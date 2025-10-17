#include "lsmindex.h"
#include "vectorindeximpl.hpp"
#include "utils.h"
#include "lib/stringinfo.h"
#include "storage/itemptr.h"
#include "storage/shmem.h"
#include "storage/dsm.h"
#include "tasksend.h"
#include <stdint.h>
#include <time.h>
#include "lsm_merge_worker.h"

LSMIndexBuffer *SharedLSMIndexBuffer = NULL;
MemtableBuffer *SharedMemtableBuffer = NULL;

// helper functions for memtables
ConcurrentMemTable
MT_FROM_SLOTIDX(int slot_num)
{
    return &SharedMemtableBuffer->slots[slot_num].mt;
}

static inline ConcurrentMemTable
MT_FROM_SLOT(MemtableBufferSlot *slot)
{
    return &slot->mt;
}

void *
VEC_BASE_FROM_MT(ConcurrentMemTable mt)
{
    /* payload starts right after header; blob is already aligned */
    return (void *)mt->vector_blob;
}

static inline Size
VEC_BYTES_PER_ROW(const ConcurrentMemTable mt)
{
    return (Size)mt->dim * (Size)mt->elem_size;
}

static inline void *
VEC_PTR_AT(ConcurrentMemTable mt, uint32 row_idx)
{
    char  *base   = (char *)VEC_BASE_FROM_MT(mt);
    Size stride = MAXALIGN(VEC_BYTES_PER_ROW(mt)); /* keep rows aligned */
    return (void *)(base + (Size)row_idx * stride);
}

// static inline size_t
// vec_bytes(const ConcurrentMemTable t)
// {
//     return (size_t)t->dim * (size_t)t->elem_size;
// }

// // Compute pointer to i-th vector.
// static inline void *
// vec_ptr(const ConcurrentMemTable t, uint32 i)
// {
//     return t->vector_base + (size_t)i * vec_bytes(t);
// }


void
lsm_index_buffer_shmem_initialize()
{
    elog(DEBUG1, "enter lsm_index_buffer_shmem_initialize");

    bool found;

    // initialize SharedLSMIndexBuffer
    SharedLSMIndexBuffer = (LSMIndexBuffer *)
        ShmemInitStruct("LSM Index Buffer",
                        MAXALIGN(sizeof(LSMIndexBuffer)),
                        &found);
    LWLockPadded *mt_tranche = GetNamedLWLockTranche(LSM_MEMTABLE_LWTRANCHE);
    // LWLockPadded *seg_tranche = GetNamedLWLockTranche(LSM_SEGMENT_LWTRANCHE);
    if (!found)
    {
        for (int i = 0; i < INDEX_BUF_SIZE; i++)
        {
            pg_atomic_write_u32(&SharedLSMIndexBuffer->slots[i].valid, 0);
            LWLockInitialize(&SharedLSMIndexBuffer->slots[i].lsmIndex.mt_lock, LSM_MEMTABLE_LWTRANCHE_ID);
            SharedLSMIndexBuffer->slots[i].lsmIndex.mt_lock = &mt_tranche[i].lock;
            // SharedLSMIndexBuffer->slots[i].lsmIndex.seg_lock = &seg_tranche[i].lock;
            SharedLSMIndexBuffer->slots[i].lsmIndex.growing_memtable_idx = MT_IDX_INVALID;
            SharedLSMIndexBuffer->slots[i].lsmIndex.growing_memtable_id  = 0;
        }
    }
    // initialize SharedMemtableBuffer
    Pointer base = (ConcurrentMemTable *)
        ShmemInitStruct("Memtable Buffer",
                        MAXALIGN(sizeof(MemtableBuffer)),
                        &found);
    // first chunk is the buffer info
    SharedMemtableBuffer = (MemtableBuffer *)base;
    if (!found)
    {
        memset(SharedMemtableBuffer, 0, sizeof(MemtableBuffer));

        for (int i = 0; i < MEMTABLE_BUF_SIZE; i++)
        {
            MemtableBufferSlot *slot = &SharedMemtableBuffer->slots[i];
            ConcurrentMemTable   mt  = MT_FROM_SLOT(slot);

            pg_atomic_write_u32(&slot->ref_count, 0);

            /* Initialize header fields */
            memset(mt, 0, sizeof(*mt));
        }
    }
}

static inline SegmentId
alloc_segment_id(LSMIndex lsm)
{
    /* Unique, monotonic id; no lock needed */
    return (SegmentId) pg_atomic_fetch_add_u32(&lsm->next_segment_id, 1);
}

static int
register_lsm_index()
{
    // find an empty slot
    int slot_num = -1;
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        uint32 expected = 0;
        if (pg_atomic_compare_exchange_u32(&SharedLSMIndexBuffer->slots[i].valid, &expected, 1))
        {
            slot_num = i;
            break;
        }
    }
    if (slot_num == -1)
    {
        // no free slot - randomly evict one
        elog(ERROR, "[register_lsm_index] no free slot");
    }
    return slot_num;
}

static int
register_and_set_memtable(LSMIndex lsm)
{
    // find an empty slot
    int slot_num = -1;
    for (int i = 0; i < MEMTABLE_BUF_SIZE; i++)
    {
        uint32 expected = 0;
        if (pg_atomic_compare_exchange_u32(&SharedMemtableBuffer->slots[i].ref_count, &expected, 1))
        {
            slot_num = i;
            break;
        }
    }
    // set memtable
    if (slot_num < 0)
    {
        return -1;
    }

    ConcurrentMemTable mt = MT_FROM_SLOTIDX(slot_num);  
    mt->rel = lsm->indexRelId;
    mt->memtable_id = alloc_segment_id(lsm);
    // elog(DEBUG1, "[register_and_set_memtable] mt->memtable_id = %d", mt->memtable_id);
    mt->dim = lsm->dim;
    mt->elem_size = lsm->elem_size;
    // calculate capacity
    Size vecbytes = (Size)mt->dim * (Size)mt->elem_size;
    uint32 cap_by_bytes = (uint32)(MEMTABLE_VECTOR_ARRAY_SIZE_BYTES / vecbytes);
    uint32 cap = Min(cap_by_bytes, (uint32)MEMTABLE_MAX_CAPACITY);
    mt->capacity = cap;
    pg_atomic_write_u32(&mt->current_size, 0);
    pg_atomic_write_u32(&mt->ready_cnt, 0);
    pg_atomic_write_u32(&mt->sealed, 0);
    pg_atomic_write_u32(&mt->flush_claimed, 0);
    pg_atomic_write_u32(&mt->max_ready_id, UINT32_MAX); /* No slots ready initially */
    
    MemSet(mt->ready, 0, cap);
    MemSet(mt->bitmap, 0, sizeof(uint8_t) * MEMTABLE_BITMAP_SIZE);

    pg_write_barrier();
    elog(DEBUG1, "[register_and_set_memtable]  mt->memtable_id = %d, slot_num = %d",  mt->memtable_id, slot_num);
    return slot_num;
}

// require an exclusive lock
// try to allocate a new memtable slot from the global buffer, waiting if none
static int
allocate_new_growing_memtable(LSMIndex lsm)
{
    for (;;)
    {
        int mt_idx = register_and_set_memtable(lsm);
        elog(DEBUG1, "[allocate_new_growing_memtable] mt_idx = %d", mt_idx);
        if (mt_idx >= 0)
        {
            return mt_idx;
        }
        pg_usleep(1000);
    }
}

// persist the index segment to disk (for initial build)
static void
persist_index_segment(LSMIndex lsm, SegmentId id_start, SegmentId id_end, uint64_t count, int64_t *tids, uint8_t *bitmap, void *index_bin, IndexType index_type)
{
    PrepareFlushMetaData prep;
    prep.start_sid = id_start;
    prep.end_sid = id_end;
    prep.valid_rows = count;
    prep.index_type = index_type;

    // initialize the mapping
    prep.map_size = sizeof(int64_t) * count;
    Pointer mapping_ptr = palloc(prep.map_size);
    memcpy(mapping_ptr, tids, prep.map_size);
    prep.map_ptr = mapping_ptr;

    // initialize the bitmap
    prep.bitmap_size = GET_BITMAP_SIZE(count);
    Pointer bitmap_ptr = palloc(prep.bitmap_size);
    if (bitmap == NULL)
    {
        MemSet(bitmap_ptr, 0x00, prep.bitmap_size);
    }
    else
    {
        memcpy(bitmap_ptr, bitmap, prep.bitmap_size);
    }
    prep.bitmap_ptr = bitmap_ptr;

    // set the index binary
    prep.index_bin = index_bin;

    flush_segment_to_disk(lsm, &prep);
    /* Persist overall LSM index metadata (index_type, dim, elem_size) */
    write_lsm_index_metadata(lsm);

    pfree(mapping_ptr);
    pfree(bitmap_ptr);
}

static dsm_segment * 
try_dsm_attach(dsm_handle hdl)
{
    dsm_segment *seg = dsm_find_mapping(hdl);
    if (seg == NULL)  // not attached in this process yet
        seg = dsm_attach(hdl);
    return seg;
}

// no locks are needed when building
void
build_lsm_index(IndexType type, Oid relId, void *vector_index, int64_t *tids, uint32_t dim, uint32_t elem_size, uint64_t count)
{
    elog(DEBUG1, "enter build_lsm_index");
    int slot_num = register_lsm_index();
    if (slot_num == -1)
    {
        elog(ERROR, "[build_lsm_index] no free slot to register a new lsm index");
    }
    // elog(DEBUG1, "[build_lsm_index] slot_num = %d", slot_num);
    
    LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[slot_num];
    slot->lsmIndex.indexRelId = relId;
    slot->lsmIndex.index_type = type;
    slot->lsmIndex.dim = dim;
    slot->lsmIndex.elem_size = elem_size;
    pg_atomic_write_u32(&slot->lsmIndex.next_segment_id, START_SEGMENT_ID + 1);

    // serialize the index
    void *index_bin_set;
    IndexSerialize(vector_index, &index_bin_set);

    // persist the index segment
    persist_index_segment(&slot->lsmIndex, START_SEGMENT_ID, START_SEGMENT_ID, count, tids, NULL, index_bin_set, type);

    index_build_blocking(relId, slot_num);

    // update segment array
    add_to_segment_array(slot_num, relId, START_SEGMENT_ID, START_SEGMENT_ID, count, type, 0.0f);

    // initialize the memtable
    for (int i = 0; i < MEMTABLE_NUM; i++)
    {
        slot->lsmIndex.memtable_idxs[i] = -1;
    }
    slot->lsmIndex.memtable_count = 0;
    int mt_idx = allocate_new_growing_memtable(&slot->lsmIndex);
    slot->lsmIndex.growing_memtable_idx = mt_idx;
    slot->lsmIndex.growing_memtable_id = MT_FROM_SLOTIDX(mt_idx)->memtable_id;
    // elog(DEBUG1, "[build_lsm_index] initialized the memtable, growing_memtable_idx = %d, growing_memtable_id = %d", 
    //         slot->lsmIndex.growing_memtable_idx, slot->lsmIndex.growing_memtable_id);

    // index will be freed outside this function
}

static void
load_lsm_index(Oid index_relid, uint32_t slot_idx)
{
    elog(DEBUG1, "[load_lsm_index] loading index: relId = %u to slot %u", index_relid, slot_idx);
    
    LSMIndex lsm = &SharedLSMIndexBuffer->slots[slot_idx].lsmIndex;
    
    // Read LSM index metadata from disk
    IndexType index_type;
    uint32_t dim, elem_size;
    if (!read_lsm_index_metadata(index_relid, &index_type, &dim, &elem_size))
    {
        elog(ERROR, "[load_lsm_index] Failed to read LSM index metadata for index %u", index_relid);
    }
    
    // Initialize LSM index structure
    lsm->indexRelId = index_relid;
    lsm->index_type = index_type;
    lsm->dim = dim;
    lsm->elem_size = elem_size;
    // FIXME: check this
    pg_atomic_write_u32(&lsm->next_segment_id, START_SEGMENT_ID + 1);
    
    // Load all flushed segments from disk via vector index worker
    index_load_blocking(index_relid, slot_idx);

    // Initialize memtables
    for (int i = 0; i < MEMTABLE_NUM; i++)
    {
        lsm->memtable_idxs[i] = -1;
    }
    lsm->memtable_count = 0;
    
    // Allocate a new growing memtable
    int mt_idx = allocate_new_growing_memtable(lsm);
    lsm->growing_memtable_idx = mt_idx;
    lsm->growing_memtable_id = MT_FROM_SLOTIDX(mt_idx)->memtable_id;
    
    elog(DEBUG1, "[load_lsm_index] successfully loaded LSM index %u", index_relid);
}

int
get_lsm_index_idx(Oid index_relid)
{    
    // check if it's already in the buffer
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        if (pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid) && SharedLSMIndexBuffer->slots[i].lsmIndex.indexRelId == index_relid)
        {
            return i;
        }
    }
    
    elog(DEBUG1, "[get_lsm_index_idx] the requested lsm_index is not in the buffer");
    // find an empty slot
    int slot_num = register_lsm_index();
    if (slot_num == -1)
    {
        elog(ERROR, "[get_lsm_index_idx] no free slot to register a new lsm index");
        return NULL;
    }
    load_lsm_index(index_relid, slot_num);
    return slot_num;
}

LSMIndex
get_lsm_index(Oid index_relid)
{
    // check if it's already in the buffer
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        if (pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid) && SharedLSMIndexBuffer->slots[i].lsmIndex.indexRelId == index_relid)
        {
            return &SharedLSMIndexBuffer->slots[i].lsmIndex;
        }
    }
    
    elog(DEBUG1, "[get_lsm_index] the requested lsm_index is not in the buffer");
    // find an empty slot
    int slot_num = register_lsm_index();
    if (slot_num == -1)
    {
        elog(ERROR, "[get_lsm_index] no free slot to register a new lsm index");
        return NULL;
    }
    load_lsm_index(index_relid, slot_num);
    return &SharedLSMIndexBuffer->slots[slot_num].lsmIndex;
}

/* Helper function to update max_ready_id atomically */
static inline void
update_max_ready_id(ConcurrentMemTable t, uint32 slot_id)
{
    uint32 current_max = pg_atomic_read_u32(&t->max_ready_id);
    
    /* Handle initial case where no slots are ready yet */
    if (current_max == UINT32_MAX) {
        if (slot_id == 0) {
            /* First slot (0) is ready, try to set max_ready_id to 0 */
            pg_atomic_compare_exchange_u32(&t->max_ready_id, &current_max, 0);
        }
        return;
    }
    
    /* Only update if this slot extends the contiguous ready range */
    if (slot_id == current_max + 1) {
        /* Find the new maximum contiguous ready ID (inclusive) */
        uint32 new_max = current_max + 1;
        while (new_max < t->capacity && t->ready[new_max] == 1) {
            new_max++;
        }
        new_max--; /* Convert back to inclusive (last ready slot) */
        
        /* Atomically update max_ready_id if we found a larger contiguous range */
        while (new_max > current_max) {
            if (pg_atomic_compare_exchange_u32(&t->max_ready_id, &current_max, new_max)) {
                break;
            }
            /* If CAS failed, re-read current value and try again if still beneficial */
            if (current_max >= new_max) {
                break;
            }
        }
    }
}

// /* Helper function to refresh max_ready_id during search operations */
// static inline void
// refresh_max_ready_id(ConcurrentMemTable t)
// {
//     uint32 current_max = pg_atomic_read_u32(&t->max_ready_id);
    
//     /* Handle initial case where no slots are ready yet */
//     if (current_max == UINT32_MAX) {
//         if (t->capacity > 0 && t->ready[0] == 1) {
//             /* Find the maximum contiguous ready range starting from 0 */
//             uint32 new_max = 0;
//             while (new_max + 1 < t->capacity && t->ready[new_max + 1] == 1) {
//                 new_max++;
//             }
//             pg_atomic_compare_exchange_u32(&t->max_ready_id, &current_max, new_max);
//         }
//         return;
//     }
    
//     /* Scan forward from current max + 1 to find new contiguous ready range */
//     uint32 new_max = current_max;
//     while (new_max + 1 < t->capacity && t->ready[new_max + 1] == 1) {
//         new_max++;
//     }
    
//     /* Update if we found a larger contiguous range */
//     if (new_max > current_max) {
//         pg_atomic_compare_exchange_u32(&t->max_ready_id, &current_max, new_max);
//     }
// }

/* Publish/consume ordering helpers. */
static inline void
publish_slot_release(ConcurrentMemTable t, uint32 i)
{
    // Ensure data writes (tid/vector) are globally visible before ready=1.
    pg_write_barrier();
    t->ready[i] = 1;
    pg_atomic_add_fetch_u32(&t->ready_cnt, 1);
    
    /* Update max_ready_id if this extends the contiguous ready range */
    update_max_ready_id(t, i);
}

static inline bool
slot_is_ready_acquire(const ConcurrentMemTable t, uint32 i)
{
    if (t->ready[i] == 0)
        return false;
    /* Pairs with writer’s release so payload reads don’t get reordered before the check. */
    pg_read_barrier();
    return true;
}

/* Reserve exactly one slot. Returns UINT32_MAX if full/frozen. */
static inline uint32
reserve_slot(ConcurrentMemTable t)
{
    /* Stop new reservations if sealed. */
    if (pg_atomic_read_u32(&t->sealed) == 1)
        return UINT32_MAX;

    uint32 i = pg_atomic_fetch_add_u32(&t->current_size, 1);
    if (__builtin_expect(i >= t->capacity, 0))
    {
        // set sealed
        pg_atomic_write_u32(&t->sealed, 1);
        return UINT32_MAX;
    }
    return i;
}

typedef void (*row_consumer_fn)(uint32 i, int64_t tid, const void *vec, void *arg);
static inline void
scan_ready_rows(const ConcurrentMemTable t, row_consumer_fn fn, void *arg)
{
    /* Snapshot how many slots were ever handed out */
    uint32 upto = pg_atomic_read_u32(&t->current_size);
    if (upto > t->capacity) upto = t->capacity;

    for (uint32 i = 0; i < upto; i++)
    {
        if (!slot_is_ready_acquire(t, i))
            continue;

        int64_t tid = t->tids[i];
        void *vec = VEC_PTR_AT(t, i);
        fn(i, tid, vec, arg);
    }
}

// push a sealed memtable into memtable_idxs[], waiting if full
static void
enqueue_sealed_memtable(LSMIndex lsm, int mt_idx)
{
    for (;;)
    {
        LWLockAcquire(lsm->mt_lock, LW_EXCLUSIVE);
        if (lsm->memtable_count < MEMTABLE_NUM)
        {
            lsm->memtable_idxs[lsm->memtable_count++] = mt_idx;
            LWLockRelease(lsm->mt_lock);
            return;
        }
        // wait if full
        LWLockRelease(lsm->mt_lock);
        pg_usleep(1000);
    }
}

static void
rotate_growing_memtable(LSMIndex lsm)
{
    int cur_idx;
    ConcurrentMemTable cur;
    SegmentId cur_id;

    LWLockAcquire(lsm->mt_lock, LW_EXCLUSIVE);

    cur_idx = lsm->growing_memtable_idx;
    // if another rotation is in process, just wait outside
    if (cur_idx == MT_IDX_ROTATING)
    {
        LWLockRelease(lsm->mt_lock);
        while(lsm->growing_memtable_idx == MT_IDX_ROTATING)
            pg_usleep(50);
        return;
    }

    if (cur_idx < 0 || cur_idx >= MEMTABLE_BUF_SIZE)
    {
        LWLockRelease(lsm->mt_lock);
        elog(ERROR, "[rotate_growing_memtable] invalid growing_memtable_idx = %d", cur_idx);
    }

    cur = MT_FROM_SLOTIDX(cur_idx);
    cur_id = cur->memtable_id;

    // if not sealed, no need to rotate
    if (pg_atomic_read_u32(&cur->sealed) == 0)
    {
        LWLockRelease(lsm->mt_lock);
        return;
    }

    lsm->growing_memtable_idx = MT_IDX_ROTATING;
    lsm->growing_memtable_id  = cur_id;

    LWLockRelease(lsm->mt_lock);

    // wait for publish 
    // wait can be done in the background
    // wait_for_publish(cur);
    enqueue_sealed_memtable(lsm, cur_idx);

    int new_idx = allocate_new_growing_memtable(lsm);

    LWLockAcquire(lsm->mt_lock, LW_EXCLUSIVE);
    lsm->growing_memtable_idx = new_idx;
    lsm->growing_memtable_id = MT_FROM_SLOTIDX(new_idx)->memtable_id;
    LWLockRelease(lsm->mt_lock);
}

// TODO: 
// potential optimization: start rotate_growing_memtable before the current growing memtable is full.

void 
insert_lsm_index(Relation index, const void *vector, const int64_t tid)
{
    LSMIndex lsm_index = get_lsm_index(index->rd_id);

    for (;;)
    {
        LWLockAcquire(lsm_index->mt_lock, LW_SHARED);
        int gidx = lsm_index->growing_memtable_idx;
        if (gidx == MT_IDX_ROTATING)
        {
            LWLockRelease(lsm_index->mt_lock);
            pg_usleep(50);
            continue;
        }
        
        if (gidx < 0 || gidx >= MEMTABLE_BUF_SIZE)
        {
            LWLockRelease(lsm_index->mt_lock);
            elog(ERROR, "[insert_lsm_index] invalid growing memtable idx");
        }

        ConcurrentMemTable mt = MT_FROM_SLOTIDX(gidx);
        Assert(mt->in_use);
        uint32_t i = reserve_slot(mt);
        LWLockRelease(lsm_index->mt_lock);

        if (i != UINT32_MAX)
        {
            mt->tids[i] = tid;
            memcpy(VEC_PTR_AT(mt, i), vector, VEC_BYTES_PER_ROW(mt));
            publish_slot_release(mt, i);
            return;
        }

        // slow path: current growing memtable is sealed. 
        rotate_growing_memtable(lsm_index);        
    }
}

// Comparison function for sorting DistancePair by distance
static int 
compare_distance(const void *a, const void *b)
{
    DistancePair *pairA = (DistancePair*)a;
    DistancePair *pairB = (DistancePair*)b;

    if (pairA->distance < pairB->distance)
        return -1;
    else if (pairA->distance > pairB->distance)
        return 1;
    else
        return 0;
}


// pairs_1 and pairs_2 should be sorted, return_pairs should be pre-allocated
static int 
merge_top_k(DistancePair *pairs_1, DistancePair *pairs_2, int num_1, int num_2, int top_k, DistancePair *merge_pair)
{
    // elog(DEBUG1, "enter merge_top_k");

    int i = 0, j = 0, k = 0;

    while (k < top_k && (i < num_1 || j < num_2)) {
        if (i < num_1 && (j >= num_2 || pairs_1[i].distance <= pairs_2[j].distance)) {
            merge_pair[k++] = pairs_1[i++];
        } else if (j < num_2) {
            merge_pair[k++] = pairs_2[j++];
        }
    }

    return k;
}

// Use BruteForceSearch for contiguous ready range + ComputeDistance for remaining slots
static int
search_memtable_common(ConcurrentMemTable mt, const void *query_vector, int k, DistancePair *top_k_pairs, bool check_ready)
{
    uint32_t valid_num = Min(pg_atomic_read_u32(&mt->current_size), mt->capacity);
    Assert(valid_num == mt->ready_cnt);

    if (valid_num == 0) {
        return 0;
    }
    
    if (check_ready) {
        /* Optimization: use max_ready_id to split into BruteForceSearch + individual ComputeDistance */
        uint32_t max_ready = pg_atomic_read_u32(&mt->max_ready_id);

        /* Handle case where no slots are ready yet */
        if (max_ready == UINT32_MAX) {
            /* All slots need ready check - use individual ComputeDistance */
            DistancePair *dist_pairs = (DistancePair *) palloc(sizeof(DistancePair) * valid_num);
            int actual_num = 0;
            
            for (uint32_t i = 0; i < valid_num; i++)
            {
                if (!slot_is_ready_acquire(mt, i))
                    continue;
                if (IS_SLOT_SET(mt->bitmap, i)) // marked as "excluded"
                    continue;
                dist_pairs[actual_num].distance = ComputeDistance((const float *)VEC_PTR_AT(mt, i), (const float *)query_vector, mt->dim);
                dist_pairs[actual_num].id = mt->tids[i];
                actual_num++;
            }
            
            if (actual_num == 0) {
                pfree(dist_pairs);
                return 0;
            }
            
            // Sort and return top k
            qsort(dist_pairs, actual_num, sizeof(DistancePair), compare_distance);
            int result_count = Min(k, actual_num);
            for (int i = 0; i < result_count; i++) {
                top_k_pairs[i] = dist_pairs[i];
            }
            pfree(dist_pairs);
            return result_count;
        } else {
            uint32_t ready_limit = Min(max_ready + 1, valid_num);
            topKVector *brute_force_result = NULL;
            DistancePair *remaining_pairs = NULL;
            int brute_force_count = 0;
            int remaining_count = 0;
            
            /* Part 1: Use ComputeDistance for remaining slots with ready check (do this first) */
            if (ready_limit < valid_num) {
                remaining_pairs = (DistancePair *) palloc(sizeof(DistancePair) * (valid_num - ready_limit));
                for (uint32_t i = ready_limit; i < valid_num; i++)
                {
                    if (!slot_is_ready_acquire(mt, i))
                        continue;
                    if (IS_SLOT_SET(mt->bitmap, i)) // marked as "excluded"
                        continue;
                    remaining_pairs[remaining_count].distance = ComputeDistance((const float *)VEC_PTR_AT(mt, i), (const float *)query_vector, mt->dim);
                    remaining_pairs[remaining_count].id = mt->tids[i];
                    remaining_count++;
                }
                // Sort remaining pairs
                if (remaining_count > 0) {
                    qsort(remaining_pairs, remaining_count, sizeof(DistancePair), compare_distance);
                }
            }
            
            /* Part 2: Use BruteForceSearch for contiguous ready range (0 to max_ready) (do this second) */
            if (ready_limit > 0) {
                // Call BruteForceSearch using the existing mt->bitmap directly
                brute_force_result = BruteForceSearch(
                    (const float *)VEC_BASE_FROM_MT(mt),
                    (const float *)query_vector,
                    mt->bitmap,  // Use the existing bitmap directly
                    ready_limit,
                    k,
                    mt->dim
                );
                brute_force_count = brute_force_result ? brute_force_result->num_results : 0;
            }
            
            /* Part 3: Merge results from ComputeDistance and BruteForceSearch */
            if (brute_force_count == 0 && remaining_count == 0) {
                if (brute_force_result) free_topk_vector(brute_force_result);
                if (remaining_pairs) pfree(remaining_pairs);
                return 0;
            }
            
            // Convert BruteForceSearch result to DistancePair format
            DistancePair *brute_force_pairs = NULL;
            if (brute_force_count > 0) {
                brute_force_pairs = (DistancePair *) palloc(sizeof(DistancePair) * brute_force_count);
                for (int i = 0; i < brute_force_count; i++) {
                    brute_force_pairs[i].distance = brute_force_result->distances[i];
                    brute_force_pairs[i].id = mt->tids[brute_force_result->vids[i]]; // Convert vector index to TID
                }
            }
            
            // Merge the two sorted arrays directly into top_k_pairs (remaining_pairs first, then brute_force_pairs)
            int merged_count = merge_top_k(remaining_pairs, brute_force_pairs, remaining_count, brute_force_count, k, top_k_pairs);
            
            // Cleanup
            if (brute_force_result) free_topk_vector(brute_force_result);
            if (brute_force_pairs) pfree(brute_force_pairs);
            if (remaining_pairs) pfree(remaining_pairs);
            
            return merged_count;
        }
    } else {
        // Use BruteForceSearch for the entire sealed memtable using existing mt->bitmap
        topKVector *result = BruteForceSearch(
            (const float *)VEC_BASE_FROM_MT(mt),
            (const float *)query_vector,
            mt->bitmap,  // Use the existing bitmap directly
            valid_num,
            k,
            mt->dim
        );
        
        int result_count = result ? result->num_results : 0;
        
        // Convert result to output format
        for (int i = 0; i < result_count; i++) {
            top_k_pairs[i].distance = result->distances[i];
            top_k_pairs[i].id = mt->tids[result->vids[i]]; // Convert vector index to TID
        }
        
        // Cleanup
        if (result) free_topk_vector(result);
        
        return result_count;
    }
}

static int
search_growing_memtable(ConcurrentMemTable mt, const void *query_vector, int k, DistancePair *top_k_pairs)
{
    return search_memtable_common(mt, query_vector, k, top_k_pairs, /*check_ready=*/true);
}

static int
search_sealed_memtable(ConcurrentMemTable mt, const void *query_vector, int k, DistancePair *top_k_pairs)
{
    return search_memtable_common(mt, query_vector, k, top_k_pairs, /*check_ready=*/false);
}
// FIXME: how to decide the search parameters for each segment?
// for now we just use the same parameters for all segments
TopKTuples
search_lsm_index(Relation index, const void *vector, int k, int nprobe_efs)
{
    // elog(DEBUG1, "enter search_lsm_index, k = %d, nprobe_efs = %d", k, nprobe_efs);

    LSMIndex lsm = get_lsm_index(index->rd_id);
    Assert(lsm);

    // step 1. get a snapshot of the current memtables
    LSMSnapshot lsm_snapshot;
    
    LWLockAcquire(lsm->mt_lock, LW_SHARED);

    lsm_snapshot.gidx = lsm->growing_memtable_idx;
    lsm_snapshot.gmt_id = MT_FROM_SLOTIDX(lsm_snapshot.gidx)->memtable_id;
    pg_atomic_add_fetch_u32(&SharedMemtableBuffer->slots[lsm_snapshot.gidx].ref_count, 1);
    lsm_snapshot.scount = lsm->memtable_count;
    for (int i = 0; i < lsm_snapshot.scount; i++)
    {
        lsm_snapshot.sidxs[i] = lsm->memtable_idxs[i];
        lsm_snapshot.smt_ids[i] = MT_FROM_SLOTIDX(lsm_snapshot.sidxs[i])->memtable_id;
        pg_atomic_add_fetch_u32(&SharedMemtableBuffer->slots[lsm_snapshot.sidxs[i]].ref_count, 1);
    }

    LWLockRelease(lsm->mt_lock);
    // elog(DEBUG1, "[search_lsm_index] Snapshot: gidx = %d, gmt_id = %d, scount = %d", lsm_snapshot.gidx, lsm_snapshot.gmt_id, lsm_snapshot.scount);

    // step 2. issue a search task to the vector search process
    vector_search_send(index->rd_id, (float *)vector, lsm->dim, lsm->elem_size, k, nprobe_efs, lsm_snapshot);
    
    // step 3. search the growing memtables (update the reference count after search)
    DistancePair *final_pairs, *pair_1;
    int num_1;

    DistancePair *gmt_pairs = palloc(sizeof(DistancePair) * k);
    int gmt_num = search_growing_memtable(MT_FROM_SLOTIDX(lsm_snapshot.gidx), vector, k, gmt_pairs);
    num_1 = gmt_num;
    pair_1 = gmt_pairs;
    pg_atomic_add_fetch_u32(&SharedMemtableBuffer->slots[lsm_snapshot.gidx].ref_count, -1);
    // elog(DEBUG1, "[search_lsm_index] searched the growing memtables, num_1 = %d", num_1);

    // step 4. search the immutable memtables (update the reference count after searches)
    for (int i = 0; i < lsm_snapshot.scount; i++)
    {
        DistancePair *smt_pairs = palloc(sizeof(DistancePair) * k);
        int smt_num = search_sealed_memtable(MT_FROM_SLOTIDX(lsm_snapshot.sidxs[i]), vector, k, smt_pairs);
        // merge smt_pairs into pair_1
        final_pairs = palloc(sizeof(DistancePair) * k);
        int merge_num = merge_top_k(pair_1, smt_pairs, num_1, smt_num, k, final_pairs);
        pfree(pair_1);
        pfree(smt_pairs);
        num_1 = merge_num;
        pair_1 = final_pairs;
        pg_atomic_add_fetch_u32(&SharedMemtableBuffer->slots[lsm_snapshot.sidxs[i]].ref_count, -1);
        // elog(DEBUG1, "[search_lsm_index] searched an immutable memtable, smt_num = %d, num_1 = %d", smt_num, num_1);
    }
    // elog(DEBUG1, "[search_lsm_index] searched the immutable memtables");
    
    // step 5. merge the results (wait for the results from the vector search process if not ready yet)
    VectorSearchResult vs_result = vector_search_get_result();
    int64_t *res_id = vs_search_result_id_at(vs_result);
    float *res_dist = vs_search_result_dist_at(vs_result);
    DistancePair *segment_pairs = palloc(sizeof(DistancePair) * k);
    for (int i = 0; i < vs_result->result_count; i++)
    {
        segment_pairs[i].id = res_id[i];
        segment_pairs[i].distance = res_dist[i];
    }
    // merge segment_pairs into pair_1
    final_pairs = palloc(sizeof(DistancePair) * k);
    int final_num = merge_top_k(pair_1, segment_pairs, num_1, vs_result->result_count, k, final_pairs);
    pfree(pair_1);
    pfree(segment_pairs);

    TopKTuples topk_result = {
        .num_results = final_num,
        .pairs = final_pairs
    };
    // elog(DEBUG1, "[search_lsm_index] return %d", topk_result.num_results);
    return topk_result;
}

// -------------------------------- for debugging ----------------------------------
// static void*
// get_pointer_from_cached_segment(dsm_handle handle)
// {
//     dsm_segment *seg = dsm_find_mapping(handle);
//     if (seg == NULL)
//     {
//         seg = dsm_attach(handle);
//         dsm_pin_mapping(seg);
//     }
//     return dsm_segment_address(seg); 
// }

// // TODO: for debugging
// static void 
// vector_search(Oid index_oid, float *query, int dim, Size elem_size, int topk, int efs_nprobe, LSMSnapshot lsm_snapshot, topKVector *topk_result)
// {   
//     // elog(DEBUG1, "enter vector_search");
//     DistancePair *final_pairs = NULL, *pairs_1 = NULL;
//     int num_1;

//     // traverse all flushed segments
//     int lsm_idx = get_lsm_index_idx(index_oid);
//     LSMIndex lsm = &SharedLSMIndexBuffer->slots[lsm_idx].lsmIndex;
//     Assert(lsm);
//     uint32_t idx = 0;
//     do
//     {
//         // skip the flushed segment if its segment id is in the snapshot
//         bool found = false;
//         for (int i = 0; i < lsm_snapshot.scount; i++)
//         {
//             // TODO: this requires that we should never merge flushed segments whose segment idx are in the snapshot
//             if (lsm_snapshot.smt_ids[i] <= lsm->flushed_segments[idx].segment_id_end &&
//                 lsm_snapshot.smt_ids[i] >= lsm->flushed_segments[idx].segment_id_start)
//             {
//                 Assert(lsm->flushed_segments[idx].segment_id_end == lsm->flushed_segments[idx].segment_id_start);
//                 found = true;
//                 break;
//             }
//         }
//         found = found || (lsm_snapshot.gmt_id <= lsm->flushed_segments[idx].segment_id_end &&
//                         lsm_snapshot.gmt_id >= lsm->flushed_segments[idx].segment_id_start);
//         if (!found)
//         {
            
//             // conduct search
//             topKVector *segment_result;
//             uint8_t *bitmap = (uint8_t *)get_pointer_from_cached_segment(lsm->flushed_segments[idx].bitmap);
//             int64_t *mapping = (int64_t *)get_pointer_from_cached_segment(lsm->flushed_segments[idx].mapping);
//             // dsm_segment *bm_seg = dsm_attach(lsm->flushed_segments[idx].bitmap);
//             // void *bitmap = dsm_segment_address(bm_seg);
//             // dsm_segment *map_seg = dsm_attach(lsm->flushed_segments[idx].mapping);
//             // int64_t *mapping = (int64_t *)dsm_segment_address(map_seg);
//             segment_result = VectorIndexSearch(lsm->index_type, lsm_idx, idx, 
//                                                 NULL, 
//                                                 lsm->flushed_segments[idx].vec_count, 
//                                                 query, 
//                                                 topk, 
//                                                 efs_nprobe);
//             // elog(DEBUG1, "[vector_search] returned from VectorIndexSearch, segment_result->num_results = %d", segment_result->num_results);
//             // merge the result
//             int seg_result_num = segment_result->num_results;
//             pairs_1 = palloc(sizeof(DistancePair) * topk);
            
//             // TODO: for evaluation - timing instrumentation
//             struct timespec start_time, end_time;
//             clock_gettime(CLOCK_MONOTONIC, &start_time);

//             for (int i = 0; i < seg_result_num; i++)
//             {
//                 pairs_1[i].distance = segment_result->distances[i];
//                 int pos = segment_result->vids[i];
//                 pairs_1[i].id = mapping[pos];
//             }

//             // TODO: for evaluation - calculate and log execution time
//             clock_gettime(CLOCK_MONOTONIC, &end_time);
//             long execution_time_us = (end_time.tv_sec - start_time.tv_sec) * 1000000L + 
//                                     (end_time.tv_nsec - start_time.tv_nsec) / 1000L;
            
//             // Static arrays to store last 10000 execution times
//             static long execution_times[10000];
//             static int call_count = 0;
//             static int array_index = 0;
            
//             // Store current execution time
//             execution_times[array_index] = execution_time_us;
//             array_index = (array_index + 1) % 10000;
//             call_count++;
            
//             // Log statistics every 10000 calls
//             if (call_count % 10000 == 0) {
//                 long total_time = 0;
//                 long min_time = LONG_MAX;
//                 long max_time = 0;
                
//                 // Calculate stats for the last 10000 calls
//                 for (int i = 0; i < 10000; i++) {
//                     total_time += execution_times[i];
//                     if (execution_times[i] < min_time) min_time = execution_times[i];
//                     if (execution_times[i] > max_time) max_time = execution_times[i];
//                 }
                
//                 double avg_time = (double)total_time / 10000.0;
//                 elog(DEBUG1, "[LSMIndexSearch] Stats for last 10000 calls - Avg: %.2fμs, Min: %ldμs, Max: %ldμs", 
//                      avg_time, min_time, max_time);
//             }
//             // TODO: for evaluation (end here)
//             free_topk_vector(segment_result);
//             // if (pairs_1 == NULL)
//             // {
//             num_1 = seg_result_num;
//             // }
//             // else
//             // {
//             //     final_pairs = palloc(sizeof(DistancePair) * topk);
//             //     int merge_num = merge_top_k(pairs_1, segment_pairs, num_1, seg_result_num, topk, final_pairs);
//             //     pfree(pairs_1);
//             //     pfree(segment_pairs);
//             //     num_1 = merge_num;
//             //     pairs_1 = final_pairs;
//             // }
//             // dsm_detach(bm_seg);
//             // dsm_detach(map_seg);
//         }
//         idx = lsm->flushed_segments[idx].next_idx;
//     } while (idx != lsm->tail_idx);

//     // return the result
//     topk_result->num_results = num_1;
//     for (int i = 0; i < num_1; i++)
//     {
//         topk_result->distances[i] = pairs_1[i].distance;
//         topk_result->vids[i] = pairs_1[i].id;
//     }
//     pfree(pairs_1);
//     // elog(DEBUG1, "[vector_search] topk_result->num_results = %d", topk_result->num_results);    
// }

// TopKTuples
// search_lsm_index(Relation index, const void *vector, int k, int nprobe_efs)
// {
//     // elog(DEBUG1, "enter search_lsm_index, k = %d, nprobe_efs = %d", k, nprobe_efs);

//     LSMIndex lsm = get_lsm_index(index->rd_id);
//     Assert(lsm);

//     // step 1. get a snapshot of the current memtables
//     LSMSnapshot lsm_snapshot;
    
//     LWLockAcquire(lsm->mt_lock, LW_SHARED);

//     lsm_snapshot.gidx = lsm->growing_memtable_idx;
//     lsm_snapshot.gmt_id = MT_FROM_SLOTIDX(lsm_snapshot.gidx)->memtable_id;
//     pg_atomic_add_fetch_u32(&SharedMemtableBuffer->slots[lsm_snapshot.gidx].ref_count, 1);
//     lsm_snapshot.scount = lsm->memtable_count;
//     for (int i = 0; i < lsm_snapshot.scount; i++)
//     {
//         lsm_snapshot.sidxs[i] = lsm->memtable_idxs[i];
//         lsm_snapshot.smt_ids[i] = MT_FROM_SLOTIDX(lsm_snapshot.sidxs[i])->memtable_id;
//         pg_atomic_add_fetch_u32(&SharedMemtableBuffer->slots[lsm_snapshot.sidxs[i]].ref_count, 1);
//     }

//     LWLockRelease(lsm->mt_lock);
//     // elog(DEBUG1, "[search_lsm_index] Snapshot: gidx = %d, gmt_id = %d, scount = %d", lsm_snapshot.gidx, lsm_snapshot.gmt_id, lsm_snapshot.scount);

//     // TODO: for debugging
//     // // step 2. issue a search task to the vector search process
//     // vector_search_send(index->rd_id, (float *)vector, lsm->dim, lsm->elem_size, k, nprobe_efs, lsm_snapshot);
//     // conduct vector search on segments locally
//     topKVector *topk_result = (topKVector *) palloc(sizeof(topKVector));
//     topk_result->num_results = k;
//     topk_result->distances = (float *) palloc(sizeof(float) * k);
//     topk_result->vids = (int64_t *) palloc(sizeof(int64_t) * k);    
//     vector_search(index->rd_id, (float *)vector, lsm->dim, lsm->elem_size, k, nprobe_efs, lsm_snapshot, topk_result);

//     // step 5. merge the results (wait for the results from the vector search process if not ready yet)
//     int64_t *res_id = topk_result->vids;
//     float *res_dist = topk_result->distances;
//     DistancePair *segment_pairs = palloc(sizeof(DistancePair) * k);
//     for (int i = 0; i < topk_result->num_results; i++)
//     {
//         segment_pairs[i].id = res_id[i];
//         segment_pairs[i].distance = res_dist[i];
//     }

//     TopKTuples topk_result_2 = {
//         .num_results = topk_result->num_results,
//         .pairs = segment_pairs
//     };
//     free_topk_vector(topk_result);
//     // elog(DEBUG1, "[search_lsm_index] return %d", topk_result_2.num_results);
//     return topk_result_2;
// }