#include "lsmindex.h"
#include "access/xlog.h"          /* RecoveryInProgress() */
#include "replication_gucs.h"
#include "replication_rmgr.h"
#include "storage/block.h"
#include "storage/bufmgr.h"
#include "storage/itemid.h"
#include "utils/elog.h"
#include "vectorindeximpl.hpp"
#include "utils.h"
#include "lib/stringinfo.h"
#include "storage/itemptr.h"
#include "storage/shmem.h"
#include "storage/dsm.h"
#include "tasksend.h"
#include <stdbool.h>
#include <stdint.h>
#include <time.h>
#include "statuspage.h"
#include "catalog/index.h"
#include "access/table.h"
#include "access/tableam.h"
#include "vector.h"
#include "access/heapam.h"
#include "portability/instr_time.h"
#include "miscadmin.h"
#include "storage/proc.h"
#include "utils/wait_event.h"

/* Backing storage for pgvector.storage_base_dir GUC. */
char *vector_storage_base_dir = NULL;

const char *
get_vector_storage_dir(void)
{
    /* Per-backend cache. PGC_POSTMASTER ensures the value can't change after
     * postmaster start, so a one-shot resolution is safe. */
    static char resolved[MAXPGPATH] = {0};

    if (resolved[0] != '\0')
        return resolved;

    if (vector_storage_base_dir != NULL && vector_storage_base_dir[0] != '\0')
    {
        /* User-specified path; ensure trailing slash. */
        size_t len = strlen(vector_storage_base_dir);
        if (vector_storage_base_dir[len - 1] == '/')
            snprintf(resolved, sizeof(resolved), "%s", vector_storage_base_dir);
        else
            snprintf(resolved, sizeof(resolved), "%s/", vector_storage_base_dir);
    }
    else
    {
        /* Default: <DataDir>/pgvector_storage/. DataDir is guaranteed to be
         * set by the time any backend calls into this extension. */
        snprintf(resolved, sizeof(resolved), "%s/pgvector_storage/", DataDir);
    }
    return resolved;
}

LSMIndexBuffer *SharedLSMIndexBuffer = NULL;
IndexRecoveryCoordinator *SharedIndexRecoveryCoordinator = NULL;
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

/* VEC_BYTES_PER_ROW and VEC_PTR_AT are now static inline in lsmindex.h */

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
    LWLockPadded *flushed_release_tranche = GetNamedLWLockTranche(LSM_FLUSHED_RELEASE_LWTRANCHE);
    LWLockPadded *index_buffer_tranche = GetNamedLWLockTranche(LSM_INDEX_BUFFER_LWTRANCHE);
    // LWLockPadded *seg_tranche = GetNamedLWLockTranche(LSM_SEGMENT_LWTRANCHE);
    if (!found)
    {
        // Initialize the buffer-level lock
        LWLockInitialize(&index_buffer_tranche[0].lock, LSM_INDEX_BUFFER_LWTRANCHE_ID);
        SharedLSMIndexBuffer->lock = &index_buffer_tranche[0].lock;
        
        for (int i = 0; i < INDEX_BUF_SIZE; i++)
        {
            pg_atomic_write_u32(&SharedLSMIndexBuffer->slots[i].valid, (uint32) LSM_SLOT_FREE);
            ConditionVariableInit(&SharedLSMIndexBuffer->slots[i].state_cv);
            pg_atomic_init_u32(&SharedLSMIndexBuffer->slots[i].state_error, 0);
            SharedLSMIndexBuffer->slots[i].request_db_oid    = InvalidOid;
            SharedLSMIndexBuffer->slots[i].request_db_userid = InvalidOid;
            LWLockInitialize(&SharedLSMIndexBuffer->slots[i].lsmIndex.mt_lock, LSM_MEMTABLE_LWTRANCHE_ID);
            SharedLSMIndexBuffer->slots[i].lsmIndex.mt_lock = &mt_tranche[i].lock;
            LWLockInitialize(&flushed_release_tranche[i].lock, LSM_FLUSHED_RELEASE_LWTRANCHE_ID);
            SharedLSMIndexBuffer->slots[i].lsmIndex.flushed_release_lock = &flushed_release_tranche[i].lock;
            // SharedLSMIndexBuffer->slots[i].lsmIndex.seg_lock = &seg_tranche[i].lock;
            SharedLSMIndexBuffer->slots[i].lsmIndex.growing_memtable_idx = MT_IDX_INVALID;
            SharedLSMIndexBuffer->slots[i].lsmIndex.growing_memtable_id  = 0;
            pg_atomic_init_u32(&SharedLSMIndexBuffer->slots[i].lsmIndex.flushed_not_released_count, 0);
            pg_atomic_init_u32(&SharedLSMIndexBuffer->slots[i].lsmIndex.releasing_in_progress, 0);
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

            /* Initialize the vacuum lock for this memtable */
            LWLockInitialize(&mt->vacuum_lock, LSM_MEMTABLE_VACUUM_LWTRANCHE_ID);
        }
    }

    SharedIndexRecoveryCoordinator = (IndexRecoveryCoordinator *)
        ShmemInitStruct("Index Recovery Coordinator",
                        MAXALIGN(sizeof(IndexRecoveryCoordinator)),
                        &found);
    if (!found)
    {
        SharedIndexRecoveryCoordinator->worker_pgprocno = -1;
    }
}

static inline SegmentId
alloc_segment_id(LSMIndex lsm)
{
    /* Unique, monotonic id; no lock needed */
    return (SegmentId) pg_atomic_fetch_add_u32(&lsm->next_segment_id, 1);
}

/*
 * claim_or_find_lsm_index_slot — internal helper for register_lsm_index and
 * register_lsm_index_for_redo. Looks up an existing slot for index_relid;
 * if absent, claims a FREE slot, writes (db_oid, db_userid, indexRelId),
 * signals the IndexRecoveryWorker, and waits for recovery to finish.
 *
 * Returns the slot index on success. Throws via elog(ERROR) on failure.
 */
static int
claim_or_find_lsm_index_slot(Oid index_relid, Oid db_oid, Oid db_userid)
{
    LWLockAcquire(SharedLSMIndexBuffer->lock, LW_EXCLUSIVE);

    /* Double-check: another backend may have finished loading while we waited */
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        uint32 valid = pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid);
        if (valid == (uint32) LSM_SLOT_FREE ||
            SharedLSMIndexBuffer->slots[i].lsmIndex.indexRelId != index_relid)
            continue;

        LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[i];
        LWLockRelease(SharedLSMIndexBuffer->lock);

        if (is_writable(valid))
            return i;  /* recovery already done; load (if needed) is the reader's job */

        /* RECOVERING: worker is running recovery — wait on CV until done or failed */
        PG_TRY();
        {
            ConditionVariablePrepareToSleep(&slot->state_cv);
            while (pg_atomic_read_u32(&slot->valid) == (uint32) LSM_SLOT_RECOVERING)
                ConditionVariableSleep(&slot->state_cv, PG_WAIT_EXTENSION);
            ConditionVariableCancelSleep();
        }
        PG_CATCH();
        {
            ConditionVariableCancelSleep();
            PG_RE_THROW();
        }
        PG_END_TRY();

        if (is_writable(pg_atomic_read_u32(&slot->valid)) &&
            pg_atomic_read_u32(&slot->state_error) == 0)
            return i;

        elog(ERROR, "[claim_or_find_lsm_index_slot] index %u recovery failed (state_error=%u, valid=%u)",
             index_relid,
             pg_atomic_read_u32(&slot->state_error),
             pg_atomic_read_u32(&slot->valid));
    }

    /* Claim a free slot: CAS FREE→RECOVERING while holding the buffer lock */
    int slot_num = -1;
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        uint32 expected = (uint32) LSM_SLOT_FREE;
        if (pg_atomic_compare_exchange_u32(&SharedLSMIndexBuffer->slots[i].valid,
                                           &expected, (uint32) LSM_SLOT_RECOVERING))
        {
            slot_num = i;
            break;
        }
    }

    if (slot_num == -1)
    {
        LWLockRelease(SharedLSMIndexBuffer->lock);
        elog(ERROR, "[claim_or_find_lsm_index_slot] no free slot");
    }

    LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[slot_num];
    pg_atomic_write_u32(&slot->state_error, 0);
    slot->request_db_oid    = db_oid;
    slot->request_db_userid = db_userid;
    slot->lsmIndex.indexRelId    = index_relid; // if the backend dies before setting indexRelId, the worker will return an error

    LWLockRelease(SharedLSMIndexBuffer->lock);

    /* Signal the IndexRecoveryWorker — it will see valid==2 and process this slot */
    int32 wpgprocno = SharedIndexRecoveryCoordinator->worker_pgprocno;
    if (wpgprocno < 0)
    {
        /* Worker not running — reset slot and error */
        pg_atomic_write_u32(&slot->valid, (uint32) LSM_SLOT_FREE);
        ConditionVariableBroadcast(&slot->state_cv);
        elog(ERROR, "[claim_or_find_lsm_index_slot] IndexRecoveryWorker is not running");
    }
    SetLatch(&ProcGlobal->allProcs[wpgprocno].procLatch);

    /* Block until recovery finishes or fails */
    PG_TRY();
    {
        ConditionVariablePrepareToSleep(&slot->state_cv);
        while (pg_atomic_read_u32(&slot->valid) == (uint32) LSM_SLOT_RECOVERING)
            ConditionVariableSleep(&slot->state_cv, PG_WAIT_EXTENSION);
        ConditionVariableCancelSleep();
    }
    PG_CATCH();
    {
        ConditionVariableCancelSleep();
        PG_RE_THROW();
    }
    PG_END_TRY();

    if (is_writable(pg_atomic_read_u32(&slot->valid)) &&
        pg_atomic_read_u32(&slot->state_error) == 0)
        return slot_num;

    elog(ERROR, "[claim_or_find_lsm_index_slot] index %u recovery failed (state_error=%u, valid=%u)",
         index_relid,
         pg_atomic_read_u32(&slot->state_error),
         pg_atomic_read_u32(&slot->valid));
}

static int
register_lsm_index(Oid index_relid)
{
    return claim_or_find_lsm_index_slot(index_relid, MyDatabaseId, GetUserId());
}

/*
 * register_lsm_index_for_redo — variant for the WAL redo path on the
 * standby. The startup process has no MyDatabaseId / GetUserId(); the
 * dbOid is read from the WAL record. db_userid = InvalidOid means the
 * IndexRecoveryWorker will connect as the bootstrap superuser.
 *
 * TODO(multi-DB): today the IndexRecoveryWorker is a single bgworker that
 * binds (sticky) to the first dbOid it sees. With one DB this works.
 * For multi-DB standby, the worker must become a per-DB pool (mirroring
 * segment_fetcher_main's per-DB registration in vector.c).
 */
int
register_lsm_index_for_redo(Oid db_oid, Oid index_relid)
{
    if (!OidIsValid(db_oid))
        elog(ERROR, "[register_lsm_index_for_redo] dbOid is InvalidOid");
    return claim_or_find_lsm_index_slot(index_relid, db_oid, InvalidOid);
}

/*
 * claim_lsm_index_slot_for_create_redo — direct FREE -> WRITABLE transition
 * for INDEX_CREATE WAL redo. See header comment in lsmindex.h.
 *
 * Why this skips the IndexRecoveryWorker entirely:
 *   For a freshly-created index, status pages have just been initialized
 *   to empty via CreateStatusMetaPage / InitializeStatusMemtableArray
 *   (those WAL records replayed just before INDEX_CREATE). No segments
 *   exist on disk yet (persist_index_segment's local file write happens
 *   between INDEX_CREATE and SegmentCreated on the primary, and the
 *   standby's fetcher pulls only after SegmentCreated redo enqueues).
 *   No memtables exist in heap to reconstruct. recover_lsm_index_internal
 *   would scan empty inputs and produce empty output; running it is the
 *   wrong tool. More importantly, its index_open(AccessShareLock) would
 *   deadlock against the primary's CREATE-INDEX-held AccessExclusiveLock
 *   that the standby has applied via STANDBY_LOCK WAL.
 *
 * Field population mirrors build_lsm_index on the primary, minus the
 * allocate_new_growing_memtable step — growing_memtable_idx /
 * growing_memtable_id stay at their shmem-init defaults
 * (MT_IDX_INVALID / 0); the first dpv REGISTER_MEMTABLE redo for this
 * index will create the growing memtable via dpv_standby_register_memtable.
 */
int
claim_lsm_index_slot_for_create_redo(Oid db_oid, Oid index_relid,
                                     uint32 index_type, uint32 dim,
                                     uint32 elem_size)
{
    int slot_num = -1;
    LSMIndexBufferSlot *slot;

    LWLockAcquire(SharedLSMIndexBuffer->lock, LW_EXCLUSIVE);

    /* Idempotency: if a slot already exists for this indexRelId in any
     * non-FREE state, leave it alone (a concurrent backend register_lsm_index
     * or a prior re-fire of this redo set it up). Broadcast in case a
     * waiter is still pending. */
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        uint32 v = pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid);
        if (v != (uint32) LSM_SLOT_FREE &&
            SharedLSMIndexBuffer->slots[i].lsmIndex.indexRelId == index_relid)
        {
            LWLockRelease(SharedLSMIndexBuffer->lock);
            ConditionVariableBroadcast(&SharedLSMIndexBuffer->slots[i].state_cv);
            return i;
        }
    }

    /* Claim a free slot directly to WRITABLE — no RECOVERING intermediate,
     * no recovery worker signal. */
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        uint32 expected = (uint32) LSM_SLOT_FREE;
        if (pg_atomic_compare_exchange_u32(&SharedLSMIndexBuffer->slots[i].valid,
                                           &expected, (uint32) LSM_SLOT_WRITABLE))
        {
            slot_num = i;
            break;
        }
    }
    if (slot_num == -1)
    {
        LWLockRelease(SharedLSMIndexBuffer->lock);
        elog(WARNING,
             "[claim_lsm_index_slot_for_create_redo] no free slot for index %u"
             " — subsequent redo for this index will skip in-memory effects",
             index_relid);
        return -1;
    }

    slot = &SharedLSMIndexBuffer->slots[slot_num];
    pg_atomic_write_u32(&slot->state_error, 0);
    slot->request_db_oid    = db_oid;
    slot->request_db_userid = InvalidOid;

    /* Populate LSMIndex header (mirror build_lsm_index lines 498-502, 528-534). */
    slot->lsmIndex.indexRelId = index_relid;
    slot->lsmIndex.index_type = (IndexType) index_type;
    slot->lsmIndex.dim        = dim;
    slot->lsmIndex.elem_size  = elem_size;
    pg_atomic_write_u32(&slot->lsmIndex.next_segment_id, START_SEGMENT_ID + 1);

    for (int i = 0; i < MEMTABLE_NUM; i++)
        slot->lsmIndex.memtable_idxs[i] = -1;
    slot->lsmIndex.memtable_count = 0;
    pg_atomic_write_u32(&slot->lsmIndex.flushed_not_released_count, 0);
    pg_atomic_write_u32(&slot->lsmIndex.releasing_in_progress, 0);

    LWLockRelease(SharedLSMIndexBuffer->lock);

    /* Wake any waiters (none expected in normal flow, but safe). */
    ConditionVariableBroadcast(&slot->state_cv);
    return slot_num;
}

int
register_and_set_memtable(LSMIndex lsm, Relation index, bool is_recovery, SegmentId assigned_sid)
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
    if (is_recovery)
    {
        mt->memtable_id = assigned_sid;
    }
    else
    {
        mt->memtable_id = alloc_segment_id(lsm);
    }
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

    // update the status page
    if (!is_recovery)
    {
        RegisterStatusMemtable(index, mt->memtable_id);
    }

    elog(DEBUG1, "[register_and_set_memtable]  mt->memtable_id = %d, slot_num = %d",  mt->memtable_id, slot_num);
    return slot_num;
}

// require an exclusive lock
// try to allocate a new memtable slot from the global buffer, waiting if none
static int
allocate_new_growing_memtable(LSMIndex lsm, Relation index, bool is_recovery, SegmentId assigned_sid)
{
    for (;;)
    {
        int mt_idx = register_and_set_memtable(lsm, index, is_recovery, assigned_sid);
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
    prep.delete_count = 0;

    // set the index binary
    prep.index_bin = index_bin;

    // set the offsets
    SegmentOffsetRange *offsets = palloc(sizeof(SegmentOffsetRange) * 1);
    offsets[0].sid = prep.start_sid;
    offsets[0].start_offset = 0;
    offsets[0].end_offset = count;
    prep.offsets = offsets;

    flush_segment_to_disk(lsm->indexRelId, &prep);
    /* Persist overall LSM index metadata (index_type, dim, elem_size) */
    write_lsm_index_metadata(lsm);

    if (dpv_replication_role == DPV_ROLE_PRIMARY)
    {
        /* Safe: caller holds is_compacting (rebuild/merge) or single-threaded flush
         * worker — no concurrent writer can advance the version between
         * flush_segment_to_disk and this lookup. */
        uint32_t new_version =
            find_latest_segment_version(lsm->indexRelId, prep.start_sid, prep.end_sid);
        dpv_emit_segment_created(lsm->indexRelId, prep.start_sid, prep.end_sid, new_version);
    }

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
build_lsm_index(IndexType type, Relation index, void *vector_index, int64_t *tids, uint32_t dim, uint32_t elem_size, uint64_t count)
{
    elog(DEBUG1, "enter build_lsm_index");
    Oid relId = RelationGetRelid(index);

    /* Claim a free slot directly (CAS FREE→RECOVERING) — do NOT signal the
     * load worker; build_lsm_index constructs the slot in-place and
     * publishes LSM_SLOT_QUERYABLE itself once build is complete. We use
     * RECOVERING here so concurrent waiters block until we finish. */
    int slot_num = -1;
    LWLockAcquire(SharedLSMIndexBuffer->lock, LW_EXCLUSIVE);
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        uint32 expected = (uint32) LSM_SLOT_FREE;
        if (pg_atomic_compare_exchange_u32(&SharedLSMIndexBuffer->slots[i].valid,
                                           &expected, (uint32) LSM_SLOT_RECOVERING))
        {
            slot_num = i;
            break;
        }
    }
    LWLockRelease(SharedLSMIndexBuffer->lock);

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

    /*
     * Emit xl_dpv_index_create UNCONDITIONALLY (not gated on replication_role).
     *
     * Two reasons it must always emit, even on non-replicated setups where
     * pgvector.replication_role = DISABLED:
     *
     *  1. Side-channel signal: standby creates <storage>/<oid>/ + writes
     *     per-index metadata.  (Inert if there's no standby; cheap on disk.)
     *
     *  2. **Pre-establish the LSMIndexBufferSlot in LSM_SLOT_WRITABLE** on
     *     the standby BEFORE any subsequent dpv memtable record fires.
     *     dpv_emit_register_memtable / dpv_emit_update_max_sid / etc. in
     *     statuspage.c are unconditional (Plan 1 needs them to replicate
     *     memtable state on any hot-standby). If INDEX_CREATE were gated
     *     out, those records would land on a standby with no slot for the
     *     new index → redo_update_max_sid's get_lsm_index_idx(for_redo=true)
     *     would slow-path into register_lsm_index_for_redo → claim slot
     *     RECOVERING → signal IndexRecoveryWorker → worker calls
     *     index_open(AccessShareLock) → blocks on the AccessExclusiveLock
     *     primary's still-running CREATE INDEX holds (propagated via
     *     STANDBY_LOCK WAL) → and the startup process driving WAL replay
     *     is the one waiting on the recovery worker. Deadlock.
     *
     * redo_index_create's claim_lsm_index_slot_for_create_redo establishes
     * the slot in WRITABLE directly (no recovery worker, no index_open).
     * Subsequent memtable redo records then fast-path through
     * get_lsm_index_idx and the deadlock is avoided.
     *
     * Cost: one ~20-byte WAL record per CREATE INDEX. Negligible.
     */
    dpv_emit_index_create(relId, (uint32) type, dim, elem_size);

    // serialize the index
    // if the index type is Diskann, the index is already written to disk during building
    void *index_bin_set = NULL;
    if (type != DISKANN)
    {
        IndexSerialize(vector_index, &index_bin_set);
    }

    // persist the index segment
    persist_index_segment(&slot->lsmIndex, START_SEGMENT_ID, START_SEGMENT_ID, count, tids, NULL, index_bin_set, type);

    index_build_blocking(relId, slot_num);

    // initialize the memtable
    for (int i = 0; i < MEMTABLE_NUM; i++)
    {
        slot->lsmIndex.memtable_idxs[i] = -1;
    }
    slot->lsmIndex.memtable_count = 0;
    pg_atomic_init_u32(&slot->lsmIndex.flushed_not_released_count, 0);
    pg_atomic_init_u32(&slot->lsmIndex.releasing_in_progress, 0);
    int mt_idx = allocate_new_growing_memtable(&slot->lsmIndex, index, false, NULL);
    slot->lsmIndex.growing_memtable_idx = mt_idx;
    slot->lsmIndex.growing_memtable_id = MT_FROM_SLOTIDX(mt_idx)->memtable_id;
    /*
     * Publish LSM_SLOT_QUERYABLE: build_lsm_index already populated the
     * FlushedSegmentPool via index_build_blocking() above, so search is
     * allowed immediately. Broadcast so any waiters in RECOVERING wake up.
     */
    pg_write_barrier();
    pg_atomic_write_u32(&slot->valid, (uint32) LSM_SLOT_QUERYABLE);
    ConditionVariableBroadcast(&slot->state_cv);
    // elog(DEBUG1, "[build_lsm_index] initialized the memtable, growing_memtable_idx = %d, growing_memtable_id = %d",
    //         slot->lsmIndex.growing_memtable_idx, slot->lsmIndex.growing_memtable_id);

    // vector_index will be freed outside this function
}

// Comparison function for descending sort of SegmentId array
static int
compare_sids_desc(const void *a, const void *b)
{
    SegmentId sid_a = *(const SegmentId *)a;
    SegmentId sid_b = *(const SegmentId *)b;
    if (sid_a > sid_b)
        return -1;
    if (sid_a < sid_b)
        return 1;
    return 0;
}

// Comparison function for ascending sort of int64_t array (for binary search)
static int
compare_int64(const void *a, const void *b)
{
    int64_t val_a = *(const int64_t *)a;
    int64_t val_b = *(const int64_t *)b;
    if (val_a < val_b)
        return -1;
    if (val_a > val_b)
        return 1;
    return 0;
}

static inline void publish_slot_release(ConcurrentMemTable t, uint32 i);
static bool test_consistency(Relation heap_rel, LSMIndex lsm_index, int lsm_slot_idx);

void
recover_lsm_index_internal(Oid index_relid, uint32_t slot_idx)
{
    Relation index = index_open(index_relid, AccessShareLock);
    elog(DEBUG1, "[recover_lsm_index_internal] recovering index: relId = %u to slot %u", index_relid, slot_idx);
    
    // Note: Concurrency and status checks are already handled in register_lsm_index()
    // and get_lsm_index_idx()/get_lsm_index() before calling this function.
    // By the time we reach here, the slot should be in LSM_SLOT_RECOVERING with the correct Oid.
    
    LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[slot_idx];

    // read the max memtable sid from the status page
    SegmentId status_growing_sid = GetStatusGrowingSid(index, MAIN_FORKNUM);
    int num_sids;
    SegmentId *sids = GetStatusMemtableSids(index, MAIN_FORKNUM, &num_sids);
    // sort the sids descendingly
    if (sids != NULL && num_sids > 0)
    {
        qsort(sids, num_sids, sizeof(SegmentId), compare_sids_desc);
    }
    Assert(sids[0] <= max_memtable_sid);
    SegmentId max_memtable_sid = sids[0] == status_growing_sid ? status_growing_sid : status_growing_sid - 1;

    /*
     * No index_load_blocking here. The FlushedSegmentPool is populated
     * on demand by the first reader via ensure_index_loaded(); recovery
     * leaves the slot in LSM_SLOT_WRITABLE so writers and standby redo
     * callbacks can proceed without paying segment-load latency.
     */

    LSMIndex lsm = &SharedLSMIndexBuffer->slots[slot_idx].lsmIndex;

    // Read LSM index metadata from disk
    IndexType index_type;
    uint32_t dim, elem_size;
    if (!read_lsm_index_metadata(index_relid, &index_type, &dim, &elem_size))
    {
        elog(ERROR, "[recover_lsm_index_internal] Failed to read LSM index metadata for index %u", index_relid);
        Assert(false);
    }
    // Initialize LSM index structure
    lsm->indexRelId = index_relid;
    lsm->index_type = index_type;
    lsm->dim = dim;
    lsm->elem_size = elem_size;
    
    // Scan segment metadata files to get all segments
    SegmentFileInfo files[MAX_SEGMENTS_COUNT];
    int file_count = scan_segment_metadata_files(index_relid, files, MAX_SEGMENTS_COUNT);

    // Recovery - start timing
    instr_time recovery_start_time;
    INSTR_TIME_SET_CURRENT(recovery_start_time);

    // Recovery
    // For segments that include sids > max_memtable_sid, mark those vectors as deleted
    for (int i = file_count - 1; i >= 0; i--)
    {
        // elog(DEBUG1, "[recover_lsm_index_internal] In recovery step 1, checking segment %u-%u", files[i].start_sid, files[i].end_sid);
        // Check if this segment includes any sids greater than max_memtable_sid
        if (files[i].end_sid > max_memtable_sid)
        {
            // elog(DEBUG1, "[recover_lsm_index_internal] In recovery step 1, processing segment %u-%u", files[i].start_sid, files[i].end_sid);
            SegmentId start_sid_disk, end_sid_disk;
            uint32_t valid_rows;
            IndexType seg_index_type;
            
            // Read segment metadata to get actual sid range and valid_rows
            if (read_lsm_segment_metadata(index_relid, files[i].start_sid, files[i].end_sid, files[i].version,
                                         &start_sid_disk, &end_sid_disk, &valid_rows, &seg_index_type))
            {
                // Only process if the segment actually has sids > max_memtable_sid
                if (end_sid_disk > max_memtable_sid)
                {
                    // Load bitmap file (use latest version)
                    uint8_t *bitmap = NULL;
                    uint32_t delete_count;
                    load_bitmap_file(index_relid, start_sid_disk, end_sid_disk, LOAD_LATEST_VERSION, &bitmap, true, &delete_count);
                    Assert(bitmap != NULL); 
                    
                    // Load offset file to map sids to vector indices (use latest version)
                    SegmentOffsetRange *offsets = NULL;
                    load_offset_file(index_relid, start_sid_disk, end_sid_disk, LOAD_LATEST_VERSION, &offsets, true);
                    Assert(offsets != NULL);
                    
                    // Iterate through all offset ranges and mark vectors with sid > max_memtable_sid
                    uint32_t segment_count = end_sid_disk - start_sid_disk + 1;
                    for (uint32_t sid_idx = 0; sid_idx < segment_count; sid_idx++)
                    {
                        SegmentOffsetRange *offset_range = &offsets[sid_idx];
                        
                        // Only process entries where sid > max_memtable_sid
                        if (offset_range->sid <= max_memtable_sid)
                            continue;
                        
                        // Skip empty ranges (no vectors for this sid)
                        if (offset_range->start_offset == offset_range->end_offset)
                            continue;
                        
                        // Set all bits in the range [start_offset, end_offset) to 1
                        for (Size vec_idx = offset_range->start_offset; vec_idx < offset_range->end_offset; vec_idx++)
                        {
                            if (vec_idx < valid_rows && !IS_SLOT_SET(bitmap, vec_idx))
                            {
                                SET_SLOT(bitmap, vec_idx);
                                delete_count++;
                            }
                        }
                    }
                    
                    // Find latest version and subversion for writing new bitmap
                    uint32_t version = find_latest_segment_version(index_relid, start_sid_disk, end_sid_disk);
                    uint32_t latest_subversion = find_latest_bitmap_subversion(index_relid, start_sid_disk, end_sid_disk, version);
                    uint32_t next_subversion = (latest_subversion == UINT32_MAX) ? 0 : latest_subversion + 1;
                    
                    // Write updated bitmap with new subversion
                    Size bitmap_size = GET_BITMAP_SIZE(valid_rows);
                    write_bitmap_file_with_subversion(index_relid, start_sid_disk, end_sid_disk,
                                                      version, next_subversion, bitmap, bitmap_size, delete_count);

                    /*
                     * No SEGMENT_UPDATE_VACUUM round-trip here: index_load_blocking
                     * has not run yet, and when it does it will pick up this new
                     * bitmap subversion via load_bitmap_file(... LOAD_LATEST_VERSION ...).
                     */

                    elog(DEBUG1, "[recover_lsm_index_internal] Vacuumed segment %u-%u v%u: marked all sids > %u as deleted, new delete_count=%u",
                         start_sid_disk, end_sid_disk, version, max_memtable_sid, delete_count);
                    
                    // Clean up
                    pfree(bitmap);
                    pfree(offsets);
                }
            }
        }
        else {
            break;
        }
    }

    // Recovery step 2: For segments that include sid in sids array, mark vectors as deleted if the corresponding tids are not in the status page
    if (sids != NULL && num_sids > 0 && file_count > 0 && files[file_count - 1].end_sid >= sids[num_sids - 1])
    {
        // TODO: for debugging
        elog(DEBUG1, "[recover_lsm_index_internal] In recovery step 2, checking segments that include sid in sids array");

        int f_idx = file_count - 1;
        int s_idx = 0;
        
        while (f_idx >= 0 && s_idx < num_sids)
        {
            // skip the sid if it is greater than the end_sid of the largest segment
            if (sids[s_idx] > files[f_idx].end_sid)
            {
                s_idx++;
                continue;
            }
            // check whether the segment contains any sid in sids array
            if (files[f_idx].start_sid <= sids[s_idx] && files[f_idx].end_sid >= sids[s_idx])
            {
                // TODO: for debugging
                elog(DEBUG1, "[recover_lsm_index_internal] In recovery step 2, processing segment %u-%u", files[f_idx].start_sid, files[f_idx].end_sid);
                SegmentId start_sid_disk, end_sid_disk;
                uint32_t valid_rows;
                IndexType seg_index_type;
                
                // Read segment metadata to get actual sid range and valid_rows
                if (read_lsm_segment_metadata(index_relid, files[f_idx].start_sid, files[f_idx].end_sid, files[f_idx].version,
                                             &start_sid_disk, &end_sid_disk, &valid_rows, &seg_index_type))
                {
                    // load the mapping file, bitmap file and offset file
                    int64_t *mapping = NULL;
                    load_mapping_file(index_relid, start_sid_disk, end_sid_disk, LOAD_LATEST_VERSION, &mapping, true);
                    Assert(mapping != NULL);
                    
                    uint8_t *bitmap = NULL;
                    uint32_t delete_count;
                    load_bitmap_file(index_relid, start_sid_disk, end_sid_disk, LOAD_LATEST_VERSION, &bitmap, true, &delete_count);
                    Assert(bitmap != NULL);
                    
                    SegmentOffsetRange *offsets = NULL;
                    load_offset_file(index_relid, start_sid_disk, end_sid_disk, LOAD_LATEST_VERSION, &offsets, true);
                    Assert(offsets != NULL);
                    
                    bool bitmap_changed = false;
                    uint32_t segment_count = end_sid_disk - start_sid_disk + 1;
                    
                    // find all sids that are in sids array
                    for (uint32_t sid_idx = 0; sid_idx < segment_count; sid_idx++)
                    {
                        SegmentOffsetRange *offset_range = &offsets[sid_idx];
                        SegmentId current_sid = offset_range->sid;
                        
                        // Check if this sid is in the sids array
                        bool sid_in_array = false;
                        for (int check_s_idx = s_idx; check_s_idx < num_sids; check_s_idx++)
                        {
                            if (sids[check_s_idx] == current_sid)
                            {
                                sid_in_array = true;
                                break;
                            }
                        }
                        
                        if (!sid_in_array)
                            continue;
                        
                        // Skip empty ranges (no vectors for this sid)
                        if (offset_range->start_offset == offset_range->end_offset)
                            continue;
                        
                        // For each sid, Call GetStatusMemtableTids to get the tids in the status page
                        int num_status_tids = 0;
                        ItemPointerData *status_tids = GetStatusMemtableTids(index, MAIN_FORKNUM, current_sid, &num_status_tids);
                        Assert(status_tids != NULL && num_status_tids > 0);

                        // FIXME: this part can be optimized
                        // Convert status tids to int64 and sort for efficient lookup
                        int64_t *status_tids_int64 = (int64_t *) palloc(sizeof(int64_t) * num_status_tids);
                        for (int t_idx = 0; t_idx < num_status_tids; t_idx++)
                        {
                            status_tids_int64[t_idx] = ItemPointerToInt64(&status_tids[t_idx]);
                        }
                        
                        // Sort for binary search (tids are approximately in order, but sorting ensures correctness)
                        qsort(status_tids_int64, num_status_tids, sizeof(int64_t), compare_int64);
                        
                        // Generate new bitmap for this sid's range
                        Size range_start = offset_range->start_offset;
                        Size range_end = offset_range->end_offset;
                        
                        if (range_start < range_end)
                        {
                            // Generate bitmap: set bit to 1 if tid is NOT in status_tids
                            // Use binary search for efficient lookup (O(log n) per lookup)
                            // Since tids are approximately in order, binary search is efficient
                            uint32_t additional_deleted = 0;
                            
                            for (Size vec_idx = range_start; vec_idx < range_end; vec_idx++)
                            {
                                int64_t vec_tid = mapping[vec_idx];
                                
                                // Binary search to check if tid is in status_tids
                                int64_t *found = (int64_t *) bsearch(&vec_tid, status_tids_int64, num_status_tids, 
                                                                        sizeof(int64_t), compare_int64);
                                
                                // If tid not found in status page, mark as deleted
                                if (found == NULL)
                                {
                                    // Check if bit was already set (preserve existing deletions)
                                    if (!IS_SLOT_SET(bitmap, vec_idx))
                                    {
                                        SET_SLOT(bitmap, vec_idx);
                                        additional_deleted++;
                                    }
                                }
                            }
                            
                            delete_count += additional_deleted;
                            if (additional_deleted > 0)
                                bitmap_changed = true;
                        }
                        
                        pfree(status_tids_int64);
                        
                        if (status_tids != NULL)
                        {
                            pfree(status_tids);
                        }
                    }
                    
                    // Flush the modified bitmap to disk
                    if (bitmap_changed)
                    {
                        uint32_t version = find_latest_segment_version(index_relid, start_sid_disk, end_sid_disk);
                        uint32_t latest_subversion = find_latest_bitmap_subversion(index_relid, start_sid_disk, end_sid_disk, version);
                        uint32_t next_subversion = (latest_subversion == UINT32_MAX) ? 0 : latest_subversion + 1;
                        
                        Size bitmap_size = GET_BITMAP_SIZE(valid_rows);
                        write_bitmap_file_with_subversion(index_relid, start_sid_disk, end_sid_disk,
                                                          version, next_subversion, bitmap, bitmap_size, delete_count);

                        /*
                         * No SEGMENT_UPDATE_VACUUM round-trip here: same reasoning
                         * as recovery step 1. The bitmap subversion on disk is the
                         * source of truth; the next load reads it directly.
                         */

                        elog(DEBUG1, "[recover_lsm_index_internal] Vacuumed segment %u-%u v%u: marked vectors with missing tids as deleted, new delete_count=%u",
                             start_sid_disk, end_sid_disk, version, delete_count);
                    }
                    
                    // Clean up
                    pfree(mapping);
                    pfree(bitmap);
                    pfree(offsets);
                }
            }

            f_idx--;
        }
    }

    // Recovery - end timing and output overhead
    instr_time recovery_end_time;
    INSTR_TIME_SET_CURRENT(recovery_end_time);
    INSTR_TIME_SUBTRACT(recovery_end_time, recovery_start_time);
    double recovery_overhead_ms = INSTR_TIME_GET_MILLISEC(recovery_end_time);
    elog(DEBUG1, "[recover_lsm_index_internal] Recovery step 1&2 overhead: %.3f ms", recovery_overhead_ms);
    
    // Initialize sealed memtable array
    for (int i = 0; i < MEMTABLE_NUM; i++)
    {
        lsm->memtable_idxs[i] = -1;
    }
    lsm->memtable_count = 0;
    // Initialize and update the flushed_not_released array
    pg_atomic_init_u32(&lsm->flushed_not_released_count, 0);
    pg_atomic_init_u32(&lsm->releasing_in_progress, 0);
    for (int sid_idx = num_sids - 1; sid_idx >= 0; sid_idx--)
    {
        if (sids[sid_idx] <= files[file_count - 1].end_sid)
        {
            uint32_t count = pg_atomic_read_u32(&lsm->flushed_not_released_count);
            lsm->flushed_not_released[count] = sids[sid_idx];
            pg_atomic_write_u32(&lsm->flushed_not_released_count, count + 1);
        }        
        else {
            break;
        }
    }

    // prepare the heap relation
    Oid heap_relid = index->rd_index->indrelid;
    AttrNumber attnum = index->rd_index->indkey.values[0];
    Relation heap_rel = table_open(heap_relid, AccessShareLock);
    TupleTableSlot *heap_slot = table_slot_create(heap_rel, NULL);

    bool growing_memtable_allocated = false;

    // Recovery - start timing
    INSTR_TIME_SET_CURRENT(recovery_start_time);

    // Recovery step 3: construct the memtables
    for (int i = num_sids - 1; i >= 0; i--)
    {
        // those sids are already flushed to disk as segments
        if (sids[i] <= files[file_count - 1].end_sid)
        {
            continue;
        }

        // read status pages to get the tids
        int num_status_tids = 0;
        ItemPointerData *status_tids = GetStatusMemtableTids(index, MAIN_FORKNUM, sids[i], &num_status_tids);

        ConcurrentMemTable current_mt;
        int mt_idx;
        // allocate a new growing memtable
        mt_idx = allocate_new_growing_memtable(lsm, index, true, sids[i]);
        current_mt = MT_FROM_SLOTIDX(mt_idx);

        // iterate the tids to construct the memtable
        for (int t_idx = 0; t_idx < num_status_tids; t_idx++)
        {            
            if (table_tuple_fetch_row_version(heap_rel, &status_tids[t_idx], SnapshotAny, heap_slot)) {
                
                uint32_t i = pg_atomic_fetch_add_u32(&current_mt->current_size, 1);
                current_mt->tids[i] = ItemPointerToInt64(&status_tids[t_idx]);

                // extract the vector attribute
                bool isnull;
                Datum vector_datum = slot_getattr(heap_slot, attnum, &isnull);
                if (isnull)
                    elog(ERROR, "failed to fetch the vector data in the LSM index recovery phase");

                Vector *vec = (Vector *) PG_DETOAST_DATUM_COPY(vector_datum);
                
                memcpy(VEC_PTR_AT(current_mt, i), vec->x, VEC_BYTES_PER_ROW(current_mt));

                pfree(vec);

                publish_slot_release(current_mt, i);
            }
            else {
                elog(DEBUG1, "[recover_lsm_index_internal] failed to fetch the vector data in the LSM index recovery phase, potential table pruning detected");
            }
        }

        if (sids[i] == status_growing_sid) { // the current sid is the growing memtable
            lsm->growing_memtable_idx = mt_idx;
            lsm->growing_memtable_id = MT_FROM_SLOTIDX(mt_idx)->memtable_id;
            // update the next_segment_id
            pg_atomic_write_u32(&lsm->next_segment_id, lsm->growing_memtable_id + 1);
            growing_memtable_allocated = true;
        }
        else {
            // seal the memtable
            pg_atomic_write_u32(&current_mt->sealed, 1);
            // enqueue the sealed memtable
            if (lsm->memtable_count < MEMTABLE_NUM)
            {
                lsm->memtable_idxs[lsm->memtable_count++] = mt_idx;
            }
            else {
                elog(ERROR, "[recover_lsm_index_internal] no free slot to register a new memtable");
            }
        }
    }

    // Recovery - end timing and output overhead
    INSTR_TIME_SET_CURRENT(recovery_end_time);
    INSTR_TIME_SUBTRACT(recovery_end_time, recovery_start_time);
    recovery_overhead_ms = INSTR_TIME_GET_MILLISEC(recovery_end_time);
    elog(DEBUG1, "[recover_lsm_index_internal] Recovery step 3 overhead: %.3f ms", recovery_overhead_ms);

    if (!growing_memtable_allocated) {
        // update the next_segment_id
        pg_atomic_write_u32(&lsm->next_segment_id, sids[0] + 1);
        // we should not allocate a new growing memtable here, it will be allocated in the insert path (within a transaction)
    }
    
    pg_write_barrier();

    // // test the consistency of the LSM index
    // bool consistency_ok = test_consistency(heap_rel, lsm, slot_idx);
    // if (!consistency_ok) {
    //     elog(ERROR, "[recover_lsm_index_internal] consistency check failed for LSM index %u", index_relid);
    //     Assert(false);
    // }

    table_close(heap_rel, AccessShareLock);
    index_close(index, AccessShareLock);
    /*
     * Publish LSM_SLOT_WRITABLE. The FlushedSegmentPool is NOT yet
     * initialized — readers must call ensure_index_loaded() to drive the
     * WRITABLE -> LOADING_INDEX -> QUERYABLE transition. Writers and
     * standby redo callbacks operate on the memtable only and may proceed.
     */
    pg_write_barrier();
    pg_atomic_write_u32(&slot->valid, (uint32) LSM_SLOT_WRITABLE);

    elog(DEBUG1, "[recover_lsm_index_internal] successfully recovered LSM index %u (WRITABLE)", index_relid);
}

// no recovery here, just check if the index is in the buffer and if it is writable; wait if it is recovering
int
lookup_lsm_index_idx(Oid index_relid)
{
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        uint32 valid = pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid);
        if (valid == (uint32) LSM_SLOT_FREE ||
            SharedLSMIndexBuffer->slots[i].lsmIndex.indexRelId != index_relid)
            continue;

        if (is_writable(valid))
            return i;

        /* RECOVERING: wait for the IndexRecoveryWorker to finish recovery */
        LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[i];
        ConditionVariablePrepareToSleep(&slot->state_cv);
        while (pg_atomic_read_u32(&slot->valid) == (uint32) LSM_SLOT_RECOVERING)
            ConditionVariableSleep(&slot->state_cv, PG_WAIT_EXTENSION);
        ConditionVariableCancelSleep();

        if (is_writable(pg_atomic_read_u32(&slot->valid)) &&
            pg_atomic_read_u32(&slot->state_error) == 0)
            return i;

        elog(DEBUG1, "[lookup_lsm_index_idx] index %u recovery failed", index_relid);
        return -1;
    }

    elog(DEBUG1, "[lookup_lsm_index_idx] index %u not in buffer", index_relid);
    return -1;
}

// Resolve indexRelId to a slot index. See header for parameter contract.
int
get_lsm_index_idx(Oid index_relid, bool for_redo, Oid db_oid)
{
    /* Fast path: lock-free scan for an already-registered slot. */
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        uint32 valid = pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid);
        if (valid == (uint32) LSM_SLOT_FREE ||
            SharedLSMIndexBuffer->slots[i].lsmIndex.indexRelId != index_relid)
            continue;

        if (is_writable(valid))
            return i;

        /* RECOVERING: worker is running recovery — wait on CV */
        {
            LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[i];
            ConditionVariablePrepareToSleep(&slot->state_cv);
            while (pg_atomic_read_u32(&slot->valid) == (uint32) LSM_SLOT_RECOVERING)
                ConditionVariableSleep(&slot->state_cv, PG_WAIT_EXTENSION);
            ConditionVariableCancelSleep();

            if (is_writable(pg_atomic_read_u32(&slot->valid)) &&
                pg_atomic_read_u32(&slot->state_error) == 0)
                return i;

            elog(ERROR, "[get_lsm_index_idx] index %u recovery failed (state_error=%u, valid=%u)",
                 index_relid,
                 pg_atomic_read_u32(&slot->state_error),
                 pg_atomic_read_u32(&slot->valid));
        }
    }

    /*
     * Slow path: not in buffer — claim a slot and signal recovery. Branch
     * on for_redo to pick the right registration entry point (the WAL
     * redo path has no MyDatabaseId / GetUserId(); db_oid comes from the
     * WAL record instead).
     */
    elog(DEBUG1, "[get_lsm_index_idx] the requested lsm_index is not in the buffer");
    if (for_redo)
        return register_lsm_index_for_redo(db_oid, index_relid);
    return register_lsm_index(index_relid);
}

/*
 * dpv_initial_segment_on_disk — true if <storage_base>/<oid>/ contains at
 * least one segment-metadata file whose sid range covers START_SEGMENT_ID,
 * i.e. the initial-build segment or a successor that subsumed it via merge.
 *
 * Used as the standby queryability barrier: a SELECT against an index built
 * after replication started must wait for the build's segment files to be
 * fetched before the first WRITABLE->LOADING_INDEX transition. Without this
 * gate, index_load_blocking would find an empty <oid>/ and the slot would
 * publish QUERYABLE over zero segments, violating heap<->index consistency
 * for the WAL-replayed-but-not-yet-fetched window.
 */
static bool
dpv_initial_segment_on_disk(Oid indexRelId)
{
    SegmentFileInfo files[MAX_SEGMENTS_COUNT];
    int n = scan_segment_metadata_files(indexRelId, files, MAX_SEGMENTS_COUNT);
    for (int i = 0; i < n; i++)
    {
        if (files[i].start_sid <= START_SEGMENT_ID &&
            START_SEGMENT_ID <= files[i].end_sid)
            return true;
    }
    return false;
}

/*
 * ensure_index_loaded — drive the WRITABLE -> LOADING_INDEX -> QUERYABLE
 * transition for slot_idx and block until QUERYABLE.
 *
 * Preconditions:
 *   - slot_idx is a valid index returned by register_lsm_index() / friends.
 *   - The slot is at least WRITABLE.
 *
 * Returns true if the slot reaches QUERYABLE. Returns false if a concurrent
 * load attempt failed; the caller should propagate an error (the slot
 * reverts to WRITABLE and state_error is set).
 */
static bool
ensure_index_loaded(int slot_idx)
{
    LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[slot_idx];
    uint32 v = pg_atomic_read_u32(&slot->valid);

    if (v == (uint32) LSM_SLOT_QUERYABLE)
        return true;

    if (v == (uint32) LSM_SLOT_WRITABLE)
    {
        uint32 expected = (uint32) LSM_SLOT_WRITABLE;
        if (pg_atomic_compare_exchange_u32(&slot->valid, &expected,
                                           (uint32) LSM_SLOT_LOADING_INDEX))
        {
            /* Winner: clear any stale state_error from a prior attempt. */
            pg_atomic_write_u32(&slot->state_error, 0);

            PG_TRY();
            {
                /*
                 * Plan 2 §10 standby queryability barrier (CREATE INDEX
                 * after replication start). On a hot standby with the side
                 * channel active, we may be the first SELECT after the
                 * SegmentCreated WAL was redone but BEFORE the fetcher
                 * pulled its files. Loading now would publish QUERYABLE
                 * over an empty pool and the heap<->index consistency
                 * invariant would break.
                 *
                 * Gate on dpv_replication_role == DPV_ROLE_STANDBY (not
                 * just RecoveryInProgress): in non-replicated setups (e.g.,
                 * Plan 1 tests with role=DISABLED but hot_standby=on) no
                 * segment fetcher is running, so waiting for a file that
                 * will never arrive would always time out and ERROR out
                 * legitimate Plan 1 queries that only need memtable
                 * reconstruction from heap. The fetcher exists if and only
                 * if role=STANDBY; that's the precise condition under
                 * which we should wait.
                 *
                 * We block here, holding LOADING_INDEX so concurrent
                 * readers wait on state_cv exactly once. Only the
                 * initial-build segment is required; later segments
                 * arriving via WAL+adoption are integrated asynchronously
                 * after first load (Plan 2 §10's stated v1 behavior).
                 */
                if (RecoveryInProgress() &&
                    dpv_replication_role == DPV_ROLE_STANDBY)
                {
                    Oid     relid     = slot->lsmIndex.indexRelId;
                    int     waited_ms = 0;

                    while (!dpv_initial_segment_on_disk(relid))
                    {
                        if (waited_ms >= dpv_replication_fetch_wait_timeout_ms)
                            ereport(ERROR,
                                    (errcode(ERRCODE_T_R_DEADLOCK_DETECTED),
                                     errmsg("standby fetch barrier: initial-build segment "
                                            "for index %u not yet fetched after %d ms",
                                            relid,
                                            dpv_replication_fetch_wait_timeout_ms),
                                     errhint("Increase pgvector.replication_fetch_wait_timeout "
                                             "or wait for the segment fetcher to catch up.")));
                        pg_usleep(100 * 1000);  /* 100 ms */
                        waited_ms += 100;
                        CHECK_FOR_INTERRUPTS();
                    }
                }

                index_load_blocking(slot->lsmIndex.indexRelId, slot_idx);
                pg_write_barrier();
                pg_atomic_write_u32(&slot->valid, (uint32) LSM_SLOT_QUERYABLE);
            }
            PG_CATCH();
            {
                pg_atomic_write_u32(&slot->state_error, 1);
                pg_atomic_write_u32(&slot->valid, (uint32) LSM_SLOT_WRITABLE);
                ConditionVariableBroadcast(&slot->state_cv);
                PG_RE_THROW();
            }
            PG_END_TRY();

            ConditionVariableBroadcast(&slot->state_cv);
            return true;
        }
        /* CAS lost — someone else owns the transition; fall through to wait. */
    }

    /*
     * Wait until the slot leaves LOADING_INDEX.
     *   - QUERYABLE: success.
     *   - WRITABLE: the winner errored and reverted; we report failure.
     */
    PG_TRY();
    {
        ConditionVariablePrepareToSleep(&slot->state_cv);
        for (;;)
        {
            v = pg_atomic_read_u32(&slot->valid);
            if (v == (uint32) LSM_SLOT_QUERYABLE)
            {
                ConditionVariableCancelSleep();
                return true;
            }
            if (v == (uint32) LSM_SLOT_WRITABLE)
            {
                ConditionVariableCancelSleep();
                return false;
            }
            ConditionVariableSleep(&slot->state_cv, PG_WAIT_EXTENSION);
        }
    }
    PG_CATCH();
    {
        ConditionVariableCancelSleep();
        PG_RE_THROW();
    }
    PG_END_TRY();
}

LSMIndex
get_lsm_index(Relation index)
{
    Oid index_relid = RelationGetRelid(index);
    int slot_num;

    /* Fast path: scan for an already-registered slot */
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        uint32 valid = pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid);
        if (valid == (uint32) LSM_SLOT_FREE ||
            SharedLSMIndexBuffer->slots[i].lsmIndex.indexRelId != index_relid)
            continue;

        if (is_writable(valid))
            return &SharedLSMIndexBuffer->slots[i].lsmIndex;

        /* RECOVERING: worker is running recovery — wait on CV */
        {
            LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[i];
            ConditionVariablePrepareToSleep(&slot->state_cv);
            while (pg_atomic_read_u32(&slot->valid) == (uint32) LSM_SLOT_RECOVERING)
                ConditionVariableSleep(&slot->state_cv, PG_WAIT_EXTENSION);
            ConditionVariableCancelSleep();

            if (is_writable(pg_atomic_read_u32(&slot->valid)) &&
                pg_atomic_read_u32(&slot->state_error) == 0)
                return &slot->lsmIndex;

            elog(ERROR, "[get_lsm_index] index %u recovery failed (state_error=%u, valid=%u)",
                 index_relid,
                 pg_atomic_read_u32(&slot->state_error),
                 pg_atomic_read_u32(&slot->valid));
        }
    }

    /* Slow path: not in buffer — register and wait for worker to load */
    elog(DEBUG1, "[get_lsm_index] the requested lsm_index is not in the buffer");
    slot_num = register_lsm_index(index_relid);
    return &SharedLSMIndexBuffer->slots[slot_num].lsmIndex;
}

LSMIndex
get_lsm_index_for_read(Relation index)
{
    Oid index_relid = RelationGetRelid(index);
    int slot_num = -1;

    /* Resolve to a slot index; reuse the fast-path logic of get_lsm_index_idx. */
    slot_num = get_lsm_index_idx(index_relid, false, InvalidOid);
    if (slot_num < 0)
        elog(ERROR, "[get_lsm_index_for_read] failed to resolve LSM index slot for %u",
             index_relid);

    if (!ensure_index_loaded(slot_num))
        elog(ERROR, "[get_lsm_index_for_read] index %u load failed (state_error=%u)",
             index_relid,
             pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[slot_num].state_error));

    return &SharedLSMIndexBuffer->slots[slot_num].lsmIndex;
}

/* update_max_ready_id is now a static inline in lsmindex.h */

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

/* publish_slot_release is now a static inline in lsmindex.h */

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
rotate_growing_memtable(LSMIndex lsm, Relation index, bool need_enqueue)
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

    if (cur_idx >= MEMTABLE_BUF_SIZE)
    {
        LWLockRelease(lsm->mt_lock);
        elog(ERROR, "[rotate_growing_memtable] invalid growing_memtable_idx = %d", cur_idx);
    }

    if (need_enqueue)
    {
        cur = MT_FROM_SLOTIDX(cur_idx);
        cur_id = cur->memtable_id;

        // if not sealed, no need to rotate
        if (pg_atomic_read_u32(&cur->sealed) == 0)
        {
            LWLockRelease(lsm->mt_lock);
            return;
        }
        lsm->growing_memtable_id  = cur_id;
    }
    lsm->growing_memtable_idx = MT_IDX_ROTATING;

    LWLockRelease(lsm->mt_lock);

    int new_idx = allocate_new_growing_memtable(lsm, index, false, NULL);

    if (need_enqueue)
        enqueue_sealed_memtable(lsm, cur_idx);

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
    LSMIndex lsm_index = get_lsm_index(index);

    // Check for flushed but not released memtables and release them
    // Only one backend should release at a time to avoid duplicate work
    uint32_t count = pg_atomic_read_u32(&lsm_index->flushed_not_released_count);
    if (count > 0)
    {
        // Check if another backend is already releasing
        uint32_t expected = 0;
        if (pg_atomic_compare_exchange_u32(&lsm_index->releasing_in_progress, &expected, 1))
        {
            // We successfully claimed the releasing task
            // Now acquire the lock before releasing
            LWLockAcquire(lsm_index->flushed_release_lock, LW_EXCLUSIVE);
            
            // Re-read count while holding lock (it might have changed)
            count = pg_atomic_read_u32(&lsm_index->flushed_not_released_count);
            if (count > 0)
            {
                // Read barrier to ensure we see the array writes made before count was incremented
                pg_read_barrier();
                
                // Atomically clear the count and get the items to release
                uint32_t items_to_release = pg_atomic_exchange_u32(&lsm_index->flushed_not_released_count, 0);
                
                // Release all flushed memtables from status pages
                for (uint32_t i = 0; i < items_to_release && i < MEMTABLE_NUM; i++)
                {
                    ReleaseStatusMemtable(index, lsm_index->flushed_not_released[i]);
                }
            }
            
            // Release lock and clear the flag
            LWLockRelease(lsm_index->flushed_release_lock);
            pg_atomic_write_u32(&lsm_index->releasing_in_progress, 0);
        }
        // If another backend is already releasing, skip (they'll handle it)
    }

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

        if (gidx == MT_IDX_INVALID)
        {
            LWLockRelease(lsm_index->mt_lock);
            rotate_growing_memtable(lsm_index, index, false);
            continue;
        }
        
        if (gidx >= MEMTABLE_BUF_SIZE)
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

            /* update the status page (GenericXLog page-level WAL) */
            AddToStatusMemtable(index, MAIN_FORKNUM, mt->memtable_id, i, Int64ToItemPointer(tid));

            /*
             * Semantic WAL for replication: carries slot_index + tid + vector
             * data so the standby can materialize this memtable entry without
             * doing a heap fetch.
             */
            {
                ItemPointerData tid_ip = Int64ToItemPointer(tid);
                dpv_emit_add_to_memtable(RelationGetRelid(index), mt->memtable_id,
                                         i, &tid_ip,
                                         VEC_PTR_AT(mt, i), VEC_BYTES_PER_ROW(mt));
            }

            publish_slot_release(mt, i);
            return;
        }

        // slow path: current growing memtable is sealed. 
        rotate_growing_memtable(lsm_index, index, true);        
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

/*
 * Same streaming top-K merge as merge_top_k, but drops duplicate TIDs:
 * before appending a candidate, linearly scan the already-output prefix and
 * skip if the id was already emitted. O(top_k^2) worst case; fine for the
 * typical k=10..100 search width.
 *
 * Needed at the memtable+segment merge in search_lsm_index because a merged
 * segment whose [start..end] range partially overlaps the backend's memtable
 * snapshot is now searched on both sides (the worker because the segment is
 * not fully covered, the backend because some of its sids still belong to
 * sealed memtables). The overlapping slice produces identical TIDs from
 * both sources; this dedup keeps each TID exactly once.
 */
static int
merge_top_k_dedup(DistancePair *pairs_1, DistancePair *pairs_2,
                  int num_1, int num_2, int top_k, DistancePair *merge_pair)
{
    int i = 0, j = 0, k = 0;

    while (k < top_k && (i < num_1 || j < num_2))
    {
        DistancePair candidate;
        bool dup = false;
        int m;

        if (i < num_1 && (j >= num_2 || pairs_1[i].distance <= pairs_2[j].distance))
            candidate = pairs_1[i++];
        else
            candidate = pairs_2[j++];

        for (m = 0; m < k; m++)
        {
            if (merge_pair[m].id == candidate.id)
            {
                dup = true;
                break;
            }
        }
        if (!dup)
            merge_pair[k++] = candidate;
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
    /*
     * Read path: must ensure the FlushedSegmentPool is populated
     * (QUERYABLE), not just that recovery is done.
     */
    LSMIndex lsm = get_lsm_index_for_read(index);
    Assert(lsm);

    // step 1. get a snapshot of the current memtables
    LSMSnapshot lsm_snapshot;

    /*
     * [DIAG-H1] insert_lsm_index / bulk_delete_lsm_index both treat
     * MT_IDX_ROTATING (-2) and MT_IDX_INVALID (-1) as a transient state and
     * spin until growing_memtable_idx becomes a valid slot index. The search
     * path previously dereferenced MT_FROM_SLOTIDX(gidx) and indexed
     * SharedMemtableBuffer->slots[gidx] unconditionally, which is an OOB
     * read/atomic-write into shared memory whenever rotate_growing_memtable
     * is between line ~1583 (set ROTATING, release lock) and line ~1593
     * (set new idx). Spin-retry to match the insert/vacuum paths and log
     * every occurrence so we can correlate with the recall drop.
     */
    for (;;)
    {
        LWLockAcquire(lsm->mt_lock, LW_SHARED);

        lsm_snapshot.gidx = lsm->growing_memtable_idx;
        if (lsm_snapshot.gidx >= 0 && lsm_snapshot.gidx < MEMTABLE_BUF_SIZE)
            break;  /* valid slot, proceed under the held SHARED lock */

        LWLockRelease(lsm->mt_lock);
        fprintf(stderr,
                "[search_lsm_index][DIAG-H1] hit transient gidx=%d (ROTATING=-2/INVALID=-1), retrying\n",
                lsm_snapshot.gidx);
        pg_usleep(50);
    }

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
    }
    
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
    // dedup by TID: a merged segment that partially overlaps the snapshot is
    // now searched on both sides, so the overlapping slice yields identical
    // TIDs from memtable and segment results.
    final_pairs = palloc(sizeof(DistancePair) * k);
    int final_num = merge_top_k_dedup(pair_1, segment_pairs, num_1, vs_result->result_count, k, final_pairs);
    pfree(pair_1);
    pfree(segment_pairs);

    TopKTuples topk_result = {
        .num_results = final_num,
        .pairs = final_pairs
    };
    return topk_result;
}

IndexBulkDeleteResult *
bulk_delete_lsm_index(Relation index, IndexBulkDeleteResult *stats, IndexBulkDeleteCallback callback, void *callback_state)
{
    elog(DEBUG1, "[bulk_delete_lsm_index] enter bulk_delete_lsm_index");
    if (stats == NULL)
    {
        stats = (IndexBulkDeleteResult *) palloc0(sizeof(IndexBulkDeleteResult));
    }
    
    int lsm_idx = get_lsm_index_idx(RelationGetRelid(index), false, InvalidOid);
    if (lsm_idx < 0)
    {
        return stats;
    }
    
    Oid indexRelId = index->rd_id;
    LSMIndex lsm = &SharedLSMIndexBuffer->slots[lsm_idx].lsmIndex;
    
    // Track memtable IDs that have been vacuumed to avoid duplicates
    SegmentId vacuumed_memtable_ids[MEMTABLE_NUM + 1]; // +1 for growing memtable
    uint32_t vacuumed_count = 0;
    
    // step 1. vacuum the growing memtable
    /*
    * 1. acquire the vacuum lock of the growing memtable
    * 2. iterate over all ready slots in the growing memtable
    * 3. for all ready slots in the growing memtable, use the callback function to check if the corresponding vector is deleted
    * 4. if deleted, mark the slot as deleted in the bitmap
    * 5. if not deleted, continue
    * 6. check if the growing memtable is persistent on disk
    * 7. if persistent, flush the growing memtable to disk
    * 8. notify the index worker to update the bitmap (latest segment files)
    * 9. release the vacuum lock of the growing memtable
    */
    // elog(DEBUG1, "[bulk_delete_lsm_index] step 1. vacuum the growing memtable");
    SegmentId growing_memtable_id = 0;  // 0 indicates no growing memtable
    {
        int gidx = -1;
        LWLockAcquire(lsm->mt_lock, LW_SHARED);
        gidx = lsm->growing_memtable_idx;
        
        if (gidx >= 0 && gidx != MT_IDX_INVALID && gidx != MT_IDX_ROTATING)
        {
            // Increment reference count while holding mt_lock
            pg_atomic_add_fetch_u32(&SharedMemtableBuffer->slots[gidx].ref_count, 1);
            growing_memtable_id = MT_FROM_SLOTIDX(gidx)->memtable_id;
        }
        LWLockRelease(lsm->mt_lock);
        
        if (gidx >= 0 && gidx != MT_IDX_INVALID && gidx != MT_IDX_ROTATING)
        {
            ConcurrentMemTable mt = MT_FROM_SLOTIDX(gidx);
            
            // Acquire vacuum lock
            LWLockAcquire(&mt->vacuum_lock, LW_EXCLUSIVE);
            
            // Iterate over all ready slots
            uint32_t current_size = pg_atomic_read_u32(&mt->current_size);
            uint32_t valid_size = (current_size > mt->capacity) ? mt->capacity : current_size;

            bool bitmap_changed = false;

            /* Plan 3 refactor: collect entries for the unified per-sid WAL.
             * The growing memtable has exactly one sid (mt->memtable_id);
             * we emit one xl_dpv_vacuum_tombstones record per vacuum batch. */
            xl_dpv_vacuum_entry *vacuum_entries = NULL;
            int vacuum_entry_count = 0;
            if (valid_size > 0 && dpv_replication_role == DPV_ROLE_PRIMARY)
                vacuum_entries = (xl_dpv_vacuum_entry *)
                    palloc(sizeof(xl_dpv_vacuum_entry) * valid_size);

            // Iterate over ready slots (for growing memtable, we check ready flag)
            for (uint32_t i = 0; i < valid_size; i++)
            {
                if (mt->ready[i] == 1)  // Slot is ready
                {
                    int64_t tid_int64 = mt->tids[i];
                    ItemPointerData tid = Int64ToItemPointer(tid_int64);
                    if (callback(&tid, callback_state))
                    {
                        // Vector is deleted, mark it in bitmap
                        SET_SLOT(mt->bitmap, i);
                        // remove the tid from the status page
                        RemoveFromStatusMemtable(index, MAIN_FORKNUM, mt->memtable_id, i, tid);
                        bitmap_changed = true;
                        stats->tuples_removed++;
                        if (vacuum_entries)
                        {
                            vacuum_entries[vacuum_entry_count].sid_local_idx = i;
                            vacuum_entries[vacuum_entry_count].tid           = tid_int64;
                            vacuum_entry_count++;
                        }
                    }
                    else {
                        stats->num_index_tuples++;
                    }
                }
            }

            // Check if memtable is persistent on disk (has been flushed)
            // A memtable is persistent if there's a segment file for its memtable_id
            SegmentId start_sid_disk, end_sid_disk;
            uint32_t valid_rows;
            IndexType seg_index_type;
            bool is_persistent = read_lsm_segment_metadata(indexRelId, mt->memtable_id, mt->memtable_id, 
                                                          UINT32_MAX, &start_sid_disk, &end_sid_disk, 
                                                          &valid_rows, &seg_index_type);
            
            if (bitmap_changed)
            {
                // Calculate delete_count from bitmap
                uint32_t delete_count = 0;
                for (uint32_t i = 0; i < valid_size; i++)
                {
                    if (IS_SLOT_SET(mt->bitmap, i))
                    {
                        delete_count++;
                    }
                }
                if (is_persistent)
                {
                    // Find latest version and subversion
                    uint32_t version = find_latest_segment_version(indexRelId, mt->memtable_id, mt->memtable_id);
                    uint32_t latest_subversion = find_latest_bitmap_subversion(indexRelId, mt->memtable_id, mt->memtable_id, version);
                    uint32_t next_subversion = (latest_subversion == UINT32_MAX) ? 0 : latest_subversion + 1;

                    /*
                     * Spec §11: WAL must be emitted AND flushed BEFORE the
                     * subversion bitmap file hits disk, so a standby that
                     * adopted this memtable as a segment can replay vacuum
                     * tombstones in order with merges/rebuilds.
                     */
                    if (dpv_replication_role == DPV_ROLE_PRIMARY)
                    {
                        XLogRecPtr lsn = dpv_emit_vacuum_tombstones(
                            indexRelId,
                            mt->memtable_id,             /* sid */
                            version,                     /* owner_version */
                            next_subversion,
                            /* is_memtable_owner */ false,
                            vacuum_entries, vacuum_entry_count);
                        XLogFlush(lsn);
                    }

                    // Write updated bitmap with subversion
                    write_bitmap_file_with_subversion(indexRelId, mt->memtable_id, mt->memtable_id,
                                                    version, next_subversion, mt->bitmap,
                                                    GET_BITMAP_SIZE(valid_size), delete_count);

                    // notify the index worker to update the bitmap
                    (void) segment_update_blocking(lsm_idx, indexRelId, SEGMENT_UPDATE_VACUUM,
                                                   start_sid_disk, end_sid_disk, 0,
                                                   NULL, 0, NULL, 0);
                    elog(DEBUG1, "[bulk_delete_lsm_index] notified the index worker to update the bitmap for growing memtable %u", mt->memtable_id);
                }
                else {
                    if (dpv_replication_role == DPV_ROLE_PRIMARY &&
                        vacuum_entry_count > 0)
                    {
                        /* Memtable not yet flushed: subversion N/A (use sentinel).
                         * No XLogFlush needed — memtables have no §11 ordering
                         * invariant to protect (their bitmap-for-memtable file
                         * is advisory; the standby reconstructs from WAL). */
                        (void) dpv_emit_vacuum_tombstones(
                            indexRelId,
                            mt->memtable_id,             /* sid */
                            0,                           /* owner_version */
                            UINT32_MAX,                  /* subversion sentinel */
                            /* is_memtable_owner */ true,
                            vacuum_entries, vacuum_entry_count);
                    }
                    write_bitmap_for_memtable(indexRelId, mt->memtable_id, mt->bitmap, GET_BITMAP_SIZE(valid_size), delete_count);
                    elog(DEBUG1, "[bulk_delete_lsm_index] wrote the bitmap for growing memtable %u to disk", mt->memtable_id);
                }
            }

            if (vacuum_entries)
                pfree(vacuum_entries);

            // Release vacuum lock
            LWLockRelease(&mt->vacuum_lock);

            // Decrement reference count (atomic operation, no lock needed)
            pg_atomic_add_fetch_u32(&SharedMemtableBuffer->slots[gidx].ref_count, -1);

            // Record this memtable as vacuumed
            vacuumed_memtable_ids[vacuumed_count++] = growing_memtable_id;
        }
    }
    
    // step 2. vacuum the immutable memtables
    /*
    * 1. iterate over all immutable memtables
    * 2. for each immutable memtable, acquire the vacuum lock
    * 3. iterate over all slots in the immutable memtable
    * 4. for each slot, use the callback function to check if the corresponding vector is deleted
    * 5. if deleted, mark the slot as deleted in the bitmap
    * 6. if not deleted, continue
    * 7. check if the immutable memtable is persistent on disk
    * 8. if persistent, flush the immutable memtable to disk, and notify the index worker to update the bitmap
    * 9. release the vacuum lock of the immutable memtable
    */
    // elog(DEBUG1, "[bulk_delete_lsm_index] step 2. vacuum the immutable memtables");
    {
        // Process immutable memtables one by one, checking if they're still in the list
        // to avoid processing memtables that were removed while we were working
        for (;;)
        {
            SegmentId memtable_id_to_vacuum = 0;  // 0 indicates no memtable to vacuum
            int mt_idx_to_vacuum = -1;
            
            // Get next immutable memtable to vacuum (if any)
            LWLockAcquire(lsm->mt_lock, LW_SHARED);
            
            // Skip immutable memtables with sid >= growing_memtable_id recorded in step 1
            // (already vacuumed in step 1 or newer than the one we vacuumed)
            
            // Find first immutable memtable that hasn't been vacuumed yet
            // Iterate backwards to process more recent memtables first
            uint32_t memtable_count = lsm->memtable_count;
            for (int i = (int)memtable_count - 1; i >= 0; i--)
            {
                if (i >= (int)MEMTABLE_NUM)
                    elog(ERROR, "[bulk_delete_lsm_index] memtable index out of range");
                    
                int mt_idx = lsm->memtable_idxs[i];
                if (mt_idx < 0 || mt_idx >= MEMTABLE_BUF_SIZE)
                    continue;
                
                ConcurrentMemTable mt = MT_FROM_SLOTIDX(mt_idx);
                SegmentId mt_id = mt->memtable_id;
                
                // Skip if sid >= growing memtable sid (already vacuumed in step 1 or newer)
                if (mt_id >= growing_memtable_id)
                    continue;
                
                // Skip if already vacuumed
                bool already_vacuumed = false;
                for (uint32_t j = 0; j < vacuumed_count; j++)
                {
                    if (vacuumed_memtable_ids[j] == mt_id)
                    {
                        already_vacuumed = true;
                        break;
                    }
                }
                
                if (!already_vacuumed)
                {
                    memtable_id_to_vacuum = mt_id;
                    mt_idx_to_vacuum = mt_idx;
                    // Increment reference count while holding mt_lock
                    pg_atomic_add_fetch_u32(&SharedMemtableBuffer->slots[mt_idx].ref_count, 1);
                    break;
                }
            }
            
            LWLockRelease(lsm->mt_lock);
            
            // If no more memtables to vacuum, break
            if (mt_idx_to_vacuum < 0)
                break;
            
            // Verify memtable still exists and get it
            ConcurrentMemTable mt = MT_FROM_SLOTIDX(mt_idx_to_vacuum);
            if (mt->rel != indexRelId || mt->memtable_id != memtable_id_to_vacuum)
            {
                elog(ERROR, "[bulk_delete_lsm_index] memtable was removed or changed"); // this should not happen as we increment the reference count while holding the mt_lock
                // Memtable was removed or changed, decrement ref count before skipping
                pg_atomic_add_fetch_u32(&SharedMemtableBuffer->slots[mt_idx_to_vacuum].ref_count, -1);
                continue;
            }
            
            // Acquire vacuum lock
            LWLockAcquire(&mt->vacuum_lock, LW_EXCLUSIVE);
            
            // Iterate over all slots (immutable memtables are sealed, so all slots are valid)
            uint32_t current_size = pg_atomic_read_u32(&mt->current_size);
            uint32_t valid_size = (current_size > mt->capacity) ? mt->capacity : current_size;

            bool bitmap_changed = false;

            /* Plan 3 refactor: collect entries for the unified per-sid WAL.
             * Each immutable memtable has exactly one sid (mt->memtable_id);
             * we emit one xl_dpv_vacuum_tombstones record per vacuum batch. */
            xl_dpv_vacuum_entry *vacuum_entries = NULL;
            int vacuum_entry_count = 0;
            if (valid_size > 0 && dpv_replication_role == DPV_ROLE_PRIMARY)
                vacuum_entries = (xl_dpv_vacuum_entry *)
                    palloc(sizeof(xl_dpv_vacuum_entry) * valid_size);

            for (uint32_t j = 0; j < valid_size; j++)
            {
                int64_t tid_int64 = mt->tids[j];
                ItemPointerData tid = Int64ToItemPointer(tid_int64);
                if (callback(&tid, callback_state))
                {
                    // Vector is deleted, mark it in bitmap
                    SET_SLOT(mt->bitmap, j);
                    // remove the tid from the status page
                    RemoveFromStatusMemtable(index, MAIN_FORKNUM, mt->memtable_id, j, tid);
                    bitmap_changed = true;
                    stats->tuples_removed++;
                    if (vacuum_entries)
                    {
                        vacuum_entries[vacuum_entry_count].sid_local_idx = j;
                        vacuum_entries[vacuum_entry_count].tid           = tid_int64;
                        vacuum_entry_count++;
                    }
                }
                else {
                    stats->num_index_tuples++;
                }
            }

            // Check if memtable is persistent on disk
            SegmentId start_sid_disk, end_sid_disk;
            uint32_t valid_rows;
            IndexType seg_index_type;
            bool is_persistent = read_lsm_segment_metadata(indexRelId, mt->memtable_id, mt->memtable_id, 
                                                          UINT32_MAX, &start_sid_disk, &end_sid_disk, 
                                                          &valid_rows, &seg_index_type);
            
            if (bitmap_changed)
            {
                // Calculate delete_count from bitmap
                uint32_t delete_count = 0;
                for (uint32_t i = 0; i < valid_size; i++)
                {
                    if (IS_SLOT_SET(mt->bitmap, i))
                    {
                        delete_count++;
                    }
                }
                if (is_persistent)
                {
                    // Find latest version and subversion
                    uint32_t version = find_latest_segment_version(indexRelId, mt->memtable_id, mt->memtable_id);
                    uint32_t latest_subversion = find_latest_bitmap_subversion(indexRelId, mt->memtable_id, mt->memtable_id, version);
                    uint32_t next_subversion = (latest_subversion == UINT32_MAX) ? 0 : latest_subversion + 1;

                    /*
                     * Spec §11: WAL must be emitted AND flushed BEFORE the
                     * subversion bitmap file hits disk, so a standby that
                     * adopted this memtable as a segment can replay vacuum
                     * tombstones in order with merges/rebuilds.
                     */
                    if (dpv_replication_role == DPV_ROLE_PRIMARY)
                    {
                        XLogRecPtr lsn = dpv_emit_vacuum_tombstones(
                            indexRelId,
                            mt->memtable_id,             /* sid */
                            version,                     /* owner_version */
                            next_subversion,
                            /* is_memtable_owner */ false,
                            vacuum_entries, vacuum_entry_count);
                        XLogFlush(lsn);
                    }

                    // Write updated bitmap with subversion
                    write_bitmap_file_with_subversion(indexRelId, mt->memtable_id, mt->memtable_id,
                                                    version, next_subversion, mt->bitmap,
                                                    GET_BITMAP_SIZE(valid_size), delete_count);

                    // notify the index worker to update the bitmap
                    (void) segment_update_blocking(lsm_idx, indexRelId, SEGMENT_UPDATE_VACUUM,
                                                   start_sid_disk, end_sid_disk, 0,
                                                   NULL, 0, NULL, 0);
                    elog(DEBUG1, "[bulk_delete_lsm_index] notified the index worker to update the bitmap for immutable memtable %u", mt->memtable_id);
                }
                else
                {
                    if (dpv_replication_role == DPV_ROLE_PRIMARY &&
                        vacuum_entry_count > 0)
                    {
                        /* Memtable not yet flushed: subversion N/A (use sentinel).
                         * No XLogFlush needed — memtables have no §11 ordering
                         * invariant to protect (their bitmap-for-memtable file
                         * is advisory; the standby reconstructs from WAL). */
                        (void) dpv_emit_vacuum_tombstones(
                            indexRelId,
                            mt->memtable_id,             /* sid */
                            0,                           /* owner_version */
                            UINT32_MAX,                  /* subversion sentinel */
                            /* is_memtable_owner */ true,
                            vacuum_entries, vacuum_entry_count);
                    }
                    write_bitmap_for_memtable(indexRelId, mt->memtable_id, mt->bitmap, GET_BITMAP_SIZE(valid_size), delete_count);
                    elog(DEBUG1, "[bulk_delete_lsm_index] wrote the bitmap for immutable memtable %u to disk", mt->memtable_id);
                }
            }

            /* Per-iteration free; never leak across memtables. */
            if (vacuum_entries)
                pfree(vacuum_entries);

            // Release vacuum lock
            LWLockRelease(&mt->vacuum_lock);

            // Decrement reference count (atomic operation, no lock needed)
            pg_atomic_add_fetch_u32(&SharedMemtableBuffer->slots[mt_idx_to_vacuum].ref_count, -1);

            // Record this memtable as vacuumed
            if (vacuumed_count < MEMTABLE_NUM + 1)
            {
                vacuumed_memtable_ids[vacuumed_count++] = mt->memtable_id;
            }
        }
    }

    // step 3. vacuum the flushed segments
    /*
    * 1. iterate over all flushed segments
    * 2. for each flushed segment, acquire the vacuum lock
    * 3. load the bitmap and the mapping of the latest version of the flushed segment
    * 4. conduct vacuum on the flushed segment to generate a new bitmap
    * 5. flush the flushed segment to disk
    * 6. notify the index worker to update the bitmap
    * 7. release the vacuum lock of the flushed segment
    */
    /* Step 3: vacuum flushed segments via disk scan */
    {
        SegmentFileInfo files[MAX_SEGMENTS_COUNT];
        int file_count = scan_segment_metadata_files(indexRelId, files, MAX_SEGMENTS_COUNT);

        for (int fi = 0; fi < file_count; fi++)
        {
retry_segment:;
            SegmentId start_sid_disk, end_sid_disk;
            uint32_t valid_rows;
            IndexType seg_index_type;
            uint32_t seg_version = files[fi].version;

            if (!read_lsm_segment_metadata(indexRelId,
                                           files[fi].start_sid, files[fi].end_sid,
                                           seg_version,
                                           &start_sid_disk, &end_sid_disk,
                                           &valid_rows, &seg_index_type))

            {
                // FIXME: This will be a potential bug: If we remove old segment files in the future, we need to handle this case.
                elog(ERROR, "[bulk_delete_lsm_index] segment %u-%u version %u gone between scan and read", files[fi].start_sid, files[fi].end_sid, seg_version);
                continue;
            }

            /* Skip segments that match an already-vacuumed memtable */
            bool already_vacuumed = false;
            if (start_sid_disk == end_sid_disk)
            {
                for (uint32_t vi = 0; vi < vacuumed_count; vi++)
                {
                    if (start_sid_disk == vacuumed_memtable_ids[vi])
                    {
                        already_vacuumed = true;
                        break;
                    }
                }
            }
            if (already_vacuumed)
                continue;

            /* Load bitmap + mapping at this version */
            uint8_t *bitmap_ptr = NULL;
            int64_t *mapping_ptr = NULL;
            uint32_t delete_count;
            load_bitmap_file(indexRelId, start_sid_disk, end_sid_disk,
                             seg_version, &bitmap_ptr, true, &delete_count);
            load_mapping_file(indexRelId, start_sid_disk, end_sid_disk,
                              seg_version, &mapping_ptr, true);

            if (bitmap_ptr == NULL || mapping_ptr == NULL)
            {
                if (bitmap_ptr)  pfree(bitmap_ptr);
                if (mapping_ptr) pfree(mapping_ptr);
                continue;
            }

            /* Walk vectors and apply callback */
            bool bitmap_changed = false;

            /*
             * Spec §11: collect (sid_local_idx, tid) entries while marking, so
             * we can emit an xl_dpv_vacuum_tombstones WAL record and XLogFlush
             * BEFORE writing the bitmap subversion file. mapping_ptr is already
             * int64_t here — no encoding conversion needed.
             */
            xl_dpv_vacuum_entry *vacuum_entries = NULL;
            int vacuum_entry_count = 0;
            if (valid_rows > 0 && dpv_replication_role == DPV_ROLE_PRIMARY &&
                start_sid_disk == end_sid_disk)
                vacuum_entries = (xl_dpv_vacuum_entry *)
                    palloc(sizeof(xl_dpv_vacuum_entry) * valid_rows);

            /*
             * Plan 3 refactor: for multi-sid segments, group vacuumed entries
             * by sid via the offset file. seg_offsets[k] gives (sid,
             * start_offset, end_offset) for sid index k. per_sid_entries[k]
             * and per_sid_count[k] are the buffer + size for sid k.
             */
            SegmentOffsetRange     *seg_offsets      = NULL;
            uint32_t                seg_sid_count    = 0;
            xl_dpv_vacuum_entry **per_sid_entries = NULL;
            int                    *per_sid_count    = NULL;
            uint32_t                cur_off_idx      = 0;  /* advancing pointer */

            if (valid_rows > 0 && dpv_replication_role == DPV_ROLE_PRIMARY &&
                start_sid_disk != end_sid_disk)
            {
                /* Multi-sid path: load offset table and allocate per-sid buffers. */
                load_offset_file(indexRelId, start_sid_disk, end_sid_disk,
                                 seg_version, &seg_offsets, true);
                seg_sid_count = (uint32_t) (end_sid_disk - start_sid_disk + 1);
                per_sid_entries = (xl_dpv_vacuum_entry **)
                    palloc0(sizeof(xl_dpv_vacuum_entry *) * seg_sid_count);
                per_sid_count = (int *) palloc0(sizeof(int) * seg_sid_count);
                for (uint32_t k = 0; k < seg_sid_count; k++)
                {
                    Size span = seg_offsets[k].end_offset - seg_offsets[k].start_offset;
                    if (span > 0)
                        per_sid_entries[k] = (xl_dpv_vacuum_entry *)
                            palloc(sizeof(xl_dpv_vacuum_entry) * span);
                }
            }

            for (uint32_t i = 0; i < valid_rows; i++)
            {
                if (!IS_SLOT_SET(bitmap_ptr, i))
                {
                    ItemPointerData tid = Int64ToItemPointer(mapping_ptr[i]);
                    if (callback(&tid, callback_state))
                    {
                        SET_SLOT(bitmap_ptr, i);
                        bitmap_changed = true;
                        stats->tuples_removed++;
                        delete_count++;
                        if (dpv_replication_role == DPV_ROLE_PRIMARY)
                        {
                            if (start_sid_disk == end_sid_disk)
                            {
                                /* Single-sid: sid_local_idx == i. */
                                vacuum_entries[vacuum_entry_count].sid_local_idx = i;
                                vacuum_entries[vacuum_entry_count].tid           = mapping_ptr[i];
                                vacuum_entry_count++;
                            }
                            else
                            {
                                /* Multi-sid: locate sid via offset table.
                                 * cur_off_idx advances monotonically with i
                                 * (declared at outer scope in Step 1), giving
                                 * O(n) total grouping. */
                                while (cur_off_idx < seg_sid_count &&
                                       i >= seg_offsets[cur_off_idx].end_offset)
                                    cur_off_idx++;
                                if (cur_off_idx >= seg_sid_count)
                                    elog(ERROR,
                                         "[bulk_delete_lsm_index] i=%u beyond segment offsets",
                                         i);

                                int k = (int) cur_off_idx;
                                int slot = per_sid_count[k]++;
                                per_sid_entries[k][slot].sid_local_idx =
                                    (uint32) (i - seg_offsets[k].start_offset);
                                per_sid_entries[k][slot].tid = mapping_ptr[i];
                            }
                        }
                    }
                    else
                    {
                        stats->num_index_tuples++;
                    }
                }
            }

            pfree(mapping_ptr);

            if (bitmap_changed)
            {
                uint32_t latest_sub = find_latest_bitmap_subversion(indexRelId,
                                          start_sid_disk, end_sid_disk, seg_version);
                uint32_t next_sub = (latest_sub == UINT32_MAX) ? 0 : latest_sub + 1;
                Size bitmap_size = GET_BITMAP_SIZE(valid_rows);

                /*
                 * Spec §11: WAL must be emitted AND flushed BEFORE the
                 * subversion bitmap file hits disk, so the standby sees
                 * tombstones ordered correctly with any subsequent
                 * merge/rebuild on this segment.
                 */
                if (dpv_replication_role == DPV_ROLE_PRIMARY)
                {
                    if (start_sid_disk == end_sid_disk)
                    {
                        /* Single-sid fast path. */
                        if (vacuum_entry_count > 0)
                        {
                            XLogRecPtr lsn = dpv_emit_vacuum_tombstones(
                                indexRelId,
                                start_sid_disk,
                                seg_version,
                                next_sub,
                                /* is_memtable_owner */ false,
                                vacuum_entries, vacuum_entry_count);
                            XLogFlush(lsn);
                        }
                    }
                    else
                    {
                        /* Multi-sid: one record per touched sid. Collect each
                         * XLogInsert's LSN; flush the max once. */
                        XLogRecPtr max_lsn = InvalidXLogRecPtr;
                        for (uint32_t k = 0; k < seg_sid_count; k++)
                        {
                            if (per_sid_count[k] == 0)
                                continue;
                            XLogRecPtr lsn = dpv_emit_vacuum_tombstones(
                                indexRelId,
                                seg_offsets[k].sid,
                                seg_version,
                                next_sub,
                                /* is_memtable_owner */ false,
                                per_sid_entries[k],
                                per_sid_count[k]);
                            if (lsn > max_lsn) max_lsn = lsn;
                        }
                        if (max_lsn != InvalidXLogRecPtr)
                            XLogFlush(max_lsn);
                    }
                }

                write_bitmap_file_with_subversion(indexRelId, start_sid_disk, end_sid_disk,
                                                  seg_version, next_sub,
                                                  bitmap_ptr, bitmap_size, delete_count);

                int result = segment_update_blocking(lsm_idx, indexRelId,
                                 SEGMENT_UPDATE_VACUUM,
                                 start_sid_disk, end_sid_disk,
                                 seg_version,  /* expected_version */
                                 NULL, 0,
                                 NULL, 0);
                if (result == 1)  /* RETRY */
                {
                    pfree(bitmap_ptr);
                    if (vacuum_entries)
                        pfree(vacuum_entries);
                    if (per_sid_entries)
                    {
                        for (uint32_t k = 0; k < seg_sid_count; k++)
                            if (per_sid_entries[k]) pfree(per_sid_entries[k]);
                        pfree(per_sid_entries);
                    }
                    if (per_sid_count)  pfree(per_sid_count);
                    if (seg_offsets)    pfree(seg_offsets);
                    /*
                     * The segment was rebuilt or merged away while we were
                     * computing deletions. Re-scan disk to find the current
                     * segment covering start_sid_disk and retry.
                     */
                    SegmentFileInfo retry_files[MAX_SEGMENTS_COUNT];
                    int retry_count = scan_segment_metadata_files(indexRelId,
                                          retry_files, MAX_SEGMENTS_COUNT);
                    for (int rfi = 0; rfi < retry_count; rfi++)
                    {
                        if (retry_files[rfi].start_sid <= start_sid_disk &&
                            start_sid_disk <= retry_files[rfi].end_sid)
                        {
                            files[fi] = retry_files[rfi];
                            goto retry_segment;
                        }
                    }
                    /* Segment completely gone — skip */
                    continue;
                }
            }

            if (vacuum_entries)
                pfree(vacuum_entries);
            if (per_sid_entries)
            {
                for (uint32_t k = 0; k < seg_sid_count; k++)
                    if (per_sid_entries[k]) pfree(per_sid_entries[k]);
                pfree(per_sid_entries);
            }
            if (per_sid_count)  pfree(per_sid_count);
            if (seg_offsets)    pfree(seg_offsets);
            pfree(bitmap_ptr);
        }
    }

    elog(DEBUG1, "[bulk_delete_lsm_index] completed bulk_delete_lsm_index for index %u: tuples_removed=%.0f, num_index_tuples=%.0f",
         indexRelId, stats->tuples_removed, stats->num_index_tuples);
    
    return stats;
}

// Forward declaration for TestConsistencyContext
typedef struct TestConsistencyContext
{
    BlockNumber block_id;
    OffsetNumber *offset_numbers;
    int total_count;
    uint8_t *bitmap; // 1 stands for visible, 0 stands for invisible
    int valid_count;
    uint8_t *visit_map;
    int visit_count;
    int visit_valid_count;
} TestConsistencyContext;

// Helper function to check if a TID exists in the heap and mark it as visited
static bool
check_and_mark_tid(ItemPointerData tid, TestConsistencyContext *ctx)
{
    BlockNumber blkno = BlockIdGetBlockNumber(&tid.ip_blkid);
    OffsetNumber offno = tid.ip_posid;

    // Find the offset in the context using binary search (offset_numbers is in ascending order)
    int idx = -1;
    int left = 0;
    int right = ctx->total_count - 1;
    
    while (left <= right)
    {
        int mid = left + (right - left) / 2;
        OffsetNumber mid_offno = ctx->offset_numbers[mid];
        
        if (mid_offno == offno)
        {
            idx = mid;
            break;
        }
        else if (mid_offno < offno)
        {
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }
    
    if (idx < 0)
    {
        elog(DEBUG1, "[test_consistency] TID (%u,%u) offset %u not found in block %u", 
             blkno, offno, offno, blkno);
        return false;
    }
    
    // Mark as visited
    if (!IS_SLOT_SET(ctx->visit_map, idx))
    {
        SET_SLOT(ctx->visit_map, idx);
        ctx->visit_count++;
        
        // If this TID is visible in the heap, increment visit_valid_count
        if (IS_SLOT_SET(ctx->bitmap, idx))
        {
            ctx->visit_valid_count++;
        }
    }
    
    return true;
}

static bool
test_consistency(Relation heap_rel, LSMIndex lsm_index, int lsm_slot_idx)
{
    elog(DEBUG1, "[test_consistency] testing consistency of the LSM index %u", lsm_index->indexRelId);

    // get the curren snapshot 
    Snapshot snapshot = GetActiveSnapshot();
    // initialize the heap scan
    BufferAccessStrategy bas = GetAccessStrategy(BAS_BULKREAD);
    BlockNumber nblocks = RelationGetNumberOfBlocks(heap_rel);
    TestConsistencyContext *context = (TestConsistencyContext *) palloc(sizeof(TestConsistencyContext) * nblocks);
    for (int i = 0; i < nblocks; i++)
    {
        context[i].block_id = InvalidBlockNumber;
        context[i].offset_numbers = NULL;
        context[i].total_count = 0;
        context[i].bitmap = NULL;
        context[i].valid_count = 0;
        context[i].visit_map = NULL;
        context[i].visit_count = 0;
        context[i].visit_valid_count = 0;
    }

    for (BlockNumber blkno = 0; blkno < nblocks; blkno++)
    {
        Buffer buf = ReadBufferExtended(heap_rel, MAIN_FORKNUM, blkno,
            RBM_NORMAL, bas);

        LockBuffer(buf, BUFFER_LOCK_SHARE);

        Page page = BufferGetPage(buf);

        if (!PageIsNew(page) && !PageIsEmpty(page))
        {
            OffsetNumber maxoff = PageGetMaxOffsetNumber(page);
            context[blkno].block_id = blkno;
            context[blkno].offset_numbers = (OffsetNumber *) palloc(sizeof(OffsetNumber) * maxoff);
            context[blkno].bitmap = (uint8_t *) palloc(GET_BITMAP_SIZE(maxoff));
            context[blkno].visit_map = (uint8_t *) palloc(GET_BITMAP_SIZE(maxoff));
            // set all the slots in the bitmap and visit_map to 0
            memset(context[blkno].bitmap, 0, GET_BITMAP_SIZE(maxoff));
            memset(context[blkno].visit_map, 0, GET_BITMAP_SIZE(maxoff));

            for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off++)
            {
                ItemId itemId = PageGetItemId(page, off);
                int idx = context[blkno].total_count;
                context[blkno].offset_numbers[context[blkno].total_count++] = off;

                if (ItemIdIsNormal(itemId))
                {
                    HeapTupleData tuple;

                    tuple.t_len = ItemIdGetLength(itemId);
                    ItemPointerSet(&tuple.t_self, blkno, off);
                    tuple.t_tableOid = RelationGetRelid(heap_rel);
                    tuple.t_data = (HeapTupleHeader) PageGetItem(page, itemId);

                    bool visible = HeapTupleSatisfiesVisibility(&tuple, snapshot, buf);
                    if (visible)
                    {
                        SET_SLOT(context[blkno].bitmap, idx);
                        context[blkno].valid_count++;
                    }
                    else
                        CLEAR_SLOT(context[blkno].bitmap, idx);
                }
                else {
                    // if (ItemIdIsRedirected(itemId))
                    //     elog(DEBUG1, "[test_consistency] redirected item id");

                    // if (ItemIdIsDead(itemId))
                    //     elog(DEBUG1, "[test_consistency] dead item id");

                    // if (!ItemIdIsUsed(itemId))
                    //     elog(DEBUG1, "[test_consistency] unused item id, table pruning detected");
                    CLEAR_SLOT(context[blkno].bitmap, idx);
                }
            }
        }
        UnlockReleaseBuffer(buf);
    }

    // Visibility check 1: all existing tid in the lsm index should be in the heap
    // iterate over all lsm index segments, load the bitmap and mapping file, check if the tids are in the heap 
    // iterate over all sealed memtables and the growing memtable, check if the tids are in the heap 
    
    bool consistency_ok = true;
    Oid index_relid = lsm_index->indexRelId;
    
    // Step 1: Iterate over all flushed segments via disk scan
    {
        SegmentFileInfo seg_files[MAX_SEGMENTS_COUNT];
        int seg_file_count = scan_segment_metadata_files(index_relid, seg_files, MAX_SEGMENTS_COUNT);

        for (int sfi = 0; sfi < seg_file_count; sfi++)
        {
            SegmentId start_sid, end_sid;
            uint32_t vec_count;
            IndexType seg_index_type;
            uint32_t version = seg_files[sfi].version;
            uint8_t *bitmap_ptr = NULL;
            int64_t *mapping_ptr = NULL;
            uint32_t delete_count;

            if (!read_lsm_segment_metadata(index_relid,
                                           seg_files[sfi].start_sid, seg_files[sfi].end_sid,
                                           version,
                                           &start_sid, &end_sid,
                                           &vec_count, &seg_index_type))
                continue;
            load_bitmap_file(index_relid, start_sid, end_sid, version, &bitmap_ptr, true, &delete_count);
            load_mapping_file(index_relid, start_sid, end_sid, version, &mapping_ptr, true);

            if (bitmap_ptr != NULL && mapping_ptr != NULL)
            {
                for (uint32_t i = 0; i < vec_count; i++)
                {
                    if (!IS_SLOT_SET(bitmap_ptr, i))
                    {
                        int64_t tid_int64 = mapping_ptr[i];
                        ItemPointerData tid = Int64ToItemPointer(tid_int64);
                        for (int j = 0; j < nblocks; j++)
                        {
                            if (context[j].block_id == BlockIdGetBlockNumber(&tid.ip_blkid))
                            {
                                if (!check_and_mark_tid(tid, &context[j]))
                                    consistency_ok = false;
                                break;
                            }
                        }
                    }
                }
            }
            else
            {
                elog(ERROR, "[test_consistency] failed to load bitmap and mapping files for segment %u-%u", start_sid, end_sid);
                if (bitmap_ptr) pfree(bitmap_ptr);
                if (mapping_ptr) pfree(mapping_ptr);
                return false;
            }

            if (bitmap_ptr) pfree(bitmap_ptr);
            if (mapping_ptr) pfree(mapping_ptr);
        }
    }
    
    // Step 2: Iterate over sealed memtables
    LWLockAcquire(lsm_index->mt_lock, LW_SHARED);
    uint32_t memtable_count = lsm_index->memtable_count;
    int32_t *memtable_idxs = lsm_index->memtable_idxs;
    int32_t growing_memtable_idx = lsm_index->growing_memtable_idx;
    LWLockRelease(lsm_index->mt_lock);

    // Process sealed memtables
    for (uint32_t i = 0; i < memtable_count; i++)
    {
        int mt_idx = memtable_idxs[i];
        if (mt_idx < 0 || mt_idx >= MEMTABLE_BUF_SIZE)
            continue;
        
        ConcurrentMemTable mt = MT_FROM_SLOTIDX(mt_idx);
        
        // Verify this memtable belongs to this index
        if (mt->rel != index_relid)
            continue;
        
        uint32_t current_size = pg_atomic_read_u32(&mt->current_size);
        uint32_t valid_size = (current_size > mt->capacity) ? mt->capacity : current_size;
        
        // Iterate over all slots in the sealed memtable
        for (uint32_t j = 0; j < valid_size; j++)
        {
            // Check if slot is not deleted (bitmap bit is 0 means not deleted)
            if (!IS_SLOT_SET(mt->bitmap, j))
            {
                int64_t tid_int64 = mt->tids[j];
                ItemPointerData tid = Int64ToItemPointer(tid_int64);
                
                // Check if TID exists in heap and mark as visited
                for (int j = 0; j < nblocks; j++)
                {
                    if (context[j].block_id == BlockIdGetBlockNumber(&tid.ip_blkid))
                    {
                        if (!check_and_mark_tid(tid, &context[j]))
                            consistency_ok = false;
                        break;
                    }
                }
            }
        }
    }

    // Step 3: Iterate over growing memtable
    if (growing_memtable_idx >= 0 && growing_memtable_idx != MT_IDX_INVALID && growing_memtable_idx != MT_IDX_ROTATING)
    {
        ConcurrentMemTable mt = MT_FROM_SLOTIDX(growing_memtable_idx);
        
        // Verify this memtable belongs to this index
        if (mt->rel == index_relid)
        {
            uint32_t current_size = pg_atomic_read_u32(&mt->current_size);
            uint32_t valid_size = (current_size > mt->capacity) ? mt->capacity : current_size;
            
            // Iterate over ready slots in the growing memtable
            for (uint32_t j = 0; j < valid_size; j++)
            {
                // For growing memtable, check ready flag
                if (!IS_SLOT_SET(mt->bitmap, j))
                {
                    int64_t tid_int64 = mt->tids[j];
                    ItemPointerData tid = Int64ToItemPointer(tid_int64);
                    
                    // Check if TID exists in heap and mark as visited
                    for (int j = 0; j < nblocks; j++)
                    {
                        if (context[j].block_id == BlockIdGetBlockNumber(&tid.ip_blkid))
                        {
                            if (!check_and_mark_tid(tid, &context[j]))
                                consistency_ok = false;
                            break;
                        }
                    }
                }
            }
        }
    }

    // Visibility check 2: all visible tid in the heap should be existing in the lsm index     
    for (BlockNumber blkno = 0; blkno < nblocks; blkno++)
    {
        TestConsistencyContext *ctx = &context[blkno];
        
        if (ctx->block_id == InvalidBlockNumber || ctx->offset_numbers == NULL)
            continue;
        
        // Check if all visible TIDs in this block are visited
        if (ctx->valid_count > 0)
        {
            // valid_count is the number of visible TIDs in the heap
            // visit_valid_count is the number of visible TIDs that were visited (exist in LSM index)
            if (ctx->valid_count != ctx->visit_valid_count)
            {
                consistency_ok = false;
                
                // Output missing TIDs
                for (int i = 0; i < ctx->total_count; i++)
                {
                    // Check if this TID is visible but not visited
                    if (IS_SLOT_SET(ctx->bitmap, i) && !IS_SLOT_SET(ctx->visit_map, i))
                    {
                        OffsetNumber offno = ctx->offset_numbers[i];
                        ItemPointerData tid;
                        BlockIdSet(&tid.ip_blkid, blkno);
                        tid.ip_posid = offno;
                        
                        elog(WARNING, "[test_consistency] Missing TID in LSM index: (%u,%u)", 
                             blkno, offno);
                    }
                }
            }
        }
    }
        
    if (!consistency_ok)
    {
        elog(ERROR, "[test_consistency] Consistency check failed: some visible TIDs in heap are missing in LSM index");
    }
    else {
        elog(DEBUG1, "[test_consistency] Consistency check passed");
    }
    
    return consistency_ok;
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