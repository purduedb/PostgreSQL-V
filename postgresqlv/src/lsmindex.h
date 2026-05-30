#ifndef LSMINDEX_H
#define LSMINDEX_H

#include "postgres.h"
#include "access/genam.h"
#include "storage/lwlock.h"
#include "storage/condition_variable.h"

#include "utils.h"
#include <sys/stat.h>
#include <limits.h>

typedef uint32_t SegmentId;
typedef enum IndexType
{
    FLAT, // no indexing
    IVFFLAT, 
    HNSW,
    DISKANN
}IndexType;

// Global macro to indicate whether DiskANN is used
// Set this to 1 to enable DiskANN, 0 to use HNSW
#ifndef IS_DISK_BASED
#define IS_DISK_BASED 0
#endif

// Set this to 1 to enable 2-phase mmap cold-start on IndexLoadTaskType:
//   phase 1 = mmap-load all segments quickly → signal backend immediately;
//   phase 2 = background upgrade each segment to full in-memory.
// Set to 0 to skip mmap and fully load all segments before signaling.
// The on-disk format is mmap-compatible regardless of this flag.
#ifndef ENABLE_MMAP_COLDSTART
#define ENABLE_MMAP_COLDSTART 1
#endif

// FIXME: find a better way to manage memtables' memory
// memtable
#define MEMTABLE_MAX_CAPACITY 50000 // 50K
// bitmap
#define MEMTABLE_BITMAP_SIZE ((MEMTABLE_MAX_CAPACITY + 7) / 8) // in bytes
#define IS_SLOT_SET(bitmap, i)   (bitmap[(i) >> 3] & (1 << ((i) & 7)))
#define SET_SLOT(bitmap, i)      (bitmap[(i) >> 3] |= (1 << ((i) & 7)))
#define CLEAR_SLOT(bitmap, i)    (bitmap[(i) >> 3] &= ~(1 << ((i) & 7)))
#define GET_BITMAP_SIZE(size) (((size) + 7) / 8)

#define MEMTABLE_VECTOR_ARRAY_SIZE_BYTES (128 * 1024 * 1024) // 128MB
#define MEMTABLE_SIZE_BYTES ( MAXALIGN(sizeof(ConcurrentMemTableData)) + MEMTABLE_VECTOR_ARRAY_SIZE_BYTES )
#define MEMTABLE_BUF_SIZE 8 // number of memtable slots
#define MEMTABLE_NUM 4 // number of memtables per relation
#define MAX_SEGMENTS_COUNT 1024 // max number of segments per index
#define MAX_SEGMENTS_SLOT_NUM 1024 // max number of segments in buffer
#define INDEX_BUF_SIZE 8
#define START_SEGMENT_ID 3

// locks
#define LSM_MEMTABLE_LWTRANCHE "lsm_memtable_rwlock"
#define LSM_MEMTABLE_LWTRANCHE_ID 1001
#define LSM_SEGMENT_LWTRANCHE "lsm_segment_rwlock"
#define LSM_SEGMENT_LWTRANCHE_ID 1002
#define LSM_MEMTABLE_VACUUM_LWTRANCHE "lsm_memtable_vacuum_lock"
#define LSM_MEMTABLE_VACUUM_LWTRANCHE_ID 1005
#define LSM_FLUSHED_RELEASE_LWTRANCHE "lsm_flushed_release_lock"
#define LSM_FLUSHED_RELEASE_LWTRANCHE_ID 1006
#define LSM_INDEX_BUFFER_LWTRANCHE "lsm_index_buffer_lock"
#define LSM_INDEX_BUFFER_LWTRANCHE_ID 1007

// Sentinel for rotation-in-progress
#define MT_IDX_INVALID   (-1)
#define MT_IDX_ROTATING  (-2)

typedef struct ConcurrentMemTableData {
    // VectorId start_vid;
    Oid rel;
    SegmentId memtable_id;                                        
    uint32 capacity;
    pg_atomic_uint32 current_size; 
    pg_atomic_uint32 ready_cnt; 
    pg_atomic_uint32 sealed;
    pg_atomic_uint32 flush_claimed; /* 0 = not claimed, 1 = claimed */
    pg_atomic_uint32 max_ready_id; /* maximum ID where all slots before it and itself are ready */

    // The metadata of the vector 
    uint32_t dim;
    uint32_t elem_size; // the size of each dimension

    int64_t tids[MEMTABLE_MAX_CAPACITY];
    uint8_t bitmap[MEMTABLE_BITMAP_SIZE];   // 1 = "masked out / exclude this id", and 0 = keep
    uint8_t ready[MEMTABLE_MAX_CAPACITY];   // 0 = not ready, 1 = ready
    
    // LWLock for vacuum operations - prevents flushing while vacuum is active
    LWLock vacuum_lock;

    // followed by an in-memory array of vectors
#if defined(pg_attribute_aligned)
    char vector_blob[MEMTABLE_VECTOR_ARRAY_SIZE_BYTES] pg_attribute_aligned(MAXIMUM_ALIGNOF);
#else
    /* Preserve alignment even without attribute support */
    union {
        char vector_blob[MEMTABLE_VECTOR_ARRAY_SIZE_BYTES];
        long _force_align;
    } u;
#   define vector_blob u.vector_blob
#endif
} ConcurrentMemTableData;
typedef ConcurrentMemTableData * ConcurrentMemTable;

typedef struct MemtableBufferSlot
{
    /*
        * Memtable reference count:
        * the buffer slot is free when ref_count is 0
        * when the table is registered, set to 1
        * when a backend perform search on it, increase by 1
        * when a backend finishes search, decrease by 1
        * when the background process flushes it, decrease by 1
        * when it reaches 0, it's free again
    */
    pg_atomic_uint32 ref_count;   /* atomic flag */
#if defined(pg_attribute_aligned)
    ConcurrentMemTableData mt pg_attribute_aligned(MAXIMUM_ALIGNOF);
#else
    ConcurrentMemTableData mt;
#endif
}   MemtableBufferSlot;

typedef struct MemtableBuffer
{
    MemtableBufferSlot slots[MEMTABLE_BUF_SIZE];
}   MemtableBuffer;

extern MemtableBuffer *SharedMemtableBuffer;

typedef struct LSMIndexData
{
    Oid indexRelId;
    // TODO: should be per segment
    IndexType index_type; // IVFFLAT or HNSW
    uint32_t dim;
    uint32_t elem_size; // the size of each dimension
    pg_atomic_uint32 next_segment_id;

    // // flushed segments
    // LWLock *seg_lock;
    // FlushedSegmentData flushed_segments[MAX_SEGMENTS_COUNT];
    // uint32_t flushed_segment_count;
    // uint32_t tail_idx;
    // uint32_t insert_idx;

    // memtable
    LWLock *mt_lock;
    int32 growing_memtable_idx;
    SegmentId growing_memtable_id;
    int32_t memtable_idxs[MEMTABLE_NUM]; // the idx of the memtable in the global memtable array
    uint32_t memtable_count; // number of immutable memtables for this index
    
    // Track memtables that have been flushed but not yet released in status pages
    // Use atomic operations for lock-free reads in insert path
    LWLock *flushed_release_lock; // lock for flushed_not_released tracking (used in both flush and release paths)
    SegmentId flushed_not_released[MEMTABLE_NUM]; // SegmentIds of flushed but not released memtables
    pg_atomic_uint32 flushed_not_released_count; // number of flushed but not released memtables (atomic for lock-free reads)
    pg_atomic_uint32 releasing_in_progress; // 1 if a backend is currently releasing memtables, 0 otherwise
} LSMIndexData;
typedef LSMIndexData * LSMIndex;

/*
 * LSMSlotState — lifecycle of an LSMIndexBufferSlot.
 *
 *   FREE          slot is not in use.
 *   RECOVERING    IndexRecoveryWorker is running recover_lsm_index_internal.
 *   WRITABLE      recovery done. Memtables are usable. FlushedSegmentPool
 *                 is NOT initialized — no segment search is possible yet.
 *   LOADING_INDEX a reader is calling index_load_blocking to populate the pool.
 *   QUERYABLE     pool is initialized; search is allowed.
 *
 * Numeric values are chosen so that the natural progression matches the
 * lifecycle order, but callers SHOULD use the predicates below rather than
 * relying on numeric ordering.
 */
typedef enum LSMSlotState {
    LSM_SLOT_FREE          = 0,
    LSM_SLOT_RECOVERING    = 1,
    LSM_SLOT_WRITABLE      = 2,
    LSM_SLOT_LOADING_INDEX = 3,
    LSM_SLOT_QUERYABLE     = 4
} LSMSlotState;

static inline bool
is_writable(uint32 v)
{
    return v == (uint32) LSM_SLOT_WRITABLE
        || v == (uint32) LSM_SLOT_LOADING_INDEX
        || v == (uint32) LSM_SLOT_QUERYABLE;
}

static inline bool
is_queryable(uint32 v)
{
    return v == (uint32) LSM_SLOT_QUERYABLE;
}

typedef struct LSMIndexBufferSlot
{
    pg_atomic_uint32 valid;      /* LSMSlotState; see enum above */
    ConditionVariable state_cv;   /* backends sleep here; worker broadcasts on done/fail */
    pg_atomic_uint32 state_error; /* 0=ok, 1=failed; set before broadcast */
    Oid request_db_oid;          /* database OID written by the claiming backend */
    Oid request_db_userid;       /* role OID written by the claiming backend */
    LSMIndexData lsmIndex;
}   LSMIndexBufferSlot;

typedef struct LSMIndexBuffer
{
    LWLock *lock;  /* Lock to protect index registration and loading */
    LSMIndexBufferSlot slots[INDEX_BUF_SIZE];
}   LSMIndexBuffer;

extern LSMIndexBuffer *SharedLSMIndexBuffer;

/* Shared coordinator so backends can signal the IndexRecoveryWorker */
typedef struct IndexRecoveryCoordinator
{
    int32 worker_pgprocno; /* pgprocno of the recovery worker; -1 = not running */
} IndexRecoveryCoordinator;

extern IndexRecoveryCoordinator *SharedIndexRecoveryCoordinator;

void lsm_index_buffer_shmem_initialize();
void recover_lsm_index_internal(Oid index_relid, uint32_t slot_idx);
void build_lsm_index(IndexType type, Relation index, void *vector_index, int64_t *tids, uint32_t dim, uint32_t elem_size, uint64_t count);
void insert_lsm_index(Relation index, const void *vector, const int64_t tid);
LSMIndex get_lsm_index(Relation index);
/*
 * get_lsm_index_for_read — like get_lsm_index, but additionally drives the
 * WRITABLE -> QUERYABLE transition before returning. Use this from read
 * paths (search). Writers (insert) should keep calling get_lsm_index.
 */
LSMIndex get_lsm_index_for_read(Relation index);
int lookup_lsm_index_idx(Oid index_relid);
/*
 * get_lsm_index_idx — resolve indexRelId to a slot index in
 * SharedLSMIndexBuffer. Lock-free fast scan first; falls through to a
 * slot-claim that requires the exclusive buffer lock only on cold miss.
 *
 * for_redo selects which registration path to use on cold miss:
 *   - false: backend path, register_lsm_index() (uses MyDatabaseId / GetUserId()).
 *            db_oid must be InvalidOid (unused).
 *   - true:  WAL redo path, register_lsm_index_for_redo() (uses db_oid).
 *            db_oid must be a valid OID from the WAL record.
 */
int get_lsm_index_idx(Oid index_relid, bool for_redo, Oid db_oid);
TopKTuples search_lsm_index(Relation index, const void *vector, int k, int nprobe_efs);
IndexBulkDeleteResult *bulk_delete_lsm_index(Relation index, IndexBulkDeleteResult *stats, IndexBulkDeleteCallback callback, void *callback_state);

// storage
/*
 * Base directory for pgvector LSM segment storage.
 *
 * Set via the GUC `pgvector.storage_base_dir`. When the GUC is empty (the
 * default), the resolved path is `<DataDir>/pgvector_storage/`. The
 * returned string always ends in a trailing '/'.
 *
 * Use get_vector_storage_dir() rather than reading the GUC directly so the
 * default-resolution logic stays in one place.
 */
extern char *vector_storage_base_dir;        /* raw GUC value (may be empty) */
extern const char *get_vector_storage_dir(void);

// Offset file structure
typedef struct SegmentOffsetRange
{
    SegmentId sid;
    Size start_offset;
    Size end_offset;
} SegmentOffsetRange;

// flushed segment
typedef struct PrepareFlushMetaData
{
    SegmentId start_sid;
    SegmentId end_sid;
    uint32_t valid_rows;
    // dsm_handle map_hdl;
    void *map_ptr;
    Size map_size;
    // dsm_handle bitmap_hdl;
    void *bitmap_ptr;
    Size bitmap_size;
    uint32_t delete_count;  // Number of deleted vectors in this segment
    void *index_bin;
    IndexType index_type;
    SegmentOffsetRange *offsets;  // Offset ranges for each SegmentId in the segment
} PrepareFlushMetaData;
typedef PrepareFlushMetaData* PrepareFlushMeta;

// IO
typedef struct SegmentFileInfo
{
    SegmentId start_sid;
    SegmentId end_sid;
    uint32_t version;
} SegmentFileInfo;
int scan_segment_metadata_files(Oid indexRelId, SegmentFileInfo *files, int max_files);
void flush_segment_to_disk(Oid indexRelId, PrepareFlushMeta prep);
void write_lsm_index_metadata(LSMIndex lsm);
bool read_lsm_index_metadata(Oid indexRelId, IndexType *index_type, uint32_t *dim, uint32_t *elem_size);
bool read_lsm_segment_metadata(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version,
					  SegmentId *out_start_sid, SegmentId *out_end_sid, uint32_t *valid_rows, IndexType *index_type);
uint32_t find_latest_segment_version(Oid indexRelId, SegmentId start_sid, SegmentId end_sid);
uint32_t find_latest_bitmap_subversion(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version);
void load_index_file(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version, IndexType index_type, void **index, bool use_mmap);
bool read_bitmap_delete_count(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version, uint32_t *delete_count_out);
void load_bitmap_file(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version, uint8_t **bitmap, bool pg_alloc, uint32_t *delete_count_out);
void write_bitmap_file_with_subversion(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version, uint32_t subversion, const uint8_t *bitmap, Size bitmap_size, uint32_t delete_count);
void write_bitmap_for_memtable(Oid indexRelId, SegmentId memtable_id, uint8_t *bitmap, Size bitmap_size, uint32_t delete_count);
void load_mapping_file(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version, int64_t **mapping, bool pg_alloc);

// Offset file functions
void write_offset_file(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version, const SegmentOffsetRange *offsets);
void load_offset_file(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version, SegmentOffsetRange **offsets, bool pg_alloc);

#define LOAD_LATEST_VERSION UINT32_MAX

// helper functions (memtable)
ConcurrentMemTable MT_FROM_SLOTIDX(int slot_num);
void *VEC_BASE_FROM_MT(ConcurrentMemTable mt);

/*
 * Inline helpers shared between lsmindex.c and standby_memtable.c.
 * These must live in the header so that both translation units can use them
 * without a separate .o dependency.
 */

static inline Size
VEC_BYTES_PER_ROW(const ConcurrentMemTable mt)
{
    return (Size)mt->dim * (Size)mt->elem_size;
}

static inline void *
VEC_PTR_AT(ConcurrentMemTable mt, uint32 row_idx)
{
    char  *base  = (char *) VEC_BASE_FROM_MT(mt);
    Size stride  = MAXALIGN(VEC_BYTES_PER_ROW(mt));
    return (void *)(base + (Size)row_idx * stride);
}

/* Forward declaration needed by publish_slot_release. */
static inline void update_max_ready_id(ConcurrentMemTable t, uint32 slot_id);

static inline void
publish_slot_release(ConcurrentMemTable t, uint32 i)
{
    /* Ensure data writes (tid/vector) are globally visible before ready=1. */
    pg_write_barrier();
    t->ready[i] = 1;
    pg_atomic_add_fetch_u32(&t->ready_cnt, 1);
    update_max_ready_id(t, i);
}

static inline void
update_max_ready_id(ConcurrentMemTable t, uint32 slot_id)
{
    uint32 current_max = pg_atomic_read_u32(&t->max_ready_id);

    /* Handle initial case where no slots are ready yet */
    if (current_max == UINT32_MAX)
    {
        if (slot_id == 0)
            pg_atomic_compare_exchange_u32(&t->max_ready_id, &current_max, 0);
        return;
    }

    /* Only update if this slot extends the contiguous ready range */
    if (slot_id == current_max + 1)
    {
        uint32 new_max = current_max + 1;
        while (new_max < t->capacity && t->ready[new_max] == 1)
            new_max++;
        new_max--;  /* convert to inclusive */

        while (new_max > current_max)
        {
            if (pg_atomic_compare_exchange_u32(&t->max_ready_id, &current_max, new_max))
                break;
            if (current_max >= new_max)
                break;
        }
    }
}

/* Exposed for standby_memtable.c (replication redo). */
int register_and_set_memtable(LSMIndex lsm, Relation index, bool is_recovery,
                               SegmentId assigned_sid);

/*
 * register_lsm_index_for_redo — like register_lsm_index but for use from
 * standby WAL redo callbacks, which have no MyDatabaseId / GetUserId().
 * The caller (a redo callback in the startup process) reads dbOid from
 * the WAL record.
 *
 * Returns the slot index, or throws via elog(ERROR) on failure.
 *
 * TODO(multi-DB): IndexRecoveryWorker is currently a single bgworker that
 * binds sticky to the first dbOid it serves. Multi-DB requires a per-DB
 * worker pool.
 */
extern int register_lsm_index_for_redo(Oid db_oid, Oid index_relid);

/*
 * claim_lsm_index_slot_for_create_redo — direct slot claim used by
 * redo_index_create on the standby. Bypasses the IndexRecoveryWorker:
 * for a brand-new index, there is nothing to recover (status pages are
 * empty, no segments on disk, no memtables in heap). Avoids the deadlock
 * where the recovery worker's index_open(AccessShareLock) would block
 * waiting for the AccessExclusiveLock held by primary's CREATE INDEX,
 * while the startup process is itself blocked waiting for the recovery
 * worker. The slot transitions directly FREE -> WRITABLE. Idempotent.
 *
 * Returns the slot index, or -1 if no free slot was available.
 */
extern int claim_lsm_index_slot_for_create_redo(Oid db_oid, Oid index_relid,
                                                uint32 index_type,
                                                uint32 dim,
                                                uint32 elem_size);

#endif