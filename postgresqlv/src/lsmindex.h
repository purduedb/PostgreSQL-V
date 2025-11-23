#ifndef LSMINDEX_H
#define LSMINDEX_H

#include "postgres.h"
#include "access/genam.h"
#include "storage/lwlock.h"

#include "utils.h"
#include <sys/stat.h>
#include <limits.h>

typedef uint32_t SegmentId;
typedef enum IndexType
{
    FLAT, // no indexing
    IVFFLAT, 
    HNSW
}IndexType;

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

typedef struct LSMIndexBufferSlot
{
    pg_atomic_uint32 valid;   /* atomic flag */
    LSMIndexData lsmIndex;
}   LSMIndexBufferSlot;

typedef struct LSMIndexBuffer
{
    LSMIndexBufferSlot slots[INDEX_BUF_SIZE];
}   LSMIndexBuffer;

extern LSMIndexBuffer *SharedLSMIndexBuffer;

void lsm_index_buffer_shmem_initialize();
void build_lsm_index(IndexType type, Oid relId, void *vector_index, int64_t *tids, uint32_t dim, uint32_t elem_size, uint64_t count);
void insert_lsm_index(Relation index, const void *vector, const int64_t tid);
LSMIndex get_lsm_index(Oid index_relid);
int get_lsm_index_idx(Oid index_relid);
TopKTuples search_lsm_index(Relation index, const void *vector, int k, int nprobe_efs);
IndexBulkDeleteResult *bulk_delete_lsm_index(Relation index, IndexBulkDeleteResult *stats, IndexBulkDeleteCallback callback, void *callback_state);

// storage
#define VECTOR_STORAGE_BASE_DIR "/ssd_root/liu4127/pg_vector_extension_indexes/"

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
void load_index_file(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version, IndexType index_type, void **index);
bool read_bitmap_delete_count(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version, uint32_t *delete_count_out);
void load_bitmap_file(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version, uint8_t **bitmap, bool pg_alloc, uint32_t *delete_count_out);
void write_bitmap_file_with_subversion(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version, uint32_t subversion, const uint8_t *bitmap, Size bitmap_size, uint32_t delete_count);
void write_bitmap_for_memtable(Oid indexRelId, SegmentId memtable_id, uint8_t *bitmap, Size bitmap_size, uint32_t delete_count);
void load_mapping_file(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version, int64_t **mapping, bool pg_alloc);
#define LOAD_LATEST_VERSION UINT32_MAX

// helper functions (memtable)
ConcurrentMemTable MT_FROM_SLOTIDX(int slot_num);
void *VEC_BASE_FROM_MT(ConcurrentMemTable mt);
#endif