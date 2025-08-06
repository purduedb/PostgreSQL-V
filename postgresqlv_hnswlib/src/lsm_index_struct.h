#ifndef LSM_INDEX_STRUCT_H
#define LSM_INDEX_STRUCT_H

#include "postgres.h"
#include "access/genam.h"
#include "access/parallel.h"
#include "faiss_index.hpp"
#include "storage/lwlock.h"

typedef int64_t VectorId;
typedef uint32_t SegmentId;

typedef enum IndexType
{
    IVFFLAT, 
    HNSW
}IndexType;

// memtable
#define MEMTABLE_MAX_SIZE 10000 // 10K
// bitmap
#define MEMTABLE_BITMAP_SIZE ((MEMTABLE_MAX_SIZE + 7) / 8) // in bytes
#define IS_SLOT_SET(bitmap, i)   (bitmap[(i) >> 3] & (1 << ((i) & 7)))
#define SET_SLOT(bitmap, i)      (bitmap[(i) >> 3] |= (1 << ((i) & 7)))
#define CLEAR_SLOT(bitmap, i)    (bitmap[(i) >> 3] &= ~(1 << ((i) & 7)))
#define GET_BITMAP_SIZE(size) (((size) + 7) / 8)

typedef struct ConcurrentMemTableData {
    VectorId start_vid;
    uint32 max_size;
    pg_atomic_uint32 current_size;

    // The metadata of the vector 
    uint32_t dim;
    uint32_t elem_size; // the size of each dimension

    // use atomic flag per slot
    // pg_atomic_uint32 written[MEMTABLE_MAX_SIZE];
    int64_t tids[MEMTABLE_MAX_SIZE];
    uint8_t bitmap[MEMTABLE_BITMAP_SIZE];

    // followed by an in-memory array of vectors
} ConcurrentMemTableData;
typedef ConcurrentMemTableData * ConcurrentMemTable;

// FIXME: we now store the bitmap and mapping in a continuous array; however, further optimizations are required to support sparse VId
typedef struct SegmentData
{
    pg_atomic_uint32 valid; // can act as a lock when there are multiple background workers
    // TODO: load index, bitmap, mapping lazily when needed
    // pg_atomic_uint32 loaded;

    SegmentId segmentId;
    VectorId lowestVid;
    VectorId highestVid;

    size_t indexSize;
    dsm_handle index;   // SegmentIndexData
    size_t mapSize;
    dsm_handle mapping;
    size_t bitmapSize;
    dsm_handle bitmap;
    // other segment metadata here

    // local segment cache (can only be used by one backend)
    // int deserialized_index_cache_slot;
    dsm_segment * segment_index_cached_seg;
    dsm_segment * segment_mapping_cached_seg;
    dsm_segment * segment_bitmap_cached_seg;
} SegmentData;

#define DEFAULT_SEALED_MEMTABLE_CAPACITY 4
#define DEFAULT_SEGMENT_CAPACITY 100
#define START_SEGMENT_ID 3
#define INVALID_SEGMENT_ID 0
#define FLUSHED_SEGMENT_ID 1

#define LSM_TRANCHE_ID 1000
#define LSM_LOCK_TRANCHE_NAME "lsm_index_lock"

typedef struct LSMIndexData
{
    // FIXME: If you initialized lsm_index in local memory or forgot to use ShmemInitStruct() to allocate and register the lock, then LWLockAcquire() will crash.
    // concurrency bt the background writer and the backend (deletion)
    LWLock *lock;
    // TODO: initialize recovered flag
    pg_atomic_uint32 recovered;

    // The metadata of the vector 
    IndexType index_type;
    uint32_t dim;
    uint32_t elem_size;
    pg_atomic_uint32 currentSegmentId;

    // flushed segments
    // uint32_t segmentNum; 
    // limitation: the segment buffer may overflow due to the naive implementation
    SegmentData segments[DEFAULT_SEGMENT_CAPACITY];

    // in-memory memtables 
    SegmentId memtableID;
    dsm_handle memtable;
    pg_atomic_uint32 sealedMemtableIds[DEFAULT_SEALED_MEMTABLE_CAPACITY];
    dsm_handle sealedMemtables[DEFAULT_SEALED_MEMTABLE_CAPACITY];

    // local segment cache (can only be used by one backend)
    dsm_segment * memtableCachedSeg;
    dsm_segment * sealedCachedSegs[DEFAULT_SEALED_MEMTABLE_CAPACITY];
} LSMIndexData;
typedef LSMIndexData * LSMIndex;

typedef struct LSMIndexBufferSlot
{
    pg_atomic_uint32 valid;

    Oid indexRelId;
    dsm_handle lsm_handle;

    // local segment cache (can only be used by one backend)
    dsm_segment * lsm_index_cached_seg;
}   LSMIndexBufferSlot;

#define INDEX_BUF_SIZE 4
typedef struct LSMIndexBuffer
{
    LSMIndexBufferSlot slots[INDEX_BUF_SIZE];
}   LSMIndexBuffer;

extern LSMIndexBuffer *SharedLSMIndexBuffer;

void lsm_index_buffer_shmem_initialize();

int64_t ItemPointerToInt64(const ItemPointer tid);

ItemPointerData Int64ToItemPointer(int64_t encoded);

Pointer get_lsm_memtable_pointer(uint32_t pos, ConcurrentMemTable mt);

VectorId insert_lsm_index(Relation index, const void *vector, int64_t tid);

int build_lsm_index(IndexType type, Oid relId, void *vector_index, int64_t *tids, uint32_t dim, uint32_t elem_size, VectorId lowest_vid, VectorId highest_vid);

TopKTuples search_lsm_index(Relation index, const float *query_vector, int top_k, int nprobe_efs);

void bulk_delete_lsm_index(Relation index, VectorId *vec_ids, int delete_num);

// write and load files
typedef enum SegmentFileKind
{
    SEGMENT_INDEX,
    SEGMENT_BITMAP,
    SEGMENT_MAPPING
} SegmentFileKind;

void write_lsm_index_metadata(Oid relId, LSMIndex lsm_index);

void write_segment_file(Oid indexRelId, SegmentId segmentId, const void *data, Size size, SegmentFileKind kind);

void load_segment_file(Oid indexRelId, SegmentData *segment, uint32_t segment_id, SegmentFileKind kind);

// ------------------- not yet implemented --------------------------
// void free_lsm_index(uint32_t slot_num);

#endif

/*
    TODO
    - move the cached segments out of the structure and store them in the local memory of each process instead of the shared memory
    - 
*/