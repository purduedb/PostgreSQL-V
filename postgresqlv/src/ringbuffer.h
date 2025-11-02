#ifndef RINGBUFFER_H
#define RINGBUFFER_H

#include "postgres.h"
#include "access/relscan.h"
#include "catalog/pg_type.h"
#include "fmgr.h"
#include "funcapi.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "storage/lwlock.h"
#include "storage/proc.h"
#include "storage/shmem.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/lsyscache.h"
#include "pgstat.h"

#include "utils.h"
#include "lsmindex.h"

#define MAX_DIM 1024
#define MAX_TOPK 1000
#define VECTOR_SEARCH_RING_TRANCHE "vector search ring buffer"
#define VECTOR_SEARCH_RING_TRANCHE_ID 1000

typedef enum
{
    VectorSearchTaskType,
    IndexBuildTaskType,
    IndexLoadTaskType,
    SegmentUpdateTaskType
} VectorTaskType;

// typedef union 
// {
//     int efs;
//     int nprobe;
// }VectorSearchParam;

typedef struct
{
    int32_t gidx;
    int32_t scount;
    int32_t sidxs[MEMTABLE_NUM];
    SegmentId gmt_id;
    int32_t smt_ids[MEMTABLE_NUM];
}LSMSnapshot;

// TODO: we need to implement a finer lock on each task slot
// TODO: remove the latch on the worker's side (The worker will just poll the ring buffer)
// define the vector search task data structure
typedef struct {
    int backend_pgprocno;
    Oid index_relid;
    Size vector_dim; // do we need this?
    int topk;
    int efs_nprobe;
    LSMSnapshot snapshot;
} VectorSearchTaskData;
// followed by vector data
typedef VectorSearchTaskData* VectorSearchTask;

// define the lsm index build data structure
typedef struct {
    int backend_pgprocno;
    // Oid heap_relid;
    Oid index_relid;
    int lsm_idx;
} IndexBuildTaskData;
typedef IndexBuildTaskData* IndexBuildTask;

// define the flushed segment update structure
// This structure is reused for different types of segment operations:
// - Regular segment updates (old_segment_idx_0 = -1, old_segment_idx_1 = -1)
// - Index rebuild for flat segments (old_segment_idx_0 = segment_idx, old_segment_idx_1 = -1, operation_type = REBUILD_FLAT)
// - Index rebuild for deletion ratio (old_segment_idx_0 = segment_idx, old_segment_idx_1 = -1, operation_type = REBUILD_DELETION)
// - Segment merging (old_segment_idx_0 = source_seg_idx, old_segment_idx_1 = target_seg_idx, operation_type = MERGE)
typedef struct {
    int backend_pgprocno;
    Oid index_relid;
    int lsm_idx;

    int operation_type;     // 0=regular update, 1=rebuild_flat, 2=rebuild_deletion, 3=merge

    // the merged segment information
    SegmentId start_sid;
    SegmentId end_sid;
} SegmentUpdateTaskData;
typedef SegmentUpdateTaskData* SegmentUpdateTask;

// Operation type constants for SegmentUpdateTaskData
#define SEGMENT_UPDATE_REGULAR 0
#define SEGMENT_UPDATE_REBUILD_FLAT 1
#define SEGMENT_UPDATE_REBUILD_DELETION 2
#define SEGMENT_UPDATE_MERGE 3

// define the index load task structure
typedef struct {
    int backend_pgprocno;
    Oid index_relid;
    int lsm_idx;
} IndexLoadTaskData;
typedef IndexLoadTaskData* IndexLoadTask;

typedef struct {
    volatile int status; // 0: empty, 1: done, 2: error
    volatile int result_count;
    Size result_size;
} VectorSearchResultData;
// followed by result data (may include both vids and distances)
typedef VectorSearchResultData* VectorSearchResult;

typedef struct {
    VectorTaskType type;
    Size dsm_size;
    dsm_handle handle; // used by IndexBuild and SegmentUpdate
} TaskSlot;
// [Retionale] Why we do not use dsm_handle for vector search? we do not want to allocate a new dsm for each vector search call

// the ring buffer is included in shared memory
typedef struct{
    // ring buffer
    LWLock *lock;
    int ring_size;
    int head;
    int tail;
    int count;

    int worker_pgprocno;

    Size task_handles_ofst; // initialize to hold `MaxBackends` handles

    Size search_task_stride;
    Size search_result_stride;
} RingBufferShmem;

extern RingBufferShmem *ring_buffer_shmem;
extern VectorSearchTaskData *vector_search_task_pool;
extern VectorSearchResultData *vector_search_result_pool;

Size calculate_ring_buffer_shmem_size();
Size calculate_vector_search_task_pool_size();
Size calculate_vector_search_result_pool_size();
void ring_buffer_init(void);

// helper functions for ring buffer operations
TaskSlot *vs_task_at(int idx);
float* vs_search_result_dist_at(VectorSearchResult result);
int64_t* vs_search_result_id_at(VectorSearchResult result);
float* vs_search_task_vector_at(VectorSearchTask task);
// Size get_vec_search_task_size(int dim, Size elem_size);
// Size get_vec_search_result_size(int top_k, Size id_size, Size dist_size);
VectorSearchTask vs_search_task_at(int idx);
VectorSearchResult vs_search_result_at(int pgprocno);
int get_ring_buffer_slot();
// attach task, release task, write result, etc.

// we don't need a ring buffer as each connection can only send one task at a time

#endif