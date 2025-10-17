#ifndef LSM_MERGE_WORKER_H
#define LSM_MERGE_WORKER_H

#include "c.h"
#include "postgres.h"
#include "lsmindex.h"
#include "ringbuffer.h"

// Configuration constants
#define MERGE_DELETION_RATIO_THRESHOLD 0.3f  // 30% deletion ratio threshold
#define MAX_MERGE_WORKERS 4                  // Maximum number of merge workers
#define MERGE_TASK_QUEUE_SIZE 1024           // Size of merge task queue
#define MAX_SEGMENTS_SIZE 10000000            // The max number of vectors in a segment, the segment will not be merged if the number of vectors is larger than this value
#define THRESHOLD_SMALL_SEGMENT_SIZE 1000000   // The min number of vectors in a segment, the segment will be merged if the number of vectors is smaller than this value

// Segment information structure for merge worker (scanned from disk)
typedef struct {
    bool in_used;
    bool is_compacting;
    uint32_t next_idx;  // Linked list pointer
    uint32_t prev_idx;  // Linked list pointer

    SegmentId start_sid;
    SegmentId end_sid;
    uint32_t vec_count;
    IndexType index_type;

    // staticstics
    // TODO: 
    float deletion_ratio;
} MergeSegmentInfo;

typedef struct {
    int operation_type; // 0=regular update, 1=rebuild_flat, 2=rebuild_deletion, 3=merge
    uint32_t segment_idx0;
    uint32_t segment_idx1;
    SegmentId start_sid;
    SegmentId end_sid;
    TimestampTz start_time;
    Oid index_relid;   // Index this task belongs to
} MergeTaskData;

// Merge worker state
typedef struct {
    int worker_id;
    bool active;
    MergeTaskData current_task;
} MergeWorkerState;

// Shared segment information for all merge workers
typedef struct {
    LWLock *lock;
    Oid current_index_relid;       // Index this segment array belongs to

    // segment information
    MergeSegmentInfo segments[MAX_SEGMENTS_COUNT];
    uint32_t segment_count;
    uint32_t tail_idx;             // Index of the last segment in the linked list
    uint32_t insert_idx;           // Next available slot for insertion
    SegmentId max_end_segment_id;  // Current maximum end segment ID
    
    // staticstics
    pg_atomic_uint32 flat_count; // number of flat segments
    pg_atomic_uint32 memtable_capacity_le_count; // count of segments with vec_count <= MEMTABLE_MAX_CAPACITY
    pg_atomic_uint32 small_segment_le_count;     // count of segments with vec_count <= THRESHOLD_SMALL_SEGMENT_SIZE
} SharedSegmentArray;

// Global merge worker management
typedef struct {
    LWLock *lock;
    MergeWorkerState workers[MAX_MERGE_WORKERS];
    int active_worker_count;
    SharedSegmentArray segment_arrays[INDEX_BUF_SIZE]; // One per LSM index
} MergeWorkerManager;

// Function declarations
extern MergeWorkerManager *merge_worker_manager;

// Initialization and management
void initialize_merge_worker_manager(void);

// Linked list management functions (similar to FlushedSegmentPool)
void add_to_segment_array(uint32_t lsm_idx, Oid index_relid, SegmentId start_sid, SegmentId end_sid, uint32_t vec_count, IndexType index_type, float deletion_ratio);

// Merge operations
void rebuild_index(Oid index_relid, int lsm_idx, uint32_t segment_idx, IndexType target_type);
void merge_adjacent_segments(Oid index_relid, int lsm_idx, uint32_t segment_idx0, uint32_t segment_idx1);

// Background worker main function
void lsm_merge_worker_main(Datum main_arg);

// Internal function for sending tasks to vector index worker
void send_merge_task_to_vector_worker(SegmentUpdateTaskData *task_data);

#endif // LSM_MERGE_WORKER_H
