#include "postgres.h"
#include "fmgr.h"                         // For PG_MODULE_MAGIC
#include "miscadmin.h"                    // For MyProcPid, MyDatabaseId, etc.
#include "storage/ipc.h"                  // For on_proc_exit
#include "storage/lwlock.h"               // For LWLock (if you use them)
#include "storage/shmem.h"                // For ShmemInitStruct
#include "storage/dsm.h"                  // For DSM functions (dsm_create, attach, etc.)
#include "storage/pg_shmem.h"             // For shared memory setup
#include "storage/proc.h"                 // For PGPROC and current backend info
#include "storage/latch.h"                // For latch functions

#include "postmaster/bgworker.h"         // Required for background worker APIs
#include "postmaster/postmaster.h"       // For BackgroundWorkerStartTime
#include "tcop/tcopprot.h"               // For ReadyForQuery
#include "utils/builtins.h"              // For elog(), etc.
#include "utils/elog.h"
#include "utils/memutils.h"              // For MemoryContext
#include "utils/guc.h"                   // For custom GUCs if any
#include "storage/pmsignal.h"
#include "utils/wait_event.h"
#include "portability/instr_time.h"
#include "storage/proc.h"
#include <sys/types.h>
#include <time.h>

#include "vector_index_worker.h"
#include "lsmindex.h"
#include "vectorindeximpl.hpp"
#include "lsm_segment.h"
#include "lsm_merge_worker.h"
#include <pthread.h>
#include <stdatomic.h>

static volatile sig_atomic_t got_sighup = false;
static volatile sig_atomic_t got_sigterm = false;

// Thread pool for maintenance tasks
#define MAINTENANCE_THREAD_POOL_SIZE 4
#define MAINTENANCE_TASK_QUEUE_SIZE 128

typedef struct MaintenanceTask {
    TaskSlot *task_slot;
    struct MaintenanceTask *next;
} MaintenanceTask;

typedef struct MaintenanceThreadPool {
    pthread_t threads[MAINTENANCE_THREAD_POOL_SIZE];
    pthread_mutex_t queue_mutex;
    pthread_cond_t queue_cond;
    MaintenanceTask *task_queue_head;
    MaintenanceTask *task_queue_tail;
    atomic_int queue_size;
    atomic_int shutdown;
    int num_threads;
} MaintenanceThreadPool;

static MaintenanceThreadPool *maintenance_pool = NULL;

// Worker thread function for maintenance tasks
static void *
maintenance_worker_thread(void *arg)
{   
    // fprintf(stderr, "enter maintenance_worker_thread\n");

    MaintenanceThreadPool *pool = (MaintenanceThreadPool *)arg;
    
    while (!atomic_load(&pool->shutdown)) {
        MaintenanceTask *task = NULL;
        
        // Acquire lock and wait for a task
        pthread_mutex_lock(&pool->queue_mutex);
        
        while (pool->task_queue_head == NULL && !atomic_load(&pool->shutdown)) {
            // fprintf(stderr, "[maintenance_worker] Thread waiting for task, queue_size = %d\n", atomic_load(&pool->queue_size));
            pthread_cond_wait(&pool->queue_cond, &pool->queue_mutex);
            // fprintf(stderr, "[maintenance_worker] Thread woke up from condition wait\n");
        }
        
        if (atomic_load(&pool->shutdown) && pool->task_queue_head == NULL) {
            pthread_mutex_unlock(&pool->queue_mutex);
            break;
        }
        
        // Dequeue a task
        task = pool->task_queue_head;
        if (task != NULL) {
            pool->task_queue_head = task->next;
            if (pool->task_queue_head == NULL) {
                pool->task_queue_tail = NULL;
            }
            atomic_fetch_sub(&pool->queue_size, 1);
            // fprintf(stderr, "[maintenance_worker] Dequeued task, queue_size now = %d\n", atomic_load(&pool->queue_size));
        }
        
        pthread_mutex_unlock(&pool->queue_mutex);
        
        if (task != NULL) {
            // Handle the maintenance task
            TaskSlot *task_slot = task->task_slot;
            PGPROC *client = NULL;
            dsm_segment *task_seg = NULL;
            
            switch (task_slot->type) {
            case IndexBuildTaskType:
            {
                fprintf(stderr, "[maintenance_worker] IndexBuildTaskType case entered\n");
                dsm_handle task_hdl = task_slot->handle;
                task_seg = dsm_attach(task_hdl);
                IndexBuildTask build_task = (IndexBuildTask) dsm_segment_address(task_seg);
                                
                Oid index_relid = build_task->index_relid;
                int lsm_idx = build_task->lsm_idx;
                
                // initialize flushed segments
                FlushedSegmentPool *pool_seg = get_flushed_segment_pool(lsm_idx);
                initialize_segment_pool(pool_seg);
                uint32_t seg_idx = reserve_flushed_segment(pool_seg);
                if(seg_idx == -1) {
                    fprintf(stderr, "[maintenance_worker] ERROR: no free flushed segment slot\n");
                } else {
                    load_and_set_segment(index_relid, seg_idx, &pool_seg->flushed_segments[seg_idx], START_SEGMENT_ID, START_SEGMENT_ID, LOAD_LATEST_VERSION);
                    register_flushed_segment(pool_seg, seg_idx);
                }
                
                client = &ProcGlobal->allProcs[build_task->backend_pgprocno];
                dsm_detach(task_seg);
                break;
            }
            case SegmentUpdateTaskType:
            {
                fprintf(stderr, "[maintenance_worker] SegmentUpdateTaskType\n");
                dsm_handle task_hdl = task_slot->handle;
                task_seg = dsm_attach(task_hdl);
                SegmentUpdateTask task = (SegmentUpdateTask) dsm_segment_address(task_seg);
                
                Oid index_relid = task->index_relid;
                int lsm_idx = task->lsm_idx;
                
                // Handle different types of segment update operations
                switch (task->operation_type)
                {
                    case SEGMENT_UPDATE_REGULAR:
                    {
                        FlushedSegmentPool *pool_seg = get_flushed_segment_pool(lsm_idx);
                        
                        pthread_rwlock_wrlock(&pool_seg->seg_lock);
                        uint32_t seg_idx = reserve_flushed_segment(pool_seg);
                        pthread_rwlock_unlock(&pool_seg->seg_lock);
                        
                        load_and_set_segment(index_relid, seg_idx, &pool_seg->flushed_segments[seg_idx], task->start_sid, task->end_sid, LOAD_LATEST_VERSION);
                        
                        pthread_rwlock_wrlock(&pool_seg->seg_lock);
                        register_flushed_segment(pool_seg, seg_idx);
                        pthread_rwlock_unlock(&pool_seg->seg_lock);
                        break;
                    }
                    
                    case SEGMENT_UPDATE_REBUILD_FLAT:
                    case SEGMENT_UPDATE_REBUILD_DELETION:
                    {
                        const char *rebuild_type = (task->operation_type == SEGMENT_UPDATE_REBUILD_FLAT) ? 
                            "IndexRebuildFlat" : "IndexRebuildDeletion";
                        fprintf(stderr, "[maintenance_worker] %s task received, start_sid = %d, end_sid = %d\n", 
                             rebuild_type, task->start_sid, task->end_sid);
                        
                        FlushedSegmentPool *pool_seg = get_flushed_segment_pool(lsm_idx);
                        
                        pthread_rwlock_rdlock(&pool_seg->seg_lock);
                        uint32_t old_seg_idx = find_segment_by_sids(pool_seg, task->start_sid, task->end_sid);
                        pthread_rwlock_unlock(&pool_seg->seg_lock);
                        
                        pthread_rwlock_wrlock(&pool_seg->seg_lock);
                        uint32_t new_seg_idx = reserve_flushed_segment(pool_seg);
                        pthread_rwlock_unlock(&pool_seg->seg_lock);
                        if (new_seg_idx == -1) {
                            fprintf(stderr, "[maintenance_worker] ERROR: No free flushed segment slot available for rebuild\n");
                            break;
                        }
                        
                        load_and_set_segment(index_relid, new_seg_idx, &pool_seg->flushed_segments[new_seg_idx], task->start_sid, task->end_sid, LOAD_LATEST_VERSION);
                        
                        pthread_rwlock_wrlock(&pool_seg->seg_lock);
                        replace_flushed_segment(pool_seg, old_seg_idx, -1, new_seg_idx);
                        pthread_rwlock_unlock(&pool_seg->seg_lock);
                        
                        fprintf(stderr, "[maintenance_worker] Rebuild completed for segment %d-%d, replaced old at slot %u with new at slot %u\n", 
                             task->start_sid, task->end_sid, old_seg_idx, new_seg_idx);
                        break;
                    }
                    
                    case SEGMENT_UPDATE_MERGE:
                    {
                        fprintf(stderr, "[maintenance_worker] SegmentMerge task received: merging segments to create segment %d-%d\n", 
                             task->start_sid, task->end_sid);
                        
                        FlushedSegmentPool *pool_seg = get_flushed_segment_pool(lsm_idx);
                        
                        uint32_t old_seg_idx_0, old_seg_idx_1;
                        pthread_rwlock_rdlock(&pool_seg->seg_lock);
                        find_two_adjacent_segments(pool_seg, task->start_sid, task->end_sid, &old_seg_idx_0, &old_seg_idx_1);
                        pthread_rwlock_unlock(&pool_seg->seg_lock);
                        
                        if (old_seg_idx_0 == -1 || old_seg_idx_1 == -1)
                        {
                            fprintf(stderr, "[maintenance_worker] ERROR: Failed to find two adjacent segments for merge\n");
                            break;
                        }
                        fprintf(stderr, "[maintenance_worker] Found adjacent segments %u and %u for merge\n", old_seg_idx_0, old_seg_idx_1);
                        
                        pthread_rwlock_wrlock(&pool_seg->seg_lock);
                        uint32_t new_seg_idx = reserve_flushed_segment(pool_seg);
                        pthread_rwlock_unlock(&pool_seg->seg_lock);
                        if (new_seg_idx == -1) {
                            fprintf(stderr, "[maintenance_worker] ERROR: No free flushed segment slot available for merge\n");
                            break;
                        }
                        
                        load_and_set_segment(index_relid, new_seg_idx, &pool_seg->flushed_segments[new_seg_idx], task->start_sid, task->end_sid, LOAD_LATEST_VERSION);
                        
                        pthread_rwlock_wrlock(&pool_seg->seg_lock);
                        replace_flushed_segment(pool_seg, old_seg_idx_0, old_seg_idx_1, new_seg_idx);
                        pthread_rwlock_unlock(&pool_seg->seg_lock);
                        
                        fprintf(stderr, "[maintenance_worker] SegmentMerge completed, replaced segments %u and %u with segment %u\n", 
                             old_seg_idx_0, old_seg_idx_1, new_seg_idx);
                        break;
                    }
                    
                    case SEGMENT_UPDATE_VACUUM:
                    {
                        fprintf(stderr, "[maintenance_worker] SegmentVacuum task received: reloading bitmap for segment %d-%d\n", 
                             task->start_sid, task->end_sid);
                        
                        FlushedSegmentPool *pool_seg = get_flushed_segment_pool(lsm_idx);
                        
                        pthread_rwlock_rdlock(&pool_seg->seg_lock);
                        uint32_t seg_idx = find_segment_by_sids(pool_seg, task->start_sid, task->end_sid);
                        pthread_rwlock_unlock(&pool_seg->seg_lock);
                        
                        FlushedSegment segment = &pool_seg->flushed_segments[seg_idx];
                        
                        if (segment->bitmap_ptr == NULL) {
                            fprintf(stderr, "[maintenance_worker] ERROR: Segment %d-%d has NULL bitmap pointer, cannot perform vacuum\n", 
                                 task->start_sid, task->end_sid);
                            break;
                        }
                        
                        uint8_t *new_bitmap_ptr = NULL;
                        load_bitmap_file(index_relid, task->start_sid, task->end_sid, LOAD_LATEST_VERSION, &new_bitmap_ptr, false);
                        
                        if (new_bitmap_ptr == NULL) {
                            fprintf(stderr, "[maintenance_worker] ERROR: Failed to load new bitmap for segment %d-%d during vacuum\n", 
                                 task->start_sid, task->end_sid);
                            break;
                        }
                        
                        Size bitmap_size = GET_BITMAP_SIZE(segment->vec_count);
                        uint8_t *current_bitmap = segment->bitmap_ptr;
                        
                        for (Size i = 0; i < bitmap_size; i++) {
                            current_bitmap[i] |= new_bitmap_ptr[i];
                        }
                        
                        free(new_bitmap_ptr);
                        
                        fprintf(stderr, "[maintenance_worker] SegmentVacuum completed, merged new bitmap into segment %d-%d at slot %u\n", 
                             task->start_sid, task->end_sid, seg_idx);
                        break;
                    }
                    
                    default:
                        fprintf(stderr, "[maintenance_worker] WARNING: Unknown segment update operation type %d\n", task->operation_type);
                        break;
                }
                client = &ProcGlobal->allProcs[task->backend_pgprocno];
                dsm_detach(task_seg);
                break;
            }
            case IndexLoadTaskType:
            {
                fprintf(stderr, "[maintenance_worker] IndexLoadTaskType\n");
                dsm_handle task_hdl = task_slot->handle;
                task_seg = dsm_attach(task_hdl);
                IndexLoadTask task = (IndexLoadTask) dsm_segment_address(task_seg);
                
                Oid index_relid = task->index_relid;
                int lsm_idx = task->lsm_idx;
                
                FlushedSegmentPool *pool_seg = get_flushed_segment_pool(lsm_idx);
                initialize_segment_pool(pool_seg);
                load_all_segments_from_disk(index_relid, pool_seg);
                
                client = &ProcGlobal->allProcs[task->backend_pgprocno];
                dsm_detach(task_seg);
                break;
            }
            default:
                fprintf(stderr, "[maintenance_worker] Unknown task type %d\n", task_slot->type);
                break;
            }
            
            // Notify the client backend
            if (client != NULL) {
                SetLatch(&client->procLatch);
            }
            
            // Free the task structure
            free(task);
        }
    }
    
    return NULL;
}

// Initialize the maintenance thread pool
static void
init_maintenance_thread_pool(void)
{
    if (maintenance_pool != NULL) {
        return; // Already initialized
    }
    
    maintenance_pool = (MaintenanceThreadPool *)malloc(sizeof(MaintenanceThreadPool));
    if (maintenance_pool == NULL) {
        elog(ERROR, "[init_maintenance_thread_pool] Failed to allocate thread pool");
        return;
    }
    
    maintenance_pool->task_queue_head = NULL;
    maintenance_pool->task_queue_tail = NULL;
    atomic_init(&maintenance_pool->queue_size, 0);
    atomic_init(&maintenance_pool->shutdown, 0);
    maintenance_pool->num_threads = MAINTENANCE_THREAD_POOL_SIZE;
    
    pthread_mutex_init(&maintenance_pool->queue_mutex, NULL);
    pthread_cond_init(&maintenance_pool->queue_cond, NULL);
    
    // Create worker threads
    for (int i = 0; i < maintenance_pool->num_threads; i++) {
        if (pthread_create(&maintenance_pool->threads[i], NULL, maintenance_worker_thread, maintenance_pool) != 0) {
            elog(ERROR, "[init_maintenance_thread_pool] Failed to create worker thread %d", i);
        }
    }
    
    elog(DEBUG1, "[init_maintenance_thread_pool] Initialized maintenance thread pool with %d threads", maintenance_pool->num_threads);
}

// Submit a maintenance task to the thread pool
static void
submit_maintenance_task(TaskSlot *task_slot)
{
    elog(DEBUG1, "[submit_maintenance_task] submitting maintenance task, task_slot->type = %d", task_slot->type);

    if (maintenance_pool == NULL) {
        init_maintenance_thread_pool();
    }
    
    // Check if queue is full
    if (atomic_load(&maintenance_pool->queue_size) >= MAINTENANCE_TASK_QUEUE_SIZE) {
        elog(WARNING, "[submit_maintenance_task] Task queue is full, dropping task");
        return;
    }
    
    // Allocate task structure
    MaintenanceTask *task = (MaintenanceTask *)malloc(sizeof(MaintenanceTask));
    if (task == NULL) {
        elog(ERROR, "[submit_maintenance_task] Failed to allocate task structure");
        return;
    }
    
    task->task_slot = task_slot;
    task->next = NULL;
    
    // Enqueue the task
    pthread_mutex_lock(&maintenance_pool->queue_mutex);
    
    if (maintenance_pool->task_queue_tail == NULL) {
        maintenance_pool->task_queue_head = task;
        maintenance_pool->task_queue_tail = task;
    } else {
        maintenance_pool->task_queue_tail->next = task;
        maintenance_pool->task_queue_tail = task;
    }
    
    atomic_fetch_add(&maintenance_pool->queue_size, 1);
    
    pthread_cond_signal(&maintenance_pool->queue_cond);
    pthread_mutex_unlock(&maintenance_pool->queue_mutex);
}

static void
vq_sighup(SIGNAL_ARGS)
{
    int save_errno = errno;
    got_sighup = true;
    SetLatch(MyLatch);
    errno = save_errno;
}

static void
vq_sigterm(SIGNAL_ARGS)
{
    int save_errno = errno;
    got_sigterm = true;
    /* Arrange to break out of WaitLatch */
    SetLatch(MyLatch);
    errno = save_errno;
    /* let the main loop see WL_EXIT_ON_PM_DEATH */
}

static void handle_task(TaskSlot *task_slot, int slot_idx);
static void vector_search(VectorSearchTask task, VectorSearchResult result);

void 
vector_index_worker_main(Datum main_arg)
{
    elog(DEBUG1, "enter vector_index_worker_main");

    // Initialize the maintenance thread pool in the background worker process
    init_maintenance_thread_pool();
    
    // FIXME: check this block
    pqsignal(SIGHUP, vq_sighup);
    pqsignal(SIGTERM, vq_sigterm);
    BackgroundWorkerUnblockSignals();
    // LWLockRegisterTranche(VECTOR_SEARCH_RING_TRANCHE_ID, VECTOR_SEARCH_RING_TRANCHE);

    // FIXME: set worker_pgprocno in the init stage
    LWLockAcquire(ring_buffer_shmem->lock, LW_EXCLUSIVE);
    ring_buffer_shmem->worker_pgprocno = MyProc->pgprocno;
    LWLockRelease(ring_buffer_shmem->lock);

    ResetLatch(MyLatch);
    for (;;)
    {
        int rc = WaitLatch(MyLatch,
                           WL_LATCH_SET | WL_POSTMASTER_DEATH,
                           0, /* no timeout */
                           0  /* no wait-event reporting */);
        ResetLatch(MyLatch);
        
        if (rc & WL_POSTMASTER_DEATH)
        {
            proc_exit(0);
        }
        
        // for graceful shutdown
        if (got_sigterm)
        {
            proc_exit(0);
        }
        
        // Postgres uses this to tell backends/workers to reload configuration
        if (got_sighup)
        {
            got_sighup = false;
            ProcessConfigFile(PGC_SIGHUP);
        }

        for (;;)
        {
            TaskSlot *task_slot;

            LWLockAcquire(ring_buffer_shmem->lock, LW_EXCLUSIVE);
            if (ring_buffer_shmem->count == 0)
            {
                LWLockRelease(ring_buffer_shmem->lock);
                break;
            }

            int idx = ring_buffer_shmem->head;
            task_slot = vs_task_at(idx);

            handle_task(task_slot, idx);

            ring_buffer_shmem->head = (ring_buffer_shmem->head + 1) % ring_buffer_shmem->ring_size;
            ring_buffer_shmem->count--;
            // elog(DEBUG1, "[vector_index_worker_main] ring buffer's head = %d, ring buffer's count = %d", ring_buffer_shmem->head, ring_buffer_shmem->count);
            LWLockRelease(ring_buffer_shmem->lock);
        }
    }
}

void 
vector_index_worker_init(void)
{
    // NOTE: Do NOT initialize the maintenance thread pool here!
    // The thread pool must be initialized in the background worker process
    // (vector_index_worker_main), not during shared memory initialization.
    // This function is called from shmem_startup() which runs in every backend,
    // and we want the threads to run in the background worker process only.
}

// the lock is held
static void
handle_task(TaskSlot *task_slot, int slot_idx)
{
    switch (task_slot->type)
    {
    case VectorSearchTaskType:
    {
        // get the task
        VectorSearchTask task = vs_search_task_at(slot_idx);

        // conduct the search (concurrent search will handle merging, writing results, and setting latch)
        VectorSearchResult result = vs_search_result_at(task->backend_pgprocno);
        vector_search(task, result);

        // Note: The latch is set by the concurrent search function (last-finisher continuation)
        // No need to set it here
        break;
    }
    case IndexBuildTaskType:
    case SegmentUpdateTaskType:
    case IndexLoadTaskType:
    {
        // Submit maintenance tasks to the thread pool
        // The worker thread will handle the task and notify the backend
        submit_maintenance_task(task_slot);
        break;
    }
    default:
        break;
    }
}

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
    the backend process need to snapshot the memtables' segment ids it needs to search before sending tasks to the worker
    the vector index search process is only responsible for conducting searches on sealed segments
*/
static void 
vector_search(VectorSearchTask task, VectorSearchResult result)
{
    // traverse all flushed segments
    int lsm_idx = get_lsm_index_idx(task->index_relid);
    FlushedSegmentPool *pool = get_flushed_segment_pool(lsm_idx);
    Assert(lsm);
    
    // Acquire segment lock to prevent concurrent segment updates during search
    // Use shared lock to allow multiple concurrent searches but prevent updates
    pthread_rwlock_rdlock(&pool->seg_lock);
    
    // Take a snapshot of segment structure to avoid holding lock too long
    uint32_t tail_idx_snapshot = pool->tail_idx;
    
    // Create a local copy of segment indices to traverse
    uint32_t segment_indices[MAX_SEGMENTS_COUNT];
    uint32_t segment_count = 0;
    uint32_t idx = pool->head_idx;

    // Build segment traversal list while holding the lock
    if (idx != (uint32_t)-1) {
        do {
            if (pool->flushed_segments[idx].in_used) {
                // Increment reference count when adding to traversal list
                atomic_fetch_add(&pool->flushed_segments[idx].ref_count, 1);
                segment_indices[segment_count++] = idx;
            }
            if (idx == tail_idx_snapshot) break;
            idx = pool->flushed_segments[idx].next_idx;
        } while (true);
    }
    
    // Release lock early to minimize contention
    pthread_rwlock_unlock(&pool->seg_lock);
    
    // Filter segments and build SegmentSearchInfo array for segments that need to be searched
    // Note: this should be freed after the search is complete by the last-finisher thread, but not in this function
    SegmentSearchInfo *segments_to_search = malloc(sizeof(SegmentSearchInfo) * segment_count);
    uint32_t search_count = 0;
    
    for (uint32_t i = 0; i < segment_count; i++) {
        uint32_t seg_idx = segment_indices[i];
        FlushedSegment segment = &pool->flushed_segments[seg_idx];
        
        // Check if this segment should be skipped (if its segment id is in the snapshot)
        bool found = false;
        for (int j = 0; j < task->snapshot.scount; j++)
        {
            if (task->snapshot.smt_ids[j] <= segment->segment_id_end &&
                task->snapshot.smt_ids[j] >= segment->segment_id_start)
            {
                Assert(segment->segment_id_end == segment->segment_id_start);
                found = true;
                break;
            }
        }
        found = found || (task->snapshot.gmt_id <= segment->segment_id_end &&
                        task->snapshot.gmt_id >= segment->segment_id_start);
        
        if (!found)
        {   
            // This segment needs to be searched - add it to the search list
            segments_to_search[search_count].index_type = segment->index_type;
            segments_to_search[search_count].index_ptr = segment->index_ptr;
            segments_to_search[search_count].bitmap_ptr = segment->bitmap_ptr;
            segments_to_search[search_count].vec_count = segment->vec_count;
            segments_to_search[search_count].map_ptr = segment->map_ptr;
            segments_to_search[search_count].segment_idx = seg_idx;
            search_count++;
        } else {
            // Skip this segment - decrement reference count
            decrement_flushed_segment_ref_count(pool, seg_idx);
        }
    }

    // Get the client backend process for setting the latch
    PGPROC *client = &ProcGlobal->allProcs[task->backend_pgprocno];
    
    // Conduct concurrent search on all segments that need to be searched
    // The concurrent function will handle merging, writing results, and setting the latch
    if (search_count > 0) {
        // Submit concurrent searches - function returns immediately
        // The last-finisher thread will handle merging, writing results, 
        // decrementing reference counts, and setting the latch
        ConcurrentVectorSearchOnSegments(
            segments_to_search,
            search_count,
            vs_search_task_vector_at(task),
            task->topk,
            task->efs_nprobe,
            result,
            client,
            pool  // Pass pool pointer for reference count decrementing
        );
        
        // Note: Reference counts will be decremented by the last-finisher thread
        // after all searches complete. Do not decrement them here.
    } else {
        // No segments to search - set empty result and notify client
        result->result_count = 0;
        SetLatch(&client->procLatch);
    }

}

// TaskDesc *task_decs = NULL;

// void 
// vector_index_worker_init(void)
// {
//     bool found;
//     task_decs = ShmemInitStruct("knowhere task ring",
//                                   sizeof(TaskDesc), &found);
// }

// // a global flag to track whether the background worker received a SIGTERM
// static volatile sig_atomic_t got_sigterm = false;

// static inline bool
// OurPostmasterIsAlive(void)
// {
//  if (likely(!postmaster_possibly_dead))
//      return true;
//  return PostmasterIsAliveInternal();
// }

// void 
// vector_index_worker_main(Datum main_arg)
// {
//     pqsignal(SIGTERM, die);
//     BackgroundWorkerUnblockSignals();

//     /* publish worker proc (guard with a lightweight lock if you prefer) */
//     task_decs->worker_proc = MyProc;

//     while (!got_sigterm)
//     {
//         CHECK_FOR_INTERRUPTS();

//         int rc = WaitLatch(&MyProc->procLatch,
//                            WL_LATCH_SET | WL_POSTMASTER_DEATH,
//                            0, /* no timeout */
//                            0  /* no wait-event reporting */);
//         ResetLatch(&MyProc->procLatch);
//         if (rc & WL_POSTMASTER_DEATH)
//             proc_exit(1);

//         /* echo back to client if present */
//         PGPROC *client = task_decs->client_proc;   /* take a stable snapshot */
//         if (client)
//             SetLatch(&client->procLatch);
//     }
// }

// test communication overhead
// add the following to sql/vector.sql
// CREATE OR REPLACE FUNCTION pg_knowhere_ping_us(iters int)
// RETURNS float8
// AS 'MODULE_PATHNAME', 'pg_knowhere_ping_us'
// LANGUAGE C STRICT;

// PG_FUNCTION_INFO_V1(pg_knowhere_ping_us);
// Datum
// pg_knowhere_ping_us(PG_FUNCTION_ARGS)
// {
//     int32 iters = PG_GETARG_INT32(0);
//     if (iters <= 0) iters = 1000;

//     while (task_decs->worker_proc == NULL)
//         sleep(1);  /* wait for worker to start */

//     /* register myself as the client */
//     task_decs->client_proc = MyProc;

//     instr_time total; INSTR_TIME_SET_ZERO(total);

//     for (int i = 0; i < iters; i++)
//     {
//         instr_time t0, t1;

//         INSTR_TIME_SET_CURRENT(t0);

//         /* wake worker */
//         SetLatch(&task_decs->worker_proc->procLatch);

//         /* wait for echo (worker will SetLatch on our proc) */
//         (void) WaitLatch(&MyProc->procLatch,
//                          WL_LATCH_SET | WL_POSTMASTER_DEATH,
//                          0,  /* no timeout */
//                          0);
//         ResetLatch(&MyProc->procLatch);

//         INSTR_TIME_SET_CURRENT(t1);
//         INSTR_TIME_SUBTRACT(t1, t0);
//         INSTR_TIME_ADD(total, t1);
//     }

//     double avg_us = INSTR_TIME_GET_DOUBLE(total) * 1e6 / (double) iters;
//     PG_RETURN_FLOAT8(avg_us);
// }