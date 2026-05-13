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
#include <unistd.h>

#include "vector_index_worker.h"
#include "lsmindex.h"
#include "vectorindeximpl.hpp"
#include "lsm_segment.h"
#include "catalog/index.h"
#include <pthread.h>
#include <stdatomic.h>

static void *merge_worker_thread(void *arg);  /* implemented in Task 8 */
static void signal_merge_pool(void);           /* forward declaration */

static volatile sig_atomic_t got_sighup = false;
static volatile sig_atomic_t got_sigterm = false;

// Thread pool for maintenance tasks
#define MAINTENANCE_THREAD_POOL_SIZE 4
// Must be larger than MAX_SEGMENTS_COUNT to absorb one upgrade task per segment
// plus headroom for regular tasks.
#define MAINTENANCE_TASK_QUEUE_SIZE 1152

// Data for an internal (non-ring-buffer) mmap->full-memory upgrade task.
typedef struct {
    Oid index_relid;
    int lsm_idx;
    uint32_t segment_idx;   // slot in FlushedSegmentPool
    SegmentId start_sid;
    SegmentId end_sid;
    uint32_t version;
    IndexType index_type;
} InternalUpgradeTaskData;

typedef struct MaintenanceTask {
    TaskSlot *task_slot;
    VectorTaskType task_type;
    // Copied task data (one of the following unions)
    union {
        IndexBuildTaskData build_task;
        SegmentUpdateTaskData update_task;
        IndexLoadTaskData load_task;
        InternalUpgradeTaskData upgrade_task;
    } task_data;
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

/* Merge thread pool — dedicated pool for long-running merge/rebuild tasks.
 * Threads wake on a condition variable, scan FlushedSegmentPool themselves,
 * and update it directly without going through the ring buffer. */
typedef struct MergeThreadPool {
    pthread_t threads[MERGE_WORKERS_COUNT];
    pthread_mutex_t mutex;
    pthread_cond_t work_available;
    atomic_int pending_signals;
    atomic_int shutdown;
    int num_threads;
} MergeThreadPool;

static MergeThreadPool *merge_pool = NULL;

static void submit_internal_task(VectorTaskType task_type, void *data, size_t data_size);

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
            PGPROC *client = NULL;
            
            switch (task->task_type) {
            case IndexBuildTaskType:
            {
                fprintf(stderr, "[maintenance_worker] IndexBuildTaskType case entered\n");
                IndexBuildTaskData *build_task = &task->task_data.build_task;
                                
                Oid index_relid = build_task->index_relid;
                int lsm_idx = build_task->lsm_idx;
                
                // initialize flushed segments
                FlushedSegmentPool *pool_seg = get_flushed_segment_pool(lsm_idx);
                initialize_segment_pool(pool_seg);
                uint32_t seg_idx = reserve_flushed_segment(pool_seg);
                if(seg_idx == -1) {
                    fprintf(stderr, "[maintenance_worker] ERROR: no free flushed segment slot\n");
                } else {
                    load_and_set_segment(index_relid, seg_idx, &pool_seg->flushed_segments[seg_idx], START_SEGMENT_ID, START_SEGMENT_ID, LOAD_LATEST_VERSION, false);
                    register_flushed_segment(pool_seg, seg_idx);
                }
                
                client = &ProcGlobal->allProcs[build_task->backend_pgprocno];
                break;
            }
            case SegmentUpdateTaskType:
            {
                fprintf(stderr, "[maintenance_worker] SegmentUpdateTaskType\n");
                SegmentUpdateTaskData *update_task = &task->task_data.update_task;
                
                Oid index_relid = update_task->index_relid;
                int lsm_idx = update_task->lsm_idx;
                
                // Handle different types of segment update operations
                switch (update_task->operation_type)
                {
                    case SEGMENT_UPDATE_REGULAR:
                    {
                        FlushedSegmentPool *pool_seg = get_flushed_segment_pool(lsm_idx);
                        
                        pthread_rwlock_wrlock(&pool_seg->seg_lock);
                        uint32_t seg_idx = reserve_flushed_segment(pool_seg);
                        pthread_rwlock_unlock(&pool_seg->seg_lock);
                        
                        load_and_set_segment(index_relid, seg_idx, &pool_seg->flushed_segments[seg_idx], update_task->start_sid, update_task->end_sid, LOAD_LATEST_VERSION, false);
                        
                        pthread_rwlock_wrlock(&pool_seg->seg_lock);
                        register_flushed_segment(pool_seg, seg_idx);
                        pthread_rwlock_unlock(&pool_seg->seg_lock);
                        signal_merge_pool();  /* NEW: new segment fully loaded, trigger merge scan */
                        break;
                    }

                    case SEGMENT_UPDATE_VACUUM:
                    {
                        fprintf(stderr, "[maintenance_worker] SEGMENT_UPDATE_VACUUM %d-%d expected_version=%u\n",
                                update_task->start_sid, update_task->end_sid, update_task->expected_version);

                        FlushedSegmentPool *pool_seg = get_flushed_segment_pool(lsm_idx);

                        /* Step 1: find segment by exact (start_sid, end_sid) under read lock */
                        pthread_rwlock_rdlock(&pool_seg->seg_lock);
                        uint32_t seg_idx = find_segment_by_sids(pool_seg,
                                               update_task->start_sid, update_task->end_sid);
                        if (seg_idx != (uint32_t)-1)
                            atomic_fetch_add(&pool_seg->flushed_segments[seg_idx].ref_count, 1);
                        pthread_rwlock_unlock(&pool_seg->seg_lock);

                        if (seg_idx == (uint32_t)-1)
                        {
                            /* Segment merged away → tell backend to retry with new segment */
                            vs_search_result_at(update_task->backend_pgprocno)->maint_status = 1; /* RETRY */
                            client = &ProcGlobal->allProcs[update_task->backend_pgprocno];
                            break;
                        }

                        FlushedSegmentData *seg = &pool_seg->flushed_segments[seg_idx];

                        /* Step 2: acquire per_seg_mutex, check version */
                        pthread_mutex_lock(&seg->per_seg_mutex);

                        int retry = 0;
                        if (seg->version != update_task->expected_version || !seg->in_used)
                        {
                            retry = 1;  /* version changed (rebuild happened) or segment gone */
                        }
                        else
                        {
                            /* Load latest bitmap subversion and OR into in-memory bitmap */
                            uint8_t *new_bitmap = NULL;
                            uint32_t new_delete_count;
                            load_bitmap_file(index_relid, update_task->start_sid, update_task->end_sid,
                                             LOAD_LATEST_VERSION, &new_bitmap, false, &new_delete_count);
                            if (new_bitmap != NULL)
                            {
                                Size bitmap_size = GET_BITMAP_SIZE(seg->vec_count);
                                for (Size bi = 0; bi < bitmap_size; bi++)
                                    seg->bitmap_ptr[bi] |= new_bitmap[bi];
                                seg->delete_count = new_delete_count;
                                free(new_bitmap);
                            }
                        }

                        pthread_mutex_unlock(&seg->per_seg_mutex);

                        /* Step 3: decrement ref count; write result; signal backend */
                        decrement_flushed_segment_ref_count(pool_seg, seg_idx);

                        vs_search_result_at(update_task->backend_pgprocno)->maint_status = retry;
                        client = &ProcGlobal->allProcs[update_task->backend_pgprocno];

                        fprintf(stderr, "[maintenance_worker] SEGMENT_UPDATE_VACUUM %s for %d-%d\n",
                                retry ? "RETRY" : "OK", update_task->start_sid, update_task->end_sid);
                        break;
                    }
                    
                    default:
                        fprintf(stderr, "[maintenance_worker] WARNING: Unknown segment update operation type %d\n", update_task->operation_type);
                        break;
                }
                client = &ProcGlobal->allProcs[update_task->backend_pgprocno];
                break;
            }
            case IndexLoadTaskType:
            {
                IndexLoadTaskData *load_task = &task->task_data.load_task;
                Oid index_relid = load_task->index_relid;
                int lsm_idx = load_task->lsm_idx;
                FlushedSegmentPool *pool_seg;

#if ENABLE_MMAP_COLDSTART
                {
                    uint32_t seg_idx;

                    // TODO: for debugging
                    fprintf(stderr, "[maintenance_worker] IndexLoadTaskType: starting 2-phase mmap cold-start"
                            " for index %u lsm_idx=%d\n", index_relid, lsm_idx);

                    pool_seg = get_flushed_segment_pool(lsm_idx);
                    initialize_segment_pool(pool_seg);

                    // Phase 1: mmap-load all segments (fast) so backends can search immediately.
                    load_all_segments_from_disk_mmap(index_relid, pool_seg);

                    // Signal backend — segments are now searchable via mmap.
                    // TODO: for debugging
                    fprintf(stderr, "[maintenance_worker] IndexLoadTaskType: phase 1 complete,"
                            " signaling backend pgprocno=%d\n", load_task->backend_pgprocno);
                    SetLatch(&ProcGlobal->allProcs[load_task->backend_pgprocno].procLatch);
                    client = NULL;  // suppress the post-switch SetLatch
                    signal_merge_pool();  /* mmap-loaded segments are now searchable */

                    // Phase 2: submit one upgrade task per mmap-loaded segment.
                    {
                        int upgrade_count = 0;
                        pthread_rwlock_rdlock(&pool_seg->seg_lock);
                        seg_idx = pool_seg->head_idx;
                        while (seg_idx != (uint32_t)-1) {
                            FlushedSegment seg = &pool_seg->flushed_segments[seg_idx];
                            if (atomic_load(&seg->load_state) == (int)SEG_MMAP_LOADED) {
                                InternalUpgradeTaskData upg;
                                upg.index_relid  = index_relid;
                                upg.lsm_idx      = lsm_idx;
                                upg.segment_idx  = seg_idx;
                                upg.start_sid    = seg->segment_id_start;
                                upg.end_sid      = seg->segment_id_end;
                                upg.version      = LOAD_LATEST_VERSION;
                                upg.index_type   = seg->index_type;
                                submit_internal_task(InternalSegmentUpgradeTaskType,
                                                     &upg, sizeof(upg));
                                upgrade_count++;
                            }
                            seg_idx = seg->next_idx;
                        }
                        pthread_rwlock_unlock(&pool_seg->seg_lock);
                        // TODO: for debugging
                        fprintf(stderr, "[maintenance_worker] IndexLoadTaskType: phase 2 submitted"
                                " %d upgrade tasks\n", upgrade_count);
                    }
                }
#else
                // TODO: for debugging
                fprintf(stderr, "[maintenance_worker] IndexLoadTaskType: fully loading index %u lsm_idx=%d"
                        " (mmap cold-start disabled)\n", index_relid, lsm_idx);

                pool_seg = get_flushed_segment_pool(lsm_idx);
                initialize_segment_pool(pool_seg);
                load_all_segments_from_disk(index_relid, pool_seg);
                signal_merge_pool();  /* segments fully loaded, trigger merge scan */

                // TODO: for debugging
                fprintf(stderr, "[maintenance_worker] IndexLoadTaskType: full load complete,"
                        " signaling backend pgprocno=%d\n", load_task->backend_pgprocno);
                client = &ProcGlobal->allProcs[load_task->backend_pgprocno];
#endif
                break;
            }
            case InternalSegmentUpgradeTaskType:
            {
                InternalUpgradeTaskData *upg = &task->task_data.upgrade_task;
                FlushedSegmentPool *pool_seg = get_flushed_segment_pool(upg->lsm_idx);
                void *new_index_ptr = NULL;
                uint32_t new_slot_idx;
                FlushedSegment seg;
                Size bitmap_size;
                Size map_size;
                uint8_t *new_bitmap;
                int64_t *new_map;
                FlushedSegmentData *ns;

                fprintf(stderr, "[maintenance_worker] InternalSegmentUpgrade: upgrading"
                        " segment %u-%u (slot=%u) to full in-memory\n",
                        upg->start_sid, upg->end_sid, upg->segment_idx);

                load_index_file(upg->index_relid, upg->start_sid, upg->end_sid,
                                upg->version, upg->index_type, &new_index_ptr, false);

                if (new_index_ptr == NULL) {
                    fprintf(stderr,
                            "[maintenance_worker] InternalSegmentUpgrade: failed to load"
                            " index for segment %u-%u\n",
                            upg->start_sid, upg->end_sid);
                    break;
                }

                pthread_rwlock_wrlock(&pool_seg->seg_lock);
                seg = &pool_seg->flushed_segments[upg->segment_idx];

                if (seg->in_used &&
                    seg->segment_id_start == upg->start_sid &&
                    seg->segment_id_end   == upg->end_sid   &&
                    atomic_load(&seg->load_state) == (int)SEG_MMAP_LOADED)
                {
                    new_slot_idx = reserve_flushed_segment(pool_seg);
                    if (new_slot_idx != (uint32_t)-1)
                    {
                        bitmap_size = GET_BITMAP_SIZE(seg->vec_count);
                        map_size    = sizeof(int64_t) * seg->vec_count;
                        new_bitmap  = (uint8_t *) malloc(bitmap_size);
                        new_map     = (int64_t *) malloc(map_size);

                        if (new_bitmap != NULL && new_map != NULL)
                        {
                            memcpy(new_bitmap, seg->bitmap_ptr, bitmap_size);
                            memcpy(new_map,    seg->map_ptr,    map_size);

                            ns = &pool_seg->flushed_segments[new_slot_idx];
                            ns->segment_id_start = seg->segment_id_start;
                            ns->segment_id_end   = seg->segment_id_end;
                            ns->vec_count        = seg->vec_count;
                            ns->index_type       = seg->index_type;
                            ns->delete_count     = seg->delete_count;
                            ns->version          = seg->version;
                            ns->is_compacting    = false;
                            ns->index_ptr        = new_index_ptr;
                            ns->bitmap_ptr       = new_bitmap;
                            ns->map_ptr          = new_map;
                            atomic_store(&ns->load_state, (int)SEG_FULLY_LOADED);
                            /* Old slot keeps its original bitmap_ptr/map_ptr/index_ptr.
                             * cleanup_flushed_segment frees all three when old slot's
                             * ref_count reaches 0 (i.e. once all in-flight mmap searches
                             * finish).  New slot owns its independent deep copies. */
                            replace_flushed_segment(pool_seg, upg->segment_idx,
                                                    (uint32_t)-1, new_slot_idx);
                            pthread_rwlock_unlock(&pool_seg->seg_lock);

                            signal_merge_pool();
                            fprintf(stderr,
                                    "[maintenance_worker] InternalSegmentUpgrade: upgraded"
                                    " segment %u-%u to full in-memory\n",
                                    upg->start_sid, upg->end_sid);
                        }
                        else
                        {
                            free(new_bitmap);
                            free(new_map);
                            pool_seg->flushed_segments[new_slot_idx].in_used = false;
                            pthread_rwlock_unlock(&pool_seg->seg_lock);
                            IndexFree(new_index_ptr);
                            fprintf(stderr,
                                    "[maintenance_worker] InternalSegmentUpgrade: malloc"
                                    " failed for segment %u-%u\n",
                                    upg->start_sid, upg->end_sid);
                        }
                    }
                    else
                    {
                        pthread_rwlock_unlock(&pool_seg->seg_lock);
                        IndexFree(new_index_ptr);
                        fprintf(stderr,
                                "[maintenance_worker] InternalSegmentUpgrade: no free"
                                " slot for segment %u-%u, discarding\n",
                                upg->start_sid, upg->end_sid);
                    }
                }
                else
                {
                    pthread_rwlock_unlock(&pool_seg->seg_lock);
                    IndexFree(new_index_ptr);
                    fprintf(stderr,
                            "[maintenance_worker] InternalSegmentUpgrade: segment"
                            " %u-%u no longer valid, discarding\n",
                            upg->start_sid, upg->end_sid);
                }

                client = NULL;
                break;
            }
            default:
                fprintf(stderr, "[maintenance_worker] Unknown task type %d\n", task->task_type);
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

static void
init_merge_thread_pool(void)
{
    if (merge_pool != NULL)
        return;

    merge_pool = (MergeThreadPool *) malloc(sizeof(MergeThreadPool));
    if (merge_pool == NULL)
        elog(ERROR, "[init_merge_thread_pool] malloc failed");

    pthread_mutex_init(&merge_pool->mutex, NULL);
    pthread_cond_init(&merge_pool->work_available, NULL);
    atomic_init(&merge_pool->pending_signals, 0);
    atomic_init(&merge_pool->shutdown, 0);
    merge_pool->num_threads = MERGE_WORKERS_COUNT;

    for (int i = 0; i < merge_pool->num_threads; i++)
    {
        if (pthread_create(&merge_pool->threads[i], NULL,
                           merge_worker_thread, merge_pool) != 0)
            elog(ERROR, "[init_merge_thread_pool] failed to create merge thread %d", i);
    }

    elog(DEBUG1, "[init_merge_thread_pool] started %d merge threads", merge_pool->num_threads);
}

static void
shutdown_merge_thread_pool(void)
{
    if (merge_pool == NULL)
        return;

    atomic_store(&merge_pool->shutdown, 1);
    pthread_mutex_lock(&merge_pool->mutex);
    pthread_cond_broadcast(&merge_pool->work_available);
    pthread_mutex_unlock(&merge_pool->mutex);

    for (int i = 0; i < merge_pool->num_threads; i++)
        pthread_join(merge_pool->threads[i], NULL);

    pthread_mutex_destroy(&merge_pool->mutex);
    pthread_cond_destroy(&merge_pool->work_available);
    free(merge_pool);
    merge_pool = NULL;
    elog(DEBUG1, "[shutdown_merge_thread_pool] merge threads joined and cleaned up");
}

static void
signal_merge_pool(void)
{
    if (merge_pool == NULL)
        return;
    pthread_mutex_lock(&merge_pool->mutex);
    atomic_fetch_add(&merge_pool->pending_signals, 1);
    pthread_cond_signal(&merge_pool->work_available);
    pthread_mutex_unlock(&merge_pool->mutex);
}

/* ----------------------------------------------------------------
 * Merge scheduling helpers (Task 5)
 * Ported from lsm_merge_worker.c; operate on FlushedSegmentPool
 * (process-local, pthread-locked) instead of SharedSegmentArray.
 * ---------------------------------------------------------------- */

typedef struct {
    int operation_type;        /* SEGMENT_UPDATE_REBUILD_FLAT/DELETION/MERGE */
    int lsm_idx;
    Oid index_relid;
    uint32_t segment_idx0;
    uint32_t segment_idx1;     /* (uint32_t)-1 for rebuild ops */
    SegmentId merged_start_sid;
    SegmentId merged_end_sid;
    uint32_t merged_vec_count;
    IndexType merged_index_type;
    uint32_t merged_delete_count;
} MergeTaskLocal;

/*
 * Choose a merge partner for the segment at `idx`.
 * Prefers the smaller of the two neighbours. Skips segments that are
 * DISKANN, already compacting, or not fully loaded.
 * Must be called with pool->seg_lock held (read or write).
 */
static bool
choose_adjacent_smaller_pool(FlushedSegmentPool *pool, uint32_t idx,
                              uint32_t *chosen_adj_idx)
{
    uint32_t prev = pool->flushed_segments[idx].prev_idx;
    uint32_t next = pool->flushed_segments[idx].next_idx;

#define CANDIDATE_OK(i) \
    ((i) != (uint32_t)-1 && \
     pool->flushed_segments[(i)].in_used && \
     !pool->flushed_segments[(i)].is_compacting && \
     pool->flushed_segments[(i)].index_type != DISKANN && \
     atomic_load(&pool->flushed_segments[(i)].load_state) == (int)SEG_FULLY_LOADED)

    bool have_prev = CANDIDATE_OK(prev);
    bool have_next = CANDIDATE_OK(next);
#undef CANDIDATE_OK

    if (!have_prev && !have_next) return false;
    if (have_prev && !have_next)  { *chosen_adj_idx = prev; return true; }
    if (!have_prev && have_next)  { *chosen_adj_idx = next; return true; }

    *chosen_adj_idx = (pool->flushed_segments[prev].vec_count <=
                       pool->flushed_segments[next].vec_count) ? prev : next;
    return true;
}

static bool
claim_merge_task_pool(int lsm_idx, uint32_t segment_idx0, uint32_t segment_idx1,
                      int task_type, MergeTaskLocal *task)
{
    FlushedSegmentPool *pool = get_flushed_segment_pool(lsm_idx);

    pthread_rwlock_wrlock(&pool->seg_lock);

    /* Validate segment 0 */
    if (!pool->flushed_segments[segment_idx0].in_used ||
        pool->flushed_segments[segment_idx0].is_compacting ||
        atomic_load(&pool->flushed_segments[segment_idx0].load_state) != (int)SEG_FULLY_LOADED)
    {
        pthread_rwlock_unlock(&pool->seg_lock);
        return false;
    }

    /* Validate segment 1 for merge ops */
    if (task_type == SEGMENT_UPDATE_MERGE)
    {
        if (segment_idx1 == (uint32_t)-1 ||
            !pool->flushed_segments[segment_idx1].in_used ||
            pool->flushed_segments[segment_idx1].is_compacting ||
            atomic_load(&pool->flushed_segments[segment_idx1].load_state) != (int)SEG_FULLY_LOADED)
        {
            pthread_rwlock_unlock(&pool->seg_lock);
            return false;
        }
    }

    /* Claim */
    pool->flushed_segments[segment_idx0].is_compacting = true;
    if (task_type == SEGMENT_UPDATE_MERGE && segment_idx1 != (uint32_t)-1)
        pool->flushed_segments[segment_idx1].is_compacting = true;

    /* Fill task */
    task->operation_type    = task_type;
    task->lsm_idx           = lsm_idx;
    task->index_relid       = SharedLSMIndexBuffer->slots[lsm_idx].lsmIndex.indexRelId;
    task->segment_idx0      = segment_idx0;
    task->segment_idx1      = segment_idx1;
    task->merged_start_sid  = pool->flushed_segments[segment_idx0].segment_id_start;
    task->merged_end_sid    = (task_type == SEGMENT_UPDATE_MERGE)
                              ? pool->flushed_segments[segment_idx1].segment_id_end
                              : pool->flushed_segments[segment_idx0].segment_id_end;

    pthread_rwlock_unlock(&pool->seg_lock);
    return true;
}

static bool
traverse_and_check_priority_pool(int lsm_idx, int priority_type, MergeTaskLocal *task)
{
    FlushedSegmentPool *pool = get_flushed_segment_pool(lsm_idx);

    pthread_rwlock_rdlock(&pool->seg_lock);

    if (pool->head_idx == (uint32_t)-1 ||
        !pool->flushed_segments[pool->head_idx].in_used)
    {
        pthread_rwlock_unlock(&pool->seg_lock);
        return false;
    }

    uint32_t cur = pool->head_idx;
    while (cur != (uint32_t)-1)
    {
        FlushedSegmentData *seg = &pool->flushed_segments[cur];

        if (!seg->in_used || seg->is_compacting ||
            atomic_load(&seg->load_state) != (int)SEG_FULLY_LOADED)
        {
            cur = seg->next_idx;
            continue;
        }

        bool should_claim = false;
        uint32_t adj = (uint32_t)-1;

        switch (priority_type)
        {
            case 1: /* FLAT → rebuild flat */
                should_claim = (seg->index_type == FLAT);
                break;
            case 2: /* vec_count <= MEMTABLE_MAX_CAPACITY → merge */
                if (seg->index_type == DISKANN) break;
                if (seg->vec_count <= MEMTABLE_MAX_CAPACITY &&
                    choose_adjacent_smaller_pool(pool, cur, &adj) &&
                    pool->flushed_segments[adj].vec_count < MAX_SEGMENTS_SIZE &&
                    seg->vec_count + pool->flushed_segments[adj].vec_count < MAX_SEGMENTS_SIZE)
                    should_claim = true;
                break;
            case 3: /* high deletion ratio → rebuild deletion */
                if (seg->vec_count > 0 &&
                    (float)seg->delete_count / (float)seg->vec_count > MERGE_DELETION_RATIO_THRESHOLD)
                    should_claim = true;
                break;
            case 4: /* vec_count <= THRESHOLD_SMALL_SEGMENT_SIZE → merge */
                if (seg->index_type == DISKANN) break;
                if (seg->vec_count <= THRESHOLD_SMALL_SEGMENT_SIZE &&
                    choose_adjacent_smaller_pool(pool, cur, &adj) &&
                    pool->flushed_segments[adj].vec_count < MAX_SEGMENTS_SIZE &&
                    seg->vec_count + pool->flushed_segments[adj].vec_count < MAX_SEGMENTS_SIZE)
                    should_claim = true;
                break;
            case 5: /* vec_count < MAX_SEGMENTS_SIZE → merge */
                if (seg->index_type == DISKANN) break;
                if (seg->vec_count < MAX_SEGMENTS_SIZE &&
                    choose_adjacent_smaller_pool(pool, cur, &adj) &&
                    pool->flushed_segments[adj].vec_count < MAX_SEGMENTS_SIZE &&
                    seg->vec_count + pool->flushed_segments[adj].vec_count < MAX_SEGMENTS_SIZE)
                    should_claim = true;
                break;
        }

        if (should_claim)
        {
            /*
             * Order by segment_id_start (not pool array index) while still
             * holding the read lock so the IDs are stable.  Pool slots are
             * reused after merges, so Min/Max of pool indices can place the
             * higher-SID segment first, producing an underflowing
             * merged_seg_count and corrupting replace_flushed_segment's
             * linked-list surgery.
             */
            bool adj_lower;
            uint32_t lo, hi;
            int task_type;

            adj_lower = (adj != (uint32_t)-1) &&
                        pool->flushed_segments[adj].segment_id_start <
                        pool->flushed_segments[cur].segment_id_start;
            lo = adj_lower ? adj : cur;
            hi = (adj == (uint32_t)-1) ? (uint32_t)-1 : (adj_lower ? cur : adj);

            pthread_rwlock_unlock(&pool->seg_lock);

            task_type = (priority_type == 1) ? SEGMENT_UPDATE_REBUILD_FLAT :
                        (priority_type == 3) ? SEGMENT_UPDATE_REBUILD_DELETION :
                                               SEGMENT_UPDATE_MERGE;

            if (claim_merge_task_pool(lsm_idx, lo, hi, task_type, task))
                return true;

            /* Claim failed — another thread raced us; restart scan from head */
            pthread_rwlock_rdlock(&pool->seg_lock);
            cur = pool->head_idx;
            continue;
        }

        cur = seg->next_idx;
    }

    pthread_rwlock_unlock(&pool->seg_lock);
    return false;
}

static bool
scan_and_claim_merge_task_pool(MergeTaskLocal *task)
{
    /* Priority 1: FLAT segments → rebuild */
    for (int lsm_idx = 0; lsm_idx < INDEX_BUF_SIZE; lsm_idx++)
    {
        if (!pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[lsm_idx].valid))
            continue;
        FlushedSegmentPool *pool = get_flushed_segment_pool(lsm_idx);
        if (pg_atomic_read_u32(&pool->flat_count) > 0)
            if (traverse_and_check_priority_pool(lsm_idx, 1, task)) return true;
    }
    /* Priority 2: small segments (≤ memtable capacity) → merge */
    for (int lsm_idx = 0; lsm_idx < INDEX_BUF_SIZE; lsm_idx++)
    {
        if (!pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[lsm_idx].valid))
            continue;
        FlushedSegmentPool *pool = get_flushed_segment_pool(lsm_idx);
        if (pg_atomic_read_u32(&pool->memtable_capacity_le_count) > 0)
            if (traverse_and_check_priority_pool(lsm_idx, 2, task)) return true;
    }
    /* Priority 3: high deletion ratio → rebuild */
    for (int lsm_idx = 0; lsm_idx < INDEX_BUF_SIZE; lsm_idx++)
    {
        if (!pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[lsm_idx].valid))
            continue;
        if (traverse_and_check_priority_pool(lsm_idx, 3, task)) return true;
    }
    /* Priority 4: small segments (≤ threshold) → merge */
    for (int lsm_idx = 0; lsm_idx < INDEX_BUF_SIZE; lsm_idx++)
    {
        if (!pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[lsm_idx].valid))
            continue;
        FlushedSegmentPool *pool = get_flushed_segment_pool(lsm_idx);
        if (pg_atomic_read_u32(&pool->small_segment_le_count) > 0)
            if (traverse_and_check_priority_pool(lsm_idx, 4, task)) return true;
    }
    /* Priority 5: any segment pair → merge */
    for (int lsm_idx = 0; lsm_idx < INDEX_BUF_SIZE; lsm_idx++)
    {
        if (!pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[lsm_idx].valid))
            continue;
        if (traverse_and_check_priority_pool(lsm_idx, 5, task)) return true;
    }
    return false;
}

/* ----------------------------------------------------------------
 * End of merge scheduling helpers (Task 5)
 * ---------------------------------------------------------------- */

/*
 * Rebuild a single segment's index, filtering deleted vectors.
 * Updates FlushedSegmentPool directly on completion.
 * Called from merge threads — uses malloc, not palloc.
 * The segment must already be claimed (is_compacting = true) via claim_merge_task_pool.
 */
static void
rebuild_index_pool(MergeTaskLocal *task)
{
    /* All declarations up front — required for C99 compatibility (-Wdeclaration-after-statement). */
    FlushedSegmentPool    *pool;
    FlushedSegmentData    *seg;
    Oid                    index_relid;
    SegmentId              start_sid;
    SegmentId              end_sid;
    uint32_t               version;
    IndexType              target_type;
    void                  *old_index_ptr;
    uint8_t               *snapshot_bitmap;
    uint32_t               delete_count;
    int64_t               *old_mapping_ptr;
    SegmentOffsetRange    *old_offsets_ptr;
    void                  *new_index_ptr;
    int                    new_index_count;
    int                    M, efConstruction, lists;
    uint8_t               *fresh_bitmap;
    uint32_t               fresh_delete_count;
    void                  *new_index_bin;
    uint32_t               segment_count;
    Size                   new_map_size;
    int64_t               *new_mapping;
    uint8_t               *new_bitmap;
    SegmentOffsetRange    *new_offsets;
    uint32_t               new_delete_count;
    uint32_t               new_slot_idx;
    uint32_t               new_version;
    PrepareFlushMetaData   prep;
    Size                   prev_end;

    pool        = get_flushed_segment_pool(task->lsm_idx);
    seg         = &pool->flushed_segments[task->segment_idx0];
    index_relid = task->index_relid;
    start_sid   = seg->segment_id_start;
    end_sid     = seg->segment_id_end;

    version     = find_latest_segment_version(index_relid, start_sid, end_sid);
    target_type = SharedLSMIndexBuffer->slots[task->lsm_idx].lsmIndex.index_type;

    /* --- Phase 1: load data from disk (no lock) --- */
    old_index_ptr    = NULL;
    load_index_file(index_relid, start_sid, end_sid, version, seg->index_type,
                    &old_index_ptr, false);

    snapshot_bitmap  = NULL;
    load_bitmap_file(index_relid, start_sid, end_sid, version,
                     &snapshot_bitmap, false, &delete_count);

    old_mapping_ptr  = NULL;
    load_mapping_file(index_relid, start_sid, end_sid, version, &old_mapping_ptr, false);

    old_offsets_ptr  = NULL;
    load_offset_file(index_relid, start_sid, end_sid, version, &old_offsets_ptr, false);

    /* --- Phase 2: build new index from snapshot bitmap (no lock) --- */
    new_index_ptr   = NULL;
    new_index_count = 0;
    M = 32; efConstruction = 400; lists = 1024;
    MergeIndex(old_index_ptr, snapshot_bitmap, (int)seg->vec_count,
               seg->index_type, target_type,
               &new_index_ptr, &new_index_count,
               M, efConstruction, lists);

    /* --- Phase 3: acquire per_seg_mutex, reload bitmap to capture concurrent vacuum --- */
    pthread_mutex_lock(&seg->per_seg_mutex);

    fresh_bitmap = NULL;
    load_bitmap_file(index_relid, start_sid, end_sid, version,
                     &fresh_bitmap, false, &fresh_delete_count);

    /* Serialize new index */
    new_index_bin = NULL;
    IndexSerialize(new_index_ptr, &new_index_bin);

    /* Build new mapping and bitmap */
    segment_count   = end_sid - start_sid + 1;
    new_map_size    = sizeof(int64_t) * (Size)new_index_count;
    new_mapping     = (int64_t *) malloc(new_map_size);
    new_bitmap      = (uint8_t *) calloc(1, GET_BITMAP_SIZE(new_index_count));
    new_offsets     = (SegmentOffsetRange *) calloc(segment_count, sizeof(SegmentOffsetRange));

    if (new_mapping == NULL || new_bitmap == NULL || new_offsets == NULL)
    {
        fprintf(stderr, "[rebuild_index_pool] allocation failed for segment %u-%u\n",
                start_sid, end_sid);
        free(new_mapping);
        free(new_bitmap);
        free(new_offsets);
        free(fresh_bitmap);
        free(old_mapping_ptr);
        free(old_offsets_ptr);
        free(snapshot_bitmap);
        IndexFree(old_index_ptr);
        IndexFree(new_index_ptr);
        /* Release locks before returning */
        pthread_mutex_unlock(&seg->per_seg_mutex);
        /* Clear is_compacting under seg_lock write */
        pthread_rwlock_wrlock(&pool->seg_lock);
        seg->is_compacting = false;
        pthread_rwlock_unlock(&pool->seg_lock);
        return;
    }

    new_delete_count = 0;

    /* Initialise offset sentinels */
    for (uint32_t j = 0; j < segment_count; j++)
    {
        new_offsets[j].sid          = old_offsets_ptr[j].sid;
        new_offsets[j].start_offset = (Size)-1;  /* SIZE_MAX sentinel */
        new_offsets[j].end_offset   = 0;
    }

    {
        int write_idx    = 0;
        uint32_t cur_sid_idx = 0;
        for (int i = 0; i < (int)seg->vec_count; i++)
        {
            /* Advance offset window */
            while (cur_sid_idx < segment_count)
            {
                if (old_offsets_ptr[cur_sid_idx].start_offset ==
                    old_offsets_ptr[cur_sid_idx].end_offset)
                    cur_sid_idx++;
                else if (i >= (int)old_offsets_ptr[cur_sid_idx].end_offset)
                    cur_sid_idx++;
                else
                    break;
            }

            if (!IS_SLOT_SET(snapshot_bitmap, i))  /* not deleted in original snapshot */
            {
                new_mapping[write_idx] = old_mapping_ptr[i];

                /* Apply vacuum deletions from fresh reload */
                if (IS_SLOT_SET(fresh_bitmap, i))
                {
                    SET_SLOT(new_bitmap, write_idx);
                    new_delete_count++;
                }

                /* Update offset range */
                if (cur_sid_idx < segment_count &&
                    i >= (int)old_offsets_ptr[cur_sid_idx].start_offset &&
                    i <  (int)old_offsets_ptr[cur_sid_idx].end_offset)
                {
                    if (new_offsets[cur_sid_idx].start_offset == (Size)-1)
                        new_offsets[cur_sid_idx].start_offset = (Size)write_idx;
                    new_offsets[cur_sid_idx].end_offset = (Size)(write_idx + 1);
                }
                write_idx++;
            }
        }
        /* write_idx should equal new_index_count */
    }

    /* Fix empty offset ranges */
    prev_end = 0;
    for (uint32_t j = 0; j < segment_count; j++)
    {
        if (new_offsets[j].start_offset == (Size)-1)
            new_offsets[j].start_offset = new_offsets[j].end_offset = prev_end;
        else
            prev_end = new_offsets[j].end_offset;
    }

    free(snapshot_bitmap);

    /* Write new segment to disk */
    prep.start_sid    = start_sid;
    prep.end_sid      = end_sid;
    prep.valid_rows   = (uint32_t)new_index_count;
    prep.index_type   = target_type;
    prep.index_bin    = new_index_bin;
    prep.bitmap_ptr   = new_bitmap;
    prep.bitmap_size  = GET_BITMAP_SIZE(new_index_count);
    prep.delete_count = new_delete_count;
    prep.map_ptr      = new_mapping;
    prep.map_size     = new_map_size;
    prep.offsets      = new_offsets;
    flush_segment_to_disk(index_relid, &prep);
    /* flush_segment_to_disk takes ownership of index_bin; do not free separately */

    new_version = find_latest_segment_version(index_relid, start_sid, end_sid);

    /* --- Phase 4: reserve new slot, populate, replace old slot.
     * replace_flushed_segment decrements old slot's ref_count; its index/bitmap/map
     * are freed by cleanup_flushed_segment when the last in-flight search finishes. */

    pthread_rwlock_wrlock(&pool->seg_lock);
    new_slot_idx = reserve_flushed_segment(pool);
    pthread_rwlock_unlock(&pool->seg_lock);

    if (new_slot_idx == (uint32_t)-1)
    {
        fprintf(stderr, "[rebuild_index_pool] no free slot for rebuilt segment %u-%u\n",
                start_sid, end_sid);
        IndexFree(new_index_ptr);
        free(new_bitmap);
        free(new_mapping);
        free(new_offsets);
        free(fresh_bitmap);
        IndexFree(old_index_ptr);
        free(old_mapping_ptr);
        free(old_offsets_ptr);
        pthread_mutex_unlock(&seg->per_seg_mutex);
        pthread_rwlock_wrlock(&pool->seg_lock);
        seg->is_compacting = false;
        pthread_rwlock_unlock(&pool->seg_lock);
        return;
    }

    {
        FlushedSegmentData *ns = &pool->flushed_segments[new_slot_idx];
        ns->segment_id_start = start_sid;
        ns->segment_id_end   = end_sid;
        ns->index_ptr        = new_index_ptr;
        ns->bitmap_ptr       = new_bitmap;
        ns->map_ptr          = new_mapping;
        ns->vec_count        = (Size)new_index_count;
        ns->index_type       = target_type;
        ns->delete_count     = new_delete_count;
        ns->version          = new_version;
        ns->is_compacting    = false;
        atomic_store(&ns->load_state, (int)SEG_FULLY_LOADED);
    }

    pthread_rwlock_wrlock(&pool->seg_lock);
    replace_flushed_segment(pool, task->segment_idx0, (uint32_t)-1, new_slot_idx);
    if (target_type == FLAT)
        pg_atomic_fetch_add_u32(&pool->flat_count, 1);
    if ((uint32_t)new_index_count <= MEMTABLE_MAX_CAPACITY)
        pg_atomic_fetch_add_u32(&pool->memtable_capacity_le_count, 1);
    if ((uint32_t)new_index_count <= THRESHOLD_SMALL_SEGMENT_SIZE)
        pg_atomic_fetch_add_u32(&pool->small_segment_le_count, 1);
    pthread_rwlock_unlock(&pool->seg_lock);

    pthread_mutex_unlock(&seg->per_seg_mutex);

    /* Free only private-to-rebuild buffers (not shared with any search thread).
     * Old slot's index/bitmap/map are freed by cleanup_flushed_segment. */
    IndexFree(old_index_ptr);
    free(fresh_bitmap);
    free(old_mapping_ptr);
    free(old_offsets_ptr);
    free(new_offsets);

    fprintf(stderr, "[rebuild_index_pool] rebuilt segment %u-%u → %d vectors (v%u)\n",
            start_sid, end_sid, new_index_count, new_version);
}

/*
 * Merge two adjacent segments into one.
 * Updates FlushedSegmentPool directly on completion.
 * Called from merge threads — uses malloc, not palloc.
 * Both segments must already be claimed (is_compacting = true).
 * The two per_seg_mutex locks are acquired in ascending start_sid order
 * to prevent deadlock between concurrent merge threads.
 */
static void
merge_adjacent_segments_pool(MergeTaskLocal *task)
{
    /* --- All declarations hoisted to function top (C89 compliance) --- */
    FlushedSegmentPool    *pool;
    FlushedSegmentData    *seg0, *seg1;
    Oid                    index_relid;
    SegmentId              s0_start, s0_end, s1_start, s1_end;
    uint32_t               version0, version1;
    void                  *index0_ptr, *index1_ptr;
    uint8_t               *bitmap0, *bitmap1;
    uint32_t               dc0, dc1;
    int64_t               *mapping0, *mapping1;
    SegmentOffsetRange    *offsets0, *offsets1;
    void                  *larger_index, *smaller_index;
    int64_t               *larger_mapping, *smaller_mapping;
    SegmentOffsetRange    *larger_offsets, *smaller_offsets;
    uint32_t               larger_sid_count, smaller_sid_count;
    int                    larger_count, smaller_count;
    IndexType              larger_type, smaller_type;
    int                    merged_count;
    void                  *merged_index;
    FlushedSegmentData    *first_lock, *second_lock;
    uint8_t               *fresh_bitmap0, *fresh_bitmap1;
    uint32_t               fresh_dc0, fresh_dc1;
    uint8_t               *fresh_larger_bitmap, *fresh_smaller_bitmap;
    int                    total_count;
    uint32_t               merged_delete_count;
    Size                   merged_bitmap_size;
    uint8_t               *merged_bitmap;
    int64_t               *merged_mapping;
    uint32_t               merged_seg_count;
    SegmentOffsetRange    *merged_offsets;
    void                  *merged_index_bin;
    PrepareFlushMetaData   prep;
    uint32_t               new_version;
    uint32_t               new_slot_idx;
    void                  *smaller_disk_ptr;

    pool        = get_flushed_segment_pool(task->lsm_idx);
    seg0        = &pool->flushed_segments[task->segment_idx0];
    seg1        = &pool->flushed_segments[task->segment_idx1];
    index_relid = task->index_relid;

    s0_start = seg0->segment_id_start;
    s0_end   = seg0->segment_id_end;
    s1_start = seg1->segment_id_start;
    s1_end   = seg1->segment_id_end;

    version0 = find_latest_segment_version(index_relid, s0_start, s0_end);
    version1 = find_latest_segment_version(index_relid, s1_start, s1_end);

    /* --- Phase 1: load both indices, bitmaps, mappings, offsets (no lock) --- */
    index0_ptr = NULL;
    index1_ptr = NULL;
    load_index_file(index_relid, s0_start, s0_end, version0, seg0->index_type, &index0_ptr, false);
    load_index_file(index_relid, s1_start, s1_end, version1, seg1->index_type, &index1_ptr, false);

    bitmap0 = NULL;
    bitmap1 = NULL;
    load_bitmap_file(index_relid, s0_start, s0_end, version0, &bitmap0, false, &dc0);
    load_bitmap_file(index_relid, s1_start, s1_end, version1, &bitmap1, false, &dc1);

    mapping0 = NULL;
    mapping1 = NULL;
    load_mapping_file(index_relid, s0_start, s0_end, version0, &mapping0, false);
    load_mapping_file(index_relid, s1_start, s1_end, version1, &mapping1, false);

    offsets0 = NULL;
    offsets1 = NULL;
    load_offset_file(index_relid, s0_start, s0_end, version0, &offsets0, false);
    load_offset_file(index_relid, s1_start, s1_end, version1, &offsets1, false);

    /* Determine which segment is larger for MergeTwoIndices */
    if (seg0->vec_count >= seg1->vec_count) {
        larger_index      = index0_ptr;    smaller_index   = index1_ptr;
        larger_mapping    = mapping0;      smaller_mapping = mapping1;
        larger_offsets    = offsets0;      smaller_offsets = offsets1;
        larger_sid_count  = s0_end - s0_start + 1;
        smaller_sid_count = s1_end - s1_start + 1;
        larger_count      = (int)seg0->vec_count;  smaller_count = (int)seg1->vec_count;
        larger_type       = seg0->index_type;       smaller_type  = seg1->index_type;
    } else {
        larger_index      = index1_ptr;    smaller_index   = index0_ptr;
        larger_mapping    = mapping1;      smaller_mapping = mapping0;
        larger_offsets    = offsets1;      smaller_offsets = offsets0;
        larger_sid_count  = s1_end - s1_start + 1;
        smaller_sid_count = s0_end - s0_start + 1;
        larger_count      = (int)seg1->vec_count;  smaller_count = (int)seg0->vec_count;
        larger_type       = seg1->index_type;       smaller_type  = seg0->index_type;
    }

    /* Track the smaller disk-loaded pointer for cleanup; the larger becomes merged_index
     * (owned by the pool via seg0->index_ptr) and must NOT be freed in this function. */
    smaller_disk_ptr = (seg0->vec_count >= seg1->vec_count) ? index1_ptr : index0_ptr;

    /* --- Phase 2: merge indices (no lock) --- */
    merged_count = 0;
    merged_index = MergeTwoIndices(larger_index, larger_count, larger_type,
                                   smaller_index, smaller_count, smaller_type,
                                   &merged_count);
    /* MergeTwoIndices returns the larger_index pointer with smaller merged in */

    /* --- Phase 3: acquire per_seg_mutex for both (ascending start_sid order) --- */
    first_lock  = (s0_start <= s1_start) ? seg0 : seg1;
    second_lock = (s0_start <= s1_start) ? seg1 : seg0;
    pthread_mutex_lock(&first_lock->per_seg_mutex);
    pthread_mutex_lock(&second_lock->per_seg_mutex);

    /* Reload bitmaps to capture any vacuum deletions that landed during Phase 2 */
    fresh_bitmap0 = NULL;
    fresh_bitmap1 = NULL;
    load_bitmap_file(index_relid, s0_start, s0_end, version0, &fresh_bitmap0, false, &fresh_dc0);
    load_bitmap_file(index_relid, s1_start, s1_end, version1, &fresh_bitmap1, false, &fresh_dc1);

    /* Re-point larger/smaller fresh bitmaps */
    fresh_larger_bitmap  = (seg0->vec_count >= seg1->vec_count) ? fresh_bitmap0 : fresh_bitmap1;
    fresh_smaller_bitmap = (seg0->vec_count >= seg1->vec_count) ? fresh_bitmap1 : fresh_bitmap0;

    total_count          = larger_count + smaller_count;
    merged_delete_count  = fresh_dc0 + fresh_dc1;

    /* Build merged bitmap */
    merged_bitmap_size = GET_BITMAP_SIZE(total_count);
    merged_bitmap      = (uint8_t *) calloc(1, merged_bitmap_size);

    /* Build merged mapping */
    merged_mapping = (int64_t *) malloc(sizeof(int64_t) * (Size)total_count);

    /* Build merged offsets */
    merged_seg_count = task->merged_end_sid - task->merged_start_sid + 1;
    merged_offsets   =
        (SegmentOffsetRange *) calloc(merged_seg_count, sizeof(SegmentOffsetRange));

    if (merged_bitmap == NULL || merged_mapping == NULL || merged_offsets == NULL)
    {
        fprintf(stderr, "[merge_adjacent_segments_pool] allocation failed for segments %u-%u + %u-%u\n",
                s0_start, s0_end, s1_start, s1_end);
        free(merged_bitmap);
        free(merged_mapping);
        free(merged_offsets);
        free(fresh_bitmap0);
        free(fresh_bitmap1);
        free(bitmap0); free(bitmap1);
        free(mapping0); free(mapping1);
        free(offsets0); free(offsets1);
        IndexFree(smaller_disk_ptr);  /* only free the smaller disk copy; larger is merged_index, not yet owned by pool */
        pthread_mutex_unlock(&second_lock->per_seg_mutex);
        pthread_mutex_unlock(&first_lock->per_seg_mutex);
        pthread_rwlock_wrlock(&pool->seg_lock);
        seg0->is_compacting = false;
        seg1->is_compacting = false;
        pthread_rwlock_unlock(&pool->seg_lock);
        return;
    }

    /* Populate merged bitmap from fresh bitmaps */
    for (int i = 0; i < larger_count; i++)
        if (IS_SLOT_SET(fresh_larger_bitmap, i)) SET_SLOT(merged_bitmap, i);
    for (int i = 0; i < smaller_count; i++)
        if (IS_SLOT_SET(fresh_smaller_bitmap, i)) SET_SLOT(merged_bitmap, larger_count + i);
    /* Clear stray bits in last byte */
    {
        int rem = total_count % 8;
        if (rem) merged_bitmap[merged_bitmap_size - 1] &= (uint8_t)((1 << rem) - 1);
    }

    /* Populate merged mapping: larger vectors first, then smaller */
    for (int i = 0; i < larger_count; i++)
        merged_mapping[i] = larger_mapping[i];
    for (int i = 0; i < smaller_count; i++)
        merged_mapping[larger_count + i] = smaller_mapping[i];

    /* Populate merged offsets: larger offsets first, then smaller (offset-adjusted) */
    for (uint32_t i = 0; i < larger_sid_count; i++)
        merged_offsets[i] = larger_offsets[i];
    for (uint32_t i = 0; i < smaller_sid_count; i++)
    {
        merged_offsets[larger_sid_count + i] = smaller_offsets[i];
        merged_offsets[larger_sid_count + i].start_offset += (Size)larger_count;
        merged_offsets[larger_sid_count + i].end_offset   += (Size)larger_count;
    }

    /* Serialize and flush to disk */
    merged_index_bin = NULL;
    IndexSerialize(merged_index, &merged_index_bin);

    prep.start_sid    = task->merged_start_sid;
    prep.end_sid      = task->merged_end_sid;
    prep.valid_rows   = (uint32_t)merged_count;
    prep.index_type   = larger_type;
    prep.index_bin    = merged_index_bin;
    prep.bitmap_ptr   = merged_bitmap;
    prep.bitmap_size  = merged_bitmap_size;
    prep.delete_count = merged_delete_count;
    prep.map_ptr      = merged_mapping;
    prep.map_size     = sizeof(int64_t) * (Size)merged_count;
    prep.offsets      = merged_offsets;
    flush_segment_to_disk(index_relid, &prep);

    new_version = find_latest_segment_version(index_relid,
                      task->merged_start_sid, task->merged_end_sid);

    /* --- Phase 4: reserve new slot, populate, replace both old slots.
     * replace_flushed_segment decrements both old slots' ref_counts; their
     * index/bitmap/map are freed by cleanup_flushed_segment when the last
     * in-flight search on each slot drops its reference. */

    pthread_rwlock_wrlock(&pool->seg_lock);
    new_slot_idx = reserve_flushed_segment(pool);
    pthread_rwlock_unlock(&pool->seg_lock);

    if (new_slot_idx == (uint32_t)-1)
    {
        fprintf(stderr,
                "[merge_adjacent_segments_pool] no free slot for merged segment %u-%u\n",
                task->merged_start_sid, task->merged_end_sid);
        IndexFree(merged_index);
        free(merged_bitmap);
        free(merged_mapping);
        free(merged_offsets);
        IndexFree(smaller_disk_ptr);
        free(bitmap0); free(bitmap1);
        free(mapping0); free(mapping1);
        free(offsets0); free(offsets1);
        free(fresh_bitmap0); free(fresh_bitmap1);
        pthread_mutex_unlock(&second_lock->per_seg_mutex);
        pthread_mutex_unlock(&first_lock->per_seg_mutex);
        pthread_rwlock_wrlock(&pool->seg_lock);
        seg0->is_compacting = false;
        seg1->is_compacting = false;
        pthread_rwlock_unlock(&pool->seg_lock);
        return;
    }

    {
        FlushedSegmentData *ns = &pool->flushed_segments[new_slot_idx];
        ns->segment_id_start = task->merged_start_sid;
        ns->segment_id_end   = task->merged_end_sid;
        ns->index_ptr        = merged_index;
        ns->bitmap_ptr       = merged_bitmap;
        ns->map_ptr          = merged_mapping;
        ns->vec_count        = (Size)merged_count;
        ns->index_type       = larger_type;
        ns->delete_count     = merged_delete_count;
        ns->version          = new_version;
        ns->is_compacting    = false;
        atomic_store(&ns->load_state, (int)SEG_FULLY_LOADED);
    }

    pthread_rwlock_wrlock(&pool->seg_lock);
    replace_flushed_segment(pool, task->segment_idx0, task->segment_idx1, new_slot_idx);
    if (larger_type == FLAT)
        pg_atomic_fetch_add_u32(&pool->flat_count, 1);
    {
        uint32_t nv = (uint32_t)merged_count;
        if (nv <= MEMTABLE_MAX_CAPACITY)
            pg_atomic_fetch_add_u32(&pool->memtable_capacity_le_count, 1);
        if (nv <= THRESHOLD_SMALL_SEGMENT_SIZE)
            pg_atomic_fetch_add_u32(&pool->small_segment_le_count, 1);
    }
    pthread_rwlock_unlock(&pool->seg_lock);

    pthread_mutex_unlock(&second_lock->per_seg_mutex);
    pthread_mutex_unlock(&first_lock->per_seg_mutex);

    /* Free only private-to-merge buffers (not shared with any search thread).
     * Old slots' index/bitmap/map are freed by cleanup_flushed_segment. */
    IndexFree(smaller_disk_ptr);
    free(bitmap0);       free(bitmap1);
    free(mapping0);      free(mapping1);
    free(offsets0);      free(offsets1);
    free(fresh_bitmap0); free(fresh_bitmap1);
    free(merged_offsets);

    fprintf(stderr,
            "[merge_adjacent_segments_pool] merged %u-%u + %u-%u → %u-%u (%d vectors v%u)\n",
            s0_start, s0_end, s1_start, s1_end,
            task->merged_start_sid, task->merged_end_sid, merged_count, new_version);
}

static void *
merge_worker_thread(void *arg)
{
    MergeThreadPool *pool = (MergeThreadPool *) arg;

    while (!atomic_load(&pool->shutdown))
    {
        MergeTaskLocal task;
        bool claimed;

        /* Wait for a wakeup signal */
        pthread_mutex_lock(&pool->mutex);
        while (!atomic_load(&pool->shutdown) && atomic_load(&pool->pending_signals) == 0)
            pthread_cond_wait(&pool->work_available, &pool->mutex);
        if (atomic_load(&pool->pending_signals) > 0)
            atomic_fetch_sub(&pool->pending_signals, 1);
        pthread_mutex_unlock(&pool->mutex);

        if (atomic_load(&pool->shutdown))
            break;

        /* Scan for a task and execute it */
        claimed = scan_and_claim_merge_task_pool(&task);

        if (!claimed)
            continue;

        fprintf(stderr, "[merge_worker_thread] claimed task op=%d lsm=%d seg0=%u seg1=%u\n",
                task.operation_type, task.lsm_idx, task.segment_idx0, task.segment_idx1);

        switch (task.operation_type)
        {
            case SEGMENT_UPDATE_REBUILD_FLAT:
            case SEGMENT_UPDATE_REBUILD_DELETION:
                rebuild_index_pool(&task);
                break;
            case SEGMENT_UPDATE_MERGE:
                merge_adjacent_segments_pool(&task);
                break;
            default:
                fprintf(stderr, "[merge_worker_thread] unknown op %d\n", task.operation_type);
                break;
        }

        /* Signal again — this completion may enable another merge */
        signal_merge_pool();
    }

    return NULL;
}

// Submit an internal task (not from the ring buffer) directly to the maintenance pool.
// Used for phase-2 mmap->full-memory upgrade tasks spawned by the IndexLoad handler.
static void
submit_internal_task(VectorTaskType task_type, void *data, size_t data_size)
{
    MaintenanceTask *task;

    if (maintenance_pool == NULL) {
        fprintf(stderr, "[submit_internal_task] maintenance pool not initialized\n");
        return;
    }

    if (atomic_load(&maintenance_pool->queue_size) >= MAINTENANCE_TASK_QUEUE_SIZE) {
        fprintf(stderr, "[submit_internal_task] task queue full, dropping upgrade task\n");
        return;
    }

    task = (MaintenanceTask *)malloc(sizeof(MaintenanceTask));
    if (task == NULL) {
        fprintf(stderr, "[submit_internal_task] malloc failed\n");
        return;
    }

    task->task_slot = NULL;
    task->task_type = task_type;
    memcpy(&task->task_data, data, data_size);
    task->next = NULL;

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
    
    // Attach DSM segment, copy data, then detach (all in PostgreSQL backend worker process context)
    dsm_handle task_hdl = task_slot->handle;
    dsm_segment *task_seg = dsm_attach(task_hdl);
    if (task_seg == NULL) {
        elog(ERROR, "[submit_maintenance_task] Failed to attach DSM segment");
        return;
    }
    
    // Allocate task structure
    MaintenanceTask *task = (MaintenanceTask *)malloc(sizeof(MaintenanceTask));
    if (task == NULL) {
        elog(ERROR, "[submit_maintenance_task] Failed to allocate task structure");
        dsm_detach(task_seg);
        return;
    }
    
    task->task_slot = task_slot;
    task->task_type = task_slot->type;
    
    // Copy task data from DSM segment based on task type
    switch (task_slot->type) {
        case IndexBuildTaskType:
        {
            IndexBuildTask src_task = (IndexBuildTask) dsm_segment_address(task_seg);
            task->task_data.build_task = *src_task;  // Copy the structure
            break;
        }
        case SegmentUpdateTaskType:
        {
            SegmentUpdateTask src_task = (SegmentUpdateTask) dsm_segment_address(task_seg);
            task->task_data.update_task = *src_task;  // Copy the structure
            break;
        }
        case IndexLoadTaskType:
        {
            IndexLoadTask src_task = (IndexLoadTask) dsm_segment_address(task_seg);
            task->task_data.load_task = *src_task;  // Copy the structure
            break;
        }
        default:
            elog(ERROR, "[submit_maintenance_task] Unknown task type %d", task_slot->type);
            free(task);
            dsm_detach(task_seg);
            return;
    }
    
    // Detach DSM segment now that we've copied the data
    dsm_detach(task_seg);
    
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
    init_merge_thread_pool();
    
    // FIXME: check this block
    pqsignal(SIGHUP, vq_sighup);
    pqsignal(SIGTERM, vq_sigterm);
    BackgroundWorkerUnblockSignals();
    // LWLockRegisterTranche(VECTOR_SEARCH_RING_TRANCHE_ID, VECTOR_SEARCH_RING_TRANCHE);

    LWLockAcquire(ring_buffer_shmem->lock, LW_EXCLUSIVE);
    ring_buffer_shmem->worker_pgprocno = MyProc->pgprocno;
    elog(DEBUG1, "[vector_index_worker] started, pgprocno=%d", MyProc->pgprocno);
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
            shutdown_merge_thread_pool();
            proc_exit(0);
        }

        // for graceful shutdown
        if (got_sigterm)
        {
            shutdown_merge_thread_pool();
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
    int lsm_idx = get_lsm_index_idx_no_loading(task->index_relid);
    FlushedSegmentPool *pool = get_flushed_segment_pool(lsm_idx);

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
                // FIXME: how is this guaranteed?
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