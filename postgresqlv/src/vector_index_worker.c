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

static volatile sig_atomic_t got_sighup = false;
static volatile sig_atomic_t got_sigterm = false;

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
    
}

// the lock is held
static void
handle_task(TaskSlot *task_slot, int slot_idx)
{
    switch (task_slot->type)
    {
    case VectorSearchTaskType:
    {
        // elog(DEBUG1, "[handle_task] task_slot->type = VectorSearch");
        // get the task
        VectorSearchTask task = vs_search_task_at(slot_idx);

        // conduct the search
        VectorSearchResult result = vs_search_result_at(task->backend_pgprocno);
        vector_search(task, result);

        // notify the client
        PGPROC *client = &ProcGlobal->allProcs[task->backend_pgprocno];
        SetLatch(&client->procLatch);
        break;
    }
    case IndexBuildTaskType:
    {
        elog(DEBUG1, "[handle_task] task_slot->type = IndexBuildTaskType");
        dsm_handle task_hdl = task_slot->handle;
        dsm_segment *task_seg = dsm_attach(task_hdl);
        IndexBuildTask task = (IndexBuildTask) dsm_segment_address(task_seg);

        Oid index_relid = task->index_relid;
        int lsm_idx = task->lsm_idx;

        // initialize flushed segments
        FlushedSegmentPool *pool = get_flushed_segment_pool(lsm_idx);
        initialize_segment_pool(pool);
        uint32_t seg_idx = reserve_flushed_segment(pool);
        if(seg_idx == -1)
        {
            elog(ERROR, "[build_lsm_index] no free flushed segment slot");
        }
        load_and_set_segment(index_relid, seg_idx, &pool->flushed_segments[seg_idx], START_SEGMENT_ID, START_SEGMENT_ID);
        register_flushed_segment(pool, seg_idx);

        // notify the client
        PGPROC *client = &ProcGlobal->allProcs[task->backend_pgprocno];
        dsm_detach(task_seg);
        SetLatch(&client->procLatch);
        break;
    }
    case SegmentUpdateTaskType:
    {
        elog(DEBUG1, "[handle_task] task_slot->type = SegmentUpdate");
        dsm_handle task_hdl = task_slot->handle;
        dsm_segment *task_seg = dsm_attach(task_hdl);
        SegmentUpdateTask task = (SegmentUpdateTask) dsm_segment_address(task_seg);

        Oid index_relid = task->index_relid;
        int lsm_idx = task->lsm_idx;

        // Handle different types of segment update operations
        switch (task->operation_type)
        {
            case SEGMENT_UPDATE_REGULAR:
            {
                // Regular segment update (existing functionality)
                FlushedSegmentPool *pool = get_flushed_segment_pool(lsm_idx);

                pthread_rwlock_wrlock(&pool->seg_lock);
                uint32_t seg_idx = reserve_flushed_segment(pool);
                pthread_rwlock_unlock(&pool->seg_lock);

                load_and_set_segment(index_relid, seg_idx, &pool->flushed_segments[seg_idx], task->start_sid, task->end_sid);
                
                pthread_rwlock_wrlock(&pool->seg_lock);
                register_flushed_segment(pool, seg_idx);
                pthread_rwlock_unlock(&pool->seg_lock);
                break;
            }
            
            case SEGMENT_UPDATE_REBUILD_FLAT:
            case SEGMENT_UPDATE_REBUILD_DELETION:
            {
                // Index rebuild for flat segments or high deletion ratio segments
                const char *rebuild_type = (task->operation_type == SEGMENT_UPDATE_REBUILD_FLAT) ? 
                    "IndexRebuildFlat" : "IndexRebuildDeletion";
                elog(DEBUG1, "[handle_task] %s task received, start_sid = %d, end_sid = %d", 
                     rebuild_type, task->start_sid, task->end_sid);
                
                FlushedSegmentPool *pool = get_flushed_segment_pool(lsm_idx);
                
                // Step 1: Find the old segment
                pthread_rwlock_rdlock(&pool->seg_lock);
                uint32_t old_seg_idx = find_segment_by_sids(pool, task->start_sid, task->end_sid);
                pthread_rwlock_unlock(&pool->seg_lock);
                
                // Step 2: Reserve a new segment slot
                pthread_rwlock_wrlock(&pool->seg_lock);
                uint32_t new_seg_idx = reserve_flushed_segment(pool);
                pthread_rwlock_unlock(&pool->seg_lock);
                if (new_seg_idx == -1) {
                    elog(ERROR, "[handle_task] No free flushed segment slot available for rebuild");
                    break;
                }

                // Step 3: Load the rebuilt segment from disk
                load_and_set_segment(index_relid, new_seg_idx, &pool->flushed_segments[new_seg_idx], task->start_sid, task->end_sid);
                
                // Step 4: Replace old segment with new segment atomically
                pthread_rwlock_wrlock(&pool->seg_lock);
                replace_flushed_segment(pool, old_seg_idx, -1, new_seg_idx);
                pthread_rwlock_unlock(&pool->seg_lock);
                
                elog(DEBUG1, "[handle_task] Rebuild completed for segment %d-%d, replaced old at slot %u with new at slot %u", 
                     task->start_sid, task->end_sid, old_seg_idx, new_seg_idx);
                break;
            }
            
            case SEGMENT_UPDATE_MERGE:
            {
                // Segment merging
                elog(DEBUG1, "[handle_task] SegmentMerge task received: merging segments to create segment %d-%d", 
                     task->start_sid, task->end_sid);
                
                FlushedSegmentPool *pool = get_flushed_segment_pool(lsm_idx);

                // Step 1: Find the two adjacent old segments
                uint32_t old_seg_idx_0, old_seg_idx_1;
                pthread_rwlock_rdlock(&pool->seg_lock);
                find_two_adjacent_segments(pool, task->start_sid, task->end_sid, &old_seg_idx_0, &old_seg_idx_1);
                pthread_rwlock_unlock(&pool->seg_lock);
                
                if (old_seg_idx_0 == -1 || old_seg_idx_1 == -1)
                {
                    elog(ERROR, "[handle_task] Failed to find two adjacent segments for merge");
                    break;
                }
                elog(DEBUG1, "[handle_task] Found adjacent segments %u and %u for merge", old_seg_idx_0, old_seg_idx_1);
                
                // Step 2: Reserve a new segment slot
                pthread_rwlock_wrlock(&pool->seg_lock);
                uint32_t new_seg_idx = reserve_flushed_segment(pool);
                pthread_rwlock_unlock(&pool->seg_lock);
                if (new_seg_idx == -1) {
                    elog(ERROR, "[handle_task] No free flushed segment slot available for merge");
                    break;
                }

                // Step 3: Load the merged segment from disk
                load_and_set_segment(index_relid, new_seg_idx, &pool->flushed_segments[new_seg_idx], task->start_sid, task->end_sid);
                
                // Step 4: Replace old segments with new segment atomically
                pthread_rwlock_wrlock(&pool->seg_lock);
                replace_flushed_segment(pool, old_seg_idx_0, old_seg_idx_1, new_seg_idx);
                pthread_rwlock_unlock(&pool->seg_lock);

                elog(DEBUG1, "[handle_task] SegmentMerge completed, replaced segments %u and %u with segment %u", 
                     old_seg_idx_0, old_seg_idx_1, new_seg_idx);
                break;
            }
            
            default:
                elog(WARNING, "[handle_task] Unknown segment update operation type %d", task->operation_type);
                break;
        }
        // notify the merge worker
        PGPROC *client = &ProcGlobal->allProcs[task->backend_pgprocno];
        dsm_detach(task_seg);
        SetLatch(&client->procLatch);
        break;
    }
    case IndexLoadTaskType:
    {
        elog(DEBUG1, "[handle_task] task_slot->type = IndexLoad");
        dsm_handle task_hdl = task_slot->handle;
        dsm_segment *task_seg = dsm_attach(task_hdl);
        IndexLoadTask task = (IndexLoadTask) dsm_segment_address(task_seg);

        Oid index_relid = task->index_relid;
        int lsm_idx = task->lsm_idx;

        FlushedSegmentPool *pool = get_flushed_segment_pool(lsm_idx);
        initialize_segment_pool(pool);
        // Load all segments from disk for this index
        load_all_segments_from_disk(index_relid, pool);

        // notify the client
        PGPROC *client = &ProcGlobal->allProcs[task->backend_pgprocno];
        dsm_detach(task_seg);
        SetLatch(&client->procLatch);
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
    // elog(DEBUG1, "enter vector_search");
    
    DistancePair *final_pairs = NULL, *pairs_1 = NULL;
    int num_1;

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
    do {
        if (pool->flushed_segments[idx].in_used) {
            // Increment reference count when adding to traversal list
            atomic_fetch_add(&pool->flushed_segments[idx].ref_count, 1);
            segment_indices[segment_count++] = idx;
        }
        if (idx == tail_idx_snapshot) break;
        idx = pool->flushed_segments[idx].next_idx;
    } while (true);
    // TODO: for debugging
    elog(DEBUG1, "[vector_search] segment_count = %d", segment_count);

    // Release lock early to minimize contention
    pthread_rwlock_unlock(&pool->seg_lock);
    
    // Now traverse segments without holding the lock
    for (uint32_t i = 0; i < segment_count; i++) {
        uint32_t idx = segment_indices[i];
        
        // TODO: for debugging
        elog(DEBUG1, "[vector_search] considering search on segment %u-%u, idx = %u", pool->flushed_segments[idx].segment_id_start, pool->flushed_segments[idx].segment_id_end, idx);
        // skip the flushed segment if its segment id is in the snapshot
        bool found = false;
        for (int j = 0; j < task->snapshot.scount; j++)
        {
            if (task->snapshot.smt_ids[j] <= pool->flushed_segments[idx].segment_id_end &&
                task->snapshot.smt_ids[j] >= pool->flushed_segments[idx].segment_id_start)
            {
                Assert(pool->flushed_segments[idx].segment_id_end == pool->flushed_segments[idx].segment_id_start);
                found = true;
                break;
            }
        }
        found = found || (task->snapshot.gmt_id <= pool->flushed_segments[idx].segment_id_end &&
                        task->snapshot.gmt_id >= pool->flushed_segments[idx].segment_id_start);
        if (!found)
        {   
            // TODO: for debugging
            elog(DEBUG1, "[vector_search] conducting search on segment %u-%u", pool->flushed_segments[idx].segment_id_start, pool->flushed_segments[idx].segment_id_end);
            // conduct search
            topKVector *segment_result;
            int64_t *mapping = pool->flushed_segments[idx].map_ptr;

            segment_result = VectorIndexSearch(pool->flushed_segments[idx].index_type, pool->flushed_segments[idx].index_ptr, pool->flushed_segments[idx].bitmap_ptr, 
                                                pool->flushed_segments[idx].vec_count, 
                                                vs_search_task_vector_at(task), 
                                                task->topk, 
                                                task->efs_nprobe);
            // elog(DEBUG1, "[vector_search] returned from VectorIndexSearch, segment_result->num_results = %d", segment_result->num_results);

            // merge the result
            DistancePair *segment_pairs = palloc(sizeof(DistancePair) * task->topk);
            for (int k = 0; k < segment_result->num_results; k++)
            {
                segment_pairs[k].distance = segment_result->distances[k];
                int pos = segment_result->vids[k];
                segment_pairs[k].id = mapping[pos];
            }
            if (pairs_1 == NULL)
            {
                num_1 = segment_result->num_results;
                pairs_1 = segment_pairs;
            }
            else
            {
                final_pairs = palloc(sizeof(DistancePair) * task->topk);
                int merge_num = merge_top_k(pairs_1, segment_pairs, num_1, segment_result->num_results, task->topk, final_pairs);
                pfree(pairs_1);
                pfree(segment_pairs);
                num_1 = merge_num;
                pairs_1 = final_pairs;
            }
            free_topk_vector(segment_result);
        }
        
        // Decrement reference count for this segment (whether searched or skipped)
        decrement_flushed_segment_ref_count(pool, idx);
    }

    // return the result
    result->result_count = num_1;
    float* res_dist = vs_search_result_dist_at(result);
    int64_t* res_id = vs_search_result_id_at(result);
    for (int i = 0; i < num_1; i++)
    {
        res_dist[i] = pairs_1[i].distance;
        res_id[i] = pairs_1[i].id;
    }
    pfree(pairs_1);
    
    // elog(DEBUG1, "[vector_search] result->result_count = %d", result->result_count);    
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