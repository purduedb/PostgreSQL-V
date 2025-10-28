#include "tasksend.h"


static void
notify_and_return()
{
    PGPROC *worker = &ProcGlobal->allProcs[ring_buffer_shmem->worker_pgprocno];
    SetLatch(&worker->procLatch);
}

static void
wait_for_latch()
{
    int rc = WaitLatch(MyLatch,
                       WL_LATCH_SET | WL_POSTMASTER_DEATH,
                       0, /* no timeout */
                       0  /* no wait-event reporting */);
    ResetLatch(MyLatch);
}

void
vector_search_send(Oid index_oid, float *query, int dim, Size elem_size, int topk, int efs_nprobe, LSMSnapshot lsm_snapshot)
{
    if(ring_buffer_shmem == NULL)
        ereport(ERROR, (errmsg("vector search shmem not initialized")));
    
    LWLockAcquire(ring_buffer_shmem->lock, LW_EXCLUSIVE);
    if (ring_buffer_shmem->count == ring_buffer_shmem->ring_size)
    {
        LWLockRelease(ring_buffer_shmem->lock);
        elog(ERROR, "vector search ring buffer full");
        return;
    }
    
    int slot = get_ring_buffer_slot();
    TaskSlot *task_slot = vs_task_at(slot);
    task_slot->type = VectorSearch;
    // we do not use handle to send vector search task
    task_slot->dsm_size = 0;
    task_slot->handle = 0;

    // TODO: can we release the lock before filling the task?
    // fill the task
    VectorSearchTask task = vs_search_task_at(slot);
    task->index_relid = index_oid;
    task->backend_pgprocno = MyProc->pgprocno;
    task->vector_dim = dim;
    task->topk = topk;
    task->efs_nprobe = efs_nprobe;
    task->snapshot = lsm_snapshot;
    float *vec = vs_search_task_vector_at(task);
    memcpy(vec, query, sizeof(float) * (Size)task->vector_dim);

    LWLockRelease(ring_buffer_shmem->lock);

    // notify the worker and return without blocking
    notify_and_return();
}

VectorSearchResult
vector_search_get_result(void)
{
    wait_for_latch();

    VectorSearchResult result = vs_search_result_at(MyProc->pgprocno);
    return result;
}

// TODO: do we need to protect `lsm_index_idx`? replace lsm_index_idx with the relation Oid?
void
index_build_blocking(Oid indexRelId, int lsm_index_idx)
{
    elog(DEBUG1, "enter index_build_blocking with lsm_index_idx = %d", lsm_index_idx);
    if(ring_buffer_shmem == NULL)
        elog(ERROR, "[index_build_blocking] vector search shmem not initialized");

    // create IndexBuildTask
    Size task_seg_size = sizeof(IndexBuildTaskData);
    dsm_segment *task_seg = dsm_create(task_seg_size, 0);
    if (task_seg == NULL)
        elog(ERROR, "[index_build_blocking] Failed to alocate dynamic shared memory segment");
    IndexBuildTask task = (IndexBuildTask) dsm_segment_address(task_seg);
    dsm_handle task_hdl = dsm_segment_handle(task_seg);

    // fill IndexBuildTask
    task->backend_pgprocno = MyProc->pgprocno;
    task->index_relid = indexRelId;
    task->lsm_idx = lsm_index_idx;
    
    LWLockAcquire(ring_buffer_shmem->lock, LW_EXCLUSIVE);
    if (ring_buffer_shmem->count == ring_buffer_shmem->ring_size)
    {
        LWLockRelease(ring_buffer_shmem->lock);
        elog(ERROR, "[index_build_blocking] vector search ring buffer full");
        dsm_detach(task_seg);
        return;
    }

    int slot = get_ring_buffer_slot();
    TaskSlot *task_slot = vs_task_at(slot);
    task_slot->type = IndexBuild;
    task_slot->dsm_size = task_seg_size;
    task_slot->handle = task_hdl;

    // Notify worker before releasing lock to prevent race condition
    PGPROC *worker = &ProcGlobal->allProcs[ring_buffer_shmem->worker_pgprocno];
    ResetLatch(MyLatch);
    SetLatch(&worker->procLatch);

    LWLockRelease(ring_buffer_shmem->lock);
    // elog(DEBUG1, "[index_build_blocking] slot = %d", slot);

    // Now wait for completion
    int rc = WaitLatch(MyLatch,
                       WL_LATCH_SET | WL_POSTMASTER_DEATH,
                       0, /* no timeout */
                       0  /* no wait-event reporting */);
    ResetLatch(MyLatch);
    dsm_detach(task_seg);
    elog(DEBUG1, "[index_build_blocking] IndexBuild task completed and DSM segment detached");
}

void
segment_update_blocking(int lsm_index_idx, Oid index_relid, int operation_type, SegmentId start_sid, SegmentId end_sid)
{
    if(ring_buffer_shmem == NULL)
        ereport(ERROR, (errmsg("[segment_update_blocking] vector search shmem not initialized")));

    // create SegmentUpdateTask
    Size task_seg_size = sizeof(SegmentUpdateTaskData);
    dsm_segment *task_seg = dsm_create(task_seg_size, 0);
    if (task_seg == NULL)
        elog(ERROR, "[segment_update_blocking] Failed to alocate dynamic shared memory segment");
    SegmentUpdateTask task = (SegmentUpdateTask) dsm_segment_address(task_seg);
    dsm_handle task_hdl = dsm_segment_handle(task_seg);

    // fill SegmentUpdateTask
    task->backend_pgprocno = MyProc->pgprocno;
    task->index_relid = index_relid;
    task->lsm_idx = lsm_index_idx;
    task->operation_type = operation_type;
    task->start_sid = start_sid;
    task->end_sid = end_sid;

    LWLockAcquire(ring_buffer_shmem->lock, LW_EXCLUSIVE);
    if (ring_buffer_shmem->count == ring_buffer_shmem->ring_size)
    {
        LWLockRelease(ring_buffer_shmem->lock);
        elog(ERROR, "[segment_update_blocking] vector search ring buffer full");
        dsm_detach(task_seg);
        return;
    }

    int slot = get_ring_buffer_slot();
    TaskSlot *task_slot = vs_task_at(slot);
    task_slot->type = SegmentUpdate;
    task_slot->dsm_size = task_seg_size;
    task_slot->handle = task_hdl;

    // Notify worker before releasing lock to prevent race condition
    PGPROC *worker = &ProcGlobal->allProcs[ring_buffer_shmem->worker_pgprocno];
    ResetLatch(MyLatch);
    SetLatch(&worker->procLatch);

    LWLockRelease(ring_buffer_shmem->lock);

    // Now wait for completion
    int rc = WaitLatch(MyLatch,
                       WL_LATCH_SET | WL_POSTMASTER_DEATH,
                       0, /* no timeout */
                       0  /* no wait-event reporting */);
    ResetLatch(MyLatch);
    dsm_detach(task_seg);
    elog(DEBUG1, "[segment_update_blocking] SegmentUpdate task completed and DSM segment detached");
}

void
index_load_blocking(Oid index_relid, int lsm_index_idx)
{
    elog(DEBUG1, "enter index_load_blocking with index_relid = %u, lsm_index_idx = %d", index_relid, lsm_index_idx);
    if(ring_buffer_shmem == NULL)
        elog(ERROR, "[index_load_blocking] vector search shmem not initialized");

    // create IndexLoadTask
    Size task_seg_size = sizeof(IndexLoadTaskData);
    dsm_segment *task_seg = dsm_create(task_seg_size, 0);
    if (task_seg == NULL)
        elog(ERROR, "[index_load_blocking] Failed to allocate dynamic shared memory segment");
    IndexLoadTask task = (IndexLoadTask) dsm_segment_address(task_seg);
    dsm_handle task_hdl = dsm_segment_handle(task_seg);

    // fill IndexLoadTask
    task->backend_pgprocno = MyProc->pgprocno;
    task->index_relid = index_relid;
    task->lsm_idx = lsm_index_idx;
    
    LWLockAcquire(ring_buffer_shmem->lock, LW_EXCLUSIVE);
    if (ring_buffer_shmem->count == ring_buffer_shmem->ring_size)
    {
        LWLockRelease(ring_buffer_shmem->lock);
        elog(ERROR, "[index_load_blocking] vector search ring buffer full");
        dsm_detach(task_seg);
        return;
    }

    int slot = get_ring_buffer_slot();
    TaskSlot *task_slot = vs_task_at(slot);
    task_slot->type = IndexLoad;
    task_slot->dsm_size = task_seg_size;
    task_slot->handle = task_hdl;

    // Notify worker before releasing lock to prevent race condition
    PGPROC *worker = &ProcGlobal->allProcs[ring_buffer_shmem->worker_pgprocno];
    ResetLatch(MyLatch);
    SetLatch(&worker->procLatch);

    LWLockRelease(ring_buffer_shmem->lock);

    // Now wait for completion
    int rc = WaitLatch(MyLatch,
                       WL_LATCH_SET | WL_POSTMASTER_DEATH,
                       0, /* no timeout */
                       0  /* no wait-event reporting */);
    ResetLatch(MyLatch);
    dsm_detach(task_seg);
    elog(DEBUG1, "[index_load_blocking] IndexLoad task completed and DSM segment detached");
}