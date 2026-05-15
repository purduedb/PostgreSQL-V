#include "tasksend.h"
#include "utils/elog.h"

/* Timeout per WaitLatch iteration while waiting for the worker (milliseconds). */
#define WAIT_LATCH_TIMEOUT_MS 1000L

static void
notify_worker(void)
{
    PGPROC *worker = &ProcGlobal->allProcs[ring_buffer_shmem->worker_pgprocno];
    SetLatch(&worker->procLatch);
}

void
vector_search_send(Oid index_oid, float *query, int dim, Size elem_size, int topk, int efs_nprobe, LSMSnapshot lsm_snapshot)
{
    int slot;
    TaskSlot *task_slot;
    VectorSearchTask task;
    float *vec;

    if (ring_buffer_shmem == NULL)
        ereport(ERROR, (errmsg("vector search shmem not initialized")));

    LWLockAcquire(ring_buffer_shmem->lock, LW_EXCLUSIVE);
    if (ring_buffer_shmem->count == ring_buffer_shmem->ring_size)
    {
        LWLockRelease(ring_buffer_shmem->lock);
        elog(ERROR, "vector search ring buffer full");
        return;
    }

    slot = get_ring_buffer_slot();
    // elog(DEBUG1, "[vector_search_send] enqueued search: index=%u dim=%d topk=%d slot=%d",
    //      index_oid, dim, topk, slot);
    task_slot = vs_task_at(slot);
    task_slot->type = VectorSearchTaskType;
    task_slot->dsm_size = 0;
    task_slot->handle = 0;

    task = vs_search_task_at(slot);
    task->index_relid = index_oid;
    task->backend_pgprocno = MyProcNumber;
    task->vector_dim = dim;
    task->topk = topk;
    task->efs_nprobe = efs_nprobe;
    task->snapshot = lsm_snapshot;

    vec = vs_search_task_vector_at(task);
    memcpy(vec, query, sizeof(float) * (Size)task->vector_dim);

    LWLockRelease(ring_buffer_shmem->lock);

    notify_worker();
}

VectorSearchResult
vector_search_get_result(void)
{
    int rc;

    for (;;)
    {
        rc = WaitLatch(MyLatch,
                       WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
                       WAIT_LATCH_TIMEOUT_MS,
                       0);
        ResetLatch(MyLatch);

        if (rc & WL_POSTMASTER_DEATH)
            ereport(ERROR, (errmsg("[vector_search_get_result] postmaster died")));

        if (rc & WL_LATCH_SET)
        {
            // elog(DEBUG1, "[vector_search_get_result] result received");
            break;
        }
        /* WL_TIMEOUT: keep waiting. */
    }

    return vs_search_result_at(MyProcNumber);
}

/*
 * Enqueue a DSM-backed maintenance task and wait for the worker to complete
 * it.  A kill -9 on the worker triggers a full cluster restart, so all
 * backends die via WL_POSTMASTER_DEATH before any retry is needed.
 */
static void
submit_and_wait_maintenance(VectorTaskType task_type, dsm_handle task_hdl,
                             Size task_seg_size, const char *caller_name)
{
    int slot;
    TaskSlot *task_slot;
    PGPROC *worker;
    int rc;

    ResetLatch(MyLatch);

    LWLockAcquire(ring_buffer_shmem->lock, LW_EXCLUSIVE);

    if (ring_buffer_shmem->count == ring_buffer_shmem->ring_size)
    {
        LWLockRelease(ring_buffer_shmem->lock);
        elog(ERROR, "[%s] vector search ring buffer full", caller_name);
    }

    slot = get_ring_buffer_slot();
    elog(DEBUG1, "[%s] submitting maintenance task: slot=%d", caller_name, slot);
    task_slot = vs_task_at(slot);
    task_slot->type = task_type;
    task_slot->dsm_size = task_seg_size;
    task_slot->handle = task_hdl;

    worker = &ProcGlobal->allProcs[ring_buffer_shmem->worker_pgprocno];
    SetLatch(&worker->procLatch);

    LWLockRelease(ring_buffer_shmem->lock);

    for (;;)
    {
        rc = WaitLatch(MyLatch,
                       WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
                       WAIT_LATCH_TIMEOUT_MS,
                       0);
        ResetLatch(MyLatch);

        if (rc & WL_POSTMASTER_DEATH)
            ereport(ERROR,
                    (errmsg("[%s] postmaster died while waiting for worker",
                            caller_name)));

        if (rc & WL_LATCH_SET)
        {
            elog(DEBUG1, "[%s] maintenance task completed", caller_name);
            break;
        }
        /* WL_TIMEOUT: keep waiting. */
    }
}

void
index_build_blocking(Oid indexRelId, int lsm_index_idx)
{
    Size task_seg_size = sizeof(IndexBuildTaskData);
    dsm_segment *task_seg;
    IndexBuildTask task;

    elog(DEBUG1, "enter index_build_blocking with lsm_index_idx = %d", lsm_index_idx);
    if (ring_buffer_shmem == NULL)
        elog(ERROR, "[index_build_blocking] vector search shmem not initialized");

    task_seg = dsm_create(task_seg_size, 0);
    if (task_seg == NULL)
        elog(ERROR, "[index_build_blocking] Failed to allocate dynamic shared memory segment");

    task = (IndexBuildTask) dsm_segment_address(task_seg);
    task->backend_pgprocno = MyProcNumber;
    task->index_relid = indexRelId;
    task->lsm_idx = lsm_index_idx;

    submit_and_wait_maintenance(IndexBuildTaskType, dsm_segment_handle(task_seg),
                                task_seg_size, "index_build_blocking");

    dsm_detach(task_seg);
    elog(DEBUG1, "[index_build_blocking] IndexBuildTask completed");
}

int
segment_update_blocking(int lsm_index_idx, Oid index_relid, int operation_type,
                         SegmentId start_sid, SegmentId end_sid,
                         uint32_t expected_version)
{
    Size task_seg_size = sizeof(SegmentUpdateTaskData);
    dsm_segment *task_seg;
    SegmentUpdateTask task;
    int result;

    if (ring_buffer_shmem == NULL)
        ereport(ERROR, (errmsg("[segment_update_blocking] vector search shmem not initialized")));

    task_seg = dsm_create(task_seg_size, 0);
    if (task_seg == NULL)
        elog(ERROR, "[segment_update_blocking] Failed to allocate DSM segment");

    task = (SegmentUpdateTask) dsm_segment_address(task_seg);
    task->backend_pgprocno = MyProcNumber;
    task->index_relid = index_relid;
    task->lsm_idx = lsm_index_idx;
    task->operation_type = operation_type;
    task->start_sid = start_sid;
    task->end_sid = end_sid;
    task->expected_version = expected_version;

    /* Clear maint_status before submitting so a stale value cannot be misread */
    vs_search_result_at(MyProcNumber)->maint_status = 0;

    submit_and_wait_maintenance(SegmentUpdateTaskType, dsm_segment_handle(task_seg),
                                task_seg_size, "segment_update_blocking");

    result = vs_search_result_at(MyProcNumber)->maint_status;

    dsm_detach(task_seg);
    elog(DEBUG1, "[segment_update_blocking] completed, result=%d", result);
    return result;
}

void
index_load_blocking(Oid index_relid, int lsm_index_idx)
{
    Size task_seg_size = sizeof(IndexLoadTaskData);
    dsm_segment *task_seg;
    IndexLoadTask task;

    elog(DEBUG1, "enter index_load_blocking with index_relid = %u, lsm_index_idx = %d",
         index_relid, lsm_index_idx);
    if (ring_buffer_shmem == NULL)
        elog(ERROR, "[index_load_blocking] vector search shmem not initialized");

    task_seg = dsm_create(task_seg_size, 0);
    if (task_seg == NULL)
        elog(ERROR, "[index_load_blocking] Failed to allocate dynamic shared memory segment");

    task = (IndexLoadTask) dsm_segment_address(task_seg);
    task->backend_pgprocno = MyProcNumber;
    task->index_relid = index_relid;
    task->lsm_idx = lsm_index_idx;

    submit_and_wait_maintenance(IndexLoadTaskType, dsm_segment_handle(task_seg),
                                task_seg_size, "index_load_blocking");

    dsm_detach(task_seg);
    elog(DEBUG1, "[index_load_blocking] IndexLoad task completed");
}
