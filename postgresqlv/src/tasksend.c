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

    /*
     * Publish "result not ready" before the worker can possibly run.  The
     * matching loop in vector_search_get_result() spins on status and uses
     * the latch only as a wakeup hint, so a spurious MyLatch set (sinval
     * catchup, procsignal, ...) can no longer make the backend read a
     * stale/half-written result slot.  Written under the ring lock:
     * LWLockRelease has release semantics, ordering this store before the
     * worker dequeues the task under the same lock.
     */
    vs_search_result_at(MyProcNumber)->status = 0;

    LWLockRelease(ring_buffer_shmem->lock);

    notify_worker();
}

VectorSearchResult
vector_search_get_result(void)
{
    VectorSearchResult result = vs_search_result_at(MyProcNumber);

    /*
     * Wait until the worker (or its last-finishing folly thread) flips
     * status to 1.  A bare latch wakeup cannot be trusted: MyLatch is the
     * backend's shared process latch and PostgreSQL sets it for many
     * unrelated reasons whose traffic scales with backend count.  We use
     * reset-before-check to avoid a lost wakeup: if the worker sets
     * status + latch between our ResetLatch and WaitLatch, WaitLatch
     * returns immediately and the next iteration observes status == 1.
     */
    for (;;)
    {
        int rc;

        ResetLatch(MyLatch);

        if (result->status == 1)
        {
            pg_read_barrier();  /* observe result_count + payload after status */
            break;
        }

        rc = WaitLatch(MyLatch,
                       WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
                       WAIT_LATCH_TIMEOUT_MS,
                       0);

        if (rc & WL_POSTMASTER_DEATH)
            ereport(ERROR, (errmsg("[vector_search_get_result] postmaster died")));
        /* WL_LATCH_SET / WL_TIMEOUT: loop and re-check status. */
    }

    return result;
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
                         uint32_t expected_version,
                         const SegmentId *memtable_cover,
                         int memtable_cover_count,
                         const DpvVacuumGroup *groups,
                         int n_groups)
{
    Size task_struct_size = sizeof(SegmentUpdateTaskData);
    Size task_seg_size;
    dsm_segment *task_seg;
    SegmentUpdateTask task;
    int result;
    int cover_count;

    if (ring_buffer_shmem == NULL)
        ereport(ERROR, (errmsg("[segment_update_blocking] vector search shmem not initialized")));

    /*
     * Plan 3 refactor: the DSM trailer carries per-sid groups instead of a
     * flat tid array. Layout (starting at byte offset
     * sizeof(SegmentUpdateTaskData)):
     *
     *   uint32 n_groups
     *   For each group:
     *     uint32     sid
     *     uint32     n_tids
     *     int64_t    tids[n_tids]
     *
     * Sizes use natural alignment; the int64_t arrays land on 8-aligned
     * offsets because the SegmentUpdateTaskData header is 8-aligned and
     * we keep (sid, n_tids) as two uint32 (8 bytes) before each tids[].
     */
    Size payload_size = sizeof(uint32);     /* n_groups field */
    if (groups == NULL || n_groups <= 0)
    {
        n_groups = 0;
    }
    else
    {
        for (int g = 0; g < n_groups; g++)
            payload_size += sizeof(uint32) * 2 +
                            (Size) groups[g].n_tids * sizeof(int64_t);
    }

    task_seg_size = task_struct_size + payload_size;

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

    /* Copy memtable cover for ADOPT tasks; zero it out for all other ops. */
    if (memtable_cover != NULL && memtable_cover_count > 0)
    {
        cover_count = memtable_cover_count;
        if (cover_count > MEMTABLE_NUM + 1)
            cover_count = MEMTABLE_NUM + 1;
        task->memtable_cover_count = (uint8) cover_count;
        for (int i = 0; i < cover_count; i++)
            task->memtable_cover[i] = memtable_cover[i];
    }
    else
    {
        task->memtable_cover_count = 0;
    }

    /* Pack the per-sid group payload. */
    {
        char *p = (char *) dsm_segment_address(task_seg) + task_struct_size;
        uint32 hdr_ngroups = (uint32) n_groups;
        memcpy(p, &hdr_ngroups, sizeof(uint32));
        p += sizeof(uint32);
        for (int g = 0; g < n_groups; g++)
        {
            uint32 sid_u32 = (uint32) groups[g].sid;
            uint32 n       = groups[g].n_tids;
            memcpy(p, &sid_u32, sizeof(uint32));  p += sizeof(uint32);
            memcpy(p, &n,       sizeof(uint32));  p += sizeof(uint32);
            if (n > 0)
                memcpy(p, groups[g].tids, (Size) n * sizeof(int64_t));
            p += (Size) n * sizeof(int64_t);
        }
    }

    /* Carry the *byte length* of the group payload so the consumer doesn't
     * recompute it. Repurpose the existing deletion_tids_count field by
     * reading it as a byte count (consumer is updated to match). */
    task->deletion_tids_count = (uint32) payload_size;

    /* Clear maint_status before submitting so a stale value cannot be misread */
    vs_search_result_at(MyProcNumber)->maint_status = 0;

    submit_and_wait_maintenance(SegmentUpdateTaskType, dsm_segment_handle(task_seg),
                                task_seg_size, "segment_update_blocking");

    result = vs_search_result_at(MyProcNumber)->maint_status;

    dsm_detach(task_seg);
    elog(DEBUG1, "[segment_update_blocking] completed, result=%d", result);
    return result;
}

int
dpv_send_adopt_task(int lsm_idx, Oid index_relid,
                    SegmentId start_sid, SegmentId end_sid,
                    uint32_t version,
                    const SegmentId *memtable_cover,
                    int memtable_cover_count,
                    const DpvVacuumGroup *groups,
                    int n_groups)
{
    return segment_update_blocking(lsm_idx, index_relid, SEGMENT_UPDATE_ADOPT,
                                    start_sid, end_sid, version,
                                    memtable_cover, memtable_cover_count,
                                    groups, n_groups);
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
