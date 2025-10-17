#include "ringbuffer.h"

RingBufferShmem *ring_buffer_shmem = NULL;
VectorSearchTaskData *vector_search_task_pool = NULL;
VectorSearchResultData *vector_search_result_pool = NULL;

Size
calculate_ring_buffer_shmem_size()
{
    Size size = 0;

    size = add_size(size, MAXALIGN(sizeof(RingBufferShmem)));

    Size task_stride = MAXALIGN(sizeof(TaskSlot));
    size = add_size(size, mul_size(task_stride, (Size)MaxBackends));

    Size result_stride = MAXALIGN(sizeof(ResultSlot));
    size = add_size(size, mul_size(result_stride, (Size)MaxBackends));

    return size;
}

Size
calculate_vector_search_task_pool_size()
{
    Size task_stride = add_size(MAXALIGN(sizeof(VectorSearchTaskData)), mul_size(sizeof(float), (Size)MAX_DIM));
    Size size = mul_size(task_stride, (Size)MaxBackends);
    return size;
}

Size
calculate_vector_search_result_pool_size()
{
    // id + distance
    Size result_stride = add_size(MAXALIGN(sizeof(VectorSearchResultData)), mul_size((Size)MAX_TOPK, sizeof(int64_t)));
    result_stride = add_size(result_stride, mul_size(MAX_TOPK, sizeof(float)));
    Size size = mul_size(result_stride, (Size)MaxBackends);
    return size;
}

// helper functions for ring buffer operations
TaskSlot *
vs_task_at(int idx)
{
    return (TaskSlot *)((char *)ring_buffer_shmem + ring_buffer_shmem->task_handles_ofst + (Size)idx * sizeof(TaskSlot));
}

ResultSlot *
vs_result_at(int pgprocno)
{
    return (ResultSlot *)((char *)ring_buffer_shmem + ring_buffer_shmem->result_handles_ofst + (Size)pgprocno * sizeof(ResultSlot));
}

float *
vs_search_task_vector_at(VectorSearchTask task)
{
    return (float*)((char*)task + MAXALIGN(sizeof(VectorSearchTaskData)));
}

VectorSearchTask
vs_search_task_at(int idx)
{
    return (VectorSearchTaskData *)((char *)vector_search_task_pool + (Size)idx * ring_buffer_shmem->search_task_stride);
}

VectorSearchResult
vs_search_result_at(int pgprocno)
{
    return (VectorSearchResultData *)((char *)vector_search_result_pool + (Size)pgprocno * ring_buffer_shmem->search_result_stride);
}

float*
vs_search_result_dist_at(VectorSearchResult result)
{
    return (float*)((char*)result + MAXALIGN(sizeof(VectorSearchResultData)) + mul_size((Size)result->result_count, sizeof(int64_t)));
}

int64_t*
vs_search_result_id_at(VectorSearchResult result)
{
    return (int64_t*)((char*)result + MAXALIGN(sizeof(VectorSearchResultData)));
}

// Size 
// get_vec_search_task_size(int dim, Size elem_size)
// {
//     Size s = MAXALIGN(sizeof(VectorSearchTaskData));
//     s = add_size(s, mul_size(elem_size, dim));
//     return s;
// }

// Size
// get_vec_search_result_size(int top_k, Size id_size, Size dist_size)
// {
//     Size s = MAXALIGN(sizeof(VectorSearchResultData));
//     s = add_size(s, mul_size(top_k, id_size));
//     s = add_size(s, mul_size(top_k, dist_size));
//     return s;
// }

// Assume the lock is already acquired and ring_buffer_shmem->count < ring_buffer_shmem->ring_size
int 
get_ring_buffer_slot()
{
    int slot = ring_buffer_shmem->tail;
    ring_buffer_shmem->tail = (ring_buffer_shmem->tail + 1) % ring_buffer_shmem->ring_size;
    ring_buffer_shmem->count++;
    return slot;
}

void
ring_buffer_init(void)
{
    elog(DEBUG1, "[ring_buffer_init] initializing vector search ring buffer");
    bool found = false;
    LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
    Size total_size = calculate_ring_buffer_shmem_size();
    ring_buffer_shmem = ShmemInitStruct("vector search ring buffer",
                                          total_size, &found);
    if (!found)
    {
        // first time initialization
        memset(ring_buffer_shmem, 0, total_size);

        // assign LWLock from named tranche
        LWLockPadded *tr = GetNamedLWLockTranche(VECTOR_SEARCH_RING_TRANCHE);
        ring_buffer_shmem->lock = &tr[0].lock;

        ring_buffer_shmem->ring_size = MaxBackends; // each backend can have one task
        ring_buffer_shmem->head = 0;
        ring_buffer_shmem->tail = 0;
        ring_buffer_shmem->count = 0;
        ring_buffer_shmem->worker_pgprocno = -1;

        ring_buffer_shmem->task_handles_ofst = MAXALIGN(sizeof(RingBufferShmem));
        Size task_stride = MAXALIGN(sizeof(TaskSlot));
        ring_buffer_shmem->result_handles_ofst = ring_buffer_shmem->task_handles_ofst + mul_size(task_stride, (Size)MaxBackends);

        ring_buffer_shmem->search_task_stride = add_size(MAXALIGN(sizeof(VectorSearchTaskData)), mul_size(sizeof(float), (Size)MAX_DIM));
        Size result_stride = add_size(MAXALIGN(sizeof(VectorSearchResultData)), mul_size((Size)MAX_TOPK, sizeof(int64_t)));
        result_stride = add_size(result_stride, mul_size(MAX_TOPK, sizeof(float)));
        ring_buffer_shmem->search_result_stride = result_stride;
    }

    // vector_search_task_pool
    total_size = calculate_vector_search_task_pool_size();
    vector_search_task_pool = ShmemInitStruct("vector search task pool",
                                                total_size, &found);
    if (!found)
    {
        memset(vector_search_task_pool, 0, total_size);
    }
    // vector_search_result_pool
    total_size = calculate_vector_search_result_pool_size();
    vector_search_result_pool = ShmemInitStruct("vector search result pool",
                                                total_size, &found);
    if (!found)
    {
        memset(vector_search_result_pool, 0, total_size);
        // initialize the result slots
        for (int i = 0; i < MaxBackends; i++)
        {
            VectorSearchResultData *result = vs_search_result_at(i);
            result->status = 0;
            result->result_count = 0;
            result->result_size = 0;
        }
    }

    LWLockRelease(AddinShmemInitLock);
}