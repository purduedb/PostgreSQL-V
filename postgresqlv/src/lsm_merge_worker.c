#include "postgres.h"
#include "postmaster/postmaster.h"
#include "postmaster/bgworker.h"
#include "tcop/tcopprot.h"
#include "storage/shmem.h"
#include "storage/lwlock.h"
#include "storage/pmsignal.h"
#include "miscadmin.h"
#include "utils/timestamp.h"

#include "lsm_merge_worker.h"
#include "lsmindex.h"
#include "vectorindeximpl.hpp"
#include "tasksend.h"
#include "ringbuffer.h"

// Global merge worker manager
MergeWorkerManager *merge_worker_manager = NULL;

// Worker ID for this instance
static int current_worker_id = -1;

// Signal handling
static volatile sig_atomic_t got_sigterm = false;
static volatile sig_atomic_t got_sighup = false;

static void
merge_worker_sigterm(SIGNAL_ARGS)
{
    int save_errno = errno;
    got_sigterm = true;
    SetLatch(MyLatch);
    errno = save_errno;
}

static void
merge_worker_sighup(SIGNAL_ARGS)
{
    int save_errno = errno;
    got_sighup = true;
    SetLatch(MyLatch);
    errno = save_errno;
}

static inline bool
OurPostmasterIsAlive(void)
{
    if (likely(!postmaster_possibly_dead))
        return true;
    return PostmasterIsAliveInternal();
}

// Initialize merge segment array
static void
initialize_merge_segment_array(SharedSegmentArray *seg_array)
{
    seg_array->current_index_relid = InvalidOid;
    seg_array->segment_count = 0;
    seg_array->tail_idx = -1;  // -1 indicates empty list
    seg_array->insert_idx = 0;
    seg_array->max_end_segment_id = 0;
    pg_atomic_init_u32(&seg_array->flat_count, 0);
    pg_atomic_init_u32(&seg_array->memtable_capacity_le_count, 0);
    pg_atomic_init_u32(&seg_array->small_segment_le_count, 0);
    
    // Initialize all segments as unused
    for (int i = 0; i < MAX_SEGMENTS_COUNT; i++)
    {
        seg_array->segments[i].in_used = false;
        seg_array->segments[i].is_compacting = false;
        seg_array->segments[i].next_idx = -1;  // -1 indicates end of list
        seg_array->segments[i].prev_idx = -1;  
    }
}

// Initialize the merge worker manager
void
initialize_merge_worker_manager(void)
{
    bool found;
    
    LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
    
    merge_worker_manager = (MergeWorkerManager *)
        ShmemInitStruct("Merge Worker Manager",
                       sizeof(MergeWorkerManager),
                       &found);
    
    if (!found)
    {
        // First time through, so initialize
        // Get a LWLock from the named tranche
        LWLockPadded *tr = GetNamedLWLockTranche("LSM Merge Worker");
        merge_worker_manager->lock = &tr[0].lock;
        merge_worker_manager->active_worker_count = 0;
        
        // Initialize worker states
        for (int i = 0; i < MAX_MERGE_WORKERS; i++)
        {
            merge_worker_manager->workers[i].worker_id = i;
            merge_worker_manager->workers[i].active = false;
        }
        
        // Initialize shared segment arrays
        for (int i = 0; i < INDEX_BUF_SIZE; i++)
        {
            SharedSegmentArray *seg_array = &merge_worker_manager->segment_arrays[i];
            // Get LWLock from the named tranche (we'll need one per segment array)
            LWLockPadded *tr = GetNamedLWLockTranche("LSM Merge Worker");
            seg_array->lock = &tr[i + 1].lock; // Use different locks for each array
            
            initialize_merge_segment_array(seg_array);
        }
        
        elog(DEBUG1, "[initialize_merge_worker_manager] Initialized merge worker manager with shared segment arrays");
    }
    
    LWLockRelease(AddinShmemInitLock);
}

// Statistics update functions

// Update statistics when adding a segment
static void
update_statistics_on_add(SharedSegmentArray *seg_array, IndexType index_type, uint32_t vec_count)
{
    if (index_type == FLAT)
    {
        (void) pg_atomic_add_fetch_u32(&seg_array->flat_count, 1);
    }
    if (vec_count <= MEMTABLE_MAX_CAPACITY)
    {
        (void) pg_atomic_add_fetch_u32(&seg_array->memtable_capacity_le_count, 1);
    }
    if (vec_count <= THRESHOLD_SMALL_SEGMENT_SIZE)
    {
        (void) pg_atomic_add_fetch_u32(&seg_array->small_segment_le_count, 1);
    }
}

// Update statistics when removing a segment
static void
update_statistics_on_remove(SharedSegmentArray *seg_array, IndexType index_type, uint32_t vec_count)
{
    if (index_type == FLAT)
    {
        (void) pg_atomic_add_fetch_u32(&seg_array->flat_count, -1);
    }
    if (vec_count <= MEMTABLE_MAX_CAPACITY)
    {
        (void) pg_atomic_add_fetch_u32(&seg_array->memtable_capacity_le_count, -1);
    }
    if (vec_count <= THRESHOLD_SMALL_SEGMENT_SIZE)
    {
        (void) pg_atomic_add_fetch_u32(&seg_array->small_segment_le_count, -1);
    }
}

// Linked list management functions (similar to FlushedSegmentPool)

// Reserve a segment slot (similar to reserve_flushed_segment) (need to hold the lock)
static uint32_t
reserve_merge_segment_slot(SharedSegmentArray *seg_array)
{
    elog(DEBUG1, "enter reserve_merge_segment_slot, insert_idx = %u", seg_array->insert_idx);
    
    for (uint32_t idx = seg_array->insert_idx; idx < MAX_SEGMENTS_COUNT; ++idx)
    {
        if (!seg_array->segments[idx].in_used)
        {
            seg_array->insert_idx = idx + 1;
            seg_array->segments[idx].in_used = true;
            seg_array->segments[idx].is_compacting = false;
            elog(DEBUG1, "reserve_merge_segment_slot successfully, idx = %u", idx);
            return idx;
        }
    }
    return -1;  // No free slot
}

// Register a segment in the linked list (similar to register_flushed_segment) (need to hold the lock)
static void
register_merge_segment(SharedSegmentArray *seg_array, uint32_t idx)
{
    elog(DEBUG1, "enter register_merge_segment for idx = %u", idx);
    
    // Set next_idx
    if (seg_array->tail_idx != -1)
        seg_array->segments[seg_array->tail_idx].next_idx = idx;
    seg_array->segments[idx].prev_idx = seg_array->tail_idx;
    seg_array->tail_idx = idx;
    seg_array->segments[idx].next_idx = -1;  // This is now the tail
    ++seg_array->segment_count;
    
    // Update statistics using unified function
    update_statistics_on_add(seg_array, seg_array->segments[idx].index_type, seg_array->segments[idx].vec_count);
    
    elog(DEBUG1, "register_merge_segment successfully added to list, idx = %u", idx);
}

// Remove a segment from the linked list (need to hold the lock)
static void
remove_from_segment_array(SharedSegmentArray *seg_array, uint32_t idx)
{
    elog(DEBUG1, "enter remove_merge_segment for idx = %u", idx);
    
    if (!seg_array->segments[idx].in_used)
    {
        elog(WARNING, "remove_merge_segment: segment %u is not in use", idx);
        return;
    }
    
    // Update the linked list
    uint32_t prev_idx = seg_array->segments[idx].prev_idx;
    uint32_t next_idx = seg_array->segments[idx].next_idx;
    if (prev_idx != -1)
    {
        // Not the first segment - update previous segment's next_idx
        seg_array->segments[prev_idx].next_idx = seg_array->segments[idx].next_idx;
    }
    else {
        elog(ERROR, "remove_merge_segment: segment %u is the first segment", idx);
    }
    if (next_idx != -1)
    {
        // Not the last segment - update next segment's prev_idx
        seg_array->segments[next_idx].prev_idx = prev_idx;
    }
    
    if (seg_array->tail_idx == idx)
    {
        // This was the tail - update tail_idx
        seg_array->tail_idx = prev_idx;
    }
    
    // Mark segment as unused
    // Update statistics using unified function before marking as unused
    update_statistics_on_remove(seg_array, seg_array->segments[idx].index_type, seg_array->segments[idx].vec_count);
    
    seg_array->segments[idx].in_used = false;
    seg_array->segments[idx].next_idx = -1;
    seg_array->segments[idx].prev_idx = -1;
    --seg_array->segment_count;
    
    elog(DEBUG1, "remove_merge_segment successfully removed from list, idx = %u", idx);
}

// Add a segment to the linked list (need to hold the lock)
void
add_to_segment_array(uint32_t lsm_idx, Oid index_relid, SegmentId start_sid, SegmentId end_sid, uint32_t vec_count, IndexType index_type, float deletion_ratio)
{
    SharedSegmentArray *seg_array = &merge_worker_manager->segment_arrays[lsm_idx];
    LWLockAcquire(seg_array->lock, LW_EXCLUSIVE);
    uint32_t segment_idx = reserve_merge_segment_slot(seg_array);
    LWLockRelease(seg_array->lock);

    seg_array->segments[segment_idx].start_sid = start_sid;
    seg_array->segments[segment_idx].end_sid = end_sid;
    seg_array->segments[segment_idx].vec_count = vec_count;
    seg_array->segments[segment_idx].index_type = index_type;
    seg_array->segments[segment_idx].deletion_ratio = deletion_ratio;

    LWLockAcquire(seg_array->lock, LW_EXCLUSIVE);
    // Set the index relid for this segment array if not already set
    if (seg_array->current_index_relid == InvalidOid)
    {
        seg_array->current_index_relid = index_relid;
    }
    else 
    {
        // check if the index relid is the same
        Assert(seg_array->current_index_relid == index_relid);
    }
    register_merge_segment(seg_array, segment_idx);
    LWLockRelease(seg_array->lock);
}

// Merge operations

// rebuild index (for segments whose index_type is FLAT or has high deletion ratio)
// this rebuilding will always filter out the vectors with deletion for both SEGMENT_UPDATE_REBUILD_DELETION and SEGMENT_UPDATE_REBUILD_FLAT
static void 
rebuild_index(MergeTaskData *task, IndexType target_type)
{
    elog(DEBUG1, "enter rebuild_index for slot %u, segment %u, index_type = %d, delete_ratio = %f, target_type = %d", 
        task->lsm_idx, task->segment_idx0, merge_worker_manager->segment_arrays[task->lsm_idx].segments[task->segment_idx0].index_type, 
        merge_worker_manager->segment_arrays[task->lsm_idx].segments[task->segment_idx0].deletion_ratio, target_type);
    
    SharedSegmentArray *seg_array = &merge_worker_manager->segment_arrays[task->lsm_idx];
    MergeSegmentInfo *segment = &seg_array->segments[task->segment_idx0];
    Oid index_relid = seg_array->current_index_relid;
    
    // Load the existing index from disk
    void *old_index_ptr = NULL;
    load_index_file(index_relid, segment->start_sid, segment->end_sid, segment->index_type, &old_index_ptr);
    if (old_index_ptr == NULL) {
        elog(ERROR, "rebuild_index: failed to load index from disk for segment %u-%u", 
             segment->start_sid, segment->end_sid);
        return;
    }
    
    // Load the bitmap from disk
    uint8_t *old_bitmap_ptr = NULL;
    load_bitmap_file(index_relid, segment->start_sid, segment->end_sid, &old_bitmap_ptr);
    if (old_bitmap_ptr == NULL) {
        elog(ERROR, "rebuild_index: failed to load bitmap from disk for segment %u-%u", 
             segment->start_sid, segment->end_sid);
        IndexFree(old_index_ptr);
        return;
    }
    
    // Create new index with filtered vectors
    void *new_index_ptr = NULL;
    int new_index_count = 0;
    
    // Get index parameters from LSM index metadata
    IndexType current_index_type;
    uint32_t dim, elem_size;
    if (!read_lsm_index_metadata(index_relid, &current_index_type, &dim, &elem_size)) {
        elog(ERROR, "rebuild_index: failed to read LSM index metadata for index %u", index_relid);
        IndexFree(old_index_ptr);
        pfree(old_bitmap_ptr);
        return;
    }
    
    // TODO: make the parameters configurable
    // Default parameters for new index (could be made configurable)
    int M = 16;                    // HNSW M parameter
    int efConstruction = 200;      // HNSW efConstruction parameter
    int lists = 1024;              // IVFFLAT nlist parameter
    
    // Call MergeIndex to rebuild with bitmap filtering
    MergeIndex(old_index_ptr, old_bitmap_ptr, segment->vec_count, 
               segment->index_type, target_type, 
               &new_index_ptr, &new_index_count,
               M, efConstruction, lists);
    
    if (new_index_ptr == NULL) {
        elog(ERROR, "rebuild_index: MergeIndex failed for segment %u-%u", 
             segment->start_sid, segment->end_sid);
        IndexFree(old_index_ptr);
        pfree(old_bitmap_ptr);
        return;
    }
    
    // Serialize the new index
    void *new_index_bin = NULL;
    IndexSerialize(new_index_ptr, &new_index_bin);
    if (new_index_bin == NULL) {
        elog(ERROR, "rebuild_index: failed to serialize new index for segment %u-%u", 
             segment->start_sid, segment->end_sid);
        IndexFree(old_index_ptr);
        IndexFree(new_index_ptr);
        pfree(old_bitmap_ptr);
        return;
    }
    
    // Generate new bitmap since all vectors in rebuilt index are valid (deleted ones filtered out)
    uint8_t *new_bitmap_ptr = (uint8_t *) palloc0(GET_BITMAP_SIZE(new_index_count));
    if (new_bitmap_ptr == NULL) {
        elog(ERROR, "rebuild_index: failed to allocate memory for new bitmap");
        IndexFree(old_index_ptr);
        IndexFree(new_index_ptr);
        pfree(old_bitmap_ptr);
        return;
    }
    // Set all bits to 1 since all vectors in the rebuilt index are valid
    memset(new_bitmap_ptr, 0x00, GET_BITMAP_SIZE(new_index_count));

    // Update the segment information in task
    task->merged_index_type = target_type;
    task->merged_vec_count = new_index_count;
    task->merged_deletion_ratio = 0.0f; // Reset deletion ratio after rebuild
    
    // Prepare flush metadata
    PrepareFlushMetaData prep;
    prep.start_sid = task->merged_start_sid;
    prep.end_sid = task->merged_end_sid;
    prep.valid_rows = new_index_count;
    prep.index_type = target_type;
    prep.index_bin = new_index_bin;
    prep.bitmap_ptr = new_bitmap_ptr;
    prep.bitmap_size = GET_BITMAP_SIZE(new_index_count); // Use new count for bitmap size
    prep.map_ptr = NULL; // Mapping not needed for rebuild
    prep.map_size = 0;
    
    // Use flush_segment_to_disk to write everything to disk
    flush_segment_to_disk(index_relid, &prep);
    
    // Cleanup
    IndexFree(old_index_ptr);
    IndexFree(new_index_ptr);
    pfree(old_bitmap_ptr);
    pfree(new_bitmap_ptr);
    
    elog(DEBUG1, "rebuild_index: successfully rebuilt index for segment %u-%u, new count = %d", 
         segment->start_sid, segment->end_sid, new_index_count);
}

// merge two adjacent segments into one
// this merging will not filter out the vectors with deletion
static void 
merge_adjacent_segments(MergeTaskData *task)
{
    elog(DEBUG1, "enter merge_adjacent_segments for slot %u, segments %u and %u", 
         task->lsm_idx, task->segment_idx0, task->segment_idx1);
    
    SharedSegmentArray *seg_array = &merge_worker_manager->segment_arrays[task->lsm_idx];
    MergeSegmentInfo *segment0 = &seg_array->segments[task->segment_idx0];
    MergeSegmentInfo *segment1 = &seg_array->segments[task->segment_idx1];
    Oid index_relid = seg_array->current_index_relid;
    
    // Load both indices from disk
    void *index0_ptr = NULL;
    load_index_file(index_relid, segment0->start_sid, segment0->end_sid, segment0->index_type, &index0_ptr);
    if (index0_ptr == NULL) {
        elog(ERROR, "merge_adjacent_segments: failed to load index0 from disk for segment %u-%u", 
             segment0->start_sid, segment0->end_sid);
        return;
    }
    
    void *index1_ptr = NULL;
    load_index_file(index_relid, segment1->start_sid, segment1->end_sid, segment1->index_type, &index1_ptr);
    if (index1_ptr == NULL) {
        elog(ERROR, "merge_adjacent_segments: failed to load index1 from disk for segment %u-%u", 
             segment1->start_sid, segment1->end_sid);
        IndexFree(index0_ptr);
        return;
    }
    
    // Load both bitmaps from disk
    uint8_t *bitmap0_ptr = NULL;
    load_bitmap_file(index_relid, segment0->start_sid, segment0->end_sid, &bitmap0_ptr);
    if (bitmap0_ptr == NULL) {
        elog(ERROR, "merge_adjacent_segments: failed to load bitmap0 from disk for segment %u-%u", 
             segment0->start_sid, segment0->end_sid);
        IndexFree(index0_ptr);
        IndexFree(index1_ptr);
        return;
    }
    
    uint8_t *bitmap1_ptr = NULL;
    load_bitmap_file(index_relid, segment1->start_sid, segment1->end_sid, &bitmap1_ptr);
    if (bitmap1_ptr == NULL) {
        elog(ERROR, "merge_adjacent_segments: failed to load bitmap1 from disk for segment %u-%u", 
             segment1->start_sid, segment1->end_sid);
        IndexFree(index0_ptr);
        IndexFree(index1_ptr);
        pfree(bitmap0_ptr);
        return;
    }
    
    // Merge the two indices
    void *merged_index_ptr = NULL;
    uint8_t *merged_bitmap_ptr = NULL;
    int merged_count = 0;
    IndexType merged_index_type;
    float merged_deletion_ratio;
    
    // TODO: make the parameters configurable
    MergeTwoIndices(index0_ptr, bitmap0_ptr, segment0->vec_count, segment0->index_type, segment0->deletion_ratio,
                    index1_ptr, bitmap1_ptr, segment1->vec_count, segment1->index_type, segment1->deletion_ratio,
                    &merged_index_ptr, &merged_bitmap_ptr, &merged_count,
                    &merged_index_type, &merged_deletion_ratio);
    
    if (merged_index_ptr == NULL) {
        elog(ERROR, "merge_adjacent_segments: MergeTwoIndices failed for segments %u-%u and %u-%u", 
             segment0->start_sid, segment0->end_sid, segment1->start_sid, segment1->end_sid);
        IndexFree(index0_ptr);
        IndexFree(index1_ptr);
        pfree(bitmap0_ptr);
        pfree(bitmap1_ptr);
        return;
    }
    
    // Serialize the merged index
    void *merged_index_bin = NULL;
    IndexSerialize(merged_index_ptr, &merged_index_bin);
    if (merged_index_bin == NULL) {
        elog(ERROR, "merge_adjacent_segments: failed to serialize merged index");
        IndexFree(index0_ptr);
        IndexFree(index1_ptr);
        IndexFree(merged_index_ptr);
        pfree(bitmap0_ptr);
        pfree(bitmap1_ptr);
        pfree(merged_bitmap_ptr);
        return;
    }
    
    // Update the segment information in task using values computed by MergeTwoIndices
    task->merged_index_type = merged_index_type;
    task->merged_vec_count = merged_count;
    task->merged_deletion_ratio = merged_deletion_ratio;
    
    // Prepare flush metadata for the merged segment
    PrepareFlushMetaData prep;
    prep.start_sid = task->merged_start_sid;
    prep.end_sid = task->merged_end_sid;
    prep.valid_rows = merged_count;
    prep.index_type = merged_index_type; // Use the computed merged index type
    prep.index_bin = merged_index_bin;
    prep.bitmap_ptr = merged_bitmap_ptr;
    prep.bitmap_size = GET_BITMAP_SIZE(merged_count);
    prep.map_ptr = NULL; // Mapping not needed for merge
    prep.map_size = 0;
    
    // Write the merged segment to disk
    flush_segment_to_disk(index_relid, &prep);
    
    // Cleanup
    IndexFree(index0_ptr);
    IndexFree(index1_ptr);
    IndexFree(merged_index_ptr);
    pfree(bitmap0_ptr);
    pfree(bitmap1_ptr);
    pfree(merged_bitmap_ptr);
    
    elog(DEBUG1, "merge_adjacent_segments: successfully merged segments %u-%u and %u-%u into %u-%u, merged count = %d", 
         segment0->start_sid, segment0->end_sid, segment1->start_sid, segment1->end_sid,
         task->merged_start_sid, task->merged_end_sid, merged_count);
}

// Claim a merge task
static bool
claim_merge_task(int worker_id, int lsm_idx, uint32_t segment_idx0, uint32_t segment_idx1, int task_type)
{
    SharedSegmentArray *seg_array = &merge_worker_manager->segment_arrays[lsm_idx];
    LWLockAcquire(seg_array->lock, LW_EXCLUSIVE);

    // Validate task type requirements
    if (task_type == SEGMENT_UPDATE_MERGE)
    {
        // For merge operations, both segments must be valid
        if (segment_idx0 < 0 || segment_idx0 >= MAX_SEGMENTS_COUNT || segment_idx1 == -1)
        {
            LWLockRelease(seg_array->lock);
            return false;
        }
    }
    else if (task_type == SEGMENT_UPDATE_REBUILD_FLAT || task_type == SEGMENT_UPDATE_REBUILD_DELETION)
    {
        // For rebuild operations, segment_idx1 must be -1
        if (segment_idx0 < 0 || segment_idx0 >= MAX_SEGMENTS_COUNT || segment_idx1 != -1)
        {
            LWLockRelease(seg_array->lock);
            return false;
        }
    }

    // check if the segment is used (segment_idx1 might be -1)
    if (!seg_array->segments[segment_idx0].in_used || (segment_idx1 != -1 && !seg_array->segments[segment_idx1].in_used))
    {
        LWLockRelease(seg_array->lock);
        return false;
    }
    // check if the segment is already claimed
    if (seg_array->segments[segment_idx0].is_compacting || (segment_idx1 != -1 && seg_array->segments[segment_idx1].is_compacting))
    {
        LWLockRelease(seg_array->lock);
        return false;
    }

    // claim the task
    seg_array->segments[segment_idx0].is_compacting = true;
    if (segment_idx1 != -1)
    {
        seg_array->segments[segment_idx1].is_compacting = true;
    }

    LWLockRelease(seg_array->lock);

    // update the task
    merge_worker_manager->workers[worker_id].current_task.operation_type = task_type;
    merge_worker_manager->workers[worker_id].current_task.segment_idx0 = segment_idx0;
    merge_worker_manager->workers[worker_id].current_task.segment_idx1 = segment_idx1;
    merge_worker_manager->workers[worker_id].current_task.merged_start_sid = seg_array->segments[segment_idx0].start_sid;
    // Set end_sid based on task type
    if (task_type == SEGMENT_UPDATE_MERGE)
    {
        // For merge operations, use the end_sid of the target segment (segment_idx1)
        merge_worker_manager->workers[worker_id].current_task.merged_end_sid = seg_array->segments[segment_idx1].end_sid;
    }
    else
    {
        // For rebuild operations, use the end_sid of the segment being rebuilt (segment_idx0)
        merge_worker_manager->workers[worker_id].current_task.merged_end_sid = seg_array->segments[segment_idx0].end_sid;
    }
    merge_worker_manager->workers[worker_id].current_task.start_time = GetCurrentTimestamp();
    merge_worker_manager->workers[worker_id].current_task.lsm_idx = lsm_idx;
    return true;
}

// TODO: we also need to handle the case that the task is not conducted successfully
// finish the task
static void
finish_merge_task(int worker_id)
{
    MergeWorkerState *worker = &merge_worker_manager->workers[worker_id];
    MergeTaskData *task = &worker->current_task;
    
    if (!worker->active || task->operation_type == -1)
    {
        elog(WARNING, "[finish_merge_task] Worker %d has no active task to finish", worker_id);
        return;
    }
    
    elog(DEBUG1, "[finish_merge_task] Finishing task for worker %d, operation_type = %d", 
         worker_id, task->operation_type);
    
    // Get the segment array
    SharedSegmentArray *seg_array = &merge_worker_manager->segment_arrays[task->lsm_idx];
    
    // step 1: notify the vector index worker that the task is finished
    // Send completion notification to vector index worker
    segment_update_blocking(task->lsm_idx, seg_array->current_index_relid, task->operation_type, task->merged_start_sid, task->merged_end_sid);
    elog(DEBUG1, "[finish_merge_task] Completion notification sent to vector index worker");
    
    // step 2: update the segment information in the segment array
    LWLockAcquire(seg_array->lock, LW_EXCLUSIVE);
    
    switch (task->operation_type)
    {
        case SEGMENT_UPDATE_REBUILD_FLAT:
        case SEGMENT_UPDATE_REBUILD_DELETION:
        {
            // For rebuild operations, we only need to update the segment information of the segment being rebuilt
            uint32_t segment_idx = task->segment_idx0;
            MergeSegmentInfo *segment = &seg_array->segments[segment_idx];
            
            if (segment->in_used && segment->is_compacting)
            {
                // Update statistics: remove old values before updating segment
                update_statistics_on_remove(seg_array, segment->index_type, segment->vec_count);
                
                // Update segment metadata with merged information
                segment->index_type = task->merged_index_type;
                segment->vec_count = task->merged_vec_count;
                segment->deletion_ratio = task->merged_deletion_ratio;
                segment->is_compacting = false;
                
                // Update statistics: add new values after updating segment
                update_statistics_on_add(seg_array, segment->index_type, segment->vec_count);
                
                elog(DEBUG1, "[finish_merge_task] Rebuild completed for segment %u", segment_idx);
            }
            else {
                elog(ERROR, "[finish_merge_task] Rebuild failed for segment %u", segment_idx);
            }
            break;
        }
        
        case SEGMENT_UPDATE_MERGE:
        {
            // For merge operations, we need to update the segment information of the first segment with the merged segment information
            // and remove the second segment information from the segment array
            uint32_t segment_idx0 = task->segment_idx0;
            uint32_t segment_idx1 = task->segment_idx1;
            
            MergeSegmentInfo *segment0 = &seg_array->segments[segment_idx0];
            MergeSegmentInfo *segment1 = &seg_array->segments[segment_idx1];
            
            if (segment0->in_used && segment0->is_compacting && 
                segment1->in_used && segment1->is_compacting)
            {
                // Update statistics: remove old values from segment0 before updating
                update_statistics_on_remove(seg_array, segment0->index_type, segment0->vec_count);
                
                // Update segment0 to represent the merged segment
                segment0->end_sid = task->merged_end_sid;
                segment0->vec_count = task->merged_vec_count;
                segment0->index_type = task->merged_index_type;
                segment0->deletion_ratio = task->merged_deletion_ratio;
                segment0->is_compacting = false;
                
                // Update statistics: add new values for segment0 after updating
                update_statistics_on_add(seg_array, segment0->index_type, segment0->vec_count);
                
                // Remove segment1 from the segment array (this will update statistics for segment1)
                remove_from_segment_array(seg_array, segment_idx1);
                
                elog(DEBUG1, "[finish_merge_task] Merge completed: segment %u merged into segment %u", 
                     segment_idx1, segment_idx0);
            }
            else {
                elog(ERROR, "[finish_merge_task] Merge failed for segments %u and %u", segment_idx0, segment_idx1);
            }
            break;
        }
        
        default:
            elog(WARNING, "[finish_merge_task] Unknown operation type %d", task->operation_type);
            break;
    }
    
    LWLockRelease(seg_array->lock);
    
    // step 3: reset `is_compacting` flag for the segment being rebuilt or merged
    // (This is already done above in the switch statement)
    
    // Reset worker status
    worker->current_task.operation_type = -1;
    worker->current_task.segment_idx0 = -1;
    worker->current_task.segment_idx1 = -1;
    worker->current_task.lsm_idx = -1;
    worker->current_task.start_time = 0;
    worker->current_task.merged_start_sid = -1;
    worker->current_task.merged_end_sid = -1;
    worker->current_task.merged_vec_count = 0;
    worker->current_task.merged_index_type = -1;
    worker->current_task.merged_deletion_ratio = 0.0f;
    
    elog(DEBUG1, "[finish_merge_task] Task finished for worker %d", worker_id);
}

// Choose an adjacent segment (prev or next) that is in use and not compacting.
// Prefer the one with smaller vec_count when both exist.
// (Need to hold the shared lock)
static bool
choose_adjacent_smaller(SharedSegmentArray *seg_array, uint32_t idx, uint32_t *chosen_adj_idx)
{
    uint32_t prev_idx = seg_array->segments[idx].prev_idx;
    uint32_t next_idx = seg_array->segments[idx].next_idx;
    bool have_prev = (prev_idx != (uint32_t)-1) && seg_array->segments[prev_idx].in_used && !seg_array->segments[prev_idx].is_compacting;
    bool have_next = (next_idx != (uint32_t)-1) && seg_array->segments[next_idx].in_used && !seg_array->segments[next_idx].is_compacting;

    if (!have_prev && !have_next)
        return false;
    if (have_prev && !have_next)
    {
        *chosen_adj_idx = prev_idx;
        return true;
    }
    if (!have_prev && have_next)
    {
        *chosen_adj_idx = next_idx;
        return true;
    }

    if (seg_array->segments[prev_idx].vec_count <= seg_array->segments[next_idx].vec_count)
        *chosen_adj_idx = prev_idx;
    else
        *chosen_adj_idx = next_idx;
    return true;
}

/*
 * Scan all segment arrays and try to claim one task for current worker
 * Priority (high -> low):
 * 1) FLAT segments => SEGMENT_UPDATE_REBUILD_FLAT
 * 2) vec_count <= MEMTABLE_MAX_CAPACITY => SEGMENT_UPDATE_MERGE with adjacent smaller
 * 3) deletion_ratio > MERGE_DELETION_RATIO_THRESHOLD => SEGMENT_UPDATE_REBUILD_DELETION
 * 4) vec_count <= THRESHOLD_SMALL_SEGMENT_SIZE => SEGMENT_UPDATE_MERGE with adjacent smaller
 * 5) vec_count <= MAX_SEGMENTS_SIZE => SEGMENT_UPDATE_MERGE with adjacent smaller
 */
// Helper function to traverse and check segments for a specific priority
static bool
traverse_and_check_priority(int worker_id, int lsm_idx, int priority_type)
{
    uint32_t adj;
    
    SharedSegmentArray *seg_array = &merge_worker_manager->segment_arrays[lsm_idx];
    LWLockAcquire(seg_array->lock, LW_SHARED);
    if (!seg_array->segments[0].in_used)
    {
        LWLockRelease(seg_array->lock);
        return false;
    }
    
    uint32_t cur = 0;
    while (true)
    {
        if (!seg_array->segments[cur].in_used)
        {
            uint32_t next = seg_array->segments[cur].next_idx;
            if (next == (uint32_t)-1)
                break;
            cur = next;
            continue;
        }
        
        bool should_claim = false;
        uint32_t adj_segment = (uint32_t)-1;
        
        switch (priority_type)
        {
            case 1: // FLAT segments
                if (seg_array->segments[cur].index_type == FLAT)
                {
                    should_claim = true;
                }
                break;
                
            case 2: // vec_count <= MEMTABLE_MAX_CAPACITY
                if (seg_array->segments[cur].vec_count <= MEMTABLE_MAX_CAPACITY)
                {
                    if (choose_adjacent_smaller(seg_array, cur, &adj_segment))
                    {
                        should_claim = true;
                    }
                }
                break;
                
            case 3: // deletion_ratio > MERGE_DELETION_RATIO_THRESHOLD
                if (seg_array->segments[cur].deletion_ratio > MERGE_DELETION_RATIO_THRESHOLD)
                {
                    should_claim = true;
                }
                break;
                
            case 4: // vec_count <= THRESHOLD_SMALL_SEGMENT_SIZE
                if (seg_array->segments[cur].vec_count <= THRESHOLD_SMALL_SEGMENT_SIZE)
                {
                    if (choose_adjacent_smaller(seg_array, cur, &adj_segment))
                    {
                        should_claim = true;
                    }
                }
                break;
                
            case 5: // vec_count <= MAX_SEGMENTS_SIZE
                if (seg_array->segments[cur].vec_count <= MAX_SEGMENTS_SIZE)
                {
                    if (choose_adjacent_smaller(seg_array, cur, &adj_segment))
                    {
                        should_claim = true;
                    }
                }
                break;
        }
        
        if (should_claim)
        {
            LWLockRelease(seg_array->lock);
            
            // Determine task type
            int task_type;
            if (priority_type == 1)
                task_type = SEGMENT_UPDATE_REBUILD_FLAT;
            else if (priority_type == 3)
                task_type = SEGMENT_UPDATE_REBUILD_DELETION;
            else
                task_type = SEGMENT_UPDATE_MERGE;
            
            if (claim_merge_task(worker_id, lsm_idx, cur, adj_segment, task_type))
                return true;
                
            // Re-acquire lock to continue traversal
            LWLockAcquire(seg_array->lock, LW_SHARED);
            
            // Check if cur is still valid after re-acquiring lock
            // If not, reset to 0 to restart traversal
            if (!seg_array->segments[cur].in_used)
            {
                cur = 0;
                continue;
            }
        }
        
        uint32_t next = seg_array->segments[cur].next_idx;
        if (next == (uint32_t)-1)
            break;
        cur = next;
    }
    
    LWLockRelease(seg_array->lock);
    return false;
}

static bool
scan_and_claim_merge_task(int worker_id)
{
    int lsm_idx;
    
    /* Priority 1: FLAT segments => rebuild flat (highest priority) */
    for (lsm_idx = 0; lsm_idx < INDEX_BUF_SIZE; lsm_idx++)
    {
        SharedSegmentArray *seg_array = &merge_worker_manager->segment_arrays[lsm_idx];
        
        if (seg_array->current_index_relid == InvalidOid)
            continue;
            
        uint32 flat_cnt = pg_atomic_read_u32(&seg_array->flat_count);
        if (flat_cnt == 0)
            continue;
            
        if (traverse_and_check_priority(worker_id, lsm_idx, 1))
            return true;
    }
    
    /* Priority 2: vec_count <= MEMTABLE_MAX_CAPACITY => merge with adjacent smaller */
    for (lsm_idx = 0; lsm_idx < INDEX_BUF_SIZE; lsm_idx++)
    {
        SharedSegmentArray *seg_array = &merge_worker_manager->segment_arrays[lsm_idx];
        
        if (seg_array->current_index_relid == InvalidOid)
            continue;
            
        uint32 mem_le_cnt = pg_atomic_read_u32(&seg_array->memtable_capacity_le_count);
        if (mem_le_cnt == 0)
            continue;
            
        if (traverse_and_check_priority(worker_id, lsm_idx, 2))
            return true;
    }
    
    /* Priority 3: deletion_ratio > MERGE_DELETION_RATIO_THRESHOLD => rebuild deletion */
    for (lsm_idx = 0; lsm_idx < INDEX_BUF_SIZE; lsm_idx++)
    {
        SharedSegmentArray *seg_array = &merge_worker_manager->segment_arrays[lsm_idx];
        
        if (seg_array->current_index_relid == InvalidOid)
            continue;
            
        if (traverse_and_check_priority(worker_id, lsm_idx, 3))
            return true;
    }
    
    /* Priority 4: vec_count <= THRESHOLD_SMALL_SEGMENT_SIZE => merge with adjacent smaller */
    for (lsm_idx = 0; lsm_idx < INDEX_BUF_SIZE; lsm_idx++)
    {
        SharedSegmentArray *seg_array = &merge_worker_manager->segment_arrays[lsm_idx];
        
        if (seg_array->current_index_relid == InvalidOid)
            continue;
            
        uint32 small_le_cnt = pg_atomic_read_u32(&seg_array->small_segment_le_count);
        if (small_le_cnt == 0)
            continue;
            
        if (traverse_and_check_priority(worker_id, lsm_idx, 4))
            return true;
    }
    
    /* Priority 5: vec_count <= MAX_SEGMENTS_SIZE => merge with adjacent smaller (lowest priority) */
    for (lsm_idx = 0; lsm_idx < INDEX_BUF_SIZE; lsm_idx++)
    {
        SharedSegmentArray *seg_array = &merge_worker_manager->segment_arrays[lsm_idx];
        
        if (seg_array->current_index_relid == InvalidOid)
            continue;
            
        if (traverse_and_check_priority(worker_id, lsm_idx, 5))
            return true;
    }

    return false;
}

// Register this worker
static void
register_merge_worker(int worker_id)
{
    LWLockAcquire(merge_worker_manager->lock, LW_EXCLUSIVE);
    
    if (worker_id >= 0 && worker_id < MAX_MERGE_WORKERS)
    {
        merge_worker_manager->workers[worker_id].active = true;
        merge_worker_manager->active_worker_count++;
        current_worker_id = worker_id;
        
        elog(DEBUG1, "[register_merge_worker] Registered worker %d", worker_id);
    }
    
    LWLockRelease(merge_worker_manager->lock);
}

// Unregister this worker
static void
unregister_merge_worker(int worker_id)
{
    LWLockAcquire(merge_worker_manager->lock, LW_EXCLUSIVE);
    
    if (worker_id >= 0 && worker_id < MAX_MERGE_WORKERS)
    {
        merge_worker_manager->workers[worker_id].active = false;
        merge_worker_manager->active_worker_count--;
        
        elog(DEBUG1, "[unregister_merge_worker] Unregistered worker %d", worker_id);
    }
    
    LWLockRelease(merge_worker_manager->lock);
}

// TODO:
// Main background worker function
void
lsm_merge_worker_main(Datum main_arg)
{
    // Set up signal handling
    pqsignal(SIGTERM, merge_worker_sigterm);
    pqsignal(SIGHUP, merge_worker_sighup);
    BackgroundWorkerUnblockSignals();
    
    // Initialize worker manager if not already done
    if (merge_worker_manager == NULL)
        initialize_merge_worker_manager();
    
    // Register this worker
    int worker_id = DatumGetInt32(main_arg);
    register_merge_worker(worker_id);
    
    elog(LOG, "[lsm_merge_worker_main] Merge worker %d started", worker_id);
    
    while (!got_sigterm)
    {
        CHECK_FOR_INTERRUPTS();
        
        if (!OurPostmasterIsAlive())
        {
            elog(LOG, "[lsm_merge_worker_main] Postmaster is dead, exiting");
            break;
        }

        // Try to claim a merge task
        bool claimed = scan_and_claim_merge_task(worker_id);
        
        if (claimed)
        {
            MergeWorkerState *worker = &merge_worker_manager->workers[worker_id];
            MergeTaskData *task = &worker->current_task;
            
            elog(DEBUG1, "[lsm_merge_worker_main] Worker %d claimed task, operation_type = %d", 
                 worker_id, task->operation_type);
            
            // Execute the claimed task based on its operation type
            switch (task->operation_type)
            {
                case SEGMENT_UPDATE_REBUILD_FLAT:
                    // Rebuild segment with flat index
                    rebuild_index(task, FLAT);
                    break;
                    
                case SEGMENT_UPDATE_REBUILD_DELETION:
                    // Rebuild segment to remove deleted vectors
                    IndexType target_type = SharedLSMIndexBuffer->slots[task->lsm_idx].lsmIndex.index_type;
                    rebuild_index(task, target_type);
                    break;
                    
                case SEGMENT_UPDATE_MERGE:
                    // Merge two adjacent segments
                    merge_adjacent_segments(task);
                    break;
                    
                default:
                    elog(ERROR, "[lsm_merge_worker_main] Unknown operation type %d", task->operation_type);
                    break;
            }
            
            // Finish the task (sends notification and updates metadata)
            finish_merge_task(worker_id);
        }
        else
        {
            // No task claimed, sleep for a short time before retrying
            ResetLatch(MyLatch);
            WaitLatch(MyLatch, WL_LATCH_SET | WL_TIMEOUT, 100, 0);
        }
    }
    
    // Cleanup
    unregister_merge_worker(worker_id);
    elog(LOG, "[lsm_merge_worker_main] Merge worker %d shutting down", worker_id);
    
    proc_exit(0);
}