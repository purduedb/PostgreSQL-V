/*
 * Example implementation of segment merging with reference counting
 * This file demonstrates how to safely merge flushed segments using
 * the reference counting mechanism.
 */

#include "postgres.h"
#include "lsmindex.h"

/*
 * Example function to safely merge two segments
 * This would be called during background merging operations
 */
void
merge_flushed_segments_example(LSMIndex lsm, uint32_t seg1_idx, uint32_t seg2_idx, uint32_t new_seg_idx)
{
    FlushedSegment seg1 = &lsm->flushed_segments[seg1_idx];
    FlushedSegment seg2 = &lsm->flushed_segments[seg2_idx];
    FlushedSegment new_seg = &lsm->flushed_segments[new_seg_idx];
    
    elog(DEBUG1, "[merge_flushed_segments_example] Merging segments %d and %d into %d", 
         seg1_idx, seg2_idx, new_seg_idx);
    
    // Acquire exclusive lock to prevent concurrent searches
    LWLockAcquire(lsm->seg_lock, LW_EXCLUSIVE);
    
    // Check if segments are still valid and have zero reference count
    if (!seg1->in_used || !seg2->in_used) {
        elog(WARNING, "[merge_flushed_segments_example] One or both segments are not in use");
        LWLockRelease(lsm->seg_lock);
        return;
    }
    
    if (pg_atomic_read_u32(&seg1->ref_count) != 0 || pg_atomic_read_u32(&seg2->ref_count) != 0) {
        elog(WARNING, "[merge_flushed_segments_example] Segments still have active references");
        LWLockRelease(lsm->seg_lock);
        return;
    }
    
    // Perform the actual merge operation here
    // This would involve:
    // 1. Creating a new merged index
    // 2. Loading data from both segments
    // 3. Building the merged index
    // 4. Persisting to disk
    
    // For this example, we'll just simulate the merge
    elog(DEBUG1, "[merge_flushed_segments_example] Performing merge operation...");
    
    // After successful merge, safely delete the old segments
    cleanup_flushed_segment(lsm, seg1_idx);
    cleanup_flushed_segment(lsm, seg2_idx);
    
    // Register the new merged segment
    register_flushed_segment(lsm, new_seg_idx);
    
    LWLockRelease(lsm->seg_lock);
    
    elog(DEBUG1, "[merge_flushed_segments_example] Merge completed successfully");
}

/*
 * Example function to safely delete a segment when merging is not possible
 * This would be called when a segment needs to be removed without merging
 */
void
delete_flushed_segment_example(LSMIndex lsm, uint32_t seg_idx)
{
    FlushedSegment segment = &lsm->flushed_segments[seg_idx];
    
    elog(DEBUG1, "[delete_flushed_segment_example] Attempting to delete segment %d", seg_idx);
    
    // Acquire exclusive lock
    LWLockAcquire(lsm->seg_lock, LW_EXCLUSIVE);
    
    // Check if segment is still valid
    if (!segment->in_used) {
        elog(WARNING, "[delete_flushed_segment_example] Segment %d is not in use", seg_idx);
        LWLockRelease(lsm->seg_lock);
        return;
    }
    
    // Check if segment has active references
    if (pg_atomic_read_u32(&segment->ref_count) != 0) {
        elog(WARNING, "[delete_flushed_segment_example] Segment %d still has active references, cannot delete", seg_idx);
        LWLockRelease(lsm->seg_lock);
        return;
    }
    
    // Safe to delete
    cleanup_flushed_segment(lsm, seg_idx);
    
    LWLockRelease(lsm->seg_lock);
    
    elog(DEBUG1, "[delete_flushed_segment_example] Segment %d deleted successfully", seg_idx);
}
