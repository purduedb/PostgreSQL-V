#include "lsm_segment.h"
#include "vectorindeximpl.hpp"

static FlushedSegmentPool segment_pool[INDEX_BUF_SIZE];

FlushedSegmentPool *get_flushed_segment_pool(int idx)
{
    return &segment_pool[idx];
}

// ------------------------ flushed segment ------------------------
void initialize_segment_pool(FlushedSegmentPool *pool)
{
    pool->flushed_segment_count = 0;
    pool->head_idx = -1;
    pool->tail_idx = -1;
    pool->insert_idx = 0;
    for(int i = 0; i < MAX_SEGMENTS_COUNT; ++i)
    {
        pool->flushed_segments[i].in_used = false;
        pool->flushed_segments[i].next_idx = -1;
        pool->flushed_segments[i].prev_idx = -1;
        atomic_store(&pool->flushed_segments[i].ref_count, 0);
    }
}
// need to acquire the lock first
uint32_t
reserve_flushed_segment(FlushedSegmentPool *pool)
{
    elog(DEBUG1, "enter reserve_flushed_segment, insert_idx = %d", pool->insert_idx);
    for(int idx = pool->insert_idx; idx < MAX_SEGMENTS_COUNT; ++idx)
    {
        if(!pool->flushed_segments[idx].in_used)
        {
            pool->insert_idx = idx + 1;
            pool->flushed_segments[idx].in_used = true;
            elog(DEBUG1, "reserve_flushed_segment successfully, idx = %d", idx);
            return idx;
        }
    }
    return -1;
}

void
register_flushed_segment(FlushedSegmentPool *pool, uint32_t idx)
{
    elog(DEBUG1, "enter register_flushed_segment for idx = %d", idx);
    
    // Initialize reference count for the new segment
    atomic_store(&pool->flushed_segments[idx].ref_count, 1);
    
    // Link to the end of the list
    uint32_t prev_tail = -1;
    if(pool->tail_idx != -1) {
        pool->flushed_segments[pool->tail_idx].next_idx = idx;
        prev_tail = pool->tail_idx;
    }
    pool->flushed_segments[idx].prev_idx = prev_tail;
    pool->flushed_segments[idx].next_idx = -1;
    pool->tail_idx = idx;
    
    // Update head_idx if this is the first segment
    if (pool->head_idx == -1)
    {
        pool->head_idx = idx;
    }
    
    ++pool->flushed_segment_count;
    elog(DEBUG1, "register_flushed_segment successfully added to list, idx = %d", idx);
}

// Reference counting functions for flushed segments
// This can only be called by the vector index worker
void
decrement_flushed_segment_ref_count(FlushedSegmentPool *pool, uint32_t segment_idx)
{
    FlushedSegment segment = &pool->flushed_segments[segment_idx];
    
    // Just decrement without checking
    (void) atomic_fetch_sub(&segment->ref_count, 1);
    // if the reference count is 0, cleanup the segment
    if (atomic_load(&segment->ref_count) == 0) {
        cleanup_flushed_segment(pool, segment_idx);
    }
}

void
cleanup_flushed_segment(FlushedSegmentPool *pool, uint32_t segment_idx)
{
    FlushedSegment segment = &pool->flushed_segments[segment_idx];
    
    elog(DEBUG1, "[cleanup_flushed_segment] Cleaning up segment idx = %d", segment_idx);
    
    // Free the index, bitmap, and map
    if (segment->index_ptr != NULL)
    {
        IndexFree(segment->index_ptr);
        segment->index_ptr = NULL;
    }
    if (segment->bitmap_ptr != NULL)
    {
        free(segment->bitmap_ptr);
        segment->bitmap_ptr = NULL;
    }
    if (segment->map_ptr != NULL)
    {
        free(segment->map_ptr);
        segment->map_ptr = NULL;
    }

    // No need to unlink the segment in this function, the segment is already unlinked when the reference count reaches 0
    // Mark segment as unused
    segment->in_used = false;
    segment->next_idx = -1;
    segment->prev_idx = -1;
    
    // Update segment count
    --pool->flushed_segment_count;
    
    atomic_store(&segment->ref_count, 0);
    
    elog(DEBUG1, "[cleanup_flushed_segment] Segment idx = %d cleaned up and unlinked from list", segment_idx);
}

// Find a segment by its segment IDs
// Returns the segment index if found, -1 otherwise
// Note: requires lock to be held
uint32_t
find_segment_by_sids(FlushedSegmentPool *pool, SegmentId start_sid, SegmentId end_sid)
{
    // Iterate through all segments to find matching one
    for (uint32_t idx = 0; idx < MAX_SEGMENTS_COUNT; idx++)
    {
        FlushedSegment seg = &pool->flushed_segments[idx];
        if (seg->in_used && 
            seg->segment_id_start == start_sid && 
            seg->segment_id_end == end_sid)
        {
            elog(DEBUG1, "[find_segment_by_ids] Found segment %u-%u at idx %u", start_sid, end_sid, idx);
            return idx;
        }
    }
    
    elog(ERROR, "[find_segment_by_ids] Segment %u-%u not found", start_sid, end_sid);
    return -1;
}

// Find two adjacent segments that together cover the target range
// Returns the indices in seg_idx_0 and seg_idx_1
// Note: requires lock to be held
void
find_two_adjacent_segments(FlushedSegmentPool *pool, SegmentId target_start_sid, SegmentId target_end_sid,
                           uint32_t *seg_idx_0, uint32_t *seg_idx_1)
{
    elog(DEBUG1, "[find_two_adjacent_segments] Looking for segments covering range %u-%u", 
         target_start_sid, target_end_sid);
    
    // Iterate through the linked list to find the segments
    uint32_t idx = pool->head_idx;
    
    while (idx != -1)
    {
        FlushedSegment seg = &pool->flushed_segments[idx];
        
        if (seg->in_used)
        {
            // Check if this segment starts at or before the target start
            if (seg->segment_id_start == target_start_sid)
            {
                // This is the first segment, now check if it needs an adjacent segment
                if (seg->segment_id_end >= target_end_sid)
                {
                    // Single segment covers the entire range - error case
                    elog(ERROR, "[find_two_adjacent_segments] Single segment %u-%u already covers range %u-%u, should use single segment replacement", 
                         seg->segment_id_start, seg->segment_id_end, target_start_sid, target_end_sid);
                    *seg_idx_0 = -1;
                    *seg_idx_1 = -1;
                    return;
                }
                
                // Need to find the adjacent segment
                uint32_t next_idx = seg->next_idx;
                if (next_idx == -1)
                {
                    elog(ERROR, "[find_two_adjacent_segments] No adjacent segment found for range %u-%u", 
                         target_start_sid, target_end_sid);
                    *seg_idx_0 = -1;
                    *seg_idx_1 = -1;
                    return;
                }
                
                FlushedSegment next_seg = &pool->flushed_segments[next_idx];
                if (next_seg->segment_id_end == target_end_sid)
                {
                    // Found the two segments
                    *seg_idx_0 = idx;
                    *seg_idx_1 = next_idx;
                    elog(DEBUG1, "[find_two_adjacent_segments] Found segments: %u-%u and %u-%u covering range %u-%u",
                         seg->segment_id_start, seg->segment_id_end,
                         next_seg->segment_id_start, next_seg->segment_id_end,
                         target_start_sid, target_end_sid);
                    return;
                }
                else
                {
                    elog(ERROR, "[find_two_adjacent_segments] Adjacent segment %u-%u does not cover the range", 
                         next_seg->segment_id_start, next_seg->segment_id_end);
                    *seg_idx_0 = -1;
                    *seg_idx_1 = -1;
                    return;
                }
            }
        }
        else {
            elog(ERROR, "[find_two_adjacent_segments] No adjacent segments found for range %u-%u", 
                 target_start_sid, target_end_sid);
        }
        
        // Move to next segment
        if (idx == pool->tail_idx)
        {
            break;
        }
        idx = seg->next_idx;
    }
    
    elog(ERROR, "[find_two_adjacent_segments] Could not find segments covering range %u-%u", 
         target_start_sid, target_end_sid);
    *seg_idx_0 = -1;
    *seg_idx_1 = -1;
}

// Replace an old segment with a new segment atomically
// Unlinks the old segment and links the new segment in its place
// the order of the old segments is important
// The caller must hold the write lock
void
replace_flushed_segment(FlushedSegmentPool *pool, uint32_t old_seg_idx_0, uint32_t old_seg_idx_1, uint32_t new_seg_idx)
{
    elog(DEBUG1, "[replace_flushed_segment] Replacing old segment idx = %u with new segment idx = %u", 
         old_seg_idx_0, new_seg_idx);
    
    // Get the old segment's position before unlinking
    FlushedSegment old_seg_0 = &pool->flushed_segments[old_seg_idx_0];
    FlushedSegment old_seg_1 = old_seg_idx_1 != -1 ? &pool->flushed_segments[old_seg_idx_1] : NULL;
    FlushedSegment new_seg = &pool->flushed_segments[new_seg_idx];
    uint32_t prev_idx = old_seg_0->prev_idx;
    uint32_t next_idx = old_seg_idx_1 != -1 ? old_seg_1->next_idx : old_seg_0->next_idx;

    // Decrement ref count of old segment
    decrement_flushed_segment_ref_count(pool, old_seg_idx_0);
    if(old_seg_idx_1 != -1) {
        decrement_flushed_segment_ref_count(pool, old_seg_idx_1);
    }
    
    // 1. Update the head and the tail if necessary
    if (pool->head_idx == old_seg_idx_0)
    {
        // This was the head - update head_idx to new segment
        pool->head_idx = new_seg_idx;
    }
    if ((old_seg_idx_1 == -1 && pool->tail_idx == old_seg_idx_0) || (old_seg_idx_1 != -1 && pool->tail_idx == old_seg_idx_1))
    {
        // This was the tail - update tail_idx to new segment
        pool->tail_idx = new_seg_idx;
    }
    
    // 2. Set the prev and next of the new segment
    new_seg->prev_idx = prev_idx;
    new_seg->next_idx = next_idx;
    
    // Update the previous segment's next_idx to point to new segment
    if (prev_idx != -1)
    {
        pool->flushed_segments[prev_idx].next_idx = new_seg_idx;
    }
    
    // Update the next segment's prev_idx to point to new segment
    if (next_idx != -1)
    {
        pool->flushed_segments[next_idx].prev_idx = new_seg_idx;
    }
    
    // 3. Update the ref count of the new segment
    atomic_store(&new_seg->ref_count, 1);

    // update the flushed_segment_count if necessary
    if(old_seg_idx_1 != -1) {
        --pool->flushed_segment_count;
    }
    
    elog(DEBUG1, "[replace_flushed_segment] Successfully replaced segment %u with segment %u", 
         old_seg_idx_0, new_seg_idx);
}

// -------------------------------- IO ----------------------------------

// FIXME: concurrency issue
void 
load_all_segments_from_disk(Oid index_oid, FlushedSegmentPool *pool)
{
    SegmentFileInfo files[MAX_SEGMENTS_COUNT];
    int file_count = scan_segment_metadata_files(index_oid, files, MAX_SEGMENTS_COUNT);
    
    if (file_count == 0)
    {
        elog(DEBUG1, "[load_all_segments_from_disk] No segment metadata files found for index %u", index_oid);
        return;
    }
    
    elog(DEBUG1, "[load_all_segments_from_disk] Found %d segment metadata files", file_count);
    
    // Load segments in order
    for (int i = 0; i < file_count; i++)
    {
        uint32_t seg_idx = reserve_flushed_segment(pool);
        if (seg_idx == -1)
        {
            elog(ERROR, "no free flushed segment slot");
        }

        FlushedSegment segment = &pool->flushed_segments[seg_idx];
        load_and_set_segment(index_oid, seg_idx, segment, files[i].start_sid, files[i].end_sid);
        
        register_flushed_segment(pool, seg_idx);
    }

    elog(DEBUG1, "[load_all_segments_from_disk] Loaded %d segments, tail_idx = %d", file_count, pool->tail_idx);
}

// handled by vector index worker (for index build)
void 
load_and_set_segment(Oid indexRelId, uint32_t segment_idx, FlushedSegment segment, SegmentId start_sid, SegmentId end_sid)
{
    SegmentId start_sid_disk, end_sid_disk;
    uint32_t valid_rows;
    IndexType seg_index_type;
    
    if (read_lsm_segment_metadata(indexRelId, start_sid, end_sid, 
                             &start_sid_disk, &end_sid_disk, &valid_rows, &seg_index_type))
    {
        segment->segment_id_start = start_sid_disk;
        segment->segment_id_end = end_sid_disk;
        segment->vec_count = valid_rows;
        segment->index_type = seg_index_type;

        load_index_file(indexRelId, start_sid, end_sid, seg_index_type, &segment->index_ptr);
        load_bitmap_file(indexRelId, start_sid, end_sid, &segment->bitmap_ptr, false);
        load_mapping_file(indexRelId, start_sid, end_sid, &segment->map_ptr, false);

        segment->in_used = true;
        elog(DEBUG1, "[load_segment_from_disk] loaded segment %u-%u with %u vectors", start_sid, end_sid, valid_rows);
    }
    else 
    {
        elog(ERROR, "[load_segment_from_disk] Failed to read segment metadata for segment %u-%u", start_sid, end_sid);
    }
    pg_write_barrier();
}


