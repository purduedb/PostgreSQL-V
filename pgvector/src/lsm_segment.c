#include "lsm_segment.h"

static FlushedSegmentPool segment_pool[INDEX_BUF_SIZE];

FlushedSegmentPool *get_flushed_segment_pool(int idx)
{
    return &segment_pool[idx];
}

// ------------------------ flushed segment ------------------------
void initialize_segment_pool(FlushedSegmentPool *pool)
{
    pool->flushed_segment_count = 0;
    pool->tail_idx = -1;
    pool->insert_idx = 0;
    for(int i = 0; i < MAX_SEGMENTS_COUNT; ++i)
    {
        pool->flushed_segments[i].in_used = false;
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
    
    // set next_idx
    if(pool->tail_idx != -1)
        pool->flushed_segments[pool->tail_idx].next_idx = idx;
    pool->tail_idx = idx;
    ++pool->flushed_segment_count;
    elog(DEBUG1, "register_flushed_segment successfully added to list, idx = %d", idx);
}

// Reference counting functions for flushed segments
// This can only be called by the vector index worker
void
decrement_flushed_segment_ref_count(FlushedSegmentPool *pool, uint32_t segment_idx)
{
    FlushedSegment segment = &pool->flushed_segments[segment_idx];
    int old_count = atomic_fetch_sub(&segment->ref_count, 1);
    
    // FIXME: disable this for now
    // if (old_count == 1) {
    //     // Reference count reached zero, cleanup the segment
    //     cleanup_flushed_segment(pool, segment_idx);
    // }
}

void
cleanup_flushed_segment(FlushedSegmentPool *pool, uint32_t segment_idx)
{
    FlushedSegment segment = &pool->flushed_segments[segment_idx];
    
    elog(DEBUG1, "[cleanup_flushed_segment] Cleaning up segment idx = %d", segment_idx);
    
    // TODO: free the index, bitmap, map

    atomic_store(&segment->ref_count, 0);
    
    elog(DEBUG1, "[cleanup_flushed_segment] Segment idx = %d cleaned up", segment_idx);
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
        load_bitmap_file(indexRelId, start_sid, end_sid, &segment->bitmap_ptr);
        load_mapping_file(indexRelId, start_sid, end_sid, &segment->map_ptr);

        segment->in_used = true;
        elog(DEBUG1, "[load_segment_from_disk] loaded segment %u-%u with %u vectors", start_sid, end_sid, valid_rows);
    }
    else 
    {
        elog(ERROR, "[load_segment_from_disk] Failed to read segment metadata for segment %u-%u", start_sid, end_sid);
    }
    pg_write_barrier();
}


