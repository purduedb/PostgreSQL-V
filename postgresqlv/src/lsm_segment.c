#include "lsm_segment.h"
#include "vectorindeximpl.hpp"
#include "vector_index_worker.h"

static FlushedSegmentPool segment_pool[INDEX_BUF_SIZE];

FlushedSegmentPool *get_flushed_segment_pool(int idx)
{
    return &segment_pool[idx];
}

// ------------------------ flushed segment ------------------------
void initialize_segment_pool(FlushedSegmentPool *pool)
{
    pool->seg_lock = (pthread_rwlock_t) PTHREAD_RWLOCK_INITIALIZER;
    pool->flushed_segment_count = 0;
    pool->head_idx = (uint32_t)-1;
    pool->tail_idx = (uint32_t)-1;
    pool->insert_idx = 0;

    pg_atomic_init_u32(&pool->flat_count, 0);
    pg_atomic_init_u32(&pool->memtable_capacity_le_count, 0);
    pg_atomic_init_u32(&pool->small_segment_le_count, 0);

    for (int i = 0; i < MAX_SEGMENTS_COUNT; i++)
    {
        pool->flushed_segments[i].in_used = false;
        pool->flushed_segments[i].next_idx = (uint32_t)-1;
        pool->flushed_segments[i].prev_idx = (uint32_t)-1;
        atomic_store(&pool->flushed_segments[i].ref_count, 0);
        atomic_store(&pool->flushed_segments[i].load_state, (int)SEG_NOT_LOADED);

        pool->flushed_segments[i].delete_count = 0;
        pool->flushed_segments[i].is_compacting = false;
        pool->flushed_segments[i].version = 0;
        pool->flushed_segments[i].offsets       = NULL;
        pool->flushed_segments[i].offsets_count = 0;
        pthread_mutex_init(&pool->flushed_segments[i].per_seg_mutex, NULL);
    }
}
/* Update pool-level aggregate statistics atomics.
 * delta must be +1 (segment added) or -1 (segment removed).
 * For delta = -1, pass (uint32_t)(-1) which wraps correctly via unsigned arithmetic.
 * Caller must ensure counters do not underflow below 0.
 */
static void
update_pool_stats(FlushedSegmentPool *pool, IndexType index_type, uint32_t vec_count, int delta)
{
    uint32_t d = (uint32_t)delta;  /* -1 becomes UINT32_MAX; unsigned add wraps correctly */
    if (index_type == FLAT)
        pg_atomic_fetch_add_u32(&pool->flat_count, d);
    if (vec_count <= MEMTABLE_MAX_CAPACITY)
        pg_atomic_fetch_add_u32(&pool->memtable_capacity_le_count, d);
    if (vec_count <= THRESHOLD_SMALL_SEGMENT_SIZE)
        pg_atomic_fetch_add_u32(&pool->small_segment_le_count, d);
}

// need to acquire the lock first
uint32_t
reserve_flushed_segment(FlushedSegmentPool *pool)
{
    fprintf(stderr, "enter reserve_flushed_segment, insert_idx = %d\n", pool->insert_idx);
    for(int idx = pool->insert_idx; idx < MAX_SEGMENTS_COUNT; ++idx)
    {
        if(!pool->flushed_segments[idx].in_used)
        {
            pool->insert_idx = idx + 1;
            pool->flushed_segments[idx].in_used = true;
            fprintf(stderr, "reserve_flushed_segment successfully, idx = %d\n", idx);
            return idx;
        }
    }
    return -1;
}

void
register_flushed_segment(FlushedSegmentPool *pool, uint32_t idx)
{
    FlushedSegment new_seg = &pool->flushed_segments[idx];
    SegmentId new_start = new_seg->segment_id_start;

    // Initialize reference count for the new segment
    atomic_store(&new_seg->ref_count, 1);

    /*
     * Insert into the linked list in ascending segment_id_start order.
     *
     * On the primary, segments are produced in sid order, so the new
     * segment usually belongs at the tail; the walk-from-tail loop below
     * exits on the first iteration (O(1)). On the standby, out-of-order
     * arrivals via segment_fetcher_main can register a small-sid segment
     * after a large-sid one — in that case the walk traverses backward
     * until it finds the correct insertion point (O(N)).
     *
     * Sid-order is required by several consumers that traverse the list
     * directly: scan_and_claim_merge_task_pool's adjacency scan,
     * choose_adjacent_smaller_pool, find_two_adjacent_segments, the
     * dpv_pool_adopt case-E tiling walk, and find_strictly_containing.
     */
    if (pool->head_idx == (uint32_t) -1)
    {
        /* Empty list. */
        new_seg->prev_idx = (uint32_t) -1;
        new_seg->next_idx = (uint32_t) -1;
        pool->head_idx = idx;
        pool->tail_idx = idx;
    }
    else
    {
        uint32_t cur = pool->tail_idx;
        while (cur != (uint32_t) -1 &&
               pool->flushed_segments[cur].segment_id_start > new_start)
        {
            cur = pool->flushed_segments[cur].prev_idx;
        }

        if (cur == (uint32_t) -1)
        {
            /* All existing segments have higher start_sid — insert at head. */
            uint32_t old_head = pool->head_idx;
            new_seg->prev_idx = (uint32_t) -1;
            new_seg->next_idx = old_head;
            pool->flushed_segments[old_head].prev_idx = idx;
            pool->head_idx = idx;
        }
        else
        {
            /* Insert after `cur`. */
            uint32_t after = pool->flushed_segments[cur].next_idx;
            new_seg->prev_idx = cur;
            new_seg->next_idx = after;
            pool->flushed_segments[cur].next_idx = idx;
            if (after != (uint32_t) -1)
                pool->flushed_segments[after].prev_idx = idx;
            else
                pool->tail_idx = idx;  /* inserted at tail */
        }
    }

    ++pool->flushed_segment_count;

    update_pool_stats(pool,
                      new_seg->index_type,
                      new_seg->vec_count,
                      +1);

    fprintf(stderr, "register_flushed_segment successfully added to list, idx = %d (start_sid=%u)\n",
            idx, new_start);
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
    if (segment->offsets != NULL)
    {
        free(segment->offsets);
        segment->offsets = NULL;
    }
    segment->offsets_count = 0;

    // No need to unlink the segment in this function, the segment is already unlinked when the reference count reaches 0
    // Mark segment as unused
    segment->in_used = false;
    segment->next_idx = -1;
    segment->prev_idx = -1;
    
    // Update segment count
    --pool->flushed_segment_count;

    /* Reset per-segment mutex for potential reuse */
    pthread_mutex_destroy(&pool->flushed_segments[segment_idx].per_seg_mutex);
    pthread_mutex_init(&pool->flushed_segments[segment_idx].per_seg_mutex, NULL);

    atomic_store(&segment->ref_count, 0);
    
    // Update insert_idx to allow reuse of this slot in reserve_flushed_segment
    // Try to acquire write lock to update insert_idx safely
    // If we can't acquire it (e.g., already held by caller), we skip the update
    // This is safe because reserve_flushed_segment will eventually wrap around and find the slot
    int lock_acquired = pthread_rwlock_trywrlock(&pool->seg_lock);
    if (lock_acquired == 0)
    {
        // Successfully acquired lock - read insert_idx while holding lock to avoid race condition
        // Update insert_idx if this segment is before current insert_idx
        uint32_t current_insert_idx = pool->insert_idx;
        if (segment_idx < current_insert_idx)
        {
            pool->insert_idx = segment_idx;
        }
        pthread_rwlock_unlock(&pool->seg_lock);
    }
    // If lock acquisition failed, another thread holds it - skip insert_idx update
    // This is safe because reserve_flushed_segment will eventually find freed slots
    
    fprintf(stderr, "[cleanup_flushed_segment] Segment idx = %d cleaned up and unlinked from list\n", segment_idx);
}

// Cleanup for a slot reserved + loaded but never linked into the list.
// Frees the loaded buffers and clears in_used; does NOT touch list/stats/count.
// Caller must hold the write lock on pool->seg_lock.
void
discard_reserved_segment(FlushedSegmentPool *pool, uint32_t segment_idx)
{
    FlushedSegment segment = &pool->flushed_segments[segment_idx];

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
    if (segment->offsets != NULL)
    {
        free(segment->offsets);
        segment->offsets = NULL;
    }
    segment->offsets_count = 0;

    segment->in_used = false;
    segment->next_idx = -1;
    segment->prev_idx = -1;
    atomic_store(&segment->ref_count, 0);

    /* Rewind insert_idx so the slot is preferred for reuse. We already
     * hold the write lock here (caller's contract), so don't attempt a
     * second try-acquire. */
    if (segment_idx < pool->insert_idx)
    {
        pool->insert_idx = segment_idx;
    }

    fprintf(stderr, "[discard_reserved_segment] Segment idx = %u discarded (loaded buffers freed)\n",
            segment_idx);
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
            fprintf(stderr, "[find_segment_by_ids] Found segment %u-%u at idx %u\n", start_sid, end_sid, idx);
            return idx;
        }
    }
    
    fprintf(stderr, "[find_segment_by_ids] Segment %u-%u not found\n", start_sid, end_sid);
    return -1;
}

// Find two adjacent segments that together cover the target range
// Returns the indices in seg_idx_0 and seg_idx_1
// Note: requires lock to be held
void
find_two_adjacent_segments(FlushedSegmentPool *pool, SegmentId target_start_sid, SegmentId target_end_sid,
                           uint32_t *seg_idx_0, uint32_t *seg_idx_1)
{
    fprintf(stderr, "[find_two_adjacent_segments] Looking for segments covering range %u-%u\n", 
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
                    fprintf(stderr, "[find_two_adjacent_segments] Single segment %u-%u already covers range %u-%u, should use single segment replacement\n", 
                         seg->segment_id_start, seg->segment_id_end, target_start_sid, target_end_sid);
                    *seg_idx_0 = -1;
                    *seg_idx_1 = -1;
                    return;
                }
                
                // Need to find the adjacent segment
                uint32_t next_idx = seg->next_idx;
                if (next_idx == -1)
                {
                    fprintf(stderr, "[find_two_adjacent_segments] No adjacent segment found for range %u-%u\n", 
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
                    fprintf(stderr, "[find_two_adjacent_segments] Found segments: %u-%u and %u-%u covering range %u-%u\n",
                         seg->segment_id_start, seg->segment_id_end,
                         next_seg->segment_id_start, next_seg->segment_id_end,
                         target_start_sid, target_end_sid);
                    return;
                }
                else
                {
                    fprintf(stderr, "[find_two_adjacent_segments] Adjacent segment %u-%u does not cover the range\n", 
                         next_seg->segment_id_start, next_seg->segment_id_end);
                    *seg_idx_0 = -1;
                    *seg_idx_1 = -1;
                    return;
                }
            }
        }
        else {
            fprintf(stderr, "[find_two_adjacent_segments] No adjacent segments found for range %u-%u\n", 
                 target_start_sid, target_end_sid);
        }
        
        // Move to next segment
        if (idx == pool->tail_idx)
        {
            break;
        }
        idx = seg->next_idx;
    }
    
    fprintf(stderr, "[find_two_adjacent_segments] Could not find segments covering range %u-%u\n", 
         target_start_sid, target_end_sid);
    *seg_idx_0 = -1;
    *seg_idx_1 = -1;
}

// Replace N old segments (contiguous in linked list, ascending sid order) with
// one new segment. Caller must hold the write lock on pool->seg_lock.
void
replace_flushed_segments_n(FlushedSegmentPool *pool,
                            const uint32_t *old_seg_idxs, int old_count,
                            uint32_t new_seg_idx)
{
    FlushedSegment first_old;
    FlushedSegment last_old;
    FlushedSegment new_seg;
    uint32_t prev_idx;
    uint32_t next_idx;

    if (old_count <= 0)
    {
        fprintf(stderr, "[replace_flushed_segments_n] old_count=%d; nothing to replace\n", old_count);
        return;
    }

    first_old = &pool->flushed_segments[old_seg_idxs[0]];
    last_old  = &pool->flushed_segments[old_seg_idxs[old_count - 1]];
    new_seg   = &pool->flushed_segments[new_seg_idx];
    prev_idx  = first_old->prev_idx;
    next_idx  = last_old->next_idx;

    fprintf(stderr, "[replace_flushed_segments_n] Replacing %d old segments (first=%u, last=%u) with new segment idx=%u\n",
            old_count, old_seg_idxs[0], old_seg_idxs[old_count - 1], new_seg_idx);

    /*
     * Logical-removal point: decrement pool stats for each old segment NOW,
     * while they still hold their in_used metadata (index_type, vec_count).
     * Stats track logical-list membership; the segments are about to leave
     * the list. We do this BEFORE decrement_flushed_segment_ref_count so
     * cleanup_flushed_segment (which fires when ref_count hits 0) doesn't
     * need to know about stats anymore.
     */
    for (int i = 0; i < old_count; i++)
    {
        FlushedSegment old = &pool->flushed_segments[old_seg_idxs[i]];
        update_pool_stats(pool, old->index_type, old->vec_count, -1);
    }

    /*
     * Decrement ref counts on all old segments. If any drops to 0,
     * cleanup_flushed_segment runs (it uses trywrlock for insert_idx, so
     * safe even though we hold the write lock here).
     */
    for (int i = 0; i < old_count; i++)
    {
        decrement_flushed_segment_ref_count(pool, old_seg_idxs[i]);
    }

    /* Update head/tail if the replaced span included either end. */
    if (pool->head_idx == old_seg_idxs[0])
    {
        pool->head_idx = new_seg_idx;
    }
    if (pool->tail_idx == old_seg_idxs[old_count - 1])
    {
        pool->tail_idx = new_seg_idx;
    }

    /* Wire the new segment into the list in place of the old span. */
    new_seg->prev_idx = prev_idx;
    new_seg->next_idx = next_idx;

    if (prev_idx != (uint32_t) -1)
    {
        pool->flushed_segments[prev_idx].next_idx = new_seg_idx;
    }
    if (next_idx != (uint32_t) -1)
    {
        pool->flushed_segments[next_idx].prev_idx = new_seg_idx;
    }

    atomic_store(&new_seg->ref_count, 1);

    /*
     * Logical-insertion point: increment pool stats for the new segment
     * now that it is wired into the list. Pairs with the -1 done above
     * for each old segment.
     */
    update_pool_stats(pool, new_seg->index_type, new_seg->vec_count, +1);

    /* Net change in segment count: removed old_count, added 1. */
    pool->flushed_segment_count -= (old_count - 1);

    fprintf(stderr, "[replace_flushed_segments_n] Successfully replaced %d old segments with segment %u\n",
            old_count, new_seg_idx);
}

// -------------------------------- IO ----------------------------------

/*
 * load_all_segments_from_disk — populate the pool from the on-disk
 * segment files for an index. Called from the IndexLoadTaskType handler
 * (vector_index_worker.c) during slot LOADING_INDEX transition.
 *
 * Concurrency: the on-disk file set is NOT quiescent during this call.
 *   - Primary: lsm_flush_one_pending can write a new segment while we
 *     load; the file is missed here but picked up by the trailing
 *     SEGMENT_UPDATE_REGULAR maintenance task (which waits via
 *     wait_for_slot_queryable, then runs the dedup-protected register).
 *   - Standby: segment_fetcher_main pulls files in arbitrary order. The
 *     disk may have gaps in sid coverage — scan_segment_metadata_files
 *     tolerates these and returns whatever's present.
 *   - Merges/rebuilds: skipped by scan_and_claim_merge_task_pool for
 *     non-queryable slots (see Task 10), so no concurrent pool mutation
 *     races with this loader for the same slot.
 *
 * Missed-during-load segments converge via the corresponding maintenance
 * task path after the slot reaches QUERYABLE.
 */
void
load_all_segments_from_disk(Oid index_oid, FlushedSegmentPool *pool)
{
    SegmentFileInfo files[MAX_SEGMENTS_COUNT];
    int file_count = scan_segment_metadata_files(index_oid, files, MAX_SEGMENTS_COUNT);
    
    if (file_count == 0)
    {
        fprintf(stderr, "[load_all_segments_from_disk] No segment metadata files found for index %u\n", index_oid);
        return;
    }
    
    fprintf(stderr, "[load_all_segments_from_disk] Found %d segment metadata files\n", file_count);
    
    // Load segments in order
    for (int i = 0; i < file_count; i++)
    {
        uint32_t seg_idx = reserve_flushed_segment(pool);
        if (seg_idx == (uint32_t)-1)
        {
            fprintf(stderr, "no free flushed segment slot\n");
            continue;
        }

        FlushedSegment segment = &pool->flushed_segments[seg_idx];
        // Use version from SegmentFileInfo to avoid recalculating it
        load_and_set_segment(index_oid, seg_idx, segment, files[i].start_sid, files[i].end_sid, files[i].version, false);

        register_flushed_segment(pool, seg_idx);
    }

    fprintf(stderr, "[load_all_segments_from_disk] Loaded %d segments, tail_idx = %d\n", file_count, pool->tail_idx);
}

// Mmap-backed variant: loads each segment via mmap so the worker can become
// searchable immediately after a restart, before the full in-memory upgrade.
void
load_all_segments_from_disk_mmap(Oid index_oid, FlushedSegmentPool *pool)
{
    SegmentFileInfo files[MAX_SEGMENTS_COUNT];
    int file_count = scan_segment_metadata_files(index_oid, files, MAX_SEGMENTS_COUNT);

    if (file_count == 0)
    {
        fprintf(stderr, "[load_all_segments_from_disk_mmap] No segment metadata files found for index %u\n", index_oid);
        return;
    }

    fprintf(stderr, "[load_all_segments_from_disk_mmap] Found %d segment metadata files\n", file_count);

    for (int i = 0; i < file_count; i++)
    {
        uint32_t seg_idx = reserve_flushed_segment(pool);
        if (seg_idx == (uint32_t)-1)
        {
            fprintf(stderr, "[load_all_segments_from_disk_mmap] no free flushed segment slot\n");
            continue;
        }

        FlushedSegment segment = &pool->flushed_segments[seg_idx];
        load_and_set_segment(index_oid, seg_idx, segment, files[i].start_sid, files[i].end_sid, files[i].version, true);

        register_flushed_segment(pool, seg_idx);
    }

    fprintf(stderr, "[load_all_segments_from_disk_mmap] Loaded %d segments (mmap), tail_idx = %d\n", file_count, pool->tail_idx);
}

// handled by vector index worker (for index build)
void
load_and_set_segment(Oid indexRelId, uint32_t segment_idx, FlushedSegment segment, SegmentId start_sid, SegmentId end_sid, uint32_t version, bool use_mmap)
{
    SegmentId start_sid_disk, end_sid_disk;
    uint32_t valid_rows;
    IndexType seg_index_type;

    // TODO: for debugging
    fprintf(stderr, "[load_and_set_segment] loading segment %u-%u slot=%u (mmap=%d)\n",
            start_sid, end_sid, segment_idx, (int)use_mmap);

    // If version is UINT32_MAX, find latest version once to avoid multiple directory scans
    if (version == UINT32_MAX)
    {
        version = find_latest_segment_version(indexRelId, start_sid, end_sid);
    }

    if (read_lsm_segment_metadata(indexRelId, start_sid, end_sid, version,
                             &start_sid_disk, &end_sid_disk, &valid_rows, &seg_index_type))
    {
        segment->segment_id_start = start_sid_disk;
        segment->segment_id_end = end_sid_disk;
        segment->vec_count = valid_rows;
        segment->index_type = seg_index_type;

        load_index_file(indexRelId, start_sid, end_sid, version, seg_index_type, &segment->index_ptr, use_mmap);
        uint32_t delete_count;
        load_bitmap_file(indexRelId, start_sid, end_sid, version, &segment->bitmap_ptr, false, &delete_count);
        load_mapping_file(indexRelId, start_sid, end_sid, version, &segment->map_ptr, false);

        /*
         * Plan 3 refactor: cache the per-sid offset table on the segment.
         * Used by vacuum redo (segment_vacuum_redo.c) and adoption union
         * (segment_adoption.c) for the O(n) 2-pointer merge.
         */
        load_offset_file(indexRelId, start_sid, end_sid, version,
                         &segment->offsets, false);
        segment->offsets_count = (uint32_t) (end_sid - start_sid + 1);

        segment->in_used = true;
        segment->delete_count = delete_count;  /* already computed by load_bitmap_file */
        segment->version = version;
        segment->is_compacting = false;
        atomic_store(&segment->load_state,
                     use_mmap ? (int)SEG_MMAP_LOADED : (int)SEG_FULLY_LOADED);
        fprintf(stderr, "[load_segment_from_disk] loaded segment %u-%u v%u with %u vectors (mmap=%d)\n",
                start_sid, end_sid, version, valid_rows, (int)use_mmap);
    }
    else
    {
        fprintf(stderr, "[load_segment_from_disk] Failed to read segment metadata for segment %u-%u v%u\n", start_sid, end_sid, version);
    }
    pg_write_barrier();
}


