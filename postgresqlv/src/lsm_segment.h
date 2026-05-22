#ifndef LSM_SEGMENT_H
#define LSM_SEGMENT_H

#include "lsmindex.h"
// atomic_int should be defined before including this header in C++ mode
#ifndef __cplusplus
#include <stdatomic.h>
#endif
#include <pthread.h>

typedef enum SegmentLoadState
{
    SEG_NOT_LOADED  = 0,
    SEG_MMAP_LOADED = 1,  // index loaded via mmap; upgrade to full in-memory pending
    SEG_FULLY_LOADED = 2  // index fully resident in memory
} SegmentLoadState;

typedef struct FlushedSegmentData
{
    bool in_used;

    SegmentId segment_id_start;
    SegmentId segment_id_end;
    Size vec_count;

    // Size mapSize;
    // dsm_handle mapping;
    int64_t *map_ptr;
    IndexType index_type;
    void *index_ptr;
    uint8_t *bitmap_ptr;

    uint32_t next_idx;
    uint32_t prev_idx;  // Previous node in the linked list

    // Reference counting for safe segment deletion during merging
    atomic_int ref_count;

    // Load state: SEG_NOT_LOADED, SEG_MMAP_LOADED, or SEG_FULLY_LOADED
    atomic_int load_state;

    /* merge metadata (formerly in SharedSegmentArray) */
    uint32_t delete_count;         /* # deleted vectors; updated under seg_lock write */
    bool is_compacting;            /* claimed by a merge thread; set/cleared under seg_lock write */
    uint32_t version;              /* current on-disk version; used by SEGMENT_UPDATE_VACUUM */
    /* Per-sid offset table for this segment. Allocated in load_and_set_segment
     * via load_offset_file; freed in cleanup_flushed_segment /
     * discard_reserved_segment. Indexed by sid order; offsets[k].sid is the
     * physical sid, [start_offset, end_offset) is its slice of map_ptr /
     * bitmap_ptr. offsets_count == (segment_id_end - segment_id_start + 1). */
    SegmentOffsetRange *offsets;
    uint32_t            offsets_count;
    pthread_mutex_t per_seg_mutex; /* vacuum-merge bitmap coordination */
} FlushedSegmentData;
typedef FlushedSegmentData * FlushedSegment;

typedef struct FlushedSegmentPool
{
    pthread_rwlock_t seg_lock;
    FlushedSegmentData flushed_segments[MAX_SEGMENTS_COUNT];
    uint32_t flushed_segment_count;
    uint32_t head_idx;
    uint32_t tail_idx;
    uint32_t insert_idx;

    /* aggregate statistics for merge scheduling fast-path */
    pg_atomic_uint32 flat_count;
    pg_atomic_uint32 memtable_capacity_le_count;
    pg_atomic_uint32 small_segment_le_count;
} FlushedSegmentPool;

// helper functions (segment)
FlushedSegmentPool *get_flushed_segment_pool(int idx);
void initialize_segment_pool(FlushedSegmentPool *pool);
uint32_t reserve_flushed_segment(FlushedSegmentPool *pool);
void register_flushed_segment(FlushedSegmentPool *pool, uint32_t idx);
void decrement_flushed_segment_ref_count(FlushedSegmentPool *pool, uint32_t segment_idx);
void cleanup_flushed_segment(FlushedSegmentPool *pool, uint32_t segment_idx);

/*
 * discard_reserved_segment — free the buffers loaded into a slot that was
 * reserved via reserve_flushed_segment + load_and_set_segment but never
 * linked into the list (e.g. when adoption is stale-discarded after the
 * disk load). Does NOT touch list links, flushed_segment_count, or pool
 * stats — those are managed by register/replace, which the caller never
 * reached.
 *
 * Caller must hold the write lock on pool->seg_lock.
 */
void discard_reserved_segment(FlushedSegmentPool *pool, uint32_t segment_idx);
uint32_t find_segment_by_sids(FlushedSegmentPool *pool, SegmentId start_sid, SegmentId end_sid);
void find_two_adjacent_segments(FlushedSegmentPool *pool, SegmentId target_start_sid, SegmentId target_end_sid, 
                                 uint32_t *seg_idx_0, uint32_t *seg_idx_1);
/*
 * Replace N old segments with one new segment in the pool's linked list.
 * The old segments must be contiguous in the list (ascending sid order);
 * old_seg_idxs[0] is the lowest, old_seg_idxs[old_count-1] is the
 * highest. Caller must hold the write lock on pool->seg_lock.
 * Pre-condition: old_count >= 1.
 */
void replace_flushed_segments_n(FlushedSegmentPool *pool,
                                 const uint32_t *old_seg_idxs, int old_count,
                                 uint32_t new_seg_idx);
// IO
void load_all_segments_from_disk(Oid index_oid, FlushedSegmentPool *pool);
void load_all_segments_from_disk_mmap(Oid index_oid, FlushedSegmentPool *pool);
void load_and_set_segment(Oid indexRelId, uint32_t segment_idx, FlushedSegment segment, SegmentId start_sid, SegmentId end_sid, uint32_t version, bool use_mmap);

#endif