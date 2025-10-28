#ifndef LSM_SEGMENT_H
#define LSM_SEGMENT_H

#include "lsmindex.h"
#include <stdatomic.h>
#include <pthread.h>

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
} FlushedSegmentPool;

// helper functions (segment)
FlushedSegmentPool *get_flushed_segment_pool(int idx);
void initialize_segment_pool(FlushedSegmentPool *pool);
uint32_t reserve_flushed_segment(FlushedSegmentPool *pool);
void register_flushed_segment(FlushedSegmentPool *pool, uint32_t idx);
void decrement_flushed_segment_ref_count(FlushedSegmentPool *pool, uint32_t segment_idx);
void cleanup_flushed_segment(FlushedSegmentPool *pool, uint32_t segment_idx);
uint32_t find_segment_by_sids(FlushedSegmentPool *pool, SegmentId start_sid, SegmentId end_sid);
void find_two_adjacent_segments(FlushedSegmentPool *pool, SegmentId target_start_sid, SegmentId target_end_sid, 
                                 uint32_t *seg_idx_0, uint32_t *seg_idx_1);
void replace_flushed_segment(FlushedSegmentPool *pool, uint32_t old_seg_idx_0, uint32_t old_seg_idx_1, uint32_t new_seg_idx);
// IO
void load_all_segments_from_disk(Oid index_oid, FlushedSegmentPool *pool);                     
void load_and_set_segment(Oid indexRelId, uint32_t segment_idx, FlushedSegment segment, SegmentId start_sid, SegmentId end_sid);

#endif