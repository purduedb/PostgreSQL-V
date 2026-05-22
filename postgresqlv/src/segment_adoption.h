#ifndef DPV_SEGMENT_ADOPTION_H
#define DPV_SEGMENT_ADOPTION_H

#include "postgres.h"
#include "lsm_segment.h"

typedef enum {
    DPV_ADOPT_ADOPTED         = 0,
    DPV_ADOPT_STALE_DISCARD   = 1,
    DPV_ADOPT_INDEX_UNLOADED  = 2,
} DpvAdoptionOutcome;

/*
 * dpv_pool_adopt — pool-side adoption only. Does NOT touch lsm->mt_lock.
 * The caller (segment_fetcher_main) is responsible for releasing
 * memtables in [start_sid, end_sid] under LW_EXCLUSIVE on mt_lock after
 * this returns ADOPTED or INDEX_UNLOADED.
 *
 * memtable_cover lists the sids the caller intends to release; we use it
 * to compute the residual range R = [start_sid, end_sid] \ memtable_cover
 * that pool segments must exactly tile. Sorted ascending.
 *
 * Returns:
 *   DPV_ADOPT_ADOPTED        — pool successfully updated (or already
 *                               consistent, e.g., no pool work needed).
 *                               Caller proceeds with memtable release.
 *   DPV_ADOPT_STALE_DISCARD  — pool does not match expected residual; the
 *                               new segment cannot be safely adopted.
 *                               Caller must NOT release memtables.
 *   DPV_ADOPT_INDEX_UNLOADED — slot is not queryable; pool can't be
 *                               touched. Caller still releases memtables
 *                               (segment file is durably on disk).
 */
/*
 * Per-sid grouped deletion-tid payload from segment_fetcher's cover scan.
 * One group per cover memtable (each memtable carries exactly one sid).
 * tids[] is in insertion order (== mt->tids order); not sorted.
 */
typedef struct DpvVacuumGroup
{
    SegmentId      sid;
    uint32         n_tids;
    const int64_t *tids;
} DpvVacuumGroup;

/*
 * groups / n_groups — Plan 3 refactor. Per-sid grouped deletion bits from
 * the cover memtables, captured by the fetcher under
 * mt_lock SHARED + vacuum_lock SHARED before this task was submitted.
 * Cases D and E walk new_seg->map_ptr[offsets[sid] slice) in lockstep
 * against group->tids[] (2-pointer merge over the shared offset-order
 * invariant) and SET_SLOT each match into the new segment's bitmap,
 * so tombstones that were applied to a memtable on the standby
 * (e.g. via vacuum tombstone redo) are preserved when adoption drops
 * the memtable. NULL/0 means "no tombstones to carry forward" — common
 * case (zero overhead).
 */
extern DpvAdoptionOutcome dpv_pool_adopt(int lsm_idx, Oid indexRelId,
                                          SegmentId start_sid, SegmentId end_sid,
                                          uint32 version,
                                          const SegmentId *memtable_cover,
                                          int memtable_cover_count,
                                          const DpvVacuumGroup *groups,
                                          int n_groups);

#endif
