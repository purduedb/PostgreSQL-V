/*
 * segment_adoption.c — Plan 2 pool-only adoption (spec §8).
 *
 * dpv_pool_adopt is called from vector_index_worker_main's maintenance
 * pthread. It does NOT acquire lsm->mt_lock (a PG LWLock). Memtable
 * identification and release is handled entirely by segment_fetcher_main
 * (a regular PG bgworker process), which passes the cover set via the
 * ADOPT task payload and releases memtables after the worker returns.
 *
 * The pool's seg_lock is pthread_rwlock_t (process-private), so any
 * thread calling adoption must be inside the same process as the pool's
 * lifecycle owner (the vector_index_worker process).
 *
 * Plan 2 limitations (Plan 3 refactor status):
 *  1. Segment-predecessor bitmap union — IMPLEMENTED via
 *     union_segment_bitmap_via_offset_merge for Cases A and E. For each
 *     sid shared between old and new segment, walk both map_ptr slices
 *     in lockstep with a 2-pointer merge over the (sid, offset)-order
 *     invariant — old's surviving tids are a subsequence of new's
 *     (compaction-only). Bits whose tid was compacted out of the new
 *     segment (REBUILD_DELETION) are silently dropped.
 *  2. Memtable-predecessor bitmap union — IMPLEMENTED.
 *     The architectural commitment that this pthread does NOT acquire
 *     lsm->mt_lock is preserved: instead, segment_fetcher_main (a real
 *     bgworker process that already holds mt_lock for cover scanning)
 *     captures per-sid groups of set-bit tids across the cover memtables
 *     and marshals them through the ADOPT task DSM trailer. dpv_pool_adopt
 *     receives (groups, n_groups) and calls
 *     union_deletion_entries_into_new_seg before replace_flushed_segments_n
 *     in Case D (memtable-only cover) and Case E (memtable prefix/suffix
 *     plus pool-segment tile). Each group's tids[] is in insertion order;
 *     the function walks new_seg->map_ptr[offsets[sid] slice) and tids[]
 *     with a 2-pointer merge, matching by tid. Tombstones applied to a
 *     memtable on the standby survive the memtable being dropped.
 *  3. **Pool insert_idx leak.** When replace_flushed_segments_n retires the
 *     old slots, cleanup_flushed_segment attempts pthread_rwlock_trywrlock
 *     to rewind pool->insert_idx; under adoption (which holds the write
 *     lock), this trywrlock always fails. Pre-existing pattern; out of
 *     Plan 2/3 scope.
 *
 * The original "group size > 2 unsupported" limitation has been lifted —
 * adoption now uses replace_flushed_segments_n which accepts up to
 * MAX_SEGMENTS_COUNT contiguous old segments. This matters for standby
 * out-of-order arrivals where a new segment's range may be tiled by
 * many small pool segments.
 */

#include "postgres.h"
#include "miscadmin.h"
#include "storage/lwlock.h"
#include "utils/elog.h"

#include <pthread.h>

#include "lsmindex.h"
#include "lsm_segment.h"
#include "segment_adoption.h"


/* Scan the pool for a segment whose range strictly contains [a,b]. */
static uint32
find_strictly_containing(FlushedSegmentPool *pool, SegmentId a, SegmentId b)
{
    uint32 idx = pool->head_idx;
    while (idx != (uint32) -1)
    {
        FlushedSegment seg = &pool->flushed_segments[idx];
        if (seg->in_used &&
            seg->segment_id_start <= a && seg->segment_id_end >= b &&
            !(seg->segment_id_start == a && seg->segment_id_end == b))
        {
            return idx;
        }
        if (idx == pool->tail_idx)
            break;
        idx = seg->next_idx;
    }
    return (uint32) -1;
}

/*
 * Union the OLD segment's deleted-bit set into NEW's bitmap, translating
 * via tid: for each bit set in old->bitmap_ptr, look up its tid in old's
 * mapping, then resolve to new's local_idx via the sorted-permutation aux.
 *
 * Both segments must already be loaded (map_ptr, bitmap_ptr non-NULL).
 * The caller holds pool->seg_lock in WRITE mode, so no other thread can
 * touch either segment.
 *
 * Bits whose tid no longer exists in `new` are silently dropped — the
 * tuple was compacted out by REBUILD_DELETION and is no longer indexable.
 *
 * `new`'s delete_count is bumped only for bits that flip from 0 to 1.
 */
static void
union_segment_bitmap_via_offset_merge(FlushedSegment new_seg, FlushedSegment old_seg)
{
    int applied = 0;

    if (old_seg->bitmap_ptr == NULL || old_seg->map_ptr == NULL ||
        new_seg->bitmap_ptr == NULL || new_seg->map_ptr == NULL ||
        old_seg->offsets == NULL    || new_seg->offsets == NULL)
        return;

    /* Walk new's offsets[] (typically the smaller superset of sids), looking up
     * each sid in old's offsets[]. The arrays are small (one entry per sid in
     * the range), so a linear scan inside the inner lookup is fine — for very
     * large merges this could become binary search, but in practice
     * offsets_count is a handful. */
    for (uint32 ni = 0; ni < new_seg->offsets_count; ni++)
    {
        SegmentId sid = new_seg->offsets[ni].sid;
        int       oi  = -1;
        for (uint32 k = 0; k < old_seg->offsets_count; k++)
            if (old_seg->offsets[k].sid == sid) { oi = (int) k; break; }
        if (oi < 0)
            continue;       /* sid only in new (impossible at adoption time) */

        Size old_lo = old_seg->offsets[oi].start_offset;
        Size old_hi = old_seg->offsets[oi].end_offset;
        Size new_lo = new_seg->offsets[ni].start_offset;
        Size new_hi = new_seg->offsets[ni].end_offset;

        Size i_old = old_lo, i_new = new_lo;
        while (i_old < old_hi && i_new < new_hi)
        {
            if (old_seg->map_ptr[i_old] == new_seg->map_ptr[i_new])
            {
                if (IS_SLOT_SET(old_seg->bitmap_ptr, i_old) &&
                    !IS_SLOT_SET(new_seg->bitmap_ptr, i_new))
                {
                    SET_SLOT(new_seg->bitmap_ptr, i_new);
                    applied++;
                }
                i_old++; i_new++;
            }
            else
            {
                /* old has a tid that new compacted out. */
                i_old++;
            }
        }
    }

    if (applied > 0)
        new_seg->delete_count += applied;
}

/*
 * Plan 3 refactor — memtable-predecessor union at adoption.
 *
 * For each per-sid group of deletion tids from the fetcher's cover scan,
 * walk new_seg's map_ptr[offsets[sid] slice) in lockstep with group->tids[].
 * The fetcher captured tids[] in mt->tids insertion order, which equals
 * the order they appear in new_seg's map_ptr for that sid (the segment was
 * just flushed from this memtable, or merged from segments that preserve
 * relative order per sid).
 *
 * Caller holds pool->seg_lock in WRITE mode.
 */
static void
union_deletion_entries_into_new_seg(FlushedSegment new_seg,
                                     const DpvVacuumGroup *groups, int n_groups)
{
    int applied = 0;

    if (new_seg->bitmap_ptr == NULL || new_seg->map_ptr == NULL ||
        new_seg->offsets == NULL || groups == NULL || n_groups <= 0)
        return;

    for (int g = 0; g < n_groups; g++)
    {
        const DpvVacuumGroup *gp = &groups[g];
        int   oi = -1;

        if (gp->n_tids == 0 || gp->tids == NULL) continue;

        for (uint32 k = 0; k < new_seg->offsets_count; k++)
            if (new_seg->offsets[k].sid == gp->sid) { oi = (int) k; break; }
        if (oi < 0)
            continue;       /* sid not present in new_seg (shouldn't happen). */

        Size m_lo = new_seg->offsets[oi].start_offset;
        Size m_hi = new_seg->offsets[oi].end_offset;
        Size m    = m_lo;
        uint32 w  = 0;

        while (m < m_hi && w < gp->n_tids)
        {
            if (new_seg->map_ptr[m] == gp->tids[w])
            {
                if (!IS_SLOT_SET(new_seg->bitmap_ptr, m))
                {
                    SET_SLOT(new_seg->bitmap_ptr, m);
                    applied++;
                }
                w++;
                m++;
            }
            else
            {
                /* new has a tid that wasn't deleted (cover memtable's bitmap
                 * had 0 here). Advance new's pointer. */
                m++;
            }
        }
    }

    if (applied > 0)
        new_seg->delete_count += applied;
}

/*
 * dpv_pool_adopt — pool-side adoption. Does NOT acquire lsm->mt_lock.
 *
 * The fetcher (segment_fetcher_main) passes the memtable_cover: the set of
 * memtable sids it found in [start_sid, end_sid] under LW_SHARED and will
 * release under LW_EXCLUSIVE after we return.
 *
 * We compute the residual R = [start_sid, end_sid] \ memtable_cover and
 * verify that pool segments exactly tile R. The I/O (load_and_set_segment)
 * is done outside the write lock; the slot is reserved first so no one else
 * can claim it.
 *
 * Locking structure (revised):
 *   - Reserve slot (briefly under write lock; nothing else).
 *   - Load segment from disk (no lock).
 *   - Acquire write lock and re-run the full Case A/B/D/E decision against
 *     the *current* pool state, then execute. This avoids false discards
 *     caused by compatible intermediate pool changes during the I/O — a
 *     concurrent merge that turns [3,3][4,4][5,5] into [3,4][5,5] no
 *     longer invalidates an in-flight adoption of [3,5]. The decision is
 *     guaranteed to reflect the live state at execute time.
 *
 * Cases:
 *   A — exact-range match in pool with lower version → replace it.
 *   B — pool has a strictly-containing wider segment → stale.
 *   D — memtable_cover covers the full range; no pool overlap → append.
 *   E — pool segments exactly tile R; replace them with the new segment.
 */
DpvAdoptionOutcome
dpv_pool_adopt(int lsm_idx, Oid indexRelId,
               SegmentId start_sid, SegmentId end_sid, uint32 version,
               const SegmentId *memtable_cover, int memtable_cover_count,
               const DpvVacuumGroup *groups, int n_groups)
{
    LSMIndexBufferSlot *slot;
    FlushedSegmentPool *pool;
    DpvAdoptionOutcome outcome = DPV_ADOPT_STALE_DISCARD;
    uint32 new_slot = (uint32) -1;
    bool   new_slot_used = false;

    /*
     * Residual range R = [r_start, r_end] that pool segments must tile.
     * r_empty = true means memtable_cover covers the full [start_sid, end_sid].
     * R depends only on the input args (start_sid, end_sid, memtable_cover);
     * it does NOT depend on pool state, so it can be computed before any lock.
     */
    SegmentId r_start = start_sid;
    SegmentId r_end   = end_sid;
    bool      r_empty = false;

    if (lsm_idx < 0 || lsm_idx >= INDEX_BUF_SIZE)
        return DPV_ADOPT_INDEX_UNLOADED;
    slot = &SharedLSMIndexBuffer->slots[lsm_idx];
    if (!is_queryable(pg_atomic_read_u32(&slot->valid)))
        return DPV_ADOPT_INDEX_UNLOADED;
    if (slot->lsmIndex.indexRelId != indexRelId)
        return DPV_ADOPT_INDEX_UNLOADED;

    pool = get_flushed_segment_pool(lsm_idx);

    /* ------------------------------------------------------------------
     * Compute residual R from memtable_cover (sorted ascending).
     *
     * We only handle the common cases where the cover forms a contiguous
     * prefix, suffix, full span, or empty of [start_sid, end_sid].
     * Non-contiguous coverage is rejected defensively.
     * ------------------------------------------------------------------ */
    if (memtable_cover_count > 0)
    {
        /* Walk down from end_sid to find how far the suffix is covered. */
        SegmentId suffix_low = end_sid + 1;  /* start of covered suffix (exclusive if == end_sid+1) */
        for (SegmentId s = end_sid; s >= start_sid && s != (SegmentId) -1; s--)
        {
            bool found = false;
            for (int i = 0; i < memtable_cover_count; i++)
            {
                if (memtable_cover[i] == s) { found = true; break; }
            }
            if (!found) break;
            suffix_low = s;
            if (s == 0) break;
        }

        /* Walk up from start_sid to find how far the prefix is covered. */
        SegmentId prefix_high = start_sid - 1; /* end of covered prefix (exclusive if == start_sid-1) */
        for (SegmentId s = start_sid; s <= end_sid; s++)
        {
            bool found = false;
            for (int i = 0; i < memtable_cover_count; i++)
            {
                if (memtable_cover[i] == s) { found = true; break; }
            }
            if (!found) break;
            prefix_high = s;
        }

        bool full_cover   = (suffix_low == start_sid && prefix_high == end_sid);
        bool clean_suffix = (suffix_low <= end_sid && prefix_high < start_sid);
        bool clean_prefix = (suffix_low > end_sid && prefix_high >= start_sid);

        if (full_cover)
        {
            r_empty = true;
        }
        else if (clean_suffix)
        {
            r_start = start_sid;
            r_end   = suffix_low - 1;
        }
        else if (clean_prefix)
        {
            r_start = prefix_high + 1;
            r_end   = end_sid;
        }
        else
        {
            /* Non-contiguous coverage — defensive STALE_DISCARD. */
            elog(ERROR,
                 "[dpv_pool_adopt] non-contiguous memtable cover for [%u,%u]; STALE_DISCARD",
                 start_sid, end_sid);
            return DPV_ADOPT_STALE_DISCARD;
        }
    }

    /* ------------------------------------------------------------------
     * Brief lock: reserve a slot only.
     *
     * No pool inspection here. Doing inspection in this window would just
     * produce a snapshot that the final lock section has to re-check
     * against live state anyway — the snapshot is pure overhead.
     * ------------------------------------------------------------------ */
    pthread_rwlock_wrlock(&pool->seg_lock);
    new_slot = reserve_flushed_segment(pool);
    pthread_rwlock_unlock(&pool->seg_lock);
    if (new_slot == (uint32) -1)
    {
        elog(WARNING,
             "[dpv_pool_adopt] no free pool slot for indexRelId=%u range=[%u,%u] v=%u",
             indexRelId, start_sid, end_sid, version);
        return DPV_ADOPT_STALE_DISCARD;
    }
    new_slot_used = true;

    /* ------------------------------------------------------------------
     * Phase 2: slow disk I/O — no locks held.
     * ------------------------------------------------------------------ */
    load_and_set_segment(indexRelId, new_slot,
                         &pool->flushed_segments[new_slot],
                         start_sid, end_sid, version, false);
    pool->flushed_segments[new_slot].version = version;

    /* ------------------------------------------------------------------
     * Phase 3: split-lock decision + execution with bounded retry.
     *
     * Lock-acquisition order to avoid deadlock with vacuum-redo's worker
     * handler (which takes pool->seg_lock SHARED briefly, then releases
     * before per_seg_mutex):
     *
     *   3.1  acquire pool->seg_lock SHARED
     *   3.2  decide case (A/B/D/E); snapshot each predecessor as
     *        (seg_idx, start_sid, end_sid, version)
     *   3.3  release pool->seg_lock
     *   3.4  acquire per_seg_mutex on snapshot in ascending start_sid order
     *        (snapshot is already pool-order, which is ascending — no sort)
     *   3.5  re-validate each snapshot (in_used + range + version match);
     *        on staleness, release per_seg_mutex (reverse order) and retry
     *   3.6  union bitmap into new_seg (pre-validated predecessors)
     *   3.7  acquire pool->seg_lock EXCLUSIVE briefly; replace; release
     *   3.8  release per_seg_mutex (reverse order)
     *
     * Why per_seg_mutex is acquired WITHOUT seg_lock held: any other
     * mutator (vacuum-redo worker, concurrent adoption) follows the same
     * "release seg_lock before per_seg_mutex" rule, so we cannot form a
     * cycle of (holds seg_lock SH, wants per_seg) vs (holds per_seg, wants
     * seg_lock EX). Same pattern as merge_adjacent_segments_pool.
     * ------------------------------------------------------------------ */
#define MAX_ADOPT_RETRIES 4
    {
        typedef struct PredSnap {
            uint32    seg_idx;
            SegmentId start;
            SegmentId end;
            uint32    version;
        } PredSnap;

        PredSnap snaps[MAX_SEGMENTS_COUNT];
        int      n_snaps;
        int      case_kind;   /* 0=undecided, 1=A, 2=stale, 3=D, 4=E */
        bool     adopted = false;
        int      retry;

        for (retry = 0; retry < MAX_ADOPT_RETRIES && !adopted; retry++)
        {
            n_snaps   = 0;
            case_kind = 0;

            /* 3.1-3.2: decide case + snapshot under SHARED. */
            pthread_rwlock_rdlock(&pool->seg_lock);

            /* Case A: exact-range match. */
            {
                uint32 found = find_segment_by_sids(pool, start_sid, end_sid);
                if (found != (uint32) -1)
                {
                    FlushedSegmentData *existing = &pool->flushed_segments[found];
                    if (existing->in_used && version > existing->version)
                    {
                        case_kind = 1;
                        snaps[0].seg_idx = found;
                        snaps[0].start   = existing->segment_id_start;
                        snaps[0].end     = existing->segment_id_end;
                        snaps[0].version = existing->version;
                        n_snaps = 1;
                    }
                    else
                    {
                        case_kind = 2;   /* equal/higher version already present */
                    }
                }
            }

            /* Case B: strictly-containing wider segment. */
            if (case_kind == 0 &&
                find_strictly_containing(pool, start_sid, end_sid) != (uint32) -1)
            {
                case_kind = 2;
            }

            /* Case D: memtables cover full range. */
            if (case_kind == 0 && r_empty)
            {
                case_kind = 3;
                n_snaps = 0;
            }

            /* Case E: pool segments must exactly tile [r_start, r_end]. */
            if (case_kind == 0)
            {
                uint32    cur     = pool->head_idx;
                SegmentId tile_lo = r_start;
                int       e_count = 0;
                bool      tile_ok = true;

                while (cur != (uint32) -1)
                {
                    FlushedSegmentData *seg = &pool->flushed_segments[cur];
                    if (seg->segment_id_end < r_start)
                    {
                        cur = seg->next_idx;
                        if (cur == pool->head_idx) break;
                        continue;
                    }
                    if (seg->segment_id_start > r_end)
                        break;
                    if (seg->segment_id_start != tile_lo ||
                        seg->segment_id_end   >  r_end   ||
                        e_count               >= MAX_SEGMENTS_COUNT)
                    {
                        tile_ok = false;
                        break;
                    }
                    snaps[e_count].seg_idx = cur;
                    snaps[e_count].start   = seg->segment_id_start;
                    snaps[e_count].end     = seg->segment_id_end;
                    snaps[e_count].version = seg->version;
                    e_count++;
                    tile_lo = seg->segment_id_end + 1;
                    cur     = seg->next_idx;
                }

                if (!tile_ok || tile_lo != r_end + 1)
                    case_kind = 2;
                else
                {
                    case_kind = 4;
                    n_snaps   = e_count;
                }
            }

            pthread_rwlock_unlock(&pool->seg_lock);

            /* Stale → terminal STALE_DISCARD (no retry helps). */
            if (case_kind == 2)
            {
                outcome = DPV_ADOPT_STALE_DISCARD;
                break;
            }

            /* 3.4: acquire per_seg_mutex on predecessors in ascending
             * start_sid order. snaps[] is already in pool-list order
             * (Case A has 1 entry; Case E walked head→tail which is
             * ascending start_sid). No explicit sort needed. */
            for (int i = 0; i < n_snaps; i++)
                pthread_mutex_lock(
                    &pool->flushed_segments[snaps[i].seg_idx].per_seg_mutex);

            /* 3.5: re-validate snapshot. Slot identity AND version must
             * still match — pool slots are recycled, so a different
             * segment with a coincident version could occupy the slot
             * after our SH release. */
            {
                bool stale = false;
                for (int i = 0; i < n_snaps; i++)
                {
                    FlushedSegmentData *p =
                        &pool->flushed_segments[snaps[i].seg_idx];
                    if (!p->in_used ||
                        p->segment_id_start != snaps[i].start ||
                        p->segment_id_end   != snaps[i].end   ||
                        p->version          != snaps[i].version)
                    {
                        stale = true;
                        break;
                    }
                }
                if (stale)
                {
                    /* Release in reverse order. */
                    for (int i = n_snaps - 1; i >= 0; i--)
                        pthread_mutex_unlock(
                            &pool->flushed_segments[snaps[i].seg_idx].per_seg_mutex);
                    continue;   /* retry */
                }
            }

            /* 3.6: union under per_seg_mutex (predecessors are pinned). */
            {
                FlushedSegment new_seg = &pool->flushed_segments[new_slot];

                if (case_kind == 1)         /* Case A */
                {
                    union_segment_bitmap_via_offset_merge(new_seg,
                        &pool->flushed_segments[snaps[0].seg_idx]);
                }
                else if (case_kind == 3)    /* Case D */
                {
                    union_deletion_entries_into_new_seg(new_seg, groups, n_groups);
                }
                else if (case_kind == 4)    /* Case E */
                {
                    for (int i = 0; i < n_snaps; i++)
                    {
                        FlushedSegment old_seg =
                            &pool->flushed_segments[snaps[i].seg_idx];
                        union_segment_bitmap_via_offset_merge(new_seg, old_seg);
                    }
                    union_deletion_entries_into_new_seg(new_seg, groups, n_groups);
                }
            }

            /* 3.7: EX → replace → release EX. per_seg_mutex still held. */
            pthread_rwlock_wrlock(&pool->seg_lock);
            if (case_kind == 3)             /* Case D: just append */
            {
                register_flushed_segment(pool, new_slot);
            }
            else                            /* Case A / Case E: replace */
            {
                uint32 idxs[MAX_SEGMENTS_COUNT];
                for (int i = 0; i < n_snaps; i++)
                    idxs[i] = snaps[i].seg_idx;
                replace_flushed_segments_n(pool, idxs, n_snaps, new_slot);
            }
            pthread_rwlock_unlock(&pool->seg_lock);

            /* 3.8: release per_seg_mutex in reverse order. */
            for (int i = n_snaps - 1; i >= 0; i--)
                pthread_mutex_unlock(
                    &pool->flushed_segments[snaps[i].seg_idx].per_seg_mutex);

            new_slot_used = false;
            outcome       = DPV_ADOPT_ADOPTED;
            adopted       = true;
        }

        /* Retry budget exhausted with no adoption AND no terminal stale. */
        if (!adopted && outcome != DPV_ADOPT_STALE_DISCARD)
        {
            elog(WARNING,
                 "[dpv_pool_adopt] retry budget exhausted for [%u,%u] v=%u; STALE_DISCARD",
                 start_sid, end_sid, version);
            outcome = DPV_ADOPT_STALE_DISCARD;
        }
    }

    /* ------------------------------------------------------------------
     * finalize: discard the reserved slot if we never installed it.
     * Must hold pool->seg_lock WRITE for discard_reserved_segment.
     * ------------------------------------------------------------------ */
    if (new_slot_used)
    {
        pthread_rwlock_wrlock(&pool->seg_lock);
        discard_reserved_segment(pool, new_slot);
        pthread_rwlock_unlock(&pool->seg_lock);
    }
    return outcome;
}
