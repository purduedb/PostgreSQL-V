#include "postgres.h"
#include "miscadmin.h"
#include "storage/lwlock.h"
#include "utils/elog.h"

#include "lsmindex.h"
#include "lsm_segment.h"
#include "lsmindex_io.h"        /* write_bitmap_file_with_subversion, load_*_file */
#include "tasksend.h"           /* dpv_send_vacuum_refresh_async */
#include "segment_vacuum_redo.h"

/*
 * segment_vacuum_redo.c — standby WAL redo for unified vacuum tombstones.
 *
 * This runs in the startup process (WAL recovery), NOT in the
 * vector_index_worker. Critical consequence:
 *
 *   - `static FlushedSegmentPool segment_pool[]` (lsm_segment.c) is
 *     process-local memory. The worker owns the populated copy; the
 *     startup process sees zero-initialized memory. We MUST NOT walk it
 *     or touch any FlushedSegmentData fields from here.
 *   - SharedMemtableBuffer (memtables) IS in shared memory; we can take
 *     mt->vacuum_lock and write mt->bitmap safely.
 *   - On-disk bitmap subversion files are owned by the filesystem; the
 *     startup process can read+write them directly, just like the primary.
 *
 * Routing:
 *   1. If a memtable for hdr->sid exists in SharedMemtableBuffer (via
 *      lsm->memtable_idxs[]) → apply to it. Shmem-safe.
 *   2. If a segment metadata file containing hdr->sid exists on disk →
 *      read current bitmap/mapping/offsets, apply via 2-pointer merge,
 *      write new subversion file. Then fire a non-blocking
 *      SEGMENT_UPDATE_VACUUM task so the worker refreshes its in-memory
 *      pool bitmap (no-op if the segment isn't loaded).
 *   3. If neither found → ERROR (data lost, indicates a §11 ordering
 *      violation or out-of-order WAL/file delivery bug).
 *
 * Memtable and segment file are independent stores; both may exist
 * simultaneously during the transient window between adoption and
 * RELEASE_MEMTABLE replay. Both are updated when both exist.
 */

/* ----------------------------------------------------------------------
 * Memtable branch — shared memory, safe from startup process.
 * ---------------------------------------------------------------------- */
/*
 * Apply the WAL entries to `mt`. Caller MUST already hold
 * mt->vacuum_lock EXCLUSIVE (see apply_to_memtable_for_sid for the lock
 * order required to avoid deadlock with segment_fetcher_main).
 */
static void
apply_entries_to_memtable_locked(ConcurrentMemTable mt,
                                 const xl_dpv_vacuum_tombstones *hdr,
                                 const xl_dpv_vacuum_entry *entries)
{
    if (hdr->is_memtable_owner == 1)
    {
        /* Direct slot indexing — sid_local_idx == memtable slot. */
        for (uint32 w = 0; w < hdr->entry_count; w++)
        {
            uint32 li = entries[w].sid_local_idx;
            if (li < mt->capacity)
                SET_SLOT(mt->bitmap, li);
        }
    }
    else
    {
        /* Primary already flushed; standby hasn't released the memtable
         * yet (RELEASE_MEMTABLE WAL not replayed). 2-pointer walk against
         * mt->tids in insertion order. */
        uint32 cur = pg_atomic_read_u32(&mt->current_size);
        uint32 cap = (cur > mt->capacity) ? mt->capacity : cur;
        uint32 w   = 0;
        for (uint32 j = 0; j < cap && w < hdr->entry_count; j++)
        {
            if (mt->tids[j] == entries[w].tid)
            {
                SET_SLOT(mt->bitmap, j);
                w++;
            }
        }
    }
}

/*
 * Find the memtable for `sid` and apply the WAL entries to it.
 *
 * Lock-order contract (must be obeyed to avoid deadlock with the fetcher,
 * see comment in segment_fetcher.c around the pre-flight cover scan):
 *
 *   Phase A:  acquire mt_lock SHARED, locate the matching ConcurrentMemTable
 *             pointer, RELEASE mt_lock.
 *   Phase B:  acquire mt->vacuum_lock EXCLUSIVE (no mt_lock held).
 *             Re-validate mt->memtable_id == sid (defense against future
 *             concurrent recycling; today it cannot happen because slot
 *             recycling on the standby goes through register-memtable WAL
 *             redo in the same startup process, sequenced with us).
 *             Apply, release.
 *
 * The fetcher holds vacuum_lock SHARED across an mt_lock EXCLUSIVE
 * acquisition; if we held mt_lock SHARED while requesting vacuum_lock
 * EXCLUSIVE, the two paths could deadlock (cycle: fetcher waits for
 * mt_lock EX which we hold SHARED; we wait for vacuum_lock EX which
 * fetcher holds SHARED).
 */
static bool
apply_to_memtable_for_sid(LSMIndexBufferSlot *slot, SegmentId sid,
                          const xl_dpv_vacuum_tombstones *hdr,
                          const xl_dpv_vacuum_entry *entries)
{
    LSMIndex           lsm       = &slot->lsmIndex;
    ConcurrentMemTable target_mt = NULL;

    /* ---- Phase A: locate under mt_lock SHARED, then release. ---- */
    LWLockAcquire(lsm->mt_lock, LW_SHARED);

    /* Sealed/immutable memtables: only the first memtable_count entries
     * in memtable_idxs[] are valid. */
    for (uint32_t i = 0; i < lsm->memtable_count && target_mt == NULL; i++)
    {
        int32_t            gidx = lsm->memtable_idxs[i];
        ConcurrentMemTable mt;

        if (gidx < 0) continue;
        mt = MT_FROM_SLOTIDX(gidx);
        if (mt != NULL && mt->memtable_id == sid)
            target_mt = mt;
    }

    /* Growing memtable lives in a separate slot. Sids are unique across
     * sealed + growing, so this is mutually exclusive with the loop above. */
    if (target_mt == NULL &&
        lsm->growing_memtable_idx != MT_IDX_INVALID &&
        lsm->growing_memtable_idx != MT_IDX_ROTATING)
    {
        ConcurrentMemTable g = MT_FROM_SLOTIDX(lsm->growing_memtable_idx);
        if (g != NULL && g->memtable_id == sid)
            target_mt = g;
    }

    LWLockRelease(lsm->mt_lock);

    if (target_mt == NULL)
        return false;

    /* ---- Phase B: vacuum_lock EXCLUSIVE on the saved pointer. ---- */
    LWLockAcquire(&target_mt->vacuum_lock, LW_EXCLUSIVE);

    /* Re-validate after the lock drop. Today this should always be true:
     * SharedMemtableBuffer slot recycling on the standby happens via
     * dpv_standby_register_memtable, itself a WAL redo callback, and WAL
     * replay is single-threaded in the startup process — it cannot
     * interleave with our own redo call. The check is defense-in-depth in
     * case that invariant is broken in the future. */
    if (target_mt->memtable_id != sid)
    {
        LWLockRelease(&target_mt->vacuum_lock);
        return false;
    }

    apply_entries_to_memtable_locked(target_mt, hdr, entries);

    LWLockRelease(&target_mt->vacuum_lock);
    return true;
}

/* ----------------------------------------------------------------------
 * Disk file branch — file I/O only, no FlushedSegmentPool access.
 * ---------------------------------------------------------------------- */

/* Scan on-disk metadata files for the one whose [start, end] contains sid.
 * Returns true and fills out_* on hit; false otherwise. */
static bool
find_disk_segment_for_sid(Oid indexRelId, SegmentId sid,
                          SegmentId *out_start, SegmentId *out_end,
                          uint32_t *out_version)
{
    SegmentFileInfo files[MAX_SEGMENTS_COUNT];
    int             n = scan_segment_metadata_files(indexRelId, files,
                                                     MAX_SEGMENTS_COUNT);

    for (int i = 0; i < n; i++)
    {
        if (files[i].start_sid <= sid && sid <= files[i].end_sid)
        {
            *out_start   = files[i].start_sid;
            *out_end     = files[i].end_sid;
            *out_version = files[i].version;
            return true;
        }
    }
    return false;
}

/* Find the offset-table slice for `sid` in a loaded SegmentOffsetRange[]
 * of `count` entries. Returns -1 on miss. */
static int
offset_index_for_sid(const SegmentOffsetRange *offsets, uint32 count,
                     SegmentId sid)
{
    int lo = 0;
    int hi = (int) count - 1;
    while (lo <= hi)
    {
        int mid = lo + ((hi - lo) >> 1);
        SegmentId mid_sid = offsets[mid].sid;
        if (mid_sid == sid) return mid;
        if (mid_sid < sid)  lo = mid + 1;
        else                hi = mid - 1;
    }
    return -1;
}

/*
 * Apply the vacuum batch to the on-disk bitmap subversion file for the
 * segment that owns hdr->sid, then synchronously notify the
 * vector_index_worker to refresh its in-memory bitmap.
 *
 * Retry loop: if the worker reports retry=1 (segment merged or rebuilt
 * out from under us between disk-scan and worker dispatch), re-scan disk
 * — the post-replace segment's metadata is on disk (the §11 protocol
 * orders SegmentReplaced WAL after vacuum WAL on the primary, but the
 * fetcher delivers files asynchronously; either OLD or NEW may be on
 * disk when we re-scan). Re-apply against the new target and re-dispatch.
 *
 * The worker's SEGMENT_UPDATE_VACUUM handler serializes against adoption
 * via pool->seg_lock SH + per_seg_mutex. Together with adoption's new
 * Phase 3 (per_seg_mutex pinning of predecessors), this ensures the
 * vacuum bits land in whichever segment is current at the moment the
 * worker processes them — no silent loss.
 *
 * Returns true if at least one apply succeeded; false otherwise.
 */
#define MAX_VACUUM_REDO_RETRIES  4

static bool
apply_to_disk_file_for_sid(int lsm_idx, Oid indexRelId,
                           const xl_dpv_vacuum_tombstones *hdr,
                           const xl_dpv_vacuum_entry *entries)
{
    int attempt;

    for (attempt = 0; attempt < MAX_VACUUM_REDO_RETRIES; attempt++)
    {
        SegmentId disk_start, disk_end;
        uint32_t  disk_version;
        SegmentOffsetRange *offsets    = NULL;
        int64_t  *mapping_ptr          = NULL;
        uint8_t  *bitmap               = NULL;
        uint32_t  delete_count_in_file = 0;
        int       oi;
        Size      start_off, end_off;
        Size      bitmap_size;
        SegmentId out_start, out_end;
        uint32_t  valid_rows;
        IndexType seg_index_type;
        uint32_t  latest_sub;
        uint32_t  use_subversion;
        int       worker_status;

        if (!find_disk_segment_for_sid(indexRelId, hdr->sid,
                                       &disk_start, &disk_end, &disk_version))
        {
            /* No segment file on disk contains this sid. */
            return false;
        }

        if (!read_lsm_segment_metadata(indexRelId, disk_start, disk_end, disk_version,
                                       &out_start, &out_end, &valid_rows, &seg_index_type))
        {
            elog(WARNING,
                 "[dpv_apply_vacuum_tombstones] metadata gone for [%u,%u] v=%u",
                 disk_start, disk_end, disk_version);
            return false;
        }

        load_offset_file(indexRelId, disk_start, disk_end, disk_version,
                         &offsets, true);
        if (offsets == NULL)
        {
            elog(WARNING,
                 "[dpv_apply_vacuum_tombstones] offset file missing for [%u,%u] v=%u",
                 disk_start, disk_end, disk_version);
            return false;
        }

        {
            uint32 count = (uint32) (disk_end - disk_start + 1);
            oi = offset_index_for_sid(offsets, count, hdr->sid);
        }
        if (oi < 0)
        {
            elog(WARNING,
                 "[dpv_apply_vacuum_tombstones] sid %u not in offset table of [%u,%u]",
                 hdr->sid, disk_start, disk_end);
            pfree(offsets);
            return false;
        }

        start_off = offsets[oi].start_offset;
        end_off   = offsets[oi].end_offset;

        /* Load latest bitmap subversion (mutable copy) and mapping. */
        load_bitmap_file(indexRelId, disk_start, disk_end, disk_version,
                         &bitmap, true, &delete_count_in_file);
        load_mapping_file(indexRelId, disk_start, disk_end, disk_version,
                          &mapping_ptr, true);
        if (bitmap == NULL || mapping_ptr == NULL)
        {
            elog(WARNING,
                 "[dpv_apply_vacuum_tombstones] failed to load bitmap/mapping for [%u,%u] v=%u",
                 disk_start, disk_end, disk_version);
            if (bitmap)      pfree(bitmap);
            if (mapping_ptr) pfree(mapping_ptr);
            pfree(offsets);
            return false;
        }

        /*
         * Apply.
         *
         * Fast path when the on-disk version matches the primary's
         * owner_version AND the primary's owner was a flushed segment
         * (not a memtable): sid_local_idx is bytewise valid against
         * mapping_ptr.
         *
         * Otherwise 2-pointer merge: mapping[start_off..end_off) and
         * entries[].tid are in the same relative physical-offset order
         * (compaction-only invariant). Walk mapping; advance the WAL
         * cursor only on tid match.
         */
        {
            bool fast_path = (disk_version == hdr->owner_version &&
                              hdr->is_memtable_owner == 0);
            uint32 applied = 0;

            if (fast_path)
            {
                for (uint32 w = 0; w < hdr->entry_count; w++)
                {
                    Size m = start_off + entries[w].sid_local_idx;
                    if (m < end_off && !IS_SLOT_SET(bitmap, m))
                    {
                        SET_SLOT(bitmap, m);
                        applied++;
                    }
                }
            }
            else
            {
                uint32 w = 0;
                for (Size m = start_off; m < end_off && w < hdr->entry_count; m++)
                {
                    if (mapping_ptr[m] == entries[w].tid)
                    {
                        if (!IS_SLOT_SET(bitmap, m))
                        {
                            SET_SLOT(bitmap, m);
                            applied++;
                        }
                        w++;
                    }
                }
                /* Unmatched trailing WAL entries: tids compacted out of
                 * the standby's on-disk mapping by a later rebuild —
                 * silently drop. */
            }

            delete_count_in_file += applied;
        }

        /*
         * Pick a local subversion number: hdr->subversion is the counter
         * value the primary assigned for its own (start_sid, end_sid,
         * version) target. If our disk target differs (because the
         * standby has a different segment for this sid), that number is
         * for a different counter. Use the next subversion local to our
         * target.
         */
        latest_sub = find_latest_bitmap_subversion(indexRelId,
                                                    disk_start, disk_end,
                                                    disk_version);
        use_subversion = (latest_sub == UINT32_MAX) ? 0 : latest_sub + 1;

        /* Persist new subversion file. */
        bitmap_size = GET_BITMAP_SIZE(valid_rows);
        write_bitmap_file_with_subversion(indexRelId, disk_start, disk_end,
                                          disk_version, use_subversion,
                                          bitmap, bitmap_size,
                                          delete_count_in_file);

        pfree(bitmap);
        pfree(mapping_ptr);
        pfree(offsets);

        /*
         * Synchronously notify the worker. Three outcomes:
         *   0 = OK — segment in pool (or slot not queryable, treated as
         *       success since disk file is durable).
         *   1 = RETRY — segment merged/rebuilt; re-scan and retry.
         *   anything else — treat as success (defensive).
         *
         * segment_update_blocking uses MyProcNumber for the result slot;
         * the ring buffer's result pool is sized for MaxBackends +
         * NUM_AUXILIARY_PROCS so the startup process can use it.
         */
        worker_status = segment_update_blocking(lsm_idx, indexRelId,
                                                SEGMENT_UPDATE_VACUUM,
                                                disk_start, disk_end,
                                                disk_version,
                                                NULL, 0, NULL, 0);

        if (worker_status == 0)
            return true;

        if (worker_status == 1)
        {
            elog(DEBUG1,
                 "[dpv_apply_vacuum_tombstones] worker RETRY for sid %u "
                 "(target [%u,%u] v=%u merged/rebuilt); re-resolving disk segment",
                 hdr->sid, disk_start, disk_end, disk_version);
            continue;
        }

        /* Unexpected status — assume durable on disk is good enough. */
        elog(DEBUG1,
             "[dpv_apply_vacuum_tombstones] unexpected worker status %d for sid %u; treating as OK",
             worker_status, hdr->sid);
        return true;
    }

    elog(WARNING,
         "[dpv_apply_vacuum_tombstones] retry budget exhausted for sid %u; "
         "disk file written but worker pool may not reflect it until next reload",
         hdr->sid);
    return true;
}

/* ----------------------------------------------------------------------
 * Entry point — runs in startup process during WAL recovery.
 * ---------------------------------------------------------------------- */
void
dpv_apply_vacuum_tombstones(const xl_dpv_vacuum_tombstones *hdr,
                            const xl_dpv_vacuum_entry *entries)
{
    int                 slot_idx;
    LSMIndexBufferSlot *slot;
    bool                mem_handled;
    bool                file_handled;

    slot_idx = lookup_lsm_index_idx(hdr->indexRelId);
    if (slot_idx < 0)
        return;
    slot = &SharedLSMIndexBuffer->slots[slot_idx];

    /*
     * Apply to memtable (if present) AND to on-disk segment file (if
     * present). Both are independent durable stores; both must be kept
     * consistent. The transient window between adoption and
     * RELEASE_MEMTABLE replay is when both exist; outside that window
     * exactly one exists. The §11 ordering invariant guarantees at least
     * one always exists for a well-formed WAL stream.
     */
    mem_handled  = apply_to_memtable_for_sid(slot, hdr->sid, hdr, entries);
    file_handled = apply_to_disk_file_for_sid(slot_idx, hdr->indexRelId,
                                               hdr, entries);

    if (!mem_handled && !file_handled)
        elog(ERROR,
             "dpv_apply_vacuum_tombstones: sid %u for index %u has no "
             "memtable in shmem and no segment file on disk (entry_count=%u, "
             "owner_version=%u, is_memtable_owner=%u). This indicates a §11 "
             "ordering violation or out-of-order WAL/file delivery.",
             hdr->sid, hdr->indexRelId, hdr->entry_count,
             hdr->owner_version, hdr->is_memtable_owner);
}
