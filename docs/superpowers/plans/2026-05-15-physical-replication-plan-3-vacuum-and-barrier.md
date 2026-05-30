# Physical Replication — Plan 3: Vacuum WAL + Standby Barrier

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** Plan 1 and Plan 2 are merged. Memtable lifecycle records and segment-lifecycle records replicate, and segments adopt under identity bitmap translation.

**Goal:** Add the `SegmentVacuumTombstones` WAL record, the **WAL-emit-before-file-write protocol** (spec §11 — the most load-bearing single decision in the design), the lazy sorted-permutation auxiliary structure for tid-based bitmap translation (spec §13.3), version-aware bitmap merge in adoption, and the standby-side guards: disable write workers on the standby, hold indexes "not queryable" at attach until the fetcher catches up since restartpoint.

After Plan 3, the system is end-to-end correct under concurrent vacuum + merge on the primary, and a standby restart while fetches are pending is recoverable.

**Architecture:** Vacuum (`bulk_delete_lsm_index`) emits `SegmentVacuumTombstones` WAL **before** writing the bitmap subversion file at three call sites; redo on the standby applies the bitmap delta to the matching segment (fast path: same range+version; slow paths: predecessor segments via tid lookup; or memtable target). The aux structure is built lazily on first slow-path touch per segment. Write workers (`lsm_index_bgworker`, merge thread pool) are gated on `!RecoveryInProgress()`. At index attach (in `load_lsm_index_internal`), the standby refuses queries until the fetcher confirms every `SegmentCreated/Replaced` since restartpoint has a corresponding file on disk.

**Tech Stack:** PG 17, C. No new external dependencies.

---

## File structure

| File | Role | Action |
| --- | --- | --- |
| [pgvector/src/replication_rmgr.h](../../../pgvector/src/replication_rmgr.h) | Add `SegmentVacuumTombstones` record type and payload | Modify |
| [pgvector/src/replication_rmgr.c](../../../pgvector/src/replication_rmgr.c) | Add emit helper and redo callback (delegating slow-path work to a helper) | Modify |
| [pgvector/src/segment_vacuum_redo.h](../../../pgvector/src/segment_vacuum_redo.h) | Public API: `dpv_apply_vacuum_tombstones(...)` | Create |
| [pgvector/src/segment_vacuum_redo.c](../../../pgvector/src/segment_vacuum_redo.c) | Fast + 3 slow paths; writes a local subversion file once per WAL record | Create |
| [pgvector/src/sorted_perm.h](../../../pgvector/src/sorted_perm.h) | API for the lazy sorted-permutation aux structure | Create |
| [pgvector/src/sorted_perm.c](../../../pgvector/src/sorted_perm.c) | Build on first touch, cache for segment lifetime | Create |
| [pgvector/src/segment_adoption.c](../../../pgvector/src/segment_adoption.c) | Replace identity bitmap merge with version-aware merge using tid lookup | Modify |
| [pgvector/src/lsmindex.c](../../../pgvector/src/lsmindex.c) | At three sites in `bulk_delete_lsm_index`, emit `SegmentVacuumTombstones` + `XLogFlush` BEFORE writing the bitmap subversion file | Modify |
| [pgvector/src/lsmbackground.c](../../../pgvector/src/lsmbackground.c) | At entry to `lsm_index_bgworker_main`, `if (RecoveryInProgress()) proc_exit(0);` | Modify |
| [pgvector/src/vector_index_worker.c](../../../pgvector/src/vector_index_worker.c) | Skip `init_merge_thread_pool` under recovery; gate maintenance task dispatch | Modify |
| [pgvector/src/index_load_worker.c](../../../pgvector/src/index_load_worker.c) | At end of `load_lsm_index_internal`, on standby, set per-slot `not_queryable_until_lsn` to current WAL replay LSN; clear when fetcher catches up | Modify |
| [pgvector/src/segment_fetcher.c](../../../pgvector/src/segment_fetcher.c) | After draining queue, clear `not_queryable_until_lsn` for indexes whose pending fetches all resolve to `done` | Modify |
| [pgvector/src/lsm_segment.c](../../../pgvector/src/lsm_segment.c) (search path) | Before serving a query, check `not_queryable_until_lsn`; wait on a CV up to `pgvector_replication_fetch_wait_timeout` then ereport recovery conflict if not satisfied | Modify |
| [pgvector/src/replication_gucs.c](../../../pgvector/src/replication_gucs.c) | Add `pgvector_replication_fetch_wait_timeout` GUC | Modify |
| [pgvector/test/t/120_replication_vacuum_simple.pl](../../../pgvector/test/t/120_replication_vacuum_simple.pl) | Vacuum on primary → standby's segment bitmap reflects deletes | Create |
| [pgvector/test/t/121_replication_vacuum_merge_race.pl](../../../pgvector/test/t/121_replication_vacuum_merge_race.pl) | Concurrent vacuum + merge on primary; standby converges | Create |
| [pgvector/test/t/122_replication_rebuild_translation.pl](../../../pgvector/test/t/122_replication_rebuild_translation.pl) | Rebuild with compaction; standby's bitmap translation handles it | Create |
| [pgvector/test/t/123_replication_attach_barrier.pl](../../../pgvector/test/t/123_replication_attach_barrier.pl) | Standby restart with in-flight fetch; queries on affected index wait then succeed | Create |

---

## Phase 1 — `SegmentVacuumTombstones` WAL record

### Task 1.1: Define the record

**Files:**
- Modify: `pgvector/src/replication_rmgr.h`

- [ ] **Step 1: Add the info bit and payload structs**

```c
#define XLOG_DPV_SEGMENT_VACUUM_TOMBSTONES  0x70

typedef struct {
    uint32          local_idx;   /* index within the owner segment's slot array */
    ItemPointerData tid;         /* heap tid (stable across rebuild/merge)        */
} xl_dpv_vacuum_entry;

typedef struct {
    Oid       indexRelId;
    SegmentId owner_start_sid;   /* the segment whose subversion bitmap is being  */
    SegmentId owner_end_sid;     /* updated; on the primary, this is the segment  */
    uint32    owner_version;     /* that the bitmap-subversion file is named for. */
    uint32    subversion;        /* the new subversion number — used by standby   */
                                 /* to name its locally-written subversion file.  */
    uint16    is_memtable_target;/* 1 = the owner is still a memtable on primary  */
                                 /* (no subversion file; redo updates mt->bitmap) */
    uint16    entry_count;       /* number of xl_dpv_vacuum_entry following.      */
    /* followed by xl_dpv_vacuum_entry entries[entry_count] */
} xl_dpv_segment_vacuum_tombstones;
```

- [ ] **Step 2: Add emit prototype**

```c
extern XLogRecPtr dpv_emit_segment_vacuum_tombstones(
    Oid indexRelId,
    SegmentId owner_start_sid, SegmentId owner_end_sid, uint32 owner_version,
    uint32 subversion, bool is_memtable_target,
    const xl_dpv_vacuum_entry *entries, int entry_count);
```

### Task 1.2: Implement emit + redo

**Files:**
- Modify: `pgvector/src/replication_rmgr.c`

- [ ] **Step 1: Add to dispatcher + identify**

```c
case XLOG_DPV_SEGMENT_VACUUM_TOMBSTONES: redo_segment_vacuum_tombstones(record); break;
...
case XLOG_DPV_SEGMENT_VACUUM_TOMBSTONES: return "SEGMENT_VACUUM_TOMBSTONES";
```

- [ ] **Step 2: Emit helper**

```c
#include "segment_vacuum_redo.h"

XLogRecPtr
dpv_emit_segment_vacuum_tombstones(Oid indexRelId,
                                   SegmentId owner_start_sid, SegmentId owner_end_sid,
                                   uint32 owner_version, uint32 subversion,
                                   bool is_memtable_target,
                                   const xl_dpv_vacuum_entry *entries, int entry_count)
{
    xl_dpv_segment_vacuum_tombstones hdr = {
        .indexRelId         = indexRelId,
        .owner_start_sid    = owner_start_sid,
        .owner_end_sid      = owner_end_sid,
        .owner_version      = owner_version,
        .subversion         = subversion,
        .is_memtable_target = is_memtable_target ? 1 : 0,
        .entry_count        = (uint16) entry_count,
    };
    XLogRecPtr lsn;

    XLogBeginInsert();
    XLogRegisterData((char *) &hdr, sizeof(hdr));
    if (entry_count > 0)
        XLogRegisterData((char *) entries, entry_count * sizeof(xl_dpv_vacuum_entry));
    lsn = XLogInsert(RM_DPV_REPLICATION_ID, XLOG_DPV_SEGMENT_VACUUM_TOMBSTONES);
    return lsn;
}
```

- [ ] **Step 3: Redo callback (delegates to segment_vacuum_redo.c)**

```c
static void
redo_segment_vacuum_tombstones(XLogReaderState *r)
{
    char *data = XLogRecGetData(r);
    xl_dpv_segment_vacuum_tombstones *hdr = (xl_dpv_segment_vacuum_tombstones *) data;
    xl_dpv_vacuum_entry *entries = (xl_dpv_vacuum_entry *) (data + sizeof(*hdr));

    if (!InHotStandby)
        return;  /* primary crash recovery: load_lsm_index_internal reconciles */

    if (dpv_standby_wait_if_loading(hdr->indexRelId) != 1)
        return;

    dpv_apply_vacuum_tombstones(hdr, entries);
}
```

### Task 1.3: Build and commit

- [ ] **Step 1: Build, commit**

```bash
cd pgvector && make 2>&1 | tail -20
git add pgvector/src/replication_rmgr.h pgvector/src/replication_rmgr.c
git commit -m "feat: SegmentVacuumTombstones WAL record type"
```

---

## Phase 2 — Sorted-permutation aux structure

### Task 2.1: API

**Files:**
- Create: `pgvector/src/sorted_perm.h`

- [ ] **Step 1: Header**

```c
#ifndef DPV_SORTED_PERM_H
#define DPV_SORTED_PERM_H

#include "postgres.h"
#include "storage/itemptr.h"
#include "lsm_segment.h"  /* FlushedSegment, segment->map_ptr, segment->N */

/*
 * On first slow-path touch of a segment, build sorted_idx[N] such that
 * segment->map_ptr[sorted_idx[i]] is in tid order. Cached for the segment's
 * lifetime. Returns true on success, false on OOM (rare).
 *
 * Idempotent: subsequent calls return immediately.
 */
extern bool dpv_sorted_perm_ensure(FlushedSegment seg);

/*
 * Look up `tid` in `seg`'s mapping via the sorted permutation. Returns the
 * `local_idx` if found, -1 otherwise. Caller must have dpv_sorted_perm_ensure
 * succeeded on `seg` first (or this function calls ensure internally if not).
 */
extern int32 dpv_sorted_perm_find(FlushedSegment seg, ItemPointer tid);

/* Free aux memory (called at segment unload). */
extern void dpv_sorted_perm_free(FlushedSegment seg);

#endif
```

### Task 2.2: Implementation

**Files:**
- Create: `pgvector/src/sorted_perm.c`

- [ ] **Step 1: Build + lookup**

```c
#include "postgres.h"
#include "miscadmin.h"
#include "storage/lwlock.h"
#include "utils/elog.h"
#include "utils/memutils.h"
#include "lsm_segment.h"
#include "sorted_perm.h"

#include <stdlib.h>  /* qsort_r */

typedef struct {
    ItemPointerData *map_ptr;  /* mapping: local_idx → tid */
} CmpCtx;

static int
cmp_idx(const void *a, const void *b, void *ctx_)
{
    CmpCtx *ctx = (CmpCtx *) ctx_;
    uint32 ia = *(const uint32 *) a;
    uint32 ib = *(const uint32 *) b;
    return ItemPointerCompare(&ctx->map_ptr[ia], &ctx->map_ptr[ib]);
}

bool
dpv_sorted_perm_ensure(FlushedSegment seg)
{
    /* Single-flag init guard. Use the segment's existing per_seg_mutex. */
    LWLockAcquire(&seg->per_seg_mutex, LW_EXCLUSIVE);
    if (seg->sorted_idx != NULL)
    {
        LWLockRelease(&seg->per_seg_mutex);
        return true;
    }

    {
        uint32 *idx;
        uint32 N = seg->N;  /* number of entries in the segment */
        CmpCtx ctx = { .map_ptr = seg->map_ptr };

        idx = MemoryContextAlloc(TopMemoryContext, sizeof(uint32) * N);
        if (idx == NULL)
        {
            LWLockRelease(&seg->per_seg_mutex);
            return false;
        }
        for (uint32 i = 0; i < N; i++) idx[i] = i;
        qsort_r(idx, N, sizeof(uint32), cmp_idx, &ctx);
        seg->sorted_idx = idx;
    }
    LWLockRelease(&seg->per_seg_mutex);
    return true;
}

int32
dpv_sorted_perm_find(FlushedSegment seg, ItemPointer tid)
{
    uint32 *idx;
    uint32 N;
    int lo, hi;

    if (seg->sorted_idx == NULL)
    {
        if (!dpv_sorted_perm_ensure(seg))
            return -1;
    }
    idx = seg->sorted_idx;
    N   = seg->N;
    lo  = 0; hi = (int) N - 1;
    while (lo <= hi)
    {
        int mid = lo + ((hi - lo) >> 1);
        int cmp = ItemPointerCompare(&seg->map_ptr[idx[mid]], tid);
        if (cmp == 0)
            return (int32) idx[mid];
        if (cmp < 0) lo = mid + 1;
        else         hi = mid - 1;
    }
    return -1;
}

void
dpv_sorted_perm_free(FlushedSegment seg)
{
    if (seg->sorted_idx != NULL)
    {
        pfree(seg->sorted_idx);
        seg->sorted_idx = NULL;
    }
}
```

- [ ] **Step 2: Add `sorted_idx` field to `FlushedSegment`**

In [pgvector/src/lsm_segment.h](../../../pgvector/src/lsm_segment.h), inside `FlushedSegmentData`, add:

```c
    /* Lazy sorted permutation over map_ptr, in tid order. Allocated on first
     * slow-path touch by dpv_sorted_perm_ensure. NULL = not built yet. */
    uint32 *sorted_idx;
```

In the segment's unload path (e.g., `unload_segment` or `replace_flushed_segment` when an old segment is being freed), call `dpv_sorted_perm_free(seg)`.

- [ ] **Step 3: Build, commit**

```bash
# Add src/sorted_perm.o to OBJS in pgvector/Makefile
cd pgvector && make 2>&1 | tail -10
git add pgvector/src/sorted_perm.h pgvector/src/sorted_perm.c \
        pgvector/src/lsm_segment.h pgvector/src/lsm_segment.c \
        pgvector/Makefile
git commit -m "feat: lazy sorted-permutation aux structure for tid lookup"
```

---

## Phase 3 — Vacuum redo logic (fast + slow paths)

### Task 3.1: Header

**Files:**
- Create: `pgvector/src/segment_vacuum_redo.h`

- [ ] **Step 1: Header**

```c
#ifndef DPV_SEGMENT_VACUUM_REDO_H
#define DPV_SEGMENT_VACUUM_REDO_H

#include "postgres.h"
#include "replication_rmgr.h"

extern void dpv_apply_vacuum_tombstones(
    const xl_dpv_segment_vacuum_tombstones *hdr,
    const xl_dpv_vacuum_entry *entries);

#endif
```

### Task 3.2: Implementation

**Files:**
- Create: `pgvector/src/segment_vacuum_redo.c`

The implementation routes based on what state the standby holds for `[owner_start, owner_end]` at redo time. There are four paths:

1. **Fast path** — pool has a segment with exact range AND version.
2. **Slow path 1 (rebuild race)** — pool has a segment with exact range but older version. The pulled-segment file for the new version is in flight; until it arrives, route tombstones to the predecessor via tid lookup.
3. **Slow path 2 (merge race)** — pool has multiple narrower segments overlapping the range. Find each tid in the predecessor that owns it.
4. **Slow path 3 (memtable target)** — owner is still a memtable on this standby (the segment file for the flush hasn't arrived yet, or the primary's record set `is_memtable_target=1`). Update `mt->bitmap` directly via the encoded `local_idx` (memtables have stable slot indexes).

For paths 1, 2, and 3a (segment targets), write a local subversion file once per record after the bitmap update. Path 3 (memtable target) does not write a subversion file — memtables have no persistent bitmap.

- [ ] **Step 1: Implementation**

```c
#include "postgres.h"
#include "miscadmin.h"
#include "storage/lwlock.h"
#include "utils/elog.h"

#include "lsm_segment.h"
#include "lsmindex.h"
#include "lsmindex_io.h"      /* write_bitmap_file_with_subversion */
#include "sorted_perm.h"
#include "standby_memtable.h"
#include "segment_vacuum_redo.h"

static FlushedSegment
find_segment_exact(LSMIndexBufferSlot *slot, SegmentId start, SegmentId end)
{
    return find_segment_by_sids(slot->pool, start, end);  /* existing helper */
}

static FlushedSegment
find_segment_overlap(LSMIndexBufferSlot *slot, SegmentId start, SegmentId end,
                     int idx)
{
    /* Return the idx-th existing pool entry whose range overlaps [start,end].
     * Existing pool iteration utilities — wrap them here. */
    /* (Implementation: linear scan over pool->entries under seg_lock read.) */
    return NULL;  /* TODO: fill in using pool iteration helpers */
}

static void
apply_entries_to_segment(FlushedSegment seg, const xl_dpv_vacuum_entry *entries,
                         int n, bool use_local_idx_directly)
{
    LWLockAcquire(&seg->per_seg_mutex, LW_EXCLUSIVE);
    for (int i = 0; i < n; i++)
    {
        int32 li;
        if (use_local_idx_directly)
        {
            li = (int32) entries[i].local_idx;
        }
        else
        {
            li = dpv_sorted_perm_find(seg, &entries[i].tid);
            if (li < 0)
                continue;  /* tid not in this segment; spec §10 slow paths note it lives in exactly one predecessor */
        }
        SET_SLOT(seg->bitmap, li);
    }
    seg->delete_count += n;  /* approximate; for diagnostics */
    LWLockRelease(&seg->per_seg_mutex);
}

static void
write_local_subversion(FlushedSegment seg, uint32 subversion)
{
    /* Reuse the primary's writer. write_bitmap_file_with_subversion is in
     * lsmindex_io.c. */
    write_bitmap_file_with_subversion(seg, subversion);
}

void
dpv_apply_vacuum_tombstones(const xl_dpv_segment_vacuum_tombstones *hdr,
                            const xl_dpv_vacuum_entry *entries)
{
    LSMIndexBufferSlot *slot = lookup_lsm_index_buffer_slot(hdr->indexRelId);

    if (slot == NULL || pg_atomic_read_u32(&slot->valid) != 1)
        return;

    /* Path 3 (memtable target): owner is still a memtable. */
    if (hdr->is_memtable_target)
    {
        ConcurrentMemTable mt = find_memtable_by_sid(slot, hdr->owner_start_sid);
        if (mt == NULL)
        {
            /* Memtable absent on this standby — must have been adopted into a
             * segment. Per spec §10, vacuum batches also emit a segment-targeted
             * record, so the deletion will land via that. Skip here. */
            return;
        }
        LWLockAcquire(&mt->mt_lock, LW_EXCLUSIVE);
        for (int i = 0; i < hdr->entry_count; i++)
            SET_SLOT(mt->bitmap, entries[i].local_idx);
        LWLockRelease(&mt->mt_lock);
        return;  /* no subversion file for memtables */
    }

    /* Path 1: fast path — exact range, exact version. */
    LWLockAcquire(&slot->pool->seg_lock, LW_SHARED);
    {
        FlushedSegment seg = find_segment_exact(slot, hdr->owner_start_sid,
                                                  hdr->owner_end_sid);
        if (seg && seg->version == hdr->owner_version)
        {
            apply_entries_to_segment(seg, entries, hdr->entry_count,
                                      /*use_local_idx_directly=*/true);
            write_local_subversion(seg, hdr->subversion);
            LWLockRelease(&slot->pool->seg_lock);
            return;
        }
        if (seg && seg->version < hdr->owner_version)
        {
            /* Path 2 (rebuild race): same range, older version. Apply via tid. */
            apply_entries_to_segment(seg, entries, hdr->entry_count,
                                      /*use_local_idx_directly=*/false);
            write_local_subversion(seg, hdr->subversion);
            LWLockRelease(&slot->pool->seg_lock);
            return;
        }
    }
    LWLockRelease(&slot->pool->seg_lock);

    /* Path 3a (merge race): multiple narrower predecessors. Iterate the pool
     * and apply each tid to whichever predecessor owns it. */
    LWLockAcquire(&slot->pool->seg_lock, LW_SHARED);
    {
        int overlap_i = 0;
        FlushedSegment seg;
        bool any_applied = false;

        while ((seg = find_segment_overlap(slot, hdr->owner_start_sid,
                                            hdr->owner_end_sid, overlap_i++)) != NULL)
        {
            /* For each entry not yet applied, attempt tid lookup in seg. */
            int applied = 0;
            LWLockAcquire(&seg->per_seg_mutex, LW_EXCLUSIVE);
            for (int i = 0; i < hdr->entry_count; i++)
            {
                int32 li = dpv_sorted_perm_find(seg, &entries[i].tid);
                if (li >= 0)
                {
                    SET_SLOT(seg->bitmap, li);
                    applied++;
                }
            }
            seg->delete_count += applied;
            LWLockRelease(&seg->per_seg_mutex);
            if (applied > 0)
            {
                write_local_subversion(seg, hdr->subversion);
                any_applied = true;
            }
        }

        if (!any_applied)
        {
            /* Path 4: memtable target despite is_memtable_target==0 — happens
             * when the primary's flush+vacuum interleaved oddly. Fall back to
             * looking up each tid in memtables. */
            for (int i = 0; i < hdr->entry_count; i++)
            {
                /* Iterate slot->memtable_buffer; for each memtable mt: linear
                 * search mt->tids[] for entries[i].tid; if found, set bit.
                 * Memtables are small; linear is fine. */
                apply_to_any_memtable(slot, &entries[i].tid);
            }
        }
    }
    LWLockRelease(&slot->pool->seg_lock);
}

static void
apply_to_any_memtable(LSMIndexBufferSlot *slot, ItemPointer tid)
{
    /* Iterate slot->memtable_buffer; for the memtable whose tids[] contains tid,
     * SET_SLOT(mt->bitmap, j). Use existing memtable iteration helpers. */
    /* (Trivial implementation — fill in based on memtable_buffer layout in lsmindex.h.) */
}
```

The functions `find_segment_overlap`, `apply_to_any_memtable` are static helpers — implement them by reading the pool / memtable iteration utilities in [lsm_segment.c](../../../pgvector/src/lsm_segment.c) and [lsmindex.c](../../../pgvector/src/lsmindex.c).

- [ ] **Step 2: Build, commit**

```bash
# Add src/segment_vacuum_redo.o to OBJS
cd pgvector && make 2>&1 | tail -10
git add pgvector/src/segment_vacuum_redo.h pgvector/src/segment_vacuum_redo.c pgvector/Makefile
git commit -m "feat: vacuum tombstone redo (fast + slow paths)"
```

---

## Phase 4 — WAL-emit-before-file-write protocol on primary

**This is the most important phase of the entire replication effort.** Read spec §11 in full before starting. The protocol:

> For every event that produces a new bitmap subversion file:
>   1. Compute the change.
>   2. `XLogInsert(SegmentVacuumTombstones)` to obtain `LSN_record`.
>   3. `XLogFlush(LSN_record)` so the WAL is durable.
>   4. Write the file via temp-then-rename.

If any of these steps is reordered, the merge-race scenario in spec §11 produces a window where the standby's index is inconsistent with the standby's heap. The reordering is **silent** under most workloads and only manifests under concurrent vacuum + merge — exactly the kind of bug that escapes manual testing.

### Task 4.1: Identify the three call sites

**Files:**
- Read: `pgvector/src/lsmindex.c` — `bulk_delete_lsm_index`

The three sites per spec §11:
1. [lsmindex.c:1703](../../../pgvector/src/lsmindex.c#L1703) — step 1 (memtable already persisted, vacuum applies to its segment)
2. [lsmindex.c:1868](../../../pgvector/src/lsmindex.c#L1868) — step 2 (immutable memtable persisted)
3. [lsmindex.c:1994](../../../pgvector/src/lsmindex.c#L1994) — step 3 (segment-only vacuum)

- [ ] **Step 1: Read the three sites**

Open [lsmindex.c:1700-2010](../../../pgvector/src/lsmindex.c#L1700-L2010) and locate the three `write_bitmap_file_with_subversion` invocations. At each, identify:
- The segment being modified (`FlushedSegment seg`).
- The new subversion number being assigned (let's call it `new_subversion`).
- The list of `(local_idx, tid)` entries being deleted (in step 1 and 2, also the per-tid memtable tombstones).

### Task 4.2: Convert site 3 (segment-only vacuum) first — simplest

**Files:**
- Modify: `pgvector/src/lsmindex.c`

At line 1994 surroundings, the structure is roughly:

```c
// Before:
LWLockAcquire(&seg->per_seg_mutex, LW_EXCLUSIVE);
... apply tombstones to seg->bitmap in memory ...
new_subversion = ++seg->subversion;
LWLockRelease(&seg->per_seg_mutex);

write_bitmap_file_with_subversion(seg, new_subversion);
```

- [ ] **Step 1: Insert WAL emit + flush BEFORE `write_bitmap_file_with_subversion`**

```c
#include "replication_rmgr.h"
#include "replication_gucs.h"
#include "access/xlog.h"
...

// After: bitmap-in-memory update + new_subversion assignment, BEFORE the file write.
if (dpv_replication_role == DPV_ROLE_PRIMARY)
{
    XLogRecPtr lsn;
    /* `entries` here is the same (local_idx, tid) array we just used to update
     * the in-memory bitmap. Construct it as xl_dpv_vacuum_entry[]. */
    lsn = dpv_emit_segment_vacuum_tombstones(
        RelationGetRelid(index),
        seg->start_sid, seg->end_sid, seg->version,
        new_subversion, /*is_memtable_target=*/false,
        entries, entry_count);
    XLogFlush(lsn);  /* CRITICAL: this is the §11 protocol */
}

write_bitmap_file_with_subversion(seg, new_subversion);
```

You will need to construct the `xl_dpv_vacuum_entry` array from the data already in scope. If the existing code uses a different shape (e.g., a `BulkDeleteWorkData` struct that stores `(local_idx, tid)` per to-be-deleted entry), translate. If the local-idx values aren't materialized in scope, capture them at the point where bits are set — wrap the bit-setting loop in a translation that emits both the bit-set and the entry collection.

### Task 4.3: Convert sites 1 and 2 (memtable-persisted paths)

Sites 1 and 2 are vacuum-on-a-memtable-whose-corresponding-segment-already-exists. They emit BOTH `MemtableTombstone` (one per tid, replaced in Plan 1) AND `SegmentVacuumTombstones` (one batch). The latter targets the segment whose flush replaced the memtable (so `is_memtable_target=false`); but if a primary-side race means the segment file does not exist yet on the primary at the time of vacuum, the spec says to mark `is_memtable_target=true`.

Read the existing code carefully to identify which branch fires when. The simplest robust rule: if the vacuum loop's current owner is `mt` (memtable handle, no segment file yet), set `is_memtable_target=true` and pass `owner_start_sid = mt->sid`, `owner_end_sid = mt->sid`, `owner_version = 0`. Otherwise, treat as a segment target.

- [ ] **Step 1: At sites 1 and 2, apply the same `XLogInsert + XLogFlush` protocol BEFORE the bitmap file write.**

If `MemtableTombstone` records are already being emitted per tid earlier in the loop (Plan 1, Task 4.4), keep that emission — it makes the memtable bitmap durable on the standby for the case where the segment hasn't been adopted yet. Then emit `SegmentVacuumTombstones` once per batch as above, immediately before `write_bitmap_file_with_subversion`.

### Task 4.4: Build, run existing tests, commit

```bash
cd pgvector && make && make install
prove -I test/perl test/t/110_replication_segment_flush.pl
prove -I test/perl test/t/111_replication_segment_merge.pl
prove -I test/perl test/t/112_replication_queue_restart.pl
git add pgvector/src/lsmindex.c
git commit -m "feat: WAL-emit-before-file-write protocol for vacuum (spec §11)"
```

---

## Phase 5 — Version-aware bitmap merge in adoption

Plan 2 used identity translation at adoption. Now that we have `dpv_sorted_perm_find`, replace the identity merge with a tid-keyed union: for each set bit in the predecessor's bitmap, look up the corresponding tid in the predecessor's mapping, then resolve to the new segment's `local_idx` via `dpv_sorted_perm_find(new_seg, tid)` and set that bit.

### Task 5.1: Replace identity merge

**Files:**
- Modify: `pgvector/src/segment_adoption.c`

- [ ] **Step 1: Replace the identity translation in `load_and_swap_in_segment_grouped`**

The structure becomes:

```c
static void
union_bitmap_via_tid(FlushedSegment new_seg, FlushedSegment old_seg)
{
    /* old_seg may have a different mapping → tid layout from new_seg.
     * For each bit set in old_seg->bitmap, look up its tid in old_seg's
     * mapping, then resolve to new_seg's local_idx via sorted permutation. */
    uint32 N_old = old_seg->N;
    for (uint32 i = 0; i < N_old; i++)
    {
        if (!IS_SLOT_SET(old_seg->bitmap, i))
            continue;
        ItemPointer tid = &old_seg->map_ptr[i];
        int32 new_li = dpv_sorted_perm_find(new_seg, tid);
        if (new_li >= 0)
            SET_SLOT(new_seg->bitmap, new_li);
        /* If new_li < 0: this tid is no longer present in new_seg
         * (compacted out by REBUILD_DELETION). No action needed —
         * the tid is gone from the index entirely. */
    }
}
```

And for a memtable predecessor:

```c
static void
union_bitmap_from_memtable(FlushedSegment new_seg, ConcurrentMemTable mt)
{
    for (uint32 i = 0; i < mt->slot_count; i++)
    {
        if (!IS_SLOT_SET(mt->bitmap, i))
            continue;
        ItemPointer tid = &mt->tids[i];
        int32 new_li = dpv_sorted_perm_find(new_seg, tid);
        if (new_li >= 0)
            SET_SLOT(new_seg->bitmap, new_li);
    }
}
```

Wire both into the group adoption — call once per predecessor segment, and once per predecessor memtable. Remove the identity-translation block and its accompanying `NOTE` comment.

- [ ] **Step 2: Run existing tests, commit**

```bash
prove -I test/perl test/t/110_replication_segment_flush.pl
prove -I test/perl test/t/111_replication_segment_merge.pl
git add pgvector/src/segment_adoption.c
git commit -m "feat: tid-keyed bitmap union during adoption"
```

---

## Phase 6 — Disable write workers on the standby

### Task 6.1: Gate `lsm_index_bgworker_main`

**Files:**
- Modify: `pgvector/src/lsmbackground.c`

- [ ] **Step 1: At the very top of `lsm_index_bgworker_main`** ([lsmbackground.c:252](../../../pgvector/src/lsmbackground.c#L252)), after the signal-handler setup, add:

```c
#include "utils/wait_event.h"
#include "access/xlog.h"
...
    /* Replication: this worker performs flushes (WAL-emitting writes). On a
     * standby it has no work; exit. The worker will be restarted on promotion
     * via bgw_restart_time. */
    if (RecoveryInProgress())
        proc_exit(0);
```

### Task 6.2: Gate merge thread pool inside vector_index_worker

**Files:**
- Modify: `pgvector/src/vector_index_worker.c`

- [ ] **Step 1: In `vector_index_worker_main`**, locate the call to `init_merge_thread_pool` around [vector_index_worker.c:1572](../../../pgvector/src/vector_index_worker.c#L1572). Wrap:

```c
if (!RecoveryInProgress())
    init_merge_thread_pool(...);
else
    elog(LOG, "vector_index_worker: skipping merge pool init under recovery");
```

- [ ] **Step 2: In the maintenance task dispatcher** (where `SEGMENT_UPDATE_*` tasks are dequeued and dispatched), early-skip with a warning if under recovery — those tasks shouldn't originate locally on the standby, but defense-in-depth:

```c
if (RecoveryInProgress())
{
    elog(WARNING, "vector_index_worker: ignoring local maintenance task on standby");
    continue;
}
```

- [ ] **Step 3: Build, run existing tests, commit**

```bash
cd pgvector && make && make install
prove -I test/perl test/t/110_replication_segment_flush.pl
git add pgvector/src/lsmbackground.c pgvector/src/vector_index_worker.c
git commit -m "feat: disable write workers on standby under recovery"
```

---

## Phase 7 — Standby query barrier

### Task 7.1: GUC

**Files:**
- Modify: `pgvector/src/replication_gucs.h`, `pgvector/src/replication_gucs.c`

- [ ] **Step 1: Add the timeout GUC**

```c
extern int dpv_replication_fetch_wait_timeout_ms;
```

In the `.c`:

```c
int dpv_replication_fetch_wait_timeout_ms = 30000;
...
    DefineCustomIntVariable("pgvector.replication_fetch_wait_timeout",
        "Time (ms) a standby query waits for fetcher catchup before erroring.",
        NULL, &dpv_replication_fetch_wait_timeout_ms, 30000,
        100, INT_MAX, PGC_SIGHUP, GUC_UNIT_MS, NULL, NULL, NULL);
```

### Task 7.2: Add per-slot `not_queryable_until_lsn`

**Files:**
- Modify: `pgvector/src/lsmindex.h`

- [ ] **Step 1: Add field to `LSMIndexBufferSlot`**

```c
    /* Replication: backends waiting for fetcher catchup sleep on this CV
     * while pg_atomic_read_u64(&not_queryable_until_lsn) > 0. The fetcher
     * sets it to 0 when all SegmentCreated/Replaced records since the value
     * of restartpoint_lsn at attach time have corresponding files locally. */
    pg_atomic_uint64 not_queryable_until_lsn;
    ConditionVariable not_queryable_cv;
```

Initialize these in the slot-creation path (search for where `LSMIndexBufferSlot` is set up — likely in `lsm_index_buffer_shmem_initialize` in [lsmindex.c](../../../pgvector/src/lsmindex.c)).

### Task 7.3: At attach time, set the barrier on the standby

**Files:**
- Modify: `pgvector/src/index_load_worker.c` or `lsmindex.c` (`load_lsm_index_internal`)

- [ ] **Step 1: At the end of `load_lsm_index_internal`**, on the standby only, set the barrier:

```c
#include "access/xlog.h"
...
    if (RecoveryInProgress())
    {
        XLogRecPtr lsn = GetXLogReplayRecPtr(NULL);
        pg_atomic_write_u64(&slot->not_queryable_until_lsn, (uint64) lsn);
        ConditionVariableBroadcast(&slot->not_queryable_cv);
    }
```

`GetXLogReplayRecPtr` returns the standby's last-replayed LSN. The barrier is "queries must wait until the fetcher confirms every SegmentCreated/Replaced record up to this LSN is reflected on disk."

### Task 7.4: Clear the barrier from the fetcher

**Files:**
- Modify: `pgvector/src/segment_fetcher.c`

After draining the queue (the bgworker's main loop sees no pending entries for a configurable threshold), check each `LSMIndexBufferSlot` and clear its barrier:

- [ ] **Step 1: After each successful adoption**, the fetcher checks if `not_queryable_until_lsn != 0` for that index, and if so:
  - Scans the queue for pending entries with `source_lsn < not_queryable_until_lsn` for this index.
  - If none remain, clears the barrier and broadcasts the CV.

Sketch:

```c
static void
maybe_clear_barrier(Oid indexRelId)
{
    LSMIndexBufferSlot *slot = lookup_lsm_index_buffer_slot(indexRelId);
    XLogRecPtr barrier_lsn;
    if (slot == NULL) return;
    barrier_lsn = (XLogRecPtr) pg_atomic_read_u64(&slot->not_queryable_until_lsn);
    if (barrier_lsn == 0) return;

    /* Walk the queue dir; if any pending entry for indexRelId has source_lsn
     * <= barrier_lsn, we still have work. Otherwise, clear. */
    if (!queue_has_pending_below(indexRelId, barrier_lsn))
    {
        pg_atomic_write_u64(&slot->not_queryable_until_lsn, 0);
        ConditionVariableBroadcast(&slot->not_queryable_cv);
    }
}
```

`queue_has_pending_below` is a new helper in `pending_fetch_queue.c` that scans the queue dir.

### Task 7.5: Backend-side wait at search entry

**Files:**
- Modify: `pgvector/src/lsmindex.c` (`search_lsm_index` at [lsmindex.c:1525](../../../pgvector/src/lsmindex.c#L1525)) or the per-search entry function

- [ ] **Step 1: At the top of `search_lsm_index`** (or wherever a per-search slot resolution happens on the standby), wait for the barrier:

```c
#include "replication_gucs.h"
...
    if (RecoveryInProgress())
    {
        XLogRecPtr barrier = (XLogRecPtr) pg_atomic_read_u64(&slot->not_queryable_until_lsn);
        if (barrier != 0)
        {
            TimestampTz start = GetCurrentTimestamp();
            ConditionVariablePrepareToSleep(&slot->not_queryable_cv);
            for (;;)
            {
                if (pg_atomic_read_u64(&slot->not_queryable_until_lsn) == 0)
                    break;
                if (TimestampDifferenceExceeds(start, GetCurrentTimestamp(),
                                                dpv_replication_fetch_wait_timeout_ms))
                {
                    ConditionVariableCancelSleep();
                    ereport(ERROR,
                            (errcode(ERRCODE_T_R_QUERY_CANCELED),
                             errmsg("pgvector index is unavailable on standby: "
                                    "fetcher catchup timed out after %d ms",
                                    dpv_replication_fetch_wait_timeout_ms)));
                }
                ConditionVariableTimedSleep(&slot->not_queryable_cv,
                                             dpv_replication_fetch_wait_timeout_ms,
                                             WAIT_EVENT_EXTENSION);
            }
            ConditionVariableCancelSleep();
        }
    }
```

- [ ] **Step 2: Build, commit**

```bash
cd pgvector && make && make install
git add pgvector/src/lsmindex.h pgvector/src/lsmindex.c \
        pgvector/src/index_load_worker.c \
        pgvector/src/segment_fetcher.c pgvector/src/pending_fetch_queue.c \
        pgvector/src/replication_gucs.h pgvector/src/replication_gucs.c
git commit -m "feat: standby query barrier until fetcher catchup"
```

---

## Phase 8 — Integration tests

### Task 8.1: Vacuum replicates

**Files:**
- Create: `pgvector/test/t/120_replication_vacuum_simple.pl`

- [ ] **Step 1: Write the test**

```perl
use strict;
use warnings FATAL => 'all';
use lib 'test/perl';
use DpvReplication qw(setup_primary setup_standby wait_catchup);
use Test::More;

my $primary = setup_primary();
my $standby = setup_standby($primary);

my $dim = 32;
my $array_sql = join(",", ('random()') x $dim);

$primary->safe_psql('postgres', 'CREATE EXTENSION vector;');
$primary->safe_psql('postgres', qq(
    CREATE TABLE t (id int, v vector($dim));
    CREATE INDEX t_v_idx ON t USING <YOUR_LSM_AM_NAME> (v vector_l2_ops);
));

# Drive enough rows to flush at least one segment, then vacuum half of them.
$primary->safe_psql('postgres',
    "INSERT INTO t SELECT i, ARRAY[$array_sql]::vector FROM generate_series(1, 50000) i;");
$primary->safe_psql('postgres', "SELECT pg_sleep(3);");  # allow flush
wait_catchup($primary, $standby);

# Wait for the standby to adopt.
for (my $i = 0; $i < 30; $i++) {
    last if $standby->safe_psql('postgres',
        "SELECT count(*) FROM pg_ls_dir('pgvector_storage/_pending_fetches')") eq '0';
    sleep 1;
}

$primary->safe_psql('postgres', "DELETE FROM t WHERE id <= 25000;");
$primary->safe_psql('postgres', "VACUUM t;");
wait_catchup($primary, $standby);

# After vacuum, the index should return only 25000 rows.
my $count = $standby->safe_psql('postgres', q(
    SET enable_seqscan = off;
    SELECT count(*) FROM (
        SELECT * FROM t
        ORDER BY v <-> '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector
        LIMIT 50000
    ) sub;
));
is($count, '25000', "vacuum tombstones replicate to standby");

done_testing();
```

- [ ] **Step 2: Run, commit**

```bash
prove -I test/perl test/t/120_replication_vacuum_simple.pl
git add pgvector/test/t/120_replication_vacuum_simple.pl
git commit -m "test: vacuum tombstones replicate to standby"
```

### Task 8.2: Concurrent vacuum + merge race

**Files:**
- Create: `pgvector/test/t/121_replication_vacuum_merge_race.pl`

This is the §11 motivating scenario. Hard to provoke deterministically with normal SQL — the race requires vacuum and merge to interleave inside a specific window of time on the primary. The test approach:

1. Build the standby on a slow disk / under high concurrency to amplify timing.
2. On the primary, run vacuum and merge concurrently (in separate sessions).
3. After both complete, verify the standby's pool bitmap matches the primary's.

Without deterministic injection points, this test is best-effort. A stronger test adds a debugging hook (compile-time `#define DPV_TEST_INJECT_VACUUM_MERGE_RACE`) that pauses vacuum after `XLogFlush` but before the file write, and lets merge run in between. For v1 testing, accept the best-effort variant and document the gap.

- [ ] **Step 1: Test**

```perl
use strict;
use warnings FATAL => 'all';
use lib 'test/perl';
use DpvReplication qw(setup_primary setup_standby wait_catchup);
use Test::More;

my $primary = setup_primary();
my $standby = setup_standby($primary);

my $dim = 32;
my $array_sql = join(",", ('random()') x $dim);

$primary->safe_psql('postgres', 'CREATE EXTENSION vector;');
$primary->safe_psql('postgres', qq(
    CREATE TABLE t (id int, v vector($dim));
    CREATE INDEX t_v_idx ON t USING <YOUR_LSM_AM_NAME> (v vector_l2_ops);
));

# Drive multiple flushes.
for my $batch (1..4) {
    $primary->safe_psql('postgres', "INSERT INTO t SELECT i, ARRAY[$array_sql]::vector "
                                  . "FROM generate_series(1 + 30000*($batch-1), 30000*$batch) i;");
    $primary->safe_psql('postgres', "SELECT pg_sleep(1);");
}
$primary->safe_psql('postgres', "SELECT pg_sleep(3);");
wait_catchup($primary, $standby);

# Now run vacuum and merge concurrently.
my $vacuum_pid = fork();
if ($vacuum_pid == 0) {
    $primary->safe_psql('postgres', "DELETE FROM t WHERE id % 4 = 0; VACUUM t;");
    exit 0;
}
my $merge_pid = fork();
if ($merge_pid == 0) {
    $primary->safe_psql('postgres', "SELECT pgvector_force_merge('t_v_idx'::regclass);");
    exit 0;
}
waitpid($vacuum_pid, 0);
waitpid($merge_pid,  0);

$primary->safe_psql('postgres', "SELECT pg_sleep(2);");
wait_catchup($primary, $standby);

# Allow fetcher to drain.
for (my $i = 0; $i < 30; $i++) {
    last if $standby->safe_psql('postgres',
        "SELECT count(*) FROM pg_ls_dir('pgvector_storage/_pending_fetches')") eq '0';
    sleep 1;
}

# Verify primary and standby return the same row count.
my $primary_count = $primary->safe_psql('postgres', q(
    SET enable_seqscan = off;
    SELECT count(*) FROM (SELECT * FROM t ORDER BY v <-> '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector LIMIT 200000) sub;
));
my $standby_count = $standby->safe_psql('postgres', q(
    SET enable_seqscan = off;
    SELECT count(*) FROM (SELECT * FROM t ORDER BY v <-> '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector LIMIT 200000) sub;
));
is($standby_count, $primary_count,
   "standby and primary agree on row count after concurrent vacuum+merge");

done_testing();
```

- [ ] **Step 2: Run, commit**

```bash
prove -I test/perl test/t/121_replication_vacuum_merge_race.pl
git add pgvector/test/t/121_replication_vacuum_merge_race.pl
git commit -m "test: concurrent vacuum+merge converges on standby"
```

### Task 8.3: Rebuild with compaction (translation)

**Files:**
- Create: `pgvector/test/t/122_replication_rebuild_translation.pl`

- [ ] **Step 1: Test**

```perl
use strict;
use warnings FATAL => 'all';
use lib 'test/perl';
use DpvReplication qw(setup_primary setup_standby wait_catchup);
use Test::More;

my $primary = setup_primary();
my $standby = setup_standby($primary);

my $dim = 32;
my $array_sql = join(",", ('random()') x $dim);

$primary->safe_psql('postgres', 'CREATE EXTENSION vector;');
$primary->safe_psql('postgres', qq(
    CREATE TABLE t (id int, v vector($dim));
    CREATE INDEX t_v_idx ON t USING <YOUR_LSM_AM_NAME> (v vector_l2_ops);
));
$primary->safe_psql('postgres',
    "INSERT INTO t SELECT i, ARRAY[$array_sql]::vector FROM generate_series(1, 60000) i;");
$primary->safe_psql('postgres', "SELECT pg_sleep(2);");
wait_catchup($primary, $standby);

# Mark half deleted, then trigger a REBUILD_DELETION on the primary that compacts.
$primary->safe_psql('postgres', "DELETE FROM t WHERE id % 2 = 0; VACUUM t;");
$primary->safe_psql('postgres', "SELECT pgvector_force_rebuild('t_v_idx'::regclass);");
$primary->safe_psql('postgres', "SELECT pg_sleep(2);");
wait_catchup($primary, $standby);

for (my $i = 0; $i < 30; $i++) {
    last if $standby->safe_psql('postgres',
        "SELECT count(*) FROM pg_ls_dir('pgvector_storage/_pending_fetches')") eq '0';
    sleep 1;
}

# The standby should reflect the post-rebuild state: 30000 rows.
my $count = $standby->safe_psql('postgres', q(
    SET enable_seqscan = off;
    SELECT count(*) FROM (SELECT * FROM t ORDER BY v <-> '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector LIMIT 60000) sub;
));
is($count, '30000', "standby reflects post-rebuild compacted state");

done_testing();
```

`pgvector_force_rebuild` is a test helper analogous to `pgvector_force_merge`. Add to [vector.c](../../../pgvector/src/vector.c) if missing.

- [ ] **Step 2: Run, commit**

```bash
prove -I test/perl test/t/122_replication_rebuild_translation.pl
git add pgvector/test/t/122_replication_rebuild_translation.pl pgvector/src/vector.c
git commit -m "test: rebuild with compaction translates correctly on standby"
```

### Task 8.4: Standby attach barrier

**Files:**
- Create: `pgvector/test/t/123_replication_attach_barrier.pl`

The scenario: standby has a fresh `LSMIndexBufferSlot`. Pending fetches exist. A backend on the standby attempts a query; it must wait, then succeed when the fetcher catches up.

- [ ] **Step 1: Test**

```perl
use strict;
use warnings FATAL => 'all';
use lib 'test/perl';
use DpvReplication qw(setup_primary setup_standby wait_catchup);
use Test::More;

my $primary = setup_primary();
$primary->append_conf('postgresql.conf', q(
pgvector.replication_fetch_wait_timeout = 30000
));
$primary->restart;

my $standby = setup_standby($primary);
$standby->append_conf('postgresql.conf', q(
pgvector.replication_fetch_wait_timeout = 30000
));
$standby->restart;

my $dim = 32;
my $array_sql = join(",", ('random()') x $dim);

$primary->safe_psql('postgres', 'CREATE EXTENSION vector;');
$primary->safe_psql('postgres', qq(
    CREATE TABLE t (id int, v vector($dim));
    CREATE INDEX t_v_idx ON t USING <YOUR_LSM_AM_NAME> (v vector_l2_ops);
    INSERT INTO t SELECT i, ARRAY[$array_sql]::vector FROM generate_series(1, 50000) i;
));
$primary->safe_psql('postgres', "SELECT pg_sleep(2);");
wait_catchup($primary, $standby);

# Crash-restart the standby while fetches may be pending.
$standby->stop('immediate');
$standby->start;

# Query should wait on the barrier, then succeed.
my $count = $standby->safe_psql('postgres', q(
    SET enable_seqscan = off;
    SELECT count(*) FROM (SELECT * FROM t ORDER BY v <-> '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector LIMIT 50000) sub;
));
is($count, '50000', "standby query waits for fetcher and then returns all rows");

done_testing();
```

- [ ] **Step 2: Run, commit**

```bash
prove -I test/perl test/t/123_replication_attach_barrier.pl
git add pgvector/test/t/123_replication_attach_barrier.pl
git commit -m "test: standby attach barrier waits for fetcher catchup"
```

---

## Self-review checklist for Plan 3

- [ ] **§11 protocol enforced at all three vacuum sites.** Grep for `write_bitmap_file_with_subversion` in [lsmindex.c](../../../pgvector/src/lsmindex.c); each occurrence must be preceded by `dpv_emit_segment_vacuum_tombstones + XLogFlush` (under the `dpv_replication_role == DPV_ROLE_PRIMARY` guard).
- [ ] **Standby fast-path/slow-path coverage.** Tests 120, 121, 122 exercise all four paths (fast, rebuild race, merge race, memtable target). 121 covers concurrent vacuum+merge.
- [ ] **`sorted_idx` field added to `FlushedSegment`** and freed on segment unload.
- [ ] **Adoption now uses tid-keyed bitmap union** for both segment and memtable predecessors.
- [ ] **Write workers (`lsm_index_bgworker`, merge pool) gated on `!RecoveryInProgress()`** at entry.
- [ ] **Attach barrier set on standby after `load_lsm_index_internal`**, cleared by fetcher after queue drains for that index.
- [ ] **All test files from Plans 1 & 2 still pass.**
- [ ] **GUC `pgvector.replication_fetch_wait_timeout` is documented in the project's GUC docs (or added inline to a CLAUDE.md/README section if no GUC docs exist yet).**

---

## Open follow-ons (out of v1)

These are explicitly out-of-scope for v1 per spec §17, but noted here so future work can pick them up:

1. **Non-blocking redo during index load (Option C, spec §9 future).** Replace the current "block on `load_cv`" with a per-index ring buffer drained by the loader.
2. **Promotion / failover.** Brand-new design space.
3. **Multi-standby fan-out.** The current file server is single-connection; supporting many concurrent connections requires either a per-connection worker pool on the primary or moving to an `accept + fork-worker` model.
4. **Logical replication.** Out of scope.
5. **Removing `VECTOR_STORAGE_BASE_DIR` as a GUC and unifying with PG's tablespace machinery.** A future cleanup.
6. **Batched subversion writes.** Spec §10 notes per-redo subversion writes; if profiling shows fsync rate is a bottleneck, batch.

---

## End of Plan 3
