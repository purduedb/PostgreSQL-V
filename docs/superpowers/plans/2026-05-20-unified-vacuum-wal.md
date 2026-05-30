# Unified per-sid Vacuum WAL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Git policy for this repository:** The user owns all git operations (commits, branches, pushes). DO NOT run `git add`, `git commit`, or any other state-changing git command. Build verification (`make`) and regression tests (`prove`) are the appropriate checkpoints between phases.

**Goal:** Refactor Plan 3's vacuum replication to use a single per-sid WAL record, drop the `sorted_perm` aux structure, and apply tombstones via O(n) 2-pointer merges over the offset-order invariant.

**Architecture:** Replaces the two existing WAL types (`XLOG_DPV_MEMTABLE_TOMBSTONE` 0x20 and the old `XLOG_DPV_SEGMENT_VACUUM_TOMBSTONES` 0x80) with one record (`XLOG_DPV_VACUUM_TOMBSTONES`) per touched sid per vacuum batch. The primary consults the segment offset file at emit time to attribute each vacuumed `local_idx` to a specific sid. The standby caches `SegmentOffsetRange[]` on `FlushedSegmentData`, then applies each record via a single advancing-pointer walk over `seg->map_ptr[offsets[sid].start_offset..end_offset)`. Adoption's bitmap union helpers switch to the same 2-pointer merge, eliminating `sorted_perm` entirely. Boundary signaling per sid is intrinsic — each WAL record is one sid's batch.

**Tech Stack:** PostgreSQL 17 custom rmgr (`replication_rmgr.{c,h}`), C, GenericXLog + XLogInsert + XLogFlush, DSM for ADOPT task payload.

**Spec:** [`docs/superpowers/specs/2026-05-20-unified-vacuum-wal-design.md`](../specs/2026-05-20-unified-vacuum-wal-design.md)

---

## File structure

| File | Role | Action |
|------|------|--------|
| [pgvector/src/replication_rmgr.h](../../pgvector/src/replication_rmgr.h) | Defines new `xl_dpv_vacuum_tombstones`, retires old types, declares new emit helper. | Modify |
| [pgvector/src/replication_rmgr.c](../../pgvector/src/replication_rmgr.c) | New emit helper + redo callback dispatch; deletes old emit/redo for the two retired types. | Modify |
| [pgvector/src/segment_vacuum_redo.h](../../pgvector/src/segment_vacuum_redo.h) | Updated signature for `dpv_apply_vacuum_tombstones`. | Modify |
| [pgvector/src/segment_vacuum_redo.c](../../pgvector/src/segment_vacuum_redo.c) | Rewritten with pool-first / memtable-fallback routing + 2-pointer merge; drops slow paths. | Modify |
| [pgvector/src/lsm_segment.h](../../pgvector/src/lsm_segment.h) | Adds `offsets[]` + `offsets_count` to `FlushedSegmentData`; removes `sorted_idx`. | Modify |
| [pgvector/src/lsm_segment.c](../../pgvector/src/lsm_segment.c) | Allocates/frees `offsets[]` in load/cleanup/discard paths; removes `dpv_sorted_perm_free` calls. | Modify |
| [pgvector/src/lsmindex.c](../../pgvector/src/lsmindex.c) | Three `bulk_delete_lsm_index` sites switch to per-sid emission; multi-sid uses offset file. | Modify |
| [pgvector/src/statuspage.c](../../pgvector/src/statuspage.c) | Removes the standalone `dpv_emit_memtable_tombstone` call. | Modify |
| [pgvector/src/segment_adoption.h](../../pgvector/src/segment_adoption.h) | Declares `DpvVacuumGroup`; updates `dpv_pool_adopt` signature. | Modify |
| [pgvector/src/segment_adoption.c](../../pgvector/src/segment_adoption.c) | New `union_segment_bitmap_via_offset_merge` and `union_deletion_entries_into_new_seg`; signature change. | Modify |
| [pgvector/src/segment_fetcher.c](../../pgvector/src/segment_fetcher.c) | Per-sid grouping of cover deletion bits; updated ADOPT payload assembly. | Modify |
| [pgvector/src/tasksend.h](../../pgvector/src/tasksend.h) | `dpv_send_adopt_task` / `segment_update_blocking` signature change for group payload. | Modify |
| [pgvector/src/tasksend.c](../../pgvector/src/tasksend.c) | DSM trailer serialization rewritten for groups. | Modify |
| [pgvector/src/vector_index_worker.c](../../pgvector/src/vector_index_worker.c) | DSM trailer deserialization rewritten for groups; field renames. | Modify |
| [pgvector/src/standby_memtable.h](../../pgvector/src/standby_memtable.h) | Removes `dpv_standby_memtable_tombstone` declaration. | Modify |
| [pgvector/src/standby_memtable.c](../../pgvector/src/standby_memtable.c) | Removes `dpv_standby_memtable_tombstone` definition. | Modify |
| [pgvector/src/sorted_perm.h](../../pgvector/src/sorted_perm.h) | Deleted. | Delete |
| [pgvector/src/sorted_perm.c](../../pgvector/src/sorted_perm.c) | Deleted. | Delete |
| [pgvector/Makefile](../../pgvector/Makefile) | Removes `src/sorted_perm.o` from OBJS. | Modify |

---

## Phase 1 — Add new WAL infrastructure (coexisting with old)

This phase introduces the new record type under a **temporary op code 0x90** and the new emit/redo helpers, all coexisting with the existing 0x20 / 0x80 records. At the end of this phase nothing on the primary or standby actually uses the new code path yet — we just have the scaffolding compiled in. The temporary op code is promoted to 0x80 in Phase 7 once the old code is removed.

### Task 1.1: Define new WAL header and entry structs in `replication_rmgr.h`

**Files:**
- Modify: `pgvector/src/replication_rmgr.h`

- [ ] **Step 1: Add new op code (temporary slot)**

In [`replication_rmgr.h`](../../pgvector/src/replication_rmgr.h), insert AFTER the existing `XLOG_DPV_SEGMENT_VACUUM_TOMBSTONES 0x80` line:

```c
/*
 * Plan 3 refactor: unified per-sid vacuum record. During Phase 1–6 of the
 * refactor this lives under op code 0x90 so it coexists with the legacy
 * 0x20 / 0x80 records. Phase 7 retires the legacy records and moves this
 * to 0x80 permanently.
 */
#define XLOG_DPV_VACUUM_TOMBSTONES         0x90
```

- [ ] **Step 2: Add new payload structs**

Insert AFTER the new `#define` line:

```c
/*
 * Per-entry payload for the unified vacuum record. Entries are emitted in
 * ascending sid_local_idx order — i.e. physical offset order within the
 * sid's range in the primary's map_ptr / mt->tids.
 */
typedef struct
{
    uint32  sid_local_idx;
    int64_t tid;
} xl_dpv_vacuum_entry_v2;

/*
 * Variable-length record. One record per sid touched by a vacuum batch.
 * Payload layout on the wire:
 *   header (32 B, 8-aligned)
 *   xl_dpv_vacuum_entry_v2 entries[entry_count]
 */
typedef struct
{
    Oid       dbOid;                /*  4 */
    Oid       indexRelId;           /*  4 */
    SegmentId sid;                  /*  4 — the exact sid this batch vacuums. */
    uint32    owner_version;        /*  4 — primary's segment version at emit
                                     *     time; 0 if owner is a memtable.   */
    uint32    subversion;           /*  4 — new subversion for the bitmap
                                     *     file; UINT32_MAX if memtable owner. */
    uint32    is_memtable_owner;    /*  4 — 1 = owner is memtable on primary;
                                     *     0 = owner is a flushed segment.   */
    uint32    entry_count;          /*  4 */
    uint32    _pad;                 /*  4 — keeps total 32 B / 8-aligned.    */
    /* followed by xl_dpv_vacuum_entry_v2 entries[entry_count] */
} xl_dpv_vacuum_tombstones;

StaticAssertDecl(sizeof(xl_dpv_vacuum_tombstones) == 32,
                 "xl_dpv_vacuum_tombstones must be exactly 32 bytes");
StaticAssertDecl(sizeof(xl_dpv_vacuum_tombstones) % 8 == 0,
                 "xl_dpv_vacuum_tombstones must be 8-aligned so the entry "
                 "trailer's int64_t tid is naturally aligned");
StaticAssertDecl(sizeof(xl_dpv_vacuum_entry_v2) == 16,
                 "xl_dpv_vacuum_entry_v2 must be exactly 16 bytes");
```

> The temporary type names `xl_dpv_vacuum_tombstones` / `xl_dpv_vacuum_entry_v2` differ from the legacy `xl_dpv_segment_vacuum_tombstones` / `xl_dpv_vacuum_entry` to avoid C type-name collisions during the transition. Phase 7 renames `_v2` → unsuffixed after the legacy types are removed.

- [ ] **Step 3: Add new emit-helper prototype**

Insert near the bottom of [`replication_rmgr.h`](../../pgvector/src/replication_rmgr.h), after `dpv_emit_segment_vacuum_tombstones`:

```c
extern XLogRecPtr dpv_emit_vacuum_tombstones(
    Oid indexRelId,
    SegmentId sid, uint32 owner_version, uint32 subversion,
    bool is_memtable_owner,
    const xl_dpv_vacuum_entry_v2 *entries, int entry_count);
```

### Task 1.2: Implement new emit helper in `replication_rmgr.c`

**Files:**
- Modify: `pgvector/src/replication_rmgr.c`

- [ ] **Step 1: Add emit helper**

At the end of [`replication_rmgr.c`](../../pgvector/src/replication_rmgr.c) (after `dpv_emit_segment_vacuum_tombstones`), add:

```c
XLogRecPtr
dpv_emit_vacuum_tombstones(Oid indexRelId,
                           SegmentId sid, uint32 owner_version, uint32 subversion,
                           bool is_memtable_owner,
                           const xl_dpv_vacuum_entry_v2 *entries, int entry_count)
{
    xl_dpv_vacuum_tombstones hdr;

    Assert(entry_count >= 0);

    hdr = (xl_dpv_vacuum_tombstones) {
        .dbOid             = MyDatabaseId,
        .indexRelId        = indexRelId,
        .sid               = sid,
        .owner_version     = owner_version,
        .subversion        = subversion,
        .is_memtable_owner = is_memtable_owner ? 1u : 0u,
        .entry_count       = (uint32) entry_count,
        ._pad              = 0,
    };

    XLogBeginInsert();
    XLogRegisterData((char *) &hdr, sizeof(hdr));
    if (entry_count > 0)
        XLogRegisterData((char *) entries,
                         (Size) entry_count * sizeof(xl_dpv_vacuum_entry_v2));
    return XLogInsert(RM_DPV_REPLICATION_ID, XLOG_DPV_VACUUM_TOMBSTONES);
}
```

### Task 1.3: Add redo callback dispatch + identify string

**Files:**
- Modify: `pgvector/src/replication_rmgr.c`

- [ ] **Step 1: Declare new redo callback**

In the forward-declarations block near the top of [`replication_rmgr.c`](../../pgvector/src/replication_rmgr.c) (after `static void redo_segment_vacuum_tombstones(XLogReaderState *r);`), add:

```c
static void redo_vacuum_tombstones(XLogReaderState *r);
```

- [ ] **Step 2: Add dispatcher entry**

In `dpv_replication_redo` (the `switch (info)` block), add a case AFTER the existing `XLOG_DPV_SEGMENT_VACUUM_TOMBSTONES` case:

```c
        case XLOG_DPV_VACUUM_TOMBSTONES:   redo_vacuum_tombstones(record);   break;
```

- [ ] **Step 3: Add identify string**

In `dpv_replication_identify`, add a case AFTER `XLOG_DPV_SEGMENT_VACUUM_TOMBSTONES`:

```c
        case XLOG_DPV_VACUUM_TOMBSTONES:   return "VACUUM_TOMBSTONES";
```

- [ ] **Step 4: Define `redo_vacuum_tombstones`**

At the end of the redo-callback block (right after `redo_segment_vacuum_tombstones` in the file), add:

```c
static void
redo_vacuum_tombstones(XLogReaderState *r)
{
    char                     *data    = XLogRecGetData(r);
    xl_dpv_vacuum_tombstones *hdr     = (xl_dpv_vacuum_tombstones *) data;
    xl_dpv_vacuum_entry_v2   *entries =
        (xl_dpv_vacuum_entry_v2 *) (data + sizeof(*hdr));

    if (!InHotStandby)
        return;

    PG_TRY();
    {
        (void) get_lsm_index_idx(hdr->indexRelId, /* for_redo */ true, hdr->dbOid);
    }
    PG_CATCH();
    {
        FlushErrorState();
        return;
    }
    PG_END_TRY();

    /* Plan 3 refactor: route via the new unified entry point. The function
     * signature is updated in Phase 3; until then we keep the legacy
     * routing untouched and leave this callback inert by forwarding to a
     * no-op stub. Phase 3 implements the real apply logic. */
    dpv_apply_vacuum_tombstones_v2(hdr, entries);
}
```

> The function `dpv_apply_vacuum_tombstones_v2` is declared in Task 1.4 and implemented in Phase 3. During Phases 1–2 it is a no-op stub so the build links.

### Task 1.4: Add new apply function header + stub

**Files:**
- Modify: `pgvector/src/segment_vacuum_redo.h`
- Modify: `pgvector/src/segment_vacuum_redo.c`

- [ ] **Step 1: Add prototype**

In [`segment_vacuum_redo.h`](../../pgvector/src/segment_vacuum_redo.h), AFTER the existing `dpv_apply_vacuum_tombstones` declaration, add:

```c
extern void dpv_apply_vacuum_tombstones_v2(
    const xl_dpv_vacuum_tombstones *hdr,
    const xl_dpv_vacuum_entry_v2 *entries);
```

- [ ] **Step 2: Add no-op stub**

At the very end of [`segment_vacuum_redo.c`](../../pgvector/src/segment_vacuum_redo.c), add:

```c
/*
 * Phase 1 placeholder. The real implementation (pool-first/memtable-fallback
 * routing + 2-pointer merge) lands in Phase 3. Until then this is unreachable
 * because no primary site emits XLOG_DPV_VACUUM_TOMBSTONES yet.
 */
void
dpv_apply_vacuum_tombstones_v2(const xl_dpv_vacuum_tombstones *hdr,
                                const xl_dpv_vacuum_entry_v2 *entries)
{
    (void) hdr;
    (void) entries;
    elog(WARNING,
         "[dpv_apply_vacuum_tombstones_v2] unexpected pre-Phase-3 invocation; ignoring");
}
```

### Task 1.5: Build verification for Phase 1

- [ ] **Step 1: Run make**

```bash
cd pgvector && make 2>&1 | tail -30
```

Expected: build succeeds with no errors. Warnings about `dpv_apply_vacuum_tombstones_v2` being defined-but-unused-in-a-translation-unit are acceptable; warnings about the new struct types are bugs (revisit Task 1.1 alignment).

---

## Phase 2 — Cache `offsets[]` on `FlushedSegmentData`

### Task 2.1: Add `offsets[]` fields to `FlushedSegmentData`

**Files:**
- Modify: `pgvector/src/lsm_segment.h`

- [ ] **Step 1: Add fields**

In [`lsm_segment.h`](../../pgvector/src/lsm_segment.h) inside `FlushedSegmentData`, AFTER the `uint32_t *sorted_idx;` field (the `sorted_idx` field is removed later in Phase 7), add:

```c
    /* Per-sid offset table for this segment. Allocated in load_and_set_segment
     * via load_offset_file; freed in cleanup_flushed_segment /
     * discard_reserved_segment. Indexed by sid order; offsets[k].sid is the
     * physical sid, [start_offset, end_offset) is its slice of map_ptr /
     * bitmap_ptr. offsets_count == (segment_id_end - segment_id_start + 1). */
    SegmentOffsetRange *offsets;
    uint32_t            offsets_count;
```

### Task 2.2: Allocate offsets[] in `load_and_set_segment`

**Files:**
- Modify: `pgvector/src/lsm_segment.c`

- [ ] **Step 1: Load offset file after mapping**

In [`lsm_segment.c:586`](../../pgvector/src/lsm_segment.c#L586) (inside `load_and_set_segment`), insert AFTER the `load_mapping_file(...)` line and BEFORE `segment->in_used = true;`:

```c
        /*
         * Plan 3 refactor: cache the per-sid offset table on the segment.
         * Used by vacuum redo (segment_vacuum_redo.c) and adoption union
         * (segment_adoption.c) for the O(n) 2-pointer merge.
         */
        load_offset_file(indexRelId, start_sid, end_sid, version,
                         &segment->offsets, false);
        segment->offsets_count = (uint32_t) (end_sid - start_sid + 1);
```

> `load_offset_file` already allocates with `palloc` when `pg_alloc=true` and with `malloc` when `pg_alloc=false`. We pass `false` so freeing in cleanup uses `free()`, matching `map_ptr` / `bitmap_ptr` ownership.

### Task 2.3: Free offsets[] in `cleanup_flushed_segment`

**Files:**
- Modify: `pgvector/src/lsm_segment.c`

- [ ] **Step 1: Free in cleanup path**

In [`lsm_segment.c:182-186`](../../pgvector/src/lsm_segment.c#L182-L186) (inside `cleanup_flushed_segment`), AFTER the `map_ptr` free block and BEFORE `dpv_sorted_perm_free(segment);`, insert:

```c
    if (segment->offsets != NULL)
    {
        free(segment->offsets);
        segment->offsets = NULL;
    }
    segment->offsets_count = 0;
```

### Task 2.4: Free offsets[] in `discard_reserved_segment`

**Files:**
- Modify: `pgvector/src/lsm_segment.c`

- [ ] **Step 1: Free in discard path**

In [`lsm_segment.c:244-248`](../../pgvector/src/lsm_segment.c#L244-L248) (inside `discard_reserved_segment`), AFTER the `map_ptr` free block and BEFORE `dpv_sorted_perm_free(segment);`, insert:

```c
    if (segment->offsets != NULL)
    {
        free(segment->offsets);
        segment->offsets = NULL;
    }
    segment->offsets_count = 0;
```

### Task 2.5: Build verification for Phase 2

- [ ] **Step 1: Run make**

```bash
cd pgvector && make 2>&1 | tail -20
```

Expected: build succeeds.

---

## Phase 3 — Rewrite standby redo with new routing and 2-pointer merge

### Task 3.1: Replace Phase 1 stub with real apply implementation

**Files:**
- Modify: `pgvector/src/segment_vacuum_redo.c`

- [ ] **Step 1: Delete the Phase 1 stub**

In [`segment_vacuum_redo.c`](../../pgvector/src/segment_vacuum_redo.c), delete the `dpv_apply_vacuum_tombstones_v2` function added in Task 1.4 Step 2 (the WARNING-only stub).

- [ ] **Step 2: Add helper: binary-search offsets[] for a sid**

At the end of [`segment_vacuum_redo.c`](../../pgvector/src/segment_vacuum_redo.c) (before the closing `#endif`/EOF if any), add:

```c
/*
 * Binary search seg->offsets[] (sorted ascending by sid, length offsets_count)
 * for an entry matching `sid`. Returns the index into offsets[] on hit, -1
 * on miss. The offset file is exactly (end_sid - start_sid + 1) entries,
 * one per sid in [segment_id_start, segment_id_end].
 */
static int
find_offset_index_for_sid(const FlushedSegmentData *seg, SegmentId sid)
{
    int lo = 0;
    int hi = (int) seg->offsets_count - 1;

    if (seg->offsets == NULL || hi < 0)
        return -1;

    while (lo <= hi)
    {
        int mid = lo + ((hi - lo) >> 1);
        SegmentId mid_sid = seg->offsets[mid].sid;
        if (mid_sid == sid) return mid;
        if (mid_sid < sid)  lo = mid + 1;
        else                hi = mid - 1;
    }
    return -1;
}
```

- [ ] **Step 3: Add helper: apply to a segment (fast + 2-pointer slow)**

Append:

```c
/*
 * Apply a vacuum batch to a standby-side segment. Caller holds
 * pool->seg_lock at least SHARED; this function takes seg->per_seg_mutex
 * for the bitmap update.
 *
 * Returns true if at least one bit flipped 0→1 (caller may then persist a
 * subversion file).
 */
static bool
apply_to_segment(FlushedSegment seg, int sid_offset_index,
                 const xl_dpv_vacuum_tombstones *hdr,
                 const xl_dpv_vacuum_entry_v2 *entries)
{
    SegmentOffsetRange slice = seg->offsets[sid_offset_index];
    Size start_off = slice.start_offset;
    Size end_off   = slice.end_offset;     /* exclusive */
    bool any_flipped = false;
    bool fast_path   = (seg->version == hdr->owner_version &&
                        hdr->is_memtable_owner == 0);

    pthread_mutex_lock(&seg->per_seg_mutex);

    if (fast_path)
    {
        for (uint32 w = 0; w < hdr->entry_count; w++)
        {
            Size m = start_off + entries[w].sid_local_idx;
            if (m < end_off &&
                !IS_SLOT_SET(seg->bitmap_ptr, m))
            {
                SET_SLOT(seg->bitmap_ptr, m);
                seg->delete_count++;
                any_flipped = true;
            }
        }
    }
    else
    {
        /* Cross-version OR primary-was-memtable: 2-pointer merge.
         * WAL tids are in physical-offset order of the primary's sid range;
         * standby's seg->map_ptr[start_off..end_off) is in the same relative
         * order (compaction-only invariant). */
        uint32 w = 0;
        for (Size m = start_off; m < end_off && w < hdr->entry_count; m++)
        {
            if (seg->map_ptr[m] == entries[w].tid)
            {
                if (!IS_SLOT_SET(seg->bitmap_ptr, m))
                {
                    SET_SLOT(seg->bitmap_ptr, m);
                    seg->delete_count++;
                    any_flipped = true;
                }
                w++;
            }
            /* else: standby-side tid not in this WAL batch — skip. */
        }
        /* Trailing unmatched WAL entries are silently dropped (compacted
         * out of the standby by a later rebuild). */
    }

    pthread_mutex_unlock(&seg->per_seg_mutex);
    return any_flipped;
}
```

- [ ] **Step 4: Add helper: apply to a memtable**

Append:

```c
/*
 * Apply a vacuum batch to a memtable on the standby. Returns true if the
 * matching memtable was found (regardless of whether any bits flipped).
 * Caller already holds lsm->mt_lock SHARED via the routing layer below.
 */
static bool
apply_to_memtable_for_sid(LSMIndexBufferSlot *slot, SegmentId sid,
                          const xl_dpv_vacuum_tombstones *hdr,
                          const xl_dpv_vacuum_entry_v2 *entries)
{
    LSMIndex lsm = &slot->lsmIndex;
    bool     found = false;

    for (int i = 0; i < MEMTABLE_NUM; i++)
    {
        int32_t            gidx = lsm->memtable_idxs[i];
        ConcurrentMemTable mt;

        if (gidx < 0) continue;
        mt = MT_FROM_SLOTIDX(gidx);
        if (mt == NULL || mt->memtable_id != sid) continue;

        LWLockAcquire(&mt->vacuum_lock, LW_EXCLUSIVE);
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
            /* Primary already flushed; standby hasn't adopted. 2-pointer
             * walk against mt->tids in insertion order. */
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
        LWLockRelease(&mt->vacuum_lock);
        found = true;
        break;
    }
    return found;
}
```

- [ ] **Step 5: Replace `dpv_apply_vacuum_tombstones_v2` with full implementation**

Append the final entry point:

```c
void
dpv_apply_vacuum_tombstones_v2(const xl_dpv_vacuum_tombstones *hdr,
                                const xl_dpv_vacuum_entry_v2 *entries)
{
    int                 slot_idx;
    LSMIndexBufferSlot *slot;
    FlushedSegmentPool *pool;
    LSMIndex            lsm;

    slot_idx = lookup_lsm_index_idx(hdr->indexRelId);
    if (slot_idx < 0)
        return;
    slot = &SharedLSMIndexBuffer->slots[slot_idx];
    lsm  = &slot->lsmIndex;

    pool = get_flushed_segment_pool(slot_idx);
    if (pool == NULL)
        return;

    /* ---- Pool first ------------------------------------------------- */
    {
        bool   segment_handled = false;
        bool   any_flipped     = false;
        uint32 found_seg_idx   = (uint32) -1;
        int    sid_off_index   = -1;

        pthread_rwlock_rdlock(&pool->seg_lock);

        for (uint32 idx = pool->head_idx;
             idx != (uint32) -1;
             idx = pool->flushed_segments[idx].next_idx)
        {
            FlushedSegment seg = &pool->flushed_segments[idx];
            if (!seg->in_used) {
                if (idx == pool->tail_idx) break;
                continue;
            }
            if (seg->segment_id_start <= hdr->sid &&
                hdr->sid <= seg->segment_id_end)
            {
                int oi = find_offset_index_for_sid(seg, hdr->sid);
                if (oi >= 0)
                {
                    any_flipped = apply_to_segment(seg, oi, hdr, entries);
                    found_seg_idx = idx;
                    sid_off_index = oi;
                    segment_handled = true;
                }
                break;     /* sids are unique across the pool */
            }
            if (idx == pool->tail_idx) break;
        }

        pthread_rwlock_unlock(&pool->seg_lock);

        if (segment_handled)
        {
            /* Persist a subversion file only when the owner-on-primary was
             * a flushed segment AND we actually flipped at least one bit. */
            if (any_flipped && hdr->is_memtable_owner == 0)
            {
                FlushedSegment seg = &pool->flushed_segments[found_seg_idx];
                pthread_rwlock_rdlock(&pool->seg_lock);
                pthread_mutex_lock(&seg->per_seg_mutex);
                {
                    Size bitmap_size = GET_BITMAP_SIZE(seg->vec_count);
                    write_bitmap_file_with_subversion(hdr->indexRelId,
                                                     seg->segment_id_start,
                                                     seg->segment_id_end,
                                                     seg->version,
                                                     hdr->subversion,
                                                     seg->bitmap_ptr,
                                                     bitmap_size,
                                                     seg->delete_count);
                }
                pthread_mutex_unlock(&seg->per_seg_mutex);
                pthread_rwlock_unlock(&pool->seg_lock);
            }
            (void) sid_off_index;   /* unused after this point */
            return;
        }
    }

    /* ---- Memtable fallback ----------------------------------------- */
    LWLockAcquire(lsm->mt_lock, LW_SHARED);
    (void) apply_to_memtable_for_sid(slot, hdr->sid, hdr, entries);
    LWLockRelease(lsm->mt_lock);

    /* If neither a segment nor a memtable holds the sid right now, the
     * tombstone is silently dropped here. The primary's bitmap subversion
     * file (or eventual segment file via the fetcher) carries the bits;
     * adoption will union them in via union_deletion_entries_into_new_seg
     * when the segment arrives. */
}
```

- [ ] **Step 6: Add required includes**

Confirm the following includes are present at the top of [`segment_vacuum_redo.c`](../../pgvector/src/segment_vacuum_redo.c); add any that are missing:

```c
#include "postgres.h"
#include "miscadmin.h"
#include "storage/lwlock.h"
#include "utils/elog.h"
#include "access/xlog.h"
#include <pthread.h>

#include "lsmindex.h"
#include "lsm_segment.h"
#include "lsmindex_io.h"        /* write_bitmap_file_with_subversion */
#include "segment_vacuum_redo.h"
```

### Task 3.2: Build verification for Phase 3

- [ ] **Step 1: Run make**

```bash
cd pgvector && make 2>&1 | tail -20
```

Expected: build succeeds.

---

## Phase 4 — Switch primary emit sites to new WAL

Each task below replaces *one* of the three vacuum sites in `bulk_delete_lsm_index`. The other sites continue to emit the legacy `xl_dpv_segment_vacuum_tombstones` (and the per-tid `MEMTABLE_TOMBSTONE`) until they're switched. Order: simplest first.

### Task 4.1: Switch site 1 — growing memtable

**Files:**
- Modify: `pgvector/src/lsmindex.c`

- [ ] **Step 1: Replace the `vacuum_entries` palloc + per-entry collection + emit block**

In [`lsmindex.c`](../../pgvector/src/lsmindex.c) inside the growing-memtable branch (currently lines ~2035 to ~2129):

**Before** (current snippet, for reference only — search for the block beginning with the comment "Spec §11: collect (local_idx, tid) entries"):

```c
            xl_dpv_vacuum_entry *vacuum_entries = NULL;
            int vacuum_entry_count = 0;
            if (valid_size > 0 && dpv_replication_role == DPV_ROLE_PRIMARY)
                vacuum_entries = (xl_dpv_vacuum_entry *) palloc(sizeof(xl_dpv_vacuum_entry) * valid_size);
            ...
                        if (vacuum_entries)
                        {
                            vacuum_entries[vacuum_entry_count].local_idx = i;
                            vacuum_entries[vacuum_entry_count].tid       = tid_int64;
                            vacuum_entry_count++;
                        }
            ...
                    if (dpv_replication_role == DPV_ROLE_PRIMARY)
                    {
                        XLogRecPtr lsn = dpv_emit_segment_vacuum_tombstones(
                            indexRelId,
                            mt->memtable_id, mt->memtable_id, version,
                            next_subversion,
                            vacuum_entries, vacuum_entry_count);
                        XLogFlush(lsn);
                    }
            ...
            if (vacuum_entries)
                pfree(vacuum_entries);
```

**After** (do these three edits inside the same branch):

Edit A — replace the buffer type:

```c
            /* Plan 3 refactor: collect entries for the unified per-sid WAL.
             * The growing memtable has exactly one sid (mt->memtable_id);
             * we emit one xl_dpv_vacuum_tombstones record per vacuum batch. */
            xl_dpv_vacuum_entry_v2 *vacuum_entries = NULL;
            int vacuum_entry_count = 0;
            if (valid_size > 0 && dpv_replication_role == DPV_ROLE_PRIMARY)
                vacuum_entries = (xl_dpv_vacuum_entry_v2 *)
                    palloc(sizeof(xl_dpv_vacuum_entry_v2) * valid_size);
```

Edit B — replace the per-entry append (inside the `if (callback(...))` branch). Find:

```c
                        if (vacuum_entries)
                        {
                            vacuum_entries[vacuum_entry_count].local_idx = i;
                            vacuum_entries[vacuum_entry_count].tid       = tid_int64;
                            vacuum_entry_count++;
                        }
```

Replace with:

```c
                        if (vacuum_entries)
                        {
                            vacuum_entries[vacuum_entry_count].sid_local_idx = i;
                            vacuum_entries[vacuum_entry_count].tid           = tid_int64;
                            vacuum_entry_count++;
                        }
```

Edit C — replace the emit call. Find the two branches inside `if (bitmap_changed)`:

```c
                if (is_persistent)
                {
                    ...
                    if (dpv_replication_role == DPV_ROLE_PRIMARY)
                    {
                        XLogRecPtr lsn = dpv_emit_segment_vacuum_tombstones(
                            indexRelId,
                            mt->memtable_id, mt->memtable_id, version,
                            next_subversion,
                            vacuum_entries, vacuum_entry_count);
                        XLogFlush(lsn);
                    }
                    ...
                }
                else {
                    write_bitmap_for_memtable(...);
                    ...
                }
```

Replace the persistent-branch `if (dpv_replication_role == DPV_ROLE_PRIMARY) { ... XLogFlush(lsn); }` block with:

```c
                    if (dpv_replication_role == DPV_ROLE_PRIMARY)
                    {
                        XLogRecPtr lsn = dpv_emit_vacuum_tombstones(
                            indexRelId,
                            mt->memtable_id,             /* sid */
                            version,                     /* owner_version */
                            next_subversion,
                            /* is_memtable_owner */ false,
                            vacuum_entries, vacuum_entry_count);
                        XLogFlush(lsn);
                    }
```

And ADD a primary-role emit in the `else { /* non-persistent */ }` branch, **immediately before** the existing `write_bitmap_for_memtable(...)` call:

```c
                    if (dpv_replication_role == DPV_ROLE_PRIMARY &&
                        vacuum_entry_count > 0)
                    {
                        /* Memtable not yet flushed: subversion N/A (use sentinel).
                         * No XLogFlush needed — memtables have no §11 ordering
                         * invariant to protect (their bitmap-for-memtable file
                         * is advisory; the standby reconstructs from WAL). */
                        (void) dpv_emit_vacuum_tombstones(
                            indexRelId,
                            mt->memtable_id,             /* sid */
                            0,                           /* owner_version */
                            UINT32_MAX,                  /* subversion sentinel */
                            /* is_memtable_owner */ true,
                            vacuum_entries, vacuum_entry_count);
                    }
                    write_bitmap_for_memtable(...);   /* keep the existing call */
```

> Do NOT delete the existing `write_bitmap_for_memtable` call; just insert the new emit immediately before it.

### Task 4.2: Switch site 2 — immutable memtables

**Files:**
- Modify: `pgvector/src/lsmindex.c`

- [ ] **Step 1: Apply the same three edits as Task 4.1 to the immutable-memtable branch**

The immutable-memtable branch is structurally identical to the growing-memtable branch (currently lines ~2240 to ~2331 in [`lsmindex.c`](../../pgvector/src/lsmindex.c#L2240-L2331)). Apply the three edits from Task 4.1 (Edit A — replace buffer type to `xl_dpv_vacuum_entry_v2`; Edit B — rename `local_idx` → `sid_local_idx` in the append; Edit C — replace the emit call with `dpv_emit_vacuum_tombstones`, including the non-persistent-branch addition).

> The two branches use the same identifiers (`mt`, `vacuum_entries`, `vacuum_entry_count`, `next_subversion`, `is_persistent`, `bitmap_changed`), so the edits are mechanically the same.

### Task 4.3: Switch site 3 — flushed segments (single-sid fast path)

**Files:**
- Modify: `pgvector/src/lsmindex.c`

- [ ] **Step 1: Replace the per-segment vacuum_entries collection + emit block**

In [`lsmindex.c`](../../pgvector/src/lsmindex.c) inside the flushed-segments branch (currently lines ~2423 to ~2516 — search for "vacuum_entries" inside this branch).

This step handles the **single-sid case only** (`start_sid_disk == end_sid_disk`). Task 4.4 adds the multi-sid path.

Replace the `xl_dpv_vacuum_entry *vacuum_entries` declaration and population block (analogous to sites 1 and 2) with:

Edit A — buffer:

```c
            xl_dpv_vacuum_entry_v2 *vacuum_entries = NULL;
            int vacuum_entry_count = 0;
            if (valid_rows > 0 && dpv_replication_role == DPV_ROLE_PRIMARY)
                vacuum_entries = (xl_dpv_vacuum_entry_v2 *)
                    palloc(sizeof(xl_dpv_vacuum_entry_v2) * valid_rows);
```

Edit B — append entry (rename field):

```c
                        if (vacuum_entries)
                        {
                            vacuum_entries[vacuum_entry_count].sid_local_idx = i;
                            vacuum_entries[vacuum_entry_count].tid           = mapping_ptr[i];
                            vacuum_entry_count++;
                        }
```

Edit C — replace the emit call. Find:

```c
                if (dpv_replication_role == DPV_ROLE_PRIMARY)
                {
                    XLogRecPtr lsn = dpv_emit_segment_vacuum_tombstones(
                        indexRelId,
                        start_sid_disk, end_sid_disk, seg_version,
                        next_sub,
                        vacuum_entries, vacuum_entry_count);
                    XLogFlush(lsn);
                }
```

Replace with:

```c
                if (dpv_replication_role == DPV_ROLE_PRIMARY &&
                    vacuum_entry_count > 0)
                {
                    if (start_sid_disk == end_sid_disk)
                    {
                        /* Single-sid segment: sid_local_idx == i; emit one record. */
                        XLogRecPtr lsn = dpv_emit_vacuum_tombstones(
                            indexRelId,
                            start_sid_disk,            /* sid */
                            seg_version,               /* owner_version */
                            next_sub,                  /* subversion */
                            /* is_memtable_owner */ false,
                            vacuum_entries, vacuum_entry_count);
                        XLogFlush(lsn);
                    }
                    else
                    {
                        /* Multi-sid segment — handled in Task 4.4. Until that
                         * task lands, fall back to a single record covering
                         * the whole range (sid = start_sid_disk) which is
                         * INCORRECT for multi-sid; this is an intentional
                         * transient state. Task 4.4 fixes it. */
                        XLogRecPtr lsn = dpv_emit_vacuum_tombstones(
                            indexRelId,
                            start_sid_disk,
                            seg_version,
                            next_sub,
                            /* is_memtable_owner */ false,
                            vacuum_entries, vacuum_entry_count);
                        XLogFlush(lsn);
                    }
                }
```

> The temporary multi-sid fallback is intentional and is replaced in Task 4.4. Don't run multi-sid regression tests between 4.3 and 4.4.

### Task 4.4: Switch site 3 — flushed segments (multi-sid path)

**Files:**
- Modify: `pgvector/src/lsmindex.c`

- [ ] **Step 1: Add multi-sid scratch buffer declarations**

In the flushed-segments branch, immediately AFTER the `xl_dpv_vacuum_entry_v2 *vacuum_entries = NULL; int vacuum_entry_count = 0;` declarations from Task 4.3 Edit A, ADD these additional locals at the same scope:

```c
            /*
             * Plan 3 refactor: for multi-sid segments, group vacuumed entries
             * by sid via the offset file. seg_offsets[k] gives (sid,
             * start_offset, end_offset) for sid index k. per_sid_entries[k]
             * and per_sid_count[k] are the buffer + size for sid k.
             */
            SegmentOffsetRange     *seg_offsets      = NULL;
            uint32_t                seg_sid_count    = 0;
            xl_dpv_vacuum_entry_v2 **per_sid_entries = NULL;
            int                    *per_sid_count    = NULL;
            uint32_t                cur_off_idx      = 0;  /* advancing pointer */

            if (vacuum_entries != NULL && start_sid_disk != end_sid_disk)
            {
                /* Multi-sid path: load offset table and allocate per-sid buffers. */
                load_offset_file(indexRelId, start_sid_disk, end_sid_disk,
                                 seg_version, &seg_offsets, true);
                seg_sid_count = (uint32_t) (end_sid_disk - start_sid_disk + 1);
                per_sid_entries = (xl_dpv_vacuum_entry_v2 **)
                    palloc0(sizeof(xl_dpv_vacuum_entry_v2 *) * seg_sid_count);
                per_sid_count = (int *) palloc0(sizeof(int) * seg_sid_count);
                for (uint32_t k = 0; k < seg_sid_count; k++)
                {
                    Size span = seg_offsets[k].end_offset - seg_offsets[k].start_offset;
                    if (span > 0)
                        per_sid_entries[k] = (xl_dpv_vacuum_entry_v2 *)
                            palloc(sizeof(xl_dpv_vacuum_entry_v2) * span);
                }
            }
```

- [ ] **Step 2: Replace per-entry append to dispatch by sid**

Inside the `for (uint32_t i = 0; i < valid_rows; i++)` loop, replace the body of `if (vacuum_entries) { ... }` (the field-rename from Task 4.3 Edit B) with sid-aware dispatch:

```c
                        if (vacuum_entries)
                        {
                            if (start_sid_disk == end_sid_disk)
                            {
                                /* Single-sid: sid_local_idx == i. */
                                vacuum_entries[vacuum_entry_count].sid_local_idx = i;
                                vacuum_entries[vacuum_entry_count].tid           = mapping_ptr[i];
                                vacuum_entry_count++;
                            }
                            else
                            {
                                /* Multi-sid: locate sid via offset table.
                                 * cur_off_idx advances monotonically with i
                                 * (declared at outer scope in Step 1), giving
                                 * O(n) total grouping. */
                                while (cur_off_idx < seg_sid_count &&
                                       i >= seg_offsets[cur_off_idx].end_offset)
                                    cur_off_idx++;
                                if (cur_off_idx >= seg_sid_count)
                                    elog(ERROR,
                                         "[bulk_delete_lsm_index] i=%u beyond segment offsets",
                                         i);

                                int k = (int) cur_off_idx;
                                int slot = per_sid_count[k]++;
                                per_sid_entries[k][slot].sid_local_idx =
                                    (uint32) (i - seg_offsets[k].start_offset);
                                per_sid_entries[k][slot].tid = mapping_ptr[i];
                            }
                        }
```

- [ ] **Step 3: Replace the emit block to emit one record per sid**

Replace the entire `if (dpv_replication_role == DPV_ROLE_PRIMARY && vacuum_entry_count > 0) { ... }` block from Task 4.3 Edit C with:

```c
                if (dpv_replication_role == DPV_ROLE_PRIMARY)
                {
                    if (start_sid_disk == end_sid_disk)
                    {
                        /* Single-sid fast path. */
                        if (vacuum_entry_count > 0)
                        {
                            XLogRecPtr lsn = dpv_emit_vacuum_tombstones(
                                indexRelId,
                                start_sid_disk,
                                seg_version,
                                next_sub,
                                /* is_memtable_owner */ false,
                                vacuum_entries, vacuum_entry_count);
                            XLogFlush(lsn);
                        }
                    }
                    else
                    {
                        /* Multi-sid: one record per touched sid. Collect each
                         * XLogInsert's LSN; flush the max once. */
                        XLogRecPtr max_lsn = InvalidXLogRecPtr;
                        for (uint32_t k = 0; k < seg_sid_count; k++)
                        {
                            if (per_sid_count[k] == 0)
                                continue;
                            XLogRecPtr lsn = dpv_emit_vacuum_tombstones(
                                indexRelId,
                                seg_offsets[k].sid,
                                seg_version,
                                next_sub,
                                /* is_memtable_owner */ false,
                                per_sid_entries[k],
                                per_sid_count[k]);
                            if (lsn > max_lsn) max_lsn = lsn;
                        }
                        if (max_lsn != InvalidXLogRecPtr)
                            XLogFlush(max_lsn);
                    }
                }
```

- [ ] **Step 4: Free the multi-sid scratch buffers**

At the bottom of the per-segment loop (after the existing `pfree(bitmap_ptr);` and the `if (vacuum_entries) pfree(vacuum_entries);`), add:

```c
            if (per_sid_entries)
            {
                for (uint32_t k = 0; k < seg_sid_count; k++)
                    if (per_sid_entries[k]) pfree(per_sid_entries[k]);
                pfree(per_sid_entries);
            }
            if (per_sid_count)  pfree(per_sid_count);
            if (seg_offsets)    pfree(seg_offsets);
```

> Also add matching `pfree(...)` calls in any early-exit / `goto retry_segment` paths within this branch — search for `pfree(bitmap_ptr)` and add the four new pfree calls alongside each occurrence.

### Task 4.5: Build verification for Phase 4

- [ ] **Step 1: Run make**

```bash
cd pgvector && make 2>&1 | tail -20
```

Expected: build succeeds.

---

## Phase 5 — Stop emitting `MEMTABLE_TOMBSTONE` from `statuspage.c`

### Task 5.1: Remove the standalone emit

**Files:**
- Modify: `pgvector/src/statuspage.c`

- [ ] **Step 1: Delete the emit line**

In [`statuspage.c:842`](../../pgvector/src/statuspage.c#L842), delete the single line:

```c
                dpv_emit_memtable_tombstone(RelationGetRelid(index), sid, slot_index);
```

> Keep everything else in `RemoveFromStatusMemtable` (the `GenericXLogFinish(state)`, the `UnlockReleaseBuffer` calls, the `return true`). Only the standalone semantic WAL emission goes away.

- [ ] **Step 2: Update the comment**

Replace the comment two lines above (`/* Emit semantic WAL record after all GenericXLog work is done. */`) with:

```c
                /* No semantic WAL emission here: the unified xl_dpv_vacuum_tombstones
                 * record emitted by bulk_delete_lsm_index carries the per-tid
                 * deletion info. GenericXLog above handles the heap status-page
                 * page-level WAL. */
```

### Task 5.2: Build verification for Phase 5

- [ ] **Step 1: Run make**

```bash
cd pgvector && make 2>&1 | tail -20
```

Expected: build succeeds. The function `dpv_emit_memtable_tombstone` is now defined-but-unused; Phase 7 deletes it.

---

## Phase 6 — Adoption rewrite

### Task 6.1: Define `DpvVacuumGroup`

**Files:**
- Modify: `pgvector/src/segment_adoption.h`

- [ ] **Step 1: Add type**

Open [`segment_adoption.h`](../../pgvector/src/segment_adoption.h). Above the `dpv_pool_adopt` declaration, insert:

```c
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
```

- [ ] **Step 2: Update `dpv_pool_adopt` declaration**

Find the existing declaration `extern DpvAdoptionOutcome dpv_pool_adopt(...)` in [`segment_adoption.h`](../../pgvector/src/segment_adoption.h). Replace the trailing two parameters:

```c
/* Before: */
extern DpvAdoptionOutcome dpv_pool_adopt(int lsm_idx, Oid indexRelId,
                                          SegmentId start_sid, SegmentId end_sid, uint32 version,
                                          const SegmentId *memtable_cover, int memtable_cover_count,
                                          const int64_t *deletion_tids, int n_deletion_tids);

/* After: */
extern DpvAdoptionOutcome dpv_pool_adopt(int lsm_idx, Oid indexRelId,
                                          SegmentId start_sid, SegmentId end_sid, uint32 version,
                                          const SegmentId *memtable_cover, int memtable_cover_count,
                                          const DpvVacuumGroup *groups, int n_groups);
```

### Task 6.2: Replace `union_segment_bitmap_via_tid` with offset-merge

**Files:**
- Modify: `pgvector/src/segment_adoption.c`

- [ ] **Step 1: Delete the existing function**

In [`segment_adoption.c:96-134`](../../pgvector/src/segment_adoption.c#L96-L134), delete the entire `union_segment_bitmap_via_tid` function (the whole 39-line block).

- [ ] **Step 2: Add the new function**

In the same place, insert:

```c
/*
 * Union the OLD segment's deleted bits into NEW's bitmap via a 2-pointer
 * merge over the shared (sid, offset)-order invariant. For every sid that
 * appears in both segments' offset tables, walk both map_ptr slices in
 * lockstep: when tids match, propagate the bit; when old has a tid not in
 * new (compaction), skip past it. O(sum of old slice lengths) total.
 *
 * Caller holds pool->seg_lock in WRITE mode.
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
```

### Task 6.3: Replace `union_deletion_tids_into_new_seg` with offset-merge groups

**Files:**
- Modify: `pgvector/src/segment_adoption.c`

- [ ] **Step 1: Delete existing function**

Delete the entire `union_deletion_tids_into_new_seg` function ([`segment_adoption.c:155-183`](../../pgvector/src/segment_adoption.c#L155-L183)).

- [ ] **Step 2: Add new function**

Insert in its place:

```c
/*
 * Plan 3 Part B (refactored). For each per-sid group of deletion tids from
 * the fetcher's cover scan, walk new_seg's map_ptr[offsets[sid] slice) in
 * lockstep with group->tids[]. The fetcher captured tids[] in mt->tids
 * insertion order, which equals the order they appear in new_seg's
 * map_ptr for that sid (the segment was just flushed from this memtable
 * — or merged from segments that preserve relative order per sid).
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
```

### Task 6.4: Update `dpv_pool_adopt` signature and call sites

**Files:**
- Modify: `pgvector/src/segment_adoption.c`

- [ ] **Step 1: Update function signature**

In [`segment_adoption.c:214-217`](../../pgvector/src/segment_adoption.c#L214-L217):

```c
/* Before */
DpvAdoptionOutcome
dpv_pool_adopt(int lsm_idx, Oid indexRelId,
               SegmentId start_sid, SegmentId end_sid, uint32 version,
               const SegmentId *memtable_cover, int memtable_cover_count,
               const int64_t *deletion_tids, int n_deletion_tids)

/* After */
DpvAdoptionOutcome
dpv_pool_adopt(int lsm_idx, Oid indexRelId,
               SegmentId start_sid, SegmentId end_sid, uint32 version,
               const SegmentId *memtable_cover, int memtable_cover_count,
               const DpvVacuumGroup *groups, int n_groups)
```

- [ ] **Step 2: Update Case A call site**

In [`segment_adoption.c:353`](../../pgvector/src/segment_adoption.c#L353), replace:

```c
                union_segment_bitmap_via_tid(new_seg, existing);
```

with:

```c
                union_segment_bitmap_via_offset_merge(new_seg, existing);
```

- [ ] **Step 3: Update Case D call site**

In [`segment_adoption.c:382-383`](../../pgvector/src/segment_adoption.c#L382-L383), replace:

```c
        union_deletion_tids_into_new_seg(&pool->flushed_segments[new_slot],
                                          deletion_tids, n_deletion_tids);
```

with:

```c
        union_deletion_entries_into_new_seg(&pool->flushed_segments[new_slot],
                                             groups, n_groups);
```

- [ ] **Step 4: Update Case E call sites**

In [`segment_adoption.c:464`](../../pgvector/src/segment_adoption.c#L464), replace:

```c
                    union_segment_bitmap_via_tid(new_seg, old_seg);
```

with:

```c
                    union_segment_bitmap_via_offset_merge(new_seg, old_seg);
```

In [`segment_adoption.c:471-472`](../../pgvector/src/segment_adoption.c#L471-L472), replace:

```c
            union_deletion_tids_into_new_seg(new_seg,
                                              deletion_tids, n_deletion_tids);
```

with:

```c
            union_deletion_entries_into_new_seg(new_seg, groups, n_groups);
```

- [ ] **Step 5: Remove `#include "sorted_perm.h"`**

In [`segment_adoption.c:58`](../../pgvector/src/segment_adoption.c#L58), delete the line:

```c
#include "sorted_perm.h"
```

### Task 6.5: Update `tasksend.h` / `tasksend.c` signatures

**Files:**
- Modify: `pgvector/src/tasksend.h`
- Modify: `pgvector/src/tasksend.c`

- [ ] **Step 1: Update `tasksend.h` declarations**

In [`tasksend.h`](../../pgvector/src/tasksend.h), replace the existing declarations of `segment_update_blocking` and `dpv_send_adopt_task`. Find their trailing `const int64_t *deletion_tids, int n_deletion_tids` arguments and replace them with `const DpvVacuumGroup *groups, int n_groups`.

Add this include at the top of [`tasksend.h`](../../pgvector/src/tasksend.h) if not already present:

```c
#include "segment_adoption.h"   /* DpvVacuumGroup */
```

- [ ] **Step 2: Update `segment_update_blocking` parameters and DSM packing**

In [`tasksend.c:172-178`](../../pgvector/src/tasksend.c#L172-L178), replace the signature trailing arguments:

```c
/* Before */
                         const int64_t *deletion_tids,
                         int n_deletion_tids)
/* After */
                         const DpvVacuumGroup *groups,
                         int n_groups)
```

Replace the DSM payload sizing + packing block (lines 195-241 approximately):

```c
    /*
     * Plan 3 refactor: the DSM trailer carries per-sid groups instead of a
     * flat tid array. Layout (starting at byte offset
     * sizeof(SegmentUpdateTaskData)):
     *
     *   uint32 n_groups
     *   For each group:
     *     uint32     sid
     *     uint32     n_tids
     *     int64_t    tids[n_tids]
     *
     * Sizes use natural alignment; the int64_t arrays land on 8-aligned
     * offsets because the SegmentUpdateTaskData header is 8-aligned and
     * we keep (sid, n_tids) as two uint32 (8 bytes) before each tids[].
     */
    Size payload_size = sizeof(uint32);     /* n_groups field */
    if (groups == NULL || n_groups <= 0)
    {
        n_groups = 0;
    }
    else
    {
        for (int g = 0; g < n_groups; g++)
            payload_size += sizeof(uint32) * 2 +
                            (Size) groups[g].n_tids * sizeof(int64_t);
    }

    task_seg_size = task_struct_size + payload_size;

    task_seg = dsm_create(task_seg_size, 0);
    if (task_seg == NULL)
        elog(ERROR, "[segment_update_blocking] Failed to allocate DSM segment");

    task = (SegmentUpdateTask) dsm_segment_address(task_seg);
    task->backend_pgprocno = MyProcNumber;
    task->index_relid = index_relid;
    task->lsm_idx = lsm_index_idx;
    task->operation_type = operation_type;
    task->start_sid = start_sid;
    task->end_sid = end_sid;
    task->expected_version = expected_version;

    /* Copy memtable cover for ADOPT tasks; zero it out for all other ops. */
    if (memtable_cover != NULL && memtable_cover_count > 0)
    {
        cover_count = memtable_cover_count;
        if (cover_count > MEMTABLE_NUM + 1)
            cover_count = MEMTABLE_NUM + 1;
        task->memtable_cover_count = (uint8) cover_count;
        for (int i = 0; i < cover_count; i++)
            task->memtable_cover[i] = memtable_cover[i];
    }
    else
    {
        task->memtable_cover_count = 0;
    }

    /* Pack the per-sid group payload. */
    {
        char *p = (char *) dsm_segment_address(task_seg) + task_struct_size;
        uint32 hdr_ngroups = (uint32) n_groups;
        memcpy(p, &hdr_ngroups, sizeof(uint32));
        p += sizeof(uint32);
        for (int g = 0; g < n_groups; g++)
        {
            uint32 sid_u32 = (uint32) groups[g].sid;
            uint32 n       = groups[g].n_tids;
            memcpy(p, &sid_u32, sizeof(uint32));  p += sizeof(uint32);
            memcpy(p, &n,       sizeof(uint32));  p += sizeof(uint32);
            if (n > 0)
                memcpy(p, groups[g].tids, (Size) n * sizeof(int64_t));
            p += (Size) n * sizeof(int64_t);
        }
    }

    /* Carry the *byte length* of the group payload so the consumer doesn't
     * recompute it. Repurpose the existing deletion_tids_count field by
     * reading it as a byte count (consumer is updated to match). */
    task->deletion_tids_count = (uint32) payload_size;
```

Delete the old per-tid trailer block (the `int64_t *trailer = ... ; memcpy(trailer, deletion_tids, ...);` lines, around [`tasksend.c:234-241`](../../pgvector/src/tasksend.c#L234-L241)).

- [ ] **Step 3: Update `dpv_send_adopt_task` signature and body**

In [`tasksend.c:256-269`](../../pgvector/src/tasksend.c#L256-L269), replace:

```c
int
dpv_send_adopt_task(int lsm_idx, Oid index_relid,
                    SegmentId start_sid, SegmentId end_sid,
                    uint32_t version,
                    const SegmentId *memtable_cover,
                    int memtable_cover_count,
                    const DpvVacuumGroup *groups,
                    int n_groups)
{
    return segment_update_blocking(lsm_idx, index_relid, SEGMENT_UPDATE_ADOPT,
                                    start_sid, end_sid, version,
                                    memtable_cover, memtable_cover_count,
                                    groups, n_groups);
}
```

### Task 6.6: Update `vector_index_worker.c` DSM trailer reader and call site

**Files:**
- Modify: `pgvector/src/vector_index_worker.c`

- [ ] **Step 1: Replace task fields**

Find the `task->deletion_tids` / `task->deletion_tids_count` field declarations (around [`vector_index_worker.c:145-146`](../../pgvector/src/vector_index_worker.c#L145-L146)). Replace with:

```c
    /* Plan 3 refactor: per-sid grouped deletion payload from the fetcher. */
    DpvVacuumGroup *groups;
    int             n_groups;
    int64_t        *groups_tids_buf;   /* backing storage for all group tids */
```

- [ ] **Step 2: Replace zero-initializers**

There are two places that NULL these out (around [`vector_index_worker.c:1730-1731`](../../pgvector/src/vector_index_worker.c#L1730-L1731) and [`vector_index_worker.c:1781-1782`](../../pgvector/src/vector_index_worker.c#L1781-L1782)). Replace each two-line block with:

```c
    task->groups            = NULL;
    task->n_groups          = 0;
    task->groups_tids_buf   = NULL;
```

- [ ] **Step 3: Replace the DSM trailer reader**

Find the block that copies the old `deletion_tids` trailer (around [`vector_index_worker.c:1806-1823`](../../pgvector/src/vector_index_worker.c#L1806-L1823)). Replace the entire `if (src_task->deletion_tids_count > 0) { ... }` block with:

```c
            if (src_task->operation_type == SEGMENT_UPDATE_ADOPT &&
                src_task->deletion_tids_count > 0)
            {
                /* deletion_tids_count is now repurposed as the trailer byte
                 * length (Plan 3 refactor). Parse the per-sid group layout:
                 *   uint32 n_groups; (sid, n_tids, tids[n_tids])* */
                Size  payload_bytes = (Size) src_task->deletion_tids_count;
                char *src = (char *) src_task + sizeof(SegmentUpdateTaskData);
                char *src_end = src + payload_bytes;

                uint32 ng;
                if (payload_bytes < sizeof(uint32))
                    elog(ERROR, "[submit_maintenance_task] ADOPT trailer too short");
                memcpy(&ng, src, sizeof(uint32));
                src += sizeof(uint32);

                if (ng > 0)
                {
                    /* Allocate the groups[] array and a single flat tids buffer
                     * to back all groups' tids[] pointers. */
                    task->groups = (DpvVacuumGroup *) malloc(ng * sizeof(DpvVacuumGroup));
                    if (task->groups == NULL)
                        elog(ERROR, "[submit_maintenance_task] malloc(groups) failed");

                    /* Two-pass: pass 1 counts total tids; pass 2 fills. */
                    char *scan = src;
                    Size total_tids = 0;
                    for (uint32 g = 0; g < ng; g++)
                    {
                        if (scan + 2 * sizeof(uint32) > src_end)
                            elog(ERROR, "[submit_maintenance_task] truncated group header");
                        uint32 sid_u32, n;
                        memcpy(&sid_u32, scan, sizeof(uint32)); scan += sizeof(uint32);
                        memcpy(&n,       scan, sizeof(uint32)); scan += sizeof(uint32);
                        total_tids += n;
                        if (scan + (Size) n * sizeof(int64_t) > src_end)
                            elog(ERROR, "[submit_maintenance_task] truncated group tids");
                        scan += (Size) n * sizeof(int64_t);
                    }

                    task->groups_tids_buf = (int64_t *) malloc(
                        total_tids ? total_tids * sizeof(int64_t) : 1);
                    if (task->groups_tids_buf == NULL)
                        elog(ERROR, "[submit_maintenance_task] malloc(tids_buf) failed");

                    scan = src;
                    int64_t *next_tid = task->groups_tids_buf;
                    for (uint32 g = 0; g < ng; g++)
                    {
                        uint32 sid_u32, n;
                        memcpy(&sid_u32, scan, sizeof(uint32)); scan += sizeof(uint32);
                        memcpy(&n,       scan, sizeof(uint32)); scan += sizeof(uint32);
                        task->groups[g].sid    = (SegmentId) sid_u32;
                        task->groups[g].n_tids = n;
                        task->groups[g].tids   = (n > 0) ? next_tid : NULL;
                        if (n > 0)
                        {
                            memcpy(next_tid, scan, (Size) n * sizeof(int64_t));
                            next_tid += n;
                            scan     += (Size) n * sizeof(int64_t);
                        }
                    }
                    task->n_groups = (int) ng;
                }
            }
```

- [ ] **Step 4: Replace the free path**

Find the `task->deletion_tids != NULL` free block (around [`vector_index_worker.c:658-661`](../../pgvector/src/vector_index_worker.c#L658-L661)). Replace with:

```c
            if (task->groups != NULL)
            {
                free(task->groups);
                task->groups   = NULL;
                task->n_groups = 0;
            }
            if (task->groups_tids_buf != NULL)
            {
                free(task->groups_tids_buf);
                task->groups_tids_buf = NULL;
            }
```

- [ ] **Step 5: Update the `dpv_pool_adopt` call site**

Find the existing call (around [`vector_index_worker.c:437-438`](../../pgvector/src/vector_index_worker.c#L437-L438)). Replace the trailing two arguments:

```c
/* Before */
                                                     task->deletion_tids,
                                                     (int) task->deletion_tids_count);
/* After */
                                                     task->groups,
                                                     task->n_groups);
```

> Verify by reading the lines immediately above the change: the call should be `dpv_pool_adopt(lsm_idx, indexRelId, start_sid, end_sid, version, memtable_cover, memtable_cover_count, task->groups, task->n_groups);`.

### Task 6.7: Update `segment_fetcher.c` to build per-sid groups

**Files:**
- Modify: `pgvector/src/segment_fetcher.c`

- [ ] **Step 1: Replace the flat-tid collection with per-group collection**

In [`segment_fetcher.c:750-779`](../../pgvector/src/segment_fetcher.c#L750-L779) (the block beginning with `int64_t *deletion_tids = NULL;`), replace the entire block — from `int64_t *deletion_tids = NULL;` through the closing brace of the per-`mci` loop that appends tids — with:

```c
                /*
                 * Plan 3 refactor: collect deletion bits as per-sid groups.
                 * One group per cover memtable. Each group's tids[] is a
                 * subarray of mt->tids in insertion order — directly
                 * referenced (no copy) since vacuum_lock SHARED is held on
                 * each cover memtable until after dpv_send_adopt_task
                 * returns (so mt->tids[] cannot mutate).
                 */
                DpvVacuumGroup *groups = (memtable_cover_count > 0)
                    ? (DpvVacuumGroup *) palloc(sizeof(DpvVacuumGroup) * memtable_cover_count)
                    : NULL;
                int64_t **per_group_tids =
                    (memtable_cover_count > 0)
                    ? (int64_t **) palloc(sizeof(int64_t *) * memtable_cover_count)
                    : NULL;
                int n_groups = 0;

                for (int mci = 0; mci < memtable_cover_count; mci++)
                {
                    ConcurrentMemTable mt = cover_mts[mci];
                    uint32 cur_size = pg_atomic_read_u32(&mt->current_size);
                    uint32 valid_size = (cur_size > mt->capacity)
                                        ? mt->capacity : cur_size;

                    /* First pass: count set bits to size the per-group buffer. */
                    int n_set = 0;
                    for (uint32 j = 0; j < valid_size; j++)
                        if (IS_SLOT_SET(mt->bitmap, j)) n_set++;

                    if (n_set == 0)
                        continue;

                    int64_t *g_tids = (int64_t *) palloc(sizeof(int64_t) * n_set);
                    int      g_n    = 0;
                    /* Second pass: copy tids in insertion order. */
                    for (uint32 j = 0; j < valid_size; j++)
                        if (IS_SLOT_SET(mt->bitmap, j))
                            g_tids[g_n++] = mt->tids[j];

                    groups[n_groups].sid    = mt->memtable_id;
                    groups[n_groups].n_tids = (uint32) g_n;
                    groups[n_groups].tids   = g_tids;
                    per_group_tids[n_groups] = g_tids;
                    n_groups++;
                }
```

- [ ] **Step 2: Update the `dpv_send_adopt_task` call**

Locate the existing call (around [`segment_fetcher.c:787-795`](../../pgvector/src/segment_fetcher.c#L787-L795)). Replace the trailing two arguments:

```c
/* Before */
                    adopt_result = dpv_send_adopt_task(lsm_idx,
                                                       e->hdr.indexRelId,
                                                       e->hdr.start_sid,
                                                       e->hdr.end_sid,
                                                       e->hdr.version,
                                                       memtable_cover,
                                                       memtable_cover_count,
                                                       deletion_tids,
                                                       n_deletion_tids);
/* After */
                    adopt_result = dpv_send_adopt_task(lsm_idx,
                                                       e->hdr.indexRelId,
                                                       e->hdr.start_sid,
                                                       e->hdr.end_sid,
                                                       e->hdr.version,
                                                       memtable_cover,
                                                       memtable_cover_count,
                                                       groups,
                                                       n_groups);
```

- [ ] **Step 3: Update the free path**

Replace the existing free block (around [`segment_fetcher.c:811-812`](../../pgvector/src/segment_fetcher.c#L811-L812)):

```c
/* Before */
                if (deletion_tids != NULL)
                    pfree(deletion_tids);
/* After */
                if (groups != NULL)
                {
                    for (int g = 0; g < n_groups; g++)
                        if (per_group_tids[g] != NULL) pfree(per_group_tids[g]);
                    pfree(groups);
                }
                if (per_group_tids != NULL)
                    pfree(per_group_tids);
```

### Task 6.8: Build verification for Phase 6

- [ ] **Step 1: Run make**

```bash
cd pgvector && make 2>&1 | tail -40
```

Expected: build succeeds. The only remaining users of `sorted_perm.{h,c}` and the legacy WAL types are dead-code paths cleaned up in Phase 7.

---

## Phase 7 — Delete `sorted_perm` and legacy WAL types, promote op code

### Task 7.1: Remove `sorted_idx` field and free calls

**Files:**
- Modify: `pgvector/src/lsm_segment.h`
- Modify: `pgvector/src/lsm_segment.c`

- [ ] **Step 1: Remove field**

In [`lsm_segment.h`](../../pgvector/src/lsm_segment.h), delete the `uint32_t *sorted_idx;` field and its preceding comment block (the 4-line comment that begins with "Lazy sorted permutation over map_ptr").

- [ ] **Step 2: Remove `dpv_sorted_perm_free` calls**

In [`lsm_segment.c:187`](../../pgvector/src/lsm_segment.c#L187), delete:

```c
    dpv_sorted_perm_free(segment);
```

In [`lsm_segment.c:249`](../../pgvector/src/lsm_segment.c#L249), delete:

```c
    dpv_sorted_perm_free(segment);
```

- [ ] **Step 3: Remove `#include "sorted_perm.h"`**

In [`lsm_segment.c:2`](../../pgvector/src/lsm_segment.c#L2), delete:

```c
#include "sorted_perm.h"
```

### Task 7.2: Delete `sorted_perm.{h,c}` and remove from Makefile

**Files:**
- Delete: `pgvector/src/sorted_perm.h`
- Delete: `pgvector/src/sorted_perm.c`
- Modify: `pgvector/Makefile`

- [ ] **Step 1: Delete the source files**

```bash
rm pgvector/src/sorted_perm.h pgvector/src/sorted_perm.c
```

- [ ] **Step 2: Remove from Makefile OBJS**

Open [`pgvector/Makefile`](../../pgvector/Makefile). Find the line containing `src/sorted_perm.o` in the `OBJS` variable. Delete the token (and its trailing `\` or surrounding whitespace as appropriate).

### Task 7.3: Remove `#include "sorted_perm.h"` and unused stubs from `segment_vacuum_redo.c`

**Files:**
- Modify: `pgvector/src/segment_vacuum_redo.c`

- [ ] **Step 1: Delete the include**

In [`segment_vacuum_redo.c`](../../pgvector/src/segment_vacuum_redo.c) (line 10 in the original file), delete:

```c
#include "sorted_perm.h"
```

- [ ] **Step 2: Delete the legacy apply implementation**

Delete the entire `dpv_apply_vacuum_tombstones` function (the multi-path routing helper) and ALL its static helpers (`try_fast_or_rebuild_race`, `try_merge_race`, `apply_to_memtable_by_sid`, `apply_to_any_memtable`, `apply_entries_and_persist`) and forward declarations at the top of the file. The new file should contain only the helpers and entry point added in Task 3.1 plus the includes.

### Task 7.4: Delete legacy WAL types from `replication_rmgr.h`

**Files:**
- Modify: `pgvector/src/replication_rmgr.h`

- [ ] **Step 1: Delete legacy op codes**

Delete the line:

```c
#define XLOG_DPV_MEMTABLE_TOMBSTONE    0x20
```

Delete the line:

```c
#define XLOG_DPV_SEGMENT_VACUUM_TOMBSTONES  0x80
```

- [ ] **Step 2: Promote new op code to 0x80**

Change:

```c
#define XLOG_DPV_VACUUM_TOMBSTONES         0x90
```

to:

```c
#define XLOG_DPV_VACUUM_TOMBSTONES         0x80
```

Also delete the explanatory comment block above it that says "this lives under op code 0x90 so it coexists ... Phase 7 retires the legacy records and moves this to 0x80 permanently." Replace with:

```c
/*
 * Unified per-sid vacuum tombstone record. Each record covers one sid's
 * vacuum batch; a multi-sid segment vacuum on the primary emits one
 * record per touched sid (all flushed together before the bitmap file).
 */
```

- [ ] **Step 3: Delete legacy struct types**

Delete the `xl_dpv_memtable_tombstone` struct (the 8-line block).

Delete the old `xl_dpv_vacuum_entry` struct (the 4-line block with `uint32 local_idx; int64_t tid;`).

Delete the old `xl_dpv_segment_vacuum_tombstones` struct (the entire block including its `StaticAssertDecl`s).

- [ ] **Step 4: Rename `xl_dpv_vacuum_entry_v2` → `xl_dpv_vacuum_entry`**

Use a single replace-all across the file: `xl_dpv_vacuum_entry_v2` → `xl_dpv_vacuum_entry`.

- [ ] **Step 5: Delete legacy emit prototypes**

Delete the prototype lines:

```c
extern XLogRecPtr dpv_emit_memtable_tombstone(Oid indexRelId, SegmentId sid,
                                              uint32 slot_index);
```

```c
extern XLogRecPtr dpv_emit_segment_vacuum_tombstones(
    Oid indexRelId,
    SegmentId owner_start_sid, SegmentId owner_end_sid, uint32 owner_version,
    uint32 subversion,
    const xl_dpv_vacuum_entry *entries, int entry_count);
```

> The first one used the OLD `xl_dpv_vacuum_entry` typedef name — confirm that after the Step 4 rename, this prototype refers to the new struct, which is fine because we're deleting it.

### Task 7.5: Delete legacy code from `replication_rmgr.c`

**Files:**
- Modify: `pgvector/src/replication_rmgr.c`

- [ ] **Step 1: Delete legacy dispatcher cases**

In `dpv_replication_redo`, delete the two cases:

```c
        case XLOG_DPV_MEMTABLE_TOMBSTONE: redo_memtable_tombstone(record); break;
```

```c
        case XLOG_DPV_SEGMENT_VACUUM_TOMBSTONES: redo_segment_vacuum_tombstones(record); break;
```

- [ ] **Step 2: Delete legacy identify entries**

In `dpv_replication_identify`, delete the two cases:

```c
        case XLOG_DPV_MEMTABLE_TOMBSTONE: return "MEMTABLE_TOMBSTONE";
```

```c
        case XLOG_DPV_SEGMENT_VACUUM_TOMBSTONES: return "SEGMENT_VACUUM_TOMBSTONES";
```

- [ ] **Step 3: Delete legacy forward declarations and callback bodies**

Delete the forward decls:

```c
static void redo_memtable_tombstone(XLogReaderState *r);
static void redo_segment_vacuum_tombstones(XLogReaderState *r);
```

Delete the entire `redo_memtable_tombstone` function body (around lines 138-157 of the current file).

Delete the entire `redo_segment_vacuum_tombstones` function body (around lines 353-374 of the current file).

- [ ] **Step 4: Delete legacy emit helpers**

Delete the entire `dpv_emit_memtable_tombstone` function (around lines 416-429).

Delete the entire `dpv_emit_segment_vacuum_tombstones` function (around lines 522-548).

- [ ] **Step 5: Rename `xl_dpv_vacuum_entry_v2` → `xl_dpv_vacuum_entry` and `dpv_apply_vacuum_tombstones_v2` → `dpv_apply_vacuum_tombstones`**

Two single-file replace-all operations:

- `xl_dpv_vacuum_entry_v2` → `xl_dpv_vacuum_entry`
- `dpv_apply_vacuum_tombstones_v2` → `dpv_apply_vacuum_tombstones`

### Task 7.6: Delete `dpv_standby_memtable_tombstone`

**Files:**
- Modify: `pgvector/src/standby_memtable.h`
- Modify: `pgvector/src/standby_memtable.c`

- [ ] **Step 1: Delete prototype**

In [`standby_memtable.h:25`](../../pgvector/src/standby_memtable.h#L25), delete the line:

```c
extern void dpv_standby_memtable_tombstone(Oid indexRelId, SegmentId sid,
                                            uint32 slot_index);
```

(The exact line may span two lines depending on formatting — delete the full declaration including any trailing argument lines.)

- [ ] **Step 2: Delete function body**

In [`standby_memtable.c`](../../pgvector/src/standby_memtable.c) (around line 218), delete the entire `dpv_standby_memtable_tombstone` function (including its leading comment block, around lines 213-250).

### Task 7.7: Rename apply function in `segment_vacuum_redo.h`

**Files:**
- Modify: `pgvector/src/segment_vacuum_redo.h`

- [ ] **Step 1: Delete the legacy declaration**

Delete:

```c
extern void dpv_apply_vacuum_tombstones(
    const xl_dpv_segment_vacuum_tombstones *hdr,
    const xl_dpv_vacuum_entry *entries);
```

- [ ] **Step 2: Rename `_v2`**

Replace the existing `dpv_apply_vacuum_tombstones_v2` declaration with:

```c
extern void dpv_apply_vacuum_tombstones(
    const xl_dpv_vacuum_tombstones *hdr,
    const xl_dpv_vacuum_entry *entries);
```

### Task 7.8: Final rename pass in `segment_vacuum_redo.c`

**Files:**
- Modify: `pgvector/src/segment_vacuum_redo.c`

- [ ] **Step 1: Apply replace-all in segment_vacuum_redo.c**

In [`segment_vacuum_redo.c`](../../pgvector/src/segment_vacuum_redo.c):

- `xl_dpv_vacuum_entry_v2` → `xl_dpv_vacuum_entry`
- `dpv_apply_vacuum_tombstones_v2` → `dpv_apply_vacuum_tombstones`

### Task 7.9: Build verification for Phase 7

- [ ] **Step 1: Run make**

```bash
cd pgvector && make 2>&1 | tail -40
```

Expected: build succeeds with no errors and no warnings about undefined references or unused functions related to the refactor.

- [ ] **Step 2: Confirm no lingering references**

```bash
grep -rn "sorted_perm\|dpv_sorted_perm\|xl_dpv_memtable_tombstone\|dpv_emit_memtable_tombstone\|dpv_standby_memtable_tombstone\|dpv_emit_segment_vacuum_tombstones\|xl_dpv_segment_vacuum_tombstones\|XLOG_DPV_MEMTABLE_TOMBSTONE\|XLOG_DPV_SEGMENT_VACUUM_TOMBSTONES\|_v2" pgvector/src/ | grep -v "\.o:" | grep -v "\.cache/"
```

Expected: no output. Any matches are leftover references that must be cleaned up before regression testing.

---

## Phase 8 — Regression tests

The existing Plan 3 test suite covers all four redo paths (fast / cross-version / multi-segment / memtable target) which collapse into the new routing logic. The Plan 1/2 tests cover the upstream layers.

### Task 8.1: Install built extension

- [ ] **Step 1: Run install**

```bash
cd pgvector && make install 2>&1 | tail -10
```

Expected: install succeeds. (Permissions on `pg_config`'s install destination may require the maintainer's local layout; the install script `./install_pgvector.sh` is the fallback.)

### Task 8.2: Run upstream regression tests (Plans 1 + 2)

- [ ] **Step 1: Run Plan 1/2 tests**

```bash
cd pgvector && prove -I test/perl test/t/110_replication_segment_flush.pl test/t/111_replication_segment_merge.pl test/t/112_replication_queue_restart.pl 2>&1 | tail -30
```

Expected: all three tests pass. Any regression here means the refactor accidentally broke an upstream invariant — bisect by phase.

### Task 8.3: Run Plan 3 regression suite

- [ ] **Step 1: Run vacuum tests**

```bash
cd pgvector && prove -I test/perl test/t/120_replication_vacuum_simple.pl 2>&1 | tail -20
```

Expected: pass — covers the segment-target fast path.

```bash
cd pgvector && prove -I test/perl test/t/121_replication_vacuum_merge_race.pl 2>&1 | tail -20
```

Expected: pass — covers the concurrent vacuum + merge case (the §11 motivating scenario; verifies the multi-sid emit path).

```bash
cd pgvector && prove -I test/perl test/t/122_replication_rebuild_translation.pl 2>&1 | tail -20
```

Expected: pass — covers rebuild-with-compaction (verifies the cross-version 2-pointer slow path on the standby).

```bash
cd pgvector && prove -I test/perl test/t/123_replication_attach_barrier.pl 2>&1 | tail -20
```

Expected: pass — covers the load-time barrier from Plan 2 §10 (not the rejected per-query barrier).

### Task 8.4: Final smoke run

- [ ] **Step 1: Run all replication tests together**

```bash
cd pgvector && prove -I test/perl test/t/11*_replication_*.pl test/t/12*_replication_*.pl 2>&1 | tail -40
```

Expected: all tests pass.

---

## Self-review checklist

- [ ] All four refactor requirements have implementation tasks:
  - Unify the two tombstone WAL types — Phase 1 (define new) + Phase 5 (stop legacy emit) + Phase 7 (delete legacy).
  - Include sid in vacuum WAL records (with offset-file consult on primary) — Phase 1 (schema) + Phase 4.4 (offset-file consult at emit).
  - Per-sid granularity (Option B) — Phase 1 (schema) + Phase 4 (per-sid emission).
  - Per-sid boundary signaling — intrinsic to Option B (each record is one sid's boundary).
- [ ] Downstream adjustments:
  - Ordered tids within each sid — Phase 4 (emit walks `mapping_ptr` in order).
  - Single advancing pointer on standby — Phase 3 (2-pointer merge).
  - Simplified vacuum adoption logic — Phase 6 (offset-merge functions, no sorted_perm).
  - `sorted_perm` removal — Phase 7.
- [ ] §11 ordering invariant preserved at all three emit sites — Phase 4.1, 4.2, 4.3, 4.4 each emit + XLogFlush before `write_bitmap_file_with_subversion`.
- [ ] DSM payload schema change reflected end-to-end: producer (Phase 6.7), packer (Phase 6.5), consumer (Phase 6.6), adoption-side (Phase 6.4).
- [ ] No `git commit` steps included (per repo policy).
- [ ] All file paths absolute or repo-relative; no placeholders.

---

## Open follow-ons (out of this refactor)

These are valid future work that the spec explicitly excluded:

1. **Batched per-record fsync.** If profiling shows fsync rate is a bottleneck under heavy multi-sid vacuum, multiple sids could be packed into one record with a sub-header. Today each sid is its own record; one XLogFlush per batch already amortizes the cost.
2. **Promotion / failover.** Out of scope per spec §9.
3. **Multi-standby fan-out.** Out of scope per spec §17 of the original design.

---

## End of Plan
