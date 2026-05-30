# Unified per-sid Vacuum WAL — refactor of Plan 3

**Status:** design approved 2026-05-20.
**Scope:** refactor of the already-merged Plan 3 (`docs/superpowers/plans/2026-05-15-physical-replication-plan-3-vacuum-and-barrier.md`). Does not re-design upstream replication phases (Plans 1, 2) or downstream phases not yet started.
**Non-goals:** per-query standby barrier (rejected); changes to the on-disk format of `map_ptr` or the offset file; new GUCs.

---

## 1. Motivation

Plan 3 shipped with two overlapping WAL record types for vacuumed tuples in a primary memtable:

- `XLOG_DPV_MEMTABLE_TOMBSTONE` (`0x20`) — one record per vacuumed tid, emitted from `RemoveFromStatusMemtable` in [statuspage.c:842](../../pgvector/src/statuspage.c#L842).
- `XLOG_DPV_SEGMENT_VACUUM_TOMBSTONES` (`0x80`) — one batched record per vacuum site in [bulk_delete_lsm_index](../../pgvector/src/lsmindex.c#L1968), covering the same tids.

This is a 2× WAL emission for the same logical event. The batched record also identifies the owner by `(owner_start_sid, owner_end_sid, owner_version)` — a *range*. For a multi-sid (merged) segment, the standby has to consult an in-memory `sorted_perm` aux structure to translate every WAL `tid` into a `local_idx` in whichever segment it now holds. That aux structure is built lazily under a per-segment mutex (qsort of up to N elements; allocated on first touch and kept for the segment's lifetime).

Two observations make a tighter design possible:

1. **`map_ptr` order is invariant per sid.** A flushed segment's `map_ptr[start_offset..end_offset)` for any sid is the original memtable's `tids[]` for that sid. Merges concatenate sid ranges in order. `REBUILD_DELETION` removes entries but never reorders. So across all versions/subversions of segments that contain a given sid, the surviving tids appear in the same relative order.

2. **Per-sid is the natural unit of concurrency.** Vacuum on the primary touches one or more sids per segment-vacuum batch. Adoption on the standby installs whole segments. Both producer and consumer reason naturally about sids. Range-keyed routing forces the standby into an O(N) tid lookup it shouldn't need.

The refactor unifies the two WAL types into one *per-sid* record, makes the standby apply it with a single advancing pointer over `map_ptr` within the sid's range, and drops the `sorted_perm` aux entirely.

---

## 2. The new WAL record

### 2.1 Header

```c
typedef struct {
    uint32  sid_local_idx;   /* local index within the sid's range:
                              * for a segment owner, this is
                              * (segment_local_idx − offsets[sid].start_offset);
                              * for a memtable owner, this is the memtable slot. */
    int64_t tid;             /* heap tid encoded as int64_t (same form as map_ptr). */
} xl_dpv_vacuum_entry;

typedef struct {
    Oid       dbOid;                /* 4 */
    Oid       indexRelId;           /* 4 */
    SegmentId sid;                  /* 4 — the exact sid this batch vacuums.
                                     * For a multi-sid segment, the primary
                                     * emits one of these records per sid. */
    uint32    owner_version;        /* 4 — version of the segment containing
                                     * `sid` on the primary at WAL-emit time;
                                     * 0 if owner is still a memtable. */
    uint32    subversion;           /* 4 — new subversion the primary is about
                                     * to write for the bitmap file;
                                     * UINT32_MAX sentinel if owner is a
                                     * memtable (no subversion file). */
    uint32    is_memtable_owner;    /* 4 — 1 if owner is memtable on primary,
                                     * else 0. Optimization hint only; the
                                     * standby routes by local state (§4.1). */
    uint32    entry_count;          /* 4 */
    uint32    _pad;                 /* 4 — keeps total 32B / 8-aligned so the
                                     * int64_t tid in the entry trailer is
                                     * naturally aligned. */
    /* followed by xl_dpv_vacuum_entry entries[entry_count],
     * EMITTED IN ASCENDING sid_local_idx ORDER — i.e. in the physical-offset
     * order of map_ptr within the sid's range. */
} xl_dpv_vacuum_tombstones;
/* Static asserts at the C declaration site:
 *   sizeof(xl_dpv_vacuum_tombstones) == 32
 *   sizeof(xl_dpv_vacuum_tombstones) % 8 == 0
 *   sizeof(xl_dpv_vacuum_entry) == 16
 */
```

Op code: `XLOG_DPV_VACUUM_TOMBSTONES = 0x80` (reusing the slot that previously held `XLOG_DPV_SEGMENT_VACUUM_TOMBSTONES`). The old `0x20` (`XLOG_DPV_MEMTABLE_TOMBSTONE`) is permanently retired.

### 2.2 Why one record per sid

- Each record encloses one sid's vacuum batch atomically. The record *is* the per-sid boundary; no separate `BEGIN_VACUUM_SID` / `END_VACUUM_SID` markers are needed.
- A multi-sid segment vacuum on the primary emits one record per touched sid; all are XLogInsert'd, then a single `XLogFlush` on the latest LSN occurs before the new bitmap subversion file is written. The §11 protocol is preserved.
- The standby applies each record independently against whichever owner it locally has for that sid. Routing is per-record, not per-batch.

### 2.3 Why `sid_local_idx` is included alongside `tid`

- **Fast path.** When the standby has the same segment version as the primary at WAL-emit time, the `sid_local_idx` directly indexes `map_ptr[offsets[sid].start_offset + sid_local_idx]` — no comparison needed.
- **Memtable owner direct-apply.** When the primary's owner is a memtable and the standby still has that memtable, `sid_local_idx` equals the memtable slot index — direct indexing into `mt->bitmap`.
- **Cross-version paths still work** because entries are in ascending `sid_local_idx`, which equals ascending physical-offset order within the sid. Combined with the offset-order invariant (§1), this lets the standby walk its own `map_ptr` slice with a single advancing WAL pointer and match by tid. The `sid_local_idx` from the primary is ignored on these paths.

---

## 3. Primary-side emission

### 3.1 Three call sites in `bulk_delete_lsm_index`

| Site | Lines (current) | Owner type | Single sid? |
|------|----------------|-----------|-------------|
| 1 — growing memtable | [lsmindex.c:2003-2138](../../pgvector/src/lsmindex.c#L2003-L2138) | memtable | yes |
| 2 — immutable memtables | [lsmindex.c:2155-2344](../../pgvector/src/lsmindex.c#L2155-L2344) | memtable | yes |
| 3 — flushed segments via disk scan | [lsmindex.c:2357-2518](../../pgvector/src/lsmindex.c#L2357-L2518) | segment | possibly multi-sid |

### 3.2 Sites 1 and 2 — memtable vacuum

Each memtable has exactly one sid (`mt->memtable_id`). Walk `mt->ready[]` / `mt->tids[]` in slot order; for each deleted slot, append `{sid_local_idx=i, tid=mt->tids[i]}`. Entries are naturally in ascending `sid_local_idx`.

Two sub-cases depend on `is_persistent`:

- **`is_persistent` (memtable already flushed to a segment on primary):** emit with `is_memtable_owner=0`, `owner_version=find_latest_segment_version(...)`, `subversion=next_subversion`. `XLogInsert` → `XLogFlush(lsn)` → `write_bitmap_file_with_subversion(...)`. §11 ordering invariant preserved.
- **Non-persistent (memtable still in heap):** emit with `is_memtable_owner=1`, `owner_version=0`, `subversion=UINT32_MAX`. No `XLogFlush` required before `write_bitmap_for_memtable` — memtables have no subversion-file ordering invariant to protect; the file is a hint and the standby reconstructs from WAL.

`RemoveFromStatusMemtable` ([statuspage.c:842](../../pgvector/src/statuspage.c#L842)) **no longer emits its own WAL**. The line `dpv_emit_memtable_tombstone(...)` is deleted; the GenericXLog work for the heap status page tuple stays (orthogonal concern).

### 3.3 Site 3 — flushed segment vacuum

After loading `bitmap_ptr` and `mapping_ptr` for the segment at `(start_sid_disk, end_sid_disk, seg_version)`:

1. **Single-sid fast path** (`start_sid_disk == end_sid_disk`): no offset file needed. `sid_local_idx == i`. Emit one record with `sid = start_sid_disk`.

2. **Multi-sid path:**
   1. `load_offset_file(indexRelId, start_sid_disk, end_sid_disk, seg_version, &offsets, true)` → `SegmentOffsetRange[]` with one entry per sid.
   2. Allocate per-sid scratch buffers (one `xl_dpv_vacuum_entry *` per sid, sized up to that sid's range).
   3. Walk `mapping_ptr[]` once in ascending index order. Maintain a single advancing pointer into `offsets[]` (since `mapping_ptr` is laid out in sid-then-offset order). For each deleted entry at global index `i`: append `{sid_local_idx = i - offsets[k].start_offset, tid = mapping_ptr[i]}` to sid `offsets[k].sid`'s buffer.
   4. For each sid that collected at least one entry: `XLogInsert` via `dpv_emit_vacuum_tombstones(...)`, capturing each returned LSN.
   5. `XLogFlush(max_lsn)` — single flush, covers all per-sid records emitted in this batch.
   6. `write_bitmap_file_with_subversion(...)` — §11 ordering invariant preserved (WAL durable before file write).

All emission is gated by `dpv_replication_role == DPV_ROLE_PRIMARY` as today.

### 3.4 Wire-format example

A vacuum that deletes 3 tids in sid 5 (memtable still in heap), and 7 tids in sid 8 + 2 tids in sid 9 of a merged segment, emits **three** WAL records: one for sid 5 (`is_memtable_owner=1`), one for sid 8, one for sid 9 (both `is_memtable_owner=0`, same `owner_version` and `subversion` since they share a segment). All three are flushed together before the segment's new bitmap subversion file is written.

---

## 4. Standby-side redo

### 4.1 Routing

A single function `dpv_apply_vacuum_tombstones(hdr, entries)` replaces the four-path routing in [segment_vacuum_redo.c](../../pgvector/src/segment_vacuum_redo.c). For a given `hdr->sid`:

1. **Check the pool first.** Walk the pool linearly to find the (at most one) segment whose range contains `hdr->sid`. If found → apply to that segment (§4.2). Done.
2. **Fall back to memtable.** Only if no pool segment contains `hdr->sid`, check `memtable_idxs[]` for a memtable with `memtable_id == hdr->sid`. If found → apply to that memtable (§4.3). Done.
3. **Silent skip otherwise.** Neither owner present. The bits will arrive via the side-channel file fetch (the bitmap subversion file the primary wrote contains them) and be loaded by the eventual `load_and_set_segment` call during adoption.

**Order matters.** In the transient window between `dpv_pool_adopt` installing a new segment and the startup process replaying `RELEASE_MEMTABLE` for the memtable that was folded into it, the standby holds **both** owners for that sid. Applying to the segment is the only durable choice: the memtable's bitmap is about to be discarded, and after adoption completes, future memtable bit-flips no longer propagate into the segment (adoption captures the cover bits only at adoption time, in `union_deletion_entries_into_new_seg`).

The `hdr->is_memtable_owner` flag is **not** consulted for routing — only as an optimization hint inside the memtable branch (§4.3). Routing is driven entirely by the standby's local state.

### 4.2 Segment branch — 2-pointer merge

The standby's segment now caches its offset file as `seg->offsets[]` (see §5). Find the slice for `hdr->sid` via binary search over `seg->offsets[]` → `[start_offset, end_offset)`.

```c
pthread_mutex_lock(&seg->per_seg_mutex);

bool fast_path = (seg->version == hdr->owner_version &&
                  !hdr->is_memtable_owner);

if (fast_path) {
    /* sid_local_idx is bytewise valid; no comparison needed. */
    for (uint32 w = 0; w < hdr->entry_count; w++) {
        Size m = start_offset + entries[w].sid_local_idx;
        if (m < end_offset && !IS_SLOT_SET(seg->bitmap_ptr, m)) {
            SET_SLOT(seg->bitmap_ptr, m);
            seg->delete_count++;
        }
    }
} else {
    /* Cross-version or memtable-owner-on-primary: walk both with a
     * single advancing WAL pointer. WAL entries are in physical-offset
     * order of the primary's sid range, which is the same relative
     * order as the standby's sid range (invariant §1). */
    uint32 w = 0;
    for (Size m = start_offset; m < end_offset && w < hdr->entry_count; m++) {
        if (seg->map_ptr[m] == entries[w].tid) {
            if (!IS_SLOT_SET(seg->bitmap_ptr, m)) {
                SET_SLOT(seg->bitmap_ptr, m);
                seg->delete_count++;
            }
            w++;
        }
        /* else: standby-side tid not in this WAL batch — skip. */
    }
    /* Unmatched trailing WAL entries (w < entry_count) are tids
     * compacted out of the standby by a later rebuild — silently drop. */
}

pthread_mutex_unlock(&seg->per_seg_mutex);
```

After the merge, if any bit flipped 0→1 AND `hdr->is_memtable_owner == 0`, the standby writes a local subversion file via `write_bitmap_file_with_subversion(indexRelId, seg->segment_id_start, seg->segment_id_end, seg->version, hdr->subversion, seg->bitmap_ptr, ...)`. If `is_memtable_owner == 1`, no subversion file is written — the new bits will be picked up next time the segment's bitmap file is written from a primary-side segment vacuum, or persisted via `union_deletion_entries_into_new_seg` at adoption time if this segment is later replaced.

Cost: O(end_offset − start_offset) per record on the slow path; O(entry_count) on the fast path. No allocation, no comparator, no aux.

### 4.3 Memtable branch

Two sub-cases inside the memtable branch:

- **`hdr->is_memtable_owner == 1` AND memtable found:** `sid_local_idx` equals the slot index. Direct indexing — no walk:
  ```c
  LWLockAcquire(&mt->vacuum_lock, LW_EXCLUSIVE);
  for (w = 0; w < hdr->entry_count; w++) {
      uint32 li = entries[w].sid_local_idx;
      if (li < mt->capacity) SET_SLOT(mt->bitmap, li);
  }
  LWLockRelease(&mt->vacuum_lock);
  ```
- **`hdr->is_memtable_owner == 0` AND memtable found** (primary flushed already; standby's segment hasn't arrived yet): `sid_local_idx` no longer corresponds to a memtable slot. Use the 2-pointer walk against `mt->tids[0..current_size)` — same as §4.2's slow path, replacing `seg->map_ptr` with `mt->tids` and `seg->bitmap_ptr` with `mt->bitmap`.

In both sub-cases the bits stay on the memtable. When the standby's segment for this sid later arrives and adoption runs `union_deletion_entries_into_new_seg` (§5.2), those bits propagate into the new segment via the same offset-order merge.

---

## 5. Adoption and `sorted_perm` removal

### 5.1 New `FlushedSegmentData` field

```c
typedef struct FlushedSegmentData {
    /* ... existing fields ... */
    SegmentOffsetRange *offsets;     /* per-sid [start_offset, end_offset). */
    uint32              offsets_count;
    /* sorted_idx field removed. */
} FlushedSegmentData;
```

`offsets[]` is allocated in `load_and_set_segment` via `load_offset_file(...)` (this function already runs there for some code paths; the call site moves so every load populates `offsets[]`). Freed in `cleanup_flushed_segment` and `discard_reserved_segment` alongside `map_ptr` / `bitmap_ptr`.

### 5.2 `union_segment_bitmap_via_offset_merge`

Replaces `union_segment_bitmap_via_tid` ([segment_adoption.c:96](../../pgvector/src/segment_adoption.c#L96)). For each sid in the intersection of `old->offsets[]` and `new->offsets[]`:

```c
SegmentId sid = /* shared sid */;
Size old_lo = lookup_offset(old, sid).start_offset;
Size old_hi = lookup_offset(old, sid).end_offset;
Size new_lo = lookup_offset(new, sid).start_offset;
Size new_hi = lookup_offset(new, sid).end_offset;

Size i_old = old_lo, i_new = new_lo;
while (i_old < old_hi && i_new < new_hi) {
    if (old->map_ptr[i_old] == new->map_ptr[i_new]) {
        if (IS_SLOT_SET(old->bitmap_ptr, i_old) &&
            !IS_SLOT_SET(new->bitmap_ptr, i_new)) {
            SET_SLOT(new->bitmap_ptr, i_new);
            new->delete_count++;
        }
        i_old++; i_new++;
    } else {
        /* old has a tid that new compacted out. */
        i_old++;
    }
}
```

Justification: `new`'s tids for this sid are a subsequence (in same relative order) of `old`'s — compaction only removes. So a 2-pointer merge in O(old_hi − old_lo) is sufficient.

### 5.3 `union_deletion_entries_into_new_seg`

Replaces `union_deletion_tids_into_new_seg` ([segment_adoption.c:155](../../pgvector/src/segment_adoption.c#L155)). The fetcher now collects memtable cover bits as per-sid groups (one group per cover memtable, since each memtable carries one sid). Group representation:

```c
typedef struct {
    SegmentId    sid;
    const int64_t *tids;   /* tids of *set* bits in mt->bitmap, in insertion order. */
    uint32        n_tids;
} DpvVacuumGroup;
```

Per group, binary-search `new->offsets[]` for `sid` to get `[start, end)`, then 2-pointer merge between `new->map_ptr[start..end)` and `group->tids[]`. Same algorithm shape as §5.2.

### 5.4 ADOPT task payload schema change

`segment_fetcher.c` populates the ADOPT task DSM trailer. The current flat `int64_t *deletion_tids` becomes a sequence of groups; trailer layout:

```
uint32 n_groups;
[ SegmentId sid; uint32 n_tids; int64_t tids[n_tids]; ] * n_groups
```

`dpv_pool_adopt` now takes `(const DpvVacuumGroup *groups, int n_groups)` instead of `(const int64_t *deletion_tids, int n_deletion_tids)`.

### 5.5 Files / fields deleted

- `pgvector/src/sorted_perm.h`
- `pgvector/src/sorted_perm.c`
- `src/sorted_perm.o` line from `pgvector/Makefile`
- `uint32_t *sorted_idx` field on `FlushedSegmentData` ([lsm_segment.h:50](../../pgvector/src/lsm_segment.h#L50))
- `dpv_sorted_perm_free(segment)` calls in [lsm_segment.c:187](../../pgvector/src/lsm_segment.c#L187) and [lsm_segment.c:249](../../pgvector/src/lsm_segment.c#L249)
- All `#include "sorted_perm.h"` lines (segment_adoption.c, segment_vacuum_redo.c, lsm_segment.c)
- Old `xl_dpv_memtable_tombstone` struct, `dpv_emit_memtable_tombstone` helper, `redo_memtable_tombstone` callback, `dpv_standby_memtable_tombstone` standby handler, `XLOG_DPV_MEMTABLE_TOMBSTONE` op code, dispatcher case, identify-string entry
- Old `xl_dpv_segment_vacuum_tombstones` struct, the per-record `local_idx + tid` layout that lacked `sid` (replaced by the new schema reusing op code `0x80`)
- `dpv_emit_segment_vacuum_tombstones` helper (replaced by `dpv_emit_vacuum_tombstones`)

---

## 6. §11 ordering invariant — unchanged

For every primary-side vacuum that produces a new bitmap subversion file, the order remains:

1. Compute the in-memory bitmap delta and the per-sid entry buffers.
2. `XLogInsert` one record per touched sid.
3. `XLogFlush(max_lsn)` — single flush over all records for this batch.
4. `write_bitmap_file_with_subversion` for the segment.

Step 3 may flush multiple records together; this is sound because all four records belong to the same logical vacuum batch, and any reorder of WAL inserts within a single flush window does not violate the "WAL durable before file write" guarantee.

---

## 7. Concurrency notes

- **Standby per-record locking** is unchanged — `seg->per_seg_mutex` for the segment branch, `mt->vacuum_lock` for the memtable branch. No new locks; `sorted_perm`'s internal mutex acquisition disappears with the structure.
- **Per-sid scoping is the boundary.** Within one WAL record, all entries belong to one sid. Concurrent control on the standby is naturally per-sid: two redo invocations for different sids never need to coordinate (they touch disjoint slices of `map_ptr` and `bitmap_ptr`). Two for the same sid serialize on `seg->per_seg_mutex` or `mt->vacuum_lock`.
- **Pool-walk correctness on the standby.** The "find the segment that contains `sid`" walk takes `pool->seg_lock` read-shared. Adoption takes it write-exclusive — so adoption cannot complete while a redo callback holds the read lock. If the walk finds no segment, the redo callback releases the read lock before checking memtables; a concurrent adoption that installs a segment in that window will pick up the bits later via `union_deletion_entries_into_new_seg`.

---

## 8. Tests

All existing Plan 3 tests apply unchanged:

- `pgvector/test/t/120_replication_vacuum_simple.pl`
- `pgvector/test/t/121_replication_vacuum_merge_race.pl`
- `pgvector/test/t/122_replication_rebuild_translation.pl`
- `pgvector/test/t/123_replication_attach_barrier.pl`

Plans 1 / 2 regression tests:

- `pgvector/test/t/110_replication_segment_flush.pl`
- `pgvector/test/t/111_replication_segment_merge.pl`
- `pgvector/test/t/112_replication_queue_restart.pl`

No new tests required for the refactor; the existing suite covers all four standby paths (fast, rebuild race, merge race, memtable target) which collapse into the new routing logic.

---

## 9. Risks and follow-ons

- **Wire-format break.** Any in-flight WAL using `0x20` or the old `0x80` layout will fail to replay. Acceptable — the project is pre-release; fresh test clusters are the norm.
- **Adoption DSM payload change.** Callers of `dpv_pool_adopt` outside `segment_fetcher.c` (none today) would also need updating. No grep hits at present.
- **Multi-sid vacuum batches emit more WAL records than today.** A vacuum touching N sids in a merged segment emits N records (was 1). Each is small (header ~32B + entries 12B each). For a typical workload this is a wash, since the old layout's larger range header is replaced by a smaller per-sid header. If profiling later shows fsync rate is a bottleneck under heavy multi-sid vacuum, batching multiple sids into a single record with a sid-index sub-header is a straightforward follow-on.
- **`SegmentOffsetRange` memory.** Caching offsets[] on every loaded segment adds 16B × sid_count per segment to resident memory. For a segment with 64 sids (extreme) this is 1 KB — negligible against the segment's own multi-megabyte footprint.

---

## End
