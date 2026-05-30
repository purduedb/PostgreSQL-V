# Decouple LSM Recovery from Index Load

**Date:** 2026-05-16
**Status:** Design — awaiting review before implementation
**Scope:** `pgvector/src/lsmindex.{c,h}`, `pgvector/src/vector_index_worker.c`, plus all callers that check `LSMIndexBufferSlot.valid`.

---

## 1. Motivation

`load_lsm_index_internal` ([lsmindex.c:527](../../../pgvector/src/lsmindex.c#L527)) today does two things in sequence:

1. Calls `index_load_blocking` to populate the in-memory `FlushedSegmentPool` from disk.
2. Runs three "recovery" steps that reconcile the in-memory state against the status page (vacuum past-max-sid vectors, vacuum missing-tid vectors, rebuild memtables).

Steps 1 and 2 of recovery write new bitmap subversions to disk and then issue `segment_update_blocking(... SEGMENT_UPDATE_VACUUM ...)` to make the worker re-merge those subversions into its in-memory bitmap. This is wasted work: the worker just loaded the bitmap from disk milliseconds before, and the disk loader already resolves the latest subversion.

We want to:

- **Swap the order**: run recovery first against on-disk state only, then load segments into the in-memory pool. The loader naturally picks up the recovery-written subversions, so the per-segment `SEGMENT_UPDATE_VACUUM` round-trips disappear.
- **Split slot state into "Writable" and "Queryable"**: recovery alone is enough for writers and standby redo callbacks; loading is only needed for queries. This is the foundation for a future change that lets a hot-standby keep its memtable up to date via WAL redo without ever loading segment indexes.

The standby integration is **out of scope** for this change. This spec covers only the swap, the new state machine, and the worker-side gating that makes it safe for maintenance tasks to fire against a slot that has been recovered but not loaded. Follow-on PRs are listed in §9.

---

## 2. State machine

Replace the current ternary `valid` field (0=free, 1=ready, 2=loading) on `LSMIndexBufferSlot` with a five-state enum:

```c
typedef enum LSMSlotState {
    LSM_SLOT_FREE          = 0,
    LSM_SLOT_RECOVERING    = 1,   /* recovery worker is in recover_lsm_index_internal */
    LSM_SLOT_WRITABLE      = 2,   /* recovery done; FlushedSegmentPool NOT initialized */
    LSM_SLOT_LOADING_INDEX = 3,   /* a reader is calling index_load_blocking */
    LSM_SLOT_QUERYABLE     = 4    /* segments mmap- or fully-loaded; pool initialized */
} LSMSlotState;
```

Predicates (added to `lsmindex.h`):

- `is_writable(v) := v == WRITABLE || v == LOADING_INDEX || v == QUERYABLE`
- `is_queryable(v) := v == QUERYABLE`

The existing `slot->load_error` and `slot->load_cv` are reused. `load_cv` is broadcast on every transition out of a `LOADING_*` state.

Transitions:

```
FREE --register--> RECOVERING --recovery done--> WRITABLE
WRITABLE --reader CAS--> LOADING_INDEX --index_load_blocking done--> QUERYABLE
```

There is no transition back from `QUERYABLE`. Once a slot is loaded, it is loaded for the lifetime of the slot.

---

## 3. Rename

`load_lsm_index_internal` is misleading after the swap. Rename to **`recover_lsm_index_internal`**. The enum constant is **`LSM_SLOT_RECOVERING`** (not `LOADING_RECOVERY`).

Everything else keeps its current name:

- `IndexLoadWorker`, `IndexLoadCoordinator`, `SharedIndexLoadCoordinator`, `index_load_worker.c/.h` — unchanged. The recovery worker reuses the existing infrastructure.
- `index_load_blocking`, `IndexLoadTaskType`, `IndexLoadTaskData` — unchanged. These genuinely still describe the segment-load task on the `vector_index_worker`, which is separate from recovery.
- `slot->load_error`, `slot->load_cv` — unchanged. The CV is now used for two transitions (RECOVERING→WRITABLE and LOADING_INDEX→QUERYABLE), but a single CV per slot is enough since waiters re-check `valid` on every wake.

---

## 4. Recovery worker: `recover_lsm_index_internal`

New body shape:

```c
void recover_lsm_index_internal(Oid index_relid, uint32_t slot_idx)
{
    Relation index = index_open(index_relid, AccessShareLock);
    LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[slot_idx];
    LSMIndex lsm = &slot->lsmIndex;

    /* 1. Status-page reads */
    SegmentId status_growing_sid = GetStatusGrowingSid(index, MAIN_FORKNUM);
    int num_sids;
    SegmentId *sids = GetStatusMemtableSids(index, MAIN_FORKNUM, &num_sids);
    if (sids != NULL && num_sids > 0)
        qsort(sids, num_sids, sizeof(SegmentId), compare_sids_desc);
    SegmentId max_memtable_sid =
        (sids && num_sids ? (sids[0] == status_growing_sid
                              ? status_growing_sid
                              : status_growing_sid - 1)
                          : 0);

    /* 2. Initialize LSMIndex header fields */
    IndexType index_type; uint32_t dim, elem_size;
    if (!read_lsm_index_metadata(index_relid, &index_type, &dim, &elem_size))
        elog(ERROR, "[recover_lsm_index_internal] metadata read failed for %u",
             index_relid);
    lsm->indexRelId = index_relid;
    lsm->index_type = index_type;
    lsm->dim        = dim;
    lsm->elem_size  = elem_size;

    /* 3. Scan segment metadata files */
    SegmentFileInfo files[MAX_SEGMENTS_COUNT];
    int file_count = scan_segment_metadata_files(index_relid, files,
                                                 MAX_SEGMENTS_COUNT);

    /* 4. Recovery step 1: vacuum segments with end_sid > max_memtable_sid.
     *    Same body as today, but WITHOUT the trailing
     *    segment_update_blocking(... SEGMENT_UPDATE_VACUUM ...) call.
     *    Disk subversion is still written by write_bitmap_file_with_subversion.
     */

    /* 5. Recovery step 2: vacuum segments overlapping memtable sids.
     *    Same body as today, but WITHOUT the trailing segment_update_blocking.
     */

    /* 6. Initialize sealed memtable array; populate flushed_not_released.
     *    (Unchanged from today.)
     */

    /* 7. Recovery step 3: construct memtables from heap + status pages.
     *    (Unchanged from today — opens heap_rel internally, closes at end.)
     */

    /* heap_rel / index relation cleanup as today. */

    /* 8. Publish WRITABLE — NOT QUERYABLE */
    pg_write_barrier();
    pg_atomic_write_u32(&slot->valid, LSM_SLOT_WRITABLE);
    ConditionVariableBroadcast(&slot->load_cv);
}
```

Key deltas vs. today's `load_lsm_index_internal`:

- `index_load_blocking(index_relid, slot_idx)` at [lsmindex.c:551](../../../pgvector/src/lsmindex.c#L551) is **removed**.
- `segment_update_blocking(... SEGMENT_UPDATE_VACUUM ...)` inside recovery step 1 at [lsmindex.c:645-646](../../../pgvector/src/lsmindex.c#L645) is **removed**.
- `segment_update_blocking(... SEGMENT_UPDATE_VACUUM ...)` inside recovery step 2 at [lsmindex.c:805-806](../../../pgvector/src/lsmindex.c#L805) is **removed**.
- Terminal state is `LSM_SLOT_WRITABLE`. The transition to `LSM_SLOT_QUERYABLE` happens later, triggered by a reader.

---

## 5. Reader-triggered on-demand load

A new helper centralizes the CAS + wait pattern. Defined in `lsmindex.c`:

```c
static bool
ensure_index_loaded(int slot_idx)
{
    LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[slot_idx];
    uint32 v = pg_atomic_read_u32(&slot->valid);

    if (v == LSM_SLOT_QUERYABLE)
        return true;

    if (v == LSM_SLOT_WRITABLE)
    {
        uint32 expected = LSM_SLOT_WRITABLE;
        if (pg_atomic_compare_exchange_u32(&slot->valid, &expected,
                                           LSM_SLOT_LOADING_INDEX))
        {
            /* Winner clears any stale load_error before retrying. */
            pg_atomic_write_u32(&slot->load_error, 0);
            PG_TRY();
            {
                index_load_blocking(slot->lsmIndex.indexRelId, slot_idx);
                pg_write_barrier();
                pg_atomic_write_u32(&slot->valid, LSM_SLOT_QUERYABLE);
            }
            PG_CATCH();
            {
                pg_atomic_write_u32(&slot->load_error, 1);
                pg_atomic_write_u32(&slot->valid, LSM_SLOT_WRITABLE);
                ConditionVariableBroadcast(&slot->load_cv);
                PG_RE_THROW();
            }
            PG_END_TRY();
            ConditionVariableBroadcast(&slot->load_cv);
            return true;
        }
        /* CAS lost — fall through to wait. */
    }

    /* Wait for the in-flight loader (or back to WRITABLE on error). */
    ConditionVariablePrepareToSleep(&slot->load_cv);
    for (;;)
    {
        v = pg_atomic_read_u32(&slot->valid);
        if (v == LSM_SLOT_QUERYABLE) { ConditionVariableCancelSleep(); return true; }
        if (v == LSM_SLOT_WRITABLE)  { ConditionVariableCancelSleep(); return false; }
        ConditionVariableSleep(&slot->load_cv, PG_WAIT_EXTENSION);
    }
}
```

Entry-point wiring:

| Function | Today (waits for) | New (waits for) | Triggers load? |
|---|---|---|---|
| `insert_lsm_index` (via `get_lsm_index`) | `valid == 1` | `is_writable(v)` | No |
| `search_lsm_index` (via `get_lsm_index_for_read`) | `valid == 1` | `is_writable(v)`, then `ensure_index_loaded` | **Yes** |
| `bulk_delete_lsm_index` (via `get_lsm_index_idx`) | `valid == 1` | `is_writable(v)` | No |
| `register_lsm_index` slow path | `valid == 1` | `is_writable(v)` | No |
| `get_lsm_index_idx_no_loading` (standby, fetcher introspection) | `valid == 1` for "loaded" | `is_writable(v)` for "slot has data"; callers separately check `valid == QUERYABLE` if they need the pool | No |

The read path adds one call to `ensure_index_loaded` after `get_lsm_index` returns. To keep call sites tidy, expose a thin wrapper:

```c
LSMIndex get_lsm_index_for_read(Relation index);
/*  - waits until is_writable(slot->valid), as get_lsm_index does today,
 *  - additionally guarantees slot->valid == LSM_SLOT_QUERYABLE before returning,
 *  - throws on load error.
 */
```

`search_lsm_index` becomes the only caller of `get_lsm_index_for_read` for now. `get_lsm_index` keeps its current name and is used by writers (e.g., `insert_lsm_index`); its semantics change to "wait until `is_writable(v)`".

---

## 6. Worker-side gating

Every maintenance task handler in [vector_index_worker.c](../../../pgvector/src/vector_index_worker.c) checks slot state on entry. If `!is_queryable(v)`, the handler logs at `DEBUG1`, sets `maint_status = 0`, signals the backend, and breaks. Disk-side state is unaffected, so the eventual `index_load_blocking` picks up the latest bitmap subversion.

Helper added once in `vector_index_worker.c`:

```c
static inline bool
slot_is_queryable(int lsm_idx)
{
    if (lsm_idx < 0 || lsm_idx >= INDEX_BUF_SIZE)
        return false;
    return pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[lsm_idx].valid)
           == LSM_SLOT_QUERYABLE;
}
```

Per task type:

| Task | Today's behavior | New gate |
|---|---|---|
| `SEGMENT_UPDATE_VACUUM` ([vector_index_worker.c:222-283](../../../pgvector/src/vector_index_worker.c#L222)) | finds segment in pool, ORs latest bitmap subversion | `if (!slot_is_queryable(lsm_idx)) { maint_status=0; break; }` |
| `SEGMENT_UPDATE_ADOPT` ([vector_index_worker.c:285-313](../../../pgvector/src/vector_index_worker.c#L285)) | calls `dpv_attempt_adoption`; returns `maint_status=2` for `INDEX_UNLOADED` | Optional fast-path: if `!slot_is_queryable`, set `maint_status=2` directly without calling `dpv_attempt_adoption`. Net behavior preserved. |
| `SEGMENT_UPDATE_REGULAR` ([vector_index_worker.c:205+](../../../pgvector/src/vector_index_worker.c#L205)) | post-flush bitmap merge | Same gate. The flush itself wrote disk-side bitmap; no-op on in-memory is safe. |
| `SEGMENT_UPDATE_REBUILD_FLAT` / `_REBUILD_DELETION` / `_MERGE` ([vector_index_worker.c:1516-1520](../../../pgvector/src/vector_index_worker.c#L1516)) | claimed and executed by internal merge-thread pool inside `vector_index_worker` | Same gate on dispatch. Additionally, `scan_and_claim_merge_task_pool` ([vector_index_worker.c:826+](../../../pgvector/src/vector_index_worker.c#L826)) should skip slots where `!slot_is_queryable` so tasks are not claimed in the first place. The pool is empty for WRITABLE-only slots, but the defensive check avoids paying the per-slot scan cost. |
| `IndexLoadTaskType` ([vector_index_worker.c:322-395](../../../pgvector/src/vector_index_worker.c#L322)) | initializes pool, loads segments | **No gate.** This is the task that performs WRITABLE→QUERYABLE; `ensure_index_loaded` has already CAS'd to LOADING_INDEX. |
| `InternalSegmentUpgradeTaskType` ([vector_index_worker.c:396+](../../../pgvector/src/vector_index_worker.c#L396)) | mmap → full upgrade | Inherits gating from its parent IndexLoadTask. No new check. |

Edge cases:

1. **LOADING_INDEX in-flight + concurrent VACUUM.** A bulk_delete is running while a reader has transitioned WRITABLE→LOADING_INDEX. A VACUUM task may arrive at the worker before, during, or after the load. In all three the gate fails (`valid != QUERYABLE`) until the load finishes, so:
   - Before/during: gate fires, no-op success. Disk subversion already written.
   - After: gate passes, normal in-memory bitmap merge. Disk subversion is still the latest.
   All three outcomes are correct.

2. **Merge thread scan on WRITABLE-only slots.** `scan_and_claim_merge_task_pool` iterates `SharedLSMIndexBuffer->slots`. WRITABLE-only slots have an empty `FlushedSegmentPool` (it is populated only by `index_load_blocking`), so the inner loop finds nothing — but the per-slot atomics counters (`pool->flat_count` etc.) are still touched. The added `slot_is_queryable` check is defensive and cheap.

---

## 7. Concurrency, error paths, crash safety

**Concurrent registration during recovery.** Unchanged from today. `register_lsm_index` CASes `valid` FREE→RECOVERING; losers wait on `load_cv`.

**Recovery error.** If `recover_lsm_index_internal` throws, the worker resets `valid` to `FREE`, sets `load_error=1`, broadcasts `load_cv`. Waiters re-check, mirroring [lsmindex.c:976-983](../../../pgvector/src/lsmindex.c#L976).

**Load error.** If `index_load_blocking` throws inside `ensure_index_loaded`, the helper reverts `valid` from LOADING_INDEX to WRITABLE, sets `load_error=1`, broadcasts, re-throws. The slot stays writable; only the load is retryable. Subsequent readers CAS WRITABLE→LOADING_INDEX and clear `load_error` at the start of each attempt.

**Crash safety.** Every disk write in recovery is a self-contained bitmap subversion file with a monotonically increasing subversion number. The loader always resolves `LOAD_LATEST_VERSION`. Possible crash points:

- **Mid-recovery (after some subversions written, before others)**: on restart, recovery re-runs and re-applies the same disk reads. Bitmap OR is idempotent; recomputed `delete_count` is the same. Stale partial subversions are simply superseded by the next attempt.
- **Between recovery and load**: identical to "never-loaded-yet" steady state.
- **Mid-load**: pool is in volatile shared memory and cleared on restart; recovery re-runs; reader re-triggers load.

**Subversion monotonicity.** Recovery uses `find_latest_bitmap_subversion + 1`. Only one recovery runs per slot. `bulk_delete` cannot race with recovery: it waits for `>= WRITABLE`, which is after recovery completes.

**Performance.** Recovery's per-segment `SEGMENT_UPDATE_VACUUM` round-trips disappear — at minimum a `min(file_count, num_sids)`-scale set of synchronous round-trips for cold start. The existing `DEBUG1 [load_lsm_index_internal] Recovery step 1&2 overhead` log will reflect the reduction.

---

## 8. Build & static checks

- `make` in `pgvector/` builds clean (no new warnings).
- `compile_commands.json` regenerated if the file list changes; `clangd` reports no new diagnostics.
- Every existing `valid == 1` / `valid == 2` literal across the codebase is audited and converted to the appropriate named predicate. Specifically the sites at: [lsmindex.c:191,198,964,968,997,1001,1012,1037,1041,1052](../../../pgvector/src/lsmindex.c#L191), [standby_memtable.c:27](../../../pgvector/src/standby_memtable.c#L27).
- The two `segment_update_blocking(... SEGMENT_UPDATE_VACUUM ...)` calls inside the renamed `recover_lsm_index_internal` are removed.
- The `index_load_blocking` call inside `recover_lsm_index_internal` is removed.
- `LSMSlotState` enum is defined in [lsmindex.h](../../../pgvector/src/lsmindex.h); no raw integer state values remain.

Manual functional verification will be performed by the maintainer.

---

## 9. Out of scope (follow-on changes)

- Standby redo callbacks calling `register_lsm_index` to drive WAL-side recovery.
- Standby memtable eviction when a segment-arrival redo signals adoption.
- `flushed_not_released` simplification or replacement.
- Demand-driven `index_load_blocking` from a query on a hot-standby (the mechanism in §5 generalizes naturally, but exercising it on standby is a separate change).
