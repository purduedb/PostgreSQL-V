# Crash Resilience, mmap-Compatible Index Format, and mmap Cold-Start

**Date:** 2026-05-06  
**Branch:** main  
**Files modified:** `tasksend.c`, `ringbuffer.h`, `ringbuffer.c`, `vector_index_worker.c`, `lsm_segment.h`, `lsm_segment.c`, `lsmindex.h`, `lsmindex_io.c`, `lsm_merge_worker.c`, `vectorindeximpl.hpp`, `vectorindeximpl.cpp`

---

## Motivation

Three related problems motivated this work:

1. **Backends hung indefinitely** when the vector-index worker crashed. `WaitLatch` was called with `timeout=0` (infinite), so a crash left every connected backend stuck forever until the postmaster died.
2. **The on-disk index format** used a multi-key BinarySet envelope (`[name_size][data_size][name][data]...`) that is incompatible with Knowhere's `DeserializeFromFile` API, which reads raw Faiss bytes directly and is the only path that supports `enable_mmap`.
3. **Cold-start latency** after a worker restart was dominated by loading all segment indexes fully into memory before the index became searchable. With large datasets this could take minutes.

---

## Feature 1 — Worker Crash Resilience

### Crash scenario

The target crash is **`kill -9 <worker_pid>`** (SIGKILL). This is fundamentally different from `kill -SIGTERM`:

| Signal | Effect |
|--------|--------|
| SIGKILL | PostgreSQL postmaster detects abnormal worker exit and enters crash recovery: it sends SIGTERM to every other backend, then reinitializes shared memory. All backends and the old shared memory state are gone before the new worker starts. |
| SIGTERM | Only the specific worker is restarted. Backends survive; shared memory is intact. |

Because the worker is registered with `BGWORKER_SHMEM_ACCESS`, an unclean exit (SIGKILL) tells the postmaster that shared memory may be corrupted, triggering a full cluster restart.

### Mechanism

No crash-detection state is stored in shared memory. Backends detect the crash through PostgreSQL's native latch mechanism:

1. The postmaster sends SIGTERM to all backends during crash recovery.
2. Each backend's `die()` signal handler fires, setting `ProcDiePending` and calling `SetLatch(MyLatch)`.
3. `WaitLatch` — called in both `vector_search_get_result` and `submit_and_wait_maintenance` — includes the `WL_POSTMASTER_DEATH` flag. When crash recovery begins, this condition fires, causing `WaitLatch` to return.
4. The backend checks `rc & WL_POSTMASTER_DEATH` and immediately calls `ereport(ERROR, ...)`.
5. At the next `CHECK_FOR_INTERRUPTS()`, the transaction is aborted cleanly.

Because all backends die before the new worker starts and shared memory is reinitialized, there is no stale state to reason about. The ring buffer, result slots, and all other shared structures are fresh when the new worker begins accepting tasks.

### Worker startup block (`vector_index_worker.c`)

```c
LWLockAcquire(ring_buffer_shmem->lock, LW_EXCLUSIVE);
ring_buffer_shmem->worker_pgprocno = MyProc->pgprocno;
elog(DEBUG1, "[vector_index_worker] started, pgprocno=%d", MyProc->pgprocno);
LWLockRelease(ring_buffer_shmem->lock);
```

Only `worker_pgprocno` is set. No epoch counter, no ring buffer reset — neither is needed because shared memory is always reinitialized before this block runs.

### Backend-side changes (`tasksend.c`)

**Vector search** (`vector_search_send` / `vector_search_get_result`):

- `vector_search_send` enqueues the task and wakes the worker. No epoch snapshot.
- `vector_search_get_result` loops on `WaitLatch` with a 1-second timeout. On each wakeup it checks `WL_POSTMASTER_DEATH` first (errors the backend) and then `WL_LATCH_SET` (result ready, breaks out of the loop). `WL_TIMEOUT` continues waiting.

**Maintenance tasks** (`index_build_blocking`, `segment_update_blocking`, `index_load_blocking`):

All three are unified behind a single `submit_and_wait_maintenance` helper that:
1. Calls `ResetLatch(MyLatch)` **before** acquiring the LWLock (so no completion signal from the worker can arrive between enqueue and `WaitLatch` and be silently cleared).
2. Enqueues the task and wakes the worker under the LWLock.
3. Waits in a 1-second timeout loop checking `WL_POSTMASTER_DEATH` (error) and `WL_LATCH_SET` (done).

---

## Feature 2 — mmap-Compatible On-Disk Index Format

### Problem with the old format

`IndexBinarySetFlush` previously called `write_binary_set`, which serialized the full Knowhere `BinarySet` as a multi-key envelope. Knowhere's `DeserializeFromFile` (the only API that supports `enable_mmap`) calls `faiss::read_index(filename)` directly and expects a file containing only the raw Faiss byte payload for one index — not the envelope.

### New format

`IndexBinarySetFlush` now:
1. Calls `get_primary_key_name(index_type)` to map the `IndexType` enum to the Knowhere `BinarySet` key name for that index type:
   - `FLAT` → `knowhere::IndexEnum::INDEX_FAISS_IDMAP`
   - `IVFFLAT` → `knowhere::IndexEnum::INDEX_FAISS_IVFFLAT`
   - `HNSW` → `knowhere::IndexEnum::INDEX_HNSW`
   - `DISKANN` → `nullptr` (falls back to `write_binary_set`; DISKANN manages its own disk format)
2. Extracts the binary blob for that key from the `BinarySet`.
3. Writes only those raw bytes to disk using `FileIOWriter`.

`IndexLoadAndSave` now calls `kindex.DeserializeFromFile(path, conf)` for FLAT/IVFFLAT/HNSW, passing `conf["enable_mmap"] = use_mmap`. The `use_mmap` flag is threaded through the call stack:

```
IndexLoadTaskType handler
  → load_all_segments_from_disk_mmap(pool)        [use_mmap=true]
  → load_and_set_segment(..., use_mmap=true)
  → load_index_file(..., use_mmap=true)
  → IndexLoadAndSave(path, index_type, indexPtr, use_mmap=true)
```

Regular (non-cold-start) paths pass `use_mmap=false` (same call chain, e.g., `load_all_segments_from_disk`, merge worker calls).

### Backward compatibility

**Existing segment files written by the old format are not readable by the new code.** The files in `VECTOR_STORAGE_BASE_DIR` (`/ssd_root/liu4127/pg_vector_extension_indexes/`) must be purged before deploying this change.

---

## Feature 3 — mmap Cold-Start

### Overview

On `IndexLoadTaskType`, the worker now performs a 2-phase load:

**Phase 1 (fast path):** Load all segments using mmap (`DeserializeFromFile` with `enable_mmap=true`). This maps index bytes directly from disk without reading them into memory. Once all segments are mmap-loaded, the backend that requested the load is signaled immediately — the index is now searchable with mmap-backed access.

**Phase 2 (background upgrade):** For each mmap-loaded segment, the worker submits an `InternalSegmentUpgradeTaskType` task to the `MaintenanceThreadPool`. Each upgrade task:
1. Loads the segment fully into memory (`use_mmap=false`).
2. Acquires the pool write lock.
3. Swaps `seg->index_ptr` to the new in-memory pointer.
4. Sets `load_state = SEG_FULLY_LOADED`.
5. Releases the write lock.
6. Frees the old mmap-backed index pointer.

### `SegmentLoadState` enum

Added to `lsm_segment.h` and stored as `atomic_int load_state` in `FlushedSegmentData`:

```c
typedef enum SegmentLoadState {
    SEG_NOT_LOADED  = 0,
    SEG_MMAP_LOADED = 1,
    SEG_FULLY_LOADED = 2,
} SegmentLoadState;
```

`initialize_segment_pool` initializes each slot to `SEG_NOT_LOADED` alongside `ref_count`.

### `InternalSegmentUpgradeTaskType`

Added to the `VectorTaskType` enum in `ringbuffer.h` with a comment that it is internal-only and never enters the ring buffer. It is submitted directly via `submit_internal_task()` to the `MaintenanceThreadPool` (no DSM, no ring buffer slot).

`MAINTENANCE_TASK_QUEUE_SIZE` was raised from 128 to 1152 to accommodate up to `MAX_SEGMENTS_COUNT=1024` pending upgrade tasks plus headroom for regular maintenance tasks.

### Compile-time flag: `ENABLE_MMAP_COLDSTART`

Defined in `lsmindex.h` alongside `IS_DISK_BASED`:

```c
#ifndef ENABLE_MMAP_COLDSTART
#define ENABLE_MMAP_COLDSTART 1
#endif
```

| Value | Behavior |
|-------|----------|
| `1` (default) | 2-phase load: mmap phase 1 → signal backend → background upgrade phase 2 |
| `0` | Direct full load: all segments fully loaded into memory before the backend is signaled |

Override at build time via `make PG_CPPFLAGS="-DENABLE_MMAP_COLDSTART=0"`.

**The on-disk format** (raw Faiss bytes written by `IndexBinarySetFlush`) **is mmap-compatible regardless of this flag.** Setting the flag to 0 simply skips the mmap loading path; `IndexLoadAndSave` is still called with `use_mmap=false` for all segments.

When `ENABLE_MMAP_COLDSTART=0`, the `IndexLoadTaskType` handler calls `load_all_segments_from_disk` (full in-memory load) and then signals the backend via the normal `client` pointer path — identical in behavior to the pre-mmap-coldstart implementation, but using the new on-disk format.

---

## Correctness Invariants for mmap Cold-Start

Three invariants were verified after implementation. One required a fix.

---

### Invariant 1 — Rebuild/merge must not run while source segments are `SEG_MMAP_LOADED`

**Status:** Fixed (was not enforced).

**Why it matters:** `replace_flushed_segment` → `cleanup_flushed_segment` calls `IndexFree(index_ptr)` on the old slot. If the old slot is still `SEG_MMAP_LOADED`, that frees the mmap-backed index while searches that snapshotted the pointer before the swap are still reading it (use-after-free). The upgrade task guard correctly discards the stale upgrade task after the slot is replaced, so the final in-memory state is correct — but the premature free is hazardous.

**Fix:** Both the `SEGMENT_UPDATE_REBUILD_FLAT`/`SEGMENT_UPDATE_REBUILD_DELETION` and `SEGMENT_UPDATE_MERGE` handlers in `vector_index_worker.c` now spin-wait (1 ms intervals, 30-second timeout with a warning) after finding the source segment(s), blocking until `atomic_load(&seg->load_state) != SEG_MMAP_LOADED` before calling `replace_flushed_segment`. The rebuild and merge themselves load from disk files — they do not read from the in-memory index pointer — so the wait is purely for safety of the old pointer's lifetime.

Note: `SEGMENT_UPDATE_REGULAR` (new flushed segment) is unaffected — it creates a new slot, not replacing an existing mmap one.

---

### Invariant 2 — Vacuum must not cause inconsistencies during segment loading

**Status:** Satisfied. No change needed.

**Why it is safe:**

- `bitmap_ptr` is always heap-allocated regardless of `load_state`. `load_and_set_segment` always calls `load_bitmap_file(..., use_mmap=false)` even during the mmap cold-start path. Only the index file (`index_ptr`) is conditionally mmap-backed.
- The `SEGMENT_UPDATE_VACUUM` handler merges new deletion marks into `bitmap_ptr`. It does not touch `index_ptr` or `load_state`.
- The `InternalSegmentUpgradeTaskType` handler swaps only `index_ptr` and `load_state`. It does not touch `bitmap_ptr`.
- These two operations can therefore run concurrently on the same segment without conflict.
- Deletion marks applied by vacuum during the mmap phase survive the upgrade intact: after the upgrade swaps `index_ptr`, the segment's `bitmap_ptr` is unchanged and reflects all deletions.

Note: there is a pre-existing race between `SEGMENT_UPDATE_VACUUM` and `replace_flushed_segment` (a concurrent rebuild/merge can free `bitmap_ptr` via `cleanup_flushed_segment` while vacuum is still reading it outside the lock), but this predates the mmap cold-start feature.

---

### Invariant 3 — `use_mmap=true` is used only for cold-start loading

**Status:** Satisfied. No change needed.

`use_mmap=true` is passed to `load_and_set_segment` only from `load_all_segments_from_disk_mmap`, which is called exclusively from the `IndexLoadTaskType` handler. `IndexLoadTaskType` is submitted only by `index_load_blocking`, which is called only from `load_lsm_index`, which fires when a backend first accesses an index whose pool is not yet initialized (i.e., after a worker start or restart).

All other loading paths use `use_mmap=false`:

| Path | `use_mmap` |
|------|-----------|
| `IndexBuildTaskType` handler | `false` |
| `SEGMENT_UPDATE_REGULAR` handler | `false` |
| `SEGMENT_UPDATE_REBUILD_FLAT/DELETION` handler | `false` |
| `SEGMENT_UPDATE_MERGE` handler | `false` |
| `load_all_segments_from_disk` (non-cold-start) | `false` |
| Merge worker (`lsm_merge_worker.c`) | `false` |

---

## Bugs Fixed During Quality Review (C1–C4)

| ID | File | Fix |
|----|------|-----|
| C1 | `lsm_segment.c` | Added `continue` after `reserve_flushed_segment` returns `(uint32_t)-1` in `load_all_segments_from_disk`. Previously fell through to `pool->flushed_segments[0xFFFFFFFF]` — an out-of-bounds array access. |
| C2 | `tasksend.c` | Moved `ResetLatch(MyLatch)` to before `LWLockAcquire` in `submit_and_wait_maintenance`. Previously the reset was inside the lock, creating a window where the worker could complete and signal the backend before the latch was cleared, causing a spurious 1-second wait. |
| C3 | `lsm_segment.c` | Added `atomic_store(&seg->load_state, SEG_NOT_LOADED)` to `initialize_segment_pool`. Without this, `load_state` was uninitialized for slots that failed metadata loading, causing the upgrade phase-2 check to read garbage. |

---

## Remaining Known Issues (Not Yet Fixed)

### C5 — mmap swap bypasses ref-count: potential use-after-free in concurrent search

**File:** `vector_index_worker.c` (`InternalSegmentUpgradeTaskType` handler)  
**Severity:** Critical (use-after-free)

`vector_search` snapshots `seg->index_ptr` under a read lock, releases the lock, then passes the pointer to `ConcurrentVectorSearchOnSegments` outside any lock. The upgrade handler swaps `index_ptr` and then calls `IndexFree(old_index_ptr)` after releasing the write lock. A concurrent search that snapshotted the old pointer before the swap will dereference freed mmap-backed memory.

The fix requires the upgrade path to go through the existing ref-count/cleanup system (`increment_flushed_segment_ref_count` / `cleanup_flushed_segment`) rather than calling `IndexFree` directly, so the old mmap pointer is only freed after all in-flight searches on it complete.

---

### I1 — `InternalSegmentUpgradeTaskType` not guarded in ring-buffer dispatch

**File:** `vector_index_worker.c` (`handle_task` switch)  
**Severity:** Important (silent bug if invariant violated)

`handle_task` dispatches ring-buffer task types but has only `default: break` — a corrupted slot with `type == InternalSegmentUpgradeTaskType` would be silently swallowed. An explicit `case InternalSegmentUpgradeTaskType: elog(ERROR, ...)` makes the "never enters the ring buffer" contract machine-checked.

---

### I2 — `get_primary_key_name` conflates DISKANN and unknown index types

**File:** `vectorindeximpl.cpp`  
**Severity:** Important (unknown types silently call `write_binary_set`)

Both `DISKANN` and any unrecognized `IndexType` value hit `default: return nullptr`, causing `IndexBinarySetFlush` to fall back to `write_binary_set` for truly unknown types rather than reporting an error. The `default` case should call `elog(ERROR, ...)` and `DISKANN` should be an explicit `case`.

---

### I3 — `load_all_segments_from_disk` and `load_all_segments_from_disk_mmap` are near-identical (DRY)

**File:** `lsm_segment.c`  
**Severity:** Important (maintenance burden)

The two functions differ only in the `use_mmap` argument passed to `load_and_set_segment`. Should collapse into one `static` internal function with a `bool use_mmap` parameter, with both public functions as thin wrappers.

---

### I4 — `IndexLoadAndSave` has three identical switch arms (FLAT/IVFFLAT/HNSW)

**File:** `vectorindeximpl.cpp`  
**Severity:** Important (DRY, `get_primary_key_name` is not the single source of truth)

The FLAT, IVFFLAT, and HNSW cases in the `switch` over `index_type` are structurally identical — only the string constant differs, and `get_primary_key_name` already provides that mapping. Merge into one `case FLAT: case IVFFLAT: case HNSW:` block using `get_primary_key_name`.

---

### I5 — Enqueue logic duplicated in `submit_maintenance_task` and `submit_internal_task`

**File:** `vector_index_worker.c`  
**Severity:** Important (DRY)

Both functions end with the same mutex-lock / tail-append / `atomic_fetch_add` / `cond_signal` / unlock sequence. Extract to a shared `enqueue_maintenance_task(MaintenanceTask*)` helper.

---

### I6 — `register_flushed_segment` called unconditionally when `load_and_set_segment` fails

**File:** `lsm_segment.c`  
**Severity:** Important (null dereference in search)

If `read_lsm_segment_metadata` returns false inside `load_and_set_segment`, the segment slot remains `in_used=true` (set by `reserve_flushed_segment`) with `index_ptr=NULL`, `bitmap_ptr=NULL`, and `map_ptr=NULL`. `register_flushed_segment` is still called, adding this slot to the linked list. Subsequent searches will dereference null pointers. `load_and_set_segment` should return a status and callers should skip `register_flushed_segment` on failure.

---

### I7 — Dead commented-out `IndexDeserializeAndSave` function

**File:** `vectorindeximpl.cpp`  
**Severity:** Low (dead code)

~35 lines of commented-out code superseded by `IndexLoadAndSave`. References globals that no longer exist. Should be deleted.
