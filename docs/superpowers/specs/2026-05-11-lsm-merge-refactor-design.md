# LSM Merge Worker Refactoring Design

**Date:** 2026-05-11  
**Status:** Approved for implementation

---

## 1. Motivation

The current architecture has two separate PostgreSQL background processes managing index segment metadata:

- **LSM Merge Worker** (×2): owns `SharedSegmentArray` in shared memory, scans for merge/rebuild candidates, and notifies the Vector Index Worker via the ring buffer when a task completes.
- **Vector Index Worker**: owns `FlushedSegmentPool` in process-local memory, handles search and segment updates.

Because both processes track overlapping segment state, they must be kept in sync through shared memory, LWLocks, and ring buffer round-trips. This adds synchronisation overhead and makes the code difficult to reason about.

The goal is to make the Vector Index Worker the **single owner of all segment metadata** by moving merge and rebuild logic into dedicated threads inside it.

---

## 2. Architecture Overview

### What is removed

- `lsm_merge_worker.c` and `lsm_merge_worker.h` — deleted entirely
- `SharedSegmentArray`, `MergeWorkerManager`, `MergeSegmentInfo`, `MergeTaskData`, `MergeWorkerState` structs
- `merge_worker_manager` global pointer
- `LSM_MERGE_SEGMENT_BITMAP_LWTRANCHE` and `"LSM Merge Worker"` LWLock tranches and their `RequestNamedLWLockTranche` / `RequestAddinShmemSpace` calls in `_PG_init`
- `MERGE_WORKERS_COUNT` background worker registration loop in `vector.c`
- `add_to_segment_array()` call in `lsmbackground.c`
- `initialize_merge_worker_manager()` call in `shmem_startup`
- `#include "lsm_merge_worker.h"` from `vector.c` and `lsmbackground.c`

### What is added

- `MergeThreadPool` — dedicated pthread pool (`MERGE_WORKERS_COUNT = 2` threads) inside the Vector Index Worker
- New fields in `FlushedSegmentData`: `delete_count`, `is_compacting`, `version`, `per_seg_mutex`
- New aggregate statistics atomics in `FlushedSegmentPool`: `flat_count`, `memtable_capacity_le_count`, `small_segment_le_count`
- `expected_version` field in `SegmentUpdateTaskData` (meaningful only for `SEGMENT_UPDATE_VACUUM`)
- Retry logic in `bulk_delete_lsm_index` for the version-mismatch and segment-not-found cases
- `MERGE_WORKERS_COUNT` and merge scheduling thresholds move to `vector_index_worker.h`

### Process topology

Three OS processes remain after the refactoring.

**PostgreSQL backends** run `bulk_delete_lsm_index` during vacuum. They enumerate segments by scanning disk via `scan_segment_metadata_files()`, compute deletions, write bitmap subversions, then send `SEGMENT_UPDATE_VACUUM` through the ring buffer and block.

**LSM Background Worker** (`lsm_index_bgworker_main`) is unchanged. It flushes full memtables to disk and sends `SEGMENT_UPDATE_REGULAR` through the ring buffer. The `add_to_segment_array()` call is simply removed — the flush notification to VIW already carries all necessary information.

**Vector Index Worker** (`vector_index_worker_main`) becomes the single owner of all segment metadata. It contains three internal pthread components:

- **Main event loop**: reads tasks off the ring buffer and dispatches them — search tasks handled inline, all other task types submitted to `MaintenanceThreadPool`.
- **MaintenanceThreadPool** (existing, 4 threads): handles `SEGMENT_UPDATE_REGULAR` (add segment to `FlushedSegmentPool`, signal `MergeThreadPool`), `SEGMENT_UPDATE_VACUUM` (acquire `per_seg_mutex`, version check, reload bitmap), `IndexLoad`, `IndexBuild`, and mmap upgrades (`InternalSegmentUpgradeTaskType`). `SEGMENT_UPDATE_REBUILD` and `SEGMENT_UPDATE_MERGE` no longer arrive via the ring buffer.
- **MergeThreadPool** (new, 2 threads): sleeps on a condition variable, scans `FlushedSegmentPool` for candidates, executes merge/rebuild, and updates `FlushedSegmentPool` directly — no ring buffer call needed.

### Key communication changes

| Path | Before | After |
|------|---------|-------|
| Merge completion → VIW | Ring buffer (`segment_update_blocking`) | Direct `FlushedSegmentPool` update (same process) |
| New segment → merge threads | `add_to_segment_array()` via shared memory | Condition variable signal inside VIW |
| Vacuum ↔ merge synchronisation | Shared-memory `bitmap_lock` (LWLock) | In-process `per_seg_mutex` (pthread_mutex_t) + retry protocol |

---

## 3. Data Structure Changes

### `FlushedSegmentData` — four fields added (`lsm_segment.h`)

```c
typedef struct FlushedSegmentData {
    /* existing fields unchanged */
    bool in_used;
    SegmentId segment_id_start;
    SegmentId segment_id_end;
    Size vec_count;
    int64_t *map_ptr;
    IndexType index_type;
    void *index_ptr;
    uint8_t *bitmap_ptr;
    uint32_t next_idx;
    uint32_t prev_idx;
    atomic_int ref_count;
    atomic_int load_state;

    /* new */
    uint32_t delete_count;         // deletion ratio scheduling; updated under seg_lock write
    bool is_compacting;            // claimed by a merge thread; set/cleared under seg_lock write
    uint32_t version;              // current on-disk version; used by SEGMENT_UPDATE_VACUUM check
    pthread_mutex_t per_seg_mutex; // vacuum-merge bitmap coordination (see Section 6)
} FlushedSegmentData;
```

`version` is set when a segment is first loaded (`find_latest_segment_version`) and incremented each time a merge or rebuild writes a new segment version to disk.

### `FlushedSegmentPool` — three aggregate counters added (`lsm_segment.h`)

```c
typedef struct FlushedSegmentPool {
    /* existing fields unchanged */
    pthread_rwlock_t seg_lock;
    FlushedSegmentData flushed_segments[MAX_SEGMENTS_COUNT];
    uint32_t flushed_segment_count;
    uint32_t head_idx;
    uint32_t tail_idx;
    uint32_t insert_idx;

    /* new */
    pg_atomic_uint32 flat_count;
    pg_atomic_uint32 memtable_capacity_le_count;
    pg_atomic_uint32 small_segment_le_count;
} FlushedSegmentPool;
```

These atomics allow merge threads to skip a full pool scan when there is obviously no work, preserving the fast-path scheduling behaviour of the current merge workers.

### `SegmentUpdateTaskData` — one field added (`ringbuffer.h`)

```c
typedef struct {
    int backend_pgprocno;
    Oid index_relid;
    int lsm_idx;
    int operation_type;
    SegmentId start_sid;
    SegmentId end_sid;
    uint32_t expected_version;  // new; meaningful only for SEGMENT_UPDATE_VACUUM
} SegmentUpdateTaskData;
```

### `MergeThreadPool` — new struct (`vector_index_worker.c`)

Unlike `MaintenanceThreadPool`, merge threads do not consume a task queue — they scan `FlushedSegmentPool` themselves. The pool only needs a condition variable for wakeup signals.

```c
typedef struct MergeThreadPool {
    pthread_t threads[MERGE_WORKERS_COUNT];
    pthread_mutex_t mutex;
    pthread_cond_t work_available;
    atomic_int shutdown;
} MergeThreadPool;
```

---

## 4. Merge Thread Pool and Scheduling

### Initialisation

`init_merge_thread_pool()` is called at the start of `vector_index_worker_main` alongside `init_maintenance_thread_pool()`. It initialises the mutex, condition variable, and spawns `MERGE_WORKERS_COUNT` threads each running `merge_worker_thread()`.

### Wakeup triggers

Merge threads sleep on `MergeThreadPool.work_available`. Four events signal it:

1. A maintenance thread completes `SEGMENT_UPDATE_REGULAR` — a new segment entered `FlushedSegmentPool`
2. A maintenance thread completes `IndexLoad` — cold-start segments are available
3. A maintenance thread completes `InternalSegmentUpgradeTaskType` — a segment transitions from `SEG_MMAP_LOADED` to `SEG_FULLY_LOADED`
4. A merge thread finishes a task — may satisfy merge conditions on a neighbouring segment

### Precondition: `SEG_FULLY_LOADED`

Merge and rebuild operate only on segments with `load_state == SEG_FULLY_LOADED`. Segments in `SEG_NOT_LOADED` or `SEG_MMAP_LOADED` are skipped during candidate scanning. This avoids busy-waiting on mmap-upgrading segments and prevents operating on partially-loaded index data.

### Task claim

Each merge thread wakes and runs the same 5-priority scan as `scan_and_claim_merge_task()` today, against `FlushedSegmentPool` instead of `SharedSegmentArray`. Fast-path atomics are checked first. Candidate conditions additionally require `!segment->is_compacting` and `segment->load_state == SEG_FULLY_LOADED`.

Claiming a candidate:
1. Acquire `seg_lock` (read) — find candidate satisfying priority criteria
2. Release `seg_lock`
3. Acquire `seg_lock` (write) — re-verify candidate still valid; set `is_compacting = true`
4. Release `seg_lock`

If the candidate was taken between steps 2 and 3, retry from the same priority level.

### Task execution — rebuild (`rebuild_index`)

1. Load bitmap, mapping, and index from disk — no lock held (long I/O step)
2. Run `MergeIndex` to build the new index from the snapshot bitmap
3. Acquire `per_seg_mutex`
4. Read fresh `FlushedSegmentData.bitmap_ptr` — captures any vacuum updates that landed during step 2
5. Construct the new segment's bitmap from the fresh in-memory state
6. Write new segment files to disk (new version number)
7. Acquire `seg_lock` (write): swap `FlushedSegmentData` fields (`index_ptr`, `bitmap_ptr`, `map_ptr`, `vec_count`, `index_type`, `delete_count`, `version`), free old pointers, clear `is_compacting`, update pool statistics atomics
8. Release `seg_lock`
9. Release `per_seg_mutex`
10. Signal `MergeThreadPool.work_available`

### Task execution — merge adjacent (`merge_adjacent_segments`)

Same structure as rebuild, with two segments:

- Step 3: acquire both `per_seg_mutex` locks in ascending `start_sid` order (consistent ordering prevents deadlock)
- Step 7: under `seg_lock` write — update the first `FlushedSegmentData` to represent the merged segment (expanded `segment_id_end`, updated counts); remove the second entry via `cleanup_flushed_segment`

### Memory allocation

All `palloc`/`pfree` in `rebuild_index` and `merge_adjacent_segments` become `malloc`/`free`. Merge threads run without a PostgreSQL memory context; the existing maintenance threads already follow this convention.

### Eliminated paths

`finish_merge_task()` and its `segment_update_blocking()` call are removed entirely. Merge threads update `FlushedSegmentPool` directly.

---

## 5. Vacuum Changes

### `bulk_delete_lsm_index` — flushed segments step

The `SharedSegmentArray` traversal is replaced with a disk-based scan:

```
1. scan_segment_metadata_files(indexRelId, files, MAX_SEGMENTS_COUNT)
   → flat array of (start_sid, end_sid, version) for all on-disk segments

2. For each file entry:
   a. read_lsm_segment_metadata(...) → get vec_count, index_type
      skip if file cannot be read (disappeared between scan and read)
   b. skip if (start_sid == end_sid) and in vacuumed_memtable_ids list
   c. Load bitmap + mapping from disk at the recorded version
   d. Walk vec_count positions; apply callback; mark deletions
   e. If bitmap changed:
        write_bitmap_file_with_subversion(...)
        result = segment_update_blocking(..., SEGMENT_UPDATE_VACUUM,
                     start_sid, end_sid, expected_version = version)
        if result == RETRY:
            re-scan disk; find segment where file.start_sid <= start_sid <= file.end_sid
            if found with new (start_sid', end_sid', version'):
                go to step (c) for the covering segment
```

No `bitmap_lock`, no shared-memory traversal, no `in_used` check.

### `SEGMENT_UPDATE_VACUUM` handler (inside `MaintenanceThreadPool`)

```
1. Acquire seg_lock (read) — find segment by exact (start_sid, end_sid); increment ref_count
2. Release seg_lock
3. If not found → signal backend RETRY; return
4. Acquire per_seg_mutex
5. Check segment->version == task->expected_version:
     Match   → load latest bitmap subversion from disk
               OR new bits into FlushedSegmentData.bitmap_ptr
               update segment->delete_count
               release per_seg_mutex; decrement ref_count
               signal backend: OK
     Mismatch → release per_seg_mutex; decrement ref_count
               signal backend: RETRY
```

---

## 6. Concurrency Model

### Locks and their scopes

| Lock | Type | Guards |
|------|------|--------|
| `FlushedSegmentPool.seg_lock` | `pthread_rwlock_t` | Linked-list structure, `in_used`, `is_compacting`, statistics atomics |
| `FlushedSegmentData.per_seg_mutex` | `pthread_mutex_t` | `bitmap_ptr` consistency between vacuum maintenance thread and merge thread; `version` field |
| `FlushedSegmentData.ref_count` | `atomic_int` | Prevents segment memory from being freed while a pointer is held |
| `MergeThreadPool.mutex` | `pthread_mutex_t` | `work_available` condition variable |

### Lock ordering — no circular wait

- **Maintenance thread (vacuum)**: `seg_lock` (read) → find segment, increment ref_count → **release `seg_lock`** → `per_seg_mutex`. Never holds `seg_lock` while blocking on `per_seg_mutex`.
- **Merge thread (finalisation)**: `per_seg_mutex` → `seg_lock` (write, brief swap) → release `seg_lock` → release `per_seg_mutex`.

No circular wait exists: maintenance threads always release `seg_lock` before attempting `per_seg_mutex`.

**Two-segment merge**: both `per_seg_mutex` locks acquired in ascending `start_sid` order.

### Correctness of the two bitmap races

**Race 1 — vacuum maintenance thread wins `per_seg_mutex` first:**
Maintenance thread ORs deletions D into `bitmap_ptr`, releases mutex. Merge thread acquires mutex, reads fresh `bitmap_ptr` = B₀ ∪ D. New segment embeds all deletions. ✓

**Race 2 — merge thread wins `per_seg_mutex` first:**
Merge thread reads B₀ (vacuum not landed yet), writes new segment version V+1, swaps `FlushedSegmentData`, releases mutex. Maintenance thread acquires mutex, detects version mismatch V+1 ≠ V, signals RETRY. Backend re-reads version V+1 mapping and bitmap, recomputes deletions in new position space, writes subversion of V+1, resends with `expected_version = V+1`. On retry, version matches. ✓

### RETRY semantics — two cases

| Case | Detection | Retry action |
|------|-----------|--------------|
| Segment rebuilt in place (same sid range, new version) | Version mismatch at step 5 | Re-read same `(start_sid, end_sid)` at new version; recompute deletions |
| Segment merged away (different sid range) | Not found at step 3 | Re-scan disk; find file covering `start_sid`; process that segment |

The merged-away case arises when merge completes entirely — including its `per_seg_mutex`-protected bitmap reload — before vacuum generates D_A at all. Since D_A does not yet exist on disk or in memory when merge reads the bitmap, `per_seg_mutex` cannot prevent this. RETRY + re-scan is the sole recovery mechanism and is correct.

Re-processing positions from a previously-vacuumed adjacent segment is idempotent (setting an already-set bitmap bit has no effect).

---

## 7. Files Changed

| File | Change |
|------|--------|
| `pgvector/src/lsm_merge_worker.c` | Deleted |
| `pgvector/src/lsm_merge_worker.h` | Deleted |
| `pgvector/src/lsm_segment.h` | Add `delete_count`, `is_compacting`, `version`, `per_seg_mutex` to `FlushedSegmentData`; add three atomic counters to `FlushedSegmentPool` |
| `pgvector/src/lsm_segment.c` | Initialise new fields in `initialize_segment_pool`; update `register_flushed_segment`, `replace_flushed_segment`, `cleanup_flushed_segment` to maintain statistics and `per_seg_mutex` |
| `pgvector/src/ringbuffer.h` | Add `expected_version` to `SegmentUpdateTaskData` |
| `pgvector/src/vector_index_worker.h` | Add `MERGE_WORKERS_COUNT` and merge scheduling thresholds |
| `pgvector/src/vector_index_worker.c` | Add `MergeThreadPool`; `merge_worker_thread()` with scan-claim-execute loop; `init_merge_thread_pool()`; signal merge pool from maintenance threads on `SEGMENT_UPDATE_REGULAR`, `IndexLoad`, and `InternalSegmentUpgradeTaskType` completion; update `SEGMENT_UPDATE_VACUUM` handler with `per_seg_mutex` and version check |
| `pgvector/src/lsmbackground.c` | Remove `add_to_segment_array()` call; remove `#include "lsm_merge_worker.h"` |
| `pgvector/src/lsmindex.c` | Replace `SharedSegmentArray` traversal in `bulk_delete_lsm_index` with `scan_segment_metadata_files()`; add retry logic; pass `expected_version` in `segment_update_blocking` |
| `pgvector/src/vector.c` | Remove merge worker registration loop; remove LWLock tranche registrations for merge worker and bitmap locks; remove `RequestAddinShmemSpace` for `MergeWorkerManager`; remove `initialize_merge_worker_manager()` from `shmem_startup`; remove `#include "lsm_merge_worker.h"` |
| `pgvector/Makefile` | Remove `src/lsm_merge_worker.o` from `OBJS` |
