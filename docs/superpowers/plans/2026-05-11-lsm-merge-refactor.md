# LSM Merge Worker Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the standalone LSM Merge Worker background processes and move all merge/rebuild logic into dedicated threads inside the Vector Index Worker, making VIW the single owner of all index segment metadata.

**Architecture:** A new `MergeThreadPool` (2 pthreads) is added inside `vector_index_worker_main`. Merge metadata (`delete_count`, `is_compacting`, `version`, `per_seg_mutex`) moves from shared-memory `SharedSegmentArray` into `FlushedSegmentData`. Vacuum coordination uses a per-segment `pthread_mutex_t` and a RETRY protocol instead of a shared-memory `LWLock`.

**Tech Stack:** C17, PostgreSQL background worker API, pthreads, PostgreSQL LWLocks (for remaining shared-memory structures), pg_atomic primitives.

**Spec:** `docs/superpowers/specs/2026-05-11-lsm-merge-refactor-design.md`

---

## File Map

| File | Action | Summary |
|------|--------|---------|
| `pgvector/src/lsm_segment.h` | Modify | Add 4 fields to `FlushedSegmentData`; add 3 atomic counters to `FlushedSegmentPool` |
| `pgvector/src/lsm_segment.c` | Modify | Init/manage new fields in pool functions |
| `pgvector/src/ringbuffer.h` | Modify | Add `expected_version` to `SegmentUpdateTaskData`; add `maint_status` to `VectorSearchResultData` |
| `pgvector/src/tasksend.h` | Modify | Change `segment_update_blocking` signature to accept `expected_version`, return `int` |
| `pgvector/src/tasksend.c` | Modify | Implement new signature; read `maint_status` after waking |
| `pgvector/src/vector_index_worker.h` | Modify | Move merge constants here from `lsm_merge_worker.h` |
| `pgvector/src/vector_index_worker.c` | Modify | Add `MergeThreadPool`; port merge scheduling, `rebuild_index`, `merge_adjacent_segments`; add `merge_worker_thread`; update VACUUM handler; signal merge pool |
| `pgvector/src/lsmbackground.c` | Modify | Remove `add_to_segment_array()` call and include |
| `pgvector/src/lsmindex.c` | Modify | Replace `SharedSegmentArray` traversal in `bulk_delete_lsm_index` with disk scan; add retry |
| `pgvector/src/vector.c` | Modify | Remove merge worker registrations, LWLock tranches, shmem allocation |
| `pgvector/Makefile` | Modify | Remove `src/lsm_merge_worker.o` from `OBJS` |
| `pgvector/src/lsm_merge_worker.c` | Delete | |
| `pgvector/src/lsm_merge_worker.h` | Delete | |

---

## Task 1: Update data structure headers

**Files:**
- Modify: `pgvector/src/lsm_segment.h`
- Modify: `pgvector/src/ringbuffer.h`
- Modify: `pgvector/src/vector_index_worker.h`

- [ ] **Step 1: Add merge metadata fields to `FlushedSegmentData` in `lsm_segment.h`**

Add `#include <pthread.h>` at the top if not already present. Add four new fields at the end of `FlushedSegmentData`, before the closing `}`:

```c
    // Reference counting for safe segment deletion during merging
    atomic_int ref_count;

    // Load state: SEG_NOT_LOADED, SEG_MMAP_LOADED, or SEG_FULLY_LOADED
    atomic_int load_state;

    /* NEW: merge metadata (formerly in SharedSegmentArray) */
    uint32_t delete_count;         /* # deleted vectors; updated under seg_lock write */
    bool is_compacting;            /* claimed by a merge thread; set/cleared under seg_lock write */
    uint32_t version;              /* current on-disk version; used by SEGMENT_UPDATE_VACUUM */
    pthread_mutex_t per_seg_mutex; /* vacuum-merge bitmap coordination */
```

- [ ] **Step 2: Add aggregate statistics atomics to `FlushedSegmentPool` in `lsm_segment.h`**

Add three atomic counters at the end of `FlushedSegmentPool`, before the closing `}`:

```c
    uint32_t insert_idx;

    /* NEW: aggregate statistics for merge scheduling fast-path */
    pg_atomic_uint32 flat_count;
    pg_atomic_uint32 memtable_capacity_le_count;
    pg_atomic_uint32 small_segment_le_count;
```

- [ ] **Step 3: Add `expected_version` to `SegmentUpdateTaskData` in `ringbuffer.h`**

```c
typedef struct {
    int backend_pgprocno;
    Oid index_relid;
    int lsm_idx;
    int operation_type;
    SegmentId start_sid;
    SegmentId end_sid;
    uint32_t expected_version;  /* NEW: used only by SEGMENT_UPDATE_VACUUM */
} SegmentUpdateTaskData;
```

- [ ] **Step 4: Add `maint_status` to `VectorSearchResultData` in `ringbuffer.h`**

This field lets the VIW communicate OK vs. RETRY to the backend after `SEGMENT_UPDATE_VACUUM`. It is written by the maintenance thread before signalling the backend's latch, and read by the backend in `segment_update_blocking`.

```c
typedef struct {
    volatile int status;       /* search result: 0=empty, 1=done, 2=error */
    volatile int result_count;
    Size result_size;
    volatile int maint_status; /* NEW: maintenance result: 0=OK, 1=RETRY */
} VectorSearchResultData;
```

- [ ] **Step 5: Move merge constants to `vector_index_worker.h`**

Add to `vector_index_worker.h` (these will be removed from `lsm_merge_worker.h` in Task 13):

```c
/* Merge scheduling constants (formerly in lsm_merge_worker.h) */
#define MERGE_WORKERS_COUNT              2
#define MERGE_DELETION_RATIO_THRESHOLD   0.3f
#define MAX_SEGMENTS_SIZE                5000000
#define THRESHOLD_SMALL_SEGMENT_SIZE     1000000
```

- [ ] **Step 6: Compile check**

```bash
cd /home/liu4127/postgresql/decoupled_pgvector/pgvector && make 2>&1 | head -40
```

Expected: compile errors in `lsm_segment.c` and `lsm_merge_worker.c` about missing fields/symbols — these will be fixed in subsequent tasks. The goal here is to confirm no syntax errors in the headers themselves.

---

## Task 2: Update `tasksend.h` and `tasksend.c`

**Files:**
- Modify: `pgvector/src/tasksend.h`
- Modify: `pgvector/src/tasksend.c`

- [ ] **Step 1: Update `segment_update_blocking` signature in `tasksend.h`**

Change the declaration to accept `expected_version` and return `int` (0=OK, 1=RETRY):

```c
int segment_update_blocking(int lsm_index_idx, Oid index_relid, int operation_type,
                             SegmentId start_sid, SegmentId end_sid,
                             uint32_t expected_version);
```

- [ ] **Step 2: Update `segment_update_blocking` implementation in `tasksend.c`**

Replace the existing `segment_update_blocking` function:

```c
int
segment_update_blocking(int lsm_index_idx, Oid index_relid, int operation_type,
                         SegmentId start_sid, SegmentId end_sid,
                         uint32_t expected_version)
{
    Size task_seg_size = sizeof(SegmentUpdateTaskData);
    dsm_segment *task_seg;
    SegmentUpdateTask task;

    if (ring_buffer_shmem == NULL)
        ereport(ERROR, (errmsg("[segment_update_blocking] vector search shmem not initialized")));

    task_seg = dsm_create(task_seg_size, 0);
    if (task_seg == NULL)
        elog(ERROR, "[segment_update_blocking] Failed to allocate DSM segment");

    task = (SegmentUpdateTask) dsm_segment_address(task_seg);
    task->backend_pgprocno = MyProc->pgprocno;
    task->index_relid = index_relid;
    task->lsm_idx = lsm_index_idx;
    task->operation_type = operation_type;
    task->start_sid = start_sid;
    task->end_sid = end_sid;
    task->expected_version = expected_version;

    /* Clear maint_status before submitting so a stale value cannot be misread */
    vs_search_result_at(MyProc->pgprocno)->maint_status = 0;

    submit_and_wait_maintenance(SegmentUpdateTaskType, dsm_segment_handle(task_seg),
                                task_seg_size, "segment_update_blocking");

    int result = vs_search_result_at(MyProc->pgprocno)->maint_status;

    dsm_detach(task_seg);
    elog(DEBUG1, "[segment_update_blocking] completed, result=%d", result);
    return result;
}
```

- [ ] **Step 3: Fix all existing callers of `segment_update_blocking`**

The callers that do not need the RETRY result should pass `0` as `expected_version` and cast the return value away. In `pgvector/src/lsmbackground.c`:

```c
/* Was: segment_update_blocking(slot_idx, lsm->indexRelId, SEGMENT_UPDATE_REGULAR, prep.start_sid, prep.end_sid); */
(void) segment_update_blocking(slot_idx, lsm->indexRelId, SEGMENT_UPDATE_REGULAR,
                                prep.start_sid, prep.end_sid, 0);
```

In `pgvector/src/lsmindex.c`, the two non-vacuum callers (index build and memtable vacuum) also pass `0` and discard the result:

```c
(void) segment_update_blocking(lsm_idx, indexRelId, SEGMENT_UPDATE_VACUUM,
                                start_sid_disk, end_sid_disk, 0);
```

(The main vacuum path for flushed segments will be handled separately in Task 11.)

- [ ] **Step 4: Compile check**

```bash
cd /home/liu4127/postgresql/decoupled_pgvector/pgvector && make 2>&1 | grep -E "error:|warning:" | head -30
```

Expected: errors only in `lsm_merge_worker.c` (uses old `segment_update_blocking` signature from `finish_merge_task`) and `lsm_segment.c` (new fields not initialised yet). All other files should compile cleanly.

---

## Task 3: Update `lsm_segment.c` — initialise and manage new fields

**Files:**
- Modify: `pgvector/src/lsm_segment.c`

- [ ] **Step 1: Add `#include <pthread.h>` if not already present**

At the top of `lsm_segment.c`, ensure:
```c
#include <pthread.h>
```

- [ ] **Step 2: Update `initialize_segment_pool` to initialise new fields**

In `initialize_segment_pool`, after the existing per-slot initialisation loop, add initialisation of the new pool-level atomics and per-segment fields:

```c
void
initialize_segment_pool(FlushedSegmentPool *pool)
{
    pool->seg_lock = (pthread_rwlock_t) PTHREAD_RWLOCK_INITIALIZER;
    pool->flushed_segment_count = 0;
    pool->head_idx = (uint32_t)-1;
    pool->tail_idx = (uint32_t)-1;
    pool->insert_idx = 0;

    /* NEW: initialise aggregate statistics */
    pg_atomic_init_u32(&pool->flat_count, 0);
    pg_atomic_init_u32(&pool->memtable_capacity_le_count, 0);
    pg_atomic_init_u32(&pool->small_segment_le_count, 0);

    for (int i = 0; i < MAX_SEGMENTS_COUNT; i++)
    {
        pool->flushed_segments[i].in_used = false;
        pool->flushed_segments[i].ref_count = 1; /* base reference */
        atomic_store(&pool->flushed_segments[i].load_state, (int)SEG_NOT_LOADED);

        /* NEW fields */
        pool->flushed_segments[i].delete_count = 0;
        pool->flushed_segments[i].is_compacting = false;
        pool->flushed_segments[i].version = 0;
        pthread_mutex_init(&pool->flushed_segments[i].per_seg_mutex, NULL);
    }
}
```

- [ ] **Step 3: Add a static helper to update statistics atomics**

Add this helper just before `register_flushed_segment`:

```c
/* Call with the segment's index_type and vec_count; delta is +1 or -1 */
static void
update_pool_stats(FlushedSegmentPool *pool, IndexType index_type, uint32_t vec_count, int delta)
{
    if (index_type == FLAT)
        pg_atomic_fetch_add_u32(&pool->flat_count, (uint32_t)delta);
    if (vec_count <= MEMTABLE_MAX_CAPACITY)
        pg_atomic_fetch_add_u32(&pool->memtable_capacity_le_count, (uint32_t)delta);
    if (vec_count <= THRESHOLD_SMALL_SEGMENT_SIZE)
        pg_atomic_fetch_add_u32(&pool->small_segment_le_count, (uint32_t)delta);
}
```

Note: `MEMTABLE_MAX_CAPACITY` is in `lsmindex.h` and `THRESHOLD_SMALL_SEGMENT_SIZE` will be in `vector_index_worker.h` after Task 1. Add `#include "vector_index_worker.h"` to `lsm_segment.c`.

- [ ] **Step 4: Update `register_flushed_segment` to maintain statistics**

After the existing code that links the segment into the list, add:

```c
    /* NEW: update aggregate statistics */
    update_pool_stats(pool,
                      pool->flushed_segments[idx].index_type,
                      pool->flushed_segments[idx].vec_count,
                      +1);
```

- [ ] **Step 5: Update `cleanup_flushed_segment` to maintain statistics and destroy mutex**

Before marking the segment as unused, add:

```c
    /* NEW: update aggregate statistics */
    update_pool_stats(pool,
                      pool->flushed_segments[idx].index_type,
                      pool->flushed_segments[idx].vec_count,
                      -1);
    pthread_mutex_destroy(&pool->flushed_segments[idx].per_seg_mutex);
    pthread_mutex_init(&pool->flushed_segments[idx].per_seg_mutex, NULL);
```

- [ ] **Step 6: Update `replace_flushed_segment` to maintain statistics**

`replace_flushed_segment` removes one or two old segments and inserts a new one. After the removal of old segments and before returning, add stat updates:

```c
    /* NEW: update stats for removed old segments */
    update_pool_stats(pool, pool->flushed_segments[old_seg_idx_0].index_type,
                      pool->flushed_segments[old_seg_idx_0].vec_count, -1);
    if (old_seg_idx_1 != (uint32_t)-1)
        update_pool_stats(pool, pool->flushed_segments[old_seg_idx_1].index_type,
                          pool->flushed_segments[old_seg_idx_1].vec_count, -1);
    /* stats for new segment are added by register_flushed_segment called after this */
```

- [ ] **Step 7: Compile check**

```bash
cd /home/liu4127/postgresql/decoupled_pgvector/pgvector && make 2>&1 | grep "error:" | head -20
```

Expected: errors only in `lsm_merge_worker.c` and the VIW file where old code still references removed structures.

---

## Task 4: Add `MergeThreadPool` and initialisation to `vector_index_worker.c`

**Files:**
- Modify: `pgvector/src/vector_index_worker.c`

- [ ] **Step 1: Add includes and `MergeThreadPool` struct**

Near the top of `vector_index_worker.c`, after the existing includes, add:

```c
#include "lsm_segment.h"  /* already present; verify */

/* Merge thread pool — dedicated pool for long-running merge/rebuild tasks */
#define MERGE_TASK_QUEUE_SIZE 64  /* max pending wakeup signals; pool scans on each wake */

typedef struct MergeThreadPool {
    pthread_t threads[MERGE_WORKERS_COUNT];
    pthread_mutex_t mutex;
    pthread_cond_t work_available;
    atomic_int pending_signals;  /* count of unprocessed wakeup signals */
    atomic_int shutdown;
    int num_threads;
} MergeThreadPool;

static MergeThreadPool *merge_pool = NULL;
```

- [ ] **Step 2: Add `init_merge_thread_pool` declaration and stub**

We will fill in the thread function (`merge_worker_thread`) in Task 7. For now, forward-declare it and write the init function:

```c
static void *merge_worker_thread(void *arg);  /* defined in Task 7 */

static void
init_merge_thread_pool(void)
{
    if (merge_pool != NULL)
        return;

    merge_pool = (MergeThreadPool *) malloc(sizeof(MergeThreadPool));
    if (merge_pool == NULL)
        elog(ERROR, "[init_merge_thread_pool] malloc failed");

    pthread_mutex_init(&merge_pool->mutex, NULL);
    pthread_cond_init(&merge_pool->work_available, NULL);
    atomic_init(&merge_pool->pending_signals, 0);
    atomic_init(&merge_pool->shutdown, 0);
    merge_pool->num_threads = MERGE_WORKERS_COUNT;

    for (int i = 0; i < merge_pool->num_threads; i++)
    {
        if (pthread_create(&merge_pool->threads[i], NULL,
                           merge_worker_thread, merge_pool) != 0)
            elog(ERROR, "[init_merge_thread_pool] failed to create merge thread %d", i);
    }

    elog(DEBUG1, "[init_merge_thread_pool] started %d merge threads", merge_pool->num_threads);
}
```

- [ ] **Step 3: Add `shutdown_merge_thread_pool`**

```c
static void
shutdown_merge_thread_pool(void)
{
    if (merge_pool == NULL)
        return;

    atomic_store(&merge_pool->shutdown, 1);
    pthread_mutex_lock(&merge_pool->mutex);
    pthread_cond_broadcast(&merge_pool->work_available);
    pthread_mutex_unlock(&merge_pool->mutex);

    for (int i = 0; i < merge_pool->num_threads; i++)
        pthread_join(merge_pool->threads[i], NULL);

    pthread_mutex_destroy(&merge_pool->mutex);
    pthread_cond_destroy(&merge_pool->work_available);
    free(merge_pool);
    merge_pool = NULL;
}
```

- [ ] **Step 4: Add helper to signal merge threads**

```c
static void
signal_merge_pool(void)
{
    if (merge_pool == NULL)
        return;
    pthread_mutex_lock(&merge_pool->mutex);
    pthread_cond_signal(&merge_pool->work_available);
    pthread_mutex_unlock(&merge_pool->mutex);
}
```

- [ ] **Step 5: Call `init_merge_thread_pool` in `vector_index_worker_main`**

In `vector_index_worker_main`, after the existing `init_maintenance_thread_pool()` call:

```c
    init_maintenance_thread_pool();
    init_merge_thread_pool();   /* NEW */
```

And before `proc_exit(0)` at graceful shutdown, add:

```c
    shutdown_merge_thread_pool();  /* NEW */
```

- [ ] **Step 6: Compile check**

```bash
cd /home/liu4127/postgresql/decoupled_pgvector/pgvector && make 2>&1 | grep "error:" | head -20
```

Expected: no new errors from this file. `merge_worker_thread` is forward-declared so the linker error will come when linking; that is expected at this stage.

---

## Task 5: Port merge scheduling logic to `vector_index_worker.c`

**Files:**
- Modify: `pgvector/src/vector_index_worker.c`

These functions are ported from `lsm_merge_worker.c` and adapted to work against `FlushedSegmentPool` instead of `SharedSegmentArray`. The key differences: use `pthread_rwlock_t seg_lock` instead of `LWLock`, use `FlushedSegmentData` fields directly, add `load_state == SEG_FULLY_LOADED` precondition.

- [ ] **Step 1: Port `choose_adjacent_smaller` → `choose_adjacent_smaller_pool`**

```c
/*
 * Choose a merge partner for the segment at `idx` in `pool`.
 * Prefers the smaller of the two neighbours. Skips segments that are
 * DISKANN, already compacting, or not fully loaded.
 * Must be called with pool->seg_lock held (read or write).
 */
static bool
choose_adjacent_smaller_pool(FlushedSegmentPool *pool, uint32_t idx,
                              uint32_t *chosen_adj_idx)
{
    uint32_t prev = pool->flushed_segments[idx].prev_idx;
    uint32_t next = pool->flushed_segments[idx].next_idx;

#define CANDIDATE_OK(i) \
    ((i) != (uint32_t)-1 && \
     pool->flushed_segments[(i)].in_used && \
     !pool->flushed_segments[(i)].is_compacting && \
     pool->flushed_segments[(i)].index_type != DISKANN && \
     atomic_load(&pool->flushed_segments[(i)].load_state) == (int)SEG_FULLY_LOADED)

    bool have_prev = CANDIDATE_OK(prev);
    bool have_next = CANDIDATE_OK(next);
#undef CANDIDATE_OK

    if (!have_prev && !have_next) return false;
    if (have_prev && !have_next)  { *chosen_adj_idx = prev; return true; }
    if (!have_prev && have_next)  { *chosen_adj_idx = next; return true; }

    *chosen_adj_idx = (pool->flushed_segments[prev].vec_count <=
                       pool->flushed_segments[next].vec_count) ? prev : next;
    return true;
}
```

- [ ] **Step 2: Add `MergeTaskLocal` struct**

```c
typedef struct {
    int operation_type;        /* SEGMENT_UPDATE_REBUILD_FLAT/DELETION/MERGE */
    int lsm_idx;
    Oid index_relid;
    uint32_t segment_idx0;
    uint32_t segment_idx1;     /* -1 for rebuild ops */
    SegmentId merged_start_sid;
    SegmentId merged_end_sid;
    uint32_t merged_vec_count;
    IndexType merged_index_type;
    uint32_t merged_delete_count;
} MergeTaskLocal;
```

- [ ] **Step 3: Port `claim_merge_task` → `claim_merge_task_pool`**

```c
static bool
claim_merge_task_pool(int lsm_idx, uint32_t segment_idx0, uint32_t segment_idx1,
                      int task_type, MergeTaskLocal *task)
{
    FlushedSegmentPool *pool = get_flushed_segment_pool(lsm_idx);

    pthread_rwlock_wrlock(&pool->seg_lock);

    /* Validate both segments are still usable */
    if (!pool->flushed_segments[segment_idx0].in_used ||
        pool->flushed_segments[segment_idx0].is_compacting ||
        atomic_load(&pool->flushed_segments[segment_idx0].load_state) != (int)SEG_FULLY_LOADED)
    {
        pthread_rwlock_unlock(&pool->seg_lock);
        return false;
    }
    if (task_type == SEGMENT_UPDATE_MERGE)
    {
        if (segment_idx1 == (uint32_t)-1 ||
            !pool->flushed_segments[segment_idx1].in_used ||
            pool->flushed_segments[segment_idx1].is_compacting ||
            atomic_load(&pool->flushed_segments[segment_idx1].load_state) != (int)SEG_FULLY_LOADED)
        {
            pthread_rwlock_unlock(&pool->seg_lock);
            return false;
        }
    }

    /* Claim */
    pool->flushed_segments[segment_idx0].is_compacting = true;
    if (task_type == SEGMENT_UPDATE_MERGE && segment_idx1 != (uint32_t)-1)
        pool->flushed_segments[segment_idx1].is_compacting = true;

    /* Fill task */
    task->operation_type    = task_type;
    task->lsm_idx           = lsm_idx;
    task->index_relid       = SharedLSMIndexBuffer->slots[lsm_idx].lsmIndex.indexRelId;
    task->segment_idx0      = segment_idx0;
    task->segment_idx1      = segment_idx1;
    task->merged_start_sid  = pool->flushed_segments[segment_idx0].segment_id_start;
    task->merged_end_sid    = (task_type == SEGMENT_UPDATE_MERGE)
                              ? pool->flushed_segments[segment_idx1].segment_id_end
                              : pool->flushed_segments[segment_idx0].segment_id_end;

    pthread_rwlock_unlock(&pool->seg_lock);
    return true;
}
```

- [ ] **Step 4: Port `traverse_and_check_priority` → `traverse_and_check_priority_pool`**

```c
static bool
traverse_and_check_priority_pool(int lsm_idx, int priority_type, MergeTaskLocal *task)
{
    FlushedSegmentPool *pool = get_flushed_segment_pool(lsm_idx);

    pthread_rwlock_rdlock(&pool->seg_lock);

    if (pool->head_idx == (uint32_t)-1 ||
        !pool->flushed_segments[pool->head_idx].in_used)
    {
        pthread_rwlock_unlock(&pool->seg_lock);
        return false;
    }

    uint32_t cur = pool->head_idx;
    while (cur != (uint32_t)-1)
    {
        FlushedSegmentData *seg = &pool->flushed_segments[cur];

        if (!seg->in_used || seg->is_compacting ||
            atomic_load(&seg->load_state) != (int)SEG_FULLY_LOADED)
        {
            cur = seg->next_idx;
            continue;
        }

        bool should_claim = false;
        uint32_t adj = (uint32_t)-1;

        switch (priority_type)
        {
            case 1: /* FLAT → rebuild flat */
                should_claim = (seg->index_type == FLAT);
                break;
            case 2: /* vec_count <= MEMTABLE_MAX_CAPACITY → merge */
                if (seg->index_type == DISKANN) break;
                if (seg->vec_count <= MEMTABLE_MAX_CAPACITY &&
                    choose_adjacent_smaller_pool(pool, cur, &adj) &&
                    pool->flushed_segments[adj].vec_count < MAX_SEGMENTS_SIZE &&
                    seg->vec_count + pool->flushed_segments[adj].vec_count < MAX_SEGMENTS_SIZE)
                    should_claim = true;
                break;
            case 3: /* high deletion ratio → rebuild deletion */
                if (seg->vec_count > 0 &&
                    (float)seg->delete_count / seg->vec_count > MERGE_DELETION_RATIO_THRESHOLD)
                    should_claim = true;
                break;
            case 4: /* vec_count <= THRESHOLD_SMALL_SEGMENT_SIZE → merge */
                if (seg->index_type == DISKANN) break;
                if (seg->vec_count <= THRESHOLD_SMALL_SEGMENT_SIZE &&
                    choose_adjacent_smaller_pool(pool, cur, &adj) &&
                    pool->flushed_segments[adj].vec_count < MAX_SEGMENTS_SIZE &&
                    seg->vec_count + pool->flushed_segments[adj].vec_count < MAX_SEGMENTS_SIZE)
                    should_claim = true;
                break;
            case 5: /* vec_count < MAX_SEGMENTS_SIZE → merge */
                if (seg->index_type == DISKANN) break;
                if (seg->vec_count < MAX_SEGMENTS_SIZE &&
                    choose_adjacent_smaller_pool(pool, cur, &adj) &&
                    pool->flushed_segments[adj].vec_count < MAX_SEGMENTS_SIZE &&
                    seg->vec_count + pool->flushed_segments[adj].vec_count < MAX_SEGMENTS_SIZE)
                    should_claim = true;
                break;
        }

        if (should_claim)
        {
            pthread_rwlock_unlock(&pool->seg_lock);

            int task_type = (priority_type == 1) ? SEGMENT_UPDATE_REBUILD_FLAT :
                            (priority_type == 3) ? SEGMENT_UPDATE_REBUILD_DELETION :
                                                   SEGMENT_UPDATE_MERGE;
            uint32_t lo = (adj == (uint32_t)-1) ? cur : Min(cur, adj);
            uint32_t hi = (adj == (uint32_t)-1) ? (uint32_t)-1 : Max(cur, adj);

            if (claim_merge_task_pool(lsm_idx, lo, hi, task_type, task))
                return true;

            pthread_rwlock_rdlock(&pool->seg_lock);
            cur = pool->head_idx;   /* restart after failed claim */
            continue;
        }

        cur = seg->next_idx;
    }

    pthread_rwlock_unlock(&pool->seg_lock);
    return false;
}
```

- [ ] **Step 5: Port `scan_and_claim_merge_task` → `scan_and_claim_merge_task_pool`**

```c
static bool
scan_and_claim_merge_task_pool(MergeTaskLocal *task)
{
    for (int lsm_idx = 0; lsm_idx < INDEX_BUF_SIZE; lsm_idx++)
    {
        if (pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[lsm_idx].valid) != 1)
            continue;
        FlushedSegmentPool *pool = get_flushed_segment_pool(lsm_idx);
        if (pg_atomic_read_u32(&pool->flat_count) > 0)
            if (traverse_and_check_priority_pool(lsm_idx, 1, task)) return true;
    }
    for (int lsm_idx = 0; lsm_idx < INDEX_BUF_SIZE; lsm_idx++)
    {
        if (pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[lsm_idx].valid) != 1)
            continue;
        FlushedSegmentPool *pool = get_flushed_segment_pool(lsm_idx);
        if (pg_atomic_read_u32(&pool->memtable_capacity_le_count) > 0)
            if (traverse_and_check_priority_pool(lsm_idx, 2, task)) return true;
    }
    for (int lsm_idx = 0; lsm_idx < INDEX_BUF_SIZE; lsm_idx++)
    {
        if (pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[lsm_idx].valid) != 1)
            continue;
        if (traverse_and_check_priority_pool(lsm_idx, 3, task)) return true;
    }
    for (int lsm_idx = 0; lsm_idx < INDEX_BUF_SIZE; lsm_idx++)
    {
        if (pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[lsm_idx].valid) != 1)
            continue;
        FlushedSegmentPool *pool = get_flushed_segment_pool(lsm_idx);
        if (pg_atomic_read_u32(&pool->small_segment_le_count) > 0)
            if (traverse_and_check_priority_pool(lsm_idx, 4, task)) return true;
    }
    for (int lsm_idx = 0; lsm_idx < INDEX_BUF_SIZE; lsm_idx++)
    {
        if (pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[lsm_idx].valid) != 1)
            continue;
        if (traverse_and_check_priority_pool(lsm_idx, 5, task)) return true;
    }
    return false;
}
```

- [ ] **Step 6: Compile check**

```bash
cd /home/liu4127/postgresql/decoupled_pgvector/pgvector && make 2>&1 | grep "error:" | head -20
```

Expected: no errors from the new functions. Remaining errors are in `lsm_merge_worker.c` and the not-yet-updated handler in `vector_index_worker.c`.

---

## Task 6: Port `rebuild_index` to `vector_index_worker.c`

**Files:**
- Modify: `pgvector/src/vector_index_worker.c`

This function is ported from `lsm_merge_worker.c` with four key changes:
1. `palloc`/`pfree` → `malloc`/`free` (no PostgreSQL memory context in merge threads)
2. `merge_worker_manager->segment_arrays[lsm_idx].segments[idx]` → `FlushedSegmentData` fields accessed directly via `MergeTaskLocal`
3. `LWLockAcquire(&segment->bitmap_lock, ...)` → `pthread_mutex_lock(&seg->per_seg_mutex)`
4. No `finish_merge_task` call — update `FlushedSegmentPool` directly at the end

- [ ] **Step 1: Add required includes to `vector_index_worker.c`**

Ensure these includes are present at the top:
```c
#include "vectorindeximpl.hpp"
#include "lsm_segment.h"
#include "lsmindex.h"
```

- [ ] **Step 2: Add `rebuild_index_pool`**

```c
/*
 * Rebuild a single segment's index, filtering deleted vectors.
 * Updates FlushedSegmentPool directly on completion.
 * Called from merge threads — must use malloc, not palloc.
 */
static void
rebuild_index_pool(MergeTaskLocal *task)
{
    FlushedSegmentPool *pool     = get_flushed_segment_pool(task->lsm_idx);
    FlushedSegmentData *seg      = &pool->flushed_segments[task->segment_idx0];
    Oid index_relid              = task->index_relid;
    SegmentId start_sid          = seg->segment_id_start;
    SegmentId end_sid            = seg->segment_id_end;

    uint32_t version = find_latest_segment_version(index_relid, start_sid, end_sid);
    IndexType target_type = SharedLSMIndexBuffer->slots[task->lsm_idx].lsmIndex.index_type;

    /* --- Phase 1: load data from disk (no lock) --- */
    void *old_index_ptr = NULL;
    load_index_file(index_relid, start_sid, end_sid, version, seg->index_type,
                    &old_index_ptr, false);

    uint8_t *old_bitmap_ptr = NULL;
    uint32_t delete_count;
    load_bitmap_file(index_relid, start_sid, end_sid, version,
                     &old_bitmap_ptr, false, &delete_count);

    int64_t *old_mapping_ptr = NULL;
    load_mapping_file(index_relid, start_sid, end_sid, version, &old_mapping_ptr, false);

    SegmentOffsetRange *old_offsets_ptr = NULL;
    load_offset_file(index_relid, start_sid, end_sid, version, &old_offsets_ptr, false);

    /* --- Phase 2: build new index from snapshot bitmap (no lock) --- */
    void *new_index_ptr = NULL;
    int new_index_count = 0;
    int M = 32, efConstruction = 400, lists = 1024;
    MergeIndex(old_index_ptr, old_bitmap_ptr, seg->vec_count,
               seg->index_type, target_type,
               &new_index_ptr, &new_index_count,
               M, efConstruction, lists);

    uint8_t *original_bitmap_ptr = old_bitmap_ptr;
    old_bitmap_ptr = NULL;

    /* --- Phase 3: acquire per_seg_mutex, reload bitmap to capture concurrent vacuum --- */
    pthread_mutex_lock(&seg->per_seg_mutex);

    /* Reload bitmap from disk to capture any vacuum deletions during phase 2 */
    load_bitmap_file(index_relid, start_sid, end_sid, version,
                     &old_bitmap_ptr, false, &delete_count);

    /* Serialize new index */
    void *new_index_bin = NULL;
    IndexSerialize(new_index_ptr, &new_index_bin);

    /* Build new mapping and bitmap */
    uint32_t segment_count = end_sid - start_sid + 1;
    Size new_map_size   = sizeof(int64_t) * (Size)new_index_count;
    int64_t *new_mapping_ptr = (int64_t *) malloc(new_map_size);
    uint8_t *new_bitmap_ptr  = (uint8_t *) calloc(1, GET_BITMAP_SIZE(new_index_count));
    SegmentOffsetRange *new_offsets_ptr =
        (SegmentOffsetRange *) calloc(segment_count, sizeof(SegmentOffsetRange));
    uint32_t new_delete_count = 0;

    /* Initialise offset sentinels */
    for (uint32_t j = 0; j < segment_count; j++)
    {
        new_offsets_ptr[j].sid = old_offsets_ptr[j].sid;
        new_offsets_ptr[j].start_offset = SIZE_MAX;
        new_offsets_ptr[j].end_offset   = 0;
    }

    {
        int write_idx = 0;
        uint32_t cur_sid_idx = 0;
        for (int i = 0; i < (int)seg->vec_count; i++)
        {
            while (cur_sid_idx < segment_count)
            {
                if (old_offsets_ptr[cur_sid_idx].start_offset ==
                    old_offsets_ptr[cur_sid_idx].end_offset)
                    cur_sid_idx++;
                else if (i >= (int)old_offsets_ptr[cur_sid_idx].end_offset)
                    cur_sid_idx++;
                else
                    break;
            }

            if (!IS_SLOT_SET(original_bitmap_ptr, i))
            {
                new_mapping_ptr[write_idx] = old_mapping_ptr[i];
                if (IS_SLOT_SET(old_bitmap_ptr, i))
                {
                    SET_SLOT(new_bitmap_ptr, write_idx);
                    new_delete_count++;
                }
                if (cur_sid_idx < segment_count &&
                    i >= (int)old_offsets_ptr[cur_sid_idx].start_offset &&
                    i <  (int)old_offsets_ptr[cur_sid_idx].end_offset)
                {
                    if (new_offsets_ptr[cur_sid_idx].start_offset == SIZE_MAX)
                        new_offsets_ptr[cur_sid_idx].start_offset = write_idx;
                    new_offsets_ptr[cur_sid_idx].end_offset = write_idx + 1;
                }
                write_idx++;
            }
        }
        Assert(write_idx == new_index_count);
    }

    /* Fix empty offset ranges */
    Size prev_end = 0;
    for (uint32_t j = 0; j < segment_count; j++)
    {
        if (new_offsets_ptr[j].start_offset == SIZE_MAX)
            new_offsets_ptr[j].start_offset = new_offsets_ptr[j].end_offset = prev_end;
        else
            prev_end = new_offsets_ptr[j].end_offset;
    }

    free(original_bitmap_ptr);

    /* Write new segment to disk (new version) */
    task->merged_start_sid    = start_sid;
    task->merged_end_sid      = end_sid;
    task->merged_index_type   = target_type;
    task->merged_vec_count    = new_index_count;
    task->merged_delete_count = new_delete_count;

    PrepareFlushMetaData prep;
    prep.start_sid   = start_sid;
    prep.end_sid     = end_sid;
    prep.valid_rows  = new_index_count;
    prep.index_type  = target_type;
    prep.index_bin   = new_index_bin;
    prep.bitmap_ptr  = new_bitmap_ptr;
    prep.bitmap_size = GET_BITMAP_SIZE(new_index_count);
    prep.delete_count = new_delete_count;
    prep.map_ptr     = new_mapping_ptr;
    prep.map_size    = new_map_size;
    prep.offsets     = new_offsets_ptr;
    flush_segment_to_disk(index_relid, &prep);
    /* flush_segment_to_disk frees new_index_bin internally */

    uint32_t new_version = find_latest_segment_version(index_relid, start_sid, end_sid);

    /* --- Phase 4: update FlushedSegmentPool under seg_lock write (while holding per_seg_mutex) --- */
    pthread_rwlock_wrlock(&pool->seg_lock);

    /* Swap in-memory segment data */
    void *old_index_in_mem  = seg->index_ptr;
    uint8_t *old_bitmap_mem = seg->bitmap_ptr;
    int64_t *old_map_mem    = seg->map_ptr;

    seg->index_ptr    = new_index_ptr;
    seg->bitmap_ptr   = new_bitmap_ptr;
    seg->map_ptr      = new_mapping_ptr;
    seg->vec_count    = new_index_count;
    seg->index_type   = target_type;
    seg->delete_count = new_delete_count;
    seg->version      = new_version;
    seg->is_compacting = false;

    /* Update pool statistics */
    /* (old stats were for old index_type/vec_count; remove and re-add) */
    /* update_pool_stats called automatically by register path is not invoked here
       because we are modifying in-place, so do it manually: */
    update_pool_stats(pool, seg->index_type, seg->vec_count, -1); /* remove old */
    update_pool_stats(pool, target_type, new_index_count, +1);    /* add new */

    pthread_rwlock_unlock(&pool->seg_lock);
    pthread_mutex_unlock(&seg->per_seg_mutex);

    /* Free old in-memory data */
    IndexFree(old_index_in_mem);
    IndexFree(old_index_ptr);
    free(old_bitmap_mem);
    free(old_map_mem);
    free(old_bitmap_ptr);
    free(old_mapping_ptr);
    free(old_offsets_ptr);
    free(new_offsets_ptr);

    fprintf(stderr, "[rebuild_index_pool] rebuilt segment %u-%u → %d vectors\n",
            start_sid, end_sid, new_index_count);
}
```

- [ ] **Step 2: Compile check**

```bash
cd /home/liu4127/postgresql/decoupled_pgvector/pgvector && make 2>&1 | grep "error:" | head -20
```

Expected: no new errors from `rebuild_index_pool`.

---

## Task 7: Port `merge_adjacent_segments` to `vector_index_worker.c`

**Files:**
- Modify: `pgvector/src/vector_index_worker.c`

Same four adaptations as Task 6. Two `per_seg_mutex` locks are acquired in ascending `start_sid` order to prevent deadlock.

- [ ] **Step 1: Add `merge_adjacent_segments_pool`**

```c
static void
merge_adjacent_segments_pool(MergeTaskLocal *task)
{
    FlushedSegmentPool *pool = get_flushed_segment_pool(task->lsm_idx);
    FlushedSegmentData *seg0 = &pool->flushed_segments[task->segment_idx0];
    FlushedSegmentData *seg1 = &pool->flushed_segments[task->segment_idx1];
    Oid index_relid          = task->index_relid;

    uint32_t version0 = find_latest_segment_version(index_relid,
                            seg0->segment_id_start, seg0->segment_id_end);
    uint32_t version1 = find_latest_segment_version(index_relid,
                            seg1->segment_id_start, seg1->segment_id_end);

    /* --- Phase 1: load both indices and bitmaps (no lock) --- */
    void *index0_ptr = NULL, *index1_ptr = NULL;
    load_index_file(index_relid, seg0->segment_id_start, seg0->segment_id_end,
                    version0, seg0->index_type, &index0_ptr, false);
    load_index_file(index_relid, seg1->segment_id_start, seg1->segment_id_end,
                    version1, seg1->index_type, &index1_ptr, false);

    uint8_t *bitmap0_ptr = NULL, *bitmap1_ptr = NULL;
    uint32_t delete_count0, delete_count1;
    load_bitmap_file(index_relid, seg0->segment_id_start, seg0->segment_id_end,
                     version0, &bitmap0_ptr, false, &delete_count0);
    load_bitmap_file(index_relid, seg1->segment_id_start, seg1->segment_id_end,
                     version1, &bitmap1_ptr, false, &delete_count1);

    int64_t *mapping0_ptr = NULL, *mapping1_ptr = NULL;
    load_mapping_file(index_relid, seg0->segment_id_start, seg0->segment_id_end,
                      version0, &mapping0_ptr, false);
    load_mapping_file(index_relid, seg1->segment_id_start, seg1->segment_id_end,
                      version1, &mapping1_ptr, false);

    SegmentOffsetRange *offsets0_ptr = NULL, *offsets1_ptr = NULL;
    load_offset_file(index_relid, seg0->segment_id_start, seg0->segment_id_end,
                     version0, &offsets0_ptr, false);
    load_offset_file(index_relid, seg1->segment_id_start, seg1->segment_id_end,
                     version1, &offsets1_ptr, false);

    /* Determine larger/smaller for MergeTwoIndices */
    void     *larger_index, *smaller_index;
    uint8_t  *larger_bitmap, *smaller_bitmap;
    int64_t  *larger_mapping, *smaller_mapping;
    SegmentOffsetRange *larger_offsets, *smaller_offsets;
    uint32_t  larger_sid_count, smaller_sid_count;
    int       larger_count, smaller_count;
    IndexType larger_type, smaller_type;

    if (seg0->vec_count >= seg1->vec_count) {
        larger_index   = index0_ptr;   smaller_index   = index1_ptr;
        larger_bitmap  = bitmap0_ptr;  smaller_bitmap  = bitmap1_ptr;
        larger_mapping = mapping0_ptr; smaller_mapping = mapping1_ptr;
        larger_offsets = offsets0_ptr; smaller_offsets = offsets1_ptr;
        larger_sid_count  = seg0->segment_id_end - seg0->segment_id_start + 1;
        smaller_sid_count = seg1->segment_id_end - seg1->segment_id_start + 1;
        larger_count   = seg0->vec_count; smaller_count = seg1->vec_count;
        larger_type    = seg0->index_type; smaller_type  = seg1->index_type;
    } else {
        larger_index   = index1_ptr;   smaller_index   = index0_ptr;
        larger_bitmap  = bitmap1_ptr;  smaller_bitmap  = bitmap0_ptr;
        larger_mapping = mapping1_ptr; smaller_mapping = mapping0_ptr;
        larger_offsets = offsets1_ptr; smaller_offsets = offsets0_ptr;
        larger_sid_count  = seg1->segment_id_end - seg1->segment_id_start + 1;
        smaller_sid_count = seg0->segment_id_end - seg0->segment_id_start + 1;
        larger_count   = seg1->vec_count; smaller_count = seg0->vec_count;
        larger_type    = seg1->index_type; smaller_type  = seg0->index_type;
    }

    /* --- Phase 2: merge indices (no lock) --- */
    int merged_count = 0;
    void *merged_index_ptr = MergeTwoIndices(larger_index, larger_count, larger_type,
                                              smaller_index, smaller_count, smaller_type,
                                              &merged_count);
    Assert(merged_index_ptr == larger_index);

    /* --- Phase 3: acquire per_seg_mutex for both segments (ascending start_sid order) --- */
    FlushedSegmentData *first_lock  = (seg0->segment_id_start <= seg1->segment_id_start) ? seg0 : seg1;
    FlushedSegmentData *second_lock = (seg0->segment_id_start <= seg1->segment_id_start) ? seg1 : seg0;
    pthread_mutex_lock(&first_lock->per_seg_mutex);
    pthread_mutex_lock(&second_lock->per_seg_mutex);

    /* Reload bitmaps to capture concurrent vacuum */
    free(bitmap0_ptr); free(bitmap1_ptr);
    load_bitmap_file(index_relid, seg0->segment_id_start, seg0->segment_id_end,
                     version0, &bitmap0_ptr, false, &delete_count0);
    load_bitmap_file(index_relid, seg1->segment_id_start, seg1->segment_id_end,
                     version1, &bitmap1_ptr, false, &delete_count1);
    /* Re-point larger/smaller bitmaps after reload */
    if (seg0->vec_count >= seg1->vec_count)
        { larger_bitmap = bitmap0_ptr; smaller_bitmap = bitmap1_ptr; }
    else
        { larger_bitmap = bitmap1_ptr; smaller_bitmap = bitmap0_ptr; }

    int total_count = larger_count + smaller_count;
    Assert(merged_count == total_count);

    /* Build merged bitmap, mapping, offsets */
    Size merged_bitmap_size = GET_BITMAP_SIZE(total_count);
    uint8_t *merged_bitmap  = (uint8_t *) calloc(1, merged_bitmap_size);
    for (int i = 0; i < larger_count; i++)
        if (IS_SLOT_SET(larger_bitmap, i)) SET_SLOT(merged_bitmap, i);
    for (int i = 0; i < smaller_count; i++)
        if (IS_SLOT_SET(smaller_bitmap, i)) SET_SLOT(merged_bitmap, larger_count + i);
    int rem = total_count % 8;
    if (rem) merged_bitmap[merged_bitmap_size - 1] &= (uint8_t)((1 << rem) - 1);

    uint32_t merged_delete_count = delete_count0 + delete_count1;

    int64_t *merged_mapping = (int64_t *) malloc(sizeof(int64_t) * (Size)total_count);
    for (int i = 0; i < larger_count; i++)  merged_mapping[i]               = larger_mapping[i];
    for (int i = 0; i < smaller_count; i++) merged_mapping[larger_count + i] = smaller_mapping[i];

    uint32_t merged_seg_count = task->merged_end_sid - task->merged_start_sid + 1;
    SegmentOffsetRange *merged_offsets =
        (SegmentOffsetRange *) calloc(merged_seg_count, sizeof(SegmentOffsetRange));
    for (int i = 0; i < (int)larger_sid_count; i++)
        merged_offsets[i] = larger_offsets[i];
    for (int i = 0; i < (int)smaller_sid_count; i++) {
        merged_offsets[larger_sid_count + i] = smaller_offsets[i];
        merged_offsets[larger_sid_count + i].start_offset += larger_count;
        merged_offsets[larger_sid_count + i].end_offset   += larger_count;
    }

    void *merged_index_bin = NULL;
    IndexSerialize(merged_index_ptr, &merged_index_bin);

    task->merged_index_type   = larger_type;
    task->merged_vec_count    = merged_count;
    task->merged_delete_count = merged_delete_count;

    PrepareFlushMetaData prep;
    prep.start_sid   = task->merged_start_sid;
    prep.end_sid     = task->merged_end_sid;
    prep.valid_rows  = merged_count;
    prep.index_type  = larger_type;
    prep.index_bin   = merged_index_bin;
    prep.bitmap_ptr  = merged_bitmap;
    prep.bitmap_size = merged_bitmap_size;
    prep.delete_count = merged_delete_count;
    prep.map_ptr     = merged_mapping;
    prep.map_size    = sizeof(int64_t) * (Size)merged_count;
    prep.offsets     = merged_offsets;
    flush_segment_to_disk(index_relid, &prep);

    uint32_t new_version = find_latest_segment_version(index_relid,
                               task->merged_start_sid, task->merged_end_sid);

    /* --- Phase 4: update FlushedSegmentPool under seg_lock write --- */
    pthread_rwlock_wrlock(&pool->seg_lock);

    /* Update seg0 to represent the merged segment */
    void *old_index0_mem    = seg0->index_ptr;
    uint8_t *old_bitmap0_mem = seg0->bitmap_ptr;
    int64_t *old_map0_mem    = seg0->map_ptr;

    update_pool_stats(pool, seg0->index_type, seg0->vec_count, -1);
    seg0->segment_id_end = task->merged_end_sid;
    seg0->vec_count      = merged_count;
    seg0->index_type     = larger_type;
    seg0->delete_count   = merged_delete_count;
    seg0->version        = new_version;
    seg0->index_ptr      = merged_index_ptr;
    seg0->bitmap_ptr     = merged_bitmap;
    seg0->map_ptr        = merged_mapping;
    seg0->is_compacting  = false;
    update_pool_stats(pool, larger_type, merged_count, +1);

    /* Remove seg1 from the pool */
    void *old_index1_mem    = seg1->index_ptr;
    uint8_t *old_bitmap1_mem = seg1->bitmap_ptr;
    int64_t *old_map1_mem    = seg1->map_ptr;
    cleanup_flushed_segment(pool, task->segment_idx1);

    pthread_rwlock_unlock(&pool->seg_lock);
    pthread_mutex_unlock(&second_lock->per_seg_mutex);
    pthread_mutex_unlock(&first_lock->per_seg_mutex);

    /* Free old data */
    IndexFree(old_index0_mem);
    IndexFree(old_index1_mem);
    free(old_bitmap0_mem); free(old_bitmap1_mem);
    free(old_map0_mem);    free(old_map1_mem);
    free(bitmap0_ptr);     free(bitmap1_ptr);
    free(mapping0_ptr);    free(mapping1_ptr);
    free(offsets0_ptr);    free(offsets1_ptr);
    free(merged_offsets);

    fprintf(stderr, "[merge_adjacent_segments_pool] merged %u-%u + %u-%u → %u-%u (%d vectors)\n",
            seg0->segment_id_start, seg0->segment_id_end,
            seg1->segment_id_start, seg1->segment_id_end,
            task->merged_start_sid, task->merged_end_sid, merged_count);
}
```

- [ ] **Step 2: Compile check**

```bash
cd /home/liu4127/postgresql/decoupled_pgvector/pgvector && make 2>&1 | grep "error:" | head -20
```

---

## Task 8: Add `merge_worker_thread` main loop and wire into VIW

**Files:**
- Modify: `pgvector/src/vector_index_worker.c`

- [ ] **Step 1: Implement `merge_worker_thread`**

```c
static void *
merge_worker_thread(void *arg)
{
    MergeThreadPool *pool = (MergeThreadPool *) arg;

    while (!atomic_load(&pool->shutdown))
    {
        /* Wait for a wakeup signal */
        pthread_mutex_lock(&pool->mutex);
        while (!atomic_load(&pool->shutdown) && atomic_load(&pool->pending_signals) == 0)
            pthread_cond_wait(&pool->work_available, &pool->mutex);
        if (atomic_load(&pool->pending_signals) > 0)
            atomic_fetch_sub(&pool->pending_signals, 1);
        pthread_mutex_unlock(&pool->mutex);

        if (atomic_load(&pool->shutdown))
            break;

        /* Scan for a task and execute it */
        MergeTaskLocal task;
        bool claimed = scan_and_claim_merge_task_pool(&task);

        if (!claimed)
            continue;

        fprintf(stderr, "[merge_worker_thread] claimed task op=%d lsm=%d seg0=%u seg1=%u\n",
                task.operation_type, task.lsm_idx, task.segment_idx0, task.segment_idx1);

        switch (task.operation_type)
        {
            case SEGMENT_UPDATE_REBUILD_FLAT:
            case SEGMENT_UPDATE_REBUILD_DELETION:
                rebuild_index_pool(&task);
                break;
            case SEGMENT_UPDATE_MERGE:
                merge_adjacent_segments_pool(&task);
                break;
            default:
                fprintf(stderr, "[merge_worker_thread] unknown op %d\n", task.operation_type);
                break;
        }

        /* Signal again — this completion may enable another merge */
        signal_merge_pool();
    }

    return NULL;
}
```

- [ ] **Step 2: Update `signal_merge_pool` to increment `pending_signals`**

Replace the earlier stub with:

```c
static void
signal_merge_pool(void)
{
    if (merge_pool == NULL)
        return;
    pthread_mutex_lock(&merge_pool->mutex);
    atomic_fetch_add(&merge_pool->pending_signals, 1);
    pthread_cond_signal(&merge_pool->work_available);
    pthread_mutex_unlock(&merge_pool->mutex);
}
```

- [ ] **Step 3: Signal merge pool from maintenance thread completions**

In the `maintenance_worker_thread` function inside the `switch (task->task_type)` block, add `signal_merge_pool()` after each relevant case:

After `SEGMENT_UPDATE_REGULAR` completes (segment added to pool):
```c
case SEGMENT_UPDATE_REGULAR:
{
    /* ... existing load code ... */
    signal_merge_pool();   /* NEW */
    break;
}
```

After `IndexLoadTaskType` phase-1 mmap signals the backend:
```c
    /* After load_all_segments_from_disk_mmap and before submitting upgrade tasks */
    signal_merge_pool();   /* NEW */
```

After `InternalSegmentUpgradeTaskType` successfully upgrades a segment:
```c
    /* After atomic_store(&seg->load_state, (int)SEG_FULLY_LOADED) */
    signal_merge_pool();   /* NEW */
```

- [ ] **Step 4: Compile check**

```bash
cd /home/liu4127/postgresql/decoupled_pgvector/pgvector && make 2>&1 | grep "error:" | head -20
```

Expected: no new errors.

---

## Task 9: Update `SEGMENT_UPDATE_VACUUM` handler in `vector_index_worker.c`

**Files:**
- Modify: `pgvector/src/vector_index_worker.c`

- [ ] **Step 1: Replace the existing `SEGMENT_UPDATE_VACUUM` case**

In `maintenance_worker_thread`, inside `case SegmentUpdateTaskType:`, replace the `SEGMENT_UPDATE_VACUUM` sub-case with:

```c
case SEGMENT_UPDATE_VACUUM:
{
    fprintf(stderr, "[maintenance_worker] SEGMENT_UPDATE_VACUUM %d-%d expected_version=%u\n",
            update_task->start_sid, update_task->end_sid, update_task->expected_version);

    FlushedSegmentPool *pool_seg = get_flushed_segment_pool(lsm_idx);

    /* Step 1: find segment by exact (start_sid, end_sid) under read lock */
    pthread_rwlock_rdlock(&pool_seg->seg_lock);
    uint32_t seg_idx = find_segment_by_sids(pool_seg,
                           update_task->start_sid, update_task->end_sid);
    if (seg_idx != (uint32_t)-1)
        atomic_fetch_add(&pool_seg->flushed_segments[seg_idx].ref_count, 1);
    pthread_rwlock_unlock(&pool_seg->seg_lock);

    if (seg_idx == (uint32_t)-1)
    {
        /* Segment merged away → tell backend to retry with new segment */
        vs_search_result_at(update_task->backend_pgprocno)->maint_status = 1; /* RETRY */
        client = &ProcGlobal->allProcs[update_task->backend_pgprocno];
        break;
    }

    FlushedSegmentData *seg = &pool_seg->flushed_segments[seg_idx];

    /* Step 2: acquire per_seg_mutex, check version */
    pthread_mutex_lock(&seg->per_seg_mutex);

    int retry = 0;
    if (seg->version != update_task->expected_version || !seg->in_used)
    {
        retry = 1;  /* version changed (rebuild happened) or segment gone */
    }
    else
    {
        /* Load latest bitmap subversion and OR into in-memory bitmap */
        uint8_t *new_bitmap = NULL;
        uint32_t new_delete_count;
        load_bitmap_file(index_relid, update_task->start_sid, update_task->end_sid,
                         LOAD_LATEST_VERSION, &new_bitmap, false, &new_delete_count);
        if (new_bitmap != NULL)
        {
            Size bitmap_size = GET_BITMAP_SIZE(seg->vec_count);
            for (Size bi = 0; bi < bitmap_size; bi++)
                seg->bitmap_ptr[bi] |= new_bitmap[bi];
            seg->delete_count = new_delete_count;
            free(new_bitmap);
        }
    }

    pthread_mutex_unlock(&seg->per_seg_mutex);

    /* Step 3: decrement ref count; write result; signal backend */
    decrement_flushed_segment_ref_count(pool_seg, seg_idx);

    vs_search_result_at(update_task->backend_pgprocno)->maint_status = retry;
    client = &ProcGlobal->allProcs[update_task->backend_pgprocno];

    fprintf(stderr, "[maintenance_worker] SEGMENT_UPDATE_VACUUM %s for %d-%d\n",
            retry ? "RETRY" : "OK", update_task->start_sid, update_task->end_sid);
    break;
}
```

- [ ] **Step 2: Remove `SEGMENT_UPDATE_REBUILD_FLAT`, `SEGMENT_UPDATE_REBUILD_DELETION`, `SEGMENT_UPDATE_MERGE` cases from the maintenance handler**

These operations are now handled directly by merge threads and no longer arrive via the ring buffer. Delete those three sub-cases from the `case SegmentUpdateTaskType:` switch. If any caller still sends these types through the ring buffer they will fall to `default` and log a warning; no functional path should trigger them after Task 12.

- [ ] **Step 3: Compile check**

```bash
cd /home/liu4127/postgresql/decoupled_pgvector/pgvector && make 2>&1 | grep "error:" | head -20
```

---

## Task 10: Update `lsmbackground.c`

**Files:**
- Modify: `pgvector/src/lsmbackground.c`

- [ ] **Step 1: Remove `add_to_segment_array` call**

In `lsm_flush_one_pending`, delete the line:

```c
    // step 7. update the segment array
    add_to_segment_array(slot_idx, lsm->indexRelId, prep.start_sid, prep.end_sid, prep.valid_rows, prep.index_type, prep.delete_count);
```

The merge pool is now signalled inside the VIW maintenance thread after processing `SEGMENT_UPDATE_REGULAR`.

- [ ] **Step 2: Remove the `lsm_merge_worker.h` include**

Delete:
```c
#include "lsm_merge_worker.h"
```

- [ ] **Step 3: Compile check**

```bash
cd /home/liu4127/postgresql/decoupled_pgvector/pgvector && make 2>&1 | grep "error:" | head -20
```

---

## Task 11: Update `bulk_delete_lsm_index` in `lsmindex.c`

**Files:**
- Modify: `pgvector/src/lsmindex.c`

- [ ] **Step 1: Remove `lsm_merge_worker.h` include and `merge_worker_manager` guard**

Delete:
```c
#include "lsm_merge_worker.h"
```

Remove the early-return guard at the start of the flushed segments step:
```c
        if (merge_worker_manager == NULL)
        {
            return stats;
        }
```

- [ ] **Step 2: Replace the `SharedSegmentArray` traversal with a disk scan**

Replace the entire "step 3: vacuum the flushed segments" block (from `SharedSegmentArray *seg_array = ...` through the closing `LWLockRelease(seg_array->lock)`) with:

```c
    /* Step 3: vacuum flushed segments via disk scan */
    {
        SegmentFileInfo files[MAX_SEGMENTS_COUNT];
        int file_count = scan_segment_metadata_files(indexRelId, files, MAX_SEGMENTS_COUNT);

        for (int fi = 0; fi < file_count; fi++)
        {
retry_segment:;
            SegmentId start_sid_disk, end_sid_disk;
            uint32_t valid_rows;
            IndexType seg_index_type;
            uint32_t seg_version = files[fi].version;

            if (!read_lsm_segment_metadata(indexRelId,
                                           files[fi].start_sid, files[fi].end_sid,
                                           seg_version,
                                           &start_sid_disk, &end_sid_disk,
                                           &valid_rows, &seg_index_type))
                continue;  /* file gone between scan and read */

            /* Skip segments that match an already-vacuumed memtable */
            bool already_vacuumed = false;
            if (start_sid_disk == end_sid_disk)
            {
                for (uint32_t vi = 0; vi < vacuumed_count; vi++)
                {
                    if (start_sid_disk == vacuumed_memtable_ids[vi])
                    {
                        already_vacuumed = true;
                        break;
                    }
                }
            }
            if (already_vacuumed)
                continue;

            /* Load bitmap + mapping at this version */
            uint8_t *bitmap_ptr = NULL;
            int64_t *mapping_ptr = NULL;
            uint32_t delete_count;
            load_bitmap_file(indexRelId, start_sid_disk, end_sid_disk,
                             seg_version, &bitmap_ptr, true, &delete_count);
            load_mapping_file(indexRelId, start_sid_disk, end_sid_disk,
                              seg_version, &mapping_ptr, true);

            if (bitmap_ptr == NULL || mapping_ptr == NULL)
            {
                if (bitmap_ptr)  pfree(bitmap_ptr);
                if (mapping_ptr) pfree(mapping_ptr);
                continue;
            }

            /* Walk vectors and apply callback */
            bool bitmap_changed = false;
            for (uint32_t i = 0; i < valid_rows; i++)
            {
                if (!IS_SLOT_SET(bitmap_ptr, i))
                {
                    ItemPointerData tid = Int64ToItemPointer(mapping_ptr[i]);
                    if (callback(&tid, callback_state))
                    {
                        SET_SLOT(bitmap_ptr, i);
                        bitmap_changed = true;
                        stats->tuples_removed++;
                        delete_count++;
                    }
                    else
                    {
                        stats->num_index_tuples++;
                    }
                }
            }

            pfree(mapping_ptr);

            if (bitmap_changed)
            {
                uint32_t latest_sub = find_latest_bitmap_subversion(indexRelId,
                                          start_sid_disk, end_sid_disk, seg_version);
                uint32_t next_sub = (latest_sub == UINT32_MAX) ? 0 : latest_sub + 1;
                Size bitmap_size = GET_BITMAP_SIZE(valid_rows);
                write_bitmap_file_with_subversion(indexRelId, start_sid_disk, end_sid_disk,
                                                  seg_version, next_sub,
                                                  bitmap_ptr, bitmap_size, delete_count);

                int result = segment_update_blocking(lsm_idx, indexRelId,
                                 SEGMENT_UPDATE_VACUUM,
                                 start_sid_disk, end_sid_disk,
                                 seg_version);  /* expected_version */
                if (result == 1)  /* RETRY */
                {
                    pfree(bitmap_ptr);
                    /*
                     * The segment was rebuilt or merged away while we were
                     * computing deletions. Re-scan disk to find the current
                     * segment covering start_sid_disk and retry.
                     */
                    SegmentFileInfo retry_files[MAX_SEGMENTS_COUNT];
                    int retry_count = scan_segment_metadata_files(indexRelId,
                                          retry_files, MAX_SEGMENTS_COUNT);
                    for (int rfi = 0; rfi < retry_count; rfi++)
                    {
                        if (retry_files[rfi].start_sid <= start_sid_disk &&
                            start_sid_disk <= retry_files[rfi].end_sid)
                        {
                            files[fi] = retry_files[rfi];
                            goto retry_segment;
                        }
                    }
                    /* Segment completely gone — skip */
                    continue;
                }
            }

            pfree(bitmap_ptr);
        }
    }
```

- [ ] **Step 3: Compile check**

```bash
cd /home/liu4127/postgresql/decoupled_pgvector/pgvector && make 2>&1 | grep "error:" | head -20
```

Expected: no errors in `lsmindex.c`. Remaining errors only in `lsm_merge_worker.c`.

---

## Task 12: Update `vector.c` — remove merge worker registrations

**Files:**
- Modify: `pgvector/src/vector.c`

- [ ] **Step 1: Remove `#include "lsm_merge_worker.h"`**

Delete that line from the includes section.

- [ ] **Step 2: Remove merge worker LWLock tranche requests from `_PG_init`**

Delete these three blocks:

```c
	// reserve LWLock tranche for merge worker manager (need multiple locks for segment arrays)
	RequestNamedLWLockTranche("LSM Merge Worker", INDEX_BUF_SIZE + 1); // +1 for main manager lock
	LWLockRegisterTranche(1003, "LSM Merge Worker");
	
	// reserve LWLock tranche for merge segment bitmap locks
	// Need INDEX_BUF_SIZE * MAX_SEGMENTS_COUNT locks (one per segment per index buffer slot)
	RequestNamedLWLockTranche(LSM_MERGE_SEGMENT_BITMAP_LWTRANCHE, INDEX_BUF_SIZE * MAX_SEGMENTS_COUNT);
	LWLockRegisterTranche(LSM_MERGE_SEGMENT_BITMAP_LWTRANCHE_ID, LSM_MERGE_SEGMENT_BITMAP_LWTRANCHE);
```

And:

```c
	// reserve shared memory for merge worker manager
	RequestAddinShmemSpace(sizeof(MergeWorkerManager));
```

- [ ] **Step 3: Remove `initialize_merge_worker_manager()` from `shmem_startup`**

Delete:
```c
	// initialize merge worker manager
	initialize_merge_worker_manager();
```

- [ ] **Step 4: Remove merge worker background worker registration loop**

Delete the entire loop:
```c
	// initialize and register multiple lsm merge workers
	elog(DEBUG1, "[_PG_init]register lsm merge workers");
	for (int i = 0; i < MERGE_WORKERS_COUNT; i++)
	{
		BackgroundWorker lsm_merge_worker;
		...
		RegisterBackgroundWorker(&lsm_merge_worker);
		...
	}
	elog(DEBUG1, "[_PG_init]register lsm merge workers finished (total: %d)", MERGE_WORKERS_COUNT);
```

- [ ] **Step 5: Compile check**

```bash
cd /home/liu4127/postgresql/decoupled_pgvector/pgvector && make 2>&1 | grep "error:" | head -20
```

Expected: errors only in `lsm_merge_worker.c` (the file itself, not callers of it).

---

## Task 13: Delete merge worker files and update `Makefile`

**Files:**
- Delete: `pgvector/src/lsm_merge_worker.c`
- Delete: `pgvector/src/lsm_merge_worker.h`
- Modify: `pgvector/Makefile`

- [ ] **Step 1: Remove `lsm_merge_worker.o` from `Makefile`**

In `pgvector/Makefile`, find the `OBJS` variable and remove `src/lsm_merge_worker.o`:

```makefile
# Before:
OBJS = ... src/lsm_merge_worker.o ...
# After: remove that entry
```

- [ ] **Step 2: Delete the merge worker source files**

```bash
rm /home/liu4127/postgresql/decoupled_pgvector/pgvector/src/lsm_merge_worker.c
rm /home/liu4127/postgresql/decoupled_pgvector/pgvector/src/lsm_merge_worker.h
```

- [ ] **Step 3: Full build**

```bash
cd /home/liu4127/postgresql/decoupled_pgvector/pgvector && make 2>&1
```

Expected: clean build with no errors. Warnings about unused variables are acceptable at this stage.

---

## Task 14: Build verification and functional test

**Files:** None (verification only)

- [ ] **Step 1: Clean build**

```bash
cd /home/liu4127/postgresql/decoupled_pgvector/pgvector && make clean && make 2>&1
```

Expected: `vector.so` produced with no errors.

- [ ] **Step 2: Install**

```bash
cd /home/liu4127/postgresql/decoupled_pgvector/pgvector && ./install_pgvector.sh
```

Expected: files copied to PostgreSQL install tree.

- [ ] **Step 3: Restart PostgreSQL and verify background workers**

Restart the PostgreSQL server, then check the process list:

```bash
ps aux | grep -E "VectorIndexWorker|LSMBackgroundWorker|LSMMergeWorker"
```

Expected: `VectorIndexWorker` and `LSMBackgroundWorker` are present. **No** `LSMMergeWorker` processes exist.

- [ ] **Step 4: Basic smoke test**

```sql
-- In psql connected to a test database:
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE smoke (id serial, v vector(4));
CREATE INDEX smoke_hnsw ON smoke USING hnsw (v vector_l2_ops);
INSERT INTO smoke (v) SELECT array_fill(random()::float4, ARRAY[4])::vector(4)
  FROM generate_series(1, 1000);
SELECT id, v <-> '[0.1, 0.2, 0.3, 0.4]' AS dist
FROM smoke
ORDER BY dist
LIMIT 5;
```

Expected: query returns 5 rows without error.

- [ ] **Step 5: Vacuum test**

```sql
DELETE FROM smoke WHERE id % 10 = 0;
VACUUM smoke;
SELECT COUNT(*) FROM smoke;
```

Expected: `VACUUM` completes without error; count is 900.

- [ ] **Step 6: Run existing Python concurrency test**

```bash
cd /home/liu4127/postgresql/decoupled_pgvector/pgvector/test
python3 concurrent_insert_query.py
```

Expected: test completes without assertion errors or crashes.
