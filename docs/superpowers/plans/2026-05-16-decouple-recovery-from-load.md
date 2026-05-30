# Decouple LSM Recovery from Index Load — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Swap the order of recovery and index-load inside `load_lsm_index_internal`, introduce a `WRITABLE → QUERYABLE` state split on `LSMIndexBufferSlot`, and gate worker maintenance tasks against the new state so it is safe to keep a slot recovered-but-not-loaded.

**Architecture:** Recovery becomes a disk-only pass that runs in `IndexLoadWorker` and ends at the new `LSM_SLOT_WRITABLE` state. The segment-pool load is deferred until a reader arrives, at which point a CAS-driven helper transitions `WRITABLE → LOADING_INDEX → QUERYABLE` and calls `index_load_blocking`. Every worker-side maintenance task (vacuum, merge, rebuild, adopt) checks the slot state and no-ops when the slot is not yet queryable.

**Tech Stack:** C (PostgreSQL extension built with PGXS), pthread/atomics for shared state, condition variables for slot transitions.

**Spec:** [docs/superpowers/specs/2026-05-16-decouple-recovery-from-load-design.md](../specs/2026-05-16-decouple-recovery-from-load-design.md).

**Project conventions:**
- This project does not run `git commit` or `git add` from inside agents. Every task ends with a **Commit checkpoint** that gives suggested commit-message text; the maintainer commits manually outside the working tree.
- Functional verification is manual (per spec §8). Each task's verification step is a `make` build check plus a description of what to inspect.
- All `make` commands run from `pgvector/` (the extension root).

---

## File Map

| File | Responsibility | Change kind |
|---|---|---|
| `pgvector/src/lsmindex.h` | State enum, predicates, function declarations | Modify |
| `pgvector/src/lsmindex.c` | Recovery body, register helpers, on-demand load helper, read-path wrapper | Modify (rename + body changes + new functions) |
| `pgvector/src/index_load_worker.c` | Call site of the renamed recovery function + log strings | Modify |
| `pgvector/src/standby_memtable.c` | Slot-state probes used by standby redo | Modify |
| `pgvector/src/vector_index_worker.c` | Worker-side maintenance dispatch + merge-thread scan | Modify (add gates) |

No new files. No header rename. No file removed.

---

## Task 1: Add `LSMSlotState` enum and predicates in `lsmindex.h`

**Files:**
- Modify: `pgvector/src/lsmindex.h:166-174` (LSMIndexBufferSlot definition + nearby)

This task adds the new types but does not touch any state-check call site. Subsequent tasks migrate call sites one file at a time.

- [ ] **Step 1: Open `pgvector/src/lsmindex.h` and locate `LSMIndexBufferSlot`**

Confirm the current shape at lines 166-174:

```c
typedef struct LSMIndexBufferSlot
{
    pg_atomic_uint32 valid;      /* 0=free, 1=loaded, 2=load-in-progress */
    ConditionVariable load_cv;   /* backends sleep here; worker broadcasts on done/fail */
    pg_atomic_uint32 load_error; /* 0=ok, 1=failed; set before broadcast */
    Oid request_db_oid;
    Oid request_db_userid;
    LSMIndexData lsmIndex;
}   LSMIndexBufferSlot;
```

- [ ] **Step 2: Insert the `LSMSlotState` enum and predicates immediately before `LSMIndexBufferSlot`**

Edit `pgvector/src/lsmindex.h`. Find:

```c
typedef struct LSMIndexBufferSlot
{
    pg_atomic_uint32 valid;      /* 0=free, 1=loaded, 2=load-in-progress */
```

Replace with:

```c
/*
 * LSMSlotState — lifecycle of an LSMIndexBufferSlot.
 *
 *   FREE          slot is not in use.
 *   RECOVERING    IndexLoadWorker is running recover_lsm_index_internal.
 *   WRITABLE      recovery done. Memtables are usable. FlushedSegmentPool
 *                 is NOT initialized — no segment search is possible yet.
 *   LOADING_INDEX a reader is calling index_load_blocking to populate the pool.
 *   QUERYABLE     pool is initialized; search is allowed.
 *
 * Numeric values are chosen so that the natural progression matches the
 * lifecycle order, but callers SHOULD use the predicates below rather than
 * relying on numeric ordering.
 */
typedef enum LSMSlotState {
    LSM_SLOT_FREE          = 0,
    LSM_SLOT_RECOVERING    = 1,
    LSM_SLOT_WRITABLE      = 2,
    LSM_SLOT_LOADING_INDEX = 3,
    LSM_SLOT_QUERYABLE     = 4
} LSMSlotState;

static inline bool
is_writable(uint32 v)
{
    return v == (uint32) LSM_SLOT_WRITABLE
        || v == (uint32) LSM_SLOT_LOADING_INDEX
        || v == (uint32) LSM_SLOT_QUERYABLE;
}

static inline bool
is_queryable(uint32 v)
{
    return v == (uint32) LSM_SLOT_QUERYABLE;
}

typedef struct LSMIndexBufferSlot
{
    pg_atomic_uint32 valid;      /* LSMSlotState; see enum above */
```

(Only the comment on the `valid` field changes inside the struct. The rest of the struct stays identical.)

- [ ] **Step 3: Build**

Run from `pgvector/`:

```bash
make
```

Expected: clean build. No new warnings. If `is_writable` / `is_queryable` collide with an existing identifier elsewhere, rename them (e.g., `lsm_state_is_writable`) — grep first to confirm no collisions:

```bash
grep -rn 'is_writable\|is_queryable' src/ | grep -v lsmindex.h
```

Expected: no hits before this change.

- [ ] **Step 4: Commit checkpoint**

Suggested message:

```
lsmindex: introduce LSMSlotState enum and writable/queryable predicates

Adds the state machine vocabulary; no behavior change. Callers are still
using raw integer state values and will be migrated in subsequent commits.
```

The maintainer commits manually.

---

## Task 2: Rename `load_lsm_index_internal` → `recover_lsm_index_internal`

**Files:**
- Modify: `pgvector/src/lsmindex.h:193` (declaration)
- Modify: `pgvector/src/lsmindex.c:526-957` (definition)
- Modify: `pgvector/src/index_load_worker.c:166` (call site) plus comment/log strings at lines 10, 167, 175
- Modify (comments only, for consistency): `pgvector/src/vector_index_worker.c:306`, `pgvector/src/segment_adoption.h:28`, `pgvector/src/replication_rmgr.c:159`, `pgvector/src/segment_fetcher.c:443`, `pgvector/src/lsmindex_io.h:52`

This is a pure rename — the body and log prefix strings inside the function will be edited in Task 3 to match.

- [ ] **Step 1: Update the declaration in `lsmindex.h`**

Edit `pgvector/src/lsmindex.h:193`. Find:

```c
void load_lsm_index_internal(Oid index_relid, uint32_t slot_idx);
```

Replace with:

```c
void recover_lsm_index_internal(Oid index_relid, uint32_t slot_idx);
```

- [ ] **Step 2: Update the definition in `lsmindex.c`**

Edit `pgvector/src/lsmindex.c`. Find (line 526-527):

```c
void
load_lsm_index_internal(Oid index_relid, uint32_t slot_idx)
```

Replace with:

```c
void
recover_lsm_index_internal(Oid index_relid, uint32_t slot_idx)
```

Inside the function body, find and replace **each** of the log-prefix strings (do not use `replace_all` blindly — match the exact strings):

| Find | Replace with |
|---|---|
| `"[load_lsm_index_internal] loading index: relId = %u to slot %u"` | `"[recover_lsm_index_internal] recovering index: relId = %u to slot %u"` |
| `"[load_lsm_index_internal] Failed to read LSM index metadata for index %u"` | `"[recover_lsm_index_internal] Failed to read LSM index metadata for index %u"` |
| `"[load_lsm_index_internal] Vacuumed segment %u-%u v%u: marked all sids > %u as deleted, new delete_count=%u"` | `"[recover_lsm_index_internal] Vacuumed segment %u-%u v%u: marked all sids > %u as deleted, new delete_count=%u"` |
| `"[load_lsm_index_internal] In recovery step 2, checking segments that include sid in sids array"` | `"[recover_lsm_index_internal] In recovery step 2, checking segments that include sid in sids array"` |
| `"[load_lsm_index_internal] In recovery step 2, processing segment %u-%u"` | `"[recover_lsm_index_internal] In recovery step 2, processing segment %u-%u"` |
| `"[load_lsm_index_internal] Vacuumed segment %u-%u v%u: marked vectors with missing tids as deleted, new delete_count=%u"` | `"[recover_lsm_index_internal] Vacuumed segment %u-%u v%u: marked vectors with missing tids as deleted, new delete_count=%u"` |
| `"[load_lsm_index_internal] Recovery step 1&2 overhead: %.3f ms"` | `"[recover_lsm_index_internal] Recovery step 1&2 overhead: %.3f ms"` |
| `"[load_lsm_index_internal] Recovery step 3 overhead: %.3f ms"` | `"[recover_lsm_index_internal] Recovery step 3 overhead: %.3f ms"` |
| `"[load_lsm_index_internal] failed to fetch the vector data in the LSM index recovery phase, potential table pruning detected"` | `"[recover_lsm_index_internal] failed to fetch the vector data in the LSM index recovery phase, potential table pruning detected"` |
| `"[load_lsm_index_internal] no free slot to register a new memtable"` | `"[recover_lsm_index_internal] no free slot to register a new memtable"` |
| `"[load_lsm_index_internal] consistency check failed for LSM index %u"` | `"[recover_lsm_index_internal] consistency check failed for LSM index %u"` (this one is inside a commented-out block — update for consistency) |
| `"[load_lsm_index_internal] successfully loaded LSM index %u"` | `"[recover_lsm_index_internal] successfully recovered LSM index %u"` |

To find any leftovers, run:

```bash
grep -n 'load_lsm_index_internal' src/lsmindex.c
```

Expected: no hits.

- [ ] **Step 3: Update the caller in `index_load_worker.c`**

Edit `pgvector/src/index_load_worker.c`. Find at line 166:

```c
                    load_lsm_index_internal(slot->lsmIndex.indexRelId, (uint32_t) i);
                    /* load_lsm_index_internal sets valid=1 itself */
```

Replace with:

```c
                    recover_lsm_index_internal(slot->lsmIndex.indexRelId, (uint32_t) i);
                    /* recover_lsm_index_internal sets valid to LSM_SLOT_WRITABLE itself */
```

Find at line 10 (file header comment block):

```c
 * load_lsm_index_internal(), sets valid=1 (or load_error=1 + valid=0 on
```

Replace with:

```c
 * recover_lsm_index_internal(), sets valid=LSM_SLOT_WRITABLE (or load_error=1 + valid=0 on
```

Find at line 175 (warning message):

```c
                         "[IndexLoadWorker] load_lsm_index_internal failed for index %u slot %d",
```

Replace with:

```c
                         "[IndexLoadWorker] recover_lsm_index_internal failed for index %u slot %d",
```

- [ ] **Step 4: Update remaining cross-file comments for consistency**

These are comments only — no behavior depends on them. Update so future readers do not get confused.

`pgvector/src/vector_index_worker.c:306` — find:

```c
                         *                       for load_lsm_index_internal")
```

Replace with:

```c
                         *                       for recover_lsm_index_internal")
```

`pgvector/src/segment_adoption.h:28` — find:

```
 *                      Files persist on disk; load_lsm_index_internal will
```

Replace with:

```
 *                      Files persist on disk; recover_lsm_index_internal will
```

`pgvector/src/replication_rmgr.c:159` — find:

```c
        return;  /* primary crash recovery — load_lsm_index_internal handles it */
```

Replace with:

```c
        return;  /* primary crash recovery — recover_lsm_index_internal handles it */
```

`pgvector/src/segment_fetcher.c:443` — find:

```c
     * at CREATE INDEX time on the primary; load_lsm_index_internal reads it
```

Replace with:

```c
     * at CREATE INDEX time on the primary; recover_lsm_index_internal reads it
```

`pgvector/src/lsmindex_io.h:52` — find:

```c
 * read by load_lsm_index_internal on first-touch. The side channel must
```

Replace with:

```c
 * read by recover_lsm_index_internal on first-touch. The side channel must
```

- [ ] **Step 5: Verify no `load_lsm_index_internal` references remain**

Run:

```bash
grep -rn 'load_lsm_index_internal' src/
```

Expected: zero hits.

- [ ] **Step 6: Build**

Run from `pgvector/`:

```bash
make
```

Expected: clean build, no warnings.

- [ ] **Step 7: Commit checkpoint**

Suggested message:

```
lsmindex: rename load_lsm_index_internal -> recover_lsm_index_internal

Pure rename. The function body still calls index_load_blocking and still
publishes valid=1 at the end; subsequent commits split off the load phase.
```

---

## Task 3: Remove the two `SEGMENT_UPDATE_VACUUM` round-trips inside recovery steps 1 and 2

**Files:**
- Modify: `pgvector/src/lsmindex.c:643-646` (recovery step 1)
- Modify: `pgvector/src/lsmindex.c:803-806` (recovery step 2)

These calls update the in-memory bitmap of segments that the worker has just loaded. Once Task 4 removes the load from this function, the segments will not be in the pool yet, so these calls would no-op anyway — and the disk-side bitmap subversion has already been written. Removing them first keeps the diff focused.

- [ ] **Step 1: Remove the recovery step 1 VACUUM call**

Edit `pgvector/src/lsmindex.c`. Find lines 642-647:

```c
                    write_bitmap_file_with_subversion(index_relid, start_sid_disk, end_sid_disk, 
                                                      version, next_subversion, bitmap, bitmap_size, delete_count);
                    
                    // FIXME: can be done in parallel
                    // Notify index worker to update with SEGMENT_UPDATE_VACUUM
                    (void) segment_update_blocking(slot_idx, index_relid, SEGMENT_UPDATE_VACUUM,
                                                   start_sid_disk, end_sid_disk, 0);
                    
                    elog(DEBUG1, "[recover_lsm_index_internal] Vacuumed segment %u-%u v%u: marked all sids > %u as deleted, new delete_count=%u",
```

Replace with:

```c
                    write_bitmap_file_with_subversion(index_relid, start_sid_disk, end_sid_disk, 
                                                      version, next_subversion, bitmap, bitmap_size, delete_count);
                    
                    /*
                     * No SEGMENT_UPDATE_VACUUM round-trip here: index_load_blocking
                     * has not run yet, and when it does it will pick up this new
                     * bitmap subversion via load_bitmap_file(... LOAD_LATEST_VERSION ...).
                     */
                    
                    elog(DEBUG1, "[recover_lsm_index_internal] Vacuumed segment %u-%u v%u: marked all sids > %u as deleted, new delete_count=%u",
```

(Note: the log-message string already uses `recover_lsm_index_internal` from Task 2.)

- [ ] **Step 2: Remove the recovery step 2 VACUUM call**

Edit `pgvector/src/lsmindex.c`. Find lines 801-809:

```c
                        write_bitmap_file_with_subversion(index_relid, start_sid_disk, end_sid_disk, 
                                                          version, next_subversion, bitmap, bitmap_size, delete_count);
                        
                        // Notify index worker to update the segment with SEGMENT_UPDATE_VACUUM
                        (void) segment_update_blocking(slot_idx, index_relid, SEGMENT_UPDATE_VACUUM,
                                                       start_sid_disk, end_sid_disk, 0);
                        
                        elog(DEBUG1, "[recover_lsm_index_internal] Vacuumed segment %u-%u v%u: marked vectors with missing tids as deleted, new delete_count=%u",
```

Replace with:

```c
                        write_bitmap_file_with_subversion(index_relid, start_sid_disk, end_sid_disk, 
                                                          version, next_subversion, bitmap, bitmap_size, delete_count);
                        
                        /*
                         * No SEGMENT_UPDATE_VACUUM round-trip here: same reasoning
                         * as recovery step 1. The bitmap subversion on disk is the
                         * source of truth; the next load reads it directly.
                         */
                        
                        elog(DEBUG1, "[recover_lsm_index_internal] Vacuumed segment %u-%u v%u: marked vectors with missing tids as deleted, new delete_count=%u",
```

- [ ] **Step 3: Verify no `segment_update_blocking` calls remain inside `recover_lsm_index_internal`**

Run:

```bash
awk '/^recover_lsm_index_internal/,/^[a-zA-Z].*\(/' src/lsmindex.c | grep -n 'segment_update_blocking' || echo "none in function body"
```

(If the awk pattern misbehaves on the file, just open `src/lsmindex.c` in an editor and confirm visually that the function body from `void recover_lsm_index_internal` up to its closing `}` contains no `segment_update_blocking` calls.)

Expected: no hits inside the function body.

- [ ] **Step 4: Build**

```bash
make
```

Expected: clean build. The `slot_idx` parameter is now unused inside the recovery steps but still used elsewhere in the function (memtable construction); should not produce an unused-variable warning.

- [ ] **Step 5: Commit checkpoint**

Suggested message:

```
recover_lsm_index_internal: drop SEGMENT_UPDATE_VACUUM round-trips

Recovery writes new bitmap subversions to disk; the worker no longer needs
to merge them into an in-memory pool that does not yet exist (after Task 4)
or that was just initialized milliseconds ago (today). The on-disk
subversion is authoritative; the next index_load_blocking picks it up via
LOAD_LATEST_VERSION.
```

---

## Task 4: Remove `index_load_blocking` from recovery and publish `LSM_SLOT_WRITABLE` at the end

**Files:**
- Modify: `pgvector/src/lsmindex.c:551` (the index_load_blocking call)
- Modify: `pgvector/src/lsmindex.c:953-956` (the terminal state write)

After this task, `recover_lsm_index_internal` is pure recovery. Readers will trigger the load later via the helper added in Task 6.

- [ ] **Step 1: Remove the `index_load_blocking` call**

Edit `pgvector/src/lsmindex.c`. Find lines 549-552:

```c
    SegmentId max_memtable_sid = sids[0] == status_growing_sid ? status_growing_sid : status_growing_sid - 1;

    // Load all flushed segments from disk via vector index worker
    index_load_blocking(index_relid, slot_idx);

    LSMIndex lsm = &SharedLSMIndexBuffer->slots[slot_idx].lsmIndex;
```

Replace with:

```c
    SegmentId max_memtable_sid = sids[0] == status_growing_sid ? status_growing_sid : status_growing_sid - 1;

    /*
     * No index_load_blocking here. The FlushedSegmentPool is populated
     * on demand by the first reader via ensure_index_loaded(); recovery
     * leaves the slot in LSM_SLOT_WRITABLE so writers and standby redo
     * callbacks can proceed without paying segment-load latency.
     */

    LSMIndex lsm = &SharedLSMIndexBuffer->slots[slot_idx].lsmIndex;
```

- [ ] **Step 2: Update the terminal state write**

Edit `pgvector/src/lsmindex.c`. Find lines 951-956:

```c
    table_close(heap_rel, AccessShareLock);
    index_close(index, AccessShareLock);
    // Mark as fully loaded (valid = 1) - use write barrier to ensure all writes are visible
    pg_atomic_write_u32(&slot->valid, 1);

    elog(DEBUG1, "[recover_lsm_index_internal] successfully recovered LSM index %u", index_relid);
}
```

Replace with:

```c
    table_close(heap_rel, AccessShareLock);
    index_close(index, AccessShareLock);
    /*
     * Publish LSM_SLOT_WRITABLE. The FlushedSegmentPool is NOT yet
     * initialized — readers must call ensure_index_loaded() to drive the
     * WRITABLE -> LOADING_INDEX -> QUERYABLE transition. Writers and
     * standby redo callbacks operate on the memtable only and may proceed.
     */
    pg_write_barrier();
    pg_atomic_write_u32(&slot->valid, (uint32) LSM_SLOT_WRITABLE);
    ConditionVariableBroadcast(&slot->load_cv);

    elog(DEBUG1, "[recover_lsm_index_internal] successfully recovered LSM index %u (WRITABLE)", index_relid);
}
```

(The `pg_write_barrier()` was already implicit before the `pg_atomic_write_u32` via the existing barrier at line 942; making it explicit and adding the broadcast aligns with the helper added in later tasks. The slot's `load_cv` was previously broadcast by the `IndexLoadWorker` outer wrapper after this function returned — keeping a broadcast inside the function is harmless because the outer code's broadcast still runs on success and on error.)

- [ ] **Step 3: Build**

```bash
make
```

Expected: clean build.

- [ ] **Step 4: Inspect**

Sanity check the function body. Open `src/lsmindex.c` and confirm:
- `recover_lsm_index_internal` no longer calls `index_load_blocking`.
- The terminal state set is `LSM_SLOT_WRITABLE`, not `1`.
- No `segment_update_blocking(... SEGMENT_UPDATE_VACUUM ...)` calls remain inside the function body.

- [ ] **Step 5: Commit checkpoint**

Suggested message:

```
recover_lsm_index_internal: stop populating FlushedSegmentPool

The function now ends at LSM_SLOT_WRITABLE; segment-pool initialization
is deferred to ensure_index_loaded() (added in a follow-up commit), which
runs on first reader access. Callers that still expect a fully-loaded
slot will be migrated next.
```

---

## Task 5: Migrate `valid == 1` / `valid == 2` checks in `lsmindex.c`

**Files:**
- Modify: `pgvector/src/lsmindex.c:184-286` (`register_lsm_index`)
- Modify: `pgvector/src/lsmindex.c:960-988` (`get_lsm_index_idx_no_loading`)
- Modify: `pgvector/src/lsmindex.c:992-1026` (`get_lsm_index_idx`)
- Modify: `pgvector/src/lsmindex.c:1029-1067` (`get_lsm_index`)

After this task, every state probe inside `lsmindex.c` uses the named constants and `is_writable` predicate.

Note: this task migrates **writer-side** semantics: callers wait for `is_writable(v)` (i.e., recovery is done). Readers are wired to additionally call `ensure_index_loaded` in Task 8; for now `get_lsm_index` returns at `WRITABLE` and `search_lsm_index` would temporarily lose query correctness if invoked. This is **expected** — the codebase does not crash, but search results during this interim window are unreliable. The maintainer should defer search-path testing until Task 8 lands.

- [ ] **Step 1: Migrate `register_lsm_index`**

Edit `pgvector/src/lsmindex.c`. Find the function body starting at line 184. Make the following edits.

Find at line 192:

```c
        if (valid == 0 || SharedLSMIndexBuffer->slots[i].lsmIndex.indexRelId != index_relid)
            continue;

        LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[i];
        LWLockRelease(SharedLSMIndexBuffer->lock);

        if (valid == 1)
            return i;  /* already fully loaded */

        /* valid == 2: worker is loading — wait on CV until done or failed */
        PG_TRY();
        {
            ConditionVariablePrepareToSleep(&slot->load_cv);
            while (pg_atomic_read_u32(&slot->valid) == 2)
                ConditionVariableSleep(&slot->load_cv, PG_WAIT_EXTENSION);
            ConditionVariableCancelSleep();
        }
        PG_CATCH();
        {
            ConditionVariableCancelSleep();
            PG_RE_THROW();
        }
        PG_END_TRY();

        if (pg_atomic_read_u32(&slot->valid) == 1 &&
            pg_atomic_read_u32(&slot->load_error) == 0)
            return i;
```

Replace with:

```c
        if (valid == (uint32) LSM_SLOT_FREE ||
            SharedLSMIndexBuffer->slots[i].lsmIndex.indexRelId != index_relid)
            continue;

        LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[i];
        LWLockRelease(SharedLSMIndexBuffer->lock);

        if (is_writable(valid))
            return i;  /* recovery already done; load (if needed) is the reader's job */

        /* RECOVERING: worker is running recovery — wait on CV until done or failed */
        PG_TRY();
        {
            ConditionVariablePrepareToSleep(&slot->load_cv);
            while (pg_atomic_read_u32(&slot->valid) == (uint32) LSM_SLOT_RECOVERING)
                ConditionVariableSleep(&slot->load_cv, PG_WAIT_EXTENSION);
            ConditionVariableCancelSleep();
        }
        PG_CATCH();
        {
            ConditionVariableCancelSleep();
            PG_RE_THROW();
        }
        PG_END_TRY();

        if (is_writable(pg_atomic_read_u32(&slot->valid)) &&
            pg_atomic_read_u32(&slot->load_error) == 0)
            return i;
```

Find at line 230-235:

```c
    /* Claim a free slot: CAS 0→2 while holding the buffer lock */
    int slot_num = -1;
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        uint32 expected = 0;
        if (pg_atomic_compare_exchange_u32(&SharedLSMIndexBuffer->slots[i].valid, &expected, 2))
        {
            slot_num = i;
            break;
        }
    }
```

Replace with:

```c
    /* Claim a free slot: CAS FREE→RECOVERING while holding the buffer lock */
    int slot_num = -1;
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        uint32 expected = (uint32) LSM_SLOT_FREE;
        if (pg_atomic_compare_exchange_u32(&SharedLSMIndexBuffer->slots[i].valid,
                                           &expected, (uint32) LSM_SLOT_RECOVERING))
        {
            slot_num = i;
            break;
        }
    }
```

Find at line 256-258:

```c
    if (wpgprocno < 0)
    {
        /* Worker not running — reset slot and error */
        pg_atomic_write_u32(&slot->valid, 0);
        ConditionVariableBroadcast(&slot->load_cv);
        elog(ERROR, "[register_lsm_index] IndexLoadWorker is not running");
    }
```

Replace with:

```c
    if (wpgprocno < 0)
    {
        /* Worker not running — reset slot and error */
        pg_atomic_write_u32(&slot->valid, (uint32) LSM_SLOT_FREE);
        ConditionVariableBroadcast(&slot->load_cv);
        elog(ERROR, "[register_lsm_index] IndexLoadWorker is not running");
    }
```

Find at line 264-280:

```c
    /* Block until the worker finishes or fails */
    PG_TRY();
    {
        ConditionVariablePrepareToSleep(&slot->load_cv);
        while (pg_atomic_read_u32(&slot->valid) == 2)
            ConditionVariableSleep(&slot->load_cv, PG_WAIT_EXTENSION);
        ConditionVariableCancelSleep();
    }
    PG_CATCH();
    {
        ConditionVariableCancelSleep();
        PG_RE_THROW();
    }
    PG_END_TRY();

    if (pg_atomic_read_u32(&slot->valid) == 1 &&
        pg_atomic_read_u32(&slot->load_error) == 0)
        return slot_num;
```

Replace with:

```c
    /* Block until recovery finishes or fails */
    PG_TRY();
    {
        ConditionVariablePrepareToSleep(&slot->load_cv);
        while (pg_atomic_read_u32(&slot->valid) == (uint32) LSM_SLOT_RECOVERING)
            ConditionVariableSleep(&slot->load_cv, PG_WAIT_EXTENSION);
        ConditionVariableCancelSleep();
    }
    PG_CATCH();
    {
        ConditionVariableCancelSleep();
        PG_RE_THROW();
    }
    PG_END_TRY();

    if (is_writable(pg_atomic_read_u32(&slot->valid)) &&
        pg_atomic_read_u32(&slot->load_error) == 0)
        return slot_num;
```

- [ ] **Step 2: Migrate `get_lsm_index_idx_no_loading`**

Edit `pgvector/src/lsmindex.c`. Find at lines 960-988:

```c
int
get_lsm_index_idx_no_loading(Oid index_relid)
{
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        uint32 valid = pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid);
        if (valid == 0 || SharedLSMIndexBuffer->slots[i].lsmIndex.indexRelId != index_relid)
            continue;

        if (valid == 1)
            return i;

        /* valid == 2: wait for the IndexLoadWorker */
        LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[i];
        ConditionVariablePrepareToSleep(&slot->load_cv);
        while (pg_atomic_read_u32(&slot->valid) == 2)
            ConditionVariableSleep(&slot->load_cv, PG_WAIT_EXTENSION);
        ConditionVariableCancelSleep();

        if (pg_atomic_read_u32(&slot->valid) == 1 &&
            pg_atomic_read_u32(&slot->load_error) == 0)
            return i;

        elog(DEBUG1, "[get_lsm_index_idx_no_loading] index %u loading failed", index_relid);
        return -1;
    }

    elog(DEBUG1, "[get_lsm_index_idx_no_loading] index %u not in buffer", index_relid);
    return -1;
}
```

Replace with:

```c
int
get_lsm_index_idx_no_loading(Oid index_relid)
{
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        uint32 valid = pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid);
        if (valid == (uint32) LSM_SLOT_FREE ||
            SharedLSMIndexBuffer->slots[i].lsmIndex.indexRelId != index_relid)
            continue;

        if (is_writable(valid))
            return i;

        /* RECOVERING: wait for the IndexLoadWorker to finish recovery */
        LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[i];
        ConditionVariablePrepareToSleep(&slot->load_cv);
        while (pg_atomic_read_u32(&slot->valid) == (uint32) LSM_SLOT_RECOVERING)
            ConditionVariableSleep(&slot->load_cv, PG_WAIT_EXTENSION);
        ConditionVariableCancelSleep();

        if (is_writable(pg_atomic_read_u32(&slot->valid)) &&
            pg_atomic_read_u32(&slot->load_error) == 0)
            return i;

        elog(DEBUG1, "[get_lsm_index_idx_no_loading] index %u recovery failed", index_relid);
        return -1;
    }

    elog(DEBUG1, "[get_lsm_index_idx_no_loading] index %u not in buffer", index_relid);
    return -1;
}
```

- [ ] **Step 3: Migrate `get_lsm_index_idx`**

Edit `pgvector/src/lsmindex.c`. Find at lines 994-1020 (the loop body inside `get_lsm_index_idx`):

```c
    /* Fast path: scan for an already-registered slot */
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        uint32 valid = pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid);
        if (valid == 0 || SharedLSMIndexBuffer->slots[i].lsmIndex.indexRelId != index->rd_id)
            continue;

        if (valid == 1)
            return i;

        /* valid == 2: worker is loading — wait on CV */
        {
            LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[i];
            ConditionVariablePrepareToSleep(&slot->load_cv);
            while (pg_atomic_read_u32(&slot->valid) == 2)
                ConditionVariableSleep(&slot->load_cv, PG_WAIT_EXTENSION);
            ConditionVariableCancelSleep();

            if (pg_atomic_read_u32(&slot->valid) == 1 &&
                pg_atomic_read_u32(&slot->load_error) == 0)
                return i;

            elog(ERROR, "[get_lsm_index_idx] index %u loading failed (load_error=%u, valid=%u)",
                 index->rd_id,
                 pg_atomic_read_u32(&slot->load_error),
                 pg_atomic_read_u32(&slot->valid));
        }
    }
```

Replace with:

```c
    /* Fast path: scan for an already-registered slot */
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        uint32 valid = pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid);
        if (valid == (uint32) LSM_SLOT_FREE ||
            SharedLSMIndexBuffer->slots[i].lsmIndex.indexRelId != index->rd_id)
            continue;

        if (is_writable(valid))
            return i;

        /* RECOVERING: worker is running recovery — wait on CV */
        {
            LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[i];
            ConditionVariablePrepareToSleep(&slot->load_cv);
            while (pg_atomic_read_u32(&slot->valid) == (uint32) LSM_SLOT_RECOVERING)
                ConditionVariableSleep(&slot->load_cv, PG_WAIT_EXTENSION);
            ConditionVariableCancelSleep();

            if (is_writable(pg_atomic_read_u32(&slot->valid)) &&
                pg_atomic_read_u32(&slot->load_error) == 0)
                return i;

            elog(ERROR, "[get_lsm_index_idx] index %u recovery failed (load_error=%u, valid=%u)",
                 index->rd_id,
                 pg_atomic_read_u32(&slot->load_error),
                 pg_atomic_read_u32(&slot->valid));
        }
    }
```

- [ ] **Step 4: Migrate `get_lsm_index`**

Edit `pgvector/src/lsmindex.c`. Find at lines 1035-1061 (the loop body inside `get_lsm_index`):

```c
    /* Fast path: scan for an already-registered slot */
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        uint32 valid = pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid);
        if (valid == 0 || SharedLSMIndexBuffer->slots[i].lsmIndex.indexRelId != index_relid)
            continue;

        if (valid == 1)
            return &SharedLSMIndexBuffer->slots[i].lsmIndex;

        /* valid == 2: worker is loading — wait on CV */
        {
            LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[i];
            ConditionVariablePrepareToSleep(&slot->load_cv);
            while (pg_atomic_read_u32(&slot->valid) == 2)
                ConditionVariableSleep(&slot->load_cv, PG_WAIT_EXTENSION);
            ConditionVariableCancelSleep();

            if (pg_atomic_read_u32(&slot->valid) == 1 &&
                pg_atomic_read_u32(&slot->load_error) == 0)
                return &slot->lsmIndex;

            elog(ERROR, "[get_lsm_index] index %u loading failed (load_error=%u, valid=%u)",
                 index_relid,
                 pg_atomic_read_u32(&slot->load_error),
                 pg_atomic_read_u32(&slot->valid));
        }
    }
```

Replace with:

```c
    /* Fast path: scan for an already-registered slot */
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        uint32 valid = pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid);
        if (valid == (uint32) LSM_SLOT_FREE ||
            SharedLSMIndexBuffer->slots[i].lsmIndex.indexRelId != index_relid)
            continue;

        if (is_writable(valid))
            return &SharedLSMIndexBuffer->slots[i].lsmIndex;

        /* RECOVERING: worker is running recovery — wait on CV */
        {
            LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[i];
            ConditionVariablePrepareToSleep(&slot->load_cv);
            while (pg_atomic_read_u32(&slot->valid) == (uint32) LSM_SLOT_RECOVERING)
                ConditionVariableSleep(&slot->load_cv, PG_WAIT_EXTENSION);
            ConditionVariableCancelSleep();

            if (is_writable(pg_atomic_read_u32(&slot->valid)) &&
                pg_atomic_read_u32(&slot->load_error) == 0)
                return &slot->lsmIndex;

            elog(ERROR, "[get_lsm_index] index %u recovery failed (load_error=%u, valid=%u)",
                 index_relid,
                 pg_atomic_read_u32(&slot->load_error),
                 pg_atomic_read_u32(&slot->valid));
        }
    }
```

- [ ] **Step 5: Verify no raw `valid == 1` / `valid == 2` literals remain in `lsmindex.c`**

```bash
grep -n 'valid) == 1\|valid) == 2\|valid == 1\|valid == 2\|&expected, 1)\|&expected, 2)' src/lsmindex.c
```

Expected: zero hits. (CAS calls now use named constants; comparisons use predicates or named constants.)

- [ ] **Step 6: Build**

```bash
make
```

Expected: clean build.

- [ ] **Step 7: Commit checkpoint**

Suggested message:

```
lsmindex.c: migrate slot-state checks to LSMSlotState predicates

register_lsm_index, get_lsm_index, get_lsm_index_idx, and
get_lsm_index_idx_no_loading now return at is_writable(v) (recovery done)
rather than the previous "fully loaded" semantics. Readers will be wired
to additionally wait for QUERYABLE in a follow-up commit.
```

---

## Task 6: Migrate `valid` checks in `standby_memtable.c`

**Files:**
- Modify: `pgvector/src/standby_memtable.c:17-31` (`find_loaded_slot`)
- Modify: `pgvector/src/standby_memtable.c:73-99` (`dpv_standby_wait_if_loading`)

Standby callbacks only need the memtable to be usable — they do not touch the FlushedSegmentPool. Switch their state check to `is_writable`.

- [ ] **Step 1: Migrate `find_loaded_slot`**

Edit `pgvector/src/standby_memtable.c`. Find at lines 12-31:

```c
/*
 * find_loaded_slot — return the LSMIndexBufferSlot for indexRelId if it is
 * currently valid==1 (fully loaded), else NULL.
 */
static LSMIndexBufferSlot *
find_loaded_slot(Oid indexRelId)
{
    int slot_idx = get_lsm_index_idx_no_loading(indexRelId);
    LSMIndexBufferSlot *slot;

    if (slot_idx < 0)
        return NULL;

    slot = &SharedLSMIndexBuffer->slots[slot_idx];
    if (pg_atomic_read_u32(&slot->valid) != 1)
        return NULL;

    return slot;
}
```

Replace with:

```c
/*
 * find_loaded_slot — return the LSMIndexBufferSlot for indexRelId if it is
 * currently writable (recovery done; FlushedSegmentPool may or may not be
 * initialized — standby redo only needs the memtable, not the pool).
 * Returns NULL otherwise.
 */
static LSMIndexBufferSlot *
find_loaded_slot(Oid indexRelId)
{
    int slot_idx = get_lsm_index_idx_no_loading(indexRelId);
    LSMIndexBufferSlot *slot;

    if (slot_idx < 0)
        return NULL;

    slot = &SharedLSMIndexBuffer->slots[slot_idx];
    if (!is_writable(pg_atomic_read_u32(&slot->valid)))
        return NULL;

    return slot;
}
```

- [ ] **Step 2: Migrate `dpv_standby_wait_if_loading`**

Edit `pgvector/src/standby_memtable.c`. Find at lines 73-99:

```c
/*
 * dpv_standby_wait_if_loading — if the slot for indexRelId is in
 * valid==2 (load in progress), sleep on load_cv until it settles.
 * Returns the resulting valid value, or 0 if the slot is absent.
 */
int
dpv_standby_wait_if_loading(Oid indexRelId)
{
    int slot_idx = get_lsm_index_idx_no_loading(indexRelId);
    LSMIndexBufferSlot *slot;
    uint32 observed;

    if (slot_idx < 0)
        return 0;  /* not in buffer; caller skips in-memory effect */

    slot = &SharedLSMIndexBuffer->slots[slot_idx];
    observed = pg_atomic_read_u32(&slot->valid);
    if (observed != 2)
        return (int) observed;

    /* Slow path: wait for the loader. */
    ConditionVariablePrepareToSleep(&slot->load_cv);
    for (;;)
    {
        observed = pg_atomic_read_u32(&slot->valid);
        if (observed != 2)
            break;
        ConditionVariableSleep(&slot->load_cv, PG_WAIT_EXTENSION);
    }
    ConditionVariableCancelSleep();
    return (int) observed;
}
```

Replace with:

```c
/*
 * dpv_standby_wait_if_loading — if the slot for indexRelId is in
 * LSM_SLOT_RECOVERING, sleep on load_cv until it settles.
 *
 * NOTE: this function intentionally does NOT wait on LSM_SLOT_LOADING_INDEX.
 * Standby redo callbacks only touch the memtable, which is already
 * available in any is_writable() state.
 *
 * Returns the resulting valid value, or 0 if the slot is absent.
 */
int
dpv_standby_wait_if_loading(Oid indexRelId)
{
    int slot_idx = get_lsm_index_idx_no_loading(indexRelId);
    LSMIndexBufferSlot *slot;
    uint32 observed;

    if (slot_idx < 0)
        return 0;  /* not in buffer; caller skips in-memory effect */

    slot = &SharedLSMIndexBuffer->slots[slot_idx];
    observed = pg_atomic_read_u32(&slot->valid);
    if (observed != (uint32) LSM_SLOT_RECOVERING)
        return (int) observed;

    /* Slow path: wait for recovery to finish. */
    ConditionVariablePrepareToSleep(&slot->load_cv);
    for (;;)
    {
        observed = pg_atomic_read_u32(&slot->valid);
        if (observed != (uint32) LSM_SLOT_RECOVERING)
            break;
        ConditionVariableSleep(&slot->load_cv, PG_WAIT_EXTENSION);
    }
    ConditionVariableCancelSleep();
    return (int) observed;
}
```

- [ ] **Step 3: Verify no raw `valid == 1` / `valid == 2` literals remain in `standby_memtable.c`**

```bash
grep -n 'valid) == 1\|valid) == 2\|valid != 1\|valid != 2' src/standby_memtable.c
```

Expected: zero hits.

- [ ] **Step 4: Build**

```bash
make
```

Expected: clean build.

- [ ] **Step 5: Commit checkpoint**

Suggested message:

```
standby_memtable: accept any is_writable state for redo callbacks

Standby redo callbacks only need the memtable; they never touch the
FlushedSegmentPool. Use is_writable() so callbacks can keep up with WAL
even while the index is recovered-but-not-loaded.
```

---

## Task 7: Add the `ensure_index_loaded` helper in `lsmindex.c`

**Files:**
- Modify: `pgvector/src/lsmindex.c` — add a new static helper near the other slot-state helpers (e.g., just before `get_lsm_index` at the current line ~1028)

The helper compiles but is not yet called. Task 8 wires it into `search_lsm_index`.

- [ ] **Step 1: Add the helper**

Edit `pgvector/src/lsmindex.c`. Find the function `get_lsm_index` (line 1029):

```c
LSMIndex
get_lsm_index(Relation index)
{
```

Insert the following helper immediately above (after the closing `}` of `get_lsm_index_idx` at line 1026 and the blank line that follows):

```c
/*
 * ensure_index_loaded — drive the WRITABLE -> LOADING_INDEX -> QUERYABLE
 * transition for slot_idx and block until QUERYABLE.
 *
 * Preconditions:
 *   - slot_idx is a valid index returned by register_lsm_index() / friends.
 *   - The slot is at least WRITABLE.
 *
 * Returns true if the slot reaches QUERYABLE. Returns false if a concurrent
 * load attempt failed; the caller should propagate an error (the slot
 * reverts to WRITABLE and load_error is set).
 */
static bool
ensure_index_loaded(int slot_idx)
{
    LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[slot_idx];
    uint32 v = pg_atomic_read_u32(&slot->valid);

    if (v == (uint32) LSM_SLOT_QUERYABLE)
        return true;

    if (v == (uint32) LSM_SLOT_WRITABLE)
    {
        uint32 expected = (uint32) LSM_SLOT_WRITABLE;
        if (pg_atomic_compare_exchange_u32(&slot->valid, &expected,
                                           (uint32) LSM_SLOT_LOADING_INDEX))
        {
            /* Winner: clear any stale load_error from a prior attempt. */
            pg_atomic_write_u32(&slot->load_error, 0);

            PG_TRY();
            {
                index_load_blocking(slot->lsmIndex.indexRelId, slot_idx);
                pg_write_barrier();
                pg_atomic_write_u32(&slot->valid, (uint32) LSM_SLOT_QUERYABLE);
            }
            PG_CATCH();
            {
                pg_atomic_write_u32(&slot->load_error, 1);
                pg_atomic_write_u32(&slot->valid, (uint32) LSM_SLOT_WRITABLE);
                ConditionVariableBroadcast(&slot->load_cv);
                PG_RE_THROW();
            }
            PG_END_TRY();

            ConditionVariableBroadcast(&slot->load_cv);
            return true;
        }
        /* CAS lost — someone else owns the transition; fall through to wait. */
    }

    /*
     * Wait until the slot leaves LOADING_INDEX.
     *   - QUERYABLE: success.
     *   - WRITABLE: the winner errored and reverted; we report failure.
     */
    PG_TRY();
    {
        ConditionVariablePrepareToSleep(&slot->load_cv);
        for (;;)
        {
            v = pg_atomic_read_u32(&slot->valid);
            if (v == (uint32) LSM_SLOT_QUERYABLE)
            {
                ConditionVariableCancelSleep();
                return true;
            }
            if (v == (uint32) LSM_SLOT_WRITABLE)
            {
                ConditionVariableCancelSleep();
                return false;
            }
            ConditionVariableSleep(&slot->load_cv, PG_WAIT_EXTENSION);
        }
    }
    PG_CATCH();
    {
        ConditionVariableCancelSleep();
        PG_RE_THROW();
    }
    PG_END_TRY();
}
```

- [ ] **Step 2: Build**

```bash
make
```

Expected: clean build. A `-Wunused-function` warning on `ensure_index_loaded` is **acceptable for one commit** because Task 8 will wire it. If the project's `-Werror` flag is active and this warning is fatal, add `pg_attribute_unused()` to the declaration:

```c
static bool ensure_index_loaded(int slot_idx) pg_attribute_unused();
```

(Check the Makefile for `-Werror`; if absent, the warning is informational only.)

- [ ] **Step 3: Inspect**

Confirm visually that the helper handles all three observed states (`QUERYABLE`, `WRITABLE`, `LOADING_INDEX`) and that the CAS error path resets `valid` back to `WRITABLE` *and* broadcasts.

- [ ] **Step 4: Commit checkpoint**

Suggested message:

```
lsmindex: add ensure_index_loaded helper (not yet wired)

CAS-driven WRITABLE -> LOADING_INDEX -> QUERYABLE transition. One winner
calls index_load_blocking; losers and second-arrivers wait on load_cv.
The helper is added in advance of wiring it into the read path so the
build stays green between tasks.
```

---

## Task 8: Add `get_lsm_index_for_read` and wire `search_lsm_index`

**Files:**
- Modify: `pgvector/src/lsmindex.h` (add declaration)
- Modify: `pgvector/src/lsmindex.c` — add `get_lsm_index_for_read` and call it from `search_lsm_index` (current call at line 1516)

- [ ] **Step 1: Declare `get_lsm_index_for_read` in `lsmindex.h`**

Edit `pgvector/src/lsmindex.h`. Find the existing declarations (around line 196-200):

```c
LSMIndex get_lsm_index(Relation index);
int get_lsm_index_idx_no_loading(Oid index_relid);
int get_lsm_index_idx(Relation index);
TopKTuples search_lsm_index(Relation index, const void *vector, int k, int nprobe_efs);
```

Replace with:

```c
LSMIndex get_lsm_index(Relation index);
/*
 * get_lsm_index_for_read — like get_lsm_index, but additionally drives the
 * WRITABLE -> QUERYABLE transition before returning. Use this from read
 * paths (search). Writers (insert) should keep calling get_lsm_index.
 */
LSMIndex get_lsm_index_for_read(Relation index);
int get_lsm_index_idx_no_loading(Oid index_relid);
int get_lsm_index_idx(Relation index);
TopKTuples search_lsm_index(Relation index, const void *vector, int k, int nprobe_efs);
```

- [ ] **Step 2: Define `get_lsm_index_for_read` in `lsmindex.c`**

Edit `pgvector/src/lsmindex.c`. Find the end of `get_lsm_index` (around line 1067):

```c
    /* Slow path: not in buffer — register and wait for worker to load */
    elog(DEBUG1, "[get_lsm_index] the requested lsm_index is not in the buffer");
    slot_num = register_lsm_index(index_relid);
    return &SharedLSMIndexBuffer->slots[slot_num].lsmIndex;
}
```

Insert the new function immediately after the closing brace:

```c
    /* Slow path: not in buffer — register and wait for worker to load */
    elog(DEBUG1, "[get_lsm_index] the requested lsm_index is not in the buffer");
    slot_num = register_lsm_index(index_relid);
    return &SharedLSMIndexBuffer->slots[slot_num].lsmIndex;
}

LSMIndex
get_lsm_index_for_read(Relation index)
{
    Oid index_relid = RelationGetRelid(index);
    int slot_num = -1;

    /* Resolve to a slot index; reuse the fast-path logic of get_lsm_index_idx. */
    slot_num = get_lsm_index_idx(index);
    if (slot_num < 0)
        elog(ERROR, "[get_lsm_index_for_read] failed to resolve LSM index slot for %u",
             index_relid);

    if (!ensure_index_loaded(slot_num))
        elog(ERROR, "[get_lsm_index_for_read] index %u load failed (load_error=%u)",
             index_relid,
             pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[slot_num].load_error));

    return &SharedLSMIndexBuffer->slots[slot_num].lsmIndex;
}
```

- [ ] **Step 3: Wire `search_lsm_index`**

Edit `pgvector/src/lsmindex.c`. Find at line 1514-1517:

```c
TopKTuples
search_lsm_index(Relation index, const void *vector, int k, int nprobe_efs)
{
    LSMIndex lsm = get_lsm_index(index);
    Assert(lsm);
```

Replace with:

```c
TopKTuples
search_lsm_index(Relation index, const void *vector, int k, int nprobe_efs)
{
    /*
     * Read path: must ensure the FlushedSegmentPool is populated
     * (QUERYABLE), not just that recovery is done.
     */
    LSMIndex lsm = get_lsm_index_for_read(index);
    Assert(lsm);
```

- [ ] **Step 4: Build**

```bash
make
```

Expected: clean build. The `-Wunused-function` warning on `ensure_index_loaded` (if it was raised in Task 7) should disappear.

- [ ] **Step 5: Sanity-check call sites**

The only reader call site is `search_lsm_index`. Confirm:

```bash
grep -n 'get_lsm_index_for_read' src/
```

Expected: one definition (`lsmindex.c`), one declaration (`lsmindex.h`), one caller (`lsmindex.c` inside `search_lsm_index`).

```bash
grep -n 'get_lsm_index(' src/ | grep -v 'get_lsm_index_for_read\|get_lsm_index_idx'
```

Expected: writer paths only (notably `insert_lsm_index`). No `search_lsm_index` references.

- [ ] **Step 6: Commit checkpoint**

Suggested message:

```
search_lsm_index: route through get_lsm_index_for_read

Readers now drive the WRITABLE -> QUERYABLE transition via
ensure_index_loaded() on first access; writers continue to return at
WRITABLE via get_lsm_index. This is the user-visible split between
"recovered" and "loaded" slot states.
```

---

## Task 9: Add `slot_is_queryable` helper and gate maintenance handlers in `vector_index_worker.c`

**Files:**
- Modify: `pgvector/src/vector_index_worker.c` — add helper near the top of the file (after `#include` block; pick a spot before the first task-handler use)
- Modify: handlers for `SEGMENT_UPDATE_REGULAR`, `SEGMENT_UPDATE_VACUUM`, `SEGMENT_UPDATE_REBUILD_FLAT`, `SEGMENT_UPDATE_REBUILD_DELETION`, `SEGMENT_UPDATE_MERGE`, `SEGMENT_UPDATE_ADOPT` inside the dispatcher around line 194-320.

Apply identical gating: on entry, `if (!slot_is_queryable(lsm_idx)) { set maint_status, signal backend, break; }`.

- [ ] **Step 1: Add `slot_is_queryable` helper**

Edit `pgvector/src/vector_index_worker.c`. Locate a stable spot near the top of the file just after the `#include` block (search for the first `static` declaration). Add immediately after the last `#include`:

```c
/*
 * slot_is_queryable — true iff the slot at lsm_idx is currently in
 * LSM_SLOT_QUERYABLE. Used to gate maintenance task handlers so they
 * no-op against slots that are recovered but not yet loaded.
 */
static inline bool
slot_is_queryable(int lsm_idx)
{
    if (lsm_idx < 0 || lsm_idx >= INDEX_BUF_SIZE)
        return false;
    return pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[lsm_idx].valid)
           == (uint32) LSM_SLOT_QUERYABLE;
}
```

If the file already has its own static-helper grouping convention, place the helper alongside existing ones; the exact location is not load-bearing as long as it precedes all uses.

- [ ] **Step 2: Gate `SEGMENT_UPDATE_REGULAR`**

Edit `pgvector/src/vector_index_worker.c`. Find at lines 205-219:

```c
                    case SEGMENT_UPDATE_REGULAR:
                    {
                        FlushedSegmentPool *pool_seg = get_flushed_segment_pool(lsm_idx);
                        
                        pthread_rwlock_wrlock(&pool_seg->seg_lock);
                        uint32_t seg_idx = reserve_flushed_segment(pool_seg);
                        pthread_rwlock_unlock(&pool_seg->seg_lock);
                        
                        load_and_set_segment(index_relid, seg_idx, &pool_seg->flushed_segments[seg_idx], update_task->start_sid, update_task->end_sid, LOAD_LATEST_VERSION, false);
                        
                        pthread_rwlock_wrlock(&pool_seg->seg_lock);
                        register_flushed_segment(pool_seg, seg_idx);
                        pthread_rwlock_unlock(&pool_seg->seg_lock);
                        signal_merge_pool();  /* NEW: new segment fully loaded, trigger merge scan */
                        break;
                    }
```

Replace with:

```c
                    case SEGMENT_UPDATE_REGULAR:
                    {
                        if (!slot_is_queryable(lsm_idx))
                        {
                            elog(DEBUG1, "[maintenance_worker] SEGMENT_UPDATE_REGULAR: skip — slot %d not queryable",
                                 lsm_idx);
                            vs_search_result_at(update_task->backend_pgprocno)->maint_status = 0;
                            client = &ProcGlobal->allProcs[update_task->backend_pgprocno];
                            break;
                        }

                        FlushedSegmentPool *pool_seg = get_flushed_segment_pool(lsm_idx);
                        
                        pthread_rwlock_wrlock(&pool_seg->seg_lock);
                        uint32_t seg_idx = reserve_flushed_segment(pool_seg);
                        pthread_rwlock_unlock(&pool_seg->seg_lock);
                        
                        load_and_set_segment(index_relid, seg_idx, &pool_seg->flushed_segments[seg_idx], update_task->start_sid, update_task->end_sid, LOAD_LATEST_VERSION, false);
                        
                        pthread_rwlock_wrlock(&pool_seg->seg_lock);
                        register_flushed_segment(pool_seg, seg_idx);
                        pthread_rwlock_unlock(&pool_seg->seg_lock);
                        signal_merge_pool();  /* NEW: new segment fully loaded, trigger merge scan */
                        break;
                    }
```

- [ ] **Step 3: Gate `SEGMENT_UPDATE_VACUUM`**

Edit `pgvector/src/vector_index_worker.c`. Find at lines 222-225:

```c
                    case SEGMENT_UPDATE_VACUUM:
                    {
                        fprintf(stderr, "[maintenance_worker] SEGMENT_UPDATE_VACUUM %d-%d expected_version=%u\n",
                                update_task->start_sid, update_task->end_sid, update_task->expected_version);
```

Replace with:

```c
                    case SEGMENT_UPDATE_VACUUM:
                    {
                        fprintf(stderr, "[maintenance_worker] SEGMENT_UPDATE_VACUUM %d-%d expected_version=%u\n",
                                update_task->start_sid, update_task->end_sid, update_task->expected_version);

                        if (!slot_is_queryable(lsm_idx))
                        {
                            elog(DEBUG1, "[maintenance_worker] SEGMENT_UPDATE_VACUUM: skip — slot %d not queryable",
                                 lsm_idx);
                            vs_search_result_at(update_task->backend_pgprocno)->maint_status = 0;
                            client = &ProcGlobal->allProcs[update_task->backend_pgprocno];
                            break;
                        }
```

- [ ] **Step 4: Gate `SEGMENT_UPDATE_ADOPT`**

Edit `pgvector/src/vector_index_worker.c`. Find at lines 285-294:

```c
                    case SEGMENT_UPDATE_ADOPT:
                    {
                        DpvAdoptionOutcome adopt_result;

                        fprintf(stderr,
                                "[maintenance_worker] SEGMENT_UPDATE_ADOPT %d-%d v=%u\n",
                                update_task->start_sid, update_task->end_sid,
                                update_task->expected_version);

                        adopt_result = dpv_attempt_adoption(update_task->lsm_idx,
```

Replace with:

```c
                    case SEGMENT_UPDATE_ADOPT:
                    {
                        DpvAdoptionOutcome adopt_result;

                        fprintf(stderr,
                                "[maintenance_worker] SEGMENT_UPDATE_ADOPT %d-%d v=%u\n",
                                update_task->start_sid, update_task->end_sid,
                                update_task->expected_version);

                        if (!slot_is_queryable(lsm_idx))
                        {
                            elog(DEBUG1, "[maintenance_worker] SEGMENT_UPDATE_ADOPT: skip — slot %d not queryable (INDEX_UNLOADED)",
                                 lsm_idx);
                            /*
                             * Match the existing dpv_attempt_adoption path:
                             * INDEX_UNLOADED maps to maint_status=2 ("skip,
                             * files persist on disk for recover_lsm_index_internal").
                             */
                            vs_search_result_at(update_task->backend_pgprocno)->maint_status = 2;
                            client = &ProcGlobal->allProcs[update_task->backend_pgprocno];
                            break;
                        }

                        adopt_result = dpv_attempt_adoption(update_task->lsm_idx,
```

- [ ] **Step 5: Gate the merge-thread dispatch site for REBUILD/MERGE**

REBUILD and MERGE tasks are dispatched inside `merge_worker_thread`, not the maintenance dispatcher. Task 10 will additionally make `scan_and_claim_merge_task_pool` skip non-queryable slots; the gate here is belt-and-suspenders against any stale claim.

Edit `pgvector/src/vector_index_worker.c`. Find at lines 1505-1526:

```c
        /* Scan for a task and execute it */
        claimed = scan_and_claim_merge_task_pool(&task);

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
```

Replace with:

```c
        /* Scan for a task and execute it */
        claimed = scan_and_claim_merge_task_pool(&task);

        if (!claimed)
            continue;

        fprintf(stderr, "[merge_worker_thread] claimed task op=%d lsm=%d seg0=%u seg1=%u\n",
                task.operation_type, task.lsm_idx, task.segment_idx0, task.segment_idx1);

        /*
         * Belt-and-suspenders: scan_and_claim_merge_task_pool already filters
         * non-queryable slots, and QUERYABLE never regresses, so this gate
         * should be unreachable in practice. Keep it so the invariant is
         * locally checkable when reading this function in isolation.
         */
        if (!slot_is_queryable(task.lsm_idx))
        {
            elog(DEBUG1, "[merge_worker_thread] skip op=%d — slot %d not queryable",
                 task.operation_type, task.lsm_idx);
            signal_merge_pool();
            continue;
        }

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
```

- [ ] **Step 6: Build**

```bash
make
```

Expected: clean build.

- [ ] **Step 7: Inspect**

Open `vector_index_worker.c` and confirm each maintenance case has the gate at the top. Walk through manually:

| Case | Has gate? |
|---|---|
| `SEGMENT_UPDATE_REGULAR` | ✓ |
| `SEGMENT_UPDATE_VACUUM` | ✓ |
| `SEGMENT_UPDATE_ADOPT` | ✓ |
| `SEGMENT_UPDATE_REBUILD_FLAT` / `_DELETION` / `_MERGE` | ✓ |
| `IndexLoadTaskType` | **No gate** (intentional — this task drives the transition itself) |
| `InternalSegmentUpgradeTaskType` | **No gate** (inherits from parent IndexLoadTask) |

- [ ] **Step 8: Commit checkpoint**

Suggested message:

```
vector_index_worker: gate maintenance handlers on slot_is_queryable

Vacuum, regular post-flush update, adopt, rebuild, and merge handlers
all return a no-op status when the slot is recovered-but-not-loaded
(WRITABLE or LOADING_INDEX). Disk-side state is unaffected; the eventual
index_load_blocking picks up the latest bitmap subversion. ADOPT
preserves its existing INDEX_UNLOADED -> maint_status=2 protocol.
```

---

## Task 10: Skip non-queryable slots in `scan_and_claim_merge_task_pool`

**Files:**
- Modify: `pgvector/src/vector_index_worker.c:826-870` (`scan_and_claim_merge_task_pool`)

Five identical `if (!pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[lsm_idx].valid)) continue;` checks should be tightened to skip any slot that is not yet queryable.

- [ ] **Step 1: Tighten the slot skip in all five priority loops**

Edit `pgvector/src/vector_index_worker.c`. Find at lines 829-867 (five priority loops, each starting with):

```c
        if (!pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[lsm_idx].valid))
            continue;
```

Replace **each** of the five occurrences (use `replace_all = true` only if the file contains *no other* `if (!pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[lsm_idx].valid))` patterns outside this function; otherwise do them one by one):

```c
        if (!slot_is_queryable(lsm_idx))
            continue;
```

Verify only `scan_and_claim_merge_task_pool` is affected:

```bash
grep -n '!pg_atomic_read_u32(&SharedLSMIndexBuffer->slots\[lsm_idx\].valid)' src/vector_index_worker.c
```

Expected: zero hits after the change (or only hits outside `scan_and_claim_merge_task_pool`, which should not be touched).

- [ ] **Step 2: Build**

```bash
make
```

Expected: clean build.

- [ ] **Step 3: Commit checkpoint**

Suggested message:

```
vector_index_worker: merge scan skips non-queryable slots

scan_and_claim_merge_task_pool now uses slot_is_queryable() instead of
just checking valid != FREE. WRITABLE-only slots have empty pools so the
inner loop would find nothing anyway, but skipping at the outer loop
avoids the per-slot atomics fetch and aligns with the rest of the
worker-side gating.
```

---

## Task 11: Final integration check

**Files:** none modified — verification only.

- [ ] **Step 1: Full grep audit**

```bash
grep -rn 'valid) == 1\|valid) == 2\|valid == 1\|valid == 2\|valid != 1\|valid != 2' src/
```

Expected: zero hits, or only hits inside comments / unrelated counters (manually inspect each survivor).

```bash
grep -rn 'load_lsm_index_internal' src/
```

Expected: zero hits.

```bash
grep -n 'pg_atomic_write_u32.*&slot->valid.*, 1)\|pg_atomic_write_u32.*&slot->valid.*, 2)' src/
```

Expected: zero hits (every state write goes through `LSM_SLOT_*` constants).

```bash
grep -n 'pg_atomic_write_u32.*&slot->valid.*, 0)' src/
```

Expected: hits only in the IndexLoadWorker error path (`index_load_worker.c`) and the `register_lsm_index` worker-not-running path (`lsmindex.c`). These intentionally set FREE; they could be migrated to `LSM_SLOT_FREE` but it is a pure cosmetic change — if convenient, edit them now; otherwise leave for a follow-up.

- [ ] **Step 2: Clean build**

```bash
make clean
make
```

Expected: build succeeds with no warnings.

- [ ] **Step 3: clangd / compile-commands check**

If the project has a `compile_commands.json` generator (e.g., `bear -- make`), regenerate it and inspect that clangd reports no new diagnostics. Otherwise this step is informational only.

- [ ] **Step 4: Hand off to maintainer for functional verification**

The maintainer will manually verify the following on a fresh test cluster (per spec §8 — no automated test runs as part of this plan):

1. Cold-start, writer-first: open DB, run an INSERT against a persisted LSM index, verify the slot is `LSM_SLOT_WRITABLE` (visible in `DEBUG1` logs from this PR) and there is no `IndexLoadTask` message.
2. Cold-start, reader-first: open DB, run a SELECT, verify WRITABLE → LOADING_INDEX → QUERYABLE transition in the logs and that the query returns correct results.
3. Cold-start, `VACUUM` first: confirm bulk_delete finishes; worker logs show "skip VACUUM, slot not queryable"; a subsequent SELECT loads the latest bitmap subversion and reflects the deletions.

- [ ] **Step 5: Final commit checkpoint**

Suggested message:

```
docs: spec & plan for decouple-recovery-from-load (already committed)

This is the integration / audit task — no code change here. All
preceding commits together implement the design at
docs/superpowers/specs/2026-05-16-decouple-recovery-from-load-design.md.
```
