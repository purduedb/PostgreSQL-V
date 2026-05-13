# Index Loading Background Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all index loading and crash-recovery logic out of backend processes and into a dedicated `IndexLoadWorker` background worker, so that backend cancellation or death can never leave shared state in a partially initialized limbo.

**Architecture:** Backends atomically claim an `LSMIndexBufferSlot` (CAS `valid` 0→2), write their `MyDatabaseId` and `GetUserId()` into the slot, then signal the `IndexLoadWorker` and block on a per-slot `ConditionVariable`. The worker (the sole entity that ever writes `valid=1`) performs the full `load_lsm_index_internal()` including crash-recovery, then broadcasts on the slot's CV to wake all waiting backends. On worker restart the postmaster reschedules it; it re-scans for stuck `valid=2` slots and re-processes them without any backend involvement.

**Tech Stack:** C, PostgreSQL background worker API (`BGWORKER_BACKEND_DATABASE_CONNECTION`), `ConditionVariable`, `pg_atomic_*`, `LWLock`, `WaitLatch`/`SetLatch`, `StartTransactionCommand`/`CommitTransactionCommand`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `pgvector/src/lsmindex.h` | Modify | Add `ConditionVariable load_cv`, `pg_atomic_uint32 load_error`, `Oid request_db_oid`, `Oid request_db_userid` to `LSMIndexBufferSlot`; add `IndexLoadCoordinator` struct + `extern`; add `load_lsm_index_internal` declaration |
| `pgvector/src/lsmindex.c` | Modify | Update `lsm_index_buffer_shmem_initialize()`; rename `load_lsm_index()` → `load_lsm_index_internal()` (changes signature + opens relation itself); rewrite `register_lsm_index()` to signal worker + CV wait; simplify `get_lsm_index_idx()`, `get_lsm_index()`, `get_lsm_index_idx_no_loading()` to remove spin-polls |
| `pgvector/src/index_load_worker.h` | Create | Declare `index_load_worker_init()`, `index_load_worker_main()` |
| `pgvector/src/index_load_worker.c` | Create | Worker main loop, crash-recovery scan, calls `load_lsm_index_internal()`, broadcasts on completion/failure |
| `pgvector/src/vector.c` | Modify | Register new BGW in `_PG_init()` and add `RequestAddinShmemSpace` for `IndexLoadCoordinator` |
| `pgvector/Makefile` | Modify | Add `src/index_load_worker.o` to `OBJS` |

---

## Background: Current State vs. Target State

**Current (broken) flow:**
```
Backend: get_lsm_index() → register_lsm_index() [CAS 0→2] → load_lsm_index() → valid=1
                                                                ↑ dies here → valid stays 2 forever
Other backends: spin on pg_usleep(1000) forever if first backend dies
```

**Target flow:**
```
Backend: get_lsm_index() → register_lsm_index() [CAS 0→2, signal worker] → ConditionVariableSleep
Worker:                                                                      → load_lsm_index_internal() → valid=1 → Broadcast
Other backends:                                                                                          ← wake, see valid=1, return
Crash + restart: worker scans for valid=2 stubs → re-loads → broadcasts → backends wake
```

---

## State Machine for `LSMIndexBufferSlot.valid`

```
0 (UNLOADED / free)
  ↓  CAS by any backend claiming this slot
2 (LOAD_IN_PROGRESS)
  ↓  written by IndexLoadWorker on success
1 (LOADED — terminal, readable by backends)
  (On worker error: write load_error=1, then reset valid→0, then broadcast)
```

---

## Task 1 — Add shared-memory coordination fields to `lsmindex.h` and `lsmindex.c`

**Files:**
- Modify: `pgvector/src/lsmindex.h`
- Modify: `pgvector/src/lsmindex.c` (`lsm_index_buffer_shmem_initialize` only)

- [ ] **Step 1: Add `#include` for `ConditionVariable` in `lsmindex.h`**

  After the existing `#include "storage/lwlock.h"` line (line 6), add:

  ```c
  #include "storage/condition_variable.h"
  ```

- [ ] **Step 2: Add new fields to `LSMIndexBufferSlot` in `lsmindex.h`**

  Replace the current `LSMIndexBufferSlot` struct (lines 165–169):

  ```c
  typedef struct LSMIndexBufferSlot
  {
      pg_atomic_uint32 valid;   /* atomic flag: 0 = free, 1 = loaded, 2 = loading */
      LSMIndexData lsmIndex;
  }   LSMIndexBufferSlot;
  ```

  with:

  ```c
  typedef struct LSMIndexBufferSlot
  {
      pg_atomic_uint32 valid;      /* 0=free, 1=loaded, 2=load-in-progress */
      ConditionVariable load_cv;   /* backends sleep here; worker broadcasts on done/fail */
      pg_atomic_uint32 load_error; /* 0=ok, 1=failed; set before broadcast */
      Oid request_db_oid;          /* database OID written by the claiming backend */
      Oid request_db_userid;       /* role OID written by the claiming backend */
      LSMIndexData lsmIndex;
  }   LSMIndexBufferSlot;
  ```

- [ ] **Step 3: Add `IndexLoadCoordinator` struct and extern to `lsmindex.h`**

  After the `LSMIndexBuffer` extern declaration (after `extern LSMIndexBuffer *SharedLSMIndexBuffer;`), add:

  ```c
  /* Shared coordinator so backends can signal the IndexLoadWorker */
  typedef struct IndexLoadCoordinator
  {
      pg_atomic_int32 worker_pgprocno; /* pgprocno of the load worker; -1 = not running */
  } IndexLoadCoordinator;

  extern IndexLoadCoordinator *SharedIndexLoadCoordinator;
  ```

- [ ] **Step 4: Add `load_lsm_index_internal` declaration to `lsmindex.h`**

  In the function declarations section (near `get_lsm_index`, `build_lsm_index`, etc.), add:

  ```c
  void load_lsm_index_internal(Oid index_relid, uint32_t slot_idx);
  ```

- [ ] **Step 5: Add `IndexLoadCoordinator` global and initialize it in `lsmindex.c`**

  After `LSMIndexBuffer *SharedLSMIndexBuffer = NULL;` (line 24 of `lsmindex.c`), add:

  ```c
  IndexLoadCoordinator *SharedIndexLoadCoordinator = NULL;
  ```

- [ ] **Step 6: Initialize new slot fields in `lsm_index_buffer_shmem_initialize()`**

  Inside the `!found` block for `SharedLSMIndexBuffer` (the `for (int i = 0; i < INDEX_BUF_SIZE; i++)` loop), after the existing `pg_atomic_write_u32(&SharedLSMIndexBuffer->slots[i].valid, 0);` call, add:

  ```c
  ConditionVariableInit(&SharedLSMIndexBuffer->slots[i].load_cv);
  pg_atomic_init_u32(&SharedLSMIndexBuffer->slots[i].load_error, 0);
  SharedLSMIndexBuffer->slots[i].request_db_oid    = InvalidOid;
  SharedLSMIndexBuffer->slots[i].request_db_userid = InvalidOid;
  ```

- [ ] **Step 7: Initialize `SharedIndexLoadCoordinator` in `lsm_index_buffer_shmem_initialize()`**

  At the end of `lsm_index_buffer_shmem_initialize()` (just before the closing brace), add:

  ```c
  SharedIndexLoadCoordinator = (IndexLoadCoordinator *)
      ShmemInitStruct("Index Load Coordinator",
                      MAXALIGN(sizeof(IndexLoadCoordinator)),
                      &found);
  if (!found)
  {
      pg_atomic_write_s32(&SharedIndexLoadCoordinator->worker_pgprocno, -1);
  }
  ```

- [ ] **Step 8: Verify the build compiles**

  ```bash
  cd /home/liu4127/postgresql/decoupled_pgvector/pgvector && make 2>&1 | head -40
  ```

  Expected: compilation errors about missing `index_load_worker.o` or undefined `load_lsm_index_internal` are acceptable at this stage; no errors about the struct changes.


---

## Task 2 — Create `index_load_worker.h`

**Files:**
- Create: `pgvector/src/index_load_worker.h`

- [ ] **Step 1: Write the header**

  ```c
  #ifndef INDEX_LOAD_WORKER_H
  #define INDEX_LOAD_WORKER_H

  #include "postgres.h"

  void index_load_worker_init(void);
  void index_load_worker_main(Datum main_arg);

  #endif
  ```


---

## Task 3 — Add `src/index_load_worker.o` to Makefile

**Files:**
- Modify: `pgvector/Makefile`

- [ ] **Step 1: Find the OBJS line**

  ```bash
  grep -n "OBJS\s*=" /home/liu4127/postgresql/decoupled_pgvector/pgvector/Makefile | head -5
  ```

- [ ] **Step 2: Append the new object**

  Locate the `OBJS = ...` assignment in `pgvector/Makefile`. Add `src/index_load_worker.o` to the list, following the same style as other entries (e.g., after `src/vector_index_worker.o`):

  ```makefile
  OBJS = ... \
         src/vector_index_worker.o \
         src/index_load_worker.o \
         ...
  ```

---

## Task 4 — Register the new background worker in `vector.c`

**Files:**
- Modify: `pgvector/src/vector.c`

- [ ] **Step 1: Add the include for the new header**

  Near the top of `vector.c`, after the existing worker includes, add:

  ```c
  #include "index_load_worker.h"
  ```

- [ ] **Step 2: Add `RequestAddinShmemSpace` for `IndexLoadCoordinator`**

  In `_PG_init()`, after the existing `RequestAddinShmemSpace(4000000000L);` call (line ~101), add:

  ```c
  RequestAddinShmemSpace(MAXALIGN(sizeof(IndexLoadCoordinator)));
  ```

- [ ] **Step 3: Register the BGW**

  In `_PG_init()`, after the existing `RegisterBackgroundWorker` call for `lsm_index_worker` (around line 133), add:

  ```c
  elog(DEBUG1, "[_PG_init] register index load worker");
  BackgroundWorker index_load_worker;
  memset(&index_load_worker, 0, sizeof(BackgroundWorker));
  index_load_worker.bgw_flags = BGWORKER_SHMEM_ACCESS |
                                BGWORKER_BACKEND_DATABASE_CONNECTION;
  index_load_worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
  index_load_worker.bgw_restart_time = 1;  /* restart 1s after crash */
  snprintf(index_load_worker.bgw_name, BGW_MAXLEN, "IndexLoadWorker");
  snprintf(index_load_worker.bgw_library_name, BGW_MAXLEN, "vector.so");
  snprintf(index_load_worker.bgw_function_name, BGW_MAXLEN, "index_load_worker_main");
  index_load_worker.bgw_main_arg = (Datum) 0;
  index_load_worker.bgw_notify_pid = 0;
  RegisterBackgroundWorker(&index_load_worker);
  elog(DEBUG1, "[_PG_init] register index load worker finished");
  ```

  **Why `BGWORKER_BACKEND_DATABASE_CONNECTION`:** The worker calls `BackgroundWorkerInitializeConnectionByOid()` so it can open relations and read heap tuples during recovery step 3 (memtable reconstruction).


---

## Task 5 — Rename and adapt `load_lsm_index()` → `load_lsm_index_internal()` in `lsmindex.c`

**Files:**
- Modify: `pgvector/src/lsmindex.c`

This is the largest refactor step. The current `static void load_lsm_index(Relation index, uint32_t slot_idx)` (line ~435):
- Takes an already-open `Relation index`
- Calls database-accessing functions using that `Relation`
- Sets `slot->valid = 1` at the very end

After this task it becomes `void load_lsm_index_internal(Oid index_relid, uint32_t slot_idx)`:
- Opens the index relation itself
- Does everything the old function did
- Still sets `slot->valid = 1` on success (the worker broadcasts after)
- Is no longer `static` (called from `index_load_worker.c`)

- [ ] **Step 1: Change the function from `static` to non-static and update its signature**

  Find the function declaration line (around line 435):
  ```c
  static void
  load_lsm_index(Relation index, uint32_t slot_idx)
  ```

  Replace with:
  ```c
  void
  load_lsm_index_internal(Oid index_relid, uint32_t slot_idx)
  ```

- [ ] **Step 2: Open and close the index relation inside the function**

  The current function body begins with:
  ```c
  {
      Oid index_relid = RelationGetRelid(index);
      elog(DEBUG1, "[load_lsm_index] loading index: relId = %u to slot %u", index_relid, slot_idx);
  ```

  Replace the opening of the function body with:
  ```c
  {
      Relation index = index_open(index_relid, AccessShareLock);
      elog(DEBUG1, "[load_lsm_index_internal] loading index: relId = %u to slot %u",
           index_relid, slot_idx);
  ```

  Then, find the two `table_close` calls at the end of the function. After the second one (currently `table_close(heap_rel, AccessShareLock);`), add:
  ```c
  index_close(index, AccessShareLock);
  ```

  Also update the `elog` on the final success line to use the new name:
  ```c
  elog(DEBUG1, "[load_lsm_index_internal] successfully loaded LSM index %u", index_relid);
  ```

- [ ] **Step 3: Add missing include for `index_open` / `index_close`**

  At the top of `lsmindex.c`, verify `catalog/index.h` is already included (it already is, at line 17). If not, add it:

  ```c
  #include "catalog/index.h"
  ```

- [ ] **Step 4: Verify the function compiles in isolation**

  ```bash
  cd /home/liu4127/postgresql/decoupled_pgvector/pgvector && make src/lsmindex.o 2>&1 | head -30
  ```

  Expected: warnings are acceptable; no errors about `load_lsm_index_internal`.

---

## Task 6 — Rewrite `register_lsm_index()` and simplify `get_lsm_index*()` in `lsmindex.c`

**Files:**
- Modify: `pgvector/src/lsmindex.c`

This removes all spin-polls (`pg_usleep` loops) and the direct call to `load_lsm_index_internal()` from the backend side. Instead, backends signal the worker and sleep on the per-slot `ConditionVariable`.

- [ ] **Step 1: Add include for `storage/proc.h` and `index_load_worker.h` at the top of `lsmindex.c`**

  `lsmindex.c` likely already includes `storage/proc.h` transitively, but make it explicit. After the existing includes, add if not already present:

  ```c
  #include "storage/proc.h"
  #include "index_load_worker.h"
  ```

- [ ] **Step 2: Replace the entire `register_lsm_index()` function body**

  Find the current `register_lsm_index(Oid index_relid)` function (lines ~145–219) and replace its entire body:

  ```c
  static int
  register_lsm_index(Oid index_relid)
  {
      LWLockAcquire(SharedLSMIndexBuffer->lock, LW_EXCLUSIVE);

      /* Double-check: another backend may have finished loading while we waited */
      for (int i = 0; i < INDEX_BUF_SIZE; i++)
      {
          uint32 valid = pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid);
          if (valid == 0 || SharedLSMIndexBuffer->slots[i].lsmIndex.indexRelId != index_relid)
              continue;

          LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[i];
          LWLockRelease(SharedLSMIndexBuffer->lock);

          if (valid == 1)
              return i;  /* already fully loaded */

          /* valid == 2: worker is loading — wait on CV until done or failed */
          ConditionVariablePrepareToSleep(&slot->load_cv);
          while (pg_atomic_read_u32(&slot->valid) == 2)
              ConditionVariableSleep(&slot->load_cv, WAIT_EVENT_EXTENSION);
          ConditionVariableCancelSleep();

          if (pg_atomic_read_u32(&slot->valid) == 1 &&
              pg_atomic_read_u32(&slot->load_error) == 0)
              return i;

          elog(ERROR, "[register_lsm_index] index %u loading failed (load_error=%u, valid=%u)",
               index_relid,
               pg_atomic_read_u32(&slot->load_error),
               pg_atomic_read_u32(&slot->valid));
      }

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

      if (slot_num == -1)
      {
          LWLockRelease(SharedLSMIndexBuffer->lock);
          elog(ERROR, "[register_lsm_index] no free slot");
      }

      LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[slot_num];
      slot->lsmIndex.indexRelId    = index_relid;
      pg_atomic_write_u32(&slot->load_error, 0);
      slot->request_db_oid    = MyDatabaseId;
      slot->request_db_userid = GetUserId();

      LWLockRelease(SharedLSMIndexBuffer->lock);

      /* Signal the IndexLoadWorker — it will see valid==2 and process this slot */
      int32 wpgprocno = pg_atomic_read_s32(&SharedIndexLoadCoordinator->worker_pgprocno);
      if (wpgprocno < 0)
      {
          /* Worker not running — reset slot and error */
          pg_atomic_write_u32(&slot->valid, 0);
          ConditionVariableBroadcast(&slot->load_cv);
          elog(ERROR, "[register_lsm_index] IndexLoadWorker is not running");
      }
      SetLatch(&ProcGlobal->allProcs[wpgprocno].procLatch);

      /* Block until the worker finishes or fails */
      ConditionVariablePrepareToSleep(&slot->load_cv);
      while (pg_atomic_read_u32(&slot->valid) == 2)
          ConditionVariableSleep(&slot->load_cv, WAIT_EVENT_EXTENSION);
      ConditionVariableCancelSleep();

      if (pg_atomic_read_u32(&slot->valid) == 1 &&
          pg_atomic_read_u32(&slot->load_error) == 0)
          return slot_num;

      elog(ERROR, "[register_lsm_index] index %u loading failed (load_error=%u, valid=%u)",
           index_relid,
           pg_atomic_read_u32(&slot->load_error),
           pg_atomic_read_u32(&slot->valid));
  }
  ```

- [ ] **Step 3: Rewrite `get_lsm_index_idx()` — remove spin-poll and direct load call**

  Replace the current `get_lsm_index_idx()` (lines ~902–956) with:

  ```c
  int
  get_lsm_index_idx(Relation index)
  {
      /* Fast path: slot already fully loaded (lock-free) */
      for (int i = 0; i < INDEX_BUF_SIZE; i++)
      {
          if (pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid) == 1 &&
              SharedLSMIndexBuffer->slots[i].lsmIndex.indexRelId == index->rd_id)
              return i;
      }

      /* Slow path: claim slot, signal worker, wait */
      return register_lsm_index(index->rd_id);
  }
  ```

- [ ] **Step 4: Rewrite `get_lsm_index()` — same pattern**

  Replace the current `get_lsm_index()` (lines ~958–1011) with:

  ```c
  LSMIndex
  get_lsm_index(Relation index)
  {
      Oid index_relid = RelationGetRelid(index);

      /* Fast path: slot already fully loaded (lock-free) */
      for (int i = 0; i < INDEX_BUF_SIZE; i++)
      {
          if (pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid) == 1 &&
              SharedLSMIndexBuffer->slots[i].lsmIndex.indexRelId == index_relid)
              return &SharedLSMIndexBuffer->slots[i].lsmIndex;
      }

      /* Slow path: claim slot, signal worker, wait */
      int slot_num = register_lsm_index(index_relid);
      return &SharedLSMIndexBuffer->slots[slot_num].lsmIndex;
  }
  ```

- [ ] **Step 5: Rewrite `get_lsm_index_idx_no_loading()` — replace spin-poll with CV wait**

  This function is called from the `vector_index_worker_main` process (a BGW with PGPROC), so it can use `ConditionVariable`.

  Replace the current `get_lsm_index_idx_no_loading()` (lines ~866–900) with:

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
              ConditionVariableSleep(&slot->load_cv, WAIT_EVENT_EXTENSION);
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

- [ ] **Step 6: Remove the now-unreachable old call to `load_lsm_index` in `get_lsm_index_idx` and `get_lsm_index`**

  These calls have been removed by the rewrites above. Verify no references to the old `load_lsm_index` (the un-renamed version) remain:

  ```bash
  grep -n "load_lsm_index[^_]" /home/liu4127/postgresql/decoupled_pgvector/pgvector/src/lsmindex.c
  ```

  Expected: no output (only `load_lsm_index_internal` references should remain).

- [ ] **Step 7: Compile**

  ```bash
  cd /home/liu4127/postgresql/decoupled_pgvector/pgvector && make src/lsmindex.o 2>&1 | head -30
  ```

  Expected: clean compile or warnings only.

---

## Task 7 — Implement `index_load_worker.c`

**Files:**
- Create: `pgvector/src/index_load_worker.c`

This is the new background worker. Key design points:
- **Database connection:** Called once via `BackgroundWorkerInitializeConnectionByOid` using the `request_db_oid` of the first slot it finds. All subsequent loads are assumed to be from the same database (research-project assumption; add a `TODO` for multi-DB support).
- **Crash-recovery:** On startup the worker scans for slots already in state 2 (stuck from a previous crash) and processes them before waiting for new signals.
- **Error handling:** On `load_lsm_index_internal()` failure, sets `load_error=1`, resets `valid=0`, and broadcasts — backends see a non-1 `valid` after waking and call `elog(ERROR)`. The worker remains alive to process the next request.
- **Transaction wrapping:** `load_lsm_index_internal()` calls PostgreSQL catalog/heap functions; these must run inside `StartTransactionCommand()`/`CommitTransactionCommand()`.

- [ ] **Step 1: Write `index_load_worker.c`**

  ```c
  #include "postgres.h"
  #include "fmgr.h"
  #include "miscadmin.h"
  #include "postmaster/bgworker.h"
  #include "postmaster/postmaster.h"
  #include "storage/condition_variable.h"
  #include "storage/ipc.h"
  #include "storage/latch.h"
  #include "storage/proc.h"
  #include "storage/shmem.h"
  #include "tcop/tcopprot.h"
  #include "utils/elog.h"
  #include "utils/memutils.h"
  #include "utils/wait_event.h"
  #include "access/xact.h"

  #include "index_load_worker.h"
  #include "lsmindex.h"

  static volatile sig_atomic_t got_sigterm = false;

  static void
  handle_sigterm(SIGNAL_ARGS)
  {
      got_sigterm = true;
      SetLatch(MyLatch);
  }

  /* Process one slot: run load_lsm_index_internal inside a transaction,
   * then set valid=1 and broadcast (or set load_error=1, reset valid=0, broadcast on failure). */
  static void
  process_pending_slot(int slot_idx)
  {
      LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[slot_idx];
      Oid index_relid = slot->lsmIndex.indexRelId;

      elog(DEBUG1, "[index_load_worker] loading index %u into slot %d", index_relid, slot_idx);

      StartTransactionCommand();
      PG_TRY();
      {
          load_lsm_index_internal(index_relid, slot_idx);
          /* load_lsm_index_internal sets slot->valid = 1 on success */
          CommitTransactionCommand();
      }
      PG_CATCH();
      {
          AbortCurrentTransaction();

          ErrorData *edata = CopyErrorData();
          FlushErrorState();
          elog(WARNING, "[index_load_worker] failed to load index %u: %s",
               index_relid, edata->message);
          FreeErrorData(edata);

          pg_atomic_write_u32(&slot->load_error, 1);
          pg_write_barrier();
          pg_atomic_write_u32(&slot->valid, 0);
      }
      PG_END_TRY();

      /* Wake all backends waiting on this slot (whether success or failure) */
      pg_write_barrier();
      ConditionVariableBroadcast(&slot->load_cv);
  }

  void
  index_load_worker_main(Datum main_arg)
  {
      bool db_connected = false;

      pqsignal(SIGTERM, handle_sigterm);
      BackgroundWorkerUnblockSignals();

      /* Publish our PGPROC slot so backends can signal us */
      pg_atomic_write_s32(&SharedIndexLoadCoordinator->worker_pgprocno,
                          (int32) MyProc->pgprocno);

      elog(LOG, "[index_load_worker] started (pgprocno=%d)", MyProc->pgprocno);

      for (;;)
      {
          if (got_sigterm)
          {
              pg_atomic_write_s32(&SharedIndexLoadCoordinator->worker_pgprocno, -1);
              elog(LOG, "[index_load_worker] received SIGTERM, exiting");
              proc_exit(0);
          }

          bool found_work = false;

          for (int i = 0; i < INDEX_BUF_SIZE; i++)
          {
              LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[i];

              if (pg_atomic_read_u32(&slot->valid) != 2)
                  continue;
              if (slot->lsmIndex.indexRelId == InvalidOid)
                  continue;
              if (slot->request_db_oid == InvalidOid)
                  continue;

              /* Connect to the database on first work item (once per process lifetime) */
              if (!db_connected)
              {
                  BackgroundWorkerInitializeConnectionByOid(
                      slot->request_db_oid,
                      slot->request_db_userid,
                      0 /* flags */);
                  db_connected = true;
                  elog(LOG, "[index_load_worker] connected to database %u", slot->request_db_oid);
              }

              found_work = true;
              process_pending_slot(i);
          }

          if (!found_work)
          {
              (void) WaitLatch(MyLatch,
                               WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
                               1000L,   /* 1-second polling fallback */
                               WAIT_EVENT_EXTENSION);
              ResetLatch(MyLatch);
          }
      }
  }
  ```

- [ ] **Step 2: Verify the full build compiles**

  ```bash
  cd /home/liu4127/postgresql/decoupled_pgvector/pgvector && make 2>&1 | tail -20
  ```

  Expected: `vector.so` built successfully. No linker errors.

---

## Known Limitations and TODOs

1. **Single-database assumption:** `BackgroundWorkerInitializeConnectionByOid` is called once per worker process lifetime. If indices from multiple databases need loading, each database requires a separate `IndexLoadWorker` process. This would require dynamic BGW spawning (`RegisterDynamicBackgroundWorker`) and is left for future work. Add a `TODO` comment in `index_load_worker.c`.

2. **`get_lsm_index_idx_no_loading` in pthreads:** If this function is ever called from the maintenance pthreads inside `vector_index_worker`, `ConditionVariableSleep` will crash (no PGPROC). Verify the call sites are only in BGW main-process context. Add an `Assert(IsBackgroundWorker || IsUnderPostmaster)` check.

3. **Worker not-running window:** Between postmaster restart and worker registration (`pg_atomic_write_s32` of `worker_pgprocno`), backends calling `register_lsm_index` will see `wpgprocno == -1` and get an error. The 1-second `bgw_restart_time` limits this window. A more robust solution would retry with exponential backoff.

4. **`valid` reset on load failure:** When the worker sets `valid=0` after failure, the slot is freed and any new backend can reclaim it. If the failure is transient (e.g., a recoverable I/O error), the next backend will trigger a re-load automatically. If the failure is permanent (corrupt files), backends will keep retrying and failing — there is no permanent blacklist. This is acceptable for now.
