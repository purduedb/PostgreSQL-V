# Segment-Search Completion Handshake Fix

**Date:** 2026-05-30  
**Branch:** main  
**Files modified:** `tasksend.c`, `vectorindeximpl.cpp`, `vectorindeximpl.hpp`, `vector_index_worker.c`

---

This document records a concurrency fix for the decoupled pgvector LSM index
search path. Under concurrent load the **segment portion of a search could be
silently dropped**, collapsing recall from a stable ~0.92 (8 clients) to
~0.74 (32 clients).

It covers:

- the symptom and how it was diagnosed,
- the two root causes (a broken completion handshake and a query-vector
  aliasing race),
- the fix applied to each file,
- how to verify.

---

## Symptom

- Recall was stable at **~0.91–0.93** with 8 concurrent clients.
- At 32 clients, recall intermittently dropped to **0.74–0.79**.
- The worst-recall steps were also the **fastest** (3–4 s), while the slowest
  steps kept recall ≥ 0.88.

A search that returns *faster* while producing *worse* recall is the signature
of a premature return (the backend stops waiting and returns early), not of a
slow index builder — a slow builder only raises latency, it never lowers it.

---

## Architecture (relevant path)

The segment search is asynchronous and crosses a process + thread-pool
boundary:

1. The backend dispatches the segment search to the vector-index worker via
   the ring buffer (`vector_search_send`), does its local memtable searches,
   then blocks in `vector_search_get_result`.
2. The worker fans the per-segment searches onto a folly thread pool and
   returns immediately. The **last folly thread to finish** writes
   `result_count` + payload into the backend's result slot and calls
   `SetLatch` on the backend.
3. The backend wakes from the latch and reads the slot.

Each backend has a dedicated `VectorSearchResult` slot keyed by `pgprocno`.

---

## Root cause 1 — broken completion handshake

`VectorSearchResultData` has a `status` field (`0 = empty`, `1 = done`) in
`ringbuffer.h` that was designed for exactly this handoff — but on the search
path it was **never set to 1 and never checked**.

The wait was effectively *"sleep until some latch set, then read the slot
unconditionally."* The problem:

- `MyLatch` is the backend's **shared process latch**. PostgreSQL sets it for
  many unrelated reasons (sinval catchup, `SIGUSR1`/procsignal, etc.), and
  that traffic **scales with backend count**.
- `vector_search_send` never reset the latch before publishing the task.

So at 32 clients a spurious latch set arriving between *send* and *wait* made
`WaitLatch` return immediately; the backend read its slot **before** the folly
threads had written this query's results, getting the previous query's stale
results (or a half-written payload). That stale segment result was merged with
the correct memtable result, degrading recall. At 8 clients latch traffic is
light, so the wait almost always coincided with the real completion.

---

## Root cause 2 — query-vector aliasing

`ConcurrentVectorSearchOnSegments` stored the caller's `query_vector` pointer
directly (`ctx->query_vector = query_vector`). That pointer aims into a single
reused worker scratch buffer. The folly threads read the vector
**asynchronously after the call returns**, while the worker loops and
overwrites that same buffer for the next search. Under 32-client churn a new
search could overwrite the buffer mid-flight, so some segments were searched
with the **wrong query vector**. This is also load-dependent and compounds the
recall loss.

---

## The fix

The design is unchanged (search-everything-and-merge); only the completion
handshake and the query-vector lifetime are corrected. Latency may rise under
load, but segment results can no longer be dropped.

### `tasksend.c`
- `vector_search_send`: set the backend's `result->status = 0` under the ring
  lock before notifying the worker (clears any stale done flag; `LWLockRelease`
  provides release ordering before the worker dequeues under the same lock).
- `vector_search_get_result`: loop **reset-before-check** on `status == 1`,
  using the latch only as a wakeup hint, with a `pg_read_barrier()` before
  reading the payload. Reset-before-check avoids lost wakeups.

### `vectorindeximpl.cpp`
- Last-finisher thread: after writing `result_count` + payload, issue a
  release fence, then set `result->status = 1` (before `SetLatch`).
- Copy the query vector into ctx-owned memory (`malloc` + `memcpy`, new `dim`
  parameter) instead of holding the ring-slot pointer; free it with the rest
  of `ctx` in the last-finisher cleanup.

### `vectorindeximpl.hpp`
- Declaration updated to add the `int dim` parameter.

### `vector_index_worker.c`
- Pass `(int) task->vector_dim` to `ConcurrentVectorSearchOnSegments`.
- Empty-result path (no segments to search): write `result_count = 0`,
  `pg_write_barrier()`, set `status = 1`, then `SetLatch`.

---

## Memory ordering summary

| Side    | Order                                                              |
|---------|-------------------------------------------------------------------|
| Writer  | write `result_count` + payload → release barrier → `status = 1` → `SetLatch` |
| Reader  | `ResetLatch` → check `status == 1` → `pg_read_barrier()` → read payload |

The reader treats the latch purely as a hint; correctness comes from the
`status` flag plus the barrier pairing.

---

## Verification

- Build is green (`make`, no errors; `vector.so` relinks).
- The real confirmation is the recall workload: rebuild-install and re-run the
  **32-client** concurrency test. Recall should return to the stable ~0.92
  instead of the 0.74–0.79 dips.

---

## Out of scope (related, not changed)

The maintenance path (`submit_and_wait_maintenance` in `tasksend.c`) reads
`maint_status` with the same latch-trust pattern but no done-flag gate. It is
partially protected — it `ResetLatch`es before submitting and callers
pre-clear `maint_status` — but a spurious wakeup could in principle make a
vacuum/adopt `RETRY` read as `OK`. Hardening it the same way (a dedicated done
flag) is a candidate follow-up.

---

## Files changed

- `pgvector/src/tasksend.c`
- `pgvector/src/vectorindeximpl.cpp`
- `pgvector/src/vectorindeximpl.hpp`
- `pgvector/src/vector_index_worker.c`
