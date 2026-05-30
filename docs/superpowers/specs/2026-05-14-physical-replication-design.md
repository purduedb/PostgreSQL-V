# Physical Replication Support for `decoupled_pgvector`

**Status:** Design — pending implementation plan.
**Date:** 2026-05-14
**Scope:** Read-only standby (v1). No failover, no multi-standby coordination, no cascading replication.

---

## 1. Goals

1. **Index segments propagate to the secondary.** The secondary's `VectorIndexWorker` loads propagated segments into local memory; the secondary always knows which segments are currently visible.
2. **Status pages replay into memtables on the secondary.** The secondary materializes the memtable state described by replicated status-page updates.
3. **Vacuum is correct under replication.** Both memtable vacuum and segment-bitmap vacuum propagate. Index vacuum is visible at or before the corresponding heap vacuum, from a backend's point of view. Index vacuum becomes visible only after it has been fully propagated.

## 2. Non-goals (v1)

- **Failover / promotion.** The standby is read-only; if the primary fails, recovery to a working primary is operational (manual). Promotion is out of scope.
- **Multi-standby coordination.** Single standby; no cascading.
- **Initial sync.** The standby's first-time state — segment files and PG cluster — is brought up via `pg_basebackup` plus an external `rsync` of `VECTOR_STORAGE_BASE_DIR`. Steady-state replication starts from there.
- **Cross-version index changes.** The design assumes the primary and secondary run identical binaries.

## 3. Background

The fork replicates two kinds of state today, and only one of them rides PG's WAL:


| State                                                                  | Storage                                                                                         | Replication today                                                                                                                                    | Notes                                                                                                                                                                                                                                                      |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status pages (memtable metadata + per-memtable tid lists)              | PG buffer cache, fork's main fork pages                                                         | PG WAL via `GenericXLog` ([statuspage.c:33](../../../pgvector/src/statuspage.c#L33), [statuspage.c:262](../../../pgvector/src/statuspage.c#L262), …) | Replicates automatically. No extension hook for redo.                                                                                                                                                                                                      |
| Index segments (HNSW/DiskANN structure, mapping, bitmap, offset files) | Files under `VECTOR_STORAGE_BASE_DIR` ([lsmindex.h:203](../../../pgvector/src/lsmindex.h#L203)) | None                                                                                                                                                 | Atomic via temp-then-rename ([lsmindex_io.c:718](../../../pgvector/src/lsmindex_io.c#L718))                                                                                                                                                                |
| Bitmap subversions (vacuum result on a flushed segment)                | Files alongside segments (`bitmap_<s>_<e>_v<V>_s<S>`)                                           | None                                                                                                                                                 | Written by `write_bitmap_file_with_subversion` ([lsmindex_io.c:826](../../../pgvector/src/lsmindex_io.c#L826)); paired with in-memory updates via `SEGMENT_UPDATE_VACUUM` ([vector_index_worker.c:193](../../../pgvector/src/vector_index_worker.c#L193)). |
| Memtables (vector data, bitmap, ready flags)                           | Shared memory (`SharedMemtableBuffer`)                                                          | None directly; reconstructible from status pages + heap                                                                                              | Recovery rebuilds in `load_lsm_index_internal` ([lsmindex.c:495](../../../pgvector/src/lsmindex.c#L495)).                                                                                                                                                  |


The challenges:

- **Two channels for state-modifying events.** Status-page changes ride WAL; segments and bitmap subversions don't. The secondary's view of segments lags WAL by however long a side-channel pull takes.
- **Vacuum is split across three places.** Memtable vacuum is recorded in status pages (rides WAL today); segment-bitmap vacuum is direct disk writes (does not ride WAL); heap vacuum is in PG's standard WAL. Coordinating these three into a single visibility model on the standby is non-trivial.

(Note: `ReleaseStatusMemtable` ([lsmindex.c:1272](../../../pgvector/src/lsmindex.c#L1272)) being asynchronous w.r.t. flush is *not* a challenge for this design — see §10. The standby's memtable in `SharedMemtableBuffer` is decoupled from `Release` WAL replay; the memtable persists until adoption explicitly drops it, regardless of when `Release` replays.)

The design below addresses the challenges by (a) replacing `GenericXLog` with a custom extension rmgr that emits semantic records, (b) introducing a side channel for *segment-content* files only, and (c) routing all vacuum effects through the WAL via that rmgr.

## 4. High-level architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PRIMARY                                                                     │
│                                                                             │
│  Event producers:                                                           │
│    • Backends (insert, vacuum)                                              │
│    • lsm_index_bgworker (flush)                                             │
│    • merge worker threads (merge, rebuild)                                  │
│                                                                             │
│  All producers ──> custom rmgr ──XLogInsert──> PG WAL ────────────┐         │
│      records: Register / Add / MemtableTombstone / UpdateMaxSid / │         │
│               Release / SegmentCreated / SegmentReplaced /        │         │
│               SegmentVacuumTombstones                             │         │
│                                                                   │         │
│  Some producers also write files locally:                         │         │
│    • lsm_index_bgworker      → segment files (flush output)       │         │
│    • merge worker threads    → segment files (merge / rebuild)    │         │
│    • Backends (vacuum)       → bitmap subversion files            │         │
│                          │                                        │         │
│                          v                                        │         │
│                  VECTOR_STORAGE_BASE_DIR                          │         │
│                          │                                        │         │
│                          v                                        │         │
│                  ┌────────────────┐                               │         │
│                  │ side-channel   │  (serves base segment files   │         │
│                  │ file server    │   only; subversions stay      │         │
│                  └────────┬───────┘   local — see §13.1)          │         │
└───────────────────────────┼───────────────────────────────────────┼─────────┘
                            │                                       │
                            │ pull                                  │ replicate WAL
                            │                                       │
┌───────────────────────────┼───────────────────────────────────────┼─────────┐
│ STANDBY                   │                                       │         │
│                           v                                       v         │
│                  ┌────────────────┐                       ┌──────────────┐  │
│                  │ SegmentFetcher │                       │ startup proc │  │
│                  │ bgworker(s)    │                       │ (rmgr redo)  │  │
│                  └────────┬───────┘                       └──────┬───────┘  │
│                           │                                      │          │
│                           │ writes pulled files;                 │ drives   │
│                           │ adopts when coverage rule met        │ memtable │
│                           │                                      │ + bitmap │
│                           v                                      v          │
│                  ┌─────────────────────────────────────────────────┐        │
│                  │ FlushedSegmentPool                              │        │
│                  │ SharedMemtableBuffer                            │        │
│                  └─────────────────────┬───────────────────────────┘        │
│                                        │ search                             │
│                                        v                                    │
│                  backends + vector_index_worker (search-only on standby)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

The WAL carries semantic events for memtable lifecycle, segment lifecycle, and vacuum. The side channel carries only **segment-content files** produced by flush and merge/rebuild. Subversion files are *not* on the side channel — they are reconstructed locally on the standby from WAL redo.

## 5. Subproblems

The design breaks down into six subproblems. They have well-defined interfaces and can be implemented and tested in roughly the order listed.


| ID  | Subproblem                             | Key mechanism                                                                                 |
| --- | -------------------------------------- | --------------------------------------------------------------------------------------------- |
| A   | Side-channel segment propagation       | Pull-based fetcher worker on secondary; primary file server                                   |
| B   | Custom rmgr WAL records                | Replace `GenericXLog` for status-page ops; add segment-lifecycle and vacuum records           |
| C   | Coverage-based segment adoption        | Adopt iff sids are covered by current pool/memtable buffer                                    |
| D   | Standby memtable replayer              | rmgr redo callbacks materialize memtables from WAL                                            |
| E   | Cross-channel coordination             | Pending-fetch queue; async adoption; vacuum applied to whatever rep is current                |
| F   | Standby worker topology and query path | Disable write workers under `RecoveryInProgress`; add `SegmentFetcher`; search path unchanged |


Each is described below.

---

## 6. Subproblem A — Side-channel segment propagation

### What's transported

Only **segment-content files** flow over the side channel:

- `index_<s>_<e>_v<V>` (Faiss/HNSW serialized blob; chunked at 1 GiB per `write_segment_file` at [lsmindex_io.c:111](../../../pgvector/src/lsmindex_io.c#L111))
- `mapping_<s>_<e>_v<V>`
- `offset_<s>_<e>_v<V>`
- `bitmap_<s>_<e>_v<V>` (the **base** bitmap — no subversion suffix)
- `metadata_<s>_<e>_v<V>` (always pulled last; its atomic-rename presence is the marker that the segment is complete on the secondary, matching the primary's discipline at [lsmindex_io.c:395](../../../pgvector/src/lsmindex_io.c#L395))

**Bitmap subversion files are not pulled.** Vacuum on a segment runs entirely through the rmgr WAL channel; the secondary writes its own local subversion files via redo (see Subproblem B and the WAL-emit-before-file protocol in §11).

### Transport

A custom protocol over TCP. Primary runs a small file-server worker (registered by `_PG_init` in [vector.c:68](../../../pgvector/src/vector.c#L68)) that accepts `(indexRelId, start_sid, end_sid, version, file_kind)` requests and streams the named file from `VECTOR_STORAGE_BASE_DIR`. Secondary runs one or more `SegmentFetcher` bgworkers that pop pending-fetch entries from a persistent queue and pull the five files.

### Idempotence and partial-pull safety

- Files written via temp-then-rename, mirroring [write_segment_file](../../../pgvector/src/lsmindex_io.c#L111) and [write_lsm_segment_metadata](../../../pgvector/src/lsmindex_io.c#L366).
- `metadata_`* rename happens last: presence = complete.
- Fetcher checks `metadata_*` existence before initiating a pull; if present, skip (already on disk, e.g. from rsync base sync).
- Same-target fetch attempts deduplicate via the queue.

### Failure modes

- **Primary unreachable.** Fetcher retries with exponential backoff; queue persists.
- **Partial pull, secondary crash.** On restart, `.tmp` files are cleaned; queue re-enqueues the entry; pull restarts.
- **File missing on primary.** Logs error; queue entry marked failed; standby is degraded until manual recovery.

### Open questions for implementation

- Authentication / authorization for the file server (likely shared secret or TLS cert).
- Pull parallelism (proposed: 2–4 workers).
- Configurable target throughput / rate limit to avoid swamping the primary.

---

## 7. Subproblem B — Custom rmgr WAL records

### Why a custom rmgr

The standby needs *semantic* redo callbacks to materialize the in-memory memtable buffer and the bitmap state of the segment pool. `GenericXLog` only ships full-page images; the standby's buffer cache gets the page changes correctly, but there is no per-record callback to drive `SharedMemtableBuffer` updates. A custom rmgr (PG 15+; `RegisterCustomRmgr`) gives us that hook for free, and lets us add new record types for segment lifecycle and vacuum.

### Record types


| Record                                                                                          | Carries                                           | Emitted at                                                                                                              | Standby redo                                                                     |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `Register(indexRelId, sid)`                                                                     | sid                                               | `RegisterStatusMemtable` ([statuspage.c:273](../../../pgvector/src/statuspage.c#L273))                                  | Allocate a memtable slot for sid; updates status page (full-page-image fallback) |
| `Add(indexRelId, sid, slot_index, tid)`                                                         | sid, slot, tid                                    | `AddToStatusMemtable` ([statuspage.c:465](../../../pgvector/src/statuspage.c#L465))                                     | Fetch vector from heap by tid, copy into memtable slot; updates status page      |
| `MemtableTombstone(indexRelId, sid, slot_index)`                                                | sid, slot                                         | `RemoveFromStatusMemtable` ([statuspage.c:750](../../../pgvector/src/statuspage.c#L750))                                | Set bit in memtable bitmap; updates status page                                  |
| `UpdateMaxSid(indexRelId, sid)`                                                                 | sid                                               | `UpdateMaxMemtableSid` ([statuspage.c:250](../../../pgvector/src/statuspage.c#L250))                                    | Update standby's status meta `max_memtable_sid`                                  |
| `Release(indexRelId, sid)`                                                                      | sid                                               | `ReleaseStatusMemtable` ([statuspage.c:86](../../../pgvector/src/statuspage.c#L86))                                     | Status-page cleanup only; does **not** touch standby's memtable shmem            |
| `SegmentCreated(indexRelId, start_sid, end_sid, version)`                                       | sid range, version                                | After `flush_segment_to_disk` ([lsmindex_io.c:718](../../../pgvector/src/lsmindex_io.c#L718)) returns                   | Enqueue pull task                                                                |
| `SegmentReplaced(indexRelId, old_segs[], new_seg, new_version)`                                 | replaced sid ranges, new range/version            | After merge/rebuild writes new file                                                                                     | Enqueue pull task; mark old segments superseded                                  |
| `SegmentVacuumTombstones(indexRelId, owner_start_sid, owner_end_sid, owner_version, entries[])` | owner identity + `(local_idx, tid)` per tombstone | After computing deletions in `bulk_delete_lsm_index` step 3 ([lsmindex.c:1898](../../../pgvector/src/lsmindex.c#L1898)) | Apply bits to matching segment or fallback (see §11)                             |


All records carry `indexRelId` so the redo can find the right `LSMIndexBufferSlot`. All records are durable via `XLogInsert` (and `XLogFlush` where ordering matters; see §11).

### Status-page changes that get replaced

`GenericXLog` use sites in [statuspage.c](../../../pgvector/src/statuspage.c) become custom-rmgr `XLogInsert` calls. On the standby (steady-state replay), the redo callback applies the same page modification (page-level idempotent ops like `PageAddItem` / `PageIndexTupleDelete`) **plus** the semantic side effects on the secondary's in-memory state. On the primary during crash recovery, see the next subsection.

### Redo behavior on primary crash recovery (v1)

**Policy:** On primary crash recovery, every redo callback for our custom rmgr applies the page-level change (via PG's standard `XLogReadBufferForRedoExtended` machinery — same as `GenericXLog` would today) and **skips all extension-specific side effects**:

- No `SharedMemtableBuffer` updates.
- No `FlushedSegmentPool` updates.
- No subversion-file writes from `SegmentVacuumTombstones` redo.
- No fetch-queue enqueues from `SegmentCreated` / `SegmentReplaced` redo.

Detection in code uses PG's standard mechanism — `RecoveryInProgress() && !StandbyMode` is the primary-crash-recovery condition — checked at the top of each redo callback.

**Why this is safe and sufficient for v1.** Post-recovery, the first backend that touches an LSM index invokes `load_lsm_index_internal` ([lsmindex.c:495](../../../pgvector/src/lsmindex.c#L495)), whose disk-scan and reconciliation logic (recovery steps 1, 2, 3) already rebuilds `SharedMemtableBuffer` and `FlushedSegmentPool` from disk state. The page-level changes that *do* happen during recovery (status-page modifications via our callback's page-level work) bring the status pages to their post-crash state — which is exactly what `load_lsm_index_internal` reads to drive the rebuild. So the extension-side state ends up correct without needing to run any extension-specific side effect during recovery.

**v1 trade-off.** If the primary crashes between `XLogInsert(SegmentVacuumTombstones) + XLogFlush` and `write_bitmap_file_with_subversion` (the §11 WAL-emit-before-file protocol), the subversion file is absent on disk after recovery — and because the redo callback's file-write is skipped under this policy, the file stays absent. The vacuum's bitmap delta is therefore not reflected post-recovery. The next VACUUM re-detects the same dead tuples (heap visibility is unchanged) and reissues the delta, so the deletion is eventually applied — just delayed.

**Why "do nothing extension-side" rather than "do the same as standby".** The full extension-side redo path on the standby relies on infrastructure that isn't yet up during primary's crash recovery — `FlushedSegmentPool` isn't initialized for any index, the `SegmentFetcher` isn't running, the index isn't in `SharedLSMIndexBuffer`. Trying to drive those code paths in a partial-startup state would introduce uncertainty about which initialization invariants hold; the conservative choice for v1 is to skip them entirely and let `load_lsm_index_internal` do its existing reconciliation. We may revisit this in a future version to make extension-side state durable independently of `load_lsm_index_internal`'s lazy re-derivation.

### Why not keep `GenericXLog`

A "hook on top of `GenericXLog` redo" would minimize the diff, but PG doesn't offer a per-record extension hook on built-in `GenericXLog`. Replacing those few call sites with a custom rmgr is the smallest change that gets us the semantic callback we need.

---

## 8. Subproblem C — Coverage-based segment adoption

### The adoption rule

> A newly-fetched segment file with sid range `[a, b]` is **adopted** iff the standby's local pool plus memtable buffer already contains a contiguous group `G` of memtables/segments whose combined sid coverage is **exactly** `[a, b]`.
>
> Adoption atomically removes `G` and inserts the new segment in its place (under `pool->seg_lock` write + `lsm->mt_lock`).
>
> Outcomes for a **loaded** index, by the pool state for sids in `[a, b]`:
>
> | Pool state | Outcome |
> |---|---|
> | Exactly covered by `G`; `G` is not a single same-range segment (e.g., memtables, multiple segments, wider-then-trimmed group) | **Adopt** — flush / merge case. Replace `G` with the new segment. |
> | Exactly covered by `G`; `G` is a single segment with range `= [a, b]`; `arrival.version > G.version` | **Adopt** — rebuild case. Replace `G` with the new segment. |
> | Exactly covered by `G`; `G` is a single segment with range `= [a, b]`; `arrival.version ≤ G.version` | **Stale** — older or duplicate pull arriving after the rebuild result has already been adopted. Discard. |
> | `[a, b]` is a strict subset of some existing segment's range | **Stale** — superseded by a merge that arrived first. Discard. |
>
> Outcome for an **unloaded** index (no `LSMIndexBufferSlot` for this `indexRelId`): adoption is **skipped**; the file persists on disk and is incorporated by `load_lsm_index_internal` ([lsmindex.c:540](../../../pgvector/src/lsmindex.c#L540)) on first backend access (its `scan_segment_metadata_files` picks up the file along with any base-synced segments).

**Invariant: for a loaded index, the "no coverage" case cannot occur.** Pulls are enqueued only at `SegmentCreated` / `SegmentReplaced` redo. By the time the standby reaches that redo, WAL ordering guarantees:

- For a flush of memtable `M`: `Register(M)`, every `Add(M, …)`, and `Register(M+1)` (emitted during `rotate_growing_memtable` at [lsmindex.c:1186](../../../pgvector/src/lsmindex.c#L1186) — flush only happens after seal, seal only after rotation) have all been redone. Memtable `M` is therefore in `SharedMemtableBuffer`, populated, and sealed.
- For a merge `[s1, s2] → s_new`: each input segment's own `SegmentCreated` / `SegmentReplaced` is WAL-ordered earlier; each input is either already an adopted segment in the pool or still represented by sealed memtables.

Subsequent WAL replay only adds state — it never strips a prerequisite, because the only operation that could strip one is *another* adoption replacing it with a wider segment, in which case the present pull's range becomes subset of that wider segment and the outcome is "stale" rather than "no coverage."

### Why this is sufficient for snapshot correctness

- Memtable state on the standby is driven by rmgr redo synchronously with WAL replay. At any replay LSN, the memtable buffer reflects all inserts/tombstones with WAL LSN ≤ replay LSN.
- A segment created from memtable M is *the same data as memtable M* — choice of representation is a storage decision, not a correctness one.
- Top-k search returns tids; PG's `HeapTupleSatisfiesVisibility` filters at heap-fetch time. Any "extra" entry the index might surface (because of replay being ahead of the query's snapshot LSN) is dropped at the heap step. No false positives.
- The search path's existing snapshot-of-pool behavior in `vector_search` ([vector_index_worker.c:1716](../../../pgvector/src/vector_index_worker.c#L1716)) and `search_lsm_index` ([lsmindex.c:1525](../../../pgvector/src/lsmindex.c#L1525)) is unchanged — it already takes a ref-counted snapshot and is unaffected by concurrent adoption.

### Walk-through of representative cases


| Scenario                          | Pool / memtables before            | Arrival                              | Outcome                                                                 |
| --------------------------------- | ---------------------------------- | ------------------------------------ | ----------------------------------------------------------------------- |
| Plain flush                       | memtables 5,6,7                    | seg[5,5]                             | Replace memtable 5.                                                     |
| Pull lags WAL                     | memtables 5,6 (no Register(7) yet) | seg[5,5]                             | Adopt — only needs memtable 5, which is sealed at SegmentCreated(5) redo time. |
| Merge result                      | seg[5,5], seg[6,6]                 | seg[5,6]                             | Replace both.                                                           |
| Merge before flushes              | memtables 5, 6                     | seg[5,6]                             | Replace memtables 5 & 6 (both sealed by the time SegmentReplaced redoes). |
| Flush after merge already adopted | seg[5,6]                           | seg[5,5]                             | Stale; discard.                                                         |
| Out-of-order pulls                | memtables 5,6,7                    | seg[5,6] then seg[5,5] then seg[6,6] | Adopt [5,6]; later [5,5] and [6,6] stale.                               |
| Rebuild                           | seg[5,5] v1                        | seg[5,5] v2                          | Replace seg[5,5] (same range, new version).                             |
| Older version arrives late        | seg[5,5] v2                        | seg[5,5] v1                          | Stale; discard (`arrival.version < pool.version`).                      |


### Adoption: bitmap merge

Adoption replaces a group `G` of standby-local representations with the pulled segment. The pulled bitmap is at primary's state at flush/merge time (see §11 for the protocol guarantee that pulled state is never ahead of the standby's WAL-replayed state). The standby's local bitmaps on `G` may have additional WAL-applied deletions accumulated during the pending window.

Adoption builds the new segment's bitmap as the **union** of the pulled bitmap and the (translated) local bitmaps. Translation is required when the local representation is at a different *version* (e.g., the predecessor is `[5,5] v1` but the pulled segment was built from `[5,5] v2`, after a rebuild that compacted out deleted entries). Translation is **tid-based**, using a lazy sorted-permutation auxiliary structure on the pulled segment's mapping (see §13.3).

---

## 9. Subproblem D — Standby memtable replayer

### Mechanism

Driven by the custom-rmgr redo callbacks on:

- `Register(sid)` → if the index is loaded, allocate a memtable slot via `register_and_set_memtable(is_recovery=true, sid)`-equivalent path ([lsmindex.c:268](../../../pgvector/src/lsmindex.c#L268)). The standby's recovery flag suppresses the primary-side `RegisterStatusMemtable` call (because WAL replay already updated the status page).
- `Add(sid, slot_index, tid)` → if the memtable for `sid` exists, look up the heap row by tid (`table_tuple_fetch_row_version` with `SnapshotAny`, as already done in recovery at [lsmindex.c:854](../../../pgvector/src/lsmindex.c#L854)), copy the vector into `mt->vector_blob` at the named slot, set `mt->tids[slot_index] = tid`, publish via `publish_slot_release`. Slot allocation is deterministic because the primary's slot index is included in the record.
- `MemtableTombstone(sid, slot_index)` → if the memtable for `sid` exists in `SharedMemtableBuffer`, `SET_SLOT(mt->bitmap, slot_index)`. Otherwise the in-memory effect is **skipped** — see "When the memtable is absent" below. The page-level status-page modification is always applied via PG's buffer-cache redo.
- `Release(sid)` → status-page cleanup; **does not** free the standby's `SharedMemtableBuffer` slot. The slot is freed by adoption (Subproblem E), not by `Release`.

### When the memtable is absent at redo time

`MemtableTombstone` redo is the one record type that can encounter an absent memtable for two distinct reasons:

1. **Index not yet loaded** on this standby (no `LSMIndexBufferSlot` for `indexRelId`). The page-level status-page change still applies via PG's buffer-cache redo. The in-memory bitmap update is skipped. When `load_lsm_index_internal` later runs, it rebuilds the memtable from heap using post-redo status-page state, which already reflects every preceding `MemtableTombstone` — so the rebuilt memtable is equivalent to one that had been kept current by in-memory redo throughout.

2. **Memtable already adopted into a segment** on this standby. The primary's `bulk_delete_lsm_index` emits both `MemtableTombstone` (per tid) and `SegmentVacuumTombstones` (per batch, later in WAL) when vacuuming a memtable whose corresponding segment already exists on disk — the `is_persistent` branch at [lsmindex.c:1680](../../../pgvector/src/lsmindex.c#L1680). The standby's pull and adoption for that segment can complete between the two records' replay. `MemtableTombstone` then finds no memtable — in-memory effect skipped. The subsequent `SegmentVacuumTombstones` in the same vacuum batch then applies the deletion to the adopted segment's bitmap. No deletion is lost.

`Register` and `Add` cannot encounter case 2: by the WAL-ordering argument of §8, adoption only fires after `SegmentCreated` redo, and `SegmentCreated` is itself WAL-ordered after every `Register(M)` and `Add(M, …)` for the flushed memtable. They can only encounter case 1 (index unloaded), with the same "page-level applies; in-memory skipped; rebuild handles it" pattern.

The general principle: rmgr redo for memtable records applies the page-level change unconditionally; the in-memory effect on `SharedMemtableBuffer` is conditional on the memtable being present. Correctness rests on the fact that disk-resident state — status pages for index-load reconstruction, and `SegmentVacuumTombstones` for adopted-segment deletions — is independently kept current.

### Why this is single-threaded-safe

rmgr redo runs in the startup process, which is single-threaded. No concurrent backends mutate the memtable while redo runs (backends only read). The standby's `ConcurrentMemTable` already supports concurrent reads against a writing flow; redo plays the role of the single writer.

### Initial load

When an index is first touched on the standby, `load_lsm_index_internal` ([lsmindex.c:495](../../../pgvector/src/lsmindex.c#L495)) runs as today: scan disk segments, read status pages, materialize memtables from heap. This is the initial bootstrap, identical to crash recovery on the primary. After load, ongoing rmgr redo keeps the state current.

### Redo during index loading (v1 policy: block)

A coordination problem arises between the startup process running rmgr redo and the IndexLoadWorker running `load_lsm_index_internal`. The load reads status pages and heap *sequentially*; between any two of its reads, the startup process can redo a WAL record that mutates the same status pages.

If redo applied the "absent memtable" policy literally during load (treat `valid==2` like `valid==0`, skipping in-memory effects), records replayed during the load window would have their page-level effects applied but their in-memory effects lost permanently — `load_lsm_index_internal`'s reconstruction missed them, and subsequent redos (with `valid==1`) don't retroactively close the gap. Queries on the standby would miss those entries until the next reload (typically the next standby restart). Correctness break.

**v1 policy: block.** When a redo callback for a status-page record encounters `valid==2` for the target index, it sleeps on the existing `load_cv` (`ConditionVariable` in `LSMIndexBufferSlot`) until `valid` transitions to 1 (success) or 0 (failure). The pattern mirrors the existing backend-side wait at [lsmindex.c:181-193](../../../pgvector/src/lsmindex.c#L181). After the wake, the redo callback re-checks `valid` and either applies in-memory + page-level (if 1) or skips the in-memory effect (if 0, per the §9 absent-memtable policy).

**Consequence:** WAL replay stalls for the duration of an in-progress load. Because the startup process is single-threaded, the stall affects records for *all* indexes, not just the loading one. For lazy loading on a first user query, the stall overlaps with that query's wait — net user-visible latency is roughly unchanged. For workloads that touch many indexes shortly after a fresh standby start, total stall ≈ sum of individual load durations, which can be mitigated by triggering eager loads at standby startup (a startup hook that opens each known LSM index) so all loads complete before any user-facing workload arrives. PG's `max_standby_streaming_delay` is the standard backstop if replay lag becomes a problem under sustained load.

**Future direction (Option C — non-blocking queue):** A production-grade replacement for v2+ would replace the wait with a per-index ring buffer: when redo sees `valid==2`, it appends `(record_lsn, semantic fields)` to the buffer and returns immediately without stalling replay. `load_lsm_index_internal`, after building its initial state, atomically drains the queue, applies the queued records' in-memory effects in WAL order, and sets `valid=1`. This decouples replay from load progress at the cost of extra mechanism (queue size bound, drain-vs-append ordering at the `valid=2→1` transition, recovery if the queue overflows). Tracked in §16.

### Heap reads during redo

`Add(sid, slot, tid)` redo reads a heap row from the standby's heap. This requires the heap to be at-or-ahead of the WAL record's LSN. Since the `Add` record is emitted *after* the heap insert on the primary, and WAL replay is in order, the heap row is guaranteed to be present in the standby's heap by the time the `Add` is redone. (PG's heap-insert WAL is replayed before our `Add`.)

---

## 10. Subproblem E — Cross-channel coordination

### The persistent fetch queue

A persistent on-disk queue under `VECTOR_STORAGE_BASE_DIR/_pending_fetches/`. One small file per pending entry, each entry holding:

```
indexRelId       Oid
start_sid        SegmentId
end_sid          SegmentId
version          uint32
kind             {Created, Replaced}
source_lsn       XLogRecPtr      // for diagnostics; not used by adoption logic
status           {pending, fetching, done, failed}
offsets[]        for Replaced records only — see §13.5
```

### Why persistent (not derived from WAL on restart)

If the queue were in-memory only and rebuilt by replaying WAL from the restartpoint, then the restartpoint would have to be held back to before any pending fetch's LSN — coupling our liveness to PG's checkpoint logic. Persisting the queue decouples that. Standby restart: read the queue directory, re-enqueue everything still pending.

### Queue interactions

- **Redo callbacks** for `SegmentCreated` and `SegmentReplaced` write a new queue entry, then return. The entry's existence is the durable signal; the in-memory queue is just an optimization.
- **SegmentFetcher workers** pop entries, pull files, attempt adoption, then mark `done`. Per §8, adoption for a loaded index always terminates as either "adopt" or "stale-discard" — there is no "pending" outcome. For an unloaded index, adoption is skipped (file stays on disk for `load_lsm_index_internal` to pick up later); the queue entry is still marked `done`.
- **Failures** (timeout, file-not-found, checksum) mark `failed`; bumped by a periodic retry. Cluster-wide diagnostic counter exposed via a function in the extension.

### Vacuum WAL during pending fetch

When `SegmentVacuumTombstones(target=[a,b] vK, …)` is redone and the matching segment is not yet adopted:

1. **Fast path**: pool has a segment with exactly `(a, b, vK)` — apply bits to its bitmap, write a local subversion file.
2. **Slow path 1 (predecessor with same range, older version — rebuild race)**: lazy sorted-permutation lookup on the pulled segment's mapping (built on-demand) — see §13.3 for cost analysis. Find each tombstone tid in the predecessor's mapping, set its bit.
3. **Slow path 2 (predecessors with narrower ranges — merge race)**: same tid lookup, but across multiple predecessor segments. Each tid lives in exactly one predecessor.
4. **Slow path 3 (memtables not yet flushed)**: tid lookup against `mt->tids[]` (linear; memtables are small).

For the three segment-targeted paths (fast path, slow paths 1 and 2), the local subversion file is written **once per rmgr record, after the in-memory bitmap update**, with the primary's subversion number (carried in the rmgr record). One WAL record ↔ one subversion file write, with one fsync apiece. This matches the primary's discipline at [lsmindex_io.c:826](../../../pgvector/src/lsmindex_io.c#L826) (one `write_bitmap_file_with_subversion` per call-site invocation).

For slow path 3 (memtable target), there is **no** subversion file write — memtables have no disk-backed bitmap to subversion. The in-memory `mt->bitmap` update is the only effect; durability comes from WAL re-replay on the next standby restart (which is fine because adoption hasn't dropped the memtable yet, so re-replay reliably re-applies the tombstone).

On adoption, the union step in §8 picks up any local subversions written during the pending window.

**Why per-redo rather than batched/deferred?** The in-memory bitmap is only durable in two places: the rmgr WAL records that produced it, and the on-disk subversion file. If PG's restartpoint advances past those WAL records before a deferred file write happens, in-memory state would be lost on the next standby restart. Per-redo writes keep us decoupled from PG's restartpoint advancement (matching the same reasoning that motivated the persisted fetch queue earlier in this section). Batching is a possible follow-on if fsync rate is measured to be a bottleneck.

### Memtable lifetime relative to `Release` WAL

The standby's `SharedMemtableBuffer` slot for sid M persists until *adoption* drops it — not until `Release(M)` is redone. `Release` redo only touches status pages. This makes the cross-channel race tractable: even if `Release(M)` replays before the segment is fetched, queries continue to find sid M's data in the memtable.

### Standby crash before pull

If the standby crashes between `Release(M)` redo and the segment fetch completing, the `SharedMemtableBuffer` slot is lost (it's in shmem). On restart, the status page has no entry for M (it was released). `load_lsm_index_internal`'s rebuild logic doesn't recreate the memtable. The data is *only* reachable via the segment file.

If the segment file has been fetched (metadata present), it's loaded normally. If not, the index is **not queryable** for that sid until the fetcher catches up.

We make this safe by **holding the index `not queryable`** at attach time until the SegmentFetcher confirms that every `SegmentCreated` / `SegmentReplaced` record in WAL since the restartpoint corresponds to a file on disk. Queries against this index wait (recovery-conflict-style, bounded by a GUC analogous to `max_standby_streaming_delay`) until the fetcher catches up, then proceed.

---

## 11. Critical protocol — WAL emit before file write

**This is the most load-bearing single decision in the design. It must be enforced rigidly.**

### What the protocol says

For every event that produces a new file on disk and a corresponding rmgr WAL record, the primary must:

1. Compute the change (vacuum sweep, merge output, flush serialization).
2. `XLogInsert(...)` to obtain LSN_record.
3. `XLogFlush(LSN_record)` so the WAL is durable.
4. Write the file via temp-then-rename.

The order matters because of how concurrent operations interleave on the primary.

### Why this matters: the merge-race scenario

Concurrent vacuum + merge on the primary can race in `bulk_delete_lsm_index` step 3 ([lsmindex.c:1898](../../../pgvector/src/lsmindex.c#L1898)) vs `merge_adjacent_segments_pool` ([vector_index_worker.c:1085](../../../pgvector/src/vector_index_worker.c#L1085)). Vacuum does not hold `per_seg_mutex` while writing its subversion file ([lsmindex.c:1994](../../../pgvector/src/lsmindex.c#L1994)). Merge reloads bitmaps inside `per_seg_mutex` at [vector_index_worker.c:1196](../../../pgvector/src/vector_index_worker.c#L1196).

If vacuum writes its subversion file *before* emitting WAL:

- Merge could reload the subversion (bitmap now has X marked), bake X into the merged segment file, and emit `SegmentReplaced` (LSN_m).
- Vacuum then emits `SegmentVacuumTombstones` (LSN_v) **with LSN_v > LSN_m**.

The standby would then:

- Replay LSN_m → enqueue pull of merged segment.
- Pulled file arrives with X marked.
- Adoption fires while replay is still between LSN_m and LSN_v.
- During this window, the standby's adopted segment has X marked, but the standby's WAL-replayed state does *not* have the tombstone applied.

A backend on the standby with snapshot LSN < LSN_v can see the heap tuple (heap-vacuum WAL is even further ahead) but cannot find it in the index. **Correctness break.**

### Why the protocol fixes it

If vacuum emits WAL *first* (`XLogInsert + XLogFlush`), then writes the file:

- LSN_v is assigned before any concurrent operation can observe the file.
- Merge, even if its reload happens after vacuum's file write, must produce its `SegmentReplaced` at LSN_m > LSN_v.
- WAL stream is well-ordered: LSN_v < LSN_m.
- Standby applies `SegmentVacuumTombstones` first, then pulls the merge result.
- At adoption, pulled bitmap and local bitmap both reflect the deletion. No window.

### Specific call sites

The protocol applies at three call sites in `bulk_delete_lsm_index`:


| Site                                                                                             | Today                                                           | Change                                                                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [lsmindex.c:1703](../../../pgvector/src/lsmindex.c#L1703) (memtable already persisted, step 1)   | `write_bitmap_file_with_subversion` → `segment_update_blocking` | Insert `XLogInsert(SegmentVacuumTombstones) + XLogFlush` before `write_bitmap_file_with_subversion`. Also insert `XLogInsert(MemtableTombstone)` per tid earlier in the loop (replacing the `RemoveFromStatusMemtable` GenericXLog use). |
| [lsmindex.c:1868](../../../pgvector/src/lsmindex.c#L1868) (immutable memtable persisted, step 2) | Same                                                            | Same                                                                                                                                                                                                                                     |
| [lsmindex.c:1994](../../../pgvector/src/lsmindex.c#L1994) (segment-only vacuum, step 3)          | Same                                                            | Insert `XLogInsert + XLogFlush` before `write_bitmap_file_with_subversion`.                                                                                                                                                              |


Merge and flush also emit WAL after their file writes (`SegmentReplaced`, `SegmentCreated`); the protocol here is the inverse — file write first, then WAL — because the WAL is what triggers the standby to pull. This direction is safe because the file is finalized (via atomic rename) before the WAL says it exists; the standby never tries to pull a file that hasn't been completely written.

### Implementation note

This protocol must be unmistakably present in the implementation. The plan should include unit tests that assert:

- `SegmentVacuumTombstones` LSN < timestamp of corresponding subversion file's last-modify.
- `SegmentReplaced` / `SegmentCreated` LSN > timestamp of corresponding metadata-file rename.

(In production these timestamps aren't directly available, but a synthetic test fixture can capture both.)

### Residual: index-ahead-of-heap window

Even with the protocol, there's a brief window between replaying `SegmentVacuumTombstones` and the corresponding heap-vacuum WAL where a backend with an old snapshot could find an index entry deleted but the heap row still visible to its snapshot. This is **identical to PG built-in btree behavior** — btree's vacuum WAL also precedes heap-vacuum WAL — and is addressed by either `hot_standby_feedback=on` (primary refuses to vacuum tuples visible to any standby snapshot) or PG's recovery-conflict cancellation when heap-vacuum WAL is replayed. We inherit both for free. The design doc should note this as a known characteristic, not a bug.

---

## 12. Subproblem F — Standby worker topology and query barrier

### Workers under `RecoveryInProgress()`


| Worker                                                                                                                           | On primary                          | On standby                                                                                                              |
| -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `lsm_index_bgworker_main` ([lsmbackground.c:252](../../../pgvector/src/lsmbackground.c#L252))                                    | Polls for sealed memtables, flushes | **Disabled** — `RecoveryInProgress()` check at entry; immediate `proc_exit(0)`                                          |
| Merge thread pool inside `vector_index_worker` ([vector_index_worker.c:1572](../../../pgvector/src/vector_index_worker.c#L1572)) | Started by `init_merge_thread_pool` | **Not started** under recovery; checked at startup                                                                      |
| `vector_index_worker_main`                                                                                                       | Serves search + maintenance         | **Search-only** — `SEGMENT_UPDATE_`* tasks never originate locally because there's no flush/merge/vacuum on the standby |
| `index_load_worker_main` ([index_load_worker.c:70](../../../pgvector/src/index_load_worker.c#L70))                               | First-touch loading                 | Same — needed for standby attach                                                                                        |
| `SegmentFetcher` (new)                                                                                                           | Not present                         | Pulls files from primary; drives adoption                                                                               |
| Custom-rmgr redo                                                                                                                 | Runs in startup process             | Drives memtable buffer + bitmap updates                                                                                 |


`bulk_delete_lsm_index` is reachable only via `amblulkdelete` invoked by VACUUM, which doesn't run on the standby. No code-path guard needed; it just never gets called.

### Search path

Identical to the primary's `search_lsm_index` ([lsmindex.c:1525](../../../pgvector/src/lsmindex.c#L1525)). No standby-specific query barrier for the insertion path — see Subproblem C.

For the **standby restart edge case** (data unreachable until fetcher catches up), the index is held in a "not queryable" state at attach until the fetcher confirms all `SegmentCreated` / `SegmentReplaced` records since restartpoint correspond to local files. A backend that hits an index in this state waits up to `max_standby_streaming_delay`-style timeout, then is canceled. This GUC is added: `pgvector_replication_fetch_wait_timeout` (default 30s).

For vacuum, no barrier needed — index vacuum is WAL-ordered before heap vacuum (see §11 residual paragraph).

---

## 13. Detailed design decisions

### 13.1 Pull base file only; vacuum subversions are local

The secondary pulls only base files (`bitmap_<s>_<e>_v<V>` with no subversion suffix). The secondary's own redo writes local subversion files via `SegmentVacuumTombstones`. Adoption code reads the latest local subversion (which has been advanced by redo) and unions with the pulled bitmap.

**Why not pull the latest subversion at fetch time?** Two reasons:

1. Pulled state could be ahead of WAL replay state, opening Case 1 of the user-flagged correctness scenarios (see §15.1).
2. Pulling base only keeps the side channel low-bandwidth (one file per flush/merge); subversions, which can be much smaller individually but more numerous, ride WAL instead.

### 13.2 `SegmentVacuumTombstones` encoding — chosen: `(local_idx, tid)`

The chosen encoding carries `(uint32 local_idx, int64 tid)` per tombstone. 12 bytes per entry.

- **Fast path** (matching segment in pool): use `local_idx` directly — O(1) per entry, just `SET_SLOT(seg->bitmap, local_idx)`.
- **Slow path** (predecessor segment in pool, e.g., during a pending fetch of a rebuild or merge target): look up `tid` in the predecessor's mapping via a lazy sorted-permutation auxiliary structure (§13.3). O(log N) per entry.

### 13.3 Lazy sorted-permutation auxiliary structure for tid lookup

For each segment, a `uint32 sorted_idx[N]` is built **on first slow-path touch**, where `sorted_idx` is a permutation over `[0, N)` such that `seg->map_ptr[sorted_idx[i]]` is in tid order. Binary search on `seg->map_ptr` via `sorted_idx` resolves tid → `local_idx` in O(log N).

Memory cost: 4 N bytes per segment that's been slow-path-touched. For HNSW with dim=768 (typical embedding), this is ~0.1% of segment size. For dim=128, ~0.6%. For DiskANN (off-heap vectors), more visible but still <10%. Cached for the segment's lifetime.

Build cost: O(N log N) once per segment, on the first vacuum WAL record that hits the slow path.

### 13.4 Alternatives to #13.3 (recorded, not chosen for v1)

**Alternative #2 — eager sorted permutation.** Build the `sorted_idx` array at segment-attach time inside `load_and_set_segment` ([lsm_segment.c:430](../../../pgvector/src/lsm_segment.c#L430)). Trade-off: pays memory and CPU for every segment regardless of whether the slow path is ever exercised. Promote to this if profiling shows the lazy first-touch build is a perceptible stall on standby.

**Alternative #3 — `valid_id` encoding + offset enrichment of `SegmentReplaced`.** Replace `(local_idx, tid)` with `(local_idx, valid_id)` in `SegmentVacuumTombstones`, where `valid_id` is the rank of the entry among non-deleted entries in its segment. Enrich `SegmentReplaced` with the merged segment's offset map: `{(sid, start_offset), …}` (typically 16 bytes for a two-input merge). 

Resulting properties:

- **Fast path** unchanged — `local_idx` direct.
- **Rebuild race** — single O(N + M) walk of predecessor's `map_ptr` with an offset cursor counting non-deleted entries; no aux structure.
- **Merge race** — offset map (from `SegmentReplaced`) translates `local_idx_merged → (source_sid, source_local_idx)` arithmetically; no aux structure.
- WAL size: 33% smaller (`8` bytes vs `12` bytes per tombstone).

Required invariants for #3 (must hold or this encoding breaks):

- `REBUILD_DELETION` preserves the rank of surviving entries.
- Merge preserves source's `local_index` ordering verbatim (no compaction at merge time; true today at [vector_index_worker.c:1252-1256](../../../pgvector/src/vector_index_worker.c#L1252-L1256)).

**Adoption translation still needs tid lookup** in #3 as well: when the pulled segment finally lands and the secondary unions in its local bitmap from predecessor segments at different versions, identifying matching slots across versions requires the stable tid identifier.

**Choice for v1:** #1 (lazy sorted permutation with `(local_idx, tid)` encoding). It's the most robust to future changes in merge/rebuild semantics — tids are stable identifiers regardless of layout — and the WAL size cost is bounded. #3 is an attractive optimization if vacuum WAL volume becomes a measured concern; the design doc records it so we know what changes if we adopt it later.

### 13.5 `SegmentReplaced` carries offset info (regardless of encoding choice)

Even with the `(local_idx, tid)` encoding chosen for v1, including the offset map in `SegmentReplaced` is useful: it lets the standby make adoption decisions and resolve overlap relationships without first having to pull the merged file's offset file. The offset map is small (one entry per source sid; usually 2 entries for a two-input merge).

For v1: include `offsets[]` in `SegmentReplaced` as a forward-compatibility hook. The redo path doesn't strictly need it under the chosen encoding (we'll use tid lookup for cross-channel races), but having it costs almost nothing and unlocks the alternative-#3 path with no further format change.

### 13.6 Asynchronous segment pull

The custom-rmgr redo for `SegmentCreated` and `SegmentReplaced` **must not** perform the pull synchronously. Pulls are bandwidth-bound (multi-MB to multi-GB per segment); doing them inline blocks the standby's startup process for the entire pull duration, stalling all subsequent WAL replay including memtable inserts on unrelated indexes.

Architecture (also see §10):

```
Redo callback (startup process)
  └─ Append entry to persistent fetch queue.  Return.  (~5 µs)

SegmentFetcher worker (background, 2–4 workers in parallel)
  ├─ Pop entry from queue.
  ├─ Check if metadata file already present (from rsync base sync, or
  │  from a prior partial run).  If yes, skip pull; go to adoption.
  ├─ Pull index / mapping / offset / bitmap / metadata files
  │  (metadata last; atomic rename = "complete").
  ├─ Attempt adoption (coverage rule, §8).
  │  Loaded index, coverage matches    → replace pool entries, union bitmaps.
  │  Loaded index, range already       → stale; delete the pulled file.
  │    subsumed by wider segment
  │  Index not loaded                  → skip; file persists for
  │                                      load_lsm_index_internal to pick up.
  └─ Mark queue entry done.
```

Out-of-order adoption is handled by the coverage rule (§8 walk-through table).

### 13.7 Build-system hooks

- `RegisterCustomRmgr` requires PG ≥ 15. The Makefile's `PG_CONFIG` ([pgvector/Makefile](../../../pgvector/Makefile)) must point at a PG 15+ build.
- New OBJS: `replication_rmgr.o`, `segment_fetcher.o`, `replication_server.o` (or analogous names).
- The extension control file needs no change; this is all internal.

---

## 14. Invariants

The design relies on these invariants. They must be asserted (where checkable) and documented at the relevant call sites.

1. **PG vacuum order is preserved.** `amblulkdelete` runs before `lazy_vacuum_heap` in PG's vacuum loop. The standby therefore sees index-vacuum WAL before heap-vacuum WAL.
2. **Memtable slot-index ≡ segment vector-index for a pure flush.** `prepare_for_flushing` ([lsmbackground.c:88](../../../pgvector/src/lsmbackground.c#L88)) writes vectors in slot order with no remapping. Cross-representation tombstone application (memtable ↔ freshly-flushed segment) relies on this.
3. **Tids are stable across rebuild and merge.** A tuple's `(block, offset)` identity doesn't change because we reorganize the index.
4. **Rebuild-deletion removes only entries already marked in the prior version's bitmap.** A tid deleted by vacuum on `vN+1` must have been present (and not yet vacuumed) in `vN`.
5. **Merge preserves all non-deleted source tids.** Inherently true from [vector_index_worker.c:1242-1245](../../../pgvector/src/vector_index_worker.c#L1242-L1245).
6. **Primary emits `SegmentVacuumTombstones` WAL before writing the corresponding subversion file.** This is the WAL-emit-before-file protocol of §11.
7. **Primary emits `SegmentCreated` / `SegmentReplaced` WAL after writing (and atomically renaming) the corresponding files.** This is the inverse direction of §11.
8. **Merge picks up the latest bitmap before producing the merged file.** Already true at [vector_index_worker.c:1196](../../../pgvector/src/vector_index_worker.c#L1196).
9. **Standby's `RecoveryInProgress()` is true on standby for the lifetime of the standby's read role.** Workers gate on this.
10. **Custom-rmgr redo callbacks are idempotent.** WAL replay may re-apply a record after a standby restart; bit-OR and slot-array writes are naturally idempotent.

## 15. Failure scenarios and how the design handles them

### 15.1 The three correctness cases the design must defeat


| Case                                                                                       | Source                                                                                         | How the design prevents it                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Segment "from the future" — pulled bitmap reflects vacuum the standby hasn't replayed      | Cross-channel race if pulled file includes subversions written post-WAL-emit but pre-WAL-flush | Pull base file only (§13.1) + WAL-emit-before-file-write protocol (§11). Pulled bitmap is at flush/merge LSN; standby's WAL replay at adoption ≥ that LSN, so all prior vacuums are already locally applied.                                                                           |
| Segment "from the past" — pulled bitmap predates a vacuum the standby has already replayed | Cross-channel race in reverse                                                                  | Union-at-adoption (§8) preserves local WAL-applied bitmap. Pulled bitmap contributes only what it has; local bitmap covers everything else.                                                                                                                                            |
| Partially-vacuumed segment from concurrent vacuum + merge                                  | Vacuum + merge interleave on primary; merged file written with X but not Y                     | Tid-based slow-path redo (§13.3); vacuum's retry-loop in `bulk_delete_lsm_index` step 3 ([lsmindex.c:1915](../../../pgvector/src/lsmindex.c#L1915)) re-issues the WAL for Y on the merged segment; union-at-adoption combines everything. End state on standby converges to primary's. |


### 15.2 Standby crash before pull completes

Memtable shmem is lost; status pages have been cleaned. Data not in any segment file yet.

Mitigated by holding the index in "not queryable" state at attach until the fetcher catches up. Queries wait up to a GUC timeout, then are canceled. After fetcher catches up, the index becomes queryable. No data loss because the segment file exists (or will exist) on the primary.

### 15.3 Primary unreachable

Fetcher retries with exponential backoff. Standby remains in degraded state for affected indexes (existing pool serves queries, but no new segments arrive). Resolves when primary returns.

### 15.4 Partial pull mid-flight on standby crash

Temp files cleaned at standby restart. Queue re-enqueues the entry. Pull restarts from scratch (we don't bother with range-resume in v1).

### 15.5 Misaligned PG cluster state (rsync skew)

If the operator runs the initial `rsync` of `VECTOR_STORAGE_BASE_DIR` while the primary is concurrently writing, some files may be partial. Mitigation: rsync should run after `pg_basebackup` start LSN; metadata files that fail integrity checks (corrupt or truncated) are re-fetched.

## 16. Open questions

These are deferred from this design and need to be settled during planning or early implementation. They are *not* gating items for the overall direction.

1. **Side-channel protocol concrete shape.** TCP vs HTTP, authentication mechanism, framing. Default proposal: raw TCP with a length-prefixed binary protocol and a shared-secret token; revisit if cluster-management constraints require HTTP.
2. **Configurable `VECTOR_STORAGE_BASE_DIR`.** Today hardcoded ([lsmindex.h:203](../../../pgvector/src/lsmindex.h#L203)). For multi-node, must become a GUC or env-var driven path.
3. **Restartpoint integration.** Persisted queue (chosen) decouples our liveness from PG's restartpoint, but we should verify that standby checkpoints don't cause data loss in pathological corner cases (e.g., very-long-running pending fetch combined with WAL recycling).
4. **Backpressure on WAL volume from bulk vacuums.** A 1-million-tid vacuum emits ~12 MB of WAL in `SegmentVacuumTombstones` records. Acceptable for v1 but worth measuring. The §13.4 alternative (encoding #3) saves 33% if this becomes an issue.
5. **Concurrent redo + adoption synchronization.** Redo writes local subversion files on the standby while the fetcher might be reading/replacing in the same segment slot. We need a clear ownership story for the per-segment data structures; current locks (`pool->seg_lock`, `per_seg_mutex`) should suffice but the design needs an explicit lock-order documentation.
6. **GUC names.** Pre-finalize: `pgvector_replication_role`, `pgvector_replication_primary_host`, `pgvector_replication_primary_port`, `pgvector_replication_shared_secret`, `pgvector_replication_fetch_wait_timeout`, `pgvector_replication_fetch_parallelism`.
7. **Non-blocking redo during index loading (Option C, planned for v2+).** v1 blocks WAL replay while a `valid==2` index is being loaded by the IndexLoadWorker (see §9, "Redo during index loading"). The follow-on is a per-index ring buffer the redo callback appends to without stalling, drained by the load worker before it sets `valid=1`. Needs a queue size bound, a clean drain-vs-append ordering at the `valid=2→1` transition, and an overflow-recovery story (presumably: fall back to the v1 blocking behavior). Worth doing once the v1 stall pattern shows up in production measurements.

## 17. Out of scope for v1 (explicit non-goals revisited)

- Promotion / failover. (The new primary would need to know which standbys have which segments, when it's safe to recycle WAL slot for a vacuum, etc. None of that infrastructure is designed here.)
- Multi-standby fan-out. (One file server on primary serving many standbys is feasible later.)
- Cascading replication.
- Logical replication (decoupled-pgvector indexes are extension state; logical decoding would require a separate, much larger design.)

## 18. References to existing code (consolidated)


| Concern                                    | File                                                                 | Function / location                                                                  |
| ------------------------------------------ | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Status page primitives                     | [statuspage.c](../../../pgvector/src/statuspage.c)                   | `Register/Add/Remove/Release/Update`*                                                |
| Memtable buffer                            | [lsmindex.h](../../../pgvector/src/lsmindex.h)                       | `ConcurrentMemTableData`, `MemtableBuffer`                                           |
| Memtable replay (bootstrap)                | [lsmindex.c](../../../pgvector/src/lsmindex.c)                       | `load_lsm_index_internal`                                                            |
| Flush worker                               | [lsmbackground.c](../../../pgvector/src/lsmbackground.c)             | `lsm_index_bgworker_main`, `prepare_for_flushing`                                    |
| Vector index worker (search + maintenance) | [vector_index_worker.c](../../../pgvector/src/vector_index_worker.c) | `vector_index_worker_main`, maintenance pool, merge pool                             |
| Index-load worker                          | [index_load_worker.c](../../../pgvector/src/index_load_worker.c)     | `index_load_worker_main`                                                             |
| Segment pool                               | [lsm_segment.c](../../../pgvector/src/lsm_segment.c)                 | `register_flushed_segment`, `replace_flushed_segment`, `find_segment_by_sids`        |
| Segment IO                                 | [lsmindex_io.c](../../../pgvector/src/lsmindex_io.c)                 | `flush_segment_to_disk`, `write_bitmap_file_with_subversion`, `load_and_set_segment` |
| Vacuum entry point                         | [lsmindex.c](../../../pgvector/src/lsmindex.c)                       | `bulk_delete_lsm_index`                                                              |
| Merge                                      | [vector_index_worker.c](../../../pgvector/src/vector_index_worker.c) | `merge_adjacent_segments_pool`                                                       |
| Extension init                             | [vector.c](../../../pgvector/src/vector.c)                           | `_PG_init`                                                                           |


---

*End of design.*