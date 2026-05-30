# Replica Search Performance Evaluation — Design

**Status:** Design — pending implementation plan.
**Date:** 2026-05-26
**Scope:** Two evaluation drivers + guidance for measuring read-only-standby search behavior of the physical-replication path, using the msturing-10M sliding-window runbook.

---

## 1. Goals

Measure how the read-only standby performs against the primary on the
`msturing-10M_slidingwindow_runbook.yaml` runbook (96-dim float32, **400 ops**:
steps 1–100 insert 5M vectors; steps 101–400 cycle `[search, delete, insert]`,
i.e. 100 logical rounds of 10K searches / 50K deletes / 50K inserts).

- **Experiment A — Recall vs. step.** Sequential ("non-mixed") runbook on the
  primary. Before each search step, wait for the standby to catch up; compute
  average recall on **both** primary and standby for that step.
  Output: `experiment_a.csv` with columns `step,primary_recall,standby_recall`.
- **Experiment B — Throughput vs. time.** Mixed-mode runbook on the primary
  (group size 3). On the standby, client threads issue random queries from the
  query set continuously (no pausing); record per-query completion timestamps
  and aggregate into per-second QPS. On the primary, record a wall-clock
  timestamp at the start of each runbook step.
  Outputs: `standby_throughput.csv` (`timestamp_ms,qps`) and
  `step_boundaries.csv` (`step,timestamp_ms`).

## 2. Confirmed decisions

| Topic | Decision |
|-------|----------|
| Cluster provisioning | **Externally managed.** Primary and standby are started/configured by the operator; drivers connect via `host:port` (no `pg_ctl`/`initdb` from the scripts). |
| Language / tooling | **C++**, reusing the proven helpers in `pgvector_test.cpp`. Two new standalone programs + a shared header; `pgvector_test.cpp` is left untouched. |
| Topology | **Different hosts.** Primary and standby run on separate machines → the two CSVs are produced on two hosts and need a shared timeline (see §6, clock module). |
| "Standby caught up" (Exp A) | **WAL-replay catchup only** — wait until the standby's `pg_last_wal_replay_lsn()` ≥ the primary's `pg_current_wal_lsn()` captured at step start. No segment-fetcher settle wait (correctness holds via memtable redo; fetcher is efficiency-only). An optional `--standby-settle-ms` knob exists, **default 0**. |
| Exp A start mode | **Both supported.** `--start-step` / `--build-index-before` are configurable: seed-at-101 (5M pre-inserted + index built + replicated) for fast iteration, or full-from-1 for a self-contained run. |

## 3. Non-goals

- No cluster setup/teardown, `pg_basebackup`, or GUC editing by the scripts (operator does this; the guidance doc lists the required GUCs).
- No ground-truth generation — GT `.npy` files already exist in `/ssd_root/liu4127/msturing_runbook_gt` (100 files, steps 101…400 stride 3, matching `compute_ground_truth.cpp`'s hashing). Exp B needs no GT.
- No failover/promotion, no multi-standby, no new SQL monitoring functions.
- No modification of extension C/C++ source. This is test/benchmark tooling only.

## 4. Components & file layout

All new files under `pgvector/test/runbooks/replication/` (sibling to `pgvector_test.cpp` in the parent dir, sharing its dataset/GT conventions), plus one guidance doc under `docs/`.

```
pgvector/test/runbooks/replication/
  runbook_common.hpp          # shared helpers (new)
  replica_recall_eval.cpp     # Experiment A (new)
  replica_throughput_eval.cpp # Experiment B, --role primary|standby (new)
  Makefile                    # build targets for the two programs (new)
docs/
  replica_search_evaluation.md # configure + conduct guidance (new)
```

### 4.1 `runbook_common.hpp`

Self-contained header (no dependency on `pgvector_test.cpp`) holding clean,
standalone versions of the helpers the two programs share:

- `DataLoader` — `.fvecs` / `.fbin` loaders **and** a seekable on-demand reader (`open_vec_reader` / `read_vec`) so the 5M-vector dataset is not fully resident.
- `vector_to_sql(const float*, dim)` — `'[...]'` literal, `%.6f`.
- `gt_filename(runbook_id, step, active_ranges)` — MD5 over the active-ranges JSON, first 4 bytes hex (byte-for-byte identical to `compute_ground_truth.cpp` / `pgvector_test.cpp`).
- `load_gt_npy(path, num_queries, k)` — npy reader (6-byte magic, version, uint16 header len, then `num_queries*k` int32).
- Runbook reader (yaml-cpp): parse `dataset_name` section, produce ordered steps `{step_num, op, start, end, k}`, skipping `max_pts`/`query`/`groundtruth`; helper to build the `active_ranges` prefix up to a given start step (for GT hashing parity).
- `ThreadLocalConnection` — per-thread libpq connection, as in `pgvector_test.cpp`.
- **Clock module** (see §6): `prime_clock_offset(primary_conninfo)`, `now_primary_ms()`.

### 4.2 `replica_recall_eval.cpp` — Experiment A

Connects to **primary** and **standby**. Walks the runbook **sequentially**.

Flow per step (within `[--start-step, --end-step]`, default 101–400):
1. If `--build-index-before == step` and not yet built: `CREATE INDEX … hnsw|ivfflat` on the **primary**; wait for catchup.
2. `insert` → batched `INSERT … ON CONFLICT DO NOTHING` on the **primary** (reusing `--checkpoint-size` batching); push `["insert",s,e]` to active_ranges.
3. `delete` → `DELETE … WHERE id >= s+offset AND id < e+offset` on the **primary**; push `["delete",s,e]`.
4. `search`:
   a. **wait_catchup**: read `pg_current_wal_lsn()` on primary; poll standby `pg_last_wal_replay_lsn()` until `pg_wal_lsn_diff(replay, target) >= 0` (timeout via `--catchup-timeout-sec`); optional `--standby-settle-ms` sleep (default 0).
   b. Load GT `.npy` via `gt_filename(runbook_id, step, active_ranges)`; if missing, log + record `NaN` and continue.
   c. Compute avg recall over `N = min(--num-verify-queries, |queries|)` (default 10000) on the **primary** (queries parallelized with `--threads`; recall is order-independent). Recall per query = |returned ∩ gt[:k]| / k.
   d. Same on the **standby**.
   e. Append `step,primary_recall,standby_recall` (6-dp) to `experiment_a.csv` and also emit a one-line progress diag.

Mutations are primary-only; the standby receives them via replication. `k` comes from the runbook step (default 100), matching the GT column count.

### 4.3 `replica_throughput_eval.cpp` — Experiment B

One process per host, run concurrently. `--role` selects behavior.

**`--role primary`** (on/near the primary host, connects to the primary):
- Mixed-mode walk from `--mixed-mode-start` (default 101) in groups of `--mixed-size` (default 3), reusing the shuffled work-array scheme from `pgvector_test.cpp`'s mixed mode (insert/delete/search ops of a group interleaved across `--threads`). Searches here are pure load (no recall).
- At the **start of each group**, append `step,timestamp_ms` to `step_boundaries.csv`, where `step` is the group's lead step number (101, 104, …) and `timestamp_ms = now_primary_ms()`. (Under mixed mode, ops inside a group run concurrently, so the group start is the only meaningful boundary; documented in the guidance.)

**`--role standby`** (on the standby host, connects to the standby for queries, and to the primary once for clock priming):
- Spawn `--threads` worker threads. Each loops until stop: pick a uniformly random query vector from the query set, run `SELECT id FROM <table> ORDER BY vec <-> '<lit>' LIMIT <k>` (with `SET hnsw.ef_search`/`ivfflat.probes` once per session), record completion `now_primary_ms()` into a per-thread buffer.
- Stop on SIGINT/SIGTERM or after `--duration-sec` (default 0 = until signaled).
- On stop: merge per-thread completion timestamps, bucket by `floor(ts/1000)*1000`, write `standby_throughput.csv` (`timestamp_ms,qps`), one row per second that had ≥1 completion.

### 4.4 `Makefile`

Targets `replica_recall_eval` and `replica_throughput_eval`, built like `pgvector_test.cpp`:
```
g++ -O3 -std=c++17 -mavx2 -mfma -fopenmp <src> -o <bin> \
    -I$(shell pg_config --includedir) -lpq -lyaml-cpp -lcrypto -pthread
```
`PG_INCLUDE` overridable; `make all` builds both.

## 5. CLI surface (summary)

Shared: `--dataset --queries --runbook --dataset-name --gt-dir --index-type {hnsw,ivfflat} --hnsw-m --hnsw-ef-construction --hnsw-ef-search --ivfflat-lists --ivfflat-probes --table-name --dataset-offset --threads --checkpoint-size`.

`replica_recall_eval` adds: `--primary-host/--primary-port/--primary-db/--primary-user`, `--standby-host/--standby-port/--standby-db/--standby-user`, `--start-step` (101), `--end-step` (400), `--build-index-before` (0=off), `--num-verify-queries` (10000), `--catchup-timeout-sec` (300), `--standby-settle-ms` (0), `--out experiment_a.csv`.

`replica_throughput_eval` adds: `--role {primary,standby}`, primary conn flags (both roles, for clock priming), standby conn flags (standby role), `--mixed-mode-start` (101), `--mixed-size` (3), `--duration-sec` (0), `--assume-synced-clocks`, `--out` (`step_boundaries.csv` / `standby_throughput.csv` by role).

## 6. Clock module (different-host timeline)

The two CSVs are written on two hosts. To put them on one timeline without
requiring NTP, every emitted timestamp is expressed in **primary-DB-clock ms**:

```
best_rtt = +inf; offset = 0
repeat K times (default 5):
    t0 = steady_local_ms()
    p  = SELECT (extract(epoch from clock_timestamp())*1000)::bigint  -- on PRIMARY
    t1 = steady_local_ms()
    rtt = t1 - t0
    if rtt < best_rtt: best_rtt = rtt; offset = p - (t0 + rtt/2)
now_primary_ms() := steady_local_ms() + offset
```

- `--role primary` primes against the primary it already drives (tiny RTT).
- `--role standby` opens one extra short-lived connection to the **primary** purely to prime the offset (it has the primary host:port for this), then closes it; all query work stays on the standby.
- `--assume-synced-clocks` skips priming and uses `CLOCK_REALTIME` directly (documented NTP assumption).

Re-priming is one-shot by default (runbook B is minutes-scale; steady-clock drift is negligible); a periodic re-prime is a possible follow-on if drift is observed.

## 7. Outputs

- `experiment_a.csv`: header `step,primary_recall,standby_recall`; one row per search step (101,104,…,400); recall 6-dp; missing-GT rows emit `NaN`.
- `standby_throughput.csv`: header `timestamp_ms,qps`; `timestamp_ms` = epoch-ms at the start of each 1-second bucket (primary-DB clock); `qps` = completed queries in that second.
- `step_boundaries.csv`: header `step,timestamp_ms`; one row per mixed group (primary-DB clock).

A zeroed/relative time axis for plotting is left to post-processing (the guidance shows a one-liner). No plotting code is shipped.

## 8. Guidance doc (`docs/replica_search_evaluation.md`) outline

1. Prereqs: dataset/query/runbook/GT paths; build (`make` in `test/runbooks/replication/`); PG 15+ with `shared_preload_libraries='vector'`.
2. Replication GUCs for primary and standby (from the design / `130_*.pl`): `wal_level=replica`, `max_wal_senders`, `pgvector.replication_role`, `pgvector.replication_primary_host/port`, `pgvector.replication_shared_secret`, `pgvector.storage_base_dir`, `pgvector.replication_fetch_parallelism`, `hot_standby=on`; initial sync = `pg_basebackup` + `rsync` of `storage_base_dir`.
3. Seeding for Exp A (seed-at-101 vs full-from-1) and index build.
4. Run Experiment A → `experiment_a.csv`; how to read it.
5. Run Experiment B: **launch order** — start `--role standby` first (begins querying), then `--role primary`; stop the standby driver (SIGINT) when the primary finishes; clock-sync note for different hosts.
6. CSV schemas + a short plotting snippet.
7. Troubleshooting: catchup timeouts, GT filename mismatch, fetcher lag.

## 9. Risks / open items

- **Recall delta from fetch lag (Exp A).** With catchup-only waiting, the standby may briefly serve from memtables while the primary serves from a flushed segment, surfacing a small recall delta that is a lag artifact, not a correctness bug. Documented; `--standby-settle-ms` available if a fairer comparison is wanted.
- **Cross-host clock accuracy (Exp B).** RTT/2 estimation bounds error to a few ms over a LAN — well under the 1-second QPS bucket. `--assume-synced-clocks` is the fallback.
- **Standby driver lifetime.** Cross-host start/stop is operator-coordinated (SIGINT or `--duration-sec`); no automatic primary-completion detection in v1.
- **`--dataset-offset` parity.** Insert/delete row ids must match the GT generator's convention; default 0, matching the existing GT set.
```
