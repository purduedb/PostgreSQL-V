#!/usr/bin/env python3
"""
vector_restart_client.py
Benchmark client for vector-index restart / cold-start latency experiments.

Schedules queries at a fixed QPS per client regardless of how long each takes.
Records one CSV row per query so you can plot latency vs. time and add vertical
markers for crash, restart, mmap-loaded, and fully-loaded events.

CSV columns
-----------
client_id            integer: which client sent this query
query_id             integer: monotonically increasing per client
scheduled_ts         float: seconds since experiment start when query was due
start_ts             float: seconds since experiment start when query actually started
end_ts               float: seconds since experiment start when query ended
latency_ms           float: end_ts - start_ts  (milliseconds)
corrected_latency_ms float: end_ts - scheduled_ts  (milliseconds; captures schedule slip)
status               str:   ok | timeout | sql_error | connection_error
error                str:   short error description on failure, empty on success

Examples
--------
# Single client, 20 QPS, 3-minute run
python3.12 vector_restart_client.py \\
  --host localhost --port 5434 --user liu4127 --dbname postgres \\
  --query-vectors /ssd_root/dataset/turing10m/msturing-query.fvecs \\
  --clients 1 --total-qps 20 --duration 180 --timeout-ms 50 \\
  --k 100 --table vectors \\
  --output /ssd_root/liu4127/output_1c.csv \\
  --summary-output /ssd_root/liu4127/summary_1c.csv

# 16 clients, 160 QPS total
python3.12 vector_restart_client.py \\
  --host localhost --port 5434 --user liu4127 --dbname postgres \\
  --query-vectors /ssd_root/dataset/turing10m/msturing-query.fvecs \\
  --clients 16 --total-qps 160 --duration 180 --timeout-ms 50 \\
  --k 100 --table vectors \\
  --output /ssd_root/liu4127/output_16c.csv \\
  --summary-output /ssd_root/liu4127/summary_16c.csv
"""

import argparse
import asyncio
import csv
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import asyncpg
import numpy as np


# ── Vector file loaders ───────────────────────────────────────────────────────

def _read_fvecs(path: str, limit) -> np.ndarray:
    vecs = []
    with open(path, "rb") as f:
        while limit is None or len(vecs) < limit:
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            (dim,) = struct.unpack_from("<i", hdr)
            buf = f.read(4 * dim)
            if len(buf) < 4 * dim:
                break
            vecs.append(np.frombuffer(buf, dtype="<f4").copy())
    return np.stack(vecs)


def _read_bvecs(path: str, limit) -> np.ndarray:
    vecs = []
    with open(path, "rb") as f:
        while limit is None or len(vecs) < limit:
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            (dim,) = struct.unpack_from("<i", hdr)
            buf = f.read(dim)
            if len(buf) < dim:
                break
            vecs.append(np.frombuffer(buf, dtype=np.uint8).astype(np.float32))
    return np.stack(vecs)


def _read_fbin(path: str, limit) -> np.ndarray:
    with open(path, "rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
        count = n if limit is None else min(n, limit)
        data = np.frombuffer(f.read(count * d * 4), dtype="<f4").copy()
    return data.reshape(count, d)


def load_vectors(path: str, limit=None) -> np.ndarray:
    ext = Path(path).suffix.lower()
    if ext == ".fvecs":
        return _read_fvecs(path, limit)
    if ext == ".bvecs":
        return _read_bvecs(path, limit)
    if ext == ".fbin":
        return _read_fbin(path, limit)
    raise ValueError(f"Unsupported vector file format: {ext!r}. Use .fvecs / .bvecs / .fbin")


# ── Connection helpers ────────────────────────────────────────────────────────

async def _open_conn(args) -> asyncpg.Connection:
    conn = await asyncpg.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password or None,
        database=args.dbname,
        # No global command_timeout; we impose per-query timeouts manually.
        command_timeout=None,
    )
    await conn.execute(f"SET hnsw.ef_search = {args.ef_search}")
    await conn.execute(f"SET ivfflat.probes  = {args.nprobe}")
    return conn


async def connect_with_retry(args, *, max_wait: float = 60.0) -> asyncpg.Connection:
    """Retry every 0.5 s until connected or max_wait seconds elapsed."""
    deadline = time.monotonic() + max_wait
    last_exc: Exception | None = None
    while True:
        try:
            return await _open_conn(args)
        except Exception as e:
            last_exc = e
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"Could not connect to {args.host}:{args.port} after {max_wait:.0f}s"
            ) from last_exc
        await asyncio.sleep(min(0.5, remaining))


# ── Client worker ─────────────────────────────────────────────────────────────

# Exceptions that indicate the connection to PostgreSQL was lost.
_CONN_ERRORS = (
    asyncpg.ConnectionDoesNotExistError,
    asyncpg.InterfaceError,
    OSError,
)

_SQL_TEMPLATE = "SELECT id FROM {table} ORDER BY vec <-> $1::vector LIMIT $2"


def _vec_literal(vec: np.ndarray) -> str:
    """Format a numpy vector as a pgvector literal string '[x,y,…]'."""
    return "[" + ",".join(f"{v:.6g}" for v in vec) + "]"


async def client_worker(
    cid: int,
    args,
    qvecs: np.ndarray,
    out_q: asyncio.Queue,
    t0: float,           # loop.time() at experiment start
    stop_ev: asyncio.Event,
) -> None:
    """
    Schedules queries at qps_per_client.

    Schedule is open-loop: next_sched advances by `interval` after each query
    regardless of how long the query took.  After a slow query next_sched will
    be in the past, so the next query fires immediately — no artificial idle time.

    After a server crash (connection_error), next_sched is reset to now so we
    don't burst through the accumulated backlog when the server comes back.
    """
    loop = asyncio.get_running_loop()
    qps = args.qps_per_client
    interval = 1.0 / qps

    # Effective per-query timeout per spec:
    #   if timeout_s * qps < 1 → use timeout_s
    #   otherwise             → cap at 1/qps  (one interval)
    raw_to = args.timeout_ms / 1000.0
    eff_to = raw_to if raw_to * qps < 1.0 else interval

    sql = _SQL_TEMPLATE.format(table=args.table)
    n_vecs = len(qvecs)

    conn = await connect_with_retry(args)

    # Stagger clients so their first queries are spread evenly within one interval.
    stagger = cid * interval / max(args.clients, 1)
    next_sched = t0 + stagger   # absolute loop time for next scheduled query

    qid = 0
    while not stop_ev.is_set():
        if loop.time() - t0 >= args.duration:
            break

        # Sleep until the next scheduled slot (skip if already past).
        delay = next_sched - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)

        if stop_ev.is_set():
            break

        sched_abs = next_sched          # absolute scheduled start
        actual_start = loop.time()
        scheduled_ts = sched_abs - t0
        start_ts = actual_start - t0
        next_sched += interval          # advance schedule unconditionally

        vec_lit = _vec_literal(qvecs[qid % n_vecs])
        status = "ok"
        error = ""
        end_abs = actual_start          # updated on every path below
        reconnected = False

        try:
            # Enforce timeout via asyncio.wait_for so it fires even if asyncpg's
            # internal state is confused after repeated server-side cancels.
            await asyncio.wait_for(conn.fetch(sql, vec_lit, args.k), timeout=eff_to)
            end_abs = loop.time()

        except (asyncio.TimeoutError, asyncpg.QueryCanceledError):
            end_abs = loop.time()
            status = "timeout"
            error = f">{eff_to * 1000:.0f}ms"
            # Always close and reconnect after a timeout.  asyncpg sends a
            # server-side cancel; if we reuse the same connection the next
            # query can inherit stale cancel state from the PostgreSQL backend,
            # causing it to be cancelled immediately without a 50ms wait.
            try:
                await asyncio.wait_for(conn.close(), timeout=1.0)
            except Exception:
                pass
            conn = await connect_with_retry(args)
            reconnected = True

        except _CONN_ERRORS as exc:
            end_abs = loop.time()
            status = "connection_error"
            error = repr(exc)[:120]
            try:
                await conn.close()
            except Exception:
                pass
            conn = await connect_with_retry(args)
            reconnected = True

        except asyncpg.PostgresError as exc:
            end_abs = loop.time()
            status = "sql_error"
            error = repr(exc)[:120]
            if conn.is_closed():
                try:
                    await conn.close()
                except Exception:
                    pass
                conn = await connect_with_retry(args)
                reconnected = True

        except Exception as exc:
            end_abs = loop.time()
            status = "connection_error"
            error = repr(exc)[:120]
            try:
                await conn.close()
            except Exception:
                pass
            conn = await connect_with_retry(args)
            reconnected = True

        # After reconnecting from a crash the schedule may be far in the past.
        # Reset to now so we don't burst through the accumulated backlog.
        if reconnected:
            next_sched = loop.time()

        end_ts = end_abs - t0
        lat_ms = (end_abs - actual_start) * 1e3
        clat_ms = (end_abs - sched_abs) * 1e3

        await out_q.put((
            cid, qid,
            f"{scheduled_ts:.6f}", f"{start_ts:.6f}", f"{end_ts:.6f}",
            f"{lat_ms:.3f}", f"{clat_ms:.3f}",
            status, error,
        ))
        qid += 1

    try:
        await conn.close()
    except Exception:
        pass


# ── CSV writer task ───────────────────────────────────────────────────────────

_CSV_HEADER = (
    "client_id", "query_id",
    "scheduled_ts", "start_ts", "end_ts",
    "latency_ms", "corrected_latency_ms",
    "status", "error",
)


async def csv_writer_task(out_q: asyncio.Queue, path: str, flush_s: float = 2.0) -> int:
    """
    Drains out_q and writes rows to CSV.  Flushes to disk every flush_s seconds.
    Terminates when it receives None (sentinel) from the queue.
    Returns the number of rows written.
    """
    written = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_CSV_HEADER)
        last_flush = time.monotonic()

        while True:
            try:
                row = await asyncio.wait_for(out_q.get(), timeout=flush_s)
            except asyncio.TimeoutError:
                f.flush()
                last_flush = time.monotonic()
                continue

            if row is None:     # sentinel: flush and exit
                f.flush()
                break

            w.writerow(row)
            written += 1
            out_q.task_done()

            if time.monotonic() - last_flush >= flush_s:
                f.flush()
                last_flush = time.monotonic()

    return written


# ── Summary ───────────────────────────────────────────────────────────────────

def write_summary(summary_path: str, data_path: str, args, elapsed_s: float) -> None:
    """Re-reads the per-query CSV and writes one-row summary (easy to stack across runs)."""
    status_cnt: Counter = Counter()
    lats_ok: list[float] = []

    with open(data_path, newline="") as f:
        for row in csv.DictReader(f):
            status_cnt[row["status"]] += 1
            if row["status"] == "ok":
                lats_ok.append(float(row["latency_ms"]))

    total = sum(status_cnt.values())
    lats = np.array(lats_ok) if lats_ok else np.array([float("nan")])

    def pct(p):
        return f"{np.nanpercentile(lats, p):.3f}" if lats_ok else "nan"

    header = [
        "clients", "total_qps", "duration_s", "timeout_ms", "k", "ef_search", "nprobe",
        "total_queries", "ok", "timeout", "sql_error", "connection_error",
        "achieved_qps", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "p999_ms",
    ]
    row = [
        args.clients, args.total_qps, f"{elapsed_s:.2f}", args.timeout_ms,
        args.k, args.ef_search, args.nprobe,
        total,
        status_cnt.get("ok", 0),
        status_cnt.get("timeout", 0),
        status_cnt.get("sql_error", 0),
        status_cnt.get("connection_error", 0),
        f"{total / elapsed_s:.2f}",
        f"{np.nanmean(lats):.3f}" if lats_ok else "nan",
        pct(50), pct(95), pct(99), pct(99.9),
    ]

    # Append if file already exists (to stack multiple runs), write header only once.
    write_header = not Path(summary_path).exists()
    with open(summary_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def run(args, qvecs: np.ndarray) -> None:
    out_q: asyncio.Queue = asyncio.Queue(maxsize=200_000)
    stop_ev = asyncio.Event()

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    wall0 = time.monotonic()

    writer = asyncio.create_task(csv_writer_task(out_q, args.output))

    workers = [
        asyncio.create_task(client_worker(cid, args, qvecs, out_q, t0, stop_ev))
        for cid in range(args.clients)
    ]

    worker_results = await asyncio.gather(*workers, return_exceptions=True)
    elapsed = time.monotonic() - wall0

    for i, r in enumerate(worker_results):
        if isinstance(r, Exception):
            print(f"[client {i}] exited with exception: {r!r}", file=sys.stderr)

    # Signal the CSV writer to flush and exit.
    await out_q.put(None)
    written = await writer

    print(
        f"{written} queries in {elapsed:.1f}s ({written / elapsed:.1f} QPS achieved)"
        f"  →  {args.output}"
    )

    if args.summary_output:
        write_summary(args.summary_output, args.output, args, elapsed)
        print(f"Summary → {args.summary_output}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Vector-index restart benchmark client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    c = p.add_argument_group("PostgreSQL connection")
    c.add_argument("--host", default="localhost")
    c.add_argument("--port", type=int, default=5432)
    c.add_argument("--user", default="postgres")
    c.add_argument("--password", default=None)
    c.add_argument("--dbname", default="postgres")

    w = p.add_argument_group("Workload")
    w.add_argument(
        "--query-vectors", required=True, metavar="PATH",
        help="Query vector file (.fvecs / .bvecs / .fbin)",
    )
    w.add_argument(
        "--max-query-vectors", type=int, default=None, metavar="N",
        help="Load at most N query vectors (default: all)",
    )
    w.add_argument("--clients", type=int, default=1,
                   help="Number of parallel client coroutines (default: 1)")
    qps = w.add_mutually_exclusive_group(required=True)
    qps.add_argument("--total-qps", type=float, metavar="QPS",
                     help="Total QPS shared across all clients")
    qps.add_argument("--per-client-qps", type=float, metavar="QPS",
                     help="QPS per client")
    w.add_argument("--duration", type=float, default=120.0,
                   help="Experiment duration in seconds (default: 120)")
    w.add_argument(
        "--timeout-ms", type=float, default=1000.0, metavar="MS",
        help=(
            "Per-query client-side timeout in ms (default: 1000). "
            "Capped at 1000/qps_per_client if that is smaller."
        ),
    )

    q = p.add_argument_group("Query parameters")
    q.add_argument("--k", type=int, default=10, help="Top-k neighbors (default: 10)")
    q.add_argument("--table", default="vectors", help="Table name (default: vectors)")
    q.add_argument("--ef-search", type=int, default=100, dest="ef_search",
                   help="hnsw.ef_search session parameter (default: 100)")
    q.add_argument("--nprobe", type=int, default=10,
                   help="ivfflat.probes session parameter (default: 10)")

    o = p.add_argument_group("Output")
    o.add_argument("--output", required=True, metavar="CSV",
                   help="Per-query CSV output path")
    o.add_argument("--summary-output", default=None, dest="summary_output", metavar="CSV",
                   help="Aggregate summary CSV path (optional)")

    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()

    if args.clients < 1:
        p.error("--clients must be at least 1")

    # Resolve QPS
    if args.total_qps is not None:
        args.qps_per_client = args.total_qps / args.clients
    else:
        args.qps_per_client = args.per_client_qps
        args.total_qps = args.per_client_qps * args.clients

    if args.qps_per_client <= 0:
        p.error("Resolved qps_per_client must be positive")

    # Load query vectors
    print(f"Loading {args.query_vectors} …", flush=True)
    qvecs = load_vectors(args.query_vectors, args.max_query_vectors)
    if len(qvecs) == 0:
        p.error("No query vectors loaded — check --query-vectors path and format")
    print(f"  {len(qvecs)} vectors, dim={qvecs.shape[1]}", flush=True)

    # Compute and display effective timeout
    raw_to = args.timeout_ms / 1000.0
    eff_to = raw_to if raw_to * args.qps_per_client < 1.0 else 1.0 / args.qps_per_client
    print(
        f"clients={args.clients}  total-QPS={args.total_qps:.1f}"
        f"  per-client-QPS={args.qps_per_client:.2f}"
        f"  effective-timeout={eff_to * 1000:.1f}ms"
        f"  duration={args.duration:.0f}s",
        flush=True,
    )
    print(f"Output → {args.output}", flush=True)

    try:
        asyncio.run(run(args, qvecs))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
