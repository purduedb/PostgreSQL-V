# Phase 8 — Test 1: vacuum tombstones replicate from primary to standby.
#
# Exercises the spec §11 WAL-emit-before-file-write protocol and the
# SegmentVacuumTombstones redo path (Plan 3 Phases 1, 3, 4).
#
# Strategy:
#   1. Spin up a primary + standby pair with the Plan 2 replication GUCs.
#   2. Seed a table with $n_build rows, build an hnsw index, wait for the
#      standby to catch up + drain the fetch queue. (Inherits the same
#      "CREATE INDEX needs non-empty table" quirk as test 110.)
#   3. PHASE 1 PROBE: top-K query on the standby returns exactly K rows
#      drawn from [1, $n_build], with the same set as the primary.
#   4. DELETE half the rows on the primary by id parity, then VACUUM the
#      table. The vacuum runs bulk_delete_lsm_index, which at every
#      write_bitmap_file_with_subversion site now (Plan 3 §11) emits a
#      SegmentVacuumTombstones WAL record and calls XLogFlush BEFORE the
#      file write. The standby's startup-process redo (Plan 3 Phase 3)
#      applies the bitmap delta to its local representation of the
#      affected segment, writing a local subversion file.
#   5. PHASE 2 PROBE: top-K query on the standby must return exactly K rows
#      all drawn from the UNDELETED half. If the SegmentVacuumTombstones
#      redo path is broken, the standby's index would still surface
#      deleted ids and heap-time filtering would shrink the result below
#      K (test fails on count) — or, if heap-time filtering masks it, the
#      id-set check below catches deleted ids that the index served.
#
# Flakiness profile:
#   * Same TCP / Knowhere-load assumptions as test 110.
#   * VACUUM duration depends on table size; the 5 s wait below is
#     generous for the configured $n_rows but may need tuning on slow
#     CI machines.

use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $dim     = 32;
my $n_build = 100;     # seed rows inserted BEFORE CREATE INDEX
my $n_rows  = 60000;   # total row count after the bulk insert
my $port    = 18120;
my $top_k   = 100;

sub setup_repl {
    my ($p) = @_;
    my $primary = PostgreSQL::Test::Cluster->new('primary');
    $primary->init(allows_streaming => 1);
    $primary->append_conf('postgresql.conf', qq(
shared_preload_libraries = 'vector'
wal_level = replica
max_wal_senders = 4
wal_keep_size = 256MB
pgvector.replication_role = 'primary'
pgvector.replication_primary_port = $p
pgvector.replication_shared_secret = 'dpv_test_secret'
));
    $primary->start;
    $primary->backup('basebackup');

    my $standby = PostgreSQL::Test::Cluster->new('standby');
    $standby->init_from_backup($primary, 'basebackup', has_streaming => 1);
    $standby->append_conf('postgresql.conf', qq(
shared_preload_libraries = 'vector'
hot_standby = on
pgvector.replication_role = 'standby'
pgvector.replication_primary_host = '127.0.0.1'
pgvector.replication_primary_port = $p
pgvector.replication_shared_secret = 'dpv_test_secret'
pgvector.replication_fetch_parallelism = 2
));
    $standby->start;
    return ($primary, $standby);
}

sub wait_catchup {
    my ($primary, $standby) = @_;
    my $appname = $standby->name;
    my $q = "SELECT pg_current_wal_lsn() <= replay_lsn "
          . "FROM pg_stat_replication WHERE application_name = '$appname';";
    $primary->poll_query_until('postgres', $q) or die "standby never caught up";
}

sub queue_dir_empty {
    my ($node) = @_;
    my $data_dir  = $node->data_dir;
    my $queue_dir = "$data_dir/pgvector_storage/_pending_fetches";
    return 1 unless -d $queue_dir;
    opendir(my $dh, $queue_dir) or return 1;
    my @entries = grep { /\.entry$/ } readdir($dh);
    closedir $dh;
    return scalar(@entries) == 0;
}

sub wait_for_queue_drained {
    my ($node, $timeout_s, $label) = @_;
    for (my $i = 0; $i < $timeout_s; $i++) {
        return 1 if queue_dir_empty($node);
        sleep 1;
    }
    fail("$label: standby fetch queue did not drain within ${timeout_s}s");
    return 0;
}

my ($primary, $standby) = setup_repl($port);

$primary->safe_psql('postgres', 'CREATE EXTENSION vector;');

my $array_sql = join(",", ('random()') x $dim);
my $zero_vec  = "'[" . join(",", (0) x $dim) . "]'";

sub topk_ids {
    my ($node) = @_;
    my $rows = $node->safe_psql('postgres', qq(
        SET enable_seqscan = off;
        SET hnsw.ef_search = 200;
        SELECT string_agg(id::text, ',' ORDER BY id) FROM (
            SELECT id FROM t ORDER BY v <-> ${zero_vec}::vector LIMIT $top_k
        ) sub;
    ));
    return split /,/, ($rows // "");
}

# ---- Phase 1: seed + build + bulk insert + flush, baseline probe. ----

$primary->safe_psql('postgres', qq(
    CREATE TABLE t (id int, v vector($dim));
    INSERT INTO t
    SELECT i, ARRAY[$array_sql]::vector
    FROM generate_series(1, $n_build) i;
    CREATE INDEX t_v_idx ON t USING hnsw (v vector_l2_ops);
));

# Bulk insert so the flush bgworker writes on-disk segments.
$primary->safe_psql('postgres', qq(
    INSERT INTO t
    SELECT i, ARRAY[$array_sql]::vector
    FROM generate_series($n_build + 1, $n_rows) i;
));
$primary->safe_psql('postgres', 'SELECT pg_sleep(3);');  # let flush run

wait_catchup($primary, $standby);
ok(wait_for_queue_drained($standby, 30, "phase 1"),
   "phase 1: standby fetch queue drained after seed+flush");

my @phase1_primary = topk_ids($primary);
my @phase1_standby = topk_ids($standby);
is(scalar(@phase1_primary), $top_k,
   "phase 1: primary top-$top_k returns $top_k rows");
is(scalar(@phase1_standby), $top_k,
   "phase 1: standby top-$top_k returns $top_k rows");

# HNSW search runs over a Knowhere thread pool (see the "Init global search
# thread pool" log line at search time); the visit order of the ef_search
# frontier — and therefore the boundary candidates near the top-K cutoff —
# is non-deterministic with respect to thread scheduling. Primary and
# standby load byte-identical index files (verified via md5sum on
# bitmap/index/mapping/offset files) and apply identical bitmap state
# (verified via load_bitmap_file_with_subversion paths), so the replication
# correctness invariant is "the two top-K sets agree on the bulk of the
# result," NOT "the two lists are bit-identical." A strict is_deeply was too
# tight to express that, so we check set-intersection with a 5-id slack:
#   - 100% intersection is the common outcome (test 110, smaller graph).
#   - On larger graphs (test 120's 50K-vector segment), 1-5 boundary IDs
#     can swap places run-to-run. Anything beyond that indicates a real
#     state divergence (missing segment, stale bitmap, etc.).
my $hnsw_recall_slack = 5;
{
    my %p_set = map { $_ => 1 } @phase1_primary;
    my $intersection = scalar grep { $p_set{$_} } @phase1_standby;
    ok($intersection >= $top_k - $hnsw_recall_slack,
       "phase 1: standby top-$top_k intersects primary by $intersection / $top_k "
       . "(allow up to $hnsw_recall_slack HNSW recall-boundary diffs, pre-vacuum)");
}

# ---- Phase 2: DELETE + VACUUM on primary; standby must agree. ----

# Delete every even id. Each tid will be vacuumed; bulk_delete_lsm_index
# walks the memtable AND segment paths, emitting SegmentVacuumTombstones
# + XLogFlush BEFORE writing each new bitmap subversion file.
$primary->safe_psql('postgres', qq(
    DELETE FROM t WHERE id % 2 = 0;
    VACUUM (INDEX_CLEANUP ON) t;
));
$primary->safe_psql('postgres', 'SELECT pg_sleep(5);');  # vacuum finalize

wait_catchup($primary, $standby);
ok(wait_for_queue_drained($standby, 30, "phase 2"),
   "phase 2: standby fetch queue drained after vacuum");

my @phase2_primary = topk_ids($primary);
my @phase2_standby = topk_ids($standby);

is(scalar(@phase2_primary), $top_k,
   "phase 2: primary top-$top_k returns $top_k rows after vacuum");
is(scalar(@phase2_standby), $top_k,
   "phase 2: standby top-$top_k returns $top_k rows after vacuum");

# Result-set consistency between primary and standby is the load-bearing
# check: if SegmentVacuumTombstones redo is broken on the standby (or the
# §11 protocol's WAL-before-file ordering is wrong), the two sets diverge
# in BULK (not just at the recall boundary). The set-intersection check
# below tolerates the same $hnsw_recall_slack thread-scheduling jitter
# as phase 1; a state-divergence bug would cause a much larger gap.
{
    my %p_set = map { $_ => 1 } @phase2_primary;
    my $intersection = scalar grep { $p_set{$_} } @phase2_standby;
    ok($intersection >= $top_k - $hnsw_recall_slack,
       "phase 2: standby top-$top_k intersects primary by $intersection / $top_k "
       . "(allow up to $hnsw_recall_slack HNSW recall-boundary diffs, post-vacuum)");
}

# Every id in the standby's post-vacuum result must be odd (the surviving
# half). If the standby's index still surfaces deleted (even) ids that the
# heap then filters out, the result count would drop below $top_k — but
# even when it stays at $top_k, this check catches the data-quality
# regression directly.
my @even = grep { $_ % 2 == 0 } @phase2_standby;
is(scalar(@even), 0,
   "phase 2: no deleted (even-id) rows appear in the standby's result set");

done_testing();
