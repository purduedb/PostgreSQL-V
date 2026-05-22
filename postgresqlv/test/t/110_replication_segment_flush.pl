# Phase 7 — Test 1: segment flush replicates and is adopted by standby.
#
# Strategy:
#   1. Spin up a primary + standby pair configured with the Plan 2 GUCs
#      (pgvector.replication_role, primary port, shared secret).
#   2. Create the vector extension and a table on the primary, seed it with
#      $n_build rows, then create the hnsw index (CREATE INDEX on an empty
#      table is unsupported in this fork). Initial HNSW build emits a
#      SegmentCreated WAL record covering the seed segment.
#   3. PHASE 1 PROBE: wait for catchup + fetch queue drain, then run a top-100
#      kNN query on the standby. Result must contain exactly 100 ids drawn
#      from the seed range [1, $n_build].
#   4. Bulk insert ~50000 - $n_build additional rows so the primary's memtable
#      fills and the LSM flush bgworker writes one or more on-disk segments.
#      Each flush emits a SegmentCreated WAL record and enqueues a pending
#      fetch on the standby.
#   5. PHASE 2 PROBE: wait for catchup + fetch queue drain, then run another
#      top-100 kNN query. Result must contain exactly 100 ids. We also check
#      that at least one returned id is from the post-seed range, proving the
#      flushed segments are actually being searched on the standby.
#
# Flakiness profile:
#   * The "drain in 30s" timeout depends on TCP throughput from primary to
#     standby and Knowhere load time per segment. On slow disks or under
#     parallel test load this may legitimately need more time.
#   * The flush bgworker fires when the memtable exceeds a size threshold;
#     ~50000 rows of dim=32 should be well over that, but a future tweak to
#     the threshold could invalidate this assumption.
#   * top-100 with a fixed query vector (origin) is well-defined for HNSW
#     even with low ef_search; the test does not rely on exact recall, only
#     on the index returning *some* 100 valid ids.
use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $dim     = 32;
my $n_build = 100;     # seed rows inserted BEFORE CREATE INDEX
my $n_rows  = 50000;   # total row count after the bulk insert
my $port    = 18110;

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

my ($primary, $standby) = setup_repl($port);

$primary->safe_psql('postgres', 'CREATE EXTENSION vector;');

my $array_sql = join(",", ('random()') x $dim);

sub wait_for_queue_drained {
    my ($node, $timeout_s, $label) = @_;
    for (my $i = 0; $i < $timeout_s; $i++) {
        return 1 if queue_dir_empty($node);
        sleep 1;
    }
    fail("$label: standby fetch queue did not drain within ${timeout_s}s");
    return 0;
}

my $zero_vec = "'[" . join(",", (0) x $dim) . "]'";
my $top_k    = 100;

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

# ---- Phase 1: seed table, build index, verify standby can query the seed ----

# CREATE INDEX on an empty table is unsupported in this fork; seed first.
$primary->safe_psql('postgres', qq(
    CREATE TABLE t (id int, v vector($dim));
    INSERT INTO t
    SELECT i, ARRAY[$array_sql]::vector
    FROM generate_series(1, $n_build) i;
    CREATE INDEX t_v_idx ON t USING hnsw (v vector_l2_ops);
));

wait_catchup($primary, $standby);
ok(wait_for_queue_drained($standby, 30, "phase 1"),
   "phase 1: standby fetch queue drained after initial index build");

my @phase1_primary = topk_ids($primary);
my @phase1_standby = topk_ids($standby);
is(scalar(@phase1_primary), $top_k,
   "phase 1: primary top-$top_k returns $top_k rows");
is(scalar(@phase1_standby), $top_k,
   "phase 1: standby top-$top_k returns $top_k rows");
is_deeply(\@phase1_standby, \@phase1_primary,
   "phase 1: standby top-$top_k matches primary top-$top_k (sorted id set)");

# ---- Phase 2: bulk insert, verify standby search reaches new rows ----

$primary->safe_psql('postgres', qq(
    INSERT INTO t
    SELECT i, ARRAY[$array_sql]::vector
    FROM generate_series($n_build + 1, $n_rows) i;
));

# Let the flush bgworker write the segment(s) and emit SegmentCreated WAL.
$primary->safe_psql('postgres', 'SELECT pg_sleep(2);');

wait_catchup($primary, $standby);
ok(wait_for_queue_drained($standby, 30, "phase 2"),
   "phase 2: standby fetch queue drained after bulk insert + flush");

my @phase2_primary = topk_ids($primary);
my @phase2_standby = topk_ids($standby);
is(scalar(@phase2_primary), $top_k,
   "phase 2: primary top-$top_k returns $top_k rows");
is(scalar(@phase2_standby), $top_k,
   "phase 2: standby top-$top_k returns $top_k rows");
is_deeply(\@phase2_standby, \@phase2_primary,
   "phase 2: standby top-$top_k matches primary top-$top_k (sorted id set)");

# Cross-phase sanity: with ~500x more rows now visible, the top-100 should
# differ from phase 1. If it doesn't, the flushed segments were never
# searched and adoption silently failed.
my %phase1_set = map { $_ => 1 } @phase1_primary;
my $changed    = grep { !$phase1_set{$_} } @phase2_primary;
ok($changed > 0,
   "phase 2: primary top-$top_k contains $changed ids not in phase 1 result "
   . "(proves the new rows are actually queryable)");

done_testing();
