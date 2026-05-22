# Phase 7 — Test 2: segment merge replicates.
#
# This test attempts to exercise the merge bgworker code path on the primary
# by inserting enough rows to (a) cause multiple memtable flushes and (b)
# give merge_adjacent_segments_pool a chance to fire. When it does, the
# primary emits a SegmentReplaced WAL record (custom rmgr) which the standby
# replays by enqueueing a pull of the new merged segment and superseding the
# old ones in its in-memory segment array.
#
# NOTE: This test does NOT guarantee that a merge actually happened on the
# primary. It exercises the same code paths as test 110 with a larger row
# count, increasing the likelihood that merge_adjacent_segments_pool runs
# and emits SegmentReplaced WAL. If merge doesn't fire (depends on workload
# thresholds), this test degrades into "test 110 with 100k rows".
# A future task could add a SQL helper (e.g. pgvector_force_merge('t_v_idx'))
# to force merge synchronously and make this test deterministic.
#
# Flakiness profile:
#   * Whether merge actually fires is workload-dependent. The correctness
#     check (row count) is unaffected by whether merge ran.
#   * Same drain-timeout caveats as test 110.
use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $dim     = 32;
my $n_build = 100;       # seed rows inserted BEFORE CREATE INDEX
my $n_rows  = 100000;    # total row count after the bulk insert
my $port    = 18111;

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

my $array_sql = join(",", ('random()') x $dim);

# ---- Phase 1: seed + index; verify standby matches primary on the seed ----

# CREATE INDEX on an empty table is unsupported by this fork. Seed first.
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
   "phase 1: standby top-$top_k matches primary top-$top_k");

# ---- Phase 2: large insert (multiple flushes + hopefully a merge) ----

$primary->safe_psql('postgres', qq(
    INSERT INTO t
    SELECT i, ARRAY[$array_sql]::vector
    FROM generate_series($n_build + 1, $n_rows) i;
));

# Wait long enough that flushes complete and the merge worker has a chance.
$primary->safe_psql('postgres', 'SELECT pg_sleep(5);');

wait_catchup($primary, $standby);
ok(wait_for_queue_drained($standby, 60, "phase 2"),
   "phase 2: standby fetch queue drained after flushes (+ optional merge)");

my @phase2_primary = topk_ids($primary);
my @phase2_standby = topk_ids($standby);
is(scalar(@phase2_primary), $top_k,
   "phase 2: primary top-$top_k returns $top_k rows");
is(scalar(@phase2_standby), $top_k,
   "phase 2: standby top-$top_k returns $top_k rows");
is_deeply(\@phase2_standby, \@phase2_primary,
   "phase 2: standby top-$top_k matches primary top-$top_k after flushes "
   . "(and merge if it fired)");

# Cross-phase sanity: post-seed rows must influence the top-k.
my %phase1_set = map { $_ => 1 } @phase1_primary;
my $changed    = grep { !$phase1_set{$_} } @phase2_primary;
ok($changed > 0,
   "phase 2: primary top-$top_k contains $changed ids not in phase 1 "
   . "(proves new rows are queryable)");

done_testing();
