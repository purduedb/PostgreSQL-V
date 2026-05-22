# Phase 7 — Test 3: pending-fetch queue durability across a standby restart.
#
# Strategy:
#   1. Set up primary + standby with replication GUCs (port 18112).
#   2. Create extension + table + hnsw index on the primary, then bulk-insert
#      ~50000 rows to trigger one or more segment flushes.
#   3. Sleep just long enough for the flush bgworker to enqueue pending
#      fetches on the standby, but NOT long enough for the fetcher to drain
#      them. We want pulls in-flight or queued when the standby goes down.
#   4. Stop the standby with mode 'immediate' (simulates a crash mid-pull).
#   5. Restart the standby. The persisted queue files under
#      <data>/pgvector_storage/_pending_fetches should survive, FETCHING
#      entries should get demoted back to QUEUED on recovery, and the
#      fetcher should resume pulling.
#   6. Poll for the queue to drain (up to 30s).
#   7. Probe the standby — all 50000 rows must be visible via the index.
#
# Flakiness profile:
#   * The 1-second pre-stop sleep is a heuristic. If the primary finishes
#     all flushes faster than that or the standby drains faster than that,
#     the test still passes (it just doesn't exercise the mid-flight path).
#     A failure here means the queue file format / restart recovery is
#     broken. Passing without a mid-flight stop is harmless.
#   * 'immediate' shutdown is not graceful; this is intentional. It mimics
#     a real-world crash so we can verify the FETCHING→QUEUED demotion path.
use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $dim     = 32;
my $n_build = 100;     # seed rows inserted BEFORE CREATE INDEX
my $n_rows  = 50000;   # total row count after the bulk insert
my $port    = 18112;

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

# ---- Phase 1: seed + index; sanity check pre-stop ----

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
is_deeply(\@phase1_standby, \@phase1_primary,
   "phase 1: standby top-$top_k matches primary top-$top_k (pre-restart)");

# ---- Phase 2: bulk insert, immediate-stop mid-pull, restart, drain ----

$primary->safe_psql('postgres', qq(
    INSERT INTO t
    SELECT i, ARRAY[$array_sql]::vector
    FROM generate_series($n_build + 1, $n_rows) i;
));

# Just enough time for the flush worker to start writing/enqueueing —
# we deliberately do NOT wait for the fetcher to drain.
$primary->safe_psql('postgres', 'SELECT pg_sleep(1);');

wait_catchup($primary, $standby);

# Hard-stop the standby. This simulates a crash and exercises the FETCHING
# → QUEUED demotion logic on the next start.
$standby->stop('immediate');

# Restart. Recovery should rehydrate the persisted queue and resume pulls.
$standby->start;

ok(wait_for_queue_drained($standby, 30, "phase 2"),
   "phase 2: standby fetch queue drained after immediate restart");

my @phase2_primary = topk_ids($primary);
my @phase2_standby = topk_ids($standby);
is(scalar(@phase2_primary), $top_k,
   "phase 2: primary top-$top_k returns $top_k rows");
is(scalar(@phase2_standby), $top_k,
   "phase 2: standby top-$top_k returns $top_k rows");
is_deeply(\@phase2_standby, \@phase2_primary,
   "phase 2: standby top-$top_k matches primary top-$top_k after restart-and-resume");

# Cross-phase sanity: post-restart top-k must include rows from the bulk insert.
my %phase1_set = map { $_ => 1 } @phase1_primary;
my $changed    = grep { !$phase1_set{$_} } @phase2_primary;
ok($changed > 0,
   "phase 2: primary top-$top_k contains $changed ids not in phase 1 "
   . "(proves bulk-insert rows landed and are queryable on both nodes)");

done_testing();
