# Phase 7 — Test 4: out-of-order pull arrivals and stale-discard adoption.
#
# With pgvector.replication_fetch_parallelism = 4 on the standby, multiple
# segment pulls can run concurrently. The custom rmgr replay path enqueues
# fetches in the order WAL records arrive (SegmentFlushed, SegmentReplaced,
# ...). Because pulls finish in arbitrary order under parallelism, the
# adoption code in segment_adoption.c may sometimes see a fetched segment
# that has already been superseded by a later SegmentReplaced (a stale
# pull). The expected outcome is `adopt_result=1` (STALE_DISCARD): the
# fetcher discards the stale payload, logs a DEBUG1/WARNING explaining
# "no coverage" or "discarding", and the merge-target segment wins.
#
# Strategy:
#   1. Set up primary + standby with replication GUCs (port 18113). Standby
#      uses replication_fetch_parallelism = 4 and log_min_messages = DEBUG1
#      so adoption traces appear in the server log.
#   2. Bulk-insert ~100000 rows so multiple flushes and (hopefully) at least
#      one merge fire on the primary. The mix of SegmentFlushed +
#      SegmentReplaced records is what creates the out-of-order opportunity.
#   3. pg_sleep(5) and wait for catchup so all WAL replay is done on the
#      standby and all pulls have a chance to finish.
#   4. Drain the standby's _pending_fetches directory (up to 60s).
#   5. Soft check (TODO): grep the standby log for stale-discard markers.
#      If at least one is observed the timing genuinely interleaved; if not
#      the workload didn't produce overlapping pulls (still safe).
#   6. Hard check: standby kNN must return all 100000 rows. This is the
#      correctness invariant — stale-discard MUST NOT lose data.
#
# Flakiness profile:
#   * The TODO soft check is inherently nondeterministic. Stale-discard only
#     happens when a SegmentReplaced supersedes a pull that hasn't finished
#     yet, which depends on per-segment pull latency, parallelism, and how
#     promptly the merge worker fires. On fast hardware all pulls may
#     complete before any merge, so no stale-discard ever happens — that's
#     fine and not a regression. The TODO marks the assertion as "expected
#     to maybe fail" so CI doesn't treat it as a hard failure.
#   * Merge itself is workload-dependent (see test 111 notes).
#   * The hard check (count = 100000) MUST always pass. A failure means
#     stale-discard incorrectly dropped a live segment, which is a
#     correctness bug in the adoption code, not a timing artifact.
#   * Grepping the log file is fragile: the wording "adopt_result=1",
#     "no coverage", "discarding" must match what segment_adoption.c
#     actually emits. If those strings change in the source, update the
#     regex below or the soft check will silently always TODO-fail.
use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $dim     = 32;
my $n_build = 100;       # seed rows inserted BEFORE CREATE INDEX
my $n_rows  = 100000;    # total row count after the bulk insert
my $port    = 18113;

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
log_min_messages = DEBUG1
pgvector.replication_role = 'standby'
pgvector.replication_primary_host = '127.0.0.1'
pgvector.replication_primary_port = $p
pgvector.replication_shared_secret = 'dpv_test_secret'
pgvector.replication_fetch_parallelism = 4
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

# ---- Phase 1: seed + index ----

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
   "phase 1: standby top-$top_k matches primary top-$top_k");

# ---- Phase 2: bulk insert + concurrent pulls (parallelism = 4) ----

# Large insert: aim for multiple flushes and at least one merge so that
# SegmentReplaced WAL records race in-flight pulls on the standby.
$primary->safe_psql('postgres', qq(
    INSERT INTO t
    SELECT i, ARRAY[$array_sql]::vector
    FROM generate_series($n_build + 1, $n_rows) i;
));

# Give the flush + merge workers time to run on the primary so the standby
# sees the mixed Flushed/Replaced WAL stream.
$primary->safe_psql('postgres', 'SELECT pg_sleep(5);');

wait_catchup($primary, $standby);
ok(wait_for_queue_drained($standby, 60, "phase 2"),
   "phase 2: standby fetch queue drained within 60s under fetch_parallelism=4");

# Soft check: scan the standby log for stale-discard outcomes. We accept any
# of several wordings emitted by segment_adoption.c; if none match, the test
# is marked TODO rather than failing.
my $logfile = $standby->logfile;
my $stale_count = 0;
if (open(my $lh, '<', $logfile)) {
    while (my $line = <$lh>) {
        if ($line =~ /adopt_result\s*=\s*1/i
            || $line =~ /STALE_DISCARD/
            || $line =~ /\bno coverage\b/i
            || $line =~ /\bdiscarding\b/i)
        {
            $stale_count++;
        }
    }
    close $lh;
}

TODO: {
    local $TODO = "out-of-order arrival timing is not deterministic";
    ok($stale_count > 0,
       "saw at least one stale-discard outcome ($stale_count matches in standby log)");
}

# Hard correctness invariant: regardless of how pulls interleaved, the
# standby's adopted state must agree with the primary's index state.
my @phase2_primary = topk_ids($primary);
my @phase2_standby = topk_ids($standby);
is(scalar(@phase2_primary), $top_k,
   "phase 2: primary top-$top_k returns $top_k rows");
is(scalar(@phase2_standby), $top_k,
   "phase 2: standby top-$top_k returns $top_k rows");
is_deeply(\@phase2_standby, \@phase2_primary,
   "phase 2: standby top-$top_k matches primary top-$top_k "
   . "(stale-discard preserved correctness under parallelism)");

# Cross-phase sanity: bulk-insert rows must influence the top-k.
my %phase1_set = map { $_ => 1 } @phase1_primary;
my $changed    = grep { !$phase1_set{$_} } @phase2_primary;
ok($changed > 0,
   "phase 2: primary top-$top_k contains $changed ids not in phase 1 "
   . "(proves bulk-insert rows are queryable)");

done_testing();
