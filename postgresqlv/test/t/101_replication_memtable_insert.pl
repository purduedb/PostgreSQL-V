# Verify that vector inserts on the primary AFTER the standby has loaded
# its index propagate to the standby's SharedMemtableBuffer via the custom
# rmgr's Add WAL records — i.e., this test exercises the rmgr replay path
# (not the lazy-load shortcut).
#
# Strategy:
#   1. Build a small index on the primary, insert a "phase 1" memtable row
#      with a known vector v1.
#   2. Wait for catchup, then run a query on the standby for v1. This
#      triggers the standby's IndexLoadWorker, which builds the in-memory
#      memtable from the (GenericXLog-replicated) status pages + heap.
#      Confirms the load path works and v1 is reachable.
#   3. NOW insert a "phase 2" memtable row with a different known vector v2.
#      The standby's index is already loaded (valid=1), so the Add WAL record
#      MUST be applied via dpv_standby_add_to_memtable to be visible.
#   4. Wait for catchup, query the standby for v2. The closest row must be v2
#      with distance 0. If the custom rmgr replay is broken, v2 won't appear.
use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $dim = 32;
my $n_build = 100;

my $primary = PostgreSQL::Test::Cluster->new('primary');
$primary->init(allows_streaming => 1);
$primary->append_conf('postgresql.conf', qq(
shared_preload_libraries = 'vector'
wal_level = replica
max_wal_senders = 4
wal_keep_size = 256MB
));
$primary->start;
$primary->backup('basebackup');

my $standby = PostgreSQL::Test::Cluster->new('standby');
$standby->init_from_backup($primary, 'basebackup', has_streaming => 1);
$standby->append_conf('postgresql.conf', qq(
shared_preload_libraries = 'vector'
hot_standby = on
));
$standby->start;

sub wait_catchup {
    my $appname = $standby->name;
    my $q = "SELECT pg_current_wal_lsn() <= replay_lsn "
          . "FROM pg_stat_replication WHERE application_name = '$appname';";
    $primary->poll_query_until('postgres', $q) or die "standby never caught up";
}

$primary->safe_psql('postgres', 'CREATE EXTENSION vector;');
my $array_sql = join(",", ('random()') x $dim);

# Two probe vectors with values well outside the random-uniform-[0,1) range
# so they're unambiguous nearest-neighbour candidates for themselves.
my $v1_lit = "'[" . join(",", (7) x $dim) . "]'";   # all-sevens
my $v2_lit = "'[" . join(",", (-5) x $dim) . "]'";  # all-negative-fives

# Phase 1: build the index, then insert one row with known vector v1.
$primary->safe_psql('postgres', qq(
    CREATE TABLE t (id int, v vector($dim));
    INSERT INTO t SELECT i, ARRAY[$array_sql]::vector FROM generate_series(1, $n_build) i;
    CREATE INDEX t_v_idx ON t USING hnsw (v vector_l2_ops);
    INSERT INTO t VALUES (201, ${v1_lit}::vector);
));
wait_catchup();

# Standby query for v1. This is the first index touch on the standby, so it
# triggers the IndexLoadWorker. The memtable is built from status pages + heap.
my $idx_query_v1 = qq(
    SET enable_seqscan = off;
    SET hnsw.ef_search = 2000;
    SELECT id FROM t ORDER BY v <-> ${v1_lit}::vector LIMIT 1;
);
my $standby_id_v1 = $standby->safe_psql('postgres', $idx_query_v1);
is($standby_id_v1, '201',
   "standby finds v1 (id=201) after first-touch lazy load — load + GenericXLog path works");

# Phase 2: with the standby's index already loaded, insert another known row.
# This Add MUST be applied via dpv_standby_add_to_memtable (custom rmgr redo)
# to be visible — the load already ran, so there's no second-chance rebuild.
$primary->safe_psql('postgres', "INSERT INTO t VALUES (202, ${v2_lit}::vector);");
wait_catchup();

# Query the standby for v2. If the rmgr replay materialized this Add into the
# SharedMemtableBuffer, the nearest neighbour to v2 is id=202 (distance 0).
my $idx_query_v2 = qq(
    SET enable_seqscan = off;
    SET hnsw.ef_search = 2000;
    SELECT id FROM t ORDER BY v <-> ${v2_lit}::vector LIMIT 1;
);
my $standby_id_v2 = $standby->safe_psql('postgres', $idx_query_v2);
is($standby_id_v2, '202',
   "standby finds v2 (id=202) inserted AFTER first load — custom rmgr Add replay works");

# Sanity: heap count is correct on both sides.
my $heap_count_s = $standby->safe_psql('postgres', 'SELECT count(*) FROM t;');
is($heap_count_s, ($n_build + 2), "heap on standby reflects both phase-1 and phase-2 inserts");

done_testing();
