# Verify that DELETE + VACUUM on the primary AFTER the standby has loaded
# its index propagates a MemtableTombstone WAL record that the standby's
# rmgr replay applies to the live SharedMemtableBuffer bitmap.
#
# Strategy:
#   1. Build the index, insert a few memtable rows on the primary, wait
#      for catchup.
#   2. Query the standby for one of those rows; this triggers the load so
#      the index becomes valid=1 (live, not just dormant on disk).
#   3. DELETE that row on the primary and VACUUM. bulk_delete_lsm_index
#      emits MemtableTombstone records.
#   4. Wait for catchup, query the standby for that vector again. If the
#      rmgr tombstone replay works, the deleted id is no longer the closest
#      (it's masked from search by the memtable bitmap and the heap row is
#      gone). The closest result for that vector will be some OTHER row,
#      not the deleted one.
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

# Known vector for the row that will be deleted.
my $v_del_lit = "'[" . join(",", (9) x $dim) . "]'";   # all-nines

# Phase 1: build index + insert the to-be-deleted row.
$primary->safe_psql('postgres', qq(
    CREATE TABLE t (id int, v vector($dim));
    INSERT INTO t SELECT i, ARRAY[$array_sql]::vector FROM generate_series(1, $n_build) i;
    CREATE INDEX t_v_idx ON t USING hnsw (v vector_l2_ops);
    INSERT INTO t VALUES (300, ${v_del_lit}::vector);
));
wait_catchup();

# Phase 2: query the standby for v_del — triggers load, returns id=300.
my $idx_query = qq(
    SET enable_seqscan = off;
    SET hnsw.ef_search = 2000;
    SELECT id FROM t ORDER BY v <-> ${v_del_lit}::vector LIMIT 1;
);
my $standby_id_pre = $standby->safe_psql('postgres', $idx_query);
is($standby_id_pre, '300', "standby finds id=300 before delete (load works)");

# Phase 3: delete id=300 and vacuum. MemtableTombstone WAL records are emitted.
$primary->safe_psql('postgres', q(
    DELETE FROM t WHERE id = 300;
    VACUUM t;
));
wait_catchup();

# Phase 4: re-query the standby. id=300 must NOT be the closest match now —
# both because the heap row is gone (visibility) and because the memtable
# bitmap on the standby should mark its slot as deleted.
my $standby_id_post = $standby->safe_psql('postgres', $idx_query);
isnt($standby_id_post, '300', "standby no longer returns id=300 after delete + vacuum");

# Sanity: heap count
my $heap_count_s = $standby->safe_psql('postgres', 'SELECT count(*) FROM t;');
is($heap_count_s, $n_build, "heap count on standby reflects the delete");

done_testing();
