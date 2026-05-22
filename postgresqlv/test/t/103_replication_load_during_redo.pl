# Verify that rows inserted on the primary AFTER the standby's basebackup
# but BEFORE the standby ever touches the index are still queryable on the
# standby once it catches up. This exercises the standby's lazy-load path
# running on top of status pages that were rebuilt entirely from replicated
# WAL (GenericXLog + custom rmgr records).
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

$primary->safe_psql('postgres', 'CREATE EXTENSION vector;');
my $array_sql = join(",", ('random()') x $dim);

# Build the index BEFORE the basebackup. The build flushes a segment for
# the initial $n_build rows.
$primary->safe_psql('postgres', qq(
    CREATE TABLE t (id int, v vector($dim));
    INSERT INTO t SELECT i, ARRAY[$array_sql]::vector FROM generate_series(1, $n_build) i;
    CREATE INDEX t_v_idx ON t USING hnsw (v vector_l2_ops);
));
$primary->backup('basebackup');

# Known vector inserted AFTER the basebackup — only reaches the standby via WAL.
my $v_post_lit = "'[" . join(",", (3) x $dim) . "]'";
$primary->safe_psql('postgres', "INSERT INTO t VALUES (400, ${v_post_lit}::vector);");

my $standby = PostgreSQL::Test::Cluster->new('standby');
$standby->init_from_backup($primary, 'basebackup', has_streaming => 1);
$standby->append_conf('postgresql.conf', qq(
shared_preload_libraries = 'vector'
hot_standby = on
));
$standby->start;

my $appname = $standby->name;
my $caughtup = "SELECT pg_current_wal_lsn() <= replay_lsn "
             . "FROM pg_stat_replication WHERE application_name = '$appname';";
$primary->poll_query_until('postgres', $caughtup)
    or die "standby never caught up";

# Heap sanity.
my $heap_count_s = $standby->safe_psql('postgres', 'SELECT count(*) FROM t;');
is($heap_count_s, ($n_build + 1), "heap on standby includes the post-backup insert");

# Query the standby for the post-backup vector. The standby has never touched
# this index before; this query triggers the first-touch load. The load reads
# status pages (kept current by GenericXLog replay), builds the memtable from
# heap, and the post-backup row should be returned.
my $idx_query = qq(
    SET enable_seqscan = off;
    SET hnsw.ef_search = 2000;
    SELECT id FROM t ORDER BY v <-> ${v_post_lit}::vector LIMIT 1;
);
my $standby_id = $standby->safe_psql('postgres', $idx_query);
is($standby_id, '400',
   "standby finds the post-backup row via first-touch lazy load after WAL replay");

done_testing();
