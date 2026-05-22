# Smoke test: primary + streaming replica with the vector extension loaded
use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $primary = PostgreSQL::Test::Cluster->new('primary');
$primary->init(allows_streaming => 1);
$primary->append_conf('postgresql.conf', qq(
shared_preload_libraries = 'vector'
wal_level = replica
max_wal_senders = 4
wal_keep_size = 64MB
));
$primary->start;
$primary->backup('basebackup');

my $standby = PostgreSQL::Test::Cluster->new('standby');
$standby->init_from_backup($primary, 'basebackup', has_streaming => 1);
$standby->append_conf('postgresql.conf', qq(
shared_preload_libraries = 'vector'
));
$standby->start;

$primary->safe_psql('postgres', 'CREATE EXTENSION vector;');

# Wait for the CREATE EXTENSION WAL to replay on the standby
my $appname = $standby->name;
my $caughtup = "SELECT pg_current_wal_lsn() <= replay_lsn "
             . "FROM pg_stat_replication WHERE application_name = '$appname';";
$primary->poll_query_until('postgres', $caughtup)
    or die "standby never caught up";

# Verify extension is present on standby too
my $ver = $standby->safe_psql('postgres',
    "SELECT extversion FROM pg_extension WHERE extname = 'vector';");
ok($ver ne '', "vector extension is visible on standby (got: $ver)");

# Verify our custom rmgr registered without crash and replicates simple work.
$primary->safe_psql('postgres', q(
    CREATE TABLE t (id int, v vector(4));
    INSERT INTO t SELECT i, ARRAY[i*1.0, i+0.1, i+0.2, i+0.3]::vector
        FROM generate_series(1, 10) i;
));
$primary->poll_query_until('postgres', $caughtup);

my $count = $standby->safe_psql('postgres', 'SELECT count(*) FROM t;');
is($count, '10', "10 rows replicated to standby");

done_testing();
