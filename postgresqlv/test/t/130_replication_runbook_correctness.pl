# 130_replication_runbook_correctness.pl
#
# Physical-replication correctness sweep for pgvector, driven by the same
# YAML runbook + .fvecs/.fbin dataset + .npy ground-truth files used by
# pgvector/test/runbooks/pgvector_test.cpp.
#
# Strategy
# --------
# 1. Spin up a primary + streaming standby pair (Plan-2 replication GUCs).
# 2. Optionally run a handful of trivially-deterministic edge cases first
#    (empty table, single row, delete-all, reinsert-after-delete).
# 3. Walk through the runbook sequentially on the PRIMARY:
#       - insert / delete steps just mutate primary state (and update
#         active_ranges for ground-truth filename hashing).
#       - search  steps trigger the SOLE correctness check:
#             a. wait for the standby's WAL replay to catch up
#             b. run the same N queries against primary and standby
#             c. compute average recall on each side against the
#                pre-computed .npy ground truth
#             d. assert |avg_p - avg_s| <= $recall_tolerance
#    (per-side recall is always reported as TAP diag so a regression
#    that drifts both sides equally is visible.)
# 4. Optional failure scenarios — restart standby / crash+restart
#    primary / SIGSTOP the segment_fetcher — are off by default; each
#    is fired exactly once at $failure_inject_at_step and the NEXT
#    search step's verification is the assertion.
#
# What this test does NOT verify (by design)
# ------------------------------------------
#   * Heap integrity (vector.* is not in the heap-management path).
#   * Per-query result-set equality (HNSW search is multi-threaded so
#     even the ordered ID set is not deterministic across two runs of
#     the SAME query — only the *average recall* is meaningfully
#     comparable across primary and standby).
#   * Performance / throughput / latency.
#
# Prereqs the user must set up (paths or env vars at top of file):
#   - .fvecs/.fbin dataset and query files
#   - YAML runbook (same one fed to pgvector_test.cpp)
#   - directory of pre-computed .npy ground-truth files (produced by
#     compute_ground_truth.cpp). Filename convention matches the C++
#     test: <runbook_name>_step<N>_ranges<8hex>_gt.npy
#     where 8hex is the first 4 bytes of MD5 over the JSON of
#     active_ranges built so far.
#
# This test takes a long time on real runbooks; you almost certainly
# want --start-step / --end-step (env vars) to scope it down.

use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;
use Digest::MD5 qw(md5_hex);
use File::Spec;
use File::Path qw(make_path);
use Cwd qw(abs_path);
use List::Util qw(min sum);

$| = 1;   # unbuffered stdout so progress diags appear live under prove

# ============================================================
# CONFIGURATION  (edit here, or override via env vars)
# ============================================================

# ---- Dataset + runbook ----
my $dataset_path = $ENV{DATASET_PATH}
    // '/ssd_root/dataset/turing10m/msturing-10M.fvecs';
my $query_path   = $ENV{QUERY_PATH}
    // '/ssd_root/dataset/turing10m/msturing-query.fvecs';
my $runbook_path = $ENV{RUNBOOK_PATH}
    // File::Spec->catfile($ENV{PWD} // '.', '..', 'runbooks',
                           'msturing-10M_slidingwindow_1M_runbook.yaml');
my $dataset_name = $ENV{DATASET_NAME} // 'msturing-10M';
my $gt_dir       = $ENV{GT_DIR}
    // '/ssd_root/liu4127/msturing_runbook_gt';
my $dataset_offset = $ENV{DATASET_OFFSET} // 0;

# Limit the runbook walk (1-based; 0 = no limit)
my $start_step         = $ENV{START_STEP}         // 101;
my $end_step           = $ENV{END_STEP}           // 400;
my $build_index_before = $ENV{BUILD_INDEX_BEFORE} // 101;

# ---- Index ----
my $index_type           = $ENV{INDEX_TYPE} // 'hnsw';   # 'hnsw' | 'ivfflat'
my $hnsw_m               = $ENV{HNSW_M}               // 16;
my $hnsw_ef_construction = $ENV{HNSW_EF_CONSTRUCTION} // 40;
my $hnsw_ef_search       = $ENV{HNSW_EF_SEARCH}       // 200;
my $ivfflat_lists        = $ENV{IVFFLAT_LISTS}        // 100;
my $ivfflat_probes       = $ENV{IVFFLAT_PROBES}       // 10;

# ---- Verification ----
my $num_verify_queries = $ENV{NUM_VERIFY_QUERIES} // 10000;
my $recall_tolerance   = $ENV{RECALL_TOLERANCE}   // 0.02;

# ---- Cluster / workload ----
my $dpv_port      = $ENV{DPV_PORT}      // 18130;
my $insert_batch  = $ENV{INSERT_BATCH}  // 500;
my $shared_secret = 'dpv_test_secret';

# Emit an intra-insert progress line every this many rows (0 = silent).
my $insert_progress_every = $ENV{INSERT_PROGRESS_EVERY} // 10000;

# If set, initialize the PRIMARY from this existing PostgreSQL data directory
# (e.g. one produced by pgvector_test.cpp with the first 100 steps already
# inserted) instead of running initdb. The STANDBY is then created by
# pg_basebackup of that primary, so it inherits the same seed. Pair this with
# START_STEP=101 to skip the steps already materialized in the seed.
# Requirements:
#   * the source cluster MUST be stopped (no postmaster.pid) so the copy is
#     physically consistent;
#   * it SHOULD be a data dir created by the SAME Postgres build the test runs
#     under (pg_build_17), else catalog/version mismatch will refuse to start;
#   * the seed must already contain `CREATE EXTENSION vector` and the `vectors`
#     table (both true for a pgvector_test.cpp run).
my $source_pgdata = $ENV{SOURCE_PGDATA} // '/ssd_root/liu4127/postgresql_vec_runbook_insert';

# ---- Edge cases (comment out to skip) ----
my %enable_edge_case = (
    # empty_table_query          => 1,
    # single_row_state           => 1,
    # delete_all_then_query      => 1,
    # reinsert_after_full_delete => 1,
);

# ---- Optional failure scenarios (flip to 1 to enable) ----
my %enable_failure_scenario = (
    standby_restart_mid_workload    => 0,
    primary_crash_restart           => 0,
    fetcher_pause_query_correctness => 0,
);
my $failure_inject_at_step = $ENV{FAILURE_INJECT_AT_STEP} // 0;  # 0 = midpoint

# ============================================================
# DATA HELPERS
# ============================================================

sub vec_to_sql_literal {
    my ($vec) = @_;
    return "[" . join(",", map { sprintf("%.6f", $_) } @$vec) . "]";
}

sub load_fvecs {
    my ($path) = @_;
    open my $fh, '<:raw', $path or die "open $path: $!";
    my $buf;
    read($fh, $buf, 4) == 4 or die "fvecs $path: truncated dim";
    my $dim = unpack 'l<', $buf;
    die "fvecs $path: nonsense dim=$dim" if $dim <= 0 || $dim > 65536;
    my @st = stat $fh;
    my $rec_bytes = 4 + 4 * $dim;
    my $n = int($st[7] / $rec_bytes);
    seek($fh, 0, 0) or die "seek $path: $!";
    my @vecs;
    for (1 .. $n) {
        read($fh, $buf, 4);
        read($fh, $buf, 4 * $dim) == 4 * $dim
            or die "fvecs $path: truncated payload";
        push @vecs, [ unpack 'f<*', $buf ];
    }
    close $fh;
    return (\@vecs, $dim, $n);
}

sub load_fbin {
    my ($path) = @_;
    open my $fh, '<:raw', $path or die "open $path: $!";
    my $buf;
    read($fh, $buf, 8) == 8 or die "fbin $path: short header";
    my ($n, $dim) = unpack 'l<l<', $buf;
    die "fbin $path: invalid n=$n d=$dim" if $n <= 0 || $dim <= 0;
    my @vecs;
    for (1 .. $n) {
        read($fh, $buf, 4 * $dim) == 4 * $dim
            or die "fbin $path: truncated payload";
        push @vecs, [ unpack 'f<*', $buf ];
    }
    close $fh;
    return (\@vecs, $dim, $n);
}

sub load_dataset_file {
    my ($path) = @_;
    return load_fbin($path)  if $path =~ /\.fbin$/i;
    return load_fvecs($path) if $path =~ /\.fvecs$/i;
    die "unknown dataset extension: $path (need .fvecs or .fbin)";
}

# Seekable, on-demand vector reader. The full msturing dataset is 10M x 100
# floats; loading it into Perl scalars costs ~32 GB and makes every safe_psql
# fork() crawl. The runbook only inserts a subset of vectors, so read each one
# from disk on demand instead. Returns a handle; use read_vec($r, $i).
sub open_vec_reader {
    my ($path) = @_;
    open my $fh, '<:raw', $path or die "open $path: $!";
    my $buf;
    if ($path =~ /\.fbin$/i) {
        read($fh, $buf, 8) == 8 or die "fbin $path: short header";
        my ($n, $dim) = unpack 'l<l<', $buf;
        die "fbin $path: invalid n=$n d=$dim" if $n <= 0 || $dim <= 0;
        return { fh => $fh, dim => $dim, n => $n, fmt => 'fbin' };
    }
    elsif ($path =~ /\.fvecs$/i) {
        read($fh, $buf, 4) == 4 or die "fvecs $path: truncated dim";
        my $dim = unpack 'l<', $buf;
        die "fvecs $path: nonsense dim=$dim" if $dim <= 0 || $dim > 65536;
        my @st = stat $fh;
        my $n = int($st[7] / (4 + 4 * $dim));
        return { fh => $fh, dim => $dim, n => $n, fmt => 'fvecs' };
    }
    die "unknown dataset extension: $path (need .fvecs or .fbin)";
}

sub read_vec {
    my ($r, $i) = @_;
    my $dim = $r->{dim};
    # fvecs: each record is [int32 dim][dim float32]; floats of vec i at
    #        i*(4+4*dim)+4.  fbin: 8-byte header then contiguous float32.
    my $off = $r->{fmt} eq 'fvecs' ? $i * (4 + 4 * $dim) + 4
                                   : 8 + $i * 4 * $dim;
    seek($r->{fh}, $off, 0) or die "seek vec $i: $!";
    my $buf;
    read($r->{fh}, $buf, 4 * $dim) == 4 * $dim
        or die "read vec $i: truncated";
    return [ unpack 'f<*', $buf ];
}

# Mirrors compute_ground_truth.cpp / pgvector_test.cpp filename scheme:
#   ranges JSON: [["insert",0,1000],["delete",500,800],...]
#   md5 of that string, hex; take FIRST 4 bytes (= 8 hex chars).
sub gt_filename {
    my ($runbook_name, $step_num, $ranges_ref) = @_;
    my $ranges_str = '[' . join(',', map {
        '["' . $_->[0] . '",' . $_->[1] . ',' . $_->[2] . ']'
    } @$ranges_ref) . ']';
    my $hash = substr(md5_hex($ranges_str), 0, 8);
    return "${runbook_name}_step${step_num}_ranges${hash}_gt.npy";
}

# Read a .npy file produced by compute_ground_truth.cpp:
# 6-byte magic + 2-byte version + uint16 header length + header dict +
# (num_queries * k) int32 payload.
sub load_gt_npy {
    my ($path, $num_queries, $k) = @_;
    open my $fh, '<:raw', $path or die "open $path: $!";
    my $buf;
    read($fh, $buf, 6) == 6 or die "$path: short magic";
    read($fh, $buf, 2) == 2 or die "$path: short version";
    read($fh, $buf, 2) == 2 or die "$path: short header length";
    my $hlen = unpack 'v', $buf;
    read($fh, $buf, $hlen) == $hlen or die "$path: short header body";
    my @gt;
    for (1 .. $num_queries) {
        read($fh, $buf, 4 * $k) == 4 * $k or die "$path: truncated payload";
        push @gt, [ unpack 'l<*', $buf ];
    }
    close $fh;
    return \@gt;
}

# Tiny line-oriented parser for the runbook YAML subset we actually
# use: scalar key:value plus 2-space-indented nested maps. No flow
# style, no arrays, no anchors. Sufficient for the runbooks shipped
# with this fork.
sub parse_runbook_yaml {
    my ($path) = @_;
    open my $fh, '<', $path or die "open $path: $!";
    my $root = {};
    my @stack   = ($root);
    my @indents = (-1);
    while (my $line = <$fh>) {
        chomp $line;
        $line =~ s/#.*$//;
        next if $line =~ /^\s*$/;
        my ($lead) = $line =~ /^(\s*)/;
        my $ilen = length $lead;
        my $body = $line;
        $body =~ s/^\s+//;
        while (@indents > 1 && $indents[-1] >= $ilen) {
            pop @stack;
            pop @indents;
        }
        if ($body =~ /^([^:]+):\s*(.*)$/) {
            my ($k, $v) = ($1, $2);
            $k =~ s/^\s+|\s+$//g;
            $v =~ s/^\s+|\s+$//g;
            if ($v eq '') {
                my $child = {};
                $stack[-1]->{$k} = $child;
                push @stack,   $child;
                push @indents, $ilen;
            }
            else {
                $stack[-1]->{$k} = $v;
            }
        }
    }
    close $fh;
    return $root;
}

# Every key under the dataset section is a step EXCEPT the reserved
# metadata keys (matches pgvector_test.cpp / compute_ground_truth.cpp).
# Step keys may be bare integers ("1".."400") or "stepN"; sort them by
# their embedded integer so 10 follows 9 (NOT 1, 10, 2, ...).
sub ordered_step_keys {
    my ($section) = @_;
    my @keys = grep {
        $_ ne 'max_pts' && $_ ne 'query' && $_ ne 'groundtruth'
    } keys %$section;
    return sort {
        my ($na) = $a =~ /(\d+)/;
        my ($nb) = $b =~ /(\d+)/;
        ((defined $na ? $na : 0) <=> (defined $nb ? $nb : 0)) or ($a cmp $b);
    } @keys;
}

# ============================================================
# CLUSTER HELPERS
# ============================================================

# Populate a (not-yet-initialized) Cluster node's pgdata from an existing data
# directory, then rewrite the connection settings so the TAP framework can
# reach it. Replaces $node->init for the from-seed path.
sub init_primary_from_existing {
    my ($node, $source) = @_;
    BAIL_OUT("SOURCE_PGDATA not a directory: $source") unless -d $source;
    BAIL_OUT("SOURCE_PGDATA missing PG_VERSION (not a data dir?): $source")
        unless -f "$source/PG_VERSION";
    BAIL_OUT("SOURCE_PGDATA has a postmaster.pid — stop that cluster first so "
           . "the copy is consistent: $source")
        if -f "$source/postmaster.pid";

    diag("initializing primary from existing data dir: $source");
    # Physical copy of the whole data dir into the framework-managed pgdata
    # (-T: treat dest as the copy itself, not a directory to copy into).
    PostgreSQL::Test::Utils::system_or_bail('cp', '-aT', $source,
                                            $node->data_dir);
    chmod 0700, $node->data_dir;

    # ALTER SYSTEM settings from the source cluster may pin the old port or
    # storage_base_dir and would override our overrides (auto.conf is read
    # last). Clear it; everything we need is set via postgresql.conf below.
    my $auto = $node->data_dir . '/postgresql.auto.conf';
    if (-f $auto) {
        open my $af, '>', $auto
            or BAIL_OUT("cannot truncate $auto: $!");
        print $af "# cleared by 130_replication_runbook_correctness.pl\n";
        close $af;
    }

    # The copied postgresql.conf still points at the original cluster's port /
    # socket dir. append_conf writes LAST, so these win.
    my $host = $node->host;
    my $port = $node->port;
    if ($host =~ m{^/}) {
        $node->append_conf('postgresql.conf', qq(
listen_addresses = ''
unix_socket_directories = '$host'
port = $port
));
    }
    else {
        $node->append_conf('postgresql.conf', qq(
listen_addresses = '$host'
port = $port
));
    }
    # Trust auth for the local test user, plus replication for the standby's
    # basebackup. (The seed's pg_hba may not cover the framework's user.)
    $node->append_conf('pg_hba.conf', "local all all trust");
    $node->append_conf('pg_hba.conf', "local replication all trust");
    if ($host !~ m{^/}) {
        $node->append_conf('pg_hba.conf', "host all all 127.0.0.1/32 trust");
        $node->append_conf('pg_hba.conf',
                           "host replication all 127.0.0.1/32 trust");
    }
}

sub make_cluster {
    my $primary = PostgreSQL::Test::Cluster->new('primary130');
    if ($source_pgdata) {
        init_primary_from_existing($primary, $source_pgdata);
    }
    else {
        $primary->init(allows_streaming => 1);
    }
    # storage_base_dir MUST be absolute (Postgres backends run with CWD = the
    # data dir, so a relative path resolves wrong) and the base must pre-exist
    # (the extension's mkdir of <base>/<reloid> is non-recursive). Place it
    # under the node basedir (sibling of pgdata) so pg_basebackup doesn't copy
    # the primary's segment files into the standby.
    my $primary_storage = abs_path($primary->basedir) . '/pgvector_storage';
    make_path($primary_storage);
    $primary->append_conf('postgresql.conf', qq(
shared_preload_libraries = 'vector'
wal_level = replica
max_wal_senders = 8
max_worker_processes = 32
wal_keep_size = 1GB
pgvector.replication_role = 'primary'
pgvector.replication_primary_port = $dpv_port
pgvector.replication_shared_secret = '$shared_secret'
pgvector.storage_base_dir = '$primary_storage'
));
    $primary->start;
    $primary->backup('basebackup');

    my $standby = PostgreSQL::Test::Cluster->new('standby130');
    $standby->init_from_backup($primary, 'basebackup', has_streaming => 1);
    my $standby_storage = abs_path($standby->basedir) . '/pgvector_storage';
    make_path($standby_storage);
    $standby->append_conf('postgresql.conf', qq(
shared_preload_libraries = 'vector'
hot_standby = on
max_worker_processes = 32
pgvector.replication_role = 'standby'
pgvector.replication_primary_host = '127.0.0.1'
pgvector.replication_primary_port = $dpv_port
pgvector.replication_shared_secret = '$shared_secret'
pgvector.replication_fetch_parallelism = 2
pgvector.storage_base_dir = '$standby_storage'
));
    $standby->start;
    return ($primary, $standby);
}

sub wait_catchup {
    my ($primary, $standby) = @_;
    my $appname = $standby->name;
    my $q = "SELECT pg_current_wal_lsn() <= replay_lsn "
          . "FROM pg_stat_replication WHERE application_name = '$appname';";
    $primary->poll_query_until('postgres', $q)
        or die "standby never caught up";
}

sub set_search_params_sql {
    return $index_type eq 'ivfflat'
        ? "SET ivfflat.probes = $ivfflat_probes;"
        : "SET hnsw.ef_search = $hnsw_ef_search;";
}

sub create_table_sql {
    my ($dim) = @_;
    return "CREATE TABLE vectors (id BIGINT PRIMARY KEY, vec vector($dim));";
}

sub create_index_sql {
    if ($index_type eq 'ivfflat') {
        return "CREATE INDEX ON vectors USING ivfflat (vec vector_l2_ops) "
             . "WITH (lists = $ivfflat_lists);";
    }
    return "CREATE INDEX ON vectors USING hnsw (vec vector_l2_ops) "
         . "WITH (m = $hnsw_m, ef_construction = $hnsw_ef_construction);";
}

# ============================================================
# BATCHED WORKLOAD HELPERS  (primary only)
# ============================================================

sub insert_range_batched {
    my ($node, $dreader, $start, $end, $label) = @_;
    my $total = $end - $start;
    my $done  = 0;
    my $next_report = $insert_progress_every;
    for (my $b = $start; $b < $end; $b += $insert_batch) {
        my $e = min($b + $insert_batch, $end);
        my @rows;
        for my $i ($b .. $e - 1) {
            my $row_id = $i + $dataset_offset;
            my $lit    = vec_to_sql_literal(read_vec($dreader, $i));
            push @rows, "($row_id, '$lit')";
        }
        my $sql = "INSERT INTO vectors (id, vec) VALUES "
                . join(",", @rows)
                . " ON CONFLICT (id) DO NOTHING;";
        my ($ret, $out, $err) =
            $node->psql('postgres', $sql, on_error_die => 0);
        die "insert batch [$b,$e) failed: $err" if $ret != 0;
        $done += ($e - $b);
        if ($label && $insert_progress_every > 0 && $done >= $next_report) {
            diag(sprintf("    %s: inserted %d/%d rows", $label, $done, $total));
            $next_report += $insert_progress_every;
        }
    }
}

sub delete_range_batched {
    my ($node, $start, $end) = @_;
    my $a = $start + $dataset_offset;
    my $b = $end   + $dataset_offset;
    $node->safe_psql('postgres',
        "DELETE FROM vectors WHERE id >= $a AND id < $b;");
}

# ============================================================
# QUERY + RECALL
# ============================================================

sub recall_one {
    my ($returned_ref, $gt_row_ref, $k) = @_;
    return 0.0 if !@$returned_ref;
    my %gt = map { $_ => 1 } @$gt_row_ref[0 .. min($k, scalar @$gt_row_ref) - 1];
    my $hits = 0;
    for my $id (@$returned_ref) {
        $hits++ if exists $gt{$id};
    }
    return $hits / $k;
}

# Run all $num_q queries in ONE psql session (the index loads once per session
# and we pay a single fork/connect), delimiting each result set with an \echo
# marker so we can split the combined output. This replaces $num_q separate
# safe_psql calls — the per-query psql spawn was the dominant cost.
sub avg_recall_on_node {
    my ($node, $queries_ref, $gt_ref, $k, $num_q) = @_;

    my $script = set_search_params_sql() . "\n";
    for my $q (0 .. $num_q - 1) {
        my $lit = vec_to_sql_literal($queries_ref->[$q]);
        $script .= "\\echo __Q${q}__\n";
        $script .= "SELECT id FROM vectors "
                 . "ORDER BY vec <-> '$lit'::vector LIMIT $k;\n";
    }

    my $out = $node->safe_psql('postgres', $script);

    # Parse: lines "__Q<n>__" start query n's result block; numeric lines
    # within are returned ids.
    my @buckets;
    my $cur = -1;
    for my $line (split /\n/, ($out // '')) {
        if ($line =~ /^__Q(\d+)__$/) { $cur = $1; $buckets[$cur] = []; next; }
        next if $cur < 0;
        push @{ $buckets[$cur] }, $line if $line =~ /^\d+$/;
    }

    my @per_query;
    for my $q (0 .. $num_q - 1) {
        push @per_query, recall_one($buckets[$q] // [], $gt_ref->[$q], $k);
    }
    my $avg = @per_query ? (sum(@per_query) / scalar @per_query) : 0.0;
    return ($avg, \@per_query);
}

# ============================================================
# EDGE CASES  (trivially-deterministic invariants only)
# ============================================================

sub edge_case_empty_table {
    my ($primary, $standby, $dim) = @_;
    return unless $enable_edge_case{empty_table_query};

    $primary->safe_psql('postgres', 'DROP TABLE IF EXISTS vectors;');
    $primary->safe_psql('postgres', create_table_sql($dim));
    wait_catchup($primary, $standby);

    my $zero = "'[" . join(",", (0) x $dim) . "]'";
    my $pc = $primary->safe_psql('postgres',
        "SELECT count(*) FROM (SELECT id FROM vectors "
      . "ORDER BY vec <-> ${zero}::vector LIMIT 10) s;");
    my $sc = $standby->safe_psql('postgres',
        "SELECT count(*) FROM (SELECT id FROM vectors "
      . "ORDER BY vec <-> ${zero}::vector LIMIT 10) s;");
    is($pc, '0', 'edge: empty table primary returns 0 rows');
    is($sc, '0', 'edge: empty table standby returns 0 rows');
}

sub edge_case_single_row {
    my ($primary, $standby, $dim) = @_;
    return unless $enable_edge_case{single_row_state};

    $primary->safe_psql('postgres', 'DROP TABLE IF EXISTS vectors;');
    $primary->safe_psql('postgres', create_table_sql($dim));
    my $one = "'[" . join(",", (1) x $dim) . "]'";
    $primary->safe_psql('postgres',
        "INSERT INTO vectors (id, vec) VALUES (42, ${one}::vector);");
    wait_catchup($primary, $standby);

    my $pid = $primary->safe_psql('postgres',
        "SELECT id FROM vectors ORDER BY vec <-> ${one}::vector LIMIT 1;");
    my $sid = $standby->safe_psql('postgres',
        "SELECT id FROM vectors ORDER BY vec <-> ${one}::vector LIMIT 1;");
    is($pid, '42', 'edge: single-row primary returns id 42');
    is($sid, '42', 'edge: single-row standby returns id 42');
}

sub edge_case_delete_all {
    my ($primary, $standby, $dim) = @_;
    return unless $enable_edge_case{delete_all_then_query};

    $primary->safe_psql('postgres', 'DROP TABLE IF EXISTS vectors;');
    $primary->safe_psql('postgres', create_table_sql($dim));
    my $one = "'[" . join(",", (1) x $dim) . "]'";
    $primary->safe_psql('postgres', qq(
        INSERT INTO vectors (id, vec)
        SELECT g, ${one}::vector FROM generate_series(1, 50) g;
    ));
    $primary->safe_psql('postgres', 'DELETE FROM vectors;');
    wait_catchup($primary, $standby);

    my $pc = $primary->safe_psql('postgres', 'SELECT count(*) FROM vectors;');
    my $sc = $standby->safe_psql('postgres', 'SELECT count(*) FROM vectors;');
    is($pc, '0', 'edge: delete-all primary count = 0');
    is($sc, '0', 'edge: delete-all standby count = 0');
}

sub edge_case_reinsert_after_full_delete {
    my ($primary, $standby, $dim) = @_;
    return unless $enable_edge_case{reinsert_after_full_delete};

    $primary->safe_psql('postgres', 'DROP TABLE IF EXISTS vectors;');
    $primary->safe_psql('postgres', create_table_sql($dim));
    my $a   = "'[" . join(",", (1) x $dim) . "]'";
    my $b   = "'[" . join(",", (9) x $dim) . "]'";
    $primary->safe_psql('postgres',
        "INSERT INTO vectors (id, vec) VALUES (7, ${a}::vector);");
    $primary->safe_psql('postgres', 'DELETE FROM vectors WHERE id = 7;');
    $primary->safe_psql('postgres',
        "INSERT INTO vectors (id, vec) VALUES (7, ${b}::vector);");
    wait_catchup($primary, $standby);

    # Query for b — closest on each side must be id=7 with the NEW vec
    # (i.e. distance ~ 0). If the standby still has the stale row, the
    # distance will be nonzero.
    my $pd = $primary->safe_psql('postgres',
        "SELECT round((vec <-> ${b}::vector)::numeric, 4) FROM vectors WHERE id = 7;");
    my $sd = $standby->safe_psql('postgres',
        "SELECT round((vec <-> ${b}::vector)::numeric, 4) FROM vectors WHERE id = 7;");
    is($pd, '0.0000', 'edge: reinsert primary has new vec for id 7');
    is($sd, '0.0000', 'edge: reinsert standby has new vec for id 7');
}

# ============================================================
# FAILURE-SCENARIO HOOKS  (all off by default)
# ============================================================

sub maybe_fire_restart_scenarios {
    my ($step_num, $primary, $standby) = @_;
    return unless $failure_inject_at_step
                && $step_num == $failure_inject_at_step;

    if ($enable_failure_scenario{standby_restart_mid_workload}) {
        diag("[failure] restarting standby at step $step_num");
        $standby->stop('immediate');
        $standby->start;
    }
    if ($enable_failure_scenario{primary_crash_restart}) {
        diag("[failure] crash+restart primary at step $step_num");
        $primary->stop('immediate');
        $primary->start;
    }
}

sub find_fetcher_pids {
    my ($standby) = @_;
    my $out = $standby->safe_psql('postgres', q(
        SELECT pid FROM pg_stat_activity
        WHERE backend_type = 'DpvSegmentFetcher';
    ));
    return () if !defined $out || $out eq '';
    return grep { /^\d+$/ } split /\n/, $out;
}

sub maybe_pause_fetcher {
    my ($step_num, $standby) = @_;
    return [] unless $enable_failure_scenario{fetcher_pause_query_correctness}
                  && $failure_inject_at_step
                  && $step_num == $failure_inject_at_step;
    my @pids = find_fetcher_pids($standby);
    if (!@pids) {
        diag("[failure] fetcher pause requested but no DpvSegmentFetcher "
            ."found on standby; skipping");
        return [];
    }
    for my $pid (@pids) {
        diag("[failure] SIGSTOP fetcher pid=$pid");
        kill 'STOP', $pid;
    }
    return \@pids;
}

sub resume_fetcher {
    my ($paused_pids_ref) = @_;
    return unless $paused_pids_ref && @$paused_pids_ref;
    for my $pid (@$paused_pids_ref) {
        diag("[failure] SIGCONT fetcher pid=$pid");
        kill 'CONT', $pid;
    }
}

# ============================================================
# RUNBOOK EXECUTION  (the heart of the test)
# ============================================================

# A "step" we actually act on (filters out max_pts / query / groundtruth).
sub collect_steps {
    my ($section) = @_;
    my @out;
    my $n = 0;
    for my $key (ordered_step_keys($section)) {
        $n++;
        my $node = $section->{$key};
        next unless ref($node) eq 'HASH';
        my $op = $node->{operation};
        next if !defined $op;
        push @out, {
            step_num => $n,
            step_key => $key,
            op       => $op,
            start    => $node->{start},
            end      => $node->{end},
            k        => $node->{k},
        };
    }
    return @out;
}

# Build the active_ranges prefix the C++ test uses when --start-step > 1,
# so the .npy filename hash agrees with whatever compute_ground_truth.cpp
# wrote.
sub seed_active_ranges_up_to {
    my ($steps_ref, $upto_step) = @_;
    my @ranges;
    return \@ranges if $upto_step <= 1;
    for my $s (@$steps_ref) {
        last if $s->{step_num} >= $upto_step;
        next unless $s->{op} eq 'insert' || $s->{op} eq 'delete';
        push @ranges, [ $s->{op}, $s->{start}, $s->{end} ];
    }
    return \@ranges;
}

sub run_runbook {
    my ($primary, $standby, $dreader, $queries_ref) = @_;

    my $runbook = parse_runbook_yaml($runbook_path);
    my $section = $runbook->{$dataset_name}
        or BAIL_OUT("dataset_name '$dataset_name' not present in runbook");
    my @steps = collect_steps($section);
    BAIL_OUT("runbook has no actionable steps") if !@steps;

    my $runbook_id = $dataset_name;
    $runbook_id =~ tr/-/_/;

    my $active_ranges = seed_active_ranges_up_to(\@steps, $start_step);
    my $index_created = 0;
    my $verified = 0;

    # How many steps actually fall in the requested range (for "i/total").
    my $total_in_range = grep {
        (!$start_step || $_->{step_num} >= $start_step)
            && (!$end_step || $_->{step_num} <= $end_step)
    } @steps;
    my $processed = 0;

    for my $s (@steps) {
        my $n = $s->{step_num};
        next if $start_step > 0 && $n < $start_step;
        last if $end_step   > 0 && $n > $end_step;

        $processed++;
        my $desc = ($s->{op} eq 'search')
            ? "k=" . ($s->{k} // 100)
            : "rows [$s->{start},$s->{end})";
        diag(sprintf("[%d/%d] step %d (%s): %s %s",
             $processed, $total_in_range, $n, $s->{step_key},
             $s->{op}, $desc));

        # Build index at the configured boundary
        if (!$index_created && $build_index_before > 0
            && $n == $build_index_before)
        {
            diag("    building $index_type index on current rows "
                ."(this can take a while)...");
            $primary->safe_psql('postgres', create_index_sql());
            $index_created = 1;
            wait_catchup($primary, $standby);
            diag("    index built and replicated");
        }

        if ($s->{op} eq 'insert') {
            insert_range_batched($primary, $dreader,
                                 $s->{start}, $s->{end}, "step $n");
            push @$active_ranges, [ 'insert', $s->{start}, $s->{end} ];
        }
        elsif ($s->{op} eq 'delete') {
            delete_range_batched($primary, $s->{start}, $s->{end});
            push @$active_ranges, [ 'delete', $s->{start}, $s->{end} ];
        }
        elsif ($s->{op} eq 'search') {
            # Default k MUST match compute_ground_truth.cpp (k=100) so the
            # GT npy column count and our LIMIT agree.
            my $k = $s->{k} // 100;
            my $num_q = min($num_verify_queries, scalar @$queries_ref);

            maybe_fire_restart_scenarios($n, $primary, $standby);
            wait_catchup($primary, $standby);

            my $paused = maybe_pause_fetcher($n, $standby);

            my $gt_file = File::Spec->catfile(
                $gt_dir, gt_filename($runbook_id, $n, $active_ranges));
            if (! -f $gt_file) {
                resume_fetcher($paused);
                fail("step $n ($s->{step_key}): ground-truth file missing: "
                    . $gt_file);
                next;
            }
            my $gt = load_gt_npy($gt_file, $num_q, $k);

            my ($avg_p, $per_p) =
                avg_recall_on_node($primary, $queries_ref, $gt, $k, $num_q);
            my ($avg_s, $per_s) =
                avg_recall_on_node($standby, $queries_ref, $gt, $k, $num_q);

            resume_fetcher($paused);

            my $delta = abs($avg_p - $avg_s);
            diag(sprintf(
                "step %d (%s) k=%d num_q=%d  recall_primary=%.4f  "
                ."recall_standby=%.4f  delta=%.4f  fetcher_paused=%s",
                $n, $s->{step_key}, $k, $num_q,
                $avg_p, $avg_s, $delta,
                (@$paused ? 'yes' : 'no')));

            cmp_ok($delta, '<=', $recall_tolerance,
                "step $n ($s->{step_key}): recall delta within tolerance "
                . "(p=$avg_p s=$avg_s tol=$recall_tolerance)");
            $verified++;
        }
        else {
            diag("step $n ($s->{step_key}): unknown op '$s->{op}', skipping");
        }
    }

    # A search-free step range would otherwise leave the subtest empty,
    # which Test::More treats as a failure.
    ok(1, "no search steps in range [$start_step,$end_step]; nothing to verify")
        if !$verified;
}

# ============================================================
# MAIN
# ============================================================

# Sanity-check inputs that are cheap to verify before bringing up a cluster.
BAIL_OUT("INDEX_TYPE must be 'hnsw' or 'ivfflat' (got: $index_type)")
    unless $index_type eq 'hnsw' || $index_type eq 'ivfflat';
BAIL_OUT("dataset not readable: $dataset_path") unless -r $dataset_path;
BAIL_OUT("queries not readable: $query_path")   unless -r $query_path;
BAIL_OUT("runbook not readable: $runbook_path") unless -r $runbook_path;
BAIL_OUT("gt_dir not a directory: $gt_dir")     unless -d $gt_dir;

# Dataset: open an on-demand reader (no full load — avoids a ~32 GB resident
# Perl process that would make every psql fork crawl). Only insert steps touch
# it, and only for their own range.
my $dreader = open_vec_reader($dataset_path);
my $dim     = $dreader->{dim};
diag("dataset reader: $dreader->{n} vectors, dim=$dim (on-demand, not loaded)");

# Queries: small enough to load fully (10k x dim).
diag("loading queries: $query_path");
my ($queries_ref, $qdim, $n_q) = load_dataset_file($query_path);
diag("queries loaded: $n_q queries, dim=$qdim");

BAIL_OUT("dataset dim ($dim) != query dim ($qdim)") if $dim != $qdim;

if ($source_pgdata && $start_step <= 1) {
    diag("WARNING: SOURCE_PGDATA is set but START_STEP=$start_step; the seed "
        ."already contains those steps. Set START_STEP=101 to skip them.");
}

my ($primary, $standby) = make_cluster();
# IF NOT EXISTS so this is a no-op when the seed already created the extension.
$primary->safe_psql('postgres', 'CREATE EXTENSION IF NOT EXISTS vector;');
wait_catchup($primary, $standby);

my $ext = $standby->safe_psql('postgres',
    "SELECT extversion FROM pg_extension WHERE extname = 'vector';");
isnt($ext, '', "vector extension visible on standby (got: $ext)");

# When reusing a seed, confirm the table came across and replicated.
if ($source_pgdata) {
    my $pc = $primary->safe_psql('postgres',
        "SELECT count(*) FROM vectors;");
    my $sc = $standby->safe_psql('postgres',
        "SELECT count(*) FROM vectors;");
    diag("seed 'vectors' rows: primary=$pc standby=$sc");
    is($sc, $pc, "seed 'vectors' rows replicated to standby ($pc rows)");
}

# Edge cases each DROP+recreate the vectors table, which would destroy a
# reused seed — skip them entirely when initializing from an existing dir.
# Otherwise skip only if every edge case is disabled (an empty subtest is
# itself a failure under Test::More).
if ($source_pgdata) {
    diag('edge cases skipped (reusing SOURCE_PGDATA; they would drop the seed)');
}
elsif (grep { $_ } values %enable_edge_case) {
    subtest 'edge cases' => sub {
        edge_case_empty_table($primary, $standby, $dim);
        edge_case_single_row($primary, $standby, $dim);
        edge_case_delete_all($primary, $standby, $dim);
        edge_case_reinsert_after_full_delete($primary, $standby, $dim);
    };
}
else {
    diag('all edge cases disabled; skipping edge-case subtest');
}

# Clean slate for the runbook walk — but ONLY for a true fresh start. When
# reusing a seed (or resuming mid-runbook), the table must be preserved.
my $fresh_start = (!$source_pgdata && $start_step <= 1);
if ($fresh_start) {
    $primary->safe_psql('postgres', 'DROP TABLE IF EXISTS vectors;');
    $primary->safe_psql('postgres', create_table_sql($dim));
    wait_catchup($primary, $standby);
}
else {
    diag("reusing existing 'vectors' table "
        ."(source=" . ($source_pgdata ? 'yes' : 'no')
        .", start_step=$start_step)");
}

subtest "runbook ($index_type)" => sub {
    run_runbook($primary, $standby, $dreader, $queries_ref);
};

done_testing();
