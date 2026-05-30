// replica_recall_eval.cpp — Experiment A: recall vs. step, primary & standby.
// Sequential runbook walk on the primary; recall computed on both sides
// after the standby's WAL replay catches up. Output: experiment_a.csv.
#include "runbook_common.hpp"
#include <libpq-fe.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include <cstring>
#include <thread>
#include <chrono>
#include <cmath>

using rbc::Step; using rbc::Range;

struct Args {
    std::string dataset, queries, runbook, dataset_name, gt_dir, table = "vectors";
    std::string p_host="localhost", p_port="5432", p_db="postgres", p_user="postgres", p_pass;
    std::string s_host="localhost", s_port="5433", s_db="postgres", s_user="postgres", s_pass;
    std::string index_type = "hnsw", out = "experiment_a.csv";
    int hnsw_ef_search = 200, ivfflat_probes = 10;
    long start_step = 101, end_step = 400, build_index_before = 0;
    long num_verify = 10000, dataset_offset = 0, checkpoint = 1000;
    int threads = 8, catchup_timeout_sec = 300, settle_ms = 0;
    // index build params (used only if build_index_before > 0)
    int hnsw_m = 16, hnsw_ef_construction = 40, ivfflat_lists = 100;
};

static std::string conninfo(const std::string& h,const std::string& p,const std::string& db,
                            const std::string& u,const std::string& pw){
    std::string s = "host="+h+" port="+p+" dbname="+db+" user="+u;
    if (!pw.empty()) s += " password="+pw;
    return s;
}

static std::string set_search_param(const Args& a){
    return a.index_type=="ivfflat" ? "SET ivfflat.probes="+std::to_string(a.ivfflat_probes)
                                   : "SET hnsw.ef_search="+std::to_string(a.hnsw_ef_search);
}

// Wait until standby replay_lsn >= primary current LSN at call time.
static void wait_catchup(PGconn* primary, PGconn* standby, int timeout_sec){
    std::string target = rbc::scalar_query(primary, "SELECT pg_current_wal_lsn()");
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    for (;;) {
        std::string ok = rbc::scalar_query(standby,
            "SELECT (pg_last_wal_replay_lsn() >= '"+target+"'::pg_lsn)");
        if (ok == "t") return;
        if (std::chrono::steady_clock::now() > deadline)
            throw std::runtime_error("standby catchup timeout (target "+target+")");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Average recall@k over the first num_q queries, parallelized across threads.
// Connections are opened per call (one per worker thread) and closed at the end,
// so switching targets (primary vs. standby) across calls always uses the right
// host — we deliberately do NOT use a cached thread_local connection here.
static double avg_recall(const std::string& conninfo_str, const Args& a,
                         const std::vector<float>& queries, size_t dim,
                         const std::vector<std::vector<int32_t>>& gt, int k, long num_q){
    std::vector<double> per(num_q, 0.0);
    std::vector<PGconn*> conns(a.threads, nullptr);
    #pragma omp parallel for num_threads(a.threads) schedule(dynamic,1)
    for (long q = 0; q < num_q; ++q) {
        int tid = omp_get_thread_num();
        if (!conns[tid]) {
            PGconn* nc = PQconnectdb(conninfo_str.c_str());
            if (PQstatus(nc) != CONNECTION_OK) { PQfinish(nc); continue; }
            // Set search params once per connection (not per query).
            PGresult* sr = PQexec(nc, set_search_param(a).c_str());
            if (PQresultStatus(sr) != PGRES_COMMAND_OK)
                std::cerr << "[warn] SET search param failed: " << PQerrorMessage(nc) << "\n";
            PQclear(sr);
            conns[tid] = nc;
        }
        PGconn* c = conns[tid];
        std::string sql = "SELECT id FROM " + a.table + " ORDER BY vec <-> " +
                          rbc::vector_to_sql(queries.data()+q*dim, dim) +
                          " LIMIT " + std::to_string(k);
        PGresult* r = PQexec(c, sql.c_str());
        if (PQresultStatus(r) == PGRES_TUPLES_OK) {
            std::unordered_set<int32_t> g(gt[q].begin(), gt[q].begin()+std::min((size_t)k, gt[q].size()));
            int hits = 0, nr = PQntuples(r);
            for (int i = 0; i < nr; ++i)
                if (!PQgetisnull(r,i,0) && g.count((int32_t)std::atoi(PQgetvalue(r,i,0)))) ++hits;
            per[q] = (double)hits / (double)k;
        }
        PQclear(r);
    }
    for (PGconn* c : conns) if (c) PQfinish(c);
    double sum = 0; for (double x : per) sum += x;
    return num_q ? sum / (double)num_q : 0.0;
}

static void insert_range(PGconn* p, rbc::VecReader& dr, const Args& a, long s, long e){
    for (long b = s; b < e; b += a.checkpoint) {
        long end = std::min(b + a.checkpoint, e);
        std::string sql = "INSERT INTO " + a.table + " (id, vec) VALUES ";
        for (long i = b; i < end; ++i) {
            if (i > b) sql += ",";
            auto v = rbc::read_vec(dr, i);
            // vector_to_sql returns '[...]'; substr(1) drops its leading quote, the
            // literal ' before it restores it, and the ) closes the VALUES tuple.
            sql += "(" + std::to_string(i + a.dataset_offset) + ",'" +
                   rbc::vector_to_sql(v.data(), v.size()).substr(1) + ")";
        }
        sql += " ON CONFLICT (id) DO NOTHING;";
        PGresult* r = PQexec(p, sql.c_str());
        if (PQresultStatus(r) != PGRES_COMMAND_OK) {
            std::string msg = std::string("insert failed: ")+PQerrorMessage(p);
            PQclear(r); throw std::runtime_error(msg);
        }
        PQclear(r);
    }
}

static void delete_range(PGconn* p, const Args& a, long s, long e){
    std::string sql = "DELETE FROM " + a.table + " WHERE id >= " +
        std::to_string(s + a.dataset_offset) + " AND id < " + std::to_string(e + a.dataset_offset) + ";";
    PGresult* r = PQexec(p, sql.c_str());
    if (PQresultStatus(r) != PGRES_COMMAND_OK) {
        std::string msg = std::string("delete failed: ")+PQerrorMessage(p);
        PQclear(r); throw std::runtime_error(msg);
    }
    PQclear(r);
}

static void create_index(PGconn* p, const Args& a){
    std::string sql = a.index_type=="ivfflat"
        ? "CREATE INDEX ON "+a.table+" USING ivfflat (vec vector_l2_ops) WITH (lists="+std::to_string(a.ivfflat_lists)+");"
        : "CREATE INDEX ON "+a.table+" USING hnsw (vec vector_l2_ops) WITH (m="+std::to_string(a.hnsw_m)+
          ", ef_construction="+std::to_string(a.hnsw_ef_construction)+");";
    PGresult* r = PQexec(p, sql.c_str());
    if (PQresultStatus(r) != PGRES_COMMAND_OK) {
        std::string msg = std::string("create index failed: ")+PQerrorMessage(p);
        PQclear(r); throw std::runtime_error(msg);
    }
    PQclear(r);
}

static void usage(const char* prog){
    std::cerr <<
      "Usage: " << prog << " [options]\n"
      "  --dataset PATH --queries PATH --runbook PATH --dataset-name NAME --gt-dir DIR (required)\n"
      "  --primary-host H --primary-port P --primary-db D --primary-user U [--primary-pass PW]\n"
      "  --standby-host H --standby-port P --standby-db D --standby-user U [--standby-pass PW]\n"
      "  --index-type hnsw|ivfflat  --hnsw-ef-search N  --ivfflat-probes N\n"
      "  --start-step N (101)  --end-step N (400)  --build-index-before N (0=off)\n"
      "  --num-verify-queries N (10000)  --threads N (8)  --checkpoint-size N (1000)\n"
      "  --dataset-offset N (0)  --catchup-timeout-sec N (300)  --standby-settle-ms N (0)\n"
      "  --hnsw-m N --hnsw-ef-construction N --ivfflat-lists N  --table-name NAME  --out FILE\n";
}

int main(int argc, char** argv){
    Args a;
    auto need = [&](int& i){ if (i+1>=argc){ usage(argv[0]); std::exit(2);} return std::string(argv[++i]); };
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s=="--dataset") a.dataset=need(i);
        else if (s=="--queries") a.queries=need(i);
        else if (s=="--runbook") a.runbook=need(i);
        else if (s=="--dataset-name") a.dataset_name=need(i);
        else if (s=="--gt-dir") a.gt_dir=need(i);
        else if (s=="--primary-host") a.p_host=need(i);
        else if (s=="--primary-port") a.p_port=need(i);
        else if (s=="--primary-db") a.p_db=need(i);
        else if (s=="--primary-user") a.p_user=need(i);
        else if (s=="--primary-pass") a.p_pass=need(i);
        else if (s=="--standby-host") a.s_host=need(i);
        else if (s=="--standby-port") a.s_port=need(i);
        else if (s=="--standby-db") a.s_db=need(i);
        else if (s=="--standby-user") a.s_user=need(i);
        else if (s=="--standby-pass") a.s_pass=need(i);
        else if (s=="--index-type") a.index_type=need(i);
        else if (s=="--hnsw-ef-search") a.hnsw_ef_search=std::stoi(need(i));
        else if (s=="--ivfflat-probes") a.ivfflat_probes=std::stoi(need(i));
        else if (s=="--start-step") a.start_step=std::stol(need(i));
        else if (s=="--end-step") a.end_step=std::stol(need(i));
        else if (s=="--build-index-before") a.build_index_before=std::stol(need(i));
        else if (s=="--num-verify-queries") a.num_verify=std::stol(need(i));
        else if (s=="--threads") a.threads=std::stoi(need(i));
        else if (s=="--checkpoint-size") a.checkpoint=std::stol(need(i));
        else if (s=="--dataset-offset") a.dataset_offset=std::stol(need(i));
        else if (s=="--catchup-timeout-sec") a.catchup_timeout_sec=std::stoi(need(i));
        else if (s=="--standby-settle-ms") a.settle_ms=std::stoi(need(i));
        else if (s=="--hnsw-m") a.hnsw_m=std::stoi(need(i));
        else if (s=="--hnsw-ef-construction") a.hnsw_ef_construction=std::stoi(need(i));
        else if (s=="--ivfflat-lists") a.ivfflat_lists=std::stoi(need(i));
        else if (s=="--table-name") a.table=need(i);
        else if (s=="--out") a.out=need(i);
        else { std::cerr << "unknown arg: " << s << "\n"; usage(argv[0]); return 2; }
    }
    if (a.dataset.empty()||a.queries.empty()||a.runbook.empty()||a.dataset_name.empty()||a.gt_dir.empty()){
        usage(argv[0]); return 2;
    }

    std::string runbook_id = a.dataset_name;
    for (char& c : runbook_id) if (c=='-') c='_';

    auto dr = rbc::open_vec_reader(a.dataset);
    size_t qn=0, qdim=0;
    auto queries = rbc::DataLoader::load(a.queries, qn, qdim);
    if (qdim != dr.dim) { std::cerr << "dim mismatch\n"; return 1; }
    auto steps = rbc::read_runbook(a.runbook, a.dataset_name);

    std::string pconn = conninfo(a.p_host,a.p_port,a.p_db,a.p_user,a.p_pass);
    std::string sconn = conninfo(a.s_host,a.s_port,a.s_db,a.s_user,a.s_pass);
    PGconn* primary = rbc::connect_or_die(pconn);
    PGconn* standby = rbc::connect_or_die(sconn);

    std::ofstream csv(a.out);
    csv << "step,primary_recall,standby_recall\n";

    long num_q = std::min<long>(a.num_verify, (long)qn);
    bool index_built = false;
    std::vector<Range> ranges = rbc::ranges_up_to(steps, a.start_step);

    // Progress: count steps in range for an [done/total] counter; time each step.
    long total = 0, done = 0;
    for (const auto& st : steps)
        if ((long)st.step_num >= a.start_step && (a.end_step<=0 || (long)st.step_num<=a.end_step)) ++total;

    for (const auto& st : steps) {
        if ((long)st.step_num < a.start_step) continue;
        if (a.end_step > 0 && (long)st.step_num > a.end_step) break;
        ++done;
        auto t0 = std::chrono::steady_clock::now();
        auto secs = [&]{ return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count(); };
        std::cerr << "[" << done << "/" << total << "] step " << st.step_num << " " << st.op;

        if (!index_built && a.build_index_before>0 && (long)st.step_num==a.build_index_before){
            std::cerr << " | building index..." << std::flush;
            create_index(primary, a); index_built = true;
            wait_catchup(primary, standby, a.catchup_timeout_sec);
        }

        if (st.op == "insert") {
            std::cerr << " [" << st.start << "," << st.end << ")" << std::flush;
            insert_range(primary, dr, a, st.start, st.end);
            ranges.push_back({"insert", st.start, st.end});
            std::cerr << " (" << std::fixed << std::setprecision(1) << secs() << "s)\n";
        } else if (st.op == "delete") {
            std::cerr << " [" << st.start << "," << st.end << ")" << std::flush;
            delete_range(primary, a, st.start, st.end);
            ranges.push_back({"delete", st.start, st.end});
            std::cerr << " (" << std::fixed << std::setprecision(1) << secs() << "s)\n";
        } else if (st.op == "search") {
            std::cerr << " k=" << st.k << " | catchup..." << std::flush;
            wait_catchup(primary, standby, a.catchup_timeout_sec);
            if (a.settle_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(a.settle_ms));
            std::string gtf = a.gt_dir + "/" + rbc::gt_filename(runbook_id, st.step_num, ranges);
            std::ifstream probe(gtf);
            if (!probe.good()) {
                std::cerr << " MISSING GT " << gtf << "\n";
                csv << st.step_num << ",NaN,NaN\n"; csv.flush(); continue;
            }
            auto gt = rbc::load_gt_npy(gtf, num_q, st.k);
            double rp = avg_recall(pconn, a, queries, qdim, gt, st.k, num_q);
            double rs = avg_recall(sconn, a, queries, qdim, gt, st.k, num_q);
            csv << st.step_num << "," << std::fixed << std::setprecision(6) << rp << "," << rs << "\n";
            csv.flush();
            std::cerr << std::fixed << std::setprecision(4)
                      << " p=" << rp << " s=" << rs << " d=" << std::fabs(rp-rs)
                      << " (" << std::setprecision(1) << secs() << "s)\n";
        }
    }
    PQfinish(primary); PQfinish(standby);
    std::cerr << "done -> " << a.out << "\n";
    return 0;
}

// ./replica_recall_eval \
// --dataset /ssd_root/dataset/turing10m/msturing-10M.fvecs \
// --queries /ssd_root/dataset/turing10m/msturing-query.fvecs \
// --runbook ../msturing-10M_slidingwindow_runbook.yaml \
// --dataset-name msturing-10M --gt-dir /ssd_root/liu4127/msturing_runbook_gt \
// --primary-host 127.0.0.1 --primary-port 5434 --primary-db postgres --primary-user liu4127 \
// --standby-host 10.145.21.39 --standby-port 5434 --standby-db postgres --standby-user liu4127 \
// --start-step 101 --build-index-before 101 \
// --index-type hnsw --hnsw-ef-search 200 --hnsw-m 16 --hnsw-ef-construction 40 \
// --table-name vectors --out experiment_a.csv --threads 8