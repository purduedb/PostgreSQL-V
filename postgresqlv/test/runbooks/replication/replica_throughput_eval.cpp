// replica_throughput_eval.cpp — Experiment B: throughput vs. time.
//   --role primary : mixed-mode runbook on the primary; emits step_boundaries.csv
//   --role standby : continuous random queries on the standby; emits standby_throughput.csv
// All timestamps are primary-DB-clock ms (RTT/2-corrected) so the two CSVs,
// written on different hosts, share one timeline.
#include "runbook_common.hpp"
#include <libpq-fe.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <random>
#include <atomic>
#include <csignal>
#include <vector>
#include <algorithm>
#include <cstring>

using rbc::Step;

struct Args {
    std::string dataset, queries, runbook, dataset_name, table = "vectors";
    std::string p_host="localhost", p_port="5432", p_db="postgres", p_user="postgres", p_pass;
    std::string s_host="localhost", s_port="5433", s_db="postgres", s_user="postgres", s_pass;
    std::string role, index_type = "hnsw", out;
    int hnsw_ef_search = 200, ivfflat_probes = 10, threads = 16;
    long mixed_mode_start = 101, mixed_size = 3, checkpoint = 1000, dataset_offset = 0;
    long duration_sec = 0;
    bool assume_synced = false;
};

static std::atomic<bool> g_stop{false};
static void on_signal(int){ g_stop.store(true); }

static std::string conninfo(const std::string& h,const std::string& p,const std::string& db,
                            const std::string& u,const std::string& pw){
    std::string s="host="+h+" port="+p+" dbname="+db+" user="+u;
    if(!pw.empty()) s+=" password="+pw; return s;
}
static std::string set_search_param(const Args& a){
    return a.index_type=="ivfflat" ? "SET ivfflat.probes="+std::to_string(a.ivfflat_probes)
                                   : "SET hnsw.ef_search="+std::to_string(a.hnsw_ef_search);
}

static long long prime_offset(const Args& a){
    if (a.assume_synced) {
        // now_ms() is steady_clock (ms since an arbitrary epoch, e.g. boot), so
        // emitted = now_ms()+offset. To land on the Unix-epoch timeline (which is
        // what NTP keeps in sync across hosts), offset = system_epoch - steady_epoch.
        using namespace std::chrono;
        long long sys_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        return sys_ms - rbc::now_ms();
    }
    return rbc::prime_clock_offset(conninfo(a.p_host,a.p_port,a.p_db,a.p_user,a.p_pass));
}

// ---------------- role=standby ----------------
static int run_standby(const Args& a){
    long long offset = prime_offset(a);
    size_t qn=0,qdim=0;
    auto queries = rbc::DataLoader::load(a.queries, qn, qdim);
    rbc::ThreadConn::conninfo = conninfo(a.s_host,a.s_port,a.s_db,a.s_user,a.s_pass);

    std::vector<std::vector<long long>> buckets(a.threads);
    long long t_end = a.duration_sec>0 ? rbc::now_ms()+offset + a.duration_sec*1000 : 0;
    std::atomic<long> total{0};   // cumulative completed queries, for the heartbeat
    std::cerr << "[standby] querying with " << a.threads << " threads"
              << (a.duration_sec>0 ? " for "+std::to_string(a.duration_sec)+"s" : " (Ctrl-C to stop)")
              << "...\n";

    #pragma omp parallel num_threads(a.threads)
    {
        int tid = omp_get_thread_num();
        std::mt19937 gen(std::random_device{}() ^ (tid*2654435761u));
        std::uniform_int_distribution<size_t> pick(0, qn-1);
        PGconn* c = rbc::ThreadConn::get();
        { PGresult* r = PQexec(c, set_search_param(a).c_str()); PQclear(r); }
        long long hb_at = rbc::now_ms()+offset; long hb_count = 0;   // tid-0 heartbeat state
        while (!g_stop.load(std::memory_order_relaxed)) {
            if (t_end && (rbc::now_ms()+offset) >= t_end) break;
            size_t q = pick(gen);
            std::string sql = "SELECT id FROM "+a.table+" ORDER BY vec <-> "+
                              rbc::vector_to_sql(queries.data()+q*qdim, qdim)+" LIMIT 100";
            PGresult* r = PQexec(c, sql.c_str());
            bool ok = (PQresultStatus(r)==PGRES_TUPLES_OK);
            PQclear(r);
            if (ok) { buckets[tid].push_back(rbc::now_ms()+offset); total.fetch_add(1, std::memory_order_relaxed); }
            if (tid == 0) {
                long long now = rbc::now_ms()+offset;
                if (now - hb_at >= 5000) {
                    long c2 = total.load(std::memory_order_relaxed);
                    std::cerr << "[standby] " << c2 << " queries, ~"
                              << (long)((c2-hb_count)*1000.0/(now-hb_at)) << " q/s\n";
                    hb_at = now; hb_count = c2;
                }
            }
        }
    }
    std::vector<long long> all;
    for (auto& b : buckets) all.insert(all.end(), b.begin(), b.end());
    auto rows = rbc::bucket_qps(all);
    std::ofstream csv(a.out.empty()? "standby_throughput.csv" : a.out);
    csv << "timestamp_ms,qps\n";
    for (auto& kv : rows) csv << kv.first << "," << kv.second << "\n";
    std::cerr << "standby: "<<all.size()<<" queries over "<<rows.size()<<" seconds -> "
              << (a.out.empty()?"standby_throughput.csv":a.out) << "\n";
    return 0;
}

// ---------------- role=primary ----------------
static int run_primary(const Args& a){
    long long offset = prime_offset(a);
    // One VecReader per worker thread: a std::ifstream cannot be seek+read
    // concurrently, so each OpenMP thread gets its own handle to the dataset.
    std::vector<rbc::VecReader> readers(a.threads);
    for (int t = 0; t < a.threads; ++t) readers[t] = rbc::open_vec_reader(a.dataset);
    size_t qn=0,qdim=0;
    auto queries = rbc::DataLoader::load(a.queries, qn, qdim);
    auto steps = rbc::read_runbook(a.runbook, a.dataset_name);
    rbc::ThreadConn::conninfo = conninfo(a.p_host,a.p_port,a.p_db,a.p_user,a.p_pass);

    std::ofstream csv(a.out.empty()? "step_boundaries.csv" : a.out);
    csv << "step,timestamp_ms\n";

    // Walk in groups of mixed_size starting at mixed_mode_start.
    size_t i = 0;
    while (i < steps.size()) {
        if ((long)steps[i].step_num < a.mixed_mode_start) { ++i; continue; }
        // collect a group of up to mixed_size steps
        std::vector<Step> group;
        for (; i < steps.size() && (long)group.size() < a.mixed_size; ++i)
            group.push_back(steps[i]);
        if (group.empty()) break;

        csv << group.front().step_num << "," << (rbc::now_ms()+offset) << "\n";
        csv.flush();

        // Build a shuffled work array: one entry per op-item, value = index into group.
        std::vector<size_t> work;
        std::vector<long> item_count(group.size(), 0);
        for (size_t g = 0; g < group.size(); ++g) {
            long items = (group[g].op=="search") ? (long)qn : (group[g].end - group[g].start);
            item_count[g] = items;
            for (long t = 0; t < items; ++t) work.push_back(g);
        }
        std::shuffle(work.begin(), work.end(), std::mt19937(std::random_device{}()));
        std::vector<std::atomic<long>> next(group.size());
        for (auto& n : next) n.store(0);

        auto gstart = std::chrono::steady_clock::now();
        std::cerr << "[primary] step " << group.front().step_num << " mixed, "
                  << work.size() << " ops..." << std::flush;

        #pragma omp parallel for num_threads(a.threads) schedule(dynamic,100)
        for (size_t w = 0; w < work.size(); ++w) {
            int tid = omp_get_thread_num();
            size_t g = work[w];
            long item = next[g].fetch_add(1, std::memory_order_relaxed);
            if (item >= item_count[g]) continue;
            PGconn* c = rbc::ThreadConn::get();
            if (!c) continue;
            const Step& st = group[g];
            if (st.op == "insert") {
                long vid = st.start + item;
                auto v = rbc::read_vec(readers[tid], vid);
                std::string sql = "INSERT INTO "+a.table+" (id,vec) VALUES ("+
                    std::to_string(vid+a.dataset_offset)+",'"+
                    rbc::vector_to_sql(v.data(),v.size()).substr(1)+
                    ") ON CONFLICT (id) DO NOTHING";
                PGresult* r = PQexec(c, sql.c_str()); PQclear(r);
            } else if (st.op == "delete") {
                long vid = st.start + item;
                std::string sql = "DELETE FROM "+a.table+" WHERE id="+std::to_string(vid+a.dataset_offset);
                PGresult* r = PQexec(c, sql.c_str()); PQclear(r);
            } else if (st.op == "search") {
                { PGresult* r = PQexec(c, set_search_param(a).c_str()); PQclear(r); }
                std::string sql = "SELECT id FROM "+a.table+" ORDER BY vec <-> "+
                    rbc::vector_to_sql(queries.data()+item*qdim, qdim)+" LIMIT "+std::to_string(st.k);
                PGresult* r = PQexec(c, sql.c_str()); PQclear(r);
            }
        }
        std::cerr << " (" << std::fixed << std::setprecision(1)
                  << std::chrono::duration<double>(std::chrono::steady_clock::now()-gstart).count()
                  << "s)\n";
    }
    std::cerr << "primary: workload complete -> " << (a.out.empty()?"step_boundaries.csv":a.out) << "\n";
    return 0;
}

static void usage(const char* prog){
    std::cerr <<
      "Usage: " << prog << " --role primary|standby [options]\n"
      "  --dataset PATH --queries PATH --runbook PATH --dataset-name NAME (primary needs all; standby needs --queries)\n"
      "  --primary-host/-port/-db/-user/-pass   (both roles: clock priming; primary: workload)\n"
      "  --standby-host/-port/-db/-user/-pass   (standby role: query target)\n"
      "  --index-type hnsw|ivfflat --hnsw-ef-search N --ivfflat-probes N\n"
      "  --mixed-mode-start N (101) --mixed-size N (3) --threads N (16)\n"
      "  --checkpoint-size N (1000) --dataset-offset N (0) --duration-sec N (0=until SIGINT)\n"
      "  --assume-synced-clocks  --table-name NAME --out FILE\n";
}

int main(int argc, char** argv){
    Args a;
    auto need=[&](int& i){ if(i+1>=argc){usage(argv[0]);std::exit(2);} return std::string(argv[++i]); };
    for (int i=1;i<argc;++i){ std::string s=argv[i];
        if      (s=="--role") a.role=need(i);
        else if (s=="--dataset") a.dataset=need(i);
        else if (s=="--queries") a.queries=need(i);
        else if (s=="--runbook") a.runbook=need(i);
        else if (s=="--dataset-name") a.dataset_name=need(i);
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
        else if (s=="--mixed-mode-start") a.mixed_mode_start=std::stol(need(i));
        else if (s=="--mixed-size") a.mixed_size=std::stol(need(i));
        else if (s=="--threads") a.threads=std::stoi(need(i));
        else if (s=="--checkpoint-size") a.checkpoint=std::stol(need(i));
        else if (s=="--dataset-offset") a.dataset_offset=std::stol(need(i));
        else if (s=="--duration-sec") a.duration_sec=std::stol(need(i));
        else if (s=="--assume-synced-clocks") a.assume_synced=true;
        else if (s=="--table-name") a.table=need(i);
        else if (s=="--out") a.out=need(i);
        else { std::cerr<<"unknown arg: "<<s<<"\n"; usage(argv[0]); return 2; }
    }
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    try {
        if (a.role == "standby") {
            if (a.queries.empty()) { usage(argv[0]); return 2; }
            return run_standby(a);
        } else if (a.role == "primary") {
            if (a.dataset.empty()||a.queries.empty()||a.runbook.empty()||a.dataset_name.empty()){ usage(argv[0]); return 2; }
            return run_primary(a);
        }
        usage(argv[0]); return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
