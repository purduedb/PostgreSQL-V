#ifndef RUNBOOK_COMMON_HPP
#define RUNBOOK_COMMON_HPP
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <openssl/md5.h>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cctype>
#include <libpq-fe.h>
#include <chrono>
#include <map>
#include <thread>
#include <mutex>
namespace rbc {

inline std::string vector_to_sql(const float* vec, size_t dim) {
    std::ostringstream ss;
    ss << "'[";
    for (size_t i = 0; i < dim; ++i) {
        if (i) ss << ",";
        ss << std::fixed << std::setprecision(6) << vec[i];
    }
    ss << "]'";
    return ss.str();
}

struct DataLoader {
    static std::vector<float> load_fvecs(const std::string& path, size_t& n, size_t& dim) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + path);
        int32_t d; f.read(reinterpret_cast<char*>(&d), 4);
        dim = (size_t)d;
        f.seekg(0, std::ios::end);
        size_t bytes = (size_t)f.tellg();
        n = bytes / ((dim + 1) * sizeof(float));
        f.seekg(0, std::ios::beg);
        std::vector<float> data(n * dim);
        for (size_t i = 0; i < n; ++i) {
            f.read(reinterpret_cast<char*>(&d), 4);
            f.read(reinterpret_cast<char*>(&data[i * dim]), dim * sizeof(float));
        }
        return data;
    }
    static std::vector<float> load_fbin(const std::string& path, size_t& n, size_t& dim) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + path);
        int32_t nn, d;
        f.read(reinterpret_cast<char*>(&nn), 4);
        f.read(reinterpret_cast<char*>(&d), 4);
        if (nn <= 0 || d <= 0) throw std::runtime_error("bad fbin header: " + path);
        n = (size_t)nn; dim = (size_t)d;
        std::vector<float> data(n * dim);
        f.read(reinterpret_cast<char*>(data.data()), n * dim * sizeof(float));
        return data;
    }
    static std::vector<float> load(const std::string& path, size_t& n, size_t& dim) {
        auto p = path.rfind('.');
        if (p == std::string::npos) throw std::runtime_error("no extension: " + path);
        std::string ext = path.substr(p + 1);
        if (ext == "fbin")  return load_fbin(path, n, dim);
        if (ext == "fvecs") return load_fvecs(path, n, dim);
        throw std::runtime_error("unknown extension: " + path);
    }
};

struct VecReader {
    std::ifstream fh;
    size_t dim = 0, n = 0;
    std::string fmt;   // "fvecs" | "fbin"
};

inline VecReader open_vec_reader(const std::string& path) {
    VecReader r;
    r.fh.open(path, std::ios::binary);
    if (!r.fh) throw std::runtime_error("cannot open " + path);
    auto p = path.rfind('.');
    std::string ext = (p == std::string::npos) ? "" : path.substr(p + 1);
    if (ext == "fbin") {
        int32_t nn, d;
        r.fh.read(reinterpret_cast<char*>(&nn), 4);
        r.fh.read(reinterpret_cast<char*>(&d), 4);
        if (nn <= 0 || d <= 0) throw std::runtime_error("bad fbin header: " + path);
        r.n = (size_t)nn; r.dim = (size_t)d; r.fmt = "fbin";
    } else if (ext == "fvecs") {
        int32_t d; r.fh.read(reinterpret_cast<char*>(&d), 4);
        r.dim = (size_t)d;
        r.fh.seekg(0, std::ios::end);
        size_t bytes = (size_t)r.fh.tellg();
        r.n = bytes / ((r.dim + 1) * sizeof(float));
        r.fmt = "fvecs";
    } else {
        throw std::runtime_error("unknown extension: " + path);
    }
    return r;
}

inline std::vector<float> read_vec(VecReader& r, size_t i) {
    size_t off = (r.fmt == "fvecs") ? i * (4 + 4 * r.dim) + 4
                                    : 8 + i * 4 * r.dim;
    r.fh.clear();
    r.fh.seekg((std::streamoff)off, std::ios::beg);
    std::vector<float> v(r.dim);
    r.fh.read(reinterpret_cast<char*>(v.data()), r.dim * sizeof(float));
    return v;
}

struct Range { std::string op; long start; long end; };

inline std::string gt_filename(const std::string& runbook_id, size_t step,
                               const std::vector<Range>& ranges) {
    std::ostringstream rs;
    rs << "[";
    for (size_t i = 0; i < ranges.size(); ++i) {
        if (i) rs << ",";
        rs << "[\"" << ranges[i].op << "\"," << ranges[i].start << "," << ranges[i].end << "]";
    }
    rs << "]";
    std::string s = rs.str();
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5(reinterpret_cast<const unsigned char*>(s.data()), s.size(), digest);
    std::ostringstream hx;
    for (int i = 0; i < 4; ++i)
        hx << std::hex << std::setw(2) << std::setfill('0') << (int)digest[i];
    std::ostringstream fn;
    fn << runbook_id << "_step" << step << "_ranges" << hx.str() << "_gt.npy";
    return fn.str();
}

inline std::vector<std::vector<int32_t>> load_gt_npy(const std::string& path,
                                                     size_t num_queries, size_t k) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open GT " + path);
    char buf[6]; f.read(buf, 6); f.read(buf, 2);   // magic, version
    uint16_t hlen; f.read(reinterpret_cast<char*>(&hlen), 2);
    f.seekg(hlen, std::ios::cur);
    // GT payload is int32; use int32_t storage so the read byte-count matches
    // the element size on every platform.
    std::vector<std::vector<int32_t>> gt(num_queries, std::vector<int32_t>(k));
    for (size_t i = 0; i < num_queries; ++i)
        f.read(reinterpret_cast<char*>(gt[i].data()), (std::streamsize)(k * sizeof(int32_t)));
    if (!f) throw std::runtime_error("truncated GT " + path);
    return gt;
}

struct Step {
    size_t step_num;       // 1-based, over all non-reserved keys in sorted order
    std::string step_key;  // original YAML key
    std::string op;        // "insert" | "delete" | "search"
    long start = 0, end = 0;
    int  k = 100;
};

// Extract the first integer embedded in a key ("101" -> 101, "step101" -> 101).
inline long key_int(const std::string& s) {
    std::string d; for (char c : s) if (std::isdigit((unsigned char)c)) d += c;
    return d.empty() ? 0 : std::stol(d);
}

inline std::vector<Step> read_runbook(const std::string& path, const std::string& dataset_name) {
    YAML::Node root = YAML::LoadFile(path);
    YAML::Node sec = root[dataset_name];
    if (!sec) throw std::runtime_error("dataset '" + dataset_name + "' not in runbook");
    std::vector<std::string> keys;
    for (auto it = sec.begin(); it != sec.end(); ++it) {
        std::string key = it->first.as<std::string>();
        if (key == "max_pts" || key == "query" || key == "groundtruth") continue;
        keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end(),
              [](const std::string& a, const std::string& b){ return key_int(a) < key_int(b); });
    std::vector<Step> out;
    size_t n = 0;
    for (const auto& key : keys) {
        ++n;
        YAML::Node node = sec[key];
        if (!node["operation"]) continue;
        Step s;
        s.step_num = n; s.step_key = key;
        s.op = node["operation"].as<std::string>();
        if (node["start"]) s.start = node["start"].as<long>();
        if (node["end"])   s.end   = node["end"].as<long>();
        if (node["k"])     s.k     = node["k"].as<int>();
        out.push_back(s);
    }
    return out;
}

// Active-ranges prefix: all insert/delete steps with step_num < upto_step.
inline std::vector<Range> ranges_up_to(const std::vector<Step>& steps, size_t upto_step) {
    std::vector<Range> r;
    for (const auto& s : steps) {
        if (s.step_num >= upto_step) break;
        if (s.op == "insert" || s.op == "delete") r.push_back({s.op, s.start, s.end});
    }
    return r;
}

// ---- monotonic local clock in ms ----
// Uses steady_clock (immune to NTP steps) so RTT brackets in prime_clock_offset
// are never distorted. The steady epoch is arbitrary, but it cancels out:
// every emitted timestamp is now_ms()+offset, and offset = primary_ms-(now_ms@sample),
// so emitted values land on the primary's Unix-epoch-ms timeline.
inline long long now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

// ---- clock offset estimation (pure, testable) ----
struct ClockSample { long long t0; long long primary_ms; long long t1; };

inline long long offset_from_samples(const std::vector<ClockSample>& samples) {
    long long best_rtt = -1, offset = 0;
    for (const auto& s : samples) {
        long long rtt = s.t1 - s.t0;
        if (best_rtt < 0 || rtt < best_rtt) {
            best_rtt = rtt;
            offset = s.primary_ms - (s.t0 + rtt / 2);
        }
    }
    return offset;
}

// ---- per-second QPS bucketing (pure, testable) ----
inline std::vector<std::pair<long long,long long>> bucket_qps(const std::vector<long long>& completions_ms) {
    std::map<long long,long long> m;
    for (long long t : completions_ms) m[(t / 1000) * 1000]++;
    return std::vector<std::pair<long long,long long>>(m.begin(), m.end());
}

// ---- libpq helpers ----
inline PGconn* connect_or_die(const std::string& conninfo) {
    PGconn* c = PQconnectdb(conninfo.c_str());
    if (PQstatus(c) != CONNECTION_OK) {
        // PQconnectdb returns a non-null handle even on failure; free it.
        std::string msg = std::string("connect failed: ") + PQerrorMessage(c);
        PQfinish(c);
        throw std::runtime_error(msg);
    }
    return c;
}

inline std::string scalar_query(PGconn* c, const std::string& sql) {
    PGresult* r = PQexec(c, sql.c_str());
    std::string out;
    if (PQresultStatus(r) == PGRES_TUPLES_OK && PQntuples(r) > 0 && !PQgetisnull(r, 0, 0))
        out = PQgetvalue(r, 0, 0);
    PQclear(r);
    return out;
}

// Prime offset by sampling the PRIMARY server clock K times (smallest-RTT wins).
inline long long prime_clock_offset(const std::string& primary_conninfo, int k = 5) {
    PGconn* c = connect_or_die(primary_conninfo);
    std::vector<ClockSample> samples;
    for (int i = 0; i < k; ++i) {
        long long t0 = now_ms();
        std::string p = scalar_query(c, "SELECT (extract(epoch from clock_timestamp())*1000)::bigint");
        long long t1 = now_ms();
        if (!p.empty()) samples.push_back({t0, std::stoll(p), t1});
    }
    PQfinish(c);
    if (samples.empty()) throw std::runtime_error("clock priming: no samples from primary");
    return offset_from_samples(samples);
}

// Per-thread libpq connection (one conninfo set at startup).
struct ThreadConn {
    static inline std::string conninfo;   // C++17 inline static — no separate definition needed
    static PGconn* get() {
        thread_local PGconn* c = nullptr;
        if (!c) c = connect_or_die(conninfo);
        return c;
    }
};

}  // namespace rbc
#endif
