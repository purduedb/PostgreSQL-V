# Replica Search Performance Evaluation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two C++ drivers (plus a shared header, a Makefile, and a guidance doc) that measure read-only-standby search behavior of the physical-replication path on the msturing-10M sliding-window runbook — Experiment A (recall vs. step) and Experiment B (throughput vs. time).

**Architecture:** A self-contained header `runbook_common.hpp` holds pure, unit-tested helpers (dataset/GT/YAML/vector-SQL/connection/clock). Two standalone programs build on it: `replica_recall_eval` (sequential walk on the primary, recall on both sides) and `replica_throughput_eval --role {primary,standby}` (mixed-mode load on the primary while the standby is queried continuously). The two drivers run on different hosts and share one timeline by expressing every timestamp in primary-DB-clock ms (RTT/2-corrected). The extension C/C++ source is not touched.

**Tech Stack:** C++17, libpq, yaml-cpp, OpenSSL MD5, OpenMP, pthreads. Build with `g++` (same flags as `pgvector_test.cpp`), not PGXS.

---

## Conventions for this plan

- **Git commits are the user's responsibility.** Per project convention the implementer must NOT run `git add`/`git commit`. Each task ends with a **Checkpoint** (stop for review) instead of a commit.
- **Live-DB behavior cannot be tested here** (the operator owns the PostgreSQL lifecycle; no `pg_ctl`/`initdb` from scripts). Verification therefore relies on: (a) runnable unit tests for the pure helpers in `runbook_common.hpp`, (b) clean compilation + argument-parsing smoke tests for the two drivers, and (c) operator run-instructions captured in the guidance doc. Tasks 1–6 are fully test-runnable now; Tasks 7–8 verify by build + usage smoke.
- **Paths.** Work happens in `pgvector/test/runbooks/replication/`. The reference monolith to port from is `pgvector/test/runbooks/pgvector_test.cpp` (the implementer should open it).
- **Reference constants** (verified): dataset/query dim = 96; runbook `msturing-10M_slidingwindow_runbook.yaml` = 400 ops (steps 1–100 insert, then `[search,delete,insert]` ×100); GT dir `/ssd_root/liu4127/msturing_runbook_gt`; N=10000 queries, k=100; `runbook_id` = dataset name with `-`→`_` (`msturing-10M` → `msturing_10M`).

---

## File Structure

```
pgvector/test/runbooks/replication/
  runbook_common.hpp           # Tasks 1-6: shared helpers (pure + libpq + clock)
  test_runbook_common.cpp      # Tasks 1-6: assert-based unit tests
  replica_recall_eval.cpp      # Task 7: Experiment A
  replica_throughput_eval.cpp  # Task 8: Experiment B (--role primary|standby)
  Makefile                     # Tasks 1,7,8: build test + both drivers
docs/
  replica_search_evaluation.md # Task 9: configure + conduct guidance
```

Responsibilities: `runbook_common.hpp` is the single source of shared logic (one header, included by all three `.cpp`). Each driver is one focused program. Tests live in one file that links only the header (no DB).

---

## Task 1: Build skeleton + `vector_to_sql`

**Files:**
- Create: `pgvector/test/runbooks/replication/runbook_common.hpp`
- Create: `pgvector/test/runbooks/replication/test_runbook_common.cpp`
- Create: `pgvector/test/runbooks/replication/Makefile`

- [ ] **Step 1: Write the failing test**

Create `test_runbook_common.cpp` with a minimal assert framework plus the first test:

```cpp
// test_runbook_common.cpp — unit tests for runbook_common.hpp (no DB needed)
#include "runbook_common.hpp"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include <functional>

static int g_checks = 0, g_failures = 0;
#define CHECK(cond) do { ++g_checks; if(!(cond)) { ++g_failures; \
    std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " << #cond << "\n"; } } while(0)
#define CHECK_STR_EQ(a,b) do { ++g_checks; std::string _a=(a), _b=(b); if(_a!=_b){ ++g_failures; \
    std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": [" << _a << "] != [" << _b << "]\n"; } } while(0)

static std::vector<std::pair<std::string,std::function<void()>>> g_tests;
struct Reg { Reg(const char* n, std::function<void()> f){ g_tests.push_back({n,f}); } };
#define TEST(name) static void name(); static Reg reg_##name(#name, name); static void name()

TEST(test_vector_to_sql) {
    std::vector<float> v = {1.0f, 2.5f, -3.0f};
    CHECK_STR_EQ(rbc::vector_to_sql(v.data(), v.size()), "'[1.000000,2.500000,-3.000000]'");
}

int main() {
    for (auto& t : g_tests) { std::cout << "RUN  " << t.first << "\n"; t.second(); }
    std::cout << (g_failures ? "\nFAILED " : "\nPASSED ")
              << (g_checks - g_failures) << "/" << g_checks << " checks\n";
    return g_failures ? 1 : 0;
}
```

Create `Makefile`:

```make
# Build flags mirror pgvector_test.cpp.
PG_INCLUDE ?= $(shell pg_config --includedir)
CXX        ?= g++
CXXFLAGS   ?= -O3 -std=c++17 -mavx2 -mfma -fopenmp -Wall
INCLUDES    = -I$(PG_INCLUDE)
LIBS        = -lpq -lyaml-cpp -lcrypto -pthread

.PHONY: all test clean
all: replica_recall_eval replica_throughput_eval

test: test_runbook_common
	./test_runbook_common

test_runbook_common: test_runbook_common.cpp runbook_common.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) test_runbook_common.cpp -o $@ $(LIBS)

replica_recall_eval: replica_recall_eval.cpp runbook_common.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) replica_recall_eval.cpp -o $@ $(LIBS)

replica_throughput_eval: replica_throughput_eval.cpp runbook_common.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) replica_throughput_eval.cpp -o $@ $(LIBS)

clean:
	rm -f test_runbook_common replica_recall_eval replica_throughput_eval
```

Create `runbook_common.hpp` with only the header guard + includes (no `vector_to_sql` yet, so the test fails to compile):

```cpp
#ifndef RUNBOOK_COMMON_HPP
#define RUNBOOK_COMMON_HPP
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cstdint>
namespace rbc {
}  // namespace rbc
#endif
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd pgvector/test/runbooks/replication && make test`
Expected: compile error — `'vector_to_sql' is not a member of 'rbc'`.

- [ ] **Step 3: Write minimal implementation**

Add inside `namespace rbc` in `runbook_common.hpp`:

```cpp
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make test`
Expected: `PASSED 1/1 checks`, exit 0.

- [ ] **Step 5: Checkpoint** — stop for review (no git commit; user handles git).

---

## Task 2: `DataLoader` (fvecs/fbin) + seekable reader

**Files:**
- Modify: `pgvector/test/runbooks/replication/runbook_common.hpp`
- Modify: `pgvector/test/runbooks/replication/test_runbook_common.cpp`

- [ ] **Step 1: Write the failing tests**

Add to `test_runbook_common.cpp`:

```cpp
#include <fstream>
#include <cstdio>

// Write a tiny .fvecs: 2 vectors of dim 3.
static std::string write_tmp_fvecs() {
    std::string path = "/tmp/rbc_test.fvecs";
    std::ofstream f(path, std::ios::binary);
    int32_t dim = 3;
    float rows[2][3] = {{1,2,3},{4,5,6}};
    for (int i = 0; i < 2; ++i) {
        f.write(reinterpret_cast<const char*>(&dim), 4);
        f.write(reinterpret_cast<const char*>(rows[i]), 12);
    }
    return path;
}

// Write a tiny .fbin: 2 vectors of dim 3.
static std::string write_tmp_fbin() {
    std::string path = "/tmp/rbc_test.fbin";
    std::ofstream f(path, std::ios::binary);
    int32_t n = 2, dim = 3;
    float rows[2][3] = {{7,8,9},{10,11,12}};
    f.write(reinterpret_cast<const char*>(&n), 4);
    f.write(reinterpret_cast<const char*>(&dim), 4);
    f.write(reinterpret_cast<const char*>(rows), 24);
    return path;
}

TEST(test_load_fvecs) {
    size_t n=0, dim=0;
    auto v = rbc::DataLoader::load(write_tmp_fvecs(), n, dim);
    CHECK(n == 2); CHECK(dim == 3);
    CHECK(v[0]==1 && v[1]==2 && v[2]==3 && v[3]==4 && v[4]==5 && v[5]==6);
}

TEST(test_load_fbin) {
    size_t n=0, dim=0;
    auto v = rbc::DataLoader::load(write_tmp_fbin(), n, dim);
    CHECK(n == 2); CHECK(dim == 3);
    CHECK(v[0]==7 && v[5]==12);
}

TEST(test_seekable_reader) {
    auto r = rbc::open_vec_reader(write_tmp_fvecs());
    CHECK(r.dim == 3); CHECK(r.n == 2);
    auto v1 = rbc::read_vec(r, 1);   // second vector
    CHECK(v1.size()==3 && v1[0]==4 && v1[1]==5 && v1[2]==6);
    auto r2 = rbc::open_vec_reader(write_tmp_fbin());
    auto v0 = rbc::read_vec(r2, 0);
    CHECK(v0[0]==7 && v0[2]==9);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test`
Expected: compile error — `DataLoader` / `open_vec_reader` / `read_vec` not in `rbc`.

- [ ] **Step 3: Write minimal implementation**

Add `#include <fstream>` and `#include <stdexcept>` to `runbook_common.hpp`, and inside `namespace rbc`:

```cpp
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make test`
Expected: `PASSED` with all checks passing.

- [ ] **Step 5: Checkpoint** — stop for review.

---

## Task 3: GT `.npy` loader

**Files:**
- Modify: `runbook_common.hpp`
- Modify: `test_runbook_common.cpp`

- [ ] **Step 1: Write the failing test**

Add to `test_runbook_common.cpp`:

```cpp
// Write a synthetic npy matching compute_ground_truth.cpp's format:
// 6-byte magic, 2-byte version, uint16 header len, header bytes, then n*k int32.
static std::string write_tmp_npy(int n, int k) {
    std::string path = "/tmp/rbc_test_gt.npy";
    std::ofstream f(path, std::ios::binary);
    const char magic[6] = {(char)0x93,'N','U','M','P','Y'};
    f.write(magic, 6);
    char ver[2] = {1, 0}; f.write(ver, 2);
    std::string hdr = "{'descr':'<i4','fortran_order':False,'shape':(" +
                      std::to_string(n) + "," + std::to_string(k) + "),} ";
    while ((10 + hdr.size()) % 64 != 0) hdr += ' ';
    hdr.back() = '\n';
    uint16_t hlen = (uint16_t)hdr.size();
    f.write(reinterpret_cast<const char*>(&hlen), 2);
    f.write(hdr.data(), hlen);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < k; ++j) { int32_t id = i * 1000 + j; f.write(reinterpret_cast<const char*>(&id), 4); }
    return path;
}

TEST(test_load_gt_npy_synthetic) {
    auto gt = rbc::load_gt_npy(write_tmp_npy(5, 4), 5, 4);
    CHECK(gt.size() == 5);
    CHECK(gt[0].size() == 4);
    CHECK(gt[0][0] == 0 && gt[0][3] == 3);
    CHECK(gt[2][1] == 2001);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test`
Expected: compile error — `load_gt_npy` not in `rbc`.

- [ ] **Step 3: Write minimal implementation**

Add inside `namespace rbc`:

```cpp
inline std::vector<std::vector<int>> load_gt_npy(const std::string& path,
                                                 size_t num_queries, size_t k) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open GT " + path);
    char buf[6]; f.read(buf, 6); f.read(buf, 2);   // magic, version
    uint16_t hlen; f.read(reinterpret_cast<char*>(&hlen), 2);
    f.seekg(hlen, std::ios::cur);
    std::vector<std::vector<int>> gt(num_queries, std::vector<int>(k));
    for (size_t i = 0; i < num_queries; ++i)
        f.read(reinterpret_cast<char*>(gt[i].data()), (std::streamsize)(k * sizeof(int32_t)));
    if (!f) throw std::runtime_error("truncated GT " + path);
    return gt;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make test`
Expected: `PASSED`.

- [ ] **Step 5: Checkpoint** — stop for review.

---

## Task 4: `gt_filename` (MD5 over active-ranges JSON)

**Files:**
- Modify: `runbook_common.hpp`
- Modify: `test_runbook_common.cpp`

- [ ] **Step 1: Write the failing test**

The active-ranges JSON and MD5-first-4-bytes scheme must match `compute_ground_truth.cpp` byte-for-byte. Expected values were computed independently:
- `[]` → `d7517139`
- `[["insert",0,50000]]` → `80dd090e`

Add to `test_runbook_common.cpp`:

```cpp
TEST(test_gt_filename_empty) {
    std::vector<rbc::Range> r;
    CHECK_STR_EQ(rbc::gt_filename("msturing_10M", 7, r),
                 "msturing_10M_step7_rangesd7517139_gt.npy");
}
TEST(test_gt_filename_single) {
    std::vector<rbc::Range> r = {{"insert", 0, 50000}};
    CHECK_STR_EQ(rbc::gt_filename("msturing_10M", 101, r),
                 "msturing_10M_step101_ranges80dd090e_gt.npy");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test`
Expected: compile error — `rbc::Range` and `rbc::gt_filename` undefined.

- [ ] **Step 3: Write minimal implementation**

Add `#include <openssl/md5.h>` to `runbook_common.hpp`, and inside `namespace rbc`:

```cpp
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make test`
Expected: `PASSED`.

- [ ] **Step 5: Checkpoint** — stop for review.

---

## Task 5: Runbook YAML reader + ordered steps + ranges-up-to (integration test)

**Files:**
- Modify: `runbook_common.hpp`
- Modify: `test_runbook_common.cpp`

- [ ] **Step 1: Write the failing test**

This test ties the whole GT path together against the real runbook + real GT filename (`e4c4025c` for step 101's 100-insert prefix). It is skipped (not failed) if the runbook file is absent, so the suite stays portable.

Add to `test_runbook_common.cpp`:

```cpp
TEST(test_runbook_reader_real) {
    const char* rb = "../msturing-10M_slidingwindow_runbook.yaml";
    std::ifstream probe(rb);
    if (!probe.good()) { std::cout << "  (skip: runbook not found)\n"; return; }
    auto steps = rbc::read_runbook(rb, "msturing-10M");
    CHECK(steps.size() == 400);
    CHECK(steps[0].op == "insert" && steps[0].start == 0 && steps[0].end == 50000);
    CHECK(steps[0].step_num == 1);
    CHECK(steps[100].op == "search" && steps[100].step_num == 101);   // 0-based index 100

    auto ranges = rbc::ranges_up_to(steps, 101);   // inserts of steps 1..100
    CHECK(ranges.size() == 100);
    CHECK(ranges[0].op == "insert" && ranges[0].start == 0 && ranges[0].end == 50000);
    CHECK(ranges[99].end == 5000000);

    CHECK_STR_EQ(rbc::gt_filename("msturing_10M", 101, ranges),
                 "msturing_10M_step101_rangese4c4025c_gt.npy");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test`
Expected: compile error — `rbc::Step`, `rbc::read_runbook`, `rbc::ranges_up_to` undefined.

- [ ] **Step 3: Write minimal implementation**

Add `#include <yaml-cpp/yaml.h>`, `#include <algorithm>`, `#include <cctype>` to `runbook_common.hpp`, and inside `namespace rbc`:

```cpp
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make test`
Expected: `PASSED` (or the skip line if the runbook path differs — in that case run from the `replication/` dir so `../msturing-10M_slidingwindow_runbook.yaml` resolves).

- [ ] **Step 5: Checkpoint** — stop for review.

---

## Task 6: Connections + clock module + QPS bucketing

**Files:**
- Modify: `runbook_common.hpp`
- Modify: `test_runbook_common.cpp`

- [ ] **Step 1: Write the failing tests**

Two pure functions are unit-testable without a DB: the clock-offset estimator (pick the sample with the smallest RTT, correct by RTT/2) and the per-second QPS bucketer. Add to `test_runbook_common.cpp`:

```cpp
TEST(test_offset_from_samples) {
    // Each sample: {t0_local, primary_ms, t1_local}. Best = smallest RTT.
    std::vector<rbc::ClockSample> s = {
        {1000, 5000, 1040},   // rtt 40, mid 1020, offset 5000-1020 = 3980
        {2000, 6010, 2010},   // rtt 10, mid 2005, offset 6010-2005 = 4005  <- chosen
    };
    CHECK(rbc::offset_from_samples(s) == 4005);
}

TEST(test_bucket_qps) {
    // completions at ms: 1000,1500,1999 (second 1) ; 2500 (second 2)
    std::vector<long long> c = {1000, 1500, 1999, 2500};
    auto rows = rbc::bucket_qps(c);   // vector<pair<bucket_ms, qps>> sorted
    CHECK(rows.size() == 2);
    CHECK(rows[0].first == 1000 && rows[0].second == 3);
    CHECK(rows[1].first == 2000 && rows[1].second == 1);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test`
Expected: compile error — `ClockSample`, `offset_from_samples`, `bucket_qps` undefined.

- [ ] **Step 3: Write minimal implementation**

Add `#include <libpq-fe.h>`, `#include <chrono>`, `#include <map>`, `#include <thread>`, `#include <mutex>` to `runbook_common.hpp`, and inside `namespace rbc`:

```cpp
// ---- monotonic-ish local wall clock in ms (epoch-based so offset maps to primary epoch) ----
inline long long now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
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
    if (PQstatus(c) != CONNECTION_OK)
        throw std::runtime_error(std::string("connect failed: ") + PQerrorMessage(c));
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
    static std::string conninfo;
    static PGconn* get() {
        thread_local PGconn* c = nullptr;
        if (!c) c = connect_or_die(conninfo);
        return c;
    }
};
```

In `runbook_common.hpp`, after the namespace close-brace problem: define the static member once. Add immediately **before** `#endif`, **outside** the namespace is not needed — instead define inside the header guarded by an `inline` variable. Replace the `ThreadConn` static declaration line `static std::string conninfo;` with:

```cpp
    static inline std::string conninfo;   // C++17 inline static — no separate definition needed
```

(So `ThreadConn` uses an `inline static` member; no `.cpp` definition required.)

- [ ] **Step 4: Run test to verify it passes**

Run: `make test`
Expected: `PASSED` for all checks including `test_offset_from_samples` and `test_bucket_qps`.

- [ ] **Step 5: Checkpoint** — stop for review.

---

## Task 7: `replica_recall_eval.cpp` — Experiment A

**Files:**
- Create: `pgvector/test/runbooks/replication/replica_recall_eval.cpp`
- (Makefile already has the target from Task 1.)

This driver connects to primary + standby, walks the runbook sequentially, and at each search step computes recall on both sides after WAL catchup. Live-DB correctness is verified by the operator (guidance, Task 9); here we verify it **compiles** and its **usage smoke** behaves.

- [ ] **Step 1: Write the program**

Create `replica_recall_eval.cpp`:

```cpp
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
static double avg_recall(const std::string& conninfo_str, const Args& a,
                         const std::vector<float>& queries, size_t dim,
                         const std::vector<std::vector<int>>& gt, int k, long num_q){
    rbc::ThreadConn::conninfo = conninfo_str;
    std::vector<double> per(num_q, 0.0);
    #pragma omp parallel for num_threads(a.threads) schedule(dynamic,1)
    for (long q = 0; q < num_q; ++q) {
        PGconn* c = rbc::ThreadConn::get();
        if (!c) continue;
        { PGresult* r = PQexec(c, set_search_param(a).c_str()); PQclear(r); }
        std::string sql = "SELECT id FROM " + a.table + " ORDER BY vec <-> " +
                          rbc::vector_to_sql(queries.data()+q*dim, dim) +
                          " LIMIT " + std::to_string(k);
        PGresult* r = PQexec(c, sql.c_str());
        if (PQresultStatus(r) == PGRES_TUPLES_OK) {
            std::unordered_set<int> g(gt[q].begin(), gt[q].begin()+std::min((size_t)k, gt[q].size()));
            int hits = 0, nr = PQntuples(r);
            for (int i = 0; i < nr; ++i) if (g.count(std::stoi(PQgetvalue(r,i,0)))) ++hits;
            per[q] = (double)hits / (double)k;
        }
        PQclear(r);
    }
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
            sql += "(" + std::to_string(i + a.dataset_offset) + ",'" +
                   rbc::vector_to_sql(v.data(), v.size()).substr(1);   // drop leading quote
        }
        sql += " ON CONFLICT (id) DO NOTHING;";
        PGresult* r = PQexec(p, sql.c_str());
        if (PQresultStatus(r) != PGRES_COMMAND_OK)
            throw std::runtime_error(std::string("insert failed: ")+PQerrorMessage(p));
        PQclear(r);
    }
}
```

> Note for the implementer: `vector_to_sql` returns `'[..]'` (single-quoted). The `INSERT` builds `(id,'[..]')`; the snippet above appends `"'"` then `vector_to_sql(...).substr(1)` so the result is `(id,'[..]')` with exactly one opening quote. Verify the produced SQL string by eye on first build (print one batch) — it must read `(123,'[1.0,...]')`.

Continue `replica_recall_eval.cpp`:

```cpp
static void delete_range(PGconn* p, const Args& a, long s, long e){
    std::string sql = "DELETE FROM " + a.table + " WHERE id >= " +
        std::to_string(s + a.dataset_offset) + " AND id < " + std::to_string(e + a.dataset_offset) + ";";
    PGresult* r = PQexec(p, sql.c_str());
    if (PQresultStatus(r) != PGRES_COMMAND_OK)
        throw std::runtime_error(std::string("delete failed: ")+PQerrorMessage(p));
    PQclear(r);
}

static void create_index(PGconn* p, const Args& a){
    std::string sql = a.index_type=="ivfflat"
        ? "CREATE INDEX ON "+a.table+" USING ivfflat (vec vector_l2_ops) WITH (lists="+std::to_string(a.ivfflat_lists)+");"
        : "CREATE INDEX ON "+a.table+" USING hnsw (vec vector_l2_ops) WITH (m="+std::to_string(a.hnsw_m)+
          ", ef_construction="+std::to_string(a.hnsw_ef_construction)+");";
    PGresult* r = PQexec(p, sql.c_str());
    if (PQresultStatus(r) != PGRES_COMMAND_OK)
        throw std::runtime_error(std::string("create index failed: ")+PQerrorMessage(p));
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

    for (const auto& st : steps) {
        if ((long)st.step_num < a.start_step) continue;
        if (a.end_step > 0 && (long)st.step_num > a.end_step) break;

        if (!index_built && a.build_index_before>0 && (long)st.step_num==a.build_index_before){
            std::cerr << "[step "<<st.step_num<<"] building index...\n";
            create_index(primary, a); index_built = true;
            wait_catchup(primary, standby, a.catchup_timeout_sec);
        }

        if (st.op == "insert") {
            insert_range(primary, dr, a, st.start, st.end);
            ranges.push_back({"insert", st.start, st.end});
        } else if (st.op == "delete") {
            delete_range(primary, a, st.start, st.end);
            ranges.push_back({"delete", st.start, st.end});
        } else if (st.op == "search") {
            wait_catchup(primary, standby, a.catchup_timeout_sec);
            if (a.settle_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(a.settle_ms));
            std::string gtf = a.gt_dir + "/" + rbc::gt_filename(runbook_id, st.step_num, ranges);
            std::ifstream probe(gtf);
            if (!probe.good()) {
                std::cerr << "[step "<<st.step_num<<"] MISSING GT "<<gtf<<"\n";
                csv << st.step_num << ",NaN,NaN\n"; csv.flush(); continue;
            }
            auto gt = rbc::load_gt_npy(gtf, num_q, st.k);
            double rp = avg_recall(pconn, a, queries, qdim, gt, st.k, num_q);
            double rs = avg_recall(sconn, a, queries, qdim, gt, st.k, num_q);
            csv << st.step_num << "," << std::fixed << std::setprecision(6) << rp << "," << rs << "\n";
            csv.flush();
            std::cerr << "[step "<<st.step_num<<"] primary="<<rp<<" standby="<<rs
                      <<" delta="<<std::fabs(rp-rs)<<"\n";
        }
    }
    PQfinish(primary); PQfinish(standby);
    std::cerr << "done -> " << a.out << "\n";
    return 0;
}
```

- [ ] **Step 2: Build**

Run: `make replica_recall_eval`
Expected: compiles and links with no errors (warnings OK).

- [ ] **Step 3: Usage smoke test**

Run: `./replica_recall_eval` (no args)
Expected: prints the `Usage:` block to stderr, exit code 2 (`echo $?` → `2`).

Run: `./replica_recall_eval --dataset x --queries y --runbook z --dataset-name n --gt-dir d --start-step 999 --end-step 0` against unreachable DBs
Expected: it fails fast at the dataset open / connect step with a clear error (not a crash) — this confirms argument wiring without needing a live DB. (Use a real small `--dataset`/`--queries`/`--runbook` if you want it to reach the connect step.)

- [ ] **Step 4: Checkpoint** — stop for review. Full functional verification is the operator run in Task 9.

---

## Task 8: `replica_throughput_eval.cpp` — Experiment B

**Files:**
- Create: `pgvector/test/runbooks/replication/replica_throughput_eval.cpp`
- (Makefile already has the target from Task 1.)

`--role primary` runs the mixed-mode workload and emits `step_boundaries.csv`; `--role standby` hammers random queries and emits `standby_throughput.csv`. The mixed-mode group loop is ported from `pgvector_test.cpp` (the shuffled work-array scheme, see `pgvector_test.cpp:1404-1519`), simplified: no recall, no crash sim, and a step-boundary timestamp recorded at each group start.

- [ ] **Step 1: Write the program**

Create `replica_throughput_eval.cpp`:

```cpp
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
    if (a.assume_synced) return 0;
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

    #pragma omp parallel num_threads(a.threads)
    {
        int tid = omp_get_thread_num();
        std::mt19937 gen(std::random_device{}() ^ (tid*2654435761u));
        std::uniform_int_distribution<size_t> pick(0, qn-1);
        PGconn* c = rbc::ThreadConn::get();
        { PGresult* r = PQexec(c, set_search_param(a).c_str()); PQclear(r); }
        while (!g_stop.load(std::memory_order_relaxed)) {
            if (t_end && (rbc::now_ms()+offset) >= t_end) break;
            size_t q = pick(gen);
            std::string sql = "SELECT id FROM "+a.table+" ORDER BY vec <-> "+
                              rbc::vector_to_sql(queries.data()+q*qdim, qdim)+" LIMIT 100";
            PGresult* r = PQexec(c, sql.c_str());
            bool ok = (PQresultStatus(r)==PGRES_TUPLES_OK);
            PQclear(r);
            if (ok) buckets[tid].push_back(rbc::now_ms()+offset);
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
    auto dr = rbc::open_vec_reader(a.dataset);
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

        #pragma omp parallel for num_threads(a.threads) schedule(dynamic,100)
        for (size_t w = 0; w < work.size(); ++w) {
            size_t g = work[w];
            long item = next[g].fetch_add(1, std::memory_order_relaxed);
            if (item >= item_count[g]) continue;
            PGconn* c = rbc::ThreadConn::get();
            if (!c) continue;
            const Step& st = group[g];
            if (st.op == "insert") {
                long vid = st.start + item;
                auto v = rbc::read_vec(dr, vid);
                std::string sql = "INSERT INTO "+a.table+" (id,vec) VALUES ("+
                    std::to_string(vid+a.dataset_offset)+",'"+
                    rbc::vector_to_sql(v.data(),v.size()).substr(1)+
                    " ON CONFLICT (id) DO NOTHING";
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

    if (a.role == "standby") {
        if (a.queries.empty()) { usage(argv[0]); return 2; }
        return run_standby(a);
    } else if (a.role == "primary") {
        if (a.dataset.empty()||a.queries.empty()||a.runbook.empty()||a.dataset_name.empty()){ usage(argv[0]); return 2; }
        return run_primary(a);
    }
    usage(argv[0]); return 2;
}
```

> Same `vector_to_sql` quoting note as Task 7 applies to the INSERT builder here. Print one generated INSERT on first build to confirm it reads `(123,'[...]')`.

- [ ] **Step 2: Build**

Run: `make replica_throughput_eval`
Expected: compiles and links cleanly.

- [ ] **Step 3: Usage smoke test**

Run: `./replica_throughput_eval` (no args)
Expected: `Usage:` block, exit 2.

Run: `./replica_throughput_eval --role standby --queries /nonexistent.fvecs --assume-synced-clocks`
Expected: fails fast with a clear "cannot open" error (confirms standby branch wiring without a DB).

- [ ] **Step 4: Build everything together**

Run: `make all && make test`
Expected: both drivers build; unit suite still `PASSED`.

- [ ] **Step 5: Checkpoint** — stop for review.

---

## Task 9: Guidance doc

**Files:**
- Create: `docs/replica_search_evaluation.md`

- [ ] **Step 1: Write the guidance**

Create `docs/replica_search_evaluation.md` covering (use the exact section list below; fill each with concrete commands using the verified defaults — dim 96, GT dir `/ssd_root/liu4127/msturing_runbook_gt`, runbook `pgvector/test/runbooks/msturing-10M_slidingwindow_runbook.yaml`, dataset `/ssd_root/dataset/turing10m/msturing-10M.fvecs`, queries `/ssd_root/dataset/turing10m/msturing-query.fvecs`):

1. **Overview** — the two experiments and their CSV outputs.
2. **Prerequisites** — PG 15+ built with this extension; `shared_preload_libraries='vector'`; dataset/query/runbook/GT files; build the drivers:
   ```bash
   cd pgvector/test/runbooks/replication && make all
   ```
3. **Replication setup** (operator-run; the scripts do NOT configure PG). Primary `postgresql.conf`:
   ```
   wal_level = replica
   max_wal_senders = 8
   max_worker_processes = 32
   wal_keep_size = 1GB
   shared_preload_libraries = 'vector'
   pgvector.replication_role = 'primary'
   pgvector.replication_primary_port = <P>
   pgvector.replication_shared_secret = '<secret>'
   pgvector.storage_base_dir = '<abs path>'
   ```
   Standby `postgresql.conf`:
   ```
   hot_standby = on
   max_worker_processes = 32
   shared_preload_libraries = 'vector'
   pgvector.replication_role = 'standby'
   pgvector.replication_primary_host = '<primary ip>'
   pgvector.replication_primary_port = <P>
   pgvector.replication_shared_secret = '<secret>'
   pgvector.replication_fetch_parallelism = 2
   pgvector.storage_base_dir = '<abs path>'
   ```
   Initial sync: `pg_basebackup` of the primary into the standby's data dir, then `rsync` of `storage_base_dir`. Confirm streaming with `SELECT * FROM pg_stat_replication` on the primary.
4. **Seeding for Experiment A** — two workflows:
   - *Seed-at-101*: insert steps 1–100 (5M rows) + `CREATE INDEX` once on the primary, let it replicate, then run with `--start-step 101` (no `--build-index-before`).
   - *Full-from-1*: run with `--start-step 1 --build-index-before 101` (table must already exist: `CREATE EXTENSION vector; CREATE TABLE vectors(id bigint primary key, vec vector(96));`).
5. **Run Experiment A** — full command:
   ```bash
   ./replica_recall_eval \
     --dataset /ssd_root/dataset/turing10m/msturing-10M.fvecs \
     --queries /ssd_root/dataset/turing10m/msturing-query.fvecs \
     --runbook ../msturing-10M_slidingwindow_runbook.yaml \
     --dataset-name msturing-10M \
     --gt-dir /ssd_root/liu4127/msturing_runbook_gt \
     --primary-host <P_IP> --primary-port <P_PORT> --primary-db postgres --primary-user <U> \
     --standby-host <S_IP> --standby-port <S_PORT> --standby-db postgres --standby-user <U> \
     --hnsw-ef-search 200 --num-verify-queries 10000 --threads 8 \
     --start-step 101 --end-step 400 --out experiment_a.csv
   ```
   Reading `experiment_a.csv`: `step,primary_recall,standby_recall`; a persistent nonzero `standby−primary` delta suggests replication divergence; a transient delta near a flush can be fetch lag (raise `--standby-settle-ms`).
6. **Run Experiment B** — launch order matters (two hosts):
   - On the **standby host**, start the query load first (runs until SIGINT):
     ```bash
     ./replica_throughput_eval --role standby \
       --queries /ssd_root/dataset/turing10m/msturing-query.fvecs \
       --standby-host 127.0.0.1 --standby-port <S_PORT> --standby-db postgres --standby-user <U> \
       --primary-host <P_IP> --primary-port <P_PORT> --primary-db postgres --primary-user <U> \
       --hnsw-ef-search 200 --threads 16 --out standby_throughput.csv
     ```
     (It connects to the primary once to prime the clock offset, then queries only the standby.)
   - On the **primary host**, run the mixed-mode workload:
     ```bash
     ./replica_throughput_eval --role primary \
       --dataset /ssd_root/dataset/turing10m/msturing-10M.fvecs \
       --queries /ssd_root/dataset/turing10m/msturing-query.fvecs \
       --runbook ../msturing-10M_slidingwindow_runbook.yaml \
       --dataset-name msturing-10M \
       --primary-host 127.0.0.1 --primary-port <P_PORT> --primary-db postgres --primary-user <U> \
       --mixed-mode-start 101 --mixed-size 3 --threads 16 --out step_boundaries.csv
     ```
   - When the primary finishes, send SIGINT (Ctrl-C) to the standby driver; it writes `standby_throughput.csv`.
   - Note: both CSVs use primary-DB-clock ms, so they overlay directly. If clocks are NTP-synced you may pass `--assume-synced-clocks` to skip priming.
7. **Plotting** (optional one-liner) — e.g. zero the time axis:
   ```python
   import pandas as pd
   t = pd.read_csv("standby_throughput.csv"); b = pd.read_csv("step_boundaries.csv")
   t0 = t.timestamp_ms.min()
   t["sec"]=(t.timestamp_ms-t0)/1000; b["sec"]=(b.timestamp_ms-t0)/1000
   # plot t.sec vs t.qps; overlay vertical lines at b.sec labeled b.step
   ```
8. **Troubleshooting** — catchup timeout (check `pg_stat_replication`/standby logs); GT filename mismatch (the `--dataset-name`/`--start-step`/`--dataset-offset` must match how GT was generated; the active-ranges hash is printed in the missing-GT error); fetcher lag (`pgvector.replication_fetch_parallelism`).

- [ ] **Step 2: Verify the doc renders and commands are internally consistent**

Run: `grep -n "replica_recall_eval\|replica_throughput_eval\|msturing" docs/replica_search_evaluation.md | head`
Expected: commands reference the binaries built in `pgvector/test/runbooks/replication/` and the verified paths; flag names match the `usage()` text in Tasks 7–8.

- [ ] **Step 3: Checkpoint** — stop for review.

---

## Self-Review notes (author)

- **Spec coverage:** Exp A driver (Task 7) ↔ spec §4.2; Exp B driver (Task 8) ↔ §4.3; shared header pieces (Tasks 1–6) ↔ §4.1; clock module (Task 6) ↔ §6; outputs (Tasks 7–8) ↔ §7; Makefile (Task 1) ↔ §4.4; guidance (Task 9) ↔ §8. `--standby-settle-ms` default 0 ↔ §2. Both start modes (start-step/build-index-before) ↔ §2.
- **Type consistency:** `rbc::Range{op,start,end}`, `rbc::Step{step_num,step_key,op,start,end,k}`, `rbc::VecReader`, `rbc::ClockSample{t0,primary_ms,t1}`, `rbc::ThreadConn::conninfo` (inline static) are defined in Tasks 2/4/5/6 and used consistently in Tasks 7–8. `gt_filename(runbook_id, step, ranges)`, `read_runbook(path, dataset_name)`, `ranges_up_to(steps, upto)`, `bucket_qps(vec)`, `offset_from_samples(vec)`, `prime_clock_offset(conninfo,k)`, `now_ms()`, `vector_to_sql(ptr,dim)`, `load_gt_npy(path,n,k)` — signatures match across tasks.
- **Known sharp edge:** the INSERT-builder `vector_to_sql(...).substr(1)` quoting trick (Tasks 7 & 8) is flagged inline with a "print one batch to confirm" step. If preferred, replace with explicit `"(" + id + ",'" + bracketed + "')"` where `bracketed` strips both quotes — functionally equivalent.
```
