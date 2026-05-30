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

int main() {
    for (auto& t : g_tests) { std::cout << "RUN  " << t.first << "\n"; t.second(); }
    std::cout << (g_failures ? "\nFAILED " : "\nPASSED ")
              << (g_checks - g_failures) << "/" << g_checks << " checks\n";
    return g_failures ? 1 : 0;
}
