/**
 * Standalone perf test for mmap vs resident load of INDEX_HNSW (Faiss HNSW) in knowhere.
 *
 * Flow:
 *  - build INDEX_HNSW from randomly generated vectors
 *  - serialize the underlying Faiss index bytes to a single file
 *  - load the same file twice via DeserializeFromFile():
 *      (1) enable_mmap=false (resident)
 *      (2) enable_mmap=true  (mmap-backed IO reader)
 *  - run identical batched searches and report timings
 *
 * Usage:
 *   ./hnsw_mmap_perf_test [required options are none]
 *
 * Options (similar style to knowhere_concurrent_query_random.cpp):
 *   --nb <int>                 number of base vectors (default: 100000)
 *   --nq <int>                 number of query vectors (default: 1000)
 *   --dim <int>                vector dimension (default: 128)
 *   --k <int>                  top-k (default: 10)
 *   --metric-type <L2|IP|COSINE> metric type (default: L2)
 *   --M <int>                  HNSW graph degree (default: 16)
 *   --efConstruction <int>     build-time efConstruction (default: 200)
 *   --ef <int>                 search-time ef (default: 64)
 *   --seed <int>               RNG seed (default: 42)
 *   --warmup <int>             warmup queries before timing (default: 50)
 *   --runs <int>               timed search runs per mode (default: 5)
 *   --index-file <path>        index file path to write (default: /tmp/knowhere_hnsw_mmap_perf.index)
 *   --skip-build               skip build and only load+search existing index-file
 *
 * Dataset input (optional; if not provided, random data is generated):
 *   --base-file <path>         base vectors file (.fvecs/.bvecs/.fbin)
 *   --query-file <path>        query vectors file (.fvecs/.bvecs/.fbin)
 *   --ground-truth-file <path> ground truth IDs file (.ivecs) (optional; enables recall@k)
 *   --use-bvecs                override vector file format to .bvecs (for base/query)
 *   --use-fbin                 override vector file format to .fbin (for base/query)
 *   --base-skip <int>          skip first N base vectors (default: 0)
 *   --base-num <int>           read at most N base vectors (default: -1, all)
 *   --query-skip <int>         skip first N query vectors (default: 0)
 *   --query-num <int>          read at most N query vectors (default: -1, all)
 */

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include "knowhere/bitsetview.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/version.h"

namespace {

// Vector file format (auto-detected from extension, or overridden by --use-bvecs/--use-fbin)
enum class VectorFileFormat { Fvecs, Bvecs, Fbin };

static VectorFileFormat
detect_vector_format(const std::string& filename) {
    std::string ext;
    size_t dot = filename.rfind('.');
    if (dot != std::string::npos && dot + 1 < filename.size()) {
        ext = filename.substr(dot + 1);
        for (char& c : ext) {
            if (c >= 'A' && c <= 'Z') c += ('a' - 'A');
        }
    }
    if (ext == "fbin") return VectorFileFormat::Fbin;
    if (ext == "bvecs") return VectorFileFormat::Bvecs;
    if (ext == "fvecs") return VectorFileFormat::Fvecs;
    return VectorFileFormat::Fvecs;  // default
}

struct Options {
    int64_t nb = 100000;
    int64_t nq = 1000;
    int64_t dim = 128;
    int k = 10;
    std::string metric_type = knowhere::metric::L2;
    int hnsw_M = 16;
    int ef_construction = 200;
    int ef_search = 64;
    int seed = 42;
    int warmup = 50;
    int runs = 5;
    std::string index_file = "/tmp/knowhere_hnsw_mmap_perf.index";
    bool skip_build = false;

    // dataset inputs (optional)
    std::string base_file;
    std::string query_file;
    std::string ground_truth_file;
    bool use_bvecs = false;
    bool use_fbin = false;
    int64_t base_skip = 0;
    int64_t base_num = -1;
    int64_t query_skip = 0;
    int64_t query_num = -1;
};

static void
print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --nb <int>                     (default: 100000)\n"
              << "  --nq <int>                     (default: 1000)\n"
              << "  --dim <int>                    (default: 128)\n"
              << "  --k <int>                      (default: 10)\n"
              << "  --metric-type <L2|IP|COSINE>   (default: L2)\n"
              << "  --M <int>                      (default: 16)\n"
              << "  --efConstruction <int>         (default: 200)\n"
              << "  --ef <int>                     (default: 64)\n"
              << "  --seed <int>                   (default: 42)\n"
              << "  --warmup <int>                 (default: 50)\n"
              << "  --runs <int>                   (default: 5)\n"
              << "  --index-file <path>            (default: /tmp/knowhere_hnsw_mmap_perf.index)\n"
              << "  --skip-build                   (default: false)\n"
              << "\n"
              << "Dataset inputs (optional; if not provided, random vectors are generated):\n"
              << "  --base-file <path>             base vectors (.fvecs/.bvecs/.fbin)\n"
              << "  --query-file <path>            query vectors (.fvecs/.bvecs/.fbin)\n"
              << "  --ground-truth-file <path>     ground truth IDs (.ivecs)\n"
              << "  --use-bvecs                    override base/query format to .bvecs\n"
              << "  --use-fbin                     override base/query format to .fbin\n"
              << "  --base-skip <int>              skip first N base vectors (default: 0)\n"
              << "  --base-num <int>               read at most N base vectors (default: -1, all)\n"
              << "  --query-skip <int>             skip first N query vectors (default: 0)\n"
              << "  --query-num <int>              read at most N query vectors (default: -1, all)\n";
}

static bool
is_metric_valid(const std::string& m) {
    return m == knowhere::metric::L2 || m == knowhere::metric::IP || m == knowhere::metric::COSINE;
}

static int64_t
parse_i64(const std::string& s) {
    size_t pos = 0;
    long long v = std::stoll(s, &pos);
    if (pos != s.size()) {
        throw std::invalid_argument("invalid integer: " + s);
    }
    return static_cast<int64_t>(v);
}

static int
parse_i32(const std::string& s) {
    auto v = parse_i64(s);
    if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
        throw std::out_of_range("integer out of range: " + s);
    }
    return static_cast<int>(v);
}

static Options
parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + flag);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--nb") {
            opt.nb = parse_i64(need_value("--nb"));
        } else if (arg == "--nq") {
            opt.nq = parse_i64(need_value("--nq"));
        } else if (arg == "--dim") {
            opt.dim = parse_i64(need_value("--dim"));
        } else if (arg == "--k") {
            opt.k = parse_i32(need_value("--k"));
        } else if (arg == "--metric-type") {
            opt.metric_type = need_value("--metric-type");
        } else if (arg == "--M") {
            opt.hnsw_M = parse_i32(need_value("--M"));
        } else if (arg == "--efConstruction") {
            opt.ef_construction = parse_i32(need_value("--efConstruction"));
        } else if (arg == "--ef") {
            opt.ef_search = parse_i32(need_value("--ef"));
        } else if (arg == "--seed") {
            opt.seed = parse_i32(need_value("--seed"));
        } else if (arg == "--warmup") {
            opt.warmup = parse_i32(need_value("--warmup"));
        } else if (arg == "--runs") {
            opt.runs = parse_i32(need_value("--runs"));
        } else if (arg == "--index-file") {
            opt.index_file = need_value("--index-file");
        } else if (arg == "--skip-build") {
            opt.skip_build = true;
        } else if (arg == "--base-file") {
            opt.base_file = need_value("--base-file");
        } else if (arg == "--query-file") {
            opt.query_file = need_value("--query-file");
        } else if (arg == "--ground-truth-file") {
            opt.ground_truth_file = need_value("--ground-truth-file");
        } else if (arg == "--use-bvecs") {
            opt.use_bvecs = true;
        } else if (arg == "--use-fbin") {
            opt.use_fbin = true;
        } else if (arg == "--base-skip") {
            opt.base_skip = parse_i64(need_value("--base-skip"));
        } else if (arg == "--base-num") {
            opt.base_num = parse_i64(need_value("--base-num"));
        } else if (arg == "--query-skip") {
            opt.query_skip = parse_i64(need_value("--query-skip"));
        } else if (arg == "--query-num") {
            opt.query_num = parse_i64(need_value("--query-num"));
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }

    if (opt.nb <= 0 || opt.nq <= 0 || opt.dim <= 0) {
        throw std::invalid_argument("nb/nq/dim must be > 0");
    }
    if (opt.k <= 0) {
        throw std::invalid_argument("k must be > 0");
    }
    if (!is_metric_valid(opt.metric_type)) {
        throw std::invalid_argument("metric-type must be L2, IP, or COSINE");
    }
    if (opt.hnsw_M <= 0 || opt.ef_construction <= 0 || opt.ef_search <= 0) {
        throw std::invalid_argument("M/efConstruction/ef must be > 0");
    }
    if (opt.warmup < 0 || opt.runs <= 0) {
        throw std::invalid_argument("warmup must be >= 0 and runs must be > 0");
    }
    if (opt.base_skip < 0 || opt.query_skip < 0) {
        throw std::invalid_argument("base-skip/query-skip must be >= 0");
    }

    if (opt.use_bvecs && opt.use_fbin) {
        throw std::invalid_argument("use-bvecs and use-fbin are mutually exclusive");
    }

    return opt;
}

// File format structures (copied/simplified from knowhere_concurrent_query_random.cpp)
struct FvecsReader {
    std::ifstream file;
    int dim;

    explicit FvecsReader(const std::string& filename) : file(filename, std::ios::binary) {
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        int32_t d;
        file.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        if (!file) {
            throw std::runtime_error("Failed to read fvecs dim from file: " + filename);
        }
        dim = d;
        file.seekg(0, std::ios::beg);
    }

    void read_all_vectors(std::vector<std::vector<float>>& vectors, int64_t skip = 0, int64_t max_vectors = -1) {
        file.clear();
        file.seekg(0, std::ios::beg);
        vectors.clear();

        for (int64_t i = 0; i < skip; i++) {
            int32_t vec_dim;
            file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int32_t));
            if (!file || file.eof()) break;
            file.seekg(vec_dim * sizeof(float), std::ios::cur);
        }

        int64_t count = 0;
        while (true) {
            if (max_vectors > 0 && count >= max_vectors) break;
            int32_t vec_dim;
            file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int32_t));
            if (!file || file.eof()) break;
            std::vector<float> vec(vec_dim);
            file.read(reinterpret_cast<char*>(vec.data()), vec_dim * sizeof(float));
            if (!file) break;
            vectors.push_back(std::move(vec));
            count++;
        }
    }
};

struct BvecsReader {
    std::ifstream file;
    int dim;

    explicit BvecsReader(const std::string& filename) : file(filename, std::ios::binary) {
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        int32_t d;
        file.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        if (!file) {
            throw std::runtime_error("Failed to read bvecs dim from file: " + filename);
        }
        dim = d;
        file.seekg(0, std::ios::beg);
    }

    void read_all_vectors(std::vector<std::vector<float>>& vectors, int64_t skip = 0, int64_t max_vectors = -1) {
        file.clear();
        file.seekg(0, std::ios::beg);
        vectors.clear();

        for (int64_t i = 0; i < skip; i++) {
            int32_t vec_dim;
            file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int32_t));
            if (!file || file.eof()) break;
            file.seekg(vec_dim, std::ios::cur);
        }

        int64_t count = 0;
        while (true) {
            if (max_vectors > 0 && count >= max_vectors) break;
            int32_t vec_dim;
            file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int32_t));
            if (!file || file.eof()) break;
            std::vector<uint8_t> bvec(vec_dim);
            file.read(reinterpret_cast<char*>(bvec.data()), vec_dim);
            if (!file) break;
            std::vector<float> vec(vec_dim);
            for (int i = 0; i < vec_dim; i++) vec[i] = static_cast<float>(bvec[i]);
            vectors.push_back(std::move(vec));
            count++;
        }
    }
};

// fbin format: 8-byte header (n:int32, d:int32 little-endian), then n*d float32 contiguous
struct FbinReader {
    std::ifstream file;
    int dim;
    int64_t num_vectors;
    int64_t data_offset;

    explicit FbinReader(const std::string& filename) : file(filename, std::ios::binary) {
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        int32_t n, d;
        file.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        if (!file || n <= 0 || d <= 0) {
            throw std::runtime_error("Invalid fbin header: n=" + std::to_string(n) + " d=" + std::to_string(d));
        }
        num_vectors = static_cast<int64_t>(n);
        dim = d;
        data_offset = 8;
    }

    void read_all_vectors(std::vector<std::vector<float>>& vectors, int64_t skip = 0, int64_t max_vectors = -1) {
        vectors.clear();
        int64_t to_read = (max_vectors < 0) ? (num_vectors - skip) : std::min(max_vectors, num_vectors - skip);
        if (to_read <= 0) return;
        int64_t start_offset = data_offset + skip * static_cast<int64_t>(dim) * sizeof(float);
        file.clear();
        file.seekg(start_offset, std::ios::beg);

        for (int64_t i = 0; i < to_read; i++) {
            std::vector<float> vec(dim);
            file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
            if (!file) break;
            vectors.push_back(std::move(vec));
        }
    }
};

struct IvecsReader {
    std::ifstream file;
    int k;

    IvecsReader(const std::string& filename, int expected_k) : k(expected_k) {
        file.open(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
    }

    void read_all_ground_truth(std::vector<std::vector<int32_t>>& ground_truth, int64_t skip = 0, int64_t max_queries = -1) {
        file.clear();
        file.seekg(0, std::ios::beg);
        ground_truth.clear();

        // Skip queries
        for (int64_t i = 0; i < skip; i++) {
            int32_t vec_k;
            file.read(reinterpret_cast<char*>(&vec_k), sizeof(int32_t));
            if (!file || file.eof()) break;
            file.seekg(static_cast<std::streamoff>(vec_k) * sizeof(int32_t), std::ios::cur);
        }

        int64_t count = 0;
        while (true) {
            if (max_queries > 0 && count >= max_queries) break;
            int32_t vec_k;
            file.read(reinterpret_cast<char*>(&vec_k), sizeof(int32_t));
            if (!file || file.eof()) break;
            std::vector<int32_t> gt(vec_k);
            file.read(reinterpret_cast<char*>(gt.data()), vec_k * sizeof(int32_t));
            if (!file) break;
            if (static_cast<int>(gt.size()) > k) gt.resize(k);
            ground_truth.push_back(std::move(gt));
            count++;
        }
    }
};

static double
calculate_recall(const std::vector<std::vector<int64_t>>& predicted, const std::vector<std::vector<int32_t>>& ground_truth,
                 int k) {
    if (predicted.empty() || ground_truth.empty()) return 0.0;
    int64_t total_correct = 0;
    int64_t queries_with_results = 0;
    int64_t num_queries = std::min<int64_t>(predicted.size(), ground_truth.size());
    for (int64_t i = 0; i < num_queries; i++) {
        if (predicted[i].empty() || ground_truth[i].empty()) continue;
        queries_with_results++;
        std::set<int64_t> gt_set;
        int gt_k = std::min(k, static_cast<int>(ground_truth[i].size()));
        for (int j = 0; j < gt_k; j++) gt_set.insert(static_cast<int64_t>(ground_truth[i][j]));
        int pred_k = std::min(k, static_cast<int>(predicted[i].size()));
        for (int j = 0; j < pred_k; j++) {
            if (gt_set.find(predicted[i][j]) != gt_set.end()) total_correct++;
        }
    }
    if (queries_with_results == 0) return 0.0;
    return static_cast<double>(total_correct) / (queries_with_results * k);
}

static std::vector<float>
gen_random_vectors(int64_t rows, int64_t dim, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(static_cast<size_t>(rows * dim));
    for (auto& x : data) {
        x = dist(rng);
    }
    return data;
}

static void
write_raw_index_bytes_to_file(const std::string& filename, const knowhere::BinarySet& binset,
                              const std::string& key_name) {
    auto bin = binset.GetByName(key_name);
    if (!bin || !bin->data || bin->size <= 0) {
        throw std::runtime_error("Serialize() did not produce binary for key: " + key_name);
    }

    fs::path p(filename);
    if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
    }

    std::ofstream out(filename, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("cannot open index file for write: " + filename);
    }
    out.write(reinterpret_cast<const char*>(bin->data.get()), bin->size);
    if (!out.good()) {
        throw std::runtime_error("failed writing index bytes to file: " + filename);
    }
    out.close();
}

struct Timings {
    double load_ms = 0.0;
    double warmup_ms = 0.0;
    std::vector<double> run_ms;
};

static Timings
load_and_benchmark(const std::string& label, const std::string& index_file, bool enable_mmap,
                   const knowhere::DataSetPtr& queries, const knowhere::Json& base_search_conf,
                   int warmup_q, int runs, std::vector<std::vector<int64_t>>* out_ids = nullptr) {
    Timings t;
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto idx_result = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version);
    if (!idx_result.has_value()) {
        throw std::runtime_error("failed to create INDEX_HNSW index");
    }
    auto index = idx_result.value();

    knowhere::Json load_conf;
    load_conf[knowhere::meta::METRIC_TYPE] = base_search_conf.at(knowhere::meta::METRIC_TYPE);
    load_conf["enable_mmap"] = enable_mmap;

    auto t0 = std::chrono::steady_clock::now();
    auto st = index.DeserializeFromFile(index_file, load_conf);
    auto t1 = std::chrono::steady_clock::now();
    t.load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (st != knowhere::Status::success) {
        throw std::runtime_error("DeserializeFromFile failed for " + label + " (status=" + std::to_string((int)st) + ")");
    }

    knowhere::BitsetView bitset(nullptr);

    // Warmup: run search on first warmup_q queries (if any).
    if (warmup_q > 0) {
        int64_t qdim = queries->GetDim();
        int64_t qrows = queries->GetRows();
        int64_t n = std::min<int64_t>(qrows, warmup_q);
        auto warmup_ds = knowhere::GenDataSet(n, qdim, (float*)queries->GetTensor());
        t0 = std::chrono::steady_clock::now();
        auto res = index.Search(warmup_ds, base_search_conf, bitset);
        t1 = std::chrono::steady_clock::now();
        t.warmup_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (!res.has_value()) {
            throw std::runtime_error("warmup Search failed for " + label);
        }
    }

    t.run_ms.reserve(runs);
    for (int r = 0; r < runs; ++r) {
        t0 = std::chrono::steady_clock::now();
        auto res = index.Search(queries, base_search_conf, bitset);
        t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (!res.has_value()) {
            throw std::runtime_error("Search failed for " + label);
        }
        if (out_ids && r == runs - 1) {
            // capture ids from last run (for recall computation)
            auto ds = res.value();
            const int64_t* ids = ds->GetIds();
            int64_t nq = ds->GetRows();
            int64_t k = ds->GetDim();
            out_ids->clear();
            out_ids->resize(nq);
            if (ids) {
                for (int64_t i = 0; i < nq; i++) {
                    (*out_ids)[i].assign(ids + i * k, ids + i * k + k);
                }
            }
        }
        t.run_ms.push_back(ms);
    }

    // Print a small summary here (keeps the executable self-contained).
    auto min_ms = *std::min_element(t.run_ms.begin(), t.run_ms.end());
    auto max_ms = *std::max_element(t.run_ms.begin(), t.run_ms.end());
    double sum_ms = 0.0;
    for (auto v : t.run_ms) sum_ms += v;
    double avg_ms = sum_ms / t.run_ms.size();

    const double qps = (queries->GetRows() * 1000.0) / avg_ms;
    std::cout << "\n[" << label << "] enable_mmap=" << (enable_mmap ? "true" : "false") << "\n";
    std::cout << "  load_ms=" << std::fixed << std::setprecision(2) << t.load_ms
              << " warmup_ms=" << t.warmup_ms << "\n";
    std::cout << "  search_ms(avg/min/max over " << runs << " runs) = "
              << avg_ms << " / " << min_ms << " / " << max_ms << "\n";
    std::cout << "  QPS(avg) = " << std::fixed << std::setprecision(2) << qps << "\n";

    return t;
}

}  // namespace

int
main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        std::cout << "=== Knowhere HNSW mmap perf test ===\n";
        std::cout << "index_type=" << knowhere::IndexEnum::INDEX_HNSW << " metric_type=" << opt.metric_type << "\n";
        std::cout << "nb=" << opt.nb << " nq=" << opt.nq << " dim=" << opt.dim << " k=" << opt.k << "\n";
        std::cout << "M=" << opt.hnsw_M << " efConstruction=" << opt.ef_construction << " ef=" << opt.ef_search << "\n";
        std::cout << "seed=" << opt.seed << " warmup=" << opt.warmup << " runs=" << opt.runs << "\n";
        std::cout << "index_file=" << opt.index_file << " skip_build=" << (opt.skip_build ? "true" : "false") << "\n";
        if (!opt.base_file.empty()) {
            std::cout << "base_file=" << opt.base_file << " base_skip=" << opt.base_skip << " base_num=" << opt.base_num
                      << "\n";
        }
        if (!opt.query_file.empty()) {
            std::cout << "query_file=" << opt.query_file << " query_skip=" << opt.query_skip
                      << " query_num=" << opt.query_num << "\n";
        }
        if (!opt.ground_truth_file.empty()) {
            std::cout << "ground_truth_file=" << opt.ground_truth_file << "\n";
        }

        // Load or generate base/query vectors.
        std::vector<float> base_contig;
        std::vector<float> query_contig;
        int64_t base_rows = opt.nb;
        int64_t query_rows = opt.nq;
        int64_t dim = opt.dim;

        auto resolve_fmt = [&](const std::string& filename) -> VectorFileFormat {
            if (opt.use_fbin) return VectorFileFormat::Fbin;
            if (opt.use_bvecs) return VectorFileFormat::Bvecs;
            return detect_vector_format(filename);
        };

        auto load_vectors = [&](const std::string& filename, int64_t skip, int64_t max_num, int64_t& out_rows,
                                int64_t& out_dim, std::vector<float>& out_contig) {
            std::vector<std::vector<float>> vecs;
            auto fmt = resolve_fmt(filename);
            if (fmt == VectorFileFormat::Fbin) {
                FbinReader r(filename);
                r.read_all_vectors(vecs, skip, max_num);
                out_dim = r.dim;
            } else if (fmt == VectorFileFormat::Bvecs) {
                BvecsReader r(filename);
                r.read_all_vectors(vecs, skip, max_num);
                out_dim = r.dim;
            } else {
                FvecsReader r(filename);
                r.read_all_vectors(vecs, skip, max_num);
                out_dim = r.dim;
            }
            if (vecs.empty()) {
                throw std::runtime_error("no vectors loaded from: " + filename);
            }
            out_rows = static_cast<int64_t>(vecs.size());
            out_contig.resize(static_cast<size_t>(out_rows * out_dim));
            for (int64_t i = 0; i < out_rows; i++) {
                if (static_cast<int64_t>(vecs[i].size()) != out_dim) {
                    throw std::runtime_error("inconsistent vector dim in: " + filename);
                }
                std::memcpy(out_contig.data() + i * out_dim, vecs[i].data(), out_dim * sizeof(float));
            }
        };

        if (!opt.base_file.empty() && !opt.skip_build) {
            int64_t bd = 0;
            load_vectors(opt.base_file, opt.base_skip, opt.base_num, base_rows, bd, base_contig);
            dim = bd;
        } else if (!opt.skip_build) {
            base_contig = gen_random_vectors(opt.nb, opt.dim, opt.seed);
        }

        if (!opt.query_file.empty()) {
            int64_t qd = 0;
            load_vectors(opt.query_file, opt.query_skip, opt.query_num, query_rows, qd, query_contig);
            if (!opt.skip_build) {
                if (qd != dim) {
                    throw std::runtime_error("query dim does not match base dim");
                }
            } else {
                dim = qd;
            }
        } else {
            query_contig = gen_random_vectors(opt.nq, opt.dim, opt.seed + 1);
        }

        if (!opt.skip_build) {
            opt.nb = base_rows;
            opt.dim = dim;
        }
        opt.nq = query_rows;

        auto query_ds = knowhere::GenDataSet(opt.nq, dim, query_contig.data());

        // Build+save if requested.
        if (!opt.skip_build) {
            auto train_ds = knowhere::GenDataSet(opt.nb, dim, base_contig.data());

            auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
            auto idx_result =
                knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version);
            if (!idx_result.has_value()) {
                std::cerr << "failed to create INDEX_HNSW\n";
                return 1;
            }
            auto index = idx_result.value();

            knowhere::Json build_conf = {
                {knowhere::meta::DIM, dim},
                {knowhere::meta::METRIC_TYPE, opt.metric_type},
                {knowhere::indexparam::HNSW_M, opt.hnsw_M},
                {knowhere::indexparam::EFCONSTRUCTION, opt.ef_construction},
            };

            auto t0 = std::chrono::steady_clock::now();
            auto st = index.Build(train_ds, build_conf);
            auto t1 = std::chrono::steady_clock::now();
            double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (st != knowhere::Status::success) {
                std::cerr << "Build failed (status=" << (int)st << ")\n";
                return 1;
            }
            std::cout << "build_ms=" << std::fixed << std::setprecision(2) << build_ms
                      << " count=" << index.Count() << "\n";

            // Serialize to BinarySet, but write only the raw underlying index bytes to disk.
            knowhere::BinarySet binset;
            st = index.Serialize(binset);
            if (st != knowhere::Status::success) {
                std::cerr << "Serialize failed (status=" << (int)st << ")\n";
                return 1;
            }
            write_raw_index_bytes_to_file(opt.index_file, binset, knowhere::IndexEnum::INDEX_HNSW);
            std::cout << "index_saved_bytes=" << binset.GetByName(knowhere::IndexEnum::INDEX_HNSW)->size << "\n";
        } else {
            if (!fs::exists(opt.index_file)) {
                std::cerr << "index_file does not exist: " << opt.index_file << "\n";
                return 1;
            }
        }

        knowhere::Json search_conf = {
            {knowhere::meta::DIM, dim},
            {knowhere::meta::METRIC_TYPE, opt.metric_type},
            {knowhere::meta::TOPK, opt.k},
            {knowhere::indexparam::EF, opt.ef_search},
        };

        // Benchmark resident first, then mmap-backed.
        std::vector<std::vector<int64_t>> ids_resident;
        std::vector<std::vector<int64_t>> ids_mmap;
        (void)load_and_benchmark("resident", opt.index_file, /*enable_mmap=*/false, query_ds, search_conf, opt.warmup,
                                 opt.runs, opt.ground_truth_file.empty() ? nullptr : &ids_resident);
        (void)load_and_benchmark("mmap", opt.index_file, /*enable_mmap=*/true, query_ds, search_conf, opt.warmup,
                                 opt.runs, opt.ground_truth_file.empty() ? nullptr : &ids_mmap);

        if (!opt.ground_truth_file.empty()) {
            std::vector<std::vector<int32_t>> gt;
            IvecsReader gt_reader(opt.ground_truth_file, opt.k);
            gt_reader.read_all_ground_truth(gt, opt.query_skip, opt.query_num);
            double recall_res = calculate_recall(ids_resident, gt, opt.k);
            double recall_mmap = calculate_recall(ids_mmap, gt, opt.k);
            std::cout << "\nRecall@" << opt.k << " (resident) = " << std::fixed << std::setprecision(4) << recall_res
                      << "\n";
            std::cout << "Recall@" << opt.k << " (mmap)     = " << std::fixed << std::setprecision(4) << recall_mmap
                      << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }
}

