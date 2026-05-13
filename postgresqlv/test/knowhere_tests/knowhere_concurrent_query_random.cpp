// Random-concurrent query test based on knowhere_concurrent_query.cpp.
// Differences:
//   - Each worker thread repeatedly runs queries, picking a random query
//     vector on every iteration (with replacement).
//   - Execution runs for a configurable wall-clock duration instead of
//     exhausting a fixed query list once.
//   - No ground-truth / recall computation; we only report latency/throughput.

#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <memory>
#include <random>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

// Include knowhere headers
#include "knowhere/index/index.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/binaryset.h"
#include "knowhere/object.h"

// Index serialization helpers (copied from knowhere_concurrent_query.cpp)
struct FileIOWriter {
    std::ofstream fs;
    std::string name;

    explicit FileIOWriter(const std::string& fname) {
        name = fname;
        fs.open(name, std::ios::out | std::ios::binary);
        if (!fs.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + fname);
        }
    }

    ~FileIOWriter() {
        if (fs.is_open()) {
            fs.close();
        }
    }

    size_t operator()(void* ptr, size_t size) {
        fs.write(reinterpret_cast<char*>(ptr), size);
        return size;
    }
};

struct FileIOReader {
    std::ifstream fs;
    std::string name;

    explicit FileIOReader(const std::string& fname) {
        name = fname;
        fs.open(name, std::ios::in | std::ios::binary);
        if (!fs.is_open()) {
            throw std::runtime_error("Cannot open file for reading: " + fname);
        }
    }

    ~FileIOReader() {
        if (fs.is_open()) {
            fs.close();
        }
    }

    size_t operator()(void* ptr, size_t size) {
        fs.read(reinterpret_cast<char*>(ptr), size);
        return fs.gcount();
    }

    int64_t size() {
        fs.seekg(0, fs.end);
        int64_t len = fs.tellg();
        fs.seekg(0, fs.beg);
        return len;
    }
};

static void
write_binary_set(knowhere::BinarySet& binary_set, const std::string& filename) {
    FileIOWriter writer(filename);

    const auto& m = binary_set.binary_map_;
    for (auto it = m.begin(); it != m.end(); ++it) {
        const std::string& name = it->first;
        size_t name_size = name.length();
        const knowhere::BinaryPtr data = it->second;
        size_t data_size = data->size;

        writer(&name_size, sizeof(name_size));
        writer(&data_size, sizeof(data_size));
        writer((void*)name.c_str(), name_size);
        writer(data->data.get(), data_size);
    }
}

static void
read_binary_set(knowhere::BinarySet& binary_set, const std::string& filename) {
    FileIOReader reader(filename);
    int64_t file_size = reader.size();
    if (file_size < 0) {
        throw std::runtime_error("Cannot determine file size: " + filename);
    }

    int64_t offset = 0;
    while (offset < file_size) {
        size_t name_size, data_size;
        reader(&name_size, sizeof(size_t));
        offset += sizeof(size_t);
        reader(&data_size, sizeof(size_t));
        offset += sizeof(size_t);

        if (name_size == 0 && data_size == 0) {
            // Sentinel value indicating end of file
            break;
        }

        std::string name;
        name.resize(name_size);
        reader((void*)name.data(), name_size);
        offset += name_size;
        auto data = new uint8_t[data_size];
        reader(data, data_size);
        offset += data_size;

        std::shared_ptr<uint8_t[]> data_ptr(data);
        binary_set.Append(name, data_ptr, data_size);
    }
}

static void
write_index(knowhere::Index<knowhere::IndexNode>& index, const std::string& filename) {
    knowhere::BinarySet binary_set;
    index.Serialize(binary_set);
    write_binary_set(binary_set, filename);
}

static void
read_index(knowhere::Index<knowhere::IndexNode>& index, const std::string& filename, const knowhere::Json& conf) {
    knowhere::BinarySet binary_set;
    read_binary_set(binary_set, filename);
    index.Deserialize(binary_set, conf);
}

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

static const char*
format_extension(VectorFileFormat fmt) {
    switch (fmt) {
        case VectorFileFormat::Fbin: return ".fbin";
        case VectorFileFormat::Bvecs: return ".bvecs";
        case VectorFileFormat::Fvecs: return ".fvecs";
    }
    return ".fvecs";
}

// File format structures (subset of those in knowhere_concurrent_query.cpp)
struct FvecsReader {
    std::ifstream file;
    int dim;
    int64_t num_vectors;

    FvecsReader(const std::string& filename) : file(filename, std::ios::binary) {
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        // Read first vector to get dimension
        int32_t d;
        file.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        dim = d;
        file.seekg(0, std::ios::beg);
        num_vectors = 0;  // Not counting, will read as needed
    }

    void read_all_vectors(std::vector<std::vector<float>>& vectors, int64_t skip = 0, int64_t max_vectors = -1, bool show_progress = true) {
        file.seekg(0, std::ios::beg);
        vectors.clear();

        // Skip vectors
        if (show_progress && skip > 0) {
            std::cout << "Skipping " << skip << " vectors...";
            std::cout.flush();
        }
        for (int64_t i = 0; i < skip; i++) {
            int32_t vec_dim;
            file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int32_t));
            if (!file || file.eof()) break;
            file.seekg(vec_dim * sizeof(float), std::ios::cur);
        }
        if (show_progress && skip > 0) {
            std::cout << " done\n";
        }

        // Read vectors
        int64_t count = 0;

        while (true) {
            if (max_vectors > 0 && count >= max_vectors) break;
            int32_t vec_dim;
            file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int32_t));
            if (!file || file.eof()) break;
            std::vector<float> vec(vec_dim);
            file.read(reinterpret_cast<char*>(vec.data()), vec_dim * sizeof(float));
            vectors.push_back(std::move(vec));
            count++;
        }
        if (show_progress) {
            std::cout << count << " loaded\n";
        }
    }
};

struct BvecsReader {
    std::ifstream file;
    int dim;
    int64_t num_vectors;

    BvecsReader(const std::string& filename) : file(filename, std::ios::binary) {
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        // Read first vector to get dimension
        int32_t d;
        file.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        dim = d;
        file.seekg(0, std::ios::beg);
        num_vectors = 0;  // Not counting, will read as needed
    }

    void read_all_vectors(std::vector<std::vector<float>>& vectors, int64_t skip = 0, int64_t max_vectors = -1, bool show_progress = true) {
        file.seekg(0, std::ios::beg);
        vectors.clear();

        // Skip vectors
        if (show_progress && skip > 0) {
            std::cout << "Skipping " << skip << " vectors...";
            std::cout.flush();
        }
        for (int64_t i = 0; i < skip; i++) {
            int32_t vec_dim;
            file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int32_t));
            if (!file || file.eof()) break;
            file.seekg(vec_dim, std::ios::cur);
        }
        if (show_progress && skip > 0) {
            std::cout << " done\n";
        }

        // Read vectors
        int64_t count = 0;

        while (true) {
            if (max_vectors > 0 && count >= max_vectors) break;
            int32_t vec_dim;
            file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int32_t));
            if (!file || file.eof()) break;
            std::vector<uint8_t> bvec(vec_dim);
            file.read(reinterpret_cast<char*>(bvec.data()), vec_dim);
            // Convert uint8_t to float
            std::vector<float> vec(vec_dim);
            for (int i = 0; i < vec_dim; i++) {
                vec[i] = static_cast<float>(bvec[i]);
            }
            vectors.push_back(std::move(vec));
            count++;
        }
        if (show_progress) {
            std::cout << "\n";
        }
    }
};

// fbin format: 8-byte header (n:int32, d:int32 little-endian), then n*d float32 contiguous
struct FbinReader {
    std::ifstream file;
    int dim;
    int64_t num_vectors;
    int64_t data_offset;  // byte offset where vector data starts (after header)

    FbinReader(const std::string& filename) : file(filename, std::ios::binary) {
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

    void read_all_vectors(std::vector<std::vector<float>>& vectors, int64_t skip = 0, int64_t max_vectors = -1, bool show_progress = true) {
        vectors.clear();
        // max_vectors < 0: read all remaining; max_vectors >= 0: read up to max_vectors (0 means read none)
        int64_t to_read = (max_vectors < 0)
            ? (num_vectors - skip)
            : std::min(max_vectors, num_vectors - skip);
        if (to_read <= 0) return;

        if (show_progress && skip > 0) {
            std::cout << "Skipping " << skip << " vectors...";
            std::cout.flush();
        }
        int64_t start_offset = data_offset + skip * static_cast<int64_t>(dim) * sizeof(float);
        file.seekg(start_offset, std::ios::beg);

        if (show_progress && skip > 0) {
            std::cout << " done\n";
        }
        if (show_progress) {
            std::cout << "Loading " << to_read << " vectors...";
            std::cout.flush();
        }

        for (int64_t i = 0; i < to_read; i++) {
            std::vector<float> vec(dim);
            file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
            if (!file) break;
            vectors.push_back(std::move(vec));
        }
        if (show_progress) {
            std::cout << " " << vectors.size() << " loaded\n";
        }
    }
};

// Statistics structure (same as in knowhere_concurrent_query.cpp, but used only for timing/count)
struct QueryStats {
    std::atomic<int64_t> query_count{0};
    std::atomic<double> total_time{0.0};
    std::atomic<int> completed_threads{0};
    double query_time{0.0};  // Wall-clock time for the entire query run
    std::mutex print_mutex;

    void add_query(double elapsed) {
        query_count.fetch_add(1, std::memory_order_relaxed);
        double old_val = total_time.load(std::memory_order_relaxed);
        while (!total_time.compare_exchange_weak(old_val, old_val + elapsed,
                                                 std::memory_order_relaxed)) {
            // Retry on failure
        }
    }
};

// Random-query worker function:
//   - Loops until stop_flag is set.
//   - On each iteration, picks a random query vector and runs a search.
//   - Does NOT store per-query results; only aggregates timing statistics.
void query_worker_random(
    knowhere::Index<knowhere::IndexNode>* index,
    const std::vector<std::vector<float>>& query_vectors,
    QueryStats& stats,
    int thread_id,
    int k,
    int ef_search,
    int nprobe,
    bool is_hnsw,
    std::atomic<bool>& stop_flag
) {
    if (query_vectors.empty()) {
        return;
    }

    knowhere::BitsetView bitset_view(nullptr);

    // Per-thread RNG
    auto seed = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    ) ^ (static_cast<uint64_t>(thread_id) + 0x9e3779b97f4a7c15ULL);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> dist(0, static_cast<int64_t>(query_vectors.size()) - 1);

    while (!stop_flag.load(std::memory_order_relaxed)) {
        int64_t query_idx = dist(rng);
        const std::vector<float>& qvec = query_vectors[query_idx];
        const float* query_vec = qvec.data();
        int dim = static_cast<int>(qvec.size());

        // Configure search
        knowhere::Json conf;
        conf[knowhere::meta::TOPK] = k;
        conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        if (is_hnsw) {
            conf[knowhere::indexparam::EF] = ef_search;
        } else {
            conf[knowhere::indexparam::NPROBE] = nprobe;
        }

        // Generate query dataset
        auto dataset = knowhere::GenDataSet(1, dim, query_vec);

        // Perform search
        auto start_time = std::chrono::high_resolution_clock::now();
        (void)index->Search(dataset, conf, bitset_view);
        auto end_time = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(end_time - start_time).count();
        stats.add_query(elapsed);
    }

    stats.completed_threads++;
}

int main(int argc, char* argv[]) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0] << " <base_file> <query_file> <ground_truth_file_unused> "
                  << "<index_type> <num_threads> <k> <ef_search/nprobe/search_list_size> [options]\n"
                  << "  base_file: Path to .fvecs, .bvecs, or .fbin file for building/loading index\n"
                  << "  query_file: Path to .fvecs, .bvecs, or .fbin file for queries\n"
                  << "  ground_truth_file_unused: Placeholder to keep CLI consistent; ignored\n"
                  << "  index_type: 'hnsw' or 'ivfflat'\n"
                  << "  num_threads: Number of concurrent query threads\n"
                  << "  k: Top-k for search\n"
                  << "  ef_search/nprobe: ef_search for HNSW, nprobe for IVFFLAT (single value or comma-separated list)\n"
                  << "Options:\n"
                  << "  Format is auto-detected from file extension (.fvecs, .bvecs, .fbin).\n"
                  << "  --use-bvecs: Override to .bvecs format\n"
                  << "  --use-fbin: Override to .fbin format\n"
                  << "  --M <value>: HNSW M parameter (default: 16)\n"
                  << "  --ef-construction <value>: HNSW ef_construction (default: 64)\n"
                  << "  --nlist <value>: IVFFLAT nlist parameter (default: 100)\n"
                  << "  --base-skip <value>: Skip N vectors from base file before reading (default: 0)\n"
                  << "  --base-num <value>: Number of vectors to read from base file (default: all)\n"
                  << "  --query-skip <value>: Skip N queries from query file before reading (default: 0)\n"
                  << "  --query-num <value>: Number of queries to read from query file (default: all)\n"
                  << "  --save-index <path>: Save the built index to file (optional)\n"
                  << "  --load-index <path>: Load index from file instead of building (optional)\n"
                  << "  --duration <seconds>: Total wall-clock time to run queries (default: 10.0)\n"
                  << "  --max-vectors <value>: [Deprecated] Use --base-num instead\n"
                  << "  --max-queries <value>: [Deprecated] Use --query-num instead\n";
        return 1;
    }

    std::string base_file = argv[1];
    std::string query_file = argv[2];
    // Ground truth file is intentionally ignored in this random test
    // to keep the CLI consistent with knowhere_concurrent_query.cpp.
    std::string ground_truth_file_unused = argv[3];
    (void)ground_truth_file_unused;
    std::string index_type = argv[4];
    int num_threads = std::stoi(argv[5]);
    int k = std::stoi(argv[6]);
    std::string ef_search_nprobe_str = argv[7];

    // Parse ef_search/nprobe/search_list_size list (comma-separated or single value)
    std::vector<int> ef_search_nprobe_list;
    std::stringstream ss(ef_search_nprobe_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // Trim whitespace
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        if (!item.empty()) {
            ef_search_nprobe_list.push_back(std::stoi(item));
        }
    }

    if (ef_search_nprobe_list.empty()) {
        std::cerr << "Error: At least one ef_search/nprobe value must be specified\n";
        return 1;
    }

    bool use_bvecs = false;
    bool use_fbin = false;
    VectorFileFormat base_format;
    VectorFileFormat query_format;
    int M = 16;
    int ef_construction = 40;
    int nlist = 100;
    int64_t base_skip = 0;
    int64_t base_num = -1;
    int64_t query_skip = 0;
    int64_t query_num = -1;
    int64_t max_vectors = -1;  // Deprecated, for backward compatibility
    int64_t max_queries = -1;  // Deprecated, for backward compatibility
    std::string save_index_path;
    std::string load_index_path;
    double duration_seconds = 10.0;  // Default run duration

    // Parse optional arguments
    for (int i = 8; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--use-bvecs") {
            use_bvecs = true;
        } else if (arg == "--use-fbin") {
            use_fbin = true;
        } else if (arg == "--M" && i + 1 < argc) {
            M = std::stoi(argv[++i]);
        } else if (arg == "--ef-construction" && i + 1 < argc) {
            ef_construction = std::stoi(argv[++i]);
        } else if (arg == "--nlist" && i + 1 < argc) {
            nlist = std::stoi(argv[++i]);
        } else if (arg == "--base-skip" && i + 1 < argc) {
            base_skip = std::stoll(argv[++i]);
        } else if (arg == "--base-num" && i + 1 < argc) {
            base_num = std::stoll(argv[++i]);
        } else if (arg == "--query-skip" && i + 1 < argc) {
            query_skip = std::stoll(argv[++i]);
        } else if (arg == "--query-num" && i + 1 < argc) {
            query_num = std::stoll(argv[++i]);
        } else if (arg == "--save-index" && i + 1 < argc) {
            save_index_path = argv[++i];
        } else if (arg == "--load-index" && i + 1 < argc) {
            load_index_path = argv[++i];
        } else if (arg == "--duration" && i + 1 < argc) {
            duration_seconds = std::stod(argv[++i]);
        } else if (arg == "--max-vectors" && i + 1 < argc) {
            max_vectors = std::stoll(argv[++i]);
            // For backward compatibility, set base_num if not already set
            if (base_num == -1) {
                base_num = max_vectors;
            }
        } else if (arg == "--max-queries" && i + 1 < argc) {
            max_queries = std::stoll(argv[++i]);
            // For backward compatibility, set query_num if not already set
            if (query_num == -1) {
                query_num = max_queries;
            }
        }
    }

    if (duration_seconds <= 0.0) {
        std::cerr << "Error: --duration must be positive\n";
        return 1;
    }

    // Auto-detect format from file extension (or use --use-bvecs/--use-fbin override)
    if (use_fbin) {
        base_format = VectorFileFormat::Fbin;
        query_format = VectorFileFormat::Fbin;
    } else if (use_bvecs) {
        base_format = VectorFileFormat::Bvecs;
        query_format = VectorFileFormat::Bvecs;
    } else {
        base_format = detect_vector_format(base_file);
        query_format = detect_vector_format(query_file);
    }

    bool is_hnsw = (index_type == "hnsw");
    bool is_ivfflat = (index_type == "ivfflat");

    if (!is_hnsw && !is_ivfflat) {
        std::cerr << "Error: index_type must be 'hnsw' or 'ivfflat'\n";
        return 1;
    }

    std::cout << "=== Knowhere Concurrent Random Query Test ===\n";
    std::cout << "Base file: " << base_file << "\n";
    std::cout << "Query file: " << query_file << "\n";
    std::cout << "Index type: " << index_type << "\n";
    std::cout << "Number of threads: " << num_threads << "\n";
    std::cout << "Top-k: " << k << "\n";
    if (is_hnsw) {
        std::cout << "ef_search values: ";
        for (size_t i = 0; i < ef_search_nprobe_list.size(); i++) {
            std::cout << ef_search_nprobe_list[i];
            if (i < ef_search_nprobe_list.size() - 1) std::cout << ", ";
        }
        std::cout << "\n";
        std::cout << "M: " << M << "\n";
        std::cout << "ef_construction: " << ef_construction << "\n";
    } else {
        std::cout << "nprobe values: ";
        for (size_t i = 0; i < ef_search_nprobe_list.size(); i++) {
            std::cout << ef_search_nprobe_list[i];
            if (i < ef_search_nprobe_list.size() - 1) std::cout << ", ";
        }
        std::cout << "\n";
        std::cout << "nlist: " << nlist << "\n";
    }
    std::cout << "Base file format: " << format_extension(base_format) << "\n";
    std::cout << "Query file format: " << format_extension(query_format) << "\n";
    if (base_skip > 0 || base_num > 0) {
        std::cout << "Base vectors: skip=" << base_skip;
        if (base_num > 0) {
            std::cout << ", num=" << base_num;
        } else {
            std::cout << ", num=all";
        }
        std::cout << "\n";
    }
    if (query_skip > 0 || query_num > 0) {
        std::cout << "Query vectors: skip=" << query_skip;
        if (query_num > 0) {
            std::cout << ", num=" << query_num;
        } else {
            std::cout << ", num=all";
        }
        std::cout << "\n";
    }
    if (!load_index_path.empty()) {
        std::cout << "Load index from: " << load_index_path << "\n";
    }
    if (!save_index_path.empty()) {
        std::cout << "Save index to: " << save_index_path << "\n";
    }
    std::cout << "Run duration: " << duration_seconds << " seconds per setting\n";
    std::cout << "\n";

    // Step 1: Get dimension (skip if loading index)
    int dim = 0;
    int64_t total_vectors_to_load = 0;

    if (load_index_path.empty()) {
        // Get dimension and total vectors count
        std::cout << "Step 1: Preparing to load base vectors...\n";

        try {
            if (base_format == VectorFileFormat::Fbin) {
                FbinReader reader(base_file);
                dim = reader.dim;
                total_vectors_to_load = (base_num > 0) ? std::min(base_num, reader.num_vectors) : reader.num_vectors;
                std::cout << "Dimension: " << dim << "\n";
                std::cout << "Total vectors in file: " << reader.num_vectors << "\n";
                std::cout << "Will load " << total_vectors_to_load << " vectors";
                if (base_skip > 0) {
                    std::cout << " (skipping first " << base_skip << ")";
                }
                std::cout << "\n";
            } else if (base_format == VectorFileFormat::Bvecs) {
                BvecsReader reader(base_file);
                dim = reader.dim;
                total_vectors_to_load = (base_num > 0) ? base_num : -1;  // -1 means read until EOF
                std::cout << "Dimension: " << dim << "\n";
                std::cout << "Will load " << (total_vectors_to_load > 0 ? std::to_string(total_vectors_to_load) : "all remaining") << " vectors";
                if (base_skip > 0) {
                    std::cout << " (skipping first " << base_skip << ")";
                }
                std::cout << "\n";
            } else {
                FvecsReader reader(base_file);
                dim = reader.dim;
                total_vectors_to_load = (base_num > 0) ? base_num : -1;  // -1 means read until EOF
                std::cout << "Dimension: " << dim << "\n";
                std::cout << "Will load " << (total_vectors_to_load > 0 ? std::to_string(total_vectors_to_load) : "all remaining") << " vectors";
                if (base_skip > 0) {
                    std::cout << " (skipping first " << base_skip << ")";
                }
                std::cout << "\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Error reading base file: " << e.what() << "\n";
            return 1;
        }
        std::cout << "\n";
    } else {
        // When loading index, we'll get dimension from the index
        std::cout << "Step 1: Skipping base vector loading (will load index instead)\n\n";
    }

    // Step 2: Build or load index
    knowhere::Index<knowhere::IndexNode> index;
    bool index_loaded = false;

    if (!load_index_path.empty()) {
        bool index_file_valid = false;
        bool should_build_index = false;

        if (fs::exists(load_index_path)) {
            auto file_size = fs::file_size(load_index_path);
            if (file_size > 0) {
                index_file_valid = true;
                std::cout << "Step 2: Loading index from file...\n";
                std::cout << "Index file: " << load_index_path << " (size: " << file_size << " bytes)\n";
            } else {
                std::cout << "WARNING: Index file exists but is empty (0 bytes): " << load_index_path << "\n";
                std::cout << "Will build index from base file instead.\n\n";
                should_build_index = true;
            }
        } else {
            std::cout << "WARNING: Index file does not exist: " << load_index_path << "\n";
            std::cout << "Will build index from base file instead.\n\n";
            should_build_index = true;
        }

        if (index_file_valid) {
            auto start_load = std::chrono::high_resolution_clock::now();

            auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
            knowhere::Index<knowhere::IndexNode> loaded_index;

            if (is_hnsw) {
                auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
                    knowhere::IndexEnum::INDEX_HNSW, version);
                if (!idx.has_value()) {
                    std::cerr << "Error creating HNSW index\n";
                    return 1;
                }
                loaded_index = idx.value();
            } else {
                auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
                    knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, version);
                if (!idx.has_value()) {
                    std::cerr << "Error creating IVFFLAT index\n";
                    return 1;
                }
                loaded_index = idx.value();
            }

            knowhere::Json conf;
            conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;

            try {
                read_index(loaded_index, load_index_path, conf);
                index = loaded_index;
                index_loaded = true;

                // Get dimension from loaded index
                dim = index.Dim();

                auto end_load = std::chrono::high_resolution_clock::now();
                double load_time = std::chrono::duration<double>(end_load - start_load).count();
                std::cout << "Index loaded in " << std::fixed << std::setprecision(4) << load_time << " seconds\n";
                std::cout << "Index dimension: " << dim << "\n";
                std::cout << "Index count: " << index.Count() << "\n";

                if (index.Count() <= 0) {
                    std::cout << "WARNING: Index file exists but index is empty after loading!\n";
                    std::cout << "Will build index from base file instead.\n\n";
                    should_build_index = true;
                    index_loaded = false;
                } else {
                    std::cout << "\n";
                }
            } catch (const std::exception& e) {
                std::cout << "WARNING: Error loading index: " << e.what() << "\n";
                std::cout << "Will build index from base file instead.\n\n";
                should_build_index = true;
                index_loaded = false;
            }
        }

        if (should_build_index) {
            if (base_file.empty()) {
                std::cerr << "ERROR: Cannot build index - base file not provided!\n";
                return 1;
            }

            if (dim == 0) {
                try {
                    if (base_format == VectorFileFormat::Fbin) {
                        FbinReader reader(base_file);
                        dim = reader.dim;
                    } else if (base_format == VectorFileFormat::Bvecs) {
                        BvecsReader reader(base_file);
                        dim = reader.dim;
                    } else {
                        FvecsReader reader(base_file);
                        dim = reader.dim;
                    }
                    std::cout << "Dimension from base file: " << dim << "\n";
                } catch (const std::exception& e) {
                    std::cerr << "Error reading base file to get dimension: " << e.what() << "\n";
                    return 1;
                }
            }

            load_index_path.clear();
        }
    }

    // Build index if not loaded
    if (load_index_path.empty() || !index_loaded) {
        std::cout << "Step 2: Building index in batches...\n";
        auto start_build = std::chrono::high_resolution_clock::now();

        const int64_t batch_size = 100000;
        int64_t total_loaded = 0;
        int64_t total_added = 0;

        auto version = knowhere::Version::GetCurrentVersion().VersionNumber();

        // Create index
        if (is_hnsw) {
            auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
                knowhere::IndexEnum::INDEX_HNSW, version);
            if (!idx.has_value()) {
                std::cerr << "Error creating HNSW index\n";
                return 1;
            }
            index = idx.value();
        } else {
            auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
                knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, version);
            if (!idx.has_value()) {
                std::cerr << "Error creating IVFFLAT index\n";
                return 1;
            }
            index = idx.value();
        }

        knowhere::Json conf;
        conf[knowhere::meta::DIM] = dim;
        conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;

        if (is_hnsw) {
            conf[knowhere::indexparam::M] = M;
            conf[knowhere::indexparam::EFCONSTRUCTION] = ef_construction;
        } else {
            conf[knowhere::indexparam::NLIST] = nlist;
        }

        {
            std::unique_ptr<FvecsReader> fvecs_reader;
            std::unique_ptr<BvecsReader> bvecs_reader;
            std::unique_ptr<FbinReader> fbin_reader;

            if (base_format == VectorFileFormat::Fbin) {
                fbin_reader = std::make_unique<FbinReader>(base_file);
            } else if (base_format == VectorFileFormat::Bvecs) {
                bvecs_reader = std::make_unique<BvecsReader>(base_file);
            } else {
                fvecs_reader = std::make_unique<FvecsReader>(base_file);
            }

            // For IVFFLAT, train first on a sample
            if (is_ivfflat) {
                int64_t train_samples = std::min(static_cast<int64_t>(50) * static_cast<int64_t>(nlist), total_vectors_to_load);
                if (train_samples > 0) {
                    std::cout << "Training IVFFLAT index on " << train_samples << " samples...";
                    std::cout.flush();

                    std::vector<std::vector<float>> train_vectors;
                    if (base_format == VectorFileFormat::Fbin) {
                        fbin_reader->read_all_vectors(train_vectors, base_skip, train_samples, false);
                    } else if (base_format == VectorFileFormat::Bvecs) {
                        bvecs_reader->read_all_vectors(train_vectors, base_skip, train_samples, false);
                    } else {
                        fvecs_reader->read_all_vectors(train_vectors, base_skip, train_samples, false);
                    }

                    std::vector<float> train_data(train_samples * dim);
                    for (int64_t i = 0; i < train_samples; i++) {
                        std::memcpy(train_data.data() + i * dim, train_vectors[i].data(), dim * sizeof(float));
                    }

                    auto train_dataset = knowhere::GenDataSet(train_samples, dim, train_data.data());
                    knowhere::Json train_conf;
                    train_conf[knowhere::meta::ROWS] = train_samples;
                    train_conf[knowhere::meta::DIM] = dim;
                    train_conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
                    train_conf[knowhere::indexparam::NLIST] = nlist;
                    auto train_res = index.Train(train_dataset, train_conf);
                    if (train_res != knowhere::Status::success) {
                        std::cerr << "Error training IVFFLAT index\n";
                        return 1;
                    }
                    std::cout << " done\n";
                }
            }

            if (total_vectors_to_load == 0) {
                if (base_num > 0) {
                    total_vectors_to_load = base_num;
                } else {
                    total_vectors_to_load = -1;
                }
            }

            std::cout << "Loading and adding vectors in batches of " << batch_size << ":\n";
            std::cout << "Target: " << (total_vectors_to_load > 0 ? std::to_string(total_vectors_to_load) : "all") << " vectors\n";
            std::cout << "Progress: ";
            std::cout.flush();

            int last_reported_pct = -1;
            int64_t progress_interval = 0;
            if (total_vectors_to_load > 0) {
                progress_interval = std::max(static_cast<int64_t>(1), total_vectors_to_load / 100);
            }

            bool first_batch_processed = false;
            int batch_number = 0;

            bool loop_condition = (total_vectors_to_load < 0 || total_loaded < total_vectors_to_load);
            if (!loop_condition) {
                std::cerr << "ERROR: Build loop will not execute!\n";
                std::cerr << "  total_vectors_to_load: " << total_vectors_to_load << "\n";
                std::cerr << "  total_loaded: " << total_loaded << "\n";
                return 1;
            }

            while (total_vectors_to_load < 0 || total_loaded < total_vectors_to_load) {
                batch_number++;
                int64_t current_batch_size;
                if (total_vectors_to_load > 0) {
                    current_batch_size = std::min(batch_size, total_vectors_to_load - total_loaded);
                } else {
                    current_batch_size = batch_size;
                }
                int64_t current_skip = base_skip + total_loaded;

                std::vector<std::vector<float>> batch_vectors;
                if (base_format == VectorFileFormat::Fbin) {
                    fbin_reader->read_all_vectors(batch_vectors, current_skip, current_batch_size, false);
                } else if (base_format == VectorFileFormat::Bvecs) {
                    bvecs_reader->read_all_vectors(batch_vectors, current_skip, current_batch_size, false);
                } else {
                    fvecs_reader->read_all_vectors(batch_vectors, current_skip, current_batch_size, false);
                }

                if (batch_vectors.empty()) {
                    if (total_vectors_to_load < 0) {
                        if (batch_number == 1) {
                            std::cerr << "\nERROR: First batch is empty! Base file appears to be empty or unreadable.\n";
                            std::cerr << "Base file: " << base_file << "\n";
                            return 1;
                        }
                        break;
                    } else {
                        std::cerr << "\nError: Reached EOF before loading all requested vectors\n";
                        std::cerr << "Loaded: " << total_loaded << ", Expected: " << total_vectors_to_load << "\n";
                        return 1;
                    }
                }

                total_loaded += batch_vectors.size();

                int64_t batch_count = static_cast<int64_t>(batch_vectors.size());
                std::vector<float> flat_batch(batch_count * dim);
                for (int64_t i = 0; i < batch_count; i++) {
                    std::memcpy(flat_batch.data() + i * dim, batch_vectors[i].data(), dim * sizeof(float));
                }

                auto batch_dataset = knowhere::GenDataSet(batch_count, dim, flat_batch.data());
                conf[knowhere::meta::ROWS] = batch_count;

                if (index.Count() == -1 || index.Count() == 0) {
                    if (!first_batch_processed) {
                        std::cout << "\nBuilding index with first batch (" << batch_count << " vectors)...";
                        std::cout.flush();
                        first_batch_processed = true;
                    }
                    auto build_res = index.Build(batch_dataset, conf);
                    if (build_res != knowhere::Status::success) {
                        std::cerr << "\nError building index with first batch (status: " << static_cast<int>(build_res) << ")\n";
                        return 1;
                    }
                    int64_t after_build_count = index.Count();
                    if (first_batch_processed && after_build_count > 0) {
                        std::cout << " done (index count: " << after_build_count << ")\n";
                        std::cout << "Progress: ";
                        std::cout.flush();
                    } else if (after_build_count <= 0) {
                        std::cerr << "\nWARNING: Index count is " << after_build_count << " after build! This is unexpected.\n";
                    }
                } else {
                    auto add_res = index.Add(batch_dataset, conf);
                    if (add_res != knowhere::Status::success) {
                        std::cerr << "\nError adding batch to index\n";
                        return 1;
                    }
                }

                total_added += batch_count;

                if (total_vectors_to_load > 0) {
                    int current_pct = (total_added * 100) / total_vectors_to_load;
                    if (current_pct > last_reported_pct) {
                        std::cout << total_added << "/" << total_vectors_to_load << " (" << current_pct << "%) ";
                        std::cout.flush();
                        last_reported_pct = current_pct;
                    }
                } else {
                    if (progress_interval > 0 && total_added % progress_interval == 0) {
                        std::cout << total_added << " ";
                        std::cout.flush();
                    } else if (progress_interval == 0) {
                        progress_interval = std::max(static_cast<int64_t>(1), batch_size);
                    }
                }
            }

            std::cout << "\n";

            if (total_added == 0) {
                std::cerr << "ERROR: No vectors were added to the index!\n";
                return 1;
            }

            auto end_build = std::chrono::high_resolution_clock::now();
            double build_time = std::chrono::duration<double>(end_build - start_build).count();
            std::cout << "Index built in " << std::fixed << std::setprecision(4) << build_time << " seconds\n";

            int64_t index_count = index.Count();
            std::cout << "Index count: " << index_count << "\n";
            std::cout << "Total vectors added: " << total_added << "\n";
            std::cout << "Index dimension: " << index.Dim() << "\n";

            if (index.Count() <= 0 || total_added == 0) {
                std::cerr << "ERROR: Index is empty after building! Index count is " << index.Count()
                          << ", total vectors added: " << total_added << "\n";
                return 1;
            }

            if (!save_index_path.empty()) {
                std::cout << "Saving index to: " << save_index_path << "\n";
                try {
                    write_index(index, save_index_path);
                    std::cout << "Index saved successfully\n";
                } catch (const std::exception& e) {
                    std::cerr << "Error saving index: " << e.what() << "\n";
                    return 1;
                }
            }
            std::cout << "\n";
        }
    }

    // Step 3: Load queries
    std::cout << "Step 3: Loading queries...\n";
    std::vector<std::vector<float>> query_vectors;

    try {
        if (query_format == VectorFileFormat::Fbin) {
            FbinReader reader(query_file);
            if (query_skip > 0) {
                std::cout << "Skipping first " << query_skip << " queries\n";
            }
            reader.read_all_vectors(query_vectors, query_skip, query_num);
        } else if (query_format == VectorFileFormat::Bvecs) {
            BvecsReader reader(query_file);
            if (query_skip > 0) {
                std::cout << "Skipping first " << query_skip << " queries\n";
            }
            reader.read_all_vectors(query_vectors, query_skip, query_num);
        } else {
            FvecsReader reader(query_file);
            if (query_skip > 0) {
                std::cout << "Skipping first " << query_skip << " queries\n";
            }
            reader.read_all_vectors(query_vectors, query_skip, query_num);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading queries: " << e.what() << "\n";
        return 1;
    }

    if (query_vectors.empty()) {
        std::cerr << "ERROR: No queries loaded from query file; cannot run random queries.\n";
        return 1;
    }

    std::cout << "Loaded " << query_vectors.size() << " queries\n\n";

    // Validate index before running queries
    if (index.Count() <= 0) {
        std::cerr << "ERROR: Cannot run queries - index is empty! Index count: " << index.Count() << "\n";
        return 1;
    }
    std::cout << "Index validation: Index contains " << index.Count() << " vectors, ready for queries\n\n";

    // Step 4: Run random queries for each ef_search/nprobe value
    std::cout << "Step 4: Running RANDOM queries for " << ef_search_nprobe_list.size() << " different "
              << (is_hnsw ? "ef_search" : "nprobe") << " value(s)...\n\n";

    struct StatsSnapshot {
        int64_t query_count;
        double total_time;
        double query_time;
    };
    std::vector<std::pair<int, StatsSnapshot>> all_stats;

    for (size_t ef_idx = 0; ef_idx < ef_search_nprobe_list.size(); ef_idx++) {
        int ef_search_nprobe = ef_search_nprobe_list[ef_idx];

        std::cout << "========================================\n";
        std::cout << "Running RANDOM queries with " << (is_hnsw ? "ef_search" : "nprobe")
                  << " = " << ef_search_nprobe << " (" << (ef_idx + 1) << "/"
                  << ef_search_nprobe_list.size() << ")\n";
        std::cout << "========================================\n";

        QueryStats stats;
        std::atomic<bool> stop_flag(false);

        std::vector<std::thread> threads;
        auto start_query = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_threads; i++) {
            threads.emplace_back(query_worker_random,
                                 &index,
                                 std::cref(query_vectors),
                                 std::ref(stats),
                                 i,
                                 k,
                                 ef_search_nprobe,
                                 ef_search_nprobe,
                                 is_hnsw,
                                 std::ref(stop_flag));
        }

        // Let threads run for the requested duration
        std::this_thread::sleep_for(std::chrono::duration<double>(duration_seconds));
        stop_flag.store(true, std::memory_order_relaxed);

        // Wait for threads to finish
        for (auto& t : threads) {
            t.join();
        }

        auto end_query = std::chrono::high_resolution_clock::now();
        double query_time = std::chrono::duration<double>(end_query - start_query).count();

        std::cout << "\nRandom query run completed in " << std::fixed << std::setprecision(4)
                  << query_time << " seconds (target duration " << duration_seconds << " seconds)\n";
        std::cout << "Total queries: " << stats.query_count.load() << "\n";
        std::cout << "Total query time (sum over queries): " << std::fixed << std::setprecision(4)
                  << stats.total_time.load() << " seconds\n";
        if (stats.query_count.load() > 0) {
            std::cout << "Average query time: " << std::fixed << std::setprecision(6)
                      << stats.total_time.load() / stats.query_count.load() << " seconds\n";
            std::cout << "Query throughput (wall-clock): " << std::fixed << std::setprecision(2)
                      << stats.query_count.load() / query_time << " queries/second\n";
        }
        std::cout << "\n";

        stats.query_time = query_time;

        StatsSnapshot snapshot;
        snapshot.query_count = stats.query_count.load();
        snapshot.total_time = stats.total_time.load();
        snapshot.query_time = stats.query_time;
        all_stats.push_back({ef_search_nprobe, snapshot});
    }

    // Step 5: Summary for all ef_search/nprobe values
    std::cout << "\n========================================\n";
    std::cout << "SUMMARY FOR ALL " << (is_hnsw ? "EF_SEARCH" : "NPROBE") << " VALUES (CONCURRENCY = " << num_threads << ")\n";
    std::cout << "========================================\n";
    std::cout << std::left << std::setw(12) << (is_hnsw ? "ef_search" : "nprobe")
              << std::setw(18) << "Avg Query Time"
              << std::setw(18) << "Throughput (q/s)"
              << std::setw(18) << "Total Queries" << "\n";
    std::cout << std::string(66, '-') << "\n";

    for (size_t i = 0; i < all_stats.size(); i++) {
        int ef_val = all_stats[i].first;
        StatsSnapshot& stats = all_stats[i].second;

        double avg_time = 0.0;
        double throughput = 0.0;
        if (stats.query_count > 0) {
            avg_time = stats.total_time / stats.query_count;
            if (stats.query_time > 0.0) {
                throughput = stats.query_count / stats.query_time;
            }
        }

        std::cout << std::left << std::setw(12) << ef_val
                  << std::setw(18) << std::fixed << std::setprecision(6) << avg_time
                  << std::setw(18) << std::setprecision(2) << throughput
                  << std::setw(18) << stats.query_count << "\n";
    }
    std::cout << "\n";

    return 0;
}

