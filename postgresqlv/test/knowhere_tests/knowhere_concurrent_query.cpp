#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <set>
#include <sstream>
#include <memory>

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

// Index serialization helpers
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

// File format structures
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
    
    std::vector<float> read_vector(int64_t index) {
        file.seekg(0, std::ios::beg);
        for (int64_t i = 0; i < index; i++) {
            int32_t vec_dim;
            file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int32_t));
            file.seekg(vec_dim * sizeof(float), std::ios::cur);
        }
        int32_t vec_dim;
        file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int32_t));
        std::vector<float> vec(vec_dim);
        file.read(reinterpret_cast<char*>(vec.data()), vec_dim * sizeof(float));
        return vec;
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
    
    std::vector<float> read_vector(int64_t index) {
        file.seekg(0, std::ios::beg);
        for (int64_t i = 0; i < index; i++) {
            int32_t vec_dim;
            file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int32_t));
            file.seekg(vec_dim, std::ios::cur);
        }
        int32_t vec_dim;
        file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int32_t));
        std::vector<uint8_t> bvec(vec_dim);
        file.read(reinterpret_cast<char*>(bvec.data()), vec_dim);
        // Convert uint8_t to float
        std::vector<float> vec(vec_dim);
        for (int i = 0; i < vec_dim; i++) {
            vec[i] = static_cast<float>(bvec[i]);
        }
        return vec;
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
        int64_t total_to_read = (max_vectors > 0) ? max_vectors : num_vectors;
        int64_t progress_interval = std::max(static_cast<int64_t>(1), total_to_read / 20);  // Report every 5%
        
        if (show_progress) {
            std::cout << "Loading vectors: ";
            std::cout.flush();
        }
        
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

struct IvecsReader {
    std::ifstream file;
    int k;
    int64_t num_queries;
    
    IvecsReader(const std::string& filename, int expected_k) : k(expected_k) {
        file.open(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        // Count queries (allow file to have more results than expected_k)
        num_queries = 0;
        while (true) {
            int32_t vec_k;
            file.read(reinterpret_cast<char*>(&vec_k), sizeof(int32_t));
            if (!file || file.eof()) break;
            // Allow file to have more results than expected_k, but warn if less
            if (vec_k < expected_k) {
                throw std::runtime_error("Ground truth file has k=" + std::to_string(vec_k) + 
                                       " but need at least k=" + std::to_string(expected_k) + 
                                       " at query " + std::to_string(num_queries));
            }
            file.seekg(vec_k * sizeof(int32_t), std::ios::cur);
            num_queries++;
        }
        file.clear();
        file.seekg(0, std::ios::beg);
    }
    
    std::vector<int32_t> read_ground_truth(int64_t query_index) {
        file.seekg(0, std::ios::beg);
        for (int64_t i = 0; i < query_index; i++) {
            int32_t vec_k;
            file.read(reinterpret_cast<char*>(&vec_k), sizeof(int32_t));
            file.seekg(vec_k * sizeof(int32_t), std::ios::cur);
        }
        int32_t vec_k;
        file.read(reinterpret_cast<char*>(&vec_k), sizeof(int32_t));
        std::vector<int32_t> gt(vec_k);
        file.read(reinterpret_cast<char*>(gt.data()), vec_k * sizeof(int32_t));
        return gt;
    }
    
    void read_all_ground_truth(std::vector<std::vector<int32_t>>& ground_truth, int64_t skip = 0, int64_t max_queries = -1) {
        file.seekg(0, std::ios::beg);
        ground_truth.clear();
        
        // Skip queries
        for (int64_t i = 0; i < skip; i++) {
            int32_t vec_k;
            file.read(reinterpret_cast<char*>(&vec_k), sizeof(int32_t));
            if (!file || file.eof()) break;
            file.seekg(vec_k * sizeof(int32_t), std::ios::cur);
        }
        
        // Read ground truth
        int64_t count = 0;
        while (true) {
            if (max_queries > 0 && count >= max_queries) break;
            int32_t vec_k;
            file.read(reinterpret_cast<char*>(&vec_k), sizeof(int32_t));
            if (!file || file.eof()) break;
            std::vector<int32_t> gt(vec_k);
            file.read(reinterpret_cast<char*>(gt.data()), vec_k * sizeof(int32_t));
            // Only keep the first k results (even if file has more)
            if (static_cast<int>(gt.size()) > k) {
                gt.resize(k);
            }
            ground_truth.push_back(std::move(gt));
            count++;
        }
    }
};

// Statistics structure
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

// Query worker function
void query_worker(
    knowhere::Index<knowhere::IndexNode>* index,
    std::queue<std::pair<int64_t, std::vector<float>>>& query_queue,
    std::mutex& queue_mutex,
    std::vector<std::vector<int64_t>>& results,
    std::mutex& results_mutex,
    QueryStats& stats,
    int thread_id,
    int k,
    int ef_search,
    int nprobe,
    bool is_hnsw,
    std::atomic<bool>& stop_flag
) {
    knowhere::BitsetView bitset_view(nullptr);
    
    while (true) {
        if (stop_flag.load()) {
            // Check if queue is empty
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (query_queue.empty()) break;
        }
        
        std::pair<int64_t, std::vector<float>> query_item;
        bool got_item = false;
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (!query_queue.empty()) {
                query_item = query_queue.front();
                query_queue.pop();
                got_item = true;
            }
        }
        
        if (!got_item) {
            if (stop_flag.load()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        int64_t query_idx = query_item.first;
        const float* query_vec = query_item.second.data();
        int dim = static_cast<int>(query_item.second.size());
        
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
        auto res = index->Search(dataset, conf, bitset_view);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        double elapsed = std::chrono::duration<double>(end_time - start_time).count();
        stats.add_query(elapsed);
        
        // Extract results
        if (res.has_value()) {
            const int64_t* ids = res.value()->GetIds();
            const float* distances = res.value()->GetDistance();
            
            // Knowhere search returns exactly k results (or fewer if index has fewer vectors)
            // Since we requested k results and search succeeded, we should have results
            if (ids == nullptr || distances == nullptr) {
                // Search returned null pointers - add empty result
                std::lock_guard<std::mutex> lock(results_mutex);
                if (query_idx >= static_cast<int64_t>(results.size())) {
                    results.resize(query_idx + 1);
                }
                results[query_idx] = std::vector<int64_t>();  // Empty result
            } else {
                // Use k as the result count (we requested k results)
                std::vector<int64_t> result_ids(ids, ids + k);
                
                std::lock_guard<std::mutex> lock(results_mutex);
                if (query_idx >= static_cast<int64_t>(results.size())) {
                    results.resize(query_idx + 1);
                }
                results[query_idx] = std::move(result_ids);
            }
        } else {
            // Search failed - add empty result
            std::lock_guard<std::mutex> lock(results_mutex);
            if (query_idx >= static_cast<int64_t>(results.size())) {
                results.resize(query_idx + 1);
            }
            results[query_idx] = std::vector<int64_t>();  // Empty result
        }
    }
    
    stats.completed_threads++;
}

// Calculate recall
double calculate_recall(
    const std::vector<std::vector<int64_t>>& predicted,
    const std::vector<std::vector<int32_t>>& ground_truth,
    int k
) {
    if (predicted.empty() || ground_truth.empty()) {
        return 0.0;
    }
    
    int64_t total_correct = 0;
    int64_t queries_with_results = 0;
    
    int64_t num_queries = std::min(static_cast<int64_t>(predicted.size()), 
                                   static_cast<int64_t>(ground_truth.size()));
    
    for (int64_t i = 0; i < num_queries; i++) {
        if (predicted[i].empty()) continue;
        
        queries_with_results++;
        std::set<int64_t> gt_set(ground_truth[i].begin(), 
                                 ground_truth[i].begin() + std::min(k, static_cast<int>(ground_truth[i].size())));
        
        int correct = 0;
        int pred_k = std::min(k, static_cast<int>(predicted[i].size()));
        for (int j = 0; j < pred_k; j++) {
            if (gt_set.find(predicted[i][j]) != gt_set.end()) {
                correct++;
            }
        }
        total_correct += correct;
    }
    
    if (queries_with_results == 0) {
        return 0.0;
    }
    
    return static_cast<double>(total_correct) / (queries_with_results * k);
}

int main(int argc, char* argv[]) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0] << " <base_file> <query_file> <ground_truth_file> "
                  << "<index_type> <num_threads> <k> <ef_search/nprobe/search_list_size> [options]\n"
                  << "  base_file: Path to .fvecs or .bvecs file for building index\n"
                  << "  query_file: Path to .fvecs or .bvecs file for queries\n"
                  << "  ground_truth_file: Path to .ivecs file for ground truth\n"
                  << "  index_type: 'hnsw' or 'ivfflat'\n"
                  << "  num_threads: Number of concurrent query threads\n"
                  << "  k: Top-k for search\n"
                  << "  ef_search/nprobe: ef_search for HNSW, nprobe for IVFFLAT (single value or comma-separated list)\n"
                  << "Options:\n"
                  << "  --use-bvecs: Use .bvecs format (default: .fvecs)\n"
                  << "  --M <value>: HNSW M parameter (default: 16)\n"
                  << "  --ef-construction <value>: HNSW ef_construction (default: 64)\n"
                  << "  --nlist <value>: IVFFLAT nlist parameter (default: 100)\n"
                  << "  --base-skip <value>: Skip N vectors from base file before reading (default: 0)\n"
                  << "  --base-num <value>: Number of vectors to read from base file (default: all)\n"
                  << "  --query-skip <value>: Skip N queries from query file before reading (default: 0)\n"
                  << "  --query-num <value>: Number of queries to read from query file (default: all)\n"
                  << "  --save-index <path>: Save the built index to file (optional)\n"
                  << "  --load-index <path>: Load index from file instead of building (optional)\n"
                  << "  --max-vectors <value>: [Deprecated] Use --base-num instead\n"
                  << "  --max-queries <value>: [Deprecated] Use --query-num instead\n";
        return 1;
    }
    
    std::string base_file = argv[1];
    std::string query_file = argv[2];
    std::string ground_truth_file = argv[3];
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
    
    // Parse optional arguments
    for (int i = 8; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--use-bvecs") {
            use_bvecs = true;
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
    
    bool is_hnsw = (index_type == "hnsw");
    bool is_ivfflat = (index_type == "ivfflat");
    
    if (!is_hnsw && !is_ivfflat) {
        std::cerr << "Error: index_type must be 'hnsw' or 'ivfflat'\n";
        return 1;
    }
    
    std::cout << "=== Knowhere Concurrent Query Test ===\n";
    std::cout << "Base file: " << base_file << "\n";
    std::cout << "Query file: " << query_file << "\n";
    std::cout << "Ground truth file: " << ground_truth_file << "\n";
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
    std::cout << "File format: " << (use_bvecs ? ".bvecs" : ".fvecs") << "\n";
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
    std::cout << "\n";
    
    // Step 1: Get dimension (skip if loading index)
    int dim = 0;
    int64_t total_vectors_to_load = 0;
    
    if (load_index_path.empty()) {
        // Get dimension and total vectors count
        std::cout << "Step 1: Preparing to load base vectors...\n";
        
        try {
            if (use_bvecs) {
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
        // Check if index file exists and is valid
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
            // Try to load the index
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
                
                // Validate that index is not empty
                if (index.Count() <= 0) {
                    std::cout << "WARNING: Index file exists but index is empty after loading!\n";
                    std::cout << "Will build index from base file instead.\n\n";
                    should_build_index = true;
                    index_loaded = false;  // Reset flag so we build instead
                } else {
                    std::cout << "\n";
                }
            } catch (const std::exception& e) {
                std::cout << "WARNING: Error loading index: " << e.what() << "\n";
                std::cout << "Will build index from base file instead.\n\n";
                should_build_index = true;
                index_loaded = false;  // Reset flag so we build instead
            }
        }
        
        // If index file is invalid/empty/missing, build from base file
        if (should_build_index) {
            if (base_file.empty()) {
                std::cerr << "ERROR: Cannot build index - base file not provided!\n";
                std::cerr << "Please provide base_file as the first argument, or build the index separately.\n";
                return 1;
            }
            
            // Need to get dimension from base file first
            if (dim == 0) {
                try {
                    if (use_bvecs) {
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
            
            // Fall through to build index (will continue below)
            load_index_path.clear();  // Clear so we don't try to load again
        }
    }
    
    // Build index if not loaded
    if (load_index_path.empty() || !index_loaded) {
        // Build index from vectors in batches
        if (!index_loaded && !load_index_path.empty()) {
            std::cout << "Step 2: Building index (previous load failed or index was empty)...\n";
        } else {
            std::cout << "Step 2: Building index in batches...\n";
        }
        auto start_build = std::chrono::high_resolution_clock::now();
        
        const int64_t batch_size = 100000;  // Load and add 100k vectors at a time
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
            // IVFFLAT specific configuration
            conf[knowhere::indexparam::NLIST] = nlist;
        }
        
        // Load and add vectors in batches
        {
            // Open file readers
            std::unique_ptr<FvecsReader> fvecs_reader;
            std::unique_ptr<BvecsReader> bvecs_reader;
            
            if (use_bvecs) {
                bvecs_reader = std::make_unique<BvecsReader>(base_file);
            } else {
                fvecs_reader = std::make_unique<FvecsReader>(base_file);
            }
            
            // For IVFFLAT, train first on a sample
            if (is_ivfflat) {
                int64_t train_samples = std::min(static_cast<int64_t>(256) * static_cast<int64_t>(nlist), total_vectors_to_load);
                if (train_samples > 0) {
                std::cout << "Training IVFFLAT index on " << train_samples << " samples...";
                std::cout.flush();
                
                // Load training samples
                std::vector<std::vector<float>> train_vectors;
                if (use_bvecs) {
                    bvecs_reader->read_all_vectors(train_vectors, base_skip, train_samples, false);
                } else {
                    fvecs_reader->read_all_vectors(train_vectors, base_skip, train_samples, false);
                }
                
                // Flatten training vectors
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
            
            // Load and add vectors in batches
            // Ensure total_vectors_to_load is set correctly for building
            if (total_vectors_to_load == 0) {
                // If not set, try to get it from base_num parameter
                if (base_num > 0) {
                    total_vectors_to_load = base_num;
                } else {
                    total_vectors_to_load = -1;  // Read until EOF
                }
            }
            
            std::cout << "Loading and adding vectors in batches of " << batch_size << ":\n";
            std::cout << "Target: " << (total_vectors_to_load > 0 ? std::to_string(total_vectors_to_load) : "all") << " vectors\n";
            std::cout << "Progress: ";
            std::cout.flush();
            
            // Track progress reporting (every 1%)
            int last_reported_pct = -1;
            int64_t progress_interval = 0;
            if (total_vectors_to_load > 0) {
                progress_interval = std::max(static_cast<int64_t>(1), total_vectors_to_load / 100);  // Every 1%
            }
            
            bool first_batch_processed = false;
            int batch_number = 0;
            
            // Debug: Check loop condition
            bool loop_condition = (total_vectors_to_load < 0 || total_loaded < total_vectors_to_load);
            if (!loop_condition) {
                std::cerr << "ERROR: Build loop will not execute!\n";
                std::cerr << "  total_vectors_to_load: " << total_vectors_to_load << "\n";
                std::cerr << "  total_loaded: " << total_loaded << "\n";
                std::cerr << "  Condition: total_vectors_to_load < 0 || total_loaded < total_vectors_to_load\n";
                return 1;
            }
            
            while (total_vectors_to_load < 0 || total_loaded < total_vectors_to_load) {
                batch_number++;
            int64_t current_batch_size;
            if (total_vectors_to_load > 0) {
                current_batch_size = std::min(batch_size, total_vectors_to_load - total_loaded);
            } else {
                current_batch_size = batch_size;  // Read until EOF
            }
            int64_t current_skip = base_skip + total_loaded;
            
            // Load batch
            std::vector<std::vector<float>> batch_vectors;
            if (use_bvecs) {
                bvecs_reader->read_all_vectors(batch_vectors, current_skip, current_batch_size, false);
            } else {
                fvecs_reader->read_all_vectors(batch_vectors, current_skip, current_batch_size, false);
            }
            
            if (batch_vectors.empty()) {
                // EOF reached
                if (total_vectors_to_load < 0) {
                    // We were reading until EOF, so this is expected
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
            
            // Flatten batch
            int64_t batch_count = static_cast<int64_t>(batch_vectors.size());
            std::vector<float> flat_batch(batch_count * dim);
            for (int64_t i = 0; i < batch_count; i++) {
                std::memcpy(flat_batch.data() + i * dim, batch_vectors[i].data(), dim * sizeof(float));
            }
            
            // Add to index
            auto batch_dataset = knowhere::GenDataSet(batch_count, dim, flat_batch.data());
            conf[knowhere::meta::ROWS] = batch_count;
            
            if (index.Count() == -1 || index.Count() == 0) {
                // First batch - build
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
                // Subsequent batches - add
                auto add_res = index.Add(batch_dataset, conf);
                if (add_res != knowhere::Status::success) {
                    std::cerr << "\nError adding batch to index\n";
                    return 1;
                }
            }
            
            total_added += batch_count;
            
            // Print progress every 1%
            if (total_vectors_to_load > 0) {
                int current_pct = (total_added * 100) / total_vectors_to_load;
                // Report when we cross a 1% boundary
                if (current_pct > last_reported_pct) {
                    std::cout << total_added << "/" << total_vectors_to_load << " (" << current_pct << "%) ";
                    std::cout.flush();
                    last_reported_pct = current_pct;
                }
            } else {
                // When reading until EOF, report every progress_interval vectors
                if (progress_interval > 0 && total_added % progress_interval == 0) {
                    std::cout << total_added << " ";
                    std::cout.flush();
                } else if (progress_interval == 0) {
                    // First time, set interval based on batch size
                    progress_interval = std::max(static_cast<int64_t>(1), batch_size);
                }
            }
            }  // End of while loop
            
            std::cout << "\n";
            
            // Check if any vectors were processed
            if (total_added == 0) {
                std::cerr << "ERROR: No vectors were added to the index!\n";
                std::cerr << "Total loaded: " << total_loaded << ", Total added: " << total_added << "\n";
                std::cerr << "This usually means:\n";
                std::cerr << "  1. The base file is empty or cannot be read\n";
                std::cerr << "  2. --base-skip is too large (skipping all vectors)\n";
                std::cerr << "  3. --base-num is 0 or negative\n";
                std::cerr << "  4. File format mismatch (check --use-bvecs flag)\n";
                return 1;
            }
            
            auto end_build = std::chrono::high_resolution_clock::now();
            double build_time = std::chrono::duration<double>(end_build - start_build).count();
            std::cout << "Index built in " << std::fixed << std::setprecision(4) << build_time << " seconds\n";
            
            // Verify index state immediately after build
            int64_t index_count = index.Count();
            std::cout << "Index count: " << index_count << "\n";
            std::cout << "Total vectors added: " << total_added << "\n";
            std::cout << "Index dimension: " << index.Dim() << "\n";
            
            // Additional diagnostic: check if index is in a valid state
            if (index_count == -1) {
                std::cerr << "WARNING: Index count is -1, which typically means the index was not built.\n";
                std::cerr << "This can happen if:\n";
                std::cerr << "  1. Build() was never called (no batches processed)\n";
                std::cerr << "  2. Build() failed silently\n";
                std::cerr << "  3. Index object is in an invalid state\n";
                std::cerr << "Checking if build actually happened...\n";
                if (!first_batch_processed) {
                    std::cerr << "ERROR: First batch was never processed! The build loop may not have executed.\n";
                    std::cerr << "total_vectors_to_load: " << total_vectors_to_load << "\n";
                    std::cerr << "total_loaded: " << total_loaded << "\n";
                }
            }
            
            // Validate that index is not empty
            if (index.Count() <= 0 || total_added == 0) {
                std::cerr << "ERROR: Index is empty after building! Index count is " << index.Count() 
                          << ", total vectors added: " << total_added << "\n";
                std::cerr << "This will cause all queries to return no results (recall = 0.0000).\n";
                if (total_added == 0) {
                    std::cerr << "No vectors were loaded from the base file. Please check:\n";
                    std::cerr << "  - Base file path: " << base_file << "\n";
                    std::cerr << "  - Base file format (--use-bvecs flag if needed)\n";
                    std::cerr << "  - --base-skip and --base-num parameters\n";
                } else if (index.Count() <= 0) {
                    std::cerr << "Vectors were loaded but index build failed. Please check:\n";
                    std::cerr << "  - Index type and parameters\n";
                    std::cerr << "  - Error messages during build\n";
                }
                return 1;
            }
            
            // Save index if requested
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
        }  // End of batch loading
    }  // End of if (load_index_path.empty() || !index_loaded)
    
    // Step 3: Load queries
    std::cout << "Step 3: Loading queries...\n";
    std::vector<std::vector<float>> query_vectors;
    
    try {
        if (use_bvecs) {
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
    
    std::cout << "Loaded " << query_vectors.size() << " queries\n\n";
    
    // Step 4: Load ground truth
    std::cout << "Step 4: Loading ground truth...\n";
    std::vector<std::vector<int32_t>> ground_truth;
    
    try {
        IvecsReader reader(ground_truth_file, k);
        if (query_skip > 0) {
            std::cout << "Skipping first " << query_skip << " ground truth entries\n";
        }
        reader.read_all_ground_truth(ground_truth, query_skip, query_num);
    } catch (const std::exception& e) {
        std::cerr << "Error loading ground truth: " << e.what() << "\n";
        return 1;
    }
    
    std::cout << "Loaded ground truth for " << ground_truth.size() << " queries\n\n";
    
    // Validate index before running queries
    if (index.Count() <= 0) {
        std::cerr << "ERROR: Cannot run queries - index is empty! Index count: " << index.Count() << "\n";
        std::cerr << "Please rebuild or reload the index with valid data.\n";
        return 1;
    }
    std::cout << "Index validation: Index contains " << index.Count() << " vectors, ready for queries\n\n";
    
    // Step 5: Run queries for each ef_search/nprobe value
    std::cout << "Step 5: Running queries for " << ef_search_nprobe_list.size() << " different " 
              << (is_hnsw ? "ef_search" : "nprobe") << " value(s)...\n\n";
    
    // Store results for each ef_search/nprobe value
    std::vector<std::pair<int, std::vector<std::vector<int64_t>>>> all_results;
    // Store stats separately (QueryStats has atomic members, so we store values)
    struct StatsSnapshot {
        int64_t query_count;
        double total_time;
        double query_time;
    };
    std::vector<std::pair<int, StatsSnapshot>> all_stats;
    
    for (size_t ef_idx = 0; ef_idx < ef_search_nprobe_list.size(); ef_idx++) {
        int ef_search_nprobe = ef_search_nprobe_list[ef_idx];
        
        std::cout << "========================================\n";
        std::cout << "Running queries with " << (is_hnsw ? "ef_search" : "nprobe") 
                  << " = " << ef_search_nprobe << " (" << (ef_idx + 1) << "/" 
                  << ef_search_nprobe_list.size() << ")\n";
        std::cout << "========================================\n";
        
        // Prepare query queue
        std::queue<std::pair<int64_t, std::vector<float>>> query_queue;
        std::mutex queue_mutex;
        
        for (int64_t i = 0; i < static_cast<int64_t>(query_vectors.size()); i++) {
            query_queue.push({i, query_vectors[i]});
        }
        
        // Run concurrent queries
        std::vector<std::vector<int64_t>> results;
        std::mutex results_mutex;
        QueryStats stats;
        std::atomic<bool> stop_flag(false);
        
        std::vector<std::thread> threads;
        auto start_query = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_threads; i++) {
            threads.emplace_back(query_worker,
                               &index,
                               std::ref(query_queue),
                               std::ref(queue_mutex),
                               std::ref(results),
                               std::ref(results_mutex),
                               std::ref(stats),
                               i,
                               k,
                               ef_search_nprobe,
                               ef_search_nprobe,
                               is_hnsw,
                               std::ref(stop_flag));
        }
        
        // Wait for all queries to complete
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (query_queue.empty()) {
                stop_flag = true;
                break;
            }
        }
        
        // Wait for all threads
        for (auto& t : threads) {
            t.join();
        }
        
        auto end_query = std::chrono::high_resolution_clock::now();
        double query_time = std::chrono::duration<double>(end_query - start_query).count();
        
        std::cout << "\nQuery completed in " << std::fixed << std::setprecision(4) 
                  << query_time << " seconds\n";
        std::cout << "Total queries: " << stats.query_count.load() << "\n";
        std::cout << "Total query time: " << std::fixed << std::setprecision(4) 
                  << stats.total_time.load() << " seconds\n";
        if (stats.query_count.load() > 0) {
            std::cout << "Average query time: " << std::fixed << std::setprecision(6)
                      << stats.total_time.load() / stats.query_count.load() << " seconds\n";
            std::cout << "Query throughput: " << std::fixed << std::setprecision(2)
                      << stats.query_count.load() / query_time << " queries/second\n";
        }
        
        // Calculate recall for this ef_search/nprobe value
        double recall = calculate_recall(results, ground_truth, k);
        
        // Diagnostic: count queries with results
        int64_t queries_with_results = 0;
        int64_t total_result_ids = 0;
        for (const auto& result : results) {
            if (!result.empty()) {
                queries_with_results++;
                total_result_ids += result.size();
            }
        }
        
        std::cout << "Recall@" << k << ": " << std::fixed << std::setprecision(4) << recall << "\n";
        std::cout << "Diagnostics: " << queries_with_results << "/" << results.size() 
                  << " queries returned results, " << total_result_ids << " total result IDs\n";
        if (queries_with_results == 0) {
            std::cerr << "WARNING: No queries returned any results! This suggests the index is empty or not properly built.\n";
            std::cerr << "Index count: " << index.Count() << "\n";
        }
        std::cout << "\n";
        
        // Store query_time for this run (for accurate throughput calculation)
        stats.query_time = query_time;
        
        // Store results and stats
        all_results.push_back({ef_search_nprobe, results});
        StatsSnapshot snapshot;
        snapshot.query_count = stats.query_count.load();
        snapshot.total_time = stats.total_time.load();
        snapshot.query_time = stats.query_time;
        all_stats.push_back({ef_search_nprobe, snapshot});
    }
    
    // Step 6: Summary for all ef_search/nprobe values
    std::cout << "\n========================================\n";
    std::cout << "SUMMARY FOR ALL " << (is_hnsw ? "EF_SEARCH" : "NPROBE") << " VALUES (CONCURRENCY = " << num_threads << ")\n";
    std::cout << "========================================\n";
    std::cout << std::left << std::setw(12) << (is_hnsw ? "ef_search" : "nprobe")
              << std::setw(12) << "Recall@" << std::to_string(k)
              << std::setw(18) << "Avg Query Time"
              << std::setw(18) << "Throughput (q/s)" << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (size_t i = 0; i < all_results.size(); i++) {
        int ef_val = all_results[i].first;
        double recall = calculate_recall(all_results[i].second, ground_truth, k);
        StatsSnapshot& stats = all_stats[i].second;
        
        double avg_time = 0.0;
        double throughput = 0.0;
        if (stats.query_count > 0) {
            avg_time = stats.total_time / stats.query_count;
            // Calculate throughput from wall-clock time
            if (stats.query_time > 0.0) {
                throughput = stats.query_count / stats.query_time;
            }
        }
        
        std::cout << std::left << std::setw(12) << ef_val
                  << std::setw(12) << std::fixed << std::setprecision(4) << recall
                  << std::setw(18) << std::setprecision(6) << avg_time
                  << std::setw(18) << std::setprecision(2) << throughput << "\n";
    }
    std::cout << "\n";
    
    return 0;
}

// Build HNSW index and save it (with custom M and ef_construction)
// ./knowhere_concurrent_query /ssd_root/dataset/sift/bigann_base.bvecs \
// /ssd_root/dataset/sift/bigann_query.bvecs /ssd_root/dataset/sift/gnd/idx_10M.ivecs \
// hnsw 1 100  "100,200,300,400,500,600,700,800,900,1000" \
//     --use-bvecs --base-num 10000000 --M 16 --ef-construction 40 \
//     --save-index /ssd_root/liu4127/knowhere_indexes/hnsw_sift10M_index.idxls

// Load existing index (much faster, skips building)
// ./knowhere_concurrent_query /ssd_root/dataset/sift/bigann_base.bvecs \
// /ssd_root/dataset/sift/bigann_query.bvecs /ssd_root/dataset/sift/gnd/idx_10M.ivecs \
// hnsw 1 100  "100,200,300,400,500,600,700,800,900,1000" \
//     --use-bvecs --load-index /ssd_root/liu4127/knowhere_indexes/hnsw_sift10M_index.idxls

// Build IVFFLAT index with different parameters and save
// ./knowhere_concurrent_query /ssd_root/dataset/sift/bigann_base.bvecs \
// /ssd_root/dataset/sift/bigann_query.bvecs /ssd_root/dataset/sift/gnd/idx_10M.ivecs \
// ivfflat 1 100 "10,20,50,100" \
//     --use-bvecs --base-num 10000000 --nlist 3162 \
//     --save-index /ssd_root/liu4127/knowhere_indexes/ivfflat_sift10M_index.idx