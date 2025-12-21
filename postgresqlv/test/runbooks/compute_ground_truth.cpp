// compute_ground_truth.cpp
//
// Standalone script to compute all ground truth files for search steps in a runbook
// Uses concurrent threads to compute multiple GT files in parallel
//
// Compilation command (with SIMD optimizations):
//   g++ -O3 -std=c++17 -mavx2 -mfma -fopenmp compute_ground_truth.cpp \
//       -o compute_ground_truth \
//       -lyaml-cpp -lcrypto -pthread
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <algorithm>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <atomic>
#include <omp.h>
#include <yaml-cpp/yaml.h>
#include <openssl/md5.h>

// SIMD headers
#ifdef __AVX2__
#include <immintrin.h>
#define USE_AVX2
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#define USE_SSE
#elif defined(__SSE__)
#include <xmmintrin.h>
#define USE_SSE
#endif

// ===================================================
// SIMD-optimized L2 distance computation
// ===================================================
namespace SIMDUtils {
    inline float l2_distance_simd(const float* a, const float* b, size_t dim) {
        float dist_squared = 0.0f;
        
#ifdef USE_AVX2
        const size_t simd_width = 8;
        size_t i = 0;
        __m256 sum_vec = _mm256_setzero_ps();
        
        for (; i + simd_width <= dim; i += simd_width) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 diff = _mm256_sub_ps(va, vb);
            __m256 sq = _mm256_mul_ps(diff, diff);
            sum_vec = _mm256_add_ps(sum_vec, sq);
        }
        
        __m128 sum128_lo = _mm256_extractf128_ps(sum_vec, 0);
        __m128 sum128_hi = _mm256_extractf128_ps(sum_vec, 1);
        __m128 sum128 = _mm_add_ps(sum128_lo, sum128_hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        dist_squared = _mm_cvtss_f32(sum128);
        
        for (; i < dim; i++) {
            float diff = a[i] - b[i];
            dist_squared += diff * diff;
        }
        
#elif defined(USE_SSE)
        const size_t simd_width = 4;
        size_t i = 0;
        __m128 sum_vec = _mm_setzero_ps();
        
        for (; i + simd_width <= dim; i += simd_width) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 diff = _mm_sub_ps(va, vb);
            __m128 sq = _mm_mul_ps(diff, diff);
            sum_vec = _mm_add_ps(sum_vec, sq);
        }
        
        sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
        sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
        dist_squared = _mm_cvtss_f32(sum_vec);
        
        for (; i < dim; i++) {
            float diff = a[i] - b[i];
            dist_squared += diff * diff;
        }
        
#else
        for (size_t i = 0; i < dim; i++) {
            float diff = a[i] - b[i];
            dist_squared += diff * diff;
        }
#endif
        
        return std::sqrt(dist_squared);
    }
}

// ===================================================
// Data loader
// ===================================================
class DataLoader {
public:
    static std::vector<float> load_fvecs(const std::string& filename, size_t& num_vectors, size_t& dim) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        int d;
        file.read(reinterpret_cast<char*>(&d), sizeof(int));
        dim = d;
        file.seekg(0, std::ios::end);
        size_t file_size = (size_t)file.tellg();
        num_vectors = file_size / ((dim + 1) * sizeof(float));
        file.seekg(0, std::ios::beg);

        std::vector<float> data(num_vectors * dim);

        for (size_t i = 0; i < num_vectors; i++) {
            file.read(reinterpret_cast<char*>(&d), sizeof(int));
            file.read(reinterpret_cast<char*>(&data[i * dim]), dim * sizeof(float));
        }

        file.close();
        return data;
    }
};

// Generate GT filename (consistent with pgvector_test.cpp logic)
std::string get_gt_filename(const std::string& runbook_name, size_t step_num,
                            const std::vector<std::tuple<std::string, size_t, size_t>>& ranges) {
    std::stringstream ranges_str;
    ranges_str << "[";
    for (size_t i = 0; i < ranges.size(); i++) {
        if (i > 0) ranges_str << ",";
        ranges_str << "[\"" << std::get<0>(ranges[i]) << "\","
                   << std::get<1>(ranges[i]) << ","
                   << std::get<2>(ranges[i]) << "]";
    }
    ranges_str << "]";

    unsigned char hash[MD5_DIGEST_LENGTH];
    auto s = ranges_str.str();
    MD5(reinterpret_cast<const unsigned char*>(s.c_str()), s.size(), hash);

    std::stringstream hash_str;
    for (int i = 0; i < 4; i++) {
        hash_str << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }

    std::stringstream filename;
    filename << runbook_name << "_step" << step_num
             << "_ranges" << hash_str.str() << "_gt.npy";
    return filename.str();
}

// Save .npy GT file
void save_gt_npy(const std::string& filepath, const std::vector<std::vector<int>>& gt) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create file: " + filepath);
    }

    const char magic[] = "\x93NUMPY";
    const uint8_t ver_major = 0x01, ver_minor = 0x00;

    size_t rows = gt.size();
    size_t cols = rows ? gt[0].size() : 0;
    std::stringstream header_ss;
    header_ss << "{'descr': '<i4', 'fortran_order': False, 'shape': ("
              << rows << ", " << cols << ")}";

    std::string header = header_ss.str();
    size_t preamble = 6 + 2 + 2;
    size_t header_len = header.size() + 1;
    size_t pad = (64 - ((preamble + header_len) % 64)) % 64;
    header.append(pad, ' ');
    header.push_back('\n');
    uint16_t header_size_le = (uint16_t)header.size();

    file.write(magic, 6);
    file.put((char)ver_major);
    file.put((char)ver_minor);
    file.write(reinterpret_cast<const char*>(&header_size_le), 2);
    file.write(header.data(), (std::streamsize)header.size());

    for (const auto& row : gt) {
        file.write(reinterpret_cast<const char*>(row.data()),
                   (std::streamsize)(cols * sizeof(int)));
    }

    file.close();
    std::cout << "  Saved: " << filepath << std::endl;
}

// Compute ground truth for a specific search step with progress reporting
std::vector<std::vector<int>> compute_gt_for_step(
    float* dataset, float* queries, size_t num_queries, size_t k, size_t dim,
    const std::set<size_t>& active_indices, bool show_progress = true) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> gt(num_queries, std::vector<int>(k, -1));

    if (active_indices.empty()) {
        if (show_progress) {
            std::cout << "    → No active indices, returning empty GT" << std::endl;
        }
        return gt;
    }

    std::vector<size_t> active_idx(active_indices.begin(), active_indices.end());
    
    if (show_progress) {
        std::cout << "    → Computing distances for " << num_queries << " queries against " 
                  << active_idx.size() << " active vectors..." << std::endl;
    }

    std::atomic<size_t> completed_queries(0);
    const size_t progress_interval = std::max<size_t>(1, num_queries / 10); // Update every 10%

    #pragma omp parallel for
    for (size_t q = 0; q < num_queries; q++) {
        std::vector<std::pair<float, size_t>> distances;
        distances.reserve(active_idx.size());

        const float* qv = queries + q * dim;
        for (size_t idx : active_idx) {
            const float* dv = dataset + idx * dim;
            float dist = SIMDUtils::l2_distance_simd(qv, dv, dim);
            distances.emplace_back(dist, idx);
        }

        size_t take = std::min(k, distances.size());
        std::partial_sort(distances.begin(), distances.begin() + take, distances.end());
        for (size_t i = 0; i < take; i++) {
            gt[q][i] = (int)distances[i].second;
        }

        // Progress reporting
        if (show_progress) {
            size_t completed = ++completed_queries;
            if (completed % progress_interval == 0 || completed == num_queries) {
                double progress = 100.0 * completed / num_queries;
                #pragma omp critical
                {
                    std::cout << "    → Progress: " << std::fixed << std::setprecision(1) << progress 
                              << "% (" << completed << "/" << num_queries << " queries)" << std::endl;
                }
            }
        }
    }

    if (show_progress) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double elapsed_sec = duration.count() / 1000.0;
        double qps = num_queries / elapsed_sec;
        
        std::cout << "    → Completed: " << num_queries << " queries in " 
                  << std::fixed << std::setprecision(2) << elapsed_sec << "s "
                  << "(" << std::setprecision(0) << qps << " queries/sec)" << std::endl;
    }

    return gt;
}


void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " [options]" << std::endl;
    std::cerr << "\nRequired arguments:" << std::endl;
    std::cerr << "  --dataset <path>            Path to dataset .fvecs file" << std::endl;
    std::cerr << "  --queries <path>            Path to queries .fvecs file" << std::endl;
    std::cerr << "  --runbook <path>            Path to runbook YAML file" << std::endl;
    std::cerr << "  --dataset-name <name>       Dataset name in runbook" << std::endl;
    std::cerr << "  --gt-dir <path>             Ground truth directory" << std::endl;
    std::cerr << "\nOptional arguments:" << std::endl;
    std::cerr << "  --num-threads <n>           Number of threads for parallel computation (default: number of CPU cores)" << std::endl;
}

int main(int argc, char** argv) {
    std::string dataset_path, query_path, runbook_path, dataset_name, gt_dir;
    int num_threads = omp_get_max_threads();

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            dataset_path = argv[++i];
        } else if (arg == "--queries" && i + 1 < argc) {
            query_path = argv[++i];
        } else if (arg == "--runbook" && i + 1 < argc) {
            runbook_path = argv[++i];
        } else if (arg == "--dataset-name" && i + 1 < argc) {
            dataset_name = argv[++i];
        } else if (arg == "--gt-dir" && i + 1 < argc) {
            gt_dir = argv[++i];
        } else if (arg == "--num-threads" && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    if (dataset_path.empty() || query_path.empty() || 
        runbook_path.empty() || dataset_name.empty() || gt_dir.empty()) {
        std::cerr << "Error: --dataset, --queries, --runbook, --dataset-name, and --gt-dir are required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    // Set OpenMP thread count
    omp_set_num_threads(num_threads);

    // Ensure GT directory exists
    std::string cmd = "mkdir -p " + gt_dir;
    (void)system(cmd.c_str());

    std::cout << "========================================" << std::endl;
    std::cout << "GROUND TRUTH COMPUTATION" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Dataset: " << dataset_path << std::endl;
    std::cout << "Queries: " << query_path << std::endl;
    std::cout << "Runbook: " << runbook_path << std::endl;
    std::cout << "Dataset name: " << dataset_name << std::endl;
    std::cout << "GT directory: " << gt_dir << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "========================================\n" << std::endl;

    try {
        // Load dataset
        std::cout << "Loading dataset..." << std::endl;
        size_t num_vectors, data_dim;
        auto dataset = DataLoader::load_fvecs(dataset_path, num_vectors, data_dim);
        std::cout << "✔ Dataset loaded: " << num_vectors << " vectors, dim=" << data_dim << std::endl;

        // Load queries
        std::cout << "Loading queries..." << std::endl;
        size_t num_queries, query_dim;
        auto queries = DataLoader::load_fvecs(query_path, num_queries, query_dim);
        std::cout << "✔ Queries loaded: " << num_queries << " queries, dim=" << query_dim << std::endl;

        if (data_dim != query_dim) {
            throw std::runtime_error("Dataset and query dimensions do not match");
        }

        // Load runbook
        std::cout << "Loading runbook..." << std::endl;
        YAML::Node runbook = YAML::LoadFile(runbook_path);
        auto operations = runbook[dataset_name];
        std::cout << "✔ Runbook loaded" << std::endl;

        std::string runbook_name = dataset_name;
        std::replace(runbook_name.begin(), runbook_name.end(), '-', '_');

        // Single pass: build active_indices cumulatively and compute GT immediately at search steps
        std::set<size_t> active_indices;
        std::vector<std::tuple<std::string, size_t, size_t>> active_ranges;
        size_t step_num = 0;
        size_t search_count = 0;

        std::cout << "\nSimulating operations and computing ground truth..." << std::endl;
        std::cout << "========================================" << std::endl;

        // Single pass: modify active_indices cumulatively and compute GT immediately when meeting search steps
        for (auto it = operations.begin(); it != operations.end(); ++it) {
            std::string step_key = it->first.as<std::string>();
            if (step_key == "max_pts" || step_key == "query" || step_key == "groundtruth") {
                continue;
            }

            step_num++;
            auto step = it->second;
            std::string op = step["operation"].as<std::string>();

            if (op == "insert") {
                size_t start = step["start"].as<size_t>(0);
                size_t end = step["end"].as<size_t>(0);
                
                // Simulate insert operation
                std::cout << "Step " << step_num << " [" << step_key << "]: INSERT [" 
                          << start << ":" << end << "]" << std::endl;
                
                for (size_t i = start; i < end; i++) {
                    active_indices.insert(i);
                }
                active_ranges.emplace_back("insert", start, end);
                
                std::cout << "  → Active indices: " << active_indices.size() << std::endl;
                
            } else if (op == "delete") {
                size_t start = step["start"].as<size_t>(0);
                size_t end = step["end"].as<size_t>(0);
                
                // Simulate delete operation
                std::cout << "Step " << step_num << " [" << step_key << "]: DELETE [" 
                          << start << ":" << end << "]" << std::endl;
                
                size_t deleted_count = 0;
                for (size_t i = start; i < end; i++) {
                    if (active_indices.erase(i) > 0) {
                        deleted_count++;
                    }
                }
                active_ranges.emplace_back("delete", start, end);
                
                std::cout << "  → Deleted: " << deleted_count 
                          << ", Active indices: " << active_indices.size() << std::endl;
                
            } else if (op == "search") {
                size_t k = step["k"].as<size_t>(100);
                search_count++;
                
                std::cout << "Step " << step_num << " [" << step_key << "]: SEARCH (k=" 
                          << k << ", active_indices=" << active_indices.size() << ")" << std::endl;
                
                // Generate filename
                std::string gt_filename = get_gt_filename(runbook_name, step_num, active_ranges);
                std::string gt_filepath = gt_dir + "/" + gt_filename;

                // Check if file already exists
                std::ifstream check(gt_filepath, std::ios::binary);
                if (check.good()) {
                    check.close();
                    std::cout << "  → Skipping (already exists): " << gt_filename << std::endl;
                    continue;
                }

                // Compute GT immediately using current active_indices
                std::cout << "  → Computing ground truth for " << num_queries 
                          << " queries (k=" << k << ")..." << std::endl;
                auto gt = compute_gt_for_step(
                    dataset.data(), queries.data(), num_queries, k, data_dim, active_indices, true);

                // Save GT
                std::cout << "  → Saving ground truth file..." << std::endl;
                save_gt_npy(gt_filepath, gt);
                std::cout << "  → ✓ Completed: " << gt_filename << std::endl;
            }
        }

        std::cout << "========================================" << std::endl;
        std::cout << "\nCompleted: " << search_count << " search step(s) processed" << std::endl;

        std::cout << "\n========================================" << std::endl;
        std::cout << "ALL GROUND TRUTH FILES COMPUTED" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


// ./compute_ground_truth
//   --dataset /ssd_root/dataset/turing10m/msturing-10M.fvecs
//   --queries /ssd_root/dataset/turing10m/msturing-query.fvecs
//   --runbook msturing-10M_slidingwindow_runbook.yaml
//   --dataset-name msturing-10M
//   --gt-dir /ssd_root/liu4127/msturing_runbook_gt
//   --num-threads 32