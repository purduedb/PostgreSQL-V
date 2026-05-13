// pgvector_test.cpp
//
// pgvector HNSW sliding window test script - fine-grained parallel version
// Parallel logic consistent with hnswlib sliding_window_test.cpp:
//   - Insert: each vector is an independent task
//   - Delete: each vector is an independent task
//   - Search: each query is an independent task
//
// Key design:
//   - Use thread_local connections to avoid connection pool contention
//   - Use Prepared Statements to reduce SQL parsing overhead
//   - Same checkpoint_size batching + internal fine-grained parallelism as hnswlib
//
// Compilation command (with SIMD optimizations):
//   g++ -O3 -std=c++17 -mavx2 -mfma -fopenmp pgvector_test.cpp \
//       -o pgvector_test \
//       -I/usr/include/postgresql -lpq -lyaml-cpp -lcrypto -pthread
// Note: -mavx2 enables AVX2 SIMD instructions for faster L2 distance computation
//       For SSE-only systems, use -msse4.1 instead
//       For systems without SIMD, the code will automatically fall back to scalar operations
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <unordered_set>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <mutex>
#include <atomic>
#include <memory>
#include <random>
#include <libpq-fe.h>
#include <omp.h>
#include <yaml-cpp/yaml.h>
#include <openssl/md5.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdlib>
#include <cerrno>
#include <cstring>

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

// ==================================
// Configuration structure
// ==================================
struct Config {
    // PostgreSQL connection parameters
    std::string DB_HOST = "localhost";
    std::string DB_PORT = "5432";
    std::string DB_NAME = "vector_benchmark";
    std::string DB_USER = "postgres";
    std::string DB_PASSWORD = "";
    
    // Index type: "hnsw" or "ivfflat"
    std::string INDEX_TYPE = "hnsw";

    // pgvector HNSW index parameters (used when INDEX_TYPE == "hnsw")
    int HNSW_M = 48;
    int HNSW_EF_CONSTRUCTION = 400;
    int HNSW_EF_SEARCH = 200;

    // pgvector IVFFlat index parameters (used when INDEX_TYPE == "ivfflat")
    int IVFFLAT_LISTS = 100;
    int IVFFLAT_PROBES = 10;
    
    // Dataset paths
    std::string DATASET_PATH;
    std::string QUERY_PATH;
    std::string RUNBOOK_PATH;
    std::string DATASET_NAME;
    std::string GT_DIR;
    
    // Dimension will be auto-detected from dataset
    size_t DIMENSION = 0;

    // Table and dataset offset
    std::string TABLE_NAME = "vectors";
    size_t DATASET_OFFSET = 0;
};

// ===================================================
// SIMD-optimized L2 distance computation
// ===================================================
namespace SIMDUtils {
    // Compute L2 distance between two vectors using SIMD
    inline float l2_distance_simd(const float* a, const float* b, size_t dim) {
        float dist_squared = 0.0f;
        
#ifdef USE_AVX2
        // AVX2: process 8 floats at a time
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
        
        // Horizontal sum of 8 floats (more efficient method)
        __m128 sum128_lo = _mm256_extractf128_ps(sum_vec, 0);
        __m128 sum128_hi = _mm256_extractf128_ps(sum_vec, 1);
        __m128 sum128 = _mm_add_ps(sum128_lo, sum128_hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        dist_squared = _mm_cvtss_f32(sum128);
        
        // Handle remainder
        for (; i < dim; i++) {
            float diff = a[i] - b[i];
            dist_squared += diff * diff;
        }
        
#elif defined(USE_SSE)
        // SSE: process 4 floats at a time
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
        
        // Horizontal sum of 4 floats
        sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
        sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
        dist_squared = _mm_cvtss_f32(sum_vec);
        
        // Handle remainder
        for (; i < dim; i++) {
            float diff = a[i] - b[i];
            dist_squared += diff * diff;
        }
        
#else
        // Scalar fallback
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
    // Load .fvecs format: each vector is [int32 dim][float32 * dim]
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

    // Load .fbin format: 8-byte header (n:int32, d:int32 little-endian), then n*d float32 contiguous
    static std::vector<float> load_fbin(const std::string& filename, size_t& num_vectors, size_t& dim) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        int32_t n, d;
        file.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        if (!file || n <= 0 || d <= 0) {
            throw std::runtime_error("Invalid fbin header: n=" + std::to_string(n) + " d=" + std::to_string(d));
        }
        num_vectors = static_cast<size_t>(n);
        dim = static_cast<size_t>(d);

        std::vector<float> data(num_vectors * dim);
        file.read(reinterpret_cast<char*>(data.data()), num_vectors * dim * sizeof(float));
        if (!file || file.gcount() != static_cast<std::streamsize>(num_vectors * dim * sizeof(float))) {
            throw std::runtime_error("Failed to read fbin data: file truncated or read error");
        }

        file.close();
        return data;
    }

    // Load vector file, auto-detecting format from extension (.fvecs or .fbin)
    static std::vector<float> load(const std::string& filename, size_t& num_vectors, size_t& dim) {
        size_t ext_pos = filename.rfind('.');
        if (ext_pos == std::string::npos) {
            throw std::runtime_error("Cannot detect file format: no extension in " + filename);
        }
        std::string ext = filename.substr(ext_pos + 1);
        if (ext == "fbin") {
            return load_fbin(filename, num_vectors, dim);
        } else if (ext == "fvecs") {
            return load_fvecs(filename, num_vectors, dim);
        } else {
            throw std::runtime_error("Unsupported file format: ." + ext + " (use .fvecs or .fbin)");
        }
    }
};

// ==========================
// Timer
// ==========================
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;

public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed_seconds() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000000.0;
    }
};

// ============================================================================
// Thread-local connection manager
// ============================================================================
class ThreadLocalConnection {
private:
    static std::string conninfo;
    static std::vector<PGconn*> all_connections;
    static std::mutex conn_mutex;

public:
    static void init(const std::string& host, const std::string& port,
                     const std::string& dbname, const std::string& user, 
                     const std::string& password) {
        std::stringstream ss;
        ss << "host=" << host << " port=" << port << " dbname=" << dbname << " user=" << user;
        if (!password.empty()) {
            ss << " password=" << password;
        }
        conninfo = ss.str();
    }

    static PGconn* get() {
        thread_local PGconn* conn = nullptr;
        
        if (conn == nullptr) {
            conn = PQconnectdb(conninfo.c_str());
            if (PQstatus(conn) != CONNECTION_OK) {
                std::cerr << "Thread connection failed: " << PQerrorMessage(conn) << std::endl;
                PQfinish(conn);
                conn = nullptr;
                return nullptr;
            }
            
            // Record connection for cleanup
            std::lock_guard<std::mutex> lock(conn_mutex);
            all_connections.push_back(conn);
        }
        
        return conn;
    }

    static void cleanup() {
        std::lock_guard<std::mutex> lock(conn_mutex);
        for (auto conn : all_connections) {
            if (conn) PQfinish(conn);
        }
        all_connections.clear();
    }

    static PGconn* get_main() {
        static PGconn* main_conn = nullptr;
        if (main_conn == nullptr) {
            main_conn = PQconnectdb(conninfo.c_str());
            if (PQstatus(main_conn) != CONNECTION_OK) {
                std::cerr << "Main connection failed: " << PQerrorMessage(main_conn) << std::endl;
                PQfinish(main_conn);
                main_conn = nullptr;
            }
        }
        return main_conn;
    }
};

std::string ThreadLocalConnection::conninfo;
std::vector<PGconn*> ThreadLocalConnection::all_connections;
std::mutex ThreadLocalConnection::conn_mutex;

// =========================================================
// pgvector sliding window test class
// =========================================================
class PGVectorSlidingWindowTest {
private:
    std::set<size_t> active_indices;
    size_t dim;
    std::string table_name;
    std::string gt_dir;
    size_t dataset_offset;
    std::vector<std::tuple<std::string, size_t, size_t>> active_ranges;
    std::string index_type;
    int hnsw_m;
    int hnsw_ef_construction;
    int hnsw_ef_search;
    int ivfflat_lists;
    int ivfflat_probes;
    
    // Crash simulation
    bool simulate_crash = false;
    size_t crash_target_step = 0;
    size_t crash_target_operation = 0;
    std::atomic<size_t> current_operation_count{0};
    std::string db_host;
    std::string db_port;
    
    // Skip recall computation
    bool skip_recall_computation = false;
    
    // Transaction batching (only used when mixed mode is disabled)
    size_t transaction_batch_size = 0;  // 0 means disabled (each operation is independent)

    // Statistics structure
    struct DetailedStats {
        struct OperationStats {
            std::string name;
            std::string type;
            double time;
            size_t successful = 0;
            size_t failed = 0;
            double recall = 0.0;
            size_t k = 0;
            size_t active_count = 0;
            size_t start_idx = 0;
            size_t end_idx = 0;
            int num_threads = 1;
            double throughput_per_thread = 0.0;
        };

        struct CategoryStats {
            int count = 0;
            double total_time = 0.0;
            double min_time = std::numeric_limits<double>::max();
            double max_time = 0.0;
            size_t total_successful = 0;
            size_t total_failed = 0;
            std::vector<double> recall_values;
            double min_recall = 1.0;
            double max_recall = 0.0;
            double max_throughput_per_thread = 0.0;
            double min_throughput_per_thread = std::numeric_limits<double>::max();
        };

        double total_time = 0.0;
        std::vector<OperationStats> operations;
        CategoryStats insert_stats;
        CategoryStats delete_stats;
        CategoryStats search_stats;
        double peak_insert_throughput = 0.0;
        double peak_delete_throughput = 0.0;
        double peak_search_qps = 0.0;
        double peak_insert_throughput_per_thread = 0.0;
        double peak_delete_throughput_per_thread = 0.0;
        double peak_search_qps_per_thread = 0.0;
        int max_threads_used = 0;
    } stats;

    // Generate SQL string representation of vector
    std::string vector_to_sql(const float* vec, size_t dim) {
        std::stringstream ss;
        ss << "'[";
        for (size_t i = 0; i < dim; i++) {
            if (i > 0) ss << ",";
            ss << std::fixed << std::setprecision(6) << vec[i];
        }
        ss << "]'";
        return ss.str();
    }

    // Kill PostgreSQL master process to simulate crash
    void kill_postgres_master() {
        std::cout << "\n[CRASH SIMULATION] Killing PostgreSQL master process..." << std::endl;
        
        // Method 1: Find postmaster process by port using lsof or ss
        std::stringstream cmd;
        cmd << "lsof -ti:" << db_port << " | head -1";
        FILE* pipe = popen(cmd.str().c_str(), "r");
        if (pipe) {
            char buffer[128];
            if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                std::string pid_str(buffer);
                // Remove newline
                pid_str.erase(pid_str.find_last_not_of(" \n\r\t") + 1);
                if (!pid_str.empty()) {
                    pid_t pid = std::atoi(pid_str.c_str());
                    if (pid > 0) {
                        std::cout << "[CRASH SIMULATION] Found postmaster PID: " << pid << std::endl;
                        // Kill with SIGKILL for immediate termination
                        if (kill(pid, SIGKILL) == 0) {
                            std::cout << "[CRASH SIMULATION] Successfully killed PostgreSQL master process (PID: " << pid << ")" << std::endl;
                            pclose(pipe);
                            return;
                        } else {
                            std::cerr << "[CRASH SIMULATION] Failed to kill process " << pid << ": " << strerror(errno) << std::endl;
                        }
                    }
                }
            }
            pclose(pipe);
        }
        
        // Method 2: Try using pg_ctl if available (requires data directory, less reliable)
        // Method 3: Fallback - try to find postgres process listening on the port
        std::stringstream cmd2;
        cmd2 << "ss -tlnp | grep :" << db_port << " | awk '{print $6}' | cut -d',' -f2 | head -1";
        pipe = popen(cmd2.str().c_str(), "r");
        if (pipe) {
            char buffer[128];
            if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                std::string pid_str(buffer);
                pid_str.erase(pid_str.find_last_not_of(" \n\r\t") + 1);
                if (!pid_str.empty()) {
                    pid_t pid = std::atoi(pid_str.c_str());
                    if (pid > 0) {
                        std::cout << "[CRASH SIMULATION] Found process via ss: " << pid << std::endl;
                        if (kill(pid, SIGKILL) == 0) {
                            std::cout << "[CRASH SIMULATION] Successfully killed PostgreSQL master process (PID: " << pid << ")" << std::endl;
                            pclose(pipe);
                            return;
                        }
                    }
                }
            }
            pclose(pipe);
        }
        
        std::cerr << "[CRASH SIMULATION] Warning: Could not find PostgreSQL master process to kill" << std::endl;
    }

    // Check if we should trigger crash at this operation
    bool should_crash(size_t current_step) {
        if (!simulate_crash) return false;
        
        if (current_step == crash_target_step) {
            size_t op_count = current_operation_count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (op_count == crash_target_operation) {
                std::cout << "\n[CRASH SIMULATION] Target reached: Step " << current_step 
                          << ", Operation " << op_count << std::endl;
                return true;
            }
        }
        return false;
    }

    // Generate GT filename (consistent with HNSW test script GT filename logic)
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

    // Load .npy GT file
    std::vector<std::vector<int>> load_gt_npy(const std::string& filepath,
                                              size_t expected_queries, size_t k) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            return {};
        }

        char magic[6];
        file.read(magic, 6);
        char ver[2];
        file.read(ver, 2);

        uint16_t header_len = 0;
        file.read(reinterpret_cast<char*>(&header_len), 2);
        file.seekg(header_len, std::ios::cur);

        std::vector<std::vector<int>> gt(expected_queries, std::vector<int>(k));
        for (size_t i = 0; i < expected_queries; i++) {
            file.read(reinterpret_cast<char*>(gt[i].data()),
                      (std::streamsize)(k * sizeof(int)));
        }

        file.close();
        std::cout << "Loaded ground truth from " << filepath << std::endl;
        return gt;
    }



    void updateCategoryStats(DetailedStats::CategoryStats& cat_stats,
                             const DetailedStats::OperationStats& op_stats) {
        cat_stats.count++;
        cat_stats.total_time += op_stats.time;
        cat_stats.min_time = std::min(cat_stats.min_time, op_stats.time);
        cat_stats.max_time = std::max(cat_stats.max_time, op_stats.time);
        cat_stats.total_successful += op_stats.successful;
        cat_stats.total_failed += op_stats.failed;

        if (op_stats.time > 0) {
            double throughput = (double)(op_stats.successful) / op_stats.time;
            double throughput_per_thread = throughput / op_stats.num_threads;
            
            cat_stats.max_throughput_per_thread = std::max(cat_stats.max_throughput_per_thread, throughput_per_thread);
            cat_stats.min_throughput_per_thread = std::min(cat_stats.min_throughput_per_thread, throughput_per_thread);
            
            if (op_stats.type == "insert") {
                stats.peak_insert_throughput = std::max(stats.peak_insert_throughput, throughput);
                stats.peak_insert_throughput_per_thread = std::max(stats.peak_insert_throughput_per_thread, throughput_per_thread);
            } else if (op_stats.type == "delete") {
                stats.peak_delete_throughput = std::max(stats.peak_delete_throughput, throughput);
                stats.peak_delete_throughput_per_thread = std::max(stats.peak_delete_throughput_per_thread, throughput_per_thread);
            } else if (op_stats.type == "search") {
                stats.peak_search_qps = std::max(stats.peak_search_qps, throughput);
                stats.peak_search_qps_per_thread = std::max(stats.peak_search_qps_per_thread, throughput_per_thread);
            }
        }

        stats.max_threads_used = std::max(stats.max_threads_used, op_stats.num_threads);

        if (op_stats.type == "search" && op_stats.recall > 0) {
            cat_stats.recall_values.push_back(op_stats.recall);
            cat_stats.min_recall = std::min(cat_stats.min_recall, op_stats.recall);
            cat_stats.max_recall = std::max(cat_stats.max_recall, op_stats.recall);
        }
    }

public:
    PGVectorSlidingWindowTest(size_t dimension,
                               const std::string& gt_directory,
                               const std::string& db_host,
                               const std::string& db_port,
                               const std::string& db_name,
                               const std::string& db_user,
                               const std::string& db_password,
                               const std::string& index_type_param,
                               int hnsw_m,
                               int hnsw_ef_construction,
                               int hnsw_ef_search,
                               int ivfflat_lists_param,
                               int ivfflat_probes_param,
                               bool enable_crash_simulation = false,
                               size_t crash_step = 0,
                               size_t crash_operation = 0,
                               bool skip_recall = false,
                               size_t txn_batch_size = 0,
                               const std::string& table_name_param = "vectors",
                               size_t dataset_offset_param = 0)
        : dim(dimension), table_name(table_name_param), gt_dir(gt_directory),
          dataset_offset(dataset_offset_param),
          index_type(index_type_param),
          hnsw_m(hnsw_m), hnsw_ef_construction(hnsw_ef_construction), hnsw_ef_search(hnsw_ef_search),
          ivfflat_lists(ivfflat_lists_param), ivfflat_probes(ivfflat_probes_param),
          simulate_crash(enable_crash_simulation), crash_target_step(crash_step),
          crash_target_operation(crash_operation), db_host(db_host), db_port(db_port),
          skip_recall_computation(skip_recall), transaction_batch_size(txn_batch_size) {
        
        // Initialize connection manager
        ThreadLocalConnection::init(
            db_host, db_port, db_name, db_user, db_password
        );

        // Ensure GT directory exists
        std::string cmd = "mkdir -p " + gt_dir;
        (void)system(cmd.c_str());
    }

    ~PGVectorSlidingWindowTest() {
        ThreadLocalConnection::cleanup();
    }

    // Initialize database table
    void setup(bool recreate_table = true) {
        PGconn* conn = ThreadLocalConnection::get_main();
        if (!conn) {
            throw std::runtime_error("No database connection available");
        }

        std::cout << "Setting up database..." << std::endl;

        // Create pgvector extension
        PGresult* res = PQexec(conn, "CREATE EXTENSION IF NOT EXISTS vector");
        PQclear(res);

        if (recreate_table) {
            // Drop old table
            res = PQexec(conn, ("DROP TABLE IF EXISTS " + table_name).c_str());
            PQclear(res);

            // Create new table
            std::stringstream create_sql;
            create_sql << "CREATE TABLE " << table_name << " ("
                       << "id BIGINT PRIMARY KEY, "
                       << "vec vector(" << dim << ")"
                       << ")";
            res = PQexec(conn, create_sql.str().c_str());
            if (PQresultStatus(res) != PGRES_COMMAND_OK) {
                throw std::runtime_error("Failed to create table: " + std::string(PQerrorMessage(conn)));
            }
            PQclear(res);

            std::cout << "✔ Table created: " << table_name << std::endl;
            std::cout << "✔ Database setup complete (index will be created after initial data load)" << std::endl;
        } else {
            std::cout << "✔ Using existing table: " << table_name << std::endl;
        }
    }

    // Create index (HNSW or IVFFlat based on index_type)
    void create_index() {
        PGconn* conn = ThreadLocalConnection::get_main();
        Timer timer;

        PGresult* res = PQexec(conn, "SET maintenance_work_mem = '8GB'");
        PQclear(res);

        if (index_type == "ivfflat") {
            std::cout << "Creating IVFFlat index..." << std::endl;
            std::stringstream index_sql;
            index_sql << "CREATE INDEX ON " << table_name
                      << " USING ivfflat (vec vector_l2_ops) WITH (lists = " << ivfflat_lists << ")";
            res = PQexec(conn, index_sql.str().c_str());
            if (PQresultStatus(res) != PGRES_COMMAND_OK) {
                throw std::runtime_error("Failed to create IVFFlat index: " + std::string(PQerrorMessage(conn)));
            }
            PQclear(res);
            double elapsed = timer.elapsed_seconds();
            std::cout << "✔ IVFFlat index created in " << elapsed << "s" << std::endl;
        } else {
            std::cout << "Creating HNSW index..." << std::endl;
            std::stringstream index_sql;
            index_sql << "CREATE INDEX ON " << table_name
                      << " USING hnsw (vec vector_l2_ops) WITH ("
                      << "m = " << hnsw_m << ", "
                      << "ef_construction = " << hnsw_ef_construction
                      << ")";
            res = PQexec(conn, index_sql.str().c_str());
            if (PQresultStatus(res) != PGRES_COMMAND_OK) {
                throw std::runtime_error("Failed to create HNSW index: " + std::string(PQerrorMessage(conn)));
            }
            PQclear(res);
            double elapsed = timer.elapsed_seconds();
            std::cout << "✔ HNSW index created in " << elapsed << "s" << std::endl;
        }
    }

    // ========================================================================
    // Parallel insert - consistent with hnswlib test logic
    // Outer: batch by checkpoint_size
    // Inner: fine-grained parallel, each vector is an independent task
    // ========================================================================
    std::map<std::string, double> parallel_insert(float* data, size_t start, size_t end,
                                                   int num_threads, size_t checkpoint_size,
                                                   const std::string& step_name, size_t step_num = 0) {
        Timer timer;
        std::cout << "Starting parallel insert: range [" << start << ":" << end
                  << "], threads=" << num_threads << std::endl;

        size_t successful_inserts = 0;
        size_t failed_inserts = 0;

        // Batch processing
        for (size_t batch_start = start; batch_start < end; batch_start += checkpoint_size) {
            size_t batch_end = std::min(batch_start + checkpoint_size, end);

            if (transaction_batch_size > 0) {
                // Transaction batching mode: group operations into transactions
                // Calculate number of transaction batches
                size_t num_txn_batches = (batch_end - batch_start + transaction_batch_size - 1) / transaction_batch_size;
                
                // Process transaction batches in parallel
                #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
                for (size_t txn_idx = 0; txn_idx < num_txn_batches; txn_idx++) {
                    PGconn* conn = ThreadLocalConnection::get();
                    if (!conn) continue;
                    
                    // Check for crash simulation
                    if (should_crash(step_num)) {
                        #pragma omp critical
                        {
                            if (simulate_crash) {
                                kill_postgres_master();
                                simulate_crash = false;
                            }
                        }
                        continue;
                    }
                    
                    // Calculate the range for this transaction batch
                    size_t txn_start = batch_start + txn_idx * transaction_batch_size;
                    size_t txn_end = std::min(txn_start + transaction_batch_size, batch_end);
                    
                    // Execute all INSERTs within a single transaction
                    PGresult* begin_res = PQexec(conn, "BEGIN");
                    if (PQresultStatus(begin_res) == PGRES_COMMAND_OK) {
                        PQclear(begin_res);
                        
                        for (size_t i = txn_start; i < txn_end; i++) {
                            size_t row_id = i + dataset_offset;
                            std::stringstream sql;
                            sql << "INSERT INTO " << table_name << " (id, vec) VALUES ("
                                << row_id << "," << vector_to_sql(data + i * dim, dim)
                                << ") ON CONFLICT (id) DO NOTHING";
                            
                            PGresult* res = PQexec(conn, sql.str().c_str());
                            PQclear(res);
                            // Note: individual INSERT failures are ignored (ON CONFLICT DO NOTHING)
                        }
                        
                        PGresult* commit_res = PQexec(conn, "COMMIT");
                        PQclear(commit_res);
                    } else {
                        PQclear(begin_res);
                    }
                }
                
                // After batch completion, serially update active_indices
                for (size_t i = batch_start; i < batch_end; i++) {
                    active_indices.insert(i);
                }
                
                // Assume all inserts succeeded
                successful_inserts += (batch_end - batch_start);
            } else {
                // Original behavior: each operation is independent (dynamic scheduling)
                #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
                for (size_t i = batch_start; i < batch_end; i++) {
                    PGconn* conn = ThreadLocalConnection::get();
                    if (!conn) continue;

                    // Check for crash simulation (only one thread should trigger)
                    if (should_crash(step_num)) {
                        #pragma omp critical
                        {
                            if (simulate_crash) {
                                kill_postgres_master();
                                simulate_crash = false;  // Prevent multiple kills
                            }
                        }
                        // Exit this thread - crash has been triggered
                        continue;
                    }

                    // Build single INSERT SQL
                    size_t row_id = i + dataset_offset;
                    std::stringstream sql;
                    sql << "INSERT INTO " << table_name << " (id, vec) VALUES ("
                        << row_id << "," << vector_to_sql(data + i * dim, dim)
                        << ") ON CONFLICT (id) DO NOTHING";

                    PGresult* res = PQexec(conn, sql.str().c_str());
                    PQclear(res);
                    // Note: not updating counter here, as hnswlib version also assumes all succeed
                }
                
                // After batch completion, serially update active_indices
                for (size_t i = batch_start; i < batch_end; i++) {
                    active_indices.insert(i);
                }
                
                // Assume all inserts succeeded
                successful_inserts += (batch_end - batch_start);
            }
        }

        // Record active range
        active_ranges.emplace_back("insert", start, end);

        double elapsed = timer.elapsed_seconds();
        double throughput = successful_inserts / elapsed;
        double throughput_per_thread = throughput / num_threads;

        std::cout << "Insert completed in " << elapsed << "s: "
                  << successful_inserts << " successful, " << failed_inserts << " failed"
                  << "\n  Throughput: " << std::fixed << std::setprecision(0)
                  << throughput << " vec/s (total), "
                  << throughput_per_thread << " vec/s/thread" << std::endl;

        // Record statistics
        DetailedStats::OperationStats op_stats;
        op_stats.name = step_name;
        op_stats.type = "insert";
        op_stats.time = elapsed;
        op_stats.successful = successful_inserts;
        op_stats.failed = failed_inserts;
        op_stats.active_count = active_indices.size();
        op_stats.start_idx = start;
        op_stats.end_idx = end;
        op_stats.num_threads = num_threads;
        op_stats.throughput_per_thread = throughput_per_thread;

        stats.operations.push_back(op_stats);
        updateCategoryStats(stats.insert_stats, op_stats);

        return {
            {"successful", (double)successful_inserts},
            {"failed", (double)failed_inserts},
            {"time", elapsed}
        };
    }

    // ================================================
    // Parallel delete - consistent with hnswlib version logic
    // ================================================
    std::map<std::string, double> parallel_delete(size_t start, size_t end,
                                                   int num_threads, size_t checkpoint_size,
                                                   const std::string& step_name, size_t step_num = 0) {
        Timer timer;
        std::cout << "Starting parallel delete: range [" << start << ":" << end
                  << "], threads=" << num_threads << std::endl;

        size_t successful_deletes = 0;
        size_t failed_deletes = 0;

        // Batch processing
        for (size_t batch_start = start; batch_start < end; batch_start += checkpoint_size) {
            size_t batch_end = std::min(batch_start + checkpoint_size, end);
            
            // Track which deletes succeeded (thread-safe using mutex)
            std::vector<bool> delete_success(batch_end - batch_start, false);
            std::mutex delete_success_mutex;
            
            if (transaction_batch_size > 0) {
                // Transaction batching mode: group operations into transactions
                // Calculate number of transaction batches
                size_t num_txn_batches = (batch_end - batch_start + transaction_batch_size - 1) / transaction_batch_size;
                
                // Process transaction batches in parallel
                #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
                for (size_t txn_idx = 0; txn_idx < num_txn_batches; txn_idx++) {
                    PGconn* conn = ThreadLocalConnection::get();
                    if (!conn) continue;
                    
                    // Check for crash simulation
                    if (should_crash(step_num)) {
                        #pragma omp critical
                        {
                            if (simulate_crash) {
                                kill_postgres_master();
                                simulate_crash = false;
                            }
                        }
                        continue;
                    }
                    
                    // Calculate the range for this transaction batch
                    size_t txn_start = batch_start + txn_idx * transaction_batch_size;
                    size_t txn_end = std::min(txn_start + transaction_batch_size, batch_end);
                    
                    // Execute each DELETE individually within a transaction to track results
                    PGresult* begin_res = PQexec(conn, "BEGIN");
                    if (PQresultStatus(begin_res) == PGRES_COMMAND_OK) {
                        PQclear(begin_res);
                        
                        for (size_t i = txn_start; i < txn_end; i++) {
                            size_t row_id = i + dataset_offset;
                            std::stringstream sql;
                            sql << "DELETE FROM " << table_name << " WHERE id = " << row_id;
                            
                            PGresult* res = PQexec(conn, sql.str().c_str());
                            if (PQresultStatus(res) == PGRES_COMMAND_OK) {
                                // Check if any rows were actually deleted
                                char* tuples = PQcmdTuples(res);
                                if (tuples && std::atoi(tuples) > 0) {
                                    size_t offset = i - batch_start;
                                    std::lock_guard<std::mutex> lock(delete_success_mutex);
                                    delete_success[offset] = true;
                                }
                            }
                            PQclear(res);
                        }
                        
                        PGresult* commit_res = PQexec(conn, "COMMIT");
                        PQclear(commit_res);
                    } else {
                        PQclear(begin_res);
                    }
                }
            } else {
                // Original behavior: each operation is independent (dynamic scheduling)
                #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
                for (size_t i = batch_start; i < batch_end; i++) {
                    PGconn* conn = ThreadLocalConnection::get();
                    if (!conn) continue;

                    // Check for crash simulation
                    if (should_crash(step_num)) {
                        #pragma omp critical
                        {
                            if (simulate_crash) {
                                kill_postgres_master();
                                simulate_crash = false;  // Prevent multiple kills
                            }
                        }
                        continue;
                    }

                    // Single DELETE
                    size_t row_id = i + dataset_offset;
                    std::stringstream sql;
                    sql << "DELETE FROM " << table_name << " WHERE id = " << row_id;

                    PGresult* res = PQexec(conn, sql.str().c_str());
                    
                    // Check if delete succeeded and actually deleted a row
                    if (PQresultStatus(res) == PGRES_COMMAND_OK) {
                        char* tuples = PQcmdTuples(res);
                        if (tuples && std::atoi(tuples) > 0) {
                            size_t offset = i - batch_start;
                            std::lock_guard<std::mutex> lock(delete_success_mutex);
                            delete_success[offset] = true;
                        }
                    }
                    
                    PQclear(res);
                }
            }
            
            // After batch completion, serially update active_indices and count results
            for (size_t i = batch_start; i < batch_end; i++) {
                size_t offset = i - batch_start;
                if (delete_success[offset]) {
                    active_indices.erase(i);
                    successful_deletes++;
                } else {
                    failed_deletes++;
                }
            }
        }

        // Record active range
        active_ranges.emplace_back("delete", start, end);

        double elapsed = timer.elapsed_seconds();
        double throughput = successful_deletes / elapsed;
        double throughput_per_thread = throughput / num_threads;

        std::cout << "Delete completed in " << elapsed << "s: "
                  << successful_deletes << " successful, " << failed_deletes << " failed"
                  << "\n  Throughput: " << std::fixed << std::setprecision(0)
                  << throughput << " vec/s (total), "
                  << throughput_per_thread << " vec/s/thread" << std::endl;

        // Record statistics
        DetailedStats::OperationStats op_stats;
        op_stats.name = step_name;
        op_stats.type = "delete";
        op_stats.time = elapsed;
        op_stats.successful = successful_deletes;
        op_stats.failed = failed_deletes;
        op_stats.active_count = active_indices.size();
        op_stats.start_idx = start;
        op_stats.end_idx = end;
        op_stats.num_threads = num_threads;
        op_stats.throughput_per_thread = throughput_per_thread;

        stats.operations.push_back(op_stats);
        updateCategoryStats(stats.delete_stats, op_stats);

        return {
            {"successful", (double)successful_deletes},
            {"failed", (double)failed_deletes},
            {"time", elapsed}
        };
    }

    // ========================================================================
    // Parallel search - consistent with hnswlib version logic
    // ========================================================================
    std::map<std::string, double> parallel_search(float* dataset, float* queries,
                                                  size_t num_queries, size_t k,
                                                  int num_threads, size_t step_num,
                                                  const std::string& runbook_name,
                                                  const std::string& step_name,
                                                  bool skip_recall = false) {
        Timer timer;

        std::vector<std::vector<int>> gt;
        if (!skip_recall) {
            // Load GT file (must exist - should be pre-computed using compute_ground_truth.cpp)
            std::string gt_filename = get_gt_filename(runbook_name, step_num, active_ranges);
            std::string gt_filepath = gt_dir + "/" + gt_filename;

            std::ifstream check(gt_filepath, std::ios::binary);
            if (!check.good()) {
                throw std::runtime_error("Ground truth file not found: " + gt_filepath + 
                                       "\nPlease run compute_ground_truth.cpp first to generate all GT files.");
            }
            check.close();
            std::cout << "Loading ground truth from " << gt_filepath << std::endl;
            gt = load_gt_npy(gt_filepath, num_queries, k);
            if (gt.empty() || gt.size() != num_queries) {
                throw std::runtime_error("Failed to load ground truth from " + gt_filepath);
            }
        }

        std::cout << "Starting parallel search: " << num_queries
                  << " queries, k=" << k << ", threads=" << num_threads;
        if (skip_recall) {
            std::cout << " (recall computation skipped)";
        }
        std::cout << std::endl;

        size_t total_correct = 0;
        size_t total_total = 0;

        // Fine-grained parallel search
        #pragma omp parallel for num_threads(num_threads) reduction(+:total_correct,total_total) schedule(dynamic, 1)
        for (size_t q = 0; q < num_queries; q++) {
            PGconn* conn = ThreadLocalConnection::get();
            if (!conn) continue;

            // Check for crash simulation
            if (should_crash(step_num)) {
                #pragma omp critical
                {
                    if (simulate_crash) {
                        kill_postgres_master();
                        simulate_crash = false;  // Prevent multiple kills
                    }
                }
                continue;
            }

            // Set search parameters (HNSW or IVFFlat)
            std::stringstream set_sql;
            if (index_type == "ivfflat") {
                set_sql << "SET ivfflat.probes = " << ivfflat_probes;
            } else {
                set_sql << "SET hnsw.ef_search = " << hnsw_ef_search;
            }
            PGresult* set_res = PQexec(conn, set_sql.str().c_str());
            PQclear(set_res);

            // Build search SQL
            std::stringstream sql;
            sql << "SELECT id FROM " << table_name
                << " ORDER BY vec <-> " << vector_to_sql(queries + q * dim, dim)
                << " LIMIT " << k;

            PGresult* res = PQexec(conn, sql.str().c_str());
            
            if (PQresultStatus(res) == PGRES_TUPLES_OK && !skip_recall) {
                int nrows = PQntuples(res);
                
                // Calculate recall
                std::unordered_set<int> gt_set(gt[q].begin(), gt[q].end());
                for (int i = 0; i < nrows; i++) {
                    int id = std::stoi(PQgetvalue(res, i, 0));
                    if (gt_set.find(id) != gt_set.end()) {
                        total_correct++;
                    }
                    total_total++;
                }
            }
            
            PQclear(res);
        }

        double elapsed = timer.elapsed_seconds();
        double recall = skip_recall ? 0.0 : (total_total > 0 ? (double)total_correct / (double)total_total : 0.0);
        double qps = num_queries / elapsed;
        double qps_per_thread = qps / num_threads;

        std::cout << "Search completed in " << elapsed << "s";
        if (!skip_recall) {
            std::cout << ": Recall@" << k << " = " << std::fixed << std::setprecision(4) << recall;
        }
        std::cout << "\n  QPS: " << std::fixed << std::setprecision(0) << qps
                  << " (total), " << qps_per_thread << " q/s/thread" << std::endl;

        // Record statistics
        DetailedStats::OperationStats op_stats;
        op_stats.name = step_name;
        op_stats.type = "search";
        op_stats.time = elapsed;
        op_stats.successful = num_queries;
        op_stats.failed = 0;
        op_stats.recall = recall;
        op_stats.k = k;
        op_stats.active_count = active_indices.size();
        op_stats.num_threads = num_threads;
        op_stats.throughput_per_thread = qps_per_thread;

        stats.operations.push_back(op_stats);
        updateCategoryStats(stats.search_stats, op_stats);

        return {
            {"recall", recall},
            {"total_correct", (double)total_correct},
            {"total_total", (double)total_total},
            {"time", elapsed}
        };
    }

    // Set crash simulation target (random step and operation)
    void set_crash_target(size_t start_step, size_t end_step, 
                          const std::string& runbook_file, const std::string& dataset_name,
                          size_t num_queries) {
        if (!simulate_crash) return;
        
        YAML::Node runbook = YAML::LoadFile(runbook_file);
        auto operations = runbook[dataset_name];
        
        // Collect all steps in range to calculate total operations
        std::vector<std::pair<size_t, size_t>> step_operations;  // step_num -> num_operations
        size_t step_num = 0;
        
        for (auto it = operations.begin(); it != operations.end(); ++it) {
            std::string step_key = it->first.as<std::string>();
            if (step_key == "max_pts" || step_key == "query" || step_key == "groundtruth") {
                continue;
            }
            step_num++;
            
            if (start_step > 0 && step_num < start_step) continue;
            if (end_step > 0 && step_num > end_step) break;
            
            auto step = it->second;
            std::string op = step["operation"].as<std::string>();
            size_t num_ops = 0;
            
            if (op == "insert" || op == "delete") {
                size_t start = step["start"].as<size_t>(0);
                size_t end = step["end"].as<size_t>(0);
                num_ops = end - start;
            } else if (op == "search") {
                num_ops = num_queries;
            }
            
            if (num_ops > 0) {
                step_operations.push_back({step_num, num_ops});
            }
        }
        
        if (step_operations.empty()) {
            std::cerr << "[CRASH SIMULATION] Warning: No operations found in range, disabling crash simulation" << std::endl;
            simulate_crash = false;
            return;
        }
        
        // Randomly select a step
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> step_dist(0, step_operations.size() - 1);
        size_t selected_idx = step_dist(gen);
        crash_target_step = step_operations[selected_idx].first;
        
        // Randomly select an operation within that step
        std::uniform_int_distribution<size_t> op_dist(1, step_operations[selected_idx].second);
        crash_target_operation = op_dist(gen);
        
        std::cout << "\n[CRASH SIMULATION] Target set: Step " << crash_target_step 
                  << ", Operation " << crash_target_operation << " (out of " 
                  << step_operations[selected_idx].second << " operations in this step)" << std::endl;
    }

    // Execute runbook
    void execute_runbook(const std::string& runbook_file, const std::string& dataset_name,
                         float* dataset, float* queries, size_t num_queries,
                         int num_threads, size_t checkpoint_size,
                         size_t start_step = 0, size_t end_step = 0, size_t build_index_before = 0,
                         size_t mixed_mode_start = 0, size_t mixed_size = 0) {
        Timer overall_timer;

        YAML::Node runbook = YAML::LoadFile(runbook_file);
        auto operations = runbook[dataset_name];
        
        // Set crash target if crash simulation is enabled
        if (simulate_crash) {
            set_crash_target(start_step, end_step, runbook_file, dataset_name, num_queries);
        }

        // Initialize database - only drop and recreate table if starting from the beginning
        bool recreate_table = (start_step == 0 || start_step == 1);
        setup(recreate_table);

        std::string runbook_name = dataset_name;
        std::replace(runbook_name.begin(), runbook_name.end(), '-', '_');

        bool index_created = false;
        size_t step_num = 0;
        bool in_mixed_mode = (mixed_mode_start > 0 && mixed_size > 0);

        // First pass: build active_ranges from the beginning (needed for correct GT filename hash)
        // This ensures the hash matches compute_ground_truth.cpp which processes all steps
        if (start_step > 0) {
            size_t range_step_num = 0;
            for (auto it = operations.begin(); it != operations.end(); ++it) {
                std::string step_key = it->first.as<std::string>();
                if (step_key == "max_pts" || step_key == "query" || step_key == "groundtruth") {
                    continue;
                }
                range_step_num++;
                if (range_step_num >= start_step) {
                    break;  // Stop when we reach start_step
                }
                auto step = it->second;
                std::string op = step["operation"].as<std::string>();
                if (op == "insert" || op == "delete") {
                    size_t start = step["start"].as<size_t>(0);
                    size_t end = step["end"].as<size_t>(0);
                    active_ranges.emplace_back(op, start, end);
                }
            }
        }

        // Collect all steps first for mixed mode processing
        struct StepInfo {
            size_t step_num;
            std::string step_key;
            YAML::Node step;
            std::string op;
        };
        std::vector<StepInfo> all_steps;

        for (auto it = operations.begin(); it != operations.end(); ++it) {
            std::string step_key = it->first.as<std::string>();
            if (step_key == "max_pts" || step_key == "query" || step_key == "groundtruth") {
                continue;
            }
            step_num++;
            if (start_step > 0 && step_num < start_step) {
                continue;
            }
            if (end_step > 0 && step_num > end_step) {
                break;
            }
            auto step = it->second;
            std::string op = step["operation"].as<std::string>();
            all_steps.push_back({step_num, step_key, step, op});
        }

        // Execute operations - sequential or mixed mode
        step_num = 0;
        for (size_t i = 0; i < all_steps.size(); i++) {
            const auto& step_info = all_steps[i];
            step_num = step_info.step_num;

            // Create index before the specified step
            if (build_index_before > 0 && !index_created && step_num == build_index_before) {
                create_index();
                index_created = true;
            }

            // Check if we should enter mixed mode
            bool should_use_mixed_mode = in_mixed_mode && step_num >= mixed_mode_start;
            
            if (should_use_mixed_mode) {
                // Collect steps for this mixed mode group
                std::vector<StepInfo> mixed_steps;
                for (size_t j = i; j < all_steps.size() && mixed_steps.size() < mixed_size; j++) {
                    if (all_steps[j].step_num >= mixed_mode_start) {
                        mixed_steps.push_back(all_steps[j]);
                        i = j;  // Update outer loop index
                    } else {
                        break;
                    }
                }

                if (!mixed_steps.empty()) {
                    std::cout << "\n========================================" << std::endl;
                    std::cout << "MIXED MODE: Processing " << mixed_steps.size() 
                              << " steps concurrently with " << num_threads << " threads" << std::endl;
                    std::cout << "========================================" << std::endl;

                    Timer mixed_timer;
                    
                    // Structure to hold step information (no atomic members - they're stored separately)
                    struct StepInfo {
                        std::string op;
                        size_t start;
                        size_t end;
                        size_t total_items;
                        size_t step_num;
                        std::string step_key;
                        size_t k;  // For search operations
                    };
                    
                    std::vector<StepInfo> step_infos;
                    // Store atomic counters separately using unique_ptr (atomic is not copyable/movable)
                    // We use unique_ptr to avoid vector reallocation issues
                    std::vector<std::unique_ptr<std::atomic<size_t>>> step_next_items;
                    std::vector<size_t> step_successful_counts(mixed_steps.size(), 0);
                    std::vector<size_t> step_failed_counts(mixed_steps.size(), 0);
                    std::mutex counters_mutex;  // For protecting counter updates
                    std::mutex stats_mutex;  // For protecting active_indices updates
                    
                    // Initialize step information
                    size_t total_operations = 0;
                    step_infos.reserve(mixed_steps.size());
                    for (size_t step_idx = 0; step_idx < mixed_steps.size(); step_idx++) {
                        const auto& mixed_step = mixed_steps[step_idx];
                        
                        std::string op = mixed_step.op;
                        size_t step_num = mixed_step.step_num;
                        std::string step_key = mixed_step.step_key;
                        size_t start, end, total_items, k = 0;
                        
                        if (mixed_step.op == "insert" || mixed_step.op == "delete") {
                            start = mixed_step.step["start"].as<size_t>(0);
                            end = mixed_step.step["end"].as<size_t>(0);
                            total_items = end - start;
                            k = 0;
                        } else if (mixed_step.op == "search") {
                            start = 0;
                            end = num_queries;
                            total_items = num_queries;
                            k = mixed_step.step["k"].as<size_t>(100);
                        } else {
                            continue;  // Skip unknown operations
                        }
                        
                        // Add step info (copyable struct)
                        step_infos.push_back({op, start, end, total_items, step_num, step_key, k});
                        // Add atomic counter separately using unique_ptr
                        step_next_items.push_back(std::make_unique<std::atomic<size_t>>(0));
                        
                        total_operations += total_items;
                    }
                    
                    if (step_infos.empty()) {
                        std::cout << "No valid operations in mixed mode" << std::endl;
                        continue;
                    }
                    
                    // Display work distribution
                    std::cout << "Work distribution:" << std::endl;
                    for (size_t i = 0; i < step_infos.size(); i++) {
                        const auto& info = step_infos[i];
                        std::cout << "  Step " << info.step_num << " (" << info.step_key 
                                  << "): " << info.op << " - " << info.total_items 
                                  << " operations" << std::endl;
                    }
                    std::cout << "  Total operations: " << total_operations << std::endl;
                    
                    // Generate array: each element is a step index (0 to mixed_size-1)
                    // Count of step index i equals the number of operations in step i
                    std::vector<size_t> step_index_array;
                    step_index_array.reserve(total_operations);
                    for (size_t step_idx = 0; step_idx < step_infos.size(); step_idx++) {
                        for (size_t i = 0; i < step_infos[step_idx].total_items; i++) {
                            step_index_array.push_back(step_idx);
                        }
                    }
                    
                    // Shuffle the array to randomize work distribution
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::shuffle(step_index_array.begin(), step_index_array.end(), gen);
                    
                    std::cout << "  Generated shuffled work array of size " << step_index_array.size() << std::endl;
                    
                    // Process the step_index_array in parallel
                    // Each thread processes elements from the array and executes the corresponding operation
                    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 100)
                    for (size_t array_idx = 0; array_idx < step_index_array.size(); array_idx++) {
                        size_t step_idx = step_index_array[array_idx];
                        StepInfo& step_info = step_infos[step_idx];
                        
                        // Atomically get the next operation index within this step (maintains order)
                        size_t item_idx = step_next_items[step_idx]->fetch_add(1, std::memory_order_relaxed);
                        
                        // Safety check (should never happen if array is correctly generated)
                        if (item_idx >= step_info.total_items) {
                            continue;
                        }
                        
                        PGconn* conn = ThreadLocalConnection::get();
                        if (!conn) {
                            std::lock_guard<std::mutex> lock(counters_mutex);
                            step_failed_counts[step_idx]++;
                            continue;
                        }
                        
                        // Check for crash simulation
                        if (should_crash(step_info.step_num)) {
                            #pragma omp critical
                            {
                                if (simulate_crash) {
                                    kill_postgres_master();
                                    simulate_crash = false;  // Prevent multiple kills
                                }
                            }
                            continue;
                        }
                        
                        // Execute the operation
                        if (step_info.op == "insert") {
                            size_t vec_idx = step_info.start + item_idx;
                            size_t row_id = vec_idx + dataset_offset;
                            std::stringstream sql;
                            sql << "INSERT INTO " << table_name << " (id, vec) VALUES ("
                                << row_id << "," << vector_to_sql(dataset + vec_idx * dim, dim)
                                << ") ON CONFLICT (id) DO NOTHING";
                            PGresult* res = PQexec(conn, sql.str().c_str());
                            {
                                std::lock_guard<std::mutex> lock(counters_mutex);
                                if (PQresultStatus(res) == PGRES_COMMAND_OK) {
                                    step_successful_counts[step_idx]++;
                                } else {
                                    step_failed_counts[step_idx]++;
                                }
                            }
                            PQclear(res);
                        }
                        else if (step_info.op == "delete") {
                            size_t vec_idx = step_info.start + item_idx;
                            size_t row_id = vec_idx + dataset_offset;
                            std::stringstream sql;
                            sql << "DELETE FROM " << table_name << " WHERE id = " << row_id;
                            PGresult* res = PQexec(conn, sql.str().c_str());
                            {
                                std::lock_guard<std::mutex> lock(counters_mutex);
                                if (PQresultStatus(res) == PGRES_COMMAND_OK) {
                                    step_successful_counts[step_idx]++;
                                } else {
                                    step_failed_counts[step_idx]++;
                                }
                            }
                            PQclear(res);
                        }
                        else if (step_info.op == "search") {
                            if (queries != nullptr) {
                                // Set search parameters (HNSW or IVFFlat)
                                std::stringstream set_sql;
                                if (index_type == "ivfflat") {
                                    set_sql << "SET ivfflat.probes = " << ivfflat_probes;
                                } else {
                                    set_sql << "SET hnsw.ef_search = " << hnsw_ef_search;
                                }
                                PGresult* set_res = PQexec(conn, set_sql.str().c_str());
                                PQclear(set_res);
                                
                                // Build search SQL
                                std::stringstream sql;
                                sql << "SELECT id FROM " << table_name
                                    << " ORDER BY vec <-> " << vector_to_sql(queries + item_idx * dim, dim)
                                    << " LIMIT " << step_info.k;
                                PGresult* res = PQexec(conn, sql.str().c_str());
                                {
                                    std::lock_guard<std::mutex> lock(counters_mutex);
                                    if (PQresultStatus(res) == PGRES_TUPLES_OK) {
                                        step_successful_counts[step_idx]++;
                                    } else {
                                        step_failed_counts[step_idx]++;
                                    }
                                }
                                PQclear(res);
                            }
                        }
                    }
                    
                    // Final update of active_indices and active_ranges
                    {
                        std::lock_guard<std::mutex> lock(stats_mutex);
                        for (const auto& step_info : step_infos) {
                            if (step_info.op == "insert") {
                                for (size_t i = step_info.start; i < step_info.end; i++) {
                                    active_indices.insert(i);
                                }
                                active_ranges.emplace_back("insert", step_info.start, step_info.end);
                            } else if (step_info.op == "delete") {
                                for (size_t i = step_info.start; i < step_info.end; i++) {
                                    active_indices.erase(i);
                                }
                                active_ranges.emplace_back("delete", step_info.start, step_info.end);
                            }
                        }
                    }
                    
                    double mixed_elapsed = mixed_timer.elapsed_seconds();
                    
                    // Record statistics for each step
                    for (size_t step_idx = 0; step_idx < step_infos.size(); step_idx++) {
                        const auto& step_info = step_infos[step_idx];
                        size_t successful = step_successful_counts[step_idx];
                        size_t failed = step_failed_counts[step_idx];
                        double throughput = successful / mixed_elapsed;
                        
                        std::cout << "  [Mixed] Step " << step_info.step_num << " - " 
                                  << step_info.step_key << " (" << step_info.op << "): "
                                  << successful << " successful, " << failed << " failed, "
                                  << std::fixed << std::setprecision(2) << mixed_elapsed << "s, "
                                  << std::setprecision(0) << throughput << " ops/s" << std::endl;
                        
                        DetailedStats::OperationStats op_stats;
                        op_stats.name = step_info.step_key;
                        op_stats.type = step_info.op;
                        op_stats.time = mixed_elapsed;
                        op_stats.successful = successful;
                        op_stats.failed = failed;
                        op_stats.active_count = active_indices.size();
                        op_stats.start_idx = step_info.start;
                        op_stats.end_idx = step_info.end;
                        op_stats.num_threads = num_threads;
                        op_stats.throughput_per_thread = throughput / num_threads;
                        if (step_info.op == "search") {
                            op_stats.recall = 0.0;  // Skip recall in mixed mode
                            op_stats.k = step_info.k;
                        }
                        
                        stats.operations.push_back(op_stats);
                        if (step_info.op == "insert") {
                            updateCategoryStats(stats.insert_stats, op_stats);
                        } else if (step_info.op == "delete") {
                            updateCategoryStats(stats.delete_stats, op_stats);
                        } else if (step_info.op == "search") {
                            updateCategoryStats(stats.search_stats, op_stats);
                        }
                    }
                    
                    std::cout << "✔ Mixed mode completed in " << std::fixed << std::setprecision(2) 
                              << mixed_elapsed << "s" << std::endl;
                }
            } else {
                // Sequential mode (original behavior)
                std::cout << "\n========================================" << std::endl;
                std::cout << "Step " << step_num << " - " << step_info.step_key << ": " << step_info.op << std::endl;
                std::cout << "========================================" << std::endl;

                std::map<std::string, double> result;

                if (step_info.op == "insert") {
                    size_t start = step_info.step["start"].as<size_t>(0);
                    size_t end = step_info.step["end"].as<size_t>(0);
                    result = parallel_insert(dataset, start, end, num_threads, checkpoint_size, step_info.step_key, step_num);
                }
                else if (step_info.op == "delete") {
                    size_t start = step_info.step["start"].as<size_t>(0);
                    size_t end = step_info.step["end"].as<size_t>(0);
                    result = parallel_delete(start, end, num_threads, checkpoint_size, step_info.step_key, step_num);
                }
                else if (step_info.op == "search") {
                    if (queries != nullptr) {
                        size_t k = step_info.step["k"].as<size_t>(100);
                        result = parallel_search(dataset, queries, num_queries, k,
                                                 num_threads, step_num, runbook_name, step_info.step_key, skip_recall_computation);
                    } else {
                        std::cout << "Skipping search - no queries available" << std::endl;
                        result = { {"time", 0.0} };
                    }
                }

                std::cout << "✔ Step " << step_info.step_key << " completed in "
                          << result["time"] << " seconds" << std::endl;
            }
        }

        stats.total_time = overall_timer.elapsed_seconds();
        print_summary();
    }

    void print_summary() {
        std::cout << "\n╔══════════════════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                      PGVECTOR SLIDING WINDOW TEST SUMMARY                    ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝" << std::endl;

        std::cout << "\n┌───────────────────────── OVERALL STATISTICS ────────────────────────────────┐" << std::endl;
        std::cout << "│ Total Test Time: " << std::fixed << std::setprecision(2) 
                  << stats.total_time << " seconds" << std::endl;
        std::cout << "│ Total Operations: " << stats.operations.size() << std::endl;
        std::cout << "│   - Inserts: " << stats.insert_stats.count << std::endl;
        std::cout << "│   - Deletes: " << stats.delete_stats.count << std::endl;
        std::cout << "│   - Searches: " << stats.search_stats.count << std::endl;
        std::cout << "│ Max Threads Used: " << stats.max_threads_used << std::endl;
        std::cout << "└──────────────────────────────────────────────────────────────────────────────┘" << std::endl;

        std::cout << "\n┌───────────────────────── PERFORMANCE SUMMARY ───────────────────────────────┐" << std::endl;
        std::cout << "│ Peak INSERT Throughput: " << std::fixed << std::setprecision(0) 
                  << stats.peak_insert_throughput << " vec/s" << std::endl;
        std::cout << "│   Per-thread: " << stats.peak_insert_throughput_per_thread << " vec/s/thread" << std::endl;
        std::cout << "│ Peak DELETE Throughput: " << stats.peak_delete_throughput << " vec/s" << std::endl;
        std::cout << "│   Per-thread: " << stats.peak_delete_throughput_per_thread << " vec/s/thread" << std::endl;
        std::cout << "│ Peak SEARCH QPS: " << stats.peak_search_qps << " q/s" << std::endl;
        std::cout << "│   Per-thread: " << stats.peak_search_qps_per_thread << " q/s/thread" << std::endl;

        if (!stats.search_stats.recall_values.empty()) {
            double avg_recall = 0;
            for (double r : stats.search_stats.recall_values) avg_recall += r;
            avg_recall /= stats.search_stats.recall_values.size();
            
            std::cout << "│" << std::endl;
            std::cout << "│ RECALL Statistics:" << std::endl;
            std::cout << "│   Average: " << std::fixed << std::setprecision(4) << avg_recall << std::endl;
            std::cout << "│   Min: " << stats.search_stats.min_recall << std::endl;
            std::cout << "│   Max: " << stats.search_stats.max_recall << std::endl;
        }
        std::cout << "└──────────────────────────────────────────────────────────────────────────────┘" << std::endl;

        // Detailed operation log
        std::cout << "\n┌───────────────────────── DETAILED OPERATION LOG ────────────────────────────┐" << std::endl;
        std::cout << "│ " << std::left 
                  << std::setw(12) << "Step" 
                  << std::setw(8) << "Type"
                  << std::setw(10) << "Time(s)"
                  << std::setw(10) << "Success"
                  << std::setw(8) << "Threads"
                  << std::setw(12) << "TP/Thread"
                  << std::setw(10) << "Recall" << " │" << std::endl;
        std::cout << "│ " << std::string(68, '-') << " │" << std::endl;
        
        for (const auto& op : stats.operations) {
            std::cout << "│ " << std::left 
                      << std::setw(12) << op.name.substr(0, 11)
                      << std::setw(8) << op.type.substr(0, 7)
                      << std::setw(10) << std::fixed << std::setprecision(2) << op.time
                      << std::setw(10) << op.successful
                      << std::setw(8) << op.num_threads
                      << std::setw(12) << std::fixed << std::setprecision(0) << op.throughput_per_thread;
            
            if (op.type == "search" && op.recall > 0) {
                std::cout << std::setw(10) << std::fixed << std::setprecision(4) << op.recall;
            } else {
                std::cout << std::setw(10) << "-";
            }
            
            std::cout << " │" << std::endl;
        }
        std::cout << "└──────────────────────────────────────────────────────────────────────────────┘" << std::endl;

        std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║            TEST COMPLETED SUCCESSFULLY                       ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    }
};

// ===============================
// Main function
// ===============================
void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " <num_threads> [options]" << std::endl;
    std::cerr << "\nRequired arguments:" << std::endl;
    std::cerr << "  <num_threads>              Number of threads for parallel operations" << std::endl;
    std::cerr << "\nOptional arguments:" << std::endl;
    std::cerr << "  --checkpoint-size <size>    Checkpoint size for batching (default: 1000)" << std::endl;
    std::cerr << "  --dataset <path>            Path to dataset .fvecs file (required)" << std::endl;
    std::cerr << "  --queries <path>             Path to queries .fvecs file (required)" << std::endl;
    std::cerr << "  --runbook <path>            Path to runbook YAML file (required)" << std::endl;
    std::cerr << "  --dataset-name <name>        Dataset name in runbook (required)" << std::endl;
    std::cerr << "  --gt-dir <path>             Ground truth directory (default: ./ground_truth)" << std::endl;
    std::cerr << "  --start-step <num>          First step to process (1-based, default: process all)" << std::endl;
    std::cerr << "  --end-step <num>            Last step to process (1-based, default: process all)" << std::endl;
    std::cerr << "  --build-index-before <num>  Create index before this step (1-based, default: auto)" << std::endl;
    std::cerr << "  --mixed-mode-start <num>    Start step for mixed mode (1-based, default: disabled)" << std::endl;
    std::cerr << "  --mixed-size <size>         Number of steps per group in mixed mode (default: 0, disabled)" << std::endl;
    std::cerr << "  --db-host <host>            PostgreSQL host (default: localhost)" << std::endl;
    std::cerr << "  --db-port <port>            PostgreSQL port (default: 5432)" << std::endl;
    std::cerr << "  --db-name <name>            PostgreSQL database name (default: vector_benchmark)" << std::endl;
    std::cerr << "  --db-user <user>            PostgreSQL user (default: postgres)" << std::endl;
    std::cerr << "  --db-password <password>    PostgreSQL password (default: empty)" << std::endl;
    std::cerr << "  --index-type <type>         Index type: hnsw or ivfflat (default: hnsw)" << std::endl;
    std::cerr << "  --hnsw-m <m>                HNSW M parameter (default: 48, used when index-type=hnsw)" << std::endl;
    std::cerr << "  --hnsw-ef-construction <ef> HNSW ef_construction parameter (default: 400)" << std::endl;
    std::cerr << "  --hnsw-ef-search <ef>       HNSW ef_search parameter (default: 200)" << std::endl;
    std::cerr << "  --ivfflat-lists <n>         IVFFlat lists parameter (default: 100, used when index-type=ivfflat)" << std::endl;
    std::cerr << "  --ivfflat-probes <n>        IVFFlat probes parameter (default: 10)" << std::endl;
    std::cerr << "  --table-name <name>         Table name (default: vectors)" << std::endl;
    std::cerr << "  --dataset-offset <n>        Offset added to vec_idx for insert/delete (default: 0)" << std::endl;
    std::cerr << "  --simulate-crash            Enable crash simulation (kills PostgreSQL at random step/operation)" << std::endl;
    std::cerr << "  --skip-recall               Skip recall computation for search operations (default: false)" << std::endl;
    std::cerr << "  --transaction-batch-size <n> Group n operations per transaction (default: 0, disabled)" << std::endl;
    std::cerr << "                               Only applies when mixed mode is NOT enabled" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    Config config;
    int num_threads = std::atoi(argv[1]);
    size_t checkpoint_size = 1000;
    size_t start_step = 0;
    size_t end_step = 0;
    size_t build_index_before = 0;
    size_t mixed_mode_start = 0;
    size_t mixed_size = 0;
    bool simulate_crash = false;
    bool skip_recall = false;
    size_t transaction_batch_size = 0;

    // Parse command line arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--checkpoint-size" && i + 1 < argc) {
            checkpoint_size = (size_t)std::atoi(argv[++i]);
        } else if (arg == "--dataset" && i + 1 < argc) {
            config.DATASET_PATH = argv[++i];
        } else if (arg == "--queries" && i + 1 < argc) {
            config.QUERY_PATH = argv[++i];
        } else if (arg == "--runbook" && i + 1 < argc) {
            config.RUNBOOK_PATH = argv[++i];
        } else if (arg == "--dataset-name" && i + 1 < argc) {
            config.DATASET_NAME = argv[++i];
        } else if (arg == "--gt-dir" && i + 1 < argc) {
            config.GT_DIR = argv[++i];
        } else if (arg == "--start-step" && i + 1 < argc) {
            start_step = (size_t)std::atoi(argv[++i]);
        } else if (arg == "--end-step" && i + 1 < argc) {
            end_step = (size_t)std::atoi(argv[++i]);
        } else if (arg == "--build-index-before" && i + 1 < argc) {
            build_index_before = (size_t)std::atoi(argv[++i]);
        } else if (arg == "--mixed-mode-start" && i + 1 < argc) {
            mixed_mode_start = (size_t)std::atoi(argv[++i]);
        } else if (arg == "--mixed-size" && i + 1 < argc) {
            mixed_size = (size_t)std::atoi(argv[++i]);
        } else if (arg == "--db-host" && i + 1 < argc) {
            config.DB_HOST = argv[++i];
        } else if (arg == "--db-port" && i + 1 < argc) {
            config.DB_PORT = argv[++i];
        } else if (arg == "--db-name" && i + 1 < argc) {
            config.DB_NAME = argv[++i];
        } else if (arg == "--db-user" && i + 1 < argc) {
            config.DB_USER = argv[++i];
        } else if (arg == "--db-password" && i + 1 < argc) {
            config.DB_PASSWORD = argv[++i];
        } else if (arg == "--index-type" && i + 1 < argc) {
            config.INDEX_TYPE = argv[++i];
            if (config.INDEX_TYPE != "hnsw" && config.INDEX_TYPE != "ivfflat") {
                std::cerr << "Error: --index-type must be 'hnsw' or 'ivfflat'" << std::endl;
                return 1;
            }
        } else if (arg == "--hnsw-m" && i + 1 < argc) {
            config.HNSW_M = std::atoi(argv[++i]);
        } else if (arg == "--hnsw-ef-construction" && i + 1 < argc) {
            config.HNSW_EF_CONSTRUCTION = std::atoi(argv[++i]);
        } else if (arg == "--hnsw-ef-search" && i + 1 < argc) {
            config.HNSW_EF_SEARCH = std::atoi(argv[++i]);
        } else if (arg == "--ivfflat-lists" && i + 1 < argc) {
            config.IVFFLAT_LISTS = std::atoi(argv[++i]);
        } else if (arg == "--ivfflat-probes" && i + 1 < argc) {
            config.IVFFLAT_PROBES = std::atoi(argv[++i]);
        } else if (arg == "--table-name" && i + 1 < argc) {
            config.TABLE_NAME = argv[++i];
        } else if (arg == "--dataset-offset" && i + 1 < argc) {
            config.DATASET_OFFSET = (size_t)std::atoll(argv[++i]);
        } else if (arg == "--simulate-crash") {
            simulate_crash = true;
        } else if (arg == "--skip-recall") {
            skip_recall = true;
        } else if (arg == "--transaction-batch-size" && i + 1 < argc) {
            transaction_batch_size = (size_t)std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    if (config.DATASET_PATH.empty() || config.QUERY_PATH.empty() || 
        config.RUNBOOK_PATH.empty() || config.DATASET_NAME.empty()) {
        std::cerr << "Error: --dataset, --queries, --runbook, and --dataset-name are required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    if (config.GT_DIR.empty()) {
        config.GT_DIR = "./ground_truth";
    }

    // Validate: transaction batching only works when mixed mode is NOT enabled
    if (transaction_batch_size > 0 && mixed_mode_start > 0 && mixed_size > 0) {
        std::cerr << "Error: --transaction-batch-size cannot be used with mixed mode" << std::endl;
        std::cerr << "Transaction batching is only available when mixed mode is disabled" << std::endl;
        return 1;
    }

    // Set OpenMP thread count
    omp_set_num_threads(num_threads);

    std::cout << "========================================" << std::endl;
    std::cout << "PGVECTOR SLIDING WINDOW TEST (C++) [" << config.INDEX_TYPE << "]" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Threads: " << num_threads << std::endl;
    std::cout << "  Dataset: " << config.DATASET_PATH << std::endl;
    std::cout << "  Queries: " << config.QUERY_PATH << std::endl;
    std::cout << "  Runbook: " << config.RUNBOOK_PATH << std::endl;
    std::cout << "  Dataset name: " << config.DATASET_NAME << std::endl;
    std::cout << "  GT directory: " << config.GT_DIR << std::endl;
    std::cout << "  Checkpoint size: " << checkpoint_size << std::endl;
    if (start_step > 0) {
        std::cout << "  Start step: " << start_step << std::endl;
    }
    if (end_step > 0) {
        std::cout << "  End step: " << end_step << std::endl;
    }
    if (build_index_before > 0) {
        std::cout << "  Build index before step: " << build_index_before << std::endl;
    }
    if (mixed_mode_start > 0 && mixed_size > 0) {
        std::cout << "  Mixed mode: enabled (start at step " << mixed_mode_start 
                  << ", mixed size: " << mixed_size << ")" << std::endl;
    }
    if (simulate_crash) {
        std::cout << "  Crash simulation: ENABLED (will kill PostgreSQL at random step/operation)" << std::endl;
    }
    if (skip_recall) {
        std::cout << "  Skip recall computation: ENABLED" << std::endl;
    }
    if (transaction_batch_size > 0) {
        std::cout << "  Transaction batching: enabled (" << transaction_batch_size 
                  << " operations per transaction)" << std::endl;
    }
    std::cout << "  Table name: " << config.TABLE_NAME << std::endl;
    if (config.DATASET_OFFSET > 0) {
        std::cout << "  Dataset offset: " << config.DATASET_OFFSET << std::endl;
    }
    std::cout << "  DB Host: " << config.DB_HOST << std::endl;
    std::cout << "  DB Port: " << config.DB_PORT << std::endl;
    std::cout << "  DB Name: " << config.DB_NAME << std::endl;
    std::cout << "  DB User: " << config.DB_USER << std::endl;
    std::cout << "  Index type: " << config.INDEX_TYPE << std::endl;
    if (config.INDEX_TYPE == "ivfflat") {
        std::cout << "  IVFFlat lists: " << config.IVFFLAT_LISTS << std::endl;
        std::cout << "  IVFFlat probes: " << config.IVFFLAT_PROBES << std::endl;
    } else {
        std::cout << "  HNSW M: " << config.HNSW_M << std::endl;
        std::cout << "  HNSW ef_construction: " << config.HNSW_EF_CONSTRUCTION << std::endl;
        std::cout << "  HNSW ef_search: " << config.HNSW_EF_SEARCH << std::endl;
    }
    std::cout << "========================================\n" << std::endl;

    try {
        // Load dataset
        std::cout << "Loading dataset..." << std::endl;
        size_t num_vectors, data_dim;
        auto dataset = DataLoader::load(config.DATASET_PATH, num_vectors, data_dim);
        std::cout << "✔ Dataset loaded: " << num_vectors << " vectors, dim=" << data_dim << std::endl;

        // Load queries
        std::cout << "Loading queries..." << std::endl;
        size_t num_queries, query_dim;
        auto queries = DataLoader::load(config.QUERY_PATH, num_queries, query_dim);
        std::cout << "✔ Queries loaded: " << num_queries << " queries, dim=" << query_dim << std::endl;

        // Auto-detect dimension and validate
        if (data_dim != query_dim) {
            throw std::runtime_error("Dataset and query dimensions do not match: " + 
                                   std::to_string(data_dim) + " vs " + std::to_string(query_dim));
        }
        config.DIMENSION = data_dim;
        std::cout << "✔ Auto-detected dimension: " << config.DIMENSION << std::endl;

        // Create test object
        PGVectorSlidingWindowTest test(
            config.DIMENSION,
            config.GT_DIR,
            config.DB_HOST,
            config.DB_PORT,
            config.DB_NAME,
            config.DB_USER,
            config.DB_PASSWORD,
            config.INDEX_TYPE,
            config.HNSW_M,
            config.HNSW_EF_CONSTRUCTION,
            config.HNSW_EF_SEARCH,
            config.IVFFLAT_LISTS,
            config.IVFFLAT_PROBES,
            simulate_crash,
            0,  // crash_step (will be set by set_crash_target)
            0,  // crash_operation (will be set by set_crash_target)
            skip_recall,
            transaction_batch_size,
            config.TABLE_NAME,
            config.DATASET_OFFSET
        );

        // Execute test
        test.execute_runbook(config.RUNBOOK_PATH, config.DATASET_NAME,
                             dataset.data(), queries.data(), num_queries,
                             num_threads, checkpoint_size,
                             start_step, end_step, build_index_before,
                             mixed_mode_start, mixed_size);

        std::cout << "\n========================================" << std::endl;
        std::cout << "TEST COMPLETED SUCCESSFULLY" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


// ./pgvector_test 1 \
// --dataset /ssd_root/dataset/turing10m/msturing-10M.fvecs \
// --queries /ssd_root/dataset/turing10m/msturing-query.fvecs \
// --runbook msturing-10M_slidingwindow_runbook.yaml \
// --dataset-name msturing-10M \
// --checkpoint-size 1000 \
// --gt-dir /ssd_root/liu4127/msturing_runbook_gt \
// --db-host localhost \
// --db-port 5434 \
// --db-name postgres \
// --db-user liu4127 \
// --db-password "" \
// --hnsw-m 16 \
// --hnsw-ef-construction 40 \
// --hnsw-ef-search 200 \
// --start-step 101  --end-step 400  --build-index-before 101 \
// --mixed-mode-start 101  --mixed-size 3


// test recovery
// ./pgvector_test 1 \
// --dataset /ssd_root/dataset/turing10m/msturing-10M.fvecs \
// --queries /ssd_root/dataset/turing10m/msturing-query.fvecs \
// --runbook msturing-10M_slidingwindow_runbook.yaml \
// --dataset-name msturing-10M \
// --checkpoint-size 1000 \
// --gt-dir /ssd_root/liu4127/msturing_runbook_gt \
// --db-host localhost \
// --db-port 5434 \
// --db-name postgres \
// --db-user liu4127 \
// --db-password "" \
// --hnsw-m 16 \
// --hnsw-ef-construction 40 \
// --hnsw-ef-search 200 \
// --start-step 101  --end-step 400  \
// --mixed-mode-start 101  --mixed-size 3  --simulate-crash

// ./pgvector_test 1 \
// --dataset /ssd_root/dataset/turing10m/msturing-10M.fvecs \
// --queries /ssd_root/dataset/turing10m/msturing-query.fvecs \
// --runbook msturing-10M_slidingwindow_runbook.yaml \
// --dataset-name msturing-10M \
// --checkpoint-size 1000 \
// --gt-dir /ssd_root/liu4127/msturing_runbook_gt \
// --db-host localhost \
// --db-port 5434 \
// --db-name postgres \
// --db-user liu4127 \
// --db-password "" \
// --hnsw-m 16 \
// --hnsw-ef-construction 40 \
// --hnsw-ef-search 200 \
// --start-step 101  --end-step 101


// ./pgvector_test 1 \
// --dataset /ssd_root/dataset/turing10m/msturing-10M.fvecs \
// --queries /ssd_root/dataset/turing10m/msturing-query.fvecs \
// --runbook msturing-10M_slidingwindow_runbook.yaml \
// --dataset-name msturing-10M \
// --checkpoint-size 1000 \
// --gt-dir /ssd_root/liu4127/msturing_runbook_gt \
// --db-host localhost \
// --db-port 5434 \
// --db-name postgres \
// --db-user liu4127 \
// --db-password "" \
// --hnsw-m 16 \
// --hnsw-ef-construction 40 \
// --hnsw-ef-search 200 \
// --start-step 101  --end-step 101  --build-index-before 101
//
// IVFFlat example:
// ./pgvector_test 1 \
// --dataset <path> --queries <path> --runbook <path> --dataset-name <name> \
// --index-type ivfflat --ivfflat-lists 100 --ivfflat-probes 10 \
// --checkpoint-size 1000 --gt-dir ./ground_truth \
// --db-host localhost --db-port 5432 --db-name vector_benchmark --db-user postgres