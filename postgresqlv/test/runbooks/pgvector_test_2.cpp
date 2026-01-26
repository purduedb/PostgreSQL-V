// pgvector_test_2.cpp
//
// pgvector HNSW sliding window test script - multi-threaded version with QPS control
// This script runs the runbook with multiple threads in mixed mode, allowing
// the user to set the QPS (queries per second) rate per thread for sending requests.
//
// Compilation command:
//   g++ -O3 -std=c++17 -mavx2 -mfma -fopenmp pgvector_test_2.cpp \
//       -o pgvector_test_2 \
//       -I/usr/include/postgresql -lpq -lyaml-cpp -lcrypto -pthread
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <random>
#include <libpq-fe.h>
#include <yaml-cpp/yaml.h>
#include <openssl/md5.h>
#include <thread>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdlib>
#include <cerrno>
#include <omp.h>
#include <mutex>
#include <atomic>
#include <limits>
#include <memory>

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
    
    // pgvector HNSW index parameters
    int HNSW_M = 48;
    int HNSW_EF_CONSTRUCTION = 400;
    int HNSW_EF_SEARCH = 200;
    
    // Dataset paths
    std::string DATASET_PATH;
    std::string QUERY_PATH;
    std::string RUNBOOK_PATH;
    std::string DATASET_NAME;
    std::string GT_DIR;
    
    // Dimension will be auto-detected from dataset
    size_t DIMENSION = 0;
    
    // QPS rate limiting
    double TARGET_QPS = 0.0;  // 0 means no rate limiting
};

// ===================================================
// Rate limiter for QPS control
// ===================================================
class RateLimiter {
private:
    double qps;
    double min_interval;  // Minimum time between operations (seconds)
    std::chrono::high_resolution_clock::time_point last_operation_time;
    std::mt19937 gen;
    std::uniform_real_distribution<double> jitter_dist;

public:
    RateLimiter(double target_qps) : qps(target_qps) {
        if (qps > 0) {
            min_interval = 1.0 / qps;
        } else {
            min_interval = 0.0;
        }
        last_operation_time = std::chrono::high_resolution_clock::now();
        std::random_device rd;
        gen = std::mt19937(rd());
        // Small jitter to avoid thundering herd (0-5% of interval)
        jitter_dist = std::uniform_real_distribution<double>(0.0, min_interval * 0.05);
    }

    void wait_if_needed() {
        if (qps <= 0) {
            return;  // No rate limiting
        }

        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(now - last_operation_time).count();
        
        double wait_time = min_interval - elapsed;
        if (wait_time > 0) {
            // Add small random jitter to avoid synchronization
            double jitter = jitter_dist(gen);
            std::this_thread::sleep_for(std::chrono::duration<double>(wait_time + jitter));
        }
        
        last_operation_time = std::chrono::high_resolution_clock::now();
    }
};

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
// pgvector sliding window test class (multi-threaded with QPS control)
// =========================================================
class PGVectorSlidingWindowTest {
private:
    std::set<size_t> active_indices;
    size_t dim;
    std::string table_name;
    std::string gt_dir;
    std::vector<std::tuple<std::string, size_t, size_t>> active_ranges;
    int hnsw_m;
    int hnsw_ef_construction;
    int hnsw_ef_search;
    double target_qps;
    int num_threads;
    
    // Crash simulation
    bool simulate_crash = false;
    size_t crash_target_step = 0;
    size_t crash_target_operation = 0;
    std::atomic<size_t> current_operation_count{0};
    std::string db_host;
    std::string db_port;
    
    // Skip recall computation
    bool skip_recall_computation = false;
    
    // Stop flag - set to true when crash happens or first failure occurs
    std::atomic<bool> should_stop{false};
    
    // Mutex for statistics updates
    std::mutex stats_mutex;
    
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
            double actual_qps = 0.0;
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
        };

        double total_time = 0.0;
        std::vector<OperationStats> operations;
        CategoryStats insert_stats;
        CategoryStats delete_stats;
        CategoryStats search_stats;
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
        
        // Method 2: Fallback - try to find postgres process listening on the port
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

        if (op_stats.type == "search" && op_stats.recall > 0) {
            cat_stats.recall_values.push_back(op_stats.recall);
            cat_stats.min_recall = std::min(cat_stats.min_recall, op_stats.recall);
            cat_stats.max_recall = std::max(cat_stats.max_recall, op_stats.recall);
        }
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

public:
    PGVectorSlidingWindowTest(size_t dimension,
                               const std::string& gt_directory,
                               const std::string& db_host,
                               const std::string& db_port,
                               const std::string& db_name,
                               const std::string& db_user,
                               const std::string& db_password,
                               int hnsw_m,
                               int hnsw_ef_construction,
                               int hnsw_ef_search,
                               double qps,
                               int threads,
                               bool enable_crash_simulation = false,
                               bool skip_recall = false)
        : dim(dimension), table_name("vectors"), gt_dir(gt_directory),
          hnsw_m(hnsw_m), hnsw_ef_construction(hnsw_ef_construction), 
          hnsw_ef_search(hnsw_ef_search), target_qps(qps), num_threads(threads),
          simulate_crash(enable_crash_simulation), db_host(db_host), 
          db_port(db_port), skip_recall_computation(skip_recall) {
        
        // Initialize thread-local connection manager
        ThreadLocalConnection::init(db_host, db_port, db_name, db_user, db_password);

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
                       << "embedding vector(" << dim << ")"
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

    // Create HNSW index
    void create_index() {
        Timer timer;
        PGconn* conn = ThreadLocalConnection::get_main();
        if (!conn) {
            throw std::runtime_error("No database connection available");
        }

        std::cout << "Creating HNSW index..." << std::endl;

        PGresult* res = PQexec(conn, "SET maintenance_work_mem = '8GB'");
        PQclear(res);

        std::stringstream index_sql;
        index_sql << "CREATE INDEX ON " << table_name 
                  << " USING hnsw (embedding vector_l2_ops) WITH ("
                  << "m = " << hnsw_m << ", "
                  << "ef_construction = " << hnsw_ef_construction
                  << ")";
        
        res = PQexec(conn, index_sql.str().c_str());
        if (PQresultStatus(res) != PGRES_COMMAND_OK) {
            throw std::runtime_error("Failed to create index: " + std::string(PQerrorMessage(conn)));
        }
        PQclear(res);

        double elapsed = timer.elapsed_seconds();
        std::cout << "✔ HNSW index created in " << elapsed << "s" << std::endl;
    }

    // Execute runbook in mixed mode with QPS control
    void execute_runbook(const std::string& runbook_file, const std::string& dataset_name,
                         float* dataset, float* queries, size_t num_queries,
                         size_t start_step = 0, size_t end_step = 0, 
                         size_t build_index_before = 0,
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
        if (start_step > 0) {
            size_t range_step_num = 0;
            for (auto it = operations.begin(); it != operations.end(); ++it) {
                std::string step_key = it->first.as<std::string>();
                if (step_key == "max_pts" || step_key == "query" || step_key == "groundtruth") {
                    continue;
                }
                range_step_num++;
                if (range_step_num >= start_step) {
                    break;
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

        // Collect all steps
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

        // Initialize rate limiter
        RateLimiter rate_limiter(target_qps);

        // Execute operations - mixed mode only
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
                        i = j;
                    } else {
                        break;
                    }
                }

                if (!mixed_steps.empty()) {
                    std::cout << "\n========================================" << std::endl;
                    std::cout << "MIXED MODE: Processing " << mixed_steps.size() 
                              << " steps with " << num_threads << " threads, QPS=" << target_qps << " per thread" << std::endl;
                    std::cout << "========================================" << std::endl;

                    Timer mixed_timer;
                    
                    // Structure to hold step information
                    struct StepInfo {
                        std::string op;
                        size_t start;
                        size_t end;
                        size_t total_items;
                        size_t step_num;
                        std::string step_key;
                        size_t k;
                    };
                    
                    std::vector<StepInfo> step_infos;
                    // Use atomic counters for thread-safe operation tracking
                    std::vector<std::unique_ptr<std::atomic<size_t>>> step_next_items;
                    std::vector<size_t> step_successful_counts(mixed_steps.size(), 0);
                    std::vector<size_t> step_failed_counts(mixed_steps.size(), 0);
                    std::mutex counters_mutex;  // For protecting counter updates
                    
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
                            continue;
                        }
                        
                        step_infos.push_back({op, start, end, total_items, step_num, step_key, k});
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
                    
                    // Generate array: each element is a step index
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
                    
                    // Process the step_index_array in parallel with rate limiting per thread
                    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 100)
                    for (size_t array_idx = 0; array_idx < step_index_array.size(); array_idx++) {
                        // Thread-local rate limiter (each thread maintains its own QPS)
                        thread_local RateLimiter* rate_limiter = nullptr;
                        if (rate_limiter == nullptr) {
                            rate_limiter = new RateLimiter(target_qps);
                        }
                        
                        // Get thread-local connection
                        PGconn* conn = ThreadLocalConnection::get();
                        if (!conn) {
                            continue;
                        }
                        
                        size_t step_idx = step_index_array[array_idx];
                        StepInfo& step_info = step_infos[step_idx];
                        
                        // Atomically get the next operation index within this step
                        size_t item_idx = step_next_items[step_idx]->fetch_add(1, std::memory_order_relaxed);
                        
                        // Safety check
                        if (item_idx >= step_info.total_items) {
                            continue;
                        }
                        
                        // Check if we should stop (check atomic flag)
                        if (should_stop.load(std::memory_order_relaxed)) {
                            continue;
                        }
                        
                        // Check for crash simulation
                        if (should_crash(step_info.step_num)) {
                            #pragma omp critical
                            {
                                if (simulate_crash) {
                                    kill_postgres_master();
                                    simulate_crash = false;  // Prevent multiple kills
                                    should_stop.store(true, std::memory_order_relaxed);
                                    std::cout << "\n[CRASH SIMULATION] Stopping execution after crash" << std::endl;
                                }
                            }
                            continue;
                        }
                        
                        // Check if we should stop after crash check
                        if (should_stop.load(std::memory_order_relaxed)) {
                            continue;
                        }
                        
                        // Rate limiting: wait if needed to maintain QPS (per thread)
                        rate_limiter->wait_if_needed();
                        
                        // Check connection status before operation
                        if (PQstatus(conn) != CONNECTION_OK) {
                            std::cerr << "\n[ERROR] Database connection lost. Stopping execution." << std::endl;
                            std::cerr << "[ERROR] Connection error: " << PQerrorMessage(conn) << std::endl;
                            should_stop.store(true, std::memory_order_relaxed);
                            continue;
                        }
                        
                        // Execute the operation
                        bool operation_failed = false;
                        if (step_info.op == "insert") {
                            size_t vec_idx = step_info.start + item_idx;
                            std::stringstream sql;
                            sql << "INSERT INTO " << table_name << " (id, embedding) VALUES ("
                                << vec_idx << "," << vector_to_sql(dataset + vec_idx * dim, dim)
                                << ") ON CONFLICT (id) DO NOTHING";
                            PGresult* res = PQexec(conn, sql.str().c_str());
                            if (PQresultStatus(res) == PGRES_COMMAND_OK) {
                                std::lock_guard<std::mutex> lock(counters_mutex);
                                step_successful_counts[step_idx]++;
                            } else {
                                std::lock_guard<std::mutex> lock(counters_mutex);
                                step_failed_counts[step_idx]++;
                                operation_failed = true;
                                std::cerr << "\n[ERROR] Insert operation failed: " << PQerrorMessage(conn) << std::endl;
                            }
                            PQclear(res);
                        }
                        else if (step_info.op == "delete") {
                            size_t vec_idx = step_info.start + item_idx;
                            std::stringstream sql;
                            sql << "DELETE FROM " << table_name << " WHERE id = " << vec_idx;
                            PGresult* res = PQexec(conn, sql.str().c_str());
                            if (PQresultStatus(res) == PGRES_COMMAND_OK) {
                                char* tuples = PQcmdTuples(res);
                                std::lock_guard<std::mutex> lock(counters_mutex);
                                if (tuples && std::atoi(tuples) > 0) {
                                    step_successful_counts[step_idx]++;
                                } else {
                                    step_failed_counts[step_idx]++;
                                }
                            } else {
                                std::lock_guard<std::mutex> lock(counters_mutex);
                                step_failed_counts[step_idx]++;
                                operation_failed = true;
                                std::cerr << "\n[ERROR] Delete operation failed: " << PQerrorMessage(conn) << std::endl;
                            }
                            PQclear(res);
                        }
                        else if (step_info.op == "search") {
                            if (queries != nullptr) {
                                // Set search parameters
                                std::stringstream set_sql;
                                set_sql << "SET hnsw.ef_search = " << hnsw_ef_search;
                                PGresult* set_res = PQexec(conn, set_sql.str().c_str());
                                PQclear(set_res);
                                
                                // Build search SQL
                                std::stringstream sql;
                                sql << "SELECT id FROM " << table_name
                                    << " ORDER BY embedding <-> " << vector_to_sql(queries + item_idx * dim, dim)
                                    << " LIMIT " << step_info.k;
                                PGresult* res = PQexec(conn, sql.str().c_str());
                                if (PQresultStatus(res) == PGRES_TUPLES_OK) {
                                    std::lock_guard<std::mutex> lock(counters_mutex);
                                    step_successful_counts[step_idx]++;
                                } else {
                                    std::lock_guard<std::mutex> lock(counters_mutex);
                                    step_failed_counts[step_idx]++;
                                    operation_failed = true;
                                    std::cerr << "\n[ERROR] Search operation failed: " << PQerrorMessage(conn) << std::endl;
                                }
                                PQclear(res);
                            }
                        }
                        
                        // If operation failed, set stop flag
                        if (operation_failed) {
                            should_stop.store(true, std::memory_order_relaxed);
                            std::cerr << "[ERROR] Stopping execution after first failure." << std::endl;
                            continue;
                        }
                        
                        // Check connection status after operation
                        if (PQstatus(conn) != CONNECTION_OK) {
                            std::cerr << "\n[ERROR] Database connection lost during operation. Stopping execution." << std::endl;
                            std::cerr << "[ERROR] Connection error: " << PQerrorMessage(conn) << std::endl;
                            should_stop.store(true, std::memory_order_relaxed);
                            continue;
                        }
                    }
                    
                    // Check if we stopped early
                    if (should_stop.load(std::memory_order_relaxed)) {
                        std::cout << "\n[WARNING] Execution stopped early due to crash or failure." << std::endl;
                        std::cout << "[WARNING] Statistics below reflect partial execution." << std::endl;
                    }
                    
                    // Final update of active_indices and active_ranges
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
                    
                    double mixed_elapsed = mixed_timer.elapsed_seconds();
                    
                    // Record statistics for each step
                    for (size_t step_idx = 0; step_idx < step_infos.size(); step_idx++) {
                        const auto& step_info = step_infos[step_idx];
                        size_t successful = step_successful_counts[step_idx];
                        size_t failed = step_failed_counts[step_idx];
                        double throughput = successful / mixed_elapsed;
                        double actual_qps = (target_qps > 0) ? target_qps : throughput;
                        
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
                        op_stats.actual_qps = actual_qps;
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
                std::cout << "\n========================================" << std::endl;
                std::cout << "Step " << step_num << " - " << step_info.step_key << ": " << step_info.op << std::endl;
                std::cout << "========================================" << std::endl;
                std::cout << "Note: This script only supports mixed mode. Skipping sequential step." << std::endl;
            }
        }

        stats.total_time = overall_timer.elapsed_seconds();
        print_summary();
    }

    void print_summary() {
        std::cout << "\n╔══════════════════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║              PGVECTOR SLIDING WINDOW TEST SUMMARY (Multi-threaded)           ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝" << std::endl;

        std::cout << "\n┌───────────────────────── OVERALL STATISTICS ────────────────────────────────┐" << std::endl;
        std::cout << "│ Total Test Time: " << std::fixed << std::setprecision(2) 
                  << stats.total_time << " seconds" << std::endl;
        std::cout << "│ Total Operations: " << stats.operations.size() << std::endl;
        std::cout << "│   - Inserts: " << stats.insert_stats.count << std::endl;
        std::cout << "│   - Deletes: " << stats.delete_stats.count << std::endl;
        std::cout << "│   - Searches: " << stats.search_stats.count << std::endl;
        std::cout << "│ Threads: " << num_threads << std::endl;
        std::cout << "│ Target QPS per thread: " << (target_qps > 0 ? std::to_string(target_qps) : "unlimited") << std::endl;
        std::cout << "│ Total Target QPS: " << (target_qps > 0 ? std::to_string(target_qps * num_threads) : "unlimited") << std::endl;
        std::cout << "└──────────────────────────────────────────────────────────────────────────────┘" << std::endl;

        std::cout << "\n┌───────────────────────── DETAILED OPERATION LOG ────────────────────────────┐" << std::endl;
        std::cout << "│ " << std::left 
                  << std::setw(12) << "Step" 
                  << std::setw(8) << "Type"
                  << std::setw(10) << "Time(s)"
                  << std::setw(10) << "Success"
                  << std::setw(12) << "Actual QPS"
                  << std::setw(10) << "Recall" << " │" << std::endl;
        std::cout << "│ " << std::string(68, '-') << " │" << std::endl;
        
        for (const auto& op : stats.operations) {
            std::cout << "│ " << std::left 
                      << std::setw(12) << op.name.substr(0, 11)
                      << std::setw(8) << op.type.substr(0, 7)
                      << std::setw(10) << std::fixed << std::setprecision(2) << op.time
                      << std::setw(10) << op.successful
                      << std::setw(12) << std::fixed << std::setprecision(0) << op.actual_qps;
            
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
    std::cerr << "Usage: " << prog_name << " [options]" << std::endl;
    std::cerr << "\nRequired arguments:" << std::endl;
    std::cerr << "  --num-threads <n>            Number of threads for parallel operations" << std::endl;
    std::cerr << "  --qps <qps>                  Target QPS (queries per second) per thread for rate limiting" << std::endl;
    std::cerr << "  --dataset <path>            Path to dataset .fvecs file" << std::endl;
    std::cerr << "  --queries <path>             Path to queries .fvecs file" << std::endl;
    std::cerr << "  --runbook <path>            Path to runbook YAML file" << std::endl;
    std::cerr << "  --dataset-name <name>       Dataset name in runbook" << std::endl;
    std::cerr << "  --mixed-mode-start <num>    Start step for mixed mode (1-based)" << std::endl;
    std::cerr << "  --mixed-size <size>         Number of steps per group in mixed mode" << std::endl;
    std::cerr << "\nOptional arguments:" << std::endl;
    std::cerr << "  --gt-dir <path>             Ground truth directory (default: ./ground_truth)" << std::endl;
    std::cerr << "  --start-step <num>          First step to process (1-based, default: process all)" << std::endl;
    std::cerr << "  --end-step <num>            Last step to process (1-based, default: process all)" << std::endl;
    std::cerr << "  --build-index-before <num>  Create index before this step (1-based, default: auto)" << std::endl;
    std::cerr << "  --db-host <host>            PostgreSQL host (default: localhost)" << std::endl;
    std::cerr << "  --db-port <port>            PostgreSQL port (default: 5432)" << std::endl;
    std::cerr << "  --db-name <name>            PostgreSQL database name (default: vector_benchmark)" << std::endl;
    std::cerr << "  --db-user <user>            PostgreSQL user (default: postgres)" << std::endl;
    std::cerr << "  --db-password <password>    PostgreSQL password (default: empty)" << std::endl;
    std::cerr << "  --hnsw-m <m>                HNSW M parameter (default: 48)" << std::endl;
    std::cerr << "  --hnsw-ef-construction <ef> HNSW ef_construction parameter (default: 400)" << std::endl;
    std::cerr << "  --hnsw-ef-search <ef>        HNSW ef_search parameter (default: 200)" << std::endl;
    std::cerr << "  --simulate-crash            Enable crash simulation (kills PostgreSQL at random step/operation)" << std::endl;
    std::cerr << "  --skip-recall               Skip recall computation for search operations (default: false)" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    Config config;
    int num_threads = 1;
    double target_qps = 0.0;
    size_t start_step = 0;
    size_t end_step = 0;
    size_t build_index_before = 0;
    size_t mixed_mode_start = 0;
    size_t mixed_size = 0;
    bool simulate_crash = false;
    bool skip_recall = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--num-threads" && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
        } else if (arg == "--qps" && i + 1 < argc) {
            target_qps = std::atof(argv[++i]);
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
        } else if (arg == "--hnsw-m" && i + 1 < argc) {
            config.HNSW_M = std::atoi(argv[++i]);
        } else if (arg == "--hnsw-ef-construction" && i + 1 < argc) {
            config.HNSW_EF_CONSTRUCTION = std::atoi(argv[++i]);
        } else if (arg == "--hnsw-ef-search" && i + 1 < argc) {
            config.HNSW_EF_SEARCH = std::atoi(argv[++i]);
        } else if (arg == "--simulate-crash") {
            simulate_crash = true;
        } else if (arg == "--skip-recall") {
            skip_recall = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    if (num_threads <= 0) {
        std::cerr << "Error: --num-threads is required and must be > 0" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    if (target_qps <= 0) {
        std::cerr << "Error: --qps is required and must be > 0" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    if (config.DATASET_PATH.empty() || config.QUERY_PATH.empty() || 
        config.RUNBOOK_PATH.empty() || config.DATASET_NAME.empty()) {
        std::cerr << "Error: --dataset, --queries, --runbook, and --dataset-name are required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    if (mixed_mode_start == 0 || mixed_size == 0) {
        std::cerr << "Error: --mixed-mode-start and --mixed-size are required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    if (config.GT_DIR.empty()) {
        config.GT_DIR = "./ground_truth";
    }

    // Set OpenMP thread count
    omp_set_num_threads(num_threads);

    std::cout << "========================================" << std::endl;
    std::cout << "PGVECTOR HNSW SLIDING WINDOW TEST (C++)" << std::endl;
    std::cout << "Multi-threaded with QPS control" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Threads: " << num_threads << std::endl;
    std::cout << "  Target QPS per thread: " << target_qps << std::endl;
    std::cout << "  Total target QPS: " << (target_qps * num_threads) << std::endl;
    std::cout << "  Dataset: " << config.DATASET_PATH << std::endl;
    std::cout << "  Queries: " << config.QUERY_PATH << std::endl;
    std::cout << "  Runbook: " << config.RUNBOOK_PATH << std::endl;
    std::cout << "  Dataset name: " << config.DATASET_NAME << std::endl;
    std::cout << "  GT directory: " << config.GT_DIR << std::endl;
    if (start_step > 0) {
        std::cout << "  Start step: " << start_step << std::endl;
    }
    if (end_step > 0) {
        std::cout << "  End step: " << end_step << std::endl;
    }
    if (build_index_before > 0) {
        std::cout << "  Build index before step: " << build_index_before << std::endl;
    }
    std::cout << "  Mixed mode: enabled (start at step " << mixed_mode_start 
              << ", mixed size: " << mixed_size << ")" << std::endl;
    std::cout << "  DB Host: " << config.DB_HOST << std::endl;
    std::cout << "  DB Port: " << config.DB_PORT << std::endl;
    std::cout << "  DB Name: " << config.DB_NAME << std::endl;
    std::cout << "  DB User: " << config.DB_USER << std::endl;
    std::cout << "  HNSW M: " << config.HNSW_M << std::endl;
    std::cout << "  HNSW ef_construction: " << config.HNSW_EF_CONSTRUCTION << std::endl;
    std::cout << "  HNSW ef_search: " << config.HNSW_EF_SEARCH << std::endl;
    if (simulate_crash) {
        std::cout << "  Crash simulation: ENABLED (will kill PostgreSQL at random step/operation)" << std::endl;
    }
    if (skip_recall) {
        std::cout << "  Skip recall computation: ENABLED" << std::endl;
    }
    std::cout << "========================================\n" << std::endl;

    try {
        // Load dataset
        std::cout << "Loading dataset..." << std::endl;
        size_t num_vectors, data_dim;
        auto dataset = DataLoader::load_fvecs(config.DATASET_PATH, num_vectors, data_dim);
        std::cout << "✔ Dataset loaded: " << num_vectors << " vectors, dim=" << data_dim << std::endl;

        // Load queries
        std::cout << "Loading queries..." << std::endl;
        size_t num_queries, query_dim;
        auto queries = DataLoader::load_fvecs(config.QUERY_PATH, num_queries, query_dim);
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
            config.HNSW_M,
            config.HNSW_EF_CONSTRUCTION,
            config.HNSW_EF_SEARCH,
            target_qps,
            num_threads,
            simulate_crash,
            skip_recall
        );

        // Execute test
        test.execute_runbook(config.RUNBOOK_PATH, config.DATASET_NAME,
                             dataset.data(), queries.data(), num_queries,
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

// ./pgvector_test_2 --num-threads 16 --qps 50 --dataset /ssd_root/dataset/turing10m/msturing-10M.fvecs \
// --queries /ssd_root/dataset/turing10m/msturing-query.fvecs --runbook msturing-10M_slidingwindow_1M_runbook.yaml \
// --dataset-name msturing-10M --gt-dir /ssd_root/liu4127/msturing_runbook_gt \
// --db-host localhost --db-port 5434 --db-name postgres --db-user liu4127 --db-password "" \
// --hnsw-m 16 --hnsw-ef-construction 40 --hnsw-ef-search 200 \
// --start-step 21  --end-step 21 --mixed-mode-start 21  --mixed-size 3 --skip-recall

// ./pgvector_test_2 --num-threads 16 --qps 50 --dataset /ssd_root/dataset/turing10m/msturing-10M.fvecs \
// --queries /ssd_root/dataset/turing10m/msturing-query.fvecs --runbook msturing-10M_slidingwindow_1M_runbook.yaml \
// --dataset-name msturing-10M --gt-dir /ssd_root/liu4127/msturing_runbook_gt \
// --db-host localhost --db-port 5434 --db-name postgres --db-user liu4127 --db-password "" \
// --hnsw-m 16 --hnsw-ef-construction 40 --hnsw-ef-search 200 \
// --start-step 21  --end-step 60  --mixed-mode-start 21  --mixed-size 3  --simulate-crash