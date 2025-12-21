import struct
import numpy as np
import psycopg2
import argparse
import time
import sys
import threading
from queue import Queue
from collections import defaultdict

from psycopg2 import sql


def read_bvecs(file_path, num_vectors=None):
    with open(file_path, "rb") as f:
        vectors = []
        while True:
            len_bytes = f.read(4)
            if not len_bytes:
                break
            dim = struct.unpack("i", len_bytes)[0]
            # Read byte-type values, each dimension is 1 byte
            vec = np.frombuffer(f.read(dim), dtype=np.uint8)
            vectors.append(vec)
            if num_vectors and len(vectors) >= num_vectors:
                break
    return np.vstack(vectors)

def read_fvecs(file_path, num_vectors=None):
    with open(file_path, "rb") as f:
        vectors = []
        while True:
            len_bytes = f.read(4)
            if not len_bytes:
                break
            dim = struct.unpack("i", len_bytes)[0]
            vec = np.frombuffer(f.read(4 * dim), dtype=np.float32)
            vectors.append(vec)
            if num_vectors and len(vectors) >= num_vectors:
                break
    return np.vstack(vectors)

def read_ivecs(file_path, expected_k=10000):
    """
    Reads an .ivecs ground truth file where each entry is prefixed by an int32
    indicating the number of neighbors (should be `expected_k`, usually 1000).
    """
    vecs = []
    with open(file_path, "rb") as f:
        query_index = 0
        while True:
            len_bytes = f.read(4)
            if not len_bytes:
                break  # EOF
            dim = struct.unpack("i", len_bytes)[0]
            if dim != expected_k:
                raise ValueError(f"Expected {expected_k} neighbors but got {dim} at query {query_index}")
            vec_data = f.read(4 * dim)
            vec = struct.unpack(f"{dim}i", vec_data)
            vecs.append(vec)
            query_index += 1
    return vecs

def read_fbin(file_path, num_vectors=None):
    with open(file_path, "rb") as f:
        # Read number of vectors and dimension
        header = np.fromfile(f, dtype=np.int32, count=2)
        if len(header) < 2:
            raise ValueError("Invalid fbin file: header too short")
        n, d = header
        print(f"File contains {n} vectors of dimension {d}")
        
        # If num_vectors is specified, limit the read
        count = n if num_vectors is None else min(num_vectors, n)
        
        # Read vectors as float32
        data = np.fromfile(f, dtype=np.float32, count=count * d)
        vectors = data.reshape(count, d)
        
    return vectors

def load_groundtruth_bin_file(filename):
    with open(filename, "rb") as f:
        # Read npts and ndims as 32-bit integers
        npts, ndims = struct.unpack("ii", f.read(8))

        # Read npts * ndims uint32_t (ground truth ids)
        num_ids = npts * ndims
        id_data = f.read(num_ids * 4)  # 4 bytes per uint32
        ids = np.frombuffer(id_data, dtype=np.uint32).reshape(npts, ndims)

        # Read npts * ndims floats (distances)
        dist_data = f.read(npts * ndims * 4)
        dists = np.frombuffer(dist_data, dtype=np.float32).reshape(npts, ndims)

    # Return only the ground truth ids as a list of lists
    return ids.tolist()

def create_table(conn, dim, tablename = "bigann_vectors"):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"DROP TABLE IF EXISTS {tablename};")
        cur.execute(f"""
            CREATE TABLE {tablename} (
                id SERIAL PRIMARY KEY,
                vec vector({dim})
            );
        """)
        conn.commit()

def insert_vectors(conn, vectors, tablename = "bigann_vectors"):
    with conn.cursor() as cur:
        for vec in vectors:
            cur.execute(f"INSERT INTO {tablename} (vec) VALUES (%s);", (vec.tolist(),))
        conn.commit()

def create_index(conn, index_type, index_params, tablename = "bigann_vectors"):
    with conn.cursor() as cur:
        print("Checking for existing index and dropping if it exists...")
        cur.execute("DROP INDEX IF EXISTS bigann_vector_idx;")
        cur.execute("DROP INDEX IF EXISTS bigann_ivf_idx;")
        
        if index_type == "ivfflat":
            nlist = index_params.get("nlist", 100)
            print(f"Creating IVF_FLAT index with nlist = {nlist}...")
            start_time = time.time()
            index_sql = f"""
                CREATE INDEX bigann_vector_idx ON {tablename}
                USING ivfflat (vec vector_l2_ops)
                WITH (lists = %s);
            """
            cur.execute(index_sql, (nlist,))
            elapsed_time = time.time() - start_time
            print(f"IVFFLAT Index creation done in {elapsed_time:.4f} seconds.\n")
        
        elif index_type == "hnsw":
            m = index_params.get("M", 32)
            ef_construction = index_params.get("ef_construction", 64)
            print(f"Creating HNSW index with M = {m}, ef_construction = {ef_construction}...")
            start_time = time.time()
            index_sql = f"""
                CREATE INDEX bigann_vector_idx ON {tablename}
                USING hnsw (vec vector_l2_ops)
                WITH (M = %s, ef_construction = %s);
            """
            cur.execute(index_sql, (m, ef_construction))
            elapsed_time = time.time() - start_time
            print(f"HNSW Index creation done in {elapsed_time:.4f} seconds.\n")
        
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        conn.commit()
        print("Index creation complete.\n")


def search_queries(conn, queries, top_k=10, nprobe=20, ef_search=400, tablename = "bigann_vectors"):
    """Sequential search queries on a single connection (for compatibility)"""
    with conn.cursor() as cur:
        cur.execute(f"SET ivfflat.probes = {nprobe};")
        cur.execute(f"SET hnsw.ef_search = {ef_search};")
        print(f"SET hnsw.ef_search = {ef_search}, ivfflat.probes = {nprobe}")
        total_time = 0
        all_results = []
        for q in queries:
            start = time.time()
            cur.execute(f"""
                SELECT id
                FROM {tablename}
                ORDER BY vec <-> %s::vector
                LIMIT %s;
            """, (q.tolist(), top_k))
            # Subtract 1 from each ID to get zero-based indexing
            ids = [row[0] - 1 for row in cur.fetchall()]
            elapsed = time.time() - start
            total_time += elapsed
            all_results.append(ids)
        print(f"\nAverage search time: {total_time / len(queries):.4f} seconds")
        return all_results


def search_queries_concurrent(conn_params, queries, top_k=10, nprobe=20, ef_search=400, 
                              tablename="bigann_vectors", num_connections=4, all_queries_per_thread=False):
    """
    Execute queries concurrently using multiple connections.
    
    Args:
        conn_params: Dictionary with connection parameters (host, port, user, password, dbname)
        queries: List of query vectors
        top_k: Number of results to return per query
        nprobe: Number of probes for IVF_FLAT
        ef_search: ef_search parameter for HNSW
        tablename: Name of the table to query
        num_connections: Number of concurrent connections to use
        all_queries_per_thread: If True, each thread executes all queries. If False, queries are distributed across threads.
    
    Returns:
        List of results, where each result is a list of IDs for that query
        (When all_queries_per_thread=True, returns results from the first thread only)
    """
    if all_queries_per_thread:
        print(f"Running concurrent search with {num_connections} connections (each thread executes all {len(queries)} queries)")
    else:
        print(f"Running concurrent search with {num_connections} connections (queries distributed across threads)")
    print(f"SET hnsw.ef_search = {ef_search}, ivfflat.probes = {nprobe}")
    
    # Results dictionary: query_index -> list of IDs
    results = {}
    results_lock = threading.Lock()
    
    # Timing statistics
    timing_stats = {
        'total_time': 0.0,
        'query_count': 0,
        'lock': threading.Lock()
    }
    
    if all_queries_per_thread:
        # Mode: Each thread executes all queries
        def worker_thread_all_queries(thread_id):
            """Worker thread that executes all queries"""
            # Create a connection for this thread
            conn = psycopg2.connect(
                host=conn_params['host'],
                port=conn_params['port'],
                user=conn_params['user'],
                password=conn_params['password'],
                dbname=conn_params['dbname']
            )
            
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SET ivfflat.probes = {nprobe};")
                    cur.execute(f"SET hnsw.ef_search = {ef_search};")
                    
                    thread_total_time = 0.0
                    thread_query_count = 0
                    thread_results = []
                    
                    # Execute all queries
                    for query_index, query_vec in enumerate(queries):
                        start = time.time()
                        cur.execute(f"""
                            SELECT id
                            FROM {tablename}
                            ORDER BY vec <-> %s::vector
                            LIMIT %s;
                        """, (query_vec.tolist(), top_k))
                        
                        # Subtract 1 from each ID to get zero-based indexing
                        ids = [row[0] - 1 for row in cur.fetchall()]
                        elapsed = time.time() - start
                        
                        thread_results.append(ids)
                        thread_total_time += elapsed
                        thread_query_count += 1
                    
                    # Store results from first thread only (for recall calculation)
                    if thread_id == 0:
                        with results_lock:
                            for i, ids in enumerate(thread_results):
                                results[i] = ids
                    
                    # Update global timing stats
                    with timing_stats['lock']:
                        timing_stats['total_time'] += thread_total_time
                        timing_stats['query_count'] += thread_query_count
                    
            finally:
                conn.close()
        
        worker_func = worker_thread_all_queries
    else:
        # Mode: Distribute queries across threads using a queue
        # Create a queue of query indices
        query_queue = Queue()
        for i, query in enumerate(queries):
            query_queue.put((i, query))
        
        def worker_thread_distributed(thread_id):
            """Worker thread that processes queries from the queue"""
            # Create a connection for this thread
            conn = psycopg2.connect(
                host=conn_params['host'],
                port=conn_params['port'],
                user=conn_params['user'],
                password=conn_params['password'],
                dbname=conn_params['dbname']
            )
            
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SET ivfflat.probes = {nprobe};")
                    cur.execute(f"SET hnsw.ef_search = {ef_search};")
                    
                    thread_total_time = 0.0
                    thread_query_count = 0
                    
                    while True:
                        try:
                            # Get next query from queue (timeout to allow checking if done)
                            query_index, query_vec = query_queue.get(timeout=1)
                        except:
                            # Queue is empty, we're done
                            break
                        
                        # Execute the query
                        start = time.time()
                        cur.execute(f"""
                            SELECT id
                            FROM {tablename}
                            ORDER BY vec <-> %s::vector
                            LIMIT %s;
                        """, (query_vec.tolist(), top_k))
                        
                        # Subtract 1 from each ID to get zero-based indexing
                        ids = [row[0] - 1 for row in cur.fetchall()]
                        elapsed = time.time() - start
                        
                        # Store results
                        with results_lock:
                            results[query_index] = ids
                        
                        thread_total_time += elapsed
                        thread_query_count += 1
                        
                        query_queue.task_done()
                    
                    # Update global timing stats
                    with timing_stats['lock']:
                        timing_stats['total_time'] += thread_total_time
                        timing_stats['query_count'] += thread_query_count
                    
            finally:
                conn.close()
        
        worker_func = worker_thread_distributed
    
    # Start worker threads
    start_time = time.time()
    threads = []
    for i in range(num_connections):
        thread = threading.Thread(target=worker_func, args=(i,))
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    total_elapsed = time.time() - start_time
    
    # Reconstruct results in the correct order
    all_results = [results[i] for i in range(len(queries))]
    
    total_queries_executed = timing_stats['query_count']
    print(f"\nTotal wall-clock time: {total_elapsed:.4f} seconds")
    print(f"Total queries executed: {total_queries_executed}")
    print(f"Total query execution time: {timing_stats['total_time']:.4f} seconds")
    if all_queries_per_thread:
        print(f"Average search time per query (per thread): {timing_stats['total_time'] / total_queries_executed:.4f} seconds")
        print(f"Throughput: {total_queries_executed / total_elapsed:.2f} queries/second")
    else:
        print(f"Average search time per query: {timing_stats['total_time'] / len(queries):.4f} seconds")
        print(f"Throughput: {len(queries) / total_elapsed:.2f} queries/second")
    
    return all_results

    
def compute_recall(predicted_ids, ground_truth_ids, top_k=10):
    total_correct = 0
    for i in range(len(predicted_ids)):
        pred = predicted_ids[i][:top_k]
        if i < len(ground_truth_ids):
            gt = set(ground_truth_ids[i][:top_k])
        else:
            gt = set()

        correct = sum(1 for pid in pred if pid in gt)
        total_correct += correct

        # if i < 5:
        #     print(f"Query {i}:")
        #     print(f"  Predicted IDs:     {pred}")
        #     print(f"  Ground Truth Top-{top_k}: {list(gt)}")
        #     print(f"  Correct matches:   {correct}")

    recall = total_correct / (len(predicted_ids) * top_k)
    print(f"\nRecall@{top_k}: {recall:.4f}")

# TODO: for evaluation
def get_vector_search_timing_stats(conn):
    """Query vector search timing statistics from PostgreSQL"""
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM pg_vector_search_timing_stats();")
        result = cur.fetchone()
        if result:
            call_count, total_time_us, average_time_us = result
            print(f"\n=== Vector Search Timing Statistics ===")
            print(f"Total calls: {call_count}")
            print(f"Total time: {total_time_us:.2f} μs")
            print(f"Average time per call: {average_time_us:.2f} μs")
            return result
        else:
            print("No timing statistics available yet")
            return None
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do-insert", action="store_true", help="Insert base vectors into table")
    parser.add_argument("--do-index", action="store_true", help="Create IVF_FLAT index")
    parser.add_argument("--do-query", action="store_true", help="Run similarity search queries")
    parser.add_argument("--file", help="Path to .bvecs file for insert")
    parser.add_argument("--queries", help="Path to query .bvecs file for search")
    parser.add_argument("--gnd", help="Path to groundtruth .ivecs file for calculating recall")
    parser.add_argument("--num", type=int, default=1000, help="Number of base vectors to insert")
    parser.add_argument("--nq", type=int, default=10, help="Number of queries to run")
    parser.add_argument("--k", type=int, default=100, help="topk")
    parser.add_argument("--index-type", choices=["ivfflat", "hnsw"], default="ivfflat", help="Type of vector index to use")
    parser.add_argument("--M", type=int, default=16, help="HNSW: number of neighbors per node")
    parser.add_argument("--ef-construction", type=int, default=64, help="HNSW: size of dynamic list for graph construction")
    parser.add_argument("--nlist", type=int, default=1000, help="Number of lists for IVF_FLAT index")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=5432, type=int)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--dbname", default="postgres")
    parser.add_argument("--tablename", default="test_table")
    parser.add_argument("--num-connections", type=int, default=4, help="Number of concurrent connections for queries")
    parser.add_argument("--sequential", action="store_true", help="Use sequential query execution instead of concurrent")
    parser.add_argument("--all-queries-per-thread", action="store_true", help="Each thread executes all queries (for stress testing)")
    args = parser.parse_args()

    if not (args.do_insert or args.do_index or args.do_query):
        print("Error: Please specify at least one of --do-insert, --do-index, or --do-query")
        sys.exit(1)

    conn_params = {
        'host': args.host,
        'port': args.port,
        'user': args.user,
        'password': args.password,
        'dbname': args.dbname
    }

    conn = psycopg2.connect(**conn_params)

    if args.do_insert:
        if not args.file:
            print("Error: --file is required for --do-insert")
            sys.exit(1)
        print(f"Reading base vectors from: {args.file}")
        base_vectors = read_fvecs(args.file, args.num)
        print(f"Loaded {len(base_vectors)} base vectors of dimension {base_vectors.shape[1]}")
        print("Creating table...")
        create_table(conn, base_vectors.shape[1], args.tablename)
        print("Inserting vectors...")
        insert_vectors(conn, base_vectors, args.tablename)
        print("Insertion done.\n")

    if args.do_index:
        index_params = {
            "nlist": args.nlist,
            "M": args.M,
            "ef_construction": args.ef_construction,
        }
        print(f"Creating {args.index_type.upper()} index...")
        create_index(conn, args.index_type, index_params=index_params, tablename=args.tablename)

    
    if args.do_query:
        if not args.gnd:
            print("Error: --gnd is required for --do-query")
            sys.exit(1)
        print(f"Reading ground truth from: {args.gnd}")
        
        ground_truth = read_ivecs(args.gnd, 1000)
        if not args.queries:
            print("Error: --queries is required for --do-query")
            sys.exit(1)
        print(f"Reading query vectors from: {args.queries}")
        queries = read_bvecs(args.queries, args.nq)
        print(f"Loaded {len(queries)} queries of dimension {queries.shape[1]}")
        
        # Use concurrent or sequential search based on flag
        if args.sequential:
            print("Using sequential query execution")
            def search_func(conn, queries, top_k, nprobe, ef_search, tablename):
                return search_queries(conn, queries, top_k, nprobe, ef_search, tablename)
        else:
            if args.all_queries_per_thread:
                print(f"Using concurrent query execution with {args.num_connections} connections (each thread executes all queries)")
            else:
                print(f"Using concurrent query execution with {args.num_connections} connections (queries distributed)")
            def search_func(conn, queries, top_k, nprobe, ef_search, tablename):
                return search_queries_concurrent(conn_params, queries, top_k, nprobe, ef_search, tablename, args.num_connections, args.all_queries_per_thread)
        
        print("Running similarity search (r1)...")
        predicted = search_func(conn, queries, args.k, 2, 100, args.tablename)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r2)...")
        predicted = search_func(conn, queries, args.k, 3, 100, args.tablename)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r3)...")
        predicted = search_func(conn, queries, args.k, 5, 100, args.tablename)
        compute_recall(predicted, ground_truth, args.k)
        
        # print("Running similarity search (r4)...")
        # predicted = search_func(conn, queries, args.k, 10, 300, args.tablename)
        # compute_recall(predicted, ground_truth, args.k)
        
        # print("Running similarity search (r5)...")
        # predicted = search_func(conn, queries, args.k, 3, 100, args.tablename)
        # compute_recall(predicted, ground_truth, args.k)
        
        # print("Running similarity search (r6)...")
        # predicted = search_func(conn, queries, args.k, 5, 200, args.tablename)
        # compute_recall(predicted, ground_truth, args.k)
        
        # print("Running similarity search (r7)...")
        # predicted = search_func(conn, queries, args.k, 10, 300, args.tablename)
        # compute_recall(predicted, ground_truth, args.k)
        
        # print("Running similarity search (r8)...")
        # predicted = search_func(conn, queries, args.k, 20, 400, args.tablename)
        # compute_recall(predicted, ground_truth, args.k)
        
        # print("Running similarity search (r9)...")
        # predicted = search_func(conn, queries, args.k, 30, 500, args.tablename)
        # compute_recall(predicted, ground_truth, args.k)
        
        # print("Running similarity search (r10)...")
        # predicted = search_func(conn, queries, args.k, 40, 600, args.tablename)
        # compute_recall(predicted, ground_truth, args.k)
        
        # print("Running similarity search (r9)...")
        # predicted = search_func(conn, queries, args.k, 30, 700, args.tablename)
        # compute_recall(predicted, ground_truth, args.k)
        
        # print("Running similarity search (r10)...")
        # predicted = search_func(conn, queries, args.k, 40, 800, args.tablename)
        # compute_recall(predicted, ground_truth, args.k)
        
        # print("Running similarity search (r9)...")
        # predicted = search_func(conn, queries, args.k, 30, 900, args.tablename)
        # compute_recall(predicted, ground_truth, args.k)
        
        # print("Running similarity search (r9)...")
        # predicted = search_func(conn, queries, args.k, 30, 1000, args.tablename)
        # compute_recall(predicted, ground_truth, args.k)
        
    conn.close()

if __name__ == "__main__":
    main()

# hnsw build and query sift1M
# python static_search_concurrency.py --do-index --do-query \
#   --file /ssd_root/dataset/sift/sift_base.fvecs \
#   --queries /ssd_root/dataset/sift/sift_query.fvecs \
#   --gnd /ssd_root/dataset/sift/sift_groundtruth.ivecs \
#   --num 1000000 --nq 10000 --k 100 --num-connections 2 \
#   --index-type hnsw --M 16 --ef-construction 40 \
#   --host localhost --port 543* --user liu4127 --dbname postgres --tablename bigann_vectors

# hnsw build and query sift10M
# python static_search_concurrency.py --do-index --do-query \
#   --queries /ssd_root/dataset/sift/bigann_query.bvecs \
#   --gnd /ssd_root/dataset/sift/gnd/idx_10M.ivecs \
#   --num 1000000 --nq 10000 --k 100 --num-connections 1 \
#   --index-type hnsw --M 16 --ef-construction 40 \
#   --host localhost --port 543* --user liu4127 --dbname postgres --tablename bigann_vectors