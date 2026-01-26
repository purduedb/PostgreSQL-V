import struct
import numpy as np
import psycopg2
import argparse
import time
import sys

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

def read_bvecs_batch(file_handle, batch_size, dim=None):
    """Read a batch of vectors from an open file handle. Returns (vectors, eof_reached)"""
    vectors = []
    for _ in range(batch_size):
        len_bytes = file_handle.read(4)
        if not len_bytes:
            break
        vec_dim = struct.unpack("i", len_bytes)[0]
        if dim is None:
            dim = vec_dim
        elif dim != vec_dim:
            raise ValueError(f"Dimension mismatch: expected {dim}, got {vec_dim}")
        # Read byte-type values, each dimension is 1 byte
        vec = np.frombuffer(file_handle.read(vec_dim), dtype=np.uint8)
        vectors.append(vec)
    
    if len(vectors) == 0:
        return None, True
    return np.vstack(vectors), False

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

def read_fvecs_batch(file_handle, batch_size, dim=None):
    """Read a batch of vectors from an open file handle. Returns (vectors, eof_reached)"""
    vectors = []
    for _ in range(batch_size):
        len_bytes = file_handle.read(4)
        if not len_bytes:
            break
        vec_dim = struct.unpack("i", len_bytes)[0]
        if dim is None:
            dim = vec_dim
        elif dim != vec_dim:
            raise ValueError(f"Dimension mismatch: expected {dim}, got {vec_dim}")
        vec = np.frombuffer(file_handle.read(4 * vec_dim), dtype=np.float32)
        vectors.append(vec)
    
    if len(vectors) == 0:
        return None, True
    return np.vstack(vectors), False

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

def read_fbin_batch(file_handle, batch_size, dim, header_read=False):
    """Read a batch of vectors from an open fbin file handle. 
    Returns (vectors, eof_reached, header_read).
    Note: header_read should be True after first call."""
    if not header_read:
        # Read header on first call
        header = np.fromfile(file_handle, dtype=np.int32, count=2)
        if len(header) < 2:
            return None, True, True
        n, d = header
        if dim is None:
            dim = d
        elif dim != d:
            raise ValueError(f"Dimension mismatch: expected {dim}, got {d}")
    
    # Read batch_size vectors
    data = np.fromfile(file_handle, dtype=np.float32, count=batch_size * dim)
    if len(data) == 0:
        return None, True, True
    
    actual_count = len(data) // dim
    if actual_count < batch_size:
        # EOF reached
        vectors = data.reshape(actual_count, dim)
        return vectors, True, True
    
    vectors = data.reshape(batch_size, dim)
    return vectors, False, True

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
        # Use executemany for better performance
        data = [(vec.tolist(),) for vec in vectors]
        cur.executemany(f"INSERT INTO {tablename} (vec) VALUES (%s);", data)
        conn.commit()

def create_index(conn, index_type, index_params, tablename = "bigann_vectors"):
    with conn.cursor() as cur:
        print("Checking for existing index and dropping if it exists...")
        cur.execute("DROP INDEX IF EXISTS bigann_vector_idx;")
        cur.execute("DROP INDEX IF EXISTS bigann_ivf_idx;")
        cur.execute("DROP INDEX IF EXISTS bigann_diskann_idx;")
        
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
        
        elif index_type == "diskann":
            max_degree = index_params.get("max_degree", 68)
            ef_construction = index_params.get("ef_construction", 75)
            pq_code_budget_gb = index_params.get("pq_code_budget_gb", None)
            build_dram_budget_gb = index_params.get("build_dram_budget_gb", 30.0)
            
            print(f"Creating DiskANN index with max_degree = {max_degree}, ef_construction = {ef_construction}...")
            if pq_code_budget_gb is not None:
                print(f"  pq_code_budget_gb = {pq_code_budget_gb}")
            print(f"  build_dram_budget_gb = {build_dram_budget_gb}")
            
            start_time = time.time()
            
            # Try USING diskann first, fallback to hnsw if needed
            # Note: DiskANN might use the same access method as HNSW but with different parameters
            # Adjust the SQL syntax based on your pgvectorscale implementation
            if pq_code_budget_gb is not None:
                index_sql = f"""
                    CREATE INDEX bigann_vector_idx ON {tablename}
                    USING diskann (vec vector_l2_ops)
                    WITH (max_degree = %s, ef_construction = %s, 
                          pq_code_budget_gb = %s, build_dram_budget_gb = %s);
                """
                cur.execute(index_sql, (max_degree, ef_construction, pq_code_budget_gb, build_dram_budget_gb))
            else:
                index_sql = f"""
                    CREATE INDEX bigann_vector_idx ON {tablename}
                    USING diskann (vec vector_l2_ops)
                    WITH (max_degree = %s, ef_construction = %s, build_dram_budget_gb = %s);
                """
                cur.execute(index_sql, (max_degree, ef_construction, build_dram_budget_gb))
            
            elapsed_time = time.time() - start_time
            print(f"DiskANN Index creation done in {elapsed_time:.4f} seconds.\n")
        
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        conn.commit()
        print("Index creation complete.\n")


def search_queries(conn, queries, top_k=10, nprobe=20, ef_search=400, search_list_size=None, 
                   query_search_list_size=None, query_rescore=None, tablename="bigann_vectors", index_type="hnsw"):
    with conn.cursor() as cur:
        # Set search parameters based on index type
        if index_type == "ivfflat":
            cur.execute(f"SET ivfflat.probes = {nprobe};")
            print(f"SET ivfflat.probes = {nprobe}")
        elif index_type == "hnsw":
            cur.execute(f"SET hnsw.ef_search = {ef_search};")
            print(f"SET hnsw.ef_search = {ef_search}")
        elif index_type == "diskann":
            # DiskANN uses search_list_size instead of ef_search
            if search_list_size is None:
                search_list_size = ef_search  # Use ef_search as default if not specified
            cur.execute(f"SET diskann.search_list_size = {search_list_size};")
            print(f"SET diskann.search_list_size = {search_list_size}")
            
            # Set query_search_list_size if provided
            if query_search_list_size is not None:
                cur.execute(f"SET diskann.query_search_list_size = {query_search_list_size};")
                print(f"SET diskann.query_search_list_size = {query_search_list_size}")
            
            # Set query_rescore if provided
            if query_rescore is not None:
                cur.execute(f"SET diskann.query_rescore = {query_rescore};")
                print(f"SET diskann.query_rescore = {query_rescore}")

        total_time = 0
        all_results = []

        for i, q in enumerate(queries):
            start = time.time()
            cur.execute(f"""
                SELECT id
                FROM {tablename}
                ORDER BY embedding <-> %s::vector
                LIMIT %s;
            """, (q.tolist(), top_k))

            ids = [row[0] - 1 for row in cur.fetchall()]
            elapsed = time.time() - start
            total_time += elapsed
            all_results.append(ids)

        print(f"\nAverage search time: {total_time / len(queries):.4f} seconds")
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

    recall = total_correct / (len(predicted_ids) * top_k)
    print(f"\nRecall@{top_k}: {recall:.4f}")

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
    parser.add_argument("--do-index", action="store_true", help="Create index")
    parser.add_argument("--do-query", action="store_true", help="Run similarity search queries")
    parser.add_argument("--file", help="Path to .bvecs/.fvecs/.fbin file for insert")
    parser.add_argument("--queries", help="Path to query .bvecs/.fvecs file for search")
    parser.add_argument("--gnd", help="Path to groundtruth .ivecs file for calculating recall")
    parser.add_argument("--num", type=int, default=1000, help="Number of base vectors to insert")
    parser.add_argument("--nq", type=int, default=10, help="Number of queries to run")
    parser.add_argument("--k", type=int, default=100, help="topk")
    parser.add_argument("--index-type", choices=["ivfflat", "hnsw", "diskann"], default="diskann", help="Type of vector index to use")
    parser.add_argument("--M", type=int, default=16, help="HNSW: number of neighbors per node")
    parser.add_argument("--ef-construction", type=int, default=75, help="HNSW/DiskANN: size of dynamic list for graph construction")
    parser.add_argument("--nlist", type=int, default=1000, help="Number of lists for IVF_FLAT index")
    parser.add_argument("--max-degree", type=int, default=68, help="DiskANN: max_degree parameter")
    parser.add_argument("--pq-code-budget-gb", type=float, default=None, help="DiskANN: pq_code_budget_gb parameter (auto-calculated if not specified)")
    parser.add_argument("--build-dram-budget-gb", type=float, default=30.0, help="DiskANN: build_dram_budget_gb parameter")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=5432, type=int)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--dbname", default="postgres")
    parser.add_argument("--tablename", default="test_table")
    parser.add_argument("--batch-size", type=int, default=10000, help="Number of vectors to load and insert per batch")
    parser.add_argument("--search-list-size", type=int, default=None, help="DiskANN: search_list_size parameter (uses ef_search value if not specified)")
    parser.add_argument("--query-search-list-size", type=int, default=None, help="DiskANN: query_search_list_size parameter")
    parser.add_argument("--query-rescore", type=int, default=None, help="DiskANN: query_rescore parameter")
    args = parser.parse_args()

    if not (args.do_insert or args.do_index or args.do_query):
        print("Error: Please specify at least one of --do-insert, --do-index, or --do-query")
        sys.exit(1)

    conn = psycopg2.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.dbname
    )

    if args.do_insert:
        if not args.file:
            print("Error: --file is required for --do-insert")
            sys.exit(1)
        
        # Determine file type from extension
        file_ext = args.file.lower()
        if file_ext.endswith('.bvecs'):
            read_batch_fn = read_bvecs_batch
        elif file_ext.endswith('.fvecs'):
            read_batch_fn = read_fvecs_batch
        elif file_ext.endswith('.fbin'):
            read_batch_fn = read_fbin_batch
        else:
            # Default to bvecs
            read_batch_fn = read_bvecs_batch
        
        print(f"Reading base vectors from: {args.file}")
        print(f"Batch size: {args.batch_size}")
        
        # First, determine dimension by reading file header or first vector
        is_fbin = file_ext.endswith('.fbin')
        with open(args.file, "rb") as f:
            if is_fbin:
                # fbin has header
                header = np.fromfile(f, dtype=np.int32, count=2)
                if len(header) < 2:
                    print("Error: Invalid fbin file: header too short")
                    sys.exit(1)
                n_total, dim = header
                print(f"File contains {n_total} vectors of dimension {dim}")
            else:
                # bvecs/fvecs: read first vector to get dimension
                len_bytes = f.read(4)
                if not len_bytes:
                    print("Error: Empty file")
                    sys.exit(1)
                dim = struct.unpack("i", len_bytes)[0]
                n_total = None  # Unknown for bvecs/fvecs
        
        print("Creating table...")
        create_table(conn, dim, args.tablename)
        print("Inserting vectors in batches...")
        
        total_inserted = 0
        batch_num = 0
        start_time = time.time()
        header_read = False  # Track if header has been read for fbin files
        
        # Determine total vectors for progress calculation
        total_vectors = args.num if args.num else (n_total if n_total else None)
        
        with open(args.file, "rb") as f:
            if is_fbin:
                # Skip header - it will be handled by read_fbin_batch
                pass
            
            while True:
                if args.num and total_inserted >= args.num:
                    break
                
                # Calculate how many vectors to read in this batch
                remaining = args.num - total_inserted if args.num else args.batch_size
                batch_size = min(args.batch_size, remaining) if args.num else args.batch_size
                
                # Read batch
                batch_read_start = time.time()
                if is_fbin:
                    batch_vectors, eof, header_read = read_fbin_batch(f, batch_size, dim, header_read)
                else:
                    batch_vectors, eof = read_batch_fn(f, batch_size, dim)
                
                if batch_vectors is None or eof:
                    break
                
                # Insert batch
                insert_start = time.time()
                insert_vectors(conn, batch_vectors, args.tablename)
                insert_time = time.time() - insert_start
                total_inserted += len(batch_vectors)
                batch_num += 1
                
                # Calculate progress metrics
                elapsed = time.time() - start_time
                rate = total_inserted / elapsed if elapsed > 0 else 0
                
                # Show progress after every batch
                if total_vectors:
                    percentage = (total_inserted / total_vectors) * 100
                    remaining_vectors = total_vectors - total_inserted
                    eta_seconds = remaining_vectors / rate if rate > 0 else 0
                    eta_minutes = eta_seconds / 60
                    
                    print(f"Batch {batch_num}: Inserted {total_inserted:,}/{total_vectors:,} vectors "
                          f"({percentage:.1f}%) | "
                          f"Rate: {rate:.0f} vec/s | "
                          f"Batch time: {insert_time:.2f}s | "
                          f"ETA: {eta_minutes:.1f} min", flush=True)
                else:
                    print(f"Batch {batch_num}: Inserted {total_inserted:,} vectors | "
                          f"Rate: {rate:.0f} vec/s | "
                          f"Batch time: {insert_time:.2f}s", flush=True)
        
        elapsed_time = time.time() - start_time
        final_rate = total_inserted / elapsed_time if elapsed_time > 0 else 0
        print(f"\n{'='*70}")
        print(f"Insertion complete!")
        print(f"Total vectors: {total_inserted:,}")
        print(f"Total batches: {batch_num}")
        print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"Average rate: {final_rate:.0f} vectors/second")
        print(f"{'='*70}\n")

    if args.do_index:
        index_params = {
            "nlist": args.nlist,
            "M": args.M,
            "ef_construction": args.ef_construction,
            "max_degree": args.max_degree,
            "pq_code_budget_gb": args.pq_code_budget_gb,
            "build_dram_budget_gb": args.build_dram_budget_gb,
        }
        print(f"Creating {args.index_type.upper()} index...")
        create_index(conn, args.index_type, index_params=index_params, tablename=args.tablename)

    
    if args.do_query:
        if not args.gnd:
            print("Error: --gnd is required for --do-query")
            sys.exit(1)
        print(f"Reading ground truth from: {args.gnd}")
        
        ground_truth = read_ivecs(args.gnd, 100)
        if not args.queries:
            print("Error: --queries is required for --do-query")
            sys.exit(1)
        print(f"Reading query vectors from: {args.queries}")
        
        # Determine query file type
        query_ext = args.queries.lower()
        if query_ext.endswith('.bvecs'):
            queries = read_bvecs(args.queries, args.nq)
        elif query_ext.endswith('.fvecs'):
            queries = read_fvecs(args.queries, args.nq)
        elif query_ext.endswith('.fbin'):
            queries = read_fbin(args.queries, args.nq)
        else:
            print("Error: Unsupported query file type")
            sys.exit(1)
        
        print(f"Loaded {len(queries)} queries of dimension {queries.shape[1]}")
        print(f"Queries: {queries}")
        
        # Use search_list_size for diskann, ef_search for others
        search_list_size = args.search_list_size
        query_search_list_size = args.query_search_list_size
        query_rescore = args.query_rescore
        
        print("Running similarity search (r1)...")
        predicted = search_queries(conn, queries, args.k, 2, 100, search_list_size, 
                                  query_search_list_size, query_rescore, args.tablename, args.index_type)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r2)...")
        predicted = search_queries(conn, queries, args.k, 3, 100, search_list_size,
                                  query_search_list_size, query_rescore, args.tablename, args.index_type)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r3)...")
        predicted = search_queries(conn, queries, args.k, 5, 200, search_list_size,
                                  query_search_list_size, query_rescore, args.tablename, args.index_type)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r4)...")
        predicted = search_queries(conn, queries, args.k, 10, 300, search_list_size,
                                  query_search_list_size, query_rescore, args.tablename, args.index_type)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r5)...")
        predicted = search_queries(conn, queries, args.k, 3, 100, search_list_size,
                                  query_search_list_size, query_rescore, args.tablename, args.index_type)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r6)...")
        predicted = search_queries(conn, queries, args.k, 5, 200, search_list_size,
                                  query_search_list_size, query_rescore, args.tablename, args.index_type)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r7)...")
        predicted = search_queries(conn, queries, args.k, 10, 300, search_list_size,
                                  query_search_list_size, query_rescore, args.tablename, args.index_type)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r8)...")
        predicted = search_queries(conn, queries, args.k, 20, 400, search_list_size,
                                  query_search_list_size, query_rescore, args.tablename, args.index_type)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r9)...")
        predicted = search_queries(conn, queries, args.k, 30, 500, search_list_size,
                                  query_search_list_size, query_rescore, args.tablename, args.index_type)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r10)...")
        predicted = search_queries(conn, queries, args.k, 40, 600, search_list_size,
                                  query_search_list_size, query_rescore, args.tablename, args.index_type)
        compute_recall(predicted, ground_truth, args.k)

    conn.close()

if __name__ == "__main__":
    main()

# Example usage for DiskANN:
# python static_search_pgvectorscale.py --do-insert --do-index --do-query \
#   --file /ssd_root/dataset/sift/sift_base.fvecs \
#   --queries /ssd_root/dataset/sift/sift_query.fvecs \
#   --gnd /ssd_root/dataset/sift/sift_groundtruth.ivecs \
#   --num 1000000 --nq 10000 --k 100 \
#   --index-type diskann --max-degree 68 --ef-construction 75 \
#   --build-dram-budget-gb 30.0 \
#   --query-search-list-size 200 --query-rescore 100 \
#   --host localhost --port 5432 --user postgres --dbname postgres --tablename bigann_vectors

