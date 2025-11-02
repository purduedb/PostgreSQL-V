import struct
import numpy as np
import psycopg2
import argparse
import time
import sys

from psycopg2 import sql


def read_bvecs(file_path, num_vectors=None, skip=0):
    with open(file_path, "rb") as f:
        vectors = []
        # skip entries
        for _ in range(skip):
            len_bytes = f.read(4)
            if not len_bytes:
                break
            dim = struct.unpack("i", len_bytes)[0]
            f.seek(dim, 1)  # skip dim bytes (uint8)
        # read entries
        while True:
            len_bytes = f.read(4)
            if not len_bytes:
                break
            dim = struct.unpack("i", len_bytes)[0]
            vec = np.frombuffer(f.read(dim), dtype=np.uint8)
            vectors.append(vec)
            if num_vectors and len(vectors) >= num_vectors:
                break
    if not vectors:
        return np.empty((0, 0), dtype=np.uint8)
    return np.vstack(vectors)

def read_fvecs(file_path, num_vectors=None, skip=0):
    with open(file_path, "rb") as f:
        vectors = []
        # skip entries
        for _ in range(skip):
            len_bytes = f.read(4)
            if not len_bytes:
                break
            dim = struct.unpack("i", len_bytes)[0]
            f.seek(4 * dim, 1)  # skip dim float32s
        # read entries
        while True:
            len_bytes = f.read(4)
            if not len_bytes:
                break
            dim = struct.unpack("i", len_bytes)[0]
            vec = np.frombuffer(f.read(4 * dim), dtype=np.float32)
            vectors.append(vec)
            if num_vectors and len(vectors) >= num_vectors:
                break
    if not vectors:
        return np.empty((0, 0), dtype=np.float32)
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
        header = np.fromfile(f, dtype=np.int32, count=2)
        if len(header) < 2:
            raise ValueError("Invalid fbin file: header too short")
        n, d = header
        print(f"File contains {n} vectors of dimension {d}")
        count = n if num_vectors is None else min(num_vectors, n)
        data = np.fromfile(f, dtype=np.float32, count=count * d)
        vectors = data.reshape(count, d)
    return vectors

def load_groundtruth_bin_file(filename):
    with open(filename, "rb") as f:
        npts, ndims = struct.unpack("ii", f.read(8))
        num_ids = npts * ndims
        id_data = f.read(num_ids * 4)
        ids = np.frombuffer(id_data, dtype=np.uint32).reshape(npts, ndims)
        dist_data = f.read(npts * ndims * 4)
        dists = np.frombuffer(dist_data, dtype=np.float32).reshape(npts, ndims)
    return ids.tolist()

def create_table(conn, dim, tablename="bigann_vectors"):
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

def insert_vectors(conn, vectors, tablename="bigann_vectors"):
    with conn.cursor() as cur:
        for vec in vectors:
            cur.execute(f"INSERT INTO {tablename} (vec) VALUES (%s);", (vec.tolist(),))
        conn.commit()

def create_index(conn, index_type, index_params, tablename="bigann_vectors"):
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

def search_queries(conn, queries, top_k=10, nprobe=20, ef_search=400, tablename="bigann_vectors"):
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
            ids = [row[0] - 1 for row in cur.fetchall()]  # zero-based
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

def main():
    parser = argparse.ArgumentParser()
    # Round 1 (original)
    parser.add_argument("--do-insert", action="store_true", help="Insert base vectors into table (round 1)")
    parser.add_argument("--file", help="Path to .fvecs/.bvecs file for round-1 insert")
    parser.add_argument("--num", type=int, default=1000, help="Number of base vectors to insert (round 1)")
    # Index
    parser.add_argument("--do-index", action="store_true", help="Create vector index")
    parser.add_argument("--index-type", choices=["ivfflat", "hnsw"], default="ivfflat", help="Type of vector index to use")
    parser.add_argument("--M", type=int, default=16, help="HNSW: number of neighbors per node")
    parser.add_argument("--ef-construction", type=int, default=64, help="HNSW: size of dynamic list for graph construction")
    parser.add_argument("--nlist", type=int, default=1000, help="Number of lists for IVF_FLAT index")
    # Round 2 (new)
    parser.add_argument("--do-insert2", action="store_true",
                        help="Insert additional vectors AFTER indexing (round 2)")
    parser.add_argument("--file2", help="Optional path to .fvecs/.bvecs file for round-2 insert (defaults to --file)")
    parser.add_argument("--num2", type=int, default=0,
                        help="How many vectors to insert in round 2 (required if --do-insert2)")
    parser.add_argument("--skip2", type=int, default=0,
                        help="How many vectors to skip before inserting in round 2")
    # Query & recall
    parser.add_argument("--do-query", action="store_true", help="Run similarity search queries")
    parser.add_argument("--queries", help="Path to query .fvecs file for search")
    parser.add_argument("--gnd", help="Path to groundtruth .ivecs file for calculating recall")
    parser.add_argument("--nq", type=int, default=10, help="Number of queries to run")
    parser.add_argument("--k", type=int, default=100, help="topk")
    # DB & table
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=5432, type=int)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--dbname", default="postgres")
    parser.add_argument("--tablename", default="test_table")
    args = parser.parse_args()

    if not (args.do_insert or args.do_index or args.do_query or args.do_insert2):
        print("Error: Please specify at least one of --do-insert, --do-index, --do-query, or --do-insert2")
        sys.exit(1)

    conn = psycopg2.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.dbname
    )

    # Round 1 insert (and table creation)
    if args.do_insert:
        if not args.file:
            print("Error: --file is required for --do-insert")
            sys.exit(1)
        print(f"Reading base vectors from: {args.file}")
        # prefer fvecs; if needed, you can switch to read_bvecs here
        base_vectors = read_fvecs(args.file, args.num)
        if base_vectors.size == 0:
            print("Error: no vectors loaded for round 1 insert")
            sys.exit(1)
        print(f"Loaded {len(base_vectors)} base vectors of dimension {base_vectors.shape[1]}")
        print("Creating table...")
        create_table(conn, base_vectors.shape[1], args.tablename)
        print("Inserting vectors (round 1)...")
        insert_vectors(conn, base_vectors, args.tablename)
        print("Round 1 insertion done.\n")

    # Index build
    if args.do_index:
        index_params = {
            "nlist": args.nlist,
            "M": args.M,
            "ef_construction": args.ef_construction,
        }
        print(f"Creating {args.index_type.upper()} index...")
        create_index(conn, args.index_type, index_params=index_params, tablename=args.tablename)

    # Round 2 insert (post-index)
    if args.do_insert2:
        if args.num2 <= 0:
            print("Error: --num2 must be > 0 when using --do-insert2")
            sys.exit(1)
        src_file = args.file2 if args.file2 else args.file
        if not src_file:
            print("Error: --file2 (or --file) is required for --do-insert2")
            sys.exit(1)
        print(f"Reading round-2 vectors from: {src_file} (skip {args.skip2}, take {args.num2})")
        # Try fvecs first; switch to bvecs if you store bytes.
        round2_vectors = read_fvecs(src_file, num_vectors=args.num2, skip=args.skip2)
        if round2_vectors.size == 0:
            print("Warning: No vectors loaded for round 2 insert (check file/skip/num2).")
        else:
            print(f"Loaded {len(round2_vectors)} vectors for round 2 (dim {round2_vectors.shape[1]})")
            print("Inserting vectors (round 2, post-index)...")
            insert_vectors(conn, round2_vectors, args.tablename)
            print("Round 2 insertion done.\n")

    # Query & recall
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
        queries = read_fvecs(args.queries, args.nq)
        print(f"Loaded {len(queries)} queries of dimension {queries.shape[1]}")
        
        # # sleep if needed
        # print("Waiting 10 seconds before running queries...")
        # time.sleep(10)
        
        print("Running similarity search (r1)...")
        predicted = search_queries(conn, queries, args.k, 2, 100, args.tablename)
        compute_recall(predicted, ground_truth, args.k)
        
        # print("Running similarity search (r3)...")
        # predicted = search_queries(conn, queries, args.k, 3, 100, args.tablename)
        # compute_recall(predicted, ground_truth, args.k)
        
        # print("Running similarity search (r4)...")
        # predicted = search_queries(conn, queries, args.k, 5, 200, args.tablename)
        # compute_recall(predicted, ground_truth, args.k)
        
        # print("Running similarity search (r5)...")
        # predicted = search_queries(conn, queries, args.k, 10, 300, args.tablename)
        # compute_recall(predicted, ground_truth, args.k)
        
        # print("Running similarity search (r6)...")
        # predicted = search_queries(conn, queries, args.k, 20, 400, args.tablename)
        # compute_recall(predicted, ground_truth, args.k)
        
        # print("Running similarity search (r7)...")
        # predicted = search_queries(conn, queries, args.k, 30, 500, args.tablename)
        # compute_recall(predicted, ground_truth, args.k)

    conn.close()

if __name__ == "__main__":
    main()

# hnsw build and query sift1M
# python dynamic_search.py --do-index --do-insert2 --do-query \
#   --file /ssd_root/dataset/sift/sift_base.fvecs \
#   --queries /ssd_root/dataset/sift/sift_query.fvecs \
#   --gnd /ssd_root/dataset/sift/sift_groundtruth.ivecs \
#   --num 800000 --num2 200000 --skip2 800000 --nq 10000 --k 100 \
#   --index-type hnsw --M 16 --ef-construction 40 \
#   --host localhost --port 543* --user liu4127 --dbname postgres --tablename bigann_vectors
