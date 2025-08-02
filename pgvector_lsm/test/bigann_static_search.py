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

def read_ivecs(file_path, expected_k=1000):
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

def create_table(conn, dim):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("DROP TABLE IF EXISTS bigann_vectors;")
        cur.execute(f"""
            CREATE TABLE bigann_vectors (
                id SERIAL PRIMARY KEY,
                vec vector({dim})
            );
        """)
        conn.commit()

def insert_vectors(conn, vectors):
    with conn.cursor() as cur:
        for vec in vectors:
            cur.execute("INSERT INTO bigann_vectors (vec) VALUES (%s);", (vec.tolist(),))
        conn.commit()

def create_index(conn, index_type, index_params):
    with conn.cursor() as cur:
        print("Checking for existing index and dropping if it exists...")
        cur.execute("DROP INDEX IF EXISTS bigann_vector_idx;")
        cur.execute("DROP INDEX IF EXISTS bigann_ivf_idx;")
        
        if index_type == "ivfflat":
            nlist = index_params.get("nlist", 100)
            print(f"Creating IVF_FLAT index with nlist = {nlist}...")
            start_time = time.time()
            index_sql = sql.SQL("""
                CREATE INDEX bigann_vector_idx ON bigann_vectors
                USING ivfflat (vec vector_l2_ops)
                WITH (lists = %s);
            """)
            cur.execute(index_sql, (nlist,))
            elapsed_time = time.time() - start_time
            print(f"IVFFLAT Index creation done in {elapsed_time:.4f} seconds.\n")
        
        elif index_type == "hnsw":
            m = index_params.get("M", 32)
            ef_construction = index_params.get("ef_construction", 64)
            print(f"Creating HNSW index with M = {m}, ef_construction = {ef_construction}...")
            start_time = time.time()
            index_sql = sql.SQL("""
                CREATE INDEX bigann_vector_idx ON bigann_vectors
                USING hnsw (vec vector_l2_ops)
                WITH (M = %s, ef_construction = %s);
            """)
            cur.execute(index_sql, (m, ef_construction))
            elapsed_time = time.time() - start_time
            print(f"HNSW Index creation done in {elapsed_time:.4f} seconds.\n")
        
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        conn.commit()
        print("Index creation complete.\n")


def search_queries(conn, queries, top_k=10, nprobe=20, ef_search=400):
    with conn.cursor() as cur:
        cur.execute(f"SET ivfflat.probes = {nprobe};")
        cur.execute(f"SET hnsw.ef_search = {ef_search};")
        print(f"SET hnsw.ef_search = {ef_search}, ivfflat.probes = {nprobe}")
        total_time = 0
        all_results = []
        for q in queries:
            start = time.time()
            cur.execute("""
                SELECT id
                FROM bigann_vectors
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

        # print(f"Query {i}:")
        # print(f"  Predicted IDs:     {pred}")
        # print(f"  Ground Truth Top-{top_k}: {list(gt)}")
        # print(f"  Correct matches:   {correct}")

    recall = total_correct / (len(predicted_ids) * top_k)
    print(f"\nRecall@{top_k}: {recall:.4f}")
    
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
        print(f"Reading base vectors from: {args.file}")
        base_vectors = read_bvecs(args.file, args.num)
        print(f"Loaded {len(base_vectors)} base vectors of dimension {base_vectors.shape[1]}")
        print("Creating table...")
        create_table(conn, base_vectors.shape[1])
        print("Inserting vectors...")
        insert_vectors(conn, base_vectors)
        print("Insertion done.\n")

    if args.do_index:
        index_params = {
            "nlist": args.nlist,
            "M": args.M,
            "ef_construction": args.ef_construction,
        }
        print(f"Creating {args.index_type.upper()} index...")
        create_index(conn, args.index_type, index_params=index_params)


    print(f"Reading ground truth from: {args.gnd}")
    ground_truth = read_ivecs(args.gnd)
    
    if args.do_query:
        if not args.queries:
            print("Error: --queries is required for --do-query")
            sys.exit(1)
        print(f"Reading query vectors from: {args.queries}")
        queries = read_bvecs(args.queries, args.nq)
        print(f"Loaded {len(queries)} queries of dimension {queries.shape[1]}")
        
        print("Running similarity search (r1)...")
        predicted = search_queries(conn, queries, args.k, 2, 80)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r2)...")
        predicted = search_queries(conn, queries, args.k, 2, 80)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r3)...")
        predicted = search_queries(conn, queries, args.k, 3, 100)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r4)...")
        predicted = search_queries(conn, queries, args.k, 5, 200)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r5)...")
        predicted = search_queries(conn, queries, args.k, 10, 300)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r6)...")
        predicted = search_queries(conn, queries, args.k, 20, 400)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r7)...")
        predicted = search_queries(conn, queries, args.k, 30, 500)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r8)...")
        predicted = search_queries(conn, queries, args.k, 40, 600)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r9)...")
        predicted = search_queries(conn, queries, args.k, 50, 700)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r10)...")
        predicted = search_queries(conn, queries, args.k, 60, 800)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r11)...")
        predicted = search_queries(conn, queries, args.k, 70, 900)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r12)...")
        predicted = search_queries(conn, queries, args.k, 80, 1000)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r13)...")
        predicted = search_queries(conn, queries, args.k, 90, 1100)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r14)...")
        predicted = search_queries(conn, queries, args.k, 100, 1200)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r15)...")
        predicted = search_queries(conn, queries, args.k, 110, 1300)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r16)...")
        predicted = search_queries(conn, queries, args.k, 120, 1400)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r17)...")
        predicted = search_queries(conn, queries, args.k, 130, 1500)
        compute_recall(predicted, ground_truth, args.k)

    conn.close()

if __name__ == "__main__":
    main()


# diskann 1M hnsw w/ insertion
# python bigann_static_search.py --do-insert --do-index --do-query \
#   --file /ssd_root/dataset/sift/bigann_base.bvecs \
#   --queries /ssd_root/dataset/sift/bigann_query.bvecs \
#   --gnd /ssd_root/dataset/sift/gnd/idx_1M.ivecs \
#   --num 1000000 --nq 10000 --k 100 \
#   --index-type hnsw --M 16 --ef-construction 40 \
#   --host localhost --port 5434 --user liu4127 --dbname postgres

# diskann 1M hnsw w/o insertion
# python bigann_static_search.py --do-index --do-query \
#   --file /ssd_root/dataset/sift/bigann_base.bvecs \
#   --queries /ssd_root/dataset/sift/bigann_query.bvecs \
#   --gnd /ssd_root/dataset/sift/gnd/idx_1M.ivecs \
#   --num 1000000 --nq 10000 --k 100 \
#   --index-type hnsw --M 16 --ef-construction 40 \
#   --host localhost --port 5434 --user liu4127 --dbname postgres

# diskann 10M hnsw w/o insertion
# python bigann_static_search.py --do-index --do-query \
#   --file /ssd_root/dataset/sift/bigann_base.bvecs \
#   --queries /ssd_root/dataset/sift/bigann_query.bvecs \
#   --gnd /ssd_root/dataset/sift/gnd/idx_10M.ivecs \
#   --num 10000000 --nq 10000 --k 100 \
#   --index-type hnsw --M 16 --ef-construction 40 \
#   --host localhost --port 5434 --user liu4127 --dbname postgres

# diskann 10M ivfflat w/ insertion
# python bigann_static_search.py --do-insert --do-index --do-query \
#   --file /ssd_root/dataset/sift/bigann_base.bvecs \
#   --queries /ssd_root/dataset/sift/bigann_query.bvecs \
#   --gnd /ssd_root/dataset/sift/gnd/idx_10M.ivecs \
#   --num 10000000 --nq 10000 --k 100 \
#   --index-type ivfflat --nlist 3162 \
#   --host localhost --port 5434 --user liu4127 --dbname postgres

