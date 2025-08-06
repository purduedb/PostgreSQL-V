import struct
import numpy as np
import psycopg2
import argparse
import time
import sys

def read_bvecs(file_path, num_vectors=None, skip_vectors=0):
    vectors = []
    with open(file_path, "rb") as f:
        skipped = 0
        while skipped < skip_vectors:
            len_bytes = f.read(4)
            if not len_bytes:
                break
            dim = struct.unpack("i", len_bytes)[0]
            f.seek(dim, 1)  # skip vector
            skipped += 1
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

def create_index(conn, nlist):
    with conn.cursor() as cur:
        print("Checking for existing index and dropping if it exists...")
        cur.execute("DROP INDEX IF EXISTS bigann_ivf_idx;")
        print("Creating new IVF_FLAT index with nlist = {nlist}...")
        start_time = time.time()
        cur.execute(f"CREATE INDEX bigann_ivf_idx ON bigann_vectors USING ivfflat (vec vector_ip_ops) WITH (lists = {nlist});")
        elapsed_time = time.time() - start_time
        print(f"Index creation done in {elapsed_time:.4f} seconds.\n")
        conn.commit()

def search_queries(conn, queries, top_k=10, nprobe=20):
    with conn.cursor() as cur:
        cur.execute(f"SET ivfflat.probes = {nprobe};")
        total_time = 0
        all_results = []
        for q in queries:
            start = time.time()
            cur.execute("""
                SELECT id
                FROM bigann_vectors
                ORDER BY vec <#> %s::vector
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
    parser.add_argument("--file", help="Path to .bvecs file for insert")
    parser.add_argument("--queries", help="Path to query .bvecs file for search")
    parser.add_argument("--gnd", help="Path to groundtruth .ivecs file for calculating recall")
    parser.add_argument("--num", type=int, default=1000, help="Number of base vectors to insert")
    parser.add_argument("--total-num", type=int, default=1000, help="Number of total vectors to insert")
    parser.add_argument("--nq", type=int, default=10, help="Number of queries to run")
    parser.add_argument("--k", type=int, default=100, help="topk")
    parser.add_argument("--nlist", type=int, default=1000, help="Number of lists for IVF_FLAT index")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=5432, type=int)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--dbname", default="postgres")
    args = parser.parse_args()

    conn = psycopg2.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.dbname
    )

    if not args.file:
        print("Error: --file is required")
        sys.exit(1)
    # print(f"Reading base vectors from: {args.file}")
    # base_vectors = read_bvecs(args.file, args.num)
    # print(f"Loaded {len(base_vectors)} base vectors of dimension {base_vectors.shape[1]}")
    # print("Creating table...")
    # create_table(conn, base_vectors.shape[1])
    # print("Inserting vectors...")
    # insert_vectors(conn, base_vectors)
    # print("Insertion done.\n")
        
    print(f"Creating IVF_FLAT index with {args.nlist} lists...")
    create_index(conn, args.nlist)
    print("Index creation done.\n")
    
    print(f"Inserting {args.total_num - args.num} additional vectors after the first {args.num}...")
    additional_vectors = read_bvecs(args.file, num_vectors=args.total_num - args.num, skip_vectors=args.num)
    print(f"Loaded {len(additional_vectors)} additional vectors")
    insert_vectors(conn, additional_vectors)
    print("Second insertion done.")

    print(f"Reading ground truth from: {args.gnd}")
    ground_truth = read_ivecs(args.gnd)
    
    if not args.queries:
        print("Error: --queries is required to conduct queries")
        sys.exit(1)
    print(f"Reading query vectors from: {args.queries}")
    queries = read_bvecs(args.queries, args.nq)
    print(f"Loaded {len(queries)} queries of dimension {queries.shape[1]}")
    
    print("Running similarity search (r1)...")
    predicted = search_queries(conn, queries, args.k)
    compute_recall(predicted, ground_truth, args.k)
    
    print("Running similarity search (r2)...")
    predicted = search_queries(conn, queries, args.k)
    compute_recall(predicted, ground_truth, args.k)
    
    time.sleep(3)
    
    print("Running similarity search (r3)...")
    predicted = search_queries(conn, queries, args.k)
    compute_recall(predicted, ground_truth, args.k)
    
    print("Running similarity search (r4)...")
    predicted = search_queries(conn, queries, args.k)
    compute_recall(predicted, ground_truth, args.k)

    conn.close()

if __name__ == "__main__":
    main()


# python bigann_dynamic_search.py \
#   --file /ssd_root/liu4127/sift/bigann_base.bvecs \
#   --queries /ssd_root/liu4127/sift/bigann_query.bvecs \
#   --gnd /ssd_root/liu4127/sift/gnd/idx_1M.ivecs \
#   --num 980_000 --total-num 1_000_000 --nq 1  --k 100  --nlist 1000 \
#   --host localhost --port 5434 --user liu4127 --dbname postgres