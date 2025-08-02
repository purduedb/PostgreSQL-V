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
            f.seek(dim, 1)
            skipped += 1

        read = 0
        while True:
            if num_vectors is not None and read >= num_vectors:
                break
            len_bytes = f.read(4)
            if not len_bytes:
                break
            dim = struct.unpack("i", len_bytes)[0]
            vec = np.frombuffer(f.read(dim), dtype=np.uint8)
            vectors.append(vec)
            read += 1
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

def search_queries(conn, queries, top_k=10, nprobe=100):
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
    parser.add_argument("--do-insert", action="store_true", help="Insert base vectors into table")
    parser.add_argument("--do-index", action="store_true", help="Create IVF_FLAT index")
    parser.add_argument("--do-query", action="store_true", help="Run similarity search queries")
    parser.add_argument("--file", help="Path to .bvecs file for insert")
    parser.add_argument("--queries", help="Path to query .bvecs file for search")
    parser.add_argument("--gnd", help="Path to groundtruth .ivecs file for calculating recall")
    parser.add_argument("--num", type=int, default=1000, help="Number of base vectors to insert")
    parser.add_argument("--nq", type=int, default=10, help="Number of queries to run")
    parser.add_argument("--k", type=int, default=100, help="topk")
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

        print(f"Opening .bvecs file: {args.file}")
        inserted = 0
        batch_size = 1_000_000
        dim = None

        while inserted < args.num:
            to_read = min(batch_size, args.num - inserted)
            print(f"Reading vectors {inserted} to {inserted + to_read - 1}")
            batch_vectors = read_bvecs(args.file, num_vectors=to_read, skip_vectors=inserted)

            if dim is None:
                dim = batch_vectors.shape[1]
                print(f"Creating table with dimension {dim}")
                create_table(conn, dim)

            print(f"Inserting batch of {len(batch_vectors)} vectors...")
            insert_vectors(conn, batch_vectors)
            inserted += len(batch_vectors)
            print(f"Total inserted: {inserted}/{args.num}")

        print("All vectors inserted.")
        
    if args.do_index:
        print(f"Creating IVF_FLAT index with {args.nlist} lists...")
        create_index(conn, args.nlist)
        print("Index creation done.\n")

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
        predicted = search_queries(conn, queries, args.k)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r2)...")
        predicted = search_queries(conn, queries, args.k)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r3)...")
        predicted = search_queries(conn, queries, args.k)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r4)...")
        predicted = search_queries(conn, queries, args.k)
        compute_recall(predicted, ground_truth, args.k)
        
        print("Running similarity search (r5)...")
        predicted = search_queries(conn, queries, args.k)
        compute_recall(predicted, ground_truth, args.k)

    
    conn.close()


if __name__ == "__main__":
    main()

# create & query (100M) nprobes = 100
# python bigann_insert.py  --do-insert  --do-index --do-query \
#   --file /ssd_root/liu4127/sift/bigann_base.bvecs \
#   --queries /ssd_root/liu4127/sift/bigann_query.bvecs \
#   --gnd /ssd_root/liu4127/sift/gnd/idx_10M.ivecs \
#   --num 100_000_000 --nq 10_000  --k 100  --nlist 10000 \
#   --host localhost --port 5434 --user liu4127 --dbname postgres