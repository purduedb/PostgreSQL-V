import struct
import numpy as np
import psycopg2
import argparse
import time
import sys
import random

def read_fvecs(file_path, num_vectors=None, skip_vectors=0):
    vectors = []
    with open(file_path, "rb") as f:
        skipped = 0
        while skipped < skip_vectors:
            len_bytes = f.read(4)
            if not len_bytes:
                break
            dim = struct.unpack("i", len_bytes)[0]
            f.seek(4 * dim, 1)  # skip vector
            skipped += 1
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

def create_table(conn, dim):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("DROP TABLE IF EXISTS sift_vectors;")
        cur.execute(f"""
            CREATE TABLE sift_vectors (
                id SERIAL PRIMARY KEY,
                vec vector({dim})
            );
        """)
        conn.commit()

def insert_vectors(conn, vectors):
    with conn.cursor() as cur:
        for vec in vectors:
            cur.execute("INSERT INTO sift_vectors (vec) VALUES (%s);", (vec.tolist(),))
        conn.commit()

def create_index(conn):
    with conn.cursor() as cur:
        print("Creating IVF_FLAT index...")
        cur.execute("CREATE INDEX IF NOT EXISTS sift_ivf_idx ON sift_vectors USING ivfflat (vec vector_ip_ops) WITH (lists = 100);")
        conn.commit()

def search_queries(conn, queries, top_k=10):
    with conn.cursor() as cur:
        total_time = 0
        for i, q in enumerate(queries):
            start = time.time()
            cur.execute("""
                SELECT id, vec <#> %s::vector AS distance
                FROM sift_vectors
                ORDER BY distance
                LIMIT %s;
            """, (q.tolist(), top_k))
            results = cur.fetchall()
            elapsed = time.time() - start
            total_time += elapsed
            print(f"Query {i + 1}: top-{top_k} results in {elapsed:.4f} seconds")
        print(f"\nAverage search time: {total_time / len(queries):.4f} seconds")

def delete_vectors(conn, delete_ids):
    with conn.cursor() as cur:
        for vec_id in delete_ids:
            cur.execute("DELETE FROM sift_vectors WHERE id = %s;", (vec_id,))
        conn.commit()
        print(f"Deleted {len(delete_ids)} vectors from the table.")
        
def vacuum_table(conn, table_name):
    print(f"Running VACUUM on {table_name}...")

    # Temporarily enable autocommit
    original_autocommit = conn.autocommit
    conn.autocommit = True

    with conn.cursor() as cur:
        cur.execute(f"VACUUM {table_name};")

    # Restore original autocommit setting
    conn.autocommit = original_autocommit
    print("VACUUM completed (bulkdelete should be triggered).\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to .fvecs file for insert")
    parser.add_argument("--queries", help="Path to query .fvecs file for search")
    parser.add_argument("--num", type=int, default=1000, help="Number of base vectors to insert")
    parser.add_argument("--num2", type=int, default=1000, help="Number of base vectors to insert in round 2")
    parser.add_argument("--nq", type=int, default=10, help="Number of queries to run")
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
        print("Error: --file is required for --do-insert")
        sys.exit(1)
    print(f"Reading base vectors from: {args.file}")
    base_vectors = read_fvecs(args.file, args.num)
    print(f"Loaded {len(base_vectors)} base vectors of dimension {base_vectors.shape[1]}")
    print("Creating table...")
    create_table(conn, base_vectors.shape[1])
    print("Inserting vectors...")
    insert_vectors(conn, base_vectors)
    print("Insertion done.\n")

    print("Creating IVF_FLAT index...")
    create_index(conn)
    print("Index creation done.\n")
    
    if not args.queries:
        print("Error: --queries is required for --do-query")
        sys.exit(1)
    print(f"Reading query vectors from: {args.queries}")
    queries = read_fvecs(args.queries, args.nq)
    print(f"Loaded {len(queries)} queries of dimension {queries.shape[1]}")
    print("Running similarity search...")
    search_queries(conn, queries)
    
    # insert after search
    print(f"Insert after search")
    print(f"Inserting {args.num2} additional vectors after the first {args.num}...")
    additional_vectors = read_fvecs(args.file, num_vectors=args.num2, skip_vectors=args.num)
    print(f"Loaded {len(additional_vectors)} additional vectors")
    insert_vectors(conn, additional_vectors)
    print("Second insertion done.")
    
    time.sleep(3)
    
    # conduct query after the second round insertion
    if not args.queries:
        print("Error: --queries is required for --do-query")
        sys.exit(1)
    print(f"Reading query vectors from: {args.queries}")
    queries = read_fvecs(args.queries, args.nq)
    print(f"Loaded {len(queries)} queries of dimension {queries.shape[1]}")
    print("Running similarity search...")
    search_queries(conn, queries)
        
    # Deletion phase
    print("Deleting half of the initially inserted vectors (based on id)...")
    total_num = args.num + args.num2
    delete_ids = random.sample(range(1, total_num + 1), total_num // 2)
    delete_vectors(conn, delete_ids)
    print("Deletion done.\n")
    
    # Trigger index bulkdelete by vacuum
    vacuum_table(conn, "sift_vectors")
    
    time.sleep(3)
    
    # Query after deletion
    print("Running similarity search after deletion...")
    queries = read_fvecs(args.queries, args.nq)
    print(f"Loaded {len(queries)} queries of dimension {queries.shape[1]}")
    search_queries(conn, queries)

    time.sleep(3)

    conn.close()

if __name__ == "__main__":
    main()

# do everything
# python sift_dynamic_search.py   --file /ssd_root/liu4127/sift/sift_base.fvecs   --queries /ssd_root/liu4127/sift/sift_query.fvecs   --num 10000 --num2 10003 --nq 1   --host localhost --port 5434 --user liu4127 --dbname postgres