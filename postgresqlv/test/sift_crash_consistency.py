import struct
import numpy as np
import psycopg2
import argparse
import time
import sys
import random
import os
import subprocess
import threading

# Global variable to track the total number of vectors inserted
total_inserted_vectors = 0

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
    global total_inserted_vectors, live_vectors
    with conn.cursor() as cur:
        for vec in vectors:
            cur.execute("INSERT INTO sift_vectors (vec) VALUES (%s);", (vec.tolist(),))
            total_inserted_vectors += 1  # Update the global inserted counter
            live_vectors.add(total_inserted_vectors)  # Add the inserted vector ID to live vectors
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

delete_time = 0
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

def trigger_crash(data_dir):
    """
    Simulates a crash by killing the PostgreSQL process using its PID.
    
    :param data_dir: The directory where PostgreSQL's data is stored.
    :return: True if the crash was triggered, False if no crash occurred.
    """
    # Construct the path to postmaster.pid based on the provided data directory
    pid_file = os.path.join(data_dir, "postmaster.pid")
    
    if os.path.exists(pid_file):
        with open(pid_file, "r") as f:
            pid = f.readline().strip()
            print(f"Simulating crash by killing PostgreSQL process with PID {pid}...")
            # Use 'kill' to send SIGKILL to the process
            subprocess.run(["kill", "-9", pid])
            return True
    else:
        print(f"Error: {pid_file} does not exist. PostgreSQL might not be running or the PID file location is incorrect.")
    
    return False

def check_consistency():
    # Placeholder for consistency check
    # Implement the logic for checking consistency after recovery
    print("Consistency check placeholder")

def crash_thread(data_dir):
    # Thread that triggers random crashes during testing
    while True:
        time.sleep(random.randint(0, 1))  # Random delay between 5-15 seconds
        if random.random() < 1:  # 20% chance to trigger crash
            print("Triggering random crash in separate thread...")
            if trigger_crash(data_dir):
                print("A crash is triggered.")
                # time.sleep(5)  # Wait for recovery
                # check_consistency()
            
# Global set to track live vectors
live_vectors = set()

def decide_deletion_ids(deletion_pattern, delete_count):
    """
    Function to decide deletion IDs based on a given pattern.
    
    :param deletion_pattern: The pattern for deletion: "random" or "sliding_window".
    :param delete_count: Number of vectors to delete (for random deletion).
    :return: List of IDs to delete.
    """
    global live_vectors
    
    if deletion_pattern == "random":
        # Random deletion: delete a fraction of the vectors

        # Ensure we don't delete already deleted vectors
        live_ids = list(live_vectors)  # Convert the set of live vectors to a list
        if delete_count > len(live_ids):
            raise ValueError("Not enough live vectors to delete")
        
        delete_ids = random.sample(live_ids, delete_count)
        for vec_id in delete_ids:
            live_vectors.remove(vec_id)  # Remove from live set after deletion

    # elif deletion_pattern == "sliding_window": 

    else:
        raise ValueError("Invalid deletion pattern specified.")
    
    return delete_ids

def main():
    global total_inserted_vectors
    global delete_time

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to .fvecs file for insert")
    parser.add_argument("--queries", help="Path to query .fvecs file for search")
    parser.add_argument("--num", type=int, default=1000, help="Number of base vectors to insert")
    parser.add_argument("--nq", type=int, default=10, help="Number of queries to run")
    parser.add_argument("--insert-batch", type=int, default=1000, help="Number of vectors to insert in each round")
    parser.add_argument("--delete-batch", type=int, default=1000, help="Number of vectors to delete in each round")
    parser.add_argument("--deletion-pattern", choices=["random", "sliding_window"], default="random", help="Pattern for deletion")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=5432, type=int)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--dbname", default="postgres")
    parser.add_argument("--data-dir", default="/var/lib/postgresql/data")
    parser.add_argument("--trigger-crash", action="store_true", help="allocate a thread to trigger a crash")
    args = parser.parse_args()

    conn = psycopg2.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.dbname
    )

    if args.trigger_crash:
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
        print("Error: --queries is required")
        sys.exit(1)
    print(f"Reading query vectors from: {args.queries}")
    queries = read_fvecs(args.queries, args.nq)
    print(f"Loaded {len(queries)} queries of dimension {queries.shape[1]}")
    print("Running similarity search...")
    search_queries(conn, queries)

    # Start the crash simulation thread
    # if args.trigger_crash:
    #     crash_thread_instance = threading.Thread(target=crash_thread, args=(args.data_dir,), daemon=True)
    #     crash_thread_instance.start()
    
    # Loop for crash consistency testing
    while True:
        # Insert more vectors, delete vectors, and vacuum in sequence
        print(f"Inserting {args.insert_batch} additional vectors after the first {total_inserted_vectors}...")
        additional_vectors = read_fvecs(args.file, num_vectors=args.insert_batch, skip_vectors=total_inserted_vectors)
        print(f"Loaded {len(additional_vectors)} additional vectors")
        insert_vectors(conn, additional_vectors)
        print("Second insertion done.")
        
        # Decide deleted IDs based on deletion pattern
        print(f"Deleting vectors based on {args.deletion_pattern} pattern...")
        delete_ids = decide_deletion_ids("random", args.delete_batch)
        delete_vectors(conn, delete_ids)
        print("Deletion done.\n")

        # Vacuum the table
        # vacuum_table(conn, "sift_vectors")
        
        # FIXME: for debugging
        if delete_time == 3:
            # time.sleep(5)
            trigger_crash(args.data_dir)
        delete_time += 1

    # Closing connection at the end of testing
    conn.close()

if __name__ == "__main__":
    main()

# python sift_crash_consistency.py   --file /ssd_root/dataset/sift/sift_base.fvecs   --queries /ssd_root/dataset/sift/sift_query.fvecs   --num 100000  --nq 1  --insert-batch 3000 --delete-batch 1000  --host localhost --port 5434 --user liu4127 --dbname postgres  --data-dir /ssd_root/liu4127/postgresql_vec  --trigger-crash
# python sift_crash_consistency.py   --file /ssd_root/dataset/sift/sift_base.fvecs   --queries /ssd_root/dataset/sift/sift_query.fvecs   --num 100000  --nq 1  --insert-batch 3000 --delete-batch 1000  --host localhost --port 5434 --user liu4127 --dbname postgres  --data-dir /ssd_root/liu4127/postgresql_vec
