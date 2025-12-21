import struct
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import argparse
import time
import sys
import threading
from queue import Queue
from collections import defaultdict


def read_fvecs(file_path, num_vectors=None, skip=0):
    """Read vectors from .fvecs file format"""
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


def read_bvecs(file_path, num_vectors=None, skip=0):
    """Read vectors from .bvecs file format"""
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


def create_table(conn, dim, tablename="test_vectors"):
    """Create a table for storing vectors"""
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


def insert_vectors_batch(conn, vectors, batch_size=100, tablename="test_vectors", start_id=None):
    """Insert vectors in batches for better performance
    
    Args:
        conn: Database connection
        vectors: Array of vectors to insert
        batch_size: Size of each batch
        tablename: Name of the table
        start_id: Starting ID for the first vector (1-based). If None, uses auto-increment.
    """
    with conn.cursor() as cur:
        max_id = None
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            if start_id is not None:
                # Insert with explicit IDs
                batch_start_id = start_id + i
                values = [(batch_start_id + j, vec.tolist()) for j, vec in enumerate(batch)]
                execute_values(
                    cur,
                    f"INSERT INTO {tablename} (id, vec) VALUES %s;",
                    values
                )
                # Track the maximum ID we've inserted
                max_id = batch_start_id + len(batch) - 1
            else:
                # Insert without IDs (auto-increment)
                execute_values(
                    cur,
                    f"INSERT INTO {tablename} (vec) VALUES %s;",
                    [(vec.tolist(),) for vec in batch]
                )
        conn.commit()
        
        # Update the sequence to be at least as high as the maximum ID we inserted
        # This prevents conflicts if auto-increment is used later
        # Note: With concurrent inserts, this is done per-thread, but PostgreSQL's setval
        # is safe for concurrent use and will use the maximum value
        if max_id is not None:
            with conn.cursor() as cur:
                # Update sequence to be at least max_id (using GREATEST to handle concurrent updates)
                cur.execute(f"""
                    SELECT setval(pg_get_serial_sequence('{tablename}', 'id'), 
                                  GREATEST(COALESCE((SELECT MAX(id) FROM {tablename}), 0), {max_id}), 
                                  true);
                """)
            conn.commit()


def create_index(conn, index_type, index_params, tablename="test_vectors"):
    """Create a vector index"""
    with conn.cursor() as cur:
        print("Dropping existing index if it exists...")
        cur.execute(f"DROP INDEX IF EXISTS {tablename}_idx;")
        
        if index_type == "ivfflat":
            nlist = index_params.get("nlist", 100)
            print(f"Creating IVF_FLAT index with nlist = {nlist}...")
            start_time = time.time()
            index_sql = f"""
                CREATE INDEX {tablename}_idx ON {tablename}
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
                CREATE INDEX {tablename}_idx ON {tablename}
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


def insert_worker(conn_params, insert_queue, stats, tablename="test_vectors", thread_id=0, progress_interval=1000):
    """Worker thread that inserts vectors from the queue
    
    Each item in the queue should be a tuple: (vectors, start_id) where:
    - vectors: numpy array of vectors to insert
    - start_id: starting ID (1-based) for the first vector in the batch
    """
    conn = psycopg2.connect(
        host=conn_params['host'],
        port=conn_params['port'],
        user=conn_params['user'],
        password=conn_params['password'],
        dbname=conn_params['dbname']
    )
    
    try:
        inserted_count = 0
        total_time = 0.0
        
        while True:
            try:
                # Get next vector batch from queue (timeout to allow checking if done)
                item = insert_queue.get(timeout=1)
                if item is None:  # Sentinel value to signal completion
                    break
                
                # Extract vectors and start_id from queue item
                if isinstance(item, tuple) and len(item) == 2:
                    vectors, start_id = item
                else:
                    # Backward compatibility: if item is just vectors, use None for start_id
                    vectors = item
                    start_id = None
                
                start_time = time.time()
                
                # Insert vectors with explicit IDs
                insert_vectors_batch(conn, vectors, batch_size=100, tablename=tablename, start_id=start_id)
                
                elapsed = time.time() - start_time
                batch_size = len(vectors)
                inserted_count += batch_size
                total_time += elapsed
                
                insert_queue.task_done()
                
                # Update statistics and check for progress reporting
                current_time = time.time()
                with stats['lock']:
                    stats['inserted_count'] += batch_size
                    stats['insert_time'] += elapsed
                    current_total = stats['inserted_count']
                    start_time = stats.get('start_time')
                    if start_time is None:
                        start_time = current_time
                    last_progress = stats.get('last_insert_progress', 0)
                    last_progress_time = stats.get('last_progress_time')
                    if last_progress_time is None:
                        last_progress_time = current_time
                    
                    # Report progress every progress_interval vectors or every 5 seconds
                    should_report = (current_total - last_progress >= progress_interval) or (current_time - last_progress_time >= 5.0)
                    
                    if should_report:
                        elapsed_total = current_time - start_time
                        rate = current_total / elapsed_total if elapsed_total > 0 else 0
                        stats['last_insert_progress'] = current_total
                        stats['last_progress_time'] = current_time
                        print(f"[INSERT PROGRESS] Total inserted: {current_total:,} vectors | "
                              f"Rate: {rate:.0f} vectors/sec | "
                              f"Thread {thread_id}: {inserted_count:,} vectors")
                
            except:
                # Queue timeout or empty, check if we should continue
                if insert_queue.empty():
                    break
        
        # Update final statistics
        with stats['lock']:
            stats['insert_threads_completed'] += 1
        
        print(f"Insert thread {thread_id}: completed - inserted {inserted_count:,} vectors in {total_time:.4f} seconds "
              f"({inserted_count/total_time:.0f} vectors/sec)")
        
    finally:
        conn.close()


def query_worker_once(conn_params, query_queue, results, stats, top_k=10, nprobe=20, 
                      ef_search=400, tablename="test_vectors", thread_id=0, progress_interval=100):
    """Worker thread that executes queries from the queue once (for after-insert mode)"""
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
            
            query_count = 0
            total_time = 0.0
            
            while True:
                try:
                    # Get next query from queue (timeout to allow checking if done)
                    item = query_queue.get(timeout=1)
                    if item is None:  # Sentinel value to signal completion
                        break
                    
                    query_index, query_vec = item
                    start_time = time.time()
                    
                    # Execute the query
                    cur.execute(f"""
                        SELECT id
                        FROM {tablename}
                        ORDER BY vec <-> %s::vector
                        LIMIT %s;
                    """, (query_vec.tolist(), top_k))
                    
                    # Subtract 1 from each ID to get zero-based indexing
                    ids = [row[0] - 1 for row in cur.fetchall()]
                    elapsed = time.time() - start_time
                    
                    # Store results
                    with results['lock']:
                        results[query_index] = ids
                    
                    query_count += 1
                    total_time += elapsed
                    
                    # Update statistics and check for progress reporting
                    current_time = time.time()
                    with stats['lock']:
                        stats['query_count'] += 1
                        stats['query_time'] += elapsed
                        current_total = stats['query_count']
                        start_time = stats.get('start_time')
                        if start_time is None:
                            start_time = current_time
                        last_progress = stats.get('last_query_progress', 0)
                        last_progress_time = stats.get('last_progress_time')
                        if last_progress_time is None:
                            last_progress_time = current_time
                        
                        # Report progress every progress_interval queries or every 5 seconds
                        should_report = (current_total - last_progress >= progress_interval) or (current_time - last_progress_time >= 5.0)
                        
                        if should_report:
                            elapsed_total = current_time - start_time
                            rate = current_total / elapsed_total if elapsed_total > 0 else 0
                            avg_time = stats['query_time'] / current_total if current_total > 0 else 0
                            stats['last_query_progress'] = current_total
                            stats['last_progress_time'] = current_time
                            print(f"[QUERY PROGRESS] Total queries: {current_total:,} | "
                                  f"Rate: {rate:.1f} queries/sec | "
                                  f"Avg time: {avg_time*1000:.2f}ms | "
                                  f"Thread {thread_id}: {query_count:,} queries")
                    
                    query_queue.task_done()
                    
                except:
                    # Queue timeout or empty, check if we should continue
                    if query_queue.empty():
                        break
            
            # Update final statistics
            with stats['lock']:
                stats['query_threads_completed'] += 1
            
            print(f"Query thread {thread_id}: completed - {query_count:,} queries in {total_time:.4f} seconds "
                  f"({query_count/total_time:.1f} queries/sec)" if total_time > 0 else f"Query thread {thread_id}: completed - {query_count:,} queries")
    
    finally:
        conn.close()


def query_worker(conn_params, query_queue, round_results, results, stats, stop_flag, round_complete_event,
                 round_lock, top_k=10, nprobe=20, ef_search=400, tablename="test_vectors", 
                 thread_id=0, progress_interval=100, ground_truth=None, num_queries=None):
    """Worker thread that collaborates with other threads to process queries in rounds (for concurrent mode)
    
    All threads work together on each round: they pull queries from a shared queue until the round is complete.
    After each complete round, one thread computes and prints recall if ground truth is available.
    """
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
            
            query_count = 0
            total_time = 0.0
            round_count = 0
            
            # Keep processing rounds until stop flag is set
            while not stop_flag.is_set():
                # Process queries for current round from shared queue
                round_query_count = 0
                got_sentinel = False
                
                while not stop_flag.is_set() and not got_sentinel:
                    try:
                        # Get next query from queue (timeout to allow checking stop flag)
                        item = query_queue.get(timeout=1)
                        if item is None:  # Sentinel value - this thread's portion of round is complete
                            query_queue.task_done()
                            got_sentinel = True
                            break
                        
                        query_index, query_vec = item
                        start_time = time.time()
                        
                        # Execute the query
                        cur.execute(f"""
                            SELECT id
                            FROM {tablename}
                            ORDER BY vec <-> %s::vector
                            LIMIT %s;
                        """, (query_vec.tolist(), top_k))
                        
                        # Subtract 1 from each ID to get zero-based indexing
                        ids = [row[0] - 1 for row in cur.fetchall()]
                        elapsed = time.time() - start_time
                        
                        # Store results for current round (shared across threads)
                        with round_results['lock']:
                            round_results[query_index] = ids
                        
                        # Store results in global results (only from first round to avoid overwriting)
                        if round_count == 0:
                            with results['lock']:
                                if query_index not in results:
                                    results[query_index] = ids
                        
                        query_count += 1
                        round_query_count += 1
                        total_time += elapsed
                        
                        query_queue.task_done()
                        
                        # Update statistics and check for progress reporting
                        current_time = time.time()
                        with stats['lock']:
                            stats['query_count'] += 1
                            stats['query_time'] += elapsed
                            current_total = stats['query_count']
                            start_time = stats.get('start_time')
                            if start_time is None:
                                start_time = current_time
                            last_progress = stats.get('last_query_progress', 0)
                            last_progress_time = stats.get('last_progress_time')
                            if last_progress_time is None:
                                last_progress_time = current_time
                            
                            # Report progress every progress_interval queries or every 5 seconds
                            should_report = (current_total - last_progress >= progress_interval) or (current_time - last_progress_time >= 5.0)
                            
                            if should_report:
                                elapsed_total = current_time - start_time
                                rate = current_total / elapsed_total if elapsed_total > 0 else 0
                                avg_time = stats['query_time'] / current_total if current_total > 0 else 0
                                stats['last_query_progress'] = current_total
                                stats['last_progress_time'] = current_time
                                print(f"[QUERY PROGRESS] Total queries: {current_total:,} | "
                                      f"Rate: {rate:.1f} queries/sec | "
                                      f"Avg time: {avg_time*1000:.2f}ms | "
                                      f"Thread {thread_id}: Round {round_count + 1}, {round_query_count} queries")
                    
                    except:
                        # Queue timeout or empty - check if we should continue
                        if stop_flag.is_set():
                            break
                        # If queue is empty and we haven't got sentinel, wait a bit
                        if query_queue.empty():
                            time.sleep(0.01)
                
                # Wait for all threads to finish current round (coordinator will set this)
                # Use timeout to periodically check stop flag
                if not stop_flag.is_set():
                    while not round_complete_event.wait(timeout=0.5):
                        if stop_flag.is_set():
                            break
                
                # One thread computes recall for the completed round
                with round_lock:
                    # Check if recall has already been computed for this round
                    if stats.get('last_computed_round', -1) < round_count + 1:
                        stats['last_computed_round'] = round_count + 1
                        should_compute_recall = True
                    else:
                        should_compute_recall = False
                
                if should_compute_recall and ground_truth is not None and num_queries is not None:
                    # Reconstruct results in order for this round
                    with round_results['lock']:
                        round_results_ordered = []
                        for i in range(num_queries):
                            if i in round_results:
                                round_results_ordered.append(round_results[i])
                            else:
                                round_results_ordered.append([])
                    
                    # Compute recall for this round
                    print(f"\n[ROUND {round_count + 1} COMPLETE] All threads finished. Computing recall...")
                    recall = compute_recall(round_results_ordered, ground_truth, top_k, verbose=True)
                    print(f"[ROUND {round_count + 1}] Recall@{top_k}: {recall:.4f}\n")
                
                round_count += 1
                
                # Clear round results for next round (preserve the lock)
                with round_results['lock']:
                    # Remove all keys except 'lock'
                    keys_to_remove = [k for k in round_results.keys() if k != 'lock']
                    for key in keys_to_remove:
                        del round_results[key]
                
                # Reset event for next round
                round_complete_event.clear()
            
            # Update final statistics
            with stats['lock']:
                stats['query_threads_completed'] += 1
                stats['query_rounds'] = max(stats.get('query_rounds', 0), round_count)
            
            print(f"Query thread {thread_id}: completed - {query_count:,} queries in {round_count} rounds, "
                  f"total time {total_time:.4f} seconds ({query_count/total_time:.1f} queries/sec)" if total_time > 0 else f"Query thread {thread_id}: completed - {query_count:,} queries in {round_count} rounds")
    
    finally:
        conn.close()


def compute_recall(predicted_ids, ground_truth_ids, top_k=10, verbose=True):
    """Compute recall@k by comparing predicted results with ground truth
    
    Only includes queries that have non-empty results in the calculation.
    
    Args:
        predicted_ids: List of predicted ID lists
        ground_truth_ids: List of ground truth ID lists
        top_k: Top-k for recall calculation
        verbose: If True, print recall information. If False, only return value.
    """
    total_correct = 0
    queries_with_results = 0
    
    for i in range(len(predicted_ids)):
        pred = predicted_ids[i][:top_k]
        
        # Skip queries with no results
        if not pred:
            continue
        
        queries_with_results += 1
        
        if i < len(ground_truth_ids):
            gt = set(ground_truth_ids[i][:top_k])
        else:
            gt = set()
        
        correct = sum(1 for pid in pred if pid in gt)
        total_correct += correct
    
    if queries_with_results == 0:
        if verbose:
            print(f"Recall@{top_k}: N/A (no queries with results)")
        return 0.0
    
    recall = total_correct / (queries_with_results * top_k)
    if verbose:
        print(f"Recall@{top_k}: {recall:.4f} (calculated over {queries_with_results} queries with results)")
        if queries_with_results < len(predicted_ids):
            print(f"  Note: {len(predicted_ids) - queries_with_results} queries had no results and were excluded")
    return recall


def main():
    parser = argparse.ArgumentParser(description="Test concurrent insert and query operations")
    
    # Database connection
    parser.add_argument("--host", default="localhost", help="Database host")
    parser.add_argument("--port", default=5432, type=int, help="Database port")
    parser.add_argument("--user", default="postgres", help="Database user")
    parser.add_argument("--password", default="postgres", help="Database password")
    parser.add_argument("--dbname", default="postgres", help="Database name")
    parser.add_argument("--tablename", default="test_vectors", help="Table name")
    
    # Initial setup
    parser.add_argument("--do-setup", action="store_true", help="Create table and optionally insert initial vectors")
    parser.add_argument("--file", help="Path to .fvecs/.bvecs file for initial insert")
    parser.add_argument("--num-initial", type=int, default=0, help="Number of initial vectors to insert (before concurrent operations)")
    parser.add_argument("--use-bvecs", action="store_true", help="Use .bvecs format instead of .fvecs")
    
    # Index creation
    parser.add_argument("--do-index", action="store_true", help="Create vector index before concurrent operations")
    parser.add_argument("--index-type", choices=["ivfflat", "hnsw"], default="hnsw", help="Index type")
    parser.add_argument("--M", type=int, default=16, help="HNSW: number of neighbors per node")
    parser.add_argument("--ef-construction", type=int, default=64, help="HNSW: ef_construction parameter")
    parser.add_argument("--nlist", type=int, default=1000, help="IVF_FLAT: number of lists")
    
    # Concurrent insert configuration
    parser.add_argument("--do-insert", action="store_true", help="Run concurrent insert operations")
    parser.add_argument("--insert-file", help="Path to .fvecs/.bvecs file for concurrent inserts")
    parser.add_argument("--num-insert", type=int, default=1000, help="Total number of vectors to insert concurrently")
    parser.add_argument("--insert-skip", type=int, default=0, help="Skip this many vectors before starting concurrent inserts")
    parser.add_argument("--num-insert-threads", type=int, default=2, help="Number of concurrent insert threads")
    parser.add_argument("--insert-batch-size", type=int, default=100, help="Number of vectors per insert batch")
    
    # Concurrent query configuration
    parser.add_argument("--do-query", action="store_true", help="Run query operations")
    parser.add_argument("--query-file", help="Path to query .fvecs/.bvecs file")
    parser.add_argument("--num-query", type=int, default=100, help="Number of queries to run")
    parser.add_argument("--num-query-threads", type=int, default=2, help="Number of concurrent query threads")
    parser.add_argument("--query-after-insert", action="store_true", help="Run queries after insertion completes (instead of concurrently). If set, queries run once through the query set.")
    parser.add_argument("--k", type=int, default=10, help="Top-k for search")
    parser.add_argument("--nprobe", type=int, default=20, help="IVF_FLAT: number of probes")
    parser.add_argument("--ef-search", type=int, default=100, help="HNSW: ef_search parameter")
    
    # Ground truth for recall calculation
    parser.add_argument("--gnd", help="Path to groundtruth .ivecs file for calculating recall")
    
    # Timing
    parser.add_argument("--duration", type=int, default=0, help="Run concurrent operations for this many seconds (0 = run until queues are empty)")
    parser.add_argument("--wait-after-setup", type=int, default=0, help="Wait N seconds after setup before starting concurrent operations")
    
    args = parser.parse_args()
    
    if not (args.do_setup or args.do_insert or args.do_query or args.do_index):
        print("Error: Please specify at least one of --do-setup, --do-index, --do-insert, or --do-query")
        sys.exit(1)
    
    conn_params = {
        'host': args.host,
        'port': args.port,
        'user': args.user,
        'password': args.password,
        'dbname': args.dbname
    }
    
    conn = psycopg2.connect(**conn_params)
    
    try:
        # Step 1: Setup (create table, optionally insert initial vectors)
        if args.do_setup:
            print("=" * 60)
            print("STEP 1: Setup")
            print("=" * 60)
            
            if args.num_initial > 0:
                if not args.file:
                    print("Error: --file is required when --num-initial > 0")
                    sys.exit(1)
                
                print(f"Reading initial vectors from: {args.file}")
                if args.use_bvecs:
                    initial_vectors = read_bvecs(args.file, args.num_initial)
                else:
                    initial_vectors = read_fvecs(args.file, args.num_initial)
                
                if initial_vectors.size == 0:
                    print("Error: No vectors loaded")
                    sys.exit(1)
                
                print(f"Loaded {len(initial_vectors)} vectors of dimension {initial_vectors.shape[1]}")
                print("Creating table...")
                create_table(conn, initial_vectors.shape[1], args.tablename)
                print("Inserting initial vectors...")
                insert_vectors_batch(conn, initial_vectors, batch_size=100, tablename=args.tablename)
                print(f"Inserted {len(initial_vectors)} initial vectors.\n")
            else:
                # Just create empty table - need to know dimension
                if not args.file:
                    print("Error: --file is required to determine vector dimension (even if --num-initial=0)")
                    sys.exit(1)
                
                # Read first vector to get dimension
                if args.use_bvecs:
                    sample = read_bvecs(args.file, 1)
                else:
                    sample = read_fvecs(args.file, 1)
                
                if sample.size == 0:
                    print("Error: Could not read sample vector to determine dimension")
                    sys.exit(1)
                
                dim = sample.shape[1]
                print(f"Creating empty table with dimension {dim}...")
                create_table(conn, dim, args.tablename)
                print("Table created.\n")
        
        # Step 2: Create index (optional)
        if args.do_index:
            print("=" * 60)
            print("STEP 2: Create Index")
            print("=" * 60)
            
            index_params = {
                "nlist": args.nlist,
                "M": args.M,
                "ef_construction": args.ef_construction,
            }
            create_index(conn, args.index_type, index_params=index_params, tablename=args.tablename)
        
        # Wait before starting concurrent operations
        if args.wait_after_setup > 0:
            print(f"Waiting {args.wait_after_setup} seconds before starting concurrent operations...")
            time.sleep(args.wait_after_setup)
        
        # Step 3: Concurrent insert and query operations
        print("=" * 60)
        print("STEP 3: Concurrent Insert and Query Operations")
        print("=" * 60)
        
        # Statistics tracking
        stats = {
            'inserted_count': 0,
            'insert_time': 0.0,
            'insert_threads_completed': 0,
            'query_count': 0,
            'query_time': 0.0,
            'query_threads_completed': 0,
            'query_rounds': 0,
            'start_time': None,  # Will be set when operations start
            'last_insert_progress': 0,  # Last reported insert count
            'last_query_progress': 0,  # Last reported query count
            'last_progress_time': None,  # Last progress report time
            'lock': threading.Lock()
        }
        
        # Results dictionary for queries
        query_results = {
            'lock': threading.Lock()
        }
        
        # Stop flag for query threads (set when insertion is complete)
        query_stop_flag = threading.Event()
        
        insert_threads = []
        query_threads = []
        insert_queue = Queue()
        query_list = []  # List of (index, vector) tuples for queries
        query_queue = None  # Shared queue for concurrent query mode
        round_results = {'lock': threading.Lock()}  # Shared results for current round
        round_complete_event = None  # Event to signal round completion
        round_lock = None  # Lock for coordinating recall computation
        coordinator_thread = None  # Coordinator thread reference
        
        # Prepare insert queue
        if args.do_insert:
            if not args.insert_file:
                print("Error: --insert-file is required for --do-insert")
                sys.exit(1)
            
            print(f"Loading {args.num_insert} vectors for concurrent insertion from: {args.insert_file} (skip {args.insert_skip}, take {args.num_insert})")
            if args.use_bvecs:
                insert_vectors = read_bvecs(args.insert_file, args.num_insert, args.insert_skip)
            else:
                insert_vectors = read_fvecs(args.insert_file, args.num_insert, args.insert_skip)
            
            if insert_vectors.size == 0:
                print("Error: No vectors loaded for insertion")
                sys.exit(1)
            
            print(f"Loaded {len(insert_vectors)} vectors for insertion")
            
            # Split vectors into batches and add to queue with explicit IDs
            # IDs are based on file offset: start from (insert_skip + 1) since IDs are 1-based
            # This ensures IDs match the vector's position in the file, even with concurrent insertion
            batch_size = args.insert_batch_size
            base_id = args.insert_skip + 1  # IDs are 1-based, file offset is 0-based
            for i in range(0, len(insert_vectors), batch_size):
                batch = insert_vectors[i:i+batch_size]
                start_id = base_id + i
                # Put tuple of (vectors, start_id) in queue
                insert_queue.put((batch, start_id))
            
            print(f"Created {insert_queue.qsize()} insert batches")
            print(f"Vector IDs will range from {base_id} to {base_id + len(insert_vectors) - 1} (matching file offsets)")
        
        # Prepare query data structure
        query_vectors = None
        query_list = []  # For concurrent mode (repeated cycling)
        query_queue_after = Queue()  # For after-insert mode (run once)
        if args.do_query:
            if not args.query_file:
                print("Error: --query-file is required for --do-query")
                sys.exit(1)
            
            print(f"Loading {args.num_query} query vectors from: {args.query_file}")
            if args.use_bvecs:
                query_vectors = read_bvecs(args.query_file, args.num_query)
            else:
                query_vectors = read_fvecs(args.query_file, args.num_query)
            
            if query_vectors.size == 0:
                print("Error: No query vectors loaded")
                sys.exit(1)
            
            print(f"Loaded {len(query_vectors)} query vectors")
            
            if args.query_after_insert:
                # For after-insert mode: create queue for one-time execution
                for i, query_vec in enumerate(query_vectors):
                    query_queue_after.put((i, query_vec))
                print(f"Queries will run once after insertion completes ({len(query_vectors)} queries)")
            else:
                # For concurrent mode: create shared queue and query list for refilling
                query_list = [(i, query_vec) for i, query_vec in enumerate(query_vectors)]
                query_queue = Queue()  # Shared queue for all threads to collaborate on each round
                print(f"Query threads will collaborate on {len(query_list)} queries per round during insertion")
        
        # Set start time for progress reporting
        if args.do_insert or args.do_query:
            with stats['lock']:
                stats['start_time'] = time.time()
        
        # Start insert threads
        if args.do_insert:
            print(f"\nStarting {args.num_insert_threads} insert threads...")
            # Progress interval: report every 1000 vectors or every 5 seconds
            progress_interval = max(1000, args.num_insert // 100)  # Adaptive based on total
            for i in range(args.num_insert_threads):
                thread = threading.Thread(
                    target=insert_worker,
                    args=(conn_params, insert_queue, stats, args.tablename, i, progress_interval)
                )
                thread.start()
                insert_threads.append(thread)
        
        # Load ground truth if available (needed for per-round recall calculation)
        ground_truth = None
        if args.do_query and args.gnd:
            print(f"Loading ground truth from: {args.gnd}")
            ground_truth = read_ivecs(args.gnd, args.k)
        
        # Start query threads (only if running concurrently, not after insertion)
        if args.do_query and not args.query_after_insert:
            # Initialize round coordination objects
            round_complete_event = threading.Event()
            round_lock = threading.Lock()
            
            # Initialize stats for round tracking
            with stats['lock']:
                stats['last_computed_round'] = -1
            
            print(f"Starting {args.num_query_threads} query threads...")
            print(f"Search parameters: ivfflat.probes = {args.nprobe}, hnsw.ef_search = {args.ef_search}")
            print("Query threads will collaborate on queries per round until insertion completes")
            if ground_truth:
                print("Recall will be computed and printed after each complete round")
            
            # Progress interval: report every 100 queries or every 5 seconds
            progress_interval = max(100, args.num_query // 10)  # Adaptive based on query count
            
            # Start a coordinator thread to refill queue for each round
            def round_coordinator():
                """Coordinates rounds by refilling the queue and signaling completion"""
                round_num = 0
                try:
                    while not query_stop_flag.is_set():
                        # Wait for current round to complete (all queries processed and sentinels consumed)
                        # Use timeout to periodically check stop flag
                        # Wait for queue to be empty, but check stop flag periodically
                        max_wait = 300  # Maximum wait time in 0.1s increments (30 seconds)
                        wait_count = 0
                        while (not query_queue.empty() or query_queue.unfinished_tasks > 0) and wait_count < max_wait:
                            if query_stop_flag.is_set():
                                break
                            time.sleep(0.1)
                            wait_count += 1
                        
                        # If stop flag is set, signal completion and exit
                        if query_stop_flag.is_set():
                            round_complete_event.set()  # Signal threads to exit
                            break
                        
                        # Signal that round is complete (all threads will see this)
                        round_complete_event.set()
                        
                        # Small delay to allow recall computation
                        time.sleep(0.2)
                        
                        # Check stop flag again before refilling
                        if query_stop_flag.is_set():
                            break
                        
                        # Refill queue for next round
                        round_num += 1
                        round_complete_event.clear()
                        
                        if query_stop_flag.is_set():
                            break
                        
                        # Add all queries to queue for next round
                        for query_item in query_list:
                            if query_stop_flag.is_set():
                                break
                            query_queue.put(query_item)
                        
                        # Add sentinel values for each thread to signal round end
                        if not query_stop_flag.is_set():
                            for _ in range(args.num_query_threads):
                                query_queue.put(None)
                except Exception as e:
                    # If any error occurs, signal completion and exit
                    print(f"Coordinator thread error: {e}")
                    round_complete_event.set()
                finally:
                    # Ensure event is set so threads can exit
                    round_complete_event.set()
            
            # Start coordinator thread
            coordinator_thread = threading.Thread(target=round_coordinator, daemon=True)
            coordinator_thread.start()
            
            # Fill queue for first round
            for query_item in query_list:
                query_queue.put(query_item)
            # Add sentinel values for each thread
            for _ in range(args.num_query_threads):
                query_queue.put(None)
            
            # Start query worker threads
            for i in range(args.num_query_threads):
                thread = threading.Thread(
                    target=query_worker,
                    args=(conn_params, query_queue, round_results, query_results, stats, query_stop_flag,
                          round_complete_event, round_lock, args.k, args.nprobe, args.ef_search, 
                          args.tablename, i, progress_interval, ground_truth, len(query_list))
                )
                thread.start()
                query_threads.append(thread)
        
        # Run for specified duration or until insert queue is empty
        start_time = time.time()
        if args.duration > 0:
            print(f"\nRunning concurrent operations for {args.duration} seconds...")
            time.sleep(args.duration)
            
            # Signal insert threads to stop
            if args.do_insert:
                for _ in range(args.num_insert_threads):
                    insert_queue.put(None)
        else:
            print("\nRunning concurrent operations until insert queue is empty...")
            # Wait for insert queue to be processed
            if args.do_insert:
                insert_queue.join()
                # Signal insert threads to stop
                for _ in range(args.num_insert_threads):
                    insert_queue.put(None)
        
        # Wait for insert threads to complete
        if args.do_insert:
            print("Waiting for insert threads to complete...")
            for thread in insert_threads:
                thread.join()
            print("All insert threads completed")
            
            # Update sequence one final time to ensure it's synchronized
            # This is a safety measure after all concurrent inserts complete
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT setval(pg_get_serial_sequence('{args.tablename}', 'id'), 
                                  COALESCE((SELECT MAX(id) FROM {args.tablename}), 1), 
                                  true);
                """)
            conn.commit()
            print("Sequence synchronized with maximum ID")
        
        # Handle queries based on mode
        if args.do_query:
            if args.query_after_insert:
                # Run queries after insertion completes (once through the query set)
                print("\n" + "=" * 60)
                print("STEP 4: Running Queries After Insertion")
                print("=" * 60)
                print(f"Starting {args.num_query_threads} query threads...")
                print(f"Search parameters: ivfflat.probes = {args.nprobe}, hnsw.ef_search = {args.ef_search}")
                print("Queries will run once through the query set")
                
                # Set start time for query operations
                with stats['lock']:
                    stats['start_time'] = time.time()
                
                # Progress interval: report every 100 queries or every 5 seconds
                progress_interval = max(100, args.num_query // 10)  # Adaptive based on query count
                for i in range(args.num_query_threads):
                    thread = threading.Thread(
                        target=query_worker_once,
                        args=(conn_params, query_queue_after, query_results, stats, args.k, 
                              args.nprobe, args.ef_search, args.tablename, i, progress_interval)
                    )
                    thread.start()
                    query_threads.append(thread)
                
                # Wait for query queue to be processed
                query_queue_after.join()
                
                # Signal threads to stop
                for _ in range(args.num_query_threads):
                    query_queue_after.put(None)
            else:
                # Signal concurrent query threads to stop immediately (insertion is complete)
                print("Signaling query threads to stop immediately...")
                query_stop_flag.set()
                # Set the round complete event to unblock any waiting threads
                if round_complete_event is not None:
                    round_complete_event.set()
                # Clear any remaining items in queue to allow threads to exit
                if query_queue is not None:
                    # Drain the queue to allow threads to exit
                    try:
                        while not query_queue.empty():
                            try:
                                query_queue.get_nowait()
                                query_queue.task_done()
                            except:
                                break
                    except:
                        pass
        
        # Wait for query threads to complete
        if args.do_query:
            print("Waiting for query threads to complete...")
            for thread in query_threads:
                thread.join()
            print("All query threads completed")
        
        total_elapsed = time.time() - start_time
        
        # Print statistics
        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)
        print(f"Total wall-clock time: {total_elapsed:.4f} seconds")
        
        if args.do_insert:
            print(f"\nInsert Statistics:")
            print(f"  Total vectors inserted: {stats['inserted_count']}")
            print(f"  Total insert time: {stats['insert_time']:.4f} seconds")
            if stats['inserted_count'] > 0:
                print(f"  Average insert time per vector: {stats['insert_time'] / stats['inserted_count']:.6f} seconds")
                print(f"  Insert throughput: {stats['inserted_count'] / total_elapsed:.2f} vectors/second")
        
        if args.do_query:
            print(f"\nQuery Statistics:")
            if args.query_after_insert:
                print(f"  Mode: Queries run after insertion (once through query set)")
            else:
                print(f"  Mode: Queries run concurrently with insertion (repeated cycles)")
            print(f"  Total queries executed: {stats['query_count']}")
            if stats['query_rounds'] > 0:
                print(f"  Total query rounds: {stats['query_rounds']} (across all threads)")
                if query_vectors is not None and len(query_vectors) > 0:
                    print(f"  Average rounds per thread: {stats['query_rounds'] / args.num_query_threads:.1f}")
            print(f"  Total query time: {stats['query_time']:.4f} seconds")
            if stats['query_count'] > 0:
                print(f"  Average query time: {stats['query_time'] / stats['query_count']:.4f} seconds")
                print(f"  Query throughput: {stats['query_count'] / total_elapsed:.2f} queries/second")
            
            # Compute final recall if ground truth is provided
            # (For concurrent mode, recall was already computed per round)
            if args.gnd:
                if not args.query_after_insert:
                    # In concurrent mode, we already computed recall per round
                    # Just show final summary based on first round results
                    print(f"\n[FINAL RECALL SUMMARY]")
                    print(f"Based on first round results (from {len(query_results)} queries)")
                else:
                    # In after-insert mode, compute recall now
                    print(f"\nReading ground truth from: {args.gnd}")
                    if ground_truth is None:
                        ground_truth = read_ivecs(args.gnd, args.k)
                
                # Reconstruct results in order
                all_results = []
                missing_queries = []
                for i in range(len(query_vectors)):
                    if i in query_results:
                        all_results.append(query_results[i])
                    else:
                        all_results.append([])  # Query not completed
                        missing_queries.append(i)
                
                # Check if we have complete results
                if missing_queries:
                    print(f"\n⚠️  WARNING: Final recall calculation may be inaccurate!")
                    print(f"   Missing results for {len(missing_queries)} out of {len(query_vectors)} queries")
                    if len(missing_queries) <= 10:
                        print(f"   Missing query indices: {missing_queries}")
                    else:
                        print(f"   Missing query indices (first 10): {missing_queries[:10]} ...")
                    
                    if args.query_after_insert:
                        print(f"   This should not happen in after-insert mode. Check for errors.")
                    else:
                        print(f"   In concurrent mode, queries were stopped mid-cycle.")
                        print(f"   Recall was already computed per round above.")
                    
                    print(f"\n   Computing final recall using available results...")
                    print(f"   (Recall will be calculated only for queries with results)")
                else:
                    if args.query_after_insert:
                        print(f"✓ All {len(query_vectors)} queries have results - recall calculation will be accurate")
                
                # Compute recall (suppress per-round printing if in concurrent mode)
                recall = compute_recall(all_results, ground_truth, args.k, verbose=args.query_after_insert)
                if not args.query_after_insert:
                    print(f"Final Recall@{args.k}: {recall:.4f} (based on first round results)")
        
        print("\n" + "=" * 60)
        print("CONCURRENT OPERATIONS COMPLETED")
        print("=" * 60)
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()

# # Setup: create table, insert initial vectors, create index
# python concurrent_insert_query.py --do-setup --do-index \
#   --file /ssd_root/dataset/sift/sift_base.fvecs --num-initial 800000 \
#   --index-type hnsw --M 16 --ef-construction 40 \
#   --host localhost --port 5434 --user liu4127 --dbname postgres --tablename bigann_vectors

# # Create index
# python concurrent_insert_query.py --do-index \
#   --index-type hnsw --M 16 --ef-construction 40 \
#   --host localhost --port 5434 --user liu4127 --dbname postgres --tablename bigann_vectors

# # Run concurrent inserts and queries
# python concurrent_insert_query.py --do-insert --do-query \
#   --insert-file /ssd_root/dataset/sift/sift_base.fvecs --num-insert 200000 --insert-skip 800000 \
#   --num-insert-threads 1 --insert-batch-size 100 \
#   --query-file /ssd_root/dataset/sift/sift_query.fvecs --num-query 10000 \
#   --num-query-threads 1 --k 100 --ef-search 100 \
#   --gnd /ssd_root/dataset/sift/sift_groundtruth.ivecs \
#   --host localhost --port 5434 --user liu4127 --dbname postgres --tablename bigann_vectors