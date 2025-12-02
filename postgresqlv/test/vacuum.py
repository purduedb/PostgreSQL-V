import struct
import numpy as np
import psycopg2
import argparse
import time
import sys

from psycopg2 import sql


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


def insert_vectors(conn, vectors, tablename="test_vectors"):
    """Insert vectors into the table"""
    with conn.cursor() as cur:
        for vec in vectors:
            cur.execute(f"INSERT INTO {tablename} (vec) VALUES (%s);", (vec.tolist(),))
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


def delete_vectors(conn, start_id, end_id, tablename="test_vectors"):
    """Delete vectors with IDs in the range [start_id, end_id]"""
    with conn.cursor() as cur:
        print(f"Deleting vectors with IDs from {start_id} to {end_id}...")
        cur.execute(f"DELETE FROM {tablename} WHERE id >= %s AND id <= %s;", (start_id, end_id))
        deleted_count = cur.rowcount
        conn.commit()
        print(f"Deleted {deleted_count} vectors.\n")
        return deleted_count


def delete_vectors_multiple_of(conn, n, tablename="test_vectors"):
    """Delete vectors where id is a multiple of n (id % n == 0)"""
    with conn.cursor() as cur:
        print(f"Deleting vectors where id is a multiple of {n}...")
        # Get all IDs that are multiples of n before deletion
        # Use regular string formatting to avoid f-string issues with % operator
        cur.execute("SELECT id FROM {} WHERE id %% %s = 0 ORDER BY id;".format(tablename), (n,))
        deleted_ids = [row[0] for row in cur.fetchall()]
        
        # Delete the vectors
        cur.execute("DELETE FROM {} WHERE id %% %s = 0;".format(tablename), (n,))
        deleted_count = cur.rowcount
        conn.commit()
        print(f"Deleted {deleted_count} vectors (IDs that are multiples of {n}).\n")
        return deleted_ids, deleted_count


def vacuum_table(conn, table_name):
    """Run VACUUM on the table"""
    print(f"Running VACUUM on {table_name}...")

    # Temporarily enable autocommit
    original_autocommit = conn.autocommit
    conn.autocommit = True

    with conn.cursor() as cur:
        cur.execute(f"VACUUM {table_name};")

    # Restore original autocommit setting
    conn.autocommit = original_autocommit
    print("VACUUM completed (bulkdelete should be triggered).\n")


def search_queries(conn, queries, top_k=10, nprobe=20, ef_search=400, tablename="test_vectors"):
    """Run similarity search queries and return results (zero-based IDs)"""
    with conn.cursor() as cur:
        cur.execute(f"SET ivfflat.probes = {nprobe};")
        cur.execute(f"SET hnsw.ef_search = {ef_search};")
        print(f"Search parameters: ivfflat.probes = {nprobe}, hnsw.ef_search = {ef_search}")
        all_results = []
        for q in queries:
            cur.execute(f"""
                SELECT id
                FROM {tablename}
                ORDER BY vec <-> %s::vector
                LIMIT %s;
            """, (q.tolist(), top_k))
            # Subtract 1 from each ID to get zero-based indexing (matching ground truth format)
            ids = [row[0] - 1 for row in cur.fetchall()]
            all_results.append(ids)
        return all_results


def compute_exact_ground_truth(conn, queries, top_k=10, tablename="test_vectors"):
    """Compute exact ground truth by running sequential scan (no index) - returns zero-based IDs"""
    print("Computing exact ground truth (sequential scan)...")
    with conn.cursor() as cur:
        # Disable index scan to force sequential scan for exact results
        cur.execute("SET enable_indexscan = off;")
        all_results = []
        for q in queries:
            cur.execute(f"""
                SELECT id
                FROM {tablename}
                ORDER BY vec <-> %s::vector
                LIMIT %s;
            """, (q.tolist(), top_k))
            # Subtract 1 from each ID to get zero-based indexing (matching ground truth format)
            ids = [row[0] - 1 for row in cur.fetchall()]
            all_results.append(ids)
        # Re-enable index scan
        cur.execute("SET enable_indexscan = on;")
    print(f"Computed ground truth for {len(all_results)} queries")
    return all_results


def compute_recall(predicted_ids, ground_truth_ids, top_k=10):
    """Compute recall@k by comparing predicted results with ground truth"""
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
    return recall


def verify_search_results(search_results, deleted_ids, test_name="Search"):
    """Verify that deleted vector IDs are not in search results"""
    deleted_set = set(deleted_ids)
    violations = []
    
    for query_idx, result_ids in enumerate(search_results):
        found_deleted = [vid for vid in result_ids if vid in deleted_set]
        if found_deleted:
            violations.append((query_idx, found_deleted))
    
    if violations:
        print(f"❌ {test_name} FAILED: Found deleted vectors in search results!")
        for query_idx, deleted_found in violations[:5]:  # Show first 5 violations
            print(f"  Query {query_idx}: Found deleted IDs {deleted_found}")
        if len(violations) > 5:
            print(f"  ... and {len(violations) - 5} more violations")
        return False
    else:
        print(f"✅ {test_name} PASSED: No deleted vectors found in search results")
        return True


def get_all_vector_ids(conn, tablename="test_vectors"):
    """Get all vector IDs currently in the table"""
    with conn.cursor() as cur:
        cur.execute(f"SELECT id FROM {tablename} ORDER BY id;")
        return [row[0] for row in cur.fetchall()]


def main():
    parser = argparse.ArgumentParser(description="Test index vacuum functionality")
    # Database connection
    parser.add_argument("--host", default="localhost", help="Database host")
    parser.add_argument("--port", default=5432, type=int, help="Database port")
    parser.add_argument("--user", default="postgres", help="Database user")
    parser.add_argument("--password", default="postgres", help="Database password")
    parser.add_argument("--dbname", default="postgres", help="Database name")
    parser.add_argument("--tablename", default="test_vectors", help="Table name")
    
    # Round 1 insertion
    parser.add_argument("--skip-insert1", action="store_true", help="Skip round 1 insertion (vectors already in table)")
    parser.add_argument("--file", help="Path to .fvecs file for round-1 insert (required unless --skip-insert1)")
    parser.add_argument("--num1", type=int, default=1000, help="Number of vectors in round 1 (for ID range inference)")
    
    # Round 2 insertion (optional)
    parser.add_argument("--do-insert2", action="store_true", help="Insert vectors in round 2")
    parser.add_argument("--file2", help="Path to .fvecs file for round-2 insert (defaults to --file)")
    parser.add_argument("--num2", type=int, default=0, help="Number of vectors to insert in round 2")
    parser.add_argument("--skip2", type=int, default=0, help="Skip this many vectors before round 2 insert")
    
    # Index creation
    parser.add_argument("--do-index", action="store_true", help="Create vector index")
    parser.add_argument("--index-type", choices=["ivfflat", "hnsw"], default="hnsw", help="Index type")
    parser.add_argument("--M", type=int, default=16, help="HNSW: number of neighbors per node")
    parser.add_argument("--ef-construction", type=int, default=64, help="HNSW: ef_construction parameter")
    parser.add_argument("--nlist", type=int, default=1000, help="IVF_FLAT: number of lists")
    
    # Deletion
    parser.add_argument("--delete-start", type=int, help="Start ID for deletion (inclusive)")
    parser.add_argument("--delete-end", type=int, help="End ID for deletion (inclusive)")
    parser.add_argument("--delete-round1", action="store_true", help="Delete all vectors from round 1")
    parser.add_argument("--delete-round2", action="store_true", help="Delete all vectors from round 2")
    parser.add_argument("--delete-multiple-of", type=int, help="Delete vectors where id is a multiple of n (id %% n == 0)")
    
    # Search and verification
    parser.add_argument("--queries", help="Path to query .fvecs file")
    parser.add_argument("--gnd", help="Path to groundtruth .ivecs file for calculating recall (optional, will compute if not provided)")
    parser.add_argument("--nq", type=int, default=10, help="Number of queries to run")
    parser.add_argument("--k", type=int, default=10, help="Top-k for search")
    parser.add_argument("--nprobe", type=int, default=20, help="IVF_FLAT: number of probes")
    parser.add_argument("--ef-search", type=int, default=100, help="HNSW: ef_search parameter")
    
    args = parser.parse_args()

    # Validate that only one delete pattern is specified
    delete_patterns = []
    if args.delete_round1:
        delete_patterns.append("--delete-round1")
    if args.delete_round2:
        delete_patterns.append("--delete-round2")
    if args.delete_start is not None or args.delete_end is not None:
        delete_patterns.append("--delete-start/--delete-end")
    if args.delete_multiple_of is not None:
        delete_patterns.append("--delete-multiple-of")
    
    if len(delete_patterns) > 1:
        print("Error: Only one delete pattern can be specified at a time.")
        print(f"Found {len(delete_patterns)} delete patterns: {', '.join(delete_patterns)}")
        sys.exit(1)
    
    if args.delete_start is not None and args.delete_end is None:
        print("Error: --delete-end must be specified when using --delete-start")
        sys.exit(1)
    
    if args.delete_end is not None and args.delete_start is None:
        print("Error: --delete-start must be specified when using --delete-end")
        sys.exit(1)

    # Connect to database
    conn = psycopg2.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.dbname
    )

    try:
        # Step 1: First round insertion
        if args.skip_insert1:
            print("STEP 1: First Round Insertion (SKIPPED - vectors already in table)")
            # Infer the IDs of round 1 vectors (assuming they start at 1)
            round1_start_id = 1
            round1_end_id = args.num1
            print(f"Assuming round 1 vectors have IDs: {round1_start_id} to {round1_end_id}\n")
        else:
            print("STEP 1: First Round Insertion")
            
            if not args.file:
                print("Error: --file is required for round 1 insertion (or use --skip-insert1)")
                sys.exit(1)
            
            print(f"Reading vectors from: {args.file}")
            round1_vectors = read_fvecs(args.file, args.num1)
            if round1_vectors.size == 0:
                print("Error: No vectors loaded for round 1")
                sys.exit(1)
            
            print(f"Loaded {len(round1_vectors)} vectors of dimension {round1_vectors.shape[1]}")
            print("Creating table...")
            create_table(conn, round1_vectors.shape[1], args.tablename)
            print("Inserting vectors (round 1)...")
            insert_vectors(conn, round1_vectors, args.tablename)
            
            # Infer the IDs of round 1 vectors (SERIAL starts at 1)
            round1_start_id = 1
            round1_end_id = args.num1
            print(f"Round 1 vectors have IDs: {round1_start_id} to {round1_end_id}\n")
        
        # Step 2: Create index (optional, but recommended)
        if args.do_index:
            print("STEP 2: Create Index")
            index_params = {
                "nlist": args.nlist,
                "M": args.M,
                "ef_construction": args.ef_construction,
            }
            create_index(conn, args.index_type, index_params=index_params, tablename=args.tablename)


        # Step 3: Second round insertion (optional)
        round2_start_id = None
        round2_end_id = None
        
        if args.do_insert2:
            print("STEP 3: Second Round Insertion")
            
            if args.num2 <= 0:
                print("Error: --num2 must be > 0 when using --do-insert2")
                sys.exit(1)
            
            src_file = args.file2 if args.file2 else args.file
            if not src_file:
                print("Error: --file2 (or --file) is required for --do-insert2")
                sys.exit(1)
            
            print(f"Reading vectors from: {src_file} (skip {args.skip2}, take {args.num2})")
            round2_vectors = read_fvecs(src_file, num_vectors=args.num2, skip=args.skip2)
            if round2_vectors.size == 0:
                print("Warning: No vectors loaded for round 2")
            else:
                print(f"Loaded {len(round2_vectors)} vectors for round 2")
                print("Inserting vectors (round 2)...")
                insert_vectors(conn, round2_vectors, args.tablename)
                
                # Infer the IDs of round 2 vectors (continuous after round 1)
                round2_start_id = round1_end_id + 1
                round2_end_id = round1_end_id + args.num2
                print(f"Round 2 vectors have IDs: {round2_start_id} to {round2_end_id}\n")

        # Step 4: Delete vectors
        print("STEP 4: Delete Vectors")
        
        deleted_ids = []
        
        if args.delete_round1:
            deleted_ids.extend(range(round1_start_id, round1_end_id + 1))
            delete_vectors(conn, round1_start_id, round1_end_id, args.tablename)
        
        if args.delete_round2 and round2_start_id is not None:
            deleted_ids.extend(range(round2_start_id, round2_end_id + 1))
            delete_vectors(conn, round2_start_id, round2_end_id, args.tablename)
        
        if args.delete_start is not None and args.delete_end is not None:
            deleted_ids.extend(range(args.delete_start, args.delete_end + 1))
            delete_vectors(conn, args.delete_start, args.delete_end, args.tablename)
        
        if args.delete_multiple_of is not None:
            if args.delete_multiple_of <= 0:
                print("Error: --delete-multiple-of must be a positive integer")
                sys.exit(1)
            multiple_deleted_ids, _ = delete_vectors_multiple_of(conn, args.delete_multiple_of, args.tablename)
            deleted_ids.extend(multiple_deleted_ids)
        
        if not deleted_ids:
            print("Warning: No vectors were deleted. Specify --delete-start/--delete-end, --delete-round1/--delete-round2, or --delete-multiple-of")
        else:
            print(f"Total deleted vector IDs: {len(deleted_ids)}")

        # Step 5: Run VACUUM
        print("STEP 5: Run VACUUM")
        vacuum_table(conn, args.tablename)
        
        # Wait 10 seconds after vacuum before running queries
        print("Waiting 10 seconds after vacuum before running queries...")
        time.sleep(10)

        # Step 6: Conduct similarity search and verify correctness
        print("STEP 6: Similarity Search and Verification")
        
        if not args.queries:
            print("Warning: --queries not provided, skipping search verification")
        else:
            print(f"Reading query vectors from: {args.queries}")
            queries = read_fvecs(args.queries, args.nq)
            print(f"Loaded {len(queries)} query vectors")
            
            # Run search
            print("\nRunning similarity search...")
            search_results = search_queries(
                conn, queries, args.k, args.nprobe, args.ef_search, args.tablename
            )
            
            # Get ground truth: from file if provided, otherwise compute exact
            if args.gnd:
                print(f"Reading ground truth from: {args.gnd}")
                ground_truth = read_ivecs(args.gnd, args.k)
            else:
                # Compute exact ground truth by sequential scan
                ground_truth = compute_exact_ground_truth(conn, queries, args.k, args.tablename)
            
            # Compute recall
            recall = compute_recall(search_results, ground_truth, args.k)
            
            # # Verify results
            # if deleted_ids:
            #     success = verify_search_results(search_results, deleted_ids, "Vacuum Test")
            #     if not success:
            #         sys.exit(1)
            # else:
            #     print("No vectors were deleted, skipping verification")
            
            # # Additional verification: check that all returned IDs exist in the table
            # all_existing_ids = set(get_all_vector_ids(conn, args.tablename))
            # all_valid = True
            # for query_idx, result_ids in enumerate(search_results):
            #     invalid_ids = [vid for vid in result_ids if vid not in all_existing_ids]
            #     if invalid_ids:
            #         print(f"❌ Query {query_idx}: Found non-existent IDs {invalid_ids}")
            #         all_valid = False
            
            # if all_valid:
            #     print("✅ All returned IDs exist in the table")
        
        print("TEST COMPLETED SUCCESSFULLY")

    finally:
        conn.close()


if __name__ == "__main__":
    main()


  
# hnsw build and query sift1M
# python vacuum.py --skip-insert1 --do-index --do-insert2 \
#   --file /ssd_root/dataset/sift/sift_base.fvecs \
#   --queries /ssd_root/dataset/sift/sift_query.fvecs \
#   --num1 800000 --num2 200000 --skip2 800000 --nq 10000 --k 100 \
#   --index-type hnsw --M 16 --ef-construction 40 \
#   --delete-multiple-of 3 \
#   --host localhost --port 5434 --user liu4127 --dbname postgres --tablename bigann_vectors