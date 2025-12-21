# #!/usr/bin/env python3
# """
# Test script for runbook execution.

# This script reads a runbook YAML file, executes the operations (insert, delete, search),
# and compares search results with ground truth files.
# """

# import struct
# import numpy as np
# import psycopg2
# import yaml
# import argparse
# import os
# import sys
# import time
# from pathlib import Path


# def read_usbin(file_path):
#     """
#     Read ground truth file in usbin format.
    
#     Format:
#     [Header - 8 bytes]
#     ├── uint32 n_queries    (4 bytes) - Number of query vectors
#     └── uint32 k            (4 bytes) - Number of nearest neighbors (typically 100)
    
#     [Data - n_queries * k * 8 bytes]
#     ├── int32[n_queries, k]  IDs      - Database vector indices (4 bytes each)
#     └── float32[n_queries, k] Distances - Corresponding distances (4 bytes each)
#     """
#     with open(file_path, "rb") as f:
#         # Read header
#         header = f.read(8)
#         if len(header) < 8:
#             raise ValueError(f"Invalid usbin file: header too short in {file_path}")
        
#         n_queries, k = struct.unpack("II", header)
        
#         # Read IDs (int32)
#         id_data = f.read(n_queries * k * 4)
#         ids = np.frombuffer(id_data, dtype=np.int32).reshape(n_queries, k)
        
#         # Read distances (float32)
#         dist_data = f.read(n_queries * k * 4)
#         distances = np.frombuffer(dist_data, dtype=np.float32).reshape(n_queries, k)
    
#     return ids.tolist(), distances.tolist()


# def read_fvecs(file_path, num_vectors=None, skip=0):
#     """Read vectors from .fvecs file format"""
#     with open(file_path, "rb") as f:
#         vectors = []
#         # skip entries
#         for _ in range(skip):
#             len_bytes = f.read(4)
#             if not len_bytes:
#                 break
#             dim = struct.unpack("i", len_bytes)[0]
#             f.seek(4 * dim, 1)  # skip dim float32s
#         # read entries
#         while True:
#             len_bytes = f.read(4)
#             if not len_bytes:
#                 break
#             dim = struct.unpack("i", len_bytes)[0]
#             vec = np.frombuffer(f.read(4 * dim), dtype=np.float32)
#             vectors.append(vec)
#             if num_vectors and len(vectors) >= num_vectors:
#                 break
#     if not vectors:
#         return np.empty((0, 0), dtype=np.float32)
#     return np.vstack(vectors)


# def read_bvecs(file_path, num_vectors=None, skip=0):
#     """Read vectors from .bvecs file format"""
#     with open(file_path, "rb") as f:
#         vectors = []
#         # skip entries
#         for _ in range(skip):
#             len_bytes = f.read(4)
#             if not len_bytes:
#                 break
#             dim = struct.unpack("i", len_bytes)[0]
#             f.seek(dim, 1)  # skip dim bytes (uint8)
#         # read entries
#         while True:
#             len_bytes = f.read(4)
#             if not len_bytes:
#                 break
#             dim = struct.unpack("i", len_bytes)[0]
#             vec = np.frombuffer(f.read(dim), dtype=np.uint8)
#             vectors.append(vec)
#             if num_vectors and len(vectors) >= num_vectors:
#                 break
#     if not vectors:
#         return np.empty((0, 0), dtype=np.uint8)
#     return np.vstack(vectors)


# def read_fbin(file_path, num_vectors=None):
#     """
#     Read vectors from .fbin file format.
    
#     Format:
#     - Header: int32 n (number of vectors), int32 d (dimension)
#     - Data: n * d float32 values
#     """
#     with open(file_path, "rb") as f:
#         # Read number of vectors and dimension
#         header = np.fromfile(f, dtype=np.int32, count=2)
#         if len(header) < 2:
#             raise ValueError("Invalid fbin file: header too short")
#         n, d = header
#         print(f"  File contains {n} vectors of dimension {d}")
        
#         # If num_vectors is specified, limit the read
#         count = n if num_vectors is None else min(num_vectors, n)
        
#         # Read vectors as float32
#         data = np.fromfile(f, dtype=np.float32, count=count * d)
#         vectors = data.reshape(count, d)
        
#     return vectors


# def create_table(conn, dim, tablename="test_vectors", drop_if_exists=True):
#     """Create a table for storing vectors"""
#     with conn.cursor() as cur:
#         cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
#         if drop_if_exists:
#             cur.execute(f"DROP TABLE IF EXISTS {tablename};")
#         cur.execute(f"""
#             CREATE TABLE IF NOT EXISTS {tablename} (
#                 id SERIAL PRIMARY KEY,
#                 vec vector({dim})
#             );
#         """)
#         conn.commit()


# def insert_vectors_range(conn, vectors, start_idx, end_idx, tablename="test_vectors", start_id=None):
#     """
#     Insert vectors from start_idx to end_idx (exclusive) into the database.
    
#     Args:
#         conn: Database connection
#         vectors: Array of all vectors
#         start_idx: Starting index in vectors array (0-based, inclusive)
#         end_idx: Ending index in vectors array (0-based, exclusive, like Python ranges)
#         tablename: Name of the table
#         start_id: Starting ID for the first vector (1-based). If None, uses auto-increment.
#     """
#     with conn.cursor() as cur:
#         batch_vectors = vectors[start_idx:end_idx]
#         if len(batch_vectors) == 0:
#             return
        
#         # Insert with explicit IDs if start_id is provided
#         if start_id is not None:
#             values = [(start_id + i, vec.tolist()) for i, vec in enumerate(batch_vectors)]
#             from psycopg2.extras import execute_values
#             execute_values(
#                 cur,
#                 f"INSERT INTO {tablename} (id, vec) VALUES %s;",
#                 values
#             )
#             max_id = start_id + len(batch_vectors) - 1
#             # Update the sequence
#             cur.execute(f"""
#                 SELECT setval(pg_get_serial_sequence('{tablename}', 'id'), 
#                               GREATEST(COALESCE((SELECT MAX(id) FROM {tablename}), 0), {max_id}), 
#                               true);
#             """)
#         else:
#             # Insert without IDs (auto-increment)
#             from psycopg2.extras import execute_values
#             execute_values(
#                 cur,
#                 f"INSERT INTO {tablename} (vec) VALUES %s;",
#                 [(vec.tolist(),) for vec in batch_vectors]
#             )
#         conn.commit()


# def delete_vectors_range(conn, start_id, end_id, tablename="test_vectors"):
#     """Delete vectors with IDs in the range [start_id, end_id] (inclusive)"""
#     with conn.cursor() as cur:
#         cur.execute(f"DELETE FROM {tablename} WHERE id >= %s AND id <= %s;", (start_id, end_id))
#         deleted_count = cur.rowcount
#         conn.commit()
#         return deleted_count


# def create_index(conn, index_type, index_params, tablename="test_vectors", index_name="test_vector_idx"):
#     """
#     Create an index on the vector column.
    
#     Args:
#         conn: Database connection
#         index_type: Type of index ("hnsw" or "ivfflat")
#         index_params: Dictionary with index parameters
#             - For HNSW: M, ef_construction
#             - For IVFFlat: nlist
#         tablename: Name of the table
#         index_name: Name of the index to create
#     """
#     with conn.cursor() as cur:
#         print(f"Checking for existing index and dropping if it exists...")
#         cur.execute(f"DROP INDEX IF EXISTS {index_name};")
        
#         if index_type == "ivfflat":
#             nlist = index_params.get("nlist", 100)
#             print(f"Creating IVF_FLAT index with nlist = {nlist}...")
#             start_time = time.time()
#             index_sql = f"""
#                 CREATE INDEX {index_name} ON {tablename}
#                 USING ivfflat (vec vector_l2_ops)
#                 WITH (lists = %s);
#             """
#             cur.execute(index_sql, (nlist,))
#             elapsed_time = time.time() - start_time
#             print(f"IVFFLAT Index creation done in {elapsed_time:.4f} seconds.\n")
        
#         elif index_type == "hnsw":
#             m = index_params.get("M", 32)
#             ef_construction = index_params.get("ef_construction", 64)
#             print(f"Creating HNSW index with M = {m}, ef_construction = {ef_construction}...")
#             start_time = time.time()
#             index_sql = f"""
#                 CREATE INDEX {index_name} ON {tablename}
#                 USING hnsw (vec vector_l2_ops)
#                 WITH (M = %s, ef_construction = %s);
#             """
#             cur.execute(index_sql, (m, ef_construction))
#             elapsed_time = time.time() - start_time
#             print(f"HNSW Index creation done in {elapsed_time:.4f} seconds.\n")
        
#         else:
#             raise ValueError(f"Unsupported index type: {index_type}")
        
#         conn.commit()
#         print("Index creation complete.\n")


# def search_queries(conn, queries, top_k=100, ef_search=400, nprobe=20, tablename="test_vectors"):
#     """
#     Search for nearest neighbors of query vectors.
    
#     Returns:
#         List of lists, where each inner list contains the IDs (0-based) of the top_k nearest neighbors
#     """
#     with conn.cursor() as cur:
#         cur.execute(f"SET ivfflat.probes = {nprobe};")
#         cur.execute(f"SET hnsw.ef_search = {ef_search};")
        
#         all_results = []
#         for q in queries[:100]:
#             cur.execute(f"""
#                 SELECT id
#                 FROM {tablename}
#                 ORDER BY vec <-> %s::vector
#                 LIMIT %s;
#             """, (q.tolist(), top_k))
#             # Subtract 1 from each ID to get zero-based indexing
#             ids = [row[0] - 1 for row in cur.fetchall()]
#             all_results.append(ids)
        
#         return all_results


# def compute_recall(predicted_ids, ground_truth_ids, top_k=100):
#     """
#     Compute recall@top_k.
    
#     Args:
#         predicted_ids: List of lists of predicted IDs (0-based)
#         ground_truth_ids: List of lists of ground truth IDs (0-based)
#         top_k: Number of top results to consider
    
#     Returns:
#         Recall value (0.0 to 1.0)
#     """
#     total_correct = 0
#     total_expected = 0
    
#     for i in range(len(predicted_ids)):
#         pred = set(predicted_ids[i][:top_k])
#         if i < len(ground_truth_ids):
#             gt = set(ground_truth_ids[i][:top_k])
#         else:
#             gt = set()
        
#         correct = len(pred & gt)
#         total_correct += correct
#         total_expected += len(gt)
    
#     if total_expected == 0:
#         return 0.0
    
#     recall = total_correct / total_expected
#     return recall


# def load_runbook(runbook_path):
#     """Load and parse the runbook YAML file"""
#     with open(runbook_path, 'r') as f:
#         runbook = yaml.safe_load(f)
#     return runbook


# def get_ground_truth_path(gt_dir, step_num):
#     """
#     Get the path to the ground truth file for a given step.
    
#     Args:
#         gt_dir: Directory containing ground truth files
#         step_num: Step number
    
#     Returns:
#         Path to the ground truth file (step{step_num}.gt100)
#     """
#     gt_file = os.path.join(gt_dir, f"step{step_num}.gt100")
#     return gt_file


# def main():
#     parser = argparse.ArgumentParser(description="Test runbook execution")
#     parser.add_argument("--runbook", required=True, help="Path to runbook YAML file")
#     parser.add_argument("--base-vectors", required=True, help="Path to base vectors file (.fvecs or .bvecs)")
#     parser.add_argument("--query-vectors", required=True, help="Path to query vectors file (.fvecs or .bvecs)")
#     parser.add_argument("--host", default="localhost", help="PostgreSQL host")
#     parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port")
#     parser.add_argument("--user", default="postgres", help="PostgreSQL user")
#     parser.add_argument("--password", default="postgres", help="PostgreSQL password")
#     parser.add_argument("--dbname", default="postgres", help="PostgreSQL database name")
#     parser.add_argument("--tablename", default="test_vectors", help="Table name")
#     parser.add_argument("--k", type=int, default=100, help="Number of nearest neighbors to retrieve")
#     parser.add_argument("--ef-search", type=int, default=400, help="HNSW ef_search parameter")
#     parser.add_argument("--nprobe", type=int, default=20, help="IVFFlat nprobe parameter")
#     parser.add_argument("--vector-format", choices=["fvecs", "bvecs", "fbin"], default="fvecs",
#                         help="Format of vector files")
#     parser.add_argument("--query-format", choices=["fvecs", "bvecs", "fbin"], default=None,
#                         help="Format of query vector file (defaults to --vector-format if not specified)")
#     parser.add_argument("--gt-dir", default=None,
#                         help="Directory containing ground truth files. If not specified, "
#                              "defaults to {runbook_name}_gt in the same directory as the runbook")
#     parser.add_argument("--build-index-before-step", type=int, default=None,
#                         help="Build index before this step number starts")
#     parser.add_argument("--index-type", choices=["hnsw", "ivfflat"], default="hnsw",
#                         help="Type of index to build")
#     parser.add_argument("--index-name", default="test_vector_idx",
#                         help="Name of the index to create")
#     parser.add_argument("--M", type=int, default=32,
#                         help="HNSW: number of neighbors per node")
#     parser.add_argument("--ef-construction", type=int, default=64,
#                         help="HNSW: size of dynamic list for graph construction")
#     parser.add_argument("--nlist", type=int, default=1000,
#                         help="IVFFlat: number of lists")
#     parser.add_argument("--start-step", type=int, default=None,
#                         help="First step to process (inclusive). If not specified, starts from the first step")
#     parser.add_argument("--end-step", type=int, default=None,
#                         help="Last step to process (inclusive). If not specified, processes until the last step")
#     parser.add_argument("--drop-table", action="store_true", default=False,
#                         help="Drop and recreate the table at the start. By default, the table is not dropped.")
#     args = parser.parse_args()
    
#     # Load runbook
#     print(f"Loading runbook from {args.runbook}...")
#     runbook_data = load_runbook(args.runbook)
    
#     # Get the first (and typically only) dataset name
#     dataset_name = list(runbook_data.keys())[0]
#     operations = runbook_data[dataset_name]
    
#     print(f"Dataset: {dataset_name}")
#     print(f"Number of operations: {len(operations)}")
    
#     # Determine ground truth directory
#     if args.gt_dir:
#         gt_dir = args.gt_dir
#     else:
#         # Default: infer from runbook name
#         runbook_dir = os.path.dirname(args.runbook)
#         runbook_name = os.path.splitext(os.path.basename(args.runbook))[0]
#         gt_dir = os.path.join(runbook_dir, f"{runbook_name}_gt")
    
#     print(f"Ground truth directory: {gt_dir}")
    
#     # Auto-detect format from file extension if not explicitly specified
#     def detect_format(file_path, default_format):
#         """Detect file format from extension or filename pattern"""
#         file_lower = file_path.lower()
#         # Check for exact extension matches first
#         if file_lower.endswith('.fbin') or '.fbin' in file_lower:
#             return 'fbin'
#         elif file_lower.endswith('.fvecs') or '.fvecs' in file_lower:
#             return 'fvecs'
#         elif file_lower.endswith('.bvecs') or '.bvecs' in file_lower:
#             return 'bvecs'
#         return default_format
    
#     # Load base vectors
#     print(f"\nLoading base vectors from {args.base_vectors}...")
#     base_format = detect_format(args.base_vectors, args.vector_format)
#     if base_format != args.vector_format:
#         print(f"  Auto-detected format: {base_format} (from file extension)")
#     if base_format == "fvecs":
#         base_vectors = read_fvecs(args.base_vectors)
#     elif base_format == "bvecs":
#         base_vectors = read_bvecs(args.base_vectors)
#     elif base_format == "fbin":
#         base_vectors = read_fbin(args.base_vectors)
#     else:
#         raise ValueError(f"Unsupported base vector format: {base_format}")
#     print(f"Loaded {len(base_vectors)} base vectors of dimension {base_vectors.shape[1]}")
    
#     # Load query vectors
#     print(f"\nLoading query vectors from {args.query_vectors}...")
#     query_format_default = args.query_format if args.query_format is not None else args.vector_format
#     query_format = detect_format(args.query_vectors, query_format_default)
#     if query_format != query_format_default:
#         print(f"  Auto-detected format: {query_format} (from file extension)")
#     if query_format == "fvecs":
#         query_vectors = read_fvecs(args.query_vectors)
#     elif query_format == "bvecs":
#         query_vectors = read_bvecs(args.query_vectors)
#     elif query_format == "fbin":
#         query_vectors = read_fbin(args.query_vectors)
#     else:
#         raise ValueError(f"Unsupported query format: {query_format}")
#     print(f"Loaded {len(query_vectors)} query vectors of dimension {query_vectors.shape[1]}")
    
#     # Connect to database
#     print(f"\nConnecting to database...")
#     conn = psycopg2.connect(
#         host=args.host,
#         port=args.port,
#         user=args.user,
#         password=args.password,
#         dbname=args.dbname
#     )
    
#     # Create table
#     if args.drop_table:
#         print(f"\nDropping and creating table {args.tablename}...")
#     else:
#         print(f"\nCreating table {args.tablename} (if not exists)...")
#     create_table(conn, base_vectors.shape[1], args.tablename, drop_if_exists=args.drop_table)
    
#     # Track current max ID for inserts
#     current_max_id = 0
    
#     # Prepare index parameters if index building is requested
#     index_params = None
#     if args.build_index_before_step is not None:
#         index_params = {
#             "M": args.M,
#             "ef_construction": args.ef_construction,
#             "nlist": args.nlist,
#         }
#         print(f"\nIndex will be built before step {args.build_index_before_step}")
#         print(f"Index type: {args.index_type}")
#         if args.index_type == "hnsw":
#             print(f"  M: {args.M}, ef_construction: {args.ef_construction}")
#         else:
#             print(f"  nlist: {args.nlist}")
    
#     # Execute operations
#     print(f"\nExecuting operations...")
#     search_step_count = 0
#     index_built = False
    
#     # Filter out non-numeric keys (like 'max_pts') and sort by integer value
#     # YAML may load numeric keys as integers or strings, so we need to handle both
#     step_numbers = []
#     for key in operations.keys():
#         if isinstance(key, int):
#             step_numbers.append(key)
#         elif isinstance(key, str):
#             try:
#                 step_numbers.append(int(key))
#             except ValueError:
#                 # Skip non-numeric keys like 'max_pts'
#                 continue
#         else:
#             # Skip other types
#             continue
    
#     # Sort step numbers
#     step_numbers = sorted(step_numbers)
    
#     # Filter by step range if specified
#     if args.start_step is not None:
#         step_numbers = [s for s in step_numbers if s >= args.start_step]
#         print(f"Filtered to steps >= {args.start_step}: {len(step_numbers)} steps remaining")
    
#     if args.end_step is not None:
#         step_numbers = [s for s in step_numbers if s <= args.end_step]
#         print(f"Filtered to steps <= {args.end_step}: {len(step_numbers)} steps remaining")
    
#     if len(step_numbers) == 0:
#         print("Error: No steps to process after filtering")
#         sys.exit(1)
    
#     print(f"Processing {len(step_numbers)} steps: {step_numbers[0]} to {step_numbers[-1]}")
    
#     for step_num in step_numbers:
#         # Build index before the specified step if requested
#         if (args.build_index_before_step is not None and 
#             step_num == args.build_index_before_step and 
#             not index_built):
#             print(f"\n--- Building index before step {step_num} ---")
#             create_index(conn, args.index_type, index_params, args.tablename, args.index_name)
#             index_built = True
        
#         # Try both integer and string key (YAML may load keys as either)
#         op = operations.get(step_num) or operations.get(str(step_num))
#         if op is None:
#             print(f"  Warning: Could not find operation for step {step_num}")
#             continue
#         op_type = op.get("operation")
        
#         print(f"\n--- Step {step_num}: {op_type} ---")
        
#         if op_type == "insert":
#             start = op.get("start")
#             end = op.get("end")
#             if start is None or end is None:
#                 print(f"  Warning: Missing start or end for insert operation")
#                 continue
            
#             if start < 0 or end < 0 or start >= len(base_vectors) or end > len(base_vectors):
#                 print(f"  Warning: Indices out of range (start={start}, end={end} (exclusive), max={len(base_vectors)})")
#                 continue
            
#             if start >= end:
#                 print(f"  Warning: Invalid range (start={start} >= end={end})")
#                 continue
            
#             # Insert vectors with 1-based IDs
#             # end is exclusive, so we insert vectors [start, end)
#             num_vectors = end - start
#             insert_start_id = start + 1
#             insert_end_id = end  # end is exclusive, so last ID is end (not end+1)
#             print(f"  Inserting vectors {start} to {end} (exclusive, IDs {insert_start_id} to {insert_end_id})...")
#             insert_vectors_range(conn, base_vectors, start, end, args.tablename, insert_start_id)
#             current_max_id = max(current_max_id, insert_end_id)
#             print(f"  Inserted {num_vectors} vectors")
        
#         elif op_type == "delete":
#             start = op.get("start")
#             end = op.get("end")
#             if start is None or end is None:
#                 print(f"  Warning: Missing start or end for delete operation")
#                 continue
            
#             if start > end:
#                 print(f"  Warning: Invalid range (start={start} > end={end})")
#                 continue
            
#             # Delete vectors with 1-based IDs
#             # end is exclusive, so we delete IDs [start+1, end) which is [start+1, end-1] inclusive
#             # But since end is exclusive, the last ID to delete is 'end' (not end+1)
#             # Example: start=0, end=50000 (exclusive) means delete IDs 1 to 50000
#             delete_start_id = start + 1
#             delete_end_id = end  # end is exclusive, so last ID is end (not end+1)
#             print(f"  Deleting vectors with IDs {delete_start_id} to {delete_end_id} (inclusive)...")
#             deleted_count = delete_vectors_range(conn, delete_start_id, delete_end_id, args.tablename)
#             print(f"  Deleted {deleted_count} vectors")
        
#         elif op_type == "search":
#             print(f"  Performing search with {len(query_vectors)} queries...")
#             predicted_ids = search_queries(
#                 conn, query_vectors, args.k, args.ef_search, args.nprobe, args.tablename
#             )
            
#             # Load and compare with ground truth
#             gt_file = get_ground_truth_path(gt_dir, step_num)
#             if os.path.exists(gt_file):
#                 print(f"  Loading ground truth from {gt_file}...")
#                 gt_ids, gt_distances = read_usbin(gt_file)
                
#                 recall = compute_recall(predicted_ids, gt_ids, args.k)
#                 print(f"  Recall@{args.k}: {recall:.4f}")
                
#                 search_step_count += 1
#             else:
#                 print(f"  Warning: Ground truth file not found: {gt_file}")
        
#         else:
#             print(f"  Warning: Unknown operation type: {op_type}")
    
#     print(f"\n=== Summary ===")
#     print(f"Total search operations with ground truth: {search_step_count}")
#     if index_built:
#         print(f"Index built before step {args.build_index_before_step}")
    
#     conn.close()
#     print("\nDone!")


# if __name__ == "__main__":
#     main()


# # insertion phase:
# # python test_runbook.py \
# #   --runbook msturing-10M_slidingwindow_runbook.yaml \
# #   --start-step 1  --end-step 100 \
# #   --base-vectors /ssd_root/liu4127/big-ann-benchmarks/data/MSTuringANNS/base1b.fbin.crop_nb_10000000 \
# #   --build-index-before-step 101  --index-type hnsw  --M 16 --ef-construction 40 \
# #   --gt-dir msturing-10M_slidingwindow_gt \
# #   --query-vectors /ssd_root/dataset/turing10m/msturing-query.fvecs \
# #   --host localhost --port 5434 --user liu4127 \
# #   --dbname postgres --tablename turing10m_vectors \
# #   --k 100 --ef-search 100 --nprobe 20 \
# #   --vector-format fvecs

# # python test_runbook.py \
# #   --runbook msturing-10M_slidingwindow_runbook.yaml \
# #   --start-step 101 \
# #   --base-vectors /ssd_root/liu4127/big-ann-benchmarks/data/MSTuringANNS/base1b.fbin.crop_nb_10000000 \
# #   --build-index-before-step 101  --index-type hnsw  --M 16  --ef-construction 40 \
# #   --gt-dir msturing-10M_slidingwindow_gt \
# #   --query-format fbin  --query-vectors /ssd_root/dataset/turing10m/msturing-query.fvecs \
# #   --host localhost --port 5434 --user liu4127 \
# #   --dbname postgres --tablename turing10m_vectors \
# #   --k 100 --ef-search 100 --nprobe 20 \
# #   --vector-format fvecs

# # python test_runbook.py \
# #   --runbook msturing-10M_slidingwindow_runbook.yaml \
# #   --start-step 101  --end-step 101 \
# #   --base-vectors /ssd_root/liu4127/big-ann-benchmarks/data/MSTuringANNS/base1b.fbin.crop_nb_10000000 \
# #   --index-type hnsw  --M 16  --ef-construction 40 \
# #   --gt-dir msturing-10M_slidingwindow_gt \
# #   --query-format fbin  --query-vectors /ssd_root/dataset/turing10m/msturing-query.fvecs \
# #   --host localhost --port 5434 --user liu4127 \
# #   --dbname postgres --tablename turing10m_vectors \
# #   --k 100 --ef-search 100 --nprobe 20 \
# #   --vector-format fvecs
#!/usr/bin/env python3
"""
Test script for runbook execution.

This script reads a runbook YAML file, executes the operations (insert, delete, search),
and compares search results with ground truth files.
"""

import struct
import numpy as np
import psycopg2
import yaml
import argparse
import os
import sys
import time
from pathlib import Path


def read_usbin(file_path):
    """
    Read ground truth file in usbin format.
    
    Format:
    [Header - 8 bytes]
    ├── uint32 n_queries    (4 bytes) - Number of query vectors
    └── uint32 k            (4 bytes) - Number of nearest neighbors (typically 100)
    
    [Data - n_queries * k * 8 bytes]
    ├── int32[n_queries, k]  IDs      - Database vector indices (4 bytes each)
    └── float32[n_queries, k] Distances - Corresponding distances (4 bytes each)
    """
    with open(file_path, "rb") as f:
        # Read header
        header = f.read(8)
        if len(header) < 8:
            raise ValueError(f"Invalid usbin file: header too short in {file_path}")
        
        n_queries, k = struct.unpack("II", header)
        
        # Read IDs (int32)
        id_data = f.read(n_queries * k * 4)
        ids = np.frombuffer(id_data, dtype=np.int32).reshape(n_queries, k)
        
        # Read distances (float32)
        dist_data = f.read(n_queries * k * 4)
        distances = np.frombuffer(dist_data, dtype=np.float32).reshape(n_queries, k)
    
    return ids.tolist(), distances.tolist()


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


def read_fbin(file_path, num_vectors=None):
    """
    Read vectors from .fbin file format.
    
    Format:
    - Header: int32 n (number of vectors), int32 d (dimension)
    - Data: n * d float32 values
    """
    with open(file_path, "rb") as f:
        # Read number of vectors and dimension
        header = np.fromfile(f, dtype=np.int32, count=2)
        if len(header) < 2:
            raise ValueError("Invalid fbin file: header too short")
        n, d = header
        print(f"  File contains {n} vectors of dimension {d}")
        
        # If num_vectors is specified, limit the read
        count = n if num_vectors is None else min(num_vectors, n)
        
        # Read vectors as float32
        data = np.fromfile(f, dtype=np.float32, count=count * d)
        vectors = data.reshape(count, d)
        
    return vectors


def create_table(conn, dim, tablename="test_vectors", drop_if_exists=True):
    """Create a table for storing vectors"""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        if drop_if_exists:
            cur.execute(f"DROP TABLE IF EXISTS {tablename};")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {tablename} (
                id SERIAL PRIMARY KEY,
                vec vector({dim})
            );
        """)
        conn.commit()


def insert_vectors_range(conn, vectors, start_idx, end_idx, tablename="test_vectors", start_id=None):
    """
    Insert vectors from start_idx to end_idx (exclusive) into the database.
    
    Args:
        conn: Database connection
        vectors: Array of all vectors
        start_idx: Starting index in vectors array (0-based, inclusive)
        end_idx: Ending index in vectors array (0-based, exclusive, like Python ranges)
        tablename: Name of the table
        start_id: Starting ID for the first vector (1-based). If None, uses auto-increment.
    """
    with conn.cursor() as cur:
        batch_vectors = vectors[start_idx:end_idx]
        if len(batch_vectors) == 0:
            return
        
        # Insert with explicit IDs if start_id is provided
        if start_id is not None:
            values = [(start_id + i, vec.tolist()) for i, vec in enumerate(batch_vectors)]
            from psycopg2.extras import execute_values
            execute_values(
                cur,
                f"INSERT INTO {tablename} (id, vec) VALUES %s;",
                values
            )
            max_id = start_id + len(batch_vectors) - 1
            # Update the sequence
            cur.execute(f"""
                SELECT setval(pg_get_serial_sequence('{tablename}', 'id'), 
                              GREATEST(COALESCE((SELECT MAX(id) FROM {tablename}), 0), {max_id}), 
                              true);
            """)
        else:
            # Insert without IDs (auto-increment)
            from psycopg2.extras import execute_values
            execute_values(
                cur,
                f"INSERT INTO {tablename} (vec) VALUES %s;",
                [(vec.tolist(),) for vec in batch_vectors]
            )
        conn.commit()


def delete_vectors_range(conn, start_id, end_id, tablename="test_vectors"):
    """Delete vectors with IDs in the range [start_id, end_id] (inclusive)"""
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {tablename} WHERE id >= %s AND id <= %s;", (start_id, end_id))
        deleted_count = cur.rowcount
        conn.commit()
        return deleted_count


def create_index(conn, index_type, index_params, tablename="test_vectors", index_name="test_vector_idx"):
    """
    Create an index on the vector column.
    
    Args:
        conn: Database connection
        index_type: Type of index ("hnsw" or "ivfflat")
        index_params: Dictionary with index parameters
            - For HNSW: M, ef_construction
            - For IVFFlat: nlist
        tablename: Name of the table
        index_name: Name of the index to create
    """
    with conn.cursor() as cur:
        print(f"Checking for existing index and dropping if it exists...")
        cur.execute(f"DROP INDEX IF EXISTS {index_name};")
        
        if index_type == "ivfflat":
            nlist = index_params.get("nlist", 100)
            print(f"Creating IVF_FLAT index with nlist = {nlist}...")
            start_time = time.time()
            index_sql = f"""
                CREATE INDEX {index_name} ON {tablename}
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
                CREATE INDEX {index_name} ON {tablename}
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


def search_queries(conn, queries, top_k=100, ef_search=400, nprobe=20, tablename="test_vectors", db_id_to_tag=None):
    """
    Search for nearest neighbors of query vectors.
    
    Args:
        db_id_to_tag: Optional mapping from database ID (1-based) to tag (0-based index in original dataset).
                     If None, assumes DB_ID - 1 = tag (simple case with no deletions).
    
    Returns:
        List of lists, where each inner list contains the tags (0-based indices in original dataset) of the top_k nearest neighbors
    """
    with conn.cursor() as cur:
        cur.execute(f"SET ivfflat.probes = {nprobe};")
        cur.execute(f"SET hnsw.ef_search = {ef_search};")
        
        all_results = []
        # Process all queries, not just first 100
        for q in queries:
            cur.execute(f"""
                SELECT id
                FROM {tablename}
                ORDER BY vec <-> %s::vector
                LIMIT %s;
            """, (q.tolist(), top_k))
            
            # Convert database IDs to tags (0-based indices in original dataset)
            # Ground truth uses tags (0-based indices in original dataset), not database IDs
            if db_id_to_tag is not None and len(db_id_to_tag) > 0:
                # Use mapping if provided and non-empty
                ids = []
                for row in cur.fetchall():
                    db_id = row[0]
                    tag = db_id_to_tag.get(db_id)
                    if tag is None:
                        # Fallback: assume DB_ID - 1 = tag (works if IDs were set explicitly to match positions)
                        tag = db_id - 1
                    ids.append(tag)
            else:
                # Simple case: assume DB_ID - 1 = tag (works if we always insert with explicit IDs)
                ids = [row[0] - 1 for row in cur.fetchall()]
            
            all_results.append(ids)
        
        return all_results


def compute_recall(predicted_ids, ground_truth_ids, top_k=100):
    """
    Compute recall@top_k.
    
    Args:
        predicted_ids: List of lists of predicted IDs (0-based)
        ground_truth_ids: List of lists of ground truth IDs (0-based)
        top_k: Number of top results to consider
    
    Returns:
        Recall value (0.0 to 1.0)
    """
    total_correct = 0
    total_expected = 0
    
    for i in range(len(predicted_ids)):
        pred = set(predicted_ids[i][:top_k])
        if i < len(ground_truth_ids):
            gt = set(ground_truth_ids[i][:top_k])
        else:
            gt = set()
        
        correct = len(pred & gt)
        total_correct += correct
        total_expected += len(gt)
    
    if total_expected == 0:
        return 0.0
    
    recall = total_correct / total_expected
    return recall


def load_runbook(runbook_path):
    """Load and parse the runbook YAML file"""
    with open(runbook_path, 'r') as f:
        runbook = yaml.safe_load(f)
    return runbook


def get_ground_truth_path(gt_dir, step_num):
    """
    Get the path to the ground truth file for a given step.
    
    Args:
        gt_dir: Directory containing ground truth files
        step_num: Step number
    
    Returns:
        Path to the ground truth file (step{step_num}.gt100)
    """
    gt_file = os.path.join(gt_dir, f"step{step_num}.gt100")
    return gt_file


def read_tags_file(file_path):
    """
    Read the tags file format.
    
    Format:
    - uint32 tags.size (number of tags)
    - uint32 1 (always 1)
    - uint32[tags.size] tags (the tag values)
    
    Returns:
        Tuple of (tags array, size, one_value)
    """
    with open(file_path, "rb") as f:
        size_bytes = f.read(4)
        if len(size_bytes) < 4:
            raise ValueError(f"Invalid tags file: header too short in {file_path}")
        size = struct.unpack("I", size_bytes)[0]
        
        one_bytes = f.read(4)
        one = struct.unpack("I", one_bytes)[0]
        
        tags_data = f.read(size * 4)
        tags = np.frombuffer(tags_data, dtype=np.uint32)
    
    return tags, size, one


def get_tags_file_path(gt_dir, step_num):
    """
    Get the path to the tags file for a given step.
    
    Args:
        gt_dir: Directory containing ground truth files
        step_num: Step number
    
    Returns:
        Path to the tags file (step{step_num}.tags)
    """
    tags_file = os.path.join(gt_dir, f"step{step_num}.tags")
    return tags_file


def main():
    parser = argparse.ArgumentParser(description="Test runbook execution")
    parser.add_argument("--runbook", required=True, help="Path to runbook YAML file")
    parser.add_argument("--base-vectors", required=True, help="Path to base vectors file (.fvecs or .bvecs)")
    parser.add_argument("--query-vectors", required=True, help="Path to query vectors file (.fvecs or .bvecs)")
    parser.add_argument("--host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--user", default="postgres", help="PostgreSQL user")
    parser.add_argument("--password", default="postgres", help="PostgreSQL password")
    parser.add_argument("--dbname", default="postgres", help="PostgreSQL database name")
    parser.add_argument("--tablename", default="test_vectors", help="Table name")
    parser.add_argument("--k", type=int, default=100, help="Number of nearest neighbors to retrieve")
    parser.add_argument("--ef-search", type=int, default=400, help="HNSW ef_search parameter")
    parser.add_argument("--nprobe", type=int, default=20, help="IVFFlat nprobe parameter")
    parser.add_argument("--vector-format", choices=["fvecs", "bvecs", "fbin"], default="fvecs",
                        help="Format of vector files")
    parser.add_argument("--query-format", choices=["fvecs", "bvecs", "fbin"], default=None,
                        help="Format of query vector file (defaults to --vector-format if not specified)")
    parser.add_argument("--gt-dir", default=None,
                        help="Directory containing ground truth files. If not specified, "
                             "defaults to {runbook_name}_gt in the same directory as the runbook")
    parser.add_argument("--build-index-before-step", type=int, default=None,
                        help="Build index before this step number starts")
    parser.add_argument("--index-type", choices=["hnsw", "ivfflat"], default="hnsw",
                        help="Type of index to build")
    parser.add_argument("--index-name", default="test_vector_idx",
                        help="Name of the index to create")
    parser.add_argument("--M", type=int, default=32,
                        help="HNSW: number of neighbors per node")
    parser.add_argument("--ef-construction", type=int, default=64,
                        help="HNSW: size of dynamic list for graph construction")
    parser.add_argument("--nlist", type=int, default=1000,
                        help="IVFFlat: number of lists")
    parser.add_argument("--start-step", type=int, default=None,
                        help="First step to process (inclusive). If not specified, starts from the first step")
    parser.add_argument("--end-step", type=int, default=None,
                        help="Last step to process (inclusive). If not specified, processes until the last step")
    parser.add_argument("--drop-table", action="store_true", default=False,
                        help="Drop and recreate the table at the start. By default, the table is not dropped.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Print debug information about ID mappings and search results")
    args = parser.parse_args()
    
    # Load runbook
    print(f"Loading runbook from {args.runbook}...")
    runbook_data = load_runbook(args.runbook)
    
    # Get the first (and typically only) dataset name
    dataset_name = list(runbook_data.keys())[0]
    operations = runbook_data[dataset_name]
    
    print(f"Dataset: {dataset_name}")
    print(f"Number of operations: {len(operations)}")
    
    # Determine ground truth directory
    if args.gt_dir:
        gt_dir = args.gt_dir
    else:
        # Default: infer from runbook name
        runbook_dir = os.path.dirname(args.runbook)
        runbook_name = os.path.splitext(os.path.basename(args.runbook))[0]
        gt_dir = os.path.join(runbook_dir, f"{runbook_name}_gt")
    
    print(f"Ground truth directory: {gt_dir}")
    
    # Auto-detect format from file extension if not explicitly specified
    def detect_format(file_path, default_format):
        """Detect file format from extension or filename pattern"""
        file_lower = file_path.lower()
        # Check for exact extension matches first
        if file_lower.endswith('.fbin') or '.fbin' in file_lower:
            return 'fbin'
        elif file_lower.endswith('.fvecs') or '.fvecs' in file_lower:
            return 'fvecs'
        elif file_lower.endswith('.bvecs') or '.bvecs' in file_lower:
            return 'bvecs'
        return default_format
    
    # Load base vectors
    print(f"\nLoading base vectors from {args.base_vectors}...")
    base_format = detect_format(args.base_vectors, args.vector_format)
    if base_format != args.vector_format:
        print(f"  Auto-detected format: {base_format} (from file extension)")
    if base_format == "fvecs":
        base_vectors = read_fvecs(args.base_vectors)
    elif base_format == "bvecs":
        base_vectors = read_bvecs(args.base_vectors)
    elif base_format == "fbin":
        base_vectors = read_fbin(args.base_vectors)
    else:
        raise ValueError(f"Unsupported base vector format: {base_format}")
    print(f"Loaded {len(base_vectors)} base vectors of dimension {base_vectors.shape[1]}")
    
    # Load query vectors
    print(f"\nLoading query vectors from {args.query_vectors}...")
    query_format_default = args.query_format if args.query_format is not None else args.vector_format
    query_format = detect_format(args.query_vectors, query_format_default)
    if query_format != query_format_default:
        print(f"  Auto-detected format: {query_format} (from file extension)")
    if query_format == "fvecs":
        query_vectors = read_fvecs(args.query_vectors)
    elif query_format == "bvecs":
        query_vectors = read_bvecs(args.query_vectors)
    elif query_format == "fbin":
        query_vectors = read_fbin(args.query_vectors)
    else:
        raise ValueError(f"Unsupported query format: {query_format}")
    print(f"Loaded {len(query_vectors)} query vectors of dimension {query_vectors.shape[1]}")
    
    # Connect to database
    print(f"\nConnecting to database...")
    conn = psycopg2.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.dbname
    )
    
    # Create table
    if args.drop_table:
        print(f"\nDropping and creating table {args.tablename}...")
    else:
        print(f"\nCreating table {args.tablename} (if not exists)...")
    create_table(conn, base_vectors.shape[1], args.tablename, drop_if_exists=args.drop_table)
    
    # Track current max ID for inserts
    current_max_id = 0
    
    # Maintain mapping from database ID (1-based) to tag (0-based index in original dataset)
    # This is needed because ground truth uses tags, not database IDs
    # We build the mapping from operations we process, not from existing database IDs,
    # because we can't reliably determine the tag mapping without knowing the insertion history.
    db_id_to_tag = {}
    if not args.drop_table:
        with conn.cursor() as cur:
            cur.execute(f"SELECT id FROM {args.tablename} ORDER BY id;")
            existing_ids = [row[0] for row in cur.fetchall()]
            if existing_ids:
                current_max_id = max(existing_ids)
                print(f"  Found {len(existing_ids)} existing vectors in table (max ID: {current_max_id})")
                print(f"  Warning: Existing vectors found. Mapping will be built from operations in this run.")
                print(f"  If you need to use existing vectors from a previous run, you may need to")
                print(f"  process all operations from the beginning or use --drop-table to start fresh.")
    
    # Prepare index parameters if index building is requested
    index_params = None
    if args.build_index_before_step is not None:
        index_params = {
            "M": args.M,
            "ef_construction": args.ef_construction,
            "nlist": args.nlist,
        }
        print(f"\nIndex will be built before step {args.build_index_before_step}")
        print(f"Index type: {args.index_type}")
        if args.index_type == "hnsw":
            print(f"  M: {args.M}, ef_construction: {args.ef_construction}")
        else:
            print(f"  nlist: {args.nlist}")
    
    # Execute operations
    print(f"\nExecuting operations...")
    search_step_count = 0
    index_built = False
    
    # Filter out non-numeric keys (like 'max_pts') and sort by integer value
    # YAML may load numeric keys as integers or strings, so we need to handle both
    step_numbers = []
    for key in operations.keys():
        if isinstance(key, int):
            step_numbers.append(key)
        elif isinstance(key, str):
            try:
                step_numbers.append(int(key))
            except ValueError:
                # Skip non-numeric keys like 'max_pts'
                continue
        else:
            # Skip other types
            continue
    
    # Sort step numbers
    step_numbers = sorted(step_numbers)
    
    # Filter by step range if specified
    if args.start_step is not None:
        step_numbers = [s for s in step_numbers if s >= args.start_step]
        print(f"Filtered to steps >= {args.start_step}: {len(step_numbers)} steps remaining")
    
    if args.end_step is not None:
        step_numbers = [s for s in step_numbers if s <= args.end_step]
        print(f"Filtered to steps <= {args.end_step}: {len(step_numbers)} steps remaining")
    
    if len(step_numbers) == 0:
        print("Error: No steps to process after filtering")
        sys.exit(1)
    
    print(f"Processing {len(step_numbers)} steps: {step_numbers[0]} to {step_numbers[-1]}")
    
    for step_num in step_numbers:
        # Build index before the specified step if requested
        if (args.build_index_before_step is not None and 
            step_num == args.build_index_before_step and 
            not index_built):
            print(f"\n--- Building index before step {step_num} ---")
            create_index(conn, args.index_type, index_params, args.tablename, args.index_name)
            index_built = True
        
        # Try both integer and string key (YAML may load keys as either)
        op = operations.get(step_num) or operations.get(str(step_num))
        if op is None:
            print(f"  Warning: Could not find operation for step {step_num}")
            continue
        op_type = op.get("operation")
        
        print(f"\n--- Step {step_num}: {op_type} ---")
        
        if op_type == "insert":
            start = op.get("start")
            end = op.get("end")
            if start is None or end is None:
                print(f"  Warning: Missing start or end for insert operation")
                continue
            
            if start < 0 or end < 0 or start >= len(base_vectors) or end > len(base_vectors):
                print(f"  Warning: Indices out of range (start={start}, end={end} (exclusive), max={len(base_vectors)})")
                continue
            
            if start >= end:
                print(f"  Warning: Invalid range (start={start} >= end={end})")
                continue
            
            # Insert vectors with 1-based IDs
            # end is exclusive, so we insert vectors [start, end)
            num_vectors = end - start
            insert_start_id = start + 1
            insert_end_id = end  # end is exclusive, so last ID is end (not end+1)
            print(f"  Inserting vectors {start} to {end} (exclusive, IDs {insert_start_id} to {insert_end_id})...")
            insert_vectors_range(conn, base_vectors, start, end, args.tablename, insert_start_id)
            current_max_id = max(current_max_id, insert_end_id)
            
            # Update mapping: database ID -> tag (original position)
            for i in range(start, end):
                db_id = i + 1
                db_id_to_tag[db_id] = i
            
            print(f"  Inserted {num_vectors} vectors")
        
        elif op_type == "delete":
            start = op.get("start")
            end = op.get("end")
            if start is None or end is None:
                print(f"  Warning: Missing start or end for delete operation")
                continue
            
            if start > end:
                print(f"  Warning: Invalid range (start={start} > end={end})")
                continue
            
            # Delete vectors with 1-based IDs
            # end is exclusive, so we delete IDs [start+1, end) which is [start+1, end-1] inclusive
            # But since end is exclusive, the last ID to delete is 'end' (not end+1)
            # Example: start=0, end=50000 (exclusive) means delete IDs 1 to 50000
            delete_start_id = start + 1
            delete_end_id = end  # end is exclusive, so last ID is end (not end+1)
            print(f"  Deleting vectors with IDs {delete_start_id} to {delete_end_id} (inclusive)...")
            deleted_count = delete_vectors_range(conn, delete_start_id, delete_end_id, args.tablename)
            
            # Remove deleted IDs from mapping
            for db_id in range(delete_start_id, delete_end_id + 1):
                db_id_to_tag.pop(db_id, None)
            
            print(f"  Deleted {deleted_count} vectors")
        
        elif op_type == "search":
            print(f"  Performing search with {len(query_vectors)} queries...")
            predicted_ids = search_queries(
                conn, query_vectors, args.k, args.ef_search, args.nprobe, args.tablename, db_id_to_tag
            )
            
            # Load and compare with ground truth
            gt_file = get_ground_truth_path(gt_dir, step_num)
            if os.path.exists(gt_file):
                print(f"  Loading ground truth from {gt_file}...")
                gt_ids, gt_distances = read_usbin(gt_file)
                
                # Optionally validate using tags file (if available)
                tags_file = get_tags_file_path(gt_dir, step_num)
                if os.path.exists(tags_file) and args.debug:
                    print(f"  Loading tags file from {tags_file} for validation...")
                    tags, tags_size, _ = read_tags_file(tags_file)
                    tags_set = set(tags)
                    
                    # Validate that all ground truth IDs are valid tags
                    invalid_ids = []
                    for query_idx in range(min(10, len(gt_ids))):  # Check first 10 queries
                        for gt_id in gt_ids[query_idx]:
                            if gt_id not in tags_set:
                                invalid_ids.append((query_idx, gt_id))
                    
                    if invalid_ids:
                        print(f"  Warning: Found {len(invalid_ids)} invalid ground truth IDs (not in tags file)")
                        if len(invalid_ids) <= 10:
                            print(f"    Examples: {invalid_ids[:10]}")
                    else:
                        print(f"  Validation: All checked ground truth IDs are valid tags (tags file has {tags_size} tags)")
                
                if args.debug:
                    print(f"  Debug: Number of queries - predicted: {len(predicted_ids)}, ground truth: {len(gt_ids)}")
                    if len(predicted_ids) > 0 and len(gt_ids) > 0:
                        print(f"  Debug: First query predicted IDs (first 10): {predicted_ids[0][:10]}")
                        print(f"  Debug: First query ground truth IDs (first 10): {gt_ids[0][:10]}")
                        print(f"  Debug: Mapping size: {len(db_id_to_tag) if db_id_to_tag else 0}")
                        if len(predicted_ids[0]) > 0:
                            # Check if predicted IDs are in ground truth
                            pred_set = set(predicted_ids[0][:10])
                            gt_set = set(gt_ids[0][:10])
                            intersection = pred_set & gt_set
                            print(f"  Debug: Intersection of first 10 predicted and GT IDs: {intersection}")
                            if db_id_to_tag and len(predicted_ids[0]) > 0:
                                sample_tag = predicted_ids[0][0]
                                sample_db_id = sample_tag + 1  # Convert tag to DB ID (assuming DB_ID = tag + 1)
                                mapped_tag = db_id_to_tag.get(sample_db_id, 'NOT IN MAP')
                                print(f"  Debug: Sample - predicted tag {sample_tag} -> assumed DB ID {sample_db_id} -> mapped tag {mapped_tag}")
                                
                                # Also check a ground truth ID
                                if len(gt_ids[0]) > 0:
                                    gt_tag = gt_ids[0][0]
                                    gt_db_id = gt_tag + 1
                                    gt_mapped_tag = db_id_to_tag.get(gt_db_id, 'NOT IN MAP')
                                    print(f"  Debug: GT sample - GT tag {gt_tag} -> assumed DB ID {gt_db_id} -> mapped tag {gt_mapped_tag}")
                
                # Only compare queries that we actually processed
                num_queries_to_compare = min(len(predicted_ids), len(gt_ids))
                if num_queries_to_compare < len(gt_ids):
                    print(f"  Warning: Only comparing first {num_queries_to_compare} queries (ground truth has {len(gt_ids)})")
                
                recall = compute_recall(predicted_ids[:num_queries_to_compare], gt_ids[:num_queries_to_compare], args.k)
                print(f"  Recall@{args.k}: {recall:.4f}")
                
                search_step_count += 1
            else:
                print(f"  Warning: Ground truth file not found: {gt_file}")
        
        else:
            print(f"  Warning: Unknown operation type: {op_type}")
    
    print(f"\n=== Summary ===")
    print(f"Total search operations with ground truth: {search_step_count}")
    if index_built:
        print(f"Index built before step {args.build_index_before_step}")
    
    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()


# insertion phase:
# python test_runbook.py \
#   --runbook msturing-10M_slidingwindow_runbook.yaml \
#   --start-step 1  --end-step 100 \
#   --base-vectors /ssd_root/liu4127/big-ann-benchmarks/data/MSTuringANNS/base1b.fbin.crop_nb_10000000 \
#   --build-index-before-step 101  --index-type hnsw  --M 16 --ef-construction 40 \
#   --gt-dir msturing-10M_slidingwindow_gt \
#   --query-vectors /ssd_root/dataset/turing10m/msturing-query.fvecs \
#   --host localhost --port 5434 --user liu4127 \
#   --dbname postgres --tablename turing10m_vectors \
#   --k 100 --ef-search 100 --nprobe 20 \
#   --vector-format fvecs

# python test_runbook.py \
#   --runbook msturing-10M_slidingwindow_runbook.yaml \
#   --start-step 101 \
#   --base-vectors /ssd_root/liu4127/big-ann-benchmarks/data/MSTuringANNS/base1b.fbin.crop_nb_10000000 \
#   --build-index-before-step 101  --index-type hnsw  --M 16  --ef-construction 40 \
#   --gt-dir msturing-10M_slidingwindow_gt \
#   --query-format fbin  --query-vectors /ssd_root/dataset/turing10m/msturing-query.fvecs \
#   --host localhost --port 5434 --user liu4127 \
#   --dbname postgres --tablename turing10m_vectors \
#   --k 100 --ef-search 100 --nprobe 20 \
#   --vector-format fvecs

# python test_runbook.py \
#   --runbook msturing-10M_slidingwindow_runbook.yaml \
#   --start-step 101  --end-step 101 \
#   --base-vectors /ssd_root/liu4127/big-ann-benchmarks/data/MSTuringANNS/base1b.fbin.crop_nb_10000000 \
#   --index-type hnsw  --M 16  --ef-construction 40 \
#   --gt-dir msturing-10M_slidingwindow_gt \
#   --query-format fbin  --query-vectors /ssd_root/dataset/turing10m/msturing-query.fvecs \
#   --host localhost --port 5434 --user liu4127 \
#   --dbname postgres --tablename turing10m_vectors \
#   --k 100 --ef-search 100 --nprobe 20 \
#   --vector-format fvecs