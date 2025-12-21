#!/usr/bin/env python3
"""
Script to read vector query files (.bvecs/.fvecs) and generate a SQL file
containing all queries as SQL statements.
"""

import struct
import numpy as np
import argparse
import sys


def read_bvecs(file_path, num_vectors=None):
    """Read .bvecs file (byte vectors)"""
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


def read_fvecs(file_path, num_vectors=None):
    """Read .fvecs file (float vectors)"""
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


def vector_to_postgres_format(vector, is_byte_vector=False):
    """
    Convert a numpy vector to PostgreSQL vector format string.
    For byte vectors, convert to float for PostgreSQL.
    """
    if is_byte_vector:
        # Convert byte vector to float
        vec_float = vector.astype(np.float32)
    else:
        vec_float = vector.astype(np.float32)
    
    # Convert to list and format as PostgreSQL vector string: [1.0, 2.0, 3.0, ...]
    vec_list = vec_float.tolist()
    vec_str = "[" + ",".join(str(x) for x in vec_list) + "]"
    return vec_str


def generate_sql_file(vectors, output_file, tablename="bigann_vectors", top_k=100, 
                      is_byte_vector=False, begin_transaction=True, efs_nprobe=None, index_type=None):
    """
    Generate SQL file with all queries.
    
    Args:
        vectors: numpy array of vectors (shape: [num_queries, dim])
        output_file: path to output SQL file
        tablename: name of the table to query
        top_k: number of results to return per query
        is_byte_vector: whether vectors are byte vectors (need conversion)
        begin_transaction: whether to wrap all queries in a transaction
        efs_nprobe: value to set for hnsw.ef_search or ivfflat.probes (optional)
        index_type: index type to use - "hnsw" or "ivfflat" (required if efs_nprobe is set)
    """
    with open(output_file, 'w') as f:
        # Write header comment
        f.write(f"-- Generated SQL file with {len(vectors)} queries\n")
        f.write(f"-- Table: {tablename}\n")
        f.write(f"-- Top-K: {top_k}\n")
        f.write(f"-- Vector dimension: {vectors.shape[1]}\n")
        if efs_nprobe is not None:
            param_name = "ef_search" if index_type == "hnsw" else "probes"
            f.write(f"-- {index_type.upper()}.{param_name}: {efs_nprobe}\n")
        f.write("\n")
        
        # Begin transaction if requested
        if begin_transaction:
            f.write("BEGIN;\n\n")
            # Set efs_nprobe at the beginning of transaction if provided
            if efs_nprobe is not None:
                if index_type == "hnsw":
                    f.write(f"SET LOCAL hnsw.ef_search = {efs_nprobe};\n")
                elif index_type == "ivfflat":
                    f.write(f"SET LOCAL ivfflat.probes = {efs_nprobe};\n")
                else:
                    raise ValueError(f"Invalid index_type: {index_type}. Must be 'hnsw' or 'ivfflat'")
                f.write("\n")
        
        # Generate SQL for each query
        for i, vec in enumerate(vectors):
            vec_str = vector_to_postgres_format(vec, is_byte_vector)
            # Use string literal format for PostgreSQL vector
            sql = f"""SELECT id
FROM {tablename}
ORDER BY vec <-> '{vec_str}'::vector
LIMIT {top_k};
"""
            f.write(f"-- Query {i + 1}\n")
            f.write(sql)
            f.write("\n")
        
        # Commit transaction if we began one
        if begin_transaction:
            f.write("\nCOMMIT;\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SQL file from vector query file (.bvecs/.fvecs)"
    )
    parser.add_argument("--input", required=True, 
                       help="Path to input .bvecs or .fvecs file")
    parser.add_argument("--output", required=True,
                       help="Path to output SQL file")
    parser.add_argument("--tablename", default="bigann_vectors",
                       help="Name of the table to query (default: bigann_vectors)")
    parser.add_argument("--top-k", type=int, default=100,
                       help="Number of results per query (default: 100)")
    parser.add_argument("--num-queries", type=int, default=None,
                       help="Number of queries to process (default: all)")
    parser.add_argument("--no-transaction", action="store_true",
                       help="Don't wrap queries in BEGIN/COMMIT transaction")
    parser.add_argument("--format", choices=["auto", "bvecs", "fvecs"], default="auto",
                       help="File format (default: auto-detect from extension)")
    parser.add_argument("--efs-nprobe", type=int, default=None,
                       help="Set hnsw.ef_search or ivfflat.probes to this value at the beginning of transaction")
    parser.add_argument("--index-type", choices=["hnsw", "ivfflat"], default=None,
                       help="Index type to use with --efs-nprobe: 'hnsw' sets ef_search, 'ivfflat' sets probes")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.efs_nprobe is not None and args.index_type is None:
        print("Error: --index-type must be specified when --efs-nprobe is provided")
        sys.exit(1)
    
    # Determine file format
    if args.format == "auto":
        if args.input.endswith(".bvecs"):
            file_format = "bvecs"
        elif args.input.endswith(".fvecs"):
            file_format = "fvecs"
        else:
            print(f"Error: Cannot auto-detect format from file extension. Use --format to specify.")
            sys.exit(1)
    else:
        file_format = args.format
    
    # Read vectors
    print(f"Reading {file_format} file: {args.input}")
    try:
        if file_format == "bvecs":
            vectors = read_bvecs(args.input, args.num_queries)
        else:  # fvecs
            vectors = read_fvecs(args.input, args.num_queries)
        
        print(f"Loaded {len(vectors)} vectors of dimension {vectors.shape[1]}")
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Generate SQL file
    print(f"Generating SQL file: {args.output}")
    try:
        generate_sql_file(
            vectors, 
            args.output,
            tablename=args.tablename,
            top_k=args.top_k,
            is_byte_vector=(file_format == "bvecs"),
            begin_transaction=not args.no_transaction,
            efs_nprobe=args.efs_nprobe,
            index_type=args.index_type
        )
        print(f"Successfully generated SQL file with {len(vectors)} queries")
    except Exception as e:
        print(f"Error generating SQL file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Generate SQL from .fvecs file
# python generate_query_sql.py --input /ssd_root/dataset/sift/sift_query.fvecs --output sift_queries.sql \
#     --tablename bigann_vectors --num-queries 10000 --efs-nprobe 400 --index-type hnsw --top-k 100

# Generate SQL without transaction wrapper
    # --no-transaction