#!/usr/bin/env python3
"""
Script to create a query table and insert query vectors from a file.
"""

import struct
import numpy as np
import psycopg2
import argparse
import sys

from psycopg2 import sql


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


def create_query_table(conn, dim, tablename="sift_queries"):
    """Create a query table with id and v columns"""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"DROP TABLE IF EXISTS {tablename};")
        cur.execute(f"""
            CREATE TABLE {tablename} (
                id SERIAL PRIMARY KEY,
                v vector({dim})
            );
        """)
        conn.commit()
        print(f"Created table {tablename} with columns: id (int), v (vector({dim}))")


def insert_query_vectors(conn, vectors, tablename="sift_queries"):
    """Insert query vectors into the table"""
    with conn.cursor() as cur:
        print(f"Inserting {len(vectors)} query vectors...")
        for i, vec in enumerate(vectors):
            # Convert byte vectors to float for PostgreSQL
            if vec.dtype == np.uint8:
                vec_float = vec.astype(np.float32)
            else:
                vec_float = vec.astype(np.float32)
            
            cur.execute(f"INSERT INTO {tablename} (v) VALUES (%s);", (vec_float.tolist(),))
            
            if (i + 1) % 1000 == 0:
                print(f"  Inserted {i + 1}/{len(vectors)} vectors...")
        
        conn.commit()
        print(f"Successfully inserted {len(vectors)} query vectors into {tablename}")


def main():
    parser = argparse.ArgumentParser(
        description="Create query table and insert query vectors from file"
    )
    parser.add_argument("--file", required=True,
                       help="Path to query .bvecs or .fvecs file")
    parser.add_argument("--num", type=int, default=None,
                       help="Number of query vectors to insert (default: all)")
    parser.add_argument("--format", choices=["auto", "bvecs", "fvecs"], default="auto",
                       help="File format (default: auto-detect from extension)")
    parser.add_argument("--host", default="localhost",
                       help="Database host (default: localhost)")
    parser.add_argument("--port", default=5432, type=int,
                       help="Database port (default: 5432)")
    parser.add_argument("--user", default="postgres",
                       help="Database user (default: postgres)")
    parser.add_argument("--password", default="postgres",
                       help="Database password (default: postgres)")
    parser.add_argument("--dbname", default="postgres",
                       help="Database name (default: postgres)")
    parser.add_argument("--tablename", default="sift_queries",
                       help="Query table name (default: sift_queries)")
    
    args = parser.parse_args()
    
    # Determine file format
    if args.format == "auto":
        if args.file.endswith(".bvecs"):
            file_format = "bvecs"
        elif args.file.endswith(".fvecs"):
            file_format = "fvecs"
        else:
            print(f"Error: Cannot auto-detect format from file extension. Use --format to specify.")
            sys.exit(1)
    else:
        file_format = args.format
    
    # Read query vectors
    print(f"Reading {file_format} file: {args.file}")
    try:
        if file_format == "bvecs":
            vectors = read_bvecs(args.file, args.num)
        else:  # fvecs
            vectors = read_fvecs(args.file, args.num)
        
        print(f"Loaded {len(vectors)} query vectors of dimension {vectors.shape[1]}")
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Connect to database
    try:
        conn = psycopg2.connect(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            dbname=args.dbname
        )
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)
    
    # Create table and insert vectors
    try:
        create_query_table(conn, vectors.shape[1], args.tablename)
        insert_query_vectors(conn, vectors, args.tablename)
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

# Example usage:
# python insert_query_table.py --file /ssd_root/dataset/sift/bigann_query.bvecs \
#   --tablename sift_queries --num 10000 \
#   --host localhost --port 5434 --user liu4127 --dbname postgres