import struct
import numpy as np
import psycopg2
import argparse
import time
import sys

from psycopg2 import sql

def read_bvecs(file_path, num_vectors=0, skip_vectors=0):
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

def insert_vectors(conn, vectors):
    with conn.cursor() as cur:
        for vec in vectors:
            cur.execute("INSERT INTO bigann_vectors (vec) VALUES (%s);", (vec.tolist(),))
        conn.commit()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to .bvecs file for insert")
    parser.add_argument("--skip-num", type=int, default=1000, help="Number of total vectors to skip")
    parser.add_argument("--num", type=int, default=1000, help="Number of base vectors to insert")
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
    base_vectors = read_bvecs(args.file, args.num, args.skip_num)
    print(f"Loaded {len(base_vectors)} base vectors of dimension {base_vectors.shape[1]}")
    print("Inserting vectors...")
    insert_vectors(conn, base_vectors)
    print("Insertion done.\n")

    conn.close()

if __name__ == "__main__":
    main()


# python bigann_insert_after_create.py \
#   --file /ssd_root/dataset/sift/bigann_base.bvecs \
#   --num 100000 --skip-num 10000000\
#   --host localhost --port 5434 --user liu4127 --dbname postgres
