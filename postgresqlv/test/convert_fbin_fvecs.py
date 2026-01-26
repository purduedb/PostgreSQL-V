#!/usr/bin/env python3
"""
Convert .fbin (n:int32, d:int32, followed by float32 data) to .fvecs.

Usage examples:
  python3 fbin_to_fvecs.py --input base.fbin --output base.fvecs
  python3 fbin_to_fvecs.py -i base.fbin -o base.fvecs --count 100000
"""

import argparse
import os
import struct
import sys
from typing import Tuple

import numpy as np


def read_fbin_header(fp) -> Tuple[int, int]:
    hdr = fp.read(8)
    if len(hdr) != 8:
        raise ValueError("File too small to be a valid .fbin (missing 8-byte header).")
    n, d = struct.unpack("<ii", hdr)  # little-endian int32, int32
    if n <= 0 or d <= 0:
        raise ValueError(f"Invalid header values: n={n}, d={d}")
    return n, d


def convert_fbin_to_fvecs(
    input_path: str,
    output_path: str,
    count: int | None,
    chunk_vectors: int = 8192,
) -> None:
    with open(input_path, "rb") as fin:
        n, d = read_fbin_header(fin)

        to_convert = n if count is None else min(count, n)
        if to_convert < 0:
            raise ValueError("--count must be >= 0")

        # Validate file size (best-effort sanity check)
        fin.seek(0, os.SEEK_END)
        file_size = fin.tell()
        expected_min = 8 + (n * d * 4)
        if file_size < expected_min:
            raise ValueError(
                f"File seems truncated: size={file_size} bytes, "
                f"expected at least {expected_min} bytes for n={n}, d={d}."
            )
        fin.seek(8, os.SEEK_SET)  # jump to data start

        with open(output_path, "wb") as fout:
            # Prepare per-vector dim prefix (little-endian int32)
            dim_prefix = struct.pack("<i", d)

            remaining = to_convert
            while remaining > 0:
                cur = min(chunk_vectors, remaining)

                # Read cur*d float32 values
                num_floats = cur * d
                # np.fromfile works on real file objects; it reads binary quickly.
                data = np.fromfile(fin, dtype="<f4", count=num_floats)
                if data.size != num_floats:
                    raise ValueError(
                        f"Unexpected EOF while reading data: "
                        f"wanted {num_floats} float32, got {data.size}."
                    )
                data = data.reshape(cur, d)

                # Write as fvecs: [int32 d][float32 * d] per vector
                # We'll write vector-by-vector to keep the format exact.
                # (This is still fast enough in chunks.)
                for i in range(cur):
                    fout.write(dim_prefix)
                    fout.write(data[i].astype("<f4", copy=False).tobytes(order="C"))

                remaining -= cur

    print(f"Converted {to_convert} / {n} vectors (dim={d}) -> {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert FBIN to FVECs with optional count limit.")
    p.add_argument("-i", "--input", required=True, help="Input .fbin file")
    p.add_argument("-o", "--output", required=True, help="Output .fvecs file")
    p.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of vectors to convert (default: all)",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=8192,
        help="Vectors per chunk for streaming (default: 8192)",
    )
    args = p.parse_args()

    convert_fbin_to_fvecs(
        input_path=args.input,
        output_path=args.output,
        count=args.count,
        chunk_vectors=args.chunk,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
