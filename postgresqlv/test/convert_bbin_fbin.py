#!/usr/bin/env python3
"""
Script to read vectors from .bbin file and convert them to .fbin file.
.bbin format: Header (2 int32: n, d) followed by n*d uint8 values
.fbin format: Header (2 int32: n, d) followed by n*d float32 values
"""

import struct
import numpy as np
import argparse
import sys

# Maximum number of vectors to read at a time (1M)
MAX_BATCH_SIZE = 1000000


def read_bbin_batch(file_handle, batch_size, dim, header_read=False):
    """
    Read a batch of vectors from .bbin file.
    
    Args:
        file_handle: Open file handle (binary mode)
        batch_size: Maximum number of vectors to read in this batch
        dim: Dimension of vectors
        header_read: Whether the header has already been read
    
    Returns:
        Tuple of (vectors as numpy array, eof_reached)
    """
    if not header_read:
        # Read header: number of vectors and dimension
        header = np.fromfile(file_handle, dtype=np.int32, count=2)
        if len(header) < 2:
            raise ValueError("Invalid bbin file: header too short")
        n, d = header
        if d != dim:
            raise ValueError(f"Dimension mismatch: expected {dim}, got {d}")
    
    # Read vectors as uint8
    data = np.fromfile(file_handle, dtype=np.uint8, count=batch_size * dim)
    if len(data) == 0:
        return None, True
    
    # Check if we got a complete batch
    actual_count = len(data) // dim
    if actual_count < batch_size:
        # EOF reached
        vectors = data.reshape(actual_count, dim)
        return vectors, True
    
    vectors = data.reshape(batch_size, dim)
    return vectors, False


def write_fbin_header(file_path, num_vectors, dim, update_only=False):
    """
    Write or update header in .fbin file.
    
    Args:
        file_path: Path to output .fbin file
        num_vectors: Total number of vectors
        dim: Dimension of vectors
        update_only: If True, only update the num_vectors field (first int32)
    """
    if update_only:
        # Update only the num_vectors field (first int32)
        with open(file_path, "r+b") as f:
            f.seek(0)
            np.array([num_vectors], dtype=np.int32).tofile(f)
    else:
        # Write full header (will truncate if file exists)
        with open(file_path, "wb") as f:
            header = np.array([num_vectors, dim], dtype=np.int32)
            header.tofile(f)


def append_fbin_vectors(file_path, vectors):
    """
    Append vectors to .fbin file (header must already be written).
    
    Args:
        file_path: Path to output .fbin file
        vectors: Numpy array of vectors (uint8, will be converted to float32)
    """
    # Convert uint8 to float32
    vectors_float = vectors.astype(np.float32)
    
    with open(file_path, "ab") as f:
        vectors_float.tofile(f)


def main():
    parser = argparse.ArgumentParser(
        description="Convert vectors from .bbin file to .fbin file"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input .bbin file"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to output .fbin file"
    )
    parser.add_argument(
        "-n", "--num-vectors",
        type=int,
        default=None,
        help="Number of vectors to read and convert (default: all vectors)"
    )
    
    args = parser.parse_args()
    
    try:
        # First, read the header to get total vectors and dimension
        with open(args.input_file, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=2)
            if len(header) < 2:
                raise ValueError("Invalid bbin file: header too short")
            n_total, dim = header
            print(f"File contains {n_total} vectors of dimension {dim}")
        
        # Determine how many vectors to convert
        num_to_convert = n_total if args.num_vectors is None else min(args.num_vectors, n_total)
        print(f"Converting {num_to_convert} vectors in batches of up to {MAX_BATCH_SIZE}...")
        
        # Write header to output file (we'll update num_vectors at the end if needed)
        write_fbin_header(args.output_file, num_to_convert, dim)
        
        # Process vectors in batches
        total_converted = 0
        batch_num = 0
        last_percent = -1  # Track last percentage printed
        
        with open(args.input_file, "rb") as f_in:
            # Skip the header (2 int32 = 8 bytes)
            f_in.seek(8)
            
            while total_converted < num_to_convert:
                # Calculate batch size
                remaining = num_to_convert - total_converted
                batch_size = min(MAX_BATCH_SIZE, remaining)
                
                # Read batch (header already skipped)
                vectors, eof = read_bbin_batch(f_in, batch_size, dim, header_read=True)
                
                if vectors is None or eof:
                    break
                
                # Write batch
                append_fbin_vectors(args.output_file, vectors)
                
                total_converted += vectors.shape[0]
                batch_num += 1
                
                # Calculate current percentage
                current_percent = int(100.0 * total_converted / num_to_convert)
                
                # Print progress every 1% or when complete
                if current_percent > last_percent or total_converted == num_to_convert:
                    print(f"  Processed {total_converted:,} / {num_to_convert:,} vectors ({current_percent}%)")
                    last_percent = current_percent
        
        # Update header with actual number of vectors converted
        if total_converted != num_to_convert:
            write_fbin_header(args.output_file, total_converted, dim, update_only=True)
            print(f"Note: Converted {total_converted} vectors (less than requested)")
        
        print(f"\nConversion completed successfully!")
        print(f"  Total vectors converted: {total_converted:,}")
        print(f"  Dimension: {dim}")
        print(f"  Output file: {args.output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

# python3 convert_bbin_fbin.py input.bbin output.fbin
# python3 convert_bbin_fbin.py /ssd_root/dataset/sift/bigann_base.bbin /ssd_root/dataset/sift/bigann_base_100M.fbin -n 100000000  # Convert only first 100M vectors