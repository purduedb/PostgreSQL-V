#!/usr/bin/env python3
"""
Convert ground truth file from custom binary format to .ivecs format.

The custom format is:
- 4 bytes: int32 npts (number of queries)
- 4 bytes: int32 ndims (number of neighbors per query)
- npts * ndims * 4 bytes: uint32_t IDs (npts rows, ndims columns)
- npts * ndims * 4 bytes: float distances (npts rows, ndims columns)

The .ivecs format is:
- For each query:
  - 4 bytes: int32 k (number of neighbors)
  - k * 4 bytes: k int32 neighbor IDs
"""

import struct
import numpy as np
import argparse
import sys


def convert_groundtruth_to_ivecs(input_file, output_file):
    """
    Convert ground truth from custom binary format to .ivecs format.
    
    Args:
        input_file: Path to input file in custom binary format
        output_file: Path to output .ivecs file
    """
    print(f"Reading ground truth from: {input_file}")
    
    with open(input_file, "rb") as f:
        # Read npts and ndims as 32-bit integers
        header = f.read(8)
        if len(header) < 8:
            raise ValueError("Invalid ground truth file: header too short")
        
        npts, ndims = struct.unpack("ii", header)
        print(f"Found {npts} queries with {ndims} neighbors each")
        
        # Read npts * ndims uint32_t (ground truth ids)
        num_ids = npts * ndims
        id_data = f.read(num_ids * 4)  # 4 bytes per uint32
        if len(id_data) < num_ids * 4:
            raise ValueError(f"Invalid ground truth file: expected {num_ids * 4} bytes for IDs, got {len(id_data)}")
        
        ids = np.frombuffer(id_data, dtype=np.uint32).reshape(npts, ndims)
        
        # Read distances (we don't need them for .ivecs, but we verify the file structure)
        dist_data = f.read(num_ids * 4)  # 4 bytes per float
        if len(dist_data) < num_ids * 4:
            raise ValueError(f"Invalid ground truth file: expected {num_ids * 4} bytes for distances, got {len(dist_data)}")
        
        # Check if there's more data (shouldn't be)
        remaining = f.read(1)
        if remaining:
            print("Warning: File contains more data than expected")
    
    print(f"Writing .ivecs format to: {output_file}")
    
    with open(output_file, "wb") as f:
        for i in range(npts):
            # Write ndims as int32
            f.write(struct.pack("i", ndims))
            
            # Write the ndims neighbor IDs as int32
            # Convert uint32 to int32 (should be safe for valid IDs)
            neighbor_ids = ids[i].astype(np.int32)
            f.write(neighbor_ids.tobytes())
    
    print(f"Successfully converted {npts} queries to .ivecs format")
    print(f"Output file: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ground truth file from custom binary format to .ivecs format"
    )
    parser.add_argument(
        "input_file",
        help="Path to input ground truth file in custom binary format"
    )
    parser.add_argument(
        "output_file",
        help="Path to output .ivecs file"
    )
    
    args = parser.parse_args()
    
    try:
        convert_groundtruth_to_ivecs(args.input_file, args.output_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

