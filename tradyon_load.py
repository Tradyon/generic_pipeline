#!/usr/bin/env python3
"""
CLI tool to load and validate shipment data.
"""

import argparse
import os
import sys
from src.utils.data_loader import load_data, validate_shipment_data

def main():
    parser = argparse.ArgumentParser(description="Load and validate shipment data.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=True, help="Path to output CSV file (e.g., shipment_master.csv).")
    parser.add_argument("--low-memory", action="store_true", help="Enable pandas low_memory mode (default: False).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
        
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    try:
        print(f"Loading data from {args.input}...")
        df = load_data(args.input, args.output, low_memory=args.low_memory)
        print("Validating data...")
        validate_shipment_data(df)
        print(f"Success! Data loaded to {args.output}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
