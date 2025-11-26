#!/usr/bin/env python3
"""
CLI tool for post-processing (schema master, combining attributes).
"""

import argparse
import os
import sys
import logging
from src.utils.post_processing import build_schema_master, combine_shipment_attributes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    parser = argparse.ArgumentParser(description="Post-process classification results.")
    parser.add_argument("--input-dir", required=True, help="Directory containing per-product CSV files.")
    parser.add_argument("--output-dir", required=True, help="Directory for output files.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 1. Build Product Schema Master
        product_schema_master_csv = os.path.join(args.output_dir, "product_schema_master.csv")
        print("Building product schema master...")
        build_schema_master(
            input_dir=args.input_dir,
            output_csv=product_schema_master_csv
        )
        
        # 2. Combine Shipment Attributes (JSON)
        shipment_id_to_attr_json = os.path.join(args.output_dir, "shipment_id_to_attr.json")
        print("Combining shipment attributes (JSON)...")
        combine_shipment_attributes(
            input_dir=args.input_dir,
            pattern="*.csv",
            output_path=shipment_id_to_attr_json
        )
        
        # 3. Combine Shipment Attributes (CSV)
        shipment_id_to_attr_csv = os.path.join(args.output_dir, "shipment_id_to_attr.csv")
        print("Combining shipment attributes (CSV)...")
        combine_shipment_attributes(
            input_dir=args.input_dir,
            pattern="*.csv",
            output_path=shipment_id_to_attr_csv
        )
        
        print("Post-processing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
