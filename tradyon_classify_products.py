#!/usr/bin/env python3
"""
CLI tool to classify products using LLM.
"""

import argparse
import os
import sys
import logging
from src.classifiers.product_classifier import classify_products

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    parser = argparse.ArgumentParser(description="Classify products using LLM.")
    parser.add_argument("--input", required=True, help="Path to input CSV file (e.g., shipment_master.csv).")
    parser.add_argument("--output", required=True, help="Path to output CSV file (e.g., shipment_master_classified.csv).")
    parser.add_argument("--products-definition", required=True, help="Path to products definition JSON.")
    parser.add_argument("--model", default="gemini-2.0-flash", help="LLM model name.")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size.")
    parser.add_argument("--max-workers", type=int, default=8, help="Max workers.")
    parser.add_argument("--batch-timeout", type=int, default=300, help="Seconds to wait per batch before marking as Unclassified.")
    parser.add_argument("--checkpoint-file", help="Path to checkpoint file.")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from checkpoint.")
    parser.add_argument("--low-memory", action="store_true", help="Enable pandas low_memory mode (default: False).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
        
    if not os.path.exists(args.products_definition):
        print(f"Error: Products definition file not found: {args.products_definition}")
        sys.exit(1)
        
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    try:
        classify_products(
            input_csv=args.input,
            products_definition_path=args.products_definition,
            output_csv=args.output,
            checkpoint_file=args.checkpoint_file,
            model_name=args.model,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            batch_timeout=args.batch_timeout,
            resume=not args.no_resume,
            low_memory=args.low_memory
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
