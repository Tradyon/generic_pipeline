#!/usr/bin/env python3
"""
CLI tool to classify product attributes using LLM.
"""

import argparse
import os
import sys
import logging
from src.classifiers.attribute_classifier.classifier import classify_attributes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    parser = argparse.ArgumentParser(description="Classify product attributes using LLM.")
    parser.add_argument("--input", required=True, help="Path to classified shipment CSV (e.g., shipment_master_classified.csv).")
    parser.add_argument("--output-dir", required=True, help="Directory for output files.")
    parser.add_argument("--product-attributes-schema", required=True, help="Path to product attributes schema JSON.")
    parser.add_argument("--attribute-definitions", required=True, help="Path to attribute definitions JSON.")
    parser.add_argument("--model", default="gemini-2.0-flash", help="LLM model name.")
    parser.add_argument("--items-per-call", type=int, default=10, help="Items per LLM call.")
    parser.add_argument("--token-budget", type=int, default=200_000_000, help="Max token budget.")
    parser.add_argument("--max-workers", type=int, default=20, help="Max parallel workers across products.")
    parser.add_argument("--checkpoint-file", help="Path to checkpoint file.")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from checkpoint.")
    parser.add_argument("--low-memory", action="store_true", help="Enable pandas low_memory mode (default: False).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)
    if not os.path.exists(args.product_attributes_schema):
        logging.error(f"Product attributes schema not found: {args.product_attributes_schema}")
        sys.exit(1)
    if not os.path.exists(args.attribute_definitions):
        logging.error(f"Attribute definitions not found: {args.attribute_definitions}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Config overrides
    config = {
        'model_name': args.model,
        'items_per_call': args.items_per_call,
        'max_token_budget': args.token_budget,
        'max_workers': args.max_workers,
    }
    
    # Define output paths
    output_json = os.path.join(args.output_dir, "classifications.json")
    output_csv = os.path.join(args.output_dir, "classifications_flat.csv")
    per_product_dir = os.path.join(args.output_dir, "per_product_classifications")
    
    try:
        classify_attributes(
            input_csv=args.input,
            product_attributes_schema_path=args.product_attributes_schema,
            attribute_definitions_path=args.attribute_definitions,
            output_json=output_json,
            output_csv=output_csv,
            output_folder=per_product_dir,
            checkpoint_file=args.checkpoint_file,
            config=config,
            resume=not args.no_resume,
            low_memory=args.low_memory
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
