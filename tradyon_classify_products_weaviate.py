#!/usr/bin/env python3
"""
CLI tool to classify products using Weaviate Hybrid Search (Vector + LLM).
"""

import argparse
import os
import sys
import logging
import pandas as pd
from src.classifiers.weaviate_product_classifier import WeaviateProductClassifier
from src.utils.config_models import load_product_definition

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    parser = argparse.ArgumentParser(description="Classify products using Weaviate Hybrid Search.")
    parser.add_argument("--input", required=True, help="Path to input CSV file (e.g., shipment_master.csv).")
    parser.add_argument("--output", required=True, help="Path to output CSV file (e.g., shipment_master_classified.csv).")
    parser.add_argument("--products-definition", required=True, help="Path to products definition JSON.")
    parser.add_argument("--class-name", default="ProductClassification", help="Weaviate class name.")
    parser.add_argument("--model", default="gemini-2.0-flash", help="LLM model name.")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size.")
    parser.add_argument("--max-workers", type=int, default=8, help="Max workers.")
    parser.add_argument("--batch-timeout", type=int, default=900, help="Seconds to wait per batch before marking as Unclassified.")
    parser.add_argument("--checkpoint-file", help="Path to checkpoint file.")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from checkpoint.")
    parser.add_argument("--low-memory", action="store_true", help="Enable pandas low_memory mode.")
    parser.add_argument("--similarity-threshold", type=float, default=0.85, help="Similarity threshold for vector match.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking.")
    
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
        # Load product definition
        products_def = load_product_definition(args.products_definition)
        
        # Initialize classifier
        classifier = WeaviateProductClassifier(
            products_definition=products_def,
            class_name=args.class_name,
            model_name=args.model,
            similarity_threshold=args.similarity_threshold,
            checkpoint_file=args.checkpoint_file,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            batch_timeout=args.batch_timeout,
            use_reranking=not args.no_rerank
        )
        
        # Load input data
        df = pd.read_csv(args.input, low_memory=args.low_memory)
        if 'goods_shipped' not in df.columns:
            raise ValueError("Input CSV must contain 'goods_shipped' column")
            
        # Run classification
        result_df = classifier.classify_dataframe(
            df=df,
            resume=not args.no_resume,
            low_memory=args.low_memory
        )
        
        # Save output
        result_df.to_csv(args.output, index=False)
        print(f"Classification complete. Results saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
