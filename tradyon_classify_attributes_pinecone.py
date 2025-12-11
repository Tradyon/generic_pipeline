#!/usr/bin/env python3
"""
CLI tool to classify product attributes using Pinecone hybrid search with LLM fallback.

This tool provides:
1. Multilingual vector search with multilingual-e5-large
2. Per-attribute granular matching for precision
3. Reranking for improved accuracy
4. LLM fallback for new/unknown products
5. Knowledge base updates with high-confidence results
6. Batch processing with checkpointing
"""

import argparse
import os
import sys
import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Handle both 'export VAR=val' and 'VAR=val' formats
                    if line.startswith('export '):
                        line = line[7:]
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        os.environ[key] = value
                        
                        # Map GOOGLE_API_KEY to GEMINI_API_KEY if needed
                        if key == 'GOOGLE_API_KEY' and 'GEMINI_API_KEY' not in os.environ:
                            os.environ['GEMINI_API_KEY'] = value

load_env_file()

from src.classifiers.pinecone_attribute_classifier import (
    PineconeAttributeClassifier,
    AttributeClassificationResult
)
from src.classifiers.attribute_classifier.utils import TokenTracker, sanitize_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    """Load classification checkpoint."""
    if not os.path.exists(checkpoint_file):
        return {
            'products': {},
            'token_usage': 0,
            'timestamp': datetime.now().isoformat(),
            'version': 2
        }
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'products' not in data:
            data['products'] = {}
        logger.info(f"Loaded checkpoint with {len(data['products'])} products")
        return data
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return {
            'products': {},
            'token_usage': 0,
            'timestamp': datetime.now().isoformat(),
            'version': 2
        }


def save_checkpoint(checkpoint_file: str, state: Dict[str, Any]):
    """Save classification checkpoint."""
    try:
        state['timestamp'] = datetime.now().isoformat()
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        logger.debug('Checkpoint saved')
    except Exception as e:
        logger.error(f'Failed to save checkpoint: {e}')


def classify_product_group(
    hs_code: str,
    category: str,
    goods_to_shipments: Dict[str, List[str]],
    classifier: PineconeAttributeClassifier,
    token_tracker: TokenTracker,
    config: Dict[str, Any],
    checkpoint_state: Dict[str, Any],
    checkpoint_file: Optional[str]
) -> Dict[str, Any]:
    """
    Classify attributes for a single HS code + product category group.
    
    Args:
        hs_code: 4-digit HS code
        category: Product category name
        goods_to_shipments: Dict mapping goods description to list of shipment IDs
        classifier: PineconeAttributeClassifier instance
        token_tracker: Token usage tracker
        config: Configuration dictionary
        checkpoint_state: Checkpoint state dictionary
        checkpoint_file: Path to checkpoint file
    
    Returns:
        Dictionary with classification results for this product group
    """
    product_key = f"{hs_code}_{category}"
    
    # Check if already processed
    if product_key in checkpoint_state.get('products', {}):
        logger.info(f"Skipping already processed: {product_key}")
        return checkpoint_state['products'][product_key]
    
    goods_list = list(goods_to_shipments.keys())
    logger.info(f"Classifying {len(goods_list)} unique goods for {product_key}")
    
    # Process in batches
    items_per_call = config.get('items_per_call', 10)
    all_results = []
    
    total_batches = (len(goods_list) + items_per_call - 1) // items_per_call
    
    for batch_num in range(total_batches):
        start_idx = batch_num * items_per_call
        end_idx = min(start_idx + items_per_call, len(goods_list))
        batch_goods = goods_list[start_idx:end_idx]
        
        # Classify batch using hybrid approach
        batch_results = classifier.classify_batch_hybrid(
            goods_list=batch_goods,
            hs_code=hs_code,
            category=category,
            token_tracker=token_tracker,
            batch_num=batch_num + 1,
            total_batches=total_batches,
            config=config
        )
        
        all_results.extend(batch_results)
        
        # Update token usage
        checkpoint_state['token_usage'] = token_tracker.used
        
        # Save checkpoint periodically
        if checkpoint_file and (batch_num + 1) % 5 == 0:
            save_checkpoint(checkpoint_file, checkpoint_state)
    
    # Store high-confidence results back to Pinecone
    if config.get('store_new_classifications', True):
        classifier.store_new_classifications(
            all_results,
            min_similarity=config.get('min_similarity_for_storage', 0.95)
        )
    
    # Convert results to dictionary format
    product_results = {}
    for result in all_results:
        product_results[result.goods_shipped] = result.attributes
    
    # Save to checkpoint
    checkpoint_state['products'][product_key] = product_results
    if checkpoint_file:
        save_checkpoint(checkpoint_file, checkpoint_state)
    
    return product_results


def save_results(
    results: Dict[str, Dict[str, Dict[str, str]]],
    output_json: str,
    output_csv: str,
    output_folder: str,
    shipment_data: pd.DataFrame
):
    """
    Save classification results to JSON, CSV, and per-product CSV files.
    
    Args:
        results: Nested dict {product_key: {goods: {attr: value}}}
        output_json: Path for JSON output
        output_csv: Path for flat CSV output
        output_folder: Folder for per-product CSVs
        shipment_data: Original shipment dataframe
    """
    # Save JSON
    logger.info(f"Saving results to {output_json}")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Build flat CSV
    rows = []
    for product_key, goods_dict in results.items():
        parts = product_key.split('_', 1)
        if len(parts) != 2:
            continue
        hs_code, category = parts
        
        for goods_shipped, attributes in goods_dict.items():
            row = {
                'hs_code': hs_code,
                'category': category,
                'goods_shipped': goods_shipped,
                **attributes
            }
            rows.append(row)
    
    if rows:
        df_flat = pd.DataFrame(rows)
        logger.info(f"Saving flat CSV to {output_csv}")
        df_flat.to_csv(output_csv, index=False, encoding='utf-8')
    
    # Save per-product CSVs
    os.makedirs(output_folder, exist_ok=True)
    
    for product_key, goods_dict in results.items():
        parts = product_key.split('_', 1)
        if len(parts) != 2:
            continue
        hs_code, category = parts
        
        # Match with shipment data
        product_shipments = shipment_data[
            (shipment_data['hs_code'] == hs_code) &
            (shipment_data['category'] == category)
        ].copy()
        
        # Add attributes
        for col in goods_dict.get(list(goods_dict.keys())[0], {}).keys():
            product_shipments[col] = product_shipments['goods_shipped'].map(
                lambda g: goods_dict.get(g, {}).get(col, 'None')
            )
        
        # Save
        filename = sanitize_filename(f"hs_{hs_code}_{category}.csv")
        filepath = os.path.join(output_folder, filename)
        product_shipments.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Saved {len(product_shipments)} records to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Classify product attributes using Pinecone hybrid search"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to classified shipment CSV (with 'category' column)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output files"
    )
    parser.add_argument(
        "--product-attributes-schema",
        required=True,
        help="Path to product attributes schema JSON"
    )
    parser.add_argument(
        "--attribute-definitions",
        required=True,
        help="Path to attribute definitions JSON"
    )
    parser.add_argument(
        "--pinecone-index",
        default="attribute-classification",
        help="Pinecone index name (default: attribute-classification)"
    )
    parser.add_argument(
        "--embedding-model",
        default="multilingual-e5-large",
        help="Embedding model (default: multilingual-e5-large for multilingual)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for vector matches (default: 0.75)"
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="LLM model for fallback (default: gemini-2.0-flash)"
    )
    parser.add_argument(
        "--items-per-call",
        type=int,
        default=10,
        help="Items per LLM call (default: 10)"
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=200_000_000,
        help="Max token budget (default: 200M)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="Max parallel workers (default: 20)"
    )
    parser.add_argument(
        "--checkpoint-file",
        help="Path to checkpoint file (default: <output-dir>/checkpoint.json)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint"
    )
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="Don't store new classifications to Pinecone"
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable pandas low_memory mode"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    if not os.path.exists(args.product_attributes_schema):
        logger.error(f"Product attributes schema not found: {args.product_attributes_schema}")
        sys.exit(1)
    if not os.path.exists(args.attribute_definitions):
        logger.error(f"Attribute definitions not found: {args.attribute_definitions}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set checkpoint file
    checkpoint_file = args.checkpoint_file
    if not checkpoint_file:
        checkpoint_file = os.path.join(args.output_dir, "checkpoint.json")
    
    # Load checkpoint
    checkpoint_state = {} if args.no_resume else load_checkpoint(checkpoint_file)
    
    # Configuration
    config = {
        'model_name': args.model,
        'items_per_call': args.items_per_call,
        'max_token_budget': args.token_budget,
        'max_workers': args.max_workers,
        'temperature': 0.0,
        'use_structured_output': True,
        'deterministic_first_pass': True,
        'enable_heuristic_fill': True,
        'allow_out_of_schema_values': False,
        'store_new_classifications': not args.no_store,
        'min_similarity_for_storage': 0.95,
    }
    
    logger.info("=== Pinecone Attribute Classification ===")
    logger.info(f"Input CSV: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Pinecone index: {args.pinecone_index}")
    logger.info(f"Embedding model: {args.embedding_model}")
    logger.info(f"Similarity threshold: {args.similarity_threshold}")
    logger.info(f"LLM model: {args.model}")
    logger.info(f"Store new classifications: {not args.no_store}")
    
    # Load shipment data
    logger.info("Loading shipment data...")
    df = pd.read_csv(args.input, low_memory=args.low_memory, dtype={'hs_code': str})
    
    required = ['hs_code', 'goods_shipped', 'category']
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Input CSV missing columns: {missing}")
        sys.exit(1)
    
    # Normalize data
    df['hs_code'] = df['hs_code'].astype(str).str.strip().str[:4].str.zfill(4)
    df['goods_shipped'] = df['goods_shipped'].astype(str).str.strip()
    df['category'] = df['category'].astype(str).str.strip()
    df = df.dropna(subset=['goods_shipped'])
    
    logger.info(f"Loaded {len(df)} shipment records")
    logger.info(f"Unique goods descriptions: {df['goods_shipped'].nunique()}")
    
    # Initialize classifier
    logger.info("Initializing Pinecone classifier...")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        logger.error("PINECONE_API_KEY environment variable not set")
        sys.exit(1)
    
    # LLMClient will auto-load GOOGLE_API_KEY from .env for Gemini models
    classifier = PineconeAttributeClassifier(
        index_name=args.pinecone_index,
        product_attributes_schema_path=args.product_attributes_schema,
        attribute_definitions_path=args.attribute_definitions,
        model_name=args.model,
        api_key=None,  # Let LLMClient load from .env
        embedding_model=args.embedding_model,
        similarity_threshold=args.similarity_threshold
    )
    
    # Initialize token tracker
    token_tracker = TokenTracker(config['max_token_budget'])
    
    # Group by HS code + category
    logger.info("Grouping goods by HS code and category...")
    product_groups = []
    
    for (hs_code, category), group_df in df.groupby(['hs_code', 'category']):
        goods_to_shipments = {}
        for goods in group_df['goods_shipped'].unique():
            if pd.notna(goods) and str(goods).strip():
                goods_to_shipments[str(goods)] = []
        
        if goods_to_shipments:
            product_groups.append({
                'hs_code': hs_code,
                'category': category,
                'goods_to_shipments': goods_to_shipments,
                'count': len(goods_to_shipments)
            })
    
    logger.info(f"Found {len(product_groups)} product groups to classify")
    for group in product_groups:
        logger.info(f"  {group['hs_code']}_{group['category']}: {group['count']} unique goods")
    
    # Process product groups
    logger.info("Starting classification...")
    start_time = time.time()
    
    all_results = checkpoint_state.get('products', {})
    
    with tqdm(total=len(product_groups), desc="Product groups") as pbar:
        for group in product_groups:
            product_key = f"{group['hs_code']}_{group['category']}"
            
            if product_key in all_results:
                logger.info(f"Skipping already processed: {product_key}")
                pbar.update(1)
                continue
            
            try:
                results = classify_product_group(
                    hs_code=group['hs_code'],
                    category=group['category'],
                    goods_to_shipments=group['goods_to_shipments'],
                    classifier=classifier,
                    token_tracker=token_tracker,
                    config=config,
                    checkpoint_state=checkpoint_state,
                    checkpoint_file=checkpoint_file
                )
                
                all_results[product_key] = results
                
            except Exception as e:
                logger.error(f"Error processing {product_key}: {e}", exc_info=True)
            
            pbar.update(1)
    
    elapsed = time.time() - start_time
    
    # Save results
    output_json = os.path.join(args.output_dir, "classifications.json")
    output_csv = os.path.join(args.output_dir, "classifications_flat.csv")
    per_product_dir = os.path.join(args.output_dir, "per_product_classifications")
    
    save_results(all_results, output_json, output_csv, per_product_dir, df)
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info("Checkpoint file removed after successful completion")
    
    # Print statistics
    stats = classifier.get_stats()
    
    logger.info(f"\n=== Classification Complete ===")
    logger.info(f"Time elapsed: {elapsed:.1f}s")
    logger.info(f"Total classifications: {stats['total_classifications']}")
    logger.info(f"Vector matches: {stats['vector_matches']} ({stats['vector_match_rate']*100:.1f}%)")
    logger.info(f"LLM fallbacks: {stats['llm_fallbacks']} ({stats['llm_fallback_rate']*100:.1f}%)")
    logger.info(f"Total LLM tokens: {token_tracker.used}")
    logger.info(f"Cost savings: {stats['vector_match_rate']*100:.1f}% (vector matches avoided LLM calls)")
    
    logger.info("\nClassification completed successfully!")


if __name__ == "__main__":
    main()
