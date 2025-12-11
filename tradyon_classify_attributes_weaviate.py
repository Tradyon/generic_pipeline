#!/usr/bin/env python3
"""
CLI tool to classify product attributes using Weaviate Hybrid Search (Vector + LLM).
"""

import argparse
import os
import sys
import logging
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Any

from src.classifiers.weaviate_attribute_classifier import WeaviateAttributeClassifier, AttributeClassificationResult
from src.classifiers.attribute_classifier.utils import TokenTracker, sanitize_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_flat_products(csv_path: str, low_memory: bool = False) -> pd.DataFrame:
    """Load and prepare classified shipment data."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path, low_memory=low_memory, dtype={'hs_code': str})
    required = ['hs_code', 'goods_shipped', 'category']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Input CSV missing columns: {missing}")
    
    df = df.copy()
    df['hs_code'] = df['hs_code'].astype(str).str.strip().str[:4].str.zfill(4)
    df['goods_shipped'] = df['goods_shipped'].astype(str).str.strip()
    df['category'] = df['category'].astype(str).str.strip()
    
    if 'shipment_id' in df.columns:
        df['shipment_id'] = df['shipment_id'].astype(str).str.strip()
    else:
        df['shipment_id'] = [str(i) for i in range(1, len(df) + 1)]
    
    df = df.dropna(subset=['goods_shipped'])
    return df[['hs_code', 'category', 'goods_shipped', 'shipment_id']]

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
    except Exception as e:
        logger.error(f'Failed to save checkpoint: {e}')

def main():
    parser = argparse.ArgumentParser(description="Classify product attributes using Weaviate Hybrid Search.")
    parser.add_argument("--input", required=True, help="Path to classified shipment CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for output files.")
    parser.add_argument("--product-attributes-schema", required=True, help="Path to product attributes schema JSON.")
    parser.add_argument("--attribute-definitions", required=True, help="Path to attribute definitions JSON.")
    parser.add_argument("--class-name", default="AttributeClassification", help="Weaviate class name.")
    parser.add_argument("--model", default="gemini-2.0-flash", help="LLM model name.")
    parser.add_argument("--items-per-call", type=int, default=10, help="Items per batch.")
    parser.add_argument("--token-budget", type=int, default=200_000_000, help="Max token budget.")
    parser.add_argument("--checkpoint-file", default="weaviate_attribute_checkpoint.json", help="Path to checkpoint file.")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from checkpoint.")
    parser.add_argument("--low-memory", action="store_true", help="Enable pandas low_memory mode.")
    parser.add_argument("--update-kb", action="store_true", help="Update knowledge base with new high-confidence classifications.")
    parser.add_argument("--min-similarity", type=float, default=0.95, help="Min similarity to update KB.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Classifier
    classifier = WeaviateAttributeClassifier(
        class_name=args.class_name,
        product_attributes_schema_path=args.product_attributes_schema,
        attribute_definitions_path=args.attribute_definitions,
        model_name=args.model,
        top_k=10,
        rerank_top_n=5
    )
    
    # Load Data
    logger.info("Loading data...")
    df = load_flat_products(args.input, low_memory=args.low_memory)
    
    # Group goods by hs4+product
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (hs4, product), sub in df.groupby(['hs_code', 'category']):
        sub2 = sub[['goods_shipped', 'shipment_id']].dropna(subset=['goods_shipped'])
        goods_to_shipments: Dict[str, List[str]] = {}
        for r in sub2.to_dict(orient='records'):
            g = r['goods_shipped']
            sid = r['shipment_id']
            if not isinstance(g, str) or not g.strip():
                continue
            goods_to_shipments.setdefault(g, []).append(str(sid))
        
        unique_goods = list(goods_to_shipments.keys())
        if unique_goods:
            grouped.setdefault(hs4, {})[product] = {
                'goods': sorted(unique_goods),
                'shipments': goods_to_shipments
            }
            
    # Checkpoint
    if args.no_resume:
        state = {
            'products': {},
            'token_usage': 0,
            'timestamp': datetime.now().isoformat(),
            'version': 2
        }
    else:
        state = load_checkpoint(args.checkpoint_file)
        
    tracker = TokenTracker(args.token_budget)
    tracker.used = int(state.get('token_usage', 0))
    
    # Process
    hs_list = sorted(grouped.keys())
    
    for hs4 in hs_list:
        products_items = list(grouped[hs4].items())
        for product, pdata in products_items:
            key = f"{hs4}::{product}"
            prod_state = state['products'].setdefault(key, {'completed': 0, 'classifications': []})
            
            goods_list = pdata['goods']
            shipments_map = pdata['shipments']
            
            # Filter already done
            done_goods = {c['goods_shipped'] for c in prod_state['classifications']}
            remaining_goods = [g for g in goods_list if g not in done_goods]
            
            if not remaining_goods:
                logger.info(f"Product {product} ({hs4}) already completed.")
                continue
                
            logger.info(f"Processing {product} ({hs4}): {len(remaining_goods)} goods remaining")
            
            total_batches = (len(remaining_goods) + args.items_per_call - 1) // args.items_per_call
            
            for b_start in range(0, len(remaining_goods), args.items_per_call):
                if tracker.remaining() <= 0:
                    logger.warning('Token budget exhausted.')
                    break
                
                batch_goods = remaining_goods[b_start:b_start+args.items_per_call]
                batch_num = (b_start // args.items_per_call) + 1
                
                logger.info(f"Batch {batch_num}/{total_batches} for {product}")
                
                # Hybrid Classification
                results = classifier.classify_batch_hybrid(
                    goods_list=batch_goods,
                    hs_code=hs4,
                    category=product,
                    token_tracker=tracker,
                    batch_num=batch_num,
                    total_batches=total_batches,
                    config={}
                )
                
                # Update Knowledge Base if requested
                if args.update_kb:
                    classifier.store_new_classifications(
                        [r for r in results if r is not None], 
                        min_similarity=args.min_similarity
                    )
                
                # Save results
                for res in results:
                    if res is None: continue
                    
                    # Convert to dict for JSON serialization
                    res_dict = {
                        'goods_shipped': res.goods_shipped,
                        'attributes': res.attributes,
                        'similarity_scores': res.similarity_scores,
                        'method': res.method,
                        'metadata': res.metadata,
                        'shipment_ids': shipments_map.get(res.goods_shipped, [])
                    }
                    prod_state['classifications'].append(res_dict)
                
                prod_state['completed'] = len(prod_state['classifications'])
                state['token_usage'] = tracker.usage()
                save_checkpoint(args.checkpoint_file, state)
                
            # Save per-product CSV
            safe_prod = sanitize_filename(product)
            csv_name = f"hs_{hs4}_{safe_prod}.csv"
            csv_path = os.path.join(args.output_dir, "per_product_classifications", csv_name)
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
            rows = []
            for c in prod_state['classifications']:
                shipments = c.get('shipment_ids', [])
                all_ids = "|".join(shipments)
                
                for sid in shipments:
                    base = {
                        'hs_code': hs4,
                        'product': product,
                        'goods_shipped': c['goods_shipped'],
                        'shipment_id': sid,
                        'shipment_ids': all_ids,
                        'method': c.get('method', 'unknown')
                    }
                    for k, v in c['attributes'].items():
                        base[f"attr_{k}"] = v
                    rows.append(base)
            
            if rows:
                pd.DataFrame(rows).to_csv(csv_path, index=False)
    
    # Save aggregated outputs
    output_json = os.path.join(args.output_dir, "classifications.json")
    output_csv = os.path.join(args.output_dir, "classifications_flat.csv")
    
    all_rows = []
    all_json = {}
    
    for key, p_state in state['products'].items():
        if '::' in key:
            hs4, product = key.split('::', 1)
        else:
            hs4, product = "0000", key
        
        for c in p_state['classifications']:
            all_json.setdefault(hs4, {}).setdefault(product, []).append(c)
            
            base = {
                'hs_code': hs4,
                'product': product,
                'goods_shipped': c['goods_shipped'],
                'shipment_ids': "|".join(c.get('shipment_ids', [])),
                'method': c.get('method', 'unknown')
            }
            for k, v in c['attributes'].items():
                base[f"attr_{k}"] = v
            all_rows.append(base)
            
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_json, f, indent=2)
        
    if all_rows:
        pd.DataFrame(all_rows).to_csv(output_csv, index=False)
        
    # Print stats
    stats = classifier.get_stats()
    logger.info("Classification Stats:")
    logger.info(json.dumps(stats, indent=2))
    
    # Close connection
    classifier.close()

if __name__ == "__main__":
    main()
