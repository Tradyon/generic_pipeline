#!/usr/bin/env python3
"""
Populate Weaviate with existing attribute classifications.

This script:
1. Reads existing attribute classification results (JSON or CSV)
2. Upserts attribute-goods mappings to Weaviate with local embeddings
3. Uses multilingual-e5-large for better multilingual support

Schema:
- Class: AttributeClassification
- Properties: goods_shipped, attribute_name, attribute_value, hs_code, category, etc.

This enables precise attribute-specific vector search with high accuracy using filters.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Any, Set
from collections import defaultdict
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).parent.parent / '.env'
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

load_env_file()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.weaviate_client import WeaviateClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_classifications_from_csv(csv_path: str, text_field: str = 'goods_shipped') -> List[Dict[str, Any]]:
    """
    Load classifications from CSV file.
    Supports both 'flat' classification CSVs (with attr_ columns) and raw shipment CSVs.
    """
    df = pd.read_csv(csv_path)
    records = []
    
    # Identify attribute columns (starting with attr_)
    attr_cols = [c for c in df.columns if c.startswith('attr_')]
    
    for _, row in df.iterrows():
        base_record = {
            'goods_shipped': row.get(text_field, row.get('goods_description', '')),
            'hs_code': str(row.get('hs_code', '')),
            'category': row.get('product', 'Uncategorized'),
            '_id': str(row.get('shipment_id', '')) or str(row.get('shipment_ids', ''))
        }
        
        if not base_record['goods_shipped']:
            continue
            
        if attr_cols:
            # If we have attribute columns, create a record for each non-null attribute
            for col in attr_cols:
                val = row[col]
                if pd.notna(val) and val != 'None' and str(val).strip():
                    record = base_record.copy()
                    record['attribute_name'] = col.replace('attr_', '')
                    record['attribute_value'] = str(val)
                    records.append(record)
        else:
            # Raw CSV without attributes - just index the text
            # We create a dummy attribute to ensure it's indexed
            record = base_record.copy()
            record['attribute_name'] = 'RawText'
            record['attribute_value'] = 'True'
            records.append(record)
            
    return records

def load_classifications_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Load attribute classifications from JSON file.
    
    JSON structure:
    {
        "hs4_product": {
            "goods_description": {
                "attribute1": "value1",
                "attribute2": "value2",
                ...
            }
        }
    }
    
    Returns:
        List of classification records
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    for hs4, products_dict in data.items():
        # Check if the structure is nested {hs4: {product: ...}} or flat {hs4_product: ...}
        
        # Case 1: Flat structure {hs4_product: ...}
        if '_' in hs4 and not isinstance(products_dict, dict):
             # This matches the old logic, but let's be robust
             pass

        # Case 2: Nested structure {hs4: {product: [list of items]}}
        # Based on the user's file content: "1006": { "Aromatic Long Grain Rice...": [ ... ] }
        if isinstance(products_dict, dict):
            for product, items_list in products_dict.items():
                # items_list is a list of dicts based on the file content
                if isinstance(items_list, list):
                    for item in items_list:
                        if not isinstance(item, dict): continue
                        
                        goods_shipped = item.get('goods_shipped')
                        attributes = item.get('attributes')
                        
                        if goods_shipped and attributes:
                            records.append({
                                'hs_code': hs4,
                                'category': product,
                                'goods_shipped': goods_shipped,
                                'attributes': attributes
                            })
                # Handle case where it might be {product: {goods: attributes}} (old format?)
                elif isinstance(items_list, dict):
                     for goods_shipped, attributes in items_list.items():
                        if isinstance(attributes, dict):
                            records.append({
                                'hs_code': hs4,
                                'category': product,
                                'goods_shipped': goods_shipped,
                                'attributes': attributes
                            })
        
        # Case 3: Old flat key format fallback
        elif '_' in hs4:
             parts = hs4.split('_', 1)
             if len(parts) == 2:
                 real_hs4, real_product = parts
                 # ... logic for flat format if needed, but the file shows nested
                 pass
    
    return records






def prepare_weaviate_records(
    classifications: List[Dict[str, Any]],
    text_field: str = "goods_shipped"
) -> List[Dict[str, Any]]:
    """
    Prepare records for Weaviate upsert.
    
    Args:
        classifications: List of classification records
        text_field: Name of text field for embeddings
    
    Returns:
        List of records ready for Weaviate
    """
    weaviate_records = []
    
    # Track unique goods per attribute to avoid duplicates
    seen_combinations = set()
    
    for item in classifications:
        hs_code = item['hs_code']
        category = item['category']
        goods_shipped = item['goods_shipped']
        attributes = item['attributes']
        
        # Create a record for each attribute separately
        for attr_name, attr_value in attributes.items():
            # Skip None values
            if attr_value in ['None', 'none', '', None]:
                continue
            
            # Check if we've seen this combination before
            # Key: hs_code|category|attribute|value|goods
            combination_key = f"{hs_code}|{category}|{attr_name}|{attr_value}|{goods_shipped}"
            if combination_key in seen_combinations:
                continue
            seen_combinations.add(combination_key)
            
            # Create record
            record = {
                text_field: goods_shipped,
                "attribute_name": attr_name,
                "attribute_value": attr_value,
                "hs_code": hs_code,
                "category": category,
                "confidence": 1.0,  # Assume 1.0 for manual/existing data
                "classification_method": "manual_load",
                "_id": f"{hs_code}_{category}_{attr_name}_{hash(goods_shipped + attr_value)}"
            }
            
            weaviate_records.append(record)
            
    return weaviate_records


def main():
    parser = argparse.ArgumentParser(description='Populate Weaviate with attribute classifications')
    parser.add_argument('--input', required=True, help='Input JSON or CSV file with classifications')
    parser.add_argument('--class-name', default='AttributeClassification', help='Weaviate class name')
    parser.add_argument('--text-field', default='goods_shipped', help='Name of text field')
    parser.add_argument('--batch-size', type=int, default=25, help='Batch size for upsert')
    parser.add_argument('--embedding-model', default='intfloat/multilingual-e5-large', help='Embedding model name')
    parser.add_argument('--recreate', action='store_true', help='Recreate the class if it exists')
    
    args = parser.parse_args()
    
    # Initialize Weaviate client
    logger.info(f"Initializing Weaviate client for class '{args.class_name}'")
    client = WeaviateClient(
        class_name=args.class_name,
        text_field=args.text_field,
        embedding_model_name=args.embedding_model
    )

    if args.recreate:
        logger.info(f"Deleting existing class '{args.class_name}'...")
        client.client.collections.delete(args.class_name)
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    if args.input.endswith('.json'):
        classifications = load_classifications_from_json(args.input)
        logger.info(f"Loaded {len(classifications)} classification groups")
        
        # Prepare records for Weaviate
        logger.info("Preparing records for Weaviate...")
        records = prepare_weaviate_records(classifications, args.text_field)
    elif args.input.endswith('.csv'):
        records = load_classifications_from_csv(args.input, args.text_field)
    else:
        logger.error("Input file must be .json or .csv")
        sys.exit(1)
        
    logger.info(f"Prepared {len(records)} individual attribute records")
    
    if not records:
        logger.warning("No records to upsert")
        sys.exit(0)
        
    # Upsert to Weaviate
    logger.info(f"Upserting records to Weaviate (batch size {args.batch_size})...")
    
    # Pass all records to the client and let it handle batching/multithreading
    client.upsert_records(records, batch_size=args.batch_size)
        
    logger.info("Population complete!")
    client.close()


if __name__ == "__main__":
    main()
