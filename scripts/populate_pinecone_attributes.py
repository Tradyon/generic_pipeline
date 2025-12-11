#!/usr/bin/env python3
"""
Populate Pinecone index with existing attribute classifications.

This script:
1. Reads existing attribute classification results (JSON or CSV)
2. Creates per-attribute namespaces for granular search
3. Upserts attribute-goods mappings to Pinecone with integrated embeddings
4. Uses multilingual-e5-large for better multilingual support

Namespace structure: hs_<code>_<product>_<attribute>
Example: hs_0901_Bulk_Commodity_Green_Coffee_Coffee_Variety

This enables precise attribute-specific vector search with high accuracy.
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
                        
                        # Map GOOGLE_API_KEY to GEMINI_API_KEY if needed
                        if key == 'GOOGLE_API_KEY' and 'GEMINI_API_KEY' not in os.environ:
                            os.environ['GEMINI_API_KEY'] = value

load_env_file()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.pinecone_client import PineconeClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    for key, goods_dict in data.items():
        # Parse hs4_product key
        parts = key.split('_', 1)
        if len(parts) != 2:
            logger.warning(f"Skipping invalid key format: {key}")
            continue
        
        hs4, product = parts
        
        for goods_shipped, attributes in goods_dict.items():
            if not isinstance(attributes, dict):
                continue
            
            records.append({
                'hs_code': hs4,
                'category': product,
                'goods_shipped': goods_shipped,
                'attributes': attributes
            })
    
    return records


def load_classifications_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load attribute classifications from flat CSV file.
    
    CSV columns: hs_code, category/product, goods_shipped, attribute1, attribute2, ...
    
    Returns:
        List of classification records
    """
    df = pd.read_csv(csv_path, dtype={'hs_code': str})
    
    # Handle both 'category' and 'product' column names
    if 'product' in df.columns and 'category' not in df.columns:
        df['category'] = df['product']
    
    required_cols = ['hs_code', 'category', 'goods_shipped']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    
    # Get attribute columns (only columns starting with 'attr_')
    attribute_cols = [c for c in df.columns if c.startswith('attr_')]
    
    records = []
    for _, row in df.iterrows():
        hs_code = str(row['hs_code'])[:4].zfill(4)
        category = str(row['category'])
        goods_shipped = str(row['goods_shipped'])
        
        attributes = {}
        for attr_col in attribute_cols:
            value = row[attr_col]
            if pd.notna(value):
                attributes[attr_col] = str(value)
        
        if attributes:  # Only include records with at least one attribute
            records.append({
                'hs_code': hs_code,
                'category': category,
                'goods_shipped': goods_shipped,
                'attributes': attributes
            })
    
    return records


def prepare_attribute_records(
    classifications: List[Dict[str, Any]],
    text_field: str = "goods_shipped"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Prepare records grouped by namespace (hs_code + category + attribute).
    
    Args:
        classifications: List of classification records
        text_field: Name of text field for embeddings
    
    Returns:
        Dictionary mapping namespace to list of records
    """
    namespace_records = defaultdict(list)
    
    # Track unique goods per attribute to avoid duplicates
    seen_combinations = defaultdict(set)
    
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
            
            # Generate namespace
            safe_category = category.replace(" ", "_").replace("/", "_")
            safe_attribute = attr_name.replace(" ", "_").replace("/", "_")
            namespace = f"hs_{hs_code}_{safe_category}_{safe_attribute}"
            
            # Check if we've seen this combination before
            combination_key = f"{namespace}|{goods_shipped}|{attr_value}"
            if combination_key in seen_combinations[namespace]:
                continue
            seen_combinations[namespace].add(combination_key)
            
            # Create record
            record_id = f"{hs_code}_{category}_{attr_name}_{hash(goods_shipped + attr_value)}"
            
            record = {
                "_id": record_id,
                text_field: goods_shipped,
                "attribute_name": attr_name,
                "attribute_value": attr_value,
                "hs_code": hs_code,
                "category": category,
            }
            
            namespace_records[namespace].append(record)
    
    return dict(namespace_records)


def populate_index(
    input_path: str,
    index_name: str,
    embedding_model: str = "multilingual-e5-large",
    text_field: str = "goods_shipped",
    batch_size: int = 50,
    dry_run: bool = False
):
    """
    Populate Pinecone index with attribute classifications.
    
    Args:
        input_path: Path to JSON or CSV file with classifications
        index_name: Name of Pinecone index
        embedding_model: Embedding model to use
        text_field: Name of text field
        batch_size: Batch size for upserts
        dry_run: If True, don't actually upsert
    """
    logger.info(f"Loading classifications from {input_path}")
    
    # Load classifications
    if input_path.endswith('.json'):
        classifications = load_classifications_from_json(input_path)
    elif input_path.endswith('.csv'):
        classifications = load_classifications_from_csv(input_path)
    else:
        raise ValueError("Input file must be .json or .csv")
    
    logger.info(f"Loaded {len(classifications)} classification records")
    
    # Prepare records by namespace
    logger.info("Preparing records by namespace...")
    namespace_records = prepare_attribute_records(classifications, text_field)
    
    logger.info(f"Prepared {len(namespace_records)} namespaces")
    for namespace, records in list(namespace_records.items())[:5]:
        logger.info(f"  {namespace}: {len(records)} records")
    if len(namespace_records) > 5:
        logger.info(f"  ... and {len(namespace_records) - 5} more namespaces")
    
    # Calculate total unique goods-attribute combinations
    total_records = sum(len(records) for records in namespace_records.values())
    logger.info(f"Total unique goods-attribute combinations: {total_records}")
    
    if dry_run:
        logger.info("DRY RUN - not upserting to Pinecone")
        return
    
    # Initialize Pinecone client (reads API key from environment)
    logger.info(f"Initializing Pinecone client for index '{index_name}'")
    
    # Get API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    
    from pinecone import Pinecone
    pc = Pinecone(api_key=api_key)
    
    # Create index if it doesn't exist with integrated embeddings
    if not pc.has_index(index_name):
        logger.info(f"Creating index '{index_name}' with integrated embeddings")
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": embedding_model,
                "field_map": {"text": text_field}
            }
        )
        logger.info(f"Index '{index_name}' created successfully")
    else:
        logger.info(f"Index '{index_name}' already exists")
    
    index = pc.Index(index_name)
    logger.info(f"Index '{index_name}' is ready")
    
    # Upsert records by namespace - DIRECTLY to Pinecone, bypassing PineconeClient
    # This preserves the attribute schema (attribute_name, attribute_value)
    logger.info("Upserting records to Pinecone...")
    total_upserted = 0
    
    with tqdm(total=len(namespace_records), desc="Namespaces") as pbar:
        for namespace, records in namespace_records.items():
            try:
                # Batch upsert - use Pinecone SDK directly, not PineconeClient
                # This keeps our attribute schema intact
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    index.upsert_records(namespace, batch)
                    total_upserted += len(batch)
                
                pbar.set_postfix({'upserted': total_upserted})
            except Exception as e:
                logger.error(f"Error upserting to namespace '{namespace}': {e}")
            
            pbar.update(1)
    
    logger.info(f"Successfully upserted {total_upserted} records across {len(namespace_records)} namespaces")
    
    # Get index stats
    try:
        stats = index.describe_index_stats()
        logger.info(f"\nIndex statistics:")
        logger.info(f"  Total vectors: {stats.total_vector_count}")
        logger.info(f"  Dimension: {stats.dimension}")
        logger.info(f"  Namespaces: {len(stats.namespaces)}")
        
        # Show top namespaces by record count
        if stats.namespaces:
            sorted_ns = sorted(
                stats.namespaces.items(),
                key=lambda x: x[1].vector_count,
                reverse=True
            )
            logger.info(f"\n  Top namespaces by record count:")
            for ns_name, ns_stats in sorted_ns[:10]:
                logger.info(f"    {ns_name}: {ns_stats.vector_count} vectors")
    except Exception as e:
        logger.warning(f"Could not get index stats: {e}")
    
    logger.info("\nPopulation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Populate Pinecone index with attribute classifications"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to classifications file (JSON or CSV)"
    )
    parser.add_argument(
        "--index-name",
        default="attribute-classification",
        help="Name of Pinecone index (default: attribute-classification)"
    )
    parser.add_argument(
        "--embedding-model",
        default="multilingual-e5-large",
        help="Embedding model (default: multilingual-e5-large for multilingual support)"
    )
    parser.add_argument(
        "--text-field",
        default="goods_shipped",
        help="Name of text field (default: goods_shipped)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for upserts (default: 50)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually upsert, just prepare records"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Get API key from environment
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key and not args.dry_run:
        logger.error("PINECONE_API_KEY environment variable not set")
        sys.exit(1)
    
    try:
        populate_index(
            input_path=args.input,
            index_name=args.index_name,
            embedding_model=args.embedding_model,
            text_field=args.text_field,
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
    except Exception as e:
        logger.error(f"Error populating index: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
