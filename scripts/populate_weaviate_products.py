"""
Populate Weaviate with existing product classifications.

This script loads previously classified shipment data and populates
the Weaviate vector database to create an initial knowledge base.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.weaviate_client import WeaviateClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_classified_data(csv_path: str) -> pd.DataFrame:
    """
    Load previously classified shipment data.
    
    Args:
        csv_path: Path to CSV with 'goods_shipped' and 'category' columns
        
    Returns:
        DataFrame with classification data
    """
    logger.info(f"Loading classified data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['goods_shipped', 'category', 'hs_code']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Filter out unclassified or empty records
    df = df[df['category'].notna()].copy()
    df = df[df['category'] != 'Unclassified'].copy()
    df = df[df['goods_shipped'].notna()].copy()
    df = df[df['goods_shipped'].str.strip() != ''].copy()
    
    logger.info(f"Loaded {len(df)} valid classified records")
    
    return df


def prepare_records(df: pd.DataFrame) -> List[Dict]:
    """
    Prepare records for Weaviate upsert.
    
    Args:
        df: DataFrame with classification data
        
    Returns:
        List of record dictionaries
    """
    logger.info("Preparing records for Weaviate...")
    
    # Group by unique goods_shipped to count shipments per description
    grouped = df.groupby(['goods_shipped', 'category', 'hs_code']).size().reset_index(name='shipment_count')
    
    records = []
    for _, row in grouped.iterrows():
        record = {
            'goods_shipped': row['goods_shipped'].strip(),
            'category': row['category'],
            'hs_code': str(row['hs_code']),
            'confidence': 1.0,  # Historical data assumed correct
            'classification_method': 'historical_llm',
            'shipment_count': int(row['shipment_count'])
        }
        records.append(record)
    
    logger.info(f"Prepared {len(records)} unique records")
    logger.info(f"Category distribution:")
    category_counts = grouped['category'].value_counts()
    for category, count in category_counts.items():
        logger.info(f"  {category}: {count}")
    
    return records


def populate_index(
    csv_path: str,
    class_name: str,
    embedding_model: str,
    batch_size: int = 100
):
    """
    Populate Weaviate index.
    
    Args:
        csv_path: Path to input CSV
        class_name: Weaviate class name
        embedding_model: Embedding model name
        batch_size: Batch size for upsert
    """
    # Load data
    df = load_classified_data(csv_path)
    
    # Prepare records
    records = prepare_records(df)
    
    if not records:
        logger.warning("No records to upsert")
        return
    
    # Initialize Weaviate client
    client = WeaviateClient(
        class_name=class_name,
        embedding_model_name=embedding_model,
        text_field="goods_shipped"
    )
    
    # Upsert records
    logger.info(f"Upserting {len(records)} records to class '{class_name}'")
    
    # Process in chunks
    total_records = len(records)
    for i in range(0, total_records, batch_size):
        batch = records[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(total_records + batch_size - 1)//batch_size}")
        client.upsert_records(batch, batch_size=batch_size)
        
    logger.info("Population complete!")
    client.close()


def main():
    parser = argparse.ArgumentParser(description="Populate Weaviate with product classifications")
    parser.add_argument("--input", required=True, help="Path to classified shipment CSV")
    parser.add_argument("--class-name", default="ProductClassification", help="Weaviate class name")
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-large", help="Embedding model name")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for upsert")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
        
    try:
        populate_index(
            csv_path=args.input,
            class_name=args.class_name,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    load_dotenv()
    main()
