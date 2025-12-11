"""
Populate Pinecone index with existing product classifications.

This script loads previously classified shipment data and populates
the Pinecone vector database to create an initial knowledge base.
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

from src.utils.pinecone_client import PineconeClient

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
    Prepare records for Pinecone upsert.
    
    Args:
        df: DataFrame with classification data
        
    Returns:
        List of record dictionaries
    """
    logger.info("Preparing records for Pinecone...")
    
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
    index_name: str,
    namespace: str,
    embedding_model: str = "llama-text-embed-v2",
    text_field: str = "content",
    batch_size: int = 96,
    create_if_missing: bool = True
):
    """
    Populate Pinecone index with classified data.
    
    Args:
        csv_path: Path to classified CSV file
        index_name: Name of Pinecone index
        namespace: Namespace to use
        embedding_model: OpenAI embedding model
        dimension: Embedding dimension
        batch_size: Batch size for operations
        create_if_missing: Create index if it doesn't exist
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize Pinecone client
    logger.info(f"Initializing Pinecone client for index '{index_name}'")
    client = PineconeClient(
        index_name=index_name,
        embedding_model=embedding_model,
        text_field=text_field
    )
    
    # Create index if needed
    if create_if_missing:
        client.create_index()
    
    # Load and prepare data
    df = load_classified_data(csv_path)
    records = prepare_records(df)
    
    # Upsert to Pinecone
    logger.info(f"Starting upsert to namespace '{namespace}'")
    client.upsert_records(
        namespace=namespace,
        records=records,
        batch_size=batch_size
    )
    
    # Get and display stats
    stats = client.get_index_stats(namespace=namespace)
    logger.info(f"Index statistics:")
    logger.info(f"  Total vectors: {stats.get('total_vector_count', 'N/A')}")
    logger.info(f"  Namespace vectors: {stats.get('namespace_vector_count', 'N/A')}")
    logger.info(f"  Dimension: {stats.get('dimension', 'N/A')}")
    
    logger.info("Population complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Populate Pinecone index with existing product classifications"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to classified CSV file (with goods_shipped, category, hs_code columns)"
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="product-classification-1006",
        help="Name of Pinecone index (default: product-classification-1006)"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="hs_1006",
        help="Namespace to use (default: hs_1006)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="llama-text-embed-v2",
        help="Pinecone integrated embedding model (default: llama-text-embed-v2)"
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="content",
        help="Field name for text content (default: content)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for operations (default: 100)"
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="Don't create index if missing (fail instead)"
    )
    
    args = parser.parse_args()
    
    populate_index(
        csv_path=args.csv,
        index_name=args.index_name,
        namespace=args.namespace,
        embedding_model=args.embedding_model,
        text_field=args.text_field,
        batch_size=args.batch_size,
        create_if_missing=not args.no_create
    )


if __name__ == "__main__":
    main()
