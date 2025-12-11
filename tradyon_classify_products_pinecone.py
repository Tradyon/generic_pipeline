#!/usr/bin/env python3
"""
CLI tool to classify products using Pinecone hybrid search with LLM fallback.

This tool uses vector similarity search via Pinecone for fast classification
of known patterns, falling back to LLM for novel goods descriptions.
"""

import argparse
import os
import sys
import logging
import pandas as pd
from dotenv import load_dotenv

from src.utils.config_models import load_product_definition
from src.classifiers.pinecone_product_classifier import PineconeProductClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def classify_products_pinecone(
    input_csv: str,
    products_definition_path: str,
    output_csv: str,
    pinecone_index_name: str,
    pinecone_namespace: str,
    similarity_threshold: float = 0.85,
    checkpoint_file: str = None,
    model_name: str = "gemini-2.0-flash",
    embedding_model: str = "llama-text-embed-v2",
    text_field: str = "content",
    batch_size: int = 10,
    max_workers: int = 8,
    batch_timeout: int = 900,
    use_reranking: bool = True,
    resume: bool = True,
    low_memory: bool = False,
    create_index: bool = True
) -> pd.DataFrame:
    """
    Classify products using Pinecone hybrid search.
    
    Parameters:
    -----------
    input_csv : str
        Path to input CSV file with shipment data
    products_definition_path : str
        Path to products definition JSON
    output_csv : str
        Path to output CSV file
    pinecone_index_name : str
        Name of Pinecone index
    pinecone_namespace : str
        Namespace within the index
    similarity_threshold : float
        Minimum similarity score for vector match (0.0-1.0)
    checkpoint_file : str, optional
        Path to checkpoint file
    model_name : str
        LLM model name for fallback
    embedding_model : str
        OpenAI embedding model name
    embedding_dimension : int
        Dimension of embeddings
    batch_size : int
        Number of goods per batch
    max_workers : int
        Number of parallel workers
    batch_timeout : int
        Timeout in seconds per batch
    use_reranking : bool
        Whether to use Pinecone reranking
    resume : bool
        Resume from checkpoint if exists
    low_memory : bool
        Use low memory mode for pandas
    create_index : bool
        Create Pinecone index if it doesn't exist
        
    Returns:
    --------
    pd.DataFrame
        Classified DataFrame
    """
    # Load environment variables
    load_dotenv()
    
    logger.info("=== Pinecone Product Classification ===")
    logger.info(f"Input CSV: {input_csv}")
    logger.info(f"Products definition: {products_definition_path}")
    logger.info(f"Output CSV: {output_csv}")
    logger.info(f"Pinecone index: {pinecone_index_name}")
    logger.info(f"Namespace: {pinecone_namespace}")
    logger.info(f"Similarity threshold: {similarity_threshold}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Embedding model: {embedding_model}")
    
    # Load products definition
    logger.info("Loading products definition...")
    products_definition = load_product_definition(products_definition_path)
    logger.info(f"Loaded {len(products_definition.product_categories)} product categories")
    
    # Load shipment data
    logger.info("Loading shipment data...")
    df = pd.read_csv(input_csv, low_memory=low_memory)
    logger.info(f"Loaded {len(df)} shipment records")
    logger.info(f"Unique goods descriptions: {df['goods_shipped'].nunique()}")
    
    # Initialize classifier
    logger.info("Initializing Pinecone classifier...")
    classifier = PineconeProductClassifier(
        products_definition=products_definition,
        pinecone_index_name=pinecone_index_name,
        pinecone_namespace=pinecone_namespace,
        model_name=model_name,
        similarity_threshold=similarity_threshold,
        embedding_model=embedding_model,
        text_field=text_field,
        checkpoint_file=checkpoint_file,
        batch_size=batch_size,
        max_workers=max_workers,
        batch_timeout=batch_timeout,
        use_reranking=use_reranking,
        create_index=create_index
    )
    
    # Classify
    logger.info("Starting classification...")
    df_classified = classifier.classify_dataframe(
        df=df,
        resume=resume,
        low_memory=low_memory
    )
    
    # Save output
    logger.info(f"Saving results to {output_csv}...")
    df_classified.to_csv(output_csv, index=False)
    logger.info(f"Successfully saved {len(df_classified)} classified records")
    
    # Final statistics
    logger.info("\n=== Classification Summary ===")
    logger.info(f"Total records: {len(df_classified)}")
    logger.info(f"Vector matches: {classifier.vector_match_count}")
    logger.info(f"LLM fallbacks: {classifier.llm_fallback_count}")
    logger.info(f"Total tokens used: {classifier.total_tokens_used}")
    
    cost_savings_pct = (classifier.vector_match_count / 
                       (classifier.vector_match_count + classifier.llm_fallback_count) * 100 
                       if (classifier.vector_match_count + classifier.llm_fallback_count) > 0 else 0)
    logger.info(f"Cost savings: {cost_savings_pct:.1f}% (vector matches avoided LLM calls)")
    
    return df_classified


def main():
    parser = argparse.ArgumentParser(
        description="Classify products using Pinecone hybrid search with LLM fallback."
    )
    
    # Required arguments
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file (e.g., shipment_master.csv)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV file (e.g., shipment_master_classified.csv)"
    )
    parser.add_argument(
        "--products-definition",
        required=True,
        help="Path to products definition JSON"
    )
    
    # Pinecone settings
    parser.add_argument(
        "--pinecone-index",
        default="product-classification-1006",
        help="Pinecone index name (default: product-classification-1006)"
    )
    parser.add_argument(
        "--pinecone-namespace",
        default="hs_1006",
        help="Pinecone namespace (default: hs_1006)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Minimum similarity score for vector match (default: 0.85)"
    )
    parser.add_argument(
        "--no-reranking",
        action="store_true",
        help="Disable Pinecone reranking"
    )
    parser.add_argument(
        "--no-create-index",
        action="store_true",
        help="Don't create index if missing (fail instead)"
    )
    
    # Embedding settings
    parser.add_argument(
        "--embedding-model",
        default="llama-text-embed-v2",
        help="Pinecone integrated embedding model (default: llama-text-embed-v2)"
    )
    parser.add_argument(
        "--text-field",
        default="content",
        help="Field name for text content (default: content)"
    )
    
    # LLM settings
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="LLM model name for fallback (default: gemini-2.0-flash)"
    )
    
    # Processing settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size (default: 10)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Max workers (default: 8)"
    )
    parser.add_argument(
        "--batch-timeout",
        type=int,
        default=900,
        help="Batch timeout in seconds (default: 900)"
    )
    parser.add_argument(
        "--checkpoint-file",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from checkpoint"
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable pandas low_memory mode"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
        
    if not os.path.exists(args.products_definition):
        print(f"Error: Products definition file not found: {args.products_definition}")
        sys.exit(1)
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Run classification
    try:
        classify_products_pinecone(
            input_csv=args.input,
            products_definition_path=args.products_definition,
            output_csv=args.output,
            pinecone_index_name=args.pinecone_index,
            pinecone_namespace=args.pinecone_namespace,
            similarity_threshold=args.similarity_threshold,
            checkpoint_file=args.checkpoint_file,
            model_name=args.model,
            embedding_model=args.embedding_model,
            text_field=args.text_field,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            batch_timeout=args.batch_timeout,
            use_reranking=not args.no_reranking,
            resume=not args.no_resume,
            low_memory=args.low_memory,
            create_index=not args.no_create_index
        )
        logger.info("Classification completed successfully!")
    except Exception as e:
        logger.error(f"Classification failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
