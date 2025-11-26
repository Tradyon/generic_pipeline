"""
Products Classifier Module

Classify shipment goods descriptions into product categories using LLM.
"""

import pandas as pd
import json
import time
import os
import threading
import pickle
import logging
import time
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime

from src.utils.config_models import ProductDefinition, load_product_definition
from src.utils.llm_client import LLMClient


logger = logging.getLogger(__name__)


def create_classification_schema(product_categories: List[str]) -> Dict[str, Any]:
    """
    Create a JSON schema for batch classification.
    
    Parameters:
    -----------
    product_categories : List[str]
        List of allowed product category names
    
    Returns:
    --------
    Dict[str, Any]
        JSON schema for batch classification
    """
    enum_values = list(product_categories)
    if "Unclassified" not in enum_values:
        enum_values.append("Unclassified")
    return {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": enum_values
                }
            }
        },
        "required": ["classifications"]
    }


class ProductClassifier:
    """Product classifier with LLM-based batch classification."""

    def __init__(
        self,
        products_definition: Union[ProductDefinition, Dict[str, Any]],
        model_name: str = "gemini-2.0-flash",
        checkpoint_file: Optional[str] = None,
        batch_size: int = 10,
        max_workers: int = 8,
        batch_timeout: int = 300,
        api_key: Optional[str] = None
    ):
        """
        Initialize the product classifier.
        
        Parameters:
        -----------
        products_definition : Dict[str, Any]
            Dictionary with 'product_categories' and 'category_descriptions'
        model_name : str
            Gemini model name
        checkpoint_file : str, optional
            Path to checkpoint file for resume capability
        batch_size : int
            Number of goods descriptions per API call
        max_workers : int
            Number of parallel workers
        api_key : str, optional
            Google API key (uses environment variable if not provided)
        """
        if isinstance(products_definition, ProductDefinition):
            definition = products_definition
        else:
            definition = ProductDefinition(
                hs_code=str(products_definition.get('hs_code', "")).strip(),
                product_categories=[
                    str(c).strip() for c in products_definition.get('product_categories', []) if str(c).strip()
                ],
                category_descriptions={
                    str(k).strip(): str(v).strip()
                    for k, v in (products_definition.get('category_descriptions', {}) or {}).items()
                },
                metadata={
                    k: v
                    for k, v in products_definition.items()
                    if k not in {'hs_code', 'product_categories', 'category_descriptions'}
                }
            )

        self.product_definition = definition
        self.product_categories = definition.product_categories
        self.category_descriptions = definition.category_descriptions
        
        # Create dynamic schema
        self.classification_schema = create_classification_schema(self.product_categories)
        
        self.model = LLMClient(
            model_name=model_name,
            api_key=api_key
        )
        
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.checkpoint_file = checkpoint_file or "classification_checkpoint.pkl"
        self.total_tokens_used = 0
        self.start_time = None
        self.lock = threading.Lock()
        self.batch_timeout = batch_timeout
    
    def build_prompt(self, goods_descriptions: List[str]) -> str:
        """Build classification prompt from product definition."""
        descriptions_text = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(goods_descriptions)])
        
        # Build category descriptions text
        category_text_parts = []
        for cat in self.product_categories:
            desc = self.category_descriptions.get(cat, "")
            if desc:
                category_text_parts.append(f"- {cat}: {desc}")
            else:
                category_text_parts.append(f"- {cat}")
        
        category_text = "\n".join(category_text_parts)
        
        prompt = f"""
        You are a careful classifier. Assign each goods description to exactly ONE category from the allowed list.

        Allowed categories:
        {category_text}

        GOODS (preserve order):
        {descriptions_text}

        Output JSON ONLY:
        {{
          "classifications": [
            "Category for item 1",
            "Category for item 2",
            ...
          ]
        }}
        - Length of classifications MUST equal the number of goods above.
        - Use only category names from the allowed list.
        - If unsure, output "Unclassified".
        - No explanations, no extra keys, no markdown.
        """
        
        return prompt
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // 4
    
    def classify_batch_products(self, goods_descriptions: List[str]) -> List[str]:
        """Classify a batch of product descriptions in a single API call."""
        if not goods_descriptions:
            return []
        
        prompt = self.build_prompt(goods_descriptions)
        
        try:
            input_tokens = self.count_tokens(prompt)
            time.sleep(0.1)  # Rate limiting
            
            response = self.model.generate(prompt, schema=self.classification_schema)
            text = getattr(response, "text", "")
            try:
                result = json.loads(text)
            except Exception:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        result = json.loads(text[start : end + 1])
                    except Exception:
                        result = {}
                else:
                    result = {}

            categories = result.get("classifications") if isinstance(result, dict) else None
            if not isinstance(categories, list):
                categories = ["Unclassified"] * len(goods_descriptions)
            elif len(categories) != len(goods_descriptions):
                categories = (categories + ["Unclassified"] * len(goods_descriptions))[: len(goods_descriptions)]
            
            output_tokens = len(categories) * 3
            
            with self.lock:
                self.total_tokens_used += input_tokens + output_tokens
            
            return categories
        except Exception as e:
            logger.error(f"Error classifying batch: {str(e)}")
            return ["Unclassified"] * len(goods_descriptions)
    
    def process_batch_with_results(self, batch_goods: List[str], batch_index: int) -> tuple:
        """Process a single batch and return results with batch index."""
        try:
            batch_results = self.classify_batch_products(batch_goods)
            return batch_index, batch_goods, batch_results, None
        except Exception as e:
            logger.error(f"Error processing batch {batch_index}: {str(e)}")
            return batch_index, batch_goods, ["Unclassified"] * len(batch_goods), str(e)
    
    def save_checkpoint(self, df: pd.DataFrame, processed_count: int, processed_goods: set):
        """Save current progress to checkpoint file."""
        checkpoint_data = {
            'dataframe': df,
            'processed_count': processed_count,
            'processed_goods': processed_goods,
            'total_tokens_used': self.total_tokens_used,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Checkpoint saved: {processed_count} unique goods processed, {self.total_tokens_used} tokens used")
    
    def load_checkpoint(self) -> tuple:
        """Load progress from checkpoint file."""
        if not os.path.exists(self.checkpoint_file):
            return None, 0, set()
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            processed_goods = checkpoint_data.get('processed_goods', set())
            processed_count = checkpoint_data.get('processed_count', 0)
            
            logger.info(f"Checkpoint loaded: {processed_count} unique goods processed")
            logger.info(f"Previous session used {checkpoint_data['total_tokens_used']} tokens")
            
            return checkpoint_data['dataframe'], processed_count, processed_goods
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return None, 0, set()
    
    def classify_dataframe(self, df: pd.DataFrame, resume: bool = True) -> pd.DataFrame:
        """
        Classify goods in DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'goods_shipped' column
        resume : bool
            Whether to resume from checkpoint
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with added 'category' column
        """
        logger.info(f"Starting classification of {len(df)} records")
        
        # Load checkpoint if resuming
        df_checkpoint = None
        start_index = 0
        processed_goods = set()
        
        if resume:
            result = self.load_checkpoint()
            if result[0] is not None:
                df_checkpoint, start_index, processed_goods = result
        
        if df_checkpoint is not None:
            df = df_checkpoint
        else:
            if 'goods_shipped' not in df.columns:
                raise ValueError("DataFrame must contain 'goods_shipped' column")
            df['category'] = ''
        
        # Create unique goods mapping
        unique_goods = df['goods_shipped'].unique()
        logger.info(f"Found {len(unique_goods)} unique goods descriptions")
        
        # Create mapping from goods_shipped to indices
        goods_to_indices = {}
        for idx, goods in enumerate(df['goods_shipped']):
            if goods not in goods_to_indices:
                goods_to_indices[goods] = []
            goods_to_indices[goods].append(idx)
        
        # Filter out already processed goods
        remaining_goods = [goods for goods in unique_goods if goods not in processed_goods]
        logger.info(f"Remaining unique goods to process: {len(remaining_goods)}")
        
        if not remaining_goods:
            logger.info("All goods have been processed!")
            return df
        
        self.start_time = time.time()
        
        # Process unique goods in batches with multi-threading
        total_batches = (len(remaining_goods) + self.batch_size - 1) // self.batch_size
        
        # Create batches
        batches = []
        for i in range(0, len(remaining_goods), self.batch_size):
            batch_end = min(i + self.batch_size, len(remaining_goods))
            batch_goods = remaining_goods[i:batch_end]
            batches.append((i // self.batch_size, batch_goods))
        
        logger.info(f"Processing {len(batches)} batches with {self.max_workers} workers")
        
        with tqdm(total=len(remaining_goods), desc="Classifying unique goods") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_batch = {}
                start_times = {}
                for batch_index, batch_goods in batches:
                    fut = executor.submit(self.process_batch_with_results, batch_goods, batch_index)
                    future_to_batch[fut] = (batch_index, batch_goods)
                    start_times[fut] = time.time()

                pending = set(future_to_batch.keys())
                while pending:
                    done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)

                    # Handle timeouts individually
                    now = time.time()
                    to_cancel = [f for f in pending if now - start_times.get(f, now) > self.batch_timeout]
                    for fut in to_cancel:
                        batch_index, batch_goods = future_to_batch.get(fut, (-1, []))
                        logger.warning(f"Batch {batch_index} exceeded timeout; marking {len(batch_goods)} goods as Unclassified.")
                        fut.cancel()
                        for goods in batch_goods:
                            for idx in goods_to_indices.get(goods, []):
                                df.at[idx, 'category'] = "Unclassified"
                            processed_goods.add(goods)
                        pbar.update(len(batch_goods))
                        pending.discard(fut)

                    for future in done:
                        batch_index, batch_goods = future_to_batch.get(future, (-1, []))
                        try:
                            batch_index, batch_goods, batch_results, error = future.result()
                        except Exception as exc:
                            logger.warning(f"Batch {batch_index} raised exception; marking goods as Unclassified. error={exc}")
                            for goods in batch_goods:
                                for idx in goods_to_indices.get(goods, []):
                                    df.at[idx, 'category'] = "Unclassified"
                                processed_goods.add(goods)
                            pbar.update(len(batch_goods))
                            continue

                        logger.info(f"Completed batch {batch_index + 1}/{total_batches} ({len(batch_goods)} goods)")

                        if error:
                            logger.warning(f"Batch {batch_index} had errors but was processed: {error}")

                        for goods, category in zip(batch_goods, batch_results):
                            indices = goods_to_indices[goods]
                            for idx in indices:
                                df.at[idx, 'category'] = category
                            processed_goods.add(goods)

                        pbar.update(len(batch_goods))

                        if len(processed_goods) % 50 == 0:
                            self.save_checkpoint(df, len(processed_goods), processed_goods)

                    # If no futures are done and none pending, break to avoid infinite loop
                    if not pending and not done:
                        break
        
        # Final checkpoint cleanup
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            logger.info("Checkpoint file cleaned up")
        
        self.print_summary(df)
        return df
    
    def print_summary(self, df: pd.DataFrame):
        """Print classification summary statistics."""
        logger.info("\n=== Classification Summary ===")
        
        category_counts = df['category'].value_counts()
        logger.info("\nCategory Distribution:")
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"  {category}: {count} ({percentage:.1f}%)")
        
        unique_goods = df['goods_shipped'].unique()
        total_records = len(df)
        logger.info(f"\nDeduplication Statistics:")
        logger.info(f"  Total records: {total_records:,}")
        logger.info(f"  Unique goods descriptions: {len(unique_goods):,}")
        logger.info(f"  Deduplication ratio: {len(unique_goods)/total_records:.2%}")
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        logger.info(f"\nToken Usage:")
        logger.info(f"  Total tokens used: {self.total_tokens_used:,}")
        logger.info(f"  Processing time: {elapsed_time:.1f} seconds")


def classify_products(
    input_csv: str,
    products_definition_path: str,
    output_csv: str,
    checkpoint_file: Optional[str] = None,
    model_name: str = "gemini-2.0-flash",
    batch_size: int = 10,
    max_workers: int = 8,
    batch_timeout: int = 300,
    resume: bool = True,
    low_memory: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Classify products in a CSV file using LLM.
    
    Parameters:
    -----------
    input_csv : str
        Path to input CSV file
    products_definition_path : str
        Path to products definition JSON
    output_csv : str
        Path to output CSV file
    checkpoint_file : str, optional
        Path to checkpoint file
    model_name : str, optional
        LLM model name
    batch_size : int, optional
        Batch size for LLM calls
    max_workers : int, optional
        Max workers for parallel processing
    resume : bool, optional
        Whether to resume from checkpoint
    low_memory : bool, optional
        Pandas read_csv low_memory flag
    **kwargs
        Additional arguments (e.g., api_key)
        
    Returns:
    --------
    pd.DataFrame
        Classified DataFrame with 'category' column
    """
    # Load products definition via shared helpers
    products_definition = load_product_definition(products_definition_path)
    if not products_definition.product_categories:
        raise ValueError("products_definition must contain at least one product category")
    
    # Load input data
    logger.info(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv, low_memory=low_memory)
    
    # Initialize classifier and run
    classifier = ProductClassifier(
        products_definition=products_definition,
        model_name=model_name,
        checkpoint_file=checkpoint_file,
        batch_size=batch_size,
        max_workers=max_workers,
        batch_timeout=batch_timeout,
        api_key=kwargs.get("api_key")
    )
    classified_df = classifier.classify_dataframe(df, resume=resume)
    
    # Persist results
    os.makedirs(os.path.dirname(output_csv), exist_ok=True) if os.path.dirname(output_csv) else None
    classified_df.to_csv(output_csv, index=False)
    logger.info(f"Saved classified data to {output_csv}")
    
    return classified_df
