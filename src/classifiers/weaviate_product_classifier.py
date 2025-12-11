"""
Weaviate-based Products Classifier Module

Hybrid classification using Weaviate vector search with LLM fallback.
Uses semantic similarity for fast classification of known patterns,
falls back to LLM for novel goods descriptions.
"""

import pandas as pd
import json
import time
import os
import threading
import pickle
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime

from src.utils.config_models import ProductDefinition
from src.utils.llm_client import LLMClient
from src.utils.weaviate_client import WeaviateClient


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


class WeaviateProductClassifier:
    """Hybrid product classifier using Weaviate vector search with LLM fallback."""

    def __init__(
        self,
        products_definition: Union[ProductDefinition, Dict[str, Any]],
        class_name: str = "ProductClassification",
        model_name: str = "gemini-2.0-flash",
        similarity_threshold: float = 0.85,
        embedding_model: str = "intfloat/multilingual-e5-large",
        text_field: str = "goods_shipped",
        checkpoint_file: Optional[str] = None,
        batch_size: int = 10,
        max_workers: int = 8,
        batch_timeout: int = 900,
        use_reranking: bool = True,
        api_key: Optional[str] = None,
        weaviate_url: Optional[str] = None,
        weaviate_api_key: Optional[str] = None
    ):
        """
        Initialize the Weaviate product classifier.
        
        Parameters:
        -----------
        products_definition : Union[ProductDefinition, Dict[str, Any]]
            Product definition with categories and descriptions
        class_name : str
            Name of Weaviate class
        model_name : str
            LLM model name for fallback classification
        similarity_threshold : float
            Minimum similarity score for vector match (0.0-1.0)
        embedding_model : str
            Embedding model name
        text_field : str
            Field name for text content
        checkpoint_file : str, optional
            Path to checkpoint file for resume capability
        batch_size : int
            Number of goods descriptions per batch
        max_workers : int
            Number of parallel workers
        batch_timeout : int
            Timeout in seconds for batch processing
        use_reranking : bool
            Whether to use reranking
        api_key : str, optional
            Google API key (uses environment variable if not provided)
        weaviate_url : str, optional
            Weaviate URL
        weaviate_api_key : str, optional
            Weaviate API Key
        """
        # Parse product definition
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
        self.hs_code = definition.hs_code
        
        # Weaviate settings
        self.class_name = class_name
        self.similarity_threshold = similarity_threshold
        self.use_reranking = use_reranking
        
        # Initialize Weaviate client
        logger.info(f"Initializing Weaviate client for class '{class_name}'")
        self.weaviate_client = WeaviateClient(
            url=weaviate_url,
            api_key=weaviate_api_key,
            class_name=class_name,
            embedding_model_name=embedding_model,
            text_field=text_field
        )
        
        # Create dynamic schema for LLM fallback
        self.classification_schema = create_classification_schema(self.product_categories)
        
        # Initialize LLM client for fallback
        self.llm_client = LLMClient(
            model_name=model_name,
            api_key=api_key
        )
        
        # Processing settings
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.checkpoint_file = checkpoint_file or "weaviate_classification_checkpoint.pkl"
        self.batch_timeout = batch_timeout
        
        # Tracking
        self.total_tokens_used = 0
        self.vector_match_count = 0
        self.llm_fallback_count = 0
        self.total_similarity_score = 0.0
        self.start_time = None
        self.lock = threading.Lock()
        
        # Warmup models
        self.weaviate_client.warmup()

    def close(self):
        """Close the Weaviate client connection."""
        if self.weaviate_client:
            self.weaviate_client.close()
    
    def classify_single_weaviate(self, goods_description: str) -> Tuple[str, float, str]:
        """
        Classify a single goods description using Weaviate.
        
        Parameters:
        -----------
        goods_description : str
            Goods description to classify
            
        Returns:
        --------
        Tuple[str, float, str]
            (category, confidence_score, method)
            method is either 'vector_match' or 'no_match'
        """
        try:
            # Query Weaviate with reranking
            matches = self.weaviate_client.search_with_rerank(
                query_text=goods_description,
                hs_code=self.hs_code,  # Filter by HS code
                top_k=10,  # Get more candidates for reranking
                rerank_top_n=5
            )
            
            # Check if we have a good match
            if matches and matches[0]['score'] >= self.similarity_threshold:
                # Access fields
                category = matches[0]['fields'].get('category', 'Unclassified')
                confidence = matches[0]['score']
                
                with self.lock:
                    self.vector_match_count += 1
                    self.total_similarity_score += confidence
                
                return category, confidence, 'vector_match'
            
            return "Unclassified", 0.0, 'no_match'
            
        except Exception as e:
            logger.warning(f"Weaviate query error: {e}")
            return "Unclassified", 0.0, 'error'
    
    def build_llm_prompt(self, goods_descriptions: List[str]) -> str:
        """Build classification prompt for LLM fallback."""
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
        
        prompt = f"""You are a careful classifier. Assign each goods description to exactly ONE category from the allowed list.

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

If uncertain, use "Unclassified". Array length MUST match {len(goods_descriptions)} items."""
        
        return prompt
    
    def classify_batch_llm(self, goods_batch: List[str]) -> List[str]:
        """
        Classify a batch of goods using LLM.
        
        Parameters:
        -----------
        goods_batch : List[str]
            List of goods descriptions
            
        Returns:
        --------
        List[str]
            List of category assignments
        """
        try:
            prompt = self.build_llm_prompt(goods_batch)
            
            # Call LLM with schema
            response = self.llm_client.generate(
                prompt=prompt,
                schema=self.classification_schema,
                temperature=0.0
            )
            
            # Track tokens
            with self.lock:
                self.total_tokens_used += response.total_tokens
                self.llm_fallback_count += len(goods_batch)
            
            # Parse response
            try:
                result = json.loads(response.content)
                classifications = result.get('classifications', [])
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON, extracting from text")
                # Fallback extraction
                import re
                match = re.search(r'\{[\s\S]*\}', response.content)
                if match:
                    result = json.loads(match.group(0))
                    classifications = result.get('classifications', [])
                else:
                    classifications = []
            
            # Validate length
            if len(classifications) != len(goods_batch):
                logger.warning(f"Classification count mismatch: expected {len(goods_batch)}, got {len(classifications)}")
                classifications = classifications + ["Unclassified"] * (len(goods_batch) - len(classifications))
                classifications = classifications[:len(goods_batch)]
            
            return classifications
            
        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            return ["Unclassified"] * len(goods_batch)
    
    def classify_batch_hybrid(self, goods_batch: List[str]) -> List[Tuple[str, float, str]]:
        """
        Classify a batch using hybrid approach: Weaviate first, LLM for misses.
        
        Parameters:
        -----------
        goods_batch : List[str]
            List of goods descriptions
            
        Returns:
        --------
        List[Tuple[str, float, str]]
            List of (category, confidence, method) tuples
        """
        results = []
        llm_needed = []
        llm_indices = []
        
        # Try Weaviate for all items
        for i, goods in enumerate(goods_batch):
            category, confidence, method = self.classify_single_weaviate(goods)
            
            if method == 'vector_match':
                results.append((category, confidence, method))
            else:
                # Mark for LLM fallback
                results.append(None)
                llm_needed.append(goods)
                llm_indices.append(i)
        
        # Use LLM for items that didn't match
        if llm_needed:
            logger.debug(f"Using LLM fallback for {len(llm_needed)} items")
            llm_categories = self.classify_batch_llm(llm_needed)
            
            # Fill in LLM results
            for idx, category in zip(llm_indices, llm_categories):
                results[idx] = (category, 0.0, 'llm_fallback')
            
            # Store new classifications in Weaviate for future use
            self.store_new_classifications(llm_needed, llm_categories)
        
        return results
    
    def store_new_classifications(self, goods_list: List[str], categories: List[str]) -> None:
        """
        Store newly classified goods in Weaviate for future retrieval.
        
        Parameters:
        -----------
        goods_list : List[str]
            List of goods descriptions
        categories : List[str]
            Corresponding category assignments
        """
        try:
            records = []
            for goods, category in zip(goods_list, categories):
                if category != "Unclassified":
                    records.append({
                        'goods_shipped': goods,
                        'category': category,
                        'hs_code': self.hs_code,
                        'confidence': 0.8,  # Lower confidence for LLM-derived
                        'classification_method': 'llm_fallback',
                        'shipment_count': 1
                    })
            
            if records:
                self.weaviate_client.upsert_records(
                    records=records,
                    batch_size=100
                )
                logger.debug(f"Stored {len(records)} new classifications in Weaviate")
                
        except Exception as e:
            logger.warning(f"Failed to store new classifications: {e}")
    
    def save_checkpoint(self, df: pd.DataFrame, processed_goods: set, processed_count: int) -> None:
        """Save checkpoint for resume capability."""
        checkpoint = {
            'dataframe': df.copy(),
            'processed_count': processed_count,
            'processed_goods': processed_goods,
            'total_tokens_used': self.total_tokens_used,
            'vector_match_count': self.vector_match_count,
            'llm_fallback_count': self.llm_fallback_count,
            'total_similarity_score': self.total_similarity_score,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Checkpoint saved: {processed_count} goods processed")
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint if exists."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                logger.info(f"Loaded checkpoint from {checkpoint['timestamp']}")
                logger.info(f"Resuming from {checkpoint['processed_count']} processed goods")
                
                # Restore state
                self.total_tokens_used = checkpoint.get('total_tokens_used', 0)
                self.vector_match_count = checkpoint.get('vector_match_count', 0)
                self.llm_fallback_count = checkpoint.get('llm_fallback_count', 0)
                self.total_similarity_score = checkpoint.get('total_similarity_score', 0.0)
                
                return checkpoint
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return None
    
    def classify_dataframe(
        self,
        df: pd.DataFrame,
        resume: bool = True,
        low_memory: bool = False
    ) -> pd.DataFrame:
        """
        Classify goods descriptions in a DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'goods_shipped' column
        resume : bool
            Whether to resume from checkpoint
        low_memory : bool
            Use memory-efficient processing
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added 'category' column
        """
        self.start_time = time.time()
        
        # Try to load checkpoint
        checkpoint = None
        if resume:
            checkpoint = self.load_checkpoint()
        
        if checkpoint:
            df = checkpoint['dataframe']
            processed_goods = checkpoint['processed_goods']
            processed_count = checkpoint['processed_count']
        else:
            df['category'] = None
            processed_goods = set()
            processed_count = 0
        
        # Get unique goods descriptions
        unique_goods = df['goods_shipped'].unique()
        unique_goods = [g for g in unique_goods if g not in processed_goods]
        
        total_unique = len(unique_goods)
        logger.info(f"Total unique goods to classify: {total_unique}")
        logger.info(f"Already processed: {processed_count}")
        
        # Create mapping from goods to dataframe indices
        goods_to_indices = {}
        for idx, row in df.iterrows():
            goods = row['goods_shipped']
            if goods not in goods_to_indices:
                goods_to_indices[goods] = []
            goods_to_indices[goods].append(idx)
        
        # Create batches
        batches = []
        for i in range(0, len(unique_goods), self.batch_size):
            batch = unique_goods[i:i + self.batch_size]
            batches.append((i // self.batch_size, batch))
        
        logger.info(f"Processing {len(batches)} batches with {self.max_workers} workers")
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {}
            start_times = {}
            
            for batch_index, batch_goods in batches:
                fut = executor.submit(self.classify_batch_hybrid, batch_goods)
                future_to_batch[fut] = (batch_index, batch_goods)
                start_times[fut] = time.time()
            
            pending = set(future_to_batch.keys())
            pbar = tqdm(total=len(batches), desc="Classifying batches")
            
            while pending:
                done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                
                # Check for timeouts
                now = time.time()
                to_cancel = [f for f in pending if now - start_times.get(f, now) > self.batch_timeout]
                
                for fut in to_cancel:
                    batch_index, batch_goods = future_to_batch[fut]
                    logger.warning(f"Batch {batch_index} timeout after {self.batch_timeout}s, marking as Unclassified")
                    fut.cancel()
                    
                    # Mark as unclassified
                    for goods in batch_goods:
                        for idx in goods_to_indices.get(goods, []):
                            df.at[idx, 'category'] = 'Unclassified'
                        processed_goods.add(goods)
                    
                    processed_count += len(batch_goods)
                    pbar.update(1)
                
                # Process completed futures
                for fut in done:
                    batch_index, batch_goods = future_to_batch[fut]
                    
                    try:
                        results = fut.result()
                        
                        # Apply results to dataframe
                        for goods, (category, confidence, method) in zip(batch_goods, results):
                            for idx in goods_to_indices.get(goods, []):
                                df.at[idx, 'category'] = category
                            processed_goods.add(goods)
                        
                        processed_count += len(batch_goods)
                        
                        # Save checkpoint every 50 goods
                        if processed_count % 50 == 0:
                            self.save_checkpoint(df, processed_goods, processed_count)
                        
                    except Exception as e:
                        logger.error(f"Batch {batch_index} failed: {e}")
                        for goods in batch_goods:
                            for idx in goods_to_indices.get(goods, []):
                                df.at[idx, 'category'] = 'Unclassified'
                            processed_goods.add(goods)
                        processed_count += len(batch_goods)
                    
                    pbar.update(1)
            
            pbar.close()
        
        # Clean up checkpoint on success
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            logger.info("Checkpoint file removed after successful completion")
        
        # Log statistics
        elapsed = time.time() - self.start_time
        logger.info(f"\nClassification complete in {elapsed:.2f}s")
        logger.info(f"Total goods classified: {processed_count}")
        logger.info(f"Vector matches: {self.vector_match_count} ({self.vector_match_count/processed_count*100:.1f}%)")
        logger.info(f"LLM fallbacks: {self.llm_fallback_count} ({self.llm_fallback_count/processed_count*100:.1f}%)")
        if self.vector_match_count > 0:
            avg_similarity = self.total_similarity_score / self.vector_match_count
            logger.info(f"Average similarity score: {avg_similarity:.3f}")
        logger.info(f"Total LLM tokens used: {self.total_tokens_used}")
        
        # Calculate category distribution
        category_counts = df['category'].value_counts()
        logger.info("\nCategory distribution:")
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} ({count/len(df)*100:.1f}%)")
        
        return df
