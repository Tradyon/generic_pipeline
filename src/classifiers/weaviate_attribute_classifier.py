"""
Weaviate-based hybrid attribute classifier with multilingual support.

This classifier uses a hybrid approach:
1. Vector search first: Query Weaviate for similar goods descriptions
2. LLM fallback: For low similarity matches, use LLM classification
3. Knowledge base update: Store new high-confidence classifications

Key features:
- Multilingual support with multilingual-e5-large model (local)
- Per-attribute granular matching for higher precision
- Reranking for improved accuracy (local bge-reranker)
- Attribute-specific similarity thresholds
- Batch processing with parallel execution
- Checkpointing for fault tolerance
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm

from src.utils.weaviate_client import WeaviateClient
from src.utils.llm_client import LLMClient
from src.utils.config_models import (
    AttributeDefinitions,
    AttributeSchema,
    load_attribute_definitions,
    load_attribute_schema,
)
from src.classifiers.attribute_classifier.llm import classify_batch
from src.classifiers.attribute_classifier.utils import TokenTracker

logger = logging.getLogger(__name__)


@dataclass
class AttributeClassificationResult:
    """Result of attribute classification for a single goods description."""
    goods_shipped: str
    hs_code: str
    category: str
    attributes: Dict[str, str]  # attribute_name -> value
    similarity_scores: Dict[str, float]  # attribute_name -> similarity score
    method: str  # 'vector' or 'llm'
    metadata: Dict[str, Any]  # additional metadata


class WeaviateAttributeClassifier:
    """
    Hybrid attribute classifier using Weaviate vector search with LLM fallback.
    
    Optimized for multilingual accuracy and precision with:
    - multilingual-e5-large embedding model
    - Per-attribute granular search
    - Reranking for relevance
    - Configurable similarity thresholds
    """
    
    def __init__(
        self,
        class_name: str,
        product_attributes_schema_path: str,
        attribute_definitions_path: str,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        embedding_model: str = "intfloat/multilingual-e5-large",
        text_field: str = "goods_shipped",
        similarity_threshold: float = 0.75,
        per_attribute_threshold: Optional[Dict[str, float]] = None,
        top_k: int = 10,
        rerank_top_n: int = 5,
        weaviate_url: Optional[str] = None,
        weaviate_api_key: Optional[str] = None
    ):
        """
        Initialize Weaviate attribute classifier.
        
        Args:
            class_name: Name of Weaviate class
            product_attributes_schema_path: Path to product attributes schema JSON
            attribute_definitions_path: Path to attribute definitions JSON
            model_name: LLM model for fallback classification
            api_key: API key for LLM
            embedding_model: Embedding model name
            text_field: Name of text field in records
            similarity_threshold: Default similarity threshold for vector matches
            per_attribute_threshold: Custom thresholds per attribute
            top_k: Number of candidates to retrieve
            rerank_top_n: Number of results after reranking
            weaviate_url: URL for Weaviate instance
            weaviate_api_key: API Key for Weaviate instance
        """
        self.class_name = class_name
        self.text_field = text_field
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.per_attribute_threshold = per_attribute_threshold or {}
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        
        # Initialize Weaviate client
        logger.info(f"Initializing Weaviate client for class '{class_name}'")
        self.weaviate_client = WeaviateClient(
            url=weaviate_url,
            api_key=weaviate_api_key,
            class_name=class_name,
            embedding_model_name=embedding_model,
            text_field=text_field
        )
        
        # Warmup models to prevent race conditions in threads
        self.weaviate_client.warmup()
        
        # Initialize LLM client for fallback
        self.llm_client = LLMClient(model_name=model_name, api_key=api_key)
        
        # Load schemas
        self.schema_model = load_attribute_schema(product_attributes_schema_path)
        self.attr_definitions_model = load_attribute_definitions(attribute_definitions_path)
        self.attr_definitions = self.attr_definitions_model.definitions
        
        # Build attribute schema map: {hs4: {product: {attribute: [values]}}}
        self.schema_map = self._build_schema_map()
        
        # Statistics
        self.stats = {
            'vector_matches': 0,
            'llm_fallbacks': 0,
            'total_classifications': 0,
            'total_llm_tokens': 0,
            'attribute_accuracy': {},  # Per-attribute vector match rate
        }
    
    def close(self):
        """Close the Weaviate client connection."""
        if self.weaviate_client:
            self.weaviate_client.close()

    def _build_schema_map(self) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        """Build normalized schema map from schema model."""
        schema_map = {}
        for hs4, entry in self.schema_model.entries.items():
            products = {}
            for product_name, attr_set in entry.products.items():
                products[product_name] = attr_set.normalized_values()
            if products:
                schema_map[hs4] = products
        return schema_map
    
    def classify_single_attribute(
        self,
        goods_description: str,
        hs_code: str,
        category: str,
        attribute: str,
        allowed_values: List[str]
    ) -> Tuple[str, float, str]:
        """
        Classify a single attribute for a goods description.
        
        Args:
            goods_description: Text description of goods
            hs_code: HS code (4-digit)
            category: Product category
            attribute: Attribute name
            allowed_values: List of allowed values for this attribute
        
        Returns:
            Tuple of (value, similarity_score, method)
            - value: Classified attribute value or "None"
            - similarity_score: Confidence score (0-1)
            - method: "vector" or "llm"
        """
        threshold = self.per_attribute_threshold.get(attribute, self.similarity_threshold)
        
        try:
            # Query Weaviate with reranking
            # Note: We pass filters directly instead of namespace
            results = self.weaviate_client.search_with_rerank(
                query_text=goods_description,
                hs_code=hs_code,
                category=category,
                attribute_name=attribute,
                top_k=self.top_k,
                rerank_top_n=self.rerank_top_n
            )
            
            # search_with_rerank returns a list of matches
            if not results or not isinstance(results, list) or len(results) == 0:
                return "None", 0.0, "vector_no_match"
            
            # Get top match - results is a list of dicts
            top_hit = results[0]
            if not isinstance(top_hit, dict):
                logger.error(f"Expected dict but got {type(top_hit)}: {top_hit}")
                return "None", 0.0, "vector_error"
                
            similarity = top_hit.get('score', 0.0)
            
            # Extract attribute value from fields
            fields = top_hit.get('fields', {})
            value = fields.get('attribute_value', 'None')
            
            # Check if similarity meets threshold
            if similarity >= threshold:
                # Validate value is in allowed list
                if value in allowed_values or value == 'None':
                    return value, similarity, "vector"
                else:
                    logger.warning(
                        f"Vector match returned out-of-schema value '{value}' "
                        f"for attribute '{attribute}'. Treating as no match."
                    )
                    return "None", similarity, "vector_invalid"
            else:
                return "None", similarity, "vector_low_similarity"
        
        except Exception as e:
            logger.error(f"Weaviate query error for attribute '{attribute}': {e}")
            return "None", 0.0, "vector_error"
    
    def classify_single_goods_weaviate(
        self,
        goods_description: str,
        hs_code: str,
        category: str
    ) -> AttributeClassificationResult:
        """
        Classify all attributes for a single goods description using Weaviate.
        
        Args:
            goods_description: Text description of goods
            hs_code: HS code (4-digit)
            category: Product category
        
        Returns:
            AttributeClassificationResult with all attributes classified
        """
        hs4 = str(hs_code)[:4].zfill(4)
        
        # Get attribute schema for this product
        if hs4 not in self.schema_map or category not in self.schema_map[hs4]:
            logger.warning(f"No schema found for HS {hs4}, category '{category}'")
            return AttributeClassificationResult(
                goods_shipped=goods_description,
                hs_code=hs4,
                category=category,
                attributes={},
                similarity_scores={},
                method="no_schema",
                metadata={'error': 'No schema found'}
            )
        
        attr_values = self.schema_map[hs4][category]
        attributes = {}
        similarity_scores = {}
        method_counts = {'vector': 0, 'other': 0}
        
        # Classify each attribute independently
        for attribute, allowed_values in attr_values.items():
            value, similarity, method = self.classify_single_attribute(
                goods_description, hs_code, category, attribute, allowed_values
            )
            
            attributes[attribute] = value
            similarity_scores[attribute] = similarity
            
            if method == "vector":
                method_counts['vector'] += 1
            else:
                method_counts['other'] += 1
        
        # Determine overall method
        overall_method = "vector" if method_counts['vector'] > 0 else "no_match"
        
        return AttributeClassificationResult(
            goods_shipped=goods_description,
            hs_code=hs4,
            category=category,
            attributes=attributes,
            similarity_scores=similarity_scores,
            method=overall_method,
            metadata={
                'vector_matches': method_counts['vector'],
                'total_attributes': len(attr_values),
                'avg_similarity': sum(similarity_scores.values()) / len(similarity_scores) if similarity_scores else 0.0
            }
        )
    
    def classify_batch_llm(
        self,
        goods_list: List[str],
        hs_code: str,
        category: str,
        token_tracker: TokenTracker,
        batch_num: int,
        total_batches: int,
        config: Dict[str, Any]
    ) -> List[AttributeClassificationResult]:
        """
        Classify a batch of goods using LLM.
        
        Args:
            goods_list: List of goods descriptions
            hs_code: HS code (4-digit)
            category: Product category
            token_tracker: Token usage tracker
            batch_num: Current batch number
            total_batches: Total number of batches
            config: Configuration dictionary
        
        Returns:
            List of AttributeClassificationResult objects
        """
        hs4 = str(hs_code)[:4].zfill(4)
        
        if hs4 not in self.schema_map or category not in self.schema_map[hs4]:
            logger.warning(f"No schema found for HS {hs4}, category '{category}'")
            return [
                AttributeClassificationResult(
                    goods_shipped=g,
                    hs_code=hs4,
                    category=category,
                    attributes={},
                    similarity_scores={},
                    method="no_schema",
                    metadata={'error': 'No schema found'}
                )
                for g in goods_list
            ]
        
        attr_values = self.schema_map[hs4][category]
        
        # Call LLM batch classifier
        llm_results = classify_batch(
            hs4=hs4,
            product=category,
            attr_values=attr_values,
            goods_list=goods_list,
            model=self.llm_client,
            tracker=token_tracker,
            batch_num=batch_num,
            total_batches=total_batches,
            attr_definitions=self.attr_definitions,
            config=config
        )
        
        # Convert to AttributeClassificationResult
        results = []
        for llm_result in llm_results:
            results.append(
                AttributeClassificationResult(
                    goods_shipped=llm_result['goods_shipped'],
                    hs_code=hs4,
                    category=category,
                    attributes=llm_result['attributes'],
                    similarity_scores={attr: 0.0 for attr in llm_result['attributes']},
                    method="llm",
                    metadata=llm_result.get('_validation_meta', {})
                )
            )
        
        return results
    
    def classify_batch_hybrid(
        self,
        goods_list: List[str],
        hs_code: str,
        category: str,
        token_tracker: TokenTracker,
        batch_num: int,
        total_batches: int,
        config: Dict[str, Any]
    ) -> List[AttributeClassificationResult]:
        """
        Classify a batch using hybrid approach: Weaviate first, LLM fallback.
        
        Args:
            goods_list: List of goods descriptions
            hs_code: HS code
            category: Product category
            token_tracker: Token usage tracker
            batch_num: Current batch number
            total_batches: Total number of batches
            config: Configuration dictionary
        
        Returns:
            List of AttributeClassificationResult objects
        """
        results = [None] * len(goods_list)
        llm_fallback_goods = []
        llm_fallback_indices = []
        
        # Step 1: Try Weaviate for all goods in parallel
        # Use up to 10 workers or len(goods_list)
        max_workers = min(len(goods_list), 10)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.classify_single_goods_weaviate, goods, hs_code, category): idx
                for idx, goods in enumerate(goods_list)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    weaviate_result = future.result()
                    
                    # Check if we have enough vector matches
                    vector_match_count = weaviate_result.metadata.get('vector_matches', 0)
                    total_attrs = weaviate_result.metadata.get('total_attributes', 1)
                    match_rate = vector_match_count / total_attrs if total_attrs > 0 else 0.0
                    
                    # Only use LLM fallback if NO attributes matched (0% match rate)
                    # This preserves partial vector matches and only uses LLM when truly needed
                    if match_rate == 0.0 and weaviate_result.method != "no_schema":
                        # Will be added to fallback list later
                        pass
                    else:
                        results[idx] = weaviate_result
                        if weaviate_result.method == "vector":
                            self.stats['vector_matches'] += 1
                            
                except Exception as e:
                    logger.error(f"Error classifying goods at index {idx}: {e}")
        
        # Step 2: Identify items needing LLM fallback
        for idx, res in enumerate(results):
            if res is None:
                llm_fallback_goods.append(goods_list[idx])
                llm_fallback_indices.append(idx)
        
        # Step 3: LLM fallback for zero-match goods only
        if llm_fallback_goods:
            logger.info(
                f"Using LLM fallback for {len(llm_fallback_goods)}/{len(goods_list)} goods "
                f"(0% vector match - no attributes found)"
            )
            
            llm_results = self.classify_batch_llm(
                llm_fallback_goods,
                hs_code,
                category,
                token_tracker,
                batch_num,
                total_batches,
                config
            )
            
            # Fill in LLM results
            for idx, llm_result in zip(llm_fallback_indices, llm_results):
                results[idx] = llm_result
                self.stats['llm_fallbacks'] += 1
        
        self.stats['total_classifications'] += len(goods_list)
        
        return results
    
    def store_new_classifications(
        self,
        results: List[AttributeClassificationResult],
        min_similarity: float = 0.95
    ):
        """
        Store high-confidence classifications back to Weaviate knowledge base.
        
        Args:
            results: List of classification results
            min_similarity: Minimum average similarity to store (default 0.95)
        """
        records_to_upsert = []
        
        for result in results:
            # Only store high-confidence results
            if result.method != "vector" and result.method != "llm":
                continue
            
            avg_similarity = result.metadata.get('avg_similarity', 0.0)
            
            # For vector matches, require high similarity
            # For LLM results, store only if we have deterministic matches
            if result.method == "vector" and avg_similarity < min_similarity:
                continue
            
            # Store each attribute separately
            for attribute, value in result.attributes.items():
                if value == "None":
                    continue
                
                # Create record for this attribute
                # Note: We map keys to match WeaviateClient.upsert_records expectation
                record = {
                    "_id": f"{result.hs_code}_{result.category}_{attribute}_{hash(result.goods_shipped)}",
                    self.text_field: result.goods_shipped,
                    "attribute_name": attribute,
                    "attribute_value": value,
                    "hs_code": result.hs_code,
                    "category": result.category,
                    "confidence": result.similarity_scores.get(attribute, 0.0),
                    "classification_method": result.method,
                    "timestamp": datetime.now().isoformat()
                }
                
                records_to_upsert.append(record)
        
        # Upsert to Weaviate
        if records_to_upsert:
            try:
                self.weaviate_client.upsert_records(records_to_upsert)
                logger.info(f"Stored {len(records_to_upsert)} new attribute classifications to knowledge base")
            except Exception as e:
                logger.error(f"Failed to store records to Weaviate: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get classification statistics."""
        stats = self.stats.copy()
        
        if stats['total_classifications'] > 0:
            stats['vector_match_rate'] = stats['vector_matches'] / stats['total_classifications']
            stats['llm_fallback_rate'] = stats['llm_fallbacks'] / stats['total_classifications']
        else:
            stats['vector_match_rate'] = 0.0
            stats['llm_fallback_rate'] = 0.0
        
        return stats
