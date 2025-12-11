"""
Pinecone-based hybrid attribute classifier with multilingual support.

This classifier uses a hybrid approach:
1. Vector search first: Query Pinecone for similar goods descriptions
2. LLM fallback: For low similarity matches, use LLM classification
3. Knowledge base update: Store new high-confidence classifications

Key features:
- Multilingual support with multilingual-e5-large model
- Per-attribute granular matching for higher precision
- Reranking for improved accuracy
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

from src.utils.pinecone_client import PineconeClient
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


class PineconeAttributeClassifier:
    """
    Hybrid attribute classifier using Pinecone vector search with LLM fallback.
    
    Optimized for multilingual accuracy and precision with:
    - multilingual-e5-large embedding model
    - Per-attribute granular search
    - Reranking for relevance
    - Configurable similarity thresholds
    """
    
    def __init__(
        self,
        index_name: str,
        product_attributes_schema_path: str,
        attribute_definitions_path: str,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        embedding_model: str = "multilingual-e5-large",
        text_field: str = "goods_shipped",
        similarity_threshold: float = 0.75,
        per_attribute_threshold: Optional[Dict[str, float]] = None,
        top_k: int = 10,
        rerank_top_n: int = 5,
    ):
        """
        Initialize Pinecone attribute classifier.
        
        Args:
            index_name: Name of Pinecone index
            product_attributes_schema_path: Path to product attributes schema JSON
            attribute_definitions_path: Path to attribute definitions JSON
            model_name: LLM model for fallback classification
            api_key: API key for Pinecone and LLM
            embedding_model: Pinecone embedding model (multilingual-e5-large for multilingual)
            text_field: Name of text field in records
            similarity_threshold: Default similarity threshold for vector matches
            per_attribute_threshold: Custom thresholds per attribute
            top_k: Number of candidates to retrieve
            rerank_top_n: Number of results after reranking
        """
        self.index_name = index_name
        self.text_field = text_field
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.per_attribute_threshold = per_attribute_threshold or {}
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        
        # Initialize Pinecone client (reads API key from environment)
        logger.info(f"Initializing Pinecone client for index '{index_name}'")
        self.pinecone_client = PineconeClient(
            index_name=index_name,
            embedding_model=embedding_model,
            text_field=text_field
        )
        
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
    
    def _get_namespace(self, hs_code: str, category: str, attribute: str) -> str:
        """Generate namespace for attribute-specific search."""
        hs4 = str(hs_code)[:4].zfill(4)
        # Namespace format: hs_<code>_<product>_attr_<attribute>
        # Example: hs_9011_Bulk_Commodity_Green_Coffee_attr_Coffee_Variety
        safe_category = category.replace(" ", "_").replace("/", "_")
        safe_attribute = attribute.replace(" ", "_").replace("/", "_")
        return f"hs_{hs4}_{safe_category}_attr_{safe_attribute}"
    
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
        namespace = self._get_namespace(hs_code, category, attribute)
        threshold = self.per_attribute_threshold.get(attribute, self.similarity_threshold)
        
        try:
            # Query Pinecone with reranking
            results = self.pinecone_client.search_with_rerank(
                query_text=goods_description,
                namespace=namespace,
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
            
            # Extract attribute value from metadata
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
            logger.error(f"Pinecone query error for attribute '{attribute}': {e}")
            return "None", 0.0, "vector_error"
    
    def classify_single_goods_pinecone(
        self,
        goods_description: str,
        hs_code: str,
        category: str
    ) -> AttributeClassificationResult:
        """
        Classify all attributes for a single goods description using Pinecone.
        
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
        Classify a batch using hybrid approach: Pinecone first, LLM fallback.
        
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
        results = []
        llm_fallback_goods = []
        llm_fallback_indices = []
        
        # Step 1: Try Pinecone for all goods
        for idx, goods in enumerate(goods_list):
            pinecone_result = self.classify_single_goods_pinecone(goods, hs_code, category)
            
            # Check if we have enough vector matches
            vector_match_count = pinecone_result.metadata.get('vector_matches', 0)
            total_attrs = pinecone_result.metadata.get('total_attributes', 1)
            match_rate = vector_match_count / total_attrs if total_attrs > 0 else 0.0
            
            # Only use LLM fallback if NO attributes matched (0% match rate)
            # This preserves partial vector matches and only uses LLM when truly needed
            if match_rate == 0.0 and pinecone_result.method != "no_schema":
                llm_fallback_goods.append(goods)
                llm_fallback_indices.append(idx)
                results.append(None)  # Placeholder
            else:
                results.append(pinecone_result)
                if pinecone_result.method == "vector":
                    self.stats['vector_matches'] += 1
        
        # Step 2: LLM fallback for zero-match goods only
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
        Store high-confidence classifications back to Pinecone knowledge base.
        
        Args:
            results: List of classification results
            min_similarity: Minimum average similarity to store (default 0.95)
        """
        # Group results by namespace (hs_code + category + attribute)
        namespace_records = {}
        
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
                
                namespace = self._get_namespace(result.hs_code, result.category, attribute)
                
                if namespace not in namespace_records:
                    namespace_records[namespace] = []
                
                # Create record for this attribute
                record = {
                    "_id": f"{result.hs_code}_{result.category}_{attribute}_{hash(result.goods_shipped)}",
                    self.text_field: result.goods_shipped,
                    "attribute_name": attribute,
                    "attribute_value": value,
                    "hs_code": result.hs_code,
                    "category": result.category,
                    "similarity": result.similarity_scores.get(attribute, 0.0),
                    "method": result.method,
                    "timestamp": datetime.now().isoformat()
                }
                
                namespace_records[namespace].append(record)
        
        # Upsert to Pinecone
        stored_count = 0
        for namespace, records in namespace_records.items():
            if records:
                try:
                    self.pinecone_client.upsert_records(namespace, records)
                    stored_count += len(records)
                    logger.debug(f"Stored {len(records)} records to namespace '{namespace}'")
                except Exception as e:
                    logger.error(f"Failed to store records to namespace '{namespace}': {e}")
        
        if stored_count > 0:
            logger.info(f"Stored {stored_count} new attribute classifications to knowledge base")
    
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
