"""
Pinecone client for vector-based product classification.

Uses Pinecone's integrated embedding models (no external API required).
Provides utilities for:
- Creating and managing Pinecone indexes with integrated embeddings
- Upserting text records (Pinecone handles embedding automatically)
- Querying with similarity search and reranking
"""

import os
import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime

from pinecone import Pinecone
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PineconeClient:
    """Client for managing Pinecone vector database operations with integrated embeddings."""
    
    def __init__(
        self,
        index_name: str,
        embedding_model: str = "llama-text-embed-v2",
        text_field: str = "content",
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        """
        Initialize Pinecone client with integrated embedding model.
        
        Args:
            index_name: Name of the Pinecone index
            embedding_model: Pinecone integrated embedding model
                - "llama-text-embed-v2" (recommended, multilingual)
                - "multilingual-e5-large" (1024 dims, multilingual)
            text_field: Field name to use for text content (e.g., "content", "text")
            metric: Distance metric (cosine, euclidean, dotproduct)
            cloud: Cloud provider (aws, gcp, azure)
            region: Cloud region
        """
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.text_field = text_field
        self.metric = metric
        self.cloud = cloud
        self.region = region
        
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        self.pc = Pinecone(api_key=api_key)
        
        # Index reference (lazy loaded)
        self._index = None
    
    def create_index(self) -> None:
        """Create Pinecone index with integrated embedding model if it doesn't exist."""
        if self.pc.has_index(self.index_name):
            logger.info(f"Index '{self.index_name}' already exists")
            
            # Verify it has integrated inference
            try:
                index_info = self.pc.describe_index(self.index_name)
                # Check if index has integrated inference configured
                if not hasattr(index_info, 'embed') or not index_info.embed:
                    logger.warning(
                        f"Index '{self.index_name}' exists but does not have integrated embeddings configured.\n"
                        f"Please delete it first: pc.delete_index('{self.index_name}')"
                    )
                    raise ValueError(
                        f"Index '{self.index_name}' is not configured for integrated inference. "
                        f"Delete it and re-run to create with proper configuration."
                    )
            except Exception as e:
                logger.warning(f"Could not verify index configuration: {e}")
            
            return
        
        logger.info(f"Creating index '{self.index_name}' with integrated embedding model '{self.embedding_model}'")
        
        # Create index with integrated embeddings (Pinecone handles embedding automatically)
        self.pc.create_index_for_model(
            name=self.index_name,
            cloud=self.cloud,
            region=self.region,
            embed={
                "model": self.embedding_model,
                "field_map": {"text": self.text_field}  # Map "text" input to our field name
            }
        )
        
        logger.info(f"Index '{self.index_name}' created successfully with integrated embeddings")
    
    @property
    def index(self):
        """Lazy load index connection."""
        if self._index is None:
            if not self.pc.has_index(self.index_name):
                raise ValueError(f"Index '{self.index_name}' does not exist. Call create_index() first.")
            self._index = self.pc.Index(self.index_name)
        return self._index
    
    @staticmethod
    def generate_id(text: str) -> str:
        """Generate deterministic ID from text using hash."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def upsert_records(
        self,
        namespace: str,
        records: List[Dict[str, Any]],
        batch_size: int = 96  # Max for text records per Pinecone docs
    ) -> None:
        """
        Upsert text records to Pinecone index with integrated embeddings.
        
        Pinecone will automatically generate embeddings using the integrated model.
        
        Args:
            namespace: Namespace to upsert to
            records: List of records with format:
                {
                    'goods_shipped': str,  # The text content to embed
                    'category': str,
                    'hs_code': str,
                    'confidence': float (optional),
                    'classification_method': str (optional),
                    'shipment_count': int (optional)
                }
            batch_size: Number of records per upsert batch (max 96 for text records)
        """
        logger.info(f"Upserting {len(records)} text records to namespace '{namespace}'")
        
        # Prepare records for upsert (Pinecone will handle embedding)
        pinecone_records = []
        for record in records:
            record_id = self.generate_id(record['goods_shipped'])
            
            # Build record with text field and metadata
            pinecone_record = {
                '_id': record_id,
                self.text_field: record['goods_shipped'],  # Text to be embedded
                'category': record['category'],
                'hs_code': record.get('hs_code', ''),
                'confidence': record.get('confidence', 1.0),
                'classification_method': record.get('classification_method', 'initial'),
                'shipment_count': record.get('shipment_count', 1),
                'created_at': datetime.utcnow().isoformat()
            }
            pinecone_records.append(pinecone_record)
        
        # Upsert in batches (max 96 for text records, 2MB total per batch)
        for i in tqdm(range(0, len(pinecone_records), batch_size), desc="Upserting to Pinecone"):
            batch = pinecone_records[i:i + batch_size]
            self.index.upsert_records(namespace, batch)
        
        logger.info(f"Successfully upserted {len(pinecone_records)} records")
    
    def search_with_rerank(
        self,
        namespace: str,
        query_text: str,
        top_k: int = 10,
        rerank_top_n: int = 5,
        rerank_model: str = "bge-reranker-v2-m3",
        rank_fields: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search Pinecone with semantic search and reranking.
        
        Uses integrated embedding model for query, then reranks for better relevance.
        
        Args:
            namespace: Namespace to query
            query_text: Text to search for
            top_k: Number of initial candidates to retrieve
            rerank_top_n: Number of results after reranking
            rerank_model: Reranking model (bge-reranker-v2-m3 recommended)
            rank_fields: Fields to use for reranking (e.g., ["content"])
            filter_dict: Metadata filters (optional)
            
        Returns:
            List of matches with format:
                {
                    'id': str,
                    'score': float,
                    'fields': dict (contains text content and metadata)
                }
        """
        # Build query dict
        query_dict = {
            "top_k": top_k,
            "inputs": {
                "text": query_text  # Pinecone will embed this automatically
            }
        }
        
        # Add filter only if provided
        if filter_dict:
            query_dict["filter"] = filter_dict
        
        # Build rerank dict
        rerank_dict = {
            "model": rerank_model,
            "top_n": rerank_top_n,
            "rank_fields": rank_fields or [self.text_field]
        }
        
        # Execute search with reranking
        results = self.index.search(
            namespace=namespace,
            query=query_dict,
            rerank=rerank_dict
        )
        
        # Format results - use dict-style access as per AGENTS.md
        matches = []
        if 'result' in results and 'hits' in results['result']:
            for hit in results['result']['hits']:
                matches.append({
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'fields': hit.get('fields', {})
                })
        
        return matches
    
    def search_batch(
        self,
        namespace: str,
        query_texts: List[str],
        top_k: int = 5,
        use_reranking: bool = True,
        rerank_top_n: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Search multiple texts in batch.
        
        Args:
            namespace: Namespace to query
            query_texts: List of texts to search for
            top_k: Number of results per query
            use_reranking: Whether to use reranking
            rerank_top_n: Number of results after reranking
            filter_dict: Metadata filters
            
        Returns:
            List of match lists (one per query)
        """
        all_results = []
        
        for query_text in tqdm(query_texts, desc="Searching Pinecone"):
            if use_reranking:
                results = self.search_with_rerank(
                    namespace=namespace,
                    query_text=query_text,
                    top_k=top_k,
                    rerank_top_n=rerank_top_n,
                    filter_dict=filter_dict
                )
            else:
                # Simple search without reranking
                query_dict = {
                    "top_k": top_k,
                    "inputs": {"text": query_text}
                }
                if filter_dict:
                    query_dict["filter"] = filter_dict
                
                results_raw = self.index.search(
                    namespace=namespace,
                    query=query_dict
                )
                
                results = []
                if 'result' in results_raw and 'hits' in results_raw['result']:
                    for hit in results_raw['result']['hits']:
                        results.append({
                            'id': hit['_id'],
                            'score': hit['_score'],
                            'fields': hit.get('fields', {})
                        })
            
            all_results.append(results)
        
        return all_results
    
    def get_index_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about the index."""
        stats = self.index.describe_index_stats()
        
        if namespace and hasattr(stats, 'namespaces') and namespace in stats.namespaces:
            return {
                'total_vector_count': stats.total_vector_count,
                'namespace_vector_count': stats.namespaces[namespace].vector_count,
                'dimension': stats.dimension
            }
        
        return {
            'total_vector_count': stats.total_vector_count,
            'dimension': stats.dimension,
            'namespaces': list(stats.namespaces.keys()) if hasattr(stats, 'namespaces') else []
        }
    
    def delete_namespace(self, namespace: str) -> None:
        """Delete all vectors in a namespace."""
        logger.warning(f"Deleting all vectors in namespace '{namespace}'")
        self.index.delete(delete_all=True, namespace=namespace)
        logger.info(f"Namespace '{namespace}' cleared")
