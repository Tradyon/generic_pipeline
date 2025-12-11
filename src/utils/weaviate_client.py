"""
Weaviate client for vector-based product classification.

Replicates the functionality of PineconeClient but using Weaviate.
Features:
- Local embedding generation using sentence-transformers (matching Pinecone's integrated model)
- Local reranking (matching Pinecone's integrated reranker)
- Class-based schema with filters (replacing Pinecone namespaces)
"""

import os
import logging
import hashlib
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from tqdm import tqdm

import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import Filter, Rerank, MetadataQuery
import weaviate.classes as wvc

logger = logging.getLogger(__name__)

class WeaviateClient:
    """Client for managing Weaviate vector database operations."""
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        class_name: str = "AttributeClassification",
        embedding_model_name: str = "intfloat/multilingual-e5-large",
        rerank_model_name: str = "cross-encoder-ms-marco-MiniLM-L-6-v2",
        text_field: str = "goods_shipped"
    ):
        """
        Initialize Weaviate client.
        
        Args:
            url: Weaviate URL (default: env WEAVIATE_URL or localhost)
            api_key: Weaviate API Key (default: env WEAVIATE_API_KEY)
            class_name: Name of the class/collection
            embedding_model_name: HuggingFace model for embeddings
            rerank_model_name: HuggingFace model for reranking
            text_field: Field to vectorize
        """
        self.url = url or os.getenv("WEAVIATE_URL", "http://localhost:8080")
        self.api_key = api_key or os.getenv("WEAVIATE_API_KEY", "")
        self.class_name = class_name
        self.text_field = text_field
        
        # Connect to Weaviate
        headers = {}
        if os.getenv("OPENAI_API_KEY"):
            headers["X-OpenAI-Api-Key"] = os.getenv("OPENAI_API_KEY")
            
        if self.api_key:
            self.client = weaviate.connect_to_wcs(
                cluster_url=self.url,
                auth_credentials=weaviate.auth.AuthApiKey(self.api_key),
                headers=headers
            )
        else:
            self.client = weaviate.connect_to_local(
                headers=headers
            )
            
        # Initialize models lazily
        self.embedding_model_name = embedding_model_name
        self.rerank_model_name = rerank_model_name
        
    def warmup(self):
        """No-op for server-side models."""
        pass

    def close(self):
        """Close the Weaviate client connection."""
        if self.client:
            self.client.close()

    def ensure_schema(self):
        """Create the schema if it doesn't exist."""
        if self.client.collections.exists(self.class_name):
            logger.info(f"Class '{self.class_name}' already exists")
            return

        logger.info(f"Creating class '{self.class_name}'")
        self.client.collections.create(
            name=self.class_name,
            vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
            reranker_config=Configure.Reranker.transformers(),
            properties=[
                Property(name="text_vector", data_type=DataType.TEXT),
                Property(name=self.text_field, data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="attribute_name", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="attribute_value", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="hs_code", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="category", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="confidence", data_type=DataType.NUMBER, skip_vectorization=True),
                Property(name="classification_method", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="original_id", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="created_at", data_type=DataType.DATE, skip_vectorization=True),
                Property(name="shipment_count", data_type=DataType.INT, skip_vectorization=True),
            ]
        )
        logger.info(f"Class '{self.class_name}' created")

    def generate_id(self, text: str) -> str:
        """Generate deterministic ID."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def upsert_records(
        self,
        records: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> None:
        """
        Upsert records to Weaviate.
        
        Args:
            records: List of dicts containing data.
                     Must have 'goods_shipped', 'hs_code', 'category'.
                     Optional: 'attribute_name', 'attribute_value'.
        """
        self.ensure_schema()
        collection = self.client.collections.get(self.class_name)
        
        logger.info(f"Upserting {len(records)} records to Weaviate...")
        # Use fixed_size batching with concurrency for faster imports
        with collection.batch.fixed_size(batch_size=batch_size, concurrent_requests=4) as batch:
            for i, record in enumerate(records):
                # Prepare properties
                # E5 models require "passage: " prefix for documents
                raw_text = record[self.text_field]
                vector_text = f"passage: {raw_text}"

                properties = {
                    "text_vector": vector_text,
                    self.text_field: raw_text,
                    "hs_code": record["hs_code"],
                    "category": record["category"],
                    "confidence": record.get("confidence", 1.0),
                    "classification_method": record.get("classification_method", "initial"),
                    "original_id": record.get("_id", ""),
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                
                # Add optional fields if present
                if "attribute_name" in record:
                    properties["attribute_name"] = record["attribute_name"]
                if "attribute_value" in record:
                    properties["attribute_value"] = record["attribute_value"]
                if "shipment_count" in record:
                    properties["shipment_count"] = record["shipment_count"]
                
                # Weaviate uses UUIDs. We can generate a deterministic UUID from our ID.
                # Or let Weaviate generate one.
                # Using generate_uuid5 with a namespace is good practice for deterministic IDs.
                seed = record.get("_id", self.generate_id(record[self.text_field] + str(record.get("attribute_name", ""))))
                uuid = weaviate.util.generate_uuid5(seed)
                
                batch.add_object(
                    properties=properties,
                    uuid=uuid
                )
                
        if len(collection.batch.failed_objects) > 0:
            logger.error(f"Failed to import {len(collection.batch.failed_objects)} objects")
            for failed in collection.batch.failed_objects[:5]:
                logger.error(f"Error: {failed.message}")

    def search_with_rerank(
        self,
        query_text: str,
        hs_code: Optional[str] = None,
        category: Optional[str] = None,
        attribute_name: Optional[str] = None,
        top_k: int = 10,
        rerank_top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search with vector similarity and server-side reranking.
        
        Args:
            query_text: The search query
            hs_code: Filter by HS code (optional)
            category: Filter by category (optional)
            attribute_name: Filter by attribute name (optional)
            top_k: Initial candidates
            rerank_top_n: Final results
        """
        collection = self.client.collections.get(self.class_name)
        
        # Build filter
        filters = None
        
        if hs_code:
            f = Filter.by_property("hs_code").equal(hs_code)
            filters = f if filters is None else filters & f
            
        if category:
            f = Filter.by_property("category").equal(category)
            filters = f if filters is None else filters & f
            
        if attribute_name:
            f = Filter.by_property("attribute_name").equal(attribute_name)
            filters = f if filters is None else filters & f
        
        # E5 models require "query: " prefix for queries
        search_query = query_text
        if not search_query.startswith("query: "):
            search_query = f"query: {search_query}"

        try:
            # Use Weaviate's built-in reranker
            response = collection.query.near_text(
                query=search_query,
                limit=top_k,
                filters=filters,
                rerank=Rerank(
                    prop=self.text_field,
                    query=query_text  # Use raw query for reranker
                ),
                return_metadata=MetadataQuery(score=True)
            )
            
            if not response.objects:
                return []
                
            # Format result to match Pinecone style for compatibility
            final_results = []
            for obj in response.objects[:rerank_top_n]:
                props = obj.properties
                # Strip "passage: " prefix from text field if present, to keep it clean for the caller
                if self.text_field in props and isinstance(props[self.text_field], str):
                    if props[self.text_field].startswith("passage: "):
                        props[self.text_field] = props[self.text_field][9:]

                final_results.append({
                    "id": str(obj.uuid),
                    "score": obj.metadata.rerank_score if obj.metadata.rerank_score is not None else obj.metadata.score,
                    "fields": props
                })
                
            return final_results
            
        except Exception as e:
            logger.error(f"Error in search_with_rerank: {e}")
            return []
            
        return final_results

    def close(self):
        self.client.close()
