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
from datetime import datetime
from tqdm import tqdm

import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import Filter
import weaviate.classes as wvc

# Try importing embedding libraries
try:
    import torch
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_LOCAL_MODELS = True
except ImportError:
    HAS_LOCAL_MODELS = False

logger = logging.getLogger(__name__)

class WeaviateClient:
    """Client for managing Weaviate vector database operations."""
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        class_name: str = "AttributeClassification",
        embedding_model_name: str = "intfloat/multilingual-e5-large",
        rerank_model_name: str = "BAAI/bge-reranker-v2-m3",
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
        self._embedding_model = None
        self._rerank_model = None
        
    @property
    def embedding_model(self):
        if not self._embedding_model:
            if not HAS_LOCAL_MODELS:
                raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
            
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
                
            logger.info(f"Loading embedding model: {self.embedding_model_name} on {device}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
        return self._embedding_model

    @property
    def rerank_model(self):
        if not self._rerank_model:
            if not HAS_LOCAL_MODELS:
                raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
            
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"

            logger.info(f"Loading rerank model: {self.rerank_model_name} on {device}")
            self._rerank_model = CrossEncoder(self.rerank_model_name, device=device)
        return self._rerank_model

    def ensure_schema(self):
        """Create the schema if it doesn't exist."""
        if self.client.collections.exists(self.class_name):
            logger.info(f"Class '{self.class_name}' already exists")
            return

        logger.info(f"Creating class '{self.class_name}'")
        self.client.collections.create(
            name=self.class_name,
            vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
            properties=[
                Property(name=self.text_field, data_type=DataType.TEXT),
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

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        # e5 models need "query: " or "passage: " prefix. 
        # For goods descriptions (passages), we usually use "passage: "
        # But let's check how Pinecone does it. Pinecone's integrated might handle it.
        # Standard e5 usage: "passage: " for docs, "query: " for queries.
        passages = [f"passage: {t}" for t in texts]
        embeddings = self.embedding_model.encode(passages, normalize_embeddings=True)
        return embeddings.tolist()

    def _generate_query_embedding(self, text: str) -> List[float]:
        """Generate embedding for a query."""
        query = f"query: {text}"
        embedding = self.embedding_model.encode(query, normalize_embeddings=True)
        return embedding.tolist()

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
                properties = {
                    self.text_field: record[self.text_field],
                    "hs_code": record["hs_code"],
                    "category": record["category"],
                    "confidence": record.get("confidence", 1.0),
                    "classification_method": record.get("classification_method", "initial"),
                    "original_id": record.get("_id", ""),
                    "created_at": datetime.utcnow().isoformat() + "Z"
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
        Search with vector similarity and local reranking.
        
        Args:
            query_text: The search query
            hs_code: Filter by HS code (optional)
            category: Filter by category (optional)
            attribute_name: Filter by attribute name (optional)
            top_k: Initial candidates
            rerank_top_n: Final results
        """
        collection = self.client.collections.get(self.class_name)
        
        # 1. Vector Search
        # query_vector = self._generate_query_embedding(query_text)
        
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
        
        response = collection.query.near_text(
            query=query_text,
            limit=top_k,
            filters=filters,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        
        if not response.objects:
            return []
            
        # 2. Reranking
        # Prepare pairs for cross-encoder: [[query, doc1], [query, doc2], ...]
        candidates = []
        for obj in response.objects:
            candidates.append({
                "id": str(obj.uuid),
                "score": 1 - obj.metadata.distance, # Convert distance to similarity
                "fields": obj.properties,
                "text": obj.properties[self.text_field]
            })
            
        if not candidates:
            return []
            
        pairs = [[query_text, c["text"]] for c in candidates]
        scores = self.rerank_model.predict(pairs)
        
        # Attach scores and sort
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])
            # Normalize sigmoid score if needed, but raw logits/scores are fine for sorting
            
        # Sort by rerank score descending
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Format result to match Pinecone style for compatibility
        final_results = []
        for c in candidates[:rerank_top_n]:
            final_results.append({
                "id": c["id"],
                "score": c["rerank_score"], # Use rerank score as final score
                "fields": c["fields"]
            })
            
        return final_results
            
        return final_results

    def close(self):
        self.client.close()
