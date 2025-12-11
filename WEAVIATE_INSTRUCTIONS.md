# Weaviate Migration & Usage Instructions

This guide details how to run the newly migrated Weaviate-based pipeline for product and attribute classification.

## 1. Prerequisites

### Environment Setup

Ensure you have `uv` installed and dependencies synced:

```bash
uv sync
```

### Start Weaviate

Start a local Weaviate instance using Docker:

```bash
docker-compose up -d
```

This will start Weaviate at `http://localhost:8080`.

### Environment Variables

Ensure your `.env` file contains the necessary API keys (OpenAI/Gemini for LLM, and optionally Weaviate if you aren't using the local docker setup).

```bash
# .env
WEAVIATE_URL=http://localhost:8080
# WEAVIATE_API_KEY=  # Leave empty for local docker instance with anonymous access
GEMINI_API_KEY=...
# or
OPENAI_API_KEY=...
```

## 2. Populate Knowledge Base (Optional but Recommended)

If you have existing classified data (e.g., from the Pinecone run or manual labeling), populate Weaviate to enable the vector search component.

### Populate Products

Seeds the `ProductClassification` class.

```bash
uv run python scripts/populate_weaviate_products.py \
  --input output/coffee/shipment_master_classified.csv \
  --class-name ProductClassification
```

### Populate Attributes

Seeds the `AttributeClassification` class.

```bash
uv run python scripts/populate_weaviate_attributes.py \
  --input output/coffee/classifications.json \
  --class-name AttributeClassification
```

## 3. Run Classification Pipelines

### Product Classification

Classifies products into categories (e.g., "Coffee", "Rice") using Hybrid Search (Weaviate + LLM).

```bash
uv run python tradyon_classify_products_weaviate.py \
  --input sample_coffee_shipment_master.csv \
  --output output/weaviate_run/shipment_master_classified.csv \
  --products-definition config/products_definition.json \
  --model gemini-2.0-flash
```

### Attribute Classification

Classifies attributes (e.g., "Variety", "Grade") for the products identified above.

```bash
uv run python tradyon_classify_attributes_weaviate.py \
  --input output/weaviate_run/shipment_master_classified.csv \
  --output-dir output/weaviate_run \
  --product-attributes-schema config/product_attributes_schema.json \
  --attribute-definitions config/attribute_definitions.json \
  --model gemini-2.0-flash \
  --update-kb  # Optional: Adds new high-confidence findings back to Weaviate
```

## 4. Key Differences from Pinecone Pipeline

| Feature | Pinecone Implementation | Weaviate Implementation |
| :--- | :--- | :--- |
| **Database** | Pinecone (Cloud) | Weaviate (Local/Cloud) |
| **Embeddings** | Integrated (Server-side) | Local (`sentence-transformers`) |
| **Reranking** | Integrated (Server-side) | Local (`cross-encoder`) |
| **Isolation** | Namespaces | Filters (`hs_code`, `category`, etc.) |
| **Client** | `PineconeClient` | `WeaviateClient` |

## 5. Troubleshooting

* **Connection Refused**: Ensure Docker container is running (`docker ps`).
* **Missing Models**: The first run will download embedding models (~2GB) to `~/.cache/huggingface`. Ensure you have internet access.
* **Memory Issues**: If running out of RAM, reduce `--batch-size` (default 10) or `--max-workers`.
