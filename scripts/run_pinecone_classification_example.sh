#!/bin/bash
#
# Example workflow for Pinecone-based product classification
# This demonstrates the complete process from setup to analysis
#

set -e  # Exit on error

echo "========================================"
echo "Pinecone Classification System - Example"
echo "========================================"

# Configuration
INPUT_CSV="output/rice/shipment_master.csv"
EXISTING_CLASSIFICATIONS="output/rice/shipment_master_classified.csv"
OUTPUT_CSV="output/rice/shipment_master_classified_pinecone.csv"
PRODUCTS_DEF="config/rice/products_definition.json"
INDEX_NAME="product-classification-1006"
NAMESPACE="hs_1006"

echo ""
echo "Step 1: Test Pinecone Connection"
echo "--------------------------------"
python scripts/test_pinecone_setup.py

echo ""
echo "Step 2: Populate Pinecone Index"
echo "--------------------------------"
echo "Loading existing classifications into Pinecone..."

python scripts/populate_pinecone_index.py \
  --csv "$EXISTING_CLASSIFICATIONS" \
  --index-name "$INDEX_NAME" \
  --namespace "$NAMESPACE" \
  --embedding-model text-embedding-3-small \
  --dimension 1536 \
  --batch-size 100

echo ""
echo "Step 3: Run Hybrid Classification"
echo "----------------------------------"
echo "Classifying new data with Pinecone + LLM fallback..."

python tradyon_classify_products_pinecone.py \
  --input "$INPUT_CSV" \
  --output "$OUTPUT_CSV" \
  --products-definition "$PRODUCTS_DEF" \
  --pinecone-index "$INDEX_NAME" \
  --pinecone-namespace "$NAMESPACE" \
  --similarity-threshold 0.85 \
  --batch-size 10 \
  --max-workers 8

echo ""
echo "Step 4: Analyze Performance"
echo "---------------------------"
echo "Comparing Pinecone vs LLM results..."

python scripts/analyze_classification_performance.py \
  --pinecone-results "$OUTPUT_CSV" \
  --llm-results "$EXISTING_CLASSIFICATIONS" \
  --checkpoint pinecone_classification_checkpoint.pkl \
  --output classification_analysis.json

echo ""
echo "========================================"
echo "✅ Classification Complete!"
echo "========================================"
echo ""
echo "Results:"
echo "  - Classified data: $OUTPUT_CSV"
echo "  - Analysis report: classification_analysis.json"
echo ""
echo "Check the logs above for performance metrics."
