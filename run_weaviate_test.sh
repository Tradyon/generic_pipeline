#!/bin/bash
# End-to-end testing script for Weaviate Attribute Classification
# 
# This script runs the complete workflow:
# 1. Generate LLM baseline classifications
# 2. Populate Weaviate knowledge base
# 3. Run Weaviate hybrid classification
# 4. Analyze performance comparison
#
# Prerequisites:
#   export WEAVIATE_URL="http://localhost:8080" (or your WCS URL)
#   export WEAVIATE_API_KEY="your-api-key" (optional for local)
#   export GEMINI_API_KEY="your-api-key"

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Weaviate Attribute Classification Test${NC}"
echo -e "${BLUE}========================================${NC}"

# Load .env file if it exists
if [ -f .env ]; then
    echo "Loading environment from .env file..."
    # Parse .env file (handle both export and non-export formats)
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$line" ]] && continue
        
        # Remove 'export ' prefix if present
        line="${line#export }"
        
        # Extract key=value, handling spaces around =
        if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.+)$ ]]; then
            key="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"
            # Remove quotes from value
            value="${value%\"}"
            value="${value#\"}"
            value="${value%\'}"
            value="${value#\'}"
            
            export "$key=$value"
        fi
    done < .env
fi

# Map GOOGLE_API_KEY to GEMINI_API_KEY if needed
if [ -z "$GEMINI_API_KEY" ] && [ -n "$GOOGLE_API_KEY" ]; then
    export GEMINI_API_KEY="$GOOGLE_API_KEY"
    echo "Using GOOGLE_API_KEY as GEMINI_API_KEY"
fi

# Check prerequisites
echo "Checking environment variables..."
echo "WEAVIATE_URL: ${WEAVIATE_URL:-http://localhost:8080}"
echo "GEMINI_API_KEY: ${GEMINI_API_KEY:0:10}..."

if [ -z "$GEMINI_API_KEY" ]; then
    echo -e "${YELLOW}Error: GEMINI_API_KEY or GOOGLE_API_KEY not set${NC}"
    echo ""
    echo "Add to .env file:"
    echo '  GOOGLE_API_KEY = "your-google-api-key"'
    echo ""
    echo "Or export manually:"
    echo "  export GEMINI_API_KEY='your-api-key'"
    exit 1
fi

# Configuration
INPUT_CSV="output/coffee/shipment_master_classified_weaviate.csv"
LLM_OUTPUT_DIR="output/coffee_attributes_llm"
WEAVIATE_OUTPUT_DIR="output/coffee_attributes_weaviate"
SCHEMA_FILE="config/product_attributes_schema_example.json"
DEFINITIONS_FILE="config/attribute_definitions_example.json"
CLASS_NAME="AttributeClassification"
EMBEDDING_MODEL="intfloat/multilingual-e5-large"

# Ensure input CSV exists (using sample if needed)
if [ ! -f "$INPUT_CSV" ]; then
    if [ -f "sample_coffee_shipment_master.csv" ]; then
        echo "Preparing input data from sample_coffee_shipment_master.csv..."
        uv run python scripts/prepare_test_data.py "sample_coffee_shipment_master.csv" "prepared_input.csv"
        INPUT_CSV="prepared_input.csv"
    else
        echo -e "${YELLOW}Error: Input CSV not found and sample not available.${NC}"
        exit 1
    fi
fi

echo -e "\n${GREEN}Step 1: Generate LLM Baseline Classifications${NC}"
echo "Input: $INPUT_CSV"
echo "Output: $LLM_OUTPUT_DIR"

if [ -f "$LLM_OUTPUT_DIR/classifications_flat.csv" ]; then
    echo -e "${YELLOW}LLM classifications already exist, skipping...${NC}"
else
    uv run python tradyon_classify_attributes.py \
        --input "$INPUT_CSV" \
        --output-dir "$LLM_OUTPUT_DIR" \
        --product-attributes-schema "$SCHEMA_FILE" \
        --attribute-definitions "$DEFINITIONS_FILE" \
        --items-per-call 5 \
        --max-workers 4
fi

# Extract coffee-only records for Knowledge Base
echo -e "\n${GREEN}Extracting coffee-only records for Knowledge Base...${NC}"
COFFEE_CSV="$LLM_OUTPUT_DIR/coffee_classifications.csv"

# Check if already exists
if [ -f "$COFFEE_CSV" ]; then
    echo -e "${YELLOW}Coffee classifications already extracted, skipping...${NC}"
else
    # Extract header
    head -1 "$LLM_OUTPUT_DIR/classifications_flat.csv" > "$COFFEE_CSV"
    # Extract coffee records (HS code 9011)
    grep "^9011," "$LLM_OUTPUT_DIR/classifications_flat.csv" >> "$COFFEE_CSV" || true
    
    COFFEE_COUNT=$(wc -l < "$COFFEE_CSV")
    echo "Extracted $((COFFEE_COUNT - 1)) coffee records"
fi

echo -e "\n${GREEN}Step 2: Populate Weaviate Knowledge Base${NC}"
echo "Class: $CLASS_NAME"
echo "Embedding model: $EMBEDDING_MODEL"

uv run python scripts/populate_weaviate_attributes.py \
    --input "$COFFEE_CSV" \
    --class-name "$CLASS_NAME" \
    --embedding-model "$EMBEDDING_MODEL" \
    --batch-size 50

echo -e "\n${GREEN}Step 3: Run Weaviate Hybrid Classification${NC}"
echo "Input: $INPUT_CSV"
echo "Output: $WEAVIATE_OUTPUT_DIR"

uv run python tradyon_classify_attributes_weaviate.py \
    --input "$INPUT_CSV" \
    --output-dir "$WEAVIATE_OUTPUT_DIR" \
    --product-attributes-schema "$SCHEMA_FILE" \
    --attribute-definitions "$DEFINITIONS_FILE" \
    --class-name "$CLASS_NAME" \
    --items-per-call 5 \
    --no-resume

echo -e "\n${GREEN}Step 4: Analyze Performance${NC}"

uv run python scripts/analyze_attribute_performance.py \
    --pinecone "$WEAVIATE_OUTPUT_DIR/classifications_flat.csv" \
    --llm "$LLM_OUTPUT_DIR/classifications_flat.csv" \
    --output "weaviate_attribute_classification_analysis.json"

echo -e "\n${GREEN}Step 5: Cleanup Weaviate${NC}"
uv run python scripts/cleanup_weaviate.py --class-name "$CLASS_NAME" --force

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Testing Complete!${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "\nResults:"
echo "  LLM classifications: $LLM_OUTPUT_DIR/classifications_flat.csv"
echo "  Weaviate classifications: $WEAVIATE_OUTPUT_DIR/classifications_flat.csv"
echo "  Analysis report: weaviate_attribute_classification_analysis.json"

echo -e "\n${GREEN}Next steps:${NC}"
echo "  1. Review analysis report for accuracy metrics"
echo "  2. Check vector match rate"
echo "  3. Examine disagreements for quality issues"
