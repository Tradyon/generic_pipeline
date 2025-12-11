#!/bin/bash
# End-to-end testing script for Pinecone Attribute Classification
# 
# This script runs the complete workflow:
# 1. Generate LLM baseline classifications
# 2. Populate Pinecone knowledge base
# 3. Run Pinecone hybrid classification
# 4. Analyze performance comparison
#
# Prerequisites:
#   export PINECONE_API_KEY="your-api-key"
#   export GEMINI_API_KEY="your-api-key"

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Pinecone Attribute Classification Test${NC}"
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
echo "PINECONE_API_KEY: ${PINECONE_API_KEY:0:10}..." # Show first 10 chars
echo "GEMINI_API_KEY: ${GEMINI_API_KEY:0:10}..."

if [ -z "$PINECONE_API_KEY" ]; then
    echo -e "${YELLOW}Error: PINECONE_API_KEY not set${NC}"
    echo ""
    echo "Add to .env file:"
    echo '  PINECONE_API_KEY = "your-pinecone-api-key"'
    echo ""
    echo "Or export manually:"
    echo "  export PINECONE_API_KEY='your-api-key'"
    exit 1
fi

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
INPUT_CSV="output/coffee/shipment_master_classified_pinecone.csv"
LLM_OUTPUT_DIR="output/coffee_attributes_llm"
PINECONE_OUTPUT_DIR="output/coffee_attributes_pinecone"
SCHEMA_FILE="config/product_attributes_schema_example.json"
DEFINITIONS_FILE="config/attribute_definitions_example.json"
INDEX_NAME="coffee-attrs-clean"
EMBEDDING_MODEL="multilingual-e5-large"

echo -e "\n${GREEN}Step 1: Generate LLM Baseline Classifications${NC}"
echo "Input: $INPUT_CSV"
echo "Output: $LLM_OUTPUT_DIR"

if [ -f "$LLM_OUTPUT_DIR/classifications_flat.csv" ]; then
    echo -e "${YELLOW}LLM classifications already exist, skipping...${NC}"
else
    python tradyon_classify_attributes.py \
        --input "$INPUT_CSV" \
        --output-dir "$LLM_OUTPUT_DIR" \
        --product-attributes-schema "$SCHEMA_FILE" \
        --attribute-definitions "$DEFINITIONS_FILE" \
        --items-per-call 5 \
        --max-workers 4
fi

# Extract coffee-only records
echo -e "\n${GREEN}Extracting coffee-only records...${NC}"
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

echo -e "\n${GREEN}Step 2: Populate Pinecone Knowledge Base${NC}"
echo "Index: $INDEX_NAME"
echo "Embedding model: $EMBEDDING_MODEL"

python scripts/populate_pinecone_attributes.py \
    --input "$COFFEE_CSV" \
    --index-name "$INDEX_NAME" \
    --embedding-model "$EMBEDDING_MODEL" \
    --batch-size 50

echo -e "\n${GREEN}Step 3: Run Pinecone Hybrid Classification${NC}"
echo "Input: $INPUT_CSV"
echo "Output: $PINECONE_OUTPUT_DIR"

python tradyon_classify_attributes_pinecone.py \
    --input "$INPUT_CSV" \
    --output-dir "$PINECONE_OUTPUT_DIR" \
    --product-attributes-schema "$SCHEMA_FILE" \
    --attribute-definitions "$DEFINITIONS_FILE" \
    --pinecone-index "$INDEX_NAME" \
    --embedding-model "$EMBEDDING_MODEL" \
    --similarity-threshold 0.75 \
    --items-per-call 5 \
    --max-workers 4

echo -e "\n${GREEN}Step 4: Analyze Performance${NC}"

python scripts/analyze_attribute_performance.py \
    --pinecone "$PINECONE_OUTPUT_DIR/classifications_flat.csv" \
    --llm "$LLM_OUTPUT_DIR/classifications_flat.csv" \
    --output "attribute_classification_analysis.json"

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Testing Complete!${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "\nResults:"
echo "  LLM classifications: $LLM_OUTPUT_DIR/classifications_flat.csv"
echo "  Pinecone classifications: $PINECONE_OUTPUT_DIR/classifications_flat.csv"
echo "  Analysis report: attribute_classification_analysis.json"

echo -e "\n${GREEN}Next steps:${NC}"
echo "  1. Review analysis report for accuracy metrics"
echo "  2. Check vector match rate (target: >85%)"
echo "  3. Examine disagreements for quality issues"
echo "  4. Tune similarity threshold if needed"
echo "  5. Test with multilingual data"
