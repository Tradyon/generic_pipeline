#!/bin/bash
# Helper script to run Pinecone attribute classification with API keys from .env file
#
# Usage:
#   1. Create a .env file with your API keys:
#      echo 'export PINECONE_API_KEY="your-key"' > .env
#      echo 'export GEMINI_API_KEY="your-key"' >> .env
#
#   2. Run this script:
#      ./run_pinecone_test.sh

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading environment from .env file..."
    source .env
else
    echo "No .env file found. Checking for environment variables..."
fi

# Check if variables are set
if [ -z "$PINECONE_API_KEY" ]; then
    echo "Error: PINECONE_API_KEY not set"
    echo ""
    echo "Create a .env file with:"
    echo "  export PINECONE_API_KEY=\"your-pinecone-key\""
    echo "  export GEMINI_API_KEY=\"your-gemini-key\""
    echo ""
    echo "Or set manually:"
    echo "  export PINECONE_API_KEY=\"your-key\""
    echo "  export GEMINI_API_KEY=\"your-key\""
    exit 1
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY not set"
    exit 1
fi

# Run the test script
exec ./test_pinecone_attributes.sh
