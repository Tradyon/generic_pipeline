#!/usr/bin/env python3
"""
Cleanup Weaviate schema/class.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if line.startswith('export '):
                        line = line[7:]
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        os.environ[key] = value

load_env_file()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.weaviate_client import WeaviateClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Delete Weaviate class.")
    parser.add_argument("--class-name", required=True, help="Weaviate class name to delete.")
    parser.add_argument("--force", action="store_true", help="Force deletion without confirmation.")
    
    args = parser.parse_args()
    
    client = WeaviateClient(class_name=args.class_name)
    
    if client.client.collections.exists(args.class_name):
        if not args.force:
            confirm = input(f"Are you sure you want to delete class '{args.class_name}'? [y/N]: ")
            if confirm.lower() != 'y':
                logger.info("Aborted.")
                return

        logger.info(f"Deleting class '{args.class_name}'...")
        client.client.collections.delete(args.class_name)
        logger.info(f"Class '{args.class_name}' deleted.")
    else:
        logger.info(f"Class '{args.class_name}' does not exist.")

if __name__ == "__main__":
    main()
