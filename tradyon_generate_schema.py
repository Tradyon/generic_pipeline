#!/usr/bin/env python3
"""
CLI tool to generate schemas using LLM.
"""

import argparse
import os
import sys
import logging
from src.schema.generator import SchemaGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    parser = argparse.ArgumentParser(description="Generate schemas using LLM.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Subcommand: product-definition
    parser_prod = subparsers.add_parser("product-definition", help="Generate product definition from raw shipments.")
    parser_prod.add_argument("--hs-code", required=True, help="HS-4 code (e.g., 0904).")
    parser_prod.add_argument("--input", required=True, help="Path to shipment master CSV.")
    parser_prod.add_argument("--output-dir", required=True, help="Output directory for schema.")
    parser_prod.add_argument("--model", default="gemini-2.0-flash", help="LLM model name.")
    parser_prod.add_argument("--max-products", type=int, help="Maximum number of product categories to infer (overrides auto ratio).")
    parser_prod.add_argument(
        "--product-refinement-rounds",
        type=int,
        default=3,
        help="Number of iterative refinement rounds when inferring product categories.",
    )
    
    # Subcommand: attribute-schema
    parser_attr = subparsers.add_parser("attribute-schema", help="Generate attribute schema from classified shipments.")
    parser_attr.add_argument("--hs-code", required=True, help="HS-4 code.")
    parser_attr.add_argument("--input", required=True, help="Path to classified shipment CSV.")
    parser_attr.add_argument("--products-definition", required=True, help="Path to products definition JSON.")
    parser_attr.add_argument("--output-dir", required=True, help="Output directory for schema.")
    parser_attr.add_argument("--model", default="gemini-2.0-flash", help="LLM model name.")
    parser_attr.add_argument("--use-response-schema", action="store_true", help="Enforce JSON response schema (stricter, reduces parsing errors).")
    parser_attr.add_argument("--attribute-goods-per-call", type=int, help="Max goods passed per LLM call when inferring attributes (default 250).")
    parser_attr.add_argument("--max-total-samples", type=int, help="Cap total goods samples per category (default 1000).")
    parser_attr.add_argument("--attribute-max-workers", type=int, help="Max parallel workers for attribute inference (default 8).")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.command == "product-definition":
            gen_config = {
                "model_name": args.model,
                "product_refinement_rounds": args.product_refinement_rounds,
            }
            if args.max_products and args.max_products > 0:
                gen_config["max_categories"] = args.max_products
            generator = SchemaGenerator(generation_config=gen_config)
            result = generator.generate_product_definition(
                hs_code=args.hs_code,
                shipment_csv=args.input,
                output_dir=args.output_dir
            )
            print(f"Product definition generated at: {result.products_definition_path}")
            
        elif args.command == "attribute-schema":
            gen_config = {"model_name": args.model}
            if args.use_response_schema:
                gen_config["use_response_schema"] = True
            if args.attribute_goods_per_call and args.attribute_goods_per_call > 0:
                gen_config["attribute_goods_per_call"] = args.attribute_goods_per_call
            if args.max_total_samples and args.max_total_samples > 0:
                gen_config["max_total_samples"] = args.max_total_samples
            if args.attribute_max_workers and args.attribute_max_workers > 0:
                gen_config["attribute_max_workers"] = args.attribute_max_workers

            generator = SchemaGenerator(generation_config=gen_config)
            result = generator.generate_attribute_configs_from_classifications(
                hs_code=args.hs_code,
                classified_csv=args.input,
                product_definition_path=args.products_definition,
                output_dir=args.output_dir,
                overwrite=True
            )
            print(f"Attribute schema generated at: {result.product_attributes_schema_path}")
            print(f"Attribute definitions generated at: {result.attribute_definitions_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
