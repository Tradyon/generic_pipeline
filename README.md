# Tradyon Generic Pipeline

LLM workflow to classify shipments into products and extract attributes.

## Setup

- Python >= 3.9
- Install deps: `uv sync` (or `pip install -r requirements.txt`)
- Keys: set `OPENROUTER_API_KEY` (OpenRouter/OpenAI) or `GOOGLE_API_KEY` (Gemini). Optional `OPENROUTER_BASE_URL`.

## Commands (all flags shown)

### 1) Load & validate

```bash
uv run python tradyon_load.py \
  --input <raw.csv> \            # required
  --output <shipment_master.csv> \ # required
  [--low-memory]                  # optional: pandas low_memory
```

### 2) Generate product definition

```bash
uv run python tradyon_generate_schema.py product-definition \
  --hs-code <HS4> \              # required
  --input <shipment_master.csv> \ # required
  --output-dir <config_dir> \    # required
  [--model <llm_model>] \        # optional, default gemini-2.0-flash
  [--max-products <int>]         # optional: cap categories (recommended)
```

### 3) Classify products

```bash
uv run python tradyon_classify_products.py \
  --input <shipment_master.csv> \                 # required
  --output <shipment_master_classified.csv> \     # required
  --products-definition <products_definition.json> \ # required
  [--model <llm_model>] \                         # optional, default gemini-2.0-flash
  [--batch-size <int>] \                          # optional, default 10
  [--max-workers <int>] \                         # optional, default 5
  [--checkpoint-file <path>] \                    # optional
  [--no-resume] \                                 # optional: ignore checkpoint
  [--low-memory]                                  # optional
```

### 4) Generate attribute schema

```bash
uv run python tradyon_generate_schema.py attribute-schema \
  --hs-code <HS4> \                                # required
  --input <shipment_master_classified.csv> \       # required
  --products-definition <products_definition.json> \ # required
  --output-dir <config_dir> \                      # required
  [--model <llm_model>]                            # optional
```

### 5) Classify attributes

```bash
uv run python tradyon_classify_attributes.py \
  --input <shipment_master_classified.csv> \        # required
  --output-dir <output_dir> \                       # required
  --product-attributes-schema <product_attributes_schema.json> \ # required
  --attribute-definitions <attribute_definitions.json> \         # required
  [--model <llm_model>] \                           # optional, default gemini-2.0-flash
  [--items-per-call <int>] \                        # optional, default 10
  [--max-workers <int>] \                           # optional, default 20 (parallel products)
  [--token-budget <int>] \                          # optional, default 20000000
  [--checkpoint-file <path>] \                      # optional
  [--no-resume] \                                   # optional
  [--low-memory]                                    # optional
```

### 6) Post-process outputs

```bash
uv run python tradyon_post_process.py \
  --input-dir <per_product_classifications_dir> \ # required
  --output-dir <output_dir>                       # required
```

## Notes

- Product inference uses multiple rounds over up to 1,000 samples and honors `is_multi_product_shipment` to drop mixed rows.
- Override `--model` with OpenRouter IDs like `deepseek/deepseek-v3.2-exp`; the client enforces JSON where supported.

## Strict Attribute Enforcement

The pipeline is configured to strictly enforce attribute values from the schema. It will **not** invent new values or output "custom" values. If an attribute value is not found in the schema and cannot be inferred, it will be set to "None".

## Native Gemini Support (via OpenAI Compatibility)

For Gemini models (e.g., `gemini-2.0-flash`), the pipeline uses the OpenAI SDK pointing to Google's OpenAI compatibility endpoint. This allows using the standard OpenAI client while leveraging Gemini's features. Note that `propertyOrdering` is passed in the JSON schema to ensure correct field ordering in structured outputs.

## Examples

Coffee sample (Gemini flash 2.0):

```bash
uv run python tradyon_load.py --input ./sample_coffee_shipment_master.csv --output ./output/coffee/shipment_master.csv
uv run python tradyon_generate_schema.py product-definition --hs-code 0901 --input ./output/coffee/shipment_master.csv --output-dir ./config
uv run python tradyon_classify_products.py --input ./output/coffee/shipment_master.csv --output ./output/coffee/shipment_master_classified.csv --products-definition config/products_definition_example.json --batch-size 5 --max-workers 5
uv run python tradyon_generate_schema.py attribute-schema --hs-code 0901 --input ./output/coffee/shipment_master_classified.csv --products-definition config/products_definition_example.json --output-dir ./config
uv run python tradyon_classify_attributes.py --input ./output/coffee/shipment_master_classified.csv --output-dir ./output/coffee --product-attributes-schema config/product_attributes_schema_example.json --attribute-definitions config/attribute_definitions_example.json --items-per-call 5 --max-workers 10
uv run python tradyon_post_process.py --input-dir ./output/coffee/per_product_classifications --output-dir ./output/coffee
```

Coffee sample (Deepseek / any OpenRouter model):

```bash
uv run python tradyon_load.py --input ./sample_coffee_shipment_master.csv --output ./output/coffee/shipment_master.csv
uv run python tradyon_generate_schema.py product-definition --hs-code 0901 --input ./output/coffee/shipment_master.csv --output-dir ./config --model deepseek/deepseek-v3.2-exp
uv run python tradyon_classify_products.py --input ./output/coffee/shipment_master.csv --output ./output/coffee/shipment_master_classified.csv --products-definition config/products_definition_example.json --batch-size 5 --max-workers 5 --model deepseek/deepseek-v3.2-exp
uv run python tradyon_generate_schema.py attribute-schema --hs-code 0901 --input ./output/coffee/shipment_master_classified.csv --products-definition config/products_definition_example.json --output-dir ./config --model deepseek/deepseek-v3.2-exp
uv run python tradyon_classify_attributes.py --input ./output/coffee/shipment_master_classified.csv --output-dir ./output/coffee --product-attributes-schema config/product_attributes_schema_example.json --attribute-definitions config/attribute_definitions_example.json --items-per-call 5 --max-workers 10 --model deepseek/deepseek-v3.2-exp
uv run python tradyon_post_process.py --input-dir ./output/coffee/per_product_classifications --output-dir ./output/coffee
```
