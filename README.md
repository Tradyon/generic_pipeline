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
  --output <shipment_master.csv> \# required
  [--low-memory]                 # optional: pandas low_memory
```

### 2) Generate product definition

```bash
uv run python tradyon_generate_schema.py product-definition \
  --hs-code <HS4> \              # required
  --input <shipment_master.csv> \# required
  --output-dir <config_dir> \    # required
  [--model <llm_model>] \        # optional, default gemini-2.0-flash
  [--max-products <int>]         # optional: cap categories
```

### 3) Classify products

```bash
uv run python tradyon_classify_products.py \
  --input <shipment_master.csv> \                 # required
  --output <shipment_master_classified.csv> \     # required
  --products-definition <products_definition.json> \# required
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
