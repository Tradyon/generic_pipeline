# Generic Product Classification Pipeline


## Overview

This pipeline processes trade shipment data through multiple stages:

1. **Data Loading**: Load and prepare shipment data (We need to edit this for our pipeline)
2. **Product Classification**: Classify goods into product categories
3. **Attribute Classification**: Extract and classify product-specific attributes
4. **Output Generation**: Generate structured outputs and summaries


## Features

- **Product-agnostic**: Adapts to any product domain via configuration
- **LLM-powered**: Uses Google Gemini for intelligent classification
- **Deterministic matching**: Fast pattern matching before LLM classification
- **Checkpoint support**: Resume interrupted processing
- **Parallel processing**: Multi-threaded batch classification
- **Validation & auditing**: Track out-of-schema values and validation stats
- **Flexible outputs**: Per-product CSVs, aggregated JSON, flat CSV

## Installation

### Prerequisites

- Python 3.8+
- Google Gemini API key

### Dependencies

```bash
pip install pandas google-generativeai pydantic tqdm python-dotenv
```

### Environment Setup

Create a `.env` file or set environment variable:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

## Directory Structure

```
generic_pipeline/
├── config/                              # Example configuration files
│   ├── products_definition_example.json
│   ├── product_attributes_schema_example.json
│   └── attribute_definitions_example.json
├── utils/                               # Helper utilities
│   ├── build_product_schema_master.py
│   └── combined_shipment_id_to_attr_json.py
├── data_loader.py                       # Data loading template
├── products_classifier.py               # Product classification module
├── product_attribute_classifier.py      # Attribute classification module
├── pipeline_runner.py                   # Main orchestration script
└── README.md                            # This file
```

## Configuration Files

### 1. Products Definition (`products_definition.json`)

Defines product categories for classification.

**Format:**
```json
{
  "hs_code": "9011",
  "product_categories": [
    "Category A",
    "Category B",
    "Category C"
  ],
  "category_descriptions": {
    "Category A": "Description to guide LLM classification",
    "Category B": "Another description",
    "Category C": "Yet another description"
  }
}
```

**Fields:**
- `hs_code`: Harmonized System code (4-digit string)
- `product_categories`: List of category names
- `category_descriptions`: Optional descriptions for each category

### 2. Product Attributes Schema (`product_attributes_schema.json`)

Defines attributes and allowed values for each product category.

**Format:**
```json
{
  "hs_code": {
    "version": "2025-01-01T00:00:00Z",
    "products": {
      "Category A": {
        "attributes": {
          "Attribute_1": ["Value1", "Value2", "Value3"],
          "Attribute_2": ["ValueA", "ValueB", "ValueC"]
        }
      },
      "Category B": {
        "attributes": {
          "Attribute_1": ["Different", "Values"],
          "Attribute_3": ["More", "Options"]
        }
      }
    }
  }
}
```

**Structure:**
- Top level: HS code keys
- `version`: ISO 8601 timestamp for tracking
- `products`: Product categories (matching products_definition)
- `attributes`: Attribute types and their allowed values

### 3. Attribute Definitions (`attribute_definitions.json`)

Provides descriptions for attributes to guide LLM classification.

**Format:**
```json
{
  "attr_Attribute_1": "Description of what this attribute represents",
  "attr_Attribute_2": "Another attribute description",
  "attr_Attribute_3": "Yet another description"
}
```

**Note:** Keys must be prefixed with `attr_` followed by the attribute name.

## Usage

### Quick Start

1. **Implement Data Loader**

Edit `data_loader.py` to load your data:

```python
def load_data(input_source: str, output_csv: str) -> pd.DataFrame:
    # Your custom loading logic
    df = pd.read_csv(input_source)
    
    # Ensure required columns exist
    # Required: hs_code, goods_shipped, shipment_id
    
    df.to_csv(output_csv, index=False)
    return df
```

2. **Create Configuration Files**

Based on the examples in `config/`, create your own:
- `my_products_definition.json`
- `my_attributes_schema.json`
- `my_attribute_definitions.json`

3. **Run Pipeline**

```bash
python pipeline_runner.py \
  --data-source ./raw_data.csv \
  --products-definition ./config/my_products_definition.json \
  --product-attributes-schema ./config/my_attributes_schema.json \
  --attribute-definitions ./config/my_attribute_definitions.json \
  --output-dir ./output
```

### Command-Line Arguments

**Required:**
- `--data-source`: Input data path
- `--products-definition`: Products definition JSON path
- `--product-attributes-schema`: Attributes schema JSON path
- `--attribute-definitions`: Attribute definitions JSON path
- `--output-dir`: Output directory

**Optional:**
- `--skip-data-loading`: Skip data loading step
- `--skip-product-classification`: Skip product classification
- `--skip-attribute-classification`: Skip attribute classification
- `--no-helpers`: Skip helper utilities
- `--no-resume`: Don't resume from checkpoints
- `--model`: LLM model name (default: `gemini-2.0-flash`)
- `--batch-size`: Batch size (default: 10)
- `--max-workers`: Parallel workers (default: 5)
- `--token-budget`: Max tokens (default: 20000000)
- `--items-per-call`: Items per LLM call for attributes (default: 10)

### Example: Coffee Pipeline

```bash
python pipeline_runner.py \
  --data-source ../coffee_pipeline/coffee_shipment_master.csv \
  --products-definition ./config/products_definition_example.json \
  --product-attributes-schema ./config/product_attributes_schema_example.json \
  --attribute-definitions ./config/attribute_definitions_example.json \
  --output-dir ./output/coffee \
  --model gemini-2.0-flash \
  --batch-size 10
```

## Outputs

The pipeline generates the following outputs in the specified output directory:

```
output_dir/
├── shipment_master.csv                      # Loaded data
├── shipment_master_classified.csv           # With product categories
├── classifications.json                      # Aggregated classifications
├── classifications_flat.csv                  # Flat CSV with all attributes
├── product_schema_master.csv                # Schema summary
├── shipment_id_to_attr.json                 # Shipment ID mapping
├── per_product_classifications/             # Per-product CSVs
│   ├── hs_9011_Bulk_Commodity_Green_Coffee.csv
│   ├── hs_9011_Specialty_Green_Coffee.csv
│   └── ...
├── checkpoints/                             # Checkpoint files
│   ├── product_classification.pkl
│   └── attribute_classification.json
└── pipeline.log                             # Execution log
```

### Output Formats

**Per-Product CSV:**
```csv
hs_code,product,shipment_id,goods_shipped,attr_Variety,attr_Origin,attr_Grade,...
9011,Bulk Commodity,123,Coffee beans arabica,Arabica,Brazil,Grade 1,...
```

**Classifications JSON:**
```json
{
  "metadata": {
    "timestamp": "2025-01-01T00:00:00Z",
    "products_processed": 3,
    "token_usage": 150000
  },
  "products": {
    "9011::Product Name": {
      "completed": 100,
      "classifications": [...]
    }
  }
}
```

## Advanced Usage

### Resume from Checkpoint

The pipeline automatically saves checkpoints. To resume:

```bash
python pipeline_runner.py \
  --data-source ./data.csv \
  --products-definition ./config/products.json \
  --product-attributes-schema ./config/attributes.json \
  --attribute-definitions ./config/definitions.json \
  --output-dir ./output
  # Checkpoints in ./output/checkpoints/ are used automatically
```

### Skip Completed Steps

If you've already completed certain steps:

```bash
python pipeline_runner.py \
  --skip-data-loading \
  --skip-product-classification \
  ...
```

### Configuration Override

Adjust classification parameters:

```bash
python pipeline_runner.py \
  --model gemini-2.5-flash-lite \
  --batch-size 20 \
  --max-workers 10 \
  --token-budget 50000000 \
  ...
```

## Extending the Data Loader

The `data_loader.py` template must be implemented for your data source.

**Required columns in output:**
- `hs_code`: Harmonized System code (string)
- `goods_shipped`: Text description of goods
- `shipment_id`: Unique shipment identifier

**Optional but recommended:**
- `date`, `shipment_origin`, `shipment_destination`
- `port_of_lading`, `port_of_unlading`
- `value_of_goods_usd`, `weight_in_kg`
- Additional metadata columns

**Example implementations:**

**From CSV:**
```python
def load_data(input_source: str, output_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_source)
    df['hs_code'] = df['hs_code'].astype(str).str[:4]
    df.to_csv(output_csv, index=False)
    return df
```

**From Database:**
```python
def load_data(input_source: str, output_csv: str) -> pd.DataFrame:
    import sqlalchemy
    engine = sqlalchemy.create_engine(input_source)
    df = pd.read_sql("SELECT * FROM shipments", engine)
    df.to_csv(output_csv, index=False)
    return df
```

**From API:**
```python
def load_data(input_source: str, output_csv: str) -> pd.DataFrame:
    import requests
    response = requests.get(input_source)
    data = response.json()
    df = pd.DataFrame(data['records'])
    df.to_csv(output_csv, index=False)
    return df
```

## Classification Logic

### Product Classification

1. Loads products definition JSON
2. Creates dynamic Pydantic schema from categories
3. Batches goods descriptions
4. Classifies using LLM with structured output
5. Applies results to all duplicate goods descriptions
6. Saves checkpoints every 50 unique goods

### Attribute Classification

1. **Deterministic Matching**: Fast pattern-based extraction
   - Whole-token matching with boundary detection
   - Negation awareness ("not ground" → no match for "ground")
   - Longest-value precedence

2. **LLM Classification**: Structured attribute extraction
   - Batched processing with schema validation
   - Support for out-of-schema values
   - Merge deterministic and LLM results

3. **Heuristic Enrichment**: Fill missing attributes
   - Simple substring matching for obvious values
   - Applied to attributes still marked as 'None'

4. **Validation**: Track data quality
   - Invalid value detection and coercion
   - Out-of-schema value counting
   - Comprehensive statistics

## Troubleshooting

### Data Loader Not Implemented

**Error:** `NotImplementedError: data_loader.load_data() is a template function`

**Solution:** Implement the `load_data()` function in `data_loader.py`

### API Key Missing

**Error:** `ValueError: Google API key must be provided`

**Solution:** Set `GOOGLE_API_KEY` environment variable or pass `api_key` parameter

### Schema Mismatch

**Error:** `KeyError: product not found in schema`

**Solution:** Ensure all products in `products_definition.json` have entries in `product_attributes_schema.json`

### Token Budget Exhausted

**Warning:** `Token budget exhausted; stopping`

**Solution:** Increase `--token-budget` or process data in smaller batches

### Checkpoint Corruption

**Error:** `Failed to load checkpoint`

**Solution:** Delete checkpoint files and restart with `--no-resume`

## Performance Tips

1. **Optimize Batch Size**: Larger batches = fewer API calls but longer processing per batch
2. **Adjust Workers**: More workers = faster parallel processing (up to API rate limits)
3. **Use Deterministic First Pass**: Enabled by default, significantly reduces LLM calls
4. **Enable Checkpointing**: Resume long-running jobs without starting over
5. **Filter Data**: Process only relevant HS codes to save tokens

## API Costs

Approximate token usage (using Gemini):
- Product classification: ~100-200 tokens per goods description
- Attribute classification: ~500-1000 tokens per goods description (varies by attributes)

Monitor usage in pipeline output:
```
Token Usage: 1,500,000 (remaining 18,500,000)
```

