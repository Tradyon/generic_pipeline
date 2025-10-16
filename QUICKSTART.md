# Quick Start Guide


## Prerequisites

1. Python 3.8 or higher
2. Google Gemini API key

## Step 1: Install Dependencies

```bash
cd generic_pipeline
pip install -r requirements.txt
```

## Step 2: Set API Key

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
GOOGLE_API_KEY=your-api-key-here
```

## Step 3: Implement Data Loader

Edit `data_loader.py` and replace the `load_data()` function:

```python
def load_data(input_source: str, output_csv: str) -> pd.DataFrame:
    # Example: Load from CSV
    df = pd.read_csv(input_source)
    
    # Ensure required columns
    assert 'hs_code' in df.columns
    assert 'goods_shipped' in df.columns
    assert 'shipment_id' in df.columns
    
    # Clean HS codes
    df['hs_code'] = df['hs_code'].astype(str).str[:4]
    
    # Save and return
    df.to_csv(output_csv, index=False)
    return df
```

## Step 4: Prepare Configuration Files

Copy and customize the example configs in `config/`:

```bash
cp config/products_definition_example.json config/my_products.json
cp config/product_attributes_schema_example.json config/my_attributes.json
cp config/attribute_definitions_example.json config/my_definitions.json
```

Edit these files to match your product domain.

## Step 5: Run the Pipeline

```bash
python pipeline_runner.py \
  --data-source /path/to/your/data.csv \
  --products-definition config/my_products.json \
  --product-attributes-schema config/my_attributes.json \
  --attribute-definitions config/my_definitions.json \
  --output-dir ./output
```

## Step 6: Check Results

Results will be in the `output/` directory:

```
output/
├── shipment_master_classified.csv       # Classified products
├── classifications_flat.csv             # All attributes
├── per_product_classifications/         # Per-product CSVs
└── classifications.json                 # Full JSON output
```

## Testing with Example Data

To test with the coffee example:

```bash
# Assuming you have coffee_pipeline data
python pipeline_runner.py \
  --data-source ../coffee_pipeline/coffee_shipment_master.csv \
  --products-definition config/products_definition_example.json \
  --product-attributes-schema config/product_attributes_schema_example.json \
  --attribute-definitions config/attribute_definitions_example.json \
  --output-dir ./output/test \
  --batch-size 5 \
  --items-per-call 5
```

## Common Issues

### "Data loader not implemented"
→ You need to implement the `load_data()` function in `data_loader.py`

### "API key missing"
→ Set `GOOGLE_API_KEY` environment variable

### "Missing required columns"
→ Ensure your data has `hs_code`, `goods_shipped`, and `shipment_id` columns

### "Schema mismatch"
→ Check that product names match between `products_definition.json` and `product_attributes_schema.json`

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Customize configuration files for your domain
3. Adjust classification parameters (batch size, token budget, etc.)
4. Monitor `pipeline.log` for detailed execution logs

## Getting Help

Check the logs:
```bash
tail -f pipeline.log
```

Verify configuration:
```bash
python -c "import json; print(json.dumps(json.load(open('config/my_products.json')), indent=2))"
```

Test imports:
```bash
python -c "import data_loader, products_classifier, product_attribute_classifier; print('OK')"
```

