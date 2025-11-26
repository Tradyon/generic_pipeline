"""
Build Product Schema Master

Build a schema summary from per-product classification CSV files.
"""

import os
import json
import re
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd


MISSING_STRINGS = {"", "none", "null", "nan", "na", "n/a", "undefined"}


def _clean_value(val: object) -> Optional[str]:
    """Clean a raw cell value, preserving original casing and spaces; return None if missing."""
    if pd.isna(val):
        return None
    s = str(val).strip().strip('"').strip("'")
    if not s:
        return None
    if s.strip().lower() in MISSING_STRINGS:
        return None
    
    # Reject values that are too long (likely hallucinations or raw data)
    if len(s) > 60:
        return None
        
    return s


def _to_snake_case(name: str) -> str:
    """Convert PascalCase/camelCase/Title_Case to snake_case while respecting existing underscores."""
    # Normalize dashes and spaces to underscores
    s = re.sub(r"[\-\s]+", "_", name)
    # Handle transitions: Acronyms followed by words (e.g., 'HTTPServer' -> 'HTTP_Server')
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    # Lower-to-Upper transitions (e.g., 'originCountry' -> 'origin_Country')
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    return s.lower().strip("_")


def _attr_to_snake(col: str, drop_prefix: bool = True) -> str:
    """Normalize attribute column to snake_case and optionally drop 'attr_' prefix."""
    if col.startswith("attr_"):
        suffix = col[len("attr_"):]
        snake_suffix = _to_snake_case(suffix)
        return snake_suffix if drop_prefix else f"attr_{snake_suffix}"
    # Fallback for non-attr prefixed columns
    return _to_snake_case(col)


def product_name_from_filename(filename: str) -> str:
    """Extract product name from CSV filename."""
    name, _ = os.path.splitext(os.path.basename(filename))
    parts = name.split("_")
    # Attempt to strip leading 'hs' and code like '9011'
    if len(parts) >= 3 and parts[0].lower() == "hs" and parts[1].isdigit():
        product_parts = parts[2:]
    else:
        # Fallback: everything after potential 'hs_<code>_'
        product_parts = parts[2:] if len(parts) > 2 else parts
    # Join with spaces and tidy casing
    product = " ".join(product_parts).replace("  ", " ").strip()
    # Replace remaining underscores if any
    product = product.replace("_", " ")
    return product


def build_schema_for_csv(csv_path: str) -> Dict[str, List[str]]:
    """
    Build a mapping of attribute_name -> list of possible values (strings),
    considering only columns prefixed with 'attr_'.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    schema: Dict[str, List[str]] = {}
    attr_cols = [c for c in df.columns if c.startswith("attr_")]
    for col in attr_cols:
        # Convert column name to snake_case and drop attr_ prefix in the output
        snake_case_col = _attr_to_snake(col, drop_prefix=True)
        # Clean values and collect uniques, preserving original form
        raw_series = df[col].map(_clean_value).dropna()
        seen = set()
        values: List[str] = []
        for v in raw_series:
            key = v.strip().lower()
            if key not in seen:
                seen.add(key)
                values.append(v)
        schema[snake_case_col] = values
    return schema


def build_schema_master(input_dir: str, output_csv: str, file_pattern: str = "*.csv") -> pd.DataFrame:
    """
    Build product schema master from per-product CSV files.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing product CSV files
    output_csv : str
        Path to write product_schema_master.csv
    file_pattern : str
        Glob pattern for CSV files (default: "*.csv")
    
    Returns:
    --------
    pd.DataFrame
        Schema master DataFrame
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    rows = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        fpath = os.path.join(input_dir, fname)
        
        # Skip writing the output file itself if scanning the same directory
        try:
            if os.path.abspath(fpath) == os.path.abspath(output_csv):
                continue
        except Exception:
            pass
        
        try:
            product = product_name_from_filename(fname)
            schema = build_schema_for_csv(fpath)
            rows.append({
                "product": product,
                "file": fname,
                "schema_json": json.dumps(schema, ensure_ascii=False)
            })
        except Exception as e:
            rows.append({
                "product": product_name_from_filename(fname),
                "file": fname,
                "schema_json": json.dumps({"__error__": [str(e)]}, ensure_ascii=False)
            })
    
    if not rows:
        raise RuntimeError(f"No CSV files found in {input_dir}")
    
    out_dir = os.path.dirname(output_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote schema for {len(rows)} products to {output_csv}")
    
    return out_df


def find_csv_files(input_dir: Path, pattern: str) -> List[Path]:
    """Return list of CSV files matching pattern in the input directory (non-recursive)."""
    return sorted(input_dir.glob(pattern))


def extract_attr_columns(header: List[str]) -> List[str]:
    """Identify columns that start with 'attr_' (case-sensitive)."""
    return [c for c in header if c.startswith("attr_")]


def to_snake_case_combined(key: str) -> str:
    """Convert arbitrary strings to lowercase snake_case suitable for JSON keys."""
    s = key.strip()
    # Insert underscore between camelCase and PascalCase boundaries
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    # Replace spaces, hyphens, slashes and other non-word chars with underscores
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    # Trim underscores and lowercase
    return s.strip("_").lower()


def load_attributes(files: List[Path], shipment_id_col: str = "shipment_id", product_col: str = "product") -> Dict[str, Dict[str, object]]:
    """
    Load and combine attributes from multiple CSV files.
    """
    combined: Dict[str, Dict[str, object]] = {}
    
    for path in files:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            if shipment_id_col not in reader.fieldnames:
                # Skip files that don't have the ID column
                continue
            
            attr_cols = extract_attr_columns(reader.fieldnames or [])
            for row in reader:
                shipment_id = (row.get(shipment_id_col) or "").strip()
                if not shipment_id:
                    continue
                
                entry = combined.setdefault(shipment_id, {"attrs": {}, "product": None})
                attrs: Dict[str, str] = entry["attrs"]  # type: ignore[assignment]
                
                if product_col in (reader.fieldnames or []):
                    prod_raw = row.get(product_col)
                    if prod_raw is not None:
                        prod = str(prod_raw).strip()
                        if prod:
                            entry["product"] = prod
                
                for col in attr_cols:
                    raw_val = row.get(col)
                    if raw_val is None:
                        continue
                    val = str(raw_val).strip()
                    if val == "" or val.lower() in {"nan", "none", "null"}:
                        continue
                    key = col[len("attr_"):]
                    snake_key = to_snake_case_combined(key)
                    if not snake_key:
                        continue
                    attrs[snake_key] = val
    
    return combined


def combine_shipment_attributes(input_dir: str, pattern: str, output_path: str, shipment_id_col: str = "shipment_id") -> Dict[str, Any]:
    """
    Combine shipment ID to attribute mappings from multiple CSV files.
    """
    input_dir_path = Path(input_dir).expanduser().resolve()
    output_path_obj = Path(output_path)
    if not output_path_obj.is_absolute():
        output_path_obj = input_dir_path / output_path
    
    files = find_csv_files(input_dir_path, pattern)
    if not files:
        print(f"Warning: No files found in '{input_dir_path}' matching pattern '{pattern}'.")
        return {}
    
    print(f"Found {len(files)} file(s): {[p.name for p in files]}")
    combined = load_attributes(files, shipment_id_col=shipment_id_col)
    
    # Write output based on extension
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path_obj.suffix.lower()
    
    if ext == ".json":
        # Single JSON object mapping shipment_id -> { attr: value }
        attrs_only = {sid: (entry.get("attrs") or {}) for sid, entry in combined.items()}
        with output_path_obj.open("w", encoding="utf-8") as out_f:
            json.dump(attrs_only, out_f, ensure_ascii=False, indent=2)
    else:
        # Three-column CSV: shipment_id, product, attrs_json
        with output_path_obj.open("w", encoding="utf-8", newline="") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(["shipment_id", "product", "attrs_json"])
            for shipment_id in sorted(combined.keys()):
                entry = combined[shipment_id]
                product = entry.get("product") or ""
                attrs_json = json.dumps(entry.get("attrs") or {}, ensure_ascii=False)
                writer.writerow([shipment_id, product, attrs_json])
    
    print(f"Wrote {len(combined)} shipment(s) with attributes to '{output_path_obj}'.")
    
    return combined


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build product schema master from product CSVs.")
    parser.add_argument("--input-dir", required=True, help="Directory containing product CSV files")
    parser.add_argument("--output-csv", required=True, help="Path to write product_schema_master.csv")
    parser.add_argument("--pattern", default="*.csv", help="File pattern (default: *.csv)")
    args = parser.parse_args()
    
    build_schema_master(args.input_dir, args.output_csv, args.pattern)

