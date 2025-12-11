import pandas as pd
import sys
import os

def prepare_data(input_path, output_path):
    print(f"Preparing data from {input_path} to {output_path}")
    df = pd.read_csv(input_path)
    
    # Add category if missing
    if 'category' not in df.columns:
        print("Adding default category: Bulk Commodity Green Coffee")
        df['category'] = 'Bulk Commodity Green Coffee'
        
    # Ensure hs_code is string and padded
    if 'hs_code' in df.columns:
        df['hs_code'] = df['hs_code'].astype(str).str.strip().str[:4].str.zfill(4)
        
    df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prepare_test_data.py <input_csv> <output_csv>")
        sys.exit(1)
    
    prepare_data(sys.argv[1], sys.argv[2])
