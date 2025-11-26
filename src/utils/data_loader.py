"""
Data Loader Module

Functions for loading shipment data into the pipeline from various sources.
"""

import pandas as pd
from typing import Optional


def load_data_from_bigquery(
    project_id: str,
    dataset_id: str,
    table_id: str,
    output_csv: str,
    query: Optional[str] = None,
    filters: Optional[dict] = None
) -> pd.DataFrame:
    """
    Load shipment data from Google BigQuery.
    
    This function queries BigQuery and prepares the data for classification.
    
    Parameters:
    -----------
    project_id : str
        GCP project ID
    dataset_id : str
        BigQuery dataset ID
    table_id : str
        BigQuery table ID
    output_csv : str
        Path where the loaded data should be written as CSV
    query : str, optional
        Custom SQL query (if None, queries entire table)
    filters : dict, optional
        Dictionary of filters to apply (e.g., {'hs_code': '9011'})
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with required columns, ready for product classification
    
    Required Columns in BigQuery Table:
    ------------------------------------
    - hs_code: Harmonized System code (string)
    - goods_shipped: Text description of goods
    - shipment_id: Unique identifier for each shipment
    
    Optional but recommended columns:
    - date, shipment_origin, shipment_destination
    - port_of_lading, port_of_unlading
    - value_of_goods_usd, weight_in_kg
    
    Example Implementation:
    -----------------------
    ```python
    from google.cloud import bigquery
    
    client = bigquery.Client(project=project_id)
    
    if query:
        df = client.query(query).to_dataframe()
    else:
        query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"
        if filters:
            where_clauses = [f"{k} = '{v}'" for k, v in filters.items()]
            query += " WHERE " + " AND ".join(where_clauses)
        df = client.query(query).to_dataframe()
    
    # Clean and standardize
    df['hs_code'] = df['hs_code'].astype(str).str[:4]
    df['goods_shipped'] = df['goods_shipped'].fillna('').str.strip()
    df['shipment_id'] = df['shipment_id'].astype(str)
    
    # Save and return
    df.to_csv(output_csv, index=False)
    return df
    ```
    
    Setup:
    ------
    1. Install: `pip install google-cloud-bigquery`
    2. Authenticate: `gcloud auth application-default login`
    3. Set project: `gcloud config set project YOUR_PROJECT_ID`
    
    Raises:
    -------
    NotImplementedError
        This template function must be implemented by the user
    """
    raise NotImplementedError(
        "load_data_from_bigquery() is a template function.\n"
        "Please implement BigQuery loading logic.\n\n"
        "Example:\n"
        "--------\n"
        "from google.cloud import bigquery\n"
        "client = bigquery.Client(project=project_id)\n"
        "query = f'SELECT * FROM `{project_id}.{dataset_id}.{table_id}` LIMIT 1000'\n"
        "df = client.query(query).to_dataframe()\n"
        "df['hs_code'] = df['hs_code'].astype(str)\n"
        "df.to_csv(output_csv, index=False)\n"
        "return df\n"
    )


def load_data(input_source: str, output_csv: str, low_memory: bool = False) -> pd.DataFrame:
    """
    Load shipment data from a CSV file.
    
    This is the default implementation that loads data from CSV files.
    For BigQuery or other sources, use load_data_from_bigquery() or implement
    your own loader.
    
    Parameters:
    -----------
    input_source : str
        Path to CSV file
    output_csv : str
        Path where the loaded data should be written
    low_memory : bool, optional
        Pandas read_csv low_memory flag. Default is False (safer for mixed types).
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with required columns, ready for product classification
    
    Required Output Columns:
    ------------------------
    - hs_code: Harmonized System code (string, e.g., "090111" or "9011")
    - goods_shipped: Text description of goods (primary field for classification)
    - shipment_id: Unique identifier for each shipment
    
    Optional but recommended columns:
    - date: Shipment date (YYYY-MM-DD format)
    - shipment_destination: Destination country/location
    - shipment_origin: Origin country/location
    - port_of_lading: Port name
    - port_of_lading_country: Country of port
    - port_of_unlading: Destination port name
    - port_of_unlading_country: Destination port country
    - trade_direction: Import/Export
    - transport_method: Maritime/Air/Truck/Rail
    - is_containerized: Boolean or Yes/No
    - value_of_goods_usd: Numeric value in USD
    - weight_in_kg: Weight in kilograms
    
    Additional columns are preserved but not required.
    """
    # Load CSV
    df = pd.read_csv(input_source, low_memory=low_memory)
    
    # Verify required columns
    required_cols = ['hs_code', 'goods_shipped', 'shipment_id']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    
    # Clean and standardize
    df['hs_code'] = df['hs_code'].astype(str).str.strip()
    df['goods_shipped'] = df['goods_shipped'].fillna('').astype(str).str.strip()
    df['shipment_id'] = df['shipment_id'].astype(str).str.strip()
    
    # Remove rows with empty goods_shipped
    initial_count = len(df)
    df = df[df['goods_shipped'] != '']
    removed = initial_count - len(df)
    if removed > 0:
        print(f"Removed {removed} rows with empty goods_shipped")
    
    # Save processed data
    df.to_csv(output_csv, index=False)
    print(f"Loaded {len(df)} records from {input_source}")
    print(f"Saved to {output_csv}")
    
    return df


def validate_shipment_data(df: pd.DataFrame) -> bool:
    """
    Validate that the loaded DataFrame has required columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Loaded shipment data
    
    Returns:
    --------
    bool
        True if validation passes
    
    Raises:
    -------
    ValueError
        If required columns are missing or data is invalid
    """
    required_columns = ['hs_code', 'goods_shipped', 'shipment_id']
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check for empty critical columns
    if df['goods_shipped'].isna().all() or (df['goods_shipped'] == '').all():
        raise ValueError("'goods_shipped' column is entirely empty")
    
    if df['shipment_id'].isna().all() or (df['shipment_id'] == '').all():
        raise ValueError("'shipment_id' column is entirely empty")
    
    # Check for valid HS codes
    if df['hs_code'].isna().all() or (df['hs_code'] == '').all():
        raise ValueError("'hs_code' column is entirely empty")
    
    print(f"✓ Validation passed: {len(df)} records")
    print(f"  - Unique HS codes: {df['hs_code'].nunique()}")
    print(f"  - Unique goods descriptions: {df['goods_shipped'].nunique()}")
    print(f"  - Unique shipments: {df['shipment_id'].nunique()}")
    
    return True
