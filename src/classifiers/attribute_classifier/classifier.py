"""
Main attribute classification logic.
"""

import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd
from tqdm import tqdm

from src.utils.config_models import (
    AttributeDefinitions,
    AttributeSchema,
    load_attribute_definitions as load_attribute_definitions_model,
    load_attribute_schema,
)
from src.utils.llm_client import LLMClient
from .utils import TokenTracker, sanitize_filename
from .llm import classify_batch
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def load_product_attribute_schema(path: str) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """Load attribute/value schema via shared config models."""
    schema_model: AttributeSchema = load_attribute_schema(path)
    normalized: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for hs4, entry in schema_model.entries.items():
        products: Dict[str, Dict[str, List[str]]] = {}
        for product_name, attr_set in entry.products.items():
            products[product_name] = attr_set.normalized_values()
        if products:
            normalized[hs4] = products
    return normalized


def load_attribute_definitions(path: str) -> Dict[str, str]:
    """Load attribute definitions for prompting via shared config models."""
    try:
        definitions_model: AttributeDefinitions = load_attribute_definitions_model(path)
    except FileNotFoundError:
        logger.warning(f"Attribute definitions file not found: {path}")
        return {}
    except Exception as exc:
        logger.warning(f"Failed to load attribute definitions: {exc}")
        return {}

    logger.info(f"Loaded {len(definitions_model.definitions)} attribute definitions")
    return definitions_model.definitions


def load_flat_products(csv_path: str, low_memory: bool = False) -> pd.DataFrame:
    """Load and prepare classified shipment data."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path, low_memory=low_memory, dtype={'hs_code': str})
    required = ['hs_code', 'goods_shipped', 'category']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Input CSV missing columns: {missing}")
    
    df = df.copy()
    df['hs_code'] = df['hs_code'].astype(str).str.strip().str[:4].str.zfill(4)
    df['goods_shipped'] = df['goods_shipped'].astype(str).str.strip()
    df['category'] = df['category'].astype(str).str.strip()
    
    if 'shipment_id' in df.columns:
        df['shipment_id'] = df['shipment_id'].astype(str).str.strip()
    else:
        df['shipment_id'] = [str(i) for i in range(1, len(df) + 1)]
    
    df = df.dropna(subset=['goods_shipped'])
    return df[['hs_code', 'category', 'goods_shipped', 'shipment_id']]


def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    """Load classification checkpoint."""
    if not os.path.exists(checkpoint_file):
        return {
            'products': {},
            'token_usage': 0,
            'timestamp': datetime.now().isoformat(),
            'version': 2
        }
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'products' not in data:
            data['products'] = {}
        return data
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return {
            'products': {},
            'token_usage': 0,
            'timestamp': datetime.now().isoformat(),
            'version': 2
        }


def save_checkpoint(checkpoint_file: str, state: Dict[str, Any]):
    """Save classification checkpoint."""
    try:
        state['timestamp'] = datetime.now().isoformat()
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        logger.info('Checkpoint saved.')
    except Exception as e:
        logger.error(f'Failed to save checkpoint: {e}')


def classify_attributes(
    input_csv: str,
    product_attributes_schema_path: str,
    attribute_definitions_path: str,
    output_json: str,
    output_csv: str,
    output_folder: str,
    checkpoint_file: Optional[str] = None,
    resume: bool = True,
    low_memory: bool = False,
    config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Classify product attributes from goods descriptions.
    """
    logger.info('Starting attribute classification...')
    
    # Default config
    default_config = {
        'hs_filter': None,
        'sample_goods_per_product': None,
        'items_per_call': 10,
        'max_workers': 20,
        'product_batch_workers': 4,
        'max_token_budget': 200_000_000,
        'model_name': 'gemini-2.0-flash',
        'retry_delay': 4,
        'max_retries': 3,
        'temperature': 0.0,
        'use_structured_output': True,
        'save_overall_csv': True,
        'validation_mode': 'coerce',
        'log_invalid_values': True,
        'record_raw_values': True,
        'enable_heuristic_fill': True,
        'max_hints_preview': 25,
        'deterministic_first_pass': True,
        'deterministic_min_token_chars': 3,
        'enable_negation_guard': True,
        'deterministic_negation_window_chars': 28,
        'dry_run_deterministic_only': False,
        'debug_deterministic_examples': 0,
        'allow_out_of_schema_values': False,
        'add_custom_flag_columns': False,
    }
    
    if config:
        default_config.update(config)
    config = default_config
    
    # Configure API
    model = LLMClient(model_name=config['model_name'], api_key=api_key)
    
    # Load schemas
    schema_model = load_attribute_schema(product_attributes_schema_path)
    schema_map = load_product_attribute_schema(product_attributes_schema_path)
    hs_available = set(schema_model.entries.keys())
    attr_definitions = load_attribute_definitions(attribute_definitions_path)
    
    # Load data
    df = load_flat_products(input_csv, low_memory=low_memory)
    if config['hs_filter']:
        allowed = {str(h)[:4] for h in config['hs_filter']}
        df = df[df['hs_code'].isin(allowed)]
    df = df[df['hs_code'].isin(hs_available)]
    
    # Group goods by hs4+product
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (hs4, product), sub in df.groupby(['hs_code', 'category']):
        sub2 = sub[['goods_shipped', 'shipment_id']].dropna(subset=['goods_shipped'])
        goods_to_shipments: Dict[str, List[str]] = {}
        for r in sub2.to_dict(orient='records'):
            g = r['goods_shipped']
            sid = r['shipment_id']
            if not isinstance(g, str) or not g.strip():
                continue
            if sid is None or (isinstance(sid, float) and pd.isna(sid)) or not isinstance(sid, str) or not str(sid).strip():
                sid = ''
            goods_to_shipments.setdefault(g, []).append(str(sid))
        
        unique_goods = list(goods_to_shipments.keys())
        if config['sample_goods_per_product'] and isinstance(config['sample_goods_per_product'], int):
            unique_goods = unique_goods[:config['sample_goods_per_product']]
        
        if unique_goods:
            grouped.setdefault(hs4, {})[product] = {
                'goods': sorted(unique_goods),
                'shipments': goods_to_shipments
            }
    
    checkpoint_file = checkpoint_file or 'attribute_classification_checkpoint.json'
    if resume:
        state = load_checkpoint(checkpoint_file)
    else:
        state = {
            'products': {},
            'token_usage': 0,
            'timestamp': datetime.now().isoformat(),
            'version': 2
        }
    tracker = TokenTracker(config['max_token_budget'])
    tracker.used = int(state.get('token_usage', 0))
    
    items_per_call = max(1, int(config['items_per_call']))
    os.makedirs(output_folder, exist_ok=True)

    hs_list = sorted(grouped.keys())
    max_workers = max(1, int(config.get('max_workers', 4)))
    for hs4 in hs_list:
        products_items = list(grouped[hs4].items())
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for product, pdata in products_items:
                key = f"{hs4}::{product}"
                existing = state.get('products', {}).get(key, {})
                done_count = len(existing.get('classifications', []))
                total_goods = len(pdata.get('goods', []))
                if done_count:
                    logger.info("Resuming product %s (%s): %s/%s goods already classified", product, hs4, done_count, total_goods)
                else:
                    logger.info("Processing new product %s (%s): %s goods", product, hs4, total_goods)

                fut = executor.submit(
                    _process_product,
                    hs4,
                    product,
                    pdata,
                    schema_map,
                    attr_definitions,
                    items_per_call,
                    tracker,
                    state,
                    checkpoint_file,
                    output_folder,
                    config,
                    model,
                )
                futures[fut] = product
            for future in tqdm(as_completed(futures), total=len(futures), desc=f'HS {hs4}', leave=False):
                future.result()
            if tracker.remaining() <= 0:
                logger.warning('Token budget exhausted; stopping.')
                break
            
            attr_values = schema_map.get(hs4, {}).get(product)
            if not attr_values:
                continue
            
            goods_list = pdata['goods']
            shipments_map = pdata['shipments']
            key = f"{hs4}::{product}"
            prod_state = state['products'].setdefault(key, {'completed': 0, 'classifications': []})
            
            # Merge shipment ids for already-classified goods
            existing_index = {c['goods_shipped']: c for c in prod_state['classifications']}
            for g, sids in shipments_map.items():
                if g in existing_index:
                    existing = existing_index[g].setdefault('shipment_ids', [])
                    for sid in sids:
                        if sid not in existing:
                            existing.append(sid)
            
            done_goods = {c['goods_shipped'] for c in prod_state['classifications']}
            remaining_goods = [g for g in goods_list if g not in done_goods]
            total_batches = (len(remaining_goods) + items_per_call - 1) // items_per_call if remaining_goods else 0
            
            for b_start in range(0, len(remaining_goods), items_per_call):
                if tracker.remaining() <= 0:
                    logger.warning('Token budget exhausted mid-product.')
                    break
                
                batch_goods = remaining_goods[b_start:b_start+items_per_call]
                batch_num = (b_start // items_per_call) + 1
                results = classify_batch(hs4, product, attr_values, batch_goods, model, tracker, batch_num, total_batches, attr_definitions, config)
                
                if results and '_validation_meta' in results[0]:
                    meta = results[0].pop('_validation_meta')
                    state.setdefault('validation_stats', {'invalid': 0, 'total': 0})
                    state['validation_stats']['invalid'] += meta.get('invalid', 0)
                
                for res in results:
                    g = res['goods_shipped']
                    res['shipment_ids'] = shipments_map.get(g, [])
                    prod_state['classifications'].append(res)
                
                prod_state['completed'] = len(prod_state['classifications'])
                state['token_usage'] = tracker.usage()
                save_checkpoint(checkpoint_file, state)
            
            # Save per-product CSV
            safe_prod = sanitize_filename(product)
            csv_name = f"hs_{hs4}_{safe_prod}.csv"
            csv_path = os.path.join(output_folder, csv_name)
            
            rows = []
            for c in prod_state['classifications']:
                shipments = [s for s in (c.get('shipment_ids') or []) if isinstance(s, str) and s.strip()]
                shipments = shipments or ['']
                all_ids = "|".join(shipments)

                for sid in shipments:
                    base = {
                        'hs_code': hs4,
                        'product': product,
                        'goods_shipped': c['goods_shipped'],
                        'shipment_id': sid,
                        'shipment_ids': all_ids,
                    }
                    for k, v in c['attributes'].items():
                        base[f"attr_{k}"] = v
                    rows.append(base)
            
            if rows:
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                logger.info(f"Saved per-product CSV {csv_name} ({len(rows)} rows)")

    # Save aggregated outputs
    if config['save_overall_csv']:
        all_rows = []
        all_json = {}
        
        for key, p_state in state['products'].items():
            if '::' in key:
                hs4, product = key.split('::', 1)
            else:
                hs4, product = "0000", key
            
            for c in p_state['classifications']:
                # JSON structure
                all_json.setdefault(hs4, {}).setdefault(product, []).append(c)
                
                # Flat CSV structure
                base = {
                    'hs_code': hs4,
                    'product': product,
                    'goods_shipped': c['goods_shipped'],
                    'shipment_ids': "|".join(c.get('shipment_ids', []))
                }
                for k, v in c['attributes'].items():
                    base[f"attr_{k}"] = v
                all_rows.append(base)
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(all_json, f, indent=2)
        logger.info(f"Saved aggregated JSON -> {output_json}")
        
        if all_rows:
            pd.DataFrame(all_rows).to_csv(output_csv, index=False)
            logger.info(f"Saved combined CSV -> {output_csv} (rows={len(all_rows)})")
    
    return {
        'metadata': {
            'products_processed': len(state['products']),
            'token_usage': tracker.usage(),
            'validation_stats': state.get('validation_stats', {})
        },
        'classifications': state['products']
    }
def _process_product(
    hs4: str,
    product: str,
    pdata: Dict[str, Any],
    schema_map: Dict[str, Dict[str, Dict[str, List[str]]]],
    attr_definitions: Dict[str, str],
    items_per_call: int,
    tracker: TokenTracker,
    state: Dict[str, Any],
    checkpoint_file: str,
    output_folder: str,
    config: Dict[str, Any],
    model: Any,
):
    """Process a single product (per HS4) for attribute classification."""
    attr_values = schema_map.get(hs4, {}).get(product)
    if not attr_values:
        return

    goods_list = pdata['goods']
    shipments_map = pdata['shipments']
    key = f"{hs4}::{product}"
    prod_state = state['products'].setdefault(key, {'completed': 0, 'classifications': []})

    # Merge shipment ids for already-classified goods
    existing_index = {c['goods_shipped']: c for c in prod_state['classifications']}
    for g, sids in shipments_map.items():
        if g in existing_index:
            existing = existing_index[g].setdefault('shipment_ids', [])
            for sid in sids:
                if sid not in existing:
                    existing.append(sid)

    done_goods = {c['goods_shipped'] for c in prod_state['classifications']}
    remaining_goods = [g for g in goods_list if g not in done_goods]
    total_batches = (len(remaining_goods) + items_per_call - 1) // items_per_call if remaining_goods else 0

    batches = []
    for b_start in range(0, len(remaining_goods), items_per_call):
        batch_goods = remaining_goods[b_start:b_start+items_per_call]
        batch_num = (b_start // items_per_call) + 1
        batches.append((batch_goods, batch_num))

    batch_workers = max(1, int(config.get('product_batch_workers', 1)))
    with ThreadPoolExecutor(max_workers=batch_workers) as executor:
        future_to_batch = {
            executor.submit(
                classify_batch,
                hs4,
                product,
                attr_values,
                batch_goods,
                model,
                tracker,
                batch_num,
                total_batches,
                attr_definitions,
                config,
            ): (batch_goods, batch_num)
            for batch_goods, batch_num in batches
        }
        for future in as_completed(future_to_batch):
            if tracker.remaining() <= 0:
                logger.warning('Token budget exhausted mid-product.')
                break
            try:
                results = future.result()
            except Exception as exc:
                logger.warning("Batch failed for product %s: %s", product, exc)
                results = []

            if results and '_validation_meta' in results[0]:
                meta = results[0].pop('_validation_meta')
                state.setdefault('validation_stats', {'invalid': 0, 'total': 0})
                state['validation_stats']['invalid'] += meta.get('invalid', 0)

            for res in results:
                g = res['goods_shipped']
                res['shipment_ids'] = shipments_map.get(g, [])
                prod_state['classifications'].append(res)

            prod_state['completed'] = len(prod_state['classifications'])
            state['token_usage'] = tracker.usage()
            save_checkpoint(checkpoint_file, state)

    # Save per-product CSV
    safe_prod = sanitize_filename(product)
    csv_name = f"hs_{hs4}_{safe_prod}.csv"
    csv_path = os.path.join(output_folder, csv_name)

    rows = []
    for c in prod_state['classifications']:
        shipments = [s for s in (c.get('shipment_ids') or []) if isinstance(s, str) and s.strip()]
        shipments = shipments or ['']
        all_ids = "|".join(shipments)

        for sid in shipments:
            base = {
                'hs_code': hs4,
                'product': product,
                'goods_shipped': c['goods_shipped'],
                'shipment_id': sid,
                'shipment_ids': all_ids,
            }
            for k, v in c['attributes'].items():
                base[f"attr_{k}"] = v
            rows.append(base)

    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info(f"Saved per-product CSV {csv_name} ({len(rows)} rows)")
