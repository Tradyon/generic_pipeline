#!/usr/bin/env python3
"""
Analyze and compare Pinecone vs LLM attribute classification performance.

Metrics:
- Accuracy: Agreement rate between methods
- Speed: Time to classify
- Cost: Token usage and estimated cost
- Precision: Per-attribute match rates
- Multilingual quality: Performance on non-English text
"""

import argparse
import json
import logging
import sys
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def load_classifications(csv_path: str) -> pd.DataFrame:
    """Load attribute classifications from CSV."""
    df = pd.read_csv(csv_path, dtype={'hs_code': str})
    logger.info(f"Loaded {len(df)} records from {csv_path}")
    return df


def compare_classifications(
    pinecone_df: pd.DataFrame,
    llm_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compare Pinecone and LLM classifications.
    
    Returns:
        Dictionary with comparison metrics
    """
    # Merge on key columns
    merged = pinecone_df.merge(
        llm_df,
        on=['hs_code', 'product', 'goods_shipped'],
        how='inner',
        suffixes=('_pinecone', '_llm')
    )
    
    logger.info(f"Comparing {len(merged)} matching records")
    
    # Get attribute columns (those starting with 'attr_')
    attr_cols = [c for c in pinecone_df.columns if c.startswith('attr_')]
    
    results = {
        'total_records': len(merged),
        'total_attributes': len(attr_cols),
        'attribute_comparison': {},
        'overall_agreement': 0.0,
        'per_attribute_agreement': {},
        'disagreements': [],
    }
    
    total_comparisons = 0
    total_agreements = 0
    
    for attr in attr_cols:
        pinecone_col = f"{attr}_pinecone"
        llm_col = f"{attr}_llm"
        
        if pinecone_col not in merged.columns or llm_col not in merged.columns:
            continue
        
        # Compare values
        valid_comparisons = merged[[pinecone_col, llm_col]].notna().all(axis=1)
        agreements = merged[valid_comparisons][pinecone_col] == merged[valid_comparisons][llm_col]
        
        n_comparisons = valid_comparisons.sum()
        n_agreements = agreements.sum()
        
        if n_comparisons > 0:
            agreement_rate = n_agreements / n_comparisons
            
            results['per_attribute_agreement'][attr] = {
                'agreement_rate': agreement_rate,
                'total_comparisons': n_comparisons,
                'agreements': n_agreements,
                'disagreements': n_comparisons - n_agreements
            }
            
            total_comparisons += n_comparisons
            total_agreements += n_agreements
            
            # Collect sample disagreements
            disagreements = merged[valid_comparisons & ~agreements]
            if len(disagreements) > 0:
                for _, row in disagreements.head(5).iterrows():
                    results['disagreements'].append({
                        'goods_shipped': row['goods_shipped'],
                        'attribute': attr,
                        'pinecone_value': row[pinecone_col],
                        'llm_value': row[llm_col]
                    })
    
    if total_comparisons > 0:
        results['overall_agreement'] = total_agreements / total_comparisons
    
    return results


def analyze_multilingual_performance(
    df: pd.DataFrame,
    language_patterns: Dict[str, str]
) -> Dict[str, Any]:
    """
    Analyze performance on multilingual data.
    
    Args:
        df: DataFrame with classifications
        language_patterns: Dict mapping language name to regex pattern
    
    Returns:
        Dictionary with per-language metrics
    """
    results = {}
    
    for lang, pattern in language_patterns.items():
        lang_df = df[df['goods_shipped'].str.contains(pattern, case=False, na=False)]
        
        if len(lang_df) > 0:
            results[lang] = {
                'record_count': len(lang_df),
                'sample_goods': lang_df['goods_shipped'].head(3).tolist()
            }
    
    return results


def generate_report(
    comparison: Dict[str, Any],
    pinecone_stats: Dict[str, Any],
    llm_stats: Dict[str, Any],
    output_file: str
):
    """Generate analysis report."""
    report = {
        'comparison': comparison,
        'pinecone_stats': pinecone_stats,
        'llm_stats': llm_stats,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=NpEncoder)
    
    logger.info(f"Report saved to {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("ATTRIBUTE CLASSIFICATION ANALYSIS REPORT")
    print("="*80)
    
    print(f"\n📊 Overall Comparison:")
    print(f"  Total records compared: {comparison['total_records']}")
    print(f"  Total attributes: {comparison['total_attributes']}")
    print(f"  Overall agreement: {comparison['overall_agreement']*100:.2f}%")
    
    print(f"\n🎯 Per-Attribute Agreement Rates:")
    sorted_attrs = sorted(
        comparison['per_attribute_agreement'].items(),
        key=lambda x: x[1]['agreement_rate'],
        reverse=True
    )
    for attr, metrics in sorted_attrs[:10]:
        attr_name = attr.replace('attr_', '')
        print(f"  {attr_name:30s}: {metrics['agreement_rate']*100:5.1f}% "
              f"({metrics['agreements']}/{metrics['total_comparisons']})")
    
    if len(sorted_attrs) > 10:
        print(f"  ... and {len(sorted_attrs) - 10} more attributes")
    
    print(f"\n⚠️  Sample Disagreements (first 10):")
    for i, disagreement in enumerate(comparison['disagreements'][:10], 1):
        attr_name = disagreement['attribute'].replace('attr_', '')
        print(f"\n  {i}. Goods: {disagreement['goods_shipped'][:60]}")
        print(f"     Attribute: {attr_name}")
        print(f"     Pinecone: {disagreement['pinecone_value']}")
        print(f"     LLM: {disagreement['llm_value']}")
    
    if pinecone_stats:
        print(f"\n🚀 Pinecone Performance:")
        print(f"  Vector match rate: {pinecone_stats.get('vector_match_rate', 0)*100:.1f}%")
        print(f"  LLM fallback rate: {pinecone_stats.get('llm_fallback_rate', 0)*100:.1f}%")
        print(f"  Cost savings: {pinecone_stats.get('vector_match_rate', 0)*100:.1f}%")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Pinecone vs LLM attribute classification performance"
    )
    parser.add_argument(
        "--pinecone",
        required=True,
        help="Path to Pinecone classification CSV"
    )
    parser.add_argument(
        "--llm",
        required=True,
        help="Path to LLM classification CSV"
    )
    parser.add_argument(
        "--output",
        default="attribute_classification_analysis.json",
        help="Output JSON file for detailed report"
    )
    parser.add_argument(
        "--pinecone-stats",
        help="Optional: JSON file with Pinecone runtime stats"
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading Pinecone classifications from {args.pinecone}")
    pinecone_df = load_classifications(args.pinecone)
    
    logger.info(f"Loading LLM classifications from {args.llm}")
    llm_df = load_classifications(args.llm)
    
    # Compare classifications
    logger.info("Comparing classifications...")
    comparison = compare_classifications(pinecone_df, llm_df)
    
    # Load Pinecone stats if available
    pinecone_stats = {}
    if args.pinecone_stats:
        try:
            with open(args.pinecone_stats, 'r') as f:
                pinecone_stats = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load Pinecone stats: {e}")
    
    # Generate report
    generate_report(
        comparison,
        pinecone_stats,
        {},  # LLM stats placeholder
        args.output
    )


if __name__ == "__main__":
    main()
