"""
Analyze and compare classification performance between Pinecone and pure LLM approaches.

This script analyzes classification results to measure:
- Accuracy comparison
- Speed improvements
- Cost savings
- Similarity score distributions
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_classification_results(csv_path: str) -> pd.DataFrame:
    """Load classification results CSV."""
    logger.info(f"Loading classification results from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} records")
    return df


def analyze_category_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze category distribution in results."""
    logger.info("Analyzing category distribution...")
    
    category_counts = df['category'].value_counts()
    total = len(df)
    
    distribution = {}
    for category, count in category_counts.items():
        distribution[category] = {
            'count': int(count),
            'percentage': round(count / total * 100, 2)
        }
    
    return {
        'total_records': total,
        'unique_categories': len(category_counts),
        'distribution': distribution
    }


def compare_classifications(
    pinecone_csv: str,
    llm_csv: str
) -> Dict[str, Any]:
    """
    Compare classifications between Pinecone and LLM approaches.
    
    Parameters:
    -----------
    pinecone_csv : str
        Path to Pinecone classification results
    llm_csv : str
        Path to pure LLM classification results
        
    Returns:
    --------
    Dict[str, Any]
        Comparison metrics
    """
    logger.info("Comparing Pinecone vs LLM classifications...")
    
    df_pinecone = pd.read_csv(pinecone_csv)
    df_llm = pd.read_csv(llm_csv)
    
    # Ensure same records
    if len(df_pinecone) != len(df_llm):
        logger.warning(f"Record count mismatch: Pinecone={len(df_pinecone)}, LLM={len(df_llm)}")
    
    # Merge on shipment_id or goods_shipped
    merge_key = 'shipment_id' if 'shipment_id' in df_pinecone.columns else 'goods_shipped'
    
    merged = df_pinecone.merge(
        df_llm[[merge_key, 'category']],
        on=merge_key,
        suffixes=('_pinecone', '_llm'),
        how='inner'
    )
    
    logger.info(f"Comparing {len(merged)} matching records")
    
    # Calculate agreement
    merged['agrees'] = merged['category_pinecone'] == merged['category_llm']
    agreement_rate = merged['agrees'].sum() / len(merged) * 100
    
    # Find disagreements
    disagreements = merged[~merged['agrees']]
    
    # Category-wise agreement
    category_agreement = {}
    for category in df_llm['category'].unique():
        cat_records = merged[merged['category_llm'] == category]
        if len(cat_records) > 0:
            cat_agreement = cat_records['agrees'].sum() / len(cat_records) * 100
            category_agreement[category] = {
                'total': len(cat_records),
                'agreement_rate': round(cat_agreement, 2)
            }
    
    return {
        'total_compared': len(merged),
        'agreement_rate': round(agreement_rate, 2),
        'disagreements': len(disagreements),
        'category_agreement': category_agreement,
        'sample_disagreements': disagreements[[merge_key, 'category_pinecone', 'category_llm']].head(10).to_dict('records')
    }


def analyze_performance_metrics(checkpoint_file: str = None) -> Dict[str, Any]:
    """
    Analyze performance metrics from checkpoint or logs.
    
    Parameters:
    -----------
    checkpoint_file : str, optional
        Path to checkpoint pickle file with metrics
        
    Returns:
    --------
    Dict[str, Any]
        Performance metrics
    """
    if not checkpoint_file or not Path(checkpoint_file).exists():
        logger.warning("No checkpoint file found, skipping performance analysis")
        return {}
    
    import pickle
    
    logger.info(f"Loading checkpoint from {checkpoint_file}")
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
    
    total_goods = checkpoint.get('processed_count', 0)
    vector_matches = checkpoint.get('vector_match_count', 0)
    llm_fallbacks = checkpoint.get('llm_fallback_count', 0)
    total_similarity = checkpoint.get('total_similarity_score', 0.0)
    
    metrics = {
        'total_classified': total_goods,
        'vector_matches': vector_matches,
        'llm_fallbacks': llm_fallbacks,
        'vector_match_rate': round(vector_matches / total_goods * 100, 2) if total_goods > 0 else 0,
        'avg_similarity_score': round(total_similarity / vector_matches, 3) if vector_matches > 0 else 0,
        'cost_savings_pct': round(vector_matches / total_goods * 100, 2) if total_goods > 0 else 0
    }
    
    return metrics


def generate_report(
    output_path: str,
    category_analysis: Dict[str, Any],
    comparison: Dict[str, Any] = None,
    performance: Dict[str, Any] = None
):
    """Generate comprehensive analysis report."""
    logger.info(f"Generating report to {output_path}")
    
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'category_distribution': category_analysis,
    }
    
    if comparison:
        report['classification_comparison'] = comparison
    
    if performance:
        report['performance_metrics'] = performance
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("CLASSIFICATION ANALYSIS REPORT")
    print("="*80)
    
    print(f"\n📊 Category Distribution:")
    print(f"  Total records: {category_analysis['total_records']}")
    print(f"  Categories: {category_analysis['unique_categories']}")
    for cat, info in category_analysis['distribution'].items():
        print(f"    {cat}: {info['count']} ({info['percentage']}%)")
    
    if comparison:
        print(f"\n🔍 Pinecone vs LLM Comparison:")
        print(f"  Records compared: {comparison['total_compared']}")
        print(f"  Agreement rate: {comparison['agreement_rate']}%")
        print(f"  Disagreements: {comparison['disagreements']}")
        
        if comparison['sample_disagreements']:
            print(f"\n  Sample disagreements:")
            for i, row in enumerate(comparison['sample_disagreements'][:5], 1):
                print(f"    {i}. Pinecone: {row.get('category_pinecone')} | LLM: {row.get('category_llm')}")
    
    if performance:
        print(f"\n⚡ Performance Metrics:")
        print(f"  Total classified: {performance['total_classified']}")
        print(f"  Vector matches: {performance['vector_matches']} ({performance['vector_match_rate']}%)")
        print(f"  LLM fallbacks: {performance['llm_fallbacks']}")
        print(f"  Avg similarity: {performance['avg_similarity_score']}")
        print(f"  Cost savings: {performance['cost_savings_pct']}%")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze classification performance"
    )
    
    parser.add_argument(
        "--pinecone-results",
        required=True,
        help="Path to Pinecone classification CSV"
    )
    parser.add_argument(
        "--llm-results",
        help="Path to pure LLM classification CSV (for comparison)"
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to checkpoint file with performance metrics"
    )
    parser.add_argument(
        "--output",
        default="classification_analysis.json",
        help="Output JSON report path (default: classification_analysis.json)"
    )
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    # Analyze Pinecone results
    df_pinecone = load_classification_results(args.pinecone_results)
    category_analysis = analyze_category_distribution(df_pinecone)
    
    # Compare with LLM if provided
    comparison = None
    if args.llm_results:
        comparison = compare_classifications(args.pinecone_results, args.llm_results)
    
    # Analyze performance
    performance = None
    if args.checkpoint:
        performance = analyze_performance_metrics(args.checkpoint)
    
    # Generate report
    generate_report(
        output_path=args.output,
        category_analysis=category_analysis,
        comparison=comparison,
        performance=performance
    )


if __name__ == "__main__":
    main()
