#!/usr/bin/env python3
"""
Comprehensive test to find the optimal balance between text cleaning aggressiveness 
and embedding effectiveness for transaction grouping
"""

import sys
import csv
import time
import re
from pathlib import Path

# Add the pythonHandler directory to path
import os
pythonHandler_path = str(Path(__file__).parent.parent / 'pythonHandler')
abs_path = str(Path(pythonHandler_path).absolute())
sys.path.insert(0, abs_path)

from utils.text_utils import CleaningConfig, clean_and_group_transactions, clean_text


def create_cleaning_levels():
    """Create different levels of text cleaning aggressiveness"""
    
    def minimal_clean(text):
        """Level 1: Minimal cleaning - only remove obvious noise"""
        text = re.sub(r'\d{4}-\*{4}-\*{4}-\d{4}', '', text)  # Remove card numbers
        text = re.sub(r'\b\d{10,}\b', '', text)  # Remove very long numeric codes
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text
    
    def moderate_clean(text):
        """Level 2: Moderate cleaning - remove codes but keep location/type info"""
        text = re.sub(r'\d{4}-\*{4}-\*{4}-\d{4}', '', text)  # Remove card numbers
        text = re.sub(r'\b\d{6,}\b', '', text)  # Remove long numeric codes
        text = re.sub(r'\b[A-Z]{2,3}\d{3,}\b', '', text)  # Remove reference codes
        text = re.sub(r'\*', '', text)  # Remove asterisks
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text
    
    def aggressive_clean(text):
        """Level 3: Aggressive cleaning - current implementation"""
        return clean_text(text)
    
    return {
        'minimal': minimal_clean,
        'moderate': moderate_clean, 
        'aggressive': aggressive_clean
    }


def load_transaction_sample(csv_file, limit=300):
    """Load a sample of transaction data"""
    csv_path = Path(__file__).parent / 'test_data' / csv_file
    descriptions = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            details = row.get('Details', '').strip()
            particulars = row.get('Particulars', '').strip()
            code = row.get('Code', '').strip()
            
            combined_parts = [p for p in [details, particulars, code] if p]
            if combined_parts:
                descriptions.append(' '.join(combined_parts))
    
    return descriptions


def analyze_grouping_results(original_descriptions, final_descriptions, processing_time, test_name):
    """Analyze and return grouping results"""
    original_unique = len(set(original_descriptions))
    final_unique = len(set(final_descriptions))
    
    group_counts = {}
    for desc in final_descriptions:
        group_counts[desc] = group_counts.get(desc, 0) + 1
    
    sorted_groups = sorted(group_counts.items(), key=lambda x: x[1], reverse=True)
    
    large_groups = len([g for g in sorted_groups if g[1] >= 10])
    medium_groups = len([g for g in sorted_groups if g[1] >= 5])
    small_groups = len([g for g in sorted_groups if g[1] >= 3])
    singletons = len([g for g in sorted_groups if g[1] == 1])
    
    reduction_percent = ((original_unique - final_unique) / original_unique * 100) if original_unique > 0 else 0
    
    return {
        'test_name': test_name,
        'processing_time': processing_time,
        'original_unique': original_unique,
        'final_unique': final_unique,
        'reduction_percent': reduction_percent,
        'large_groups': large_groups,
        'medium_groups': medium_groups,
        'small_groups': small_groups,
        'singletons': singletons,
        'top_groups': sorted_groups[:5],
        'effectiveness_score': large_groups + (medium_groups * 0.5) + (reduction_percent * 0.1)
    }


def main():
    print("🔬 COMPREHENSIVE TEXT CLEANING vs EMBEDDING OPTIMIZATION ANALYSIS")
    print("=" * 80)
    
    # Load test data
    csv_file = "ANZ Transactions Nov 2024 to May 2025.csv"
    descriptions = load_transaction_sample(csv_file, limit=200)
    print(f"📝 Loaded {len(descriptions)} transaction descriptions")
    
    # Get different cleaning levels
    cleaning_levels = create_cleaning_levels()
    
    # Test configurations
    test_configs = []
    
    # For each cleaning level, test with and without embeddings
    for level_name, clean_func in cleaning_levels.items():
        # Without embeddings
        test_configs.append({
            'name': f'{level_name.title()} cleaning only',
            'cleaning_level': level_name,
            'clean_func': clean_func,
            'use_embeddings': False
        })
        
        # With embeddings - different configurations
        embedding_configs = [
            ('similarity', 0.7),
            ('similarity', 0.6),
            ('hdbscan', None)
        ]
        
        for method, threshold in embedding_configs:
            config_name = f'{level_name.title()} + {method}'
            if threshold:
                config_name += f' {threshold}'
            
            test_configs.append({
                'name': config_name,
                'cleaning_level': level_name,
                'clean_func': clean_func,
                'use_embeddings': True,
                'embedding_method': method,
                'embedding_threshold': threshold
            })
    
    results = []
    
    # Run all test configurations
    for config in test_configs:
        print(f"\n🧪 Testing: {config['name']}")
        
        start_time = time.time()
        
        try:
            if not config['use_embeddings']:
                # Just apply the cleaning function
                cleaned_descriptions = [config['clean_func'](desc) for desc in descriptions]
                processing_time = time.time() - start_time
                
            else:
                # Use embeddings with pre-cleaned text
                pre_cleaned = [config['clean_func'](desc) for desc in descriptions]
                
                # Configure embedding settings
                embedding_config = CleaningConfig()
                embedding_config.use_embedding_grouping = True
                embedding_config.embedding_clustering_method = config['embedding_method']
                if config['embedding_threshold']:
                    embedding_config.embedding_similarity_threshold = config['embedding_threshold']
                
                # Apply embeddings to pre-cleaned text
                cleaned_descriptions, groups = clean_and_group_transactions(pre_cleaned, embedding_config)
                processing_time = time.time() - start_time
            
            # Analyze results
            result = analyze_grouping_results(descriptions, cleaned_descriptions, processing_time, config['name'])
            results.append(result)
            
            print(f"   ⏱️  Time: {result['processing_time']:.2f}s")
            print(f"   📊 Unique: {result['original_unique']} → {result['final_unique']} ({result['reduction_percent']:.1f}% reduction)")
            print(f"   🎯 Groups: Large(10+)={result['large_groups']}, Medium(5+)={result['medium_groups']}, Small(3+)={result['small_groups']}")
            print(f"   🏆 Effectiveness score: {result['effectiveness_score']:.1f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Print comprehensive comparison
    print(f"\n📊 COMPREHENSIVE RESULTS COMPARISON")
    print("=" * 100)
    print(f"{'Configuration':<25} | {'Large':<5} | {'Med':<3} | {'Red%':<5} | {'Time':<5} | {'Score':<5}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['test_name']:<25} | {result['large_groups']:>5} | {result['medium_groups']:>3} | {result['reduction_percent']:>5.1f} | {result['processing_time']:>5.1f} | {result['effectiveness_score']:>5.1f}")
    
    # Find optimal configurations
    best_effectiveness = max(results, key=lambda x: x['effectiveness_score'])
    best_large_groups = max(results, key=lambda x: x['large_groups'])
    fastest = min(results, key=lambda x: x['processing_time'])
    best_reduction = max(results, key=lambda x: x['reduction_percent'])
    
    print(f"\n🏆 OPTIMAL CONFIGURATIONS:")
    print(f"   Best overall effectiveness: {best_effectiveness['test_name']} (score: {best_effectiveness['effectiveness_score']:.1f})")
    print(f"   Most large groups: {best_large_groups['test_name']} ({best_large_groups['large_groups']} large groups)")
    print(f"   Highest reduction: {best_reduction['test_name']} ({best_reduction['reduction_percent']:.1f}% reduction)")
    print(f"   Fastest processing: {fastest['test_name']} ({fastest['processing_time']:.2f}s)")
    
    # Show effectiveness analysis
    print(f"\n💡 KEY INSIGHTS:")
    
    # Compare cleaning levels
    cleaning_only_results = [r for r in results if 'only' in r['test_name']]
    if len(cleaning_only_results) >= 2:
        print(f"   📈 Text Cleaning Impact:")
        for result in cleaning_only_results:
            level = result['test_name'].split()[0].lower()
            print(f"      {level.title()}: {result['large_groups']} large groups, {result['reduction_percent']:.1f}% reduction")
    
    # Compare embedding impact for each cleaning level
    print(f"   🤖 Embedding Impact by Cleaning Level:")
    for level in ['minimal', 'moderate', 'aggressive']:
        level_results = [r for r in results if level in r['test_name'].lower()]
        if len(level_results) >= 2:
            baseline = next((r for r in level_results if 'only' in r['test_name']), None)
            best_embedding = max([r for r in level_results if 'only' not in r['test_name']], 
                               key=lambda x: x['large_groups'], default=None)
            
            if baseline and best_embedding:
                improvement = best_embedding['large_groups'] - baseline['large_groups']
                time_cost = best_embedding['processing_time'] - baseline['processing_time']
                print(f"      {level.title()}: +{improvement} large groups, +{time_cost:.1f}s processing time")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    
    if best_effectiveness['use_embeddings'] if 'use_embeddings' in locals() else 'embedding' in best_effectiveness['test_name'].lower():
        print(f"   ✅ USE EMBEDDINGS: {best_effectiveness['test_name']} provides best overall results")
        print(f"      - {best_effectiveness['large_groups']} large groups")
        print(f"      - {best_effectiveness['reduction_percent']:.1f}% reduction")
        print(f"      - {best_effectiveness['processing_time']:.1f}s processing time")
    else:
        print(f"   ⚠️  EMBEDDINGS NOT BENEFICIAL: {best_effectiveness['test_name']} performs best")
        print(f"      Basic text cleaning is sufficient for this dataset.")
    
    print(f"\n✅ Analysis complete! Use these insights to optimize your configuration.")


if __name__ == "__main__":
    main()
