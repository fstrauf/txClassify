#!/usr/bin/env python3
"""
Direct test of embedding optimization with different configurations
This bypasses the Flask API and tests the core functionality directly
"""

import os
import sys
import json
import time
from datetime import datetime
import csv
from pathlib import Path

# Add the pythonHandler directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pythonHandler'))

from utils.text_utils import clean_and_group_transactions, CleaningConfig, clean_text

def load_transaction_data(csv_file):
    """Load transaction data from CSV file"""
    csv_path = Path(__file__).parent / 'test_data' / csv_file
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    transactions = []
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        # Try to detect the CSV format
        sample = file.read(1024)
        file.seek(0)
        
        # Use csv.Sniffer to detect delimiter
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        
        reader = csv.DictReader(file, delimiter=delimiter)
        
        for row in reader:
            # Look for description-like fields
            description = None
            code = None
            
            # Check various possible column names
            for key, value in row.items():
                key_lower = key.lower().strip()
                if key_lower in ['description', 'narrative', 'details']:
                    description = value.strip() if value else None
                elif key_lower in ['code']:
                    code = value.strip() if value else None
            
            if description:
                # Combine description and code if both exist
                combined_desc = description
                if code and code.strip():
                    combined_desc = f"{description} {code}"
                
                transactions.append({
                    'description': description,
                    'code': code,
                    'combined': combined_desc.strip()
                })
    
    print(f"Loaded {len(transactions)} transactions from {csv_file}")
    return transactions

def run_configuration_test(descriptions, config_name, config):
    """Run a single configuration test"""
    print(f"\n🧪 Testing Configuration: {config_name}")
    print(f"   Use embedding grouping: {config.use_embedding_grouping}")
    if config.use_embedding_grouping:
        print(f"   Clustering method: {config.embedding_clustering_method}")
        print(f"   Similarity threshold: {config.embedding_similarity_threshold}")
    
    start_time = time.time()
    
    try:
        if config.use_embedding_grouping:
            cleaned_descriptions, groups = clean_and_group_transactions(descriptions, config)
        else:
            # Regular cleaning only
            cleaned_descriptions = [clean_text(desc) for desc in descriptions]
            groups = {}
    
        processing_time = time.time() - start_time
        
        # Analyze results
        original_unique = len(set(descriptions))
        cleaned_unique = len(set(cleaned_descriptions))
        
        # Group analysis
        group_counts = {}
        for cleaned in cleaned_descriptions:
            group_counts[cleaned] = group_counts.get(cleaned, 0) + 1
        
        sorted_groups = sorted(group_counts.items(), key=lambda x: x[1], reverse=True)
        large_groups = len([g for g in sorted_groups if g[1] >= 10])
        singletons = len([g for g in sorted_groups if g[1] == 1])
        
        reduction_percent = ((original_unique - cleaned_unique) / original_unique * 100) if original_unique > 0 else 0
        singleton_percent = (singletons / len(sorted_groups) * 100) if sorted_groups else 0
        
        result = {
            'config_name': config_name,
            'processing_time': processing_time,
            'original_unique': original_unique,
            'cleaned_unique': cleaned_unique,
            'reduction_percent': reduction_percent,
            'total_groups': len(sorted_groups),
            'large_groups': large_groups,
            'singletons': singletons,
            'singleton_percent': singleton_percent,
            'largest_group': max([g[1] for g in sorted_groups]) if sorted_groups else 0,
            'embedding_groups': len(groups) if groups else 0,
            'top_5_groups': sorted_groups[:5]
        }
        
        print(f"   ✅ Completed in {processing_time:.2f}s")
        print(f"   📊 {original_unique} → {cleaned_unique} unique ({reduction_percent:.1f}% reduction)")
        print(f"   🎯 {large_groups} large groups, {singletons} singletons ({singleton_percent:.1f}%)")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("🧪 DIRECT EMBEDDING OPTIMIZATION COMPARISON TEST")
    print("=" * 50)
    
    # Load test data
    csv_file = "ANZ Transactions Nov 2024 to May 2025.csv"
    
    try:
        transactions = load_transaction_data(csv_file)
        if not transactions:
            print("❌ No transactions loaded")
            return
            
        descriptions = [t['combined'] for t in transactions]
        print(f"📝 Testing with {len(descriptions)} transaction descriptions")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Define test configurations
    test_configs = [
        ("Basic text cleaning (no embeddings)", CleaningConfig()),
        
        ("Similarity 0.8 (current)", lambda: setattr(CleaningConfig(), 'use_embedding_grouping', True) or 
                                             setattr(CleaningConfig(), 'embedding_clustering_method', 'similarity') or
                                             setattr(CleaningConfig(), 'embedding_similarity_threshold', 0.8) or CleaningConfig()),
        
        ("Similarity 0.7 (lower threshold)", lambda: setattr(CleaningConfig(), 'use_embedding_grouping', True) or 
                                                     setattr(CleaningConfig(), 'embedding_clustering_method', 'similarity') or
                                                     setattr(CleaningConfig(), 'embedding_similarity_threshold', 0.7) or CleaningConfig()),
        
        ("Similarity 0.6 (lowest threshold)", lambda: setattr(CleaningConfig(), 'use_embedding_grouping', True) or 
                                                      setattr(CleaningConfig(), 'embedding_clustering_method', 'similarity') or
                                                      setattr(CleaningConfig(), 'embedding_similarity_threshold', 0.6) or CleaningConfig()),
    ]
    
    # Create configs properly
    configs = []
    
    # Basic config
    basic_config = CleaningConfig()
    basic_config.use_embedding_grouping = False
    configs.append(("Basic text cleaning (no embeddings)", basic_config))
    
    # Similarity 0.8
    sim_08_config = CleaningConfig()
    sim_08_config.use_embedding_grouping = True
    sim_08_config.embedding_clustering_method = 'similarity'
    sim_08_config.embedding_similarity_threshold = 0.8
    configs.append(("Similarity 0.8 (current)", sim_08_config))
    
    # Similarity 0.7
    sim_07_config = CleaningConfig()
    sim_07_config.use_embedding_grouping = True
    sim_07_config.embedding_clustering_method = 'similarity'
    sim_07_config.embedding_similarity_threshold = 0.7
    configs.append(("Similarity 0.7 (lower threshold)", sim_07_config))
    
    # Similarity 0.6
    sim_06_config = CleaningConfig()
    sim_06_config.use_embedding_grouping = True
    sim_06_config.embedding_clustering_method = 'similarity'
    sim_06_config.embedding_similarity_threshold = 0.6
    configs.append(("Similarity 0.6 (lowest threshold)", sim_06_config))
    
    # HDBSCAN if available
    try:
        import hdbscan
        hdbscan_config = CleaningConfig()
        hdbscan_config.use_embedding_grouping = True
        hdbscan_config.embedding_clustering_method = 'hdbscan'
        hdbscan_config.embedding_similarity_threshold = 0.8
        configs.append(("HDBSCAN clustering", hdbscan_config))
    except ImportError:
        print("⚠️  HDBSCAN not available, skipping that configuration")
    
    # Run tests
    results = []
    for config_name, config in configs:
        result = run_configuration_test(descriptions, config_name, config)
        if result:
            results.append(result)
    
    # Print comparison summary
    print(f"\n📊 CONFIGURATION COMPARISON RESULTS:")
    print(f"{'Configuration':<30} | {'Groups':<6} | {'Large':<5} | {'Reduce%':<7} | {'Time':<6}")
    print(f"{'-' * 30} | {'-' * 6} | {'-' * 5} | {'-' * 7} | {'-' * 6}")
    
    for r in results:
        print(f"{r['config_name'][:29]:<30} | {r['total_groups']:<6} | {r['large_groups']:<5} | {r['reduction_percent']:<7.1f} | {r['processing_time']:<6.1f}")
    
    # Find best configurations
    if results:
        best_large_groups = max(results, key=lambda x: x['large_groups'])
        best_reduction = max(results, key=lambda x: x['reduction_percent'])
        
        print(f"\n🏆 RECOMMENDATIONS:")
        print(f"   Best for large groups: {best_large_groups['config_name']} ({best_large_groups['large_groups']} large groups)")
        print(f"   Best for reduction: {best_reduction['config_name']} ({best_reduction['reduction_percent']:.1f}% reduction)")
        
        if best_large_groups['large_groups'] < 15:
            print(f"\n⚠️  INSIGHT: Even the best configuration only created {best_large_groups['large_groups']} large groups.")
            print(f"   This suggests the data might have inherently high diversity, or text cleaning")
            print(f"   is removing important distinguishing information.")
            
            # Show top groups from best config
            print(f"\n🔍 TOP GROUPS from {best_large_groups['config_name']}:")
            for i, (group_name, count) in enumerate(best_large_groups['top_5_groups'][:10]):
                print(f"   {i+1:2}. {group_name[:60]:<60} ({count:3} transactions)")
        
        # Save detailed results
        log_file = f"logs/direct_comparison_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        os.makedirs('logs', exist_ok=True)
        
        with open(log_file, 'w') as f:
            json.dump({
                'test_timestamp': datetime.now().isoformat(),
                'csv_file': csv_file,
                'total_transactions': len(descriptions),
                'results': results
            }, f, indent=2)
            
        print(f"\n📁 Detailed results saved to: {log_file}")
    
    print(f"\n✅ Comparison test completed!")

if __name__ == "__main__":
    main()
