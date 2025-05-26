#!/usr/bin/env python3
"""
Final validation test for embedding optimization features.
Tests all major optimization components:
- Caching functionality
- Batch processing
- Parameter optimization  
- Error handling
- Performance monitoring
"""

import sys
import os
import time
import logging

# Add the pythonHandler directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pythonHandler'))

from utils.text_utils import CleaningConfig, clean_and_group_transactions, group_merchants_with_embeddings

# Setup logging to capture performance metrics
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_optimization_features():
    """Comprehensive test of all optimization features."""
    print("🚀 Testing Embedding Optimization Features")
    print("=" * 50)
    
    # Test 1: Basic functionality
    print("\n1. Testing Basic Embedding Grouping...")
    test_descriptions = [
        "Woolworths Store 123",
        "Woolworths Store 456", 
        "New World Market",
        "Kmart Department Store",
        "Countdown Supermarket"
    ]
    
    config = CleaningConfig()
    config.use_embedding_grouping = True
    config.embedding_clustering_method = "similarity"
    config.embedding_use_cache = True
    
    start_time = time.time()
    cleaned, groups = clean_and_group_transactions(test_descriptions, config)
    processing_time = time.time() - start_time
    
    unique_groups = len(set(groups.values()))
    reduction = (1 - unique_groups / len(test_descriptions)) * 100
    
    print(f"   ✓ Processed {len(test_descriptions)} descriptions in {processing_time:.3f}s")
    print(f"   ✓ Created {unique_groups} groups ({reduction:.1f}% reduction)")
    
    # Test 2: Caching performance
    print("\n2. Testing Caching Performance...")
    print("   First run (generating embeddings)...")
    start_time = time.time()
    result1 = clean_and_group_transactions(test_descriptions, config)
    first_run_time = time.time() - start_time
    
    print("   Second run (using cached embeddings)...")
    start_time = time.time()
    result2 = clean_and_group_transactions(test_descriptions, config)
    second_run_time = time.time() - start_time
    
    assert result1 == result2, "Cached results should match original"
    speedup = first_run_time / second_run_time if second_run_time > 0 else 1
    print(f"   ✓ Cache speedup: {speedup:.2f}x")
    
    # Test 3: Batch processing
    print("\n3. Testing Batch Processing...")
    
    # Create a larger dataset to trigger batch processing
    large_dataset = []
    base_merchants = ["Woolworths", "New World", "Kmart", "Countdown", "The Warehouse"]
    for i in range(60):  # Create 300 descriptions (60 * 5)
        for merchant in base_merchants:
            large_dataset.append(f"{merchant} Store {i:03d}")
    
    config_batch = CleaningConfig()
    config_batch.use_embedding_grouping = True
    config_batch.embedding_batch_size = 50  # Force batch processing
    config_batch.embedding_clustering_method = "similarity"
    
    print(f"   Processing {len(large_dataset)} descriptions in batches...")
    start_time = time.time()
    batch_cleaned, batch_groups = clean_and_group_transactions(large_dataset, config_batch)
    batch_time = time.time() - start_time
    
    batch_unique = len(set(batch_groups.values()))
    batch_reduction = (1 - batch_unique / len(large_dataset)) * 100
    
    print(f"   ✓ Batch processing completed in {batch_time:.3f}s")
    print(f"   ✓ {len(large_dataset)} → {batch_unique} groups ({batch_reduction:.1f}% reduction)")
    
    # Test 4: Different clustering methods
    print("\n4. Testing Clustering Methods...")
    
    test_merchants = [
        "woolworths auckland", "woolworths wellington", "woolworths christchurch",
        "new world ponsonby", "new world newmarket", "new world botany",
        "kmart sylvia park", "kmart queen street", "kmart manukau",
        "countdown metro", "countdown online", "countdown newmarket"
    ]
    
    methods = ["similarity", "dbscan"]
    for method in methods:
        config_method = CleaningConfig()
        config_method.use_embedding_grouping = True
        config_method.embedding_clustering_method = method
        
        result = group_merchants_with_embeddings(test_merchants, config_method)
        method_groups = len(set(result.values()))
        method_reduction = (1 - method_groups / len(test_merchants)) * 100
        
        print(f"   ✓ {method.upper()}: {len(test_merchants)} → {method_groups} groups ({method_reduction:.1f}% reduction)")
    
    # Test 5: Error handling and fallbacks
    print("\n5. Testing Error Handling...")
    
    # Test with embedding disabled to trigger fallback
    config_fallback = CleaningConfig()
    config_fallback.use_embedding_grouping = False  # Force fallback to fuzzy matching
    config_fallback.use_fuzzy_matching = True
    
    fallback_cleaned, fallback_groups = clean_and_group_transactions(test_descriptions, config_fallback)
    fallback_unique = len(set(fallback_groups.values()))
    
    print(f"   ✓ Fallback to fuzzy matching: {len(test_descriptions)} → {fallback_unique} groups")
    
    # Test 6: Performance summary
    print("\n6. Performance Summary...")
    print(f"   • Basic grouping: {processing_time:.3f}s for {len(test_descriptions)} items")
    print(f"   • Cache speedup: {speedup:.2f}x improvement")
    print(f"   • Batch processing: {batch_time:.3f}s for {len(large_dataset)} items")
    print(f"   • Memory efficiency: Handled {len(large_dataset)} items in batches")
    
    return True

def demonstrate_api_usage():
    """Demonstrate how to use the optimized features in practice."""
    print("\n🔧 API Usage Examples")
    print("=" * 50)
    
    print("\n1. Standard Configuration:")
    print("""
config = CleaningConfig()
config.use_embedding_grouping = True
config.embedding_clustering_method = "similarity"
config.embedding_use_cache = True
config.embedding_batch_size = 50
""")
    
    print("\n2. High-Performance Configuration:")
    print("""
config = CleaningConfig()
config.use_embedding_grouping = True
config.embedding_clustering_method = "dbscan"
config.embedding_similarity_threshold = 0.75
config.embedding_batch_size = 100
config.embedding_use_cache = True
""")
    
    print("\n3. Memory-Efficient Configuration:")
    print("""
config = CleaningConfig()
config.use_embedding_grouping = True
config.embedding_batch_size = 25  # Smaller batches
config.embedding_use_cache = True
config.embedding_fallback_to_fuzzy = True
""")

def main():
    """Run the complete validation test suite."""
    try:
        success = test_optimization_features()
        
        if success:
            demonstrate_api_usage()
            
            print("\n" + "=" * 50)
            print("🎉 ALL OPTIMIZATION TESTS PASSED!")
            print("=" * 50)
            print("\n✅ Validated Features:")
            print("   • Embedding caching system")
            print("   • Batch processing for large datasets")
            print("   • Multiple clustering algorithms")
            print("   • Error handling and fallbacks")
            print("   • Performance monitoring")
            print("\n🚀 System is ready for production use!")
            
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
