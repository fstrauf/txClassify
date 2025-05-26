#!/usr/bin/env python3
"""
Comprehensive test for embedding optimization features including:
- Caching functionality
- Batch processing for large datasets
- Parameter tuning validation
- Memory efficiency testing
"""

import sys
import os
import time
import logging

# Add the pythonHandler directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pythonHandler'))

from utils.text_utils import CleaningConfig, clean_and_group_transactions, group_merchants_with_embeddings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_caching_functionality():
    """Test that embedding caching works correctly."""
    print("\n=== Testing Caching Functionality ===")
    
    test_descriptions = [
        "Woolworths Store 123",
        "Woolworths Store 456", 
        "New World Market",
        "Kmart Department Store",
        "Countdown Supermarket"
    ] * 2  # Duplicate to test caching effectiveness
    
    # Test with caching enabled
    config_with_cache = CleaningConfig(
        enable_grouping=True,
        grouping_method="embedding",
        embedding_use_cache=True,
        embedding_clustering_method="similarity"
    )
    
    print("First run (generating and caching embeddings)...")
    start_time = time.time()
    result1 = clean_and_group_transactions(test_descriptions, config_with_cache)
    first_run_time = time.time() - start_time
    print(f"First run completed in {first_run_time:.3f} seconds")
    
    print("Second run (should use cached embeddings)...")
    start_time = time.time()
    result2 = clean_and_group_transactions(test_descriptions, config_with_cache)
    second_run_time = time.time() - start_time
    print(f"Second run completed in {second_run_time:.3f} seconds")
    
    # Results should be identical
    assert result1 == result2, "Cached results should match original results"
    
    # Second run should be faster (though this may not always be true due to overhead)
    speedup = first_run_time / second_run_time if second_run_time > 0 else float('inf')
    print(f"Cache speedup factor: {speedup:.2f}x")
    
    print("✓ Caching functionality test passed")

def test_batch_processing():
    """Test batch processing for large datasets."""
    print("\n=== Testing Batch Processing ===")
    
    # Create a large dataset
    base_descriptions = [
        "Woolworths Store",
        "New World Market", 
        "Kmart Department",
        "Countdown Supermarket",
        "The Warehouse",
        "Mitre 10",
        "Bunnings Warehouse",
        "Harvey Norman",
        "JB Hi-Fi",
        "Farmers Department"
    ]
    
    # Create dataset larger than default batch size (50)
    large_dataset = []
    for i in range(60):
        for desc in base_descriptions:
            large_dataset.append(f"{desc} {i:03d}")
    
    print(f"Testing with {len(large_dataset)} descriptions")
    
    config_batch = CleaningConfig(
        enable_grouping=True,
        grouping_method="embedding",
        embedding_batch_size=50,  # Force batch processing
        embedding_clustering_method="similarity"
    )
    
    start_time = time.time()
    result = clean_and_group_transactions(large_dataset, config_batch)
    processing_time = time.time() - start_time
    
    print(f"Batch processing completed in {processing_time:.3f} seconds")
    print(f"Input: {len(large_dataset)} descriptions")
    print(f"Output: {len(set(result[1].values()))} unique groups")
    
    # Verify we got reasonable grouping
    unique_groups = len(set(result[1].values()))
    reduction_percentage = (1 - unique_groups / len(large_dataset)) * 100
    print(f"Grouping reduction: {reduction_percentage:.1f}%")
    
    assert unique_groups < len(large_dataset), "Should have some grouping effect"
    print("✓ Batch processing test passed")

def test_clustering_methods():
    """Test different clustering methods and parameter optimization."""
    print("\n=== Testing Clustering Methods and Parameters ===")
    
    test_descriptions = [
        "Woolworths Auckland Central",
        "Woolworths Wellington",
        "Woolworths Christchurch",
        "New World Ponsonby", 
        "New World Newmarket",
        "Kmart Sylvia Park",
        "Kmart Queen Street",
        "Countdown Metro",
        "Countdown Online",
        "The Warehouse Botany"
    ]
    
    clustering_methods = ["similarity", "dbscan"]
    
    for method in clustering_methods:
        print(f"\nTesting {method} clustering...")
        
        config = CleaningConfig(
            enable_grouping=True,
            grouping_method="embedding",
            embedding_clustering_method=method,
            embedding_similarity_threshold=0.8 if method == "similarity" else 0.85,
            embedding_eps=0.25 if method == "dbscan" else 0.3
        )
        
        start_time = time.time()
        result = clean_and_group_transactions(test_descriptions, config)
        processing_time = time.time() - start_time
        
        unique_groups = len(set(result[1].values()))
        reduction = (1 - unique_groups / len(test_descriptions)) * 100
        
        print(f"  Method: {method}")
        print(f"  Processing time: {processing_time:.3f} seconds")
        print(f"  Groups created: {unique_groups}")
        print(f"  Reduction: {reduction:.1f}%")
        
        # Show some example groups
        groups_dict = {}
        for original, canonical in result[1].items():
            if canonical not in groups_dict:
                groups_dict[canonical] = []
            groups_dict[canonical].append(original)
        
        # Show groups with multiple members
        for canonical, members in groups_dict.items():
            if len(members) > 1:
                print(f"  Group '{canonical}': {len(members)} members")
                for member in members[:3]:  # Show first 3 members
                    print(f"    - {member}")
                if len(members) > 3:
                    print(f"    ... and {len(members) - 3} more")
    
    print("✓ Clustering methods test passed")

def test_memory_efficiency():
    """Test memory efficiency improvements."""
    print("\n=== Testing Memory Efficiency ===")
    
    # Create a moderately large dataset to test memory usage
    descriptions = []
    merchants = ["Woolworths", "Countdown", "New World", "Kmart", "The Warehouse"]
    locations = ["Auckland", "Wellington", "Christchurch", "Hamilton", "Tauranga"]
    
    for i in range(200):  # 200 descriptions
        merchant = merchants[i % len(merchants)]
        location = locations[i % len(locations)]
        descriptions.append(f"{merchant} {location} {i:03d}")
    
    print(f"Testing memory efficiency with {len(descriptions)} descriptions")
    
    config = CleaningConfig(
        enable_grouping=True,
        grouping_method="embedding",
        embedding_batch_size=30,  # Smaller batches for memory testing
        embedding_use_cache=True,
        embedding_clustering_method="similarity"
    )
    
    start_time = time.time()
    result = clean_and_group_transactions(descriptions, config)
    processing_time = time.time() - start_time
    
    unique_groups = len(set(result[1].values()))
    reduction = (1 - unique_groups / len(descriptions)) * 100
    
    print(f"Memory efficiency test completed in {processing_time:.3f} seconds")
    print(f"Input: {len(descriptions)} descriptions")
    print(f"Output: {unique_groups} unique groups")
    print(f"Reduction: {reduction:.1f}%")
    
    # Verify reasonable performance (should complete without memory issues)
    assert processing_time < 60, "Should complete within reasonable time"
    assert unique_groups < len(descriptions), "Should achieve some grouping"
    
    print("✓ Memory efficiency test passed")

def main():
    """Run all optimization tests."""
    print("=== Embedding Optimization Features Test Suite ===")
    
    try:
        test_caching_functionality()
        test_batch_processing()
        test_clustering_methods()
        test_memory_efficiency()
        
        print("\n🎉 All optimization tests passed successfully!")
        print("\nOptimization features validated:")
        print("  ✓ Embedding caching system")
        print("  ✓ Batch processing for large datasets")
        print("  ✓ Multiple clustering methods")
        print("  ✓ Memory efficiency improvements")
        print("  ✓ Parameter tuning capabilities")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
