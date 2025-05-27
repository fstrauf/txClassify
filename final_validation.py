#!/usr/bin/env python3
"""
Final validation that the Nova Energy cross-batch merging fix is working.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pythonHandler'))

from utils.text_utils import group_merchants_with_embeddings, CleaningConfig, compute_similarity_score

def validate_fix():
    print("=== FINAL VALIDATION OF NOVA ENERGY FIX ===")
    
    # Test 1: Direct similarity computation
    print("\n1. Testing compute_similarity_score function:")
    config = CleaningConfig()
    score = compute_similarity_score("nova energy", "nova energy onlineeftpos", config)
    print(f"   Similarity score: {score:.4f}")
    print(f"   Threshold: {config.embedding_similarity_threshold}")
    print(f"   Should group: {'YES' if score >= config.embedding_similarity_threshold else 'NO'}")
    
    # Test 2: Small batch test (from our earlier successful test)
    print("\n2. Testing with small batches to force cross-batch merging:")
    config.embedding_batch_size = 2  # Force cross-batch scenario
    
    test_names = [
        "test merchant 1",
        "nova energy", 
        "test merchant 2", 
        "nova energy onlineeftpos"
    ]
    
    result = group_merchants_with_embeddings(test_names, config)
    
    nova_group = result.get("nova energy")
    nova_online_group = result.get("nova energy onlineeftpos")
    
    print(f"   'nova energy' -> '{nova_group}'")
    print(f"   'nova energy onlineeftpos' -> '{nova_online_group}'")
    print(f"   Grouped together: {'YES' if nova_group == nova_online_group else 'NO'}")
    
    # Test 3: Larger realistic test
    print("\n3. Testing with larger dataset (realistic scenario):")
    config.embedding_batch_size = 50  # Normal batch size
    
    # Create a larger test dataset that will definitely span multiple batches
    large_test = []
    
    # Add some filler merchants to force Nova Energy into different batches
    for i in range(45):
        large_test.append(f"merchant {i:02d}")
    
    large_test.append("nova energy")  # This will be in batch 1
    
    # Add more merchants
    for i in range(45, 90):
        large_test.append(f"merchant {i:02d}")
        
    large_test.append("nova energy onlineeftpos")  # This will be in batch 2
    
    # Add even more merchants  
    for i in range(90, 100):
        large_test.append(f"merchant {i:02d}")
    
    print(f"   Total merchants: {len(large_test)}")
    print(f"   Batch size: {config.embedding_batch_size}")
    print(f"   Expected batches: {(len(large_test) + config.embedding_batch_size - 1) // config.embedding_batch_size}")
    
    result_large = group_merchants_with_embeddings(large_test, config)
    
    nova_group_large = result_large.get("nova energy")
    nova_online_group_large = result_large.get("nova energy onlineeftpos")
    
    print(f"   'nova energy' -> '{nova_group_large}'")
    print(f"   'nova energy onlineeftpos' -> '{nova_online_group_large}'")
    print(f"   Grouped together: {'YES' if nova_group_large == nova_online_group_large else 'NO'}")
    
    # Summary
    print("\n=== SUMMARY ===")
    
    similarity_ok = score >= config.embedding_similarity_threshold
    small_batch_ok = nova_group == nova_online_group
    large_batch_ok = nova_group_large == nova_online_group_large
    
    print(f"✓ Direct similarity computation: {'PASS' if similarity_ok else 'FAIL'}")
    print(f"✓ Small batch cross-merging: {'PASS' if small_batch_ok else 'FAIL'}")
    print(f"✓ Large batch cross-merging: {'PASS' if large_batch_ok else 'FAIL'}")
    
    all_passed = similarity_ok and small_batch_ok and large_batch_ok
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! The Nova Energy cross-batch merging fix is working correctly.")
        print("\nThe issue has been resolved:")
        print("- Cross-batch merging now uses embedding similarity instead of Levenshtein similarity")
        print("- Nova Energy entries will be grouped together regardless of which batch they appear in")
        print("- The fix maintains backward compatibility with Levenshtein fallback if embeddings fail")
    else:
        print("\n❌ Some tests failed. The fix may need additional work.")
    
    return all_passed

if __name__ == "__main__":
    success = validate_fix()
    sys.exit(0 if success else 1)
