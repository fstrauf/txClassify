#!/usr/bin/env python3
"""
Simple test for embedding optimization features.
"""

import sys
import os
import time

# Add the pythonHandler directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pythonHandler'))

from utils.text_utils import CleaningConfig, clean_and_group_transactions

def test_basic_optimization():
    """Test basic optimization features."""
    print("=== Testing Basic Optimization Features ===")
    
    test_descriptions = [
        "Woolworths Store 123",
        "Woolworths Store 456", 
        "New World Market",
        "Kmart Department Store"
    ]
    
    # Test with caching enabled
    config = CleaningConfig(
        enable_grouping=True,
        grouping_method="embedding",
        embedding_use_cache=True,
        embedding_clustering_method="similarity"
    )
    
    print("Running embedding-based grouping with caching...")
    start_time = time.time()
    result = clean_and_group_transactions(test_descriptions, config)
    processing_time = time.time() - start_time
    
    print(f"Processing completed in {processing_time:.3f} seconds")
    print(f"Input: {len(test_descriptions)} descriptions")
    print(f"Output: {len(set(result[1].values()))} unique groups")
    
    # Show results
    groups_dict = {}
    for original, canonical in result[1].items():
        if canonical not in groups_dict:
            groups_dict[canonical] = []
        groups_dict[canonical].append(original)
    
    print("Groups:")
    for canonical, members in groups_dict.items():
        print(f"  '{canonical}': {len(members)} members")
        for member in members:
            print(f"    - {member}")
    
    print("✓ Basic optimization test completed successfully")

if __name__ == "__main__":
    test_basic_optimization()
