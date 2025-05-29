#!/usr/bin/env python3
"""Simple test to check if our embedding optimization is working"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pythonHandler'))

print("🧪 Simple Embedding Test")
print("=" * 30)

try:
    from utils.text_utils import clean_text, CleaningConfig
    print("✅ Successfully imported text_utils")
    
    # Test basic cleaning
    test_descriptions = [
        "PAYPAL *SPOTIFY ABC123 4029357733 NSW",
        "PAYPAL *SPOTIFY DEF456 4029357733 NSW", 
        "PAYPAL *SPOTIFY GHI789 4029357733 NSW",
        "WOOLWORTHS 1234 BONDI NSW",
        "WOOLWORTHS 5678 BONDI NSW",
        "COLES 9876 SYDNEY NSW"
    ]
    
    print(f"\n📝 Testing with {len(test_descriptions)} descriptions:")
    for i, desc in enumerate(test_descriptions):
        print(f"  {i+1}. {desc}")
    
    # Test basic cleaning
    print(f"\n🧹 Basic cleaning results:")
    for i, desc in enumerate(test_descriptions):
        cleaned = clean_text(desc)
        print(f"  {i+1}. {cleaned}")
    
    # Test with embedding grouping
    try:
        from utils.text_utils import clean_and_group_transactions
        
        config = CleaningConfig()
        config.use_embedding_grouping = True
        config.embedding_clustering_method = 'similarity'
        config.embedding_similarity_threshold = 0.7
        
        print(f"\n🤖 Testing embedding grouping (similarity 0.7)...")
        cleaned_descriptions, groups = clean_and_group_transactions(test_descriptions, config)
        
        print(f"Cleaned descriptions:")
        for i, desc in enumerate(cleaned_descriptions):
            print(f"  {i+1}. {desc}")
            
        print(f"\nGroups found: {len(groups)}")
        for group_key, group_items in groups.items():
            print(f"  Group '{group_key}': {group_items}")
            
        print("✅ Embedding grouping test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in embedding grouping: {e}")
        import traceback
        traceback.print_exc()
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Simple test completed!")
