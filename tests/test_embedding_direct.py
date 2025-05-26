#!/usr/bin/env python3
"""Test script for embedding-based grouping functionality."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pythonHandler'))

from utils.text_utils import CleaningConfig, clean_and_group_transactions

def test_embedding_grouping():
    """Test the embedding-based grouping functionality."""
    
    print("=== Testing Embedding-Based Grouping ===")
    
    # Test data with similar merchants
    test_descriptions = [
        "4835-****-****-0311  Df Woolworths N",
        "4835-****-****-0329  Df Woolworths N", 
        "4835-****-****-0311  Df Woolworths O",
        "4835-****-****-0329  Df New World Mt",
        "4835-****-****-0311  Df New World Mt",
        "4835-****-****-0329  Df New World Bl",
        "4835-****-****-0311  Df Kmart - Bayf",
        "4835-****-****-0329  Df Kmart",
    ]
    
    print(f"Testing with {len(test_descriptions)} descriptions:")
    for i, desc in enumerate(test_descriptions):
        print(f"  {i+1}: {desc}")
    
    print("\n1. Testing regular cleaning (no grouping)...")
    config_regular = CleaningConfig()
    config_regular.use_embedding_grouping = False
    config_regular.use_fuzzy_matching = False
    
    try:
        cleaned_regular, groups_regular = clean_and_group_transactions(test_descriptions, config_regular)
        print(f"   Regular cleaning successful. Groups: {len(groups_regular)}")
        for i, (orig, clean) in enumerate(zip(test_descriptions, cleaned_regular)):
            print(f"     {i+1}: '{orig}' -> '{clean}'")
    except Exception as e:
        print(f"   Regular cleaning failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n2. Testing fuzzy matching grouping...")
    config_fuzzy = CleaningConfig()
    config_fuzzy.use_embedding_grouping = False
    config_fuzzy.use_fuzzy_matching = True
    config_fuzzy.similarity_threshold = 0.8
    
    try:
        cleaned_fuzzy, groups_fuzzy = clean_and_group_transactions(test_descriptions, config_fuzzy)
        print(f"   Fuzzy grouping successful. Groups: {len(groups_fuzzy)}")
        
        # Count unique group representatives
        unique_groups = set(groups_fuzzy.values())
        print(f"   Unique groups found: {len(unique_groups)}")
        
        # Show grouping results
        for representative in unique_groups:
            members = [k for k, v in groups_fuzzy.items() if v == representative]
            print(f"     Group '{representative}': {len(members)} members")
            for member in members:
                print(f"       - '{member}'")
                
    except Exception as e:
        print(f"   Fuzzy grouping failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n3. Testing embedding-based grouping...")
    config_embedding = CleaningConfig()
    config_embedding.use_embedding_grouping = True
    config_embedding.embedding_clustering_method = "similarity"  # Start with simplest method
    config_embedding.embedding_similarity_threshold = 0.8
    
    try:
        cleaned_embedding, groups_embedding = clean_and_group_transactions(test_descriptions, config_embedding)
        print(f"   Embedding grouping successful. Groups: {len(groups_embedding)}")
        
        # Count unique group representatives
        unique_groups = set(groups_embedding.values())
        print(f"   Unique groups found: {len(unique_groups)}")
        
        # Show grouping results
        for representative in unique_groups:
            members = [k for k, v in groups_embedding.items() if v == representative]
            print(f"     Group '{representative}': {len(members)} members")
            for member in members:
                print(f"       - '{member}'")
                
    except Exception as e:
        print(f"   Embedding grouping failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Check if embeddings are available
        try:
            from utils.local_embedding_utils import generate_embeddings
            print("   Embedding utils are available")
            
            # Test generating embeddings for a few texts
            test_texts = ["woolworths", "kmart", "new world"]
            embeddings = generate_embeddings(test_texts)
            print(f"   Generated embeddings shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'unknown'}")
            
        except Exception as embed_e:
            print(f"   Embedding generation failed: {embed_e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== Test completed ===")
    return True

if __name__ == "__main__":
    test_embedding_grouping()
