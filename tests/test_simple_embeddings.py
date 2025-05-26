#!/usr/bin/env python3
"""Simple test for embedding generation."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pythonHandler'))

def test_simple_embeddings():
    """Test if embeddings can be generated."""
    print("Testing embedding generation...")
    
    try:
        from utils.local_embedding_utils import generate_embeddings
        
        test_texts = ["woolworths", "kmart", "new world", "woolworths store"]
        print(f"Testing with texts: {test_texts}")
        
        embeddings = generate_embeddings(test_texts)
        
        if embeddings is not None:
            print(f"SUCCESS: Generated embeddings with shape: {embeddings.shape}")
            
            # Test similarity calculation
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(embeddings)
            print(f"Similarity matrix shape: {similarities.shape}")
            
            # Check if woolworths entries are similar
            woolworths_indices = [i for i, text in enumerate(test_texts) if 'woolworths' in text.lower()]
            if len(woolworths_indices) >= 2:
                sim_score = similarities[woolworths_indices[0], woolworths_indices[1]]
                print(f"Similarity between woolworths entries: {sim_score:.3f}")
                
            return True
            
        else:
            print("FAILED: Could not generate embeddings")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_embeddings()
    exit(0 if success else 1)
