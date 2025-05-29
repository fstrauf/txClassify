#!/usr/bin/env python3
"""
Debug script to test the comprehensive analysis step by step
"""

import sys
import csv
import time
from pathlib import Path

print("Starting debug analysis...")

# Add the pythonHandler directory to path
import os
pythonHandler_path = str(Path(__file__).parent.parent / 'pythonHandler')
abs_path = str(Path(pythonHandler_path).absolute())
sys.path.insert(0, abs_path)

print("Python path updated")

try:
    from utils.text_utils import CleaningConfig, clean_and_group_transactions, clean_text
    print("Successfully imported text_utils")
except Exception as e:
    print(f"Error importing text_utils: {e}")
    sys.exit(1)

def load_transaction_sample(csv_file, limit=50):
    """Load a small sample of transaction data for testing"""
    csv_path = Path(__file__).parent / 'test_data' / csv_file
    print(f"Loading CSV from: {csv_path}")
    
    descriptions = []
    
    try:
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
        
        print(f"Loaded {len(descriptions)} descriptions")
        return descriptions
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []

def simple_test():
    """Run a simple test"""
    print("\n=== SIMPLE EMBEDDING TEST ===")
    
    # Load test data
    descriptions = load_transaction_sample("ANZ Transactions Nov 2024 to May 2025.csv", 50)
    if not descriptions:
        return
    
    print(f"Sample descriptions:")
    for i, desc in enumerate(descriptions[:3]):
        print(f"  {i+1}. {desc}")
    
    # Test basic cleaning
    print("\n--- Testing basic cleaning ---")
    start_time = time.time()
    cleaned_basic = [clean_text(desc) for desc in descriptions]
    basic_time = time.time() - start_time
    
    unique_original = len(set(descriptions))
    unique_basic = len(set(cleaned_basic))
    basic_reduction = ((unique_original - unique_basic) / unique_original * 100)
    
    print(f"Basic cleaning: {unique_original} → {unique_basic} ({basic_reduction:.1f}% reduction)")
    print(f"Processing time: {basic_time:.2f}s")
    
    # Test with embeddings
    print("\n--- Testing with embeddings ---")
    start_time = time.time()
    
    config = CleaningConfig()
    config.use_embedding_grouping = True
    config.embedding_clustering_method = 'similarity'
    config.embedding_similarity_threshold = 0.7
    
    try:
        cleaned_embeddings, groups = clean_and_group_transactions(descriptions, config)
        embedding_time = time.time() - start_time
        
        unique_embeddings = len(set(cleaned_embeddings))
        embedding_reduction = ((unique_original - unique_embeddings) / unique_original * 100)
        
        print(f"With embeddings: {unique_original} → {unique_embeddings} ({embedding_reduction:.1f}% reduction)")
        print(f"Processing time: {embedding_time:.2f}s")
        print(f"Improvement: {embedding_reduction - basic_reduction:.1f} percentage points")
        
    except Exception as e:
        print(f"Error with embeddings: {e}")

if __name__ == "__main__":
    simple_test()
