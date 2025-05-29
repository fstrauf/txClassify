#!/usr/bin/env python3
"""
Test script for the new universal categorization endpoint.
"""

import requests
import json
import sys
import os
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:5003"
API_KEY = "test_api_key_placeholder"  # Replace with actual test API key

def test_universal_categorization():
    """Test the new /categorize endpoint."""
    
    # Test data - mix of different transaction types
    test_transactions = [
        "UBER EATS SYDNEY NSW",
        "WOOLWORTHS METRO 123",
        {
            "description": "SALARY DEPOSIT CBA",
            "money_in": True,
            "amount": 2500.00
        },
        "GOOGLE ONE STORAGE",
        "NOVA ENERGY BILL",
        "NETFLIX SUBSCRIPTION",
        {
            "description": "ATM WITHDRAWAL ANZ",
            "money_in": False,
            "amount": 100.00
        },
        "RENT PAYMENT - JOHN SMITH",
        "COLES CENTRAL SYDNEY",
        "UBER RIDE FARE"
    ]
    
    print("Testing Universal Categorization Endpoint")
    print("=" * 50)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Test transactions: {len(test_transactions)}")
    print()
    
    # Make the request
    try:
        response = requests.post(
            f"{API_BASE_URL}/categorize",
            headers={
                "Content-Type": "application/json",
                "X-API-Key": API_KEY,
                "Accept": "application/json",
            },
            json={"transactions": test_transactions}
        )
        
        print(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Status: {result.get('status')}")
            print(f"Message: {result.get('message')}")
            
            if 'processing_info' in result:
                info = result['processing_info']
                print(f"Processing Info:")
                print(f"  - Total transactions: {info.get('total_transactions')}")
                print(f"  - Unique groups: {info.get('unique_groups')}")
                print(f"  - Processing time: {info.get('processing_time_seconds')}s")
                print(f"  - Categories used: {info.get('categories_used')}")
                print(f"  - Method: {info.get('method')}")
            
            if 'results' in result:
                print(f"\nResults Summary:")
                results = result['results']
                
                # Group by category for summary
                category_counts = {}
                for r in results:
                    cat = r.get('predicted_category', 'Unknown')
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                
                for category, count in sorted(category_counts.items()):
                    print(f"  - {category}: {count} transactions")
                
                print(f"\nDetailed Results:")
                for i, result_item in enumerate(results):
                    print(f"  {i+1}. '{result_item.get('narrative')}' → {result_item.get('predicted_category')}")
                    if result_item.get('adjustment_info', {}).get('group_representative'):
                        rep = result_item['adjustment_info']['group_representative']
                        if rep != result_item.get('cleaned_narrative'):
                            print(f"     Grouped with: '{rep}'")
            
            return True
            
        else:
            print("❌ Request failed!")
            try:
                error_data = response.json()
                print(f"Error: {error_data}")
            except:
                print(f"Error response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed!")
        print("Make sure the Flask server is running on localhost:5003")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_api_availability():
    """Test if the API is running and accessible."""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print("✅ API is online")
            print(f"Available endpoints: {data.get('endpoints', [])}")
            return True
        else:
            print("❌ API responded with error")
            return False
    except:
        print("❌ API is not accessible")
        return False

if __name__ == "__main__":
    print("Universal Categorization Endpoint Test")
    print("=" * 50)
    
    # Check if API key is provided
    if len(sys.argv) > 1:
        API_KEY = sys.argv[1]
        print(f"Using provided API key: {API_KEY[:8]}...")
    else:
        print("⚠️  Using placeholder API key. Provide real API key as argument if needed.")
    
    print()
    
    # Test API availability first
    if not test_api_availability():
        print("\n❌ Cannot proceed - API not available")
        sys.exit(1)
    
    print()
    
    # Test the universal categorization endpoint
    success = test_universal_categorization()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
