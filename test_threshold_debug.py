import requests
import json

def test_embedding_with_debug():
    print("Testing embedding grouping with detailed output...")
    
    # Test data
    test_data = {
        "descriptions": ["nova energy", "nova energy onlineeftpos"],
        "use_embedding_grouping": True,
        "embedding_similarity_threshold": 0.6
    }
    
    headers = {
        'Content-Type': 'application/json',
        'X-API-Key': 'test-api-key-123',
        'X-User-ID': 'test-user-123'
    }
    
    print(f"Request data: {json.dumps(test_data, indent=2)}")
    
    # Test with different thresholds
    thresholds = [0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    
    for threshold in thresholds:
        print(f"\n{'='*50}")
        print(f"Testing with threshold: {threshold}")
        print('='*50)
        
        test_data["embedding_similarity_threshold"] = threshold
        
        try:
            response = requests.post(
                'http://localhost/clean_text',
                json=test_data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                groups = result.get('groups', {})
                
                print(f"✅ Success! Number of groups: {len(groups)}")
                
                for group_name, group_items in groups.items():
                    print(f"  Group: {group_name}")
                    print(f"    Items: {group_items}")
                
                # Check if Nova entries are grouped together
                if len(groups) == 1:
                    print("🎉 SUCCESS: Nova Energy entries are grouped together!")
                    break
                else:
                    print("❌ Nova Energy entries are still separate")
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_embedding_with_debug()
