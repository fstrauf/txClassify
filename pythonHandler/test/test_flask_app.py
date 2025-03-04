import logging
import requests
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger('test_flask_app')

# Load environment variables
load_dotenv()

def test_health_endpoint():
    """Test the health endpoint of the Flask app."""
    try:
        # Get the base URL from environment or use default
        base_url = os.environ.get("BACKEND_API", "http://localhost:5001")
        
        # Make a request to the health endpoint
        logger.info(f"Testing health endpoint at {base_url}/health")
        response = requests.get(f"{base_url}/health")
        
        # Check the response
        if response.status_code == 200:
            logger.info(f"Health endpoint returned status code {response.status_code}")
            logger.info(f"Response: {response.json()}")
            return True
        else:
            logger.error(f"Health endpoint returned status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error testing health endpoint: {str(e)}")
        return False

def test_debug_validate_key():
    """Test the debug/validate-key endpoint with a valid API key."""
    try:
        # Get the base URL and API key from environment
        base_url = os.environ.get("BACKEND_API", "http://localhost:5001")
        api_key = os.environ.get("TEST_API_KEY")
        
        if not api_key:
            logger.error("TEST_API_KEY environment variable not found")
            return False
        
        # Make a request to the debug/validate-key endpoint
        logger.info(f"Testing debug/validate-key endpoint at {base_url}/debug/validate-key")
        headers = {"X-API-Key": api_key}
        response = requests.get(f"{base_url}/debug/validate-key", headers=headers)
        
        # Check the response
        if response.status_code == 200:
            logger.info(f"Debug/validate-key endpoint returned status code {response.status_code}")
            logger.info(f"Response: {response.json()}")
            return True
        else:
            logger.error(f"Debug/validate-key endpoint returned status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error testing debug/validate-key endpoint: {str(e)}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting Flask app tests")
    
    # Test health endpoint
    health_result = test_health_endpoint()
    logger.info(f"Health endpoint test {'passed' if health_result else 'failed'}")
    
    # Test debug/validate-key endpoint
    if os.environ.get("TEST_API_KEY"):
        validate_key_result = test_debug_validate_key()
        logger.info(f"Debug/validate-key endpoint test {'passed' if validate_key_result else 'failed'}")
    else:
        logger.warning("Skipping debug/validate-key test - TEST_API_KEY not set")
    
    logger.info("Flask app tests completed")

if __name__ == "__main__":
    main() 