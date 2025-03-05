#!/usr/bin/env python3
import os
import sys
import logging
import uuid
import secrets
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import the main module
script_dir = os.path.dirname(os.path.abspath(__file__))
pythonHandler_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(pythonHandler_dir)

# Add both the pythonHandler directory and the project root to the path
sys.path.insert(0, pythonHandler_dir)
sys.path.insert(0, project_root)

# Load environment variables from .env file
load_dotenv()

def setup_test_account():
    """Set up a test account in the database."""
    try:
        # Import the prisma client
        from pythonHandler.utils.prisma_client import prisma_client
        
        # Connect to the database
        logger.info("Connecting to the database...")
        prisma_client.connect()
        
        # Generate a test user ID
        test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
        
        # Generate a test API key
        test_api_key = f"test_key_{secrets.token_hex(16)}"
        
        # Check if the test account already exists
        existing_account = prisma_client.account.find_first(
            where={"userId": test_user_id}
        )
        
        if existing_account:
            logger.info(f"Test account already exists with ID: {existing_account.id}")
            
            # Check if the API key exists
            existing_api_key = prisma_client.api_key.find_first(
                where={"accountId": existing_account.id}
            )
            
            if existing_api_key:
                logger.info(f"Test API key already exists: {existing_api_key.key}")
                test_api_key = existing_api_key.key
            else:
                # Create a new API key for the existing account
                new_api_key = prisma_client.api_key.create(
                    data={
                        "key": test_api_key,
                        "accountId": existing_account.id
                    }
                )
                logger.info(f"Created new API key for existing account: {new_api_key.key}")
        else:
            # Create a new test account
            new_account = prisma_client.account.create(
                data={
                    "userId": test_user_id,
                    "email": f"{test_user_id}@example.com",
                    "api_keys": {
                        "create": {
                            "key": test_api_key
                        }
                    }
                }
            )
            logger.info(f"Created new test account with ID: {new_account.id}")
            logger.info(f"Created new API key: {test_api_key}")
        
        # Disconnect from the database
        prisma_client.disconnect()
        
        # Write the test account details to a file
        test_env_file = os.path.join(pythonHandler_dir, '.test.env')
        with open(test_env_file, 'w') as f:
            f.write(f"TEST_USER_ID={test_user_id}\n")
            f.write(f"TEST_API_KEY={test_api_key}\n")
        
        logger.info(f"Test account details written to {test_env_file}")
        
        # Return the test account details
        return {
            "user_id": test_user_id,
            "api_key": test_api_key
        }
    except Exception as e:
        logger.error(f"Error setting up test account: {e}")
        return None

if __name__ == "__main__":
    test_account = setup_test_account()
    if test_account:
        logger.info(f"Test account set up successfully:")
        logger.info(f"User ID: {test_account['user_id']}")
        logger.info(f"API Key: {test_account['api_key']}")
    else:
        logger.error("Failed to set up test account") 