import os
import sys
import logging
import unittest
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import from main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the validate_api_key function
from pythonHandler.main import validate_api_key

class TestApiKeyValidation(unittest.TestCase):
    """Test cases for API key validation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test class by connecting to the database."""
        logger.info("Setting up TestApiKeyValidation class")
        try:
            # Test database connection
            cls.conn = psycopg2.connect(
                host=os.environ.get("PGHOST_UNPOOLED"),
                database=os.environ.get("PGDATABASE"),
                user=os.environ.get("PGUSER"),
                password=os.environ.get("PGPASSWORD"),
                sslmode="require",
                connect_timeout=5
            )
            logger.info("Connected to database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    @classmethod
    def tearDownClass(cls):
        """Tear down the test class by disconnecting from the database."""
        logger.info("Tearing down TestApiKeyValidation class")
        try:
            # Close the database connection
            if hasattr(cls, 'conn') and cls.conn:
                cls.conn.close()
                logger.info("Disconnected from database")
        except Exception as e:
            logger.error(f"Failed to disconnect from database: {str(e)}")
    
    def test_valid_api_key(self):
        """Test validation with a valid API key."""
        # Get the API key from the environment
        api_key = os.environ.get("TEST_API_KEY")
        
        if not api_key:
            logger.warning("No TEST_API_KEY environment variable found, skipping test")
            self.skipTest("No TEST_API_KEY environment variable found")
            return
        
        try:
            # Validate the API key
            user_id = validate_api_key(api_key)
            logger.info(f"API key validation successful! User ID: {user_id}")
            self.assertIsNotNone(user_id)
            self.assertTrue(len(user_id) > 0)
        except Exception as e:
            logger.error(f"API key validation failed: {str(e)}")
            self.fail(f"API key validation raised an exception: {str(e)}")
    
    def test_invalid_api_key(self):
        """Test validation with an invalid API key."""
        invalid_api_key = "invalid-api-key"
        
        with self.assertRaises(Exception) as context:
            validate_api_key(invalid_api_key)
        
        logger.info(f"Expected exception raised: {str(context.exception)}")
        self.assertIn("Invalid API key", str(context.exception))
    
    def test_empty_api_key(self):
        """Test validation with an empty API key."""
        empty_api_key = ""
        
        with self.assertRaises(Exception) as context:
            validate_api_key(empty_api_key)
        
        logger.info(f"Expected exception raised: {str(context.exception)}")
        self.assertIn("API key validation failed", str(context.exception))
    
    def test_get_account_by_api_key(self):
        """Test getting an account by API key directly using psycopg2."""
        # Get the API key from the environment
        api_key = os.environ.get("TEST_API_KEY")
        
        if not api_key:
            logger.warning("No TEST_API_KEY environment variable found, skipping test")
            self.skipTest("No TEST_API_KEY environment variable found")
            return
        
        try:
            # Get account by API key using psycopg2
            with self.__class__.conn.cursor() as cur:
                cur.execute('SELECT "userId" FROM "account" WHERE api_key = %s', (api_key,))
                result = cur.fetchone()
                
                logger.info(f"Account retrieval successful! User ID: {result[0] if result else 'None'}")
                
                self.assertIsNotNone(result)
                self.assertIsNotNone(result[0])
        except Exception as e:
            logger.error(f"Account retrieval failed: {str(e)}")
            self.fail(f"Account retrieval raised an exception: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting API key validation tests")
    unittest.main()
    logger.info("API key validation tests complete") 