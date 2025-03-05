#!/usr/bin/env python3
import os
import sys
import unittest
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the test suite with proper environment setup."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Add the current directory to the path
    sys.path.append(script_dir)
    
    # Set test environment variables if not already set
    if not os.environ.get("BACKEND_API"):
        os.environ["BACKEND_API"] = "http://localhost:5001"
    
    if not os.environ.get("FLASK_ENV"):
        os.environ["FLASK_ENV"] = "testing"
    
    # Discover and run tests
    logger.info("Discovering tests...")
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('test', pattern='test_*.py')
    
    logger.info(f"Found {test_suite.countTestCases()} test cases")
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return non-zero exit code if tests failed
    if not result.wasSuccessful():
        logger.error("Tests failed")
        return 1
    
    logger.info("All tests passed")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 