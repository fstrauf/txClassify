import os
import sys
import logging
from pythonHandler.utils.prisma_client import PrismaClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_prisma_client():
    """Test the PrismaClient class."""
    try:
        logger.info("Testing PrismaClient class...")

        # Create a PrismaClient instance
        logger.info("Creating PrismaClient instance...")
        client = PrismaClient()
        logger.info("PrismaClient instance created successfully")

        # Connect to the database
        logger.info("Connecting to the database...")
        try:
            client.connect()
            logger.info("Connected to database successfully")

            # Try to get an account by API key
            logger.info("Testing get_account_by_api_key method...")
            try:
                # This is just a test, so we're using a fake API key
                account = client.get_account_by_api_key("test-api-key")
                if account:
                    logger.info(f"Found account: {account}")
                else:
                    logger.info(
                        "No account found with the test API key (this is expected)"
                    )
            except Exception as e:
                logger.warning(f"get_account_by_api_key failed: {str(e)}")

            # Disconnect from the database
            client.disconnect()
            logger.info("Disconnected from database successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"Error testing PrismaClient: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_prisma_client()
    if success:
        logger.info("PrismaClient test completed successfully")
        sys.exit(0)
    else:
        logger.error("PrismaClient test failed")
        sys.exit(1)
