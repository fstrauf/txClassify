import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_prisma():
    """Test if Prisma works correctly."""
    try:
        logger.info("Testing Prisma...")

        # Import Prisma
        from prisma import Prisma

        logger.info("Prisma module imported successfully")

        # Create a Prisma client
        client = Prisma()
        logger.info("Prisma client created successfully")

        # Try to connect to the database
        if os.environ.get("DATABASE_URL") or os.environ.get("DIRECT_URL"):
            client.connect()
            logger.info("Connected to database successfully")

            # Try a simple query
            try:
                # Try to get all accounts
                accounts = client.account.find_many()
                logger.info(f"Found {len(accounts)} accounts")
            except Exception as e:
                logger.warning(f"Query failed: {str(e)}")

            client.disconnect()
            logger.info("Disconnected from database successfully")
        else:
            logger.warning(
                "DATABASE_URL or DIRECT_URL not set, skipping connection test"
            )

        return True
    except Exception as e:
        logger.error(f"Error testing Prisma: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_prisma()
    if success:
        logger.info("Prisma test completed successfully")
        sys.exit(0)
    else:
        logger.error("Prisma test failed")
        sys.exit(1)
