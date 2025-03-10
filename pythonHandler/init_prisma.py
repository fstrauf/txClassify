import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_prisma():
    """Initialize Prisma following the documentation."""
    try:
        logger.info("Initializing Prisma...")

        # Get the absolute path to the schema file
        schema_path = Path(__file__).parent.parent / "prisma" / "schema.prisma"
        logger.info(f"Schema path: {schema_path}")

        # Check if schema file exists
        if not schema_path.exists():
            logger.error(f"Schema file not found at {schema_path}")
            return False

        # Generate the Prisma client
        logger.info("Generating Prisma client...")
        os.system(f"python -m prisma generate --schema={schema_path}")

        # Test if the client works
        logger.info("Testing Prisma client...")
        try:
            from prisma import Prisma

            client = Prisma()
            client.connect()
            logger.info("Prisma client connected successfully")
            client.disconnect()
            logger.info("Prisma client disconnected successfully")
            return True
        except Exception as e:
            logger.error(f"Error testing Prisma client: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"Error initializing Prisma: {str(e)}")
        return False


if __name__ == "__main__":
    success = init_prisma()
    if success:
        logger.info("Prisma initialized successfully")
        sys.exit(0)
    else:
        logger.error("Prisma initialization failed")
        sys.exit(1)
