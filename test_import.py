import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_import():
    """Test the import issue."""
    try:
        logger.info("Testing import...")
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
        logger.info(f"sys.path: {sys.path}")

        # Try to import the module
        try:
            from pythonHandler.utils.prisma_client import prisma_client

            logger.info("Successfully imported prisma_client using absolute import")
        except ImportError as e:
            logger.error(
                f"Failed to import prisma_client using absolute import: {str(e)}"
            )

        # Try to import using a relative import
        try:
            os.chdir("pythonHandler")
            logger.info(f"Changed directory to: {os.getcwd()}")
            from utils.prisma_client import prisma_client

            logger.info("Successfully imported prisma_client using relative import")
        except ImportError as e:
            logger.error(
                f"Failed to import prisma_client using relative import: {str(e)}"
            )

        return True
    except Exception as e:
        logger.error(f"Error testing import: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_import()
    if success:
        logger.info("Import test completed")
        sys.exit(0)
    else:
        logger.error("Import test failed")
        sys.exit(1)
