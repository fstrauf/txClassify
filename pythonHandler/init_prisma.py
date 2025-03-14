import os
import sys
import logging
import subprocess
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

        # First, check if Node.js and Prisma CLI are available
        logger.info("Checking if Node.js and Prisma CLI are available...")
        node_result = subprocess.run(
            ["node", "--version"], capture_output=True, text=True
        )
        if node_result.returncode != 0:
            logger.error("Node.js is not available. Please install Node.js.")
            return False
        logger.info(f"Node.js version: {node_result.stdout.strip()}")

        prisma_result = subprocess.run(
            ["prisma", "--version"], capture_output=True, text=True
        )
        if prisma_result.returncode != 0:
            logger.error("Prisma CLI is not available. Please install Prisma CLI.")
            return False
        logger.info(f"Prisma CLI version: {prisma_result.stdout.strip()}")

        # Generate the Prisma client using the Prisma CLI directly
        logger.info("Generating Prisma client using Prisma CLI...")
        cli_result = subprocess.run(
            ["prisma", "generate", f"--schema={schema_path}"],
            capture_output=True,
            text=True,
        )

        if cli_result.returncode != 0:
            logger.error(
                f"Failed to generate Prisma client using CLI: {cli_result.stderr}"
            )
            return False

        logger.info(f"Prisma CLI output: {cli_result.stdout}")

        # Now generate using the Python module
        logger.info("Generating Prisma client using Python module...")
        py_result = subprocess.run(
            ["python", "-m", "prisma", "generate", f"--schema={schema_path}"],
            capture_output=True,
            text=True,
        )

        if py_result.returncode != 0:
            logger.error(
                f"Failed to generate Prisma client using Python module: {py_result.stderr}"
            )
            return False

        logger.info(f"Python module output: {py_result.stdout}")

        # Test if the client works
        logger.info("Testing Prisma client...")
        try:
            from prisma import Prisma

            logger.info("Prisma module imported successfully")

            client = Prisma()
            logger.info("Prisma client created successfully")

            client.connect()
            logger.info("Connected to database successfully")
            client.disconnect()
            logger.info("Disconnected from database successfully")
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
