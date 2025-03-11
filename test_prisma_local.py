import os
import sys
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_prisma_local():
    """Test Prisma setup locally."""
    try:
        logger.info("Testing Prisma setup locally...")

        # Step 1: Check if prisma package is installed
        logger.info("Checking if prisma package is installed...")
        try:
            import prisma

            logger.info(f"Prisma package is installed. Version: {prisma.__version__}")
        except ImportError:
            logger.error("Prisma package is not installed. Installing...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "prisma==0.15.0"], check=True
            )
            logger.info("Prisma package installed.")

        # Step 2: Check if Node.js is installed
        logger.info("Checking if Node.js is installed...")
        node_result = subprocess.run(
            ["node", "--version"], capture_output=True, text=True
        )
        if node_result.returncode != 0:
            logger.error("Node.js is not installed. Please install Node.js.")
            return False
        logger.info(f"Node.js version: {node_result.stdout.strip()}")

        # Step 3: Install the correct Prisma CLI version
        logger.info("Installing Prisma CLI version 5.17.0...")
        subprocess.run(["npm", "install", "-g", "prisma@5.17.0"], check=True)

        # Verify Prisma CLI version
        prisma_result = subprocess.run(
            ["npx", "prisma@5.17.0", "--version"], capture_output=True, text=True
        )
        if prisma_result.returncode != 0:
            logger.error("Failed to verify Prisma CLI version.")
            return False
        logger.info(f"Prisma CLI version: {prisma_result.stdout.strip()}")

        # Step 4: Get the schema path
        schema_path = Path("prisma/schema.prisma")
        if not schema_path.exists():
            logger.error(f"Schema file not found at {schema_path}")
            return False
        logger.info(f"Schema path: {schema_path.absolute()}")

        # Step 5: Generate Prisma client using CLI with the correct version
        logger.info("Generating Prisma client using CLI...")
        cli_result = subprocess.run(
            ["npx", "prisma@5.17.0", "generate", f"--schema={schema_path}"],
            capture_output=True,
            text=True,
        )

        if cli_result.returncode != 0:
            logger.error(
                f"Failed to generate Prisma client using CLI: {cli_result.stderr}"
            )
            return False

        logger.info(f"Prisma CLI output: {cli_result.stdout}")

        # Step 6: Generate Prisma client using Python module
        logger.info("Generating Prisma client using Python module...")
        # Set the environment variable to debug the generator
        os.environ["PRISMA_PY_DEBUG_GENERATOR"] = "1"

        py_result = subprocess.run(
            [sys.executable, "-m", "prisma", "generate", f"--schema={schema_path}"],
            capture_output=True,
            text=True,
        )

        if py_result.returncode != 0:
            logger.error(
                f"Failed to generate Prisma client using Python module: {py_result.stderr}"
            )
            return False

        logger.info(f"Python module output: {py_result.stdout}")

        # Step 7: Test if the client works
        logger.info("Testing Prisma client...")
        try:
            from prisma import Prisma

            client = Prisma()
            logger.info("Prisma client created successfully")

            # Check if DATABASE_URL is set
            if not os.environ.get("DATABASE_URL") and not os.environ.get("DIRECT_URL"):
                logger.warning(
                    "DATABASE_URL or DIRECT_URL environment variable not set. Cannot test connection."
                )
                logger.info(
                    "Prisma client creation successful, but connection not tested."
                )
                return True

            # Try to connect
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
            return True
        except Exception as e:
            logger.error(f"Error testing Prisma client: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"Error testing Prisma locally: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_prisma_local()
    if success:
        logger.info("Prisma local test completed successfully")
        sys.exit(0)
    else:
        logger.error("Prisma local test failed")
        sys.exit(1)
