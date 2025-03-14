import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_prisma_client():
    """Generate the Prisma client."""
    try:
        logger.info("Generating Prisma client...")
        schema_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "prisma", "schema.prisma")
        )
        logger.info(f"Schema path: {schema_path}")

        # Check if schema file exists
        if not os.path.exists(schema_path):
            logger.error(f"Schema file not found at {schema_path}")
            sys.exit(1)

        # Generate the client
        cmd = f"prisma generate --schema={schema_path}"
        logger.info(f"Running command: {cmd}")
        result = os.system(cmd)

        if result != 0:
            logger.error(f"Failed to generate Prisma client. Exit code: {result}")
            sys.exit(1)

        logger.info("Prisma client generated successfully")
    except Exception as e:
        logger.error(f"Error generating Prisma client: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    generate_prisma_client()
