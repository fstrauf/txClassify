import os
import psycopg2
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get database connection string from environment
DATABASE_URL = os.getenv("DATABASE_URL")


def test_connection():
    """Test database connection and check table existence"""
    if not DATABASE_URL:
        logger.error("DATABASE_URL environment variable not set")
        return False

    try:
        # Connect to database
        logger.info(f"Connecting to database with URL: {DATABASE_URL[:10]}...")
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        logger.info("Connection successful")

        # Check tables
        tables = ["account", "embedding", "webhook_results"]
        for table in tables:
            cursor.execute(
                f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public'
                AND table_name = '{table}'
            );
            """
            )
            exists = cursor.fetchone()[0]
            logger.info(f"Table '{table}' exists: {exists}")

            if exists:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"  - {table} has {count} rows")

        # Close connection
        cursor.close()
        conn.close()

        return True
    except Exception as e:
        logger.error(f"Error testing database connection: {str(e)}")
        return False


if __name__ == "__main__":
    if test_connection():
        logger.info("Database connection test completed successfully")
    else:
        logger.error("Database connection test failed")
