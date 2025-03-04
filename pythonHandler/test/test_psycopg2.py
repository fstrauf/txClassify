import os
import logging
import psycopg2
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger('test_psycopg2')

# Load environment variables
load_dotenv()

def main():
    logger.info("Testing direct PostgreSQL connection with psycopg2")
    
    # Get database URL from environment
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not found")
        return
    
    logger.info(f"Using DATABASE_URL: {database_url[:20]}...")
    
    try:
        # Connect to the database
        logger.info("Connecting to database...")
        conn = psycopg2.connect(database_url)
        logger.info("Connected to database successfully!")
        
        # Create a cursor
        cursor = conn.cursor()
        
        # Test query - get all accounts
        logger.info("Fetching accounts...")
        cursor.execute("SELECT * FROM account LIMIT 5")
        accounts = cursor.fetchall()
        logger.info(f"Found {len(accounts)} accounts")
        
        # Print column names
        column_names = [desc[0] for desc in cursor.description]
        logger.info(f"Account columns: {column_names}")
        
        # Print first account (if any)
        if accounts:
            first_account = accounts[0]
            logger.info(f"First account: {first_account}")
        
        # Test query - get webhook results
        logger.info("Fetching webhook results...")
        cursor.execute("SELECT * FROM webhook_results LIMIT 5")
        webhook_results = cursor.fetchall()
        logger.info(f"Found {len(webhook_results)} webhook results")
        
        # Print column names
        column_names = [desc[0] for desc in cursor.description]
        logger.info(f"Webhook result columns: {column_names}")
        
        # Print first webhook result (if any)
        if webhook_results:
            first_result = webhook_results[0]
            logger.info(f"First webhook result: {first_result}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise
    finally:
        # Close the connection
        if 'conn' in locals() and conn:
            conn.close()
            logger.info("Disconnected from database")

if __name__ == "__main__":
    main() 