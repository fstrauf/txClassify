#!/usr/bin/env python3
"""Test script to verify psycopg2 database connection is working properly."""
import os
from dotenv import load_dotenv
from utils.prisma_client import prisma_client
import logging

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    """Test the database connection."""
    # Load environment variables
    load_dotenv()
    
    # Get database URL
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        logger.error("DATABASE_URL environment variable not found")
        return False
    
    logger.info(f"Using database URL: {database_url[:20]}...")
    
    # Test connection
    try:
        logger.info("Testing database connection...")
        connected = prisma_client.connect()
        if connected:
            logger.info("✅ Connected to database successfully!")
        else:
            logger.error("❌ Failed to connect to database")
            return False
            
        # Test query execution
        logger.info("Testing query execution...")
        try:
            # Simple query to test connection
            result = prisma_client.execute_query("SELECT 1 as test")
            logger.info(f"✅ Query executed successfully: {result}")
            
            # Test account table
            try:
                account_count = prisma_client.execute_query("SELECT COUNT(*) FROM account")
                logger.info(f"✅ Found {account_count[0]['count']} accounts in the database")
            except Exception as e:
                logger.warning(f"⚠️ Could not query account table: {str(e)}")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to execute query: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error connecting to database: {str(e)}")
        return False
    finally:
        # Disconnect from database
        try:
            prisma_client.disconnect()
            logger.info("Disconnected from database")
        except Exception as e:
            logger.error(f"Error disconnecting from database: {str(e)}")

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("✅ Database connection test completed successfully!")
    else:
        logger.error("❌ Database connection test failed!") 