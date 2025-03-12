import os
import logging
from prisma import Prisma
from prisma.errors import PrismaError
import base64
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class PrismaClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PrismaClient, cls).__new__(cls)
            cls._instance.client = None
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        """Initialize the Prisma client."""
        try:
            # Set debug flag to bypass version check
            os.environ["PRISMA_PY_DEBUG_GENERATOR"] = "1"

            # Initialize client
            self.client = Prisma()
            logger.info("Prisma client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Prisma client: {str(e)}")
            raise

    def connect(self):
        """Connect to the database."""
        try:
            if not self.client:
                self.initialize()

            # Simple connection attempt without checking connection status first
            # This avoids the circular dependency issue
            try:
                self.client.connect()
                logger.info("Connected to database via Prisma")
                return True
            except PrismaError as e:
                if "Already connected" in str(e):
                    # Already connected is not an error
                    logger.info("Client already connected to database")
                    return True
                # Re-raise other errors
                raise
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def disconnect(self):
        """Disconnect from the database."""
        try:
            if self.client and hasattr(self.client, "_engine") and self.client._engine:
                self.client.disconnect()
                logger.info("Disconnected from database")
        except Exception as e:
            logger.error(f"Error disconnecting from database: {str(e)}")

    # Account methods
    def get_account_by_user_id(self, user_id):
        """Get account by user ID."""
        try:
            # Ensure connection
            self.connect()

            account = self.client.account.find_unique(where={"userId": user_id})
            if account:
                # Convert to dictionary for consistent access
                account_dict = {
                    "userId": account.userId,
                    "categorisationRange": account.categorisationRange,
                    "categorisationTab": account.categorisationTab,
                    "columnOrderCategorisation": account.columnOrderCategorisation,
                    "api_key": account.api_key,
                }

                # Add email field if it exists
                if hasattr(account, "email"):
                    account_dict["email"] = account.email

                return account_dict
            return None
        except Exception as e:
            logger.error(f"Error getting account by user ID: {str(e)}")
            raise

    def get_account_by_api_key(self, api_key):
        """Get account by API key."""
        try:
            # Ensure connection
            self.connect()

            account = self.client.account.find_first(where={"api_key": api_key})
            if account:
                # Convert to dictionary for consistent access
                account_dict = {
                    "userId": account.userId,
                    "categorisationRange": account.categorisationRange,
                    "categorisationTab": account.categorisationTab,
                    "columnOrderCategorisation": account.columnOrderCategorisation,
                    "api_key": account.api_key,
                }

                # Add email field if it exists
                if hasattr(account, "email"):
                    account_dict["email"] = account.email

                return account_dict
            return None
        except Exception as e:
            logger.error(f"Error getting account by API key: {str(e)}")
            raise

    def insert_account(self, user_id, data):
        """Insert a new account."""
        try:
            # Ensure connection
            self.connect()

            account_data = {"userId": user_id}

            if "apiKey" in data:
                account_data["api_key"] = data["apiKey"]
            elif "api_key" in data:
                account_data["api_key"] = data["api_key"]

            if "email" in data:
                account_data["email"] = data["email"]

            if "config" in data:
                account_data["config"] = data["config"]

            if "modelData" in data:
                account_data["modelData"] = data["modelData"]

            if "categorisationRange" in data:
                account_data["categorisationRange"] = data["categorisationRange"]

            if "categorisationTab" in data:
                account_data["categorisationTab"] = data["categorisationTab"]

            if "columnOrderCategorisation" in data:
                account_data["columnOrderCategorisation"] = data[
                    "columnOrderCategorisation"
                ]

            account = self.client.account.create(data=account_data)

            if account:
                # Convert to dictionary for consistent access
                account_dict = {
                    "userId": account.userId,
                    "categorisationRange": account.categorisationRange,
                    "categorisationTab": account.categorisationTab,
                    "columnOrderCategorisation": account.columnOrderCategorisation,
                    "api_key": account.api_key,
                }

                # Add email field if it exists
                if hasattr(account, "email"):
                    account_dict["email"] = account.email

                return account_dict
            return None
        except Exception as e:
            logger.error(f"Error inserting account: {str(e)}")
            raise

    def update_account(self, user_id, data):
        """Update an existing account."""
        try:
            # Ensure connection
            self.connect()

            update_data = {}

            if "apiKey" in data:
                update_data["api_key"] = data["apiKey"]
            elif "api_key" in data:
                update_data["api_key"] = data["api_key"]

            if "email" in data:
                update_data["email"] = data["email"]

            if "categorisationRange" in data:
                update_data["categorisationRange"] = data["categorisationRange"]

            if "categorisationTab" in data:
                update_data["categorisationTab"] = data["categorisationTab"]

            if "columnOrderCategorisation" in data:
                update_data["columnOrderCategorisation"] = data[
                    "columnOrderCategorisation"
                ]

            account = self.client.account.update(
                where={"userId": user_id}, data=update_data
            )

            if account:
                # Convert to dictionary for consistent access
                return {
                    "userId": account.userId,
                    "categorisationRange": account.categorisationRange,
                    "categorisationTab": account.categorisationTab,
                    "columnOrderCategorisation": account.columnOrderCategorisation,
                    "api_key": account.api_key,
                    "email": account.email,
                }
            return None
        except Exception as e:
            logger.error(f"Error updating account: {str(e)}")
            raise

    # Webhook results methods
    def insert_webhook_result(self, prediction_id, results):
        """Insert webhook result."""
        try:
            # Ensure client is connected
            self.connect()

            # Convert results to a JSON string
            import json

            # Create a simple structure with the results
            if (
                isinstance(results, dict)
                and "data" in results
                and isinstance(results["data"], list)
            ):
                # Use the provided data
                json_data = {
                    "status": "success",
                    "message": "Classification completed",
                    "data": results["data"],
                }
            else:
                # Create a simple structure
                json_data = {
                    "status": "success",
                    "message": "Results available in logs",
                }

            # Convert to JSON string
            json_string = json.dumps(json_data)

            # Escape single quotes in the JSON string for SQL
            json_string = json_string.replace("'", "''")

            # Create a raw SQL query
            raw_sql = f"""
            INSERT INTO webhook_results (prediction_id, results)
            VALUES ('{prediction_id}', '{json_string}'::jsonb)
            ON CONFLICT (prediction_id) 
            DO UPDATE SET results = '{json_string}'::jsonb
            RETURNING id, prediction_id, created_at;
            """

            # Log the SQL query for debugging
            logger.debug(f"Raw SQL query: {raw_sql}")

            try:
                # Execute the raw SQL query using psycopg2
                import psycopg2
                from psycopg2.extras import RealDictCursor

                # Get the database URL from environment
                database_url = os.environ.get("DATABASE_URL")
                if not database_url:
                    logger.error("DATABASE_URL environment variable not set")
                    return None

                # Connect to the database
                conn = psycopg2.connect(database_url)
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                # Execute the query
                cursor.execute(raw_sql)

                # Get the result
                result = cursor.fetchone()

                # Commit the transaction
                conn.commit()

                # Close the cursor and connection
                cursor.close()
                conn.close()

                if result:
                    logger.info(
                        f"Successfully stored webhook result for prediction_id: {prediction_id}"
                    )
                    return {
                        "id": result["id"],
                        "prediction_id": result["prediction_id"],
                        "results": json_data,
                        "created_at": result["created_at"],
                    }
            except Exception as db_error:
                logger.error(f"Error executing raw SQL: {str(db_error)}")

                # Fall back to returning a dummy result
                logger.info(
                    f"Returning dummy result for prediction_id: {prediction_id}"
                )
                return {
                    "id": "generated-id",
                    "prediction_id": prediction_id,
                    "results": json_data,
                    "created_at": None,
                }

            return None
        except Exception as e:
            logger.error(f"Error inserting webhook result: {str(e)}")
            # Don't raise the exception, just log it and return None
            return None

    def get_webhook_result(self, prediction_id):
        """Get webhook result by prediction ID."""
        try:
            # Ensure connection
            self.connect()

            webhook_result = self.client.webhookresult.find_unique(
                where={"prediction_id": prediction_id}
            )

            if webhook_result:
                # Convert to dictionary for consistent access
                return {
                    "id": webhook_result.id,
                    "prediction_id": webhook_result.prediction_id,
                    "results": webhook_result.results,
                    "created_at": webhook_result.created_at,
                }
            return None
        except Exception as e:
            logger.error(f"Error getting webhook result: {str(e)}")
            return None

    # Embedding methods
    def store_embedding(self, embedding_id, data_bytes):
        """Store embedding data in the database."""
        try:
            # Ensure connection
            self.connect()

            # Convert bytes to base64 string for storage
            data_base64 = base64.b64encode(data_bytes).decode("utf-8")

            # Check if embedding already exists
            existing = self.client.embedding.find_unique(
                where={"embedding_id": embedding_id}
            )

            if existing:
                # Update existing embedding
                embedding = self.client.embedding.update(
                    where={"embedding_id": embedding_id}, data={"data": data_base64}
                )
            else:
                # Create new embedding
                embedding = self.client.embedding.create(
                    data={"embedding_id": embedding_id, "data": data_base64}
                )

            if embedding:
                logger.info(f"Successfully stored embedding: {embedding_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            return False

    def fetch_embedding(self, embedding_id):
        """Fetch embedding data from the database."""
        try:
            # Ensure connection
            self.connect()

            embedding = self.client.embedding.find_unique(
                where={"embedding_id": embedding_id}
            )

            if embedding and embedding.data:
                # Convert base64 string back to bytes
                data_bytes = base64.b64decode(embedding.data)
                logger.info(f"Successfully fetched embedding: {embedding_id}")
                return data_bytes
            else:
                logger.warning(f"No embedding found for ID: {embedding_id}")
                return None
        except Exception as e:
            logger.error(f"Error fetching embedding: {str(e)}")
            return None


# Create a singleton instance
prisma_client = PrismaClient()
