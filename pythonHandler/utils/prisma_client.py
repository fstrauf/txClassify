import os
import logging
from prisma import Prisma
from prisma.errors import PrismaError
import base64
from datetime import datetime
import json

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

    # API Usage Tracking methods
    def track_api_usage(self, api_key):
        """
        Track API key usage by incrementing request count and updating last used timestamp.

        Args:
            api_key: The API key to track usage for

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure connection
            self.connect()

            # Find the account with this API key
            account = self.client.account.find_first(where={"api_key": api_key})
            if not account:
                logger.warning(
                    f"No account found for API key: {api_key[:4]}...{api_key[-4:]}"
                )
                return False

            # Update the account with new request count and timestamp
            self.client.account.update(
                where={"userId": account.userId},
                data={
                    "requestsCount": (
                        account.requestsCount + 1
                        if hasattr(account, "requestsCount")
                        else 1
                    ),
                    "lastUsed": datetime.now(),
                },
            )

            logger.info(f"Tracked API usage for account: {account.userId}")
            return True

        except Exception as e:
            logger.error(f"Error tracking API usage: {str(e)}")
            return False

    def track_embedding_creation(self, api_key, embedding_id):
        """
        Track API key usage for embedding creation and link the embedding to the account.

        Args:
            api_key: The API key used for the request
            embedding_id: The ID of the embedding being created

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure connection
            self.connect()

            # First, track the API usage
            self.track_api_usage(api_key)

            # Find the account with this API key
            account = self.client.account.find_first(where={"api_key": api_key})
            if not account:
                logger.warning(
                    f"No account found for API key: {api_key[:4]}...{api_key[-4:]}"
                )
                return False

            # Update the embedding with the account ID
            embedding = self.client.embedding.find_unique(
                where={"embedding_id": embedding_id}
            )
            if not embedding:
                logger.warning(f"No embedding found with ID: {embedding_id}")
                return False

            self.client.embedding.update(
                where={"embedding_id": embedding_id}, data={"accountId": account.userId}
            )

            logger.info(f"Linked embedding {embedding_id} to account {account.userId}")
            return True

        except Exception as e:
            logger.error(f"Error tracking embedding creation: {str(e)}")
            return False

    def get_account_usage_stats(self, user_id):
        """
        Get usage statistics for an account.

        Args:
            user_id: The user ID to get statistics for

        Returns:
            dict: Dictionary with usage statistics or None if an error occurred
        """
        try:
            # Ensure connection
            self.connect()

            # Get account with usage fields
            account = self.client.account.find_unique(
                where={"userId": user_id}, include={"embeddings": True}
            )

            if not account:
                logger.warning(f"No account found for user ID: {user_id}")
                return None

            # Count embeddings
            embeddings_count = len(account.embeddings) if account.embeddings else 0

            return {
                "user_id": account.userId,
                "email": account.email if hasattr(account, "email") else None,
                "requests_count": (
                    account.requestsCount if hasattr(account, "requestsCount") else 0
                ),
                "last_used": account.lastUsed,
                "embeddings_count": embeddings_count,
            }

        except Exception as e:
            logger.error(f"Error getting account usage stats: {str(e)}")
            return None

    # Webhook results methods
    def insert_webhook_result(self, prediction_id, results):
        """Insert webhook result."""
        try:
            # Ensure client is connected
            self.connect()

            logger.info(f"Inserting webhook result for prediction_id: {prediction_id}")

            # Ensure results is properly JSON serialized
            if isinstance(results, dict) or isinstance(results, list):
                try:
                    # Convert dict/list to json string
                    results_json = json.dumps(results)
                    logger.info(
                        f"Converted dict/list to JSON string for webhook result storage"
                    )
                except Exception as json_error:
                    logger.error(f"Error converting results to JSON: {json_error}")
                    # If JSON conversion fails, try to store a minimal dict with just spreadsheet_id
                    if isinstance(results, dict) and "spreadsheet_id" in results:
                        try:
                            results_json = json.dumps(
                                {"spreadsheet_id": results["spreadsheet_id"]}
                            )
                            logger.info("Created minimal JSON with just spreadsheet_id")
                        except:
                            results_json = json.dumps(
                                {"error": "Failed to convert original data"}
                            )
                    else:
                        results_json = json.dumps(
                            {"error": "Failed to convert original data"}
                        )
            else:
                # If it's already a string, assume it's JSON
                results_json = (
                    results
                    if isinstance(results, str)
                    else json.dumps({"error": "Invalid data type"})
                )

            try:
                # Use Prisma client to create or update webhook result
                webhook_result = self.client.webhookresult.upsert(
                    where={"prediction_id": prediction_id},
                    data={
                        "create": {
                            "prediction_id": prediction_id,
                            "results": results_json,
                        },
                        "update": {"results": results_json},
                    },
                )

                if webhook_result:
                    logger.info(
                        f"Successfully stored webhook result for prediction_id: {prediction_id}"
                    )
                    return {
                        "id": webhook_result.id,
                        "prediction_id": webhook_result.prediction_id,
                        "results": results,
                        "created_at": webhook_result.created_at,
                    }
                return None

            except Exception as prisma_error:
                logger.error(
                    f"Error using Prisma to insert webhook result: {str(prisma_error)}"
                )

                # If Prisma fails, log the issue
                logger.warning(
                    f"Prisma operation failed, webhook result may not be stored properly"
                )
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
                # Log the raw webhook result for debugging
                logger.info(
                    f"Raw webhook result: id='{webhook_result.id}' prediction_id='{webhook_result.prediction_id}'"
                )

                # Convert to dictionary for consistent access
                result = {
                    "id": webhook_result.id,
                    "prediction_id": webhook_result.prediction_id,
                    "created_at": webhook_result.created_at,
                }

                # Add results field if it exists and is valid
                if hasattr(webhook_result, "results") and webhook_result.results:
                    try:
                        # First check if it's already a dictionary or other object
                        if isinstance(webhook_result.results, (dict, list)):
                            return webhook_result.results

                        # If it's a string, try to parse it as JSON
                        if isinstance(webhook_result.results, str):
                            try:
                                parsed_results = json.loads(webhook_result.results)
                                logger.info(
                                    f"Successfully parsed JSON results from webhook_result"
                                )
                                return parsed_results
                            except json.JSONDecodeError as e:
                                logger.error(f"Error parsing webhook result JSON: {e}")
                                result["results"] = webhook_result.results
                        else:
                            # Non-string, non-dict, just return it as is
                            logger.warning(
                                f"Webhook result is neither string nor dict: {type(webhook_result.results)}"
                            )
                            return webhook_result.results
                    except Exception as e:
                        logger.error(f"Error processing webhook result: {str(e)}")
                        result["results"] = webhook_result.results

                return result
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

            # Convert bytes to base64 string in one go
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
                # Convert base64 string back to bytes in one go
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
