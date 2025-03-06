import os
import logging
from prisma import Prisma
from prisma.errors import PrismaError
import base64

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
            self.client = Prisma()
            logger.info("Prisma client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Prisma client: {str(e)}")
            raise

    def connect(self):
        """Connect to the database."""
        if not self.client:
            self.initialize()
        try:
            self.client.connect()
            logger.info("Connected to database via Prisma")
        except PrismaError as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def disconnect(self):
        """Disconnect from the database."""
        if self.client:
            try:
                self.client.disconnect()
                logger.info("Disconnected from database")
            except PrismaError as e:
                logger.error(f"Error disconnecting from database: {str(e)}")

    # Account methods
    def get_account_by_user_id(self, user_id):
        """Get account by user ID."""
        try:
            # Ensure client is connected
            if not hasattr(self.client, "_engine") or not self.client._engine:
                self.connect()

            account = self.client.account.find_unique(where={"userId": user_id})
            if account:
                # Convert to dictionary for consistent access
                return {
                    "userId": account.userId,
                    "categorisationRange": account.categorisationRange,
                    "categorisationTab": account.categorisationTab,
                    "columnOrderCategorisation": account.columnOrderCategorisation,
                    "api_key": account.api_key,
                }
            return None
        except PrismaError as e:
            logger.error(f"Error getting account by user ID: {str(e)}")
            raise

    def get_account_by_api_key(self, api_key):
        """Get account by API key."""
        try:
            # Ensure client is connected
            if not hasattr(self.client, "_engine") or not self.client._engine:
                self.connect()

            account = self.client.account.find_first(where={"api_key": api_key})
            if account:
                # Convert to dictionary for consistent access
                return {
                    "userId": account.userId,
                    "categorisationRange": account.categorisationRange,
                    "categorisationTab": account.categorisationTab,
                    "columnOrderCategorisation": account.columnOrderCategorisation,
                    "api_key": account.api_key,
                }
            return None
        except PrismaError as e:
            logger.error(f"Error getting account by API key: {str(e)}")
            raise

    def insert_account(self, user_id, data):
        """Insert a new account."""
        try:
            # Ensure client is connected
            if not hasattr(self.client, "_engine") or not self.client._engine:
                self.connect()

            account_data = {"userId": user_id}

            if "apiKey" in data:
                account_data["api_key"] = data["apiKey"]
            elif "api_key" in data:
                account_data["api_key"] = data["api_key"]

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
                return {
                    "userId": account.userId,
                    "categorisationRange": account.categorisationRange,
                    "categorisationTab": account.categorisationTab,
                    "columnOrderCategorisation": account.columnOrderCategorisation,
                    "api_key": account.api_key,
                }
            return None
        except PrismaError as e:
            logger.error(f"Error inserting account: {str(e)}")
            raise

    def update_account(self, user_id, data):
        """Update an existing account."""
        try:
            # Ensure client is connected
            if not hasattr(self.client, "_engine") or not self.client._engine:
                self.connect()

            update_data = {}

            if "apiKey" in data:
                update_data["api_key"] = data["apiKey"]
            elif "api_key" in data:
                update_data["api_key"] = data["api_key"]

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
                }
            return None
        except PrismaError as e:
            logger.error(f"Error updating account: {str(e)}")
            raise

    # Webhook results methods
    def insert_webhook_result(self, prediction_id, results):
        """Insert webhook result."""
        try:
            # Ensure client is connected
            if not hasattr(self.client, "_engine") or not self.client._engine:
                self.connect()

            webhook_result = self.client.webhookresult.create(
                data={"prediction_id": prediction_id, "results": results}
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
        except PrismaError as e:
            logger.error(f"Error inserting webhook result: {str(e)}")
            raise

    def get_webhook_result(self, prediction_id):
        """Get webhook result by prediction ID."""
        try:
            # Ensure client is connected
            if not hasattr(self.client, "_engine") or not self.client._engine:
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
        except PrismaError as e:
            logger.error(f"Error getting webhook result: {str(e)}")
            raise

    # Embedding methods
    def store_embedding(self, file_name, data_bytes):
        """Store embedding data in the database."""
        try:
            # Ensure client is connected
            if not hasattr(self.client, "_engine") or not self.client._engine:
                self.connect()

            # Convert bytes to base64 string for storage
            data_base64 = base64.b64encode(data_bytes).decode("utf-8")

            # Check if embedding already exists
            existing = self.client.embedding.find_unique(where={"file_name": file_name})

            if existing:
                # Update existing embedding
                embedding = self.client.embedding.update(
                    where={"file_name": file_name}, data={"data": data_base64}
                )
            else:
                # Create new embedding
                embedding = self.client.embedding.create(
                    data={"file_name": file_name, "data": data_base64}
                )

            if embedding:
                logger.info(f"Successfully stored embedding: {file_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            raise

    def fetch_embedding(self, file_name):
        """Fetch embedding data from the database."""
        try:
            # Ensure client is connected
            if not hasattr(self.client, "_engine") or not self.client._engine:
                self.connect()

            embedding = self.client.embedding.find_unique(
                where={"file_name": file_name}
            )

            if embedding and embedding.data:
                # Convert base64 string back to bytes
                data_bytes = base64.b64decode(embedding.data)
                logger.info(f"Successfully fetched embedding: {file_name}")
                return data_bytes
            logger.warning(f"No embedding found for file: {file_name}")
            return None
        except Exception as e:
            logger.error(f"Error fetching embedding: {str(e)}")
            raise


# Create a singleton instance
prisma_client = PrismaClient()
