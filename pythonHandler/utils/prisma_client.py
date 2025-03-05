import os
import logging
import asyncio
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
    
    async def connect(self):
        """Connect to the database."""
        if not self.client:
            self.initialize()
        try:
            await self.client.connect()
            logger.info("Connected to database via Prisma")
        except PrismaError as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    async def disconnect(self):
        """Disconnect from the database."""
        if self.client:
            try:
                await self.client.disconnect()
                logger.info("Disconnected from database")
            except PrismaError as e:
                logger.error(f"Error disconnecting from database: {str(e)}")
    
    # Synchronous wrapper methods
    def sync_connect(self):
        """Synchronous wrapper for connect."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.connect())
        finally:
            loop.close()
    
    def sync_disconnect(self):
        """Synchronous wrapper for disconnect."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.disconnect())
        finally:
            loop.close()
    
    def sync_get_account_by_user_id(self, user_id):
        """Synchronous wrapper for get_account_by_user_id."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.get_account_by_user_id(user_id))
        finally:
            loop.close()
    
    def sync_get_account_by_api_key(self, api_key):
        """Synchronous wrapper for get_account_by_api_key."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.get_account_by_api_key(api_key))
        finally:
            loop.close()
    
    def sync_insert_account(self, user_id, data):
        """Synchronous wrapper for insert_account."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.insert_account(user_id, data))
        finally:
            loop.close()
    
    def sync_update_account(self, user_id, data):
        """Synchronous wrapper for update_account."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.update_account(user_id, data))
        finally:
            loop.close()
    
    def sync_insert_webhook_result(self, prediction_id, results):
        """Synchronous wrapper for insert_webhook_result."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.insert_webhook_result(prediction_id, results))
        finally:
            loop.close()
    
    def sync_get_webhook_result(self, prediction_id):
        """Synchronous wrapper for get_webhook_result."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.get_webhook_result(prediction_id))
        finally:
            loop.close()
    
    def sync_store_embedding(self, file_name, data_bytes):
        """Synchronous wrapper for store_embedding."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.store_embedding(file_name, data_bytes))
        finally:
            loop.close()
    
    def sync_fetch_embedding(self, file_name):
        """Synchronous wrapper for fetch_embedding."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.fetch_embedding(file_name))
        finally:
            loop.close()
    
    # Account methods
    async def get_account_by_user_id(self, user_id):
        """Get account by user ID."""
        try:
            account = await self.client.account.find_unique(
                where={"userId": user_id}
            )
            if account:
                # Convert to dictionary for consistent access
                return {
                    "userId": account.userId,
                    "categorisationRange": account.categorisationRange,
                    "categorisationTab": account.categorisationTab,
                    "columnOrderCategorisation": account.columnOrderCategorisation,
                    "api_key": account.api_key
                }
            return None
        except PrismaError as e:
            logger.error(f"Error getting account by user ID: {str(e)}")
            raise
    
    async def get_account_by_api_key(self, api_key):
        """Get account by API key."""
        try:
            account = await self.client.account.find_first(
                where={"api_key": api_key}
            )
            if account:
                # Convert to dictionary for consistent access
                return {
                    "userId": account.userId,
                    "categorisationRange": account.categorisationRange,
                    "categorisationTab": account.categorisationTab,
                    "columnOrderCategorisation": account.columnOrderCategorisation,
                    "api_key": account.api_key
                }
            return None
        except PrismaError as e:
            logger.error(f"Error getting account by API key: {str(e)}")
            raise
    
    async def insert_account(self, user_id, data):
        """Insert a new account."""
        try:
            account_data = {
                "userId": user_id
            }
            
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
                account_data["columnOrderCategorisation"] = data["columnOrderCategorisation"]
            
            account = await self.client.account.create(
                data=account_data
            )
            
            if account:
                # Convert to dictionary for consistent access
                return {
                    "userId": account.userId,
                    "categorisationRange": account.categorisationRange,
                    "categorisationTab": account.categorisationTab,
                    "columnOrderCategorisation": account.columnOrderCategorisation,
                    "api_key": account.api_key
                }
            return None
        except PrismaError as e:
            logger.error(f"Error inserting account: {str(e)}")
            raise
    
    async def update_account(self, user_id, data):
        """Update an existing account."""
        try:
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
                update_data["columnOrderCategorisation"] = data["columnOrderCategorisation"]
            
            account = await self.client.account.update(
                where={"userId": user_id},
                data=update_data
            )
            
            if account:
                # Convert to dictionary for consistent access
                return {
                    "userId": account.userId,
                    "categorisationRange": account.categorisationRange,
                    "categorisationTab": account.categorisationTab,
                    "columnOrderCategorisation": account.columnOrderCategorisation,
                    "api_key": account.api_key
                }
            return None
        except PrismaError as e:
            logger.error(f"Error updating account: {str(e)}")
            raise
    
    # Webhook results methods
    async def insert_webhook_result(self, prediction_id, results):
        """Insert webhook result."""
        try:
            webhook_result = await self.client.webhookresult.create(
                data={
                    "prediction_id": prediction_id,
                    "results": results
                }
            )
            
            if webhook_result:
                # Convert to dictionary for consistent access
                return {
                    "id": webhook_result.id,
                    "prediction_id": webhook_result.prediction_id,
                    "results": webhook_result.results,
                    "created_at": webhook_result.created_at
                }
            return None
        except PrismaError as e:
            logger.error(f"Error inserting webhook result: {str(e)}")
            raise
    
    async def get_webhook_result(self, prediction_id):
        """Get webhook result by prediction ID."""
        try:
            webhook_result = await self.client.webhookresult.find_unique(
                where={"prediction_id": prediction_id}
            )
            
            if webhook_result:
                # Convert to dictionary for consistent access
                return {
                    "id": webhook_result.id,
                    "prediction_id": webhook_result.prediction_id,
                    "results": webhook_result.results,
                    "created_at": webhook_result.created_at
                }
            return None
        except PrismaError as e:
            logger.error(f"Error getting webhook result: {str(e)}")
            raise
            
    # Embedding methods
    async def store_embedding(self, file_name, data_bytes):
        """Store embedding data in the database."""
        try:
            # Convert bytes to base64 string for storage
            data_base64 = base64.b64encode(data_bytes).decode('utf-8')
            
            # Check if embedding already exists
            existing = await self.client.embedding.find_unique(
                where={"file_name": file_name}
            )
            
            if existing:
                # Update existing embedding
                embedding = await self.client.embedding.update(
                    where={"file_name": file_name},
                    data={"data": data_base64}
                )
            else:
                # Create new embedding
                embedding = await self.client.embedding.create(
                    data={
                        "file_name": file_name,
                        "data": data_base64
                    }
                )
            
            if embedding:
                logger.info(f"Successfully stored embedding: {file_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            raise
    
    async def fetch_embedding(self, file_name):
        """Fetch embedding data from the database."""
        try:
            embedding = await self.client.embedding.find_unique(
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