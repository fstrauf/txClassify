import os
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

logger = logging.getLogger(__name__)


class Database:
    def __init__(self):
        self.conn = None
        self.cursor = None
        self.connect()

    def connect(self):
        """Connect to the PostgreSQL database."""
        try:
            database_url = os.environ.get("DATABASE_URL")
            if not database_url:
                raise ValueError("DATABASE_URL environment variable not found")

            logger.info(f"Connecting to database with URL: {database_url[:20]}...")
            self.conn = psycopg2.connect(database_url)
            logger.info("Connected to database successfully")
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise

    def disconnect(self):
        """Disconnect from the database."""
        if self.conn:
            try:
                self.conn.close()
                logger.info("Disconnected from database")
            except Exception as e:
                logger.error(f"Error disconnecting from database: {str(e)}")

    def execute_query(self, query, params=None, fetch=True):
        """Execute a SQL query with optional parameters."""
        try:
            # Create a new cursor for each query
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)

            # Execute the query
            cursor.execute(query, params)

            if fetch:
                # Fetch results
                results = cursor.fetchall()
                cursor.close()
                return results
            else:
                # Commit changes
                self.conn.commit()
                cursor.close()
                return True
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            self.conn.rollback()
            raise

    def get_account_by_user_id(self, user_id):
        """Get account by user ID."""
        query = 'SELECT * FROM "accounts" WHERE "userId" = %s'
        results = self.execute_query(query, (user_id,))
        return results[0] if results else None

    def get_user_by_api_key(self, api_key):
        """Get user by API key."""
        query = 'SELECT * FROM "users" WHERE api_key = %s'
        results = self.execute_query(query, (api_key,))
        return results[0] if results else None

    def get_account_by_api_key(self, api_key):
        """Get account by API key (deprecated, use get_user_by_api_key)."""
        # For backward compatibility
        logger.warning(
            "get_account_by_api_key is deprecated, use get_user_by_api_key instead"
        )
        user = self.get_user_by_api_key(api_key)
        if not user:
            return None
        return self.get_account_by_user_id(user["id"])

    def insert_account(self, user_id, data):
        """Insert a new account."""
        query = """
            INSERT INTO "accounts" ("userId", "type", "provider", "providerAccountId", "categorisationRange", "categorisationTab", "columnOrderCategorisation", "created_at")
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
        """

        params = (
            user_id,
            data.get("type", "oauth"),
            data.get("provider", "email"),
            data.get("providerAccountId", data.get("email")),
            data.get("categorisationRange"),
            data.get("categorisationTab"),
            (
                json.dumps(data.get("columnOrderCategorisation"))
                if data.get("columnOrderCategorisation")
                else None
            ),
            datetime.now(),
        )

        results = self.execute_query(query, params)
        return results[0] if results else None

    def update_account(self, account_id, data):
        """Update an existing account."""
        # Build the SET clause dynamically based on provided data
        set_clauses = []
        params = []

        if "categorisationRange" in data:
            set_clauses.append('"categorisationRange" = %s')
            params.append(data["categorisationRange"])

        if "categorisationTab" in data:
            set_clauses.append('"categorisationTab" = %s')
            params.append(data["categorisationTab"])

        if "columnOrderCategorisation" in data:
            set_clauses.append('"columnOrderCategorisation" = %s')
            params.append(json.dumps(data["columnOrderCategorisation"]))

        if "requestsCount" in data:
            set_clauses.append('"requestsCount" = %s')
            params.append(data["requestsCount"])

        if "lastUsed" in data:
            set_clauses.append('"lastUsed" = %s')
            params.append(data["lastUsed"])

        if not set_clauses:
            logger.warning("No data provided for update")
            return None

        # Add the WHERE clause parameter
        params.append(account_id)

        query = f"""
            UPDATE "accounts"
            SET {", ".join(set_clauses)}
            WHERE id = %s
            RETURNING *
        """

        results = self.execute_query(query, params)
        return results[0] if results else None

    def insert_webhook_result(self, prediction_id, results):
        """Insert webhook result."""
        query = """
            INSERT INTO webhook_results (prediction_id, results)
            VALUES (%s, %s)
            RETURNING *
        """

        params = (prediction_id, json.dumps(results))

        results = self.execute_query(query, params)
        return results[0] if results else None

    def get_webhook_result(self, prediction_id):
        """Get webhook result by prediction ID."""
        query = "SELECT * FROM webhook_results WHERE prediction_id = %s"
        results = self.execute_query(query, (prediction_id,))
        return results[0] if results else None


# Create a singleton instance
db = Database()
