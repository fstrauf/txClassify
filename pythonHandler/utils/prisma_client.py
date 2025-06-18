import os
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import base64
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class DBClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DBClient, cls).__new__(cls)
            cls._instance.conn = None
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        """Initialize the database connection."""
        try:
            logger.info("Initializing database client")
            self.connect()
        except Exception as e:
            logger.error(f"Failed to initialize database client: {str(e)}")
            raise

    def connect(self):
        """Connect to the database."""
        try:
            if self.conn:
                # Try a simple query to check if connection is still alive
                try:
                    cursor = self.conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                    return True
                except Exception:
                    # Connection is dead, close it to be sure
                    try:
                        self.conn.close()
                    except:
                        pass
                    self.conn = None

            # Connect to the database
            database_url = os.environ.get("DATABASE_URL")
            if not database_url:
                raise ValueError("DATABASE_URL environment variable not found")

            logger.info(f"Connecting to database with URL: {database_url[:20]}...")
            self.conn = psycopg2.connect(database_url)
            logger.info("Connected to database successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def disconnect(self):
        """Disconnect from the database."""
        try:
            if self.conn:
                self.conn.close()
                self.conn = None
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

    # User and Account methods
    def get_account_by_user_id(self, user_id):
        """Get account by user ID."""
        try:
            self.connect()
            query = 'SELECT * FROM accounts WHERE "userId" = %s'
            results = self.execute_query(query, (user_id,))
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Error getting account by user ID: {str(e)}")
            raise

    def get_user_by_api_key(self, api_key):
        """Get user by API key."""
        try:
            self.connect()
            query = "SELECT * FROM users WHERE api_key = %s"
            results = self.execute_query(query, (api_key,))
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Error getting user by API key: {str(e)}")
            raise

    def get_account_by_provider_ids(self, provider, provider_account_id):
        """Get account by provider and provider account ID."""
        try:
            self.connect()
            query = 'SELECT * FROM accounts WHERE provider = %s AND "providerAccountId" = %s'
            results = self.execute_query(query, (provider, provider_account_id))
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Error getting account by provider IDs: {str(e)}")
            raise

    def insert_account(self, user_id, data):
        """Insert a new account."""
        try:
            self.connect()
            query = """
                INSERT INTO accounts (
                    "userId", "type", "provider", "providerAccountId", "refresh_token", 
                    "access_token", "expires_at", "token_type", "scope", "id_token", 
                    "session_state", "categorisationRange", "categorisationTab", 
                    "columnOrderCategorisation", "created_at"
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
            """

            params = (
                user_id,
                data.get("type"),
                data.get("provider"),
                data.get("providerAccountId"),
                data.get("refresh_token"),
                data.get("access_token"),
                data.get("expires_at"),
                data.get("token_type"),
                data.get("scope"),
                data.get("id_token"),
                data.get("session_state"),
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
        except Exception as e:
            logger.error(f"Error inserting account: {str(e)}")
            raise

    def update_account(self, account_id, data):
        """Update an existing account."""
        try:
            self.connect()
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

            if "appBetaOptIn" in data:
                set_clauses.append('"appBetaOptIn" = %s')
                params.append(data["appBetaOptIn"])

            if not set_clauses:
                logger.warning("No data provided for update")
                return None

            # Add the WHERE clause parameter
            params.append(account_id)

            query = f"""
                UPDATE accounts
                SET {", ".join(set_clauses)}
                WHERE id = %s
                RETURNING *
            """

            results = self.execute_query(query, params)
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Error updating account: {str(e)}")
            raise

    def update_user(self, user_id, data):
        """Update an existing user."""
        try:
            self.connect()
            # Build the SET clause dynamically based on provided data
            user_set_clauses = []
            user_params = []

            # Handle user table updates
            if "api_key" in data:
                user_set_clauses.append("api_key = %s")
                user_params.append(data["api_key"])

            if "email" in data:
                user_set_clauses.append("email = %s")
                user_params.append(data["email"])

            # Update user table if there are changes
            user_result = None
            if user_set_clauses:
                user_params.append(user_id)
                user_query = f"""
                    UPDATE users
                    SET {", ".join(user_set_clauses)}
                    WHERE id = %s
                    RETURNING *
                """
                user_results = self.execute_query(user_query, user_params)
                user_result = user_results[0] if user_results else None

            # Handle subscription updates (create or update in subscriptions table)
            subscription_data = {}
            if "subscriptionStatus" in data:
                subscription_data["status"] = data["subscriptionStatus"]
            if "subscriptionPlan" in data:
                subscription_data["plan"] = data["subscriptionPlan"]
            if "billingCycle" in data:
                subscription_data["billingCycle"] = data["billingCycle"]

            if subscription_data:
                # Check if user has an existing subscription
                check_query = """
                    SELECT id FROM subscriptions 
                    WHERE "userId" = %s 
                    ORDER BY "createdAt" DESC 
                    LIMIT 1
                """
                existing_subs = self.execute_query(check_query, (user_id,))
                
                if existing_subs:
                    # Update existing subscription
                    sub_set_clauses = []
                    sub_params = []
                    
                    for key, value in subscription_data.items():
                        sub_set_clauses.append(f'"{key}" = %s')
                        sub_params.append(value)
                    
                    sub_params.append(existing_subs[0]["id"])
                    sub_query = f"""
                        UPDATE subscriptions
                        SET {", ".join(sub_set_clauses)}, "updatedAt" = NOW()
                        WHERE id = %s
                        RETURNING *
                    """
                    self.execute_query(sub_query, sub_params)
                else:
                    # Create new subscription with defaults
                    sub_query = """
                        INSERT INTO subscriptions ("userId", status, plan, "billingCycle", "currentPeriodStart", "currentPeriodEnd")
                        VALUES (%s, %s, %s, %s, NOW(), NOW() + INTERVAL '1 month')
                        RETURNING *
                    """
                    self.execute_query(sub_query, (
                        user_id,
                        subscription_data.get("status", "ACTIVE"),
                        subscription_data.get("plan", "FREE"),
                        subscription_data.get("billingCycle", "MONTHLY")
                    ))

            # Return user data if updated, otherwise get current user
            if user_result:
                return user_result
            else:
                get_user_query = "SELECT * FROM users WHERE id = %s"
                user_results = self.execute_query(get_user_query, (user_id,))
                return user_results[0] if user_results else None

        except Exception as e:
            logger.error(f"Error updating user: {str(e)}")
            raise

    def get_user_subscription_status(self, user_id):
        """Get the subscription status for a user."""
        try:
            self.connect()
            # Query the subscriptions table to get the most recent active subscription
            query = '''
                SELECT status FROM subscriptions 
                WHERE "userId" = %s 
                ORDER BY "createdAt" DESC 
                LIMIT 1
            '''
            results = self.execute_query(query, (user_id,))
            if results and results[0].get("status"):
                return results[0]["status"]
            else:
                # Return None if user not found or no subscription found
                logger.warning(f"Subscription status not found for user ID: {user_id}")
                return None
                
        except Exception as e:
            logger.error(
                f"Error getting user subscription status for {user_id}: {str(e)}"
            )
            # For development/testing, if we can't check subscription, allow access
            logger.warning(f"Defaulting to TRIALING status for user {user_id} due to error")
            return "TRIALING"

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
            self.connect()

            # Find the user with this API key
            user = self.get_user_by_api_key(api_key)
            if not user:
                logger.warning(
                    f"No user found for API key: {str(api_key)[:4]}...{str(api_key)[-4:]}"
                )
                return False

            # Find the account for this user
            account = self.get_account_by_user_id(user["id"])
            if not account:
                logger.warning(f"No account found for user ID: {str(user['id'])}")
                return False

            # Calculate new request count
            current_count = account.get("requestsCount", 0)
            new_count = current_count + 1 if current_count is not None else 1

            # Update the account with new request count and timestamp
            query = """
                UPDATE accounts
                SET "requestsCount" = %s, "lastUsed" = %s
                WHERE id = %s
            """

            # Pass parameters with correct types
            self.execute_query(
                query,
                (
                    int(new_count),  # Integer in the database schema
                    datetime.now(),
                    str(account["id"]),
                ),
                fetch=False,
            )

            logger.info(f"Tracked API usage for user: {str(user['id'])}")
            return True

        except Exception as e:
            logger.error(f"Error tracking API usage: {str(e)}")
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
            self.connect()

            # Get user details
            user_query = "SELECT * FROM users WHERE id = %s"
            user_results = self.execute_query(user_query, (user_id,))
            if not user_results:
                logger.warning(f"No user found for user ID: {user_id}")
                return None

            user = user_results[0]

            # Get account
            account = self.get_account_by_user_id(user_id)
            if not account:
                logger.warning(f"No account found for user ID: {user_id}")
                return None

            # Count embeddings
            embedding_query = (
                'SELECT COUNT(*) as count FROM embeddings WHERE "accountId" = %s'
            )
            results = self.execute_query(embedding_query, (account["id"],))
            embeddings_count = results[0]["count"] if results else 0

            # Get subscription status from subscriptions table
            subscription_query = """
                SELECT status, plan FROM subscriptions 
                WHERE "userId" = %s 
                ORDER BY "createdAt" DESC 
                LIMIT 1
            """
            subscription_results = self.execute_query(subscription_query, (user_id,))
            subscription_status = None
            subscription_plan = None
            if subscription_results:
                subscription_status = subscription_results[0].get("status")
                subscription_plan = subscription_results[0].get("plan")

            return {
                "user_id": user_id,
                "email": user.get("email"),
                "requests_count": account.get("requestsCount", 0),
                "last_used": account.get("lastUsed"),
                "embeddings_count": embeddings_count,
                "subscription_plan": subscription_plan,
                "subscription_status": subscription_status,
            }

        except Exception as e:
            logger.error(f"Error getting account usage stats: {str(e)}")
            return None

    # Webhook results methods
    def insert_webhook_result(self, prediction_id, results):
        """Insert webhook result."""
        try:
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

            # First check if the record exists
            check_query = 'SELECT id FROM "webhookResults" WHERE prediction_id = %s'
            existing = self.execute_query(check_query, (prediction_id,))

            if existing:
                # Update existing record
                query = """
                    UPDATE "webhookResults"
                    SET results = %s
                    WHERE prediction_id = %s
                    RETURNING *
                """
                results = self.execute_query(query, (results_json, prediction_id))
            else:
                # Insert new record
                query = """
                    INSERT INTO "webhookResults" (prediction_id, results)
                    VALUES (%s, %s)
                    RETURNING *
                """
                results = self.execute_query(query, (prediction_id, results_json))

            logger.info(
                f"Successfully stored webhook result for prediction_id: {prediction_id}"
            )
            return results[0] if results else None

        except Exception as e:
            logger.error(f"Error inserting webhook result: {str(e)}")
            # Don't raise the exception, just log it and return None
            return None

    def get_webhook_result(self, prediction_id):
        """Get webhook result by prediction ID."""
        try:
            self.connect()

            query = 'SELECT * FROM "webhookResults" WHERE prediction_id = %s'
            results = self.execute_query(query, (prediction_id,))

            if results:
                result = dict(results[0])
                # Log the raw webhook result for debugging
                logger.info(
                    f"Raw webhook result: id='{result['id']}' prediction_id='{result['prediction_id']}'"
                )

                # Add results field if it exists and is valid
                if "results" in result and result["results"]:
                    try:
                        # If it's a string, try to parse it as JSON
                        if isinstance(result["results"], str):
                            try:
                                parsed_results = json.loads(result["results"])
                                logger.info(
                                    f"Successfully parsed JSON results from webhook_result"
                                )
                                return parsed_results
                            except json.JSONDecodeError as e:
                                logger.error(f"Error parsing webhook result JSON: {e}")
                        else:
                            # Non-string, return as is
                            return result["results"]
                    except Exception as e:
                        logger.error(f"Error processing webhook result: {str(e)}")

                return result
            return None
        except Exception as e:
            logger.error(f"Error getting webhook result: {str(e)}")
            return None

    # Embedding methods
    def store_embedding(self, embedding_id, data_bytes, user_id):
        """Store embedding data in the database, including the userId."""
        try:
            self.connect()

            # Convert bytes to base64 string
            data_base64 = base64.b64encode(data_bytes).decode("utf-8")

            # Check if embedding already exists
            check_query = "SELECT embedding_id FROM embeddings WHERE embedding_id = %s"
            existing = self.execute_query(check_query, (embedding_id,))

            if existing:
                # Update existing embedding
                # Ensure userId is also set during update
                query = """
                    UPDATE embeddings
                    SET data = %s, updated_at = %s, "userId" = %s
                    WHERE embedding_id = %s
                """
                self.execute_query(
                    query,
                    (data_base64, datetime.now(), user_id, embedding_id),
                    fetch=False,  # Added user_id
                )
            else:
                # Create new embedding, including userId
                query = """
                    INSERT INTO embeddings (embedding_id, data, created_at, updated_at, "userId")
                    VALUES (%s, %s, %s, %s, %s)
                """
                self.execute_query(
                    query,
                    (
                        embedding_id,
                        data_base64,
                        datetime.now(),
                        datetime.now(),
                        user_id,
                    ),  # Added user_id
                    fetch=False,
                )

            logger.info(
                f"Successfully stored embedding: {embedding_id} for user {user_id}"
            )
            return True
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            return False

    def fetch_embedding(self, embedding_id):
        """Fetch embedding data from the database."""
        try:
            self.connect()

            query = "SELECT data FROM embeddings WHERE embedding_id = %s"
            results = self.execute_query(query, (embedding_id,))

            if results and "data" in results[0]:
                # Convert base64 string back to bytes
                data_bytes = base64.b64decode(results[0]["data"])
                logger.info(f"Successfully fetched embedding: {embedding_id}")
                return data_bytes
            else:
                logger.warning(f"No embedding found for ID: {embedding_id}")
                return None
        except Exception as e:
            logger.error(f"Error fetching embedding: {str(e)}")
            return None

    # --- Async Job Methods --- 
    def create_async_job(self, job_id: str, user_id: str, job_type: str, status: str = "PENDING") -> Optional[Dict[str, Any]]:
        """Create a new job in the async_jobs table."""
        try:
            self.connect()
            query = """
                INSERT INTO async_jobs (id, "userId", job_type, status, "createdAt", "predictionId")
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING *
            """
            params = (
                job_id,
                user_id,
                job_type,
                status,
                datetime.now(),
                job_id # Assuming predictionId can be same as id or is used for other purposes
            )
            results = self.execute_query(query, params, fetch=True)
            logger.info(f"Created async job: {job_id} for user {user_id} with type {job_type}")
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Error creating async job {job_id}: {str(e)}")
            # raise # Decide if you want to raise or return None/False
            return None

    def update_async_job_status(
        self, 
        job_id: str, 
        status: str, 
        result_data: Optional[Dict[str, Any]] = None, 
        error_message: Optional[str] = None
    ) -> bool:
        """Update the status, result_data, or error for an async job."""
        try:
            self.connect()
            
            set_clauses = ['status = %s', '"completedAt" = %s']
            params = [status, datetime.now()]

            if result_data is not None:
                set_clauses.append('result_data = %s')
                params.append(json.dumps(result_data)) # Serialize dict to JSON string
            
            if error_message is not None:
                set_clauses.append('error = %s')
                params.append(error_message)
            
            params.append(job_id) # For the WHERE clause

            query = f"""
                UPDATE async_jobs
                SET {", ".join(set_clauses)}
                WHERE id = %s
            """
            
            self.execute_query(query, tuple(params), fetch=False)
            logger.info(f"Updated async job {job_id} to status {status}")
            return True
        except Exception as e:
            logger.error(f"Error updating async job {job_id} to status {status}: {str(e)}")
            return False

    def get_async_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get an async job by its ID."""
        try:
            self.connect()
            query = 'SELECT * FROM async_jobs WHERE id = %s'
            results = self.execute_query(query, (job_id,))
            if results:
                logger.debug(f"Fetched async job: {job_id}")
                # Ensure result_data is a dict if it was stored as JSON string
                job_data = results[0]
                if isinstance(job_data.get('result_data'), str):
                    try:
                        job_data['result_data'] = json.loads(job_data['result_data'])
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse result_data for job {job_id} in get_async_job_by_id")
                        # Keep it as string or set to None/error indicator based on preference
                return job_data
            else:
                logger.warning(f"Async job not found: {job_id}")
                return None
        except Exception as e:
            logger.error(f"Error fetching async job {job_id}: {str(e)}")
            return None


# Create a singleton instance
db_client = DBClient()
