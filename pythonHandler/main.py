import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import replicate
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
import sys
import uuid
import time
from typing import List, Optional
from utils.prisma_client import prisma_client
import psycopg2
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

# Load environment variables
load_dotenv()


# === Request Validation Models ===
class TransactionBase(BaseModel):
    """Base model for transaction data containing description field."""

    description: str

    @field_validator("description")
    @classmethod
    def description_must_not_be_empty(cls, v):
        v = v.strip() if v else ""
        if not v:
            raise ValueError("Description cannot be empty")
        return v


class Transaction(TransactionBase):
    """Transaction model with narrative and category fields."""

    Category: Optional[str] = None


class TrainRequest(BaseModel):
    """Request model for the /train endpoint."""

    transactions: List[Transaction]
    expenseSheetId: str = Field(..., min_length=1)
    userId: Optional[str] = None

    @field_validator("transactions")
    @classmethod
    def transactions_not_empty(cls, v):
        if not v or len(v) < 10:
            raise ValueError("At least 10 valid transactions required for training")
        return v


class ClassifyRequest(BaseModel):
    """Request model for the /classify endpoint."""

    transactions: List[str]
    spreadsheetId: str = Field(..., min_length=1)
    sheetName: Optional[str] = "new_transactions"
    categoryColumn: Optional[str] = "E"
    startRow: Optional[str] = "1"

    @field_validator("transactions")
    @classmethod
    def transactions_not_empty(cls, v):
        if not v:
            raise ValueError("No transactions provided")
        return v


class ApiKeyRequest(BaseModel):
    """Request model for the /api-key endpoint."""

    userId: str = Field(..., min_length=1)


class UserConfigRequest(BaseModel):
    """Request model for the /user-config endpoint."""

    userId: str = Field(..., min_length=1)
    apiKey: Optional[str] = None


# Configure logging
logging.basicConfig(
    stream=sys.stdout,  # Log to stdout for Docker/Gunicorn to capture
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define backend API URL for webhooks
BACKEND_API = os.environ.get("BACKEND_API", "http://localhost:5001")
logger.info(f"Backend API URL: {BACKEND_API}")

# Log startup information
logger.info("=== Main Application Starting ===")
logger.info(f"Environment: {os.environ.get('FLASK_ENV', 'production')}")

# Connect to the database with a simple retry mechanism
connected = False
max_retries = 3
retry_count = 0

while not connected and retry_count < max_retries:
    try:
        logger.info(f"Connecting to database (attempt {retry_count + 1}/{max_retries})")
        prisma_client.connect()
        logger.info("Successfully connected to database on startup")
        connected = True
    except Exception as e:
        retry_count += 1
        logger.error(f"Database connection attempt {retry_count} failed: {e}")
        if retry_count < max_retries:
            # Simple backoff strategy
            sleep_time = 1 * retry_count
            logger.info(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
        else:
            logger.warning(
                "All connection attempts failed, will retry on first request"
            )

# Even if we couldn't connect, continue with app startup
# The app will try to reconnect on each request


@app.teardown_appcontext
def shutdown_db_connection(exception=None):
    """Disconnect from the database when the application context ends."""
    try:
        # Only disconnect if we're shutting down the app
        if exception and isinstance(exception, Exception):
            logger.info("Disconnecting from database due to exception")
            prisma_client.disconnect()
            logger.info("Disconnected from database")
    except Exception as e:
        logger.error(f"Error disconnecting from database: {e}")


@app.before_request
def ensure_db_connection():
    """Ensure database connection is active before each request."""
    try:
        # Use the simplified connect method
        prisma_client.connect()
    except Exception as e:
        logger.error(f"Failed to ensure database connection: {e}")
        logger.warning("Request may fail due to database connection issues")


@app.route("/")
def home():
    """Home endpoint"""
    logger.info("Home endpoint accessed")
    return jsonify(
        {
            "status": "online",
            "version": "1.0.0",
            "endpoints": [
                "/classify",
                "/train",
                "/health",
                "/api-key",
                "/status/:prediction_id",
            ],
        }
    )


@app.route("/health")
def health():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed")

    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {"app": "healthy"},
    }

    # Check database connection
    try:
        # Use psycopg2 to check database connection
        conn = psycopg2.connect(
            host=os.environ.get("PGHOST_UNPOOLED"),
            database=os.environ.get("PGDATABASE"),
            user=os.environ.get("PGUSER"),
            password=os.environ.get("PGPASSWORD"),
            sslmode="require",
            connect_timeout=5,
        )

        # Execute a simple query
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            result = cur.fetchone()

        # Close the connection
        conn.close()

        if result and result[0] == 1:
            health_status["components"]["database"] = "healthy"
        else:
            health_status["components"]["database"] = "unhealthy"
            health_status["status"] = "degraded"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["components"]["database"] = "unhealthy"
        health_status["status"] = "degraded"
        health_status["database_error"] = str(e)

    # Return health status
    return jsonify(health_status)


def validate_api_key(api_key: str) -> str:
    """Validate API key and return user ID if valid."""
    try:
        # Clean the API key
        api_key = api_key.strip()
        logger.info(f"Validating API key (length: {len(api_key)})")
        logger.debug(f"API key first/last 4 chars: {api_key[:4]}...{api_key[-4:]}")

        # Log the query we're about to make
        logger.info("Querying database for API key validation")

        # Use Prisma client to find account by API key
        account = prisma_client.get_account_by_api_key(api_key)

        if not account:
            logger.error(
                f"No account found for API key: {api_key[:4]}...{api_key[-4:]}"
            )
            raise Exception("Invalid API key - no matching account found")

        # Log the found user data (excluding sensitive info)
        logger.info(f"Found user data - userId: {account['userId']}")

        if not account["userId"]:
            logger.error("User data found but missing userId")
            raise Exception("Invalid user configuration - missing userId")

        # Track API usage on successful validation
        try:
            prisma_client.track_api_usage(api_key)
            logger.debug(f"Tracked API usage for key: {api_key[:4]}...{api_key[-4:]}")
        except Exception as tracking_error:
            # Don't fail the validation if tracking fails
            logger.warning(f"Error tracking API usage: {tracking_error}")

        return account["userId"]

    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        logger.error(f"Full error details: {e}")
        logger.error(
            f"API key validation failed for key: {api_key[:4]}...{api_key[-4:]}"
        )
        raise Exception(f"API key validation failed: {str(e)}")


def update_process_status(status_text: str, mode: str, user_id: str) -> None:
    """Log process status without database updates."""
    try:
        logger.info(
            f"Process status update - mode: {mode}, user: {user_id}, status: {status_text}"
        )
    except Exception as e:
        logger.error(f"Error logging process status: {e}")
        logger.error(
            f"Status update attempted - mode: {mode}, user: {user_id}, status: {status_text}"
        )


def clean_text(text: str) -> str:
    """Clean transaction description text while preserving business names."""
    # Convert to string and strip whitespace
    text = str(text).strip()

    # Remove only transaction-specific metadata
    patterns = [
        r"\s*\d{2,4}[-/]\d{2}[-/]\d{2,4}",  # Dates
        r"\s*\d{2}:\d{2}(?::\d{2})?",  # Times
        r"\s*Card\s+[xX*]+\d{4}",  # Card numbers
        r"\s*\|\s*[\d\.]+$",  # Amount at end
        r"\s*\|\s*[A-Z0-9\s]+$",  # Reference codes
        r"\s+(?:Value Date|Card ending|ref|reference)\s*:?.*$",  # Transaction metadata
        r"(?i)\s+(?:AUS|USA|UK|NS|CYP)$",  # Country codes at end
        r"\s+\([^)]*\)$",  # Anything in parentheses at the end
    ]

    # Apply patterns one by one
    for pattern in patterns:
        text = re.sub(pattern, "", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text.strip()


def run_prediction(descriptions: list) -> dict:
    """Run prediction using Replicate API."""
    try:
        model = replicate.models.get("replicate/all-mpnet-base-v2")
        version = model.versions.get(
            "b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305"
        )

        # Create the prediction without webhook
        logger.info("Creating prediction without webhook (using polling)")
        prediction = replicate.predictions.create(
            version=version,
            input={"text_batch": json.dumps(descriptions)},
        )

        return prediction
    except Exception as e:
        logger.error(f"Error running prediction: {e}")
        raise


def store_embeddings(data: np.ndarray, embedding_id: str) -> bool:
    """Store embeddings in database using Prisma.

    Args:
        data: The numpy array containing the embeddings
        embedding_id: The identifier for the embeddings (e.g., spreadsheet_id or spreadsheet_id_index)

    Returns:
        bool: True if the embeddings were successfully stored, False otherwise
    """
    try:
        logger.info(f"Storing embeddings with ID: {embedding_id}, shape: {data.shape}")

        # Convert numpy array to bytes
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp_file:
            np.savez_compressed(temp_file, data)
            temp_file_path = temp_file.name

        # Read the compressed data as bytes
        with open(temp_file_path, "rb") as f:
            data_bytes = f.read()
            logger.info(f"Read {len(data_bytes)} bytes from temporary file")

        # Clean up temp file
        os.unlink(temp_file_path)

        # Store in database using Prisma
        result = prisma_client.store_embedding(embedding_id, data_bytes)
        logger.info(f"Embedding storage result: {result}")

        # Try to link embedding to account if we have an API key in the request context
        try:
            # Check if there's an API key in the current request context
            if hasattr(request, "headers") and request.headers.get("X-API-Key"):
                api_key = request.headers.get("X-API-Key")
                prisma_client.track_embedding_creation(api_key, embedding_id)
                logger.info(f"Linked embedding {embedding_id} to account via API key")
        except Exception as tracking_error:
            # Don't fail the storage if tracking fails
            logger.warning(f"Error tracking embedding creation: {tracking_error}")

        return result
    except Exception as e:
        logger.error(f"Error storing embeddings: {e}")
        return False


def fetch_embeddings(embedding_id: str) -> np.ndarray:
    """Fetch embeddings from database.

    Args:
        embedding_id: The identifier for the embeddings to fetch

    Returns:
        np.ndarray: The embeddings as a numpy array, or empty array if not found
    """
    try:
        logger.info(f"Fetching embeddings with ID: {embedding_id}")

        # Fetch from database using Prisma
        data_bytes = prisma_client.fetch_embedding(embedding_id)

        if not data_bytes:
            logger.warning(f"No embeddings found with ID: {embedding_id}")
            return np.array([])

        logger.info(f"Retrieved {len(data_bytes)} bytes from database")

        # Convert bytes to numpy array
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp_file:
            temp_file.write(data_bytes)
            temp_file_path = temp_file.name

        # Load the numpy array from the temporary file
        try:
            with np.load(temp_file_path, allow_pickle=True) as data:
                # Get the first array in the npz file
                array_name = list(data.files)[0]
                embeddings = data[array_name]
                logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
        except Exception as e:
            logger.error(f"Error loading numpy array from bytes: {e}")
            embeddings = np.array([])
        finally:
            # Clean up temp file
            os.unlink(temp_file_path)

        return embeddings
    except Exception as e:
        logger.error(f"Error fetching embeddings: {e}")
        return np.array([])


@app.route("/train", methods=["POST"])
def train_model():
    """Train the model with new data."""
    try:
        # Log all incoming request details
        logger.info("=== Incoming Training Request ===")
        logger.info(f"Headers: {dict(request.headers)}")

        # Get request data
        data = request.get_json()
        if not data:
            return create_error_response("Missing request data", 400)

        logger.info(f"Request data: {data}")

        # Validate request data
        validated_data, error_response = validate_request_data(TrainRequest, data)
        if error_response:
            return error_response

        # Extract validated data
        transactions = validated_data.transactions
        sheet_id = validated_data.expenseSheetId

        # Get user ID either from API key validation or payload
        user_id = None
        api_key = request.headers.get("X-API-Key")

        # First try to validate API key if provided
        if api_key:
            try:
                user_id = validate_api_key(api_key)
                logger.info(f"Got user_id from API key validation: {user_id}")

                # Track API usage explicitly for the training endpoint
                prisma_client.track_api_usage(api_key)
                logger.info(f"Tracked API usage for training endpoint")
            except Exception as e:
                # If API key validation fails and we're not in fallback mode, return error
                if not validated_data.userId:
                    logger.error(
                        f"API key validation failed and no userId provided in payload: {str(e)}"
                    )
                    return create_error_response(
                        f"API key validation failed: {str(e)}", 401
                    )
                logger.error(f"API key validation failed: {str(e)}")
                logger.info("Falling back to userId from payload")

        # Create or update user configuration
        try:
            # Check if user config exists
            account = prisma_client.get_account_by_user_id(user_id)

            if not account:
                # Create new user config
                default_config = {
                    "userId": user_id,
                    "categorisationRange": "A:Z",
                    "columnOrderCategorisation": {
                        "categoryColumn": "E",
                        "descriptionColumn": "C",
                    },
                    "categorisationTab": None,
                    "api_key": api_key if api_key else None,
                }
                prisma_client.insert_account(user_id, default_config)
                logger.info(f"Created new user configuration for {user_id}")
            else:
                # Update API key if it's provided and different from stored one
                if api_key and (
                    not account.get("api_key") or account.get("api_key") != api_key
                ):
                    prisma_client.update_account(user_id, {"api_key": api_key})
                    logger.info(f"Updated API key for user {user_id}")
                logger.info(f"Found existing user configuration for {user_id}")

        except Exception as e:
            logger.warning(f"Error managing user configuration: {str(e)}")
            # Continue with training even if config creation fails

        # Convert transactions to DataFrame
        transactions_data = [t.dict() for t in transactions]
        df = pd.DataFrame(transactions_data)
        logger.info(f"DataFrame created with columns: {df.columns.tolist()}")

        # Clean descriptions
        df["description"] = df["description"].apply(clean_text)
        df = df.drop_duplicates(subset=["description"])

        # Store training data index with proper dtype
        df["item_id"] = range(len(df))

        # Log the categories we're training with
        unique_categories = df["Category"].unique().tolist()
        logger.info(
            f"Training with {len(unique_categories)} unique categories: {unique_categories}"
        )
        logger.info(f"Category column length: {df['Category'].str.len().describe()}")

        # Create structured array for index data
        index_data = np.array(
            [
                (i, desc, cat)
                for i, (desc, cat) in enumerate(
                    zip(df["description"].values, df["Category"].values)
                )
            ],
            dtype=[
                ("item_id", np.int32),
                ("description", "U256"),  # Unicode string up to 256 chars
                ("category", "U128"),  # Unicode string up to 128 chars
            ],
        )

        # Store index data
        store_embeddings(index_data, f"{sheet_id}_index")
        logger.info(f"Stored index data with {len(index_data)} entries")

        # Create a placeholder for the embeddings so we know it's expected
        # This will be replaced with the actual embeddings when the webhook is called
        placeholder = np.array([[0.0] * 768])  # Standard embedding size
        store_embeddings(placeholder, f"{sheet_id}")
        logger.info(f"Stored placeholder embeddings for {sheet_id}")

        # Verify index data was stored correctly
        try:
            stored_index = fetch_embeddings(f"{sheet_id}_index")
            logger.info(f"Verified index data with shape: {stored_index.shape}")
        except Exception as e:
            logger.warning(f"Could not verify stored index data: {e}")

        # Get descriptions for embedding
        descriptions = df["description"].tolist()
        logger.info(f"Prepared {len(descriptions)} descriptions for embedding")

        # Create prediction using the run_prediction function
        prediction = run_prediction(descriptions)

        logger.info(f"Created prediction with ID: {prediction.id}")

        # Wait for the prediction to complete
        attempt = 0
        max_attempts = 60  # Maximum number of attempts

        # Start with shorter checks for the first few attempts
        initial_delays = [2, 3, 5, 10, 15]  # First few checks are faster

        while prediction.status != "succeeded" and attempt < max_attempts:
            if prediction.status == "failed":
                logger.error(f"Training prediction failed: {prediction.error}")
                update_process_status("failed", "training", user_id)
                return jsonify({"status": "failed", "error": prediction.error}), 500

            try:
                # Use initial delays for first few attempts, then fixed delay
                if attempt < len(initial_delays):
                    delay = initial_delays[attempt]
                else:
                    delay = 5  # Fixed 5-second delay after initial attempts

                logger.info(
                    f"Waiting {delay} seconds before checking training prediction status again (attempt {attempt+1}/{max_attempts})"
                )
                time.sleep(delay)

                # Reload prediction status
                try:
                    prediction.reload()
                except Exception as reload_error:
                    logger.warning(f"Error reloading prediction status: {reload_error}")
                    # If reload fails, continue to next attempt rather than failing
                    attempt += 1
                    continue

                attempt += 1

            except Exception as e:
                logger.warning(f"Error during training status check: {e}")
                # Don't increment attempt on error, just continue with shorter delay
                time.sleep(2)
                continue

        if attempt >= max_attempts:
            logger.warning(
                f"Timed out waiting for training prediction to complete, checking local database"
            )

            # Check if we have a webhook result
            try:
                webhook_result = prisma_client.get_webhook_result(prediction.id)
                if webhook_result and webhook_result.get("status") == "success":
                    logger.info("Found successful webhook result, training completed")
                    return jsonify(
                        {"status": "completed", "prediction_id": prediction.id}
                    )
            except Exception as db_error:
                logger.warning(f"Error checking webhook result: {db_error}")

            # If no webhook result but we haven't failed, continue processing
            logger.info("No webhook result found, but continuing with processing")
            return jsonify({"status": "processing", "prediction_id": prediction.id})

        elif prediction.status == "succeeded":
            logger.info("Training prediction completed successfully")

            # Process the embeddings from the prediction output
            if (
                prediction.output
                and isinstance(prediction.output, list)
                and len(prediction.output) > 0
            ):
                try:
                    # Extract embeddings from the response
                    embeddings = np.array(
                        [item["embedding"] for item in prediction.output],
                        dtype=np.float32,
                    )
                    logger.info(f"Extracted embeddings with shape: {embeddings.shape}")

                    # Store the embeddings
                    store_result = store_embeddings(embeddings, f"{sheet_id}")
                    logger.info(f"Stored embeddings for {sheet_id}: {store_result}")

                    # Store webhook result
                    result = {
                        "user_id": user_id,
                        "status": "success",
                        "embeddings_shape": str(embeddings.shape),
                        "spreadsheet_id": sheet_id,
                    }
                    prisma_client.insert_webhook_result(prediction.id, result)
                except Exception as e:
                    logger.error(f"Error processing training embeddings: {e}")
            else:
                logger.warning("No embeddings found in training prediction output")

        # Update status
        update_process_status("processing", "training", user_id)

        return jsonify({"status": "processing", "prediction_id": prediction.id})

    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        return create_error_response(str(e), 500)


@app.route("/classify", methods=["POST"])
def classify_transactions():
    """Classify transactions endpoint - directly using Replicate API."""
    try:
        # Validate API key
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return create_error_response("Missing API key", 401)

        # Get request data
        data = request.get_json()
        if not data:
            return create_error_response("No data provided", 400)

        logger.info(
            f"Classify request received with {len(data.get('transactions', []))} transactions"
        )

        # Validate with Pydantic model
        validated_data, error_response = validate_request_data(ClassifyRequest, data)
        if error_response:
            return error_response

        # Extract validated data
        transaction_descriptions = validated_data.transactions  # List of strings
        spreadsheet_id = validated_data.spreadsheetId
        sheet_name = validated_data.sheetName
        category_column = validated_data.categoryColumn
        start_row = validated_data.startRow

        # Log all important parameters
        logger.info(
            f"Processing classify request with spreadsheet_id: {spreadsheet_id}"
        )

        # Validate API key and get user ID
        try:
            user_id = validate_api_key(api_key)
            logger.info(f"API key validated for user: {user_id}")

            # Track API usage
            prisma_client.track_api_usage(api_key)
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return create_error_response(f"API key validation failed: {str(e)}", 401)

        # Verify training data exists
        try:
            trained_data = fetch_embeddings(f"{spreadsheet_id}_index")
            if len(trained_data) == 0:
                error_msg = "No training data found. Please train the model first."
                logger.error(f"{error_msg} for spreadsheet_id: {spreadsheet_id}")
                return create_error_response(error_msg, 400)
        except Exception as e:
            error_msg = f"Error accessing training data: {str(e)}"
            logger.error(f"{error_msg} for spreadsheet_id: {spreadsheet_id}")
            return create_error_response(error_msg, 400)

        # Clean and prepare descriptions
        cleaned_descriptions = []
        original_descriptions = []
        for desc in transaction_descriptions:
            if desc:
                cleaned = clean_text(desc)
                cleaned_descriptions.append(cleaned)
                original_descriptions.append(desc)
            else:
                logger.warning(f"Empty description found in transactions")

        if not cleaned_descriptions:
            logger.error("No valid descriptions found in transactions")
            return create_error_response(
                "No valid descriptions found in transactions", 400
            )

        # Get embeddings for descriptions using Replicate
        logger.info(f"Getting embeddings for {len(cleaned_descriptions)} descriptions")

        # Create prediction
        model = replicate.models.get("replicate/all-mpnet-base-v2")
        version = model.versions.get(
            "b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305"
        )

        # Create prediction without webhook
        prediction = replicate.predictions.create(
            version=version, input={"text_batch": json.dumps(cleaned_descriptions)}
        )

        logger.info(f"Created prediction with ID: {prediction.id}")

        # Store config in database for later retrieval
        config_data = {
            "user_id": user_id,
            "spreadsheet_id": str(spreadsheet_id),  # Ensure spreadsheet_id is a string
            "sheet_name": sheet_name,
            "category_column": category_column,
            "start_row": start_row,
            "original_descriptions": original_descriptions,
        }

        # Log the config data before storage
        logger.info(f"Storing config data: {json.dumps(config_data)}")

        # Store the configuration
        store_result = prisma_client.insert_webhook_result(prediction.id, config_data)

        # Verify storage was successful
        if store_result:
            logger.info(
                f"Successfully stored configuration with spreadsheet_id: {spreadsheet_id}"
            )
        else:
            logger.warning(
                f"Failed to store configuration, will attempt direct database access"
            )
            # Additional fallback could be implemented here if needed

        # Return the prediction ID for status checking
        return jsonify(
            {
                "status": "processing",
                "prediction_id": prediction.id,
                "message": "Classification started. Check status endpoint for updates.",
            }
        )

    except Exception as e:
        logger.error(f"Error in classify_transactions: {e}")
        return create_error_response(str(e), 500)


@app.route("/status/<prediction_id>", methods=["GET"])
def get_prediction_status(prediction_id):
    """Get the status of a prediction directly from Replicate API."""
    try:
        logger.info(f"Getting status for prediction: {prediction_id}")

        # Get prediction directly from Replicate
        prediction = replicate.predictions.get(prediction_id)

        if not prediction:
            logger.warning(f"Prediction {prediction_id} not found in Replicate")
            return create_error_response("Prediction not found", 404)

        # Get basic prediction status
        status = prediction.status
        logger.info(f"Prediction {prediction_id} status: {status}")

        # Get config data for this prediction if it exists
        config_data = prisma_client.get_webhook_result(prediction_id)

        # Log retrieved config data
        logger.info(f"Retrieved webhook result for {prediction_id}: {config_data}")

        # Log config data for debugging
        if config_data and isinstance(config_data, dict) and config_data.get("results"):
            config_data = config_data.get("results")
            logger.info(f"Using results field from webhook_result")

        if config_data and isinstance(config_data, dict):
            spreadsheet_id = config_data.get("spreadsheet_id")
            logger.info(f"Retrieved config with spreadsheet_id: {spreadsheet_id}")
        else:
            logger.warning(f"No config data found for prediction {prediction_id}")
            # Create empty config_data if it doesn't exist
            config_data = {}

        # If prediction is still processing, return simple status
        if status == "processing":
            return jsonify(
                {"status": "processing", "message": "Processing in progress"}
            )

        # If prediction failed, return error
        elif status == "failed":
            error_message = prediction.error or "Unknown error occurred"
            logger.error(f"Prediction {prediction_id} failed: {error_message}")
            return create_error_response(f"Prediction failed: {error_message}", 500)

        # If prediction succeeded, process the results
        elif status == "succeeded":
            logger.info(f"Prediction {prediction_id} succeeded")

            # If succeeded but no output, return error
            if not prediction.output:
                logger.error(f"Prediction {prediction_id} succeeded but has no output")
                return create_error_response(
                    "Prediction succeeded but has no output", 500
                )

            # If we have config data, process the results
            if config_data:
                try:
                    spreadsheet_id = config_data.get("spreadsheet_id")
                    user_id = config_data.get("user_id")
                    original_descriptions = config_data.get("original_descriptions", [])

                    # Handle missing spreadsheet_id
                    if not spreadsheet_id:
                        logger.error(
                            f"Missing spreadsheet_id in config data for prediction {prediction_id}"
                        )
                        return create_error_response(
                            "Configuration error: Missing spreadsheet ID", 400
                        )

                    logger.info(
                        f"Processing results with config: spreadsheet_id={spreadsheet_id}, user_id={user_id}"
                    )

                    # Get trained embeddings
                    trained_embeddings = fetch_embeddings(f"{spreadsheet_id}")
                    trained_data = fetch_embeddings(f"{spreadsheet_id}_index")

                    # Check if we have valid embeddings
                    if trained_embeddings.size == 0 or trained_data.size == 0:
                        logger.error(
                            f"No valid embeddings or training data found for spreadsheet_id: {spreadsheet_id}"
                        )
                        return create_error_response(
                            f"No valid training data found for spreadsheet ID: {spreadsheet_id}",
                            400,
                        )

                    # Process embeddings from Replicate
                    if (
                        isinstance(prediction.output, list)
                        and len(prediction.output) > 0
                    ):
                        # Extract embeddings from the prediction output
                        new_embeddings = np.array(
                            [item["embedding"] for item in prediction.output],
                            dtype=np.float32,
                        )
                        logger.info(
                            f"Extracted {len(new_embeddings)} embeddings from prediction output"
                        )

                        # Calculate similarities
                        similarities = cosine_similarity(
                            new_embeddings, trained_embeddings
                        )
                        best_matches = similarities.argmax(axis=1)

                        # Get predicted categories and confidence scores
                        results = []
                        for i, idx in enumerate(best_matches):
                            try:
                                # Extract the category directly from the structured array
                                category = None

                                # Try different field names and positions to get the category
                                if "category" in trained_data.dtype.names:
                                    category = str(trained_data[idx]["category"])
                                elif "Category" in trained_data.dtype.names:
                                    category = str(trained_data[idx]["Category"])
                                else:
                                    # Fallback to index 2 which should be the category field
                                    category = str(trained_data[idx][2])

                                similarity_score = float(similarities[i][idx])

                                results.append(
                                    {
                                        "predicted_category": category,
                                        "similarity_score": similarity_score,
                                        "narrative": (
                                            original_descriptions[i]
                                            if i < len(original_descriptions)
                                            else ""
                                        ),
                                    }
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error processing prediction {i}: {str(e)}"
                                )
                                results.append(
                                    {
                                        "predicted_category": "Unknown",
                                        "similarity_score": 0.0,
                                        "narrative": (
                                            original_descriptions[i]
                                            if i < len(original_descriptions)
                                            else ""
                                        ),
                                    }
                                )

                        # Store the results in a simple format
                        result_data = {
                            "status": "success",
                            "data": results,
                            "config": {
                                "spreadsheet_id": spreadsheet_id,
                                "sheet_name": config_data.get("sheet_name"),
                                "category_column": config_data.get("category_column"),
                                "start_row": config_data.get("start_row"),
                            },
                        }

                        # Store the results in the database
                        prisma_client.insert_webhook_result(prediction.id, result_data)

                        # Return the results
                        return jsonify(
                            {
                                "status": "completed",
                                "message": "Processing completed successfully",
                                "results": results,
                                "config": {
                                    "categoryColumn": config_data.get(
                                        "category_column"
                                    ),
                                    "startRow": config_data.get("start_row"),
                                    "sheetName": config_data.get("sheet_name"),
                                    "spreadsheetId": spreadsheet_id,
                                },
                            }
                        )
                    else:
                        logger.warning(
                            f"Prediction output is not in expected format: {type(prediction.output)}"
                        )
                        return create_error_response(
                            "Prediction output is not in expected format", 500
                        )

                except Exception as e:
                    logger.error(f"Error processing classification results: {e}")
                    return create_error_response(
                        f"Error processing results: {str(e)}", 500
                    )
            else:
                # If we don't have config data but have output, return raw output
                logger.warning(f"No config data found for prediction {prediction_id}")
                return jsonify(
                    {
                        "status": "completed",
                        "message": "Processing completed but configuration not found",
                        "raw_output": {
                            "output_type": type(prediction.output).__name__,
                            "output_length": (
                                len(prediction.output)
                                if isinstance(prediction.output, list)
                                else "n/a"
                            ),
                            "sample": (
                                prediction.output[0]
                                if isinstance(prediction.output, list)
                                and len(prediction.output) > 0
                                else prediction.output
                            ),
                        },
                    }
                )

        # For any other status
        else:
            return jsonify(
                {"status": status, "message": f"Prediction has status: {status}"}
            )

    except Exception as e:
        logger.error(f"Error getting prediction status: {e}")
        return create_error_response(str(e), 500)


@app.route("/api-key", methods=["GET", "POST"])
def manage_api_key():
    """Get or generate API key for a user."""
    try:
        # For GET requests, we need to validate the user
        if request.method == "GET":
            # Check for existing API key in header
            api_key = request.headers.get("X-API-Key")
            if api_key:
                try:
                    # Validate the API key
                    user_id = validate_api_key(api_key)
                    logger.info(f"API key validated for user: {user_id}")

                    # Track API usage explicitly for API key management
                    prisma_client.track_api_usage(api_key)
                    logger.info(f"Tracked API usage for API key management endpoint")

                    # Return the API key
                    return jsonify(
                        {"status": "success", "user_id": user_id, "api_key": api_key}
                    )
                except Exception as e:
                    logger.error(f"API key validation failed: {e}")
                    return create_error_response(
                        f"API key validation failed: {str(e)}", 401
                    )

            # If no API key in header, check for user_id in query params
            user_id = request.args.get("userId")
            if not user_id:
                return create_error_response("Missing userId parameter", 400)

            # If user_id is an email, prefix it with google-oauth2|
            if "@" in user_id and not user_id.startswith("google-oauth2|"):
                user_id = f"google-oauth2|{user_id}"

            # Use psycopg2 to execute the query
            conn = psycopg2.connect(
                host=os.environ.get("PGHOST_UNPOOLED"),
                database=os.environ.get("PGDATABASE"),
                user=os.environ.get("PGUSER"),
                password=os.environ.get("PGPASSWORD"),
                sslmode="require",
                connect_timeout=5,
            )

            # Execute query to find account by user ID
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT "userId", api_key FROM "account" WHERE "userId" = %s',
                    (user_id,),
                )
                result = cur.fetchone()

                # Close the connection
                conn.close()

                if not result:
                    return create_error_response("User not found", 404)

                # Return the API key if it exists
                if result[1]:  # api_key is the second column
                    return jsonify(
                        {"status": "success", "user_id": user_id, "api_key": result[1]}
                    )
                else:
                    return (
                        jsonify(
                            {
                                "status": "not_found",
                                "user_id": user_id,
                                "message": "No API key found for this user",
                            }
                        ),
                        404,
                    )

        # For POST requests, we generate a new API key
        elif request.method == "POST":
            data = request.get_json()
            if not data:
                return create_error_response("Missing request data", 400)

            # Validate request data
            validated_data, error_response = validate_request_data(ApiKeyRequest, data)
            if error_response:
                return error_response

            user_id = validated_data.userId

            # If user_id is an email, prefix it with google-oauth2|
            if "@" in user_id and not user_id.startswith("google-oauth2|"):
                user_id = f"google-oauth2|{user_id}"

            # Generate a new API key (UUID)
            new_api_key = str(uuid.uuid4())

            # Use psycopg2 to execute the query
            conn = psycopg2.connect(
                host=os.environ.get("PGHOST_UNPOOLED"),
                database=os.environ.get("PGDATABASE"),
                user=os.environ.get("PGUSER"),
                password=os.environ.get("PGPASSWORD"),
                sslmode="require",
                connect_timeout=5,
            )

            # Check if user exists and update or create accordingly
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT "userId" FROM "account" WHERE "userId" = %s', (user_id,)
                )
                user_exists = cur.fetchone() is not None

                if user_exists:
                    # Update existing account
                    cur.execute(
                        'UPDATE "account" SET api_key = %s WHERE "userId" = %s',
                        (new_api_key, user_id),
                    )
                    logger.info(f"Updated API key for user {user_id}")
                else:
                    # Create new account with default config
                    default_config = json.dumps(
                        {"categoryColumn": "E", "descriptionColumn": "C"}
                    )

                    cur.execute(
                        'INSERT INTO "account" ("userId", api_key, "categorisationRange", "categorisationTab", "columnOrderCategorisation") VALUES (%s, %s, %s, %s, %s)',
                        (user_id, new_api_key, "A:Z", None, default_config),
                    )
                    logger.info(f"Created new user with API key: {user_id}")

                # Commit the transaction
                conn.commit()

            # Close the connection
            conn.close()

            return jsonify(
                {"status": "success", "user_id": user_id, "api_key": new_api_key}
            )

    except Exception as e:
        logger.error(f"Error managing API key: {e}")
        return create_error_response(str(e), 500)


@app.route("/user-config", methods=["GET"])
def get_user_config():
    """Get user configuration."""
    try:
        # Try to get userId from query params first
        user_id = request.args.get("userId")

        # If no userId in query params, try data
        if not user_id and request.is_json:
            data = request.get_json()
            if data:
                # Validate request data
                validated_data, error_response = validate_request_data(
                    UserConfigRequest, data
                )
                if error_response:
                    return error_response
                user_id = validated_data.userId

                # Check for API key in the validated data
                api_key = validated_data.apiKey
            else:
                return create_error_response("Missing userId parameter", 400)
        else:
            # Get API key from query params or headers
            api_key = request.args.get("apiKey") or request.headers.get("X-API-Key")

        if not user_id:
            return create_error_response("Missing userId parameter", 400)

        # If user_id is an email, prefix it with google-oauth2|
        if "@" in user_id and not user_id.startswith("google-oauth2|"):
            user_id = f"google-oauth2|{user_id}"

        # Track API usage if API key is provided
        if api_key:
            try:
                prisma_client.track_api_usage(api_key)
                logger.info(f"Tracked API usage for user config endpoint")
            except Exception as tracking_error:
                # Don't fail the request if tracking fails
                logger.warning(f"Error tracking API usage: {tracking_error}")

        response = prisma_client.get_account_by_user_id(user_id)
        if not response:
            return create_error_response("User not found", 404)

        # Update API key if provided
        if api_key:
            prisma_client.update_account(user_id, {"api_key": api_key})
            logger.info(f"Updated API key for user {user_id}")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error getting user config: {e}")
        return create_error_response(str(e), 500)


@app.route("/api-usage", methods=["GET"])
def get_api_usage():
    """Get API usage statistics for an account."""
    try:
        # Require API key authentication
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return create_error_response("Missing API key", 401)

        # Validate API key and get user ID
        try:
            user_id = validate_api_key(api_key)
            logger.info(f"API key validated for user: {user_id}")
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return create_error_response(f"API key validation failed: {str(e)}", 401)

        # Get usage statistics
        usage_stats = prisma_client.get_account_usage_stats(user_id)
        if not usage_stats:
            return create_error_response("Failed to retrieve usage statistics", 500)

        return jsonify(usage_stats), 200

    except Exception as e:
        logger.error(f"Error getting API usage statistics: {e}")
        return create_error_response(str(e), 500)


# === Helper Functions ===
def validate_request_data(model_class, data):
    """
    Validate request data using a Pydantic model

    Args:
        model_class: The Pydantic model class to use for validation
        data: The data to validate

    Returns:
        tuple: (validated_data, None) if validation succeeds, (None, error_response) if validation fails
    """
    try:
        # Validate request data using the Pydantic model
        validated_data = model_class(**data)
        return validated_data, None
    except ValidationError as e:
        # Extract validation errors
        error_details = []
        for error in e.errors():
            location = ".".join(str(loc) for loc in error["loc"])
            error_details.append(
                {"location": location, "message": error["msg"], "type": error["type"]}
            )

        # Create error response
        error_response = {"error": "Validation error", "details": error_details}
        logger.warning(f"Validation error: {error_details}")
        return None, (jsonify(error_response), 400)


def create_error_response(message, status_code=400, details=None):
    """
    Create a standardized error response

    Args:
        message: The error message
        status_code: The HTTP status code
        details: Additional error details

    Returns:
        tuple: (jsonify(error_response), status_code)
    """
    error_response = {"error": message}

    if details:
        error_response["details"] = details

    return jsonify(error_response), status_code


# Add a function to clean up old webhook results if needed
def cleanup_old_webhook_results():
    """Clean up old webhook results from the database to prevent excessive growth."""
    try:
        # Find webhook results older than 7 days
        cutoff_date = datetime.now() - timedelta(days=7)

        # Use Prisma client to delete old webhook results
        # This is a placeholder - implement the actual deletion logic using your Prisma client
        logger.info(f"Cleaning up webhook results older than {cutoff_date}")

        # Example implementation:
        # prisma_client.delete_old_webhook_results(cutoff_date)
        # logger.info(f"Cleaned up webhook results older than {cutoff_date}")
    except Exception as e:
        logger.error(f"Error cleaning up old webhook results: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    app.run(host="0.0.0.0", port=port)
