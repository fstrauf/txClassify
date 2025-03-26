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
import time
from typing import List, Optional
from utils.prisma_client import prisma_client
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator
from functools import wraps

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
        # Use prisma_client to check database connection
        connected = prisma_client.connect()

        if connected:
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


def validate_api_key(api_key: str, track_usage: bool = True) -> str:
    """
    Validate API key and return user ID if valid.

    Args:
        api_key: The API key to validate
        track_usage: Whether to track API usage (default: True)

    Returns:
        str: User ID if valid, empty string if invalid
    """
    if not api_key:
        logger.error("Empty API key provided")
        return ""

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
            return ""

        # Log the found user data (excluding sensitive info)
        logger.info(f"Found user data - userId: {account['userId']}")

        if not account["userId"]:
            logger.error("User data found but missing userId")
            return ""

        # Track API usage on successful validation if requested
        if track_usage:
            try:
                prisma_client.track_api_usage(api_key)
                logger.debug(
                    f"Tracked API usage for key: {api_key[:4]}...{api_key[-4:]}"
                )
            except Exception as tracking_error:
                # Don't fail the validation if tracking fails
                logger.warning(f"Error tracking API usage: {tracking_error}")

        return account["userId"]

    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        logger.error(
            f"API key validation failed for key: {api_key[:4] if len(api_key) >= 4 else ''}...{api_key[-4:] if len(api_key) >= 4 else ''}"
        )
        return ""


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
        model = replicate.models.get("beautyyuyanli/multilingual-e5-large")
        version = model.versions.get(
            "a06276a89f1a902d5fc225a9ca32b6e8e6292b7f3b136518878da97c458e2bad"
        )

        # Create the prediction without webhook
        logger.info(
            "Creating prediction with multilingual-e5-large model (using polling)"
        )
        prediction = replicate.predictions.create(
            version=version,
            input={"texts": json.dumps(descriptions), "normalize_embeddings": True},
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


def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return create_error_response("No API key provided", 401)

        user_id = validate_api_key(api_key)
        if not user_id:
            return create_error_response("Invalid API key", 401)

        # Add user_id to request context
        request.user_id = user_id
        request.api_key = api_key

        return f(*args, **kwargs)

    return decorated_function


@app.route("/train", methods=["POST"])
@require_api_key
def train_model():
    """Train the model with new data."""
    try:
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

        # Get user_id from request context (set by decorator)
        user_id = request.user_id
        logger.info(f"Processing request for user: {user_id}")

        # Create or update user configuration
        try:
            account = prisma_client.get_account_by_user_id(user_id)

            if not account:
                default_config = {
                    "userId": user_id,
                    "categorisationRange": "A:Z",
                    "columnOrderCategorisation": {
                        "categoryColumn": "E",
                        "descriptionColumn": "C",
                    },
                    "categorisationTab": None,
                    "api_key": request.api_key,
                }
                prisma_client.insert_account(user_id, default_config)
                logger.info(f"Created new user configuration for {user_id}")
            else:
                if request.api_key and (
                    not account.get("api_key")
                    or account.get("api_key") != request.api_key
                ):
                    prisma_client.update_account(user_id, {"api_key": request.api_key})
                    logger.info(f"Updated API key for user {user_id}")
                logger.info(f"Found existing user configuration for {user_id}")

        except Exception as e:
            logger.warning(f"Error managing user configuration: {str(e)}")

        # Convert transactions to DataFrame
        transactions_data = [t.model_dump() for t in transactions]
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
        placeholder = np.array([[0.0] * 1024])  # E5-large embedding size is 1024
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

        # Store initial configuration for status endpoint
        config_data = {
            "user_id": user_id,
            "spreadsheet_id": str(sheet_id),
            "status": "processing",
            "embeddings_data": {
                "index_data": index_data.tolist(),
                "descriptions": descriptions,
            },
        }

        # Store the configuration
        try:
            store_result = prisma_client.insert_webhook_result(
                prediction.id, config_data
            )
            logger.info(f"Stored initial configuration for prediction {prediction.id}")
        except Exception as e:
            logger.error(f"Error storing initial configuration: {e}")
            # Continue anyway to provide prediction ID to client

        # Update status
        update_process_status("processing", "training", user_id)

        return jsonify(
            {
                "status": "processing",
                "prediction_id": prediction.id,
                "message": "Training started. Check status endpoint for updates.",
            }
        )

    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        return create_error_response(str(e), 500)


@app.route("/classify", methods=["POST"])
@require_api_key
def classify_transactions():
    """Classify transactions endpoint."""
    try:
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
        transaction_descriptions = validated_data.transactions
        spreadsheet_id = validated_data.spreadsheetId
        sheet_name = validated_data.sheetName
        category_column = validated_data.categoryColumn
        start_row = validated_data.startRow

        # Log all important parameters
        logger.info(
            f"Processing classify request with spreadsheet_id: {spreadsheet_id}"
        )

        # Get user_id from request context (set by decorator)
        user_id = request.user_id
        logger.info(f"API key validated for user: {user_id}")

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
        model = replicate.models.get("beautyyuyanli/multilingual-e5-large")
        version = model.versions.get(
            "a06276a89f1a902d5fc225a9ca32b6e8e6292b7f3b136518878da97c458e2bad"
        )

        # Create prediction without webhook
        logger.info("Creating prediction with multilingual-e5-large model")
        prediction = replicate.predictions.create(
            version=version,
            input={
                "texts": json.dumps(cleaned_descriptions),
                "normalize_embeddings": True,
            },
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
        try:
            # First try to see if there's already a record for this prediction ID
            existing_config = prisma_client.get_webhook_result(prediction.id)
            if existing_config:
                logger.info(
                    f"Found existing configuration for prediction {prediction.id}, updating it"
                )
                # If it exists, make sure we merge rather than overwrite it
                if isinstance(existing_config, dict):
                    existing_config.update(config_data)
                    store_result = prisma_client.insert_webhook_result(
                        prediction.id, existing_config
                    )
                else:
                    # If it's not a dict, just overwrite
                    store_result = prisma_client.insert_webhook_result(
                        prediction.id, config_data
                    )
            else:
                # If it doesn't exist, create a new record
                store_result = prisma_client.insert_webhook_result(
                    prediction.id, config_data
                )

            if store_result:
                logger.info(
                    f"Successfully stored configuration with spreadsheet_id: {spreadsheet_id}"
                )
            else:
                logger.warning("Failed to store configuration - trying direct approach")
                # Try a more direct approach
                minimal_config = {
                    "spreadsheet_id": str(spreadsheet_id),
                    "user_id": user_id,
                }
                prisma_client.insert_webhook_result(prediction.id, minimal_config)
        except Exception as e:
            logger.error(f"Error storing configuration: {e}")
            # Create a minimal configuration record
            try:
                minimal_config = {"spreadsheet_id": str(spreadsheet_id)}
                prisma_client.insert_webhook_result(prediction.id, minimal_config)
                logger.info("Created minimal configuration after error")
            except Exception as inner_e:
                logger.error(f"Failed to create minimal configuration: {inner_e}")
                # Continue anyway to provide the prediction ID to the client

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
        try:
            prediction = replicate.predictions.get(prediction_id)
            if not prediction:
                logger.warning(f"Prediction {prediction_id} not found in Replicate")
                return create_error_response("Prediction not found", 404)
        except Exception as e:
            logger.error(f"Error fetching prediction from Replicate: {e}")
            return create_error_response(f"Error fetching prediction: {str(e)}", 500)

        # Get basic prediction status
        status = prediction.status
        logger.info(f"Prediction {prediction_id} status: {status}")

        # Get config data for this prediction if it exists
        config_data = prisma_client.get_webhook_result(prediction_id)
        logger.info(
            f"Raw config data: {json.dumps(config_data) if config_data else 'None'}"
        )

        if not config_data:
            config_data = {}

        # Ensure config_data is a dictionary
        if not isinstance(config_data, dict):
            logger.warning(f"Non-dictionary config data retrieved: {type(config_data)}")
            try:
                # Try to convert to dictionary if it's string JSON
                if isinstance(config_data, str):
                    config_data = json.loads(config_data)
                else:
                    config_data = {}
            except Exception as json_error:
                logger.error(
                    f"Error converting config data to dictionary: {json_error}"
                )
                config_data = {}

        # Get spreadsheet_id from config_data, handling different possible locations
        spreadsheet_id = None

        # Try direct access first
        if "spreadsheet_id" in config_data:
            spreadsheet_id = config_data.get("spreadsheet_id")
            logger.info(
                f"Found spreadsheet_id directly in config_data: {spreadsheet_id}"
            )

        # Try nested in results
        elif isinstance(config_data, dict) and "results" in config_data:
            if (
                isinstance(config_data["results"], dict)
                and "spreadsheet_id" in config_data["results"]
            ):
                spreadsheet_id = config_data["results"].get("spreadsheet_id")
                logger.info(f"Found spreadsheet_id in nested results: {spreadsheet_id}")
                # Update config_data to use the nested structure
                config_data = config_data["results"]

        # Try nested in config
        elif isinstance(config_data, dict) and "config" in config_data:
            if (
                isinstance(config_data["config"], dict)
                and "spreadsheet_id" in config_data["config"]
            ):
                spreadsheet_id = config_data["config"].get("spreadsheet_id")
                logger.info(f"Found spreadsheet_id in config section: {spreadsheet_id}")

        # Look for spreadsheetId (camelCase variant)
        elif "spreadsheetId" in config_data:
            spreadsheet_id = config_data.get("spreadsheetId")
            logger.info(
                f"Found spreadsheetId (camelCase) in config_data: {spreadsheet_id}"
            )

        # Final fallback - try to search all nested dictionaries for spreadsheet_id
        if not spreadsheet_id:

            def search_dict_for_key(d, target_key):
                if isinstance(d, dict):
                    for k, v in d.items():
                        if k == target_key:
                            return v
                        if isinstance(v, (dict, list)):
                            result = search_dict_for_key(v, target_key)
                            if result:
                                return result
                elif isinstance(d, list):
                    for item in d:
                        if isinstance(item, (dict, list)):
                            result = search_dict_for_key(item, target_key)
                            if result:
                                return result
                return None

            spreadsheet_id = search_dict_for_key(
                config_data, "spreadsheet_id"
            ) or search_dict_for_key(config_data, "spreadsheetId")
            if spreadsheet_id:
                logger.info(
                    f"Found spreadsheet_id through deep search: {spreadsheet_id}"
                )
            else:
                logger.warning(
                    f"No spreadsheet_id found anywhere in config for prediction {prediction_id}"
                )

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

            # Get embeddings from prediction output
            # The multilingual-e5-large model returns embeddings directly in an array of arrays
            embeddings = np.array(prediction.output, dtype=np.float32)
            logger.info(
                f"Extracted {len(embeddings)} embeddings from prediction output"
            )

            # Process based on whether this is a training or classification prediction
            if "embeddings_data" in config_data:  # Training prediction
                try:
                    sheet_id = config_data.get("spreadsheet_id")
                    if not sheet_id:
                        return create_error_response(
                            "Missing spreadsheet_id in config", 500
                        )

                    # Store the embeddings
                    store_result = store_embeddings(embeddings, f"{sheet_id}")
                    logger.info(f"Stored embeddings for {sheet_id}: {store_result}")

                    # Store success result
                    result = {
                        "user_id": config_data.get("user_id"),
                        "status": "success",
                        "embeddings_shape": str(embeddings.shape),
                        "spreadsheet_id": sheet_id,
                    }
                    prisma_client.insert_webhook_result(prediction.id, result)

                    return jsonify(
                        {
                            "status": "completed",
                            "message": "Training completed successfully",
                            "spreadsheet_id": sheet_id,
                        }
                    )

                except Exception as e:
                    logger.error(f"Error processing training results: {e}")
                    return create_error_response(
                        f"Error processing training results: {str(e)}", 500
                    )

            # Handle classification prediction (existing code)
            elif spreadsheet_id:
                try:
                    user_id = config_data.get("user_id", "unknown")

                    # Get original descriptions from config data
                    original_descriptions = []
                    if "original_descriptions" in config_data:
                        original_descriptions = config_data.get(
                            "original_descriptions", []
                        )
                    elif isinstance(config_data, dict) and "data" in config_data:
                        if isinstance(config_data["data"], list):
                            original_descriptions = [
                                item.get("narrative", "")
                                for item in config_data["data"]
                                if "narrative" in item
                            ]

                    logger.info(
                        f"Processing results with config: spreadsheet_id={spreadsheet_id}, user_id={user_id}, descriptions={len(original_descriptions)}"
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

                    # Process embeddings from Replicate - ensure output is valid
                    if (
                        not isinstance(prediction.output, list)
                        or len(prediction.output) == 0
                    ):
                        logger.warning(
                            f"Prediction output is not in expected format: {type(prediction.output)}"
                        )
                        return create_error_response(
                            "Prediction output is not in expected format", 500
                        )

                    # Extract embeddings from the prediction output
                    # The multilingual-e5-large model returns embeddings directly in an array of arrays
                    new_embeddings = np.array(prediction.output, dtype=np.float32)
                    logger.info(
                        f"Extracted {len(new_embeddings)} embeddings from prediction output"
                    )

                    # Calculate similarities
                    similarities = cosine_similarity(new_embeddings, trained_embeddings)
                    best_matches = similarities.argmax(axis=1)

                    # Get predicted categories and confidence scores
                    results = []
                    for i, idx in enumerate(best_matches):
                        try:
                            # Extract the category directly from the structured array
                            category = "Unknown"

                            # Try different field names and positions to get the category
                            if "category" in trained_data.dtype.names:
                                category = str(trained_data[idx]["category"])
                            elif "Category" in trained_data.dtype.names:
                                category = str(trained_data[idx]["Category"])
                            else:
                                # Fallback to index 2 which should be the category field
                                try:
                                    category = str(trained_data[idx][2])
                                except:
                                    logger.warning(
                                        f"Failed to extract category from index 2, using default"
                                    )

                            similarity_score = float(similarities[i][idx])

                            # Get original description or empty string if index is out of range
                            narrative = (
                                original_descriptions[i]
                                if i < len(original_descriptions)
                                else ""
                            )

                            results.append(
                                {
                                    "predicted_category": category,
                                    "similarity_score": similarity_score,
                                    "narrative": narrative,
                                }
                            )
                        except Exception as e:
                            logger.error(f"Error processing prediction {i}: {str(e)}")
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
                    store_result = prisma_client.insert_webhook_result(
                        prediction.id, result_data
                    )
                    if not store_result:
                        logger.warning(
                            f"Failed to store results for prediction {prediction_id}"
                        )
                        # Try a direct update as a fallback
                        try:
                            logger.info("Attempting direct update as fallback")
                            # Make sure spreadsheet_id is explicitly included again for safety
                            if (
                                "config" in result_data
                                and "spreadsheet_id" not in result_data["config"]
                            ):
                                result_data["config"]["spreadsheet_id"] = spreadsheet_id

                            # Try a simpler structure if needed
                            simple_result = {
                                "status": "success",
                                "spreadsheet_id": spreadsheet_id,
                                "results": result_data["data"],
                            }
                            prisma_client.insert_webhook_result(
                                prediction.id, simple_result
                            )
                        except Exception as fallback_error:
                            logger.error(
                                f"Fallback storage also failed: {fallback_error}"
                            )

                    # Return the results
                    return jsonify(
                        {
                            "status": "completed",
                            "message": "Processing completed successfully",
                            "results": results,
                            "config": {
                                "categoryColumn": config_data.get("category_column"),
                                "startRow": config_data.get("start_row"),
                                "sheetName": config_data.get("sheet_name"),
                                "spreadsheetId": spreadsheet_id,
                            },
                        }
                    )
                except Exception as e:
                    logger.error(f"Error processing classification results: {e}")
                    return create_error_response(
                        f"Error processing results: {str(e)}", 500
                    )
            else:
                # If we don't have config data with spreadsheet_id but have output, return raw output
                logger.warning(
                    f"No spreadsheet_id found for prediction {prediction_id}"
                )
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


@app.route("/user-config", methods=["GET"])
@require_api_key
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
            else:
                return create_error_response("Missing userId parameter", 400)

        if not user_id:
            return create_error_response("Missing userId parameter", 400)

        # If user_id is an email, prefix it with google-oauth2|
        if "@" in user_id and not user_id.startswith("google-oauth2|"):
            user_id = f"google-oauth2|{user_id}"

        # Track API usage (already handled by decorator)
        response = prisma_client.get_account_by_user_id(user_id)
        if not response:
            return create_error_response("User not found", 404)

        # Update API key if it changed
        if request.api_key != response.get("api_key"):
            prisma_client.update_account(user_id, {"api_key": request.api_key})
            logger.info(f"Updated API key for user {user_id}")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error getting user config: {e}")
        return create_error_response(str(e), 500)


@app.route("/api-usage", methods=["GET"])
@require_api_key
def get_api_usage():
    """Get API usage statistics for an account."""
    try:
        # Get user_id from request context (set by decorator)
        user_id = request.user_id
        logger.info(f"Getting API usage stats for user: {user_id}")

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

        # Log validation errors
        logger.warning(f"Validation error: {error_details}")

        # Create error response using our helper function
        return None, create_error_response(
            message="Validation error", status_code=400, details=error_details
        )


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
    # Log the error
    logger.error(f"Error response being generated: {message} (Code: {status_code})")

    # Create error response object
    error_response = {"status": "error", "error": message, "code": status_code}

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
