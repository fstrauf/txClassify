import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import replicate
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
import sys
import time
from typing import List, Optional, Union, Dict, Any
from utils.prisma_client import prisma_client
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator
from functools import wraps
import io
from flasgger import Swagger, swag_from

# Define global constants for Replicate model
REPLICATE_MODEL_NAME = "beautyyuyanli/multilingual-e5-large"
REPLICATE_MODEL_VERSION = (
    "a06276a89f1a902d5fc225a9ca32b6e8e6292b7f3b136518878da97c458e2bad"
)
EMBEDDING_DIMENSION = 1024  # Embedding dimension for the model


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
    expenseSheetId: Optional[str] = None
    userId: Optional[str] = None

    @field_validator("transactions")
    @classmethod
    def transactions_not_empty(cls, v):
        if not v or len(v) < 10:
            raise ValueError("At least 10 valid transactions required for training")
        return v


class TransactionInput(BaseModel):
    description: str
    money_in: Optional[bool] = (
        None  # True for income/credit/positive transactions, False for expense/debit/negative transactions
    )


class ClassifyRequest(BaseModel):
    """Request model for the /classify endpoint."""

    transactions: List[Union[str, TransactionInput]]
    spreadsheetId: Optional[str] = None
    sheetName: Optional[str] = "new_transactions"
    categoryColumn: Optional[str] = "E"
    startRow: Optional[str] = "1"

    @field_validator("transactions")
    @classmethod
    def transactions_not_empty(cls, v):
        if not v:
            raise ValueError("At least one transaction is required")
        for tx in v:
            if isinstance(tx, dict) and "description" not in tx:
                raise ValueError("Transaction objects must have a 'description' field")
        return v


class UserConfigRequest(BaseModel):
    """Request model for the /user-config endpoint."""

    userId: str = Field(..., min_length=1)
    apiKey: Optional[str] = None


# Configure logging
logging.basicConfig(
    stream=sys.stdout,  # Log to stdout for Docker/Gunicorn to capture
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure Swagger
# List of endpoints to exclude from Swagger documentation
EXCLUDED_SWAGGER_ENDPOINTS = ["/clean_text", "/api-usage", "/user-config"]

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: not any(
                rule.rule.startswith(endpoint)
                for endpoint in EXCLUDED_SWAGGER_ENDPOINTS
            ),
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs",
}

swagger_template = {
    "info": {
        "title": "Transaction Classification API",
        "description": "API for training and classifying financial transactions",
        "version": "1.0.0",
        "contact": {"email": "support@txclassify.com"},
    },
    "securityDefinitions": {
        "ApiKeyAuth": {"type": "apiKey", "name": "X-API-Key", "in": "header"}
    },
    "security": [{"ApiKeyAuth": []}],
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

# Define backend API URL for webhooks
BACKEND_API = os.environ.get("BACKEND_API", "http://localhost:5001")
logger.info(f"Backend API URL: {BACKEND_API}")

# Log startup information
logger.info("=== Main Application Starting ===")
logger.info(f"Environment: {os.environ.get('FLASK_ENV', 'production')}")
logger.info(
    f"Using embedding model: {REPLICATE_MODEL_NAME}, version: {REPLICATE_MODEL_VERSION[:8]}..."
)

# Connect to the database with a simple retry mechanism
connected = False
max_retries = 3
retry_count = 0

while not connected and retry_count < max_retries:
    try:
        logger.info(f"Connecting to database (attempt {retry_count + 1}/{max_retries})")
        connected = prisma_client.connect()
        logger.info("Successfully connected to database on startup")
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
@swag_from(
    {
        "responses": {
            200: {
                "description": "API is online and ready to use",
                "examples": {
                    "application/json": {
                        "status": "online",
                        "version": "1.0.0",
                        "endpoints": [
                            "/classify",
                            "/train",
                            "/health",
                            "/status/:prediction_id",
                        ],
                    }
                },
            }
        }
    }
)
def home():
    """Home endpoint - API information"""
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
@swag_from(
    {
        "responses": {
            200: {
                "description": "Health check status",
                "examples": {
                    "application/json": {
                        "status": "healthy",
                        "timestamp": "2023-01-01T00:00:00.000Z",
                        "components": {"app": "healthy", "database": "healthy"},
                    }
                },
            }
        }
    }
)
def health():
    """Health check endpoint to verify the API is functioning correctly"""
    logger.info("Health check endpoint accessed")

    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {"app": "healthy"},
    }

    # Check database connection
    try:
        # Use db to check database connection
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
    """Validate API key and return user ID if valid.

    Args:
        api_key: API key to validate
        track_usage: Whether to track API usage for this validation

    Returns:
        str: User ID if valid, empty string if invalid
    """
    if not api_key:
        logger.error("Empty API key provided")
        return ""

    try:
        # Clean the API key
        api_key = api_key.strip()

        # Use Prisma client to find user by API key
        user = prisma_client.get_user_by_api_key(api_key)

        if not user:
            logger.error("Invalid API key provided")
            return ""

        # Log the found user data (excluding sensitive info)
        logger.info(f"API key validated for userId: {str(user['id'])}")

        if not user["id"]:
            logger.error("User data found but missing id")
            return ""

        # Track API usage on successful validation if requested
        if track_usage:
            try:
                prisma_client.track_api_usage(api_key)
            except Exception as tracking_error:
                # Don't fail the validation if tracking fails
                logger.warning(f"Error tracking API usage: {tracking_error}")

        return user["id"]

    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
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
    """Clean transaction description text while preserving core business information."""
    # Convert to string and strip whitespace
    text = str(text).strip().upper()

    # Remove common payment method patterns
    payment_patterns = [
        r"\s*(?:TAP AND PAY|CONTACTLESS|PIN PURCHASE|EFTPOS|DEBIT CARD|CREDIT CARD)",
        r"\s*(?:PURCHASE|PAYMENT|TRANSFER|DIRECT DEBIT)",
        r"\s*(?:APPLE PAY|GOOGLE PAY|SAMSUNG PAY)",
        r"\s*(?:POS\s+PURCHASE|POS\s+PAYMENT)",
    ]
    for pattern in payment_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove transaction metadata
    patterns = [
        # Remove dates in various formats
        r"\s*\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}",
        r"\s*\d{2,4}[-/.]\d{1,2}[-/.]\d{1,2}",
        # Remove times
        r"\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?",
        # Remove card numbers and references
        r"\s*(?:CARD|REF|REFERENCE|TXN)\s*[#]?\s*[xX*]?\d+",
        r"\s*\d{4}[xX*]+\d{4}",
        # Remove amounts and currency
        r"\s*\$?\d+\.\d{2}\s*(?:AUD|USD|EUR|GBP)?",
        r"\s*(?:AUD|USD|EUR|GBP)\s*\$?\d+\.\d{2}",
        # Remove common transaction IDs and references
        r"\s*(?:ID|REF|REFERENCE|TXN|TRANS|INV|INVOICE)\s*[:#]?\s*\d+",
        r"\s*\d{6,}",  # Remove long numbers (likely reference numbers)
        # Remove dates and timestamps in various formats
        r"\s*\d{8,14}",  # YYYYMMDD, YYYYMMDDHHmm, etc.
        # Remove common location/terminal identifiers
        r"\s+(?:T/C|QPS|AU|NS|TERMINAL|TID|MID)\s*[:#]?\s*\d*",
        # Remove store/branch numbers
        r"\s+(?:STORE|BRANCH|LOCATION|LOC)\s*[:#]?\s*\d+",
        r"\s+#\s*\d+",
        r"\s+\d{2,4}(?:\s|$)",  # Standalone 2-4 digit numbers (likely store numbers)
        # Remove business suffixes
        r"\s+(?:PTY\s*LTD|P/?L|LIMITED|AUSTRALIA(?:N)?|CORPORATION|CORP|INC|LLC)",
        # Remove common prefixes
        r"^(?:SQ|LIV|SMP|MWA|EZI|SP|PP)\s*[\*#]?\s*",
        # Remove transaction types
        r"^(?:POS|ATM|DD|SO|BP|AP)\s+",
        r"^(?:CRED\s+VOUCHER|PENDING|RETURN|REFUND|CREDIT|DEBIT)\s+",
        # Remove anything in parentheses or brackets
        r"\s*\([^)]*\)",
        r"\s*\[[^\]]*\]",
        # Remove URLs and email addresses
        r"\s*(?:WWW|HTTP|HTTPS).*$",
        r"\s*\S+@\S+\.\S+",
        # Remove state/country codes at the end
        r"\s+(?:NSW|VIC|QLD|SA|WA|NT|ACT|TAS|AUS|USA|UK|NZ)$",
        # Remove extra spaces between digits
        r"(\d)\s+(\d)",
        # Remove special characters and multiple spaces
        r"[^\w\s-]",
    ]

    # Apply patterns one by one
    for pattern in patterns:
        text = re.sub(
            pattern, r"\1\2" if r"\1" in pattern else "", text, flags=re.IGNORECASE
        )

    # Remove extra whitespace and normalize spaces
    text = " ".join(text.split())

    # Trim long transaction names
    words = text.split()
    if len(words) > 4:  # If more than 4 words
        text = " ".join(words[:3])

    # Remove any remaining noise words at the end
    noise_words = {
        "VALUE",
        "DATE",
        "DIRECT",
        "DEBIT",
        "CREDIT",
        "CARD",
        "PAYMENT",
        "PURCHASE",
    }
    words = text.split()
    if len(words) > 1 and words[-1] in noise_words:
        text = " ".join(words[:-1])

    return text.strip()


def run_prediction(descriptions: list) -> dict:
    """Run prediction using Replicate API.

    Args:
        descriptions: List of descriptions to process
    """
    try:
        start_time = time.time()
        total_items = len(descriptions)
        logger.info(
            f"Starting prediction with model: {REPLICATE_MODEL_NAME} for {total_items} items"
        )

        # Clean descriptions and ensure they are strings
        cleaned_descriptions = [str(desc).strip() for desc in descriptions]

        model = replicate.models.get(REPLICATE_MODEL_NAME)
        version = model.versions.get(REPLICATE_MODEL_VERSION)

        # Format input exactly as shown in documentation
        # Note: The API expects 'texts' to be a JSON string of a list of strings
        input_texts = json.dumps(cleaned_descriptions)
        prediction = replicate.predictions.create(
            version=version,
            input={
                "texts": input_texts,
                "batch_size": 32,
                "normalize_embeddings": True,
            },
        )

        if not prediction or not prediction.id:
            raise Exception("Failed to create prediction (no prediction ID returned)")

        logger.info(
            f"Prediction created with ID: {prediction.id} (time: {time.time() - start_time:.2f}s)"
        )
        return prediction

    except Exception as e:
        logger.error(f"Prediction creation failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        if hasattr(e, "response"):
            logger.error(
                f"Response status: {e.response.status_code if hasattr(e.response, 'status_code') else 'unknown'}"
            )
            logger.error(
                f"Response content: {e.response.text if hasattr(e.response, 'text') else 'unknown'}"
            )
        raise


def store_embeddings(data: np.ndarray, embedding_id: str, user_id: str) -> bool:
    """Store embeddings in database with quantization, including userId."""
    try:
        logger.info(
            f"Storing embeddings with ID: {embedding_id}, shape: {data.shape}, for User: {user_id}"
        )
        start_time = time.time()

        # Prepare data before database operation
        buffer = io.BytesIO()

        # Handle structured arrays (like index data) differently
        if data.dtype.names is not None:
            dtype_dict = {
                "names": data.dtype.names,
                "formats": [
                    data.dtype.fields[name][0].str for name in data.dtype.names
                ],
            }
            metadata = {"is_structured": True, "dtype": dtype_dict, "shape": data.shape}
            np.savez(buffer, metadata=metadata, data=data.tobytes())
        else:
            # For regular arrays (embeddings), use quantization
            if data.dtype == np.float32:
                # Avoid division by zero if data is all zeros
                abs_max = np.max(np.abs(data))
                scale = (
                    abs_max / 127 if abs_max > 0 else 1.0
                )  # Default scale to 1 if max is 0
                quantized_data = (data / scale).astype(np.int8)
                metadata = {
                    "is_structured": False,
                    "scale": scale,
                    "shape": data.shape,
                    "quantized": True,
                }
                np.savez_compressed(buffer, metadata=metadata, data=quantized_data)
            else:
                metadata = {
                    "is_structured": False,
                    "quantized": False,
                    "shape": data.shape,
                }
                np.savez_compressed(buffer, metadata=metadata, data=data)

        data_bytes = buffer.getvalue()
        buffer.close()

        # Store in database with retry logic
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                # Correctly call the singular method name in prisma_client
                result = prisma_client.store_embedding(
                    embedding_id, data_bytes, user_id
                )

                # Removed the redundant tracking code below, as userId is stored directly

                logger.info(f"Embeddings stored in {time.time() - start_time:.2f}s")
                return result

            except Exception as e:
                if "connection pool" in str(e).lower() or "timeout" in str(e).lower():
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Database operation failed (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        continue
                # Log the error before re-raising or returning False
                logger.error(
                    f"Attempt {attempt + 1} failed to store embedding {embedding_id}: {e}"
                )
                # Re-raise the exception on the last attempt or if it's not a pool/timeout error
                if attempt == max_retries - 1 or not (
                    "connection pool" in str(e).lower() or "timeout" in str(e).lower()
                ):
                    raise

        # If loop completes without success (should only happen if pool/timeout errors persist)
        return False

    except Exception as e:
        # Catch potential errors during data prep or final re-raise from loop
        logger.error(f"Error storing embedding {embedding_id}: {e}", exc_info=True)
        return False


def fetch_embeddings(embedding_id: str) -> np.ndarray:
    """Fetch embeddings from database."""
    try:
        start_time = time.time()
        logger.info(f"Fetching embeddings with ID: {embedding_id}")

        # Fetch from database with retry logic
        max_retries = 3
        retry_delay = 1  # seconds
        data_bytes = None

        for attempt in range(max_retries):
            try:
                data_bytes = prisma_client.fetch_embedding(embedding_id)
                break
            except Exception as e:
                if "connection pool" in str(e).lower() or "timeout" in str(e).lower():
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Database fetch failed (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                raise

        if not data_bytes:
            logger.warning(f"No embeddings found with ID: {embedding_id}")
            return np.array([])

        # Convert bytes back to numpy array
        try:
            buffer = io.BytesIO(data_bytes)
            with np.load(buffer, allow_pickle=True) as loaded:
                metadata = loaded["metadata"].item()
                data = loaded["data"]

                if metadata.get("is_structured", False):
                    # Reconstruct structured array
                    dtype_dict = metadata["dtype"]
                    dtype = np.dtype(
                        {"names": dtype_dict["names"], "formats": dtype_dict["formats"]}
                    )
                    embeddings = np.frombuffer(data, dtype=dtype)
                    if len(metadata["shape"]) > 1:
                        embeddings = embeddings.reshape(metadata["shape"])
                else:
                    # Handle regular arrays
                    if metadata.get("quantized", False):
                        scale = metadata["scale"]
                        embeddings = data.astype(np.float32) * scale
                    else:
                        embeddings = data

            buffer.close()
            logger.info(
                f"Embeddings loaded in {time.time() - start_time:.2f}s, shape: {embeddings.shape}"
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error loading numpy array from bytes: {e}")
            return np.array([])

    except Exception as e:
        logger.error(f"Error fetching embeddings: {e}")
        return np.array([])


def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")

        if not api_key:
            logger.error("No API key provided in request headers")
            return create_error_response("No API key provided", 401)

        user_id = validate_api_key(api_key)
        if not user_id:
            logger.error("Invalid API key provided")
            return create_error_response("Invalid API key", 401)

        # Add user_id to request context
        request.user_id = user_id
        request.api_key = api_key

        return f(*args, **kwargs)

    return decorated_function


@app.route("/train", methods=["POST"])
@require_api_key
@swag_from(
    {
        "tags": ["Training"],
        "summary": "Train transaction classifier",
        "description": "Trains the model with new transaction data",
        "parameters": [
            {
                "name": "X-API-Key",
                "in": "header",
                "type": "string",
                "required": True,
                "description": "API Key for authentication",
            },
            {
                "name": "body",
                "in": "body",
                "required": True,
                "schema": {
                    "type": "object",
                    "required": ["transactions", "expenseSheetId"],
                    "properties": {
                        "transactions": {
                            "type": "array",
                            "description": "List of transactions to train with",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {"type": "string"},
                                    "Category": {"type": "string"},
                                },
                            },
                        },
                        "expenseSheetId": {"type": "string"},
                        "userId": {"type": "string"},
                    },
                },
            },
        ],
        "responses": {
            200: {
                "description": "Training started successfully",
                "examples": {
                    "application/json": {
                        "status": "processing",
                        "prediction_id": "abcd1234",
                        "message": "Training started. Check status endpoint for updates.",
                    }
                },
            },
            400: {"description": "Invalid request data"},
            401: {"description": "Invalid or missing API key"},
            500: {"description": "Server error"},
        },
    }
)
def train_model():
    """Train the model with new data."""
    try:
        start_time = time.time()
        logger.info("=== Incoming Training Request ===")

        # Get request data
        data = request.get_json()
        if not data:
            return create_error_response("Missing request data", 400)

        # Validate request data
        validated_data, error_response = validate_request_data(TrainRequest, data)
        if error_response:
            return error_response

        # Extract validated data
        transactions = validated_data.transactions
        # Use user_id as the main identifier
        sheet_id = validated_data.expenseSheetId or f"user_{request.user_id}"
        user_id = request.user_id
        logger.info(
            f"Training request received - User: {user_id}, Items: {len(transactions)}"
        )

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
            else:
                if request.api_key and (
                    not account.get("api_key")
                    or account.get("api_key") != request.api_key
                ):
                    prisma_client.update_account(user_id, {"api_key": request.api_key})

        except Exception as e:
            logger.warning(f"Error managing user configuration: {str(e)}")

        # Convert transactions to DataFrame
        transactions_data = [t.model_dump() for t in transactions]
        df = pd.DataFrame(transactions_data)

        # Clean descriptions
        df["description"] = df["description"].apply(clean_text)
        df = df.drop_duplicates(subset=["description"])

        # Store training data index with proper dtype
        df["item_id"] = range(len(df))

        # Log the categories we're training with
        unique_categories = df["Category"].unique().tolist()
        logger.info(
            f"Training with {len(df)} transactions across {len(unique_categories)} categories"
        )

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
                ("description", "U256"),
                ("category", "U128"),
            ],
        )

        # Store index data
        store_embeddings(index_data, f"{user_id}_index", user_id)
        logger.info(f"Stored index data with {len(index_data)} entries")

        # Create a placeholder for the embeddings
        placeholder = np.array([[0.0] * EMBEDDING_DIMENSION])
        store_embeddings(placeholder, f"{user_id}", user_id)

        # Get descriptions for embedding
        descriptions = df["description"].tolist()

        # Create prediction using the run_prediction function
        prediction = run_prediction(descriptions)

        # Store initial configuration for status endpoint
        context = {
            "user_id": user_id,
            "status": "processing",
            "type": "training",
            "transaction_count": len(transactions),
            "created_at": datetime.now().isoformat(),
        }

        # Store the configuration
        try:
            prisma_client.insert_webhook_result(prediction.id, context)
            logger.info(
                f"Stored initial training context for prediction {prediction.id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to store initial context for training prediction {prediction.id}: {e}",
                exc_info=True,
            )
            # If storing context fails, we might still proceed but log it.
            # Depending on requirements, you might want to return an error here.

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"Training request processed in {elapsed_time:.2f}s")

        return (
            jsonify(
                {
                    "status": "processing",
                    "prediction_id": prediction.id,
                    "message": "Training started. Check status endpoint for updates.",
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error in train_model: {e}", exc_info=True)
        return create_error_response(str(e), 500)


@app.route("/classify", methods=["POST"])
@require_api_key
@swag_from(
    {
        "tags": ["Classification"],
        "summary": "Classify transactions",
        "description": "Classifies a list of transaction descriptions",
        "parameters": [
            {
                "name": "X-API-Key",
                "in": "header",
                "type": "string",
                "required": True,
                "description": "API Key for authentication",
            },
            {
                "name": "body",
                "in": "body",
                "required": True,
                "schema": {
                    "type": "object",
                    "required": ["transactions"],
                    "properties": {
                        "transactions": {
                            "type": "array",
                            "description": "List of transaction descriptions to classify",
                            "items": {"type": "string"},
                        }
                    },
                },
            },
        ],
        "responses": {
            200: {
                "description": "Classification started successfully",
                "examples": {
                    "application/json": {
                        "status": "processing",
                        "prediction_id": "abcd1234",
                        "message": "Classification started. Check status endpoint for updates.",
                    }
                },
            },
            400: {"description": "Invalid request data"},
            401: {"description": "Invalid or missing API key"},
            500: {"description": "Server error"},
        },
    }
)
def classify_transactions_async():
    """Starts the asynchronous classification process."""
    try:
        start_time = time.time()
        user_id = request.user_id

        # 1. Validate Request Data
        data = request.get_json()
        if not data:
            return create_error_response("No data provided", 400)
        # Use the updated ClassifyRequest model
        validated_data, error_response = validate_request_data(ClassifyRequest, data)
        if error_response:
            return error_response

        # 2. Prepare Input Data for Embedding and Context Storage
        transactions_input_for_context = []
        descriptions_to_embed = []
        for tx_input_item in validated_data.transactions:
            # Handle both string and object inputs
            if isinstance(tx_input_item, str):
                desc = tx_input_item
                money_in = None  # Mark as unknown
            else:  # It's a TransactionInput object (validated by Pydantic)
                desc = tx_input_item.description
                money_in = tx_input_item.money_in  # Could be True, False, or None

            transactions_input_for_context.append(
                {"description": desc, "money_in": money_in}
            )
            # Use original description for embedding
            descriptions_to_embed.append(desc)

        if not descriptions_to_embed:
            return create_error_response(
                "No valid transaction descriptions provided", 400
            )

        logger.info(
            f"Starting classification job for {len(transactions_input_for_context)} txns, user {user_id}"
        )

        # 3. Start Replicate Prediction (DO NOT WAIT)
        try:
            prediction = run_prediction(
                descriptions_to_embed
            )  # Gets embeddings for the descriptions
        except Exception as e:
            logger.error(f"Failed to start Replicate prediction: {e}", exc_info=True)
            return create_error_response(
                f"Failed to start embedding prediction: {str(e)}", 502
            )

        # 4. Store Context for later processing in /status
        #    Store ALL necessary info, including transactions_input, in the main record
        context = {
            "user_id": user_id,
            "status": "processing",  # Initial status stored in DB
            "type": "classification",
            "transactions_input": transactions_input_for_context,  # Store original list with flags
            "created_at": datetime.now().isoformat(),
        }
        try:
            # Store the full context in the single record associated with prediction_id
            prisma_client.insert_webhook_result(prediction.id, context)
            logger.info(f"Stored full context for prediction {prediction.id}")

        except Exception as e:
            logger.error(
                f"Failed to store context for prediction {prediction.id}: {e}",
                exc_info=True,
            )
            return create_error_response(f"Failed to store job context: {str(e)}", 500)

        # 5. Return Prediction ID Immediately
        elapsed_time = time.time() - start_time
        logger.info(
            f"/classify request handled in {elapsed_time:.2f}s, prediction {prediction.id} started."
        )

        # Return 202 Accepted status code might be more appropriate for async start
        return (
            jsonify(
                {
                    "status": "processing",
                    "prediction_id": prediction.id,
                    "message": "Classification job started. Poll the /status endpoint.",
                }
            ),
            202,
        )  # Use 202 Accepted

    except Exception as e:
        logger.error(f"Critical error in /classify start: {e}", exc_info=True)
        return create_error_response(
            f"An unexpected server error occurred: {str(e)}", 500
        )


@app.route("/status/<prediction_id>", methods=["GET"])
@require_api_key  # Secure the status endpoint
@swag_from(
    {
        "tags": ["Status"],
        "summary": "Get prediction status",
        "description": "Get the current status of a classification or training job",
        "parameters": [
            {
                "name": "prediction_id",
                "in": "path",
                "type": "string",
                "required": True,
                "description": "ID of the prediction to check",
            }
        ],
        "responses": {
            200: {
                "description": "Prediction status",
                "examples": {
                    "application/json": {
                        "status": "completed",
                        "message": "Processing completed successfully",
                        "results": [
                            {
                                "predicted_category": "Food & Dining",
                                "similarity_score": 0.95,
                                "narrative": "UBER EATS",
                            }
                        ],
                    }
                },
            },
            404: {"description": "Prediction not found or context missing"},
            500: {"description": "Server error"},
            502: {"description": "Error communicating with prediction provider"},
        },
    }
)
def get_classification_or_training_status(prediction_id):
    """Gets status for Training OR Classification jobs."""
    try:
        start_time = time.time()
        requesting_user_id = request.user_id

        # 1. Check Replicate Status First
        try:
            logger.info(f"Checking Replicate status for {prediction_id}")
            prediction = replicate.predictions.get(prediction_id)
            replicate_status = prediction.status
            logger.info(f"Replicate status for {prediction_id}: {replicate_status}")
        except Exception as e:
            logger.warning(
                f"Failed to get prediction {prediction_id} from Replicate: {e}"
            )
            # If Replicate fails, check if we already have a final status stored in *our* DB
            stored_result = prisma_client.get_webhook_result(prediction_id)
            if isinstance(stored_result, dict) and stored_result.get("status") in [
                "completed",
                "failed",
            ]:
                # Security check
                if stored_result.get("user_id") != requesting_user_id:
                    logger.warning(
                        f"User {requesting_user_id} permission denied for {prediction_id}"
                    )
                    return create_error_response(
                        "Permission denied or prediction not found", 404
                    )

                logger.info(
                    f"Returning previously stored final status for {prediction_id} after Replicate fetch error."
                )

                return jsonify(stored_result)
            else:
                # No final status in our DB and Replicate failed
                return create_error_response(
                    "Prediction not found or provider error.", 404
                )

        # 2. Handle Non-Successful Replicate Statuses
        if replicate_status == "starting" or replicate_status == "processing":
            return jsonify(
                {
                    "status": "processing",
                    "message": "Job is processing on provider.",
                    "progress": 50,
                }
            )
        elif replicate_status == "failed":
            error_message = prediction.error or "Unknown prediction error from provider"
            logger.error(
                f"Prediction {prediction_id} failed on Replicate: {error_message}"
            )
            # Store failure status in our DB
            final_db_record = {
                "status": "failed",
                "error": str(error_message),
                "user_id": requesting_user_id,  # Store requesting user ID
                "type": "unknown",  # We might not know the type if context is lost
            }
            try:
                # Attempt to fetch context to get the type, but don't fail if it's missing
                context = prisma_client.get_webhook_result(prediction_id)
                if isinstance(context, dict) and context.get("type"):
                    final_db_record["type"] = context.get("type")
            except Exception as ctx_err:
                logger.warning(
                    f"Could not fetch context to determine type for failed job {prediction_id}: {ctx_err}"
                )

            prisma_client.insert_webhook_result(prediction_id, final_db_record)
            # Return 500 or maybe 502 (Bad Gateway) since the backend provider failed?
            return jsonify(final_db_record), 500
        elif replicate_status != "succeeded":
            logger.warning(
                f"Prediction {prediction_id} has unexpected Replicate status: {replicate_status}"
            )
            # Store failure status in our DB
            final_db_record = {
                "status": "failed",
                "error": f"Unexpected prediction status from provider: {replicate_status}",
                "user_id": requesting_user_id,
            }
            prisma_client.insert_webhook_result(prediction_id, final_db_record)
            return jsonify(final_db_record), 500

        # --- If we reach here, Replicate status is "succeeded" ---
        logger.info(
            f"Prediction {prediction_id} succeeded on Replicate. Fetching context and processing..."
        )
        # Add a small delay to allow DB write from /classify to propagate
        time.sleep(1)

        process_start_time = time.time()

        # 3. Fetch Job Context from *our* Database (Crucial Step)
        try:
            context = prisma_client.get_webhook_result(prediction_id)
            if (
                not isinstance(context, dict)
                or not context.get("user_id")
                or not context.get("type")
            ):
                logger.error(
                    f"CRITICAL: Valid context for {prediction_id} not found in DB after Replicate success."
                )
                # Store a specific failure status in DB
                prisma_client.insert_webhook_result(
                    prediction_id,
                    {
                        "status": "failed",
                        "error": "Internal Error: Job context lost after successful prediction.",
                        "user_id": requesting_user_id,  # Log who requested it
                    },
                )
                # Return 500 Internal Server Error
                return create_error_response("Job context lost during processing", 500)

            job_type = context.get("type")
            user_id = context.get("user_id")

            # Security check: Ensure requesting user owns this job context
            if user_id != requesting_user_id:
                logger.error(
                    f"Permission denied: User {requesting_user_id} attempting to access job {prediction_id} owned by {user_id}"
                )
                return create_error_response("Permission denied", 403)

        except Exception as db_err:
            logger.error(
                f"CRITICAL: Failed to fetch context for {prediction_id} from DB: {db_err}",
                exc_info=True,
            )
            # Don't store failure here, as the DB error might be transient
            return create_error_response(
                "Failed to retrieve job context from database", 500
            )

        # 4. Get Embeddings from Replicate Output
        try:
            if not prediction.output or not isinstance(prediction.output, list):
                raise ValueError("Invalid or missing prediction output from Replicate")
            embeddings = np.array(prediction.output, dtype=np.float32)
        except Exception as e:
            logger.error(
                f"Failed to get/process embeddings from Replicate output for {prediction_id}: {e}"
            )
            final_db_record = {
                "status": "failed",
                "error": "Failed to process prediction output",
                "user_id": user_id,
                "type": job_type,
            }
            prisma_client.insert_webhook_result(prediction_id, final_db_record)
            return jsonify(final_db_record), 500

        # 5. Process based on Job Type
        # --- Handle TRAINING Completion ---
        if job_type == "training":
            logger.info(
                f"Processing TRAINING completion for prediction {prediction_id}, user {user_id}"
            )
            try:
                store_result = store_embeddings(embeddings, f"{user_id}", user_id)
                if not store_result:
                    # Log error but maybe don't fail the whole process?
                    # Or maybe we should? If embeddings aren't stored, training didn't really complete.
                    raise Exception("Failed to store training embeddings in database")

                final_db_record = {
                    "status": "completed",
                    "message": "Training completed successfully",
                    "user_id": user_id,
                    "type": "training",
                    "completed_at": datetime.now().isoformat(),
                }
                prisma_client.insert_webhook_result(prediction_id, final_db_record)
                logger.info(
                    f"Training job {prediction_id} processing completed in {time.time() - process_start_time:.2f}s"
                )
                return jsonify(final_db_record)
            except Exception as e:
                logger.error(
                    f"Error storing/finalizing training results for {prediction_id}: {e}",
                    exc_info=True,
                )
                final_db_record = {
                    "status": "failed",
                    "error": f"Failed to store training results: {str(e)}",
                    "user_id": user_id,
                    "type": "training",
                }
                prisma_client.insert_webhook_result(prediction_id, final_db_record)
                return jsonify(final_db_record), 500

        # --- Handle CLASSIFICATION Completion ---
        elif job_type == "classification":
            logger.info(
                f"Processing CLASSIFICATION completion for {prediction_id}, user {user_id}"
            )

            # Fetch the transactions_input directly from the context fetched earlier
            transactions_input = context.get("transactions_input")

            # Validate that transactions_input exists in the context
            if not transactions_input or not isinstance(transactions_input, list):
                logger.error(
                    f"Missing or invalid transactions_input in main context for {prediction_id}"
                )
                final_db_record = {
                    "status": "failed",
                    "error": "Internal Error: Missing transaction data for classification",
                    "user_id": user_id,
                    "type": "classification",
                }
                prisma_client.insert_webhook_result(prediction_id, final_db_record)
                return jsonify(final_db_record), 500

            logger.info(
                f"Using {len(transactions_input)} transactions from main context for {prediction_id}"
            )

            # Validate counts
            if len(embeddings) != len(transactions_input):
                logger.error(
                    f"Embedding/transaction count mismatch for {prediction_id}: {len(embeddings)} vs {len(transactions_input)}"
                )
                final_db_record = {
                    "status": "failed",
                    "error": "Internal Error: Embedding count mismatch",
                    "user_id": user_id,
                    "type": "classification",
                }
                prisma_client.insert_webhook_result(prediction_id, final_db_record)
                return jsonify(final_db_record), 500

            try:
                # Store embeddings for potential future recalculation
                embeddings_id = f"{prediction_id}_embeddings"
                if not store_embeddings(embeddings, embeddings_id, user_id):
                    logger.error(
                        f"Failed to store classification embeddings {embeddings_id} for {prediction_id}"
                    )
                    # Decide if this is fatal - maybe just warn?
                    # For now, let's make it fatal as recalculation would fail.
                    raise Exception(
                        f"Failed to store classification embeddings {embeddings_id}"
                    )

                # Run the local categorization pipeline
                initial_results = _apply_initial_categorization(
                    transactions_input, embeddings, user_id
                )
                if any(
                    "Error:" in res.get("predicted_category", "")
                    for res in initial_results
                ):
                    logger.error(
                        f"Categorization pipeline failed for {prediction_id} (likely missing training data)."
                    )
                    raise Exception("Categorization failed (model/index missing?)")

                results_after_refunds = _detect_refunds(
                    initial_results, embeddings, user_id
                )
                final_results_raw = _detect_transfers(results_after_refunds)

                # Clean results for final response
                final_results_clean = []
                for res in final_results_raw:
                    cleaned_res = res.copy()
                    final_results_clean.append(cleaned_res)

                # Store final status in DB (without results for privacy)
                db_record = {
                    "status": "completed",
                    "message": "Classification completed successfully",
                    "type": "classification",
                    "user_id": user_id,
                    "transaction_count": len(final_results_clean),
                    "embeddings_id": embeddings_id,  # Reference stored embeddings
                    "completed_at": datetime.now().isoformat(),
                }
                prisma_client.insert_webhook_result(prediction_id, db_record)

                # Return full results to client
                response_payload = {
                    "status": "completed",
                    "message": "Classification completed successfully",
                    "results": final_results_clean,
                    "type": "classification",
                }
                logger.info(
                    f"Classification job {prediction_id} processing completed in {time.time() - process_start_time:.2f}s"
                )
                return jsonify(response_payload)

            except Exception as e:
                logger.error(
                    f"Error during classification processing for {prediction_id}: {e}",
                    exc_info=True,
                )
                final_db_record = {
                    "status": "failed",
                    "error": f"Error processing classification results: {str(e)}",
                    "user_id": user_id,
                    "type": "classification",
                }
                prisma_client.insert_webhook_result(prediction_id, final_db_record)
                return jsonify(final_db_record), 500

        # --- Handle Unknown Job Type ---
        else:
            logger.error(
                f"Unknown job type '{job_type}' in context for {prediction_id}"
            )
            final_db_record = {
                "status": "failed",
                "error": f"Internal Error: Unknown job type '{job_type}'",
                "user_id": user_id,
            }
            prisma_client.insert_webhook_result(prediction_id, final_db_record)
            return jsonify(final_db_record), 500

    except Exception as e:
        logger.error(
            f"Critical error in /status endpoint for {prediction_id}: {e}",
            exc_info=True,
        )
        return create_error_response(
            f"An unexpected server error occurred: {str(e)}", 500
        )


@app.route("/user-config", methods=["GET"])
@require_api_key
@swag_from(
    {
        "tags": ["User"],
        "summary": "Get user configuration",
        "description": "Retrieve configuration for a specific user",
        "parameters": [
            {
                "name": "X-API-Key",
                "in": "header",
                "type": "string",
                "required": True,
                "description": "API Key for authentication",
            },
            {
                "name": "userId",
                "in": "query",
                "type": "string",
                "required": False,
                "description": "User ID to get configuration for",
            },
        ],
        "responses": {
            200: {
                "description": "User configuration",
                "examples": {
                    "application/json": {
                        "userId": "user123",
                        "categorisationRange": "A:Z",
                        "columnOrderCategorisation": {
                            "categoryColumn": "E",
                            "descriptionColumn": "C",
                        },
                    }
                },
            },
            400: {"description": "Missing userId parameter"},
            401: {"description": "Invalid or missing API key"},
            404: {"description": "User not found"},
            500: {"description": "Server error"},
        },
    }
)
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
@swag_from(
    {
        "tags": ["User"],
        "summary": "Get API usage statistics",
        "description": "Get API usage statistics for the authenticated user",
        "parameters": [
            {
                "name": "X-API-Key",
                "in": "header",
                "type": "string",
                "required": True,
                "description": "API Key for authentication",
            }
        ],
        "responses": {
            200: {
                "description": "API usage statistics",
                "examples": {
                    "application/json": {
                        "total_requests": 150,
                        "daily_requests": 25,
                        "weekly_requests": 75,
                        "monthly_requests": 150,
                        "last_request": "2023-01-01T00:00:00.000Z",
                    }
                },
            },
            401: {"description": "Invalid or missing API key"},
            500: {"description": "Server error"},
        },
    }
)
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


@app.route("/clean_text", methods=["POST"])
@require_api_key
@swag_from(
    {
        "tags": ["Utilities"],
        "summary": "Clean transaction descriptions",
        "description": "Clean and normalize transaction descriptions",
        "parameters": [
            {
                "name": "X-API-Key",
                "in": "header",
                "type": "string",
                "required": True,
                "description": "API Key for authentication",
            },
            {
                "name": "body",
                "in": "body",
                "required": True,
                "schema": {
                    "type": "object",
                    "required": ["descriptions"],
                    "properties": {
                        "descriptions": {
                            "type": "array",
                            "description": "List of transaction descriptions to clean",
                            "items": {"type": "string"},
                        }
                    },
                },
            },
        ],
        "responses": {
            200: {
                "description": "Cleaned descriptions",
                "examples": {
                    "application/json": {
                        "cleaned_descriptions": ["UBER EATS", "AMAZON", "NETFLIX"]
                    }
                },
            },
            400: {"description": "Invalid request data"},
            401: {"description": "Invalid or missing API key"},
            429: {"description": "Daily API limit exceeded"},
            500: {"description": "Server error"},
        },
    }
)
def clean_text_endpoint():
    """Clean a batch of transaction descriptions."""
    try:
        data = request.get_json()
        if not data or "descriptions" not in data:
            return jsonify({"error": "Missing descriptions in request"}), 400

        descriptions = data["descriptions"]
        if not isinstance(descriptions, list):
            return jsonify({"error": "Descriptions must be a list"}), 400

        # Get user_id from request context (set by decorator)
        user_id = request.user_id

        # Track API usage
        try:
            # Get current usage
            usage_stats = prisma_client.get_account_usage_stats(user_id)
            if usage_stats:
                daily_requests = usage_stats.get("daily_requests", 0)
                if daily_requests > 10000:  # Limit to 10k requests per day
                    return (
                        jsonify(
                            {
                                "error": "Daily API limit exceeded. Please try again tomorrow or contact support to increase your limit."
                            }
                        ),
                        429,
                    )
        except Exception as e:
            logger.warning(f"Error checking API usage for user {user_id}: {e}")
            # Continue processing if usage check fails

        # Process descriptions
        cleaned_descriptions = []
        invalid_descriptions = []

        for i, desc in enumerate(descriptions):
            try:
                if not isinstance(desc, str):
                    invalid_descriptions.append(
                        {
                            "index": i,
                            "description": str(desc),
                            "error": "Description must be a string",
                        }
                    )
                    cleaned_descriptions.append("")
                    continue

                if len(desc) > 500:  # Limit description length
                    invalid_descriptions.append(
                        {
                            "index": i,
                            "description": desc[:50] + "...",
                            "error": "Description too long (max 500 characters)",
                        }
                    )
                    cleaned_descriptions.append(desc[:500])
                    continue

                cleaned = clean_text(desc)
                cleaned_descriptions.append(cleaned)

            except Exception as e:
                invalid_descriptions.append(
                    {"index": i, "description": desc, "error": str(e)}
                )
                cleaned_descriptions.append("")

        response = {
            "cleaned_descriptions": cleaned_descriptions,
        }

        if invalid_descriptions:
            response["warnings"] = {
                "invalid_descriptions": invalid_descriptions,
                "message": "Some descriptions could not be processed properly",
            }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in clean_text endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500


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
        # Find webhook results older than 7 days for general results
        webhook_cutoff_date = datetime.now() - timedelta(days=7)

        # Use longer retention period (30 days) for embeddings and context data
        embeddings_cutoff_date = datetime.now() - timedelta(days=30)

        logger.info(f"Cleaning up webhook results older than {webhook_cutoff_date}")
        logger.info(f"Cleaning up embeddings older than {embeddings_cutoff_date}")

        # Clean up general webhook results
        deleted_count = prisma_client.delete_old_webhook_results(webhook_cutoff_date)
        logger.info(
            f"Cleaned up {deleted_count} webhook results older than {webhook_cutoff_date}"
        )

        # Clean up embeddings and context data (stored with different IDs)
        # This requires a new function in the prisma client that can delete entries
        # with IDs containing '_embeddings' or '_context' patterns
        try:
            deleted_embeddings = prisma_client.delete_old_embeddings_and_contexts(
                embeddings_cutoff_date
            )
            logger.info(
                f"Cleaned up {deleted_embeddings} embeddings and contexts older than {embeddings_cutoff_date}"
            )
        except Exception as e:
            logger.error(f"Error cleaning up embeddings and contexts: {e}")

    except Exception as e:
        logger.error(f"Error cleaning up old webhook results: {e}")


def _apply_initial_categorization(
    transactions_input: List[Dict[str, Any]], input_embeddings: np.ndarray, user_id: str
) -> List[Dict[str, Any]]:
    """Performs initial categorization based on similarity and money_in flag."""
    results = []
    try:
        trained_embeddings = fetch_embeddings(f"{user_id}")
        trained_data = fetch_embeddings(f"{user_id}_index")

        if trained_embeddings.size == 0 or trained_data.size == 0:
            logger.error(
                f"No training data/index found for user {user_id} during categorization"
            )
            return [
                {
                    "narrative": tx["description"],
                    "predicted_category": "Error: Model/Index not found",
                    "similarity_score": 0.0,
                    "money_in": tx.get("money_in"),
                    "amount": tx.get("amount"),  # Store amount if available
                    "adjustment_info": {"reason": "Training data or index missing"},
                }
                for tx in transactions_input
            ]

        if not trained_data.dtype.names or "category" not in trained_data.dtype.names:
            logger.error(
                f"Trained index data for user {user_id} missing 'category' field."
            )
            return [
                {
                    "narrative": tx["description"],
                    "predicted_category": "Error: Invalid Index",
                    "similarity_score": 0.0,
                    "money_in": tx.get("money_in"),
                    "amount": tx.get("amount"),  # Store amount if available
                }
                for tx in transactions_input
            ]

        similarities = cosine_similarity(input_embeddings, trained_embeddings)

        for i, tx in enumerate(transactions_input):
            money_in = tx.get("money_in")  # Could be True, False, or None
            # Removed is_expense calculation as it's no longer used for compatibility check

            current_similarities = similarities[i]
            sorted_indices = np.argsort(-current_similarities)

            best_match_idx = -1
            best_category = "Unknown"
            best_score = 0.0
            # Removed original_best_category and adjustment_reason initialization related to compatibility

            # Simplified logic: Find the top match directly
            if len(sorted_indices) > 0:
                try:
                    best_match_idx = sorted_indices[0]
                    if best_match_idx < len(trained_data):
                        best_category = str(trained_data[best_match_idx]["category"])
                        best_score = float(current_similarities[best_match_idx])
                    else:
                        logger.warning(
                            f"Best match index {best_match_idx} out of bounds for trained_data (len {len(trained_data)})"
                        )
                except IndexError:
                    logger.error(
                        f"IndexError accessing trained_data at index {best_match_idx}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing best match index {best_match_idx}: {e}"
                    )

            # Always use the best match found by similarity
            results.append(
                {
                    "narrative": tx["description"],
                    "predicted_category": best_category,
                    "similarity_score": best_score,
                    "money_in": money_in,
                    "amount": tx.get("amount"),  # Store amount if available
                    # Removed adjustment_info as compatibility adjustments are gone
                }
            )

    except Exception as e:
        logger.error(f"Error during initial categorization: {e}", exc_info=True)

    return results


def _detect_refunds(
    initial_results: List[Dict[str, Any]], input_embeddings: np.ndarray, user_id: str
) -> List[Dict[str, Any]]:
    """Identifies potential refunds among credit transactions by comparing expense category scores.
    NOTE: This function's logic is currently disabled as per user request to remove
    dependency on hardcoded EXPENSE_CATEGORIES.
    """
    logger.info("Refund detection logic is currently bypassed.")
    # Original logic removed/commented out as it relied on EXPENSE_CATEGORIES
    # try:
    #     trained_embeddings = fetch_embeddings(f"{user_id}")
    #     trained_data = fetch_embeddings(f"{user_id}_index")
    #
    #     if trained_embeddings.size == 0 or trained_data.size == 0:
    #         logger.warning(
    #             f"Cannot detect refunds, no training data/index for user {user_id}"
    #         )
    #         return initial_results
    #
    #     similarities = cosine_similarity(input_embeddings, trained_embeddings)
    #
    #     for i, result in enumerate(initial_results):
    #         # Only check transactions marked as money_in (potential credits/refunds)
    #         # And skip those already definitively classified as income types by initial step
    #         # (Although initial step no longer uses INCOME_CATEGORIES for filtering)
    #         if result["money_in"] is True:
    #             current_similarities = similarities[i]
    #             sorted_indices = np.argsort(-current_similarities)
    #
    #             best_expense_score = -1.0
    #             best_expense_category = None
    #
    #             # Find the highest scoring *expense* category for this transaction
    #             for trained_idx in sorted_indices:
    #                 try:
    #                     if trained_idx >= len(trained_data):
    #                         continue
    #                     category = str(trained_data[trained_idx]["category"])
    #
    #                     # RELIES ON HARDCODED SET:
    #                     # if category in EXPENSE_CATEGORIES and category != REFUND_CATEGORY:
    #                     #    best_expense_score = float(current_similarities[trained_idx])
    #                     #    best_expense_category = category
    #                     #    break # Found the best *expense* match
    #                 except Exception as e:
    #                     logger.warning(
    #                         f"Error checking index {trained_idx} during refund detection: {e}"
    #                     )
    #                     continue
    #
    #             # Define a threshold - how much higher must the expense score be?
    #             refund_confidence_threshold = 0.05 # Example threshold
    #
    #             # If we found an expense category & its score is significantly higher
    #             # than the currently assigned category (which might be 'Unknown' or a low-confidence match)
    #             is_potential_refund = False
    #             if best_expense_category and (
    #                 result["predicted_category"] == "Unknown" or
    #                 best_expense_score > (result["similarity_score"] + refund_confidence_threshold)
    #             ):
    #                 is_potential_refund = True
    #
    #                 # Re-categorize as the best matching *expense* category, marking as refund candidate
    #                 original_category = result["predicted_category"]
    #                 result["predicted_category"] = best_expense_category
    #                 result["similarity_score"] = best_expense_score # Update score too
    #                 reason = f"Potential refund: Matched expense category '{best_expense_category}' (score {best_expense_score:.2f}) better than original '{original_category}' (score {result['similarity_score']:.2f})"
    #
    #                 if "adjustment_info" not in result:
    #                       result["adjustment_info"] = {}
    #                 result["adjustment_info"]["is_refund_candidate"] = True
    #                 result["adjustment_info"]["refund_detection_reason"] = reason
    #                 result["adjustment_info"]["adjusted"] = True
    #
    #                 logger.info(f"Potential refund detected: '{result['narrative']}' -> '{best_expense_category}'")
    #
    # except Exception as e:
    #     logger.error(f"Error during refund detection: {e}", exc_info=True)
    # Pass results through without modification
    return initial_results


def _detect_transfers(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detects potential transfers by matching income/expense descriptions."""
    try:
        description_map = {}
        for i, res in enumerate(results):
            norm_desc = re.sub(r"\s+", " ", res["narrative"].lower().strip())
            norm_desc = re.sub(r"\b\d{4,}\b", "", norm_desc)
            norm_desc = re.sub(r"\s+", " ", norm_desc).strip()

            if not norm_desc:
                continue

            if norm_desc not in description_map:
                description_map[norm_desc] = {"income": [], "expense": []}

            if res["money_in"] is False:
                description_map[norm_desc]["expense"].append((i, res.get("amount")))
            elif res["money_in"] is True:
                if (
                    not res.get("adjustment_info", {}).get("is_refund_candidate")
                    and res["predicted_category"] != "Transfer in"
                ):
                    description_map[norm_desc]["income"].append((i, res.get("amount")))

        matched_indices = set()
        for norm_desc, groups in description_map.items():
            # Generate possible pairs considering amounts
            possible_pairs = []

            for income_entry in groups["income"]:
                income_idx, income_amount = income_entry
                if income_idx in matched_indices:
                    continue

                for expense_entry in groups["expense"]:
                    expense_idx, expense_amount = expense_entry
                    if expense_idx in matched_indices:
                        continue

                    # Skip if either amount is None
                    if income_amount is None or expense_amount is None:
                        continue

                    # Check if amounts match (opposite signs but same absolute value)
                    # Allow for small floating point differences
                    if abs(abs(income_amount) - abs(expense_amount)) < 0.01:
                        possible_pairs.append((income_idx, expense_idx))

            # Process the valid pairs
            for income_idx, expense_idx in possible_pairs:
                if income_idx in matched_indices or expense_idx in matched_indices:
                    continue

                results[income_idx]["predicted_category"] = "Transfer in"
                if "adjustment_info" not in results[income_idx]:
                    results[income_idx]["adjustment_info"] = {}
                results[income_idx]["adjustment_info"][
                    "transfer_detection_reason"
                ] = f"Paired with expense index {expense_idx} ('{results[expense_idx]['narrative']}', amount: {results[expense_idx].get('amount')})"
                results[income_idx]["adjustment_info"]["adjusted"] = True

                results[expense_idx]["predicted_category"] = "Transfer out"
                if "adjustment_info" not in results[expense_idx]:
                    results[expense_idx]["adjustment_info"] = {}
                results[expense_idx]["adjustment_info"][
                    "transfer_detection_reason"
                ] = f"Paired with income index {income_idx} ('{results[income_idx]['narrative']}', amount: {results[income_idx].get('amount')})"
                results[expense_idx]["adjustment_info"]["adjusted"] = True

                matched_indices.add(income_idx)
                matched_indices.add(expense_idx)
                logger.info(
                    f"Detected transfer pair: '{results[income_idx]['narrative']}' (amount: {results[income_idx].get('amount')}) <-> '{results[expense_idx]['narrative']}' (amount: {results[expense_idx].get('amount')})"
                )

    except Exception as e:
        logger.error(f"Error during transfer detection: {e}", exc_info=True)

    return results


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    app.run(host="0.0.0.0", port=port)
