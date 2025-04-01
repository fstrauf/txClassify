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
from typing import List, Optional
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


class ClassifyRequest(BaseModel):
    """Request model for the /classify endpoint."""

    transactions: List[str]
    spreadsheetId: Optional[str] = None
    sheetName: Optional[str] = "new_transactions"
    categoryColumn: Optional[str] = "E"
    startRow: Optional[str] = "1"

    @field_validator("transactions")
    @classmethod
    def transactions_not_empty(cls, v):
        if not v:
            raise ValueError("At least one transaction is required")
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
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
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


def store_embeddings(data: np.ndarray, embedding_id: str) -> bool:
    """Store embeddings in database with quantization."""
    try:
        logger.info(f"Storing embeddings with ID: {embedding_id}, shape: {data.shape}")
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
                scale = np.max(np.abs(data)) / 127
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
                result = prisma_client.store_embedding(embedding_id, data_bytes)

                # Try to link embedding to account if we have an API key
                try:
                    if hasattr(request, "headers") and request.headers.get("X-API-Key"):
                        api_key = request.headers.get("X-API-Key")
                        prisma_client.track_embedding_creation(
                            api_key, str(embedding_id)
                        )
                except Exception as e:
                    logger.warning(f"Failed to track embedding creation: {str(e)}")

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
                raise

        return False

    except Exception as e:
        logger.error(f"Error storing embeddings: {e}")
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

        # Get user_id from request context (set by decorator)
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
        store_embeddings(index_data, f"{user_id}_index")
        logger.info(f"Stored index data with {len(index_data)} entries")

        # Create a placeholder for the embeddings
        placeholder = np.array([[0.0] * EMBEDDING_DIMENSION])
        store_embeddings(placeholder, f"{user_id}")

        # Get descriptions for embedding
        descriptions = df["description"].tolist()

        # Create prediction using the run_prediction function
        prediction = run_prediction(descriptions)

        # Store initial configuration for status endpoint
        config_data = {
            "user_id": user_id,
            "status": "processing",
            "embeddings_data": {
                "index_data": index_data.tolist(),
                "descriptions": descriptions,
            },
        }

        # Store the configuration
        try:
            prisma_client.insert_webhook_result(prediction.id, config_data)
        except Exception as e:
            logger.error(f"Error storing initial configuration: {e}")

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"Training request processed in {elapsed_time:.2f}s")

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
def classify_transactions():
    """Classify transactions endpoint."""
    try:
        start_time = time.time()

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

        # Get user_id from request context (set by decorator)
        user_id = request.user_id

        # Verify training data exists - we can use the user_id to fetch the trained model
        # instead of relying on spreadsheet_id
        try:
            # Use user_id to fetch embeddings
            trained_data = fetch_embeddings(f"{user_id}_index")
            if len(trained_data) == 0:
                error_msg = "No training data found. Please train the model first."
                logger.error(f"{error_msg} for user: {user_id}")
                return create_error_response(error_msg, 400)
        except Exception as e:
            error_msg = f"Error accessing training data: {str(e)}"
            logger.error(f"{error_msg} for user: {user_id}")
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
        logger.info(
            f"Processing {len(cleaned_descriptions)} descriptions for classification"
        )

        # Create prediction using the model
        model = replicate.models.get(REPLICATE_MODEL_NAME)
        version = model.versions.get(REPLICATE_MODEL_VERSION)

        # Create prediction without webhook
        prediction = replicate.predictions.create(
            version=version,
            input={
                "texts": json.dumps(cleaned_descriptions),
                "batch_size": 32,
                "normalize_embeddings": True,
            },
        )

        logger.info(f"Classification prediction created with ID: {prediction.id}")

        # Store config in database for later retrieval - simplified version
        config_data = {
            "user_id": user_id,
            "original_descriptions": original_descriptions,
        }

        # Store the configuration
        try:
            prisma_client.insert_webhook_result(prediction.id, config_data)
        except Exception as e:
            logger.error(f"Error storing configuration: {e}")

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"Classification request processed in {elapsed_time:.2f}s")

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
            404: {"description": "Prediction not found"},
            500: {"description": "Server error"},
        },
    }
)
def get_prediction_status(prediction_id):
    """Get the status of a prediction directly from Replicate API."""
    try:
        start_time = time.time()
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

        # Enhanced error logging for failed predictions
        if status == "failed":
            error_msg = str(prediction.error) if prediction.error else "Unknown error"
            error_type = (
                type(prediction.error).__name__ if prediction.error else "Unknown"
            )
            logger.error(f"Prediction failed with error type {error_type}: {error_msg}")

            # Log additional prediction details
            logger.error("Prediction details:")
            logger.error(
                f"  Created at: {prediction.created_at if hasattr(prediction, 'created_at') else 'unknown'}"
            )
            logger.error(
                f"  Started at: {prediction.started_at if hasattr(prediction, 'started_at') else 'unknown'}"
            )
            logger.error(
                f"  Completed at: {prediction.completed_at if hasattr(prediction, 'completed_at') else 'unknown'}"
            )
            logger.error(
                f"  Model: {prediction.model if hasattr(prediction, 'model') else 'unknown'}"
            )
            logger.error(
                f"  Version: {prediction.version if hasattr(prediction, 'version') else 'unknown'}"
            )

            if hasattr(prediction, "logs") and prediction.logs:
                logger.error(f"Prediction logs: {prediction.logs}")

            if hasattr(prediction, "input"):
                logger.error(f"Input configuration: {prediction.input}")

            return create_error_response(f"Prediction failed: {error_msg}", 500)

        # Get config data for this prediction if it exists
        config_data = prisma_client.get_webhook_result(prediction_id)

        if not config_data:
            config_data = {}

        # If prediction failed with a retryable error, attempt to retry
        if status == "failed" and "code: pa" in str(prediction.error).lower():
            logger.info(f"Retrying interrupted prediction {prediction_id}")

            # Extract necessary data from config
            if isinstance(config_data, dict):
                if "original_descriptions" in config_data:
                    descriptions = config_data["original_descriptions"]
                elif (
                    "embeddings_data" in config_data
                    and "descriptions" in config_data["embeddings_data"]
                ):
                    descriptions = config_data["embeddings_data"]["descriptions"]
                else:
                    return create_error_response(
                        "Cannot retry: missing description data", 500
                    )

                try:
                    # Create new prediction with retry mechanism
                    new_prediction = run_prediction(descriptions)

                    # Update the webhook result with the new prediction ID
                    config_data["retried_from"] = prediction_id
                    prisma_client.insert_webhook_result(new_prediction.id, config_data)

                    return jsonify(
                        {
                            "status": "retrying",
                            "old_prediction_id": prediction_id,
                            "new_prediction_id": new_prediction.id,
                            "message": "Previous prediction failed, created new prediction",
                        }
                    )

                except Exception as retry_error:
                    logger.error(f"Error retrying prediction: {retry_error}")
                    return create_error_response(
                        f"Failed to retry prediction: {str(retry_error)}", 500
                    )

        # Ensure config_data is a dictionary
        if not isinstance(config_data, dict):
            try:
                # Try to convert to dictionary if it's string JSON
                if isinstance(config_data, str):
                    config_data = json.loads(config_data)
                else:
                    config_data = {}
            except Exception:
                config_data = {}

        # Get spreadsheet_id from config_data
        spreadsheet_id = None

        # Try direct access first
        if "spreadsheet_id" in config_data:
            spreadsheet_id = config_data.get("spreadsheet_id")
        # Try nested in results
        elif isinstance(config_data, dict) and "results" in config_data:
            if (
                isinstance(config_data["results"], dict)
                and "spreadsheet_id" in config_data["results"]
            ):
                spreadsheet_id = config_data["results"].get("spreadsheet_id")
                # Update config_data to use the nested structure
                config_data = config_data["results"]
        # Try nested in config
        elif isinstance(config_data, dict) and "config" in config_data:
            if (
                isinstance(config_data["config"], dict)
                and "spreadsheet_id" in config_data["config"]
            ):
                spreadsheet_id = config_data["config"].get("spreadsheet_id")
        # Look for spreadsheetId (camelCase variant)
        elif "spreadsheetId" in config_data:
            spreadsheet_id = config_data.get("spreadsheetId")

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
            logger.info(f"Prediction {prediction_id} succeeded, processing results")
            process_start_time = time.time()

            # If succeeded but no output, return error
            if not prediction.output:
                logger.error(f"Prediction {prediction_id} succeeded but has no output")
                return create_error_response(
                    "Prediction succeeded but has no output", 500
                )

            # Get embeddings from prediction output
            embeddings = np.array(prediction.output, dtype=np.float32)
            logger.info(f"Received {len(embeddings)} embeddings from prediction")

            # Process based on whether this is a training or classification prediction
            if "embeddings_data" in config_data:  # Training prediction
                try:
                    # Get user_id from config_data
                    user_id = config_data.get("user_id")
                    if not user_id:
                        # Fallback to the request's user_id
                        user_id = request.user_id or "unknown"
                        logger.warning(
                            f"No user_id in config, using fallback: {user_id}"
                        )

                    # Store the embeddings with user_id
                    store_result = store_embeddings(embeddings, f"{user_id}")

                    # Store success result
                    result = {
                        "user_id": user_id,
                        "status": "success",
                        "embeddings_shape": str(embeddings.shape),
                    }
                    prisma_client.insert_webhook_result(prediction.id, result)

                    # Log total execution time
                    total_time = time.time() - start_time
                    logger.info(
                        f"Status request for training processed in {total_time:.2f}s"
                    )

                    return jsonify(
                        {
                            "status": "completed",
                            "message": "Training completed successfully",
                            "user_id": user_id,
                        }
                    )

                except Exception as e:
                    logger.error(f"Error processing training results: {e}")
                    return create_error_response(
                        f"Error processing training results: {str(e)}", 500
                    )
            # Handle classification prediction
            else:
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
                        f"Processing classification results for {len(original_descriptions)} items"
                    )

                    # Get trained embeddings using user_id instead of spreadsheet_id
                    trained_embeddings = fetch_embeddings(f"{user_id}")
                    trained_data = fetch_embeddings(f"{user_id}_index")

                    # Check if we have valid embeddings
                    if trained_embeddings.size == 0 or trained_data.size == 0:
                        logger.error(
                            f"No valid embeddings or training data found for user: {user_id}"
                        )
                        return create_error_response(
                            f"No valid training data found for this user. Please train the model first.",
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
                    new_embeddings = np.array(prediction.output, dtype=np.float32)

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
                                    pass

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
                        "user_id": user_id,
                    }

                    # Store the results in the database
                    prisma_client.insert_webhook_result(prediction.id, result_data)

                    # Log total execution time
                    total_time = time.time() - start_time
                    process_time = time.time() - process_start_time
                    logger.info(
                        f"Classified {len(results)} items in {process_time:.2f}s (total: {total_time:.2f}s)"
                    )

                    # Return the results
                    return jsonify(
                        {
                            "status": "completed",
                            "message": "Processing completed successfully",
                            "results": results,
                        }
                    )
                except Exception as e:
                    logger.error(f"Error processing classification results: {e}")
                    return create_error_response(
                        f"Error processing results: {str(e)}", 500
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
        # Find webhook results older than 7 days
        cutoff_date = datetime.now() - timedelta(days=7)

        # Use Prisma client to delete old webhook results
        # This is a placeholder - implement the actual deletion logic using your Prisma client
        logger.info(f"Cleaning up webhook results older than {cutoff_date}")

        # Example implementation:
        # prisma_client.delete_old_webhook_results(cutoff_date)
        # logger.info(f"Cleaned up webhook results older than {cutoff_date}")

        # Now implemented:
        deleted_count = prisma_client.delete_old_webhook_results(cutoff_date)
        logger.info(
            f"Cleaned up {deleted_count} webhook results older than {cutoff_date}"
        )
    except Exception as e:
        logger.error(f"Error cleaning up old webhook results: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    app.run(host="0.0.0.0", port=port)
