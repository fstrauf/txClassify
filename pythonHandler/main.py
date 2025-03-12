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
import gc
import sys
import uuid
import time
from typing import List
from threading import Thread
from .utils.prisma_client import prisma_client
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Dictionary to store prediction data
predictions_db = {}

# Cache for Replicate API responses
replicate_cache = {}

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

# Connect to the database
try:
    prisma_client.connect()
    logger.info("Connected to database")
except Exception as e:
    logger.error(f"Failed to connect to database: {e}")
    logger.warning(
        "Continuing without database connection. Some features may not work correctly."
    )


@app.teardown_appcontext
def shutdown_db_connection(exception=None):
    """Disconnect from the database when the application context ends."""
    try:
        # We don't want to disconnect after each request in testing mode
        # as it causes issues with subsequent requests
        if os.environ.get("FLASK_ENV") != "testing":
            prisma_client.disconnect()
        logger.info("Disconnected from database")
    except Exception as e:
        logger.error(f"Error disconnecting from database: {e}")


@app.before_request
def ensure_db_connection():
    """Ensure database connection is active before each request."""
    try:
        # Check if client exists and is connected
        if (
            not hasattr(prisma_client.client, "_engine")
            or not prisma_client.client._engine
        ):
            logger.info("Database connection not active, reconnecting...")
            prisma_client.connect()
            logger.info("Reconnected to database")
    except Exception as e:
        logger.error(f"Failed to ensure database connection: {e}")


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


def run_prediction(
    mode: str,
    sheet_id: str,
    user_id: str,
    descriptions: list,
    sheet_name: str = None,
    category_column: str = None,
    use_webhook: bool = None,
) -> dict:
    """Run prediction using Replicate API."""
    try:
        model = replicate.models.get("replicate/all-mpnet-base-v2")
        version = model.versions.get(
            "b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305"
        )

        # Determine whether to use webhooks
        # If use_webhook is None, check environment variable
        if use_webhook is None:
            use_webhook = os.environ.get("USE_WEBHOOKS", "false").lower() == "true"

        # Set up prediction parameters
        prediction_params = {
            "version": version,
            "input": {"text_batch": json.dumps(descriptions)},
        }

        # Only add webhook if enabled and BACKEND_API is set
        if (
            use_webhook
            and BACKEND_API
            and not BACKEND_API.startswith("http://localhost")
        ):
            # Create webhook URL with all necessary parameters
            webhook_params = [f"spreadsheetId={sheet_id}", f"userId={user_id}"]
            if sheet_name:
                webhook_params.append(f"sheetName={sheet_name}")
            if category_column:
                webhook_params.append(f"categoryColumn={category_column}")

            webhook = f"{BACKEND_API}/{mode}/webhook?{'&'.join(webhook_params)}"
            logger.info(f"Using webhook URL: {webhook}")

            prediction_params["webhook"] = webhook
            prediction_params["webhook_events_filter"] = ["completed"]
        else:
            logger.info(
                "Webhooks disabled or invalid BACKEND_API. Using polling instead."
            )

        # Create the prediction
        prediction = replicate.predictions.create(**prediction_params)

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


def update_sheet_log(
    sheet_id: str, status: str, message: str, details: str = ""
) -> None:
    """Log sheet update without writing directly to the sheet."""
    try:
        # Just log the message instead of writing to the sheet
        logger.info(
            f"Sheet log update - sheet_id: {sheet_id}, status: {status}, message: {message}, details: {details}"
        )
    except Exception as e:
        logger.error(f"Error logging sheet update: {e}")


@app.route("/train", methods=["POST"])
def train_model():
    """Train the model with new data."""
    try:
        # Log all incoming request details
        logger.info("=== Incoming Training Request ===")
        logger.info(f"Headers: {dict(request.headers)}")

        # Get request data
        data = request.get_json()
        logger.info(f"Request data: {data}")

        if not data:
            logger.error("Missing request data")
            return jsonify({"error": "Missing request data"}), 400

        if not isinstance(data, dict):
            logger.error(
                f"Invalid request format - expected JSON object, got {type(data)}"
            )
            return (
                jsonify({"error": "Invalid request format - expected JSON object"}),
                400,
            )

        if "transactions" not in data:
            logger.error("Missing transactions data in request")
            return jsonify({"error": "Missing transactions data"}), 400

        # Support both parameter names for backward compatibility
        sheet_id = data.get("spreadsheetId") or data.get("expenseSheetId")
        if not sheet_id or not isinstance(sheet_id, str) or len(sheet_id.strip()) == 0:
            logger.error(f"Invalid or missing spreadsheetId: {sheet_id}")
            return jsonify({"error": "Invalid or missing spreadsheetId"}), 400

        # Get user ID either from API key validation or payload
        user_id = None
        api_key = request.headers.get("X-API-Key")

        # First try to validate API key if provided
        if api_key:
            try:
                user_id = validate_api_key(api_key)
                logger.info(f"Got user_id from API key validation: {user_id}")
            except Exception as e:
                # If API key validation fails and we're not in fallback mode, return error
                if not data.get("userId"):
                    logger.error(
                        f"API key validation failed and no userId provided in payload: {str(e)}"
                    )
                    return (
                        jsonify({"error": f"API key validation failed: {str(e)}"}),
                        401,
                    )
                logger.error(f"API key validation failed: {str(e)}")
                logger.info("Falling back to userId from payload")

        # If no user_id from API key, try payload
        if not user_id:
            user_id = data.get("userId")
            if user_id:
                # If user_id is an email, prefix it with google-oauth2|
                if "@" in user_id and not user_id.startswith("google-oauth2|"):
                    user_id = f"google-oauth2|{user_id}"
                logger.info(f"Got user_id from payload: {user_id}")
            else:
                logger.error("No valid user_id found in API key or payload")
                return (
                    jsonify(
                        {
                            "error": "No valid user ID found. Please provide either a valid API key or userId in the request."
                        }
                    ),
                    401,
                )

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

        # Convert transactions to DataFrame with error handling
        try:
            logger.info(
                f"Converting transactions to DataFrame. Sample: {data['transactions'][:2]}"
            )
            df = pd.DataFrame(data["transactions"])
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error converting transactions to DataFrame: {str(e)}")
            return jsonify({"error": "Invalid transaction data format"}), 400

        # Validate required columns
        required_columns = ["Narrative", "Category"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return (
                jsonify({"error": f"Missing required columns: {missing_columns}"}),
                400,
            )

        # Validate data quality
        df["Narrative"] = df["Narrative"].astype(str).str.strip()
        df["Category"] = df["Category"].astype(str).str.strip()

        if df["Narrative"].empty or df["Narrative"].isna().any():
            logger.error("Invalid or empty narratives found")
            return jsonify({"error": "Invalid or empty narratives found"}), 400

        if df["Category"].empty or df["Category"].isna().any():
            logger.error("Invalid or empty categories found")
            return jsonify({"error": "Invalid or empty categories found"}), 400

        # Clean descriptions
        df["description"] = df["Narrative"].apply(clean_text)
        df = df.drop_duplicates(subset=["description"])

        if len(df) < 10:  # Minimum required for meaningful training
            return (
                jsonify(
                    {"error": "At least 10 valid transactions required for training"}
                ),
                400,
            )

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

        # Determine whether to use webhooks based on environment variable
        use_webhooks = os.environ.get("USE_WEBHOOKS", "false").lower() == "true"

        # Create prediction using the run_prediction function
        prediction = run_prediction(
            mode="train",
            sheet_id=sheet_id,
            user_id=user_id,
            descriptions=descriptions,
            sheet_name=data.get("sheetName"),
            category_column=data.get("categoryColumn"),
            use_webhook=use_webhooks,
        )

        logger.info(f"Created prediction with ID: {prediction.id}")

        # Update status
        update_process_status("processing", "training", user_id)

        return jsonify({"status": "processing", "prediction_id": prediction.id})

    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/classify", methods=["POST"])
def classify_transactions():
    """Classify transactions endpoint."""
    try:
        # Validate API key
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return jsonify({"error": "Missing API key"}), 401

        # Get and validate request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate required fields
        required_fields = ["transactions", "spreadsheetId"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Extract data
        transactions = data["transactions"]
        spreadsheet_id = data["spreadsheetId"]
        sheet_name = data.get(
            "sheetName", "new_transactions"
        )  # Default to "new_transactions" if not provided
        category_column = data.get(
            "categoryColumn", "E"
        )  # Default to column E if not provided
        start_row = data.get("startRow", "1")  # Default to row 1 if not provided

        if not transactions:
            return jsonify({"error": "No transactions provided"}), 400

        # Validate API key and get user ID
        try:
            user_id = validate_api_key(api_key)
            logger.info(f"API key validated for user: {user_id}")

            # Update API key in user account if needed
            try:
                response = prisma_client.get_account_by_user_id(user_id)
                if response:
                    existing_config = response
                    if (
                        not existing_config.get("api_key")
                        or existing_config.get("api_key") != api_key
                    ):
                        prisma_client.update_account(user_id, {"api_key": api_key})
                        logger.info(f"Updated API key for user {user_id}")
            except Exception as e:
                logger.warning(f"Error updating API key in user account: {e}")
                # Continue with classification even if update fails
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return jsonify({"error": f"API key validation failed: {str(e)}"}), 401

        # Verify training data exists
        try:
            trained_data = fetch_embeddings(f"{spreadsheet_id}_index")
            if len(trained_data) == 0:
                error_msg = "No training data found. Please train the model first."
                return jsonify({"error": error_msg}), 400
        except Exception as e:
            error_msg = "No training data found. Please train the model first."
            return jsonify({"error": error_msg}), 400

        # Start classification process
        prediction_id = str(uuid.uuid4())

        # Store prediction metadata
        predictions_db[prediction_id] = {
            "status": "processing",
            "user_id": user_id,
            "spreadsheet_id": spreadsheet_id,
            "sheet_name": sheet_name,  # Store sheet name
            "category_column": category_column,  # Store category column
            "start_row": start_row,  # Store start row
            "total_transactions": len(transactions),
            "processed_transactions": 0,
        }

        # Start classification in background
        Thread(
            target=process_classification, args=(prediction_id, transactions, user_id)
        ).start()

        return jsonify({"status": "processing", "prediction_id": prediction_id})

    except Exception as e:
        logger.error(f"Error in classify_transactions: {e}")
        return jsonify({"error": str(e)}), 500


def process_classification(prediction_id: str, transactions: List[dict], user_id: str):
    """Process classification in background."""
    try:
        predictions_db[prediction_id]["status"] = "classifying"

        # Get trained embeddings and categories
        spreadsheet_id = predictions_db[prediction_id]["spreadsheet_id"]
        sheet_name = predictions_db[prediction_id].get("sheet_name", "new_transactions")

        logger.info(f"Fetching trained embeddings for spreadsheet_id: {spreadsheet_id}")
        trained_embeddings = fetch_embeddings(f"{spreadsheet_id}")
        trained_data = fetch_embeddings(f"{spreadsheet_id}_index")

        logger.info(f"Trained embeddings size: {len(trained_embeddings)}")
        logger.info(f"Trained data size: {len(trained_data)}")

        if len(trained_embeddings) == 0 or len(trained_data) == 0:
            logger.error(f"No training data found for spreadsheet_id: {spreadsheet_id}")
            logger.error(
                f"Embeddings size: {len(trained_embeddings)}, Index size: {len(trained_data)}"
            )

            # Check if the files exist in the database
            try:
                embedding_exists = (
                    prisma_client.fetch_embedding(f"{spreadsheet_id}") is not None
                )
                index_exists = (
                    prisma_client.fetch_embedding(f"{spreadsheet_id}_index") is not None
                )
                logger.error(
                    f"Embedding file exists in DB: {embedding_exists}, Index file exists in DB: {index_exists}"
                )
            except Exception as e:
                logger.error(f"Error checking if embedding files exist: {e}")

            predictions_db[prediction_id]["status"] = "failed"
            predictions_db[prediction_id]["error"] = "No training data found"
            return

        # Clean and prepare descriptions
        descriptions = []
        for t in transactions:
            if t.get("description"):
                descriptions.append(clean_text(t["description"]))
            else:
                logger.warning(f"No description field found in transaction: {t}")

        if not descriptions:
            logger.error("No valid descriptions found in transactions")
            predictions_db[prediction_id]["status"] = "failed"
            predictions_db[prediction_id][
                "error"
            ] = "No valid descriptions found in transactions"
            return

        # Get embeddings for new descriptions
        logger.info("Initializing Replicate model for embeddings")

        # Determine whether to use webhooks based on environment variable
        use_webhooks = os.environ.get("USE_WEBHOOKS", "false").lower() == "true"

        # Create prediction using the run_prediction function
        prediction = run_prediction(
            mode="classify",
            sheet_id=spreadsheet_id,
            user_id=user_id,
            descriptions=descriptions,
            sheet_name=sheet_name,
            category_column=category_column,
            use_webhook=use_webhooks,
        )

        logger.info(f"Created prediction with ID: {prediction.id}")

        # Store prediction ID for reference
        predictions_db[prediction_id]["replicate_prediction_id"] = prediction.id

        # Optimize the waiting strategy - use shorter initial checks
        attempt = 0
        max_attempts = 60  # Maximum number of attempts

        # Start with shorter checks for the first few attempts
        initial_delays = [2, 3, 5, 10, 15]  # First few checks are faster

        while prediction.status != "succeeded" and attempt < max_attempts:
            if prediction.status == "failed":
                predictions_db[prediction_id]["status"] = "failed"
                predictions_db[prediction_id]["error"] = prediction.error
                return

            # Use initial delays for first few attempts, then exponential backoff
            if attempt < len(initial_delays):
                delay = initial_delays[attempt]
            else:
                # Cap at 60 seconds for later attempts
                delay = min(20 * (1.5 ** (attempt - len(initial_delays))), 60)

            logger.info(
                f"Waiting {delay} seconds before checking prediction status again (attempt {attempt+1}/{max_attempts})"
            )
            time.sleep(delay)

            # Reload prediction status
            prediction.reload()
            attempt += 1

            # Update progress information
            predictions_db[prediction_id][
                "status_message"
            ] = f"Waiting for embeddings (attempt {attempt}/{max_attempts})"

            # If we've been waiting for a while, check if webhook results are already available
            if attempt > 5:
                try:
                    webhook_result = prisma_client.get_webhook_result(prediction_id)
                    if webhook_result:
                        logger.info(
                            f"Found webhook results while waiting for embeddings (attempt {attempt})"
                        )
                        predictions_db[prediction_id]["status"] = "completed"
                        predictions_db[prediction_id][
                            "status_message"
                        ] = "Classification completed successfully"
                        return
                except Exception as e:
                    logger.warning(
                        f"Error checking webhook results during waiting: {e}"
                    )

        if attempt >= max_attempts:
            predictions_db[prediction_id]["status"] = "failed"
            predictions_db[prediction_id]["error"] = "Timed out waiting for embeddings"
            return

        # If we're using webhooks, we don't need to process the results here
        # The webhook will handle it
        predictions_db[prediction_id]["status"] = "waiting_for_webhook"
        predictions_db[prediction_id][
            "status_message"
        ] = "Embeddings completed, waiting for webhook to process results"
        logger.info(
            f"Embeddings completed for prediction {prediction_id}, waiting for webhook to process results"
        )

    except Exception as e:
        logger.error(f"Error in process_classification: {e}")
        predictions_db[prediction_id]["status"] = "failed"
        predictions_db[prediction_id]["error"] = str(e)


@app.route("/classify/webhook", methods=["POST"])
def classify_webhook():
    """Handle classification webhook from Replicate."""
    try:
        data = request.get_json()
        sheet_id = request.args.get("spreadsheetId")
        user_id = request.args.get("userId")
        sheet_name = request.args.get("sheetName")
        start_row = request.args.get("startRow", "2")  # Default to row 2
        category_column = request.args.get("categoryColumn", "E")  # Default to column E

        # Log all parameters for debugging
        logger.info(
            f"Webhook parameters: sheet_id={sheet_id}, user_id={user_id}, sheet_name={sheet_name}, start_row={start_row}, category_column={category_column}"
        )

        if not all([data, sheet_id, user_id]):
            error_msg = "Missing required parameters: data, spreadsheetId, or userId"
            logger.error(error_msg)
            if sheet_id:
                update_sheet_log(sheet_id, "ERROR", error_msg)
            return jsonify({"error": error_msg}), 400

        # Default sheet_name to "new_transactions" if not provided
        if not sheet_name:
            logger.warning(
                "sheet_name not provided in webhook, defaulting to 'new_transactions'"
            )
            sheet_name = "new_transactions"

        # Get new embeddings
        try:
            new_embeddings = np.array(
                [item["embedding"] for item in data["output"]], dtype=np.float32
            )
            logger.info(f"Processed {len(new_embeddings)} new embeddings")
        except Exception as e:
            error_msg = f"Error processing embeddings: {str(e)}"
            logger.error(error_msg)
            update_sheet_log(sheet_id, "ERROR", error_msg)
            return jsonify({"error": error_msg}), 400

        # Get trained embeddings and categories
        try:
            trained_embeddings = fetch_embeddings(f"{sheet_id}")
            trained_data = fetch_embeddings(f"{sheet_id}_index")
            logger.info(
                f"Retrieved {len(trained_embeddings)} trained embeddings and {len(trained_data)} training data points"
            )

            # Add debug logging to see the structure of trained_data
            if len(trained_data) > 0:
                logger.info(f"First training data point: {trained_data[0]}")
                logger.info(f"Training data dtype: {trained_data.dtype}")
                logger.info(f"Training data shape: {trained_data.shape}")

                # Log a few sample categories to verify content
                sample_categories = [
                    (
                        trained_data[i]["category"]
                        if "category" in trained_data.dtype.names
                        else (
                            trained_data[i]["Category"]
                            if "Category" in trained_data.dtype.names
                            else trained_data[i][1]
                        )
                    )
                    for i in range(min(5, len(trained_data)))
                ]
                logger.info(f"Sample categories: {sample_categories}")

            if len(trained_embeddings) == 0 or len(trained_data) == 0:
                error_msg = "No training data found"
                logger.error(error_msg)
                update_sheet_log(sheet_id, "ERROR", error_msg)
                return jsonify({"error": error_msg}), 400
        except Exception as e:
            error_msg = f"Error fetching training data: {str(e)}"
            logger.error(error_msg)
            update_sheet_log(sheet_id, "ERROR", error_msg)
            return jsonify({"error": error_msg}), 400

        # Calculate similarities
        try:
            similarities = cosine_similarity(new_embeddings, trained_embeddings)
            best_matches = similarities.argmax(axis=1)
            logger.info(f"Calculated similarities with shape {similarities.shape}")
        except Exception as e:
            error_msg = f"Error calculating similarities: {str(e)}"
            logger.error(error_msg)
            update_sheet_log(sheet_id, "ERROR", error_msg)
            return jsonify({"error": error_msg}), 400

        # Get predicted categories and confidence scores
        results = []
        for i, idx in enumerate(best_matches):
            try:
                # Add debug logging for the first few matches
                if i < 3:
                    logger.info(f"Match {i}: trained_data[{idx}] = {trained_data[idx]}")

                # Extract the category directly from the structured array
                # This should be the actual category, not the description
                category = None

                # Try different field names and positions to get the category
                if "category" in trained_data.dtype.names:
                    category = str(trained_data[idx]["category"])
                elif "Category" in trained_data.dtype.names:
                    category = str(trained_data[idx]["Category"])
                else:
                    # Fallback to index 1 which should be the category field
                    category = str(trained_data[idx][1])

                similarity_score = float(
                    similarities[i][idx]
                )  # Get the similarity score

                # Log the extracted category and score
                if i < 3:
                    logger.info(
                        f"Extracted category: '{category}', similarity score: {similarity_score:.2f}"
                    )

                results.append(
                    {
                        "predicted_category": category,
                        "similarity_score": similarity_score,
                    }
                )
            except Exception as e:
                logger.error(f"Error processing prediction {i}: {str(e)}")
                logger.error(
                    f"trained_data type: {type(trained_data)}, shape: {trained_data.shape if hasattr(trained_data, 'shape') else 'unknown'}"
                )
                if i < 3 and idx < len(trained_data):
                    logger.error(f"trained_data[{idx}] = {trained_data[idx]}")
                results.append(
                    {"predicted_category": "Unknown", "similarity_score": 0.0}
                )

        logger.info(f"Generated {len(results)} predictions")

        # Instead of writing to the sheet directly, return the results
        status_msg = f"Generated {len(results)} predictions for sheet '{sheet_name}'"
        update_sheet_log(sheet_id, "INFO", status_msg)

        try:
            # Store webhook result with results data
            prediction_id = request.args.get("prediction_id", "unknown")

            try:
                # Format results for storage
                serializable_results = {
                    "status": "success",
                    "message": "Classification completed",
                    "data": [],
                }

                # Add each result with simple types
                for r in results:
                    serializable_results["data"].append(
                        {
                            "predicted_category": str(r.get("predicted_category", "")),
                            "similarity_score": float(r.get("similarity_score", 0)),
                            "narrative": (
                                str(r.get("narrative", "")) if "narrative" in r else ""
                            ),
                        }
                    )

                # Store the results
                webhook_result = prisma_client.insert_webhook_result(
                    prediction_id, serializable_results
                )
                if webhook_result:
                    logger.info(f"Stored webhook result for sheet_id: {sheet_id}")
                else:
                    logger.warning(
                        f"Failed to store webhook result for sheet_id: {sheet_id}"
                    )
            except Exception as e:
                logger.warning(f"Error storing webhook result: {e}")

            status_msg = "Classification completed successfully"
            update_process_status("completed", "classify", user_id)
            update_sheet_log(sheet_id, "SUCCESS", status_msg)

            # Return the results to be handled by the Google Apps Script
            return jsonify(
                {
                    "status": "success",
                    "results": results,
                    "config": {
                        "categoryColumn": category_column,
                        "startRow": start_row,
                        "sheetName": sheet_name,
                        "spreadsheetId": sheet_id,
                    },
                }
            )

        except Exception as e:
            error_msg = f"Error processing results: {str(e)}"
            logger.error(error_msg)
            update_sheet_log(sheet_id, "ERROR", error_msg)
            return jsonify({"error": error_msg}), 500

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in classify webhook: {error_msg}")
        if user_id:
            update_process_status(f"Error: {str(e)}", "classify", user_id)
        if sheet_id:
            update_sheet_log(
                sheet_id, "ERROR", f"Classification webhook failed: {error_msg}"
            )
        return jsonify({"error": error_msg}), 500


@app.route("/status/<prediction_id>", methods=["GET"])
def get_prediction_status(prediction_id):
    """Get the status of a prediction."""
    try:
        try:
            webhook_result = prisma_client.get_webhook_result(prediction_id)
            if webhook_result:
                logger.info(
                    f"Found completed webhook results for prediction {prediction_id}"
                )
                webhook_data = webhook_result

                # Extract the actual results from the webhook data
                results_data = []
                if webhook_data.get("results"):
                    # Handle different result structures
                    if isinstance(webhook_data["results"], dict):
                        if webhook_data["results"].get("data"):
                            # Old structure: {"data": [...]}
                            results_data = webhook_data["results"]["data"]
                        elif webhook_data["results"].get("status") and webhook_data[
                            "results"
                        ].get("data"):
                            # Another structure: {"status": "success", "data": [...]}
                            results_data = webhook_data["results"]["data"]
                    elif isinstance(webhook_data["results"], list):
                        # New structure: direct list
                        results_data = webhook_data["results"]
                    logger.info(
                        f"Extracted {len(results_data)} results from webhook data"
                    )

                # Get configuration from predictions_db if available
                category_column = "E"  # Default
                start_row = "1"  # Default
                sheet_name = "Sheet1"  # Default
                spreadsheet_id = None

                if prediction_id in predictions_db:
                    pred_data = predictions_db[prediction_id]
                    category_column = pred_data.get("category_column", category_column)
                    start_row = pred_data.get("start_row", start_row)
                    sheet_name = pred_data.get("sheet_name", sheet_name)
                    spreadsheet_id = pred_data.get("spreadsheet_id", spreadsheet_id)

                # Return a response that includes the actual results
                return jsonify(
                    {
                        "status": "completed",
                        "message": "Classification completed successfully",
                        "results": results_data,  # Include the actual results array
                        "config": {
                            "categoryColumn": category_column,
                            "startRow": start_row,
                            "sheetName": sheet_name,
                            "spreadsheetId": spreadsheet_id,
                        },
                        "result": {
                            "results": {"status": "success", "data": results_data}
                        },  # Keep the original result format for backward compatibility
                    }
                )
        except Exception as e:
            logger.warning(f"Error checking webhook results: {e}")

        # Then check if we have this prediction in our local dictionary
        if prediction_id in predictions_db:
            logger.info(f"Found prediction {prediction_id} in local predictions_db")
            prediction_data = predictions_db[prediction_id]

            # This helps in cases where the webhook has completed but we missed it
            if prediction_data.get("status") == "waiting_for_webhook":
                try:
                    # Quick check for webhook results
                    webhook_check = prisma_client.get_webhook_result(prediction_id)
                    if webhook_check:
                        # We found a result, get the full data
                        webhook_data = webhook_check

                        # Extract results and return them
                        results_data = []
                        if webhook_data.get("results"):
                            # Handle different result structures
                            if isinstance(webhook_data["results"], dict):
                                if webhook_data["results"].get("data"):
                                    # Old structure: {"data": [...]}
                                    results_data = webhook_data["results"]["data"]
                                elif webhook_data["results"].get(
                                    "status"
                                ) and webhook_data["results"].get("data"):
                                    # Another structure: {"status": "success", "data": [...]}
                                    results_data = webhook_data["results"]["data"]
                            elif isinstance(webhook_data["results"], list):
                                # New structure: direct list
                                results_data = webhook_data["results"]
                            logger.info(
                                f"Found webhook results on second check: {len(results_data)} results"
                            )

                        return jsonify(
                            {
                                "status": "completed",
                                "message": "Classification completed successfully",
                                "results": results_data,
                                "config": {
                                    "categoryColumn": prediction_data.get(
                                        "category_column", "E"
                                    ),
                                    "startRow": prediction_data.get("start_row", "1"),
                                    "sheetName": prediction_data.get(
                                        "sheet_name", "Sheet1"
                                    ),
                                    "spreadsheetId": prediction_data.get(
                                        "spreadsheet_id"
                                    ),
                                },
                                "result": {
                                    "results": {
                                        "status": "success",
                                        "data": results_data,
                                    }
                                },
                            }
                        )
                except Exception as e:
                    logger.warning(f"Error on second webhook check: {e}")

            # Return the status from predictions_db
            return jsonify(
                {
                    "status": prediction_data.get("status", "processing"),
                    "message": prediction_data.get(
                        "status_message", "Processing in progress"
                    ),
                    "processed_transactions": prediction_data.get(
                        "processed_transactions", 0
                    ),
                    "total_transactions": prediction_data.get("total_transactions", 0),
                }
            )

        # Check if we have a cached response that's less than 1 minute old
        current_time = datetime.now()
        if prediction_id in replicate_cache:
            cache_entry = replicate_cache[prediction_id]
            cache_age = current_time - cache_entry["timestamp"]

            # Use cached response if it's less than 1 minute old
            if cache_age < timedelta(minutes=1):
                logger.info(
                    f"Using cached response for prediction {prediction_id} (age: {cache_age.total_seconds():.1f}s)"
                )
                return jsonify(cache_entry["response"])

        # Try to get prediction from Replicate
        try:
            prediction = replicate.predictions.get(prediction_id)

            if not prediction:
                logger.warning(f"Prediction {prediction_id} not found in Replicate")
                response_data = {
                    "status": "not_found",
                    "message": "Prediction not found in Replicate",
                }
                # Cache the response
                replicate_cache[prediction_id] = {
                    "timestamp": current_time,
                    "response": response_data,
                }
                return jsonify(response_data), 404

            # Return status based on prediction state
            status = prediction.status
            if status == "succeeded":
                response_data = {
                    "status": "completed",
                    "message": "Processing completed successfully",
                }
            elif status == "failed":
                response_data = {"status": "failed", "error": prediction.error}
            else:
                response_data = {"status": status, "message": "Processing in progress"}

            # Cache the response
            replicate_cache[prediction_id] = {
                "timestamp": current_time,
                "response": response_data,
            }

            return jsonify(response_data)

        except Exception as e:
            logger.warning(f"Error fetching prediction from Replicate: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    except Exception as e:
        logger.error(f"Error getting prediction status: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


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

                    # Return the API key
                    return jsonify(
                        {"status": "success", "user_id": user_id, "api_key": api_key}
                    )
                except Exception as e:
                    logger.error(f"API key validation failed: {e}")
                    return (
                        jsonify({"error": f"API key validation failed: {str(e)}"}),
                        401,
                    )

            # If no API key in header, check for user_id in query params
            user_id = request.args.get("userId")
            if not user_id:
                return jsonify({"error": "Missing userId parameter"}), 400

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
                    return jsonify({"error": "User not found"}), 404

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
            if not data or "userId" not in data:
                return jsonify({"error": "Missing userId in request body"}), 400

            user_id = data["userId"]

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
        return jsonify({"error": str(e)}), 500


@app.route("/user-config", methods=["GET"])
def get_user_config():
    """Get user configuration."""
    try:
        user_id = request.args.get("userId")
        if not user_id:
            return jsonify({"error": "Missing userId parameter"}), 400

        response = prisma_client.get_account_by_user_id(user_id)
        if not response:
            return jsonify({"error": "User not found"}), 404

        # Update API key if provided
        api_key = request.args.get("apiKey")
        if api_key:
            prisma_client.update_account(user_id, {"api_key": api_key})
            logger.info(f"Updated API key for user {user_id}")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error getting user config: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/webhook", methods=["POST"])
def webhook():
    """Handle webhook from Replicate."""
    # Extract parameters from request
    spreadsheet_id = request.args.get("spreadsheetId")
    user_id = request.args.get("userId")

    # Get prediction_id from headers or request
    prediction_id = request.headers.get(
        "X-Replicate-Prediction-Id"
    ) or request.args.get("prediction_id")

    logger.info(
        f"Received webhook for spreadsheet_id: {spreadsheet_id}, user_id: {user_id}, prediction_id: {prediction_id}"
    )

    if not all([spreadsheet_id, user_id]):
        error_msg = f"Missing required parameters: spreadsheet_id={spreadsheet_id}, user_id={user_id}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 400

    try:
        # Parse JSON data
        if not request.is_json:
            error_msg = "Request must be JSON"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400

        data = request.get_json(force=True)
        logger.info(f"Webhook data keys: {data.keys()}")

        # Check if this is a training webhook (from the URL path)
        is_training = "/train/" in request.path

        if is_training:
            # Process embeddings for training
            try:
                # Extract embeddings from the response
                if (
                    "output" in data
                    and isinstance(data["output"], list)
                    and len(data["output"]) > 0
                ):
                    # Standard Replicate format
                    embeddings = np.array(
                        [item["embedding"] for item in data["output"]], dtype=np.float32
                    )
                    logger.info(f"Extracted embeddings with shape: {embeddings.shape}")

                    # Store the embeddings
                    store_result = store_embeddings(embeddings, f"{spreadsheet_id}")
                    logger.info(
                        f"Stored embeddings for {spreadsheet_id}: {store_result}"
                    )

                    # Update status
                    update_process_status("completed", "training", user_id)

                    # Store webhook result
                    if prediction_id:
                        result = {
                            "user_id": user_id,
                            "status": "success",
                            "embeddings_shape": str(embeddings.shape),
                            "spreadsheet_id": spreadsheet_id,
                        }
                        prisma_client.insert_webhook_result(prediction_id, result)

                    return jsonify({"status": "success"}), 200
                else:
                    error_msg = "Invalid data format in webhook response"
                    logger.error(error_msg)
                    return jsonify({"error": error_msg}), 400
            except Exception as e:
                error_msg = f"Error processing training webhook: {str(e)}"
                logger.error(error_msg)
                return jsonify({"error": error_msg}), 500
        else:
            # Handle other webhook types (classify, etc.)
            logger.info("Non-training webhook received")
            return jsonify({"status": "success"}), 200

    except Exception as e:
        error_msg = f"Error in webhook: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500


@app.route("/webhook-result", methods=["GET"])
def get_webhook_result():
    """Get webhook result."""
    try:
        prediction_id = request.args.get("prediction_id")
        if not prediction_id:
            return jsonify({"error": "Missing prediction_id parameter"}), 400

        webhook_result = prisma_client.get_webhook_result(prediction_id)
        if not webhook_result:
            return (
                jsonify(
                    {
                        "status": "pending",
                        "message": "No result found for this prediction ID",
                    }
                ),
                404,
            )

        return jsonify(webhook_result), 200
    except Exception as e:
        logger.error(f"Error getting webhook result: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/check-webhook", methods=["GET"])
def check_webhook():
    """Check if webhook has been processed."""
    try:
        prediction_id = request.args.get("prediction_id")
        if not prediction_id:
            return jsonify({"error": "Missing prediction_id parameter"}), 400

        webhook_check = prisma_client.get_webhook_result(prediction_id)
        if webhook_check:
            return jsonify({"status": "processed"}), 200
        else:
            return jsonify({"status": "pending"}), 200
    except Exception as e:
        logger.error(f"Error checking webhook: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/train/webhook", methods=["POST"])
def train_webhook():
    """Handle training webhook from Replicate."""
    logger.info("Received training webhook, redirecting to main webhook handler")
    # Add mode=train to the request args to ensure it's treated as a training webhook
    request.args = request.args.copy()
    request.args["mode"] = "train"
    return webhook()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
