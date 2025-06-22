import os
import json
import sentry_sdk
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import uuid
from utils.prisma_client import db_client
from flasgger import Swagger, swag_from
from pydantic import ValidationError

# Initialize Sentry
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    # Add data like request headers and IP for users,
    # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
    send_default_pii=True,
)

# Import from new modules
from config import (
    REPLICATE_MODEL_NAME,
    REPLICATE_MODEL_VERSION,
    BACKEND_API,
)
from models import (
    TrainRequest,
    ClassifyRequest,
    UserConfigRequest,
    FinancialAnalyticsRequest,
)
from utils.text_utils import clean_text, clean_and_group_transactions
from utils.request_utils import (
    validate_request_data,
    create_error_response,
    require_api_key,
)
from services.training_service import process_training_request
from services.classification_service import process_classification_request
from services.status_service import get_and_process_status
from services.universal_categorization_service import process_universal_categorization_request
from services.financial_analytics_service import process_financial_analytics_request

# Configure logging
logging.basicConfig(
    stream=sys.stdout,  # Log to stdout for Docker/Gunicorn to capture
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
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

# === ThreadPoolExecutor for background tasks ===
# Adjust max_workers as needed based on server resources and expected load
# For CPU-bound tasks like embedding generation, number of CPU cores is a good starting point.
# For I/O-bound tasks (if any were dominant), could be higher.
# Since our tasks are CPU-bound for ~10-15s, a small number is fine to prevent resource exhaustion.
executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 2)

# === Constants Moved to config.py ===
# BACKEND_API is imported
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
        connected = db_client.connect()
        logger.info("Successfully connected to database on startup")
    except Exception as e:
        retry_count += 1
        logger.error(f"Database connection attempt {retry_count} failed: {e}")
        if retry_count < max_retries:
            sleep_time = 1 * retry_count
            logger.info(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
        else:
            logger.warning(
                "All connection attempts failed, will retry on first request"
            )


@app.teardown_appcontext
def shutdown_db_connection(exception=None):
    """Disconnect from the database when the application context ends."""
    try:
        if exception and isinstance(exception, Exception):
            logger.info("Disconnecting from database due to exception")
            db_client.disconnect()
            logger.info("Disconnected from database")
    except Exception as e:
        logger.error(f"Error disconnecting from database: {e}")


@app.before_request
def ensure_db_connection():
    """Ensure database connection is active before each request."""
    try:
        db_client.connect()
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
                            "/auto-classify",
                            "/train",
                            "/financial-analytics",
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
                "/auto-classify",
                "/train",
                "/financial-analytics",
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
    try:
        connected = db_client.connect()
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
    return jsonify(health_status)


@app.route("/sentry-debug")
def trigger_error():
    """Test endpoint to verify Sentry error reporting"""
    logger.info("Sentry debug endpoint accessed - triggering test error")
    division_by_zero = 1 / 0


@app.route("/train", methods=["POST"])
@require_api_key
@swag_from(
    {
        "tags": ["Training"],
        "summary": "Train transaction classifier",
        "description": (
            "Trains the model with new transaction data. Requires a minimum of 10 transactions. "
            "The user is identified via the X-API-Key. "
            "If training completes within ~8.5s, a 200 OK is returned with the completion status. "
            "If it takes longer, a 202 Accepted is returned, and the client must poll the /status endpoint."
        ),
        "parameters": [
            {
                "name": "X-API-Key",
                "in": "header",
                "type": "string",
                "required": True,
                "description": "API Key for authentication (implicitly provides user context)",
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
                            "description": "Array of transaction objects (minimum 10) with descriptions and categories.",
                            "minItems": 10,
                            "items": {
                                "type": "object",
                                "required": ["description", "Category"],
                                "properties": {
                                    "description": {
                                        "type": "string",
                                        "description": "The transaction description or narrative.",
                                        "example": "WOOLWORTHS 2099 Dee Why AU AUS"
                                    },
                                    "Category": {
                                        "type": "string",
                                        "description": "The category label for the transaction.",
                                        "example": "Groceries"
                                    },
                                },
                            },
                        },
                        "expenseSheetId": {
                            "type": "string",
                            "description": (
                                "(Optional) Client-specific identifier for the training dataset (e.g., a spreadsheet ID). "
                                "Not used by the core training service but can be passed through."
                            ),
                            "minLength": 1,
                            "example": "spreadsheet-unique-id-123"
                        }
                    },
                    "example": {
                        "transactions": [
                            {"description": "TRANSPORTFORNSW TAP SYDNEY AUS Card Value Date: 23/01/2024", "Category": "Transport"},
                            {"description": "WOOLWORTHS 2099 Dee Why AU AUS Card Value Date: 28/01/2024", "Category": "Groceries"},
                            {"description": "RivaReno Gelato Barangaroo NS AUS Card Value Date: 28/01/2024", "Category": "DinnerBars"},
                            {"description": "Harris Farm Markets NS AUS Card Value Date: 28/01/2024", "Category": "Groceries"},
                            {"description": "TREACHERY CAMP PL SEAL ROCKS NS AUS Card Value Date: 27/01/2024", "Category": "Travel"},
                            {"description": "MED*ALDIMobile CHATSWOOD AU AUS Card Value Date: 27/01/2024", "Category": "Utility"},
                            {"description": "ADOBE CREATIVE CLOUD Sydney AU AUS Card Value Date: 27/01/2024", "Category": "Utility"},
                            {"description": "COTTON ON MEGA 2957 FORSTER NS AUS Card Value Date: 27/01/2024", "Category": "Shopping"},
                            {"description": "ALDI STORES - DEE WHY AU", "Category": "Groceries"},
                            {"description": "Transfer to other Bank NetBank Dee Why Pde", "Category": "Living"}
                        ]
                        # "expenseSheetId": "test-sheet-id-for-training-optional"
                    }
                },
            },
        ],
        "responses": {
            200: {
                "description": (
                    "Training completed synchronously OR training initiated and will complete asynchronously. "
                    "If synchronous, 'status' will be 'completed'. "
                    "If asynchronous (but responded before timeout), 'status' will be 'processing' and 'prediction_id' will be provided for polling."
                ),
                "examples": {
                    "application/json_sync_completed": {
                        "status": "completed",
                        "message": "Training completed successfully for model: user-xyz-123.",
                        "model_id": "user-xyz-123",
                        "unique_description_count": 150,
                        "category_count": 12,
                        "training_duration_seconds": 5.2
                    },
                    "application/json_async_started_200": {
                        "status": "processing",
                        "prediction_id": "train_job_abc789",
                        "message": "Training started. Check status endpoint for updates."
                    }
                },
            },
            202: {
                "description": "Training accepted for background processing due to exceeding initial processing time. Poll /status/{prediction_id} for completion.",
                "examples": {
                    "application/json": {
                        "status": "processing",
                        "prediction_id": "train_job_def456",
                        "message": "Training request accepted and is processing in the background."
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
        logger.info("=== Incoming Training Request ===")
        data = request.get_json()
        if not data:
            return create_error_response("Missing request data", 400)

        validated_data, error_response = validate_request_data(TrainRequest, data)
        if error_response:
            return error_response

        job_id = str(uuid.uuid4())
        api_key_user_id = request.user_id

        # 1. Create PENDING job record immediately
        db_client.create_async_job(job_id, api_key_user_id, "training", "PENDING")

        # 2. Submit actual work to executor
        future = executor.submit(
            _run_training_task, job_id, api_key_user_id, validated_data
        )

        try:
            # 3. Try to get result with timeout
            job_result_dict = future.result(timeout=8.5)

            status_code = (
                job_result_dict.get("error_code", 200)
                if isinstance(job_result_dict, dict)
                else 200
            )
            if (
                isinstance(job_result_dict, dict)
                and job_result_dict.get("status") == "error"
            ):
                status_code = job_result_dict.get("error_code", 500)

            return jsonify(job_result_dict), status_code

        except TimeoutError:
            logger.info(
                f"Training job {job_id} for user {api_key_user_id} taking longer, returning 202."
            )
            return (
                jsonify(
                    {
                        "status": "processing",
                        "prediction_id": job_id,
                        "message": "Training request accepted and is processing in the background.",
                    }
                ),
                202,
            )
        except Exception as e:
            logger.error(
                f"Error managing future for training job {job_id}: {e}", exc_info=True
            )
            db_client.update_async_job_status(
                job_id, "FAILED", error_message=f"Internal error managing job: {str(e)}"
            )
            return (
                jsonify(
                    {
                        "status": "error",
                        "error_message": "Failed to process training request due to an internal error.",
                        "prediction_id": job_id,
                    }
                ),
                500,
            )

    except Exception as e:
        logger.error(f"Error in train_model endpoint: {e}", exc_info=True)
        return create_error_response(
            f"Server error in training endpoint: {str(e)}", 500
        )


@app.route("/classify", methods=["POST"])
@require_api_key
@swag_from(
    {
        "tags": ["Classification"],
        "summary": "Classify transactions",
        "description": (
            "Classifies a list of transaction descriptions. The user is identified via the X-API-Key. "
            "If processing takes longer than the gateway timeout (e.g., ~8.5 seconds on Render), "
            "this endpoint returns a 202 Accepted with a `prediction_id`. The client should then poll "
            "the `/status/{prediction_id}` endpoint to get the final results. "
        ),
        "parameters": [
            {
                "name": "X-API-Key",
                "in": "header",
                "type": "string",
                "required": True,
                "description": "API Key for authentication.",
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
                            "items": {
                                "oneOf": [
                                    {"type": "string"},
                                    {
                                        "$ref": "#/definitions/TransactionInput",
                                    },
                                ]
                            },
                            "description": "List of transaction descriptions (strings) or transaction objects.",
                            "example": [
                                "Coffee with John",
                                "Salary payment MAY",
                                {
                                    "description": "Grocery shopping at local store",
                                    "amount": -55.20,
                                    "money_in": False,
                                },
                                {
                                    "description": "Online course subscription",
                                    "amount": -19.99
                                },
                                {
                                    "description": "Client payment for project X",
                                    "amount": 1200.00,
                                    "money_in": True
                                }
                            ],
                        },
                        "user_categories": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "name": {"type": "string"},
                                },
                                "required": ["id", "name"]
                            },
                            "description": (
                                "(Optional) List of user-defined categories to assist the LLM, especially for few-shot learning or specific user preferences. "
                                "Each category should have an 'id' and a 'name'."
                            ),
                            "example": [
                                {"id": "cat_travel", "name": "Travel Expenses"},
                                {"id": "cat_utilities", "name": "Utilities"},
                            ],
                        },
                    },
                },
            },
        ],
        "definitions": {
            "TransactionInput": {
                "type": "object",
                "required": ["description"],
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "The transaction description or narrative.",
                        "example": "UBER TRIP SYDNEY"
                    },
                    "money_in": {
                        "type": "boolean",
                        "description": "Indicates if the transaction is income/credit (true) or an expense/debit (false). Helps in disambiguation, especially for transfer detection.",
                        "example": False
                    },
                    "amount": { 
                        "type": "number",
                        "format": "float",
                        "description": "(Optional) The numerical amount of the transaction. Used for transfer detection.",
                        "example": 25.50
                    }
                },
            },
        },
        "responses": {
            200: {
                "description": "Classification completed synchronously",
                "examples": {
                    "application/json": {
                        "status": "completed",
                        "message": "Classification completed successfully.",
                        "prediction_id": "clf_job_sync_abc123",
                        "type": "classification",
                        "results": [
                            {
                                "narrative": "UBER EATS SYDNEY",
                                "cleaned_narrative": "UBER EATS SYDNEY",
                                "predicted_category": "Food & Dining",
                                "predicted_category_name": "Food & Dining",
                                "similarity_score": 0.9523,
                                "second_predicted_category": "Transport",
                                "second_similarity_score": 0.1200,
                                "money_in": False,
                                "amount": 25.50,
                                "adjustment_info": {
                                    "is_low_confidence": False,
                                    "reason": None,
                                    "llm_assisted": True,
                                    "llm_model": "gpt-3.5-turbo",
                                    "original_embedding_category": "Takeaway",
                                    "original_embedding_category_name": "Takeaway Food",
                                    "original_similarity_score": 0.65,
                                    "transfer_detection_reason": None,
                                    "adjusted": True,
                                    "original_category": "Takeaway",
                                    "matched_transfer_key": None,
                                    "is_refund_candidate": False
                                },
                                "debug_info": {
                                    "reason_code": "LOW_ABS_CONF",
                                    "best_score": 0.45,
                                    "threshold": 0.7,
                                    "evaluated_category": "Groceries",
                                    "neighbor_categories": ["Groceries", "Shopping", "Groceries"],
                                    "neighbor_cleaned_descs": ["WOOLWORTHS METRO", "KMART ONLINE", "COLES CENTRAL"],
                                    "neighbor_original_descs": ["WOOLWORTHS METRO 123", "KMART ONLINE ORDER", "COLES CENTRAL SYDNEY"],
                                    "neighbor_scores": [0.45, 0.42, 0.41],
                                    "second_best_category": "Shopping",
                                    "difference": 0.03,
                                    "rel_threshold": 0.1,
                                    "unique_neighbor_categories": ["Groceries", "Shopping"]
                                }
                            },
                            {
                                "narrative": "Salary Deposit CBA",
                                "cleaned_narrative": "SALARY DEPOSIT CBA",
                                "predicted_category": "Income",
                                "predicted_category_name": "Income",
                                "similarity_score": 0.9910,
                                "second_predicted_category": "Unknown",
                                "second_similarity_score": 0.0500,
                                "money_in": True,
                                "amount": 2500.00,
                                "adjustment_info": {"adjusted": False},
                                "debug_info": None
                            }
                        ]
                    }
                }
            },
            202: {"description": "Classification accepted for background processing"},
            400: {"description": "Invalid request data"},
            401: {"description": "Invalid or missing API key"},
            409: {
                "description": "Conflict (e.g., embedding mismatch, requires re-training)"
            },
            500: {"description": "Server error"},
        },
    }
)
def classify_transactions_async():
    """Attempts synchronous classification first, falls back to async if needed."""
    try:
        user_id = request.user_id  # Get user_id associated with the API key
        data = request.get_json()
        if not data:
            return create_error_response("Missing request data", 400)

        # Validate request data using Pydantic model
        try:
            validated_data = ClassifyRequest(**data)
        except ValidationError as e:
            logger.error(f"Validation error for /classify: {e.json()}")
            # Provide a more user-friendly error message
            error_details = e.errors()
            first_error = error_details[0]["msg"] if error_details else "Invalid data"
            return create_error_response(f"Validation error: {first_error}", 400)

        # Generate a unique job ID
        job_id = str(uuid.uuid4())

        # --- Critical Section: Record Job Intention --- #
        try:
            db_client.create_async_job(job_id, user_id, "classification", "PENDING")
            logger.info(f"Created PENDING job record {job_id} for user {user_id}")
        except Exception as db_err:
            logger.error(
                f"Failed to create PENDING job record for job {job_id}: {db_err}",
                exc_info=True,
            )
            # Fail fast if we can't even record the job
            return create_error_response("Failed to initiate classification job.", 500)
        # --- End Critical Section --- #

        # Submit actual work to the executor
        # Pass the full validated_data object, which includes user_categories
        future = executor.submit(
            _run_classification_task,
            job_id,  # Pass the generated job ID
            user_id,  # Pass the user ID
            validated_data,  # Pass the whole validated request model
        )

        # Try to get the result synchronously with a timeout
        try:
            # Wait for the result for up to ~8.5 seconds
            job_result_dict = future.result(timeout=8.5)

            # Check the structure and status of the result from the task
            if not isinstance(job_result_dict, dict):
                logger.error(
                    f"Task for job {job_id} returned unexpected type: {type(job_result_dict)}"
                )
                # Update job status to FAILED as task result was bad
                db_client.update_async_job_status(
                    job_id,
                    "FAILED",
                    error_message="Internal error: Invalid result format from task.",
                )
                return create_error_response(
                    "Internal error during classification processing.", 500
                )

            # Determine status code based on result
            status_code = 200  # Default success
            if job_result_dict.get("status") == "error":
                status_code = job_result_dict.get("error_code", 500)

            logger.info(
                f"Classification job {job_id} completed synchronously. Returning {status_code}."
            )
            # Return the full result dictionary (contains status, message, results/error)
            return jsonify(job_result_dict), status_code

        except TimeoutError:
            # Task didn't complete within the timeout
            logger.info(
                f"Classification job {job_id} for user {user_id} taking longer, returning 202 Accepted."
            )
            # The job status remains PENDING or will be updated by the background task
            return (
                jsonify(
                    {
                        "status": "processing",
                        "prediction_id": job_id,
                        "message": "Classification request accepted and is processing in the background.",
                    }
                ),
                202,
            )

        except Exception as future_err:
            # Error retrieving result from the future (e.g., task raised an unhandled exception)
            logger.error(
                f"Error retrieving result for classification job {job_id}: {future_err}",
                exc_info=True,
            )
            # The background task *should* handle its own errors and update the DB,
            # but we update here just in case it failed before it could.
            db_client.update_async_job_status(
                job_id,
                "FAILED",
                error_message=f"Internal error managing job future: {str(future_err)}",
            )
            return (
                jsonify(
                    {
                        "status": "error",
                        "error_message": "Failed to process classification request due to an internal error.",
                        "prediction_id": job_id,
                    }
                ),
                500,
            )

    except Exception as route_err:
        # Catch-all for errors in the route handler itself (before job submission)
        logger.error(
            f"Error in /classify route for user {request.user_id if hasattr(request, 'user_id') else 'Unknown'}: {route_err}",
            exc_info=True,
        )
        return create_error_response(
            f"Server error in classification endpoint: {str(route_err)}", 500
        )


@app.route("/auto-classify", methods=["POST"])
@require_api_key
@swag_from(
    {
        "tags": ["Universal Categorization"],
        "summary": "Auto-classify transactions using universal categories",
        "description": (
            "Auto-classifies transactions using improved text cleaning, grouping, and LLM-based categorization "
            "with predefined categories. This endpoint bypasses user training data and provides immediate "
            "auto-classification using a comprehensive set of predefined transaction categories. "
            "The process includes advanced text cleaning, similarity-based grouping of similar merchants, "
            "and AI-powered categorization."
        ),
        "parameters": [
            {
                "name": "X-API-Key",
                "in": "header",
                "type": "string",
                "required": True,
                "description": "API Key for authentication.",
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
                            "items": {
                                "oneOf": [
                                    {"type": "string"},
                                    {
                                        "type": "object",
                                        "required": ["description"],
                                        "properties": {
                                            "description": {
                                                "type": "string",
                                                "description": "Transaction description text",
                                                "example": "UBER EATS SYDNEY NSW"
                                            },
                                            "money_in": {
                                                "type": "boolean",
                                                "description": "Whether this is an incoming transaction (credit)",
                                                "example": False
                                            },
                                            "amount": {
                                                "type": "number",
                                                "description": "Transaction amount (positive number)",
                                                "example": 25.50
                                            }
                                        }
                                    }
                                ]
                            },
                            "description": "List of transaction descriptions or transaction objects to auto-classify",
                            "example": [
                                "UBER EATS SYDNEY NSW",
                                {
                                    "description": "SALARY DEPOSIT",
                                    "money_in": True,
                                    "amount": 2500.00
                                }
                            ]
                        }
                    }
                }
            }
        ],
        "responses": {
            200: {
                "description": "Universal categorization completed successfully",
                "examples": {
                    "application/json": {
                        "status": "completed",
                        "message": "Universal categorization completed successfully",
                        "results": [
                            {
                                "narrative": "UBER EATS SYDNEY NSW",
                                "cleaned_narrative": "uber eats sydney",
                                "predicted_category": "FOOD_AND_DRINK",
                                "similarity_score": 1.0,
                                "second_predicted_category": None,
                                "second_similarity_score": 0.0,
                                "money_in": False,
                                "amount": 25.50,
                                "adjustment_info": {
                                    "universal_categorization": True,
                                    "group_representative": "uber eats sydney",
                                    "llm_assisted": True,
                                    "llm_model": "gpt-4.1-2025-04-14",
                                    "bypassed_training_data": True
                                },
                                "debug_info": {
                                    "method": "universal_categorization",
                                    "group_size": 1,
                                    "predefined_categories_count": 16
                                }
                            },
                            {
                                "narrative": "SALARY DEPOSIT",
                                "cleaned_narrative": "salary deposit",
                                "predicted_category": "INCOME",
                                "similarity_score": 1.0,
                                "second_predicted_category": None,
                                "second_similarity_score": 0.0,
                                "money_in": True,
                                "amount": 2500.00,
                                "adjustment_info": {
                                    "universal_categorization": True,
                                    "group_representative": "salary deposit",
                                    "llm_assisted": True,
                                    "llm_model": "gpt-4.1-2025-04-14",
                                    "bypassed_training_data": True
                                },
                                "debug_info": {
                                    "method": "universal_categorization",
                                    "group_size": 1,
                                    "predefined_categories_count": 16
                                }
                            }
                        ],
                        "processing_info": {
                            "total_transactions": 2,
                            "unique_groups": 2,
                            "processing_time_seconds": 1.23,
                            "categories_used": 16,
                            "method": "universal_categorization"
                        }
                    }
                }
            },
            400: {"description": "Invalid request data"},
            401: {"description": "Invalid or missing API key"},
            500: {"description": "Server error"},
            503: {"description": "LLM categorization service unavailable"}
        }
    }
)
def auto_classify_transactions_universal():
    """Universal auto-classification using predefined categories and LLM."""
    try:
        user_id = request.user_id  # Get user_id associated with the API key
        data = request.get_json()
        if not data:
            return create_error_response("Missing request data", 400)

        # Validate request data using same model as classify endpoint
        try:
            validated_data = ClassifyRequest(**data)
        except ValidationError as e:
            logger.error(f"Validation error for /auto-classify: {e.json()}")
            error_details = e.errors()
            first_error = error_details[0]["msg"] if error_details else "Invalid data"
            return create_error_response(f"Validation error: {first_error}", 400)

        # Process universal auto-classification (synchronous only)
        result_dict = process_universal_categorization_request(validated_data, user_id)

        # Determine status code based on result
        status_code = 200
        if result_dict.get("status") == "error":
            status_code = result_dict.get("error_code", 500)

        logger.info(f"Universal categorization completed for user {user_id}. Returning {status_code}.")
        return jsonify(result_dict), status_code

    except Exception as route_err:
        logger.error(
            f"Error in /auto-classify route for user {request.user_id if hasattr(request, 'user_id') else 'Unknown'}: {route_err}",
            exc_info=True,
        )
        return create_error_response(
            f"Server error in universal auto-classification endpoint: {str(route_err)}", 500
        )


@app.route("/status/<prediction_id>", methods=["GET"])
@require_api_key
@swag_from(
    {
        "tags": ["Status"],
        "summary": "Get prediction status",
        "description": (
            "Retrieves the current status and, if completed, the results of an asynchronous job "
            "(either classification or training) identified by its prediction_id.\n\n"
            "- For **completed classification jobs**: The response includes the final classification `results` "
            "(array of categorized transactions), the job `type` ('classification'), overall `status` ('completed'), "
            "and any passthrough `config` data that was associated with the job.\n"
            "- For **completed training jobs**: The response includes a summary `message`, details like `model_id`, "
            "`unique_description_count`, `category_count`, the job `type` ('training'), and overall `status` ('completed').\n"
            "- For **jobs still processing**: The response indicates `status` ('processing') and the job `type`.\n"
            "- For **failed jobs**: The response indicates `status` ('failed'), the job `type`, and may include an `error_message` or `error_details`.\n\n"
            "The user is identified via the X-API-Key and can only access the status of their own jobs."
        ),
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
                "description": (
                    "Successfully retrieved job status. The body will contain the detailed status. "
                    "If the job is completed, results (for classification) or summary details (for training) will be included. "
                    "See main endpoint description for variations based on job type and state (processing, failed)."
                ),
                "examples": {
                    "application/json": { # Using the completed classification as the primary example for 200 OK
                        "status": "completed",
                        "message": "Classification completed successfully.",
                        "prediction_id": "clf_job_abc123",
                        "type": "classification",
                        "results": [
                            {
                                "narrative": "UBER EATS SYDNEY",
                                "cleaned_narrative": "UBER EATS SYDNEY",
                                "predicted_category": "Food & Dining",
                                "predicted_category_name": "Food & Dining",
                                "similarity_score": 0.9523,
                                "second_predicted_category": "Transport",
                                "second_similarity_score": 0.1200,
                                "money_in": False,
                                "amount": 25.50,
                                "adjustment_info": {
                                    "is_low_confidence": False,
                                    "reason": None,
                                    "llm_assisted": True,
                                    "llm_model": "gpt-3.5-turbo",
                                    "original_embedding_category": "Takeaway",
                                    "original_embedding_category_name": "Takeaway Food",
                                    "original_similarity_score": 0.65,
                                    "transfer_detection_reason": None,
                                    "adjusted": True,
                                    "original_category": "Takeaway",
                                    "matched_transfer_key": None,
                                    "is_refund_candidate": False
                                },
                                "debug_info": {
                                    "reason_code": "LOW_ABS_CONF",
                                    "best_score": 0.45,
                                    "threshold": 0.7,
                                    "evaluated_category": "Groceries",
                                    "neighbor_categories": ["Groceries", "Shopping", "Groceries"],
                                    "neighbor_cleaned_descs": ["WOOLWORTHS METRO", "KMART ONLINE", "COLES CENTRAL"],
                                    "neighbor_original_descs": ["WOOLWORTHS METRO 123", "KMART ONLINE ORDER", "COLES CENTRAL SYDNEY"],
                                    "neighbor_scores": [0.45, 0.42, 0.41],
                                    "second_best_category": "Shopping",
                                    "difference": 0.03,
                                    "rel_threshold": 0.1,
                                    "unique_neighbor_categories": ["Groceries", "Shopping"]
                                }
                            }
                            # Add more example results items if desired, or keep it concise
                        ],
                        "config": { 
                            "categoryColumn": "D",
                            "startRow": "2",
                            "sheetName": "NewTransactionsToClassify",
                            "spreadsheetId": "test-sheet-id-for-classification"
                        }
                    }
                    # Other examples (training_completed, processing, failed_job) are still valuable 
                    # for documentation but might not be directly rendered by default in all UIs 
                    # if only one example per media type is shown. The main endpoint description covers them.
                }
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
        requesting_user_id = request.user_id

        # New call to the refactored status service, which returns a dictionary
        status_dict = get_and_process_status(prediction_id, requesting_user_id)

        # Determine status code based on the dictionary returned by status_service
        # status_service now returns error_code if applicable
        response_status_code = 200  # Default for PENDING, PROCESSING, COMPLETED
        if isinstance(status_dict, dict) and status_dict.get("status") == "error":
            response_status_code = status_dict.get("error_code", 500)
        elif isinstance(status_dict, dict) and status_dict.get("status") == "failed":
            response_status_code = status_dict.get(
                "error_code", 500
            )  # Or a more specific client error if needed

        return jsonify(status_dict), response_status_code

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
        user_id_query = request.args.get("userId")
        user_id = None

        if not user_id_query and request.is_json:
            data = request.get_json()
            if data:
                validated_data, error_response = validate_request_data(
                    UserConfigRequest, data
                )
                if error_response:
                    return error_response
                user_id = validated_data.userId
            else:
                return create_error_response("Missing userId parameter", 400)
        elif user_id_query:
            user_id = user_id_query
        else:
            return create_error_response("Missing userId parameter", 400)

        if not user_id:
            # This case should be covered above, but as a safeguard
            return create_error_response("Missing userId parameter", 400)

        # If user_id is an email, prefix it (assuming this logic is desired)
        if "@" in user_id and not user_id.startswith("google-oauth2|"):
            user_id = f"google-oauth2|{user_id}"

        # Ensure the API key user can only access their own config
        if request.user_id != user_id:
            logger.warning(
                f"API Key mismatch: User {request.user_id} tried to access config for {user_id}. Denying."
            )
            return create_error_response(
                "Permission denied: API key does not match requested user ID", 403
            )

        response = db_client.get_account_by_user_id(user_id)
        if not response:
            # Create default if not found?
            # Or return 404? Returning 404 for now.
            return create_error_response("User configuration not found", 404)

        # Update API key in DB if provided key is different (ensure user owns the config first)
        if request.api_key != response.get("api_key"):
            try:
                db_client.update_user(request.user_id, {"api_key": request.api_key})
                logger.info(f"Updated API key for user {user_id}")
            except Exception as db_update_err:
                logger.error(
                    f"Failed to update API key for user {user_id}: {db_update_err}"
                )
                # Continue returning the config even if update fails

        # Remove sensitive info before returning?
        # response.pop('api_key', None)

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
        user_id = request.user_id
        logger.info(f"Getting API usage stats for user: {user_id}")
        usage_stats = db_client.get_account_usage_stats(user_id)
        if not usage_stats:
            # Return default usage if not found?
            logger.warning(f"No usage stats found for user {user_id}")
            return (
                jsonify(
                    {
                        "total_requests": 0,
                        "daily_requests": 0,
                        "weekly_requests": 0,
                        "monthly_requests": 0,
                        "last_request": None,
                    }
                ),
                200,
            )
            # Or return error?
            # return create_error_response("Failed to retrieve usage statistics", 500)

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
                        },
                        "use_embedding_grouping": {
                            "type": "boolean",
                            "description": "Whether to use embedding-based grouping",
                            "default": False,
                        },
                        "embedding_clustering_method": {
                            "type": "string",
                            "description": "Clustering method for embeddings: hdbscan, dbscan, hierarchical, similarity",
                            "default": "hdbscan",
                        },
                        "embedding_similarity_threshold": {
                            "type": "number",
                            "description": "Similarity threshold for embedding clustering",
                            "default": 0.85,
                        },
                    },
                },
            },
        ],
        "responses": {
            200: {
                "description": "Cleaned descriptions",
                "examples": {
                    "application/json": {
                        "cleaned_descriptions": ["UBER EATS", "AMAZON", "NETFLIX"],
                        "groups": {
                            "UBER EATS": ["UBER EATS"],
                            "AMAZON": ["AMAZON"],
                            "NETFLIX": ["NETFLIX"]
                        }
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
            return create_error_response("Missing descriptions in request", 400)

        descriptions = data["descriptions"]
        if not isinstance(descriptions, list):
            return create_error_response("Descriptions must be a list", 400)

        # Extract grouping parameters
        use_embedding_grouping = data.get("use_embedding_grouping", False)
        embedding_clustering_method = data.get("embedding_clustering_method", "hdbscan")
        embedding_similarity_threshold = data.get("embedding_similarity_threshold", 0.85)

        user_id = request.user_id

        # Check API usage limit
        try:
            usage_stats = db_client.get_account_usage_stats(user_id)
            if usage_stats and usage_stats.get("daily_requests", 0) > 10000:
                return create_error_response("Daily API limit exceeded.", 429)
        except Exception as e:
            logger.warning(f"Error checking API usage for user {user_id}: {e}")

        if use_embedding_grouping:
            # Use the enhanced clean_and_group_transactions function
            try:
                # Create a custom config for the embedding grouping
                from utils.text_utils import CleaningConfig
                config = CleaningConfig()
                config.use_embedding_grouping = True
                config.embedding_clustering_method = embedding_clustering_method
                config.embedding_similarity_threshold = embedding_similarity_threshold
                
                # Call the function with the correct signature
                cleaned_descriptions, grouping_dict = clean_and_group_transactions(descriptions, config)
                
                # Convert grouping_dict to the expected format for the API response
                # grouping_dict maps individual cleaned text to representative text
                # We need to create groups where each group contains all texts that map to the same representative
                groups = {}
                for original_desc, cleaned_desc in zip(descriptions, cleaned_descriptions):
                    representative = grouping_dict.get(cleaned_desc, cleaned_desc)
                    if representative not in groups:
                        groups[representative] = []
                    groups[representative].append(cleaned_desc)
                
                response = {
                    "cleaned_descriptions": cleaned_descriptions,
                    "groups": groups
                }
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Error in embedding-based grouping: {str(e)}")
                # Fallback to regular cleaning if embedding grouping fails
                logger.warning("Falling back to regular text cleaning")
                use_embedding_grouping = False

        # Regular cleaning (fallback or when grouping is not requested)
        cleaned_descriptions = []
        invalid_descriptions = []
        for i, desc in enumerate(descriptions):
            try:
                if not isinstance(desc, str):
                    raise TypeError("Description must be a string")
                if len(desc) > 500:
                    raise ValueError("Description too long (max 500 characters)")

                cleaned = clean_text(desc)
                cleaned_descriptions.append(cleaned)
            except (TypeError, ValueError) as ve:
                invalid_descriptions.append(
                    {
                        "index": i,
                        "description": str(desc)[:50] + "...",
                        "error": str(ve),
                    }
                )
                cleaned_descriptions.append("")
            except Exception as e:
                logger.error(
                    f"Unexpected error cleaning desc '{str(desc)[:50]}...': {e}"
                )
                invalid_descriptions.append(
                    {
                        "index": i,
                        "description": str(desc)[:50] + "...",
                        "error": "Processing error",
                    }
                )
                cleaned_descriptions.append("")

        response = {"cleaned_descriptions": cleaned_descriptions}
        if invalid_descriptions:
            response["warnings"] = {
                "invalid_descriptions": invalid_descriptions,
                "message": "Some descriptions could not be processed properly",
            }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in clean_text endpoint: {str(e)}")
        return create_error_response(str(e), 500)


@app.route("/financial-analytics", methods=["POST"])
@require_api_key
@swag_from(
    {
        "tags": ["Analytics"],
        "summary": "Advanced financial analytics for transactions",
        "description": (
            "Provides comprehensive financial analytics including vendor intelligence, "
            "spending patterns, anomaly detection, savings opportunities, and cash flow predictions. "
            "Analyzes transaction data to generate actionable insights for personal finance management."
        ),
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
                            "description": "List of transaction objects for analysis",
                            "items": {
                                "type": "object",
                                "required": ["description", "amount", "category"],
                                "properties": {
                                    "description": {
                                        "type": "string",
                                        "description": "Transaction description",
                                        "example": "UBER EATS SYDNEY"
                                    },
                                    "amount": {
                                        "type": "number",
                                        "description": "Transaction amount (negative for expenses, positive for income)",
                                        "example": -25.50
                                    },
                                    "category": {
                                        "type": "string",
                                        "description": "Transaction category",
                                        "example": "DinnerBars"
                                    },
                                    "date": {
                                        "type": "string",
                                        "description": "Transaction date (ISO format)",
                                        "example": "2024-01-15T10:30:00Z"
                                    },
                                    "money_in": {
                                        "type": "boolean",
                                        "description": "Whether this is an income transaction",
                                        "example": False
                                    }
                                }
                            }
                        },
                        "analysis_types": {
                            "type": "array",
                            "description": "Types of analysis to perform (optional)",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "vendor_intelligence",
                                    "anomaly_detection",
                                    "spending_patterns",
                                    "savings_opportunities",
                                    "cash_flow_prediction"
                                ]
                            },
                            "example": ["vendor_intelligence", "anomaly_detection"]
                        }
                    },
                    "example": {
                        "transactions": [
                            {
                                "description": "UBER EATS SYDNEY",
                                "amount": -25.50,
                                "category": "DinnerBars",
                                "date": "2024-01-15T10:30:00Z",
                                "money_in": False
                            },
                            {
                                "description": "SALARY DEPOSIT",
                                "amount": 2500.00,
                                "category": "Income",
                                "date": "2024-01-01T09:00:00Z",
                                "money_in": True
                            }
                        ]
                    }
                }
            }
        ],
        "responses": {
            200: {
                "description": "Financial analytics completed successfully",
                "examples": {
                    "application/json": {
                        "user_id": "user123",
                        "analysis_period": {
                            "start_date": "2024-01-01T09:00:00Z",
                            "end_date": "2024-01-15T10:30:00Z",
                            "total_transactions": 2
                        },
                        "insights": {
                            "vendor_intelligence": {
                                "vendors": [
                                    {
                                        "name": "UBER EATS",
                                        "total_spent": 25.50,
                                        "average_transaction": 25.50,
                                        "visit_count": 1
                                    }
                                ],
                                "insights": [
                                    "You spent $26 at UBER EATS (1 visits, avg $25.50/visit)"
                                ]
                            },
                            "anomaly_detection": {
                                "anomalies": [],
                                "insights": []
                            },
                            "spending_patterns": {
                                "patterns": {
                                    "day_of_week": {
                                        "weekend_average": 0.0,
                                        "weekday_average": 25.50
                                    },
                                    "category_distribution": {
                                        "DinnerBars": {
                                            "amount": 25.50,
                                            "percentage": 100.0
                                        }
                                    }
                                },
                                "insights": []
                            },
                            "savings_opportunities": {
                                "opportunities": [
                                    {
                                        "type": "category_reduction",
                                        "category": "DinnerBars",
                                        "current_spending": 25.50,
                                        "potential_savings": 5.10,
                                        "recommendation": "Reduce dinnerbars spending by 20%"
                                    }
                                ],
                                "insights": [
                                    "Reduce dinnerbars spending by 20% to save ~$5"
                                ]
                            },
                            "cash_flow_prediction": {
                                "predictions": {
                                    "weekly_spending_estimate": 6.38,
                                    "weekly_income_estimate": 625.00,
                                    "weekly_net": 618.62
                                },
                                "insights": [
                                    "You save approximately $619 per week"
                                ]
                            }
                        }
                    }
                }
            },
            400: {"description": "Invalid request data"},
            401: {"description": "Invalid or missing API key"},
            500: {"description": "Server error"}
        }
    }
)
def financial_analytics():
    """Advanced financial analytics endpoint."""
    try:
        data = request.get_json()
        if not data:
            return create_error_response("No data provided", 400)

        # Validate the request using Pydantic
        try:
            validated_data = FinancialAnalyticsRequest(**data)
        except ValidationError as e:
            logger.error(f"Validation error in financial analytics: {e}")
            return create_error_response(f"Invalid request data: {e}", 400)

        user_id = request.user_id
        logger.info(f"Processing financial analytics for user {user_id} with {len(validated_data.transactions)} transactions")

        # Convert Pydantic models to dictionaries for processing
        transactions_dict = [tx.dict() for tx in validated_data.transactions]
        
        # Extract excluded categories from request data
        excluded_categories = getattr(validated_data, 'excluded_categories', [])
        
        # Process the financial analytics
        result = process_financial_analytics_request(transactions_dict, user_id, excluded_categories)
        
        if "error" in result:
            logger.error(f"Financial analytics processing failed for user {user_id}: {result['error']}")
            return create_error_response(result["error"], 500)
        
        logger.info(f"Financial analytics completed successfully for user {user_id}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in financial analytics endpoint: {str(e)}", exc_info=True)
        return create_error_response(f"Financial analytics failed: {str(e)}", 500)


# === Helper functions for background tasks ===
def _run_training_task(job_id: str, user_id: str, validated_data: TrainRequest):
    """Runs the training process and updates the async job status."""
    logger.info(f"Starting background training task for job {job_id}, user {user_id}")
    try:
        db_client.update_async_job_status(job_id, "PROCESSING")
        # Assuming process_training_request handles its own errors and returns a dict
        result_dict = process_training_request(validated_data, user_id)

        final_status = "COMPLETED" if result_dict.get("status") != "error" else "FAILED"
        error_msg = (
            result_dict.get("error_message") if final_status == "FAILED" else None
        )
        result_data_json = (
            json.dumps(result_dict) if final_status == "COMPLETED" else None
        )

        db_client.update_async_job_status(
            job_id, final_status, result_data=result_data_json, error_message=error_msg
        )
        logger.info(
            f"Background training task for job {job_id} finished with status: {final_status}"
        )
        return result_dict  # Return the result for synchronous case

    except Exception as e:
        logger.error(
            f"Unhandled exception in _run_training_task for job {job_id}: {e}",
            exc_info=True,
        )
        db_client.update_async_job_status(
            job_id, "FAILED", error_message=f"Internal error: {str(e)}"
        )
        # Return an error structure consistent with what process_training_request might return
        return {
            "status": "error",
            "error_message": f"Internal error during training task: {str(e)}",
            "error_code": 500,
        }


def _run_classification_task(
    job_id: str, user_id: str, validated_data: ClassifyRequest
):
    """Runs the classification process and updates the async job status."""
    logger.info(
        f"Starting background classification task for job {job_id}, user {user_id}"
    )
    try:
        db_client.update_async_job_status(job_id, "PROCESSING")
        # Call the main processing function, passing the full validated data
        # process_classification_request now handles extracting user_categories internally
        result_dict = process_classification_request(validated_data, user_id)

        final_status = "COMPLETED" if result_dict.get("status") != "error" else "FAILED"
        error_msg = (
            result_dict.get("error_message") if final_status == "FAILED" else None
        )
        result_data_json = (
            json.dumps(result_dict) if final_status == "COMPLETED" else None
        )

        db_client.update_async_job_status(
            job_id, final_status, result_data=result_data_json, error_message=error_msg
        )
        logger.info(
            f"Background classification task for job {job_id} finished with status: {final_status}"
        )
        return result_dict  # Return the result for synchronous case

    except Exception as e:
        logger.error(
            f"Unhandled exception in _run_classification_task for job {job_id}: {e}",
            exc_info=True,
        )
        db_client.update_async_job_status(
            job_id, "FAILED", error_message=f"Internal error: {str(e)}"
        )
        # Return an error structure consistent with what process_classification_request might return
        return {
            "status": "error",
            "error_message": f"Internal error during classification task: {str(e)}",
            "error_code": 500,
        }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    # Consider adding call to cleanup_old_webhook_results() on startup?
    # cleanup_old_webhook_results()
    app.run(host="0.0.0.0", port=port)
