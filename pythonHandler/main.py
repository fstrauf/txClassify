import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import replicate
import logging
import sys
import time
from typing import List, Optional, Union, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import uuid
from utils.prisma_client import db_client
from dotenv import load_dotenv
import io
from flasgger import Swagger, swag_from
from pydantic import ValidationError

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
)
from utils.text_utils import clean_text
from utils.request_utils import (
    validate_request_data,
    create_error_response,
    require_api_key,
)
from services.training_service import process_training_request
from services.classification_service import process_classification_request
from services.status_service import get_and_process_status

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
        db_client.create_async_job(job_id, api_key_user_id, 'training', 'PENDING')

        # 2. Submit actual work to executor
        future = executor.submit(
            _run_training_task, 
            job_id, 
            api_key_user_id, 
            validated_data
        )

        try:
            # 3. Try to get result with timeout
            job_result_dict = future.result(timeout=8.5)
            
            status_code = job_result_dict.get("error_code", 200) if isinstance(job_result_dict, dict) else 200
            if isinstance(job_result_dict, dict) and job_result_dict.get("status") == "error":
                status_code = job_result_dict.get("error_code", 500)

            return jsonify(job_result_dict), status_code

        except TimeoutError:
            logger.info(f"Training job {job_id} for user {api_key_user_id} taking longer, returning 202.")
            return jsonify({
                "status": "processing", 
                "prediction_id": job_id, 
                "message": "Training request accepted and is processing in the background."
            }), 202
        except Exception as e:
            logger.error(f"Error managing future for training job {job_id}: {e}", exc_info=True)
            db_client.update_async_job_status(job_id, 'FAILED', error_message=f"Internal error managing job: {str(e)}")
            return jsonify({
                "status": "error", 
                "error_message": "Failed to process training request due to an internal error.",
                "prediction_id": job_id
            }), 500

    except Exception as e:
        logger.error(f"Error in train_model endpoint: {e}", exc_info=True)
        return create_error_response(f"Server error in training endpoint: {str(e)}", 500)


@app.route("/classify", methods=["POST"])
@require_api_key
@swag_from(
    {
        "tags": ["Classification"],
        "summary": "Classify transactions",
        "description": "Classifies a list of transaction descriptions. If processing takes longer than ~8.5 seconds, returns a 202 Accepted with a prediction_id for polling.",
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
                    "$ref": "#/definitions/ClassifyRequest" # Reference the model definition
                },
            },
        ],
        "definitions": { # Add definition for Swagger UI
            "ClassifyRequest": {
                "type": "object",
                "required": ["transactions"],
                "properties": {
                    "transactions": {
                        "type": "array",
                        "description": "List of transactions (strings or objects) to classify",
                        "items": {
                            "oneOf": [
                                {"type": "string"},
                                {"$ref": "#/definitions/TransactionInput"}
                            ]
                        }
                    },
                    "user_categories": {
                        "type": "array",
                        "description": "(Optional) List of user category objects ({id, name}) for LLM assist",
                        "items": {
                             "type": "object",
                             "properties": {
                                 "id": {"type": "string"},
                                 "name": {"type": "string"}
                             }
                         }
                    },
                    "spreadsheetId": {"type": "string", "description": "(Optional) Google Sheet ID for context/output"},
                    "sheetName": {"type": "string", "default": "new_transactions", "description": "(Optional) Sheet name"},
                    "categoryColumn": {"type": "string", "default": "E", "description": "(Optional) Category column letter"},
                    "startRow": {"type": "string", "default": "1", "description": "(Optional) Starting row number"}
                }
            },
            "TransactionInput": {
                 "type": "object",
                 "required": ["description"],
                 "properties": {
                     "description": {"type": "string"},
                     "money_in": {"type": "boolean", "description": "True for income, False for expense"}
                     # Amount might also be useful here if available from frontend
                 }
            }
        },
        "responses": {
            200: {"description": "Classification completed synchronously"},
            202: {"description": "Classification accepted for background processing"},
            400: {"description": "Invalid request data"},
            401: {"description": "Invalid or missing API key"},
            409: {"description": "Conflict (e.g., embedding mismatch, requires re-training)"},
            500: {"description": "Server error"},
        },
    }
)
def classify_transactions_async():
    """Attempts synchronous classification first, falls back to async if needed."""
    try:
        user_id = request.user_id # Get user_id associated with the API key
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
            first_error = error_details[0]['msg'] if error_details else "Invalid data"
            return create_error_response(f"Validation error: {first_error}", 400)

        # Generate a unique job ID
        job_id = str(uuid.uuid4())

        # --- Critical Section: Record Job Intention --- #
        try:
            db_client.create_async_job(job_id, user_id, 'classification', 'PENDING')
            logger.info(f"Created PENDING job record {job_id} for user {user_id}")
        except Exception as db_err:
            logger.error(f"Failed to create PENDING job record for job {job_id}: {db_err}", exc_info=True)
            # Fail fast if we can't even record the job
            return create_error_response("Failed to initiate classification job.", 500)
        # --- End Critical Section --- #

        # Submit actual work to the executor
        # Pass the full validated_data object, which includes user_categories
        future = executor.submit(
            _run_classification_task,
            job_id,         # Pass the generated job ID
            user_id,        # Pass the user ID
            validated_data  # Pass the whole validated request model
        )

        # Try to get the result synchronously with a timeout
        try:
            # Wait for the result for up to ~8.5 seconds
            job_result_dict = future.result(timeout=8.5)

            # Check the structure and status of the result from the task
            if not isinstance(job_result_dict, dict):
                 logger.error(f"Task for job {job_id} returned unexpected type: {type(job_result_dict)}")
                 # Update job status to FAILED as task result was bad
                 db_client.update_async_job_status(job_id, 'FAILED', error_message="Internal error: Invalid result format from task.")
                 return create_error_response("Internal error during classification processing.", 500)

            # Determine status code based on result
            status_code = 200 # Default success
            if job_result_dict.get("status") == "error":
                status_code = job_result_dict.get("error_code", 500)
            
            logger.info(f"Classification job {job_id} completed synchronously. Returning {status_code}.")
            # Return the full result dictionary (contains status, message, results/error)
            return jsonify(job_result_dict), status_code

        except TimeoutError:
            # Task didn't complete within the timeout
            logger.info(f"Classification job {job_id} for user {user_id} taking longer, returning 202 Accepted.")
            # The job status remains PENDING or will be updated by the background task
            return jsonify({
                "status": "processing",
                "prediction_id": job_id,
                "message": "Classification request accepted and is processing in the background."
            }), 202

        except Exception as future_err:
            # Error retrieving result from the future (e.g., task raised an unhandled exception)
            logger.error(f"Error retrieving result for classification job {job_id}: {future_err}", exc_info=True)
            # The background task *should* handle its own errors and update the DB,
            # but we update here just in case it failed before it could.
            db_client.update_async_job_status(job_id, 'FAILED', error_message=f"Internal error managing job future: {str(future_err)}")
            return jsonify({
                "status": "error",
                "error_message": "Failed to process classification request due to an internal error.",
                "prediction_id": job_id
            }), 500

    except Exception as route_err:
        # Catch-all for errors in the route handler itself (before job submission)
        logger.error(f"Error in /classify route for user {request.user_id if hasattr(request, 'user_id') else 'Unknown'}: {route_err}", exc_info=True)
        return create_error_response(f"Server error in classification endpoint: {str(route_err)}", 500)


@app.route("/status/<prediction_id>", methods=["GET"])
@require_api_key
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
        requesting_user_id = request.user_id

        # New call to the refactored status service, which returns a dictionary
        status_dict = get_and_process_status(prediction_id, requesting_user_id)
        
        # Determine status code based on the dictionary returned by status_service
        # status_service now returns error_code if applicable
        response_status_code = 200 # Default for PENDING, PROCESSING, COMPLETED
        if isinstance(status_dict, dict) and status_dict.get("status") == "error":
            response_status_code = status_dict.get("error_code", 500)
        elif isinstance(status_dict, dict) and status_dict.get("status") == "failed":
             response_status_code = status_dict.get("error_code", 500) # Or a more specific client error if needed

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
            return create_error_response("Missing descriptions in request", 400)

        descriptions = data["descriptions"]
        if not isinstance(descriptions, list):
            return create_error_response("Descriptions must be a list", 400)

        user_id = request.user_id

        # Check API usage limit
        try:
            usage_stats = db_client.get_account_usage_stats(user_id)
            if usage_stats and usage_stats.get("daily_requests", 0) > 10000:
                return create_error_response("Daily API limit exceeded.", 429)
        except Exception as e:
            logger.warning(f"Error checking API usage for user {user_id}: {e}")

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


def cleanup_old_webhook_results():
    """Clean up old webhook results from the database."""
    try:
        webhook_cutoff_date = datetime.now() - timedelta(days=7)
        embeddings_cutoff_date = datetime.now() - timedelta(days=30)

        logger.info(f"Cleaning up webhook results older than {webhook_cutoff_date}")
        # deleted_count = db_client.delete_old_webhook_results(webhook_cutoff_date)
        # logger.info(
            # f"Cleaned up {deleted_count} webhook results older than {webhook_cutoff_date}"
        # )

        logger.info(
            f"Cleaning up embeddings/contexts older than {embeddings_cutoff_date}"
        )
        try:
            # Assumes a method exists in prisma_client
            # deleted_embeds = db_client.delete_old_embeddings_and_contexts(
            #     embeddings_cutoff_date
            # )
            # logger.info(
            #     f"Cleaned up {deleted_embeds} embeddings/contexts older than {embeddings_cutoff_date}"
            # )
            pass # Placeholder if methods are removed for now
        except AttributeError:
            logger.warning(
                "db_client.delete_old_embeddings_and_contexts method not found. Skipping cleanup."
            )
        except Exception as e:
            logger.error(f"Error cleaning up embeddings and contexts: {e}")

    except Exception as e:
        logger.error(f"Error cleaning up old webhook results: {e}")


# === Helper functions for background tasks ===
def _run_training_task(job_id: str, user_id: str, validated_data: TrainRequest):
    """Runs the training process and updates the async job status."""
    logger.info(f"Starting background training task for job {job_id}, user {user_id}")
    try:
        db_client.update_async_job_status(job_id, 'PROCESSING')
        # Assuming process_training_request handles its own errors and returns a dict
        result_dict = process_training_request(validated_data, user_id)

        final_status = 'COMPLETED' if result_dict.get("status") != "error" else 'FAILED'
        error_msg = result_dict.get("error_message") if final_status == 'FAILED' else None
        result_data_json = json.dumps(result_dict) if final_status == 'COMPLETED' else None

        db_client.update_async_job_status(job_id, final_status, result_data=result_data_json, error_message=error_msg)
        logger.info(f"Background training task for job {job_id} finished with status: {final_status}")
        return result_dict # Return the result for synchronous case

    except Exception as e:
        logger.error(f"Unhandled exception in _run_training_task for job {job_id}: {e}", exc_info=True)
        db_client.update_async_job_status(job_id, 'FAILED', error_message=f"Internal error: {str(e)}")
        # Return an error structure consistent with what process_training_request might return
        return {"status": "error", "error_message": f"Internal error during training task: {str(e)}", "error_code": 500}

def _run_classification_task(job_id: str, user_id: str, validated_data: ClassifyRequest):
    """Runs the classification process and updates the async job status."""
    logger.info(f"Starting background classification task for job {job_id}, user {user_id}")
    try:
        db_client.update_async_job_status(job_id, 'PROCESSING')
        # Call the main processing function, passing the full validated data
        # process_classification_request now handles extracting user_categories internally
        result_dict = process_classification_request(validated_data, user_id)

        final_status = 'COMPLETED' if result_dict.get("status") != "error" else 'FAILED'
        error_msg = result_dict.get("error_message") if final_status == 'FAILED' else None
        result_data_json = json.dumps(result_dict) if final_status == 'COMPLETED' else None

        db_client.update_async_job_status(job_id, final_status, result_data=result_data_json, error_message=error_msg)
        logger.info(f"Background classification task for job {job_id} finished with status: {final_status}")
        return result_dict # Return the result for synchronous case

    except Exception as e:
        logger.error(f"Unhandled exception in _run_classification_task for job {job_id}: {e}", exc_info=True)
        db_client.update_async_job_status(job_id, 'FAILED', error_message=f"Internal error: {str(e)}")
        # Return an error structure consistent with what process_classification_request might return
        return {"status": "error", "error_message": f"Internal error during classification task: {str(e)}", "error_code": 500}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    # Consider adding call to cleanup_old_webhook_results() on startup?
    # cleanup_old_webhook_results()
    app.run(host="0.0.0.0", port=port)
