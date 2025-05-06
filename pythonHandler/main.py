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
from utils.prisma_client import prisma_client
from dotenv import load_dotenv
import io
from flasgger import Swagger, swag_from

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
        connected = prisma_client.connect()
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
            prisma_client.disconnect()
            logger.info("Disconnected from database")
    except Exception as e:
        logger.error(f"Error disconnecting from database: {e}")


@app.before_request
def ensure_db_connection():
    """Ensure database connection is active before each request."""
    try:
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
    try:
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

        # Call the service function
        response, status_code = process_training_request(
            validated_data, request.user_id, request.api_key
        )
        return response, status_code

    except Exception as e:
        logger.error(f"Error in train_model endpoint: {e}", exc_info=True)
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
    """Attempts synchronous classification first, falls back to async if needed."""
    try:
        user_id = request.user_id

        data = request.get_json()
        if not data:
            return create_error_response("No data provided", 400)
        validated_data, error_response = validate_request_data(ClassifyRequest, data)
        if error_response:
            return error_response

        # Call the service function
        response, status_code = process_classification_request(validated_data, user_id)
        return response, status_code

    except Exception as e:
        logger.error(f"Critical error in /classify endpoint: {e}", exc_info=True)
        return create_error_response(
            f"An unexpected server error occurred: {str(e)}", 500
        )


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

        # Call the service function
        response, status_code = get_and_process_status(
            prediction_id, requesting_user_id
        )
        return response, status_code

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

        response = prisma_client.get_account_by_user_id(user_id)
        if not response:
            # Create default if not found?
            # Or return 404? Returning 404 for now.
            return create_error_response("User configuration not found", 404)

        # Update API key in DB if provided key is different (ensure user owns the config first)
        if request.api_key != response.get("api_key"):
            try:
                prisma_client.update_account(user_id, {"api_key": request.api_key})
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
        usage_stats = prisma_client.get_account_usage_stats(user_id)
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
            usage_stats = prisma_client.get_account_usage_stats(user_id)
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
        deleted_count = prisma_client.delete_old_webhook_results(webhook_cutoff_date)
        logger.info(
            f"Cleaned up {deleted_count} webhook results older than {webhook_cutoff_date}"
        )

        logger.info(
            f"Cleaning up embeddings/contexts older than {embeddings_cutoff_date}"
        )
        try:
            # Assumes a method exists in prisma_client
            deleted_embeds = prisma_client.delete_old_embeddings_and_contexts(
                embeddings_cutoff_date
            )
            logger.info(
                f"Cleaned up {deleted_embeds} embeddings/contexts older than {embeddings_cutoff_date}"
            )
        except AttributeError:
            logger.warning(
                "prisma_client.delete_old_embeddings_and_contexts method not found. Skipping cleanup."
            )
        except Exception as e:
            logger.error(f"Error cleaning up embeddings and contexts: {e}")

    except Exception as e:
        logger.error(f"Error cleaning up old webhook results: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    # Consider adding call to cleanup_old_webhook_results() on startup?
    # cleanup_old_webhook_results()
    app.run(host="0.0.0.0", port=port)
