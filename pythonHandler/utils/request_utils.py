"""Utility functions for handling Flask requests (validation, decorators, errors)."""

from functools import wraps
from flask import request, jsonify
from pydantic import ValidationError
import logging
from utils.prisma_client import db_client
from utils.db_utils import validate_api_key

logger = logging.getLogger(__name__)


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

        # Check subscription status
        try:
            subscription_status = db_client.get_user_subscription_status(user_id)
            # Allowed statuses based on schema.ts
            allowed_statuses = ["ACTIVE", "TRIALING"]

            if subscription_status not in allowed_statuses:
                logger.warning(
                    f"User {user_id} has invalid subscription status: {subscription_status}"
                )
                return create_error_response(
                    "API key does not have a valid subscription (ACTIVE or TRIALING).",
                    403,
                )

            logger.info(
                f"User {user_id} subscription status ({subscription_status}) validated."
            )

        except Exception as e:
            logger.error(
                f"Failed to verify subscription status for user {user_id}: {e}",
                exc_info=True,
            )
            # Fail closed - if we can't check status, deny access
            return create_error_response("Could not verify subscription status.", 500)

        # Add user_id to request context
        request.user_id = user_id
        request.api_key = api_key

        return f(*args, **kwargs)

    return decorated_function
