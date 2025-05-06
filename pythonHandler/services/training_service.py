"""Service layer for handling training requests."""

import time
import logging
import json
from datetime import datetime
import numpy as np
import pandas as pd
import replicate
from flask import jsonify

from utils.prisma_client import prisma_client
from utils.text_utils import clean_text
from utils.embedding_utils import store_embeddings
from utils.replicate_utils import run_prediction
from config import EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


def process_training_request(validated_data, user_id, api_key):
    """Processes the validated training request data."""
    try:
        transactions = validated_data.transactions
        # Use user_id as the main identifier
        # sheet_id = validated_data.expenseSheetId or f"user_{user_id}" # sheet_id not used currently
        logger.info(
            f"Processing training request - User: {user_id}, Items: {len(transactions)}"
        )

        # Create or update user configuration (Account table)
        try:
            account = prisma_client.get_account_by_user_id(user_id)

            if not account:
                # Define default config structure, aligning with Prisma schema if possible
                default_config = {
                    "userId": user_id,
                    "categorisationRange": "A:Z",  # Example default
                    "columnOrderCategorisation": {
                        "categoryColumn": "E",  # Example default
                        "descriptionColumn": "C",  # Example default
                    },
                    "categorisationTab": None,
                    "api_key": api_key,  # Store the API key used for training
                    # Add other fields from Account schema with defaults if necessary
                }
                prisma_client.insert_account(user_id, default_config)
                logger.info(f"Created default account config for user {user_id}")
            else:
                # Update API key if it has changed or is missing
                if api_key and (
                    not account.get("api_key") or account.get("api_key") != api_key
                ):
                    prisma_client.update_account(user_id, {"api_key": api_key})
                    logger.info(f"Updated API key for user {user_id}")

        except Exception as e:
            # Log but don't necessarily fail the training if config update fails
            logger.warning(f"Error managing user account configuration: {str(e)}")

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
            f"Training with {len(df)} unique descriptions across {len(unique_categories)} categories for user {user_id}"
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
                ("description", "U256"),  # Match length used in categorization
                ("category", "U128"),  # Match length used in categorization
            ],
        )

        # Store index data
        index_id = f"{user_id}_index"
        if not store_embeddings(index_data, index_id, user_id):
            logger.error(f"Failed to store training index {index_id}")
            # Return an error response as index is crucial
            return (
                jsonify(
                    {"status": "error", "error": "Failed to store training index data"}
                ),
                500,
            )
        logger.info(
            f"Stored training index data {index_id} with {len(index_data)} entries"
        )

        # Create a placeholder for the embeddings (to ensure the record exists)
        embedding_id = f"{user_id}"
        placeholder = np.array([[0.0] * EMBEDDING_DIMENSION], dtype=np.float32)
        store_embeddings(placeholder, embedding_id, user_id)
        logger.info(f"Stored/updated placeholder embedding record {embedding_id}")

        # Get descriptions for embedding
        descriptions = df["description"].tolist()

        # Create prediction using the run_prediction function
        prediction = run_prediction(descriptions)
        prediction_id = prediction.id

        # Store initial configuration for status endpoint
        context = {
            "user_id": user_id,
            "status": "processing",
            "type": "training",
            "transaction_count": len(
                transactions
            ),  # Original count before deduplication
            "unique_description_count": len(df),  # Count after deduplication
            "category_count": len(unique_categories),
            "created_at": datetime.now().isoformat(),
            "embedding_id": embedding_id,  # ID for the main embeddings
            "index_id": index_id,  # ID for the index data
        }

        try:
            prisma_client.insert_webhook_result(prediction_id, context)
            logger.info(
                f"Stored initial training context for prediction {prediction_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to store initial context for training prediction {prediction_id}: {e}",
                exc_info=True,
            )
            # Decide if this is fatal. For now, return error as status check depends on it.
            return (
                jsonify(
                    {"status": "error", "error": "Failed to store training job context"}
                ),
                500,
            )

        # === Try Synchronous Completion (Poll for 10 seconds) ===
        SYNC_TIMEOUT = 10  # seconds (can be adjusted)
        sync_end_time = time.time() + SYNC_TIMEOUT
        poll_interval = 0.5  # seconds

        while time.time() < sync_end_time:
            try:
                # Check Replicate prediction status
                current_prediction = replicate.predictions.get(prediction_id)
                replicate_status = current_prediction.status

                if replicate_status == "succeeded":
                    logger.info(
                        f"Training prediction {prediction_id} completed synchronously."
                    )
                    # Process successful completion immediately
                    embeddings = np.array(current_prediction.output, dtype=np.float32)

                    # Validate embedding dimension
                    if embeddings.shape[1] != EMBEDDING_DIMENSION:
                        logger.error(
                            f"Incorrect embedding dimension received: {embeddings.shape[1]}, expected {EMBEDDING_DIMENSION}"
                        )
                        raise ValueError("Incorrect embedding dimension")

                    store_result = store_embeddings(embeddings, embedding_id, user_id)
                    if not store_result:
                        # Log error, but maybe don't raise exception here, let status endpoint handle retry?
                        # For synchronous, it might be better to fail hard.
                        logger.error(
                            f"Failed to store training embeddings {embedding_id} synchronously."
                        )
                        raise Exception(
                            "Failed to store training embeddings synchronously."
                        )
                    logger.info(
                        f"Stored final training embeddings {embedding_id} synchronously."
                    )

                    # Update DB status
                    final_db_record = {
                        "status": "completed",
                        "message": "Training completed successfully (synchronous)",
                        "user_id": user_id,
                        "type": "training",
                        "completed_at": datetime.now().isoformat(),
                        # Include counts from context for consistency
                        "transaction_count": context["transaction_count"],
                        "unique_description_count": context["unique_description_count"],
                        "category_count": context["category_count"],
                    }
                    prisma_client.insert_webhook_result(prediction_id, final_db_record)

                    # Return synchronous success response
                    return jsonify(final_db_record), 200

                elif replicate_status == "failed":
                    error_msg = (
                        current_prediction.error
                        or "Unknown Replicate error during training"
                    )
                    logger.error(
                        f"Training prediction {prediction_id} failed synchronously: {error_msg}"
                    )
                    # Update DB status
                    final_db_record = {
                        "status": "failed",
                        "error": f"Training failed during prediction: {str(error_msg)}",
                        "user_id": user_id,
                        "type": "training",
                    }
                    prisma_client.insert_webhook_result(prediction_id, final_db_record)
                    # Return error response
                    return jsonify(final_db_record), 500

                # If still processing, wait and poll again
                logger.debug(
                    f"Polling training status for {prediction_id}, current: {replicate_status}"
                )
                time.sleep(poll_interval)

            except Exception as poll_error:
                logger.warning(
                    f"Error during synchronous polling for {prediction_id}: {poll_error}"
                )
                # If polling fails significantly, break and return async response
                # (Could add more sophisticated error handling here)
                break  # Exit loop on poll error

        # --- Fallback to Asynchronous Response ---
        # If the loop finished without success/failure (i.e., timed out or poll error)
        logger.info(
            f"Training {prediction_id} did not complete within {SYNC_TIMEOUT}s, returning async response."
        )
        return (
            jsonify(
                {
                    "status": "processing",
                    "prediction_id": prediction_id,
                    "message": "Training started. Check status endpoint for updates.",
                }
            ),
            202,  # Use 202 Accepted for async processing
        )

    except Exception as e:
        logger.error(f"Error processing training request: {e}", exc_info=True)
        # Generic internal server error for unexpected issues in the service layer
        return (
            jsonify(
                {
                    "status": "error",
                    "error": "Internal server error during training processing",
                }
            ),
            500,
        )
