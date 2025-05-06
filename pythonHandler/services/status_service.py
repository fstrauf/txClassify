"""Service layer for handling prediction status checks."""

import logging
import time
import replicate
import numpy as np
from datetime import datetime
from flask import jsonify

from utils.prisma_client import prisma_client
from utils.embedding_utils import store_embeddings, fetch_embeddings
from utils.request_utils import create_error_response
from config import EMBEDDING_DIMENSION

# Import classification pipeline functions
from services.classification_service import (
    _apply_initial_categorization,
    _detect_refunds,
    _detect_transfers,
)

logger = logging.getLogger(__name__)


def get_and_process_status(prediction_id, requesting_user_id):
    """Gets status for Training OR Classification jobs, processes if complete."""
    try:
        process_start_time = time.time()

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
            # Check internal DB for final status as fallback
            try:
                stored_result = prisma_client.get_webhook_result(prediction_id)
                if isinstance(stored_result, dict) and stored_result.get("status") in [
                    "completed",
                    "failed",
                ]:
                    # Security check
                    if stored_result.get("user_id") != requesting_user_id:
                        logger.warning(
                            f"User {requesting_user_id} permission denied for stored result {prediction_id}"
                        )
                        return create_error_response(
                            "Permission denied or prediction not found", 404
                        )

                    logger.info(
                        f"Returning stored final status for {prediction_id} after Replicate fetch error."
                    )
                    status_code = (
                        200 if stored_result.get("status") == "completed" else 500
                    )
                    return jsonify(stored_result), status_code
                else:
                    logger.info(
                        f"No final status found internally for {prediction_id} after Replicate fetch error."
                    )
                    # Return a 200 indicating local/synchronous completion
                    return (
                        jsonify(
                            {
                                "status": "completed",
                                "message": "Local jobs complete synchronously. This status endpoint is primarily for asynchronous Replicate jobs.",
                                "prediction_id": prediction_id,
                            }
                        ),
                        200,
                    )
            except Exception as db_err:
                logger.error(
                    f"Error checking internal DB for {prediction_id} status after Replicate error: {db_err}"
                )
                return create_error_response(
                    "Prediction provider error and internal DB check failed.", 500
                )

        # 2. Handle Non-Successful Replicate Statuses
        if replicate_status == "starting" or replicate_status == "processing":
            return (
                jsonify(
                    {
                        "status": "processing",
                        "message": "Job is processing.",
                        "provider_status": replicate_status,
                    }
                ),
                200,
            )

        elif replicate_status == "failed":
            error_message = prediction.error or "Unknown prediction error from provider"
            logger.error(
                f"Prediction {prediction_id} failed on Replicate: {error_message}"
            )
            # Ensure failure is logged in our DB
            try:
                context = prisma_client.get_webhook_result(prediction_id)
                job_type = (
                    context.get("type", "unknown")
                    if isinstance(context, dict)
                    else "unknown"
                )
                user_id = context.get("user_id") if isinstance(context, dict) else None

                if user_id and user_id != requesting_user_id:
                    return create_error_response("Permission denied", 403)

                final_db_record = {
                    "status": "failed",
                    "error": f"Provider Error: {str(error_message)}",
                    "user_id": user_id or requesting_user_id,
                    "type": job_type,
                    "completed_at": datetime.now().isoformat(),
                }
                prisma_client.insert_webhook_result(prediction_id, final_db_record)
                return jsonify(final_db_record), 500
            except Exception as db_err:
                logger.error(
                    f"Failed to update DB status for Replicate failure {prediction_id}: {db_err}"
                )
                return jsonify({"status": "failed", "error": str(error_message)}), 500

        elif replicate_status != "succeeded":
            logger.warning(
                f"Prediction {prediction_id} has unexpected Replicate status: {replicate_status}"
            )
            try:
                context = prisma_client.get_webhook_result(prediction_id)
                job_type = (
                    context.get("type", "unknown")
                    if isinstance(context, dict)
                    else "unknown"
                )
                user_id = context.get("user_id") if isinstance(context, dict) else None
                if user_id and user_id != requesting_user_id:
                    return create_error_response("Permission denied", 403)

                final_db_record = {
                    "status": "failed",
                    "error": f"Unexpected provider status: {replicate_status}",
                    "user_id": user_id or requesting_user_id,
                    "type": job_type,
                    "completed_at": datetime.now().isoformat(),
                }
                prisma_client.insert_webhook_result(prediction_id, final_db_record)
                return jsonify(final_db_record), 500
            except Exception as db_err:
                logger.error(
                    f"Failed to update DB status for unexpected Replicate status {prediction_id}: {db_err}"
                )
                return (
                    jsonify(
                        {
                            "status": "failed",
                            "error": f"Unexpected provider status: {replicate_status}",
                        }
                    ),
                    500,
                )

        # --- If Replicate status is "succeeded" --- #
        logger.info(
            f"Prediction {prediction_id} succeeded on Replicate. Processing results..."
        )

        # 3. Fetch Job Context from our DB
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
                try:
                    prisma_client.insert_webhook_result(
                        prediction_id,
                        {
                            "status": "failed",
                            "error": "Internal Error: Job context lost",
                            "user_id": requesting_user_id,
                        },
                    )
                except:
                    pass
                return create_error_response("Job context lost during processing", 500)

            job_type = context.get("type")
            user_id = context.get("user_id")

            if user_id != requesting_user_id:
                return create_error_response("Permission denied", 403)

            # Check if already processed
            if (
                context.get("status") == "completed"
                or context.get("status") == "failed"
            ):
                logger.info(
                    f"Job {prediction_id} already has final status '{context.get('status')}' in DB. Re-processing."
                )
                # Allow re-processing by falling through

        except Exception as db_err:
            logger.error(
                f"CRITICAL: Failed to fetch context for {prediction_id} from DB: {db_err}",
                exc_info=True,
            )
            return create_error_response(
                "Failed to retrieve job context from database", 500
            )

        # 4. Get Embeddings from Replicate Output
        try:
            if not prediction.output or not isinstance(prediction.output, list):
                raise ValueError("Invalid or missing prediction output")
            embeddings = np.array(prediction.output, dtype=np.float32)
            if len(embeddings.shape) < 2 or embeddings.shape[1] != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Incorrect embedding dimension/shape: {embeddings.shape}"
                )
        except Exception as e:
            logger.error(
                f"Failed to get/process embeddings from output for {prediction_id}: {e}"
            )
            final_db_record = {
                "status": "failed",
                "error": f"Failed to process prediction output: {str(e)}",
                "user_id": user_id,
                "type": job_type,
                "completed_at": datetime.now().isoformat(),
            }
            prisma_client.insert_webhook_result(prediction_id, final_db_record)
            return jsonify(final_db_record), 500

        # 5. Process based on Job Type
        # --- TRAINING --- #
        if job_type == "training":
            logger.info(
                f"Processing TRAINING completion for {prediction_id}, user {user_id}"
            )
            try:
                embedding_id = context.get("embedding_id", f"{user_id}")
                store_result = store_embeddings(embeddings, embedding_id, user_id)
                if not store_result:
                    raise Exception(
                        f"Failed to store training embeddings {embedding_id}"
                    )
                logger.info(f"Stored final training embeddings {embedding_id}.")

                t_count = context.get("transaction_count", "N/A")
                u_desc_count = context.get("unique_description_count", "N/A")
                cat_count = context.get("category_count", "N/A")
                final_db_record = {
                    "status": "completed",
                    "message": f"Training completed successfully ({u_desc_count} unique descriptions)",
                    "user_id": user_id,
                    "type": "training",
                    "completed_at": datetime.now().isoformat(),
                    "transaction_count": t_count,
                    "unique_description_count": u_desc_count,
                    "category_count": cat_count,
                }
                prisma_client.insert_webhook_result(prediction_id, final_db_record)
                logger.info(
                    f"Training job {prediction_id} processing completed in {time.time() - process_start_time:.2f}s"
                )
                return jsonify(final_db_record), 200
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
                    "completed_at": datetime.now().isoformat(),
                }
                prisma_client.insert_webhook_result(prediction_id, final_db_record)
                return jsonify(final_db_record), 500

        # --- CLASSIFICATION --- #
        elif job_type == "classification":
            logger.info(
                f"Processing CLASSIFICATION completion for {prediction_id}, user {user_id}"
            )
            transactions_input = context.get("transactions_input")
            if not transactions_input or not isinstance(transactions_input, list):
                logger.error(
                    f"Missing/invalid transactions_input in context for {prediction_id}"
                )
                final_db_record = {
                    "status": "failed",
                    "error": "Internal Error: Missing transaction data",
                    "user_id": user_id,
                    "type": "classification",
                    "completed_at": datetime.now().isoformat(),
                }
                prisma_client.insert_webhook_result(prediction_id, final_db_record)
                return jsonify(final_db_record), 500

            if len(embeddings) != len(transactions_input):
                logger.error(
                    f"Embedding/transaction count mismatch for {prediction_id}: {len(embeddings)} vs {len(transactions_input)}"
                )
                final_db_record = {
                    "status": "failed",
                    "error": "Internal Error: Embedding count mismatch",
                    "user_id": user_id,
                    "type": "classification",
                    "completed_at": datetime.now().isoformat(),
                }
                prisma_client.insert_webhook_result(prediction_id, final_db_record)
                return jsonify(final_db_record), 500

            try:
                embeddings_id = context.get(
                    "embeddings_id", f"{prediction_id}_embeddings"
                )
                if not store_embeddings(embeddings, embeddings_id, user_id):
                    logger.error(
                        f"Failed to store classification embeddings {embeddings_id} for {prediction_id}"
                    )
                    raise Exception(
                        f"Failed to store classification embeddings {embeddings_id}"
                    )

                # Run categorization pipeline
                initial_results = _apply_initial_categorization(
                    transactions_input, embeddings, user_id
                )
                if any(
                    "Error:" in res.get("predicted_category", "")
                    for res in initial_results
                ):
                    first_error = next(
                        (
                            res["predicted_category"]
                            for res in initial_results
                            if "Error:" in res.get("predicted_category", "")
                        ),
                        "Categorization failed",
                    )
                    raise Exception(f"Categorization Error: {first_error}")

                results_after_refunds = _detect_refunds(
                    initial_results, embeddings, user_id
                )
                final_results_raw = _detect_transfers(results_after_refunds)
                final_results_clean = final_results_raw  # Keep internal fields for now

                db_record = {
                    "status": "completed",
                    "message": "Classification completed successfully",
                    "type": "classification",
                    "user_id": user_id,
                    "transaction_count": len(final_results_clean),
                    "embeddings_id": embeddings_id,
                    "completed_at": datetime.now().isoformat(),
                }
                prisma_client.insert_webhook_result(prediction_id, db_record)

                response_payload = {
                    "status": "completed",
                    "message": "Classification completed successfully",
                    "results": final_results_clean,
                    "prediction_id": prediction_id,
                    "type": "classification",
                }
                logger.info(
                    f"Classification job {prediction_id} processing completed in {time.time() - process_start_time:.2f}s"
                )
                return jsonify(response_payload), 200

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
                    "completed_at": datetime.now().isoformat(),
                }
                prisma_client.insert_webhook_result(prediction_id, final_db_record)
                return jsonify(final_db_record), 500

        # --- Unknown Job Type --- #
        else:
            logger.error(
                f"Unknown job type '{job_type}' found in context for {prediction_id}"
            )
            final_db_record = {
                "status": "failed",
                "error": f"Internal Error: Unknown job type '{job_type}'",
                "user_id": user_id,
                "completed_at": datetime.now().isoformat(),
            }
            prisma_client.insert_webhook_result(prediction_id, final_db_record)
            return jsonify(final_db_record), 500

    except Exception as e:
        logger.error(
            f"Critical error in get_and_process_status for {prediction_id}: {e}",
            exc_info=True,
        )
        return create_error_response(
            "An unexpected server error occurred while checking status", 500
        )
