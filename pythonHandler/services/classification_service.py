"""Service layer for handling classification requests and categorization pipeline."""

import logging
import re
import os  # Import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import time
import replicate
from datetime import datetime
from flask import jsonify

from utils.embedding_utils import fetch_embeddings, store_embeddings
from utils.prisma_client import prisma_client
from utils.replicate_utils import run_prediction
from utils.request_utils import create_error_response
from config import EMBEDDING_DIMENSION
from utils.text_utils import clean_text  # Import clean_text

logger = logging.getLogger(__name__)


def process_classification_request(validated_data, user_id, sync_timeout):
    """Processes the validated classification request data."""
    try:
        start_time = time.time()

        # 1. Prepare Input Data from validated request
        transactions_input_for_context = []
        original_descriptions = []  # Store original descriptions
        # descriptions_to_embed = [] # This will now be cleaned descriptions

        for tx_input_item in validated_data.transactions:
            if isinstance(tx_input_item, str):
                desc = tx_input_item
                money_in = None
                amount = None  # Assume amount is not available for string inputs
            else:  # It's a TransactionInput object
                desc = tx_input_item.description
                money_in = tx_input_item.money_in
                amount = (
                    tx_input_item.amount if hasattr(tx_input_item, "amount") else None
                )

            original_descriptions.append(desc)  # Store original

        if not original_descriptions:
            logger.error(
                f"No valid descriptions found in classification request for user {user_id}"
            )
            return create_error_response(
                "No valid transaction descriptions provided", 400
            )

        # --- Clean Descriptions ---
        try:
            # Use the imported clean_text function
            cleaned_descriptions = clean_text(original_descriptions)
            if len(cleaned_descriptions) != len(original_descriptions):
                logger.error(
                    f"Mismatch in length after cleaning: {len(original_descriptions)} vs {len(cleaned_descriptions)}"
                )
                # Handle error appropriately - maybe raise exception or return error response
                raise ValueError("Cleaning resulted in description count mismatch.")
            logger.info(
                f"Cleaned {len(original_descriptions)} descriptions for user {user_id}"
            )

            # Prepare context data with both original and cleaned descriptions
            for i, original_desc in enumerate(original_descriptions):
                tx_input = validated_data.transactions[
                    i
                ]  # Get original input item again
                money_in = None
                amount = None
                if not isinstance(tx_input, str):
                    money_in = tx_input.money_in
                    amount = tx_input.amount if hasattr(tx_input, "amount") else None

                transactions_input_for_context.append(
                    {
                        "original_description": original_desc,
                        "cleaned_description": cleaned_descriptions[i],
                        "money_in": money_in,
                        "amount": amount,
                    }
                )

        except Exception as clean_error:
            logger.error(
                f"Error cleaning descriptions for user {user_id}: {clean_error}",
                exc_info=True,
            )
            return create_error_response(
                f"Failed during text cleaning: {str(clean_error)}", 500
            )
        # --- End Cleaning ---

        logger.info(
            f"Processing classification for {len(transactions_input_for_context)} txns, user {user_id}"
        )

        # 2. Start Replicate Prediction using CLEANED descriptions
        try:
            prediction = run_prediction(
                cleaned_descriptions
            )  # Use cleaned descriptions
            prediction_id = prediction.id
        except Exception as e:
            logger.error(
                f"Failed to start Replicate prediction for classification: {e}",
                exc_info=True,
            )
            # Use create_error_response format
            return create_error_response(
                f"Failed to start embedding prediction: {str(e)}", 502
            )

        # 3. Store Initial Context
        context = {
            "user_id": user_id,
            "status": "processing",
            "type": "classification",
            "transactions_input": transactions_input_for_context,  # Store prepared input
            "created_at": datetime.now().isoformat(),
        }
        try:
            prisma_client.insert_webhook_result(prediction_id, context)
            logger.info(
                f"Stored initial classification context for prediction {prediction_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to store context for classification prediction {prediction_id}: {e}"
            )
            return create_error_response(f"Failed to store job context: {str(e)}", 500)

        # 4. Try Synchronous Completion
        sync_end_time = time.time() + sync_timeout
        poll_interval = 0.5

        while time.time() < sync_end_time:
            try:
                current_prediction = replicate.predictions.get(prediction_id)
                replicate_status = current_prediction.status

                if replicate_status == "succeeded":
                    logger.info(
                        f"Classification prediction {prediction_id} completed synchronously."
                    )
                    try:
                        embeddings = np.array(
                            current_prediction.output, dtype=np.float32
                        )

                        # Validate embedding dimension
                        if embeddings.shape[1] != EMBEDDING_DIMENSION:
                            logger.error(
                                f"Incorrect embedding dimension received: {embeddings.shape[1]}, expected {EMBEDDING_DIMENSION}"
                            )
                            raise ValueError("Incorrect embedding dimension")
                        if embeddings.shape[0] != len(transactions_input_for_context):
                            logger.error(
                                f"Embedding count mismatch: {embeddings.shape[0]} vs {len(transactions_input_for_context)}"
                            )
                            raise ValueError("Embedding count mismatch")

                        # Store embeddings (optional for sync, but good practice)
                        embeddings_id = f"{prediction_id}_embeddings"
                        # Store with cleaned descriptions as context if needed? Or just user_id?
                        # Let's assume store_embeddings only needs user_id for now.
                        if not store_embeddings(embeddings, embeddings_id, user_id):
                            logger.warning(
                                f"Failed to store embeddings {embeddings_id} synchronously."
                            )
                            # Don't necessarily fail the request, but log it.

                        # --- Run Categorization Pipeline ---
                        # Pass the context list which now contains original & cleaned descriptions
                        initial_results = _apply_initial_categorization(
                            transactions_input_for_context, embeddings, user_id
                        )
                        # Check for errors from categorization itself
                        if any(
                            "Error:" in res.get("predicted_category", "")
                            for res in initial_results
                        ):
                            logger.error(
                                f"Categorization pipeline failed synchronously for {prediction_id}. Check previous logs."
                            )
                            # Find the first error message
                            first_error = next(
                                (
                                    res["predicted_category"]
                                    for res in initial_results
                                    if "Error:" in res.get("predicted_category", "")
                                ),
                                "Categorization failed",
                            )
                            raise Exception(first_error)

                        results_after_refunds = _detect_refunds(
                            initial_results,
                            embeddings,
                            user_id,  # Pass embeddings if needed by future logic
                        )
                        final_results = _detect_transfers(results_after_refunds)
                        # --- End Pipeline ---

                        # Update status in DB
                        completion_record = {
                            "status": "completed",
                            "message": "Classification completed successfully (synchronous)",
                            "type": "classification",
                            "user_id": user_id,
                            "transaction_count": len(final_results),
                            "embeddings_id": embeddings_id,
                            "completed_at": datetime.now().isoformat(),
                        }
                        prisma_client.insert_webhook_result(
                            prediction_id, completion_record
                        )

                        # Return synchronous response with results
                        return (
                            jsonify(
                                {
                                    "status": "completed",
                                    "message": "Classification completed successfully",
                                    "results": final_results,
                                    "prediction_id": prediction_id,
                                }
                            ),
                            200,
                        )

                    except Exception as process_error:
                        logger.error(
                            f"Error processing classification results synchronously for {prediction_id}: {process_error}",
                            exc_info=True,
                        )
                        # Update DB to reflect processing failure
                        prisma_client.insert_webhook_result(
                            prediction_id,
                            {
                                "status": "failed",
                                "error": f"Failed to process results synchronously: {str(process_error)}",
                                "user_id": user_id,
                                "type": "classification",
                            },
                        )
                        # Return an error to the user
                        return create_error_response(
                            f"Failed to process classification results: {str(process_error)}",
                            500,
                        )

                elif replicate_status == "failed":
                    error_msg = (
                        current_prediction.error
                        or "Unknown Replicate error during classification"
                    )
                    logger.error(
                        f"Classification prediction {prediction_id} failed synchronously: {error_msg}"
                    )
                    # Update DB status
                    prisma_client.insert_webhook_result(
                        prediction_id,
                        {
                            "status": "failed",
                            "error": f"Classification failed during prediction: {str(error_msg)}",
                            "user_id": user_id,
                            "type": "classification",
                        },
                    )
                    return create_error_response(
                        f"Classification failed on provider: {error_msg}", 502
                    )

                # If still processing, wait
                logger.debug(
                    f"Polling classification status for {prediction_id}, current: {replicate_status}"
                )
                time.sleep(poll_interval)

            except Exception as poll_error:
                logger.warning(
                    f"Error during synchronous polling for classification {prediction_id}: {poll_error}"
                )
                # Break the loop on error and return async response
                break

        # 5. Fallback to Asynchronous Response
        logger.info(
            f"Classification {prediction_id} did not complete within {sync_timeout}s, returning async response."
        )
        return (
            jsonify(
                {
                    "status": "processing",
                    "prediction_id": prediction_id,
                    "message": "Classification in progress. Check status endpoint for results.",
                }
            ),
            202,  # HTTP 202 Accepted
        )

    except Exception as e:
        logger.error(
            f"Critical error processing classification request: {e}", exc_info=True
        )
        return create_error_response(
            "Internal server error during classification processing", 500
        )


# === Categorization Pipeline Helpers ===


def _apply_initial_categorization(
    transactions_input: List[
        Dict[str, Any]
    ],  # Now contains original_description, cleaned_description, etc.
    input_embeddings: np.ndarray,
    user_id: str,
) -> List[Dict[str, Any]]:
    """Performs initial categorization based on similarity and returns top 2 matches."""
    results = []
    # Define confidence thresholds
    MIN_ABSOLUTE_CONFIDENCE = 0.85
    MIN_RELATIVE_CONFIDENCE_DIFF = 0.03
    NEIGHBOR_COUNT = 3

    # --- Check for Debug Mode ---
    is_debug = os.getenv("TX_CLASSIFY_DEBUG") == "true"
    logger.debug(f"Classification debug mode: {is_debug}")

    try:
        trained_embeddings = fetch_embeddings(f"{user_id}")
        trained_data = fetch_embeddings(f"{user_id}_index")

        if trained_embeddings.size == 0 or trained_data.size == 0:
            logger.error(
                f"No training data/index found for user {user_id} during categorization"
            )
            # Return error marker in results
            return [
                {
                    # Use original_description for the final output narrative
                    "narrative": tx["original_description"],
                    "cleaned_narrative": tx[
                        "cleaned_description"
                    ],  # Add cleaned for debugging
                    "predicted_category": "Error: Model/Index not found",
                    "similarity_score": 0.0,
                    "money_in": tx.get("money_in"),
                    "amount": tx.get("amount"),
                    "adjustment_info": {"reason": "Training data or index missing"},
                    "debug_info": None,  # Ensure field exists
                }
                for tx in transactions_input
            ]

        if not trained_data.dtype.names or "category" not in trained_data.dtype.names:
            logger.error(
                f"Trained index data for user {user_id} missing 'category' field."
            )
            return [
                {
                    # Use original_description for the final output narrative
                    "narrative": tx["original_description"],
                    "cleaned_narrative": tx[
                        "cleaned_description"
                    ],  # Add cleaned for debugging
                    "predicted_category": "Error: Invalid Index",
                    "similarity_score": 0.0,
                    "money_in": tx.get("money_in"),
                    "amount": tx.get("amount"),
                    "debug_info": None,  # Ensure field exists
                }
                for tx in transactions_input
            ]

        # Ensure input embeddings have the correct shape (N, D)
        if len(input_embeddings.shape) != 2:
            logger.error(
                f"Input embeddings have incorrect shape: {input_embeddings.shape}"
            )
            raise ValueError("Input embeddings shape mismatch")

        # Ensure trained embeddings have the correct shape (M, D)
        if len(trained_embeddings.shape) != 2:
            logger.error(
                f"Trained embeddings have incorrect shape: {trained_embeddings.shape}"
            )
            # Attempt to reshape if it's a flat array from placeholder loading
            if (
                len(trained_embeddings.shape) == 1
                and trained_embeddings.size == input_embeddings.shape[1]
            ):
                trained_embeddings = trained_embeddings.reshape(1, -1)
                logger.warning("Reshaped trained embeddings from flat array.")
            else:
                raise ValueError("Trained embeddings shape mismatch")

        # Check dimension compatibility
        if input_embeddings.shape[1] != trained_embeddings.shape[1]:
            logger.error(
                f"Embedding dimension mismatch: Input {input_embeddings.shape[1]}, Trained {trained_embeddings.shape[1]}"
            )
            raise ValueError("Embedding dimension mismatch")

        similarities = cosine_similarity(input_embeddings, trained_embeddings)

        for i, tx in enumerate(transactions_input):
            money_in = tx.get("money_in")
            amount = tx.get("amount")
            cleaned_desc = tx[
                "cleaned_description"
            ]  # Use cleaned description for logging/comparison if needed
            original_desc = tx[
                "original_description"
            ]  # Use original for final output narrative

            current_similarities = similarities[i]
            num_trained_samples = similarities.shape[1]
            if num_trained_samples == 0:
                logger.warning(
                    f"No trained samples to compare against for narrative: {cleaned_desc}"  # Log cleaned desc
                )
                results.append(
                    {
                        "narrative": original_desc,  # Return original desc
                        "cleaned_narrative": cleaned_desc,  # Add cleaned for debugging
                        "predicted_category": "Unknown",
                        "similarity_score": 0.0,
                        "second_predicted_category": "Unknown",
                        "second_similarity_score": 0.0,
                        "money_in": money_in,
                        "amount": amount,
                        "adjustment_info": {"unknown_reason": "No trained samples"},
                        "debug_info": None,  # Ensure field exists
                    }
                )
                continue

            sorted_indices = np.argsort(-current_similarities)[
                : min(num_trained_samples, NEIGHBOR_COUNT + 5)
            ]

            best_match_idx = -1
            best_category = "Unknown"
            best_score = 0.0
            second_best_category = "Unknown"
            second_best_score = 0.0
            neighbor_categories = []
            neighbor_scores = []  # Store neighbor scores for debug

            # Find top N valid neighbors and their categories/scores
            valid_neighbors_found = 0
            processed_neighbor_indices = []
            for k_idx in sorted_indices:
                if k_idx < len(trained_data):
                    try:
                        neighbor_category = str(trained_data[k_idx]["category"])
                        neighbor_score = float(current_similarities[k_idx])
                        processed_neighbor_indices.append(k_idx)

                        if valid_neighbors_found == 0:
                            best_match_idx = k_idx
                            best_category = neighbor_category
                            best_score = neighbor_score
                        elif valid_neighbors_found == 1:
                            second_best_category = neighbor_category
                            second_best_score = neighbor_score

                        if valid_neighbors_found < NEIGHBOR_COUNT:
                            neighbor_categories.append(neighbor_category)
                            neighbor_scores.append(neighbor_score)  # Store score

                        valid_neighbors_found += 1
                        if valid_neighbors_found >= NEIGHBOR_COUNT + 2:
                            break

                    except IndexError as e:
                        logger.warning(
                            f"IndexError accessing trained_data at index {k_idx}: {e}"
                        )
                    except Exception as e:
                        logger.error(f"Error processing neighbor at index {k_idx}: {e}")
                else:
                    logger.warning(
                        f"Neighbor index {k_idx} out of bounds for trained_data (len {len(trained_data)})"
                    )

            # Determine final category based on thresholds and neighbor consistency
            unique_neighbor_categories = set(neighbor_categories)
            has_conflicting_neighbors = len(unique_neighbor_categories) > 1
            final_category = "Unknown"
            reason = ""
            debug_details: Optional[Dict[str, Any]] = None  # Initialize debug info

            # --- Refined Decision Logic ---
            meets_absolute_confidence = best_score >= MIN_ABSOLUTE_CONFIDENCE
            meets_relative_confidence = (
                best_score - second_best_score
            ) >= MIN_RELATIVE_CONFIDENCE_DIFF
            neighbors_are_consistent = not has_conflicting_neighbors

            if meets_absolute_confidence:
                if meets_relative_confidence:
                    # Standout case: High absolute confidence AND clear winner over the second match. Ignore neighbors.
                    final_category = best_category
                elif neighbors_are_consistent:
                    # Not a clear winner, but absolute confidence met AND neighbors agree.
                    final_category = best_category
                    reason = f"Accepted due to neighbor consistency despite low relative confidence (Diff: {best_score - second_best_score:.2f})"
                    if is_debug:
                        debug_details = {
                            "reason_code": "ACCEPTED_BY_NEIGHBORS",
                            "best_score": best_score,
                            "second_best_score": second_best_score,
                            "difference": best_score - second_best_score,
                            "threshold": MIN_RELATIVE_CONFIDENCE_DIFF,
                            "best_category": best_category,
                            "neighbor_categories": neighbor_categories,
                        }
                # else: final_category remains "Unknown" because absolute confidence met, but relative confidence is low AND neighbors conflict.

            # --- Logging for "Unknown" Cases ---
            if final_category == "Unknown":
                # Determine the primary reason for being "Unknown"
                if not meets_absolute_confidence:
                    reason = f"Low absolute confidence ({best_score:.2f} < {MIN_ABSOLUTE_CONFIDENCE})"
                    if is_debug:
                        debug_details = {
                            "reason_code": "LOW_ABS_CONF",
                            "best_score": best_score,
                            "threshold": MIN_ABSOLUTE_CONFIDENCE,
                            "rejected_best_category": best_category,
                        }
                elif not meets_relative_confidence and not neighbors_are_consistent:
                    reason = f"Low relative confidence (Diff: {best_score - second_best_score:.2f} < {MIN_RELATIVE_CONFIDENCE_DIFF}) AND Conflicting neighbors: {list(unique_neighbor_categories)}"
                    if is_debug:
                        debug_details = {
                            "reason_code": "LOW_REL_CONF_AND_CONFLICTING_NEIGHBORS",
                            "best_score": best_score,
                            "second_best_score": second_best_score,
                            "difference": best_score - second_best_score,
                            "rel_threshold": MIN_RELATIVE_CONFIDENCE_DIFF,
                            "neighbor_categories": neighbor_categories,
                            "unique_neighbor_categories": list(
                                unique_neighbor_categories
                            ),
                            "rejected_best_category": best_category,
                            "second_best_category": second_best_category,
                        }
                # This case should theoretically not be hit if the logic above is correct,
                # but included for completeness.
                elif (
                    not meets_relative_confidence
                ):  # Implies neighbors_are_consistent was false
                    reason = f"Low relative confidence, neighbors conflicted (Diff: {best_score - second_best_score:.2f})"
                    if is_debug:
                        debug_details = {
                            "reason_code": "LOW_REL_CONF_NEIGHBORS_CONFLICTED",  # Should be covered above
                            "best_score": best_score,
                            "second_best_score": second_best_score,
                            "rejected_best_category": best_category,
                        }
                else:  # Should not happen
                    reason = "Defaulted to Unknown (Unexpected state)"
                    if is_debug:
                        debug_details = {
                            "reason_code": "UNKNOWN_DEFAULT",
                            "rejected_best_category": best_category,
                        }

                logger.info(
                    f"Narrative '{cleaned_desc}' (Original: '{original_desc}') classified as Unknown: {reason}"  # Log both
                )

            results.append(
                {
                    "narrative": original_desc,  # Return original
                    "cleaned_narrative": cleaned_desc,  # Add cleaned for debugging
                    "predicted_category": final_category,
                    "similarity_score": best_score,
                    "second_predicted_category": second_best_category,
                    "second_similarity_score": second_best_score,
                    "money_in": money_in,
                    "amount": amount,
                    "adjustment_info": (
                        {"unknown_reason": reason}
                        if final_category == "Unknown"
                        else None
                    ),
                    "debug_info": debug_details,  # Add debug info here
                }
            )

    except ValueError as ve:
        logger.error(f"ValueError during initial categorization: {ve}")
        # Return error for all transactions if a fundamental error like shape mismatch occurs
        return [
            {
                "narrative": tx["original_description"],  # Return original
                "cleaned_narrative": tx["cleaned_description"],  # Add cleaned
                "predicted_category": f"Error: {str(ve)}",
                "similarity_score": 0.0,
                "money_in": tx.get("money_in"),
                "amount": tx.get("amount"),
                "debug_info": None,  # Ensure field exists
            }
            for tx in transactions_input
        ]
    except Exception as e:
        logger.error(
            f"Unexpected error during initial categorization: {e}", exc_info=True
        )
        # Generic error for unexpected issues
        return [
            {
                "narrative": tx["original_description"],  # Return original
                "cleaned_narrative": tx["cleaned_description"],  # Add cleaned
                "predicted_category": "Error: Categorization Failed",
                "similarity_score": 0.0,
                "money_in": tx.get("money_in"),
                "amount": tx.get("amount"),
                "debug_info": None,  # Ensure field exists
            }
            for tx in transactions_input
        ]

    return results


def _detect_refunds(
    initial_results: List[Dict[str, Any]], input_embeddings: np.ndarray, user_id: str
) -> List[Dict[str, Any]]:
    """Identifies potential refunds among credit transactions.
    Currently bypassed as logic was removed.
    """
    logger.info("Refund detection logic is currently bypassed.")
    # Placeholder: Simply return results without modification
    for res in initial_results:
        if (
            "adjustment_info" not in res or res["adjustment_info"] is None
        ):  # Initialize if needed
            res["adjustment_info"] = {}
        res["adjustment_info"]["is_refund_candidate"] = False  # Ensure flag is set

    return initial_results


def _detect_transfers(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detects potential transfers by matching income/expense descriptions and amounts."""
    try:
        # Note: results contains 'narrative' (original) and 'cleaned_narrative'
        # We should match transfers based on the CLEANED narrative for robustness
        description_map = {}

        # Clean descriptions slightly differently for transfer matching: lowercase, normalize spaces
        def normalize_transfer_desc(desc):
            # Use the already cleaned description if available, otherwise clean the original
            # Assuming the input `desc` here is the *original* narrative from the results list
            # If we use cleaned_narrative directly, no need for this function?
            # Let's assume we still need some normalization *specific* to transfer matching
            norm = re.sub(r"\s+", " ", str(desc).lower().strip())
            # Optionally remove frequent but irrelevant details like dates/times if they interfere
            # norm = re.sub(r"\b\d{1,2}[-/.]\d{1,2}(?:[-/.]\d{2,4})?\b", "", norm) # Simple date removal
            # norm = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]m)?\b", "", norm) # Simple time removal
            # Remove long numbers (potential IDs) but keep amounts
            norm = re.sub(r"\b\d{6,}\b", "", norm)
            norm = re.sub(r"\s+", " ", norm).strip()
            return norm

        for i, res in enumerate(results):
            # Use the cleaned narrative for matching
            cleaned_desc = res.get("cleaned_narrative", "")  # Use cleaned narrative
            if not cleaned_desc:
                # Fallback to cleaning the original narrative if cleaned one is missing
                cleaned_desc = normalize_transfer_desc(res.get("narrative", ""))

            amount = res.get("amount")
            money_in = res.get("money_in")  # True, False, or None

            if not cleaned_desc or amount is None or money_in is None:
                continue  # Skip entries without description, amount, or direction

            if cleaned_desc not in description_map:
                description_map[cleaned_desc] = {"income": [], "expense": []}

            # Store index, amount, and original category
            entry = (i, amount, res["predicted_category"])

            if money_in is False:
                description_map[cleaned_desc]["expense"].append(entry)
            elif money_in is True:
                # Add income candidate unless it was already marked as a refund (if refund logic existed)
                # or already classified as Transfer in
                is_refund = res.get("adjustment_info", {}).get(
                    "is_refund_candidate", False
                )
                is_already_transfer = res["predicted_category"] == "Transfer in"
                if not is_refund and not is_already_transfer:
                    description_map[cleaned_desc]["income"].append(entry)

        matched_indices = set()
        for cleaned_desc_key, groups in description_map.items():
            # Try to find exact amount matches first
            expense_indices_processed = set()
            income_indices_processed = set()

            for inc_idx, inc_amount, inc_cat in groups["income"]:
                if inc_idx in matched_indices or inc_idx in income_indices_processed:
                    continue

                found_match = False
                for exp_idx, exp_amount, exp_cat in groups["expense"]:
                    if (
                        exp_idx in matched_indices
                        or exp_idx in expense_indices_processed
                    ):
                        continue

                    # Check for exact amount match (absolute values)
                    if abs(abs(inc_amount) - abs(exp_amount)) < 0.01:
                        # Found a pair!
                        results[inc_idx]["predicted_category"] = "Transfer in"
                        if (
                            "adjustment_info" not in results[inc_idx]
                            or results[inc_idx]["adjustment_info"] is None
                        ):
                            results[inc_idx]["adjustment_info"] = {}
                        results[inc_idx]["adjustment_info"][
                            "transfer_detection_reason"
                        ] = f"Paired with expense index {exp_idx} (Amount Match)"
                        results[inc_idx]["adjustment_info"]["adjusted"] = True
                        results[inc_idx]["adjustment_info"][
                            "original_category"
                        ] = inc_cat  # Store original

                        results[exp_idx]["predicted_category"] = "Transfer out"
                        if (
                            "adjustment_info" not in results[exp_idx]
                            or results[exp_idx]["adjustment_info"] is None
                        ):
                            results[exp_idx]["adjustment_info"] = {}
                        results[exp_idx]["adjustment_info"][
                            "transfer_detection_reason"
                        ] = f"Paired with income index {inc_idx} (Amount Match)"
                        results[exp_idx]["adjustment_info"]["adjusted"] = True
                        results[exp_idx]["adjustment_info"][
                            "original_category"
                        ] = exp_cat  # Store original

                        # Optionally add the matched cleaned description to adjustment_info
                        results[inc_idx]["adjustment_info"][
                            "matched_transfer_key"
                        ] = cleaned_desc_key
                        results[exp_idx]["adjustment_info"][
                            "matched_transfer_key"
                        ] = cleaned_desc_key

                        matched_indices.add(inc_idx)
                        matched_indices.add(exp_idx)
                        income_indices_processed.add(inc_idx)
                        expense_indices_processed.add(exp_idx)
                        logger.info(
                            f"Detected transfer pair (Exact Amount): '{results[inc_idx]['narrative']}' ({inc_amount}) <-> '{results[exp_idx]['narrative']}' ({exp_amount})"
                        )
                        found_match = True
                        break  # Move to next income item once matched
                # End inner expense loop
            # End outer income loop

    except Exception as e:
        logger.error(f"Error during transfer detection: {e}", exc_info=True)
        # Don't halt processing, just log the error

    return results
