"""Service layer for handling classification requests and categorization pipeline."""

import logging
import re
import os  # Import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import time

from utils.embedding_utils import fetch_embeddings

# from utils.prisma_client import prisma_client
# from utils.replicate_utils import run_prediction
from utils.request_utils import create_error_response
from config import (
    EMBEDDING_DIMENSION,
    MIN_ABSOLUTE_CONFIDENCE,
    MIN_RELATIVE_CONFIDENCE_DIFF,
    NEIGHBOR_COUNT,
)  # Import constants
from utils.text_utils import clean_text  # Import clean_text
from utils.local_embedding_utils import generate_embeddings  # Import local utils

logger = logging.getLogger(__name__)


def process_classification_request(validated_data, user_id):
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

        # 2. Generate Embeddings Locally (Synchronous)
        try:
            embeddings = generate_embeddings(cleaned_descriptions)

            if embeddings is None:
                logger.error(
                    f"Failed to generate embeddings locally during classification for user {user_id}"
                )
                return create_error_response(
                    "Failed to generate classification embeddings", 500
                )

            # Validate dimensions
            if embeddings.shape[0] != len(transactions_input_for_context):
                logger.error(
                    f"Embedding count mismatch after local generation: {embeddings.shape[0]} vs {len(transactions_input_for_context)}"
                )
                return create_error_response(
                    "Embedding count mismatch during classification", 500
                )

            # Assuming EMBEDDING_DIMENSION is still relevant or can be inferred
            if embeddings.shape[1] != EMBEDDING_DIMENSION:
                logger.warning(
                    f"Local embedding dimension ({embeddings.shape[1]}) differs from config ({EMBEDDING_DIMENSION}). Using local dimension."
                )
                # Proceeding, but ensure comparison logic handles potential dimension differences

            # --- Run Categorization Pipeline (Now directly after local embedding) ---
            initial_results = _apply_initial_categorization(
                transactions_input_for_context, embeddings, user_id
            )

            # Check for errors from categorization itself
            if any(
                "Error:" in res.get("predicted_category", "") for res in initial_results
            ):
                logger.error(
                    f"Categorization pipeline failed locally for user {user_id}. Check previous logs."
                )
                first_error_message = next(
                    (
                        res["predicted_category"]
                        for res in initial_results
                        if "Error:" in res.get("predicted_category", "")
                    ),
                    "Categorization failed",
                )

                logger.info(f"Extracted first_error_message: '{first_error_message}'")

                # Check for specific "Embedding dimension mismatch" error
                if "Embedding dimension mismatch" in first_error_message:
                    logger.warning(
                        f"Embedding dimension mismatch for user {user_id}. Instructing to re-train."
                    )
                    return {
                        "status": "error",
                        "error_message": "Model retraining required: The embedding model has been updated. Please re-train your model using the spreadsheet addon before classifying new transactions.",
                        "error_code": 409 # 409 Conflict is appropriate here
                    }

                # For other errors from categorization pipeline
                return {
                    "status": "error",
                    "error_message": f"Categorization failed: {first_error_message}",
                    "error_code": 500
                }

            results_after_refunds = _detect_refunds(
                initial_results, embeddings, user_id
            )
            final_results = _detect_transfers(results_after_refunds)
            # --- End Pipeline ---

            # Return synchronous success response
            return {
                "status": "completed",
                "message": "Classification completed successfully",
                "results": final_results,
            }

        except Exception as process_error:
            logger.error(
                f"Error processing classification results locally for user {user_id}: {process_error}",
                exc_info=True,
            )
            # Return an error to the user
            return {
                "status": "error",
                "error_message": f"Failed to process classification results: {str(process_error)}",
                "error_code": 500
            }

    except Exception as e:
        logger.error(
            f"Critical error processing classification request: {e}", exc_info=True
        )
        return {
            "status": "error",
            "error_message": "Internal server error during classification processing",
            "error_code": 500
        }


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
    # Define confidence thresholds - MOVED TO config.py
    # MIN_ABSOLUTE_CONFIDENCE = 0.70  # Minimum similarity score to consider a match valid
    # MIN_RELATIVE_CONFIDENCE_DIFF = (
    #     0.03  # Minimum difference between best and second best score
    # )
    # NEIGHBOR_COUNT = 3  # Number of neighbors to consider for consistency check

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
            final_category = best_category  # Default to the best match found
            debug_details: Optional[Dict[str, Any]] = None  # Initialize debug info
            is_low_confidence = False  # NEW flag
            human_readable_reason = ""  # NEW variable for user-facing reason

            # --- Refined Decision Logic (Calculate Reason Only) ---
            meets_absolute_confidence = best_score >= MIN_ABSOLUTE_CONFIDENCE
            meets_relative_confidence = (
                best_score - second_best_score
            ) >= MIN_RELATIVE_CONFIDENCE_DIFF
            neighbors_are_consistent = not has_conflicting_neighbors

            # Original logic preserved here to determine the *reason* for potential low confidence
            # but we no longer change final_category based on this.
            if not meets_absolute_confidence:
                # adjustment_needed = True # No longer needed
                is_low_confidence = True
                human_readable_reason = (
                    "Match score was below the confidence threshold."
                )
                # reason = f"Low absolute confidence ({best_score:.2f} < {MIN_ABSOLUTE_CONFIDENCE})" # Original reason for debug
                if is_debug:
                    # ... (debug_details remains the same)
                    debug_details = {
                        "reason_code": "LOW_ABS_CONF",
                        "best_score": best_score,
                        "threshold": MIN_ABSOLUTE_CONFIDENCE,
                        "evaluated_category": best_category,  # Note: this category *was* returned
                        "neighbor_categories": neighbor_categories[:NEIGHBOR_COUNT],
                        "neighbor_cleaned_descs": cleaned_descs[:NEIGHBOR_COUNT],
                        "neighbor_original_descs": original_descs[:NEIGHBOR_COUNT],
                        "neighbor_scores": neighbor_scores[:NEIGHBOR_COUNT],
                    }
            # --- Check relative confidence ONLY if best != second best ---
            elif best_category != second_best_category:
                if not meets_relative_confidence:  # Low relative difference
                    is_low_confidence = True  # Flag it
                    if not neighbors_are_consistent:
                        # adjustment_needed = True # No longer needed
                        human_readable_reason = "Top two matches were very similar and had conflicting neighbor categories."
                        # reason = f"Low relative confidence (Diff: {best_score - second_best_score:.2f} < {MIN_RELATIVE_CONFIDENCE_DIFF}) AND Conflicting neighbors: {list(unique_neighbor_categories)}" # Original reason
                        if is_debug:
                            # ... (debug_details remains the same)
                            debug_details = {
                                "reason_code": "LOW_REL_CONF_AND_CONFLICTING_NEIGHBORS",
                                "best_score": best_score,
                                "second_best_score": second_best_score,
                                "difference": best_score - second_best_score,
                                "rel_threshold": MIN_RELATIVE_CONFIDENCE_DIFF,
                                "neighbor_categories": neighbor_categories[
                                    :NEIGHBOR_COUNT
                                ],
                                "unique_neighbor_categories": list(
                                    unique_neighbor_categories
                                ),
                                "neighbor_cleaned_descs": cleaned_descs[
                                    :NEIGHBOR_COUNT
                                ],
                                "neighbor_original_descs": original_descs[
                                    :NEIGHBOR_COUNT
                                ],
                                "neighbor_scores": neighbor_scores[:NEIGHBOR_COUNT],
                                "evaluated_category": best_category,
                                "second_best_category": second_best_category,
                            }
                    else:  # Neighbors are consistent
                        # adjustment_needed = True # No longer needed
                        human_readable_reason = "Top two matches were very similar, but neighbors agreed with the top match."
                        # reason = f"Neighbor consistency override: Low relative confidence (Diff: {best_score - second_best_score:.2f} < {MIN_RELATIVE_CONFIDENCE_DIFF}) but neighbors consistent." # Original reason
                        if is_debug:
                            # ... (debug_details remains the same)
                            debug_details = {
                                "reason_code": "ACCEPTED_BY_NEIGHBORS_LOW_REL_CONF",
                                "best_score": best_score,
                                "second_best_score": second_best_score,
                                "difference": best_score - second_best_score,
                                "threshold": MIN_RELATIVE_CONFIDENCE_DIFF,
                                "evaluated_category": best_category,
                                "neighbor_categories": neighbor_categories[
                                    :NEIGHBOR_COUNT
                                ],
                                "neighbor_cleaned_descs": cleaned_descs[
                                    :NEIGHBOR_COUNT
                                ],
                                "neighbor_original_descs": original_descs[
                                    :NEIGHBOR_COUNT
                                ],
                                "neighbor_scores": neighbor_scores[:NEIGHBOR_COUNT],
                            }
            # No need for the complex "Unknown" logging section anymore, as we always return best_category

            # --- Append Result ---
            adjustment_info_final = None
            if is_low_confidence:
                adjustment_info_final = {
                    "is_low_confidence": True,
                    "reason": human_readable_reason,  # Use the new human-readable reason
                }
            # Note: If refund logic is added later, merge `is_refund_candidate` here.
            # Example merge: if adjustment_info_final and refund_info: adjustment_info_final.update(refund_info)
            # Currently refund is bypassed, so we just use the confidence info if present.

            results.append(
                {
                    "narrative": original_desc,  # Return original
                    "cleaned_narrative": cleaned_desc,  # Add cleaned for debugging
                    "predicted_category": final_category,  # Always best_category now
                    "similarity_score": best_score,
                    "second_predicted_category": second_best_category,
                    "second_similarity_score": second_best_score,
                    "money_in": money_in,
                    "amount": amount,
                    "adjustment_info": adjustment_info_final,  # Use the potentially populated dict
                    "debug_info": debug_details,  # Add debug info here if generated
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
