"""Service layer for handling training requests."""

import logging
import numpy as np
import pandas as pd

from utils.text_utils import clean_text
from utils.embedding_utils import store_embeddings
from utils.local_embedding_utils import generate_embeddings
from config import EMBEDDING_DIMENSION

try:
    from utils.memory_utils import MemoryMonitor, log_memory_usage, force_garbage_collection
    from utils.performance_config import get_performance_config, update_cleaning_config_for_performance
    PERFORMANCE_UTILS_AVAILABLE = True
except ImportError:
    PERFORMANCE_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)


def process_training_request(validated_data, user_id):
    """Processes the validated training request data."""
    # Use memory monitoring if available
    if PERFORMANCE_UTILS_AVAILABLE:
        with MemoryMonitor(f"Training request for user {user_id}"):
            return _process_training_request_internal(validated_data, user_id)
    else:
        return _process_training_request_internal(validated_data, user_id)


def _process_training_request_internal(validated_data, user_id):
    """Internal training request processing with memory optimizations."""
    # temporary artificial delay to test async training 10 seconds
    try:        
        transactions = validated_data.transactions
        # Use user_id as the main identifier
        # sheet_id = validated_data.expenseSheetId or f"user_{user_id}" # sheet_id not used currently
        logger.info(
            f"Processing training request - User: {user_id}, Items: {len(transactions)}"
        )
        
        if PERFORMANCE_UTILS_AVAILABLE:
            log_memory_usage("Training start")
        
        # Convert transactions to DataFrame
        transactions_data = [t.model_dump() for t in transactions]
        df = pd.DataFrame(transactions_data)

        # --- Clean Descriptions (Corrected Method) ---
        if "description" not in df.columns or df["description"].isnull().all():
            # Handle cases with missing or all-null description column
            logger.error(
                f"Training data for user {user_id} is missing 'description' column or all values are null."
            )
            return {
                "status": "error",
                "error_message": "Training data must contain valid descriptions.",
                "error_code": 400
            }

        original_descriptions = df["description"].astype(str).tolist()  # Ensure strings
        try:
            # Use simple text cleaning without grouping for training (1:1 mapping required)
            from utils.text_utils import CleaningConfig
            config = CleaningConfig()
            config.use_embedding_grouping = False
            config.use_fuzzy_matching = False
            config.use_fuzzy_matching_post_clean = False
            
            # Apply performance optimizations
            if PERFORMANCE_UTILS_AVAILABLE:
                update_cleaning_config_for_performance(config, len(original_descriptions))
                log_memory_usage("Before text cleaning")
            
            cleaned_descriptions = clean_text(original_descriptions, config)
            
            if PERFORMANCE_UTILS_AVAILABLE:
                log_memory_usage("After text cleaning")
            if len(cleaned_descriptions) != len(original_descriptions):
                logger.error(
                    f"Training Clean Error: Length mismatch for user {user_id}. Input {len(original_descriptions)}, Output {len(cleaned_descriptions)}"
                )
                return {
                    "status": "error",
                    "error_message": "Cleaning produced incorrect number of descriptions.",
                    "error_code": 500
                }
            df["cleaned_description"] = cleaned_descriptions
        except Exception as clean_error:
            logger.error(
                f"Error during description cleaning in training for user {user_id}: {clean_error}",
                exc_info=True,
            )
            return {
                "status": "error",
                "error_message": f"Failed during text cleaning: {str(clean_error)}",
                "error_code": 500
            }

        # Drop duplicates based on the *cleaned* description
        df = df.drop_duplicates(subset=["cleaned_description"])
        # --- End Cleaning ---

        # Store training data index with proper dtype
        df["item_id"] = range(len(df))

        # Log the categories we're training with
        # Ensure 'Category' column exists
        if "Category" not in df.columns:
            logger.error(
                f"Training data for user {user_id} is missing 'Category' column."
            )
            return {
                "status": "error",
                "error_message": "Training data must contain a 'Category' column.",
                "error_code": 400
            }

        unique_categories = df["Category"].dropna().unique().tolist()
        logger.info(
            f"Training with {len(df)} unique descriptions across {len(unique_categories)} categories for user {user_id}"
        )

        # Create structured array for index data
        index_data = np.array(
            [
                (i, desc, cat)
                for i, (desc, cat) in enumerate(
                    zip(df["cleaned_description"].values, df["Category"].values)
                )
            ],
            dtype=[
                ("item_id", np.int32),
                ("description", "U256"),  # Storing the CLEANED description
                ("category", "U128"),  # Match length used in categorization
            ],
        )

        # Store index data
        index_id = f"{user_id}_index"
        if not store_embeddings(index_data, index_id, user_id):
            logger.error(f"Failed to store training index {index_id}")
            return {
                "status": "error",
                "error_message": "Failed to store training index data",
                "error_code": 500
            }
        logger.info(
            f"Stored training index data {index_id} with {len(index_data)} entries"
        )

        # Create a placeholder for the embeddings (to ensure the record exists)
        embedding_id = f"{user_id}"
        placeholder = np.array([[0.0] * EMBEDDING_DIMENSION], dtype=np.float32)
        store_embeddings(placeholder, embedding_id, user_id)
        logger.info(f"Stored/updated placeholder embedding record {embedding_id}")

        # Get descriptions for embedding
        descriptions_to_embed = df["cleaned_description"].tolist()

        try:
            # === Generate Embeddings Locally with Memory Optimization ===
            if PERFORMANCE_UTILS_AVAILABLE:
                log_memory_usage("Before embedding generation")
                perf_config = get_performance_config()
                embedding_batch_size = perf_config.get_embedding_generation_batch_size()
            else:
                embedding_batch_size = 10  # Conservative default
            
            embeddings = generate_embeddings(descriptions_to_embed, batch_size=embedding_batch_size)
            
            if PERFORMANCE_UTILS_AVAILABLE:
                log_memory_usage("After embedding generation")

            if embeddings is None:
                logger.error(
                    f"Failed to generate embeddings locally for user {user_id}"
                )
                return {
                    "status": "error",
                    "error_message": "Failed to generate training embeddings",
                    "error_code": 500
                }

            # Validate embedding dimension (assuming generate_embeddings returns non-empty on success)
            if embeddings.shape[0] != len(descriptions_to_embed):
                logger.error(
                    f"Embedding count mismatch after local generation: {embeddings.shape[0]} vs {len(descriptions_to_embed)}"
                )
                return {
                    "status": "error",
                    "error_message": "Embedding count mismatch during training",
                    "error_code": 500
                }

            # Assuming EMBEDDING_DIMENSION is still relevant or can be inferred
            # If the local model dimension differs, update EMBEDDING_DIMENSION in config.py or handle dynamically
            if embeddings.shape[1] != EMBEDDING_DIMENSION:
                logger.warning(
                    f"Local embedding dimension ({embeddings.shape[1]}) differs from config ({EMBEDDING_DIMENSION}). Using local dimension."
                )
                # Potentially update config or handle dimension mismatch? For now, proceed.

            # Store the locally generated embeddings
            store_result = store_embeddings(embeddings, embedding_id, user_id)
            if not store_result:
                logger.error(
                    f"Failed to store locally generated training embeddings {embedding_id}"
                )
                return {
                    "status": "error",
                    "error_message": "Failed to store training embeddings",
                    "error_code": 500
                }

            logger.info(
                f"Stored locally generated training embeddings {embedding_id} successfully."
            )

            # Since generation is synchronous, training is complete here.
            # We can optionally log the completion status to the database if needed
            # (e.g., using a generic ID or timestamp if no prediction_id exists)
            # For simplicity, we'll just return success directly.

            return {
                "status": "completed",
                "message": "Training completed successfully",
                "unique_description_count": len(df),
                "category_count": len(unique_categories),
            }

        except Exception as local_gen_error:
            logger.error(
                f"Error during local embedding generation/storage for user {user_id}: {local_gen_error}",
                exc_info=True,
            )
            return {
                "status": "error",
                "error_message": f"Error during training process: {str(local_gen_error)}",
                "error_code": 500
            }

    except Exception as e:
        logger.error(f"Error processing training request: {e}", exc_info=True)
        # Generic internal server error for unexpected issues in the service layer
        return {
            "status": "error",
            "error_message": "Internal server error during training processing",
            "error_code": 500
        }
