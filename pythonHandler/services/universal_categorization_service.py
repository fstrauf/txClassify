"""Service layer for universal categorization that bypasses training data."""

import logging
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from utils.text_utils import clean_and_group_transactions, CleaningConfig
from utils.local_embedding_utils import generate_embeddings
from utils.openai_utils import categorize_with_openai, client as openai_client, DEFAULT_OPENAI_MODEL
from utils.request_utils import create_error_response

logger = logging.getLogger(__name__)

def load_predefined_categories() -> List[str]:
    """Load predefined categories from the categories.json file."""
    try:
        categories_file = Path(__file__).parent.parent / "data" / "categories.json"
        with open(categories_file, 'r') as f:
            categories = json.load(f)
        
        if not isinstance(categories, list):
            logger.error(f"Expected categories.json to contain a list, got {type(categories)}")
            return []
        
        logger.info(f"Loaded {len(categories)} predefined categories")
        return categories
    except Exception as e:
        logger.error(f"Failed to load predefined categories: {e}", exc_info=True)
        return []

def process_universal_categorization_request(validated_data, user_id: str):
    """Process universal categorization request without using training data."""
    try:
        start_time = time.time()
        
        # 1. Extract transaction descriptions
        original_descriptions = []
        transactions_input_for_context = []
        
        for tx_input_item in validated_data.transactions:
            if isinstance(tx_input_item, str):
                desc = tx_input_item
                money_in = None
                amount = None
            else:  # It's a TransactionInput object
                desc = tx_input_item.description
                money_in = tx_input_item.money_in
                amount = tx_input_item.amount if hasattr(tx_input_item, "amount") else None

            original_descriptions.append(desc)
            transactions_input_for_context.append({
                "original_description": desc,
                "money_in": money_in,
                "amount": amount,
            })

        if not original_descriptions:
            logger.error(f"No valid descriptions found in universal categorization request for user {user_id}")
            return create_error_response("No valid transaction descriptions provided", 400)

        # 2. Load predefined categories
        predefined_categories = load_predefined_categories()
        if not predefined_categories:
            logger.error("No predefined categories available for universal categorization")
            return create_error_response("System error: predefined categories not available", 500)

        # 3. Clean and group transactions using improved algorithm
        try:
            config = CleaningConfig()
            config.use_embedding_grouping = True
            config.embedding_clustering_method = "similarity"  # Use conservative similarity method
            config.embedding_similarity_threshold = 0.85  # Conservative threshold
            
            logger.info(f"Cleaning and grouping {len(original_descriptions)} transactions for user {user_id}")
            cleaned_descriptions, grouping_dict = clean_and_group_transactions(original_descriptions, config)
            
            if len(cleaned_descriptions) != len(original_descriptions):
                logger.error(f"Mismatch in length after cleaning: {len(original_descriptions)} vs {len(cleaned_descriptions)}")
                raise ValueError("Cleaning resulted in description count mismatch.")
                
            logger.info(f"Successfully cleaned and grouped {len(original_descriptions)} transactions into {len(set(grouping_dict.values()))} groups")
            
        except Exception as clean_error:
            logger.error(f"Error cleaning descriptions for universal categorization, user {user_id}: {clean_error}", exc_info=True)
            return create_error_response(f"Failed during text cleaning: {str(clean_error)}", 500)

        # 4. Prepare transactions with cleaned descriptions for context
        for i, original_desc in enumerate(original_descriptions):
            transactions_input_for_context[i]["cleaned_description"] = cleaned_descriptions[i]

        # 5. Create representative transactions for each group for LLM categorization
        group_representatives = {}
        group_members = {}
        
        # Build groups from the grouping dictionary
        for i, (original_desc, cleaned_desc) in enumerate(zip(original_descriptions, cleaned_descriptions)):
            representative = grouping_dict.get(cleaned_desc, cleaned_desc)
            
            if representative not in group_representatives:
                group_representatives[representative] = {
                    "description": original_desc,  # Use original description for LLM
                    "amount": transactions_input_for_context[i].get("amount"),
                    "money_in": transactions_input_for_context[i].get("money_in"),
                }
                group_members[representative] = []
            
            group_members[representative].append(i)  # Store transaction indices

        # 6. Send grouped transactions to LLM for categorization
        if not openai_client:
            logger.error("OpenAI client not available for universal categorization")
            return create_error_response("LLM categorization service not available", 503)

        try:
            # Prepare transactions for OpenAI
            transactions_for_llm = []
            for rep_desc, rep_data in group_representatives.items():
                transactions_for_llm.append({
                    "description": rep_data["description"],
                    "amount": rep_data["amount"],
                    "money_in": rep_data["money_in"],
                })

            logger.info(f"Sending {len(transactions_for_llm)} representative transactions to LLM for categorization")
            
            # Call OpenAI for categorization
            llm_results = categorize_with_openai(transactions_for_llm, predefined_categories, DEFAULT_OPENAI_MODEL)
            
            if not llm_results:
                logger.error("LLM categorization returned empty results")
                return create_error_response("LLM categorization failed", 500)

            # Create mapping from representative description to category
            rep_to_category = {}
            for result in llm_results:
                if result.get("description") and result.get("suggested_category"):
                    rep_to_category[result["description"]] = result["suggested_category"]

            logger.info(f"LLM categorized {len(rep_to_category)} groups successfully")
            
        except Exception as llm_error:
            logger.error(f"Error during LLM categorization for user {user_id}: {llm_error}", exc_info=True)
            return create_error_response(f"Failed during LLM categorization: {str(llm_error)}", 500)

        # 7. Apply categorization results to all transactions
        final_results = []
        
        for i, original_desc in enumerate(original_descriptions):
            cleaned_desc = cleaned_descriptions[i]
            representative = grouping_dict.get(cleaned_desc, cleaned_desc)
            
            # Find the representative's original description to get the category
            rep_original_desc = None
            for rep_desc, rep_data in group_representatives.items():
                if rep_desc == representative:
                    rep_original_desc = rep_data["description"]
                    break
            
            # Get the category from LLM results
            predicted_category = rep_to_category.get(rep_original_desc, "GENERAL_SERVICES")  # Default fallback
            
            result = {
                "narrative": original_desc,
                "cleaned_narrative": cleaned_desc,
                "predicted_category": predicted_category,
                "similarity_score": 1.0,  # Not applicable for universal categorization
                "second_predicted_category": None,
                "second_similarity_score": 0.0,
                "money_in": transactions_input_for_context[i].get("money_in"),
                "amount": transactions_input_for_context[i].get("amount"),
                "adjustment_info": {
                    "universal_categorization": True,
                    "group_representative": representative,
                    "llm_assisted": True,
                    "llm_model": DEFAULT_OPENAI_MODEL,
                    "bypassed_training_data": True,
                },
                "debug_info": {
                    "method": "universal_categorization",
                    "group_size": len(group_members.get(representative, [])),
                    "predefined_categories_count": len(predefined_categories),
                }
            }
            
            final_results.append(result)

        processing_time = time.time() - start_time
        logger.info(f"Universal categorization completed for user {user_id} in {processing_time:.2f}s")

        return {
            "status": "completed",
            "message": "Universal categorization completed successfully",
            "results": final_results,
            "processing_info": {
                "total_transactions": len(original_descriptions),
                "unique_groups": len(group_representatives),
                "processing_time_seconds": round(processing_time, 2),
                "categories_used": len(predefined_categories),
                "method": "universal_categorization"
            }
        }

    except Exception as e:
        logger.error(f"Critical error processing universal categorization request: {e}", exc_info=True)
        return {
            "status": "error",
            "error_message": "Internal server error during universal categorization processing",
            "error_code": 500
        }
