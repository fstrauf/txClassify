"""Utilities for interacting with the OpenAI API."""

import logging
import os
import json
import time # Added for potential delays if needed
from typing import List, Dict, Any, Optional

# Import the openai library - ensure it's in requirements.txt and installed
try:
    import openai
except ImportError:
    raise ImportError("OpenAI library not found. Please install it: pip install openai")

logger = logging.getLogger(__name__)

# Initialize OpenAI client using API key from environment variable
# It's good practice to handle the case where the key might be missing
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.warning("OPENAI_API_KEY environment variable not set. OpenAI features will be disabled.")
    client = None
else:
    try:
        client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        client = None

# Define constants for the OpenAI model and potentially default prompts
DEFAULT_OPENAI_MODEL = "gpt-4.1-2025-04-14" # Use a model that supports JSON mode
OPENAI_BATCH_SIZE = 15 # Process transactions in batches
OPENAI_TIMEOUT = 45 # Increased Timeout in seconds for API calls
OPENAI_MAX_RETRIES = 2 # Number of retries on timeout or specific errors

# System prompt instructing the model
SYSTEM_PROMPT = """
You are an expert assistant helping to categorize bank transactions. You will receive a list of transactions and a list of valid category names.
Your task is to analyze each transaction and choose the single most appropriate category name ONLY from the provided list of valid categories.
You MUST return your response as a single valid JSON object with a single key "batch_results". The value of "batch_results" MUST be a JSON array, where each object in the array corresponds to one input transaction and has the following format:
{
  "description": "The original transaction description exactly as provided",
  "suggested_category": "The chosen category name from the list OR the literal string 'None' if no valid category applies"
}
Do NOT include any explanations, apologies, or text outside this JSON object structure. Ensure the 'description' field in your output exactly matches the input description for mapping purposes.
"""

def categorize_with_openai(
    transactions_to_categorize: List[Dict[str, Any]],
    user_category_names: List[str],
    model: str = DEFAULT_OPENAI_MODEL
) -> List[Dict[str, Optional[str]]]: # Return list of dicts
    """
    Attempts to categorize low-confidence transactions using OpenAI's API in batches.

    Args:
        transactions_to_categorize: List of transaction dicts ({'description': ..., 'amount': ..., 'money_in': ...}).
        user_category_names: List of valid category names.
        model: The OpenAI model to use (must support JSON mode).

    Returns:
        List of dicts, same length as input, mapping description to suggested category:
        e.g., [{'description': '...', 'suggested_category': 'CategoryName' or None}]
    """
    if not client:
        logger.warning("OpenAI client not initialized. Skipping OpenAI categorization.")
        # Return the expected structure with None for categories
        return [{"description": tx.get('description'), "suggested_category": None} for tx in transactions_to_categorize]

    if not user_category_names:
        logger.warning("No user category names provided for OpenAI categorization. Skipping.")
        return [{"description": tx.get('description'), "suggested_category": None} for tx in transactions_to_categorize]

    # Create a set for faster validation
    valid_categories_set = set(user_category_names)
    all_results_map: Dict[str, Optional[str]] = {} # Map description -> category

    logger.info(f"Starting OpenAI categorization for {len(transactions_to_categorize)} transactions in batches of {OPENAI_BATCH_SIZE} using model {model}.")

    # --- Process in Batches --- #
    for i in range(0, len(transactions_to_categorize), OPENAI_BATCH_SIZE):
        batch = transactions_to_categorize[i:i + OPENAI_BATCH_SIZE]
        batch_descriptions = [tx.get('description', '[Missing Description]') for tx in batch]
        logger.info(f"Processing batch {i // OPENAI_BATCH_SIZE + 1}: {len(batch)} transactions.")

        prompt_batch_data = [
            {
                "description": tx.get('description'),
                "amount": tx.get('amount'),
                "money_in": tx.get('money_in')
            } for tx in batch
        ]

        user_prompt = f"""Please categorize the following transactions based ONLY on the valid categories provided below. Return your response as a JSON object with a "batch_results" key, as instructed.

Transactions:
```json
{json.dumps(prompt_batch_data, indent=2)}
```

Valid Categories:
"""
        user_prompt += "\n".join([f"- {name}" for name in user_category_names])
        user_prompt += "\n\nJSON Output (ensure root is an object with 'batch_results' key):\n"

        retries = 0
        while retries <= OPENAI_MAX_RETRIES:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    timeout=OPENAI_TIMEOUT,
                )
                response_content = response.choices[0].message.content

                if not response_content:
                    logger.error(f"OpenAI returned empty content for batch: {batch_descriptions[0]}")
                    # Mark all in batch as None, break retry loop for this batch
                    for desc in batch_descriptions:
                        if desc not in all_results_map: all_results_map[desc] = None
                    break # Break retry loop

                # --- Parse and Validate JSON Response --- #
                try:
                    ai_output = json.loads(response_content)
                    results_list = None
                    if isinstance(ai_output, dict) and "batch_results" in ai_output and isinstance(ai_output.get("batch_results"), list):
                        results_list = ai_output["batch_results"]
                    else:
                        logger.error(f"OpenAI JSON response was not in the expected format {{'batch_results': [...]}}. Batch: {batch_descriptions[0]}. Response: {response_content[:500]}")

                    if results_list is None:
                        for desc in batch_descriptions:
                            if desc not in all_results_map: all_results_map[desc] = None
                        break # Break retry loop, parsing failed fundamentally

                    # --- Process Validated Batch Results --- #
                    batch_results_map = {}
                    for item in results_list:
                        if not isinstance(item, dict) or "description" not in item or "suggested_category" not in item:
                            logger.warning(f"Skipping invalid item format in OpenAI response: {item}")
                            continue # Skip this item

                        desc = item["description"]
                        cat = item["suggested_category"]
                        validated_cat = None

                        if cat == "None":
                            validated_cat = None
                        elif isinstance(cat, str) and cat in valid_categories_set:
                            validated_cat = cat
                        else:
                            logger.warning(f"OpenAI suggested invalid category '{cat}' for description '{desc}'. Discarding.")
                            validated_cat = None

                        if desc not in batch_results_map:
                             batch_results_map[desc] = validated_cat
                        else:
                            logger.warning(f"Duplicate description '{desc}' found in OpenAI batch response. Using first valid result.")

                    # Map validated batch results to the main results map
                    for desc in batch_descriptions:
                         # Ensure every input description gets an entry, even if AI missed it
                         if desc not in all_results_map:
                             all_results_map[desc] = batch_results_map.get(desc, None)

                    # Successfully processed batch, break retry loop
                    break

                except json.JSONDecodeError as json_err:
                    logger.error(f"Failed to decode JSON from OpenAI for batch: {batch_descriptions[0]}. Error: {json_err}. Response: {response_content[:500]}")
                    # Mark all as None for this batch, break retry loop
                    for desc in batch_descriptions:
                         if desc not in all_results_map: all_results_map[desc] = None
                    break # Break retry loop

            # --- Handle API Errors with Retry Logic --- #
            except openai.APITimeoutError:
                 logger.warning(f"OpenAI API call timed out for batch: {batch_descriptions[0]}. Retry {retries + 1}/{OPENAI_MAX_RETRIES}")
                 retries += 1
                 if retries > OPENAI_MAX_RETRIES:
                     logger.error(f"OpenAI API call failed after {OPENAI_MAX_RETRIES} retries (Timeout).")
                     for desc in batch_descriptions:
                         if desc not in all_results_map: all_results_map[desc] = None
                 else:
                    time.sleep(1) # Simple backoff
            except openai.APIStatusError as status_err: # Catch specific status errors like 429 Rate Limit
                logger.warning(f"OpenAI API status error for batch: {batch_descriptions[0]}. Status: {status_err.status_code}. Retry {retries + 1}/{OPENAI_MAX_RETRIES}")
                retries += 1
                if status_err.status_code == 429 or retries > OPENAI_MAX_RETRIES: # Stop retrying on rate limit or max retries
                    logger.error(f"OpenAI API call failed (Status: {status_err.status_code}) after {retries -1} retries.")
                    for desc in batch_descriptions:
                        if desc not in all_results_map: all_results_map[desc] = None
                    break # Break retry loop
                else:
                     time.sleep(retries + 1) # Exponential backoff might be better
            except openai.APIError as api_err: # More general API errors
                logger.error(f"OpenAI API error for batch: {batch_descriptions[0]}. Error: {api_err}. No retry.")
                for desc in batch_descriptions:
                     if desc not in all_results_map: all_results_map[desc] = None
                break # Break retry loop, non-retryable API error
            except Exception as e:
                logger.error(f"Unexpected error during OpenAI call for batch: {batch_descriptions[0]}. Error: {e}", exc_info=True)
                for desc in batch_descriptions:
                     if desc not in all_results_map: all_results_map[desc] = None
                break # Break retry loop, unexpected error
        # End of retry while loop
    # --- End of Batch Loop --- #

    # --- Consolidate final results in the original order --- #
    final_ordered_results: List[Dict[str, Optional[str]]] = []
    for tx in transactions_to_categorize:
        desc = tx.get('description')
        suggested_category = all_results_map.get(desc) if desc else None
        final_ordered_results.append({
            "description": desc,
            "suggested_category": suggested_category
        })

    if len(final_ordered_results) != len(transactions_to_categorize):
         logger.error(f"Final OpenAI results length mismatch: expected {len(transactions_to_categorize)}, got {len(final_ordered_results)}. This indicates a bug.")
         # Fallback if something went wrong during consolidation
         return [{"description": tx.get('description'), "suggested_category": None} for tx in transactions_to_categorize]

    logger.info(f"Completed OpenAI categorization attempt for {len(transactions_to_categorize)} transactions.")
    return final_ordered_results

# Example usage (for testing purposes, would be called from classification_service)
# if __name__ == '__main__':
#     # Ensure OPENAI_API_KEY is set in your environment for this example
#     logging.basicConfig(level=logging.INFO)
#     test_transactions = [
#         {'description': 'Coffee Shop Purchase', 'amount': -5.50, 'money_in': False},
#         {'description': 'Monthly Salary Deposit', 'amount': 3000, 'money_in': True},
#         {'description': 'Netflix Subscription', 'amount': -15.99, 'money_in': False},
#     ]
#     test_categories = ["Groceries", "Dining Out", "Income", "Subscriptions", "Utilities", "Shopping"]
#
#     categorized_results = categorize_with_openai(test_transactions, test_categories)
#     print("OpenAI Categorization Results (Placeholder):")
#     for tx, result in zip(test_transactions, categorized_results):
#         print(f"  - Desc: '{tx['description']}', Suggested Category: {result}") 