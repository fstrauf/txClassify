import csv
import os
import sys
from collections import defaultdict
import logging
import urllib.request  # Added for API calls
import json  # Added for API calls

# Attempt to load .env file
try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:
    print(
        "python-dotenv library not found. Please install it with 'pip install python-dotenv' to load .env files."
    )
    load_dotenv = None  # Ensure load_dotenv is None if import fails
    find_dotenv = None

# Configure basic logging to see output from text_utils and the script itself
# spaCy can be quite verbose on INFO, so setting to WARNING for cleaner default output from it.
# Script's own messages will use logger.info which will be visible.
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set our script's logger to INFO

# Correctly add paths for imports
# SCRIPT_DIR is where this script lives, e.g., txClassify/tests/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT is txClassify/
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Add PROJECT_ROOT to sys.path so 'pythonHandler' can be found as a package
sys.path.insert(0, PROJECT_ROOT)

# Removed PYTHONHANDLER_DIR sys.path modification as clean_text is now via API

# Load .env file from project root *before* accessing env variables for API config
if load_dotenv and find_dotenv:  # Check if import was successful
    # Try to find .env automatically, starting from script dir and going up
    # Or specify the project root directly if find_dotenv is problematic
    dotenv_path = find_dotenv(usecwd=True)  # Checks current working directory first
    if (
        not dotenv_path
    ):  # If not found in cwd, try relative to project root (common for tests)
        dotenv_path = os.path.join(PROJECT_ROOT, ".env")

    if os.path.exists(dotenv_path):
        loaded_ok = load_dotenv(dotenv_path, override=True)  # ADDED override=True
        if loaded_ok:
            logger.info(
                f"Loaded environment variables from: {dotenv_path} (with override)"
            )
        else:
            # This case might occur if .env is empty or load_dotenv itself returns False
            logger.warning(
                f"Attempted to load .env from {dotenv_path} but load_dotenv returned False/None."
            )
    else:
        logger.info(
            f".env file not found at {dotenv_path} (or by find_dotenv). Relying on manually set environment variables."
        )
else:
    logger.info(
        "dotenv library (load_dotenv/find_dotenv) not available. Relying on manually set environment variables."
    )

# API Configuration - Load from environment variables (now possibly populated by dotenv)
API_BASE_URL = os.getenv("TEST_TARGET_API_URL", "http://localhost:3005")
API_KEY = os.getenv("TEST_API_KEY")


def call_clean_text_api(descriptions_batch: list[str]) -> list[str]:
    """
    Calls the /clean_text API endpoint to clean a batch of descriptions.
    """
    if not API_KEY:
        logger.error(
            "TEST_API_KEY environment variable is not set (or not loaded from .env). Cannot call /clean_text API."
        )
        logger.warning("Returning original descriptions due to missing API_KEY.")
        return descriptions_batch

    if not descriptions_batch:
        return []

    payload = {"descriptions": descriptions_batch}
    data = json.dumps(payload).encode("utf-8")

    url = f"{API_BASE_URL}/clean_text"
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("X-API-Key", API_KEY)
    req.add_header("Accept", "application/json")

    logger.debug(
        f"Calling /clean_text API for {len(descriptions_batch)} descriptions. URL: {url}"
    )

    try:
        with urllib.request.urlopen(
            req, timeout=60
        ) as response:  # Increased timeout for potentially larger batches
            response_content = response.read().decode("utf-8")
            if response.status == 200:
                response_data = json.loads(response_content)
                if "cleaned_descriptions" in response_data and isinstance(
                    response_data["cleaned_descriptions"], list
                ):
                    logger.debug(
                        f"/clean_text API call successful. Received {len(response_data['cleaned_descriptions'])} cleaned descriptions."
                    )
                    return response_data["cleaned_descriptions"]
                else:
                    logger.error(
                        f"/clean_text API response missing 'cleaned_descriptions' or not a list. Response: {response_data}"
                    )
                    return descriptions_batch
            else:
                logger.error(
                    f"/clean_text API call failed with status {response.status}: {response_content}"
                )
                return descriptions_batch
    except urllib.error.HTTPError as e:
        # HTTPError can also be caught by URLError, but this gives more specific status
        error_content = "N/A"
        try:
            error_content = e.read().decode("utf-8")
        except Exception:
            pass
        logger.error(
            f"HTTP error calling /clean_text API ({e.code} {e.reason}): {error_content}"
        )
        return descriptions_batch
    except urllib.error.URLError as e:
        # This can include [Errno 61] Connection refused if server is down
        logger.error(
            f"Network error or other URLError calling /clean_text API: {e.reason}. URL: {url}"
        )
        return descriptions_batch
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to decode JSON response from /clean_text API: {e}. Response was: {response_content if 'response_content' in locals() else 'N/A'}"
        )
        return descriptions_batch
    except Exception as e:
        logger.error(f"Unexpected error calling /clean_text API: {e}", exc_info=True)
        return descriptions_batch


def process_transactions(csv_filepath):
    """
    Reads transactions from a CSV file, cleans the 'Details' field using
    the /clean_text API, and groups them by the cleaned details.
    """
    cleaned_transactions_groups = defaultdict(
        lambda: {"count": 0, "total_amount": 0.0, "original_descriptions": set()}
    )

    if not os.path.exists(csv_filepath):
        logger.error(f"CSV file not found at {csv_filepath}")
        return

    logger.info(
        f"Starting transaction processing from: {csv_filepath} using API: {API_BASE_URL}"
    )
    if not API_KEY:
        logger.warning(
            "TEST_API_KEY is not set (or not loaded from .env). API calls to /clean_text will likely fail or return originals."
        )

    rows_processed = 0
    BATCH_SIZE = 100  # Process in batches of 100
    current_batch_data = []  # Stores dicts of {'original': details, 'amount': amount}

    try:
        with open(csv_filepath, mode="r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            if "Details" not in reader.fieldnames or "Amount" not in reader.fieldnames:
                logger.error("CSV file must contain 'Details' and 'Amount' columns.")
                return

            for i, row in enumerate(reader):
                rows_processed += 1
                details = row.get("Details")
                amount_str = row.get("Amount")

                if not details:
                    logger.warning(
                        f"Skipping row {i+2} (header is row 1, data starts row 2) due to missing or empty 'Details'."
                    )
                    continue

                try:
                    amount = float(amount_str) if amount_str else 0.0
                except ValueError:
                    logger.warning(
                        f"Could not parse amount '{amount_str}' in row {i+2}. Using 0.0 for amount."
                    )
                    amount = 0.0

                current_batch_data.append(
                    {"original": details, "amount": amount, "row_num": i + 2}
                )

                if len(current_batch_data) >= BATCH_SIZE:
                    descriptions_to_clean = [
                        item["original"] for item in current_batch_data
                    ]
                    cleaned_descriptions_from_api = call_clean_text_api(
                        descriptions_to_clean
                    )

                    if len(cleaned_descriptions_from_api) == len(current_batch_data):
                        for idx, cleaned_detail in enumerate(
                            cleaned_descriptions_from_api
                        ):
                            original_item = current_batch_data[idx]
                            group = cleaned_transactions_groups[cleaned_detail]
                            group["count"] += 1
                            group["total_amount"] += original_item["amount"]
                            if (
                                len(group["original_descriptions"]) < 5
                            ):  # Store up to 5 examples
                                group["original_descriptions"].add(
                                    original_item["original"]
                                )
                    else:
                        logger.error(
                            f"Mismatch in batch sizes after API call: API returned {len(cleaned_descriptions_from_api)}, expected {len(current_batch_data)}. Skipping this batch processing."
                        )

                    logger.info(
                        f"Processed batch ending at row {i+2}. Total rows examined: {rows_processed}"
                    )
                    current_batch_data = []  # Reset batch

            # Process any remaining transactions in the last batch
            if current_batch_data:
                logger.info(
                    f"Processing final batch of {len(current_batch_data)} transactions."
                )
                descriptions_to_clean = [
                    item["original"] for item in current_batch_data
                ]
                cleaned_descriptions_from_api = call_clean_text_api(
                    descriptions_to_clean
                )

                if len(cleaned_descriptions_from_api) == len(current_batch_data):
                    for idx, cleaned_detail in enumerate(cleaned_descriptions_from_api):
                        original_item = current_batch_data[idx]
                        group = cleaned_transactions_groups[cleaned_detail]
                        group["count"] += 1
                        group["total_amount"] += original_item["amount"]
                        if len(group["original_descriptions"]) < 5:
                            group["original_descriptions"].add(
                                original_item["original"]
                            )
                else:
                    logger.error(
                        f"Mismatch in final batch sizes: API returned {len(cleaned_descriptions_from_api)}, expected {len(current_batch_data)}. Skipping final batch processing."
                    )
            current_batch_data = []

    except FileNotFoundError:
        logger.error(f"Could not find the CSV file at: {csv_filepath}")
        return
    except Exception as e:
        logger.error(
            f"An error occurred while processing the CSV file: {e}", exc_info=True
        )
        return

    logger.info(f"Finished processing {rows_processed} transactions.")
    logger.info("\n--- Transaction Grouping Results (Sorted by Count) ---")

    sorted_groups = sorted(
        cleaned_transactions_groups.items(),
        key=lambda item: item[1]["count"],
        reverse=True,
    )

    for cleaned_name, data in sorted_groups:
        print(f'\nCleaned Name: "{cleaned_name}"')
        print(f"  Count: {data['count']}")
        print(f"  Total Amount: {data['total_amount']:.2f}")
        if data["original_descriptions"]:
            print(f"  Example Original Descriptions (up to 5 unique):")
            for desc in list(data["original_descriptions"])[:5]:
                print(f'    - "{desc}"')
        else:
            print("  No original descriptions captured for this group.")


if __name__ == "__main__":
    csv_file_path = os.path.join(
        PROJECT_ROOT, "tests", "test_data", "ANZ Transactions Nov 2024 to May 2025.csv"
    )

    logger.info("Starting transaction cleaning and grouping script (API mode).")
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Target API URL: {API_BASE_URL}")
    logger.info(
        f"API Key Loaded: {'Yes' if API_KEY else 'No - TEST_API_KEY not set (or not loaded from .env)!'}"
    )
    logger.info(f"Attempting to load CSV from: {csv_file_path}")

    process_transactions(csv_file_path)
    logger.info("Script finished.")
