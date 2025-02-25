import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime
import replicate
from supabase import create_client, Client
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import re
import logging
import gc
import sys
import uuid
import time
from typing import List
from threading import Thread
from functools import lru_cache
from datetime import datetime, timedelta

# Dictionary to store prediction data
predictions_db = {}

# Cache for Replicate API responses
replicate_cache = {}

# Configure logging
logging.basicConfig(
    stream=sys.stdout,  # Log to stdout for Docker/Gunicorn to capture
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize Flask app with logging
app = Flask(__name__)
CORS(app)

# Log startup information
logger.info("=== Main Application Starting ===")
logger.info(f"Environment: {os.environ.get('FLASK_ENV', 'production')}")

@app.route('/')
def home():
    """Home endpoint"""
    logger.info("Home endpoint accessed")
    return jsonify({
        "status": "online",
        "version": "1.0.0",
        "endpoints": [
            "/classify",
            "/train",
            "/health",
            "/api-key",
            "/status/:prediction_id"
        ]
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed")
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

# Initialize Supabase client with logging
try:
    logger.info("=== Initializing Supabase ===")
    supabase_url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
    supabase_key = os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    
    if not all([supabase_url, supabase_key]):
        logger.error("Missing required environment variables for Supabase")
        raise ValueError("Missing required environment variables: NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY")
    
    logger.info(f"Supabase URL configured: {supabase_url}")
    logger.info(f"Supabase key length: {len(supabase_key)}")
    
    supabase = create_client(supabase_url, supabase_key)
    logger.info("Supabase client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Supabase: {str(e)}")
    raise

BACKEND_API = os.environ.get("BACKEND_API")

def validate_api_key(api_key: str) -> str:
    """Validate API key and return user ID if valid."""
    try:
        # Clean the API key
        api_key = api_key.strip()
        logger.info(f"Validating API key (length: {len(api_key)})")
        logger.debug(f"API key first/last 4 chars: {api_key[:4]}...{api_key[-4:]}")
        
        # Log the query we're about to make
        logger.info("Querying Supabase for API key validation")
        
        # First try exact match
        response = supabase.table("account").select("*").eq("api_key", api_key).execute()
        logger.debug(f"Exact match query response count: {len(response.data) if response.data else 0}")
        
        if not response.data:
            # Try case-insensitive match as fallback
            logger.info("No exact match found, trying case-insensitive match")
            response = supabase.table("account").select("*").ilike("api_key", api_key).execute()
            logger.debug(f"Case-insensitive query response count: {len(response.data) if response.data else 0}")
            
        if not response.data:
            logger.error(f"No account found for API key: {api_key[:4]}...{api_key[-4:]}")
            raise Exception("Invalid API key - no matching account found")
            
        # Log the found user data (excluding sensitive info)
        user_data = response.data[0]
        logger.info(f"Found user data - userId: {user_data.get('userId')}")
        logger.debug(f"User data keys: {list(user_data.keys())}")
        
        if not user_data.get("userId"):
            logger.error("User data found but missing userId")
            raise Exception("Invalid user configuration - missing userId")
        
        return user_data["userId"]
        
    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        logger.error(f"Full error details: {e}")
        logger.error(f"API key validation failed for key: {api_key[:4]}...{api_key[-4:]}")
        raise Exception(f"API key validation failed: {str(e)}")

def update_process_status(status_text: str, mode: str, user_id: str) -> None:
    """Log process status without database updates."""
    try:
        logger.info(f"Process status update - mode: {mode}, user: {user_id}, status: {status_text}")
    except Exception as e:
        logger.error(f"Error logging process status: {e}")
        logger.error(f"Status update attempted - mode: {mode}, user: {user_id}, status: {status_text}")

def clean_text(text: str) -> str:
    """Clean transaction description text while preserving business names."""
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Remove only transaction-specific metadata
    patterns = [
        r'\s*\d{2,4}[-/]\d{2}[-/]\d{2,4}',  # Dates
        r'\s*\d{2}:\d{2}(?::\d{2})?',  # Times
        r'\s*Card\s+[xX*]+\d{4}',  # Card numbers
        r'\s*\|\s*[\d\.]+$',  # Amount at end
        r'\s*\|\s*[A-Z0-9\s]+$',  # Reference codes
        r'\s+(?:Value Date|Card ending|ref|reference)\s*:?.*$',  # Transaction metadata
        r'(?i)\s+(?:AUS|USA|UK|NS|CYP)$',  # Country codes at end
        r'\s+\([^)]*\)$',  # Anything in parentheses at the end
    ]
    
    # Apply patterns one by one
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def run_prediction(mode: str, sheet_id: str, user_id: str, descriptions: list, sheet_name: str = None, category_column: str = None) -> dict:
    """Run prediction using Replicate API."""
    try:
        model = replicate.models.get("replicate/all-mpnet-base-v2")
        version = model.versions.get(
            "b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305"
        )

        # Create webhook URL with all necessary parameters
        webhook_params = [
            f"spreadsheetId={sheet_id}",
            f"userId={user_id}"
        ]
        if sheet_name:
            webhook_params.append(f"sheetName={sheet_name}")
        if category_column:
            webhook_params.append(f"categoryColumn={category_column}")
            
        webhook = f"{BACKEND_API}/{mode}/webhook?{'&'.join(webhook_params)}"
        
        prediction = replicate.predictions.create(
            version=version,
            input={"text_batch": json.dumps(descriptions)},
            webhook=webhook,
            webhook_events_filter=["completed"]
        )
        
        return prediction
    except Exception as e:
        logger.error(f"Error running prediction: {e}")
        raise

def store_embeddings(bucket_name: str, file_name: str, data: np.ndarray) -> None:
    """Store embeddings in Supabase storage."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp_file:
            np.savez_compressed(temp_file, data)
            temp_file_path = temp_file.name

        with open(temp_file_path, "rb") as f:
            supabase.storage.from_(bucket_name).upload(
                file_name,
                f,
                file_options={"x-upsert": "true"}
            )

        os.unlink(temp_file_path)
    except Exception as e:
        logger.error(f"Error storing embeddings: {e}")
        raise

def fetch_embeddings(bucket_name: str, file_name: str) -> np.ndarray:
    """Fetch embeddings from Supabase storage."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp_file:
            response = supabase.storage.from_(bucket_name).download(file_name)
            temp_file.write(response)
            temp_file_path = temp_file.name

        data = np.load(temp_file_path, allow_pickle=True)["arr_0"]
        os.unlink(temp_file_path)
        
        # Check if this is index data with the old format and migrate if needed
        if file_name.endswith('_index.npy') and data.dtype.names and 'Category' in data.dtype.names and 'category' not in data.dtype.names:
            logger.info(f"Detected old format index data in {file_name}, migrating to new format")
            # Create a new array with the updated field name
            new_data = np.array([(item['item_id'], item['Category'], item['description']) for item in data],
                              dtype=[('item_id', int), ('category', 'U100'), ('description', 'U500')])
            
            # Store the migrated data
            sheet_id = file_name.split('_index.npy')[0]
            logger.info(f"Storing migrated index data for sheet_id: {sheet_id}")
            store_embeddings(bucket_name, file_name, new_data)
            
            # Return the migrated data
            return new_data
            
        return data
    except Exception as e:
        logger.error(f"Error fetching embeddings: {e}")
        raise

def update_sheet_log(sheet_id: str, status: str, message: str, details: str = '') -> None:
    """Log sheet update without writing directly to the sheet."""
    try:
        # Just log the message instead of writing to the sheet
        logger.info(f"Sheet log update - sheet_id: {sheet_id}, status: {status}, message: {message}, details: {details}")
    except Exception as e:
        logger.error(f"Error logging sheet update: {e}")

@app.route("/train", methods=["POST"])
def train_model():
    """Train the model with new data."""
    try:
        # Log all incoming request details
        logger.info("=== Incoming Training Request ===")
        logger.info(f"Headers: {dict(request.headers)}")
        
        # Get request data
        data = request.get_json()
        logger.info(f"Request data: {data}")
        
        if not data:
            logger.error("Missing request data")
            return jsonify({"error": "Missing request data"}), 400
            
        if not isinstance(data, dict):
            logger.error(f"Invalid request format - expected JSON object, got {type(data)}")
            return jsonify({"error": "Invalid request format - expected JSON object"}), 400
            
        if 'transactions' not in data:
            logger.error("Missing transactions data in request")
            return jsonify({"error": "Missing transactions data"}), 400
            
        # Support both parameter names for backward compatibility
        sheet_id = data.get('spreadsheetId') or data.get('expenseSheetId')
        if not sheet_id or not isinstance(sheet_id, str) or len(sheet_id.strip()) == 0:
            logger.error(f"Invalid or missing spreadsheetId: {sheet_id}")
            return jsonify({"error": "Invalid or missing spreadsheetId"}), 400
            
        # Get user ID either from API key validation or payload
        user_id = None
        api_key = request.headers.get('X-API-Key')
        
        # First try to validate API key if provided
        if api_key:
            try:
                user_id = validate_api_key(api_key)
                logger.info(f"Got user_id from API key validation: {user_id}")
            except Exception as e:
                # If API key validation fails and we're not in fallback mode, return error
                if not data.get('userId'):
                    logger.error(f"API key validation failed and no userId provided in payload: {str(e)}")
                    return jsonify({"error": f"API key validation failed: {str(e)}"}), 401
                logger.error(f"API key validation failed: {str(e)}")
                logger.info("Falling back to userId from payload")
        
        # If no user_id from API key, try payload
        if not user_id:
            user_id = data.get('userId')
            if user_id:
                # If user_id is an email, prefix it with google-oauth2|
                if '@' in user_id and not user_id.startswith('google-oauth2|'):
                    user_id = f"google-oauth2|{user_id}"
                logger.info(f"Got user_id from payload: {user_id}")
            else:
                logger.error("No valid user_id found in API key or payload")
                return jsonify({"error": "No valid user ID found. Please provide either a valid API key or userId in the request."}), 401
            
        # Create or update user configuration
        try:
            # Check if user config exists
            response = supabase.table("account").select("*").eq("userId", user_id).execute()
            
            if not response.data:
                # Create new user config
                default_config = {
                    "userId": user_id,
                    "categorisationRange": "A:Z",
                    "columnOrderCategorisation": {
                        "categoryColumn": "E",
                        "descriptionColumn": "C"
                    },
                    "categorisationTab": None,
                    "api_key": api_key if api_key else None
                }
                supabase.table("account").insert(default_config).execute()
                logger.info(f"Created new user configuration for {user_id}")
            else:
                # Update API key if it's provided and different from stored one
                existing_config = response.data[0]
                if api_key and (not existing_config.get("api_key") or existing_config.get("api_key") != api_key):
                    supabase.table("account").update({"api_key": api_key}).eq("userId", user_id).execute()
                    logger.info(f"Updated API key for user {user_id}")
                logger.info(f"Found existing user configuration for {user_id}")
                
        except Exception as e:
            logger.warning(f"Error managing user configuration: {str(e)}")
            # Continue with training even if config creation fails
            
        # Convert transactions to DataFrame with error handling
        try:
            logger.info(f"Converting transactions to DataFrame. Sample: {data['transactions'][:2]}")
            df = pd.DataFrame(data['transactions'])
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error converting transactions to DataFrame: {str(e)}")
            return jsonify({"error": "Invalid transaction data format"}), 400
            
        # Validate required columns
        required_columns = ['Narrative', 'Category']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return jsonify({
                "error": f"Missing required columns: {missing_columns}"
            }), 400
            
        # Validate data quality
        df['Narrative'] = df['Narrative'].astype(str).str.strip()
        df['Category'] = df['Category'].astype(str).str.strip()
        
        if df['Narrative'].empty or df['Narrative'].isna().any():
            logger.error("Invalid or empty narratives found")
            return jsonify({"error": "Invalid or empty narratives found"}), 400
            
        if df['Category'].empty or df['Category'].isna().any():
            logger.error("Invalid or empty categories found")
            return jsonify({"error": "Invalid or empty categories found"}), 400
            
        # Clean descriptions
        df["description"] = df["Narrative"].apply(clean_text)
        df = df.drop_duplicates(subset=["description"])
        
        if len(df) < 10:  # Minimum required for meaningful training
            return jsonify({"error": "At least 10 valid transactions required for training"}), 400
        
        # Store training data index with proper dtype
        df["item_id"] = range(len(df))
        
        # Log the categories we're training with
        unique_categories = df['Category'].unique().tolist()
        logger.info(f"Training with {len(unique_categories)} unique categories: {unique_categories}")
        logger.info(f"Category column length: {df['Category'].str.len().describe()}")
        
        # Store training data with clear separation between category and description
        # Use a structured array with named fields for clarity
        # IMPORTANT: We're using df["Category"] for the category field, not df["Narrative"]
        training_data = np.array(list(zip(df["item_id"], df["Category"], df["description"])), 
                              dtype=[('item_id', int), ('category', 'U100'), ('description', 'U500')])
        
        # Log the first few entries to verify structure
        logger.info(f"First training data entry: {training_data[0]}")
        logger.info(f"Training data dtype: {training_data.dtype}")
        
        # Log a few sample entries to verify we're storing the correct category
        for i in range(min(5, len(training_data))):
            logger.info(f"Training entry {i}: id={training_data[i]['item_id']}, category='{training_data[i]['category']}', description='{training_data[i]['description']}'")
        
        store_embeddings("txclassify", f"{sheet_id}_index.npy", training_data)
        
        # Run prediction
        prediction = run_prediction("train", sheet_id, user_id, df["description"].tolist(), sheet_name=data.get('sheetName'), category_column=data.get('categoryColumn'))
        
        return jsonify({
            "status": "processing",
            "prediction_id": prediction.id
        })
        
    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/train/webhook", methods=["POST"])
def training_webhook():
    """Handle training webhook from Replicate."""
    sheet_id = request.args.get("spreadsheetId")
    user_id = request.args.get("userId")
    embeddings_shape = None  # Initialize embeddings_shape variable
    
    try:
        # Log incoming request details
        logger.info(f"Received webhook request for sheet_id: {sheet_id}, user_id: {user_id}")
        
        if not all([sheet_id, user_id]):
            error_msg = "Missing required parameters: spreadsheetId and userId"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400

        # Ensure user account exists
        try:
            response = supabase.table("account").select("*").eq("userId", user_id).execute()
            if not response.data:
                # Create new account with all fields from schema
                default_config = {
                    "userId": user_id,
                    "categorisationRange": "A:Z",
                    "columnOrderCategorisation": {
                        "categoryColumn": "E",
                        "descriptionColumn": "C"
                    },
                    "categorisationTab": None,  # Set to None as it's being deprecated
                    "api_key": None  # Include api_key field but set to None
                }
                supabase.table("account").insert(default_config).execute()
                logger.info(f"Created new account for user {user_id}")
        except Exception as e:
            logger.warning(f"Error checking/creating user account: {e}")
            # Continue with webhook processing even if account creation fails

        # Get prediction_id from multiple sources with logging
        prediction_id = None
        prediction_source = None
        
        # 1. Try Replicate header (most reliable)
        if 'X-Replicate-Prediction-Id' in request.headers:
            prediction_id = request.headers.get('X-Replicate-Prediction-Id')
            prediction_source = "Replicate header"
            
        # 2. Try custom header (backup)
        elif 'Prediction-Id' in request.headers:
            prediction_id = request.headers.get('Prediction-Id')
            prediction_source = "Custom header"
            
        # 3. Try query parameters
        elif request.args.get('prediction_id'):
            prediction_id = request.args.get('prediction_id')
            prediction_source = "Query parameter"
            
        # 4. Try JSON body as last resort
        elif request.is_json:
            try:
                data = request.get_json(force=True)
                if data.get('id'):
                    prediction_id = data.get('id')
                    prediction_source = "JSON body"
            except Exception as e:
                logger.warning(f"Error parsing request JSON for prediction ID: {e}")

        if prediction_id:
            logger.info(f"Found prediction_id: {prediction_id} from {prediction_source}")
        else:
            error_msg = "Missing prediction ID - not found in headers, query params, or body"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400

        # Check if we've already processed this webhook
        try:
            webhook_results = supabase.table("webhook_results").select("*").eq("prediction_id", prediction_id).execute()
            if webhook_results.data:
                logger.info(f"Webhook already processed for prediction_id: {prediction_id}")
                return jsonify({"status": "success", "message": "Already processed"}), 200
        except Exception as e:
            logger.warning(f"Error checking webhook results: {e}")

        # Parse and validate JSON data
        if not request.is_json:
            error_msg = "Request must be JSON"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
            
        try:
            data = request.get_json(force=True)
        except Exception as e:
            error_msg = f"Failed to parse JSON: {str(e)}"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400

        if not data or not isinstance(data, dict):
            error_msg = "Invalid JSON data format"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400

        # Process embeddings
        try:
            embeddings = np.array([item["embedding"] for item in data["output"]], dtype=np.float32)
            embeddings_shape = embeddings.shape  # Store shape before clearing embeddings
            logger.info(f"Successfully processed embeddings with shape: {embeddings_shape}")
            
            # Store embeddings
            store_embeddings("txclassify", f"{sheet_id}.npy", embeddings)
            
            # Log information about the stored embeddings
            logger.info(f"Stored embeddings for sheet_id: {sheet_id} with shape: {embeddings_shape}")
            
            # Try to load the index file to verify it exists and has the right structure
            try:
                index_data = fetch_embeddings("txclassify", f"{sheet_id}_index.npy")
                logger.info(f"Verified index data - shape: {index_data.shape}, dtype: {index_data.dtype}")
                if len(index_data) > 0:
                    logger.info(f"Sample index entry: {index_data[0]}")
            except Exception as e:
                logger.warning(f"Could not verify index data: {e}")
            
            # Clear memory
            del embeddings
            gc.collect()
            
        except (KeyError, TypeError) as e:
            error_msg = f"Invalid embedding data structure: {str(e)}"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
        
        # Store webhook result with minimal data
        try:
            result = {
                "prediction_id": prediction_id,
                "results": {
                    "user_id": user_id,
                    "status": "success",
                    "embeddings_shape": embeddings_shape
                }
            }
            supabase.table("webhook_results").insert(result).execute()
            logger.info(f"Stored webhook result for prediction_id: {prediction_id}")
        except Exception as e:
            logger.warning(f"Error storing webhook result: {e}")
        
        # Update status
        update_process_status("completed", "training", user_id)
        
        logger.info("Training webhook completed successfully")
        return jsonify({
            "status": "success",
            "message": "Training completed successfully"
        })
        
    except Exception as e:
        error_msg = f"Error in training webhook: {str(e)}"
        logger.error(error_msg)
        if user_id:
            update_process_status(f"Error: {str(e)}", "training", user_id)
        return jsonify({"error": error_msg}), 500

@app.route("/classify", methods=["POST"])
def classify_transactions():
    """Classify transactions endpoint."""
    try:
        # Validate API key
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return jsonify({"error": "Missing API key"}), 401

        # Get and validate request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate required fields
        required_fields = ["transactions", "spreadsheetId"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Extract data
        transactions = data["transactions"]
        spreadsheet_id = data["spreadsheetId"]
        sheet_name = data.get("sheetName", "new_transactions")  # Default to "new_transactions" if not provided
        category_column = data.get("categoryColumn", "E")  # Default to column E if not provided
        start_row = data.get("startRow", "1")  # Default to row 1 if not provided

        if not transactions:
            return jsonify({"error": "No transactions provided"}), 400

        # Validate API key and get user ID
        try:
            user_id = validate_api_key(api_key)
            logger.info(f"API key validated for user: {user_id}")
            
            # Update API key in user account if needed
            try:
                response = supabase.table("account").select("*").eq("userId", user_id).execute()
                if response.data:
                    existing_config = response.data[0]
                    if not existing_config.get("api_key") or existing_config.get("api_key") != api_key:
                        supabase.table("account").update({"api_key": api_key}).eq("userId", user_id).execute()
                        logger.info(f"Updated API key for user {user_id}")
            except Exception as e:
                logger.warning(f"Error updating API key in user account: {e}")
                # Continue with classification even if update fails
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return jsonify({"error": f"API key validation failed: {str(e)}"}), 401

        # Verify training data exists
        try:
            trained_data = fetch_embeddings("txclassify", f"{spreadsheet_id}_index.npy")
            if len(trained_data) == 0:
                error_msg = "No training data found. Please train the model first."
                return jsonify({"error": error_msg}), 400
        except Exception as e:
            error_msg = "No training data found. Please train the model first."
            return jsonify({"error": error_msg}), 400

        # Start classification process
        prediction_id = str(uuid.uuid4())
        
        # Store prediction metadata
        predictions_db[prediction_id] = {
            "status": "processing",
            "user_id": user_id,
            "spreadsheet_id": spreadsheet_id,
            "sheet_name": sheet_name,  # Store sheet name
            "category_column": category_column,  # Store category column
            "start_row": start_row,  # Store start row
            "total_transactions": len(transactions),
            "processed_transactions": 0
        }

        # Start classification in background
        Thread(
            target=process_classification,
            args=(prediction_id, transactions, user_id)
        ).start()

        return jsonify({
            "status": "processing",
            "prediction_id": prediction_id
        })

    except Exception as e:
        logger.error(f"Error in classify_transactions: {e}")
        return jsonify({"error": str(e)}), 500

def process_classification(prediction_id: str, transactions: List[dict], user_id: str):
    """Process classification in background."""
    try:
        predictions_db[prediction_id]["status"] = "classifying"
        
        # Get trained embeddings and categories
        spreadsheet_id = predictions_db[prediction_id]["spreadsheet_id"]
        sheet_name = predictions_db[prediction_id].get("sheet_name", "new_transactions")
        
        trained_embeddings = fetch_embeddings("txclassify", f"{spreadsheet_id}.npy")
        trained_data = fetch_embeddings("txclassify", f"{spreadsheet_id}_index.npy")
        
        if len(trained_embeddings) == 0 or len(trained_data) == 0:
            predictions_db[prediction_id]["status"] = "failed"
            predictions_db[prediction_id]["error"] = "No training data found"
            return

        # Clean and prepare descriptions
        descriptions = [clean_text(t["description"]) for t in transactions if t.get("description")]
        
        # Get embeddings for new descriptions
        model = replicate.models.get("replicate/all-mpnet-base-v2")
        version = model.versions.get("b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305")
        
        # Create webhook URL with all necessary parameters
        webhook_params = [
            f"spreadsheetId={spreadsheet_id}",
            f"userId={user_id}",
            f"sheetName={sheet_name}",
            f"prediction_id={prediction_id}"
        ]
        
        # Add start row parameter if available
        start_row = predictions_db[prediction_id].get("start_row", "1")
        webhook_params.append(f"startRow={start_row}")
        
        # Add category column parameter if available
        category_column = predictions_db[prediction_id].get("category_column", "E")
        webhook_params.append(f"categoryColumn={category_column}")
        
        webhook = f"{BACKEND_API}/classify/webhook?{'&'.join(webhook_params)}"
        logger.info(f"Setting up webhook URL: {webhook}")
        
        # Get embeddings for new descriptions with webhook
        prediction = replicate.predictions.create(
            version=version,
            input={"text_batch": json.dumps(descriptions)},
            webhook=webhook,
            webhook_events_filter=["completed"]
        )
        
        # Store prediction ID for reference
        predictions_db[prediction_id]["replicate_prediction_id"] = prediction.id
        
        # Optimize the waiting strategy - use shorter initial checks
        attempt = 0
        max_attempts = 60  # Maximum number of attempts
        
        # Start with shorter checks for the first few attempts
        initial_delays = [2, 3, 5, 10, 15]  # First few checks are faster
        
        while prediction.status != "succeeded" and attempt < max_attempts:
            if prediction.status == "failed":
                predictions_db[prediction_id]["status"] = "failed"
                predictions_db[prediction_id]["error"] = prediction.error
                return
            
            # Use initial delays for first few attempts, then exponential backoff
            if attempt < len(initial_delays):
                delay = initial_delays[attempt]
            else:
                # Cap at 60 seconds for later attempts
                delay = min(20 * (1.5 ** (attempt - len(initial_delays))), 60)
                
            logger.info(f"Waiting {delay} seconds before checking prediction status again (attempt {attempt+1}/{max_attempts})")
            time.sleep(delay)
            
            # Reload prediction status
            prediction.reload()
            attempt += 1
            
            # Update progress information
            predictions_db[prediction_id]["status_message"] = f"Waiting for embeddings (attempt {attempt}/{max_attempts})"
            
            # If we've been waiting for a while, check if webhook results are already available
            if attempt > 5:
                try:
                    webhook_results = supabase.table("webhook_results").select("id").eq("prediction_id", prediction_id).execute()
                    if webhook_results.data:
                        logger.info(f"Found webhook results while waiting for embeddings (attempt {attempt})")
                        predictions_db[prediction_id]["status"] = "completed"
                        predictions_db[prediction_id]["status_message"] = "Classification completed successfully"
                        return
                except Exception as e:
                    logger.warning(f"Error checking webhook results during waiting: {e}")
        
        if attempt >= max_attempts:
            predictions_db[prediction_id]["status"] = "failed"
            predictions_db[prediction_id]["error"] = "Timed out waiting for embeddings"
            return
        
        # If we're using webhooks, we don't need to process the results here
        # The webhook will handle it
        predictions_db[prediction_id]["status"] = "waiting_for_webhook"
        predictions_db[prediction_id]["status_message"] = "Embeddings completed, waiting for webhook processing"
        logger.info(f"Embeddings completed for prediction {prediction_id}, waiting for webhook to process results")

    except Exception as e:
        logger.error(f"Error in process_classification: {e}")
        predictions_db[prediction_id]["status"] = "failed"
        predictions_db[prediction_id]["error"] = str(e)

@app.route("/classify/webhook", methods=["POST"])
def classify_webhook():
    """Handle classification webhook from Replicate."""
    try:
        data = request.get_json()
        sheet_id = request.args.get("spreadsheetId")
        user_id = request.args.get("userId")
        sheet_name = request.args.get("sheetName")
        
        # Log all parameters for debugging
        logger.info(f"Webhook parameters: sheet_id={sheet_id}, user_id={user_id}, sheet_name={sheet_name}")
        
        if not all([data, sheet_id, user_id]):
            error_msg = "Missing required parameters: data, spreadsheetId, or userId"
            logger.error(error_msg)
            if sheet_id:
                update_sheet_log(sheet_id, "ERROR", error_msg)
            return jsonify({"error": error_msg}), 400
            
        # Default sheet_name to "new_transactions" if not provided
        if not sheet_name:
            logger.warning("sheet_name not provided in webhook, defaulting to 'new_transactions'")
            sheet_name = "new_transactions"

        # Get new embeddings
        try:
            new_embeddings = np.array([item["embedding"] for item in data["output"]], dtype=np.float32)
            logger.info(f"Processed {len(new_embeddings)} new embeddings")
        except Exception as e:
            error_msg = f"Error processing embeddings: {str(e)}"
            logger.error(error_msg)
            update_sheet_log(sheet_id, "ERROR", error_msg)
            return jsonify({"error": error_msg}), 400
        
        # Get trained embeddings and categories
        try:
            trained_embeddings = fetch_embeddings("txclassify", f"{sheet_id}.npy")
            trained_data = fetch_embeddings("txclassify", f"{sheet_id}_index.npy")
            logger.info(f"Retrieved {len(trained_embeddings)} trained embeddings and {len(trained_data)} training data points")
            
            # Add debug logging to see the structure of trained_data
            if len(trained_data) > 0:
                logger.info(f"First training data point: {trained_data[0]}")
                logger.info(f"Training data dtype: {trained_data.dtype}")
                logger.info(f"Training data shape: {trained_data.shape}")
                
                # Log a few sample categories to verify content
                sample_categories = [trained_data[i]['category'] if 'category' in trained_data.dtype.names else 
                                    trained_data[i]['Category'] if 'Category' in trained_data.dtype.names else 
                                    trained_data[i][1] for i in range(min(5, len(trained_data)))]
                logger.info(f"Sample categories: {sample_categories}")
            
            if len(trained_embeddings) == 0 or len(trained_data) == 0:
                error_msg = "No training data found"
                logger.error(error_msg)
                update_sheet_log(sheet_id, "ERROR", error_msg)
                return jsonify({"error": error_msg}), 400
        except Exception as e:
            error_msg = f"Error fetching training data: {str(e)}"
            logger.error(error_msg)
            update_sheet_log(sheet_id, "ERROR", error_msg)
            return jsonify({"error": error_msg}), 400
        
        # Calculate similarities
        try:
            similarities = cosine_similarity(new_embeddings, trained_embeddings)
            best_matches = similarities.argmax(axis=1)
            logger.info(f"Calculated similarities with shape {similarities.shape}")
        except Exception as e:
            error_msg = f"Error calculating similarities: {str(e)}"
            logger.error(error_msg)
            update_sheet_log(sheet_id, "ERROR", error_msg)
            return jsonify({"error": error_msg}), 400
        
        # Get predicted categories and confidence scores
        results = []
        for i, idx in enumerate(best_matches):
            try:
                # Add debug logging for the first few matches
                if i < 3:
                    logger.info(f"Match {i}: trained_data[{idx}] = {trained_data[idx]}")
                
                # Extract the category directly from the structured array
                # This should be the actual category, not the description
                category = None
                
                # Try different field names and positions to get the category
                if 'category' in trained_data.dtype.names:
                    category = str(trained_data[idx]['category'])
                elif 'Category' in trained_data.dtype.names:
                    category = str(trained_data[idx]['Category'])
                else:
                    # Fallback to index 1 which should be the category field
                    category = str(trained_data[idx][1])
                
                similarity_score = float(similarities[i][idx])  # Get the similarity score
                
                # Log the extracted category and score
                if i < 3:
                    logger.info(f"Extracted category: '{category}', similarity score: {similarity_score:.2f}")
                
                results.append({
                    "predicted_category": category,
                    "similarity_score": similarity_score
                })
            except Exception as e:
                logger.error(f"Error processing prediction {i}: {str(e)}")
                logger.error(f"trained_data type: {type(trained_data)}, shape: {trained_data.shape if hasattr(trained_data, 'shape') else 'unknown'}")
                if i < 3 and idx < len(trained_data):
                    logger.error(f"trained_data[{idx}] = {trained_data[idx]}")
                results.append({
                    "predicted_category": "Unknown",
                    "similarity_score": 0.0
                })
        
        logger.info(f"Generated {len(results)} predictions")
        
        # Instead of writing to the sheet directly, return the results
        status_msg = f"Generated {len(results)} predictions for sheet '{sheet_name}'"
        update_sheet_log(sheet_id, "INFO", status_msg)
        
        try:
            # Get column configuration from request args
            category_column = request.args.get("categoryColumn", "E")
            start_row = int(request.args.get("startRow", "1"))  # Default to row 1 if not specified
            
            # Store webhook result with results data
            prediction_id = request.args.get("prediction_id", "unknown")
            
            try:
                supabase.table("webhook_results").insert({
                    "prediction_id": prediction_id,
                    "results": {
                        "user_id": user_id,
                        "status": "success",
                        "count": len(results),
                        "data": results
                    }
                }).execute()
                logger.info(f"Stored webhook result for sheet_id: {sheet_id}")
            except Exception as e:
                logger.warning(f"Error storing webhook result: {e}")
            
            status_msg = "Classification completed successfully"
            update_process_status("completed", "classify", user_id)
            update_sheet_log(sheet_id, "SUCCESS", status_msg)
            
            # Return the results to be handled by the Google Apps Script
            return jsonify({
                "status": "success", 
                "results": results,
                "config": {
                    "categoryColumn": category_column,
                    "startRow": start_row,
                    "sheetName": sheet_name,
                    "spreadsheetId": sheet_id
                }
            })
            
        except Exception as e:
            error_msg = f"Error processing results: {str(e)}"
            logger.error(error_msg)
            update_sheet_log(sheet_id, "ERROR", error_msg)
            return jsonify({"error": error_msg}), 500

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in classify webhook: {error_msg}")
        if user_id:
            update_process_status(f"Error: {str(e)}", "classify", user_id)
        if sheet_id:
            update_sheet_log(sheet_id, "ERROR", f"Classification webhook failed: {error_msg}")
        return jsonify({"error": error_msg}), 500

@app.route("/status/<prediction_id>", methods=["GET"])
def get_prediction_status(prediction_id):
    """Get the status of a prediction."""
    try:
        # First check if we have a completed webhook in Supabase - prioritize this check
        try:
            webhook_results = supabase.table("webhook_results").select("*").eq("prediction_id", prediction_id).execute()
            if webhook_results.data:
                logger.info(f"Found completed webhook results for prediction {prediction_id}")
                webhook_data = webhook_results.data[0]
                
                # Extract the actual results from the webhook data
                results_data = []
                if webhook_data.get("results") and webhook_data["results"].get("data"):
                    results_data = webhook_data["results"]["data"]
                    logger.info(f"Extracted {len(results_data)} results from webhook data")
                
                # Get configuration from predictions_db if available
                category_column = "E"  # Default
                start_row = "1"  # Default
                sheet_name = "Sheet1"  # Default
                spreadsheet_id = None
                
                if prediction_id in predictions_db:
                    pred_data = predictions_db[prediction_id]
                    category_column = pred_data.get("category_column", category_column)
                    start_row = pred_data.get("start_row", start_row)
                    sheet_name = pred_data.get("sheet_name", sheet_name)
                    spreadsheet_id = pred_data.get("spreadsheet_id", spreadsheet_id)
                
                # Return a response that includes the actual results
                return jsonify({
                    "status": "completed",
                    "message": "Classification completed successfully",
                    "results": results_data,  # Include the actual results array
                    "config": {
                        "categoryColumn": category_column,
                        "startRow": start_row,
                        "sheetName": sheet_name,
                        "spreadsheetId": spreadsheet_id
                    },
                    "result": {
                        "results": {
                            "status": "success",
                            "data": results_data
                        }
                    }  # Keep the original result format for backward compatibility
                })
        except Exception as e:
            logger.warning(f"Error checking webhook results: {e}")
        
        # Then check if we have this prediction in our local dictionary
        if prediction_id in predictions_db:
            logger.info(f"Found prediction {prediction_id} in local predictions_db")
            prediction_data = predictions_db[prediction_id]
            
            # If status is waiting_for_webhook, check Supabase again with a direct query
            # This helps in cases where the webhook has completed but we missed it
            if prediction_data.get("status") == "waiting_for_webhook":
                try:
                    # Quick check for webhook results
                    webhook_check = supabase.table("webhook_results").select("id").eq("prediction_id", prediction_id).execute()
                    if webhook_check.data:
                        # We found a result, get the full data
                        webhook_results = supabase.table("webhook_results").select("*").eq("prediction_id", prediction_id).execute()
                        webhook_data = webhook_results.data[0]
                        
                        # Extract results and return them
                        results_data = []
                        if webhook_data.get("results") and webhook_data["results"].get("data"):
                            results_data = webhook_data["results"]["data"]
                            logger.info(f"Found webhook results on second check: {len(results_data)} results")
                        
                        return jsonify({
                            "status": "completed",
                            "message": "Classification completed successfully",
                            "results": results_data,
                            "config": {
                                "categoryColumn": prediction_data.get("category_column", "E"),
                                "startRow": prediction_data.get("start_row", "1"),
                                "sheetName": prediction_data.get("sheet_name", "Sheet1"),
                                "spreadsheetId": prediction_data.get("spreadsheet_id")
                            },
                            "result": {
                                "results": {
                                    "status": "success",
                                    "data": results_data
                                }
                            }
                        })
                except Exception as e:
                    logger.warning(f"Error on second webhook check: {e}")
            
            # Return the status from predictions_db
            return jsonify({
                "status": prediction_data.get("status", "processing"),
                "message": prediction_data.get("status_message", "Processing in progress"),
                "processed_transactions": prediction_data.get("processed_transactions", 0),
                "total_transactions": prediction_data.get("total_transactions", 0)
            })
        
        # Check if we have a cached response that's less than 1 minute old
        current_time = datetime.now()
        if prediction_id in replicate_cache:
            cache_entry = replicate_cache[prediction_id]
            cache_age = current_time - cache_entry["timestamp"]
            
            # Use cached response if it's less than 1 minute old
            if cache_age < timedelta(minutes=1):
                logger.info(f"Using cached response for prediction {prediction_id} (age: {cache_age.total_seconds():.1f}s)")
                return jsonify(cache_entry["response"])
        
        # Try to get prediction from Replicate
        try:
            prediction = replicate.predictions.get(prediction_id)
            
            if not prediction:
                logger.warning(f"Prediction {prediction_id} not found in Replicate")
                response_data = {
                    "status": "not_found",
                    "message": "Prediction not found in Replicate"
                }
                # Cache the response
                replicate_cache[prediction_id] = {
                    "timestamp": current_time,
                    "response": response_data
                }
                return jsonify(response_data), 404
                
            # Return status based on prediction state
            status = prediction.status
            if status == "succeeded":
                response_data = {
                    "status": "completed",
                    "message": "Processing completed successfully"
                }
            elif status == "failed":
                response_data = {
                    "status": "failed",
                    "error": prediction.error
                }
            else:
                response_data = {
                    "status": status,
                    "message": "Processing in progress"
                }
                
            # Cache the response
            replicate_cache[prediction_id] = {
                "timestamp": current_time,
                "response": response_data
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.warning(f"Error fetching prediction from Replicate: {e}")
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"Error getting prediction status: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api-key', methods=['GET', 'POST'])
def manage_api_key():
    """Get or generate API key for a user."""
    try:
        # For GET requests, we need to validate the user
        if request.method == 'GET':
            # Check for existing API key in header
            api_key = request.headers.get('X-API-Key')
            if api_key:
                try:
                    # Validate the API key
                    user_id = validate_api_key(api_key)
                    logger.info(f"API key validated for user: {user_id}")
                    
                    # Return the API key
                    return jsonify({
                        "status": "success",
                        "user_id": user_id,
                        "api_key": api_key
                    })
                except Exception as e:
                    logger.error(f"API key validation failed: {e}")
                    return jsonify({"error": f"API key validation failed: {str(e)}"}), 401
            
            # If no API key in header, check for user_id in query params
            user_id = request.args.get('userId')
            if not user_id:
                return jsonify({"error": "Missing userId parameter"}), 400
                
            # If user_id is an email, prefix it with google-oauth2|
            if '@' in user_id and not user_id.startswith('google-oauth2|'):
                user_id = f"google-oauth2|{user_id}"
                
            # Look up user in database
            response = supabase.table("account").select("*").eq("userId", user_id).execute()
            if not response.data:
                return jsonify({"error": "User not found"}), 404
                
            # Return the API key if it exists
            user_data = response.data[0]
            if user_data.get("api_key"):
                return jsonify({
                    "status": "success",
                    "user_id": user_id,
                    "api_key": user_data["api_key"]
                })
            else:
                return jsonify({
                    "status": "not_found",
                    "user_id": user_id,
                    "message": "No API key found for this user"
                }), 404
        
        # For POST requests, we generate a new API key
        elif request.method == 'POST':
            data = request.get_json()
            if not data or 'userId' not in data:
                return jsonify({"error": "Missing userId in request body"}), 400
                
            user_id = data['userId']
            
            # If user_id is an email, prefix it with google-oauth2|
            if '@' in user_id and not user_id.startswith('google-oauth2|'):
                user_id = f"google-oauth2|{user_id}"
            
            # Generate a new API key (UUID)
            new_api_key = str(uuid.uuid4())
            
            # Check if user exists
            response = supabase.table("account").select("*").eq("userId", user_id).execute()
            
            if response.data:
                # Update existing user
                supabase.table("account").update({"api_key": new_api_key}).eq("userId", user_id).execute()
                logger.info(f"Updated API key for user {user_id}")
            else:
                # Create new user
                default_config = {
                    "userId": user_id,
                    "categorisationRange": "A:Z",
                    "columnOrderCategorisation": {
                        "categoryColumn": "E",
                        "descriptionColumn": "C"
                    },
                    "categorisationTab": None,
                    "api_key": new_api_key
                }
                supabase.table("account").insert(default_config).execute()
                logger.info(f"Created new user with API key: {user_id}")
            
            return jsonify({
                "status": "success",
                "user_id": user_id,
                "api_key": new_api_key
            })
            
    except Exception as e:
        logger.error(f"Error managing API key: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
