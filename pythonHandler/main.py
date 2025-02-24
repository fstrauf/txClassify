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

# Dictionary to store prediction data
predictions_db = {}

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
            "/debug/validate-key"
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

def get_user_config(user_id: str) -> dict:
    """Get user configuration from Supabase using API key-based user ID."""
    try:
        logger.info(f"Looking up user configuration for userId: {user_id}")
        
        # Query user configuration
        logger.info("Querying Supabase for user configuration...")
        try:
            response = supabase.table("account").select("*").eq("userId", user_id).execute()
            logger.info(f"Query response: {response.data}")
            
            if response.data:
                config = response.data[0]
                # Ensure all required fields exist
                if not config.get("columnOrderCategorisation"):
                    config["columnOrderCategorisation"] = {
                        "categoryColumn": "E",
                        "descriptionColumn": "C"
                    }
                if not config.get("categorisationRange"):
                    config["categorisationRange"] = "A:Z"
                if not config.get("categorisationTab"):
                    config["categorisationTab"] = None  # Set to None as it's being deprecated
                    
                # Update the configuration if we added any missing fields
                supabase.table("account").update(config).eq("userId", user_id).execute()
                logger.info(f"Updated configuration for user {user_id}")
                return config
                
        except Exception as e:
            logger.error(f"Error querying user configuration: {str(e)}")
            
        # If no configuration exists, create a default one
        logger.info(f"No configuration found, creating default for user {user_id}")
        default_config = {
            "userId": user_id,
            "categorisationRange": "A:Z",
            "columnOrderCategorisation": {
                "categoryColumn": "E",
                "descriptionColumn": "C"
            },
            "categorisationTab": None  # Set to None as it's being deprecated
        }
        
        # Insert default configuration
        try:
            insert_response = supabase.table("account").insert(default_config).execute()
            if insert_response and insert_response.data:
                logger.info(f"Created default configuration for user {user_id}")
                return default_config
            else:
                logger.error("Insert response was empty")
                raise Exception("Failed to create default configuration - empty response")
        except Exception as insert_error:
            logger.error(f"Error creating default configuration: {str(insert_error)}")
            raise Exception(f"Failed to create default configuration: {str(insert_error)}")
            
    except Exception as e:
        logger.error(f"Error in get_user_config: {str(e)}")
        raise Exception(f"User configuration error: {str(e)}")

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

def get_spreadsheet_data(sheet_id: str, range_name: str) -> list:
    """Get data from Google Sheets."""
    try:
        # Format range name to ensure proper escaping
        if '!' in range_name:
            sheet_name, cell_range = range_name.split('!')
            # Remove any existing quotes
            sheet_name = sheet_name.strip("'")
            # Add single quotes around the sheet name to handle special characters
            formatted_range = f"'{sheet_name}'!{cell_range}"
        else:
            formatted_range = range_name
        
        google_service_account = json.loads(os.environ.get("GOOGLE_SERVICE_ACCOUNT"))
        creds = Credentials.from_service_account_info(
            google_service_account,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        service = build("sheets", "v4", credentials=creds)
        result = service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range=formatted_range
        ).execute()
        return result.get("values", [])
    except Exception as e:
        logger.error(f"Error fetching spreadsheet data: {e}")
        raise

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
        return data
    except Exception as e:
        logger.error(f"Error fetching embeddings: {e}")
        raise

def update_sheet_log(sheet_id: str, status: str, message: str, details: str = '') -> None:
    """Update log sheet with status and message."""
    try:
        service = build("sheets", "v4", credentials=Credentials.from_service_account_info(
            json.loads(os.environ.get("GOOGLE_SERVICE_ACCOUNT")),
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        ))

        # Get or create Log sheet
        try:
            sheet_metadata = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
            log_sheet_exists = any(sheet['properties']['title'] == 'Log' for sheet in sheet_metadata['sheets'])
            
            if not log_sheet_exists:
                # Create Log sheet
                body = {
                    'requests': [{
                        'addSheet': {
                            'properties': {
                                'title': 'Log',
                                'gridProperties': {
                                    'frozenRowCount': 1
                                }
                            }
                        }
                    }]
                }
                service.spreadsheets().batchUpdate(spreadsheetId=sheet_id, body=body).execute()
                
                # Add headers
                headers = [['Timestamp', 'Status', 'Message', 'Details']]
                service.spreadsheets().values().update(
                    spreadsheetId=sheet_id,
                    range='Log!A1:D1',
                    valueInputOption='RAW',
                    body={'values': headers}
                ).execute()
                
                # Format headers
                format_request = {
                    'requests': [{
                        'repeatCell': {
                            'range': {
                                'sheetId': sheet_metadata['sheets'][-1]['properties']['sheetId'],
                                'startRowIndex': 0,
                                'endRowIndex': 1
                            },
                            'cell': {
                                'userEnteredFormat': {
                                    'backgroundColor': {'red': 0.95, 'green': 0.95, 'blue': 0.95},
                                    'textFormat': {'bold': True},
                                    'horizontalAlignment': 'CENTER'
                                }
                            },
                            'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                        }
                    }]
                }
                service.spreadsheets().batchUpdate(spreadsheetId=sheet_id, body=format_request).execute()

        except Exception as e:
            logger.error(f"Error checking/creating Log sheet: {e}")
            return

        # Add new log entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row = [[timestamp, status, message, details]]
        
        # Get current values to determine insert position
        result = service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range='Log!A:A'
        ).execute()
        current_rows = len(result.get('values', []))
        
        # Insert after header, limited to 1000 rows
        insert_range = f'Log!A{min(2, current_rows + 1)}:D{min(2, current_rows + 1)}'
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=insert_range,
            valueInputOption='RAW',
            body={'values': new_row}
        ).execute()
        
        # Keep only last 1000 rows
        if current_rows > 1000:
            clear_range = f'Log!A{1001}:D{current_rows}'
            service.spreadsheets().values().clear(
                spreadsheetId=sheet_id,
                range=clear_range
            ).execute()
        
    except Exception as e:
        logger.error(f"Error updating sheet log: {e}")

def ensure_default_config(user_id: str) -> None:
    """Ensure user has default configuration values."""
    try:
        logger.info(f"Ensuring default configuration for user {user_id}")
        response = supabase.table("account").select("*").eq("userId", user_id).execute()
        
        if not response.data:
            logger.error(f"No account found for user {user_id}")
            raise Exception("Account not found")
            
        account = response.data[0]
        default_config = {
            "categorisationRange": "A:Z",
            "columnOrderCategorisation": {"categoryColumn": "E", "descriptionColumn": "C"}
        }
        
        # Check which fields are missing and need to be updated
        update_fields = {}
        for key, value in default_config.items():
            if key not in account or not account[key]:
                update_fields[key] = value
                
        if update_fields:
            logger.info(f"Updating missing fields for user {user_id}: {list(update_fields.keys())}")
            supabase.table("account").update(update_fields).eq("userId", user_id).execute()
            logger.info("Successfully updated user configuration with default values")
        else:
            logger.info("User configuration already has all required fields")
            
    except Exception as e:
        logger.error(f"Error ensuring default configuration: {str(e)}")
        raise Exception(f"Failed to ensure default configuration: {str(e)}")

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
        
        if api_key:
            try:
                user_id = validate_api_key(api_key)
                logger.info(f"Got user_id from API key validation: {user_id}")
            except Exception as e:
                logger.error(f"API key validation failed: {str(e)}")
                # Don't return error yet, try userId from payload
        
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
                    "columnRange": "A:Z",
                    "categoryColumn": "E"
                }
                supabase.table("account").insert(default_config).execute()
                logger.info(f"Created new user configuration for {user_id}")
            else:
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
        training_data = np.array(list(zip(df["item_id"], df["Category"], df["description"])), 
                               dtype=[('item_id', int), ('Category', 'U100'), ('description', 'U500')])
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

        # Validate API key and get user ID
        user_id = validate_api_key(api_key)
        
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
        spreadsheet_id = data["spreadsheetId"]  # Only used for tracking

        if not transactions:
            return jsonify({"error": "No transactions provided"}), 400

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
            "spreadsheet_id": spreadsheet_id,  # Only used for tracking
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
        
        # Get embeddings for new descriptions
        prediction = replicate.predictions.create(
            version=version,
            input={"text_batch": json.dumps(descriptions)}
        )
        
        # Wait for embeddings
        while prediction.status != "succeeded":
            if prediction.status == "failed":
                predictions_db[prediction_id]["status"] = "failed"
                predictions_db[prediction_id]["error"] = prediction.error
                return
            time.sleep(1)
            prediction.reload()
        
        # Process embeddings
        new_embeddings = np.array([item["embedding"] for item in prediction.output], dtype=np.float32)
        
        # Calculate similarities
        similarities = cosine_similarity(new_embeddings, trained_embeddings)
        best_matches = similarities.argmax(axis=1)
        
        # Get predicted categories and confidence scores
        results = []
        for i, idx in enumerate(best_matches):
            try:
                category = str(trained_data[idx][1])  # Index 1 is the Category field
                similarity_score = float(similarities[i][idx])  # Get the similarity score
                results.append({
                    "description": descriptions[i],
                    "predicted_category": category,
                    "similarity_score": similarity_score
                })
            except Exception as e:
                logger.error(f"Error processing prediction {i}: {str(e)}")
                results.append({
                    "description": descriptions[i],
                    "predicted_category": "Unknown",
                    "similarity_score": 0.0
                })
            
            # Update progress
            predictions_db[prediction_id]["processed_transactions"] = i + 1

        # Store results
        predictions_db[prediction_id]["status"] = "completed"
        predictions_db[prediction_id]["results"] = results

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
        
        if not all([data, sheet_id, user_id, sheet_name]):
            error_msg = "Missing required parameters"
            if sheet_id:
                update_sheet_log(sheet_id, "ERROR", error_msg)
            return jsonify({"error": error_msg}), 400

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
                category = str(trained_data[idx][1])  # Index 1 is the Category field
                similarity_score = float(similarities[i][idx])  # Get the similarity score
                results.append({
                    "predicted_category": category,
                    "similarity_score": similarity_score
                })
            except Exception as e:
                logger.error(f"Error processing prediction {i}: {str(e)}")
                results.append({
                    "predicted_category": "Unknown",
                    "similarity_score": 0.0
                })
        
        logger.info(f"Generated {len(results)} predictions")
        
        # Update sheet with predictions
        status_msg = f"Writing {len(results)} predictions to sheet"
        update_sheet_log(sheet_id, "INFO", status_msg)
        
        try:
            service = build("sheets", "v4", credentials=Credentials.from_service_account_info(
                json.loads(os.environ.get("GOOGLE_SERVICE_ACCOUNT")),
                scopes=["https://www.googleapis.com/auth/spreadsheets"]
            ))
            
            # Get column configuration from request args
            category_column = request.args.get("categoryColumn", "E")
            
            # Write categories back to sheet starting from row 1
            category_range = f"{sheet_name}!{category_column}1:{category_column}{len(results)}"
            logger.info(f"Writing to range: {category_range}")
            
            service.spreadsheets().values().update(
                spreadsheetId=sheet_id,
                range=category_range,
                valueInputOption="USER_ENTERED",
                body={"values": [[cat["predicted_category"]] for cat in results]}
            ).execute()
            
            # Write confidence scores in the next column
            confidence_column = chr(ord(category_column) + 1)
            confidence_range = f"{sheet_name}!{confidence_column}1:{confidence_column}{len(results)}"
            
            service.spreadsheets().values().update(
                spreadsheetId=sheet_id,
                range=confidence_range,
                valueInputOption="USER_ENTERED",
                body={"values": [[f"{cat['similarity_score']:.2%}"] for cat in results]}
            ).execute()
            
            status_msg = "Classification completed successfully"
            update_process_status("completed", "classify", user_id)
            update_sheet_log(sheet_id, "SUCCESS", status_msg)
            return jsonify({"status": "success"})
            
        except Exception as e:
            error_msg = f"Error writing to sheet: {str(e)}"
            logger.error(error_msg)
            update_sheet_log(sheet_id, "ERROR", error_msg)
            return jsonify({"error": error_msg}), 500

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in classify webhook: {error_msg}")
        if user_id:
            update_process_status(f"Error: {error_msg}", "classify", user_id)
        if sheet_id:
            update_sheet_log(sheet_id, "ERROR", f"Classification webhook failed: {error_msg}")
        return jsonify({"error": error_msg}), 500

@app.route("/status/<prediction_id>", methods=["GET"])
def get_prediction_status(prediction_id):
    """Get the status of a prediction."""
    try:
        # Get prediction from Replicate
        prediction = replicate.predictions.get(prediction_id)
        
        if not prediction:
            return jsonify({"error": "Prediction not found"}), 404
            
        # Check if we have a completed webhook in Supabase
        try:
            webhook_results = supabase.table("webhook_results").select("*").eq("prediction_id", prediction_id).execute()
            if webhook_results.data:
                return jsonify({
                    "status": "completed",
                    "result": webhook_results.data[0]
                })
        except Exception as e:
            logger.warning(f"Error checking webhook results: {e}")
        
        # Return status based on prediction state
        status = prediction.status
        if status == "succeeded":
            return jsonify({
                "status": "completed",
                "message": "Processing completed successfully"
            })
        elif status == "failed":
            return jsonify({
                "status": "failed",
                "error": prediction.error
            })
        else:
            return jsonify({
                "status": status,
                "message": "Processing in progress"
            })
            
    except Exception as e:
        logger.error(f"Error getting prediction status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/debug/validate-key', methods=['GET'])
def debug_validate_key():
    """Debug endpoint to validate API key."""
    try:
        # Get API key from headers
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            logger.error("No API key provided in request headers")
            return jsonify({
                "error": "API key is required",
                "details": "Please provide X-API-Key header"
            }), 401

        # Log the API key we're about to validate (safely)
        logger.info(f"Debug validation - API key: {api_key[:4]}...{api_key[-4:]}")

        # Query Supabase directly
        logger.info("Querying Supabase for account table data")
        try:
            # First try exact match
            response = supabase.table("account").select("*").eq("api_key", api_key).execute()
            match_type = "exact"
            
            if not response.data:
                # Try case-insensitive match
                response = supabase.table("account").select("*").ilike("api_key", api_key).execute()
                match_type = "case-insensitive"
            
            logger.info(f"Query response count: {len(response.data) if response.data else 0}")
            logger.info(f"Match type used: {match_type}")
            
            if response.data:
                user_data = response.data[0]
                return jsonify({
                    "status": "success",
                    "match_type": match_type,
                    "user_id": user_data.get("userId"),
                    "account_fields": list(user_data.keys()),
                    "has_api_key": bool(user_data.get("api_key")),
                    "api_key_length": len(user_data.get("api_key", "")),
                })
            else:
                return jsonify({
                    "status": "error",
                    "error": "No matching account found",
                    "details": {
                        "provided_key_length": len(api_key),
                        "match_attempts": [match_type]
                    }
                }), 401
                
        except Exception as e:
            logger.error(f"Supabase query error: {str(e)}")
            return jsonify({
                "status": "error",
                "error": "Database query failed",
                "details": str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"Debug validation error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
