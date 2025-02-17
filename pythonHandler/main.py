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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Supabase client
supabase: Client = create_client(
    os.environ.get("NEXT_PUBLIC_SUPABASE_URL"),
    os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
)

BACKEND_API = os.environ.get("BACKEND_API")

def get_user_config(user_id: str) -> dict:
    """Get user configuration from Supabase."""
    try:
        response = supabase.table("account").select("*").eq("userId", user_id).execute()
        return response.data[0] if response.data else {}
    except Exception as e:
        logger.error(f"Error fetching user config: {e}")
        return {}

def update_process_status(status_text: str, mode: str, user_id: str) -> None:
    """Update process status in Supabase."""
    try:
        # First check if account exists
        response = supabase.table("account").select("*").eq("userId", user_id).execute()
        
        status_field = "trainingStatus" if mode == "training" else "categorisationStatus"
        
        if not response.data:
            # Create new account with status
            default_config = {
                "userId": user_id,
                "categorisationTab": "new_dump",
                "categorisationRange": "A:Z",
                "columnOrderCategorisation": {"categoryColumn": "E", "descriptionColumn": "C"},
                status_field: status_text
            }
            supabase.table("account").insert(default_config).execute()
            logger.info(f"Created new account for user {user_id} with status: {status_text}")
        else:
            # Update existing account
            supabase.table("account").update({
                status_field: status_text
            }).eq("userId", user_id).execute()
            logger.info(f"Updated status for user {user_id}: {status_text}")
            
    except Exception as e:
        logger.error(f"Error updating process status: {e}")
        # Log the full error details for debugging
        logger.error(f"Full error details: {str(e)}")
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
        google_service_account = json.loads(os.environ.get("GOOGLE_SERVICE_ACCOUNT"))
        creds = Credentials.from_service_account_info(
            google_service_account,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        service = build("sheets", "v4", credentials=creds)
        result = service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range=range_name
        ).execute()
        return result.get("values", [])
    except Exception as e:
        logger.error(f"Error fetching spreadsheet data: {e}")
        raise

def run_prediction(mode: str, sheet_id: str, user_id: str, descriptions: list) -> dict:
    """Run prediction using Replicate API."""
    try:
        model = replicate.models.get("replicate/all-mpnet-base-v2")
        version = model.versions.get(
            "b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305"
        )

        # Create webhook URL with consistent parameter names
        webhook = f"{BACKEND_API}/{mode}/webhook?spreadsheetId={sheet_id}&userId={user_id}"
        
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

@app.route("/train", methods=["POST"])
def train_model():
    """Train the model with new data."""
    try:
        data = request.get_json()
        
        # Extract data from request
        transactions = data.get("transactions")
        user_id = data.get("userId")
        
        if not transactions or not user_id:
            return jsonify({"error": "Missing required parameters: transactions and userId"}), 400
            
        # Get sheet ID from first transaction
        sheet_id = data.get("expenseSheetId")
        if not sheet_id:
            return jsonify({"error": "Missing expenseSheetId"}), 400

        # Process transactions
        df = pd.DataFrame(transactions)
        
        # Validate required columns
        if 'Narrative' not in df.columns or 'Category' not in df.columns:
            return jsonify({"error": "Missing required columns: Narrative and Category"}), 400

        # Clean descriptions
        df["description"] = df["Narrative"].apply(clean_text)
        df = df.drop_duplicates(subset=["description"])
        
        if len(df) < 10:  # Minimum required for meaningful training
            return jsonify({"error": "At least 10 valid transactions required for training"}), 400
        
        # Store training data index
        df["item_id"] = range(len(df))
        store_embeddings("txclassify", f"{sheet_id}_index.npy", df[["item_id", "Category", "description"]].to_records())
        
        # Run prediction
        prediction = run_prediction("train", sheet_id, user_id, df["description"].tolist())
        
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
                # Create new account
                default_config = {
                    "userId": user_id,
                    "categorisationTab": "new_dump",
                    "categorisationRange": "A:Z",
                    "columnOrderCategorisation": {"categoryColumn": "E", "descriptionColumn": "C"},
                    "trainingStatus": "processing",
                    "expenseSheetId": sheet_id
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
            logger.info(f"Successfully processed embeddings with shape: {embeddings.shape}")
            
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
                    "embeddings_shape": embeddings.shape
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
    """Classify new transactions."""
    try:
        data = request.get_json()
        
        # Extract and validate required parameters
        user_id = data.get("userId")
        sheet_id = data.get("spreadsheetId")
        
        if not user_id:
            error_msg = "Missing required parameter: userId"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
            
        if not sheet_id:
            error_msg = "Missing required parameter: spreadsheetId"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
            
        # Get user configuration with error handling
        try:
            config = get_user_config(user_id)
            if not config:
                error_msg = "User configuration not found"
                logger.error(f"{error_msg} for userId: {user_id}")
                return jsonify({"error": error_msg}), 400
                
            # Validate required configuration fields
            if not config.get("categorisationTab"):
                error_msg = "Missing categorisationTab in user configuration"
                logger.error(error_msg)
                return jsonify({"error": error_msg}), 400
                
            if not config.get("categorisationRange"):
                error_msg = "Missing categorisationRange in user configuration"
                logger.error(error_msg)
                return jsonify({"error": error_msg}), 400
        except Exception as e:
            error_msg = f"Error getting user configuration: {str(e)}"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400

        # Log successful parameter validation
        logger.info(f"Parameters validated - userId: {user_id}, spreadsheetId: {sheet_id}")

        # Verify training data exists
        try:
            trained_data = fetch_embeddings("txclassify", f"{sheet_id}_index.npy")
            if len(trained_data) == 0:
                error_msg = "No training data found. Please train the model first."
                update_sheet_log(sheet_id, "ERROR", error_msg)
                return jsonify({"error": error_msg}), 400
        except Exception as e:
            error_msg = "No training data found. Please train the model first."
            update_sheet_log(sheet_id, "ERROR", error_msg)
            return jsonify({"error": error_msg}), 400

        # Get sheet configuration
        sheet_range = f"{config['categorisationTab']}!{config['categorisationRange']}"
        
        # Fetch transaction data
        status_msg = "Fetching transactions"
        update_process_status(status_msg, "classify", user_id)
        update_sheet_log(sheet_id, "INFO", status_msg)
        sheet_data = get_spreadsheet_data(sheet_id, sheet_range)
        
        if not sheet_data:
            error_msg = "No transactions found to classify"
            update_sheet_log(sheet_id, "ERROR", error_msg)
            return jsonify({"error": error_msg}), 400
        
        # Get column configuration
        column_config = config.get("columnOrderCategorisation", {})
        description_column = column_config.get("descriptionColumn", "C")
        
        # Convert column letter to index (0-based)
        description_index = ord(description_column.upper()) - ord('A')
        
        # Process data without assuming headers
        descriptions = []
        for row in sheet_data:
            if len(row) > description_index:
                description = row[description_index]
                if description and str(description).strip():
                    descriptions.append({"description": str(description).strip()})
        
        if not descriptions:
            error_msg = "No valid descriptions found in the specified column"
            update_sheet_log(sheet_id, "ERROR", error_msg)
            return jsonify({"error": error_msg}), 400

        # Create DataFrame with descriptions
        df = pd.DataFrame(descriptions)
        
        # Run prediction
        status_msg = f"Getting embeddings for {len(df)} transactions"
        update_process_status(status_msg, "classify", user_id)
        update_sheet_log(sheet_id, "INFO", status_msg)
        prediction = run_prediction("classify", sheet_id, user_id, df["description"].tolist())
        
        update_sheet_log(sheet_id, "INFO", "Classification started", f"Prediction ID: {prediction.id}")
        return jsonify({
            "status": "processing",
            "prediction_id": prediction.id
        })

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in classify_transactions: {error_msg}")
        if user_id:
            update_process_status(f"Error: {error_msg}", "classify", user_id)
        if sheet_id:
            update_sheet_log(sheet_id, "ERROR", f"Classification failed: {error_msg}")
        return jsonify({"error": error_msg}), 500

@app.route("/classify/webhook", methods=["POST"])
def classify_webhook():
    """Handle classification webhook from Replicate."""
    try:
        data = request.get_json()
        sheet_id = request.args.get("spreadsheetId")
        user_id = request.args.get("userId")
        
        # Get user configuration
        response = supabase.table("account").select("*").eq("userId", user_id).execute()
        if not response.data:
            error_msg = "User configuration not found"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
            
        config = response.data[0]
        column_config = config.get("columnOrderCategorisation", {})
        if not isinstance(column_config, dict):
            column_config = {}
            
        category_column = column_config.get("categoryColumn", "E")
        
        if not all([data, sheet_id, user_id]):
            error_msg = "Missing required parameters"
            if sheet_id:
                update_sheet_log(sheet_id, "ERROR", error_msg)
            return jsonify({"error": error_msg}), 400

        # Get new embeddings
        new_embeddings = np.array([item["embedding"] for item in data["output"]])
        
        # Get trained embeddings and categories
        trained_embeddings = fetch_embeddings("txclassify", f"{sheet_id}.npy")
        trained_data = fetch_embeddings("txclassify", f"{sheet_id}_index.npy")
        
        # Calculate similarities
        similarities = cosine_similarity(new_embeddings, trained_embeddings)
        best_matches = similarities.argmax(axis=1)
        
        # Get predicted categories from structured array
        categories = []
        for idx in best_matches:
            # Get the record at index idx and convert to dictionary
            record = trained_data[idx].item()
            # Access the Category field from the dictionary
            category = str(record['Category'])
            categories.append(category)
        
        # Update sheet with predictions
        status_msg = f"Writing {len(categories)} predictions to sheet"
        update_sheet_log(sheet_id, "INFO", status_msg)
        
        service = build("sheets", "v4", credentials=Credentials.from_service_account_info(
            json.loads(os.environ.get("GOOGLE_SERVICE_ACCOUNT")),
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        ))
        
        # Write categories back to sheet starting from row 1
        category_range = f"{config['categorisationTab']}!{category_column}1:{category_column}{len(categories)}"
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=category_range,
            valueInputOption="USER_ENTERED",
            body={"values": [[cat] for cat in categories]}
        ).execute()
        
        status_msg = "Classification completed successfully"
        update_process_status("completed", "classify", user_id)
        update_sheet_log(sheet_id, "SUCCESS", status_msg)
        return jsonify({"status": "success"})

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
