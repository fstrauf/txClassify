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
        status_field = "trainingStatus" if mode == "training" else "categorisationStatus"
        supabase.table("account").update({
            status_field: status_text
        }).eq("userId", user_id).execute()
    except Exception as e:
        logger.error(f"Error updating process status: {e}")

def clean_text(text: str) -> str:
    """Clean transaction description text."""
    text = re.sub(
        r"[^\w\s]|https?://\S+|www\.\S+|https?:/\S+|[^\x00-\x7F]+|\d+|\b\w{1,2}\b|xx|Value Date|Card|AUS|USA|USD|PTY|LTD|Tap and Pay|TAP AND PAY",
        "",
        str(text).strip()
    )
    text = re.sub(r"\s+", " ", text)
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

        # Create webhook URL
        webhook = f"{BACKEND_API}/{mode}/webhook?sheetId={sheet_id}&userId={user_id}"
        
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
    try:
        data = request.get_json()
        sheet_id = request.args.get("sheetId")
        user_id = request.args.get("userId")
        
        if not all([data, sheet_id, user_id]):
            return jsonify({"error": "Missing required parameters"}), 400

        # Process embeddings
        embeddings = np.array([item["embedding"] for item in data["output"]])
        
        # Store embeddings
        store_embeddings("txclassify", f"{sheet_id}.npy", embeddings)
        
        # Update status
        update_process_status("completed", "training", user_id)
        
        return jsonify({
            "status": "success",
            "message": "Training completed successfully"
        })

    except Exception as e:
        logger.error(f"Error in training webhook: {e}")
        if user_id:
            update_process_status(f"Error: {str(e)}", "training", user_id)
        return jsonify({"error": str(e)}), 500

@app.route("/classify", methods=["POST"])
def classify_transactions():
    """Classify new transactions."""
    try:
        data = request.get_json()
        sheet_id = data.get("expenseSheetId")
        user_id = data.get("userId")
        config = get_user_config(user_id)

        if not all([sheet_id, user_id, config]):
            error_msg = "Missing required parameters"
            update_sheet_log(sheet_id, "ERROR", error_msg)
            return jsonify({"error": error_msg}), 400

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
        sheet_range = f"{config['categorisationTab']}!{config['columnRange']}"
        
        # Fetch transaction data
        status_msg = "Fetching transactions"
        update_process_status(status_msg, "classify", user_id)
        update_sheet_log(sheet_id, "INFO", status_msg)
        sheet_data = get_spreadsheet_data(sheet_id, sheet_range)
        
        if not sheet_data or len(sheet_data) < 2:
            error_msg = "No transactions found to classify"
            update_sheet_log(sheet_id, "ERROR", error_msg)
            return jsonify({"error": error_msg}), 400
        
        # Process data
        df = pd.DataFrame(sheet_data[1:], columns=sheet_data[0])
        if 'description' not in df.columns:
            error_msg = "Missing required column: description"
            update_sheet_log(sheet_id, "ERROR", error_msg)
            return jsonify({"error": error_msg}), 400

        df["description"] = df["description"].apply(clean_text)
        
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
        update_process_status(f"Error: {error_msg}", "classify", user_id)
        if sheet_id:
            update_sheet_log(sheet_id, "ERROR", f"Classification failed: {error_msg}")
        return jsonify({"error": error_msg}), 500

@app.route("/classify/webhook", methods=["POST"])
def classify_webhook():
    """Handle classification webhook from Replicate."""
    try:
        data = request.get_json()
        sheet_id = request.args.get("sheetId")
        user_id = request.args.get("userId")
        config = get_user_config(user_id)

        if not all([data, sheet_id, user_id, config]):
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
        
        # Get predicted categories
        categories = [trained_data[i]["category"] for i in best_matches]
        
        # Update sheet with predictions
        status_msg = f"Writing {len(categories)} predictions to sheet"
        update_sheet_log(sheet_id, "INFO", status_msg)
        
        service = build("sheets", "v4", credentials=Credentials.from_service_account_info(
            json.loads(os.environ.get("GOOGLE_SERVICE_ACCOUNT")),
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        ))
        
        # Write categories back to sheet
        category_range = f"{config['categorisationTab']}!{config['categoryColumn']}"
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
