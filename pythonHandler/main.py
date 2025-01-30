import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from typing import Dict, Any
import pandas as pd
import requests

from services.transaction_service import TransactionService
from services.classification_service import ClassificationService
from services.spreadsheet_service import SpreadsheetService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize services
def init_services():
    supabase_url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
    supabase_key = os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    backend_api = os.environ.get("BACKEND_API")
    google_service_account = json.loads(os.environ.get("GOOGLE_SERVICE_ACCOUNT"))

    return {
        "transaction": TransactionService(),
        "classification": ClassificationService(supabase_url, supabase_key, backend_api),
        "spreadsheet": SpreadsheetService(google_service_account)
    }

services = init_services()

def get_user_config(user_id: str) -> Dict[str, Any]:
    """Get user configuration from Supabase."""
    try:
        response = services["classification"].supabase.table("account").select("*").eq("userId", user_id).execute()
        return response.data[0] if response.data else {}
    except Exception as e:
        logger.error(f"Error fetching user config: {e}")
        return {}

def update_process_status(status_text: str, mode: str, user_id: str) -> None:
    """Update process status in Supabase."""
    try:
        status_field = "trainingStatus" if mode == "training" else "categorisationStatus"
        services["classification"].supabase.table("account").update({
            status_field: status_text
        }).eq("userId", user_id).execute()
    except Exception as e:
        logger.error(f"Error updating process status: {e}")

@app.route("/process_transactions", methods=["POST"])
def process_transactions():
    """Unified endpoint to handle transaction processing and classification."""
    try:
        data = request.get_json()
        user_id = data.get("userId")
        files_data = data.get("files")
        sheet_id = data.get("sheetId")
        
        if not all([user_id, files_data, sheet_id]):
            return jsonify({"error": "Missing required parameters"}), 400

        # Process transactions
        update_process_status("Processing transactions", "classify", user_id)
        
        # Handle Google Sheets data
        processed_files_data = []
        for file_info in files_data:
            if file_info.get("type") == "google_sheets":
                # Get sheet data
                config = file_info["config"]
                sheet_range = f"{config['tab']}!{config['range']}"
                sheet_data = services["spreadsheet"].get_sheet_data(sheet_id, sheet_range)
                
                logger.info(f"Sheet data received: {len(sheet_data) if sheet_data else 0} rows")
                if sheet_data and len(sheet_data) > 0:
                    logger.info(f"Headers: {sheet_data[0]}")
                
                # Convert to DataFrame
                if sheet_data and len(sheet_data) > 1:  # Ensure we have headers and data
                    headers = sheet_data[0]
                    data = sheet_data[1:]
                    df = pd.DataFrame(data, columns=headers)
                    logger.info(f"DataFrame columns: {df.columns.tolist()}")
                    logger.info(f"Column mapping config: {config['column_mapping']}")
                    
                    processed_files_data.append({
                        "config": config,
                        "content": df
                    })
                else:
                    logger.error("No data or headers found in sheet")
            else:
                processed_files_data.append(file_info)
        
        transactions_df = services["transaction"].process_multiple_files(processed_files_data)
        
        # Prepare for classification
        clean_transactions = services["transaction"].prepare_for_classification(transactions_df)
        
        # Run classification
        update_process_status("Running classification", "classify", user_id)
        prediction = services["classification"].run_prediction(
            "classify",
            sheet_id,
            user_id,
            "https://www.expensesorted.com/api/finishedTrainingHook",
            clean_transactions["description"].tolist()
        )
        
        return jsonify({
            "message": "Processing started",
            "transaction_count": len(transactions_df)
        })

    except Exception as e:
        logger.error(f"Error in process_transactions: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/classify", methods=["POST"])
def handle_classify_webhook():
    """Handle classification webhook from Replicate."""
    try:
        data = request.get_json()
        webhook_url = data.get("webhook")
        
        # Extract parameters from webhook URL
        params = dict(param.split('=') for param in webhook_url.split('?')[1].split('&'))
        sheet_id = params.get("sheetId")
        user_id = params.get("userId")
        run_key = params.get("runKey")
        
        # Get user configuration
        config = get_user_config(user_id)
        
        # Process embeddings
        update_process_status("Processing classification results", "classify", user_id)
        new_embeddings = services["classification"].process_embeddings(data)
        trained_embeddings = services["classification"].fetch_embeddings(f"{sheet_id}.npy")
        trained_index = services["classification"].fetch_embeddings(f"{sheet_id}_index.npy")
        
        # Get sheet configuration
        first_col, last_col, source_cols = services["spreadsheet"].get_column_range(
            config.get("columnOrderCategorisation")
        )
        _, _, target_cols = services["spreadsheet"].get_column_range(
            config.get("columnOrderTraining")
        )
        
        # Get current transactions
        sheet_range = f"{config['categorisationTab']}!{first_col}:{last_col}"
        transactions = services["spreadsheet"].get_sheet_data(sheet_id, sheet_range)
        
        # Classify and prepare data
        classified_data = services["classification"].classify_expenses(
            pd.DataFrame(transactions, columns=source_cols),
            trained_embeddings,
            new_embeddings
        )
        
        # Map indices to categories using trained_index
        df_output = pd.DataFrame.from_dict(classified_data)
        trained_categories_df = pd.DataFrame(trained_index)
        
        # Merge to get categories
        combined_df = df_output.merge(
            trained_categories_df,
            left_on="categoryIndex",
            right_on="item_id",
            how="left"
        )
        
        # Drop unnecessary columns and keep only what we need
        combined_df.drop(columns=["item_id", "categoryIndex"], inplace=True)
        
        # Prepare and append to sheet
        final_df = services["spreadsheet"].prepare_sheet_data(
            transactions,
            combined_df,
            source_cols,
            target_cols
        )
        
        # Update sheet
        update_process_status("Updating spreadsheet", "classify", user_id)
        services["spreadsheet"].append_to_sheet(
            sheet_id,
            f"{config['trainingTab']}!{first_col}:{last_col}",
            final_df.values.tolist()
        )
        
        update_process_status("completed", "classify", user_id)
        return "", 200

    except Exception as e:
        logger.error(f"Error in classify webhook: {str(e)}")
        update_process_status(f"Error: {str(e)}", "classify", user_id)
        return jsonify({"error": str(e)}), 500

@app.route("/runTraining", methods=["POST"])
def run_training():
    """Handle training data preparation and model training."""
    try:
        data = request.get_json()
        sheet_id = data.get("expenseSheetId")
        user_id = data.get("userId")
        config = get_user_config(user_id)
        
        # Get sheet configuration
        first_col, last_col, column_types = services["spreadsheet"].get_column_range(
            config.get("columnOrderTraining")
        )
        
        # Fetch training data
        update_process_status("Fetching training data", "training", user_id)
        sheet_range = f"{config['trainingTab']}!{first_col}:{last_col}"
        training_data = services["spreadsheet"].get_sheet_data(sheet_id, sheet_range)
        
        # Add debug logging
        logger.info(f"Column types from config: {column_types}")
        logger.info(f"Training data columns: {training_data[0] if training_data else 'No data'}")
        
        # Create column mapping
        column_mapping = {
            'Source': 'source',
            'Date': 'date',
            'Narrative': 'description',
            'Amount': 'amount',
            'Categories': 'category'
        }
        
        # Map the columns and create DataFrame
        header = training_data[0]
        data = training_data[1:]
        
        # Create initial DataFrame with original columns
        df_training = pd.DataFrame(data, columns=header)
        
        # Rename columns according to mapping
        df_training = df_training.rename(columns=column_mapping)
        
        # Add missing columns with default values
        if 'currency' not in df_training.columns:
            df_training['currency'] = 'AUD'  # Default currency
            
        # Ensure all required columns are present
        for col in column_types:
            if col not in df_training.columns:
                df_training[col] = None
                
        # Reorder columns to match expected order
        df_training = df_training[column_types]
        
        # Continue with rest of processing
        df_training = df_training.drop_duplicates(subset=["description"])
        df_training["item_id"] = range(len(df_training))
        
        # Store training data index
        services["classification"].store_training_data(df_training, sheet_id)
        
        # Run training
        update_process_status("Training model", "training", user_id)
        services["classification"].run_prediction(
            "training",
            sheet_id,
            user_id,
            "https://www.expensesorted.com/api/finishedTrainingHook",
            df_training["description"].tolist()
        )
        
        return jsonify({"message": "Training started"})

    except Exception as e:
        logger.error(f"Error in run_training: {str(e)}")
        update_process_status(f"Error: {str(e)}", "training", user_id)
        return jsonify({"error": str(e)}), 500

@app.route("/training", methods=["POST"])
def handle_training_webhook():
    """Handle training webhook from Replicate."""
    try:
        data = request.get_json()
        webhook_url = data.get("webhook")
        
        # Extract parameters from webhook URL
        params = dict(param.split('=') for param in webhook_url.split('?')[1].split('&'))
        sheet_id = params.get("sheetId")
        user_id = params.get("userId")
        run_key = params.get("runKey")
        
        # Process embeddings
        update_process_status("Processing training results", "training", user_id)
        embeddings = services["classification"].process_embeddings(data)
        
        # Save embeddings
        services["classification"].save_embeddings(f"{sheet_id}.npy", embeddings)
        
        # Update status
        update_process_status("completed", "training", user_id)
        
        # Forward to frontend webhook if provided
        sheet_api = params.get("sheetApi")
        if sheet_api:
            requests.post(sheet_api, json={"status": "completed"})
        
        return "", 200

    except Exception as e:
        logger.error(f"Error in training webhook: {str(e)}")
        if user_id:
            update_process_status(f"Error: {str(e)}", "training", user_id)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=3001)
