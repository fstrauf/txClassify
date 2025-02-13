import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from typing import Dict, Any
import pandas as pd
import requests
from dotenv import load_dotenv
from functools import wraps
import replicate
import numpy as np
from datetime import datetime
import tempfile

from services.transaction_service import TransactionService
from services.classification_service import ClassificationService
from services.spreadsheet_service import SpreadsheetService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({"error": "No API key provided"}), 401
            
        # Use API key as user identifier
        request.user_id = api_key[:8]  # Use first 8 chars as user ID
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def root():
    """Root endpoint"""
    return jsonify({
        "message": "TX Classify API",
        "version": "1.0.0",
        "endpoints": ["/health", "/classify", "/train"]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0"
    })

@app.route('/classify', methods=['POST'])
@require_api_key
def classify_transactions():
    """Endpoint to classify transactions"""
    try:
        # Get request data
        data = request.get_json()
        if not data or 'transactions' not in data:
            return jsonify({
                "error": "Missing transactions data"
            }), 400
            
        # Convert transactions to DataFrame
        df = pd.DataFrame(data['transactions'])
        if 'Narrative' not in df.columns:
            return jsonify({
                "error": "Missing 'Narrative' column in transactions"
            }), 400
        
        # Initialize classification service
        classifier = ClassificationService(
            supabase_url=os.environ.get("NEXT_PUBLIC_SUPABASE_URL"),
            supabase_key=os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY"),
            backend_api=request.host_url.rstrip('/')
        )
        
        # Start classification (this will trigger a webhook callback)
        prediction = classifier.classify(df, f"sheet_{request.user_id}", request.user_id)
        
        # Return prediction ID and status
        return jsonify({
            "status": "processing",
            "message": "Classification started",
            "prediction_id": prediction.id
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/classify/webhook', methods=['POST'])
def classify_webhook():
    """Handle classification webhook from Replicate"""
    try:
        data = request.get_json()
        sheet_id = request.args.get('sheetId', 'sheet_default')
        user_id = request.args.get('userId', 'user_default')
        
        logger.info(f"Received webhook for sheet {sheet_id} and user {user_id}")
        
        # Initialize classification service
        classifier = ClassificationService(
            supabase_url=os.environ.get("NEXT_PUBLIC_SUPABASE_URL"),
            supabase_key=os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY"),
            backend_api=request.host_url.rstrip('/')
        )
        
        # Process the webhook response
        logger.info("Processing webhook response")
        
        # First verify we have training data
        training_data = classifier._load_training_data(sheet_id)
        if not training_data or len(training_data['descriptions']) == 0:
            raise ValueError(f"No training data found for sheet {sheet_id}. Please train the model first.")
            
        logger.info(f"Found {len(training_data['descriptions'])} training examples")
        
        # Process embeddings and classify
        results = classifier.process_webhook_response(data, sheet_id)
        logger.info(f"Successfully classified {len(results)} transactions")
        
        # Convert results to list of dictionaries
        results_list = results.to_dict('records')
        
        # Store results in Supabase for status endpoint
        try:
            services["classification"].supabase.table("webhook_results").upsert({
                "prediction_id": data.get("id"),
                "results": results_list,
                "created_at": datetime.now().isoformat()
            }).execute()
        except Exception as e:
            logger.warning(f"Failed to store webhook results: {e}")
        
        return jsonify(results_list)
        
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        logger.error(f"Request args: {request.args}")
        logger.error(f"Request data: {data if 'data' in locals() else 'Not available'}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route("/train", methods=['POST'])
def train_model():
    """Endpoint to train the model with new data"""
    try:
        # Get request data
        data = request.get_json()
        if not data or 'transactions' not in data:
            return jsonify({"error": "Missing transactions data"}), 400
            
        # Get user ID from request data
        user_id = data.get('userId')
        if not user_id:
            return jsonify({"error": "Missing userId in request"}), 400
            
        # Convert transactions to DataFrame and validate
        df = pd.DataFrame(data['transactions'])
        required_columns = ['Narrative', 'Category']
        if not all(col in df.columns for col in required_columns):
            return jsonify({"error": f"Missing required columns: {required_columns}"}), 400
        
        # Clean data
        df = df.dropna(subset=['Narrative', 'Category'])
        df = df.drop_duplicates(subset=['Narrative'])
        
        if len(df) == 0:
            return jsonify({"error": "No valid transactions after cleaning"}), 400
            
        # Initialize classification service
        classifier = ClassificationService(
            supabase_url=os.environ.get("NEXT_PUBLIC_SUPABASE_URL"),
            supabase_key=os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY"),
            backend_api=request.host_url.rstrip('/')
        )
        
        # Get embeddings from Replicate
        sheet_id = f"sheet_{user_id}"
        
        # Store categories in a single file
        training_data = {
            'descriptions': df['Narrative'].tolist(),
            'categories': df['Category'].tolist()
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            json.dump(training_data, temp_file, ensure_ascii=False)
            temp_file_path = temp_file.name
            
        try:
            # Upload to Supabase
            with open(temp_file_path, 'rb') as f:
                classifier.supabase.storage.from_(classifier.bucket_name).upload(
                    f"{sheet_id}_training.json",
                    f,
                    file_options={"x-upsert": "true"}
                )
                
            # Start prediction only after training data is stored
            prediction = classifier.run_prediction(
                "training",
                sheet_id,
                user_id,
                df['Narrative'].tolist()
            )
            
            return jsonify({
                "status": "processing",
                "message": "Training started",
                "prediction_id": prediction.id
            })
            
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        if 'user_id' in locals():
            update_process_status(f"Error: {str(e)}", "training", user_id)
        return jsonify({"error": str(e)}), 500

@app.route('/train/webhook', methods=['POST'])
def training_webhook():
    """Handle training webhook from Replicate."""
    temp_file_path = None
    try:
        # Get sheet ID from request args
        sheet_id = request.args.get('sheetId')
        user_id = request.args.get('userId')
        
        logger.info(f"Received training webhook for sheet {sheet_id} and user {user_id}")
        
        if not sheet_id:
            raise ValueError("No sheet ID provided in webhook URL")
            
        # Parse the webhook data
        data = request.get_json()
        if not data:
            raise ValueError("No data received in webhook")
            
        if 'output' not in data:
            logger.error(f"Webhook data structure: {json.dumps(data, indent=2)}")
            raise ValueError("No output data in webhook response")
            
        # Get embeddings from the prediction output
        embeddings = []
        if isinstance(data['output'], list):
            embeddings = [item['embedding'] for item in data['output'] if 'embedding' in item]
        
        if not embeddings:
            logger.error(f"Output structure: {json.dumps(data['output'], indent=2)}")
            raise ValueError("No embeddings found in prediction output")
            
        embeddings_array = np.array(embeddings)
        logger.info(f"Processed embeddings shape: {embeddings_array.shape}")
        
        # Initialize service
        classifier = ClassificationService(
            supabase_url=os.environ.get("NEXT_PUBLIC_SUPABASE_URL"),
            supabase_key=os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY"),
            backend_api=request.host_url.rstrip('/')
        )
        
        # Get training data from stored file
        try:
            # Download training data
            response = classifier.supabase.storage.from_(classifier.bucket_name).download(
                f"{sheet_id}_training.json"
            )
            
            # Handle response based on type
            if isinstance(response, str):
                training_data = json.loads(response)
            else:
                # If it's bytes, decode it first
                training_data = json.loads(response.decode('utf-8'))
                
            descriptions = training_data.get('descriptions', [])
            categories = training_data.get('categories', [])
                    
            if not descriptions or not categories:
                raise ValueError("No training data found in stored file")
                
            if len(descriptions) != len(embeddings):
                raise ValueError(f"Mismatch between descriptions ({len(descriptions)}) and embeddings ({len(embeddings)})")
                
            # Create training data structure
            training_data = {
                'embeddings': embeddings_array,
                'descriptions': np.array(descriptions),
                'categories': np.array(categories)
            }
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as temp_file:
                np.savez_compressed(temp_file, **training_data)
                temp_file_path = temp_file.name
            
            # Upload to Supabase
            with open(temp_file_path, 'rb') as f:
                classifier.supabase.storage.from_(classifier.bucket_name).upload(
                    f"{sheet_id}_training.npz",
                    f,
                    file_options={"x-upsert": "true"}
                )
                
            # Clean up training data file
            try:
                classifier.supabase.storage.from_(classifier.bucket_name).remove([
                    f"{sheet_id}_training.json"
                ])
            except Exception as e:
                logger.warning(f"Failed to clean up training data file: {e}")
            
            logger.info(f"Successfully stored training data with {len(descriptions)} examples")
            return jsonify({"status": "success", "message": "Training data processed successfully"})
            
        except Exception as e:
            logger.error(f"Error processing training data: {str(e)}")
            raise
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing webhook: {error_msg}")
        if user_id:
            update_process_status(f"Error: {error_msg}", "training", user_id)
        return jsonify({"error": error_msg}), 500
        
    finally:
        # Clean up any temporary files
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")

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
            'Category': 'category',
            'Currency': 'currency'
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
        
        # Run training
        update_process_status("Training model", "training", user_id)
        
        # Store training data temporarily and get reference key
        training_data = df_training.to_dict('records')
        training_key = services["classification"].store_temp_training_data(training_data, sheet_id)
        
        # Pass only the training key in webhook params
        webhook_params = {
            'training_key': training_key
        }
        
        services["classification"].run_prediction(
            "training",
            sheet_id,
            user_id,
            df_training["description"].tolist(),
            webhook_params=webhook_params
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

@app.route('/status/<prediction_id>', methods=['GET'])
@require_api_key
def check_status(prediction_id):
    """Check status of a prediction"""
    try:
        # Get prediction from Replicate
        prediction = replicate.predictions.get(prediction_id)
        
        # Get webhook results from memory or storage
        webhook_results = {}
        try:
            # Try to get webhook results from Supabase
            response = services["classification"].supabase.table("webhook_results").select("*").eq("prediction_id", prediction_id).execute()
            if response.data:
                webhook_results = response.data[0].get("results", {})
        except Exception as e:
            logger.warning(f"Failed to get webhook results: {e}")
        
        # Return appropriate status
        if prediction.status == "succeeded":
            if webhook_results:
                # If we have webhook results, return those
                return jsonify({
                    "status": "completed",
                    "result": webhook_results
                })
            else:
                # If no webhook results yet, still processing
                return jsonify({
                    "status": "processing",
                    "message": "Webhook processing in progress..."
                })
        elif prediction.status == "failed":
            return jsonify({
                "status": "failed",
                "error": prediction.error
            })
        else:
            return jsonify({
                "status": "processing",
                "message": f"Operation in progress... ({prediction.status})"
            })
            
    except Exception as e:
        logger.error(f"Error checking prediction status: {str(e)}")
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
