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
import time
import traceback

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
    try:
        # Verify database connection
        services["classification"].supabase.table("webhook_results").select("count").limit(1).execute()
        
        return jsonify({
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "server": "txclassify"
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/webhook-health', methods=['GET'])
def webhook_health():
    """Health check endpoint specifically for webhooks"""
    try:
        return jsonify({
            "status": "ready",
            "service": "txclassify-webhook",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Webhook health check failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

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
        
        # Validate data quality
        if len(df) == 0:
            return jsonify({"error": "No valid transactions after cleaning"}), 400
            
        if len(df) < 10:  # Minimum required for meaningful training
            return jsonify({"error": "At least 10 valid transactions required for training"}), 400
            
        # Initialize classification service
        classifier = ClassificationService(
            supabase_url=os.environ.get("NEXT_PUBLIC_SUPABASE_URL"),
            supabase_key=os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY"),
            backend_api=request.host_url.rstrip('/')
        )
        
        # Get embeddings from Replicate
        sheet_id = f"sheet_{user_id}"
        
        # Split data into smaller chunks if too large
        CHUNK_SIZE = 1000  # Process 1000 transactions at a time
        narratives = df['Narrative'].tolist()
        categories = df['Category'].tolist()
        total_chunks = (len(narratives) + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        if total_chunks > 1:
            logger.info(f"Splitting {len(narratives)} transactions into {total_chunks} chunks")
            
            # Process first chunk
            chunk_start = 0
            chunk_end = min(CHUNK_SIZE, len(narratives))
            chunk_narratives = narratives[chunk_start:chunk_end]
            chunk_categories = categories[chunk_start:chunk_end]
            
            # Store chunk information
            chunk_info = {
                'total_chunks': total_chunks,
                'current_chunk': 1,
                'remaining_narratives': narratives[chunk_end:],
                'remaining_categories': categories[chunk_end:]
            }
            
            # Store chunk info in Supabase
            try:
                classifier.supabase.table("training_chunks").upsert({
                    "sheet_id": sheet_id,
                    "chunk_info": chunk_info,
                    "created_at": datetime.now().isoformat()
                }).execute()
            except Exception as e:
                logger.warning(f"Failed to store chunk info: {e}")
            
            # Start prediction for first chunk
            prediction = classifier.run_prediction(
                "training",
                sheet_id,
                user_id,
                chunk_narratives,
                webhook_params={
                    'descriptions': chunk_narratives,
                    'categories': chunk_categories,
                    'chunk_index': '1',
                    'total_chunks': str(total_chunks)
                }
            )
        else:
            # Process all data at once if small enough
            prediction = classifier.run_prediction(
                "training",
                sheet_id,
                user_id,
                narratives,
                webhook_params={
                    'descriptions': narratives,
                    'categories': categories
                }
            )
        
        return jsonify({
            "status": "processing",
            "message": "Training started",
            "prediction_id": prediction.id,
            "transaction_count": len(df),
            "total_chunks": total_chunks if total_chunks > 1 else 1
        })
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        if 'user_id' in locals():
            update_process_status(f"Error: {str(e)}", "training", user_id)
        return jsonify({"error": str(e)}), 500

@app.route('/train/webhook', methods=['POST'])
def training_webhook():
    """Handle training webhook from Replicate."""
    temp_file_path = None
    start_time = datetime.now()
    logger.info(f"Received webhook at {start_time.isoformat()}")
    
    try:
        sheet_id = request.args.get('sheetId')
        user_id = request.args.get('userId')
        chunk_index = request.args.get('chunk_index')
        total_chunks = request.args.get('total_chunks')
        
        logger.info(f"Processing webhook for sheet {sheet_id} and user {user_id}")
        logger.info(f"Chunk info: {chunk_index}/{total_chunks}" if chunk_index and total_chunks else "Processing all data")
        
        if not sheet_id:
            raise ValueError("No sheet ID provided in webhook URL")
            
        # Parse the webhook data with logging
        data = request.get_json()
        if not data:
            logger.error("No data received in webhook")
            logger.error(f"Request headers: {dict(request.headers)}")
            logger.error(f"Request args: {dict(request.args)}")
            raise ValueError("No data received in webhook")
            
        logger.info(f"Webhook data size: {len(str(data))} bytes")
        
        if 'output' not in data:
            logger.error(f"Webhook data structure: {json.dumps(data, indent=2)}")
            raise ValueError("No output data in webhook response")
            
        # Get embeddings with validation and logging
        embeddings = []
        if isinstance(data['output'], list):
            embeddings = [item['embedding'] for item in data['output'] if 'embedding' in item]
            logger.info(f"Found {len(embeddings)} embeddings")
            if embeddings:
                logger.info(f"First embedding dimension: {len(embeddings[0])}")
            
            if not all(len(emb) == len(embeddings[0]) for emb in embeddings):
                raise ValueError("Inconsistent embedding dimensions")
        
        if not embeddings:
            logger.error(f"Output structure: {json.dumps(data['output'], indent=2)}")
            raise ValueError("No embeddings found in prediction output")
            
        embeddings_array = np.array(embeddings)
        logger.info(f"Processed embeddings shape: {embeddings_array.shape}")
        
        # Initialize classification service for potential next chunk
        classifier = ClassificationService(
            supabase_url=os.environ.get("NEXT_PUBLIC_SUPABASE_URL"),
            supabase_key=os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY"),
            backend_api=request.host_url.rstrip('/')
        )
        
        # Check if we need to process more chunks
        if chunk_index and total_chunks and int(chunk_index) < int(total_chunks):
            try:
                # Get chunk info from Supabase
                response = classifier.supabase.table("training_chunks").select("*").eq("sheet_id", sheet_id).execute()
                if response.data:
                    chunk_info = response.data[0].get("chunk_info", {})
                    current_chunk = int(chunk_index)
                    
                    # Get next chunk of data
                    CHUNK_SIZE = 1000
                    remaining_narratives = chunk_info.get('remaining_narratives', [])
                    remaining_categories = chunk_info.get('remaining_categories', [])
                    
                    chunk_start = 0
                    chunk_end = min(CHUNK_SIZE, len(remaining_narratives))
                    chunk_narratives = remaining_narratives[chunk_start:chunk_end]
                    chunk_categories = remaining_categories[chunk_start:chunk_end]
                    
                    # Update chunk info
                    new_chunk_info = {
                        'total_chunks': int(total_chunks),
                        'current_chunk': current_chunk + 1,
                        'remaining_narratives': remaining_narratives[chunk_end:],
                        'remaining_categories': remaining_categories[chunk_end:]
                    }
                    
                    # Store updated chunk info
                    classifier.supabase.table("training_chunks").upsert({
                        "sheet_id": sheet_id,
                        "chunk_info": new_chunk_info,
                        "created_at": datetime.now().isoformat()
                    }).execute()
                    
                    # Start next chunk
                    logger.info(f"Starting chunk {current_chunk + 1}/{total_chunks}")
                    prediction = classifier.run_prediction(
                        "training",
                        sheet_id,
                        user_id,
                        chunk_narratives,
                        webhook_params={
                            'descriptions': chunk_narratives,
                            'categories': chunk_categories,
                            'chunk_index': str(current_chunk + 1),
                            'total_chunks': total_chunks
                        }
                    )
                    
                    logger.info(f"Next chunk prediction started: {prediction.id}")
                    
            except Exception as e:
                logger.error(f"Error processing next chunk: {e}")
                logger.error(traceback.format_exc())
                raise
        
        # Process current chunk results
        # Get descriptions and categories with logging
        descriptions = request.args.get('descriptions', '').split(',')
        categories = request.args.get('categories', '').split(',')
        
        logger.info(f"Received {len(descriptions)} descriptions and {len(categories)} categories")
        
        if not descriptions or not categories:
            raise ValueError("Missing descriptions or categories in webhook params")
            
        if len(descriptions) != len(embeddings):
            raise ValueError(f"Mismatch between descriptions ({len(descriptions)}) and embeddings ({len(embeddings)})")
        
        # Create training data structure
        training_data = {
            'embeddings': embeddings_array,
            'descriptions': np.array(descriptions),
            'categories': np.array(categories),
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'embedding_dimensions': embeddings_array.shape[1],
                'sample_count': len(descriptions),
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'chunk_info': f"{chunk_index}/{total_chunks}" if chunk_index and total_chunks else "complete"
            }
        }
        
        logger.info(f"Training data prepared in {training_data['metadata']['processing_time']:.2f} seconds")
        
        # Save to NPZ file with logging
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as temp_file:
            np.savez_compressed(temp_file, **training_data)
            temp_file_path = temp_file.name
            
            file_size = os.path.getsize(temp_file_path)
            logger.info(f"Training data file size: {file_size / 1024 / 1024:.2f} MB")
            
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                raise ValueError("Training data file too large")
            
            # Upload to Supabase with retry and logging
            max_retries = 3
            retry_delay = 1
            upload_success = False
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Upload attempt {attempt + 1}/{max_retries}")
                    with open(temp_file_path, 'rb') as f:
                        classifier.supabase.storage.from_(classifier.bucket_name).upload(
                            f"{sheet_id}_training{'_chunk_' + chunk_index if chunk_index else ''}.npz",
                            f.read(),
                            file_options={"x-upsert": "true"}
                        )
                    upload_success = True
                    logger.info(f"Successfully uploaded training data on attempt {attempt + 1}")
                    break
                except Exception as e:
                    logger.warning(f"Upload attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (2 ** attempt))
            
            if not upload_success:
                raise ValueError(f"Failed to upload training data after {max_retries} attempts")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Total webhook processing time: {processing_time:.2f} seconds")
        
        # Only update completion status if this is the last chunk
        if not chunk_index or int(chunk_index) == int(total_chunks):
            update_process_status("completed", "training", user_id)
        else:
            update_process_status(f"Processing chunk {chunk_index}/{total_chunks}", "training", user_id)
        
        return jsonify({
            "status": "success",
            "message": "Training data processed successfully",
            "processing_time": processing_time,
            "sample_count": len(descriptions),
            "file_size_mb": file_size / 1024 / 1024,
            "chunk_info": f"{chunk_index}/{total_chunks}" if chunk_index and total_chunks else "complete"
        })
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing webhook: {error_msg}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        if user_id:
            update_process_status(f"Error: {error_msg}", "training", user_id)
        return jsonify({"error": error_msg}), 500
        
    finally:
        # Clean up temporary files
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info("Cleaned up temporary file")
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
        # First check if we have webhook results
        webhook_results = None
        try:
            response = services["classification"].supabase.table("webhook_results").select("*").eq("prediction_id", prediction_id).execute()
            if response.data:
                webhook_results = response.data[0].get("results", {})
                logger.info(f"Found webhook results for prediction {prediction_id}")
                return jsonify({
                    "status": "completed",
                    "result": webhook_results,
                    "completion_time": response.data[0].get("created_at")
                })
        except Exception as e:
            logger.warning(f"Failed to get webhook results: {e}")
        
        # If no webhook results, check Replicate status with retries
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                prediction = replicate.predictions.get(prediction_id)
                logger.info(f"Prediction status: {prediction.status}")
                
                # Handle timestamps safely
                created_at = prediction.created_at
                if isinstance(created_at, str):
                    created_at_str = created_at
                else:
                    created_at_str = created_at.isoformat()
                
                completed_at = getattr(prediction, 'completed_at', None)
                completed_at_str = completed_at.isoformat() if completed_at and not isinstance(completed_at, str) else completed_at
                
                # Calculate elapsed time safely
                try:
                    if isinstance(created_at, str):
                        from datetime import datetime
                        created_timestamp = datetime.fromisoformat(created_at.replace('Z', '+00:00')).timestamp()
                    else:
                        created_timestamp = created_at.timestamp()
                    elapsed_time = time.time() - created_timestamp
                except Exception as e:
                    logger.warning(f"Error calculating elapsed time: {e}")
                    elapsed_time = 0
                
                status_response = {
                    "prediction_id": prediction_id,
                    "status": prediction.status,
                    "created_at": created_at_str,
                    "elapsed_time": elapsed_time,
                    "logs": prediction.logs
                }
                
                if prediction.status == "succeeded":
                    status_response.update({
                        "status": "processing",
                        "message": "Webhook processing in progress...",
                        "prediction_completed_at": completed_at_str
                    })
                elif prediction.status == "failed":
                    status_response.update({
                        "status": "failed",
                        "error": prediction.error
                    })
                else:
                    status_response.update({
                        "message": f"Prediction in progress... ({prediction.status})"
                    })
                
                return jsonify(status_response)
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"Replicate API attempt {attempt + 1} failed: {e}")
                    time.sleep(1 * (2 ** attempt))
                else:
                    if webhook_results:
                        # If we can't get prediction status but have webhook results, we're done
                        return jsonify({
                            "status": "completed",
                            "result": webhook_results
                        })
                    else:
                        raise
            
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        return jsonify({
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
