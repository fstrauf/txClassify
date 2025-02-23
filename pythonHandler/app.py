from flask import Flask, request, jsonify, g
from services.classification_service import ClassificationService
from flask_cors import CORS
import pandas as pd
import logging
from dotenv import load_dotenv
import os
from supabase import create_client
import sys
import uuid
import traceback
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s - [%(process)d] - [%(levelname)s] - %(name)s - %(message)s',
    level=logging.INFO  # Set to INFO level
)
logger = logging.getLogger('app')  # Use a specific logger for the application

# Ensure all loggers are set to INFO
logging.getLogger('werkzeug').setLevel(logging.INFO)
logging.getLogger('gunicorn').setLevel(logging.INFO)

# Custom logging filter to add request ID
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(g, 'request_id', 'no_req_id')
        return True

logger.addFilter(RequestIdFilter())

# Initialize Flask app with logging
app = Flask(__name__)
CORS(app)

@app.before_request
def log_request_info():
    """Log details about each request."""
    logger.info(f"Incoming request: {request.method} {request.path}")

@app.after_request
def log_response_info(response):
    """Log details about each response."""
    logger.info(f"Response status: {response.status}")
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    # Log the full exception with traceback
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(f"Traceback: {''.join(traceback.format_tb(e.__traceback__))}")
    return jsonify({
        "error": "Internal server error",
        "message": str(e),
        "request_id": g.request_id
    }), 500

# Log startup information
logger.info("=== Application Starting ===")
logger.info(f"Environment: {os.environ.get('FLASK_ENV', 'production')}")

# Initialize classification service with environment variables
supabase_url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
supabase_key = os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
if not all([supabase_url, supabase_key]):
    logger.error("Missing required environment variables for Supabase")
    raise ValueError("Missing required environment variables: NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY")

logger.info(f"Supabase URL configured: {supabase_url}")
logger.info(f"Supabase key length: {len(supabase_key)}")

# Initialize Supabase client with logging
try:
    logger.info("=== Initializing Supabase ===")
    supabase = create_client(supabase_url, supabase_key)
    logger.info("Supabase client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Supabase: {str(e)}")
    raise

@app.route('/')
def home():
    """Home endpoint"""
    logger.info("Home endpoint accessed")
    return jsonify({
        "status": "online",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/classify",
            "/train",
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

@app.route('/classify', methods=['POST'])
def classify_transactions():
    """Classify new transactions."""
    try:
        # Log all incoming request details
        logger.info("=== Incoming Classification Request ===")
        
        # Get API key from headers
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            logger.error("No API key provided in request headers")
            return jsonify({"error": "API key is required"}), 401

        # Validate API key and get user ID
        try:
            user_id = validate_api_key(api_key)
            logger.info(f"API key validated successfully for user: {user_id}")
        except Exception as e:
            logger.error(f"API key validation failed: {str(e)}")
            return jsonify({"error": str(e)}), 401

        # Get and validate request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract required parameters
        sheet_id = data.get("spreadsheetId")
        if not sheet_id:
            return jsonify({"error": "Missing required parameter: spreadsheetId"}), 400

        # Get transactions data
        transactions = data.get('transactions')
        if not transactions:
            return jsonify({"error": "Missing transactions data"}), 400
        if not isinstance(transactions, list):
            return jsonify({"error": "Invalid transactions format - expected array"}), 400
        if len(transactions) == 0:
            return jsonify({"error": "Empty transactions array"}), 400

        # Convert transactions to DataFrame
        try:
            df = pd.DataFrame(transactions)
            if 'description' in df.columns and 'Narrative' not in df.columns:
                df['Narrative'] = df['description']
            
            # Validate narratives
            if 'Narrative' not in df.columns:
                return jsonify({"error": "Missing 'Narrative' or 'description' column in transactions"}), 400
            
            df['Narrative'] = df['Narrative'].astype(str).str.strip()
            if df['Narrative'].empty or df['Narrative'].isna().any():
                return jsonify({"error": "Invalid or empty narratives found"}), 400
        except Exception as e:
            logger.error(f"Error processing transactions data: {str(e)}")
            return jsonify({"error": "Invalid transaction data format"}), 400

        # Initialize classification service
        try:
            classifier = ClassificationService(
                supabase_url=supabase_url,
                supabase_key=supabase_key,
                backend_api=request.host_url.rstrip('/')
            )
        except Exception as e:
            logger.error(f"Error initializing classification service: {str(e)}")
            return jsonify({"error": "Service initialization failed"}), 500

        # Check if training data exists
        try:
            classifier.fetch_embeddings("txclassify", f"{sheet_id}_index.npy")
        except Exception as e:
            logger.error(f"Error checking training data: {str(e)}")
            return jsonify({"error": "No training data found. Please train the model first."}), 400

        # Run classification
        try:
            prediction = classifier.classify(df, sheet_id, user_id)
            if not prediction or not prediction.id:
                raise ValueError("Invalid prediction response")

            logger.info(f"Successfully started classification for {len(df)} transactions. Prediction ID: {prediction.id}")
            
            return jsonify({
                "status": "processing",
                "prediction_id": prediction.id,
                "transaction_count": len(df)
            })

        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            return jsonify({"error": f"Classification failed: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Unexpected error in classify_transactions: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Endpoint to train the model with new data"""
    try:
        # Get API key from headers
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({"error": "API key is required"}), 401

        # Validate API key and get user ID
        try:
            user_id = validate_api_key(api_key)
        except Exception as e:
            return jsonify({"error": "Invalid API key"}), 401

        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request data"}), 400
            
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid request format - expected JSON object"}), 400
            
        if 'transactions' not in data:
            return jsonify({"error": "Missing transactions data"}), 400
            
        # Support both parameter names for backward compatibility
        sheet_id = data.get('spreadsheetId') or data.get('expenseSheetId')
        if not sheet_id or not isinstance(sheet_id, str) or len(sheet_id.strip()) == 0:
            return jsonify({"error": "Invalid or missing spreadsheetId"}), 400
            
        # Create or update user configuration
        try:
            # Check if user config exists
            response = supabase.table("account").select("*").eq("userId", user_id).execute()
            
            if not response.data:
                # Create new user config
                default_config = {
                    "userId": user_id,
                    "columnOrderCategorisation": {
                        "categoryColumn": "E",
                        "descriptionColumn": "C"
                    },
                    "categorisationRange": "A:Z"
                }
                supabase.table("account").insert(default_config).execute()
                logger.info(f"Created new user configuration for {user_id}")
            else:
                config = response.data[0]
                # Ensure required fields exist
                if not config.get("columnOrderCategorisation"):
                    config["columnOrderCategorisation"] = {
                        "categoryColumn": "E",
                        "descriptionColumn": "C"
                    }
                if not config.get("categorisationRange"):
                    config["categorisationRange"] = "A:Z"
                    
                # Update the configuration if needed
                supabase.table("account").update(config).eq("userId", user_id).execute()
                logger.info(f"Updated configuration for user {user_id}")
                
        except Exception as e:
            logger.warning(f"Error managing user configuration: {str(e)}")
            # Continue with training even if config creation fails
            
        # Convert transactions to DataFrame with error handling
        try:
            df = pd.DataFrame(data['transactions'])
        except Exception as e:
            logger.error(f"Error converting transactions to DataFrame: {str(e)}")
            return jsonify({"error": "Invalid transaction data format"}), 400
            
        # Validate required columns
        required_columns = ['Narrative', 'Category']
        if not all(col in df.columns for col in required_columns):
            return jsonify({
                "error": f"Missing required columns: {required_columns}"
            }), 400
            
        # Validate data quality
        df['Narrative'] = df['Narrative'].astype(str).str.strip()
        df['Category'] = df['Category'].astype(str).str.strip()
        
        if df['Narrative'].empty or df['Narrative'].isna().any():
            return jsonify({"error": "Invalid or empty narratives found"}), 400
            
        if df['Category'].empty or df['Category'].isna().any():
            return jsonify({"error": "Invalid or empty categories found"}), 400
            
        # Initialize classification service
        try:
            classifier = ClassificationService(
                supabase_url=supabase_url,
                supabase_key=supabase_key,
                backend_api=request.host_url.rstrip('/')
            )
        except Exception as e:
            logger.error(f"Error initializing classification service: {str(e)}")
            return jsonify({"error": "Service initialization failed"}), 500
            
        # Train model
        try:
            training_response = classifier.train(df, sheet_id, user_id)
            if not training_response or not training_response.id:
                raise ValueError("Invalid training response")
                
            # Log successful request
            logger.info(f"Successfully started training with {len(df)} transactions. Prediction ID: {training_response.id}")
            
            return jsonify({
                "status": "processing",
                "prediction_id": training_response.id,
                "transaction_count": len(df)
            })
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return jsonify({"error": f"Training failed: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in train_model: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

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
                    config["categorisationTab"] = "new_dump"
                    
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
            "categorisationTab": "new_dump",
            "categorisationRange": "A:Z",
            "columnOrderCategorisation": {
                "categoryColumn": "E",
                "descriptionColumn": "C"
            }
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

@app.route("/classify/webhook", methods=["POST"])
def classify_webhook():
    """Handle classification webhook from Replicate."""
    try:
        data = request.get_json()
        sheet_id = request.args.get("spreadsheetId")
        user_id = request.args.get("userId")
        
        if not all([data, sheet_id, user_id]):
            error_msg = "Missing required parameters"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400

        # Get new embeddings
        new_embeddings = np.array([item["embedding"] for item in data["output"]])
        
        # Get trained embeddings and categories
        trained_embeddings = fetch_embeddings("txclassify", f"{sheet_id}.npy")
        trained_data = fetch_embeddings("txclassify", f"{sheet_id}_index.npy")
        
        # Calculate similarities
        similarities = cosine_similarity(new_embeddings, trained_embeddings)
        best_matches = similarities.argmax(axis=1)
        
        # Get predicted categories and confidence scores
        results = []
        for i, idx in enumerate(best_matches):
            category = str(trained_data[idx][1])  # Index 1 is the Category field
            similarity_score = float(similarities[i][idx])  # Get the similarity score
            results.append({
                "predicted_category": category,
                "similarity_score": similarity_score
            })
        
        # Store webhook result
        try:
            result = {
                "prediction_id": request.headers.get('X-Replicate-Prediction-Id'),
                "results": results,
                "user_id": user_id,
                "status": "success"
            }
            supabase.table("webhook_results").insert(result).execute()
            logger.info("Stored classification results")
        except Exception as e:
            logger.warning(f"Error storing webhook result: {e}")

        return jsonify({
            "status": "success",
            "results": results
        })

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in classify webhook: {error_msg}")
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port) 