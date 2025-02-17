from flask import Flask, request, jsonify
from services.classification_service import ClassificationService
from flask_cors import CORS
import pandas as pd
import logging
from dotenv import load_dotenv
import os
from supabase import create_client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize classification service with environment variables
supabase_url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
supabase_key = os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
if not all([supabase_url, supabase_key]):
    raise ValueError("Missing required environment variables: NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY")

app = Flask(__name__)
CORS(app)

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
def classify_transactions():
    """Classify new transactions."""
    try:
        data = request.get_json()
        logger.info("Received classification request")
        
        # Extract and validate required parameters
        user_id = data.get("userId")
        sheet_id = data.get("spreadsheetId")
        
        logger.info(f"Request parameters - userId: {user_id}, spreadsheetId: {sheet_id}")
        
        if not user_id:
            error_msg = "Missing required parameter: userId"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
            
        if not sheet_id:
            error_msg = "Missing required parameter: spreadsheetId"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400
            
        # Get sheet configuration from request
        sheet_name = data.get("sheetName", "new_dump")
        description_col = data.get("descriptionColumn", "C")
        category_col = data.get("categoryColumn", "E")
        start_row = data.get("startRow", "2")
        
        logger.info(f"Sheet configuration - name: {sheet_name}, description: {description_col}, category: {category_col}, start: {start_row}")
        
        # Get and validate required parameters
        if 'transactions' not in data:
            return jsonify({"error": "Missing transactions data"}), 400
            
        if not isinstance(data['transactions'], list):
            return jsonify({"error": "Invalid transactions format - expected array"}), 400
            
        if len(data['transactions']) == 0:
            return jsonify({"error": "Empty transactions array"}), 400
            
        # Convert transactions to DataFrame with error handling
        try:
            df = pd.DataFrame(data['transactions'])
        except Exception as e:
            logger.error(f"Error converting transactions to DataFrame: {str(e)}")
            return jsonify({"error": "Invalid transaction data format"}), 400
            
        if 'Narrative' not in df.columns and 'description' not in df.columns:
            return jsonify({"error": "Missing 'Narrative' or 'description' column in transactions"}), 400
            
        # Use description field if Narrative is not present
        if 'description' in df.columns and 'Narrative' not in df.columns:
            df['Narrative'] = df['description']
            
        # Validate narratives
        df['Narrative'] = df['Narrative'].astype(str).str.strip()
        if df['Narrative'].empty or df['Narrative'].isna().any():
            return jsonify({"error": "Invalid or empty narratives found"}), 400
            
        # Create or update user configuration
        try:
            supabase = create_client(supabase_url, supabase_key)
            
            # Check if user config exists
            response = supabase.table("account").select("*").eq("userId", user_id).execute()
            
            if not response.data:
                # Create new user config
                default_config = {
                    "userId": user_id,
                    "categorisationTab": sheet_name,
                    "columnRange": "A:Z",
                    "categoryColumn": category_col
                }
                supabase.table("account").insert(default_config).execute()
                logger.info(f"Created new user configuration for {user_id}")
            else:
                logger.info(f"Found existing user configuration for {user_id}")
                
        except Exception as e:
            logger.warning(f"Error managing user configuration: {str(e)}")
            # Continue with classification even if config management fails
            
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
            
        # Check if training data exists before classification
        try:
            classifier.fetch_embeddings("txclassify", f"{sheet_id}_index.npy")
        except Exception as e:
            logger.error(f"Error checking training data: {str(e)}")
            return jsonify({"error": "No training data found. Please train the model first."}), 400
        
        # Run classification with timeout handling
        try:
            prediction = classifier.classify(df, sheet_id, user_id)
            if not prediction or not prediction.id:
                raise ValueError("Invalid prediction response")
                
            # Log successful request
            logger.info(f"Successfully started classification for {len(df)} transactions. Prediction ID: {prediction.id}")
            
            # Return prediction ID for status tracking
            return jsonify({
                "status": "processing",
                "prediction_id": prediction.id,
                "transaction_count": len(df),
                "sheet_config": {
                    "sheetName": sheet_name,
                    "descriptionColumn": description_col,
                    "categoryColumn": category_col,
                    "startRow": start_row
                }
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
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request data"}), 400
            
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid request format - expected JSON object"}), 400
            
        if 'transactions' not in data:
            return jsonify({"error": "Missing transactions data"}), 400
            
        # Get and validate required parameters
        user_id = data.get('userId')
        if not user_id or not isinstance(user_id, str) or len(user_id.strip()) == 0:
            return jsonify({"error": "Invalid or missing userId"}), 400
            
        # Support both parameter names for backward compatibility
        sheet_id = data.get('spreadsheetId') or data.get('expenseSheetId')
        if not sheet_id or not isinstance(sheet_id, str) or len(sheet_id.strip()) == 0:
            return jsonify({"error": "Invalid or missing spreadsheetId"}), 400
            
        # Create or update user configuration
        try:
            supabase = create_client(supabase_url, supabase_key)
            
            # Check if user config exists
            response = supabase.table("account").select("*").eq("userId", user_id).execute()
            
            if not response.data:
                # Create new user config
                default_config = {
                    "userId": user_id,
                    "categorisationTab": "new_dump",
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

def get_user_config(user_id: str) -> dict:
    """Get user configuration from Supabase using API key-based user ID."""
    try:
        logger.info(f"Looking up user configuration for userId: {user_id}")
        
        # Initialize Supabase client
        supabase = create_client(
            supabase_url,
            supabase_key
        )
        
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port) 