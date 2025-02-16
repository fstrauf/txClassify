from flask import Flask, request, jsonify
from services.classification_service import ClassificationService
from flask_cors import CORS
import pandas as pd
import logging
from dotenv import load_dotenv
import os

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
    """Endpoint to classify transactions"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request data"}), 400
            
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid request format - expected JSON object"}), 400
            
        if 'transactions' not in data:
            return jsonify({"error": "Missing transactions data"}), 400
            
        if not isinstance(data['transactions'], list):
            return jsonify({"error": "Invalid transactions format - expected array"}), 400
            
        if len(data['transactions']) == 0:
            return jsonify({"error": "Empty transactions array"}), 400
            
        # Get and validate required parameters
        user_id = data.get('userId')
        if not user_id or not isinstance(user_id, str) or len(user_id.strip()) == 0:
            return jsonify({"error": "Invalid or missing userId"}), 400
            
        # Support both parameter names for backward compatibility
        sheet_id = data.get('spreadsheetId') or data.get('expenseSheetId')
        if not sheet_id or not isinstance(sheet_id, str) or len(sheet_id.strip()) == 0:
            return jsonify({"error": "Invalid or missing spreadsheetId"}), 400
            
        # Convert transactions to DataFrame with error handling
        try:
            df = pd.DataFrame(data['transactions'])
        except Exception as e:
            logger.error(f"Error converting transactions to DataFrame: {str(e)}")
            return jsonify({"error": "Invalid transaction data format"}), 400
            
        if 'Narrative' not in df.columns:
            return jsonify({"error": "Missing 'Narrative' column in transactions"}), 400
            
        # Validate narratives
        df['Narrative'] = df['Narrative'].astype(str).str.strip()
        if df['Narrative'].empty or df['Narrative'].isna().any():
            return jsonify({"error": "Invalid or empty narratives found"}), 400
            
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
        # Get request data
        data = request.get_json()
        if not data or 'transactions' not in data:
            return jsonify({
                "error": "Missing transactions data"
            }), 400
            
        # Convert transactions to DataFrame
        df = pd.DataFrame(data['transactions'])
        required_columns = ['Narrative', 'Category']
        if not all(col in df.columns for col in required_columns):
            return jsonify({
                "error": f"Missing required columns: {required_columns}"
            }), 400
        
        # Initialize classification service
        classifier = ClassificationService(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            backend_api=request.host_url.rstrip('/')
        )
        
        # Train model
        training_response = classifier.train(df, "sheet_default", "user_default")
        
        # Process and store embeddings
        classifier.process_embeddings(training_response)
        classifier._store_training_data(
            training_response['embeddings'],
            df['Narrative'].tolist(),
            df['Category'].tolist(),
            "sheet_default"
        )
        
        return jsonify({
            "status": "success",
            "message": "Model trained successfully"
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port) 