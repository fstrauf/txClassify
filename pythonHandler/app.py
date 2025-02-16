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
        if not data or 'transactions' not in data:
            return jsonify({
                "error": "Missing transactions data"
            }), 400
            
        # Get required parameters
        user_id = data.get('userId')
        sheet_id = data.get('spreadsheetId')
        
        if not user_id or not sheet_id:
            return jsonify({
                "error": "Missing required parameters: userId and spreadsheetId"
            }), 400
            
        # Convert transactions to DataFrame
        df = pd.DataFrame(data['transactions'])
        if 'Narrative' not in df.columns:
            return jsonify({
                "error": "Missing 'Narrative' column in transactions"
            }), 400
        
        # Initialize classification service
        classifier = ClassificationService(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            backend_api=request.host_url.rstrip('/')
        )
        
        # Run classification
        prediction = classifier.classify(df, sheet_id, user_id)
        results = classifier.process_webhook_response(prediction, sheet_id)
        
        # Convert results to list of dictionaries
        results_list = results.to_dict('records')
        
        return jsonify(results_list)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            "error": str(e)
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