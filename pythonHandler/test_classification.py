import os
import pandas as pd
from services.classification_service import ClassificationService
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import threading
import time
from queue import Queue
from pyngrok import ngrok, conf
import socket
import requests
from werkzeug.serving import make_server

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Reduce logging verbosity for some modules
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('pyngrok').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Create Flask app for webhook handling
app = Flask(__name__)
result_queue = Queue()
server = None

def is_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def wait_for_server(port, max_retries=5):
    """Wait for the Flask server to start."""
    for i in range(max_retries):
        try:
            response = requests.get(f'http://localhost:{port}/')
            if response.status_code == 404:  # Flask is running but route doesn't exist
                return True
        except requests.exceptions.ConnectionError:
            if i < max_retries - 1:
                time.sleep(1)
                continue
        except Exception as e:
            logger.error(f"Error checking server: {str(e)}")
            if i < max_retries - 1:
                time.sleep(1)
                continue
    return False

class FlaskServerThread(threading.Thread):
    def __init__(self, app, host, port):
        threading.Thread.__init__(self)
        self.srv = make_server(host, port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        logger.info("Starting Flask server")
        self.srv.serve_forever()

    def shutdown(self):
        self.srv.shutdown()

@app.route('/training', methods=['POST'])
def handle_training_webhook():
    """Handle training completion webhook."""
    try:
        data = request.get_json()
        sheet_id = request.args.get('sheetId')
        user_id = request.args.get('userId')
        logger.info(f"Received training webhook for sheet {sheet_id} and user {user_id}")
        
        # Log only essential parts of the webhook payload
        if 'output' in data:
            num_embeddings = len(data['output'])
            logger.info(f"Received {num_embeddings} embeddings in training webhook")
        
        result_queue.put(('training', data))
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error(f"Error in training webhook: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/classify', methods=['POST'])
def handle_classify_webhook():
    """Handle classification completion webhook."""
    try:
        data = request.get_json()
        sheet_id = request.args.get('sheetId')
        user_id = request.args.get('userId')
        logger.info(f"Received classification webhook for sheet {sheet_id} and user {user_id}")
        
        # Log only essential parts of the webhook payload
        if 'output' in data:
            num_embeddings = len(data['output'])
            logger.info(f"Received {num_embeddings} embeddings in classification webhook")
        
        result_queue.put(('classify', data))
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error(f"Error in classification webhook: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

def setup_ngrok():
    """Set up ngrok tunnel for HTTPS webhook."""
    try:
        # Kill any existing ngrok processes
        ngrok.kill()
        
        # Set up ngrok configuration
        ngrok_auth_token = os.environ.get("NGROK_AUTH_TOKEN")
        if not ngrok_auth_token:
            raise ValueError("NGROK_AUTH_TOKEN environment variable is required")
            
        # Configure ngrok
        conf.get_default().auth_token = ngrok_auth_token
        
        # Find an available port
        port = 5001
        while is_port_in_use(port):
            port += 1
            if port > 5010:  # Try up to port 5010
                raise RuntimeError("Could not find an available port")
        
        logger.info(f"Using port {port} for Flask server")
        
        # Start Flask server
        global server
        server = FlaskServerThread(app, '0.0.0.0', port)
        server.daemon = True
        server.start()
        
        # Wait for server to start
        if not wait_for_server(port):
            raise RuntimeError("Flask server failed to start")
        
        logger.info("Flask server started successfully")
        
        # Start ngrok tunnel
        tunnel_options = {
            "bind_tls": True,
            "proto": "http",
            "addr": f"http://localhost:{port}"
        }
        http_tunnel = ngrok.connect(**tunnel_options)
        tunnel_url = http_tunnel.public_url
        logger.info(f"ngrok tunnel established at: {tunnel_url}")
        return tunnel_url
        
    except Exception as e:
        logger.error(f"Failed to set up ngrok: {str(e)}")
        if server:
            try:
                server.shutdown()
            except:
                pass
        raise

def main():
    try:
        # Set up ngrok tunnel and Flask server
        webhook_base_url = setup_ngrok()
        logger.info(f"Using webhook base URL: {webhook_base_url}")

        # Initialize service with Supabase credentials
        supabase_url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
        supabase_key = os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        
        if not all([supabase_url, supabase_key]):
            raise ValueError("Missing required environment variables: NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY")
        
        # Use the dynamically generated ngrok URL instead of BACKEND_API
        classifier = ClassificationService(supabase_url, supabase_key, webhook_base_url)
        
        # Load sample data
        logger.info("Loading sample data...")
        training_data = pd.read_csv('trained_data.csv')
        new_data = pd.read_csv('new_data.csv', names=['Date', 'Amount', 'Narrative', 'Currency'])
        
        print("\n=== Training Data Sample ===")
        print(training_data[['Narrative', 'Category']].head())
        print("\n=== New Data ===")
        print(new_data[['Narrative']])
        
        # Start training
        print("\n=== Starting Training ===")
        training_prediction = classifier.train(training_data, "test_sheet", "test_user")
        print(f"Training started. Prediction ID: {training_prediction.id}")
        
        # Wait for training webhook
        print("Waiting for training to complete...")
        webhook_type, training_response = result_queue.get(timeout=1200)  # 20 minute timeout
        if webhook_type != 'training':
            raise ValueError(f"Unexpected webhook type: {webhook_type}")
        
        # Process and store training data
        print("Processing training embeddings...")
        embeddings = classifier.process_embeddings(training_response)
        classifier._store_training_data(
            embeddings=embeddings,
            descriptions=training_data['Narrative'].astype(str).tolist(),
            categories=training_data['Category'].astype(str).tolist(),
            sheet_id="test_sheet"
        )
        print("Training data stored successfully")
        
        # Start classification
        print("\n=== Starting Classification ===")
        classification_prediction = classifier.classify(new_data, "test_sheet", "test_user")
        print(f"Classification started. Prediction ID: {classification_prediction.id}")
        
        # Wait for classification webhook
        print("Waiting for classification to complete...")
        webhook_type, classification_data = result_queue.get(timeout=1200)  # 20 minute timeout
        if webhook_type != 'classify':
            raise ValueError(f"Unexpected webhook type: {webhook_type}")
            
        # Process classification results
        print("\n=== Classification Results ===")
        results = classifier.process_webhook_response(classification_data, "test_sheet")
        for _, row in results.iterrows():
            print(f"\nDescription: {row['description']}")
            print(f"Predicted Category: {row['predicted_category']}")
            print(f"Similarity Score: {row['similarity_score']:.2f}")
            print(f"Matched With: {row['matched_description']}")

    except FileNotFoundError as e:
        print(f"\nError: Could not find sample data files. Please ensure 'trained_data.csv' and 'new_data.csv' are in the current directory.")
        print(f"Current directory: {os.getcwd()}")
        print(f"Error details: {str(e)}")
    except ValueError as e:
        print(f"\nConfiguration Error: {str(e)}")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise
    finally:
        # Clean up
        if server:
            try:
                server.shutdown()
            except:
                pass
        ngrok.kill()

if __name__ == "__main__":
    main() 