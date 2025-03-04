import os
import json
import pandas as pd
from dotenv import load_dotenv
import logging
from services.classification_service import ClassificationService
import numpy as np
import requests
from flask import Flask, request, jsonify
import threading
import time
from unittest.mock import patch, MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a test Flask app
test_app = Flask(__name__)

def test_storage_flow():
    """Test the basic storage and retrieval flow."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize service
        classifier = ClassificationService(
            supabase_url=os.environ.get("NEXT_PUBLIC_SUPABASE_URL"),
            supabase_key=os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY"),
            backend_api="http://localhost:5001"
        )
        
        # Create test data
        test_data = pd.DataFrame({
            'Narrative': [
                'WOOLWORTHS 2099 Dee Why AU',
                'TRANSPORTFORNSW TAP SYDNEY AUS',
                'GOOGLE*GOOGLE STORAGE Sydney AU'
            ],
            'Category': [
                'Groceries',
                'Transport',
                'Software'
            ]
        })
        
        logger.info("Step 1: Testing temporary storage...")
        # Test storing temporary data
        training_records = test_data.to_dict('records')
        training_key = classifier.store_temp_training_data(training_records, "test_sheet")
        logger.info(f"✓ Successfully stored temp data with key: {training_key}")
        
        logger.info("\nStep 2: Testing data retrieval...")
        # Test retrieving the data
        retrieved_data = classifier.get_temp_training_data(training_key)
        logger.info("✓ Successfully retrieved temp data")
        
        # Verify data integrity
        assert len(retrieved_data) == len(training_records), "Data length mismatch"
        assert all(item['Narrative'] in test_data['Narrative'].values for item in retrieved_data), "Narrative data mismatch"
        assert all(item['Category'] in test_data['Category'].values for item in retrieved_data), "Category data mismatch"
        logger.info("✓ Data integrity verified")
        
        logger.info("\nStep 3: Testing cleanup...")
        # Verify file is cleaned up
        try:
            classifier.supabase.storage.from_(classifier.bucket_name).download(f"{training_key}.json")
            logger.error("❌ File still exists (should have been cleaned up)")
        except Exception as e:
            logger.info("✓ File was properly cleaned up")
        
        logger.info("\nStorage flow tests passed successfully! 🎉")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        raise

def test_webhook_flow():
    """Test the complete webhook flow."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize service
        classifier = ClassificationService(
            supabase_url=os.environ.get("NEXT_PUBLIC_SUPABASE_URL"),
            supabase_key=os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY"),
            backend_api="http://localhost:5001"
        )
        
        # Create test data
        test_data = pd.DataFrame({
            'Narrative': [
                'WOOLWORTHS 2099 Dee Why AU',
                'TRANSPORTFORNSW TAP SYDNEY AUS',
                'GOOGLE*GOOGLE STORAGE Sydney AU'
            ],
            'Category': [
                'Groceries',
                'Transport',
                'Software'
            ]
        })
        
        logger.info("\nStep 1: Testing training initiation...")
        
        # Store training data first to ensure it exists
        training_records = test_data.to_dict('records')
        training_key = classifier.store_temp_training_data(training_records, "test_sheet")
        logger.info(f"✓ Stored training data with key: {training_key}")
        
        # Create mock prediction with the same training key
        mock_prediction = MagicMock()
        mock_prediction.id = training_key  # Use the same key
        mock_prediction.status = "succeeded"
        mock_embeddings = np.random.rand(3, 768).tolist()
        mock_prediction.output = [
            {"embedding": emb} for emb in mock_embeddings
        ]
        
        # Mock the Replicate API call
        with patch('replicate.predictions.create', return_value=mock_prediction):
            prediction = classifier.train(test_data, "test_sheet", "test_user")
            logger.info("✓ Training initiated successfully")
            
            # Simulate webhook call
            logger.info("\nStep 2: Testing webhook processing...")
            response = test_app.test_client().post(
                '/train/webhook',
                json={"output": prediction.output},
                query_string={
                    'sheetId': 'test_sheet',
                    'userId': 'test_user',
                    'training_key': training_key  # Use the same key
                }
            )
            
            assert response.status_code == 200, f"Webhook failed with status {response.status_code}"
            logger.info("✓ Webhook processed successfully")
        
        logger.info("\nWebhook flow tests completed! 🎉")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        raise

@test_app.route('/train/webhook', methods=['POST'])
def test_webhook():
    """Test webhook endpoint."""
    try:
        data = request.get_json()
        training_key = request.args.get('training_key')
        sheet_id = request.args.get('sheetId')
        user_id = request.args.get('userId')
        
        logger.info(f"Received webhook call with training_key: {training_key}")
        
        if not all([data, training_key, sheet_id, user_id]):
            missing = []
            if not data: missing.append("data")
            if not training_key: missing.append("training_key")
            if not sheet_id: missing.append("sheet_id")
            if not user_id: missing.append("user_id")
            return jsonify({"error": f"Missing required parameters: {', '.join(missing)}"}), 400
        
        # Initialize service
        classifier = ClassificationService(
            supabase_url=os.environ.get("NEXT_PUBLIC_SUPABASE_URL"),
            supabase_key=os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY"),
            backend_api="http://localhost:5001"
        )
        
        # Process embeddings
        embeddings = classifier.process_embeddings(data)
        logger.info(f"Processed embeddings shape: {embeddings.shape}")
        
        # Get training data
        training_data = classifier.get_temp_training_data(training_key)
        logger.info(f"Retrieved {len(training_data)} training examples")
        
        # Store training data with embeddings
        descriptions = [t['Narrative'] for t in training_data]
        categories = [t['Category'] for t in training_data]
        
        classifier._store_training_data(
            embeddings=embeddings,
            descriptions=descriptions,
            categories=categories,
            sheet_id=sheet_id
        )
        
        return jsonify({"status": "success", "message": "Training data processed"})
        
    except Exception as e:
        logger.error(f"Error in webhook: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Running storage flow tests...")
    test_storage_flow()
    
    logger.info("\nRunning webhook flow tests...")
    test_webhook_flow() 