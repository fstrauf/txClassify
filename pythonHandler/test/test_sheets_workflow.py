import os
import json
import pandas as pd
import logging
from dotenv import load_dotenv
from services.classification_service import ClassificationService
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockGoogleSheets:
    """Mock class to simulate Google Sheets behavior"""
    def __init__(self):
        self.log_entries = []
        
    def update_status(self, message: str):
        """Simulate status updates in the Log sheet"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        status = "INFO"
        if "error" in message.lower():
            status = "ERROR"
        elif "completed" in message.lower():
            status = "SUCCESS"
        elif "progress" in message.lower() or "processing" in message.lower():
            status = "PROCESSING"
            
        log_entry = {
            "timestamp": timestamp,
            "status": status,
            "message": message
        }
        self.log_entries.append(log_entry)
        logger.info(f"{status}: {message}")

def main():
    try:
        # Initialize services
        sheets = MockGoogleSheets()
        classifier = ClassificationService(
            supabase_url=os.environ.get("NEXT_PUBLIC_SUPABASE_URL"),
            supabase_key=os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY"),
            backend_api="http://localhost:5001"  # Local server URL
        )
        
        # Test data
        training_data = [
            {"Narrative": "UBER EATS Sydney AU", "Category": "Food & Dining"},
            {"Narrative": "NETFLIX Sydney AU", "Category": "Entertainment"},
            {"Narrative": "COLES SUPERMARKETS Sydney AU", "Category": "Groceries"},
            {"Narrative": "SPOTIFY Stockholm SE", "Category": "Entertainment"},
            {"Narrative": "AMAZON PRIME Sydney AU", "Category": "Shopping"}
        ]
        
        new_transactions = [
            {"Narrative": "UBER EATS Melbourne AU"},
            {"Narrative": "NETFLIX Melbourne AU"},
            {"Narrative": "WOOLWORTHS Sydney AU"}
        ]
        
        # 1. Training Process
        sheets.update_status("Starting training...")
        
        # Convert training data to DataFrame
        df_training = pd.DataFrame(training_data)
        sheets.update_status(f"Processing {len(df_training)} training transactions...")
        
        # Start training
        training_prediction = classifier.train(df_training, "test_sheet", "test_user")
        sheets.update_status(f"Training started with prediction ID: {training_prediction.id}")
        
        # Poll for training completion
        max_attempts = 30
        attempt = 0
        while attempt < max_attempts:
            prediction = replicate.predictions.get(training_prediction.id)
            sheets.update_status(f"Training status: {prediction.status}")
            
            if prediction.status == "succeeded":
                # Process training results
                embeddings = classifier.process_embeddings(prediction.output)
                classifier._store_training_data(
                    embeddings=embeddings,
                    descriptions=df_training['Narrative'].tolist(),
                    categories=df_training['Category'].tolist(),
                    sheet_id="test_sheet"
                )
                sheets.update_status("Training completed successfully!")
                break
            elif prediction.status == "failed":
                sheets.update_status(f"Error: Training failed - {prediction.error}")
                return
                
            time.sleep(2)
            attempt += 1
            
        if attempt >= max_attempts:
            sheets.update_status("Error: Training timed out")
            return
            
        # 2. Classification Process
        sheets.update_status("Starting classification...")
        
        # Convert new transactions to DataFrame
        df_new = pd.DataFrame(new_transactions)
        sheets.update_status(f"Processing {len(df_new)} new transactions...")
        
        # Start classification
        classification_prediction = classifier.classify(df_new, "test_sheet", "test_user")
        sheets.update_status(f"Classification started with prediction ID: {classification_prediction.id}")
        
        # Poll for classification completion
        attempt = 0
        while attempt < max_attempts:
            prediction = replicate.predictions.get(classification_prediction.id)
            sheets.update_status(f"Classification status: {prediction.status}")
            
            if prediction.status == "succeeded":
                # Process classification results
                results = classifier.process_webhook_response(prediction.output, "test_sheet")
                sheets.update_status("Classification completed successfully!")
                
                # Print results
                print("\nClassification Results:")
                print("----------------------")
                for _, row in results.iterrows():
                    print(f"\nDescription: {row['description']}")
                    print(f"Predicted Category: {row['predicted_category']}")
                    print(f"Confidence Score: {row['similarity_score']:.2f}")
                    print(f"Matched With: {row['matched_description']}")
                break
            elif prediction.status == "failed":
                sheets.update_status(f"Error: Classification failed - {prediction.error}")
                return
                
            time.sleep(2)
            attempt += 1
            
        if attempt >= max_attempts:
            sheets.update_status("Error: Classification timed out")
            return
            
        # Print log entries
        print("\nOperation Log:")
        print("-------------")
        for entry in sheets.log_entries:
            print(f"{entry['timestamp']} - {entry['status']}: {entry['message']}")
        
    except Exception as e:
        logger.error(f"Error in test workflow: {str(e)}")
        sheets.update_status(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 