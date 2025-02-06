import os
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pandas as pd
from services.classification_service import ClassificationService
import logging
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

class GoogleSheetClassifier:
    def __init__(self):
        load_dotenv()
        self.creds = self._get_credentials()
        self.service = build('sheets', 'v4', credentials=self.creds)
        self.sheets = self.service.spreadsheets()
        
        # Initialize classification service
        supabase_url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
        supabase_key = os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        backend_api = os.environ.get("BACKEND_API")
        self.classifier = ClassificationService(supabase_url, supabase_key, backend_api)

    def _get_credentials(self):
        """Get valid credentials for Google Sheets API."""
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        return creds

    def read_unclassified_transactions(self, spreadsheet_id: str, range_name: str) -> pd.DataFrame:
        """Read unclassified transactions from Google Sheet."""
        try:
            result = self.sheets.values().get(
                spreadsheetId=spreadsheet_id,
                range=range_name
            ).execute()
            
            values = result.get('values', [])
            if not values:
                logger.info('No data found in the specified range.')
                return pd.DataFrame()
            
            # Assuming columns are: Date, Amount, Narrative, Currency
            df = pd.DataFrame(values[1:], columns=values[0])  # Skip header row
            logger.info(f"Read {len(df)} transactions from sheet")
            return df
            
        except Exception as e:
            logger.error(f"Error reading from Google Sheet: {str(e)}")
            raise

    def write_classified_transactions(
        self, 
        spreadsheet_id: str, 
        range_name: str,
        results: pd.DataFrame
    ) -> None:
        """Write classified transactions back to Google Sheet."""
        try:
            # Prepare data for writing
            header = ['Date', 'Amount', 'Description', 'Currency', 'Category', 'Confidence', 'Matched With']
            values = [header]
            
            # Add rows
            for _, row in results.iterrows():
                values.append([
                    row.get('Date', ''),
                    row.get('Amount', ''),
                    row['description'],
                    row.get('Currency', ''),
                    row['predicted_category'],
                    f"{row['similarity_score']:.2f}",
                    row['matched_description']
                ])
            
            body = {
                'values': values
            }
            
            # Write to sheet
            self.sheets.values().update(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                body=body
            ).execute()
            
            logger.info(f"Wrote {len(results)} classified transactions to sheet")
            
        except Exception as e:
            logger.error(f"Error writing to Google Sheet: {str(e)}")
            raise

    def process_new_transactions(
        self,
        source_spreadsheet_id: str,
        source_range: str,
        target_spreadsheet_id: str,
        target_range: str,
        training_sheet_id: str = "test_sheet",
        user_id: str = "test_user"
    ):
        """Process new transactions from source sheet and write to target sheet."""
        try:
            # Read unclassified transactions
            transactions = self.read_unclassified_transactions(
                source_spreadsheet_id,
                source_range
            )
            
            if transactions.empty:
                logger.info("No new transactions to process")
                return
            
            # Classify transactions
            logger.info(f"Classifying {len(transactions)} transactions")
            prediction = self.classifier.classify(transactions, training_sheet_id, user_id)
            
            # Process results
            results = self.classifier.process_webhook_response(prediction, training_sheet_id)
            
            # Add original transaction data
            results['Date'] = transactions['Date']
            results['Amount'] = transactions['Amount']
            results['Currency'] = transactions['Currency']
            
            # Write results
            self.write_classified_transactions(
                target_spreadsheet_id,
                target_range,
                results
            )
            
            logger.info("Transaction classification completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing transactions: {str(e)}")
            raise

def main():
    """Main function to run the Google Sheet classifier."""
    try:
        classifier = GoogleSheetClassifier()
        
        # Get sheet IDs and ranges from environment variables
        source_sheet_id = os.environ.get("SOURCE_SHEET_ID")
        source_range = os.environ.get("SOURCE_RANGE", "Sheet1!A:D")  # Date, Amount, Narrative, Currency
        target_sheet_id = os.environ.get("TARGET_SHEET_ID")
        target_range = os.environ.get("TARGET_RANGE", "Sheet1!A:G")  # Including Category and Confidence
        
        if not all([source_sheet_id, target_sheet_id]):
            raise ValueError("Missing required environment variables: SOURCE_SHEET_ID and TARGET_SHEET_ID")
        
        classifier.process_new_transactions(
            source_sheet_id,
            source_range,
            target_sheet_id,
            target_range
        )
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 