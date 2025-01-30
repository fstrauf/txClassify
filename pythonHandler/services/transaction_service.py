from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TransactionService:
    def __init__(self):
        pass

    @staticmethod
    def parse_date(date_str: str, input_format: str) -> str:
        """Parse date string to standard format."""
        try:
            return datetime.strptime(date_str, input_format).strftime('%d/%m/%Y')
        except ValueError as e:
            logger.error(f"Date parsing error: {e}. Input: {date_str}, Format: {input_format}")
            return None

    @staticmethod
    def normalize_amount(amount_str: str) -> float:
        """Normalize amount string to float."""
        try:
            return float(str(amount_str).replace(',', '').replace(' ', ''))
        except ValueError:
            logger.error(f"Amount normalization error. Input: {amount_str}")
            return None

    def process_bank_file(self, file_data: str | pd.DataFrame, bank_config: Dict[str, Any]) -> pd.DataFrame:
        """Process a bank file or DataFrame according to its configuration."""
        try:
            # Handle input data
            if isinstance(file_data, str):
                df = pd.read_csv(
                    file_data,
                    skiprows=bank_config.get('skip_rows', 0),
                    encoding='utf-8-sig',
                    sep=bank_config.get('separator', ',')
                )
            elif isinstance(file_data, pd.DataFrame):
                df = file_data.copy()
                logger.info(f"Processing DataFrame with columns: {df.columns.tolist()}")
                # Convert column names to indices if they're not already
                df.columns = range(len(df.columns))
            else:
                raise ValueError(f"Unsupported data type: {type(file_data)}")

            # Extract column mappings
            mappings = bank_config.get('column_mapping', {})
            logger.info(f"Using column mappings: {mappings}")

            # Initialize result DataFrame with required columns
            result_df = pd.DataFrame(index=df.index)

            # Handle date column
            date_col = mappings.get('date')
            if date_col is not None:
                col_idx = int(date_col)
                if col_idx < len(df.columns):
                    result_df['date'] = df[col_idx].apply(
                        lambda x: self.parse_date(str(x), bank_config.get('date_format', '%Y-%m-%d'))
                    )
                else:
                    logger.warning(f"Date column index {col_idx} out of range")
                    result_df['date'] = None

            # Handle amount column
            amount_col = mappings.get('amount')
            if amount_col is not None:
                col_idx = int(amount_col)
                if col_idx < len(df.columns):
                    result_df['amount'] = df[col_idx].apply(self.normalize_amount)
                else:
                    logger.warning(f"Amount column index {col_idx} out of range")
                    result_df['amount'] = None

            # Handle description
            desc_col = mappings.get('description')
            if desc_col is not None:
                col_idx = int(desc_col)
                if col_idx < len(df.columns):
                    result_df['description'] = df[col_idx].astype(str)
                else:
                    logger.warning(f"Description column index {col_idx} out of range")
                    result_df['description'] = None

            # Handle currency (always AUD for now)
            result_df['currency'] = bank_config.get('defaultCurrency', 'AUD')

            # Remove any rows with missing values
            result_df = result_df.dropna()
            
            if len(result_df) == 0:
                logger.warning("No valid transactions found after processing")
            else:
                logger.info(f"Successfully processed {len(result_df)} transactions")

            return result_df

        except Exception as e:
            logger.error(f"Error processing bank file: {str(e)}")
            raise

    def process_multiple_files(self, files_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process multiple bank files and combine them."""
        all_transactions = pd.DataFrame()
        
        for file_info in files_data:
            bank_config = file_info["config"]
            file_content = file_info["content"]
            df = self.process_bank_file(file_content, bank_config)
            all_transactions = pd.concat([all_transactions, df], ignore_index=True)

        return all_transactions

    def prepare_for_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare transactions for classification."""
        df = df.copy()
        df['description'] = df['description'].apply(self.clean_text)
        return df

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and standardize text for classification."""
        import re
        
        # Convert to string and strip
        text = str(text).strip()
        
        # Remove common patterns that don't help classification
        patterns = [
            r'https?://\S+',  # URLs
            r'www\.\S+',      # Web addresses
            r'[^\x00-\x7F]+', # Non-ASCII characters
            r'\d+',           # Numbers
            r'\b\w{1,2}\b',   # 1-2 character words
            r'xx',            # Common filler
            r'Value Date|Card|AUS|USA|USD|PTY|LTD|Tap and Pay|TAP AND PAY'  # Common terms
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        
        # Remove special characters and extra spaces
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip() 