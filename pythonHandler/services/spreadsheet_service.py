import json
import pandas as pd
from typing import List, Dict, Any, Tuple
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging
import numpy as np

logger = logging.getLogger(__name__)

class SpreadsheetService:
    def __init__(self, service_account_info: Dict[str, Any]):
        """Initialize with Google service account credentials."""
        self.creds = self._get_credentials(service_account_info)
        self.service = build("sheets", "v4", credentials=self.creds)

    @staticmethod
    def _get_credentials(service_account_info: Dict[str, Any]) -> Credentials:
        """Get Google Sheets API credentials from service account info."""
        return Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )

    def get_sheet_data(self, sheet_id: str, range_name: str) -> List[List[Any]]:
        """Fetch data from Google Sheets."""
        try:
            result = (
                self.service.spreadsheets()
                .values()
                .get(spreadsheetId=sheet_id, range=range_name)
                .execute()
            )
            return result.get("values", [])
        except HttpError as e:
            if e.resp.status == 403:
                logger.error("Access to the Google Sheet was denied.")
            else:
                logger.error(f"HTTP Error while accessing the Google Sheets API: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise

    def append_to_sheet(
        self,
        sheet_id: str,
        range_name: str,
        values: List[List[Any]]
    ) -> None:
        """Append data to Google Sheets."""
        try:
            # First check if we have access
            try:
                self.service.spreadsheets().get(spreadsheetId=sheet_id).execute()
            except HttpError as e:
                if e.resp.status == 403:
                    logger.error("No write permission for the sheet. Please share the sheet with the service account email with editor access.")
                    raise Exception("Please share the sheet with the service account email (expense-sorted@txclassify.iam.gserviceaccount.com) with editor access.")
                raise

            # Then try to append
            self.service.spreadsheets().values().append(
                spreadsheetId=sheet_id,
                range=range_name,
                valueInputOption="USER_ENTERED",
                insertDataOption="INSERT_ROWS",
                body={"values": values}
            ).execute()
            logger.info("Data appended successfully.")
        except Exception as e:
            logger.error(f"Error appending to sheet: {e}")
            raise

    @staticmethod
    def get_column_range(
        column_order: List[Dict[str, Any]]
    ) -> Tuple[str, str, List[str]]:
        """Get column range and types from column order configuration."""
        if not column_order:
            return None, None, None

        sorted_list = sorted(column_order, key=lambda x: x["index"])
        first_name = sorted_list[0]["name"]
        last_name = sorted_list[-1]["name"]
        types = [item["type"] for item in sorted_list]

        return first_name, last_name, types

    @staticmethod
    def prepare_sheet_data(
        new_expenses: List[List[Any]],
        new_categories: pd.DataFrame,
        source_columns: List[str],
        target_columns: List[str]
    ) -> pd.DataFrame:
        """Prepare data for sheet update."""
        try:
            logger.info(f"Preparing sheet data with {len(new_expenses)} rows")
            logger.info(f"Source columns: {source_columns}")
            logger.info(f"Target columns: {target_columns}")

            # Create DataFrame with source columns
            df = pd.DataFrame(new_expenses, columns=source_columns)
            
            # Handle amount-based categorization
            if "amount" in df.columns:
                df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
                df["category"] = np.where(
                    df["amount"] > 0,
                    "Credit",
                    new_categories["category"].values if isinstance(new_categories, pd.DataFrame) and "category" in new_categories else ["Unknown"] * len(df)
                )
            else:
                df["category"] = new_categories["category"].values if isinstance(new_categories, pd.DataFrame) and "category" in new_categories else ["Unknown"] * len(df)

            # Handle source column
            if "source" in target_columns and "source" not in df.columns:
                df["source"] = "Bank 1"

            # Ensure all target columns exist
            for col in target_columns:
                if col not in df.columns:
                    logger.warning(f"Adding missing column: {col}")
                    df[col] = None

            # Clean up data types and handle NaN values
            if "amount" in df.columns:
                df["amount"] = df["amount"].fillna(0).astype(float)
            if "date" in df.columns:
                df["date"] = df["date"].fillna("")
            if "description" in df.columns:
                df["description"] = df["description"].fillna("")
            if "category" in df.columns:
                df["category"] = df["category"].fillna("Unknown")
            if "currency" in df.columns:
                df["currency"] = df["currency"].fillna("AUD")
            if "source" in df.columns:
                df["source"] = df["source"].fillna("Bank 1")

            # Log the final DataFrame info
            logger.info(f"Final DataFrame columns: {df.columns.tolist()}")
            logger.info(f"Final DataFrame shape: {df.shape}")

            # Reorder columns to match target
            result = df[target_columns]
            
            # Convert DataFrame to list of lists and ensure all values are strings
            values = result.values.tolist()
            cleaned_values = []
            for row in values:
                cleaned_row = []
                for val in row:
                    if pd.isna(val):
                        cleaned_row.append("")
                    elif isinstance(val, (int, float)):
                        cleaned_row.append(str(val))
                    else:
                        cleaned_row.append(str(val))
                cleaned_values.append(cleaned_row)

            return pd.DataFrame(cleaned_values, columns=target_columns)

        except Exception as e:
            logger.error(f"Error in prepare_sheet_data: {str(e)}")
            logger.error(f"new_expenses shape: {new_expenses.shape if hasattr(new_expenses, 'shape') else 'no shape'}")
            logger.error(f"new_categories: {type(new_categories)}")
            raise 