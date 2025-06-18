from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, ValidationError, field_validator
import logging

logger = logging.getLogger(__name__)


# === Request Validation Models ===
class TransactionBase(BaseModel):
    """Base model for transaction data containing description field."""

    description: str

    @field_validator("description")
    @classmethod
    def description_must_not_be_empty(cls, v):
        v = v.strip() if v else ""
        if not v:
            raise ValueError("Description cannot be empty")
        return v


class Transaction(TransactionBase):
    """Transaction model with narrative and category fields."""

    Category: Optional[str] = None


class TrainRequest(BaseModel):
    """Request model for the /train endpoint."""

    transactions: List[Transaction]
    expenseSheetId: Optional[str] = None

    @field_validator("transactions")
    @classmethod
    def transactions_not_empty(cls, v):
        if not v or len(v) < 10:
            raise ValueError("At least 10 valid transactions required for training")
        return v


class TransactionInput(BaseModel):
    description: str
    money_in: Optional[bool] = (
        None  # True for income/credit/positive transactions, False for expense/debit/negative transactions
    )
    amount: Optional[float] = None # Numerical amount of the transaction


class ClassifyRequest(BaseModel):
    """Request model for the /classify endpoint."""

    transactions: List[Union[str, TransactionInput]]
    user_categories: Optional[List[Dict[str, str]]] = None
    # spreadsheetId: Optional[str] = None # Client-side concern for Sheets Add-on
    # sheetName: Optional[str] = "new_transactions" # Client-side concern
    # categoryColumn: Optional[str] = "E" # Client-side concern
    # startRow: Optional[str] = "1" # Client-side concern

    @field_validator("transactions")
    @classmethod
    def transactions_not_empty(cls, v):
        if not v:
            raise ValueError("At least one transaction is required")
        for tx in v:
            if isinstance(tx, dict) and "description" not in tx:
                raise ValueError("Transaction objects must have a 'description' field")
        return v


class UserConfigRequest(BaseModel):
    """Request model for the /user-config endpoint."""

    userId: str = Field(..., min_length=1)
    apiKey: Optional[str] = None


class TransactionAnalyticsInput(BaseModel):
    """Input model for transaction analytics."""
    description: Optional[str] = None
    Description: Optional[str] = None  # Alternative column name
    amount: Optional[float] = None  # Negative for expenses, positive for income
    Amount_Spent: Optional[float] = None  # Alternative column name (from CSV)
    category: Optional[str] = None
    Category: Optional[str] = None  # Alternative column name
    date: Optional[str] = None  # ISO date string
    Date: Optional[str] = None  # Alternative column name
    money_in: Optional[bool] = None
    Source: Optional[str] = None  # Bank source field


class FinancialAnalyticsRequest(BaseModel):
    """Request model for the /financial-analytics endpoint."""
    
    transactions: List[TransactionAnalyticsInput]
    analysis_types: Optional[List[str]] = None  # Types of analysis to perform
    excluded_categories: Optional[List[str]] = None  # Categories to exclude from all analytics
    
    @field_validator("transactions")
    @classmethod
    def transactions_not_empty(cls, v):
        if not v:
            raise ValueError("At least one transaction is required for analytics")
        return v
