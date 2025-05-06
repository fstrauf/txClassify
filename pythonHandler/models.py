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
    userId: Optional[str] = None

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


class ClassifyRequest(BaseModel):
    """Request model for the /classify endpoint."""

    transactions: List[Union[str, TransactionInput]]
    spreadsheetId: Optional[str] = None
    sheetName: Optional[str] = "new_transactions"
    categoryColumn: Optional[str] = "E"
    startRow: Optional[str] = "1"

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
