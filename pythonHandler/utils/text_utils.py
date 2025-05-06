"""Utility functions for cleaning and processing text data."""

import re
import logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean transaction description text while preserving core business information."""
    # Convert to string and strip whitespace
    text = str(text).strip().upper()

    # Remove common payment method patterns
    payment_patterns = [
        r"\s*(?:TAP AND PAY|CONTACTLESS|PIN PURCHASE|EFTPOS|DEBIT CARD|CREDIT CARD)",
        r"\s*(?:PURCHASE|PAYMENT|TRANSFER|DIRECT DEBIT)",
        r"\s*(?:APPLE PAY|GOOGLE PAY|SAMSUNG PAY)",
        r"\s*(?:POS\s+PURCHASE|POS\s+PAYMENT)",
    ]
    for pattern in payment_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove transaction metadata
    patterns = [
        # Remove dates in various formats
        r"\s*\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}",
        r"\s*\d{2,4}[-/.]\d{1,2}[-/.]\d{1,2}",
        # Remove times
        r"\s*\d{1,2}:\d{2}(?::\d{2})?(?:s*[AaPp][Mm])?",
        # Remove card numbers and references
        r"\s*(?:CARD|REF|REFERENCE|TXN)\s*[#]?\s*[xX*]?\d+",
        r"\s*\d{4}[xX*]+\d{4}",
        # Remove amounts and currency
        r"\s*\$?d+\.d{2}\s*(?:AUD|USD|EUR|GBP)?",
        r"\s*(?:AUD|USD|EUR|GBP)\s*\$?d+\.d{2}",
        # Remove common transaction IDs and references
        r"\s*(?:ID|REF|REFERENCE|TXN|TRANS|INV|INVOICE)\s*[:#]?\s*\d+",
        r"\s*\d{6,}",  # Remove long numbers (likely reference numbers)
        # Remove dates and timestamps in various formats
        r"\s*\d{8,14}",  # YYYYMMDD, YYYYMMDDHHmm, etc.
        # Remove common location/terminal identifiers
        r"\s+(?:T/C|QPS|AU|NS|TERMINAL|TID|MID)\s*[:#]?\s*\d*",
        # Remove store/branch numbers
        r"\s+(?:STORE|BRANCH|LOCATION|LOC)\s*[:#]?\s*\d+",
        r"\s+#\s*\d+",
        r"\s+\d{2,4}(?:s|$)",  # Standalone 2-4 digit numbers (likely store numbers)
        # Remove business suffixes
        r"\s+(?:PTY\s*LTD|P/?L|LIMITED|AUSTRALIA(?:N)?|CORPORATION|CORP|INC|LLC|GMBH|SA|AG)",  # Added GMBH, SA, AG
        # Remove common prefixes
        r"^(?:SQ|LIV|SMP|MWA|EZI|SP|PP)\s*[*#]?\s*",
        # Remove transaction types
        r"^(?:POS|ATM|DD|SO|BP|AP)\s+",
        r"^(?:CRED\s+VOUCHER|PENDING|RETURN|REFUND|CREDIT|DEBIT)\s+",
        # Remove anything in parentheses or brackets
        r"\s*\([^)]*\)",
        r"\s*\[[^\]]*\]",
        # Remove URLs and email addresses
        r"\s*(?:WWW|HTTP|HTTPS).*$",
        r"\s*\S+@\S+\.\S+",
        # Remove state/country codes at the end
        r"\s+(?:NSW|VIC|QLD|SA|WA|NT|ACT|TAS|AUS|USA|UK|NZ)$",
        # Remove extra spaces between digits
        r"(\d)\s+(\d)",
        # Remove special characters and multiple spaces
        r"[^w\s-]",
    ]

    # Apply patterns one by one
    for pattern in patterns:
        text = re.sub(
            pattern, r"\1\2" if r"\1" in pattern else "", text, flags=re.IGNORECASE
        )

    # Remove extra whitespace and normalize spaces
    text = " ".join(text.split())

    # Trim long transaction names
    words = text.split()
    if len(words) > 5:  # Changed from 4 to 5
        text = " ".join(words[:4])  # Changed from 3 to 4

    # Remove any remaining noise words at the end
    noise_words = {
        "VALUE",
        "DATE",
        "DIRECT",
        "DEBIT",
        "CREDIT",
        "CARD",
        "PAYMENT",
        "PURCHASE",
    }
    words = text.split()
    if len(words) > 1 and words[-1] in noise_words:
        text = " ".join(words[:-1])

    return text.strip()
