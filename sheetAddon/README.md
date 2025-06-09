# Bank Transaction Import for Google Sheets Add-on

This Google Sheets add-on now includes a powerful bank transaction import feature that makes it easy to import and standardize transaction data from CSV files exported from various banks.

## Features

### 🏦 Multi-Bank Support
- **ANZ**: Automatically detects and combines multiple description fields
- **ASB**: Handles ASB FastNet Classic exports  
- **Westpac**: Supports Westpac online banking exports
- **BNZ**: Compatible with BNZ internet banking exports
- **Other Banks**: Generic CSV support with intelligent column detection

### 🎯 Smart Column Detection
- Automatically detects date, amount, and description columns
- Handles multiple description fields (common with ANZ)
- Supports various date formats (DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD, etc.)
- Intelligent delimiter detection (comma, semicolon, tab)

### 📊 Data Standardization
- Converts all transactions to a unified format
- Proper date formatting for Google Sheets
- Currency formatting for amounts
- Automatic debit/credit classification
- Clean, professional sheet formatting

## How to Use

### 1. Access the Feature
- Open your Google Sheet
- Go to **Add-ons** → **ExpenseSorted** → **Import Bank Transactions**

### 2. Upload Your CSV File
- Export transactions from your bank's online banking
- Drag and drop the CSV file or click to browse
- The system will automatically analyze the file structure

### 3. Configure Column Mapping
- Review the auto-detected column mappings
- Adjust mappings if needed (required: Date, Amount, Description)
- Select the appropriate date format
- Preview the data transformation

### 4. Import to Sheet
- Choose to import to the current sheet or create a new one
- Select the starting row for your data
- Click "Import Transactions" to complete the process

## Export Instructions by Bank

### ANZ
1. Log into ANZ Internet Banking
2. Go to **Accounts** → Select your account
3. Click **Export** → Choose **CSV format**
4. Select date range and download

### ASB  
1. Log into ASB FastNet Classic
2. Go to **Account Activity**
3. Click **Export** → Choose **CSV**
4. Select date range and download

### Westpac
1. Log into Westpac Online Banking  
2. Go to **Statements**
3. Click **Download CSV**
4. Choose date range and export

### BNZ
1. Log into BNZ Internet Banking
2. Go to **Accounts** → Select account  
3. Click **Export** → Choose **CSV format**
4. Select date range and download

## Data Format

The imported data will be standardized with these columns:

| Column | Description |
|--------|-------------|
| Date | Transaction date (YYYY-MM-DD format) |
| Amount | Transaction amount (positive for credits, negative for debits) |
| Description | Combined transaction description |
| Account | Account name/identifier |
| Category | Transaction category (if available) |
| Reference | Reference number or additional details |
| Debit/Credit | Transaction type indicator |

## Tips for Best Results

### 📋 File Preparation
- Export at least 30 days of transactions for better categorization
- Ensure your CSV file has headers in the first row
- Use UTF-8 encoding if you have special characters

### 🔧 Column Mapping
- **Date**: Should contain the transaction date
- **Amount**: Numerical value (negative for expenses, positive for income)
- **Description**: Main transaction description
- **Description 2**: Additional details (will be combined with main description)

### 💡 Common Issues
- **Multiple amount columns**: Choose the "Source Amount" or main transaction amount
- **Missing dates**: Ensure date column is properly mapped
- **Special characters**: Save your CSV with UTF-8 encoding

## Integration with Categorization

After importing transactions, you can use the **"Categorise New Transactions"** feature to automatically categorize all imported transactions using AI-powered classification.

## Technical Details

### Supported File Formats
- CSV files with various delimiters (comma, semicolon, tab)
- Files with quoted fields and escaped quotes
- Multiple encoding formats (UTF-8 recommended)

### Date Format Support
- `YYYY-MM-DD` (ISO format)
- `DD/MM/YYYY` (European format)  
- `MM/DD/YYYY` (US format)
- `DD.MM.YYYY` (German format)
- `YYYY/MM/DD` (Asian format)
- `YYYY-MM-DD HH:MM:SS` (with timestamps)

### Performance
- Handles files with thousands of transactions
- Batch processing for efficient import
- Progress indicators for large files

## Support

If you encounter issues with a specific bank format or need assistance:

1. Check that your CSV file has proper headers
2. Verify the date format matches your bank's export
3. Ensure required fields (Date, Amount, Description) are mapped
4. Contact support with a sample CSV file for custom bank support

---

**Note**: This add-on processes your data locally in Google Sheets. No transaction data is sent to external servers during the import process. 