/**
 * Sheet Operations Module
 * Handles importing transaction data to Google Sheets
 */

/**
 * Import transactions directly to the Expense-Detail sheet with predefined structure
 * @param {Array} processedData - Array of processed transaction objects
 * @return {Object} - Success/error result
 */
function importToExpenseDetailSheet(processedData) {
  try {
    Logger.log("Starting import to Expense-Detail sheet with " + processedData.length + " transactions");
    
    var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = spreadsheet.getSheetByName("Expense-Detail");
    
    // Create sheet if it doesn't exist
    if (!sheet) {
      Logger.log("Creating new Expense-Detail sheet");
      sheet = spreadsheet.insertSheet("Expense-Detail");
      
      // Add headers
      var headers = ["Source", "Date", "Description", "Amount Spent", "Category", "Currency Spent", "Amount in Base Currency: AUD"];
      sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
      
      // Format header row
      var headerRange = sheet.getRange(1, 1, 1, headers.length);
      headerRange.setFontWeight("bold");
      headerRange.setBackground("#f8f9fa");
      
      Logger.log("Created Expense-Detail sheet with headers");
    }
    
    // Find the next available row
    var lastRow = sheet.getLastRow();
    var startRow = lastRow + 1;
    
    Logger.log("Importing to Expense-Detail sheet starting at row " + startRow);
    
    // Prepare data for import
    var importData = [];
    var successCount = 0;
    var errorCount = 0;
    
    for (var i = 0; i < processedData.length; i++) {
      try {
        var transaction = processedData[i];
        
        // Validate transaction before import
        var validation = validateTransaction(transaction);
        if (!validation.valid) {
          Logger.log("Skipping invalid transaction: " + validation.errors.join(', '));
          errorCount++;
          continue;
        }
        
        var source = transaction["Source"] || "";
        var date = transaction["Date"] || new Date();
        var description = transaction["Description"] || "";
        var amountSpent = transaction["Amount Spent"] || 0;
        var category = transaction["Category"] || "";
        var currencySpent = transaction["Currency Spent"] || "";
        var amountBaseAUD = transaction["Amount in Base Currency: AUD"] || amountSpent;
        
        // Create row data
        var rowData = [source, date, description, amountSpent, category, currencySpent, amountBaseAUD];
        importData.push(rowData);
        successCount++;
        
      } catch (rowError) {
        Logger.log("Error processing transaction " + i + ": " + rowError.message);
        errorCount++;
      }
    }
    
    if (importData.length > 0) {
      // Import data to sheet
      var targetRange = sheet.getRange(startRow, 1, importData.length, 7);
      targetRange.setValues(importData);
      
      // Format the imported data
      formatImportedData(sheet, startRow, importData.length);
      
      Logger.log("Successfully imported " + successCount + " transactions to Expense-Detail sheet");
    }
    
    return {
      success: true,
      imported: successCount,
      errors: errorCount,
      startRow: startRow,
      endRow: startRow + importData.length - 1,
      message: "Successfully imported " + successCount + " transactions" + 
               (errorCount > 0 ? " (" + errorCount + " errors)" : "")
    };
    
  } catch (error) {
    Logger.log("Error importing to Expense-Detail sheet: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Format imported data in the sheet
 * @param {Sheet} sheet - The target sheet
 * @param {number} startRow - Starting row of imported data
 * @param {number} numRows - Number of rows imported
 */
function formatImportedData(sheet, startRow, numRows) {
  try {
    var dataRange = sheet.getRange(startRow, 1, numRows, 7);
    
    // Set alternating row colors
    for (var i = 0; i < numRows; i++) {
      var rowRange = sheet.getRange(startRow + i, 1, 1, 7);
      if (i % 2 === 0) {
        rowRange.setBackground("#f8f9fa");
      } else {
        rowRange.setBackground("#ffffff");
      }
    }
    
    // Format date column (column B)
    var dateRange = sheet.getRange(startRow, 2, numRows, 1);
    dateRange.setNumberFormat("mm/dd/yyyy");
    
    // Format amount columns (columns D and G)
    var amountRange1 = sheet.getRange(startRow, 4, numRows, 1);
    amountRange1.setNumberFormat("$#,##0.00");
    
    var amountRange2 = sheet.getRange(startRow, 7, numRows, 1);
    amountRange2.setNumberFormat("$#,##0.00");
    
    // Auto-resize columns
    sheet.autoResizeColumns(1, 7);
    
    Logger.log("Formatted imported data in rows " + startRow + " to " + (startRow + numRows - 1));
    
  } catch (error) {
    Logger.log("Error formatting imported data: " + error.message);
    // Don't fail the import if formatting fails
  }
}

/**
 * Legacy import function - kept for backwards compatibility
 * @param {Array} processedData - Processed transaction data
 * @param {string} targetSheet - Target sheet name (ignored - always imports to Expense-Detail)
 * @param {string} newSheetName - New sheet name (ignored)
 * @param {number} startingRow - Starting row (ignored - always appends)
 * @return {Object} - Import result
 */
function importToSheet(processedData, targetSheet, newSheetName, startingRow) {
  Logger.log("Legacy importToSheet called - redirecting to importToExpenseDetailSheet");
  return importToExpenseDetailSheet(processedData);
}

/**
 * Get sheet statistics
 * @param {string} sheetName - Name of the sheet to analyze
 * @return {Object} - Sheet statistics
 */
function getSheetStats(sheetName) {
  try {
    var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = spreadsheet.getSheetByName(sheetName);
    
    if (!sheet) {
      return {
        exists: false,
        error: "Sheet '" + sheetName + "' not found"
      };
    }
    
    var lastRow = sheet.getLastRow();
    var lastColumn = sheet.getLastColumn();
    var dataRows = Math.max(0, lastRow - 1); // Subtract header row
    
    // Get categories if this is the Expense-Detail sheet
    var categories = [];
    var totalAmount = 0;
    
    if (sheetName === "Expense-Detail" && lastRow > 1) {
      var dataRange = sheet.getRange(2, 1, dataRows, lastColumn);
      var data = dataRange.getValues();
      
      var categorySet = new Set();
      
      for (var i = 0; i < data.length; i++) {
        var row = data[i];
        var category = row[4]; // Category is in column E (index 4)
        var amount = row[3]; // Amount is in column D (index 3)
        
        if (category && category.toString().trim() !== '') {
          categorySet.add(category.toString().trim());
        }
        
        if (typeof amount === 'number' && !isNaN(amount)) {
          totalAmount += Math.abs(amount);
        }
      }
      
      categories = Array.from(categorySet);
    }
    
    return {
      exists: true,
      lastRow: lastRow,
      lastColumn: lastColumn,
      dataRows: dataRows,
      categories: categories,
      uniqueCategories: categories.length,
      totalAmount: totalAmount,
      hasData: dataRows > 0
    };
    
  } catch (error) {
    Logger.log("Error getting sheet stats: " + error.message);
    return {
      exists: false,
      error: error.message
    };
  }
}

/**
 * Backup sheet data before making changes
 * @param {string} sheetName - Name of sheet to backup
 * @return {Object} - Backup result
 */
function backupSheet(sheetName) {
  try {
    var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    var sourceSheet = spreadsheet.getSheetByName(sheetName);
    
    if (!sourceSheet) {
      return {
        success: false,
        error: "Sheet '" + sheetName + "' not found"
      };
    }
    
    var timestamp = Utilities.formatDate(new Date(), Session.getScriptTimeZone(), "yyyy-MM-dd_HH-mm-ss");
    var backupName = sheetName + "_backup_" + timestamp;
    
    var backupSheet = sourceSheet.copyTo(spreadsheet);
    backupSheet.setName(backupName);
    
    Logger.log("Created backup sheet: " + backupName);
    
    return {
      success: true,
      backupName: backupName,
      message: "Backup created: " + backupName
    };
    
  } catch (error) {
    Logger.log("Error creating backup: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Clean up old backup sheets (keep only last 5)
 * @param {string} baseSheetName - Base sheet name to clean backups for
 */
function cleanupOldBackups(baseSheetName) {
  try {
    var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    var sheets = spreadsheet.getSheets();
    
    var backupSheets = [];
    var backupPrefix = baseSheetName + "_backup_";
    
    for (var i = 0; i < sheets.length; i++) {
      var sheet = sheets[i];
      if (sheet.getName().startsWith(backupPrefix)) {
        backupSheets.push({
          sheet: sheet,
          name: sheet.getName(),
          timestamp: sheet.getName().replace(backupPrefix, '')
        });
      }
    }
    
    // Sort by timestamp (newest first)
    backupSheets.sort(function(a, b) {
      return b.timestamp.localeCompare(a.timestamp);
    });
    
    // Delete old backups (keep only 5 most recent)
    var deleted = 0;
    for (var j = 5; j < backupSheets.length; j++) {
      try {
        spreadsheet.deleteSheet(backupSheets[j].sheet);
        deleted++;
        Logger.log("Deleted old backup: " + backupSheets[j].name);
      } catch (deleteError) {
        Logger.log("Could not delete backup " + backupSheets[j].name + ": " + deleteError.message);
      }
    }
    
    if (deleted > 0) {
      Logger.log("Cleaned up " + deleted + " old backup sheets");
    }
    
  } catch (error) {
    Logger.log("Error cleaning up backups: " + error.message);
  }
}

/**
 * Check if user has edit permissions on the spreadsheet
 * @return {boolean} - True if user can edit
 */
function hasEditPermissions() {
  try {
    var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    var protection = spreadsheet.getProtections(SpreadsheetApp.ProtectionType.SHEET);
    
    // If there are no protections, user can edit
    if (protection.length === 0) {
      return true;
    }
    
    // Check if user is in the editor list for any protection
    var userEmail = Session.getEffectiveUser().getEmail();
    
    for (var i = 0; i < protection.length; i++) {
      var editors = protection[i].getEditors();
      for (var j = 0; j < editors.length; j++) {
        if (editors[j].getEmail() === userEmail) {
          return true;
        }
      }
    }
    
    return false;
    
  } catch (error) {
    Logger.log("Error checking edit permissions: " + error.message);
    // Assume they have permissions if we can't check
    return true;
  }
} 