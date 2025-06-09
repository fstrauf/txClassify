/**
 * API Integration Service
 * Handles all communication with the ExpenseSorted API for training and categorization
 */

/**
 * Start training the model using existing transactions from Expense-Detail sheet
 * @return {Object} - Success/error result with prediction ID
 */
function startModelTrainingForImport() {
  try {
    Logger.log("=== STARTING MODEL TRAINING FOR IMPORT ===");
    
    // Get the Expense-Detail sheet
    var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = spreadsheet.getSheetByName("Expense-Detail");
    
    if (!sheet) {
      Logger.log("ERROR: No Expense-Detail sheet found");
      return {
        success: false,
        error: "No Expense-Detail sheet found. Cannot train model without existing transaction data."
      };
    }
    
    var lastRow = sheet.getLastRow();
    Logger.log("Expense-Detail sheet has " + lastRow + " rows (including header)");
    
    if (lastRow < 2) {
      Logger.log("ERROR: Not enough data - only " + lastRow + " rows");
      return {
        success: false,
        error: "Not enough existing transactions in Expense-Detail sheet for training. At least 10 transactions are required."
      };
    }
    
    // Read all data from the sheet (skip header row)
    var dataRange = sheet.getRange(2, 1, lastRow - 1, 7); // 7 columns: Source, Date, Description, Amount Spent, Category, Currency Spent, Amount in Base Currency: AUD
    var data = dataRange.getValues();
    Logger.log("Read " + data.length + " data rows from sheet");
    
    // Prepare training transactions and track categories
    var transactions = [];
    var categoriesSeen = new Set();
    var skippedRows = 0;
    
    for (var i = 0; i < data.length; i++) {
      var row = data[i];
      var source = row[0];
      var date = row[1];
      var description = row[2];
      var amountSpent = row[3];
      var category = row[4];
      var currencySpent = row[5];
      var amountBaseAUD = row[6];
      
      // Log first few rows for debugging
      if (i < 3) {
        Logger.log("Row " + (i + 2) + ": [" + source + ", " + date + ", " + description + ", " + amountSpent + ", '" + category + "', " + currencySpent + ", " + amountBaseAUD + "]");
      }
      
      // Skip rows without description or category
      if (!description || !category || description.toString().trim() === '' || category.toString().trim() === '') {
        skippedRows++;
        if (i < 10) { // Log first 10 skips for debugging
          Logger.log("SKIPPING Row " + (i + 2) + ": Missing description (" + !!description + ") or category (" + !!category + ")");
        }
        continue;
      }
      
      var descriptionStr = description.toString().trim();
      var categoryStr = category.toString().trim();
      
      // Track unique categories
      categoriesSeen.add(categoryStr);
      
      var transaction = {
        description: descriptionStr,
        Category: categoryStr
      };
      
      // Add amount if available
      if (amountSpent && typeof amountSpent === 'number') {
        transaction.amount = amountSpent;
        transaction.money_in = amountSpent >= 0;
      }
      
      transactions.push(transaction);
    }
    
    Logger.log("=== TRAINING DATA SUMMARY ===");
    Logger.log("Total data rows: " + data.length);
    Logger.log("Skipped rows: " + skippedRows);
    Logger.log("Valid transactions: " + transactions.length);
    Logger.log("Unique categories found: " + categoriesSeen.size);
    Logger.log("Categories: " + JSON.stringify(Array.from(categoriesSeen)));
    
    if (transactions.length < 10) {
      Logger.log("ERROR: Not enough valid training data (" + transactions.length + " transactions)");
      return {
        success: false,
        error: `Not enough valid training data. Found ${transactions.length} transactions, but at least 10 are required.`
      };
    }
    
    // Log first few transactions for debugging
    Logger.log("=== SAMPLE TRAINING TRANSACTIONS ===");
    for (var j = 0; j < Math.min(5, transactions.length); j++) {
      Logger.log("TX " + (j + 1) + ": " + JSON.stringify(transactions[j]));
    }
    
    // Get service configuration
    var serviceConfig = Config.getServiceConfig();
    Logger.log("Using service URL: " + serviceConfig.serviceUrl);
    
    // Prepare payload
    var payload = JSON.stringify({
      transactions: transactions,
      userId: Session.getEffectiveUser().getEmail()
    });
    
    Logger.log("=== TRAINING API REQUEST ===");
    Logger.log("Payload size: " + payload.length + " bytes");
    Logger.log("User ID: " + Session.getEffectiveUser().getEmail());
    Logger.log("Transaction count in payload: " + transactions.length);
    
    // Make API call
    var options = {
      method: "post",
      contentType: "application/json",
      headers: { "X-API-Key": serviceConfig.apiKey },
      payload: payload,
      muteHttpExceptions: true
    };
    
    var response = UrlFetchApp.fetch(serviceConfig.serviceUrl + "/train", options);
    var responseCode = response.getResponseCode();
    var responseText = response.getContentText();
    
    Logger.log("=== TRAINING API RESPONSE ===");
    Logger.log("Response code: " + responseCode);
    Logger.log("Response text: " + responseText);
    
    if (responseCode !== 200 && responseCode !== 202) {
      Logger.log("ERROR: Training API failed with code " + responseCode);
      throw new Error(`Training API returned error ${responseCode}: ${responseText}`);
    }
    
    var result = JSON.parse(responseText);
    
    if (result.error) {
      Logger.log("ERROR: Training API returned error in response: " + result.error);
      throw new Error(`Training API error: ${result.error}`);
    }
    
    Logger.log("=== TRAINING SUCCESS ===");
    Logger.log("Prediction ID: " + (result.prediction_id || "sync_training_" + Date.now()));
    
    return {
      success: true,
      predictionId: result.prediction_id || "sync_training_" + Date.now(),
      message: "Training started successfully",
      categoriesUsed: Array.from(categoriesSeen),
      transactionCount: transactions.length
    };
    
  } catch (error) {
    Logger.log("=== TRAINING ERROR ===");
    Logger.log("Error starting training for import: " + error.message);
    Logger.log("Stack trace: " + error.stack);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Start categorization of new transactions
 * @param {Array} processedData - Array of processed transaction objects
 * @return {Object} - Success/error result with prediction ID
 */
function startTransactionCategorization(processedData) {
  try {
    Logger.log("=== STARTING TRANSACTION CATEGORIZATION ===");
    Logger.log("Processing " + processedData.length + " transactions for categorization");
    
    // Prepare transactions for categorization
    var transactions = processedData.map(function(transaction, index) {
      var tx = {
        description: transaction["Description"] || ""
      };
      
      // Add amount if available
      var amountSpent = transaction["Amount Spent"];
      if (amountSpent && typeof amountSpent === 'number') {
        tx.amount = amountSpent;
        tx.money_in = amountSpent >= 0;
      }
      
      // Log first few transactions for debugging
      if (index < 3) {
        Logger.log("Categorization TX " + (index + 1) + ": " + JSON.stringify(tx));
      }
      
      return tx;
    });
    
    Logger.log("Prepared " + transactions.length + " transactions for categorization");
    
    // Get service configuration
    var serviceConfig = Config.getServiceConfig();
    
    // Prepare payload
    var payload = JSON.stringify({
      transactions: transactions
    });
    
    Logger.log("=== CATEGORIZATION API REQUEST ===");
    Logger.log("Payload size: " + payload.length + " bytes");
    Logger.log("Service URL: " + serviceConfig.serviceUrl);
    
    // Make API call
    var options = {
      method: "post",
      contentType: "application/json",
      headers: { "X-API-Key": serviceConfig.apiKey },
      payload: payload,
      muteHttpExceptions: true
    };
    
    var response = UrlFetchApp.fetch(serviceConfig.serviceUrl + "/classify", options);
    var responseCode = response.getResponseCode();
    var responseText = response.getContentText();
    
    Logger.log("=== CATEGORIZATION API RESPONSE ===");
    Logger.log("Response code: " + responseCode);
    Logger.log("Response text length: " + responseText.length + " characters");
    
    // Log first part of response for debugging
    var responsePreview = responseText.length > 1000 ? responseText.substring(0, 1000) + "..." : responseText;
    Logger.log("Response preview: " + responsePreview);
    
    if (responseCode !== 200 && responseCode !== 202) {
      Logger.log("ERROR: Categorization API failed with code " + responseCode);
      throw new Error(`Categorization API returned error ${responseCode}: ${responseText}`);
    }
    
    var result = JSON.parse(responseText);
    
    if (result.error) {
      Logger.log("ERROR: Categorization API returned error: " + result.error);
      throw new Error(`Categorization API error: ${result.error}`);
    }
    
    // If synchronous response, extract categories immediately
    if (result.results && Array.isArray(result.results)) {
      Logger.log("=== SYNCHRONOUS CATEGORIZATION RESULTS ===");
      Logger.log("Got " + result.results.length + " categorization results");
      
      var categories = result.results.map(function(item, index) {
        var category = item.predicted_category || item.Category || item.category || 'Uncategorized';
        if (index < 5) { // Log first 5 for debugging
          Logger.log("Result " + (index + 1) + ": " + item.narrative + " -> " + category);
        }
        return category;
      });
      
      Logger.log("Extracted categories: " + JSON.stringify(categories.slice(0, 10)));
      
      return {
        success: true,
        predictionId: "sync_categorize_" + Date.now(),
        message: "Categorization completed synchronously",
        categories: categories
      };
    }
    
    Logger.log("=== ASYNCHRONOUS CATEGORIZATION STARTED ===");
    Logger.log("Prediction ID: " + (result.prediction_id || "sync_categorize_" + Date.now()));
    
    return {
      success: true,
      predictionId: result.prediction_id || "sync_categorize_" + Date.now(),
      message: "Categorization started successfully"
    };
    
  } catch (error) {
    Logger.log("=== CATEGORIZATION ERROR ===");
    Logger.log("Error starting categorization: " + error.message);
    Logger.log("Stack trace: " + error.stack);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Check categorization status and retrieve results
 * @param {string} predictionId - The prediction ID to check
 * @return {Object} - Status result with categories if completed
 */
function checkCategorizationStatus(predictionId) {
  try {
    Logger.log("=== CHECKING CATEGORIZATION STATUS ===");
    Logger.log("Prediction ID: " + predictionId);
    
    // Handle sync completion dummy IDs
    if (predictionId.startsWith("sync_categorize_")) {
      Logger.log("Sync categorization detected - marking as completed");
      return {
        status: "completed",
        message: "Categorization completed successfully",
        categories: [] // Empty for sync dummy IDs
      };
    }
    
    var serviceConfig = Config.getServiceConfig();
    
    var options = {
      headers: { "X-API-Key": serviceConfig.apiKey },
      muteHttpExceptions: true
    };
    
    var response = UrlFetchApp.fetch(serviceConfig.serviceUrl + "/status/" + predictionId, options);
    var responseCode = response.getResponseCode();
    
    Logger.log("Status check response code: " + responseCode);
    
    if (responseCode !== 200) {
      Logger.log("Status not ready, continuing to poll...");
      return {
        status: "processing",
        message: "Categorization in progress..."
      };
    }
    
    var result = JSON.parse(response.getContentText());
    Logger.log("Status response: " + JSON.stringify(result));
    
    if (result.status === "completed" || result.status === "succeeded") {
      Logger.log("=== CATEGORIZATION COMPLETED ===");
      
      // Extract categories from result
      var categories = [];
      
      // Handle direct results (synchronous response)
      if (result.results && Array.isArray(result.results)) {
        Logger.log("Extracting categories from direct results");
        categories = result.results.map(function(item, index) {
          var category = item.predicted_category || item.Category || item.category || 'Uncategorized';
          if (index < 5) {
            Logger.log("Direct result " + (index + 1) + ": " + (item.narrative || item.description) + " -> " + category);
          }
          return category;
        });
      }
      // Handle result_data (asynchronous response)
      else if (result.result_data) {
        try {
          var resultData = typeof result.result_data === 'string' ? JSON.parse(result.result_data) : result.result_data;
          Logger.log("Extracting categories from result_data");
          
          // Find the results array in the response
          var results = resultData.results || resultData.Categories || resultData;
          
          if (Array.isArray(results)) {
            categories = results.map(function(item, index) {
              var category = item.predicted_category || item.Category || item.category || 'Uncategorized';
              if (index < 5) {
                Logger.log("Async result " + (index + 1) + ": " + (item.narrative || item.description) + " -> " + category);
              }
              return category;
            });
          }
        } catch (parseError) {
          Logger.log("Error parsing categorization results: " + parseError);
          categories = [];
        }
      }
      
      Logger.log("=== FINAL CATEGORY EXTRACTION ===");
      Logger.log("Extracted " + categories.length + " categories");
      Logger.log("Sample categories: " + JSON.stringify(categories.slice(0, 10)));
      
      // Check for categories that might cause validation issues
      var uniqueCategories = Array.from(new Set(categories));
      Logger.log("Unique categories returned: " + JSON.stringify(uniqueCategories));
      
      return {
        status: "completed",
        message: "Categorization completed successfully",
        categories: categories
      };
    }
    
    Logger.log("Categorization still in progress: " + result.status);
    return {
      status: result.status || "processing",
      message: result.message || "Categorization in progress..."
    };
    
  } catch (error) {
    Logger.log("=== CATEGORIZATION STATUS ERROR ===");
    Logger.log("Error checking categorization status: " + error.message);
    return {
      status: "processing",
      message: "Checking categorization status..."
    };
  }
}

/**
 * Legacy train model function
 * @param {Object} config - Training configuration
 */
function trainModel(config) {
  try {
    Logger.log("Starting model training with config: " + JSON.stringify(config));
    
    var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = spreadsheet.getActiveSheet();
    
    var lastRow = sheet.getLastRow();
    if (lastRow < config.startRow) {
      throw new Error("No data found starting from row " + config.startRow);
    }
    
    var narrativeCol = config.narrativeCol;
    var categoryCol = config.categoryCol;
    var amountCol = config.amountCol;
    
    var narrativeColIndex = narrativeCol.charCodeAt(0) - 65;
    var categoryColIndex = categoryCol.charCodeAt(0) - 65;
    var amountColIndex = amountCol ? amountCol.charCodeAt(0) - 65 : null;
    
    var range = sheet.getRange(config.startRow, 1, lastRow - config.startRow + 1, sheet.getLastColumn());
    var data = range.getValues();
    
    var transactions = [];
    
    for (var i = 0; i < data.length; i++) {
      var row = data[i];
      var narrative = row[narrativeColIndex];
      var category = row[categoryColIndex];
      var amount = amountColIndex !== null ? row[amountColIndex] : null;
      
      if (narrative && category) {
        var transaction = {
          description: narrative.toString().trim(),
          Category: category.toString().trim()
        };
        
        if (amount && typeof amount === 'number') {
          transaction.amount = amount;
          transaction.money_in = amount >= 0;
        }
        
        transactions.push(transaction);
      }
    }
    
    if (transactions.length === 0) {
      throw new Error("No valid transactions found for training");
    }
    
    Logger.log("Prepared " + transactions.length + " transactions for training");
    
    var serviceConfig = Config.getServiceConfig();
    
    var payload = JSON.stringify({
      transactions: transactions,
      userId: Session.getEffectiveUser().getEmail()
    });
    
    var options = {
      method: "post",
      contentType: "application/json",
      headers: { "X-API-Key": serviceConfig.apiKey },
      payload: payload,
      muteHttpExceptions: true
    };
    
    var response = UrlFetchApp.fetch(serviceConfig.serviceUrl + "/train", options);
    var responseCode = response.getResponseCode();
    var responseText = response.getContentText();
    
    Logger.log("Training API response: " + responseCode + " - " + responseText);
    
    if (responseCode !== 200 && responseCode !== 202) {
      throw new Error("Training failed with status " + responseCode + ": " + responseText);
    }
    
    var result = JSON.parse(responseText);
    
    if (result.error) {
      throw new Error("Training API error: " + result.error);
    }
    
    if (result.prediction_id) {
      showPollingDialog();
      return result.prediction_id;
    } else {
      Logger.log("Training completed successfully");
      return "Training completed successfully";
    }
    
  } catch (error) {
    Logger.log("Error in trainModel: " + error.message);
    throw error;
  }
}

/**
 * Legacy categorize transactions function
 * @param {Object} config - Categorization configuration
 */
function categoriseTransactions(config) {
  try {
    Logger.log("Starting categorisation with config: " + JSON.stringify(config));
    
    var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = spreadsheet.getActiveSheet();
    
    var lastRow = sheet.getLastRow();
    var endRow = config.endRow || lastRow;
    
    if (lastRow < config.startRow) {
      throw new Error("No data found starting from row " + config.startRow);
    }
    
    var descriptionCol = config.descriptionCol;
    var categoryCol = config.categoryCol;
    var amountCol = config.amountCol;
    
    var descriptionColIndex = descriptionCol.charCodeAt(0) - 65;
    var categoryColIndex = categoryCol.charCodeAt(0) - 65;
    var amountColIndex = amountCol ? amountCol.charCodeAt(0) - 65 : null;
    
    var range = sheet.getRange(config.startRow, 1, endRow - config.startRow + 1, sheet.getLastColumn());
    var data = range.getValues();
    
    var transactions = [];
    
    for (var i = 0; i < data.length; i++) {
      var row = data[i];
      var description = row[descriptionColIndex];
      var amount = amountColIndex !== null ? row[amountColIndex] : null;
      
      if (description) {
        var transaction = {
          description: description.toString().trim()
        };
        
        if (amount && typeof amount === 'number') {
          transaction.amount = amount;
          transaction.money_in = amount >= 0;
        }
        
        transactions.push(transaction);
      }
    }
    
    if (transactions.length === 0) {
      throw new Error("No valid transactions found for categorisation");
    }
    
    Logger.log("Prepared " + transactions.length + " transactions for categorisation");
    
    var serviceConfig = Config.getServiceConfig();
    
    var payload = JSON.stringify({
      transactions: transactions
    });
    
    var options = {
      method: "post",
      contentType: "application/json",
      headers: { "X-API-Key": serviceConfig.apiKey },
      payload: payload,
      muteHttpExceptions: true
    };
    
    var response = UrlFetchApp.fetch(serviceConfig.serviceUrl + "/classify", options);
    var responseCode = response.getResponseCode();
    var responseText = response.getContentText();
    
    Logger.log("Categorisation API response: " + responseCode + " - " + responseText);
    
    if (responseCode !== 200 && responseCode !== 202) {
      throw new Error("Categorisation failed with status " + responseCode + ": " + responseText);
    }
    
    var result = JSON.parse(responseText);
    
    if (result.error) {
      throw new Error("Categorisation API error: " + result.error);
    }
    
    if (result.prediction_id) {
      showPollingDialog();
      return result.prediction_id;
    } else if (result.results) {
      // Synchronous response - write results immediately
      var categories = result.results.map(function(item) {
        return item.predicted_category || item.Category || 'Uncategorized';
      });
      
      // Write categories to sheet
      var categoryRange = sheet.getRange(config.startRow, categoryColIndex + 1, categories.length, 1);
      var categoryValues = categories.map(function(cat) { return [cat]; });
      categoryRange.setValues(categoryValues);
      
      Logger.log("Categories written to sheet successfully");
      return "Categorisation completed successfully";
    }
    
  } catch (error) {
    Logger.log("Error in categoriseTransactions: " + error.message);
    throw error;
  }
} 