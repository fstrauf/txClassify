// Add menu to the spreadsheet
function onOpen() {
  var ui = SpreadsheetApp.getUi();
  ui.createMenu('Transaction Classifier')
    .addItem('Setup Service', 'setupService')
    .addItem('Train Model', 'trainModel')
    .addItem('Classify New Transactions', 'showClassifyDialog')
    .addToUi();
}

// Show dialog to select columns for classification
function showClassifyDialog() {
  var html = HtmlService.createHtmlOutput(`
    <style>
      body { font-family: Arial, sans-serif; padding: 20px; }
      .form-group { margin-bottom: 15px; }
      label { display: block; margin-bottom: 5px; }
      select { width: 100%; padding: 5px; }
      button { padding: 8px 15px; background: #4285f4; color: white; border: none; border-radius: 3px; cursor: pointer; }
    </style>
    <div class="form-group">
      <label>Description Column:</label>
      <select id="descriptionCol">
        <option value="A">A</option>
        <option value="B">B</option>
        <option value="C" selected>C</option>
        <option value="D">D</option>
        <option value="E">E</option>
      </select>
    </div>
    <div class="form-group">
      <label>Category Column (where to write results):</label>
      <select id="categoryCol">
        <option value="D" selected>D</option>
        <option value="E">E</option>
        <option value="F">F</option>
        <option value="G">G</option>
        <option value="H">H</option>
      </select>
    </div>
    <div class="form-group">
      <label>Start Row:</label>
      <input type="number" id="startRow" value="2" min="2" style="width: 100%; padding: 5px;">
    </div>
    <button onclick="submitForm()">Classify Transactions</button>
    <script>
      function submitForm() {
        var result = {
          descriptionCol: document.getElementById('descriptionCol').value,
          categoryCol: document.getElementById('categoryCol').value,
          startRow: document.getElementById('startRow').value
        };
        google.script.run
          .withSuccessHandler(function() { google.script.host.close(); })
          .classifyTransactions(result);
      }
    </script>
  `)
    .setWidth(400)
    .setHeight(300);
  
  SpreadsheetApp.getUi().showModalDialog(html, 'Classify Transactions');
}

// Setup function to configure service URL and API key
function setupService() {
  var ui = SpreadsheetApp.getUi();
  var properties = PropertiesService.getScriptProperties();
  
  // Get service URL
  var serviceUrl = ui.prompt(
    'Setup Classification Service',
    'Enter the classification service URL:',
    ui.ButtonSet.OK_CANCEL
  );
  if (serviceUrl.getSelectedButton() !== ui.Button.OK) return;
  
  // Get API key
  var apiKey = ui.prompt(
    'Setup API Key',
    'Enter your API key (or leave blank to generate a new one):',
    ui.ButtonSet.OK_CANCEL
  );
  if (apiKey.getSelectedButton() !== ui.Button.OK) return;
  
  // If no API key provided, generate one
  var key = apiKey.getResponseText().trim();
  if (!key) {
    key = Utilities.getUuid();
    ui.alert('Generated new API key', 'Your API key is: ' + key + '\n\nPlease save this key somewhere safe.', ui.ButtonSet.OK);
  }
  
  // Store both values
  properties.setProperties({
    'CLASSIFICATION_SERVICE_URL': serviceUrl.getResponseText(),
    'API_KEY': key
  });
  
  ui.alert('Setup completed successfully!');
}

// Helper function to get stored properties
function getServiceConfig() {
  var properties = PropertiesService.getScriptProperties();
  var serviceUrl = properties.getProperty('CLASSIFICATION_SERVICE_URL');
  var apiKey = properties.getProperty('API_KEY');
  
  if (!serviceUrl || !apiKey) {
    throw new Error('Service not configured. Please use "Setup Service" first.');
  }
  
  return { serviceUrl, apiKey };
}

// Helper function to update status in sheet
function updateStatus(message) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var logSheet = ss.getSheetByName('Log');
  
  // Create Log sheet if it doesn't exist
  if (!logSheet) {
    logSheet = ss.insertSheet('Log');
    // Add headers
    logSheet.getRange('A1:C1').setValues([['Timestamp', 'Status', 'Details']]);
    logSheet.setFrozenRows(1);
    // Set column widths
    logSheet.setColumnWidth(1, 180);  // Timestamp
    logSheet.setColumnWidth(2, 150);  // Status
    logSheet.setColumnWidth(3, 400);  // Details
  }
  
  // Get current timestamp
  var timestamp = new Date().toLocaleString();
  
  // Determine status and details
  var status = "INFO";
  var details = message;
  
  if (message.toLowerCase().includes("error")) {
    status = "ERROR";
  } else if (message.toLowerCase().includes("completed")) {
    status = "SUCCESS";
  } else if (message.toLowerCase().includes("progress") || message.toLowerCase().includes("processing")) {
    status = "PROCESSING";
  }
  
  // Insert new row after header
  logSheet.insertRowAfter(1);
  
  // Write the log entry
  logSheet.getRange('A2:C2').setValues([[timestamp, status, details]]);
  
  // Color coding
  var statusCell = logSheet.getRange('B2');
  switch(status) {
    case "ERROR":
      statusCell.setBackground('#ffcdd2');  // Light red
      break;
    case "SUCCESS":
      statusCell.setBackground('#c8e6c9');  // Light green
      break;
    case "PROCESSING":
      statusCell.setBackground('#fff9c4');  // Light yellow
      break;
    default:
      statusCell.setBackground('#ffffff');  // White
  }
  
  // Keep only last 100 entries
  var lastRow = logSheet.getLastRow();
  if (lastRow > 101) {  // 1 header row + 100 log entries
    logSheet.deleteRows(102, lastRow - 101);
  }
  
  // Make sure the log is visible
  logSheet.autoResizeColumns(1, 3);
}

// Helper function to poll training/classification status
function pollStatus(predictionId, config) {
  var ui = SpreadsheetApp.getUi();
  var maxAttempts = 30;  // 30 seconds timeout
  var attempt = 0;
  
  while (attempt < maxAttempts) {
    try {
      var options = {
        method: 'get',
        headers: {
          'X-API-Key': config.apiKey
        },
        muteHttpExceptions: true
      };
      
      var response = UrlFetchApp.fetch(config.serviceUrl + '/status/' + predictionId, options);
      var result = JSON.parse(response.getContentText());
      
      if (result.status === "completed") {
        return result;
      } else if (result.status === "failed") {
        throw new Error(result.error || "Operation failed");
      }
      
      // Update status message
      updateStatus(result.message || "Processing...");
      
    } catch (error) {
      Logger.log("Error polling status: " + error);
    }
    
    // Wait 1 second before next attempt
    Utilities.sleep(1000);
    attempt++;
  }
  
  throw new Error("Operation timed out");
}

// Train model with existing categorized transactions
function trainModel() {
  var ui = SpreadsheetApp.getUi();
  
  try {
    updateStatus("Starting training...");
    var config = getServiceConfig();
    var sheet = SpreadsheetApp.getActiveSheet();
    var lastRow = sheet.getLastRow();
    var data = sheet.getRange("A2:D" + lastRow).getValues();
    
    // Filter out empty rows and prepare training data
    var transactions = data
      .filter(row => row[2] && row[3]) // Narrative and Category columns must have values
      .map(row => ({
        Narrative: row[2],
        Category: row[3]
      }));
    
    if (transactions.length === 0) {
      updateStatus("Error: No training data found");
      ui.alert('No training data found. Please ensure you have transactions with categories.');
      return;
    }
    
    updateStatus("Processing " + transactions.length + " transactions...");
    
    // Call training endpoint
    var options = {
      method: 'post',
      contentType: 'application/json',
      headers: {
        'X-API-Key': config.apiKey
      },
      payload: JSON.stringify({ transactions: transactions }),
      muteHttpExceptions: true
    };
    
    var response = UrlFetchApp.fetch(config.serviceUrl + '/train', options);
    var result = JSON.parse(response.getContentText());
    
    if (response.getResponseCode() !== 200) {
      updateStatus("Error: Training failed");
      throw new Error('Training error: ' + response.getContentText());
    }
    
    // Check if we got a prediction ID
    if (result.prediction_id) {
      updateStatus("Training in progress...");
      
      // Create a trigger to check status every minute
      var trigger = ScriptApp.newTrigger('checkTrainingStatus')
        .timeBased()
        .everyMinutes(1)
        .create();
      
      // Store prediction ID and trigger ID in user properties
      var userProperties = PropertiesService.getUserProperties();
      userProperties.setProperties({
        'PREDICTION_ID': result.prediction_id,
        'TRIGGER_ID': trigger.getUniqueId(),
        'START_TIME': new Date().getTime().toString()
      });
      
      ui.alert('Training has started! The sheet will update automatically when training is complete.\n\nYou can close this window.');
      return;
    }
    
    updateStatus("Training completed successfully!");
    ui.alert('Training completed successfully!\n\nProcessed: ' + transactions.length + ' transactions');
    
  } catch (error) {
    updateStatus("Error: " + error.toString());
    ui.alert('Error: ' + error.toString());
  }
}

function checkTrainingStatus() {
  try {
    var userProperties = PropertiesService.getUserProperties();
    var predictionId = userProperties.getProperty('PREDICTION_ID');
    var triggerId = userProperties.getProperty('TRIGGER_ID');
    var startTime = parseInt(userProperties.getProperty('START_TIME'));
    
    if (!predictionId || !triggerId) {
      // Clean up if no prediction ID or trigger ID
      if (triggerId) {
        ScriptApp.getProjectTrigger(triggerId).delete();
      }
      return;
    }
    
    // Check if we've been running for more than 30 minutes
    if (new Date().getTime() - startTime > 30 * 60 * 1000) {
      updateStatus("Error: Training timed out after 30 minutes");
      ScriptApp.getProjectTrigger(triggerId).delete();
      userProperties.deleteAllProperties();
      return;
    }
    
    var config = getServiceConfig();
    var response = UrlFetchApp.fetch(config.serviceUrl + '/status/' + predictionId, {
      headers: { 'X-API-Key': config.apiKey },
      muteHttpExceptions: true
    });
    
    var result = JSON.parse(response.getContentText());
    
    if (result.status === "completed") {
      updateStatus("Training completed successfully!");
      // Clean up
      ScriptApp.getProjectTrigger(triggerId).delete();
      userProperties.deleteAllProperties();
      return;
    } else if (result.status === "failed") {
      updateStatus("Error: Training failed - " + (result.error || "Unknown error"));
      // Clean up
      ScriptApp.getProjectTrigger(triggerId).delete();
      userProperties.deleteAllProperties();
      return;
    }
    
    // Still processing, update status
    updateStatus("Training in progress... " + (result.message || ""));
    
  } catch (error) {
    Logger.log("Error checking training status: " + error);
    // Clean up on error
    var triggerId = userProperties.getProperty('TRIGGER_ID');
    if (triggerId) {
      ScriptApp.getProjectTrigger(triggerId).delete();
    }
    userProperties.deleteAllProperties();
    updateStatus("Error: " + error.toString());
  }
}

// Main classification function
function classifyTransactions(config) {
  var sheet = SpreadsheetApp.getActiveSheet();
  var ui = SpreadsheetApp.getUi();
  
  try {
    updateStatus("Starting classification...");
    var serviceConfig = getServiceConfig();
    
    // Get all descriptions from the specified column
    var lastRow = sheet.getLastRow();
    var descriptionRange = sheet.getRange(config.descriptionCol + config.startRow + ":" + config.descriptionCol + lastRow);
    var descriptions = descriptionRange.getValues();
    
    // Filter out empty descriptions
    var transactions = descriptions
      .filter(row => row[0] && row[0].toString().trim() !== '')
      .map(row => ({
        Narrative: row[0]
      }));
    
    if (transactions.length === 0) {
      updateStatus("Error: No transactions found");
      ui.alert('No transactions found to classify');
      return;
    }
    
    updateStatus("Processing " + transactions.length + " transactions...");
    
    // Call classification service
    var options = {
      method: 'post',
      contentType: 'application/json',
      headers: {
        'X-API-Key': serviceConfig.apiKey
      },
      payload: JSON.stringify({ transactions: transactions }),
      muteHttpExceptions: true
    };
    
    var response = UrlFetchApp.fetch(serviceConfig.serviceUrl + '/classify', options);
    var result = JSON.parse(response.getContentText());
    
    if (response.getResponseCode() !== 200) {
      updateStatus("Error: Classification failed");
      throw new Error('Classification service error: ' + response.getContentText());
    }
    
    // Check if we got a prediction ID
    if (result.prediction_id) {
      updateStatus("Classification in progress...");
      
      // Store configuration for status checking
      var userProperties = PropertiesService.getUserProperties();
      userProperties.setProperties({
        'PREDICTION_ID': result.prediction_id,
        'OPERATION_TYPE': 'classify',
        'START_TIME': new Date().getTime().toString(),
        'CONFIG': JSON.stringify({
          categoryCol: config.categoryCol,
          startRow: config.startRow
        })
      });
      
      // Create a trigger to check status every minute
      var trigger = ScriptApp.newTrigger('checkOperationStatus')
        .timeBased()
        .everyMinutes(1)
        .create();
      
      userProperties.setProperty('TRIGGER_ID', trigger.getUniqueId());
      
      ui.alert('Classification has started! The sheet will update automatically when complete.\n\nYou can close this window.');
      return;
    }
  } catch (error) {
    updateStatus("Error: " + error.toString());
    ui.alert('Error: ' + error.toString());
  }
}

function checkOperationStatus() {
  var userProperties = PropertiesService.getUserProperties();
  var predictionId = userProperties.getProperty('PREDICTION_ID');
  var triggerId = userProperties.getProperty('TRIGGER_ID');
  var operationType = userProperties.getProperty('OPERATION_TYPE');
  var startTime = parseInt(userProperties.getProperty('START_TIME'));
  var config = userProperties.getProperty('CONFIG') ? JSON.parse(userProperties.getProperty('CONFIG')) : null;
  
  if (!predictionId || !triggerId) {
    // Clean up if no prediction ID or trigger ID
    if (triggerId) {
      ScriptApp.getProjectTrigger(triggerId).delete();
    }
    return;
  }
  
  try {
    // Check if we've been running for more than 30 minutes
    if (new Date().getTime() - startTime > 30 * 60 * 1000) {
      updateStatus(`Error: ${operationType} timed out after 30 minutes`);
      ScriptApp.getProjectTrigger(triggerId).delete();
      userProperties.deleteAllProperties();
      return;
    }
    
    var serviceConfig = getServiceConfig();
    var response = UrlFetchApp.fetch(serviceConfig.serviceUrl + '/status/' + predictionId, {
      headers: { 'X-API-Key': serviceConfig.apiKey },
      muteHttpExceptions: true
    });
    
    var result = JSON.parse(response.getContentText());
    Logger.log("Status response: " + JSON.stringify(result));
    
    if (result.status === "completed") {
      if (operationType === "classify" && result.result && config) {
        // Process classification results
        var sheet = SpreadsheetApp.getActiveSheet();
        var results = result.result;
        
        Logger.log("Processing results: " + JSON.stringify(results));
        
        // Write categories back to sheet
        updateStatus("Writing results to sheet...");
        var categories = results.map(r => [r.predicted_category]);
        Logger.log("Categories to write: " + JSON.stringify(categories));
        
        var categoryRange = sheet.getRange(config.categoryCol + config.startRow + ":" + config.categoryCol + (parseInt(config.startRow) + categories.length - 1));
        categoryRange.setValues(categories);
        
        // Add confidence scores in next column
        var confidenceCol = String.fromCharCode(config.categoryCol.charCodeAt(0) + 1);
        var confidences = results.map(r => [r.similarity_score]);
        var confidenceRange = sheet.getRange(confidenceCol + config.startRow + ":" + confidenceCol + (parseInt(config.startRow) + confidences.length - 1));
        confidenceRange.setValues(confidences)
          .setNumberFormat("0.00%");  // Format as percentage
        
        // Add headers if needed
        if (config.startRow === "2") {
          sheet.getRange(config.categoryCol + "1").setValue("Category");
          sheet.getRange(confidenceCol + "1").setValue("Confidence");
        }
        
        updateStatus("Classification completed successfully!");
      }
      
      // Clean up
      ScriptApp.getProjectTrigger(triggerId).delete();
      userProperties.deleteAllProperties();
      return;
      
    } else if (result.status === "failed") {
      updateStatus(`Error: ${operationType} failed - ` + (result.error || "Unknown error"));
      // Clean up
      ScriptApp.getProjectTrigger(triggerId).delete();
      userProperties.deleteAllProperties();
      return;
    }
    
    // Still processing, update status
    updateStatus(operationType + " in progress... " + (result.message || ""));
    
  } catch (error) {
    Logger.log(`Error checking ${operationType} status: ` + error);
    // Clean up on error
    if (triggerId) {
      ScriptApp.getProjectTrigger(triggerId).delete();
    }
    userProperties.deleteAllProperties();
    updateStatus("Error: " + error.toString());
  }
} 