// Service configuration
const CLASSIFICATION_SERVICE_URL = 'https://txclassify.onrender.com';

// Add menu to the spreadsheet
function onOpen() {
  var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  var menuEntries = [
    {name: "Configure API Key", functionName: "setupApiKey"},
    {name: "Train Model", functionName: "showTrainingDialog"},
    {name: "Classify New Transactions", functionName: "showClassifyDialog"}
  ];
  spreadsheet.addMenu("Transaction Classifier", menuEntries);
}

// Show dialog to select columns for classification
function showClassifyDialog() {
  var html = HtmlService.createHtmlOutput(`
    <style>
      body { font-family: Arial, sans-serif; padding: 20px; }
      .form-group { margin-bottom: 15px; }
      label { display: block; margin-bottom: 5px; }
      select { width: 100%; padding: 5px; }
      input { width: 100%; padding: 5px; }
      button { 
        padding: 8px 15px; 
        background: #4285f4; 
        color: white; 
        border: none; 
        border-radius: 3px; 
        cursor: pointer;
        position: relative;
        min-width: 120px;
      }
      button:disabled {
        background: #ccc;
        cursor: not-allowed;
      }
      .help-text { font-size: 12px; color: #666; margin-top: 4px; }
      .error { color: red; margin-top: 10px; display: none; }
      .spinner {
        display: none;
        width: 20px;
        height: 20px;
        border: 2px solid #f3f3f3;
        border-top: 2px solid #3498db;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        position: absolute;
        right: 10px;
        top: 50%;
        transform: translateY(-50%);
      }
      @keyframes spin {
        0% { transform: translateY(-50%) rotate(0deg); }
        100% { transform: translateY(-50%) rotate(360deg); }
      }
      .button-text { display: inline-block; }
      .processing-text { display: none; }
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
      <div class="help-text">Column containing transaction descriptions</div>
    </div>
    <div class="form-group">
      <label>Category Column (where to write results):</label>
      <select id="categoryCol">
        <option value="D">D</option>
        <option value="E" selected>E</option>
        <option value="F">F</option>
        <option value="G">G</option>
        <option value="H">H</option>
      </select>
      <div class="help-text">Column where categories will be written</div>
    </div>
    <div class="form-group">
      <label>Start Row:</label>
      <input type="number" id="startRow" value="1" min="1" style="width: 100%; padding: 5px;">
      <div class="help-text">First row of data to classify</div>
    </div>
    <div id="error" class="error"></div>
    <button onclick="submitForm()" id="submitBtn">
      <span class="button-text">Classify Transactions</span>
      <span class="processing-text">Processing...</span>
      <div class="spinner"></div>
    </button>
    <script>
      function submitForm() {
        var descriptionCol = document.getElementById('descriptionCol').value;
        var categoryCol = document.getElementById('categoryCol').value;
        var startRow = document.getElementById('startRow').value;
        var errorDiv = document.getElementById('error');
        var submitBtn = document.getElementById('submitBtn');
        var spinner = document.querySelector('.spinner');
        var buttonText = document.querySelector('.button-text');
        var processingText = document.querySelector('.processing-text');
        
        // Validate inputs
        if (!descriptionCol || !categoryCol || !startRow) {
          errorDiv.textContent = 'Please fill in all fields';
          errorDiv.style.display = 'block';
          return;
        }
        
        // Ensure start row is at least 1
        startRow = parseInt(startRow);
        if (isNaN(startRow) || startRow < 1) {
          errorDiv.textContent = 'Start row must be 1 or higher';
          errorDiv.style.display = 'block';
          return;
        }
        
        // Clear any previous errors
        errorDiv.style.display = 'none';
        
        // Show processing state
        submitBtn.disabled = true;
        spinner.style.display = 'block';
        buttonText.style.display = 'none';
        processingText.style.display = 'inline-block';
        
        var config = {
          descriptionCol: descriptionCol,
          categoryCol: categoryCol,
          startRow: startRow.toString()
        };
        
        google.script.run
          .withSuccessHandler(function() { 
            google.script.host.close(); 
          })
          .withFailureHandler(function(error) {
            errorDiv.textContent = error.message || 'An error occurred';
            errorDiv.style.display = 'block';
            // Reset button state
            submitBtn.disabled = false;
            spinner.style.display = 'none';
            buttonText.style.display = 'inline-block';
            processingText.style.display = 'none';
          })
          .classifyTransactions(config);
      }
      
      // Set default value and min for start row
      window.onload = function() {
        var startRowInput = document.getElementById('startRow');
        startRowInput.value = "1";
        startRowInput.min = "1";
      }
    </script>
  `)
    .setWidth(400)
    .setHeight(350);
  
  SpreadsheetApp.getUi().showModalDialog(html, 'Classify Transactions');
}

// Show dialog to select columns for training
function showTrainingDialog() {
  // Get current sheet to determine default columns
  var sheet = SpreadsheetApp.getActiveSheet();
  var headers = sheet.getRange("A1:Z1").getValues()[0];
  
  // Find default column indices
  var narrativeColDefault = columnToLetter(headers.indexOf("Narrative") + 1);
  var categoryColDefault = columnToLetter(headers.indexOf("Category") + 1);
  
  // Create column options
  var columnOptions = headers.map((header, index) => {
    if (header) {
      return `<option value="${columnToLetter(index + 1)}"${columnToLetter(index + 1) === narrativeColDefault ? ' selected' : ''}>${columnToLetter(index + 1)} (${header})</option>`;
    }
    return '';
  }).filter(Boolean).join('\n');

  var html = HtmlService.createHtmlOutput(`
    <style>
      body { font-family: Arial, sans-serif; padding: 20px; }
      .form-group { margin-bottom: 15px; }
      label { display: block; margin-bottom: 5px; }
      select { width: 100%; padding: 5px; }
      button { 
        padding: 8px 15px; 
        background: #4285f4; 
        color: white; 
        border: none; 
        border-radius: 3px; 
        cursor: pointer;
        position: relative;
        min-width: 120px;
      }
      button:disabled {
        background: #ccc;
        cursor: not-allowed;
      }
      .help-text { font-size: 12px; color: #666; margin-top: 4px; }
      .error { color: red; margin-top: 10px; display: none; }
      .spinner {
        display: none;
        width: 20px;
        height: 20px;
        border: 2px solid #f3f3f3;
        border-top: 2px solid #3498db;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        position: absolute;
        right: 10px;
        top: 50%;
        transform: translateY(-50%);
      }
      @keyframes spin {
        0% { transform: translateY(-50%) rotate(0deg); }
        100% { transform: translateY(-50%) rotate(360deg); }
      }
      .button-text { display: inline-block; }
      .processing-text { display: none; }
    </style>
    <div class="form-group">
      <label>Description Column:</label>
      <select id="narrativeCol">
        ${columnOptions}
      </select>
      <div class="help-text">Select the column containing transaction descriptions</div>
    </div>
    <div class="form-group">
      <label>Category Column:</label>
      <select id="categoryCol">
        ${headers.map((header, index) => {
          if (header) {
            return `<option value="${columnToLetter(index + 1)}"${columnToLetter(index + 1) === categoryColDefault ? ' selected' : ''}>${columnToLetter(index + 1)} (${header})</option>`;
          }
          return '';
        }).filter(Boolean).join('\n')}
      </select>
      <div class="help-text">Select the column containing categories for training</div>
    </div>
    <div class="form-group">
      <label>Start Row:</label>
      <input type="number" id="startRow" value="2" min="2" style="width: 100%; padding: 5px;">
      <div class="help-text">First row of data (usually 2 to skip headers)</div>
    </div>
    <div id="error" class="error"></div>
    <button onclick="submitForm()" id="submitBtn">
      <span class="button-text">Start Training</span>
      <span class="processing-text">Processing...</span>
      <div class="spinner"></div>
    </button>
    <script>
      function submitForm() {
        var narrativeCol = document.getElementById('narrativeCol').value;
        var categoryCol = document.getElementById('categoryCol').value;
        var startRow = document.getElementById('startRow').value;
        var errorDiv = document.getElementById('error');
        var submitBtn = document.getElementById('submitBtn');
        var spinner = document.querySelector('.spinner');
        var buttonText = document.querySelector('.button-text');
        var processingText = document.querySelector('.processing-text');
        
        // Validate inputs
        if (!narrativeCol || !categoryCol || !startRow) {
          errorDiv.textContent = 'Please fill in all fields';
          errorDiv.style.display = 'block';
          return;
        }
        
        // Clear any previous errors
        errorDiv.style.display = 'none';
        
        var config = {
          narrativeCol: narrativeCol,
          categoryCol: categoryCol,
          startRow: startRow
        };
        
        // Show processing state
        submitBtn.disabled = true;
        spinner.style.display = 'block';
        buttonText.style.display = 'none';
        processingText.style.display = 'inline-block';
        
        google.script.run
          .withSuccessHandler(function() {
            google.script.host.close();
          })
          .withFailureHandler(function(error) {
            errorDiv.textContent = error.message || 'An error occurred';
            errorDiv.style.display = 'block';
            // Reset button state
            submitBtn.disabled = false;
            spinner.style.display = 'none';
            buttonText.style.display = 'inline-block';
            processingText.style.display = 'none';
          })
          .trainModel(config);
      }
    </script>
  `)
    .setWidth(400)
    .setHeight(350);
  
  SpreadsheetApp.getUi().showModalDialog(html, 'Train Model');
}

// Setup function to configure API key
function setupApiKey() {
  var ui = SpreadsheetApp.getUi();
  var properties = PropertiesService.getScriptProperties();
  
  // Show instructions first
  var instructions = ui.alert(
    'API Key Setup',
    '1. Go to expensesorted.com and sign up for an account\n' +
    '2. After signing up, go to your account settings to get your API key\n' +
    '3. Copy the API key and paste it in the next dialog\n\n' +
    'Do you want to continue?',
    ui.ButtonSet.OK_CANCEL
  );
  
  if (instructions.getSelectedButton() !== ui.Button.OK) return;
  
  // Get API key
  var apiKey = ui.prompt(
    'Enter API Key',
    'Please enter your API key from expensesorted.com:',
    ui.ButtonSet.OK_CANCEL
  );
  
  if (apiKey.getSelectedButton() !== ui.Button.OK) return;
  
  // Validate API key is provided
  var key = apiKey.getResponseText().trim();
  if (!key) {
    ui.alert('Error', 'API key is required. Please get your API key from expensesorted.com', ui.ButtonSet.OK);
    return;
  }
  
  // Store the API key
  properties.setProperty('API_KEY', key);
  
  ui.alert('Setup completed successfully!');
}

// Helper function to get stored properties
function getServiceConfig() {
  var properties = PropertiesService.getScriptProperties();
  var apiKey = properties.getProperty('API_KEY');
  
  if (!apiKey) {
    throw new Error('API key not configured. Please go to expensesorted.com to get your API key, then use "Configure API Key" to set it up.');
  }
  
  return { 
    serviceUrl: CLASSIFICATION_SERVICE_URL,
    apiKey: apiKey,
    userId: Session.getEffectiveUser().getEmail()  // Use the user's email as userId
  };
}

// Helper function to update status in sheet
function updateStatus(message, additionalDetails = '') {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var logSheet = ss.getSheetByName('Log');
  
  // Create Log sheet if it doesn't exist
  if (!logSheet) {
    logSheet = ss.insertSheet('Log');
    // Add headers
    logSheet.getRange('A1:D1').setValues([['Timestamp', 'Status', 'Message', 'Details']]);
    logSheet.setFrozenRows(1);
    // Set column widths
    logSheet.setColumnWidth(1, 180);  // Timestamp
    logSheet.setColumnWidth(2, 100);  // Status
    logSheet.setColumnWidth(3, 300);  // Message
    logSheet.setColumnWidth(4, 400);  // Details
    
    // Format headers
    var headerRange = logSheet.getRange('A1:D1');
    headerRange.setBackground('#f3f3f3')
               .setFontWeight('bold')
               .setHorizontalAlignment('center');
  }
  
  // Get current timestamp
  var timestamp = new Date().toLocaleString();
  
  // Determine status and details
  var status = "INFO";
  if (message.toLowerCase().includes("error")) {
    status = "ERROR";
  } else if (message.toLowerCase().includes("completed") || message.toLowerCase().includes("success")) {
    status = "SUCCESS";
  } else if (message.toLowerCase().includes("progress") || message.toLowerCase().includes("processing")) {
    status = "PROCESSING";
  }
  
  // Get active sheet name and additional context
  var activeSheet = SpreadsheetApp.getActiveSheet().getName();
  var contextDetails = additionalDetails || `Active Sheet: ${activeSheet}`;
  
  // Insert new row after header
  logSheet.insertRowAfter(1);
  
  // Write the log entry
  logSheet.getRange('A2:D2').setValues([[
    timestamp,
    status,
    message,
    contextDetails
  ]]);
  
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
  
  // Format the new row
  logSheet.getRange('A2:D2')
    .setHorizontalAlignment('left')
    .setVerticalAlignment('middle')
    .setWrap(true);
  
  // Keep only last 100 entries
  var lastRow = logSheet.getLastRow();
  if (lastRow > 101) {  // 1 header row + 100 log entries
    logSheet.deleteRows(102, lastRow - 101);
  }
  
  // Make sure the Log sheet is visible
  logSheet.autoResizeColumns(1, 4);
  
  // Show the Log sheet if it's hidden
  if (logSheet.isSheetHidden()) {
    logSheet.showSheet();
  }
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

// Helper function to manage settings sheet
function getSettingsSheet() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var settingsSheet = ss.getSheetByName('Settings');
  
  // Create Settings sheet if it doesn't exist
  if (!settingsSheet) {
    settingsSheet = ss.insertSheet('Settings');
    // Add headers
    settingsSheet.getRange('A1:B1').setValues([['Setting', 'Value']]);
    settingsSheet.setFrozenRows(1);
    // Set column widths
    settingsSheet.setColumnWidth(1, 200);  // Setting name
    settingsSheet.setColumnWidth(2, 300);  // Value
    // Hide the sheet
    settingsSheet.hideSheet();
  }
  
  return settingsSheet;
}

// Helper function to update settings
function updateSetting(settingName, value) {
  var sheet = getSettingsSheet();
  var data = sheet.getDataRange().getValues();
  var rowIndex = -1;
  
  // Look for existing setting
  for (var i = 1; i < data.length; i++) {
    if (data[i][0] === settingName) {
      rowIndex = i + 1;
      break;
    }
  }
  
  if (rowIndex === -1) {
    // Add new setting at the end
    rowIndex = data.length + 1;
  }
  
  // Update the setting
  sheet.getRange(rowIndex, 1, 1, 2).setValues([[settingName, value]]);
}

// Helper function to manage stats sheet
function getStatsSheet() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var statsSheet = ss.getSheetByName('Stats');
  
  // Create Stats sheet if it doesn't exist
  if (!statsSheet) {
    statsSheet = ss.insertSheet('Stats');
    // Add headers
    statsSheet.getRange('A1:B1').setValues([['Metric', 'Value']]);
    statsSheet.setFrozenRows(1);
    
    // Set column widths
    statsSheet.setColumnWidth(1, 200);  // Metric name
    statsSheet.setColumnWidth(2, 300);  // Value
    
    // Format headers
    var headerRange = statsSheet.getRange('A1:B1');
    headerRange.setBackground('#f3f3f3')
               .setFontWeight('bold')
               .setHorizontalAlignment('center');
               
    // Add initial metrics
    statsSheet.getRange('A2:A5').setValues([
      ['Last Training Time'],
      ['Training Data Size'],
      ['Training Sheet'],
      ['Model Status']
    ]);
  }
  
  return statsSheet;
}

// Helper function to update stats
function updateStats(metric, value) {
  var sheet = getStatsSheet();
  var data = sheet.getDataRange().getValues();
  var rowIndex = -1;
  
  // Look for existing metric
  for (var i = 1; i < data.length; i++) {
    if (data[i][0] === metric) {
      rowIndex = i + 1;
      break;
    }
  }
  
  if (rowIndex === -1) {
    // Add new metric at the end
    rowIndex = data.length + 1;
    sheet.getRange(rowIndex, 1).setValue(metric);
  }
  
  // Update the value
  sheet.getRange(rowIndex, 2).setValue(value);
}

function checkTrainingStatus() {
  try {
    var userProperties = PropertiesService.getUserProperties();
    var predictionId = userProperties.getProperty('PREDICTION_ID');
    var triggerId = userProperties.getProperty('TRIGGER_ID');
    var startTime = parseInt(userProperties.getProperty('START_TIME'));
    var retryCount = parseInt(userProperties.getProperty('RETRY_COUNT') || '0');
    var maxRetries = 5;  // Maximum number of consecutive failed attempts
    
    if (!predictionId || !triggerId) {
      Logger.log("Missing predictionId or triggerId - cleaning up");
      cleanupTrigger(triggerId);
      return;
    }
    
    var minutesElapsed = Math.floor((new Date().getTime() - startTime) / (60 * 1000));
    if (minutesElapsed > 30) {  // 30 minutes timeout
      updateStatus("Error: Training timed out after 30 minutes");
      updateStats('Model Status', 'Error: Training timed out');
      cleanupTrigger(triggerId);
      userProperties.deleteAllProperties();
      return;
    }
    
    var config = getServiceConfig();
    
    // Add retry logic for fetch with exponential backoff
    var maxFetchRetries = 3;
    var response = null;
    var fetchError = null;
    
    for (var i = 0; i < maxFetchRetries; i++) {
      try {
    var options = {
      headers: {
            'X-API-Key': config.apiKey,
            'Accept': 'application/json',
            'Connection': 'keep-alive'
          },
          muteHttpExceptions: true,
          validateHttpsCertificates: true,
          followRedirects: true,
          timeout: 30000  // 30 seconds timeout
        };
        
        response = UrlFetchApp.fetch(config.serviceUrl + '/status/' + predictionId, options);
        var responseCode = response.getResponseCode();
        
        // Handle different response codes
        if (responseCode === 200) {
          fetchError = null;
          break;
        } else if (responseCode === 404) {
          // Status not found - might be temporary, retry
          fetchError = "Status not found";
          Logger.log("Status not found, will retry...");
          Utilities.sleep(Math.pow(2, i) * 1000);  // Exponential backoff
          continue;
        } else if (responseCode >= 500) {
          // Server error - retry
          fetchError = `Server error (${responseCode})`;
          Logger.log("Server error, will retry...");
          Utilities.sleep(Math.pow(2, i) * 1000);
          continue;
        } else {
          // Other errors - don't retry
          throw new Error(`Unexpected response code: ${responseCode}`);
        }
      } catch (e) {
        fetchError = e;
        if (i < maxFetchRetries - 1) {
          Logger.log(`Fetch attempt ${i + 1} failed: ${e}. Retrying...`);
          Utilities.sleep(Math.pow(2, i) * 1000);
        }
      }
    }
    
    if (fetchError) {
      retryCount++;
      userProperties.setProperty('RETRY_COUNT', retryCount.toString());
      
      if (retryCount >= maxRetries) {
        updateStatus("Error: Service unavailable after multiple retries. Please try again later.");
        updateStats('Model Status', 'Error: Service unavailable');
        cleanupTrigger(triggerId);
        userProperties.deleteAllProperties();
        return;
      }
      
      var backoffMinutes = Math.min(retryCount, 5);  // Cap at 5 minutes
      updateStatus(`Service temporarily unavailable. Retrying in ${backoffMinutes} minute(s)... (Attempt ${retryCount}/${maxRetries})`);
      return;
    }
    
    // Reset retry count on successful fetch
    if (retryCount > 0) {
      userProperties.setProperty('RETRY_COUNT', '0');
    }
    
    var responseText = response.getContentText();
    if (!responseText || responseText.trim() === '') {
      updateStatus(`Training in progress... (${minutesElapsed} min)`);
      return;
    }
    
    try {
      var result = JSON.parse(responseText);
      var statusMessage = `Training in progress... (${minutesElapsed} min)`;
      
      if (result.status) {
        statusMessage += ` - ${result.status}`;
      }
      if (result.message) {
        statusMessage += `\n${result.message}`;
      }
      
      updateStatus(statusMessage);
      
      if (result.status === "completed") {
        updateStats('Last Training Time', new Date().toLocaleString());
        updateStats('Model Status', 'Ready');
    updateStatus("Training completed successfully!");
        cleanupTrigger(triggerId);
        userProperties.deleteAllProperties();
      } else if (result.status === "failed") {
        updateStats('Model Status', 'Error: Training failed');
        updateStatus(`Error: Training failed - ${result.error || "Unknown error"}`);
        cleanupTrigger(triggerId);
        userProperties.deleteAllProperties();
      }
      
    } catch (parseError) {
      Logger.log("Error parsing response: " + parseError);
      updateStatus(`Training in progress... (${minutesElapsed} min) - Waiting for response`);
    }
    
  } catch (error) {
    Logger.log("Error checking training status: " + error);
    Logger.log("Error stack: " + error.stack);
    cleanupTrigger(triggerId);
    userProperties.deleteAllProperties();
    updateStatus("Error: " + error.toString());
    updateStats('Model Status', 'Error: ' + error.toString());
  }
}

// Helper function to clean up triggers
function cleanupTrigger(triggerId) {
  if (triggerId) {
    var triggers = ScriptApp.getProjectTriggers();
    for (var i = 0; i < triggers.length; i++) {
      if (triggers[i].getUniqueId() === triggerId) {
        ScriptApp.deleteTrigger(triggers[i]);
        break;
      }
    }
  }
}

// Main classification function
function classifyTransactions(config) {
  var sheet = SpreadsheetApp.getActiveSheet();
  var ui = SpreadsheetApp.getUi();
  
  try {
    Logger.log("Starting classification with config: " + JSON.stringify(config));
    updateStatus("Starting classification...");
    var serviceConfig = getServiceConfig();
    
    // Log service configuration
    Logger.log("Service configuration:");
    Logger.log("- URL: " + serviceConfig.serviceUrl);
    Logger.log("- API Key: " + serviceConfig.apiKey);
    
    // Store the original sheet name
    var originalSheetName = sheet.getName();
    Logger.log("Original sheet name: " + originalSheetName);
    
    // Get all descriptions from the specified column
    var lastRow = sheet.getLastRow();
    var descriptionRange = sheet.getRange(config.descriptionCol + config.startRow + ":" + config.descriptionCol + lastRow);
    var descriptions = descriptionRange.getValues();
    
    // Filter out empty descriptions
    var transactions = descriptions
      .filter(row => row[0] && row[0].toString().trim() !== '')
      .map(row => ({
        Narrative: row[0],
        description: row[0]  // Add description field as it's expected by the backend
      }));
    
    if (transactions.length === 0) {
      updateStatus("Error: No transactions found");
      ui.alert('No transactions found to classify');
      return;
    }
    
    Logger.log("Found " + transactions.length + " transactions to classify");
    updateStatus("Processing " + transactions.length + " transactions...");
    
    // Call classification service
    var options = {
      method: 'post',
      contentType: 'application/json',
      headers: {
        'X-API-Key': serviceConfig.apiKey
      },
      payload: JSON.stringify({ 
        transactions: transactions,
        spreadsheetId: sheet.getParent().getId(),
        sheetName: originalSheetName,
        columnOrderCategorisation: {
          descriptionColumn: config.descriptionCol,
          categoryColumn: config.categoryCol
        },
        categorisationRange: "A:Z",
        categorisationTab: originalSheetName,
        startRow: config.startRow
      }),
      muteHttpExceptions: true
    };
    
    Logger.log("Making classification request with API key: " + serviceConfig.apiKey);
    var response = UrlFetchApp.fetch(serviceConfig.serviceUrl + '/classify', options);
    var result = JSON.parse(response.getContentText());
    
    Logger.log("Classification service response: " + JSON.stringify(result));
    
    if (response.getResponseCode() !== 200) {
      updateStatus("Error: Classification failed");
      throw new Error('Classification service error: ' + response.getContentText());
    }
    
    // Check if we got a prediction ID
    if (result.prediction_id) {
      updateStatus("Classification in progress...");
      
      // Store configuration for status checking
      var userProperties = PropertiesService.getUserProperties();
      
      // Clear any existing properties first
      userProperties.deleteAllProperties();
      
      // Store new properties
      var properties = {
        'PREDICTION_ID': result.prediction_id,
        'OPERATION_TYPE': 'classify',
        'START_TIME': new Date().getTime().toString(),
        'ORIGINAL_SHEET_NAME': originalSheetName,
        'CONFIG': JSON.stringify({
          categoryCol: config.categoryCol,
          startRow: config.startRow,
          descriptionCol: config.descriptionCol
        })
      };
      
      Logger.log("Storing properties: " + JSON.stringify(properties));
      userProperties.setProperties(properties);
      
      // Create a trigger to check status every minute
      var trigger = ScriptApp.newTrigger('checkOperationStatus')
        .timeBased()
        .everyMinutes(1)
        .create();
      
      var triggerId = trigger.getUniqueId();
      Logger.log("Created trigger with ID: " + triggerId);
      userProperties.setProperty('TRIGGER_ID', triggerId);
      
      // Verify properties were stored
      var storedProps = userProperties.getProperties();
      Logger.log("Stored properties: " + JSON.stringify(storedProps));
      
      ui.alert('Classification has started! The sheet will update automatically when complete.\n\nYou can close this window.');
      return;
    }
  } catch (error) {
    Logger.log("Classification error: " + error.toString());
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
  var originalSheetName = userProperties.getProperty('ORIGINAL_SHEET_NAME');
  var config = userProperties.getProperty('CONFIG') ? JSON.parse(userProperties.getProperty('CONFIG')) : null;
  
  Logger.log("Starting checkOperationStatus");
  Logger.log("PredictionId: " + predictionId);
  Logger.log("TriggerId: " + triggerId);
  Logger.log("OperationType: " + operationType);
  Logger.log("Original Sheet Name: " + originalSheetName);
  Logger.log("Config: " + JSON.stringify(config));
  
  if (!predictionId || !triggerId) {
    Logger.log("Missing predictionId or triggerId - cleaning up");
    // Clean up if no prediction ID or trigger ID
    if (triggerId) {
      var triggers = ScriptApp.getProjectTriggers();
      for (var i = 0; i < triggers.length; i++) {
        if (triggers[i].getUniqueId() === triggerId) {
          ScriptApp.deleteTrigger(triggers[i]);
          break;
        }
      }
    }
    return;
  }
  
  try {
    // Check if we've been running for more than 30 minutes
    if (new Date().getTime() - startTime > 30 * 60 * 1000) {
      updateStatus(`Error: ${operationType} timed out after 30 minutes`);
      var triggers = ScriptApp.getProjectTriggers();
      for (var i = 0; i < triggers.length; i++) {
        if (triggers[i].getUniqueId() === triggerId) {
          ScriptApp.deleteTrigger(triggers[i]);
          break;
        }
      }
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
        try {
          // Get the original sheet by name
          var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
          var sheet = spreadsheet.getSheetByName(originalSheetName);
          if (!sheet) {
            throw new Error("Could not find original sheet: " + originalSheetName);
          }
          Logger.log("Retrieved original sheet: " + sheet.getName());
          updateStatus("Processing results for sheet: " + sheet.getName());

          // Get webhook results from the result field
          var webhookResults = [];
          if (Array.isArray(result.result)) {
            webhookResults = result.result;
          } else if (typeof result.result === 'object' && Array.isArray(result.result.results)) {
            webhookResults = result.result.results;
          }
          
          Logger.log("Webhook results length: " + webhookResults.length);
          Logger.log("First webhook result: " + JSON.stringify(webhookResults[0]));
          
          if (webhookResults.length === 0) {
            updateStatus("Error: No classification results found");
            throw new Error("No classification results found in webhook response");
          }
          
          // Write categories back to sheet
          updateStatus("Writing " + webhookResults.length + " categories to sheet...");
          var categories = webhookResults.map(r => [r.predicted_category]);
          Logger.log("Categories to write: " + JSON.stringify(categories.slice(0, 5)));
          
          // Calculate the range - allow starting from row 1
          var startRow = parseInt(config.startRow);
          if (isNaN(startRow)) {
            startRow = 1; // Default to 1 if parsing fails
          }
          
          // Calculate end row based on number of results
          var endRow = startRow + categories.length - 1;
          Logger.log("Writing to rows: " + startRow + " to " + endRow);
          
          // Write categories
          try {
            var categoryRangeA1 = config.categoryCol + startRow + ":" + config.categoryCol + endRow;
            Logger.log("Category range A1 notation: " + categoryRangeA1);
            updateStatus("Writing categories to range: " + categoryRangeA1);
            
            var categoryRange = sheet.getRange(categoryRangeA1);
            Logger.log("Got range object successfully");
            
            // Log the current values in the range before writing
            var currentValues = categoryRange.getValues();
            Logger.log("Current values in range: " + JSON.stringify(currentValues.slice(0, 5)));
            
            categoryRange.setValues(categories);
            Logger.log("Categories written successfully");
            
            // Verify the write
            var writtenValues = sheet.getRange(categoryRangeA1).getValues();
            Logger.log("Verification - written values: " + JSON.stringify(writtenValues.slice(0, 5)));
            updateStatus("Categories written successfully to column " + config.categoryCol);
          } catch (categoryError) {
            Logger.log("Error writing categories: " + categoryError);
            Logger.log("Error stack: " + categoryError.stack);
            updateStatus("Error writing categories: " + categoryError.toString());
            throw new Error("Failed to write categories: " + categoryError.toString());
          }
          
          // Write confidence scores
          try {
            var confidenceCol = String.fromCharCode(config.categoryCol.charCodeAt(0) + 1);
            var confidences = webhookResults.map(r => [r.similarity_score]);
            var confidenceRangeA1 = confidenceCol + startRow + ":" + confidenceCol + endRow;
            Logger.log("Confidence range A1 notation: " + confidenceRangeA1);
            updateStatus("Writing confidence scores to range: " + confidenceRangeA1);
            
            var confidenceRange = sheet.getRange(confidenceRangeA1);
            Logger.log("Got confidence range object successfully");
            
            // Log current values in confidence range
            var currentConfidences = confidenceRange.getValues();
            Logger.log("Current confidence values: " + JSON.stringify(currentConfidences.slice(0, 5)));
            
            confidenceRange.setValues(confidences)
              .setNumberFormat("0.00%");  // Format as percentage
            Logger.log("Confidence scores written successfully");
            
            // Verify the write
            var writtenConfidences = sheet.getRange(confidenceRangeA1).getValues();
            Logger.log("Verification - written confidences: " + JSON.stringify(writtenConfidences.slice(0, 5)));
            updateStatus("Confidence scores written successfully to column " + confidenceCol);
          } catch (confidenceError) {
            Logger.log("Error writing confidence scores: " + confidenceError);
            Logger.log("Error stack: " + confidenceError.stack);
            updateStatus("Error writing confidence scores: " + confidenceError.toString());
            throw new Error("Failed to write confidence scores: " + confidenceError.toString());
          }
          
          updateStatus("Classification completed successfully!");
          
        } catch (writeError) {
          Logger.log("Error writing to sheet: " + writeError);
          Logger.log("Error stack: " + writeError.stack);
          updateStatus("Error writing results: " + writeError.toString());
          throw writeError;
        }
      }
      
      // Clean up
      var triggers = ScriptApp.getProjectTriggers();
      for (var i = 0; i < triggers.length; i++) {
        if (triggers[i].getUniqueId() === triggerId) {
          ScriptApp.deleteTrigger(triggers[i]);
          break;
        }
      }
      userProperties.deleteAllProperties();
      return;
      
    } else if (result.status === "failed") {
      updateStatus(`Error: ${operationType} failed - ` + (result.error || "Unknown error"));
      // Clean up
      var triggers = ScriptApp.getProjectTriggers();
      for (var i = 0; i < triggers.length; i++) {
        if (triggers[i].getUniqueId() === triggerId) {
          ScriptApp.deleteTrigger(triggers[i]);
          break;
        }
      }
      userProperties.deleteAllProperties();
      return;
    }
    
    // Still processing, update status
    updateStatus(operationType + " in progress... " + (result.message || ""));
    
  } catch (error) {
    Logger.log(`Error checking ${operationType} status: ` + error);
    Logger.log("Error details: " + error.stack);
    // Clean up on error
    if (triggerId) {
      var triggers = ScriptApp.getProjectTriggers();
      for (var i = 0; i < triggers.length; i++) {
        if (triggers[i].getUniqueId() === triggerId) {
          ScriptApp.deleteTrigger(triggers[i]);
          break;
        }
      }
    }
    userProperties.deleteAllProperties();
    updateStatus("Error: " + error.toString());
  }
}

// Helper function to convert column number to letter
function columnToLetter(column) {
  var temp, letter = '';
  while (column > 0) {
    temp = (column - 1) % 26;
    letter = String.fromCharCode(temp + 65) + letter;
    column = (column - temp - 1) / 26;
  }
  return letter;
}

function trainModel(config) {
  var ui = SpreadsheetApp.getUi();
  
  try {
    // Validate config object
    if (!config) {
      throw new Error('Configuration object is missing');
    }

    // Validate required parameters
    if (!config.narrativeCol || !config.categoryCol || !config.startRow) {
      throw new Error('Missing required configuration parameters. Please ensure all fields are filled out.');
    }

    updateStatus("Starting training...");
    var serviceConfig = getServiceConfig();
    var sheet = SpreadsheetApp.getActiveSheet();
    var lastRow = sheet.getLastRow();
    
    // Get data from the selected columns
    var narrativeRange = sheet.getRange(config.narrativeCol + config.startRow + ":" + config.narrativeCol + lastRow);
    var categoryRange = sheet.getRange(config.categoryCol + config.startRow + ":" + config.categoryCol + lastRow);
    var narratives = narrativeRange.getValues();
    var categories = categoryRange.getValues();
    
    // Filter out empty rows and prepare training data
    var transactions = [];
    for (var i = 0; i < narratives.length; i++) {
      if (narratives[i][0] && categories[i][0]) {
        transactions.push({
          Narrative: narratives[i][0],
          Category: categories[i][0]
        });
      }
    }
    
    if (transactions.length === 0) {
      updateStatus("Error: No training data found");
      ui.alert('No training data found. Please ensure you have transactions with categories.');
      return;
    }
    
    updateStatus("Processing " + transactions.length + " transactions...");
    
    // Prepare the payload with consistent field names
    var payload = JSON.stringify({ 
      transactions: transactions,
      userId: serviceConfig.userId,
      expenseSheetId: sheet.getParent().getId(),
      columnOrderCategorisation: {
        descriptionColumn: config.narrativeCol,
        categoryColumn: config.categoryCol
      },
      categorisationRange: "A:Z",
      categorisationTab: sheet.getName()
    });
    
    // Initialize retry variables
    var maxRetries = 3;
    var retryCount = 0;
    var lastError = null;
    var response = null;
    
    // Retry loop for the initial training request
    while (retryCount < maxRetries) {
      try {
        // Add retry attempt to status
        if (retryCount > 0) {
          updateStatus(`Retrying training request (attempt ${retryCount + 1}/${maxRetries})...`);
          Utilities.sleep(Math.pow(2, retryCount) * 1000); // Exponential backoff
        }
        
        // Call training endpoint with improved options
        var options = {
          method: 'post',
          contentType: 'application/json',
          headers: {
            'X-API-Key': serviceConfig.apiKey,
            'Accept': 'application/json',
            'Connection': 'keep-alive'
          },
          payload: payload,
          muteHttpExceptions: true,
          validateHttpsCertificates: true,
          followRedirects: true,
          timeout: 30000  // 30 second timeout
        };
        
        response = UrlFetchApp.fetch(serviceConfig.serviceUrl + '/train', options);
        var responseCode = response.getResponseCode();
        
        // Handle different response codes
        if (responseCode === 200) {
          break; // Success, exit retry loop
        } else if (responseCode === 502 || responseCode === 503 || responseCode === 504) {
          // Retry on gateway errors
          lastError = `Server returned ${responseCode}`;
          throw new Error(lastError);
        } else {
          // Don't retry on other errors
          throw new Error(`Training failed with status ${responseCode}`);
        }
      } catch (e) {
        lastError = e;
        if (retryCount === maxRetries - 1) {
          // Last attempt failed
          updateStatus(`Error: Training failed after ${maxRetries} attempts. Last error: ${e.toString()}`);
          throw new Error(`Training failed after ${maxRetries} attempts: ${e.toString()}`);
        }
        retryCount++;
        continue;
      }
    }
    
    var responseText = response.getContentText();
    
    // Parse response carefully
    var result;
    try {
      result = JSON.parse(responseText);
    } catch (e) {
      updateStatus("Error: Invalid response format");
      throw new Error("Failed to parse server response: " + e.toString());
    }
    
    // Validate result structure
    if (!result || typeof result !== 'object') {
      updateStatus("Error: Invalid response structure");
      throw new Error("Invalid response structure from server");
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
        'START_TIME': new Date().getTime().toString(),
        'TRAINING_SIZE': transactions.length.toString(),
        'RETRY_COUNT': '0'  // Initialize retry count
      });
      
      ui.alert('Training has started! The sheet will update automatically when training is complete.\n\nYou can close this window.');
      return;
    } else {
      updateStatus("Error: No prediction ID received");
      throw new Error("No prediction ID received from server");
    }
    
  } catch (error) {
    Logger.log("Training error: " + error.toString());
    updateStatus("Error: " + error.toString());
    ui.alert('Error: ' + error.toString());
  }
} 