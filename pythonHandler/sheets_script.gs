// Service configuration
const CLASSIFICATION_SERVICE_URL = 'https://txclassify.onrender.com';

// Add menu to the spreadsheet
function onOpen(e) {
  var menu = SpreadsheetApp.getUi().createAddonMenu(); // Changed from addMenu to createAddonMenu
  menu.addItem("Configure API Key", "setupApiKey")
      .addItem("Train Model", "showTrainingDialog")
      .addItem("Categorise New Transactions", "showClassifyDialog")
      .addToUi();
}

// Handle add-on installation
function onInstall(e) {
  onOpen(e);
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
      <div class="help-text">First row of data to categorise</div>
    </div>
    <div id="error" class="error"></div>
    <button onclick="submitForm()" id="submitBtn">
      <span class="button-text">Categorise Transactions</span>
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
          .categoriseTransactions(config);
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
  
  SpreadsheetApp.getUi().showModalDialog(html, 'Categorise Transactions');
}

// Show dialog to select columns for training
function showTrainingDialog() {
  // Get current sheet to determine default columns
  var sheet = SpreadsheetApp.getActiveSheet();
  var lastColumn = sheet.getLastColumn();
  var headers = sheet.getRange(1, 1, 1, lastColumn).getValues()[0];
  
  // Find default column indices
  var narrativeColDefault = columnToLetter(headers.indexOf("Narrative") + 1);
  var categoryColDefault = columnToLetter(headers.indexOf("Category") + 1);
  
  // Create column options - always include all columns up to last used column
  var columnOptions = [];
  for (var i = 0; i < lastColumn; i++) {
    var letter = columnToLetter(i + 1);
    var header = headers[i] || ''; // Use empty string if header is null/undefined
    columnOptions.push(
      `<option value="${letter}"${letter === narrativeColDefault ? ' selected' : ''}>` +
      `${letter}${header ? ' (' + header + ')' : ''}</option>`
    );
  }
  var columnOptionsStr = columnOptions.join('\n');

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
        ${columnOptionsStr}
      </select>
      <div class="help-text">Select the column containing transaction descriptions</div>
    </div>
    <div class="form-group">
      <label>Category Column:</label>
      <select id="categoryCol">
        ${columnOptionsStr}
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
  var existingApiKey = properties.getProperty('API_KEY') || '';
  var userEmail = Session.getEffectiveUser().getEmail();
  
  var html = HtmlService.createHtmlOutput(`
    <style>
      body { font-family: Arial, sans-serif; padding: 20px; }
      .form-group { margin-bottom: 15px; }
      label { display: block; margin-bottom: 5px; }
      input { width: 100%; padding: 5px; margin-bottom: 10px; }
      button { 
        padding: 8px 15px; 
        background: #4285f4; 
        color: white; 
        border: none; 
        border-radius: 3px; 
        cursor: pointer;
        margin-right: 10px;
      }
      button:disabled {
        background: #ccc;
        cursor: not-allowed;
      }
      .instructions {
        background: #f5f5f5;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 20px;
      }
      .instructions ol {
        margin: 10px 0;
        padding-left: 20px;
      }
      .error { 
        color: red; 
        margin-top: 10px; 
        display: none; 
      }
      .success {
        color: green;
        margin-top: 10px;
        display: none;
      }
      .web-app-link {
        display: block;
        text-align: center;
        margin: 20px 0;
        color: #4285f4;
        text-decoration: underline;
        font-weight: bold;
      }
      .current-key {
        background: #f0f8ff;
        padding: 10px;
        border-radius: 4px;
        border: 1px solid #cce5ff;
        margin-bottom: 15px;
        word-break: break-all;
        font-family: monospace;
      }
      .key-status {
        font-weight: bold;
        margin-bottom: 5px;
      }
    </style>
    <div class="instructions">
      <strong>API Key Management:</strong>
      <p>Your API key is used to authenticate with the categorization service.</p>
      <ol>
        <li>Get your API key from the ExpenseSorted web application</li>
        <li>Copy the API key and paste it below</li>
        <li>Click "Save API Key" to configure this spreadsheet</li>
      </ol>
    </div>
    
    ${existingApiKey ? `
    <div class="current-key">
      <div class="key-status">Current API Key:</div>
      ${existingApiKey}
    </div>
    ` : ''}
    
    <div class="form-group">
      <label>API Key:</label>
      <input type="text" id="apiKey" placeholder="Enter your API key" value="${existingApiKey}">
      <div id="error" class="error"></div>
      <div id="success" class="success"></div>
    </div>
    
    <button onclick="saveApiKey()">Save API Key</button>
    
    <a href="https://expensesorted.com/api-key" target="_blank" class="web-app-link">
      Get or Generate API Key from ExpenseSorted Web App
    </a>
    
    <script>
      function saveApiKey() {
        var apiKey = document.getElementById('apiKey').value.trim();
        var errorDiv = document.getElementById('error');
        var successDiv = document.getElementById('success');
        
        // Hide any previous messages
        errorDiv.style.display = 'none';
        successDiv.style.display = 'none';
        
        if (!apiKey) {
          errorDiv.textContent = 'API key is required';
          errorDiv.style.display = 'block';
          return;
        }
        
        google.script.run
          .withSuccessHandler(function() {
            successDiv.textContent = 'API key saved successfully!';
            successDiv.style.display = 'block';
            setTimeout(function() {
              google.script.host.close();
            }, 1500);
          })
          .withFailureHandler(function(error) {
            errorDiv.textContent = error.message || 'An error occurred';
            errorDiv.style.display = 'block';
          })
          .saveApiKey(apiKey);
      }
    </script>
  `)
    .setWidth(500)
    .setHeight(450);
  
  SpreadsheetApp.getUi().showModalDialog(html, 'API Key Management');
}

// Helper function to save the API key
function saveApiKey(apiKey) {
  if (!apiKey) {
    throw new Error('API key is required');
  }
  
  var properties = PropertiesService.getScriptProperties();
  properties.setProperty('API_KEY', apiKey.trim());
  updateStatus("API key configured successfully!");
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
    var response = null;
    var fetchError = null;
    
    try {
      var options = {
        headers: {
          'X-API-Key': config.apiKey,
          'Accept': 'application/json'
        },
        muteHttpExceptions: true
      };
      
      response = UrlFetchApp.fetch(config.serviceUrl + '/status/' + predictionId, options);
      var responseCode = response.getResponseCode();
      
      if (responseCode !== 200) {
        throw new Error(`Unexpected response code: ${responseCode}`);
      }
    } catch (e) {
      fetchError = e;
      retryCount++;
      userProperties.setProperty('RETRY_COUNT', retryCount.toString());
      
      if (retryCount >= maxRetries) {
        updateStatus("Error: Service unavailable after multiple retries");
        updateStats('Model Status', 'Error: Service unavailable');
        cleanupTrigger(triggerId);
        userProperties.deleteAllProperties();
        return;
      }
      
      var backoffMinutes = Math.min(retryCount, 5);
      updateStatus(`Service temporarily unavailable. Retrying in ${backoffMinutes} minute(s)... (Attempt ${retryCount}/${maxRetries})`);
      return;
    }
    
    // Reset retry count on successful fetch
    if (retryCount > 0) {
      userProperties.setProperty('RETRY_COUNT', '0');
    }
    
    var responseText = response.getContentText();
    if (!responseText) {
      updateStatus(`Training in progress... (${minutesElapsed} min)`);
      updateStats('Model Status', 'Training in progress');
      return;
    }
    
    try {
      var result = JSON.parse(responseText);
      var statusMessage = `Training in progress... (${minutesElapsed} min)`;
      
      if (result.status) {
        statusMessage += ` - ${result.status}`;
        updateStats('Model Status', result.status);
      }
      if (result.message) {
        statusMessage += `\n${result.message}`;
      }
      
      updateStatus(statusMessage);
      
      if (result.status === "completed") {
        // Update all stats
        var sheet = SpreadsheetApp.getActiveSheet();
        updateStats('Last Training Time', new Date().toLocaleString());
        updateStats('Training Data Size', sheet.getLastRow() - 1);  // Subtract header row
        updateStats('Training Sheet', sheet.getName());
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
      updateStats('Model Status', 'Training in progress');
    }
    
  } catch (error) {
    Logger.log("Error checking training status: " + error);
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
function categoriseTransactions(config) {
  var sheet = SpreadsheetApp.getActiveSheet();
  var ui = SpreadsheetApp.getUi();
  
  try {
    Logger.log("Starting categorisation with config: " + JSON.stringify(config));
    updateStatus("Starting categorisation...");
    var serviceConfig = getServiceConfig();
    
    // Store the original sheet name
    var originalSheetName = sheet.getName();
    var spreadsheetId = sheet.getParent().getId();
    Logger.log("Original sheet name: " + originalSheetName);
    
    // Get all descriptions from the specified column
    var lastRow = sheet.getLastRow();
    var startRow = parseInt(config.startRow);
    var descriptionRange = sheet.getRange(config.descriptionCol + startRow + ":" + config.descriptionCol + lastRow);
    var descriptions = descriptionRange.getValues();
    
    // Filter out empty descriptions and prepare transactions array
    var transactions = descriptions
      .filter(row => row[0] && row[0].toString().trim() !== '')
      .map(row => ({
        description: row[0]  // Only send the description field
      }));
    
    if (transactions.length === 0) {
      updateStatus("Error: No transactions found");
      ui.alert('No transactions found to categorise');
      return;
    }
    
    Logger.log("Found " + transactions.length + " transactions to categorise");
    updateStatus("Processing " + transactions.length + " transactions...");
    
    // Call categorisation service with minimal data
    var options = {
      method: 'post',
      contentType: 'application/json',
      headers: {
        'X-API-Key': serviceConfig.apiKey
      },
      payload: JSON.stringify({ 
        transactions: transactions,
        userId: serviceConfig.userId,
        spreadsheetId: spreadsheetId,
        sheetName: originalSheetName,  // Pass the sheet name explicitly
        startRow: startRow.toString()  // Pass the start row explicitly
      }),
      muteHttpExceptions: true
    };
    
    Logger.log("Making categorisation request");
    var response = UrlFetchApp.fetch(serviceConfig.serviceUrl + '/classify', options);
    var result = JSON.parse(response.getContentText());
    
    Logger.log("Categorisation service response: " + JSON.stringify(result));
    
    if (response.getResponseCode() !== 200) {
      updateStatus("Error: Categorisation failed");
      throw new Error('Categorisation service error: ' + response.getContentText());
    }
    
    // Check if we got a prediction ID
    if (result.prediction_id) {
      updateStatus("Categorisation in progress...");
      
      // Store configuration for status checking
      var userProperties = PropertiesService.getUserProperties();
      
      // Clear any existing properties first
      userProperties.deleteAllProperties();
      
      // Store new properties
      var properties = {
        'PREDICTION_ID': result.prediction_id,
        'OPERATION_TYPE': 'categorise',
        'START_TIME': new Date().getTime().toString(),
        'ORIGINAL_SHEET_NAME': originalSheetName,
        'CONFIG': JSON.stringify({
          categoryCol: config.categoryCol,
          startRow: config.startRow,
          descriptionCol: config.descriptionCol,
          transactionCount: transactions.length,
          spreadsheetId: spreadsheetId
        })
      };
      
      Logger.log("Storing properties: " + JSON.stringify(properties));
      userProperties.setProperties(properties);
      
      // Create a trigger to check status every 5 minutes instead of every minute
      var trigger = ScriptApp.newTrigger('checkOperationStatus')
        .timeBased()
        .everyMinutes(5)
        .create();
      
      var triggerId = trigger.getUniqueId();
      Logger.log("Created trigger with ID: " + triggerId);
      userProperties.setProperty('TRIGGER_ID', triggerId);
      
      ui.alert('Categorisation has started! The sheet will update automatically when complete.\n\nYou can close this window.');
      return;
    }
  } catch (error) {
    Logger.log("Categorisation error: " + error.toString());
    updateStatus("Error: " + error.toString());
    ui.alert('Error: ' + error.toString());
  }
}

// Update the status check function to remove OAuth token
function checkOperationStatus() {
  var userProperties = PropertiesService.getUserProperties();
  var predictionId = userProperties.getProperty('PREDICTION_ID');
  var triggerId = userProperties.getProperty('TRIGGER_ID');
  var operationType = userProperties.getProperty('OPERATION_TYPE');
  var startTime = parseInt(userProperties.getProperty('START_TIME'));
  var originalSheetName = userProperties.getProperty('ORIGINAL_SHEET_NAME');
  var config = userProperties.getProperty('CONFIG') ? JSON.parse(userProperties.getProperty('CONFIG')) : null;
  var retryCount = parseInt(userProperties.getProperty('RETRY_COUNT') || '0');
  
  Logger.log("Starting checkOperationStatus");
  Logger.log("PredictionId: " + predictionId);
  Logger.log("TriggerId: " + triggerId);
  Logger.log("OperationType: " + operationType);
  Logger.log("Original Sheet Name: " + originalSheetName);
  Logger.log("Config: " + JSON.stringify(config));
  Logger.log("Retry Count: " + retryCount);
  
  if (!predictionId || !triggerId) {
    Logger.log("Missing predictionId or triggerId - cleaning up");
    cleanupTrigger(triggerId);
    return;
  }
  
  try {
    // Check if we've been running for more than 30 minutes
    if (new Date().getTime() - startTime > 30 * 60 * 1000) {
      updateStatus(`Error: ${operationType} timed out after 30 minutes`);
      cleanupTrigger(triggerId);
      userProperties.deleteAllProperties();
      return;
    }
    
    var serviceConfig = getServiceConfig();
    var response = UrlFetchApp.fetch(serviceConfig.serviceUrl + '/status/' + predictionId, {
      headers: { 'X-API-Key': serviceConfig.apiKey },
      muteHttpExceptions: true
    });
    
    // Check HTTP response code
    var responseCode = response.getResponseCode();
    if (responseCode !== 200) {
      retryCount++;
      userProperties.setProperty('RETRY_COUNT', retryCount.toString());
      
      if (retryCount >= 5) {
        updateStatus(`Error: ${operationType} failed after multiple retries`);
        cleanupTrigger(triggerId);
        userProperties.deleteAllProperties();
        return;
      }
      
      var minutesElapsed = Math.floor((new Date().getTime() - startTime) / (60 * 1000));
      updateStatus(`${operationType} in progress... (${minutesElapsed} min) - Temporary connection issue (retry ${retryCount}/5)`);
      return;
    }
    
    var result = JSON.parse(response.getContentText());
    Logger.log("Status response: " + JSON.stringify(result));
    
    // Reset retry count on successful response
    if (retryCount > 0) {
      userProperties.setProperty('RETRY_COUNT', '0');
    }
    
    // Check webhook results first
    if (result.result && result.result.results) {
      Logger.log("Found webhook results");
      if (operationType === "categorise") {
        handleClassificationResults(result, config, originalSheetName);
      } else {
        // For training, just update status and stats
        updateStats('Last Training Time', new Date().toLocaleString());
        updateStats('Training Sheet', originalSheetName);
        updateStats('Model Status', 'Ready');
        updateStatus("Training completed successfully!");
      }
      cleanupTrigger(triggerId);
      userProperties.deleteAllProperties();
      return;
    }
    
    // Handle different status types
    if (result.status === "completed") {
      // Wait for webhook results
      var minutesElapsed = Math.floor((new Date().getTime() - startTime) / (60 * 1000));
      updateStatus(`${operationType === "categorise" ? "Categorisation" : "Training"} completed, processing results... (${minutesElapsed} min)`);
      return;
    } else if (result.status === "failed") {
      updateStatus(`Error: ${operationType} failed - ` + (result.error || "Unknown error"));
      cleanupTrigger(triggerId);
      userProperties.deleteAllProperties();
      return;
    } else if (result.status === "not_found" || result.status === "unknown" || result.status === "error") {
      // Handle error statuses that indicate we should stop checking
      retryCount++;
      userProperties.setProperty('RETRY_COUNT', retryCount.toString());
      
      if (retryCount >= 3) {
        updateStatus(`Error: ${operationType} failed - ${result.message || result.error || "Unknown error"}`);
        cleanupTrigger(triggerId);
        userProperties.deleteAllProperties();
        return;
      }
      
      // Continue for a few retries in case it's a temporary issue
      var minutesElapsed = Math.floor((new Date().getTime() - startTime) / (60 * 1000));
      updateStatus(`${operationType} in progress... (${minutesElapsed} min) - Waiting for status (retry ${retryCount}/3)`);
      return;
    }
    
    // Still processing, update status
    var minutesElapsed = Math.floor((new Date().getTime() - startTime) / (60 * 1000));
    var statusMessage = `${operationType === "categorise" ? "Categorisation" : "Training"} in progress... (${minutesElapsed} min)`;
    
    if (result.status) {
      statusMessage += ` - ${result.status}`;
    }
    if (result.message) {
      statusMessage += `\n${result.message}`;
    }
    
    // Add progress information if available
    if (result.processed_transactions && result.total_transactions) {
      statusMessage += ` (${result.processed_transactions}/${result.total_transactions})`;
    }
    
    updateStatus(statusMessage);
    
  } catch (error) {
    Logger.log(`Error checking ${operationType} status: ` + error);
    Logger.log("Error details: " + error.stack);
    
    // Only cleanup on non-temporary errors
    if (error.toString().includes("Address unavailable") || error.toString().includes("Failed to fetch")) {
      var minutesElapsed = Math.floor((new Date().getTime() - startTime) / (60 * 1000));
      updateStatus(`${operationType} in progress... (${minutesElapsed} min) - temporary connection issue`);
      return;
    }
    
    cleanupTrigger(triggerId);
    userProperties.deleteAllProperties();
    updateStatus("Error: " + error.toString());
  }
}

function handleClassificationResults(result, config, originalSheetName) {
  try {
    Logger.log("Handling classification results: " + JSON.stringify(result));
    Logger.log("Config: " + JSON.stringify(config));
    Logger.log("Original sheet name: " + originalSheetName);
    
    // Get the original sheet by name
    var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = spreadsheet.getSheetByName(originalSheetName);
    if (!sheet) {
      Logger.log("Sheet not found by name, trying active sheet");
      sheet = SpreadsheetApp.getActiveSheet();
      if (sheet.getName() !== originalSheetName) {
        Logger.log("Warning: Using active sheet instead of original sheet");
        updateStatus("Warning: Using active sheet instead of original sheet", "Original sheet: " + originalSheetName);
      }
    }
    
    // Check if we have webhook results
    if (!result.result || !result.result.results) {
      Logger.log("No webhook results found in response");
      updateStatus("Categorisation completed, but no results were returned", "Check the Log sheet for details");
      return;
    }
    
    // For newer webhook format, we might not have individual results
    // Just a success status
    if (result.result.results.status === "success") {
      Logger.log("Webhook reported success but no detailed results");
      updateStatus("Categorisation completed successfully!", "Results were written directly by the webhook");
      return;
    }
    
    // For older webhook format with detailed results
    var webhookResults = result.result.results;
    if (Array.isArray(webhookResults) && webhookResults.length > 0) {
      // Write categories and confidence scores
      var startRow = parseInt(config.startRow);
      var endRow = startRow + webhookResults.length - 1;
      
      Logger.log("Writing " + webhookResults.length + " results to sheet");
      Logger.log("Start row: " + startRow + ", End row: " + endRow);
      
      // Write categories
      var categoryRange = sheet.getRange(config.categoryCol + startRow + ":" + config.categoryCol + endRow);
      categoryRange.setValues(webhookResults.map(r => [r.predicted_category]));
      
      // Write confidence scores if they exist
      if (webhookResults[0].hasOwnProperty('similarity_score')) {
        var confidenceCol = String.fromCharCode(config.categoryCol.charCodeAt(0) + 1);
        var confidenceRange = sheet.getRange(confidenceCol + startRow + ":" + confidenceCol + endRow);
        confidenceRange.setValues(webhookResults.map(r => [r.similarity_score]))
          .setNumberFormat("0.00%");
      }
      
      updateStatus("Categorisation completed successfully!");
    } else {
      Logger.log("No valid results array found in webhook response");
      updateStatus("Categorisation completed, but no valid results were found", "Check the Log sheet for details");
    }
  } catch (error) {
    Logger.log("Error handling categorisation results: " + error);
    Logger.log("Error stack: " + error.stack);
    updateStatus("Error handling categorisation results: " + error.toString());
    throw error;
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
    // Store original sheet name
    var originalSheetName = SpreadsheetApp.getActiveSheet().getName();
    
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
        'RETRY_COUNT': '0',  // Initialize retry count
        'ORIGINAL_SHEET_NAME': originalSheetName  // Store original sheet name
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