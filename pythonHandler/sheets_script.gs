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
        startRow: startRow.toString(),  // Pass the start row explicitly
        categoryColumn: config.categoryCol  // Pass the category column explicitly
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
        'LAST_CHECK_TIME': '0',
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
      
      // Create a polling UI that will check status automatically
      var pollingHtml = HtmlService.createHtmlOutput(`
        <style>
          body { font-family: Arial, sans-serif; padding: 20px; }
          #status { margin: 20px 0; padding: 10px; background: #f5f5f5; border-radius: 4px; }
          .progress-container { 
            width: 100%; 
            background-color: #f1f1f1; 
            border-radius: 4px;
            margin: 10px 0;
          }
          .progress-bar {
            width: 0%;
            height: 20px;
            background-color: #4CAF50;
            border-radius: 4px;
            text-align: center;
            line-height: 20px;
            color: white;
          }
          .success { color: green; }
          .error { color: red; }
          .info { color: blue; }
          .warning { color: orange; }
        </style>
        <h3>Categorisation Status</h3>
        <div id="status">Initializing...</div>
        <div class="progress-container">
          <div id="progress-bar" class="progress-bar">0%</div>
        </div>
        <p id="message">Starting categorisation process...</p>
        <script>
          // Poll for status every 2 seconds
          var pollInterval = 2000;
          var startTime = new Date().getTime();
          var maxTime = 30 * 60 * 1000; // 30 minutes timeout
          var completed = false;
          var errorCount = 0;
          var maxErrors = 5; // Maximum consecutive errors before showing a warning
          
          function updateProgress(percent) {
            document.getElementById('progress-bar').style.width = percent + '%';
            document.getElementById('progress-bar').innerText = percent + '%';
          }
          
          function checkStatus() {
            if (completed) return;
            
            var elapsedTime = new Date().getTime() - startTime;
            if (elapsedTime > maxTime) {
              document.getElementById('status').innerHTML = '<span class="error">Error: Operation timed out after 30 minutes</span>';
              document.getElementById('message').innerText = 'Please try again or contact support.';
              completed = true;
              return;
            }
            
            google.script.run
              .withSuccessHandler(function(result) {
                if (!result) {
                  // No result yet, keep polling
                  document.getElementById('status').innerText = 'Processing...';
                  setTimeout(checkStatus, pollInterval);
                  return;
                }
                
                // Reset error count on successful response
                errorCount = 0;
                
                if (result.error && result.status !== "in_progress") {
                  document.getElementById('status').innerHTML = '<span class="error">Error: ' + result.error + '</span>';
                  document.getElementById('message').innerText = 'Please try again or contact support.';
                  completed = true;
                  return;
                }
                
                if (result.status === 'completed') {
                  document.getElementById('status').innerHTML = '<span class="success">Categorisation completed successfully!</span>';
                  document.getElementById('message').innerText = 'You can close this window.';
                  updateProgress(100);
                  completed = true;
                  
                  // Close the dialog after 3 seconds
                  setTimeout(function() {
                    google.script.host.close();
                  }, 3000);
                  return;
                }
                
                // Update progress information
                var progressPercent = 0;
                if (result.processed_transactions && result.total_transactions) {
                  progressPercent = Math.round((result.processed_transactions / result.total_transactions) * 100);
                } else {
                  // Estimate progress based on time (up to 90%)
                  var minutesElapsed = elapsedTime / (60 * 1000);
                  progressPercent = Math.min(90, Math.round(minutesElapsed * 10)); // Roughly 10% per minute up to 90%
                }
                updateProgress(progressPercent);
                
                // Update status message
                var statusMessage = result.message || 'Processing...';
                document.getElementById('status').innerText = statusMessage;
                
                // If there's an error but we're still in progress, show it as a warning
                if (result.error && result.status === "in_progress") {
                  document.getElementById('message').innerHTML = '<span class="warning">Note: ' + result.error + '</span>';
                }
                
                // Continue polling
                setTimeout(checkStatus, pollInterval);
              })
              .withFailureHandler(function(error) {
                // Increment error count
                errorCount++;
                
                // If we've had too many consecutive errors, show a warning but keep trying
                if (errorCount >= maxErrors) {
                  document.getElementById('status').innerHTML = '<span class="warning">Warning: Multiple errors checking status</span>';
                  document.getElementById('message').innerHTML = '<span class="warning">The operation may still be in progress. Will continue trying...</span>';
                } else {
                  document.getElementById('status').innerHTML = '<span class="warning">Error checking status: ' + error + '</span>';
                  document.getElementById('message').innerText = 'Will try again in a few seconds...';
                }
                
                // Continue polling despite error, with exponential backoff
                var backoffInterval = pollInterval * Math.min(4, errorCount);
                setTimeout(checkStatus, backoffInterval);
              })
              .pollOperationStatus();
          }
          
          // Start polling immediately
          checkStatus();
        </script>
      `)
        .setWidth(450)
        .setHeight(300);
      
      SpreadsheetApp.getUi().showModalDialog(pollingHtml, 'Categorisation Progress');
      return;
    }
  } catch (error) {
    Logger.log("Categorisation error: " + error.toString());
    updateStatus("Error: " + error.toString());
    ui.alert('Error: ' + error.toString());
  }
}

// Helper function to write classification results to a sheet
function writeResultsToSheet(result, config, sheet) {
  try {
    Logger.log("Writing results to sheet with config: " + JSON.stringify(config));
    
    // Check if we have results directly in the response
    if (result.results && Array.isArray(result.results)) {
      Logger.log("Found results directly in response");
      var webhookResults = result.results;
      var categoryCol = result.config ? result.config.categoryColumn : config.categoryCol;
      var startRow = result.config ? parseInt(result.config.startRow) : parseInt(config.startRow);
      
      // Write categories and confidence scores
      var endRow = startRow + webhookResults.length - 1;
      
      Logger.log("Writing " + webhookResults.length + " results to sheet");
      Logger.log("Start row: " + startRow + ", End row: " + endRow);
      
      // Write categories
      var categoryRange = sheet.getRange(categoryCol + startRow + ":" + categoryCol + endRow);
      categoryRange.setValues(webhookResults.map(r => [r.predicted_category]));
      
      // Write confidence scores if they exist
      if (webhookResults[0].hasOwnProperty('similarity_score')) {
        var confidenceCol = String.fromCharCode(categoryCol.charCodeAt(0) + 1);
        var confidenceRange = sheet.getRange(confidenceCol + startRow + ":" + confidenceCol + endRow);
        confidenceRange.setValues(webhookResults.map(r => [r.similarity_score]))
          .setNumberFormat("0.00%");
      }
      
      updateStatus("Categorisation completed successfully!");
      return true;
    }
    
    // Check if we have results in the result.result.results.data format
    if (result.result && result.result.results && result.result.results.data && Array.isArray(result.result.results.data)) {
      Logger.log("Found results in result.result.results.data format");
      var webhookResults = result.result.results.data;
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
      return true;
    }
    
    // For newer webhook format with success status but results elsewhere
    if (result.result && result.result.results && result.result.results.status === "success") {
      Logger.log("Webhook reported success but checking for results elsewhere");
      // Check if we have results elsewhere in the response
      if (result.results && Array.isArray(result.results)) {
        Logger.log("Found results in result.results format");
        var webhookResults = result.results;
        var categoryCol = result.config ? result.config.categoryColumn : config.categoryCol;
        var startRow = result.config ? parseInt(result.config.startRow) : parseInt(config.startRow);
        
        // Write categories and confidence scores
        var endRow = startRow + webhookResults.length - 1;
        
        Logger.log("Writing " + webhookResults.length + " results to sheet");
        Logger.log("Start row: " + startRow + ", End row: " + endRow);
        
        // Write categories
        var categoryRange = sheet.getRange(categoryCol + startRow + ":" + categoryCol + endRow);
        categoryRange.setValues(webhookResults.map(r => [r.predicted_category]));
        
        // Write confidence scores if they exist
        if (webhookResults[0].hasOwnProperty('similarity_score')) {
          var confidenceCol = String.fromCharCode(categoryCol.charCodeAt(0) + 1);
          var confidenceRange = sheet.getRange(confidenceCol + startRow + ":" + confidenceCol + endRow);
          confidenceRange.setValues(webhookResults.map(r => [r.similarity_score]))
            .setNumberFormat("0.00%");
        }
        
        updateStatus("Categorisation completed successfully!");
        return true;
      }
      
      updateStatus("Categorisation completed, but no results were found to write to the sheet", "Check the Log sheet for details");
      return false;
    }
    
    // No results found in any expected format
    if (!result.result || !result.result.results) {
      Logger.log("No webhook results found in response");
      updateStatus("Categorisation completed, but no results were returned", "Check the Log sheet for details");
      return false;
    }
    
    return false;
  } catch (error) {
    Logger.log("Error writing results to sheet: " + error);
    Logger.log("Error stack: " + error.stack);
    updateStatus("Error writing results to sheet: " + error.toString());
    return false;
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
    
    // Use the helper function to write results to the sheet
    if (!writeResultsToSheet(result, config, sheet)) {
      updateStatus("Categorisation completed, but no results were found to write to the sheet", "Check the Log sheet for details");
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
      
      // Store prediction ID and other properties in user properties
      var userProperties = PropertiesService.getUserProperties();
      
      // Clear any existing properties
      userProperties.deleteAllProperties();
      
      // Store new properties
      userProperties.setProperties({
        'PREDICTION_ID': result.prediction_id,
        'START_TIME': new Date().getTime().toString(),
        'TRAINING_SIZE': transactions.length.toString(),
        'ORIGINAL_SHEET_NAME': originalSheetName,
        'OPERATION_TYPE': 'train'
      });
      
      // Create a polling UI that will check status automatically
      var pollingHtml = HtmlService.createHtmlOutput(`
        <style>
          body { font-family: Arial, sans-serif; padding: 20px; }
          #status { margin: 20px 0; padding: 10px; background: #f5f5f5; border-radius: 4px; }
          .progress-container { 
            width: 100%; 
            background-color: #f1f1f1; 
            border-radius: 4px;
            margin: 10px 0;
          }
          .progress-bar {
            width: 0%;
            height: 20px;
            background-color: #4CAF50;
            border-radius: 4px;
            text-align: center;
            line-height: 20px;
            color: white;
          }
          .success { color: green; }
          .error { color: red; }
          .info { color: blue; }
          .warning { color: orange; }
        </style>
        <h3>Training Status</h3>
        <div id="status">Initializing...</div>
        <div class="progress-container">
          <div id="progress-bar" class="progress-bar">0%</div>
        </div>
        <p id="message">Starting training process...</p>
        <script>
          // Poll for status every 2 seconds
          var pollInterval = 2000;
          var startTime = new Date().getTime();
          var maxTime = 30 * 60 * 1000; // 30 minutes timeout
          var completed = false;
          var errorCount = 0;
          var maxErrors = 5; // Maximum consecutive errors before showing a warning
          
          function updateProgress(percent) {
            document.getElementById('progress-bar').style.width = percent + '%';
            document.getElementById('progress-bar').innerText = percent + '%';
          }
          
          function checkStatus() {
            if (completed) return;
            
            var elapsedTime = new Date().getTime() - startTime;
            if (elapsedTime > maxTime) {
              document.getElementById('status').innerHTML = '<span class="error">Error: Operation timed out after 30 minutes</span>';
              document.getElementById('message').innerText = 'Please try again or contact support.';
              completed = true;
              return;
            }
            
            google.script.run
              .withSuccessHandler(function(result) {
                if (!result) {
                  // No result yet, keep polling
                  document.getElementById('status').innerText = 'Processing...';
                  setTimeout(checkStatus, pollInterval);
                  return;
                }
                
                // Reset error count on successful response
                errorCount = 0;
                
                if (result.error && result.status !== "in_progress") {
                  document.getElementById('status').innerHTML = '<span class="error">Error: ' + result.error + '</span>';
                  document.getElementById('message').innerText = 'Please try again or contact support.';
                  completed = true;
                  return;
                }
                
                if (result.status === 'completed') {
                  document.getElementById('status').innerHTML = '<span class="success">Training completed successfully!</span>';
                  document.getElementById('message').innerText = 'You can close this window.';
                  updateProgress(100);
                  completed = true;
                  
                  // Close the dialog after 3 seconds
                  setTimeout(function() {
                    google.script.host.close();
                  }, 3000);
                  return;
                }
                
                // Update progress information
                var progressPercent = 0;
                if (result.processed_transactions && result.total_transactions) {
                  progressPercent = Math.round((result.processed_transactions / result.total_transactions) * 100);
                } else {
                  // Estimate progress based on time (up to 90%)
                  var minutesElapsed = elapsedTime / (60 * 1000);
                  progressPercent = Math.min(90, Math.round(minutesElapsed * 10)); // Roughly 10% per minute up to 90%
                }
                updateProgress(progressPercent);
                
                // Update status message
                var statusMessage = result.message || 'Processing...';
                document.getElementById('status').innerText = statusMessage;
                
                // If there's an error but we're still in progress, show it as a warning
                if (result.error && result.status === "in_progress") {
                  document.getElementById('message').innerHTML = '<span class="warning">Note: ' + result.error + '</span>';
                }
                
                // Continue polling
                setTimeout(checkStatus, pollInterval);
              })
              .withFailureHandler(function(error) {
                // Increment error count
                errorCount++;
                
                // If we've had too many consecutive errors, show a warning but keep trying
                if (errorCount >= maxErrors) {
                  document.getElementById('status').innerHTML = '<span class="warning">Warning: Multiple errors checking status</span>';
                  document.getElementById('message').innerHTML = '<span class="warning">The operation may still be in progress. Will continue trying...</span>';
                } else {
                  document.getElementById('status').innerHTML = '<span class="warning">Error checking status: ' + error + '</span>';
                  document.getElementById('message').innerText = 'Will try again in a few seconds...';
                }
                
                // Continue polling despite error, with exponential backoff
                var backoffInterval = pollInterval * Math.min(4, errorCount);
                setTimeout(checkStatus, backoffInterval);
              })
              .pollOperationStatus();
          }
          
          // Start polling immediately
          checkStatus();
        </script>
      `)
        .setWidth(450)
        .setHeight(300);
      
      SpreadsheetApp.getUi().showModalDialog(pollingHtml, 'Training Progress');
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

// Function to handle client-side polling for operation status
function pollOperationStatus() {
  try {
    var userProperties = PropertiesService.getUserProperties();
    var predictionId = userProperties.getProperty('PREDICTION_ID');
    var operationType = userProperties.getProperty('OPERATION_TYPE');
    var startTime = parseInt(userProperties.getProperty('START_TIME'));
    var originalSheetName = userProperties.getProperty('ORIGINAL_SHEET_NAME');
    var config = userProperties.getProperty('CONFIG') ? JSON.parse(userProperties.getProperty('CONFIG')) : null;
    var currentTime = new Date().getTime();
    
    // If no prediction ID, return error
    if (!predictionId) {
      return { error: "No operation in progress" };
    }
    
    // Check if we've been running for more than 30 minutes
    if (currentTime - startTime > 30 * 60 * 1000) {
      var timeoutMessage = `${operationType === 'categorise' ? 'Categorisation' : 'Training'} timed out after 30 minutes`;
      updateStatus(timeoutMessage);
      userProperties.deleteAllProperties();
      return { 
        error: timeoutMessage,
        status: "timeout" 
      };
    }
    
    // Get service config
    var serviceConfig;
    try {
      serviceConfig = getServiceConfig();
    } catch (configError) {
      updateStatus("Error: " + configError.toString());
      return { error: configError.toString() };
    }
    
    // Call the service to check status
    var options = {
      headers: { 'X-API-Key': serviceConfig.apiKey },
      muteHttpExceptions: true,
      timeout: 10000 // 10 second timeout
    };
    
    var response;
    try {
      response = UrlFetchApp.fetch(serviceConfig.serviceUrl + '/status/' + predictionId, options);
    } catch (fetchError) {
      var connectionMessage = `${operationType === 'categorise' ? 'Categorisation' : 'Training'} in progress... (temporary connection issue)`;
      updateStatus(connectionMessage);
      return { 
        message: connectionMessage,
        status: "in_progress" 
      };
    }
    
    // Check HTTP response code
    var responseCode = response.getResponseCode();
    if (responseCode !== 200) {
      // For 500 errors, the server might be having issues but the operation could still be in progress
      var httpErrorMessage = `${operationType === 'categorise' ? 'Categorisation' : 'Training'} in progress... (service returned ${responseCode})`;
      updateStatus(httpErrorMessage);
      
      // For persistent 500 errors, we might need to check if the operation is actually still valid
      // We'll continue polling but log the issue
      Logger.log("Server returned error code: " + responseCode + " for prediction ID: " + predictionId);
      
      // If we've been getting errors for a while, we might want to check a different endpoint
      // or use a fallback mechanism, but for now we'll just continue polling
      return { 
        message: httpErrorMessage,
        status: "in_progress" 
      };
    }
    
    // Parse the response
    var result;
    try {
      result = JSON.parse(response.getContentText());
    } catch (parseError) {
      var parseErrorMessage = `${operationType === 'categorise' ? 'Categorisation' : 'Training'} in progress... (parsing response)`;
      updateStatus(parseErrorMessage);
      return { 
        message: parseErrorMessage,
        status: "in_progress" 
      };
    }
    
    // Get the original sheet by name
    var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = spreadsheet.getSheetByName(originalSheetName);
    if (!sheet) {
      sheet = SpreadsheetApp.getActiveSheet();
    }
    
    // Handle completed status
    if (result.status === "completed") {
      // Try to write results to sheet
      if (operationType === 'categorise' && writeResultsToSheet(result, config, sheet)) {
        var completedMessage = "Categorisation completed successfully!";
        updateStatus(completedMessage);
        userProperties.deleteAllProperties();
        return { 
          status: "completed",
          message: completedMessage 
        };
      } else if (operationType === 'train') {
        // Update training stats
        updateStats('Last Training Time', new Date().toLocaleString());
        updateStats('Training Data Size', sheet.getLastRow() - 1);
        updateStats('Training Sheet', sheet.getName());
        updateStats('Model Status', 'Ready');
        
        var trainingCompletedMessage = "Training completed successfully!";
        updateStatus(trainingCompletedMessage);
        userProperties.deleteAllProperties();
        return { 
          status: "completed",
          message: trainingCompletedMessage 
        };
      }
    } else if (result.status === "failed") {
      var errorMessage = result.error || "Unknown error";
      updateStatus("Error: " + errorMessage);
      userProperties.deleteAllProperties();
      return { 
        error: errorMessage,
        status: "failed" 
      };
    }
    
    // Still in progress, return status information
    var minutesElapsed = Math.floor((currentTime - startTime) / (60 * 1000));
    var statusMessage = `${operationType === 'categorise' ? 'Categorisation' : 'Training'} in progress... (${minutesElapsed} min)`;
    
    if (result.status) {
      statusMessage += ` - ${result.status}`;
    }
    if (result.message) {
      statusMessage += ` - ${result.message}`;
    }
    
    // Add progress information if available
    var progressInfo = {};
    var additionalDetails = "";
    if (result.processed_transactions && result.total_transactions) {
      progressInfo.processed_transactions = result.processed_transactions;
      progressInfo.total_transactions = result.total_transactions;
      additionalDetails = `Progress: ${result.processed_transactions}/${result.total_transactions} transactions (${Math.round((result.processed_transactions / result.total_transactions) * 100)}%)`;
    }
    
    // Update the log with the current status
    updateStatus(statusMessage, additionalDetails);
    
    return { 
      status: "in_progress",
      message: statusMessage,
      ...progressInfo
    };
  } catch (error) {
    Logger.log("Error in pollOperationStatus: " + error);
    updateStatus("Error in pollOperationStatus: " + error.toString());
    
    // Even if we encounter an error, we should return something that allows the UI to continue polling
    // rather than stopping the process entirely
    return { 
      status: "in_progress",
      message: "Processing... (encountered an error, will retry)",
      error: error.toString()
    };
  }
} 