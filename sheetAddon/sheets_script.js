// Service configuration
const CLASSIFICATION_SERVICE_URL = "https://txclassify.onrender.com";

// Add menu to the spreadsheet
function onOpen(e) {
  var menu = SpreadsheetApp.getUi().createAddonMenu(); // Changed from addMenu to createAddonMenu
  menu
    .addItem("Configure API Key", "setupApiKey")
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
  var html = HtmlService.createHtmlOutput(
    `
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
  `
  )
    .setWidth(400)
    .setHeight(350);

  SpreadsheetApp.getUi().showModalDialog(html, "Categorise Transactions");
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
    var header = headers[i] || ""; // Use empty string if header is null/undefined
    columnOptions.push(
      `<option value="${letter}"${letter === narrativeColDefault ? " selected" : ""}>` +
        `${letter}${header ? " (" + header + ")" : ""}</option>`
    );
  }
  var columnOptionsStr = columnOptions.join("\n");

  // Create category column options with E as default
  var categoryColumnOptions = [];
  for (var i = 0; i < lastColumn; i++) {
    var letter = columnToLetter(i + 1);
    var header = headers[i] || ""; // Use empty string if header is null/undefined
    categoryColumnOptions.push(
      `<option value="${letter}"${letter === "E" ? " selected" : ""}>` +
        `${letter}${header ? " (" + header + ")" : ""}</option>`
    );
  }
  var categoryColumnOptionsStr = categoryColumnOptions.join("\n");

  var html = HtmlService.createHtmlOutput(
    `
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
        ${categoryColumnOptionsStr}
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
  `
  )
    .setWidth(400)
    .setHeight(350);

  SpreadsheetApp.getUi().showModalDialog(html, "Train Model");
}

// Setup function to configure API key
function setupApiKey() {
  var properties = PropertiesService.getScriptProperties();
  var existingApiKey = properties.getProperty("API_KEY") || "";

  var html = HtmlService.createHtmlOutput(
    `
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
    
    ${
      existingApiKey
        ? `
    <div class="current-key">
      <div class="key-status">Current API Key:</div>
      ${existingApiKey}
    </div>
    `
        : ""
    }
    
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
  `
  )
    .setWidth(500)
    .setHeight(450);

  SpreadsheetApp.getUi().showModalDialog(html, "API Key Management");
}

// Helper function to save the API key
function saveApiKey(apiKey) {
  if (!apiKey) {
    throw new Error("API key is required");
  }

  var properties = PropertiesService.getScriptProperties();
  properties.setProperty("API_KEY", apiKey.trim());
  updateStatus("API key configured successfully!");
}

// Helper function to get stored properties
function getServiceConfig() {
  var properties = PropertiesService.getScriptProperties();
  var apiKey = properties.getProperty("API_KEY");

  if (!apiKey) {
    throw new Error(
      'API key not configured. Please go to expensesorted.com to get your API key, then use "Configure API Key" to set it up.'
    );
  }

  return {
    serviceUrl: CLASSIFICATION_SERVICE_URL,
    apiKey: apiKey,
    userId: Session.getEffectiveUser().getEmail(), // Use the user's email as userId
  };
}

// Helper function to update status in sheet
function updateStatus(message, additionalDetails = "") {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var logSheet = ss.getSheetByName("Log");

  // Create Log sheet if it doesn't exist
  if (!logSheet) {
    logSheet = ss.insertSheet("Log");
    // Add headers
    logSheet.getRange("A1:D1").setValues([["Timestamp", "Status", "Message", "Details"]]);
    logSheet.setFrozenRows(1);
    // Set column widths
    logSheet.setColumnWidth(1, 180); // Timestamp
    logSheet.setColumnWidth(2, 100); // Status
    logSheet.setColumnWidth(3, 300); // Message
    logSheet.setColumnWidth(4, 400); // Details

    // Format headers
    var headerRange = logSheet.getRange("A1:D1");
    headerRange.setBackground("#f3f3f3").setFontWeight("bold").setHorizontalAlignment("center");
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
  logSheet.getRange("A2:D2").setValues([[timestamp, status, message, contextDetails]]);

  // Color coding
  var statusCell = logSheet.getRange("B2");
  switch (status) {
    case "ERROR":
      statusCell.setBackground("#ffcdd2"); // Light red
      break;
    case "SUCCESS":
      statusCell.setBackground("#c8e6c9"); // Light green
      break;
    case "PROCESSING":
      statusCell.setBackground("#fff9c4"); // Light yellow
      break;
    default:
      statusCell.setBackground("#ffffff"); // White
  }

  // Format the new row
  logSheet.getRange("A2:D2").setHorizontalAlignment("left").setVerticalAlignment("middle").setWrap(true);

  // Keep only last 100 entries
  var lastRow = logSheet.getLastRow();
  if (lastRow > 101) {
    // 1 header row + 100 log entries
    logSheet.deleteRows(102, lastRow - 101);
  }

  // Make sure the Log sheet is visible
  logSheet.autoResizeColumns(1, 4);

  // Show the Log sheet if it's hidden
  if (logSheet.isSheetHidden()) {
    logSheet.showSheet();
  }
}

// Helper function to manage settings sheet
function getSettingsSheet() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var settingsSheet = ss.getSheetByName("Settings");

  // Create Settings sheet if it doesn't exist
  if (!settingsSheet) {
    settingsSheet = ss.insertSheet("Settings");
    // Add headers
    settingsSheet.getRange("A1:B1").setValues([["Setting", "Value"]]);
    settingsSheet.setFrozenRows(1);
    // Set column widths
    settingsSheet.setColumnWidth(1, 200); // Setting name
    settingsSheet.setColumnWidth(2, 300); // Value
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
  var statsSheet = ss.getSheetByName("Stats");

  // Create Stats sheet if it doesn't exist
  if (!statsSheet) {
    statsSheet = ss.insertSheet("Stats");
    // Add headers
    statsSheet.getRange("A1:B1").setValues([["Metric", "Value"]]);
    statsSheet.setFrozenRows(1);

    // Set column widths
    statsSheet.setColumnWidth(1, 200); // Metric name
    statsSheet.setColumnWidth(2, 300); // Value

    // Format headers
    var headerRange = statsSheet.getRange("A1:B1");
    headerRange.setBackground("#f3f3f3").setFontWeight("bold").setHorizontalAlignment("center");

    // Add initial metrics
    statsSheet
      .getRange("A2:A5")
      .setValues([["Last Training Time"], ["Training Data Size"], ["Training Sheet"], ["Model Status"]]);
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
  try {
    Logger.log("Starting categorisation with config: " + JSON.stringify(config));
    updateStatus("Starting categorisation...");

    var sheet = SpreadsheetApp.getActiveSheet();
    var serviceConfig = getServiceConfig();
    var spreadsheetId = sheet.getParent().getId();
    var lastRow = sheet.getLastRow();
    var startRow = parseInt(config.startRow);

    // Get descriptions from the specified column
    var descriptions = sheet
      .getRange(config.descriptionCol + startRow + ":" + config.descriptionCol + lastRow)
      .getValues()
      .filter((row) => row[0] && row[0].toString().trim() !== "")
      .map((row) => ({ Narrative: row[0] }));

    if (descriptions.length === 0) {
      throw new Error("No transactions found to categorise");
    }

    Logger.log("Found " + descriptions.length + " transactions to categorise");
    updateStatus("Processing " + descriptions.length + " transactions...");

    // Make API call
    var response = UrlFetchApp.fetch(serviceConfig.serviceUrl + "/classify", {
      method: "post",
      contentType: "application/json",
      headers: { "X-API-Key": serviceConfig.apiKey },
      payload: JSON.stringify({
        transactions: descriptions,
        spreadsheetId: spreadsheetId,
        userId: serviceConfig.userId,
        startRow: startRow.toString(),
        categoryColumn: config.categoryCol,
      }),
      muteHttpExceptions: true,
    });

    if (response.getResponseCode() !== 200) {
      var error = JSON.parse(response.getContentText());
      throw new Error(error.error || "Server returned error code: " + response.getResponseCode());
    }

    var result = JSON.parse(response.getContentText());

    if (!result.prediction_id) {
      throw new Error("No prediction ID received from server");
    }

    // Store properties for polling
    var userProperties = PropertiesService.getUserProperties();
    userProperties.setProperties({
      PREDICTION_ID: result.prediction_id,
      OPERATION_TYPE: "categorise",
      START_TIME: Date.now().toString(),
      CONFIG: JSON.stringify(config),
    });

    // Show progress dialog
    var html = HtmlService.createHtmlOutput(
      `
      <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        .progress-bar {
          width: 100%;
          background: #f0f0f0;
          border-radius: 4px;
          margin: 10px 0;
        }
        .progress-fill {
          width: 0%;
          height: 20px;
          background: #4CAF50;
          border-radius: 4px;
          transition: width 0.5s;
        }
        .status { margin: 10px 0; }
        .error { color: red; }
      </style>
      <div class="status" id="status">Processing ${descriptions.length} transactions...</div>
      <div class="progress-bar">
        <div class="progress-fill" id="progress"></div>
      </div>
      <script>
        const POLL_INTERVAL = 5000;
        const MAX_POLLS = 120;
        let pollCount = 0;

        function updateProgress(percent, message) {
          document.getElementById("progress").style.width = percent + "%";
          if (message) {
            document.getElementById("status").textContent = message;
          }
        }

        function pollStatus() {
          if (pollCount >= MAX_POLLS) {
            updateProgress(100, "Categorisation completed!");
            setTimeout(() => google.script.host.close(), 2000);
            return;
          }

          pollCount++;
          
          google.script.run
            .withSuccessHandler(function(result) {
              if (result.status === "completed") {
                updateProgress(100, "Categorisation completed successfully!");
                setTimeout(() => google.script.host.close(), 2000);
                return;
              }
              
              if (result.status === "failed") {
                document.getElementById("status").innerHTML = '<span class="error">Error: ' + result.message + '</span>';
                return;
              }

              updateProgress(result.progress, result.message);
              setTimeout(pollStatus, POLL_INTERVAL);
            })
            .withFailureHandler(function(error) {
              updateProgress(50, "Processing continues...");
              setTimeout(pollStatus, POLL_INTERVAL);
            })
            .pollOperationStatus();
        }

        // Start polling immediately
        pollStatus();
      </script>
    `
    )
      .setWidth(400)
      .setHeight(150);

    SpreadsheetApp.getUi().showModalDialog(html, "Categorisation Progress");
  } catch (error) {
    Logger.log("Categorisation error: " + error.toString());
    updateStatus("Error: " + error.toString());
    SpreadsheetApp.getUi().alert("Error: " + error.toString());
  }
}

// Helper function to write classification results to a sheet
function writeResultsToSheet(result, config, sheet) {
  try {
    Logger.log("Writing results to sheet with config: " + JSON.stringify(config));
    Logger.log("Result object: " + JSON.stringify(result));

    // Get results array from the response
    var webhookResults;
    if (result.results && Array.isArray(result.results)) {
      webhookResults = result.results;
    } else if (result.result && result.result.results && Array.isArray(result.result.results)) {
      webhookResults = result.result.results;
    } else if (result.result && result.result.results && result.result.results.data) {
      webhookResults = result.result.results.data;
    } else {
      Logger.log("No valid results array found in response");
      return false;
    }

    if (!webhookResults || webhookResults.length === 0) {
      Logger.log("No results to write");
      return false;
    }

    // Get configuration
    var categoryCol = result.config ? result.config.categoryColumn : config.categoryCol;
    var startRow = result.config ? parseInt(result.config.startRow) : parseInt(config.startRow);
    var endRow = startRow + webhookResults.length - 1;

    Logger.log("Writing " + webhookResults.length + " results to sheet");
    Logger.log("Category column: " + categoryCol + ", Start row: " + startRow + ", End row: " + endRow);

    // Write categories
    var categoryRange = sheet.getRange(categoryCol + startRow + ":" + categoryCol + endRow);
    categoryRange.setValues(webhookResults.map((r) => [r.predicted_category || r.category || ""]));

    // Write confidence scores if they exist
    if (webhookResults[0].hasOwnProperty("similarity_score")) {
      var confidenceCol = String.fromCharCode(categoryCol.charCodeAt(0) + 1);
      var confidenceRange = sheet.getRange(confidenceCol + startRow + ":" + confidenceCol + endRow);
      confidenceRange.setValues(webhookResults.map((r) => [r.similarity_score || 0])).setNumberFormat("0.00%");
    }

    updateStatus("Categorisation completed successfully!");
    return true;
  } catch (error) {
    Logger.log("Error writing results to sheet: " + error);
    Logger.log("Error stack: " + error.stack);
    updateStatus("Error writing results to sheet: " + error.toString());
    return false;
  }
}

// Helper function to convert column number to letter
function columnToLetter(column) {
  var temp,
    letter = "";
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
    // Validate config and get data first
    if (!config || !config.narrativeCol || !config.categoryCol || !config.startRow) {
      throw new Error("Missing required configuration parameters");
    }

    var serviceConfig = getServiceConfig();
    var sheet = SpreadsheetApp.getActiveSheet();
    var originalSheetName = sheet.getName();
    var lastRow = sheet.getLastRow();

    // Get data from the selected columns
    var narrativeRange = sheet.getRange(config.narrativeCol + config.startRow + ":" + config.narrativeCol + lastRow);
    var categoryRange = sheet.getRange(config.categoryCol + config.startRow + ":" + config.categoryCol + lastRow);
    var narratives = narrativeRange.getValues();
    var categories = categoryRange.getValues();

    // Prepare training data
    var transactions = [];
    for (var i = 0; i < narratives.length; i++) {
      if (narratives[i][0] && categories[i][0]) {
        transactions.push({
          Narrative: narratives[i][0],
          Category: categories[i][0],
        });
      }
    }

    if (transactions.length === 0) {
      throw new Error("No training data found");
    }

    // Store data needed for training
    var userProperties = PropertiesService.getUserProperties();
    userProperties.setProperties({
      TEMP_TRANSACTIONS: JSON.stringify(transactions),
      TEMP_CONFIG: JSON.stringify(config),
      TEMP_SHEET_NAME: originalSheetName,
      TEMP_SERVICE_URL: serviceConfig.serviceUrl,
    });

    // Show progress dialog BEFORE making the API call
    showTrainingProgress(transactions, config);
  } catch (error) {
    Logger.log("Training error: " + error.toString());
    updateStatus("Error: " + error.toString());
    ui.alert("Error: " + error.toString());
  }
}

function showTrainingProgress(transactions, config) {
  var html = HtmlService.createHtmlOutput(
    `
    <style>
      body { font-family: Arial, sans-serif; padding: 20px; }
      .progress-bar {
        width: 100%;
        background: #f0f0f0;
        border-radius: 4px;
        margin: 10px 0;
      }
      .progress-fill {
        width: 0%;
        height: 20px;
        background: #4CAF50;
        border-radius: 4px;
        transition: width 0.5s;
      }
      .status { margin: 10px 0; }
      .error { color: red; }
      .warning { color: orange; }
    </style>
    <div class="status" id="status">Preparing training data...</div>
    <div class="progress-bar">
      <div class="progress-fill" id="progress"></div>
    </div>
    <script>
      const POLL_INTERVAL = 5000;  // Fixed 5-second interval
      const MAX_POLLS = 120;       // 10 minutes maximum
      const RETRY_DELAY = 10000;   // 10 seconds between retries
      let pollCount = 0;
      let errorCount = 0;
      let lastProgress = 0;

      function updateProgress(percent, message, isWarning) {
        document.getElementById("progress").style.width = percent + "%";
        if (message) {
          document.getElementById("status").innerHTML = isWarning ? 
            '<span class="warning">' + message + '</span>' : 
            message;
        }
        lastProgress = percent;
      }

      function startTraining() {
        updateProgress(5, "Sending training data to server...");
        
        google.script.run
          .withSuccessHandler(function(result) {
            if (result.error) {
              document.getElementById("status").innerHTML = '<span class="error">Error: ' + result.error + '</span>';
              return;
            }
            if (result.prediction_id) {
              updateProgress(10, "Training in progress...");
              pollStatus();
            }
          })
          .withFailureHandler(function(error) {
            document.getElementById("status").innerHTML = '<span class="error">Error: ' + error + '</span>';
          })
          .startTraining();
      }

      function pollStatus() {
        if (pollCount >= MAX_POLLS) {
          updateProgress(100, "Training completed!");
          setTimeout(() => google.script.host.close(), 2000);
          return;
        }

        pollCount++;
        
        google.script.run
          .withSuccessHandler(function(result) {
            errorCount = 0; // Reset error count on success
            
            if (result.status === "completed") {
              updateProgress(100, "Training completed successfully!");
              setTimeout(() => google.script.host.close(), 2000);
              return;
            }
            
            if (result.status === "failed") {
              document.getElementById("status").innerHTML = '<span class="error">Error: ' + result.message + '</span>';
              return;
            }

            // Handle server updates/worker restarts
            if (result.message && result.message.includes("Server updating")) {
              updateProgress(lastProgress, result.message, true);
              setTimeout(pollStatus, RETRY_DELAY); // Use longer delay for server updates
              return;
            }

            // Update progress and continue polling
            updateProgress(result.progress, result.message);
            setTimeout(pollStatus, POLL_INTERVAL);
          })
          .withFailureHandler(function(error) {
            errorCount++;
            
            // After multiple errors, show warning but keep trying
            if (errorCount > 3) {
              updateProgress(lastProgress, "Server busy, still processing...", true);
            } else {
              updateProgress(lastProgress, "Processing continues...");
            }

            // Use increasing delay for consecutive errors
            setTimeout(pollStatus, Math.min(POLL_INTERVAL * errorCount, RETRY_DELAY));
          })
          .pollOperationStatus();
      }

      // Start the training process immediately
      startTraining();
    </script>
  `
  )
    .setWidth(400)
    .setHeight(150);

  SpreadsheetApp.getUi().showModalDialog(html, "Training Progress");
}

function calculateProgress(startTime) {
  var elapsed = (new Date().getTime() - startTime) / 1000; // seconds
  return Math.min(90, Math.round(elapsed / 30)); // ~3% per second up to 90%
}

function pollOperationStatus() {
  try {
    var userProperties = PropertiesService.getUserProperties();
    var predictionId = userProperties.getProperty("PREDICTION_ID");
    var serviceUrl = userProperties.getProperty("SERVICE_URL") || CLASSIFICATION_SERVICE_URL;
    var operationType = userProperties.getProperty("OPERATION_TYPE");
    var config = JSON.parse(userProperties.getProperty("CONFIG") || "{}");

    if (!predictionId) {
      return { error: "No operation in progress" };
    }

    // Simple polling with fixed interval
    var options = {
      headers: { "X-API-Key": getServiceConfig().apiKey },
      muteHttpExceptions: true,
    };

    try {
      // First try status endpoint
      var response = UrlFetchApp.fetch(serviceUrl + "/status/" + predictionId, options);
      var responseCode = response.getResponseCode();

      // If we get a 502/503/504, the worker might have restarted
      if (responseCode === 502 || responseCode === 503 || responseCode === 504) {
        // Try webhook results before giving up
        try {
          var webhookResponse = UrlFetchApp.fetch(
            serviceUrl + "/webhook-result?prediction_id=" + predictionId,
            options
          );
          var webhookResult = JSON.parse(webhookResponse.getContentText());

          if (webhookResult && webhookResult.status === "success") {
            if (operationType === "categorise") {
              var sheet = SpreadsheetApp.getActiveSheet();
              writeResultsToSheet(webhookResult, config, sheet);
            }
            userProperties.deleteAllProperties();
            return {
              status: "completed",
              message: "Operation completed successfully!",
              progress: 100,
            };
          }
        } catch (webhookError) {
          Logger.log("Webhook check error: " + webhookError);
        }

        // If webhook check fails, return processing status
        return {
          status: "in_progress",
          message: "Server updating, operation continues...",
          progress: 50,
        };
      }

      var result = JSON.parse(response.getContentText());

      // If we get a successful response
      if (result.status === "completed" || result.status === "succeeded") {
        if (result.results && operationType === "categorise") {
          var sheet = SpreadsheetApp.getActiveSheet();
          writeResultsToSheet(result, config, sheet);
        }
        userProperties.deleteAllProperties();
        return {
          status: "completed",
          message: operationType === "categorise" ? "Categorisation completed!" : "Training completed!",
          progress: 100,
        };
      }

      // If the operation failed
      if (result.status === "failed") {
        userProperties.deleteAllProperties();
        return {
          status: "failed",
          message: result.error || "Operation failed",
          progress: 0,
        };
      }

      // If still processing, calculate progress based on time
      var startTime = parseInt(userProperties.getProperty("START_TIME"));
      var elapsedMinutes = (Date.now() - startTime) / (60 * 1000);
      var progress = Math.min(90, Math.round(elapsedMinutes * 10)); // ~10% per minute up to 90%

      return {
        status: "in_progress",
        message: operationType === "categorise" ? "Categorising transactions..." : "Training in progress...",
        progress: progress,
      };
    } catch (error) {
      Logger.log("Status check error: " + error);

      // Try webhook results as fallback
      try {
        var webhookResponse = UrlFetchApp.fetch(serviceUrl + "/webhook-result?prediction_id=" + predictionId, options);
        var webhookResult = JSON.parse(webhookResponse.getContentText());

        if (webhookResult && webhookResult.status === "success") {
          if (operationType === "categorise") {
            var sheet = SpreadsheetApp.getActiveSheet();
            writeResultsToSheet(webhookResult, config, sheet);
          }
          userProperties.deleteAllProperties();
          return {
            status: "completed",
            message: "Operation completed successfully!",
            progress: 100,
          };
        }
      } catch (webhookError) {
        Logger.log("Webhook check error: " + webhookError);
      }

      // Return processing status if all checks fail
      return {
        status: "in_progress",
        message: "Processing continues...",
        progress: 50,
      };
    }
  } catch (error) {
    Logger.log("Polling error: " + error);
    return {
      status: "in_progress",
      message: "Processing continues...",
      progress: 50,
    };
  }
}

// Helper function to calculate progress
function calculateProgress(startTime) {
  var elapsedMinutes = (Date.now() - startTime) / (60 * 1000);
  return Math.min(90, Math.round(elapsedMinutes * 10)); // ~10% per minute up to 90%
}

// Global function to handle training API call
function startTraining() {
  try {
    var userProperties = PropertiesService.getUserProperties();
    var transactions = JSON.parse(userProperties.getProperty("TEMP_TRANSACTIONS"));
    var serviceConfig = getServiceConfig();
    var config = JSON.parse(userProperties.getProperty("TEMP_CONFIG"));
    var originalSheetName = userProperties.getProperty("TEMP_SHEET_NAME");
    var serviceUrl = userProperties.getProperty("TEMP_SERVICE_URL");

    // Make API call
    var payload = JSON.stringify({
      transactions: transactions,
      userId: serviceConfig.userId,
      expenseSheetId: SpreadsheetApp.getActiveSheet().getParent().getId(),
      columnOrderCategorisation: {
        descriptionColumn: config.narrativeCol,
        categoryColumn: config.categoryCol,
      },
      categorisationRange: "A:Z",
      categorisationTab: originalSheetName,
    });

    var options = {
      method: "post",
      contentType: "application/json",
      headers: {
        "X-API-Key": serviceConfig.apiKey,
      },
      payload: payload,
      muteHttpExceptions: true,
    };

    // Make the API call and get prediction ID
    var response = UrlFetchApp.fetch(serviceUrl + "/train", options);
    var responseCode = response.getResponseCode();

    // Handle non-200 responses
    if (responseCode !== 200 && responseCode !== 201) {
      return {
        error: `Server returned error code: ${responseCode}`,
        details: response.getContentText(),
      };
    }

    var result = JSON.parse(response.getContentText());

    if (!result.prediction_id) {
      return { error: "No prediction ID received from server" };
    }

    // Store properties for polling
    userProperties.setProperties({
      PREDICTION_ID: result.prediction_id,
      START_TIME: new Date().getTime().toString(),
      TRAINING_SIZE: transactions.length.toString(),
      ORIGINAL_SHEET_NAME: originalSheetName,
      OPERATION_TYPE: "train",
      SERVICE_URL: serviceUrl,
    });

    // Clean up temporary properties
    userProperties.deleteProperty("TEMP_TRANSACTIONS");
    userProperties.deleteProperty("TEMP_CONFIG");
    userProperties.deleteProperty("TEMP_SHEET_NAME");
    userProperties.deleteProperty("TEMP_SERVICE_URL");

    return { prediction_id: result.prediction_id };
  } catch (error) {
    Logger.log("Training API call error: " + error.toString());
    return { error: error.toString() };
  }
}
