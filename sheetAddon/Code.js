const CLASSIFICATION_SERVICE_URL = "https://txclassify.onrender.com";
// const CLASSIFICATION_SERVICE_URL = "https://txclassify-dev.onrender.com";

// Add menu to the spreadsheet
function onOpen(e) {
  var menu = SpreadsheetApp.getUi().createAddonMenu(); // Changed from addMenu to createAddonMenu
  menu
    .addItem("Configure API Key", "setupApiKey")
    .addItem("Train Model", "showTrainingDialog")
    .addItem("Categorise New Transactions", "showClassifyDialog")
    .addItem("Generate Expense Report", "showAnalyticsDialog")
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
      .optional-field { margin-top: 10px; }
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
    <div class="form-group optional-field">
      <label>Amount Column (optional):</label>
      <select id="amountCol">
        <option value="">-- None --</option>
        <option value="A">A</option>
        <option value="B" selected>B</option>
        <option value="C">C</option>
        <option value="D">D</option>
        <option value="E">E</option>
      </select>
      <div class="help-text">Column containing transaction amounts (positive = money in, negative = money out)</div>
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
        var amountCol = document.getElementById('amountCol').value;
        var errorDiv = document.getElementById('error');
        var submitBtn = document.getElementById('submitBtn');
        var spinner = document.querySelector('.spinner');
        var buttonText = document.querySelector('.button-text');
        var processingText = document.querySelector('.processing-text');
        
        // Validate inputs
        if (!descriptionCol || !categoryCol || !startRow) {
          errorDiv.textContent = 'Please fill in all required fields';
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
          startRow: startRow.toString(),
          amountCol: amountCol // Optional amount column
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
    .setHeight(400);

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
  var amountColDefault = columnToLetter(headers.indexOf("Amount") + 1);

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

  // Create amount column options
  var amountColumnOptions = ['<option value="">-- None --</option>'];
  for (var i = 0; i < lastColumn; i++) {
    var letter = columnToLetter(i + 1);
    var header = headers[i] || ""; // Use empty string if header is null/undefined
    amountColumnOptions.push(
      `<option value="${letter}"${letter === amountColDefault ? " selected" : ""}>` +
        `${letter}${header ? " (" + header + ")" : ""}</option>`
    );
  }
  var amountColumnOptionsStr = amountColumnOptions.join("\n");

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
      .optional-field { margin-top: 10px; }
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
    <div class="form-group optional-field">
      <label>Amount Column (optional):</label>
      <select id="amountCol">
        ${amountColumnOptionsStr}
      </select>
      <div class="help-text">Column with amounts (positive = money in, negative = money out)</div>
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
        var amountCol = document.getElementById('amountCol').value;
        var errorDiv = document.getElementById('error');
        var submitBtn = document.getElementById('submitBtn');
        var spinner = document.querySelector('.spinner');
        var buttonText = document.querySelector('.button-text');
        var processingText = document.querySelector('.processing-text');
        
        // Validate inputs
        if (!narrativeCol || !categoryCol || !startRow) {
          errorDiv.textContent = 'Please fill in all required fields';
          errorDiv.style.display = 'block';
          return;
        }
        
        // Clear any previous errors
        errorDiv.style.display = 'none';
        
        var config = {
          narrativeCol: narrativeCol,
          categoryCol: categoryCol,
          startRow: startRow,
          amountCol: amountCol // Pass optional amount column
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
    .setHeight(400); // Increased height for the new field

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
    var descriptionsData = sheet
      .getRange(config.descriptionCol + startRow + ":" + config.descriptionCol + lastRow)
      .getValues()
      .filter((row) => row[0] && row[0].toString().trim() !== "");

    // Store original descriptions for logging
    var originalDescriptions = descriptionsData.map((row) => {
      var desc = row[0];
      // Ensure it's a string
      if (typeof desc !== "string") {
        desc = String(desc);
      }
      return desc.toString().trim();
    });

    // Validate descriptions
    var invalidDescriptions = originalDescriptions.filter((desc) => !desc || desc.length === 0);
    if (invalidDescriptions.length > 0) {
      Logger.log(
        `Found ${invalidDescriptions.length} invalid descriptions out of ${originalDescriptions.length} total`
      );
      // Filter out invalid descriptions
      originalDescriptions = originalDescriptions.filter((desc) => desc && desc.length > 0);
    }

    // Get amount data if amount column is provided
    var amountData = [];
    var hasAmounts = config.amountCol && config.amountCol.trim() !== "";

    if (hasAmounts) {
      Logger.log("Reading amounts from column " + config.amountCol);
      amountData = sheet
        .getRange(config.amountCol + startRow + ":" + config.amountCol + lastRow)
        .getValues()
        .slice(0, descriptionsData.length); // Match the length of descriptions
    }

    // Prepare transactions with money_in flag if amounts are available
    var transactions = [];
    for (var i = 0; i < originalDescriptions.length; i++) {
      var description = originalDescriptions[i];

      if (hasAmounts && i < amountData.length) {
        var amount = amountData[i][0];
        var parsedAmount = null;

        // Try to parse the amount, handling various formats
        if (typeof amount === "number") {
          parsedAmount = amount;
        } else if (typeof amount === "string") {
          // Remove currency symbols and commas
          var cleanAmount = amount.replace(/[^\d.-]/g, "");
          parsedAmount = parseFloat(cleanAmount);
        }

        if (!isNaN(parsedAmount)) {
          // Create transaction object with money_in flag
          transactions.push({
            description: description,
            money_in: parsedAmount >= 0,
            amount: parsedAmount,
          });
          Logger.log(`Transaction: "${description}" with amount ${parsedAmount}, money_in: ${parsedAmount >= 0}`);
        } else {
          // If amount is invalid, just use the description
          transactions.push({
            description: description,
          });
          Logger.log(`Transaction: "${description}" without valid amount`);
        }
      } else {
        // If no amounts column or missing amount, just use the description
        transactions.push({
          description: description,
        });
      }
    }

    if (transactions.length === 0) {
      throw new Error("No transactions found to categorise");
    }

    Logger.log("Found " + transactions.length + " transactions to categorise");
    updateStatus("Processing " + transactions.length + " transactions...");

    // Prepare the payload with transactions that may include money_in flag
    var payload = JSON.stringify({
      transactions: transactions,
    });

    // Log payload size for debugging
    Logger.log("Payload size: " + payload.length + " bytes");
    if (payload.length > 1000000) {
      // If payload is very large, log a warning
      Logger.log("Warning: Large payload size may cause issues with the API");
    }

    // Make API call with retry logic
    var maxRetries = 3;
    var retryCount = 0;
    var response;
    var error;

    while (retryCount < maxRetries) {
      try {
        if (retryCount > 0) {
          Logger.log(`Retrying categorisation request (attempt ${retryCount + 1}/${maxRetries})...`);
          // Simple delay for retries
          Utilities.sleep(Math.pow(2, retryCount) * 1000);
        }

        response = UrlFetchApp.fetch(serviceConfig.serviceUrl + "/classify", {
          method: "post",
          contentType: "application/json",
          headers: { "X-API-Key": serviceConfig.apiKey },
          payload: payload,
          muteHttpExceptions: true,
        });

        // Accept 200 OK or 202 Accepted as success for starting the job
        if (response.getResponseCode() === 200 || response.getResponseCode() === 202) {
          break; // Success, exit retry loop
        } else if (
          response.getResponseCode() === 502 ||
          response.getResponseCode() === 503 ||
          response.getResponseCode() === 504
        ) {
          // Retry on gateway errors
          error = `Server returned ${response.getResponseCode()}`;
          retryCount++;
          continue;
        } else {
          // Don't retry on other errors
          throw new Error(`Server returned error code: ${response.getResponseCode()}`);
        }
      } catch (e) {
        error = e;
        if (retryCount === maxRetries - 1) {
          // Last attempt failed
          throw new Error(`Categorisation failed after ${maxRetries} attempts: ${e.toString()}`);
        }
        retryCount++;
      }
    }

    var result;
    try {
      var responseText = response.getContentText();
      Logger.log("Raw response: " + responseText);
      result = JSON.parse(responseText);

      if (result.error) {
        throw new Error("API error: " + result.error);
      }

      Logger.log("Categorisation response: " + JSON.stringify(result));
    } catch (parseError) {
      Logger.log("Error parsing response: " + parseError);
      throw new Error("Error processing server response: " + parseError.toString());
    }

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
      SERVICE_URL: serviceConfig.serviceUrl,
    });

    // Show the polling dialog
    showPollingDialog();

    // Return success
    return {
      status: "success",
      message: "Categorisation started. Please wait for results.",
      predictionId: result.prediction_id,
    };
  } catch (error) {
    Logger.log("Categorisation error: " + error.toString());
    updateStatus("Error: " + error.toString());
    SpreadsheetApp.getUi().alert("Error: " + error.toString());
    return { error: error.toString() };
  }
}

// Helper function to write classification results to a sheet
function writeResultsToSheet(result, config, sheet) {
  try {
    Logger.log("Writing results to sheet with config: " + JSON.stringify(config));

    // Extract the results array from the response, handling different response formats
    var resultsData;

    // Log the structure of the result object to help with debugging
    Logger.log("Result object structure: " + JSON.stringify(Object.keys(result)));

    // Try different paths to find the results array based on our updated API format
    if (result.results && Array.isArray(result.results)) {
      Logger.log("Using result.results array");
      resultsData = result.results;
    } else if (result.data && Array.isArray(result.data)) {
      Logger.log("Using result.data array");
      resultsData = result.data;
    } else if (result.result && typeof result.result === "object") {
      // Handle nested result structures
      if (result.result.results && Array.isArray(result.result.results)) {
        Logger.log("Using result.result.results array");
        resultsData = result.result.results;
      } else if (result.result.data && Array.isArray(result.result.data)) {
        Logger.log("Using result.result.data array");
        resultsData = result.result.data;
      }
    } else {
      // Deep search for any array property that looks like results
      for (var key in result) {
        if (
          Array.isArray(result[key]) &&
          result[key].length > 0 &&
          (result[key][0].hasOwnProperty("predicted_category") ||
            result[key][0].hasOwnProperty("Category") ||
            result[key][0].hasOwnProperty("category"))
        ) {
          Logger.log("Found results array in result." + key);
          resultsData = result[key];
          break;
        }
      }
    }

    if (!resultsData || resultsData.length === 0) {
      Logger.log("No results to write. Full result: " + JSON.stringify(result));

      // Alert user about the issue
      SpreadsheetApp.getUi().alert(
        "No classification results found in the API response. Please check logs for details."
      );
      return false;
    }

    Logger.log("Found " + resultsData.length + " results to write");
    Logger.log("Sample result: " + JSON.stringify(resultsData[0]));

    // Get configuration from either the result or the provided config
    var categoryCol = result.config && result.config.categoryColumn ? result.config.categoryColumn : config.categoryCol;
    var startRow =
      result.config && result.config.startRow ? parseInt(result.config.startRow) : parseInt(config.startRow);

    // Validate configuration
    if (!categoryCol || !startRow) {
      Logger.log("Missing required configuration: categoryCol=" + categoryCol + ", startRow=" + startRow);

      // Fallback to config from parameter if available
      categoryCol = categoryCol || config.categoryCol;
      startRow = startRow || parseInt(config.startRow);

      // If still missing, alert and return
      if (!categoryCol || !startRow) {
        SpreadsheetApp.getUi().alert("Missing required configuration for writing results.");
        return false;
      }
    }

    var endRow = startRow + resultsData.length - 1;

    Logger.log("Writing " + resultsData.length + " results to sheet");
    Logger.log("Category column: " + categoryCol + ", Start row: " + startRow + ", End row: " + endRow);

    // Safely extract categories, handling different property names
    var categories = resultsData.map(function (r) {
      return [r.predicted_category || r.category || r.Category || ""];
    });

    // Write categories
    var categoryRange = sheet.getRange(categoryCol + startRow + ":" + categoryCol + endRow);
    categoryRange.setValues(categories);

    // Write confidence scores if they exist
    var hasConfidence = resultsData.some(function (r) {
      return r.hasOwnProperty("similarity_score") || r.hasOwnProperty("confidence") || r.hasOwnProperty("score");
    });

    if (hasConfidence) {
      var confidenceCol = String.fromCharCode(categoryCol.charCodeAt(0) + 1);
      var confidenceScores = resultsData.map(function (r) {
        return [r.similarity_score || r.confidence || r.score || ""];
      });

      var confidenceRange = sheet.getRange(confidenceCol + startRow + ":" + confidenceCol + endRow);
      confidenceRange.setValues(confidenceScores);

      // Format as percentage if they look like decimals between 0-1
      if (confidenceScores.some((score) => typeof score[0] === "number" && score[0] >= 0 && score[0] <= 1)) {
        confidenceRange.setNumberFormat("0.00%");
      }
    }

    // Check if money_in data is available
    var hasMoneyInFlag = resultsData.some(function (r) {
      return r.hasOwnProperty("money_in");
    });

    if (hasMoneyInFlag) {
      // Column after confidence (or after category if no confidence)
      var moneyInCol = hasConfidence
        ? String.fromCharCode(categoryCol.charCodeAt(0) + 2)
        : String.fromCharCode(categoryCol.charCodeAt(0) + 1);

      var moneyInValues = resultsData.map(function (r) {
        if (r.money_in === true) {
          return ["IN"]; // Money in (credit)
        } else if (r.money_in === false) {
          return ["OUT"]; // Money out (debit)
        } else {
          return [""]; // Unknown
        }
      });

      var moneyInRange = sheet.getRange(moneyInCol + startRow + ":" + moneyInCol + endRow);
      moneyInRange.setValues(moneyInValues);

      // Write a header for the column if we're in the first row
      //if (startRow === 1) {
      //  sheet.getRange(moneyInCol + "1").setValue("Type");
      //}
    }

    // Check if amount data is available
    var hasAmount = resultsData.some(function (r) {
      return r.hasOwnProperty("amount") && r.amount !== null && r.amount !== undefined;
    });

    if (hasAmount) {
      // Column after money_in (or after confidence/category if no money_in)
      var amountCol = hasMoneyInFlag
        ? String.fromCharCode(moneyInCol.charCodeAt(0) + 1)
        : hasConfidence
        ? String.fromCharCode(categoryCol.charCodeAt(0) + 2)
        : String.fromCharCode(categoryCol.charCodeAt(0) + 1);

      var amountValues = resultsData.map(function (r) {
        if (r.amount !== null && r.amount !== undefined) {
          return [r.amount];
        } else {
          return [""];
        }
      });

      var amountRange = sheet.getRange(amountCol + startRow + ":" + amountCol + endRow);
      amountRange.setValues(amountValues);

      // Format as currency
      amountRange.setNumberFormat("$#,##0.00;($#,##0.00)");

      // Write a header for the column if we're in the first row
      //if (startRow === 1) {
      //  sheet.getRange(amountCol + "1").setValue("Amount");
      //}
    }

    // Update status with success message
    updateStatus(
      "Categorised " + resultsData.length + " transactions successfully!",
      "Categories written to column " + categoryCol
    );

    // Track usage stat
    updateStats("categorisations", resultsData.length);

    return true;
  } catch (error) {
    Logger.log("Error writing results to sheet: " + error.toString());
    SpreadsheetApp.getUi().alert("Error writing results: " + error.toString());
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
    var originalSheetName = sheet.getName(); // Keep for logging/context if needed
    var lastRow = sheet.getLastRow();
    var startRow = parseInt(config.startRow);

    // Get data from the selected columns
    var narrativeRange = sheet.getRange(config.narrativeCol + startRow + ":" + config.narrativeCol + lastRow);
    var categoryRange = sheet.getRange(config.categoryCol + startRow + ":" + config.categoryCol + lastRow);
    var narratives = narrativeRange.getValues();
    var categories = categoryRange.getValues();

    // Get amount data if amount column is provided
    var amountData = [];
    var hasAmounts = config.amountCol && config.amountCol.trim() !== "";
    if (hasAmounts) {
      Logger.log("Reading training amounts from column " + config.amountCol);
      amountData = sheet
        .getRange(config.amountCol + startRow + ":" + config.amountCol + lastRow)
        .getValues()
        .slice(0, narratives.length); // Match the length of narratives
    }

    // Prepare training data with amounts and money_in flag
    var transactions = [];
    for (var i = 0; i < narratives.length; i++) {
      try {
        // Skip empty rows or rows with invalid data
        if (!narratives[i][0] || !categories[i][0]) {
          continue;
        }

        // Ensure both values are strings
        var narrative = String(narratives[i][0]).trim();
        var category = String(categories[i][0]).trim();

        // Skip rows with empty data after trimming
        if (narrative === "" || category === "") {
          continue;
        }

        // Check for descriptions that might be cut off or have formatting issues
        if (narrative.length > 200) {
          // Truncate long descriptions to prevent issues
          Logger.log(`Truncating long description (${narrative.length} chars): ${narrative.substring(0, 30)}...`);
          narrative = narrative.substring(0, 200);
        }

        var transaction = {
          description: narrative,
          Category: category,
        };

        // Add amount and money_in flag if available
        if (hasAmounts && i < amountData.length) {
          var amount = amountData[i][0];
          var parsedAmount = null;

          if (typeof amount === "number") {
            parsedAmount = amount;
          } else if (typeof amount === "string") {
            var cleanAmount = amount.replace(/[^\d.-]/g, "");
            parsedAmount = parseFloat(cleanAmount);
          }

          if (!isNaN(parsedAmount)) {
            transaction.amount = parsedAmount;
            transaction.money_in = parsedAmount >= 0;
            Logger.log(`Training tx: "${narrative}" | Amt: ${parsedAmount} | MoneyIn: ${transaction.money_in}`);
          }
        }

        // Add valid transaction to the array
        transactions.push(transaction);
      } catch (rowError) {
        // Log error but continue processing
        Logger.log(`Error processing row ${i + 1}: ${rowError}`);
        continue;
      }
    }

    if (transactions.length === 0) {
      throw new Error("No valid training data found");
    }

    if (transactions.length < 10) {
      throw new Error(
        `Not enough valid transactions. Found only ${transactions.length} valid transactions after validation. At least 10 are required.`
      );
    }

    // Call startTraining directly, passing the prepared data
    // No need to store large data in properties
    var result = startTraining(transactions, serviceConfig, config);

    // Handle potential immediate errors from startTraining (e.g., API key issue)
    if (result && result.error) {
      throw new Error(result.error);
    }

    // Success: startTraining will handle showing the polling dialog
    Logger.log("Training request initiated successfully.");
  } catch (error) {
    Logger.log("Training setup error: " + error.toString());
    updateStatus("Error during training setup: " + error.toString());
    // Show error in the dialog if it's still open, otherwise use alert
    try {
      // This might fail if the dialog is already closed
      throw error; // Re-throw to be caught by the dialog's failure handler
    } catch (e) {
      ui.alert("Error during training setup: " + error.toString());
    }
  }
}

function calculateProgress(startTime) {
  var elapsedMinutes = (Date.now() - startTime) / (60 * 1000);
  return Math.min(90, Math.round(elapsedMinutes * 10)); // ~10% per minute up to 90%
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

    // Call the status endpoint
    var options = {
      headers: { "X-API-Key": getServiceConfig().apiKey },
      muteHttpExceptions: true,
    };

    var response = UrlFetchApp.fetch(serviceUrl + "/status/" + predictionId, options);
    var responseCode = response.getResponseCode();

    // Handle non-200 responses
    if (responseCode !== 200) {
      Logger.log(`Server returned ${responseCode} for prediction ${predictionId}`);
      return {
        status: "in_progress",
        message: "Server processing, please wait...",
        progress: 50,
      };
    }

    // Parse the response
    var result = JSON.parse(response.getContentText());
    Logger.log("Status response: " + JSON.stringify(result));

    // Handle different status cases
    switch (result.status) {
      case "completed":
      case "succeeded":
        Logger.log("Operation completed successfully");

        // Handle categorization results if needed
        if (operationType === "categorise" && result.results) {
          try {
            var sheet = SpreadsheetApp.getActiveSheet();
            writeResultsToSheet(result, config, sheet);
            Logger.log("Results written to sheet successfully");
          } catch (writeError) {
            Logger.log("Error writing results to sheet: " + writeError);
          }
        }

        // Clear operation state
        userProperties.deleteAllProperties();

        return {
          status: "completed",
          message: operationType === "categorise" ? "Categorisation completed!" : "Training completed!",
          progress: 100,
        };

      case "failed":
      case "error":
        Logger.log("Operation failed: " + (result.error || result.message));
        userProperties.deleteAllProperties();
        return {
          status: "failed",
          message: result.error || result.message || "Operation failed",
          progress: 0,
        };

      case "processing":
      default:
        var startTime = parseInt(userProperties.getProperty("START_TIME"));
        var progress = calculateProgress(startTime);
        var statusMessage = result.message || (operationType === "categorise" ? "Categorising..." : "Training...");

        Logger.log(`Operation in progress (${progress}%): ${statusMessage}`);
        return {
          status: "in_progress",
          message: statusMessage,
          progress: progress,
        };
    }
  } catch (error) {
    Logger.log("Error in pollOperationStatus: " + error);
    return {
      status: "in_progress",
      message: "Processing continues...",
      progress: 50,
    };
  }
}

// Global function to handle training API call
function startTraining(transactions, serviceConfig, config) {
  try {
    // Validate required data
    if (!transactions || !transactions.length) {
      return { error: "No training data available" };
    }

    if (!serviceConfig || !serviceConfig.apiKey) {
      return { error: "API key not configured" };
    }

    // Prepare payload (already contains amount and money_in from trainModel)
    var payload = JSON.stringify({
      transactions: transactions, // These now potentially include amount/money_in
      userId: serviceConfig.userId,
    });

    Logger.log("Sending training request with " + transactions.length + " transactions");
    Logger.log("Sample transaction in payload: " + JSON.stringify(transactions[0]));

    var options = {
      method: "post",
      contentType: "application/json",
      headers: {
        "X-API-Key": serviceConfig.apiKey,
      },
      payload: payload,
      muteHttpExceptions: true,
    };

    // Retry logic for API call
    var maxRetries = 3;
    var retryCount = 0;
    var response;
    var error;

    while (retryCount < maxRetries) {
      try {
        if (retryCount > 0) {
          Logger.log(`Retrying training request (attempt ${retryCount + 1}/${maxRetries})...`);
          // Simple delay for retries
          Utilities.sleep(Math.pow(2, retryCount) * 1000);
        }

        // Make the API call
        response = UrlFetchApp.fetch(serviceConfig.serviceUrl + "/train", options);
        var responseCode = response.getResponseCode();

        // Handle different response codes
        if (responseCode === 200 || responseCode === 201) {
          break; // Success, exit retry loop
        } else if (responseCode === 502 || responseCode === 503 || responseCode === 504) {
          // Retry on gateway errors
          error = `Server returned ${responseCode}`;
          retryCount++;
          continue;
        } else {
          // Don't retry on other errors
          var errorText = response.getContentText(); // Get error message from server
          Logger.log(`Training API call failed with status ${responseCode}: ${errorText}`);
          throw new Error(`Server returned error code: ${responseCode} - ${errorText}`);
        }
      } catch (e) {
        error = e;
        if (retryCount === maxRetries - 1) {
          // Last attempt failed
          return { error: `Training failed after ${maxRetries} attempts: ${e.toString()}` };
        }
        retryCount++;
      }
    }

    // Parse response
    try {
      var responseText = response.getContentText();
      Logger.log("Raw training response: " + responseText);

      var result;
      try {
        result = JSON.parse(responseText);
      } catch (jsonError) {
        Logger.log("Error parsing JSON response: " + jsonError);
        return { error: "Invalid JSON response from server" };
      }

      // Check for error response
      if (result.error) {
        Logger.log("API returned error: " + result.error);

        // Special handling for validation errors
        if (result.details && Array.isArray(result.details)) {
          var validationErrors = result.details
            .map((err) => {
              return `Field ${err.location}: ${err.message}`; // Adjusted error format
            })
            .join(", ");

          Logger.log("Validation errors: " + validationErrors);
          return { error: "Validation error: " + validationErrors };
        }

        return { error: "API error: " + result.error };
      }

      Logger.log("Training response: " + JSON.stringify(result));

      if (!result.prediction_id) {
        return { error: "No prediction ID received from server" };
      }

      // Set properties ONLY for polling
      var userProperties = PropertiesService.getUserProperties();
      userProperties.setProperties({
        PREDICTION_ID: result.prediction_id,
        OPERATION_TYPE: "training",
        START_TIME: Date.now().toString(),
        SERVICE_URL: serviceConfig.serviceUrl, // Still need service URL for polling
        // Store minimal config needed for polling/status display if any
        // (Currently, none seem essential for polling itself)
      });

      // Track the training operation in stats
      updateStats("training_operations", 1);
      updateStats("trained_transactions", transactions.length);

      // Show the polling dialog AFTER successful API call
      showPollingDialog();

      return {
        status: "processing",
        predictionId: result.prediction_id,
        message: "Training started. Check back in a few minutes.",
      };
    } catch (parseError) {
      Logger.log("Error parsing training response: " + parseError);
      return { error: "Error processing server response: " + parseError.toString() };
    }
  } catch (error) {
    Logger.log("Error in startTraining: " + error);
    return { error: error.toString() };
  }
}

// Helper function to show a polling dialog for both training and categorization
function showPollingDialog() {
  var userProperties = PropertiesService.getUserProperties();
  var operationType = userProperties.getProperty("OPERATION_TYPE");
  var title = operationType === "categorise" ? "Categorisation Progress" : "Training Progress";
  var initialMessage = operationType === "categorise" ? "Processing transactions..." : "Training model...";

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
      .success { color: green; font-weight: bold; }
      .button-container { 
        margin-top: 15px; 
        text-align: right; 
        display: none; 
      }
      .close-button {
        padding: 8px 15px;
        background: #4285f4;
        color: white;
        border: none;
        border-radius: 3px;
        cursor: pointer;
      }
    </style>
    <div class="status" id="status">${initialMessage}</div>
    <div class="progress-bar">
      <div class="progress-fill" id="progress"></div>
    </div>
    <div class="button-container" id="buttonContainer">
      <button class="close-button" onclick="closeDialog()">Close</button>
    </div>
    <script>
      const POLL_INTERVAL = 5000;  // Fixed 5-second interval
      const MAX_POLLS = 120;       // 10 minutes maximum
      let pollCount = 0;
      let errorCount = 0;
      let lastProgress = 0;
      let isCompleted = false;

      function closeDialog() {
        google.script.host.close();
      }

      function updateProgress(percent, message, isWarning, isSuccess) {
        document.getElementById("progress").style.width = percent + "%";
        if (message) {
          if (isSuccess) {
            document.getElementById("status").innerHTML = '<span class="success">' + message + '</span>';
          } else if (isWarning) {
            document.getElementById("status").innerHTML = '<span class="warning">' + message + '</span>';
          } else {
            document.getElementById("status").innerHTML = message;
          }
        }
        lastProgress = percent;
        
        // Show close button if operation is completed or failed
        if (isCompleted || percent === 100 || isSuccess) {
          document.getElementById("buttonContainer").style.display = "block";
        }
      }

      function pollStatus() {
        if (pollCount >= MAX_POLLS) {
          updateProgress(100, "Operation timed out but may still complete in the background.", true);
          isCompleted = true;
          return;
        }

        pollCount++;
        
        google.script.run
          .withSuccessHandler(function(result) {
            // Handle invalid or null result
            if (!result) {
              errorCount++;
              updateProgress(lastProgress, "Received invalid response from server, retrying...", true);
              setTimeout(pollStatus, POLL_INTERVAL);
              return;
            }
            
            // Reset error count on success
            errorCount = 0;
            
            if (result.status === "completed") {
              updateProgress(100, result.message || "Operation completed successfully!", false, true);
              isCompleted = true;
              return;
            }
            
            if (result.status === "failed") {
              document.getElementById("status").innerHTML = '<span class="error">Error: ' + (result.message || "Unknown error") + '</span>';
              document.getElementById("buttonContainer").style.display = "block";
              isCompleted = true;
              return;
            }

            // Handle server updates/worker restarts
            if (result.message && result.message.includes("Server")) {
              updateProgress(lastProgress, result.message, true);
            } else {
              updateProgress(result.progress, result.message || "Processing...");
            }
            
            if (!isCompleted) {
              setTimeout(pollStatus, POLL_INTERVAL);
            }
          })
          .withFailureHandler(function(error) {
            errorCount++;
            
            // After multiple errors, show warning but keep trying
            if (errorCount > 3) {
              updateProgress(lastProgress, "Server busy, still processing... (" + errorCount + " errors)", true);
            } else {
              updateProgress(lastProgress, "Processing continues... (Error: " + error + ")", true);
            }

            // Use fixed delay for retries
            if (!isCompleted) {
              setTimeout(pollStatus, POLL_INTERVAL);
            }
          })
          .pollOperationStatus();
      }

      // Start polling immediately
      pollStatus();
    </script>
    `
  )
    .setWidth(400)
    .setHeight(180);

  SpreadsheetApp.getUi().showModalDialog(html, title);
}

// Show dialog to select columns for analytics report
function showAnalyticsDialog() {
  // TODO: Get actual columns and headers dynamically like in showTrainingDialog
  var defaultColumns = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];
  var columnOptions = defaultColumns.map((col) => `<option value="${col}">${col}</option>`).join("\n");
  var amountColOptions = '<option value="">-- None --</option>\n' + columnOptions; // Allow optional amount

  var html = HtmlService.createHtmlOutput(
    `
    <style>
      body { font-family: Arial, sans-serif; padding: 20px; }
      .form-group { margin-bottom: 15px; }
      label { display: block; margin-bottom: 5px; font-weight: bold; }
      select, input { width: 100%; padding: 5px; box-sizing: border-box; }
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
      button:disabled { background: #ccc; cursor: not-allowed; }
      .help-text { font-size: 12px; color: #666; margin-top: 4px; }
      .error { color: red; margin-top: 10px; display: none; }
      .spinner { display: none; width: 20px; height: 20px; border: 2px solid #f3f3f3; border-top: 2px solid #3498db; border-radius: 50%; animation: spin 1s linear infinite; position: absolute; right: 10px; top: 50%; transform: translateY(-50%); }
      @keyframes spin { 0% { transform: translateY(-50%) rotate(0deg); } 100% { transform: translateY(-50%) rotate(360deg); } }
      .button-text { display: inline-block; }
      .processing-text { display: none; }
    </style>
    
    <div class="form-group">
      <label>Description Column:</label>
      <select id="descriptionCol">${columnOptions.replace('value="C"', 'value="C" selected')}</select>
      <div class="help-text">Column with transaction descriptions</div>
    </div>
    <div class="form-group">
      <label>Date Column:</label>
      <select id="dateCol">${columnOptions.replace('value="A"', 'value="A" selected')}</select>
      <div class="help-text">Column with transaction dates</div>
    </div>
    <div class="form-group">
      <label>Category Column:</label>
      <select id="categoryCol">${columnOptions.replace('value="E"', 'value="E" selected')}</select>
      <div class="help-text">Column with transaction categories</div>
    </div>
    <div class="form-group">
      <label>Amount Column:</label>
      <select id="amountCol">${amountColOptions.replace('value="B"', 'value="B" selected')}</select>
      <div class="help-text">Column with transaction amounts</div>
    </div>
    <div class="form-group">
      <label>Currency Column (optional):</label>
      <select id="currencyCol">${amountColOptions.replace('value="F"', 'value="F" selected')}</select>
      <div class="help-text">Needed for multi-currency aggregation</div>
    </div>
     <div class="form-group">
      <label>Account Column (optional):</label>
      <select id="accountCol">${amountColOptions.replace('value="G"', 'value="G" selected')}</select>
      <div class="help-text">For filtering/grouping by account</div>
    </div>
    <div class="form-group">
      <label>Start Row:</label>
      <input type="number" id="startRow" value="2" min="1">
      <div class="help-text">First row of data (usually 2 to skip headers)</div>
    </div>

    <div id="error" class="error"></div>
    <button onclick="submitAnalyticsForm()" id="submitBtn">
      <span class="button-text">Generate Report</span>
      <span class="processing-text">Generating...</span>
      <div class="spinner"></div>
    </button>
    
    <script>
      function submitAnalyticsForm() {
        var config = {
          descriptionCol: document.getElementById('descriptionCol').value,
          dateCol: document.getElementById('dateCol').value,
          categoryCol: document.getElementById('categoryCol').value,
          amountCol: document.getElementById('amountCol').value,
          currencyCol: document.getElementById('currencyCol').value,
          accountCol: document.getElementById('accountCol').value,
          startRow: document.getElementById('startRow').value
        };
        var errorDiv = document.getElementById('error');
        var submitBtn = document.getElementById('submitBtn');
        var spinner = document.querySelector('.spinner');
        var buttonText = document.querySelector('.button-text');
        var processingText = document.querySelector('.processing-text');

        // Basic validation
        if (!config.descriptionCol || !config.dateCol || !config.categoryCol || !config.amountCol || !config.startRow) {
          errorDiv.textContent = 'Please select Description, Date, Category, Amount columns and Start Row.';
          errorDiv.style.display = 'block';
          return;
        }
        
        if (parseInt(config.startRow) < 1) {
           errorDiv.textContent = 'Start row must be 1 or higher';
           errorDiv.style.display = 'block';
           return;
        }

        errorDiv.style.display = 'none';
        submitBtn.disabled = true;
        spinner.style.display = 'block';
        buttonText.style.display = 'none';
        processingText.style.display = 'inline-block';

        google.script.run
          .withSuccessHandler(function(result) { 
            // On success, the server function should return the URL
            // For now, just close dialog. Later: show the URL
            google.script.host.close(); 
            // TODO: Display the result.report_url in a new dialog/popup
            if (result && result.report_url) {
               // Create a small popup to show the link
                var htmlPopup = HtmlService.createHtmlOutput('<p style="font-family: Arial, sans-serif;">Your report is ready:</p><p><a href="' + result.report_url + '" target="_blank" rel="noopener noreferrer">' + result.report_url + '</a></p>')
                    .setWidth(400)
                    .setHeight(100);
                SpreadsheetApp.getUi().showModalDialog(htmlPopup, 'Report Link');
            } else if (result && result.message) {
                 SpreadsheetApp.getUi().alert(result.message); // Show info message
            }
          })
          .withFailureHandler(function(error) {
            errorDiv.textContent = error.message || 'An error occurred generating the report.';
            errorDiv.style.display = 'block';
            submitBtn.disabled = false;
            spinner.style.display = 'none';
            buttonText.style.display = 'inline-block';
            processingText.style.display = 'none';
          })
          .generateAnalyticsReport(config);
      }
    </script>
  `
  )
    .setWidth(450)
    .setHeight(550); // Increased height

  SpreadsheetApp.getUi().showModalDialog(html, "Generate Expense Report");
}

// Server-side function to handle analytics report generation
function generateAnalyticsReport(config) {
  Logger.log("Starting analytics report generation with config: " + JSON.stringify(config));
  updateStatus("Starting analytics report generation...");

  try {
    var serviceConfig = getServiceConfig(); // Gets API key, URL, UserID
    var sheet = SpreadsheetApp.getActiveSheet();
    var lastRow = sheet.getLastRow();
    var startRow = parseInt(config.startRow);

    if (startRow > lastRow) {
      throw new Error("Start row is beyond the last row of data.");
    }

    // Define the columns to read based on config
    // Example: Read Description, Date, Category, Amount
    // Need to dynamically determine the range to cover all selected columns.
    // This simple version assumes columns are contiguous for getRange, which might not be true.
    // A more robust way reads each column individually.

    // Simple approach: Determine min/max columns
    var columnsToRead = [
      config.descriptionCol,
      config.dateCol,
      config.categoryCol,
      config.amountCol,
      config.currencyCol,
      config.accountCol,
    ].filter(Boolean);
    if (columnsToRead.length === 0) throw new Error("No columns selected for reading.");

    // Read data for each selected column (more robust)
    var data = {};
    var numRows = lastRow - startRow + 1;

    function readColumnData(colLetter) {
      if (!colLetter) return null; // Skip if column not provided (optional)
      try {
        return sheet.getRange(colLetter + startRow + ":" + colLetter + lastRow).getValues();
      } catch (e) {
        Logger.log(`Error reading column ${colLetter}: ${e}`);
        throw new Error(`Could not read data from column ${colLetter}. Is it a valid column letter?`);
      }
    }

    data.descriptions = readColumnData(config.descriptionCol);
    data.dates = readColumnData(config.dateCol);
    data.categories = readColumnData(config.categoryCol);
    data.amounts = readColumnData(config.amountCol);
    data.currencies = readColumnData(config.currencyCol);
    data.accounts = readColumnData(config.accountCol);

    // --- Data Structuring & Cleaning ---
    var transactions = [];
    for (var i = 0; i < numRows; i++) {
      // Basic check if essential data exists for the row
      if (!data.descriptions[i][0] || !data.dates[i][0] || !data.categories[i][0] || !data.amounts[i][0]) {
        // Optionally log skipped rows or handle them differently
        // Logger.log(`Skipping row ${startRow + i} due to missing essential data.`);
        continue;
      }

      var dateValue = data.dates[i][0];
      var isoDate = "";
      try {
        // Attempt to convert date to ISO string
        if (dateValue instanceof Date) {
          isoDate = dateValue.toISOString();
        } else {
          // Handle potential string dates - this might need refinement based on actual formats
          isoDate = new Date(dateValue).toISOString();
        }
        // Basic validation if the date conversion resulted in a valid date
        if (isoDate === "Invalid Date") throw new Error("Invalid date format");
      } catch (dateError) {
        Logger.log(`Skipping row ${startRow + i} due to invalid date: ${dateValue} (${dateError})`);
        continue; // Skip row if date is invalid
      }

      var amountValue = data.amounts[i][0];
      var parsedAmount = NaN;
      if (typeof amountValue === "number") {
        parsedAmount = amountValue;
      } else if (typeof amountValue === "string") {
        // Remove currency symbols, commas, etc.
        var cleanAmount = amountValue.replace(/[^\d.-]/g, "");
        parsedAmount = parseFloat(cleanAmount);
      }

      if (isNaN(parsedAmount)) {
        Logger.log(`Skipping row ${startRow + i} due to invalid amount: ${amountValue}`);
        continue; // Skip row if amount is invalid
      }

      var transaction = {
        description: String(data.descriptions[i][0]),
        date: isoDate,
        category: String(data.categories[i][0]),
        amount: parsedAmount,
        currency: data.currencies ? String(data.currencies[i][0]) : null, // Optional
        account: data.accounts ? String(data.accounts[i][0]) : null, // Optional
      };
      transactions.push(transaction);
    }

    if (transactions.length === 0) {
      throw new Error("No valid transaction data found in the selected range/columns.");
    }

    Logger.log("Prepared " + transactions.length + " transactions for analysis.");
    // Logger.log("Sample transaction: " + JSON.stringify(transactions[0])); // Log sample if needed

    // --- Prepare Payload for Python Backend ---
    var payload = JSON.stringify({
      transactions: transactions,
      userId: serviceConfig.userId, // Include UserID for potential backend logic
      // You could add other parameters here, like requested report type
    });

    // Log payload size if large
    if (payload.length > 500000) {
      // ~0.5MB
      Logger.log("Analytics payload size: " + (payload.length / 1024).toFixed(1) + " KB");
    }
    if (payload.length > 5000000) {
      // ~5MB - getting risky for UrlFetch limits
      Logger.log(
        "Warning: Large payload size (" + (payload.length / 1024 / 1024).toFixed(1) + " MB) may exceed limits."
      );
      // Consider implementing chunking or alternative approach if this happens often
    }

    // --- Make API Call (TODO: Implement actual call) ---
    var analyticsEndpoint = serviceConfig.serviceUrl + "/analyze"; // Define your new backend endpoint

    Logger.log("Calling analytics endpoint: " + analyticsEndpoint);

    // ** Placeholder - Replace with actual UrlFetchApp call **
    // Commenting out the real call until the backend endpoint exists
    /*
    var options = {
      method: "post",
      contentType: "application/json",
      headers: { "X-API-Key": serviceConfig.apiKey },
      payload: payload,
      muteHttpExceptions: true, // Important to handle errors manually
      // Increase deadline if analysis might take longer (max 6 mins)
      // deadline: 300 // 5 minutes (optional) 
    };

    var response = UrlFetchApp.fetch(analyticsEndpoint, options);
    var responseCode = response.getResponseCode();
    var responseText = response.getContentText();

    if (responseCode >= 200 && responseCode < 300) {
      Logger.log("Analytics API call successful (Code: " + responseCode + ")");
      var result = JSON.parse(responseText);
      if (!result.report_url) {
         throw new Error("Backend response missing 'report_url'. Response: " + responseText);
      }
      updateStatus("Analytics report generated successfully!");
      // Return the result which contains the report_url
      return { report_url: result.report_url }; 
    } else {
      Logger.log("Analytics API call failed. Code: " + responseCode + ", Response: " + responseText);
      // Try to parse error message from backend if available
      var errorMessage = "Failed to generate report. Server responded with code " + responseCode;
      try {
        var errorResult = JSON.parse(responseText);
        if (errorResult && errorResult.error) {
           errorMessage += ": " + errorResult.error;
        } else if (errorResult && errorResult.message) {
            errorMessage += ": " + errorResult.message;
        } else {
            errorMessage += ". " + responseText.substring(0, 200); // Include start of response
        }
      } catch (e) { /* Ignore parsing error * / }
      throw new Error(errorMessage);
    }
    */

    // ** Temporary success message for testing UI **
    updateStatus("Data prepared, backend call skipped (placeholder).");
    SpreadsheetApp.getUi().alert(
      "Data processed (" +
        transactions.length +
        " transactions). Backend integration needed to generate the actual report URL."
    );
    return { message: "Data prepared, backend call skipped." }; // Temporary response
  } catch (error) {
    Logger.log("Error generating analytics report: " + error.toString() + "\n" + error.stack);
    updateStatus("Error generating report: " + error.toString());
    // Re-throw the error so the client-side .withFailureHandler catches it
    throw new Error("Analytics Generation Failed: " + error.message);
  }
}
