// const CLASSIFICATION_SERVICE_URL = "https://txclassify.onrender.com";
// const CLASSIFICATION_SERVICE_URL = "https://api.expensesorted.com"; // Will be removed

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
    <div class="form-group optional-field">
      <label>Amount Column (optional input):</label>
      <select id="amountCol">
        <option value="">-- None --</option>
        <option value="A">A</option>
        <option value="B" selected>B</option>
        <option value="C">C</option>
        <option value="D">D</option>
        <option value="E">E</option>
      </select>
      <div class="help-text">Column containing transaction amounts (used for prediction if provided)</div>
    </div>
    <div class="form-group optional-field">
      <label>Confidence Score Column (optional output):</label>
      <select id="confidenceCol">
        <option value="">-- None --</option>
        <option value="E">E</option>
        <option value="F" selected>F</option>
        <option value="G">G</option>
        <option value="H">H</option>
        <option value="I">I</option>
      </select>
      <div class="help-text">Column where confidence scores will be written (optional)</div>
    </div>
    <div class="form-group optional-field">
      <label>Money In/Out Column (optional output):</label>
      <select id="moneyInOutCol">
        <option value="">-- None --</option>
        <option value="F">F</option>
        <option value="G" selected>G</option>
        <option value="H">H</option>
        <option value="I">I</option>
        <option value="J">J</option>
      </select>
      <div class="help-text">Column where 'IN'/'OUT' flag will be written (optional)</div>
    </div>
    <div id="error" class="error"></div>
    <button onclick="submitForm()" id="submitBtn">
      <span class="button-text">Categorise Selected Rows</span>
      <span class="processing-text">Processing...</span>
      <div class="spinner"></div>
    </button>
    <button onclick="closeDialog()" id="closeBtn" style="display: none; background: #6c757d; margin-left: 10px;">Close</button>
    <script>
      // Declare UI variables globally within the script scope
      var errorDiv = document.getElementById('error');
      var submitBtn = document.getElementById('submitBtn');
      var spinner = document.querySelector('.spinner');
      var buttonText = document.querySelector('.button-text');
      var processingText = document.querySelector('.processing-text');
      var closeBtn = document.getElementById('closeBtn');

      function submitForm() {
        var descriptionCol = document.getElementById('descriptionCol').value;
        var categoryCol = document.getElementById('categoryCol').value;
        var amountCol = document.getElementById('amountCol').value;
        var confidenceCol = document.getElementById('confidenceCol').value;
        var moneyInOutCol = document.getElementById('moneyInOutCol').value;
        
        // Validate inputs
        if (!descriptionCol || !categoryCol) {
          errorDiv.textContent = 'Description and Category columns are required.';
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
          amountCol: amountCol,
          confidenceCol: confidenceCol,
          moneyInOutCol: moneyInOutCol
        };
        
        google.script.run
          .withSuccessHandler(function(result) {
            // Log exactly what the success handler received
            console.log("SuccessHandler received:", JSON.stringify(result));
            // Server-side will always show the polling dialog on success (sync or async).
            // So, the only job of this handler is to close the initial dialog.
            console.log("SuccessHandler: Closing initial dialog as server handled polling/completion.");
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
      
      // Add a function for the close button
      function closeDialog() {
        google.script.host.close();
      }
    </script>
  `
  )
    .setWidth(400)
    .setHeight(500);

  SpreadsheetApp.getUi().showModalDialog(html, "Categorise Selected Transactions");
}

// Show dialog to select columns for training
function showTrainingDialog() {
  // Get current sheet to determine default columns
  var sheet = SpreadsheetApp.getActiveSheet();
  var lastColumn = sheet.getLastColumn();
  var headers = sheet.getRange(1, 1, 1, lastColumn).getValues()[0];

  // Find default column indices
  var narrativeColDefault = SheetUtils.columnToLetter(headers.indexOf("Narrative") + 1);
  var categoryColDefault = SheetUtils.columnToLetter(headers.indexOf("Category") + 1);
  var amountColDefault = SheetUtils.columnToLetter(headers.indexOf("Amount") + 1);

  // Create column options - always include all columns up to last used column
  var columnOptions = [];
  for (var i = 0; i < lastColumn; i++) {
    var letter = SheetUtils.columnToLetter(i + 1);
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
    var letter = SheetUtils.columnToLetter(i + 1);
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
    var letter = SheetUtils.columnToLetter(i + 1);
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
    <button onclick="closeDialog()" id="closeBtn" style="display: none; background: #6c757d; margin-left: 10px;">Close</button>
    <script>
      // Declare UI variables globally within the script scope
      var errorDiv = document.getElementById('error');
      var submitBtn = document.getElementById('submitBtn');
      var spinner = document.querySelector('.spinner');
      var buttonText = document.querySelector('.button-text');
      var processingText = document.querySelector('.processing-text');
      var closeBtn = document.getElementById('closeBtn');

      function submitForm() {
        var narrativeCol = document.getElementById('narrativeCol').value;
        var categoryCol = document.getElementById('categoryCol').value;
        var startRow = document.getElementById('startRow').value;
        var amountCol = document.getElementById('amountCol').value;
        
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
          .withSuccessHandler(function(result) {
            // Log exactly what the success handler received
            console.log("SuccessHandler received:", JSON.stringify(result));
            // Server-side will always show the polling dialog on success (sync or async).
            // So, the only job of this handler is to close the initial dialog.
            console.log("SuccessHandler: Closing initial dialog as server handled polling/completion.");
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

      // Add a function for the close button
      function closeDialog() {
        google.script.host.close();
      }
    </script>
  `
  )
    .setWidth(400)
    .setHeight(500);

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
        min-width: 120px; /* Ensure button has a minimum width for consistent text change */
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
    
    <button onclick="saveApiKey()" id="saveApiKeyBtn">Save API Key</button>
    
    <a href="https://expensesorted.com/api-key" target="_blank" class="web-app-link">
      Get or Generate API Key from ExpenseSorted Web App
    </a>
    
    <script>
      function saveApiKey() {
        var apiKey = document.getElementById('apiKey').value.trim();
        var errorDiv = document.getElementById('error');
        var successDiv = document.getElementById('success');
        var saveBtn = document.getElementById('saveApiKeyBtn'); // Get the button
        
        // Hide any previous messages
        errorDiv.style.display = 'none';
        successDiv.style.display = 'none';
        
        if (!apiKey) {
          errorDiv.textContent = 'API key is required';
          errorDiv.style.display = 'block';
          return;
        }

        // Disable button and show loading state
        saveBtn.disabled = true;
        saveBtn.textContent = 'Saving...';
        
        google.script.run
          .withSuccessHandler(function() {
            successDiv.textContent = 'API key saved successfully!';
            successDiv.style.display = 'block';
            // Re-enable button and restore text
            saveBtn.disabled = false;
            saveBtn.textContent = 'Save API Key';
            setTimeout(function() {
              google.script.host.close();
            }, 1500);
          })
          .withFailureHandler(function(error) {
            errorDiv.textContent = error.message || 'An error occurred';
            errorDiv.style.display = 'block';
            // Re-enable button and restore text
            saveBtn.disabled = false;
            saveBtn.textContent = 'Save API Key';
          })
          .triggerSaveApiKey(apiKey); // Updated to call a global wrapper in sheets_script.js
      }
    </script>
  `
  )
    .setWidth(500)
    .setHeight(450);

  SpreadsheetApp.getUi().showModalDialog(html, "API Key Management");
}

// Global wrapper for saving API Key, callable from client-side HTML
function triggerSaveApiKey(apiKey) {
  try {
    Config.saveApiKey(apiKey); // Calls the method in config.js
    updateStatus("API key configured successfully!"); // Keep local updateStatus
    return { success: true }; // Return success to client
  } catch (e) {
    Logger.log("Error in triggerSaveApiKey: " + e.toString());
    return { error: e.message || e.toString() };
  }
}

// Helper function to get stored properties - REMOVED (was getServiceConfig, now use Config.getServiceConfig())

// Helper function to update status in sheet
function updateStatus(message, additionalDetails = "") {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var logSheet = ss.getSheetByName("Log");

  // Create Log sheet if it doesn't exist
  if (!logSheet) {
    logSheet = ss.insertSheet("Log");
    logSheet.getRange("A1:D1").setValues([["Timestamp", "Status", "Message", "Details"]]);
    logSheet.setFrozenRows(1);
    logSheet.setColumnWidth(1, 180);
    logSheet.setColumnWidth(2, 100);
    logSheet.setColumnWidth(3, 300);
    logSheet.setColumnWidth(4, 400);
    var headerRange = logSheet.getRange("A1:D1");
    headerRange.setBackground("#f3f3f3").setFontWeight("bold").setHorizontalAlignment("center");
  }

  var timestamp = new Date().toLocaleString();
  var status = "INFO";
  if (message.toLowerCase().includes("error")) {
    status = "ERROR";
  } else if (message.toLowerCase().includes("completed") || message.toLowerCase().includes("success")) {
    status = "SUCCESS";
  } else if (message.toLowerCase().includes("progress") || message.toLowerCase().includes("processing")) {
    status = "PROCESSING";
  }

  var activeSheet = SpreadsheetApp.getActiveSheet().getName();
  var contextDetails = additionalDetails || `Active Sheet: ${activeSheet}`;
  logSheet.insertRowAfter(1);
  logSheet.getRange("A2:D2").setValues([[timestamp, status, message, contextDetails]]);

  var statusCell = logSheet.getRange("B2");
  switch (status) {
    case "ERROR":
      statusCell.setBackground("#ffcdd2");
      break;
    case "SUCCESS":
      statusCell.setBackground("#c8e6c9");
      break;
    case "PROCESSING":
      statusCell.setBackground("#fff9c4");
      break;
    default:
      statusCell.setBackground("#ffffff");
  }
  logSheet.getRange("A2:D2").setHorizontalAlignment("left").setVerticalAlignment("middle").setWrap(true);
  var lastRow = logSheet.getLastRow();
  if (lastRow > 101) {
    logSheet.deleteRows(102, lastRow - 101);
  }
  logSheet.autoResizeColumns(1, 4);
  if (logSheet.isSheetHidden()) {
    logSheet.showSheet();
  }
}

// Helper function to manage settings sheet
function getSettingsSheet() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var settingsSheet = ss.getSheetByName("Settings");

  if (!settingsSheet) {
    settingsSheet = ss.insertSheet("Settings");
    settingsSheet.getRange("A1:B1").setValues([["Setting", "Value"]]);
    settingsSheet.setFrozenRows(1);
    settingsSheet.setColumnWidth(1, 200);
    settingsSheet.setColumnWidth(2, 300);
    settingsSheet.hideSheet();
  }
  return settingsSheet;
}

// Helper function to update settings
function updateSetting(settingName, value) {
  var sheet = getSettingsSheet();
  var data = sheet.getDataRange().getValues();
  var rowIndex = -1;
  for (var i = 1; i < data.length; i++) {
    if (data[i][0] === settingName) {
      rowIndex = i + 1;
      break;
    }
  }
  if (rowIndex === -1) {
    rowIndex = data.length + 1;
  }
  sheet.getRange(rowIndex, 1, 1, 2).setValues([[settingName, value]]);
}

// Helper function to manage stats sheet
function getStatsSheet() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var statsSheet = ss.getSheetByName("Stats");

  if (!statsSheet) {
    statsSheet = ss.insertSheet("Stats");
    statsSheet.getRange("A1:B1").setValues([["Metric", "Value"]]);
    statsSheet.setFrozenRows(1);
    statsSheet.setColumnWidth(1, 200);
    statsSheet.setColumnWidth(2, 300);
    var headerRange = statsSheet.getRange("A1:B1");
    headerRange.setBackground("#f3f3f3").setFontWeight("bold").setHorizontalAlignment("center");
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
  for (var i = 1; i < data.length; i++) {
    if (data[i][0] === metric) {
      rowIndex = i + 1;
      break;
    }
  }
  if (rowIndex === -1) {
    rowIndex = data.length + 1;
    sheet.getRange(rowIndex, 1).setValue(metric);
  }
  sheet.getRange(rowIndex, 2).setValue(value);
}

// Main classification function
function categoriseTransactions(config) {
  try {
    Logger.log("Starting categorisation with config: " + JSON.stringify(config));
    updateStatus("Starting categorisation..."); // Use local updateStatus

    var serviceConfig = Config.getServiceConfig(); // USE Config.getServiceConfig()
    var sheet = SpreadsheetApp.getActiveSheet();
    var spreadsheetId = sheet.getParent().getId();

    // --- Get selected range ---
    var range = null;
    var startRow, numRows, endRow, rangeA1;
    try {
      Logger.log("Attempting to get active range...");
      range = SpreadsheetApp.getActiveRange();

      if (!range) {
        throw new Error("No cells selected. Please select the transaction rows you want to categorise first.");
      }

      Logger.log("Got active range. Attempting to get range properties...");
      startRow = range.getRow();
      numRows = range.getNumRows();
      endRow = startRow + numRows - 1;
      rangeA1 = range.getA1Notation();
      Logger.log(`Selected range: ${rangeA1}, StartRow: ${startRow}, NumRows: ${numRows}, EndRow: ${endRow}`);

      // Add startRow and endRow to the config object passed from the dialog
      config.startRow = startRow;
      config.endRow = endRow;
      Logger.log("Updated config with range info: " + JSON.stringify(config));
    } catch (e) {
      Logger.log("Error getting or processing active range: " + e.toString());
      // Provide a more user-friendly error message
      throw new Error(
        "Could not get selected range. Please ensure cells are selected before running. Error details: " + e.message
      );
    }

    // --- Validate selection --- (Moved validation after successful range acquisition)
    if (numRows < 1) {
      // This check might be redundant if getActiveRange throws an error for invalid selections, but keep for safety
      throw new Error("Invalid selection. Please select at least one row containing data.");
    }

    // --- Read data from selected range ---
    // Get descriptions
    var descriptionRangeA1 = config.descriptionCol + startRow + ":" + config.descriptionCol + endRow;
    Logger.log(`Attempting to read descriptions from range: ${descriptionRangeA1}`);
    var descriptionsData = sheet.getRange(descriptionRangeA1).getValues();
    // Filter out rows where description is empty in the original selection
    var validRows = [];
    for (let i = 0; i < descriptionsData.length; i++) {
      if (descriptionsData[i][0] && descriptionsData[i][0].toString().trim() !== "") {
        validRows.push({ index: i, description: descriptionsData[i][0] });
      }
    }

    if (validRows.length === 0) {
      throw new Error("No valid descriptions found in the selected range's description column.");
    }
    Logger.log(`Found ${validRows.length} valid descriptions in selection.`);

    // Get amounts if column specified
    var amountData = [];
    var hasAmounts = config.amountCol && config.amountCol.trim() !== "";
    if (hasAmounts) {
      var amountRangeA1 = config.amountCol + startRow + ":" + config.amountCol + endRow;
      Logger.log(`Attempting to read amounts from range: ${amountRangeA1}`);
      amountData = sheet.getRange(amountRangeA1).getValues();
    } else {
      Logger.log("Amount column not specified or empty.");
    }

    // Prepare transactions only for rows with valid descriptions
    var transactions = [];
    for (var i = 0; i < validRows.length; i++) {
      var validRow = validRows[i];
      var originalIndex = validRow.index; // Index relative to the start of the *selection*
      var description = String(validRow.description).trim(); // Already known to be valid
      var transaction = { description: description };

      if (hasAmounts && originalIndex < amountData.length) {
        var amount = amountData[originalIndex][0]; // Use index relative to selection
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
          transaction.amount = parsedAmount;
          transaction.money_in = parsedAmount >= 0;
          Logger.log(`Transaction: "${description}" with amount ${parsedAmount}, money_in: ${transaction.money_in}`);
        } else {
          // If amount is invalid, just use the description
          transaction.amount = description;
          transaction.money_in = null;
          Logger.log(`Transaction: "${description}" without valid amount`);
        }
      } else {
        // If no amounts column or missing amount, just use the description
        transaction.amount = description;
        transaction.money_in = null;
      }

      transactions.push(transaction);
    }

    if (transactions.length === 0) {
      throw new Error("No transactions found to categorise");
    }

    Logger.log("Found " + transactions.length + " transactions to categorise");
    updateStatus("Processing " + transactions.length + " transactions..."); // Use local updateStatus

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

        var responseCode = response.getResponseCode();

        // Accept 200 OK or 202 Accepted as success for starting the job
        if (responseCode === 200 || responseCode === 202) {
          break; // Success, exit retry loop
        } else if (responseCode === 502 || responseCode === 503 || responseCode === 504) {
          // Retry on gateway errors
          error = `Server returned ${responseCode}`;
          retryCount++;
          continue;
        } else if (responseCode === 409) {
          // Handle 409 Conflict specifically
          var errorText = response.getContentText();
          var errorPayload = JSON.parse(errorText);
          throw new Error(errorPayload.error || "Model configuration conflict. Please check logs or contact support.");
        } else {
          // Don't retry on other errors
          var responseDetails = response.getContentText().substring(0, 200);
          throw new Error(`Server returned error code: ${responseCode}. Details: ${responseDetails}`);
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
    var responseCode = response.getResponseCode(); // Get code again outside loop

    try {
      var responseText = response.getContentText();
      Logger.log("Raw response: " + responseText);
      result = JSON.parse(responseText);

      if (result.error) {
        // Check for detailed validation errors
        if (result.details && Array.isArray(result.details)) {
          var validationErrors = result.details
            .map(function (err) {
              return `Field ${err.location}: ${err.message}`;
            })
            .join(", ");
          throw new Error("Validation error: " + validationErrors);
        } else {
          throw new Error("API error: " + result.error);
        }
      }

      Logger.log("Categorisation response: " + JSON.stringify(result));
    } catch (parseError) {
      Logger.log("Error parsing response: " + parseError);
      throw new Error("Error processing server response: " + parseError.toString() + "\nRaw response: " + responseText);
    }

    // --- Handle Sync vs Async ---
    if (responseCode === 200 && result.status === "completed") {
      // Synchronous Success!
      Logger.log("Categorisation completed synchronously.");
      updateStatus("Processing synchronous results..."); // Use local updateStatus
      try {
        var writeSuccess = SheetUtils.writeResultsToSheet(result, config, sheet); // USE SheetUtils.writeResultsToSheet
        if (writeSuccess) {
          updateStatus("Categorisation completed successfully!"); // Use local updateStatus
          return { status: "success", message: "Categorisation completed synchronously!" };
        } else {
          throw new Error("Failed to write synchronous results to sheet.");
        }
      } catch (writeError) {
        throw new Error("Error processing synchronous results: " + writeError.toString());
      }
    } else if ((responseCode === 202 || responseCode === 200) && result.prediction_id) {
      // Asynchronous start (either 202 or 200 with status=processing)
      Logger.log("Categorisation started asynchronously. Prediction ID: " + result.prediction_id);
      updateStatus("Categorisation started, processing in background..."); // Use local updateStatus

      // Store properties for polling
      var userProperties = PropertiesService.getUserProperties();
      userProperties.setProperties({
        PREDICTION_ID: result.prediction_id,
        OPERATION_TYPE: "categorise",
        START_TIME: Date.now().toString(),
        CONFIG: JSON.stringify(config),
        SHEET_ID: sheet.getSheetId().toString(),
        START_ROW: startRow.toString(),
        END_ROW: endRow.toString(),
        SERVICE_URL: serviceConfig.serviceUrl,
      });

      // Show the polling dialog
      showPollingDialog();

      // Return success for async start
      return {
        status: "processing",
        message: "Categorisation started. Please wait for results.",
        predictionId: result.prediction_id,
      };
    } else {
      // Unexpected response format or state
      Logger.log("Unexpected response state. Code: " + responseCode + ", Body: " + JSON.stringify(result));
      throw new Error("Unexpected response from server. Cannot determine status.");
    }
  } catch (error) {
    Logger.log("Categorisation error: " + error.toString());
    updateStatus("Error: " + error.toString()); // Use local updateStatus
    SpreadsheetApp.getUi().alert("Error: " + error.toString());
    return { error: error.toString() };
  }
}

// Helper function to write classification results to a sheet - REMOVED (was writeResultsToSheet, now use SheetUtils.writeResultsToSheet)

// Helper function to find the results array within the API response - REMOVED (was findResultsArray, now use SheetUtils.findResultsArray)

// Helper function to convert column number to letter - REMOVED (was columnToLetter, now use SheetUtils.columnToLetter)

function trainModel(config) {
  var ui = SpreadsheetApp.getUi();
  Logger.log("Entering trainModel (Refactored). Config: " + JSON.stringify(config));
  try {
    // Validate config and get data first
    if (!config || !config.narrativeCol || !config.categoryCol || !config.startRow) {
      throw new Error("Missing required configuration parameters");
    }

    var serviceConfig = Config.getServiceConfig(); // Updated call
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

    // === Start API Interaction Logic (Moved from startTraining) ===

    // Prepare payload
    var payload = JSON.stringify({
      transactions: transactions,
      userId: Session.getEffectiveUser().getEmail(),
    });

    Logger.log(
      "Sending training request with " +
        transactions.length +
        " transactions for user: " +
        Session.getEffectiveUser().getEmail()
    );

    var options = {
      method: "post",
      contentType: "application/json",
      headers: { "X-API-Key": serviceConfig.apiKey },
      payload: payload,
      muteHttpExceptions: true,
    };

    // Retry logic for API call
    var maxRetries = 3;
    var retryCount = 0;
    var response;
    var apiError;

    while (retryCount < maxRetries) {
      try {
        if (retryCount > 0) {
          Logger.log(`Retrying training request (attempt ${retryCount + 1}/${maxRetries})...`);
          Utilities.sleep(Math.pow(2, retryCount) * 1000);
        }
        response = UrlFetchApp.fetch(serviceConfig.serviceUrl + "/train", options);
        var responseCode = response.getResponseCode();

        if (responseCode === 200 || responseCode === 202) {
          apiError = null; // Clear error on success
          break; // Success, exit retry loop
        } else if (responseCode === 502 || responseCode === 503 || responseCode === 504) {
          apiError = `Server gateway error: ${responseCode}`;
          Logger.log(apiError + ", retrying...");
          retryCount++;
          continue;
        } else if (responseCode === 409) {
          // Handle 409 Conflict specifically
          var errorText = response.getContentText();
          var errorPayload = JSON.parse(errorText);
          apiError = errorPayload.error || "Model configuration conflict. Please re-train or contact support.";
          Logger.log(`Training API call failed (409): ${apiError}`);
          throw new Error(apiError); // Throw immediately
        } else {
          var errorText = response.getContentText();
          apiError = `Server returned error code: ${responseCode} - ${errorText.substring(0, 200)}`;
          Logger.log(`Training API call failed: ${apiError}`);
          throw new Error(apiError);
        }
      } catch (e) {
        apiError = e; // Store the caught error
        if (retryCount === maxRetries - 1) {
          Logger.log("Max retries reached for API call.");
          throw new Error(`Training failed after ${maxRetries} attempts: ${e.toString()}`);
        }
        Logger.log("Error during API fetch attempt: " + e.toString() + ", retrying...");
        retryCount++;
      }
    }

    // --- Parse Response and Handle Sync/Async ---

    var responseCode = response.getResponseCode();
    var responseText = response.getContentText();
    Logger.log("Raw API response. Code: " + responseCode + ", Body: " + responseText);

    var result;
    try {
      result = JSON.parse(responseText);
      Logger.log("Parsed JSON result: " + JSON.stringify(result));
    } catch (jsonError) {
      Logger.log("Error parsing JSON response: " + jsonError);
      throw new Error("Invalid JSON response from server: " + responseText);
    }

    // Check for error property within the JSON response itself
    if (result.error) {
      Logger.log("API returned error in JSON body: " + result.error);
      // Handle validation errors specifically if present
      if (result.details && Array.isArray(result.details)) {
        var validationErrors = result.details
          .map(function (err) {
            return `Field ${err.location}: ${err.message}`;
          })
          .join(", ");
        throw new Error("API Validation error: " + validationErrors);
      }
      throw new Error("API error: " + result.error);
    }

    // --- Final Sync/Async Decision ---

    if (responseCode === 200 && result.status === "completed") {
      // Synchronous Success BUT we still show polling dialog for consistency
      Logger.log("Training completed synchronously, but showing polling dialog anyway.");
      updateStatus("Training finished quickly, showing progress..."); // Use local updateStatus
      // Update stats immediately
      updateStats("Last Training Time", new Date().toLocaleString()); // Use local updateStats
      updateStats("Model Status", "Ready"); // Use local updateStats
      updateStats("training_operations", 1); // Use local updateStats
      updateStats("trained_transactions", transactions.length); // Use local updateStats

      // Set properties needed for the polling dialog
      var userProperties = PropertiesService.getUserProperties();
      userProperties.setProperties({
        PREDICTION_ID: "sync_complete_" + Date.now(), // Use unique dummy ID
        OPERATION_TYPE: "training",
        START_TIME: Date.now().toString(),
        SERVICE_URL: serviceConfig.serviceUrl,
        CONFIG: JSON.stringify({}), // Added for polling consistency
        SHEET_ID: sheet.getSheetId().toString(), // Added for polling consistency
        START_ROW: "0", // Placeholder for training
        END_ROW: "0", // Placeholder for training
      });

      // Show the polling dialog (which should show 100% immediately)
      showPollingDialog();

      Logger.log("trainModel finished (sync success). Returning OK to close initial dialog.");
      return { status: "ok" }; // Signal OK to close the initial dialog
    } else if ((responseCode === 202 || responseCode === 200) && result.prediction_id) {
      // Asynchronous Start
      Logger.log("Training started asynchronously (handled within trainModel). Prediction ID: " + result.prediction_id);
      updateStatus("Training started, processing in background..."); // Use local updateStatus

      // Set properties for polling (as before)
      var userProperties = PropertiesService.getUserProperties();
      userProperties.setProperties({
        PREDICTION_ID: result.prediction_id,
        OPERATION_TYPE: "training",
        START_TIME: Date.now().toString(),
        SERVICE_URL: serviceConfig.serviceUrl,
        CONFIG: JSON.stringify({}), // Added for polling consistency
        SHEET_ID: sheet.getSheetId().toString(), // Added for polling consistency
        START_ROW: "0", // Placeholder for training
        END_ROW: "0", // Placeholder for training
      });

      // Show polling dialog
      showPollingDialog();

      Logger.log("trainModel finished (async start). Returning OK to close initial dialog.");
      return { status: "ok" }; // Indicate success to client handler (polling started)
    } else {
      // Unexpected state
      Logger.log("Unexpected training response state. Code: " + responseCode + ", Body: " + JSON.stringify(result));
      throw new Error("Unexpected response from server during training.");
    }

    // === End API Interaction Logic ===
  } catch (error) {
    Logger.log("Training setup error: " + error.toString());
    // Show error in the dialog if it's still open, otherwise use alert
    try {
      // This might fail if the dialog is already closed
      // Re-throw with a clear message, ensuring error.message is captured if available
      throw new Error("Error in trainModel: " + (error.message || error.toString()));
    } catch (e) {
      // If re-throwing fails (dialog closed), show alert
      ui.alert("Error during training setup: " + (error.message || error.toString()));
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
    var configStr = userProperties.getProperty("CONFIG");
    var sheetIdStr = userProperties.getProperty("SHEET_ID");
    var startRowStr = userProperties.getProperty("START_ROW");
    var endRowStr = userProperties.getProperty("END_ROW");
    var serviceUrl = userProperties.getProperty("SERVICE_URL"); // Retrieve the service URL

    // Ensure essential properties are present
    if (!predictionId || !configStr || !sheetIdStr || !startRowStr || !endRowStr || !serviceUrl) {
      // Check serviceUrl too
      Logger.log("pollOperationStatus: Missing required user properties for polling.");
      // Attempt to gracefully handle missing properties if possible, otherwise return error
      if (predictionId && predictionId.startsWith("sync_complete_")) {
        // Handle sync completion case even if other props are missing
        Logger.log("Poll detected synchronous completion via dummy ID: " + predictionId);
        userProperties.deleteAllProperties(); // Clear state
        return {
          status: "completed",
          message: "Operation completed successfully!",
          progress: 100,
        };
      }
      // If async and properties missing, we can't reliably write results later.
      return {
        status: "failed",
        message: "Error: Could not retrieve necessary details to continue processing.",
        progress: 0,
      };
    }

    // Parse config here as it's needed for writing results
    var config = JSON.parse(configStr);
    // Add start/end row from properties into config for writeResultsToSheet consistency
    config.startRow = parseInt(startRowStr);
    config.endRow = parseInt(endRowStr);

    // Find the target sheet using its ID
    var sheet = null;
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var allSheets = ss.getSheets();
    for (var i = 0; i < allSheets.length; i++) {
      if (allSheets[i].getSheetId().toString() === sheetIdStr) {
        sheet = allSheets[i];
        break;
      }
    }
    if (!sheet) {
      Logger.log("pollOperationStatus: Could not find sheet with ID " + sheetIdStr);
      userProperties.deleteAllProperties(); // Clear properties as we can't proceed
      return { status: "failed", message: "Error: Could not find the original sheet to write results.", progress: 0 };
    }

    Logger.log("pollOperationStatus: Polling for ID: " + predictionId + " on Sheet: " + sheet.getName());

    // --- Handle SYNC completion case based on dummy ID --- (Check moved earlier)
    if (predictionId.startsWith("sync_complete_")) {
      Logger.log("Poll detected synchronous completion via dummy ID: " + predictionId);
      userProperties.deleteAllProperties(); // Clear state
      return {
        status: "completed",
        message: "Training completed successfully!", // Hardcode message here
        progress: 100,
      };
    }

    // --- Proceed with actual polling for ASYNC case ---
    var operationType = userProperties.getProperty("OPERATION_TYPE");

    // Call the status endpoint
    var options = {
      headers: { "X-API-Key": Config.getServiceConfig().apiKey }, // USE Config.getServiceConfig()
      muteHttpExceptions: true,
    };

    var response = UrlFetchApp.fetch(serviceUrl + "/status/" + predictionId, options); // Use the retrieved serviceUrl
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
        Logger.log("Operation completed successfully via polling.");

        // Handle categorization results if needed
        if (operationType === "categorise" && result) {
          try {
            var finalResultData = null;
            if (result.result_data && typeof result.result_data === "string") {
              try {
                finalResultData = JSON.parse(result.result_data);
                Logger.log("Parsed result_data from polling response.");
              } catch (parseError) {
                Logger.log("Error parsing result_data JSON: " + parseError);
                throw new Error("Completed, but failed to parse results data from server.");
              }
            } else if (result.results) {
              // Fallback for potential direct results (less likely now)
              Logger.log("Using direct results field from polling response (fallback).");
              finalResultData = result; // Pass the whole result object if results are directly in it
            } else {
              Logger.log("Completed, but no result_data or results field found in the status response.");
              throw new Error("Completed, but results data was missing from the server response.");
            }

            // Pass the retrieved sheet object, the full config, and the parsed result data
            SheetUtils.writeResultsToSheet(finalResultData, config, sheet); // USE SheetUtils.writeResultsToSheet
            Logger.log("Results written to sheet successfully from polling.");
          } catch (writeError) {
            Logger.log("Error processing or writing results to sheet from polling: " + writeError);
            // Update status to reflect the writing error but still mark as completed?
            // Let's return a slightly different completed message
            userProperties.deleteAllProperties(); // Still clear state
            return {
              status: "completed",
              message: "Categorisation complete, but error processing/writing results: " + writeError.message,
              progress: 100,
            };
          }
        } else if (operationType === "training") {
          Logger.log("Training completed successfully via polling.");
          // Update training stats if needed (consider adding this back if useful)
          // updateStats("Last Training Time", new Date().toLocaleString());
          // updateStats("Model Status", "Ready");
          // updateStats("training_operations", (parseInt(PropertiesService.getScriptProperties().getProperty("training_operations") || "0") + 1).toString());
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
