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

// Train model with existing categorized transactions
function trainModel() {
  var ui = SpreadsheetApp.getUi();
  
  try {
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
      ui.alert('No training data found. Please ensure you have transactions with categories.');
      return;
    }
    
    // Show progress
    ui.alert('Training model with ' + transactions.length + ' transactions...');
    
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
    if (response.getResponseCode() !== 200) {
      throw new Error('Training error: ' + response.getContentText());
    }
    
    ui.alert('Training completed successfully!\n\nProcessed: ' + transactions.length + ' transactions');
    
  } catch (error) {
    ui.alert('Error: ' + error.toString());
  }
}

// Main classification function
function classifyTransactions(config) {
  var sheet = SpreadsheetApp.getActiveSheet();
  var ui = SpreadsheetApp.getUi();
  
  try {
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
      ui.alert('No transactions found to classify');
      return;
    }
    
    // Show progress
    ui.alert('Processing ' + transactions.length + ' transactions...');
    
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
    if (response.getResponseCode() !== 200) {
      throw new Error('Classification service error: ' + response.getContentText());
    }
    
    var result = JSON.parse(response.getContentText());
    
    // Check if we got a prediction ID (async mode)
    if (result.prediction_id) {
      ui.alert('Classification started. Please wait a moment and try again.');
      return;
    }
    
    // Process results if we got them directly
    if (!Array.isArray(result)) {
      throw new Error('Unexpected response format');
    }
    
    // Write categories back to sheet
    var categoryCol = sheet.getRange(config.categoryCol + config.startRow);
    var categories = result.map(r => [r.predicted_category]);
    sheet.getRange(config.categoryCol + config.startRow + ":" + config.categoryCol + (parseInt(config.startRow) + categories.length - 1))
      .setValues(categories);
    
    // Optional: Add confidence scores in next column
    var confidenceCol = String.fromCharCode(config.categoryCol.charCodeAt(0) + 1);
    var confidences = result.map(r => [r.similarity_score]);
    sheet.getRange(confidenceCol + config.startRow + ":" + confidenceCol + (parseInt(config.startRow) + confidences.length - 1))
      .setValues(confidences)
      .setNumberFormat("0.00");
    
    // Add headers if needed
    if (config.startRow === "2") {
      sheet.getRange(config.categoryCol + "1").setValue("Category");
      sheet.getRange(confidenceCol + "1").setValue("Confidence");
    }
    
    ui.alert('Classification completed!\n\nProcessed: ' + transactions.length + ' transactions');
    
  } catch (error) {
    ui.alert('Error: ' + error.toString());
  }
} 