// Add menu to the spreadsheet
function onOpen() {
  var ui = SpreadsheetApp.getUi();
  ui.createMenu('Transaction Classifier')
    .addItem('Setup Service URL', 'setupService')
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

// Main classification function
function classifyTransactions(config) {
  var sheet = SpreadsheetApp.getActiveSheet();
  var ui = SpreadsheetApp.getUi();
  
  // Get service URL
  var serviceUrl = PropertiesService.getScriptProperties().getProperty('CLASSIFICATION_SERVICE_URL');
  if (!serviceUrl) {
    ui.alert('Please set up the service URL first');
    return;
  }
  
  try {
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
      payload: JSON.stringify({ transactions: transactions }),
      muteHttpExceptions: true
    };
    
    var response = UrlFetchApp.fetch(serviceUrl + '/classify', options);
    if (response.getResponseCode() !== 200) {
      throw new Error('Classification service error: ' + response.getContentText());
    }
    
    // Process results
    var results = JSON.parse(response.getContentText());
    
    // Write categories back to sheet
    var categoryCol = sheet.getRange(config.categoryCol + config.startRow);
    var categories = results.map(r => [r.predicted_category]);
    sheet.getRange(config.categoryCol + config.startRow + ":" + config.categoryCol + (parseInt(config.startRow) + categories.length - 1))
      .setValues(categories);
    
    // Optional: Add confidence scores in next column
    var confidenceCol = String.fromCharCode(config.categoryCol.charCodeAt(0) + 1);
    var confidences = results.map(r => [r.similarity_score]);
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

// Setup function to configure service URL
function setupService() {
  var ui = SpreadsheetApp.getUi();
  
  var serviceUrl = ui.prompt(
    'Setup Classification Service',
    'Enter the classification service URL:',
    ui.ButtonSet.OK_CANCEL
  );
  
  if (serviceUrl.getSelectedButton() !== ui.Button.OK) return;
  
  PropertiesService.getScriptProperties().setProperty(
    'CLASSIFICATION_SERVICE_URL',
    serviceUrl.getResponseText()
  );
  
  ui.alert('Setup completed successfully!');
} 