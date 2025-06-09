/**
 * Menu System and Simple Dialogs
 * Handles the Google Sheets add-on menu and basic configuration dialogs
 */

/**
 * Creates the add-on menu when the spreadsheet opens
 */
function onOpen() {
  const ui = SpreadsheetApp.getUi();
  
  ui.createMenu('💰 ExpenseSorted')
    .addItem('📊 Categorise New Transactions', 'previewCardInterface')
    .addItem('🎓 Train Model', 'showTrainingDialog')
    .addItem('📁 Import Bank Transactions', 'showImportDialog')
    .addSeparator()
    .addItem('⚙️ Set API Key', 'showApiKeyDialog')
    .addItem('📋 Categorise (Legacy)', 'showCategoriseDialog')
    .addSeparator()
    .addItem('🧪 Test Card Functions', 'testCardFunctions')
    .addToUi();
}

/**
 * Test function for card interface development
 */
function testCardFunctions() {
  console.log("Testing card functions...");
  previewCardInterface();
}

/**
 * Show API Key configuration dialog
 */
function showApiKeyDialog() {
  var html = HtmlService.createHtmlOutput(`
    <style>
      body { font-family: Arial, sans-serif; padding: 20px; }
      .form-group { margin-bottom: 15px; }
      label { display: block; margin-bottom: 5px; font-weight: bold; }
      input { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
      button { padding: 10px 20px; background: #4285f4; color: white; border: none; border-radius: 4px; cursor: pointer; width: 100%; }
      button:hover { background: #3367d6; }
      .help-text { font-size: 12px; color: #666; margin-top: 4px; }
      .current-key { background: #f0f7ff; padding: 10px; border-radius: 4px; margin-bottom: 15px; }
    </style>
    
    <div class="current-key">
      <strong>Current API Key:</strong> <span id="currentKey">Loading...</span>
    </div>
    
    <div class="form-group">
      <label>API Key:</label>
      <input type="password" id="apiKey" placeholder="Enter your ExpenseSorted API key">
      <div class="help-text">
        Get your API key from <a href="https://expensesorted.com/api-key" target="_blank">expensesorted.com/api-key</a>
      </div>
    </div>
    
    <button onclick="saveApiKey()">Save API Key</button>
    
    <script>
      // Load current API key
      google.script.run
        .withSuccessHandler(function(config) {
          if (config && config.apiKey) {
            document.getElementById('currentKey').textContent = '***' + config.apiKey.slice(-4);
          } else {
            document.getElementById('currentKey').textContent = 'Not set';
          }
        })
        .withFailureHandler(function(error) {
          document.getElementById('currentKey').textContent = 'Error loading';
        })
        .getServiceConfig();
      
      function saveApiKey() {
        var apiKey = document.getElementById('apiKey').value.trim();
        if (!apiKey) {
          alert('Please enter an API key');
          return;
        }
        
        google.script.run
          .withSuccessHandler(function() {
            alert('API key saved successfully!');
            google.script.host.close();
          })
          .withFailureHandler(function(error) {
            alert('Error saving API key: ' + error.message);
          })
          .triggerSaveApiKey(apiKey);
      }
    </script>
  `)
    .setWidth(400)
    .setHeight(300);

  SpreadsheetApp.getUi().showModalDialog(html, "Set API Key");
}

/**
 * Save API key to script properties
 * @param {string} apiKey - The API key to save
 */
function triggerSaveApiKey(apiKey) {
  try {
    var properties = PropertiesService.getScriptProperties();
    properties.setProperty('API_KEY', apiKey);
    
    // Test the API key by making a simple request
    var serviceConfig = Config.getServiceConfig();
    Logger.log("API key saved successfully");
    
    return { success: true };
  } catch (error) {
    Logger.log("Error saving API key: " + error.message);
    throw new Error("Failed to save API key: " + error.message);
  }
} 