// const CLASSIFICATION_SERVICE_URL = "https://txclassify.onrender.com";
// const CLASSIFICATION_SERVICE_URL = "https://api.expensesorted.com"; // Will be removed

// Legacy menu functions removed - now using modern Google Workspace Add-on architecture
// The new entry point is in card_ui.js: onHomepage() function

// TEMPORARY: Fallback menu for testing while we set up the Google Workspace Add-on
function onOpen() {
  var menu = SpreadsheetApp.getUi().createAddonMenu();
  menu
    .addItem("Import Bank Transactions", "showImportDialog")
    .addSeparator()
    .addItem("Configure API Key", "showApiKeyDialog")
    .addItem("Categorise New Transactions", "showCategoriseDialog")
    .addItem("Train Model", "showTrainingDialog")
    .addSeparator()
    .addItem("📱 Preview Modern Card Interface", "previewCardInterface")
    .addItem("🔧 Test Card Functions", "testCardFunctions")
    .addToUi();
}

// Test function to verify card functions work
function testCardFunctions() {
  try {
    // Test that card functions exist and can be called
    var homepage = onHomepage();
    var apiCard = createApiKeyCard();
    var catCard = createCategoriseCard();
    var trainCard = createTrainingCard();
    
    SpreadsheetApp.getUi().alert("✅ All card functions are working! The cards are ready for the Google Workspace Add-on sidebar.\n\nNote: Cards can only be displayed in the Google Workspace Add-on sidebar, not in dialogs.");
  } catch (e) {
    SpreadsheetApp.getUi().alert("❌ Error with card functions: " + e.toString());
  }
}

// New function to preview what the card interface looks like
function previewCardInterface() {
  var html = HtmlService.createHtmlOutput(`
    <style>
      body { font-family: 'Google Sans', Arial, sans-serif; padding: 20px; background: #f8f9fa; }
      .card { background: white; border-radius: 8px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }
      .card-header { font-size: 18px; font-weight: 500; color: #202124; margin-bottom: 16px; }
      .section-header { font-size: 14px; font-weight: 500; color: #5f6368; margin: 16px 0 8px 0; }
      .button { 
        background: #1a73e8; color: white; border: none; padding: 8px 16px; 
        border-radius: 4px; margin: 4px; cursor: pointer; width: 100%; text-align: left;
        font-size: 14px; display: block; text-decoration: none;
      }
      .button:hover { background: #1557b0; }
      .preview-note { 
        background: #e8f0fe; color: #1967d2; padding: 12px; border-radius: 4px; 
        margin-bottom: 16px; font-size: 13px; border-left: 4px solid #1967d2;
      }
      .config-button { background: #34a853; }
      .config-button:hover { background: #2d8f47; }
    </style>
    
    <div class="preview-note">
      📱 <strong>Card Interface Preview</strong><br>
      This shows what the modern Google Workspace Add-on sidebar interface looks like. 
      In production, this would appear as a sidebar in Google Sheets.
    </div>
    
    <div class="card">
      <div class="card-header">ExpenseSorted Tools</div>
      
      <div class="section-header">Transaction Management</div>
      <button class="button" onclick="alert('This would open the Import wizard')">Import Bank Transactions</button>
      <button class="button" onclick="alert('This would open the Categorise card')">Categorise New Transactions</button>
      <button class="button" onclick="alert('This would open the Training card')">Train Model</button>
      
      <div class="section-header">Configuration</div>
      <button class="button config-button" onclick="alert('This would open the API Key card')">Configure API Key</button>
    </div>
    
    <div class="card">
      <div class="card-header">Benefits of Card Interface</div>
      <ul style="color: #5f6368; font-size: 13px; line-height: 1.4;">
        <li>✅ Modern Google Workspace design</li>
        <li>✅ No need to select cell ranges manually</li>
        <li>✅ Better mobile compatibility</li>
        <li>✅ Consistent with other Google add-ons</li>
        <li>✅ Cleaner, more intuitive interface</li>
      </ul>
    </div>
    
    <div class="preview-note">
      <strong>Current Status:</strong> Your add-on has been successfully migrated! 
      The card functions are built and ready. For now, use the working dialogs in the Add-ons menu.
    </div>
  `)
    .setWidth(450)
    .setHeight(600);

  SpreadsheetApp.getUi().showModalDialog(html, "Modern Card Interface Preview");
}

// Recreate the essential dialogs for immediate use
function showApiKeyDialog() {
  var properties = PropertiesService.getScriptProperties();
  var existingApiKey = properties.getProperty("API_KEY") || "";

  var html = HtmlService.createHtmlOutput(`
    <style>
      body { font-family: Arial, sans-serif; padding: 20px; }
      .form-group { margin-bottom: 15px; }
      label { display: block; margin-bottom: 5px; font-weight: bold; }
      input { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
      button { padding: 10px 20px; background: #4285f4; color: white; border: none; border-radius: 4px; cursor: pointer; }
      button:hover { background: #3367d6; }
      .current-key { background: #f0f8ff; padding: 10px; border-radius: 4px; margin-bottom: 15px; word-break: break-all; font-family: monospace; }
    </style>
    
    ${existingApiKey ? `<div class="current-key"><strong>Current API Key:</strong><br>${existingApiKey}</div>` : ''}
    
    <div class="form-group">
      <label>API Key:</label>
      <input type="text" id="apiKey" value="${existingApiKey}" placeholder="Enter your API key">
    </div>
    
    <button onclick="saveKey()">Save API Key</button>
    
    <script>
      function saveKey() {
        var apiKey = document.getElementById('apiKey').value.trim();
        if (!apiKey) {
          alert('Please enter an API key');
          return;
        }
        
        google.script.run
          .withSuccessHandler(() => {
            alert('API key saved successfully!');
            google.script.host.close();
          })
          .withFailureHandler((error) => {
            alert('Error: ' + error.message);
          })
          .triggerSaveApiKey(apiKey);
      }
    </script>
  `)
    .setWidth(400)
    .setHeight(250);

  SpreadsheetApp.getUi().showModalDialog(html, "Configure API Key");
}

function showCategoriseDialog() {
  var html = HtmlService.createHtmlOutput(`
    <style>
      body { font-family: Arial, sans-serif; padding: 20px; }
      .form-group { margin-bottom: 15px; }
      label { display: block; margin-bottom: 5px; font-weight: bold; }
      select, input { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
      button { padding: 10px 20px; background: #4285f4; color: white; border: none; border-radius: 4px; cursor: pointer; }
      button:hover { background: #3367d6; }
      .help-text { font-size: 12px; color: #666; margin-top: 4px; }
    </style>
    
    <div class="form-group">
      <label>Description Column:</label>
      <select id="descCol">
        <option value="A">A</option><option value="B">B</option><option value="C" selected>C</option>
        <option value="D">D</option><option value="E">E</option><option value="F">F</option>
      </select>
      <div class="help-text">Column containing transaction descriptions</div>
    </div>
    
    <div class="form-group">
      <label>Category Column (Output):</label>
      <select id="catCol">
        <option value="D">D</option><option value="E" selected>E</option><option value="F">F</option>
        <option value="G">G</option><option value="H">H</option>
      </select>
      <div class="help-text">Where categories will be written</div>
    </div>
    
    <div class="form-group">
      <label>Amount Column (Optional):</label>
      <select id="amountCol">
        <option value="">-- None --</option><option value="A">A</option><option value="B" selected>B</option>
        <option value="C">C</option><option value="D">D</option>
      </select>
    </div>
    
    <div class="form-group">
      <label>Start Row:</label>
      <input type="number" id="startRow" value="2" min="1">
    </div>
    
    <div class="form-group">
      <label>End Row (optional):</label>
      <input type="number" id="endRow" min="1" placeholder="Leave empty for all rows">
    </div>
    
    <button onclick="categorize()">Categorise Transactions</button>
    
    <script>
      function categorize() {
        var config = {
          descriptionCol: document.getElementById('descCol').value,
          categoryCol: document.getElementById('catCol').value,
          amountCol: document.getElementById('amountCol').value,
          startRow: parseInt(document.getElementById('startRow').value) || 2,
          endRow: document.getElementById('endRow').value ? parseInt(document.getElementById('endRow').value) : null
        };
        
        google.script.run
          .withSuccessHandler(() => {
            alert('Categorisation started! Check the polling dialog for progress.');
            google.script.host.close();
          })
          .withFailureHandler((error) => {
            alert('Error: ' + error.message);
          })
          .categoriseTransactions(config);
      }
    </script>
  `)
    .setWidth(400)
    .setHeight(450);

  SpreadsheetApp.getUi().showModalDialog(html, "Categorise Transactions");
}

function showTrainingDialog() {
  var html = HtmlService.createHtmlOutput(`
    <style>
      body { font-family: Arial, sans-serif; padding: 20px; }
      .form-group { margin-bottom: 15px; }
      label { display: block; margin-bottom: 5px; font-weight: bold; }
      select, input { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
      button { padding: 10px 20px; background: #4285f4; color: white; border: none; border-radius: 4px; cursor: pointer; }
      button:hover { background: #3367d6; }
      .help-text { font-size: 12px; color: #666; margin-top: 4px; }
    </style>
    
    <div class="form-group">
      <label>Description Column:</label>
      <select id="descCol">
        <option value="A">A</option><option value="B">B</option><option value="C" selected>C</option>
        <option value="D">D</option><option value="E">E</option><option value="F">F</option>
      </select>
      <div class="help-text">Column with transaction descriptions</div>
    </div>
    
    <div class="form-group">
      <label>Category Column:</label>
      <select id="catCol">
        <option value="D">D</option><option value="E" selected>E</option><option value="F">F</option>
        <option value="G">G</option><option value="H">H</option>
      </select>
      <div class="help-text">Column with existing categories for training</div>
    </div>
    
    <div class="form-group">
      <label>Amount Column (Optional):</label>
      <select id="amountCol">
        <option value="">-- None --</option><option value="A">A</option><option value="B" selected>B</option>
        <option value="C">C</option><option value="D">D</option>
      </select>
    </div>
    
    <div class="form-group">
      <label>Start Row:</label>
      <input type="number" id="startRow" value="2" min="1">
      <div class="help-text">First row with data (usually 2 to skip headers)</div>
    </div>
    
    <button onclick="train()">Train Model</button>
    
    <script>
      function train() {
        var config = {
          narrativeCol: document.getElementById('descCol').value,
          categoryCol: document.getElementById('catCol').value,
          amountCol: document.getElementById('amountCol').value,
          startRow: parseInt(document.getElementById('startRow').value) || 2
        };
        
        google.script.run
          .withSuccessHandler(() => {
            alert('Training started! Check the polling dialog for progress.');
            google.script.host.close();
          })
          .withFailureHandler((error) => {
            alert('Error: ' + error.message);
          })
          .trainModel(config);
      }
    </script>
  `)
    .setWidth(400)
    .setHeight(400);

  SpreadsheetApp.getUi().showModalDialog(html, "Train Model");
}

// Enhanced Import Dialog - Full-featured version with fixed file upload
function showImportDialog() {
  var html = HtmlService.createHtmlOutput(`
    <style>
      body { 
        font-family: Arial, sans-serif; 
        padding: 20px; 
        margin: 0;
        background: #f8f9fa;
      }
      .form-group { 
        margin-bottom: 15px; 
      }
      label { 
        display: block; 
        margin-bottom: 5px; 
        font-weight: bold;
        color: #333;
      }
      select, input { 
        width: 100%; 
        padding: 10px; 
        border: 1px solid #ddd;
        border-radius: 8px;
        box-sizing: border-box;
        font-size: 14px;
      }
      button { 
        padding: 12px 24px; 
        background: #4285f4; 
        color: white; 
        border: none; 
        border-radius: 8px; 
        cursor: pointer;
        margin-right: 10px;
        margin-top: 10px;
        font-size: 14px;
        font-weight: 500;
      }
      button:hover {
        background: #3367d6;
      }
      button:disabled {
        background: #ccc;
        cursor: not-allowed;
      }
      .secondary-btn {
        background: #6c757d;
      }
      .secondary-btn:hover {
        background: #5a6268;
      }
      .help-text { 
        font-size: 12px; 
        color: #666; 
        margin-top: 4px; 
      }
      .error { 
        color: #d32f2f; 
        margin-top: 10px; 
        display: none; 
        padding: 12px;
        background: #ffebee;
        border-radius: 8px;
        border-left: 4px solid #d32f2f;
      }
      .success {
        color: #2e7d32;
        margin-top: 10px;
        display: none;
        padding: 12px;
        background: #e8f5e8;
        border-radius: 8px;
        border-left: 4px solid #2e7d32;
      }
      .file-upload-area {
        border: 2px dashed #4285f4;
        border-radius: 12px;
        padding: 40px 20px;
        text-align: center;
        margin-bottom: 20px;
        transition: all 0.3s;
        background: white;
        cursor: pointer;
      }
      .file-upload-area:hover {
        border-color: #3367d6;
        background: #f0f7ff;
      }
      .file-upload-area.dragover {
        border-color: #3367d6;
        background-color: #f0f7ff;
        transform: scale(1.02);
      }
      .hidden {
        display: none;
      }
      .step {
        margin-bottom: 30px;
        padding: 24px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      }
      .step-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 16px;
        color: #1a73e8;
        border-bottom: 2px solid #e8f0fe;
        padding-bottom: 8px;
      }
      .preview-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
        font-size: 13px;
        background: white;
        border-radius: 8px;
        overflow: hidden;
      }
      .preview-table th,
      .preview-table td {
        border: 1px solid #e0e0e0;
        padding: 12px 8px;
        text-align: left;
      }
      .preview-table th {
        background-color: #f8f9fa;
        font-weight: 600;
        color: #333;
      }
      .mapping-row {
        display: flex;
        align-items: center;
        margin-bottom: 16px;
        padding: 16px;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
      }
      .mapping-row label {
        flex: 1;
        margin: 0;
        font-weight: 600;
        color: #333;
      }
      .mapping-row select {
        flex: 1;
        margin-left: 16px;
        margin-bottom: 0;
      }
      .sample-data {
        font-size: 12px;
        color: #666;
        margin-top: 4px;
        font-style: italic;
      }
      .progress-bar {
        width: 100%;
        height: 8px;
        background-color: #f0f0f0;
        border-radius: 4px;
        overflow: hidden;
        margin: 10px 0;
      }
      .progress-fill {
        height: 100%;
        background-color: #4285f4;
        transition: width 0.3s ease;
      }
      .profile-section {
        background: #e8f0fe;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 20px;
        border: 1px solid #1a73e8;
      }
      .profile-section h4 {
        margin: 0 0 12px 0;
        color: #1a73e8;
        font-weight: 600;
      }
      .tip-box {
        background: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 8px;
        padding: 12px;
        margin: 16px 0;
      }
      .tip-box .tip-title {
        font-weight: 600;
        color: #f57c00;
        margin-bottom: 4px;
      }
      .column-header {
        font-weight: 600;
        color: #1a73e8;
        margin-bottom: 4px;
      }
      .step-indicator {
        display: flex;
        justify-content: center;
        margin-bottom: 30px;
        gap: 8px;
      }
      .step-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #e0e0e0;
      }
      .step-dot.active {
        background: #4285f4;
      }
      .step-dot.completed {
        background: #34a853;
      }
    </style>
    
    <div class="step-indicator">
      <div class="step-dot completed" id="dot1"></div>
      <div class="step-dot" id="dot2"></div>
      <div class="step-dot" id="dot3"></div>
    </div>

    <div id="step1" class="step">
      <div class="step-title">📄 Step 1: Upload Your Bank Transaction File</div>
      
      <div class="file-upload-area" onclick="triggerFileInput()">
        <div style="margin-bottom: 12px; font-size: 18px;">
          <strong>📤 Drop your CSV file here or click to browse</strong>
        </div>
        <div class="help-text">
          Supported: CSV files from ANZ, ASB, Westpac, BNZ, and other banks<br>
          We'll automatically detect your bank format and suggest field mappings
        </div>
        <input type="file" id="fileInput" accept=".csv" style="display: none;">
      </div>
      
      <div id="fileInfo" class="success hidden">
        <strong>✅ File uploaded:</strong> <span id="fileName"></span>
        <div class="help-text">File ready for analysis. Click 'Analyze File' to continue.</div>
      </div>
      
      <button onclick="analyzeFile()" id="analyzeBtn" disabled>🔍 Analyze File</button>
    </div>

    <div id="step2" class="step hidden">
      <div class="step-title">🎯 Step 2: Configure Column Mapping</div>
      
      <div class="tip-box">
        <div class="tip-title">💡 Smart Mapping</div>
        We've auto-detected your column types based on common bank formats. 
        Each field type can only be used once - just like our web app!
      </div>

      <!-- Profile Management -->
      <div class="profile-section">
        <h4>💾 Import Profiles</h4>
        <div class="form-group">
          <label>Load Saved Profile (optional):</label>
          <select id="profileSelect">
            <option value="">-- Select a saved profile --</option>
          </select>
          <div class="help-text">Profiles remember your column mappings for different banks</div>
        </div>
      </div>
      
      <div id="columnMappings"></div>
      
      <div class="form-group">
        <label>Date Format:</label>
        <select id="dateFormat">
          <option value="yyyy-MM-dd">YYYY-MM-DD (2023-12-31)</option>
          <option value="MM/dd/yyyy">MM/DD/YYYY (12/31/2023)</option>
          <option value="dd/MM/yyyy">DD/MM/YYYY (31/12/2023)</option>
          <option value="dd.MM.yyyy">DD.MM.YYYY (31.12.2023)</option>
          <option value="yyyy/MM/dd">YYYY/MM/DD (2023/12/31)</option>
        </select>
      </div>

      <!-- Profile Saving -->
      <div class="profile-section">
        <h4>💾 Save This Configuration</h4>
        <div style="display: flex; gap: 12px;">
          <div style="flex: 1;">
            <label>Profile Name (optional):</label>
            <input type="text" id="profileName" placeholder="e.g., ANZ Personal Account">
          </div>
          <div style="flex: 1;">
            <label>Bank Name (optional):</label>
            <input type="text" id="bankName" placeholder="e.g., ANZ, ASB, Westpac">
          </div>
        </div>
      </div>
      
      <button onclick="previewImport()" id="previewBtn">🔍 Preview Import</button>
    </div>

    <div id="step3" class="step hidden">
      <div class="step-title">✅ Step 3: Preview & Import</div>
      
      <div id="previewArea"></div>
      
      <div class="tip-box">
        <div class="tip-title">🎯 Import Target: Expense-Detail Sheet</div>
        Your transactions will be imported with: Source, Date, Description, Amount, Category (auto-added), Currency, Amount in AUD<br>
        <strong>💡 ANZ/NZ Banks:</strong> If you have two description fields, they'll be automatically combined!
      </div>
      
      <button onclick="importTransactions()" id="importBtn">🚀 Import to Expense-Detail Sheet</button>
      <button onclick="goBackToMapping()" class="secondary-btn">← Back to Mapping</button>
      <button onclick="closeDialog()" class="secondary-btn">Cancel</button>
    </div>

    <div id="error" class="error"></div>
    <div id="success" class="success"></div>
    
    <div id="progressArea" class="hidden">
      <div class="progress-bar">
        <div class="progress-fill" id="progressFill" style="width: 0%"></div>
      </div>
      <div id="progressText">Processing...</div>
    </div>

    <script>
      let fileData = null;
      let parsedHeaders = [];
      let previewRows = [];
      let mappedData = [];
      let savedProfiles = {}; // Store profiles in localStorage

      // Fixed file input trigger
      function triggerFileInput() {
        document.getElementById('fileInput').click();
      }

      // File input handler with proper event binding
      function initializeFileUpload() {
        const fileInput = document.getElementById('fileInput');
        const fileUploadArea = document.querySelector('.file-upload-area');
        
        if (fileInput) {
          fileInput.onchange = function(e) {
            const file = e.target.files[0];
            if (file) {
              handleFileSelect(file);
            }
          };
        }

        if (fileUploadArea) {
          fileUploadArea.ondragover = function(e) {
            e.preventDefault();
            fileUploadArea.classList.add('dragover');
          };

          fileUploadArea.ondragleave = function(e) {
            e.preventDefault();
            fileUploadArea.classList.remove('dragover');
          };

          fileUploadArea.ondrop = function(e) {
            e.preventDefault();
            fileUploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type === 'text/csv') {
              handleFileSelect(file);
            } else {
              showError('Please select a valid CSV file');
            }
          };
        }
      }

      // Initialize everything
      initializeFileUpload();
      loadSavedProfiles();

      // Profile management
      function loadSavedProfiles() {
        try {
          const profiles = localStorage.getItem('importProfiles');
          if (profiles) {
            savedProfiles = JSON.parse(profiles);
            updateProfileDropdown();
          }
        } catch (e) {
          console.log('No saved profiles found');
        }
      }

      function updateProfileDropdown() {
        const select = document.getElementById('profileSelect');
        if (!select) return;
        
        // Clear existing options except the first one
        while (select.children.length > 1) {
          select.removeChild(select.lastChild);
        }
        
        Object.keys(savedProfiles).forEach(profileId => {
          const profile = savedProfiles[profileId];
          const option = document.createElement('option');
          option.value = profileId;
          option.textContent = profile.name + (profile.bankName ? ' (' + profile.bankName + ')' : '');
          select.appendChild(option);
        });
      }

      function saveProfile() {
        const profileName = document.getElementById('profileName').value.trim();
        const bankName = document.getElementById('bankName').value.trim();
        
        if (!profileName) return;

        const mappings = {};
        parsedHeaders.forEach((header, index) => {
          const mapping = document.getElementById('mapping_' + index).value;
          if (mapping !== 'none') {
            mappings[header] = mapping;
          }
        });

        const profileId = 'profile_' + Date.now();
        savedProfiles[profileId] = {
          id: profileId,
          name: profileName,
          bankName: bankName,
          mappings: mappings,
          dateFormat: document.getElementById('dateFormat').value,
          created: new Date().toISOString()
        };

        localStorage.setItem('importProfiles', JSON.stringify(savedProfiles));
        updateProfileDropdown();
        showSuccess('Profile "' + profileName + '" saved successfully!');
      }

      function initializeProfileHandlers() {
        // Profile selection handler
        const profileSelect = document.getElementById('profileSelect');
        if (profileSelect) {
          profileSelect.onchange = function() {
            const profileId = this.value;
            if (!profileId) return;

            const profile = savedProfiles[profileId];
            if (!profile) return;

            // Apply profile settings
            if (profile.dateFormat) {
              const dateFormatSelect = document.getElementById('dateFormat');
              if (dateFormatSelect) {
                dateFormatSelect.value = profile.dateFormat;
              }
            }

            // Apply mappings
            parsedHeaders.forEach((header, index) => {
              const select = document.getElementById('mapping_' + index);
              if (select && profile.mappings[header]) {
                select.value = profile.mappings[header];
              }
            });

            showSuccess('Profile "' + profile.name + '" loaded!');
          };
        }
      }

      function handleFileSelect(file) {
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileInfo').classList.remove('hidden');
        document.getElementById('analyzeBtn').disabled = false;
        
        const reader = new FileReader();
        reader.onload = function(e) {
          fileData = e.target.result;
        };
        reader.readAsText(file);
      }

      function analyzeFile() {
        if (!fileData) {
          showError('No file selected');
          return;
        }

        showProgress('Analyzing file structure...');
        updateStepIndicator(1);
        
        google.script.run
          .withSuccessHandler(function(result) {
            hideProgress();
            if (result.success) {
              parsedHeaders = result.headers;
              previewRows = result.previewRows;
              setupColumnMappings(result.headers, result.previewRows);
              showStep(2);
              updateStepIndicator(2);
              showSuccess('File analyzed successfully! ' + result.headers.length + ' columns detected.');
            } else {
              showError(result.error || 'Failed to analyze file');
            }
          })
          .withFailureHandler(function(error) {
            hideProgress();
            showError('Error analyzing file: ' + error.message);
          })
          .analyzeCSVFile(fileData);
      }

      function setupColumnMappings(headers, preview) {
        const mappingsDiv = document.getElementById('columnMappings');
        mappingsDiv.innerHTML = '';
        
        // Initialize profile handlers now that step 2 elements exist
        initializeProfileHandlers();
        
        // Updated field types - cleaner labels without column references
        const fieldTypes = [
          { value: 'none', label: '-- Skip this column --' },
          { value: 'source', label: 'Source (Bank/Account)' },
          { value: 'date', label: 'Transaction Date' },
          { value: 'description', label: 'Description' },
          { value: 'description2', label: 'Description 2 (for ANZ/NZ banks)' },
          { value: 'amount_spent', label: 'Amount' },
          { value: 'currency_spent', label: 'Currency' },
          { value: 'amount_base_aud', label: 'Amount in AUD' }
        ];

        // Auto-detect field mappings with confidence-based scoring
        const autoMappings = autoDetectFieldTypes(headers, preview);
        const uniqueFields = new Set(); // Track assigned field types

        // Add info box with import details
        const infoBox = document.createElement('div');
        infoBox.className = 'tip-box';
        infoBox.innerHTML = 
          '<div class="tip-title">🎯 Import Target: Expense-Detail Sheet</div>' +
          'Your transactions will be imported with: Source, Date, Description, Amount, Category (auto-added), Currency, Amount in AUD<br>' +
          '<strong>💡 ANZ/NZ Banks:</strong> If you have two description fields, map both - they\\'ll be automatically combined!';
        mappingsDiv.appendChild(infoBox);

        headers.forEach((header, index) => {
          const row = document.createElement('div');
          row.className = 'mapping-row';
          
          const headerDiv = document.createElement('div');
          headerDiv.style.flex = '1';
          headerDiv.innerHTML = 
            '<div class="column-header">' + header + '</div>' +
            '<div class="sample-data">Sample: ' + (preview[0] && preview[0][index] ? preview[0][index] : 'N/A') + '</div>';
          
          const select = document.createElement('select');
          select.id = 'mapping_' + index;
          select.style.flex = '1';
          select.style.marginLeft = '16px';
          
          // Populate options
          fieldTypes.forEach(fieldType => {
            const option = document.createElement('option');
            option.value = fieldType.value;
            option.textContent = fieldType.label;
            select.appendChild(option);
          });
          
          // Apply auto-detected mapping if available and not already used
          const autoMapping = autoMappings[header];
          if (autoMapping && !uniqueFields.has(autoMapping)) {
            select.value = autoMapping;
            uniqueFields.add(autoMapping);
          }
          
          // Add change listener for unique validation
          select.onchange = function() {
            validateUniqueMappings();
          };
          
          row.appendChild(headerDiv);
          row.appendChild(select);
          mappingsDiv.appendChild(row);
        });
        
        validateUniqueMappings(); // Initial validation
      }

      function autoDetectFieldTypes(headers, previewRows) {
        // Simple auto-detection based on header names
        const mappings = {};
        const patterns = {
          'date': /date|when|time|posted/i,
          'amount_spent': /amount|debit|credit|sum|total|value/i,
          'description': /description|narrative|memo|details|payee|merchant/i,
          'source': /source|bank|account|type/i,
          'currency_spent': /currency|curr|ccy/i
        };
        
        headers.forEach(header => {
          for (const [fieldType, pattern] of Object.entries(patterns)) {
            if (pattern.test(header)) {
              mappings[header] = fieldType;
              break;
            }
          }
        });
        
        return mappings;
      }

      function validateUniqueMappings() {
        const usedMappings = new Set();
        const selects = document.querySelectorAll('[id^="mapping_"]');
        let hasError = false;
        
        selects.forEach(select => {
          const value = select.value;
          select.style.border = '1px solid #ddd'; // Reset border
          
          if (value !== 'none') {
            if (usedMappings.has(value)) {
              select.style.border = '2px solid #d32f2f'; // Red border for duplicates
              hasError = true;
            } else {
              usedMappings.add(value);
            }
          }
        });
        
        // Enable/disable preview button based on validation
        const previewBtn = document.getElementById('previewBtn');
        if (previewBtn) {
          previewBtn.disabled = hasError;
        }
        
        if (hasError) {
          showError('Each field type can only be assigned to one column. Please fix the duplicate assignments (highlighted in red).');
        } else {
          hideError();
        }
      }

      function previewImport() {
        // Validate mappings first
        const mappings = {};
        let hasRequiredField = false;
        
        parsedHeaders.forEach((header, index) => {
          const mapping = document.getElementById('mapping_' + index).value;
          if (mapping !== 'none') {
            mappings[header] = mapping;
            if (mapping === 'description') hasRequiredField = true;
          }
        });
        
        if (!hasRequiredField) {
          showError('At least one Description field must be mapped');
          return;
        }
        
        const dateFormat = document.getElementById('dateFormat').value;
        
        // Save profile if provided
        saveProfile();

        // Start enhanced import flow with training and categorization
        startEnhancedImportFlow(fileData, mappings, dateFormat);
      }

      function startEnhancedImportFlow(csvData, mappings, dateFormat) {
        showProgress('🚀 Starting enhanced import with auto-training and categorization...');
        
        google.script.run
          .withSuccessHandler(function(result) {
            if (result.success) {
              mappedData = result.processedData;
              // Start training and categorization process
              initiateTrainingAndCategorization(result.processedData);
            } else {
              hideProgress();
              showError(result.error || 'Failed to process data');
            }
          })
          .withFailureHandler(function(error) {
            hideProgress();
            showError('Error processing data: ' + error.message);
          })
          .processTransactionData(csvData, mappings, dateFormat);
      }

      function initiateTrainingAndCategorization(processedData) {
        showProgress('🎓 Step 1/3: Training model with existing transactions...');
        
        google.script.run
          .withSuccessHandler(function(trainingResult) {
            console.log('Training result:', trainingResult);
            if (trainingResult.success) {
              if (trainingResult.isAsync) {
                // Async training - start polling for completion
                console.log('Training started asynchronously, starting polling');
                startEnhancedImportPolling(trainingResult.predictionId, 'training', processedData, trainingResult.serviceUrl);
              } else {
                // Sync training completed, start categorization
                console.log('Training completed synchronously, starting categorization');
                showProgress('🎯 Step 2/3: Categorizing new transactions...');
                startCategorization(processedData);
              }
            } else {
              hideProgress();
              showError('Training failed: ' + (trainingResult.error || 'Unknown error'));
            }
          })
          .withFailureHandler(function(error) {
            hideProgress();
            showError('Error starting training: ' + error.message);
          })
          .startModelTrainingForImport();
      }

      function startCategorization(processedData) {
        google.script.run
          .withSuccessHandler(function(categorizeResult) {
            console.log('Categorization result:', categorizeResult);
            console.log('Categorization success:', categorizeResult.success);
            console.log('Categorization isAsync:', categorizeResult.isAsync);
            console.log('Categorization has categories:', categorizeResult.categories ? categorizeResult.categories.length : 'none');
            
            if (categorizeResult.success) {
              if (categorizeResult.isAsync) {
                // Async categorization - start polling for completion
                console.log('Categorization started asynchronously, starting polling');
                startEnhancedImportPolling(categorizeResult.predictionId, 'categorization', processedData, categorizeResult.serviceUrl);
              } else if (categorizeResult.categories && categorizeResult.categories.length > 0) {
                // Categorization completed with results
                console.log('Categorization completed with categories - calling finalizeCategorizedPreview');
                finalizeCategorizedPreview(processedData, categorizeResult.categories);
              } else {
                // Categorization completed but no categories
                console.log('Categorization completed but no categories returned - showing basic preview');
                hideProgress();
                displayPreview(processedData.slice(0, 5));
                showStep(3);
                updateStepIndicator(3);
                showSuccess('Ready to import ' + processedData.length + ' transactions (no categorization available)');
              }
            } else {
              console.log('Categorization failed:', categorizeResult.error);
              hideProgress();
              showError('Categorization failed: ' + (categorizeResult.error || 'Unknown error'));
            }
          })
          .withFailureHandler(function(error) {
            console.log('Categorization withFailureHandler called:', error.message);
            hideProgress();
            showError('Error starting categorization: ' + error.message);
          })
          .startTransactionCategorization(processedData);
      }

      function finalizeCategorizedPreview(processedData, categories) {
        console.log('Finalizing with categories:', categories.slice(0, 5));
        
        // Merge categories into processed data
        const categorizedData = processedData.map((transaction, index) => {
          const category = categories[index] || 'Uncategorized';
          console.log('Transaction ' + index + ': ' + transaction.Description + ' -> ' + category);
          return {
            ...transaction,
            'Category': category
          };
        });
        
        mappedData = categorizedData;
        hideProgress();
        displayPreview(categorizedData.slice(0, 5));
        showStep(3);
        updateStepIndicator(3);
        
        const categorizedCount = categories.filter(cat => cat && cat !== 'Uncategorized').length;
        showSuccess('🎉 Ready to import ' + categorizedData.length + ' transactions with ' + categorizedCount + ' auto-categorized!');
      }

      function displayPreview(data) {
        const previewArea = document.getElementById('previewArea');
        if (data.length === 0) {
          previewArea.innerHTML = '<p>No data to preview</p>';
          return;
        }
        
        const headers = ['Source', 'Date', 'Description', 'Amount Spent', 'Category', 'Currency Spent', 'Amount in Base Currency: AUD'];
        
        let tableHtml = '<table class="preview-table"><thead><tr>';
        headers.forEach(header => {
          tableHtml += '<th>' + header + '</th>';
        });
        tableHtml += '</tr></thead><tbody>';
        
        data.forEach(row => {
          tableHtml += '<tr>';
          headers.forEach(header => {
            let value = row[header] || '';
            if (header === 'Date' && value instanceof Date) {
              value = value.toLocaleDateString();
            }
            if ((header.includes('Amount') || header.includes('Spent')) && typeof value === 'number') {
              value = '$' + value.toFixed(2);
            }
            tableHtml += '<td>' + value + '</td>';
          });
          tableHtml += '</tr>';
        });
        
        tableHtml += '</tbody></table>';
        previewArea.innerHTML = '<p><strong>Preview of first 5 transactions:</strong></p>' + tableHtml;
      }

      function importTransactions() {
        if (!mappedData || mappedData.length === 0) {
          showError('No data to import');
          return;
        }
        
        showProgress('Importing transactions to Expense-Detail sheet...');
        
        google.script.run
          .withSuccessHandler(function(result) {
            hideProgress();
            if (result.success) {
              showSuccess('Successfully imported ' + result.imported + ' transactions to Expense-Detail sheet!');
              document.getElementById('importBtn').disabled = true;
            } else {
              showError(result.error || 'Import failed');
            }
          })
          .withFailureHandler(function(error) {
            hideProgress();
            showError('Error importing: ' + error.message);
          })
          .importToExpenseDetailSheet(mappedData);
      }

      function goBackToMapping() {
        showStep(2);
        updateStepIndicator(2);
      }

      function closeDialog() {
        google.script.host.close();
      }

      function showStep(stepNumber) {
        for (let i = 1; i <= 3; i++) {
          const step = document.getElementById('step' + i);
          if (step) {
            step.classList.toggle('hidden', i !== stepNumber);
          }
        }
      }

      function updateStepIndicator(currentStep) {
        for (let i = 1; i <= 3; i++) {
          const dot = document.getElementById('dot' + i);
          if (dot) {
            dot.classList.remove('active', 'completed');
            if (i < currentStep) {
              dot.classList.add('completed');
            } else if (i === currentStep) {
              dot.classList.add('active');
            }
          }
        }
      }

      function showProgress(message) {
        const progressArea = document.getElementById('progressArea');
        const progressText = document.getElementById('progressText');
        if (progressArea && progressText) {
          progressText.textContent = message;
          progressArea.classList.remove('hidden');
        }
      }

      function hideProgress() {
        const progressArea = document.getElementById('progressArea');
        if (progressArea) {
          progressArea.classList.add('hidden');
        }
      }

      function showError(message) {
        const errorDiv = document.getElementById('error');
        if (errorDiv) {
          errorDiv.textContent = message;
          errorDiv.style.display = 'block';
        }
        hideSuccess();
      }

      function hideError() {
        const errorDiv = document.getElementById('error');
        if (errorDiv) {
          errorDiv.style.display = 'none';
        }
      }

      function showSuccess(message) {
        const successDiv = document.getElementById('success');
        if (successDiv) {
          successDiv.textContent = message;
          successDiv.style.display = 'block';
        }
        hideError();
      }

      function hideSuccess() {
        const successDiv = document.getElementById('success');
        if (successDiv) {
          successDiv.style.display = 'none';
        }
      }

      function startEnhancedImportPolling(predictionId, operationType, processedData, serviceUrl) {
        console.log('Starting enhanced import polling for', operationType, 'with ID:', predictionId);
        
        let progressMessage = operationType === 'training' ? 
          '🤖 Step 1/3: Training model with existing data...' : 
          '🎯 Step 2/3: Categorizing new transactions...';
          
        showProgress(progressMessage);
        
        const maxPolls = 120; // 10 minutes at 5-second intervals
        let pollCount = 0;
        
        function pollStatus() {
          if (pollCount >= maxPolls) {
            hideProgress();
            showError(operationType + ' timed out. Please try again.');
            return;
          }
          
          pollCount++;
          
          google.script.run
            .withSuccessHandler(function(statusResult) {
              console.log('Poll result for', operationType, ':', statusResult);
              
              if (!statusResult) {
                setTimeout(pollStatus, 5000);
                return;
              }
              
              if (statusResult.status === 'completed') {
                console.log(operationType, 'completed successfully');
                
                if (operationType === 'training') {
                  // Training completed, start categorization
                  showProgress('🎯 Step 2/3: Categorizing new transactions...');
                  startCategorization(processedData);
                } else {
                  // Categorization completed, extract categories and show preview
                  let categories = [];
                  if (statusResult.result_data) {
                    try {
                      const resultData = typeof statusResult.result_data === 'string' ? 
                        JSON.parse(statusResult.result_data) : statusResult.result_data;
                      
                      if (resultData.results && Array.isArray(resultData.results)) {
                        categories = resultData.results.map(item => 
                          item.predicted_category || item.Category || item.category || 'Uncategorized'
                        );
                        console.log('Extracted', categories.length, 'categories from async result');
                      }
                    } catch (parseError) {
                      console.error('Error parsing async categorization results:', parseError);
                    }
                  }
                  
                  if (categories.length > 0) {
                    finalizeCategorizedPreview(processedData, categories);
                  } else {
                    hideProgress();
                    displayPreview(processedData.slice(0, 5));
                    showStep(3);
                    updateStepIndicator(3);
                    showSuccess('Ready to import ' + processedData.length + ' transactions (no categorization available)');
                  }
                }
                
              } else if (statusResult.status === 'failed' || statusResult.status === 'error') {
                hideProgress();
                showError(operationType + ' failed: ' + (statusResult.error || statusResult.message || 'Unknown error'));
                
              } else {
                // Still processing, continue polling
                setTimeout(pollStatus, 5000);
              }
            })
            .withFailureHandler(function(error) {
              console.error('Error polling', operationType, 'status:', error);
              setTimeout(pollStatus, 5000); // Continue polling on error
            })
            .checkJobStatus(predictionId, serviceUrl);
        }
        
        // Start polling immediately
        pollStatus();
      }
    </script>
  `)
    .setWidth(800)
    .setHeight(700);

  SpreadsheetApp.getUi().showModalDialog(html, '📤 Import Bank Transactions');
}

// Legacy showClassifyDialog function removed - now using modern card-based UI in card_ui.js
// The new categorization dialog is handled by createCategoriseCard() and handleCategorise()

// Legacy showTrainingDialog function removed - now using modern card-based UI in card_ui.js
// The new training dialog is handled by createTrainingCard() and handleTrainModel()

// Legacy setupApiKey function removed - now using modern card-based UI in card_ui.js
// The new API key configuration is handled by createApiKeyCard() and handleApiKeySave()

// Global wrapper for saving API Key, callable from client-side HTML
function triggerSaveApiKey(apiKey) {
  try {
    Config.saveApiKey(apiKey); // Calls the method in config.js
    SheetUtils.updateStatus("API key configured successfully!");
    return { success: true }; // Return success to client
  } catch (e) {
    Logger.log("Error in triggerSaveApiKey: " + e.toString());
    return { error: e.message || e.toString() };
  }
}

// Helper function to get stored properties - REMOVED (was getServiceConfig, now use Config.getServiceConfig())

// Duplicate utility functions removed - now using SheetUtils object from sheetUtils.js
// All functions now call SheetUtils.updateStatus(), SheetUtils.getSettingsSheet(), etc.

// Main classification function
function categoriseTransactions(config) {
  try {
    Logger.log("Starting categorisation with config: " + JSON.stringify(config));
    SheetUtils.updateStatus("Starting categorisation...");

    var serviceConfig = Config.getServiceConfig(); // USE Config.getServiceConfig()
    var sheet = SpreadsheetApp.getActiveSheet();
    var spreadsheetId = sheet.getParent().getId();

    // --- Get range from config (modern card-based UI) ---
    var startRow = config.startRow;
    var endRow = config.endRow;
    
    // If endRow is not specified, determine it from the sheet data
    if (!endRow) {
      var lastRow = sheet.getLastRow();
      endRow = lastRow;
    }
    
    var numRows = endRow - startRow + 1;
    Logger.log(`Processing range: StartRow: ${startRow}, EndRow: ${endRow}, NumRows: ${numRows}`);

    // --- Validate range ---
    if (numRows < 1 || startRow < 1) {
      throw new Error("Invalid row range. Please check your start and end row values.");
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
    SheetUtils.updateStatus("Processing " + transactions.length + " transactions...");

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
              SheetUtils.updateStatus("Processing synchronous results...");
      try {
        var writeSuccess = SheetUtils.writeResultsToSheet(result, config, sheet); // USE SheetUtils.writeResultsToSheet
        if (writeSuccess) {
          SheetUtils.updateStatus("Categorisation completed successfully!");
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
              SheetUtils.updateStatus("Categorisation started, processing in background...");

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
    SheetUtils.updateStatus("Error: " + error.toString());
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
      SheetUtils.updateStatus("Training finished quickly, showing progress...");
      // Update stats immediately
              SheetUtils.updateStats("Last Training Time", new Date().toLocaleString());
        SheetUtils.updateStats("Model Status", "Ready");
        SheetUtils.updateStats("training_operations", 1);
        SheetUtils.updateStats("trained_transactions", transactions.length);

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
      SheetUtils.updateStatus("Training started, processing in background...");

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
// Bank Transaction Import Functions

/**
 * Analyze CSV file structure and detect columns
 */
function analyzeCSVFile(csvData) {
  try {
    console.log("Starting CSV analysis...");
    
    if (!csvData || typeof csvData !== 'string') {
      throw new Error('Invalid CSV data provided');
    }
    
    // Split into lines
    const lines = csvData.split('\n').filter(line => line.trim().length > 0);
    
    if (lines.length < 2) {
      throw new Error('CSV file must have at least 2 lines (header + data)');
    }
    
    // Detect delimiter (comma, semicolon, tab)
    let delimiter = ',';
    const delimiters = [',', ';', '\t'];
    let maxColumns = 0;
    
    for (const testDelimiter of delimiters) {
      const testColumns = lines[0].split(testDelimiter).length;
      if (testColumns > maxColumns) {
        maxColumns = testColumns;
        delimiter = testDelimiter;
      }
    }
    
    console.log("Detected delimiter:", delimiter, "with", maxColumns, "columns");
    
    // Parse headers
    const headers = lines[0].split(delimiter).map(header => 
      header.replace(/"/g, '').trim()
    );
    
    // Parse preview rows (first 3 data rows)
    const previewRows = [];
    for (let i = 1; i < Math.min(4, lines.length); i++) {
      const values = lines[i].split(delimiter).map(value => 
        value.replace(/"/g, '').trim()
      );
      previewRows.push(values);
    }
    
    console.log("Analysis complete - Headers:", headers.length, "Preview rows:", previewRows.length);
    
    return {
      success: true,
      headers: headers,
      previewRows: previewRows,
      detectedDelimiter: delimiter,
      totalRows: lines.length - 1
    };
    
  } catch (error) {
    console.error("CSV Analysis Error:", error);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Process transaction data with column mappings
 */
function processTransactionData(csvData, mappings, dateFormat) {
  try {
    console.log("Starting data processing...", "Mappings:", Object.keys(mappings).length);
    
    if (!csvData || !mappings) {
      throw new Error('CSV data and mappings are required');
    }
    
    // Split into lines and remove empty lines
    const lines = csvData.split('\n').filter(line => line.trim().length > 0);
    
    if (lines.length < 2) {
      throw new Error('CSV file must have at least 2 lines');
    }
    
    // Detect delimiter again for consistency
    let delimiter = ',';
    const delimiters = [',', ';', '\t'];
    let maxColumns = 0;
    
    for (const testDelimiter of delimiters) {
      const testColumns = lines[0].split(testDelimiter).length;
      if (testColumns > maxColumns) {
        maxColumns = testColumns;
        delimiter = testDelimiter;
      }
    }
    
    // Parse headers
    const headers = lines[0].split(delimiter).map(header => 
      header.replace(/"/g, '').trim()
    );
    
    // Create reverse mapping (field type -> header name)
    const reverseMapping = {};
    Object.keys(mappings).forEach(header => {
      const fieldType = mappings[header];
      if (fieldType !== 'none') {
        reverseMapping[fieldType] = header;
      }
    });
    
    console.log("Reverse mapping:", reverseMapping);
    
    // Process each data row
    const processedData = [];
    
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(delimiter).map(value => 
        value.replace(/"/g, '').trim()
      );
      
      if (values.length !== headers.length) {
        console.warn(`Row ${i} has ${values.length} values but expected ${headers.length}`);
        continue; // Skip malformed rows
      }
      
      // Create row object with mapped values
      const rowData = {};
      
      // Map each field type to its value
      Object.keys(reverseMapping).forEach(fieldType => {
        const headerName = reverseMapping[fieldType];
        const headerIndex = headers.indexOf(headerName);
        
        if (headerIndex >= 0 && headerIndex < values.length) {
          let value = values[headerIndex];
          
          // Special processing for different field types
          if (fieldType === 'date') {
            // Format date consistently
            rowData[fieldType] = formatDate(value, dateFormat);
          } else if (fieldType === 'amount_spent' || fieldType === 'amount_base_aud') {
            // Parse amount as number
            const cleanAmount = value.replace(/[,$]/g, '').trim();
            rowData[fieldType] = parseFloat(cleanAmount) || 0;
          } else if (fieldType === 'description' || fieldType === 'description2') {
            // Handle multiple description fields
            if (fieldType === 'description' && reverseMapping['description2']) {
              // Combine description and description2
              const desc2Header = reverseMapping['description2'];
              const desc2Index = headers.indexOf(desc2Header);
              const desc2Value = (desc2Index >= 0 && desc2Index < values.length) ? values[desc2Index] : '';
              
              const combined = [value, desc2Value]
                .filter(d => d && d.trim().length > 0)
                .join(' - ');
              rowData[fieldType] = combined;
            } else if (fieldType === 'description2' && reverseMapping['description']) {
              // Skip description2 if we already processed it in description
              // This prevents duplication
            } else {
              rowData[fieldType] = value;
            }
          } else {
            rowData[fieldType] = value;
          }
        }
      });
      
      // Create standardized fields for expense detail sheet structure
      const standardRow = {
        'Source': rowData.source || '',
        'Date': rowData.date || '',
        'Description': rowData.description || '',
        'Amount Spent': rowData.amount_spent || 0,
        'Category': rowData.category || '',
        'Currency Spent': rowData.currency_spent || '',
        'Amount in Base Currency: AUD': rowData.amount_base_aud || rowData.amount_spent || 0
      };
      
      processedData.push(standardRow);
    }
    
    console.log("Processing complete - Processed rows:", processedData.length);
    
    return {
      success: true,
      processedData: processedData,
      totalProcessed: processedData.length
    };
    
  } catch (error) {
    console.error("Data Processing Error:", error);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Import processed data to Google Sheets
 */
function importToSheet(processedData, targetSheet, newSheetName, startingRow) {
  try {
    console.log("Starting sheet import...", "Target:", targetSheet, "Starting row:", startingRow);
    
    if (!processedData || processedData.length === 0) {
      throw new Error('No data to import');
    }
    
    const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    let sheet;
    
    // Determine target sheet
    if (targetSheet === 'new') {
      const sheetName = newSheetName || 'Bank Transactions';
      // Check if sheet already exists
      const existingSheet = spreadsheet.getSheetByName(sheetName);
      if (existingSheet) {
        sheet = existingSheet;
        console.log("Using existing sheet:", sheetName);
      } else {
        sheet = spreadsheet.insertSheet(sheetName);
        console.log("Created new sheet:", sheetName);
      }
    } else {
      sheet = spreadsheet.getActiveSheet();
      console.log("Using current sheet:", sheet.getName());
    }
    
    // Get headers from first row of data
    const headers = Object.keys(processedData[0]);
    
    // Prepare data for batch write
    const dataToWrite = [];
    
    // Add headers if starting at row 1
    if (startingRow === 1) {
      dataToWrite.push(headers);
    }
    
    // Add data rows
    processedData.forEach(row => {
      const rowValues = headers.map(header => {
        const value = row[header];
        // Format dates and numbers appropriately
        if (header === 'Date' && value) {
          return new Date(value);
        } else if (header === 'Amount' && typeof value === 'number') {
          return value;
        } else {
          return value.toString();
        }
      });
      dataToWrite.push(rowValues);
    });
    
    // Write data to sheet
    const range = sheet.getRange(startingRow, 1, dataToWrite.length, headers.length);
    range.setValues(dataToWrite);
    
    // Format the sheet nicely
    if (startingRow === 1) {
      // Format header row
      const headerRange = sheet.getRange(1, 1, 1, headers.length);
      headerRange.setBackground('#4285f4');
      headerRange.setFontColor('white');
      headerRange.setFontWeight('bold');
      
      // Auto-resize columns
      for (let i = 1; i <= headers.length; i++) {
        sheet.autoResizeColumn(i);
      }
      
      // Format amount column as currency
      const amountColIndex = headers.indexOf('Amount') + 1;
      if (amountColIndex > 0) {
        const amountRange = sheet.getRange(2, amountColIndex, processedData.length, 1);
        amountRange.setNumberFormat('$#,##0.00');
      }
      
      // Format date column
      const dateColIndex = headers.indexOf('Date') + 1;
      if (dateColIndex > 0) {
        const dateRange = sheet.getRange(2, dateColIndex, processedData.length, 1);
        dateRange.setNumberFormat('yyyy-mm-dd');
      }
    }
    
    console.log("Import complete - Imported rows:", processedData.length);
    
    return {
      success: true,
      importedCount: processedData.length,
      sheetName: sheet.getName()
    };
    
  } catch (error) {
    console.error("Sheet Import Error:", error);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Format date string according to specified format
 */
function formatDate(dateString, dateFormat) {
  if (!dateString) return '';
  
  try {
    let date;
    
    // Try to parse the date string
    if (dateFormat === 'yyyy-MM-dd') {
      // YYYY-MM-DD format
      date = new Date(dateString);
    } else if (dateFormat === 'MM/dd/yyyy') {
      // MM/DD/YYYY format
      const parts = dateString.split('/');
      if (parts.length === 3) {
        date = new Date(parts[2], parts[0] - 1, parts[1]);
      }
    } else if (dateFormat === 'dd/MM/yyyy') {
      // DD/MM/YYYY format
      const parts = dateString.split('/');
      if (parts.length === 3) {
        date = new Date(parts[2], parts[1] - 1, parts[0]);
      }
    } else if (dateFormat === 'dd.MM.yyyy') {
      // DD.MM.YYYY format
      const parts = dateString.split('.');
      if (parts.length === 3) {
        date = new Date(parts[2], parts[1] - 1, parts[0]);
      }
    } else if (dateFormat === 'yyyy/MM/dd') {
      // YYYY/MM/DD format
      const parts = dateString.split('/');
      if (parts.length === 3) {
        date = new Date(parts[0], parts[1] - 1, parts[2]);
      }
    } else {
      // Fallback - try to parse as-is
      date = new Date(dateString);
    }
    
    // Return formatted date string
    if (date && !isNaN(date.getTime())) {
      return date.toISOString().split('T')[0]; // Return as YYYY-MM-DD
    } else {
      console.warn("Could not parse date:", dateString);
      return dateString; // Return original if parsing fails
    }
    
  } catch (error) {
    console.error("Date formatting error:", error);
    return dateString; // Return original if error occurs
  }
}

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

/**
 * Start training the model using existing transactions from Expense-Detail sheet
 * @return {Object} - Success/error result with prediction ID
 */
function startModelTrainingForImport() {
  try {
    Logger.log("=== STARTING MODEL TRAINING FOR IMPORT ===");
    
    // Get the Expense-Detail sheet
    var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = spreadsheet.getSheetByName("Expense-Detail");
    
    if (!sheet) {
      Logger.log("ERROR: No Expense-Detail sheet found");
      return {
        success: false,
        error: "No Expense-Detail sheet found. Cannot train model without existing transaction data."
      };
    }
    
    var lastRow = sheet.getLastRow();
    Logger.log("Expense-Detail sheet has " + lastRow + " rows (including header)");
    
    if (lastRow < 2) {
      Logger.log("ERROR: Not enough data - only " + lastRow + " rows");
      return {
        success: false,
        error: "Not enough existing transactions in Expense-Detail sheet for training. At least 10 transactions are required."
      };
    }
    
    // Read data from the correct columns: A=Source, B=Date, C=Narrative, D=Amount, E=Category
    var dataRange = sheet.getRange(2, 1, lastRow - 1, 5); // 5 columns: A,B,C,D,E
    var data = dataRange.getValues();
    Logger.log("Read " + data.length + " data rows from sheet (columns A-E)");
    
    // Prepare training transactions and track categories
    var transactions = [];
    var categoriesSeen = new Set();
    var skippedRows = 0;
    
    for (var i = 0; i < data.length; i++) {
      var row = data[i];
      var source = row[0];        // Column A: Source
      var date = row[1];          // Column B: Date
      var description = row[2];   // Column C: Narrative (Description)
      var amountSpent = row[3];   // Column D: Amount Spent
      var category = row[4];      // Column E: Category
      
      // Log first few rows for debugging
      if (i < 3) {
        Logger.log("Row " + (i + 2) + ": [" + source + ", " + date + ", " + description + ", " + amountSpent + ", '" + category + "']");
      }
      
      // Skip rows without description or category
      if (!description || !category || description.toString().trim() === '' || category.toString().trim() === '') {
        skippedRows++;
        if (i < 10) { // Log first 10 skips for debugging
          Logger.log("SKIPPING Row " + (i + 2) + ": Missing description (" + !!description + ") or category (" + !!category + ")");
        }
        continue;
      }
      
      var descriptionStr = description.toString().trim();
      var categoryStr = category.toString().trim();
      
      // Track unique categories
      categoriesSeen.add(categoryStr);
      
      var transaction = {
        description: descriptionStr,
        Category: categoryStr
      };
      
      // Add amount if available
      if (amountSpent && typeof amountSpent === 'number') {
        transaction.amount = amountSpent;
        transaction.money_in = amountSpent >= 0;
      }
      
      transactions.push(transaction);
    }
    
    Logger.log("=== TRAINING DATA SUMMARY ===");
    Logger.log("Total data rows: " + data.length);
    Logger.log("Skipped rows: " + skippedRows);
    Logger.log("Valid transactions: " + transactions.length);
    Logger.log("Unique categories found: " + categoriesSeen.size);
    Logger.log("Categories: " + JSON.stringify(Array.from(categoriesSeen)));
    
    if (transactions.length < 10) {
      Logger.log("ERROR: Not enough valid training data (" + transactions.length + " transactions)");
      return {
        success: false,
        error: `Not enough valid training data. Found ${transactions.length} transactions, but at least 10 are required.`
      };
    }
    
    // Log first few transactions for debugging
    Logger.log("=== SAMPLE TRAINING TRANSACTIONS ===");
    for (var j = 0; j < Math.min(5, transactions.length); j++) {
      Logger.log("TX " + (j + 1) + ": " + JSON.stringify(transactions[j]));
    }
    
    // Get service configuration
    var serviceConfig = Config.getServiceConfig();
    Logger.log("Using service URL: " + serviceConfig.serviceUrl);
    
    // Prepare payload
    var payload = JSON.stringify({
      transactions: transactions,
      userId: Session.getEffectiveUser().getEmail()
    });
    
    Logger.log("=== TRAINING API REQUEST ===");
    Logger.log("Payload size: " + payload.length + " bytes");
    Logger.log("User ID: " + Session.getEffectiveUser().getEmail());
    Logger.log("Transaction count in payload: " + transactions.length);
    
    // Make API call
    var options = {
      method: "post",
      contentType: "application/json",
      headers: { "X-API-Key": serviceConfig.apiKey },
      payload: payload,
      muteHttpExceptions: true
    };
    
    var response = UrlFetchApp.fetch(serviceConfig.serviceUrl + "/train", options);
    var responseCode = response.getResponseCode();
    var responseText = response.getContentText();
    
    Logger.log("=== TRAINING API RESPONSE ===");
    Logger.log("Response code: " + responseCode);
    Logger.log("Response text: " + responseText);
    
    if (responseCode !== 200 && responseCode !== 202) {
      Logger.log("ERROR: Training API failed with code " + responseCode);
      throw new Error(`Training API returned error ${responseCode}: ${responseText}`);
    }
    
    var result = JSON.parse(responseText);
    
    if (result.error) {
      Logger.log("ERROR: Training API returned error in response: " + result.error);
      throw new Error(`Training API error: ${result.error}`);
    }
    
    Logger.log("=== TRAINING SUCCESS ===");
    Logger.log("Prediction ID: " + (result.prediction_id || "sync_training_" + Date.now()));
    
    // Handle async vs sync responses  
    if (responseCode === 202 && result.prediction_id) {
      // Asynchronous training started (202 = accepted for async processing)
      Logger.log("Training started asynchronously. Prediction ID: " + result.prediction_id);
      
      return {
        success: true,
        predictionId: result.prediction_id,
        message: "Training started asynchronously",
        categoriesUsed: Array.from(categoriesSeen),
        transactionCount: transactions.length,
        isAsync: true,
        serviceUrl: serviceConfig.serviceUrl
      };
    } else if (responseCode === 200) {
      // Synchronous training completed (200 = completed immediately)
      Logger.log("Training completed synchronously");
      
      return {
        success: true,
        predictionId: "sync_training_" + Date.now(),
        message: "Training completed successfully",
        categoriesUsed: Array.from(categoriesSeen),
        transactionCount: transactions.length,
        isAsync: false
      };
    }
    
  } catch (error) {
    Logger.log("=== TRAINING ERROR ===");
    Logger.log("Error starting training for import: " + error.message);
    Logger.log("Stack trace: " + error.stack);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Import transactions directly to the Expense-Detail sheet with predefined structure
 * @param {Array} processedData - Array of processed transaction objects
 * @return {Object} - Success/error result
 */
function importToExpenseDetailSheet(processedData) {
  try {
    Logger.log("Starting import to expense detail sheet with " + processedData.length + " transactions");
    
    // Get or create the "Expense-Detail" sheet
    var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    var targetSheet = spreadsheet.getSheetByName("Expense-Detail");
    
    if (!targetSheet) {
      // Create the sheet if it doesn't exist
      targetSheet = spreadsheet.insertSheet("Expense-Detail");
      
      // Add headers to the new sheet
      var headers = ["Source", "Date", "Description", "Amount Spent", "Category", "Currency Spent", "Amount in Base Currency: AUD"];
      targetSheet.getRange(1, 1, 1, headers.length).setValues([headers]);
      
      // Format header row
      var headerRange = targetSheet.getRange(1, 1, 1, headers.length);
      headerRange.setFontWeight("bold");
      headerRange.setBackground("#f0f0f0");
    }
    
    // Find the next empty row (starting from row 2)
    var lastRow = targetSheet.getLastRow();
    var startingRow = Math.max(2, lastRow + 1);
    
    // Transform data to match expense detail structure
    var expenseDetailData = processedData.map(function(transaction) {
      // Extract the mapped values using the exact field names from processTransactionData
      var source = transaction["Source"] || "Imported";
      var date = transaction["Date"];
      var description = transaction["Description"];
      var amountSpent = transaction["Amount Spent"] || 0;
      var category = transaction["Category"] || "";
      var currencySpent = transaction["Currency Spent"] || "";
      var amountBaseAUD = transaction["Amount in Base Currency: AUD"] || amountSpent;
      
      // The description already includes concatenation from processTransactionData
      var finalDescription = description || "";
      
      // Format date properly
      if (date && typeof date === 'string') {
        try {
          var parsedDate = new Date(date);
          if (!isNaN(parsedDate.getTime())) {
            date = parsedDate;
          }
        } catch (e) {
          Logger.log("Date parsing error: " + e.message);
        }
      }
      
      // Ensure amount is a number
      if (typeof amountSpent === 'string') {
        amountSpent = parseFloat(amountSpent.replace(/[,$]/g, '')) || 0;
      }
      if (typeof amountBaseAUD === 'string') {
        amountBaseAUD = parseFloat(amountBaseAUD.replace(/[,$]/g, '')) || amountSpent;
      }
      
      return [source, date, finalDescription, amountSpent, category, currencySpent, amountBaseAUD];
    });
    
    // Import the data
    if (expenseDetailData.length > 0) {
      var dataRange = targetSheet.getRange(startingRow, 1, expenseDetailData.length, 7);
      dataRange.setValues(expenseDetailData);
      
      // Format the data
      // Date column (B) - format as date
      var dateRange = targetSheet.getRange(startingRow, 2, expenseDetailData.length, 1);
      dateRange.setNumberFormat("dd/mm/yyyy");
      
      // Amount columns (D and G) - format as currency
      var amountRange1 = targetSheet.getRange(startingRow, 4, expenseDetailData.length, 1);
      var amountRange2 = targetSheet.getRange(startingRow, 7, expenseDetailData.length, 1);
      amountRange1.setNumberFormat("$#,##0.00");
      amountRange2.setNumberFormat("$#,##0.00");
    }
    
    Logger.log("Successfully imported " + processedData.length + " transactions to Expense-Detail sheet");
    
    return {
      success: true,
      importedCount: processedData.length,
      targetSheet: "Expense-Detail",
      startingRow: startingRow
    };
    
  } catch (error) {
    Logger.log("Error importing to Expense-Detail sheet: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Start categorization of new transactions
 * @param {Array} processedData - Array of processed transaction objects
 * @return {Object} - Success/error result with prediction ID
 */
function startTransactionCategorization(processedData) {
  try {
    Logger.log("=== STARTING TRANSACTION CATEGORIZATION ===");
    Logger.log("Processing " + processedData.length + " transactions for categorization");
    
    // Prepare transactions for categorization
    var transactions = processedData.map(function(transaction, index) {
      var tx = {
        description: transaction["Description"] || ""
      };
      
      // Add amount if available
      var amountSpent = transaction["Amount Spent"];
      if (amountSpent && typeof amountSpent === 'number') {
        tx.amount = amountSpent;
        tx.money_in = amountSpent >= 0;
      }
      
      // Log first few transactions for debugging
      if (index < 3) {
        Logger.log("Categorization TX " + (index + 1) + ": " + JSON.stringify(tx));
      }
      
      return tx;
    });
    
    Logger.log("Prepared " + transactions.length + " transactions for categorization");
    
    // Get service configuration
    var serviceConfig = Config.getServiceConfig();
    
    // Prepare payload
    var payload = JSON.stringify({
      transactions: transactions
    });
    
    Logger.log("=== CATEGORIZATION API REQUEST ===");
    Logger.log("Payload size: " + payload.length + " bytes");
    Logger.log("Service URL: " + serviceConfig.serviceUrl);
    
    // Make API call
    var options = {
      method: "post",
      contentType: "application/json",
      headers: { "X-API-Key": serviceConfig.apiKey },
      payload: payload,
      muteHttpExceptions: true
    };
    
    var response = UrlFetchApp.fetch(serviceConfig.serviceUrl + "/classify", options);
    var responseCode = response.getResponseCode();
    var responseText = response.getContentText();
    
    Logger.log("=== CATEGORIZATION API RESPONSE ===");
    Logger.log("Response code: " + responseCode);
    Logger.log("Response text length: " + responseText.length + " characters");
    
    // Log first part of response for debugging
    var responsePreview = responseText.length > 1000 ? responseText.substring(0, 1000) + "..." : responseText;
    Logger.log("Response preview: " + responsePreview);
    
    if (responseCode !== 200 && responseCode !== 202) {
      Logger.log("ERROR: Categorization API failed with code " + responseCode);
      throw new Error(`Categorization API returned error ${responseCode}: ${responseText}`);
    }
    
    var result = JSON.parse(responseText);
    
    if (result.error) {
      Logger.log("ERROR: Categorization API returned error: " + result.error);
      throw new Error(`Categorization API error: ${result.error}`);
    }
    
    // If synchronous response, extract categories immediately
    if (result.results && Array.isArray(result.results)) {
      Logger.log("=== SYNCHRONOUS CATEGORIZATION RESULTS ===");
      Logger.log("Got " + result.results.length + " categorization results");
      
      var categories = result.results.map(function(item, index) {
        var category = item.predicted_category || item.Category || item.category || 'Uncategorized';
        if (index < 5) { // Log first 5 for debugging
          Logger.log("Result " + (index + 1) + ": " + item.narrative + " -> " + category);
        }
        return category;
      });
      
      Logger.log("Extracted categories: " + JSON.stringify(categories.slice(0, 10)));
      
      return {
        success: true,
        predictionId: "sync_categorize_" + Date.now(),
        message: "Categorization completed synchronously",
        categories: categories,
        isAsync: false
      };
    }
    
    // Handle asynchronous categorization  
    if (responseCode === 202 && result.prediction_id) {
      Logger.log("=== ASYNCHRONOUS CATEGORIZATION STARTED ===");
      Logger.log("Prediction ID: " + result.prediction_id);
      
      return {
        success: true,
        predictionId: result.prediction_id,
        message: "Categorization started asynchronously",
        isAsync: true,
        serviceUrl: serviceConfig.serviceUrl
      };
    }
    
    Logger.log("=== UNKNOWN CATEGORIZATION RESPONSE ===");
    Logger.log("No results or prediction_id found, treating as sync completion");
    
    return {
      success: true,
      predictionId: "sync_categorize_" + Date.now(),
      message: "Categorization processed (unknown status)",
      isAsync: false
    };
    
  } catch (error) {
    Logger.log("=== CATEGORIZATION ERROR ===");
    Logger.log("Error starting categorization: " + error.message);
    Logger.log("Stack trace: " + error.stack);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Quick CSV import with auto-detection
 * @param {string} csvData - Raw CSV data
 * @return {Object} - Import result
 */
function quickImportCSV(csvData) {
  try {
    Logger.log("Starting quick CSV import...");
    
    // Analyze the CSV structure
    var analysisResult = analyzeCSVFile(csvData);
    if (!analysisResult.success) {
      throw new Error(analysisResult.error);
    }
    
    // Auto-detect common mappings
    var autoMappings = autoDetectMappings(analysisResult.headers, analysisResult.previewRows);
    
    // Process the data with auto-detected mappings
    var processResult = processTransactionData(csvData, autoMappings, 'dd/MM/yyyy');
    if (!processResult.success) {
      throw new Error(processResult.error);
    }
    
    // Import to Expense-Detail sheet
    var importResult = importToExpenseDetailSheet(processResult.processedData);
    
    Logger.log("Quick import completed successfully");
    return importResult;
    
  } catch (error) {
    Logger.log("Quick import error: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Auto-detect column mappings for common bank CSV formats
 * @param {Array} headers - CSV headers
 * @param {Array} previewRows - Sample data rows
 * @return {Object} - Mapping object
 */
function autoDetectMappings(headers, previewRows) {
  var mappings = {};
  
  headers.forEach(function(header, index) {
    var lowerHeader = header.toLowerCase();
    var sampleData = previewRows[0] && previewRows[0][index] ? previewRows[0][index] : '';
    
    // Date detection
    if (lowerHeader.includes('date') || lowerHeader.includes('transaction date')) {
      mappings[header] = 'date';
    }
    // Amount detection  
    else if (lowerHeader.includes('amount') || lowerHeader.includes('debit') || lowerHeader.includes('credit')) {
      mappings[header] = 'amount_spent';
    }
    // Description detection
    else if (lowerHeader.includes('desc') || lowerHeader.includes('narrative') || lowerHeader.includes('payee') || lowerHeader.includes('particulars')) {
      mappings[header] = 'description';
    }
    // Source/Account detection
    else if (lowerHeader.includes('source') || lowerHeader.includes('account')) {
      mappings[header] = 'source';
    }
    // Currency detection
    else if (lowerHeader.includes('currency')) {
      mappings[header] = 'currency_spent';
    }
    // Skip other columns by default
  });
  
  Logger.log("Auto-detected mappings: " + JSON.stringify(mappings));
  return mappings;
}

/**
 * Check categorization status and retrieve results
 * @param {string} predictionId - The prediction ID to check
 * @return {Object} - Status result with categories if completed
 */
function checkCategorizationStatus(predictionId) {
  try {
    Logger.log("=== CHECKING CATEGORIZATION STATUS ===");
    Logger.log("Prediction ID: " + predictionId);
    
    // Handle sync completion dummy IDs
    if (predictionId.startsWith("sync_categorize_")) {
      Logger.log("Sync categorization detected - marking as completed");
      return {
        status: "completed",
        message: "Categorization completed successfully",
        categories: [] // Empty for sync dummy IDs
      };
    }
    
    var serviceConfig = Config.getServiceConfig();
    
    var options = {
      headers: { "X-API-Key": serviceConfig.apiKey },
      muteHttpExceptions: true
    };
    
    var response = UrlFetchApp.fetch(serviceConfig.serviceUrl + "/status/" + predictionId, options);
    var responseCode = response.getResponseCode();
    
    Logger.log("Status check response code: " + responseCode);
    
    if (responseCode !== 200) {
      Logger.log("Status not ready, continuing to poll...");
      return {
        status: "processing",
        message: "Categorization in progress..."
      };
    }
    
    var result = JSON.parse(response.getContentText());
    Logger.log("Status response: " + JSON.stringify(result));
    
    if (result.status === "completed" || result.status === "succeeded") {
      Logger.log("=== CATEGORIZATION COMPLETED ===");
      
      // Extract categories from result
      var categories = [];
      
      // Handle direct results (synchronous response)
      if (result.results && Array.isArray(result.results)) {
        Logger.log("Extracting categories from direct results");
        categories = result.results.map(function(item, index) {
          var category = item.predicted_category || item.Category || item.category || 'Uncategorized';
          if (index < 5) {
            Logger.log("Direct result " + (index + 1) + ": " + (item.narrative || item.description) + " -> " + category);
          }
          return category;
        });
      }
      // Handle result_data (asynchronous response)
      else if (result.result_data) {
        try {
          var resultData = typeof result.result_data === 'string' ? JSON.parse(result.result_data) : result.result_data;
          Logger.log("Extracting categories from result_data");
          
          // Find the results array in the response
          var results = resultData.results || resultData.Categories || resultData;
          
          if (Array.isArray(results)) {
            categories = results.map(function(item, index) {
              var category = item.predicted_category || item.Category || item.category || 'Uncategorized';
              if (index < 5) {
                Logger.log("Async result " + (index + 1) + ": " + (item.narrative || item.description) + " -> " + category);
              }
              return category;
            });
          }
        } catch (parseError) {
          Logger.log("Error parsing categorization results: " + parseError);
          categories = [];
        }
      }
      
      Logger.log("=== FINAL CATEGORY EXTRACTION ===");
      Logger.log("Extracted " + categories.length + " categories");
      Logger.log("Sample categories: " + JSON.stringify(categories.slice(0, 10)));
      
      // Check for categories that might cause validation issues
      var uniqueCategories = Array.from(new Set(categories));
      Logger.log("Unique categories returned: " + JSON.stringify(uniqueCategories));
      
      return {
        status: "completed",
        message: "Categorization completed successfully",
        categories: categories
      };
    }
    
    Logger.log("Categorization still in progress: " + result.status);
    return {
      status: result.status || "processing",
      message: result.message || "Categorization in progress..."
    };
    
  } catch (error) {
    Logger.log("=== CATEGORIZATION STATUS ERROR ===");
    Logger.log("Error checking categorization status: " + error.message);
    return {
      status: "processing",
      message: "Checking categorization status..."
    };
  }
}

/**
 * Check job status for any prediction ID (training or categorization)
 * @param {string} predictionId - The prediction ID to check
 * @param {string} serviceUrl - The service URL to use (optional, will use config if not provided)
 * @return {Object} - Status result
 */
function checkJobStatus(predictionId, serviceUrl) {
  try {
    Logger.log("Checking job status for prediction ID: " + predictionId);
    
    var serviceConfig = Config.getServiceConfig();
    var url = (serviceUrl || serviceConfig.serviceUrl) + "/status/" + predictionId;
    
    var options = {
      headers: { "X-API-Key": serviceConfig.apiKey },
      muteHttpExceptions: true
    };
    
    var response = UrlFetchApp.fetch(url, options);
    var responseCode = response.getResponseCode();
    
    if (responseCode !== 200) {
      Logger.log("Job status check failed with code: " + responseCode);
      return {
        status: "error",
        error: "Failed to check status: HTTP " + responseCode
      };
    }
    
    var result = JSON.parse(response.getContentText());
    Logger.log("Job status check result: " + JSON.stringify(result));
    
    return result;
    
  } catch (error) {
    Logger.log("Error checking job status: " + error.message);
    return {
      status: "error", 
      error: error.message
    };
  }
}
