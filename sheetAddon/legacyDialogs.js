/**
 * Legacy Dialog Functions
 * Contains the remaining dialog-based interfaces for backwards compatibility
 * These will eventually be replaced by the modern card interface
 */

/**
 * Show categorize dialog (legacy)
 */
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

/**
 * Show training dialog (legacy)
 */
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