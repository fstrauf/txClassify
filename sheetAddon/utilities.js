/**
 * Utility Functions
 * Collection of helper functions used across the application
 */

/**
 * Format date string according to specified format
 * @param {string} dateString - Date string to format
 * @param {string} dateFormat - Target format (yyyy-MM-dd, MM/dd/yyyy, etc.)
 * @return {Date} - Formatted date object
 */
function formatDate(dateString, dateFormat) {
  if (!dateString) return null;
  
  var dateStr = dateString.toString().trim();
  if (!dateStr) return null;
  
  try {
    var year, month, day;
    
    switch (dateFormat) {
      case 'yyyy-MM-dd':
        var parts = dateStr.split('-');
        if (parts.length !== 3) throw new Error('Invalid date format');
        year = parseInt(parts[0]);
        month = parseInt(parts[1]) - 1; // Month is 0-indexed
        day = parseInt(parts[2]);
        break;
        
      case 'MM/dd/yyyy':
        var parts = dateStr.split('/');
        if (parts.length !== 3) throw new Error('Invalid date format');
        month = parseInt(parts[0]) - 1; // Month is 0-indexed
        day = parseInt(parts[1]);
        year = parseInt(parts[2]);
        break;
        
      case 'dd/MM/yyyy':
        var parts = dateStr.split('/');
        if (parts.length !== 3) throw new Error('Invalid date format');
        day = parseInt(parts[0]);
        month = parseInt(parts[1]) - 1; // Month is 0-indexed
        year = parseInt(parts[2]);
        break;
        
      case 'dd.MM.yyyy':
        var parts = dateStr.split('.');
        if (parts.length !== 3) throw new Error('Invalid date format');
        day = parseInt(parts[0]);
        month = parseInt(parts[1]) - 1; // Month is 0-indexed
        year = parseInt(parts[2]);
        break;
        
      case 'yyyy/MM/dd':
        var parts = dateStr.split('/');
        if (parts.length !== 3) throw new Error('Invalid date format');
        year = parseInt(parts[0]);
        month = parseInt(parts[1]) - 1; // Month is 0-indexed
        day = parseInt(parts[2]);
        break;
        
      default:
        // Try to parse as-is
        return new Date(dateStr);
    }
    
    if (isNaN(year) || isNaN(month) || isNaN(day)) {
      throw new Error('Invalid date components');
    }
    
    return new Date(year, month, day);
    
  } catch (error) {
    Logger.log('Error parsing date "' + dateString + '" with format "' + dateFormat + '": ' + error.message);
    // Fallback to default parsing
    try {
      return new Date(dateString);
    } catch (fallbackError) {
      Logger.log('Fallback date parsing also failed: ' + fallbackError.message);
      return null;
    }
  }
}

/**
 * Calculate progress percentage based on start time
 * @param {number} startTime - Start time in milliseconds
 * @return {number} - Progress percentage (0-100)
 */
function calculateProgress(startTime) {
  var elapsed = Date.now() - startTime;
  var estimated = 60000; // 60 seconds estimated
  return Math.min(Math.round((elapsed / estimated) * 100), 95);
}

/**
 * Show polling dialog for long-running operations
 */
function showPollingDialog() {
  var html = HtmlService.createHtmlOutput(`
    <style>
      body { 
        font-family: Arial, sans-serif; 
        padding: 20px; 
        text-align: center;
        background: #f8f9fa;
      }
      .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #4285f4;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
      }
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
      .progress-bar {
        width: 100%;
        height: 8px;
        background-color: #f0f0f0;
        border-radius: 4px;
        overflow: hidden;
        margin: 20px 0;
      }
      .progress-fill {
        height: 100%;
        background-color: #4285f4;
        transition: width 0.3s ease;
        width: 0%;
      }
      .status {
        font-size: 16px;
        margin: 20px 0;
        color: #333;
      }
      .details {
        font-size: 14px;
        color: #666;
        margin: 10px 0;
      }
      button {
        padding: 10px 20px;
        background: #6c757d;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        margin-top: 20px;
      }
      button:hover {
        background: #5a6268;
      }
    </style>
    
    <div class="spinner"></div>
    <div class="status" id="status">Processing your request...</div>
    <div class="progress-bar">
      <div class="progress-fill" id="progressFill"></div>
    </div>
    <div class="details" id="details">This may take up to 2 minutes.</div>
    
    <button onclick="google.script.host.close()">Close</button>
    
    <script>
      var startTime = Date.now();
      
      function updateProgress() {
        var elapsed = Date.now() - startTime;
        var estimated = 120000; // 2 minutes estimated
        var progress = Math.min(Math.round((elapsed / estimated) * 100), 95);
        
        document.getElementById('progressFill').style.width = progress + '%';
        
        if (elapsed < 30000) {
          document.getElementById('status').textContent = 'Initializing...';
        } else if (elapsed < 60000) {
          document.getElementById('status').textContent = 'Processing transactions...';
        } else if (elapsed < 90000) {
          document.getElementById('status').textContent = 'Training model...';
        } else {
          document.getElementById('status').textContent = 'Finalizing...';
        }
        
        // Auto-update every 2 seconds
        if (progress < 95) {
          setTimeout(updateProgress, 2000);
        } else {
          document.getElementById('status').textContent = 'Almost done...';
          document.getElementById('details').textContent = 'Processing should complete shortly.';
        }
      }
      
      // Start progress updates
      updateProgress();
    </script>
  `)
    .setWidth(400)
    .setHeight(300);

  SpreadsheetApp.getUi().showModalDialog(html, "Processing...");
}

/**
 * Poll operation status (legacy function for compatibility)
 */
function pollOperationStatus() {
  // This function is kept for backwards compatibility
  // The actual polling is now handled by individual API functions
  Logger.log("Legacy pollOperationStatus called - redirecting to new polling mechanism");
}

/**
 * Generate a unique ID for operations
 * @return {string} - Unique ID
 */
function generateUniqueId() {
  return 'op_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

/**
 * Validate email format
 * @param {string} email - Email to validate
 * @return {boolean} - True if valid email format
 */
function isValidEmail(email) {
  if (!email || typeof email !== 'string') return false;
  var emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email.trim());
}

/**
 * Safely parse JSON with error handling
 * @param {string} jsonString - JSON string to parse
 * @param {*} defaultValue - Default value if parsing fails
 * @return {*} - Parsed object or default value
 */
function safeJsonParse(jsonString, defaultValue = null) {
  try {
    return JSON.parse(jsonString);
  } catch (error) {
    Logger.log('JSON parse error: ' + error.message);
    return defaultValue;
  }
}

/**
 * Truncate text to specified length
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length
 * @return {string} - Truncated text
 */
function truncateText(text, maxLength = 50) {
  if (!text || typeof text !== 'string') return '';
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - 3) + '...';
}

/**
 * Clean and validate category names
 * @param {string} category - Category to clean
 * @return {string} - Cleaned category
 */
function cleanCategory(category) {
  if (!category || typeof category !== 'string') return '';
  
  // Remove extra whitespace and special characters
  var cleaned = category.trim().replace(/[^\w\s-]/g, '');
  
  // Limit length
  if (cleaned.length > 50) {
    cleaned = cleaned.substring(0, 50);
  }
  
  return cleaned;
}

/**
 * Format currency amount
 * @param {number} amount - Amount to format
 * @param {string} currency - Currency code (default: AUD)
 * @return {string} - Formatted currency string
 */
function formatCurrency(amount, currency = 'AUD') {
  if (typeof amount !== 'number' || isNaN(amount)) return '0.00';
  
  var symbol = '$'; // Default to dollar symbol
  if (currency === 'EUR') symbol = '€';
  if (currency === 'GBP') symbol = '£';
  
  return symbol + Math.abs(amount).toFixed(2);
}

/**
 * Validate transaction data
 * @param {Object} transaction - Transaction object to validate
 * @return {Object} - Validation result
 */
function validateTransaction(transaction) {
  var errors = [];
  
  if (!transaction) {
    errors.push('Transaction is null or undefined');
    return { valid: false, errors: errors };
  }
  
  if (!transaction.Description || transaction.Description.trim() === '') {
    errors.push('Description is required');
  }
  
  if (transaction['Amount Spent'] !== undefined && 
      (typeof transaction['Amount Spent'] !== 'number' || isNaN(transaction['Amount Spent']))) {
    errors.push('Amount Spent must be a valid number');
  }
  
  if (transaction.Date && !(transaction.Date instanceof Date) && typeof transaction.Date !== 'string') {
    errors.push('Date must be a Date object or string');
  }
  
  return {
    valid: errors.length === 0,
    errors: errors
  };
}

/**
 * Batch process array in chunks to avoid timeouts
 * @param {Array} array - Array to process
 * @param {Function} processor - Function to process each item
 * @param {number} chunkSize - Size of each chunk (default: 100)
 * @return {Array} - Processed results
 */
function processInChunks(array, processor, chunkSize = 100) {
  var results = [];
  
  for (var i = 0; i < array.length; i += chunkSize) {
    var chunk = array.slice(i, i + chunkSize);
    var chunkResults = chunk.map(processor);
    results = results.concat(chunkResults);
    
    // Add a small delay to prevent timeouts
    if (i + chunkSize < array.length) {
      Utilities.sleep(100);
    }
  }
  
  return results;
} 