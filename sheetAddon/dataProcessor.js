/**
 * Data Processing Module
 * Handles CSV file analysis and transaction data processing
 */

/**
 * Analyze CSV file structure and suggest field mappings
 * @param {string} csvData - Raw CSV data
 * @return {Object} - Analysis result with headers and preview
 */
function analyzeCSVFile(csvData) {
  try {
    if (!csvData || csvData.trim() === '') {
      throw new Error("No CSV data provided");
    }
    
    var lines = csvData.trim().split('\n');
    if (lines.length < 2) {
      throw new Error("CSV file must have at least a header row and one data row");
    }
    
    // Parse headers (first row)
    var headerLine = lines[0];
    var headers = parseCSVLine(headerLine);
    
    if (headers.length === 0) {
      throw new Error("No headers found in CSV file");
    }
    
    // Parse preview rows (next 5 rows)
    var previewRows = [];
    for (var i = 1; i < Math.min(lines.length, 6); i++) {
      var rowData = parseCSVLine(lines[i]);
      if (rowData.length > 0) {
        previewRows.push(rowData);
      }
    }
    
    Logger.log("CSV Analysis - Headers: " + JSON.stringify(headers));
    Logger.log("CSV Analysis - Preview rows: " + previewRows.length);
    
    return {
      success: true,
      headers: headers,
      previewRows: previewRows,
      totalRows: lines.length - 1 // Exclude header
    };
    
  } catch (error) {
    Logger.log("Error analyzing CSV: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Parse a single CSV line, handling quoted fields
 * @param {string} line - CSV line to parse
 * @return {Array} - Array of field values
 */
function parseCSVLine(line) {
  var result = [];
  var current = '';
  var inQuotes = false;
  
  for (var i = 0; i < line.length; i++) {
    var char = line[i];
    
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        // Handle escaped quotes
        current += '"';
        i++; // Skip next quote
      } else {
        // Toggle quote state
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      // Field separator outside quotes
      result.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }
  
  // Add the last field
  result.push(current.trim());
  
  return result;
}

/**
 * Process transaction data with field mappings
 * @param {string} csvData - Raw CSV data
 * @param {Object} mappings - Field mappings object
 * @param {string} dateFormat - Date format string
 * @return {Object} - Processing result
 */
function processTransactionData(csvData, mappings, dateFormat) {
  try {
    Logger.log("Processing transaction data with mappings: " + JSON.stringify(mappings));
    Logger.log("Date format: " + dateFormat);
    
    var lines = csvData.trim().split('\n');
    if (lines.length < 2) {
      throw new Error("Not enough data to process");
    }
    
    var headers = parseCSVLine(lines[0]);
    var processedData = [];
    var errors = [];
    
    // Skip header row and process data
    for (var i = 1; i < lines.length; i++) {
      var rowData = parseCSVLine(lines[i]);
      
      if (rowData.length === 0 || (rowData.length === 1 && rowData[0].trim() === '')) {
        continue; // Skip empty rows
      }
      
      try {
        var transaction = processTransactionRow(headers, rowData, mappings, dateFormat);
        if (transaction) {
          processedData.push(transaction);
        }
      } catch (rowError) {
        errors.push("Row " + (i + 1) + ": " + rowError.message);
        Logger.log("Error processing row " + (i + 1) + ": " + rowError.message);
      }
    }
    
    Logger.log("Processed " + processedData.length + " transactions");
    if (errors.length > 0) {
      Logger.log("Processing errors: " + JSON.stringify(errors));
    }
    
    return {
      success: true,
      processedData: processedData,
      totalProcessed: processedData.length,
      errors: errors
    };
    
  } catch (error) {
    Logger.log("Error processing transaction data: " + error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Process a single transaction row
 * @param {Array} headers - CSV headers
 * @param {Array} rowData - Row data values
 * @param {Object} mappings - Field mappings
 * @param {string} dateFormat - Date format string
 * @return {Object} - Processed transaction object
 */
function processTransactionRow(headers, rowData, mappings, dateFormat) {
  var transaction = {};
  
  // Create reverse mapping (header -> field type)
  var fieldMappings = {};
  for (var header in mappings) {
    if (mappings[header] !== 'none') {
      fieldMappings[header] = mappings[header];
    }
  }
  
  // Process each field
  for (var i = 0; i < headers.length; i++) {
    var header = headers[i];
    var value = i < rowData.length ? rowData[i] : '';
    var fieldType = fieldMappings[header];
    
    if (fieldType && value !== '') {
      switch (fieldType) {
        case 'source':
          transaction['Source'] = value.toString().trim();
          break;
          
        case 'date':
          var dateObj = formatDate(value, dateFormat);
          transaction['Date'] = dateObj || new Date();
          break;
          
        case 'description':
          transaction['Description'] = value.toString().trim();
          break;
          
        case 'description2':
          // Handle second description field for ANZ/NZ banks
          if (transaction['Description']) {
            transaction['Description'] += ' - ' + value.toString().trim();
          } else {
            transaction['Description'] = value.toString().trim();
          }
          break;
          
        case 'amount_spent':
          var amount = parseFloat(value.toString().replace(/[^-0-9.]/g, ''));
          transaction['Amount Spent'] = isNaN(amount) ? 0 : amount;
          break;
          
        case 'currency_spent':
          transaction['Currency Spent'] = value.toString().trim() || 'AUD';
          break;
          
        case 'amount_base_aud':
          var baseAmount = parseFloat(value.toString().replace(/[^-0-9.]/g, ''));
          transaction['Amount in Base Currency: AUD'] = isNaN(baseAmount) ? 0 : baseAmount;
          break;
      }
    }
  }
  
  // Set defaults for required fields
  if (!transaction['Source']) {
    transaction['Source'] = 'Unknown';
  }
  
  if (!transaction['Description']) {
    throw new Error('Description is required but not found');
  }
  
  if (!transaction['Date']) {
    transaction['Date'] = new Date();
  }
  
  if (!transaction['Amount Spent']) {
    transaction['Amount Spent'] = 0;
  }
  
  if (!transaction['Currency Spent']) {
    transaction['Currency Spent'] = 'AUD';
  }
  
  if (!transaction['Amount in Base Currency: AUD']) {
    transaction['Amount in Base Currency: AUD'] = transaction['Amount Spent'];
  }
  
  // Category will be added later by the categorization process
  transaction['Category'] = '';
  
  return transaction;
}

/**
 * Auto-detect field types based on header names and sample data
 * @param {Array} headers - CSV headers
 * @param {Array} previewRows - Sample data rows
 * @return {Object} - Suggested field mappings
 */
function autoDetectFieldTypes(headers, previewRows) {
  var suggestions = {};
  var usedTypes = new Set();
  
  // Field detection patterns
  var patterns = {
    'date': {
      headers: /date|when|time|posted|transaction.?date/i,
      priority: 1,
      validator: function(value) {
        if (!value) return false;
        // Check if value looks like a date
        return /\d{1,4}[-\/\.]\d{1,2}[-\/\.]\d{1,4}/.test(value.toString());
      }
    },
    'amount_spent': {
      headers: /amount|debit|credit|sum|total|value|spend|cost/i,
      priority: 2,
      validator: function(value) {
        if (!value) return false;
        // Check if value looks like a number
        var cleaned = value.toString().replace(/[^-0-9.]/g, '');
        return !isNaN(parseFloat(cleaned)) && cleaned !== '';
      }
    },
    'description': {
      headers: /description|narrative|memo|details|payee|merchant|reference/i,
      priority: 3,
      validator: function(value) {
        // Any non-empty text is valid for description
        return value && value.toString().trim().length > 0;
      }
    },
    'source': {
      headers: /source|bank|account|type|card/i,
      priority: 4,
      validator: function(value) {
        return value && value.toString().trim().length > 0;
      }
    },
    'description2': {
      headers: /description.?2|reference.?2|memo.?2|details.?2/i,
      priority: 5,
      validator: function(value) {
        return value && value.toString().trim().length > 0;
      }
    },
    'currency_spent': {
      headers: /currency|curr|ccy/i,
      priority: 6,
      validator: function(value) {
        return value && /^[A-Z]{3}$/.test(value.toString().trim());
      }
    }
  };
  
  // Collect potential mappings with confidence scores
  var candidates = [];
  
  for (var i = 0; i < headers.length; i++) {
    var header = headers[i];
    
    for (var fieldType in patterns) {
      var pattern = patterns[fieldType];
      var score = 0;
      
      // Header name match
      if (pattern.headers.test(header)) {
        score += 50;
      }
      
      // Sample data validation
      if (previewRows.length > 0) {
        var validSamples = 0;
        var totalSamples = 0;
        
        for (var j = 0; j < previewRows.length; j++) {
          if (i < previewRows[j].length) {
            var value = previewRows[j][i];
            if (value && value.toString().trim() !== '') {
              totalSamples++;
              if (pattern.validator(value)) {
                validSamples++;
              }
            }
          }
        }
        
        if (totalSamples > 0) {
          score += (validSamples / totalSamples) * 30;
        }
      }
      
      if (score > 20) { // Minimum threshold
        candidates.push({
          header: header,
          fieldType: fieldType,
          score: score,
          priority: pattern.priority
        });
      }
    }
  }
  
  // Sort by score (descending) then by priority (ascending)
  candidates.sort(function(a, b) {
    if (a.score !== b.score) {
      return b.score - a.score;
    }
    return a.priority - b.priority;
  });
  
  // Assign unique mappings
  for (var k = 0; k < candidates.length; k++) {
    var candidate = candidates[k];
    
    if (!usedTypes.has(candidate.fieldType)) {
      suggestions[candidate.header] = candidate.fieldType;
      usedTypes.add(candidate.fieldType);
    }
  }
  
  Logger.log("Auto-detected field mappings: " + JSON.stringify(suggestions));
  return suggestions;
}

/**
 * Validate field mappings to ensure no duplicates
 * @param {Object} mappings - Field mappings to validate
 * @return {Object} - Validation result
 */
function validateFieldMappings(mappings) {
  var usedTypes = {};
  var errors = [];
  
  for (var header in mappings) {
    var fieldType = mappings[header];
    
    if (fieldType !== 'none') {
      if (usedTypes[fieldType]) {
        errors.push("Field type '" + fieldType + "' is mapped to multiple columns: '" + 
                   usedTypes[fieldType] + "' and '" + header + "'");
      } else {
        usedTypes[fieldType] = header;
      }
    }
  }
  
  // Check for required fields
  var requiredFields = ['description'];
  for (var i = 0; i < requiredFields.length; i++) {
    var required = requiredFields[i];
    if (!usedTypes[required]) {
      errors.push("Required field '" + required + "' is not mapped to any column");
    }
  }
  
  return {
    valid: errors.length === 0,
    errors: errors,
    mappedFields: Object.keys(usedTypes)
  };
} 