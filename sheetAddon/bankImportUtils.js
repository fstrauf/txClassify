/**
 * Bank Import Utilities for Google Sheets Add-on
 * Handles CSV parsing, column mapping, and data transformation for bank transactions
 */

// Common bank column patterns for auto-detection
const BANK_COLUMN_PATTERNS = {
  // Date patterns
  date: [
    'date', 'transaction date', 'created', 'processed date', 'value date',
    'booking date', 'posted date', 'effective date'
  ],
  
  // Amount patterns  
  amount: [
    'amount', 'value', 'transaction amount', 'source amount', 'debit amount',
    'credit amount', 'net amount', 'total', 'sum'
  ],
  
  // Description patterns
  description: [
    'description', 'narrative', 'memo', 'payee', 'merchant', 'reference',
    'details', 'particulars', 'transaction details', 'tran details',
    'other party name', 'target name', 'recipient'
  ],
  
  // Secondary description patterns (for banks like ANZ that split descriptions)
  description2: [
    'analysis code', 'code', 'transaction code', 'reference number',
    'additional details', 'memo field', 'extended description'
  ],
  
  // Account patterns
  account: [
    'account', 'account number', 'account name', 'source account',
    'from account', 'account id'
  ],
  
  // Category patterns (if already present)
  category: [
    'category', 'classification', 'type', 'transaction type',
    'category code', 'expense type'
  ]
};

// Common date formats used by different banks
const DATE_FORMATS = [
  { 
    pattern: /^\d{4}-\d{2}-\d{2}/, 
    format: 'yyyy-MM-dd',
    example: '2023-12-31' 
  },
  { 
    pattern: /^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}/, 
    format: 'yyyy-MM-dd HH:mm:ss',
    example: '2023-12-31 14:30:00' 
  },
  { 
    pattern: /^\d{2}\/\d{2}\/\d{4}/, 
    format: 'MM/dd/yyyy',
    example: '12/31/2023' 
  },
  { 
    pattern: /^\d{1,2}\/\d{1,2}\/\d{4}/, 
    format: 'dd/MM/yyyy',
    example: '31/12/2023' 
  },
  { 
    pattern: /^\d{2}\.\d{2}\.\d{4}/, 
    format: 'dd.MM.yyyy',
    example: '31.12.2023' 
  },
  { 
    pattern: /^\d{4}\/\d{2}\/\d{2}/, 
    format: 'yyyy/MM/dd',
    example: '2023/12/31' 
  }
];

/**
 * Enhanced column auto-detection with confidence scoring
 */
function autoDetectColumnMapping(headers, previewRows) {
  const mappings = {};
  const confidence = {};
  
  headers.forEach((header, index) => {
    const lowerHeader = header.toLowerCase().trim();
    let bestMatch = 'none';
    let bestScore = 0;
    
    // Check each field type
    Object.keys(BANK_COLUMN_PATTERNS).forEach(fieldType => {
      const patterns = BANK_COLUMN_PATTERNS[fieldType];
      let score = 0;
      
      // Exact match gets highest score
      if (patterns.includes(lowerHeader)) {
        score = 100;
      } else {
        // Partial match scoring
        patterns.forEach(pattern => {
          if (lowerHeader.includes(pattern)) {
            score = Math.max(score, 80);
          } else if (pattern.includes(lowerHeader)) {
            score = Math.max(score, 60);
          }
        });
      }
      
      // Additional scoring based on data content
      if (previewRows.length > 0 && score > 0) {
        const sampleValue = previewRows[0][index];
        
        if (fieldType === 'date') {
          if (isValidDate(sampleValue)) {
            score += 10;
          }
        } else if (fieldType === 'amount') {
          if (isValidAmount(sampleValue)) {
            score += 10;
          }
        }
      }
      
      if (score > bestScore) {
        bestScore = score;
        bestMatch = fieldType;
      }
    });
    
    mappings[header] = bestMatch;
    confidence[header] = bestScore;
  });
  
  // Ensure uniqueness for key fields (keep highest confidence)
  const uniqueFields = ['date', 'amount'];
  uniqueFields.forEach(fieldType => {
    const candidates = Object.keys(mappings).filter(h => mappings[h] === fieldType);
    if (candidates.length > 1) {
      // Sort by confidence and keep only the best
      candidates.sort((a, b) => confidence[b] - confidence[a]);
      candidates.slice(1).forEach(header => {
        mappings[header] = 'none';
      });
    }
  });
  
  return { mappings, confidence };
}

/**
 * Validate if a string represents a valid date
 */
function isValidDate(dateString) {
  if (!dateString || typeof dateString !== 'string') return false;
  
  const date = new Date(dateString.trim());
  return date instanceof Date && !isNaN(date.getTime());
}

/**
 * Validate if a string represents a valid amount
 */
function isValidAmount(amountString) {
  if (!amountString) return false;
  
  const cleaned = amountString.toString().replace(/[,$\s]/g, '');
  const number = parseFloat(cleaned);
  return !isNaN(number) && isFinite(number);
}

/**
 * Detect date format from sample data
 */
function detectDateFormat(sampleDates) {
  for (const sample of sampleDates) {
    if (!sample) continue;
    
    const dateStr = sample.toString().trim();
    
    for (const format of DATE_FORMATS) {
      if (format.pattern.test(dateStr)) {
        return format.format;
      }
    }
  }
  
  return 'yyyy-MM-dd'; // Default format
}

/**
 * Enhanced CSV parsing with better delimiter detection
 */
function parseCSVContent(csvContent) {
  const lines = csvContent.split('\n').filter(line => line.trim().length > 0);
  
  if (lines.length < 2) {
    throw new Error('CSV file must contain at least a header row and one data row');
  }
  
  // Detect delimiter with more sophisticated logic
  const delimiter = detectDelimiter(lines[0]);
  
  // Parse with detected delimiter
  const headers = parseCSVLine(lines[0], delimiter);
  const rows = [];
  
  for (let i = 1; i < lines.length; i++) {
    try {
      const row = parseCSVLine(lines[i], delimiter);
      if (row.length === headers.length) {
        rows.push(row);
      } else {
        console.warn(`Row ${i + 1} has ${row.length} columns but expected ${headers.length}, skipping`);
      }
    } catch (error) {
      console.warn(`Error parsing row ${i + 1}: ${error.message}`);
    }
  }
  
  return {
    headers: headers,
    rows: rows,
    delimiter: delimiter
  };
}

/**
 * Detect CSV delimiter with better accuracy
 */
function detectDelimiter(headerLine) {
  const delimiters = [',', ';', '\t', '|'];
  let bestDelimiter = ',';
  let maxColumns = 0;
  
  delimiters.forEach(delimiter => {
    try {
      const columns = parseCSVLine(headerLine, delimiter);
      if (columns.length > maxColumns && columns.length > 1) {
        maxColumns = columns.length;
        bestDelimiter = delimiter;
      }
    } catch (error) {
      // Skip invalid delimiter
    }
  });
  
  return bestDelimiter;
}

/**
 * Parse a single CSV line with proper quote handling
 */
function parseCSVLine(line, delimiter) {
  const result = [];
  let current = '';
  let inQuotes = false;
  
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        // Escaped quote
        current += '"';
        i++; // Skip next quote
      } else {
        // Start or end quotes
        inQuotes = !inQuotes;
      }
    } else if (char === delimiter && !inQuotes) {
      // End of field
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
 * Bank-specific preprocessing for common formats
 */
function preprocessBankData(headers, rows, bankHint) {
  // Bank-specific preprocessing logic
  if (bankHint && bankHint.toLowerCase().includes('anz')) {
    return preprocessANZData(headers, rows);
  } else if (bankHint && bankHint.toLowerCase().includes('asb')) {
    return preprocessASBData(headers, rows);
  } else if (bankHint && bankHint.toLowerCase().includes('westpac')) {
    return preprocessWestpacData(headers, rows);
  }
  
  return { headers, rows }; // No specific preprocessing
}

/**
 * ANZ-specific preprocessing (combines multiple description fields)
 */
function preprocessANZData(headers, rows) {
  // ANZ often has 'Payee', 'Particulars', 'Code', 'Reference' columns
  // We want to auto-map these intelligently
  
  const processedHeaders = [...headers];
  const processedRows = rows.map(row => [...row]);
  
  // Look for ANZ-specific patterns and mark them appropriately
  headers.forEach((header, index) => {
    const lowerHeader = header.toLowerCase();
    if (lowerHeader.includes('payee') || lowerHeader.includes('particulars')) {
      // These should be primary description
    } else if (lowerHeader.includes('code') || lowerHeader.includes('analysis')) {
      // These should be secondary description
    }
  });
  
  return { headers: processedHeaders, rows: processedRows };
}

/**
 * ASB-specific preprocessing
 */
function preprocessASBData(headers, rows) {
  // ASB-specific logic here
  return { headers, rows };
}

/**
 * Westpac-specific preprocessing  
 */
function preprocessWestpacData(headers, rows) {
  // Westpac-specific logic here
  return { headers, rows };
}

/**
 * Generate standard transaction object from mapped data
 */
function createStandardTransaction(rowData, mappings, options = {}) {
  const transaction = {
    date: '',
    amount: 0,
    description: '',
    account: options.defaultAccount || 'Imported',
    category: '',
    reference: '',
    type: 'Unknown'
  };
  
  // Map fields based on configuration
  Object.keys(mappings).forEach(originalHeader => {
    const fieldType = mappings[originalHeader];
    const value = rowData[originalHeader];
    
    if (fieldType === 'date') {
      transaction.date = formatDateForSheets(value, options.dateFormat);
    } else if (fieldType === 'amount') {
      transaction.amount = parseAmount(value);
    } else if (fieldType === 'description') {
      transaction.description = value || '';
    } else if (fieldType === 'description2') {
      // Append secondary description
      const separator = transaction.description ? ' - ' : '';
      transaction.description += separator + (value || '');
    } else if (fieldType === 'account') {
      transaction.account = value || transaction.account;
    } else if (fieldType === 'category') {
      transaction.category = value || '';
    } else if (fieldType === 'reference') {
      transaction.reference = value || '';
    }
  });
  
  // Determine transaction type
  transaction.type = transaction.amount < 0 ? 'Debit' : 'Credit';
  
  return transaction;
}

/**
 * Parse amount string to number
 */
function parseAmount(amountStr) {
  if (!amountStr) return 0;
  
  const cleaned = amountStr.toString()
    .replace(/[,$\s]/g, '')  // Remove commas, dollar signs, spaces
    .replace(/[()]/g, '')    // Remove parentheses
    .trim();
  
  const number = parseFloat(cleaned);
  return isNaN(number) ? 0 : number;
}

/**
 * Format date for Google Sheets
 */
function formatDateForSheets(dateStr, dateFormat) {
  if (!dateStr) return new Date();
  
  try {
    const date = parseDateString(dateStr, dateFormat);
    return date instanceof Date && !isNaN(date.getTime()) ? date : new Date();
  } catch (error) {
    console.warn('Date parsing failed:', dateStr, error.message);
    return new Date();
  }
}

/**
 * Parse date string according to format
 */
function parseDateString(dateStr, format) {
  const cleanDate = dateStr.trim();
  
  if (format === 'MM/dd/yyyy') {
    const parts = cleanDate.split('/');
    if (parts.length === 3) {
      return new Date(parts[2], parts[0] - 1, parts[1]);
    }
  } else if (format === 'dd/MM/yyyy') {
    const parts = cleanDate.split('/');
    if (parts.length === 3) {
      return new Date(parts[2], parts[1] - 1, parts[0]);
    }
  } else if (format === 'dd.MM.yyyy') {
    const parts = cleanDate.split('.');
    if (parts.length === 3) {
      return new Date(parts[2], parts[1] - 1, parts[0]);
    }
  } else if (format === 'yyyy/MM/dd') {
    const parts = cleanDate.split('/');
    if (parts.length === 3) {
      return new Date(parts[0], parts[1] - 1, parts[2]);
    }
  } else {
    // Default: try to parse as-is or yyyy-MM-dd
    return new Date(cleanDate);
  }
  
  return new Date(cleanDate);
} 