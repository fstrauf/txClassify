// Test script for bulk embedding-based grouping functionality
const path = require("path");
require("dotenv").config({ path: path.resolve(__dirname, "../.env") });

const TEST_API_KEY = process.env.TEST_API_KEY;
const API_URL = process.env.TARGET_API_URL || "http://localhost";
const fs = require('fs');
const csv = require('csv-parser');

const log = (message) => {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${message}`);
};

// Load CSV data
const loadCategorizationData = (filename) => {
  return new Promise((resolve, reject) => {
    const results = [];
    const filePath = path.join(__dirname, "test_data", filename);
    
    if (!fs.existsSync(filePath)) {
      reject(new Error(`File not found: ${filePath}`));
      return;
    }
    
    fs.createReadStream(filePath)
      .pipe(csv({ 
        mapHeaders: ({ header, index }) => header.trim() 
      }))
      .on('data', (data) => {
        // Convert to the expected format
        const processed = {
          description: data['Details'] || '',
          amount: parseFloat(data['Amount']) || 0,
          code: data['Code'] || '',
          money_in: (data['Money Out'] === '' && data['Money In'] !== '') || false
        };
        results.push(processed);
      })
      .on('end', () => {
        log(`Loaded ${results.length} records from ${filename}`);
        resolve(results);
      })
      .on('error', reject);
  });
};

const testBulkEmbeddingGrouping = async () => {
  log("=== Testing Bulk Embedding-Based Grouping ===");
  
  const BULK_CSV_FILE = "ANZ Transactions Nov 2024 to May 2025.csv";

  try {
    log(`1. Loading transaction data from ${BULK_CSV_FILE}...`);
    const allTransactions = await loadCategorizationData(BULK_CSV_FILE);
    if (!allTransactions || allTransactions.length === 0) {
      throw new Error(`No transactions loaded from ${BULK_CSV_FILE}`);
    }
    log(`   Loaded ${allTransactions.length} transactions.`);

    // Combine Details (as t.description) and Code for cleaning
    const descriptionsToClean = allTransactions
      .map((t) => {
        let combinedDesc = t.description || "";
        if (t.code && String(t.code).trim() !== "") {
          combinedDesc = combinedDesc ? `${combinedDesc} ${t.code}` : t.code;
        }
        return combinedDesc.trim();
      })
      .filter((d) => d);

    log(`   Prepared ${descriptionsToClean.length} combined descriptions for cleaning.`);

    // Take a sample for testing (to avoid long processing times)
    const sampleSize = 100;
    const sampleDescriptions = descriptionsToClean.slice(0, sampleSize);
    log(`   Using sample of ${sampleSize} descriptions for testing.`);

    log("2. Testing regular cleaning without grouping...");
    const regularResponse = await fetch(`${API_URL}/clean_text`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": TEST_API_KEY,
      },
      body: JSON.stringify({ 
        descriptions: sampleDescriptions 
      }),
    });

    if (!regularResponse.ok) {
      throw new Error(`Regular cleaning failed: ${regularResponse.status}`);
    }

    const regularResult = await regularResponse.json();
    
    // Count unique cleaned descriptions
    const uniqueRegular = new Set(regularResult.cleaned_descriptions);
    log(`   Regular cleaning: ${sampleSize} descriptions -> ${uniqueRegular.size} unique cleaned descriptions`);

    log("3. Testing embedding-based grouping...");
    const embeddingResponse = await fetch(`${API_URL}/clean_text`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": TEST_API_KEY,
      },
      body: JSON.stringify({ 
        descriptions: sampleDescriptions,
        use_embedding_grouping: true,
        embedding_clustering_method: "similarity",
        embedding_similarity_threshold: 0.8
      }),
    });

    if (!embeddingResponse.ok) {
      const errorText = await embeddingResponse.text();
      throw new Error(`Embedding grouping failed: ${embeddingResponse.status} - ${errorText}`);
    }

    const embeddingResult = await embeddingResponse.json();
    
    log("4. Analyzing grouping results...");
    if (embeddingResult.groups) {
      const groupCount = Object.keys(embeddingResult.groups).length;
      log(`   Embedding-based grouping: ${sampleSize} descriptions -> ${groupCount} groups`);
      
      // Show top groups by size
      const groupSizes = Object.entries(embeddingResult.groups)
        .map(([name, members]) => ({ name, size: members.length }))
        .sort((a, b) => b.size - a.size);
      
      log(`   Top groups by size:`);
      groupSizes.slice(0, 10).forEach((group, i) => {
        log(`     ${i + 1}. "${group.name}": ${group.size} members`);
      });
      
      // Show some example groupings
      log(`   Example groupings:`);
      const exampleGroups = Object.entries(embeddingResult.groups)
        .filter(([name, members]) => members.length > 1)
        .slice(0, 5);
      
      exampleGroups.forEach(([groupName, members]) => {
        log(`     Group "${groupName}" (${members.length} members):`);
        members.slice(0, 3).forEach(member => {
          log(`       - "${member}"`);
        });
        if (members.length > 3) {
          log(`       ... and ${members.length - 3} more`);
        }
      });
      
      // Calculate grouping efficiency
      const totalMembers = Object.values(embeddingResult.groups).reduce((sum, members) => sum + members.length, 0);
      const groupingEfficiency = ((sampleSize - groupCount) / sampleSize * 100).toFixed(1);
      log(`   Grouping efficiency: ${groupingEfficiency}% reduction (${sampleSize} -> ${groupCount})`);
      
    } else {
      log("   No groups returned in response");
    }

    log("5. Comparing with different similarity thresholds...");
    const thresholds = [0.7, 0.8, 0.9];
    
    for (const threshold of thresholds) {
      const thresholdResponse = await fetch(`${API_URL}/clean_text`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": TEST_API_KEY,
        },
        body: JSON.stringify({ 
          descriptions: sampleDescriptions.slice(0, 50), // Smaller sample for speed
          use_embedding_grouping: true,
          embedding_clustering_method: "similarity",
          embedding_similarity_threshold: threshold
        }),
      });

      if (thresholdResponse.ok) {
        const thresholdResult = await thresholdResponse.json();
        if (thresholdResult.groups) {
          const groupCount = Object.keys(thresholdResult.groups).length;
          const reduction = ((50 - groupCount) / 50 * 100).toFixed(1);
          log(`     Threshold ${threshold}: 50 descriptions -> ${groupCount} groups (${reduction}% reduction)`);
        }
      }
    }

    log("\n=== Bulk Embedding-Based Grouping Test Completed Successfully ===");
    return true;

  } catch (error) {
    log(`Test failed: ${error.message}`);
    console.error(error);
    return false;
  }
};

// Run the test
testBulkEmbeddingGrouping().then(success => {
  process.exit(success ? 0 : 1);
});
