// Enhanced bulk test with multiple configuration comparisons

const fs = require('fs');
const path = require('path');
const { loadCategorizationData } = require('./test-utils');
const { logInfo, logError, logDebug } = require('./logger');

const TEST_API_KEY = process.env.TEST_API_KEY || "test-key-123";

const runBulkCleanAndGroupComparison = async (config) => {
  const testStartTime = new Date();
  const timestamp = testStartTime.toISOString().replace(/[:.]/g, '-').split('T');
  const logFileName = `bulk_comparison_${timestamp[0]}_${timestamp[1].split('.')[0]}.log`;
  const logFilePath = path.join(__dirname, '..', 'logs', logFileName);
  
  // Create logs directory if it doesn't exist
  const logsDir = path.join(__dirname, '..', 'logs');
  if (!fs.existsSync(logsDir)) {
    fs.mkdirSync(logsDir, { recursive: true });
  }
  
  // File logging function
  const logToFile = (message) => {
    const timestamp = new Date().toISOString();
    fs.appendFileSync(logFilePath, `[${timestamp}] ${message}\n`);
  };
  
  logInfo("\n=== Starting Bulk Clean and Group Comparison Test ===\n");
  logInfo(`📁 Detailed comparison logs will be saved to: ${logFilePath}`);
  const BULK_CSV_FILE = "ANZ Transactions Nov 2024 to May 2025.csv";

  try {
    logToFile("=== BULK CLEAN AND GROUP COMPARISON TEST STARTED ===");
    logToFile(`Test file: ${BULK_CSV_FILE}`);
    logToFile(`Start time: ${testStartTime.toISOString()}`);

    // Load data once
    logInfo(`1. Loading transaction data from ${BULK_CSV_FILE}...`);
    const allTransactions = await loadCategorizationData(BULK_CSV_FILE);
    if (!allTransactions || allTransactions.length === 0) {
      const errorMsg = `No transactions loaded from ${BULK_CSV_FILE}. Ensure the file exists in 'tests/test_data/' and is readable.`;
      logError(errorMsg);
      logToFile(`ERROR: ${errorMsg}`);
      return false;
    }

    // Prepare descriptions
    const descriptionsToClean = allTransactions
      .map((t) => {
        let combinedDesc = t.description || "";
        if (t.code && String(t.code).trim() !== "") {
          combinedDesc = combinedDesc ? `${combinedDesc} ${t.code}` : t.code;
        }
        return combinedDesc.trim();
      })
      .filter((d) => d);

    const originalUnique = new Set(descriptionsToClean).size;
    logInfo(`   Loaded ${allTransactions.length} transactions with ${originalUnique} unique descriptions.`);
    logToFile(`Loaded ${allTransactions.length} transactions with ${originalUnique} unique descriptions`);

    // Test configurations to compare
    const testConfigs = [
      {
        name: "Current (similarity 0.8)",
        use_embedding_grouping: true,
        embedding_clustering_method: "similarity",
        embedding_similarity_threshold: 0.8
      },
      {
        name: "Lower threshold (similarity 0.7)",
        use_embedding_grouping: true,
        embedding_clustering_method: "similarity", 
        embedding_similarity_threshold: 0.7
      },
      {
        name: "Lowest threshold (similarity 0.6)",
        use_embedding_grouping: true,
        embedding_clustering_method: "similarity",
        embedding_similarity_threshold: 0.6
      },
      {
        name: "HDBSCAN clustering",
        use_embedding_grouping: true,
        embedding_clustering_method: "hdbscan",
        embedding_similarity_threshold: 0.8
      },
      {
        name: "Basic text cleaning (no embeddings)",
        use_embedding_grouping: false
      }
    ];

    const results = [];

    for (let configIndex = 0; configIndex < testConfigs.length; configIndex++) {
      const testConfig = testConfigs[configIndex];
      const configStartTime = Date.now();
      
      console.log(`\n🧪 Testing Configuration ${configIndex + 1}/${testConfigs.length}: ${testConfig.name}`);
      logToFile(`\n=== TESTING CONFIG: ${testConfig.name} ===`);
      
      const batchSize = 50;
      const allCleanedDescriptions = [];
      const allGroups = {};
      const batchTimes = [];

      // Process in batches
      for (let i = 0; i < descriptionsToClean.length; i += batchSize) {
        const batchStartTime = Date.now();
        const batch = descriptionsToClean.slice(i, i + batchSize);
        const batchNumber = Math.floor(i / batchSize) + 1;
        const totalBatches = Math.ceil(descriptionsToClean.length / batchSize);

        const requestBody = {
          descriptions: batch,
          ...testConfig
        };

        try {
          const response = await fetch(`${config.serviceUrl}/clean_text`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "X-API-Key": config.apiKey || TEST_API_KEY,
              Accept: "application/json",
            },
            body: JSON.stringify(requestBody),
          });

          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
          }

          const result = await response.json();
          const batchTime = Date.now() - batchStartTime;
          batchTimes.push(batchTime);

          if (result.cleaned_descriptions && Array.isArray(result.cleaned_descriptions)) {
            allCleanedDescriptions.push(...result.cleaned_descriptions);
            
            if (result.groups && typeof result.groups === 'object') {
              Object.assign(allGroups, result.groups);
            }
          }
          
          // Show progress for longer tests
          if (batchNumber % 5 === 0 || batchNumber === totalBatches) {
            process.stdout.write(`   Batch ${batchNumber}/${totalBatches} (${batchTime}ms)\r`);
          }
        } catch (error) {
          logToFile(`ERROR in batch ${batchNumber}: ${error.message}`);
          // Fallback to original for this batch
          allCleanedDescriptions.push(...batch);
        }
      }

      console.log(`   Completed in ${((Date.now() - configStartTime) / 1000).toFixed(1)}s`);

      // Analyze results
      const cleanedUnique = new Set(allCleanedDescriptions).size;
      const transformationCount = descriptionsToClean.filter((orig, idx) => orig !== allCleanedDescriptions[idx]).length;
      
      // Group by cleaned descriptions
      const groups = {};
      allTransactions.forEach((transaction, index) => {
        const cleanedDetail = allCleanedDescriptions[index] || descriptionsToClean[index];
        if (!groups[cleanedDetail]) {
          groups[cleanedDetail] = { count: 0, variations: new Set() };
        }
        groups[cleanedDetail].count++;
        groups[cleanedDetail].variations.add(descriptionsToClean[index]);
      });

      const sortedGroups = Object.entries(groups).sort(([, a], [, b]) => b.count - a.count);
      const totalGroups = sortedGroups.length;
      const largeGroups = sortedGroups.filter(([, data]) => data.count >= 10).length;
      const singletons = sortedGroups.filter(([, data]) => data.count === 1).length;
      const avgBatchTime = batchTimes.length > 0 ? batchTimes.reduce((a, b) => a + b, 0) / batchTimes.length : 0;

      const configResult = {
        name: testConfig.name,
        totalTime: Date.now() - configStartTime,
        avgBatchTime: avgBatchTime,
        originalUnique: originalUnique,
        cleanedUnique: cleanedUnique,
        reductionPercent: ((originalUnique - cleanedUnique) / originalUnique * 100),
        transformationPercent: (transformationCount / descriptionsToClean.length * 100),
        totalGroups: totalGroups,
        groupReductionPercent: ((originalUnique - totalGroups) / originalUnique * 100),
        largeGroups: largeGroups,
        singletons: singletons,
        singletonPercent: (singletons / totalGroups * 100),
        avgPerGroup: (descriptionsToClean.length / totalGroups),
        largestGroup: Math.max(...sortedGroups.map(([, data]) => data.count)),
        embeddingGroupsFound: Object.keys(allGroups).length,
        top5Groups: sortedGroups.slice(0, 5).map(([name, data]) => ({
          name,
          count: data.count,
          variations: Array.from(data.variations).slice(0, 3)
        }))
      };

      results.push(configResult);
      
      // Log detailed results
      logToFile(`RESULTS FOR ${testConfig.name}:`);
      logToFile(`- Total time: ${configResult.totalTime}ms`);
      logToFile(`- Avg batch time: ${configResult.avgBatchTime.toFixed(0)}ms`);
      logToFile(`- Original unique: ${configResult.originalUnique}`);
      logToFile(`- Cleaned unique: ${configResult.cleanedUnique} (${configResult.reductionPercent.toFixed(2)}% reduction)`);
      logToFile(`- Total groups: ${configResult.totalGroups} (${configResult.groupReductionPercent.toFixed(2)}% reduction from original)`);
      logToFile(`- Large groups (10+): ${configResult.largeGroups}`);
      logToFile(`- Singletons: ${configResult.singletons} (${configResult.singletonPercent.toFixed(1)}%)`);
      logToFile(`- Largest group: ${configResult.largestGroup} transactions`);
      logToFile(`- Embedding groups found: ${configResult.embeddingGroupsFound}`);
      logToFile(`- Top 5 groups: ${JSON.stringify(configResult.top5Groups, null, 2)}`);
    }

    // Comparison analysis
    console.log(`\n📊 CONFIGURATION COMPARISON RESULTS:`);
    console.log(`${'Configuration'.padEnd(25)} | ${'Groups'.padEnd(6)} | ${'Large'.padEnd(5)} | ${'Reduce%'.padEnd(7)} | ${'Time'.padEnd(6)}`);
    console.log(`${'-'.repeat(25)} | ${'-'.repeat(6)} | ${'-'.repeat(5)} | ${'-'.repeat(7)} | ${'-'.repeat(6)}`);
    
    results.forEach(r => {
      console.log(`${r.name.padEnd(25)} | ${r.totalGroups.toString().padEnd(6)} | ${r.largeGroups.toString().padEnd(5)} | ${r.groupReductionPercent.toFixed(1).padEnd(7)} | ${(r.totalTime/1000).toFixed(1).padEnd(6)}`);
    });

    // Find best configuration
    const bestByLargeGroups = results.reduce((best, current) => 
      current.largeGroups > best.largeGroups ? current : best
    );
    
    const bestByReduction = results.reduce((best, current) => 
      current.groupReductionPercent > best.groupReductionPercent ? current : best
    );

    console.log(`\n🏆 RECOMMENDATIONS:`);
    console.log(`   Best for large groups: ${bestByLargeGroups.name} (${bestByLargeGroups.largeGroups} large groups)`);
    console.log(`   Best for reduction: ${bestByReduction.name} (${bestByReduction.groupReductionPercent.toFixed(1)}% reduction)`);
    
    if (bestByLargeGroups.largeGroups < 15) {
      console.log(`\n⚠️  INSIGHT: Even the best configuration only created ${bestByLargeGroups.largeGroups} large groups.`);
      console.log(`   This suggests the data might have inherently high diversity, or text cleaning`);
      console.log(`   is removing important distinguishing information.`);
      console.log(`\n💡 NEXT STEPS:`);
      console.log(`   1. Try reducing text cleaning aggressiveness`);
      console.log(`   2. Examine the top singleton groups to understand why they're not clustering`);
      console.log(`   3. Consider domain-specific embedding models`);
    } else {
      console.log(`\n✅ SUCCESS: Found optimal configuration with ${bestByLargeGroups.largeGroups} large groups!`);
    }

    logToFile(`\nCOMPARISON SUMMARY:`);
    logToFile(`Best for large groups: ${bestByLargeGroups.name} (${bestByLargeGroups.largeGroups} large groups)`);
    logToFile(`Best for reduction: ${bestByReduction.name} (${bestByReduction.groupReductionPercent.toFixed(1)}% reduction)`);
    logToFile(`Full results: ${JSON.stringify(results, null, 2)}`);

    const testEndTime = new Date();
    const totalTestTime = testEndTime - testStartTime;
    
    console.log(`\n✅ COMPARISON TEST COMPLETED`);
    console.log(`📁 Full analysis saved to: ${logFileName}`);
    console.log(`⏱️  Total time: ${(totalTestTime/1000).toFixed(1)}s`);
    
    logToFile(`=== COMPARISON TEST COMPLETED ===`);
    logToFile(`Total test time: ${totalTestTime}ms`);
    
    return true;

  } catch (error) {
    const errorMsg = `Bulk Comparison Test failed: ${error.message}`;
    logError(`\n${errorMsg}`);
    logDebug(error.stack);
    logToFile(`ERROR: ${errorMsg}`);
    logToFile(`ERROR STACK: ${error.stack}`);
    return false;
  }
};

module.exports = { runBulkCleanAndGroupComparison };
