#!/usr/bin/env node
// Standalone script to run bulk comparison tests

const path = require("path");
require("dotenv").config({ path: path.resolve(__dirname, "../.env") });

const { runBulkCleanAndGroupComparison } = require('../debug/test-bulk-comparison');

// Logging setup (simplified from main test file)
const LOG_LEVELS = {
  ERROR: 0,
  INFO: 1,
  DEBUG: 2,
  TRACE: 3,
};

let CURRENT_LOG_LEVEL = LOG_LEVELS.INFO;

if (process.argv.includes("--verbose")) {
  CURRENT_LOG_LEVEL = LOG_LEVELS.INFO;
} else if (process.argv.includes("--debug")) {
  CURRENT_LOG_LEVEL = LOG_LEVELS.DEBUG;
} else if (process.argv.includes("--trace")) {
  CURRENT_LOG_LEVEL = LOG_LEVELS.TRACE;
}

const log = (message, level = LOG_LEVELS.INFO) => {
  if (level <= CURRENT_LOG_LEVEL) {
    const timestamp = new Date().toISOString();
    const prefix =
      level === LOG_LEVELS.ERROR
        ? "[ERROR] "
        : level === LOG_LEVELS.DEBUG
        ? "[DEBUG] "
        : level === LOG_LEVELS.TRACE
        ? "[TRACE] "
        : "";
    console.log(`[${timestamp}] ${prefix}${message}`);
  }
};

const logError = (message) => log(message, LOG_LEVELS.ERROR);
const logInfo = (message) => log(message, LOG_LEVELS.INFO);
const logDebug = (message) => log(message, LOG_LEVELS.DEBUG);
const logTrace = (message) => log(message, LOG_LEVELS.TRACE);

// Make logging functions globally available
global.logError = logError;
global.logInfo = logInfo;
global.logDebug = logDebug;
global.logTrace = logTrace;

const main = async () => {
  console.log("\n🧪 BULK CLUSTERING COMPARISON TEST");
  console.log("==================================");
  
  // Configuration
  const config = {
    serviceUrl: process.env.TEST_TARGET_API_URL || "http://localhost:5001",
    apiKey: process.env.TEST_API_KEY || "test-key-123",
  };

  logInfo(`Using API URL: ${config.serviceUrl}`);
  logInfo(`Using API Key: ${config.apiKey ? '[SET]' : '[NOT SET]'}`);

  try {
    const success = await runBulkCleanAndGroupComparison(config);
    
    if (success) {
      console.log("\n✅ Comparison test completed successfully!");
      process.exit(0);
    } else {
      console.log("\n❌ Comparison test failed!");
      process.exit(1);
    }
  } catch (error) {
    console.error(`\n❌ Error running comparison test: ${error.message}`);
    console.error(error.stack);
    process.exit(1);
  }
};

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\n\n⚠️  Test interrupted by user');
  process.exit(1);
});

process.on('SIGTERM', () => {
  console.log('\n\n⚠️  Test terminated');
  process.exit(1);
});

// Run the test
if (require.main === module) {
  main().catch((error) => {
    console.error("Unhandled error:", error);
    process.exit(1);
  });
}
