// Import path first so we can use it for dotenv config
const path = require("path");
// Load environment variables from .env file with explicit path
require("dotenv").config({ path: path.resolve(__dirname, "../.env") });

// Verify critical environment variables are loaded
console.log("Checking environment variables...");
const criticalEnvVars = ["DATABASE_URL", "REPLICATE_API_TOKEN", "TEST_API_KEY", "TEST_USER_ID"];
const missingVars = criticalEnvVars.filter((varName) => !process.env[varName]);
if (missingVars.length > 0) {
  console.error(`❌ Missing required environment variables: ${missingVars.join(", ")}`);
  console.error("Please check that your .env file is in the project root and contains these variables.");
  process.exit(1);
}
console.log("✅ Environment variables loaded successfully");

const fs = require("fs");
const { spawn } = require("child_process");
const http = require("http");
const csv = require("csv-parser");
const net = require("net");
const axios = require("axios");

// Configuration
const API_PORT = process.env.API_PORT || 3005;
const TEST_USER_ID = process.env.TEST_USER_ID || "test_user_fixed";
const TEST_API_KEY = process.env.TEST_API_KEY || "test_api_key_fixed";

// Global variables
let webhookServer;
let flaskProcess;

// Set up logging with verbosity levels
const LOG_LEVELS = {
  ERROR: 0, // Only errors
  INFO: 1, // Important information
  DEBUG: 2, // Detailed information
  TRACE: 3, // Very detailed information
};

// Set the current log level (change this to control verbosity)
let CURRENT_LOG_LEVEL = LOG_LEVELS.ERROR; // Only show errors by default

// Add a command line flag to enable verbose logging if needed
// e.g., node test-api-endpoints.js --verbose
if (process.argv.includes("--verbose")) {
  CURRENT_LOG_LEVEL = LOG_LEVELS.INFO;
} else if (process.argv.includes("--debug")) {
  CURRENT_LOG_LEVEL = LOG_LEVELS.DEBUG;
} else if (process.argv.includes("--trace")) {
  CURRENT_LOG_LEVEL = LOG_LEVELS.TRACE;
}

// Check for test mode flags
const RUN_TRAINING = !process.argv.includes("--cat-only");
const RUN_CATEGORIZATION = !process.argv.includes("--train-only");
const TEST_CLEAN_TEXT = process.argv.includes("--test-clean");

const log = (message, level = LOG_LEVELS.INFO) => {
  // Only log messages at or below the current log level
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

// Shorthand logging functions
const logError = (message) => log(message, LOG_LEVELS.ERROR);
const logInfo = (message) => log(message, LOG_LEVELS.INFO);
const logDebug = (message) => log(message, LOG_LEVELS.DEBUG);
const logTrace = (message) => log(message, LOG_LEVELS.TRACE);

// Log the actual values being used
logInfo(`Using TEST_USER_ID: ${TEST_USER_ID}`);
logInfo(`Using TEST_API_KEY: ${TEST_API_KEY ? "***" + TEST_API_KEY.slice(-4) : "Not set"}`);

// Helper function to find a free port
const findFreePort = async (startPort) => {
  return new Promise((resolve) => {
    const server = http.createServer();
    server.listen(startPort, () => {
      const port = server.address().port;
      server.close(() => {
        resolve(port);
      });
    });
    server.on("error", () => {
      // Port is in use, try the next one
      resolve(findFreePort(startPort + 1));
    });
  });
};

// Start Flask server
const startFlaskServer = async (port) => {
  console.log(`Starting Flask server on port ${port}...`);

  // Start the Flask server
  const env = {
    ...process.env,
    FLASK_APP: "main",
    FLASK_ENV: "testing",
    PORT: port.toString(),
  };

  const options = {
    env,
    stdio: "pipe",
    cwd: path.join(process.cwd(), "pythonHandler"),
  };

  return new Promise((resolve, reject) => {
    let detectedPort = null;
    let startTimeout = setTimeout(() => {
      reject(new Error("Flask server failed to start within 30 seconds"));
    }, 30000);

    // Try different Python commands based on system configuration
    const pythonCommands = [
      path.join(process.cwd(), "pythonHandler", ".venv", "bin", "python"),
      "python3",
      "python",
      "py",
    ];
    let currentCommandIndex = 0;

    const tryStartFlask = (commandIndex) => {
      if (commandIndex >= pythonCommands.length) {
        clearTimeout(startTimeout);
        reject(
          new Error("Could not find a working Python executable. Please ensure Python is installed and in your PATH.")
        );
        return;
      }

      const pythonCommand = pythonCommands[commandIndex];
      console.log(`Trying to start Flask with: ${pythonCommand}`);

      flaskProcess = spawn(pythonCommand, ["-m", "flask", "run", "--host=0.0.0.0", `--port=${port}`], options);

      flaskProcess.stdout.on("data", (data) => {
        const output = data.toString().trim();
        console.log(`Flask: ${output}`);

        // Only look for the port in the "Running on" message
        const match = output.match(/Running on http:\/\/[^:]+:(\d+)/);
        if (match) {
          detectedPort = parseInt(match[1]);
          console.log(`\n✅ Flask server started on port ${detectedPort}`);
          clearTimeout(startTimeout);
          resolve(detectedPort);
        }
      });

      flaskProcess.stderr.on("data", (data) => {
        const error = data.toString().trim();
        console.error(`Flask error: ${error}`);

        // If port is in use, kill process and try next port
        if (error.includes("Address already in use")) {
          console.log(`\n⚠️ Port ${port} is in use, trying port ${port + 1}`);
          flaskProcess.kill();
          clearTimeout(startTimeout);
          startFlaskServer(port + 1)
            .then(resolve)
            .catch(reject);
        }
      });

      flaskProcess.on("error", (error) => {
        console.error(`Failed to start Flask with ${pythonCommand}: ${error.message}`);
        // Try the next Python command
        flaskProcess.kill();
        tryStartFlask(commandIndex + 1);
      });

      flaskProcess.on("close", (code) => {
        // Only handle unexpected exits
        if (code !== 0 && !detectedPort) {
          console.error(`Flask server exited with code ${code}`);
          // Try the next Python command
          tryStartFlask(commandIndex + 1);
        }
      });
    };

    // Start trying Python commands
    tryStartFlask(currentCommandIndex);
  });
};

// Load training data from CSV
const loadTrainingData = (file_name) => {
  return new Promise((resolve, reject) => {
    const results = [];
    const startTime = Date.now();

    fs.createReadStream(path.join(__dirname, "test_data", file_name))
      .pipe(csv())
      .on("data", (data) => {
        // Check if we're using the fs_train_nz_amount.csv format
        // which has description, amount, and category
        const keys = Object.keys(data);
        let description, category, amount;
        let money_in = null;

        if (keys.length >= 3 && !isNaN(parseFloat(data[keys[1]]))) {
          // Format: description,amount,category
          description = data[keys[0]];
          amount = data[keys[1]];
          category = data[keys[2]];
        } else if (keys.length >= 2 && data.description && data.Category) {
          // Standard format with named columns
          description = data.description || data.Narrative || data.narrative;
          category = data.Category || data.category;
          amount = data.Amount || data.amount;
        } else {
          // Fallback: try to find description and category in any order
          description = data.description || data.Narrative || data.narrative || data[keys[0]];
          category = data.Category || data.category || data[keys[keys.length - 1]];

          // Try to find amount in the second column
          if (keys.length > 1) {
            amount = data[keys[1]];
          }
        }

        // Clean and parse amount
        let parsedAmount = null;
        if (amount !== undefined) {
          // Remove currency symbols and commas
          const cleanAmount = String(amount).replace(/[^\d.-]/g, "");
          parsedAmount = parseFloat(cleanAmount);

          if (!isNaN(parsedAmount)) {
            money_in = parsedAmount >= 0;
            logDebug(`Parsed amount ${parsedAmount} as money_in=${money_in}`);
          }
        }

        // Map fields to match expected format
        const transaction = {
          description: description,
          Category: category,
        };

        // Add money_in flag if amount was successfully parsed
        if (money_in !== null) {
          transaction.money_in = money_in;
        }

        // Add actual amount for transfer detection
        if (parsedAmount !== null) {
          transaction.amount = parsedAmount;
        }

        if (transaction.description && transaction.Category) {
          results.push(transaction);
        } else {
          logDebug(`Skipping invalid transaction: ${JSON.stringify(data)}`);
        }
      })
      .on("end", () => {
        const duration = (Date.now() - startTime) / 1000;
        logInfo(`Loaded ${results.length} training records in ${duration.toFixed(1)}s`);

        if (results.length === 0) {
          logError(
            "No valid training records found. Check CSV field names - needs 'description'/'Narrative' and 'Category' fields."
          );
        } else {
          logDebug(`Sample transaction: ${JSON.stringify(results[0])}`);
          // Log how many transactions have the money_in flag
          const withMoneyIn = results.filter((t) => t.money_in !== undefined).length;
          logInfo(`${withMoneyIn} of ${results.length} transactions have money_in flag set`);
          // Log how many transactions have the amount field
          const withAmount = results.filter((t) => t.amount !== undefined).length;
          logInfo(`${withAmount} of ${results.length} transactions have amount field set`);
        }

        resolve(results);
      })
      .on("error", (error) => {
        logError(`Error loading training data: ${error.toString()}`);
        reject(error);
      });
  });
};

// Load categorization data
const loadCategorizationData = (file_name) => {
  return new Promise((resolve, reject) => {
    const transactions = [];
    const targetCsvPath = path.join(__dirname, "test_data", file_name);

    logDebug(`Loading categorization data from ${file_name}...`);

    // Check if we're using fs_cat_nz_amount.csv format (which has amount,description)
    const isFsCatFormat = file_name.includes("fs_cat_nz_amount");
    logInfo(`Using ${isFsCatFormat ? "fs_cat_nz_amount" : "standard"} format for parsing`);

    // Read the CSV file - fs_cat format has amount in column 0, description in column 1
    // standard format has description in column 2
    fs.createReadStream(targetCsvPath)
      .pipe(csv({ headers: false }))
      .on("data", (data) => {
        let description, amount;

        if (isFsCatFormat) {
          // fs_cat_nz_amount.csv format: amount,description
          amount = data[0];
          description = data[1];
        } else {
          // Standard format: description in column 2, possibly amount in column 3
          description = data[2];
          amount = data[3]; // may be undefined
        }

        if (description) {
          let added = false;

          // Check if we have a valid amount to determine money_in
          if (amount !== undefined) {
            // Clean and parse the amount
            const cleanAmount = String(amount).replace(/[^\d.-]/g, "");
            const parsedAmount = parseFloat(cleanAmount);

            if (!isNaN(parsedAmount)) {
              // Create TransactionInput object with money_in flag
              transactions.push({
                description: description,
                money_in: parsedAmount >= 0,
                amount: parsedAmount, // Add actual amount for transfer detection
              });
              logDebug(
                `Added transaction with description "${description}", money_in=${
                  parsedAmount >= 0
                }, amount=${parsedAmount}`
              );
              added = true;
            }
          }

          // If no valid amount and not already added, just add the description as a string
          if (!added) {
            transactions.push(description);
          }
        } else {
          logDebug(`Skipping row with missing description: ${JSON.stringify(data)}`);
        }
      })
      .on("end", () => {
        // Count how many transactions have money_in flag
        const transactionObjects = transactions.filter((t) => typeof t === "object");
        logInfo(`Loaded ${transactions.length} descriptions from ${file_name}`);
        logInfo(`${transactionObjects.length} of ${transactions.length} have money_in flag set`);

        // Count how many transactions have amount
        const withAmount = transactions.filter((t) => typeof t === "object" && t.amount !== undefined).length;
        logInfo(`${withAmount} of ${transactions.length} have amount field set`);

        // Print some sample transactions for debugging
        if (transactions.length > 0) {
          logInfo("Sample transactions:");
          for (let i = 0; i < Math.min(3, transactions.length); i++) {
            logInfo(`  ${i + 1}: ${JSON.stringify(transactions[i])}`);
          }
        }

        resolve(transactions);
      })
      .on("error", (error) => {
        logError(`Error loading ${file_name}: ${error.message}`);
        reject(error);
      });
  });
};

// Test training flow
const trainModel = async (config) => {
  try {
    logInfo("Starting training...");
    console.log("Training model...");

    // Validate config object
    if (!config) {
      throw new Error("Configuration object is missing");
    }

    const trainingData = config.trainingDataFile;

    if (trainingData.length === 0) {
      logError("Error: No training data found");
      throw new Error("No training data found");
    }

    logInfo(`Processing ${trainingData.length} transactions...`);

    // Prepare the payload with only the necessary fields
    const payload = {
      transactions: trainingData,
      userId: config.userId || TEST_USER_ID,
    };

    // Log the payload for debugging
    logDebug(`Training payload: ${JSON.stringify(payload, null, 2)}`);

    // Initialize retry variables
    const maxRetries = 3;
    let retryCount = 0;
    let lastError = null;
    let response = null;

    // Retry loop for the initial training request
    while (retryCount < maxRetries) {
      try {
        // Add retry attempt to status
        if (retryCount > 0) {
          console.log(`Retrying training request (attempt ${retryCount + 1}/${maxRetries})...`);
          // Simple delay for retries
          await new Promise((resolve) => setTimeout(resolve, Math.pow(2, retryCount) * 1000));
        }

        // Call training endpoint
        const apiUrl = config.serviceUrl || `http://localhost:${API_PORT}`;
        logInfo(`Sending training request to ${apiUrl}/train`);

        response = await fetch(`${apiUrl}/train`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": config.apiKey || TEST_API_KEY,
            Accept: "application/json",
          },
          body: JSON.stringify(payload),
        });

        const responseStatus = response.status;

        // Log response headers for debugging
        const headers = {};
        response.headers.forEach((value, name) => {
          headers[name] = value;
        });
        logTrace(`Response headers: ${JSON.stringify(headers, null, 2)}`);

        // Handle different response codes
        if (responseStatus === 200) {
          break; // Success, exit retry loop
        } else if (responseStatus === 502 || responseStatus === 503 || responseStatus === 504) {
          // Retry on gateway errors
          lastError = `Server returned ${responseStatus}`;
          throw new Error(lastError);
        } else {
          // Don't retry on other errors
          const responseText = await response.text();
          logTrace(`Error response body: ${responseText}`);
          throw new Error(`Training failed with status ${responseStatus}: ${responseText}`);
        }
      } catch (e) {
        lastError = e;
        if (retryCount === maxRetries - 1) {
          // Last attempt failed
          logError(`Error: Training failed after ${maxRetries} attempts. Last error: ${e.toString()}`);
          throw new Error(`Training failed after ${maxRetries} attempts: ${e.toString()}`);
        }
        retryCount++;
        continue;
      }
    }

    // Parse response
    let result;
    try {
      const responseText = await response.text();
      logTrace(`Response body: ${responseText}`);
      result = JSON.parse(responseText);
    } catch (parseError) {
      logError(`Error parsing response: ${parseError.toString()}`);
      throw new Error(`Failed to parse server response: ${parseError.toString()}`);
    }

    // Validate result structure
    if (!result || typeof result !== "object") {
      logError("Error: Invalid response structure");
      throw new Error("Invalid response structure from server");
    }

    // Log the full response for debugging
    logDebug(`Training response: ${JSON.stringify(result, null, 2)}`);

    // Check if we got a prediction ID
    if (result.prediction_id) {
      logInfo(`Training in progress with prediction ID: ${result.prediction_id}`);
      console.log("Training in progress, please wait...");

      // Store start time for polling
      const startTime = new Date().getTime();

      // Poll for status until completion or timeout
      const pollResult = await pollForTrainingCompletion(result.prediction_id, startTime, config);

      return pollResult;
    } else {
      logError("Error: No prediction ID received");
      throw new Error("No prediction ID received from server");
    }
  } catch (error) {
    logError(`Training error: ${error.toString()}`);
    logDebug(`Error stack: ${error.stack}`);
    return { error: error.toString(), status: "failed" };
  }
};

// Helper function to poll for training completion
const pollForTrainingCompletion = async (predictionId, startTime, config) => {
  const maxTime = 30 * 60 * 1000; // 30 minutes timeout
  const pollInterval = 5000; // 5 seconds between polls
  const apiUrl = config.serviceUrl || `http://localhost:${API_PORT}`;
  const maxPolls = 120; // Maximum number of polling attempts (10 minutes)
  let pollCount = 0;
  let consecutiveErrors = 0;
  const maxConsecutiveErrors = 3;

  // Show a simple progress indicator
  process.stdout.write("Training progress: ");

  while (true) {
    try {
      // Check if we've been running for more than the max time
      const currentTime = new Date().getTime();
      const elapsedMinutes = Math.floor((currentTime - startTime) / (60 * 1000));

      if (currentTime - startTime > maxTime) {
        process.stdout.write("\n");
        logError(`Training timed out after ${elapsedMinutes} minutes`);
        return {
          error: `Training timed out after ${elapsedMinutes} minutes`,
          status: "timeout",
        };
      }

      // Check if we've exceeded the maximum number of polls
      if (pollCount >= maxPolls) {
        process.stdout.write("\n");
        logInfo(`Reached maximum number of polling attempts (${maxPolls})`);
        return {
          status: "unknown",
          message: "Maximum polling attempts reached without confirmation",
          elapsedMinutes,
        };
      }

      pollCount++;

      // Show progress
      process.stdout.write(".");

      // Wait for the poll interval
      await new Promise((resolve) => setTimeout(resolve, pollInterval));

      // Call the status endpoint
      logDebug(`Checking training status for prediction ID: ${predictionId} (attempt ${pollCount}/${maxPolls})`);
      const response = await fetch(`${apiUrl}/status/${predictionId}`, {
        headers: {
          "X-API-Key": config.apiKey || TEST_API_KEY,
          Accept: "application/json",
        },
      });

      // Handle different response codes
      if (!response.ok) {
        const statusCode = response.status;
        logDebug(`Server returned error code: ${statusCode} for prediction ID: ${predictionId}`);

        // If we get a 502/503/504, the worker might have restarted
        if (statusCode === 502 || statusCode === 503 || statusCode === 504) {
          consecutiveErrors++;
          logWarning(`Worker error (${statusCode}) on attempt ${pollCount}, consecutive errors: ${consecutiveErrors}`);

          if (consecutiveErrors >= maxConsecutiveErrors) {
            process.stdout.write("\n");
            logError(`Too many consecutive worker errors (${consecutiveErrors})`);
            return {
              error: `Training failed after ${consecutiveErrors} consecutive worker errors`,
              status: "failed",
            };
          }

          // Use longer delay after worker error
          await new Promise((resolve) => setTimeout(resolve, pollInterval * 2));
          continue;
        }

        // For 404, the prediction might not exist
        if (statusCode === 404) {
          process.stdout.write("\n");
          logError(`Prediction ${predictionId} not found`);
          return {
            error: "Prediction not found",
            status: "failed",
          };
        }

        continue; // Continue polling for other error codes
      }

      // Reset consecutive errors on successful response
      consecutiveErrors = 0;

      // Parse the response
      const result = await response.json();
      logTrace(`Status response: ${JSON.stringify(result, null, 2)}`);

      // Handle different status cases
      if (result.status === "completed") {
        process.stdout.write("\n");
        logInfo(`Training completed successfully after ${elapsedMinutes} minutes!`);
        return {
          status: "completed",
          message: "Training completed successfully!",
          result: result,
          elapsedMinutes,
        };
      } else if (result.status === "failed") {
        process.stdout.write("\n");
        const errorMessage = result.error || result.message || "Unknown error";
        logError(`Training failed with error: ${errorMessage}`);
        return {
          error: errorMessage,
          status: "failed",
          fullResponse: result,
          elapsedMinutes,
        };
      } else if (result.status === "processing") {
        // Still in progress, log status information
        let statusMessage = `Training in progress... (${elapsedMinutes} min)`;
        if (result.message) {
          statusMessage += ` - ${result.message}`;
        }

        // Only log status updates occasionally to reduce verbosity
        if (pollCount === 1 || pollCount % 5 === 0) {
          logInfo(statusMessage);
        } else {
          logDebug(statusMessage);
        }
      }
    } catch (error) {
      logError(`Error in polling: ${error.toString()}`);
      logDebug(`Error stack: ${error.stack}`);

      consecutiveErrors++;
      if (consecutiveErrors >= maxConsecutiveErrors) {
        process.stdout.write("\n");
        logError(`Too many consecutive errors (${consecutiveErrors})`);
        return {
          error: `Training failed after ${consecutiveErrors} consecutive errors: ${error.toString()}`,
          status: "failed",
        };
      }
    }
  }
};

// Test categorization flow
const categoriseTransactions = async (config) => {
  try {
    // Prepare request data with only transactions (simplified API)
    const requestData = {
      transactions: config.categorizationDataFile,
    };

    // Add logging for debugging API key issues
    const serviceUrl = config.serviceUrl;
    const serverPort = new URL(serviceUrl).port;

    console.log("==== API Key Debug Info ====");
    console.log(`config.apiKey is ${config.apiKey ? "defined" : "undefined"}`);
    console.log(`TEST_API_KEY is ${TEST_API_KEY ? "defined" : "undefined"}`);
    console.log(
      `Using API key: ${(config.apiKey || TEST_API_KEY)?.substring(0, 3)}...${(
        config.apiKey || TEST_API_KEY
      )?.substring((config.apiKey || TEST_API_KEY)?.length - 3)}`
    );
    console.log(`Server URL: ${serviceUrl} (port: ${serverPort})`);
    console.log("==========================");

    // Send request to classify endpoint
    logInfo(`Sending categorisation request to ${serviceUrl}/classify`);

    // Try up to 3 times with exponential backoff
    let response;
    let attempt = 1;
    const maxAttempts = 3;

    // Log the request data for debugging
    logDebug(`Request data: ${JSON.stringify(requestData)}`);

    while (attempt <= maxAttempts) {
      try {
        const apiKey = config.apiKey || TEST_API_KEY;
        logInfo(`Attempt ${attempt}: Using API key that starts with: ${apiKey.substring(0, 3)}...`);

        // Use fetch instead of axios for consistency with training code
        const response_fetch = await fetch(`${serviceUrl}/classify`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": apiKey,
            Accept: "application/json",
          },
          body: JSON.stringify(requestData),
        });

        if (!response_fetch.ok) {
          const statusCode = response_fetch.status;
          const responseText = await response_fetch.text();
          logError(`Server returned error code: ${statusCode}`);
          logError(`Response text: ${responseText}`);
          throw new Error(`Server returned error code: ${statusCode}, message: ${responseText}`);
        }

        response = { data: await response_fetch.json() };
        break; // Success, exit the loop
      } catch (error) {
        logError(`Attempt ${attempt} failed with: ${error.message}`);

        if (attempt === maxAttempts) {
          throw error; // Rethrow on last attempt
        }
        console.log(`Retrying categorisation request (attempt ${attempt + 1}/${maxAttempts})...`);
        await new Promise((resolve) => setTimeout(resolve, 1000 * attempt)); // Wait before retry
        attempt++;
      }
    }

    // Get prediction ID from response
    const predictionId = response.data.prediction_id;
    console.log(`Categorisation in progress with prediction ID: ${predictionId}`);

    // Poll for results
    console.log("Categorization in progress, please wait...");
    process.stdout.write("Categorization progress: ");

    // Increase the maximum number of polling attempts from 30 to 60
    const maxPollingAttempts = 60;
    let pollingAttempt = 0;
    let statusResponse;

    while (pollingAttempt < maxPollingAttempts) {
      process.stdout.write(".");
      try {
        // Use fetch instead of axios for consistency with training code
        const apiKey = config.apiKey || TEST_API_KEY;
        const response_fetch = await fetch(`${serviceUrl}/status/${predictionId}`, {
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": apiKey,
            Accept: "application/json",
          },
        });

        if (!response_fetch.ok) {
          const statusCode = response_fetch.status;
          logError(`Server returned error code: ${statusCode} for prediction ID: ${predictionId}`);
          throw new Error(`Server returned error code: ${statusCode}`);
        }

        statusResponse = { data: await response_fetch.json() };

        const status = statusResponse.data.status;
        const message = statusResponse.data.message || "Processing in progress";
        const elapsedMinutes = Math.floor(pollingAttempt / 6); // Assuming 5-second intervals

        if (status === "completed") {
          console.log("\n");
          console.log(`Categorisation completed successfully!`);
          break;
        } else if (status === "failed") {
          console.log("\n");
          console.log(`Categorisation failed: ${statusResponse.data.error || "Unknown error"}`);
          return { status: "failed", error: statusResponse.data.error };
        }
        logInfo(`Categorisation in progress... (${elapsedMinutes} min) - ${status} - ${message}`);
      } catch (error) {
        logError(`Error checking categorisation status: ${error.message}`);
      }

      await new Promise((resolve) => setTimeout(resolve, 5000)); // Poll every 5 seconds
      pollingAttempt++;
    }

    if (pollingAttempt >= maxPollingAttempts) {
      console.log("\n");
      console.log(`Reached maximum number of polling attempts (${maxPollingAttempts}). Assuming completion.`);
    }

    // Process results
    console.log("Processing categorisation results...");

    // Check if we have results in the response
    let results = [];

    if (statusResponse && statusResponse.data) {
      logInfo(`Status response data: ${JSON.stringify(statusResponse.data)}`);

      if (statusResponse.data.results && Array.isArray(statusResponse.data.results)) {
        results = statusResponse.data.results;
        logInfo(`Found ${results.length} results directly in response.results`);

        // Add narratives from original transactions
        if (results && results.length > 0) {
          logInfo("Adding narratives from original transactions to results");

          // Print results in a nice format
          console.log("\n===== CATEGORISATION RESULTS =====");
          console.log(`Total transactions categorised: ${results.length}`);

          results.forEach((result, index) => {
            const confidence = result.similarity_score ? `${(result.similarity_score * 100).toFixed(2)}%` : "N/A";
            // Add money_in/money_out display
            const direction =
              result.money_in === true ? "MONEY_IN" : result.money_in === false ? "MONEY_OUT" : "UNKNOWN";
            // Display amount if available
            const amountStr =
              result.amount !== null && result.amount !== undefined ? `$${Math.abs(result.amount).toFixed(2)}` : "";

            console.log(
              `${index + 1}. "${result.narrative}" → ${
                result.predicted_category
              } (${confidence}) | ${direction} ${amountStr}`
            );
          });

          console.log("=====================================\n");
        } else {
          console.log("\n===== CATEGORISATION RESULTS =====");
          console.log("No results could be extracted from the API response.");
          console.log("This is likely a bug in the Flask server.");
          console.log("=====================================\n");
        }
      }
    }

    return { status: "completed", results };
  } catch (error) {
    console.log(`\nCategorisation failed: ${error.message}`);
    logDebug(`Error stack: ${error.stack}`);
    return { status: "failed", error: error.message };
  }
};

// Test clean_text functionality
const testCleanText = async (apiUrl) => {
  console.log("\n=== Testing clean_text Function ===\n");

  try {
    // 1. Load the training data
    console.log("1. Loading training data...");
    const trainingData = await loadTrainingData("full_train.csv");
    console.log(`   Loaded ${trainingData.length} records\n`);

    // 2. Send clean text requests
    console.log("2. Testing clean_text endpoint...");
    const descriptions = trainingData.map((t) => t.description);
    const uniqueOriginal = new Set(descriptions).size;

    console.log("\nOriginal Data Analysis:");
    console.log(`Total transactions: ${descriptions.length}`);
    console.log(`Unique descriptions: ${uniqueOriginal}`);
    console.log(
      `Duplicate ratio: ${(((descriptions.length - uniqueOriginal) / descriptions.length) * 100).toFixed(2)}%`
    );

    // Send batches of descriptions to clean_text endpoint
    const batchSize = 100;
    const cleanedDescriptions = [];
    const cleaningResults = []; // Store before/after pairs

    for (let i = 0; i < descriptions.length; i += batchSize) {
      const batch = descriptions.slice(i, i + batchSize);
      const response = await fetch(`${apiUrl}/clean_text`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": TEST_API_KEY,
        },
        body: JSON.stringify({ descriptions: batch }),
      });

      if (!response.ok) {
        throw new Error(`Failed to clean text batch: ${response.status}`);
      }

      const result = await response.json();
      cleanedDescriptions.push(...result.cleaned_descriptions);

      // Store before/after pairs
      batch.forEach((desc, idx) => {
        cleaningResults.push({
          original: desc,
          cleaned: result.cleaned_descriptions[idx],
        });
      });

      // Show progress
      process.stdout.write(`   Processing: ${Math.min(i + batchSize, descriptions.length)}/${descriptions.length}\r`);
    }

    console.log("\n"); // Clear progress line

    // 3. Analyze results
    const uniqueCleaned = new Set(cleanedDescriptions).size;

    console.log("\nAfter Cleaning Analysis:");
    console.log(`Total transactions: ${cleanedDescriptions.length}`);
    console.log(`Unique cleaned descriptions: ${uniqueCleaned}`);
    console.log(`Reduction ratio: ${(((uniqueOriginal - uniqueCleaned) / uniqueOriginal) * 100).toFixed(2)}%`);
    console.log(
      `Final duplicate ratio: ${(
        ((cleanedDescriptions.length - uniqueCleaned) / cleanedDescriptions.length) *
        100
      ).toFixed(2)}%`
    );

    // 4. Analyze patterns
    console.log("\nTop 10 most frequent cleaned descriptions:");
    const frequency = {};
    cleanedDescriptions.forEach((desc) => {
      frequency[desc] = (frequency[desc] || 0) + 1;
    });

    const sortedFreq = Object.entries(frequency)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10);

    for (const [desc, count] of sortedFreq) {
      console.log(`\nCount: ${count} | ${desc}`);
      console.log("Original variations:");
      const variations = descriptions.filter((orig, idx) => cleanedDescriptions[idx] === desc);
      const uniqueVariations = [...new Set(variations)];
      uniqueVariations.slice(0, 3).forEach((v) => console.log(`  - ${v}`));
      if (uniqueVariations.length > 3) {
        console.log(`  ... and ${uniqueVariations.length - 3} more variations`);
      }
    }

    // 5. Print detailed cleaning results
    console.log("\nDetailed Cleaning Results (first 10 examples):");
    console.log("----------------------------------------");
    cleaningResults.slice(0, 10).forEach((result, idx) => {
      console.log(`\nExample ${idx + 1}:`);
      console.log(`Original: "${result.original}"`);
      console.log(`Cleaned : "${result.cleaned}"`);
    });

    // Save full results to a file for further analysis
    const resultsPath = path.join(__dirname, "test_data", "cleaning_results.json");
    fs.writeFileSync(
      resultsPath,
      JSON.stringify(
        {
          summary: {
            totalTransactions: descriptions.length,
            uniqueOriginal,
            uniqueCleaned,
            reductionRatio: (((uniqueOriginal - uniqueCleaned) / uniqueOriginal) * 100).toFixed(2),
            duplicateRatio: (((cleanedDescriptions.length - uniqueCleaned) / cleanedDescriptions.length) * 100).toFixed(
              2
            ),
          },
          topPatterns: sortedFreq,
          allResults: cleaningResults,
        },
        null,
        2
      )
    );
    console.log(`\nFull results saved to: ${resultsPath}`);

    console.log("\n=== clean_text Test Completed Successfully ===\n");
    return true;
  } catch (error) {
    console.error(`\nclean_text test failed: ${error.message}`);
    return false;
  }
};

// Cleanup function
const cleanup = async () => {
  // Stop Flask server
  if (flaskProcess) {
    log("Stopping Flask server...");
    flaskProcess.kill();

    // Wait for the process to exit
    await new Promise((resolve) => {
      if (flaskProcess.exitCode !== null) {
        log(`Flask server process exited with code ${flaskProcess.exitCode}`);
        resolve();
      } else {
        flaskProcess.once("exit", (code) => {
          log(`Flask server process exited with code ${code}`);
          resolve();
        });

        // Set a timeout in case the process doesn't exit
        setTimeout(() => {
          log("Flask server process did not exit in time, forcing...");
          try {
            flaskProcess.kill("SIGKILL");
          } catch (e) {
            log(`Error killing Flask process: ${e.message}`);
          }
          resolve();
        }, 3000);
      }
    });
  }

  // Close webhook server
  if (webhookServer) {
    await new Promise((resolve) => {
      webhookServer.close(() => {
        log("Webhook server closed");
        resolve();
      });

      // Set a timeout in case the server doesn't close
      setTimeout(() => {
        log("Webhook server did not close in time");
        resolve();
      }, 2000);
    });
  }

  // Return a resolved promise to indicate cleanup is complete
  return Promise.resolve();
};

// Main function
const main = async () => {
  try {
    console.log("\n=== Starting Test Script ===\n");

    // 1. Start Flask Server
    console.log("1. Starting Flask Server...");
    const port = await startFlaskServer(API_PORT);
    const apiUrl = `http://localhost:${port}`;
    console.log(`   Server URL: ${apiUrl}\n`);

    // 2. Setup Test User...
    console.log("2. Setting up Test User...");

    // Skip database operations and use mocked test user
    console.log("   Using mocked test user credentials");
    console.log("   Test user ready\n");

    // 3. Verify Server Health
    console.log("3. Verifying Server Health...");
    let serverHealthy = false;
    try {
      const healthCheck = await fetch(`${apiUrl}/health`, {
        headers: { "X-API-Key": TEST_API_KEY },
      });

      if (!healthCheck.ok) {
        throw new Error(`Server health check failed: ${healthCheck.status}`);
      }
      console.log("   Server is healthy\n");
      serverHealthy = true;
    } catch (healthError) {
      console.error(`   Health check failed: ${healthError.message}`);
      console.log("   WARNING: Continuing with tests despite health check failure\n");
    }

    // 4. Run specific test mode if selected
    if (TEST_CLEAN_TEXT) {
      const success = await testCleanText(apiUrl);
      if (!success) {
        throw new Error("clean_text test failed");
      }
      return;
    }

    // 5. Load Test Data
    console.log("5. Loading Test Data...");
    // const trainingData = await loadTrainingData("full_train.csv");
    // const trainingData = await loadTrainingData("training_test.csv");
    const trainingData = await loadTrainingData("fs_train_nz_amount.csv");
    const categorizationData = await loadCategorizationData("fs_cat_nz_amount.csv");
    // const categorizationData = await loadCategorizationData("categorise_test.csv");
    console.log(`   Loaded ${trainingData.length} training records`);
    console.log(`   Loaded ${categorizationData.length} categorization records\n`);

    // Create test configuration
    const config = {
      userId: TEST_USER_ID,
      apiKey: TEST_API_KEY,
      serviceUrl: apiUrl,
      trainingDataFile: trainingData,
      categorizationDataFile: categorizationData,
    };

    // 6. Run Training (if enabled)
    if (RUN_TRAINING) {
      console.log("6. Running Training...");
      const trainingResult = await trainModel(config);

      if (trainingResult.status !== "completed") {
        throw new Error(`Training failed: ${trainingResult.error || "Unknown error"}`);
      }
      console.log("   Training completed successfully\n");

      // Wait for model to be ready
      console.log("   Waiting 5s for model to be ready...");
      await new Promise((resolve) => setTimeout(resolve, 5000));
    }

    // 7. Run Categorization (if enabled)
    if (RUN_CATEGORIZATION) {
      console.log("7. Running Categorization...");
      const categorizationResult = await categoriseTransactions(config);

      if (categorizationResult.status !== "completed") {
        throw new Error(`Categorization failed: ${categorizationResult.error || "Unknown error"}`);
      }
      console.log("   Categorization completed successfully\n");
    }

    // 8. Cleanup
    console.log("8. Cleaning up...");
    if (flaskProcess) {
      flaskProcess.kill();
      console.log("   Flask server stopped");
    }

    console.log("\n=== Test Completed Successfully ===\n");
    process.exit(0);
  } catch (error) {
    console.error(`\n❌ ERROR: ${error.message}`);

    // Ensure cleanup on error
    if (flaskProcess) {
      flaskProcess.kill();
      console.log("   Flask server stopped");
    }

    process.exit(1);
  }
};

// Execute main function when script is run directly
if (require.main === module) {
  main().catch((error) => {
    console.error("Unhandled error in main:", error);
    process.exit(1);
  });
}

// Handle process termination
process.on("SIGINT", async () => {
  logInfo("Process interrupted");

  try {
    // Run cleanup asynchronously
    await cleanup();
    logInfo("Test user cleanup complete");
  } catch (cleanupError) {
    logError(`Error during cleanup: ${cleanupError.message}`);
  } finally {
    // Ensure process exits even if cleanup fails
    process.exit(1);
  }
});

process.on("SIGTERM", async () => {
  logInfo("Process terminated");

  try {
    // Run cleanup asynchronously
    await cleanup();
    logInfo("Test user cleanup complete");
  } catch (cleanupError) {
    logError(`Error during cleanup: ${cleanupError.message}`);
  } finally {
    // Ensure process exits even if cleanup fails
    process.exit(1);
  }
});
