// Import path first so we can use it for dotenv config
const path = require("path");
// Load environment variables from .env file with explicit path
require("dotenv").config({ path: path.resolve(__dirname, "../.env") });

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

// Verify critical environment variables are loaded
logInfo("Checking environment variables...");
const criticalEnvVars = ["DATABASE_URL", "REPLICATE_API_TOKEN", "TEST_API_KEY", "TEST_USER_ID"];
const missingVars = criticalEnvVars.filter((varName) => !process.env[varName]);
if (missingVars.length > 0) {
  logError(`❌ Missing required environment variables: ${missingVars.join(", ")}`);
  logError("Please check that your .env file is in the project root and contains these variables.");
  process.exit(1);
}
logInfo("✅ Environment variables loaded successfully");

const fs = require("fs");
const csv = require("csv-parser");

// Configuration
const API_PORT = process.env.API_PORT || 3005;
const TEST_USER_ID = process.env.TEST_USER_ID || "test_user_fixed";
const TEST_API_KEY = process.env.TEST_API_KEY || "test_api_key_fixed";

// Global variables
let webhookServer;

// Check for test mode flags
const RUN_TRAINING = !process.argv.includes("--cat-only");
const RUN_CATEGORIZATION = !process.argv.includes("--train-only");
const TEST_CLEAN_TEXT = process.argv.includes("--test-clean");

// Log the actual values being used
logInfo(`Using TEST_USER_ID: ${TEST_USER_ID}`);
logInfo(`Using TEST_API_KEY: ${TEST_API_KEY ? "***" + TEST_API_KEY.slice(-4) : "Not set"}`);

// Load training data from CSV
const loadTrainingData = (file_name) => {
  return new Promise((resolve, reject) => {
    const results = [];
    const categories = new Set(); // Use a Set to store unique category names
    const startTime = Date.now();
    let headersProcessed = false;
    let descriptionIndex = -1;
    let categoryIndex = -1;
    let amountIndex = -1;

    const stream = fs
      .createReadStream(path.join(__dirname, "test_data", file_name))
      .pipe(
        csv({
          mapHeaders: ({ header, index }) => {
            const lowerHeader = header.toLowerCase().trim();
            // Allow for variations like 'Description' or 'Narrative'
            if (["description", "narrative"].includes(lowerHeader)) {
              return "description";
            }
            // Allow 'Category'
            if (lowerHeader === "category") {
              return "category";
            }
            // Allow 'Amount' or 'Amount Spent'
            if (["amount", "amount spent"].includes(lowerHeader)) {
              return "amount";
            }
            // Return null for headers we don't explicitly map to keep them if needed, or ignore others
            return null;
          },
        })
      )
      .on("headers", (headers) => {
        // Find the indices based on the mapped headers
        descriptionIndex = headers.indexOf("description");
        categoryIndex = headers.indexOf("category");
        amountIndex = headers.indexOf("amount");
        headersProcessed = true;

        logDebug(
          `Header indices found: Description=${descriptionIndex}, Category=${categoryIndex}, Amount=${amountIndex}`
        );

        // Validate required headers
        if (descriptionIndex === -1) {
          stream.destroy(); // Stop processing the stream
          return reject(new Error("CSV file must contain a header named 'Description' or 'Narrative'"));
        }
        if (categoryIndex === -1) {
          stream.destroy();
          return reject(new Error("CSV file must contain a header named 'Category'"));
        }
        if (amountIndex === -1) {
          logInfo("Optional 'Amount' or 'Amount Spent' header not found. Proceeding without amount data.");
        }
      })
      .on("data", (row) => {
        // Ensure headers were processed before handling data
        if (!headersProcessed) return;

        const description = row.description; // Access by mapped name
        const category = row.category; // Access by mapped name
        const amountValue = amountIndex !== -1 ? row.amount : undefined; // Access by mapped name if index found

        let money_in = null;
        let parsedAmount = null;

        // Process amount only if the column exists
        if (amountIndex !== -1 && amountValue !== undefined && amountValue !== null) {
          // Remove currency symbols and commas
          const cleanAmount = String(amountValue).replace(/[^\d.-]/g, "");
          parsedAmount = parseFloat(cleanAmount);

          if (!isNaN(parsedAmount)) {
            money_in = parsedAmount >= 0;
            logDebug(`Parsed amount ${parsedAmount} as money_in=${money_in}`);
          } else {
            logDebug(`Could not parse amount: "${amountValue}" for description: "${description}"`);
            parsedAmount = null; // Ensure it's null if parsing failed
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

        // Add actual amount for transfer detection if parsed
        if (parsedAmount !== null) {
          transaction.amount = parsedAmount;
        }

        // Ensure we have the required fields before adding
        if (transaction.description && transaction.Category) {
          results.push(transaction);
          categories.add(transaction.Category); // Add category name to the Set
        } else {
          logDebug(`Skipping invalid transaction data: ${JSON.stringify(row)}`);
        }
      })
      .on("end", () => {
        const duration = (Date.now() - startTime) / 1000;
        logInfo(`Loaded ${results.length} training records in ${duration.toFixed(1)}s`);
        const uniqueCategories = Array.from(categories); // Convert Set to Array
        logInfo(`Found ${uniqueCategories.length} unique categories in training data.`);

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
          logDebug(`Unique categories found: ${uniqueCategories.join(", ")}`); // Log unique categories
        }

        resolve({ transactions: results, uniqueCategories }); // Return both transactions and unique categories
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
    let headersProcessed = false;
    let descriptionIndex = -1;
    let amountIndex = -1;

    logDebug(`Loading categorization data from ${file_name}...`);

    const stream = fs
      .createReadStream(targetCsvPath)
      .pipe(
        csv({
          mapHeaders: ({ header, index }) => {
            const lowerHeader = header.toLowerCase().trim();
            // Allow for variations like 'Description' or 'Narrative'
            if (["description", "narrative"].includes(lowerHeader)) {
              return "description";
            }
            // Allow 'Amount' or 'Amount Spent'
            if (["amount", "amount spent"].includes(lowerHeader)) {
              return "amount";
            }
            return null; // Ignore other headers
          },
        })
      )
      .on("headers", (headers) => {
        descriptionIndex = headers.indexOf("description");
        amountIndex = headers.indexOf("amount");
        headersProcessed = true;

        logDebug(`Header indices found: Description=${descriptionIndex}, Amount=${amountIndex}`);

        // Validate required headers
        if (descriptionIndex === -1) {
          stream.destroy();
          return reject(new Error("Categorization CSV must contain a header named 'Description' or 'Narrative'"));
        }
        if (amountIndex === -1) {
          logInfo("Optional 'Amount' or 'Amount Spent' header not found. Proceeding without amount/money_in data.");
        }
      })
      .on("data", (row) => {
        if (!headersProcessed) return;

        const description = row.description;
        const amountValue = amountIndex !== -1 ? row.amount : undefined;

        if (description) {
          let parsedAmount = null;
          let money_in = null;

          // Check if we have a valid amount to determine money_in
          if (amountIndex !== -1 && amountValue !== undefined && amountValue !== null) {
            // Clean and parse the amount
            const cleanAmount = String(amountValue).replace(/[^\d.-]/g, "");
            parsedAmount = parseFloat(cleanAmount);

            if (!isNaN(parsedAmount)) {
              money_in = parsedAmount >= 0;
            } else {
              logDebug(`Could not parse amount: "${amountValue}" for description: "${description}"`);
              parsedAmount = null; // Ensure null if parsing failed
            }
          }

          // Create a TransactionInput object, including amount/money_in if available
          transactions.push({
            description: description,
            money_in: money_in, // Will be null if amount wasn't found or parsed
            amount: parsedAmount, // Will be null if amount wasn't found or parsed
          });
          logDebug(`Added transaction: desc="${description}", money_in=${money_in}, amount=${parsedAmount}`);
        } else {
          logDebug(`Skipping row with missing description: ${JSON.stringify(row)}`);
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
  const operationStartTime = Date.now();
  let usedPolling = false;

  try {
    logInfo("Starting training...");
    logInfo("Training model...");

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
    let predictionId = null; // To store prediction ID if async

    // Retry loop for the initial training request
    while (retryCount < maxRetries) {
      try {
        // Add retry attempt to status
        if (retryCount > 0) {
          logInfo(`Retrying training request (attempt ${retryCount + 1}/${maxRetries})...`);
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
          // Could be synchronous success OR async start (check body)
          const responseText = await response.text();
          logTrace(`Response body (status 200/202): ${responseText}`);
          const result = JSON.parse(responseText);

          if (result.status === "completed") {
            // Synchronous Success!
            logInfo("Training completed synchronously.");
            const durationMs = Date.now() - operationStartTime;
            return {
              status: "completed",
              message: "Training completed successfully (synchronous)!",
              result: result,
              durationMs: durationMs,
              usedPolling: false,
            };
          } else if (result.status === "processing" && result.prediction_id) {
            // Asynchronous start indicated by 200 status + processing status in body
            logInfo("Training started asynchronously (indicated by 200 response).");
            predictionId = result.prediction_id;
            usedPolling = true; // Polling will be used
            break; // Exit retry loop, proceed to polling
          } else {
            // Unexpected body format for 200 status
            throw new Error(`Training failed: Received status 200 but unexpected body format: ${responseText}`);
          }
        } else if (responseStatus === 202) {
          // Asynchronous start indicated by 202 status
          const responseText = await response.text();
          logTrace(`Response body (status 202): ${responseText}`);
          const result = JSON.parse(responseText);
          if (result.prediction_id) {
            logInfo("Training started asynchronously (indicated by 202 response).");
            predictionId = result.prediction_id;
            usedPolling = true; // Polling will be used
            break; // Exit retry loop, proceed to polling
          } else {
            throw new Error(`Training failed: Received status 202 but no prediction_id in body: ${responseText}`);
          }
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

    // If we exited the loop without a predictionId (and didn't return sync success),
    // it means all retries failed or an unexpected error occurred.
    if (!predictionId) {
      logError(
        `Error: Training request failed after ${maxRetries} attempts. Last error: ${
          lastError?.toString() || "Unknown error"
        }`
      );
      const durationMs = Date.now() - operationStartTime;
      throw new Error(
        `Training request failed after ${maxRetries} attempts: ${lastError?.toString() || "Unknown error"}`
      );
    }

    // Check if we got a prediction ID
    if (predictionId) {
      logInfo(`Training requires polling with prediction ID: ${predictionId}`);
      logInfo("Training in progress, please wait...");
      usedPolling = true; // Explicitly set as polling is used

      // Store start time for polling
      const pollingStartTime = new Date().getTime();

      // Poll for status until completion or timeout
      const pollResult = await pollForTrainingCompletion(predictionId, pollingStartTime, config);
      const durationMs = Date.now() - operationStartTime;

      return { ...pollResult, durationMs: durationMs, usedPolling: true };
    } else {
      logError("Error: No prediction ID received");
      const durationMs = Date.now() - operationStartTime;
      throw new Error("No prediction ID received from server"); // This will be caught by the main catch
    }
  } catch (error) {
    logError(`Training error: ${error.toString()}`);
    logDebug(`Error stack: ${error.stack}`);
    const durationMs = Date.now() - operationStartTime;
    return { error: error.toString(), status: "failed", durationMs: durationMs, usedPolling: usedPolling };
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
const categoriseTransactions = async (config, userCategories) => {
  const operationStartTime = Date.now();
  let usedPolling = false;

  try {
    // Prepare request data with transactions and categories
    const requestData = {
      transactions: config.categorizationDataFile,
      user_categories: userCategories || [], // Include the categories, default to empty array
    };

    // Add logging for debugging API key issues
    const serviceUrl = config.serviceUrl;
    const serverPort = new URL(serviceUrl).port;

    logDebug("==== API Key Debug Info ====");
    logDebug(`config.apiKey is ${config.apiKey ? "defined" : "undefined"}`);
    logDebug(`TEST_API_KEY is ${TEST_API_KEY ? "defined" : "undefined"}`);
    logDebug(
      `Using API key: ${(config.apiKey || TEST_API_KEY)?.substring(0, 3)}...${(
        config.apiKey || TEST_API_KEY
      )?.substring((config.apiKey || TEST_API_KEY)?.length - 3)}`
    );
    logDebug(`Server URL: ${serviceUrl} (port: ${serverPort})`);
    logDebug("==========================");

    // Send request to classify endpoint
    logInfo(`Sending categorisation request to ${serviceUrl}/classify`);

    // Try up to 3 times with exponential backoff
    let response;
    let attempt = 1;
    const maxAttempts = 3;
    let predictionId; // Variable to store prediction ID if we go async

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

        // --- Handle different response statuses ---
        if (response_fetch.status === 200) {
          // Synchronous success!
          logInfo("Received synchronous classification results.");
          response = { data: await response_fetch.json(), status: 200 };
          // usedPolling remains false
          break; // Exit the loop, we have results
        } else if (response_fetch.status === 202) {
          // Asynchronous processing started
          logInfo("Classification started asynchronously. Polling required.");
          response = { data: await response_fetch.json(), status: 202 };
          predictionId = response.data.prediction_id;
          usedPolling = true; // Polling will be used
          if (!predictionId) {
            throw new Error("Server started async processing but did not return a prediction ID.");
          }
          break; // Exit the loop, we need to poll
        } else {
          // Other error
          const statusCode = response_fetch.status;
          const responseText = await response_fetch.text();
          logError(`Server returned error code: ${statusCode}`);
          logError(`Response text: ${responseText}`);
          throw new Error(`Server returned error code: ${statusCode}, message: ${responseText}`);
        }
      } catch (error) {
        logError(`Attempt ${attempt} failed with: ${error.message}`);

        if (attempt === maxAttempts) {
          throw error; // Rethrow on last attempt
        }
        logInfo(`Retrying categorisation request (attempt ${attempt + 1}/${maxAttempts})...`);
        await new Promise((resolve) => setTimeout(resolve, 1000 * attempt)); // Wait before retry
        attempt++;
      }
    }

    // Check if we got a successful response (either sync or async start)
    if (!response) {
      throw new Error("Categorisation request failed after multiple attempts.");
    }

    // --- Process based on response status ---
    let results = [];
    let finalStatus = "failed"; // Default status

    if (response.status === 200) {
      // Process synchronous results directly
      if (response.data && response.data.results && response.data.status === "completed") {
        results = response.data.results;
        finalStatus = "completed";
        logInfo(`Synchronous classification completed successfully! Found ${results.length} results.`);
      } else {
        logError("Synchronous response received, but results are missing or status is not 'completed'.");
        logError("Response data:", JSON.stringify(response.data, null, 2));
        const durationMs = Date.now() - operationStartTime;
        throw new Error("Invalid synchronous response format."); // Will be caught by outer catch
      }
    } else if (response.status === 202 && predictionId) {
      // --- Start Polling for Asynchronous Results ---
      logInfo(`Categorisation in progress with prediction ID: ${predictionId}`);
      logInfo("Polling for results...");
      process.stdout.write("Categorization progress: ");

      const maxPollingAttempts = 60;
      let pollingAttempt = 0;
      let statusResponse;

      while (pollingAttempt < maxPollingAttempts) {
        process.stdout.write(".");
        try {
          // Use fetch instead of axios for consistency with training code
          const apiKey = config.apiKey || TEST_API_KEY;
          const poll_response_fetch = await fetch(`${serviceUrl}/status/${predictionId}`, {
            headers: {
              "Content-Type": "application/json",
              "X-API-Key": apiKey,
              Accept: "application/json",
            },
          });

          if (!poll_response_fetch.ok) {
            const statusCode = poll_response_fetch.status;
            logError(`Polling error: Server returned error code ${statusCode} for prediction ID: ${predictionId}`);
            // Decide if we should retry or fail based on status code (e.g., temporary vs permanent error)
            // For simplicity, we'll throw an error here, but could implement retries
            throw new Error(`Polling error: Server returned status code ${statusCode}`);
          }

          statusResponse = { data: await poll_response_fetch.json() };

          const status = statusResponse.data.status;
          const message = statusResponse.data.message || "Processing in progress";
          const elapsedMinutes = Math.floor(pollingAttempt / 12); // Assuming 5-second intervals (12 polls per minute)

          if (status === "completed") {
            process.stdout.write("\n");
            logInfo(`Categorisation completed successfully via polling!`);
            results = statusResponse.data.results || [];
            finalStatus = "completed";
            break; // Exit polling loop
          } else if (status === "failed") {
            process.stdout.write("\n");
            const errorMessage = statusResponse.data.error || "Unknown error during processing";
            logError(`Categorisation failed: ${errorMessage}`);
            const durationMs = Date.now() - operationStartTime;
            return { status: "failed", error: errorMessage, durationMs: durationMs, usedPolling: true }; // Exit function on failure
          } else if (status === "processing") {
            // Log progress occasionally
            if (pollingAttempt % 12 === 0) {
              // Log every minute
              logInfo(`Categorisation still processing... (${elapsedMinutes} min) - ${message}`);
            }
          } else {
            logWarning(`Unexpected status received during polling: ${status}`);
          }
        } catch (error) {
          logError(`Error checking categorisation status: ${error.message}`);
          // Consider adding retry logic here for network errors
          // For now, we'll just continue polling
        }

        await new Promise((resolve) => setTimeout(resolve, 5000)); // Poll every 5 seconds
        pollingAttempt++;
      } // End polling loop

      if (pollingAttempt >= maxPollingAttempts) {
        process.stdout.write("\n");
        logWarning(`Reached maximum number of polling attempts (${maxPollingAttempts}). Assuming failure.`);
        const durationMs = Date.now() - operationStartTime;
        return { status: "timeout", error: "Polling timed out", durationMs: durationMs, usedPolling: true };
      }
      // --- End Polling ---
    } else {
      // Should not happen if initial request handling is correct
      throw new Error("Invalid state after initial categorisation request.");
    }

    // --- Process Results (Common for both Sync and Async) ---
    if (finalStatus === "completed") {
      logInfo(`Status response data: ${JSON.stringify(response.data)}`); // Log the data source (sync or last poll)

      if (results && Array.isArray(results)) {
        logInfo(`Found ${results.length} results.`);

        // Log the results for debugging
        logInfo("Displaying results...");
        logInfo("\n===== CATEGORISATION RESULTS ======");
        logInfo(`Total transactions categorised: ${results.length}`);

        results.forEach((result, index) => {
          const narrative = result.narrative || "N/A";
          const category = result.predicted_category || "Error";
          const score = result.similarity_score;
          const scorePercent = score ? (score * 100).toFixed(2) : "N/A";
          const secondCategory = result.second_predicted_category || "N/A";
          const secondScore = result.second_similarity_score;
          const secondScorePercent = secondScore ? (secondScore * 100).toFixed(2) : "N/A";
          const adjustmentInfo = result.adjustment_info || {}; // Get adjustment info
          const isLlmAssisted = adjustmentInfo.llm_assisted === true;

          let logLine = `${index + 1}. \"${narrative}\"`;

          // Show cleaned narrative if different from original
          const cleaned_narrative = result.cleaned_narrative || null;
          if (cleaned_narrative && cleaned_narrative !== narrative) {
            logLine += `\n     Cleaned: "${cleaned_narrative}"`;
          }
          logLine += ` \u2192 ${category}`;

          // --- LLM Assist Info ---
          if (isLlmAssisted) {
            const llmModel = adjustmentInfo.llm_model || "Unknown LLM";
            const originalCategory = adjustmentInfo.original_embedding_category_name || "Unknown Embedding Cat";
            const originalScore = adjustmentInfo.original_similarity_score;
            const originalScorePercent = originalScore ? (originalScore * 100).toFixed(2) : "N/A";
            logLine += ` (${scorePercent}%) [LLM (${llmModel}) Assisted]`; // Add score for LLM selected category if available (might be 0)
            logLine += `\n     Original Embedding: ${originalCategory} (${originalScorePercent}%)`;
          } else {
            // If Unknown, show the rejected category and the top score
            if (category === "Unknown" && result.debug_info?.rejected_best_category) {
              logLine += ` (Rejected: ${result.debug_info.rejected_best_category} at ${scorePercent}%)`;
            } else {
              // Otherwise, show the score for the accepted category
              logLine += ` (${scorePercent}%)`;
            }
          }
          // --- End LLM Assist Info ---

          // Add second match info only if not LLM assisted (less relevant otherwise)
          if (!isLlmAssisted) {
            logLine += `\n     2nd Match: ${secondCategory} (${secondScorePercent}%)`;
          }

          // Add money direction
          if (result.money_in === true) {
            logLine += " | MONEY_IN";
          } else if (result.money_in === false) {
            logLine += " | MONEY_OUT";
          }

          // --- NEW: Add Debug Info Reason ---
          // Display the debug reason if the category is Unknown
          if (category === "Unknown" && result.debug_info) {
            const debug = result.debug_info;
            let debugReason = debug.reason_code || "Unknown reason";
            let details = "";

            if (debug.reason_code === "LOW_ABS_CONF") {
              details = `Score: ${debug.best_score?.toFixed(2)} < Threshold: ${debug.threshold?.toFixed(2)}`;
            } else if (debug.reason_code === "LOW_REL_CONF") {
              details = `Diff: ${debug.difference?.toFixed(2)} < Threshold: ${debug.threshold?.toFixed(2)} (Top: '${
                debug.best_category
              }'/${debug.best_score?.toFixed(2)}, 2nd: '${
                debug.second_best_category
              }'/${debug.second_best_score?.toFixed(2)})`;
            } else if (debug.reason_code === "CONFLICTING_NEIGHBORS") {
              const neighbors = debug.unique_neighbor_categories || [];
              details = `Neighbors: [${neighbors.join(", ")}]`;
              // Optionally add neighbor scores if available
              if (debug.neighbor_scores && debug.neighbor_categories) {
                const neighborDetails = debug.neighbor_categories
                  .map((cat, idx) => `${cat}: ${debug.neighbor_scores[idx]?.toFixed(2)}`)
                  .join(", ");
                details += ` (Scores: ${neighborDetails})`;
              }
            } else if (debug.reason_code === "DEFAULT") {
              details = `Score: ${debug.best_score?.toFixed(2)} - Defaulted to Unknown`;
            }

            logLine += `\n     DEBUG: Reason: ${debugReason} | ${details}`;
          } else if (category === "Unknown" && result.adjustment_info?.unknown_reason) {
            // Fallback to original reason if debug_info is missing but reason exists
            logLine += ` (${result.adjustment_info.unknown_reason})`;
          }
          // --- End Debug Info ---

          console.log(logLine);
        });
        console.log("=====================================\n");
      } else {
        console.log("\n===== CATEGORISATION RESULTS =====");
        console.log("Results format invalid or results array missing in the final response.");
        console.log("Final Response Data:", JSON.stringify(response.data, null, 2));
        console.log("=====================================\n");
        const durationMs = Date.now() - operationStartTime;
        return {
          status: "failed",
          error: "Invalid results format received",
          durationMs: durationMs,
          usedPolling: usedPolling,
        };
      }
    }

    // Return final status and results
    const durationMs = Date.now() - operationStartTime;
    return { status: finalStatus, results, durationMs: durationMs, usedPolling: usedPolling };
  } catch (error) {
    console.log(`\nCategorisation failed: ${error.message}`);
    logDebug(`Error stack: ${error.stack}`);
    const durationMs = Date.now() - operationStartTime;
    return { status: "failed", error: error.message, durationMs: durationMs, usedPolling: usedPolling };
  }
};

// Test clean_text functionality
const testCleanText = async (apiUrl) => {
  console.log("\n=== Testing clean_text Function ===\n");

  try {
    // 1. Load the training data
    console.log("1. Loading training data...");
    const trainingDataResult = await loadTrainingData("train_woolies_test.csv");
    const trainingData = trainingDataResult.transactions;
    const uniqueTrainingCategories = trainingDataResult.uniqueCategories; // Store unique categories
    console.log(`   Loaded ${trainingData.length} records`);
    console.log(`   Found ${uniqueTrainingCategories.length} unique categories in training data.`); // Log category count

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
  let trainingResult, categorizationResult; // Store results for performance summary
  try {
    console.log("\n=== Starting Test Script ===\n");

    let apiUrl;

    if (process.env.TEST_TARGET_API_URL) {
      console.log("1. Using Target API URL from environment variable...");
      apiUrl = process.env.TEST_TARGET_API_URL;
      console.log(`   API URL: ${apiUrl}\n`);
    } else {
      console.error("❌ ERROR: TEST_TARGET_API_URL environment variable is not set.");
      console.error(
        "Please set it in your .env file to point to your Dockerized API (e.g., http://localhost or http://localhost:3005)."
      );
      process.exit(1);
    }

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
    // const trainingData = await loadTrainingData("training_test.csv");
    // const trainingData = await loadTrainingData("full_train.csv");
    const { transactions: trainingData, uniqueCategories: uniqueTrainingCategories } = await loadTrainingData(
      "nz_train.csv"
    ); // Destructure results
    // const trainingDataResult = await loadTrainingData("train_woolies_test.csv");
    // const trainingData = trainingDataResult.transactions;
    // uniqueTrainingCategories = trainingDataResult.uniqueCategories; // Store unique categories

    // const categorizationData = await loadCategorizationData("categorise_full.csv");
    const categorizationData = await loadCategorizationData("categorise_test.csv");
    console.log(`   Loaded ${trainingData.length} training records`);
    console.log(`   Found ${uniqueTrainingCategories.length} unique categories in training data.`); // Log category count
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
      trainingResult = await trainModel(config); // Store result

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

      // Prepare user_categories for the API call [{id: string, name: string}]
      const userCategoriesForApi = uniqueTrainingCategories.map((name) => ({
        id: name, // Use name as ID for simplicity in test script
        name: name,
      }));
      logDebug(`Prepared ${userCategoriesForApi.length} categories for API: ${JSON.stringify(userCategoriesForApi)}`);

      categorizationResult = await categoriseTransactions(config, userCategoriesForApi); // Pass categories

      if (categorizationResult.status !== "completed") {
        throw new Error(`Categorization failed: ${categorizationResult.error || "Unknown error"}`);
      }
      console.log("   Categorization completed successfully\n");
    }

    // Print Performance Summary before cleanup
    console.log("\n=== Performance Summary ===");
    if (RUN_TRAINING && trainingResult) {
      const trainingDurationSec = (trainingResult.durationMs / 1000).toFixed(2);
      console.log(`   Training took: ${trainingDurationSec}s (Polling: ${trainingResult.usedPolling ? "Yes" : "No"})`);
    }
    if (RUN_CATEGORIZATION && categorizationResult) {
      const catDurationSec = (categorizationResult.durationMs / 1000).toFixed(2);
      console.log(
        `   Categorization took: ${catDurationSec}s (Polling: ${categorizationResult.usedPolling ? "Yes" : "No"})`
      );
    }
    console.log("===========================\n");

    // 8. Cleanup
    console.log("8. Cleaning up...");
    // Only stop flask process if it was started locally AND not using TEST_TARGET_API_URL
    // Removed flaskProcess kill logic

    console.log("\n=== Test Completed Successfully ===\n");
    process.exit(0);
  } catch (error) {
    console.error(`\n❌ ERROR: ${error.message}`);

    // Ensure cleanup on error
    // Removed flaskProcess kill logic in error handling

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
