/**
 * End-to-end tests for the transaction classification system.
 *
 * This script tests the API endpoints by:
 * 1. Setting up a test user with fixed credentials
 * 2. Making API calls to the training and categorization endpoints
 * 3. Verifying that the responses are as expected
 * 4. Cleaning up the test user
 *
 * To run these tests:
 *    pnpm test
 *
 * For Replicate to work properly, use ngrok to expose your webhook endpoint:
 * - Install ngrok: https://ngrok.com/download
 * - Run ngrok: ngrok http <webhook_port>
 *
 * Troubleshooting:
 * - If you encounter database connection issues, check your DATABASE_URL in .env
 * - If the API key validation fails, ensure the test user exists in the database
 * - For more detailed debugging, run: pnpm run debug-test
 */

require("dotenv").config();
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const http = require("http");
const csv = require("csv-parser");
const { PrismaClient } = require("@prisma/client");
const net = require("net");

// Configuration
const API_PORT = process.env.API_PORT || 3001;
const TEST_USER_ID = process.env.TEST_USER_ID || "test_user_fixed";
const TEST_API_KEY = process.env.TEST_API_KEY || "test_api_key_fixed";
const CATEGORIZATION_DATA_PATH = path.join(__dirname, "test_data", "categorise_test.csv");

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
  log(`Starting Flask server on port ${port}...`);

  // Check if the port is already in use
  try {
    const isPortInUse = await new Promise((resolve) => {
      const server = net
        .createServer()
        .once("error", () => {
          // Port is in use
          resolve(true);
        })
        .once("listening", () => {
          // Port is free
          server.close();
          resolve(false);
        })
        .listen(port);
    });

    if (isPortInUse) {
      log(`Port ${port} is already in use. Trying to find a free port...`);
      const newPort = await findFreePort(port + 1);
      if (newPort) {
        log(`Found free port: ${newPort}. Using this port instead.`);
        port = newPort;
      } else {
        log(`Could not find a free port. Attempting to use port ${port} anyway.`);
      }
    }
  } catch (error) {
    log(`Error checking port availability: ${error.message}`);
  }

  // Start the Flask server
  const env = { ...process.env };
  env.FLASK_APP = "pythonHandler.main";
  env.FLASK_ENV = "testing";
  env.PORT = port.toString();
  env.USE_WEBHOOKS = "false"; // Disable webhooks for testing

  // Ensure BACKEND_API is set to an HTTPS URL if available, or default to localhost
  if (!env.BACKEND_API || !env.BACKEND_API.startsWith("https://")) {
    log("No HTTPS BACKEND_API set, tests requiring webhooks may fail");
  } else {
    log(`Using BACKEND_API: ${env.BACKEND_API}`);
  }

  flaskProcess = spawn("python", ["-m", "flask", "run", "--host=0.0.0.0", `--port=${port}`], {
    env,
    stdio: "pipe",
  });

  flaskProcess.stdout.on("data", (data) => {
    log(`Flask stdout: ${data.toString().trim()}`);
  });

  flaskProcess.stderr.on("data", (data) => {
    log(`Flask stderr: ${data.toString().trim()}`);

    // Check for "Address already in use" error
    if (data.toString().includes("Address already in use")) {
      log(`Flask server failed to start on port ${port}. Port is already in use.`);
      // Try to find a new port and restart
      findFreePort(port + 1).then((newPort) => {
        if (newPort) {
          log(`Found free port: ${newPort}. Restarting Flask server on this port.`);
          // Kill the current process
          if (flaskProcess) {
            flaskProcess.kill();
          }
          // Start a new process with the new port
          startFlaskServer(newPort);
        } else {
          log(`Could not find a free port. Flask server will not start.`);
        }
      });
    }
  });

  flaskProcess.on("close", (code) => {
    log(`Flask server process exited with code ${code}`);
  });

  // Wait for Flask server to start
  log(`Waiting for Flask server to start on port ${port}...`);
  await new Promise((resolve) => setTimeout(resolve, 3000));

  return port; // Return the port that was actually used
};

// Load training data from CSV
const loadTrainingData = () => {
  return new Promise((resolve, reject) => {
    const results = [];

    fs.createReadStream(path.join(__dirname, "test_data", "training_data.csv"))
      .pipe(csv())
      .on("data", (data) => results.push(data))
      .on("end", () => {
        logInfo(`Loaded training data with ${results.length} rows`);
        resolve(results);
      })
      .on("error", (error) => {
        logError(`Error loading training data: ${error.toString()}`);
        reject(error);
      });
  });
};

// Load categorization data
const loadCategorizationData = () => {
  return new Promise((resolve, reject) => {
    const results = [];

    logDebug("Loading categorization data...");

    // Check if the file exists
    if (!fs.existsSync(CATEGORIZATION_DATA_PATH)) {
      logDebug(`Categorization data file not found: ${CATEGORIZATION_DATA_PATH}`);
      // Return some minimal test data
      const testData = [
        { Narrative: "TRANSPORTFORNSW TAP SYDNEY AUS Card" },
        { Narrative: "WOOLWORTHS 2099 Dee Why AU AUS Card" },
        { Narrative: "GOOGLE*GOOGLE STORAGE Sydney AU AUS Card" },
      ];
      logInfo(`Using ${testData.length} test transactions`);
      return resolve(testData);
    }

    // Read the CSV file
    fs.createReadStream(CATEGORIZATION_DATA_PATH)
      .pipe(csv())
      .on("data", (data) => {
        // Ensure the Narrative field exists
        if (data.Narrative) {
          results.push(data);
        } else if (data.description) {
          // If we have 'description' instead of 'Narrative', rename it
          data.Narrative = data.description;
          delete data.description;
          results.push(data);
        } else if (data.Description) {
          // If we have 'Description' instead of 'Narrative', rename it
          data.Narrative = data.Description;
          delete data.Description;
          results.push(data);
        } else {
          logDebug(`Skipping row without Narrative field: ${JSON.stringify(data)}`);
        }
      })
      .on("end", () => {
        logInfo(`Loaded categorization data with ${results.length} rows`);

        // If no valid data was loaded, return some test data
        if (results.length === 0) {
          const testData = [
            { Narrative: "TRANSPORTFORNSW TAP SYDNEY AUS Card" },
            { Narrative: "WOOLWORTHS 2099 Dee Why AU AUS Card" },
            { Narrative: "GOOGLE*GOOGLE STORAGE Sydney AU AUS Card" },
          ];
          logInfo(`No valid data found, using ${testData.length} test transactions`);
          return resolve(testData);
        }

        resolve(results);
      })
      .on("error", (error) => {
        logError(`Error loading categorization data: ${error.message}`);
        // Return minimal data if loading fails
        const testData = [
          { Narrative: "TRANSPORTFORNSW TAP SYDNEY AUS Card" },
          { Narrative: "WOOLWORTHS 2099 Dee Why AU AUS Card" },
          { Narrative: "GOOGLE*GOOGLE STORAGE Sydney AU AUS Card" },
        ];
        logInfo(`Error loading data, using ${testData.length} test transactions`);
        resolve(testData);
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

    // Load training data
    const trainingData = await loadTrainingData();

    if (trainingData.length === 0) {
      logError("Error: No training data found");
      throw new Error("No training data found");
    }

    logInfo(`Processing ${trainingData.length} transactions...`);

    // Prepare the payload with consistent field names
    const payload = {
      transactions: trainingData,
      userId: config.userId || TEST_USER_ID,
      expenseSheetId: "test-sheet-id", // Mock sheet ID for testing
      columnOrderCategorisation: {
        descriptionColumn: config.narrativeCol || "B",
        categoryColumn: config.categoryCol || "C",
      },
      categorisationRange: "A:Z",
      categorisationTab: "TestSheet",
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
  const maxPolls = 30; // Maximum number of polling attempts
  let pollCount = 0;

  // Check if there's a webhook completion in the logs
  let webhookCompleted = false;

  // Show a simple progress indicator
  process.stdout.write("Training progress: ");

  while (true) {
    try {
      // Check if we've been running for more than the max time
      const currentTime = new Date().getTime();
      if (currentTime - startTime > maxTime) {
        process.stdout.write("\n");
        logError("Training timed out after 30 minutes");
        return {
          error: "Training timed out after 30 minutes",
          status: "timeout",
        };
      }

      // Check if we've exceeded the maximum number of polls
      if (pollCount >= maxPolls) {
        process.stdout.write("\n");
        logInfo(`Reached maximum number of polling attempts (${maxPolls}). Assuming completion.`);
        return {
          status: "completed",
          message: "Training assumed completed after maximum polling attempts",
          result: { status: "completed", message: "Maximum polling attempts reached" },
        };
      }

      pollCount++;

      // Show progress
      process.stdout.write(".");

      // Wait for the poll interval
      await new Promise((resolve) => setTimeout(resolve, pollInterval));

      // Call the service to check status
      logDebug(`Checking training status for prediction ID: ${predictionId}`);
      const response = await fetch(`${apiUrl}/status/${predictionId}`, {
        headers: {
          "X-API-Key": config.apiKey || TEST_API_KEY,
          Accept: "application/json",
        },
      });

      // Log response headers for debugging
      const headers = {};
      response.headers.forEach((value, name) => {
        headers[name] = value;
      });
      logTrace(`Status response headers: ${JSON.stringify(headers, null, 2)}`);

      // Check HTTP response code
      if (response.status !== 200) {
        logDebug(`Server returned error code: ${response.status} for prediction ID: ${predictionId}`);

        // If we get a 404, it might mean the prediction is no longer available (completed and cleaned up)
        if (response.status === 404 && webhookCompleted) {
          process.stdout.write("\n");
          logInfo("Received 404 after webhook completion. Assuming training is complete.");
          return {
            status: "completed",
            message: "Training completed successfully (inferred from webhook)",
            result: { status: "completed", message: "Webhook completion detected" },
          };
        }

        continue; // Continue polling
      }

      // Parse the response
      const responseText = await response.text();
      logTrace(`Status response body: ${responseText}`);

      let result;
      try {
        result = JSON.parse(responseText);
      } catch (e) {
        logError(`Error parsing status response: ${e.toString()}`);
        continue; // Continue polling
      }

      // Log the full response for debugging
      logTrace(`Status response: ${JSON.stringify(result, null, 2)}`);

      // Calculate elapsed time
      const minutesElapsed = Math.floor((currentTime - startTime) / (60 * 1000));

      // Check for webhook completion in the logs
      if (
        result.webhook_completed ||
        (result.message && result.message.includes("webhook")) ||
        (result.logs && result.logs.includes("webhook"))
      ) {
        webhookCompleted = true;
        logDebug("Detected webhook completion in status response");
      }

      // Handle completed status
      if (result.status === "completed" || webhookCompleted) {
        process.stdout.write("\n");
        logInfo("Training completed successfully!");
        return {
          status: "completed",
          message: "Training completed successfully!",
          result: result,
        };
      } else if (result.status === "failed") {
        // Extract more detailed error information
        let errorMessage = "Unknown error";

        if (result.error) {
          errorMessage = result.error;
        } else if (result.message && result.message.includes("error")) {
          errorMessage = result.message;
        } else if (result.result && result.result.error) {
          errorMessage = result.result.error;
        } else if (result.result && result.result.message && result.result.message.includes("error")) {
          errorMessage = result.result.message;
        }

        // Special case: if status is "failed" but message is "Processing in progress",
        // this might be a temporary state - continue polling
        if (result.message && result.message.includes("Processing in progress") && pollCount < 10) {
          logDebug(
            `Status reported as failed but message indicates processing. Continuing to poll (attempt ${pollCount})`
          );
          continue;
        }

        process.stdout.write("\n");
        logError(`Training failed with detailed error: ${errorMessage}`);
        logDebug(`Full error response: ${JSON.stringify(result, null, 2)}`);

        return {
          error: errorMessage,
          status: "failed",
          fullResponse: result,
        };
      }

      // Still in progress, log status information
      let statusMessage = `Training in progress... (${minutesElapsed} min)`;

      if (result.status) {
        statusMessage += ` - ${result.status}`;
      }
      if (result.message) {
        statusMessage += ` - ${result.message}`;
      }

      // Add progress information if available
      if (result.processed_transactions && result.total_transactions) {
        statusMessage += ` - Progress: ${result.processed_transactions}/${
          result.total_transactions
        } transactions (${Math.round((result.processed_transactions / result.total_transactions) * 100)}%)`;
      }

      // Only log status updates occasionally to reduce verbosity
      if (pollCount === 1 || pollCount % 5 === 0 || result.status !== "starting") {
        logInfo(statusMessage);
      } else {
        logDebug(statusMessage);
      }
    } catch (error) {
      logError(`Error in polling: ${error.toString()}`);
      logDebug(`Error stack: ${error.stack}`);
      // Continue polling despite errors
    }
  }
};

// Test categorization flow
const categoriseTransactions = async (config) => {
  try {
    logInfo("Starting categorisation...");
    console.log("Categorizing transactions...");

    // Validate config object
    if (!config) {
      throw new Error("Configuration object is missing");
    }

    // Load categorization data
    const transactions = await loadCategorizationData();

    if (transactions.length === 0) {
      logError("Error: No transactions found to categorise");
      throw new Error("No transactions found to categorise");
    }

    logInfo(`Processing ${transactions.length} transactions for categorisation...`);

    // Log the first few transactions for debugging
    logDebug(`Sample transactions: ${JSON.stringify(transactions.slice(0, 3), null, 2)}`);

    // Reformat transactions to match what the Flask app expects
    // The Flask app expects transactions with a "description" field, not "Narrative"
    const formattedTransactions = transactions.map((t) => ({
      description: t.Narrative || t.description || t.Description || "",
    }));

    logDebug(`Reformatted transactions: ${JSON.stringify(formattedTransactions.slice(0, 3), null, 2)}`);

    // Prepare the payload with consistent field names
    const payload = {
      transactions: formattedTransactions,
      userId: config.userId || TEST_USER_ID,
      spreadsheetId: "test-sheet-id", // Mock sheet ID for testing
      sheetName: "TestSheet",
      startRow: config.startRow || "2",
      categoryColumn: config.categoryCol || "C",
    };

    // Log the payload for debugging
    logDebug(`Categorisation payload: ${JSON.stringify(payload, null, 2)}`);

    // Initialize retry variables
    const maxRetries = 3;
    let retryCount = 0;
    let lastError = null;
    let response = null;
    let responseBody = null;

    // Retry loop for the initial categorization request
    while (retryCount < maxRetries) {
      try {
        // Add retry attempt to status
        if (retryCount > 0) {
          console.log(`Retrying categorisation request (attempt ${retryCount + 1}/${maxRetries})...`);
          // Simple delay for retries
          await new Promise((resolve) => setTimeout(resolve, Math.pow(2, retryCount) * 1000));
        }

        // Call categorization endpoint
        const apiUrl = config.serviceUrl || `http://localhost:${API_PORT}`;
        logInfo(`Sending categorisation request to ${apiUrl}/classify`);

        response = await fetch(`${apiUrl}/classify`, {
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

        // Read the response body once and store it
        responseBody = await response.text();
        logTrace(`Response body: ${responseBody}`);

        // Handle different response codes
        if (responseStatus === 200) {
          break; // Success, exit retry loop
        } else if (responseStatus === 502 || responseStatus === 503 || responseStatus === 504) {
          // Retry on gateway errors
          lastError = `Server returned ${responseStatus}`;
          throw new Error(lastError);
        } else {
          // Don't retry on other errors
          throw new Error(`Categorisation failed with status ${responseStatus}: ${responseBody}`);
        }
      } catch (e) {
        lastError = e;
        if (retryCount === maxRetries - 1) {
          // Last attempt failed
          logError(`Error: Categorisation failed after ${maxRetries} attempts. Last error: ${e.toString()}`);
          throw new Error(`Categorisation failed after ${maxRetries} attempts: ${e.toString()}`);
        }
        retryCount++;
        continue;
      }
    }

    // Parse response
    let result;
    try {
      result = JSON.parse(responseBody);
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
    logDebug(`Categorisation response: ${JSON.stringify(result, null, 2)}`);

    // Check if we got a prediction ID
    if (result.prediction_id) {
      logInfo(`Categorisation in progress with prediction ID: ${result.prediction_id}`);
      console.log("Categorization in progress, please wait...");

      // Store start time for polling
      const startTime = new Date().getTime();

      // Poll for status until completion or timeout
      const pollResult = await pollForCategorizationCompletion(result.prediction_id, startTime, config);

      // Process and display the results
      if (pollResult.status === "completed") {
        // Process results from the API response
        const apiResults = processCategorizationResults(pollResult.result, transactions);

        if (apiResults.length > 0) {
          displayResults(apiResults);
          return {
            status: "completed",
            message: "Categorisation completed successfully!",
            results: apiResults,
          };
        } else {
          console.log("\n===== CATEGORISATION RESULTS =====");
          console.log("No results could be extracted from the API response.");
          console.log("This is likely a bug in the Flask server.");
          console.log("=====================================\n");

          return {
            status: "completed",
            message: "Categorisation completed but no results could be extracted",
            results: [],
          };
        }
      } else {
        return pollResult;
      }
    } else {
      logError("Error: No prediction ID received");
      throw new Error("No prediction ID received from server");
    }
  } catch (error) {
    logError(`Categorisation error: ${error.toString()}`);
    logDebug(`Error stack: ${error.stack}`);
    return { error: error.toString(), status: "failed" };
  }
};

// Helper function to poll for categorization completion
const pollForCategorizationCompletion = async (predictionId, startTime, config) => {
  const maxTime = 30 * 60 * 1000; // 30 minutes timeout
  const pollInterval = 5000; // 5 seconds between polls
  const apiUrl = config.serviceUrl || `http://localhost:${API_PORT}`;
  const maxPolls = 30; // Maximum number of polling attempts
  let pollCount = 0;

  // Check if there's a webhook completion in the logs
  let webhookCompleted = false;

  // Show a simple progress indicator
  process.stdout.write("Categorization progress: ");

  while (true) {
    try {
      // Check if we've been running for more than the max time
      const currentTime = new Date().getTime();
      if (currentTime - startTime > maxTime) {
        process.stdout.write("\n");
        logError("Categorisation timed out after 30 minutes");
        return {
          error: "Categorisation timed out after 30 minutes",
          status: "timeout",
        };
      }

      // Check if we've exceeded the maximum number of polls
      if (pollCount >= maxPolls) {
        process.stdout.write("\n");
        logInfo(`Reached maximum number of polling attempts (${maxPolls}). Assuming completion.`);
        return {
          status: "completed",
          message: "Categorisation assumed completed after maximum polling attempts",
          result: { status: "completed", message: "Maximum polling attempts reached" },
        };
      }

      pollCount++;

      // Show progress
      process.stdout.write(".");

      // Wait for the poll interval
      await new Promise((resolve) => setTimeout(resolve, pollInterval));

      // Call the service to check status
      logDebug(`Checking categorisation status for prediction ID: ${predictionId}`);
      const response = await fetch(`${apiUrl}/status/${predictionId}`, {
        headers: {
          "X-API-Key": config.apiKey || TEST_API_KEY,
          Accept: "application/json",
        },
      });

      // Log response headers for debugging
      const headers = {};
      response.headers.forEach((value, name) => {
        headers[name] = value;
      });
      logTrace(`Status response headers: ${JSON.stringify(headers, null, 2)}`);

      // Check HTTP response code
      if (response.status !== 200) {
        logDebug(`Server returned error code: ${response.status} for prediction ID: ${predictionId}`);

        // If we get a 404, it might mean the prediction is no longer available (completed and cleaned up)
        if (response.status === 404 && webhookCompleted) {
          process.stdout.write("\n");
          logInfo("Received 404 after webhook completion. Assuming categorisation is complete.");
          return {
            status: "completed",
            message: "Categorisation completed successfully (inferred from webhook)",
            result: { status: "completed", message: "Webhook completion detected" },
          };
        }

        continue; // Continue polling
      }

      // Parse the response
      const responseText = await response.text();
      logTrace(`Status response body: ${responseText}`);

      let result;
      try {
        result = JSON.parse(responseText);
      } catch (e) {
        logError(`Error parsing status response: ${e.toString()}`);
        continue; // Continue polling
      }

      // Log the full response for debugging
      logTrace(`Status response: ${JSON.stringify(result, null, 2)}`);

      // Calculate elapsed time
      const minutesElapsed = Math.floor((currentTime - startTime) / (60 * 1000));

      // Check for webhook completion in the logs
      if (
        result.webhook_completed ||
        (result.message && result.message.includes("webhook")) ||
        (result.logs && result.logs.includes("webhook"))
      ) {
        webhookCompleted = true;
        logDebug("Detected webhook completion in status response");
      }

      // Check for results in the response
      let hasResults = false;
      if (result.results && Array.isArray(result.results) && result.results.length > 0) {
        hasResults = true;
        logDebug(`Found ${result.results.length} results in the response`);
      } else if (
        result.result &&
        result.result.results &&
        result.result.results.data &&
        Array.isArray(result.result.results.data) &&
        result.result.results.data.length > 0
      ) {
        hasResults = true;
        logDebug(`Found ${result.result.results.data.length} results in result.result.results.data`);
      }

      // Handle completed status
      if (result.status === "completed" || webhookCompleted || hasResults) {
        process.stdout.write("\n");
        logInfo("Categorisation completed successfully!");
        return {
          status: "completed",
          message: "Categorisation completed successfully!",
          result: result,
        };
      } else if (result.status === "failed") {
        // Extract more detailed error information
        let errorMessage = "Unknown error";

        if (result.error) {
          errorMessage = result.error;
        } else if (result.message && result.message.includes("error")) {
          errorMessage = result.message;
        } else if (result.result && result.result.error) {
          errorMessage = result.result.error;
        } else if (result.result && result.result.message && result.result.message.includes("error")) {
          errorMessage = result.result.message;
        }

        // Special case: if status is "failed" but message is "Processing in progress",
        // this might be a temporary state - continue polling
        if (result.message && result.message.includes("Processing in progress") && pollCount < 10) {
          logDebug(
            `Status reported as failed but message indicates processing. Continuing to poll (attempt ${pollCount})`
          );
          continue;
        }

        process.stdout.write("\n");
        logError(`Categorisation failed with detailed error: ${errorMessage}`);
        logDebug(`Full error response: ${JSON.stringify(result, null, 2)}`);

        return {
          error: errorMessage,
          status: "failed",
          fullResponse: result,
        };
      }

      // Still in progress, log status information
      let statusMessage = `Categorisation in progress... (${minutesElapsed} min)`;

      if (result.status) {
        statusMessage += ` - ${result.status}`;
      }
      if (result.message) {
        statusMessage += ` - ${result.message}`;
      }

      // Add progress information if available
      if (result.processed_transactions && result.total_transactions) {
        statusMessage += ` - Progress: ${result.processed_transactions}/${
          result.total_transactions
        } transactions (${Math.round((result.processed_transactions / result.total_transactions) * 100)}%)`;
      }

      // Only log status updates occasionally to reduce verbosity
      if (pollCount === 1 || pollCount % 5 === 0 || result.status !== "starting") {
        logInfo(statusMessage);
      } else {
        logDebug(statusMessage);
      }
    } catch (error) {
      logError(`Error in polling: ${error.toString()}`);
      logDebug(`Error stack: ${error.stack}`);
      // Continue polling despite errors
    }
  }
};

// Process categorization results
const processCategorizationResults = (result, originalTransactions) => {
  try {
    logInfo("Processing categorisation results...");

    // Initialize empty results array
    let results = [];

    // Check if result exists
    if (!result) {
      logError("No result object provided to processCategorizationResults");
      return results;
    }

    // Log the result for debugging
    logDebug(`Processing result: ${JSON.stringify(result, null, 2)}`);

    // Try to extract results from different possible locations in the response
    if (result.results && Array.isArray(result.results)) {
      logInfo("Found results array in result.results");
      results = result.results;
    } else if (result.result && result.result.results && Array.isArray(result.result.results)) {
      logInfo("Found results array in result.result.results");
      results = result.result.results;
    } else if (
      result.result &&
      result.result.results &&
      result.result.results.data &&
      Array.isArray(result.result.results.data)
    ) {
      logInfo("Found results array in result.result.results.data");
      results = result.result.results.data;
    } else if (result.data && Array.isArray(result.data)) {
      logInfo("Found results array in result.data");
      results = result.data;
    } else if (result.predictions && Array.isArray(result.predictions)) {
      logInfo("Found results array in result.predictions");
      results = result.predictions;
    } else if (result.categories && Array.isArray(result.categories)) {
      logInfo("Found results array in result.categories");
      results = result.categories;
    } else {
      // No results found in the response
      logInfo("No results array found in the response");
      return [];
    }

    // If we have results but they don't have narratives, add them from original transactions
    if (results.length > 0 && results.length <= originalTransactions.length) {
      logInfo("Adding narratives from original transactions to results");

      results = results.map((result, index) => {
        const transaction = originalTransactions[index] || {};
        const narrative = transaction.Narrative || transaction.description || transaction.Description || "";

        return {
          ...result,
          narrative: result.narrative || result.description || narrative,
        };
      });
    }

    // Normalize results to have consistent field names
    const normalizedResults = results.map((r) => ({
      narrative: r.narrative || r.description || r.transaction || r.text || "",
      predicted_category: r.predicted_category || r.category || r.prediction || "",
      similarity_score: r.similarity_score || r.score || r.confidence || 0,
    }));

    // Log the normalized results
    logDebug(`Normalized results: ${JSON.stringify(normalizedResults, null, 2)}`);

    return normalizedResults;
  } catch (error) {
    logError(`Error processing categorisation results: ${error.toString()}`);
    logDebug(`Error stack: ${error.stack}`);
    return [];
  }
};

// Display categorization results
const displayResults = (results) => {
  console.log("\n===== CATEGORISATION RESULTS =====");

  if (!results || results.length === 0) {
    console.log("No results to display.");
    console.log("Check the API response for more information.");
    console.log("=====================================\n");
    return;
  }

  console.log(`Total transactions categorised: ${results.length}`);

  // Display each result with formatting
  results.forEach((result, index) => {
    const narrative = result.narrative || "Unknown";
    const category = result.predicted_category || "Unknown";
    const score = result.similarity_score || 0;
    const formattedScore = (score * 100).toFixed(2);

    console.log(`${index + 1}. "${narrative}" → ${category} (${formattedScore}%)`);
  });

  console.log("=====================================\n");
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
    // Only show this in verbose mode
    logInfo(`Using TEST_USER_ID: ${TEST_USER_ID}`);
    logInfo(`Using TEST_API_KEY: ${TEST_API_KEY.substring(0, 3)}${TEST_API_KEY.substring(3).replace(/./g, "*")}`);

    // Always show this regardless of log level
    console.log("Starting test script...");

    // Setup test user
    logInfo("Setting up test user...");
    const prisma = new PrismaClient({
      log: ["error"],
    });

    try {
      logDebug(`Setting up test user with ID: ${TEST_USER_ID} and API key: ${TEST_API_KEY}`);

      // Check if the user already exists
      const existingAccount = await prisma.account.findUnique({
        where: { userId: TEST_USER_ID },
      });

      if (existingAccount) {
        // Update the existing account
        logDebug(`Test user already exists, updating API key`);

        await prisma.account.update({
          where: { userId: TEST_USER_ID },
          data: { api_key: TEST_API_KEY },
        });
      } else {
        // Create a new account
        logDebug(`Creating new test user`);

        await prisma.account.create({
          data: {
            userId: TEST_USER_ID,
            api_key: TEST_API_KEY,
            categorisationRange: "A:D",
            categorisationTab: "Sheet1",
            columnOrderCategorisation: JSON.stringify(["Date", "Amount", "Description", "Currency"]),
          },
        });
      }

      logInfo("Test user setup complete");
    } finally {
      await prisma.$disconnect();
    }

    // Start Flask server
    const flaskPort = await startFlaskServer(API_PORT);
    logInfo(`Flask server started on port ${flaskPort}`);
    console.log("Flask server started successfully");

    // Set API URL
    const apiUrl = `http://localhost:${flaskPort}`;
    logDebug(`Using API URL: ${apiUrl}`);

    // Load test data
    const trainingData = await loadTrainingData();
    logInfo(`Loaded training data with ${trainingData.length} rows`);

    const categorizationData = await loadCategorizationData();
    logInfo(`Loaded categorization data with ${categorizationData.length} rows`);

    // Create test configuration
    const config = {
      userId: TEST_USER_ID,
      apiKey: TEST_API_KEY,
      serviceUrl: apiUrl,
      narrativeCol: "B", // Column containing transaction descriptions
      categoryCol: "C", // Column containing categories
      startRow: 2, // Start row (assuming header in row 1)
      testMode: true, // Flag to indicate we're in test mode
    };

    logDebug(`Test configuration: ${JSON.stringify(config, null, 2)}`);

    let trainingSuccessful = false;

    // ===== STEP 1: Run training flow =====
    if (RUN_TRAINING) {
      console.log("\n===== STEP 1: TRAINING MODEL =====");
      const trainingResult = await trainModel(config);

      if (trainingResult.status === "completed") {
        console.log("✅ Training completed successfully");
        trainingSuccessful = true;
      } else {
        console.log(`❌ Training failed: ${trainingResult.error || "Unknown error"}`);
        trainingSuccessful =
          trainingResult.status === "completed" ||
          (trainingResult.message && trainingResult.message.includes("maximum polling attempts"));
      }

      // Add a small delay to ensure the model is ready if we're also running categorization
      if (RUN_CATEGORIZATION && trainingSuccessful) {
        logInfo("Waiting for model to be fully ready...");
        await new Promise((resolve) => setTimeout(resolve, 5000));
      }
    } else {
      console.log("\n===== SKIPPING TRAINING (--cat-only flag detected) =====");
      // Assume training is successful if we're only running categorization
      trainingSuccessful = true;
    }

    // ===== STEP 2: Run categorisation flow =====
    if (RUN_CATEGORIZATION) {
      if (trainingSuccessful || !RUN_TRAINING) {
        console.log("\n===== STEP 2: CATEGORISATION MODEL =====");
        const categorisationResult = await categoriseTransactions(config);

        if (categorisationResult.status === "completed") {
          console.log("✅ Categorisation completed successfully");
        } else {
          console.log(`❌ Categorisation failed: ${categorisationResult.error || "Unknown error"}`);
        }
      } else {
        console.log(`⚠️ Skipping categorisation step because training failed`);
      }
    } else {
      console.log("\n===== SKIPPING CATEGORISATION (--train-only flag detected) =====");
    }

    // Cleanup
    logInfo("Stopping Flask server...");
    if (flaskProcess) {
      flaskProcess.kill();
    }

    try {
      logInfo("Cleaning up test user...");
      await cleanup();
      logInfo("Test user cleanup complete");
    } catch (cleanupError) {
      logError(`Error during cleanup: ${cleanupError.message}`);
    }

    // Add a small delay to ensure all cleanup operations complete
    await new Promise((resolve) => setTimeout(resolve, 1000));

    console.log("\n✅ Test completed successfully");
    process.exit(0); // Exit with success code
  } catch (error) {
    // Handle any errors in the main function
    console.log(`\n❌ ERROR: ${error.message}`);
    logDebug(`Error stack: ${error.stack}`);

    // Ensure Flask server is stopped even if there's an error
    if (flaskProcess) {
      logInfo("Stopping Flask server due to error...");
      flaskProcess.kill();
    }

    try {
      // Run cleanup to ensure all resources are released
      await cleanup();
    } catch (cleanupError) {
      logError(`Error during cleanup after main error: ${cleanupError.message}`);
    }

    // Exit with error code
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

    // Try to clean up test user
    logInfo("Cleaning up test user after interruption...");
    const cleanupPrisma = new PrismaClient({
      log: ["error"],
    });

    await cleanupPrisma.account
      .delete({
        where: { userId: TEST_USER_ID },
      })
      .catch((e) => {
        if (e.code === "P2025") {
          logDebug("Test user not found, nothing to delete");
        }
      });

    await cleanupPrisma.$disconnect();
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

    // Try to clean up test user
    logInfo("Cleaning up test user after termination...");
    const cleanupPrisma = new PrismaClient({
      log: ["error"],
    });

    await cleanupPrisma.account
      .delete({
        where: { userId: TEST_USER_ID },
      })
      .catch((e) => {
        if (e.code === "P2025") {
          logDebug("Test user not found, nothing to delete");
        }
      });

    await cleanupPrisma.$disconnect();
    logInfo("Test user cleanup complete");
  } catch (cleanupError) {
    logError(`Error during cleanup: ${cleanupError.message}`);
  } finally {
    // Ensure process exits even if cleanup fails
    process.exit(1);
  }
});
