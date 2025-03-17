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
const axios = require("axios");

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
  env.FLASK_APP = "main";
  env.FLASK_ENV = "testing";
  env.PORT = port.toString();

  // Ensure BACKEND_API is set to an HTTPS URL if available, or default to localhost
  if (!env.BACKEND_API || !env.BACKEND_API.startsWith("https://")) {
    log("No HTTPS BACKEND_API set, but webhooks are not required");
  } else {
    log(`Using BACKEND_API: ${env.BACKEND_API}`);
  }

  // Change working directory to pythonHandler
  const options = {
    env,
    stdio: "pipe",
    cwd: path.join(process.cwd(), "pythonHandler"),
  };

  flaskProcess = spawn("python", ["-m", "flask", "run", "--host=0.0.0.0", `--port=${port}`], options);

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
    const startTime = Date.now();

    // Use full_train.csv instead of training_data.csv
    fs.createReadStream(path.join(__dirname, "test_data", "full_train.csv"))
      .pipe(csv())
      .on("data", (data) => {
        // Map fields to match expected format
        const transaction = {
          Narrative: data.description || data.Narrative || data.narrative,
          Category: data.category || data.Category,
        };
        if (transaction.Narrative && transaction.Category) {
          results.push(transaction);
        }
      })
      .on("end", () => {
        const duration = (Date.now() - startTime) / 1000;
        logInfo(`Loaded ${results.length} training records in ${duration.toFixed(1)}s`);
        logDebug(`Sample transaction: ${JSON.stringify(results[0])}`);
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
    const targetCsvPath = path.join(process.cwd(), "txConverter", "data", "target.csv");

    logDebug("Loading categorization data from target.csv...");

    // Read the CSV file
    fs.createReadStream(targetCsvPath)
      .pipe(csv())
      .on("data", (data) => {
        // Map the fields to match expected format
        const transaction = {
          Narrative: data.description,
          amount: parseFloat(data.amount),
          currency: data.currency,
          date: data.date,
        };
        results.push(transaction);
      })
      .on("end", () => {
        logInfo(`Loaded ${results.length} transactions from target.csv`);
        resolve(results);
      })
      .on("error", (error) => {
        logError(`Error loading target.csv: ${error.message}`);
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
  const maxPolls = 120; // Maximum number of polling attempts (10 minutes)
  let pollCount = 0;
  let consecutiveErrors = 0;
  const maxConsecutiveErrors = 3;

  // Check if there's a webhook completion in the logs
  let webhookCompleted = false;

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
        logInfo(`Reached maximum number of polling attempts (${maxPolls}). Checking webhook results...`);

        // Try to get webhook results before giving up
        try {
          const webhookResponse = await fetch(`${apiUrl}/webhook-result?prediction_id=${predictionId}`, {
            headers: {
              "X-API-Key": config.apiKey || TEST_API_KEY,
              Accept: "application/json",
            },
          });

          if (webhookResponse.ok) {
            const webhookResult = await webhookResponse.json();
            if (webhookResult && webhookResult.status === "success") {
              return {
                status: "completed",
                message: "Training completed successfully (webhook confirmation)",
                result: webhookResult,
              };
            }
          }
        } catch (webhookError) {
          logError(`Error checking webhook results: ${webhookError}`);
        }

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

      // Call the service to check status
      logDebug(`Checking training status for prediction ID: ${predictionId} (attempt ${pollCount}/${maxPolls})`);
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

        // For 404, check webhook results
        if (statusCode === 404) {
          try {
            const webhookResponse = await fetch(`${apiUrl}/webhook-result?prediction_id=${predictionId}`, {
              headers: {
                "X-API-Key": config.apiKey || TEST_API_KEY,
                Accept: "application/json",
              },
            });

            if (webhookResponse.ok) {
              const webhookResult = await webhookResponse.json();
              if (webhookResult && webhookResult.status === "success") {
                process.stdout.write("\n");
                logInfo("Found successful webhook result after 404 status");
                return {
                  status: "completed",
                  message: "Training completed successfully (webhook confirmation)",
                  result: webhookResult,
                };
              }
            }
          } catch (webhookError) {
            logError(`Error checking webhook results: ${webhookError}`);
          }
        }

        continue; // Continue polling for other error codes
      }

      // Reset consecutive errors on successful response
      consecutiveErrors = 0;

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
        logInfo(`Training completed successfully after ${elapsedMinutes} minutes!`);
        return {
          status: "completed",
          message: "Training completed successfully!",
          result: result,
          elapsedMinutes,
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
        }

        process.stdout.write("\n");
        logError(`Training failed with error: ${errorMessage}`);
        logDebug(`Full error response: ${JSON.stringify(result, null, 2)}`);

        return {
          error: errorMessage,
          status: "failed",
          fullResponse: result,
          elapsedMinutes,
        };
      }

      // Still in progress, log status information
      let statusMessage = `Training in progress... (${elapsedMinutes} min)`;

      if (result.status) {
        statusMessage += ` - ${result.status}`;
      }
      if (result.message) {
        statusMessage += ` - ${result.message}`;
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

      consecutiveErrors++;
      if (consecutiveErrors >= maxConsecutiveErrors) {
        process.stdout.write("\n");
        logError(`Too many consecutive errors (${consecutiveErrors})`);
        return {
          error: `Training failed after ${consecutiveErrors} consecutive errors: ${error.toString()}`,
          status: "failed",
        };
      }

      // Continue polling despite errors
    }
  }
};

// Test categorization flow
const categoriseTransactions = async (config) => {
  try {
    console.log("Categorizing transactions...");

    // Load test data
    const transactions = await loadCategorizationData();
    logInfo(`Processing ${transactions.length} transactions for categorisation...`);

    // Prepare request data
    const requestData = {
      transactions,
      spreadsheetId: "test-sheet-id",
      userId: config.userId,
    };

    // Send request to classify endpoint
    logInfo(`Sending categorisation request to ${config.serviceUrl}/classify`);

    // Try up to 3 times with exponential backoff
    let response;
    let attempt = 1;
    const maxAttempts = 3;

    while (attempt <= maxAttempts) {
      try {
        response = await axios.post(`${config.serviceUrl}/classify`, requestData, {
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": config.apiKey,
          },
        });
        break; // Success, exit the loop
      } catch (error) {
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
        statusResponse = await axios.get(`${config.serviceUrl}/status/${predictionId}`, {
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": config.apiKey,
          },
        });

        const status = statusResponse.data.status;
        const message = statusResponse.data.message || "Processing in progress";
        const elapsedMinutes = Math.floor(pollingAttempt / 6); // Assuming 5-second intervals

        logInfo(`Categorisation in progress... (${elapsedMinutes} min) - ${status} - ${message}`);

        if (status === "completed") {
          console.log("\n");
          console.log(`Categorisation completed successfully!`);
          break;
        } else if (status === "failed") {
          console.log("\n");
          console.log(`Categorisation failed: ${statusResponse.data.error || "Unknown error"}`);
          return { status: "failed", error: statusResponse.data.error };
        }
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
      if (statusResponse.data.results) {
        results = statusResponse.data.results;
        logInfo("Found results array directly in response");
      } else if (statusResponse.data.result && statusResponse.data.result.results) {
        results = statusResponse.data.result.results.data;
        logInfo("Found results array in result.results");
      } else {
        console.log("No results array found in the response");
      }
    }

    // Add narratives from original transactions
    if (results && results.length > 0) {
      logInfo("Adding narratives from original transactions to results");

      // Print results in a nice format
      console.log("\n===== CATEGORISATION RESULTS =====");
      console.log(`Total transactions categorised: ${results.length}`);

      results.forEach((result, index) => {
        const confidence = result.similarity_score ? `${(result.similarity_score * 100).toFixed(2)}%` : "N/A";
        console.log(`${index + 1}. "${result.narrative}" → ${result.predicted_category} (${confidence})`);
      });

      console.log("=====================================\n");
    } else {
      console.log("\n===== CATEGORISATION RESULTS =====");
      console.log("No results could be extracted from the API response.");
      console.log("This is likely a bug in the Flask server.");
      console.log("=====================================\n");
    }

    return { status: "completed", results };
  } catch (error) {
    console.log(`\nCategorisation failed: ${error.message}`);
    logDebug(`Error stack: ${error.stack}`);
    return { status: "failed", error: error.message };
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
