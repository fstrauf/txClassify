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
const USE_DEV_API = process.argv.includes("--use-dev-api");
const DEV_API_URL = "https://txclassify-dev.onrender.com";

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
        } else {
          logDebug(`Skipping invalid transaction data: ${JSON.stringify(row)}`);
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
    let predictionId = null; // To store prediction ID if async

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
          // Could be synchronous success OR async start (check body)
          const responseText = await response.text();
          logTrace(`Response body (status 200/202): ${responseText}`);
          const result = JSON.parse(responseText);

          if (result.status === "completed") {
            // Synchronous Success!
            console.log("Training completed synchronously.");
            // Return a success object immediately, mimicking pollForTrainingCompletion result
            return {
              status: "completed",
              message: "Training completed successfully (synchronous)!",
              result: result,
              elapsedMinutes: 0, // Indicate immediate completion
            };
          } else if (result.status === "processing" && result.prediction_id) {
            // Asynchronous start indicated by 200 status + processing status in body
            console.log("Training started asynchronously (indicated by 200 response).");
            predictionId = result.prediction_id;
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
            console.log("Training started asynchronously (indicated by 202 response).");
            predictionId = result.prediction_id;
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
      throw new Error(
        `Training request failed after ${maxRetries} attempts: ${lastError?.toString() || "Unknown error"}`
      );
    }

    // Check if we got a prediction ID
    if (predictionId) {
      logInfo(`Training requires polling with prediction ID: ${predictionId}`);
      console.log("Training in progress, please wait...");

      // Store start time for polling
      const startTime = new Date().getTime();

      // Poll for status until completion or timeout
      const pollResult = await pollForTrainingCompletion(predictionId, startTime, config);

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
          console.log("Received synchronous classification results.");
          response = { data: await response_fetch.json(), status: 200 };
          break; // Exit the loop, we have results
        } else if (response_fetch.status === 202) {
          // Asynchronous processing started
          console.log("Classification started asynchronously. Polling required.");
          response = { data: await response_fetch.json(), status: 202 };
          predictionId = response.data.prediction_id;
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
        console.log(`Retrying categorisation request (attempt ${attempt + 1}/${maxAttempts})...`);
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
        console.log(`Synchronous classification completed successfully! Found ${results.length} results.`);
      } else {
        console.error("Synchronous response received, but results are missing or status is not 'completed'.");
        console.error("Response data:", JSON.stringify(response.data, null, 2));
        throw new Error("Invalid synchronous response format.");
      }
    } else if (response.status === 202 && predictionId) {
      // --- Start Polling for Asynchronous Results ---
      console.log(`Categorisation in progress with prediction ID: ${predictionId}`);
      console.log("Polling for results...");
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
            console.log("\n");
            console.log(`Categorisation completed successfully via polling!`);
            results = statusResponse.data.results || [];
            finalStatus = "completed";
            break; // Exit polling loop
          } else if (status === "failed") {
            console.log("\n");
            const errorMessage = statusResponse.data.error || "Unknown error during processing";
            console.log(`Categorisation failed: ${errorMessage}`);
            return { status: "failed", error: errorMessage }; // Exit function on failure
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
        console.log("\n");
        console.log(`Reached maximum number of polling attempts (${maxPollingAttempts}). Assuming failure.`);
        return { status: "timeout", error: "Polling timed out" };
      }
      // --- End Polling ---
    } else {
      // Should not happen if initial request handling is correct
      throw new Error("Invalid state after initial categorisation request.");
    }

    // --- Process Results (Common for both Sync and Async) ---
    if (finalStatus === "completed") {
      console.log("Processing final categorisation results...");
      logInfo(`Status response data: ${JSON.stringify(response.data)}`); // Log the data source (sync or last poll)

      if (results && Array.isArray(results)) {
        logInfo(`Found ${results.length} results.`);

        // Add narratives from original transactions (if needed - check if API returns them)
        // Assuming results contain narrative, category, score, money_in, amount
        if (results.length > 0) {
          logInfo("Displaying results...");

          // Print results in a nice format
          console.log("\n===== CATEGORISATION RESULTS =====");
          console.log(`Total transactions categorised: ${results.length}`);

          results.forEach((result, index) => {
            const narrative = result.narrative || "N/A"; // Handle missing narrative
            const predicted_category = result.predicted_category || "Unknown";
            const confidence = result.similarity_score ? `${(result.similarity_score * 100).toFixed(2)}%` : "N/A";
            const second_predicted_category = result.second_predicted_category || "N/A";
            const second_confidence = result.second_similarity_score
              ? `${(result.second_similarity_score * 100).toFixed(2)}%`
              : "N/A";

            // Add money_in/money_out display
            const direction =
              result.money_in === true ? "MONEY_IN" : result.money_in === false ? "MONEY_OUT" : "UNKNOWN_DIR";

            // Display amount if available
            const amountStr =
              result.amount !== null && result.amount !== undefined ? `$${Math.abs(result.amount).toFixed(2)}` : "";

            // Determine reason if Unknown
            let unknownReason = "";
            if (predicted_category === "Unknown") {
              const score = result.similarity_score;
              const secondScore = result.second_similarity_score;
              // Define thresholds here (matching Python)
              const MIN_ABSOLUTE_CONFIDENCE = 0.85;
              const MIN_RELATIVE_CONFIDENCE_DIFF = 0.05;

              if (score === null || score === undefined || score < MIN_ABSOLUTE_CONFIDENCE) {
                unknownReason = "(Low Absolute Confidence)";
              } else if (
                secondScore !== null &&
                secondScore !== undefined &&
                score - secondScore < MIN_RELATIVE_CONFIDENCE_DIFF
              ) {
                unknownReason = "(Low Relative Confidence)";
              } else {
                unknownReason = "(Conflicting Neighbors)"; // Inferred reason
              }
            }

            console.log(
              `${
                index + 1
              }. \"${narrative}\" → ${predicted_category} (${confidence}) ${unknownReason}\n     2nd Match: ${second_predicted_category} (${second_confidence}) | ${direction} ${amountStr}`
            );
          });

          console.log("=====================================\n");
        } else {
          console.log("\n===== CATEGORISATION RESULTS =====");
          console.log("No results were returned by the API, although status was 'completed'.");
          console.log("=====================================\n");
        }
      } else {
        console.log("\n===== CATEGORISATION RESULTS =====");
        console.log("Results format invalid or results array missing in the final response.");
        console.log("Final Response Data:", JSON.stringify(response.data, null, 2));
        console.log("=====================================\n");
        return { status: "failed", error: "Invalid results format received" };
      }
    }

    // Return final status and results
    return { status: finalStatus, results };
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
  // Stop Flask server only if it was started locally
  if (flaskProcess && !USE_DEV_API) {
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

    let apiUrl;

    if (USE_DEV_API) {
      console.log("1. Using Development API...");
      apiUrl = DEV_API_URL;
      console.log(`   API URL: ${apiUrl}\n`);
    } else {
      // 1. Start Flask Server Locally
      console.log("1. Starting Local Flask Server...");
      const port = await startFlaskServer(API_PORT);
      apiUrl = `http://localhost:${port}`;
      console.log(`   Server URL: ${apiUrl}\n`);
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
    const trainingData = await loadTrainingData("training_test.csv");
    // const trainingData = await loadTrainingData("training_data.csv");
    // const trainingData = await loadTrainingData("training_data_num_cat.csv");
    // const categorizationData = await loadCategorizationData("categorise_test.csv");
    const categorizationData = await loadCategorizationData("categorise_test.csv");
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
    // Only stop flask process if it was started locally
    if (flaskProcess && !USE_DEV_API) {
      flaskProcess.kill();
      console.log("   Flask server stopped");
    }

    console.log("\n=== Test Completed Successfully ===\n");
    process.exit(0);
  } catch (error) {
    console.error(`\n❌ ERROR: ${error.message}`);

    // Ensure cleanup on error
    if (flaskProcess && !USE_DEV_API) {
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
