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
const csv = require("csv-parser");

// Configuration
const API_URL = process.env.API_URL || "http://localhost:3005";
const NZ_USER_ID = process.env.NZ_USER_ID || "nz_model";
const API_KEY = process.env.TEST_API_KEY;

// Set up logging
const log = (message, isError = false) => {
  const timestamp = new Date().toISOString();
  const prefix = isError ? "[ERROR] " : "";
  console.log(`[${timestamp}] ${prefix}${message}`);
};

// Shorthand logging functions
const logError = (message) => log(message, true);
const logInfo = (message) => log(message);

// Log the actual values being used
logInfo(`Using NZ_USER_ID: ${NZ_USER_ID}`);
logInfo(`Using API_KEY: ${API_KEY ? "***" + API_KEY.slice(-4) : "Not set"}`);
logInfo(`Using API_URL: ${API_URL}`);

// Load training data from CSV
const loadTrainingData = (file_name) => {
  return new Promise((resolve, reject) => {
    const results = [];
    const startTime = Date.now();

    fs.createReadStream(path.join(__dirname, "test_data", file_name))
      .pipe(csv())
      .on("data", (data) => {
        // For NZ dataset, we expect "Merchant" and "Category" fields
        const transaction = {
          description: data.description || data.Merchant || data.merchant,
          Category: data.category || data.Category,
        };
        if (transaction.description && transaction.Category) {
          results.push(transaction);
        } else {
          logInfo(`Skipping invalid transaction: ${JSON.stringify(data)}`);
        }
      })
      .on("end", () => {
        const duration = (Date.now() - startTime) / 1000;
        logInfo(`Loaded ${results.length} training records in ${duration.toFixed(1)}s`);

        if (results.length === 0) {
          logError("No valid training records found. Check CSV field names - needs 'Merchant' and 'Category' fields.");
        } else {
          logInfo(`Sample transaction: ${JSON.stringify(results[0])}`);
        }

        resolve(results);
      })
      .on("error", (error) => {
        logError(`Error loading training data: ${error.toString()}`);
        reject(error);
      });
  });
};

// Train model with NZ data
const trainNZModel = async (transactions) => {
  try {
    logInfo("Starting training...");
    console.log("Training model with NZ data...");

    if (transactions.length === 0) {
      logError("Error: No training data found");
      throw new Error("No training data found");
    }

    logInfo(`Processing ${transactions.length} transactions...`);

    // Prepare the payload
    const payload = {
      transactions,
      userId: NZ_USER_ID,
      isDefaultModel: true, // Mark this as a default model for NZ
      country: "NZ",
    };

    // Initialize retry variables
    const maxRetries = 3;
    let retryCount = 0;
    let lastError = null;
    let response = null;

    // Retry loop for the training request
    while (retryCount < maxRetries) {
      try {
        // Add retry attempt to status
        if (retryCount > 0) {
          console.log(`Retrying training request (attempt ${retryCount + 1}/${maxRetries})...`);
          // Simple delay for retries
          await new Promise((resolve) => setTimeout(resolve, Math.pow(2, retryCount) * 1000));
        }

        // Call training endpoint
        logInfo(`Sending training request to ${API_URL}/train`);

        response = await fetch(`${API_URL}/train`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": API_KEY,
            Accept: "application/json",
          },
          body: JSON.stringify(payload),
        });

        const responseStatus = response.status;

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
          logError(`Error response body: ${responseText}`);
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

    // Check if we got a prediction ID
    if (result.prediction_id) {
      logInfo(`Training in progress with prediction ID: ${result.prediction_id}`);
      console.log("Training in progress, please wait...");

      // Store start time for polling
      const startTime = new Date().getTime();

      // Poll for status until completion or timeout
      const pollResult = await pollForTrainingCompletion(result.prediction_id, startTime);

      return pollResult;
    } else {
      logError("Error: No prediction ID received");
      throw new Error("No prediction ID received from server");
    }
  } catch (error) {
    logError(`Training error: ${error.toString()}`);
    return { error: error.toString(), status: "failed" };
  }
};

// Helper function to poll for training completion
const pollForTrainingCompletion = async (predictionId, startTime) => {
  const maxTime = 30 * 60 * 1000; // 30 minutes timeout
  const pollInterval = 5000; // 5 seconds between polls
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
      logInfo(`Checking training status for prediction ID: ${predictionId} (attempt ${pollCount}/${maxPolls})`);
      const response = await fetch(`${API_URL}/status/${predictionId}`, {
        headers: {
          "X-API-Key": API_KEY,
          Accept: "application/json",
        },
      });

      // Handle different response codes
      if (!response.ok) {
        const statusCode = response.status;
        logInfo(`Server returned error code: ${statusCode} for prediction ID: ${predictionId}`);

        // If we get a 502/503/504, the worker might have restarted
        if (statusCode === 502 || statusCode === 503 || statusCode === 504) {
          consecutiveErrors++;
          logInfo(`Worker error (${statusCode}) on attempt ${pollCount}, consecutive errors: ${consecutiveErrors}`);

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
        }
      }
    } catch (error) {
      logError(`Error in polling: ${error.toString()}`);

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

// Main function
const main = async () => {
  try {
    console.log("\n=== Starting NZ Model Training Script ===\n");

    // 1. Load NZ Training Data
    console.log("1. Loading NZ Training Data...");
    const trainingData = await loadTrainingData("nz_train.csv");
    console.log(`   Loaded ${trainingData.length} training records\n`);

    // 2. Train the Model with NZ Data
    console.log("2. Training Model with NZ Data...");
    const trainingResult = await trainNZModel(trainingData);

    if (trainingResult.status !== "completed") {
      throw new Error(`Training failed: ${trainingResult.error || "Unknown error"}`);
    }
    console.log("   Training completed successfully\n");
    console.log(`   Model ID: ${trainingResult.result?.model_id || "Unknown"}`);
    console.log(`   Default model for NZ has been created and can be used for categorizations`);

    console.log("\n=== NZ Model Training Completed Successfully ===\n");
    process.exit(0);
  } catch (error) {
    console.error(`\n❌ ERROR: ${error.message}`);
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
process.on("SIGINT", () => {
  logInfo("Process interrupted");
  process.exit(1);
});

process.on("SIGTERM", () => {
  logInfo("Process terminated");
  process.exit(1);
});
