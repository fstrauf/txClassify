/**
 * End-to-end tests for the transaction classification system.
 *
 * This script tests the API endpoints by:
 * 1. Making API calls to the training and categorization endpoints
 * 2. Verifying that the responses are as expected
 *
 * To run these tests successfully:
 * 1. First, check database connection:
 *    pnpm run test-db
 *
 * 2. Create a test user in the database:
 *    pnpm run setup-test-user
 *
 * 3. Set the TEST_USER_ID and TEST_API_KEY environment variables:
 *    export TEST_USER_ID=test_user_123
 *    export TEST_API_KEY=test_key_abc123
 *
 * 4. Run the tests:
 *    pnpm test
 *
 * 5. After testing, clean up the test user:
 *    pnpm run cleanup-test-user
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
const axios = require("axios");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const http = require("http");
const express = require("express");
const bodyParser = require("body-parser");
const csv = require("csv-parser");

// Configuration
const API_PORT = process.env.API_PORT || 3001;
const API_URL = process.env.API_URL || `http://localhost:${API_PORT}`;
const TEST_USER_ID = process.env.TEST_USER_ID || "test_user_123";
const TEST_API_KEY = process.env.TEST_API_KEY || "test_api_key";
const WEBHOOK_PORT = process.env.WEBHOOK_PORT || 3002;

// Global variables
let webhookServer;
let webhookCallbacks = {};
let flaskProcess;

// Set up logging
const log = (message) => {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${message}`);
};

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

// Helper function to get ngrok URL
const getNgrokUrl = async (port) => {
  try {
    const response = await axios.get("http://localhost:4040/api/tunnels");
    if (response.status === 200) {
      const tunnels = response.data.tunnels;

      // First try to find a tunnel for our specific port
      for (const tunnel of tunnels) {
        if (tunnel.config.addr.endsWith(port.toString())) {
          return `${tunnel.public_url}/webhook`;
        }
      }

      // If no specific port tunnel found, use the first available tunnel
      if (tunnels.length > 0) {
        log(`No tunnel found for port ${port}, using first available tunnel: ${tunnels[0].public_url}`);
        return `${tunnels[0].public_url}/webhook`;
      }
    }
    log(`Ngrok is running but no tunnels found`);
    return null;
  } catch (error) {
    log(`Failed to get ngrok URL: ${error.message}`);
    return null;
  }
};

// Start ngrok if not already running
const startNgrok = async (port) => {
  try {
    // First check if ngrok is already running with our port
    let url = await getNgrokUrl(port);
    if (url) {
      log(`Ngrok already running for port ${port}`);
      return url;
    }

    // Start ngrok
    log(`Starting ngrok for port ${port}...`);
    const ngrokProcess = spawn("ngrok", ["http", port.toString()], {
      detached: true,
      stdio: "ignore",
    });

    // Don't wait for ngrok process to exit
    ngrokProcess.unref();

    // Wait for ngrok to start and get the URL
    log("Waiting for ngrok to start...");
    let attempts = 0;
    const maxAttempts = 10;

    while (attempts < maxAttempts) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      url = await getNgrokUrl(port);
      if (url) {
        log(`Ngrok started successfully with URL: ${url}`);
        return url;
      }
      attempts++;
      log(`Waiting for ngrok to start (attempt ${attempts}/${maxAttempts})...`);
    }

    log("Failed to start ngrok after multiple attempts");
    return null;
  } catch (error) {
    log(`Error starting ngrok: ${error.message}`);
    return null;
  }
};

// Start Flask server
const startFlaskServer = async (port) => {
  log(`Starting Flask server on port ${port}...`);

  // Start the Flask server
  const env = { ...process.env };
  env.FLASK_APP = "pythonHandler.main";
  env.FLASK_ENV = "testing";
  env.PORT = port.toString();

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
  });

  // Give the server time to start
  log(`Waiting for Flask server to start on port ${port}...`);
  await new Promise((resolve) => setTimeout(resolve, 5000));

  return flaskProcess;
};

// Start webhook server
const startWebhookServer = async (port) => {
  const app = express();
  app.use(bodyParser.json());

  app.post("/webhook", (req, res) => {
    const data = req.body;
    log(`Webhook received: ${JSON.stringify(data)}`);

    // If there's a prediction ID, call the callback
    if (data.id && webhookCallbacks[data.id]) {
      webhookCallbacks[data.id](data);
      delete webhookCallbacks[data.id];
    }

    res.json({ status: "success" });
  });

  return new Promise((resolve) => {
    webhookServer = app.listen(port, () => {
      log(`Webhook server started on port ${port}`);
      resolve(webhookServer);
    });
  });
};

// Load training data
const loadTrainingData = () => {
  return new Promise((resolve) => {
    const results = [];
    fs.createReadStream(path.join(__dirname, "pythonHandler", "test_data", "training_data.csv"))
      .pipe(csv())
      .on("data", (data) => results.push(data))
      .on("end", () => {
        log(`Loaded training data with ${results.length} rows`);
        resolve(results);
      })
      .on("error", (error) => {
        log(`Error loading training data: ${error.message}`);
        // Return minimal data if loading fails
        resolve([{ Category: "Test", Narrative: "Test transaction" }]);
      });
  });
};

// Load categorization data
const loadCategorizationData = () => {
  return new Promise((resolve) => {
    const results = [];
    fs.createReadStream(path.join(__dirname, "pythonHandler", "test_data", "categorise_test.csv"))
      .pipe(csv())
      .on("data", (data) => {
        // Add Narrative column as a copy of Description if it doesn't exist
        if (!data.Narrative && data.Description) {
          data.Narrative = data.Description;
        }
        results.push(data);
      })
      .on("end", () => {
        log(`Loaded categorization data with ${results.length} rows`);
        resolve(results);
      })
      .on("error", (error) => {
        log(`Error loading categorization data: ${error.message}`);
        // Return minimal data if loading fails
        const minimalData = {
          Date: "30/12/2024",
          Amount: "-134.09",
          Description: "Test transaction",
          Currency: "AUD",
          Narrative: "Test transaction",
        };
        resolve([minimalData]);
      });
  });
};

// Test training flow
const testTrainingFlow = async (trainingData, webhookUrl) => {
  log("Starting training flow test");

  // Generate a unique test spreadsheet ID
  const testSpreadsheetId = `test_${Math.random().toString(36).substring(2, 10)}`;
  log(`Using test spreadsheet ID: ${testSpreadsheetId}`);

  // Prepare request data
  const requestData = {
    userId: TEST_USER_ID,
    spreadsheetId: testSpreadsheetId,
    sheetName: "Sheet1",
    categoryColumn: "Category",
    data: trainingData,
    transactions: trainingData,
    webhook_url: webhookUrl,
  };

  log(`API URL: ${API_URL}`);
  log(`Request data: ${JSON.stringify(requestData, null, 2)}`);
  log(`Using test user ID: ${TEST_USER_ID} and API key: ${TEST_API_KEY ? "***" : "Not set"}`);

  try {
    // Send request to training endpoint
    const response = await axios.post(`${API_URL}/train`, requestData, {
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": TEST_API_KEY,
      },
      timeout: 30000,
    });

    log(`Training response status: ${response.status}`);
    log(`Training response data: ${JSON.stringify(response.data)}`);

    // If we have a prediction ID, wait for the webhook callback
    if (response.data.prediction_id) {
      const predictionId = response.data.prediction_id;
      log(`Waiting for webhook callback for prediction ID: ${predictionId}`);

      // Set up a promise that will be resolved when the webhook is called
      const webhookPromise = new Promise((resolve) => {
        // Set a timeout to resolve the promise after 30 seconds if the webhook is not called
        const timeoutId = setTimeout(() => {
          log(`Webhook timeout for prediction ID: ${predictionId}`);
          resolve(null);
        }, 30000);

        // Set up the callback
        webhookCallbacks[predictionId] = (data) => {
          clearTimeout(timeoutId);
          log(`Webhook called for prediction ID: ${predictionId}`);
          resolve(data);
        };
      });

      // Wait for the webhook to be called
      const webhookData = await webhookPromise;
      if (webhookData) {
        log(`Webhook data: ${JSON.stringify(webhookData)}`);
      }
    }

    return {
      success: true,
      status: response.status,
      data: response.data,
      spreadsheetId: testSpreadsheetId,
    };
  } catch (error) {
    log(`Error during training request: ${error.message}`);

    if (error.response) {
      log(`Training response status: ${error.response.status}`);
      log(`Training response data: ${JSON.stringify(error.response.data)}`);

      if (error.response.status === 401) {
        log(`API key validation failed. Make sure the test user exists in the database and the API key is correct.`);
        log(`Run 'pnpm run setup-test-user' to create a test user.`);
      } else if (error.response.status === 500) {
        log(`Server error. Check the Flask server logs for more details.`);
        if (error.response.data.error && error.response.data.error.includes("Client is not connected")) {
          log(`Database connection error. Make sure the DATABASE_URL is correct and the database is running.`);
          log(`Run 'pnpm run test-db' to test the database connection.`);
        }
      }

      // For testing purposes, we'll accept various status codes
      return {
        success: error.response.status === 200,
        status: error.response.status,
        data: error.response.data,
        spreadsheetId: testSpreadsheetId,
      };
    }

    return {
      success: false,
      error: error.message,
      spreadsheetId: testSpreadsheetId,
    };
  }
};

// Test categorization flow
const testCategorizationFlow = async (categorizationData, spreadsheetId, webhookUrl) => {
  log("Starting categorization flow test");

  // Prepare request data
  const requestData = {
    userId: TEST_USER_ID,
    spreadsheetId: spreadsheetId,
    transactions: categorizationData,
    webhook_url: webhookUrl,
  };

  log(`API URL: ${API_URL}`);
  log(`Request data: ${JSON.stringify(requestData, null, 2)}`);
  log(`Using test user ID: ${TEST_USER_ID} and API key: ${TEST_API_KEY ? "***" : "Not set"}`);

  try {
    // Send request to categorization endpoint
    const response = await axios.post(`${API_URL}/classify`, requestData, {
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": TEST_API_KEY,
      },
      timeout: 30000,
    });

    log(`Categorization response status: ${response.status}`);
    log(`Categorization response data: ${JSON.stringify(response.data)}`);

    // If we have a prediction ID, wait for the webhook callback
    if (response.data.prediction_id) {
      const predictionId = response.data.prediction_id;
      log(`Waiting for webhook callback for prediction ID: ${predictionId}`);

      // Set up a promise that will be resolved when the webhook is called
      const webhookPromise = new Promise((resolve) => {
        // Set a timeout to resolve the promise after 30 seconds if the webhook is not called
        const timeoutId = setTimeout(() => {
          log(`Webhook timeout for prediction ID: ${predictionId}`);
          resolve(null);
        }, 30000);

        // Set up the callback
        webhookCallbacks[predictionId] = (data) => {
          clearTimeout(timeoutId);
          log(`Webhook called for prediction ID: ${predictionId}`);
          resolve(data);
        };
      });

      // Wait for the webhook to be called
      const webhookData = await webhookPromise;
      if (webhookData) {
        log(`Webhook data: ${JSON.stringify(webhookData)}`);
      }
    }

    return {
      success: true,
      status: response.status,
      data: response.data,
    };
  } catch (error) {
    log(`Error during categorization request: ${error.message}`);

    if (error.response) {
      log(`Categorization response status: ${error.response.status}`);
      log(`Categorization response data: ${JSON.stringify(error.response.data)}`);

      if (error.response.status === 401) {
        log(`API key validation failed. Make sure the test user exists in the database and the API key is correct.`);
        log(`Run 'pnpm run setup-test-user' to create a test user.`);
      } else if (error.response.status === 500) {
        log(`Server error. Check the Flask server logs for more details.`);
        if (error.response.data.error && error.response.data.error.includes("Client is not connected")) {
          log(`Database connection error. Make sure the DATABASE_URL is correct and the database is running.`);
          log(`Run 'pnpm run test-db' to test the database connection.`);
        }
      }

      // For testing purposes, we'll accept various status codes
      return {
        success: error.response.status === 200,
        status: error.response.status,
        data: error.response.data,
      };
    }

    return {
      success: false,
      error: error.message,
    };
  }
};

// Cleanup function
const cleanup = () => {
  // Stop Flask server
  if (flaskProcess) {
    log("Stopping Flask server...");
    flaskProcess.kill();
  }

  // Close webhook server
  if (webhookServer) {
    webhookServer.close();
    log("Webhook server closed");
  }
};

// Main function
const main = async () => {
  try {
    log(`Environment variables:`);
    log(`- DATABASE_URL: ${process.env.DATABASE_URL ? "Set (hidden for security)" : "Not set"}`);
    log(`- TEST_USER_ID: ${TEST_USER_ID}`);
    log(`- TEST_API_KEY: ${TEST_API_KEY ? "***" : "Not set"}`);

    // Start webhook server first
    const webhookPort = await findFreePort(WEBHOOK_PORT);
    await startWebhookServer(webhookPort);

    // Start ngrok and get webhook URL
    let webhookUrl = await startNgrok(webhookPort);

    if (webhookUrl) {
      log(`Using ngrok webhook URL: ${webhookUrl}`);
      // Set BACKEND_API environment variable to the ngrok URL base
      const ngrokUrlBase = webhookUrl.replace("/webhook", "");
      process.env.BACKEND_API = ngrokUrlBase;
      log(`Setting BACKEND_API to: ${ngrokUrlBase}`);
    } else {
      // Fallback to local URL with warning
      webhookUrl = `http://localhost:${webhookPort}/webhook`;
      log(`Using local webhook URL: ${webhookUrl}`);
      log("WARNING: Local webhook URL will not work with Replicate. Tests will likely fail.");
      log("Please ensure ngrok is installed and can be started.");
    }

    // Start Flask server after setting up ngrok
    const flaskPort = await findFreePort(API_PORT);
    await startFlaskServer(flaskPort);

    // Update API URL with the actual port
    const actualApiUrl = `http://localhost:${flaskPort}`;
    log(`Using API URL: ${actualApiUrl}`);

    // Load test data
    const trainingData = await loadTrainingData();
    const categorizationData = await loadCategorizationData();

    // Test training flow
    const trainingResult = await testTrainingFlow(trainingData, webhookUrl);

    // Test categorization flow
    const categorizationResult = await testCategorizationFlow(
      categorizationData,
      trainingResult.spreadsheetId,
      webhookUrl
    );

    // Print summary
    log("\n--- Test Summary ---");
    log(`Training: ${trainingResult.success ? "SUCCESS" : "FAILURE"} (Status: ${trainingResult.status})`);
    log(
      `Categorization: ${categorizationResult.success ? "SUCCESS" : "FAILURE"} (Status: ${categorizationResult.status})`
    );

    if (!trainingResult.success || !categorizationResult.success) {
      log("\n--- Troubleshooting ---");
      log("1. Check database connection: pnpm run test-db");
      log("2. Ensure test user exists: pnpm run setup-test-user");
      log("3. Verify environment variables are set correctly");
      log("4. Check Flask server logs for errors");
    }

    // Cleanup
    cleanup();

    // Exit with appropriate code
    process.exit(trainingResult.success && categorizationResult.success ? 0 : 1);
  } catch (error) {
    log(`Error in main function: ${error.message}`);
    log(`Error stack: ${error.stack}`);

    // Cleanup
    cleanup();

    process.exit(1);
  }
};

// Handle process termination
process.on("SIGINT", () => {
  log("Process interrupted");
  cleanup();
  process.exit(1);
});

process.on("SIGTERM", () => {
  log("Process terminated");
  cleanup();
  process.exit(1);
});

// Run the main function
main();
