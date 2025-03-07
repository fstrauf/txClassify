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
const axios = require("axios");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const http = require("http");
const express = require("express");
const bodyParser = require("body-parser");
const csv = require("csv-parser");
const { PrismaClient } = require("@prisma/client");

// Configuration
const API_PORT = process.env.API_PORT || 3001;
const API_URL = process.env.API_URL || `http://localhost:${API_PORT}`;
const TEST_USER_ID = process.env.TEST_USER_ID || "test_user_fixed";
const TEST_API_KEY = process.env.TEST_API_KEY || "test_api_key_fixed";
const WEBHOOK_PORT = process.env.WEBHOOK_PORT || 3002;

// Global variables
let webhookServer;
let webhookCallbacks = {};
let flaskProcess;
let pendingCallbacks = 0; // Track number of pending callbacks

// Set up logging
const log = (message) => {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${message}`);
};

// Log the actual values being used
log(`Using TEST_USER_ID: ${TEST_USER_ID}`);
log(`Using TEST_API_KEY: ${TEST_API_KEY ? "***" + TEST_API_KEY.slice(-4) : "Not set"}`);

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
          return tunnel.public_url;
        }
      }

      // If no specific port tunnel found, use the first available tunnel
      if (tunnels.length > 0) {
        log(`No tunnel found for port ${port}, using first available tunnel: ${tunnels[0].public_url}`);
        return tunnels[0].public_url;
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
  // Increase the JSON body size limit to 50MB
  app.use(bodyParser.json({ limit: "50mb" }));
  app.use(bodyParser.urlencoded({ limit: "50mb", extended: true }));

  // Handler function for webhook callbacks
  const handleWebhook = (req, res) => {
    const data = req.body;
    log(`Webhook received at ${req.path}: ${JSON.stringify(data)}`);
    log(`Webhook query params: ${JSON.stringify(req.query)}`);
    log(`Webhook headers: ${JSON.stringify(req.headers)}`);

    // If there's a prediction ID, call the callback
    if (data.id && webhookCallbacks[data.id]) {
      log(`Found callback for prediction ID: ${data.id}`);
      webhookCallbacks[data.id](data);
      delete webhookCallbacks[data.id];
      pendingCallbacks--;
      log(`Processed webhook callback. Remaining callbacks: ${pendingCallbacks}`);
    } else {
      log(`No callback found for prediction: ${JSON.stringify(data)}`);
    }

    res.json({ status: "success" });
  };

  // Register the handler for multiple paths
  app.post("/webhook", handleWebhook);
  app.post("/train/webhook", handleWebhook);
  app.post("/classify/webhook", handleWebhook);

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
      .pipe(
        csv({
          headers: false,
          mapValues: ({ header, index, value }) => {
            return value.trim();
          },
        })
      )
      .on("data", (data) => {
        // Map the columns to the expected field names
        const mappedData = {
          Date: data[0],
          Amount: data[1],
          description: data[2], // Use lowercase 'description' to match what the backend expects
          Currency: data[3],
        };
        results.push(mappedData);
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
          description: "Test transaction", // Use lowercase 'description'
          Currency: "AUD",
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
        // Set a timeout to resolve the promise after 60 seconds if the webhook is not called
        const timeoutId = setTimeout(() => {
          log(`Webhook timeout for prediction ID: ${predictionId}`);
          resolve(null);
        }, 300000); // Increase to 300000 (5 minutes)

        // Increment pending callbacks counter
        pendingCallbacks++;
        log(`Adding callback for prediction ID: ${predictionId}. Total pending: ${pendingCallbacks}`);

        // Set up the callback
        webhookCallbacks[predictionId] = (data) => {
          clearTimeout(timeoutId);
          log(`Webhook callback received for prediction ID: ${predictionId}`);
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
    }
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
        // Set a timeout to resolve the promise after 60 seconds if the webhook is not called
        const timeoutId = setTimeout(() => {
          log(`Webhook timeout for prediction ID: ${predictionId}`);
          resolve(null);
        }, 300000); // Increase to 300000 (5 minutes)

        // Increment pending callbacks counter
        pendingCallbacks++;
        log(`Adding callback for prediction ID: ${predictionId}. Total pending: ${pendingCallbacks}`);

        // Set up the callback
        webhookCallbacks[predictionId] = (data) => {
          clearTimeout(timeoutId);
          log(`Webhook callback received for prediction ID: ${predictionId}`);
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

    // Set up test user
    log("Setting up test user...");
    const prisma = new PrismaClient({
      log: ["error"],
    });

    try {
      log(`Setting up test user with ID: ${TEST_USER_ID} and API key: ${TEST_API_KEY}`);

      // Check if the user already exists
      const existingAccount = await prisma.account.findUnique({
        where: { userId: TEST_USER_ID },
      });

      if (existingAccount) {
        // Update the existing account
        log(`Test user already exists, updating API key`);

        await prisma.account.update({
          where: { userId: TEST_USER_ID },
          data: { api_key: TEST_API_KEY },
        });
      } else {
        // Create a new account
        log(`Creating new test user`);

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

      log("Test user setup complete");
    } finally {
      await prisma.$disconnect();
    }

    // Start webhook server first
    const webhookPort = await findFreePort(WEBHOOK_PORT);
    await startWebhookServer(webhookPort);

    // Start ngrok and get webhook URL
    let webhookUrl = await startNgrok(webhookPort);

    if (webhookUrl) {
      // Remove the /webhook suffix if present, as the Flask server will add the appropriate path
      webhookUrl = webhookUrl.replace("/webhook", "");
      log(`Using ngrok base URL: ${webhookUrl}`);
      // Set BACKEND_API environment variable to the ngrok URL base
      process.env.BACKEND_API = webhookUrl;
      log(`Setting BACKEND_API to: ${webhookUrl}`);
    } else {
      // Fallback to local URL with warning
      webhookUrl = `http://localhost:${webhookPort}`;
      log(`Using local webhook base URL: ${webhookUrl}`);
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

    // Run tests
    const trainingResult = await testTrainingFlow(trainingData, webhookUrl);
    const categorizationResult = await testCategorizationFlow(
      categorizationData,
      trainingResult.spreadsheetId,
      webhookUrl
    );

    // Print test summary
    log("\n--- Test Summary ---");
    log(`Training: ${trainingResult.success ? "SUCCESS" : "FAILURE"} (Status: ${trainingResult.status})`);
    log(
      `Categorization: ${categorizationResult.success ? "SUCCESS" : "FAILURE"} (Status: ${categorizationResult.status})`
    );

    // Print troubleshooting tips if tests failed
    if (!trainingResult.success || !categorizationResult.success) {
      log("\n--- Troubleshooting ---");
      log("1. Check database connection: pnpm run test-db");
      log("2. Verify environment variables are set correctly");
      log("3. Check Flask server logs for errors");
    }

    // Add a delay before cleanup to ensure all callbacks have time to arrive
    log("Waiting for any pending webhook callbacks...");
    await new Promise((resolve) => setTimeout(resolve, 30000)); // 30 second delay

    // Cleanup
    cleanup();

    // Clean up test user
    log("Cleaning up test user...");
    const cleanupPrisma = new PrismaClient({
      log: ["error"],
    });

    try {
      // Delete the test user
      await cleanupPrisma.account
        .delete({
          where: { userId: TEST_USER_ID },
        })
        .catch((e) => {
          if (e.code === "P2025") {
            log("Test user not found, nothing to delete");
          } else {
            throw e;
          }
        });

      log("Test user cleanup complete");
    } finally {
      await cleanupPrisma.$disconnect();
    }

    // Exit with appropriate code
    process.exit(trainingResult.success && categorizationResult.success ? 0 : 1);
  } catch (error) {
    log(`Error in main function: ${error.message}`);
    log(`Error stack: ${error.stack}`);

    // Cleanup
    cleanup();

    // Try to clean up test user
    try {
      log("Cleaning up test user after error...");
      const cleanupPrisma = new PrismaClient({
        log: ["error"],
      });

      await cleanupPrisma.account
        .delete({
          where: { userId: TEST_USER_ID },
        })
        .catch((e) => {
          if (e.code === "P2025") {
            log("Test user not found, nothing to delete");
          }
        });

      await cleanupPrisma.$disconnect();
      log("Test user cleanup complete");
    } catch (cleanupError) {
      log(`Error during cleanup: ${cleanupError.message}`);
    }

    process.exit(1);
  }
};

// Handle process termination
process.on("SIGINT", async () => {
  log("Process interrupted");
  cleanup();

  // Try to clean up test user
  try {
    log("Cleaning up test user after interruption...");
    const cleanupPrisma = new PrismaClient({
      log: ["error"],
    });

    await cleanupPrisma.account
      .delete({
        where: { userId: TEST_USER_ID },
      })
      .catch((e) => {
        if (e.code === "P2025") {
          log("Test user not found, nothing to delete");
        }
      });

    await cleanupPrisma.$disconnect();
    log("Test user cleanup complete");
  } catch (cleanupError) {
    log(`Error during cleanup: ${cleanupError.message}`);
  }

  process.exit(1);
});

process.on("SIGTERM", async () => {
  log("Process terminated");
  cleanup();

  // Try to clean up test user
  try {
    log("Cleaning up test user after termination...");
    const cleanupPrisma = new PrismaClient({
      log: ["error"],
    });

    await cleanupPrisma.account
      .delete({
        where: { userId: TEST_USER_ID },
      })
      .catch((e) => {
        if (e.code === "P2025") {
          log("Test user not found, nothing to delete");
        }
      });

    await cleanupPrisma.$disconnect();
    log("Test user cleanup complete");
  } catch (cleanupError) {
    log(`Error during cleanup: ${cleanupError.message}`);
  }

  process.exit(1);
});

// Run the main function
main();
