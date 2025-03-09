/**
 * Script to run the full test suite with a fixed test user.
 *
 * This script:
 * 1. Sets up a test user with fixed credentials
 * 2. Runs the API endpoint tests
 * 3. Cleans up the test user
 *
 * Usage:
 * node run-tests.js
 */

require("dotenv").config();
const { spawn } = require("child_process");
const { PrismaClient } = require("@prisma/client");

// Configuration
const TEST_USER_ID = "test_user_fixed";
const TEST_API_KEY = "test_api_key_fixed";

// Set up logging
const log = (message) => {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${message}`);
};

// Run a command and return a promise
const runCommand = (command, args) => {
  return new Promise((resolve, reject) => {
    log(`Running command: ${command} ${args.join(" ")}`);

    const childProcess = spawn(command, args, {
      stdio: "inherit", // This ensures all output is passed through to the parent process
      env: {
        ...process.env,
        TEST_USER_ID,
        TEST_API_KEY,
      },
    });

    childProcess.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Command failed with exit code ${code}`));
      }
    });

    childProcess.on("error", (error) => {
      reject(error);
    });
  });
};

// Setup test user
const setupTestUser = async () => {
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
  } catch (error) {
    log(`Error setting up test user: ${error.message}`);
    throw error;
  } finally {
    await prisma.$disconnect();
  }
};

// Clean up test user
const cleanupTestUser = async () => {
  log("Cleaning up test user...");

  const prisma = new PrismaClient({
    log: ["error"],
  });

  try {
    // Delete the test user
    await prisma.account
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
  } catch (error) {
    log(`Error cleaning up test user: ${error.message}`);
    throw error;
  } finally {
    await prisma.$disconnect();
  }
};

// Run tests
const runTests = async () => {
  log("Running API endpoint tests...");

  // Set environment variables for the test process
  process.env.TEST_USER_ID = TEST_USER_ID;
  process.env.TEST_API_KEY = TEST_API_KEY;

  log(`Using test user ID: ${TEST_USER_ID} and API key: ${TEST_API_KEY}`);

  await runCommand("node", ["test-api-endpoints.js"]);
  log("API endpoint tests complete");
};

// Main function
const main = async () => {
  try {
    // Setup test user
    await setupTestUser();

    // Run tests
    await runTests();

    // Clean up test user
    await cleanupTestUser();

    log("All tests completed successfully");
    process.exit(0);
  } catch (error) {
    log(`Error in test suite: ${error.message}`);

    // Try to clean up test user even if tests fail
    try {
      await cleanupTestUser();
    } catch (cleanupError) {
      log(`Error during cleanup: ${cleanupError.message}`);
    }

    process.exit(1);
  }
};

// Handle process termination
process.on("SIGINT", async () => {
  log("Process interrupted");
  try {
    await cleanupTestUser();
  } catch (error) {
    log(`Error during cleanup: ${error.message}`);
  }
  process.exit(1);
});

process.on("SIGTERM", async () => {
  log("Process terminated");
  try {
    await cleanupTestUser();
  } catch (error) {
    log(`Error during cleanup: ${error.message}`);
  }
  process.exit(1);
});

// Run the main function
main();
