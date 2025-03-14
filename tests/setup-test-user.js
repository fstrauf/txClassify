/**
 * Script to set up a test user in the database for end-to-end testing.
 *
 * This script:
 * 1. Creates a test user in the database with a valid API key
 * 2. Outputs the user ID and API key for use in tests
 *
 * Usage:
 * node setup-test-user.js
 */

require("dotenv").config();
const { PrismaClient } = require("@prisma/client");
const crypto = require("crypto");

// Configuration
const TEST_USER_ID = "test_user_fixed";
const TEST_API_KEY = "test_api_key_fixed";

// Set up logging
const log = (message) => {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${message}`);
};

// Main function
const main = async () => {
  log(`DATABASE_URL: ${process.env.DATABASE_URL ? "Set (hidden for security)" : "Not set"}`);

  const prisma = new PrismaClient({
    log: ["query", "info", "warn", "error"],
  });

  try {
    log(`Setting up test user with ID: ${TEST_USER_ID} and API key: ${TEST_API_KEY}`);
    log("Connecting to database...");

    // Check if the user already exists
    log("Checking if user already exists...");
    const existingAccount = await prisma.account.findUnique({
      where: { userId: TEST_USER_ID },
    });

    if (existingAccount) {
      // Update the existing account
      log(`Test user already exists with ID: ${existingAccount.id}, updating API key`);

      const updatedAccount = await prisma.account.update({
        where: { userId: TEST_USER_ID },
        data: { api_key: TEST_API_KEY },
      });

      log(
        `Updated test account: ${JSON.stringify({
          id: updatedAccount.id,
          userId: updatedAccount.userId,
          api_key: updatedAccount.api_key ? "***" : null,
        })}`
      );
    } else {
      // Create a new account
      log(`Creating new test user`);

      const newAccount = await prisma.account.create({
        data: {
          userId: TEST_USER_ID,
          api_key: TEST_API_KEY,
          categorisationRange: "A:D",
          categorisationTab: "Sheet1",
          columnOrderCategorisation: JSON.stringify(["Date", "Amount", "Description", "Currency"]),
        },
      });

      log(
        `Created test account: ${JSON.stringify({
          id: newAccount.id,
          userId: newAccount.userId,
          api_key: newAccount.api_key ? "***" : null,
        })}`
      );
    }

    // Output the user ID and API key for use in tests
    console.log(`\nTest user created successfully!`);
    console.log(`TEST_USER_ID=${TEST_USER_ID}`);
    console.log(`TEST_API_KEY=${TEST_API_KEY}`);
    console.log(`\nAdd these to your .env file or export them as environment variables before running tests.`);
  } catch (error) {
    log(`Error setting up test user: ${error.message}`);
    log(`Error details: ${JSON.stringify(error, null, 2)}`);

    if (error.code === "P1001") {
      log(
        "This error indicates that the database server cannot be reached. Check your DATABASE_URL and ensure the database is running."
      );
    } else if (error.code === "P1003") {
      log(
        "This error indicates that the database does not exist. Make sure you have created the database and run migrations."
      );
    } else if (error.code === "P2002") {
      log("This error indicates a unique constraint violation. The user ID or API key might already be in use.");
    }

    console.error(error);
    process.exit(1);
  } finally {
    log("Disconnecting from database...");
    await prisma.$disconnect();
    log("Disconnected from database");
  }
};

// Run the main function
main();
