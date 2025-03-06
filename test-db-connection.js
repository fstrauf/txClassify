/**
 * Script to test the database connection.
 *
 * This script:
 * 1. Connects to the database
 * 2. Runs a simple query to check if the connection works
 *
 * Usage:
 * node test-db-connection.js
 */

require("dotenv").config();
const { PrismaClient } = require("@prisma/client");

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
    log("Connecting to database...");

    // Test connection by running a simple query
    log("Running test query...");
    const accountCount = await prisma.account.count();
    log(`Database connection successful! Found ${accountCount} accounts.`);

    // List all accounts (without sensitive data)
    log("Listing all accounts:");
    const accounts = await prisma.account.findMany();
    accounts.forEach((account) => {
      log(
        `- Account ID: ${account.id || "N/A"}, User ID: ${account.userId}, API Key: ${account.api_key ? "***" : "None"}`
      );
    });

    log("Database connection test completed successfully!");
  } catch (error) {
    log(`Error connecting to database: ${error.message}`);
    log(`Error details: ${JSON.stringify(error, null, 2)}`);

    if (error.code === "P1001") {
      log(
        "This error indicates that the database server cannot be reached. Check your DATABASE_URL and ensure the database is running."
      );
    } else if (error.code === "P1003") {
      log(
        "This error indicates that the database does not exist. Make sure you have created the database and run migrations."
      );
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
