# Testing the Transaction Classification System

This directory contains tests for the transaction classification system.

## Test Scripts

The following npm scripts are available for testing:

- `pnpm test` or `pnpm test:all`: Run all tests (training and categorization)
- `pnpm test:train`: Run only the training test
- `pnpm test:cat`: Run only the categorization test
- `pnpm test:quiet`: Run all tests with minimal logging
- `pnpm debug-test`: Run all tests with Node.js inspector enabled for debugging

## Test Data

The test data is located in the `test_data` directory:

- `training_data.csv`: Data used for training the model
- `categorise_test.csv`: Data used for testing categorization

## Environment Variables

The tests use the following environment variables:

- `TEST_USER_ID`: The user ID to use for testing (default: "test_user_fixed")
- `TEST_API_KEY`: The API key to use for testing (default: "test_api_key_fixed")
- `API_PORT`: The port to use for the Flask API (default: 3001)
- `BACKEND_API`: The URL for the backend API (used for webhooks)
- `USE_WEBHOOKS`: Set to "true" to enable webhooks, "false" to disable (default: "false")

## Command Line Flags

The test script accepts the following command line flags:

- `--verbose`: Enable verbose logging
- `--debug`: Enable debug logging
- `--trace`: Enable trace logging
- `--train-only`: Run only the training test
- `--cat-only`: Run only the categorization test

## Troubleshooting

- If you encounter database connection issues, check your DATABASE_URL in .env
- If the API key validation fails, ensure the test user exists in the database
- For more detailed debugging, run: `pnpm run debug-test`
- If the Flask server fails to start, check if the port is already in use
