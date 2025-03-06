# Transaction Classification System

This system classifies financial transactions into categories using machine learning.

## End-to-End Testing

The end-to-end tests verify that the API endpoints for training and categorization are working correctly.

### Prerequisites

- Node.js (v14 or higher)
- Python (v3.8 or higher)
- A running database (PostgreSQL)
- pnpm (v9.15.4 or higher)

### Setup

1. Install dependencies:

```bash
pnpm install
```

2. Create a `.env` file with the following variables:

```
DATABASE_URL=postgresql://username:password@localhost:5432/database
TEST_USER_ID=test_user_123
TEST_API_KEY=test_key_abc123
```

### Running the Tests

There are several ways to run the tests:

#### Option 1: Step by step (recommended for first-time setup)

1. Test database connection:

```bash
pnpm run test-db
```

2. Set up a test user in the database:

```bash
pnpm run setup-test-user
```

3. Run the tests:

```bash
pnpm test
```

4. Clean up the test user:

```bash
pnpm run cleanup-test-user
```

#### Option 2: All-in-one command

This will set up a test user, run the tests, and clean up the test user:

```bash
pnpm run test-with-setup
```

### Troubleshooting

If you encounter issues with the tests, try the following:

1. **Database Connection Issues**

   - Check that your database is running
   - Verify the `DATABASE_URL` in your `.env` file
   - Run `pnpm run test-db` to test the database connection

2. **API Key Validation Failures**

   - Ensure the test user exists in the database
   - Run `pnpm run setup-test-user` to create or update the test user
   - Check that the `TEST_USER_ID` and `TEST_API_KEY` match what's in the database

3. **Webhook Issues**

   - For Replicate to work properly, you need a public URL for webhooks
   - Use ngrok to expose your local webhook server

4. **Debugging**
   - Run `pnpm run debug-test` to start the tests with Node.js inspector
   - Check the Flask server logs for errors

### Using ngrok for Webhook Testing

For Replicate to work properly, you need to expose your webhook endpoint using ngrok:

1. Install ngrok: https://ngrok.com/download

2. Run ngrok:

```bash
ngrok http 3002
```

3. The test script will automatically detect the ngrok URL and use it for the webhook.

## Python Testing

There's also a Python version of the end-to-end tests in `pythonHandler/test/test_end_to_end.py`.

To run the Python tests:

```bash
python -m pythonHandler.test.test_end_to_end
```

These tests also require a test user in the database, which you can set up using the Node.js script:

```bash
pnpm run setup-test-user
```

## Introduction

This is a hybrid Next.js + Python app that uses Next.js as the frontend and Flask as the API backend. One great use case of this is to write Next.js apps that use Python AI libraries on the backend.

## How It Works

`pnpm run dev` spins up flask with python and nextjs.

`python3 api/index.py` let's you test the API separately as a python scripy

````curl -m 70 -X POST "http://localhost:8080/txclassify?mode=train" \
-H "Authorization: bearer $(gcloud auth print-identity-token)" \
-F "file=@/Users/fstrauf/Documents/01_code/txClassify/all_expenses_classified.csv"


curl -m 70 -X POST "https://us-central1-txclassify.cloudfunctions.net/txclassify?mode=train" \
-H "Authorization: bearer $(gcloud auth print-identity-token)" \
-F "file=@/Users/fstrauf/Documents/01_code/txClassify/all_expenses_classified.csv"```
````
