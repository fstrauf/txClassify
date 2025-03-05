# Transaction Classification Test Suite

This directory contains tests for the transaction classification system. The tests cover various aspects of the system, from unit tests of individual components to end-to-end tests that simulate the entire workflow.

## Test Files

- `test_end_to_end.py`: Comprehensive tests that simulate the entire flow from Google Sheets to the backend service, including both training and categorization processes.
- `test_embeddings_storage.py`: Tests for the embeddings storage functionality.
- `test_api_key.py`: Tests for API key validation.
- `test_psycopg2.py`: Tests for database connectivity.
- `test_flask_app.py`: Tests for the Flask application endpoints.

## Running the Tests

You can run the tests using the provided `run_tests.py` script in the parent directory:

```bash
cd pythonHandler
./run_tests.py
```

Or you can run individual test files:

```bash
cd pythonHandler
python -m unittest test.test_end_to_end
```

## End-to-End Test with Real API Calls

For a more realistic end-to-end test that makes actual API calls to Replicate, you can use the `run_e2e_test.py` script:

```bash
cd pythonHandler
./run_e2e_test.py
```

This script:
1. Starts a Flask server on port 3001
2. Sets up an ngrok tunnel to expose the server to the internet
3. Runs the end-to-end tests using the ngrok URL as the backend API
4. Cleans up all processes when done

You can also start the Flask server and ngrok without running tests:

```bash
cd pythonHandler
./run_e2e_test.py --skip-tests
```

This is useful for manual testing or debugging.

### Prerequisites for End-to-End Testing

- Install ngrok: https://ngrok.com/download
- Set up a valid API key in your `.env` file or as an environment variable
- Ensure you have all required Python dependencies installed

## End-to-End Test Process

The end-to-end test (`test_end_to_end.py`) simulates the complete flow from training to categorization:

1. **Training Flow**:
   - Loads training data from CSV
   - Sends a training request to the API
   - Waits for the Replicate prediction to complete
   - Verifies the training status

2. **Categorization Flow**:
   - Loads categorization data from CSV
   - Sends a categorization request to the API
   - Waits for the Replicate prediction to complete
   - Verifies the categorization results

This test closely resembles real-world usage, testing both the training and categorization processes in a way that simulates how the Google Sheets addon would interact with the backend service.

## Test Data

The tests use sample data from the `pythonHandler/test_data` directory:
- `training_test.csv`: Sample data for training the model
- `categorise_test.csv`: Sample data for testing categorization

## Environment Setup

The tests automatically set up the necessary environment variables and mock external services to ensure they can run without external dependencies. However, for the end-to-end test with real API calls, you'll need:

1. A valid API key for authentication
2. Internet connectivity for Replicate API calls
3. ngrok installed for exposing your local server

## Troubleshooting

If you encounter issues with the tests:

1. Check that all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure the `.env` file is properly configured with necessary environment variables:
   ```
   TEST_API_KEY=your_api_key
   REPLICATE_API_TOKEN=your_replicate_token
   ```

3. For end-to-end tests with ngrok:
   - Make sure ngrok is installed and in your PATH
   - Check that port 3001 is available
   - Verify that you can connect to the Replicate API

4. Check the logs for detailed error messages. 