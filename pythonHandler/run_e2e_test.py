#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import threading
import logging
import signal
import argparse
import json
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global variables for process management
flask_process = None
ngrok_process = None
ngrok_url = None

def create_test_data():
    """Create test data files if they don't exist."""
    logger.info("Checking and creating test data files if needed...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_script = os.path.join(script_dir, 'test', 'create_test_data.py')
    
    if not os.path.exists(test_data_script):
        logger.error(f"Test data creation script not found: {test_data_script}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, test_data_script],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating test data: {e}")
        logger.error(e.stderr)
        return False

def start_flask_app():
    """Start the Flask application."""
    global flask_process
    
    logger.info("Starting Flask application...")
    
    # Set environment variables for Flask
    env = os.environ.copy()
    env["FLASK_APP"] = "main.py"
    env["FLASK_ENV"] = "development"
    
    # Get the project root directory (one level up from the script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Add the project root to PYTHONPATH
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{project_root}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = project_root
    
    logger.info(f"Set PYTHONPATH to include project root: {project_root}")
    
    # Start Flask in a subprocess
    flask_process = subprocess.Popen(
        ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=3001"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for Flask to start
    time.sleep(2)
    
    # Check if Flask started successfully
    if flask_process.poll() is not None:
        logger.error("Flask application failed to start")
        output, _ = flask_process.communicate()
        logger.error(f"Flask output: {output}")
        sys.exit(1)
    
    logger.info("Flask application started successfully")
    
    # Start a thread to log Flask output
    def log_output():
        for line in iter(flask_process.stdout.readline, ''):
            logger.info(f"Flask: {line.strip()}")
    
    threading.Thread(target=log_output, daemon=True).start()

def start_ngrok():
    """Start ngrok to expose the Flask app to the internet."""
    global ngrok_process, ngrok_url
    
    logger.info("Starting ngrok...")
    
    # Check if ngrok is installed
    try:
        subprocess.run(["ngrok", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ngrok is not installed or not in PATH. Please install ngrok.")
        sys.exit(1)
    
    # Start ngrok in a subprocess
    ngrok_process = subprocess.Popen(
        ["ngrok", "http", "3001", "--log=stdout"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for ngrok to start and get the public URL
    time.sleep(2)
    
    # Get ngrok public URL
    try:
        ngrok_info = subprocess.run(
            ["curl", "-s", "http://localhost:4040/api/tunnels"],
            capture_output=True,
            text=True,
            check=True
        )
        
        tunnels = json.loads(ngrok_info.stdout)
        
        if not tunnels.get("tunnels"):
            logger.error("No ngrok tunnels found")
            sys.exit(1)
        
        for tunnel in tunnels["tunnels"]:
            if tunnel["proto"] == "https":
                ngrok_url = tunnel["public_url"]
                break
        
        if not ngrok_url:
            logger.error("No HTTPS ngrok tunnel found")
            sys.exit(1)
        
        logger.info(f"ngrok started successfully. Public URL: {ngrok_url}")
        
        # Save the original BACKEND_API value
        original_backend_api = os.environ.get("BACKEND_API")
        
        # Set environment variable for the backend API URL
        os.environ["BACKEND_API"] = ngrok_url
        logger.info(f"Set BACKEND_API to {ngrok_url} (was: {original_backend_api})")
        
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        logger.error(f"Failed to get ngrok URL: {e}")
        sys.exit(1)

def verify_environment():
    """Verify that all required environment variables are set."""
    required_vars = ["REPLICATE_API_TOKEN"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.warning("Tests may fail without these variables set.")
    
    # Check for API key
    if not os.environ.get("API_KEY"):
        logger.warning("API_KEY not found in environment. Tests will use a default test key.")
    
    # Log current environment settings
    logger.info(f"BACKEND_API: {os.environ.get('BACKEND_API', 'Not set')}")
    logger.info(f"REPLICATE_API_TOKEN: {'Set' if os.environ.get('REPLICATE_API_TOKEN') else 'Not set'}")
    logger.info(f"API_KEY: {'Set' if os.environ.get('API_KEY') else 'Not set'}")
    logger.info(f"DATABASE_URL: {'Set' if os.environ.get('DATABASE_URL') else 'Not set'}")

def run_tests():
    """Run the end-to-end tests."""
    logger.info("Running end-to-end tests...")
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Set up environment for tests
    test_env = os.environ.copy()
    
    # Add the project root to PYTHONPATH
    if "PYTHONPATH" in test_env:
        test_env["PYTHONPATH"] = f"{project_root}:{test_env['PYTHONPATH']}"
    else:
        test_env["PYTHONPATH"] = project_root
    
    # Run the test directly using the test file path
    test_file = os.path.join(script_dir, 'test', 'test_end_to_end.py')
    
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return 1
    
    logger.info(f"Running test file: {test_file}")
    
    # Run the tests
    test_process = subprocess.run(
        [sys.executable, test_file],
        cwd=script_dir,
        env=test_env,
        capture_output=True,
        text=True
    )
    
    # Print test output
    logger.info("Test output:")
    logger.info(test_process.stdout)
    
    if test_process.stderr:
        logger.error("Test errors:")
        logger.error(test_process.stderr)
    
    # Return test exit code
    return test_process.returncode

def cleanup(signum=None, frame=None):
    """Clean up processes on exit."""
    logger.info("Cleaning up...")
    
    if flask_process:
        logger.info("Stopping Flask application...")
        flask_process.terminate()
        flask_process.wait(timeout=5)
    
    if ngrok_process:
        logger.info("Stopping ngrok...")
        ngrok_process.terminate()
        ngrok_process.wait(timeout=5)
    
    logger.info("Cleanup complete")

def main():
    """Main function to run the end-to-end test environment."""
    parser = argparse.ArgumentParser(description="Run end-to-end tests with Flask and ngrok")
    parser.add_argument("--skip-tests", action="store_true", help="Start Flask and ngrok but don't run tests")
    parser.add_argument("--no-ngrok", action="store_true", help="Don't start ngrok, use local Flask server only")
    args = parser.parse_args()
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        # Change to the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Get the project root directory (one level up from the script directory)
        project_root = os.path.dirname(script_dir)
        
        # Add the project root to Python path
        sys.path.insert(0, project_root)
        
        # Load environment variables from .env file
        load_dotenv()
        logger.info("Loaded environment variables from .env file")
        
        # Verify environment variables
        verify_environment()
        
        # Create test data if needed
        if not create_test_data():
            logger.warning("Failed to create test data. Tests may fail.")
        
        # Start Flask
        start_flask_app()
        
        # Start ngrok if needed
        if not args.no_ngrok:
            start_ngrok()
        else:
            logger.info("Skipping ngrok as requested. Using local Flask server only.")
            # Ensure BACKEND_API is set to localhost
            os.environ["BACKEND_API"] = "http://localhost:3001"
        
        if args.skip_tests:
            logger.info("Skipping tests as requested. Press Ctrl+C to exit.")
            # Keep the script running until interrupted
            while True:
                time.sleep(1)
        else:
            # Run the tests
            exit_code = run_tests()
            
            # Exit with the test exit code
            sys.exit(exit_code)
            
    except Exception as e:
        logger.exception(f"Error running end-to-end tests: {e}")
        sys.exit(1)
    finally:
        cleanup()

if __name__ == "__main__":
    main() 