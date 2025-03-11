import os
import sys
import logging

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Flask app from main.py
from main import app

if __name__ == "__main__":
    app.run(debug=True)
