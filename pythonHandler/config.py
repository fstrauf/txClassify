"""Configuration constants for the application."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === Replicate Model Configuration ===
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
REPLICATE_MODEL_NAME = "beautyyuyanli/multilingual-e5-large"
REPLICATE_MODEL_VERSION = (
    "a06276a8732657f45f65514a7b041131e79172397951599171753417c9d85418"
)
EMBEDDING_DIMENSION = 384  # Dimension of the embeddings (Adjust based on model)

# === Classification Confidence Configuration ===
MIN_ABSOLUTE_CONFIDENCE = 0.70  # Minimum similarity score to consider a match valid
MIN_RELATIVE_CONFIDENCE_DIFF = (
    0.03  # Minimum difference between best and second best score for high confidence
)
NEIGHBOR_COUNT = 3  # Number of neighbors to consider for consistency check

# === Backend API Configuration ===
# Define backend API URL for webhooks
BACKEND_API = os.environ.get("BACKEND_API", "http://localhost:5001")

# Database Configuration (from environment variables)
DATABASE_URL = os.getenv("DATABASE_URL")
