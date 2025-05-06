"""Configuration constants for the application."""

import os

# === Replicate Model Configuration ===
REPLICATE_MODEL_NAME = "beautyyuyanli/multilingual-e5-large"
REPLICATE_MODEL_VERSION = (
    "a06276a89f1a902d5fc225a9ca32b6e8e6292b7f3b136518878da97c458e2bad"
)
EMBEDDING_DIMENSION = 1024  # Embedding dimension for the model

# === Backend API Configuration ===
# Define backend API URL for webhooks
BACKEND_API = os.environ.get("BACKEND_API", "http://localhost:5001")
