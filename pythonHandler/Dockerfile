# 1. Choose a base Python image
FROM python:3.11-slim

# 2. Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1 # Prevents .pyc file generation

# 3. Set the working directory in the container
WORKDIR /app

# 4. Install build tools needed for C extensions AND ca-certificates for SSL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy only the requirements file first to leverage Docker's build cache.
# Assumes requirements.txt is in the pythonHandler/ directory (the build context).
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 7. Download spaCy models
RUN python -m spacy download en_core_web_lg
# Add other models here if needed, e.g.:
# RUN python -m spacy download en_core_web_sm

# 8. Copy the rest of the application code from the build context (pythonHandler/) into /app
# This comes after installing dependencies and models for better layer caching.
COPY . .

# 9. Expose the port the app runs on
EXPOSE 8000

# 10. Define the command to run your application.
# Uses shell form for $PORT substitution.
# --preload ensures the app is loaded once in the master before forking workers.
CMD gunicorn --preload main:app --bind "0.0.0.0:${PORT:-8000}" --timeout 120 --workers 2 --threads 4 --worker-class gthread