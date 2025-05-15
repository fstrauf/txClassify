"""Utility functions for interacting with the Replicate API."""

import replicate
import json
import time
import logging
from config import REPLICATE_MODEL_NAME, REPLICATE_MODEL_VERSION

logger = logging.getLogger(__name__)


def run_prediction(descriptions: list) -> dict:
    """Run prediction using Replicate API.

    Args:
        descriptions: List of descriptions to process
    """
    try:
        start_time = time.time()
        total_items = len(descriptions)
        logger.info(
            f"Starting prediction with model: {REPLICATE_MODEL_NAME} for {total_items} items"
        )

        # Clean descriptions and ensure they are strings
        cleaned_descriptions = [str(desc).strip() for desc in descriptions]

        model = replicate.models.get(REPLICATE_MODEL_NAME)
        version = model.versions.get(REPLICATE_MODEL_VERSION)

        # Format input exactly as shown in documentation
        # Note: The API expects 'texts' to be a JSON string of a list of strings
        input_texts = json.dumps(cleaned_descriptions)
        prediction = replicate.predictions.create(
            version=version,
            input={
                "texts": input_texts,
                "batch_size": 32,
                "normalize_embeddings": True,
            },
        )

        if not prediction or not prediction.id:
            raise Exception("Failed to create prediction (no prediction ID returned)")

        logger.info(
            f"Prediction created with ID: {prediction.id} (time: {time.time() - start_time:.2f}s)"
        )
        return prediction

    except Exception as e:
        logger.error(f"Prediction creation failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        if hasattr(e, "response"):
            logger.error(
                f"Response status: {e.response.status_code if hasattr(e.response, 'status_code') else 'unknown'}"
            )
            logger.error(
                f"Response content: {e.response.text if hasattr(e.response, 'text') else 'unknown'}"
            )
        raise
