"""Utility functions for handling embeddings (storage, fetching, quantization)."""

import io
import numpy as np
import time
import logging
from utils.prisma_client import prisma_client

logger = logging.getLogger(__name__)


def store_embeddings(data: np.ndarray, embedding_id: str, user_id: str) -> bool:
    """Store embeddings in database with quantization, including userId."""
    try:
        logger.info(
            f"Storing embeddings with ID: {embedding_id}, shape: {data.shape}, for User: {user_id}"
        )
        start_time = time.time()

        # Prepare data before database operation
        buffer = io.BytesIO()

        # Handle structured arrays (like index data) differently
        if data.dtype.names is not None:
            dtype_dict = {
                "names": data.dtype.names,
                "formats": [
                    data.dtype.fields[name][0].str for name in data.dtype.names
                ],
            }
            metadata = {"is_structured": True, "dtype": dtype_dict, "shape": data.shape}
            np.savez(buffer, metadata=metadata, data=data.tobytes())
        else:
            # For regular arrays (embeddings), use quantization
            if data.dtype == np.float32:
                # Avoid division by zero if data is all zeros
                abs_max = np.max(np.abs(data))
                scale = (
                    abs_max / 127 if abs_max > 0 else 1.0
                )  # Default scale to 1 if max is 0
                quantized_data = (data / scale).astype(np.int8)
                metadata = {
                    "is_structured": False,
                    "scale": scale,
                    "shape": data.shape,
                    "quantized": True,
                }
                np.savez_compressed(buffer, metadata=metadata, data=quantized_data)
            else:
                metadata = {
                    "is_structured": False,
                    "quantized": False,
                    "shape": data.shape,
                }
                np.savez_compressed(buffer, metadata=metadata, data=data)

        data_bytes = buffer.getvalue()
        buffer.close()

        # Store in database with retry logic
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                # Correctly call the singular method name in prisma_client
                result = prisma_client.store_embedding(
                    embedding_id, data_bytes, user_id
                )

                # Removed the redundant tracking code below, as userId is stored directly

                logger.info(f"Embeddings stored in {time.time() - start_time:.2f}s")
                return result

            except Exception as e:
                if "connection pool" in str(e).lower() or "timeout" in str(e).lower():
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Database operation failed (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        continue
                # Log the error before re-raising or returning False
                logger.error(
                    f"Attempt {attempt + 1} failed to store embedding {embedding_id}: {e}"
                )
                # Re-raise the exception on the last attempt or if it's not a pool/timeout error
                if attempt == max_retries - 1 or not (
                    "connection pool" in str(e).lower() or "timeout" in str(e).lower()
                ):
                    raise

        # If loop completes without success (should only happen if pool/timeout errors persist)
        return False

    except Exception as e:
        # Catch potential errors during data prep or final re-raise from loop
        logger.error(f"Error storing embedding {embedding_id}: {e}", exc_info=True)
        return False


def fetch_embeddings(embedding_id: str) -> np.ndarray:
    """Fetch embeddings from database."""
    try:
        start_time = time.time()
        logger.info(f"Fetching embeddings with ID: {embedding_id}")

        # Fetch from database with retry logic
        max_retries = 3
        retry_delay = 1  # seconds
        data_bytes = None

        for attempt in range(max_retries):
            try:
                data_bytes = prisma_client.fetch_embedding(embedding_id)
                break
            except Exception as e:
                if "connection pool" in str(e).lower() or "timeout" in str(e).lower():
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Database fetch failed (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                raise

        if not data_bytes:
            logger.warning(f"No embeddings found with ID: {embedding_id}")
            return np.array([])

        # Convert bytes back to numpy array
        try:
            buffer = io.BytesIO(data_bytes)
            with np.load(buffer, allow_pickle=True) as loaded:
                metadata = loaded["metadata"].item()
                data = loaded["data"]

                if metadata.get("is_structured", False):
                    # Reconstruct structured array
                    dtype_dict = metadata["dtype"]
                    dtype = np.dtype(
                        {"names": dtype_dict["names"], "formats": dtype_dict["formats"]}
                    )
                    embeddings = np.frombuffer(data, dtype=dtype)
                    if len(metadata["shape"]) > 1:
                        embeddings = embeddings.reshape(metadata["shape"])
                else:
                    # Handle regular arrays
                    if metadata.get("quantized", False):
                        scale = metadata["scale"]
                        embeddings = data.astype(np.float32) * scale
                    else:
                        embeddings = data

            buffer.close()
            logger.info(
                f"Embeddings loaded in {time.time() - start_time:.2f}s, shape: {embeddings.shape}"
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error loading numpy array from bytes: {e}")
            return np.array([])

    except Exception as e:
        logger.error(f"Error fetching embeddings: {e}")
        return np.array([])
