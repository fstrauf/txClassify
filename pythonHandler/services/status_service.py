"""Service layer for handling prediction status checks."""

import logging
import json

from utils.prisma_client import db_client
# from utils.request_utils import create_error_response

logger = logging.getLogger(__name__)


def get_and_process_status(prediction_id: str, requesting_user_id: str):
    """
    Fetches the status of an asynchronous job from the database.
    `prediction_id` corresponds to the `id` column in the `async_jobs` table.
    """
    try:
        logger.info(f"Fetching status for job_id: {prediction_id} by user: {requesting_user_id}")
        
        # Assume db_client has a method to fetch from 'async_jobs' by 'id'
        # Replace with your actual Prisma client call.
        # Example: job = db_client.async_jobs.find_unique(where={"id": prediction_id})
        # For now, using a hypothetical method name:
        job = db_client.get_async_job_by_id(prediction_id) 

        if not job:
            logger.warning(f"Job not found in async_jobs: {prediction_id}")
            return {
                "status": "error", 
                "error_message": "Job not found.", 
                "error_code": 404
            }

        # Verify user owns the job
        # Ensure the key in 'job' dictionary matches your table column name for user ID
        job_user_id = job.get("userId") # Or job.get("user_id") etc.
        if job_user_id != requesting_user_id:
            logger.warning(f"User {requesting_user_id} attempted to access job {prediction_id} owned by {job_user_id}")
            return {
                "status": "error", 
                "error_message": "Access denied to job status.", 
                "error_code": 403
            }

        job_status_from_db = job.get("status") # e.g., 'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'

        if job_status_from_db in ["PENDING", "PROCESSING"]:
            return {
                "status": "processing", # This is the status keyword client expects
                "message": f"Job is currently {job_status_from_db.lower() if job_status_from_db else 'unknown'}."
                # "provider_status": job_status_from_db # Optional: if client wants to see PENDING vs PROCESSING
            }
        elif job_status_from_db == "COMPLETED":
            result_data_from_db = job.get("result_data") # This could be a dict or JSON string
            final_payload = {}
            if result_data_from_db:
                try:
                    if isinstance(result_data_from_db, dict):
                        final_payload = result_data_from_db
                    elif isinstance(result_data_from_db, str):
                        final_payload = json.loads(result_data_from_db)
                    else:
                        # Should not happen if db_client stores it as JSON string or it comes from service as dict
                        logger.error(f"result_data for completed job {prediction_id} is neither dict nor string: {type(result_data_from_db)}")
                        raise json.JSONDecodeError("Invalid type for result_data", "", 0)
                        
                except json.JSONDecodeError as decode_error:
                    logger.error(f"Failed to parse result_data for completed job {prediction_id}: {result_data_from_db}. Error: {decode_error}")
                    return {
                        "status": "failed", # Treat as failed if results are corrupted
                        "error_message": "Failed to parse results from completed job.",
                        "error_code": 500 
                    }
            else: # Should not happen for a COMPLETED job, but handle defensively
                logger.warning(f"Completed job {prediction_id} has no result_data.")
                final_payload = {
                    "status": "completed",
                    "message": "Job completed but no result data found.",
                    # Include other necessary fields expected by client for a success response
                }

            # Ensure the payload has a 'status': 'completed' for consistency if not already there
            if isinstance(final_payload, dict) and final_payload.get("status") != "completed":
                 logger.debug(f"Adding/overwriting status to 'completed' for job {prediction_id}")
                 final_payload["status"] = "completed" # Ensure client gets this
            
            # Add prediction_id and job_type for context if not present in stored results
            if "prediction_id" not in final_payload:
                 final_payload["prediction_id"] = prediction_id
            if "type" not in final_payload and job.get("job_type"):
                 final_payload["type"] = job.get("job_type")
                 
            return final_payload # This is the dictionary the client expects

        elif job_status_from_db == "FAILED":
            error_details = job.get("error", "No error details provided.")
            return {
                "status": "failed", # Client-facing status
                "message": "Job processing failed.", # Generic message
                "error_details": error_details, # Specific error from the job
                # "error_code": job.get("error_code", 500) # If you store error codes
            }
        else:
            logger.error(f"Unknown job status '{job_status_from_db}' for job {prediction_id} in async_jobs table")
            return {
                "status": "error", 
                "error_message": f"Unknown or inconsistent job status: {job_status_from_db}", 
                "error_code": 500
            }

    except Exception as e:
        logger.error(f"Critical error in get_and_process_status for {prediction_id}: {e}", exc_info=True)
        # This error is for the status check itself, not a job failure
        return {
            "status": "error", 
            "error_message": "Internal server error while fetching job status.", 
            "error_code": 500
        }
