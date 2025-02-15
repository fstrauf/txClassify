@app.route('/status/<prediction_id>', methods=['GET'])
@require_api_key
def check_status(prediction_id):
    """Check status of a prediction"""
    try:
        # First check if we have webhook results
        webhook_results = None
        try:
            response = services["classification"].supabase.table("webhook_results").select("*").eq("prediction_id", prediction_id).execute()
            if response.data:
                webhook_results = response.data[0].get("results", {})
                logger.info(f"Found webhook results for prediction {prediction_id}")
                return jsonify({
                    "status": "completed",
                    "result": webhook_results,
                    "completion_time": response.data[0].get("created_at")
                })
        except Exception as e:
            logger.warning(f"Failed to get webhook results: {e}")
        
        # If no webhook results, check Replicate status with retries
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                prediction = replicate.predictions.get(prediction_id)
                logger.info(f"Prediction status: {prediction.status}")
                
                # Handle timestamps safely
                created_at = prediction.created_at
                if isinstance(created_at, str):
                    created_at_str = created_at
                else:
                    created_at_str = created_at.isoformat()
                
                completed_at = getattr(prediction, 'completed_at', None)
                completed_at_str = completed_at.isoformat() if completed_at and not isinstance(completed_at, str) else completed_at
                
                # Calculate elapsed time safely
                try:
                    if isinstance(created_at, str):
                        from datetime import datetime
                        created_timestamp = datetime.fromisoformat(created_at.replace('Z', '+00:00')).timestamp()
                    else:
                        created_timestamp = created_at.timestamp()
                    elapsed_time = time.time() - created_timestamp
                except Exception as e:
                    logger.warning(f"Error calculating elapsed time: {e}")
                    elapsed_time = 0
                
                status_response = {
                    "prediction_id": prediction_id,
                    "status": prediction.status,
                    "created_at": created_at_str,
                    "elapsed_time": elapsed_time,
                    "logs": prediction.logs
                }
                
                if prediction.status == "succeeded":
                    # If prediction succeeded but no webhook results yet
                    if not webhook_results:
                        # Try to manually trigger webhook processing
                        try:
                            logger.info("Attempting to manually process webhook for succeeded prediction")
                            webhook_results = services["classification"].process_webhook_response(prediction, prediction.output)
                            if webhook_results is not None:
                                return jsonify({
                                    "status": "completed",
                                    "result": webhook_results,
                                    "completion_time": datetime.now().isoformat()
                                })
                        except Exception as webhook_error:
                            logger.warning(f"Manual webhook processing failed: {webhook_error}")
                    
                    status_response.update({
                        "status": "processing",
                        "message": "Webhook processing in progress...",
                        "prediction_completed_at": completed_at_str
                    })
                elif prediction.status == "failed":
                    status_response.update({
                        "status": "failed",
                        "error": prediction.error
                    })
                else:
                    status_response.update({
                        "message": f"Prediction in progress... ({prediction.status})"
                    })
                
                return jsonify(status_response)
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"Replicate API attempt {attempt + 1} failed: {e}")
                    time.sleep(1 * (2 ** attempt))
                else:
                    if webhook_results:
                        # If we can't get prediction status but have webhook results, we're done
                        return jsonify({
                            "status": "completed",
                            "result": webhook_results
                        })
                    else:
                        raise
            
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        return jsonify({
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500 