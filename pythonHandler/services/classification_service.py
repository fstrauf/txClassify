import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import replicate
import tempfile
from typing import List, Dict, Any, Tuple
import logging
from supabase import create_client, Client
import re
import time

logger = logging.getLogger(__name__)

class ClassificationService:
    def __init__(self, supabase_url: str, supabase_key: str, backend_api: str):
        self.supabase = create_client(supabase_url, supabase_key)
        self.backend_api = backend_api
        self.bucket_name = "txclassify"

    def train(self, training_data: pd.DataFrame, sheet_id: str, user_id: str) -> None:
        """Train the classifier with descriptions and their categories."""
        try:
            if training_data.empty:
                raise ValueError("Training data is empty")
                
            logger.info(f"Training with {len(training_data)} transactions")
            
            # Validate required columns
            required_columns = ['Narrative', 'Category']
            missing_columns = [col for col in required_columns if col not in training_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Remove any rows with empty narratives or categories
            valid_data = training_data.dropna(subset=['Narrative', 'Category'])
            if len(valid_data) == 0:
                raise ValueError("No valid training data after removing empty values")
            
            if len(valid_data) < len(training_data):
                logger.warning(f"Dropped {len(training_data) - len(valid_data)} rows with empty values")
            
            # Store training data first
            training_records = valid_data.to_dict('records')
            training_key = self.store_temp_training_data(training_records, sheet_id)
            logger.info(f"Stored training data with key: {training_key}")
            
            # Get embeddings for descriptions
            descriptions = valid_data['Narrative'].astype(str).tolist()
            
            # Run prediction with webhook
            prediction = self.run_prediction(
                "training",
                sheet_id,
                user_id,
                descriptions,
                webhook_params={'training_key': training_key}
            )
            
            logger.info("Training request sent successfully")
            return prediction
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            # Try to clean up temporary data if it was created
            if 'training_key' in locals():
                try:
                    self.supabase.storage.from_(self.bucket_name).remove([f"{training_key}.json"])
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary data: {cleanup_error}")
            raise

    def classify(self, new_data: pd.DataFrame, sheet_id: str, user_id: str) -> None:
        """Start classification of new transactions."""
        try:
            logger.info(f"Classifying {len(new_data)} transactions")
            
            # Get descriptions for new data
            descriptions = new_data['Narrative'].astype(str).tolist()
            
            # Run prediction with webhook
            prediction = self.run_prediction(
                "classify",
                sheet_id,
                user_id,
                descriptions
            )
            
            logger.info("Classification request sent successfully")
            return prediction
            
        except Exception as e:
            logger.error(f"Error starting classification: {str(e)}")
            raise

    def run_prediction(
        self, 
        api_mode: str,
        sheet_id: str,
        user_id: str,
        texts: List[str],
        webhook_params: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Run prediction using Replicate API with webhook."""
        try:
            logger.info(f"Running prediction for {len(texts)} texts in {api_mode} mode")
            
            model = replicate.models.get("replicate/all-mpnet-base-v2")
            version = model.versions.get(
                "b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305"
            )

            # Create webhook URL - ensure no double slashes
            base_url = self.backend_api.rstrip('/')
            webhook_endpoint = f"train/webhook" if api_mode == "training" else "classify/webhook"
            webhook = f"{base_url}/{webhook_endpoint}?sheetId={sheet_id}&userId={user_id}"
            
            # Add webhook parameters if provided
            if webhook_params:
                for key, value in webhook_params.items():
                    webhook += f"&{key}={value}"
                    
            logger.info(f"Using webhook endpoint: {webhook}")

            # Create prediction with webhook
            prediction = replicate.predictions.create(
                version=version,
                input={
                    "text_batch": json.dumps(texts),
                    "webhook_url": webhook  # Pass webhook URL in input
                },
                webhook=webhook,
                webhook_events_filter=["completed"],
            )
            
            logger.info(f"Started {api_mode} with prediction ID: {prediction.id}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error in run_prediction: {str(e)}")
            raise

    def process_webhook_response(self, data: Dict[str, Any], sheet_id: str) -> pd.DataFrame:
        """Process webhook response and classify transactions."""
        try:
            logger.info("Processing webhook response")
            
            # Validate webhook data
            if not data or 'output' not in data:
                raise ValueError("Invalid webhook data: missing 'output' field")
            
            # Get embeddings from response
            new_embeddings = self.process_embeddings(data)
            logger.info(f"Processed embeddings shape: {new_embeddings.shape}")
            
            # Load trained data
            trained_data = self._load_training_data(sheet_id)
            if not trained_data:
                raise ValueError(f"No training data found for sheet {sheet_id}")
                
            logger.info(f"Loaded {len(trained_data['categories'])} training examples")
            
            # Validate embeddings dimensions match
            if new_embeddings.shape[1] != trained_data['embeddings'].shape[1]:
                raise ValueError(f"Embedding dimensions mismatch: new {new_embeddings.shape[1]} vs trained {trained_data['embeddings'].shape[1]}")
            
            # Find most similar categories
            similarities = cosine_similarity(new_embeddings, trained_data['embeddings'])
            best_matches = similarities.argmax(axis=1)
            best_scores = similarities.max(axis=1)
            
            # Get the original descriptions from the webhook response
            new_descriptions = []
            for item in data["output"]:
                text = None
                if isinstance(item, dict) and "text" in item:
                    text = item["text"]
                elif isinstance(item, dict) and "embedding" in item:
                    # If text is not in output, try to get it from input
                    if "input" in data and "text_batch" in data["input"]:
                        try:
                            texts = json.loads(data["input"]["text_batch"])
                            text = texts[len(new_descriptions)] if len(texts) > len(new_descriptions) else None
                        except (json.JSONDecodeError, IndexError):
                            pass
                text = text or "Unknown description"
                new_descriptions.append(text)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'description': new_descriptions,
                'predicted_category': [trained_data['categories'][i] for i in best_matches],
                'similarity_score': best_scores,
                'matched_description': [trained_data['descriptions'][i] for i in best_matches]
            })
            
            logger.info(f"Generated predictions for {len(results)} items")
            return results
            
        except Exception as e:
            logger.error(f"Error processing webhook response: {str(e)}")
            logger.error(f"Data structure: {json.dumps(data, indent=2) if data else 'No data'}")
            raise

    def process_embeddings(self, data: Dict[str, Any]) -> np.ndarray:
        """Convert JSON response to numpy array of embeddings."""
        try:
            embeddings = [item["embedding"] for item in data["output"]]
            return np.array(embeddings)
        except KeyError as e:
            logger.error(f"KeyError in process_embeddings: {e}")
            logger.error(f"Data structure: {data}")
            raise

    def _store_training_data(self, embeddings: np.ndarray, descriptions: List[str], 
                           categories: List[str], sheet_id: str) -> None:
        """Store training data in Supabase storage."""
        try:
            # Create a structured array with all training data
            training_data = {
                'embeddings': embeddings,
                'descriptions': descriptions,
                'categories': categories
            }
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as temp_file:
                np.savez_compressed(temp_file, **training_data)
                temp_file_path = temp_file.name
            
            # Upload to Supabase
            with open(temp_file_path, 'rb') as f:
                self.supabase.storage.from_(self.bucket_name).upload(
                    f"{sheet_id}_training.npz",
                    f,
                    file_options={"x-upsert": "true"}
                )
            
            # Clean up
            os.unlink(temp_file_path)
            logger.info(f"Stored training data for {len(descriptions)} examples")
                
        except Exception as e:
            logger.error(f"Error storing training data: {str(e)}")
            raise

    def _load_training_data(self, sheet_id: str) -> Dict[str, Any]:
        """Load training data from Supabase storage."""
        try:
            # Download to temporary file
            with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as temp_file:
                response = self.supabase.storage.from_(self.bucket_name).download(
                    f"{sheet_id}_training.npz"
                )
                temp_file.write(response)
                temp_file.flush()
                temp_file_path = temp_file.name

            try:
                # Load the data
                data = np.load(temp_file_path, allow_pickle=True)
                
                # Verify all required keys are present
                required_keys = ['embeddings', 'descriptions', 'categories']
                missing_keys = [key for key in required_keys if key not in data.files]
                if missing_keys:
                    raise ValueError(f"Training data is missing required keys: {missing_keys}")
                
                # Verify data is not empty
                if len(data['embeddings']) == 0 or len(data['descriptions']) == 0 or len(data['categories']) == 0:
                    raise ValueError("Training data is empty")
                
                # Verify data lengths match
                if not (len(data['embeddings']) == len(data['descriptions']) == len(data['categories'])):
                    raise ValueError("Mismatch in training data lengths")
                
                result = {
                    'embeddings': data['embeddings'],
                    'descriptions': data['descriptions'],
                    'categories': data['categories']
                }
                
                logger.info(f"Successfully loaded training data: {len(result['descriptions'])} examples")
                return result
                
            except Exception as e:
                raise ValueError(f"Error processing training data: {str(e)}")
            
            finally:
                # Clean up
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Error loading training data for sheet {sheet_id}: {str(e)}")
            raise ValueError(f"No training data found for sheet {sheet_id}. Please train the model first.")

    def classify_expenses(
        self,
        df_unclassified: pd.DataFrame,
        trained_embeddings: np.ndarray,
        new_embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """Classify expenses using cosine similarity."""
        try:
            desc_new_data = df_unclassified["description"]
            logger.info(f"Processing descriptions: {desc_new_data}")
            logger.info(f"Trained embeddings shape: {trained_embeddings.shape}")
            logger.info(f"New embeddings shape: {new_embeddings.shape}")

            # Calculate similarity between new embeddings and trained embeddings
            similarity_new_data = cosine_similarity(new_embeddings, trained_embeddings)
            similarity_df = pd.DataFrame(similarity_new_data)

            # Get indices of most similar items
            index_similarity = similarity_df.idxmax(axis=1)
            
            # Return the mapping information
            d_output = {
                "description": desc_new_data,
                "categoryIndex": index_similarity
            }
            return d_output

        except Exception as e:
            logger.error(f"Error in classify_expenses: {str(e)}")
            logger.error(f"df_unclassified shape: {df_unclassified.shape}")
            logger.error(f"trained_embeddings: {type(trained_embeddings)}")
            logger.error(f"new_embeddings shape: {new_embeddings.shape}")
            raise

    def save_embeddings(self, file_name: str, embeddings_array: np.ndarray) -> None:
        """Save embeddings to Supabase storage."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp_file:
            np.savez_compressed(temp_file, embeddings_array)
            file_path = temp_file.name

        try:
            with open(file_path, "rb") as f:
                self.supabase.storage.from_(self.bucket_name).upload(
                    file_name,
                    f,
                    file_options={"x-upsert": "true"}
                )
                logger.info(f"Successfully uploaded embeddings to {file_name}")

        except Exception as e:
            logger.error(f"Error uploading embeddings: {str(e)}")
            raise Exception(f"Failed to upload embeddings: {str(e)}")

        finally:
            os.unlink(file_path)

    def fetch_embeddings(self, file_name: str) -> np.ndarray:
        """Fetch embeddings from Supabase storage."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp_file:
            response = self.supabase.storage.from_(self.bucket_name).download(file_name)
            temp_file.write(response)
            temp_file.flush()

        try:
            return np.load(temp_file.name, allow_pickle=True)["arr_0"]
        finally:
            os.unlink(temp_file.name)

    def store_training_data(self, df: pd.DataFrame, sheet_id: str) -> None:
        """Store training data index."""
        try:
            # Ensure we have the required columns
            required_columns = ["item_id", "category", "description"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}, Required: {required_columns}")
            
            logger.info(f"Storing training data with columns: {df.columns.tolist()}")
            logger.info(f"Sample categories: {df['category'].head().tolist()}")
            
            # Create structured array with proper dtype
            structured_array = np.array(
                list(zip(
                    df["item_id"].astype(int).to_numpy(),
                    df["category"].astype(str).to_numpy(),
                    df["description"].astype(str).to_numpy()
                )),
                dtype=[("item_id", int), ("category", "U50"), ("description", "U500")]
            )
            
            logger.info(f"Created structured array with {len(structured_array)} entries")
            logger.info(f"First few categories in array: {structured_array['category'][:5]}")
            
            self.save_embeddings(f"{sheet_id}_index.npy", structured_array)
            
        except Exception as e:
            logger.error(f"Error in store_training_data: {str(e)}")
            logger.error(f"DataFrame info: {df.info()}")
            raise

    @staticmethod
    def _generate_timestamp() -> str:
        """Generate a timestamp for run tracking."""
        return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def store_temp_training_data(self, training_data: list, sheet_id: str) -> str:
        """Store training data temporarily in Supabase Storage."""
        try:
            # Create a unique key for this training session
            training_key = f"temp_training_{sheet_id}_{int(time.time())}"
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w+b') as temp_file:
                # Convert to JSON and encode as bytes
                json_data = json.dumps(training_data).encode('utf-8')
                temp_file.write(json_data)
                temp_file.flush()
                temp_file_path = temp_file.name
            
            try:
                # Upload to Supabase Storage
                with open(temp_file_path, 'rb') as f:
                    self.supabase.storage.from_(self.bucket_name).upload(
                        f"{training_key}.json",
                        f,
                        file_options={"x-upsert": "true"}
                    )
                logger.info(f"Stored temporary training data with key: {training_key}")
                
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
            
            return training_key
            
        except Exception as e:
            logger.error(f"Error storing temp training data: {str(e)}")
            raise

    def get_temp_training_data(self, training_key: str) -> list:
        """Retrieve temporary training data from Supabase Storage."""
        try:
            # Download to temporary file
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w+b') as temp_file:
                response = self.supabase.storage.from_(self.bucket_name).download(
                    f"{training_key}.json"
                )
                temp_file.write(response)
                temp_file.flush()
                
                # Load the data
                with open(temp_file.name, 'rb') as f:
                    training_data = json.loads(f.read().decode('utf-8'))
                
                # Clean up temporary file
                os.unlink(temp_file.name)
                
                return training_data
                
        except Exception as e:
            logger.error(f"Error retrieving temp training data: {e}")
            raise

    def cleanup_temp_training_data(self, training_key: str) -> None:
        """Clean up temporary training data from Supabase Storage."""
        try:
            self.supabase.storage.from_(self.bucket_name).remove([f"{training_key}.json"])
            logger.info(f"Cleaned up temporary training data for key: {training_key}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary training data: {e}")

    def store_embeddings(self, embeddings: np.ndarray, training_data: list) -> None:
        """Store embeddings with their corresponding categories."""
        try:
            if len(embeddings) != len(training_data):
                raise ValueError(f"Mismatch between embeddings ({len(embeddings)}) and training data ({len(training_data)})")
            
            # Extract categories from training data
            categories = [item['Category'] for item in training_data]
            descriptions = [item['Narrative'] for item in training_data]
            
            # Store training data
            self._store_training_data(
                embeddings=embeddings,
                descriptions=descriptions,
                categories=categories,
                sheet_id=None  # We'll use the default sheet ID
            )
            
            logger.info(f"Successfully stored {len(embeddings)} embeddings with categories")
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            raise 
            raise 