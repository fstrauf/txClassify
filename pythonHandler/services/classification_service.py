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
from urllib.parse import parse_qs, urlparse
import gc

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
                
            logger.info(f"Initial training data size: {len(training_data)} transactions")
            
            # Validate required columns
            required_columns = ['Narrative', 'Category']
            missing_columns = [col for col in required_columns if col not in training_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Clean and preprocess data
            valid_data = self._preprocess_training_data(training_data)
            if len(valid_data) == 0:
                raise ValueError("No valid training data after cleaning")

            # Get embeddings for all descriptions at once
            descriptions = valid_data['Narrative'].tolist()
            prediction = self.run_prediction(
                "training",
                sheet_id,
                user_id,
                descriptions,
                webhook_params={'sheet_id': sheet_id}
            )
            
            logger.info("Training request sent successfully")
            return prediction
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise

    def _preprocess_training_data(self, training_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess training data with robust error handling."""
        try:
            # 1. Convert to string and strip whitespace
            training_data['Narrative'] = training_data['Narrative'].astype(str).str.strip()
            training_data['Category'] = training_data['Category'].astype(str).str.strip()
            
            # 2. Remove rows with empty or invalid values
            valid_data = training_data.dropna(subset=['Narrative', 'Category'])
            valid_data = valid_data[
                (valid_data['Narrative'].str.len() > 0) & 
                (valid_data['Category'].str.len() > 0)
            ]
            
            # 3. Clean transaction descriptions with progress tracking
            cleaned_descriptions = []
            total = len(valid_data)
            for idx, desc in enumerate(valid_data['Narrative']):
                try:
                    cleaned = self._clean_description(desc)
                    cleaned_descriptions.append(cleaned)
                    if (idx + 1) % 1000 == 0:
                        logger.info(f"Cleaned {idx + 1}/{total} descriptions")
                except Exception as e:
                    logger.warning(f"Error cleaning description '{desc}': {e}")
                    cleaned_descriptions.append(desc)  # Use original if cleaning fails
            
            valid_data['Narrative'] = cleaned_descriptions
            
            # 4. Remove duplicates, keeping the most recent entry
            valid_data = valid_data.drop_duplicates(subset=['Narrative'], keep='last')
            
            # 5. Group similar transactions with error handling
            try:
                valid_data['Narrative_clean'] = valid_data['Narrative'].apply(self._normalize_description)
                valid_data = valid_data.drop_duplicates(subset=['Narrative_clean'], keep='last')
                valid_data = valid_data.drop('Narrative_clean', axis=1)
            except Exception as e:
                logger.warning(f"Error in normalization step: {e}")
                # Continue without normalization if it fails
            
            # Log cleaning results
            logger.info(f"After cleaning:")
            logger.info(f"- Initial size: {len(training_data)} transactions")
            logger.info(f"- Final size: {len(valid_data)} transactions")
            logger.info(f"- Removed {len(training_data) - len(valid_data)} entries")
            
            return valid_data
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def _save_training_state(self, state: dict) -> None:
        """Save training state to Supabase with retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                state_key = f"{state['sheet_id']}_training_state.json"
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(state, f)
                    temp_path = f.name
                
                with open(temp_path, 'rb') as f:
                    self.supabase.storage.from_(self.bucket_name).upload(
                        state_key,
                        f,
                        file_options={"x-upsert": "true"}
                    )
                
                os.unlink(temp_path)
                return
                
            except Exception as e:
                logger.error(f"Error saving training state (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def _get_training_state(self, sheet_id: str) -> dict:
        """Get training state from Supabase with retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                state_key = f"{sheet_id}_training_state.json"
                with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
                    response = self.supabase.storage.from_(self.bucket_name).download(state_key)
                    temp_file.write(response)
                    temp_file.flush()
                    
                    with open(temp_file.name, 'r') as f:
                        state = json.load(f)
                    
                    os.unlink(temp_file.name)
                    return state
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)

    def _clean_description(self, text: str) -> str:
        """Clean transaction description."""
        try:
            # Convert to string if not already
            text = str(text)
            
            # Generic patterns that apply to most bank transactions
            generic_patterns = [
                # Common transaction suffixes/metadata
                r'\s*\d{2,4}[-/]\d{2}[-/]\d{2,4}',  # Dates in various formats
                r'T\d{2}:\d{2}:\d{2}',  # Timestamps
                r'\s+\d{1,2}:\d{2}(:\d{2})?',  # Times
                r'\s*Card\s+[xX*]+\d{4}',  # Card numbers
                r'\s*\d{6,}',  # Long numbers (reference numbers)
                r'\s+\([^)]*\)',  # Anything in parentheses
                
                # Common business suffixes
                r'(?i)\s+(?:ltd|limited|pty|inc|llc|corporation)\.?$',
                
                # Transaction metadata
                r'(?i)\s*(?:value date|card ending|ref|reference)\s*:?.*$',
                r'\s+\|\s*[\d\.]+$',  # Amount at end
                r'\s+\|\s*[A-Z0-9\s]+$',  # Reference codes
                
                # Location/country codes
                r'\s+(?:AU|AUS|USA|UK|NS|CYP)$'
            ]
            
            # Apply generic patterns
            for pattern in generic_patterns:
                text = re.sub(pattern, '', text)
            
            # Common payment processor prefixes
            prefixes_to_remove = [
                'SQ *', 'TSG *', 'PP *', 'SP *', 'PAYPAL *', 'GOOGLE *', 'APPLE *',
                'INT\'L *', 'INTERNATIONAL *', 'THE *', 'IPY*'
            ]
            
            # Remove prefixes
            for prefix in prefixes_to_remove:
                if text.upper().startswith(prefix.upper()):
                    text = text[len(prefix):]
            
            # Remove extra whitespace and standardize spacing
            text = ' '.join(text.split())
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Error cleaning description '{text}': {str(e)}")
            return text

    def _normalize_description(self, text: str) -> str:
        """Normalize description for grouping similar transactions."""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove all numbers
            text = re.sub(r'\d+', '', text)
            
            # Keep alphanumeric, spaces, and ampersand
            text = re.sub(r'[^\w\s&]', '', text)
            
            # Generic business type patterns
            business_types = {
                # Food and Dining
                r'\b(?:cafe|coffee|restaurant|dining|takeaway|takeout|food)\b': 'food',
                
                # Retail
                r'\b(?:shop|store|market|supermarket)\b': 'store',
                
                # Transportation
                r'\b(?:taxi|uber|lyft|transport|transit|train|bus)\b': 'transport',
                
                # Fuel/Gas
                r'\b(?:fuel|petrol|gas station|service station)\b': 'fuel',
                
                # Healthcare
                r'\b(?:pharmacy|chemist|medical|doctor|health)\b': 'healthcare',
                
                # Utilities
                r'\b(?:electric|water|gas|utility|utilities)\b': 'utility',
                
                # Entertainment
                r'\b(?:cinema|movie|theatre|entertainment)\b': 'entertainment',
                
                # Financial
                r'\b(?:bank|transfer|payment|direct debit|credit)\b': 'financial'
            }
            
            # Apply business type normalization
            for pattern, replacement in business_types.items():
                text = re.sub(pattern, replacement, text)
            
            # Remove common non-descriptive words
            common_words = {'the', 'and', 'or', 'ltd', 'limited', 'pty', 'inc', 
                          'corporation', 'international', 'services', 'company',
                          'group', 'holdings', 'enterprises', 'solutions'}
            
            words = text.split()
            words = [w for w in words if w not in common_words]
            text = ' '.join(words)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Error normalizing description '{text}': {str(e)}")
            return text

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

    def verify_webhook_delivery(self, prediction_id: str, max_retries: int = 3, retry_delay: int = 5) -> bool:
        """Verify webhook delivery and trigger manual processing if needed."""
        try:
            for attempt in range(max_retries):
                # Check prediction status
                prediction = replicate.predictions.get(prediction_id)
                status = prediction.status
                
                logger.info(f"Checking prediction {prediction_id} status: {status}")
                
                if status == "succeeded":
                    # If succeeded but no webhook result, trigger manual processing
                    webhook_results = self.supabase.table("webhook_results").select("*").eq("prediction_id", prediction_id).execute()
                    
                    if not webhook_results.data:
                        logger.warning(f"Webhook not received for completed prediction {prediction_id}, manually processing")
                        self.process_webhook_response(prediction, prediction.output.get("sheetId"))
                        return True
                    return True
                    
                elif status == "failed":
                    logger.error(f"Prediction {prediction_id} failed")
                    return False
                    
                elif attempt < max_retries - 1:
                    logger.info(f"Prediction still processing, attempt {attempt + 1}/{max_retries}")
                    time.sleep(retry_delay)
                    
            return False
            
        except Exception as e:
            logger.error(f"Error verifying webhook delivery: {str(e)}")
            return False

    def _clean_text_for_embedding(self, text: str) -> str:
        """Clean text before sending to embedding model."""
        try:
            # Handle Unicode escapes first
            if isinstance(text, str):
                text = text.encode('utf-8').decode('unicode-escape')
            else:
                text = str(text)

            # Strip whitespace
            text = text.strip()
            
            # Remove reference numbers and codes after |
            if '|' in text:
                text = text.split('|')[0].strip()
            
            # Remove common suffixes with numbers
            text = re.sub(r'\s+\d+.*$', '', text)
            
            # Remove special location/branch indicators
            text = re.sub(r'\s+(?:Br|G|De|A)\s+\d+.*$', '', text)
            
            # Remove dates and times
            text = re.sub(r'\d{4}-\d{2}-\d{2}T.*Z', '', text)
            text = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?', '', text)
            
            # Normalize spaces and remove multiple spaces
            text = ' '.join(text.split())
            
            # Remove very short texts
            if len(text) < 2:
                return ""
            
            # Ensure text is properly encoded for output
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
            
            return text
            
        except Exception as e:
            logger.warning(f"Error cleaning text '{text}': {e}")
            # Return original text if cleaning fails
            return text if isinstance(text, str) else str(text)

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
            
            # Clean and validate texts
            cleaned_texts = []
            seen_texts = set()  # Track unique cleaned texts
            
            for text in texts:
                cleaned = self._clean_text_for_embedding(text)
                if not cleaned:
                    continue
                
                # Only add if we haven't seen this cleaned version before
                if cleaned not in seen_texts:
                    cleaned_texts.append(cleaned)
                    seen_texts.add(cleaned)
            
            if not cleaned_texts:
                raise ValueError("No valid texts after cleaning")
            
            logger.info(f"Cleaned and deduplicated {len(texts)} texts to {len(cleaned_texts)} unique texts")

            model = replicate.models.get("replicate/all-mpnet-base-v2")
            version = model.versions.get(
                "b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305"
            )

            # Create webhook URL with retry configuration
            base_url = self.backend_api.rstrip('/')
            webhook_endpoint = f"train/webhook" if api_mode == "training" else "classify/webhook"
            webhook = f"{base_url}/{webhook_endpoint}?sheetId={sheet_id}&userId={user_id}"
            
            # Add webhook parameters if provided
            if webhook_params:
                for key, value in webhook_params.items():
                    webhook += f"&{key}={value}"
            
            # Add retry configuration
            webhook += "&retry_count=3&retry_delay=10"
                    
            logger.info(f"Using webhook endpoint: {webhook}")

            # Create prediction with webhook and timeout
            try:
                prediction = replicate.predictions.create(
                    version=version,
                    input={
                        "text_batch": json.dumps(cleaned_texts),
                        "webhook": webhook,
                        "webhook_events_filter": ["completed"],
                        "webhook_retries": 3  # Add retry configuration
                    }
                )
            except Exception as e:
                raise ValueError(f"Failed to create prediction: {str(e)}")
            
            logger.info(f"Started {api_mode} with prediction ID: {prediction.id}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error in run_prediction: {str(e)}")
            raise

    def process_webhook_response(self, data: Dict[str, Any], sheet_id: str) -> pd.DataFrame:
        """Process webhook response and classify transactions."""
        try:
            logger.info("Processing webhook response")
            
            # Get embeddings from response
            new_embeddings = self.process_embeddings(data)
            logger.info(f"Processed embeddings shape: {new_embeddings.shape}")
            
            # Get original descriptions from the webhook response
            descriptions = []
            if 'input' in data and 'text_batch' in data['input']:
                try:
                    descriptions = json.loads(data['input']['text_batch'])
                except json.JSONDecodeError:
                    logger.warning("Could not parse text_batch from input")
            
            if not descriptions:
                logger.warning("No descriptions found in webhook data")
                descriptions = [""] * len(new_embeddings)
            
            # Store the embeddings with descriptions and categories
            training_data = {
                'embeddings': new_embeddings,
                'descriptions': np.array(descriptions),
                'categories': np.array([""] * len(descriptions))  # Will be filled by the training webhook
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
            logger.info(f"Stored training data with {len(descriptions)} examples")
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error processing webhook response: {str(e)}")
            logger.error(f"Data structure: {json.dumps(data, indent=2) if data else 'No data'}")
            raise

    def _process_classification_response(self, data: Dict[str, Any], sheet_id: str) -> pd.DataFrame:
        """Process classification response."""
        try:
            # Load trained data
            trained_data = self._load_training_data(sheet_id)
            if not trained_data:
                raise ValueError(f"No training data found for sheet {sheet_id}")
                
            logger.info(f"Loaded {len(trained_data['categories'])} training examples")
            logger.info(f"Training categories sample: {trained_data['categories'][:5]}")
            
            # Validate categories are not just numbers
            if all(cat.replace('-', '').replace('.', '').isdigit() for cat in trained_data['categories']):
                raise ValueError("Training data categories appear to be numerical values instead of category names")
            
            # Get embeddings from response
            new_embeddings = self.process_embeddings(data)
            
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
                    if "input" in data and "text_batch" in data["input"]:
                        try:
                            texts = json.loads(data["input"]["text_batch"])
                            text = texts[len(new_descriptions)] if len(texts) > len(new_descriptions) else None
                        except (json.JSONDecodeError, IndexError):
                            pass
                text = text or "Unknown description"
                new_descriptions.append(text)
            
            # Create results DataFrame
            predicted_categories = [trained_data['categories'][i] for i in best_matches]
            logger.info(f"Predicted categories: {predicted_categories}")
            
            results = pd.DataFrame({
                'description': new_descriptions,
                'predicted_category': predicted_categories,
                'similarity_score': best_scores,
                'matched_description': [trained_data['descriptions'][i] for i in best_matches]
            })
            
            logger.info(f"Generated predictions for {len(results)} items")
            logger.info(f"Sample predictions: {results['predicted_category'].head().tolist()}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing classification response: {str(e)}")
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
            # Ensure categories are strings and validate they are not just numbers
            categories = [str(cat).strip() for cat in categories]
            
            # Validate categories are not just numbers
            if all(cat.replace('-', '').replace('.', '').isdigit() for cat in categories):
                raise ValueError("Categories appear to be numerical values instead of category names")
            
            # Create combined data structure
            combined_data = {
                'embeddings': np.array(embeddings),
                'descriptions': np.array(descriptions),
                'categories': np.array(categories),
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'embedding_dimensions': embeddings.shape[1],
                    'sample_count': len(descriptions)
                }
            }
            
            # Save to temporary file with size check
            with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as temp_file:
                np.savez_compressed(temp_file, **combined_data)
                temp_file_path = temp_file.name
                
                # Check file size
                file_size = os.path.getsize(temp_file_path)
                if file_size > 50 * 1024 * 1024:  # 50MB limit
                    os.unlink(temp_file_path)
                    raise ValueError("Training data file too large")
            
            try:
                # Upload to Supabase with retry
                max_retries = 3
                retry_delay = 1
                
                for attempt in range(max_retries):
                    try:
                        with open(temp_file_path, 'rb') as f:
                            self.supabase.storage.from_(self.bucket_name).upload(
                                f"{sheet_id}_training.npz",
                                f.read(),
                                file_options={"x-upsert": "true"}
                            )
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(retry_delay * (2 ** attempt))
                
                logger.info(f"Successfully stored {len(descriptions)} training examples")
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                
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
                
                # Ensure categories are strings and not just numbers
                categories = [str(cat).strip() for cat in data['categories']]
                if all(cat.replace('-', '').replace('.', '').isdigit() for cat in categories):
                    raise ValueError("Training data categories appear to be numerical values instead of category names")
                
                result = {
                    'embeddings': data['embeddings'],
                    'descriptions': data['descriptions'],
                    'categories': categories
                }
                
                logger.info(f"Successfully loaded training data: {len(result['descriptions'])} examples")
                logger.info(f"Sample categories: {categories[:5]}")
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

    def store_temp_training_data(self, training_data: list, training_key: str) -> str:
        """Store training data temporarily in Supabase Storage."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as temp_file:
                # Write JSON data
                json.dump(training_data, temp_file, ensure_ascii=False)
                temp_file_path = temp_file.name
            
            try:
                # Upload to Supabase Storage with retries
                max_retries = 3
                retry_delay = 1  # seconds
                
                for attempt in range(max_retries):
                    try:
                        with open(temp_file_path, 'rb') as f:
                            self.supabase.storage.from_(self.bucket_name).upload(
                                f"{training_key}.json",
                                f,
                                file_options={"x-upsert": "true"}
                            )
                        logger.info(f"Successfully stored temporary training data with key: {training_key}")
                        break
                    except Exception as upload_error:
                        if attempt == max_retries - 1:
                            raise
                        logger.warning(f"Upload attempt {attempt + 1} failed: {str(upload_error)}")
                        time.sleep(retry_delay)
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            
            return training_key
            
        except Exception as e:
            logger.error(f"Error storing temp training data: {str(e)}")
            raise

    def get_temp_training_data(self, training_key: str) -> list:
        """Retrieve temporary training data from Supabase Storage."""
        try:
            # Download to temporary file with retries
            max_retries = 3
            retry_delay = 1  # seconds
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w+b') as temp_file:
                        response = self.supabase.storage.from_(self.bucket_name).download(
                            f"{training_key}.json"
                        )
                        temp_file.write(response)
                        temp_file.flush()
                        temp_file_path = temp_file.name
                        
                        # Load the data
                        with open(temp_file_path, 'r') as f:
                            training_data = json.load(f)
                        
                        # Clean up temporary file
                        os.unlink(temp_file_path)
                        
                        return training_data
                except Exception as download_error:
                    last_error = download_error
                    if attempt < max_retries - 1:
                        logger.warning(f"Download attempt {attempt + 1} failed: {str(download_error)}")
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    
            raise Exception(f"Failed to retrieve training data after {max_retries} attempts: {str(last_error)}")
                    
        except Exception as e:
            logger.error(f"Error retrieving temp training data: {e}")
            raise

    def cleanup_temp_training_data(self, training_key: str) -> None:
        """Clean up temporary training data from Supabase Storage."""
        try:
            # Double check if the file exists before attempting to delete
            try:
                # Try to download the file first to verify it exists
                self.supabase.storage.from_(self.bucket_name).download(f"{training_key}.json")
                
                # If we get here, the file exists, so we can delete it
                self.supabase.storage.from_(self.bucket_name).remove([f"{training_key}.json"])
                logger.info(f"Cleaned up temporary training data for key: {training_key}")
                
            except Exception as e:
                if "Object not found" in str(e):
                    logger.info(f"Temporary data {training_key} already cleaned up")
                else:
                    logger.warning(f"Error checking/cleaning temporary data: {e}")
            
        except Exception as e:
            logger.warning(f"Failed to clean up temporary training data: {e}")
            # Don't raise the exception - cleanup failures shouldn't break the process

    def store_embeddings(self, embeddings: np.ndarray, training_data: list, sheet_id: str = None) -> None:
        """Store embeddings with their corresponding categories."""
        try:
            if len(embeddings) != len(training_data):
                raise ValueError(f"Mismatch between embeddings ({len(embeddings)}) and training data ({len(training_data)})")
            
            # Extract categories and descriptions
            categories = [item['Category'] for item in training_data]
            descriptions = [item['Narrative'] for item in training_data]
            
            # Store training data
            self._store_training_data(
                embeddings=embeddings,
                descriptions=descriptions,
                categories=categories,
                sheet_id=sheet_id
            )
            
            logger.info(f"Successfully stored all embeddings")
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            raise 