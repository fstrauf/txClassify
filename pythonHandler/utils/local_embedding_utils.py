import spacy
import numpy as np
import logging
from typing import List, Optional
from spacy.language import Language

# Try to import sentence-transformers as a fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# --- Configuration ---
# Choose a transformer model compatible with spacy-transformers
# Option 1: A common Sentence Transformer model (good general-purpose choice)
TRANSFORMER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Option 2: A spaCy transformer pipeline (e.g., en_core_web_trf)
# TRANSFORMER_MODEL_NAME = "en_core_web_trf"
# Option 3: A different Hugging Face model
# TRANSFORMER_MODEL_NAME = "bert-base-uncased"

logger = logging.getLogger(__name__)

# Global variable to hold the loaded pipeline (simple singleton pattern)
_nlp_pipeline: Optional[Language] = None
_sentence_transformer_model: Optional = None


def load_spacy_pipeline() -> Optional[Language]:
    """
    Loads the spaCy pipeline with the specified transformer model.
    Uses a global variable to ensure the model is loaded only once.
    Returns the loaded pipeline or None if loading fails.
    """
    global _nlp_pipeline
    if _nlp_pipeline is None:
        try:
            logger.info(
                f"Loading spaCy transformer pipeline: {TRANSFORMER_MODEL_NAME}..."
            )
            # Check if it's a spaCy model name or HuggingFace name
            if "/" in TRANSFORMER_MODEL_NAME:  # Likely HuggingFace
                # Using spacy.blank("en") and adding transformer component
                # might be more robust if the HF model doesn't have a full pipeline
                _nlp_pipeline = spacy.blank("en")
                _nlp_pipeline.add_pipe(
                    "transformer",
                    config={
                        "model": {
                            "@architectures": "spacy-transformers.TransformerModel.v3",
                            "name": TRANSFORMER_MODEL_NAME,
                        }
                    },
                )
                _nlp_pipeline.initialize()  # Important to initialize the pipeline
                logger.info(
                    f"Successfully initialized blank spaCy pipeline with transformer: {TRANSFORMER_MODEL_NAME}"
                )

            else:  # Assumed to be a full spaCy pipeline name
                _nlp_pipeline = spacy.load(TRANSFORMER_MODEL_NAME)
                logger.info(
                    f"Successfully loaded spaCy pipeline: {TRANSFORMER_MODEL_NAME}"
                )

            # Verify the pipeline has the expected component (transformer)
            if "transformer" not in _nlp_pipeline.pipe_names:
                logger.warning(
                    f"Pipeline '{TRANSFORMER_MODEL_NAME}' loaded, but missing 'transformer' component."
                )
                # Decide if this is an error or acceptable based on model choice
                # For now, we'll proceed but log the warning.

        except ImportError as e:
            logger.error(
                f"ImportError loading spaCy or transformers: {e}. Ensure spacy, spacy-transformers, and torch/tensorflow are installed."
            )
            _nlp_pipeline = None  # Ensure it remains None on error
        except OSError as e:
            logger.error(
                f"OSError loading spaCy model '{TRANSFORMER_MODEL_NAME}'. It might not be downloaded or installed correctly: {e}"
            )
            logger.error(
                f"Try: python -m spacy download {TRANSFORMER_MODEL_NAME} (if it's a spaCy model) or check HuggingFace model name."
            )
            _nlp_pipeline = None  # Ensure it remains None on error
        except Exception as e:
            logger.error(
                f"Unexpected error loading spaCy pipeline '{TRANSFORMER_MODEL_NAME}': {e}",
                exc_info=True,
            )
            _nlp_pipeline = None  # Ensure it remains None on error

    return _nlp_pipeline


def generate_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    """
    Generates embeddings for a list of texts using available embedding methods.
    
    Tries in order:
    1. Sentence Transformers (if available)
    2. spaCy transformer pipeline
    3. Simple word vectors from spaCy

    Args:
        texts: A list of text strings to embed.

    Returns:
        A numpy array of embeddings (shape: num_texts x embedding_dim),
        or None if no embedding method is available.
    """
    if not texts:
        logger.warning("generate_embeddings called with empty list of texts.")
        return np.empty((0, 1))  # Return empty array with placeholder dim

    # Strategy 1: Try sentence-transformers first (most reliable)
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            global _sentence_transformer_model
            if _sentence_transformer_model is None:
                logger.info("Loading sentence-transformers model...")
                _sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence-transformers model loaded successfully")
            
            logger.info(f"Generating embeddings for {len(texts)} texts using sentence-transformers...")
            embeddings = _sentence_transformer_model.encode(texts)
            logger.info(f"Generated embeddings using sentence-transformers (shape: {embeddings.shape})")
            return embeddings
            
        except Exception as e:
            logger.warning(f"Sentence-transformers failed: {e}. Falling back to spaCy.")
    
    # Strategy 2: Try spaCy transformer pipeline
    nlp = load_spacy_pipeline()
    if nlp is not None:
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts using spaCy transformer...")
            # Process texts in batches using nlp.pipe for efficiency
            docs = list(nlp.pipe(texts))

            # Check if vectors are available
            if all(doc.has_vector for doc in docs):
                embeddings = np.array([doc.vector for doc in docs])
                logger.info(f"Extracted embeddings using doc.vector (shape: {embeddings.shape})")
                return embeddings
            else:
                logger.warning("doc.vector not available for all docs. Trying transformer data extraction.")
                # Try to extract from transformer data
                embeddings = []
                for doc in docs:
                    if hasattr(doc._, "trf_data") and doc._.trf_data is not None:
                        # Use mean pooling of transformer outputs
                        token_embeddings = doc._.trf_data.tensors[0]
                        # Simple mean pooling
                        doc_embedding = np.mean(token_embeddings, axis=0)
                        embeddings.append(doc_embedding)
                    else:
                        logger.warning("No transformer data available for document")
                        return None
                
                if embeddings:
                    embeddings = np.array(embeddings)
                    logger.info(f"Extracted embeddings using transformer data (shape: {embeddings.shape})")
                    return embeddings
                    
        except Exception as e:
            logger.warning(f"spaCy transformer pipeline failed: {e}. Trying basic spaCy.")
    
    # Strategy 3: Try basic spaCy model with word vectors
    try:
        logger.info("Trying basic spaCy model...")
        nlp_basic = spacy.load("en_core_web_md")
        docs = list(nlp_basic.pipe(texts))
        
        if all(doc.has_vector for doc in docs):
            embeddings = np.array([doc.vector for doc in docs])
            logger.info(f"Generated embeddings using basic spaCy (shape: {embeddings.shape})")
            return embeddings
            
    except Exception as e:
        logger.warning(f"Basic spaCy model failed: {e}")
    
    # Strategy 4: Very simple character-based embeddings as last resort
    try:
        logger.warning("Using simple character-based embeddings as fallback")
        embeddings = []
        max_len = min(100, max(len(text) for text in texts) if texts else 100)
        
        for text in texts:
            # Create a simple character frequency vector
            char_vector = np.zeros(256)  # ASCII character space
            for char in text.lower()[:max_len]:
                char_vector[ord(char)] += 1
            # Normalize
            if np.sum(char_vector) > 0:
                char_vector = char_vector / np.sum(char_vector)
            embeddings.append(char_vector)
        
        embeddings = np.array(embeddings)
        logger.info(f"Generated simple character embeddings (shape: {embeddings.shape})")
        return embeddings
        
    except Exception as e:
        logger.error(f"All embedding strategies failed: {e}")
        return None

        # Strategy 2: Extracting from transformer data (doc._.trf_data)
        # This is common for many HuggingFace models integrated via spacy-transformers
        # We typically need to pool the token embeddings (e.g., mean pooling of last hidden state)
        if all(
            hasattr(doc._, "trf_data") and doc._.trf_data is not None for doc in docs
        ):
            # Example: Mean pooling of the last hidden layer outputs
            # The exact structure of trf_data can vary, inspect it if needed
            # This assumes trf_data has tensors attribute with last_hidden_state
            embeddings_list = []
            for i, doc in enumerate(docs):
                # Check if tensors and last_hidden_state are available
                if (
                    hasattr(doc._, "trf_data")
                    and doc._.trf_data is not None
                    and hasattr(doc._.trf_data, "tensors")
                    and len(doc._.trf_data.tensors) > 0
                ):
                    # Shape: (num_layers, batch_size=1, num_tokens, hidden_dim)
                    # We want the last layer: [-1], squeeze batch dim: [0]
                    last_hidden_state = doc._.trf_data.tensors[-1][0]

                    # For sentence-transformers, last_hidden_state often contains the pooled output directly
                    doc_embedding = last_hidden_state

                    # --- DEBUG LOGGING ---
                    if i == 0:  # Log only for the first doc
                        logger.debug(
                            f"Shape of doc_embedding for first doc: {doc_embedding.shape}"
                        )
                    # --- END DEBUG ---

                    embeddings_list.append(doc_embedding)
                else:
                    logger.error(
                        f"Transformer data for doc '{doc.text[:50]}...' does not contain expected tensors."
                    )
                    return None  # Fail if we can't extract consistently

            if not embeddings_list:
                logger.error("Failed to extract any embeddings from transformer data.")
                return None

            embeddings = np.array(embeddings_list)
            logger.info(
                f"Extracted embeddings using mean pooling of transformer data (shape: {embeddings.shape})"
            )
            return embeddings
        else:
            logger.error(
                "Cannot extract embeddings: Neither doc.vector nor doc._.trf_data available or suitable."
            )
            return None

    except Exception as e:
        logger.error(f"Error generating embeddings: {e}", exc_info=True)
        return None


# --- Optional: Pre-load pipeline on module import ---
# Uncomment the line below if you want to load the model when the app starts
# load_spacy_pipeline()
