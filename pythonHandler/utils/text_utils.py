"""Generic utility functions for cleaning and processing financial transaction text data."""

import re
import logging
import spacy
import os
import csv
import ahocorasick
import Levenshtein
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Token, Doc
from typing import List, Union, Dict, Optional, Tuple
from config import USE_NZ_BUSINESS_DATA_MATCHING
from functools import lru_cache
import hashlib

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logging.getLogger(__name__).warning("HDBSCAN not available. Install with: pip install hdbscan")

try:
    from utils.local_embedding_utils import generate_embeddings
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.getLogger(__name__).warning("Local embedding utils not available")

logger = logging.getLogger(__name__)


# --- Define CleaningConfig Class ---
class CleaningConfig:
    def __init__(self):
        # Generic cleaning options
        self.remove_card_numbers = True
        self.remove_reference_codes = True
        self.remove_transaction_codes = True
        self.remove_merchant_codes = True
        self.normalize_whitespace = True
        self.to_lowercase = True
        
        # Advanced cleaning options
        self.use_nlp_extraction = True
        self.remove_common_payment_terms = True
        self.remove_bank_codes = True
        self.remove_urls_emails = True
        self.normalize_merchant_names = True
        
        # Grouping options
        self.use_fuzzy_matching = False  # Disable old algorithm in favor of embedding-based grouping
        self.similarity_threshold = 0.85
        
        # Embedding-based grouping options
        self.use_embedding_grouping = True  # Enable by default to use improved algorithm
        self.embedding_clustering_method = "similarity"  # "hdbscan", "dbscan", "hierarchical", "similarity"
        self.embedding_similarity_threshold = 0.85  # Conservative threshold to prevent false groupings
        self.embedding_min_cluster_size = 2
        self.embedding_eps = 0.25  # Optimized for merchant names (lower = tighter clusters)
        self.embedding_min_samples = 2  # For DBSCAN
        self.embedding_hierarchical_threshold = 0.3  # Optimized (lower = tighter clusters)
        self.embedding_fallback_to_fuzzy = True  # Fallback to fuzzy matching if embeddings fail
        self.embedding_use_cache = True  # Enable/disable embedding caching
        self.embedding_batch_size = 50  # Process embeddings in batches for memory efficiency
        
        # Legacy options for backward compatibility
        self.normalize_merchants = True
        self.use_fuzzy_matching_post_clean = True
        self.similarity_threshold_post_clean = 0.85
        self.group_by_category_post_clean = True
        self.apply_nlp_features_extraction = True
        self.apply_merchant_variants_normalization = True
        self.apply_alias_resolution = True
        self.perform_aggressive_store_number_removal = True
        self.perform_company_suffix_removal = True

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


# --- Generic Transaction Noise Patterns ---
transaction_noise_patterns = [
    # Payment method indicators
    [{"LOWER": {"IN": ["visa", "eftpos", "mastercard", "amex", "debit", "credit"]}}],
    [{"LOWER": "tap"}, {"LOWER": {"IN": ["and", "&"]}, "OP": "?"}, {"LOWER": {"IN": ["pay", "go"]}}],
    [{"LOWER": "contactless"}],
    [{"LOWER": "chip"}, {"LOWER": {"IN": ["and", "&"]}, "OP": "?"}, {"LOWER": "pin"}],
    
    # Transaction types
    [{"LOWER": {"IN": ["purchase", "payment", "transfer", "withdrawal", "deposit"]}}],
    [{"LOWER": "direct"}, {"LOWER": {"IN": ["debit", "credit"]}}],
    [{"LOWER": "automatic"}, {"LOWER": "payment"}],
    [{"LOWER": {"IN": ["atm", "pos", "eft"]}}],
    [{"LOWER": "authorised"}],
    [{"LOWER": "netbank"}],
    [{"LOWER": "commbank"}],
    [{"LOWER": "app"}],
    
    # Generic transaction codes (3-6 digits, letters, or combinations)
    [{"TEXT": {"REGEX": r"^[A-Z]{2,4}\d{2,4}$"}}],  # e.g., "DF", "SP", "IF"
    [{"TEXT": {"REGEX": r"^\d{3,6}$"}}],  # Standalone numbers
    [{"TEXT": {"REGEX": r"^[A-Z]\d+[A-Z]?$"}}],  # e.g., "C", "S3D7741"
    
    # Bank/institution codes
    [{"LOWER": {"REGEX": r"^s\d[a-z]\d+$"}}],  # ANZ style codes like "s3d7741"
    [{"TEXT": {"REGEX": r"^\*{4,}$"}}],  # Masked numbers
    
    # Common prefixes observed in logs
    [{"LOWER": "df"}],
    [{"LOWER": "if"}],
    [{"LOWER": "sp"}],
    
    # Keywords often related to noise
    [{"LOWER": {"IN": ["card", "value", "date", "ref", "reference", "txn"]}}],
    
    # Specific digit sequences (>= 5 digits)
    [{"IS_DIGIT": True, "LENGTH": {">=": 5}}],
]

# Patterns for generic location noise
location_noise_patterns = [
    # Common Country Codes/Names
    [{"LOWER": {"IN": ["au", "aus", "australia"]}}],
    [{"LOWER": {"IN": ["us", "usa"]}}],
    [{"LOWER": "united"}, {"LOWER": {"IN": ["states", "kingdom"]}}],
    [{"LOWER": {"IN": ["gb", "gbr", "uk"]}}],
    [{"LOWER": {"IN": ["ca", "can", "canada"]}}],
    [{"LOWER": {"IN": ["nz", "nzl"]}}],
    [{"LOWER": "new"}, {"LOWER": "zealand"}],
    
    # Common State/Province Codes
    [{"LOWER": {"IN": ["nsw", "vic", "qld", "wa", "sa", "tas", "act", "nt"]}}],
    
    # Common noise terms
    [{"LOWER": {"IN": ["dee", "ns"]}}],
]

# --- Custom spaCy Component ---
if not Token.has_extension("is_transaction_noise"):
    Token.set_extension("is_transaction_noise", default=False)

@Language.component("transaction_noise_filter")
def transaction_noise_filter_component(doc: Doc) -> Doc:
    """Marks transaction-specific noise patterns."""
    matcher = Matcher(doc.vocab)
    matcher.add("TRANSACTION_NOISE", transaction_noise_patterns)
    matches = matcher(doc)

    for match_id, start, end in matches:
        for i in range(start, end):
            doc[i]._.is_transaction_noise = True
    return doc

# --- Load spaCy Model ---
nlp = None
try:
    nlp = spacy.load("en_core_web_sm")
    if not nlp.has_pipe("transaction_noise_filter"):
        nlp.add_pipe("transaction_noise_filter", after="ner" if "ner" in nlp.pipe_names else True)
    logger.info("spaCy model loaded successfully")
except Exception as e:
    logger.error(f"Could not load spaCy model: {e}")

# --- NZ Business Data Loading (kept for backward compatibility) ---
NZ_BUSINESS_AUTOMATON = ahocorasick.Automaton()
NZ_BUSINESS_DATA_FILEPATH = os.getenv("NZ_BUSINESS_DATA_PATH", "data/nz_business_data.csv")

def load_nz_business_data():
    """Loads NZ business names from a CSV file into the Aho-Corasick automaton."""
    global NZ_BUSINESS_AUTOMATON
    loaded_count = 0
    if not os.path.exists(NZ_BUSINESS_DATA_FILEPATH):
        logger.warning(f"NZ business data file not found at {NZ_BUSINESS_DATA_FILEPATH}")
        return

    try:
        with open(NZ_BUSINESS_DATA_FILEPATH, mode="r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            if "ENTITY_NAME" not in reader.fieldnames:
                logger.error(f"'ENTITY_NAME' column not found in {NZ_BUSINESS_DATA_FILEPATH}")
                return
            for row in reader:
                business_name = row["ENTITY_NAME"]
                if business_name and business_name.strip():
                    NZ_BUSINESS_AUTOMATON.add_word(
                        business_name.strip().lower(), business_name.strip().lower()
                    )
                    loaded_count += 1
        if loaded_count > 0:
            NZ_BUSINESS_AUTOMATON.make_automaton()
            logger.info(f"Successfully loaded {loaded_count} NZ business names")
    except Exception as e:
        logger.error(f"Error loading NZ business data: {e}")

# Load NZ business data if enabled
if USE_NZ_BUSINESS_DATA_MATCHING:
    load_nz_business_data()

# --- Generic Cleaning Functions ---

def remove_card_numbers(text: str) -> str:
    """Remove various credit card number patterns."""
    patterns = [
        # Full card numbers (various formats)
        r'\b\d{4}[-\s]?\*{4}[-\s]?\*{4}[-\s]?\d{4}\b',
        r'\b\d{4}[-\s]?\d{2}\*{2}[-\s]?\*{4}[-\s]?\d{4}\b',
        r'\b\*{4}[-\s]?\*{4}[-\s]?\*{4}[-\s]?\d{4}\b',
        r'\b\d{4}[-\s]?\*{4}[-\s]?\*{4}[-\s]?\*{4}\b',
        
        # Partial card numbers
        r'\b\d{4}[-\s]?\*+[-\s]?\*+[-\s]?\d{1,4}\b',
        r'\bxx\d+\b',
        r'\b\*{4,}\b',
        
        # Card numbers with preceding indicators
        r'\bcard\s*(?:number\s*)?\d{4}[-\s\*]{4,}\d{0,4}\b',
        
        # Specific format from your data: "4835-****-****-0311 Df"
        r'\b\d{4}-\*{4}-\*{4}-\d{4}\s*[A-Z]{1,3}\b',
        
        # More generic masked patterns
        r'\b[*xX]{4}-[*xX]{4}-[*xX]{4}-[*xX]{4}\b',
        r'\b\d{4}[-\s]\*+[-\s]\*+[-\s]\d{4}\s*[A-Z]{1,3}\b',
        
        # Common variations
        r'\b\d{4}\*+\d{1,4}\s*[A-Z]{1,3}\b',
        r'\b\*{8,16}\s*[A-Z]{1,3}\b',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    return text

def remove_reference_codes(text: str) -> str:
    """Remove various reference and transaction codes."""
    patterns = [
        # Transaction reference codes
        r'\b(?:ref|reference|txn|transaction)[:.]?\s*[A-Z0-9]{3,}\b',
        r'\b[A-Z]{1,3}\s*\d{6,}\b',  # e.g., "G Br 134924"
        r'\b\d{6,}\b',  # Long number sequences (likely references)
        
        # Bank codes and identifiers  
        r'\b[A-Z]\d+[A-Z]?\d*\b',  # e.g., "S3D7741", "C", "A"
        r'\b\d+[A-Z]+\d*\b',  # e.g., "2770157Nova"
        
        # Date/time codes
        r'\b\d{6,}[A-Z]*\b',  # e.g., "250420131128"
        
        # Generic codes
        r'\b[A-Z]{2,4}\d+\b',
        r'\b\d+[A-Z]{2,4}\b',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    return text

def remove_bank_codes(text: str) -> str:
    """Remove bank-specific codes and identifiers."""
    patterns = [
        # ANZ ATM codes
        r'\banz\s+s\d[a-z]\d+\b',
        r'\bs\d[a-z]\d+\b',
        
        # Generic bank location codes
        r'\b(?:br|branch)\s*\d+\b',
        r'\b[A-Z]{2,}\s+\d+\b',
        
        # Currency conversion indicators
        r'\bconverted\s+at\s+[\d.]+\b',
        r'\bincludes?\s+.*?conversion\s+charge\b',
        
        # International codes
        r'\b[A-Z]{3}\s+[\d.]+\s+converted\b',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    return text

def remove_merchant_codes(text: str) -> str:
    """Remove merchant-specific codes and identifiers."""
    patterns = [
        # Store numbers and branch codes
        r'\b(?:store|branch|outlet|location)\s*#?\d+\b',
        r'\b#\d{3,}\b',
        r'\b-\s*\d{3,5}$',  # Trailing store numbers
        r'\b[A-Z]+\s*\d{3,5}$',  # e.g., "BAYF 123"
        
        # More specific location suffixes (common NZ location codes)
        # Only remove these specific known location codes, not all 2-4 letter combinations
        r'\b(?:mt|bl|ra|pt|rd|st|av|pk|sh|ctr|ct)\b$',  # Known location abbreviations
        r'\b[A-Z]\s*\d+[A-Z]?$',  # e.g., "P", "O", "N" at end
        
        # Generic merchant identifiers
        r'\b\d{4,8}[A-Z]*\b',  # Long numbers (SKUs, etc.)
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    return text

def remove_transaction_prefixes(text: str) -> str:
    """Remove common transaction type prefixes that appear before merchant names."""
    patterns = [
        # Common bank transaction prefixes
        r'^\s*[A-Z]{1,4}\s+',  # Remove 1-4 letter prefixes at start like "DF ", "IF ", "SP "
        r'^\s*(?:debit|credit|visa|eftpos|mastercard)\s+',  # Explicit payment type prefixes
        r'^\s*\d{4}-\*+\s*',  # Remove remaining card number fragments
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()

def extract_merchant_name(text: str) -> str:
    """Extract the most likely merchant name using NLP and heuristics."""
    if not nlp or not text:
        return text
    
    # Check for quoted text first (highest priority)
    quoted_match = re.search(r'"([^"]+)"', text)
    if quoted_match:
        return quoted_match.group(1).strip()
    
    # Process with spaCy
    doc = nlp(text)
    
    # Collect organization entities
    org_entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    if org_entities:
        # Return the longest organization entity
        return max(org_entities, key=len)
    
    # Fallback: extract meaningful tokens - be more inclusive
    meaningful_tokens = []
    for token in doc:
        # Skip noise tokens
        if hasattr(token._, 'is_transaction_noise') and token._.is_transaction_noise:
            continue
        if token.is_punct or token.is_space:
            continue
        # Be more inclusive - include more token types and don't skip stop words that might be part of brand names
        if token.pos_ in {"PROPN", "NOUN", "X", "ADJ", "NUM"} or not token.is_stop:
            meaningful_tokens.append(token.text)
    
    if meaningful_tokens:
        result = " ".join(meaningful_tokens)
        # If result is very short (2 characters or less), return original text instead
        if len(result.strip()) <= 2:
            return text
        return result
    
    return text

def normalize_merchant_name(text: str) -> str:
    """Normalize merchant names for better grouping."""
    if not text:
        return ""
    
    # Common merchant normalizations
    normalizations = {
        # Remove common prefixes/suffixes
        r'\b(?:the|a|an)\s+': '',
        r'\s+(?:inc|ltd|llc|corp|co|limited|corporation)\b': '',
        r'\s+(?:pty|gmbh|ag|bv)\b': '',
        
        # Normalize common variations
        r'\bst\b': 'street',
        r'\brd\b': 'road',
        r'\bave?\b': 'avenue',
        r'\bdr\b': 'drive',
        r'\bmt\b': 'mount',
        
        # Remove asterisks and special characters
        r'[*#@$%&]': '',
        r'\.com\b': '',
        
        # Normalize spacing
        r'\s+': ' ',
    }
    
    text_lower = text.lower()
    for pattern, replacement in normalizations.items():
        text_lower = re.sub(pattern, replacement, text_lower, flags=re.IGNORECASE)
    
    return text_lower.strip()

def clean_transaction_text(text: str, config: Optional[CleaningConfig] = None) -> str:
    """Main function to clean transaction text generically."""
    if not config:
        config = CleaningConfig()
    
    if not text or not isinstance(text, str):
        return ""
    
    original_text = text
    
    # Step 1: Remove URLs and emails
    if config.remove_urls_emails:
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Step 2: Remove transaction type prefixes (before detailed cleaning)
    text = remove_transaction_prefixes(text)
    
    # Step 3: Remove card numbers
    if config.remove_card_numbers:
        text = remove_card_numbers(text)
    
    # Step 4: Remove reference codes
    if config.remove_reference_codes:
        text = remove_reference_codes(text)
    
    # Step 5: Remove bank codes
    if config.remove_bank_codes:
        text = remove_bank_codes(text)
    
    # Step 6: Remove merchant codes
    if config.remove_merchant_codes:
        text = remove_merchant_codes(text)
    
    # Step 7: Normalize whitespace
    if config.normalize_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 8: Extract merchant name using NLP
    if config.use_nlp_extraction and nlp:
        text = extract_merchant_name(text)
    
    # Step 9: Normalize merchant name
    if config.normalize_merchant_names:
        text = normalize_merchant_name(text)
    
    # Step 10: Convert to lowercase
    if config.to_lowercase:
        text = text.lower()
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    # If result is empty or too short (likely over-cleaned), fall back to a simpler clean
    if not text or len(text) <= 3:  # Increased threshold from 2 to 3
        # Simple fallback: just remove obvious noise and normalize
        fallback = original_text
        fallback = remove_card_numbers(fallback)
        fallback = re.sub(r'\b\d{4,}\b', '', fallback)  # Remove long numbers only
        fallback = re.sub(r'\s+', ' ', fallback).strip()
        if fallback and len(fallback) > 3:  # Only use fallback if it's longer than 3 chars
            text = fallback.lower() if config.to_lowercase else fallback
        elif original_text:
            # Last resort: use a very minimal clean of original text
            minimal = re.sub(r'\d{4}-\*{4}-\*{4}-\d{4}', '', original_text)  # Remove card pattern
            minimal = re.sub(r'\s+', ' ', minimal).strip()
            if minimal and len(minimal) > 3:
                text = minimal.lower() if config.to_lowercase else minimal
    
    return text

def group_similar_merchants(cleaned_names: List[str], threshold: float = 0.85) -> Dict[str, str]:
    """Group merchants with similar names using fuzzy matching."""
    groups = {}
    canonical_names = {}
    
    for name in cleaned_names:
        if not name:
            continue
            
        best_match = name
        highest_score = 0.0
        
        for canonical in canonical_names.keys():
            score = Levenshtein.ratio(name, canonical)
            if score >= threshold and score > highest_score:
                highest_score = score
                best_match = canonical
        
        if highest_score < threshold:
            # This is a new canonical name
            canonical_names[name] = True
            groups[name] = name
        else:
            # This belongs to an existing group
            groups[name] = best_match
    
    return groups

# --- Embedding-based Grouping Functions ---

def compute_similarity_score(text1: str, text2: str, config: CleaningConfig) -> float:
    """
    Compute similarity between two texts using embedding similarity if available,
    falling back to Levenshtein similarity if embeddings fail.
    
    Args:
        text1: First text
        text2: Second text
        config: Cleaning configuration
        
    Returns:
        Similarity score between 0 and 1
    """
    logger.debug(f"compute_similarity_score called with: '{text1}' vs '{text2}'")
    
    if config.use_embedding_grouping:
        try:
            # Generate embeddings for both texts
            embeddings = generate_embeddings([text1, text2])
            if embeddings is not None and len(embeddings) == 2:
                # Compute cosine similarity
                similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
                similarity = similarity_matrix[0, 0]
                logger.debug(f"Embedding similarity between '{text1}' and '{text2}': {similarity:.4f}")
                return float(similarity)
            else:
                logger.warning(f"Failed to generate embeddings for similarity computation: embeddings shape = {embeddings.shape if embeddings is not None else None}")
        except Exception as e:
            logger.warning(f"Error computing embedding similarity: {e}")
    
    # Fallback to Levenshtein similarity
    levenshtein_score = Levenshtein.ratio(text1, text2)
    logger.debug(f"Levenshtein similarity between '{text1}' and '{text2}': {levenshtein_score:.4f}")
    return levenshtein_score

def group_merchants_with_embeddings(
    cleaned_names: List[str], 
    config: Optional[CleaningConfig] = None
) -> Dict[str, str]:
    """
    Group merchants using embedding-based similarity and clustering.
    
    Args:
        cleaned_names: List of cleaned merchant names
        config: Configuration for embedding grouping
        
    Returns:
        Dictionary mapping each merchant name to its canonical group name
    """
    if not config:
        config = CleaningConfig()
    
    if not EMBEDDINGS_AVAILABLE:
        logger.warning("Embeddings not available, falling back to fuzzy matching")
        if config.embedding_fallback_to_fuzzy:
            return group_similar_merchants(cleaned_names, config.similarity_threshold)
        else:
            return {name: name for name in cleaned_names if name}
    
    # Filter out empty names
    valid_names = [name for name in cleaned_names if name and name.strip()]
    if len(valid_names) < 2:
        return {name: name for name in valid_names}
    
    # For large datasets, process in batches to manage memory
    if len(valid_names) > config.embedding_batch_size:
        logger.info(f"Processing {len(valid_names)} names in batches of {config.embedding_batch_size}")
        return _process_large_dataset_in_batches(valid_names, cleaned_names, config)
    
    # Check cache first (if enabled)
    embeddings = None
    if config.embedding_use_cache:
        cached_embeddings = _get_cached_embeddings(valid_names)
        if cached_embeddings is not None:
            logger.info(f"Using cached embeddings for {len(valid_names)} merchant names")
            embeddings = cached_embeddings
    
    # Generate embeddings if not found in cache
    if embeddings is None:
        try:
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(valid_names)} merchant names")
            embeddings = generate_embeddings(valid_names)
            
            if embeddings is None or embeddings.size == 0:
                logger.warning("Failed to generate embeddings, falling back to fuzzy matching")
                if config.embedding_fallback_to_fuzzy:
                    return group_similar_merchants(cleaned_names, config.similarity_threshold)
                else:
                    return {name: name for name in valid_names}
            
            # Cache the generated embeddings (if caching enabled)
            if config.embedding_use_cache:
                _cache_embeddings(valid_names, embeddings)
        
        except Exception as e:
            logger.error(f"Error in embedding-based grouping: {e}")
            if config.embedding_fallback_to_fuzzy:
                logger.info("Falling back to fuzzy matching")
                return group_similar_merchants(cleaned_names, config.similarity_threshold)
            else:
                return {name: name for name in cleaned_names if name}
    
    # Perform clustering based on method
    # For very small datasets, use similarity clustering regardless of method
    if len(valid_names) <= 5:
        logger.info(f"Small dataset ({len(valid_names)} items), using similarity clustering")
        groups = _cluster_with_similarity(valid_names, embeddings, config)
    elif config.embedding_clustering_method == "hdbscan":
        groups = _cluster_with_hdbscan(valid_names, embeddings, config)
    elif config.embedding_clustering_method == "dbscan":
        groups = _cluster_with_dbscan(valid_names, embeddings, config)
    elif config.embedding_clustering_method == "hierarchical":
        groups = _cluster_with_hierarchical(valid_names, embeddings, config)
    elif config.embedding_clustering_method == "similarity":
        groups = _cluster_with_similarity(valid_names, embeddings, config)
    else:
        logger.warning(f"Unknown clustering method: {config.embedding_clustering_method}")
        groups = _cluster_with_similarity(valid_names, embeddings, config)
    
    # Add back any empty names that were filtered out
    for name in cleaned_names:
        if not name or not name.strip():
            groups[name] = name
            
    return groups

def _cluster_with_hdbscan(names: List[str], embeddings: np.ndarray, config: CleaningConfig) -> Dict[str, str]:
    """Cluster using HDBSCAN algorithm."""
    if not HDBSCAN_AVAILABLE:
        logger.warning("HDBSCAN not available, falling back to similarity clustering")
        return _cluster_with_similarity(names, embeddings, config)
    
    try:
        # Convert to distance metric (1 - cosine similarity)
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix
        
        # Optimize HDBSCAN parameters based on dataset size
        min_cluster_size = max(2, min(config.embedding_min_cluster_size, len(names) // 10))
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='precomputed',
            cluster_selection_epsilon=0.1  # Add parameter for better control
        )
        cluster_labels = clusterer.fit_predict(distance_matrix)
        
        # Log clustering results
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        logger.info(f"HDBSCAN clustering: {len(names)} names -> {n_clusters} clusters, {n_noise} noise points")
        
        return _create_groups_from_labels(names, cluster_labels)
        
    except Exception as e:
        logger.error(f"Error in HDBSCAN clustering: {e}")
        return _cluster_with_similarity(names, embeddings, config)

def _cluster_with_dbscan(names: List[str], embeddings: np.ndarray, config: CleaningConfig) -> Dict[str, str]:
    """Cluster using DBSCAN algorithm."""
    try:
        # Convert to distance metric (1 - cosine similarity)  
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix
        
        # Adaptive eps based on dataset size
        adaptive_eps = min(config.embedding_eps, 0.5 - (len(names) / 1000) * 0.1)
        adaptive_eps = max(0.1, adaptive_eps)  # Don't go below 0.1
        
        clusterer = DBSCAN(
            eps=adaptive_eps,
            min_samples=config.embedding_min_samples,
            metric='precomputed'
        )
        cluster_labels = clusterer.fit_predict(distance_matrix)
        
        # Log clustering results
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        logger.info(f"DBSCAN clustering (eps={adaptive_eps:.3f}): {len(names)} names -> {n_clusters} clusters, {n_noise} noise points")
        
        return _create_groups_from_labels(names, cluster_labels)
        
    except Exception as e:
        logger.error(f"Error in DBSCAN clustering: {e}")
        return _cluster_with_similarity(names, embeddings, config)

def _cluster_with_hierarchical(names: List[str], embeddings: np.ndarray, config: CleaningConfig) -> Dict[str, str]:
    """Cluster using hierarchical clustering."""
    try:
        # Convert to distance metric (1 - cosine similarity)
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distance_matrix[np.triu_indices(len(distance_matrix), k=1)], method='average')
        cluster_labels = fcluster(linkage_matrix, t=config.embedding_hierarchical_threshold, criterion='distance')
        
        # Convert to 0-based indexing
        cluster_labels = cluster_labels - 1
        
        return _create_groups_from_labels(names, cluster_labels)
        
    except Exception as e:
        logger.error(f"Error in hierarchical clustering: {e}")
        return _cluster_with_similarity(names, embeddings, config)

def _cluster_with_similarity(names: List[str], embeddings: np.ndarray, config: CleaningConfig) -> Dict[str, str]:
    """Cluster using simple similarity threshold with proper transitive grouping prevention."""
    try:
        logger.info(f"_cluster_with_similarity called with {len(names)} names and threshold {config.embedding_similarity_threshold}")
        similarity_matrix = cosine_similarity(embeddings)
        logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
        
        # Log the similarity matrix for debugging
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i < j:  # Only log upper triangle to avoid duplicates
                    logger.info(f"Similarity between '{name1}' and '{name2}': {similarity_matrix[i][j]:.4f}")
        
        groups = {}
        group_members = {}  # Track all members of each group
        
        for i, name in enumerate(names):
            logger.debug(f"Processing name {i}: '{name}'")
            best_group = None
            best_min_similarity = 0.0  # Track the best minimum similarity to group members
            
            # Check against all existing groups
            for canonical, members in group_members.items():
                # For this name to join a group, it must be similar to ALL existing members
                similarities_to_group = []
                for member_name in members:
                    member_idx = names.index(member_name)
                    score = similarity_matrix[i][member_idx]
                    similarities_to_group.append(score)
                    logger.debug(f"  Similarity to group member '{member_name}': {score:.4f}")
                
                # Check if similar to ALL members in the group
                min_similarity = min(similarities_to_group)
                logger.debug(f"  Min similarity to group '{canonical}': {min_similarity:.4f}")
                
                if min_similarity >= config.embedding_similarity_threshold:
                    # This name can join this group - pick the group with highest minimum similarity
                    if min_similarity > best_min_similarity:
                        best_min_similarity = min_similarity
                        best_group = canonical
                        logger.debug(f"    New best group candidate: '{best_group}' with min similarity {best_min_similarity:.4f}")
            
            if best_group is None:
                # This is a new group - no existing group had ALL members similar enough
                logger.info(f"'{name}' becomes new group canonical (no group had all members >= threshold {config.embedding_similarity_threshold})")
                groups[name] = name
                group_members[name] = [name]
            else:
                # This belongs to an existing group
                logger.info(f"'{name}' grouped with '{best_group}' (min group similarity {best_min_similarity:.4f} >= threshold {config.embedding_similarity_threshold})")
                groups[name] = best_group
                group_members[best_group].append(name)
        
        # Log clustering results
        n_groups = len(set(groups.values()))
        reduction = ((len(names) - n_groups) / len(names) * 100) if len(names) > 0 else 0
        logger.info(f"Similarity clustering (threshold={config.embedding_similarity_threshold}): {len(names)} names -> {n_groups} groups ({reduction:.1f}% reduction)")
        logger.info(f"Final groups: {groups}")
        logger.info(f"Group members: {group_members}")
        
        return groups
        
    except Exception as e:
        logger.error(f"Error in similarity clustering: {e}")
        return {name: name for name in names}

def _create_groups_from_labels(names: List[str], labels: np.ndarray) -> Dict[str, str]:
    """Create grouping dictionary from cluster labels."""
    groups = {}
    cluster_representatives = {}
    
    for i, (name, label) in enumerate(zip(names, labels)):
        if label == -1:  # Noise/outlier
            groups[name] = name
        else:
            if label not in cluster_representatives:
                # First name in this cluster becomes the representative
                cluster_representatives[label] = name
                groups[name] = name
            else:
                # Assign to existing representative
                groups[name] = cluster_representatives[label]
    
    return groups

# Caching for embeddings to improve performance
_embedding_cache: Dict[str, np.ndarray] = {}
_cache_max_size = 1000  # Maximum number of cached embeddings

def _get_cache_key(texts: List[str]) -> str:
    """Generate a cache key for a list of texts."""
    combined = "|".join(sorted(texts))
    return hashlib.md5(combined.encode()).hexdigest()

def _get_cached_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    """Retrieve cached embeddings if available."""
    cache_key = _get_cache_key(texts)
    return _embedding_cache.get(cache_key)

def _cache_embeddings(texts: List[str], embeddings: np.ndarray) -> None:
    """Cache embeddings for future use."""
    if len(_embedding_cache) >= _cache_max_size:
        # Remove oldest entries (simple FIFO)
        keys_to_remove = list(_embedding_cache.keys())[:len(_embedding_cache) - _cache_max_size + 1]
        for key in keys_to_remove:
            del _embedding_cache[key]
    
    cache_key = _get_cache_key(texts)
    _embedding_cache[cache_key] = embeddings

def clean_and_group_transactions(
    texts: List[str], 
    config: Optional[CleaningConfig] = None
) -> Tuple[List[str], Dict[str, str]]:
    """
    Clean transaction texts and group similar merchants.
    
    Args:
        texts: List of transaction texts to clean and group
        config: Configuration for cleaning and grouping
        
    Returns:
        Tuple of (cleaned_texts, grouping_dict)
    """
    if not config:
        config = CleaningConfig()
    
    # Clean all texts
    cleaned_texts = [clean_transaction_text(text, config) for text in texts]
    
    # Group similar merchants
    if config.use_embedding_grouping and EMBEDDINGS_AVAILABLE:
        grouping_dict = group_merchants_with_embeddings(cleaned_texts, config)
    elif config.use_fuzzy_matching:
        grouping_dict = group_similar_merchants(cleaned_texts, config.similarity_threshold)
    else:
        grouping_dict = {text: text for text in cleaned_texts}
    
    return cleaned_texts, grouping_dict

# --- Legacy Functions for Backward Compatibility ---

company_suffixes = {
    "inc", "ltd", "llc", "pty", "corp", "gmbh", "ag", "bv", "co", "limited", "corporation",
}

MERCHANT_CATEGORIES = {
    "supermarkets": ["woolworths", "pak n save", "new world", "four square", "freshchoice"],
    "fuel": ["z energy", "bp", "mobil", "caltex", "gull", "rd petroleum", "kiwi fuel"],
    "retail": ["kmart", "warehouse", "farmers", "cotton on", "noel leeming"],
    "health": ["chemist warehouse", "life pharmacy", "unichem"],
    "food": ["uber eats", "menulog", "pizza", "burger", "coffee", "cafe"],
}

MERCHANT_ALIASES = {
    "woolworths": ["woolworths n", "woolworths o", "woolies"],
    "pak n save": ["pak n save p", "pak n save fuel"],
    "new world": ["new world mt", "new world bl", "new world ra"],
    "four square": ["four square"],
    "warehouse": ["the warehous", "warehouse st"],
    "anz atm": ["anz s3d7741", "anz s3c7711", "anz s3a1490"],
}

def extract_nlp_features(text: str, config: CleaningConfig = None) -> str:
    """Legacy function - now calls the generic clean_transaction_text."""
    if not config:
        config = CleaningConfig()
    return clean_transaction_text(text, config)

def normalize_merchant_variants(text: str) -> str:
    """Normalize common merchant name variations."""
    if not text:
        return ""
    
    merchant_normalizations = {
        r"woolworths?\s*[a-z]?": "woolworths",
        r"woolies?\s*[a-z]?": "woolworths",
        r"pak\s*n\s*save\s*[a-z]?": "pak n save",
        r"new\s*world\s*[a-z]{2}": "new world",
        r"^the\s+": "",
        r"\s+the$": "",
        r"anz\s+s\d[a-z]\d+": "anz atm",
        r"kiwi\s+fuels?": "kiwi fuel",
        r"four\s+square": "four square",
    }
    
    text_lower = text.lower()
    for pattern, replacement in merchant_normalizations.items():
        text_lower = re.sub(pattern, replacement, text_lower, flags=re.IGNORECASE)
    
    return text_lower.strip()

def resolve_merchant_alias(cleaned_name: str) -> str:
    """Resolve merchant name to its canonical form using aliases."""
    if not cleaned_name:
        return ""
    
    cleaned_name_lower = cleaned_name.lower()
    for canonical, aliases in MERCHANT_ALIASES.items():
        for alias in aliases:
            if alias in cleaned_name_lower:
                return canonical
    return cleaned_name

def categorize_merchant(cleaned_name: str) -> str:
    """Assign a category to a merchant based on keywords."""
    if not cleaned_name:
        return "other"
    
    cleaned_name_lower = cleaned_name.lower()
    for category, keywords in MERCHANT_CATEGORIES.items():
        for keyword in keywords:
            if keyword in cleaned_name_lower:
                return category
    return "other"

def enhanced_clean_text(text: str, config: CleaningConfig = None) -> str:
    """Legacy function - now calls the generic clean_transaction_text."""
    if not config:
        config = CleaningConfig()
    return clean_transaction_text(text, config)

# Main function for backward compatibility
def clean_text(texts: Union[str, List[str]], config: Optional[CleaningConfig] = None) -> Union[str, List[str], Tuple[List[str], Dict[str, str]]]:
    """
    Clean transaction text(s) generically.
    
    Args:
        texts: Single text string or list of text strings
        config: Configuration for cleaning and grouping
        
    Returns:
        - If input is string: cleaned string
        - If input is list and grouping disabled: list of cleaned strings
        - If input is list and grouping enabled: tuple of (cleaned_list, grouping_dict)
    """
    if not config:
        config = CleaningConfig()
    
    if isinstance(texts, str):
        return clean_transaction_text(texts, config)
    elif isinstance(texts, list):
        # Check if we should perform grouping
        should_group = (
            config.use_embedding_grouping or 
            config.use_fuzzy_matching or 
            config.use_fuzzy_matching_post_clean
        )
        
        if should_group:
            # Return both cleaned texts and grouping
            cleaned_texts, grouping_dict = clean_and_group_transactions(texts, config)
            
            if len(cleaned_texts) != len(texts):
                logger.error(f"CRITICAL: Length mismatch after cleaning! Input: {len(texts)}, Output: {len(cleaned_texts)}")
                return texts
            
            return cleaned_texts, grouping_dict
        else:
            # Just clean, no grouping
            cleaned_texts = [clean_transaction_text(text, config) for text in texts]
            
            if len(cleaned_texts) != len(texts):
                logger.error(f"CRITICAL: Length mismatch after cleaning! Input: {len(texts)}, Output: {len(cleaned_texts)}")
                return texts
            
            return cleaned_texts
    else:
        logger.warning(f"Unexpected input type for clean_text: {type(texts)}. Returning as is.")
        return texts

def _process_large_dataset_in_batches(
    valid_names: List[str], 
    all_names: List[str], 
    config: CleaningConfig
) -> Dict[str, str]:
    """
    Process large datasets in batches to manage memory efficiently.
    
    Args:
        valid_names: List of non-empty merchant names
        all_names: Original list including empty names
        config: Configuration for embedding grouping
        
    Returns:
        Dictionary mapping each merchant name to its canonical group name
    """
    batch_size = config.embedding_batch_size
    all_groups = {}
    canonical_mapping = {}
    canonical_embeddings = {}  # Cache embeddings for canonical names
    
    # Process in batches
    for i in range(0, len(valid_names), batch_size):
        batch_names = valid_names[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_names)} names")
        
        # Get embeddings for this batch
        try:
            if config.embedding_use_cache:
                cached_embeddings = _get_cached_embeddings(batch_names)
                if cached_embeddings is not None:
                    embeddings = cached_embeddings
                else:
                    embeddings = generate_embeddings(batch_names)
                    if embeddings is not None:
                        _cache_embeddings(batch_names, embeddings)
            else:
                embeddings = generate_embeddings(batch_names)
                
            if embeddings is None or embeddings.size == 0:
                # Fallback for this batch
                for name in batch_names:
                    all_groups[name] = name
                continue
                
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
            # Fallback for this batch
            for name in batch_names:
                all_groups[name] = name
            continue
        
        # Cluster this batch
        if config.embedding_clustering_method == "similarity":
            batch_groups = _cluster_with_similarity(batch_names, embeddings, config)
        else:
            # For other clustering methods, use similarity as it's more suitable for batches
            batch_groups = _cluster_with_similarity(batch_names, embeddings, config)
        
        # Extract group representatives and their embeddings from this batch
        group_rep_to_embedding = {}
        for name, group_rep in batch_groups.items():
            if group_rep not in group_rep_to_embedding:
                # Find the embedding for this group representative
                group_rep_idx = batch_names.index(group_rep)
                group_rep_to_embedding[group_rep] = embeddings[group_rep_idx]
        
        # Merge with existing groups and resolve conflicts using efficient embedding comparison
        for name, group_rep in batch_groups.items():
            # Check if this group representative already exists in our global mapping
            existing_canonical = canonical_mapping.get(group_rep)
            if existing_canonical:
                all_groups[name] = existing_canonical
            else:
                # Check if we have a similar canonical name from previous batches
                best_match = group_rep
                best_score = 0.0
                
                if canonical_embeddings:
                    # Use efficient embedding similarity computation
                    group_rep_embedding = group_rep_to_embedding[group_rep]
                    
                    # Compute similarities to all existing canonicals in batch
                    canonical_names = list(canonical_embeddings.keys())
                    canonical_emb_matrix = np.array(list(canonical_embeddings.values()))
                    
                    if len(canonical_names) > 0:
                        # Compute cosine similarity between group_rep and all canonicals
                        group_rep_emb_matrix = np.array([group_rep_embedding])
                        similarity_scores = cosine_similarity(group_rep_emb_matrix, canonical_emb_matrix)[0]
                        
                        for j, canonical in enumerate(canonical_names):
                            score = similarity_scores[j]
                            if score >= config.embedding_similarity_threshold and score > best_score:
                                best_score = score
                                best_match = canonical
                
                if best_score >= config.embedding_similarity_threshold:
                    all_groups[name] = best_match
                    canonical_mapping[group_rep] = best_match
                else:
                    all_groups[name] = group_rep
                    canonical_mapping[group_rep] = group_rep
                    # Cache embedding for this new canonical
                    canonical_embeddings[group_rep] = group_rep_to_embedding[group_rep]
    
    # Add back any empty names that were filtered out
    for name in all_names:
        if not name or not name.strip():
            all_groups[name] = name
    
    logger.info(f"Batch processing complete. {len(valid_names)} names -> {len(set(all_groups.values()))} groups")
    return all_groups