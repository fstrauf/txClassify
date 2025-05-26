"""Utility functions for cleaning and processing text data using spaCy NER."""

import re
import logging
import spacy
import os
import csv  # Added for reading CSV
import ahocorasick  # Added for Aho-Corasick
import Levenshtein  # NEW IMPORT
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Token, Doc
from typing import List, Union, Dict
from config import USE_NZ_BUSINESS_DATA_MATCHING  # Added import

logger = logging.getLogger(__name__)


# --- Define CleaningConfig Class FIRST ---
class CleaningConfig:
    def __init__(self):
        self.remove_card_numbers = True
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
        return {
            "remove_card_numbers": self.remove_card_numbers,
            "normalize_merchants": self.normalize_merchants,
            "use_fuzzy_matching_post_clean": self.use_fuzzy_matching_post_clean,
            "similarity_threshold_post_clean": self.similarity_threshold_post_clean,
            "group_by_category_post_clean": self.group_by_category_post_clean,
            "apply_nlp_features_extraction": self.apply_nlp_features_extraction,
            "apply_merchant_variants_normalization": self.apply_merchant_variants_normalization,
            "apply_alias_resolution": self.apply_alias_resolution,
            "perform_aggressive_store_number_removal": self.perform_aggressive_store_number_removal,
            "perform_company_suffix_removal": self.perform_company_suffix_removal,
        }


# --- END CleaningConfig Class ---

# --- Aho-Corasick Automaton for NZ Businesses ---
NZ_BUSINESS_AUTOMATON = ahocorasick.Automaton()
NZ_BUSINESS_DATA_FILEPATH = os.getenv(
    "NZ_BUSINESS_DATA_PATH", "data/nz_business_data.csv"
)
# --- End Aho-Corasick ---

# --- Define Noise Patterns and Custom Component FIRST ---

# Patterns for the component added to the main pipeline (transaction noise)
transaction_noise_patterns = [
    # Common transaction noise words/phrases
    [{"LOWER": "tap"}, {"LOWER": "and", "OP": "?"}, {"LOWER": "pay"}],
    [{"LOWER": "direct"}, {"LOWER": "debit"}],
    [{"LOWER": "eftpos"}],
    [{"LOWER": "purchase"}],
    [{"LOWER": "authorised"}],
    [{"LOWER": "payment"}],
    [{"LOWER": "netbank"}],
    [{"LOWER": "commbank"}],
    [{"LOWER": "app"}],
    # Common prefixes observed in logs
    [{"LOWER": "df"}],  # Often appears with card numbers
    [{"LOWER": "if"}],  # Often appears with card numbers or online services
    [{"LOWER": "sp"}],  # Often a prefix for Shopify or similar payment processors
    # Keywords often related to noise (can be part of ORG, matcher checks this)
    [{"LOWER": "card"}],
    [{"LOWER": "value"}],
    [{"LOWER": "date"}],
    [{"LOWER": "ref"}],
    [{"LOWER": "reference"}],
    [{"LOWER": "txn"}],
    # Specific digit sequences (>= 5 digits)
    [{"IS_DIGIT": True, "LENGTH": {">=": 5}}],
]

# Patterns for generic location noise (used *inside* extract_nlp_features)
location_noise_patterns = [
    # Common Country Codes/Names (English speaking focus for now)
    [{"LOWER": "au"}],
    [{"LOWER": "aus"}],
    [{"LOWER": "australia"}],
    [{"LOWER": "us"}],
    [{"LOWER": "usa"}],
    [{"LOWER": "united"}, {"LOWER": "states"}],
    [{"LOWER": "gb"}],
    [{"LOWER": "gbr"}],
    [{"LOWER": "uk"}],
    [{"LOWER": "united"}, {"LOWER": "kingdom"}],
    [{"LOWER": "ca"}],
    [{"LOWER": "can"}],
    [{"LOWER": "canada"}],
    [{"LOWER": "nz"}],
    [{"LOWER": "nzl"}],
    [{"LOWER": "new"}, {"LOWER": "zealand"}],
    # Common AU State Codes (Example - expand as needed)
    [{"LOWER": "nsw"}],
    [{"LOWER": "vic"}],
    [{"LOWER": "qld"}],
    [{"LOWER": "wa"}],
    [{"LOWER": "sa"}],
    [{"LOWER": "tas"}],
    [{"LOWER": "act"}],
    [{"LOWER": "nt"}],
    # --- NEW --- Specific common cities/suburbs often appearing as noise
    [{"LOWER": "dee"}],  # For "DEE WHY"
    [{"LOWER": "ns"}],  # Common noise observed
]

# --- Custom spaCy Component (Handles Transaction Noise) ---
if not Token.has_extension("is_transaction_noise"):
    Token.set_extension("is_transaction_noise", default=False)


@Language.component("transaction_noise_filter")  # Component name matches add_pipe
def transaction_noise_filter_component(doc: Doc) -> Doc:
    """
    Marks transaction-specific noise patterns.
    Sets token._.is_transaction_noise = True
    """
    # Matcher is instantiated here, specific to this component call
    matcher = Matcher(doc.vocab)
    matcher.add("TRANSACTION_NOISE", transaction_noise_patterns)
    matches = matcher(doc)

    for match_id, start, end in matches:
        pattern_name = doc.vocab.strings[match_id]
        span = doc[start:end]
        logger.debug(f"Transaction Noise Matcher: '{span.text}' ({pattern_name})")
        for i in range(start, end):
            doc[i]._.is_transaction_noise = True
    return doc


# --- Load spaCy Model and Add Custom Component (AFTER component definition) ---
nlp = None
SPA_MODEL_NAME = "en_core_web_sm"


# --- Function to load NZ Business Data into Aho-Corasick Automaton ---
def load_nz_business_data():
    """Loads NZ business names from a CSV file into the Aho-Corasick automaton."""
    global NZ_BUSINESS_AUTOMATON
    loaded_count = 0
    if not os.path.exists(NZ_BUSINESS_DATA_FILEPATH):
        logger.warning(
            f"NZ business data file not found at {NZ_BUSINESS_DATA_FILEPATH}. Automaton will be empty."
        )
        return

    try:
        with open(NZ_BUSINESS_DATA_FILEPATH, mode="r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            if "ENTITY_NAME" not in reader.fieldnames:
                logger.error(
                    f"'ENTITY_NAME' column not found in {NZ_BUSINESS_DATA_FILEPATH}. Cannot load NZ business names."
                )
                return
            for row in reader:
                business_name = row["ENTITY_NAME"]
                if business_name and business_name.strip():
                    # Store names in lowercase for case-insensitive matching
                    # The value stored can be the name itself or any other identifier
                    NZ_BUSINESS_AUTOMATON.add_word(
                        business_name.strip().lower(), business_name.strip().lower()
                    )
                    loaded_count += 1
        if loaded_count > 0:
            NZ_BUSINESS_AUTOMATON.make_automaton()
            logger.info(
                f"Successfully loaded {loaded_count} NZ business names into Aho-Corasick automaton from {NZ_BUSINESS_DATA_FILEPATH}."
            )
        else:
            logger.warning(
                f"No NZ business names loaded from {NZ_BUSINESS_DATA_FILEPATH}. File might be empty or names are invalid."
            )
    except FileNotFoundError:
        logger.error(
            f"NZ business data file not found: {NZ_BUSINESS_DATA_FILEPATH}. Ensure the path is correct."
        )
    except Exception as e:
        logger.error(
            f"Error loading NZ business data from {NZ_BUSINESS_DATA_FILEPATH}: {e}",
            exc_info=True,
        )


# --- End NZ Business Data Loading ---


try:
    logger.info(f"Attempting to load spaCy model by name: {SPA_MODEL_NAME}")
    nlp = spacy.load(SPA_MODEL_NAME)
    logger.info(f"Successfully loaded spaCy model: {SPA_MODEL_NAME}")

    COMPONENT_NAME = "transaction_noise_filter"
    if "ner" in nlp.pipe_names:
        if not nlp.has_pipe(COMPONENT_NAME):
            nlp.add_pipe(COMPONENT_NAME, after="ner")
            logger.info(
                f"spaCy model '{SPA_MODEL_NAME}' loaded and '{COMPONENT_NAME}' added after ner."
            )
        else:
            logger.info(f"'{COMPONENT_NAME}' already in spaCy pipeline (after ner).")
    else:
        if not nlp.has_pipe(COMPONENT_NAME):
            nlp.add_pipe(COMPONENT_NAME, last=True)
            logger.warning(
                f"spaCy 'ner' component not found. Added '{COMPONENT_NAME}' last."
            )
        else:
            logger.warning(f"'{COMPONENT_NAME}' already in spaCy pipeline (last).")

    logger.info(f"Pipeline components: {nlp.pipe_names}")

    # --- Load NZ Business Data After NLP Model ---
    if USE_NZ_BUSINESS_DATA_MATCHING:  # Conditional loading
        load_nz_business_data()
    else:
        logger.info(
            "USE_NZ_BUSINESS_DATA_MATCHING is False. Skipping NZ business data load."
        )
    # ---

except OSError as e:
    logger.error(f"OSError loading spaCy model '{SPA_MODEL_NAME}' by name. Error: {e}")
    logger.error(
        f"Ensure '{SPA_MODEL_NAME}' was downloaded and installed correctly in the Dockerfile."
    )
    raise
except Exception as e:
    logger.error(f"Error adding custom component to spaCy pipeline: {e}")
    # nlp remains None, functions below will handle this


# --- Define Other Global Variables (e.g., company_suffixes) ---
company_suffixes = {
    "inc",
    "ltd",
    "llc",
    "pty",
    "corp",
    "gmbh",
    "ag",
    "bv",
    "co",
    "limited",
    "corporation",
}


# --- Feature Extraction and Cleaning Functions ---
def extract_nlp_features(text: str, config: CleaningConfig = None) -> str:
    """Extracts the most likely merchant name using spaCy NER, POS, Matchers, and heuristics."""
    if not config:  # Default config if none provided for direct calls to this function
        config = CleaningConfig()

    if not nlp or not text:
        return text

    text_for_spacy_pipeline = text  # Default to the input text

    # --- Step -1: Aho-Corasick Matching for NZ Businesses (PRIORITY) ---
    if USE_NZ_BUSINESS_DATA_MATCHING:  # Conditional matching
        # Match against the lowercased version of the input text
        text_to_match_lower = text.lower()
        longest_match_original_casing = None
        longest_match_len = 0

        # Check if automaton has words before iterating
        if (
            NZ_BUSINESS_AUTOMATON.kind != ahocorasick.EMPTY
        ):  # kind should be ahocorasick.AHOCORASICK
            try:  # make_automaton might not have been called if file was empty or error
                for (
                    end_index_in_lower,
                    matched_value_lower,
                ) in NZ_BUSINESS_AUTOMATON.iter(text_to_match_lower):
                    start_index_in_lower = (
                        end_index_in_lower - len(matched_value_lower) + 1
                    )
                    current_match_len = len(matched_value_lower)
                    if current_match_len > longest_match_len:
                        longest_match_len = current_match_len
                        # Extract the original cased substring from the input 'text'
                        longest_match_original_casing = text[
                            start_index_in_lower : end_index_in_lower + 1
                        ]
            except Exception as e:
                logger.error(f"Error during Aho-Corasick matching: {e}", exc_info=True)

        if longest_match_original_casing:
            text_for_spacy_pipeline = longest_match_original_casing
            logger.debug(
                f"Prioritized NZ Business Match (Aho-Corasick): '{text_for_spacy_pipeline}' from '{text}'"
            )
        else:
            # --- Step 0: Check for Quoted Text (Fallback if no NZ Biz Match or if feature is off) ---
            quoted_match = re.search(r'"(.+?)"', text)  # text is pre_cleaned_text
            if quoted_match:
                extracted_name = quoted_match.group(1).strip()
                extracted_name = re.sub(r"\\s+", " ", extracted_name).strip()
                if extracted_name:
                    text_for_spacy_pipeline = extracted_name
                    logger.debug(
                        f"Prioritized Quoted Text (NZ Biz Feature OFF): '{text_for_spacy_pipeline}' from '{text}'"
                    )
    else:
        # --- Step 0: Check for Quoted Text (Fallback if NZ Biz Match feature is off) ---
        logger.debug(
            "USE_NZ_BUSINESS_DATA_MATCHING is False. Skipping Aho-Corasick, proceeding to quoted text check."
        )
        quoted_match = re.search(r'"(.+?)"', text)
        if quoted_match:
            extracted_name = quoted_match.group(1).strip()
            extracted_name = re.sub(r"\\s+", " ", extracted_name).strip()
            if extracted_name:
                text_for_spacy_pipeline = extracted_name
                logger.debug(
                    f"Prioritized Quoted Text (NZ Biz Feature OFF): '{text_for_spacy_pipeline}' from '{text}'"
                )
    # If neither Aho-Corasick (if enabled) nor quoted text match, text_for_spacy_pipeline remains the original 'text'

    # --- Process with spaCy Pipeline (NER + Transaction Noise Matcher) ---
    # --- Disable only \"lemmatizer\" as \"parser\" is needed for noun_chunks fallback ---
    doc = nlp(text_for_spacy_pipeline, disable=["lemmatizer"])

    # --- Define POS tags to generally keep ---
    KEEP_POS = {"PROPN", "NOUN", "SYM"}  # Keep Proper Nouns, Nouns, Symbols

    # --- Step 1: Identify ORG entities via NER (for protection) ---
    org_indices = set()

    for ent in doc.ents:
        if ent.label_ == "ORG":
            for i in range(ent.start, ent.end):
                org_indices.add(i)
            logger.debug(f"NER identified ORG: '{ent.text}'")
        # else: We don't primarily filter based on other NER labels now

    # --- Step 2: Identify Generic Location Noise with dedicated Matcher ---
    location_matcher = Matcher(doc.vocab)
    location_matcher.add("LOCATION_NOISE", location_noise_patterns)
    location_matches = location_matcher(doc)
    location_noise_indices = set()
    for match_id, start, end in location_matches:
        # Mark indices for potential removal, check ORG overlap later
        for i in range(start, end):
            location_noise_indices.add(i)
        logger.debug(
            f"Location Matcher identified potential noise: '{doc[start:end].text}'"
        )

    # --- Step 3: Build Filtered List (Prioritize Noise Removal) ---
    initial_filtered_tokens = []  # List of Token objects
    transaction_noise_indices = {
        i for i, token in enumerate(doc) if token._.is_transaction_noise
    }

    for i, token in enumerate(doc):
        # Skip punctuation unless it's part of a number/currency or hyphenated word
        # Let's keep hyphens if they connect alphanumeric chars, otherwise discard.
        is_hyphen_ok = (
            token.text == "-"
            and i > 0
            and i < len(doc) - 1
            and doc[i - 1].is_alpha
            and doc[i + 1].is_alpha
        )
        if token.is_punct and not is_hyphen_ok:
            logger.debug(f"Removing '{token.text}' (POS: {token.pos_}) - Punctuation")
            continue

        # Skip stop words unless part of an ORG/PRODUCT entity
        # (Entities like "Bank of America" should keep "of")
        is_entity_part = token.ent_iob_ != "O" and token.ent_type_ in {
            "ORG",
            "PRODUCT",
        }
        if token.is_stop and not is_entity_part:
            logger.debug(f"Removing '{token.text}' (Stop word, not in ORG/PRODUCT)")
            continue

        # Check noise flags
        is_loc_match = i in location_noise_indices
        is_txn_noise = i in transaction_noise_indices  # Check the set
        is_noise = is_loc_match or is_txn_noise

        # Keep token if it's not noise
        if not is_noise:
            initial_filtered_tokens.append(token)
        else:
            noise_type = "Location Matcher" if is_loc_match else "Transaction Matcher"
            logger.debug(f"Removing '{token.text}' ({noise_type})")

    # --- Step 3.5: Refine Tokens - Discard GPE if ORG/PRODUCT appears first ---
    refined_filtered_tokens = []
    min_org_prod_token_idx = None
    min_gpe_token_idx = None

    # Find the first occurrence index *within initial_filtered_tokens*
    for idx, token in enumerate(initial_filtered_tokens):
        if token.ent_type_ in {"ORG", "PRODUCT"} and min_org_prod_token_idx is None:
            min_org_prod_token_idx = idx
        if token.ent_type_ == "GPE" and min_gpe_token_idx is None:
            min_gpe_token_idx = idx
        # Optimization: stop if both found
        if min_org_prod_token_idx is not None and min_gpe_token_idx is not None:
            break

    discard_gpe = False
    if (
        min_org_prod_token_idx is not None
        and min_gpe_token_idx is not None
        and min_org_prod_token_idx < min_gpe_token_idx
    ):
        discard_gpe = True
        logger.debug(
            f"Primary entity (ORG/PRODUCT at index {min_org_prod_token_idx}) found before location (GPE at index {min_gpe_token_idx}). Discarding GPE tokens."
        )

    if discard_gpe:
        for token in initial_filtered_tokens:
            if token.ent_type_ == "GPE":
                logger.debug(f"Discarding '{token.text}' (GPE after primary entity)")
                continue
            refined_filtered_tokens.append(token)
    else:
        # If not discarding GPE, use the initial list
        refined_filtered_tokens = initial_filtered_tokens

    # If all tokens were filtered out, return original text after basic regex
    if not refined_filtered_tokens:
        logger.warning(
            f"All tokens filtered out for '{text}'. Returning basic cleaned: '{text}'"
        )
        return text

    # --- Step 4: Build Contiguous Text Spans from Refined Tokens ---
    contiguous_token_texts = []
    current_span = []
    if refined_filtered_tokens:
        last_token_index = refined_filtered_tokens[0].i - 1
        for token in refined_filtered_tokens:
            # If current token does not follow the last one, start a new span
            if token.i != last_token_index + 1 and current_span:
                contiguous_token_texts.append(" ".join(current_span))
                current_span = []
            current_span.append(token.text)
            last_token_index = token.i
        # Add the last span
        if current_span:
            contiguous_token_texts.append(" ".join(current_span))

    if not contiguous_token_texts:
        logger.warning(
            f"No contiguous text spans formed for '{text}'. Returning basic cleaned: '{text}'"
        )
        return text

    # --- Step 5: Select Best Text Span (longest is usually best) ---
    # We primarily care about the main entity description
    final_tokens_text = max(contiguous_token_texts, key=len)
    logger.debug(f"Text after POS/Matcher/GPE filtering: '{final_tokens_text}'")

    # --- Noun Chunk Augmentation Logic (NEW) ---
    initial_derived_text = final_tokens_text
    # Define very common, usually noisy, leading/trailing stop words for noun chunks
    common_leading_trailing_stopwords = {"a", "an", "the"}

    # Only attempt noun chunk augmentation if initial result is short (e.g., 1 or 2 words)
    # and if the original text processed by spaCy (text_for_spacy_pipeline) wasn't extremely short itself.
    # This avoids unnecessary processing or over-augmentation of already specific short inputs.
    if len(initial_derived_text.split()) <= 2 and len(
        text_for_spacy_pipeline.split()
    ) > len(initial_derived_text.split()):
        candidate_chunks = []
        # Ensure doc (the spaCy processed document) is available
        if "doc" in locals() and doc is not None and doc.has_annotation("PARSER"):
            for chunk in doc.noun_chunks:
                # Check if the initial result is a substring of the chunk (case-insensitive)
                if initial_derived_text.lower() in chunk.text.lower():
                    # Lightly clean the chunk:
                    # 1. Strip common leading/trailing punctuation and digits
                    cleaned_chunk_text = chunk.text.strip(
                        " .,;:!?-()[]{}/\\\\\\\"'`0123456789"
                    )  # Escaped backslash and quote

                    # 2. Conservatively strip common leading/trailing stopwords
                    temp_chunk_words = cleaned_chunk_text.split()
                    if (
                        len(temp_chunk_words) > 1
                    ):  # Only if more than one word after initial strip
                        # Leading
                        if (
                            temp_chunk_words[0].lower()
                            in common_leading_trailing_stopwords
                        ):
                            temp_chunk_words = temp_chunk_words[1:]
                        # Trailing (check again if list is not empty)
                        if (
                            temp_chunk_words
                            and temp_chunk_words[-1].lower()
                            in common_leading_trailing_stopwords
                        ):
                            temp_chunk_words = temp_chunk_words[:-1]
                        cleaned_chunk_text = " ".join(temp_chunk_words)

                    # Further strip any remaining leading/trailing punctuation that might have been exposed
                    cleaned_chunk_text = cleaned_chunk_text.strip(
                        " .,;:!?-()[]{}/\\\\\\\"'`"
                    )  # Escaped backslash and quote

                    if cleaned_chunk_text:  # Ensure chunk is not empty after cleaning
                        candidate_chunks.append(cleaned_chunk_text)
        else:
            logger.debug(
                "Skipping noun chunk augmentation as 'doc' is not available or lacks parser annotation."
            )

        if candidate_chunks:
            # Prefer the longest valid candidate chunk
            best_chunk = max(candidate_chunks, key=len)

            # Use the chunk if it's meaningfully longer or provides more context
            # than the initial POS/Matcher derived text.
            # Avoid replacing if the chunk is just the same as initial text or shorter.
            if (
                len(best_chunk) > len(initial_derived_text)
                and best_chunk.lower() != initial_derived_text.lower()
            ):
                # Ensure we are actually augmenting, not just picking a random longer chunk.
                # Check if the best_chunk largely contains the initial_derived_text (case-insensitive),
                # OR if initial_derived_text is just a common word (e.g. "services", "on").
                # This helps confirm the chunk is an expansion of the initial, often too short, merchant name.
                if initial_derived_text.lower() in best_chunk.lower() or (
                    len(initial_derived_text.split()) == 1
                    and initial_derived_text.lower() not in company_suffixes
                    and not any(char.isdigit() for char in initial_derived_text)
                ):
                    logger.debug(
                        f"Noun chunk augmentation: Replacing '{initial_derived_text}' with '{best_chunk}'"
                    )
                    final_tokens_text = best_chunk  # Update with the augmented chunk
                else:
                    logger.debug(
                        f"Noun chunk '{best_chunk}' considered but not used as it doesn't clearly augment '{initial_derived_text}' in a related way."
                    )
            else:
                logger.debug(
                    f"Longest noun chunk '{best_chunk}' not chosen as it's not substantially better than initial '{initial_derived_text}'."
                )
        else:
            logger.debug(
                f"No suitable noun chunks found to augment '{initial_derived_text}'."
            )
    else:
        logger.debug(
            f"Skipping noun chunk augmentation for initial result '{initial_derived_text}' (either not short or input was already short)."
        )
    # --- End Noun Chunk Augmentation ---

    # --- Step 5.5: Remove Company Suffixes (NEW) ---
    if config.perform_company_suffix_removal and final_tokens_text:
        logger.debug(f"Attempting company suffix removal on: '{final_tokens_text}'")
        words = final_tokens_text.split()
        if words:
            # Check last word
            if words[-1].lower() in company_suffixes:
                logger.debug(
                    f"Removing company suffix '{words[-1]}' from '{final_tokens_text}'"
                )
                final_tokens_text = " ".join(words[:-1]).strip()
            # Check last two words if they form a known suffix (e.g. "pty ltd") - less common for current list but good practice
            elif (
                len(words) > 1
                and (words[-2].lower() + " " + words[-1].lower()) in company_suffixes
            ):
                logger.debug(
                    f"Removing company suffix '{words[-2]} {words[-1]}' from '{final_tokens_text}'"
                )
                final_tokens_text = " ".join(words[:-2]).strip()

    # --- Step 6: Final Cleanup on Selected Span ---
    cleaned_span = final_tokens_text

    # 6a. Remove standalone 3 or 4 digit numbers (potential store/branch codes)
    if config.perform_aggressive_store_number_removal:
        logger.debug(f"Attempting aggressive store number removal on: '{cleaned_span}'")
        store_number_patterns = [
            r"(?:(?<=^)|(?<=\s))#?\d{3,5}(?=\s|$)",  # Standalone 3-5 digit numbers
            r"(?:store|branch|outlet)\s*#?\d+",  # "store 123" patterns
            r"\b\d+[a-z]\b",  # "123a" style identifiers
            r"[-\s]\d{3,4}$",  # Trailing store numbers
        ]
        for pattern in store_number_patterns:
            cleaned_span = re.sub(
                pattern, "", cleaned_span, flags=re.IGNORECASE
            )  # Added IGNORECASE for robustness
            logger.debug(
                f"Step 6a - After applying store number pattern '{pattern}': '{cleaned_span}'"
            )

    # 6b. Remove leading/trailing hyphens and cleanup hyphens that might have become standalone
    # Strip leading/trailing whitespace and hyphens iteratively.
    # Handles cases like " - ", "word -", "- word"
    temp_cleaned_span = cleaned_span.strip()
    while temp_cleaned_span.startswith("-") or temp_cleaned_span.endswith("-"):
        temp_cleaned_span = temp_cleaned_span.strip(
            "-"
        ).strip()  # Strip hyphens then whitespace
    cleaned_span = temp_cleaned_span
    logger.debug(f"Step 6b - After hyphen cleanup: '{cleaned_span}'")

    # Normalize whitespace again (collapse multiple spaces, strip again)
    final_cleaned_text = re.sub(r"\\s+", " ", cleaned_span).strip()
    logger.debug(
        f"Step 6c - After final whitespace normalization: '{final_cleaned_text}'"
    )
    # Original log for context, can be removed if new logs are sufficient
    logger.debug(
        f"Final extracted features (after step 6): '{final_cleaned_text}' from input to spaCy '{text_for_spacy_pipeline}' from original pre-cleaned '{text}'"
    )

    if not final_cleaned_text:
        logger.warning(
            f"Final cleaning resulted in empty string for original: '{text}'. Returning basic cleaned: '{text}'"
        )
        return text

    return final_cleaned_text


# --- NEW FUNCTION: Normalize Merchant Variations ---
def normalize_merchant_variants(text: str) -> str:
    """Normalize common merchant name variations."""
    if not text:  # Added a check for empty input
        return ""

    # Create a mapping of variations to canonical forms
    merchant_normalizations = {
        # Supermarkets
        r"woolworths?\s*[a-z]?": "woolworths",
        r"woolies?\s*[a-z]?": "woolworths",
        r"pak\s*n\s*save\s*[a-z]?": "pak n save",
        r"new\s*world\s*[a-z]{2}": "new world",
        # Common patterns
        r"^the\s+": "",  # Remove leading "the"
        r"\s+the$": "",  # Remove trailing "the"
        r"anz\s+s\d[a-z]\d+": "anz atm",  # ANZ ATM codes
        r"kiwi\s+fuels?": "kiwi fuel",
        r"four\s+square": "four square",
    }

    text_lower = text.lower()  # Operate on lowercase text as patterns are lowercase
    for pattern, replacement in merchant_normalizations.items():
        text_lower = re.sub(
            pattern, replacement, text_lower, flags=re.IGNORECASE
        )  # Added IGNORECASE for robustness

    return text_lower.strip()


# --- NEW FUNCTION: Group Similar Merchants (Fuzzy Matching) ---
def group_similar_merchants(
    cleaned_names: List[str], similarity_threshold: float = 0.85
) -> dict:
    """Group merchants with similar names using Levenshtein distance."""
    groups = {}  # Maps a name to its canonical name
    # Stores the canonical names found so far, mapping to True (or a count, or list of members)
    canonical_names_map = {}

    # Sort names to make matching more consistent, e.g., shorter names processed first
    # or just alphabetically. For now, using the order they come in.
    for name in cleaned_names:
        if not name:  # Skip empty names
            continue

        best_match_canonical = name  # Default to itself if no better match found
        highest_score = 1.0  # Score for matching itself

        found_match = False
        for canonical_key in canonical_names_map.keys():
            score = Levenshtein.ratio(name, canonical_key)
            if score >= similarity_threshold and score > highest_score:
                # This check (score > highest_score) is a bit redundant if highest_score starts at a value
                # that any valid match for another canonical name would beat.
                # Let's refine to: if it's a good enough match to an *existing* canonical name
                highest_score = score
                best_match_canonical = canonical_key
                found_match = (
                    True  # Mark that we found a match to an existing canonical
                )
            elif (
                score >= similarity_threshold and not found_match
            ):  # If it's a good match but not better than current best (itself initially)
                # This ensures that if a name is similar to multiple canonicals, it picks one.
                # The first one it meets the threshold for, or could be the one with highest score.
                # Let's stick to highest score for now.
                # The logic for best_match_canonical and highest_score needs to be clear.
                pass  # Covered by the above condition if we want the absolute best score.

        # Refined logic for finding the best match:
        current_best_match = name  # Assume it's its own canonical form
        current_highest_score = 0.0  # Start with 0, so any valid match is better

        is_new_canonical = True
        for existing_canonical in canonical_names_map.keys():
            score = Levenshtein.ratio(name, existing_canonical)
            if score >= similarity_threshold and score > current_highest_score:
                current_highest_score = score
                current_best_match = existing_canonical
                is_new_canonical = False  # It matched an existing one

        if is_new_canonical:
            # This name establishes a new canonical group
            groups[name] = name
            canonical_names_map[name] = True  # Add to our set of known canonicals
        else:
            # This name belongs to an existing canonical group
            groups[name] = current_best_match

    return groups


# --- END NEW FUNCTION ---

# --- NEW CONSTANT: Merchant Categories ---
MERCHANT_CATEGORIES = {
    "supermarkets": [
        "woolworths",
        "pak n save",
        "new world",
        "four square",
        "freshchoice",
    ],
    "fuel": ["z energy", "bp", "mobil", "caltex", "gull", "rd petroleum", "kiwi fuel"],
    "retail": ["kmart", "warehouse", "farmers", "cotton on", "noel leeming"],
    "health": ["chemist warehouse", "life pharmacy", "unichem"],
    "food": [
        "uber eats",
        "menulog",
        "pizza",
        "burger",
        "coffee",
        "cafe",
    ],  # Added common food terms
}
# --- END NEW CONSTANT ---


# --- NEW FUNCTION: Categorize Merchant ---
def categorize_merchant(cleaned_name: str) -> str:
    """Assign a category to a merchant based on keywords."""
    if not cleaned_name:  # Added a check for empty input
        return "other"

    cleaned_name_lower = cleaned_name.lower()  # Ensure case-insensitive matching
    for category, keywords in MERCHANT_CATEGORIES.items():
        for keyword in keywords:
            if keyword in cleaned_name_lower:
                return category
    return "other"


# --- END NEW FUNCTION ---

# --- NEW CONSTANT: Merchant Aliases ---
MERCHANT_ALIASES = {
    "woolworths": ["woolworths n", "woolworths o", "woolies"],
    "pak n save": ["pak n save p", "pak n save fuel"],
    "new world": ["new world mt", "new world bl", "new world ra"],
    "four square": ["four square"],  # Already mostly canonical, but good to have
    "warehouse": ["the warehous", "warehouse st"],
    "anz atm": ["anz s3d7741", "anz s3c7711", "anz s3a1490"],  # Specific ANZ ATM codes
}
# --- END NEW CONSTANT ---


# --- NEW FUNCTION: Resolve Merchant Alias ---
def resolve_merchant_alias(cleaned_name: str) -> str:
    """Resolve merchant name to its canonical form using aliases."""
    if not cleaned_name:  # Added a check for empty input
        return ""

    cleaned_name_lower = cleaned_name.lower()  # Ensure case-insensitive matching
    for canonical, aliases in MERCHANT_ALIASES.items():
        for alias in aliases:
            if alias in cleaned_name_lower:  # Check if the alias is a substring
                return canonical  # Return the canonical name
    return cleaned_name  # Return original cleaned_name if no alias found


# --- END NEW FUNCTION ---


# --- NEW FUNCTION: Enhanced Clean Text (orchestrator) ---
def enhanced_clean_text(text: str, config: CleaningConfig = None) -> str:
    """Enhanced cleaning with multiple strategies, driven by config."""
    if not config:  # Default config if none provided
        config = CleaningConfig()

    if not text or not isinstance(text, str):  # Ensure text is a non-empty string
        return ""

    logger.debug(f"Enhanced cleaning input: '{text}' with config: {config.to_dict()}")

    cleaned_text = text
    if config.remove_card_numbers:
        logger.debug("Applying card number removal rules...")
        # Step 1: Basic direct pattern cleaning (more aggressive card/specific prefixes)
        cleaned_text = re.sub(
            r"4835-\*{4}-\*{4}-\d{4}\s*[DI]f\b", "", cleaned_text, flags=re.IGNORECASE
        )
        logger.debug(f"After specific card (4835...) removal: '{cleaned_text}'")

        cleaned_text = re.sub(
            r"\banz\s+s\d[a-z]\d+\b", "anz atm", cleaned_text, flags=re.IGNORECASE
        )
        logger.debug(f"After ANZ ATM normalization: '{cleaned_text}'")

        card_pattern_1 = r"\b\d{4}-[*xX]{4}-[*xX]{4}-\d{4}\s*(?:[A-Za-z]{1,3}\b)?"
        cleaned_text = re.sub(card_pattern_1, " ", cleaned_text, flags=re.IGNORECASE)

        card_pattern_2 = r"\b(?:card\s*)?(?:\d{4}[- ]?[*xX]{4}[- ]?[*xX]{4}[- ]?\d{4}|\d{4}[- ]?\d{2}[*xX]{2}[- ]?[*xX]{4}[- ]?\d{4}|[*xX]{4}[- ]?[*xX]{4}[- ]?[*xX]{4}[- ]?\d{4})\s*(?:[A-Za-z]{1,3}\b)?"
        cleaned_text = re.sub(card_pattern_2, " ", cleaned_text, flags=re.IGNORECASE)

        card_pattern_3 = r"\b(?:card\s*)?xx\d+\b"
        cleaned_text = re.sub(card_pattern_3, " ", cleaned_text, flags=re.IGNORECASE)
        logger.debug(f"After generic card pattern removal: '{cleaned_text}'")

    # Basic pre-cleaning (always apply these for general hygiene before NLP)
    cleaned_text = re.sub(r"https?://\S+", "", cleaned_text)  # Remove URLs
    cleaned_text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", cleaned_text
    )  # Remove emails
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()  # Normalize spaces
    logger.debug(f"After basic pre-cleaning (URL, email, spaces): '{cleaned_text}'")

    if not cleaned_text:
        logger.debug("Text empty after initial cleaning. Returning empty.")
        return ""

    # Initialize with potentially already cleaned text if NLP is skipped
    nlp_extracted_text = cleaned_text
    if config.apply_nlp_features_extraction:
        logger.debug("Applying NLP feature extraction...")
        # Pass the config to extract_nlp_features as well
        nlp_extracted_text = extract_nlp_features(cleaned_text, config)
        logger.debug(f"After extract_nlp_features: '{nlp_extracted_text}'")

    normalized_text = nlp_extracted_text
    if config.apply_merchant_variants_normalization:
        logger.debug("Applying merchant variants normalization...")
        normalized_text = normalize_merchant_variants(nlp_extracted_text)
        logger.debug(f"After normalize_merchant_variants: '{normalized_text}'")

    alias_resolved_text = normalized_text
    if config.apply_alias_resolution:
        logger.debug("Applying alias resolution...")
        alias_resolved_text = resolve_merchant_alias(normalized_text)
        logger.debug(f"After resolve_merchant_alias: '{alias_resolved_text}'")

    # Step 5: Final cleanup (ensure lowercase, collapse multiple spaces, strip)
    final_text = re.sub(r"\s+", " ", alias_resolved_text).strip().lower()
    logger.debug(f"Final output of enhanced_clean_text: '{final_text}'")

    if not final_text:
        logger.warning(
            f"Enhanced cleaning resulted in empty string for original input '{text}'. Check steps."
        )
        # Fallback strategy: could return a placeholder or re-evaluate logic for such cases.
        # For now, returning empty if all steps lead to it.

    return final_text


# --- END NEW FUNCTION ---


def clean_text(
    texts: Union[str, List[str]], config: CleaningConfig = None
) -> Union[str, List[str]]:
    """Cleans a single text string or a list of text strings using the enhanced pipeline."""
    if not config:  # Default config if none provided
        config = CleaningConfig()

    if not nlp:
        logger.error(
            "spaCy model not loaded. Returning original text(s) without full cleaning."
        )
        # Fallback to a very basic regex clean if spaCy is unavailable for some reason
        if isinstance(texts, str):
            # Apply only the most basic regex steps from enhanced_clean_text if NLP can't run
            temp_text = re.sub(
                r"4835-\*{4}-\*{4}-\d{4}\s*[DI]f\b", "", texts, flags=re.IGNORECASE
            )
            temp_text = re.sub(
                r"\banz\s+s\d[a-z]\d+\b", "anz atm", temp_text, flags=re.IGNORECASE
            )
            temp_text = re.sub(r"\s+", " ", temp_text).strip().lower()
            return temp_text
        elif isinstance(texts, list):
            results = []
            for text_item in texts:
                if isinstance(text_item, str):
                    temp_text = re.sub(
                        r"4835-\*{4}-\*{4}-\d{4}\s*[DI]f\b",
                        "",
                        text_item,
                        flags=re.IGNORECASE,
                    )
                    temp_text = re.sub(
                        r"\banz\s+s\d[a-z]\d+\b",
                        "anz atm",
                        temp_text,
                        flags=re.IGNORECASE,
                    )
                    temp_text = re.sub(r"\s+", " ", temp_text).strip().lower()
                    results.append(temp_text)
                else:
                    results.append("")  # Or handle non-strings as appropriate
            return results
        return texts  # Should not happen if types are correct

    # --- Define the core cleaning logic for a single string using the new enhanced_clean_text --- (This replaces _clean_single_text)
    def _dispatch_cleaning(original_text: any) -> str:
        if not isinstance(original_text, str):
            logger.debug(
                f"Input to _dispatch_cleaning is not a string ({type(original_text)}): '{original_text}'. Will be processed as empty string."
            )
            return enhanced_clean_text("")  # Process as empty string

        return enhanced_clean_text(original_text, config)  # Pass config

    # --- Handle input type ---
    if isinstance(texts, str):
        # Input is a single string
        return _dispatch_cleaning(texts)
    elif isinstance(texts, list):
        # Input is a list of strings
        # Original code had a critical length mismatch check, which is good.
        # Let's ensure progress logging for large lists if possible, or ensure it's efficient.
        cleaned_texts = [_dispatch_cleaning(text) for text in texts]

        # This check is important if any step could fail and return None or unexpected types
        if len(cleaned_texts) != len(texts):
            logger.error(
                f"CRITICAL: Length mismatch after loop in clean_text! Input: {len(texts)}, Output: {len(cleaned_texts)}"
            )
            # Fallback: return original texts (or partially cleaned) to avoid crashing downstream
            # This requires careful thought on what `texts` contains at this point.
            # For now, assuming _dispatch_cleaning always returns a string.
            # A more robust fallback might be to return the original `texts` list.
            # Reverting to return original `texts` for safety in case of unexpected issues.
            return texts

        return cleaned_texts
    else:
        # Handle unexpected input type for `texts`
        logger.warning(
            f"Unexpected input type for clean_text: {type(texts)}. Returning as is."
        )
        return texts  # Or raise TypeError


# === Old Regex-based clean_text (for reference, can be removed later) ===
# def clean_text_regex(text: str) -> str:
# ...
