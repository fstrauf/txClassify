"""Utility functions for cleaning and processing text data using spaCy NER."""

import re
import logging
import spacy
import os
import csv  # Added for reading CSV
import ahocorasick  # Added for Aho-Corasick
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Token, Doc
from typing import List, Union
from config import USE_NZ_BUSINESS_DATA_MATCHING  # Added import

logger = logging.getLogger(__name__)

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
def extract_nlp_features(text: str) -> str:
    """Extracts the most likely merchant name using spaCy NER, POS, Matchers, and heuristics."""
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

    # --- Step 6: Final Cleanup on Selected Span ---
    cleaned_span = final_tokens_text

    # 6a. Remove standalone 3 or 4 digit numbers (potential store/branch codes)
    # This regex looks for 3-4 digits that are preceded by start-of-string or a space,
    # and followed by a space or end-of-string.
    # Example: "MY STORE 1234" -> "MY STORE ", "1234 MY STORE" -> " MY STORE"
    # The number itself is removed. Extra spaces are handled by later normalization.
    number_removal_regex = r"(?:(?<=^)|(?<=\s))\d{3,4}(?=\s|$)"
    cleaned_span = re.sub(number_removal_regex, "", cleaned_span)
    logger.debug(
        f"Step 6a - After store number removal (regex: {number_removal_regex}): '{cleaned_span}'"
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


def clean_text(texts: Union[str, List[str]]) -> Union[str, List[str]]:
    """Cleans a single text string or a list of text strings using spaCy."""
    if not nlp:
        logger.error("spaCy model not loaded. Returning original text(s).")
        # Attempt to load NZ business data even if spaCy fails, for regex-only or future use
        if (
            USE_NZ_BUSINESS_DATA_MATCHING
            and NZ_BUSINESS_AUTOMATON.kind == ahocorasick.EMPTY
        ):
            load_nz_business_data()
        return texts

    # --- Define the core cleaning logic for a single string ---
    def _clean_single_text(
        original_text: any,
    ) -> str:  # Changed type hint for flexibility
        if not isinstance(original_text, str):
            logger.debug(
                f"Input to _clean_single_text is not a string ({type(original_text)}): '{original_text}'. Will be processed as empty string."
            )
            original_text = (
                ""  # Convert non-string to empty string for robust processing
            )

        # Basic pre-cleaning (still useful before NLP feature extraction)
        pre_cleaned_text = re.sub(r"https?://\\S+", "", original_text)
        pre_cleaned_text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", pre_cleaned_text
        )
        # Remove masked card numbers like xx9466 or Card xx0033
        pre_cleaned_text = re.sub(
            r"(card\s*)?xx\d+", "", pre_cleaned_text, flags=re.IGNORECASE
        ).strip()
        pre_cleaned_text = re.sub(r"\s+", " ", pre_cleaned_text).strip()

        if not pre_cleaned_text:
            return ""  # Return empty if basic cleaning removed everything

        extracted_feature = extract_nlp_features(pre_cleaned_text)

        # Handle edge case where cleaning results in empty string
        if not extracted_feature:
            logger.warning(
                f"Cleaning resulted in empty string for original: '{original_text}'"
            )
            # Fallback strategy:
            extracted_feature = "[empty_after_cleaning]"  # Or consider original_text?

        return extracted_feature.lower()  # Return lowercase cleaned text

    # --- Handle input type ---
    if isinstance(texts, str):
        # Input is a single string
        return _clean_single_text(texts)
    elif isinstance(texts, list):
        # Input is a list of strings
        cleaned_texts = [_clean_single_text(text) for text in texts]

        if len(cleaned_texts) != len(texts):
            # This should ideally not happen if the loop logic is correct
            logger.error(
                f"CRITICAL: Length mismatch after loop in clean_text! Input: {len(texts)}, Output: {len(cleaned_texts)}"
            )
            # Fallback: return original texts to avoid crashing downstream
            return texts

        return cleaned_texts


# === Old Regex-based clean_text (for reference, can be removed later) ===
# def clean_text_regex(text: str) -> str:
#     ...
