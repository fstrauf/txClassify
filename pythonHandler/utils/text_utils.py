"""Utility functions for cleaning and processing text data using spaCy NER."""

import re
import logging
import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Token, Doc
from typing import List, Union

logger = logging.getLogger(__name__)

# --- Define Noise Patterns for Matcher (More Generic) ---
# Expanded based on Pilot-NER rule-based repo
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
    # ".com", ".net" # Excluded as likely part of name/URL
}

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
    # Add specific noise words if consistently problematic and not caught otherwise
    [{"LOWER": "dee"}],  # For "DEE WHY"
    [{"LOWER": "ns"}],  # Common noise observed
    # [{"LOWER": "sydney"}], # Keep commented unless needed with lg model
    # [{"LOWER": "manly"}],
]


# --- Custom spaCy Component (Handles Transaction Noise) ---
if not Token.has_extension("is_transaction_noise"):
    Token.set_extension("is_transaction_noise", default=False)


@Language.component("transaction_noise_matcher")
def transaction_noise_matcher_component(doc: Doc) -> Doc:
    """
    Marks transaction-specific noise patterns, avoiding ORG overlaps.
    Sets token._.is_transaction_noise = True
    """
    matcher = Matcher(doc.vocab)
    matcher.add("TRANSACTION_NOISE", transaction_noise_patterns)
    matches = matcher(doc)

    for match_id, start, end in matches:
        # Remove the ORG check - mark noise regardless of NER label
        pattern_name = doc.vocab.strings[match_id]
        span = doc[start:end]
        logger.debug(f"Transaction Noise Matcher: '{span.text}' ({pattern_name})")
        for i in range(start, end):
            # Mark all tokens in the matched span as noise
            doc[i]._.is_transaction_noise = True
    return doc


# --- Load spaCy Model and Add Custom Component ---
nlp = None
try:
    # Load the base model
    nlp = spacy.load("en_core_web_lg")
    # Add the custom component after the 'ner' component
    if "ner" in nlp.pipe_names:
        nlp.add_pipe("transaction_noise_matcher", after="ner")
        logger.info(
            "spaCy model 'en_core_web_lg' loaded and transaction_noise_matcher added after ner."
        )
    else:
        # Add it last if ner is not present (shouldn't happen with en_core_web_lg)
        nlp.add_pipe("transaction_noise_matcher", last=True)
        logger.warning(
            "spaCy 'ner' component not found. Added transaction_noise_matcher last."
        )

    logger.info(f"Pipeline components: {nlp.pipe_names}")

except OSError as e:
    logger.error(f"spaCy model 'en_core_web_lg' not found or other OS error: {e}")
    logger.error("Please run: python -m spacy download en_core_web_lg")
    # Fallback: nlp remains None
except Exception as e:
    logger.error(f"Error adding custom component to spaCy pipeline: {e}")
    # Fallback: nlp remains None


# --- Feature Extraction and Cleaning Functions ---
def extract_nlp_features(text: str) -> str:
    """Extracts the most likely merchant name using spaCy NER, POS, Matchers, and heuristics."""
    if not nlp or not text:
        return text

    # --- Step 0: Check for Quoted Text (Rule-Based Insight) ---
    quoted_match = re.search(r"\"(.+?)\"", text)
    if quoted_match:
        extracted_name = quoted_match.group(1).strip()
        # Optional: Add minimal cleaning to quoted text if needed
        extracted_name = re.sub(r"\s+", " ", extracted_name).strip()
        if extracted_name:
            logger.debug(f"Prioritized quoted text: '{extracted_name}' from '{text}'")
            return extracted_name
        # Else, fall through if quote is empty or just whitespace

    # --- Process with spaCy Pipeline (NER + Transaction Noise Matcher) ---
    # --- Disable only \"lemmatizer\" as \"parser\" is needed for noun_chunks fallback ---
    doc = nlp(text, disable=["lemmatizer"])

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
        f"Final extracted features (after step 6): '{final_cleaned_text}' from original span '{final_tokens_text}' from doc '{text}'"
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
