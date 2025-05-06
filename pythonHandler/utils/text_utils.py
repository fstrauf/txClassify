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
        is_loc_match = i in location_noise_indices
        is_txn_noise = i in transaction_noise_indices  # Check the set
        has_keep_pos = token.pos_ in KEEP_POS

        is_noise = is_loc_match or is_txn_noise  # Combine noise flags

        # Keep token ONLY if it's NOT noise AND has a useful POS tag
        # We remove the direct ORG check here to allow noise removal within ORGs
        if not is_noise and has_keep_pos:
            initial_filtered_tokens.append(token)
        # Log removals for debugging (simplified logging)
        else:
            # Log removal reason
            if is_loc_match:
                logger.debug(f"Removing '{token.text}' (Location Matcher)")
            elif is_txn_noise:
                logger.debug(f"Removing '{token.text}' (Transaction Matcher)")
            elif not has_keep_pos:
                logger.debug(f"Removing '{token.text}' (POS: {token.pos_})")

    initial_filtered_text = " ".join([t.text for t in initial_filtered_tokens])
    logger.debug(f"Text after POS/Matcher filtering: '{initial_filtered_text}'")

    # --- Step 4: Contextual Post-Filtering (Numbers Only - applied to initial_filtered_tokens) ---
    final_tokens = []  # List of Token objects
    num_initial_tokens = len(initial_filtered_tokens)

    for idx, token in enumerate(initial_filtered_tokens):
        keep_token = True
        # Check for number removal: Remove if IS_DIGIT, length >= 5, and not adjacent to ORG in the *initial filtered* list
        if token.is_digit and len(token.text) >= 5:
            is_part_of_org = token.i in org_indices  # Check original ORG status
            prev_is_org = idx > 0 and initial_filtered_tokens[idx - 1].i in org_indices
            next_is_org = (
                idx < num_initial_tokens - 1
                and initial_filtered_tokens[idx + 1].i in org_indices
            )

            if not (is_part_of_org or prev_is_org or next_is_org):
                keep_token = False
                logger.debug(
                    f"Contextual removal: Removing digits '{token.text}' (len>=5, not near ORG)"
                )

        if keep_token:
            final_tokens.append(token)

    # --- Step 5: Final Assembly & Heuristics ---
    if not final_tokens:
        # Fallback logic 1: Use if filtering removed everything
        org_entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        if org_entities:
            extracted_name = " ".join(org_entities)
            logger.debug(
                f"Filtering removed all; falling back to ORG: '{extracted_name}'"
            )
        else:
            noun_chunks = [chunk.text for chunk in doc.noun_chunks]
            if noun_chunks:
                extracted_name = noun_chunks[0]
                logger.debug(
                    f"Filtering removed all; falling back to first Noun Chunk: '{extracted_name}'"
                )
            else:
                logger.debug(
                    f"Could not extract meaningful name from: '{text}'. Returning original (pre-regex)."
                )
                return text  # Return original text after initial regex pre-cleaning
    else:
        # Build name from final tokens
        final_tokens_text = [t.text for t in final_tokens]

        # Heuristic: Re-attach company suffix (apply before final length check)
        if len(final_tokens) >= 1 and len(initial_filtered_tokens) > len(final_tokens):
            last_kept_token_idx = final_tokens[-1].i
            if last_kept_token_idx + 1 < len(doc):
                next_token_original = doc[last_kept_token_idx + 1]
                if next_token_original.lower_ in company_suffixes:
                    is_next_token_kept = any(
                        kept_token.i == next_token_original.i
                        for kept_token in final_tokens
                    )
                    if not is_next_token_kept:
                        logger.debug(
                            f"Re-attaching company suffix: {next_token_original.text}"
                        )
                        final_tokens_text.append(next_token_original.text)

        extracted_name = " ".join(final_tokens_text)
        extracted_name = re.sub(r"\s+", " ", extracted_name).strip()

        # Heuristic: Check for very short results (<= 2 chars) AFTER suffix attachment
        # Change threshold to <= 2, improve fallback logic
        if (
            len(extracted_name) <= 2 and len(text) > 10
        ):  # Use original text length as context
            logger.warning(
                f"Result '{extracted_name}' seems too short after filtering '{text}'. Attempting fallback."
            )
            # Fallback logic 2: Try ORG -> Noun Chunk -> initial_filtered_text -> original
            org_entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
            if org_entities:
                extracted_name = " ".join(org_entities)
                logger.debug(f"Short result fallback to ORG: '{extracted_name}'")
            else:
                noun_chunks = [chunk.text for chunk in doc.noun_chunks]
                if noun_chunks:
                    extracted_name = noun_chunks[0]
                    logger.debug(
                        f"Short result fallback to Noun Chunk: '{extracted_name}'"
                    )
                else:
                    # Revert to before contextual filtering as last resort before original
                    extracted_name = re.sub(r"\s+", " ", initial_filtered_text).strip()
                    logger.debug(
                        f"Short result fallback to pre-contextual filter: '{extracted_name}'"
                    )
                    if not extracted_name:
                        logger.warning(
                            f"Short result fallback failed, returning original text: '{text}'"
                        )
                        return text  # Return original text pre-regex

    # Remove Step 6 (Final Regex Cleanup) as we rely on spaCy components now
    logger.debug(
        f"Final extracted features (before final cleanup): '{extracted_name}' from '{text}'"
    )

    # Apply minimal final cleanup (whitespace)
    final_name = re.sub(r"\s+", " ", extracted_name).strip()
    return final_name


def clean_text(texts: Union[str, List[str]]) -> Union[str, List[str]]:
    """Cleans a single text string or a list of text strings using spaCy."""
    if not nlp:
        logger.error("spaCy model not loaded. Returning original text(s).")
        return texts

    # --- Define the core cleaning logic for a single string ---
    def _clean_single_text(original_text: str) -> str:
        # Basic pre-cleaning (still useful before NLP feature extraction)
        pre_cleaned_text = re.sub(r"https?://\S+", "", original_text)
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
    else:
        # Input is neither string nor list, raise error or return as is
        logger.warning(
            f"clean_text received unexpected input type: {type(texts)}. Returning as is."
        )
        return texts


# === Old Regex-based clean_text (for reference, can be removed later) ===
# def clean_text_regex(text: str) -> str:
#     ...
