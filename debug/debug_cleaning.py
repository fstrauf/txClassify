#!/usr/bin/env python3

import sys
from pathlib import Path

# Add pythonHandler to path
sys.path.append(str(Path(__file__).parent / "pythonHandler"))

from utils.text_utils import clean_transaction_text, CleaningConfig

def debug_cleaning():
    """Debug the cleaning process for problematic examples."""
    
    # Problematic examples from the log
    examples = [
        "4835-****-****-0311  Df A 2 Z",
        "4835-****-****-0311  Df Challenge Ta", 
        "4835-****-****-0311  Df One Nz"
    ]
    
    config = CleaningConfig()
    
    print("=== Debugging Aggressive Cleaning ===\n")
    
    for example in examples:
        print(f"Original: '{example}'")
        
        # Step by step cleaning to see what happens
        text = example
        
        # Step 1: Remove card numbers
        if config.remove_card_numbers:
            from utils.text_utils import remove_card_numbers
            text = remove_card_numbers(text)
            print(f"  After card removal: '{text}'")
        
        # Step 2: Remove reference codes  
        if config.remove_reference_codes:
            from utils.text_utils import remove_reference_codes
            text = remove_reference_codes(text)
            print(f"  After ref codes: '{text}'")
        
        # Step 3: Remove bank codes
        if config.remove_bank_codes:
            from utils.text_utils import remove_bank_codes
            text = remove_bank_codes(text)
            print(f"  After bank codes: '{text}'")
        
        # Step 4: Remove merchant codes
        if config.remove_merchant_codes:
            from utils.text_utils import remove_merchant_codes
            text = remove_merchant_codes(text)
            print(f"  After merchant codes: '{text}'")
        
        # Step 5: NLP extraction
        if config.use_nlp_extraction:
            from utils.text_utils import extract_merchant_name
            text = extract_merchant_name(text)
            print(f"  After NLP extraction: '{text}'")
        
        # Step 6: Normalize merchant name
        if config.normalize_merchant_names:
            from utils.text_utils import normalize_merchant_name
            text = normalize_merchant_name(text)
            print(f"  After normalization: '{text}'")
        
        # Step 7: Lowercase
        if config.to_lowercase:
            text = text.lower()
            print(f"  After lowercase: '{text}'")
        
        # Final result
        final = clean_transaction_text(example, config)
        print(f"  Final result: '{final}'")
        print()

if __name__ == "__main__":
    debug_cleaning()
