import pandas as pd
import sys
import os
from pathlib import Path

# Add the workspace root and pythonHandler to the Python path
workspace_root = Path(__file__).parent.parent
python_handler_path = workspace_root / "pythonHandler"
sys.path.extend([str(workspace_root), str(python_handler_path)])

# Import from the correct location after refactoring
from pythonHandler.utils.text_utils import clean_text


def analyze_clean_text_effectiveness():
    """
    Analyze how the clean_text function reduces transaction counts and identify duplicate patterns.
    """
    # Read the training data
    data_file = os.path.join(workspace_root, "tests", "test_data", "full_train.csv")
    df = pd.read_csv(data_file)

    # Get initial counts
    total_original = len(df)
    unique_original = len(df["Description"].unique())

    print(f"\nOriginal Data Analysis:")
    print(f"Total transactions: {total_original}")
    print(f"Unique descriptions: {unique_original}")
    print(f"Duplicate ratio: {(total_original - unique_original) / total_original:.2%}")

    # Apply clean_text and analyze results
    df["cleaned_description"] = df["Description"].apply(clean_text)
    unique_cleaned = len(df["cleaned_description"].unique())

    print(f"\nAfter Cleaning Analysis:")
    print(f"Total transactions: {total_original}")
    print(f"Unique cleaned descriptions: {unique_cleaned}")
    print(
        f"Reduction ratio: {(unique_original - unique_cleaned) / unique_original:.2%}"
    )
    print(f"Duplicate ratio: {(total_original - unique_cleaned) / total_original:.2%}")

    # Analyze most common patterns
    print("\nTop 10 most frequent cleaned descriptions:")
    frequency = df["cleaned_description"].value_counts().head(10)
    for desc, count in frequency.items():
        print(f"Count: {count:3d} | {desc}")

        # Show original variations for this cleaned description
        originals = df[df["cleaned_description"] == desc]["Description"].unique()
        print("Original variations:")
        for orig in originals[:3]:  # Show max 3 variations
            print(f"  - {orig}")
        if len(originals) > 3:
            print(f"  ... and {len(originals)-3} more variations")
        print()

    # Save analysis to CSV for further inspection
    analysis_df = (
        df.groupby("cleaned_description")
        .agg({"Description": ["count", lambda x: list(x.unique())]})
        .reset_index()
    )
    analysis_df.columns = [
        "cleaned_description",
        "occurrence_count",
        "original_variations",
    ]
    analysis_df = analysis_df.sort_values("occurrence_count", ascending=False)

    # Save analysis to test_data directory
    output_file = os.path.join(
        workspace_root, "tests", "test_data", "clean_text_analysis.csv"
    )
    analysis_df.to_csv(output_file, index=False)
    print(f"\nDetailed analysis saved to '{output_file}'")


if __name__ == "__main__":
    analyze_clean_text_effectiveness()
