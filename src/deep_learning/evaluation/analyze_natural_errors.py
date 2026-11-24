
import re
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "deep_learning" / "reports"

INPUT_FILE = DATA_DIR / "natural_distribution_predictions.csv"


# --------------------------------------------------
# 2. Normalize ingredient descriptions
# --------------------------------------------------

def normalize_text(text):

    text = str(text).lower().strip()

    text = re.sub(r"\s+", " ", text)

    return text


# --------------------------------------------------
# 3. Main analysis
# --------------------------------------------------

def main():

    print("Loading natural-distribution predictions...")

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("Prediction file is empty.")

    df["normalized_text"] = (
        df["ingredient_text"].apply(normalize_text)
    )

    print("\nTotal products:", len(df))

    # --------------------------------------------------
    # 4. Find incorrect predictions
    # --------------------------------------------------

    errors = df[df["correct"] == False].copy()

    print("\nTotal incorrect predictions:", len(errors))

    # NOVA 2 false positives
    nova2_false_positives = errors[
        (errors["predicted_nova"] == 2)
        & (errors["nova_group"] != 2)
    ]

    print(
        "\nProducts incorrectly predicted as NOVA 2:",
        len(nova2_false_positives),
    )

    print("\nActual labels of these products:")

    print(
        nova2_false_positives["nova_group"]
        .value_counts()
        .sort_index()
    )

    # NOVA 4 predicted as NOVA 3
    nova4_as_nova3 = errors[
        (errors["nova_group"] == 4)
        & (errors["predicted_nova"] == 3)
    ]

    print(
        "\nActual NOVA 4 products predicted as NOVA 3:",
        len(nova4_as_nova3),
    )

    # --------------------------------------------------
    # 5. Display example errors
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("EXAMPLES: FALSE NOVA 2 PREDICTIONS")
    print("=" * 60)

    for _, row in nova2_false_positives.head(15).iterrows():

        print(f"\nProduct code: {row['code']}")
        print(f"Actual NOVA: {row['nova_group']}")
        print(f"Predicted NOVA: {row['predicted_nova']}")
        print(f"Ingredients: {row['ingredient_text'][:500]}")

    print("\n" + "=" * 60)
    print("EXAMPLES: NOVA 4 PREDICTED AS NOVA 3")
    print("=" * 60)

    for _, row in nova4_as_nova3.head(15).iterrows():

        print(f"\nProduct code: {row['code']}")
        print(f"Ingredients: {row['ingredient_text'][:500]}")

    # --------------------------------------------------
    # 6. Check repeated descriptions
    # --------------------------------------------------

    repeated = df[
        df.duplicated(
            subset=["normalized_text"],
            keep=False,
        )
    ].copy()

    print("\n" + "=" * 60)
    print("DUPLICATE DESCRIPTION ANALYSIS")
    print("=" * 60)

    print("Products with repeated descriptions:", len(repeated))

    print(
        "Unique repeated descriptions:",
        repeated["normalized_text"].nunique(),
    )

    conflicting = (
        repeated.groupby("normalized_text")["nova_group"]
        .nunique()
    )

    conflicting = conflicting[conflicting > 1]

    print(
        "Repeated descriptions with conflicting NOVA labels:",
        len(conflicting),
    )

    # --------------------------------------------------
    # 7. Save error report
    # --------------------------------------------------

    error_file = DATA_DIR / "natural_distribution_errors.csv"

    errors.drop(
        columns=["normalized_text"],
    ).to_csv(error_file, index=False)

    print("\nError report saved to:")
    print(error_file)


if __name__ == "__main__":
    main()
