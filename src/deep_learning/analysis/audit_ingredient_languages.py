
from pathlib import Path

import pandas as pd

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException


# Make language detection reproducible.
DetectorFactory.seed = 42


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "deep_learning" / "reports"

INPUT_FILE = DATA_DIR / "natural_distribution_predictions.csv"

OUTPUT_FILE = DATA_DIR / "ingredient_language_audit.csv"


# --------------------------------------------------
# 2. Language detection
# --------------------------------------------------

def detect_language(text):

    text = str(text).strip()

    if len(text) < 20:
        return "too_short"

    try:
        return detect(text)

    except LangDetectException:
        return "unknown"


# --------------------------------------------------
# 3. Main audit
# --------------------------------------------------

def main():

    print("Loading prediction dataset...")

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("Dataset is empty.")

    print("Products loaded:", len(df))

    print("\nDetecting ingredient text languages...")

    detected_languages = []

    for index, text in enumerate(df["ingredient_text"]):

        language = detect_language(text)

        detected_languages.append(language)

        if (index + 1) % 1000 == 0:
            print(
                f"Processed {index + 1:,}/{len(df):,} products"
            )

    df["detected_language"] = detected_languages

    # --------------------------------------------------
    # 4. Identify likely non-English descriptions
    # --------------------------------------------------

    df["likely_non_english"] = (
        (df["detected_language"] != "en")
        & (df["detected_language"] != "too_short")
        & (df["detected_language"] != "unknown")
    )

    # --------------------------------------------------
    # 5. Overall language distribution
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("INGREDIENT LANGUAGE DISTRIBUTION")
    print("=" * 60)

    language_counts = (
        df["detected_language"]
        .value_counts()
        .rename_axis("language")
        .reset_index(name="products")
    )

    language_counts["percentage"] = (
        language_counts["products"] / len(df) * 100
    )

    print(language_counts.to_string(index=False))

    # --------------------------------------------------
    # 6. Likely non-English statistics
    # --------------------------------------------------

    non_english_count = df["likely_non_english"].sum()

    print("\n" + "=" * 60)
    print("LIKELY NON-ENGLISH INGREDIENT DESCRIPTIONS")
    print("=" * 60)

    print("Likely non-English:", non_english_count)

    print(
        "Percentage:",
        f"{non_english_count / len(df) * 100:.2f}%"
    )

    # --------------------------------------------------
    # 7. Language by actual NOVA group
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("LIKELY NON-ENGLISH BY ACTUAL NOVA GROUP")
    print("=" * 60)

    print(
        pd.crosstab(
            df["nova_group"],
            df["likely_non_english"],
            margins=True,
        )
    )

    # --------------------------------------------------
    # 8. Language and prediction errors
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("LANGUAGE VS PREDICTION ACCURACY")
    print("=" * 60)

    language_performance = (
        df.groupby("detected_language")
        .agg(
            products=("code", "count"),
            accuracy=("correct", "mean"),
        )
        .sort_values("products", ascending=False)
    )

    language_performance["accuracy"] *= 100

    print(
        language_performance.round(2).to_string()
    )

    # --------------------------------------------------
    # 9. Show examples of likely non-English text
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("EXAMPLES OF LIKELY NON-ENGLISH TEXT")
    print("=" * 60)

    examples = df[df["likely_non_english"]].head(20)

    for _, row in examples.iterrows():

        print(f"\nProduct: {row['code']}")
        print(f"Detected language: {row['detected_language']}")
        print(f"Actual NOVA: {row['nova_group']}")
        print(f"Predicted NOVA: {row['predicted_nova']}")
        print(f"Correct: {row['correct']}")
        print(f"Ingredients: {row['ingredient_text'][:350]}")

    # --------------------------------------------------
    # 10. Save audit results
    # --------------------------------------------------

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nAudit saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
