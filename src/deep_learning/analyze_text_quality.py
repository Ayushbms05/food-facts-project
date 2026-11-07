
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "deep_learning"
    / "training_sample.csv"
)


def main():

    # 1. Load our original 10,000-product sample
    df = pd.read_csv(INPUT_FILE)

    text = df["ingredient_text"].fillna("").astype(str)

    # 2. Calculate text lengths and word counts
    df["character_count"] = text.str.len()
    df["word_count"] = text.str.split().str.len()

    print("\n========== TEXT LENGTH ANALYSIS ==========")

    print("\nCharacter count:")
    print(df["character_count"].describe().round(2))

    print("\nWord count:")
    print(df["word_count"].describe().round(2))

    # 3. Check unusual descriptions
    print("\n========== SHORT DESCRIPTIONS ==========")

    short_text = df[df["character_count"] < 5]

    print("Descriptions shorter than 5 characters:", len(short_text))
    print(short_text["ingredient_text"].head(10).to_string(index=False))

    # 4. Check very long descriptions
    print("\n========== LONG DESCRIPTIONS ==========")

    long_text = df[df["character_count"] > 1000]

    print("Descriptions longer than 1000 characters:", len(long_text))

    print(
        long_text[
            ["ingredient_text", "character_count"]
        ].head(5).to_string(index=False)
    )

    # 5. Check descriptions containing numbers
    print("\n========== NUMBER ANALYSIS ==========")

    contains_numbers = text.str.contains(r"\d", regex=True)

    print("Descriptions containing numbers:", contains_numbers.sum())

    # 6. Check descriptions containing punctuation
    print("\n========== PUNCTUATION ANALYSIS ==========")

    punctuation = text.str.contains(r"[^\w\s]", regex=True)

    print("Descriptions containing punctuation:", punctuation.sum())

    # 7. Check non-ASCII characters
    print("\n========== SPECIAL CHARACTER ANALYSIS ==========")

    non_ascii = text.apply(
        lambda value: any(ord(char) > 127 for char in value)
    )

    print("Descriptions containing non-ASCII characters:", non_ascii.sum())

    # 8. Check inconsistent whitespace
    print("\n========== WHITESPACE ANALYSIS ==========")

    extra_spaces = text.apply(
        lambda value: value != " ".join(value.split())
    )

    print("Descriptions with unusual whitespace:", extra_spaces.sum())

    # 9. Show examples for inspection
    print("\n========== SAMPLE INGREDIENT DESCRIPTIONS ==========")

    for description in text.sample(
        n=10,
        random_state=42
    ):
        print("-", description)

    print("\nText quality analysis complete!")


if __name__ == "__main__":
    main()
