
from pathlib import Path
import pandas as pd

# 1. Locate the prepared dataset
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "deep_learning"
    / "training_sample.csv"
)


def main():

    print("Loading prepared dataset...")

    df = pd.read_csv(INPUT_FILE)

    print("\n========== BASIC INFORMATION ==========")

    print("Total rows:", len(df))
    print("Total columns:", len(df.columns))

    # 2. Check missing values
    print("\n========== MISSING VALUES ==========")

    print(df.isnull().sum())

    # 3. Check duplicate barcodes
    print("\n========== DUPLICATE BARCODES ==========")

    print("Duplicate barcodes:", df["code"].duplicated().sum())

    # 4. Check duplicate ingredient descriptions
    print("\n========== DUPLICATE INGREDIENT TEXT ==========")

    print(
        "Duplicate ingredient descriptions:",
        df["ingredient_text"].duplicated().sum()
    )

    # 5. Check description lengths
    df["ingredient_length"] = (
        df["ingredient_text"].str.strip().str.len()
    )

    print("\n========== INGREDIENT TEXT LENGTH ==========")

    print(df["ingredient_length"].describe())

    print(
        "\nDescriptions shorter than 5 characters:",
        (df["ingredient_length"] < 5).sum()
    )

    # 6. Check unique ingredient descriptions
    print("\n========== UNIQUE INGREDIENT DESCRIPTIONS ==========")

    print(
        "Unique descriptions:",
        df["ingredient_text"].nunique()
    )

    # 7. Check NOVA distribution
    print("\n========== NOVA DISTRIBUTION ==========")

    print(df["nova_group"].value_counts().sort_index())

    print("\nData quality inspection complete!")


if __name__ == "__main__":
    main()
