
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "deep_learning" / "training_sample.csv"
)


def main():

    df = pd.read_csv(INPUT_FILE)

    df["normalized_ingredients"] = (
        df["ingredient_text"]
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Find repeated descriptions
    duplicate_rows = df[
        df.duplicated(
            subset=["normalized_ingredients"],
            keep=False
        )
    ]

    # Count distinct NOVA labels for each description
    label_counts = (
        duplicate_rows
        .groupby("normalized_ingredients")["nova_group"]
        .nunique()
    )

    conflicting = label_counts[label_counts > 1]

    print("\n========== DUPLICATE SUMMARY ==========")

    print("Total products:", len(df))
    print("Products with repeated descriptions:", len(duplicate_rows))
    print("Repeated unique descriptions:", len(label_counts))
    print("Descriptions with conflicting NOVA labels:", len(conflicting))

    print("\n========== FIRST 10 CONFLICTING EXAMPLES ==========")

    for description in conflicting.index[:10]:

        group = duplicate_rows[
            duplicate_rows["normalized_ingredients"] == description
        ]

        print("\nIngredients:", description[:150])
        print("NOVA labels:", sorted(group["nova_group"].unique().tolist()))
        print("Number of products:", len(group))

    print("\nInvestigation complete!")


if __name__ == "__main__":
    main()
