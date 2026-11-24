
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATA_PATH = Path("data/food.parquet")
BATCH_SIZE = 5000
LONG_TEXT_LIMIT = 1000


# --------------------------------------------------
# EXTRACT INGREDIENT TEXT
# --------------------------------------------------

def extract_ingredient_text(value):
    """Extract ingredient text from the structured ingredients field."""

    if value is None:
        return ""

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return extract_ingredient_text(parsed)
        except (json.JSONDecodeError, TypeError):
            return value.strip()

    if isinstance(value, list):
        parts = [extract_ingredient_text(item) for item in value]
        return " ".join(part for part in parts if part)

    if isinstance(value, dict):
        parts = []

        if value.get("text"):
            parts.append(str(value["text"]).strip())

        if value.get("ingredients"):
            nested_text = extract_ingredient_text(value["ingredients"])
            if nested_text:
                parts.append(nested_text)

        return " ".join(parts)

    return ""


# --------------------------------------------------
# ANALYZE FULL DATASET
# --------------------------------------------------

def main():

    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Make sure you are running this script from the project root folder.")
        return

    parquet_file = pq.ParquetFile(DATA_PATH)

    print("Starting full dataset analysis...")
    print(f"Dataset: {DATA_PATH}")
    print(f"Total rows in Parquet file: {parquet_file.metadata.num_rows:,}")
    print()

    english_products = 0
    eligible_products = 0
    invalid_nova = 0
    missing_ingredients = 0

    nova_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    long_text_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    text_lengths = []

    # Read the file in small batches to control memory usage.
    for batch_number, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=["ingredients", "nova_group", "lang"],
            use_threads=False,
        ),
        start=1,
    ):

        rows = batch.to_pylist()

        for row in rows:

            if row["lang"] != "en":
                continue

            english_products += 1

            # Check NOVA label.
            try:
                nova = int(row["nova_group"])
            except (TypeError, ValueError):
                invalid_nova += 1
                continue

            if nova not in (1, 2, 3, 4):
                invalid_nova += 1
                continue

            # Extract ingredient text.
            ingredient_text = extract_ingredient_text(row["ingredients"])

            if not ingredient_text.strip():
                missing_ingredients += 1
                continue

            # Count eligible products.
            eligible_products += 1
            nova_counts[nova] += 1

            text_length = len(ingredient_text)
            text_lengths.append(text_length)

            # Count long descriptions for each NOVA group.
            if text_length > LONG_TEXT_LIMIT:
                long_text_counts[nova] += 1

        if batch_number % 100 == 0:
            print(
                f"Processed approximately "
                f"{batch_number * BATCH_SIZE:,} rows..."
            )

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print("\n========== FULL DATASET ANALYSIS ==========")

    print(f"Total English products: {english_products:,}")
    print(f"Eligible English products: {eligible_products:,}")
    print(f"Invalid or missing NOVA labels: {invalid_nova:,}")
    print(f"Products with missing ingredient text: {missing_ingredients:,}")

    print("\n========== NOVA DISTRIBUTION ==========")

    for nova, count in nova_counts.items():
        percentage = (
            count / eligible_products * 100
            if eligible_products else 0
        )
        print(f"NOVA {nova}: {count:,} products ({percentage:.2f}%)")

    # --------------------------------------------------
    # LONG DESCRIPTIONS BY NOVA GROUP
    # --------------------------------------------------

    print("\n========== LONG DESCRIPTIONS BY NOVA ==========")

    total_long_descriptions = 0

    for nova, count in long_text_counts.items():

        percentage = (
            count / nova_counts[nova] * 100
            if nova_counts[nova] else 0
        )

        total_long_descriptions += count

        print(
            f"NOVA {nova}: {count:,} long descriptions "
            f"({percentage:.2f}% of that NOVA group)"
        )

    print(f"Total long descriptions: {total_long_descriptions:,}")

    # --------------------------------------------------
    # INGREDIENT TEXT LENGTH
    # --------------------------------------------------

    if text_lengths:

        lengths = np.array(text_lengths)

        print("\n========== INGREDIENT TEXT LENGTH ==========")
        print(f"Average characters: {lengths.mean():.2f}")
        print(f"Median characters: {np.median(lengths):.2f}")
        print(f"Minimum characters: {lengths.min():,}")
        print(f"Maximum characters: {lengths.max():,}")
        print(f"25th percentile: {np.percentile(lengths, 25):.2f}")
        print(f"75th percentile: {np.percentile(lengths, 75):.2f}")

        short_texts = int(np.sum(lengths < 5))

        print(f"\nDescriptions shorter than 5 characters: {short_texts:,}")

        print(
            f"Descriptions longer than {LONG_TEXT_LIMIT} characters: "
            f"{total_long_descriptions:,}"
        )

        print(
            f"Percentage of long descriptions: "
            f"{total_long_descriptions / eligible_products * 100:.2f}%"
        )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
