import csv
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATA_PATH = Path("data/food.parquet")
OUTPUT_PATH = Path("data/deep_learning/long_text_review.csv")

BATCH_SIZE = 5000
LONG_TEXT_LIMIT = 1000
SAMPLE_SIZE = 30
RANDOM_SEED = 42


# --------------------------------------------------
# EXTRACT INGREDIENT TEXT
# --------------------------------------------------

def extract_ingredient_text(value):
    """Extract ingredient text from the structured ingredients field."""

    if value is None:
        return ""

    if isinstance(value, str):
        try:
            return extract_ingredient_text(json.loads(value))
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
# MAIN ANALYSIS
# --------------------------------------------------

def main():

    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found: {DATA_PATH}")
        print("Run this script from the project root folder.")
        return

    parquet_file = pq.ParquetFile(DATA_PATH)

    rng = random.Random(RANDOM_SEED)
    selected_records = []
    long_description_count = 0

    print("Scanning for long ingredient descriptions...")
    print("Selecting a reproducible random sample of 30 products.")

    for batch_number, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=["code", "ingredients", "nova_group", "lang"],
            use_threads=False,
        ),
        start=1,
    ):

        for row in batch.to_pylist():

            if row["lang"] != "en":
                continue

            try:
                nova = int(row["nova_group"])
            except (TypeError, ValueError):
                continue

            if nova not in (1, 2, 3, 4):
                continue

            text = extract_ingredient_text(row["ingredients"])

            if len(text) <= LONG_TEXT_LIMIT:
                continue

            long_description_count += 1

            record = {
                "code": row["code"],
                "nova_group": nova,
                "text_length": len(text),
                "ingredient_text": text,
            }

            # Reservoir sampling: retain a random sample
            # without storing all 9,330+ long descriptions.
            if len(selected_records) < SAMPLE_SIZE:
                selected_records.append(record)
            else:
                position = rng.randrange(long_description_count)

                if position < SAMPLE_SIZE:
                    selected_records[position] = record

        if batch_number % 100 == 0:
            print(
                f"Processed approximately "
                f"{batch_number * BATCH_SIZE:,} rows..."
            )

    # --------------------------------------------------
    # SAVE REVIEW SAMPLE
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    selected_records.sort(
        key=lambda record: record["text_length"],
        reverse=True,
    )

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "code",
                "nova_group",
                "text_length",
                "ingredient_text",
            ],
        )

        writer.writeheader()
        writer.writerows(selected_records)

    print("\n========== REVIEW SAMPLE SUMMARY ==========")
    print(f"Total long descriptions found: {long_description_count:,}")
    print(f"Random descriptions selected: {len(selected_records)}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nSample NOVA distribution:")

    for nova in (1, 2, 3, 4):
        count = sum(
            record["nova_group"] == nova
            for record in selected_records
        )
        print(f"NOVA {nova}: {count}")

    print("\nYou can open long_text_review.csv in VS Code or Excel.")
    print("Review the ingredient_text column to identify extra information.")


if __name__ == "__main__":
    main()