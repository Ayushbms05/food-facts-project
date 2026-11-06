
import json
import random
import csv
from pathlib import Path

import pyarrow.parquet as pq


# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "food.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "deep_learning"
OUTPUT_FILE = OUTPUT_DIR / "training_sample.csv"

BATCH_SIZE = 5_000
SAMPLES_PER_CLASS = 2_500
RANDOM_SEED = 42


# --------------------------------------------------
# 2. Convert ingredient data into readable text
# --------------------------------------------------

def extract_ingredients(raw_ingredients):
    if raw_ingredients is None:
        return ""

    try:
        if isinstance(raw_ingredients, str):
            ingredients = json.loads(raw_ingredients)
        else:
            ingredients = raw_ingredients

        names = []

        def collect_names(items):
            if isinstance(items, dict):
                name = items.get("text")

                if name:
                    names.append(str(name).strip())

                nested = items.get("ingredients")
                if nested:
                    collect_names(nested)

            elif isinstance(items, list):
                for item in items:
                    collect_names(item)

        collect_names(ingredients)

        return ", ".join(name for name in names if name)

    except (TypeError, ValueError, AttributeError):
        return ""


# --------------------------------------------------
# 3. Process the dataset in small batches
# --------------------------------------------------

def main():
    print("Opening food dataset...")

    parquet_file = pq.ParquetFile(INPUT_FILE)

    print("Total products:", parquet_file.metadata.num_rows)
    print("Processing in batches of", BATCH_SIZE)

    # Separate sample buckets for NOVA groups 1, 2, 3 and 4
    samples = {1: [], 2: [], 3: [], 4: []}
    seen_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    rng = random.Random(RANDOM_SEED)

    batch_number = 0

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["code", "ingredients", "nova_group","lang"],
        use_threads=False
    ):
        batch_number += 1

        data = batch.to_pydict()

        for code, raw_ingredients, nova, lang in zip(
            data["code"],
            data["ingredients"],
            data["nova_group"],
            data["lang"]
        ):
            if lang != "en":
                continue

            if nova is None or raw_ingredients is None:
                continue

            try:
                label = int(nova)
            except (ValueError, TypeError):
                continue

            if label not in [1, 2, 3, 4]:
                continue

            ingredient_text = extract_ingredients(raw_ingredients)

            if not ingredient_text.strip():
                continue

            product = {
                "code": code,
                "ingredient_text": ingredient_text,
                "nova_group": label
            }

            # Reservoir sampling keeps a random sample
            # without storing every matching product.
            seen_counts[label] += 1
            bucket = samples[label]

            if len(bucket) < SAMPLES_PER_CLASS:
                bucket.append(product)
            else:
                position = rng.randrange(seen_counts[label])

                if position < SAMPLES_PER_CLASS:
                    bucket[position] = product

        if batch_number % 100 == 0:
            print(
                f"Processed {batch_number} batches | "
                f"Products scanned: {batch_number * BATCH_SIZE:,}"
            )

    # --------------------------------------------------
    # 4. Combine samples and save CSV
    # --------------------------------------------------

    final_samples = []

    for label in [1, 2, 3, 4]:
        final_samples.extend(samples[label])

    rng.shuffle(final_samples)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["code", "ingredient_text", "nova_group"]
        )

        writer.writeheader()
        writer.writerows(final_samples)

    # --------------------------------------------------
    # 5. Display results
    # --------------------------------------------------

    print("\nDataset preparation complete!")
    print("Total products scanned:", parquet_file.metadata.num_rows)
    print("Products with usable ingredients and labels:")

    for label in [1, 2, 3, 4]:
        print(
            f"NOVA {label}: {seen_counts[label]:,} usable | "
            f"{len(samples[label]):,} sampled"
        )

    print("\nFinal sample size:", len(final_samples))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
