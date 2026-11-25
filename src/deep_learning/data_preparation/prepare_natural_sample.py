
import random
import re
import csv
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.deep_learning.data_preparation.prepare_dataset import extract_ingredients


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = PROJECT_ROOT / "data" / "food.parquet"
DATA_DIR = PROJECT_ROOT / "data" / "deep_learning" / "datasets"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

OUTPUT_FILE = DATA_DIR / "natural_distribution_sample.csv"

BATCH_SIZE = 5_000
SAMPLE_SIZE = 10_000
RANDOM_SEED = 42


# --------------------------------------------------
# 2. Normalize ingredient text
# --------------------------------------------------

def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# --------------------------------------------------
# 3. Load descriptions already used in our splits
# --------------------------------------------------

def load_existing_texts():

    existing_texts = set()

    split_files = [
        TRAIN_PATH,
        VALIDATION_PATH,
        TEST_PATH,
    ]

    for file_path in split_files:

        print(f"Loading existing split: {file_path.name}")

        df = pd.read_csv(file_path)

        for text in df["ingredient_text"].dropna():
            existing_texts.add(normalize_text(text))

    print(
        "\nUnique descriptions to exclude:",
        f"{len(existing_texts):,}"
    )

    return existing_texts


# --------------------------------------------------
# 4. Create a natural-distribution sample
# --------------------------------------------------

def main():

    print("Loading existing training, validation and test texts...")

    existing_texts = load_existing_texts()

    print("\nOpening original Parquet dataset...")

    parquet_file = pq.ParquetFile(INPUT_FILE)

    print(
        "Total source products:",
        f"{parquet_file.metadata.num_rows:,}"
    )

    rng = random.Random(RANDOM_SEED)

    samples = []
    eligible_count = 0
    excluded_count = 0

    class_counts = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }

    batch_number = 0

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=[
            "code",
            "ingredients",
            "nova_group",
            "lang",
        ],
        use_threads=False,
    ):

        batch_number += 1
        data = batch.to_pydict()

        for code, raw_ingredients, nova, lang in zip(
            data["code"],
            data["ingredients"],
            data["nova_group"],
            data["lang"],
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

            normalized = normalize_text(ingredient_text)

            # Exclude descriptions used in existing splits.
            if normalized in existing_texts:
                excluded_count += 1
                continue

            eligible_count += 1

            class_counts[label] += 1

            product = {
                "code": code,
                "ingredient_text": ingredient_text,
                "nova_group": label,
            }

            # Reservoir sampling:
            # Every eligible product has an equal chance
            # of entering the final sample.
            if len(samples) < SAMPLE_SIZE:

                samples.append(product)

            else:

                position = rng.randrange(eligible_count)

                if position < SAMPLE_SIZE:
                    samples[position] = product

        if batch_number % 100 == 0:

            print(
                f"Processed {batch_number} batches | "
                f"Eligible products: {eligible_count:,}"
            )

    # --------------------------------------------------
    # 5. Save sample
    # --------------------------------------------------

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "code",
                "ingredient_text",
                "nova_group",
            ],
        )

        writer.writeheader()
        writer.writerows(samples)

    # --------------------------------------------------
    # 6. Display results
    # --------------------------------------------------

    sampled_counts = {
        label: sum(
            row["nova_group"] == label
            for row in samples
        )
        for label in [1, 2, 3, 4]
    }

    print("\n" + "=" * 55)
    print("NATURAL-DISTRIBUTION SAMPLE COMPLETE")
    print("=" * 55)

    print("Eligible products after exclusions:", eligible_count)
    print("Products excluded from existing splits:", excluded_count)
    print("Final sample size:", len(samples))

    print("\nClass distribution in source after exclusions:")

    for label in [1, 2, 3, 4]:

        percentage = (
            class_counts[label] / eligible_count * 100
            if eligible_count else 0
        )

        print(
            f"NOVA {label}: "
            f"{class_counts[label]:,} "
            f"({percentage:.2f}%)"
        )

    print("\nClass distribution in final sample:")

    for label in [1, 2, 3, 4]:

        percentage = (
            sampled_counts[label] / len(samples) * 100
            if samples else 0
        )

        print(
            f"NOVA {label}: "
            f"{sampled_counts[label]:,} "
            f"({percentage:.2f}%)"
        )

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
