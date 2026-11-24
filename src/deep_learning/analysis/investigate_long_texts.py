import json
import heapq
from pathlib import Path

import pyarrow.parquet as pq


DATA_PATH = Path("data/food.parquet")
BATCH_SIZE = 5000
LONG_TEXT_LIMIT = 1000
TOP_N = 20


def extract_ingredient_text(value):
    """Extract text from the structured ingredients field."""

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
            nested = extract_ingredient_text(value["ingredients"])
            if nested:
                parts.append(nested)

        return " ".join(parts)

    return ""


def main():

    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    parquet_file = pq.ParquetFile(DATA_PATH)

    # Keep only the 20 longest descriptions found.
    longest_records = []
    record_counter = 0

    print("Scanning the dataset for long ingredient descriptions...")

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["code", "ingredients", "nova_group", "lang"],
        use_threads=False,
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

            record = {
                "length": len(text),
                "code": row["code"],
                "nova": nova,
                "text": text,
            }

            record_counter += 1

            heap_item = (
                record["length"],
                record_counter,
                record,
            )

            if len(longest_records) < TOP_N:
                heapq.heappush(longest_records, heap_item)

            elif record["length"] > longest_records[0][0]:
                heapq.heapreplace(longest_records, heap_item)

    print(f"\nTop {len(longest_records)} longest descriptions:")
    print("=" * 80)

    for rank, (_, _, record) in enumerate(
        sorted(longest_records, reverse=True),
        start=1,
    ):
        print(f"\nRANK: {rank}")
        print(f"Length: {record['length']} characters")
        print(f"Product code: {record['code']}")
        print(f"NOVA group: {record['nova']}")
        print("Ingredient text (first 700 characters):")
        print(record["text"][:700])
        print("-" * 80)


if __name__ == "__main__":
    main()