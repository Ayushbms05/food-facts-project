
import json
import re
from pathlib import Path

REFERENCE_DIR = Path("data/ocr/reference")
ORIGINAL = Path("data/ocr/ocr_results.jsonl")
IMPROVED = Path("data/ocr/improved/ocr_comparison.jsonl")
OUTPUT = Path("data/ocr/ocr_accuracy_batch.jsonl")


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def tokenize(text):
    return re.findall(
        r"[^\W_]+",
        text.lower(),
        flags=re.UNICODE
    )


def edit_distance(reference, predicted):
    previous = list(range(len(predicted) + 1))

    for i, ref_word in enumerate(reference, start=1):
        current = [i]

        for j, predicted_word in enumerate(predicted, start=1):
            cost = 0 if ref_word == predicted_word else 1

            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost
            ))

        previous = current

    return previous[-1]


original_rows = read_jsonl(ORIGINAL)
improved_rows = read_jsonl(IMPROVED)

results = []

print("\n=== BATCH OCR ACCURACY EVALUATION ===")

for reference_file in sorted(REFERENCE_DIR.glob("*.txt")):
    barcode = reference_file.stem

    reference_text = reference_file.read_text(
        encoding="utf-8"
    ).strip()

    reference_words = tokenize(reference_text)

    if not reference_words:
        print("Skipping empty reference:", barcode)
        continue

    samples = []

    for row in original_rows:
        if row.get("barcode") == barcode:
            samples.append((
                "Original PSM 6",
                row.get("ocr_text", "")
            ))

    for row in improved_rows:
        if row.get("barcode") == barcode:
            samples.append((
                f"Enhanced PSM {row['psm']}",
                row.get("ocr_text", "")
            ))

    if len(samples) != 3:
        print("Incomplete OCR methods for:", barcode)
        continue

    print(f"\nProduct: {barcode}")
    print("Reference words:", len(reference_words))

    for method, text in samples:
        predicted_words = tokenize(text)

        errors = edit_distance(
            reference_words,
            predicted_words
        )

        wer = 100 * errors / len(reference_words)

        results.append({
            "barcode": barcode,
            "method": method,
            "reference_words": len(reference_words),
            "predicted_words": len(predicted_words),
            "word_errors": errors,
            "wer_percent": round(wer, 2),
            "reference_type": "manual_image_transcription"
        })

        print(f"{method}: {wer:.2f}% WER")


with OUTPUT.open("w", encoding="utf-8") as file:
    for result in results:
        file.write(json.dumps(result) + "\n")

print("\n=== FINAL SUMMARY ===")
print("Images evaluated:", len(set(r["barcode"] for r in results)))
print("OCR method evaluations:", len(results))
print("Saved to:", OUTPUT)
