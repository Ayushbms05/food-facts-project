
import json
import re
from pathlib import Path

BARCODE = "8902979052405"

REFERENCE = Path(
    f"data/ocr/reference/{BARCODE}.txt"
)

ORIGINAL = Path(
    "data/ocr/ocr_results.jsonl"
)

IMPROVED = Path(
    "data/ocr/improved/ocr_comparison.jsonl"
)

OUTPUT = Path(
    "data/ocr/ocr_accuracy_results.jsonl"
)


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def words(text):
    # Use the same normalization for reference and OCR.
    # Keep letters and numbers; ignore punctuation and case.
    return re.findall(
        r"[^\W_]+",
        text.lower(),
        flags=re.UNICODE
    )


def edit_distance(reference, predicted):
    # Word-level Levenshtein distance
    previous = list(range(len(predicted) + 1))

    for i, ref_word in enumerate(reference, start=1):
        current = [i]

        for j, pred_word in enumerate(predicted, start=1):
            substitution = (
                0 if ref_word == pred_word else 1
            )

            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + substitution
            ))

        previous = current

    return previous[-1]


if not REFERENCE.exists():
    raise SystemExit(
        f"Create the manual reference first: {REFERENCE}"
    )

reference_text = REFERENCE.read_text(
    encoding="utf-8"
).strip()

reference_words = words(reference_text)

if not reference_words:
    raise SystemExit("Reference text is empty.")


original_rows = read_jsonl(ORIGINAL)
improved_rows = read_jsonl(IMPROVED)

samples = []

for row in original_rows:
    if row.get("barcode") == BARCODE:
        samples.append((
            "Original PSM 6",
            row.get("ocr_text", "")
        ))

for row in improved_rows:
    if row.get("barcode") == BARCODE:
        samples.append((
            f"Enhanced PSM {row['psm']}",
            row.get("ocr_text", "")
        ))

if len(samples) != 3:
    raise SystemExit(
        f"Expected 3 OCR results, found {len(samples)}."
    )

results = []

print("\n=== OCR ACCURACY EVALUATION ===")
print("Barcode:", BARCODE)
print("Reference words:", len(reference_words))

for method, ocr_text in samples:
    predicted_words = words(ocr_text)

    errors = edit_distance(
        reference_words,
        predicted_words
    )

    wer = errors / len(reference_words)

    results.append({
        "barcode": BARCODE,
        "method": method,
        "reference_words": len(reference_words),
        "ocr_words": len(predicted_words),
        "word_edit_distance": errors,
        "wer": round(wer, 4),
        "wer_percent": round(wer * 100, 2),
        "reference_type": "manual_image_transcription",
    })

    print(
        f"{method}: WER = {wer * 100:.2f}%"
    )

with OUTPUT.open("w", encoding="utf-8") as file:
    for row in results:
        file.write(json.dumps(row) + "\n")

print("\n=== COMPARISON ===")

for row in sorted(results, key=lambda x: x["wer"]):
    print(
        f"{row['method']}: "
        f"{row['wer_percent']}% WER"
    )

print("\nResults saved:", OUTPUT)
