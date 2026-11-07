
import json
import shutil
import subprocess
import time
from pathlib import Path
from datetime import datetime, timezone

SOURCE = Path("data/images/ingredient_downloads.jsonl")
OUTPUT = Path("data/ocr")
TEXT_DIR = OUTPUT / "text"
RESULTS = OUTPUT / "ocr_results.jsonl"

TEXT_DIR.mkdir(parents=True, exist_ok=True)

if not SOURCE.exists():
    raise SystemExit("Ingredient download log not found.")

if shutil.which("tesseract") is None:
    raise SystemExit("Install Tesseract first.")

version = subprocess.run(
    ["tesseract", "--version"],
    capture_output=True,
    text=True
).stdout.splitlines()[0]

with SOURCE.open("r", encoding="utf-8") as file:
    images = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

results = []

print("\n=== RUNNING OCR ON FOOD IMAGES ===")

for row in images:
    if row.get("status") not in (
        "downloaded", "already_present"
    ):
        continue

    barcode = row["barcode"]
    image_path = Path(row["local_path"])
    text_path = TEXT_DIR / f"{barcode}_ingredients.txt"

    status = "failed"
    extracted_text = ""
    error = ""

    start = time.perf_counter()

    try:
        if not image_path.exists():
            raise FileNotFoundError(str(image_path))

        process = subprocess.run(
            [
                "tesseract",
                str(image_path),
                "stdout",
                "-l", "eng",
                "--psm", "6"
            ],
            capture_output=True,
            text=True,
            timeout=90
        )

        if process.returncode != 0:
            raise RuntimeError(process.stderr.strip())

        extracted_text = process.stdout.strip()

        status = (
            "success" if extracted_text
            else "empty_text"
        )

        text_path.write_text(
            extracted_text + "\n",
            encoding="utf-8"
        )

    except Exception as exc:
        error = str(exc)

    elapsed = round(
        time.perf_counter() - start, 3
    )

    result = {
        "barcode": barcode,
        "image_path": str(image_path),
        "text_path": (
            str(text_path)
            if status in ("success", "empty_text")
            else None
        ),
        "ocr_text": extracted_text,
        "character_count": len(extracted_text),
        "word_count": len(extracted_text.split()),
        "processing_seconds": elapsed,
        "status": status,
        "error": error,
        "ocr_engine": "Tesseract",
        "ocr_version": version,
        "language": "eng",
        "page_segmentation_mode": 6,
        "image_source": "Open Food Facts",
        "processed_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    results.append(result)

    print(f"\nBarcode: {barcode}")
    print("Status:", status)
    print("Words detected:", result["word_count"])
    print("Processing time:", elapsed, "seconds")
    print(
        "Text preview:",
        extracted_text[:160].replace("\n", " ")
    )

    if error:
        print("Error:", error)

with RESULTS.open("w", encoding="utf-8") as file:
    for row in results:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")

print("\n=== OCR PILOT SUMMARY ===")
print("Images processed:", len(results))
print(
    "Successful OCR:",
    sum(r["status"] == "success" for r in results)
)
print(
    "Empty OCR:",
    sum(r["status"] == "empty_text" for r in results)
)
print(
    "Failed OCR:",
    sum(r["status"] == "failed" for r in results)
)
print(
    "Total words detected:",
    sum(r["word_count"] for r in results)
)
print(
    "Total processing seconds:",
    round(sum(r["processing_seconds"] for r in results), 3)
)
print("Results saved:", RESULTS)
