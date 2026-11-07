
import json
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageOps, ImageEnhance, ImageFilter

SOURCE = Path("data/images/ingredient_downloads.jsonl")
OUTPUT = Path("data/ocr/improved")
OUTPUT.mkdir(parents=True, exist_ok=True)

records = [
    json.loads(line)
    for line in SOURCE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

results = []

for row in records:
    if row.get("status") not in ("downloaded", "already_present"):
        continue

    barcode = row["barcode"]
    original_path = Path(row["local_path"])

    if not original_path.exists():
        print("Missing image:", original_path)
        continue

    # Improve image appearance
    with Image.open(original_path) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        original_size = image.size

        # Enlarge small images without excessive scaling
        factor = min(3.0, 1800 / max(image.size))

        if factor > 1:
            new_size = (
                round(image.width * factor),
                round(image.height * factor)
            )
            image = image.resize(
                new_size, Image.Resampling.LANCZOS
            )

        image = ImageOps.autocontrast(image)
        image = ImageEnhance.Contrast(image).enhance(1.5)
        image = image.filter(
            ImageFilter.UnsharpMask(
                radius=1.2,
                percent=125,
                threshold=3
            )
        )

        improved_path = OUTPUT / f"{barcode}_improved.png"
        image.save(improved_path)

    print(f"\n=== PRODUCT {barcode} ===")
    print("Original dimensions:", original_size)

    # Compare two different OCR segmentation methods
    for psm in (6, 11):
        start = time.perf_counter()

        process = subprocess.run(
            [
                "tesseract",
                str(improved_path),
                "stdout",
                "-l", "eng",
                "--psm", str(psm)
            ],
            capture_output=True,
            text=True,
            timeout=90
        )

        elapsed = round(time.perf_counter() - start, 3)

        text = process.stdout.strip()
        status = (
            "error" if process.returncode != 0
            else "success" if text
            else "empty_text"
        )

        text_path = OUTPUT / f"{barcode}_psm{psm}.txt"
        text_path.write_text(text, encoding="utf-8")

        record = {
            "barcode": barcode,
            "psm": psm,
            "original_image": str(original_path),
            "improved_image": str(improved_path),
            "original_width": original_size[0],
            "original_height": original_size[1],
            "ocr_text": text,
            "word_count": len(text.split()),
            "processing_seconds": elapsed,
            "status": status,
            "error": process.stderr.strip()
                if process.returncode != 0 else ""
        }
        results.append(record)

        print(f"PSM {psm}: {status}")
        print("Words:", record["word_count"])
        print("Time:", elapsed, "seconds")
        print("Preview:", text[:180].replace("\n", " "))

result_path = OUTPUT / "ocr_comparison.jsonl"

with result_path.open("w", encoding="utf-8") as file:
    for record in results:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

print("\n=== IMPROVED OCR SUMMARY ===")
print("OCR attempts:", len(results))
print("Products tested:", len(set(r["barcode"] for r in results)))
print("Comparison saved:", result_path)
print("Improved images saved:", OUTPUT)
