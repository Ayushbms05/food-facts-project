
import io
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image

SOURCE = Path("data/images/balanced_image_candidates.jsonl")
OUTPUT = Path("data/images/balanced_full")
LOG = Path("data/images/balanced_download_log.jsonl")

PER_GROUP = 2
MAX_BYTES = 16_000_000

contact = os.environ.get("OFF_CONTACT_EMAIL")

if not contact:
    raise SystemExit("Set OFF_CONTACT_EMAIL first.")

user_agent = f"FoodFactsStudentProject/0.1 ({contact})"

OUTPUT.mkdir(parents=True, exist_ok=True)

with SOURCE.open(encoding="utf-8") as file:
    candidates = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

# Select equal numbers from each NOVA group
selected = []

for group in (1, 2, 3, 4):
    available = [
        row for row in candidates
        if row["nova_group"] == group
        and row["status"] == "available"
        and row.get("image_url")
    ]

    selected.extend(available[:PER_GROUP])

print("\n=== BALANCED IMAGE DOWNLOAD ===")
print("Images selected:", len(selected))

results = []

for row in selected:
    barcode = str(row["barcode"])
    group = row["nova_group"]
    original_url = row["image_url"]

    parsed = urlparse(original_url)

    record = {
        "barcode": barcode,
        "nova_group": group,
        "image_language": row.get("image_language"),
        "original_url": original_url,
        "download_url": "",
        "local_path": "",
        "width": None,
        "height": None,
        "size_bytes": 0,
        "status": "failed",
        "error": "",
        "source": "Open Food Facts"
    }

    try:
        # Only accept official Open Food Facts image URLs
        if (
            parsed.scheme != "https"
            or parsed.hostname != "images.openfoodfacts.org"
            or not parsed.path.startswith("/images/products/")
            or not barcode.isdigit()
        ):
            raise ValueError("Unexpected image URL or barcode")

        # OFF selected-image URLs commonly end with
        # .400.jpg. Replace with documented full-size
        # variant when the URL has that structure.
        full_url = re.sub(
            r"\.(?:100|200|400)\.jpg$",
            ".full.jpg",
            original_url
        )

        record["download_url"] = full_url

        request = Request(
            full_url,
            headers={"User-Agent": user_agent}
        )

        with urlopen(request, timeout=40) as response:
            final_host = urlparse(response.geturl()).hostname

            if final_host != "images.openfoodfacts.org":
                raise ValueError("Unexpected redirect host")

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if not content_type.startswith("image/"):
                raise ValueError("Response is not an image")

            content = response.read(MAX_BYTES + 1)

        if len(content) > MAX_BYTES:
            raise ValueError("Image exceeds 16 MB")

        # Verify the image can be opened
        with Image.open(io.BytesIO(content)) as image:
            image.verify()

        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
            image_format = image.format

        extensions = {
            "JPEG": ".jpg",
            "PNG": ".png",
            "WEBP": ".webp"
        }

        if image_format not in extensions:
            raise ValueError(
                f"Unsupported image format: {image_format}"
            )

        extension = extensions[image_format]

        destination = (
            OUTPUT /
            f"{barcode}_nova{group}_ingredients{extension}"
        )

        destination.write_bytes(content)

        record.update({
            "local_path": str(destination),
            "width": width,
            "height": height,
            "size_bytes": len(content),
            "status": "downloaded"
        })

        print(
            f"NOVA {group} | {barcode} | "
            f"{width}x{height} | downloaded"
        )

    except Exception as error:
        record["error"] = str(error)

        print(
            f"NOVA {group} | {barcode} | "
            f"FAILED: {error}"
        )

    results.append(record)

    time.sleep(2)

# Save the image download and dimension metadata
with LOG.open("w", encoding="utf-8") as file:
    for row in results:
        file.write(json.dumps(row) + "\n")

print("\n=== FINAL DOWNLOAD SUMMARY ===")

for group in (1, 2, 3, 4):
    group_records = [
        r for r in results
        if r["nova_group"] == group
    ]

    downloaded = sum(
        r["status"] == "downloaded"
        for r in group_records
    )

    print(
        f"NOVA {group}: "
        f"{downloaded}/{len(group_records)} downloaded"
    )

print(
    "Total downloaded:",
    sum(r["status"] == "downloaded" for r in results)
)

print("Metadata saved:", LOG)
print("Images saved:", OUTPUT)
