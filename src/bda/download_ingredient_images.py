
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

SOURCE = Path("data/images/image_metadata.jsonl")
OUTPUT = Path("data/images/ingredients")
LOG = Path("data/images/ingredient_downloads.jsonl")

OUTPUT.mkdir(parents=True, exist_ok=True)

email = os.environ.get("OFF_CONTACT_EMAIL")

if not email:
    raise SystemExit("Set OFF_CONTACT_EMAIL first.")

user_agent = f"FoodFactsStudentProject/0.1 ({email})"

# Read existing image metadata
with SOURCE.open("r", encoding="utf-8") as file:
    records = [json.loads(line) for line in file if line.strip()]

# Select ingredient images only
images = [
    row for row in records
    if row.get("image_type") == "ingredients"
    and row.get("image_url")
]

print("Ingredient images available:", len(images))

results = []
downloaded = 0

for row in images[:6]:
    barcode = row["barcode"]
    url = row["image_url"]

    parsed = urlparse(url)

    if (
        parsed.scheme != "https"
        or parsed.hostname != "images.openfoodfacts.org"
        or not barcode.isdigit()
    ):
        print("Skipped invalid URL or barcode:", barcode)
        continue

    ext = Path(parsed.path).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"

    destination = OUTPUT / f"{barcode}_ingredients{ext}"
    status = "failed"
    error_message = ""

    try:
        if destination.exists() and destination.stat().st_size > 0:
            status = "already_present"
        else:
            request = Request(
                url,
                headers={"User-Agent": user_agent}
            )

            with urlopen(request, timeout=30) as response:
                content_type = response.headers.get(
                    "Content-Type", ""
                ).lower()

                if not content_type.startswith("image/"):
                    raise ValueError("Response is not an image")

                content = response.read(8_000_001)

            if len(content) > 8_000_000:
                raise ValueError("Image exceeds 8 MB")

            destination.write_bytes(content)
            status = "downloaded"
            downloaded += 1

    except Exception as error:
        error_message = str(error)
        print("Download failed:", barcode, error_message)

    results.append({
        "barcode": barcode,
        "image_type": "ingredients",
        "image_url": url,
        "local_path": str(destination) if status != "failed" else "",
        "status": status,
        "size_bytes": (
            destination.stat().st_size
            if status != "failed" else 0
        ),
        "error": error_message,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source": "Open Food Facts"
    })

    print(barcode, status)
    time.sleep(2)

# Save a separate download log
with LOG.open("w", encoding="utf-8") as file:
    for row in results:
        file.write(json.dumps(row) + "\n")

print("\n=== INGREDIENT IMAGE RESULTS ===")
print("Available URLs:", len(images))
print("Newly downloaded:", downloaded)
print(
    "Available locally:",
    sum(r["status"] in ("downloaded", "already_present") for r in results)
)
print("Failed:", sum(r["status"] == "failed" for r in results))
print("Download log:", LOG)
