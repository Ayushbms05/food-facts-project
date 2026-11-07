
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Replace with a team contact email.
# Do not commit your personal email to GitHub.
import os
contact = os.environ.get("OFF_CONTACT_EMAIL")

if not contact:
    raise SystemExit(
        "Set OFF_CONTACT_EMAIL before running."
    )

USER_AGENT = f"FoodFactsStudentProject/0.1 ({contact})"

output_dir = Path("data/images")
output_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Select a few real Indian product barcodes
spark = (
    SparkSession.builder
    .appName("Image_Pilot")
    .master("local[1]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    codes = [
        row["code"]
        for row in (
            spark.read.parquet("data/india_candidates.parquet")
            .select("code")
            .filter(F.col("code").rlike(r"^[0-9]{8,14}$"))
            .distinct()
            .orderBy(F.rand(seed=42))
            .limit(8)
            .collect()
        )
    ]
finally:
    spark.stop()

# Step 2: Request image metadata and a few images
rows = []
downloaded = 0

for index, code in enumerate(codes):
    if index:
        time.sleep(5)  # Respect API rate limits

    api_url = (
        "https://world.openfoodfacts.org"
        f"/api/v3.6/product/{code}.json"
        "?fields=code,selected_images"
    )

    try:
        request = Request(
            api_url,
            headers={"User-Agent": USER_AGENT}
        )

        with urlopen(request, timeout=20) as response:
            payload = json.load(response)

        product = payload.get("product") or {}
        selected = product.get("selected_images") or {}

        for image_type in ("ingredients", "nutrition", "front"):
            details = selected.get(image_type) or {}
            urls = details.get("display") or details.get("small") or {}

            image_url = (
                urls.get("en") or
                next(iter(urls.values()), None)
            )

            status = "not_available"
            local_path = ""
            file_size = 0

            if image_url:
                status = "available_not_downloaded"

                if downloaded < 3:
                    try:
                        image_request = Request(
                            image_url,
                            headers={"User-Agent": USER_AGENT}
                        )

                        with urlopen(image_request, timeout=20) as response:
                            content_type = response.headers.get(
                                "Content-Type", ""
                            ).lower()
                            data = response.read(5_000_001)

                        if not content_type.startswith("image/"):
                            raise ValueError("Response is not an image")
                        if len(data) > 5_000_000:
                            raise ValueError("Image exceeds 5 MB limit")

                        extension = (
                            ".png" if "png" in content_type
                            else ".webp" if "webp" in content_type
                            else ".jpg"
                        )

                        path = output_dir / f"{code}_{image_type}{extension}"
                        path.write_bytes(data)

                        downloaded += 1
                        status = "downloaded"
                        local_path = str(path)
                        file_size = len(data)

                    except Exception as error:
                        status = f"download_error: {type(error).__name__}"

            rows.append({
                "barcode": code,
                "image_type": image_type,
                "image_url": image_url,
                "local_path": local_path,
                "size_bytes": file_size,
                "status": status,
                "source": "Open Food Facts API",
                "retrieved_at": datetime.now(
                    timezone.utc
                ).isoformat()
            })

    except Exception as error:
        print(f"Product {code}: {type(error).__name__}: {error}")
        rows.append({
            "barcode": code,
            "image_type": "product",
            "status": f"api_error: {type(error).__name__}",
            "source": "Open Food Facts API"
        })

# Step 3: Save metadata as JSONL
metadata_path = output_dir / "image_metadata.jsonl"

with metadata_path.open("w", encoding="utf-8") as file:
    for row in rows:
        file.write(json.dumps(row) + "\n")

print("\n=== IMAGE COLLECTION RESULTS ===")
print("Products checked:", len(codes))
print("Images downloaded:", downloaded)
print("Metadata records:", len(rows))
print("Metadata saved to:", metadata_path)

print("\nImage statuses:")
for row in rows:
    print(row["barcode"], row["image_type"], row["status"])
