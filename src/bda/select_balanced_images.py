
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

CONTACT = os.environ.get("OFF_CONTACT_EMAIL")

if not CONTACT:
    raise SystemExit("Set OFF_CONTACT_EMAIL first.")

USER_AGENT = f"FoodFactsStudentProject/0.1 ({CONTACT})"

OUTPUT = Path("data/images/balanced_image_candidates.jsonl")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

PER_GROUP = 8

spark = (
    SparkSession.builder
    .appName("Balanced_Image_Selection")
    .master("local[1]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    products = (
        spark.read.parquet("data/india_candidates.parquet")
        .select("code", "nova_group")
        .filter(F.col("code").rlike(r"^[0-9]{8,14}$"))
        .filter(F.col("nova_group").between(1, 4))
        .dropDuplicates(["code"])
    )

    selected = []

    for group in (1, 2, 3, 4):
        rows = (
            products
            .filter(F.col("nova_group") == group)
            .orderBy(F.rand(seed=42 + group))
            .limit(PER_GROUP)
            .collect()
        )

        for row in rows:
            selected.append({
                "barcode": row["code"],
                "nova_group": int(row["nova_group"])
            })

finally:
    spark.stop()

print("Selected products:", len(selected))

results = []

for index, product in enumerate(selected):
    if index:
        time.sleep(5)

    barcode = product["barcode"]

    url = (
        "https://world.openfoodfacts.org"
        f"/api/v3.6/product/{barcode}.json"
        "?fields=code,selected_images"
    )

    image_url = None
    language = None
    status = "missing"
    error = ""

    try:
        request = Request(
            url, headers={"User-Agent": USER_AGENT}
        )

        with urlopen(request, timeout=25) as response:
            payload = json.load(response)

        details = (
            (payload.get("product") or {})
            .get("selected_images") or {}
        )

        ingredients = details.get("ingredients") or {}
        display = ingredients.get("display") or {}

        if display:
            language = (
                "en" if "en" in display
                else sorted(display)[0]
            )
            image_url = display[language]
            status = "available"

    except Exception as exc:
        status = "error"
        error = str(exc)

    results.append({
        "barcode": barcode,
        "nova_group": product["nova_group"],
        "image_type": "ingredients",
        "image_url": image_url,
        "image_language": language,
        "status": status,
        "error": error,
        "source": "Open Food Facts API",
        "retrieved_at": datetime.now(
            timezone.utc
        ).isoformat()
    })

    print(
        f"NOVA {product['nova_group']}: "
        f"{barcode} -> {status}"
    )

with OUTPUT.open("w", encoding="utf-8") as file:
    for row in results:
        file.write(json.dumps(row) + "\n")

print("\n=== BALANCED IMAGE DATASET SUMMARY ===")

for group in (1, 2, 3, 4):
    group_rows = [
        r for r in results
        if r["nova_group"] == group
    ]

    available = sum(
        r["status"] == "available"
        for r in group_rows
    )
    errors = sum(
        r["status"] == "error"
        for r in group_rows
    )

    print(
        f"NOVA {group}: "
        f"checked={len(group_rows)}, "
        f"available={available}, "
        f"errors={errors}"
    )

print("\nTotal checked:", len(results))
print(
    "Total images available:",
    sum(r["status"] == "available" for r in results)
)
print("Metadata saved:", OUTPUT)
