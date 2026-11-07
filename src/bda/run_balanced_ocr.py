
import json
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageOps
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

INVENTORY = "data/images/balanced_image_inventory.parquet"
REVIEW = "data/images/balanced_readability_review.csv"

OUTPUT = Path("data/ocr/balanced")
OUTPUT.mkdir(parents=True, exist_ok=True)

JSON_OUTPUT = OUTPUT / "ocr_results.jsonl"
PARQUET_OUTPUT = str(OUTPUT / "ocr_audit.parquet")

MAX_SIDE = 3000

spark = (
    SparkSession.builder
    .appName("Balanced_OCR_Experiment")
    .master("local[1]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # Read the image inventory created in Step 34
    images = (
        spark.read.parquet(INVENTORY)
        .filter(F.col("download_status") == "downloaded")
        .select(
            "barcode",
            "nova_group",
            "local_path",
            "width",
            "height"
        )
    )

    # Read your manually completed CSV
    reviews = (
        spark.read.option("header", True)
        .csv(REVIEW)
        .select(
            "barcode",
            F.col("nova_group").alias("review_nova"),
            "has_ingredient_list",
            "readable_by_eye",
            "notes"
        )
    )

    combined = images.join(reviews, "barcode", "left")
    rows = combined.orderBy(
        "nova_group", "barcode"
    ).collect()

    if len(rows) != 8:
        raise ValueError(
            "Expected 8 joined images. Check review "
            "barcodes, duplicates and leading zeros."
        )

    # Validate manually entered review values
    for row in rows:
        if str(row["review_nova"]) != str(row["nova_group"]):
            raise ValueError(
                f"Missing or mismatched review: {row['barcode']}"
            )

        for field in ("has_ingredient_list", "readable_by_eye"):
            value = str(row[field]).strip().lower()
            if value not in ("yes", "no"):
                raise ValueError(
                    f"Enter yes/no for {field}: {row['barcode']}"
                )

    results = []

    print("\n=== BALANCED OCR EXPERIMENT ===")

    for row in rows:
        barcode = row["barcode"]
        nova = row["nova_group"]
        original_path = Path(row["local_path"])

        print(f"\nNOVA {nova} | Barcode {barcode}")

        ingredient_list = (
            row["has_ingredient_list"].strip().lower() == "yes"
        )
        readable = (
            row["readable_by_eye"].strip().lower() == "yes"
        )

        print("Ingredient list present:", ingredient_list)
        print("Readable by eye:", readable)

        # Prepare one common image for both OCR methods.
        # Do not apply contrast enhancement or sharpening.
        prepared_path = OUTPUT / f"{barcode}_input.jpg"
        preparation_error = ""

        try:
            with Image.open(original_path) as image:
                image = ImageOps.exif_transpose(image)
                image = image.convert("RGB")

                factor = min(
                    1.0, MAX_SIDE / max(image.size)
                )

                if factor < 1:
                    new_size = (
                        round(image.width * factor),
                        round(image.height * factor)
                    )
                    image = image.resize(
                        new_size, Image.Resampling.LANCZOS
                    )

                image.save(prepared_path, quality=95)

        except Exception as error:
            preparation_error = str(error)

        for psm in (6, 11):
            text = ""
            elapsed = 0.0
            status = "error"
            error_message = preparation_error

            if not preparation_error:
                start = time.perf_counter()

                try:
                    process = subprocess.run(
                        [
                            "tesseract",
                            str(prepared_path),
                            "stdout",
                            "-l", "eng",
                            "--psm", str(psm)
                        ],
                        capture_output=True,
                        text=True,
                        timeout=120
                    )

                    elapsed = round(
                        time.perf_counter() - start, 3
                    )

                    if process.returncode != 0:
                        raise RuntimeError(
                            process.stderr.strip()
                        )

                    text = process.stdout.strip()
                    status = (
                        "text_detected" if text
                        else "empty_text"
                    )

                except Exception as error:
                    elapsed = round(
                        time.perf_counter() - start, 3
                    )
                    error_message = str(error)

            record = {
                "barcode": barcode,
                "nova_group": int(nova),
                "psm": psm,
                "has_ingredient_list": ingredient_list,
                "readable_by_eye": readable,
                "review_notes": row["notes"] or "",
                "original_image": str(original_path),
                "ocr_input_image": (
                    str(prepared_path)
                    if not preparation_error else None
                ),
                "original_width": row["width"],
                "original_height": row["height"],
                "ocr_text": text,
                "word_count": len(text.split()),
                "processing_seconds": elapsed,
                "status": status,
                "error": error_message
            }

            results.append(record)

            print(
                f"  PSM {psm}: {status}, "
                f"words={record['word_count']}, "
                f"time={elapsed}s"
            )

            print(
                "  Preview:",
                text[:120].replace("\n", " ")
            )

    # Save semi-structured OCR output
    with JSON_OUTPUT.open(
        "w", encoding="utf-8"
    ) as file:
        for row in results:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    # Use Spark to process OCR data and save Parquet
    audit = spark.read.json(str(JSON_OUTPUT))

    audit.write.mode("overwrite").parquet(
        PARQUET_OUTPUT
    )

    print("\n=== OCR SUMMARY BY NOVA GROUP ===")

    audit.groupBy("nova_group", "psm").agg(
        F.count("*").alias("images"),
        F.sum(
            F.when(
                F.col("status") == "text_detected", 1
            ).otherwise(0)
        ).alias("nonempty_results"),
        F.round(
            F.avg("word_count"), 1
        ).alias("average_words"),
        F.round(
            F.sum("processing_seconds"), 3
        ).alias("total_seconds")
    ).orderBy(
        "nova_group", "psm"
    ).show(20, truncate=False)

    print("\n=== MANUAL QUALITY SUMMARY ===")

    audit.filter(F.col("psm") == 6).agg(
        F.count("*").alias("images"),
        F.sum(
            F.col("readable_by_eye").cast("int")
        ).alias("readable"),
        F.sum(
            F.col("has_ingredient_list").cast("int")
        ).alias("with_ingredient_list"),
        F.sum(
            (
                F.col("readable_by_eye") &
                F.col("has_ingredient_list")
            ).cast("int")
        ).alias("eligible_for_transcription")
    ).show(truncate=False)

    print("\n=== FINAL OUTPUT ===")
    print("Images tested:", len(rows))
    print("OCR attempts:", len(results))
    print("JSON results:", JSON_OUTPUT)
    print("Parquet audit:", PARQUET_OUTPUT)

finally:
    spark.stop()
