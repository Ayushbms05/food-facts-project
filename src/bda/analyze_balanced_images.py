
import csv
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("Balanced_Image_Quality_Audit")
    .master("local[1]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # 32 products checked in Step 32
    candidates = (
        spark.read.json(
            "data/images/balanced_image_candidates.jsonl"
        )
        .select(
            "barcode",
            "nova_group",
            F.col("status").alias("api_status"),
            "image_language"
        )
    )

    # 8 downloaded images from Step 33
    downloads = (
        spark.read.json(
            "data/images/balanced_download_log.jsonl"
        )
        .select(
            "barcode",
            "nova_group",
            F.col("status").alias("download_status"),
            "local_path",
            "width",
            "height",
            "size_bytes"
        )
    )

    # Existing Indian product data
    products = (
        spark.read.parquet(
            "data/india_candidates.parquet"
        )
        .select(
            F.col("code").alias("barcode"),
            F.col("nova_group").alias("source_nova")
        )
        .dropDuplicates(["barcode"])
    )

    # Join the three datasets
    inventory = (
        candidates
        .join(
            downloads,
            ["barcode", "nova_group"],
            "left"
        )
        .join(products, "barcode", "left")
    )

    inventory = (
        inventory.withColumn(
            "download_status",
            F.coalesce(
                F.col("download_status"),
                F.lit("not_selected")
            )
        )
        .withColumn(
            "total_pixels",
            F.col("width") * F.col("height")
        )
        .withColumn(
            "shortest_side",
            F.least("width", "height")
        )
        .withColumn(
            "small_image_warning",
            F.when(
                F.col("download_status") == "downloaded",
                F.col("shortest_side") < 600
            ).otherwise(False)
        )
        .withColumn(
            "nova_label_matches",
            F.col("nova_group") == F.col("source_nova")
        )
    )

    print("\n=== IMAGE QUALITY INVENTORY ===")

    inventory.filter(
        F.col("download_status") == "downloaded"
    ).select(
        "barcode",
        "nova_group",
        "width",
        "height",
        "size_bytes",
        "small_image_warning"
    ).orderBy(
        "nova_group", "barcode"
    ).show(20, truncate=False)

    print("\n=== NOVA GROUP SUMMARY ===")

    summary = inventory.groupBy("nova_group").agg(
        F.count("*").alias("checked"),
        F.sum(
            F.when(
                F.col("api_status") == "available", 1
            ).otherwise(0)
        ).alias("available"),
        F.sum(
            F.when(
                F.col("download_status") == "downloaded", 1
            ).otherwise(0)
        ).alias("downloaded"),
        F.sum(
            F.when(
                F.col("small_image_warning"), 1
            ).otherwise(0)
        ).alias("small_image_warnings")
    )

    summary.orderBy("nova_group").show(
        truncate=False
    )

    print("\n=== DATA INTEGRATION CHECK ===")

    total = inventory.count()
    linked = inventory.filter(
        F.col("nova_label_matches")
    ).count()

    downloaded = inventory.filter(
        F.col("download_status") == "downloaded"
    ).count()

    print("Total inventory records:", total)
    print("Matching NOVA labels:", linked)
    print("Downloaded image records:", downloaded)

    if total != 32 or linked != total:
        raise ValueError(
            "Unexpected record count or NOVA label mismatch."
        )

    if downloaded != 8:
        raise ValueError(
            "Expected 8 downloaded image records."
        )

    output = "data/images/balanced_image_inventory.parquet"

    inventory.write.mode("overwrite").parquet(output)

    # Create a blank manual image-review sheet.
    # Do not assume resolution guarantees readability.
    review_path = Path(
        "data/images/balanced_readability_review.csv"
    )

    rows = (
        inventory.filter(
            F.col("download_status") == "downloaded"
        )
        .select("barcode", "nova_group")
        .orderBy("nova_group", "barcode")
        .collect()
    )

    with review_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "barcode",
            "nova_group",
            "has_ingredient_list",
            "readable_by_eye",
            "notes"
        ])

        for row in rows:
            writer.writerow([
                row["barcode"],
                row["nova_group"],
                "",
                "",
                ""
            ])

    print("\n=== OUTPUT FILES ===")
    print("Inventory saved:", output)
    print("Manual review sheet:", review_path)

finally:
    spark.stop()
