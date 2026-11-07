
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("OCR_Quality_Audit")
    .master("local[1]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # Original OCR experiment
    original = (
        spark.read.json("data/ocr/ocr_results.jsonl")
        .select(
            "barcode",
            F.col("word_count").alias("original_words"),
            F.col("processing_seconds").alias(
                "original_seconds"
            ),
            F.col("ocr_text").alias("original_text"),
            F.col("status").alias("original_status")
        )
    )

    # Improved OCR experiments
    improved = spark.read.json(
        "data/ocr/improved/ocr_comparison.jsonl"
    )

    psm6 = (
        improved.filter(F.col("psm") == 6)
        .select(
            "barcode",
            F.col("word_count").alias("psm6_words"),
            F.col("processing_seconds").alias(
                "psm6_seconds"
            ),
            F.col("ocr_text").alias("psm6_text")
        )
    )

    psm11 = (
        improved.filter(F.col("psm") == 11)
        .select(
            "barcode",
            F.col("word_count").alias("psm11_words"),
            F.col("processing_seconds").alias(
                "psm11_seconds"
            ),
            F.col("ocr_text").alias("psm11_text")
        )
    )

    # Product data from our earlier BDA work
    products = (
        spark.read.parquet("data/india_candidates.parquet")
        .select(
            F.col("code").cast("string").alias("barcode"),
            "nova_group"
        )
        .dropDuplicates(["barcode"])
    )

    # Combine all OCR results
    audit = (
        original
        .join(psm6, "barcode", "inner")
        .join(psm11, "barcode", "inner")
        .join(products, "barcode", "left")
    )

    # Readability annotations based on visual inspection
    # reported by the project member.
    readable_barcodes = [
        "8902979052405",
        "8903553002809"
    ]

    audit = audit.withColumn(
        "readable_by_eye",
        F.col("barcode").isin(readable_barcodes)
    )

    # Only the first product clearly contains an
    # ingredient list in the shared OCR preview.
    # This is not an accuracy certification.
    audit = audit.withColumn(
        "ingredient_list_in_ocr_preview",
        F.col("barcode") == "8902979052405"
    )

    audit = (
        audit.withColumn(
            "psm6_word_change",
            F.col("psm6_words") - F.col("original_words")
        )
        .withColumn(
            "psm11_word_change",
            F.col("psm11_words") - F.col("original_words")
        )
    )

    print("\n=== OCR QUALITY AUDIT ===")

    audit.select(
        "barcode",
        "original_words",
        "psm6_words",
        "psm11_words",
        "readable_by_eye",
        "ingredient_list_in_ocr_preview"
    ).orderBy("barcode").show(20, truncate=False)

    print("\n=== SUMMARY ===")

    summary = audit.agg(
        F.count("*").alias("products"),
        F.sum(
            F.col("readable_by_eye").cast("int")
        ).alias("readable_images"),
        F.sum("original_words").alias("original_words"),
        F.sum("psm6_words").alias("psm6_words"),
        F.sum("psm11_words").alias("psm11_words"),
        F.round(
            F.sum("original_seconds"), 3
        ).alias("original_time"),
        F.round(
            F.sum("psm6_seconds"), 3
        ).alias("psm6_time"),
        F.round(
            F.sum("psm11_seconds"), 3
        ).alias("psm11_time")
    )

    summary.show(truncate=False)

    print("\n=== NOVA LINKAGE ===")

    audit.groupBy("nova_group").count().orderBy(
        "nova_group"
    ).show()

    output = "data/ocr/ocr_quality_audit.parquet"

    audit.write.mode("overwrite").parquet(output)

    saved = spark.read.parquet(output)

    if saved.count() != 6:
        raise ValueError(
            "Expected six products in the OCR audit."
        )

    print("\nAudit successfully saved:", output)

finally:
    spark.stop()
