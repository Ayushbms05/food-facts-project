
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("Streaming_Lookup_Audit")
    .master("local[1]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    events = spark.read.parquet(
        "data/streaming/step39/enriched_events.parquet"
    )

    products = (
        spark.read.parquet(
            "data/curated/source_aware_products.parquet"
        )
        .select(
            F.col("gtin14").alias("lookup_barcode"),
            F.col("nova_group").alias("lookup_nova")
        )
    )

    # Check actual barcode existence, not NOVA availability.
    audit = events.join(
        products,
        events.barcode == products.lookup_barcode,
        "left"
    )

    audit = audit.withColumn(
        "barcode_found",
        F.col("lookup_barcode").isNotNull()
    )

    audit = audit.withColumn(
        "nova_available",
        F.coalesce(
            F.col("lookup_nova").between(1, 4),
            F.lit(False)
        )
    )

    # The Step 39 generator used this barcode deliberately.
    audit = audit.withColumn(
        "intentionally_unknown",
        F.col("barcode") == "00000000000000"
    )

    audit = audit.withColumn(
        "lookup_category",
        F.when(
            F.col("intentionally_unknown")
            & ~F.col("barcode_found"),
            "intentional_unknown"
        )
        .when(
            F.col("intentionally_unknown")
            & F.col("barcode_found"),
            "sentinel_barcode_collision"
        )
        .when(
            ~F.col("barcode_found"),
            "unexpected_missing_barcode"
        )
        .when(
            ~F.col("nova_available"),
            "found_without_nova"
        )
        .otherwise("found_with_nova")
    )

    print("\n=== CORRECTED PRODUCT LOOKUP ===")

    audit.groupBy("lookup_category").count().orderBy(
        "lookup_category"
    ).show(truncate=False)

    print("\n=== VERIFICATION SUMMARY ===")

    audit.agg(
        F.count("*").alias("total_events"),
        F.countDistinct("event_id").alias("unique_events"),
        F.sum(
            F.col("barcode_found").cast("int")
        ).alias("barcodes_found"),
        F.sum(
            (~F.col("barcode_found")).cast("int")
        ).alias("barcodes_not_found"),
        F.sum(
            (
                F.col("barcode_found")
                & ~F.col("nova_available")
            ).cast("int")
        ).alias("found_but_missing_nova"),
        F.sum(
            F.col("intentionally_unknown").cast("int")
        ).alias("intentionally_unknown_events")
    ).show(truncate=False)

    print("\n=== ORIGINAL FLAG DISAGREEMENTS ===")

    audit.filter(
        F.col("product_found") != F.col("barcode_found")
    ).select(
        "event_id",
        "barcode",
        "product_found",
        "barcode_found",
        "lookup_nova"
    ).show(15, truncate=False)

    output = (
        "data/streaming/step39/lookup_audit_corrected.parquet"
    )

    audit.write.mode("overwrite").parquet(output)

    print("\nAudit saved:", output)

finally:
    spark.stop()
