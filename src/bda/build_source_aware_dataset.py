
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("Source_Aware_Food_Dataset")
    .master("local[1]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.files.maxPartitionBytes", "32m")
    .config("spark.sql.parquet.columnarReaderBatchSize", "256")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # Original combined product information
    products = spark.read.parquet(
        "data/integration/combined_products.parquet"
    )

    # Ingredient comparison results from Step 22
    audit = spark.read.parquet(
        "data/integration/ingredient_audit.parquet"
    ).select(
        "gtin14",
        "off_clean",
        "usda_clean",
        "agreement_category"
    )

    # Combine information using product identifiers
    combined = products.join(
        audit,
        on="gtin14",
        how="left"
    )

    has_off = F.length(
        F.trim(F.coalesce(
            F.col("off_clean"), F.lit("")
        ))
    ) > 0

    has_usda = F.length(
        F.trim(F.coalesce(
            F.col("usda_clean"), F.lit("")
        ))
    ) > 0

    valid_nova = F.col("nova_group").between(1, 4)

    # Preserve the source of model input
    combined = combined.withColumn(
        "model_text_source",
        F.when(has_off, F.lit("Open Food Facts"))
         .otherwise(F.lit("None"))
    )

    # Records that need manual investigation
    combined = combined.withColumn(
        "needs_review",
        F.when(
            F.col("agreement_category").isin(
                "still_different",
                "usda_only",
                "off_only",
                "neither"
            ),
            True
        ).otherwise(False)
    )

    # Potential DL candidate, not verified training data
    combined = combined.withColumn(
        "dl_candidate_off",
        F.when(has_off & valid_nova, True)
         .otherwise(False)
    )

    # Identify records where cleaning erased raw text
    raw_present = (
        (
            F.length(F.trim(F.coalesce(
                F.col("off_ingredients_en"), F.lit("")
            ))) > 0
        )
        |
        (
            F.length(F.trim(F.coalesce(
                F.col("usda_ingredients"), F.lit("")
            ))) > 0
        )
    )

    combined = combined.withColumn(
        "text_removed_by_cleaning",
        raw_present & ~has_off & ~has_usda
    )

    output = "data/curated/source_aware_products.parquet"

    print("Saving source-aware dataset...")
    combined.write.mode("overwrite").parquet(output)

    # Read the saved data to validate it
    result = spark.read.parquet(output)

    total = result.count()
    unique = result.select("gtin14").distinct().count()

    print("\n=== CURATED DATASET SUMMARY ===")
    print("Total records:", total)
    print("Unique product identifiers:", unique)

    if total != unique:
        raise ValueError(
            "Duplicate GTINs detected in curated data!"
        )

    print("\n=== AGREEMENT CATEGORIES ===")

    result.groupBy(
        "agreement_category"
    ).count().orderBy(
        "agreement_category"
    ).show(truncate=False)

    print("\n=== ADDITIONAL QUALITY CHECKS ===")

    result.agg(
        F.sum(
            F.col("needs_review").cast("long")
        ).alias("records_flagged_for_review"),
        F.sum(
            F.col("dl_candidate_off").cast("long")
        ).alias("off_dl_candidates"),
        F.sum(
            F.col("text_removed_by_cleaning").cast("long")
        ).alias("cleaning_removed_text")
    ).show(truncate=False)

    print("\nSaved successfully to:", output)

finally:
    spark.stop()
