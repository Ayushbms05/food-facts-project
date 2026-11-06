
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("Ingredient_Quality_Audit")
    .master("local[1]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.files.maxPartitionBytes", "32m")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


def basic_clean(name):
    text = F.lower(
        F.trim(F.coalesce(F.col(name), F.lit("")))
    )
    text = F.regexp_replace(
        text, r"^ingredients?\s*:\s*", ""
    )
    return F.trim(F.regexp_replace(text, r"\s+", " "))


def improved_clean(name):
    text = F.lower(
        F.trim(F.coalesce(F.col(name), F.lit("")))
    )

    # Remove HTML tags but keep the ingredient words
    text = F.regexp_replace(text, r"<[^>]+>", " ")

    # Handle some common HTML entities
    text = F.regexp_replace(
        text, r"&(amp|nbsp|quot|lt|gt);|&#39;", " "
    )

    # Remove introductory label
    text = F.regexp_replace(
        text, r"^ingredients?\s*:\s*", ""
    )

    # Standardize punctuation for comparison only
    text = F.regexp_replace(
        text, r"[^\p{L}\p{N}]+", " "
    )

    return F.trim(F.regexp_replace(text, r"\s+", " "))


try:
    df = spark.read.parquet(
        "data/integration/combined_products.parquet"
    ).select(
        "gtin14",
        "off_ingredients_en",
        "usda_ingredients",
        "nova_group"
    )

    data = (
        df.withColumn(
            "off_basic", basic_clean("off_ingredients_en")
        )
        .withColumn(
            "usda_basic", basic_clean("usda_ingredients")
        )
        .withColumn(
            "off_clean", improved_clean("off_ingredients_en")
        )
        .withColumn(
            "usda_clean", improved_clean("usda_ingredients")
        )
    )

    has_off = F.length(F.col("off_clean")) > 0
    has_usda = F.length(F.col("usda_clean")) > 0

    data = data.withColumn(
        "agreement_category",
        F.when(~has_off & ~has_usda, "neither")
        .when(~has_off, "usda_only")
        .when(~has_usda, "off_only")
        .when(
            F.col("off_basic") == F.col("usda_basic"),
            "same_basic"
        )
        .when(
            F.col("off_clean") == F.col("usda_clean"),
            "same_after_extra_cleanup"
        )
        .otherwise("still_different")
    )

    output = "data/integration/ingredient_audit.parquet"

    # Save without altering the original integrated dataset
    data.write.mode("overwrite").parquet(output)

    result = spark.read.parquet(output)

    print("\n=== IMPROVED INGREDIENT COMPARISON ===")

    result.groupBy(
        "agreement_category"
    ).count().orderBy(
        "agreement_category"
    ).show(truncate=False)

    print("\n=== SAMPLE REMAINING DIFFERENCES ===")

    result.filter(
        F.col("agreement_category") == "still_different"
    ).select(
        "gtin14",
        F.substring("off_clean", 1, 120)
            .alias("OFF_cleaned"),
        F.substring("usda_clean", 1, 120)
            .alias("USDA_cleaned")
    ).show(5, truncate=False)

    print("\nQuality audit saved to:", output)

finally:
    spark.stop()
