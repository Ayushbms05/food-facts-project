
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("Ingredient_Comparison")
    .master("local[1]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.files.maxPartitionBytes", "32m")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # Read our previously combined dataset
    df = spark.read.parquet(
        "data/integration/combined_products.parquet"
    ).select(
        "gtin14",
        "off_ingredients_en",
        "usda_ingredients",
        "nova_group"
    )

    # Normalize formatting for comparison
    def normalize(column):
        text = F.lower(
            F.trim(F.coalesce(F.col(column), F.lit("")))
        )

        # Remove optional INGREDIENTS: prefix
        text = F.regexp_replace(
            text, r"^ingredients?\s*:\s*", ""
        )

        # Replace repeated whitespace with one space
        return F.trim(
            F.regexp_replace(text, r"\s+", " ")
        )

    data = (
        df.withColumn(
            "off_clean", normalize("off_ingredients_en")
        )
        .withColumn(
            "usda_clean", normalize("usda_ingredients")
        )
    )

    has_off = F.length("off_clean") > 0
    has_usda = F.length("usda_clean") > 0
    both = has_off & has_usda

    same_text = both & (
        F.col("off_clean") == F.col("usda_clean")
    )

    different_text = both & (
        F.col("off_clean") != F.col("usda_clean")
    )

    def count_if(condition):
        return F.sum(
            F.when(condition, 1).otherwise(0)
        )

    results = data.agg(
        F.count("*").alias("total"),
        count_if(both).alias("both_available"),
        count_if(same_text).alias("same"),
        count_if(different_text).alias("different"),
        count_if(~has_off & has_usda).alias("usda_only"),
        count_if(has_off & ~has_usda).alias("off_only"),
        count_if(~has_off & ~has_usda).alias("neither"),
        count_if(
            both & F.col("nova_group").between(1, 4)
        ).alias("both_with_nova")
    ).first()

    print("\n=== INGREDIENT COMPARISON RESULTS ===")

    for key, value in results.asDict().items():
        print(f"{key}: {value:,}")

    print("\n=== SAMPLE DIFFERENCES ===")

    data.filter(different_text).select(
        "gtin14",
        F.substring("off_ingredients_en", 1, 100)
         .alias("OFF_ingredients"),
        F.substring("usda_ingredients", 1, 100)
         .alias("USDA_ingredients")
    ).show(5, truncate=False)

    print("\nComparison completed!")

finally:
    spark.stop()
