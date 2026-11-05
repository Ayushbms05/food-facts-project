
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("India_Food_Analysis")
    .master("local[1]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.sql.files.maxPartitionBytes", "32m")
    .config("spark.sql.parquet.columnarReaderBatchSize", "256")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    df = spark.read.parquet("data/food.parquet").select(
        "code",
        "product_name",
        "ingredients_text",
        "nova_group",
        "countries_tags"
    )

    # Check for non-empty ingredient text
    has_text = F.coalesce(
        F.exists(
            "ingredients_text",
            lambda x: x["text"].isNotNull()
            & (F.length(F.trim(x["text"])) > 0)
        ),
        F.lit(False)
    )

    # Check whether English ingredient text exists
    has_english = F.coalesce(
        F.exists(
            "ingredients_text",
            lambda x: (F.lower(x["lang"]) == "en")
            & x["text"].isNotNull()
            & (F.length(F.trim(x["text"])) > 0)
        ),
        F.lit(False)
    )

    # Select Indian products with ingredient text and NOVA labels
    india_df = df.filter(
        F.coalesce(
            F.array_contains("countries_tags", "en:india"),
            F.lit(False)
        )
        & has_text
        & F.col("nova_group").between(1, 4)
    ).cache()

    total = india_df.count()

    english_count = india_df.filter(has_english).count()

    missing_codes = india_df.filter(
        F.col("code").isNull()
        | (F.length(F.trim("code")) == 0)
    ).count()

    duplicate_codes = (
        india_df
        .filter(F.col("code").isNotNull())
        .groupBy("code")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    print("\n=== INDIA DATA QUALITY ===")
    print("Total candidate rows:", total)
    print("With English ingredients:", english_count)
    print("Without English ingredients:", total - english_count)
    print("Rows with missing barcode:", missing_codes)
    print("Barcodes appearing more than once:", duplicate_codes)

    print("\n=== INGREDIENT LANGUAGES ===")

    (
        india_df
        .select(F.explode("ingredients_text").alias("entry"))
        .filter(
            F.col("entry.text").isNotNull()
            & (F.length(F.trim(F.col("entry.text"))) > 0)
        )
        .groupBy("entry.lang")
        .count()
        .orderBy(F.desc("count"))
        .show(10)
    )

    print("\n=== SAVING INDIAN DATASET ===")

    india_df.coalesce(1).write.mode("overwrite").parquet(
        "data/india_candidates.parquet"
    )

    print("Indian candidate dataset saved successfully!")

    india_df.unpersist()

finally:
    spark.stop()
