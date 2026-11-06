
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("FoodFacts_Data_Quality")
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
        "ingredients_text",
        "nova_group",
        "countries_tags"
    )

    has_text = F.coalesce(
        F.exists(
            "ingredients_text",
            lambda x: F.length(F.trim(x["text"])) > 0
        ),
        F.lit(False)
    )

    has_nova = F.col("nova_group").between(1, 4)

    is_india = F.coalesce(
        F.array_contains("countries_tags", "en:india"),
        F.lit(False)
    )

    data = df.select(
        has_text.alias("valid_text"),
        has_nova.alias("valid_nova"),
        is_india.alias("is_india"),
        F.col("nova_group")
    )

    summary = data.agg(
        F.count("*").alias("total"),
        F.sum(F.col("valid_text").cast("long")).alias("with_text"),
        F.sum(F.col("is_india").cast("long")).alias("india"),
        F.sum(
            (F.col("valid_text") & F.col("valid_nova"))
            .cast("long")
        ).alias("global_candidates"),
        F.sum(
            (
                F.col("valid_text")
                & F.col("valid_nova")
                & F.col("is_india")
            ).cast("long")
        ).alias("india_candidates")
    ).first()

    print("\n=== DATA QUALITY RESULTS ===")

    for name in summary.asDict():
        print(f"{name}: {summary[name]:,}")

    print("\n=== INDIA NOVA DISTRIBUTION ===")

    data.filter(
        F.col("is_india")
        & F.col("valid_text")
        & F.col("valid_nova")
    ).groupBy("nova_group").count().orderBy(
        "nova_group"
    ).show()

finally:
    spark.stop()
