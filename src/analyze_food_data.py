
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("FoodFacts_Memory_Optimized")
    .master("local[1]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.sql.files.maxPartitionBytes", "32m")
    .config("spark.sql.parquet.columnarReaderBatchSize", "256")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # Read only the two columns required for this analysis
    df = spark.read.parquet("data/food.parquet").select(
        "ingredients_text",
        "nova_group"
    )

    print("\n=== DATASET ANALYSIS ===")

    has_ingredients = (
        F.col("ingredients_text").isNotNull()
        & (F.size("ingredients_text") > 0)
    )

    has_nova = F.col("nova_group").between(1, 4)

    results = df.agg(
        F.count("*").alias("total_products"),
        F.count(F.when(has_ingredients, 1)).alias("with_ingredients"),
        F.count(F.when(has_nova, 1)).alias("with_nova"),
        F.count(
            F.when(has_ingredients & has_nova, 1)
        ).alias("with_both")
    ).first()

    print("\n=== RESULTS ===")
    print("Total products:", results["total_products"])
    print("Products with ingredients:", results["with_ingredients"])
    print("Products with NOVA labels:", results["with_nova"])
    print("Products with both:", results["with_both"])

    print("\nAnalysis completed successfully!")

finally:
    spark.stop()
