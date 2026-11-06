
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("USDA_Data_Analysis")
    .master("local[1]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.sql.files.maxPartitionBytes", "32m")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    branded = (
        spark.read
        .option("header", "true")
        .option("quote", '"')
        .option("escape", '"')
        .csv("data/usda/branded_food.csv")
    )

    food = (
        spark.read
        .option("header", "true")
        .option("quote", '"')
        .option("escape", '"')
        .csv("data/usda/food.csv")
    )

    print("\n=== USDA DATASET INFORMATION ===")

    print("Branded food columns:", len(branded.columns))
    print("Food columns:", len(food.columns))

    print("\n=== ROW COUNTS ===")

    print("Branded food records:", branded.count())
    print("Food records:", food.count())

    print("\n=== BRANDED FOOD DATA QUALITY ===")

    has_barcode = (
        F.col("gtin_upc").isNotNull()
        & (F.length(F.trim("gtin_upc")) > 0)
    )

    has_ingredients = (
        F.col("ingredients").isNotNull()
        & (F.length(F.trim("ingredients")) > 0)
    )

    results = branded.agg(
        F.count("*").alias("total"),
        F.count(F.when(has_barcode, 1)).alias("with_barcode"),
        F.count(F.when(has_ingredients, 1)).alias("with_ingredients"),
        F.count(
            F.when(has_barcode & has_ingredients, 1)
        ).alias("with_both")
    ).first()

    for key, value in results.asDict().items():
        print(f"{key}: {value:,}")

    print("\n=== SAMPLE BRANDED PRODUCTS ===")

    branded.select(
        "fdc_id",
        "brand_name",
        "gtin_upc",
        "ingredients"
    ).show(5, truncate=45)

    print("\n=== SAMPLE FOOD DETAILS ===")

    food.select(
        "fdc_id",
        "description",
        "data_type"
    ).show(5, truncate=45)

    print("\nUSDA analysis completed successfully!")

finally:
    spark.stop()
