
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("USDA_Latest_Products")
    .master("local[1]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.files.maxPartitionBytes", "32m")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    base = "data/usda/"

    branded = (
        spark.read.option("header", "true")
        .option("quote", '"')
        .option("escape", '"')
        .csv(base + "branded_food.csv")
        .select(
            "fdc_id", "gtin_upc", "brand_name",
            "ingredients", "market_country",
            "modified_date"
        )
    )

    food = (
        spark.read.option("header", "true")
        .option("quote", '"')
        .option("escape", '"')
        .csv(base + "food.csv")
        .select("fdc_id", "description", "publication_date")
    )

    print("Joining USDA tables...")

    combined = branded.join(food, on="fdc_id", how="left")

    combined = combined.withColumn(
        "barcode", F.trim(F.col("gtin_upc"))
    ).filter(
        F.col("barcode").rlike("^[0-9]+$")
        & F.length("barcode").isin(8, 12, 13, 14)
    ).withColumn(
        "gtin14", F.lpad("barcode", 14, "0")
    )

    # Verify the GTIN check digit
    total = F.lit(0)

    for position in range(1, 14):
        digit = F.substring("gtin14", position, 1).cast("int")
        weight = 3 if position % 2 == 1 else 1
        total = total + digit * weight

    expected = (10 - total % 10) % 10
    actual = F.substring("gtin14", 14, 1).cast("int")

    combined = combined.filter(expected == actual)

    # Parse publication dates
    combined = combined.withColumn(
        "published_at",
        F.coalesce(
            F.try_to_timestamp(
                "publication_date", F.lit("yyyy-MM-dd")
            ),
            F.try_to_timestamp(
                "publication_date", F.lit("MM/dd/yyyy")
            )
        )
    )

    # Prefer the most recently published version
    window = Window.partitionBy("gtin14").orderBy(
        F.col("published_at").desc_nulls_last(),
        F.col("fdc_id").cast("long").desc_nulls_last()
    )

    latest = (
        combined
        .withColumn("row_num", F.row_number().over(window))
        .filter(F.col("row_num") == 1)
        .drop("row_num", "barcode", "published_at")
        .withColumn("source", F.lit("USDA"))
    )

    output = "data/integration/usda_latest.parquet"

    print("Saving latest USDA product versions...")

    latest.write.mode("overwrite").parquet(output)

    result = spark.read.parquet(output)

    print("\n=== USDA LATEST DATASET ===")
    print("Selected unique products:", result.count())

    missing_dates = result.filter(
        F.col("publication_date").isNull()
        | (F.trim("publication_date") == "")
    ).count()

    print("Records with missing publication dates:", missing_dates)

    print("\n=== SAMPLE PRODUCTS ===")
    result.select(
        "gtin14", "description",
        "publication_date", "brand_name"
    ).show(5, truncate=35)

    print("\nSaved to:", output)

finally:
    spark.stop()
