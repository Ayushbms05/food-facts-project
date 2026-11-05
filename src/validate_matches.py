
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import StorageLevel

spark = (
    SparkSession.builder
    .appName("Barcode_Validation")
    .master("local[1]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.files.maxPartitionBytes", "32m")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    shared = (
        spark.read.parquet("data/integration/shared_gtins.parquet")
        .select("gtin14")
        .distinct()
    )

    # Calculate the GS1 check digit using Spark expressions
    weighted_sum = F.lit(0)

    for position in range(1, 14):
        digit = F.substring("gtin14", position, 1).cast("int")
        weight = 3 if position % 2 == 1 else 1
        weighted_sum = weighted_sum + digit * weight

    expected_digit = (10 - (weighted_sum % 10)) % 10

    actual_digit = F.substring("gtin14", 14, 1).cast("int")

    checked = shared.withColumn(
        "checksum_valid",
        expected_digit == actual_digit
    )

    stats = checked.agg(
        F.count("*").alias("matched"),
        F.sum(F.col("checksum_valid").cast("long"))
            .alias("valid")
    ).first()

    print("\n=== BARCODE VALIDATION ===")
    print("Total matching GTINs:", stats["matched"])
    print("Valid check digits:", stats["valid"])
    print(
        "Invalid check digits:",
        stats["matched"] - stats["valid"]
    )

    # Read only the barcode column from USDA
    usda = (
        spark.read
        .option("header", "true")
        .option("quote", '"')
        .option("escape", '"')
        .csv("data/usda/branded_food.csv")
        .select(F.trim("gtin_upc").alias("barcode"))
    )

    usda = (
        usda.filter(
            F.col("barcode").rlike("^[0-9]+$")
            & F.length("barcode").isin(8, 12, 13, 14)
        )
        .select(F.lpad("barcode", 14, "0").alias("gtin14"))
    )

    counts = (
        usda.groupBy("gtin14")
        .count()
        .persist(StorageLevel.DISK_ONLY)
    )

    duplicate_stats = counts.agg(
        F.count("*").alias("unique_gtins"),
        F.sum("count").alias("records"),
        F.sum(
            F.when(F.col("count") > 1, 1).otherwise(0)
        ).alias("repeated_gtins"),
        F.max("count").alias("max_records_per_gtin")
    ).first()

    print("\n=== USDA DUPLICATE ANALYSIS ===")

    for key, value in duplicate_stats.asDict().items():
        print(f"{key}: {value:,}")

    print("\n=== TOP 5 REPEATED GTINS ===")

    counts.orderBy(F.desc("count")).show(
        5, truncate=False
    )

    counts.unpersist()

finally:
    spark.stop()
