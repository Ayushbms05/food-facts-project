
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import StorageLevel

spark = (
    SparkSession.builder
    .appName("OFF_USDA_Integration")
    .master("local[1]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.files.maxPartitionBytes", "32m")
    .config("spark.sql.parquet.columnarReaderBatchSize", "256")
    .config("spark.sql.autoBroadcastJoinThreshold", "-1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


def normalize_barcodes(df, column_name):
    return (
        df.select(
            F.trim(F.col(column_name)).alias("barcode")
        )
        .filter(
            F.col("barcode").rlike("^[0-9]+$")
            & F.length("barcode").isin(8, 12, 13, 14)
        )
        .select(
            F.lpad("barcode", 14, "0").alias("gtin14")
        )
        .distinct()
    )


try:
    print("\nReading Open Food Facts...")
    off = spark.read.parquet(
        "data/food.parquet"
    ).select("code")

    print("Reading USDA...")
    usda = (
        spark.read
        .option("header", "true")
        .option("quote", '"')
        .option("escape", '"')
        .csv("data/usda/branded_food.csv")
        .select("gtin_upc")
    )

    print("\nNormalizing barcodes...")

    off_keys = normalize_barcodes(
        off, "code"
    ).persist(StorageLevel.DISK_ONLY)

    usda_keys = normalize_barcodes(
        usda, "gtin_upc"
    ).persist(StorageLevel.DISK_ONLY)

    off_count = off_keys.count()
    usda_count = usda_keys.count()

    print("\n=== UNIQUE VALID-FORMAT GTINS ===")
    print("Open Food Facts:", off_count)
    print("USDA:", usda_count)

    print("\nMatching across sources...")

    matches = off_keys.join(
        usda_keys,
        on="gtin14",
        how="inner"
    )

    matched_count = matches.count()

    print("\n=== INTEGRATION RESULTS ===")
    print("Matching unique GTINs:", matched_count)

    print("\nSample matching barcodes:")
    matches.show(10, truncate=False)

    matches.write.mode("overwrite").parquet(
        "data/integration/shared_gtins.parquet"
    )

    print("\nMatched identifiers saved successfully!")

    off_keys.unpersist()
    usda_keys.unpersist()

finally:
    spark.stop()
