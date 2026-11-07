
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("Food_Image_Metadata_Analysis")
    .master("local[1]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # Read semi-structured JSONL image metadata
    images = spark.read.json(
        "data/images/image_metadata.jsonl"
    )

    # Read our existing Indian product records
    products = (
        spark.read.parquet(
            "data/india_candidates.parquet"
        )
        .select(
            F.col("code").alias("barcode"),
            "nova_group"
        )
        .dropDuplicates(["barcode"])
    )

    print("\n=== IMAGE METADATA ===")
    print("Metadata records:", images.count())
    print(
        "Unique products:",
        images.select("barcode").distinct().count()
    )

    # Join image information with product information
    combined = images.join(
        products,
        on="barcode",
        how="left"
    )

    # Mark images that are available online
    combined = combined.withColumn(
        "image_available",
        F.col("status").isin(
            "downloaded",
            "available_not_downloaded"
        )
    )

    print("\n=== IMAGE AVAILABILITY BY TYPE ===")

    summary = combined.groupBy("image_type").agg(
        F.count("*").alias("checked"),
        F.sum(
            F.col("image_available").cast("int")
        ).alias("available"),
        F.sum(
            (F.col("status") == "downloaded").cast("int")
        ).alias("downloaded")
    )

    summary.orderBy("image_type").show(
        truncate=False
    )

    print("\n=== IMAGE STATUS DISTRIBUTION ===")

    combined.groupBy("status").count().orderBy(
        "status"
    ).show(truncate=False)

    print("\n=== PRODUCT DATA INTEGRATION ===")

    print(
        "Image records linked to Indian products:",
        combined.filter(
            F.col("nova_group").isNotNull()
        ).count()
    )

    print("\n=== SAVING INVENTORY ===")

    output = "data/images/image_inventory.parquet"

    combined.write.mode("overwrite").parquet(output)

    print("Saved successfully:", output)

finally:
    spark.stop()
