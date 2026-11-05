
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("Food_Source_Integration")
    .master("local[1]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.files.maxPartitionBytes", "32m")
    .config("spark.sql.parquet.columnarReaderBatchSize", "256")
    .config("spark.sql.autoBroadcastJoinThreshold", "-1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    print("Reading Open Food Facts...")

    off = (
        spark.read.parquet("data/food.parquet")
        .select(
            "code", "product_name",
            "ingredients_text", "nova_group"
        )
        .withColumn("off_code", F.trim("code"))
        .filter(
            F.col("off_code").rlike("^[0-9]+$")
            & F.length("off_code").isin(8, 12, 13, 14)
        )
        .withColumn(
            "gtin14", F.lpad("off_code", 14, "0")
        )
        .withColumn(
            "off_product_name",
            F.coalesce(
                F.expr(
                    "get(filter(product_name, "
                    "x -> x.lang = 'en'), 0).text"
                ),
                F.expr(
                    "get(filter(product_name, "
                    "x -> x.lang = 'main'), 0).text"
                )
            )
        )
        .withColumn(
            "off_ingredients_en",
            F.expr(
                "get(filter(ingredients_text, "
                "x -> x.lang = 'en' AND "
                "length(trim(x.text)) > 0), 0).text"
            )
        )
        .select(
            "gtin14", "off_code",
            "off_product_name",
            "off_ingredients_en",
            "nova_group"
        )
    )

    print("Reading curated USDA products...")

    usda = (
        spark.read.parquet(
            "data/integration/usda_latest.parquet"
        )
        .select(
            "gtin14",
            F.col("fdc_id").alias("usda_fdc_id"),
            F.col("description").alias("usda_description"),
            F.col("ingredients").alias("usda_ingredients"),
            F.col("brand_name").alias("usda_brand"),
            F.col("publication_date").alias("usda_date")
        )
    )

    print("Joining both databases...")

    joined = off.join(usda, on="gtin14", how="inner")

    # Track any multiple OFF records sharing a normalized GTIN
    partition = Window.partitionBy("gtin14")

    rank_window = partition.orderBy(
        F.when(
            F.length(F.trim("off_ingredients_en")) > 0, 1
        ).otherwise(0).desc(),
        F.col("off_code").asc(),
        F.col("off_product_name").asc_nulls_last()
    )

    integrated = (
        joined
        .withColumn(
            "off_records_for_gtin",
            F.count("*").over(partition)
        )
        .withColumn(
            "row_num", F.row_number().over(rank_window)
        )
        .filter(F.col("row_num") == 1)
        .drop("row_num")
    )

    output = "data/integration/combined_products.parquet"

    print("Saving combined data...")

    integrated.write.mode("overwrite").parquet(output)

    result = spark.read.parquet(output)

    has_off = (
        F.length(F.trim(F.coalesce(
            F.col("off_ingredients_en"), F.lit("")
        ))) > 0
    )

    has_usda = (
        F.length(F.trim(F.coalesce(
            F.col("usda_ingredients"), F.lit("")
        ))) > 0
    )

    stats = result.agg(
        F.count("*").alias("matched_products"),
        F.sum(F.when(has_off, 1).otherwise(0))
            .alias("with_off_ingredients"),
        F.sum(F.when(has_usda, 1).otherwise(0))
            .alias("with_usda_ingredients"),
        F.sum(F.when(has_off & has_usda, 1).otherwise(0))
            .alias("with_both_ingredients"),
        F.sum(
            F.col("off_records_for_gtin") - 1
        ).alias("extra_off_records")
    ).first()

    print("\n=== INTEGRATION RESULTS ===")

    for key, value in stats.asDict().items():
        print(f"{key}: {value:,}")

    print("\n=== SAMPLE INTEGRATED PRODUCTS ===")

    result.select(
        "gtin14",
        "off_product_name",
        "usda_description",
        "nova_group"
    ).show(5, truncate=35)

    print("\nSaved to:", output)

finally:
    spark.stop()
