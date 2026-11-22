
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType, StringType, StructField, StructType
)

SOURCE = "data/curated/source_aware_products.parquet"
BASE = Path("data/benchmarks")
CSV_DIR = BASE / "products_csv"
PARQUET_DIR = BASE / "products_parquet"
RESULT_FILE = BASE / "step37_results.json"

BASE.mkdir(parents=True, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("BDA_CSV_vs_Parquet_Benchmark")
    .master("local[1]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.sql.files.maxPartitionBytes", "32m")
    .config("spark.sql.parquet.columnarReaderBatchSize", "256")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

SCHEMA = StructType([
    StructField("gtin14", StringType(), True),
    StructField("nova_group", IntegerType(), True),
    StructField("agreement_category", StringType(), True),
    StructField("off_chars", IntegerType(), True),
    StructField("usda_chars", IntegerType(), True),
])


def folder_bytes(path):
    return sum(
        p.stat().st_size
        for p in path.rglob("*")
        if p.is_file()
    )


def query(fmt):
    start = time.perf_counter()

    if fmt == "csv":
        df = (
            spark.read.schema(SCHEMA)
            .option("header", True)
            .csv(str(CSV_DIR))
        )
    else:
        df = spark.read.parquet(str(PARQUET_DIR))

    rows = df.groupBy("agreement_category").agg(
        F.count("*").alias("records"),
        F.sum("off_chars").alias("total_off_chars"),
        F.sum("usda_chars").alias("total_usda_chars"),
        F.sum(
            F.when(
                F.col("nova_group").between(1, 4), 1
            ).otherwise(0)
        ).alias("valid_nova")
    ).collect()

    elapsed = time.perf_counter() - start

    result = sorted(
        (
            r["agreement_category"] or "",
            r["records"],
            r["total_off_chars"],
            r["total_usda_chars"],
            r["valid_nova"]
        )
        for r in rows
    )

    return elapsed, result


try:
    # Create exactly the same flat data for both formats.
    source = spark.read.parquet(SOURCE)

    flat = source.select(
        F.col("gtin14").cast("string"),
        F.col("nova_group").cast("int"),
        "agreement_category",
        F.length(
            F.coalesce(F.col("off_clean"), F.lit(""))
        ).alias("off_chars"),
        F.length(
            F.coalesce(F.col("usda_clean"), F.lit(""))
        ).alias("usda_chars"),
    ).repartition(4)

    print("\n=== EXPORTING IDENTICAL DATA ===")

    write_times = {}

    for fmt, path in [
        ("csv", CSV_DIR),
        ("parquet", PARQUET_DIR)
    ]:
        start = time.perf_counter()

        if fmt == "csv":
            (
                flat.write
                .mode("overwrite")
                .option("header", True)
                .csv(str(path))
            )
        else:
            (
                flat.write
                .mode("overwrite")
                .option("compression", "snappy")
                .parquet(str(path))
            )

        write_times[fmt] = round(
            time.perf_counter() - start, 3
        )

        print(
            f"{fmt.upper()} export complete: "
            f"{write_times[fmt]} seconds"
        )

    sizes = {
        "csv": folder_bytes(CSV_DIR),
        "parquet": folder_bytes(PARQUET_DIR)
    }

    print("\n=== STORAGE COMPARISON ===")

    for fmt in ("csv", "parquet"):
        print(
            f"{fmt.upper()}: "
            f"{sizes[fmt] / (1024 ** 2):.2f} MiB"
        )

    # Warm-up runs. These are NOT recorded as trials.
    print("\n=== WARM-UP (NOT TIMED IN RESULTS) ===")

    _, csv_expected = query("csv")
    _, parquet_expected = query("parquet")

    if csv_expected != parquet_expected:
        raise ValueError(
            "CSV and Parquet query results do not agree!"
        )

    total = sum(r[1] for r in csv_expected)

    print("Query results match. Total products:", total)

    if total != 368055:
        raise ValueError(
            f"Unexpected total records: {total}"
        )

    print("\n=== THREE TIMED TRIALS PER FORMAT ===")

    trials = {"csv": [], "parquet": []}

    for trial in range(3):
        # Alternate order to reduce systematic bias.
        order = (
            ("csv", "parquet")
            if trial % 2 == 0
            else ("parquet", "csv")
        )

        for fmt in order:
            duration, rows = query(fmt)

            if rows != csv_expected:
                raise ValueError(
                    f"Unexpected query result for {fmt}"
                )

            trials[fmt].append(round(duration, 3))

            print(
                f"Trial {trial + 1} — "
                f"{fmt.upper()}: {duration:.3f} seconds"
            )

    print("\n=== FINAL PERFORMANCE SUMMARY ===")

    for fmt in ("csv", "parquet"):
        median = statistics.median(trials[fmt])

        print(
            f"{fmt.upper()}: "
            f"size={sizes[fmt] / (1024 ** 2):.2f} MiB, "
            f"median query={median:.3f}s, "
            f"trials={trials[fmt]}"
        )

    report = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "spark_version": spark.version,
        "input_dataset": SOURCE,
        "row_count": total,
        "columns": SCHEMA.fieldNames(),
        "spark_master": "local[1]",
        "driver_memory": "2g",
        "export_seconds_including_source_read": write_times,
        "storage_bytes": sizes,
        "query_seconds": trials,
        "median_query_seconds": {
            fmt: round(statistics.median(trials[fmt]), 3)
            for fmt in trials
        },
        "result_groups": len(csv_expected),
        "results_match": True,
        "note": (
            "Local warm-up benchmark without Spark "
            "DataFrame caching. CSV uncompressed; "
            "Parquet uses Snappy compression."
        )
    }

    RESULT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nResults saved:", RESULT_FILE)

finally:
    spark.stop()
