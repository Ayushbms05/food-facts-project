
import json
import statistics
import time
from pathlib import Path
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

SOURCE = "data/curated/source_aware_products.parquet"
OUTPUT = Path("data/benchmarks/step38")
PARTITIONS = (1, 2, 4, 8)

OUTPUT.mkdir(parents=True, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("Spark_Partition_Benchmark")
    .master("local[2]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.sql.adaptive.enabled", "false")
    .config("spark.sql.files.maxPartitionBytes", "32m")
    .config("spark.sql.parquet.columnarReaderBatchSize", "256")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


def folder_size(path):
    return sum(
        file.stat().st_size
        for file in path.rglob("*.parquet")
    )


def query(path):
    start = time.perf_counter()

    df = spark.read.parquet(str(path))

    result = df.groupBy("agreement_category").agg(
        F.count("*").alias("records"),
        F.sum(
            F.length(
                F.coalesce(
                    F.col("off_clean"), F.lit("")
                )
            )
        ).alias("off_characters"),
        F.sum(
            F.length(
                F.coalesce(
                    F.col("usda_clean"), F.lit("")
                )
            )
        ).alias("usda_characters"),
    ).collect()

    elapsed = time.perf_counter() - start

    values = sorted(
        (
            row["agreement_category"] or "",
            row["records"],
            row["off_characters"],
            row["usda_characters"]
        )
        for row in result
    )

    return elapsed, values


try:
    source = spark.read.parquet(SOURCE).select(
        "gtin14",
        "nova_group",
        "agreement_category",
        "off_clean",
        "usda_clean"
    )

    results = {}

    print("\n=== PARTITIONED DATA EXPORT ===")

    for n in PARTITIONS:
        destination = OUTPUT / f"products_p{n}"

        start = time.perf_counter()

        (
            source.repartition(n, "gtin14")
            .write
            .mode("overwrite")
            .option("compression", "snappy")
            .parquet(str(destination))
        )

        write_seconds = round(
            time.perf_counter() - start, 3
        )

        files = list(destination.glob("part-*.parquet"))
        size_mib = folder_size(destination) / (1024 ** 2)

        results[n] = {
            "partitions": n,
            "files": len(files),
            "size_mib": round(size_mib, 3),
            "write_seconds": write_seconds,
            "query_trials": []
        }

        print(
            f"Partitions={n}, "
            f"files={len(files)}, "
            f"size={size_mib:.2f} MiB, "
            f"write={write_seconds:.3f}s"
        )

        if len(files) != n:
            raise ValueError(
                f"Expected {n} Parquet files, "
                f"found {len(files)}"
            )

    print("\n=== QUERY WARM-UP ===")

    expected = None

    for n in PARTITIONS:
        _, values = query(OUTPUT / f"products_p{n}")

        if expected is None:
            expected = values
        elif values != expected:
            raise ValueError(
                f"Query results differ for {n} partitions."
            )

        print(f"{n} partitions: verified")

    total = sum(item[1] for item in expected)

    print("Total products:", total)

    if total != 368055:
        raise ValueError(
            f"Unexpected product count: {total}"
        )

    print("\n=== QUERY TIMING TRIALS ===")

    for trial in range(3):
        order = (
            PARTITIONS if trial % 2 == 0
            else tuple(reversed(PARTITIONS))
        )

        for n in order:
            duration, values = query(
                OUTPUT / f"products_p{n}"
            )

            if values != expected:
                raise ValueError(
                    f"Query mismatch for {n} partitions."
                )

            results[n]["query_trials"].append(
                round(duration, 3)
            )

            print(
                f"Trial {trial + 1}, "
                f"partitions={n}: {duration:.3f}s"
            )

    print("\n=== FINAL PARTITION BENCHMARK ===")

    for n in PARTITIONS:
        record = results[n]

        median = statistics.median(
            record["query_trials"]
        )

        record["median_query_seconds"] = round(
            median, 3
        )

        print(
            f"Partitions={n} | "
            f"files={record['files']} | "
            f"size={record['size_mib']:.2f} MiB | "
            f"write={record['write_seconds']:.3f}s | "
            f"median_query={median:.3f}s | "
            f"trials={record['query_trials']}"
        )

    report = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "spark_version": spark.version,
        "input": SOURCE,
        "records": total,
        "master": "local[2]",
        "driver_memory": "2g",
        "compression": "snappy",
        "results": [results[n] for n in PARTITIONS]
    }

    output_file = OUTPUT / "partition_results.json"

    output_file.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nResults saved:", output_file)

finally:
    spark.stop()
