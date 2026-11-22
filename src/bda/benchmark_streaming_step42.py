"""Step 42: Benchmark Spark Structured Streaming file-microbatch sizes.

Run from food-facts-project root:
    python src/bda/benchmark_streaming_step42.py

Uses only simulated scan events; no real app activity. The SAME 3,000
synthetic events are replayed in two separate experiments:
  A: 30 JSONL files * 100 events  (approximately 30 microbatches)
  B:  6 JSONL files * 500 events  (approximately  6 microbatches)

Requires data/curated/source_aware_products.parquet and PySpark.
Will not overwrite data/streaming/step41 or an existing step42 run.
"""

import json
import random
import statistics
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

SOURCE = "data/curated/source_aware_products.parquet"
ROOT = Path("data/streaming/step42")
TOTAL_EVENTS = 3000
CATEGORY_COUNTS = {
    "known_with_nova": 2100,
    "known_without_nova": 600,
    "unknown_barcode": 300,
}
SCENARIOS = (
    ("100_per_file", 100),
    ("500_per_file", 500),
)
RNG = random.Random(20261122)


def gs1_check_digit(first_13):
    weighted = sum(
        int(d) * (3 if index % 2 == 0 else 1)
        for index, d in enumerate(reversed(first_13))
    )
    return str((-weighted) % 10)


def generate_unknown_codes(products, count=20):
    proposed = set()
    while len(proposed) < 100:
        body = "99" + f"{RNG.randrange(10**11):011d}"
        proposed.add(body + gs1_check_digit(body))

    found = {
        row["gtin14"]
        for row in (
            products.filter(F.col("gtin14").isin(sorted(proposed)))
            .select("gtin14")
            .collect()
        )
    }
    absent = sorted(proposed - found)[:count]
    if len(absent) < count:
        raise ValueError("Unable to find enough absent GTINs")

    if products.filter(F.col("gtin14").isin(absent)).limit(1).count():
        raise AssertionError("An intended unknown GTIN exists in product data")
    return absent


def sample_codes(products, condition, seed, label):
    selected = [
        row["gtin14"]
        for row in (
            products.filter(condition)
            .select("gtin14")
            .orderBy(F.rand(seed=seed))
            .limit(60)
            .collect()
        )
    ]
    if not selected:
        raise ValueError(f"No available barcodes in category: {label}")
    return selected


def write_events(events, directory, file_size):
    directory.mkdir(parents=True, exist_ok=False)
    for begin in range(0, len(events), file_size):
        number = begin // file_size + 1
        path = directory / f"batch_{number:03d}.json"
        with path.open("w", encoding="utf-8") as handle:
            for event in events[begin:begin + file_size]:
                handle.write(json.dumps(event) + "\n")
    return len(list(directory.glob("batch_*.json")))


def main():
    if not Path(SOURCE).exists():
        raise SystemExit(f"Missing curated dataset: {SOURCE}")
    if ROOT.exists():
        raise SystemExit(
            f"{ROOT} already exists. To preserve reproducibility, this script "
            "will not overwrite previous Step 42 results."
        )

    spark = (
        SparkSession.builder
        .appName("BDA_Step42_Streaming_Microbatch_Benchmark")
        .master("local[2]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.files.maxPartitionBytes", "32m")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        products = (
            spark.read.parquet(SOURCE)
            .select("gtin14", "nova_group")
            .dropDuplicates(["gtin14"])
        )
        valid_nova = F.coalesce(F.col("nova_group").between(1, 4), F.lit(False))
        code_pools = {
            "known_with_nova": sample_codes(products, valid_nova, 42, "known_with_nova"),
            "known_without_nova": sample_codes(products, ~valid_nova, 43, "known_without_nova"),
            "unknown_barcode": generate_unknown_codes(products),
        }

        events = []
        for category, count in CATEGORY_COUNTS.items():
            for _ in range(count):
                events.append({
                    "barcode": RNG.choice(code_pools[category]),
                    "expected_category": category,
                    "source_type": "simulated",
                })
        RNG.shuffle(events)
        base_time = datetime.now(timezone.utc).replace(microsecond=0)
        for index, event in enumerate(events):
            event["event_id"] = f"SIM-42-{index + 1:07d}"
            event["event_time"] = (
                base_time + timedelta(seconds=index)
            ).isoformat()

        assert len(events) == TOTAL_EVENTS
        assert Counter(x["expected_category"] for x in events) == CATEGORY_COUNTS
        print("\n=== VERIFIED INPUT ===")
        print("Simulated events:", len(events))
        print("Expected categories:", CATEGORY_COUNTS)
        print("Verified absent GTINs in pool:", len(code_pools["unknown_barcode"]))
        print("This is synthetic traffic, NOT real mobile-app usage.")

        ROOT.mkdir(parents=True, exist_ok=False)
        schema = StructType([
            StructField("event_id", StringType(), False),
            StructField("event_time", StringType(), False),
            StructField("barcode", StringType(), False),
            StructField("source_type", StringType(), False),
            StructField("expected_category", StringType(), False),
        ])
        lookup = F.broadcast(products.select(
            F.col("gtin14").alias("lookup_barcode"),
            F.col("nova_group").alias("lookup_nova"),
        ))

        reports = []
        for name, per_file in SCENARIOS:
            case_root = ROOT / name
            input_dir = case_root / "incoming"
            output = case_root / "enriched_events.parquet"
            checkpoint = case_root / "checkpoint"
            progress_file = case_root / "microbatch_progress.json"
            file_count = write_events(events, input_dir, per_file)

            stream = (
                spark.readStream.schema(schema)
                .option("maxFilesPerTrigger", 1)
                .json(str(input_dir))
            )
            enriched = stream.join(
                lookup, stream.barcode == lookup.lookup_barcode, "left"
            )
            found = F.col("lookup_barcode").isNotNull()
            has_nova = F.coalesce(F.col("lookup_nova").between(1, 4), F.lit(False))
            enriched = (
                enriched.withColumn("barcode_found", found)
                .withColumn("nova_available", has_nova)
                .withColumn(
                    "actual_category",
                    F.when(~F.col("barcode_found"), "unknown_barcode")
                    .when(F.col("nova_available"), "known_with_nova")
                    .otherwise("known_without_nova"),
                )
                .withColumn(
                    "classification_matches",
                    F.col("actual_category") == F.col("expected_category"),
                )
                .withColumn("parsed_event_time", F.to_timestamp("event_time"))
            )

            progress = []

            def save_batch(batch_df, batch_id):
                started = time.perf_counter()
                cached = batch_df.persist()
                try:
                    rows = cached.count()
                    if rows:
                        cached.write.mode("append").parquet(str(output))
                finally:
                    cached.unpersist()
                elapsed = max(time.perf_counter() - started, 0.001)
                progress.append({
                    "batch_id": int(batch_id),
                    "input_rows": int(rows),
                    "foreach_batch_seconds": round(elapsed, 3),
                    "foreach_batch_rows_per_second": round(rows / elapsed, 2),
                })

            print(f"\n=== STREAMING SCENARIO: {name} ===")
            print(f"Files: {file_count}; events/file: {per_file}")
            wall_start = time.perf_counter()
            streaming_query = (
                enriched.writeStream.foreachBatch(save_batch)
                .option("checkpointLocation", str(checkpoint))
                .trigger(availableNow=True)
                .start()
            )
            streaming_query.awaitTermination()
            wall_elapsed = max(time.perf_counter() - wall_start, 0.001)
            progress_file.write_text(json.dumps(progress, indent=2), encoding="utf-8")

            saved = spark.read.parquet(str(output))
            counts = saved.agg(
                F.count("*").alias("total_events"),
                F.countDistinct("event_id").alias("unique_events"),
                F.sum((~F.col("classification_matches")).cast("long"))
                .alias("classification_mismatches"),
                F.sum((F.col("source_type") == "simulated").cast("long"))
                .alias("simulated_events"),
                F.sum(F.col("parsed_event_time").isNull().cast("long"))
                .alias("invalid_timestamps"),
            ).first().asDict()
            observed = {
                row["actual_category"]: row["count"]
                for row in saved.groupBy("actual_category").count().collect()
            }

            if counts != {
                "total_events": TOTAL_EVENTS,
                "unique_events": TOTAL_EVENTS,
                "classification_mismatches": 0,
                "simulated_events": TOTAL_EVENTS,
                "invalid_timestamps": 0,
            }:
                raise AssertionError(f"Validation failed in {name}: {counts}")
            if observed != CATEGORY_COUNTS:
                raise AssertionError(f"Category counts differ in {name}: {observed}")
            if sum(p["input_rows"] for p in progress) != TOTAL_EVENTS:
                raise AssertionError(f"Microbatch record counts differ in {name}")

            rates = [p["foreach_batch_rows_per_second"] for p in progress]
            report = {
                "scenario": name,
                "events_per_file": per_file,
                "input_files": file_count,
                "microbatches": len(progress),
                "total_events": counts["total_events"],
                "classification_mismatches": counts["classification_mismatches"],
                "wall_seconds": round(wall_elapsed, 3),
                "overall_stream_rows_per_second": round(TOTAL_EVENTS / wall_elapsed, 2),
                "median_foreach_batch_rows_per_second": round(statistics.median(rates), 2),
                "category_counts": observed,
                "json_progress_path": str(progress_file),
                "parquet_output_path": str(output),
            }
            reports.append(report)

            print("\n=== SCENARIO VALIDATION ===")
            print("Total events:", report["total_events"])
            print("Unique IDs:", counts["unique_events"])
            print("Classification mismatches:", report["classification_mismatches"])
            print("Categories:", observed)
            print("Microbatches:", report["microbatches"])
            print("Elapsed streaming wall time:", report["wall_seconds"], "s")
            print("Overall throughput:", report["overall_stream_rows_per_second"], "rows/s")
            print("Median foreachBatch rate:", report["median_foreach_batch_rows_per_second"], "rows/s")

        report_path = ROOT / "step42_results.json"
        report_path.write_text(json.dumps({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "spark_version": spark.version,
            "master": "local[2]",
            "driver_memory": "2g",
            "source": SOURCE,
            "event_provenance": "fully simulated; not real mobile scans",
            "category_targets": CATEGORY_COUNTS,
            "benchmark_note": (
                "Both experiments replay the same 3000 events. Processing time includes "
                "stream start, microbatch scheduling, join and Parquet writes, but not "
                "generation of input files or final validation. One file per microbatch "
                "via maxFilesPerTrigger=1. Local laptop only."
            ),
            "scenarios": reports,
        }, indent=2), encoding="utf-8")

        print("\n=== STEP 42 COMPARISON ===")
        for report in reports:
            print(
                f"{report['scenario']}: files={report['input_files']}, "
                f"microbatches={report['microbatches']}, "
                f"wall_time={report['wall_seconds']}s, "
                f"overall_rate={report['overall_stream_rows_per_second']} rows/s, "
                f"mismatches={report['classification_mismatches']}"
            )
        print("\nAll Step 42 validations passed.")
        print("Results saved:", report_path)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
