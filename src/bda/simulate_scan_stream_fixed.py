"""Step 41: validated, simulated food-scan events with Spark Structured Streaming.

Run from the food-facts-project root:
    python src/bda/simulate_scan_stream_fixed.py

Requires an existing data/curated/source_aware_products.parquet dataset.
Writes only to data/streaming/step41/ and never overwrites an existing run.
All events are synthetic, not actual mobile-app activity.
"""

import json
import random
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructField, StructType, StringType

SOURCE = "data/curated/source_aware_products.parquet"
ROOT = Path("data/streaming/step41")
INCOMING = ROOT / "incoming"
ENRICHED = ROOT / "enriched_events.parquet"
CHECKPOINT = ROOT / "checkpoint"
PROGRESS_FILE = ROOT / "microbatch_progress.json"

COUNTS = {
    "known_with_nova": 210,
    "known_without_nova": 60,
    "unknown_barcode": 30,
}
RNG = random.Random(20261122)


def gs1_check_digit(first_13_digits):
    """Calculate the valid final digit for a 14-digit GTIN."""
    weighted_sum = sum(
        int(digit) * (3 if index % 2 == 0 else 1)
        for index, digit in enumerate(reversed(first_13_digits))
    )
    return str((-weighted_sum) % 10)


def pick_absent_barcodes(products, n=10):
    """Generate checksum-valid GTIN-14s and verify absence in the product data."""
    proposed = set()
    while len(proposed) < 80:
        body = "99" + f"{RNG.randrange(10**11):011d}"
        proposed.add(body + gs1_check_digit(body))

    existing = {
        row["gtin14"]
        for row in (
            products.filter(F.col("gtin14").isin(sorted(proposed)))
            .select("gtin14")
            .collect()
        )
    }
    absent = sorted(proposed - existing)
    if len(absent) < n:
        raise RuntimeError("Could not generate enough absent barcodes")
    chosen = absent[:n]

    # This assertion is deliberate: a collision invalidated Step 39's test.
    if products.filter(F.col("gtin14").isin(chosen)).limit(1).count():
        raise AssertionError("An unknown test barcode exists in the database")
    return chosen


def sample_known_codes(products, condition, seed, label):
    codes = [
        row["gtin14"]
        for row in (
            products.filter(condition)
            .select("gtin14")
            .orderBy(F.rand(seed=seed))
            .limit(50)
            .collect()
        )
    ]
    if not codes:
        raise RuntimeError(f"No products found for {label}")
    return codes


def main():
    if not Path(SOURCE).exists():
        raise SystemExit(f"Missing curated dataset: {SOURCE}")
    if ROOT.exists() and any(ROOT.iterdir()):
        raise SystemExit(
            f"{ROOT} already contains output. This script will not erase "
            "a prior run or its checkpoint. Inspect that folder before rerunning."
        )

    spark = (
        SparkSession.builder
        .appName("Food_Scan_Stream_Validated_Simulation_Step41")
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
        valid_nova = F.coalesce(
            F.col("nova_group").between(1, 4), F.lit(False)
        )
        present_codes = sample_known_codes(
            products, valid_nova, 42, "known_with_nova"
        )
        no_nova_codes = sample_known_codes(
            products, ~valid_nova, 43, "known_without_nova"
        )
        missing_codes = pick_absent_barcodes(products)

        print("\n=== VERIFIED SIMULATED INPUT ===")
        print("Events requested:", COUNTS)
        print("Verified-absent GTINs in sampling pool:", len(missing_codes))
        print("Source label: simulated (NOT real mobile activity)")

        events = []
        base_time = datetime.now(timezone.utc).replace(microsecond=0)
        for expected_category, size in COUNTS.items():
            codes = {
                "known_with_nova": present_codes,
                "known_without_nova": no_nova_codes,
                "unknown_barcode": missing_codes,
            }[expected_category]
            for _ in range(size):
                events.append({
                    "barcode": RNG.choice(codes),
                    "expected_category": expected_category,
                    "source_type": "simulated",
                })
        RNG.shuffle(events)

        for i, event in enumerate(events):
            event["event_id"] = f"SIM-FIX-{i + 1:06d}"
            event["event_time"] = (
                base_time + timedelta(seconds=i)
            ).isoformat()

        INCOMING.mkdir(parents=True, exist_ok=True)
        for batch_number in range(3):
            start = batch_number * 100
            path = INCOMING / f"batch_{batch_number + 1:03d}.json"
            with path.open("w", encoding="utf-8") as handle:
                for event in events[start:start + 100]:
                    handle.write(json.dumps(event) + "\n")

        input_schema = StructType([
            StructField("event_id", StringType(), False),
            StructField("event_time", StringType(), False),
            StructField("barcode", StringType(), False),
            StructField("source_type", StringType(), False),
            StructField("expected_category", StringType(), False),
        ])

        stream = (
            spark.readStream.schema(input_schema)
            .option("maxFilesPerTrigger", 1)
            .json(str(INCOMING))
        )
        lookup = F.broadcast(products.select(
            F.col("gtin14").alias("lookup_barcode"),
            F.col("nova_group").alias("lookup_nova"),
        ))
        enriched = stream.join(
            lookup, stream.barcode == lookup.lookup_barcode, "left"
        )
        nova_available = F.coalesce(
            F.col("lookup_nova").between(1, 4), F.lit(False)
        )
        enriched = (
            enriched.withColumn(
                "barcode_found", F.col("lookup_barcode").isNotNull()
            )
            .withColumn("nova_available", nova_available)
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

        def save_microbatch(batch_df, batch_id):
            started = time.perf_counter()
            cached = batch_df.persist()
            try:
                rows = cached.count()
                if rows:
                    cached.write.mode("append").parquet(str(ENRICHED))
            finally:
                cached.unpersist()
            elapsed = max(time.perf_counter() - started, 0.001)
            record = {
                "batch_id": int(batch_id),
                "input_rows": rows,
                "elapsed_seconds": round(elapsed, 3),
                "processed_rows_per_second": round(rows / elapsed, 2),
            }
            progress.append(record)
            print(
                f"Batch {batch_id}: input_rows={rows}, "
                f"processed_rows_per_second={record['processed_rows_per_second']}"
            )

        print("\n=== MICRO-BATCH PROGRESS ===")
        query = (
            enriched.writeStream
            .foreachBatch(save_microbatch)
            .option("checkpointLocation", str(CHECKPOINT))
            .trigger(availableNow=True)
            .start()
        )
        query.awaitTermination()
        PROGRESS_FILE.write_text(
            json.dumps(progress, indent=2), encoding="utf-8"
        )

        saved = spark.read.parquet(str(ENRICHED))
        categories = {
            row["actual_category"]: row["count"]
            for row in saved.groupBy("actual_category").count().collect()
        }
        counts = saved.agg(
            F.count("*").alias("total_events"),
            F.countDistinct("event_id").alias("unique_events"),
            F.sum((~F.col("classification_matches")).cast("int"))
             .alias("classification_mismatches"),
            F.sum((F.col("source_type") == "simulated").cast("int"))
             .alias("simulated_events"),
            F.sum(F.col("parsed_event_time").isNull().cast("int"))
             .alias("invalid_timestamps"),
        ).first().asDict()

        print("\n=== CORRECTED STREAMING CATEGORIES ===")
        for category in COUNTS:
            print(f"{category}: {categories.get(category, 0)}")

        print("\n=== VALIDATION ===")
        for name, value in counts.items():
            print(f"{name}: {value}")
        print("microbatches:", len(progress))

        assert counts["total_events"] == 300, counts
        assert counts["unique_events"] == 300, counts
        assert counts["classification_mismatches"] == 0, counts
        assert counts["simulated_events"] == 300, counts
        assert counts["invalid_timestamps"] == 0, counts
        assert all(categories.get(k, 0) == v for k, v in COUNTS.items())
        assert sum(x["input_rows"] for x in progress) == 300

        print("\nAll streaming checks passed.")
        print("Saved output:", ENRICHED)
        print("Saved checkpoint:", CHECKPOINT)
        print("Saved progress:", PROGRESS_FILE)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
