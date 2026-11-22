"""Step 39: Replay clearly simulated barcode scans through Spark Structured Streaming."""
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pyspark.sql import SparkSession, functions as F, types as T

ROOT = Path('data/streaming/step39')
INPUT = ROOT / 'incoming'
OUTPUT = ROOT / 'enriched_events.parquet'
CHECKPOINT = ROOT / 'checkpoint'
PRODUCTS = 'data/curated/source_aware_products.parquet'

# Never silently duplicate simulated events or reuse an old checkpoint.
if ROOT.exists():
    raise SystemExit(
        f'{ROOT} already exists. Preserve the previous experiment, '
        'or rename that folder before running again.'
    )
INPUT.mkdir(parents=True)

spark = (
    SparkSession.builder
    .appName('Food_Scan_Structured_Streaming_Pilot')
    .master('local[2]')
    .config('spark.driver.memory', '2g')
    .config('spark.sql.shuffle.partitions', '4')
    .config('spark.sql.session.timeZone', 'UTC')
    .getOrCreate()
)
spark.sparkContext.setLogLevel('ERROR')
try:
    product_lookup = (
        spark.read.parquet(PRODUCTS)
        .select(
            F.col('gtin14').alias('barcode'),
            'nova_group',
            'agreement_category'
        )
    )
    barcodes = [
        row['barcode']
        for row in product_lookup.select('barcode').limit(40).collect()
    ]
    if not barcodes:
        raise RuntimeError('No product barcodes in the curated dataset.')

    rng = random.Random(42)
    now = datetime.now(timezone.utc)
    for batch in range(1, 4):
        path = INPUT / f'batch_{batch:03d}.json'
        with path.open('w', encoding='utf-8') as f:
            for i in range(100):
                # Every tenth event intentionally uses an unknown barcode.
                barcode = (
                    '00000000000000'
                    if i % 10 == 0
                    else rng.choice(barcodes)
                )
                event = {
                    'event_id': f'SIM-{batch:03d}-{i:03d}',
                    'barcode': barcode,
                    'event_time': (
                        now - timedelta(seconds=(3 - batch) * 100 + (99 - i))
                    ).isoformat(),
                    'scan_method': 'barcode',
                    'event_source': 'simulated'
                }
                f.write(json.dumps(event) + '\n')

    print('\n=== SIMULATED INPUT ===')
    print('Batches: 3 files x 100 events')
    print('Total generated: 300')
    print('Intentionally unknown: 30')
    print('Source label: simulated (not real mobile activity)')

    schema = T.StructType([
        T.StructField('event_id', T.StringType()),
        T.StructField('barcode', T.StringType()),
        T.StructField('event_time', T.StringType()),
        T.StructField('scan_method', T.StringType()),
        T.StructField('event_source', T.StringType())
    ])
    events = (
        spark.readStream
        .schema(schema)
        .option('maxFilesPerTrigger', 1)
        .json(str(INPUT))
        .withColumn('event_timestamp', F.to_timestamp('event_time'))
    )

    # Stream-static enrichment: scan event joined to our real curated dataset.
    enriched = (
        events.join(F.broadcast(product_lookup), 'barcode', 'left')
        .withColumn('product_found', F.col('nova_group').isNotNull())
    )

    query = (
        enriched.writeStream
        .format('parquet')
        .outputMode('append')
        .option('path', str(OUTPUT))
        .option('checkpointLocation', str(CHECKPOINT))
        .trigger(availableNow=True)
        .start()
    )
    query.awaitTermination()

    print('\n=== MICRO-BATCH PROGRESS ===')
    for entry in query.recentProgress:
        print(
            f"Batch {entry['batchId']}: "
            f"input_rows={entry['numInputRows']}, "
            f"processed_rows_per_second={entry.get('processedRowsPerSecond', 0):.2f}"
        )

    results = spark.read.parquet(str(OUTPUT))
    print('\n=== STREAMING VALIDATION ===')
    summary = results.agg(
        F.count('*').alias('total_events'),
        F.countDistinct('event_id').alias('unique_events'),
        F.sum(F.col('product_found').cast('int')).alias('known_products'),
        F.sum((~F.col('product_found')).cast('int')).alias('unknown_products'),
        F.sum((F.col('event_source') == 'simulated').cast('int'))
         .alias('simulated_events'),
        F.sum(F.col('event_timestamp').isNull().cast('int'))
         .alias('invalid_event_timestamps')
    ).first()
    for name, value in summary.asDict().items():
        print(f'{name}: {value}')

    print('\n=== ENRICHED NOVA DISTRIBUTION ===')
    results.groupBy('nova_group').count().orderBy('nova_group').show()

    if (
        summary['total_events'] != 300
        or summary['unique_events'] != 300
        or summary['simulated_events'] != 300
        or summary['invalid_event_timestamps'] != 0
    ):
        raise ValueError('Streaming record validation failed.')
    print('\nSaved output:', OUTPUT)
    print('Saved checkpoint:', CHECKPOINT)
finally:
    spark.stop()