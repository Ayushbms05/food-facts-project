"""I8.3.5a: load the verified real-data pilot Parquet into isolated PostgreSQL tables.
No full catalogue load, source modification, or model inference.
"""
import json
import re
import time
from pathlib import Path

from pyspark.sql import SparkSession, functions as F, types as T

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / 'data/backend/catalogue_pilot'
REPORT = ROOT / 'data/backend/i8_3_5a_jdbc_pilot_report.json'
URL = 'jdbc:postgresql://127.0.0.1:5432/foodfacts'
SCHEMA = 'catalogue_pilot'
DRIVER = 'org.postgresql:postgresql:42.7.14'
EXPECTED = {'off_products': 9993, 'curated_links': 8352}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    require(not REPORT.exists(), 'Pilot report already exists. Do not rerun without reviewing outputs.')
    source_report = json.loads((PILOT / 'i8_3_4a_pilot_report.json').read_text())
    require(all(source_report['output_rows_verified'][k] == v for k, v in EXPECTED.items()), 'Unexpected pilot report counts')
    passwords = re.findall(r'^POSTGRES_PASSWORD=([0-9a-f]{48})$', (ROOT / 'infra/.env').read_text(), re.M)
    require(len(passwords) == 1, 'Expected original generated password format; do not share it.')
    props = {'user': 'food_admin', 'password': passwords[0], 'driver': 'org.postgresql.Driver'}
    spark = (SparkSession.builder.appName('FoodFactsRealJDBCPilot').master('local[2]')
             .config('spark.driver.memory', '2g')
             .config('spark.sql.shuffle.partitions', '4')
             .config('spark.jars.packages', DRIVER)
             .config('spark.ui.enabled', 'false').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    try:
        def query(sql):
            return spark.read.jdbc(URL, f'({sql}) AS q', properties=props)
        schemas = query("SELECT schema_name FROM information_schema.schemata WHERE schema_name='catalogue_pilot'")
        require(schemas.count() == 1, 'Create catalogue_pilot schema first.')
        tables = query("SELECT table_name FROM information_schema.tables WHERE table_schema='catalogue_pilot'")
        require(tables.count() == 0, 'Pilot schema is not empty. Inspect it; do not overwrite existing tables.')
        frames = {}
        nul_audit = {}
        print('PREFLIGHT: checking staged rows, schemas, keys and PostgreSQL text compatibility', flush=True)
        for name, expected in EXPECTED.items():
            df = spark.read.parquet(str(PILOT/name))
            df.printSchema()
            require(all(isinstance(f.dataType, (T.StringType, T.IntegerType, T.LongType, T.BooleanType)) for f in df.schema), f'{name}: unexpected field type')
            summary = df.agg(F.count('*').alias('rows'), F.countDistinct('gtin14').alias('keys')).first()
            require(summary.rows == summary.keys == expected, f'{name}: unexpected row/key counts')
            strings = [f.name for f in df.schema if isinstance(f.dataType, T.StringType)]
            # PostgreSQL TEXT cannot contain a literal NUL byte. Do not silently remove it.
            audit = df.agg(*[F.count(F.when(F.instr(F.col(c), '\x00') > 0, 1)).alias(c) for c in strings]).first().asDict()
            nul_audit[name] = {k:v for k,v in audit.items() if v}
            frames[name] = df
        print('NUL audit (nonzero column counts):', json.dumps(nul_audit), flush=True)
        require(not any(nul_audit.values()), 'Literal NUL detected; no database writes attempted. Review text policy first.')
        require(frames['curated_links'].select('gtin14').join(frames['off_products'].select('gtin14'), 'gtin14', 'left_anti').count() == 0, 'Unmatched pilot curated key')
        results = {}
        for name, df in frames.items():
            print(f'LOAD: {SCHEMA}.{name}', flush=True)
            start = time.perf_counter()
            df.coalesce(2).write.jdbc(URL, f'{SCHEMA}.{name}', mode='errorifexists',
                                    properties={**props, 'batchsize':'1000'})
            elapsed = time.perf_counter() - start
            actual = spark.read.jdbc(URL, f'{SCHEMA}.{name}', properties=props).select(*df.columns)
            require(actual.schema.simpleString() == df.schema.simpleString(), f'{name}: JDBC type mismatch')
            n = actual.count()
            missing = df.exceptAll(actual).count()
            extra = actual.exceptAll(df).count()
            require(n == EXPECTED[name] and missing == extra == 0, f'{name}: round-trip values mismatch')
            results[name] = {'rows':n, 'missing_rows':missing, 'extra_rows':extra, 'write_seconds':elapsed, 'columns':df.columns}
            print(f'VERIFIED: {n:,} rows and all values match', flush=True)
        report = {'experiment':'I8.3.5a','spark_version':spark.version,'jdbc_driver':DRIVER,
                  'schema':SCHEMA, 'batchsize':1000, 'maximum_write_partitions':2,
                  'literal_nul_columns_nonzero':nul_audit, 'tables':results,
                  'limitations':['Pilot slice only; full data not loaded.', 'Spark-created pilot tables have no production keys/indexes.', 'Write times exclude verification and are not controlled benchmarks.', 'Failure can leave partial isolated tables; inspect before retry.']}
        REPORT.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report,indent=2))
        print('SUCCESS: real-data JDBC pilot verified; full catalogue not loaded.')
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
