"""I8.3.5b: full staged Parquet -> PostgreSQL via Spark JDBC.
Run from project root using BDA .venv and Java 17. No model inference.
A failed load leaves catalogue_loading isolated; never drop it automatically.
"""
import argparse
import hashlib
from collections import Counter
from functools import reduce
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from pyspark.sql import SparkSession, functions as F, types as T

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/backend/catalogue_full'
REPORT = ROOT / 'data/backend/i8_3_5b_full_jdbc_report.json'
PREFLIGHT = ROOT / 'data/backend/i8_3_5b_preflight_v2.json'
REPAIR_AUDIT = ROOT / 'data/backend/i8_3_5b_nul_corrections.json'
OBSERVED_NUL = {'off_products': {'product_name': 3, 'brands': 2, 'ingredients_raw': 7, 'ingredients_en': 5},
                'curated_links': {}, 'barcode_conflicts': {}}
URL = 'jdbc:postgresql://127.0.0.1:5432/foodfacts'
DRIVER = 'org.postgresql:postgresql:42.7.14'
EXPECTED = {'off_products': 4693098, 'curated_links': 368054, 'barcode_conflicts': 47}
STAGE = 'catalogue_loading'
PG_TYPES = {T.StringType: 'TEXT', T.IntegerType: 'INTEGER', T.LongType: 'BIGINT', T.BooleanType: 'BOOLEAN'}
COMPOSE = ['docker', 'compose', '-f', str(ROOT / 'infra/compose.yaml')]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def psql(sql):
    # Local container connection; password is never placed on a command line.
    cmd = COMPOSE + ['exec', '-T', 'postgres', 'psql', '-X', '-qAt', '-U', 'food_admin', '-d', 'foodfacts', '-v', 'ON_ERROR_STOP=1']
    result = subprocess.run(cmd, input=sql, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"PostgreSQL SQL command failed: {result.stderr.strip()}")
    return result.stdout.strip()


def docker_free_bytes():
    output = subprocess.check_output(COMPOSE + ['exec', '-T', 'postgres', 'df', '-Pk', '/var/lib/postgresql/data'], text=True)
    return int(output.strip().splitlines()[-1].split()[3]) * 1024


def ddl(name, df):
    fields = ['record_id BIGINT GENERATED ALWAYS AS IDENTITY'] if name == 'off_products' else []
    for field in df.schema:
        require(re.fullmatch(r'[a-z][a-z0-9_]*', field.name) is not None, 'Unexpected field name')
        require(type(field.dataType) in PG_TYPES, f'Unsupported type: {field.name}')
        col = f'"{field.name}" {PG_TYPES[type(field.dataType)]}'
        if field.name == 'gtin14':
            col += " NOT NULL CHECK (length(gtin14)=14 AND gtin14 !~ '[^0-9]')"
        if name == 'barcode_conflicts' and field.name == 'record_count':
            col += ' NOT NULL CHECK (record_count > 1)'
        fields.append(col)
    return f'CREATE TABLE {STAGE}.{name} (' + ', '.join(fields) + ');'


def digest_rows(df, columns, sign):
    # Fixed field order, explicit nulls; compare a multiset of SHA-256 row digests.
    # Unlike collecting rows, only the aggregate mismatch count returns to Python.
    return df.select(F.sha2(F.to_json(F.struct(*[F.col(c) for c in columns]),
                                      {'ignoreNullFields': 'false'}), 256).alias('digest'),
                     F.lit(sign).cast('long').alias('delta'))


def compare(source, actual):
    require(source.schema.simpleString() == actual.schema.simpleString(), 'JDBC read-back type mismatch')
    deltas = (digest_rows(source, source.columns, 1)
              .unionByName(digest_rows(actual, source.columns, -1))
              .groupBy('digest').agg(F.sum('delta').alias('delta')))
    row = deltas.agg(
        F.coalesce(F.sum(F.greatest('delta', F.lit(0))), F.lit(0)).alias('missing'),
        F.coalesce(F.sum(F.greatest(-F.col('delta'), F.lit(0))), F.lit(0)).alias('extra')).first()
    return int(row.missing), int(row.extra)



def replace_nul(column):
    return F.replace(column, F.lit('\x00'), F.lit(' '))


def test_replacement(spark):
    values = [('a\x00b',), ('\x00\x00',), ('milk',), ('café',), (None,), (r'\u0000',)]
    df = spark.createDataFrame([(i, v[0]) for i,v in enumerate(values)], 'i int, value string')
    got = [r.fixed for r in df.select('i', replace_nul(F.col('value')).alias('fixed')).orderBy('i').collect()]
    require(got == ['a b', '  ', 'milk', 'café', None, r'\u0000'], 'NUL replacement self-test failed')
    print('SUCCESS: literal NUL replacement self-test passed.', flush=True)


def repair_off(frames, audit):
    # Only this known snapshot and the four observed text columns are authorized.
    require(audit == OBSERVED_NUL, 'NUL counts differ from reported snapshot; stop and review.')
    source = frames['off_products']
    fields = sorted(audit['off_products'])
    flag = reduce(lambda a,b: a | b, [F.instr(F.col(c), '\x00') > 0 for c in fields])
    # Bounded collection: at most 17 affected source rows, plus one overflow guard.
    rows = source.filter(flag).select('gtin14','off_code', *fields,
        *[replace_nul(F.col(c)).alias('fixed_'+c) for c in fields]).limit(18).collect()
    require(0 < len(rows) <= 17, 'Unexpected number of affected source rows')
    changes = []
    counts = Counter()
    for r in rows:
        for c in fields:
            before, after = r[c], r['fixed_'+c]
            expected = None if before is None else before.replace('\x00', ' ')
            require(after == expected, 'Spark/Python replacement disagreement')
            if before != after:
                counts[c] += 1
                changes.append({'gtin14': r.gtin14, 'off_code': r.off_code,
                                'field': c, 'before': before, 'after': after,
                                'nul_characters_replaced': before.count('\x00')})
    require(dict(counts) == OBSERVED_NUL['off_products'], 'Correction audit counts do not match')
    keys = sorted({r.gtin14 for r in rows})
    curated_keys = sorted(r.gtin14 for r in frames['curated_links'].filter(F.col('gtin14').isin(keys)).select('gtin14').collect())
    # JSON escapes literal NUL for audit readability; source Parquet remains untouched.
    changes.sort(key=lambda v: json.dumps(v, sort_keys=True, ensure_ascii=True))
    details = {'experiment':'I8.3.5b', 'policy':'replace literal U+0000 with U+0020 in four OFF text columns only',
               'affected_rows':len(rows), 'affected_distinct_gtin14':len(keys),
               'affected_gtin14':keys, 'affected_curated_gtin14':curated_keys,
               'field_values_changed':len(changes), 'nul_characters_replaced':sum(c['nul_characters_replaced'] for c in changes),
               'counts_by_column':dict(counts), 'changes':changes,
               'note':'Curated source-link flags describe original staged OFF fields; no reclassification or substitution performed.'}
    payload = json.dumps(details, indent=2, sort_keys=True, ensure_ascii=True)+'\n'
    if REPAIR_AUDIT.exists():
        require(REPAIR_AUDIT.read_text() == payload, 'Existing repair audit differs; inspect before retry.')
    else:
        with REPAIR_AUDIT.open('x', encoding='utf-8') as f:
            f.write(payload)
    cleaned = source.select(*[replace_nul(F.col(c)).alias(c) if c in fields else F.col(c) for c in source.columns])
    require(cleaned.schema.simpleString() == source.schema.simpleString(), 'Correction changed field types')
    strings = [f.name for f in cleaned.schema if isinstance(f.dataType,T.StringType)]
    post = cleaned.agg(F.count('*').alias('_rows'), *[F.count(F.when(F.instr(F.col(c),'\x00')>0,1)).alias(c) for c in strings]).first().asDict()
    require(post.pop('_rows') == EXPECTED['off_products'] and not any(post.values()), 'Post-repair row/NUL audit failed')
    frames['off_products'] = cleaned
    summary = {k:v for k,v in details.items() if k not in ('changes','affected_gtin14','affected_curated_gtin14')}
    summary.update({'audit_path':str(REPAIR_AUDIT.relative_to(ROOT)), 'audit_sha256':hashlib.sha256(REPAIR_AUDIT.read_bytes()).hexdigest(),
                    'affected_curated_keys':len(curated_keys), 'remaining_literal_nul_counts':{}})
    print('REPAIR VERIFIED: '+json.dumps(summary), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preflight-only', action='store_true', help='Audit only; no database changes')
    ap.add_argument('--repair-nul', action='store_true', help='Explicitly allow the audited four-column NUL correction for the reported snapshot')
    args = ap.parse_args()
    require(not REPORT.exists(), 'Completed report exists; refusing to repeat load.')
    source_report = json.loads((DATA / 'i8_3_4b_full_report.json').read_text())
    require(all(source_report['output_rows_verified'][k] == v for k,v in EXPECTED.items()), 'Unexpected full source report counts')
    passwords = re.findall(r'^POSTGRES_PASSWORD=([0-9a-f]{48})$', (ROOT / 'infra/.env').read_text(), re.M)
    require(len(passwords) == 1, 'Expected original generated password format; do not share credentials.')
    props = {'user':'food_admin', 'password':passwords[0], 'driver':'org.postgresql.Driver', 'fetchsize':'1000'}
    require(psql("SELECT count(*) FROM pg_namespace WHERE nspname IN ('catalogue','catalogue_loading');") == '0',
            'Target or loading schema already exists. Inspect before retry; no automatic deletion.')
    host_free = shutil.disk_usage(ROOT).free
    pg_free = docker_free_bytes()
    print(f'Free host GiB: {host_free/1024**3:.2f}; PostgreSQL filesystem GiB: {pg_free/1024**3:.2f}', flush=True)
    require(min(host_free, pg_free) >= 30*1024**3, 'Less than 30 GiB free; review storage before loading (30 GiB is a conservative check, not a size guarantee).')
    spark = (SparkSession.builder.appName('FoodFactsFullJDBCLoad').master('local[2]')
             .config('spark.driver.memory','4g').config('spark.sql.shuffle.partitions','64')
             .config('spark.sql.files.maxPartitionBytes','32m')
             .config('spark.sql.parquet.columnarReaderBatchSize','256')
             .config('spark.sql.autoBroadcastJoinThreshold','-1')
             .config('spark.sql.adaptive.autoBroadcastJoinThreshold','-1')
             .config('spark.jars.packages',DRIVER).config('spark.ui.enabled','false').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    start = time.perf_counter()
    try:
        test_replacement(spark)
        # Exercise the actual credentialed JDBC connection before any changes.
        require(spark.read.jdbc(URL, '(SELECT 1 AS ready) q', properties=props).first().ready == 1, 'JDBC connection failed')
        frames, audit = {}, {}
        print('PREFLIGHT: full row counts, scalar schemas, barcode shape and literal NUL audit', flush=True)
        for name, expected in EXPECTED.items():
            df = spark.read.parquet(str(DATA/name))
            require('gtin14' in df.columns and 'record_id' not in df.columns, f'{name}: unexpected columns')
            require(all(type(f.dataType) in PG_TYPES for f in df.schema), f'{name}: unsupported type')
            strings = [f.name for f in df.schema if isinstance(f.dataType,T.StringType)]
            k = F.col('gtin14')
            row = df.agg(F.count('*').alias('_rows'),
                         F.count(F.when(k.isNull() | (F.length(k)!=14) | k.rlike('[^0-9]'),1)).alias('_bad_key'),
                         *[F.count(F.when(F.instr(F.col(c),'\x00')>0,1)).alias(c) for c in strings]).first().asDict()
            require(row.pop('_rows') == expected and row.pop('_bad_key') == 0, f'{name}: count/key check failed')
            audit[name] = {c:v for c,v in row.items() if v}
            frames[name] = df
            print(f'{name}: {expected:,} rows; literal NUL counts: {audit[name]}', flush=True)
        preflight = {'experiment':'I8.3.5b','expected_rows':EXPECTED,'literal_nul_columns_nonzero':audit,
                     'host_free_bytes':host_free,'postgres_filesystem_free_bytes':pg_free,
                     'preflight_seconds':time.perf_counter()-start,'ready':not any(audit.values())}
        PREFLIGHT.write_text(json.dumps(preflight,indent=2)+'\n')
        if any(audit.values()) and args.repair_nul:
            preflight['nul_repair'] = repair_off(frames, audit)
            preflight['ready'] = True
            preflight['preflight_seconds'] = time.perf_counter()-start
            PREFLIGHT.write_text(json.dumps(preflight,indent=2)+'\n')
        elif args.repair_nul:
            require(False, 'Expected known NUL issue not found; review snapshot before proceeding.')
        require(preflight['ready'], 'Literal NUL detected; stopped before database writes. Use explicit --repair-nul only for the reviewed snapshot.')
        if args.preflight_only:
            print('SUCCESS: full preflight passed; no database writes.', flush=True)
            return
        print('CREATE: isolated catalogue_loading schema and explicitly typed tables', flush=True)
        psql('BEGIN; CREATE SCHEMA catalogue_loading;\n' + '\n'.join(ddl(n,df) for n,df in frames.items()) + '\nCOMMIT;')
        results = {}
        for name,df in frames.items():
            print(f'LOAD: {STAGE}.{name}', flush=True)
            began = time.perf_counter()
            # Two writers, bounded JDBC batches, no overwrite. Existing table has generated record_id for OFF.
            df.coalesce(2).write.jdbc(URL, f'{STAGE}.{name}', mode='append',
                                     properties={**props,'batchsize':'1000'})
            write_seconds = time.perf_counter()-began
            count = int(psql(f'SELECT count(*) FROM {STAGE}.{name};'))
            require(count == EXPECTED[name], f'{name}: PostgreSQL count mismatch')
            key = 'record_id' if name=='off_products' else 'gtin14'
            psql(f'ALTER TABLE {STAGE}.{name} ADD PRIMARY KEY ({key});')
            print(f'VERIFY: {name}: count {count:,}; compare SHA-256 row digest multiset', flush=True)
            began = time.perf_counter()
            if name=='off_products':
                actual = spark.read.jdbc(URL, f'{STAGE}.{name}', column='record_id', lowerBound=1,
                                         upperBound=count+1, numPartitions=2, properties=props).select(*df.columns)
            else:
                actual = spark.read.jdbc(URL, f'{STAGE}.{name}', properties=props).select(*df.columns)
            missing,extra = compare(df,actual)
            require(missing==extra==0,f'{name}: digest comparison mismatch: {missing} missing, {extra} extra')
            results[name] = {'rows':count,'missing_digest_rows':missing,'extra_digest_rows':extra,
                             'write_seconds':write_seconds,'verify_seconds':time.perf_counter()-began,
                             'primary_key':key}
            print(f'VERIFIED: {count:,} rows, zero digest multiset differences', flush=True)
        print('RECONCILE: global barcode counts and curated record links in PostgreSQL', flush=True)
        # Compare the duplicate key/count table with actual product groups in SQL.
        group_sql = '''WITH g AS (SELECT gtin14,count(*) AS n FROM catalogue_loading.off_products GROUP BY gtin14),
        d AS (SELECT gtin14,n FROM g WHERE n>1),
        mismatch AS (SELECT count(*) AS n FROM d FULL JOIN catalogue_loading.barcode_conflicts c USING(gtin14)
                     WHERE d.n IS DISTINCT FROM c.record_count),
        bad_links AS (SELECT count(*) AS n FROM catalogue_loading.curated_links c LEFT JOIN g USING(gtin14)
                      WHERE g.n IS DISTINCT FROM c.record_count)
        SELECT (SELECT count(*) FROM g), (SELECT count(*) FROM d),
        (SELECT coalesce(sum(n),0) FROM d), (SELECT n FROM mismatch), (SELECT n FROM bad_links);'''
        distinct,collisions,collision_rows,conflict_diff,bad_links = map(int,psql(group_sql).split('|'))
        require((distinct,collisions,collision_rows,conflict_diff,bad_links)==(4693051,47,94,0,0), 'Global SQL reconciliation failed')
        psql('ANALYZE catalogue_loading.off_products; ANALYZE catalogue_loading.curated_links; ANALYZE catalogue_loading.barcode_conflicts;')
        print('PUBLISH: rename validated staging schema to catalogue', flush=True)
        psql('ALTER SCHEMA catalogue_loading RENAME TO catalogue;')
        report = {**preflight,'spark_version':spark.version,'jdbc_driver':DRIVER,'schema':'catalogue',
                  'batchsize':1000,'max_write_partitions':2,'tables':results,'distinct_gtin14':distinct,
                  'collision_keys':collisions,'rows_on_collision_keys':collision_rows,
                  'conflict_count_mismatches':conflict_diff,'curated_link_count_mismatches':bad_links,
                  'elapsed_seconds':time.perf_counter()-start,
                  'limitations':['Digest equality is strong probabilistic content verification, not a mathematical exact comparison.',
                  'Record IDs are generated locally and may change on a future rebuild.',
                  'Digest comparison is against the explicitly corrected OFF serving projection, not byte-identical original Parquet.',
                  'Curated link status remains based on original staged source fields; affected keys recorded separately.',
                  'OFF barcode index deliberately deferred for I8.4 before/after benchmark.',
                  'Audit rejects and unmatched curated records remain in Parquet.',
                  'Single-machine write times; not a controlled benchmark or multi-node result.']}
        REPORT.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        print('SUCCESS: full catalogue verified and published to PostgreSQL.', flush=True)
    finally:
        spark.stop()


if __name__=='__main__':
    main()
