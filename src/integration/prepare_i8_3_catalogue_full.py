"""I8.3.4b: full OFF catalogue ETL, using the verified I8.3.4a helpers.
Run with Java 17 and the original Spark .venv. No JDBC writes or inference.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F
from src.integration.prepare_i8_3_catalogue_pilot import (
    RAW, CURATED, RAW_COLS, CURATED_COLS, prepare_products, group_counts,
    curated_links, counts, self_test as pilot_self_test,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/backend/catalogue_full'
WORK = OUT.with_name(OUT.name + '.inprogress')
EXPECTED_RAW = 4_893_632
EXPECTED_CURATED = 368_055


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def barcode_status(df):
    # Same pilot rules, with explicit ASCII/length guards: Java regex '$'
    # can accept a final newline. Never accept a NULL checksum result.
    df = df.withColumn('code_trimmed', F.trim('code'))
    c = F.col('code_trimmed')
    shape = F.length(c).isin(8, 12, 13, 14) & ~c.rlike('[^0-9]')
    df = df.withColumn('_padded', F.when(shape, F.lpad(c, 14, '0')))
    total = F.expr('aggregate(sequence(1, 13), 0, (s, i) -> s + try_cast(substring(_padded, i, 1) as int) * IF(i % 2 = 1, 3, 1))')
    check = (10 - total % 10) % 10 == F.expr('try_cast(substring(_padded, 14, 1) as int)')
    return (df.withColumn('row_status',
        F.when(F.col('obsolete') == True, 'obsolete')
        .when(c.isNull() | (c == ''), 'missing_code')
        .when(~shape, 'invalid_format')
        .when(F.col('_padded') == '00000000000000', 'all_zero')
        .when(~F.coalesce(check, F.lit(False)), 'invalid_check_digit')
        .otherwise('active_valid'))
        .withColumn('gtin14', F.when(F.col('row_status') == 'active_valid', F.col('_padded')))
        .drop('_padded'))


def self_test(spark):
    pilot_self_test(spark)
    fixtures = [('036000291452',False,'active_valid'),('0036000291452',None,'active_valid'),
                ('036000291453',False,'invalid_check_digit'),('00000000',False,'all_zero'),
                ('abc',False,'invalid_format'),(None,False,'missing_code'),
                ('00000000000017',True,'obsolete'),(' 96385074 ',False,'active_valid'),
                ('036000291452\n',False,'invalid_format'),('96385074\n',False,'invalid_format'),
                ('１２３４５６７８',False,'invalid_format'),('   ',False,'missing_code')]
    actual = barcode_status(spark.createDataFrame([(a,b) for a,b,_ in fixtures], 'code string, obsolete boolean')).collect()
    require({r.code:r.row_status for r in actual} == {a:s for a,_,s in fixtures}, 'Full barcode fixture failure')
    require(all(r.gtin14 is not None for r in actual if r.row_status == 'active_valid'), 'NULL valid key')
    # Test collision handling and source comparison using synthetic data.
    prod = spark.createDataFrame([
        ('00036000291452','036000291452','Milk',1),('00036000291452','0036000291452','Milk',1),
        ('00000096385074','96385074','Oil',2),('00000000000031','00000000000031',None,4),
        ('00000000000017','00000000000017','Milk',1),('00000000000024','00000000000024','Oil',2)],
        'gtin14 string, code_trimmed string, ingredients_en string, source_nova int')
    cur = spark.createDataFrame([
        ('00036000291452','036000291452','Milk',1),('00000096385074','96385074','Oil',2),
        ('00000000000031','00000000000031',None,4),('00000000000017','00000000000017','Other',1),
        ('00000000000024','00000000000024','Oil',3),('00000000000048','00000000000048','X',1)],
        'gtin14 string, off_code string, off_ingredients_en string, nova_group int')
    result = counts(curated_links(prod, group_counts(prod), cur), 'pilot_link_status')
    require(result == {'ambiguous_in_pilot':1,'consistent_in_pilot':1,'missing_english_comparison':1,
                       'english_text_mismatch':1,'nova_mismatch':1}, f'Source link fixture failed: {result}')
    print('SUCCESS: full-scale barcode and source-link synthetic tests passed.', flush=True)


def size_bytes(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.is_dir() else path.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if not args.self_test:
        require(not OUT.exists() and not WORK.exists(), f'Existing output: {OUT} or {WORK}. Inspect/archive before rerunning.')
        require(RAW.exists() and CURATED.exists(), 'Input Parquet missing')
        require(shutil.disk_usage(ROOT).free >= 40 * 1024**3, 'Less than 40 GiB free on project volume; review storage first.')
    spark = (SparkSession.builder.appName('FoodFactsCatalogueFull').master('local[2]')
             .config('spark.driver.memory','4g').config('spark.sql.shuffle.partitions','64')
             .config('spark.sql.files.maxPartitionBytes','32m')
             .config('spark.sql.parquet.columnarReaderBatchSize','256')
             .config('spark.sql.autoBroadcastJoinThreshold','-1')
             .config('spark.sql.adaptive.autoBroadcastJoinThreshold','-1')
             .config('spark.sql.adaptive.enabled','true')
             .config('spark.ui.enabled','false').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    try:
        self_test(spark)
        if args.self_test:
            return
        start = time.perf_counter()
        stages = {}
        def finish(name, began):
            stages[name] = time.perf_counter() - began
            print(f'{name}: {stages[name]:.2f} s', flush=True)
        WORK.mkdir(parents=True, exist_ok=False)
        print('STAGE 1: scan all OFF records, validate barcodes, write staged products and rejects', flush=True)
        began = time.perf_counter()
        classified = barcode_status(spark.read.parquet(str(RAW)).select(*RAW_COLS)).persist(StorageLevel.DISK_ONLY)
        statuses = counts(classified, 'row_status')
        n = sum(statuses.values())
        p = statuses.get('active_valid',0)
        require(n == EXPECTED_RAW and p > 0, f'Unexpected raw snapshot count: {n}; expected {EXPECTED_RAW}')
        print('OFF row statuses:', statuses, flush=True)
        prepare_products(classified.filter("row_status = 'active_valid'")).write.mode('errorifexists').parquet(str(WORK/'off_products'))
        classified.filter("row_status != 'active_valid'").select('code','obsolete','row_status').write.mode('errorifexists').parquet(str(WORK/'rejected_rows'))
        classified.unpersist(blocking=True)
        finish('extract_transform_write',began)
        print('STAGE 2: read staged products, audit field coverage and global barcode collisions', flush=True)
        began = time.perf_counter()
        products = spark.read.parquet(str(WORK/'off_products'))
        coverage = products.agg(F.count('*').alias('_rows'),
            *[F.count(F.when(F.col(c).isNotNull() & (F.length(F.trim(c))>0),1)).alias(c)
              for c in ['product_name','ingredients_raw','ingredients_en','ingredients_structured_json']],
            F.count('recorded_nova').alias('recorded_nova'),
            F.count(F.when(F.col('india_tagged') == True,1)).alias('india_tagged_true'),
            F.count(F.when(F.col('india_tagged').isNull(),1)).alias('india_tagged_unknown')).first().asDict()
        require(coverage.pop('_rows') == p, 'Accepted row count mismatch')
        rejected = spark.read.parquet(str(WORK/'rejected_rows'))
        require(counts(rejected,'row_status') == {k:v for k,v in statuses.items() if k!='active_valid'}, 'Rejected status counts mismatch')
        groups = group_counts(products).persist(StorageLevel.DISK_ONLY)
        totals = groups.agg(F.count('*').alias('keys'), F.sum('pilot_record_count').alias('rows'),
            F.count(F.when(F.col('pilot_record_count') > 1,1)).alias('collisions'),
            F.sum(F.when(F.col('pilot_record_count') > 1,F.col('pilot_record_count')).otherwise(0)).alias('conflicting_rows'),
            F.count(F.when(F.col('gtin14').isNull(),1)).alias('null_keys')).first().asDict()
        require(totals['rows']==p and totals['null_keys']==0, 'Global grouping mismatch')
        unique = totals['keys']; nc = totals['collisions']
        groups.filter('pilot_record_count > 1').withColumnRenamed('pilot_record_count','record_count').write.mode('errorifexists').parquet(str(WORK/'barcode_conflicts'))
        require(spark.read.parquet(str(WORK/'barcode_conflicts')).count()==nc, 'Conflict read-back mismatch')
        finish('coverage_and_conflicts',began)
        print('STAGE 3: link curated OFF-USDA records; do not substitute ingredients or labels', flush=True)
        began = time.perf_counter()
        curated = spark.read.parquet(str(CURATED)).select(*CURATED_COLS)
        cn = curated.agg(F.count('*').alias('rows'),F.countDistinct('gtin14').alias('keys')).first()
        require(cn.rows == EXPECTED_CURATED and cn.keys == cn.rows, f'Unexpected curated count or nonunique keys: {cn}')
        # Narrow projection before join; no shuffle of the full nutrition/JSON payload.
        slim = products.select('gtin14','code_trimmed','ingredients_en','source_nova')
        links = curated_links(slim, groups, curated).withColumnRenamed('pilot_record_count','record_count')
        links = links.withColumn('link_status',F.when(F.col('pilot_link_status')=='ambiguous_in_pilot','ambiguous_barcode').when(F.col('pilot_link_status')=='consistent_in_pilot','consistent_checked_fields').otherwise(F.col('pilot_link_status'))).drop('pilot_link_status')
        links.write.mode('errorifexists').parquet(str(WORK/'curated_links'))
        written = spark.read.parquet(str(WORK/'curated_links'))
        link_status = counts(written,'link_status'); nl=sum(link_status.values())
        require(written.select('gtin14').distinct().count()==nl and nl<=min(unique,cn.rows), 'Curated link count/uniqueness mismatch')
        # Unmatched curated keys retained in a separate audit table, not silently imported.
        unmatched = curated.join(groups.select('gtin14'),'gtin14','left_anti')
        unmatched.write.mode('errorifexists').parquet(str(WORK/'unmatched_curated'))
        nu=spark.read.parquet(str(WORK/'unmatched_curated')).count()
        require(nl+nu==cn.rows, 'Curated matching reconciliation failed')
        groups.unpersist(blocking=True)
        finish('source_links',began)
        report={'experiment':'I8.3.4b','spark_version':spark.version,
                'settings':{'master':'local[2]','driver_memory':'4g','shuffle_partitions':64,'max_input_partition':'32m','parquet_batch_rows':256,'broadcast_threshold':-1},
                'source_paths':{'off':str(RAW.relative_to(ROOT)),'curated':str(CURATED.relative_to(ROOT))},
                'input_rows':n,'row_status_first_failure':statuses,'active_valid_rows':p,
                'distinct_gtin14':unique,'singleton_gtin14':unique-nc,'collision_keys':nc,
                'rows_on_collision_keys':totals['conflicting_rows'],'extra_records_per_key_total':p-unique,
                'nonempty_field_coverage':coverage,'curated_input_rows':cn.rows,'curated_links':nl,
                'unmatched_curated':nu,'curated_link_status':link_status,
                'output_rows_verified':{'off_products':p,'barcode_conflicts':nc,'curated_links':nl,'rejected_rows':n-p,'unmatched_curated':nu},
                'output_bytes':{x.name:size_bytes(x) for x in WORK.iterdir() if x.is_dir()},
                'stage_seconds':stages,'elapsed_seconds':time.perf_counter()-start,
                'limitations':['Full snapshot counts, not proof of source freshness or correct product identity.',
                    'GTIN checksum is structural only. UPC-E expansion and non-GTIN OFF codes not supported.',
                    'All active valid duplicate records retained; ambiguous keys must not be auto-resolved.',
                    'Nonblank structured JSON does not prove usable ingredient text; extraction deferred.',
                    'Curated linkage verifies selected OFF fields only, not USDA formulation identity.',
                    'Missing metadata is not proof of absence; no category-specific analytics fields added in this revision.',
                    'No PostgreSQL writes, inference, source changes or distributed cluster benchmark.']}
        (WORK/'i8_3_4b_full_report.json').write_text(json.dumps(report,indent=2)+'\n')
        require(not OUT.exists(), 'Final output appeared during run; refusing to replace it')
        WORK.rename(OUT)
        print(json.dumps(report,indent=2))
        print('SUCCESS: full catalogue reconciled, verified and published to staged Parquet.', flush=True)
    finally:
        spark.stop()


if __name__=='__main__':
    main()
