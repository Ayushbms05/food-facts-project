"""I8.3.4a: bounded real-data Spark ETL pilot, no database writes.
Run from repo root using .venv and Java 17. --self-test uses synthetic rows.
"""
import argparse
import json
import time
from pathlib import Path
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark import StorageLevel

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/food.parquet'
CURATED = ROOT / 'data/curated/source_aware_products.parquet'
OUT = ROOT / 'data/backend/catalogue_pilot'
LIMIT = 10000
RAW_COLS = ['code','product_name','brands','lang','ingredients_text','ingredients',
            'nova_group','countries_tags','allergens_tags','traces_tags','additives_tags',
            'nutriscore_grade','nutriments','nutrition_data_per','quantity','serving_size','obsolete']
CURATED_COLS = ['gtin14','off_code','off_product_name','off_ingredients_en','nova_group',
                'usda_fdc_id','usda_description','usda_ingredients','usda_brand','usda_date',
                'off_records_for_gtin','agreement_category','model_text_source','needs_review',
                'dl_candidate_off','text_removed_by_cleaning']


def barcode_status(df):
    # Validate format before any arithmetic. try_cast avoids ANSI-mode errors.
    df = df.withColumn('code_trimmed', F.trim('code'))
    shape = F.col('code_trimmed').rlike(r'^(?:[0-9]{8}|[0-9]{12}|[0-9]{13}|[0-9]{14})$')
    df = df.withColumn('_padded', F.when(shape, F.lpad('code_trimmed',14,'0')))
    total = F.expr('aggregate(sequence(1, 13), 0, (s, i) -> s + try_cast(substring(_padded, i, 1) as int) * IF(i % 2 = 1, 3, 1))')
    check = ((10 - total % 10) % 10 == F.expr('try_cast(substring(_padded, 14, 1) as int)'))
    return (df.withColumn('row_status',
        F.when(F.col('obsolete') == True, 'obsolete')
         .when(F.col('code_trimmed').isNull() | (F.col('code_trimmed') == ''), 'missing_code')
         .when(~shape, 'invalid_format')
         .when(F.col('_padded') == '00000000000000', 'all_zero')
         .when(~check, 'invalid_check_digit')
         .otherwise('active_valid'))
        .withColumn('gtin14', F.when(F.col('row_status') == 'active_valid', F.col('_padded')))
        .drop('_padded'))


def localized(field):
    usable = F.filter(F.col(field), lambda x: x['text'].isNotNull() & (F.length(F.trim(x['text'])) > 0))
    en = F.get(F.filter(usable, lambda x: x['lang'] == 'en'), 0)
    declared = F.get(F.filter(usable, lambda x: x['lang'] == F.col('lang')), 0)
    main = F.get(F.filter(usable, lambda x: x['lang'] == 'main'), 0)
    return F.coalesce(en, declared, main, F.get(usable, 0)), en


def prepare_products(valid):
    name, _ = localized('product_name')
    ing, en = localized('ingredients_text')
    valid = valid.withColumn('_name',name).withColumn('_ing',ing).withColumn('_en',en)
    def js(c):
        return F.to_json(F.col(c), {'ignoreNullFields':'false'})
    return valid.select(
        'gtin14', F.col('code').alias('off_code'), 'code_trimmed',
        F.col('_name.text').alias('product_name'), F.col('_name.lang').alias('name_language'),
        'brands', F.col('lang').alias('declared_language'),
        F.col('_ing.text').alias('ingredients_raw'), F.col('_ing.lang').alias('ingredients_language'),
        F.col('_en.text').alias('ingredients_en'), F.col('ingredients').alias('ingredients_structured_json'),
        F.when(F.col('nova_group').between(1,4), F.col('nova_group')).alias('recorded_nova'),
        F.col('nova_group').alias('source_nova'),
        js('countries_tags').alias('country_tags_json'), F.array_contains('countries_tags','en:india').alias('india_tagged'),
        js('allergens_tags').alias('allergen_tags_json'), js('traces_tags').alias('trace_tags_json'),
        js('additives_tags').alias('additive_tags_json'), 'nutriscore_grade', js('nutriments').alias('nutrients_json'),
        'nutrition_data_per','quantity','serving_size','obsolete')


def group_counts(products):
    return products.groupBy('gtin14').agg(F.count('*').alias('pilot_record_count'))


def curated_links(products, groups, curated):
    # Only keys present in this pilot. A unique sample key is not globally unique.
    keys = curated.join(groups, 'gtin14', 'inner')
    unique = products.join(groups.filter('pilot_record_count = 1').select('gtin14'),'gtin14').select(
        'gtin14', F.col('code_trimmed').alias('_code'), F.col('ingredients_en').alias('_en'),
        F.col('source_nova').alias('_nova'))
    # Do not allow empty strings to be interpreted as positive ingredient evidence.
    return (keys.join(unique,'gtin14','left')
        .withColumn('pilot_link_status',
            F.when(F.col('pilot_record_count') > 1,'ambiguous_in_pilot')
            .when(~F.trim('off_code').eqNullSafe(F.col('_code')), 'code_mismatch')
            .when(~F.col('nova_group').eqNullSafe(F.col('_nova')), 'nova_mismatch')
            .when(F.col('_en').isNull() | (F.trim('_en') == '') | F.col('off_ingredients_en').isNull() |
                  (F.trim('off_ingredients_en') == ''),'missing_english_comparison')
            .when(~F.trim('off_ingredients_en').eqNullSafe(F.trim('_en')), 'english_text_mismatch')
            .otherwise('consistent_in_pilot'))
        .drop('_code','_en','_nova'))


def self_test(spark):
    rows = [('036000291452',False),('0036000291452',None),('036000291453',False),
            ('00000000',False),('abc',False),(None,False),('00000000000017',True),(' 96385074 ',False)]
    result = barcode_status(spark.createDataFrame(rows,'code string, obsolete boolean')).collect()
    expected = ['active_valid','active_valid','invalid_check_digit','all_zero','invalid_format','missing_code','obsolete','active_valid']
    assert {r.code:r.row_status for r in result} == {r[0]:s for r,s in zip(rows,expected)}
    assert len({r.gtin14 for r in result if r.code in ('036000291452','0036000291452')}) == 1
    sample = spark.createDataFrame([('fr',[('fr','Lait'),('en','Milk')]),('fr',[('fr','Lait')]),('en',[])],
                                   'lang string, entries array<struct<lang:string,text:string>>')
    chosen,en = localized('entries')
    actual = [(r.chosen.text if r.chosen else None,r.en.text if r.en else None) for r in sample.select(chosen.alias('chosen'),en.alias('en')).collect()]
    assert actual == [('Milk','Milk'),('Lait',None),(None,None)]
    print('SUCCESS: synthetic GTIN, alias, obsolete and language-selection tests passed.')


def counts(df, field):
    return {str(r[field]):r['count'] for r in df.groupBy(field).count().collect()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    args = ap.parse_args()
    if not args.self_test and OUT.exists():
        raise FileExistsError(f'{OUT} exists; preserve or inspect previous run before repeating.')
    spark = (SparkSession.builder.appName('FoodFactsCataloguePilot').master('local[2]')
             .config('spark.driver.memory','3g').config('spark.sql.shuffle.partitions','4')
             .config('spark.sql.files.maxPartitionBytes','32m')
             .config('spark.sql.parquet.columnarReaderBatchSize','256')
             .config('spark.ui.enabled','false').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    try:
        self_test(spark)
        if args.self_test: return
        start = time.perf_counter()
        print('EXTRACT: bounded OFF slice (not random or representative)', flush=True)
        raw = spark.read.parquet(str(RAW)).select(*RAW_COLS).limit(LIMIT)
        classified = barcode_status(raw).persist(StorageLevel.DISK_ONLY)
        n = classified.count()
        assert n == LIMIT, f'Expected {LIMIT} input rows, got {n}'
        statuses = counts(classified,'row_status')
        assert sum(statuses.values()) == n
        print('TRANSFORM: validate and extract product fields', flush=True)
        products = prepare_products(classified.filter("row_status = 'active_valid'")).persist(StorageLevel.DISK_ONLY)
        p = products.count()
        assert p == statuses.get('active_valid',0) and p > 0
        groups = group_counts(products).persist(StorageLevel.DISK_ONLY)
        unique = groups.count()
        collisions = groups.filter('pilot_record_count > 1')
        nc = collisions.count()
        assert groups.agg(F.sum('pilot_record_count')).first()[0] == p
        print('LINK: compare curated records with pilot keys', flush=True)
        curated = spark.read.parquet(str(CURATED)).select(*CURATED_COLS)
        links = curated_links(products,groups,curated).persist(StorageLevel.DISK_ONLY)
        nl = links.count()
        assert links.select('gtin14').distinct().count() == nl, 'Curated duplicate key in pilot links'
        link_status = counts(links,'pilot_link_status')
        assert sum(link_status.values()) == nl
        assert nl <= unique
        coverage = products.agg(*[F.count(F.when(F.col(c).isNotNull() & (F.length(F.trim(c))>0),1)).alias(c) for c in ['product_name','ingredients_raw','ingredients_en','ingredients_structured_json']],F.count('recorded_nova').alias('recorded_nova'),F.count(F.when(F.col('india_tagged') == True,1)).alias('india_tagged_true'),F.count(F.when(F.col('india_tagged').isNull(),1)).alias('india_tagged_unknown')).first().asDict()
        print('WRITE: isolated staged Parquet tables, no PostgreSQL changes', flush=True)
        OUT.mkdir(parents=True,exist_ok=False)
        outputs = {'off_products':products,'barcode_conflicts':collisions,
                   'curated_links':links,'rejected_rows':classified.filter("row_status != 'active_valid'").select('code','obsolete','row_status')}
        expected_rows = {'off_products':p,'barcode_conflicts':nc,'curated_links':nl,'rejected_rows':n-p}
        for name, frame in outputs.items():
            path = OUT/name
            frame.write.mode('errorifexists').parquet(str(path))
            assert spark.read.parquet(str(path)).count() == expected_rows[name], f'{name}: read-back count mismatch'
        report = {'experiment':'I8.3.4a','spark_version':spark.version,'input_rows':n,
                  'row_status_first_failure':statuses,'active_valid_rows':p,'distinct_pilot_gtin14':unique,
                  'collision_keys_in_pilot':nc,'extra_records_per_key_total':p-unique,
                  'nonempty_field_coverage':coverage,'curated_links':nl,'curated_link_status':link_status,
                  'output_rows_verified':expected_rows,'elapsed_seconds':time.perf_counter()-start,
                  'limitations':['Unordered limit slice; not representative and not a fixed statistical sample.',
                                 'Curated links and collisions evaluated only within pilot; global uniqueness not established.',
                                 'Structured ingredient JSON preserved; exact DL extraction deferred to inference environment.',
                                 'No PostgreSQL writes, no model inference, no source changes.']}
        (OUT/'i8_3_4a_pilot_report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        print('SUCCESS: pilot row counts reconciled and staged Parquet counts verified.')
    finally:
        spark.stop()

if __name__ == '__main__':
    main()
