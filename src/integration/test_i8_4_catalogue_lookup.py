"""I8.4b: live read-only lookup contract checks and warm client latency."""
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
from .catalogue_lookup import connect_catalogue, lookup, self_test, normalize_barcode

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/backend/i8_4b_lookup_report.json'
BASE = ROOT / 'data/backend/i8_4a_index/after.json'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def main():
    import psycopg
    self_test()
    require(not OUT.exists(), 'Report exists; preserve previous results before another run.')
    baseline = json.loads(BASE.read_text())['fixtures']
    cases = {k: v['gtin14'] for k,v in baseline.items()}
    cases['alias_12'] = cases['known_with_nova'][-12:]
    cases['alias_13'] = cases['known_with_nova'][-13:]
    cases['invalid_checksum'] = cases['known_with_nova'][:-1]+str((int(cases['known_with_nova'][-1])+1)%10)
    cases['invalid_format'] = 'abc123'
    cases['invalid_zero'] = '00000000'
    t0 = time.perf_counter()
    with connect_catalogue() as conn:
        connection_ms = (time.perf_counter()-t0)*1000
        require(conn.execute('SHOW transaction_read_only').fetchone()['transaction_read_only']=='on', 'Read-only guard failed')
        for label,status in [('checked_link','consistent_checked_fields'),('missing_english_link','missing_english_comparison')]:
            row = conn.execute('SELECT gtin14 FROM catalogue.curated_links WHERE link_status=%s AND record_count=1 ORDER BY gtin14 LIMIT 1',(status,)).fetchone()
            require(row is not None, 'Missing expected curated fixture')
            cases[label] = row['gtin14']
        expected = {'known_with_nova':'found','known_without_nova':'found','ambiguous':'ambiguous','absent':'not_found',
                    'alias_12':'found','alias_13':'found','invalid_checksum':'invalid_barcode','invalid_format':'invalid_barcode',
                    'invalid_zero':'invalid_barcode','checked_link':'found','missing_english_link':'found'}
        first = {k:lookup(conn,code) for k,code in cases.items()}
        for k,v in first.items():
            require(v['status']==expected[k], f'{k}: unexpected lookup status')
        require(first['known_with_nova']['product']['recorded_nova'] in (1,2,3,4),'Known label mismatch')
        require(first['known_without_nova']['product']['recorded_nova'] is None,'Missing label mismatch')
        require(first['alias_12']==first['alias_13']==first['known_with_nova'],'Alias result mismatch')
        require(first['ambiguous']['matching_rows']==2 and 'product' not in first['ambiguous'],'Ambiguous selection error')
        require(first['checked_link']['source_comparison'] is not None,'Checked link not exposed')
        require(first['missing_english_link']['curated_link_status']=='missing_english_comparison' and first['missing_english_link']['source_comparison'] is None,'Unverified link exposed')
        for k in baseline:
            require(first[k]['matching_rows']==baseline[k]['rows'], 'Fixture row count changed')
            # Compare the exact projected fields from I8.4a as well.
            from .benchmark_i8_4_barcode_index import fingerprint
            rows=conn.execute('SELECT record_id, gtin14, product_name, recorded_nova, ingredients_en FROM catalogue.off_products WHERE gtin14=%s ORDER BY record_id',(cases[k],)).fetchall()
            require(fingerprint(rows)==baseline[k]['result_sha256'], 'Baseline result changed')
        for k,v in first.items():
            print(f'{k}: {v["status"]} ({v["matching_rows"]} matching rows)')
        timed = [k for k in cases if not k.startswith('invalid') and not k.startswith('alias')]
        times = {k:[] for k in timed}
        rng = random.Random(42)
        # Initial successful lookups above serve as warmup; one reused connection.
        for _ in range(30):
            rng.shuffle(timed)
            for k in timed:
                t=time.perf_counter_ns(); result=lookup(conn,cases[k]); dt=(time.perf_counter_ns()-t)/1e6
                require(result==first[k], f'{k}: lookup changed')
                times[k].append(dt)
        summary={k:{'median_ms':statistics.median(v),'min_ms':min(v),'max_ms':max(v),'trials_ms':v} for k,v in times.items()}
        sizes=conn.execute("SELECT tablename, pg_total_relation_size(quote_ident(schemaname)||'.'||quote_ident(tablename)) AS bytes FROM pg_tables WHERE schemaname='catalogue' ORDER BY tablename").fetchall()
        report={'experiment':'I8.4b','psycopg_version':psycopg.__version__,'postgres_version':conn.info.server_version,
                'connection_ms':connection_ms,'cases':{k:{'input':cases[k],'status':v['status'],'matching_rows':v['matching_rows'],'result_sha256':digest(v)} for k,v in first.items()},
                'passed_cases':len(first),'warm_client_latency':summary,'table_sizes':sizes,
                'limitations':['Six fixed database cases, 30 measured calls each; serial warmed connection, no concurrent or cold-cache test.', 'Times include Python normalization, SQL, local TCP, result decoding and decision logic; exclude connection setup and HTTP/model/mobile.', 'Prototype uses existing administrator credentials with session read-only mode; least-privilege role still needed before API deployment.', 'Source comparison is not independent validation of formulation or label.']}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open('x',encoding='utf-8') as f: json.dump(report,f,indent=2); f.write('\n')
    print(json.dumps({**report,'warm_client_latency':{k:{a:b for a,b in v.items() if a!='trials_ms'} for k,v in summary.items()}},indent=2))
    print('SUCCESS: application lookup contract and warm client timing verified.')


if __name__=='__main__':
    main()
