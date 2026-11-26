"""I8.4a: before/after B-tree barcode lookup benchmark using local psql.
No new Python packages, Spark or model required. Run --before then --after.
"""
import argparse
import hashlib
import json
import random
import re
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/backend/i8_4a_index'
TABLE = 'catalogue.off_products'
INDEX = 'idx_off_products_gtin14'
CMD = ['docker','compose','-f',str(ROOT/'infra/compose.yaml'),'exec','-T','postgres',
       'psql','-X','-qAt','-U','food_admin','-d','foodfacts','-v','ON_ERROR_STOP=1']
SETTINGS = "SET statement_timeout='15min'; SET lock_timeout='15s';"
TRIALS = 5


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def psql(sql):
    proc = subprocess.run(CMD, input=SETTINGS+'\n'+sql, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip())
    return proc.stdout.strip()


def query(code):
    require(re.fullmatch(r'[0-9]{14}',code) is not None, 'Invalid fixture shape')
    return (f'SELECT record_id, gtin14, product_name, recorded_nova, ingredients_en '
            f"FROM {TABLE} WHERE gtin14 = '{code}'")


def check_digit(prefix):
    require(re.fullmatch(r'[0-9]{13}',prefix) is not None,'Invalid prefix')
    return str((-sum(int(c)*(3 if i%2==0 else 1) for i,c in enumerate(reversed(prefix))))%10)


def result(code):
    return json.loads(psql('SELECT coalesce(jsonb_agg(to_jsonb(q) ORDER BY record_id),\'[]\'::jsonb) FROM ('+query(code)+') q;'))


def fingerprint(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=True,separators=(',',':')).encode()).hexdigest()


def snapshot():
    return json.loads(psql("""SELECT json_build_object(
      'server_version', current_setting('server_version'),
      'table_bytes', pg_table_size('catalogue.off_products'),
      'total_bytes', pg_total_relation_size('catalogue.off_products'),
      'indexes', (SELECT coalesce(json_agg(json_build_object('name',indexname,'definition',indexdef) ORDER BY indexname),'[]'::json)
                  FROM pg_indexes WHERE schemaname='catalogue' AND tablename='off_products'),
      'shared_buffers', current_setting('shared_buffers'),
      'max_parallel_workers_per_gather', current_setting('max_parallel_workers_per_gather'),
      'jit', current_setting('jit'), 'random_page_cost', current_setting('random_page_cost'));
    """))


def baseline_guard(meta):
    indexes=meta['indexes']
    require(len(indexes)==1 and 'UNIQUE INDEX' in indexes[0]['definition'] and '(record_id)' in indexes[0]['definition'],
            'Expected only the record_id primary-key index. Do not drop existing indexes; inspect first.')


def fixtures():
    keys={'known_with_nova':'00000105000011','known_without_nova':'00000105000059',
          'ambiguous':psql('SELECT gtin14 FROM catalogue.barcode_conflicts ORDER BY gtin14 LIMIT 1;')}
    # Deliberately generated structural GTINs, checked for local absence; not assigned product identifiers.
    for i in range(1,6):
        prefix=f'9999999999{i:03d}'
        candidate=prefix+check_digit(prefix)
        if not result(candidate):
            keys['absent']=candidate
            break
    require('absent' in keys,'No verified absent candidate; inspect')
    expected={'known_with_nova':1,'known_without_nova':1,'ambiguous':2,'absent':0}
    out={}
    for label,code in keys.items():
        rows=result(code)
        require(len(rows)==expected[label],f'{label}: unexpected result count')
        if label=='known_with_nova': require(rows[0]['recorded_nova'] in (1,2,3,4),'Known NOVA fixture changed')
        if label=='known_without_nova': require(rows[0]['recorded_nova'] is None,'Missing NOVA fixture changed')
        out[label]={'gtin14':code,'rows':len(rows),'result_sha256':fingerprint(rows)}
    return out


def plan_nodes(plan):
    yield plan
    for child in plan.get('Plans',[]):
        yield from plan_nodes(child)


def explain(code):
    report=json.loads(psql('EXPLAIN (ANALYZE, BUFFERS, TIMING OFF, FORMAT JSON) '+query(code)+';'))[0]
    return report


def bench(fixtures, indexed):
    results={k:[] for k in fixtures}
    for label,fixture in fixtures.items():
        explain(fixture['gtin14'])  # one unmeasured warmup per key
    rng=random.Random(42)
    for round_no in range(TRIALS):
        labels=list(fixtures)
        rng.shuffle(labels)
        for label in labels:
            f=fixtures[label]
            report=explain(f['gtin14'])
            plan=report['Plan']
            require(plan['Actual Rows']==f['rows'] and plan['Actual Loops']==1, f'{label}: EXPLAIN result mismatch')
            nodes=list(plan_nodes(plan))
            scans=[n['Node Type'] for n in nodes if 'Scan' in n['Node Type']]
            names=[n.get('Index Name') for n in nodes if n.get('Index Name')]
            require((INDEX in names) if indexed else any('Seq Scan' in s for s in scans),
                    f'{label}: unexpected scan plan {scans}, indexes {names}')
            results[label].append({'execution_ms':report['Execution Time'],'planning_ms':report['Planning Time'],
                                   'scan_nodes':scans,'shared_hit_blocks':plan.get('Shared Hit Blocks',0),
                                   'shared_read_blocks':plan.get('Shared Read Blocks',0),'explain':report})
        print(f'Completed measured round {round_no+1}/{TRIALS}',flush=True)
    summary={}
    for label,records in results.items():
        times=[r['execution_ms'] for r in records]
        summary[label]={'median_ms':statistics.median(times),'min_ms':min(times),'max_ms':max(times),
                        'trials_ms':times,'scan_nodes':records[0]['scan_nodes'],
                        'median_shared_hit_blocks':statistics.median(r['shared_hit_blocks'] for r in records),
                        'median_shared_read_blocks':statistics.median(r['shared_read_blocks'] for r in records)}
    return {'summary':summary,'raw_trials':results}


def save(path,data):
    with path.open('x',encoding='utf-8') as f:
        json.dump(data,f,indent=2); f.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--before',action='store_true'); group.add_argument('--after',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    before_path=OUT/'before.json'; after_path=OUT/'after.json'
    require(not after_path.exists(),'Final report exists; refusing rerun')
    meta=snapshot(); baseline_guard(meta)
    if args.before:
        require(not before_path.exists(),'Baseline exists; preserve it and use --after')
        count=int(psql('SELECT count(*) FROM catalogue.off_products;'))
        require(count==4693098,'Unexpected table count')
        # Loader already analyzed tables; do not change statistics between phases.
        f=fixtures(); print('Verified fixtures: '+json.dumps(f),flush=True)
        measured=bench(f,False)
        save(before_path,{'experiment':'I8.4a','phase':'before','rows':count,'metadata':meta,'fixtures':f,**measured})
        print(json.dumps(measured['summary'],indent=2))
        print('SUCCESS: baseline saved; no index created.',flush=True)
    else:
        require(before_path.exists(),'Run --before first')
        before=json.loads(before_path.read_text()); f=before['fixtures']
        require(meta==before['metadata'],'Database metadata changed since baseline; inspect before proceeding')
        for label,fixture in f.items():
            require(fingerprint(result(fixture['gtin14']))==fixture['result_sha256'],f'{label}: fixture changed')
        print('CREATE: nonunique B-tree index on gtin14',flush=True)
        began=time.perf_counter()
        psql(f'CREATE INDEX {INDEX} ON {TABLE} USING btree (gtin14);')
        build=time.perf_counter()-began
        # Record successful DDL immediately, even if subsequent checks fail.
        size=int(psql(f"SELECT pg_relation_size('catalogue.{INDEX}');"))
        print(f'Index created: {build:.3f} s (client wall time); {size:,} bytes',flush=True)
        aftermeta=snapshot()
        save(OUT/'index_created.json',{'build_wall_seconds':build,'index_bytes':size,'metadata':aftermeta})
        measured=bench(f,True)
        for label,fixture in f.items():
            require(fingerprint(result(fixture['gtin14']))==fixture['result_sha256'],f'{label}: results changed after index')
        comparison={}
        for label,a in measured['summary'].items():
            b=before['summary'][label]['median_ms']; x=a['median_ms']
            comparison[label]={'before_median_ms':b,'after_median_ms':x,'speedup':b/x if x>0 else None}
        report={'experiment':'I8.4a','phase':'after','metadata':aftermeta,'fixtures':f,'index_build_wall_seconds':build,
                'index_bytes':size,'comparison':comparison,**measured,
                'limitations':['Four fixed cases, five measured queries per case per phase, one warmup each.','EXPLAIN server execution time, not network or API latency; TIMING OFF reduces per-node timing overhead.','Caches are not cleared; before/after order and background activity can affect results.','No concurrency, cold-cache or distributed benchmark.','Index build wall time includes client process overhead; only main table index measured.']}
        save(after_path,report)
        print(json.dumps({'comparison':comparison,'index_bytes':size,'index_build_wall_seconds':build,'after_summary':measured['summary']},indent=2))
        print('SUCCESS: lookup results unchanged; barcode index verified and retained.',flush=True)


if __name__=='__main__':
    main()
