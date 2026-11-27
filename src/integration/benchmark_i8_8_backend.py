"""I8.8a: bounded local HTTP benchmark; synthetic requests, no data writes."""
import hashlib
import json
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[2]
BASE = 'http://127.0.0.1:8000'
LEVELS = (1, 4, 8)
N = 60
LOOKUPS = [
    ('known', '00000105000011', 200, 'found', 1),
    ('missing_nova', '00000105000059', 200, 'found', 1),
    ('ambiguous', '00059527070430', 409, 'ambiguous', 2),
    ('absent', '99999999990017', 404, 'not_found', 0),
    ('checked_link', '00000000005487', 200, 'found', 1),
    ('missing_english', '00000000035071', 200, 'found', 1),
]
PREDICTIONS = [
    ('milk', 'Milk', 1, [96.84, 1.71, 1.22, .23]),
    ('oil', 'Sunflower oil', 2, [.17, 98.95, .81, .07]),
    ('cheese', 'Milk, salt, bacterial culture, rennet', 3, [.24, 1.09, 98.37, .30]),
    ('biscuits', 'Wheat flour, sugar, palm oil, emulsifier, artificial flavouring', 4, [.33, .41, 2.06, 97.21]),
]
FIXTURES = {name: dict(kind='lookup', path='/products/'+key, http=http, status=status, rows=rows)
            for name,key,http,status,rows in LOOKUPS}
FIXTURES.update({name: dict(kind='prediction', path='/predict', http=200, status='predicted',
                            payload={'ingredient_text':text}, nova=nova, scores=scores)
                 for name,text,nova,scores in PREDICTIONS})
FIXTURES.update({
    'bad_barcode': dict(kind='validation', path='/products/abc123', http=400, status='invalid_barcode', rows=0),
    'ocr_review': dict(kind='validation', path='/predict', http=400, status='needs_review',
                       reason='unrecognized_short_fragment', payload={'ingredient_text':'aoa','source':'ocr','reviewed':True}),
})
WORKLOADS = {'lookup': [x[0] for x in LOOKUPS], 'prediction': [x[0] for x in PREDICTIONS], 'mixed': list(FIXTURES)}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def call(path, payload=None):
    request = Request(BASE+path, data=None if payload is None else json.dumps(payload).encode(),
                      headers={'Content-Type':'application/json', 'Connection':'close'})
    start = time.perf_counter_ns()
    try:
        response = urlopen(request, timeout=30)
    except HTTPError as exc:
        response = exc
    with response:
        code, body = response.status, json.load(response)
    return code, body, (time.perf_counter_ns()-start)/1e6


def digest(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(',',':')).encode()).hexdigest()


def check(name, code, body, baseline=None):
    f = FIXTURES[name]
    require(code == f['http'] and body.get('status') == f['status'], 'Unexpected HTTP/status')
    if 'rows' in f:
        require(body.get('matching_rows') == f['rows'], 'Unexpected row count')
        require(('product' in body) == (f['status'] == 'found'), 'Unexpected product selection')
    if name == 'known':
        require(body['product']['recorded_nova'] == 1, 'Recorded NOVA changed')
    if name == 'missing_nova':
        require(body['product']['recorded_nova'] is None, 'Missing NOVA changed')
    if name in ('checked_link','missing_english'):
        require((body['source_comparison'] is not None) == (name == 'checked_link'), 'Curated link policy changed')
    if f['kind'] == 'prediction':
        require(body.get('predicted_nova') == f['nova'] and body.get('prediction_mode') == 'original', 'Prediction class/mode changed')
        require(body.get('model') == 'distilbert_food_nova' and body.get('policy_version') == '1.0' and body.get('input_source') == 'manual', 'Model metadata changed')
        scores = [body['class_probabilities'][f'NOVA {i}'] for i in range(1,5)]
        require(all(math.isfinite(v) and 0<=v<=100 for v in scores) and abs(sum(scores)-100)<=.03, 'Invalid score vector')
        require(max(abs(a-b) for a,b in zip(scores,f['scores']))<=.1, 'Class score drift > 0.1 pp')
        require(body['confidence'] == scores[f['nova']-1] and body['confidence'] == max(scores) and 'recorded_nova' not in body, 'Prediction contract changed')
    elif baseline is not None:
        require(digest(body) == baseline, 'Response changed from verified baseline')
    if name == 'ocr_review':
        require(body.get('reason') == f['reason'] and body.get('quality_rule_version') == '0.1', 'OCR decision changed')
        require(not {'predicted_nova','confidence','class_probabilities'} & body.keys(), 'Withheld text returned prediction')


def health():
    code, body, _ = call('/health')
    require(code==200 and body=={'status':'ok','database':'ready'}, 'Health contract failed')


def observe(name, baselines):
    f = FIXTURES[name]
    start = time.perf_counter_ns()
    try:
        code, body, ms = call(f['path'],f.get('payload'))
        check(name,code,body,baselines[name])
        return dict(case=name,kind=f['kind'],http=code,ms=ms,passed=True)
    except Exception as exc:
        return dict(case=name,kind=f['kind'],ms=(time.perf_counter_ns()-start)/1e6,
                    passed=False,error=type(exc).__name__,detail=str(exc) if isinstance(exc,RuntimeError) else 'Transport or response parsing error')


def stats(rows):
    values = sorted(r['ms'] for r in rows if r['passed'])
    return dict(calls=len(rows),passed=len(values),failed=len(rows)-len(values),
                median_ms=statistics.median(values) if values else None,
                p95_ms=values[math.ceil(.95*len(values))-1] if values else None,
                min_ms=min(values) if values else None,max_ms=max(values) if values else None)


def sequence(names):
    require(N % len(names)==0,'Uneven workload')
    order = names*(N//len(names))
    random.Random(20261127).shuffle(order)
    return order


def main():
    output = ROOT/'data/backend/i8_8a_backend_benchmark.json'
    require(not output.exists(),'Report exists; preserve it before any new run')
    health()
    code, spec, _ = call('/openapi.json')
    require(code==200 and spec['info']['version']=='0.3.0','Expected verified API version 0.3.0')
    baselines = {}
    for name,f in FIXTURES.items():
        code,body,_ = call(f['path'],f.get('payload'))
        check(name,code,body)
        baselines[name] = digest(body)
    print('PASS: health, API version and 12 baseline fixtures',flush=True)
    report = dict(experiment='I8.8a',api_version='0.3.0',synthetic=True,
                  versions={p:version(p) for p in ('fastapi','uvicorn','torch','transformers','psycopg','psycopg-pool')},
                  baseline_sha256=baselines,requests_per_phase=N,concurrency_levels=LEVELS,
                  seed=20261127,phases=[],limitations=[
                      'Single local client and single Uvicorn process; fixed small fixtures; warm cache; one run per condition.',
                      'At most the configured number of workers in flight; closed-loop bounded thread pool; fresh HTTP connection per request; excludes task queue wait before worker start.',
                      'Expected HTTP 400/404/409 count as correct outcomes; throughput is completed requests per phase wall time, not prediction-only throughput for mixed phases.',
                      'Percentile is nearest-rank p95 of successful measured calls; small per-kind sample sizes.',
                      'Fixed workload and level order; no causal isolation of server bottlenecks, no Android/network, Kafka, fault injection or production capacity claim.',
                      'No source or database writes; no training; no real user scan events.'])
    try:
        for workload,names in WORKLOADS.items():
            order = sequence(names)
            for workers in LEVELS:
                # Warm each fixture once before each measured phase.
                for name in names:
                    r=observe(name,baselines)
                    require(r['passed'],f'Warm-up failed: {name}: {r.get("error")}')
                start=time.perf_counter()
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    rows=list(executor.map(lambda name:observe(name,baselines),order))
                wall=time.perf_counter()-start
                summary=stats(rows)
                phase=dict(workload=workload,workers=workers,wall_seconds=wall,
                           completed_requests_per_second=len(rows)/wall,
                           correct_requests_per_second=summary['passed']/wall,overall=summary,
                           by_kind={kind:stats([r for r in rows if r['kind']==kind]) for kind in sorted({r['kind'] for r in rows})},trials=rows)
                report['phases'].append(phase)
                print(json.dumps({k:v for k,v in phase.items() if k!='trials'}),flush=True)
                require(summary['failed']==0,f'Unexpected responses in {workload}/{workers}; inspect saved report')
        health()
        for name,f in FIXTURES.items():
            code,body,_=call(f['path'],f.get('payload'))
            check(name,code,body,baselines[name])
        report['post_benchmark_verified']=True
    except Exception as exc:
        report['failure']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['measured_calls']=sum(p['overall']['calls'] for p in report['phases'])
        report['passed_calls']=sum(p['overall']['passed'] for p in report['phases'])
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as file:
            json.dump(report,file,indent=2)
        print('Report:', output.relative_to(ROOT),flush=True)
    print(f'SUCCESS: {report["passed_calls"]}/{3*len(LEVELS)*N} measured HTTP calls correct; health and fixtures unchanged.')


if __name__=='__main__':
    main()
