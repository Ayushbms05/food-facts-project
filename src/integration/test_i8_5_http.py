"""HTTP smoke test against the running local API; standard library only."""
import json
from urllib.request import urlopen
from urllib.error import HTTPError
from pathlib import Path
from importlib.metadata import version

BASE = 'http://127.0.0.1:8000'
ROOT = Path(__file__).resolve().parents[2]


def get(path):
    try:
        response = urlopen(BASE + path, timeout=15)
    except HTTPError as exc:
        response = exc
    with response:
        return response.status, json.load(response)


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def main():
    code, health = get('/health')
    require(code == 200 and health == {'status':'ok','database':'ready'}, 'Health check failed')
    cases = [('known','00000105000011',200,'found',1),
             ('missing_nova','00000105000059',200,'found',1),
             ('ambiguous','00059527070430',409,'ambiguous',2),
             ('absent','99999999990017',404,'not_found',0),
             ('alias_12','000105000011',200,'found',1),
             ('alias_13','0000105000011',200,'found',1),
             ('bad_checksum','00000105000012',400,'invalid_barcode',0),
             ('bad_format','abc123',400,'invalid_barcode',0),
             ('zero','00000000',400,'invalid_barcode',0),
             ('checked_link','00000000005487',200,'found',1),
             ('missing_english','00000000035071',200,'found',1)]
    bodies, results = {}, {}
    for label, barcode, expected_http, expected_status, count in cases:
        status, body = get('/products/'+barcode)
        require((status,body.get('status'),body.get('matching_rows')) == (expected_http,expected_status,count), f'{label}: unexpected response')
        require(('product' in body) == (expected_status == 'found'), f'{label}: unexpected selected product')
        bodies[label] = body
        results[label] = {'http':status,'status':expected_status,'matching_rows':count}
        print(label, status, expected_status)
    require(bodies['known']['product']['recorded_nova'] in (1,2,3,4),'Known NOVA missing')
    require(bodies['missing_nova']['product']['recorded_nova'] is None,'Missing NOVA changed')
    require(bodies['known']==bodies['alias_12']==bodies['alias_13'],'Aliases differ')
    require(bodies['checked_link']['source_comparison'] is not None,'Checked link missing')
    require(bodies['missing_english']['source_comparison'] is None,'Unchecked link exposed')
    report = {'experiment':'I8.5b','versions':{p:version(p) for p in ('fastapi','uvicorn','psycopg','psycopg-pool')},'health':health,'passed_cases':len(cases),'cases':results,'limitations':['Local serial smoke test; no HTTP latency/concurrency benchmark, no model or Android integration.']}
    output = ROOT/'data/backend/i8_5b_http_report.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
    print('SUCCESS: health and 11 HTTP lookup cases verified.')


if __name__=='__main__':
    main()
