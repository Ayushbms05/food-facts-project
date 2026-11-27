"""I8.8b: controlled local PostgreSQL stop/start while API stays running."""
import argparse
import json
import subprocess
import time
from .benchmark_i8_8_backend import ROOT, FIXTURES, call, check, digest, require, health

COMPOSE = ['docker', 'compose', '-f', 'infra/compose.yaml']


def docker(*args, timeout=90):
    return subprocess.check_output([*COMPOSE, *args], cwd=ROOT, text=True, timeout=timeout).strip()


def state(cid):
    return subprocess.check_output(['docker','inspect','--format',
        '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}',cid],
        cwd=ROOT,text=True,timeout=10).strip()


def fixture(name, baselines=None):
    f = FIXTURES[name]
    code, body, ms = call(f['path'], f.get('payload'))
    check(name, code, body, None if baselines is None else baselines[name])
    return {'case':name,'http':code,'status':body['status'],'ms':ms,'sha256':digest(body)}


def unavailable(path):
    code, body, ms = call(path)
    require(code == 503 and body == {'status':'database_unavailable'},
            f'Expected database_unavailable HTTP 503 for {path}; got {code}')
    return {'path':path,'http':code,'status':body['status'],'ms':ms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-local-outage', action='store_true', required=True)
    parser.parse_args()
    out = ROOT/'data/backend/i8_8b_recovery_report.json'
    require(not out.exists(), 'Existing report; preserve it before any repeat')
    cid = docker('ps','-q','postgres')
    require(bool(cid) and len(cid.splitlines()) == 1 and state(cid)=='running healthy',
            'Expected one healthy existing postgres container')
    health()
    code, spec, _ = call('/openapi.json')
    require(code==200 and spec['info']['version']=='0.3.0','Expected API version 0.3.0')
    before = [fixture(name) for name in FIXTURES]
    baselines = {r['case']:r['sha256'] for r in before}
    report = {'experiment':'I8.8b','api_version':'0.3.0','container_id':cid,
              'before':before,'during':[], 'recovery_probes':[], 'after':[], 'passed':False,
              'limitations':['One intentional local database outage; no API restart intended.',
                'Recovery time measured from start command to first successful health probe; includes container startup and polling.',
                'Fixed fixtures, not exhaustive database durability verification; no model failure injection, network partition or real mobile clients.']}
    # Reserve report before any service changes; no overwrite of prior experiment.
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f: json.dump(report,f,indent=2)
    print('PASS: healthy database and 12 baseline fixtures',flush=True)
    try:
        try:
            print('STOP: PostgreSQL only; keep Uvicorn running',flush=True)
            docker('stop','-t','30','postgres')
            require(state(cid).startswith('exited'), 'Database container did not stop')
            for path in ('/health',FIXTURES['known']['path'],FIXTURES['absent']['path']):
                r=unavailable(path); report['during'].append(r); print('PASS:',r,flush=True)
            for name in ('bad_barcode','ocr_review','milk','oil'):
                r=fixture(name,baselines); report['during'].append(r); print('PASS:',r,flush=True)
        finally:
            # Also runs after an assertion failure or normal Ctrl+C. Hard kill/power loss may prevent it.
            print('START: restore existing PostgreSQL container',flush=True)
            start=time.monotonic()
            docker('start','--wait','--wait-timeout','60','postgres',timeout=90)
            require(docker('ps','-q','postgres')==cid and state(cid)=='running healthy', 'Expected same healthy container')
            report['container_start_wait_seconds']=time.monotonic()-start
            deadline=time.monotonic()+60
            while time.monotonic()<deadline:
                try:
                    code,body,ms=call('/health')
                    report['recovery_probes'].append({'http':code,'status':body.get('status'),'ms':ms})
                    if code==200 and body=={'status':'ok','database':'ready'}:
                        report['recovery_seconds']=time.monotonic()-start
                        break
                except (OSError,ValueError) as exc:
                    report['recovery_probes'].append({'error':type(exc).__name__})
                time.sleep(1)
            require('recovery_seconds' in report,'API did not recover within polling window')
            print('PASS: same container healthy; API health recovered',flush=True)
        report['after']=[fixture(name,baselines) for name in FIXTURES]
        health()
        report['passed']=True
    except BaseException as exc:
        report['error_type']=type(exc).__name__
        if isinstance(exc,RuntimeError): report['error']=str(exc)
        raise
    finally:
        out.write_text(json.dumps(report,indent=2)+'\n')
        print('Report:',out.relative_to(ROOT),flush=True)
    print(json.dumps(report,indent=2))
    print('SUCCESS: database outage handled; prediction available; catalogue lookup recovered without API restart.')


if __name__=='__main__':
    main()
