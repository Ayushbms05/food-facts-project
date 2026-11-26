"""I8.6b local API regression and prediction smoke test; standard library only."""
import json
import math
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from importlib.metadata import version

BASE = 'http://127.0.0.1:8000'
ROOT = Path(__file__).resolve().parents[2]


def call(path, payload=None):
    req = Request(BASE + path, data=None if payload is None else json.dumps(payload).encode(),
                  headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    try:
        response = urlopen(req, timeout=60)
    except HTTPError as exc:
        response = exc
    with response:
        code, body = response.status, json.load(response)
    return code, body, (time.perf_counter() - start) * 1000


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    output = ROOT / 'data/backend/i8_6b_prediction_http_report.json'
    require(not output.exists(), 'Report already exists; preserve it before retesting')
    code, body, _ = call('/health')
    require(code == 200 and body == {'status':'ok','database':'ready'}, 'Health contract changed')
    fixtures = [('known','00000105000011',200,'found',1),('missing_nova','00000105000059',200,'found',1),
                ('ambiguous','00059527070430',409,'ambiguous',2),('absent','99999999990017',404,'not_found',0),
                ('alias12','000105000011',200,'found',1),('alias13','0000105000011',200,'found',1),
                ('bad_checksum','00000105000012',400,'invalid_barcode',0),('bad_format','abc123',400,'invalid_barcode',0),
                ('zero','00000000',400,'invalid_barcode',0),('checked_link','00000000005487',200,'found',1),
                ('missing_english','00000000035071',200,'found',1)]
    bodies = {}
    for name, key, http, status, count in fixtures:
        code, body, _ = call('/products/' + key)
        require((code,body.get('status'),body.get('matching_rows')) == (http,status,count), name)
        require(('product' in body) == (status == 'found'), name+' selection')
        bodies[name] = body
    require(bodies['known']==bodies['alias12']==bodies['alias13'], 'Aliases changed')
    require(bodies['known']['product']['recorded_nova']==1 and bodies['missing_nova']['product']['recorded_nova'] is None, 'Recorded labels changed')
    require(bodies['checked_link']['source_comparison'] is not None and bodies['missing_english']['source_comparison'] is None, 'Curated link changed')
    print('PASS: health and all 11 original product lookup cases')
    examples = [('Milk',1,[96.84,1.71,1.22,.23]),('Sunflower oil',2,[.17,98.95,.81,.07]),
                ('Milk, salt, bacterial culture, rennet',3,[.24,1.09,98.37,.30]),
                ('Wheat flour, sugar, palm oil, emulsifier, artificial flavouring',4,[.33,.41,2.06,97.21])]
    predictions = []
    for text, expected, baseline in examples:
        code, body, elapsed = call('/predict', {'ingredient_text':text})
        require(code==200 and body.get('status')=='predicted', 'Prediction failed: '+str(code))
        require(body['predicted_nova']==expected and body['prediction_mode']=='original', 'Label/mode mismatch')
        require(body['model']=='distilbert_food_nova' and body['input_source']=='manual', 'Metadata mismatch')
        scores = [body['class_probabilities'][f'NOVA {i}'] for i in range(1,5)]
        require(all(math.isfinite(v) and 0<=v<=100 for v in scores) and abs(sum(scores)-100)<=.03, 'Invalid scores')
        require(max(abs(a-b) for a,b in zip(scores,baseline))<=.1, 'Score drift > 0.1 percentage point')
        require(body['confidence']==scores[expected-1] and 'recorded_nova' not in body, 'Prediction contract mismatch')
        predictions.append({'ingredient_text':text,'response':body,'http_ms':elapsed,'baseline_scores':baseline})
        print(f'PASS: {text}: NOVA {expected}, {body["confidence"]}%, HTTP {elapsed:.2f} ms')
    code, reviewed, _ = call('/predict', {'ingredient_text':'Milk','source':'ocr','reviewed':True})
    require(code==200 and reviewed['input_source']=='ocr' and reviewed['predicted_nova']==1, 'Reviewed OCR fixture failed')
    negatives = [('blank',{'ingredient_text':'  '},400),('symbols',{'ingredient_text':'='},400),
                 ('unreviewed_ocr',{'ingredient_text':'Milk','source':'ocr'},400),
                 ('control',{'ingredient_text':'Milk\x00'},400),('long_tokens',{'ingredient_text':'milk '*300},400),
                 ('too_many_chars',{'ingredient_text':'x'*4001},422),('missing',{},422),
                 ('number',{'ingredient_text':123},422),('research_mode',{'ingredient_text':'Milk','mode':'combined'},422)]
    errors = []
    for name,payload,expected in negatives:
        code, body, _ = call('/predict',payload)
        require(code==expected and body.get('status')==('invalid_ingredients' if expected==400 else 'invalid_request'), name+' wrong response')
        errors.append({'case':name,'http':code,'status':body['status']})
        print('PASS:',name,code)
    code, body, _ = call('/products/00000105000011')
    require(code==200 and body==bodies['known'], 'Lookup changed after prediction')
    report = {'experiment':'I8.6b','versions':{p:version(p) for p in ('fastapi','pydantic','torch','transformers','uvicorn')},
              'product_cases_passed':11,'health_passed':True,'predictions':predictions,'reviewed_ocr_fixture':reviewed,
              'invalid_cases':errors,'post_prediction_lookup_unchanged':True,
              'limitations':['Four illustrative baseline examples; score tolerance 0.1 percentage point, not independent accuracy evaluation.',
                             'Single serial HTTP call per example; no controlled latency benchmark.',
                             'Reviewed OCR example is synthetic text, not real image OCR; further input validation pending.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:
        json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
    print('SUCCESS: prediction API, 9 negative cases and 11 original lookup cases verified.')


if __name__ == '__main__':
    main()
