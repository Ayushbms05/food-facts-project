"""Offline comparison of existing basic guard and proposed quality guard."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from .ingredient_quality import assess_ingredients, RULE_VERSION
from .nova_service import validate_text, InputError

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/integration/evaluation_i6_6/i6_6_ocr_metrics.json'
OUT = ROOT / 'data/backend/i8_7a_ingredient_quality_report.json'
# Designed fixtures, not independent accuracy test data.
CASES = [
 ('milk','Milk','manual',False,'accept'), ('salt','Salt','manual',False,'accept'),
 ('rice','Rice','manual',False,'accept'), ('tea','Tea','ocr',True,'accept'),
 ('oil','Oil','ocr',True,'accept'), ('rai','Rai','ocr',True,'accept'),
 ('dal','Dal','manual',False,'accept'), ('besan','Besan','ocr',True,'accept'),
 ('regional','Maida, jaggery, ghee','manual',False,'accept'),
 ('heading_valid','Ingredients: Milk & Milk Solids','ocr',True,'accept'),
 ('additive','Wheat flour, raising agent (INS 500(ii)), salt','manual',False,'accept'),
 ('blank','','ocr',True,'reject'), ('symbols','=','ocr',True,'reject'),
 ('numbers','12345','manual',False,'reject'), ('control','Milk\x00','manual',False,'reject'),
 ('unreviewed','Milk','ocr',False,'reject'), ('heading','Ingredients','ocr',True,'reject'),
 ('heading_punct','Ingredients: 123','ocr',True,'reject'),
 ('fragment','aoa','ocr',True,'review'), ('heading_fragment','ingredients\n\nAM','ocr',True,'review'),
 ('unknown_short','xyz','manual',False,'review'),
 ('nutrition','Nutrition Facts\nEnergy 100 kcal\nProtein 5 g','ocr',True,'review'),
 ('mixed','Ingredients: Milk\nBest before: 12/2027','ocr',True,'review'),
 ('nonenglish','दूध','manual',False,'review'), # short fragment guard is not language detection
]


def basic(text, source, reviewed):
    try:
        validate_text(text, source, reviewed)
        return 'accept'
    except InputError:
        return 'reject'


def synthetic():
    rows=[]
    for name,text,source,reviewed,expected in CASES:
        result=assess_ingredients(text,source,reviewed)
        assert result['decision']==expected, (name,result,expected)
        rows.append({'case':name,'expected':expected,'basic':basic(text,source,reviewed),**result})
    return rows


def main():
    synth=synthetic()
    print(f'PASS: {len(synth)} designed fixtures')
    if not SOURCE.is_file():
        raise FileNotFoundError(f'Prior OCR results required: {SOURCE}; no need to rerun OCR if you have the saved file.')
    raw=SOURCE.read_bytes(); data=json.loads(raw)
    assert data['experiment']=='I6.6'
    rows=data['per_image']
    codes={'0760851638226','8901063367258','8901088000468','8904083301394','8906112665354','89080955'}
    assert len(rows)==12 and {(r['barcode'],r['psm']) for r in rows}=={(c,p) for c in codes for p in (6,11)}
    refs={}; results=[]
    for r in rows:
        code=r['barcode']; text=r['ocr_text']; ref=r['reference_text']
        assert isinstance(text,str) and isinstance(ref,str) and not r['ocr_error'], (code,r['psm'])
        assert refs.setdefault(code,ref)==ref
        # Deliberately simulate reviewed=True to isolate extra quality checks.
        result=assess_ingredients(text,'ocr',True)
        results.append({'barcode':code,'psm':r['psm'],'ocr_status':r['ocr_status'],
                        'wer_pct':r['wer_pct'],'prior_prediction':r['ocr_prediction'],
                        'prior_confidence_pct':r['ocr_confidence_pct'],
                        'basic':basic(text,'ocr',True),**result})
    expected_bad={('0760851638226',6):'review',('8901063367258',6):'reject',
                  ('89080955',6):'reject',('89080955',11):'review'}
    for r in results:
        k=(r['barcode'],r['psm'])
        if k in expected_bad: assert r['decision']==expected_bad[k], r
        if r['ocr_status']=='empty_text': assert r['decision']=='reject',r
    references=[{'barcode':c,**assess_ingredients(t,'ocr',True)} for c,t in sorted(refs.items())]
    assert all(r['decision']=='accept' for r in references), references
    report={'experiment':'I8.7a','rule_version':RULE_VERSION,'source_sha256':hashlib.sha256(raw).hexdigest(),
            'synthetic_passed':len(synth),'synthetic':synth,'ocr_attempts':len(results),
            'basic_ocr_counts':dict(Counter(r['basic'] for r in results)),
            'proposed_ocr_counts':dict(Counter(r['decision'] for r in results)),
            'ocr_cases':results,'reference_cases':references,
            'limitations':['Reuses six previously inspected images and known failure examples; not independent validation.',
                           'Reviewed=True simulated for offline comparison; not proof of actual review.',
                           'Short-fragment rule may flag legitimate regional or non-English text; allowlist incomplete.',
                           'Accept means no obvious issue, not verified ingredient completeness or correct NOVA.',
                           'No model inference, new OCR, database writes or API changes. Token cap remains in existing service.']}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open('x',encoding='utf-8') as f: json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k!='synthetic'},indent=2,ensure_ascii=False))
    print('SUCCESS: offline quality checks verified; API unchanged.')

if __name__=='__main__': main()
