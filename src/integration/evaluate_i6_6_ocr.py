"""Controlled ingredient-region OCR pilot; run as module from repository root."""
import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from .audit_bda_dl_overlap import code_key, text_key, read_split, FILES

ROOT = Path.cwd()
INPUT = ROOT/'data/integration/evaluation_i6_6_inputs'
OUT = ROOT/'data/integration/evaluation_i6_6'
SAMPLES = ROOT/'data/integration/evaluation_i6_3'
MODEL = ROOT/'data/deep_learning/models/distilbert_food_nova'
INDIA = ROOT/'data/integration/evaluation_i6_5/i6_5_india_metrics.json'
SOURCE = ROOT/'data/integration/evaluation_i6_4/i6_4_source_consistency.json'
MODES = (6, 11)
csv.field_size_limit(16*1024*1024)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def words(text):
    return re.findall(r'[^\W_]+',text.lower(),flags=re.UNICODE)

def distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1): cur.append(min(prev[j]+1,cur[-1]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]

def summarize(rows):
    result={}
    for psm in MODES:
        sub=[r for r in rows if r['psm']==psm]; ok=[r for r in sub if r['ocr_status']!='error']; pred=[r for r in sub if r['ocr_prediction'] is not None]
        agree=sum(r['prediction_agrees'] for r in pred)
        ref=sum(r['reference_words'] for r in ok)
        result[str(psm)]={'attempts':len(sub),'engine_errors':len(sub)-len(ok),'empty_outputs':sum(r['ocr_status']=='empty_text' for r in sub),'nonempty_predictions':len(pred),'reference_words_successful_engine_runs':ref,'word_errors_successful_engine_runs':sum(r['word_errors'] for r in ok),'pooled_wer_pct_successful_engine_runs':100*sum(r['word_errors'] for r in ok)/ref if ref else None,'prediction_agreements':agree,'agreement_pct_among_nonempty_predictions':100*agree/len(pred) if pred else None,'agreement_pct_all_attempts':100*agree/len(sub) if sub else None,'mean_confidence_change_pp_among_predictions':sum(r['confidence_change_pp'] for r in pred)/len(pred) if pred else None}
    return result

def overlap_sets():
    sampling=json.loads((SAMPLES/'i6_3_sampling_report.json').read_text())
    sets={}
    for name in FILES:
        rel=f'data/deep_learning/datasets/{name}.csv'
        if sha(ROOT/rel)!=sampling['input_sha256'][rel]: raise ValueError('DL dataset changed: '+name)
        _,c,t=read_split(name); sets['dl_'+name]=(c,t)
    for name in ('balanced_curated','natural_curated','india_evaluation','source_pairs'):
        path=(SOURCE.parent/'source_pairs.csv') if name=='source_pairs' else SAMPLES/(name+'.csv')
        expected=json.loads(SOURCE.read_text())['sample_sha256'] if name=='source_pairs' else sampling['sample_sha256'][path.name]
        if sha(path)!=expected: raise ValueError('Evaluation sample changed: '+name)
        codes,texts=set(),set()
        with path.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                codes.add(r['normalized_code'])
                for field in ('off_raw','off_clean','off_dl_structured','usda_raw','usda_clean') if name=='source_pairs' else ('off_raw','off_clean','off_dl_structured'):
                    t=text_key(r.get(field));
                    if t: texts.add(t)
        sets[name]=(codes,texts)
    return sets

def overlap(code,text,sets):
    c=code_key(code); t=text_key(text)
    return {'barcode_matches':[n for n,(cs,ts) in sets.items() if c in cs], 'exact_text_matches':[n for n,(cs,ts) in sets.items() if t and t in ts]}

def run_ocr(path,psm):
    start=time.perf_counter()
    try:
        p=subprocess.run(['tesseract',str(path),'stdout','-l','eng','--psm',str(psm)],capture_output=True,text=True,timeout=120)
        if p.returncode: raise RuntimeError(p.stderr.strip())
        t=p.stdout.strip(); return ('text_detected' if t else 'empty_text',t,'',time.perf_counter()-start)
    except (OSError,subprocess.SubprocessError,RuntimeError) as exc:
        return 'error','',str(exc),time.perf_counter()-start

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--references-reviewed',action='store_true',help='Confirm the six crop/reference pairs were visually checked before OCR/prediction.'); args=parser.parse_args()
    if not args.references_reviewed: parser.error('Review the references before confirming --references-reviewed.')
    if OUT.exists() and any(OUT.iterdir()): raise FileExistsError('Existing outputs; archive before repeating.')
    manifest=json.loads((INPUT/'manifest.json').read_text()); included=[r for r in manifest['rows'] if r['status']=='included']; excluded=[r for r in manifest['rows'] if r['status']=='excluded']
    if len(included)!=6 or len(excluded)!=2 or len({r['barcode'] for r in manifest['rows']})!=8: raise ValueError('Unexpected manifest.')
    for r in manifest['rows']:
        original=ROOT/'data/images/balanced_full'/r['original_filename']
        if sha(original)!=r['original_sha256']: raise ValueError('Original image changed: '+r['barcode'])
    for r in included:
        for field in ('crop','reference'):
            if sha(INPUT/r[field])!=r[field+'_sha256']: raise ValueError('Input hash mismatch: '+r['barcode']+' '+field)
        if not words((INPUT/r['reference']).read_text()): raise ValueError('Empty reference')
    india=json.loads(INDIA.read_text()); hashes={p.name:sha(p) for p in sorted(MODEL.iterdir()) if p.is_file()}
    if hashes!=india['model_files_sha256'] or india['experiment']!='I6.5': raise ValueError('Model differs from I6.5')
    version=subprocess.run(['tesseract','--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    langs=subprocess.run(['tesseract','--list-langs'],capture_output=True,text=True,check=True).stdout.splitlines()
    if 'eng' not in [x.strip() for x in langs]: raise ValueError('Tesseract eng language unavailable')
    sets=overlap_sets()
    from src.deep_learning.inference.predict import load_model,predict_nova
    import torch,transformers
    tokenizer,model=load_model(); model.eval()
    def predict(text):
        r=predict_nova(text,tokenizer,model); p=int(r['predicted_nova']); c=float(r['confidence']); probs={k:float(v) for k,v in r['class_probabilities'].items()}
        if p not in (1,2,3,4) or not math.isfinite(c) or not 0<=c<=100 or len(probs)!=4 or not all(math.isfinite(v) and 0<=v<=100 for v in probs.values()) or abs(sum(probs.values())-100)>0.1: raise ValueError('Invalid predictor output')
        return p,c,probs
    details=[]; calls=0; start=time.perf_counter()
    for r in included:
        code=r['barcode']; ref=(INPUT/r['reference']).read_text().strip(); rw=words(ref); rp,rc,rprobs=predict(ref); calls+=1
        for psm in MODES:
            status,text,error,elapsed=run_ocr(INPUT/r['crop'],psm); ow=words(text); edits=distance(rw,ow) if status!='error' else None
            op,oc,oprobs=(None,None,None)
            if status=='text_detected': op,oc,oprobs=predict(text); calls+=1
            row={'barcode':code,'recorded_off_nova':r['recorded_nova'],'psm':psm,'reference_text':ref,'ocr_text':text,'ocr_status':status,'ocr_error':error,'ocr_seconds':elapsed,'reference_words':len(rw),'ocr_words':len(ow),'word_errors':edits,'wer_pct':100*edits/len(rw) if edits is not None else None,'reference_prediction':rp,'reference_confidence_pct':rc,'reference_probabilities':rprobs,'ocr_prediction':op,'ocr_confidence_pct':oc,'ocr_probabilities':oprobs,'prediction_agrees':op==rp if op is not None else None,'confidence_change_pp':oc-rc if oc is not None else None,'reference_overlap':overlap(code,ref,sets),'ocr_overlap':overlap(code,text,sets)}
            details.append(row); print(f'{code} PSM {psm}: {status}; WER={row["wer_pct"]}; reference NOVA={rp}; OCR NOVA={op}',flush=True)
    report={'experiment':'I6.6','method':'Visually isolated ingredient regions; AI-assisted references reviewed by user before inference; identical unenhanced crops for PSM 6 and 11; headings retained when visible; unchanged model','manifest_sha256':sha(INPUT/'manifest.json'),'model_files_sha256':hashes,'india_report_sha256':sha(INDIA),'tesseract_version':version,'language':'eng','device':str(next(model.parameters()).device),'torch_version':torch.__version__,'transformers_version':transformers.__version__,'included_images':len(included),'excluded_images':excluded,'included_recorded_nova_counts':dict(Counter(r['recorded_nova'] for r in included)),'prediction_calls_measured':calls,'evaluation_elapsed_seconds':time.perf_counter()-start,'by_psm':summarize(details),'per_image':details,'limitations':['Six selected readable images; not balanced or representative. Crop chosen manually; not end-to-end label detection.','Reference text model prediction is a consistency baseline, not verified NOVA truth.','Prior exact matches reported, not excluded; near duplicates not screened; not an independent generalization test.','WER uses case-insensitive word tokens, ignores punctuation; empty output counts as 100% WER, engine errors excluded from pooled WER and counted separately.','Confidence is uncalibrated; Tesseract is not Android ML Kit. No training.']}
    OUT.mkdir(parents=True,exist_ok=True)
    out=OUT/'ocr_predictions.jsonl'; out.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in details),encoding='utf-8'); report['predictions_sha256']=sha(out)
    (OUT/'i6_6_ocr_metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='per_image'},indent=2,ensure_ascii=False,allow_nan=False)); print('Results saved:',OUT); print('SUCCESS: I6.6 OCR consistency pilot complete. No training performed.')

if __name__=='__main__': main()
