"""I7.1: descriptive audit of frozen I6.3c and I6.5 predictions, no inference/tuning.
Run from repository root: python -m src.integration.analyze_i7_errors
"""
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path('data/integration')
OUT = ROOT / 'evaluation_i7_1'
COHORTS = {
    'natural': ('natural_curated.csv', 'evaluation_i6_3c/i6_3c_natural_metrics.json', 'natural_predictions.csv', '92936831b02322f4a91d216108eb632be36c507585b52e9d241bb639b30b4f40', '16393599a553b95c0e602e7e23459cf241021e03263cd28370d656a447eeb2e9'),
    'india': ('india_evaluation.csv', 'evaluation_i6_5/i6_5_india_metrics.json', 'india_predictions.csv', 'c1602b19575634e152fbc2322bc32c29a85c32473ad134fc0e4514cb79bb2c45', '90888090edc0999821859f894ea993aae46bcdab427eb9c0d9441e2e06c3d07e'),
}
BINS = [(0,50),(50,70),(70,80),(80,90),(90,95),(95,100)]
csv.field_size_limit(16 * 1024 * 1024)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def stats(rows):
    n=len(rows)
    correct=sum(r['correct'] for r in rows)
    return {'rows':n,'correct':correct,'accuracy_pct':100*correct/n if n else None,
            'mean_confidence_pct':mean(r['confidence'] for r in rows) if n else None}


def summarize(rows):
    matrix=[[sum(r['actual']==a and r['prediction']==b for r in rows) for b in range(1,5)] for a in range(1,5)]
    fp=[r for r in rows if r['prediction']==2 and r['actual']!=2]
    tp=[r for r in rows if r['prediction']==r['actual']==2]
    fn=[r for r in rows if r['actual']==2 and r['prediction']!=2]
    def bins(items):
        return {f'{lo}-{hi}':stats([r for r in items if lo<=r['confidence'] and (r['confidence']<hi or hi==100)]) for lo,hi in BINS}
    return {'overall':stats(rows),'confusion_matrix':matrix,
            'confidence_bins':bins(rows),
            'high_confidence_errors_ge_90':sum(not r['correct'] and r['confidence']>=90 for r in rows),
            'nova2':{'true_positive':len(tp),'false_positive':len(fp),'false_negative':len(fn),
                'false_positives_by_actual':dict(Counter(r['actual'] for r in fp)),
                'predicted_2_confidence_bins':bins(tp+fp),
                'false_positive_mean_confidence_pct':mean(r['confidence'] for r in fp) if fp else None,
                'true_positive_mean_confidence_pct':mean(r['confidence'] for r in tp) if tp else None,
                'false_positive_ge_90':sum(r['confidence']>=90 for r in fp)},
            'nova4_to_3':stats([r for r in rows if r['actual']==4 and r['prediction']==3]),
            'nova4_to_2':stats([r for r in rows if r['actual']==4 and r['prediction']==2])}


def join(sample,predictions,cohort):
    index={r['normalized_code']:r for r in sample}
    if len(index)!=len(sample) or len(predictions)!=len(sample):
        raise ValueError('Duplicate/missing sample rows')
    result=[]; seen=set()
    for p in predictions:
        k=p['normalized_code']; s=index[k]
        if k in seen or any(s[f]!=p[f] for f in ('code','off_text_group_id','nova_group')):
            raise ValueError('Prediction/sample identity mismatch')
        seen.add(k)
        actual=int(s['nova_group'])
        if actual not in range(1,5): raise ValueError('Invalid label')
        for fmt in ('raw','structured'):
            field='off_raw' if fmt=='raw' else 'off_dl_structured'
            pred=int(p[f'{fmt}_prediction']); conf=float(p[f'{fmt}_confidence_pct'])
            probs=[float(p[f'{fmt}_p_nova{i}_pct']) for i in range(1,5)]
            if pred not in range(1,5) or not all(math.isfinite(x) and 0<=x<=100 for x in [conf,*probs]) or abs(sum(probs)-100)>0.1 or abs(conf-probs[pred-1])>0.1 or max(probs)-conf>0.1:
                raise ValueError('Invalid prediction probabilities')
            if int(p[f'{fmt}_correct'])!=int(pred==actual): raise ValueError('Wrong correctness flag')
            result.append({'cohort':cohort,'format':fmt,'code':s['code'],'normalized_code':k,'group_id':s['off_text_group_id'],'actual':actual,'prediction':pred,'correct':pred==actual,'confidence':conf,'p_nova1':probs[0],'p_nova2':probs[1],'p_nova3':probs[2],'p_nova4':probs[3], 'text':s[field]})
    return result


def main():
    if OUT.exists() and any(OUT.iterdir()): raise FileExistsError('Archive existing I7.1 output first.')
    report={'experiment':'I7.1','method':'Descriptive error audit of already-inspected frozen predictions; no new model calls or tuning.', 'inputs':{},'cohorts':{},'review_sampling_seed':42,'limitations':['Confidence bins are descriptive, not validated thresholds or calibrated probabilities.','Recorded OFF labels may be wrong; error cases need manual investigation.','India and natural curated cohorts differ in selection and class distribution.','Review CSV intentionally enriched for errors; not prevalence estimates.']}
    review=[]; rng=random.Random(42)
    for cohort,(name,rel,pname,sh,ph) in COHORTS.items():
        sample=ROOT/'evaluation_i6_3'/name; rp=ROOT/rel; pp=rp.parent/pname
        prior=json.loads(rp.read_text())
        if sha(sample)!=sh or sha(pp)!=ph or prior['sample_sha256']!=sh or prior['predictions_sha256']!=ph:
            raise ValueError(f'Frozen evidence mismatch: {cohort}')
        rows=join(read_csv(sample),read_csv(pp),cohort)
        report['inputs'][cohort]={'sample_sha256':sh,'predictions_sha256':ph,'report_sha256':sha(rp)}
        report['cohorts'][cohort]={}
        for fmt in ('raw','structured'):
            subset=[r for r in rows if r['format']==fmt]; summary=summarize(subset)
            if summary['confusion_matrix']!=prior['metrics'][fmt]['confusion_matrix_rows_actual_cols_predicted']:
                raise ValueError('Recomputed confusion matrix disagrees with original report')
            report['cohorts'][cohort][fmt]=summary
            if fmt!='structured': continue
            # Sample up to ten from each mutually exclusive diagnostic category; no ranking by confidence.
            buckets=defaultdict(list)
            for r in subset:
                a,b=r['actual'],r['prediction']
                if b==2: label='nova2_true_positive' if a==2 else f'nova2_false_positive_actual_{a}'
                elif a==2: label='nova2_false_negative'
                elif a==4 and b==3: label='nova4_to_3'
                elif a==4 and b==4: label='nova4_correct_control'
                else: continue
                buckets[label].append(r)
            for label,items in sorted(buckets.items()):
                for r in rng.sample(items,min(10,len(items))):
                    review.append({'review_category':label,**r,'manual_observation':'','reference_label_questionable':'','review_notes':''})
        print(f'{cohort}: audited {len(rows)//2:,} products',flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    review_path=OUT/'error_review_sample.csv'
    with review_path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(review[0])); w.writeheader(); w.writerows(review)
    report['review_rows']=len(review); report['review_sample_sha256']=sha(review_path)
    (OUT/'i7_1_error_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,allow_nan=False))
    print('SUCCESS: I7.1 descriptive audit complete. No inference or tuning performed.')

if __name__=='__main__': main()
