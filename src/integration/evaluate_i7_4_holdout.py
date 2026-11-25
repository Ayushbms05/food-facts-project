"""I7.4: evaluate frozen output adjustments on untouched holdout. No fitting.
Run from repository root: HF_HUB_OFFLINE=1 python -m src.integration.evaluate_i7_4_holdout
"""
import csv
import json
import tempfile
import time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from .fit_i7_3_adjustments import (BASE, REPORT, SAMPLE, MODEL, TRAIN, DEV_SHA,
    HOLD_SHA, TRAIN_SHA, MODEL_HASHES, LABELS, sha256, require, read, assess, log_probs)

DEV_OUT = BASE / 'evaluation_i7_3'
HOLD = SAMPLE.parent / 'holdout.csv'
OUT = BASE / 'evaluation_i7_4'
FROZEN_SHA = '5cab4b4772f2c0c7edc93b8dc39c10cdf55a155c4c5c9df4c8806e79e20956b0'
LOGITS_SHA = '1aef85000ff9884670ec1972fdca64c9b7e06b0e25fbe335e32a660625231c4f'
PARAMETERS = {'original': (0.0, 1.0), 'temperature_only': (0.0, 1.2877402988474436),
              'combined': (0.3, 1.326715596542784)}
QUOTAS = {1:500,2:80,3:1000,4:3500}
# Frozen before holdout inference. The gate compares the same reference metrics
# against holdout baseline, not the development baseline.
GATE = {'minimum_standardized_macro_f1_gain':0.005,
        'maximum_standardized_accuracy_drop_pp':0.5,
        'maximum_nova2_recall_drop':0.10,
        'minimum_standardized_nll_improvement':0.0001}


def text_keys(rows):
    return {' '.join(r.get(k,'').lower().split()) for r in rows
            for k in ('off_raw','off_clean','off_dl_structured') if r.get(k,'').strip()}


def validate_rows(rows, cohort):
    require(Counter(int(r['nova_group']) for r in rows)==QUOTAS, 'Unexpected quotas')
    for k in ('normalized_code','off_text_group_id'):
        require(all(r[k].strip() for r in rows) and len({r[k] for r in rows})==len(rows), f'Invalid {k}')
    require(all(r['evaluation_cohort']==cohort and r['off_dl_structured'].strip() for r in rows), 'Wrong cohort or missing text')


def collect(rows):
    import torch
    import transformers
    from src.deep_learning.inference.predict import load_model, predict_nova
    tokenizer, model = load_model(); model.eval(); device=next(model.parameters()).device
    def forward(text):
        inputs=tokenizer(text,return_tensors='pt',truncation=True,padding='max_length',max_length=256)
        with torch.inference_mode():
            z=model(**{k:v.to(device) for k,v in inputs.items()}).logits[0].detach().cpu().double().numpy()
        require(z.shape==(4,) and np.isfinite(z).all(),'Invalid logits')
        return z
    for i in np.linspace(0,len(rows)-1,20,dtype=int):
        text=rows[i]['off_dl_structured']; z=forward(text)
        old=predict_nova(text,tokenizer,model)
        p=np.exp(z-logsumexp(z))*100
        old_p=np.array([old['class_probabilities'][f'NOVA {k}'] for k in LABELS])
        require(int(old['predicted_nova'])==int(z.argmax())+1 and np.max(np.abs(p-old_p))<=.02,'Predictor parity failed')
    start=time.perf_counter(); result=[]
    for i,r in enumerate(rows,1):
        result.append(forward(r['off_dl_structured']))
        if i%250==0 or i==len(rows): print(f'Holdout: {i:,}/{len(rows):,} logits collected',flush=True)
    return np.asarray(result), {'device':str(device),'torch_version':str(torch.__version__),'transformers_version':transformers.__version__, 'elapsed_seconds':time.perf_counter()-start,'predictor_parity_checks':20}


def evaluate(z,y,w,ratio):
    results={name:assess(z,y,w,a,t,ratio) for name,(a,t) in PARAMETERS.items()}
    probs={name:np.exp(log_probs(z,a,t,ratio)) for name,(a,t) in PARAMETERS.items()}
    pred={k:v.argmax(axis=1) for k,v in probs.items()}
    require(np.array_equal(pred['original'],pred['temperature_only']), 'Temperature unexpectedly changed class')
    b=results['original']['prior_standardized']; c=results['combined']['prior_standardized']; t=results['temperature_only']['prior_standardized']
    gains={'standardized_accuracy_pp':c['accuracy_pct']-b['accuracy_pct'], 'standardized_macro_f1':c['macro_f1']-b['macro_f1'], 'nova2_recall':c['per_class']['2']['recall']-b['per_class']['2']['recall'], 'standardized_nll_improvement':b['nll']-c['nll']}
    checks={'macro_f1_gain':gains['standardized_macro_f1']>=GATE['minimum_standardized_macro_f1_gain']-1e-12,
            'accuracy_preserved':gains['standardized_accuracy_pp']>=-GATE['maximum_standardized_accuracy_drop_pp']-1e-12,
            'nova2_recall_preserved':gains['nova2_recall']>=-GATE['maximum_nova2_recall_drop']-1e-12,
            'nll_improved':gains['standardized_nll_improvement']>=GATE['minimum_standardized_nll_improvement']-1e-12}
    temp_ok=b['nll']-t['nll']>=GATE['minimum_standardized_nll_improvement']-1e-12
    decision='combined' if all(checks.values()) else ('temperature_only' if temp_ok else 'original')
    first=pred['original']==y; second=pred['combined']==y
    paired={'both_correct':int((first&second).sum()),'original_only_correct':int((first&~second).sum()),'combined_only_correct':int((~first&second).sum()),'both_wrong':int((~first&~second).sum()),'changed_predictions':int((pred['original']!=pred['combined']).sum())}
    return results,probs,{'combined_checks':checks,'combined_gains':gains,'temperature_only_nll_improvement':b['nll']-t['nll'],'temperature_only_passes':bool(temp_ok),'recommendation_for_screened_curated_structured_scope':decision},paired


def main():
    names=['holdout_predictions.csv','i7_4_holdout_metrics.json']
    require(not OUT.exists() or not any(OUT.iterdir()),'Holdout results already exist; preserve the first evaluation.')
    cfg_path=DEV_OUT/'frozen_adjustment.json'; require(sha256(cfg_path)==FROZEN_SHA,'Frozen adjustment changed')
    cfg=json.loads(cfg_path.read_text()); rep=json.loads(REPORT.read_text()); devrep=json.loads((DEV_OUT/'i7_3_development_metrics.json').read_text())
    require(cfg['experiment']=='I7.3' and cfg['input_format']=='off_dl_structured' and cfg['labels']==LABELS,'Wrong frozen config')
    require((cfg['alpha'],cfg['temperature'])==PARAMETERS['combined'],'Frozen parameters changed')
    require(devrep['frozen_adjustment_sha256']==FROZEN_SHA and devrep['development_logits_sha256']==LOGITS_SHA==cfg['development_logits_sha256']==sha256(DEV_OUT/'development_logits.csv'),'Development artifact mismatch')
    require((devrep['temperature_only']['alpha'],devrep['temperature_only']['temperature'])==PARAMETERS['temperature_only'],'Temperature-only setting changed')
    require(sha256(REPORT)==cfg['sampling_report_sha256'] and rep['development_holdout_group_overlap']==0,'Sampling report changed')
    require(sha256(SAMPLE)==DEV_SHA==cfg['development_sha256']==rep['sample_sha256']['development.csv'],'Development sample changed')
    require(sha256(HOLD)==HOLD_SHA==cfg['expected_holdout_sha256']==rep['sample_sha256']['holdout.csv'],'Holdout sample changed')
    require(sha256(TRAIN)==TRAIN_SHA,'Training file changed')
    require(cfg['model_files_sha256']==MODEL_HASHES=={p.name:sha256(p) for p in MODEL.iterdir() if p.is_file()},'Checkpoint changed')
    rows=read(HOLD); dev=read(SAMPLE); validate_rows(rows,'holdout'); validate_rows(dev,'development')
    for key in ('normalized_code','off_text_group_id'):
        require(not ({r[key] for r in rows}&{r[key] for r in dev}),f'Overlap: {key}')
    require(not (text_keys(rows)&text_keys(dev)),'Exact ingredient text overlap')
    y=np.array([int(r['nova_group'])-1 for r in rows]); train=np.array(cfg['train_prior']); target=np.array(cfg['target_prior'])
    require(np.allclose(train,np.array([1742,1758,1767,1751])/7018) and np.allclose(target,[7335/200327,695/200327,34074/200327,158223/200327]),'Unexpected priors')
    weights=np.array([rep['class_standardization_weights_both_splits'][str(k)] for k in LABELS]); require(np.allclose(weights,target/(np.array([QUOTAS[k] for k in LABELS])/len(rows))),'Unexpected weights')
    z,run=collect(rows); results,probs,decision,paired=evaluate(z,y,weights[y],np.log(target/train))
    report={'experiment':'I7.4','method':'Single frozen holdout evaluation, no fitting or parameter search', 'sample_rows':len(rows),'holdout_sha256':HOLD_SHA,'development_sha256':DEV_SHA,'frozen_adjustment_sha256':FROZEN_SHA,'development_metrics_sha256':sha256(DEV_OUT/'i7_3_development_metrics.json'),'model_files_sha256':MODEL_HASHES,'verification':{'development_holdout_code_group_text_overlap':0},'run':run,'class_standardization_weights':weights.tolist(),'parameters':PARAMETERS,'metrics':results,'paired_original_vs_combined':paired,'predeclared_acceptance_gate':GATE,'decision':decision,'limitations':['OFF labels not independently verified; near duplicates possible.','Stratified curated holdout; standardized estimates target screened curated groups, not retail prevalence.','Only 80 recorded NOVA 2 examples; rare-class estimates uncertain.','Structured input only; no fresh India or OCR validation.','Acceptance criteria are engineering choices; no tuning on holdout.']}
    OUT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUT) as d:
        tmp=Path(d)
        with (tmp/names[0]).open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f); writer.writerow(['normalized_code','off_text_group_id','nova_group',*[f'logit_nova{k}' for k in LABELS],*[field for name in PARAMETERS for field in [f'{name}_prediction',f'{name}_confidence_pct',*[f'{name}_prob_nova{k}' for k in LABELS]]]])
            for i,r in enumerate(rows):
                writer.writerow([r['normalized_code'],r['off_text_group_id'],r['nova_group'],*z[i].tolist(),*[v for name in PARAMETERS for v in [int(probs[name][i].argmax())+1,float(probs[name][i].max()*100),*probs[name][i].tolist()]]])
        report['predictions_sha256']=sha256(tmp/names[0]); (tmp/names[1]).write_text(json.dumps(report,indent=2)+'\n')
        for n in names: (tmp/n).replace(OUT/n)
    print(json.dumps(report,indent=2)); print('SUCCESS: I7.4 frozen holdout evaluation complete. No fitting or weight changes.')

if __name__=='__main__': main()
