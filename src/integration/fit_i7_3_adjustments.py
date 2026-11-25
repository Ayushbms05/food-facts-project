"""I7.3: fit output-only adjustment on frozen development data. Never reads holdout.csv.
Run: HF_HUB_OFFLINE=1 python -m src.integration.fit_i7_3_adjustments
"""
import csv
import hashlib
import json
import time
import tempfile
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

BASE = Path('data/integration')
SAMPLE = BASE / 'evaluation_i7_2/development.csv'
REPORT = SAMPLE.parent / 'i7_2_holdout_report.json'
OUT = BASE / 'evaluation_i7_3'
MODEL = Path('data/deep_learning/models/distilbert_food_nova')
TRAIN = Path('data/deep_learning/datasets/train.csv')
DEV_SHA = '0fc4c45c2c4bba6c3910f83da2e30ea304deaa6552c382bb06a3aaf38b626c6a'
HOLD_SHA = '6523cfa3fb11bc91df1d8243d4fe11e51e0f92a0cff29a0e0398f3766a6b24ea'
TRAIN_SHA = '2496f8551aa44c659ab76d2ce5405856e2b0410bfd0fbfb3dfcc92be19d9b3eb'
MODEL_HASHES = {'config.json':'d3b802f47aed03250232fbfe367bf58e34393bc0831b5853a26a2099268aa829','model.safetensors':'d688e4f77855d2797149caf80006cfd9859b0f92424e7da6d2feb48236cc371b','tokenizer.json':'0d3aef594edd5f9b53e7f814277a9171dc70ff93eb66bda6e01f7aa53997d963','tokenizer_config.json':'e1c2a61a99bda00f6c55303a210b30e2f92dcf8b555e215812e2eb583e177ffd'}
LABELS = [1,2,3,4]
ALPHAS = [i/10 for i in range(11)]
csv.field_size_limit(16*1024*1024)

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def require(ok,msg):
    if not ok: raise ValueError(msg)

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def log_probs(logits, alpha, temperature, ratio):
    # Apply class bias BEFORE temperature so calibration preserves the adjusted argmax.
    z=(logits+alpha*ratio)/temperature
    return z-logsumexp(z,axis=1,keepdims=True)

def nll(lp,y,w):
    return float(np.average(-lp[np.arange(len(y)),y],weights=w))

def fit_temperature(logits,y,w,alpha,ratio):
    objective=lambda t:nll(log_probs(logits,alpha,t,ratio),y,w)
    fit=minimize_scalar(objective,bounds=(0.25,5.0),method='bounded',options={'xatol':1e-6})
    require(fit.success,'Temperature optimization failed')
    t=float(fit.x)
    return t if objective(1.0)-objective(t)>0.0001 else 1.0

def metrics(lp,y,w):
    p=np.exp(lp); pred=np.argmax(p,axis=1); conf=p.max(axis=1); correct=pred==y
    cm=np.zeros((4,4)); np.add.at(cm,(y,pred),w)
    tp=np.diag(cm); actual=cm.sum(axis=1); predicted=cm.sum(axis=0)
    precision=np.divide(tp,predicted,out=np.zeros(4),where=predicted>0)
    recall=np.divide(tp,actual,out=np.zeros(4),where=actual>0)
    f1=np.divide(2*tp,actual+predicted,out=np.zeros(4),where=(actual+predicted)>0)
    bins=np.minimum((conf*10).astype(int),9); ece=0.0
    for i in range(10):
        mask=bins==i
        if mask.any(): ece+=w[mask].sum()/w.sum()*abs(np.average(conf[mask],weights=w[mask])-np.average(correct[mask],weights=w[mask]))
    return {'accuracy_pct':float(100*np.average(correct,weights=w)),'macro_f1':float(f1.mean()),'balanced_accuracy_pct':float(100*recall.mean()),'nll':nll(lp,y,w),'multiclass_brier':float(np.average(np.sum((p-np.eye(4)[y])**2,axis=1),weights=w)), 'ece_10_equal_width':float(ece),'mean_confidence_pct':float(100*np.average(conf,weights=w)), 'per_class':{str(k):{'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support_weight':float(actual[i]),'predicted_weight':float(predicted[i])} for i,k in enumerate(LABELS)},'confusion_matrix':cm.tolist()}

def assess(logits,y,w,alpha,t,ratio):
    lp=log_probs(logits,alpha,t,ratio)
    return {'alpha':alpha,'temperature':t,'unweighted':metrics(lp,y,np.ones(len(y))), 'prior_standardized':metrics(lp,y,w)}

def select(logits,y,w,ratio):
    baseline=assess(logits,y,w,0.0,1.0,ratio)
    candidates=[assess(logits,y,w,a,fit_temperature(logits,y,w,a,ratio),ratio) for a in ALPHAS]
    b=baseline['prior_standardized']; b2=b['per_class']['2']['recall']
    recall_floor=max(0.0,b2-0.10)
    accuracy_floor=b['accuracy_pct']-0.5
    # Fixed before inference: preserve rare-class recall and overall accuracy; require meaningful macro-F1 gain.
    eligible=[]
    for c in candidates:
        m=c['prior_standardized']
        c['passes_constraints']=bool(m['per_class']['2']['recall']+1e-12>=recall_floor and m['accuracy_pct']+1e-12>=accuracy_floor)
        if c['passes_constraints']: eligible.append(c)
    best=max(eligible,key=lambda c:(c['prior_standardized']['macro_f1'],-c['alpha']))
    if best['prior_standardized']['macro_f1']-b['macro_f1']<0.005: best=candidates[0]
    rule={'objective':'maximize prior-standardized macro F1; ties prefer smaller alpha','minimum_nova2_recall':recall_floor,'minimum_standardized_accuracy_pct':accuracy_floor,'minimum_macro_f1_gain_for_nonzero_alpha':0.005,'temperature_objective':'weighted NLL, bounds [0.25,5], retain T=1 if gain <=0.0001','alpha_grid':ALPHAS}
    return baseline,candidates,best,rule

def collect_logits(rows):
    import torch
    import transformers
    from src.deep_learning.inference.predict import load_model,predict_nova
    tokenizer,model=load_model(); model.eval(); device=next(model.parameters()).device
    def forward(text):
        inputs=tokenizer(text,return_tensors='pt',truncation=True,padding='max_length',max_length=256)
        inputs={k:v.to(device) for k,v in inputs.items()}
        with torch.inference_mode(): z=model(**inputs).logits[0].detach().cpu().double().numpy()
        require(z.shape==(4,) and np.isfinite(z).all(),'Invalid model logits')
        return z
    # Verify class mapping and tokenization against the existing public predictor.
    for idx in np.linspace(0,len(rows)-1,20,dtype=int):
        text=rows[idx]['off_dl_structured']; z=forward(text); p=np.exp(z-logsumexp(z))*100
        old=predict_nova(text,tokenizer,model)
        old_p=np.array([old['class_probabilities'][f'NOVA {k}'] for k in LABELS],float)
        require(int(old['predicted_nova'])==int(np.argmax(z))+1 and np.max(np.abs(p-old_p))<=0.02,'Direct logits differ from original predictor. Stop and inspect predict.py.')
    start=time.perf_counter(); zs=[]
    for i,r in enumerate(rows,1):
        zs.append(forward(r['off_dl_structured']))
        if i%250==0 or i==len(rows): print(f'Development: {i:,}/{len(rows):,} logits collected',flush=True)
    return np.array(zs),{'device':str(device),'torch_version':torch.__version__,'transformers_version':transformers.__version__,'elapsed_seconds':time.perf_counter()-start,'predictor_parity_checks':20}

def main():
    names=['development_logits.csv','i7_3_development_metrics.json','frozen_adjustment.json']
    require(not any((OUT/n).exists() for n in names),'Results exist; do not overwrite a frozen decision.')
    rep=json.loads(REPORT.read_text()); require(rep['experiment']=='I7.2' and rep['development_holdout_group_overlap']==0,'Unexpected sampling report')
    require(sha256(SAMPLE)==DEV_SHA==rep['sample_sha256']['development.csv'] and rep['sample_sha256']['holdout.csv']==HOLD_SHA,'Sample manifest changed')
    require(sha256(TRAIN)==TRAIN_SHA,'Original training file changed')
    require({p.name:sha256(p) for p in MODEL.iterdir() if p.is_file()}==MODEL_HASHES,'Original checkpoint changed')
    rows=read(SAMPLE); counts=Counter(int(r['nova_group']) for r in rows)
    require(counts=={1:500,2:80,3:1000,4:3500},'Unexpected development quotas')
    for k in ('normalized_code','off_text_group_id'):
        require(len({r[k] for r in rows})==len(rows) and all(r[k].strip() for r in rows),'Repeated or missing identifier')
    require(all(r['evaluation_cohort']=='development' and r['off_dl_structured'].strip() for r in rows),'Unexpected cohort or missing text')
    train=Counter(int(r['nova_group']) for r in read(TRAIN)); require(train=={1:1742,2:1758,3:1767,4:1751},'Unexpected training class counts')
    train_prior=np.array([train[k]/sum(train.values()) for k in LABELS]); target=np.array([rep['original_curated_reference_class_priors'][str(k)] for k in LABELS]); require(np.allclose(target,[7335/200327,695/200327,34074/200327,158223/200327],atol=1e-12),'Unexpected target priors')
    y=np.array([int(r['nova_group'])-1 for r in rows]); weights=np.array([rep['class_standardization_weights_both_splits'][str(k)] for k in LABELS]); require(np.allclose(weights,target/np.array([counts[k]/len(rows) for k in LABELS])),'Unexpected standardization weights')
    w=weights[y]; ratio=np.log(target/train_prior)
    logits,run=collect_logits(rows); baseline,candidates,chosen,rule=select(logits,y,w,ratio)
    config={'experiment':'I7.3','status':'development_selected_not_holdout_validated','formula':'softmax((logits + alpha * log(target_prior / train_prior)) / temperature)','input_format':'off_dl_structured','labels':LABELS,'alpha':chosen['alpha'],'temperature':chosen['temperature'],'train_prior':train_prior.tolist(),'target_prior':target.tolist(),'development_sha256':DEV_SHA,'expected_holdout_sha256':HOLD_SHA,'model_files_sha256':MODEL_HASHES,'sampling_report_sha256':sha256(REPORT),'selection_rule':rule}
    result={'experiment':'I7.3','sample_rows':len(rows),'sample_sha256':DEV_SHA,'model_files_sha256':MODEL_HASHES,'run':run,'train_prior':train_prior.tolist(),'target_prior':target.tolist(),'standardization_weights':weights.tolist(),'baseline':baseline,'temperature_only':candidates[0],'candidates':candidates,'selected':chosen,'selection_rule':rule,'limitations':['Development fit and selection metrics are optimistic, not holdout results.','Prior shift assumes roughly stable class-conditional distributions; this is an empirical adjustment, not a guaranteed correction.','Target is screened curated group distribution, not India or retail prevalence.','Structured input only. No weight training. No fresh India test.','Recorded OFF labels and exact overlap limitations remain.']}
    OUT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUT) as d:
        tmp=Path(d)
        with (tmp/names[0]).open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f); writer.writerow(['normalized_code','off_text_group_id','nova_group',*[f'logit_nova{k}' for k in LABELS]])
            for r,z in zip(rows,logits): writer.writerow([r['normalized_code'],r['off_text_group_id'],r['nova_group'],*z.tolist()])
        config['development_logits_sha256']=sha256(tmp/names[0]); (tmp/names[2]).write_text(json.dumps(config,indent=2)+'\n'); result['frozen_adjustment_sha256']=sha256(tmp/names[2]); result['development_logits_sha256']=config['development_logits_sha256']; (tmp/names[1]).write_text(json.dumps(result,indent=2)+'\n')
        for n in names: (tmp/n).replace(OUT/n)
    print(json.dumps({'experiment':'I7.3','run':run,'sample_rows':len(rows),'baseline':baseline,'temperature_only':candidates[0],'candidate_summary':[{'alpha':c['alpha'],'temperature':c['temperature'],'standardized_accuracy_pct':c['prior_standardized']['accuracy_pct'],'standardized_macro_f1':c['prior_standardized']['macro_f1'],'standardized_nova2_precision':c['prior_standardized']['per_class']['2']['precision'],'nova2_recall':c['prior_standardized']['per_class']['2']['recall'],'passes_constraints':c['passes_constraints']} for c in candidates],'selected':chosen,'selection_rule':rule,'frozen_adjustment_sha256':result['frozen_adjustment_sha256'],'development_logits_sha256':result['development_logits_sha256']},indent=2))
    print('SUCCESS: Development adjustment frozen. Holdout not opened or predicted. No model weights changed.')

if __name__=='__main__': main()
