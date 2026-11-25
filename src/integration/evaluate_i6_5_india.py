"""I6.5: paired evaluation on all 5,311 frozen India-tagged representatives.
Run from repo root: HF_HUB_OFFLINE=1 python -m src.integration.evaluate_i6_5_india
Reuses the tested I6.3c metric function; no model training or input edits.
"""
import csv
import json
import tempfile
import time
from collections import Counter
from pathlib import Path

from .evaluate_i6_3_natural import evaluate, sha256, FORMATS, LABELS
from .audit_bda_dl_overlap import code_key, text_key, read_split, FILES

ROOT = Path.cwd()
SAMPLES = ROOT / 'data/integration/evaluation_i6_3'
OUT = ROOT / 'data/integration/evaluation_i6_5'
MODEL = ROOT / 'data/deep_learning/models/distilbert_food_nova'
PRIOR = ROOT / 'data/integration/evaluation_i6_3c/i6_3c_natural_metrics.json'
SOURCE_REPORT = ROOT / 'data/integration/evaluation_i6_4/i6_4_source_consistency.json'
EXPECTED = {1: 378, 2: 53, 3: 931, 4: 3949}
HASHES = {
    'india_evaluation': 'c1602b19575634e152fbc2322bc32c29a85c32473ad134fc0e4514cb79bb2c45',
    'balanced_curated': 'e1878935db79b0c43050b4ec84fc2663dcda1cd9b880e11a8ac11674fc96a87f',
    'natural_curated': '92936831b02322f4a91d216108eb632be36c507585b52e9d241bb639b30b4f40',
}
OFF_FIELDS = ('off_raw', 'off_clean', 'off_dl_structured')
csv.field_size_limit(16 * 1024 * 1024)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def text_keys(row, fields=OFF_FIELDS):
    return {text_key(row[f]) for f in fields} - {''}


def verify_rows(rows, old_codes, old_texts, reserved):
    if Counter(int(r['nova_group']) for r in rows) != EXPECTED:
        raise ValueError('India class distribution differs from frozen sample.')
    codes, groups, texts = set(), set(), set()
    for r in rows:
        c, g = r['normalized_code'], r['off_text_group_id']
        if not c or not g or c != code_key(r['code']):
            raise ValueError('Missing or inconsistent code/group identifier.')
        if r['evaluation_cohort'] != 'india_evaluation' or any(not text_key(r[f]) for f in FORMATS.values()):
            raise ValueError('Wrong cohort or missing inference input.')
        t = text_keys(r)
        if c in codes or g in groups or t & texts:
            raise ValueError('Repeated exact code, group or text in India sample.')
        if c in old_codes or t & old_texts:
            raise ValueError('India sample overlaps earlier DL experiments.')
        codes.add(c); groups.add(g); texts.update(t)
    for name, others, fields in reserved:
        for r in others:
            if r['normalized_code'] in codes or r['off_text_group_id'] in groups or text_keys(r, fields) & texts:
                raise ValueError(f'India sample overlaps {name}. Preserve frozen files and investigate.')
    return {'unique_codes': len(codes), 'unique_groups': len(groups), 'unique_off_text_keys': len(texts),
            'previous_dl_exact_overlap': 0, 'other_evaluation_exact_overlap': 0}


def main():
    sampling = json.loads((SAMPLES/'i6_3_sampling_report.json').read_text())
    if sampling['cross_sample_group_overlap'] != 0:
        raise ValueError('Sampling report indicates overlap.')
    for name, digest in HASHES.items():
        p = SAMPLES/(name+'.csv')
        if sha256(p) != digest or sampling['sample_sha256'][p.name] != digest:
            raise ValueError(f'Frozen sample changed: {name}')
    rows = read_csv(SAMPLES/'india_evaluation.csv')
    prior = json.loads(PRIOR.read_text())
    sources = json.loads(SOURCE_REPORT.read_text())
    if prior['experiment'] != 'I6.3c' or prior['sample_sha256'] != HASHES['natural_curated']:
        raise ValueError('Unexpected prior natural report.')
    if sources['experiment'] != 'I6.4' or sources['natural_report_sha256'] != sha256(PRIOR):
        raise ValueError('Unexpected source consistency report.')
    pairs = SOURCE_REPORT.parent/'source_pairs.csv'
    if sha256(pairs) != sources['sample_sha256']:
        raise ValueError('Source pairs hash mismatch.')
    hashes = {p.name: sha256(p) for p in sorted(MODEL.iterdir()) if p.is_file()}
    if not {'model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json'} <= set(hashes) or hashes != prior['model_files_sha256'] or hashes != sources['model_files_sha256']:
        raise ValueError('Model files differ from earlier experiments.')
    old_codes, old_texts = set(), set()
    for name in FILES:
        rel = f'data/deep_learning/datasets/{name}.csv'
        if sha256(ROOT/rel) != sampling['input_sha256'][rel]:
            raise ValueError(f'Original DL dataset changed: {name}')
        _, c, t = read_split(name)
        old_codes.update(c); old_texts.update(t)
    reserved = [(name, read_csv(SAMPLES/(name+'.csv')), OFF_FIELDS) for name in ('balanced_curated', 'natural_curated')]
    reserved.append(('I6.4 source pairs', read_csv(pairs), OFF_FIELDS + ('usda_raw','usda_clean')))
    checks = verify_rows(rows, old_codes, old_texts, reserved)
    del reserved
    OUT.mkdir(parents=True, exist_ok=True)
    names = ('india_predictions.csv','i6_5_india_metrics.json')
    if any((OUT/n).exists() for n in names):
        raise FileExistsError('Existing results; archive before repeating.')
    print('Frozen India sample verified:', json.dumps(checks), flush=True)
    from src.deep_learning.inference.predict import load_model, predict_nova
    import torch, transformers
    tokenizer, model = load_model()
    model.eval()
    def predictor(text):
        return predict_nova(text, tokenizer, model)
    for field in FORMATS.values():
        predictor(rows[0][field])
    start = time.perf_counter()
    metrics, paired, details = evaluate(rows, predictor)
    elapsed = time.perf_counter()-start
    report = {'experiment':'I6.5', 'method':'Paired raw and structured OFF inference on the complete frozen deduplicated India-tagged cohort; no tuning.',
        'sample_rows':len(rows), 'sample_sha256':HASHES['india_evaluation'], 'model_files_sha256':hashes,
        'natural_report_sha256':sha256(PRIOR), 'source_report_sha256':sha256(SOURCE_REPORT),
        'device':str(next(model.parameters()).device), 'torch_version':torch.__version__, 'transformers_version':transformers.__version__,
        'formats':FORMATS, 'labels':LABELS, 'verification':checks, 'class_distribution':EXPECTED,
        'prediction_calls_measured':len(rows)*len(FORMATS), 'evaluation_elapsed_seconds':elapsed,
        'majority_baseline':{'predicted_nova':4,'accuracy_pct':100*EXPECTED[4]/len(rows),'macro_f1':0.25*2*EXPECTED[4]/(len(rows)+EXPECTED[4])},
        'metrics':metrics, 'paired_comparisons':paired,
        'limitations':['India-tagged OFF products, not necessarily made in India or representative of Indian retail.',
            'English metadata and recorded OFF reference labels not independently verified.',
            'Screened, deduplicated cohort; exact-match screening cannot rule out near duplicates.',
            'Only 53 recorded NOVA 2 examples; class-specific estimates have limited support.',
            'Confidence is uncalibrated model output; local latency is not API latency.',
            'Raw and structured formats selected after I6.3b; no training or weight tuning.']}
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        tmp = Path(tmp)
        with (tmp/names[0]).open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(details[0])); writer.writeheader(); writer.writerows(details)
        report['predictions_sha256'] = sha256(tmp/names[0])
        (tmp/names[1]).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        for n in names:
            (tmp/n).replace(OUT/n)
    print(json.dumps(report, indent=2, allow_nan=False))
    print('Results saved:', OUT)
    print('SUCCESS: I6.5 India paired evaluation complete. No training performed.')


if __name__ == '__main__':
    main()
