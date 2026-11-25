"""I6.3c: paired natural-distribution OFF representation evaluation using the existing predictor.
Run from repo root: HF_HUB_OFFLINE=1 python -m src.integration.evaluate_i6_3_natural
Reads frozen samples/model only. Refuses to overwrite completed results.
"""
import csv
import hashlib
import json
import math
import tempfile
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_fscore_support, f1_score)

ROOT = Path.cwd()
SAMPLES = ROOT / 'data/integration/evaluation_i6_3'
OUT = ROOT / 'data/integration/evaluation_i6_3c'
BALANCED_REPORT = ROOT / 'data/integration/evaluation_i6_3b/i6_3b_balanced_metrics.json'
MODEL = ROOT / 'data/deep_learning/models/distilbert_food_nova'
LABELS = [1, 2, 3, 4]
FORMATS = {'raw': 'off_raw', 'structured': 'off_dl_structured'}
csv.field_size_limit(16 * 1024 * 1024)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_frozen(folder):
    report = json.loads((folder / 'i6_3_sampling_report.json').read_text())
    path = folder / 'natural_curated.csv'
    digest = sha256(path)
    if digest != report['sample_sha256'][path.name] or digest != '92936831b02322f4a91d216108eb632be36c507585b52e9d241bb639b30b4f40':
        raise ValueError('Natural sample hash mismatch: do not regenerate or edit the frozen sample.')
    if report['cross_sample_group_overlap'] != 0:
        raise ValueError('Sampling report indicates overlap.')
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'normalized_code', 'code', 'nova_group', 'evaluation_cohort', 'off_text_group_id', *FORMATS.values()}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Unexpected sample columns.')
        rows = list(reader)
    counts = Counter(int(r['nova_group']) for r in rows)
    expected = report['samples']['natural_curated']
    if len(rows) != expected['rows'] or {str(k): counts[k] for k in LABELS} != expected['nova']:
        raise ValueError('Sample counts disagree with frozen report.')
    if len(rows) != 10000 or dict(counts) != {1:364, 2:34, 3:1688, 4:7914}:
        raise ValueError('Expected the recorded 10000-row natural class distribution.')
    for key in ('normalized_code', 'off_text_group_id'):
        if len({r[key] for r in rows}) != len(rows) or any(not r[key].strip() for r in rows):
            raise ValueError(f'Invalid or repeated {key}.')
    if any(r['evaluation_cohort'] != 'natural_curated' or any(not r[f].strip() for f in FORMATS.values()) for r in rows):
        raise ValueError('Invalid cohort or missing ingredient representation.')
    # Verify the two reserved samples and check exact group / code separation.
    for name in ('balanced_curated', 'india_evaluation'):
        other = folder / (name + '.csv')
        if sha256(other) != report['sample_sha256'][other.name]:
            raise ValueError(f'Reserved sample hash mismatch: {name}')
        keys = {k: {r[k] for r in rows} for k in ('normalized_code', 'off_text_group_id')}
        with other.open(encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                if any(r[k] in v for k,v in keys.items()):
                    raise ValueError(f'Natural sample overlaps {name}.')
    return rows, digest


def evaluate(rows, predictor):
    y = [int(r['nova_group']) for r in rows]
    predictions = {k: [] for k in FORMATS}
    scores = {k: [] for k in FORMATS}
    times = {k: [] for k in FORMATS}
    details = []
    for i, row in enumerate(rows):
        result_row = {k: row[k] for k in ('code', 'normalized_code', 'off_text_group_id', 'nova_group')}
        # Rotate order to reduce systematic order effects in timings.
        order = list(FORMATS)
        order = order[i % len(order):] + order[:i % len(order)]
        for name in order:
            start = time.perf_counter()
            result = predictor(row[FORMATS[name]])
            elapsed = (time.perf_counter() - start) * 1000
            pred = int(result['predicted_nova'])
            confidence = float(result['confidence'])
            probs = [float(result['class_probabilities'][f'NOVA {k}']) for k in LABELS]
            if pred not in LABELS or not all(math.isfinite(p) and 0 <= p <= 100 for p in [confidence, *probs]):
                raise ValueError('Unexpected prediction or probabilities.')
            if abs(sum(probs)-100) > 0.1 or abs(confidence-probs[pred-1]) > 0.1 or max(probs)-confidence > 0.1:
                raise ValueError('Inconsistent probability output; inspect predictor contract.')
            predictions[name].append(pred)
            scores[name].append(confidence)
            times[name].append(elapsed)
            result_row[f'{name}_prediction'] = pred
            result_row[f'{name}_confidence_pct'] = confidence
            result_row[f'{name}_correct'] = int(pred == y[i])
            result_row[f'{name}_elapsed_ms'] = round(elapsed, 4)
            for k, p in zip(LABELS, probs):
                result_row[f'{name}_p_nova{k}_pct'] = p
        details.append(result_row)
        if (i+1) % 250 == 0 or i+1 == len(rows):
            print(f'Evaluated {i+1:,}/{len(rows):,} products ({len(FORMATS)*(i+1):,} predictions)', flush=True)
    metrics = {}
    for name, p in predictions.items():
        precision, recall, f1, support = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
        metrics[name] = {
            'accuracy_pct': float(accuracy_score(y, p)*100),
            'macro_f1': float(f1_score(y, p, labels=LABELS, average='macro', zero_division=0)),
            'balanced_accuracy_pct': float(balanced_accuracy_score(y, p)*100),
            'correct': sum(a == b for a,b in zip(y,p)),
            'per_class': {str(k): {'precision': float(precision[j]), 'recall': float(recall[j]), 'f1': float(f1[j]), 'support': int(support[j]), 'predicted_count': p.count(k)} for j,k in enumerate(LABELS)},
            'confusion_matrix_rows_actual_cols_predicted': confusion_matrix(y,p,labels=LABELS).tolist(),
            'mean_confidence_pct': sum(scores[name])/len(y),
            'mean_call_latency_ms': sum(times[name])/len(y),
        }
    paired = {}
    for a,b in combinations(FORMATS,2):
        pa,pb = predictions[a],predictions[b]
        both = sum(x == t and z == t for t,x,z in zip(y,pa,pb))
        a_only = sum(x == t and z != t for t,x,z in zip(y,pa,pb))
        b_only = sum(x != t and z == t for t,x,z in zip(y,pa,pb))
        agree = sum(x == z for x,z in zip(pa,pb))
        paired[f'{a}_vs_{b}'] = {'both_correct': both, 'first_only_correct': a_only, 'second_only_correct': b_only,
            'both_wrong': len(y)-both-a_only-b_only, 'prediction_agreement_pct': 100*agree/len(y),
            'second_minus_first_accuracy_pp': 100*(b_only-a_only)/len(y)}
    return metrics, paired, details


def main():
    rows, digest = read_frozen(SAMPLES)
    OUT.mkdir(parents=True, exist_ok=True)
    names = ['natural_predictions.csv', 'i6_3c_natural_metrics.json']
    if any((OUT / n).exists() for n in names):
        raise FileExistsError('Results already exist. Archive them before rerunning.')
    model_hashes = {p.name: sha256(p) for p in sorted(MODEL.iterdir()) if p.is_file()}
    if not {'model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json'} <= set(model_hashes):
        raise ValueError('Missing expected checkpoint files.')
    prior = json.loads(BALANCED_REPORT.read_text(encoding='utf-8'))
    if prior['experiment'] != 'I6.3b' or prior['sample_sha256'] != 'e1878935db79b0c43050b4ec84fc2663dcda1cd9b880e11a8ac11674fc96a87f':
        raise ValueError('Unexpected balanced evaluation report.')
    if model_hashes != prior['model_files_sha256']:
        raise ValueError('Checkpoint files differ from I6.3b. Restore original files before proceeding.')
    from src.deep_learning.inference.predict import load_model, predict_nova
    import torch
    import transformers
    tokenizer, model = load_model()
    model.eval()
    def predictor(text):
        return predict_nova(text, tokenizer, model)
    print('Warming up existing predictor...', flush=True)
    for field in FORMATS.values():
        predictor(rows[0][field])
    start = time.perf_counter()
    metrics, paired, details = evaluate(rows, predictor)
    report = {'experiment': 'I6.3c', 'sample_rows': len(rows), 'sample_sha256': digest,
        'model_files_sha256': model_hashes, 'device': str(next(model.parameters()).device),
        'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
        'formats': FORMATS, 'labels': LABELS, 'prediction_calls_measured': len(rows)*len(FORMATS),
        'balanced_report_sha256': sha256(BALANCED_REPORT),
        'majority_baseline': {'predicted_nova': 4, 'accuracy_pct': 79.14, 'macro_f1': 0.25*(2*7914/(10000+7914))},
        'evaluation_elapsed_seconds': time.perf_counter()-start,
        'metrics': metrics, 'paired_comparisons': paired,
        'limitations': ['Recorded OFF labels, not independent gold standard.', 'Exact screened group sample; near duplicates may remain.', 'Natural sample represents screened deduplicated groups, not market prevalence.', 'Confidence is raw model score, not calibrated correctness.', 'Latencies are local predictor calls, not API response times.', 'Raw and structured formats selected after I6.3b; comparison-cleaned text deliberately not repeated. No weight tuning or training performed. Preserve unused groups for future evaluation if tuning is undertaken.']}
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        tmp = Path(tmp)
        with (tmp / names[0]).open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(details[0]))
            writer.writeheader()
            writer.writerows(details)
        report['predictions_sha256'] = sha256(tmp / names[0])
        (tmp / names[1]).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        for n in names:
            (tmp / n).replace(OUT / n)
    print(json.dumps(report, indent=2, allow_nan=False))
    print('Results saved:', OUT)
    print('SUCCESS: I6.3c natural paired evaluation complete. No training performed.')


if __name__ == '__main__':
    main()
