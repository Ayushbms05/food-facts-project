"""I6.4: descriptive OFF/USDA paired prediction consistency (not USDA accuracy).
Run from repo root: HF_HUB_OFFLINE=1 python -m src.integration.evaluate_i6_4_sources
Reuses I6.3c cohort deliberately; results are not independent of I6.3c.
"""
import csv
import hashlib
import json
import math
import random
import tempfile
import time
from collections import Counter
from pathlib import Path

from .audit_bda_dl_overlap import code_key, text_key, read_split, FILES
from .evaluate_i6_3_natural import read_frozen, sha256

ROOT = Path.cwd()
SAMPLES = ROOT / 'data/integration/evaluation_i6_3'
PRIOR = ROOT / 'data/integration/evaluation_i6_3c/i6_3c_natural_metrics.json'
MODEL = ROOT / 'data/deep_learning/models/distilbert_food_nova'
OUT = ROOT / 'data/integration/evaluation_i6_4'
TARGETS = {'same_basic': 667, 'same_after_extra_cleanup': 667, 'still_different': 666}
OFF_FIELDS = ('off_raw', 'off_clean', 'off_dl_structured')
USDA_FIELDS = ('usda_raw', 'usda_clean')
LABELS = (1, 2, 3, 4)
SEED = 42
csv.field_size_limit(16 * 1024 * 1024)


def keys(row, fields):
    return {text_key(row[f]) for f in fields} - {''}


def select_pairs(rows, old_codes, old_texts, reserved_codes, reserved_texts):
    """Selection is independent of predictions. One exact text key per selected pair across pairs."""
    buckets = {k: [] for k in TARGETS}
    exclusions = Counter()
    for r in rows:
        code = code_key(r['code'])
        if not code or code != r['normalized_code']:
            raise ValueError('Invalid normalized code')
        if r['agreement_category'] not in TARGETS:
            exclusions['outside_paired_categories'] += 1
            continue
        if str(r['pair_eligible']).strip().lower() not in ('true', '1'):
            exclusions['not_pair_eligible'] += 1
            continue
        if not text_key(r['off_raw']) or not text_key(r['usda_raw']):
            exclusions['missing_raw_input'] += 1
            continue
        all_keys = keys(r, OFF_FIELDS + USDA_FIELDS)
        if code in old_codes or all_keys & old_texts:
            exclusions['previous_dl_exact_overlap'] += 1
            continue
        if code in reserved_codes or all_keys & reserved_texts:
            exclusions['balanced_or_india_exact_overlap'] += 1
            continue
        buckets[r['agreement_category']].append(r)
    available = {k: len(v) for k, v in buckets.items()}
    rng = random.Random(SEED)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    # Round-robin avoids giving a single agreement stratum first choice of every text.
    selected = []
    counts = Counter()
    seen_texts, seen_codes = set(), set()
    offsets = Counter()
    while True:
        added = False
        for category, target in TARGETS.items():
            if counts[category] >= target:
                continue
            bucket = buckets[category]
            while offsets[category] < len(bucket):
                r = bucket[offsets[category]]
                offsets[category] += 1
                all_keys = keys(r, OFF_FIELDS + USDA_FIELDS)
                if r['normalized_code'] in seen_codes or all_keys & seen_texts:
                    exclusions['repeated_selected_pair_text_or_code'] += 1
                    continue
                seen_texts.update(all_keys)
                seen_codes.add(r['normalized_code'])
                selected.append(r)
                counts[category] += 1
                added = True
                break
        if not added:
            break
    if any(not counts[k] for k in TARGETS):
        raise ValueError(f'At least one agreement stratum is empty: {dict(counts)}')
    return selected, {'available_before_selected_pair_dedup': available,
        'target_by_category': TARGETS, 'selected_by_category': {k: counts[k] for k in TARGETS},
        'shortfall_by_category': {k: TARGETS[k]-counts[k] for k in TARGETS},
        'screening_and_selection_exclusions': dict(exclusions),
        'note': 'First-failure screening counts; duplicate-selection count covers visited rows only. No backfill across strata.'}


def predict_checked(predictor, text):
    start = time.perf_counter()
    r = predictor(text)
    elapsed = (time.perf_counter()-start)*1000
    p = int(r['predicted_nova'])
    c = float(r['confidence'])
    if p not in LABELS or not math.isfinite(c) or not 0 <= c <= 100:
        raise ValueError('Invalid predictor response')
    return p, c, elapsed


def evaluate(rows, predictor):
    details = []
    for i, r in enumerate(rows):
        inputs = {'off': r['off_raw'], 'usda': r['usda_raw']}
        order = list(inputs) if i % 2 == 0 else list(reversed(inputs))
        d = {k: r[k] for k in ('code', 'normalized_code', 'off_text_group_id', 'agreement_category')}
        d['recorded_off_nova'] = int(r['nova_group'])
        for name in order:
            p, c, ms = predict_checked(predictor, inputs[name])
            d.update({f'{name}_prediction': p, f'{name}_confidence_pct': c, f'{name}_latency_ms': round(ms, 4)})
        d['prediction_agreement'] = int(d['off_prediction'] == d['usda_prediction'])
        d['raw_text_exact_normalized_match'] = int(text_key(inputs['off']) == text_key(inputs['usda']))
        details.append(d)
        if (i+1)%100 == 0 or i+1 == len(rows):
            print(f'Evaluated {i+1:,}/{len(rows):,} pairs', flush=True)
    return details


def summarize(details):
    n = len(details)
    if not n:
        return {'pairs': 0}
    matrix = [[0]*4 for _ in LABELS]
    for r in details:
        matrix[r['off_prediction']-1][r['usda_prediction']-1] += 1
    exact = [r for r in details if r['raw_text_exact_normalized_match']]
    # The OFF label is used only for OFF-side accuracy, never as USDA truth.
    return {'pairs': n, 'agreement_count': sum(r['prediction_agreement'] for r in details),
        'agreement_pct': 100*sum(r['prediction_agreement'] for r in details)/n,
        'disagreement_count': sum(not r['prediction_agreement'] for r in details),
        'off_prediction_counts': {str(k): sum(r['off_prediction']==k for r in details) for k in LABELS},
        'usda_prediction_counts': {str(k): sum(r['usda_prediction']==k for r in details) for k in LABELS},
        'transition_matrix_rows_off_prediction_cols_usda_prediction': matrix,
        'off_reference_label_counts': {str(k): sum(r['recorded_off_nova']==k for r in details) for k in LABELS},
        'off_only_accuracy_against_recorded_off_label_pct': 100*sum(r['off_prediction']==r['recorded_off_nova'] for r in details)/n,
        'raw_text_exact_normalized_match_count': len(exact),
        'agreement_on_exact_normalized_raw_matches_pct': (100*sum(r['prediction_agreement'] for r in exact)/len(exact) if exact else None),
        'off_mean_confidence_pct': sum(r['off_confidence_pct'] for r in details)/n,
        'usda_mean_confidence_pct': sum(r['usda_confidence_pct'] for r in details)/n}


def main():
    rows, natural_digest = read_frozen(SAMPLES)
    sampling = json.loads((SAMPLES/'i6_3_sampling_report.json').read_text())
    prior = json.loads(PRIOR.read_text())
    if prior['experiment'] != 'I6.3c' or prior['sample_sha256'] != natural_digest:
        raise ValueError('Prior natural evaluation does not match frozen sample')
    hashes = {p.name: sha256(p) for p in sorted(MODEL.iterdir()) if p.is_file()}
    if hashes != prior['model_files_sha256']:
        raise ValueError('Checkpoint differs from natural evaluation')
    old_codes, old_texts = set(), set()
    for name in FILES:
        rel = f'data/deep_learning/datasets/{name}.csv'
        if sha256(ROOT/rel) != sampling['input_sha256'][rel]:
            raise ValueError(f'Prior DL sample changed: {name}')
        _, c, t = read_split(name)
        old_codes.update(c)
        old_texts.update(t)
    reserved_codes, reserved_texts = set(), set()
    for name in ('balanced_curated', 'india_evaluation'):
        with (SAMPLES/f'{name}.csv').open(encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                reserved_codes.add(r['normalized_code'])
                reserved_texts.update(keys(r, OFF_FIELDS))
    OUT.mkdir(parents=True, exist_ok=True)
    names = ('source_pairs.csv', 'source_predictions.csv', 'i6_4_source_consistency.json')
    if any((OUT/n).exists() for n in names):
        raise FileExistsError('Existing I6.4 outputs; archive before repeating')
    chosen, selection = select_pairs(rows, old_codes, old_texts, reserved_codes, reserved_texts)
    print(json.dumps(selection, indent=2), flush=True)
    from src.deep_learning.inference.predict import load_model, predict_nova
    import torch, transformers
    tokenizer, model = load_model()
    model.eval()
    def predictor(text):
        return predict_nova(text, tokenizer, model)
    predictor(chosen[0]['off_raw'])
    predictor(chosen[0]['usda_raw'])
    start = time.perf_counter()
    details = evaluate(chosen, predictor)
    elapsed = time.perf_counter()-start
    report = {'experiment': 'I6.4', 'method': 'Stratified matched-source descriptive comparison, raw OFF versus raw USDA using unchanged predictor; deliberate natural-cohort reuse.',
        'seed': SEED, 'natural_sample_sha256': natural_digest, 'natural_report_sha256': sha256(PRIOR),
        'model_files_sha256': hashes, 'device': str(next(model.parameters()).device),
        'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
        'selection': selection, 'prediction_calls_measured': 2*len(chosen), 'evaluation_elapsed_seconds': elapsed,
        'overall_selected_pairs': summarize(details),
        'by_agreement_category': {k: summarize([r for r in details if r['agreement_category']==k]) for k in TARGETS},
        'limitations': ['USDA has no independent NOVA reference; agreement is not accuracy, and disagreement does not identify which source is correct.',
            'Matching barcode does not prove identical formulation or time period. English ingredient language not independently verified.',
            'Stratified, screened subset of I6.3c: not independent and pooled agreement is not a population prevalence estimate.',
            'Greedy exact normalized text/code screening; no near-duplicate detection. Source agreement categories describe text matching, not semantic equivalence.',
            'Confidence is an uncalibrated model score. No tuning or training performed.']}
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        tmp = Path(tmp)
        for name, data in zip(names[:2], (chosen, details)):
            with (tmp/name).open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
        report['sample_sha256'] = sha256(tmp/names[0])
        report['predictions_sha256'] = sha256(tmp/names[1])
        (tmp/names[2]).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        for name in names:
            (tmp/name).replace(OUT/name)
    print(json.dumps(report, indent=2, allow_nan=False))
    print('Results saved:', OUT)
    print('SUCCESS: I6.4 paired prediction consistency complete. No USDA accuracy or training performed.')


if __name__ == '__main__':
    main()
