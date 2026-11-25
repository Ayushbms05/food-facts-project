"""Integration I6.1: preliminary exact barcode/text overlap audit.
Run from the repository root with the integration environment activated.
Original Parquet/CSV files are read-only; writes a new JSON audit report.
"""
import csv
import json
from collections import Counter
from pathlib import Path
from itertools import combinations

ROOT = Path.cwd()
DL = ROOT / 'data/deep_learning/datasets'
FILES = ['train', 'validation', 'test', 'natural_distribution_sample']


def code_key(value):
    code = str(value or '').strip()
    if not code:
        return ''
    # Conservative exclusion: catch missing leading zeros, including short OFF codes.
    # This is NOT GTIN validation; short codes can represent different identifiers.
    return code.zfill(14) if code.isascii() and code.isdigit() and len(code) <= 14 else code


def text_key(value):
    return ' '.join((value or '').lower().split())


def english_texts(items):
    return [entry['text'] for entry in (items or [])
            if entry and entry.get('lang') == 'en' and text_key(entry.get('text'))]


def read_split(name):
    with (DL / f'{name}.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'code', 'ingredient_text', 'nova_group'} <= set(reader.fieldnames or []):
            raise ValueError(f'Unexpected CSV schema: {name}')
        rows = list(reader)
    codes = {code_key(r['code']) for r in rows} - {''}
    texts = {text_key(r['ingredient_text']) for r in rows} - {''}
    return rows, codes, texts


def audit(dataset, kind, codes, texts):
    columns = (['gtin14', 'off_code', 'off_ingredients_en', 'off_clean',
                'usda_ingredients', 'usda_clean', 'nova_group', 'dl_candidate_off',
                'needs_review', 'agreement_category'] if kind == 'curated' else
               ['code', 'ingredients_text', 'nova_group'])
    counts = Counter()
    before, after = Counter(), Counter()
    agreement = Counter()
    for batch in dataset.to_batches(columns=columns, batch_size=4096, use_threads=False):
        for row in batch.to_pylist():
            counts['total_rows'] += 1
            label = row['nova_group']
            if kind == 'curated':
                if not row['dl_candidate_off'] or label not in (1, 2, 3, 4):
                    continue
                ids = [row['gtin14'], row['off_code']]
                raw = [row['off_ingredients_en']]
                clean = [row['off_clean']]
                usda = [row['usda_ingredients'], row['usda_clean']]
            else:
                raw = english_texts(row['ingredients_text'])
                if not raw or label not in (1, 2, 3, 4):
                    continue
                ids, clean, usda = [row['code']], [], []
            counts['eligible_rows'] += 1
            before[label] += 1
            keys = {code_key(x) for x in ids} - {''}
            hit_code = bool(keys & codes)
            hit_raw = any(text_key(x) in texts for x in raw)
            hit_clean = any(text_key(x) in texts for x in clean)
            hit_usda = any(text_key(x) in texts for x in usda)
            missing = not keys
            counts['missing_code'] += missing
            counts['barcode_overlap'] += hit_code
            counts['raw_off_text_overlap'] += hit_raw
            counts['clean_off_text_overlap'] += hit_clean
            counts['usda_text_overlap'] += hit_usda
            # Keep USDA overlap separate: OFF-only evaluation need not exclude it.
            excluded = missing or hit_code or hit_raw or hit_clean
            counts['excluded_union_off'] += excluded
            if not excluded:
                counts['remaining_off_candidates'] += 1
                after[label] += 1
                if kind == 'curated':
                    agreement[row['agreement_category'] or 'unknown'] += 1
                    counts['remaining_needs_review'] += bool(row['needs_review'])
                    if text_key(row['usda_ingredients']) and not hit_usda:
                        counts['remaining_pair_candidates'] += 1
    for key in ['total_rows','eligible_rows','missing_code','barcode_overlap',
                'raw_off_text_overlap','clean_off_text_overlap','usda_text_overlap',
                'excluded_union_off','remaining_off_candidates']:
        counts.setdefault(key, 0)
    assert counts['eligible_rows'] == counts['excluded_union_off'] + counts['remaining_off_candidates']
    result = {'counts': dict(counts), 'nova_before': dict(sorted(before.items())),
              'nova_after': dict(sorted(after.items()))}
    if kind == 'curated':
        result['remaining_agreement_categories'] = dict(sorted(agreement.items()))
    return result


def main():
    import pyarrow.dataset as ds
    splits = {name: read_split(name) for name in FILES}
    all_codes = set().union(*(s[1] for s in splits.values()))
    all_texts = set().union(*(s[2] for s in splits.values()))
    report = {'method': 'preliminary exact overlap; lowercase/whitespace normalized text; conservative numeric zero-padding; excludes all original splits and natural sample; structured ingredient overlap still pending',
              'split_rows': {name: len(s[0]) for name,s in splits.items()},
              'unique_exclusion_codes': len(all_codes), 'unique_exclusion_texts': len(all_texts),
              'split_pair_overlap': {}}
    for a,b in combinations(FILES, 2):
        report['split_pair_overlap'][f'{a} vs {b}'] = {
            'normalized_codes': len(splits[a][1] & splits[b][1]),
            'normalized_texts': len(splits[a][2] & splits[b][2])}
    for name, path in [('curated', 'data/curated/source_aware_products.parquet'),
                       ('india', 'data/india_candidates.parquet')]:
        print(f'Auditing {name}...', flush=True)
        report[name] = audit(ds.dataset(ROOT / path, format='parquet'), name, all_codes, all_texts)
    out = ROOT / 'data/integration/audits/i6_1_preliminary_overlap.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'\nAudit report saved: {out}')
    print('SUCCESS: Preliminary overlap audit complete. Structured OFF text audit still pending.')


if __name__ == '__main__':
    main()
