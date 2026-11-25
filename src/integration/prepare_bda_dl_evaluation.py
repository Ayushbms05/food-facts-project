"""I6.2: structured OFF audit and candidate pool preparation (no inference).
Run from repository root: python src/integration/prepare_bda_dl_evaluation.py
Uses the exact I6.1 helpers and original DL extraction function.
Only writes new generated files under data/integration/evaluation_i6_2/.
"""
import csv
import json
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path

from audit_bda_dl_overlap import code_key, text_key, english_texts, read_split, FILES

ROOT = Path.cwd()
OUT = ROOT / 'data/integration/evaluation_i6_2'
PATHS = {
    'curated': ROOT / 'data/curated/source_aware_products.parquet',
    'india': ROOT / 'data/india_candidates.parquet',
}
COLS = {
    'curated': ['gtin14', 'off_code', 'off_product_name', 'off_ingredients_en',
                'off_clean', 'usda_ingredients', 'usda_clean', 'nova_group',
                'dl_candidate_off', 'needs_review', 'agreement_category',
                'off_records_for_gtin'],
    'india': ['code', 'ingredients_text', 'nova_group'],
}
FIELDS = ['cohort', 'code', 'normalized_code', 'nova_group', 'product_name',
          'off_raw', 'off_clean', 'off_dl_structured', 'usda_raw', 'usda_clean',
          'agreement_category', 'needs_review', 'off_records_for_gtin',
          'pair_eligible']


def original_key(x):
    # Exact code join (apart from surrounding whitespace); never join via padding.
    return str(x or '').strip()


def preliminary(row, kind, codes, texts):
    label = row['nova_group']
    if label not in (1, 2, 3, 4):
        return None
    if kind == 'curated':
        if not row['dl_candidate_off']:
            return None
        raw = [row['off_ingredients_en']]
        clean = row['off_clean']
        ids = [row['gtin14'], row['off_code']]
        code = original_key(row['off_code'])
        key = code_key(row['gtin14'])
    else:
        raw = english_texts(row['ingredients_text'])
        if not raw:
            return None
        clean = ''
        ids = [row['code']]
        code = original_key(row['code'])
        key = code_key(code)
    keys = {code_key(x) for x in ids} - {''}
    if not keys or keys & codes:
        return None
    if any(text_key(x) in texts for x in raw) or text_key(clean) in texts:
        return None
    return dict(cohort=kind, code=code, normalized_code=key,
                nova_group=label, product_name=row.get('off_product_name') or '',
                off_raw=raw[0] or '', off_clean=clean or '',
                usda_raw=row.get('usda_ingredients') or '',
                usda_clean=row.get('usda_clean') or '',
                agreement_category=row.get('agreement_category') or '',
                needs_review=row.get('needs_review', ''),
                off_records_for_gtin=row.get('off_records_for_gtin', ''),
                _raw_variants=raw)


def candidates(ds, kind, codes, texts):
    dataset = ds.dataset(PATHS[kind], format='parquet')
    for batch in dataset.to_batches(columns=COLS[kind], batch_size=4096, use_threads=False):
        for row in batch.to_pylist():
            c = preliminary(row, kind, codes, texts)
            if c is not None:
                yield c


def evaluate_candidate(c, match, duplicate_count, texts):
    # Mutually exclusive first-failure reason; order is deliberate.
    if not c['code'] or not c['normalized_code']:
        return 'missing_exact_code'
    if code_key(c['code']) != c['normalized_code']:
        return 'normalized_id_mismatch'
    if duplicate_count != 1:
        return 'duplicate_candidate_identifier'
    if not match:
        return 'original_code_not_found'
    n, lang, label, structured, raw_json = match
    if n != 1:
        return 'ambiguous_original_code'
    if label != c['nova_group']:
        return 'original_label_mismatch'
    if lang != 'en':
        return 'original_language_not_en'
    if not text_key(c['off_raw']):
        return 'missing_raw_off_text'
    raw_keys = set(json.loads(raw_json))
    if text_key(c['off_raw']) not in raw_keys:
        return 'raw_off_text_mismatch'
    if raw_keys & texts:
        return 'original_raw_text_overlap'
    if not text_key(structured):
        return 'missing_structured_text'
    if text_key(structured) in texts:
        return 'structured_text_overlap'
    c['off_dl_structured'] = structured
    c['pair_eligible'] = bool(c['cohort'] == 'curated' and text_key(c['usda_raw'])
                              and text_key(c['usda_raw']) not in texts
                              and text_key(c['usda_clean']) not in texts)
    return 'retained'


def main():
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    # Reuse exact extraction used by Member 2; do NOT run dataset preparation.
    sys.path.insert(0, str(ROOT))
    from src.deep_learning.data_preparation.prepare_dataset import extract_ingredients
    splits = [read_split(name) for name in FILES]
    codes = set().union(*(s[1] for s in splits))
    texts = set().union(*(s[2] for s in splits))
    expected = json.loads((ROOT / 'data/integration/audits/i6_1_preliminary_overlap.json').read_text())
    report = {'method': 'I6.1 exclusions + exact original OFF code join, unique original record, label match, lang=en, matching raw OFF text, exact structured text exclusion',
              'exclusion_codes': len(codes), 'exclusion_texts': len(texts),
              'limitations': ['Exact normalized matches only; near duplicates not excluded.',
                              'Same-source OFF labels, not independently verified ground truth.',
                              'English metadata does not guarantee English ingredient text.',
                              'USDA pair eligibility is not independently verified language or label validity.',
                              'Pools may overlap across cohorts; do not combine metrics as independent samples.'],
              'cohorts': {}}
    target = set()
    key_counts = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for kind in PATHS:
        counts = Counter()
        for c in candidates(ds, kind, codes, texts):
            counts[c['normalized_code']] += 1
            if c['code']:
                target.add(c['code'])
        n = sum(counts.values())
        if n != expected[kind]['counts']['remaining_off_candidates']:
            raise ValueError(f'{kind}: I6.1 count changed ({n}); rerun preliminary audit first')
        key_counts[kind] = counts
        print(f'{kind}: {n:,} preliminary candidates', flush=True)
    print(f'Scanning original OFF for {len(target):,} exact codes...', flush=True)
    # Store potentially large ingredient strings on disk, not in a giant in-memory join.
    with tempfile.TemporaryDirectory(prefix='work_', dir=OUT) as tmp:
        work = Path(tmp)
        with sqlite3.connect(work / 'matches.sqlite') as db:
            db.execute('CREATE TABLE matches (code TEXT PRIMARY KEY, n INTEGER NOT NULL, lang TEXT, label INTEGER, structured TEXT, raw_json TEXT)')
            scan = pq.ParquetFile(ROOT / 'data/food.parquet')
            scanned = 0
            for i, batch in enumerate(scan.iter_batches(batch_size=4096,
                    columns=['code', 'lang', 'nova_group', 'ingredients', 'ingredients_text'],
                    use_threads=False), 1):
                rows = []
                scanned += batch.num_rows
                for row in batch.to_pylist():
                    code = original_key(row['code'])
                    if code not in target:
                        continue
                    structured = extract_ingredients(row['ingredients']) if row['lang'] == 'en' else ''
                    raw_keys = sorted({text_key(x) for x in english_texts(row['ingredients_text'])})
                    rows.append((code, 1, row['lang'], row['nova_group'], structured, json.dumps(raw_keys, ensure_ascii=False)))
                db.executemany('INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(code) DO UPDATE SET n=n+1', rows)
                if i % 100 == 0:
                    db.commit()
                    print(f'Original OFF rows scanned: {scanned:,}', flush=True)
            db.commit()
            report['original_rows_scanned'] = scanned
            report['exact_codes_requested'] = len(target)
            report['exact_codes_matched'] = db.execute('SELECT COUNT(*) FROM matches').fetchone()[0]
            report['ambiguous_exact_codes'] = db.execute('SELECT COUNT(*) FROM matches WHERE n>1').fetchone()[0]
            retained_codes = {}
            for kind in PATHS:
                decisions, before, after, pairs, categories = (Counter() for _ in range(5))
                retained_codes[kind] = set()
                with (work / f'{kind}_candidates.csv').open('w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
                    writer.writeheader()
                    for c in candidates(ds, kind, codes, texts):
                        before[c['nova_group']] += 1
                        match = db.execute('SELECT n,lang,label,structured,raw_json FROM matches WHERE code=?', (c['code'],)).fetchone()
                        reason = evaluate_candidate(c, match, key_counts[kind][c['normalized_code']], texts)
                        decisions[reason] += 1
                        if reason == 'retained':
                            after[c['nova_group']] += 1
                            retained_codes[kind].add(c['normalized_code'])
                            if c['pair_eligible']:
                                pairs[c['nova_group']] += 1
                                categories[c['agreement_category']] += 1
                            writer.writerow(c)
                assert sum(decisions.values()) == sum(before.values())
                assert sum(after.values()) == decisions['retained'] == len(retained_codes[kind])
                report['cohorts'][kind] = {
                    'preliminary_candidates': sum(before.values()),
                    'decision_counts': dict(decisions),
                    'retained': sum(after.values()),
                    'nova_before': dict(sorted(before.items())),
                    'nova_after': dict(sorted(after.items())),
                    'pair_eligible': sum(pairs.values()),
                    'pair_nova': dict(sorted(pairs.items())),
                    'pair_agreement_categories': dict(sorted(categories.items())),
                }
            report['retained_normalized_code_overlap_between_cohorts'] = len(retained_codes['curated'] & retained_codes['india'])
        (work / 'i6_2_structured_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        for name in ('curated_candidates.csv', 'india_candidates.csv', 'i6_2_structured_audit.json'):
            (work / name).replace(OUT / name)
    print(json.dumps(report, indent=2))
    print(f'Output folder: {OUT}')
    print('SUCCESS: structured audit complete. No inference or training performed.')


if __name__ == '__main__':
    main()
