"""I6.3a: freeze exact-text-group-disjoint OFF evaluation samples.
Run from repository root with .venv-integration active.
Standard library only, plus the existing I6.1 helper (no inference).
"""
import csv
import hashlib
import json
import random
import tempfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from audit_bda_dl_overlap import FILES, read_split, code_key, text_key

ROOT = Path.cwd()
IN = ROOT / 'data/integration/evaluation_i6_2'
OUT = ROOT / 'data/integration/evaluation_i6_3'
SEED = 42
NATURAL_TARGET = 10_000
BALANCED_PER_CLASS = 500
TEXT_FIELDS = ('off_raw', 'off_clean', 'off_dl_structured')
LABELS = (1, 2, 3, 4)
csv.field_size_limit(16 * 1024 * 1024)


class Groups:
    def __init__(self):
        self.parent = []
        self.size = []

    def add(self):
        i = len(self.parent)
        self.parent.append(i)
        self.size.append(1)
        return i

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def text_digest(text):
    return hashlib.sha256(text.encode('utf-8')).digest()


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'cohort', 'code', 'normalized_code', 'nova_group', *TEXT_FIELDS}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f'Unexpected columns in {path}')
        yield from reader


def distribution(indices, metadata):
    counts = Counter(metadata[i][1] for i in indices)
    return {str(k): counts[k] for k in LABELS}


def main():
    rng = random.Random(SEED)
    paths = {k: IN / f'{k}_candidates.csv' for k in ('curated', 'india')}
    expected = json.loads((IN / 'i6_2_structured_audit.json').read_text())
    old = [read_split(name) for name in FILES]
    old_codes = set().union(*(x[1] for x in old))
    old_texts = set().union(*(x[2] for x in old))
    if len(old_codes) != expected['exclusion_codes'] or len(old_texts) != expected['exclusion_texts']:
        raise ValueError('Previous DL exclusion sets changed; repeat I6.1/I6.2.')
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = ['balanced_curated.csv', 'natural_curated.csv', 'india_evaluation.csv', 'i6_3_sampling_report.json']
    if any((OUT / f).exists() for f in outputs):
        raise FileExistsError('I6.3 outputs already exist. Archive them before repeating this run.')
    input_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in [*paths.values(), IN / 'i6_2_structured_audit.json', *[ROOT / 'data/deep_learning/datasets' / f'{n}.csv' for n in FILES]]}
    groups = Groups()
    text_owner, code_owner = {}, {}
    metadata = []  # cohort, label, normalized code; ingredient strings streamed, not retained
    before = Counter()
    per_cohort_codes = defaultdict(set)
    for cohort, path in paths.items():
        for row in rows(path):
            label = int(row['nova_group'])
            code = row['normalized_code']
            keys = {text_key(row[f]) for f in TEXT_FIELDS} - {''}
            if (row['cohort'] != cohort or label not in LABELS or not code or
                code != code_key(row['code']) or code in per_cohort_codes[cohort] or
                not text_key(row['off_raw']) or not text_key(row['off_dl_structured']) or
                (cohort == 'curated' and not text_key(row['off_clean']))):
                raise ValueError(f'Unexpected candidate row: {cohort} {code}')
            if code in old_codes or keys & old_texts:
                raise ValueError(f'Previous DL overlap unexpectedly present: {cohort} {code}')
            i = groups.add()
            metadata.append((cohort, label, code))
            per_cohort_codes[cohort].add(code)
            before[cohort] += 1
            groups.union(i, code_owner.setdefault(code, i))
            for key in sorted(keys):
                digest = text_digest(key)
                groups.union(i, text_owner.setdefault(digest, i))
            if before[cohort] % 50_000 == 0:
                print(f'{cohort}: grouped {before[cohort]:,} rows', flush=True)
        if before[cohort] != expected['cohorts'][cohort]['retained']:
            raise ValueError(f'{cohort} count differs from I6.2 audit.')
        if distribution((i for i,m in enumerate(metadata) if m[0] == cohort), metadata) != expected['cohorts'][cohort]['nova_after']:
            raise ValueError(f'{cohort} label distribution differs from I6.2 audit.')
        print(f'{cohort}: grouped {before[cohort]:,} rows', flush=True)
    # All linked rows are a component: shared exact normalized text in ANY
    # OFF representation (even across formats), or shared normalized code.
    del text_owner, code_owner
    members = defaultdict(list)
    for i in range(len(metadata)):
        members[groups.find(i)].append(i)
    conflicts = Counter()
    removed = Counter()
    india, curated = [], []
    cross_groups = 0
    duplicate_groups = 0
    conflicting_groups = 0
    for indices in members.values():
        cohorts = {metadata[i][0] for i in indices}
        cross_groups += len(cohorts) > 1
        duplicate_groups += len(indices) > 1
        if len({metadata[i][1] for i in indices}) > 1:
            conflicting_groups += 1
            for i in indices:
                conflicts[metadata[i][0]] += 1
            continue
        # India priority: reserve every nonconflicting group containing India.
        eligible = [i for i in indices if metadata[i][0] == 'india']
        if eligible:
            chosen = rng.choice(eligible)
            india.append(chosen)
        else:
            chosen = rng.choice(indices)
            curated.append(chosen)
        for i in indices:
            if i != chosen:
                removed[metadata[i][0]] += 1
    for cohort, kept in [('curated', curated), ('india', india)]:
        assert before[cohort] == conflicts[cohort] + removed[cohort] + len(kept)
    # Natural first to avoid deliberate balanced-sampling distortion.
    natural = rng.sample(curated, min(NATURAL_TARGET, len(curated)))
    natural_set = set(natural)
    buckets = {k: [] for k in LABELS}
    for i in curated:
        if i not in natural_set:
            buckets[metadata[i][1]].append(i)
    n = min(BALANCED_PER_CLASS, *(len(v) for v in buckets.values()))
    if n == 0:
        raise ValueError(f'No balanced sample possible after natural sampling: { {k:len(v) for k,v in buckets.items()} }')
    balanced = [i for k in LABELS for i in rng.sample(buckets[k], n)]
    selection = {'balanced_curated': balanced, 'natural_curated': natural, 'india_evaluation': india}
    sample_groups = {name: {groups.find(i) for i in ids} for name,ids in selection.items()}
    for name, ids in selection.items():
        assert len(sample_groups[name]) == len(ids)
    for a,b in combinations(selection, 2):
        assert not sample_groups[a] & sample_groups[b]
    # Use component ID only for within-run audit; no claim of global formulation ID.
    chosen = {i: name for name, ids in selection.items() for i in ids}
    assert len(chosen) == sum(map(len, selection.values()))
    report = {
        'method': 'Connected exact normalized OFF text/code groups across both pools; drop all conflicting-label groups; India priority; one random representative per group; natural sample first, then balanced sample from remainder.',
        'seed': SEED, 'input_sha256': input_hashes,
        'input_rows': dict(before),
        'groups_total': len(members), 'groups_with_multiple_rows': duplicate_groups,
        'groups_shared_between_cohorts': cross_groups, 'conflicting_label_groups': conflicting_groups,
        'conflicting_rows_excluded': {k: conflicts[k] for k in paths},
        'nonrepresentative_rows_excluded': {k: removed[k] for k in paths},
        'representative_pool': {'curated': {'rows':len(curated), 'nova':distribution(curated, metadata)},
                                'india': {'rows':len(india), 'nova':distribution(india, metadata)}},
        'balanced_target_per_class': BALANCED_PER_CLASS, 'balanced_actual_per_class': n,
        'natural_target': NATURAL_TARGET,
        'samples': {name: {'rows':len(ids), 'nova':distribution(ids, metadata)} for name,ids in selection.items()},
        'remaining_unused_curated_representatives': len(curated)-len(natural)-len(balanced),
        'remaining_unused_curated_nova': distribution((i for i in curated if i not in chosen), metadata),
        'cross_sample_group_overlap': 0,
        'limitations': ['Natural sample represents the deduplicated screened curated groups, not product-weighted retail prevalence.', 'Exact lowercase/whitespace text only; not semantic or near-duplicate detection.', 'English metadata and OFF labels are not independently verified.', 'Grouping excludes USDA text; cross-source experiment needs its own checks.', 'Unused rows are not automatically a valid future test set; preserve untouched groups if tuning.'],
    }
    print('Writing sample manifests...', flush=True)
    with tempfile.TemporaryDirectory(prefix='work_', dir=OUT) as temp:
        temp = Path(temp)
        fields = list(next(rows(paths['curated'])).keys()) + ['evaluation_cohort', 'off_text_group_id']
        from contextlib import ExitStack
        written = Counter()
        with ExitStack() as stack:
            writers = {}
            for name in selection:
                f = stack.enter_context((temp / f'{name}.csv').open('w', encoding='utf-8', newline=''))
                writers[name] = csv.DictWriter(f, fieldnames=fields)
                writers[name].writeheader()
            i = 0
            for cohort,path in paths.items():
                for row in rows(path):
                    assert metadata[i] == (cohort, int(row['nova_group']), row['normalized_code'])
                    if i in chosen:
                        name = chosen[i]
                        row.update(evaluation_cohort=name, off_text_group_id=groups.find(i))
                        writers[name].writerow(row)
                        written[name] += 1
                    i += 1
            assert i == len(metadata)
        assert all(written[name] == len(ids) for name,ids in selection.items())
        report['sample_sha256'] = {f'{name}.csv': sha256_file(temp / f'{name}.csv') for name in selection}
        (temp / 'i6_3_sampling_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        for f in outputs:
            (temp / f).replace(OUT / f)
    print(json.dumps(report, indent=2))
    print(f'Output folder: {OUT}')
    print('SUCCESS: I6.3a samples frozen. No inference or training performed.')


if __name__ == '__main__':
    main()
