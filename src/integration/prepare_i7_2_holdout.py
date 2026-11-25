"""I7.2: freeze new group-disjoint development and held-out curated samples.
Run from repo root: python src/integration/prepare_i7_2_holdout.py
No model inference, tuning, or changes to earlier outputs.
"""
import csv
import json
import random
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from prepare_i6_3_samples import Groups, rows, text_digest, sha256_file, TEXT_FIELDS
from audit_bda_dl_overlap import FILES, read_split, code_key, text_key

ROOT = Path.cwd()
BASE = ROOT / 'data/integration'
OUT = BASE / 'evaluation_i7_2'
SEED = 20261125
LABELS = (1, 2, 3, 4)
TARGET = {1: 500, 2: 80, 3: 1000, 4: 3500}


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def dist(ids, meta):
    c = Counter(meta[i][1] for i in ids)
    return {str(k): c[k] for k in LABELS}


def keys(row, fields):
    return {text_key(row.get(f)) for f in fields} - {''}


def choose_samples(buckets, rng):
    dev, test = [], []
    sizes = {}
    for label in LABELS:
        n = min(TARGET[label], len(buckets[label]) // 2)
        require(n > 0, f'No disjoint dev/test samples possible for NOVA {label}')
        chosen = rng.sample(buckets[label], 2*n)
        dev.extend(chosen[:n]); test.extend(chosen[n:]); sizes[str(label)] = n
    return dev, test, sizes


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('I7.2 outputs already exist; do not overwrite frozen holdout.')
    previous_path = BASE / 'evaluation_i6_3/i6_3_sampling_report.json'
    previous = json.loads(previous_path.read_text())
    hashes = {}
    def verify(p, expected=None):
        h = sha256_file(p)
        require(expected is None or h == expected, f'Hash mismatch: {p}')
        hashes[str(p.relative_to(ROOT))] = h
        return h
    verify(previous_path)
    for relative, expected in previous['input_sha256'].items():
        verify(ROOT / relative, expected)
    old_codes, old_texts = set(), set()
    splits = {}
    for name in [*FILES, 'training_sample']:
        data, codes, texts = read_split(name)
        splits[name] = data
        old_codes.update(codes); old_texts.update(texts)
        verify(ROOT/'data/deep_learning/datasets'/f'{name}.csv')
    def triples(records):
        return Counter((r['code'], r['ingredient_text'], r['nova_group']) for r in records)
    original = triples(splits['training_sample'])
    split_union = sum((triples(splits[n]) for n in ('train','validation','test')), Counter())
    train_check = {'training_sample_rows':sum(original.values()), 'split_union_rows':sum(split_union.values()),
                   'exact_row_multiset_equal':original==split_union,
                   'training_sample_only_rows':sum((original-split_union).values()),
                   'split_union_only_rows':sum((split_union-original).values())}
    # Reconstruct all earlier connected groups, including conflicting groups and India.
    g = Groups(); meta=[]; owners={}; codes={}; before=Counter()
    paths = {c:BASE/'evaluation_i6_2'/f'{c}_candidates.csv' for c in ('curated','india')}
    hit_old=set()
    for cohort, path in paths.items():
        for r in rows(path):
            label=int(r['nova_group']); code=r['normalized_code']; txt=keys(r,TEXT_FIELDS)
            require(r['cohort']==cohort and label in LABELS and code and code==code_key(r['code']) and txt, 'Invalid candidate row')
            i=g.add(); meta.append((cohort,label,code)); before[cohort]+=1
            g.union(i,codes.setdefault(code,i))
            for t in sorted(txt):
                g.union(i,owners.setdefault(text_digest(t),i))
            if code in old_codes or txt & old_texts: hit_old.add(i)
            if before[cohort]%50000==0: print(f'{cohort}: grouped {before[cohort]:,}',flush=True)
    require(dict(before)==previous['input_rows'],'Candidate counts changed')
    members=defaultdict(list)
    for i in range(len(meta)): members[g.find(i)].append(i)
    require(len(members)==previous['groups_total'],'Group reconstruction differs from I6.3a')
    # Reserve any group touched by earlier samples, source texts, OCR or old DL data.
    reserved=defaultdict(set)
    def reserve(code, texts, reason):
        if code in codes: reserved[reason].add(g.find(codes[code]))
        for t in texts:
            i=owners.get(text_digest(t))
            if i is not None: reserved[reason].add(g.find(i))
    for i in hit_old: reserved['original_DL_including_training_sample'].add(g.find(i))
    earlier_groups=set()
    for name in ('balanced_curated','natural_curated','india_evaluation'):
        p=BASE/'evaluation_i6_3'/f'{name}.csv'; verify(p,previous['sample_sha256'][p.name])
        count=0; seen=set()
        for r in read_csv(p):
            code=r['normalized_code']; require(code in codes,'Earlier sample code missing')
            root=g.find(codes[code]); require(root==int(r['off_text_group_id']), 'Earlier group ID changed')
            require(root not in seen and root not in earlier_groups,'Earlier sample groups overlap')
            seen.add(root); count+=1
            reserve(code,keys(r,TEXT_FIELDS),name)
        require(count==previous['samples'][name]['rows'],'Earlier sample count mismatch')
        earlier_groups.update(seen)
    source_report=BASE/'evaluation_i6_4/i6_4_source_consistency.json'
    verify(source_report,'41daebe178fc0b67e96511674f181c096bedb40408df34c717218d88be1c66ec')
    sr=json.loads(source_report.read_text()); p=source_report.parent/'source_pairs.csv'; verify(p,sr['sample_sha256'])
    for r in read_csv(p):
        reserve(code_key(r['code']),keys(r,(*TEXT_FIELDS,'usda_raw','usda_clean')),'source_pairs')
    ocr_manifest=BASE/'evaluation_i6_6_inputs/manifest.json'
    verify(ocr_manifest,'e2cdbba385c1a00309992d95a79dcb68379ce9af26b903844cdda6e060ff7c98')
    manifest=json.loads(ocr_manifest.read_text())
    for r in manifest['rows']:
        reserve(code_key(r['barcode']),set(),'ocr_all_image_codes')
        if r['status']=='included':
            p=ocr_manifest.parent/r['reference']; verify(p,r['reference_sha256'])
            reserve(code_key(r['barcode']),{text_key(p.read_text())}-{''},'ocr_reference_and_output')
    ocr_report=BASE/'evaluation_i6_6/i6_6_ocr_metrics.json'; verify(ocr_report)
    ocr=json.loads(ocr_report.read_text()); require(ocr['manifest_sha256']==hashes[str(ocr_manifest.relative_to(ROOT))], 'OCR manifest mismatch')
    pp=ocr_report.parent/'ocr_predictions.jsonl'; verify(pp,ocr['predictions_sha256'])
    require(len(ocr['per_image'])==12,'Unexpected OCR report size')
    for r in ocr['per_image']:
        reserve(code_key(r['barcode']),keys(r,('reference_text','ocr_text')),'ocr_reference_and_output')
    # Exclude any group containing India, conflicts, or prior-used code/text.
    excluded=set().union(*reserved.values()) if reserved else set()
    rng=random.Random(SEED); buckets={k:[] for k in LABELS}; exclusions=Counter(); original_curated=Counter()
    for root, ids in members.items():
        labels={meta[i][1] for i in ids}
        if len(labels)!=1: exclusions['conflicting_label_groups']+=1; continue
        if any(meta[i][0]=='india' for i in ids): exclusions['groups_containing_india']+=1; continue
        label=next(iter(labels)); original_curated[label]+=1
        if root in excluded: exclusions['previously_used_or_linked_curated_groups']+=1; continue
        buckets[label].append(rng.choice(ids))
    require({str(k):original_curated[k] for k in LABELS}==previous['representative_pool']['curated']['nova'],'Original representative population changed')
    dev,test,sizes=choose_samples(buckets,rng)
    selection={'development':dev,'holdout':test}
    chosen={i:name for name,ids in selection.items() for i in ids}
    roots={name:{g.find(i) for i in ids} for name,ids in selection.items()}
    require(not roots['development'] & roots['holdout'] and not (roots['development']|roots['holdout']) & excluded,'Group overlap')
    available={str(k):len(buckets[k]) for k in LABELS}
    # Save target standardization weights separately. Original curated priors are NOT market priors.
    reference={str(k):original_curated[k]/sum(original_curated.values()) for k in LABELS}
    sample_weights={str(k):reference[str(k)]/(sizes[str(k)]/len(dev)) for k in LABELS}
    report={'experiment':'I7.2','seed':SEED,'method':'Reconstruct exact connected OFF groups; reserve all India groups, earlier samples, source pair text/code, OCR code/text and old DL including full training sample; choose one random representative per unused nonconflicting curated group; stratified disjoint development and holdout.',
            'input_sha256':hashes,'training_sample_partition_check':train_check,'input_rows':dict(before),'groups_total':len(members),
            'reserved_groups_by_reason':{k:len(v) for k,v in reserved.items()},'exclusions_first_failure':dict(exclusions),
            'unused_eligible_by_nova':available,'target_per_split':{str(k):v for k,v in TARGET.items()},'actual_per_split':sizes,
            'samples':{name:{'rows':len(ids),'nova':dist(ids,meta)} for name,ids in selection.items()},
            'original_curated_reference_class_priors':reference,'class_standardization_weights_both_splits':sample_weights,
            'remaining_unused_by_nova':{k:available[k]-2*sizes[k] for k in available},'development_holdout_group_overlap':0,
            'limitations':['Exact OFF grouping only, not semantic or near-duplicate detection.','Stratified samples intentionally enrich scarce NOVA 2; report unweighted metrics and separately prior-standardized estimates using original screened curated group distribution, not retail prevalence.','Earlier class-dependent depletion and additional exclusions may affect within-class representativeness.','No fresh India test cohort; do not tune or claim India improvement using the inspected India sample.','Recorded OFF labels not independent ground truth. Holdout must not be predicted or inspected for model selection until adjustment frozen.']}
    OUT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUT) as d:
        tmp=Path(d); written=Counter(); fields=list(next(rows(paths['curated'])).keys())+['evaluation_cohort','off_text_group_id']
        with ExitStack() as stack:
            writers={}
            for name in selection:
                f=stack.enter_context((tmp/f'{name}.csv').open('w',newline='',encoding='utf-8'))
                writers[name]=csv.DictWriter(f,fieldnames=fields); writers[name].writeheader()
            for i,r in enumerate(rows(paths['curated'])):
                require(meta[i]==('curated',int(r['nova_group']),r['normalized_code']),'Candidate row order changed')
                if i in chosen:
                    name=chosen[i]; r.update(evaluation_cohort=name,off_text_group_id=g.find(i)); writers[name].writerow(r); written[name]+=1
        require(all(written[n]==len(ids) for n,ids in selection.items()),'Output row mismatch')
        report['sample_sha256']={f'{n}.csv':sha256_file(tmp/f'{n}.csv') for n in selection}
        (tmp/'i7_2_holdout_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        for p in tmp.iterdir(): p.replace(OUT/p.name)
    print(json.dumps(report,indent=2)); print('SUCCESS: I7.2 development and holdout frozen. No inference or tuning.')

if __name__=='__main__': main()
