"""Read-only diagnosis of the DL training sample and split artifact mismatch."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path("data/deep_learning/datasets")
def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def code_key(value):
    v = value.strip()
    return v.zfill(14) if v.isascii() and v.isdigit() and len(v) <= 14 else v

def text_key(value):
    return " ".join(value.lower().split())

def summarize(a,b):
    ca,cb=Counter(a),Counter(b)
    return {
        "left_unique":len(ca),"right_unique":len(cb),
        "shared_unique":len(ca.keys() & cb.keys()),
        "left_only_rows":sum((ca-cb).values()),
        "right_only_rows":sum((cb-ca).values()),
        "multisets_equal":ca==cb,
    }

def main():
    a=read(ROOT/"training_sample.csv")
    parts={name:read(ROOT/f"{name}.csv") for name in ("train","validation","test")}
    b=[r for rs in parts.values() for r in rs]
    report={"training_sample_vs_split_union":{},"earlier_evaluation_vs_full_training_sample":{}}
    keys={
        "exact_row":lambda r:(r["code"],r["ingredient_text"],r["nova_group"]),
        "normalized_row":lambda r:(code_key(r["code"]),text_key(r["ingredient_text"]),int(r["nova_group"])),
        "normalized_code":lambda r:code_key(r["code"]),
        "normalized_ingredient_text":lambda r:text_key(r["ingredient_text"]),
    }
    for name,key in keys.items():
        report["training_sample_vs_split_union"][name]=summarize(map(key,a),map(key,b))
    report["class_counts"]={"training_sample":dict(Counter(r["nova_group"] for r in a)),
                            **{n:dict(Counter(r["nova_group"] for r in rs)) for n,rs in parts.items()}}
    ac={code_key(r["code"]) for r in a}
    at={text_key(r["ingredient_text"]) for r in a if text_key(r["ingredient_text"])}
    for name in ("balanced_curated","natural_curated","india_evaluation"):
        rs=read(Path("data/integration/evaluation_i6_3")/f"{name}.csv")
        code_hits=text_hits=either=0
        for r in rs:
            c=code_key(r["code"]) in ac
            t=any(text_key(r.get(k,"")) in at for k in ("off_raw","off_clean","off_dl_structured"))
            code_hits+=c; text_hits+=t; either+=c or t
        report["earlier_evaluation_vs_full_training_sample"][name]={"rows":len(rs),"code_overlap_rows":code_hits,"text_overlap_rows":text_hits,"either_overlap_rows":either}
    print(json.dumps(report,indent=2))
    print("Read-only audit complete. No datasets or models changed.")

if __name__=="__main__":
    main()
