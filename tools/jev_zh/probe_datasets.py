import traceback
from datasets import load_dataset, get_dataset_config_names

CANDS = [
    ("mteb/MassiveIntentClassification", ["zh-TW", "zh-CN"]),
    ("mteb/MassiveScenarioClassification", ["zh-TW"]),
    ("mteb/AmazonReviewsClassification", ["zh"]),
    ("facebook/xnli", ["zh"]),
    ("clue/clue", ["tnews", "ocnli", "iflytek", "c3"]),
    ("thu-coai/lccc", ["base"]),
]
for repo, cfgs in CANDS:
    try:
        names = get_dataset_config_names(repo)
    except Exception as e:
        print(f"[{repo}] config list FAILED: {type(e).__name__}: {str(e)[:120]}")
        names = []
    print(f"[{repo}] configs: {names[:30]}{' ...' if len(names) > 30 else ''}")
    for c in cfgs:
        if names and c not in names:
            print(f"   - {c}: not a config")
            continue
        try:
            ds = load_dataset(repo, c)
            for sp, d in ds.items():
                row = {k: (str(v)[:60]) for k, v in d[0].items()}
                print(f"   - {c}/{sp}: {len(d)} rows, cols {d.column_names}\n       e.g. {row}")
        except Exception as e:
            print(f"   - {c}: LOAD FAILED {type(e).__name__}: {str(e)[:160]}")
