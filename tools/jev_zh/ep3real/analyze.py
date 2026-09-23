#!/usr/bin/env python3
"""PREREG.md's table: test AUC overall and per tier, agreement, negation error, and test
accuracy at the dev-fitted (balanced-accuracy) threshold. Also writes the misses."""
import json, os, sys
from itertools import combinations
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, "..", "..", "jev_eval", "compare.py")).read()
exec(_src.split("def report")[0])          # auc, spearman -- without compare.py's own main
items = json.load(open(os.path.join(HERE, "ep3_all.json")))["items"]

def bal_acc(scores, gold, thr):
    tp = sum(s >= thr and g for s, g in zip(scores, gold)); fn = sum(s < thr and g for s, g in zip(scores, gold))
    tn = sum(s < thr and not g for s, g in zip(scores, gold)); fp = sum(s >= thr and not g for s, g in zip(scores, gold))
    return 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp)), (tp + tn) / len(gold)

rows = []
for tag in sys.argv[1:] or ["v2", "v3", "v1", "en"]:
    d = json.load(open(os.path.join(HERE, f"out_{tag}.json")))
    R = d["results"]; assert len(R) == len(items)
    for r, it in zip(R, items):
        assert r["state"] == it["state"]; r.update(split=it["split"], tier=it["tier"])
    mean = lambda r: sum(r["equivalent"]) / len(r["equivalent"])
    dev = [r for r in R if r["split"] == "dev"]; test = [r for r in R if r["split"] == "test"]
    cands = sorted({mean(r) for r in dev})
    thr = max(((bal_acc([mean(r) for r in dev], [r["gold"] for r in dev], t)[0], t) for t in cands))[1]
    ts, tg = [mean(r) for r in test], [r["gold"] for r in test]
    negs_ = [r for r in test if not r["gold"]]
    tier_auc = {t: auc([mean(r) for r in test if r["tier"] == t] + [mean(r) for r in negs_],
                       [1] * sum(r["tier"] == t for r in test) + [0] * len(negs_))
                for t in ("faithful", "garbled", "unmatched")}
    neg_tier = {t: auc([mean(r) for r in test if r["gold"]] + [mean(r) for r in negs_ if r["tier"] == t],
                       [1] * sum(r["gold"] for r in test) + [0] * sum(r["tier"] == t for r in negs_))
                for t in ("han", "latin")}
    n_ph = len(test[0]["equivalent"])
    by_ph = [[r["equivalent"][i] for r in test] for i in range(n_ph)]
    agree = sum(spearman(a, b) for a, b in combinations(by_ph, 2)) / (n_ph * (n_ph - 1) / 2)
    negerr = [abs(mean(r) + p - 1) for r in test for p in r["negated"]]
    ba, acc = bal_acc(ts, tg, thr)
    ba5, acc5 = bal_acc(ts, tg, 0.5)
    row = dict(model=tag, auc=auc(ts, tg), aucs_ph=[auc(s, tg) for s in by_ph], **{f"auc_{k}": v for k, v in tier_auc.items()},
               **{f"auc_vs_{k}": v for k, v in neg_tier.items()}, agree=agree, negerr=sum(negerr) / len(negerr),
               thr=thr, bal_acc=ba, acc=acc, bal_acc_05=ba5, ms=1000 * d["seconds"] / d["questions"])
    rows.append(row)
    miss = sorted([(mean(r), r["gold"], r["tier"], r["state"]) for r in test if (mean(r) >= thr) != bool(r["gold"])], reverse=True)
    json.dump(miss, open(os.path.join(HERE, f"miss_{tag}.json"), "w"), ensure_ascii=False, indent=0)
print("| model | test AUC | per phrasing | faithful | garbled | unmatched | vs Han neg | vs Latin neg | agreement | negation err | dev thr | test bal-acc (acc) | bal-acc at 0.5 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['model']} | {r['auc']:.3f} | {min(r['aucs_ph']):.2f}-{max(r['aucs_ph']):.2f} | {r['auc_faithful']:.3f} | {r['auc_garbled']:.3f} | "
          f"{r['auc_unmatched']:.3f} | {r['auc_vs_han']:.3f} | {r['auc_vs_latin']:.3f} | {r['agree']:+.2f} | {r['negerr']:.2f} | "
          f"{r['thr']:.3f} | {r['bal_acc']:.3f} ({r['acc']:.3f}) | {r['bal_acc_05']:.3f} |")
json.dump(rows, open(os.path.join(HERE, "results.json"), "w"), indent=1)
