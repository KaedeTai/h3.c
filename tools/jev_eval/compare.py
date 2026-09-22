#!/usr/bin/env python3
"""Compare the two decision paths on what a gate actually needs.

Accuracy at a fixed 0.5 is the wrong first question: a model can carry the whole
signal and still fail it by sitting off-centre. So:

AUC        : threshold-free -- is the signal there at all. 0.5 is chance.
best-thr   : accuracy at the best single threshold (the one you would fit once).
agreement  : mean pairwise Spearman between phrasings that mean the same thing.
             Does phrasing A rank the items the way phrasing B does? This is the
             stability question the caption gate got bitten by, and unlike raw
             spread it is not confounded by how confident a model is overall.
spread     : max-min of p(yes) over equivalent phrasings, for reference.
negation   : |p(claim) + p(not claim) - 1|. Does it track the word "not".
"""
import json, os, sys
from itertools import combinations

D = os.path.dirname(os.path.abspath(__file__))


def auc(scores, gold):
    pos = [s for s, g in zip(scores, gold) if g]
    neg = [s for s, g in zip(scores, gold) if not g]
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def best_threshold_acc(scores, gold):
    best = 0.0
    for t in sorted(set(scores)):
        for thr in (t, t + 1e-9):
            acc = sum((s >= thr) == bool(g) for s, g in zip(scores, gold)) / len(gold)
            best = max(best, acc)
    return best


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def report(path):
    d = json.load(open(path))
    print(f"\n=== {d['model']} ===")
    print(f"    {d['questions']} questions in {d['seconds']:.1f}s "
          f"({d['seconds'] / d['questions'] * 1000:.0f} ms per question)")
    for lang in ("en", "zh", "all"):
        rs = [r for r in d["results"] if lang == "all" or r["lang"] == lang]
        if not rs or (lang == "all" and len({r["lang"] for r in rs}) < 2):
            continue                      # "all" only means something when two languages are mixed
        gold = [r["gold"] for r in rs]
        n_ph = len(rs[0]["equivalent"])
        by_ph = [[r["equivalent"][i] for r in rs] for i in range(n_ph)]
        aucs = [auc(s, gold) for s in by_ph]
        mean_p = [sum(r["equivalent"]) / n_ph for r in rs]
        agree = [spearman(a, b) for a, b in combinations(by_ph, 2)]
        spreads = [max(r["equivalent"]) - min(r["equivalent"]) for r in rs]
        negs = [abs(m + p - 1) for r, m in zip(rs, mean_p) for p in r["negated"]]
        lo, hi = min(min(b) for b in by_ph), max(max(b) for b in by_ph)
        print(f"    {lang:>3}: AUC per phrasing {min(aucs):.2f}-{max(aucs):.2f} "
              f"(mean {sum(aucs)/len(aucs):.2f}) | AUC of the mean {auc(mean_p, gold):.2f} "
              f"| best-thr acc {best_threshold_acc(mean_p, gold):.2f}")
        print(f"         phrasing agreement (Spearman) {sum(agree)/len(agree):+.2f} "
              f"| spread {sum(spreads)/len(spreads):.2f} | p range {lo:.2f}-{hi:.2f} "
              f"| negation err {sum(negs)/len(negs):.2f}")


for f in sys.argv[1:] or ["out_openjev.json", "out_ornith.json"]:
    p = os.path.join(D, f)
    report(p) if os.path.exists(p) else print(f"\n(missing {f})")
