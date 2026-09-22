#!/usr/bin/env python3
"""Run a saved bundle over an items file (jev_eval format) and write the out_*.json that
jev_eval/compare.py reads. All phrasings of one state go through one forward pass."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "jev_eval", "typed-decisions-src", "src"))
from typed_decisions.open_jev import OpenJev  # noqa: E402

bundle, items_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
langs = sys.argv[4].split(",") if len(sys.argv) > 4 else None
data = json.load(open(items_path))
m = OpenJev.from_pretrained(bundle)
m.decide("暖機", [{"type": "noul", "instructions": "暖機"}])
out, t0 = [], time.time()
for it in data["items"]:
    if langs and it["lang"] not in langs:
        continue
    ph = data["phrasings"][it["lang"]]
    qs = [{"type": "noul", "instructions": q} for q in ph["equivalent"] + ph["negated"]]
    res = m.decide(it["state"], qs)
    k = len(ph["equivalent"])
    out.append({"lang": it["lang"], "gold": it["gold"], "state": it["state"],
                "equivalent": [r["noul"] for r in res[:k]], "negated": [r["noul"] for r in res[k:]]})
dt = time.time() - t0
n_q = sum(len(o["equivalent"]) + len(o["negated"]) for o in out)
json.dump({"model": os.path.basename(os.path.dirname(bundle.rstrip("/"))) + "/" + os.path.basename(bundle.rstrip("/")),
           "seconds": dt, "questions": n_q, "results": out},
          open(out_path, "w"), ensure_ascii=False, indent=1)
print(f"{len(out)} states / {n_q} questions in {dt:.1f}s -> {out_path}")
