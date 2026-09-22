#!/usr/bin/env python3
"""open-jev noul on the caption-gate items. One forward pass per (state, phrasing)."""
import json, os, sys, time
D = os.path.dirname(os.path.abspath(__file__))
M = os.path.expanduser("~/models/open-jev-deberta-v3-large")
sys.path.insert(0, M)
from typed_decisions.open_jev import OpenJev

data = json.load(open(os.path.join(D, "items.json")))
m = OpenJev.from_pretrained(M)

out, t0 = [], time.time()
for it in data["items"]:
    ph = data["phrasings"][it["lang"]]
    qs = [{"type": "noul", "instructions": q} for q in ph["equivalent"] + ph["negated"]]
    # all phrasings of one state in a single forward pass -- the point of the design
    res = m.decide(it["state"], qs)
    out.append({"lang": it["lang"], "gold": it["gold"], "state": it["state"],
                "equivalent": [r["noul"] for r in res[:len(ph["equivalent"])]],
                "negated": [r["noul"] for r in res[len(ph["equivalent"]):]]})
dt = time.time() - t0
n_q = sum(len(o["equivalent"]) + len(o["negated"]) for o in out)
json.dump({"model": "open-jev-deberta-v3-large", "seconds": dt, "questions": n_q, "results": out},
          open(os.path.join(D, "out_openjev.json"), "w"), ensure_ascii=False, indent=1)
print(f"{len(out)} states / {n_q} questions in {dt:.1f}s ({dt/len(out):.2f}s per state)")
