"""labels.py + lines.json -> jev_eval-format items: ep3_dev.json / ep3_test.json / ep3_all.json.

State format is the final test's ("OCR：「…」"), text only. Split is by video (a video's
lines all land on one side). At most CAP lines per video per label, so the one reference
sheet and the one phone screen do not become the whole negative class.
"""
import json, os, sys, hashlib, collections
sys.path.insert(0, os.path.dirname(__file__))
import labels
ROOT = os.path.expanduser("~/datasets/ep3_ocr")
HERE = os.path.dirname(os.path.abspath(__file__))
O = json.load(open(os.path.join(ROOT, "lines.json")))
V = json.load(open(os.path.join(ROOT, "videos.json")))
final = json.load(open(os.path.join(HERE, "..", "..", "jev_eval", "items_final_v2.json")))
CAP = 10
lab = {}
for i, o in enumerate(O):
    if o["auto"] == "dialogue": lab[i] = 1
    elif o["auto"] == "latin_upper": lab[i] = 0
lab.update(labels.OTHER); lab.update(labels.BOTTOM_SAMPLE)
per = collections.Counter(); items = []
for i in sorted(lab, key=lambda i: -O[i]["n"]):
    g = lab[i]
    if g == "x": continue
    o = O[i]; vid = o["vids"][0]
    if per[(vid, g)] >= CAP: continue
    per[(vid, g)] += 1
    if g == 1: tier = "faithful" if o["sim"] >= 0.6 else "garbled" if o["sim"] >= 0.35 else "unmatched"
    else: tier = "han" if o["han"] >= 2 else "latin"
    split = "dev" if int(hashlib.md5(V[vid].encode()).hexdigest(), 16) % 2 == 0 else "test"
    items.append({"lang": "zh", "gold": g, "state": f"OCR：「{o['text']}」", "tier": tier, "split": split,
                  "line": i, "vid": vid, "sim": o["sim"], "match": o["match"], "frame": o["frames"][0]})
note = ("EP3 real OCR (Apple Vision) of on-screen text, one line per item, hand-labelled from the frames; "
        "phrasings = items_final_v2's zh set. Built by tools/jev_zh/ep3real.")
for name, sel in [("all", items), ("dev", [x for x in items if x["split"] == "dev"]), ("test", [x for x in items if x["split"] == "test"])]:
    json.dump({"note": note, "phrasings": {"zh": final["phrasings"]["zh"]}, "items": sel},
              open(os.path.join(HERE, f"ep3_{name}.json"), "w"), ensure_ascii=False, indent=1)
    c = collections.Counter((x["gold"], x["tier"]) for x in sel)
    print(name, len(sel), dict(sorted(c.items())))
