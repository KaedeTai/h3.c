"""Line-level units: every OCR line, deduped on its folded text, with where it sat and
how close it is to a spoken line. Auto-classes are proposals; labels.json is the truth."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.expanduser("~/datasets/ep3_ocr")
EXCLUDE_VIDS = {553}   # a downloaded interview recording that sits in Downloads/ep3; not EP3 footage
import opencc
t = opencc.OpenCC("s2t").convert
L = json.load(open(os.path.join(ROOT, "script_lines.json")))
def norm(s): return re.sub(r"[^㐀-鿿豈-﫿A-Za-z0-9]", "", t(s))
def lcs(a, b):
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]
units = {}
for l in open(os.path.join(ROOT, "ocr.jsonl")):
    r = json.loads(l)
    if not r["frame"] or r["vid"] in EXCLUDE_VIDS: continue
    for b in r["lines"]:
        s = b["text"].strip()
        if b["conf"] < 0.3 or len(norm(s)) < 2: continue
        k = norm(s)
        u = units.setdefault(k, {"text": s, "n": 0, "vids": set(), "frames": [], "boxes": []})
        u["n"] += 1; u["vids"].add(r["vid"])
        if len(u["frames"]) < 3: u["frames"].append(r["frame"]); u["boxes"].append([b["x"], b["y"], b["w"], b["h"]])
out = []
for k, u in units.items():
    han = len(re.findall(r"[㐀-鿿]", k)); lat = len(re.findall(r"[A-Za-z]", k))
    best = (0.0, None)
    if han >= 2:
        for c in L:
            b = norm(c["text"]); r = lcs(k, b) / max(len(k), len(b))
            if r > best[0]: best = (r, c["text"])
    x, y, w, h = u["boxes"][0]
    if best[0] >= 0.35: auto = "dialogue"
    elif lat >= 6 and han == 0 and " " in u["text"]: auto = "pseudo_en"
    elif han == 0 and y < 0.6: auto = "latin_upper"
    else: auto = "check"
    out.append({"key": k, "text": u["text"], "n": u["n"], "vids": sorted(u["vids"]), "frames": u["frames"],
                "boxes": u["boxes"], "y": round(y, 3), "xc": round(x + w / 2, 3), "w": round(w, 3),
                "han": han, "sim": round(best[0], 3), "match": best[1], "auto": auto})
out.sort(key=lambda o: (o["auto"], -o["n"]))
json.dump(out, open(os.path.join(ROOT, "lines.json"), "w"), ensure_ascii=False, indent=0)
import collections
print(len(out), collections.Counter(o["auto"] for o in out))
