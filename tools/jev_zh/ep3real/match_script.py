"""Score every distinct OCR string against the EP3 script's spoken lines.

The video model burns in captions that echo what the character says, often as
near-miss glyphs (它只是換了一種形態 -> 官只是恢了一种梓披). Similarity is the
longest-common-subsequence ratio after folding to Traditional, against every clause
of every spoken line; the best clause wins.
"""
import json, os, re, sys
import opencc
t = opencc.OpenCC("s2t").convert
ROOT = os.path.expanduser("~/datasets/ep3_ocr")
md = open(os.path.expanduser("~/movie/ep3/zhixing_youwei_ep3.md")).read()
lines = []
for m in re.finditer(r"^\| \*\*(\d+)\*\* \|.*$", md, re.M):
    shot = int(m.group(1)); row = m.group(0)
    for q in re.findall(r"「(.+?)」", row.split("|")[5]):
        for clause in re.split(r"[，。？！、…『』\s]+|<br>", q.replace("**", "")):
            clause = clause.strip()
            if len(clause) >= 2:
                lines.append({"shot": shot, "text": clause})
# the whole sentences too, for captions that span clauses
for m in re.finditer(r"^\| \*\*(\d+)\*\* \|.*$", md, re.M):
    for q in re.findall(r"「(.+?)」", m.group(0).split("|")[5]):
        lines.append({"shot": int(m.group(1)), "text": re.sub(r"<br>|\*\*", "", q)})
json.dump(lines, open(os.path.join(ROOT, "script_lines.json"), "w"), ensure_ascii=False, indent=0)

def norm(s):
    return re.sub(r"[^㐀-鿿豈-﫿A-Za-z0-9]", "", t(s))

def lcs(a, b):
    if not a or not b: return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]

S = json.load(open(os.path.join(ROOT, "strings.json")))
for s in S:
    best = (0.0, None)
    for part in s["text"].split(" / "):
        a = norm(part)
        if len(a) < 2: continue
        for L in lines:
            b = norm(L["text"])
            r = lcs(a, b) / max(len(a), len(b))
            if r > best[0]: best = (r, L)
    s["sim"] = round(best[0], 3); s["match"] = best[1]
json.dump(S, open(os.path.join(ROOT, "strings.json"), "w"), ensure_ascii=False, indent=0)
import collections
h = collections.Counter(min(9, int(s["sim"] * 10)) for s in S)
print(len(lines), "script clauses;", sorted(h.items()))
