#!/usr/bin/env python3
"""Ornith logit-read on the same items: one forward pass, read p(yes) from the first
token's logits rather than generating. The same trick the caption gate uses."""
import json, os, sys, time
import mlx.core as mx
from mlx_lm import load

D = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.expanduser("~/models/Ornith-BigBang-MTP-zh-mlx-4bit")
data = json.load(open(os.path.join(D, "items_textonly.json")))
model, tok = load(MODEL)

YES = {"en": "Yes", "zh": "是"}
NO = {"en": "No", "zh": "否"}


def tid(word):
    ids = tok.encode(word, add_special_tokens=False)
    return ids[0]


IDS = {l: (tid(YES[l]), tid(NO[l])) for l in ("en", "zh")}


def p_yes(state, claim, lang):
    if lang == "en":
        user = (f"{state}\n\nStatement: {claim}\n"
                "Is the statement true? Answer with exactly one word, Yes or No.")
    else:
        user = (f"{state}\n\n敘述：{claim}\n"
                "這個敘述正確嗎？只回答一個字，是 或 否。")
    prompt = tok.apply_chat_template([{"role": "user", "content": user}],
                                     tokenize=False, add_generation_prompt=True)
    ids = mx.array(tok.encode(prompt))[None]
    out = model(ids)
    logits = out[0] if isinstance(out, tuple) else out
    last = logits[0, -1].astype(mx.float32)
    y, n = IDS[lang]
    pair = mx.stack([last[y], last[n]])
    return float(mx.softmax(pair)[0])


out, t0 = [], time.time()
for it in data["items"]:
    ph = data["phrasings"][it["lang"]]
    eq = [p_yes(it["state"], q, it["lang"]) for q in ph["equivalent"]]
    ng = [p_yes(it["state"], q, it["lang"]) for q in ph["negated"]]
    out.append({"lang": it["lang"], "gold": it["gold"], "state": it["state"],
                "equivalent": eq, "negated": ng})
dt = time.time() - t0
n_q = sum(len(o["equivalent"]) + len(o["negated"]) for o in out)
json.dump({"model": "Ornith-BigBang-MTP-zh-mlx-4bit", "seconds": dt, "questions": n_q,
           "results": out},
          open(os.path.join(D, "out_ornith_textonly.json"), "w"), ensure_ascii=False, indent=1)
print(f"{len(out)} states / {n_q} questions in {dt:.1f}s ({dt/n_q:.2f}s per question)")
