#!/usr/bin/env python3
"""Build the Chinese typed-decision corpus in typed_decisions' jsonl format.

Same rules as the released open-jev: public gold labels only, no synthetic answers, no teacher.
Where a label comes from a rule (a news headline is not dialogue; a C3 turn is), the rule is
the label, exactly as upstream's banking77 "product area" is a keyword rule over intent names.

Sources (states -> questions, all Chinese; simplified sources converted to Taiwan traditional
with OpenCC s2twp at p=0.7, a deterministic transform that preserves gold):

  MASSIVE zh-TW  spoken commands   choice intent (60) / choice scenario (18) / noul "is about X"
  Amazon zh      product reviews   score 5 levels / choice polarity / noul satisfied
  OCNLI          native Chinese NLI  noul claim holds (entail vs contradict only) / choice 3-way
  TNEWS          news headlines    choice category (15) / noul "is a X news item"
  C3             dialogues         choice: the dataset's own reading-comprehension question
  caption        the gate itself   noul dialogue? (x2 phrasings) / choice text category (7)

Outputs data/{train,val,test,ood-test}.jsonl for the upstream trainer, plus
dev_caption.json in the jev_eval items format for the iteration metric, and
final_textonly.json (the held-out final test with every scene description removed).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "jev_eval", "typed-decisions-src", "src"))

from typed_decisions.schema import Example, Question, NOUL_OPTIONS, write_jsonl  # noqa: E402
import pools as P  # noqa: E402

NO = list(NOUL_OPTIONS)
_cc = None


def tw(text: str, rng: random.Random, p: float = 0.7) -> str:
    global _cc
    if _cc is None:
        import opencc
        _cc = opencc.OpenCC("s2twp")
    return _cc.convert(text) if rng.random() < p else text


def to_tw(text: str) -> str:
    global _cc
    if _cc is None:
        import opencc
        _cc = opencc.OpenCC("s2twp")
    return _cc.convert(text)


# ---------------------------------------------------------------- MASSIVE (zh-TW, traditional)
def massive(split: str, n: int, rng: random.Random) -> list[Example]:
    from datasets import load_dataset
    ds = load_dataset("mteb/MassiveIntentClassification", "zh-TW", split=split)
    intents = sorted(P.MASSIVE_INTENT_ZH)
    intent_opts = [P.MASSIVE_INTENT_ZH[k] for k in intents]
    scen = sorted(P.MASSIVE_SCENARIO_ZH)
    scen_opts = [P.MASSIVE_SCENARIO_ZH[k] for k in scen]
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for j, i in enumerate(idx[:n]):
        r = ds[i]
        lab = r["label"]
        s = lab.split("_")[0]
        probe = rng.choice(scen)
        if rng.random() < 0.5:
            probe = s
        # the 60-way intent question alone is ~420 tokens; with the 18-way scenario and augmentation
        # headroom a pack reached 561 > BERT's 512. The two never share a state.
        big = j % 2 == 0
        qs = ([Question(f"ms-{split}-{i}-intent", "choice", "這句話的意圖是什麼？", intent_opts, intents.index(lab))] if big else
              [Question(f"ms-{split}-{i}-scen", "choice", "這句話屬於哪個使用情境？", scen_opts, scen.index(s))])
        qs.append(Question(f"ms-{split}-{i}-about", "noul", f"這句話和{P.MASSIVE_SCENARIO_ZH[probe]}有關。", NO, int(probe == s)))
        out.append(Example(state=r["text"], source="massive", meta={"i": i, "split": split, "label": lab, "big": big}, questions=qs))
    return out


def massive_ood(e: Example) -> list[Question]:
    lab = e.meta["label"]
    s = lab.split("_")[0]
    kind = 2 if s == "play" else (0 if (s == "qa" or lab.endswith("query") or s in ("news", "weather")) else
                                  (3 if s == "general" else 1))
    return [
        Question(f"ood-ms-{e.meta['i']}-want", "choice", "使用者想要助理做什麼？",
                 ["查詢資訊", "執行動作或控制裝置", "播放媒體", "聊天或閒聊"], kind),
        Question(f"ood-ms-{e.meta['i']}-play", "noul", "使用者是在要求播放某種媒體。", NO, int(s == "play")),
    ]


def massive_family(e: Example) -> list[Question]:
    lab = e.meta["label"]
    q = int(lab.endswith("query") or lab.startswith("qa_"))
    return [Question(f"tf-ms-{e.meta['split']}-{e.meta['i']}-isq", "noul", "這句話是在問問題、查詢資料。", NO, q)]


# ---------------------------------------------------------------- Amazon reviews (zh)
LEVELS = ["非常負面", "負面", "普通", "正面", "非常正面"]


def amazon(split: str, n: int, rng: random.Random) -> list[Example]:
    from datasets import load_dataset
    ds = load_dataset("mteb/AmazonReviewsClassification", "zh", split=split)
    by = {k: [] for k in range(5)}
    for i, lab in enumerate(ds["label"]):
        by[int(lab)].append(i)
    per = n // 5
    out = []
    for lab in range(5):
        pick = rng.sample(by[lab], min(per, len(by[lab])))
        for i in pick:
            text = ds[i]["text"].replace("\n\n", "：", 1).replace("\n", " ")[:220]
            pol = 0 if lab < 2 else (1 if lab == 2 else 2)
            out.append(Example(state=tw(text, rng), source="amazon", meta={"i": i, "split": split, "level": lab}, questions=[
                Question(f"az-{split}-{i}-level", "score", "這則評論有多正面？", LEVELS, lab),
                Question(f"az-{split}-{i}-pol", "choice", "整體的評價傾向是？", ["負面", "中立", "正面"], pol),
                Question(f"az-{split}-{i}-happy", "noul", "評論者對商品是滿意的。", NO, int(lab >= 3)),
            ]))
    rng.shuffle(out)
    return out


def amazon_ood(e: Example) -> list[Question]:
    lev, i = e.meta["level"], e.meta["i"]
    return [
        Question(f"ood-az-{i}-stars", "score", "評論者會給幾顆星？", ["1 顆星", "2 顆星", "3 顆星", "4 顆星", "5 顆星"], lev),
        Question(f"ood-az-{i}-recommend", "noul", "評論者會推薦朋友買這個商品。", NO, int(lev >= 3)),
        Question(f"ood-az-{i}-disappoint", "score", "評論者聽起來有多失望？", ["完全不會", "有一點", "還算有", "相當", "非常"], 4 - lev),
        Question(f"ood-az-{i}-tone", "choice", "評論的語氣最接近哪個詞？", ["不屑", "平淡", "熱情"], 0 if lev <= 1 else (1 if lev == 2 else 2)),
    ]


def amazon_family(e: Example) -> list[Question]:
    lev, i, sp = e.meta["level"], e.meta["i"], e.meta["split"]
    return [
        Question(f"tf-az-{sp}-{i}-again", "score", "評論者再買一次的可能性有多高？", ["絕不", "不太可能", "也許", "很可能", "一定"], lev),
        Question(f"tf-az-{sp}-{i}-mixed", "noul", "這則評論有好有壞、偏中性。", NO, int(lev == 2)),
    ]


# ---------------------------------------------------------------- OCNLI (native Chinese NLI)
NLI = ["成立", "不成立", "無法判斷"]


def ocnli(split: str, lo: int, hi: int, rng: random.Random) -> list[Example]:
    from datasets import load_dataset
    ds = load_dataset("clue/clue", "ocnli", split=split)
    names = ds.features["label"].names       # neutral, entailment, contradiction
    idx = [i for i in range(len(ds)) if ds[i]["label"] >= 0]
    rng.shuffle(idx)
    out = []
    for i in idx[lo:hi]:
        r = ds[i]
        name = names[r["label"]]
        s1, s2 = tw(r["sentence1"], rng), tw(r["sentence2"], rng)
        g3 = {"entailment": 0, "contradiction": 1, "neutral": 2}[name]
        qs = [Question(f"oc-{split}-{i}-rel", "choice", f"根據上文，這個說法：「{s2}」", NLI, g3)]
        if name != "neutral":                  # "not entailed" is not "false": neutral gets no noul
            qs.append(Question(f"oc-{split}-{i}-holds", "noul", s2, NO, int(name == "entailment")))
        out.append(Example(state=s1, source="ocnli", meta={"i": i, "split": split, "label": name, "hyp": s2}, questions=qs))
    return out


def ocnli_ood(e: Example) -> list[Question]:
    name, i, hyp = e.meta["label"], e.meta["i"], e.meta["hyp"]
    qs = [Question(f"ood-oc-{i}-rel", "choice", "上文和這句話的關係是？「" + hyp + "」",
                   ["互相支持", "互相矛盾", "沒有直接關係"], {"entailment": 0, "contradiction": 1, "neutral": 2}[name])]
    if name != "neutral":
        qs.append(Question(f"ood-oc-{i}-false", "noul", f"根據上文，「{hyp}」這個說法是錯的。", NO, int(name == "contradiction")))
    return qs


# ---------------------------------------------------------------- TNEWS (headlines)
def tnews(split: str, lo: int, hi: int, rng: random.Random) -> list[Example]:
    from datasets import load_dataset
    ds = load_dataset("clue/clue", "tnews", split=split)
    idx = [i for i in range(len(ds)) if ds[i]["label"] >= 0]
    rng.shuffle(idx)
    out = []
    for i in idx[lo:hi]:
        r = ds[i]
        cat = P.TNEWS_ZH[r["label"]]
        probe = cat if rng.random() < 0.5 else rng.choice(P.TNEWS_ZH)
        out.append(Example(state=tw(r["sentence"], rng), source="tnews", meta={"i": i, "split": split, "cat": cat}, questions=[
            Question(f"tn-{split}-{i}-cat", "choice", "這則新聞屬於哪個類別？", list(P.TNEWS_ZH), r["label"]),
            Question(f"tn-{split}-{i}-is", "noul", f"這是一則{probe}新聞。", NO, int(probe == cat)),
        ]))
    return out


def tnews_ood(e: Example) -> list[Question]:
    cat = e.meta["cat"]
    g = next(k for k, (_, s) in enumerate(P.TNEWS_COARSE) if cat in s)
    return [Question(f"ood-tn-{e.meta['i']}-desk", "choice", "這則新聞最可能出現在哪個版面？",
                     [n for n, _ in P.TNEWS_COARSE], g)]


# ---------------------------------------------------------------- C3 (dialogue reading comprehension)
SPEAKER = re.compile(r"^\s*[^：:\s]{1,4}[：:]\s*")


def c3(split: str, lo: int, hi: int, rng: random.Random) -> list[Example]:
    from datasets import load_dataset
    ds = load_dataset("clue/clue", "c3", split=split)
    idx = [i for i in range(len(ds)) if ds[i]["answer"] in ds[i]["choice"]]
    rng.shuffle(idx)
    out = []
    for i in idx[lo:hi]:
        r = ds[i]
        ctx = "\n".join(r["context"])[:300]
        conv = rng.random() < 0.7
        f = to_tw if conv else (lambda x: x)
        opts = [f(c) for c in r["choice"]]
        out.append(Example(state=f(ctx), source="c3", meta={"i": i, "split": split}, questions=[
            Question(f"c3-{split}-{i}", "choice", f(r["question"]), opts, r["choice"].index(r["answer"])),
        ]))
    return out


def c3_turns(split: str) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("clue/clue", "c3", split=split)
    turns = []
    for r in ds:
        if len(r["context"]) < 2:              # monologue passages are not dialogue
            continue
        for t in r["context"]:
            t = SPEAKER.sub("", t).strip()
            if 2 <= len(t) <= 60:
                turns.append(t)
    return sorted(set(turns))


# ---------------------------------------------------------------- the caption gate
class CaptionPools:
    def __init__(self, which: str, rng: random.Random):
        from datasets import load_dataset
        A = which == "A"
        self.rng = rng
        self.scenes = P.SCENES_A if A else P.SCENES_B
        self.sur = P.SURNAMES_A if A else P.SURNAMES_B
        self.given = P.GIVEN_A if A else P.GIVEN_B
        self.brand = P.BRAND_A if A else P.BRAND_B
        self.shoptype = P.SHOPTYPE_A if A else P.SHOPTYPE_B
        self.notice = P.NOTICE_A if A else P.NOTICE_B
        self.place = P.PLACE_A if A else P.PLACE_B
        self.news_prefix = P.NEWS_PREFIX_A if A else P.NEWS_PREFIX_B
        self.role = P.ROLE_A if A else P.ROLE_B
        self.studio = P.STUDIO_A if A else P.STUDIO_B
        self.channel = P.CHANNEL_A if A else P.CHANNEL_B
        self.ui = P.UI_A if A else P.UI_B
        self.promo = P.PROMO_A if A else P.PROMO_B
        self.bless = P.BLESS_A if A else P.BLESS_B
        self.short = P.SHORT_LINES_A if A else P.SHORT_LINES_B
        sp = "train" if A else "validation"
        self.turns = c3_turns(sp)
        m = load_dataset("mteb/MassiveIntentClassification", "zh-TW", split=sp)
        # voice-assistant commands are spoken, but "email 美君" reads like UI text: keep the
        # ones that look like a sentence someone says aloud
        self.utter = [u for u in m["text"] if len(re.findall(r"[\u4e00-\u9fff]", u)) >= 5
                      and not re.search(r"[A-Za-z]", u)]
        t = load_dataset("clue/clue", "tnews", split=sp)
        self.heads = [r["sentence"] for r in t if len(r["sentence"]) <= 40]
        f = load_dataset("clue/clue", "iflytek", split=sp)
        self.apps = [re.split(r"[，。！,!]", r["sentence"])[0][:24] for r in f]
        self.apps = [a for a in self.apps if len(a) >= 4]

    def name(self):
        return self.rng.choice(self.sur) + self.rng.choice(self.given)

    def dialogue(self) -> str:
        r = self.rng
        roll = r.random()
        if roll < 0.75:
            t = tw(r.choice(self.turns), r)
            if len(t) > 20:                                # a subtitle is one or two clauses, not a turn
                segs = [x for x in re.split(r"(?<=[，,。！？!?；;])", t) if x.strip()]
                if len(segs) > 1:
                    k = r.randrange(len(segs))
                    t = segs[k] + (segs[k + 1] if k + 1 < len(segs) and r.random() < 0.4 else "")
                t = t[:28]
        elif roll < 0.90:
            t = r.choice(self.utter)
        else:
            t = r.choice(self.short)
        t = t.strip("，,；; ")
        if r.random() < 0.12:
            t = "……" + t
        return t

    def incidental(self, cat: int) -> str:
        r = self.rng
        if cat == 1:
            k = r.random()
            if k < 0.45:
                return r.choice(self.brand) + r.choice(["", "", "堂", "號"]) + r.choice(self.shoptype)
            if k < 0.60:
                return r.choice(self.sur) + "記" + r.choice(self.shoptype)
            if k < 0.85:
                return r.choice(self.notice)
            p = r.choice(self.place)
            return r.choice([p + "站", f"往{p} {r.randint(1, 12)} 公里", f"{p}區公所"])
        if cat == 2:
            return r.choice(self.news_prefix) + tw(r.choice(self.heads), r)
        if cat == 3:
            parts = [f"{r.choice(self.role)} {self.name()}"]
            if r.random() < 0.5:
                parts.append(f"{r.choice(self.role)} {r.choice(self.studio)}{r.choice(P.STUDIO_SUFFIX)}")
            return "　".join(parts)
        if cat == 4:
            k = r.random()
            if k < 0.4:
                return f"{r.choice(self.channel)} " + r.choice(["HD", "LIVE", "新聞台", "4K"])
            if k < 0.6:
                return f"REC 00:{r.randint(0, 59):02d}:{r.randint(0, 59):02d}"
            return r.choice(self.ui)
        if cat == 5:
            if r.random() < 0.5:
                return tw(r.choice(self.apps), r)
            return r.choice(self.promo).format(n=r.randint(3, 9), item=r.choice(P.PROMO_ITEM_B))
        if cat == 6:
            return r.choice(self.bless)
        raise ValueError(cat)

    def state(self, text: str) -> str:
        r = self.rng
        if r.random() < 0.5:
            text = text.rstrip("。！？.!?")              # terminal punctuation is not a free cue
        scene = r.choice(P.SHOTS) + r.choice(self.scenes)   # drawn independently of the label
        k = r.random()
        if k < 0.55:
            return f"鏡頭：{scene}。畫面上文字的 OCR：「{text}」"
        if k < 0.75:
            return f"鏡頭：{scene}。OCR：「{text}」"
        if k < 0.90:
            return f"畫面上文字的 OCR：「{text}」"
        return f"{scene}的畫面，辨識出的文字：{text}"


def caption(split: str, n: int, pools: CaptionPools, phr_pos, phr_neg, rng) -> list[Example]:
    out = []
    for i in range(n):
        cat = 0 if rng.random() < 0.5 else rng.randint(1, 6)
        text = pools.dialogue() if cat == 0 else pools.incidental(cat)
        for bad in P.FINAL_TEST_FORBIDDEN:
            assert bad not in text, (bad, text)
        is_dlg = int(cat == 0)
        a, b = rng.sample(phr_pos, 2)
        qs = [Question(f"cap-{split}-{i}-d1", "noul", a, NO, is_dlg)]
        if rng.random() < 0.35:
            qs.append(Question(f"cap-{split}-{i}-dn", "noul", rng.choice(phr_neg), NO, 1 - is_dlg))
        else:
            qs.append(Question(f"cap-{split}-{i}-d2", "noul", b, NO, is_dlg))
        qs.append(Question(f"cap-{split}-{i}-cat", "choice", "畫面上這段文字屬於哪一類？", list(P.CAPTION_CATEGORIES), cat))
        out.append(Example(state=pools.state(text), source="caption", meta={"i": i, "split": split, "cat": cat, "text": text}, questions=qs))
    return out


def caption_ood(e: Example) -> list[Question]:
    cat, i = e.meta["cat"], e.meta["i"]
    return [
        Question(f"ood-cap-{i}-dev", "noul", P.NOUL_DEV[i % len(P.NOUL_DEV)], NO, int(cat == 0)),
        Question(f"ood-cap-{i}-src", "choice", "這段文字是從哪裡來的？", list(P.CAPTION_COARSE), P.COARSE_OF[cat]),
    ]


OOD = {"massive": massive_ood, "amazon": amazon_ood, "ocnli": ocnli_ood, "tnews": tnews_ood, "caption": caption_ood}
FAMILY = {"massive": massive_family, "amazon": amazon_family}


def build(out: str, seed: int, scale: float):
    rng = random.Random(seed)
    os.makedirs(out, exist_ok=True)

    final = json.load(open(os.path.join(HERE, "..", "jev_eval", "items_nocue.json")))
    final_phr = set(final["phrasings"]["zh"]["equivalent"] + final["phrasings"]["zh"]["negated"])
    ours = set(P.NOUL_TRAIN + P.NOUL_TRAIN_NEG + P.NOUL_DEV + P.NOUL_DEV_NEG)
    assert not (ours & final_phr), ours & final_phr
    assert not (set(P.NOUL_TRAIN) & set(P.NOUL_DEV))

    S = lambda n: max(1, int(n * scale))
    pa, pb = CaptionPools("A", rng), CaptionPools("B", rng)
    train = (massive("train", S(4500), rng) + amazon("train", S(4000), rng) + ocnli("train", 0, S(4000), rng)
             + tnews("train", 0, S(2500), rng) + c3("train", 0, S(2000), rng)
             + caption("train", S(8000), pa, P.NOUL_TRAIN, P.NOUL_TRAIN_NEG, rng))
    for e in train:
        fam = FAMILY.get(e.source)
        if fam and not e.meta.get("big"):
            e.questions += fam(e)
    val = (massive("validation", S(300), rng) + amazon("validation", S(250), rng)
           + ocnli("validation", 0, S(250), rng) + tnews("validation", 0, S(200), rng)
           + c3("validation", 0, S(150), rng)
           + caption("val", S(300), pb, P.NOUL_TRAIN, P.NOUL_TRAIN_NEG, rng))
    test = (massive("test", S(700), rng) + amazon("test", S(600), rng)
            + ocnli("validation", S(250), S(250) + S(600), rng) + tnews("validation", S(200), S(200) + S(500), rng)
            + c3("validation", S(150), S(150) + S(400), rng)
            + caption("test", S(700), pb, P.NOUL_TRAIN, P.NOUL_TRAIN_NEG, rng))
    ood = []
    for e in test:
        fn = OOD.get(e.source)
        qs = fn(e) if fn else []
        if qs:
            ood.append(Example(state=e.state, source=e.source, meta=e.meta, questions=qs))
    for x in (train, val, test):
        rng.shuffle(x)
    write_jsonl(os.path.join(out, "train.jsonl"), train)
    write_jsonl(os.path.join(out, "val.jsonl"), val)
    write_jsonl(os.path.join(out, "test.jsonl"), test)
    write_jsonl(os.path.join(out, "ood-test.jsonl"), ood)

    # iteration metric: dev phrasings on pool-B states, in the jev_eval items format
    dev_items = []
    for i in range(S(160)):
        cat = 0 if i % 2 == 0 else rng.randint(1, 6)
        text = pb.dialogue() if cat == 0 else pb.incidental(cat)
        dev_items.append({"lang": "zh", "gold": int(cat == 0), "cat": cat, "state": pb.state(text)})
    json.dump({"note": "dev set for iteration: pool-B states, NOUL_DEV phrasings; never the final test",
               "phrasings": {"zh": {"equivalent": P.NOUL_DEV[:5], "negated": P.NOUL_DEV_NEG}},
               "items": dev_items}, open(os.path.join(out, "dev_caption.json"), "w"), ensure_ascii=False, indent=1)

    # the final test with every scene stripped: only the OCR text is left to decide on
    tx = json.loads(json.dumps(final))
    for it in tx["items"]:
        m = re.search(r"(OCR[^：:]*[：:]\s*.*)$", it["state"])
        it["state"] = m.group(1) if m else it["state"]
    tx["note"] = "items_nocue.json with the shot description removed: the OCR text alone"
    json.dump(tx, open(os.path.join(out, "final_textonly.json"), "w"), ensure_ascii=False, indent=1)

    stats = {n: {"states": len(x), "questions": sum(len(e.questions) for e in x)} for n, x in
             (("train", train), ("val", val), ("test", test), ("ood", ood))}
    from collections import Counter
    stats["train_by_source"] = dict(Counter(e.source for e in train))
    import statistics as st
    cap = [e for e in train if e.source == "caption"]
    for name, sel in (("dialogue", [e for e in cap if e.meta["cat"] == 0]),
                      ("incidental", [e for e in cap if e.meta["cat"] != 0])):
        L = [len(e.meta["text"]) for e in sel]
        if L:   # length must not separate the classes on its own
            stats[f"caption_len_{name}"] = {"mean": round(st.mean(L), 1), "median": st.median(L),
                                            "p10": sorted(L)[len(L) // 10], "p90": sorted(L)[9 * len(L) // 10]}
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    build(a.out, a.seed, a.scale)
