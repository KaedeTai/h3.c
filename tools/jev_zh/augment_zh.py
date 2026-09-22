"""Chinese twin of typed_decisions.augment: same five gold-preserving ops, Chinese surface forms.

The released open-jev paraphrases and negates in English ("Is the following true of the
state? ...", "It is not the case that: ..."). Wrapping a Chinese claim in English would teach
the model a code-mixed format it never meets at inference, so every template here is Chinese.
The op set, probabilities and the pair() contract are unchanged, so the upstream trainer runs
with these swapped in and nothing else touched.
"""
from __future__ import annotations

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "jev_eval", "typed-decisions-src", "src"))
from typed_decisions.schema import Question, NOUL_OPTIONS  # noqa: E402

PARAPHRASE = {
    "choice": ["{i}", "請判斷：{i}", "從下列選項中選一個。{i}", "根據上面的內容，{i}",
               "選出最適合的答案。{i}", "請分類：{i}"],
    "score": ["{i}", "請評分：{i}", "依照下面的量表，{i}", "判斷一下：{i}", "把它放到這個量表上。{i}"],
    "noul": ["{i}", "以下敘述對上面的內容成立嗎？{i}", "說法：{i} 這正確嗎？", "這成立嗎：{i}",
             "是非題：{i}", "請判斷真假：{i}"],
}
NEGATE = ["以下說法是錯的：{i}", "並非如此：{i}", "這樣說不對：{i}", "「{i}」——這句話是錯誤的。",
          "下面這句不成立：{i}"]
SCORE_SYNONYMS = {
    3: [["低", "中", "高"], ["弱", "普通", "強"], ["幾乎沒有", "有一些", "非常"]],
    5: [["最低", "偏低", "中等", "偏高", "最高"],
        ["非常負面", "有點負面", "中性", "有點正面", "非常正面"],
        ["一分", "二分", "三分", "四分", "五分"]],
}


def paraphrase(q, rng):
    t = rng.choice(PARAPHRASE[q.kind])
    return Question(q.qid, q.kind, t.format(i=q.instructions), list(q.options), q.gold)


def shuffle(q, rng):
    if q.kind != "choice":
        return q
    idx = list(range(len(q.options)))
    rng.shuffle(idx)
    return Question(q.qid, q.kind, q.instructions, [q.options[i] for i in idx], idx.index(q.gold))


def negate(q, rng):
    if q.kind != "noul" or q.qid.endswith("~nonneg"):
        return q          # OCNLI "neutral" cannot be negated: "not entailed" is not "false"
    return Question(q.qid, "noul", rng.choice(NEGATE).format(i=q.instructions),
                    list(NOUL_OPTIONS), 1 - q.gold)


def drop(q, rng):
    if q.kind != "choice" or len(q.options) < 4:
        return q
    d = [i for i in range(len(q.options)) if i != q.gold]
    keep = sorted(rng.sample(d, max(2, len(d) - rng.randint(1, len(d) // 2))) + [q.gold])
    return Question(q.qid, "choice", q.instructions, [q.options[i] for i in keep], keep.index(q.gold))


def relabel(q, rng):
    if q.kind != "score" or len(q.options) not in SCORE_SYNONYMS:
        return q
    return Question(q.qid, "score", q.instructions, list(rng.choice(SCORE_SYNONYMS[len(q.options)])), q.gold)


OPS = [shuffle, paraphrase, drop, relabel]


def augment(q, rng, p_negate: float = 0.3):
    out = q
    for op in OPS:
        if rng.random() < 0.5:
            out = op(out, rng)
    if q.kind == "noul" and rng.random() < p_negate:
        out = negate(out, rng)
    return out


def pair(q, rng):
    a = b = q
    for op in (shuffle, paraphrase, drop, relabel):
        if rng.random() < 0.7:
            b = op(b, rng)
    if b.instructions == a.instructions and b.options == a.options:
        b = paraphrase(shuffle(a, rng), rng)
    return a, b
