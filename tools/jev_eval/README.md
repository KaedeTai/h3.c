# open-jev vs the Ornith logit read, on the caption-gate decision

Jev itself is API-only — TypeSafe has never released weights. `open-jev-deberta-v3-large`
(kotoba labs, Apache 2.0) is an independent reproduction of its *shape*: a state and any
number of typed questions go in, calibrated distributions come out of one forward pass,
nothing is generated. 0.4B, 1.6 GB.

The question here is whether that shape beats what the caption gate does today — reading
the yes/no logits off Ornith (35B-A3B, 4-bit, ~18 GB) instead of letting it generate.

## What was measured

24 hand-written states (12 en / 12 zh, balanced), each a shot description plus the OCR of
its on-screen text; the decision is "is this text a transcription of the dialogue, or
incidental text that happens to be in frame". Labels are mine. Five phrasings of the same
yes/no question plus two negated ones, so 168 questions per model.

`items.json` had a confound: every gold=1 state said "bottom"/"lower third" and every
gold=0 said "awning"/"corner"/"over black", so a model could score perfectly off the
position word alone. `items_nocue.json` says only "OCR of on-screen text" for all 24.
**The numbers below are the no-cue control** — the confounded set gave the same picture.

Accuracy at a fixed 0.5 is the wrong first question: a model can carry the whole signal
and still fail it by sitting off-centre, which is exactly what happens here. So AUC first.

## Result (items_nocue.json)

| | open-jev 0.4B | Ornith 35B-A3B logit read |
|---|---|---|
| **en** AUC (mean of phrasings) | **1.00** | 0.44 |
| **en** AUC per phrasing | 0.89 – 1.00 | 0.39 – 0.58 |
| **en** phrasing agreement (Spearman) | **+0.82** | −0.02 |
| **zh** AUC (mean of phrasings) | 0.67 | 0.56 |
| **zh** phrasing agreement | +0.06 | +0.18 |
| negation error \|p(c) + p(¬c) − 1\| | **0.07** | 0.55 |
| ms per question | **9** (warm) | 432 |
| size on disk | **1.6 GB** | ~18 GB |

- **English: open-jev solves it, Ornith is noise.** AUC 1.00 against 0.44, and the five
  phrasings rank the items the same way (+0.82) instead of independently (−0.02).
- **It tracks negation.** 0.07 against 0.55 — Ornith in Chinese answers 是 to a claim and
  to its negation alike (0.70 error), because 是 is also the copula and is a plausible
  first token for almost any Chinese answer.
- **It is off-centre, not wrong.** Its p(yes) lives in 0.09–0.76 and gold=1 items land near
  0.3–0.5, so a 0.5 threshold reads backwards. Fit the threshold; do not use 0.5. Its card
  reports ECE 0.022 in-domain, but this is far out of its training distribution.
- **Chinese is the blocker.** It is an English-only DeBERTa (`language: en`); zh AUC falls
  to 0.67 and phrasing agreement collapses to +0.06. Our actual material is Chinese.

## What this does not show

The states are *descriptions* of frames, not frames. open-jev cannot see images at all, so
this tests the judgement, not the gate end to end. It does not retract the earlier 21/21 on
real frames with a tuned prompt — but it does reproduce, on a second task, what was recorded
then: reading logits does not buy wording-invariance. Here that shows up as agreement −0.02.

24 items is small.

## Next

The recipe is cheap to redo in Chinese: their code is at `kotoba-lang/typed-decisions`, and
the released model was 18k states / 42k questions / 1 epoch / 229 s on one H100 (~$0.25).
A Chinese encoder plus our own EP3 OCR lines would close the one gap that matters.
