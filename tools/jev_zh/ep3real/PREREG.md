# EP3 real-OCR set: what will be measured, written before any model has seen it

**Why.** The final test (items_final_v2) is spent, and it showed a hole in the synthetic dev
set. This is the first set built from the footage the caption gate actually judges.

**Data.** 923 distinct EP3 videos (md5-deduped, 3.7 h) under ~/movie/ep3, ~/Downloads/ep3,
~/Movies/scenes/ep3_*, ~/ai-image/films/ep3_*. One frame every 2 s (6,422 frames), read with
Apple Vision (`vision_ocr.swift`, zh-Hant/zh-Hans/en), one OCR line = one item, deduped on
folded text. One video excluded: a downloaded interview recording that is not EP3 footage.
Labels are by looking at the frames (`labels.py` says how, and what was auto-accepted).
x-labelled lines (narration calligraphy, pseudo-English subtitle fragments, unclear) are out.
At most 10 lines per video per label. Split dev/test by video (md5 of path, parity).

| | dev | test |
|---|---|---|
| dialogue caption, faithful (sim >= 0.6 to a scripted line) | 33 | 31 |
| dialogue caption, garbled echo (0.35 - 0.6) | 51 | 48 |
| dialogue caption, unmatched (improvised line or glyph soup; confirmed on the frame) | 31 | 38 |
| scene / overlay text with >= 2 Han characters | 43 | 103 |
| scene / overlay text, Latin or digits | 62 | 66 |

**Questions.** items_final_v2's zh phrasings (5 positive, 4 negated), unchanged.

**Metrics, per model.** AUC of the mean positive-phrasing score on test, overall and per tier
(each positive tier against all negatives); phrasing agreement; negation error; accuracy on
test at the threshold that maximises balanced accuracy on dev.

**Models.** open-jev zh v1, v2, v3 (runs/), and English open-jev as the reference.

**Decision rule, fixed now.** v2 stays the release candidate unless v3's overall test AUC
beats it by more than 0.02 *and* v3 is not worse on the faithful tier. The dev-fitted
threshold of whichever is chosen is the threshold the gate will use. Nothing is trained on
this set; if a new round is needed it gets new footage.

## Result (run once, 2026-09-23, after the commit above)

| model | test AUC | per phrasing | faithful | garbled | unmatched | vs Han neg | vs Latin neg | agreement | negation err | dev thr | test bal-acc (acc) | bal-acc at 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **v2** | **0.875** | 0.87-0.88 | **0.993** | 0.911 | 0.732 | 0.875 | 0.874 | +0.99 | 0.17 | **0.824** | **0.812** (0.832) | 0.741 |
| v3 | 0.820 | 0.79-0.85 | 0.984 | 0.862 | 0.635 | 0.859 | 0.760 | +0.98 | 0.13 | 0.141 | 0.714 (0.710) | 0.738 |
| v1 | 0.883 | 0.88-0.88 | 0.991 | 0.921 | 0.747 | 0.890 | 0.872 | +1.00 | 0.19 | 0.583 | 0.827 (0.836) | 0.807 |
| open-jev (English) | 0.717 | 0.41-0.75 | 0.881 | 0.647 | 0.672 | 0.650 | 0.823 | +0.69 | 0.08 | 0.443 | 0.695 (0.703) | 0.587 |

Video-level bootstrap (1000x) of the test AUC difference: v3 - v2 = -0.092 .. -0.021 (95%),
v1 - v2 = -0.017 .. +0.036.

**Decision by the rule above: v2 stays.** v3 is worse, and the interval excludes zero. v1 is
level with v2 (interval straddles zero), so there is no case for going back. The gate's
threshold for v2 is **0.824** -- not the 0.29 fitted on synthetic dev, because real scene
text here is rarely a clean noun phrase; it is glyph soup that v2 reads as a sentence.

What the tiers say:

- **Readable dialogue captions vs everything else: 0.99.** When the OCR is faithful, v2 is
  close to perfect on real footage.
- **Garbled echoes (官只是恢了一种梓披 for 它只是換了一種形態): 0.91.** Still mostly right.
- **Unmatched glyph soup: 0.73, and this is the ceiling of a text-only judge, not a model
  problem.** The worst false positives are phone-screen and reference-sheet soup
  (你么瓜柩给氣 0.99, 份士國與鱼飲餐照午… 0.96) and the worst misses are caption soup
  (銓署簆昊耒阤肿 0.01). Nothing in those strings says which is which; *where* they sit in
  the frame does. The one feature that separates them was removed from the state on purpose,
  because in the synthetic set it leaked the label.

Next, if the gate is to use this: put position/layout in the state ("畫面下方置中，一行，
字幕底"), train on synthetic states that carry it without it being a giveaway, and test on
fresh footage -- this set is now spent.
