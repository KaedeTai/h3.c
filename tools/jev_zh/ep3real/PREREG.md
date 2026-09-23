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
