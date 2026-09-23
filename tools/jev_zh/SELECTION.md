# Release candidate: v2  (selected on dev, before the pre-registered final test was run)

Dev = data_v2/dev_caption.json: 160 pool-B states, 5 positive + 4 negated phrasings the
training never saw (2 plain negations, 2 compound "scene text, not dialogue" framings).

| | AUC per phrasing | AUC of mean | phrasing agreement | negation err | in-domain / OOD (upstream) |
|---|---|---|---|---|---|
| v1 | 0.99 – 0.99 | 0.99 | +0.97 | 0.18 | 0.821 / 0.649 |
| **v2** | **0.99 – 0.99** | **0.99** | **+0.97** | 0.19 | **0.841 / 0.668**, ECE 0.014 |
| v3 | 0.56 – 0.99 | 0.97 | +0.54 | 0.12 | 0.845 / 0.653 |

v3 did what it was built for -- the scene-side compound error went 0.82 -> 0.02 -- but broke two
plain positive phrasings on dialogue states ("這是對話的逐字字幕" 0.97 -> 0.24,
"...為了讓觀眾看懂對話而加的字幕" 0.95 -> 0.75), so a fix for one phrasing class cost another.
That is whack-a-mole against nine dev phrasings, and more rounds of it would fit the dev set
rather than the task. Stopping here.

v2 is chosen: every plain phrasing, positive or negated, is right and stable (+0.97). What it
does not handle is a claim that affirms the scene side and negates dialogue in one sentence
("這行字是場景裡看得到的東西上面的字，而不是誰說的話": 0.82 error on scene states). The
caption gate asks a fixed question it chooses, so the production recipe is: ask plain claims,
several at once (one forward pass answers all of them), and average.

The final test (jev_eval/items_final_v2.json, committed before the v2 corpus existed) is run
once, after this file is committed, on v1/v2/v3 and both baselines. v2 is the headline number
whatever the others show; nothing is retrained on its result.
