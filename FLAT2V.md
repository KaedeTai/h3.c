# FLAT2V — first frame, last frame, audio → video

MiniMax-H3 does not lip-sync to a soundtrack you give it. `--ref-audio` clones a
voice; the *words* come out of the text prompt and are denoised from noise
alongside the video. That is why lip sync used to be a lottery and why every
acceleration knob used to wreck the speech: the audio target was the only branch
in the packed layout with nothing anchoring it.

FLAT2V drives h3 in a mode where the audio is an **input**.

## The recipe

```sh
H3_DIT_VARIANT=FL2VA \
H3_AUDIO_ANCHOR=1 \
H3_REF2VA_ANCHOR=1 \
H3_REF2VA_ANCHOR_LAST=2 \
H3_VAE_INT8_FFN=1 \
./h3 -d ./MiniMax-H3-turbo-int8 -p "<the spoken line> (English scene description)" \
  --ref-image head_3to4.png --ref-image head_3to4.png \
  --ref-audio voice_fitted.wav \
  --width 384 --height 512 --frames 158 --steps 3 \
  --use-int8-row-fc2 -o out.mp4
```

51 s for 6.6 s of 384×512 video on an M5 Max cold, **39 s** for every take after
the first in one resident process, against 417 s unaccelerated.
`tools/flat2v.py` wraps all of it; `tools/take_report.py` grades the result.

### What each part is doing

| | |
|---|---|
| `H3_DIT_VARIANT=FL2VA` | FL2VA's weights were never trained on a `REF_AUDIO` segment but parse one anyway — the two checkpoints share one conditioning grammar. FL2VA carries a frame anchor through a whole clip where Ref2VA drops it by frame 2, and it has the stronger turbo LoRA (v1.1 768p vs v0.1). |
| `H3_AUDIO_ANCHOR=1` | Replaces the target audio rows with the reference on its own flow-matching trajectory after every Euler step. `x_s = (1-s)·x₀ + s·ε` is closed form, so this is exact and free. At the last step `s = 0` and those rows *are* the reference. |
| `H3_REF2VA_ANCHOR=1` | Gives reference 1 the RoPE time coordinate target frame 0 will receive. An anchor is one number: keyframes and references take the same path through the DiT and differ only in that coordinate. |
| `H3_REF2VA_ANCHOR_LAST=2` | Same for the last frame. The same image can serve both ends. |
| English scene description | If the prompt is the script, direction written in the spoken language gets spoken. Written in another language it conditions the picture silently. A positive description also removes burned-in subtitles where negation does nothing. |

## Sharp edges

**The voice clip length is exact.** It must encode to `round(frames·40/24)` audio
latent frames. 158 frames wants **6.575 s**, not 158/24 = 6.583 s. h3 refuses a
mismatch rather than misaligning.

**`--frames` must be 5 + 17k.** `h3_video_latent_t` is `((f-5)/17)·5 + 2`: a head
chunk of 2 latent frames spanning 5 real frames (1+4), then chunks of 5 spanning
17 (1+4+4+4+4). Legal: 158, 175, 192, … 311, 328. Ask for anything else and h3
rounds up, and then the fitted voice no longer matches.

**The video VAE encoder is chunk-unaware.** It downsamples time uniformly by 4,
so it is only chunk-correct when handed exactly one chunk. Encoding a 158-frame
clip returns 40 latent frames where the DiT wants 47.

**Crop to the head, not to the room — it also stops the subtitles.** Burned-in
captions were blamed on prompt wording at first. Crossing crop against prompt
says otherwise: **0 of 16** head-crop takes carry a caption and **13 of 17** wide
ones do, and the head crop stays clean even when the prompt tells it it is a wide
clinic shot. Canvas and seed only modulate the wide crop's odds. A caption
belongs to the model's prior for a short-form social clip, and what selects that
prior is what the shot looks like, not what the prompt claims about it. Grade it
with `tools/subtitle_rate.py`.

**Crop to the head, not to the room.** The DiT's cost is tokens and a token is
32×32 px of canvas, so at 480×640 the shirt, tie, bookshelf and certificates are
paying for 300 tokens a frame. A head-and-shoulders crop at 384×512 gives a face
292 px tall (56.9% of the frame) against 222 px (34.7%) for the wide crop at
480×640 — 31% more face for 36% fewer tokens, 62 s against 93 s. The anchors get
sharper too: 3.69–3.90 mean |pixel| against the wide crop's 5.00 at the same
canvas.

**But there is a canvas floor at 384×512 (192 tokens a frame).**

| canvas | tokens | wall | seeds good | anchor 0 |
|---|---|---|---|---|
| 480×640 | 300 | 101 s | 1/1 | 3.20 |
| 416×544 | 221 | 70 s | 1/1 | 4.05 |
| **384×512** | 192 | **62 s** | **4/4** | 3.69–3.90 |
| 352×448 | 154 | 48 s | 2/3 | 3.89–4.01 |
| 320×416 | 130 | 41 s | 1/2 | 5.32–5.68 |
| 288×384 | 108 | 38 s | 0/2 | 5.38–6.80 |
| 192×256 | 48 | 24 s | 0/1 | 10.11 (tail 17.75) |

The control: 288×384 with the **old wide framing** froze as well (aperture
0.0012), so the floor is the canvas and not what is in it. `h3_adapt_canvas`
normalises every request to a 768-pixel short edge — 288 is far outside anything
the checkpoint saw, and the failure is the same fixed point as the layer trim.

**References are cropped, never resized.** h3 reads conditioning images with
`H3_IMAGE_FIT_STRETCH`. A 497×589 crop squeezed into 384×512 narrows the face 11%
for the whole clip.

**Downmix to mono on the way out.** The audio latent is `[32,2,T]` and the DiT
denoises both channels from independent noise, so the L−R difference is pure
artifact: −17.3 dB side-to-mid against a mono reference.

## Grading a take

`tools/take_report.py` prints four numbers. The one that matters most is the one
that took longest to find:

- **aperture** — inner-lip gap over mouth width, s.d. over the clip. Two
  thresholds. **0.05** is the photograph line: below it nothing moves at all.
  **0.088** is the good line, calibrated on two human judgements rather than a
  study — a listener called a 0.0685 take poor and a 0.0926 take very good, same
  face, same canvas. In between, the mouth moves but under-articulates.
  **Check this first.** A still photograph with the
  soundtrack muxed onto it passes every other check: its anchors are perfect
  because nothing moved, its scene is stable for the same reason, and its
  soundtrack is exact because the soundtrack is an input.
- **anchor0 / anchorN** — mean |pixel| against the reference still. Both near 4,
  which is h3's own run-to-run noise.
- **head** — face-centroid travel between frames, in thousandths of a face
  width. **The most sensitive number here**, and the one a viewer reacts to.
  Below **2.5** the head reads as a mannequin. Everything that reuses or skips
  DiT work lands at 1.07–2.00 (core reuse at 2, 3 and 4 steps; 2 steps on its
  own); the recipe lands at 2.98–6.39. A 2-step take passed aperture at 0.0945
  and was a mannequin at 1.73.
- **scene** — mean |pixel| between consecutive frames. Low is good; the camera is
  locked off, so this is drift. It is *not* an articulation measure — it ranked a
  frozen-mouth take 15× above a correctly articulating one. It is also **not
  comparable across framings**: the same head movement fills far more of a tight
  crop, so the identical config reads 0.86 wide and 3.10 tight at one canvas.
- **audio r** — 0.9704–0.9705 on every anchored take ever run here. The missing
  0.03 is the audio VAE round trip, not the anchor.

## Knobs

| Knob | Verdict |
|---|---|
| turbo, 3 steps | **Take it.** Articulation is flat from 2 to 5 steps (aperture 0.091–0.104, which is the seed spread), because the soundtrack is an input and both end frames are pinned. h3 refuses fewer than 2. |
| but not 2 steps | 43 s instead of 51, and visibly waxier: face-region high-frequency energy 0.797 against 0.983 at 4 steps, outside the 0.935–1.064 seed spread. 3 steps sits at 0.955, inside it. Aperture cannot see this — it is texture, not movement. |
| `H3_VAE_INT8_FFN=1` | **Take it.** Video decode 54.1 s → 13.3 s at 480×640. |
| `--core-reuse 4` | **No — off by default.** It stops the head moving, which a viewer calls unnatural before any mouth measure complains: head motion 1.07–2.00 against 3.78–4.28 without it, at every step count tried. It also costs the mouth. It looked like the one accelerator worth having on the wide crop (93 s → 73 s for ~10% of the mouth movement) and is much worse on the head crop: aperture 0.0685 and face detail 0.788 at 4 steps against 0.0926 and 0.982 for plain 3 steps, and a listener called it poor before the metric threshold did. Still costs at 3 steps: 0.0762 and 0.0843 across two seeds. It holds one DiT core across steps, so it damages whatever changes fastest — and on a tight crop the mouth is 57% of the frame instead of 35%. Saving 7 s of 51 is not worth it. |
| `H3_STEEL_ATTN=1` | Off below ~600 tokens a frame. Costs 7.6% at 480×640 (300 tokens); pays at 1152×640 (858). |
| `--layers` | **No.** Buys no measurable time (l50 93 s, l45 88–101 s, l35 74–84 s — a 4-step run is dominated by model load, text encoding and VAE decode) and froze the mouth in 8 of 16 cells, non-monotonically in depth. |
| `--token-reduction` | **No.** Damages the anchors themselves: frame 0 4.25 → 5.70, tail 4.02 → 6.92. |
| `H3_AUDIO_ANCHOR=clean` | **No.** Handing the video the clean audio at step 0 is fine at full depth and collapses the mouth the moment anything else is trimmed. |
| `H3_VIDEO_INIT_SIGMA` | **No.** Seeding from a static clip (SDEdit) fails 5 for 5 even with the still laid out correctly — the remaining steps add texture, not motion. Motion has to be committed near σ=1, which is where the turbo LoRA already removed the steps. |

## Measured

| | head 384×512, 6.6 s | wide 480×640, 6.6 s | wide 480×640, 13.7 s |
|---|---|---|---|
| recipe | **62 s** | 93 s | 299 s |
| + `--core-reuse 4` | 44 s *(rejected: under-articulates)* | 73 s | — |
| aperture | 0.091–0.104 | 0.112 | 0.108 |
| anchor 0 / tail | 3.7–3.9 / 4.0–4.2 | 4.32 / 4.12 | 4.30 / 4.25 |
| tail lock vs no tail anchor | — | 4.12 (13.3) | 4.25 (8.39) |

Long form is linear instead: 26.3 s in **3m 20s** as four chained segments,
where one take of that length extrapolates to ~18 minutes — see below.

Replicated at seeds 42, 7 and 3; at three resolutions plus the head-crop ladder;
at two clip lengths; and on both the full and the stripped text encoder.

## The stripped text encoder

`h3_text_encoder.c` sets `TEXT_LAYERS = 50` and consumes hidden states, not
logits. Language layers 50–63 (12.72 GB) and `lm_head` (1.45 GB) appear nowhere
in the C or Metal sources. `tools/strip_text_encoder.py` removes them:
62.13 GB → 47.97 GB, no code change, verified on four cells against the full
bundle. The originals are kept beside the slim ones as `text_encoder_full/`.

Revert:

```sh
for B in MiniMax-H3 MiniMax-H3-turbo-int8; do
  for V in FL2VA Ref2VA; do
    rm -rf $B/$V/text_encoder && mv $B/$V/text_encoder_full $B/$V/text_encoder
  done
done
```

## Residency: run every seed in one process

`h3_cache_set_enabled(ctx, 1)` appears in exactly one place in the tree —
`h3_cli_run`. Interactive mode caches the prepared DiT, the text embedding, the
video decoder and the whole conditioning; every one-shot `-p` invocation throws
all of it away and pays ~10 s of weights plus ~3 s of reference encoding again.

Measured on the 3-step recipe: **51.0 s, then 39.4, then 39.4.**

`flat2v.py run --seeds 42,7,3` now does this automatically; `--no-resident`
restores one process per seed.

This needed a fix in h3 itself. The audio anchor is the raw `[32,2,T]` encoder
output, a different object from the patchified `REF_AUDIO` condition rows the
cache already held — so on a cache hit the anchor had nothing to write into the
target audio rows and the run refused outright with *"H3_AUDIO_ANCHOR needs a
--ref-audio clip encoded in this run"*. `conditioning_audio_anchor` is now
stored and restored alongside the rest.

## The int4 text encoder

`tools/quantize_text_encoder_int4.py` writes a bundle h3 loads directly:
**48 GB → 15 GB**, no code change to the caller.

**This is now the default, and it is the only text encoder on disk.** The swap
was made in place, inside `<bundle>/<variant>/text_encoder`, rather than by
changing anyone's `-d` argument — so the CLI, the REPL and `flat2v.py` all get it
without knowing.

The BF16 originals were then deleted to reclaim 110 GB. That is safe only
because they were byte-identical to the public release: `MiniMaxAI/MiniMax-H3`
on Hugging Face is public and ungated, all 14 shard sizes matched, and shard 14
hashed to `e45b6c9998c77ee5a6577f9f47bc76416c1d4d387169e50c4c9d3134ea51b13b` on
both sides. FL2VA and Ref2VA carried the *same* text encoder files here — one
set of inodes hardlinked four ways — so one download serves every bundle.

To go back:

```sh
tools/restore_text_encoder.sh slim   # re-download 63 GB, strip, relink
tools/restore_text_encoder.sh full   # re-download and use as-is
tools/restore_text_encoder.sh int4   # only if text_encoder_int4 is gone too
```

Nothing else in the bundle is re-downloadable: the `transformer` in
`MiniMax-H3-turbo-int8` is locally quantised and patched. Do not delete that.

Per projection `<name>` of shape [rows, cols]:

| tensor | dtype | shape | |
|---|---|---|---|
| `<name>.q4` | U8 | [rows, cols/2] | two weights a byte, low nibble first, stored as q+8 |
| `<name>.q4s` | F16 | [rows, cols/32] | one absmax scale per group of 32 along the input axis |

The original BF16 tensor is dropped, so a bundle is int4 or it is not — the
loader reaches the int4 path by the BF16 name lookup *failing*, which makes a
half-quantised checkpoint impossible to ship by accident. The group size is not
recorded anywhere; it is implied by the two shapes.

Only the 350 language-layer projections are touched. Norms are 1-D,
`embed_tokens` is a lookup rather than a matmul, and the vision tower is 1.11 GB.

**No new Metal.** On this hardware the weight buffer is unified memory, so the
dequantisation writes straight into the same bytes the GPU will read: sixteen
products are all the distinct values a group of 32 can take, so the inner loop
is a 16-entry table build and then two lookups a byte. Two load paths had to
learn it — `load_2d` and the prefetching lane reader.

Measured, 3-step recipe at 384×512, seed 42:

| | BF16 | int4 g32 |
|---|---|---|
| text encoder on disk | 48 GB | **15 GB** |
| bytes read per run | 47 GiB | 14 GiB |
| Qwen text encoder wall | 5.50 s | 4.2–4.5 s |
| peak GPU memory | 3.73 GiB | 3.73 GiB *(it already streamed)* |
| aperture | 0.1003 | 0.1048 / 0.0929 / — |
| anchor 0 / tail | 3.52 / 3.69 | 3.55 / 3.73 |

Frame difference from the BF16 take: **3.28** mean |pixel|, against **2.57** for
two BF16 runs of the same command and **5.51** for a change of seed. So int4 is
1.3× the nondeterminism floor and 0.6× a seed change.

What it buys is the footprint, which was the point. Peak memory does not move:
the text encoder already streamed layer by layer at 3.73 GiB, so int4 halves
neither that nor the DiT's 19.7 GiB.

## Quantising the text encoder — the accuracy study that preceded it

`h3_text_encoder.c` is BF16 end to end — every projection calls
`h3_gpu_linear_bf16`. The tree has a deep int8 stack (the DiT and video VAE use
it) and **no int4 anything**, so Q4 is a new-kernel project. Before writing it,
`tools/fakequant_text_encoder.py` answers whether it is worth writing: round the
350 language-layer projections through the grid, store them back as BF16, run
the unchanged pipeline.

| grid | weight error | language layers | aperture | frames moved vs BF16 |
|---|---|---|---|---|
| BF16 (today) | — | 48.76 GB | 0.1154 | 4.25 *(itself, twice)* |
| int8 per-row | 9.58% | 24.43 GB | 0.1123 | 5.44 |
| int8 g128 | 0.74% | 24.76 GB | 0.1120 | 4.34 |
| int4 g16 | 8.96% | 15.24 GB | 0.1130 | **3.52** |
| **int4 g32** | 10.53% | **13.71 GB** | 0.1099 | 4.68 |
| int4 g64 | 11.94% | **12.95 GB** | 0.1127 | 5.58 |

Every row is inside h3's own run-to-run nondeterminism, and one int4 row is
below it. Swapping between two *equivalent BF16 bundles* moves the frames 5.48
— more than int4 g32 does. The picture cannot tell.

What it buys is residency, not speed: the Qwen text encoder is 5.4 s of a 93 s
run and streams 47 GiB doing it. int4 g32 makes the bundle ~16 GB including
norms, embeddings and the vision tower — the difference between keeping it
resident on a 128 GB machine (where the DiT peaks at 20.8 GiB and the video VAE
at 15.8 GiB) and re-streaming it every run, plus maybe 3 s of the 93.

Remaining work is entirely Metal: an int4 group-wise matmul, its dequant path,
and a loader. `h3_gpu_quantize_weight_int8` and `h3_gpu_linear_int8_bf16` are
the shape to copy.

## Long form: chain segments, don't lengthen the clip

Attention is quadratic in the packed sequence and the sequence is mostly video
tokens, so 2.08× the frames costs 4.8× the time — **158 frames in 62 s, 328 in
299 s**. Four chained 158-frame segments cover **26.3 s in 3m 20s**; one take of
that length, on the same curve, is around eighteen minutes.

The seam is free because of two things the recipe already does. The soundtrack
is an *input*, so slicing it and reassembling needs no blending. And a frame
anchor is just a RoPE time coordinate on a reference image, so handing segment
N+1 the last frame of segment N as its **first** anchor is the ordinary
mechanism, not a special case.

```sh
tools/flat2v_long.py init --voice long.wav --ref head_3to4.png \
  --prompt "<script> (English scene description)" --out run/
tools/flat2v_long.py run  run/          # drives one resident REPL
tools/flat2v_long.py check run/
```

### Keep the poster as the last anchor

Each segment's first anchor is the previous segment's *output*, so errors
compound — the classic autoregressive drift. Keeping the **original poster as
every segment's last anchor** bounds it: no segment is ever more than one hop
from the original.

| segment | `--tail poster` | `--tail none` | seam step | typical step |
|---|---|---|---|---|
| 1 | 3.55 | 3.76 | — | 1.22 |
| 2 | 4.78 | 18.50 | 2.97 | 1.31 |
| 3 | 5.32 | 21.95 | 3.02 | 0.97 |
| 4 | 5.23 | 24.47 | 2.91 | 1.01 |

Distance is mean |pixel| against the original poster, 26.3 s, seed 42. With the
tail anchor it rises once and flattens; without it, it climbs and keeps
climbing. The seam measures ~3 against a within-segment step of ~1.2, which is
the same magnitude as h3's own reproduction error for *any* anchor — the four
frames across a seam show continuous pose, expression and background.

Cost: 45.4–46.4 s/segment with `--tail none`, 50.4–55.4 s with `--tail poster`.
Take the poster.

### Cut in the pauses, not on the clock

`tools/cutplan.py`. Legal lengths are 5+17k — about 0.71 s apart — so every
boundary carries ~±0.35 s of slack, which is almost always enough to reach a
breath. Spending that slack is a shortest path over the 40 Hz audio lattice:

```
cost = RMS in a 200 ms window at the boundary (over the clip's mean)
     + 1.2 · how far this span is from the target (long side weighted 1.6)
```

`init --cut silence` is the **default**; `--cut clock` restores the fixed-length
plan. `cutplan.py VOICE` prints both plans and their boundary loudness.

**This is not about the visible seam.** With `--tail poster` the seam looks the
same either way — measured 2.76–2.92 for a clock cut against 2.97–3.04 for a
silence cut, and the boundary frame's mouth is closed in both (0.001–0.003).
The poster anchor *forces* it closed. Drop the tail anchor and the same clock
cut leaves the mouth wide open at every boundary (0.219–0.306), which is what
the audio actually says — so the anchor and the audio are in direct conflict at
every seam.

**The damage lands in the middle of the segment, not at its edge.** Same voice,
same seed 42, same 4 segments, only the cut moved:

| aperture per segment | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|
| `--cut clock` | 0.0824 | 0.0999 | **0.0007** | 0.0639 | 0.0617 |
| `--cut silence` | 0.0889 | 0.0980 | 0.1172 | 0.0915 | **0.0989** |

Segment 3's boundary was the loudest of the three (2.0 × mean RMS at the cut
against 1.6 and 2.4). Forcing a closed mouth there did not produce a small
error — it dropped the whole segment into the photograph fixed point at 0.0007.
Articulation across the rest of the clip fell with it: mid-segment aperture s.d.
0.0607 for the clock cut against 0.0854 for the silence cut.

Waveform correlation is identical (0.9682 both), which is the point — `audio r`
cannot see this. Only aperture can.

So the tail anchor is not free after all; it is free *only when the cut lands
where the mouth was going to be closed anyway*. Silence cutting is what makes
`--tail poster` safe.

Measured on 24.95 s where the clock cuts were deliberately phase-shifted to land
mid-vowel (boundary loudness 2.4 / 1.6 / 3.2 against the planner's 0.0 / 0.0 /
0.0). On a clip whose pauses happen to line up with 6.575 s the two plans agree
and there is nothing to win — which is exactly why the first long-form test
looked clean.

### Dropping the tail anchor does not free the cut point

The obvious escape from the above is to keep only the first anchor, so nothing
forces the mouth shut and the cut can land anywhere. It does avoid the freeze
and it costs the identity instead. Same voice, same seed 42, same four
segments; aperture per segment, and the last segment's distance from the poster:

| | `--tail poster` | `--tail none` |
|---|---|---|
| `--cut clock` | 0.0824 0.0999 **0.0007** 0.0639 — mean 0.0617, drift 4.99 | 0.0600 0.0834 0.0825 0.0723 — mean 0.0746, drift **23.91** |
| `--cut silence` | 0.0889 0.0980 0.1172 0.0915 — mean **0.0989**, drift 5.23 | 0.0837 0.0942 0.0942 0.0951 — mean 0.0918, drift **26.69** |

Read the corners. `tail=none` trades a fixable problem for an unfixable one: it
buys back articulation at a clock cut (0.0617 → 0.0746, no frozen segment) and
the face walks away, 5 → 24. And it is not even the best articulation — with the
cut in the right place, `tail=poster` is *better* than `tail=none`, 0.0989
against 0.0918, because the poster is a second true reference rather than only a
constraint. That is the same effect the short-clip ladder measured as
`H3_REF2VA_ANCHOR_LAST`.

So the tail anchor is not a cost to be escaped. It is a cost only where the cut
is wrong.

### There is no per-segment text to split

The other worry about cutting anywhere is that a boundary lands mid-word and the
segment's prompt has to be half a sentence. It does not, because the spoken line
in the prompt is inert once the audio is anchored. Four prompts against the same
6.575 s clip, the same reference and three seeds — the correct line, no line at
all, a line truncated mid-word, and an unrelated sentence about cycling in
Taipei:

| prompt | seed 42 | seed 7 | seed 3 | mean |
|---|---|---|---|---|
| line + direction | 0.1018 | 0.0927 | 0.0989 | 0.0978 |
| direction only | 0.0955 | 0.0961 | 0.0969 | 0.0962 |
| half a line | 0.0824 | 0.1057 | 0.0916 | 0.0932 |
| wrong line | 0.0859 | 0.0928 | 0.1036 | 0.0941 |

The spread between the four prompts (s.d. of the means, 0.0021) is **a third of
the spread within one prompt across seeds** (0.0065). `audio r` is 0.970 for all
twelve takes to three decimals, and the subtitle rate is 0.0% for all of them.
An unrelated sentence produces the same mouth as the correct one.

The audio is the script. The text is set dressing, and the same string can be
handed to every segment — which is what `flat2v_long.py` already does.

### Segment lengths must be legal

`legal_frames(n) = 5 + max(1,(n-5)//17)*17`, and each slice's audio must be
`round(frames·40/24)/40` seconds exactly. Plan with **floor plus a shorter legal
tail segment**, not `ceil` — ceil leaves a 0.1 s last slice and h3 refuses a
reference clip under 2 s. (The silence planner sidesteps this: it may pad the
last slice by up to one short segment, at 0.8 cost units per second, and picks
the padding as part of the same shortest path.)

Slicing needs `-ss` **before** `-i`. As an output option it only discards
decoded frames and leaves the original timestamps, so `apad=whole_dur` believes
the stream is already long enough and the last slice comes out short.

### Two ways the join destroys the soundtrack

`ffmpeg -c copy` concat is the obvious assembly and it took waveform correlation
from **0.9704 per segment to 0.0040 for the join**. Each segment's AAC stream
carries its own encoder priming delay; four stack into ~130 ms, and the best
single lag only recovers to 0.23 because the error accumulates rather than
offsetting.

Underneath sat a second drift that would have survived that fix: h3 gives 158
frames 263 audio latent frames — 6.575 s — while 158 frames at 24 fps is 6.5833
s. Every segment's video runs **8.3 ms longer than the audio it was generated
against**. Four segments is 33 ms; twenty is 166 ms and audible.

Both go away by joining **video only**, rescaling timestamps so each segment
lasts exactly its audio span (158/6.575 = 24.03 fps — 24 was the
approximation), and laying the **original source audio** over the result:

```sh
ffmpeg -y -f concat -safe 0 -i list.txt -an -c copy video_only.mp4
ffmpeg -y -itsscale 0.998734 -i video_only.mp4 -i voice.wav \
  -map 0:v -map 1:a -c:v copy -t "$AUDIO_SECONDS" \
  -af "pan=mono|c0=0.5*c0+0.5*c1,highpass=f=60" \
  -c:a aac -b:a 128k -ac 1 -ar 32000 joined.mp4
```

Full-length correlation after the corrected join: **0.9735** — better than any
single segment, because the deliverable now carries the source audio rather than
a VAE round trip of it.

### Driver note

Never parse h3's stdout. It is block-buffered through a pipe, so a finished
segment sits on disk while the driver waits forever. `flat2v_long.py` polls the
filesystem.
