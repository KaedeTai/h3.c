# steel/ — MLX steel attention inside h3-metal

`h3_steel_attention.metal` is a flattened copy of ml-explore/mlx's NAX
(M5 TensorOps) full-attention kernel: `steel/attn/kernels/steel_attention_nax.h`
plus the headers it needs, with the mlx includes inlined and three explicit
instantiations (bf16, bq64/bk32, head_dim 64/96/128, wm4/wn1). It is loaded as
a second `MTLLibrary` when `H3_STEEL_ATTN=1` is set and `h3_gpu_sdpa()` routes
bf16, non-causal, batch-1 attention with head_dim 64/96/128 to it. Both input
layouts h3-metal uses ([N,H,D] row major and [H,N,D] head major) are expressed
as strides, so the MPSGraph transposes disappear.

Numerically verified against a CPU fp32 reference: max |diff| 0.00014 on
N=1000, H=4, D=96 for both output layouts. A 4-step turbo render with the same
seed comes out at 26 dB PSNR against the MPSGraph path — same scene and
composition, different fine detail — which is bf16 accumulation order amplified
by four sampler steps, not an error.

## Shapes that actually matter here

* DiT: 56 heads x **128** (inner 7168, qkv 21504). Not 5376/56 = 96.
* Video VAE: 32 heads x 64, sequence ~1801 per tile-chunk.
* Text refiner: 128.

## How to measure attention kernels on M5 Max (learned the hard way)

Short kernels are dominated by GPU clock state, not code:

* One process after idle runs at boost clocks; the next back-to-back process
  is already throttled. head_dim 64 at 68 ms/call barely moves; head_dim 96
  at 300 ms/call went 42 -> 17 TFLOPS between two consecutive processes of
  the *same* binary.
* So: alternate builds, `sleep 60` before every run, compare position-matched
  pairs, and report sustained (75 s of calls, mean of the last 30 s) separately
  from burst. Burst and sustained can disagree on the sign of a change.
* The "head_dim 96 is 3x slower" and "head_dim 128 is only 20 TFLOPS" numbers
  measured earlier in this project were both artefacts of run order. Paired:
  64 -> 52, 96 -> 42 (51 with the MLX unroll fix), 128 -> 44 TFLOPS burst.

Compile of the specialised pipelines (function constants) costs ~7 s on first
use per shape and is cached by Metal across processes; time the second run.

## Video VAE FFN in int8 (`H3_VAE_INT8_FFN=1`)

Opt-in, implies `H3_VAE_BF16_MLP`. The two FFN matmuls (w1 2048->16384,
w2 8192->2048, ~2/3 of the decoder's FLOPs) run on the existing M5 int8 NAX
linear (`h3_gpu_linear_int8_bf16`: per-output-channel weight scale, dynamic
per-row activation scale). The int8 kernel has no bias input, so the biases
come back through two small kernels, `h3_swiglu_bias_bf16` and
`h3_bias_add_bf16`. Attention projections and SDPA stay BF16.

* Isolated, sustained (2500 blocks back to back, M=1801): BF16 MPS path
  3.55 ms/block, int8 2.05 ms/block. w1 alone 64 -> 107 TFLOPS, w2 61 -> 91.
* Real decode, 1152x640: 1 s clip 10.1 -> 7.8 s; 3 s clip 19.3 -> 14.9 s
  (paired, 120 s cooldown between runs, int8 first). About -22%.
* Quality: 3 s, same seed, against the F32 decoder: int8 42.5 dB / SSIM
  0.984 vs BF16 43.4 dB / 0.986. Deterministic run to run.
* Weight prep (cast + quantization of 36 blocks) 0.27 s.

The first A/B of this path looked like a 2 s *loss*. It was run order: the
BF16 reference went first on a cold machine and the identical DiT in the same
logs drifted 23.8 -> 27.2 s across four runs. With 120 s cooldowns the same
DiT measures 14.8 s in every run. Same lesson as the attention kernels above.

## Energy mode matters more than any kernel above

macOS Energy Mode (System Settings > Battery, or `pmset -g` `powermode`:
0 = Automatic, 2 = High Power) was the unexplained variable behind the
354/395/363/465 s spread of identical 15 s renders. Same binary, same
config (turbo 4 steps, knobs, BF16 VAE, steel attention), 15 s @1152x640,
machine cool, 120 s between runs:

| | Automatic (cold15) | High Power |
|---|---:|---:|
| DiT denoise | 466 s | 309 s |
| video VAE (BF16) | 122 s | 89 s |
| wall | 615 s | 420 s |

Adding `H3_VAE_INT8_FFN=1` in High Power: VAE 89 -> 70 s, wall 402.5 s.
With post-processing on the same machine (RIFE 6 s + Real-ESRGAN 51 s +
ffmpeg 6 s) the whole 15 s timelapse pipeline is ~465 s, level with the
RTX 3090 reference workflow (300 + 181 s).

## Ref2VA turbo (lightx2v ref2v 4-step v0.1)

Byte-patched into `MiniMax-H3-turbo/Ref2VA` with
`~/repos/mtp-cn-tune/patch_h3_turbo.py` (`H3_VARIANT=Ref2VA H3_LORA=...`).
All 208 LoRA modules map onto the Ref2VA checkpoint; rank 128 (qkv fused
384), alpha/rank = 0.0625 as the file's metadata says. The delta is 10-30x
smaller in norm than the FL2V v1.1 768p merge, but it works: base Ref2VA at
4 steps is a blur, the LoRA'd model at 4 steps is clean.

What breaks it is `--core-reuse 4`: with the reference tokens in the stream,
skipping the core blocks turns the UI into a smear and the subject into a
blue glow. `--token-reduction` is usable but ghosts limbs slightly;
`--use-int8-row-fc2` is harmless. 3 s @1152x640: no knobs 141 s, core-reuse
84 s (broken), token-reduction 92 s, int8 fc2 133 s.

15 s @1152x640, 4 steps, int8 fc2 only, int8 VAE, steel, High Power:
DiT 1339 s + VAE 265 s = 1634 s wall (original Ref2VA 8-step run: 3515 s;
FL2VA turbo with all knobs: 402 s). The VAE took 265 s here against 70 s
after the shorter FL2VA denoise with identical settings - the machine was
22 minutes into sustained load, so High Power does not remove throttling,
it only raises the ceiling.

## Where the DiT time goes (M5 Max, 2026-09-05)

Question: fal's H3 Max renders 5 s in ~3 s on GB200; we take 232 s of
denoise for the same 5 s. Is there a software gap hiding in h3.c, or is it
the hardware? Measured with `H3_PROFILE=1` and the `H3_SKIP_ATTN=1` probe,
5 s @1152x640, turbo 4 steps, int8 fc2, steel attention, High Power:

| | denoise (4 evals) | per eval |
|---|---|---|
| full | 232.4 s | 58.1 s |
| attention skipped | 45.4 s | 11.4 s |
| => attention alone | 187.0 s | 46.7 s (80%) |

Token count: 124 frames -> latent_t 37, 72x40 latent, 2x2 patch -> 26,640
video rows + 414 audio rows = 27,054. Per block: linears
2*N*(5376*21504 + 7168*5376 + 3*5376*14336) = 2.09e13 FLOP, full
attention 4*N^2*7168 = 2.10e13 FLOP; 50 blocks.

* Linears: 1.04e15 FLOP / 11.4 s = ~92 TFLOPS. MLX's plain bf16 gemm on
  this machine measures 57 TFLOPS, so the NAX/int8 linear path is already
  above what MLX gets. Nothing to gain there.
* Attention: 1.05e15 FLOP / 46.7 s = ~22 TFLOPS. MLX's own
  `fast.scaled_dot_product_attention` at the same shape (1x56x27054x128,
  bf16) measures 23.2 TFLOPS, so h3.c's steel integration is at parity;
  the kernel itself runs at ~40% of gemm rate (softmax/rescale is not on
  the matrix units).

So the 45x gap to fal is the ~40x per-chip compute gap (B200 ~2.2 PFLOPS
dense bf16 vs M5 Max ~57-90) plus multi-GPU. The only software lever
above 1.5x is attention: it is 80% of the time at 5 s and, being
quadratic, ~92% at 15 s (measured 1339 s DiT for 15 s Ref2VA agrees with
this model to within 20%). A flash kernel that reached gemm rate would
give ~1.9x at 5 s / ~2.3x at 15 s; sparse attention (STA-style) is the
other route, but token-reduction/core-reuse already showed how badly
Ref2VA tolerates training-free approximations.

## Segmented FL2VA (5 x 3 s pinned to keyframes) - works, with two rules (2026-09-05)

Idea: attention is quadratic, so render a 15 s shot as five 3 s FL2VA
segments, each pinned at both ends (`--first-frame` / `--last-frame`) to
keyframes taken from one 3 s "fast-forward" render of the whole story, so
the anchors are mutually consistent and nothing drifts. Tool:
`tools/segmented_fl2va.py` (keyframe clip -> sharpest-frame picks -> one
interactive h3 session for the segments -> concat). Character-creation
timelapse prompt, 1152x640, turbo 4 steps, int8 fc2, layers 45, no knobs.

| | direct 15 s | 5 x 3 s |
|---|---|---|
| wall | 1665 s (DiT 1490 + VAE 116) | 769 s (clip 103 + segments 666) |

Three rounds to get there; what each one taught:

1. With the speed knobs on (core-reuse 4 + token-reduction) it was slower
   than direct (529 s vs 402 s) - once the quadratic term is gone the
   ~50 s of fixed cost per segment (VAE encode of both anchors 2 x 4 s,
   text encoder 7 s, DiT reload 16 s, VAE decode 18 s) wins - and the
   seams were jump cuts. Knobs are dead anyway (see below).
2. Clean keyframes fixed the anchors: three of four seams converged to the
   target keyframe half a second early (PSNR 32-35 dB from frame 60 on)
   instead of snapping on the last frame. But every segment still replayed
   the whole "blank canvas to finished character" arc - the first frame
   was honoured, then frames 2-5 collapsed to 13-17 dB and the canvas
   went back to a sketch.
3. That was the prompt. The shared preamble said the canvas "must start
   completely blank and rebuild the character", and the model obeyed it in
   every segment. Rewriting each segment prompt to state what is ALREADY
   on the canvas and only this segment's change ("the canvas already holds
   a clean black line-art ... the cursor fills flats ...") made all four
   seams continuous (31-38 dB across) and the 15 s monotonic: blank ->
   construction -> line art -> flats -> finished, no reverts.

Rules: (a) no speed knobs; (b) segment prompts describe the current state
and the delta, never the whole arc. Residuals: the 2-5 frames after a
pinned first frame are still a transient (floating UI panels), so
`seam_drop` (default 8, 0.33 s) cuts them at every seam; the colour
picker's hue and small panel details differ between segments, which reads
as the user having picked another colour. The fast-forward clip doubles
as a 103 s storyboard preview. The interactive h3 session does not keep
the DiT across prompts (cache key includes the prompt) - 16 s per shot
that a text-conditioning-only reset would save for every many-shot
workflow.

## Knobs off for good: clean 15 s vs the knobbed "final" (2026-09-05)

Same timelapse prompt, 1152x640, turbo 4 steps, int8 fc2, int8 VAE, steel,
seed 7, High Power. Clean = `--layers 45`, no core-reuse, no token-reduction.

| | knobs (core-reuse 4 + token-reduction, layers 50) | clean (layers 45) |
|---|---|---|
| DiT denoise | ~309 s | 1490 s |
| video VAE | 70 s | 116 s |
| wall | 402 s | 1665 s |

At frame 336 (14 s) the knobbed render has a smeared face (muddy eyes, no
hair clip), a blob for the camera and a doubled hue bar in the colour
panel - the token-reduction ghost; the clean render has a properly drawn
face, a legible camera, clean hoodie strings and a crisp colour picker.
The 4x is real but so is the damage. Defaults everywhere (KaedeStudio,
kaedecode /video/h3) are now int8 fc2 + layers 45 only; core-reuse and
token-reduction are preview-only opt-ins.

## longshot.py - the segmented pipeline as an SOP (2026-09-06)

`tools/longshot.py SCRIPT WORKDIR preview|keyframes|segments|assemble|status`
(skill: `~/.claude/skills/h3-longshot/SKILL.md`). Script = world + subject +
timed beats; every stage stops for review; `segments --only N` re-rolls one.
Timelapse run (ls3): preview 95 s, five segments 125-128 s each, assemble
with music + RIFE 48 fps + ESRGAN x2 ~3 min; ~16 min wall to a 2304x1280
deliverable, one segment re-rolled once.

What the runs taught, beyond the two rules:

* The fast-forward likes to OPEN on the final state (3 of 5 previews across
  seeds/prompts). Saying the opening state first and last in the prompt
  helps but is not reliable; re-rolling the seed is the fix (100 s each).
  Rendering an "opening frame" from the world text alone does not work
  (it painted the subject, then scribbles), and pinning a blank frame
  borrowed from ANOTHER run broke seam 0: the clip drifted to its own UI
  layout and segment 0 could not reach keyframe 1. So `first_frame` only
  from the same kind of render, otherwise none.
* Keyframe picks must reject transient popups: the sharpest frame in a
  window is often the one with a dialog box. Candidates that stray from
  the window's per-pixel median are dropped first.
* Every segment needs visible, natural progress. "line art -> line art
  tidied" made the model colour and un-colour the character on two seeds;
  a partial-colour end state (hair only) wandered on one seed and held on
  another. Prefer end states that are natural stopping points.
* Transients on both sides of a pin: 2-5 frames after a first frame and
  2-3 before a last frame (a flash of the finished character at frame 71
  of the construction-sketch segment). `seam_drop` 8 / `seam_drop_before`
  3 cut them.

## Talking head (Ref2VA) cost/quality, measured on a real person (2026-09-06)

Use case: one known person to camera, voice supplied by Breeze TTS. All our
earlier Ref2VA numbers were 15 s of UI-heavy timelapse; this measures the
shape that actually matters. 6 s, one reference photo (a face crop from real
footage), Breeze-cloned voice track, int8 fc2, steel, no speed knobs.

| | size | steps | wall | DiT | note |
|---|---|---|---|---|---|
| turbo | 1152x640 | 4 | 368 s | 270 s | identity holds, lipsync tracks |
| turbo | 768x448 | 4 | 125 s | - | see below, unusable |
| base | 1152x640 | 20 | 1669 s | 1404 s | clearly better face |

* **FL2VA cannot be driven by an audio track.** `--ref-audio` is an ordered
  reference, so it sets `reference_count`, which both selects Ref2VA and
  trips "full references cannot be combined with frame anchors". Supplying a
  voice to lipsync to is Ref2VA-only; FL2VA generates its own audio.
* **One reference image is enough for identity.** No LoRA needed for "look
  like this person" - and a LoRA would not change the compute anyway.
* **Low-res + upscale is dead for a photoreal face**, unlike the anime-style
  timelapse where it was merely lossy. 768x448 renders in a third of the
  time, but realesr-animevideov3 turns hair into a black gel and erases skin
  texture; realesrgan-x4plus leaves visible tile seams at -s 2. Faces are
  where the viewer looks. Render the face at native size.
* **base 20 vs turbo 4 is a real quality gap**, larger than on the animated
  timelapse: turbo leaves the eyes mushy and the glasses frame soft, base
  resolves eyes, catchlights, teeth and hair strands. 4.5x the wall time.
  Turbo for drafts and internal cuts, base for anything published.
* Cost per second of video is superlinear in take length (6 s = 61x
  playback, 15 s = 109x), so cut a talking head into 5-8 s takes - which is
  normal edit grammar anyway - rather than one long take.

## Third-party Ref2VA fine-tune: WarmBloodAban/Minimax-h3_Singularity (2026-09-06)

Evaluated as a possible fix for turbo Ref2VA's soft faces. Verified from 3 MB of
safetensors headers plus 20 MB of showcase video - no full download needed.

**Compatibility: yes, for the 34 GB file only.**
* `..._ref2va_v1.3_int8.safetensors` (34 GB) is structurally identical to ours:
  1035 tensors = our 535 plus a `weight_scale`/`comfy_quant` pair per quantized
  weight; all 50 DiT blocks and 2 token_refiner blocks; every shape matches
  (`qkv [21504,5376]`, `fc1 [28672,5376]`, `fc2 [5376,14336]`,
  `adaln_proj [96768,2688]`). Names differ only by a `model.diffusion_model.` prefix.
* `..._Pruned_v1.3_int8.safetensors` (21 GB) is NOT usable: it replaces
  `adaln_proj.linear.weight` with `[96768, 8]`. That is a structural change, not
  a quantization.
* Its quantization is the scheme h3.c already uses: per-row symmetric int8,
  `weight I8 [out,in]` + `weight_scale F32 [out,1]`, matching
  `h3_quantize_bf16_int8_rows` (`scale = max|w|/127`, row-major char).
  So it could be loaded straight into the existing `*_int8`/`*_scales` tensors,
  skipping both the BF16 read and the load-time quantize.

**Structural note learned on the way:** `adaln_proj` is 96768x2688 per block, so
50 blocks = 13 G params = 26 of the transformer's 66 GB - **40% of the model**.
That is what the "Pruned" build throws away.

**Not adopted.** Every video in the repo's showcase is anime/CG action (2D anime
fighters; a stylized 3D character with magic effects) - not one photoreal human.
A fine-tune showcased entirely on anime action was very likely trained on it, and
its advertised fixes read that way too ("distant face restoration" = anime faces
in wide action shots, "de-oiled" = plastic CG skin). That pushes the aesthetic
away from a photoreal talking head. Download speed here is ~1.4 MB/s (~7 h for
34 GB; parallel connections do not help, the line saturates near 2-3 MB/s), so
the cost was not worth an aesthetic bet in the wrong direction.

Revisit if the anime/CG timelapse work ever needs better base quality - the
compatibility groundwork above is done.

## Pre-quantized int8 checkpoints (2026-09-06)

h3.c read 66 GB of BF16 per transformer and then quantized the four big
per-block projections to int8 on the GPU *on every load*
(`quantize_block_mlp/qkv/attention_out`). `tools/quantize_h3_int8.py` does that
step once, offline, and h3.c now loads the int8 directly.

The converter is a port of `h3_quantize_bf16_int8_rows`, so the numbers are the
ones the GPU would have produced:

    max_abs = max |float32(w)| per row
    scale   = max_abs > 0 ? max_abs/127 : 1/127
    q       = clamp(rint(float32(w) * (127/max_abs)), -127, 127)

Layout is `<name>` I8 [rows, cols] plus `<name>_scale` F32 [rows], matching the
per-row convention ComfyUI-style int8 exports use, so the same loader reads those
too (see the Singularity note above). adaln_proj is left BF16 - it feeds a BF16
kernel - which is why the output is 47 GB rather than the 34 GB a fully
quantized export reaches.

| | BF16 checkpoint | int8 checkpoint |
|---|---|---|
| transformer on disk | 66.3 GB | 47.0 GB (71%) |
| DiT load, 384x512 Ref2VA | 14.06 s | **6.93 s** |
| DiT total (load + 4-step denoise) | 43.0 s | 37.0 s |
| conversion time | - | 28 s per transformer |

**Verifying equivalence needs a control.** The int8 render is not byte-identical
to the BF16 one - but neither are two BF16 runs of the same command. Measured on
frames 0/40/80/120, mean |diff| per pixel:

    BF16 vs BF16 (same command, twice)   2.52  11.44  3.97  4.06
    BF16 vs int8 checkpoint              3.02  10.79  4.13  5.21

h3 is not bit-deterministic run to run (GPU reduction order), and the int8
checkpoint sits inside that noise. Faces compared side by side are
indistinguishable.

Gotchas found while building it: the Ref2VA checkpoint is detected by the mere
presence of `Ref2VA/transformer/model.safetensors.index.json`, and `config.json`
must sit beside the shards - a converted directory that has neither loads as
"ordered references require the Ref2VA checkpoint" or "missing required model
file". The converter now writes both.

`MiniMax-H3-turbo-int8/` is built with hardlinks to the shared text encoder and
VAEs, so it costs 88 GB on top of the BF16 bundles rather than 161 GB. Deleting
`MiniMax-H3-turbo/` once the int8 bundle is trusted frees 123 GB.

**int8 is the default everywhere as of 2026-09-06.** `MiniMax-H3-turbo/` (BF16)
was archived to `/Volumes/2T/h3-models/MiniMax-H3-turbo` and removed from the
SSD; the external copy is exFAT, so the hardlinks materialise and it occupies
~269 GB there rather than 123 GB. Consumers now point at
`MiniMax-H3-turbo-int8/`: KaedeStudio `StudioSettings.h3LocalTurboModelDir`,
`tools/longshot.py`, `tools/segmented_fl2va.py`, and the kaedecode sidecar
(which prefers int8 -> turbo -> base, whichever exists first).
`tools/patch_h3_turbo.py` still emits BF16 by design - its output has to be
re-quantized before anything will use it.

## The reference crop must be cropped to the canvas aspect, never resized to it

Every head-shot take before 2026-09-06 15:40 (`E`, `F`, `G` in
`~/models/_wang_test`) used `wang_head_ref.png`, which was the poster region
`(122,187,497x589)` *resized* to 384x512. 497/589 = 0.844, 384/512 = 0.750, so
the reference face was squeezed 11% horizontally - and Ref2VA reproduced the
squeeze faithfully for the whole clip. It reads as "he looks thinner than the
photo", which is easy to blame on the model.

The fix is to make the crop box itself match the canvas aspect and then scale
uniformly. `wang_face34_native.png` is `(115,180,444x592)` - 444/592 = 0.750
exactly - and 444/384 = 592/512 = 1.15625, 444/480 = 592/640 = 0.925, so both
canvases are pure downscales of the same box. Assert it in code rather than
eyeballing it:

    assert crop.size[0] / w == crop.size[1] / h

Same 20-step base run, same audio, only the reference changed:

| take | canvas | reference | denoise |
|---|---|---|---|
| F | 384x512 | squeezed | 185.5 s |
| H | 384x512 | cropped 3:4 | 184.8 s |
| I | 480x640 | cropped 3:4 | 356.6 s |

So the fix is free, and the aspect error was never a speed/quality tradeoff -
just a bad crop. 480x640 is 300 tokens against 384x512's 192 (1.56x), and costs
1.93x the denoise time, close to the N^2 attention prediction. It is worth it
for a face: teeth, glasses rims and skin texture are all visibly better.

`--ref-image-size` (match|max) does not rescue this - it fits the reference to
the canvas, it does not letterbox it.

## `--ref-audio` is a voice reference, not a soundtrack - the prompt is the script

Every Ref2VA take from `A` through `I` produced audio nobody could understand,
in the doctor's voice. Transcribing the muxed audio with whisper-large-v3 shows
what it actually is:

    F  "松哥哥的更魂傾向和成語無極計 / 真實的方便質感 /
        畫面中沒有任何字幕和浮水印"
    H  "頓光領導人強大前方小齊看賞 / 聖醫師錢吉德公 / 寫實記錄片豐功"

Those are the *prompt* - "寫實紀錄片風格", "畫面中沒有任何字幕和浮水印" - read
aloud. H3 is a joint audio-video model: it generates the speech, and the text
prompt is what it says.

The layout in `h3_dit.c` says it plainly. `H3_SEG_REF_AUDIO` accumulates into
`audio_condition` and gets `schedule->audio_condition_rows` (a clean, fixed
timestep); `H3_SEG_AUDIO` accumulates into `audio_target` and gets
`schedule->audio_rows` (the denoising ladder). The reference clip conditions
timbre; the soundtrack is denoised from noise alongside the video. The upstream
ComfyUI graph agrees - `VAEDecodeAudio` on a generated latent, and no audio
input node at all.

So there is no lipsync-to-a-track mode to find, and Breeze TTS cannot drive
these shots. Put the line in the prompt instead:

| take | prompt | transcript of the result |
|---|---|---|
| J | 畫面描述 + 他說：「台詞」 | line, then mush from 3.0 s on |
| K | 台詞 only | the line, clean |

K is the recipe: **the dialogue is the prompt**. Scene description competes with
it for the audio branch and corrupts the tail, so keep visual direction short or
leave it out and carry framing with `--ref-image` and the canvas instead.

Breeze's job here is not the soundtrack - it is generating the 5-10 s voice
reference that `--ref-audio` clones.

## The audio artifact is the side channel, and mono analysis cannot see it

Reported as "obvious noise" on the turbo takes. Three measurement attempts found
nothing, because all three averaged the stereo track to mono first:

- gap-frame spectra: the long 2.9-3.5 s pause is dark in every clip
- CPP (harmonic definition): 0.95-1.05 for every take, reference included
- 5-11 kHz energy by loudness decile: generated clips mostly *quieter* than the
  reference

The first pass also reported a false result: the per-group spectra were
normalised to each group's own peak, so a highpass that removed low-frequency
rumble made the high band look worse. Any band comparison has to be absolute
dBFS with the clips gain-matched.

What is actually wrong: `h3_audio_latent` is `[32,2,T]` - genuinely stereo - and
the DiT denoises both channels from independent noise. They converge on the same
waveform only if given enough steps. Side channel relative to mid:

| take | steps | side/mid |
|---|---|---|
| reference wang_line.wav | - | true mono, no side at all |
| I 384x512 | base 20 | -29.2 dB |
| M 480x640 | base 30 | -25.3 dB |
| L 480x640 | base 20 | -23.8 dB |
| J 480x640 | turbo 4 | -18.7 dB |
| K 384x512 | turbo 4 | **-17.3 dB** |

The side content sits at 1-8 kHz (K: -34.8 / -38.1 dB), right on top of the
voice, which is why it reads as a wide phasey hiss around the speech rather than
as hiss in the silences.

The fix is `tools/clean_h3_audio.sh`: collapse to mid. The voice reference is
mono, so there is no stereo information to protect, and because the in-phase
noise in the two channels is independent too, the downmix takes another 3 dB off
what remains. whisper-large-v3 transcribes the cleaned L and M takes exactly as
before, so nothing of the speech was lost.

Order of operations that matters: more steps shrinks the side channel (turbo
-17.3 -> base 20 -23.8 -> base 30 -25.3), but mono collapse removes it entirely,
so the cheap fix dominates. Do not buy steps for audio; buy them for video.

## A speed knob is safe for the soundtrack only if it touches video rows alone

Listening test on the four 480x640 base-20 takes, ranked by how badly the speech
doubles over itself:

| take | knob | speech | picture |
|---|---|---|---|
| L | none | clean | reference |
| P | `--token-reduction` | **clean** | horizontally soft |
| O | `--core-reuse 4` | doubles for the first ~0.6 s | ok |
| N | `--layers 45` | doubles for ~2.9 s | ok |
| R | (no knob) | destroyed | - |

The spectrograms show exactly that: L and P have a dark gap at 2.93-3.52 s and
clean harmonic stacks; O fills the first 0.6 s where L is silent; N fills most of
0-2.9 s with a second harmonic texture and largely closes the gap.

`token_pool_sources()` explains it in four lines:

    if (reduced_row < dit->video_target_start) {
        *first = reduced_row; *second = reduced_row; return;
    }

Every row before `video_target_start` - text, the ordered references, and the
whole audio segment - is returned unpooled. Token reduction is a video-only
transform, so the audio branch never sees it. `--layers` and `--core-reuse`
skip or reuse *whole DiT blocks*, and the audio rows sit in the same packed
sequence, so they take the damage too. More blocks skipped = longer doubling,
which is why N is worse than O.

The cost of the one safe knob is anisotropic. The pairing runs along
`spatial_width`:

    uint32_t spatial_width = (uint32_t)dit->latent_w / 2;
    ... (local % reduced_width) * 2;

so horizontal resolution is halved and vertical is untouched. Measured as
high-frequency energy along x minus along y on frames 40/100: L +1.48 dB,
P -2.04 dB. That 3.5 dB is the "slightly squashed" look - it is real, and it is
horizontal-only by construction.

Practical rule for a talking head: **no knobs**. `--token-reduction` is the only
one that keeps the speech, and it costs horizontal detail on the one thing the
shot is about. Save it for previews.

## There is no negative prompt: the whole string is the script

Take R appended one short clause to the line - 「畫面中沒有任何字幕或浮水印。」
- and two things happened: the burned-in subtitle rendered those exact words,
and the speech was destroyed outright. Take Q appended
「固定鏡頭臉部特寫...」 and the model simply spoke it as another sentence.

So the prompt is not "description plus instructions". It is the script, and
anything in it is a candidate for being spoken and for being drawn as a subtitle.
Wardrobe and framing have to come from `--ref-image`, not from words.

Subtitles appear regardless and cannot be prompted away. Detected by the
saturated glyph yellow in the lower 45% of the frame:

| take | subtitle rows (of 640) |
|---|---|
| Q | 507-534 |
| R | 509-565 |
| L | 536-615 |

Cropping the bottom 128 px clears every case seen so far, at the cost of turning
a 480x640 render into 480x512.

## Core reuse can keep the soundtrack: re-run the blocks over the prefix

The audio target is ~2% of the sequence - 263 rows against 12,000 for a 6.5 s
480x640 shot (`audio_t = frames * H3_AUDIO_LATENT_FPS / H3_FPS`, video rows =
`latent_t * (latent_h/2) * (latent_w/2)`). Freezing it saves nothing and costs
the speech, so `--core-reuse` should never have been applied to it.

The layout makes the fix cheap. `h3_layout_pack` emits `H3_SEG_AUDIO` before
`H3_SEG_VIDEO`, and token reduction already asserts that the target video *ends*
the packed layout. So the audio side is the contiguous prefix `[0,
video_target_start)` and the video side is the contiguous tail - no repacking
needed for either half:

- the cached residual goes to the tail only, via a new `h3_gpu_add_bf16_range`
  that is just a Metal buffer offset;
- the block loop now runs on reuse steps too, with `core_reuse_audio_pass` set,
  and `run_block` clamps `rows` to `video_target_start`.

Only a full pass may refresh the cache - after an audio pass the video tail has
not been through the blocks, so `hidden - core_input` is not a core residual.
`H3_CORE_REUSE_FREEZE_AUDIO=1` restores the old behaviour as a control.

The approximation: on a reuse step the audio rows attend to text + references +
audio but not to the (stale) video rows. Full cross-attention returns on every
evaluate step.

480x640, 158 frames, base 20 steps, seed 42, dialogue-only prompt:

| take | | wall | speech |
|---|---|---|---|
| L | no knobs | 6:49 | clean |
| T | `--core-reuse 4`, audio frozen (old) | 2:46 | doubles for ~0.6 s |
| S | `--core-reuse 4` + audio pass | **2:55** | **clean** |

The audio pass costs 5.7% over the frozen path and still runs **2.33x faster
than no knobs at all**. Spectrograms: S has L's dark 2.93-3.52 s gap and sharp
harmonic stacks; T fills the first 0.6 s. whisper reads S's opening line
correctly where T gives 「代謝變慢是確立核問題」.

Still open: whether core reuse hurts the *picture* on base Ref2VA. The turbo
verdict in the section above was measured at 4 steps and does not transfer, but
the takes here diverge too much in framing to compare quality frame by frame, so
KaedeStudio still refuses core reuse on ref2va shots.

### Does core reuse hurt the picture on base Ref2VA? Not measurably.

First attempt said yes, and it was wrong twice over.

Whole-frame sharpness put L (no knobs) at 11.84 Laplacian against S's 9.28 - but
L's background is a wall of fine certificate text and S's is a plainer shelf, so
that compares content. The giveaway was the anisotropy: S read -7.05 dB
horizontal-vs-vertical without ever touching `--token-reduction`, which is
impossible as a filtering effect and obvious as a content effect (S wears a
striped tie in front of vertical shelf edges).

Restricting to the face - same person, same light, cropped by skin tone, resized
to a common width and contrast-normalised - and adding a second seed:

| | no knobs | core-reuse 4 |
|---|---|---|
| seed 42 | 0.2896 | 0.2608 (-10%) |
| seed 7 | 0.1981 | **0.2227 (+12%)** |

The two seeds disagree in sign, and seed-to-seed spread (0.198 to 0.290, 46%) is
four times the knob effect. So there is no detectable picture penalty; there is
also not enough evidence to claim there is none. Two seeds is two seeds.

### The wardrobe drift is the seed, not the knob

At seed 42 the split looked perfect: L and P (no knobs, token-reduction) kept the
white coat; N, O, S and U (layers 45, core-reuse) all put the doctor in a dark
suit. Four against two, one mechanism - core reuse stops re-reading the
reference - and it matched the old turbo verdict exactly.

Seed 7 killed it. **Both** seed-7 takes drift: the unknobbed one is in a dark
suit too, and it moves the subtitles to the top of the frame; the core-reuse one
adds a handheld press microphone and a news-interview set. The knob had nothing
to do with it.

A dialogue-only prompt gives the model no visual anchor at all, so wardrobe,
background and even subtitle placement are decided by the seed. Widening the
reference to include the chest and the white coat (take U) did not fix it either
- so this is not "the reference does not show enough coat", it is the prompt
carrying no visual constraint while the reference only pins the face.

Identity survives everywhere: it is the same man in all eight takes. What drifts
is everything the reference crop does not cover.

Practical consequence: a talking-head shot needs more than one roll. That is
affordable now - core-reuse 4 with the audio pass is 2:55 against 6:49 - so the
speedup buys takes rather than buying a single faster take.
