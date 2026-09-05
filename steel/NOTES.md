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
