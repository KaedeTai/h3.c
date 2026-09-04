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
