#ifndef H3_LORA_H
#define H3_LORA_H

/* LoRA adapters for the DiT, merged into the weights as they load.
 *
 * h3.c's projections run inside fused int8 kernels (QKV + norm + RoPE, FC1 +
 * SwiGLU + FC2), so a runtime side path would mean un-fusing the hot loop. A
 * rank-r delta is cheap to fold in instead: each targeted matrix is read,
 * given s * B @ A, and -- for an int8 weight -- requantized per row with the
 * same rule tools/quantize_h3_int8.py and the GPU quantizer use. The step
 * loop is untouched, so a LoRA costs load time only, never step time.
 *
 * The adapter files are indexed at open and read again for each weight as it
 * loads, so nothing but the index stays resident.
 *
 * Accepted keys: <prefix><module>.<A|B suffix>, where <module> is the native
 * weight name without ".weight" (blocks.7.attn.qkv_proj, token_refiner...,
 * blocks.7.adaln_proj.linear) and
 *   prefix: diffusion_model. | model.diffusion_model. | transformer. |
 *           pipe.dit. | dit. | base_model.model. | (none)
 *   A/B:    lora_A.weight / lora_B.weight, lora_A.default.weight /
 *           lora_B.default.weight, lora_down.weight / lora_up.weight
 *   alpha:  <module>.alpha (scale = alpha / rank; without it, scale = 1)
 * DoRA and full-weight patches (dora_scale, diff, diff_b) are refused rather
 * than half-applied.
 *
 * A fused attn.qkv_proj adapter's B rows are read in ComfyUI's
 * (q|k|v, head, 128) order and reordered to h3.c's (head, q|k|v, 128); see
 * h3_lora_open for the override. */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct h3_lora_set h3_lora_set;

/* spec: comma-separated items, each "PATH_OR_NAME[:STRENGTH][@comfy|@native]"
 * (the suffix is the fused-qkv row order, default comfy). A NAME is
 * looked up under $H3_LORA_DIR (default ~/models/loras/h3) as a file, as
 * NAME.safetensors, or as a directory holding exactly one .safetensors. */
h3_lora_set *h3_lora_open(const char *spec, char *error, size_t error_size);
void h3_lora_free(h3_lora_set *set);

/* 1 when any adapter targets this native weight name. */
int h3_lora_targets(const h3_lora_set *set, const char *weight_name);

/* In-place merges. Return 1 on success (a weight no adapter targets is a
 * no-op success) and 0 with a message on a shape or read error. */
int h3_lora_apply_bf16(h3_lora_set *set, const char *weight_name,
                       uint16_t *weight, uint64_t rows, uint64_t columns,
                       char *error, size_t error_size);
int h3_lora_apply_f32(h3_lora_set *set, const char *weight_name,
                      float *weight, uint64_t rows, uint64_t columns,
                      char *error, size_t error_size);
/* weight = q * scale per row; requantized in place, scales rewritten. */
int h3_lora_apply_i8(h3_lora_set *set, const char *weight_name,
                     int8_t *weight, float *scales, uint64_t rows,
                     uint64_t columns, char *error, size_t error_size);

/* One line per adapter: modules applied / modules in the file, plus the time
 * spent merging. Modules that never matched a loaded weight are named. */
void h3_lora_report(const h3_lora_set *set, FILE *stream);

#endif
