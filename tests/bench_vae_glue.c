/* Everything in a video VAE block that is not a matmul, timed at the real
 * shapes. The decode is 8.5 s of pure GPU for 28 tiles x 36 blocks = 8.4 ms a
 * block, and the matmuls and attention only account for about 6 ms of that.
 * This is where the rest is.
 */
#include "h3_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define ROWS 2805u
#define HIDDEN 2048u
#define HEADS 32u
#define HEAD_DIM 64u
#define INNER (HEADS * HEAD_DIM)
#define ROPE_HALF 24u
#define ITERS 60

/* Every dispatch goes into ONE command buffer, the way a real block does.
 * Timing them one submit at a time adds the command-buffer round trip -- about
 * 0.25 ms here -- to every measurement, which is larger than several of the
 * kernels being measured and made the first version of this file useless. */
#define TIME(label, call, bytes) do {                                          \
    if (!h3_gpu_begin(gpu)) break;                                             \
    for (int warm = 0; warm < 4; warm++)                                       \
        if (!(call)) { printf("  %-34s FAILED: %s\n", label,                   \
                              h3_gpu_error(gpu)); break; }                     \
    if (!h3_gpu_submit(gpu)) break;                                            \
    double start = now();                                                      \
    if (!h3_gpu_begin(gpu)) break;                                             \
    for (int i = 0; i < ITERS; i++) if (!(call)) break;                        \
    if (!h3_gpu_submit(gpu)) break;                                            \
    double ms = (now() - start) / ITERS * 1e3;                                  \
    printf("  %-34s %7.3f ms   %6.0f GB/s   x36 = %5.2f ms/tile\n",            \
           label, ms, (double)(bytes) / (ms * 1e-3) / 1e9, ms * 36.0);         \
} while (0)

int main(void) {
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "gpu: %s\n", error); return 1; }

    size_t hidden_n = (size_t)ROWS * HIDDEN, qkv_n = (size_t)ROWS * INNER * 3;
    size_t inner_n = (size_t)ROWS * INNER, rope_n = (size_t)ROWS * ROPE_HALF;
    size_t biggest = qkv_n;
    float *host = calloc(biggest, sizeof(*host));
    uint16_t *host16 = calloc(biggest, sizeof(*host16));
    if (!host || !host16) return 2;
    for (size_t i = 0; i < biggest; i++) { host[i] = 0.01f * (float)(i & 63); host16[i] = 0x3F00; }

    h3_gpu_tensor *hidden = h3_gpu_tensor_from_f32(gpu, host, hidden_n);
    h3_gpu_tensor *norm   = h3_gpu_tensor_new_f32(gpu, hidden_n);
    h3_gpu_tensor *branch = h3_gpu_tensor_from_f32(gpu, host, hidden_n);
    h3_gpu_tensor *weight = h3_gpu_tensor_from_f32(gpu, host, HIDDEN);
    h3_gpu_tensor *scale  = h3_gpu_tensor_from_f32(gpu, host, HIDDEN);
    h3_gpu_tensor *qkv    = h3_gpu_tensor_from_f32(gpu, host, qkv_n);
    h3_gpu_tensor *q = h3_gpu_tensor_new_f32(gpu, inner_n);
    h3_gpu_tensor *k = h3_gpu_tensor_new_f32(gpu, inner_n);
    h3_gpu_tensor *v = h3_gpu_tensor_new_f32(gpu, inner_n);
    h3_gpu_tensor *rope_cos = h3_gpu_tensor_from_f32(gpu, host, rope_n);
    h3_gpu_tensor *rope_sin = h3_gpu_tensor_from_f32(gpu, host, rope_n);
    h3_gpu_tensor *hidden16 = h3_gpu_tensor_from_bf16(gpu, host16, hidden_n);
    h3_gpu_tensor *qkv16    = h3_gpu_tensor_from_bf16(gpu, host16, qkv_n);
    h3_gpu_tensor *q16 = h3_gpu_tensor_new_bf16(gpu, inner_n);
    h3_gpu_tensor *k16 = h3_gpu_tensor_new_bf16(gpu, inner_n);
    h3_gpu_tensor *v16 = h3_gpu_tensor_new_bf16(gpu, inner_n);
    free(host); free(host16);
    if (!hidden || !norm || !branch || !weight || !scale || !qkv || !q || !k ||
        !v || !rope_cos || !rope_sin || !hidden16 || !qkv16 || !q16 || !k16 || !v16) {
        fprintf(stderr, "alloc: %s\n", h3_gpu_error(gpu)); return 3;
    }

    printf("rows=%u hidden=%u inner=%u  (one 320x320 tile)\n", ROWS, HIDDEN, INNER);
    TIME("rms_norm f32",
         h3_gpu_rms_norm_f32(gpu, norm, hidden, weight, ROWS, HIDDEN, 1e-5f),
         2 * hidden_n * 4);
    TIME("cast f32->bf16 (hidden)",
         h3_gpu_cast_f32_to_bf16(gpu, hidden16, hidden, (uint32_t)hidden_n),
         hidden_n * 6);
    TIME("cast bf16->f32 (qkv, 3x wide)",
         h3_gpu_cast_bf16_to_f32(gpu, qkv, qkv16, (uint32_t)qkv_n),
         qkv_n * 6);
    TIME("video qkv rope f32",
         h3_gpu_video_qkv_rope_f32(gpu, q, k, v, qkv, rope_cos, rope_sin,
                                   ROWS, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f),
         (qkv_n + 3 * inner_n) * 4);
    TIME("video qkv rope bf16 (fused)",
         h3_gpu_video_qkv_rope_bf16(gpu, q16, k16, v16, qkv16, rope_cos,
                                    rope_sin, ROWS, HEADS, HEAD_DIM,
                                    ROPE_HALF, 1e-5f),
         (qkv_n + 3 * inner_n) * 2);
    TIME("scale_add f32",
         h3_gpu_scale_add_f32(gpu, hidden, hidden, branch, scale, ROWS, HIDDEN),
         3 * hidden_n * 4);
    h3_gpu_free(gpu);
    return 0;
}
