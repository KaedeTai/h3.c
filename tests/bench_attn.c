/* MLX steel attention against MPSGraph SDPA, at the shapes h3 actually runs.
 *
 * Measured through the whole pipeline the two looked 11-18% apart, but this
 * machine drifts that much between identical runs as it warms, and the steel
 * runs kept landing second. So time the kernel on its own: same process, same
 * buffers, alternating, many repeats. H3_STEEL_ATTN is read when the shader
 * library is built, so each mode needs its own run of this binary.
 *
 *   DiT:       sequence 14282, 56 heads x 128   (320x576, 243 frames)
 *   video VAE: sequence  2805, 32 heads x  64   (one 320x320 tile)
 */
#include "h3_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void bench(h3_gpu *gpu, const char *name, uint32_t sequence,
                  uint32_t heads, uint32_t head_dim, int iters) {
    size_t count = (size_t)sequence * heads * head_dim;
    uint16_t *host = malloc(count * sizeof(*host));
    if (!host) { printf("  %s: out of memory\n", name); return; }
    for (size_t i = 0; i < count; i++) host[i] = 0x3F00 + (uint16_t)(i & 0x1F);
    h3_gpu_tensor *q = h3_gpu_tensor_from_bf16(gpu, host, count);
    h3_gpu_tensor *k = h3_gpu_tensor_from_bf16(gpu, host, count);
    h3_gpu_tensor *v = h3_gpu_tensor_from_bf16(gpu, host, count);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, count);
    free(host);
    if (!q || !k || !v || !out) {
        printf("  %s: alloc failed: %s\n", name, h3_gpu_error(gpu));
        return;
    }
    float scale = 1.0f / (float)head_dim;
    double best = 1e30;
    for (int pass = 0; pass < iters + 2; pass++) {
        double start = now();
        if (!h3_gpu_begin(gpu) ||
            !h3_gpu_sdpa_bf16(gpu, out, q, k, v, sequence, heads, head_dim, scale) ||
            !h3_gpu_submit(gpu)) {
            printf("  %s: dispatch failed: %s\n", name, h3_gpu_error(gpu));
            return;
        }
        double seconds = now() - start;
        if (pass >= 2 && seconds < best) best = seconds;   /* first two warm up */
    }
    double flops = 4.0 * (double)sequence * sequence * heads * head_dim;
    printf("  %-28s %8.2f ms   %7.1f TFLOP/s\n", name, best * 1e3,
           flops / best / 1e12);
    h3_gpu_tensor_free(q); h3_gpu_tensor_free(k);
    h3_gpu_tensor_free(v); h3_gpu_tensor_free(out);
}

int main(void) {
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "gpu: %s\n", error); return 1; }
    printf("H3_STEEL_ATTN=%s\n", getenv("H3_STEEL_ATTN") ? getenv("H3_STEEL_ATTN") : "(unset: MPSGraph)");
    bench(gpu, "DiT 14282 x 56 x 128", 14282, 56, 128, 5);
    bench(gpu, "video VAE 2805 x 32 x 64", 2805, 32, 64, 20);
    h3_gpu_free(gpu);
    return 0;
}
