/* Does a narrower dtype buy throughput on the M5 matrix units, or only bytes?
 *
 * On Blackwell each step down (BF16 -> FP8 -> FP4) doubles the tensor-core
 * rate, which is why the NVFP4 H3 checkpoints are so much faster than their
 * BF16 originals. Apple publishes nothing, and MLX cannot answer it because its
 * quantized matmul dequantises first. h3.c has both paths written against
 * mpp::tensor_ops, so this times them against each other on one real DiT shape.
 */
#include "h3_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define ROWS 8192u
#define KMAX 5376u
static unsigned K = 5376u;    /* swept: attention reduces over 128 and 32 */
#define Nout 21504u         /* 3 * INNER, the QKV projection */
#define ITERS 8

int main(void) {
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "gpu: %s\n", error); return 1; }
    printf("M5: %d, NAX MLP: %d, int8 MLP: %d\n",
           h3_gpu_is_m5(gpu), h3_gpu_has_nax_mlp(gpu), h3_gpu_has_int8_mlp(gpu));

    size_t in_n = (size_t)ROWS * KMAX, w_n = (size_t)Nout * KMAX, out_n = (size_t)ROWS * Nout;
    uint16_t *host = malloc(w_n * sizeof(*host));
    if (!host) return 2;
    for (size_t i = 0; i < w_n; i++) host[i] = 0x3F80 - (uint16_t)(i & 0x3F); /* ~1.0 */

    h3_gpu_tensor *input  = h3_gpu_tensor_from_bf16(gpu, host, in_n);
    h3_gpu_tensor *weight = h3_gpu_tensor_from_bf16(gpu, host, w_n);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, out_n);
    h3_gpu_tensor *w_i8   = h3_gpu_tensor_new_i8(gpu, w_n);
    h3_gpu_tensor *w_sc   = h3_gpu_tensor_new_f32(gpu, Nout);
    h3_gpu_tensor *in_i8  = h3_gpu_tensor_new_i8(gpu, in_n);
    h3_gpu_tensor *in_sc  = h3_gpu_tensor_new_f32(gpu, ROWS);
    free(host);
    if (!input || !weight || !output || !w_i8 || !w_sc || !in_i8 || !in_sc) {
        fprintf(stderr, "alloc: %s\n", h3_gpu_error(gpu)); return 3;
    }
    if (!h3_gpu_begin(gpu) ||
        !h3_gpu_quantize_weight_int8(gpu, w_i8, w_sc, weight, Nout, K) ||
        !h3_gpu_submit(gpu)) {
        fprintf(stderr, "quantise: %s\n", h3_gpu_error(gpu)); return 4;
    }

    printf("%6s %10s %10s %8s\n", "K", "bf16", "int8", "比值");
    unsigned ks[] = {128, 256, 512, 1024, 2048, 5376};
    for (unsigned ki = 0; ki < sizeof(ks)/sizeof(*ks); ki++) {
    K = ks[ki];
    double flops = 2.0 * ROWS * K * Nout;
    double rate[2] = {0, 0};
    if (!h3_gpu_begin(gpu) || !h3_gpu_quantize_weight_int8(gpu, w_i8, w_sc, weight, Nout, K) || !h3_gpu_submit(gpu)) { fprintf(stderr, "requant: %s\n", h3_gpu_error(gpu)); return 4; }
    for (int pass = 0; pass < 2; pass++) {          /* pass 0 warms up */
        double t0 = now();
        if (!h3_gpu_begin(gpu)) { fprintf(stderr, "begin\n"); return 5; }
        for (int i = 0; i < ITERS; i++)
            if (!h3_gpu_linear_bf16(gpu, output, input, weight, NULL, ROWS, K, Nout)) {
                fprintf(stderr, "  bf16 K=%u: %s\n", K, h3_gpu_error(gpu)); goto int8_pass;
            }
        if (!h3_gpu_submit(gpu)) { fprintf(stderr, "  bf16 submit K=%u: %s\n", K, h3_gpu_error(gpu)); goto int8_pass; }
        double dt = (now() - t0) / ITERS;
        if (pass) rate[0] = flops / dt / 1e12;
    }
int8_pass:
    for (int pass = 0; pass < 2; pass++) {
        double t0 = now();
        if (!h3_gpu_begin(gpu)) { fprintf(stderr, "begin\n"); return 6; }
        for (int i = 0; i < ITERS; i++)
            if (!h3_gpu_linear_int8_bf16(gpu, output, in_i8, in_sc, input,
                                         w_i8, w_sc, ROWS, K, Nout, 0)) {
                fprintf(stderr, "  int8 K=%u: %s\n", K, h3_gpu_error(gpu)); goto report;
            }
        if (!h3_gpu_submit(gpu)) { fprintf(stderr, "  int8 submit K=%u: %s\n", K, h3_gpu_error(gpu)); goto report; }
        double dt = (now() - t0) / ITERS;
        if (pass) rate[1] = flops / dt / 1e12;
    }
report:
    printf("%6u %10.1f %10.1f %8.2fx\n", K, rate[0], rate[1], rate[1]/rate[0]);
    }
    h3_gpu_free(gpu);
    return 0;
}
