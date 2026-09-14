/* Does int8 beat bf16 on attention's matmul shapes once the quantisation is
 * paid for elsewhere?
 *
 * bench_shape says int8 loses badly when the output is narrow -- 0.26x at the
 * 128x128 shape QK^T actually has. But every int8 entry point in h3.c
 * re-quantises its input on each call, while a fused attention kernel would
 * quantise Q and K once per head and reuse them across every tile of the K/V
 * sweep. H3_BENCH_SKIP_INT8_QUANT (a temporary hook, not shipped) leaves the
 * already-quantised buffers alone so the matmul can be timed on its own.
 */
#include "h3_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double timed(h3_gpu *gpu, int iters, int (*call)(void *), void *ctx) {
    (void)gpu; (void)call; (void)ctx; (void)iters; return 0;
}

int main(void) {
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "gpu: %s\n", error); return 1; }
    (void)timed;

    const unsigned ROWS = 32768, KMAX = 5376, NMAX = 21504;
    size_t big = (size_t)NMAX * KMAX;
    size_t fill = (size_t)ROWS * KMAX > big ? (size_t)ROWS * KMAX : big;
    uint16_t *host = malloc(fill * sizeof(*host));
    if (!host) return 2;
    for (size_t i = 0; i < fill; i++) host[i] = 0x3F80 - (uint16_t)(i & 0x3F);
    h3_gpu_tensor *input  = h3_gpu_tensor_from_bf16(gpu, host, (size_t)ROWS * KMAX);
    h3_gpu_tensor *weight = h3_gpu_tensor_from_bf16(gpu, host, big);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, (size_t)ROWS * NMAX);
    h3_gpu_tensor *w_i8   = h3_gpu_tensor_new_i8(gpu, big);
    h3_gpu_tensor *w_sc   = h3_gpu_tensor_new_f32(gpu, NMAX);
    h3_gpu_tensor *in_i8  = h3_gpu_tensor_new_i8(gpu, (size_t)ROWS * KMAX);
    h3_gpu_tensor *in_sc  = h3_gpu_tensor_new_f32(gpu, ROWS);
    free(host);
    if (!input || !weight || !output || !w_i8 || !w_sc || !in_i8 || !in_sc) {
        fprintf(stderr, "alloc: %s\n", h3_gpu_error(gpu)); return 3;
    }

    struct { unsigned k, n; const char *note; } shapes[] = {
        { 128,   128, "QK^T：D=128，輸出一個 tile 寬" },
        { 128,   512, "" },
        { 128,  2048, "" },
        {5376,   128, "深歸約、窄輸出" },
        {5376, 21504, "真實 QKV 投影（對照）" },
    };
    printf("%6s %7s %9s %9s %9s   %8s %8s  %s\n", "K", "N",
           "bf16", "int8全", "int8純", "全/bf16", "純/bf16", "");
    for (unsigned s = 0; s < sizeof(shapes)/sizeof(*shapes); s++) {
        unsigned K = shapes[s].k, N = shapes[s].n;
        double flops = 2.0 * ROWS * K * N;
        int iters = (int)(3e11 / flops); if (iters < 4) iters = 4; if (iters > 300) iters = 300;
        double r_bf = 0, r_full = 0, r_pure = 0;

        unsetenv("H3_BENCH_SKIP_INT8_QUANT");
        if (!h3_gpu_begin(gpu) ||
            !h3_gpu_quantize_weight_int8(gpu, w_i8, w_sc, weight, N, K) ||
            !h3_gpu_submit(gpu)) { fprintf(stderr, "requant\n"); continue; }

        for (int pass = 0; pass < 2; pass++) {
            double t0 = now(); h3_gpu_begin(gpu);
            for (int i = 0; i < iters; i++)
                if (!h3_gpu_linear_bf16(gpu, output, input, weight, NULL, ROWS, K, N)) goto full;
            if (!h3_gpu_submit(gpu)) goto full;
            if (pass) r_bf = flops * iters / (now() - t0) / 1e12;
        }
    full:
        for (int pass = 0; pass < 2; pass++) {
            double t0 = now(); h3_gpu_begin(gpu);
            for (int i = 0; i < iters; i++)
                if (!h3_gpu_linear_int8_bf16(gpu, output, in_i8, in_sc, input,
                                             w_i8, w_sc, ROWS, K, N, 0)) goto out;
            if (!h3_gpu_submit(gpu)) goto out;
            if (pass) r_full = flops * iters / (now() - t0) / 1e12;
        }
        /* The buffers now hold a valid quantisation of this input; time the
         * matmul alone, the way a fused kernel would see it. */
        setenv("H3_BENCH_SKIP_INT8_QUANT", "1", 1);
        for (int pass = 0; pass < 2; pass++) {
            double t0 = now(); h3_gpu_begin(gpu);
            for (int i = 0; i < iters; i++)
                if (!h3_gpu_linear_int8_bf16(gpu, output, in_i8, in_sc, input,
                                             w_i8, w_sc, ROWS, K, N, 0)) goto out;
            if (!h3_gpu_submit(gpu)) goto out;
            if (pass) r_pure = flops * iters / (now() - t0) / 1e12;
        }
        unsetenv("H3_BENCH_SKIP_INT8_QUANT");
    out:
        printf("%6u %7u %9.1f %9.1f %9.1f   %7.2fx %7.2fx  %s\n", K, N,
               r_bf, r_full, r_pure,
               r_bf > 0 ? r_full / r_bf : 0.0, r_bf > 0 ? r_pure / r_bf : 0.0,
               shapes[s].note);
    }
    h3_gpu_free(gpu);
    return 0;
}
