/* Is int8 slower than bf16 at a short reduction, or was that the output?
 *
 * bench_dtype swept K with the output pinned at 21504 columns, so at K = 128 the
 * call wrote 352 MB of BF16 for 45 GFLOP of work and measured the write, not the
 * matrix units. Flash attention never writes QK^T to memory at all. This sweeps
 * both dimensions so the two effects can be told apart.
 */
#include "h3_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "gpu: %s\n", error); return 1; }

    const unsigned ROWS = 32768, KMAX = 5376, NMAX = 21504;
    size_t big = (size_t)NMAX * KMAX;
    /* the input is ROWS x KMAX, which is larger than the weight; one array
     * feeds both, so it has to be sized for the larger of the two. */
    size_t fill = (size_t)ROWS * KMAX > big ? (size_t)ROWS * KMAX : big;
    uint16_t *host = malloc(fill * sizeof(*host));
    if (!host) { fprintf(stderr, "host alloc\n"); return 2; }
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
        fprintf(stderr, "alloc: %s\n", h3_gpu_error(gpu)); return 2;
    }

    struct { unsigned k, n; const char *note; } shapes[] = {
        { 128,   128, "注意力 QK^T 的形狀" },
        { 128,   512, "" },
        { 128,  2048, "" },
        { 128, 21504, "bench_dtype 用的（輸出綁）" },
        { 512,   128, "" },
        {2048,   128, "" },
        {5376,   128, "深歸約、小輸出" },
        {5376, 21504, "真實 QKV 投影" },
    };
    printf("%6s %7s %10s %10s %9s %9s  %s\n",
           "K", "N", "bf16", "int8", "比值", "輸出MB", "");
    for (unsigned s = 0; s < sizeof(shapes)/sizeof(*shapes); s++) {
        unsigned K = shapes[s].k, N = shapes[s].n;
        double flops = 2.0 * ROWS * K * N;
        int iters = (int)(4e11 / flops); if (iters < 4) iters = 4; if (iters > 400) iters = 400;
        double rate[2] = {0, 0};
        if (!h3_gpu_begin(gpu) ||
            !h3_gpu_quantize_weight_int8(gpu, w_i8, w_sc, weight, N, K) ||
            !h3_gpu_submit(gpu)) { fprintf(stderr, "requant\n"); continue; }
        for (int pass = 0; pass < 2; pass++) {
            double t0 = now(); h3_gpu_begin(gpu);
            for (int i = 0; i < iters; i++)
                if (!h3_gpu_linear_bf16(gpu, output, input, weight, NULL, ROWS, K, N)) goto i8;
            if (!h3_gpu_submit(gpu)) goto i8;
            if (pass) rate[0] = flops * iters / (now() - t0) / 1e12;
        }
    i8:
        for (int pass = 0; pass < 2; pass++) {
            double t0 = now(); h3_gpu_begin(gpu);
            for (int i = 0; i < iters; i++)
                if (!h3_gpu_linear_int8_bf16(gpu, output, in_i8, in_sc, input,
                                             w_i8, w_sc, ROWS, K, N, 0)) goto out;
            if (!h3_gpu_submit(gpu)) goto out;
            if (pass) rate[1] = flops * iters / (now() - t0) / 1e12;
        }
    out:
        printf("%6u %7u %10.1f %10.1f %8.2fx %9.0f  %s\n", K, N, rate[0], rate[1],
               rate[0] > 0 ? rate[1] / rate[0] : 0.0,
               (double)ROWS * N * 2 / 1e6, shapes[s].note);
    }
    h3_gpu_free(gpu);
    return 0;
}
