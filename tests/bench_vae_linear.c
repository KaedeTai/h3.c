/* The video VAE's attention projections do not use the NAX kernels. Why?
 *
 * h3_gpu_linear_bf16 takes the M5 TensorOps path only when `naxShape` matches
 * -- which under the default H3_NAX=qkv-attn means the two DiT shapes and
 * nothing else -- AND when the call carries no bias. The VAE's QKV and
 * attention-output projections fail both tests: their shapes are 2048->6144
 * and 2048->2048, and both pass a bias. So they land on MPSGraph, which is
 * what `linear=1022` in the decoder's profile line is counting.
 *
 * This times the same two shapes both ways at the sequence length a real
 * 320x576 tile produces, to find out what that fallback costs.
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

#define ROWS 2805u          /* 7 latent frames x 20 x 20 patches + 5 suffix */
#define ITERS 20

/* Every call goes into ONE command buffer. Submitting each one on its own adds
 * the command-buffer round trip -- about 0.25 ms -- to a 1.6 ms matmul, which
 * is enough to make the two paths look closer than they are. */
static double run(h3_gpu *gpu, h3_gpu_tensor *out, h3_gpu_tensor *in,
                  h3_gpu_tensor *w, h3_gpu_tensor *bias,
                  uint32_t k, uint32_t n) {
    if (!h3_gpu_begin(gpu)) return -1.0;
    for (int warm = 0; warm < 3; warm++)
        if (!h3_gpu_linear_bf16(gpu, out, in, w, bias, ROWS, k, n)) return -1.0;
    if (!h3_gpu_submit(gpu)) return -1.0;
    double start = now();
    if (!h3_gpu_begin(gpu)) return -1.0;
    for (int i = 0; i < ITERS; i++)
        if (!h3_gpu_linear_bf16(gpu, out, in, w, bias, ROWS, k, n)) return -1.0;
    if (!h3_gpu_submit(gpu)) return -1.0;
    double seconds = (now() - start) / ITERS;
    return 2.0 * ROWS * k * n / seconds / 1e12;
}

int main(void) {
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "gpu: %s\n", error); return 1; }

    const uint32_t shapes[][2] = {{2048, 6144}, {2048, 2048}};
    printf("rows=%u  (H3_NAX=%s)\n", ROWS,
           getenv("H3_NAX") ? getenv("H3_NAX") : "qkv-attn (default)");
    printf("%14s %12s %12s %8s\n", "shape", "with bias", "no bias", "比值");
    for (int s = 0; s < 2; s++) {
        uint32_t k = shapes[s][0], n = shapes[s][1];
        size_t in_n = (size_t)ROWS * k, w_n = (size_t)n * k, out_n = (size_t)ROWS * n;
        /* One fill buffer for both uploads, sized to the LARGER of them: the
         * activation is 2805x2048 and the 2048->2048 weight is smaller, so
         * sizing it to the weight alone reads off the end. */
        size_t fill_n = in_n > w_n ? in_n : w_n;
        uint16_t *host = malloc(fill_n * sizeof(*host));
        if (!host) return 2;
        for (size_t i = 0; i < fill_n; i++) host[i] = 0x3F80 - (uint16_t)(i & 0x3F);
        h3_gpu_tensor *in = h3_gpu_tensor_from_bf16(gpu, host, in_n);
        h3_gpu_tensor *w = h3_gpu_tensor_from_bf16(gpu, host, w_n);
        h3_gpu_tensor *bias = h3_gpu_tensor_from_bf16(gpu, host, n);
        h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, out_n);
        free(host);
        if (!in || !w || !bias || !out) {
            fprintf(stderr, "alloc: %s\n", h3_gpu_error(gpu)); return 3;
        }
        double biased = run(gpu, out, in, w, bias, k, n);
        double bare = run(gpu, out, in, w, NULL, k, n);
        printf("%9u->%-5u %10.1f%s %10.1f%s %8.2fx\n", k, n,
               biased, biased < 0 ? " ERR" : "", bare, bare < 0 ? " ERR" : "",
               biased > 0 ? bare / biased : 0.0);
        h3_gpu_tensor_free(in); h3_gpu_tensor_free(w);
        h3_gpu_tensor_free(bias); h3_gpu_tensor_free(out);
    }
    h3_gpu_free(gpu);
    return 0;
}
