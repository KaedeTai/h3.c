/* Time one DiT-shaped bf16 SDPA through h3's GPU layer.
   ./sdpa_bench [N] [reps]   (set H3_STEEL_ATTN=1 to route it to the steel NAX kernel) */
#include "../../h3_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint16_t bf16(float v) { uint32_t b; memcpy(&b, &v, 4); b += 0x7fffu + ((b >> 16) & 1u); return (uint16_t)(b >> 16); }

int main(int argc, char **argv) {
    uint32_t N = argc > 1 ? (uint32_t)atoi(argv[1]) : 37730, H = 56, D = 128;
    int reps = argc > 2 ? atoi(argv[2]) : 3;
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "gpu: %s\n", error); return 1; }
    size_t count = (size_t)N * H * D;
    uint16_t *host = malloc(count * sizeof(*host));
    srand(1);
    h3_gpu_tensor *t[3];
    for (int i = 0; i < 3; i++) {
        for (size_t j = 0; j < count; j++) host[j] = bf16(((float)rand() / RAND_MAX - 0.5f) * 3.0f);
        t[i] = h3_gpu_tensor_from_bf16(gpu, host, count);
    }
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, count);
    double flops = 4.0 * N * (double)N * H * D;
    const char *names[] = {"row-major in, row-major out", "head-major in, row-major out", "head-major in, head-major out"};
    for (int mode = 0; mode < 3; mode++) {
        for (int r = 0; r < reps; r++) {
            double t0 = now();
            h3_gpu_begin(gpu);
            if (mode) h3_gpu_set_head_major_sdpa_inputs(gpu, 1);
            int ok = mode == 2 ? h3_gpu_sdpa_bf16_head_major_output(gpu, out, t[0], t[1], t[2], N, H, D, 1.0f / sqrtf((float)D))
                               : h3_gpu_sdpa_bf16(gpu, out, t[0], t[1], t[2], N, H, D, 1.0f / sqrtf((float)D));
            ok = ok && h3_gpu_submit(gpu);
            double dt = now() - t0;
            if (!ok) { fprintf(stderr, "sdpa failed: %s\n", h3_gpu_error(gpu)); return 1; }
            printf("%-32s N=%u  %8.1f ms  %5.1f TFLOPS\n", names[mode], N, dt * 1e3, flops / dt / 1e12);
            fflush(stdout);
        }
    }
    return 0;
}
