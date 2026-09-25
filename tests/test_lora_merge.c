/* Merge one LoRA into one weight on the CPU and dump the result, so
 * tools/check_lora_merge.py can compare it with a NumPy reference.
 *   test_lora_merge SHARD_DIR WEIGHT_NAME LORA_SPEC OUT_PREFIX */
#include "h3_lora.h"
#include "h3_safetensors.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage\n"); return 2; }
    char error[512] = "";
    DIR *dir = opendir(argv[1]);
    if (!dir) return 2;
    struct dirent *entry;
    h3_st_header header, found_header;
    const h3_st_tensor *weight = NULL, *scale = NULL;
    char scale_name[256];
    snprintf(scale_name, sizeof(scale_name), "%s_scale", argv[2]);
    int have = 0;
    while ((entry = readdir(dir)) && !have) {
        if (!strstr(entry->d_name, ".safetensors")) continue;
        char path[2048];
        snprintf(path, sizeof(path), "%s/%s", argv[1], entry->d_name);
        if (!h3_st_read_header(path, &header, error, sizeof(error))) return 1;
        weight = h3_st_find(&header, argv[2]);
        scale = h3_st_find(&header, scale_name);
        if (weight) { found_header = header; have = 1; }
        else h3_st_free_header(&header);
    }
    closedir(dir);
    if (!weight) { fprintf(stderr, "no %s\n", argv[2]); return 1; }
    uint64_t rows = weight->shape[0], columns = weight->shape[1];
    size_t count = (size_t)(rows * columns);
    h3_lora_set *set = h3_lora_open(argv[3], error, sizeof(error));
    if (!set) { fprintf(stderr, "%s\n", error); return 1; }
    char path[2048];
    int ok;
    if (weight->dtype == H3_DTYPE_I8) {
        int8_t *q = malloc(count);
        float *s = malloc(rows * sizeof(float));
        ok = h3_st_read_data(&found_header, weight, q, count, error, sizeof(error)) &&
             scale && h3_st_read_data(&found_header, scale, s, rows * sizeof(float),
                                      error, sizeof(error)) &&
             h3_lora_apply_i8(set, argv[2], q, s, rows, columns, error, sizeof(error));
        if (!ok) { fprintf(stderr, "%s\n", error); return 1; }
        snprintf(path, sizeof(path), "%s.q", argv[4]);
        FILE *f = fopen(path, "wb"); fwrite(q, 1, count, f); fclose(f);
        snprintf(path, sizeof(path), "%s.s", argv[4]);
        f = fopen(path, "wb"); fwrite(s, sizeof(float), rows, f); fclose(f);
    } else {
        uint16_t *w = malloc(count * 2);
        ok = h3_st_read_data(&found_header, weight, w, count * 2, error, sizeof(error)) &&
             h3_lora_apply_bf16(set, argv[2], w, rows, columns, error, sizeof(error));
        if (!ok) { fprintf(stderr, "%s\n", error); return 1; }
        snprintf(path, sizeof(path), "%s.bf16", argv[4]);
        FILE *f = fopen(path, "wb"); fwrite(w, 2, count, f); fclose(f);
    }
    h3_lora_report(set, stderr);
    h3_lora_free(set);
    return 0;
}
