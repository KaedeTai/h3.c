#include "h3_weights.h"

#include "h3_lora.h"

#include <dirent.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct h3_weight_store {
    h3_st_header *headers;
    size_t count;
    h3_lora_set *lora;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int safetensors_name(const char *name) {
    static const char suffix[] = ".safetensors";
    size_t length = strlen(name);
    /* Hidden files are never shards: macOS writes ._name AppleDouble companions next to every file on an
       exFAT/FAT volume, and they share the .safetensors suffix. */
    if (name[0] == '.') return 0;
    return length > sizeof(suffix) - 1 &&
           strcmp(name + length - (sizeof(suffix) - 1), suffix) == 0;
}

static int compare_paths(const void *left, const void *right) {
    const char *const *a = left;
    const char *const *b = right;
    return strcmp(*a, *b);
}

static void free_paths(char **paths, size_t count) {
    if (!paths) return;
    for (size_t index = 0; index < count; index++) free(paths[index]);
    free(paths);
}

h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size) {
    if (!directory || !*directory) {
        fail(error, error_size, "weight directory is required");
        return NULL;
    }
    DIR *stream = opendir(directory);
    if (!stream) {
        fail(error, error_size, "cannot open weight directory: %s", directory);
        return NULL;
    }
    char **paths = NULL;
    size_t count = 0;
    size_t capacity = 0;
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        if (!safetensors_name(entry->d_name)) continue;
        if (count == capacity) {
            size_t next = capacity ? capacity * 2 : 8;
            char **grown = realloc(paths, next * sizeof(*grown));
            if (!grown) {
                fail(error, error_size, "out of memory listing weight shards");
                closedir(stream);
                free_paths(paths, count);
                return NULL;
            }
            paths = grown;
            capacity = next;
        }
        size_t length = strlen(directory) + strlen(entry->d_name) + 2;
        paths[count] = malloc(length);
        if (!paths[count]) {
            fail(error, error_size, "out of memory resolving a weight shard");
            closedir(stream);
            free_paths(paths, count);
            return NULL;
        }
        snprintf(paths[count], length, "%s/%s", directory, entry->d_name);
        count++;
    }
    closedir(stream);
    if (!count) {
        fail(error, error_size, "no safetensors shards in %s", directory);
        free(paths);
        return NULL;
    }
    qsort(paths, count, sizeof(*paths), compare_paths);
    h3_weight_store *store = calloc(1, sizeof(*store));
    if (!store) {
        fail(error, error_size, "out of memory creating weight store");
        free_paths(paths, count);
        return NULL;
    }
    store->headers = calloc(count, sizeof(*store->headers));
    if (!store->headers) {
        fail(error, error_size, "out of memory allocating weight headers");
        free(store);
        free_paths(paths, count);
        return NULL;
    }
    store->count = count;
    for (size_t index = 0; index < count; index++) {
        char detail[384];
        if (!h3_st_read_header(paths[index], &store->headers[index], detail,
                               sizeof(detail))) {
            fail(error, error_size, "%s", detail);
            free_paths(paths, count);
            h3_weight_store_free(store);
            return NULL;
        }
    }
    free_paths(paths, count);
    return store;
}

void h3_weight_store_free(h3_weight_store *store) {
    if (!store) return;
    for (size_t index = 0; index < store->count; index++) {
        h3_st_free_header(&store->headers[index]);
    }
    free(store->headers);
    h3_lora_free(store->lora);
    free(store);
}

size_t h3_weight_store_shards(const h3_weight_store *store) {
    return store ? store->count : 0;
}

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header) {
    if (header) *header = NULL;
    if (!store || !name) return NULL;
    for (size_t index = 0; index < store->count; index++) {
        const h3_st_tensor *tensor = h3_st_find(&store->headers[index], name);
        if (tensor) {
            if (header) *header = &store->headers[index];
            return tensor;
        }
    }
    return NULL;
}

static h3_gpu_tensor *load_tensor(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape, h3_dtype dtype,
                                  char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return NULL;
    }
    if (tensor->dtype != dtype || tensor->ndim != ndim) {
        fail(error, error_size, "weight %s has dtype/rank %s/%d, expected %s/%d",
             name, h3_dtype_name(tensor->dtype), tensor->ndim,
             h3_dtype_name(dtype), ndim);
        return NULL;
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            fail(error, error_size, "weight %s shape mismatch at dimension %d",
                 name, dimension);
            return NULL;
        }
        if (shape[dimension] && elements > UINT64_MAX / shape[dimension]) {
            fail(error, error_size, "weight %s shape overflows", name);
            return NULL;
        }
        elements *= shape[dimension];
    }
    if (elements > SIZE_MAX) {
        fail(error, error_size, "weight %s is too large for this process", name);
        return NULL;
    }
    /* A weight the LoRA will rewrite is read into its own storage rather than
     * mapped from the file (see h3_gpu_tensor_load_copy). An int8 weight's
     * scale is rewritten with it. */
    int rewritten = 0;
    if (store->lora) {
        char base[256];
        size_t length = strlen(name);
        rewritten = h3_lora_targets(store->lora, name);
        if (!rewritten && length > 6 && length - 6 < sizeof(base) &&
            !strcmp(name + length - 6, "_scale")) {
            memcpy(base, name, length - 6);
            base[length - 6] = '\0';
            rewritten = h3_lora_targets(store->lora, base);
        }
    }
    h3_gpu_tensor *result = rewritten ?
        h3_gpu_tensor_load_copy(gpu, header->path, tensor->file_offset,
                                (size_t)elements,
                                dtype == H3_DTYPE_BF16 ? H3_GPU_BF16 :
                                dtype == H3_DTYPE_I8 ? H3_GPU_I8 :
                                                       H3_GPU_F32) :
        dtype == H3_DTYPE_BF16 ?
            h3_gpu_tensor_load_bf16(gpu, header->path, tensor->file_offset,
                                    (size_t)elements) :
        dtype == H3_DTYPE_I8 ?
            h3_gpu_tensor_load_i8(gpu, header->path, tensor->file_offset,
                                  (size_t)elements) :
            h3_gpu_tensor_load_f32(gpu, header->path, tensor->file_offset,
                                   (size_t)elements);
    if (!result) {
        fail(error, error_size, "cannot load %s: %s", name, h3_gpu_error(gpu));
        return NULL;
    }
    if (store->lora && ndim == 2 && dtype != H3_DTYPE_I8 &&
        h3_lora_targets(store->lora, name)) {
        void *host = h3_gpu_tensor_host_pointer(result);
        int merged = host && (dtype == H3_DTYPE_BF16 ?
            h3_lora_apply_bf16(store->lora, name, host, shape[0], shape[1],
                               error, error_size) :
            h3_lora_apply_f32(store->lora, name, host, shape[0], shape[1],
                              error, error_size));
        if (!merged) {
            if (!host)
                fail(error, error_size, "%s: no host view to merge LoRA into",
                     name);
            h3_gpu_tensor_free(result);
            return NULL;
        }
    }
    return result;
}

int h3_weight_store_attach_lora(h3_weight_store *store, const char *spec,
                                char *error, size_t error_size) {
    if (!store) return 0;
    if (!spec || !*spec) return 1;
    h3_lora_set *lora = h3_lora_open(spec, error, error_size);
    if (!lora) return 0;
    h3_lora_free(store->lora);
    store->lora = lora;
    return 1;
}

int h3_weight_store_has_lora(const h3_weight_store *store) {
    return store && store->lora;
}

int h3_weight_lora_targets(const h3_weight_store *store, const char *name) {
    return store && store->lora && h3_lora_targets(store->lora, name);
}

int h3_weight_apply_lora_i8(h3_weight_store *store, const char *name,
                            h3_gpu_tensor *weight, h3_gpu_tensor *scales,
                            uint64_t rows, uint64_t columns,
                            char *error, size_t error_size) {
    if (!store || !store->lora || !h3_lora_targets(store->lora, name))
        return 1;
    void *q = h3_gpu_tensor_host_pointer(weight);
    void *s = h3_gpu_tensor_host_pointer(scales);
    if (!q || !s) {
        fail(error, error_size, "%s: no host view to merge LoRA into", name);
        return 0;
    }
    return h3_lora_apply_i8(store->lora, name, q, s, rows, columns,
                            error, error_size);
}

void h3_weight_lora_report(const h3_weight_store *store) {
    if (store && store->lora) h3_lora_report(store->lora, stderr);
}

h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_BF16,
                       error, error_size);
}

h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_F32,
                       error, error_size);
}

h3_gpu_tensor *h3_weight_load_i8(const h3_weight_store *store, h3_gpu *gpu,
                                 const char *name, int ndim,
                                 const uint64_t *shape,
                                 char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_I8,
                       error, error_size);
}
