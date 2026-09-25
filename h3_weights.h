#ifndef H3_WEIGHTS_H
#define H3_WEIGHTS_H

#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_weight_store h3_weight_store;

/* Open every safetensors header in a component directory without reading
 * tensor payloads. */
h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size);
void h3_weight_store_free(h3_weight_store *store);
size_t h3_weight_store_shards(const h3_weight_store *store);

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header);

/* Validate an exact BF16 shape, allocate a shared Metal buffer, and read the
 * payload directly into that buffer with no intermediate host allocation. */
h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size);
h3_gpu_tensor *h3_weight_load_i8(const h3_weight_store *store, h3_gpu *gpu,
                                 const char *name, int ndim,
                                 const uint64_t *shape,
                                 char *error, size_t error_size);
/* Attach LoRA adapters (h3_lora.h spec). Every BF16/F32 weight loaded from
 * this store afterwards has its delta merged in; int8 weights go through
 * h3_weight_apply_lora_i8, since their scales load separately. The store owns
 * the adapters. An empty or NULL spec is a no-op. */
int h3_weight_store_attach_lora(h3_weight_store *store, const char *spec,
                                char *error, size_t error_size);
int h3_weight_store_has_lora(const h3_weight_store *store);
int h3_weight_lora_targets(const h3_weight_store *store, const char *name);
int h3_weight_apply_lora_i8(h3_weight_store *store, const char *name,
                            h3_gpu_tensor *weight, h3_gpu_tensor *scales,
                            uint64_t rows, uint64_t columns,
                            char *error, size_t error_size);
void h3_weight_lora_report(const h3_weight_store *store);

h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size);

#endif
