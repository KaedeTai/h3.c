#include "h3_lora.h"

#include "h3_safetensors.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>
#include <time.h>

typedef struct {
    char *path;
    char *label;
    float strength;
    h3_st_header header;
    size_t modules;
} lora_file;

typedef struct {
    char *target;              /* native weight name */
    size_t file;
    const h3_st_tensor *a;     /* [rank, columns] */
    const h3_st_tensor *b;     /* [rows, rank] */
    float scale;               /* strength * alpha / rank */
    unsigned applied;
} lora_entry;

struct h3_lora_set {
    lora_file *files;
    size_t file_count;
    lora_entry *entries;
    size_t count;
    double seconds;
    uint64_t merged_weights;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static double now_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static char *copy_string(const char *text, size_t length) {
    char *result = malloc(length + 1);
    if (!result) return NULL;
    memcpy(result, text, length);
    result[length] = '\0';
    return result;
}

static int is_file(const char *path) {
    struct stat info;
    return stat(path, &info) == 0 && S_ISREG(info.st_mode);
}

static int is_directory(const char *path) {
    struct stat info;
    return stat(path, &info) == 0 && S_ISDIR(info.st_mode);
}

static int ends_with(const char *text, const char *suffix) {
    size_t a = strlen(text), b = strlen(suffix);
    return a >= b && !strcmp(text + a - b, suffix);
}

/* ---- name resolution ---------------------------------------------------- */

static char *single_safetensors(const char *directory, char *error,
                                size_t error_size) {
    DIR *stream = opendir(directory);
    if (!stream) return NULL;
    char *found = NULL;
    char names[512] = "";
    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        if (!ends_with(entry->d_name, ".safetensors")) continue;
        count++;
        if (strlen(names) + strlen(entry->d_name) + 3 < sizeof(names)) {
            if (names[0]) strcat(names, ", ");
            strcat(names, entry->d_name);
        }
        if (!found) {
            size_t length = strlen(directory) + strlen(entry->d_name) + 2;
            found = malloc(length);
            if (found) snprintf(found, length, "%s/%s", directory,
                                entry->d_name);
        }
    }
    closedir(stream);
    if (count == 1) return found;
    free(found);
    if (count > 1)
        fail(error, error_size,
             "LoRA directory %s holds %d files (%s); name one as DIR/FILE",
             directory, count, names);
    else
        fail(error, error_size, "LoRA directory %s has no .safetensors",
             directory);
    return NULL;
}

static char *resolve_path(const char *item, char *error, size_t error_size) {
    if (is_file(item)) return copy_string(item, strlen(item));
    const char *root = getenv("H3_LORA_DIR");
    char fallback[1024];
    if (!root || !*root) {
        const char *home = getenv("HOME");
        snprintf(fallback, sizeof(fallback), "%s/models/loras/h3",
                 home ? home : ".");
        root = fallback;
    }
    char candidate[2048];
    snprintf(candidate, sizeof(candidate), "%s/%s", root, item);
    if (is_file(candidate)) return copy_string(candidate, strlen(candidate));
    if (is_directory(candidate))
        return single_safetensors(candidate, error, error_size);
    if (is_directory(item)) return single_safetensors(item, error, error_size);
    snprintf(candidate, sizeof(candidate), "%s/%s.safetensors", root, item);
    if (is_file(candidate)) return copy_string(candidate, strlen(candidate));
    fail(error, error_size, "LoRA not found: %s (looked in %s)", item, root);
    return NULL;
}

/* ---- key parsing -------------------------------------------------------- */

enum { PART_A, PART_B, PART_ALPHA, PART_UNSUPPORTED, PART_OTHER };

static const char *strip_prefix(const char *name) {
    static const char *prefixes[] = {
        "model.diffusion_model.", "diffusion_model.", "transformer.",
        "pipe.dit.", "base_model.model.", "dit.", NULL
    };
    for (int again = 1; again;) {
        again = 0;
        for (int index = 0; prefixes[index]; index++) {
            size_t length = strlen(prefixes[index]);
            if (!strncmp(name, prefixes[index], length)) {
                name += length;
                again = 1;
            }
        }
    }
    return name;
}

/* Splits `name` into its module (written to `module`) and part. */
static int parse_key(const char *name, char *module, size_t module_size) {
    static const struct { const char *suffix; int part; } suffixes[] = {
        {".lora_A.default.weight", PART_A}, {".lora_B.default.weight", PART_B},
        {".lora_A.weight", PART_A}, {".lora_B.weight", PART_B},
        {".lora_down.weight", PART_A}, {".lora_up.weight", PART_B},
        {".alpha", PART_ALPHA},
        {".dora_scale", PART_UNSUPPORTED}, {".diff", PART_UNSUPPORTED},
        {".diff_b", PART_UNSUPPORTED}, {".lora_mid.weight", PART_UNSUPPORTED},
        {NULL, 0}
    };
    const char *stem = strip_prefix(name);
    for (int index = 0; suffixes[index].suffix; index++) {
        if (!ends_with(stem, suffixes[index].suffix)) continue;
        size_t length = strlen(stem) - strlen(suffixes[index].suffix);
        if (length + 1 > module_size) return PART_OTHER;
        memcpy(module, stem, length);
        module[length] = '\0';
        return suffixes[index].part;
    }
    return PART_OTHER;
}

static int compare_entries(const void *left, const void *right) {
    const lora_entry *a = left, *b = right;
    int order = strcmp(a->target, b->target);
    if (order) return order;
    return a->file < b->file ? -1 : a->file > b->file;
}

static float half_to_float(uint16_t value) {
    uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x3ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (!mantissa) {
            bits = sign;
        } else {
            exponent = 127 - 15 + 1;
            while (!(mantissa & 0x400u)) { mantissa <<= 1; exponent--; }
            mantissa &= 0x3ffu;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static float bf16_to_float(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t float_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((bits >> 16) | 0x40u);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

/* Read a float tensor of any supported dtype into `values`, times `scale`. */
static int read_floats(const h3_st_header *header, const h3_st_tensor *tensor,
                       float *values, float scale, char *error,
                       size_t error_size) {
    size_t count = (size_t)h3_st_tensor_elements(tensor);
    size_t bytes = count * h3_dtype_size(tensor->dtype);
    if (tensor->dtype == H3_DTYPE_F32) {
        if (!h3_st_read_data(header, tensor, values, bytes, error, error_size))
            return 0;
        if (scale != 1.0f)
            for (size_t index = 0; index < count; index++)
                values[index] *= scale;
        return 1;
    }
    uint16_t *raw = malloc(bytes ? bytes : 1);
    if (!raw) {
        fail(error, error_size, "out of memory reading %s", tensor->name);
        return 0;
    }
    if (!h3_st_read_data(header, tensor, raw, bytes, error, error_size)) {
        free(raw);
        return 0;
    }
    if (tensor->dtype == H3_DTYPE_BF16)
        for (size_t index = 0; index < count; index++)
            values[index] = bf16_to_float(raw[index]) * scale;
    else
        for (size_t index = 0; index < count; index++)
            values[index] = half_to_float(raw[index]) * scale;
    free(raw);
    return 1;
}

static int float_dtype(h3_dtype dtype) {
    return dtype == H3_DTYPE_F32 || dtype == H3_DTYPE_BF16 ||
           dtype == H3_DTYPE_F16;
}

typedef struct {
    char module[256];
    const h3_st_tensor *a;
    const h3_st_tensor *b;
    const h3_st_tensor *alpha;
} module_parts;

static module_parts *find_parts(module_parts *parts, size_t *count,
                                size_t capacity, const char *module) {
    for (size_t index = 0; index < *count; index++)
        if (!strcmp(parts[index].module, module)) return &parts[index];
    if (*count == capacity) return NULL;
    module_parts *part = &parts[(*count)++];
    memset(part, 0, sizeof(*part));
    snprintf(part->module, sizeof(part->module), "%s", module);
    return part;
}

static int index_file(h3_lora_set *set, size_t file_index, char *error,
                      size_t error_size) {
    lora_file *file = &set->files[file_index];
    const h3_st_header *header = &file->header;
    size_t capacity = header->tensor_count;
    module_parts *parts = calloc(capacity ? capacity : 1, sizeof(*parts));
    if (!parts) {
        fail(error, error_size, "out of memory indexing %s", file->path);
        return 0;
    }
    size_t count = 0;
    for (size_t index = 0; index < header->tensor_count; index++) {
        const h3_st_tensor *tensor = &header->tensors[index];
        char module[256];
        int part = parse_key(tensor->name, module, sizeof(module));
        if (part == PART_UNSUPPORTED) {
            fail(error, error_size,
                 "%s: %s is a DoRA or full-weight patch, which h3.c does not "
                 "apply", file->label, tensor->name);
            free(parts);
            return 0;
        }
        if (part == PART_OTHER) {
            fail(error, error_size,
                 "%s: unrecognised LoRA key %s", file->label, tensor->name);
            free(parts);
            return 0;
        }
        module_parts *slot = find_parts(parts, &count, capacity, module);
        if (!slot) continue;
        if (part == PART_A) slot->a = tensor;
        else if (part == PART_B) slot->b = tensor;
        else slot->alpha = tensor;
    }
    size_t added = 0;
    for (size_t index = 0; index < count; index++) {
        module_parts *part = &parts[index];
        if (!part->a || !part->b) {
            fail(error, error_size, "%s: %s has only one of its A/B halves",
                 file->label, part->module);
            free(parts);
            return 0;
        }
        if (part->a->ndim != 2 || part->b->ndim != 2 ||
            part->a->shape[0] != part->b->shape[1] ||
            !float_dtype(part->a->dtype) || !float_dtype(part->b->dtype)) {
            fail(error, error_size,
                 "%s: %s has A/B shapes or dtypes that do not pair",
                 file->label, part->module);
            free(parts);
            return 0;
        }
        float rank = (float)part->a->shape[0];
        float alpha = rank;
        if (part->alpha) {
            if (h3_st_tensor_elements(part->alpha) != 1 ||
                !float_dtype(part->alpha->dtype)) {
                fail(error, error_size, "%s: %s.alpha is not a scalar",
                     file->label, part->module);
                free(parts);
                return 0;
            }
            if (!read_floats(header, part->alpha, &alpha, 1.0f,
                             error, error_size)) {
                free(parts);
                return 0;
            }
        }
        lora_entry *entry = &set->entries[set->count + added];
        size_t length = strlen(part->module) + sizeof(".weight");
        entry->target = malloc(length);
        if (!entry->target) {
            fail(error, error_size, "out of memory indexing %s", file->path);
            free(parts);
            return 0;
        }
        snprintf(entry->target, length, "%s.weight", part->module);
        entry->file = file_index;
        entry->a = part->a;
        entry->b = part->b;
        entry->scale = file->strength * alpha / rank;
        entry->applied = 0;
        added++;
    }
    set->count += added;
    file->modules = added;
    free(parts);
    return 1;
}

h3_lora_set *h3_lora_open(const char *spec, char *error, size_t error_size) {
    if (!spec || !*spec) return NULL;
    h3_lora_set *set = calloc(1, sizeof(*set));
    char *copy = copy_string(spec, strlen(spec));
    if (!set || !copy) {
        free(set);
        free(copy);
        fail(error, error_size, "out of memory parsing H3_LORA");
        return NULL;
    }
    size_t items = 1;
    for (const char *cursor = spec; *cursor; cursor++)
        if (*cursor == ',') items++;
    set->files = calloc(items, sizeof(*set->files));
    if (!set->files) goto oom;
    size_t tensors = 0;
    char *save = NULL;
    for (char *item = strtok_r(copy, ",", &save); item;
         item = strtok_r(NULL, ",", &save)) {
        while (*item == ' ') item++;
        size_t length = strlen(item);
        while (length && item[length - 1] == ' ') item[--length] = '\0';
        if (!length) continue;
        float strength = 1.0f;
        char *colon = strrchr(item, ':');
        if (colon && colon[1]) {
            char *end = NULL;
            errno = 0;
            double parsed = strtod(colon + 1, &end);
            if (!errno && end && !*end && isfinite(parsed)) {
                strength = (float)parsed;
                *colon = '\0';
            }
        }
        lora_file *file = &set->files[set->file_count];
        file->strength = strength;
        file->label = copy_string(item, strlen(item));
        file->path = resolve_path(item, error, error_size);
        if (!file->label || !file->path) {
            free(file->label);
            free(file->path);
            file->label = file->path = NULL;
            goto failed;
        }
        if (!h3_st_read_header(file->path, &file->header, error, error_size)) {
            free(file->label);
            free(file->path);
            file->label = file->path = NULL;
            goto failed;
        }
        set->file_count++;
        tensors += file->header.tensor_count;
    }
    if (!set->file_count) {
        fail(error, error_size, "H3_LORA names no adapter");
        goto failed;
    }
    set->entries = calloc(tensors ? tensors : 1, sizeof(*set->entries));
    if (!set->entries) goto oom;
    for (size_t index = 0; index < set->file_count; index++)
        if (!index_file(set, index, error, error_size)) goto failed;
    qsort(set->entries, set->count, sizeof(*set->entries), compare_entries);
    free(copy);
    return set;
oom:
    fail(error, error_size, "out of memory opening LoRA adapters");
failed:
    free(copy);
    h3_lora_free(set);
    return NULL;
}

void h3_lora_free(h3_lora_set *set) {
    if (!set) return;
    for (size_t index = 0; index < set->count; index++)
        free(set->entries[index].target);
    for (size_t index = 0; index < set->file_count; index++) {
        h3_st_free_header(&set->files[index].header);
        free(set->files[index].path);
        free(set->files[index].label);
    }
    free(set->entries);
    free(set->files);
    free(set);
}

/* First entry for `name`, and how many consecutive entries share it. */
static lora_entry *find_range(const h3_lora_set *set, const char *name,
                              size_t *found) {
    *found = 0;
    if (!set || !set->count || !name) return NULL;
    size_t low = 0, high = set->count;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        if (strcmp(set->entries[middle].target, name) < 0) low = middle + 1;
        else high = middle;
    }
    size_t end = low;
    while (end < set->count && !strcmp(set->entries[end].target, name)) end++;
    *found = end - low;
    return *found ? &set->entries[low] : NULL;
}

int h3_lora_targets(const h3_lora_set *set, const char *weight_name) {
    size_t found;
    return find_range(set, weight_name, &found) != NULL;
}

/* ---- merge -------------------------------------------------------------- */

enum { MERGE_BF16, MERGE_F32, MERGE_I8 };

typedef struct {
    int kind;
    void *weight;
    float *scales;
    uint64_t rows;
    uint64_t columns;
    uint32_t rank;             /* summed over stacked adapters */
    const float *a;            /* [rank, columns] */
    const float *b;            /* [rows, rank], adapter scale folded in */
    uint64_t chunk_rows;
    int stochastic;            /* int8 only; see merge_chunk */
    uint64_t seed;
    int failed;
} merge_job;

static uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static uint64_t name_seed(const char *name) {
    uint64_t hash = 0xcbf29ce484222325ull;
    for (const unsigned char *c = (const unsigned char *)name; *c; c++)
        hash = (hash ^ *c) * 0x100000001b3ull;
    return mix64(hash);
}

static void merge_chunk(void *opaque, size_t chunk) {
    merge_job *job = opaque;
    uint64_t first = (uint64_t)chunk * job->chunk_rows;
    if (first >= job->rows) return;
    uint64_t count = job->rows - first < job->chunk_rows ?
        job->rows - first : job->chunk_rows;
    size_t columns = (size_t)job->columns;
    float *delta = malloc((size_t)count * columns * sizeof(float));
    if (!delta) {
        job->failed = 1;
        return;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)count, (int)columns, (int)job->rank, 1.0f,
                job->b + first * job->rank, (int)job->rank,
                job->a, (int)columns, 0.0f, delta, (int)columns);
    for (uint64_t local = 0; local < count; local++) {
        uint64_t row = first + local;
        float *d = delta + local * columns;
        if (job->kind == MERGE_BF16) {
            uint16_t *w = (uint16_t *)job->weight + row * columns;
            for (size_t column = 0; column < columns; column++)
                w[column] = float_to_bf16(bf16_to_float(w[column]) +
                                          d[column]);
        } else if (job->kind == MERGE_F32) {
            float *w = (float *)job->weight + row * columns;
            for (size_t column = 0; column < columns; column++)
                w[column] += d[column];
        } else {
            /* Dequantize, add, requantize. */
            int8_t *q = (int8_t *)job->weight + row * columns;
            float scale = job->scales[row];
            float max_abs = 0.0f;
            for (size_t column = 0; column < columns; column++) {
                float magnitude = fabsf((float)q[column] * scale + d[column]);
                if (magnitude > max_abs) max_abs = magnitude;
            }
            /* Keep the row's grid unless the merged row no longer fits it.
             * A new scale moves every grid point, so every weight in the row
             * would be re-rounded; keeping it leaves the untouched part of
             * each weight exactly where it was. Up to half a step of
             * overshoot is left to the +-127 clamp, as ordinary rounding
             * would be. On the kept grid the arithmetic stays in code units
             * (q + delta / scale), so a zero delta returns q bit for bit. */
            int regrid = max_abs > 127.5f * scale || !(scale > 0.0f);
            float old_scale = scale;
            if (regrid) {
                scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
                job->scales[row] = scale;
            }
            float inverse = 1.0f / scale;
            /* A dequantized weight sits exactly on its grid point, so a delta
             * under half a step would round straight back to the old code:
             * nearest rounding silently drops most of a typical LoRA. The
             * unquantized weight was somewhere in that cell, and assuming it
             * uniform there, quantize(w + delta) moves up a code with
             * probability frac(x). Stochastic rounding with that probability
             * keeps the delta in expectation; the hash makes it repeatable. */
            uint64_t stream = job->seed ^ mix64(row + 1);
            for (size_t column = 0; column < columns; column++) {
                /* base + step, with the rounding done on `step` alone so that
                 * floating error in base + u can never move a code. */
                float base = regrid ? 0.0f : (float)q[column];
                float step = regrid ?
                    ((float)q[column] * old_scale + d[column]) * inverse :
                    d[column] * inverse;
                float value;
                if (job->stochastic) {
                    uint64_t bits = mix64(stream + column);
                    float u = (float)(bits >> 40) * (1.0f / 16777216.0f);
                    float whole = floorf(step);
                    value = base + whole + (u < step - whole ? 1.0f : 0.0f);
                } else {
                    value = base + rintf(step);
                }
                if (value > 127.0f) value = 127.0f;
                if (value < -127.0f) value = -127.0f;
                q[column] = (int8_t)value;
            }
        }
    }
    free(delta);
}

static int merge(h3_lora_set *set, const char *name, int kind, void *weight,
                 float *scales, uint64_t rows, uint64_t columns,
                 char *error, size_t error_size) {
    size_t found;
    lora_entry *entries = find_range(set, name, &found);
    if (!entries) return 1;
    double started = now_seconds();
    uint64_t rank = 0;
    for (size_t index = 0; index < found; index++) {
        lora_entry *entry = &entries[index];
        if (entry->b->shape[0] != rows || entry->a->shape[1] != columns) {
            fail(error, error_size,
                 "LoRA %s: %s is [%llu, %llu] but the adapter is for "
                 "[%llu, %llu]", set->files[entry->file].label, name,
                 (unsigned long long)rows, (unsigned long long)columns,
                 (unsigned long long)entry->b->shape[0],
                 (unsigned long long)entry->a->shape[1]);
            return 0;
        }
        rank += entry->a->shape[0];
    }
    float *a = malloc((size_t)(rank * columns) * sizeof(float));
    float *b = malloc((size_t)(rows * rank) * sizeof(float));
    float *part = NULL;
    int ok = a && b;
    uint64_t offset = 0;
    for (size_t index = 0; ok && index < found; index++) {
        lora_entry *entry = &entries[index];
        const h3_st_header *header = &set->files[entry->file].header;
        uint64_t r = entry->a->shape[0];
        ok = read_floats(header, entry->a, a + offset * columns, 1.0f,
                         error, error_size);
        if (!ok) break;
        /* B is [rows, r]; stacking adapters puts each one's columns side by
         * side in the combined [rows, rank]. */
        free(part);
        part = malloc((size_t)(rows * r) * sizeof(float));
        ok = part && read_floats(header, entry->b, part, entry->scale,
                                 error, error_size);
        if (!ok) break;
        for (uint64_t row = 0; row < rows; row++)
            memcpy(b + row * rank + offset, part + row * r,
                   (size_t)r * sizeof(float));
        offset += r;
    }
    free(part);
    if (!ok) {
        if (!error || !*error)
            fail(error, error_size, "out of memory merging LoRA into %s",
                 name);
        free(a);
        free(b);
        return 0;
    }
    merge_job job = {
        .kind = kind, .weight = weight, .scales = scales,
        .rows = rows, .columns = columns, .rank = (uint32_t)rank,
        .a = a, .b = b, .chunk_rows = 64, .failed = 0,
        .stochastic = !(getenv("H3_LORA_ROUND") &&
                        !strcmp(getenv("H3_LORA_ROUND"), "nearest")),
        .seed = name_seed(name)
    };
    size_t chunks = (size_t)((rows + job.chunk_rows - 1) / job.chunk_rows);
    dispatch_apply_f(chunks,
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &job, merge_chunk);
    free(a);
    free(b);
    if (job.failed) {
        fail(error, error_size, "out of memory merging LoRA into %s", name);
        return 0;
    }
    for (size_t index = 0; index < found; index++) entries[index].applied++;
    set->merged_weights++;
    set->seconds += now_seconds() - started;
    return 1;
}

int h3_lora_apply_bf16(h3_lora_set *set, const char *weight_name,
                       uint16_t *weight, uint64_t rows, uint64_t columns,
                       char *error, size_t error_size) {
    return merge(set, weight_name, MERGE_BF16, weight, NULL, rows, columns,
                 error, error_size);
}

int h3_lora_apply_f32(h3_lora_set *set, const char *weight_name,
                      float *weight, uint64_t rows, uint64_t columns,
                      char *error, size_t error_size) {
    return merge(set, weight_name, MERGE_F32, weight, NULL, rows, columns,
                 error, error_size);
}

int h3_lora_apply_i8(h3_lora_set *set, const char *weight_name,
                     int8_t *weight, float *scales, uint64_t rows,
                     uint64_t columns, char *error, size_t error_size) {
    return merge(set, weight_name, MERGE_I8, weight, scales, rows, columns,
                 error, error_size);
}

void h3_lora_report(const h3_lora_set *set, FILE *stream) {
    if (!set || !stream) return;
    for (size_t file = 0; file < set->file_count; file++) {
        size_t applied = 0, shown = 0;
        for (size_t index = 0; index < set->count; index++)
            if (set->entries[index].file == file &&
                set->entries[index].applied) applied++;
        fprintf(stream, "h3: LoRA %s x%g: %zu/%zu modules merged\n",
                set->files[file].label, (double)set->files[file].strength,
                applied, set->files[file].modules);
        for (size_t index = 0; index < set->count; index++) {
            const lora_entry *entry = &set->entries[index];
            if (entry->file != file || entry->applied) continue;
            if (shown++ < 4)
                fprintf(stream, "h3:   not loaded, so not merged: %s\n",
                        entry->target);
        }
        if (shown > 4)
            fprintf(stream, "h3:   ... and %zu more\n", shown - 4);
    }
    fprintf(stream, "h3: LoRA merge %.2fs over %llu weights\n", set->seconds,
            (unsigned long long)set->merged_weights);
}
