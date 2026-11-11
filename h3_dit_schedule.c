#include "h3_dit_schedule.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TIME_INPUT = 256,
    TIME_HIDDEN = 5376,
    BLOCK_OUTPUT = H3_DIT_MODALITIES * H3_DIT_ADALN_SLOTS * H3_DIT_HIDDEN,
    FINAL_OUTPUT = 2 * H3_DIT_HIDDEN
};

/* HyperFlow H3 Flash (VideoRebirthAI/hyperflow-h3-flash) conditions every AdaLN on two times: the row's own t
 * and the endpoint r its Euler step lands on, temb(t, r) = e_t(t) + gate * (e_r(r) - e_t(t)). Its checkpoint
 * carries endpoint_time_embedder.* beside time_embedder.*, a 16-wide time embedding instead of 2688, and no
 * weights at all for the blocks it removed. Each of those is read off the weights, so the released H3 bundles
 * keep the one-time path unchanged. */
#define HYPERFLOW_DEFAULT_GATE 0.25f

struct h3_dit_schedule {
    h3_gpu *gpu;
    int steps;
    uint32_t time_rows;
    uint32_t time_dim;
    int two_time;
    uint32_t *video_rows;
    uint32_t *audio_rows;
    uint32_t *visual_condition_rows;
    uint32_t *audio_condition_rows;
    h3_gpu_tensor *blocks[H3_DIT_BLOCKS];
    h3_gpu_tensor *final;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(h3_gpu *gpu, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(gpu));
    return 0;
}

static h3_gpu_tensor *weight_f32_1d(const h3_weight_store *store, h3_gpu *gpu,
                                    const char *name, uint64_t width,
                                    char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_f32(store, gpu, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *weight_f32_2d(const h3_weight_store *store, h3_gpu *gpu,
                                    const char *name, uint64_t rows,
                                    uint64_t columns, char *error,
                                    size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_f32(store, gpu, name, 2, shape, error, error_size);
}

static h3_gpu_tensor *weight_bf16_1d(const h3_weight_store *store, h3_gpu *gpu,
                                     const char *name, uint64_t width,
                                     char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_bf16(store, gpu, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *weight_bf16_2d(const h3_weight_store *store, h3_gpu *gpu,
                                     const char *name, uint64_t rows,
                                     uint64_t columns, char *error,
                                     size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_bf16(store, gpu, name, 2, shape, error, error_size);
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

/* One row per distinct time (one-time H3) or per distinct (t, r) pair (HyperFlow). `ends` is the endpoint r of
 * each row; for one-time H3 it equals `times` and is unused. */
static int prepare_rows(h3_dit_schedule *schedule,
                        const h3_sigma_schedule *sigmas,
                        int visual_condition, int audio_condition,
                        float **times_out, float **ends_out, char *error,
                        size_t error_size) {
    schedule->steps = sigmas->steps;
    schedule->video_rows = calloc((size_t)sigmas->steps,
                                  sizeof(*schedule->video_rows));
    schedule->audio_rows = calloc((size_t)sigmas->steps,
                                  sizeof(*schedule->audio_rows));
    if (visual_condition)
        schedule->visual_condition_rows = calloc(
            (size_t)sigmas->steps, sizeof(*schedule->visual_condition_rows));
    if (audio_condition)
        schedule->audio_condition_rows = calloc(
            (size_t)sigmas->steps, sizeof(*schedule->audio_condition_rows));
    size_t capacity = 2u * (size_t)sigmas->steps + 2u;
    float *times = calloc(capacity, sizeof(*times));
    float *ends = calloc(capacity, sizeof(*ends));
    if (!schedule->video_rows || !schedule->audio_rows ||
        (visual_condition && !schedule->visual_condition_rows) ||
        (audio_condition && !schedule->audio_condition_rows) ||
        !times || !ends) {
        free(times);
        free(ends);
        fail(error, error_size, "out of memory allocating timestep row maps");
        return 0;
    }
    uint32_t count = 0;
    if (schedule->two_time) {
        /* Generated video (and text) rows step (t, r) on the video clock, generated audio rows on the audio
         * clock; they get separate rows whenever the pairs differ, even when t alone coincides (step 0). Visual
         * conditions are held at t = r = 0.999 and reference audio at t = r = 1, the official pin rule. */
        for (int step = 0; step < sigmas->steps; step++) {
            float video = 1.0f - sigmas->video[step];
            float video_end = 1.0f - sigmas->video[step + 1];
            float audio = 1.0f - sigmas->audio[step];
            float audio_end = 1.0f - sigmas->audio[step + 1];
            schedule->video_rows[step] = count;
            times[count] = video;
            ends[count++] = video_end;
            if (audio == video && audio_end == video_end) {
                schedule->audio_rows[step] = schedule->video_rows[step];
            } else {
                schedule->audio_rows[step] = count;
                times[count] = audio;
                ends[count++] = audio_end;
            }
        }
        if (visual_condition) {
            for (int step = 0; step < sigmas->steps; step++)
                schedule->visual_condition_rows[step] = count;
            times[count] = 0.999f;
            ends[count++] = 0.999f;
        }
        if (audio_condition) {
            for (int step = 0; step < sigmas->steps; step++)
                schedule->audio_condition_rows[step] = count;
            times[count] = 1.0f;
            ends[count++] = 1.0f;
        }
    } else {
        for (int step = 0; step < sigmas->steps; step++) {
            float video = 1.0f - sigmas->video[step];
            float audio = 1.0f - sigmas->audio[step];
            if (video == audio) {
                schedule->video_rows[step] = count;
                schedule->audio_rows[step] = count++;
            } else if (video < audio) {
                schedule->video_rows[step] = count++;
                schedule->audio_rows[step] = count++;
            } else {
                schedule->audio_rows[step] = count++;
                schedule->video_rows[step] = count++;
            }
        }
        uint32_t visual_condition_row = UINT32_MAX;
        uint32_t audio_condition_row = UINT32_MAX;
        if (visual_condition) visual_condition_row = count++;
        if (audio_condition) audio_condition_row = count++;
        for (int step = 0; step < sigmas->steps; step++) {
            float video = 1.0f - sigmas->video[step];
            float audio = 1.0f - sigmas->audio[step];
            if (visual_condition)
                schedule->visual_condition_rows[step] = video >= 0.999f ?
                    schedule->video_rows[step] : visual_condition_row;
            if (audio_condition)
                schedule->audio_condition_rows[step] = audio >= 1.0f ?
                    schedule->audio_rows[step] : audio_condition_row;
        }
        for (int step = 0; step < sigmas->steps; step++) {
            times[schedule->video_rows[step]] = 1.0f - sigmas->video[step];
            times[schedule->audio_rows[step]] = 1.0f - sigmas->audio[step];
        }
        if (visual_condition) times[visual_condition_row] = 0.999f;
        if (audio_condition) times[audio_condition_row] = 1.0f;
        memcpy(ends, times, capacity * sizeof(*times));
    }
    schedule->time_rows = count;
    if (!count || count > capacity) {
        free(times);
        free(ends);
        fail(error, error_size, "invalid number of timestep rows");
        return 0;
    }
    *times_out = times;
    *ends_out = ends;
    return 1;
}

/* Sinusoidal timestep features, cos then sin (diffusers Timesteps with flip_sin_to_cos, no shift). */
static float *time_features(const float *times, uint32_t rows) {
    float *features = malloc((size_t)rows * TIME_INPUT * sizeof(*features));
    if (!features) return NULL;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t index = 0; index < TIME_INPUT / 2; index++) {
            float frequency = expf(-logf(10000.0f) *
                                   (float)index / (float)(TIME_INPUT / 2));
            float angle = times[row] * frequency;
            features[(size_t)row * TIME_INPUT + index] = cosf(angle);
            features[(size_t)row * TIME_INPUT + TIME_INPUT / 2 + index] =
                sinf(angle);
        }
    }
    return features;
}

static h3_gpu_tensor *time_embeddings(const h3_weight_store *weights,
                                      h3_gpu *gpu, uint32_t rows,
                                      uint32_t time_dim,
                                      const float *features, char *error,
                                      size_t error_size) {
    h3_gpu_tensor *input = h3_gpu_tensor_from_f32(
        gpu, features, (size_t)rows * TIME_INPUT);
    h3_gpu_tensor *in_w = weight_f32_2d(weights, gpu,
        "time_embedder.proj_in.weight", TIME_HIDDEN, TIME_INPUT,
        error, error_size);
    h3_gpu_tensor *in_b = weight_f32_1d(weights, gpu,
        "time_embedder.proj_in.bias", TIME_HIDDEN, error, error_size);
    h3_gpu_tensor *out_w = weight_f32_2d(weights, gpu,
        "time_embedder.proj_out.weight", time_dim, TIME_HIDDEN,
        error, error_size);
    h3_gpu_tensor *out_b = weight_f32_1d(weights, gpu,
        "time_embedder.proj_out.bias", time_dim, error, error_size);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * TIME_HIDDEN);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * TIME_HIDDEN);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * time_dim);
    h3_gpu_tensor *bf16 = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * time_dim);
    h3_gpu_tensor *silu = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * time_dim);
    h3_gpu_tensor *result = NULL;
    h3_gpu_tensor *all[] = {input, in_w, in_b, out_w, out_b, hidden,
                            activated, output, bf16, silu};
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++) {
        if (!all[index]) {
            if (!error || !*error)
                fail(error, error_size, "cannot allocate timestep tensors: %s",
                     h3_gpu_error(gpu));
            goto cleanup;
        }
    }
    if (!gpu_op(gpu, h3_gpu_begin(gpu), error, error_size,
                "begin timestep embedding") ||
        !gpu_op(gpu, h3_gpu_linear_f32(gpu, hidden, input, in_w, in_b, rows,
                                       TIME_INPUT, TIME_HIDDEN),
                error, error_size, "timestep input projection") ||
        !gpu_op(gpu, h3_gpu_silu_f32(gpu, activated, hidden,
                                     rows * TIME_HIDDEN),
                error, error_size, "timestep SiLU") ||
        !gpu_op(gpu, h3_gpu_linear_f32(gpu, output, activated, out_w, out_b,
                                       rows, TIME_HIDDEN, time_dim),
                error, error_size, "timestep output projection") ||
        !gpu_op(gpu, h3_gpu_cast_f32_to_bf16(
                        gpu, bf16, output, rows * time_dim),
                error, error_size, "timestep BF16 cast") ||
        !gpu_op(gpu, h3_gpu_silu_bf16(gpu, silu, bf16,
                                      rows * time_dim),
                error, error_size, "timestep AdaLN SiLU") ||
        !gpu_op(gpu, h3_gpu_submit(gpu), error, error_size,
                "submit timestep embedding")) {
        goto cleanup;
    }
    result = silu;
    silu = NULL;
cleanup:
    free_tensor(&input);
    free_tensor(&in_w);
    free_tensor(&in_b);
    free_tensor(&out_w);
    free_tensor(&out_b);
    free_tensor(&hidden);
    free_tensor(&activated);
    free_tensor(&output);
    free_tensor(&bf16);
    free_tensor(&silu);
    return result;
}

/* An F32 weight copied to the host (through a shared buffer). */
static float *host_f32(const h3_weight_store *weights, h3_gpu *gpu,
                       const char *name, int ndim, const uint64_t *shape,
                       char *error, size_t error_size) {
    size_t count = 1;
    for (int axis = 0; axis < ndim; axis++) count *= (size_t)shape[axis];
    h3_gpu_tensor *tensor = h3_weight_load_f32(weights, gpu, name, ndim,
                                               shape, error, error_size);
    if (!tensor) return NULL;
    float *values = malloc(count * sizeof(*values));
    if (!values || !h3_gpu_tensor_read_f32(tensor, values, count)) {
        free(values);
        values = NULL;
        fail(error, error_size, "cannot read %s to the host", name);
    }
    h3_gpu_tensor_free(tensor);
    return values;
}

typedef struct {
    float *in_w, *in_b, *out_w, *out_b;
} host_embedder;

static void free_host_embedder(host_embedder *embedder) {
    free(embedder->in_w);
    free(embedder->in_b);
    free(embedder->out_w);
    free(embedder->out_b);
    memset(embedder, 0, sizeof(*embedder));
}

static int load_host_embedder(const h3_weight_store *weights, h3_gpu *gpu,
                              const char *prefix, uint32_t time_dim,
                              host_embedder *embedder, char *error,
                              size_t error_size) {
    char name[96];
    uint64_t in_shape[] = {TIME_HIDDEN, TIME_INPUT}, in_bias[] = {TIME_HIDDEN};
    uint64_t out_shape[] = {time_dim, TIME_HIDDEN}, out_bias[] = {time_dim};
    snprintf(name, sizeof(name), "%s.proj_in.weight", prefix);
    embedder->in_w = host_f32(weights, gpu, name, 2, in_shape, error, error_size);
    snprintf(name, sizeof(name), "%s.proj_in.bias", prefix);
    if (embedder->in_w)
        embedder->in_b = host_f32(weights, gpu, name, 1, in_bias, error, error_size);
    snprintf(name, sizeof(name), "%s.proj_out.weight", prefix);
    if (embedder->in_b)
        embedder->out_w = host_f32(weights, gpu, name, 2, out_shape, error, error_size);
    snprintf(name, sizeof(name), "%s.proj_out.bias", prefix);
    if (embedder->out_w)
        embedder->out_b = host_f32(weights, gpu, name, 1, out_bias, error, error_size);
    if (embedder->out_b) return 1;
    free_host_embedder(embedder);
    return 0;
}

/* proj_out(silu(proj_in(feature))) in double; the GPU's F32 GEMMs are not needed for a few rows, and this
 * embedding cancels heavily (an output of norm ~740 from a hidden of norm ~0.65), so it is kept exact. */
static void embed_host(const host_embedder *embedder, uint32_t time_dim,
                       const float *feature, double *hidden, double *output) {
    for (uint32_t row = 0; row < TIME_HIDDEN; row++) {
        const float *weight = embedder->in_w + (size_t)row * TIME_INPUT;
        double sum = embedder->in_b[row];
        for (uint32_t column = 0; column < TIME_INPUT; column++)
            sum += (double)weight[column] * (double)feature[column];
        hidden[row] = sum / (1.0 + exp(-sum));
    }
    for (uint32_t row = 0; row < time_dim; row++) {
        const float *weight = embedder->out_w + (size_t)row * TIME_HIDDEN;
        double sum = embedder->out_b[row];
        for (uint32_t column = 0; column < TIME_HIDDEN; column++)
            sum += (double)weight[column] * hidden[column];
        output[row] = sum;
    }
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu))
        return (uint16_t)((bits >> 16) | 0x40u);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

/* silu(temb(t, r)) for every row, as BF16, the input every AdaLN projection reads. The SiLU runs at F32 and only
 * its result is rounded to BF16, as the reference does. */
static h3_gpu_tensor *two_time_embeddings(const h3_weight_store *weights,
                                          h3_gpu *gpu, uint32_t rows,
                                          uint32_t time_dim,
                                          const float *times,
                                          const float *ends, char *error,
                                          size_t error_size) {
    host_embedder current = {0}, endpoint = {0};
    float *time_input = time_features(times, rows);
    float *end_input = time_features(ends, rows);
    double *hidden = malloc(TIME_HIDDEN * sizeof(*hidden));
    double *at_t = malloc(time_dim * sizeof(*at_t));
    double *at_r = malloc(time_dim * sizeof(*at_r));
    uint16_t *values = malloc((size_t)rows * time_dim * sizeof(*values));
    h3_gpu_tensor *result = NULL;
    float gate = HYPERFLOW_DEFAULT_GATE;
    if (!time_input || !end_input || !hidden || !at_t || !at_r || !values) {
        fail(error, error_size, "out of memory computing two-time embeddings");
        goto cleanup;
    }
    if (h3_weight_find(weights, "endpoint_time_embedder.gate", NULL)) {
        uint64_t shape[] = {1};
        float *stored = host_f32(weights, gpu, "endpoint_time_embedder.gate",
                                 1, shape, error, error_size);
        if (!stored) goto cleanup;
        gate = stored[0];
        free(stored);
    } else {
        fprintf(stderr, "h3: no endpoint_time_embedder.gate, using HyperFlow's %.2f\n", gate);
    }
    if (!load_host_embedder(weights, gpu, "time_embedder", time_dim, &current,
                            error, error_size) ||
        !load_host_embedder(weights, gpu, "endpoint_time_embedder", time_dim,
                            &endpoint, error, error_size)) goto cleanup;
    for (uint32_t row = 0; row < rows; row++) {
        embed_host(&current, time_dim, time_input + (size_t)row * TIME_INPUT,
                   hidden, at_t);
        embed_host(&endpoint, time_dim, end_input + (size_t)row * TIME_INPUT,
                   hidden, at_r);
        for (uint32_t column = 0; column < time_dim; column++) {
            float mixed = (float)(at_t[column] + (double)gate *
                                  (at_r[column] - at_t[column]));
            float activated = mixed / (1.0f + expf(-mixed));
            values[(size_t)row * time_dim + column] = f32_to_bf16(activated);
        }
    }
    result = h3_gpu_tensor_from_bf16(gpu, values, (size_t)rows * time_dim);
    if (!result)
        fail(error, error_size, "cannot upload two-time embeddings: %s",
             h3_gpu_error(gpu));
    if (getenv("H3_PROFILE")) {
        fprintf(stderr, "h3: HyperFlow two-time AdaLN, gate %.3f, %u rows:", gate, rows);
        for (uint32_t row = 0; row < rows; row++)
            fprintf(stderr, " (%.5f,%.5f)", times[row], ends[row]);
        fputc('\n', stderr);
    }
cleanup:
    free_host_embedder(&current);
    free_host_embedder(&endpoint);
    free(time_input);
    free(end_input);
    free(hidden);
    free(at_t);
    free(at_r);
    free(values);
    return result;
}

h3_dit_schedule *h3_dit_schedule_precompute(
    const h3_weight_store *weights, h3_gpu *gpu,
    const h3_sigma_schedule *sigmas, int visual_condition,
    int audio_condition,
    h3_dit_schedule_progress progress, void *progress_opaque,
    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weights || !gpu || !sigmas || sigmas->steps < 1 ||
        sigmas->steps > H3_MAX_STEPS) {
        fail(error, error_size, "invalid AdaLN schedule arguments");
        return NULL;
    }
    h3_dit_schedule *schedule = calloc(1, sizeof(*schedule));
    if (!schedule) {
        fail(error, error_size, "out of memory creating AdaLN schedule");
        return NULL;
    }
    schedule->gpu = gpu;
    const h3_st_tensor *time_out = h3_weight_find(
        weights, "time_embedder.proj_out.weight", NULL);
    if (!time_out || time_out->ndim != 2 || !time_out->shape[0] ||
        time_out->shape[0] > TIME_HIDDEN) {
        fail(error, error_size, "missing or invalid time_embedder.proj_out.weight");
        h3_dit_schedule_free(schedule);
        return NULL;
    }
    schedule->time_dim = (uint32_t)time_out->shape[0];
    schedule->two_time = h3_weight_find(
        weights, "endpoint_time_embedder.proj_in.weight", NULL) != NULL;
    float *times = NULL, *ends = NULL, *features = NULL;
    h3_gpu_tensor *time = NULL;
    if (!prepare_rows(schedule, sigmas, visual_condition, audio_condition,
                      &times, &ends, error, error_size)) goto failed;
    if (schedule->two_time) {
        time = two_time_embeddings(weights, gpu, schedule->time_rows,
                                   schedule->time_dim, times, ends,
                                   error, error_size);
    } else {
        features = time_features(times, schedule->time_rows);
        if (!features) {
            fail(error, error_size, "out of memory allocating timestep features");
            goto failed;
        }
        time = time_embeddings(weights, gpu, schedule->time_rows,
                               schedule->time_dim, features, error, error_size);
    }
    free(times);
    free(ends);
    free(features);
    times = ends = features = NULL;
    if (!time) goto failed;

    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        char weight_name[128], bias_name[128], operation[128];
        snprintf(weight_name, sizeof(weight_name),
                 "blocks.%u.adaln_proj.linear.weight", block);
        snprintf(bias_name, sizeof(bias_name),
                 "blocks.%u.adaln_proj.linear.bias", block);
        /* A block HyperFlow removed has no weights and no modulation. */
        if (schedule->two_time && !h3_weight_find(weights, weight_name, NULL)) {
            if (progress) progress((int)block + 1, (int)H3_DIT_BLOCKS,
                                   progress_opaque);
            continue;
        }
        h3_gpu_tensor *weight = weight_bf16_2d(
            weights, gpu, weight_name, BLOCK_OUTPUT, schedule->time_dim,
            error, error_size);
        h3_gpu_tensor *bias = weight_bf16_1d(
            weights, gpu, bias_name, BLOCK_OUTPUT, error, error_size);
        schedule->blocks[block] = h3_gpu_tensor_new_bf16(
            gpu, (size_t)schedule->time_rows * BLOCK_OUTPUT);
        if (!weight || !bias || !schedule->blocks[block]) {
            if (!error || !*error)
                fail(error, error_size, "cannot allocate AdaLN block %u: %s",
                     block, h3_gpu_error(gpu));
            free_tensor(&weight);
            free_tensor(&bias);
            h3_gpu_tensor_free(time);
            goto failed;
        }
        snprintf(operation, sizeof(operation), "AdaLN block %u", block);
        int ok = gpu_op(gpu, h3_gpu_begin(gpu), error, error_size, operation) &&
            gpu_op(gpu, h3_gpu_linear_bf16(
                gpu, schedule->blocks[block], time, weight, bias,
                schedule->time_rows, schedule->time_dim, BLOCK_OUTPUT),
                error, error_size, operation) &&
            gpu_op(gpu, h3_gpu_submit(gpu), error, error_size, operation);
        free_tensor(&weight);
        free_tensor(&bias);
        if (!ok) {
            h3_gpu_tensor_free(time);
            goto failed;
        }
        if (progress) progress((int)block + 1, (int)H3_DIT_BLOCKS,
                               progress_opaque);
    }

    h3_gpu_tensor *final_w = weight_bf16_2d(
        weights, gpu, "final_layer.adaln_proj.linear.weight",
        FINAL_OUTPUT, schedule->time_dim, error, error_size);
    h3_gpu_tensor *final_b = weight_bf16_1d(
        weights, gpu, "final_layer.adaln_proj.linear.bias",
        FINAL_OUTPUT, error, error_size);
    schedule->final = h3_gpu_tensor_new_bf16(
        gpu, (size_t)schedule->time_rows * FINAL_OUTPUT);
    if (!final_w || !final_b || !schedule->final ||
        !gpu_op(gpu, h3_gpu_begin(gpu), error, error_size,
                "begin final AdaLN") ||
        !gpu_op(gpu, h3_gpu_linear_bf16(
            gpu, schedule->final, time, final_w, final_b, schedule->time_rows,
            schedule->time_dim, FINAL_OUTPUT), error, error_size,
            "final AdaLN projection") ||
        !gpu_op(gpu, h3_gpu_submit(gpu), error, error_size,
                "submit final AdaLN")) {
        if ((!error || !*error) && (!final_w || !final_b || !schedule->final))
            fail(error, error_size, "cannot allocate final AdaLN tensors: %s",
                 h3_gpu_error(gpu));
        free_tensor(&final_w);
        free_tensor(&final_b);
        h3_gpu_tensor_free(time);
        goto failed;
    }
    free_tensor(&final_w);
    free_tensor(&final_b);
    if (getenv("H3_DEBUG_SCHEDULE")) {
        /* Debug: non-finite and mean |x| of the time input and the AdaLN tables. */
        const h3_gpu_tensor *tables[] = {time, schedule->blocks[0], schedule->blocks[1], schedule->final};
        const char *names[] = {"time", "block0", "block1", "final"};
        for (int t = 0; t < 4; t++) {
            if (!tables[t]) continue;
            size_t count = h3_gpu_tensor_elements(tables[t]);
            uint16_t *values = malloc(count * sizeof(*values));
            if (!values || !h3_gpu_tensor_read_bf16(tables[t], values, count)) { free(values); continue; }
            size_t bad = 0; double total = 0.0;
            for (size_t i = 0; i < count; i++) {
                uint32_t bits = (uint32_t)values[i] << 16; float v; memcpy(&v, &bits, sizeof(v));
                if (!isfinite(v)) bad++; else total += fabs((double)v);
            }
            fprintf(stderr, "h3: schedule %s: %zu values, %zu non-finite, mean|x| %.4g\n",
                    names[t], count, bad, total / (double)(count ? count : 1));
            free(values);
        }
    }
    h3_gpu_tensor_free(time);
    return schedule;

failed:
    free(times);
    free(ends);
    free(features);
    h3_dit_schedule_free(schedule);
    return NULL;
}

void h3_dit_schedule_free(h3_dit_schedule *schedule) {
    if (!schedule) return;
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        h3_gpu_tensor_free(schedule->blocks[block]);
    h3_gpu_tensor_free(schedule->final);
    free(schedule->video_rows);
    free(schedule->audio_rows);
    free(schedule->visual_condition_rows);
    free(schedule->audio_condition_rows);
    free(schedule);
}

int h3_dit_schedule_steps(const h3_dit_schedule *schedule) {
    return schedule ? schedule->steps : 0;
}

uint32_t h3_dit_schedule_time_rows(const h3_dit_schedule *schedule) {
    return schedule ? schedule->time_rows : 0;
}

uint32_t h3_dit_schedule_video_row(const h3_dit_schedule *schedule, int step) {
    return schedule && step >= 0 && step < schedule->steps ?
        schedule->video_rows[step] : UINT32_MAX;
}

uint32_t h3_dit_schedule_audio_row(const h3_dit_schedule *schedule, int step) {
    return schedule && step >= 0 && step < schedule->steps ?
        schedule->audio_rows[step] : UINT32_MAX;
}

uint32_t h3_dit_schedule_visual_condition_row(
    const h3_dit_schedule *schedule, int step) {
    return schedule && schedule->visual_condition_rows && step >= 0 &&
        step < schedule->steps ? schedule->visual_condition_rows[step] :
        UINT32_MAX;
}

uint32_t h3_dit_schedule_audio_condition_row(
    const h3_dit_schedule *schedule, int step) {
    return schedule && schedule->audio_condition_rows && step >= 0 &&
        step < schedule->steps ? schedule->audio_condition_rows[step] :
        UINT32_MAX;
}

const h3_gpu_tensor *h3_dit_schedule_block(const h3_dit_schedule *schedule,
                                           unsigned block) {
    return schedule && block < H3_DIT_BLOCKS ? schedule->blocks[block] : NULL;
}

double h3_dit_schedule_gate_score(const h3_dit_schedule *schedule,
                                  unsigned block) {
    if (!schedule || block >= H3_DIT_BLOCKS || !schedule->blocks[block])
        return -1.0;
    size_t count = (size_t)schedule->time_rows * BLOCK_OUTPUT;
    uint16_t *values = malloc(count * sizeof(*values));
    if (!values || !h3_gpu_tensor_read_bf16(schedule->blocks[block], values,
                                             count)) {
        free(values);
        return -1.0;
    }
    double total = 0.0;
    size_t samples = 0;
    for (uint32_t row = 0; row < schedule->time_rows; row++)
        for (uint32_t modality = 0; modality < H3_DIT_MODALITIES; modality++)
            for (uint32_t slot = 2; slot <= 5; slot += 3) {
                size_t base = ((size_t)row * H3_DIT_MODALITIES *
                               H3_DIT_ADALN_SLOTS +
                               (size_t)modality * H3_DIT_ADALN_SLOTS + slot) *
                              H3_DIT_HIDDEN;
                for (uint32_t column = 0; column < H3_DIT_HIDDEN; column++) {
                    uint32_t bits = (uint32_t)values[base + column] << 16;
                    float value;
                    memcpy(&value, &bits, sizeof(value));
                    total += fabs((double)value);
                }
                samples += H3_DIT_HIDDEN;
            }
    free(values);
    return samples ? total / (double)samples : -1.0;
}

void h3_dit_schedule_prune(h3_dit_schedule *schedule,
                           const uint8_t *active_blocks, size_t count) {
    if (!schedule || !active_blocks || count != H3_DIT_BLOCKS) return;
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        if (active_blocks[block]) continue;
        h3_gpu_tensor_free(schedule->blocks[block]);
        schedule->blocks[block] = NULL;
    }
}

const h3_gpu_tensor *h3_dit_schedule_final(const h3_dit_schedule *schedule) {
    return schedule ? schedule->final : NULL;
}

int h3_dit_schedule_row_map(const h3_dit_schedule *schedule, int step,
                            const h3_layout *layout,
                            const uint8_t *text_tags, size_t text_tag_count,
                            uint32_t *rows, size_t row_count) {
    if (!schedule || step < 0 || step >= schedule->steps || !layout || !rows ||
        row_count != layout->seq_len || !layout->segments ||
        (text_tags && text_tag_count != (size_t)layout->signature[0])) return 0;
    size_t text_index = 0;
    for (size_t seg_index = 0; seg_index < layout->segment_count; seg_index++) {
        const h3_segment *segment = &layout->segments[seg_index];
        if (segment->start > segment->stop || segment->stop > row_count)
            return 0;
        uint32_t time_row;
        uint32_t tag;
        switch (segment->kind) {
        case H3_SEG_TEXT:
            time_row = schedule->video_rows[step];
            for (size_t row = segment->start; row < segment->stop; row++) {
                uint32_t text_tag = text_tags ? text_tags[text_index] : 1u;
                if (text_tag >= H3_DIT_MODALITIES) return 0;
                rows[row] = time_row * H3_DIT_MODALITIES + text_tag;
                text_index++;
            }
            continue;
        case H3_SEG_COND:
        case H3_SEG_REF_IMAGE:
            if (!schedule->visual_condition_rows) return 0;
            time_row = schedule->visual_condition_rows[step];
            tag = 0;
            break;
        case H3_SEG_REF_AUDIO:
            if (!schedule->audio_condition_rows) return 0;
            time_row = schedule->audio_condition_rows[step];
            tag = 2;
            break;
        case H3_SEG_AUDIO:
            time_row = schedule->audio_rows[step];
            tag = 2;
            break;
        case H3_SEG_VIDEO:
            time_row = schedule->video_rows[step];
            tag = 0;
            break;
        default:
            return 0;
        }
        uint32_t modulation = time_row * H3_DIT_MODALITIES + tag;
        for (size_t row = segment->start; row < segment->stop; row++)
            rows[row] = modulation;
    }
    return text_index == (size_t)layout->signature[0];
}
