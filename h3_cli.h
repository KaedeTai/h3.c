#ifndef H3_CLI_H
#define H3_CLI_H

#include "h3.h"

/* Run the interactive prompt. The supplied parameters become session defaults.
 * seed_was_given preserves Iris's random-by-default interactive behavior while
 * honoring an explicit command-line --seed. */
int h3_cli_run(h3_ctx *ctx, const char *model_dir,
               const h3_params *initial, int show,
               int seed_was_given);

/* Print a warning when Ref2VA image references are combined with
 * --core-reuse > 1 or --token-reduction. Both skip or merge work inside the
 * core blocks, which is where the video tokens align to the reference tokens;
 * with the 4-step turbo LoRA the result is a smeared UI and a glowing subject
 * at every core-reuse value (measured 2, 3 and 4), and doubled limbs with
 * token reduction. Returns 1 when a warning was printed. */
int h3_warn_ref2va_knobs(const h3_params *params);

#endif
