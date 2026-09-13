# h3 as a MoneyPrinterTurbo material source

[MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) is an
assembly line: topic → LLM script → search terms → footage → TTS → subtitles →
ffmpeg. Every stage is good and none of it generates pictures — the footage
comes from stock APIs or from paid cloud video models. One of those paid
providers, `metaso_minimax`, calls **MiniMax-H3**: the same model this
repository runs locally.

These two files replace that call with `h3`, so the footage is generated on the
operator's own GPU at no per-clip cost and the rest of the pipeline is
untouched.

| file | what it is |
|---|---|
| `h3_runner.py` | drives the `h3` binary. Standard library only, no MoneyPrinterTurbo imports, so it can be tested on the machine that owns the GPU. |
| `h3_local.py` | the provider. Same signature as `volcengine_seedance.generate_videos` and friends. |
| `h3_local_provider.patch` | both files plus the four edits that wire them in. |

## Applying it

```sh
git clone https://github.com/harry0703/MoneyPrinterTurbo.git
cd MoneyPrinterTurbo
git apply /path/to/h3_local_provider.patch
```

Then in `config.toml`:

```toml
video_source = "h3_local"
h3_local_binary = "/Users/you/h3.c/h3"
```

The patch touches `app/services/material.py` (dispatch branch and an on-demand
download loop), `app/services/task.py` (preflight), `webui/Main.py` (source
list) and `config.example.toml`.

## Three decisions worth knowing

**The soundtrack is always stripped.** T2VA generates audio as well as video,
and the pipeline lays its own narration over the cut; a second voice underneath
it is never wanted. The strip is a `-c:v copy -an` remux, so it is nearly free.

**`H3_VAE_INT8_FFN=1` is set in code, not left to config.** Without it the
video VAE decode costs 69 s against 17 s on a ten-second clip, and a profile
taken without it blames the decoder for 57% of the run when its real share is
22%. That is not a flag to give anyone the chance to forget.

**The error handling is deliberately simpler than the paid providers'.** Every
cloud source in that project needs an "unconfirmed task — may already have been
billed — never retry" path. Local generation costs nothing, so a failed search
term is logged and skipped and the loop carries on.

## Measured

M5 Max, 3 steps, audio stripped, one process per clip, cold:

| term | canvas | frames | clip | wall |
|---|---|---|---|---|
| night market food stall | 320×576 (9:16) | 124 | 5.17 s | 31.2 s |
| busy taipei street at dusk | 576×320 (16:9) | 73 | 3.04 s | 17.9 s |

Canvases are the closest multiples of 32 to each aspect at about 0.18 MP.
Attention is quadratic in tokens and a bigger canvas also starts drawing
captions (see `FLAT2V.md`, *Crop to the head, not to the room*), so B-roll has
no reason to ask for more.

## Not done

One process per clip. Model load is 3–8 s against 18–31 s of denoising, so a
resident worker driven over h3's REPL would save under 10% — worth doing, but
not worth a protocol in the first version.
