"""Repack jina-ocr-v1 for mlx-vlm's `deepseekocr` module.

STATUS (2026-09-19): the weights convert and load — in 0.5 s against the
PyTorch path's 10 — and then the model answers the same hallucinated
paragraph for every image, with `Ġ` byte-BPE markers leaking into the text.
The image never reaches the tower and the detokeniser is wrong, both through
mlx-vlm directly and through oMLX's own VLM engine, so it is not the calling
convention. The cause is that `deepseekocr` is built for DeepSeek-OCR's
tokenizer and processor (vocab 102400); jina's are its own (vocab 129280,
`processing_deepseek_ocr.py`). Making this work needs a jina-shaped module in
mlx-vlm, not a config tweak. Kept so the next attempt starts here rather than
at the beginning: everything below is correct, it is the processor that is
missing.

Two changes, both mechanical:

  * `model_type` becomes `deepseekocr`. The checkpoint says `deepseek_vl_v2`,
    which routes to a module built for a SigLIP tower; jina's vision_config
    carries `clip-l-14-224` + `sam_vit_b`, which is DeepSeek-OCR v1's
    DeepEncoder, and that is what `deepseekocr` implements.
  * the `mtp_module.*` tensors are dropped. They are the FastMTP speculative
    head; mlx-vlm does not model it, so loading fails with twelve unexpected
    parameters. Dropping them costs the ~2x speculative decode and changes
    nothing about what the model outputs.
"""
import glob, json, os, shutil, sys
import mlx.core as mx
from mlx.utils import tree_flatten
import mlx.nn as nn
from mlx_vlm.utils import load, save_config, save_weights

SRC = os.path.expanduser("~/models/jina-ocr-v1")
DST = os.path.expanduser("~/models/jina-ocr-v1-mlx-8bit")
BITS = 8

os.makedirs(DST, exist_ok=True)
weights = {}
dropped = []
for shard in sorted(glob.glob(os.path.join(SRC, "*.safetensors"))):
    for key, value in mx.load(shard).items():
        if key.startswith("mtp_module."):
            dropped.append(key)
            continue
        weights[key] = value
print(f"kept {len(weights)} tensors, dropped {len(dropped)} FastMTP tensors", flush=True)

config = json.load(open(os.path.join(SRC, "config.json")))
config["model_type"] = "deepseekocr"
json.dump(config, open(os.path.join(DST, "config.json"), "w"), indent=1)
for name in os.listdir(SRC):
    if name.endswith((".py", ".json", ".jinja")) and name != "config.json" \
            and not name.startswith("model-"):
        shutil.copy2(os.path.join(SRC, name), os.path.join(DST, name))

# Build the module straight from the config. `load()` would read the shards
# itself and die on the FastMTP tensors before the filtered set gets a chance.
# Write the filtered weights first and let `load()` do the rest: it owns the
# key remapping (model.layers -> language_model.model.layers) and the expert
# stacking, and duplicating that here would be a second copy to keep right.
import glob as _glob
from mlx_vlm.utils import make_shards
shards = make_shards(weights)
fmt = "model-{:05d}-of-{:05d}.safetensors" if len(shards) > 1 else "model.safetensors"
index = {"metadata": {"total_size": sum(v.nbytes for v in weights.values())},
         "weight_map": {}}
for i, shard in enumerate(shards):
    name = fmt.format(i + 1, len(shards))
    mx.save_safetensors(os.path.join(DST, name), shard, metadata={"format": "pt"})
    for key in shard:
        index["weight_map"][key] = name
json.dump(index, open(os.path.join(DST, "model.safetensors.index.json"), "w"), indent=1)
model, processor = load(DST, trust_remote_code=True)
print("loaded from", DST, flush=True)
# Quantise the language model only: the DeepEncoder is small and the SAM
# tower's conv stack does not take kindly to 8-bit group quantisation.
def keep(path, module):
    return isinstance(module, (nn.Linear, nn.Embedding)) \
        and not path.startswith(("vision", "sam", "projector")) \
        and module.weight.shape[-1] % 64 == 0

nn.quantize(model, group_size=64, bits=BITS, class_predicate=keep)
config["quantization"] = {"group_size": 64, "bits": BITS}
save_weights(DST, model, donate_weights=True)
save_config(config, os.path.join(DST, "config.json"))
print(f"DONE {DST}", flush=True)
