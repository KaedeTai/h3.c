#!/usr/bin/env python3
"""Convert a ComfyUI int8 MiniMax-H3 export (ConvRot or plain) into h3.c's layout.

ComfyUI's `comfy_quant` int8 exports store each quantized weight as
`<name>.weight` I8 [out, in] + `<name>.weight_scale` F32 [out, 1] + a
`<name>.comfy_quant` JSON blob. With `"convrot": true` the int8 grid lives in
a rotated basis: every run of `convrot_groupsize` input columns was multiplied
by a normalized regular Hadamard matrix H before quantizing, and ComfyUI
rotates activations to match at run time. h3.c's int8 kernels do not rotate,
so the weight is brought back first -- W = (q * scale) @ H per group (H is
symmetric and orthogonal, so it is its own inverse; see comfy_kitchen
`dequantize_int8_convrot_weight`) -- and then requantized with h3.c's own
per-row rule (tools/quantize_h3_int8.py).

Fused qkv weights may also come in (3, heads, 128) row order where h3.c uses
(heads, 3, 128); each one is checked against the native weight and reordered
if needed.

Every tensor is written in the dtype and shape the matching native checkpoint
uses, so h3.c's strict loader sees exactly what it expects: the four big
per-block projections as I8 + `_scale`, adaln_proj as BF16, the patch and
time-embedder weights as F32. Names drop the `model.diffusion_model.` prefix.

--lora PATH[:STRENGTH] (repeatable) folds adapters in while the weights are
still float, before requantizing -- the exact merge, as patch_h3_turbo.py did,
rather than the load-time one h3.c has to do on an int8 bundle.

    convert_comfy_int8.py SRC.safetensors NATIVE_TRANSFORMER_DIR DST_DIR \\
        [--lora turbo.safetensors:1.0] [--shard-gb 5]
"""
import argparse, json, os, struct, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize_h3_int8 import ShardWriter, quantize_rows, read_header  # noqa: E402
from check_lora_merge import header as lora_header, read as lora_read  # noqa: E402

PREFIXES = ("model.diffusion_model.", "diffusion_model.")
NP = {"F32": np.float32, "F16": np.float16, "I8": np.int8, "U8": np.uint8,
      "BF16": np.uint16, "I32": np.int32, "I64": np.int64}


def hadamard(size):
    """comfy_kitchen `_build_hadamard`: Kronecker powers of a regular 4x4."""
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                  dtype=np.float64)
    h, n = h4, 4
    while n < size:
        h, n = np.kron(h, h4), n * 4
    if n != size:
        raise SystemExit(f"ConvRot group size {size} is not a power of 4")
    return (h / np.sqrt(size)).astype(np.float32)


class Source:
    def __init__(self, path):
        self.head, self.base = read_header(path)
        self.map = np.memmap(path, dtype=np.uint8, mode="r")
        self.names = {}
        for k in self.head:
            short = k
            for p in PREFIXES:
                if short.startswith(p):
                    short = short[len(p):]
                    break
            self.names[short] = k

    def raw(self, name):
        info = self.head[self.names[name]]
        a, b = info["data_offsets"]
        arr = self.map[self.base + a:self.base + b].view(NP[info["dtype"]])
        return arr.reshape(info["shape"]), info["dtype"]

    def has(self, name):
        return name in self.names

    def float_weight(self, name):
        """The weight as float32, whatever it is stored as."""
        arr, dtype = self.raw(name)
        stem = name[: -len(".weight")] if name.endswith(".weight") else name
        if dtype == "I8":
            quant = json.loads(bytes(self.raw(stem + ".comfy_quant")[0]).decode())
            if quant.get("format") != "int8_tensorwise":
                raise SystemExit(f"{name}: unsupported comfy_quant {quant}")
            scale = self.raw(name + "_scale")[0].astype(np.float32).reshape(-1, 1)
            w = arr.astype(np.float32) * scale if scale.size > 1 else \
                arr.astype(np.float32) * float(scale.reshape(-1)[0])
            if quant.get("convrot"):
                g = int(quant.get("convrot_groupsize", 256))
                rows, cols = w.shape
                w = (w.reshape(-1, g) @ hadamard(g)).reshape(rows, cols)
            return w
        if dtype == "BF16":
            return (arr.astype(np.uint32) << 16).view(np.float32)
        return arr.astype(np.float32)


HEADS, HEAD_DIM = 56, 128


def heads_to_native(w):
    """(3, heads, d) rows -> h3's (heads, 3, d) rows for a fused qkv weight."""
    rows, cols = w.shape
    return w.reshape(3, HEADS, HEAD_DIM, cols).transpose(1, 0, 2, 3).reshape(rows, cols)


class Native:
    """The matching native checkpoint, used as a layout reference."""

    def __init__(self, directory):
        self.head = {}
        for fn in sorted(os.listdir(directory)):
            if fn.endswith(".safetensors"):
                path = os.path.join(directory, fn)
                h, base = read_header(path)
                for k, v in h.items():
                    self.head[k] = (path, base, v)

    def rows(self, name, index):
        path, base, info = self.head[name]
        cols = info["shape"][1]
        size = {"I8": 1, "BF16": 2, "F32": 4}[info["dtype"]]
        out = []
        with open(path, "rb") as fh:
            for r in index:
                fh.seek(base + info["data_offsets"][0] + int(r) * cols * size)
                raw = np.frombuffer(fh.read(cols * size), dtype=NP[info["dtype"]])
                out.append((raw.astype(np.uint32) << 16).view(np.float32)
                           if info["dtype"] == "BF16" else raw.astype(np.float32))
        return np.stack(out)


def qkv_layout(w, native, name):
    """Which row order this fused qkv uses, decided against the native weight.
    Returns the weight in native order. The scale does not matter for a
    cosine, so the native int8 rows are compared undequantized."""
    probe = np.arange(128, w.shape[0], 997)[:24]
    ref = native.rows(name, probe)
    def score(x):
        x = x[probe]
        return float(np.mean(np.sum(x * ref, 1) /
                             (np.linalg.norm(x, axis=1) * np.linalg.norm(ref, axis=1) + 1e-30)))
    as_is, permuted = score(w), score(heads_to_native(w))
    if max(as_is, permuted) < 0.5:
        raise SystemExit(f"{name}: matches neither qkv layout (cos {as_is:.3f} / {permuted:.3f})")
    return (heads_to_native(w), "permuted") if permuted > as_is else (w, "as-is")


def to_bf16(w):
    u = np.ascontiguousarray(w, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def lora_deltas(specs, qkv_order="comfy"):
    """{native module name: [(A, B, scale)]} from --lora specs. Fused-qkv B
    rows are taken to be in ComfyUI order unless qkv_order is "native"
    (see h3_lora.h and tools/lora_qkv_layout.py)."""
    out = {}
    for spec in specs:
        path, _, strength = spec.rpartition(":")
        if not path or not strength.replace(".", "", 1).lstrip("-").isdigit():
            path, strength = spec, "1"
        h, base = lora_header(path)
        for k in h:
            for suffix in (".lora_A.weight", ".lora_A.default.weight", ".lora_down.weight"):
                if not k.endswith(suffix):
                    continue
                module = k[: -len(suffix)]
                b_key = module + suffix.replace("lora_A", "lora_B").replace("lora_down", "lora_up")
                A = lora_read(path, h, base, k).astype(np.float32)
                B = lora_read(path, h, base, b_key).astype(np.float32)
                alpha = float(lora_read(path, h, base, module + ".alpha")) \
                    if module + ".alpha" in h else A.shape[0]
                name = module
                for p in PREFIXES + ("transformer.",):
                    if name.startswith(p):
                        name = name[len(p):]
                if name.endswith("attn.qkv_proj") and qkv_order == "comfy":
                    B = heads_to_native(B)
                out.setdefault(name + ".weight", []).append(
                    (A, B, float(strength) * alpha / A.shape[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("native"); ap.add_argument("dst")
    ap.add_argument("--lora", action="append", default=[])
    ap.add_argument("--lora-qkv", choices=("comfy", "native"), default="comfy",
                    help="row order of the adapters' fused qkv (default comfy)")
    ap.add_argument("--shard-gb", type=float, default=5.0)
    a = ap.parse_args()
    src = Source(a.src)
    reference = Native(a.native)
    native = {}
    for fn in sorted(os.listdir(a.native)):
        if fn.endswith(".safetensors"):
            h, _ = read_header(os.path.join(a.native, fn))
            native.update(h)
    wanted = sorted(k for k in native if not k.endswith("_scale"))
    missing = [k for k in wanted if not src.has(k)]
    if missing:
        raise SystemExit(f"source lacks {len(missing)} native tensors, e.g. {missing[:3]}")
    deltas = lora_deltas(a.lora, a.lora_qkv)
    unused = set(deltas) - set(wanted)
    if unused:
        raise SystemExit(f"LoRA targets unknown weights, e.g. {sorted(unused)[:3]}")
    writer = ShardWriter(os.path.abspath(a.dst), int(a.shard_gb * 1e9))
    t0 = time.time(); merged = 0; layouts = {}
    for i, name in enumerate(wanted):
        info = native[name]
        want, shape = info["dtype"], info["shape"]
        w = src.float_weight(name)
        if list(w.shape) != list(shape):
            raise SystemExit(f"{name}: source {list(w.shape)} vs native {shape}")
        if name.endswith("attn.qkv_proj.weight"):
            w, how = qkv_layout(w, reference, name)
            layouts[how] = layouts.get(how, 0) + 1
        for A, B, s in deltas.get(name, ()):
            w = w + s * (B @ A); merged += 1
        if want == "I8":
            q, scale = quantize_rows(np.ascontiguousarray(w, dtype=np.float32))
            writer.add(name, "I8", shape, q.tobytes())
            writer.add(name + "_scale", "F32", [shape[0]], scale.astype(np.float32).tobytes())
        elif want == "BF16":
            writer.add(name, "BF16", shape, to_bf16(w).tobytes())
        elif want == "F32":
            writer.add(name, "F32", shape, np.ascontiguousarray(w, dtype=np.float32).tobytes())
        else:
            raise SystemExit(f"{name}: native dtype {want} not handled")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(wanted)} tensors ({time.time()-t0:.0f}s)", flush=True)
    writer.flush()
    weight_map, total = {}, 0
    for shard in sorted(f for f in os.listdir(a.dst) if f.endswith(".safetensors")):
        head, _ = read_header(os.path.join(a.dst, shard))
        for k, v in head.items():
            weight_map[k] = shard
            total += v["data_offsets"][1] - v["data_offsets"][0]
    with open(os.path.join(a.dst, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, fh, indent=1)
    cfg = os.path.join(a.native, "config.json")
    if os.path.exists(cfg):
        import shutil; shutil.copy2(cfg, os.path.join(a.dst, "config.json"))
    print(f"qkv row order: {layouts}")
    print(f"wrote {len(wanted)} tensors ({merged} LoRA merges) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
