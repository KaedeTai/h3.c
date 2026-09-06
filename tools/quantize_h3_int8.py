#!/usr/bin/env python3
"""Pre-quantize a DiT transformer directory to the int8 form h3.c produces at load.

h3.c reads BF16 weights and then quantizes the four big per-block projections to
int8 on the GPU every time it loads a model (`quantize_block_mlp/qkv/attention_out`
in h3_dit.c). That means each run reads 62 GB from disk to build ~19 GB of int8.
This script does that step once, offline, so h3.c can load the int8 directly.

The arithmetic is a faithful port of `h3_quantize_bf16_int8_rows` in
h3_shaders.metal, so the result is bit-identical to what the GPU would compute:

    max_abs = max |float32(w)| over the row
    scale   = max_abs > 0 ? max_abs / 127 : 1/127
    inverse = max_abs > 0 ? 127 / max_abs : 127
    q       = clamp(rint(float32(w) * inverse), -127, 127)        (round half to even)

Everything else in the checkpoint is copied byte for byte.

Output layout, matching what ComfyUI-style int8 checkpoints use so the same
loader reads both:
    <name>.weight        I8  [rows, cols]
    <name>.weight_scale  F32 [rows]

    quantize_h3_int8.py SRC_TRANSFORMER_DIR DST_DIR [--shard-gb 5]
"""
import argparse, json, os, struct, sys
import numpy as np

# The four projections h3.c quantizes at load. Everything else stays as-is;
# adaln_proj in particular is consumed by an BF16 kernel and is left alone.
QUANT_SUFFIXES = (
    "attn.qkv_proj.weight",
    "attn.out_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
)
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1, "I32": 4, "I64": 8, "F64": 8, "BOOL": 1}


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        head = json.loads(fh.read(n))
    head.pop("__metadata__", None)
    return head, 8 + n


def should_quantize(name):
    return name.startswith("blocks.") and any(name.endswith(s) for s in QUANT_SUFFIXES)


def bf16_to_f32(raw, count):
    u16 = np.frombuffer(raw, dtype=np.uint16, count=count)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def quantize_rows(w32):
    """Port of h3_quantize_bf16_int8_rows (clip = 1.0)."""
    max_abs = np.abs(w32).max(axis=1).astype(np.float32)
    pos = max_abs > np.float32(0)
    scale = np.where(pos, max_abs / np.float32(127), np.float32(1) / np.float32(127)).astype(np.float32)
    inverse = np.where(pos, np.float32(127) / max_abs, np.float32(127)).astype(np.float32)
    q = np.rint(w32 * inverse[:, None]).astype(np.float32)
    return np.clip(q, -127, 127).astype(np.int8), scale


class ShardWriter:
    def __init__(self, dst, limit):
        self.dst, self.limit = dst, limit
        self.index, self.entries, self.payload, self.bytes = 0, [], [], 0
        os.makedirs(dst, exist_ok=True)

    def add(self, name, dtype, shape, data):
        self.entries.append((name, dtype, list(shape), self.bytes, self.bytes + len(data)))
        self.payload.append(data); self.bytes += len(data)
        if self.bytes >= self.limit:
            self.flush()

    def flush(self):
        if not self.entries:
            return
        self.index += 1
        head = {n: {"dtype": d, "shape": s, "data_offsets": [a, b]} for n, d, s, a, b in self.entries}
        blob = json.dumps(head, separators=(",", ":")).encode()
        pad = (-len(blob)) % 8
        blob += b" " * pad
        path = os.path.join(self.dst, f"model-{self.index:05d}.safetensors")
        with open(path, "wb") as fh:
            fh.write(struct.pack("<Q", len(blob))); fh.write(blob)
            for chunk in self.payload:
                fh.write(chunk)
        print(f"  wrote {os.path.basename(path)} ({self.bytes/1e9:.1f} GB, {len(self.entries)} tensors)", flush=True)
        self.entries, self.payload, self.bytes = [], [], 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--shard-gb", type=float, default=5.0)
    a = ap.parse_args()
    src = os.path.abspath(a.src); dst = os.path.abspath(a.dst)
    shards = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    if not shards:
        sys.exit(f"no safetensors in {src}")

    writer = ShardWriter(dst, int(a.shard_gb * 1e9))
    n_quant = n_copy = 0; in_bytes = out_bytes = 0
    for shard in shards:
        path = os.path.join(src, shard)
        head, base = read_header(path)
        with open(path, "rb") as fh:
            for name in sorted(head, key=lambda k: head[k]["data_offsets"][0]):
                info = head[name]
                start, end = info["data_offsets"]
                fh.seek(base + start); raw = fh.read(end - start)
                in_bytes += len(raw)
                if should_quantize(name) and info["dtype"] == "BF16" and len(info["shape"]) == 2:
                    rows, cols = info["shape"]
                    w32 = bf16_to_f32(raw, rows * cols).reshape(rows, cols)
                    q, scale = quantize_rows(w32)
                    writer.add(name, "I8", [rows, cols], q.tobytes())
                    writer.add(name + "_scale", "F32", [rows], scale.tobytes())
                    out_bytes += q.nbytes + scale.nbytes; n_quant += 1
                else:
                    writer.add(name, info["dtype"], info["shape"], raw)
                    out_bytes += len(raw); n_copy += 1
        print(f"{shard} done ({in_bytes/1e9:.1f} GB read)", flush=True)
    writer.flush()
    # h3.c detects the Ref2VA checkpoint by the presence of this file
    # (h3.c: h3_is_file("Ref2VA/transformer/model.safetensors.index.json")),
    # so it has to exist even though the weight store itself just scans the
    # directory for *.safetensors.
    weight_map, total = {}, 0
    for shard in sorted(f for f in os.listdir(dst) if f.endswith(".safetensors")):
        head, _ = read_header(os.path.join(dst, shard))
        for k, v in head.items():
            weight_map[k] = shard
            total += v["data_offsets"][1] - v["data_offsets"][0]
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, fh, indent=1)
    print(f"index.json: {len(weight_map)} tensors")
    # Non-weight files (config.json) sit beside the shards and h3.c requires them.
    for extra in sorted(os.listdir(src)):
        if extra.endswith(".safetensors") or extra == "model.safetensors.index.json":
            continue
        target = os.path.join(dst, extra)
        if os.path.isfile(os.path.join(src, extra)) and not os.path.exists(target):
            try:
                os.link(os.path.join(src, extra), target)
            except OSError:
                import shutil; shutil.copy2(os.path.join(src, extra), target)
            print(f"  copied {extra}")
    print(f"\nquantized {n_quant}, copied {n_copy}")
    print(f"in  {in_bytes/1e9:.1f} GB -> out {out_bytes/1e9:.1f} GB  ({out_bytes/in_bytes*100:.0f}%)")


if __name__ == "__main__":
    main()
