#!/usr/bin/env python3
"""Round the text encoder's linear weights through a quantisation grid and
write them back as BF16.

"Can the text encoder go to Q4?" is two questions, and only one of them costs
engineering:

  1. Does Q4 hurt the output?   -- an accuracy question
  2. Can h3 run Q4 weights?     -- a Metal question. h3_text_encoder.c is BF16
                                   throughout; the tree has int8 kernels (the
                                   DiT and video VAE use them) and no int4
                                   anything, so Q4 is a new-kernel project.

This answers (1) without touching (2). Quantise, dequantise, store BF16. The
arithmetic h3 performs is unchanged; the numbers it performs it on are exactly
the numbers a real Q4 kernel would see. Whatever damage shows up here is a
lower bound on the real thing, which also accumulates in lower precision.

Only the 350 language-layer projections are touched: q/k/v/o and gate/up/down.
Norms are 1-D and free; embed_tokens is a lookup rather than a matmul and is a
separate decision; the vision tower is 1.11 GB and not worth the risk.

safetensors' numpy backend refuses BF16 outright ("data type 'bfloat16' not
understood"), so this walks the container itself: 8-byte little-endian header
length, JSON header, then a flat data block. Tensors that are not quantised are
copied byte for byte out of the source mapping.

    fakequant_text_encoder.py SRC DST --bits 4 --group 32
"""
import argparse, json, mmap, os, re, shutil, struct, sys
import numpy as np

LINEAR = re.compile(
    r"model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")


def read_header(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return json.loads(f.read(n)), 8 + n


def bf16_to_f32(raw):
    return (raw.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):
    """Round-to-nearest-even on the top 16 bits, the same rounding the original
    export used. Truncating instead biases every weight toward zero."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16))


def quant_dequant(w, bits, group):
    """Symmetric absmax, group-wise along the input dimension.

    The last axis is the reduction axis of the matmul, so it is the axis whose
    dynamic range one scale has to cover. Grouping along anything else would
    flatter the result.
    """
    shape = w.shape
    k = shape[-1]
    g = k if group <= 0 or k % group else group
    x = w.reshape(-1, g).astype(np.float32)
    hi = 2 ** (bits - 1) - 1
    scale = np.abs(x).max(axis=1, keepdims=True) / hi
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(x / scale), -hi - 1, hi)
    return (q * scale).reshape(shape).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src"); p.add_argument("dst")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group", type=int, default=32,
                   help="0 for one scale per output row")
    a = p.parse_args()
    os.makedirs(a.dst, exist_ok=True)

    quantised_params = 0
    errs = []
    for fn in sorted(os.listdir(a.src)):
        s, d = os.path.join(a.src, fn), os.path.join(a.dst, fn)
        if not fn.endswith(".safetensors"):
            shutil.copy2(s, d)
            continue
        with open(s, "rb") as f:
            head, base = read_header(f)
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            meta = head.pop("__metadata__", None)
            names = sorted(head, key=lambda k: head[k]["data_offsets"][0])
            blobs, touched = {}, 0
            for name in names:
                e = head[name]
                lo, hi = e["data_offsets"]
                raw = mm[base + lo: base + hi]
                if LINEAR.match(name) and e["dtype"] == "BF16":
                    w = bf16_to_f32(np.frombuffer(raw, dtype=np.uint16))
                    q = quant_dequant(w, a.bits, a.group)
                    denom = float(np.abs(w).mean()) + 1e-12
                    errs.append((float(np.abs(q - w).mean()) / denom, w.size))
                    quantised_params += w.size
                    blobs[name] = f32_to_bf16(q).tobytes()
                    touched += 1
                else:
                    blobs[name] = raw
            # rebuild the container in the source's own tensor order
            out_head, off = {}, 0
            for name in names:
                n = len(blobs[name])
                out_head[name] = {"dtype": head[name]["dtype"],
                                  "shape": head[name]["shape"],
                                  "data_offsets": [off, off + n]}
                off += n
            if meta is not None:
                out_head["__metadata__"] = meta
            js = json.dumps(out_head, separators=(",", ":")).encode()
            pad = (-len(js)) % 8
            with open(d, "wb") as g:
                g.write(struct.pack("<Q", len(js) + pad))
                g.write(js); g.write(b" " * pad)
                for name in names:
                    g.write(blobs[name])
            mm.close()
        print(f"  {fn}: {touched} quantised", flush=True)

    if errs:
        w = np.array([n for _, n in errs], dtype=float)
        e = np.array([v for v, _ in errs])
        g = a.group if a.group > 0 else 1024
        real = quantised_params * (a.bits / 8 + 2 / g) / 1e9
        print(f"\n  quantisable parameters : {quantised_params/1e9:.2f} B")
        print(f"  weight error           : {np.average(e, weights=w)*100:.2f}% "
              f"of mean |w|")
        print(f"  BF16 today             : {quantised_params*2/1e9:.2f} GB")
        print(f"  int{a.bits} g{a.group} would be   : {real:.2f} GB "
              f"(+ ~2.6 GB of norms, embeddings and vision tower)")


if __name__ == "__main__":
    main()
