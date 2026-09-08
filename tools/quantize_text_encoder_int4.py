#!/usr/bin/env python3
"""Write an int4 group-32 text encoder that h3 can load directly.

The fake-quant sweep already settled the accuracy question: rounding the 350
language-layer projections to int4 g32 moved the finished frames 4.68 mean
|pixel| against the 4.25 that two identical BF16 runs move on their own. So this
is only about getting those weights onto disk and back into h3 without going
through BF16 in between.

Layout, per projection <name> of shape [rows, cols]:

    <name>.q4   U8  [rows, cols/2]      two weights a byte, low nibble first,
                                        stored as q+8 so [-8,7] maps to [0,15]
    <name>.q4s  F16 [rows, cols/32]     one absmax scale per group of 32 along
                                        the input dimension

The original BF16 tensor is dropped. Everything else -- norms, embed_tokens,
the whole vision tower -- is copied byte for byte: they are 1-D or tiny, and
2.6 GB of the bundle is not where the problem is.

Grouping runs along the LAST axis on purpose. That is the reduction axis of the
matmul, so it is the axis whose dynamic range one scale actually has to cover;
grouping along the other one flatters the error and does not match how the
weights are read.

    quantize_text_encoder_int4.py SRC_DIR DST_DIR [--group 32]
"""
import argparse, json, mmap, os, re, shutil, struct, sys
import numpy as np

LINEAR = re.compile(
    r"model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")


def read_header(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return json.loads(f.read(n)), 8 + n


def pack_int4(w, group):
    """-> (packed uint8 [rows, cols/2], scales float16 [rows, cols/group])"""
    rows, cols = w.shape
    g = w.reshape(rows, cols // group, group).astype(np.float32)
    scale = np.abs(g).max(axis=2) / 7.0
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(g / scale[:, :, None]), -8, 7).astype(np.int8)
    q = (q + 8).astype(np.uint8).reshape(rows, cols)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return packed, scale.astype(np.float16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src"); p.add_argument("dst")
    p.add_argument("--group", type=int, default=32)
    a = p.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    weight_map, total_in, total_out = {}, 0, 0

    for fn in sorted(os.listdir(a.src)):
        s = os.path.join(a.src, fn)
        if not fn.endswith(".safetensors"):
            if fn != "model.safetensors.index.json":
                shutil.copy2(s, os.path.join(a.dst, fn))
            continue
        with open(s, "rb") as f:
            head, base = read_header(f)
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            meta = head.pop("__metadata__", None)
            names = sorted(head, key=lambda k: head[k]["data_offsets"][0])
            blobs, order, quantised = {}, [], 0
            for name in names:
                e = head[name]
                lo, hi = e["data_offsets"]
                raw = mm[base + lo: base + hi]
                total_in += hi - lo
                if LINEAR.match(name) and e["dtype"] == "BF16":
                    rows, cols = e["shape"]
                    if cols % a.group:
                        raise SystemExit(f"{name}: {cols} not a multiple of {a.group}")
                    w = ((np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16)
                         .view(np.float32).reshape(rows, cols))
                    packed, scale = pack_int4(w, a.group)
                    blobs[name + ".q4"] = (packed.tobytes(), "U8", list(packed.shape))
                    blobs[name + ".q4s"] = (scale.tobytes(), "F16", list(scale.shape))
                    order += [name + ".q4", name + ".q4s"]
                    quantised += 1
                else:
                    blobs[name] = (raw, e["dtype"], e["shape"])
                    order.append(name)
            out_head, off = {}, 0
            for name in order:
                payload, dtype, shape = blobs[name]
                out_head[name] = {"dtype": dtype, "shape": shape,
                                  "data_offsets": [off, off + len(payload)]}
                off += len(payload)
                weight_map[name] = fn
            if meta is not None:
                out_head["__metadata__"] = meta
            js = json.dumps(out_head, separators=(",", ":")).encode()
            pad = (-len(js)) % 8
            d = os.path.join(a.dst, fn)
            with open(d, "wb") as g:
                g.write(struct.pack("<Q", len(js) + pad))
                g.write(js); g.write(b" " * pad)
                for name in order:
                    g.write(blobs[name][0])
            mm.close()
        total_out += os.path.getsize(d)
        print(f"  {fn}: {quantised} projections -> int4 g{a.group}", flush=True)

    idx = os.path.join(a.src, "model.safetensors.index.json")
    if os.path.exists(idx):
        meta = json.load(open(idx)).get("metadata", {})
        meta["total_size"] = total_out
        json.dump({"metadata": meta, "weight_map": weight_map},
                  open(os.path.join(a.dst, "model.safetensors.index.json"), "w"))
    print(f"\n  {total_in/1e9:.2f} GB -> {total_out/1e9:.2f} GB "
          f"({total_out/total_in*100:.1f}%)")


if __name__ == "__main__":
    main()
