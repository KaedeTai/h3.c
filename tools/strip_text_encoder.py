#!/usr/bin/env python3
"""Strip the parts of the H3 text encoder that h3 never reads.

The checkpoint is a full Qwen3-VL: 64 language layers plus an lm_head. h3 uses
the hidden states, not logits, and `TEXT_LAYERS = 50` in h3_text_encoder.c, so
layers 50-63 and lm_head are dead weight -- 14.16 GB of 62.13 GB. Nothing in the
C or Metal sources references either.

Copies shards byte-for-byte where a shard survives whole, and rewrites only the
shards that would otherwise carry a dropped tensor, so most of the work is a
plain copy and the container layout h3's loader depends on is preserved.

    strip_text_encoder.py SRC_DIR DST_DIR [--layers 50]
"""
import argparse, json, os, shutil, struct, sys, re

KEEP_NONLAYER_PREFIXES = ("model.visual", "visual", "model.language_model.embed_tokens",
                          "model.language_model.norm")
DROP_EXACT = ("lm_head.weight", "model.lm_head.weight")


def wanted(name, layers):
    if name in DROP_EXACT or name.endswith("lm_head.weight"):
        return False
    m = re.match(r"model\.language_model\.layers\.(\d+)\.", name)
    if m:
        return int(m.group(1)) < layers
    return True


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--layers", type=int, default=50,
                    help="keep language layers [0, N); h3 reads TEXT_LAYERS=50")
    a = ap.parse_args()
    src, dst = os.path.abspath(a.src), os.path.abspath(a.dst)
    os.makedirs(dst, exist_ok=True)

    index_path = os.path.join(src, "model.safetensors.index.json")
    index = json.load(open(index_path))
    weight_map = index["weight_map"]

    shards = sorted(set(weight_map.values()))
    kept_map, dropped, kept_bytes, dropped_bytes = {}, [], 0, 0
    DT = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1, "F8_E4M3": 1}

    for shard in shards:
        header, data_start = read_header(os.path.join(src, shard))
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        keep = {k: v for k, v in tensors.items() if wanted(k, a.layers)}
        drop = {k: v for k, v in tensors.items() if k not in keep}
        for k, v in drop.items():
            el = 1
            for s in v["shape"]: el *= s
            dropped_bytes += el * DT.get(v["dtype"], 2)
            dropped.append(k)

        if not keep:
            print(f"  {shard}: dropped entirely")
            continue
        if not drop:
            # untouched shard: copy the bytes, keeping h3's expected layout
            shutil.copyfile(os.path.join(src, shard), os.path.join(dst, shard))
            for k in keep: kept_map[k] = shard
            kept_bytes += os.path.getsize(os.path.join(dst, shard))
            print(f"  {shard}: copied whole ({len(keep)} tensors)")
            continue

        # mixed shard: rebuild it with only the survivors
        out_header, offset = {}, 0
        for k in sorted(keep, key=lambda n: keep[n]["data_offsets"][0]):
            v = keep[k]
            length = v["data_offsets"][1] - v["data_offsets"][0]
            out_header[k] = {"dtype": v["dtype"], "shape": v["shape"],
                             "data_offsets": [offset, offset + length]}
            offset += length
        blob = json.dumps(out_header, separators=(",", ":")).encode()
        pad = (-len(blob)) % 8
        blob += b" " * pad
        with open(os.path.join(src, shard), "rb") as fi, \
             open(os.path.join(dst, shard), "wb") as fo:
            fo.write(struct.pack("<Q", len(blob))); fo.write(blob)
            for k in sorted(keep, key=lambda n: keep[n]["data_offsets"][0]):
                s0, s1 = keep[k]["data_offsets"]
                fi.seek(data_start + s0)
                remaining = s1 - s0
                while remaining:
                    chunk = fi.read(min(remaining, 1 << 24))
                    if not chunk: raise SystemExit(f"short read in {shard}")
                    fo.write(chunk); remaining -= len(chunk)
                kept_map[k] = shard
        kept_bytes += os.path.getsize(os.path.join(dst, shard))
        print(f"  {shard}: rebuilt, kept {len(keep)} dropped {len(drop)}")

    index["weight_map"] = kept_map
    index.setdefault("metadata", {})["total_size"] = kept_bytes
    json.dump(index, open(os.path.join(dst, "model.safetensors.index.json"), "w"),
              indent=2)
    for fn in os.listdir(src):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        s = os.path.join(src, fn)
        if os.path.isfile(s):
            shutil.copyfile(s, os.path.join(dst, fn))
    print(f"\ndropped {len(dropped)} tensors, {dropped_bytes/2**30:.2f} GB")
    print(f"result  {kept_bytes/2**30:.2f} GB in {dst}")


if __name__ == "__main__":
    main()
