#!/usr/bin/env python3
"""Which row order does an H3 LoRA's fused-qkv B use?

h3.c and the official checkpoint order attn.qkv_proj rows (head, q|k|v, 128);
ComfyUI's checkpoints use (q|k|v, head, 128). A LoRA's B carries whichever
order its trainer saw, and nothing in the file says which. Fine-tuning deltas
lean toward the base weight's principal output directions, so this measures
how much of B's column energy falls in the top-256 left singular subspace of
the base qkv weight, reading B both ways. The wrong order scatters rows
across heads and lands at the random baseline (256/21504 = 1.19%); the right
one sits clearly above it in early and middle blocks (the signal fades in
the last ten or so, hence the default block set). Checked on Singularity's
own delta against the official base: 2.2% in the right order, 1.3% in the
wrong one.

    lora_qkv_layout.py LORA.safetensors [NATIVE_TRANSFORMER_DIR] [--blocks 10,30]
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_lora_merge import header, read, find  # noqa: E402

def to_native(B):
    return B.reshape(3, 56, 128, -1).transpose(1, 0, 2, 3).reshape(B.shape)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lora")
    ap.add_argument("native", nargs="?", default=os.path.expanduser(
        "~/h3.c/MiniMax-H3-turbo-int8/FL2VA/transformer"))
    ap.add_argument("--blocks", default="2,5,10,20")
    a = ap.parse_args()
    lh, lb = header(a.lora)
    base = 256 / 21504
    votes = []
    for i in (int(x) for x in a.blocks.split(",")):
        keys = [k for k in lh if f"blocks.{i}.attn.qkv_proj.lora_B" in k and "token_refiner" not in k]
        if not keys:
            continue
        name = f"blocks.{i}.attn.qkv_proj.weight"
        p, h, b = find(a.native, name)
        W = read(p, h, b, name).astype(np.float32)
        if h[name]["dtype"] == "I8":
            ps, hs, bs = find(a.native, name + "_scale")
            W *= read(ps, hs, bs, name + "_scale").reshape(-1, 1)
        _, V = np.linalg.eigh((W.T @ W).astype(np.float64))
        U = W @ V[:, -256:].astype(np.float32)
        U /= np.linalg.norm(U, axis=0, keepdims=True)
        B = read(a.lora, lh, lb, keys[0]).astype(np.float32)
        energy = lambda X: float(np.sum((U.T @ X) ** 2) / np.sum(X ** 2))
        as_native, as_comfy = energy(B), energy(to_native(B))
        votes.append("comfy" if as_comfy > as_native else "native")
        print(f"block {i}: read as native {as_native*100:.2f}%  read as comfy {as_comfy*100:.2f}%"
              f"  (random {base*100:.2f}%)")
    if not votes:
        sys.exit("no blocks.N.attn.qkv_proj LoRA in that file")
    comfy = votes.count("comfy")
    order = "comfy" if comfy * 4 >= len(votes) * 3 else \
            "native" if comfy * 4 <= len(votes) else "unclear"
    print(f"order: {order} ({comfy}/{len(votes)} blocks say comfy) ->",
          {"comfy": "the default, nothing to add", "native": "load it as NAME@native",
           "unclear": "check a render both ways"}[order])

if __name__ == "__main__":
    main()
