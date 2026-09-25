#!/usr/bin/env python3
"""How much of a LoRA survives being merged into an already-int8 weight?

Ground truth needs the BF16 weight, which the int8 bundle no longer has, so
this takes one BF16 matrix from the official checkpoint and plays out both
histories:

  truth   quantize(W + D)                 -- merge first, as patch_h3_turbo did
  nearest requant_rint(deq(quantize(W)) + D)
  h3.c    tests/test_lora_merge.c on quantize(W) (stochastic unless
          H3_LORA_ROUND=nearest)

and scores them on random activations: how much of the LoRA's own output
X @ D^T each keeps ("kept", 1.0 = all of it) and the error left against the
unquantized X @ (W + D)^T, in units of |X @ D^T|.

    eval_lora_int8_merge.py BF16_SHARD WEIGHT_NAME LORA[:STRENGTH] ...
"""
import json, os, struct, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from check_lora_merge import header, read

shard, name = sys.argv[1:3]
specs = sys.argv[3:]
h, base = header(shard)
W = read(shard, h, base, name).astype(np.float32)
module = name[: -len(".weight")]
D = np.zeros_like(W)
for spec in specs:
    path, _, strength = spec.rpartition(":")
    lh, lbase = header(path)
    for pre in ("diffusion_model.", ""):
        for a_suf, b_suf in ((".lora_A.weight", ".lora_B.weight"),
                             (".lora_A.default.weight", ".lora_B.default.weight")):
            if pre + module + a_suf in lh:
                A = read(path, lh, lbase, pre + module + a_suf).astype(np.float32)
                B = read(path, lh, lbase, pre + module + b_suf).astype(np.float32)
                alpha = (float(read(path, lh, lbase, pre + module + ".alpha"))
                         if pre + module + ".alpha" in lh else A.shape[0])
                D += float(strength) * alpha / A.shape[0] * (B @ A)

def quantize(w):
    mx = np.abs(w).max(axis=1)
    s = np.where(mx > 0, mx / 127, 1 / 127).astype(np.float32)
    return np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8), s

q0, s0 = quantize(W)
qt, st = quantize(W + D)
qn, sn = quantize(q0.astype(np.float32) * s0[:, None] + D)

with tempfile.TemporaryDirectory() as tmp:
    blobs = [(name, "I8", list(q0.shape), q0.tobytes()),
             (name + "_scale", "F32", [len(s0)], s0.astype(np.float32).tobytes())]
    hdr, off = {}, 0
    for n, dt, shp, raw in blobs:
        hdr[n] = {"dtype": dt, "shape": shp, "data_offsets": [off, off + len(raw)]}
        off += len(raw)
    js = json.dumps(hdr).encode(); js += b" " * (-len(js) % 8)
    with open(os.path.join(tmp, "w.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(js))); f.write(js)
        for *_, raw in blobs: f.write(raw)
    exe = os.environ.get("TEST_LORA_MERGE", "/tmp/test_lora_merge")
    subprocess.run([exe, tmp, name, ",".join(specs), os.path.join(tmp, "out")],
                   check=True, stderr=subprocess.DEVNULL)
    qc = np.fromfile(os.path.join(tmp, "out.q"), np.int8).reshape(q0.shape)
    sc = np.fromfile(os.path.join(tmp, "out.s"), np.float32)

rng = np.random.default_rng(0)
X = rng.standard_normal((256, W.shape[1]), dtype=np.float32)
YD = X @ D.T
Yx = X @ (W + D).T
Y0 = X @ (q0.astype(np.float32) * s0[:, None]).T
nD = float(np.linalg.norm(YD))
print(f"{name}  |D|/|W| = {np.linalg.norm(D) / np.linalg.norm(W):.4f}")
for label, q, s in (("truth", qt, st), ("nearest", qn, sn), ("h3.c", qc, sc)):
    Y = X @ (q.astype(np.float32) * s[:, None]).T
    kept = float(np.sum((Y - Y0) * YD)) / nD ** 2
    err = float(np.linalg.norm(Y - Yx)) / nD
    print(f"  {label:8s} kept {kept:6.3f}   error {err:6.3f} x|XD|")
