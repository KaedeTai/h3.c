#!/usr/bin/env python3
"""Check h3_lora.c's int8 merge against a NumPy reference.

    check_lora_merge.py SHARD_DIR WEIGHT_NAME DUMP_PREFIX LORA:STRENGTH ...

DUMP_PREFIX.q / .s come from tests/test_lora_merge.c. The reference does the
same arithmetic in float64 -- dequantize, add sum(s * alpha/r * B @ A),
requantize per row -- so the two should agree up to rounding ties.
"""
import json, os, struct, sys
import numpy as np

def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n

def read(path, h, base, name):
    info = h[name]; a, b = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a); raw = f.read(b - a)
    dt = info["dtype"]
    if dt == "BF16":
        x = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif dt == "F16": x = np.frombuffer(raw, np.float16).astype(np.float32)
    elif dt == "F32": x = np.frombuffer(raw, np.float32)
    elif dt == "I8": x = np.frombuffer(raw, np.int8)
    else: raise SystemExit(dt)
    return x.reshape(info["shape"])

def find(shards, name):
    for fn in sorted(os.listdir(shards)):
        if not fn.endswith(".safetensors"): continue
        p = os.path.join(shards, fn); h, base = header(p)
        if name in h: return p, h, base
    raise SystemExit("missing " + name)

def main():
    shards, name, prefix = sys.argv[1:4]
    p, h, base = find(shards, name)
    q = read(p, h, base, name).astype(np.float64)
    s = read(p, h, base, name + "_scale").astype(np.float64).reshape(-1)
    w = q * s[:, None]
    module = name[: -len(".weight")]
    for spec in sys.argv[4:]:
        path, _, strength = spec.rpartition(":")
        lh, lbase = header(path)
        for pre in ("diffusion_model.", ""):
            for a_suf, b_suf in ((".lora_A.weight", ".lora_B.weight"), (".lora_A.default.weight", ".lora_B.default.weight")):
                if pre + module + a_suf in lh:
                    A = read(path, lh, lbase, pre + module + a_suf).astype(np.float64)
                    B = read(path, lh, lbase, pre + module + b_suf).astype(np.float64)
                    alpha = float(read(path, lh, lbase, pre + module + ".alpha")) if pre + module + ".alpha" in lh else A.shape[0]
                    w = w + float(strength) * alpha / A.shape[0] * (B @ A)
    mx = np.abs(w).max(axis=1)
    ref_s = np.where(mx > 0, mx / 127, 1 / 127)
    ref_q = np.clip(np.rint(w / ref_s[:, None]), -127, 127)
    got_q = np.fromfile(prefix + ".q", np.int8).reshape(q.shape).astype(np.float64)
    got_s = np.fromfile(prefix + ".s", np.float32).astype(np.float64)
    dq = np.abs(got_q - ref_q)
    print(f"{name}: scale rel err max {np.max(np.abs(got_s - ref_s) / ref_s):.2e}; "
          f"int8 codes differing {np.mean(dq > 0):.2e} (max diff {dq.max():.0f})")
    merged = got_q * got_s[:, None]
    delta = w - q * s[:, None]
    print(f"  |delta|/|W| {np.linalg.norm(delta) / np.linalg.norm(w):.4f}; "
          f"merged-vs-exact rel err {np.linalg.norm(merged - w) / np.linalg.norm(w):.2e}; "
          f"requant noise / |delta| {np.linalg.norm(merged - w) / max(np.linalg.norm(delta), 1e-30):.3f}")


if __name__ == "__main__":
    main()
