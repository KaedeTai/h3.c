"""Fuse the turbo LoRA by patching tensor bytes in place, not by re-serialising.

Rewriting the shards with mx.save_safetensors produced a checkpoint that h3.c
loads without complaint and then generates pure black frames from -- and it does
that even with the LoRA strength set to 0, i.e. with numerically identical
weights. So the container, not the arithmetic, is what h3.c is sensitive to:
its hand-written Metal loader depends on something about the original file
layout (tensor order, padding, alignment) that a fresh serialisation does not
reproduce.

This copies the original shards byte-for-byte and then overwrites only the
bytes of the 208 tensors the LoRA touches. Same dtype, same shape, same offsets,
same header -- the only thing that changes is the numbers.
"""
import json, os, shutil, struct, sys, time
import mlx.core as mx
import numpy as np

VARIANT = os.environ.get("H3_VARIANT", "FL2VA")   # FL2VA or Ref2VA
SRC = os.path.expanduser(f"~/h3.c/MiniMax-H3/{VARIANT}/transformer")
ROOT_SRC = os.path.expanduser("~/h3.c/MiniMax-H3")
DST_ROOT = os.path.expanduser(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv \
           else os.path.expanduser("~/h3.c/MiniMax-H3-turbo")
LORA = os.environ.get("H3_LORA", "/tmp/h3turbo/minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors")
STRENGTH = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 1.0
DST = os.path.join(DST_ROOT, VARIANT, "transformer")

L = mx.load(LORA)
mods = {k[len("diffusion_model."):-len(".lora_A.weight")]
        for k in L if k.endswith(".lora_A.weight")}
print(f"strength {STRENGTH}, lora covers {len(mods)} modules", flush=True)

os.makedirs(DST, exist_ok=True)
t0 = time.time(); patched = 0; rel = []
shards = sorted(f for f in os.listdir(SRC) if f.endswith(".safetensors"))
for si, sh in enumerate(shards):
    s, d = os.path.join(SRC, sh), os.path.join(DST, sh)
    if not os.path.exists(d):
        shutil.copyfile(s, d)
    with open(d, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    base_off = 8 + hlen
    W = mx.load(s)                      # read originals from the pristine copy
    with open(d, "r+b") as f:
        for k, info in hdr.items():
            if k == "__metadata__" or not k.endswith(".weight"):
                continue
            base = k[: -len(".weight")]
            if base not in mods:
                continue
            lk = "diffusion_model." + base
            A = L[lk + ".lora_A.weight"].astype(mx.float32)
            B = L[lk + ".lora_B.weight"].astype(mx.float32)
            scale = float(L[lk + ".alpha"]) / A.shape[0] * STRENGTH
            w = W[k]
            dlt = (B @ A) * scale
            if tuple(dlt.shape) != tuple(w.shape):
                raise SystemExit(f"{base}: {dlt.shape} != {tuple(w.shape)}")
            rel.append(float(mx.linalg.norm(dlt) / mx.maximum(mx.linalg.norm(w.astype(mx.float32)), 1e-6)))
            nw = (w.astype(mx.float32) + dlt).astype(mx.bfloat16)
            mx.eval(nw)
            raw = np.array(nw.view(mx.uint16), copy=False).tobytes()
            a, b = info["data_offsets"]
            if len(raw) != b - a:
                raise SystemExit(f"{base}: {len(raw)} bytes for a {b-a}-byte slot")
            f.seek(base_off + a); f.write(raw)
            patched += 1
            del A, B, dlt, nw, raw
        f.flush(); os.fsync(f.fileno())
    del W; mx.clear_cache()
    print(f"  [{si+1}/{len(shards)}] {sh}  ({time.time()-t0:.0f}s)", flush=True)

for f in os.listdir(SRC):
    if not f.endswith(".safetensors"):
        try: os.link(os.path.join(SRC, f), os.path.join(DST, f))
        except OSError: pass
for sub in os.listdir(os.path.join(ROOT_SRC, VARIANT)):
    if sub == "transformer": continue
    s = os.path.join(ROOT_SRC, VARIANT, sub); dd = os.path.join(DST_ROOT, VARIANT, sub)
    if os.path.isdir(s):
        for root, _, files in os.walk(s):
            rd = root.replace(s, dd, 1); os.makedirs(rd, exist_ok=True)
            for f in files:
                try: os.link(os.path.join(root, f), os.path.join(rd, f))
                except OSError: pass
    else:
        try: os.link(s, dd)
        except OSError: pass

rel.sort()
print(f"patched {patched} tensors; |dW|/|W| median {rel[len(rel)//2]:.4f}", flush=True)
print("wrote", DST_ROOT, f"in {time.time()-t0:.0f}s", flush=True)
