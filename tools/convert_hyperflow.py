#!/usr/bin/env python3
"""Convert a HyperFlow H3 Flash diffusers checkpoint (int8_convrot) into an h3.c transformer directory.

HyperFlow (VideoRebirthAI/hyperflow-h3-flash) is MiniMax-H3 compressed four ways; h3.c reads it as follows:

  * dropped blocks        -> no blocks.N.* tensors; h3_dit marks them inactive
  * low-rank linears      -> W = up @ down, stored like every other block matrix
                             (--keep-lowrank instead writes <name>.down/.up in BF16)
  * INT8 convrot linears  -> W = (qweight * wscale) @ blockdiag(H_256)  (H symmetric, orthonormal)
  * block matrices        -> per-row int8 + F32 weight_scale, the exact rule of tools/quantize_h3_int8.py
  * two-time conditioning -> endpoint_time_embedder.* (F32) next to time_embedder.*; AdaLN input is 16-wide
  * layout                -> q/k/v fused per head (head, q|k|v, 128); SwiGLU fc1 (up|gate) -> (gate|up)

Every layout rule here was checked against the base H3 checkpoint and the official diffusers block
(~/repos/hyperflow-h3-mlx, tests/). The directory also gets hyperflow.json, which h3 uses to pick the
4/6/8-step HyperFlow sigma grid.

  tools/convert_hyperflow.py --src ~/models/hyperflow-h3-flash/fl2va_int8_convrot --dst OUT/FL2VA/transformer
"""
import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np

H, D, FFN = 56, 128, 14336


def regular_hadamard(group):
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float64) / 2
    h, n = np.ones((1, 1)), 1
    while n < group:
        h, n = np.kron(h, h4), n * 4
    assert n == group and np.allclose(h, h.T) and np.allclose(h @ h.T, np.eye(group))
    return mx.array(h.astype(np.float32))


class Src:
    def __init__(self, d):
        self.dir = d
        self.cfg = json.loads((d / "config.json").read_text())
        self.map = json.loads((d / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
        self._file, self._data = None, None

    def get(self, k):
        f = self.map[k]
        if f != self._file:
            self._file, self._data = f, mx.load(str(self.dir / f))
        return self._data[k]

    def has(self, k):
        return k in self.map


ALIGN = 16
DTYPES = {mx.float32: "F32", mx.bfloat16: "BF16", mx.int8: "I8"}


def write_safetensors(path, tensors):
    """safetensors with the header padded so the data section, and every payload in it, starts on a 16-byte
    boundary. mx.save_safetensors leaves the header at its natural length, which put every payload at 2 mod 4:
    h3 maps transformer payloads straight into Metal buffers on M5, and F32 read from there came out garbage."""
    entries, blobs, offset = {}, [], 0
    for name, a in tensors.items():
        if a.dtype not in DTYPES:
            raise TypeError(f"{name}: unsupported dtype {a.dtype}")
        raw = np.asarray(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a).tobytes()
        pad = (-offset) % ALIGN
        if pad:
            blobs.append(b"\0" * pad)
            offset += pad
        entries[name] = {"dtype": DTYPES[a.dtype], "shape": list(a.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    header = json.dumps({"__metadata__": {"format": "pt"}, **entries}, separators=(",", ":")).encode()
    header += b" " * ((-(8 + len(header))) % ALIGN)
    with open(path, "wb") as fh:
        fh.write(len(header).to_bytes(8, "little"))
        fh.write(header)
        for blob in blobs:
            fh.write(blob)


def quantize_rows(w32):
    """tools/quantize_h3_int8.py, on the GPU: scale = absmax/127, q = clamp(rint(w * 127/absmax))."""
    max_abs = mx.max(mx.abs(w32), axis=1)
    pos = max_abs > 0
    scale = mx.where(pos, max_abs / 127.0, mx.array(1.0 / 127.0))
    inverse = mx.where(pos, 127.0 / mx.where(pos, max_abs, 1.0), mx.array(127.0))
    q = mx.clip(mx.round(w32 * inverse[:, None]), -127, 127).astype(mx.int8)
    return q, scale.astype(mx.float32)


def interleave_qkv(q, k, v):
    inn = q.shape[-1]
    return mx.stack([q.reshape(H, D, inn), k.reshape(H, D, inn), v.reshape(H, D, inn)], axis=1).reshape(3 * H * D, inn)


def swap_halves(w):
    return mx.concatenate([w[FFN:], w[:FFN]], axis=0)


def convert(src_dir, dst_dir, shard_gb=4.0, keep_lowrank=False):
    src = Src(src_dir)
    cfg = src.cfg
    if (cfg.get("quantization") or {}).get("method", "int8_convrot") != "int8_convrot":
        raise SystemExit("only int8_convrot HyperFlow checkpoints are supported")
    group = int((cfg.get("quantization") or {}).get("group", 256))
    rot = regular_hadamard(group)
    dropped = set(cfg.get("dropped_blocks") or [])
    dst_dir.mkdir(parents=True, exist_ok=True)

    out, index, state = {}, {}, {"shard": 0, "bytes": 0}

    def flush():
        if not out:
            return
        fn = f"model-{state['shard'] + 1:05d}.safetensors"
        write_safetensors(dst_dir / fn, out)
        for k in out:
            index[k] = fn
        print(f"  wrote {fn} ({state['bytes'] / 1e9:.2f} GB)", flush=True)
        out.clear()
        state["shard"] += 1
        state["bytes"] = 0

    def put(k, a):
        mx.eval(a)
        out[k] = a
        state["bytes"] += a.nbytes
        if state["bytes"] >= shard_gb * 1e9:
            flush()

    def copy(dk, mk):
        put(mk, src.get(dk))

    def matrix(name):
        """('dense', W f32) or ('lowrank', down, up)."""
        if src.has(name + ".qweight"):
            q = src.get(name + ".qweight").astype(mx.float32)
            s = src.get(name + ".wscale").astype(mx.float32)
            o, i = q.shape
            return ("dense", ((q * s[:, None]).reshape(o, i // group, group) @ rot).reshape(o, i))
        if src.has(name + ".down.weight"):
            return ("lowrank", src.get(name + ".down.weight"), src.get(name + ".up.weight"))
        return ("dense", src.get(name + ".weight").astype(mx.float32))

    def dense(p):
        return p[1] if p[0] == "dense" else p[2].astype(mx.float32) @ p[1].astype(mx.float32)

    def put_int8(mk, w32):
        q, s = quantize_rows(w32)
        put(mk + ".weight", q)
        put(mk + ".weight_scale", s)

    for dk, mk in [("proj_in", "video_patch_proj"), ("audio_proj_in", "audio_patch_proj"),
                   ("context_embedder", "condition_proj"),
                   ("proj_out", "final_layer.video_out"), ("audio_proj_out", "final_layer.audio_out"),
                   ("norm_out.linear", "final_layer.adaln_proj.linear")]:
        copy(dk + ".weight", mk + ".weight")
        copy(dk + ".bias", mk + ".bias")
    copy("norm_out.norm.weight", "final_layer.norm.weight")
    for te in ("time_embedder", "endpoint_time_embedder"):
        for dl, ml in (("linear_1", "proj_in"), ("linear_2", "proj_out")):
            copy(f"{te}.{dl}.weight", f"{te}.{ml}.weight")
            copy(f"{te}.{dl}.bias", f"{te}.{ml}.bias")
    put("endpoint_time_embedder.gate", mx.array([float(cfg["hyperflow_gate"])], dtype=mx.float32))
    theta, fd = float(cfg.get("rope_theta", 10000.0)), int(cfg.get("rope_freq_dim", 16))
    put("rope.inv_freq", mx.array((1.0 / theta ** (np.arange(0, 2 * fd, 2, dtype=np.float32) / (2 * fd))).astype(np.float32)))

    for i in range(cfg["num_refiner_layers"]):
        d, m = f"token_refiner.refiner_blocks.{i}.", f"token_refiner.blocks.{i}."
        copy(d + "norm1.weight", m + "norm1.weight")
        copy(d + "norm2.weight", m + "norm2.weight")
        copy(d + "attn.norm_q.weight", m + "attn.q_norm.weight")
        copy(d + "attn.norm_k.weight", m + "attn.k_norm.weight")
        put(m + "attn.qkv_proj.weight", interleave_qkv(*(src.get(d + f"attn.to_{x}.weight") for x in "qkv")))
        copy(d + "attn.to_out.0.weight", m + "attn.out_proj.weight")
        put(m + "mlp.fc1.weight", swap_halves(src.get(d + "ff.net.0.proj.weight")))
        copy(d + "ff.net.2.weight", m + "mlp.fc2.weight")
    copy("token_refiner.final_norm.weight", "token_refiner.final_norm.weight")

    layout = {}
    for i in range(cfg["num_layers"]):
        if i in dropped:
            layout[i] = "dropped"
            print(f"block {i:2d}: dropped", flush=True)
            continue
        d, m = f"transformer_blocks.{i}.", f"blocks.{i}."
        copy(d + "norm1.weight", m + "norm1.weight")
        copy(d + "norm2.weight", m + "norm2.weight")
        copy(d + "attn.norm_q.weight", m + "attn.q_norm.weight")
        copy(d + "attn.norm_k.weight", m + "attn.k_norm.weight")
        copy(d + "adaln_proj.linear.weight", m + "adaln_proj.linear.weight")
        copy(d + "adaln_proj.linear.bias", m + "adaln_proj.linear.bias")
        kinds = {}
        qkv = [matrix(d + f"attn.to_{x}") for x in "qkv"]
        kinds["qkv"] = qkv[0][0] if len({p[0] for p in qkv}) == 1 else "mixed"
        if keep_lowrank and kinds["qkv"] == "lowrank":
            put(m + "attn.qkv_proj.down.weight", mx.concatenate([p[1] for p in qkv], axis=0).astype(mx.bfloat16))
            for x, p in zip("qkv", qkv):
                put(m + f"attn.{x}_proj.up.weight", p[2].astype(mx.bfloat16))
        else:
            put_int8(m + "attn.qkv_proj", interleave_qkv(*(dense(p) for p in qkv)))
        for dn, mn in (("attn.to_out.0", "attn.out_proj"), ("ff.net.0.proj", "mlp.fc1"), ("ff.net.2", "mlp.fc2")):
            p = matrix(d + dn)
            kinds[mn] = p[0]
            if keep_lowrank and p[0] == "lowrank":
                up = swap_halves(p[2]) if mn == "mlp.fc1" else p[2]
                put(m + mn + ".down.weight", p[1].astype(mx.bfloat16))
                put(m + mn + ".up.weight", up.astype(mx.bfloat16))
            else:
                w = dense(p)
                put_int8(m + mn, swap_halves(w) if mn == "mlp.fc1" else w)
        layout[i] = kinds
        print(f"block {i:2d}: " + " ".join(f"{k}={v}" for k, v in kinds.items()), flush=True)
    flush()

    total = sum((dst_dir / f).stat().st_size for f in set(index.values()))
    (dst_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(index.items()))}, indent=1))
    (dst_dir / "config.json").write_text(json.dumps({
        "_class_name": "MiniMaxH3DiTModel", "hidden_size": cfg["hidden_size"], "num_layers": cfg["num_layers"],
        "token_refiner_num_layers": cfg["num_refiner_layers"], "num_attention_heads": cfg["num_attention_heads"],
        "attention_head_dim": cfg["attention_head_dim"], "ffn_hidden_size": cfg["ffn_dim"],
        "latents_dim": cfg["in_channels"], "audio_latents_dim": cfg["audio_in_channels"],
        "patch_size": cfg["patch_size"], "text_dim": cfg["text_dim"], "timestep_input_dim": cfg["freq_dim"],
        "time_embed_hidden_size": cfg["time_embed_hidden_dim"], "time_embed_dim": cfg["time_embed_dim"],
        "rope_inv_freq_len": cfg["rope_freq_dim"], "norm_eps": cfg["norm_eps"],
        "qk_norm_eps": cfg["qk_norm_eps"], "final_norm_eps": cfg["final_norm_eps"]}, indent=2))
    (dst_dir / "hyperflow.json").write_text(json.dumps({
        "hyperflow_gate": cfg["hyperflow_gate"], "dropped_blocks": sorted(dropped),
        "sigma_grids": {"4": [1.0, 0.681872, 0.205019, 0.036496],
                        "6": [1.0, 0.825217, 0.471725, 0.147401, 0.049626, 0.015517],
                        "8": [1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494]},
        "keep_lowrank": keep_lowrank, "layout": {str(k): v for k, v in layout.items()},
        "source": str(src_dir)}, indent=1))
    for extra in ("LICENSE", "NOTICE"):
        p = src_dir.parent / extra
        if p.exists():
            shutil.copy(p, dst_dir / extra)
    print("done:", dst_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--shard-gb", type=float, default=4.0)
    ap.add_argument("--keep-lowrank", action="store_true", help="write low-rank factors (BF16) instead of int8 W")
    a = ap.parse_args()
    convert(Path(a.src).expanduser(), Path(a.dst).expanduser(), a.shard_gb, a.keep_lowrank)
