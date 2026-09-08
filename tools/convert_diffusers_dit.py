#!/usr/bin/env python3
"""Convert a diffusers-layout H3 DiT into the native layout h3.c reads.

MiniMax ship the same transformer twice. FL2VA and Ref2VA use MiniMax's own
naming -- fused `attn.qkv_proj`, `mlp.fc1`, `blocks.N` -- and that is what
h3.c's loader asks for. The base text-to-audio-video transformer, and every
FastVideo checkpoint distilled from it, use the diffusers port instead:
separate `to_q`/`to_k`/`to_v`, `ff.net.0.proj`, `transformer_blocks.N`. Same
architecture, same shapes, different spelling.

Two things are not just spelling.

The loader is strict about dtype: it asks for F32 for the patch projections,
the time embedder and the final output heads, and BF16 for everything else.
The diffusers checkpoints store all of it BF16, so those tensors have to be
upcast on the way through. The dtype table is read from a real native bundle
rather than hardcoded, so it stays right if MiniMax change their minds.

And `rope.inv_freq` is a stored tensor in the native layout and a computed
constant in the diffusers one. It is copied from the reference bundle -- the
architecture is identical, so the frequencies are too.

VSA checkpoints carry one extra projection per block, `attn.to_gate_compress`,
which only FastVideo's sparse-attention kernel consumes. It is dropped, with a
count, because a dense implementation has nowhere to put it -- and that is
worth knowing about rather than silently ignoring.

    convert_diffusers_dit.py SOURCE_DIR OUT_DIR [--reference NATIVE_DIR]
"""
import argparse, json, os, re, sys

SINGLETON = {
    "proj_in.weight": "video_patch_proj.weight",
    "proj_in.bias": "video_patch_proj.bias",
    "audio_proj_in.weight": "audio_patch_proj.weight",
    "audio_proj_in.bias": "audio_patch_proj.bias",
    "context_embedder.weight": "condition_proj.weight",
    "context_embedder.bias": "condition_proj.bias",
    "time_embedder.linear_1.weight": "time_embedder.proj_in.weight",
    "time_embedder.linear_1.bias": "time_embedder.proj_in.bias",
    "time_embedder.linear_2.weight": "time_embedder.proj_out.weight",
    "time_embedder.linear_2.bias": "time_embedder.proj_out.bias",
    "norm_out.norm.weight": "final_layer.norm.weight",
    "norm_out.linear.weight": "final_layer.adaln_proj.linear.weight",
    "norm_out.linear.bias": "final_layer.adaln_proj.linear.bias",
    "proj_out.weight": "final_layer.video_out.weight",
    "proj_out.bias": "final_layer.video_out.bias",
    "audio_proj_out.weight": "final_layer.audio_out.weight",
    "audio_proj_out.bias": "final_layer.audio_out.bias",
    "token_refiner.final_norm.weight": "token_refiner.final_norm.weight",
}

# inside a block, once the block prefix has been rewritten
WITHIN = {
    "norm1.weight": "norm1.weight",
    "norm2.weight": "norm2.weight",
    "adaln_proj.linear.weight": "adaln_proj.linear.weight",
    "adaln_proj.linear.bias": "adaln_proj.linear.bias",
    "attn.norm_q.weight": "attn.q_norm.weight",
    "attn.norm_k.weight": "attn.k_norm.weight",
    "attn.to_out.0.weight": "attn.out_proj.weight",
    "ff.net.2.weight": "mlp.fc2.weight",
}
FUSED = ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight")
DROP = ("attn.to_gate_compress.weight",)

HEADS, HEAD_DIM, FFN = 56, 128, 14336


def swap_swiglu(torch, w):
    """diffusers stores the SwiGLU halves of fc1 in the opposite order.

    `ff.net.0.proj` is [28672, 5376] and so is `mlp.fc1`, and the top half of
    one is the bottom half of the other. Verified byte-exact both ways against
    MiniMax's own two spellings, at blocks 0 and 49 and in the token refiner.
    Left unswapped, gate and up trade places in every SwiGLU in the network.
    """
    return torch.cat([w[FFN:], w[:FFN]], dim=0)


def fuse_qkv(torch, q, k, v):
    """MiniMax's fused qkv_proj is PER-HEAD interleaved, not q|k|v stacked.

    Row block h*384 .. h*384+384 holds head h's q, then its k, then its v. A
    plain cat([q,k,v]) has the right shape and the wrong meaning, and the
    result is not obviously broken -- residual paths keep producing a
    structured picture, so the video still looks like a scene while the audio
    comes out as loud broadband noise. Verified byte-exact against MiniMax's
    own two spellings of the same checkpoint (transformer_ref against
    Ref2VA/transformer) at heads 0, 1, 27 and 55.
    """
    parts = []
    for head in range(HEADS):
        lo, hi = head * HEAD_DIM, (head + 1) * HEAD_DIM
        parts.extend((q[lo:hi], k[lo:hi], v[lo:hi]))
    return torch.cat(parts, dim=0)


BLOCK = re.compile(r"^transformer_blocks\.(\d+)\.(.+)$")
REFINER = re.compile(r"^token_refiner\.refiner_blocks\.(\d+)\.(.+)$")


def native_prefix(name):
    """(block prefix in native naming, remainder) or (None, name)."""
    m = BLOCK.match(name)
    if m:
        return f"blocks.{m.group(1)}.", m.group(2)
    m = REFINER.match(name)
    if m:
        return f"token_refiner.blocks.{m.group(1)}.", m.group(2)
    return None, name


SIDECAR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "native_layout", "h3_dit_native.json")


def native_layout(reference):
    """(dtype table, rope.inv_freq) -- from a real bundle, or from the sidecar.

    The loader is strict about dtype and asks for F32 for thirteen tensors --
    the patch projections, the time embedder, the output heads -- where the
    diffusers checkpoints store everything BF16. And rope.inv_freq is a stored
    tensor natively and a computed constant in the diffusers port. That is the
    entire dependency on a native bundle, so it is kept as a file instead.
    """
    if reference:
        from safetensors import safe_open
        index = os.path.join(reference, "model.safetensors.index.json")
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        table, opened = {}, {}
        for name, shard in weight_map.items():
            if shard not in opened:
                opened[shard] = safe_open(os.path.join(reference, shard), "pt")
            table[name] = opened[shard].get_slice(name).get_dtype()
        rope = safe_open(os.path.join(reference, weight_map["rope.inv_freq"]),
                         "pt").get_tensor("rope.inv_freq")
        return table, rope
    with open(SIDECAR) as f:
        data = json.load(f)
    import torch
    return data["dtypes"], torch.tensor(data["rope_inv_freq"],
                                        dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="diffusers transformer/ directory")
    ap.add_argument("out", help="native transformer/ directory to write")
    ap.add_argument("--reference", default=None,
        help="a native bundle to read dtypes and rope.inv_freq from; by "
             "default they come from native_layout/h3_dit_native.json")
    ap.add_argument("--shard-bytes", type=int, default=4_500_000_000)
    ap.add_argument("--dry-run", action="store_true",
                    help="map names only; touch no tensor data")
    a = ap.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    DRY = a.dry_run

    if not DRY:
        os.makedirs(a.out, exist_ok=True)
    want, rope = native_layout(a.reference)
    print(f"native layout declares {len(want)} tensors "
          f"({'bundle: ' + a.reference if a.reference else 'from the sidecar'})")

    src_index = os.path.join(
        a.source, "diffusion_pytorch_model.safetensors.index.json")
    with open(src_index) as f:
        source_map = json.load(f)["weight_map"]

    # group by block so the three QKV matrices meet
    handles = {}
    def get(name):
        if DRY:
            return torch.zeros(1, dtype=torch.bfloat16)
        shard = source_map[name]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(a.source, shard), "pt")
        return handles[shard].get_tensor(name)

    produced, dropped, shard_index = {}, 0, 1
    written, weight_map, pending, pending_bytes = [], {}, {}, 0
    fused_done = set()

    def flush():
        nonlocal pending, pending_bytes, shard_index
        if not pending or DRY:
            pending.clear()
            return
        name = f"model-{shard_index:05d}.safetensors"
        save_file(pending, os.path.join(a.out, name))
        for key in pending:
            weight_map[key] = name
        print(f"  wrote {name}  {len(pending)} tensors  "
              f"{pending_bytes/1e9:.2f} GB", flush=True)
        written.append(name)
        pending, pending_bytes, shard_index = {}, 0, shard_index + 1

    def emit(native_name, tensor):
        nonlocal pending_bytes
        target = want.get(native_name)
        if target == "F32" and tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        elif target == "BF16" and tensor.dtype != torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        elif target is None:
            # not in the reference bundle: keep whatever the source had
            pass
        tensor = tensor.contiguous()
        pending[native_name] = tensor
        pending_bytes += tensor.numel() * tensor.element_size()
        if DRY:
            pending.clear()
        produced[native_name] = True
        if pending_bytes >= a.shard_bytes:
            flush()

    for name in source_map:
        prefix, rest = native_prefix(name)
        if prefix is None:
            if name in SINGLETON:
                emit(SINGLETON[name], get(name))
            else:
                print(f"  ! unmapped top-level tensor: {name}")
            continue
        if rest in DROP:
            dropped += 1
            continue
        if rest == "ff.net.0.proj.weight":
            emit(prefix + "mlp.fc1.weight",
                 get(name) if DRY else swap_swiglu(torch, get(name)))
            continue
        if rest in WITHIN:
            emit(prefix + WITHIN[rest], get(name))
            continue
        if rest in FUSED:
            if prefix in fused_done:
                continue
            fused_done.add(prefix)
            source_prefix = name[:-len(rest)]      # the diffusers spelling
            q, k, v = (get(source_prefix + part) for part in FUSED)
            emit(prefix + "attn.qkv_proj.weight", fuse_qkv(torch, q, k, v))
            continue
        print(f"  ! unmapped tensor: {name}")

    # rope.inv_freq is computed in the diffusers port and stored in the native
    # one; the architecture is identical, so it is carried across verbatim.
    emit("rope.inv_freq", rope)
    flush()

    if DRY:
        missing = sorted(set(want) - set(produced))
        extra = sorted(set(produced) - set(want))
        print(f"\nDRY RUN: {len(produced)} native names would be written")
        print(f"dropped {dropped} VSA-only tensors")
        print(f"reference names NOT produced: {len(missing)}")
        for m in missing[:12]: print(f"   {m}")
        print(f"produced but absent from the reference: {len(extra)}")
        for e in extra[:12]: print(f"   {e}")
        return
    with open(os.path.join(a.out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": sum(
            os.path.getsize(os.path.join(a.out, s)) for s in written)},
            "weight_map": weight_map}, f, indent=1)
    src_config = os.path.join(a.source, "config.json")
    if os.path.exists(src_config):
        with open(src_config) as f, \
             open(os.path.join(a.out, "config.json"), "w") as g:
            g.write(f.read())

    missing = sorted(set(want) - set(produced))
    extra = sorted(set(produced) - set(want))
    print(f"\nproduced {len(produced)} tensors in {len(written)} shards")
    if dropped:
        print(f"dropped {dropped} VSA-only attn.to_gate_compress tensors "
              f"-- this checkpoint expects a sparse-attention kernel")
    print(f"reference tensors NOT produced: {len(missing)}")
    for m in missing[:12]:
        print(f"   {m}")
    print(f"produced but absent from the reference: {len(extra)}")
    for e in extra[:12]:
        print(f"   {e}")


if __name__ == "__main__":
    main()
