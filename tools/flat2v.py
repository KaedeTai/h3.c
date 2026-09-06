#!/usr/bin/env python3
"""FLAT2V - first frame, last frame, audio, to video.

Ordinary H3 generates the soundtrack: --ref-audio only clones a voice and the
words come out of the text prompt, which is why lip sync used to be a lottery.
This drives h3 in a mode where the audio is an INPUT. The reference clip is
replaced into the target audio rows at every sampler step, so the finished
soundtrack IS the file you handed in and the picture is generated to match it.

    flat2v.py WORK init  --poster P.png --voice V.wav [--line "..."]
    flat2v.py WORK crops --first x,y,w,h [--last x,y,w,h]
    flat2v.py WORK run   [--seeds 42] [--turbo] [--knobs "--core-reuse 4"]
    flat2v.py WORK check
"""
import argparse, json, math, os, subprocess, sys, shutil, tempfile
import numpy as np

H3_DIR = os.path.expanduser("~/h3.c")
H3 = os.path.join(H3_DIR, "h3")
FPS, AUDIO_LATENT_FPS = 24, 40


def align_frames(n):
    """h3 needs 4k+2 frames and at least one 22-frame decoder chunk."""
    n = max(22, int(n))
    return n - ((n - 2) % 4)


def audio_latent_frames(frames):
    return int(round(frames * AUDIO_LATENT_FPS / FPS))


def required_seconds(frames):
    """The exact clip length whose encode lands on the target's audio_t.

    h3 refuses a mismatch rather than misaligning, and this is the sharp edge
    everyone hits first: 158 video frames wants 263 audio latent frames, which
    is 6.575 s, not the 6.58 s you get from 158/24.
    """
    return audio_latent_frames(frames) / AUDIO_LATENT_FPS


def probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def state_path(w): return os.path.join(w, "flat2v.json")
def load(w):
    with open(state_path(w)) as f: return json.load(f)
def save(w, s):
    with open(state_path(w), "w") as f: json.dump(s, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------- crops

def make_crop(poster, box, width, height, out):
    """Crop at native resolution, then scale uniformly.

    h3 reads every conditioning image with H3_IMAGE_FIT_STRETCH, so a box whose
    aspect differs from the canvas is squashed and the model reproduces the
    squash for the whole clip. Refuse rather than resize.
    """
    from PIL import Image
    x, y, w, h = box
    if abs(w / h - width / height) > 1e-6:
        raise SystemExit(f"box {w}x{h} is {w/h:.4f} but the canvas is "
                         f"{width/height:.4f}; move an edge, do not resize")
    src = Image.open(poster).convert("RGB")
    if x < 0 or y < 0 or x + w > src.width or y + h > src.height:
        raise SystemExit(f"box {box} falls outside {src.width}x{src.height}")
    src.crop((x, y, x + w, y + h)).resize((width, height), Image.LANCZOS).save(out)
    print(f"  {out}  from {w}x{h} at ({x},{y})")


# ---------------------------------------------------------------- audio

def fit_voice(src, dst, frames):
    """Pad or trim the clip to exactly what this frame count needs."""
    want = required_seconds(frames)
    have = probe_duration(src)
    if have > want + 1e-3:
        print(f"  trimming {have:.3f}s -> {want:.3f}s (speech may be cut)")
        filt = f"atrim=0:{want}"
    else:
        print(f"  padding  {have:.3f}s -> {want:.3f}s")
        filt = f"apad=whole_dur={want}"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-af", filt,
                    "-ar", "24000", "-ac", "1", dst], check=True)
    return want


# ---------------------------------------------------------------- run

def run_seed(st, seed, turbo, knobs):
    tag = f"take_s{seed}" + ("_turbo" if turbo else "")
    out = os.path.join(st["workdir"], tag + ".mp4")
    if os.path.exists(out):
        print(f"  seed {seed}: already generated"); return out
    env = dict(os.environ)
    env["H3_DIT_VARIANT"] = "FL2VA"      # the checkpoint that carries an anchor
    env["H3_AUDIO_ANCHOR"] = "1"         # the soundtrack is the input
    env["H3_REF2VA_ANCHOR"] = "1"        # reference 1 takes frame 0's time
    if len(st["references"]) > 1:
        env["H3_REF2VA_ANCHOR_LAST"] = "2"
    # M5 paths that are opt-in rather than auto-detected. Tensor ops, the DiT
    # command-block split and the Qwen prefetch depth already switch on when
    # h3_gpu_is_m5() sees the device; these two do not, and attention is around
    # 40% of the denoise at this size.
    if st.get("steel", True): env.setdefault("H3_STEEL_ATTN", "1")
    if st.get("vae_int8", True): env.setdefault("H3_VAE_INT8_FFN", "1")
    cmd = [H3, "-d", st["turbo_dir"] if turbo else st["model_dir"],
           "-p", st["prompt"]]
    for r in st["references"]:
        cmd += ["--ref-image", os.path.abspath(r)]
    cmd += ["--ref-audio", os.path.abspath(st["voice_fitted"]),
            "--width", str(st["width"]), "--height", str(st["height"]),
            "--frames", str(st["frames"]),
            "--steps", str(4 if turbo else st["steps"]),
            "--seed", str(seed), "--use-int8-row-fc2"]
    if knobs: cmd += knobs.split()
    cmd += ["-o", os.path.abspath(out)]
    print(f"  seed {seed}{' turbo' if turbo else ''}: generating ...", flush=True)
    r = subprocess.run(cmd, cwd=H3_DIR, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:]); raise SystemExit(f"h3 failed on seed {seed}")
    return out


def clean_audio(src, dst):
    """The latent is [32,2,T] and the channels denoise from independent noise,
    so the L-R difference is pure artifact. The reference is mono anyway."""
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-c:v", "copy",
                    "-af", "pan=mono|c0=0.5*c0+0.5*c1,highpass=f=60",
                    "-c:a", "aac", "-b:a", "128k", "-ac", "1", "-ar", "32000",
                    dst], check=True)


# ---------------------------------------------------------------- checks

def audio_matches(mp4, reference):
    """The whole point: did the soundtrack come back as the file we gave it?"""
    import wave
    def load(p):
        w = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p, "-vn", "-ac", "1",
                        "-ar", "24000", "-c:a", "pcm_s16le", w], check=True)
        with wave.open(w) as f:
            a = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        os.unlink(w)
        return a.astype(float) / 32768
    a, b = load(mp4), load(reference)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def frame0_matches(mp4, reference):
    from PIL import Image
    p = tempfile.mktemp(suffix=".png")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mp4, "-vf",
                    "select=eq(n\\,0)", "-vframes", "1", p], check=True)
    a = np.asarray(Image.open(p).convert("RGB")).astype(float)
    b = np.asarray(Image.open(reference).convert("RGB")).astype(float)
    os.unlink(p)
    return float(np.abs(a - b).mean())


# ---------------------------------------------------------------- commands

def cmd_init(a):
    os.makedirs(a.workdir, exist_ok=True)
    frames = align_frames(a.frames if a.frames else
                          round(probe_duration(a.voice) * FPS))
    print(f"initialised {a.workdir}")
    print(f"  {frames} frames = {frames/FPS:.3f}s of video")
    print(f"  voice must be exactly {required_seconds(frames):.3f}s "
          f"({audio_latent_frames(frames)} audio latent frames)")
    fitted = os.path.join(a.workdir, "voice_fitted.wav")
    fit_voice(a.voice, fitted, frames)
    prompt = (a.line + " " if a.line else "") + a.direction
    st = {"workdir": os.path.abspath(a.workdir),
          "poster": os.path.abspath(a.poster),
          "voice": os.path.abspath(a.voice), "voice_fitted": fitted,
          "line": a.line or "", "direction": a.direction, "prompt": prompt,
          "width": a.width, "height": a.height, "frames": frames,
          "steps": a.steps,
          "model_dir": a.model_dir, "turbo_dir": a.turbo_dir,
          "steel": not a.no_steel, "vae_int8": not a.no_vae_int8,
          "references": [], "results": {}}
    save(a.workdir, st)
    print(f"  next: crops --first x,y,w,h   (aspect must be "
          f"{a.width/a.height:.4f})")


def cmd_crops(a):
    st = load(a.workdir)
    refs = []
    for name, spec in (("first", a.first), ("last", a.last)):
        if not spec: continue
        box = [int(v) for v in spec.split(",")]
        out = os.path.join(a.workdir, f"ref_{name}.png")
        make_crop(st["poster"], box, st["width"], st["height"], out)
        refs.append(out)
    if not refs: raise SystemExit("--first is required")
    st["references"] = refs
    save(a.workdir, st)
    print(f"{len(refs)} reference(s): "
          + ("frame 0 only" if len(refs) == 1 else "frame 0 and the last frame"))


def cmd_run(a):
    st = load(a.workdir)
    if not st["references"]: raise SystemExit("run `crops` first")
    for s in [int(x) for x in a.seeds.split(",")]:
        run_seed(st, s, a.turbo, a.knobs)
    save(a.workdir, st)
    print(f"next: {sys.argv[0]} {a.workdir} check")


def cmd_check(a):
    st = load(a.workdir)
    ref0 = st["references"][0]
    print(f"{'take':26s} {'audio r':>9s} {'frame0':>8s}   verdict")
    for fn in sorted(os.listdir(a.workdir)):
        if not fn.startswith("take_") or not fn.endswith(".mp4"): continue
        p = os.path.join(a.workdir, fn)
        ar = audio_matches(p, st["voice_fitted"])
        f0 = frame0_matches(p, ref0)
        ok = ar > 0.9 and f0 < 12
        print(f"{fn:26s} {ar:9.3f} {f0:8.2f}   {'ok' if ok else 'CHECK'}")
        out = os.path.join(a.workdir, fn.replace("take_", "final_"))
        clean_audio(p, out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workdir")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init"); i.set_defaults(fn=cmd_init)
    i.add_argument("--poster", required=True)
    i.add_argument("--voice", required=True, help="the actual soundtrack")
    i.add_argument("--line", help="what is said; still helps expression")
    i.add_argument("--direction", default=(
        "(locked-off camera, no zoom, no camera movement, soft even indoor "
        "light, documentary realism)"),
        help="English scene description -- another language than the spoken "
             "one, so it conditions the picture without being read aloud")
    i.add_argument("--frames", type=int, help="default: from the voice length")
    i.add_argument("--width", type=int, default=480)
    i.add_argument("--height", type=int, default=640)
    i.add_argument("--steps", type=int, default=20)
    i.add_argument("--model-dir", default="./MiniMax-H3")
    i.add_argument("--turbo-dir", default="./MiniMax-H3-turbo-int8")
    i.add_argument("--no-steel", action="store_true",
                   help="disable H3_STEEL_ATTN (M5 steel attention)")
    i.add_argument("--no-vae-int8", action="store_true",
                   help="disable H3_VAE_INT8_FFN (int8 video VAE FFN)")

    c = sub.add_parser("crops"); c.set_defaults(fn=cmd_crops)
    c.add_argument("--first", required=True, help="x,y,w,h at poster resolution")
    c.add_argument("--last", help="x,y,w,h; the same box pins both ends")

    r = sub.add_parser("run"); r.set_defaults(fn=cmd_run)
    r.add_argument("--seeds", default="42")
    r.add_argument("--turbo", action="store_true")
    r.add_argument("--knobs", default="")

    sub.add_parser("check").set_defaults(fn=cmd_check)
    a = p.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
