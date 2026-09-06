#!/usr/bin/env python3
"""Talking-head shots with MiniMax-H3 Ref2VA, as an SOP.

Everything here is a rule we paid for in a wrong take. The short version:

  * H3 generates the soundtrack; --ref-audio only clones the voice timbre. The
    prompt IS the script, so the line goes in the prompt and nothing else in the
    spoken language does.
  * Text in a language other than the spoken one conditions the picture without
    reaching the audio branch. That is where wardrobe, framing and lighting go.
  * Positive description, never negation. "no subtitles" left the subtitle rate
    unchanged; describing the shot took it to zero.
  * References must be CROPPED to the canvas aspect, never resized to it - a
    497x589 crop squeezed into 384x512 narrowed the face 11% and Ref2VA
    reproduced the squeeze for the whole clip.
  * Two references: a tight face crop for identity, a wide one carrying the
    wardrobe and set. One reference pins the face and nothing else.
  * --core-reuse 4 is fine and 2.3x faster, but only against an h3 that
    re-denoises the audio rows on reuse steps (h3_dit.c, core_reuse_audio_pass).
    --layers below 50 and --token-reduction are NOT fine: the first doubles the
    speech over itself, the second halves horizontal resolution.
  * Output is stereo from a [32,2,T] latent whose channels denoise from
    independent noise; the difference is pure artifact. Downmix to mono.

Even with all of that, wardrobe/speech/subtitle outcomes are per-seed, so the
tool generates several seeds, scores each one, and ranks them.

    talking_head.py WORKDIR init    --poster P.png --voice V.wav --line "..."
    talking_head.py WORKDIR crops   --face x,y,w,h --wide x,y,w,h
    talking_head.py WORKDIR run     [--seeds 1,3,7,11,23,42]
    talking_head.py WORKDIR score
    talking_head.py WORKDIR pick    [--out final.mp4]
    talking_head.py WORKDIR status
"""
import argparse, json, os, subprocess, sys, tempfile, shutil

H3_DIR = os.path.expanduser("~/h3.c")
H3 = os.path.join(H3_DIR, "h3")
MODEL = "./MiniMax-H3"          # base, not turbo: the turbo Ref2VA LoRA is v0.1
WHISPER = os.path.expanduser("~/.venv/bin/mlx_whisper")
WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"

DEFAULT_DIRECTION = (
    " (locked-off camera, no zoom, no camera movement, soft even indoor light,"
    " documentary realism, shallow natural depth of field)")


def state_path(w): return os.path.join(w, "talking_head.json")


def load(w):
    with open(state_path(w)) as f: return json.load(f)


def save(w, s):
    with open(state_path(w), "w") as f: json.dump(s, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------- crops

def make_crop(poster, box, width, height, out):
    """Crop at native resolution, then scale uniformly.

    Asserts the box already has the canvas aspect, because the whole point is
    that no axis is stretched. If this raises, move the box edges - do not
    "fix" it by resizing.
    """
    from PIL import Image
    x, y, w, h = box
    if abs(w / h - width / height) > 1e-6:
        raise SystemExit(
            f"box {w}x{h} is {w/h:.4f}, canvas {width}x{height} is {width/height:.4f}.\n"
            f"Widen or heighten the box so they match; never resize to fit.")
    src = Image.open(poster).convert("RGB")
    if x < 0 or y < 0 or x + w > src.width or y + h > src.height:
        raise SystemExit(f"box {box} falls outside the {src.width}x{src.height} poster")
    src.crop((x, y, x + w, y + h)).resize((width, height), Image.LANCZOS).save(out)
    print(f"  {out}  from {w}x{h} at ({x},{y}), uniform scale {w/width:.4f}")


# ---------------------------------------------------------------- scoring

def _frames(mp4, every):
    d = tempfile.mkdtemp()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mp4, "-vf",
                    f"select=not(mod(n\\,{every}))", "-vsync", "0", f"{d}/f_%04d.png"],
                   check=True)
    return d


def score_wardrobe(mp4, every=20):
    """Fraction of the shoulder blocks that is bright and desaturated.

    The coat sits either side of the tie in the lower third; sampling there and
    skipping the centre strip avoids the shirt and tie entirely. Measured
    separation is unambiguous: no take has ever landed between 2.4% and 54%.
    """
    import numpy as np
    from PIL import Image
    d = _frames(mp4, every); vals = []
    for fn in sorted(os.listdir(d)):
        a = np.asarray(Image.open(f"{d}/{fn}").convert("RGB")).astype(float)
        h, w, _ = a.shape
        band = a[int(h * 0.72):int(h * 0.92)]
        px = np.concatenate([band[:, int(w*0.06):int(w*0.30)].reshape(-1, 3),
                             band[:, int(w*0.70):int(w*0.94)].reshape(-1, 3)])
        vals.append(float((((px.mean(1) > 170) & (px.max(1) - px.min(1) < 42))).mean()))
    shutil.rmtree(d, ignore_errors=True)
    return float(np.mean(vals)) * 100


def score_subtitle(mp4, every=4):
    """Burned-in subtitle coverage.

    Both styles seen (saturated yellow, and white) share a bright glyph core
    against a dark outline; the white coat is bright with no rim, which is what
    separates them. The row-span test then drops collar edges: a subtitle line
    crosses the frame, a collar does not.
    """
    import numpy as np
    from PIL import Image
    from scipy.ndimage import binary_dilation
    d = _frames(mp4, every); vals = []
    for fn in sorted(os.listdir(d)):
        a = np.asarray(Image.open(f"{d}/{fn}").convert("RGB")).astype(int)
        H, W, _ = a.shape
        R, G, B = a[..., 0], a[..., 1], a[..., 2]
        lum = a.mean(2)
        dark = binary_dilation(lum < 95, iterations=3)
        g = (((R > 210) & (G > 180) & (B < 150) & ((G - B) > 70)) |
             ((lum > 215) & (a.max(2) - a.min(2) < 32) & dark))
        keep = np.zeros(H, bool); keep[int(H*0.55):] = True; keep[:int(H*0.18)] = True
        g[~keep] = False
        g[g.sum(1) < W * 0.18] = False
        vals.append(g.mean())
    shutil.rmtree(d, ignore_errors=True)
    return float(np.mean(vals)) * 100


def transcribe(mp4):
    if not os.path.exists(WHISPER):
        return None
    wav = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mp4, "-vn", "-ac", "1",
                    "-ar", "32000", "-c:a", "pcm_s16le", wav], check=True)
    out = tempfile.mkdtemp()
    r = subprocess.run([WHISPER, "--model", WHISPER_MODEL, "--task", "transcribe",
                        "--output-dir", out, "--output-format", "txt", wav],
                       capture_output=True, text=True)
    txts = [f for f in os.listdir(out) if f.endswith(".txt")]
    if not txts: return None
    with open(os.path.join(out, txts[0])) as f:
        return " ".join(f.read().split())


def line_accuracy(said, line):
    """Longest common subsequence against the intended line, as a percentage.

    A bag-of-characters overlap scored a take whose speech was pure noise at
    100%, because Mandarin reuses so few characters that coincidental hits fill
    the bag. LCS respects order, so a garbled take cannot borrow credit from
    characters that merely appear somewhere.
    """
    if not said: return None
    strip = "，。、！？「」()（）,. \t\n"
    want = [c for c in line if c.strip() and c not in strip]
    got = [c for c in said if c.strip() and c not in strip]
    if not want: return None
    prev = [0] * (len(got) + 1)
    for a in want:
        cur = [0]
        for j, b in enumerate(got):
            cur.append(prev[j] + 1 if a == b else max(cur[j], prev[j + 1]))
        prev = cur
    return 100.0 * prev[-1] / len(want)


# ---------------------------------------------------------------- run

def run_seed(st, seed):
    out = os.path.join(st["workdir"], f"take_s{seed}.mp4")
    if os.path.exists(out):
        print(f"  seed {seed}: already generated"); return out
    prompt = st["line"] + st["direction"]
    cmd = [H3, "-d", MODEL, "-p", prompt]
    for r in st["references"]:
        cmd += ["--ref-image", os.path.abspath(r)]
    cmd += ["--ref-audio", os.path.abspath(st["voice"]),
            "--width", str(st["width"]), "--height", str(st["height"]),
            "--frames", str(st["frames"]), "--steps", str(st["steps"]),
            "--seed", str(seed), "--use-int8-row-fc2"]
    if st["core_reuse"] > 1:
        cmd += ["--core-reuse", str(st["core_reuse"])]
    # --layers stays at h3's default 50 and --token-reduction is never passed:
    # both skip work for the audio rows too and double the speech over itself.
    cmd += ["-o", os.path.abspath(out)]
    print(f"  seed {seed}: generating ...", flush=True)
    r = subprocess.run(cmd, cwd=H3_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:]); raise SystemExit(f"h3 failed on seed {seed}")
    return out


def clean_audio(src, dst):
    """Downmix to mono and gate the residual hiss. See clean_h3_audio.sh."""
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-c:v", "copy",
                    "-af", "pan=mono|c0=0.5*c0+0.5*c1,highpass=f=60,"
                           "afftdn=nr=12:nf=-38:tn=1",
                    "-c:a", "aac", "-b:a", "128k", "-ac", "1", "-ar", "32000", dst],
                   check=True)


# ---------------------------------------------------------------- commands

def cmd_init(a):
    os.makedirs(a.workdir, exist_ok=True)
    st = {"workdir": os.path.abspath(a.workdir),
          "poster": os.path.abspath(a.poster), "voice": os.path.abspath(a.voice),
          "line": a.line, "direction": a.direction,
          "width": a.width, "height": a.height, "frames": a.frames,
          "steps": a.steps, "core_reuse": a.core_reuse,
          "references": [], "scores": {}}
    save(a.workdir, st)
    print(f"initialised {a.workdir}")
    print(f"  canvas {a.width}x{a.height}, aspect {a.width/a.height:.4f}")
    print(f"  next: {sys.argv[0]} {a.workdir} crops --face x,y,w,h --wide x,y,w,h")
    print(f"  both boxes must have aspect {a.width/a.height:.4f}")


def cmd_crops(a):
    st = load(a.workdir)
    refs = []
    for name, spec in (("face", a.face), ("wide", a.wide)):
        box = [int(v) for v in spec.split(",")]
        out = os.path.join(a.workdir, f"ref_{name}.png")
        make_crop(st["poster"], box, st["width"], st["height"], out)
        refs.append(out)
    st["references"] = refs
    save(a.workdir, st)
    print("references ready: tight face for identity, wide for wardrobe and set")


def cmd_run(a):
    st = load(a.workdir)
    if len(st["references"]) < 2:
        raise SystemExit("run `crops` first - one reference pins the face and nothing else")
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"generating {len(seeds)} takes")
    for s in seeds:
        run_seed(st, s)
    st["seeds"] = seeds
    save(a.workdir, st)
    print(f"done. next: {sys.argv[0]} {a.workdir} score")


def cmd_score(a):
    st = load(a.workdir)
    scores = {}
    print(f"{'seed':>5s} {'wardrobe':>9s} {'subtitle':>9s} {'line':>7s}")
    for s in st.get("seeds", []):
        mp4 = os.path.join(a.workdir, f"take_s{s}.mp4")
        if not os.path.exists(mp4): continue
        w = score_wardrobe(mp4)
        sub = score_subtitle(mp4)
        acc = line_accuracy(transcribe(mp4), st["line"])
        scores[str(s)] = {"wardrobe": w, "subtitle": sub, "line": acc}
        print(f"{s:5d} {w:8.1f}% {sub:8.3f}% "
              + (f"{acc:6.1f}%" if acc is not None else "     --"))
    st["scores"] = scores
    save(a.workdir, st)


def rank(scores):
    def key(kv):
        _, v = kv
        return (-(v["wardrobe"] >= 50), -(v["line"] or 0), v["subtitle"], -v["wardrobe"])
    return sorted(scores.items(), key=key)


def cmd_pick(a):
    st = load(a.workdir)
    if not st.get("scores"): raise SystemExit("run `score` first")
    order = rank(st["scores"])
    print("ranked (wardrobe pass, then line accuracy, then least subtitle):")
    for seed, v in order:
        print(f"  seed {seed:>3s}  wardrobe {v['wardrobe']:5.1f}%  "
              f"subtitle {v['subtitle']:.3f}%  line "
              + (f"{v['line']:.1f}%" if v["line"] is not None else "--"))
    best = order[0][0]
    out = a.out or os.path.join(a.workdir, "final.mp4")
    clean_audio(os.path.join(a.workdir, f"take_s{best}.mp4"), out)
    print(f"\npicked seed {best} -> {out}")
    if order[0][1]["subtitle"] > 0.05:
        print("  NOTE: this take still carries a subtitle. Re-run more seeds, or "
              "crop the band as a last resort.")


def cmd_status(a):
    st = load(a.workdir)
    print(f"workdir   {st['workdir']}")
    print(f"canvas    {st['width']}x{st['height']}  frames {st['frames']}  steps {st['steps']}")
    print(f"line      {st['line']}")
    print(f"direction {st['direction'].strip()[:100]}...")
    print(f"refs      {len(st['references'])}")
    for s in st.get("seeds", []):
        f = os.path.join(st["workdir"], f"take_s{s}.mp4")
        print(f"  seed {s:>3d}  {'generated' if os.path.exists(f) else 'missing'}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workdir")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init"); i.set_defaults(fn=cmd_init)
    i.add_argument("--poster", required=True)
    i.add_argument("--voice", required=True, help="5-10 s clip; only the timbre is used")
    i.add_argument("--line", required=True, help="what he says - this IS the prompt")
    i.add_argument("--direction", default=DEFAULT_DIRECTION,
                   help="English scene description; never in the spoken language, "
                        "and never phrased as a negation")
    i.add_argument("--width", type=int, default=480)
    i.add_argument("--height", type=int, default=640)
    i.add_argument("--frames", type=int, default=158)
    i.add_argument("--steps", type=int, default=20)
    i.add_argument("--core-reuse", type=int, default=4)

    c = sub.add_parser("crops"); c.set_defaults(fn=cmd_crops)
    c.add_argument("--face", required=True, help="x,y,w,h at poster resolution")
    c.add_argument("--wide", required=True, help="x,y,w,h - must show the wardrobe")

    r = sub.add_parser("run"); r.set_defaults(fn=cmd_run)
    r.add_argument("--seeds", default="1,3,7,11,23,42")

    sub.add_parser("score").set_defaults(fn=cmd_score)

    k = sub.add_parser("pick"); k.set_defaults(fn=cmd_pick)
    k.add_argument("--out")

    sub.add_parser("status").set_defaults(fn=cmd_status)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
