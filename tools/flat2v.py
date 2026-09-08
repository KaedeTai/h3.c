#!/usr/bin/env python3
"""FLAT2V - first frame, last frame, audio, to video.

Ordinary H3 generates the soundtrack: --ref-audio only clones a voice and the
words come out of the text prompt, which is why lip sync used to be a lottery.
This drives h3 in a mode where the audio is an INPUT. The reference clip is
replaced into the target audio rows at every sampler step, so the finished
soundtrack IS the file you handed in and the picture is generated to match it.

    flat2v.py WORK init  --poster P.png --voice V.wav [--line "..."]
    flat2v.py WORK crops --first x,y,w,h [--last x,y,w,h]
    flat2v.py WORK run   [--seeds 42] [--base]
    flat2v.py WORK check
"""
import argparse, json, math, os, re, subprocess, sys, shutil, tempfile
import numpy as np

H3_DIR = os.path.expanduser("~/h3.c")
H3 = os.path.join(H3_DIR, "h3")
FPS, AUDIO_LATENT_FPS = 24, 40


def align_frames(n):
    """h3 reads video in chunks, and only 5+17k frames are legal.

    h3_video_latent_t is ((f-5)/17)*5+2: a head chunk of 2 latent frames
    spanning 5 real ones (1+4), then chunks of 5 spanning 17 (1+4+4+4+4),
    matching h3_frame_per_token = {1,4,4,4,4}. Ask for anything else and h3
    rounds UP to the next chunk boundary, at which point the soundtrack you
    fitted no longer matches the target length and the run dies on
    "reference encodes to 0 latent frames but the target needs N".

    Rounding down here rather than up, so the fitted voice is never longer
    than the clip. 158, 175, 192, ... 311, 328.
    """
    n = max(22, int(n))
    return 5 + max(1, (n - 5) // 17) * 17


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

def generation_env(st):
    """Every knob the recipe turns on, and why each one is here."""
    env = dict(os.environ)
    env["H3_DIT_VARIANT"] = "FL2VA"      # the checkpoint that carries an anchor
    env["H3_AUDIO_ANCHOR"] = "1"         # the soundtrack is the input
    env["H3_REF2VA_ANCHOR"] = "1"        # reference 1 takes frame 0's time
    if len(st["references"]) > 1:
        env["H3_REF2VA_ANCHOR_LAST"] = "2"
    # int8 VAE is a large, unconditional win here: the video decode goes 54.1s ->
    # 13.3s at 480x640, a 4.1x cut, not the 22% an earlier note claimed at a
    # different resolution.
    if st.get("vae_int8", True): env.setdefault("H3_VAE_INT8_FFN", "1")
    # Steel attention is NOT a win at this size -- measured 68.4s -> 73.6s of
    # denoise at 480x640, and still 8% slower at 576x768 (432 tokens a frame).
    # It pays at 1152x640 (858). Opt in explicitly.
    if st.get("steel", False): env.setdefault("H3_STEEL_ATTN", "1")
    return env


def run_seed(st, seed, turbo, knobs):
    tag = f"take_s{seed}" + ("" if turbo else "_base")
    out = os.path.join(st["workdir"], tag + ".mp4")
    if os.path.exists(out):
        print(f"  seed {seed}: already generated"); return out
    env = generation_env(st)
    cmd = [H3, "-d", st["turbo_dir"] if turbo else st["model_dir"],
           "-p", st["prompt"]]
    for r in st["references"]:
        cmd += ["--ref-image", os.path.abspath(r)]
    cmd += ["--ref-audio", os.path.abspath(st["voice_fitted"]),
            "--width", str(st["width"]), "--height", str(st["height"]),
            "--frames", str(st["frames"]),
            "--steps", str(st["steps"]),
            "--seed", str(seed), "--use-int8-row-fc2"]
    if knobs: cmd += knobs.split()
    cmd += ["-o", os.path.abspath(out)]
    print(f"  seed {seed}: generating ...", flush=True)
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
          "steel": a.steel, "vae_int8": not a.no_vae_int8,
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
    if st["width"] * st["height"] < 384 * 512:
        raise SystemExit(
            f"{st['width']}x{st['height']} is below the canvas floor. "
            "352x448 froze the mouth at one seed of three, 320x416 at one of "
            "two, 288x384 at both -- and 288x384 froze on the wide framing too, "
            "so it is the canvas and not the crop. Use 384x512 or larger.")
    st["references"] = refs
    save(a.workdir, st)
    print(f"{len(refs)} reference(s): "
          + ("frame 0 only" if len(refs) == 1 else "frame 0 and the last frame"))


# --core-reuse is deliberately NOT in the recipe.
#
# It looked like the one accelerator worth having on the wide crop: 93 s -> 73 s
# for about 10% of the mouth movement. On the head crop it is much worse, and a
# listener caught it before the metric threshold did -- 4 steps + --core-reuse 4
# came back at aperture 0.0685 and face detail 0.788 against 0.0926 and 0.982
# for plain 3 steps, and was called poor. At 3 steps it still costs: 0.0762 and
# 0.0843 across two seeds.
#
# The mechanism fits the framing. Core reuse holds one DiT core across steps, so
# what it damages is whatever changes fastest -- and on a tight crop the mouth is
# 57% of the frame instead of 35%, so proportionally much more of the picture is
# the fast part. Saving 7 s of 51 is not worth it.
def run_resident(st, seeds, knobs):
    """Generate every seed inside one h3 process.

    h3 caches the prepared DiT, the text embedding, the video decoder and the
    conditioning -- but only in interactive mode: h3_cache_set_enabled(ctx, 1)
    is called in exactly one place, h3_cli_run. Every one-shot `-p` invocation
    therefore reloads ~10 s of weights and re-encodes 3 s of references that a
    seed sweep is about to use unchanged. Measured on the 3-step recipe:
    51.0 s for the first take, then 39.4 and 39.4.

    (This needed a fix in h3 itself. The audio anchor is the raw [32,2,T]
    encoder output, not the patchified REF_AUDIO condition rows, and it was not
    in the conditioning cache -- so a cache hit had nothing to write into the
    target audio rows and the run refused outright. It is cached now.)
    """
    import shlex
    out_dir = os.path.join(st["workdir"], "_resident")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    env = generation_env(st)
    cmd = [H3, "-d", st["turbo_dir"] if st.get("turbo", True) else st["model_dir"]]
    for r in st["references"]:
        cmd += ["--ref-image", os.path.abspath(r)]
    cmd += ["--ref-audio", os.path.abspath(st["voice_fitted"]),
            "--width", str(st["width"]), "--height", str(st["height"]),
            "--frames", str(st["frames"]), "--steps", str(st["steps"]),
            "--seed", str(seeds[0]), "--use-int8-row-fc2"]
    if knobs:
        cmd += shlex.split(knobs)
    script = [f"!output {out_dir}"]
    for i, seed in enumerate(seeds):
        if i:
            script.append(f"!seed {seed}")
        script.append(st["prompt"])
    script += ["!quit", ""]
    print(f"  {len(seeds)} seeds in one resident process ...", flush=True)
    r = subprocess.run(cmd, cwd=H3_DIR, env=env, input="\n".join(script),
                       capture_output=True, text=True)
    produced = re.findall(r"Done -> (\S+) \[([0-9.]+)s\]", r.stdout + r.stderr)
    if len(produced) != len(seeds):
        print((r.stderr or "")[-1500:])
        raise SystemExit(f"h3 produced {len(produced)} takes for {len(seeds)} seeds")
    outs = []
    for (path, secs), seed in zip(produced, seeds):
        dst = os.path.join(st["workdir"], f"take_s{seed}.mp4")
        shutil.move(path, dst)
        print(f"  seed {seed}: {float(secs):5.1f}s -> {os.path.basename(dst)}")
        outs.append(dst)
    shutil.rmtree(out_dir, ignore_errors=True)
    return outs


def cmd_run(a):
    st = load(a.workdir)
    if not st["references"]: raise SystemExit("run `crops` first")
    for bad in ("--layers", "--token-reduction"):
        if bad in a.knobs:
            raise SystemExit(
                f"{bad} is refused. --layers buys no measurable time at this "
                "size (l50 93 s, l45 88-101 s, l35 74-84 s -- a 4-step run is "
                "dominated by model load, text encoding and VAE decode) and it "
                "froze the mouth outright in 8 of 16 cells. --token-reduction "
                "damages the anchors themselves: frame 0 4.25 -> 5.70, tail "
                "4.02 -> 6.92.")
    if "--core-reuse" in a.knobs:
        print("  warning: --core-reuse holds one DiT core across steps, so it "
              "damps whatever moves fastest. On a head crop that is the head "
              "itself: measured head motion 1.07-2.00 against 3.78-4.28 "
              "without it, and a viewer called the result unnatural. It is off "
              "by default for that reason.", file=sys.stderr)
    st["turbo"] = not a.base
    seeds = [int(x) for x in a.seeds.split(",")]
    if len(seeds) > 1 and not a.no_resident:
        run_resident(st, seeds, a.knobs)
    else:
        for s in seeds:
            run_seed(st, s, not a.base, a.knobs)
    save(a.workdir, st)
    print(f"next: {sys.argv[0]} {a.workdir} check")


def cmd_check(a):
    """Grade every take, with the articulation check first.

    The other three numbers all pass a still photograph: its anchors are
    perfect because nothing moved, its scene is stable for the same reason,
    and its soundtrack is exact because the soundtrack is an input. Eight
    takes got through a whole night of grading that way. Aperture -- inner-lip
    gap over mouth width, s.d. over the clip -- is the only number here that
    can tell a talking head from a photograph of one.
    """
    st = load(a.workdir)
    ref0 = st["references"][0]
    sys.path.insert(0, os.path.join(H3_DIR, "tools"))
    try:
        from take_report import report, verdict
        rich = True
    except Exception as e:
        print(f"  (take_report unavailable: {e}; falling back to audio+frame0)")
        rich = False
    last = st["frames"] - 1
    if rich:
        print(f"{'take':26s} {'audio r':>8s} {'frame0':>7s} {'last':>7s} "
              f"{'apert':>7s} {'head':>6s}   verdict")
    else:
        print(f"{'take':26s} {'audio r':>9s} {'frame0':>8s}   verdict")
    for fn in sorted(os.listdir(a.workdir)):
        if not fn.startswith("take_") or not fn.endswith(".mp4"): continue
        p = os.path.join(a.workdir, fn)
        if rich:
            os.environ["FLAT2V_REF"] = os.path.abspath(ref0)
            os.environ["FLAT2V_WAV"] = os.path.abspath(st["voice_fitted"])
            m = report(p, last=last)
            print(f"{fn:26s} {m['audio_r']:8.3f} {m['anchor0']:7.2f} "
                  f"{m['anchor_last']:7.2f} {m['aperture']:7.4f} "
                  f"{m['head']:6.2f}   {verdict(m)}")
        else:
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
        "(tight head-and-shoulders framing, softly blurred interior behind, "
        "locked-off camera, no zoom, no camera movement, soft even indoor "
        "light, documentary realism)"),
        help="English scene description -- another language than the spoken "
             "one, so it conditions the picture without being read aloud")
    i.add_argument("--frames", type=int, help="default: from the voice length")
    # 384x512 head-and-shoulders, not 480x640 wide. Same face 31% larger
    # (292 px against 222), 36% fewer tokens, 62 s against 93 s, four seeds for
    # four. Do not go lower: 352x448 froze at one seed of three, 320x416 at one
    # of two, 288x384 at both -- and 288x384 froze on the WIDE framing too, so
    # the floor is the canvas, not the crop. h3_adapt_canvas normalises to a
    # 768 short edge; 288 is far outside what the checkpoint saw.
    i.add_argument("--width", type=int, default=384)
    i.add_argument("--height", type=int, default=512)
    # 3, not 4 and not 20. The step sweep is flat in articulation from 2 to 5
    # (aperture 0.091-0.104) because the soundtrack is an input and both end
    # frames are pinned, so most of what extra steps decide is already decided.
    # 2 steps does cost picture detail -- face-region high-frequency energy
    # 0.797 against 0.983 at 4 steps, outside the seed spread, and visible as a
    # waxier skin. 3 steps sits at 0.955, inside it, for 51 s against 62.
    i.add_argument("--steps", type=int, default=3)
    i.add_argument("--model-dir", default="./MiniMax-H3")
    # Both bundles now carry the int4 g32 text encoder in place -- the swap was
    # made inside <bundle>/<variant>/text_encoder rather than by changing this
    # path, so nothing here or in any other caller had to know. 48 GB -> 15 GB
    # on disk, 47 GiB -> 14 GiB read per run, output 3.28 mean |pixel| from the
    # BF16 take against a 2.57 run-to-run floor. Peak memory is unchanged; the
    # encoder already streamed layer by layer at 3.73 GiB.
    #   revert:  cd ~/h3.c && for B in MiniMax-H3 MiniMax-H3-turbo-int8; do
    #              for V in FL2VA Ref2VA; do
    #                rm -rf $B/$V/text_encoder
    #                cp -al text_encoder_slim $B/$V/text_encoder
    #              done; done
    i.add_argument("--turbo-dir", default="./MiniMax-H3-turbo-int8")
    i.add_argument("--steel", action="store_true",
                   help="enable H3_STEEL_ATTN; measured slower below ~600 "
                        "tokens a frame, worth trying at 1152x640 and up")
    i.add_argument("--no-vae-int8", action="store_true",
                   help="disable H3_VAE_INT8_FFN (int8 video VAE FFN)")

    c = sub.add_parser("crops"); c.set_defaults(fn=cmd_crops)
    c.add_argument("--first", required=True, help="x,y,w,h at poster resolution")
    c.add_argument("--last", help="x,y,w,h; the same box pins both ends")

    r = sub.add_parser("run"); r.set_defaults(fn=cmd_run)
    r.add_argument("--seeds", default="42")
    # Turbo is the default because the audio anchor removed its only cost.
    # 4 steps against 20: 93 s against 417 s, aperture 0.1124 against 0.1114 --
    # i.e. 4.5x for no measurable articulation, because everything turbo used
    # to wreck was in the audio branch and the audio branch is now an input.
    r.add_argument("--no-resident", action="store_true",
                   help="one process per seed instead of one for all of them; "
                        "costs ~12 s a take in reloaded weights")
    r.add_argument("--base", action="store_true",
                   help="use the un-distilled checkpoint at --steps instead of "
                        "the 4-step turbo one (4.5x slower, no measured gain)")
    r.add_argument("--knobs", default="",
                   help="extra h3 flags. Nothing here is recommended any more: "
                        "--core-reuse looked worth it on the wide crop and is "
                        "not on a tight one (see the note in the source). "
                        "--layers and --token-reduction are refused.")

    sub.add_parser("check").set_defaults(fn=cmd_check)
    a = p.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
