#!/usr/bin/env python3
"""Segmented FL2VA: a long shot as N short first/last-frame segments.

Why: DiT attention is quadratic in tokens, so five 3 s segments cost about
a third of one 15 s render (see steel/NOTES.md, "Where the DiT time goes").
Chaining segments on their own last frames drifts; instead every segment is
pinned at BOTH ends to keyframes that come from one short "fast-forward"
clip of the whole story, so the anchors are mutually consistent and native
resolution, and no drift accumulates.

    1. keyframe clip : one 3 s render of the whole story at 5x speed
                       (optionally pinned to --first-frame)
    2. keyframes     : N+1 frames sampled evenly from it, sharpest within a
                       small window (fast motion can leave blur)
    3. segments      : N renders of `segment` seconds, --first-frame k_i,
                       --last-frame k_{i+1}, each with its own prompt
    4. concat        : drop the duplicated seam frame, join video + audio

Everything runs in ONE interactive h3 session so the model loads once.

Spec (JSON):
{
  "model": "./MiniMax-H3-turbo", "width": 1152, "height": 640,
  "segment_seconds": 3, "seed": 7,
  "h3_args": ["--steps", "4", "--use-int8-row-fc2"],      # extra CLI flags
  "first_frame": null,                                     # optional PNG
  "keyframe_prompt": "...whole story, 5x speed...",
  "segments": ["prompt for 0-3 s", "3-6 s", ...]
}
Usage: segmented_fl2va.py spec.json OUT_DIR
"""
import json, os, re, shutil, subprocess, sys, time

H3_DIR = os.path.expanduser("~/h3.c")
FPS = 24


def align_frames(n):                       # mirrors h3_align_frame_count
    v = max(n, 5); r = (v - 5) % 17
    return v + (17 - r) if r else v


def oneline(s):
    return re.sub(r"\s+", " ", s).strip()


def run(cmd, **kw):
    print("+", " ".join(cmd) if isinstance(cmd, list) else cmd, flush=True)
    return subprocess.run(cmd, check=True, **kw)


def sharpness(png):
    from PIL import Image
    import numpy as np
    g = np.asarray(Image.open(png).convert("L"), dtype=np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def extract_frame(video, index, out):
    run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vf", f"select=eq(n\\,{index})",
         "-frames:v", "1", out])


def pick_keyframes(video, n_frames, count, out_dir, window=2):
    """count frames spread evenly over the clip; each is the sharpest within ±window."""
    picks = []
    for i in range(count):
        centre = round(i * (n_frames - 1) / (count - 1))
        best = None
        for j in range(max(0, centre - window), min(n_frames - 1, centre + window) + 1):
            tmp = os.path.join(out_dir, f"cand_{i}_{j}.png")
            extract_frame(video, j, tmp)
            s = sharpness(tmp)
            if best is None or s > best[0]:
                best = (s, j, tmp)
        dst = os.path.join(out_dir, f"key_{i}.png")
        shutil.copy(best[2], dst)
        picks.append((best[1], best[0], dst))
        print(f"keyframe {i}: frame {best[1]} (sharpness {best[0]:.0f})", flush=True)
    for f in os.listdir(out_dir):
        if f.startswith("cand_"):
            os.remove(os.path.join(out_dir, f))
    return picks


def h3_session(spec, out_dir, script_lines, log_path):
    """Feed commands to one interactive h3 process (model loads once)."""
    cmd = [os.path.join(H3_DIR, "h3"), "-d", spec["model"],
           "--width", str(spec["width"]), "--height", str(spec["height"]),
           "--seconds", str(spec.get("segment_seconds", 3))]
    if spec.get("seed") is not None:
        cmd += ["--seed", str(spec["seed"])]
    cmd += spec.get("h3_args", [])
    env = dict(os.environ, H3_VAE_INT8_FFN="1", H3_STEEL_ATTN="1")
    env.update(spec.get("env", {}))
    print("+", " ".join(cmd), flush=True)
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=H3_DIR, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                             env=env, text=True)
        p.stdin.write("\n".join(script_lines) + "\n!quit\n"); p.stdin.close()
        rc = p.wait()
    if rc != 0:
        sys.exit(f"h3 exited {rc}; see {log_path}")


def main():
    spec = json.load(open(sys.argv[1])); out_dir = os.path.abspath(sys.argv[2])
    os.makedirs(out_dir, exist_ok=True)
    seg_s = spec.get("segment_seconds", 3)
    segs = spec["segments"]; n = len(segs)
    frames = align_frames(round(seg_s * FPS))
    t0 = time.time()

    # 1. keyframe clip + 2. keyframes (or reuse a previous run's keyframes/ dir)
    kf_dir = os.path.join(out_dir, "keyframes"); os.makedirs(kf_dir, exist_ok=True)
    if spec.get("keyframes_dir"):
        keys = []
        for i in range(n + 1):
            src = os.path.join(spec["keyframes_dir"], f"key_{i}.png"); dst = os.path.join(kf_dir, f"key_{i}.png")
            shutil.copy(src, dst); keys.append((None, None, dst))
        t1 = time.time(); print(f"reusing keyframes from {spec['keyframes_dir']}", flush=True)
    else:
        kf_mp4 = os.path.join(out_dir, "keyframe_clip.mp4")
        lines = [f"!output {out_dir}", "!show off", "!open off"]
        if spec.get("first_frame"):
            lines.append(f"!first {os.path.abspath(spec['first_frame'])}")
        lines += [oneline(spec["keyframe_prompt"]), f"!save {kf_mp4}"]
        h3_session(spec, out_dir, lines, os.path.join(out_dir, "keyframe_clip.log"))
        t1 = time.time(); print(f"keyframe clip: {t1 - t0:.0f}s", flush=True)
        keys = pick_keyframes(kf_mp4, frames, n + 1, kf_dir)
        if spec.get("first_frame"):
            shutil.copy(spec["first_frame"], keys[0][2])    # the pinned first frame is exact anyway

    # 3. segments, one session
    lines = [f"!output {out_dir}", "!show off", "!open off"]
    seg_files = []
    for i, prompt in enumerate(segs):
        seg = os.path.join(out_dir, f"seg_{i}.mp4"); seg_files.append(seg)
        lines += [f"!first {keys[i][2]}", f"!last {keys[i + 1][2]}", oneline(prompt), f"!save {seg}"]
    h3_session(spec, out_dir, lines, os.path.join(out_dir, "segments.log"))
    t2 = time.time(); print(f"{n} segments: {t2 - t1:.0f}s", flush=True)

    # 4. concat, dropping the seam frame (segment i+1 starts on segment i's last frame)
    inputs, fc = [], []
    for i, seg in enumerate(seg_files):
        inputs += ["-i", seg]
        start = 1 if i else 0
        fc.append(f"[{i}:v]trim=start_frame={start},setpts=PTS-STARTPTS[v{i}];"
                  f"[{i}:a]atrim=start={start / FPS},asetpts=PTS-STARTPTS[a{i}]")
    fc.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][a]")
    final = os.path.join(out_dir, "final.mp4")
    run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(fc),
         "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", final])
    print(f"done: {final}  total {time.time() - t0:.0f}s (clip {t1 - t0:.0f}s + segments {t2 - t1:.0f}s)", flush=True)
    json.dump({"keyframes": [{"frame": f, "sharpness": s, "path": p} for f, s, p in keys],
               "seconds": {"keyframe_clip": t1 - t0, "segments": t2 - t1, "total": time.time() - t0}},
              open(os.path.join(out_dir, "report.json"), "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
