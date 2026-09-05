#!/usr/bin/env python3
"""longshot: a silent (or post-dubbed) long shot as a reviewed, segmented FL2VA render.

    script (timed beats)  ->  preview (3 s fast-forward)  ->  keyframes  ->  segments  ->  assemble

Why this shape (steel/NOTES.md, "Segmented FL2VA"): DiT attention is quadratic in
tokens, so five 3 s renders cost less than half of one 15 s render, and every
segment can be re-rolled on its own. The fast-forward clip costs ~100 s and is a
storyboard you approve BEFORE paying for the segments. Two rules make it work:
no speed knobs, and every segment prompt states what is already on screen plus
only this segment's change - never the whole arc.

Script (JSON):
{
  "name": "timelapse", "width": 1152, "height": 640, "seed": 7,
  "model": "./MiniMax-H3-turbo",
  "h3_args": ["--steps", "4", "--use-int8-row-fc2", "--layers", "45"],
  "world": "shared description of the scene / UI / cursor / sound. NO process narrative
            (nothing like 'starts blank and builds up') - that belongs in the beats.",
  "subject": "the thing that gets built / appears (a character, a product ...). Used by the
              preview and the segments, NOT by the opening-frame render - otherwise the model
              paints the finished subject into the opening frame.",
  "beats": [
    {"t": 0,  "state": "what is on screen at t=0",  "action": "what happens until the next beat"},
    {"t": 3,  "state": "...", "action": "..."},
    ...
    {"t": 15, "state": "final state"}                       # last beat: state only
  ],
  "first_frame": null,                 # optional PNG pinned as keyframe 0
  "preview_seconds": 3,
  "seam_drop": 8,                      # frames cut after every pinned first frame (transient)
  "audio":  {"keep_h3_audio": false, "music": null, "music_gain_db": -8, "fade_s": 1.5},
  "post":   {"rife_fps": 48, "upscale": 2}     # null to skip; uses ~/repos/*-ncnn-vulkan
}

Usage:
  longshot.py SCRIPT WORKDIR preview   [--seed N]        3 s fast-forward + beat sheet -> approve
  longshot.py SCRIPT WORKDIR keyframes [--enhance]       sharpest frame per beat -> keyframes/active/
  longshot.py SCRIPT WORKDIR segments  [--only 1,3]      render (or re-roll) segments -> strips
  longshot.py SCRIPT WORKDIR assemble  [--no-post]       concat, audio, RIFE + ESRGAN
  longshot.py SCRIPT WORKDIR status
Replace any keyframes/active/key_N.png by hand (same size) before `segments` to steer a seam.
"""
import argparse, json, os, re, shutil, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segmented_fl2va import align_frames, oneline, run, sharpness, extract_frame, H3_DIR, FPS  # noqa: E402

RIFE = os.path.expanduser("~/repos/rife-ncnn-vulkan/rife-ncnn-vulkan")
RIFE_MODEL = os.path.expanduser("~/repos/rife-ncnn-vulkan/rife-v4.6")
ESRGAN = os.path.expanduser("~/repos/realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan")
ESRGAN_MODELS = os.path.expanduser("~/repos/realesrgan-ncnn-vulkan/models")
DEFAULT_H3_ARGS = ["--steps", "4", "--use-int8-row-fc2", "--layers", "45"]
ARC_WORDS = ("從空白", "從零開始", "逐步重建", "從無到有", "完整流程", "整個過程")


# ---------------------------------------------------------------- script / state
def load_script(path):
    s = json.load(open(path, encoding="utf-8"))
    beats = s["beats"]
    assert len(beats) >= 2 and all(b["t"] < c["t"] for b, c in zip(beats, beats[1:])), "beats must be increasing"
    assert "action" not in beats[-1] or not beats[-1]["action"], "last beat is a state only"
    for w in ARC_WORDS:
        if w in s["world"]:
            print(f"warning: 'world' contains '{w}' - process narrative there makes every segment "
                  f"replay the whole arc; move it into the beats", file=sys.stderr)
    s.setdefault("subject", "")
    s.setdefault("model", "./MiniMax-H3-turbo"); s.setdefault("h3_args", DEFAULT_H3_ARGS)
    s.setdefault("preview_seconds", 3); s.setdefault("seam_drop", 8); s.setdefault("seed", 7)
    s.setdefault("audio", {}); s.setdefault("post", {})
    s["audio"] = {"keep_h3_audio": False, "music": None, "music_gain_db": -8, "fade_s": 1.5, **s["audio"]}
    s["post"] = {"rife_fps": 48, "upscale": 2, **s["post"]}
    return s


def state_path(wd): return os.path.join(wd, "state.json")
def load_state(wd):
    p = state_path(wd); return json.load(open(p)) if os.path.exists(p) else {}
def save_state(wd, st): json.dump(st, open(state_path(wd), "w"), indent=2, ensure_ascii=False)


def segments_of(script):
    b = script["beats"]
    return [{"index": i, "seconds": b[i + 1]["t"] - b[i]["t"], "state": b[i]["state"],
             "action": b[i].get("action", ""), "end_state": b[i + 1]["state"]} for i in range(len(b) - 1)]


# ---------------------------------------------------------------- prompts
def world_and_subject(script):
    return oneline(script["world"] + " " + script.get("subject", ""))


def preview_prompt(script):
    P = script["preview_seconds"]; b = script["beats"]; T = b[-1]["t"]
    lines = []
    for i in range(len(b) - 1):
        t0, t1 = b[i]["t"] / T * P, b[i + 1]["t"] / T * P
        lines.append(f"{t0:.1f}–{t1:.1f} 秒 {b[i]['state']}；{b[i].get('action', '')}")
    # The preview is the one prompt that MAY narrate the whole arc. The opening state has to be
    # said first, before the world text, or the model opens on the final state.
    return f"開場第一幀：{b[0]['state']}。絕對不要把最終狀態當開場。" + world_and_subject(script) + \
        f" 這是一段 {P} 秒的極速快轉，把整段流程壓縮在 {P} 秒內依序、等距演完，每一格畫面都清晰銳利、" \
        f"沒有動態模糊，場景與介面全程固定。再說一次，第一幀是：{b[0]['state']}。" + " ".join(lines) + \
        f" 最後一格停在：{b[-1]['state']}。"


def segment_prompt(script, seg):
    d = seg["seconds"]
    return world_and_subject(script) + f" 這 {d} 秒只做下面這一個階段，畫面上已經有的內容保留、不重畫、不清空、不放大。" \
        f"畫面第一幀的狀態：{seg['state']}。0:00–0:{d:02d} {seg['action']} 結尾時的狀態：{seg['end_state']}。"


# ---------------------------------------------------------------- h3
def h3_session(script, lines, log_path, seconds, seed):
    cmd = [os.path.join(H3_DIR, "h3"), "-d", script["model"], "--width", str(script["width"]),
           "--height", str(script["height"]), "--seconds", str(seconds), "--seed", str(seed), *script["h3_args"]]
    env = dict(os.environ, H3_VAE_INT8_FFN="1", H3_STEEL_ATTN="1", **script.get("env", {}))
    print("+", " ".join(cmd), flush=True)
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=H3_DIR, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, env=env, text=True)
        p.stdin.write("\n".join(lines) + "\n!quit\n"); p.stdin.close(); rc = p.wait()
    if rc != 0:
        sys.exit(f"h3 exited {rc}; see {log_path}")


def contact_strip(video, n_frames, out, count=8, scale=0.25, w=1152, h=640):
    from PIL import Image
    W, H = int(w * scale), int(h * scale)
    sheet = Image.new("RGB", (W * count, H), "black")
    for k in range(count):
        n = round(k * (n_frames - 1) / (count - 1)); tmp = out + f".{k}.png"
        extract_frame(video, n, tmp); sheet.paste(Image.open(tmp).resize((W, H)), (k * W, 0)); os.remove(tmp)
    sheet.save(out)


# ---------------------------------------------------------------- stages
def stage_preview(script, wd, a):
    seed = a.seed if a.seed is not None else script["seed"]
    P = script["preview_seconds"]; frames = align_frames(round(P * FPS))
    clip = os.path.join(wd, f"preview_seed{seed}.mp4")
    t0 = time.time()
    # A pinned first frame is optional and should come from the SAME kind of render: a
    # blank-canvas frame borrowed from another run had a different UI layout, the clip
    # drifted to its own layout, and segment 0 could no longer reach keyframe 1. When the
    # clip opens on the wrong state (it likes the final state for timelapse prompts), the
    # cheap fix is another --seed, not a foreign first frame.
    first = os.path.abspath(script["first_frame"]) if script.get("first_frame") else None
    lines = [f"!output {wd}", "!show off", "!open off"] + ([f"!first {first}"] if first else []) + \
            [preview_prompt(script), f"!save {clip}"]
    h3_session(script, lines, os.path.join(wd, "preview.log"), P, seed)
    contact_strip(clip, frames, os.path.join(wd, f"preview_seed{seed}_sheet.png"), count=len(script["beats"]),
                  w=script["width"], h=script["height"], scale=0.4)
    st = load_state(wd); st["preview"] = {"seed": seed, "clip": clip, "frames": frames, "seconds": time.time() - t0,
                                          "prompt": preview_prompt(script), "first_frame": first}
    st.pop("keyframes", None); st.pop("segments", None); st.pop("assemble", None); save_state(wd, st)
    print(f"preview: {clip} ({time.time() - t0:.0f}s). Review {clip[:-4]}_sheet.png (one column per beat), "
          f"then `keyframes`, or re-roll with --seed.", flush=True)


def stage_keyframes(script, wd, a):
    from PIL import Image
    st = load_state(wd); pv = st.get("preview") or sys.exit("run preview first")
    b = script["beats"]; T = b[-1]["t"]; F = pv["frames"]
    raw = os.path.join(wd, "keyframes", "raw"); act = os.path.join(wd, "keyframes", "active")
    os.makedirs(raw, exist_ok=True); os.makedirs(act, exist_ok=True)
    import numpy as np
    picks = []
    for i, beat in enumerate(b):
        # Candidates within ±3 frames. A fast-forward clip flashes transient UI (popups, dialogs)
        # for a frame or two, and those frames are the *sharpest* (more edges), so first drop
        # candidates that stray from the window's per-pixel median, then take the sharpest.
        centre = round(beat["t"] / T * (F - 1)); cands = []
        for j in range(max(0, centre - 3), min(F - 1, centre + 3) + 1):
            tmp = os.path.join(raw, f"cand_{i}_{j}.png"); extract_frame(pv["clip"], j, tmp)
            cands.append((j, tmp, np.asarray(Image.open(tmp).convert("L"), dtype=np.float32)))
        med = np.median(np.stack([c[2] for c in cands]), axis=0)
        dev = [float(np.abs(c[2] - med).mean()) for c in cands]
        keep = [k for k, d in enumerate(dev) if d <= 1.5 * min(dev) + 0.5]
        best = max(keep, key=lambda k: sharpness(cands[k][1]))
        j, tmp = cands[best][0], cands[best][1]; sh = sharpness(tmp)
        dst = os.path.join(raw, f"key_{i}.png"); shutil.copy(tmp, dst)
        picks.append({"beat": i, "frame": j, "sharpness": sh, "deviation": dev[best]})
        print(f"keyframe {i} (t={beat['t']}s): frame {j} sharpness {sh:.0f} (dropped {len(cands) - len(keep)} transient)", flush=True)
    for f in os.listdir(raw):
        if f.startswith("cand_"): os.remove(os.path.join(raw, f))
    if pv.get("first_frame"):
        shutil.copy(pv["first_frame"], os.path.join(raw, "key_0.png"))   # the pinned opening is exact
    for i in range(len(b)):
        src = os.path.join(raw, f"key_{i}.png"); dst = os.path.join(act, f"key_{i}.png")
        if a.enhance:
            # same-size clean-up: ESRGAN x2 then Lanczos back down. Anchors must keep the render size.
            tmp_in = os.path.join(wd, "keyframes", "_enh_in"); tmp_out = os.path.join(wd, "keyframes", "_enh_out")
            shutil.rmtree(tmp_in, ignore_errors=True); shutil.rmtree(tmp_out, ignore_errors=True)
            os.makedirs(tmp_in); os.makedirs(tmp_out); shutil.copy(src, os.path.join(tmp_in, "k.png"))
            run([ESRGAN, "-i", tmp_in, "-o", tmp_out, "-n", "realesr-animevideov3", "-s", "2", "-m", ESRGAN_MODELS])
            Image.open(os.path.join(tmp_out, "k.png")).resize((script["width"], script["height"]), Image.LANCZOS).save(dst)
            shutil.rmtree(tmp_in); shutil.rmtree(tmp_out)
        else:
            shutil.copy(src, dst)
    W, H = script["width"], script["height"]; s = 0.4
    sheet = Image.new("RGB", (int(W * s) * len(b), int(H * s)), "black")
    for i in range(len(b)):
        sheet.paste(Image.open(os.path.join(act, f"key_{i}.png")).resize((int(W * s), int(H * s))), (i * int(W * s), 0))
    sheet.save(os.path.join(wd, "keyframes_sheet.png"))
    st["keyframes"] = {"picks": picks, "enhanced": bool(a.enhance), "dir": act}; st.pop("segments", None); save_state(wd, st)
    print(f"keyframes -> {act}. Review keyframes_sheet.png; edit/replace any key_N.png (keep {W}x{H}), then `segments`.")


def stage_segments(script, wd, a):
    st = load_state(wd); kf = st.get("keyframes") or sys.exit("run keyframes first")
    segs = segments_of(script); only = set(int(x) for x in a.only.split(",")) if a.only else set(range(len(segs)))
    seed = a.seed if a.seed is not None else script["seed"]
    st.setdefault("segments", {})
    for seg in segs:
        i = seg["index"]
        if i not in only: continue
        out = os.path.join(wd, f"seg_{i}.mp4"); frames = align_frames(round(seg["seconds"] * FPS))
        lines = [f"!output {wd}", "!show off", "!open off", f"!seconds {seg['seconds']}",
                 f"!first {os.path.join(kf['dir'], f'key_{i}.png')}", f"!last {os.path.join(kf['dir'], f'key_{i + 1}.png')}",
                 f"!seed {seed + i}", segment_prompt(script, seg), f"!save {out}"]
        t0 = time.time(); h3_session(script, lines, os.path.join(wd, f"seg_{i}.log"), seg["seconds"], seed + i)
        contact_strip(out, frames, os.path.join(wd, f"seg_{i}_strip.png"), w=script["width"], h=script["height"])
        st["segments"][str(i)] = {"file": out, "frames": frames, "seed": seed + i, "seconds": time.time() - t0,
                                  "prompt": segment_prompt(script, seg)}
        save_state(wd, st)
        print(f"segment {i}: {out} ({time.time() - t0:.0f}s) -> review seg_{i}_strip.png", flush=True)
    missing = [s["index"] for s in segs if str(s["index"]) not in st["segments"]]
    print("all segments rendered; `assemble` next" if not missing else f"still missing segments {missing}")


def stage_assemble(script, wd, a):
    st = load_state(wd); segs = segments_of(script)
    files = []
    for s in segs:
        e = st.get("segments", {}).get(str(s["index"])) or sys.exit(f"segment {s['index']} not rendered")
        files.append(e["file"])
    drop = script["seam_drop"]; n = len(files)
    inputs, fc = [], []
    for i, f in enumerate(files):
        inputs += ["-i", f]; start = drop if i else 0
        fc.append(f"[{i}:v]trim=start_frame={start},setpts=PTS-STARTPTS[v{i}];[{i}:a]atrim=start={start / FPS},asetpts=PTS-STARTPTS[a{i}]")
    fc.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][a]")
    cut = os.path.join(wd, "assembled.mp4")
    run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(fc), "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", cut])
    # audio
    au = script["audio"]; final = os.path.join(wd, "final.mp4")
    if au.get("music"):
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                             "-of", "default=nw=1:nk=1", cut]).decode().strip())
        fade = au["fade_s"]; gain = au["music_gain_db"]
        music = os.path.expanduser(au["music"])
        if au["keep_h3_audio"]:
            afilter = f"[1:a]volume={gain}dB,afade=t=out:st={dur - fade}:d={fade}[m];[0:a][m]amix=inputs=2:duration=first[a]"
        else:
            afilter = f"[1:a]volume={gain}dB,afade=t=in:d={fade},afade=t=out:st={dur - fade}:d={fade}[a]"
        run(["ffmpeg", "-v", "error", "-y", "-i", cut, "-stream_loop", "-1", "-i", music, "-filter_complex", afilter,
             "-map", "0:v", "-map", "[a]", "-t", f"{dur:.3f}", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", final])
    elif au["keep_h3_audio"]:
        shutil.copy(cut, final)
    else:
        run(["ffmpeg", "-v", "error", "-y", "-i", cut, "-an", "-c:v", "copy", final])
    # seams sheet for review
    from PIL import Image
    W, H = script["width"], script["height"]; s = 0.3
    seam_at = []; acc = 0
    for i, sg in enumerate(segs):
        fr = st["segments"][str(sg["index"])]["frames"]; acc += fr - (drop if i else 0)
        if i < n - 1: seam_at.append(acc)
    sheet = Image.new("RGB", (int(W * s) * 5, int(H * s) * max(1, len(seam_at))), "black")
    for r, c in enumerate(seam_at):
        for k, nn in enumerate(range(c - 2, c + 3)):
            tmp = os.path.join(wd, "_seam.png"); extract_frame(final, nn, tmp)
            sheet.paste(Image.open(tmp).resize((int(W * s), int(H * s))), (k * int(W * s), r * int(H * s)))
    if seam_at: sheet.save(os.path.join(wd, "seams.png")); os.remove(os.path.join(wd, "_seam.png"))
    out = {"assembled": cut, "final": final}
    # post: RIFE + ESRGAN (the timelapse post.sh recipe)
    post = script["post"]
    if not a.no_post and (post.get("rife_fps") or post.get("upscale")):
        fi, fr_, fu = [os.path.join(wd, d) for d in ("frames_in", "frames_rife", "frames_up")]
        for d in (fi, fr_, fu): shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        run(["ffmpeg", "-v", "error", "-y", "-i", final, os.path.join(fi, "%08d.png")])
        N = len(os.listdir(fi)); src = fi; fps = FPS
        if post.get("rife_fps"):
            mult = int(round(post["rife_fps"] / FPS))
            run([RIFE, "-i", src, "-o", fr_, "-m", RIFE_MODEL, "-n", str(N * mult), "-j", "2:4:4"]); src = fr_; fps = FPS * mult
        if post.get("upscale"):
            run([ESRGAN, "-i", src, "-o", fu, "-n", "realesr-animevideov3", "-s", str(post["upscale"]), "-m", ESRGAN_MODELS, "-j", "2:4:4"]); src = fu
        posted = os.path.join(wd, "final_post.mp4")
        run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", os.path.join(src, "%08d.png"), "-i", final,
             "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", "-shortest", posted])
        for d in (fi, fr_, fu): shutil.rmtree(d, ignore_errors=True)
        out["final_post"] = posted
    st["assemble"] = out; save_state(wd, st)
    print("assembled:", json.dumps(out, indent=2, ensure_ascii=False))


def stage_status(script, wd, a):
    st = load_state(wd); segs = segments_of(script)
    print("preview  :", "seed %s, %.0fs" % (st["preview"]["seed"], st["preview"]["seconds"]) if st.get("preview") else "-")
    print("keyframes:", ("%d picked%s" % (len(st["keyframes"]["picks"]), ", enhanced" if st["keyframes"]["enhanced"] else "")) if st.get("keyframes") else "-")
    for s in segs:
        e = st.get("segments", {}).get(str(s["index"]))
        print(f"segment {s['index']} ({s['seconds']}s):", f"done seed {e['seed']} {e['seconds']:.0f}s" if e else "-")
    print("assemble :", st.get("assemble", "-"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script"); ap.add_argument("workdir")
    ap.add_argument("stage", choices=["preview", "keyframes", "segments", "assemble", "status"])
    ap.add_argument("--seed", type=int); ap.add_argument("--enhance", action="store_true")
    ap.add_argument("--only", help="segments: comma-separated segment indices to (re)render")
    ap.add_argument("--no-post", action="store_true")
    a = ap.parse_args()
    script = load_script(a.script); wd = os.path.abspath(a.workdir); os.makedirs(wd, exist_ok=True)
    shutil.copy(a.script, os.path.join(wd, "script.json"))
    {"preview": stage_preview, "keyframes": stage_keyframes, "segments": stage_segments,
     "assemble": stage_assemble, "status": stage_status}[a.stage](script, wd, a)


if __name__ == "__main__":
    main()
