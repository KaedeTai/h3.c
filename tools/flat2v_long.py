#!/usr/bin/env python3
"""Long-form FLAT2V: many short segments chained by their anchors.

Why segment at all, when h3 will happily take --frames 328.

Attention is quadratic in the packed sequence, and the sequence is mostly video
tokens, so doubling the clip roughly quadruples the work: 158 frames costs 62 s
and 328 costs 299 s -- 2.08x the frames for 4.8x the time. Two 158-frame
segments cost 2 x 38 s in one resident process. Segmenting turns long-form from
quadratic in length into linear, and it gets faster the longer the piece is.

The seam is free here in a way it is not for most video models, because of the
two things this pipeline already does:

  the soundtrack is an INPUT, so slicing it and concatenating the results
  reproduces the original audio exactly -- there is nothing to blend

  a frame anchor is a RoPE time coordinate on a reference image, so handing
  segment N+1 the last frame of segment N as its first anchor is the ordinary
  mechanism, not a special case

Anchor policy for the tail is the one real choice, and it is a trade:

  --tail poster   every segment also carries the original poster as its LAST
                  anchor. Identity cannot drift, because no segment is ever
                  more than one hop from the original. The cost is that every
                  segment ends near the same pose.
  --tail none     only the first anchor is set. Nothing pulls the pose back, so
                  it should look freer -- and errors compound, because each
                  segment's first anchor is the previous segment's OUTPUT.

Both are generated and measured rather than argued about; see `check`.

    flat2v_long.py WORK init --poster P.png --voice LONG.wav --line "..."
    flat2v_long.py WORK run  [--tail poster|none] [--seed 42]
    flat2v_long.py WORK check
"""
import argparse, json, os, re, shutil, subprocess, sys, time

H3_DIR = os.path.expanduser("~/h3.c")
H3 = os.path.join(H3_DIR, "h3")
FPS, AUDIO_LATENT_FPS = 24, 40
sys.path.insert(0, os.path.join(H3_DIR, "tools"))


def legal_frames(n):
    """h3 reads video in a head chunk of 5 real frames plus chunks of 17."""
    return 5 + max(1, (int(n) - 5) // 17) * 17


def required_seconds(frames):
    return round(frames * AUDIO_LATENT_FPS / FPS) / AUDIO_LATENT_FPS


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def state_path(w): return os.path.join(w, "flat2v_long.json")
def load(w):
    with open(state_path(w)) as f: return json.load(f)
def save(w, s):
    with open(state_path(w), "w") as f: json.dump(s, f, indent=2, ensure_ascii=False)


def cmd_init(a):
    # h3 runs with cwd=H3_DIR, so every path handed to it or to the REPL has to
    # be absolute -- a relative workdir silently pointed h3 at ~/h3.c/<workdir>.
    a.workdir = os.path.abspath(a.workdir)
    os.makedirs(a.workdir, exist_ok=True)
    import cutplan
    frames = legal_frames(a.segment_frames)
    seg_seconds = required_seconds(frames)
    total = duration(a.voice)

    # WHERE the cut lands matters more than how long the segments are. A
    # boundary is the single frame segment N+1 has to reproduce from a still,
    # and a still of a wide-open mouth mid-vowel is the hardest thing this
    # pipeline reproduces. Legal lengths are 5+17k -- ~0.71 s apart -- so every
    # boundary carries about +/-0.35 s of slack, which is nearly always enough
    # to reach a breath. cutplan spends that slack by shortest path.
    quiet = cutplan.envelope(a.voice)
    clock_plan, clock_rows = cutplan.clock_plan(a.voice, frames, quiet=quiet)
    if a.cut == "silence":
        plan, rows = cutplan.plan(a.voice, frames, quiet=quiet)
    else:
        plan, rows = clock_plan, clock_rows
    if not plan:
        raise SystemExit(f"{total:.2f}s is shorter than one {seg_seconds:.3f}s segment")
    spans = [required_seconds(f) for f in plan]
    print(f"{total:.2f}s of speech -> {len(plan)} segments "
          f"({'+'.join(str(f) for f in plan)} frames, "
          f"{sum(spans):.2f}s of video), cut = {a.cut}")
    cutplan.show("clock", clock_rows)
    if a.cut == "silence":
        cutplan.show("silence", rows)

    from PIL import Image
    src = Image.open(a.poster).convert("RGB")
    want = a.width / a.height
    have = src.width / src.height
    if abs(have - want) > 1e-6:
        # trim, never resize: h3 reads conditioning images with
        # H3_IMAGE_FIT_STRETCH and a squeezed reference stays squeezed all clip
        if have > want:
            nw = int(round(src.height * want)); x = (src.width - nw) // 2
            src = src.crop((x, 0, x + nw, src.height))
        else:
            nh = int(round(src.width / want))
            src = src.crop((0, 0, src.width, nh))
        print(f"  poster trimmed to {src.width}x{src.height} for {want:.4f}")
    poster = os.path.join(a.workdir, "poster.png")
    src.resize((a.width, a.height), Image.LANCZOS).save(poster)

    slices, at = [], 0.0
    for i, (f, span) in enumerate(zip(plan, spans)):
        dst = os.path.join(a.workdir, f"voice_{i:03d}.wav")
        # -ss BEFORE -i, so the slice's timestamps restart at zero. As an
        # output option it only discards decoded frames and leaves the original
        # PTS in place, and apad then believes the stream is already long
        # enough -- the last slice of a clip that needs padding came out short.
        subprocess.run(["ffmpeg", "-y", "-v", "error",
                        "-ss", f"{at:.6f}", "-i", a.voice,
                        "-af", f"apad=whole_dur={span:.6f}",
                        "-t", f"{span:.6f}",
                        "-ar", "24000", "-ac", "1", dst], check=True)
        got = duration(dst)
        if abs(got - span) > 0.01:
            raise SystemExit(f"{dst} is {got:.3f}s, needed {span:.3f}s")
        slices.append(dst); at += span
    print(f"  {len(slices)} audio slices, verified to the millisecond")

    save(a.workdir, {"workdir": os.path.abspath(a.workdir),
                     "poster": poster, "voice": os.path.abspath(a.voice),
                     "slices": slices, "plan": plan, "spans": spans,
                     "cut": a.cut, "cut_rows": rows,
                     "frames": frames, "seg_seconds": seg_seconds,
                     "line": a.line or "", "direction": a.direction,
                     "prompt": ((a.line + " ") if a.line else "") + a.direction,
                     "width": a.width, "height": a.height,
                     "steps": a.steps, "model_dir": a.model_dir})
    print(f"  next: {sys.argv[0]} {a.workdir} run")


def last_frame(mp4, dst, frames):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mp4, "-vf",
                    rf"select=eq(n\,{frames-1})", "-vframes", "1", dst], check=True)
    return dst


def cmd_run(a):
    st = load(a.workdir)
    out_dir = os.path.join(st["workdir"], f"seg_{a.tail}")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    env = dict(os.environ)
    env["H3_DIT_VARIANT"] = "FL2VA"
    env["H3_AUDIO_ANCHOR"] = "1"
    env["H3_REF2VA_ANCHOR"] = "1"
    if a.tail == "poster":
        env["H3_REF2VA_ANCHOR_LAST"] = "2"
    env.setdefault("H3_VAE_INT8_FFN", "1")

    # Segment 1 is generated from the command line so the process starts with a
    # complete reference set; every later segment edits the references in place,
    # which now keeps the DiT and the video decoder resident.
    cmd = [H3, "-d", st["model_dir"], "--ref-image", st["poster"]]
    if a.tail == "poster":
        cmd += ["--ref-image", st["poster"]]
    cmd += ["--ref-audio", st["slices"][0],
            "--width", str(st["width"]), "--height", str(st["height"]),
            "--frames", str(st["plan"][0]), "--steps", str(st["steps"]),
            "--seed", str(a.seed), "--use-int8-row-fc2"]

    script = [f"!output {out_dir}", st["prompt"]]
    # the rest are placeholders; the driver rewrites them as it goes, because
    # segment N+1's first anchor does not exist until segment N has been decoded
    print(f"  {len(st['slices'])} segments, tail anchor = {a.tail}", flush=True)
    log = open(os.path.join(out_dir, "h3.log"), "w+")
    # Do NOT parse h3's stdout for progress. Through a pipe its libc buffers in
    # blocks, so "Done -> ..." does not arrive until the buffer fills -- the
    # first attempt at this deadlocked with a finished segment on disk and a
    # driver still waiting to hear about it. The output filenames are known
    # (!output DIR numbers them video-0001 up), so poll the filesystem and time
    # it here instead.
    proc = subprocess.Popen(cmd, cwd=H3_DIR, env=env, stdin=subprocess.PIPE,
                            stdout=log, stderr=subprocess.STDOUT, text=True)

    produced, timings = [], []

    def feed(lines):
        for l in lines:
            proc.stdin.write(l + "\n")
        proc.stdin.flush()

    # FLAT2V_SEG_TIMEOUT: how long one segment may take before the driver
    # gives up. 240 s is right for an idle machine (segments measure 40-60 s),
    # but a machine deep into swap can stall a segment past it -- and giving up
    # discards every segment already computed, because the next run rmtree's
    # the output directory. Raise it rather than lose the work.
    seg_limit = float(os.environ.get("FLAT2V_SEG_TIMEOUT", "240"))

    def wait_for(path, started, limit=seg_limit):
        size = -1
        while time.time() - started < limit:
            if proc.poll() is not None:
                return None
            if os.path.exists(path):
                now = os.path.getsize(path)
                if now and now == size:            # written and settled
                    return time.time() - started
                size = now
            time.sleep(0.5)
        return None

    feed(script)
    for index in range(len(st["slices"])):
        if index:
            anchor = last_frame(produced[-1],
                                os.path.join(out_dir, f"anchor_{index:03d}.png"),
                                st["plan"][index - 1])
            refs = ["!refs clear", f"!ref-image {anchor}"]
            if st["plan"][index] != st["plan"][index - 1]:
                refs.append(f"!frames {st['plan'][index]}")
            if a.tail == "poster":
                refs.append(f"!ref-image {st['poster']}")
            refs += [f"!ref-audio {st['slices'][index]}", st["prompt"]]
            feed(refs)
        target = os.path.join(out_dir, f"video-{index+1:04d}.mp4")
        secs = wait_for(target, time.time())
        if secs is None:
            log.flush(); log.seek(0)
            tail = [l for l in log.read().replace("\r", "\n").split("\n")
                    if l.startswith("h3:") or "rror" in l][-6:]
            print("  h3 produced no segment; last complaints:")
            for l in tail: print("    " + l)
            raise SystemExit(1)
        produced.append(target); timings.append(secs)
        print(f"  segment {index+1}/{len(st['slices'])}: {secs:5.1f}s", flush=True)
    feed(["!quit"])
    proc.wait()

    # Joining needs care, and the obvious way is wrong.
    #
    # Concatenating the muxed segments with -c copy destroyed the soundtrack:
    # every segment measures r = 0.9704 against its own slice, and the join
    # measured 0.0040 against the source. Each segment's AAC stream carries its
    # own encoder priming delay, and four of them stack into ~130 ms of
    # accumulated shift.
    #
    # There is a second, subtler drift underneath that one. h3 gives 158 frames
    # 263 audio latent frames, which is 6.575 s, while 158 frames at 24 fps is
    # 6.5833 s -- so every segment's video runs 8.3 ms longer than the audio it
    # was generated against. Over four segments that is 33 ms; over twenty it is
    # 166 ms and audible.
    #
    # So: join the VIDEO only, rescale its timestamps so each segment lasts
    # exactly the audio span it was generated for (24 fps was the
    # approximation, 158/6.575 = 24.03 is the truth), and lay the ORIGINAL
    # source audio over the result. The soundtrack was an input all along; there
    # is no reason for the deliverable to carry a VAE round trip of it.
    joined = join_segments(st, out_dir, produced, a.tail)

    st.setdefault("runs", {})[a.tail] = {
        "segments": produced, "timings": timings, "joined": joined,
        "seed": a.seed}
    save(a.workdir, st)
    total = sum(timings)
    seconds_of_video = sum(st["spans"])
    print(f"\n  {total:.1f}s for {seconds_of_video:.1f}s of video "
          f"({total/seconds_of_video:.1f}x realtime)")
    print(f"  first segment {timings[0]:.1f}s, the rest "
          f"{min(timings[1:]) if len(timings)>1 else 0:.1f}"
          f"-{max(timings[1:]) if len(timings)>1 else 0:.1f}s")
    print(f"  -> {joined}")



def join_segments(st, out_dir, produced, tail):
    """Video-only concat, rescaled to the true audio clock, source audio laid over."""
    listing = os.path.join(out_dir, "segments.txt")
    with open(listing, "w") as f:
        for p in produced:
            f.write(f"file '{os.path.abspath(p)}'\n")
    video_only = os.path.join(out_dir, "video_only.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", listing, "-an", "-c", "copy", video_only], check=True)
    video_seconds = sum(st["plan"]) / FPS
    audio_seconds = sum(st["spans"])
    scale = audio_seconds / video_seconds
    joined = os.path.join(st["workdir"], f"long_{tail}.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error",
                    "-itsscale", f"{scale:.9f}", "-i", video_only,
                    "-i", st["voice"],
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", "pan=mono|c0=0.5*c0+0.5*c1,highpass=f=60",
                    "-c:a", "aac", "-b:a", "128k", "-ac", "1", "-ar", "32000",
                    "-t", f"{audio_seconds:.6f}", joined], check=True)
    print(f"  joined at {sum(st['plan'])/audio_seconds:.3f} fps "
          f"(24 fps would drift {1000*(video_seconds-audio_seconds):.0f} ms)")
    return joined



def grade_segment(st, index, path):
    from take_report import report
    from subtitle_rate import rate as subtitle_rate
    os.environ["FLAT2V_REF"] = st["poster"]
    os.environ["FLAT2V_WAV"] = st["slices"][index]
    m = report(path, last=st["plan"][index] - 1)
    m["subtitles"] = subtitle_rate(path)
    return m


SUBTITLE_LIMIT = 5.0    # percent of frames; a caption for a moment is a caption


def pick_best(candidates):
    """Best mouth among the takes that are not captioned.

    At 480x640 the picture has room to render text, and it renders the
    prompt: five of eleven segments came back with the spoken line burned in.
    Aperture cannot see a caption, so choosing by aperture alone chose them.
    """
    def corrupted(m):
        return m.get("drift_max", 0) > 2.2 * max(m.get("drift_med", 1e-9), 1e-9) + 3
    clean = [c for c in candidates
             if c[3].get("subtitles", 0.0) <= SUBTITLE_LIMIT and not corrupted(c[3])]
    if clean:
        return max(clean, key=lambda c: c[0])
    # nothing clean: prefer uncorrupted, then least captioned, and say so
    sane = [c for c in candidates if not corrupted(c[3])] or candidates
    worst = min(sane, key=lambda c: c[3].get("subtitles", 0.0))
    print(f"  no clean candidate; keeping the least captioned uncorrupted one "
          f"({worst[3].get('subtitles', 0.0):.0f}%, drift "
          f"{worst[3].get('drift_max', 0):.0f}/{worst[3].get('drift_med', 0):.0f})")
    return worst


def cmd_redo(a):
    """Re-roll one or more segments with other seeds and keep the best mouth.

    A long run is a chain, and one link in it can drop into the photograph
    fixed point -- aperture 0.0007, mouth shut for six seconds -- while every
    neighbour is fine. The run does not need repeating; the link does. Its
    first anchor is already on disk (the previous segment's last frame) and
    its last anchor is the poster, so it regenerates independently. The next
    segment's first anchor was taken from the OLD version of this one, but with
    the poster as tail anchor every version ends within reproduction error of
    the same frame, so the seam holds without cascading.
    """
    st = load(a.workdir)
    run = st["runs"][a.tail]
    out_dir = os.path.join(st["workdir"], f"seg_{a.tail}")
    env = dict(os.environ)
    env["H3_DIT_VARIANT"] = "FL2VA"; env["H3_AUDIO_ANCHOR"] = "1"
    env["H3_REF2VA_ANCHOR"] = "1"
    if a.tail == "poster": env["H3_REF2VA_ANCHOR_LAST"] = "2"
    env.setdefault("H3_VAE_INT8_FFN", "1")
    which = [int(x) - 1 for x in a.segments.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    # a fresh directory every time: !output numbers files from 1, and a poll
    # that finds a stale video-0001.mp4 from an earlier redo returns at once
    # with the wrong take
    redo_dir = os.path.join(out_dir, f"redo_{int(time.time())}")
    os.makedirs(redo_dir)
    # one resident process for every candidate: rebind makes each one cheap
    first = which[0]
    anchor0 = (st["poster"] if first == 0 else
               os.path.join(out_dir, f"anchor_{first:03d}.png"))
    cmd = [H3, "-d", st["model_dir"], "--ref-image", anchor0]
    if a.tail == "poster": cmd += ["--ref-image", st["poster"]]
    cmd += ["--ref-audio", st["slices"][first],
            "--width", str(st["width"]), "--height", str(st["height"]),
            "--frames", str(st["plan"][first]), "--steps", str(st["steps"]),
            "--seed", str(seeds[0]), "--use-int8-row-fc2"]
    log = open(os.path.join(redo_dir, "h3.log"), "a")
    proc = subprocess.Popen(cmd, cwd=H3_DIR, env=env, stdin=subprocess.PIPE,
                            stdout=log, stderr=subprocess.STDOUT, text=True)
    def feed(lines):
        for l in lines: proc.stdin.write(l + "\n")
        proc.stdin.flush()
    feed([f"!output {redo_dir}"])
    counter = 0
    replaced = []
    for index in which:
        anchor = (st["poster"] if index == 0 else
                  os.path.join(out_dir, f"anchor_{index:03d}.png"))
        old = run["segments"][index]
        candidates = []
        if os.path.exists(old):
            m = grade_segment(st, index, old)
            candidates.append((m["aperture"], old, run.get("seed", "?"), m))
        for seed in seeds:
            counter += 1
            refs = ["!refs clear", f"!ref-image {anchor}"]
            refs.append(f"!frames {st['plan'][index]}")
            if a.tail == "poster": refs.append(f"!ref-image {st['poster']}")
            refs += [f"!ref-audio {st['slices'][index]}", f"!seed {seed}", st["prompt"]]
            feed(refs)
            target = os.path.join(redo_dir, f"video-{counter:04d}.mp4")
            started, size = time.time(), -1
            while time.time() - started < 300:
                if proc.poll() is not None: break
                if os.path.exists(target):
                    now = os.path.getsize(target)
                    if now and now == size: break
                    size = now
                time.sleep(0.5)
            if not os.path.exists(target):
                print(f"  segment {index+1} seed {seed}: h3 produced nothing"); continue
            m = grade_segment(st, index, target)
            candidates.append((m["aperture"], target, seed, m))
            print(f"  segment {index+1:2d} seed {seed:3}: aperture {m['aperture']:.4f} "
                  f"head {m['head']:.2f} r {m['audio_r']:.3f} "
                  f"subtitles {m['subtitles']:.0f}% drift {m['drift_max']:.0f}/{m['drift_med']:.0f}",
                  flush=True)
        best = pick_best(candidates)
        print(f"  segment {index+1:2d} -> keeping seed {best[2]} "
              f"(aperture {best[0]:.4f})")
        if best[1] != old:
            shutil.copy(best[1], old)
            replaced.append(index + 1)
            if index + 1 < len(run["segments"]):
                last_frame(old, os.path.join(out_dir, f"anchor_{index+1:03d}.png"),
                           st["plan"][index])
    feed(["!quit"]); proc.wait()
    joined = join_segments(st, out_dir, run["segments"], a.tail)
    run.setdefault("redone", []).extend(replaced)
    save(a.workdir, st)
    print(f"  replaced segments {replaced or 'none'}; rejoined -> {joined}")


def cmd_check(a):
    """Does the seam show, and does the face drift?"""
    import numpy as np
    from PIL import Image
    from take_report import _frames, report, verdict
    from lipsync_score import mouth_curve
    st = load(a.workdir)
    poster = np.asarray(Image.open(st["poster"]).convert("RGB")).astype(np.float32)
    for tail, run in sorted(st.get("runs", {}).items()):
        print(f"\n=== tail anchor = {tail} ===")
        segs, plan = run["segments"], st["plan"]
        print(f"{'segment':9s} {'first frame vs poster':>22s} {'seam step':>11s} "
              f"{'typical step':>13s} {'mouth at cut':>15s}")
        prev_last = None
        mouths = [mouth_curve(p) for p in segs]
        for i, p in enumerate(segs):
            F = plan[i]
            fr = _frames(p, [0, 1, F // 2, F // 2 + 1, F - 1])
            drift = float(np.abs(fr[0] - poster).mean())
            # the step across the seam has to be read against the steps this
            # very segment takes internally, not against some fixed number
            typical = float(np.mean([np.abs(fr[1] - fr[0]).mean(),
                                     np.abs(fr[F//2+1] - fr[F//2]).mean()]))
            seam = (float(np.abs(fr[0] - prev_last).mean())
                    if prev_last is not None else float("nan"))
            # How open the mouth was in the frame the NEXT segment has to
            # reproduce from a still. This is the mechanism the cut planner
            # aims at: a closed mouth is an easy anchor, a stressed vowel is
            # the hardest thing this pipeline reproduces.
            gap = mouths[i][F - 1] if mouths[i] is not None else float("nan")
            print(f"{i+1:<9d} {drift:22.2f} {seam:11.2f} {typical:13.2f} "
                  f"{gap:15.3f}")
            prev_last = fr[F - 1]
        os.environ["FLAT2V_REF"] = st["poster"]
        os.environ["FLAT2V_WAV"] = st["slices"][0]
        m = report(segs[0], last=plan[0] - 1)
        print(f"  segment 1 grade: aperture {m['aperture']:.4f}  "
              f"head {m['head']:.2f}  audio r {m['audio_r']:.3f}  {verdict(m)}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workdir")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init"); i.set_defaults(fn=cmd_init)
    i.add_argument("--poster", required=True)
    i.add_argument("--voice", required=True)
    i.add_argument("--line")
    i.add_argument("--direction", default=(
        "(tight head-and-shoulders framing, softly blurred interior behind, "
        "locked-off camera, no zoom, no camera movement, soft even indoor "
        "light, documentary realism)"))
    i.add_argument("--segment-frames", type=int, default=158)
    i.add_argument("--cut", choices=("silence", "clock"), default="silence",
                   help="silence: spend the +/-0.35 s of slack every legal "
                        "length carries to land each boundary in a pause")
    i.add_argument("--width", type=int, default=384)
    i.add_argument("--height", type=int, default=512)
    i.add_argument("--steps", type=int, default=3)
    i.add_argument("--model-dir", default="./MiniMax-H3-turbo-int8")
    r = sub.add_parser("run"); r.set_defaults(fn=cmd_run)
    r.add_argument("--tail", choices=("poster", "none"), default="poster")
    r.add_argument("--seed", type=int, default=42)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    d = sub.add_parser("redo"); d.set_defaults(fn=cmd_redo)
    d.add_argument("--segments", required=True, help="1-based, comma-separated")
    d.add_argument("--seeds", default="7,3,11")
    d.add_argument("--tail", choices=("poster", "none"), default="poster")
    a = p.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
