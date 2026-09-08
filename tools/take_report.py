#!/usr/bin/env python3
"""Four numbers that decide whether a FLAT2V take is good.

Each one exists because a simpler number lied first.

  anchor0 / anchor157   mean |pixel| against the reference still. Both should
                        sit near 4, which is h3's own run-to-run noise. This is
                        the only check that the RoPE anchors held.

  scene                 mean |pixel| between consecutive frames. LOW is good:
                        the camera is locked off, so anything here is drift.

                        Two things this number is NOT. It is not "the face is
                        animated" -- a take whose mouth was frozen solid scored
                        9.41 here against 0.63 for a correctly articulating
                        one, because it was measuring background churn. And it
                        is not comparable across framings: the same head
                        movement in a tight head-and-shoulders crop moves a far
                        larger share of the frame than in a wide shot, so the
                        identical config reads 0.86 on a wide crop and 3.10 on
                        a tight one at the same canvas. Compare it only against
                        other takes of the same crop.

  aperture              standard deviation of inner-lip gap over mouth width,
                        from mediapipe landmarks. This is the articulation
                        measure, and it has two thresholds, not one.

                        0.05 is the photograph line: below it nothing is moving
                        at all (0.0008 is a still with a moving background).

                        0.088 is the good line, and it is calibrated on two
                        human judgements rather than a study -- a listener
                        called a 0.0685 take poor and a 0.0926 take very good,
                        on the same face at the same canvas. Between the two
                        thresholds the mouth moves but under-articulates, which
                        is what --core-reuse does to a tight crop. Treat the
                        upper line as a prior, not a measurement.

  head                  how far the face centroid travels between frames, in
                        thousandths of a face width. Framing-invariant, which
                        `scene` is not -- `scene` put a perfectly good take
                        (1.97) among the damped ones, and this puts it back
                        where it belongs (3.78).

                        It is the most sensitive number here. A take at 2 steps
                        read aperture 0.0945, which is healthy, and head 1.73,
                        which is a mannequin; a viewer called a --core-reuse
                        take unnatural while aperture was still passing it.
                        Anything that reuses or skips DiT work damps this first:
                        core reuse at any step count, and 2 steps on its own,
                        all land at 1.07-2.00 against 2.98-6.39 for the recipe.

  audio r               waveform correlation against the source wav. Anchored
                        audio pins this at 0.970; anything lower means the
                        soundtrack was regenerated rather than carried.

Deliberately absent: the permutation z from lipsync_score.py. It ranked a take
with a literally frozen mouth (aperture sd 0.0008) at z=2.78, above every take
that actually articulated. When the mouth curve is near-constant its permutation
null collapses and the z becomes noise over noise. Use aperture, and listen.
"""
import sys, os, subprocess, tempfile
import numpy as np
from PIL import Image

_FPS_FLAG = None


def _fps_mode_flag():
    """ffmpeg 9 removed -vsync; older builds do not know -fps_mode.

    Worked out once per process rather than assumed, because this broke mid
    session when the machine's ffmpeg moved to 9.0.1 and every take_report call
    started failing with "Unrecognized option 'vsync'".
    """
    global _FPS_FLAG
    if _FPS_FLAG is None:
        probe = subprocess.run(["ffmpeg", "-hide_banner", "-h", "full"],
                               capture_output=True, text=True)
        _FPS_FLAG = (["-fps_mode", "passthrough"]
                     if "-fps_mode" in probe.stdout else ["-vsync", "0"])
    return _FPS_FLAG



sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _dir():   return os.environ.get("FLAT2V_DIR", "/Users/kaede/models/_wang_test")
def _ref():   return os.environ.get("FLAT2V_REF", f"{_dir()}/wang_wide34_480x640.png")
def _wav():   return os.environ.get("FLAT2V_WAV", f"{_dir()}/wang_line_6575.wav")
# Read the environment per call, not at import: a caller that imports this and
# then sets FLAT2V_REF -- which is every caller with more than one reference
# image -- otherwise silently grades against the wrong still and dies on a
# shape mismatch, or worse, does not.


def _frames(path, idxs):
    d = tempfile.mkdtemp()
    sel = "+".join(rf"eq(n\,{n})" for n in idxs)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vf", f"select={sel}",
                    *_fps_mode_flag(), f"{d}/%03d.png"], check=True)
    fs = sorted(os.listdir(d))
    out = {n: np.asarray(Image.open(f"{d}/{f}").convert("RGB")).astype(np.float32)
           for n, f in zip(idxs, fs)}
    for f in fs:
        os.unlink(f"{d}/{f}")
    os.rmdir(d)
    return out


def _audio(path):
    t = tempfile.mktemp(suffix=".raw")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-ac", "1",
                    "-ar", "16000", "-f", "f32le", t], check=True)
    a = np.fromfile(t, dtype=np.float32)
    os.unlink(t)
    return a


def report(path, last=157, ref=None, src=None, with_aperture=True):
    if ref is None:
        ref = np.asarray(Image.open(_ref()).convert("RGB")).astype(np.float32)
    if src is None:
        src = _audio(_wav())
    idxs = [0, 1, 40, 41, 80, 81, 120, 121, last]
    F = _frames(path, idxs)
    scene = np.mean([np.abs(F[b] - F[a]).mean() for a, b in ((0, 1), (40, 41), (80, 81), (120, 121))])
    a = _audio(path)
    n = min(len(a), len(src))
    aa, ss = a[:n] - a[:n].mean(), src[:n] - src[:n].mean()
    r = float(np.dot(aa, ss) / (np.linalg.norm(aa) * np.linalg.norm(ss) + 1e-9))
    ap = float("nan")
    head = float("nan")
    if with_aperture:
        try:
            from lipsync_score import mouth_curve
            m = mouth_curve(path)
            if m is not None:
                m = np.asarray(m, dtype=float)
                ap = float(np.nanstd(m))
        except Exception:
            pass
        try:
            from lipsync_score import head_track
            centroids, widths = head_track(path)
            step = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
            scale = np.nanmedian(widths)
            if scale and np.isfinite(scale):
                head = float(np.nanmean(step) / scale * 1000.0)
        except Exception:
            pass
    return dict(anchor0=float(np.abs(F[0] - ref).mean()),
                anchor_last=float(np.abs(F[last] - ref).mean()),
                scene=float(scene), aperture=ap, head=head, audio_r=r,
                mb=os.path.getsize(path) / 1e6)


def verdict(m):
    # NaN compares false against everything, so an unmeasured aperture used to
    # sail through every threshold below and come out "good". It happened for
    # real: an ffmpeg upgrade broke the landmark pass and the grader kept
    # passing takes on two missing numbers.
    for name in ("aperture", "head", "audio_r"):
        if m.get(name) != m.get(name):
            return f"NOT GRADED ({name} unmeasured)"
    if m["audio_r"] < 0.95:
        return "AUDIO LOST"
    if m["aperture"] < 0.05:
        return "MOUTH FROZEN"
    if m["aperture"] < 0.088:
        return "under-articulating"
    # A head that holds still reads as a mannequin, and a viewer says so before
    # any other number complains. Measured on the head crop: everything that
    # reuses or skips DiT work lands at 1.07-2.00 (--core-reuse at 2, 3 and 4
    # steps; 2 steps without it) and everything that does not lands at
    # 2.98-6.39. The line goes in the gap.
    if m["head"] == m["head"] and m["head"] < 2.5:
        return "head too still"
    if m["anchor0"] > 8:
        return "FIRST FRAME DRIFTED"
    if m["anchor_last"] > 12:
        return "TAIL DRIFTED"
    # Deliberately loose: 3 was calibrated on a wide shot and fires on every
    # correct tight-crop take. See the note on `scene` above.
    if m["scene"] > 8:
        return "scene unstable"
    return "good"


if __name__ == "__main__":
    W = _dir()
    ref = np.asarray(Image.open(_ref()).convert("RGB")).astype(np.float32)
    src = _audio(_wav())
    print(f"{'take':24s} {'anchor0':>8s} {'anchorN':>8s} {'scene':>7s} "
          f"{'apert':>7s} {'head':>6s} {'audio r':>8s}   verdict")
    for p in sys.argv[1:]:
        p = p if os.path.sep in p else os.path.join(W, p)
        if not os.path.exists(p):
            print(f"{os.path.basename(p):24s}  MISSING")
            continue
        try:
            m = report(p, ref=ref, src=src)
        except Exception as e:
            print(f"{os.path.basename(p):24s}  ERROR {e}")
            continue
        print(f"{os.path.basename(p):24s} {m['anchor0']:8.2f} {m['anchor_last']:8.2f} "
              f"{m['scene']:7.3f} {m['aperture']:7.4f} {m['head']:6.2f} "
              f"{m['audio_r']:8.3f}   {verdict(m)}")
