#!/usr/bin/env python3
"""Is the mouth following the soundtrack?

Transcribing the audio says what was said, not whether the face said it.

Two earlier attempts failed and are worth recording. A fixed box in the lower
third scored a visibly synced take at -0.086 -- the box drifts off the mouth as
the head moves. Replacing it with a Haar face box and a dark-pixel fraction got
the lag right (~0) but only 0.25-0.40 correlation, and against a time-reversed
audio control the gap ranged from -0.009 to +0.355: not discriminating. The
proxy was the problem, so use real landmarks.

mediapipe FaceMesh gives the inner lip contour directly. Aperture is the
vertical lip gap over the mouth width, which is scale- and distance-invariant,
so head movement stops leaking into the signal.

Calibrate every number against time-reversed audio: it keeps the spectral
statistics and destroys the alignment, so the forward-minus-reversed gap is the
part that means "in sync".
"""
import subprocess, sys, os, tempfile, shutil, wave
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


_FPS_FLAG = None


def _fps_mode_flag():
    """ffmpeg 9 removed -vsync; older builds do not know -fps_mode."""
    global _FPS_FLAG
    if _FPS_FLAG is None:
        probe = subprocess.run(["ffmpeg", "-hide_banner", "-h", "full"],
                               capture_output=True, text=True)
        _FPS_FLAG = (["-fps_mode", "passthrough"]
                     if "-fps_mode" in probe.stdout else ["-vsync", "0"])
    return _FPS_FLAG


FPS = 24.0
# inner lip contour: upper and lower midpoints, and the two corners
UPPER, LOWER, LEFT, RIGHT = 13, 14, 78, 308
MODEL = os.path.expanduser("~/models/_mp/face_landmarker.task")


def _landmarker():
    """mediapipe 0.10.35 ships only the Tasks API, so this needs the model file."""
    if not os.path.exists(MODEL):
        raise SystemExit(
            f"missing {MODEL}\n  curl -sSL -o {MODEL} https://storage.googleapis.com"
            "/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
            "face_landmarker.task")
    return mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL),
            running_mode=mp_vision.RunningMode.IMAGE, num_faces=1))


def head_track(mp4):
    """Face centroid and face width per frame, in pixels.

    Whole-frame |delta| between consecutive frames -- `scene` in take_report --
    is not a head-motion measure: it mixes the head with the background and its
    size depends on how much of the frame the head fills. This does not. The
    centroid is the mean of all 478 landmarks and the width is their bounding
    box, so displacement expressed in face-widths is invariant to both framing
    and canvas.

    Returns (centroids [T,2], widths [T]) with NaN where no face was found.
    """
    d = tempfile.mkdtemp()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mp4,
                    *_fps_mode_flag(), f"{d}/f_%05d.png"], check=True)
    mesh = _landmarker()
    centroids, widths = [], []
    for fn in sorted(os.listdir(d)):
        img = cv2.cvtColor(cv2.imread(f"{d}/{fn}"), cv2.COLOR_BGR2RGB)
        res = mesh.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
        os.unlink(f"{d}/{fn}")
        if not res.face_landmarks:
            centroids.append((np.nan, np.nan)); widths.append(np.nan); continue
        lm = res.face_landmarks[0]
        h, w = img.shape[:2]
        xs = np.array([p.x for p in lm]) * w
        ys = np.array([p.y for p in lm]) * h
        centroids.append((xs.mean(), ys.mean()))
        widths.append(xs.max() - xs.min())
    shutil.rmtree(d, ignore_errors=True)
    return np.array(centroids), np.array(widths)


def mouth_curve(mp4):
    d = tempfile.mkdtemp()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mp4,
                    *_fps_mode_flag(), f"{d}/f_%05d.png"], check=True)
    mesh = _landmarker()
    vals = []
    for fn in sorted(os.listdir(d)):
        img = cv2.cvtColor(cv2.imread(f"{d}/{fn}"), cv2.COLOR_BGR2RGB)
        res = mesh.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
        if not res.face_landmarks:
            vals.append(np.nan); continue
        lm = res.face_landmarks[0]
        h, w = img.shape[:2]
        p = lambda i: np.array([lm[i].x * w, lm[i].y * h])
        gap = np.linalg.norm(p(UPPER) - p(LOWER))
        width = np.linalg.norm(p(LEFT) - p(RIGHT)) + 1e-6
        vals.append(float(gap / width))          # aperture, scale-invariant
    mesh.close()
    shutil.rmtree(d, ignore_errors=True)
    v = np.array(vals)
    if np.isnan(v).all(): return None
    idx = np.arange(len(v)); good = ~np.isnan(v)
    return np.interp(idx, idx[good], v[good])


def speech_curve(mp4, n):
    wav = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mp4, "-vn", "-ac", "1",
                    "-ar", "16000", "-c:a", "pcm_s16le", wav], check=True)
    with wave.open(wav) as f:
        y = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(float)
        sr = f.getframerate()
    os.unlink(wav)
    hop = sr / FPS
    return np.array([np.sqrt((y[int(i*hop):int((i+1)*hop)]**2).mean() + 1e-9)
                     for i in range(n)])


def z(x):
    x = x - x.mean()
    return x / (x.std() + 1e-9)


def best_corr(m, s, max_lag=8):
    b = (-2.0, 0)
    for lag in range(-max_lag, max_lag + 1):
        a = m[max(0, lag):len(m) + min(0, lag)]
        c = s[max(0, -lag):len(s) - max(0, lag)]
        n = min(len(a), len(c))
        if n < 20: continue
        r = float(np.dot(z(a[:n]), z(c[:n])) / n)
        if r > b[0]: b = (r, lag)
    return b


def score(mp4, n_null=200, seed=0):
    """True alignment against a permutation null.

    Time-reversal turned out to be a bad control: these clips are
    silence-speech-silence, an envelope roughly symmetric in time, so the
    reversed audio still correlated 0.41 with the mouth and buried the signal.
    Random circular shifts keep the envelope exactly and destroy only the
    alignment, which is what should be null.

    Validated first that the mouth curve itself is sound: on a real take its
    four most-closed frames sit at audio RMS 0.005-0.009 and its four most-open
    at 0.23-0.82, so aperture does track the soundtrack.
    """
    m = mouth_curve(mp4)
    if m is None: return None
    s = speech_curve(mp4, len(m))
    true, lag = best_corr(m, s)
    rng = np.random.default_rng(seed)
    n = len(s)
    null = []
    for _ in range(n_null):
        k = int(rng.integers(n // 8, n - n // 8))     # never a near-zero shift
        null.append(best_corr(m, np.roll(s, k))[0])
    null = np.array(null)
    zscore = (true - null.mean()) / (null.std() + 1e-9)
    return true, lag, float(null.mean()), float(null.max()), float(zscore)


if __name__ == "__main__":
    print(f"{'take':30s} {'r':>7s} {'lag':>4s} {'null mu':>8s} {'null max':>9s} {'z':>7s}   reading")
    for p in sys.argv[1:]:
        r = score(p)
        if r is None:
            print(f"{os.path.basename(p):30s}   no face detected"); continue
        true, lag, mu, mx, zs = r
        v = ("in sync" if zs > 3 and abs(lag) <= 2 else
             "offset" if zs > 3 else
             "weak" if zs > 1.5 else "NOT synced")
        print(f"{os.path.basename(p):30s} {true:7.3f} {lag:4d} {mu:8.3f} {mx:9.3f} {zs:7.2f}   {v}")
