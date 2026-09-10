#!/usr/bin/env python3
"""Does this take have a burned-in subtitle?

Reports the fraction of sampled frames carrying one, not a pixel percentage.
The pixel percentage was the first attempt and it was useless: it read 3.20% on
a head-crop take with no subtitle at all and 2.93% on a wide take that plainly
had one, because the colour test alone fires on the lab coat's lit edge and on
the framed certificates behind the shoulder. Looking at the mask overlay was
what showed it -- the number on its own was confidently wrong in both
directions.

So the colour test (from tools/desubtitle.py: saturated yellow, or white with a
dark outline nearby) is only the first stage. What actually separates a caption
from a coat edge is geometry: a subtitle is a horizontally extended run of
glyphs in a narrow band of rows near the middle of the width, while a coat edge
is a long vertical boundary and the certificates sit against the frame edge.
Stage two therefore keeps only the central width, asks each row whether masked
pixels are spread across a real span of it, and requires enough consecutive such
rows to make a line of text.

    subtitle_rate.py TAKE.mp4 [...] [--dump DIR]
"""
import os, subprocess, sys, tempfile
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

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



SAMPLE = 12          # frames per clip, evenly spaced
CENTRE = 0.70        # ignore the left/right margins: lapel, certificates
SPAN = 0.18          # a text row covers at least this much of the centre
ROWS = 7             # and a line of glyphs is at least this many rows tall


def glyph_mask(frames):
    R, G, B = frames[..., 0], frames[..., 1], frames[..., 2]
    lum = frames.mean(3)
    dark = lum < 90
    near_dark = np.stack([binary_dilation(d, iterations=4) for d in dark])
    yellow = ((G - B) > 40) & (G > 90) & (R > 90)
    white = (lum > 215) & (frames.max(3) - frames.min(3) < 30) & near_dark
    mask = yellow | white
    mask[:, :int(frames.shape[1] * 0.55)] = False   # lower band only
    return mask


TRANSITIONS = 8      # glyphs switch on/off many times across a row; a coat does not


def has_subtitle(mask_frame):
    """Geometry stage: a wide, banded run of glyph rows near the centre.

    A white lab coat with a dark tie passes the colour stage across the whole
    lower band at 480x640 -- white, next to dark -- and lit up every frame of a
    clean take as SUBTITLED. What separates a line of text from a coat is not
    how much of the row is "text-coloured" but how often it switches: glyphs
    alternate on and off dozens of times across a row, cloth once or twice.
    """
    h, w = mask_frame.shape
    lo, hi = int(w * (1 - CENTRE) / 2), int(w * (1 + CENTRE) / 2)
    roi = mask_frame[:, lo:hi]
    wide = roi.sum(1) >= (hi - lo) * SPAN
    flips = (roi[:, 1:] != roi[:, :-1]).sum(1)
    rows = wide & (flips >= TRANSITIONS)
    run = best = 0
    for flag in rows:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best >= ROWS


def rate(path, dump=None):
    work = tempfile.mkdtemp()
    n = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames",
                            "-select_streams", "v:0", "-show_entries",
                            "stream=nb_read_frames", "-of", "csv=p=0", path],
                           capture_output=True, text=True).stdout.strip() or 0)
    if not n:
        return float("nan")
    step = max(1, n // SAMPLE)
    sel = "+".join(rf"eq(n\,{i})" for i in range(0, n, step))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vf",
                    f"select={sel}", *_fps_mode_flag(), f"{work}/%03d.png"], check=True)
    names = sorted(os.listdir(work))
    frames = np.stack([np.asarray(Image.open(f"{work}/{x}").convert("RGB"))
                       for x in names]).astype(np.float32)
    mask = glyph_mask(frames)
    hits = np.array([has_subtitle(m) for m in mask])
    if dump is not None:
        os.makedirs(dump, exist_ok=True)
        pick = int(hits.argmax()) if hits.any() else \
               int(mask.reshape(len(frames), -1).sum(1).argmax())
        over = frames[pick].copy()
        over[mask[pick]] = (255, 0, 0)
        Image.fromarray(over.astype(np.uint8)).save(
            os.path.join(dump, os.path.basename(path).replace(".mp4", ".png")))
    for x in names:
        os.unlink(f"{work}/{x}")
    os.rmdir(work)
    return float(hits.mean() * 100)


if __name__ == "__main__":
    args = sys.argv[1:]
    dump = None
    if "--dump" in args:
        i = args.index("--dump"); dump = args[i + 1]; del args[i:i + 2]
    print(f"{'take':30s} {'frames with a subtitle':>23s}")
    for p in args:
        r = rate(p, dump)
        print(f"{os.path.basename(p):30s} {r:21.0f} %  {'SUBTITLED' if r > 0 else ''}")
