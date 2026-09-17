#!/usr/bin/env python3
"""Rank a batch of takes of the same shot, as JSON, for a program to read.

The three scorers in this directory print tables for a person. KaedeStudio
renders N seeds of a shot at low resolution and has to choose between them
without one, so this runs all three over a batch and emits one JSON document:
the numbers, a verdict per take, and an order.

The policy is the one the findings doc argues for, and each rule is here
because a take that broke it was unusable, not because it scored badly:

  * a burned-in subtitle disqualifies a take outright. It cannot be fixed
    downstream and it is the single most common failure at any canvas above a
    head crop (13/17 wide against 0/16 cropped).
  * a frozen take disqualifies a take outright. Mean |delta| between
    consecutive frames below the floor means the model returned a photograph,
    which happens most often on exactly the prompts that avoid subtitles.
  * among what survives, a shot carrying dialogue is ranked by how well the
    mouth follows the soundtrack, and a silent one by motion, preferring the
    take that moves least -- these are locked-off cameras, so movement in a
    silent shot is drift.

Nothing here is a quality judgement about the acting. It removes takes that
are broken and orders the rest by the one thing that is measurable.

    score_takes.py [--speech|--silent] [--vlm MODEL|--geometry] [--json OUT] take.mp4 ...
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                  # noqa: E402
import subtitle_rate                                # noqa: E402
import lipsync_score                                # noqa: E402

# A take whose consecutive frames differ by less than this is a photograph.
# Calibrated against a still encoded as video (0.000) and four real takes
# (5.7, 6.5, 6.8, 9.7), so the line sits an order of magnitude below anything
# that moved. It catches the unambiguous case -- a take that freezes for part
# of its length still averages well above this, and nothing here will find
# that one; the doc's "seven of fifteen froze" were found by eye.
FREEZE_FLOOR = 1.0

# Above this fraction of sampled frames carrying a caption the take is gone.
# Not zero: the detector fires on the occasional lit edge, and one frame in
# forty is noise rather than a subtitle.
SUBTITLE_LIMIT = 5.0


# Motion is sampled on a forced square so the reshape needs no probe, and at a
# fixed rate so the number means the same thing on a 3-second take and a
# 10-second one. Aspect distortion is irrelevant to a difference metric.
MOTION_EDGE = 128
MOTION_FPS = 6


def _frame_motion(path: str) -> float:
    """Mean |delta| between consecutive sampled frames, 0-255."""
    command = [
        "ffmpeg", "-v", "error", "-i", path,
        "-vf", f"scale={MOTION_EDGE}:{MOTION_EDGE},fps={MOTION_FPS}",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    raw = subprocess.run(command, capture_output=True).stdout
    pixels = MOTION_EDGE * MOTION_EDGE
    count = len(raw) // pixels
    if count < 2:
        return 0.0
    data = np.frombuffer(raw[:count * pixels], dtype=np.uint8)
    frames = data.reshape(count, MOTION_EDGE, MOTION_EDGE).astype(np.int16)
    return float(np.abs(np.diff(frames, axis=0)).mean())


def _has_audio(path: str) -> bool:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    return bool(probe.stdout.strip())


def score_take(path: str, speech: bool | None = None,
               vlm_model: str | None = None) -> dict:
    result = {
        "path": os.path.abspath(path),
        "name": os.path.basename(path),
        "subtitle_pct": None,
        "motion": None,
        "lipsync_r": None,
        "lipsync_lag": None,
        "lipsync_z": None,
        "verdict": "ok",
        "reasons": [],
    }
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        result["verdict"] = "missing"
        result["reasons"].append("no file")
        return result

    try:
        result["subtitle_pct"] = round(subtitle_rate.rate(path, model=vlm_model), 1)
    except Exception as exc:                       # a scorer must not take the batch down
        result["reasons"].append(f"subtitle scorer failed: {exc}")
    try:
        result["motion"] = round(_frame_motion(path), 3)
    except Exception as exc:
        result["reasons"].append(f"motion scorer failed: {exc}")

    wants_speech = _has_audio(path) if speech is None else speech
    if wants_speech:
        try:
            scored = lipsync_score.score(path)
            if scored is None:
                result["reasons"].append("no face detected")
            else:
                true, lag, _mu, _mx, zscore = scored
                result["lipsync_r"] = round(float(true), 3)
                result["lipsync_lag"] = int(lag)
                result["lipsync_z"] = round(float(zscore), 2)
        except Exception as exc:
            result["reasons"].append(f"lipsync scorer failed: {exc}")

    if result["subtitle_pct"] is not None and result["subtitle_pct"] > SUBTITLE_LIMIT:
        result["verdict"] = "subtitled"
        result["reasons"].append(
            f"{result['subtitle_pct']:.0f}% of frames carry a caption")
    elif result["motion"] is not None and result["motion"] < FREEZE_FLOOR:
        result["verdict"] = "frozen"
        result["reasons"].append(
            f"consecutive frames differ by {result['motion']:.2f}, "
            "which is a photograph")
    return result


def rank(takes: list[dict], speech: bool) -> list[dict]:
    """Order the survivors; rejected takes keep their place at the back."""
    def key(take: dict):
        if take["verdict"] != "ok":
            return (1, 0.0)
        if speech:
            # z-score first: it is the one that survived a time-reversed
            # control, where raw correlation did not.
            return (0, -(take["lipsync_z"] or -99.0))
        return (0, take["motion"] or 99.0)

    ordered = sorted(takes, key=key)
    for position, take in enumerate(ordered):
        take["rank"] = position + 1 if take["verdict"] == "ok" else None
    return ordered


def main(argv: list[str]) -> int:
    argv = list(argv)
    out_path = ""
    speech: bool | None = None
    if "--json" in argv:
        index = argv.index("--json")
        out_path = argv[index + 1]
        del argv[index:index + 2]
    if "--speech" in argv:
        argv.remove("--speech")
        speech = True
    if "--silent" in argv:
        argv.remove("--silent")
        speech = False
    vlm_model = None
    if "--vlm" in argv:
        index = argv.index("--vlm")
        vlm_model = argv[index + 1]
        del argv[index:index + 2]
    if "--geometry" in argv:
        argv.remove("--geometry")
        vlm_model = ""
    if not argv:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2

    takes = [score_take(path, speech, vlm_model) for path in argv]
    wants_speech = speech if speech is not None else any(
        take["lipsync_z"] is not None for take in takes)
    ordered = rank(takes, wants_speech)
    document = {
        "speech": wants_speech,
        "subtitle_limit": SUBTITLE_LIMIT,
        "freeze_floor": FREEZE_FLOOR,
        "takes": ordered,
        "chosen": next((t["path"] for t in ordered if t["verdict"] == "ok"), None),
    }
    text = json.dumps(document, indent=2, ensure_ascii=False)
    if out_path:
        with open(out_path, "w") as handle:
            handle.write(text)
        print(f"wrote {out_path}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
