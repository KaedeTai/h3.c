"""Drive the local h3.c binary. Standard library only, no project imports.

Kept free of MoneyPrinterTurbo types on purpose: this half is the part that has
to be testable on the machine that actually owns the GPU, where installing
moviepy and streamlit to check a subprocess call would be absurd.

h3 is MiniMax-H3 compiled to C + Metal (github.com/KaedeTai/h3.c). The T2VA
checkpoint turns a prompt into video *and* a soundtrack; this module always
strips the soundtrack, because the caller lays its own narration over the cut
and a second voice underneath it is never wanted.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass

# h3 only accepts frame counts on this lattice (the video VAE's temporal
# grouping): 5, 22, 39, ... Asking for anything else is refused outright.
_FRAME_BASE = 5
_FRAME_STEP = 17
_FPS = 24

# The VAE downsamples 16x and the patch is 2x2, so one token covers 32x32
# pixels and both edges must be multiples of 32. These are the closest such
# sizes to each aspect at roughly 0.18 MP, which is the point measured at
# ~75 s for a 10 s clip on an M5 Max; larger canvases cost quadratically more
# in attention and start drawing captions.
_ASPECT_SIZES = {
    "9:16": (320, 576),
    "16:9": (576, 320),
    "1:1": (448, 448),
}

_DEFAULT_PROMPT_TEMPLATE = (
    "({term}, cinematic stock footage, natural light, shallow depth of field, "
    "slow steady camera, ambient sound only)"
)


class H3RunnerError(RuntimeError):
    """h3 could not produce a usable clip.

    There is deliberately no "unconfirmed task" variant of this error, which
    every paid provider in this project needs: local generation costs nothing,
    so a retry is always safe and never has to be reconciled against a bill.
    """


@dataclass(frozen=True)
class H3Clip:
    path: str
    duration: float
    frames: int
    width: int
    height: int
    seed: int
    prompt: str
    wall_seconds: float


def legal_frames(seconds: float, *, max_frames: int = 243) -> int:
    """Smallest frame count h3 accepts that covers `seconds`."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise H3RunnerError(f"clip duration must be positive and finite, got {seconds!r}")
    wanted = math.ceil(seconds * _FPS)
    k = max(0, math.ceil((wanted - _FRAME_BASE) / _FRAME_STEP))
    frames = _FRAME_BASE + _FRAME_STEP * k
    return min(frames, max_frames)


def canvas_for_aspect(aspect: str, override: str = "") -> tuple[int, int]:
    """Pixel size for an aspect string, or an explicit `WIDTHxHEIGHT` override."""
    if override:
        try:
            w_text, h_text = override.lower().split("x", 1)
            width, height = int(w_text), int(h_text)
        except ValueError as exc:
            raise H3RunnerError(
                f"h3 resolution override must look like 576x320, got {override!r}"
            ) from exc
        if width % 32 or height % 32:
            raise H3RunnerError(
                f"h3 canvas edges must be multiples of 32, got {width}x{height}"
            )
        return width, height
    try:
        return _ASPECT_SIZES[aspect]
    except KeyError as exc:
        raise H3RunnerError(f"unsupported aspect {aspect!r}") from exc


def build_prompt(term: str, template: str = "") -> str:
    """Wrap a bare search term in visual direction.

    The caller hands us one to three English words ("night market food"), which
    on its own tells the model nothing about the shot. Direction written in
    English conditions the picture without being spoken; Mandarin in the prompt
    would be read aloud, so nothing here is ever in the spoken language.
    """
    cleaned = " ".join(str(term or "").split())
    if not cleaned:
        raise H3RunnerError("search term must not be empty")
    return (template or _DEFAULT_PROMPT_TEMPLATE).format(term=cleaned)


def _strip_audio(src: str, dst: str, ffmpeg: str) -> None:
    """Remux without the soundtrack. Copies the video stream, so it is cheap."""
    result = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", src, "-c:v", "copy", "-an", dst],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        raise H3RunnerError(
            f"could not strip the generated soundtrack: {result.stderr.strip()[:400]}"
        )


def render_clip(
    *,
    prompt: str,
    output_path: str,
    seconds: float,
    aspect: str = "9:16",
    resolution_override: str = "",
    steps: int = 3,
    seed: int = 42,
    binary: str = "",
    model_dir: str = "./MiniMax-H3-turbo-int8",
    timeout: float = 900.0,
    keep_audio: bool = False,
) -> H3Clip:
    """Generate one clip with h3 and return where it landed.

    One process per clip. The model load is 3-8 s against 40-75 s of denoising,
    so a resident worker would save under 10%; it is worth doing later, but not
    at the cost of a REPL protocol in the first version of this provider.
    """
    h3_binary = binary or os.environ.get("H3_BINARY", "")
    if not h3_binary:
        raise H3RunnerError("h3 binary path is not configured")
    h3_binary = os.path.abspath(os.path.expanduser(h3_binary))
    if not (os.path.isfile(h3_binary) and os.access(h3_binary, os.X_OK)):
        raise H3RunnerError(f"h3 binary is missing or not executable: {h3_binary}")

    ffmpeg = shutil.which("ffmpeg") or ""
    if not keep_audio and not ffmpeg:
        raise H3RunnerError("ffmpeg is required to strip the generated soundtrack")

    width, height = canvas_for_aspect(aspect, resolution_override)
    frames = legal_frames(seconds)
    if steps <= 0:
        raise H3RunnerError(f"steps must be positive, got {steps}")

    workdir = os.path.dirname(h3_binary)
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    raw_path = output_path if keep_audio else f"{output_path}.withaudio.mp4"

    env = dict(os.environ)
    env["H3_DIT_VARIANT"] = "T2VA"
    # Without this the video VAE decode costs more than the whole transformer:
    # 69 s against 17 s on a 10 s clip. It is the single most expensive flag to
    # forget, which is why it is set here rather than left to config.
    env["H3_VAE_INT8_FFN"] = "1"
    for stale in ("H3_AUDIO_ANCHOR", "H3_REF2VA_ANCHOR", "H3_REF2VA_ANCHOR_LAST"):
        env.pop(stale, None)

    command = [
        h3_binary, "-d", model_dir, "-p", prompt, "-o", raw_path,
        "--width", str(width), "--height", str(height),
        "--frames", str(frames), "--steps", str(steps),
        "--seed", str(seed), "--use-int8-row-fc2",
    ]

    started = time.time()
    try:
        result = subprocess.run(
            command, cwd=workdir, env=env, capture_output=True, text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise H3RunnerError(
            f"h3 did not finish within {timeout:.0f}s for prompt {prompt[:60]!r}"
        ) from exc
    wall = time.time() - started

    if result.returncode != 0:
        tail = (result.stdout or "").replace("\r", "\n").strip().splitlines()[-3:]
        raise H3RunnerError(
            f"h3 exited with {result.returncode}: {' / '.join(tail) or 'no output'}"
        )
    if not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
        raise H3RunnerError(f"h3 reported success but wrote no video to {raw_path}")

    if not keep_audio:
        _strip_audio(raw_path, output_path, ffmpeg)
        os.remove(raw_path)

    return H3Clip(
        path=output_path,
        duration=frames / _FPS,
        frames=frames,
        width=width,
        height=height,
        seed=seed,
        prompt=prompt,
        wall_seconds=wall,
    )


def new_clip_path(directory: str, prefix: str = "h3") -> str:
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"{prefix}-{uuid.uuid4().hex[:12]}.mp4")
