"""Local MiniMax-H3 video generation through the h3.c binary.

This is the same model the ``metaso_minimax`` provider reaches over the
network, run on the operator's own GPU instead. The difference that matters to
the pipeline is not speed but billing: a clip costs nothing, so none of the
paid-provider machinery — unconfirmed-task errors, stop-on-first-failure, "do
not retry because it may already have been charged" — applies here. Failures
are ordinary failures and the loop simply moves to the next search term.

Requires github.com/KaedeTai/h3.c built, and its model directory present. See
``[app] h3_local_*`` in config.example.toml.
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.services import h3_runner
from app.services.h3_runner import H3RunnerError

# Re-exported so material.py can catch one name without importing the runner.
H3LocalError = H3RunnerError

DEFAULT_STEPS = 3
DEFAULT_MIN_DURATION = 3
DEFAULT_MAX_DURATION = 10
DEFAULT_TIMEOUT = 900.0


def _setting(name: str, default: Any = "") -> Any:
    value = config.app.get(name, default)
    if isinstance(value, str):
        return value.strip()
    return value


def is_configured() -> bool:
    """True when a usable h3 binary is on disk.

    Checked before the pipeline spends time on a script, the same way the paid
    providers check their API keys — a missing binary should fail in preflight,
    not forty seconds into the first clip.
    """
    binary = str(_setting("h3_local_binary") or os.environ.get("H3_BINARY", ""))
    if not binary:
        return False
    binary = os.path.abspath(os.path.expanduser(binary))
    return os.path.isfile(binary) and os.access(binary, os.X_OK)


def _clamp_duration(minimum_duration: int) -> float:
    low = float(_setting("h3_local_min_duration", DEFAULT_MIN_DURATION) or DEFAULT_MIN_DURATION)
    high = float(_setting("h3_local_max_duration", DEFAULT_MAX_DURATION) or DEFAULT_MAX_DURATION)
    if low > high:
        raise H3LocalError(
            f"h3_local_min_duration ({low}) must not exceed h3_local_max_duration ({high})"
        )
    try:
        wanted = float(minimum_duration)
    except (TypeError, ValueError) as exc:
        raise H3LocalError(f"clip duration must be a number, got {minimum_duration!r}") from exc
    return max(low, min(high, wanted))


def generate_videos(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    *,
    output_dir: str = "",
    seed: int = 42,
) -> list[MaterialInfo]:
    """Render one clip for one search term and return it as local material.

    Signature matches ``volcengine_seedance.generate_videos`` and friends so the
    dispatcher treats it the same way, except that ``url`` is already a path on
    disk: there is nothing to download, so the caller must not try.
    """
    binary = str(_setting("h3_local_binary") or os.environ.get("H3_BINARY", ""))
    if not binary:
        raise H3LocalError("h3_local requires h3_local_binary in config")

    aspect = VideoAspect(video_aspect)
    seconds = _clamp_duration(minimum_duration)
    prompt = h3_runner.build_prompt(
        search_term, str(_setting("h3_local_prompt_template") or "")
    )
    directory = output_dir or os.path.join(os.getcwd(), "storage", "cache_videos")
    path = h3_runner.new_clip_path(directory, prefix="h3")

    clip = h3_runner.render_clip(
        prompt=prompt,
        output_path=path,
        seconds=seconds,
        aspect=aspect.value,
        resolution_override=str(_setting("h3_local_resolution") or ""),
        steps=int(_setting("h3_local_steps", DEFAULT_STEPS) or DEFAULT_STEPS),
        seed=seed,
        binary=binary,
        model_dir=str(_setting("h3_local_model_dir") or "./MiniMax-H3-turbo-int8"),
        timeout=float(_setting("h3_local_run_timeout", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT),
    )
    logger.info(
        f"h3 rendered {clip.frames} frames at {clip.width}x{clip.height} "
        f"({clip.duration:.2f}s) in {clip.wall_seconds:.1f}s: {search_term}"
    )
    return [
        MaterialInfo(
            provider="h3_local",
            url=clip.path,
            duration=int(clip.duration),
            source_info={
                "provider": "h3_local",
                "search_term": search_term,
                "asset_id": os.path.basename(clip.path),
                "width": clip.width,
                "height": clip.height,
            },
        )
    ]
