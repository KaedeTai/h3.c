#!/usr/bin/env python3
"""Choose segment boundaries that land in the pauses, not mid-vowel.

The clock cut is the obvious plan and it is only accidentally good. A boundary
is the one frame segment N+1 has to reproduce from a still image, and a still
of a wide-open mouth mid-vowel is the hardest thing this pipeline reproduces --
the same reason a photograph anchor is easy and a talking frame is not.

Segment lengths are quantised: h3 reads video as a head chunk of 5 real frames
plus chunks of 17, so legal lengths are 5+17k -- about 0.71 s apart. That
quantisation is the whole opportunity. It means every boundary carries roughly
+/-0.35 s of slack before the next legal length, which is almost always enough
to reach a breath.

Planning is a shortest path, not a greedy walk. Greedy takes the nearest pause
and then has to pay for it at every later boundary; the audio is short enough
(a 40 Hz lattice, because every legal span is a whole number of audio latent
frames) that the exact answer is a few thousand relaxations.

    cost = quietness at the boundary  +  LAMBDA * how far the span is from target

Quietness is RMS in a 200 ms window around the cut, over the clip's mean RMS,
so 0 is a real pause and 2 is a stressed vowel. The length term is asymmetric:
a shorter segment costs an extra fixed overhead per second of video, a longer
one costs quadratically more attention, and the longer side is the worse of the
two.
"""
import subprocess, numpy as np

FPS, AUDIO_LATENT_FPS = 24, 40
SR, HOP = 16000, 0.010
WINDOW = 0.10          # seconds either side of a boundary
LAMBDA = 1.2           # quietness units per unit of relative length deviation
LONG_WEIGHT = 1.6      # overshooting the target costs more than undershooting
PAD_WEIGHT = 0.8       # per second of silence padded onto the last segment


def legal_frames(n):
    return 5 + max(1, (int(n) - 5) // 17) * 17


def span_units(frames):
    """Audio latent frames -- the lattice everything is planned on."""
    return int(round(frames * AUDIO_LATENT_FPS / FPS))


def envelope(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1",
                          "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    h = int(SR * HOP)
    n = len(x) // h
    if n == 0:
        raise SystemExit(f"{path} is too short to plan against")
    r = np.sqrt((x[:n * h].reshape(n, h) ** 2).mean(1) + 1e-12)
    return r / max(r.mean(), 1e-9)


def quietness(q, seconds):
    i = int(seconds / HOP)
    w = int(WINDOW / HOP)
    lo, hi = max(0, i - w), min(len(q), i + w)
    return float(q[lo:hi].mean()) if hi > lo else 0.0


def candidate_frames(target_frames, shorter=3, longer=1):
    k = (legal_frames(target_frames) - 5) // 17
    return [5 + 17 * j for j in range(max(2, k - shorter), k + longer + 1)]


def plan(voice, target_frames=158, shorter=3, longer=1, lam=LAMBDA, quiet=None):
    """Return (frames_per_segment, report_rows). Exact shortest path."""
    q = envelope(voice) if quiet is None else quiet
    total_u = int(np.ceil(len(q) * HOP * AUDIO_LATENT_FPS))
    target_u = span_units(legal_frames(target_frames))
    options = [(f, span_units(f)) for f in candidate_frames(target_frames, shorter, longer)]
    min_u = min(u for _, u in options)

    def length_cost(u):
        d = (u - target_u) / target_u
        return lam * (LONG_WEIGHT * d if d > 0 else -d)

    INF = float("inf")
    best = {0: 0.0}
    back = {}
    order = [0]
    seen = {0}
    head = 0
    # forward relaxation over a lattice that only ever moves right
    while head < len(order):
        u = order[head]; head += 1
        base = best[u]
        for f, du in options:
            v = u + du
            if v > total_u + min_u:            # never pad by a whole segment
                continue
            c = base + length_cost(du)
            if v < total_u:
                c += quietness(q, v / AUDIO_LATENT_FPS)
            else:
                c += PAD_WEIGHT * (v - total_u) / AUDIO_LATENT_FPS
            if c < best.get(v, INF):
                best[v] = c; back[v] = (u, f)
            if v not in seen:
                seen.add(v); order.append(v)
        order.sort()                            # keep the frontier monotone

    ends = [v for v in best if v >= total_u]
    if not ends:
        raise SystemExit("no legal segmentation reaches the end of the clip")
    end = min(ends, key=lambda v: best[v])
    frames, u = [], end
    while u:
        u, f = back[u]
        frames.append(f)
    frames.reverse()

    rows, at = [], 0
    for i, f in enumerate(frames):
        at += span_units(f)
        rows.append({"segment": i + 1, "frames": f,
                     "span": span_units(f) / AUDIO_LATENT_FPS,
                     "cut_at": at / AUDIO_LATENT_FPS,
                     "quiet": (quietness(q, at / AUDIO_LATENT_FPS)
                               if i < len(frames) - 1 else None)})
    return frames, rows


def clock_plan(voice, target_frames=158, quiet=None):
    """What the fixed-length planner would have done, for comparison."""
    q = envelope(voice) if quiet is None else quiet
    total = len(q) * HOP
    f = legal_frames(target_frames)
    span = span_units(f) / AUDIO_LATENT_FPS
    full = int(total // span)
    frames = [f] * full
    rest = total - full * span
    if rest > 0.4:
        t = legal_frames(rest * FPS)
        if span_units(t) / AUDIO_LATENT_FPS >= 2.0:
            frames.append(t)
    rows, at = [], 0
    for i, fr in enumerate(frames):
        at += span_units(fr)
        rows.append({"segment": i + 1, "frames": fr,
                     "span": span_units(fr) / AUDIO_LATENT_FPS,
                     "cut_at": at / AUDIO_LATENT_FPS,
                     "quiet": (quietness(q, at / AUDIO_LATENT_FPS)
                               if i < len(frames) - 1 else None)})
    return frames, rows


def show(name, rows):
    inner = [r["quiet"] for r in rows if r["quiet"] is not None]
    print(f"  {name:8s} {len(rows)} segments, "
          f"{sum(r['span'] for r in rows):6.2f}s, "
          f"cuts at " + ", ".join(f"{r['cut_at']:.2f}s({r['quiet']:.2f})"
                                  for r in rows if r["quiet"] is not None))
    if inner:
        print(f"           mean boundary loudness {np.mean(inner):.3f}, "
              f"worst {max(inner):.3f}")


if __name__ == "__main__":
    import sys
    voice = sys.argv[1]
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 158
    q = envelope(voice)
    print(f"{voice}  {len(q)*HOP:.2f}s")
    show("clock", clock_plan(voice, target, quiet=q)[1])
    show("silence", plan(voice, target, quiet=q)[1])
