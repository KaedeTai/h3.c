#!/usr/bin/env python3
"""One resident h3 process, shared by everything that wants a clip.

Every driver in this repo has so far spawned its own `h3 -p ...` per clip,
which is honest and pays the same toll every time: start, mmap the weights,
load the tokenizer and text encoder, build the prepared DiT and the video
decoder, and only then denoise. On an M5 Max that toll is about 14.6 s against
a 75 s ten-second clip -- a fifth of the wall clock spent redoing what the
previous process had already done and then thrown away.

h3 already knows how to keep all of it: the REPL holds the prepared DiT, the
video decoder and the text embeddings in a cache (`!cache` reports them), and
h3.c is careful to REBIND a cached DiT on a prompt change rather than reload
it. What was missing was something to drive that REPL from a program instead
of a keyboard. This is that -- plus a Unix-socket server, so tools/mpt and
KaedeStudio share the one 10 GiB process instead of holding one each.

What cannot change without a restart is the environment. H3_DIT_VARIANT is
read when the DiT path is resolved and H3_VAE_INT8_FFN when the VAE loads,
and both results are then cached under keys that do not mention them, so a
process that has loaded T2VA can never be asked for FL2VA and a process that
forgot H3_VAE_INT8_FFN will decode at 69 s instead of 17 s for as long as it
lives. Those knobs are therefore the profile, and a job asking for a different
profile stops the child and starts another.

Protocol notes, for whoever has to debug this at 2am:

  * h3's stdout is the transcript and its stderr is the progress bar. The
    worker waits on stdout only.
  * A failed generation prints nothing at all to stdout -- it writes the error
    to stderr and the REPL loops. So every job is followed by `!cache`, whose
    reply is a line starting with "Cache:". That line is the fence: whatever
    arrived before it is the whole of this job's output, and if no "Done -> "
    came with it the job failed. It doubles as the residency check.
  * h3_cli.c line-buffers a piped stdout for exactly this reason. An older
    binary block-buffers it and every job here will time out.

Standard library only. This file is both a library and a command:

    h3_worker.py serve  --socket ~/.h3/worker.sock -d ./MiniMax-H3-turbo-int8
    h3_worker.py render --prompt "..." --output out.mp4 --seconds 10
    h3_worker.py status
    h3_worker.py stop
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass

DEFAULT_SOCKET = os.path.expanduser("~/.h3/worker.sock")
DEFAULT_MODEL = "./MiniMax-H3-turbo-int8"
DEFAULT_IDLE = 900.0

_FPS = 24
_FRAME_BASE = 5
_FRAME_STEP = 17

_DONE = re.compile(r"^Done -> (.+) \[([0-9.]+)s\]$")
_FENCE = "Cache:"

# Knobs this worker owns. Anything the caller does not set is REMOVED from the
# child's environment rather than inherited, so a stale H3_AUDIO_ANCHOR in the
# operator's shell cannot quietly change what a job means.
_MANAGED_ENV = (
    "H3_DIT_VARIANT",
    "H3_VAE_INT8_FFN",
    "H3_AUDIO_ANCHOR",
    "H3_REF2VA_ANCHOR",
    "H3_REF2VA_ANCHOR_LAST",
    "H3_SKIP_ATTN",
    "H3_NAX",
    "H3_SIGMA_BASE",
    "H3_BENCH_SKIP_INT8_QUANT",
    "H3_STEEL_ATTN",
    # LoRAs are merged into the weights at load, so they are part of the
    # profile: a job with a different H3_LORA restarts the child.
    "H3_LORA",
    "H3_LORA_DIR",
    "H3_LORA_ROUND",
)


class H3WorkerError(RuntimeError):
    """The worker could not produce a clip. Retrying is always safe."""


def legal_frames(seconds: float, *, max_frames: int = 243) -> int:
    """Smallest frame count h3 accepts that covers `seconds` (5, 22, 39, ...)."""
    wanted = int(-(-float(seconds) * _FPS // 1))
    k = max(0, -(-(wanted - _FRAME_BASE) // _FRAME_STEP))
    return min(_FRAME_BASE + _FRAME_STEP * k, max_frames)


@dataclass(frozen=True)
class Profile:
    """Everything a running h3 cannot change about itself."""

    binary: str
    model_dir: str
    variant: str
    int8_row_fc2: bool
    env: tuple  # sorted (name, value) pairs

    @classmethod
    def build(cls, *, binary: str = "", model_dir: str = DEFAULT_MODEL,
              variant: str = "T2VA", int8_row_fc2: bool = True,
              env: dict | None = None) -> "Profile":
        path = binary or os.environ.get("H3_BINARY", "")
        if not path:
            raise H3WorkerError("h3 binary path is not configured (--binary or H3_BINARY)")
        path = os.path.abspath(os.path.expanduser(path))
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            raise H3WorkerError(f"h3 binary is missing or not executable: {path}")
        extra = {str(k): str(v) for k, v in (env or {}).items()}
        # The single most expensive flag to forget: without it the video VAE
        # decode costs more than the whole transformer, 69 s against 17 s.
        extra.setdefault("H3_VAE_INT8_FFN", "1")
        if variant:
            extra["H3_DIT_VARIANT"] = variant
        return cls(binary=path, model_dir=model_dir, variant=variant or "",
                   int8_row_fc2=bool(int8_row_fc2),
                   env=tuple(sorted(extra.items())))

    def describe(self) -> dict:
        return {"binary": self.binary, "model_dir": self.model_dir,
                "variant": self.variant, "int8_row_fc2": self.int8_row_fc2,
                "env": dict(self.env)}


class H3Worker:
    """One long-lived h3 REPL. Not thread-safe: hold one lock per GPU."""

    def __init__(self, profile: Profile, *, log=None):
        self.profile = profile
        self._log = log or (lambda message: None)
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue = queue.Queue()
        self._errors: deque = deque(maxlen=400)
        self._activity = 0.0
        self._workdir = ""
        self.jobs = 0
        self.started_at = 0.0
        self.last_used = 0.0

    # -- lifecycle ---------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, timeout: float = 180.0) -> None:
        if self.alive:
            return
        env = {k: v for k, v in os.environ.items() if k not in _MANAGED_ENV}
        env.update(dict(self.profile.env))
        command = [self.profile.binary, "-d", self.profile.model_dir]
        if self.profile.int8_row_fc2:
            command.append("--use-int8-row-fc2")
        self._workdir = tempfile.mkdtemp(prefix="h3-worker-")
        self._lines = queue.Queue()
        self._errors.clear()
        self._activity = time.time()
        self._log(f"spawn {' '.join(command)}")
        self._proc = subprocess.Popen(
            command, cwd=os.path.dirname(self.profile.binary), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        threading.Thread(target=self._pump, args=(self._proc.stdout, self._lines),
                         daemon=True).start()
        threading.Thread(target=self._pump, args=(self._proc.stderr, None),
                         daemon=True).start()
        self.started_at = time.time()
        self.last_used = self.started_at
        self._send(f"!output {self._workdir}")
        self._send("!open off")
        self._send("!show off")
        self._fence(timeout=timeout, stall=timeout, what="startup")

    def stop(self, timeout: float = 20.0) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin:
                proc.stdin.write("!quit\n")
                proc.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._workdir:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = ""
        self._log("stopped")

    # -- plumbing ----------------------------------------------------------

    def _pump(self, stream, sink: queue.Queue | None) -> None:
        try:
            for line in stream:
                self._activity = time.time()
                line = line.rstrip("\n")
                if sink is None:
                    # Progress rewrites one line with \r; only the text after
                    # the last carriage return is worth keeping.
                    for part in line.split("\r"):
                        if part.strip():
                            self._errors.append(part.strip())
                else:
                    sink.put(line)
        except (ValueError, OSError):
            pass
        finally:
            if sink is not None:
                sink.put(None)  # EOF: the child is gone

    def _send(self, line: str) -> None:
        if not self.alive or not self._proc or not self._proc.stdin:
            raise H3WorkerError("h3 is not running")
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise H3WorkerError(f"h3 stopped reading commands: {exc}") from exc

    def _fence(self, *, timeout: float, stall: float, what: str) -> list:
        """Send `!cache` and collect stdout up to its reply.

        `stall` guards the case h3 wedges without dying: no output on either
        stream for that long and the job is declared lost. The overall
        `timeout` is the backstop for a job that is merely enormous.
        """
        self._send("!cache")
        deadline = time.time() + timeout
        collected = []
        while True:
            try:
                line = self._lines.get(timeout=1.0)
            except queue.Empty:
                if not self.alive:
                    raise H3WorkerError(
                        f"h3 exited during {what}: {self.stderr_tail()}")
                if time.time() > deadline:
                    raise H3WorkerError(
                        f"h3 did not finish {what} within {timeout:.0f}s")
                if time.time() - self._activity > stall:
                    raise H3WorkerError(
                        f"h3 produced no output for {stall:.0f}s during {what}")
                continue
            if line is None:
                raise H3WorkerError(f"h3 exited during {what}: {self.stderr_tail()}")
            if line.startswith(_FENCE):
                collected.append(line)
                return collected
            collected.append(line)

    def stderr_tail(self, count: int = 6, settle: float = 0.0) -> str:
        """The last few diagnostics.

        `settle` exists because the two streams are read by two threads and
        h3 writes the failure to stderr just before the fence arrives on
        stdout: without a moment's grace the reason for a failed job is
        routinely still in flight and the caller is told "no diagnostics".
        """
        if settle:
            time.sleep(settle)
        tail = [line for line in list(self._errors)[-count:] if line]
        return " / ".join(tail) or "no diagnostics"

    # -- work --------------------------------------------------------------

    def render(self, *, prompt: str, output: str, width: int, height: int,
               frames: int, steps: int = 3, seed: int | None = 42,
               ref_images=(), ref_audio=(), first: str = "", last: str = "",
               reuse: int = 1, layers: int = 50, core_reuse: int = 1,
               token_reduction: bool = False, render_size: str = "",
               timeout: float = 1800.0, stall: float = 420.0) -> dict:
        text = " ".join(str(prompt or "").split())
        if not text:
            raise H3WorkerError("prompt must not be empty")
        if text.startswith("!"):
            raise H3WorkerError("a prompt cannot start with '!' -- the REPL would read it as a command")
        for edge, name in ((width, "width"), (height, "height")):
            if edge <= 0 or edge % 32:
                raise H3WorkerError(f"{name} must be a positive multiple of 32, got {edge}")

        self.start()
        while True:  # drain anything left over from a previous job
            try:
                stale = self._lines.get_nowait()
            except queue.Empty:
                break
            if stale is None:
                raise H3WorkerError(f"h3 exited between jobs: {self.stderr_tail()}")

        output = os.path.abspath(os.path.expanduser(output))
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

        self._send("!refs clear")
        self._send("!first clear")
        self._send("!last clear")
        self._send(f"!size {width}x{height}")
        self._send(f"!frames {int(frames)}")
        self._send(f"!steps {int(steps)}")
        # Sent on every job rather than only when they change: the REPL is
        # sticky and a knob left over from the previous caller's shot is the
        # kind of bug that only shows up as "why is this take smeared".
        self._send(f"!reuse {int(reuse)}")
        self._send(f"!layers {int(layers)}")
        self._send(f"!core-reuse {int(core_reuse)}")
        self._send(f"!token-reduction {'on' if token_reduction else 'off'}")
        self._send(f"!render-size {render_size or 'native'}")
        self._send("!seed random" if seed is None else f"!seed {int(seed)}")
        for path in ref_images:
            self._send(f"!ref-image {os.path.abspath(os.path.expanduser(path))}")
        for path in ref_audio:
            self._send(f"!ref-audio {os.path.abspath(os.path.expanduser(path))}")
        if first:
            self._send(f"!first {os.path.abspath(os.path.expanduser(first))}")
        if last:
            self._send(f"!last {os.path.abspath(os.path.expanduser(last))}")

        started = time.time()
        self._send(text)
        lines = self._fence(timeout=timeout, stall=stall, what="a generation")
        wall = time.time() - started
        self.jobs += 1
        self.last_used = time.time()

        produced = elapsed = None
        used_seed = seed
        cache = ""
        for line in lines:
            match = _DONE.match(line)
            if match:
                produced, elapsed = match.group(1), float(match.group(2))
            elif line.startswith("Seed: ") and produced is None:
                try:
                    used_seed = int(line.split(":", 1)[1])
                except ValueError:
                    pass
            elif line.startswith(_FENCE):
                cache = line
        if not produced:
            raise H3WorkerError(
                f"h3 generated nothing: {self.stderr_tail(settle=0.5)}")
        if not os.path.exists(produced) or os.path.getsize(produced) == 0:
            raise H3WorkerError(f"h3 reported {produced} but the file is missing or empty")
        shutil.move(produced, output)

        return {
            "path": output, "seed": used_seed, "frames": int(frames),
            "width": width, "height": height, "steps": int(steps),
            "duration": int(frames) / _FPS, "layers": int(layers),
            "reuse": int(reuse), "core_reuse": int(core_reuse),
            "generate_seconds": elapsed, "wall_seconds": wall,
            "job": self.jobs, "cache": cache,
        }

    def status(self) -> dict:
        return {
            "alive": self.alive, "jobs": self.jobs,
            "uptime": round(time.time() - self.started_at, 1) if self.started_at else 0.0,
            "idle": round(time.time() - self.last_used, 1) if self.last_used else 0.0,
            "profile": self.profile.describe(),
            "pid": self._proc.pid if self._proc else None,
        }


# -- server --------------------------------------------------------------


class WorkerServer:
    """Serialise every caller onto one resident h3.

    One GPU, so one job at a time; the lock is the whole scheduler. A job whose
    profile does not match the resident child stops it and starts another,
    which is why callers should agree on a profile rather than alternating.
    """

    def __init__(self, socket_path: str, *, defaults: dict,
                 idle_seconds: float = DEFAULT_IDLE, verbose: bool = True):
        self.socket_path = os.path.abspath(os.path.expanduser(socket_path))
        self.defaults = defaults
        self.idle_seconds = idle_seconds
        self.verbose = verbose
        self._gpu = threading.Lock()
        self._worker: H3Worker | None = None
        self._stopping = threading.Event()
        self._server: socket.socket | None = None
        self.served = 0
        self.restarts = 0

    def log(self, message: str) -> None:
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    def _profile_for(self, request: dict) -> Profile:
        merged = dict(self.defaults)
        for key in ("binary", "model_dir", "variant", "int8_row_fc2"):
            if request.get(key) not in (None, ""):
                merged[key] = request[key]
        env = dict(merged.get("env") or {})
        env.update(request.get("env") or {})
        merged["env"] = env
        return Profile.build(**merged)

    def _worker_for(self, profile: Profile) -> H3Worker:
        if self._worker and self._worker.profile != profile:
            self.log(f"profile change -> restart ({self._worker.profile.variant} to {profile.variant})")
            self._worker.stop()
            self._worker = None
            self.restarts += 1
        if self._worker and not self._worker.alive:
            self.log("child died -> restart")
            self._worker.stop()
            self._worker = None
            self.restarts += 1
        if self._worker is None:
            self._worker = H3Worker(profile, log=self.log)
        return self._worker

    def handle(self, request: dict) -> dict:
        op = request.get("op", "render")
        if op == "ping":
            return {"ok": True, "pong": True}
        if op == "status":
            with self._gpu:
                worker = self._worker.status() if self._worker else {"alive": False}
            return {"ok": True, "served": self.served, "restarts": self.restarts,
                    "idle_seconds": self.idle_seconds, "worker": worker}
        if op == "shutdown":
            self._stopping.set()
            return {"ok": True, "shutting_down": True}
        if op == "release":  # drop the child, keep listening
            with self._gpu:
                if self._worker:
                    self._worker.stop()
                    self._worker = None
            return {"ok": True, "released": True}
        if op != "render":
            return {"ok": False, "error": f"unknown op {op!r}"}

        profile = self._profile_for(request)
        frames = request.get("frames")
        if not frames:
            frames = legal_frames(float(request.get("seconds") or 0) or 1.0)
        with self._gpu:
            worker = self._worker_for(profile)
            resident = worker.alive
            try:
                result = worker.render(
                    prompt=request["prompt"], output=request["output"],
                    width=int(request["width"]), height=int(request["height"]),
                    frames=int(frames), steps=int(request.get("steps") or 3),
                    seed=request.get("seed", 42),
                    ref_images=request.get("ref_images") or (),
                    ref_audio=request.get("ref_audio") or (),
                    first=request.get("first") or "",
                    last=request.get("last") or "",
                    reuse=int(request.get("reuse") or 1),
                    layers=int(request.get("layers") or 50),
                    core_reuse=int(request.get("core_reuse") or 1),
                    token_reduction=bool(request.get("token_reduction")),
                    render_size=request.get("render_size") or "",
                    timeout=float(request.get("timeout") or 1800.0),
                )
            except H3WorkerError as exc:
                if not worker.alive:  # a corpse is no use to the next caller
                    worker.stop()
                    self._worker = None
                    self.restarts += 1
                self.log(f"job failed: {exc}")
                return {"ok": False, "error": str(exc), "resident": resident}
        self.served += 1
        result["ok"] = True
        result["resident"] = resident
        self.log(f"job {result['job']} -> {result['path']} "
                 f"[{result['generate_seconds']:.1f}s, "
                 f"{'resident' if resident else 'cold'}]")
        return result

    def _serve_connection(self, connection: socket.socket) -> None:
        with connection:
            connection.settimeout(None)
            data = b""
            try:
                while not data.endswith(b"\n"):
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                if not data.strip():
                    return
                request = json.loads(data.decode("utf-8"))
                reply = self.handle(request)
            except H3WorkerError as exc:
                reply = {"ok": False, "error": str(exc)}
            except Exception as exc:  # a bad client must not take the server down
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            try:
                connection.sendall((json.dumps(reply) + "\n").encode("utf-8"))
            except OSError:
                pass

    def _reaper(self) -> None:
        while not self._stopping.wait(15.0):
            if self.idle_seconds <= 0:
                continue
            if not self._gpu.acquire(blocking=False):
                continue
            try:
                worker = self._worker
                if worker and worker.alive and worker.last_used and \
                        time.time() - worker.last_used > self.idle_seconds:
                    self.log(f"idle {self.idle_seconds:.0f}s -> releasing ~10 GiB")
                    worker.stop()
                    self._worker = None
            finally:
                self._gpu.release()

    def run(self) -> int:
        os.makedirs(os.path.dirname(self.socket_path) or ".", exist_ok=True)
        if os.path.exists(self.socket_path):
            # A live server owns its socket; a crashed one leaves it behind.
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.settimeout(2.0)
                probe.connect(self.socket_path)
                probe.close()
                print(f"h3_worker: another server is already on {self.socket_path}",
                      file=sys.stderr)
                return 1
            except OSError:
                os.unlink(self.socket_path)
            finally:
                probe.close()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        self._server.listen(16)
        self._server.settimeout(1.0)
        threading.Thread(target=self._reaper, daemon=True).start()
        self.log(f"listening on {self.socket_path}")
        try:
            while not self._stopping.is_set():
                try:
                    connection, _ = self._server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                threading.Thread(target=self._serve_connection,
                                 args=(connection,), daemon=True).start()
        except KeyboardInterrupt:
            self.log("interrupted")
        finally:
            self._stopping.set()
            if self._server:
                self._server.close()
            with self._gpu:
                if self._worker:
                    self._worker.stop()
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            self.log("gone")
        return 0


# -- client --------------------------------------------------------------


class H3WorkerClient:
    def __init__(self, socket_path: str = "", timeout: float = 2400.0):
        self.socket_path = os.path.abspath(os.path.expanduser(
            socket_path or os.environ.get("H3_WORKER_SOCKET") or DEFAULT_SOCKET))
        self.timeout = timeout

    def available(self) -> bool:
        try:
            return bool(self.request({"op": "ping"}, timeout=3.0).get("pong"))
        except Exception:
            return False

    def request(self, payload: dict, timeout: float | None = None) -> dict:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(timeout if timeout is not None else self.timeout)
        try:
            connection.connect(self.socket_path)
            connection.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            connection.shutdown(socket.SHUT_WR)
            data = b""
            while not data.endswith(b"\n"):
                chunk = connection.recv(65536)
                if not chunk:
                    break
                data += chunk
        finally:
            connection.close()
        if not data.strip():
            raise H3WorkerError("the worker closed the connection without replying")
        return json.loads(data.decode("utf-8"))

    def render(self, **job) -> dict:
        job.setdefault("op", "render")
        reply = self.request(job)
        if not reply.get("ok"):
            raise H3WorkerError(reply.get("error") or "the worker refused the job")
        return reply


# -- command line --------------------------------------------------------


def _defaults_from(args) -> dict:
    env = {}
    for item in (args.env or []):
        name, _, value = item.partition("=")
        if not name or not _:
            raise H3WorkerError(f"--env wants NAME=VALUE, got {item!r}")
        env[name] = value
    return {"binary": args.binary, "model_dir": args.model_dir,
            "variant": args.variant, "int8_row_fc2": not args.no_int8_row_fc2,
            "env": env}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--socket", default="")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="hold one h3 and answer jobs on a socket")
    serve.add_argument("-b", "--binary", default=os.environ.get("H3_BINARY", ""))
    serve.add_argument("-d", "--model-dir", default=DEFAULT_MODEL)
    serve.add_argument("--variant", default="T2VA")
    serve.add_argument("--env", action="append", help="NAME=VALUE, repeatable")
    serve.add_argument("--no-int8-row-fc2", action="store_true")
    serve.add_argument("--idle", type=float, default=DEFAULT_IDLE,
                       help="seconds of idleness before the child is released (0 = never)")
    serve.add_argument("--quiet", action="store_true")

    render = sub.add_parser("render", help="ask a running server for one clip")
    render.add_argument("-p", "--prompt", required=True)
    render.add_argument("-o", "--output", required=True)
    render.add_argument("--seconds", type=float, default=10.0)
    render.add_argument("--frames", type=int, default=0)
    render.add_argument("--size", default="320x576")
    render.add_argument("--steps", type=int, default=3)
    render.add_argument("--seed", type=int, default=42)
    render.add_argument("--ref-image", action="append", default=[])
    render.add_argument("--ref-audio", action="append", default=[])
    render.add_argument("--variant", default="")
    render.add_argument("--lora", action="append", default=[],
                        help="NAME[:STRENGTH], repeatable; sets H3_LORA for the job")

    sub.add_parser("status", help="what the server is holding")
    sub.add_parser("release", help="drop the resident child, keep the server")
    sub.add_parser("stop", help="shut the server down")

    args = parser.parse_args(argv)

    if args.command == "serve":
        server = WorkerServer(args.socket or DEFAULT_SOCKET,
                              defaults=_defaults_from(args),
                              idle_seconds=args.idle, verbose=not args.quiet)
        return server.run()

    client = H3WorkerClient(args.socket)
    if args.command in ("status", "release", "stop"):
        op = {"status": "status", "release": "release", "stop": "shutdown"}[args.command]
        print(json.dumps(client.request({"op": op}), indent=2, ensure_ascii=False))
        return 0

    width, _, height = args.size.lower().partition("x")
    job = {
        "prompt": args.prompt, "output": args.output,
        "width": int(width), "height": int(height),
        "frames": args.frames or legal_frames(args.seconds),
        "steps": args.steps, "seed": args.seed,
        "ref_images": args.ref_image, "ref_audio": args.ref_audio,
    }
    if args.variant:
        job["variant"] = args.variant
    if args.lora:
        job["env"] = {"H3_LORA": ",".join(args.lora)}
    started = time.time()
    reply = client.render(**job)
    reply["client_seconds"] = round(time.time() - started, 2)
    print(json.dumps(reply, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except H3WorkerError as exc:
        print(f"h3_worker: {exc}", file=sys.stderr)
        sys.exit(1)
