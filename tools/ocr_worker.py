#!/usr/bin/env python3
"""One resident jina-ocr-v1, shared by everything that wants to read a frame.

Loading the model costs 6.3 GB off disk and about ten seconds before the
first page is read; after that a frame comes back in one to two seconds. A
caption check samples six frames per take and a draw renders three takes, so
paying the load per call would cost more than the reading. This holds the
model and answers on a Unix socket, exactly as `h3_worker.py` holds h3.

The reply is the text the model read, not a verdict. Deciding what counts as
a burned-in caption is the caller's business -- `subtitle_rate.py` knows the
difference between a subtitle and a shop sign in the background, and this
does not.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import socket
import sys
import threading
import time

DEFAULT_SOCKET = os.path.expanduser("~/.h3/ocr.sock")
DEFAULT_MODEL = os.path.expanduser("~/models/jina-ocr-v1")
# The model answers a fixed instruction; the processor supplies it.
DEFAULT_MAX_TOKENS = 192


class OcrWorkerError(RuntimeError):
    pass


class Resident:
    """The model, loaded once, guarded by a lock because MPS is not reentrant."""

    def __init__(self, model_dir: str, device: str):
        self.model_dir = model_dir
        self.device = device
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._torch = None
        self.pages = 0
        self.loaded_at = 0.0
        self.last_used = 0.0

    @property
    def alive(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        started = time.time()
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            self.model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_dir, dtype=torch.bfloat16, trust_remote_code=True)
        self._model = model.to(torch.device(self.device))
        self.loaded_at = time.time()
        log(f"loaded {self.model_dir} on {self.device} in {self.loaded_at - started:.1f}s")

    def release(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None
            if self._torch is not None and self.device == "mps":
                try:
                    self._torch.mps.empty_cache()
                except Exception:
                    pass

    def read(self, path: str, max_new_tokens: int) -> str:
        from PIL import Image

        if not os.path.exists(path):
            raise OcrWorkerError(f"no such image: {path}")
        with self._lock:
            self.load()
            torch = self._torch
            image = Image.open(path).convert("RGB")
            inputs = self._processor.prepare_ocr_inputs(
                image, device=torch.device(self.device))
            output = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
            text = self._processor.batch_decode(output, skip_special_tokens=True)[0]
            self.pages += 1
            self.last_used = time.time()
        return strip_instruction(text)

    def status(self) -> dict:
        return {
            "alive": self.alive,
            "pages": self.pages,
            "device": self.device,
            "model": self.model_dir,
            "uptime": round(time.time() - self.loaded_at, 1) if self.loaded_at else 0.0,
            "idle": round(time.time() - self.last_used, 1) if self.last_used else 0.0,
        }


def strip_instruction(text: str) -> str:
    """Drop the prompt the decoder echoes back.

    `batch_decode` returns the whole sequence, so every answer starts with the
    processor's own instruction ("Transcribe the provided document image...").
    Leaving it in would make every frame look like it carried text.
    """
    marker = "natural reading order."
    index = text.find(marker)
    if index >= 0:
        text = text[index + len(marker):]
    return text.strip().lstrip(":").strip()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class Server:
    def __init__(self, socket_path: str, model_dir: str, device: str, idle_seconds: float):
        self.socket_path = socket_path
        self.resident = Resident(model_dir, device)
        self.idle_seconds = idle_seconds
        self.served = 0
        self._stopping = threading.Event()
        self._server: socket.socket | None = None

    def handle(self, request: dict) -> dict:
        op = request.get("op", "read")
        if op == "ping":
            return {"ok": True, "pong": True}
        if op == "status":
            return {"ok": True, "served": self.served, "worker": self.resident.status()}
        if op == "release":
            self.resident.release()
            return {"ok": True, "released": True}
        if op == "shutdown":
            self._stopping.set()
            return {"ok": True, "shutting_down": True}
        if op != "read":
            return {"ok": False, "error": f"unknown op {op!r}"}

        paths = request.get("paths")
        if isinstance(request.get("path"), str):
            paths = [request["path"]]
        if not isinstance(paths, list) or not paths:
            return {"ok": False, "error": "read needs a path or a list of paths"}
        budget = int(request.get("max_new_tokens") or DEFAULT_MAX_TOKENS)
        resident_before = self.resident.alive
        started = time.time()
        pages = []
        for path in paths:
            page_started = time.time()
            pages.append({
                "path": path,
                "text": self.resident.read(str(path), budget),
                "seconds": round(time.time() - page_started, 2),
            })
        self.served += len(pages)
        log(f"read {len(pages)} page(s) in {time.time() - started:.1f}s "
            f"[{'resident' if resident_before else 'cold'}]")
        return {"ok": True, "pages": pages, "resident": resident_before,
                "seconds": round(time.time() - started, 2)}

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
                reply = self.handle(json.loads(data.decode("utf-8")))
            except OcrWorkerError as exc:
                reply = {"ok": False, "error": str(exc)}
            except Exception as exc:  # a bad client must not take the server down
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            try:
                connection.sendall((json.dumps(reply) + "\n").encode("utf-8"))
            except OSError:
                pass

    def _reaper(self) -> None:
        while not self._stopping.wait(15.0):
            if self.idle_seconds <= 0 or not self.resident.alive:
                continue
            if self.resident.last_used and \
                    time.time() - self.resident.last_used > self.idle_seconds:
                log(f"idle {self.idle_seconds:.0f}s -> releasing the model")
                self.resident.release()

    def run(self) -> int:
        os.makedirs(os.path.dirname(self.socket_path) or ".", exist_ok=True)
        if os.path.exists(self.socket_path):
            # A live server owns its socket; a crashed one leaves it behind.
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.settimeout(2.0)
                probe.connect(self.socket_path)
                probe.close()
                print(f"another server is already on {self.socket_path}", file=sys.stderr)
                return 1
            except OSError:
                os.unlink(self.socket_path)
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        self._server.listen(16)
        self._server.settimeout(1.0)
        threading.Thread(target=self._reaper, daemon=True).start()
        log(f"listening on {self.socket_path}")
        try:
            while not self._stopping.is_set():
                try:
                    connection, _ = self._server.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self._serve_connection,
                                 args=(connection,), daemon=True).start()
        except KeyboardInterrupt:
            pass
        finally:
            self._server.close()
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass
        return 0


def request(socket_path: str, payload: dict, timeout: float = 900.0) -> dict:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(timeout)
    connection.connect(os.path.expanduser(socket_path))
    with connection:
        connection.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        buffer = b""
        while not buffer.endswith(b"\n"):
            chunk = connection.recv(65536)
            if not chunk:
                break
            buffer += chunk
    if not buffer.strip():
        raise OcrWorkerError("server closed without replying")
    return json.loads(buffer.decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="hold one model and answer on a socket")
    serve.add_argument("--model", default=DEFAULT_MODEL)
    serve.add_argument("--device", default="mps")
    serve.add_argument("--idle-seconds", type=float, default=900.0)
    read = sub.add_parser("read", help="ask a running server to read images")
    read.add_argument("images", nargs="+")
    read.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    sub.add_parser("status", help="what the server is holding")
    sub.add_parser("release", help="drop the model, keep the server")
    sub.add_parser("stop", help="shut the server down")
    args = parser.parse_args()

    if args.command == "serve":
        return Server(os.path.expanduser(args.socket), os.path.expanduser(args.model),
                      args.device, args.idle_seconds).run()
    if args.command == "read":
        reply = request(args.socket, {
            "op": "read",
            "paths": [os.path.abspath(os.path.expanduser(p)) for p in args.images],
            "max_new_tokens": args.max_new_tokens})
        if not reply.get("ok"):
            print(reply.get("error", "failed"), file=sys.stderr)
            return 1
        for page in reply["pages"]:
            print(f"=== {page['path']} ({page['seconds']}s) ===")
            print(page["text"] or "(no text)")
        return 0
    op = {"status": "status", "release": "release", "stop": "shutdown"}[args.command]
    print(json.dumps(request(args.socket, {"op": op}, timeout=30), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
