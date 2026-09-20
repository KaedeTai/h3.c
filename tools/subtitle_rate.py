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
import json, os, re, subprocess, sys, tempfile
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


# ---------------------------------------------------------------- VLM stage
#
# The geometry stage below was tuned on ONE kind of material — a white lab coat,
# head crop, plain wall — and measured on a second it is barely better than
# chance. On EP3, four checks, four wrong answers:
#
#   * a take with a phone, hands and a table in the lower third   -> 69% SUBTITLED
#   * 蘇知行's yellow 漢服 (the yellow-glyph stage reads cloth)     -> 100% SUBTITLED
#   * real footage with "只賣一個綠寶幹系" burned across the bottom -> 0%, clean
#   * real footage with no caption at all                          -> 58% SUBTITLED
#
# Both directions, so no threshold rescues it: brightness and flip-count are
# proxies for text and this material breaks the proxy. A local vision model
# reads the frame instead of guessing from pixel statistics, costs a few seconds
# against the ~130 s it takes to render the take being judged, and got all four
# of those right. Geometry stays as the fallback for when no server answers.
VLM_URL = os.environ.get("H3_VLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
VLM_MODEL = os.environ.get("H3_VLM_MODEL", "")
VLM_KEY = os.environ.get("H3_VLM_KEY", "")
VLM_SAMPLE = 6
VLM_PROMPT = ("這是一格影片畫面。畫面上有沒有「燒進畫面的字幕或說明文字」？\n"
              "招牌、商品包裝、手機介面上本來就有的字不算字幕。\n"
              "只回一個字：有，或無。")


def _ask_vlm(png_path, model, timeout=90):
    import base64, json as _json, urllib.request
    with open(png_path, "rb") as handle:
        blob = base64.b64encode(handle.read()).decode()
    body = {"model": model, "max_tokens": 8, "temperature": 0, "messages": [{
        "role": "user", "content": [
            {"type": "text", "text": VLM_PROMPT},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + blob}}]}]}
    headers = {"Content-Type": "application/json"}
    if VLM_KEY:
        headers["Authorization"] = "Bearer " + VLM_KEY
    request = urllib.request.Request(VLM_URL, data=_json.dumps(body).encode(), headers=headers)
    answer = _json.load(urllib.request.urlopen(request, timeout=timeout))
    return answer["choices"][0]["message"]["content"].strip()


def rate_vlm(path, model, samples=VLM_SAMPLE):
    """Percentage of sampled frames the vision model says carry a caption.

    Raises rather than guessing: the caller decides whether a dead server means
    "fall back to geometry" or "stop", and a scorer that silently degrades to a
    method known to be wrong is how a whole draw gets thrown away in silence.
    """
    work = tempfile.mkdtemp()
    try:
        n = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames",
                                "-select_streams", "v:0", "-show_entries",
                                "stream=nb_read_frames", "-of", "csv=p=0", path],
                               capture_output=True, text=True).stdout.strip() or 0)
        if not n:
            return float("nan")
        step = max(1, n // samples)
        sel = "+".join(rf"eq(n\,{i})" for i in range(0, n, step))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vf",
                        f"select={sel},scale=448:-1", *_fps_mode_flag(),
                        f"{work}/%03d.png"], check=True)
        names = sorted(os.listdir(work))[:samples]
        hits = 0
        for name in names:
            reply = _ask_vlm(os.path.join(work, name), model)
            # "有" anywhere and "無" absent — small models like to add a period.
            if "有" in reply and "無" not in reply:
                hits += 1
        return 100.0 * hits / max(1, len(names))
    finally:
        for name in os.listdir(work):
            os.unlink(os.path.join(work, name))
        os.rmdir(work)


# ------------------------------------------------------------------ OCR stage
#
# A step past the yes/no question: read the frame and look at what comes back.
# jina-ocr-v1 answers a fixed "transcribe this document" instruction, and on a
# frame with no text it does not invent any -- it describes the picture
# instead, in English, usually grounding the whole frame as one region:
#
#   s01  "A single water droplet hanging from a leaf[[0,0,1000,1000]]"
#   s09  "The image depicts a scene from a film or television show, ..."
#   w3   "超前一步 / Small Steps / 領先一代 / Big Impact / 2025 Total Swiss 澳門大會"
#   w8   "# Fit Solution / 人體有70兆細胞細胞或人體健康！ / ## 95.8fm Interview..."
#
# So the caption decision is "did it transcribe or did it describe", and the
# answer carries the words, which the yes/no never did. Knowing a take burned
# in 「白露秋分夜」 tells you the model is captioning its own dialogue; knowing
# it burned in a shop sign tells you the reference image did it.
#
# NOTE this is MORE sensitive than the yes/no: real signage inside the scene
# is text, and it is reported. For h3 takes that is the right way round -- the
# model drawing text at all is usually the defect -- but it means a shot that
# is meant to contain a sign will always be flagged.
OCR_SOCKET = os.environ.get("H3_OCR_SOCKET", os.path.expanduser("~/.h3/ocr.sock"))
OCR_SAMPLE = 6
# A burned-in caption persists: the geometric detector measured real ones on
# 69-100% of sampled frames. An OCR hallucination does not -- one frame of the
# EP3 cut came back "# 2 / *林建中*" where Ornith, asked to look in every
# corner, found nothing at all. One frame in six is 16.7%, which would clear a
# 5% bar and cost the take. So this method needs TWO frames to agree.
OCR_LIMIT = 33.0
# Phrases the model only uses when it is describing a picture, never when
# transcribing one. Measured over 27 frames of this project's material.
_DESCRIBER = (
    "the image depicts", "the image shows", "the image features",
    "this image", "appears to be", "is seated", "is sitting", "is standing",
    "is holding", "is wearing", "hanging from", "in a room", "on stage",
)
_BBOX = re.compile(r"\[\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]\]\s*$")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff]")


def looks_like_transcription(text: str) -> bool:
    """True when the model read words off the frame rather than describing it."""
    body = _BBOX.sub("", (text or "").strip()).strip()
    if not body:
        return False
    if _CJK.search(body):
        return True
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    # Markdown structure is the transcriber's own output shape.
    if any(line.startswith(("#", "|", "$$")) for line in lines):
        return True
    lowered = body.lower()
    if any(phrase in lowered for phrase in _DESCRIBER):
        return False
    # What is left: short fragments are signage, long prose is description.
    return len(body.split()) <= 8


def rate_ocr(path, socket_path=None, samples=OCR_SAMPLE):
    """Percentage of sampled frames carrying burned-in text, and what it said.

    Returns (percentage, [text, ...]) so the caller can put the words in the
    take's report instead of only a number.
    """
    import socket as _socket

    target = os.path.expanduser(socket_path or OCR_SOCKET)
    work = tempfile.mkdtemp()
    try:
        n = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames",
                                "-select_streams", "v:0", "-show_entries",
                                "stream=nb_read_frames", "-of", "csv=p=0", path],
                               capture_output=True, text=True).stdout.strip() or 0)
        if not n:
            return float("nan"), []
        step = max(1, n // samples)
        sel = "+".join(rf"eq(n\,{i})" for i in range(0, n, step))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vf",
                        f"select={sel},scale=640:-1", *_fps_mode_flag(),
                        f"{work}/%03d.png"], check=True)
        frames = [os.path.join(work, name) for name in sorted(os.listdir(work))][:samples]
        payload = json.dumps({"op": "read", "paths": frames,
                              "max_new_tokens": 160}).encode("utf-8")
        connection = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        connection.settimeout(900)
        connection.connect(target)
        with connection:
            connection.sendall(payload + b"\n")
            buffer = b""
            while not buffer.endswith(b"\n"):
                chunk = connection.recv(65536)
                if not chunk:
                    break
                buffer += chunk
        reply = json.loads(buffer.decode("utf-8"))
        if not reply.get("ok"):
            raise VisionUnavailable(reply.get("error", "ocr worker refused"))
        found = [page["text"] for page in reply["pages"]
                 if looks_like_transcription(page["text"])]
        return 100.0 * len(found) / max(1, len(reply["pages"])), found
    finally:
        for name in os.listdir(work):
            os.unlink(os.path.join(work, name))
        os.rmdir(work)


# ----------------------------------------------------------- noul (logits)
#
# Jev's shape, on the model this project already trusts: prefill the frame
# with the question, then read the softmax over just the two option tokens.
# Nothing is generated, so the answer cannot turn on whether the model felt
# like adding a period, and what comes back is a probability instead of a
# word.
#
# Measured on 21 labelled frames of this project's material, with the same
# question the yes/no method asks: 21/21, lowest "has text" 0.999 against
# highest "no text" 0.294 — a 0.704 gap — at 0.15-0.33 s a frame against the
# generated answer's ~2.5 s. It gets right both frames that fooled the OCR
# (the yellow 漢服 at 0.0006, EP3's cut s12 at 0.0953) and both frames where
# the generated answer had flipped between wordings.
#
# Two things this does NOT do, both measured:
#
#   * It does not remove the question's influence. The same frame under three
#     phrasings moved by up to 0.986 — but those were three different
#     questions ("any text at all" vs "a subtitle along the bottom"), and the
#     disagreement is them meaning different things, not instability. The
#     prompt IS the specification; this only makes the answer to it sharp.
#   * It does not transfer to a cheaper model. The identical code on
#     gemma-3-12b-it-4bit scores 3/21 with the classes overlapping (gap
#     -0.089). Re-measure before swapping the model.
NOUL_SAMPLE = 6
# A caption persists; one frame over the line is not one. Same two-of-six bar
# the OCR method uses, and for the same reason.
NOUL_LIMIT = 33.0
NOUL_QUESTION = VLM_PROMPT   # the same question, answered from logits


def rate_noul(path, socket_path=None, samples=NOUL_SAMPLE, question=NOUL_QUESTION):
    """Percentage of sampled frames the model puts over 0.5, and the scores.

    Returns (percentage, [probability, ...]) so the caller can show how sure
    it was rather than only how many frames crossed a line.
    """
    import socket as _socket

    target = os.path.expanduser(socket_path or OCR_SOCKET)
    work = tempfile.mkdtemp()
    try:
        n = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames",
                                "-select_streams", "v:0", "-show_entries",
                                "stream=nb_read_frames", "-of", "csv=p=0", path],
                               capture_output=True, text=True).stdout.strip() or 0)
        if not n:
            return float("nan"), []
        step = max(1, n // samples)
        sel = "+".join(rf"eq(n\,{i})" for i in range(0, n, step))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vf",
                        f"select={sel},scale=640:-1", *_fps_mode_flag(),
                        f"{work}/%03d.png"], check=True)
        frames = [os.path.join(work, name) for name in sorted(os.listdir(work))][:samples]
        payload = json.dumps({"op": "noul", "question": question,
                              "paths": frames}).encode("utf-8")
        connection = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        connection.settimeout(900)
        connection.connect(target)
        with connection:
            connection.sendall(payload + b"\n")
            buffer = b""
            while not buffer.endswith(b"\n"):
                chunk = connection.recv(65536)
                if not chunk:
                    break
                buffer += chunk
        reply = json.loads(buffer.decode("utf-8"))
        if not reply.get("ok"):
            raise VisionUnavailable(reply.get("error", "decider refused"))
        scores = [answer["p"] for answer in reply["answers"]]
        over = [p for p in scores if p > 0.5]
        return 100.0 * len(over) / max(1, len(scores)), scores
    finally:
        for name in os.listdir(work):
            os.unlink(os.path.join(work, name))
        os.rmdir(work)


class VisionUnavailable(Exception):
    """The vision model was asked for and could not be reached."""


def rate(path, dump=None, model=None, fallback=True):
    """Percentage of sampled frames carrying a burned-in caption.

    `fallback` is the whole decision. At a prompt, falling back to geometry and
    saying so is friendly. Inside an automated draw it is not: geometry rejected
    all three EP3 takes over a phone screen and a yellow 漢服, so a mistyped API
    key silently cost the shot its entire draw and the log said only "全部淘汰".
    Callers that disqualify on this number pass `fallback=False` and treat the
    exception as "caption unknown", which screens nothing but throws nothing
    away either.
    """
    model = model if model is not None else VLM_MODEL
    if model:
        try:
            return rate_vlm(path, model)
        except Exception as exc:
            if not fallback:
                raise VisionUnavailable(str(exc)) from exc
            print(f"[subtitle] 視覺模型沒有回應（{exc}），退回幾何判斷，"
                  f"這條的字幕判定不可靠", file=sys.stderr)
    return _rate_geometry(path, dump=dump)


def _rate_geometry(path, dump=None):
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
    model = None
    if "--dump" in args:
        i = args.index("--dump"); dump = args[i + 1]; del args[i:i + 2]
    if "--vlm" in args:
        i = args.index("--vlm"); model = args[i + 1]; del args[i:i + 2]
    if "--geometry" in args:
        args.remove("--geometry"); model = ""
    print(f"{'take':30s} {'frames with a subtitle':>23s}")
    for p in args:
        r = rate(p, dump, model)
        print(f"{os.path.basename(p):30s} {r:21.0f} %  {'SUBTITLED' if r > 0 else ''}")
