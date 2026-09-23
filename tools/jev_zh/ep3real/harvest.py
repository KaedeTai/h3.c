"""Pull frames out of every distinct EP3 video and read their on-screen text.

One frame every 2 s (plus one at 0.5 s so short takes get one), scaled to <=720 wide,
read with Apple Vision (vision_ocr.swift). Output: ocr.jsonl, one line per frame.
Resumable: videos already in ocr.jsonl are skipped.
"""
import json, os, subprocess, sys
ROOT = os.path.expanduser("~/datasets/ep3_ocr")
FR = os.path.join(ROOT, "frames")
paths = [l.rstrip("\n") for l in open(os.path.join(ROOT, "uniq.txt"))]
index = {p: i for i, p in enumerate(paths)}
json.dump(paths, open(os.path.join(ROOT, "videos.json"), "w"), ensure_ascii=False, indent=0)
out_path = os.path.join(ROOT, "ocr.jsonl")
done = set()
if os.path.exists(out_path):
    for l in open(out_path):
        done.add(json.loads(l)["vid"])
out = open(out_path, "a")
batch = []

def flush():
    global batch
    if not batch: return
    r = subprocess.run([os.path.join(ROOT, "vision_ocr")], input="\n".join(f for _, f in batch),
                       capture_output=True, text=True)
    got = {}
    for l in r.stdout.splitlines():
        o = json.loads(l); got[o["path"]] = o["lines"]
    for vid, f in batch:
        t = float(os.path.basename(f).split("_t")[1][:-4])
        out.write(json.dumps({"vid": vid, "t": t, "frame": f, "lines": got.get(f, [])}, ensure_ascii=False) + "\n")
    out.flush(); batch = []

for n, p in enumerate(paths):
    vid = index[p]
    if vid in done: continue
    d = os.path.join(FR, f"v{vid:04d}"); os.makedirs(d, exist_ok=True)
    for stale in os.listdir(d):  # a run killed mid-video left frames behind
        os.unlink(os.path.join(d, stale))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "0.5", "-i", p, "-vf",
                    "fps=0.5,scale='min(720,iw)':-2", "-q:v", "3", f"{d}/%04d.jpg"])
    frames = sorted(os.listdir(d))
    for k, name in enumerate(frames):
        new = os.path.join(d, f"f{k:04d}_t{0.5 + 2 * k:.1f}.jpg")
        os.rename(os.path.join(d, name), new)
        batch.append((vid, new))
    if not frames:  # empty take: record nothing so it isn't retried forever
        out.write(json.dumps({"vid": vid, "t": -1, "frame": None, "lines": []}) + "\n")
    if len(batch) >= 64: flush()
    if n % 50 == 0: print(n, len(paths), flush=True)
flush()
print("ALLDONE", flush=True)
