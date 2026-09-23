"""Contact sheets for labelling: each tile is the frame with its OCR boxes drawn and the index."""
import json, os, sys, random
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.expanduser("~/datasets/ep3_ocr")
S = json.load(open(os.path.join(ROOT, "strings.json")))
boxes = {}
for l in open(os.path.join(ROOT, "ocr.jsonl")):
    r = json.loads(l)
    if r["frame"]: boxes[r["frame"]] = r["lines"]
font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 22)
def sheet(ids, name, cols=4, tw=400):
    tiles = []
    for i in ids:
        f = S[i]["frames"][0]
        im = Image.open(f).convert("RGB"); W, H = im.size
        d = ImageDraw.Draw(im)
        for b in boxes.get(f, []):
            if b["conf"] < 0.3: continue
            d.rectangle([b["x"] * W, b["y"] * H, (b["x"] + b["w"]) * W, (b["y"] + b["h"]) * H], outline=(255, 0, 0), width=3)
        im = im.resize((tw, int(H * tw / W)))
        t = Image.new("RGB", (tw, im.size[1] + 30), (20, 20, 20)); t.paste(im, (0, 30))
        ImageDraw.Draw(t).text((4, 2), f"#{i}", fill=(255, 255, 0), font=font)
        tiles.append(t)
    th = max(t.size[1] for t in tiles); rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
    for k, t in enumerate(tiles): out.paste(t, ((k % cols) * tw, (k // cols) * th))
    p = os.path.join(ROOT, "sheets", name); os.makedirs(os.path.dirname(p), exist_ok=True)
    out.save(p, quality=80); return p
sel = json.loads(sys.argv[1]); prefix = sys.argv[2]; per = int(sys.argv[3]) if len(sys.argv) > 3 else 16
for k in range(0, len(sel), per):
    print(sheet(sel[k:k + per], f"{prefix}_{k // per:02d}.jpg"))
