"""Contact sheets over lines.json: frame thumbnail, the one line boxed in red, its index."""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.expanduser("~/datasets/ep3_ocr")
O = json.load(open(os.path.join(ROOT, "lines.json")))
font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 20)
ids = json.loads(open(sys.argv[1]).read()); prefix = sys.argv[2]; per = int(sys.argv[3]); cols = 5; tw = 320; th = 200
outdir = os.path.expanduser("~/models/_ep3_label_sheets"); os.makedirs(outdir, exist_ok=True)
for k in range(0, len(ids), per):
    chunk = ids[k:k + per]; rows = (len(chunk) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 24)), (0, 0, 0))
    for j, i in enumerate(chunk):
        o = O[i]; im = Image.open(o["frames"][0]).convert("RGB"); W, H = im.size
        x, y, w, h = o["boxes"][0]
        ImageDraw.Draw(im).rectangle([x * W - 3, y * H - 3, (x + w) * W + 3, (y + h) * H + 3], outline=(255, 0, 0), width=max(3, W // 200))
        im.thumbnail((tw, th)); tile = Image.new("RGB", (tw, th + 24), (25, 25, 25))
        tile.paste(im, ((tw - im.size[0]) // 2, 24))
        ImageDraw.Draw(tile).text((4, 1), f"{i}", fill=(255, 255, 0), font=font)
        sheet.paste(tile, ((j % cols) * tw, (j // cols) * (th + 24)))
    p = os.path.join(outdir, f"{prefix}_{k // per:02d}.jpg"); sheet.save(p, quality=82); print(p)
