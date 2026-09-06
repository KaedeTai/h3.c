#!/usr/bin/env python3
"""Remove H3's burned-in subtitles without cropping the frame.

delogo interpolates inward from the box edges, which is fine on the flat white
coat and ruinous on the tie -- any structure crossing the box gets smeared into
vertical streaks.

The shot is locked off, though, and the subtitle text changes between lines, so
for almost every pixel in the subtitle band there exist frames where no glyph
covers it. Build a per-pixel clean plate from the unmasked samples (median over
time, which also rejects the glyph outline), then composite it back only where
the mask says text. The tie survives because it is reconstructed from frames
where it was actually visible, not interpolated from the box border.

STATUS (2026-09-06): clears the subtitles completely and the white coat comes
out clean, but the tie is damaged. Two reasons, both fixable: the widened yellow
test also selects the tie's warm highlights, and the tie moves, so a static
clean plate does not line up with it. Restrict the hue test away from the tie,
or motion-compensate the plate, before trusting this on a delivery. Cropping the
subtitle band is still the reliable option.

Usage: desubtitle.py IN.mp4 OUT.mp4
"""
import numpy as np, subprocess, sys, os, tempfile
from PIL import Image

IN, OUT = sys.argv[1], sys.argv[2]
work = tempfile.mkdtemp()
subprocess.run(["ffmpeg","-y","-v","error","-i",IN,"-vsync","0",f"{work}/f_%05d.png"], check=True)
names = sorted(os.listdir(work))
frames = np.stack([np.asarray(Image.open(f"{work}/{n}").convert("RGB")) for n in names]).astype(np.float32)
T,H,W,_ = frames.shape

R,G,B = frames[...,0], frames[...,1], frames[...,2]
lum   = frames.mean(3)
# Two glyph styles seen so far: saturated yellow, and white with a dark outline.
from scipy.ndimage import binary_dilation, gaussian_filter
# A bare brightness test for the white subtitle style also selects the white lab
# coat -- it covered 31% of the frame. What separates a glyph from the coat is
# the dark outline every subtitle here is drawn with, so require one nearby.
dark = lum < 90
near_dark = np.stack([binary_dilation(d, iterations=4) for d in dark])
# The first pass left a speckle of glyph outline: the bright-core test missed the
# darker yellow-brown rim, those pixels stayed unmasked, and since the subtitle
# sits in the same place all clip the median of "clean" samples was the rim
# colour. Catch anything yellow-leaning, not just the bright core.
yellow = ((G-B)>40)&(G>90)&(R>90)
white  = (lum>215)&(frames.max(3)-frames.min(3) < 30) & near_dark
mask = yellow | white
mask[:, :int(H*0.55)] = False          # subtitles only ever sit in the lower band
if mask.sum() == 0:
    subprocess.run(["ffmpeg","-y","-v","error","-i",IN,"-c","copy",OUT], check=True)
    print("no subtitle found; copied through"); raise SystemExit

# grow the mask so anti-aliased edges and the dark outline go too
mask = np.stack([binary_dilation(m, iterations=6) for m in mask])

# Temporal first: where a pixel is uncovered in at least a few frames, the median
# of those samples rebuilds it exactly -- tie stripes included -- because it is
# real footage, not interpolation.
clean = np.where(mask[...,None], np.nan, frames)
with np.errstate(all="ignore"):
    plate = np.nanmedian(clean, axis=0)
covered = np.isnan(plate[...,0])
print(f"pixels never uncovered: {covered.mean()*100:.2f}% of frame")

# The two subtitle lines land in almost the same place, so a band of pixels is
# masked in every frame and has no temporal sample at all. Those get spatial
# inpainting instead, on the plate rather than per frame, so the repair is
# consistent across the clip and does not shimmer.
import cv2
plate_u8 = np.nan_to_num(plate, nan=0).astype(np.uint8)
plate = cv2.inpaint(plate_u8, covered.astype(np.uint8)*255, 5, cv2.INPAINT_NS).astype(np.float32)

soft = np.stack([gaussian_filter(m.astype(np.float32), 1.5) for m in mask])[...,None]
soft = np.clip(soft*1.6, 0, 1)
out = frames*(1-soft) + plate[None]*soft

os.makedirs(f"{work}/o", exist_ok=True)
for i,a in enumerate(out):
    Image.fromarray(np.clip(a,0,255).astype(np.uint8)).save(f"{work}/o/f_{i+1:05d}.png")
subprocess.run(["ffmpeg","-y","-v","error","-framerate","24","-i",f"{work}/o/f_%05d.png",
                "-i",IN,"-map","0:v","-map","1:a?","-c:v","libx264","-crf","17","-preset","slow",
                "-pix_fmt","yuv420p","-c:a","copy",OUT], check=True)
cov = mask.mean()*100
print(f"repaired {T} frames, mask covered {cov:.2f}% of pixels -> {OUT}")
