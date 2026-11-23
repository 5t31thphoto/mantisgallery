"""The firmware's own skirt fill (fill.cpp) against the Studio's (tools/mg_sway.cloth_fill), inside the hips.
    python fill_compare.py <picture.be> <fill.bin> <hips: x,y x,y ... (9)> [back]   → numbers + fill_compare.png"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tools'))
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
import mg_sway as SW
W, H = 320, 240
be, fb = sys.argv[1], sys.argv[2]
pts = [tuple(int(v) for v in p.split(',')) for p in sys.argv[3:12]]
back = len(sys.argv) > 12 and sys.argv[12] == 'back'
def rgb(a): a = a.astype(np.uint32); return np.dstack([((a >> 11) & 31) * 255 // 31, ((a >> 5) & 63) * 255 // 63, (a & 31) * 255 // 31]).astype(np.float32)
def bits(fn): return np.unpackbits(np.fromfile(fn, np.uint8), bitorder='little')[:W * H].reshape(H, W).astype(bool)
img = rgb(np.fromfile(be, '>u2').reshape(H, W))
raw = np.fromfile(fb, np.uint8); fw = rgb(raw[:W * H * 2].view('<u2').reshape(H, W)); hole = raw[W * H * 2:].reshape(H, W).astype(bool)
py = SW.cloth_fill(img, hole, bits(fb + '.forbid'), bits(fb + '.skirt'), bits(fb + '.body'), pts, back)
m = hole & binary_erosion(SW.hips_mask(pts), iterations=1)
d = np.abs(fw[m] - np.clip(py[m], 0, 255))
print(f"{m.sum()} px inside the hips: mean |diff| {d.mean():.2f}, 95th pct {np.percentile(d, 95):.1f}, max {d.max():.0f}")
ys, xs = np.nonzero(hole); y0, y1, x0, x1 = max(0, ys.min() - 4), min(H, ys.max() + 5), max(0, xs.min() - 4), min(W, xs.max() + 5)
a = img.copy(); a[hole] = np.clip(py[hole], 0, 255); b = img.copy(); b[hole] = fw[hole]
s = 4; t = [Image.fromarray(x[y0:y1, x0:x1].astype(np.uint8)).resize(((x1 - x0) * s, (y1 - y0) * s), Image.NEAREST) for x in (a, b)]
c = Image.new('RGB', (t[0].width * 2 + 6, t[0].height)); c.paste(t[0], (0, 0)); c.paste(t[1], (t[0].width + 6, 0)); c.save('fill_compare.png')
print('fill_compare.png: Studio (left), firmware (right)')
