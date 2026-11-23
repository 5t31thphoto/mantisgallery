"""Side by side: two or more runs of play.cpp (e.g. before / after a change) → one MP4, 20 fps.
    python compare.py <out.mp4> <x0,y0,x1,y1 crop | full> <dir>=<label> <dir>=<label> ...   (needs ffmpeg)"""
import os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
out, crop, runs = sys.argv[1], sys.argv[2], [a.split('=', 1) for a in sys.argv[3:]]
x0, y0, x1, y1 = (0, 0, 320, 240) if crop == 'full' else map(int, crop.split(','))
s = max(1, min(4, 300 // max(1, x1 - x0)))
cw, ch = (x1 - x0) * s, (y1 - y0) * s
try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
except Exception: font = ImageFont.load_default()
cap = {int(l.split()[0]): " ".join(l.split()[2:]) for l in open(os.path.join(runs[0][0], 'script.txt'))}
W, H = len(runs) * (cw + 4) - 4, ch + 54
W += W % 2; H += H % 2
p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '20', '-i', '-',
                      '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
for k in sorted(cap):
    c = Image.new('RGB', (W, H), (18, 18, 22)); d = ImageDraw.Draw(c)
    for j, (dr, lab) in enumerate(runs):
        a = np.fromfile(os.path.join(dr, f'f{k:04d}.raw'), '<u2').reshape(240, 320).astype(np.uint32)
        im = np.dstack([((a >> 11) & 31) * 255 // 31, ((a >> 5) & 63) * 255 // 63, (a & 31) * 255 // 31]).astype(np.uint8)
        c.paste(Image.fromarray(im[y0:y1, x0:x1]).resize((cw, ch), Image.NEAREST), (j * (cw + 4), 26))
        d.text((j * (cw + 4) + 4, 4), lab, fill=(240, 240, 240), font=font)
    d.text((4, H - 24), cap[k], fill=(200, 200, 200), font=font)
    p.stdin.write(c.tobytes())
p.stdin.close(); p.wait(); print(out)
