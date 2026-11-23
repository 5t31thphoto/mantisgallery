"""play.cpp's frames → a captioned GIF (1.5×, one palette) and MP4 (3×, sharp pixels).
    python video.py <frames dir> <out name without extension>   (needs ffmpeg on the PATH)"""
import glob, os, subprocess, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
src, out = sys.argv[1], sys.argv[2]
cap = {int(l.split()[0]): (float(l.split()[1]), " ".join(l.split()[2:])) for l in open(os.path.join(src, 'script.txt'))}
def font(n):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'C:/Windows/Fonts/segoeui.ttf', '/System/Library/Fonts/Helvetica.ttc'):
        try: return ImageFont.truetype(p, n)
        except Exception: pass
    return ImageFont.load_default()
def frames():
    for fn in sorted(glob.glob(os.path.join(src, 'f*.raw'))):
        k = int(os.path.basename(fn)[1:-4]); a = np.fromfile(fn, '<u2').reshape(240, 320).astype(np.uint32)
        yield k, Image.fromarray(np.dstack([((a >> 11) & 31) * 255 // 31, ((a >> 5) & 63) * 255 // 63, (a & 31) * 255 // 31]).astype(np.uint8))
def card(k, im, s, fs):
    c = Image.new('RGB', (320 * s, 240 * s + 14 * s), (18, 18, 22)); c.paste(im.resize((320 * s, 240 * s), Image.NEAREST), (0, 0))
    t, w = cap.get(k, (0, '')); ImageDraw.Draw(c).text((4 * s, 240 * s + 2 * s), f"{t:5.1f}s  {w}", fill=(235, 235, 235), font=fs)
    return c
tmp = tempfile.mkdtemp()
f2 = font(15); f3 = font(22)
p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '960x762', '-r', '20', '-i', '-',
                      '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out + '.mp4'], stdin=subprocess.PIPE)
for k, im in frames():
    card(k, im, 2, f2).save(os.path.join(tmp, f'f{k:04d}.png'))
    c3 = card(k, im, 3, f3)
    p.stdin.write(c3.tobytes())
p.stdin.close(); p.wait()
vf = "scale=480:-1:flags=neighbor"
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '20', '-i', os.path.join(tmp, 'f%04d.png'), '-vf', vf + ',palettegen=max_colors=256:stats_mode=full', os.path.join(tmp, 'pal.png')], check=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '20', '-i', os.path.join(tmp, 'f%04d.png'), '-i', os.path.join(tmp, 'pal.png'),
                '-lavfi', f'[0:v]{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle', out + '.gif'], check=True)
print(out + '.gif', out + '.mp4')
