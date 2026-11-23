"""A picture → what the scene builders and the harness use, exactly as the Core2 gets it (tools/mg_convert):
<name>.mg (the gallery file), <name>.be (its pixels big-endian, for the harness), <name>320.png (to look at).
    python convert.py <picture.jpg> <out dir> <name>"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tools'))
from pathlib import Path
import numpy as np
import mg_convert as C
src, out, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
C.convert_one(src, out / f'{name}.mg', 'cover')
im = C.prepare_image(src, 'cover'); im.save(out / f'{name}320.png')
np.frombuffer(C.to_rgb565_bytes(im), '<u2').astype('>u2').tofile(out / f'{name}.be')
