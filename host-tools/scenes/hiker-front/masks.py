# private: the hiker scene's masks (hand-placed outlines, snapped to the picture)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'tools'))     # (the repo's Studio tools)
import numpy as np
from PIL import Image
from scipy import ndimage as nd
import jiggle_lab as J, mg_select as S
W, H = 320, 240
img = np.asarray(Image.open(os.path.join(HERE, 'hiker320.png')).convert('RGB'))
f = img.astype(np.float32); R, G, B = f[..., 0], f[..., 1], f[..., 2]
mx, mn = f.max(-1), f.min(-1); sat = (mx - mn) / np.maximum(mx, 1); lum = f.mean(-1)
def poly(pts):
    m = np.zeros((H, W), bool); J.mask_poly(m, pts); return m
def clean(m, close=1, opn=1, minsz=12):
    if close: m = nd.binary_closing(m, iterations=close)
    if opn: m = nd.binary_opening(m, iterations=opn)
    lab, n = nd.label(m)
    if n:
        sz = nd.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, [i + 1 for i, s in enumerate(sz) if s >= minsz])
    return nd.binary_fill_holes(m) if close else m

# --- her: a rough outline → GrabCut
her_rough = poly([(161,44),(169,45),(175,50),(178,57),(179,66),(179,78),(182,82),(185,86),(187,95),(187,110),(188,125),
    (189,137),(190,145),(188,152),(183,153),(182,162),(183,174),(184,186),(186,200),(187,208),(191,214),(193,222),(192,230),
    (179,230),(177,220),(178,208),(175,195),(172,184),(169,172),(165,159),(161,154),(158,160),(155,172),(153,184),(152,196),
    (152,208),(154,215),(152,225),(141,230),(132,229),(133,220),(140,213),(142,205),(142,192),(143,180),(143,170),(140,160),
    (139,153),(135,152),(132,147),(133,138),(135,128),(136,115),(136,104),(133,98),(130,93),(131,85),(134,78),(138,70),
    (143,62),(147,54),(152,47)])
sure = poly([(156,52),(168,52),(170,72),(176,86),(178,120),(180,148),(172,150),(176,180),(180,205),(183,220),(170,152),
    (162,150),(152,150),(148,172),(147,205),(147,222),(140,226),(144,190),(142,150),(140,140),(146,120),(146,86),(154,76)])
her = S.cut(img, sure, avoid=~nd.binary_dilation(her_rough, iterations=2), prior=her_rough, reach=4)
her = clean(her & nd.binary_dilation(her_rough, iterations=2), close=1, opn=0, minsz=200)

# --- her hair: dark brown inside her, above the shoulders' line (and its long fall on her right)
hair_box = poly([(148,42),(178,43),(182,58),(181,88),(168,88),(158,80),(150,88),(146,100),(128,100),(127,84),(140,58)])
dark = (lum < 110) & (B < R + 12)
face = np.zeros((H, W), bool); yy, xx = np.mgrid[:H, :W]
face = ((xx - 162.0) / 9.5) ** 2 + ((yy - 65.0) / 12.5) ** 2 <= 1.0
hair = clean(nd.binary_dilation(her, iterations=2) & hair_box & dark & ~nd.binary_dilation(face, iterations=3), close=1, opn=0, minsz=30)
# (the hair lying on her shoulder stays put - it's what hangs free beside her that swings)
hair &= ~poly([(141,79),(153,77),(153,101),(141,101)])
# (just its outer fall swings: the hair right by her face stays, so what shows behind the fall is more of her hair)
hair &= (xx <= 143)
# (and its wisps - the fine strand ends off the fall)
hair |= nd.binary_dilation(hair, iterations=9) & (lum < 125) & (B < R + 20) & (xx <= 143) & (yy >= 58) & (yy <= 102)
hair = clean(hair, close=0, opn=0, minsz=30)
# --- her skirt: teal inside her, waist to hem
skirt_box = poly([(144,115),(178,115),(186,135),(188,150),(184,154),(138,154),(133,149),(137,130)])
teal = (B > R + 8) & (G > R)
skirt = clean(her & skirt_box & teal, close=2, opn=1, minsz=100)
# (her hands are in front of its sides: it stops at them, so they stay put when it swings)
skirt &= ~(poly([(181,134),(196,134),(196,156),(181,156)]) | poly([(124,134),(139,134),(139,156),(124,156)]))

# --- the falls: the white columns (and their mist) in the cliff
falls_box = poly([(206,58),(224,56),(224,98),(214,104),(204,98)]) | poly([(236,52),(246,52),(245,82),(236,82)])
falls = clean(falls_box & (lum > 175) & (sat < 0.22), close=1, opn=0, minsz=20)
# --- the lake (and the stream into it): blue-green water below the far shore, not her, not rocks/grass
lake_box = poly([(96,122),(150,118),(196,116),(200,104),(212,100),(216,108),(208,118),(320,119),(320,212),(262,205),(220,196),
    (196,188),(170,180),(150,168),(130,150),(110,140),(96,132)])
wcol = (B > R + 12) & (B >= G - 12) & (lum > 60)
lake = clean(lake_box & wcol & ~nd.binary_dilation(her, iterations=1), close=2, opn=1, minsz=60)

# --- the boulder (left, near) and the trees (left / right stands), the mountains (far)
boulder = poly([(0,132),(12,128),(30,129),(45,136),(56,148),(62,160),(62,178),(56,192),(44,198),(24,200),(0,196)])
boulder = S.snap(img, boulder) & nd.binary_dilation(boulder, iterations=2)
treesL = poly([(0,0),(36,0),(40,30),(48,60),(54,24),(62,28),(70,64),(74,96),(70,118),(54,118),(30,104),(16,104),(0,110)])
treesR = poly([(230,96),(244,58),(254,52),(262,74),(276,92),(286,58),(296,50),(306,78),(320,70),(320,118),(280,120),(236,118)])
green = (G > R) & (G > B - 6) & (lum < 170)
trees = clean((treesL | treesR) & green, close=2, opn=1, minsz=60)
sky = clean((B > R + 10) & (lum > 150) | ((lum > 215) & (sat < 0.12)), close=1, opn=1, minsz=40)
sky &= (np.arange(H)[:, None] < 70)
lab, n = nd.label(sky); keep = set(lab[0][lab[0] > 0]) | set(lab[:20][lab[:20] > 0].ravel()); sky = np.isin(lab, list(keep))
mount = poly([(0,0),(320,0),(320,120),(0,120)]) & ~sky & ~trees & (np.arange(H)[:, None] < 118) & ~her
mount = clean(mount, close=1, opn=1, minsz=200)
np.savez_compressed(os.path.join(HERE, 'masks.npz'), her=her, hair=hair, skirt=skirt, falls=falls, lake=lake, boulder=boulder,
                    trees=trees, sky=sky, mount=mount)
cols = dict(her=(255,140,0), hair=(255,0,200), skirt=(255,230,0), falls=(255,255,255), lake=(0,200,255), boulder=(160,90,40),
            trees=(0,255,90), sky=(120,160,255), mount=(140,140,140))
ov = f.copy()
for k in ['sky','mount','trees','boulder','lake','falls','her','hair','skirt']:
    m = locals()[k]; ov[m] = ov[m] * 0.45 + np.array(cols[k]) * 0.55
    print(k, int(m.sum()))
Image.fromarray(ov.astype(np.uint8)).resize((960, 720), Image.NEAREST).save(os.path.join(HERE, 'masks.png'))
