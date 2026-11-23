# private: the rear-view hiker scene's masks (hand-placed outlines, snapped to the picture)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'tools'))     # (the repo's Studio tools)
import numpy as np
from PIL import Image
from scipy import ndimage as nd
import jiggle_lab as J, mg_select as S
W, H = 320, 240
img = np.asarray(Image.open(os.path.join(HERE, 'rear320.png')).convert('RGB'))
f = img.astype(np.float32); R, G, B = f[..., 0], f[..., 1], f[..., 2]
mx, mn = f.max(-1), f.min(-1); sat = (mx - mn) / np.maximum(mx, 1); lum = f.mean(-1)
yy, xx = np.mgrid[:H, :W]
def poly(pts):
    m = np.zeros((H, W), bool); J.mask_poly(m, pts); return m
def clean(m, close=1, opn=1, minsz=12, fill=True):
    if close: m = nd.binary_closing(m, iterations=close)
    if opn: m = nd.binary_opening(m, iterations=opn)
    lab, n = nd.label(m)
    if n:
        sz = nd.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, [i + 1 for i, s in enumerate(sz) if s >= minsz])
    return nd.binary_fill_holes(m) if (close and fill) else m

her_rough = poly([(160,16),(170,18),(176,26),(178,40),(179,55),(186,60),(190,70),(191,90),(190,110),(191,125),(192,138),
    (187,143),(181,142),(180,160),(179,180),(178,200),(180,212),(182,222),(180,230),(164,230),(163,215),(164,200),(162,185),
    (161,170),(160,160),(159,170),(158,185),(156,200),(157,212),(156,228),(141,230),(140,218),(143,205),(142,185),(141,165),
    (141,146),(139,142),(133,143),(129,136),(130,120),(130,100),(131,80),(133,62),(142,58),(144,40),(146,26),(151,19)])
sure = poly([(155,24),(166,24),(172,50),(180,62),(186,90),(186,130),(176,138),(174,170),(174,205),(170,220),(166,205),
    (163,160),(157,160),(154,205),(150,220),(146,205),(146,170),(145,138),(134,130),(134,90),(138,64),(148,52)])
her = S.cut(img, sure, avoid=~nd.binary_dilation(her_rough, iterations=2), prior=her_rough, reach=4)
her = clean(her & nd.binary_dilation(her_rough, iterations=2), close=1, opn=0, minsz=200)
# (her ink outline goes with her)
import mg_sway as SWm
skin = SWm.is_skin(f)
# (the path between her calves isn't her)
gap = poly([(159.5,141),(161.5,141),(163.5,165),(164.5,185),(165,210),(155.5,210),(156.5,185),(157.5,165)])
her &= ~gap
her = clean(her, close=0, opn=1, minsz=200)
her = her | nd.binary_dilation(her, iterations=2) | (nd.binary_dilation(her, iterations=3) & (lum < 80) & (sat < 0.55))

# her hair: the long brown fall down her back (the top of her head stays: it doesn't hang)
hair_box = poly([(142,30),(178,30),(179,58),(176,70),(170,84),(160,90),(150,84),(144,70),(141,58)])
brown = (R > G) & (G > B) & (lum < 150) & (R - B > 25)
hair = clean(her & hair_box & brown, close=1, opn=0, minsz=60)
# her skirt: the dark teal from the waistband to the hem, stopping at her hands
skirt_box = poly([(142,104),(180,104),(188,125),(190,140),(184,143),(136,143),(130,138),(134,122)])
teal = (G > R + 10) & (B > R + 5) & (lum < 120)
skirt = clean(her & skirt_box & teal, close=2, opn=1, minsz=100)
skirt &= ~(poly([(181,118),(196,118),(196,146),(181,146)]) | poly([(124,118),(139,118),(139,146),(124,146)]))

# the lake: blue-green water on the left, not the lupines or rocks
lake_box = poly([(0,112),(40,110),(80,110),(125,114),(134,118),(132,135),(122,150),(100,158),(70,162),(55,172),(40,200),(0,206)])
wcol = (B > R + 8) & (lum > 55) & ~((R > G) & (B > G))      # (not the purple flowers)
lake = clean(lake_box & wcol & ~nd.binary_dilation(her, iterations=1), close=2, opn=1, minsz=60)
# the boulder (right, near)
boulder_r = poly([(222,140),(236,112),(262,92),(296,76),(320,74),(320,205),(300,205),(270,200),(246,186),(226,168)])
boulder = S.snap(img, boulder_r) & nd.binary_dilation(boulder_r, iterations=2)
boulder = clean(boulder, close=2, opn=1, minsz=200)
# trees: the left stand and the right stand (above the boulder)
treesL = poly([(0,0),(78,0),(92,40),(100,70),(110,96),(100,106),(60,108),(0,106)])
treesR = poly([(226,0),(320,0),(320,74),(296,76),(270,86),(240,96),(226,80),(232,40)])
green = (G > R - 10) & (G > B - 10) & (lum < 175)
trees = clean((treesL | treesR) & (green | (lum < 60)), close=2, opn=1, minsz=60)
sky = clean(((B > R + 15) & (lum > 140)) | ((lum > 215) & (sat < 0.12)), close=1, opn=1, minsz=40)
sky &= (yy < 75)
lab, n = nd.label(sky); keep = set(lab[:12][lab[:12] > 0].ravel()); sky = np.isin(lab, list(keep))
hills = (yy < 112) & ~sky & ~trees & ~her & ~boulder
hills = clean(hills, close=1, opn=1, minsz=200)
np.savez_compressed(os.path.join(HERE, 'masks.npz'), her=her, hair=hair, skirt=skirt, lake=lake, boulder=boulder,
                    trees=trees, sky=sky, hills=hills)
cols = dict(her=(255,140,0), hair=(255,0,200), skirt=(255,230,0), lake=(0,200,255), boulder=(160,90,40),
            trees=(0,255,90), sky=(120,160,255), hills=(140,140,140))
ov = f.copy()
for k in ['sky','hills','trees','boulder','lake','her','hair','skirt']:
    m = locals()[k]; ov[m] = ov[m] * 0.45 + np.array(cols[k]) * 0.55
    print(k, int(m.sum()))
Image.fromarray(ov.astype(np.uint8)).resize((960, 720), Image.NEAREST).save(os.path.join(HERE, 'masks.png'))
