# private: the hiker scene, set up through the Studio's own model → .jig (what the Core2 loads)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'tools'))     # (the repo's Studio tools)
import numpy as np
from scipy import ndimage as nd
import jiggle_lab as J, studio_model as SM, mg_sway as SW
M = dict(np.load(os.path.join(HERE, 'masks.npz')))
img = np.asarray(__import__('PIL.Image', fromlist=['Image']).open(os.path.join(HERE, 'hiker320.png')).convert('RGB'))
H, W = 240, 320
def poly(pts):
    m = np.zeros((H, W), bool); J.mask_poly(m, pts); return m
lum = img.astype(float).mean(-1)
shoeR = poly([(177,207),(187,206),(193,214),(195,224),(193,231),(177,231)]) & (lum < 200)
her = nd.binary_fill_holes(M['her'] | nd.binary_closing(shoeR, iterations=1) | M['hair'] | M['skirt'])
# (her ink outline goes with her - else it stays behind on the scene, a ghost of her edge when she slides)
her = her | nd.binary_dilation(her, iterations=1) | (nd.binary_dilation(her, iterations=3) & (lum < 120))

st = SM.Studio()
st.open(os.path.join(HERE, 'hiker.mg'))
# ground: the trail and the lake run back to the far shore; horizon about at her chest
st.par.grid = [(-40.0, 238.0), (360.0, 238.0), (192.0, 122.0), (128.0, 122.0)]
st.par.prm.update(amount=9.0, ambient=0.35, ground=1.0)
print('horizon', st.horizon())
def fig(name, mask, depth=None, feet=None):
    st.add_figure(name); L = st.layer(); L.obj.mask = mask.copy()
    if depth is not None: L.obj.depth = depth; L.obj.fixed = True
    if feet is not None: L.obj.feet = feet
fig('Her', her, feet=(162, 229))
fig('Boulder', M['boulder'], feet=(30, 199))
fig('Trees', M['trees'], depth=0.16)
fig('Mountains', M['mount'] & ~M['trees'], depth=0.07)
for i in range(len(st.par.planes)): print(st.par.planes[i].name, round(st.figure_depth(i), 3))
# water: the falls (a river pouring down) and the lake (still)
st.add_water('Falls', J.W_RIVER); L = st.layer(); L.obj.mask = M['falls'].copy()
L.obj.lines = [[(215, 58), (215, 72), (214, 86), (213, 100)], [(241, 54), (241, 66), (240, 80)]]
L.obj.prm.update(speed=2.2, amb=1.4, edge=1.0, glint=0.6)
st.add_water('Lake', J.W_STILL); L = st.layer(); L.obj.mask = M['lake'].copy()
L.obj.prm.update(amb=1.2, ripple=1.4, slosh=1.0, glint=0.5, edge=3.0)
# jiggle: her chest
for x in (154.0, 166.0):
    st.add_zone(J.ROUND, x, 98.0, 'Chest'); z = st.layer().obj; z.r = 7.5; z.material = J.NATURAL
# hair and skirt (on her)
st.add_sway('hair', 'Hair'); L = st.layer(); L.obj.mask = M['hair'].copy(); L.obj.on = 'Her'
print('locks', st.suggest_locks())
st.layer().obj.guides = []
for g in [[(143,62),(140,72),(136,82),(133,92)], [(143,72),(140,82),(138,90),(136,98)]]:
    st.sway_guide(g)
print('locks now', len(st.layer().obj.guides))
# what's behind the swinging fall: the scene and her arm, painted in from round it (the Studio's 'what's under')
import cv2
hm = nd.binary_dilation(M['hair'], iterations=3).astype(np.uint8) * 255
under = cv2.inpaint(np.ascontiguousarray(img[..., ::-1]), hm, 4, cv2.INPAINT_TELEA)[..., ::-1]
deep = nd.binary_erosion(M['hair'], iterations=3)         # (inside the fall: more of her hair, the layer under it, in shadow)
under = np.ascontiguousarray(under); under[deep] = (img[deep].astype(np.float32) * 0.8).astype(np.uint8)
st.load_under(np.ascontiguousarray(under))
__import__('PIL.Image', fromlist=['Image']).fromarray(np.ascontiguousarray(under)).save(os.path.join(HERE, 'hair_under.png'))
st.add_sway('cloth', 'Skirt'); L = st.layer(); L.obj.mask = M['skirt'].copy(); L.obj.on = 'Her'
st.sway_guide([(148.0, 118.0), (172.0, 118.0)])
print('hips', [tuple(round(v) for v in p) for p in st.hips()])
txt = st.jig_text()
open(os.path.join(HERE, 'hiker.jig'), 'w').write(txt)
print(len(txt.splitlines()), 'lines;', sorted(set(l.split(' ')[0] for l in txt.splitlines())))
