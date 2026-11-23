# private: the rear-view hiker scene, set up through the Studio's own model → .jig (what the Core2 loads)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'tools'))     # (the repo's Studio tools)
import numpy as np, cv2
from scipy import ndimage as nd
from PIL import Image
import jiggle_lab as J, studio_model as SM, mg_sway as SW
M = dict(np.load(os.path.join(HERE, 'masks.npz')))
img = np.asarray(Image.open(os.path.join(HERE, 'rear320.png')).convert('RGB'))
st = SM.Studio()
st.open(os.path.join(HERE, 'rear.mg'))
st.par.grid = [(-40.0, 238.0), (360.0, 238.0), (195.3, 112.0), (124.7, 112.0)]     # the path back to the clearing
st.par.prm.update(amount=9.0, ambient=0.35, ground=1.0)
print('horizon', st.horizon())
def fig(name, mask, depth=None, feet=None):
    st.add_figure(name); L = st.layer(); L.obj.mask = mask.copy()
    if depth is not None: L.obj.depth = depth; L.obj.fixed = True
    if feet is not None: L.obj.feet = feet
fig('Her', M['her'], feet=(160, 228))
fig('Boulder', M['boulder'], feet=(270, 205))
fig('Trees', M['trees'], depth=0.14)
fig('Hills', M['hills'] & ~M['trees'], depth=0.06)
for i in range(len(st.par.planes)): print(st.par.planes[i].name, round(st.figure_depth(i), 3))
st.add_water('Lake', J.W_STILL); L = st.layer(); L.obj.mask = M['lake'].copy()
L.obj.prm.update(amb=1.2, ripple=1.4, slosh=1.0, glint=0.5, edge=3.0)
# jiggle: her seat, under the skirt
for x in (151.5, 168.5):
    st.add_zone(J.ROUND, x, 124.0, 'Seat'); z = st.layer().obj; z.r = 9.0; z.material = J.NATURAL
# her hair: the fall down her back, three locks from the nape
st.add_sway('hair', 'Hair'); L = st.layer(); L.obj.mask = M['hair'].copy(); L.obj.on = 'Her'
for g in [[(151,42),(150,54),(151,66),(154,78)], [(160,42),(160,56),(160,70),(160,86)], [(169,42),(170,54),(169,66),(166,78)]]:
    st.sway_guide(g)
# what's behind the fall (the Studio's 'what's under'): her neck and upper back in her skin, her tank top below
# its neckline; beside her neck, the scene (painted in from round it)
import mg_sway as SWm
yy, xx = np.mgrid[:240, :320]
hz = nd.binary_dilation(M['hair'], iterations=2)
f32 = img.astype(np.float32); skin = SWm.is_skin(f32)
skin_c = np.median(f32[M['her'] & ~hz & skin & (yy >= 58) & (yy < 80)], 0)
top_c = np.median(f32[M['her'] & ~hz & ~skin & (yy >= 85) & (yy < 100) & (f32[..., 0] > f32[..., 1] + 40)], 0)
under = img.copy()
behind_body = hz & (((yy < 58) & (np.abs(xx - 160) <= 7)) | (yy >= 58)) & (yy >= 50)
under[behind_body & (yy >= 50) & (yy < 63)] = skin_c.astype(np.uint8)   # (near its roots: the hair itself - it barely moves there)
under[behind_body & (yy >= 63)] = top_c.astype(np.uint8)
bg = (hz & ~behind_body & (yy >= 50)).astype(np.uint8) * 255
under = cv2.inpaint(np.ascontiguousarray(under[..., ::-1]), bg, 5, cv2.INPAINT_TELEA)[..., ::-1]
deep = nd.binary_erosion(M['hair'], iterations=3)         # (inside the fall: more of her hair, the layer under it, in shadow)
under = np.ascontiguousarray(under); under[deep] = (img[deep].astype(np.float32) * 0.8).astype(np.uint8)
Image.fromarray(under).save(os.path.join(HERE, 'hair_under.png'))
st.load_under(np.ascontiguousarray(under))
# her skirt: seen from behind, its hips lined up on her
st.add_sway('cloth', 'Skirt'); L = st.layer(); L.obj.mask = M['skirt'].copy(); L.obj.on = 'Her'
st.sway_guide([(144.0, 108.0), (177.0, 108.0)])
print('fitted hips', [tuple(round(v) for v in p) for p in st.hips()])
L.obj.arm = [(144, 108), (138, 119), (142, 139), (159, 139), (160, 133), (161, 139), (178, 139), (182, 119), (177, 108)]
st.set_back(True)
txt = st.jig_text()
open(os.path.join(HERE, 'rear.jig'), 'w').write(txt)
print(len(txt.splitlines()), 'lines;', sorted(set(l.split(' ')[0] for l in txt.splitlines())))
