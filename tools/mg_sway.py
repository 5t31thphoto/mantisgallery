"""Sway: hair locks, pleated cloth and pendants for Mantis Gallery (the reference the Core2 follows).

Spec: things anchored along a line sway - the root stays put, the far end swings - and rigid things
on a pin swing as one piece under real gravity (tip the Core2 and they hang straight down). All of it
answers tilt, shakes, the motion echo and music; the person's body is a collider; folds and overlaps
are part of the model, drawn back to front.

  * Hair: each lock is a guide curve the user draws root → tip. The guide is a rod of nodes:
    follow-the-leader inextensibility (Müller, Kim, Chentanez 2012), a pull back to the photo's pose
    that is gravity's own for a hanging thing (g / distance from the pin), shape memory relative to the
    parent link (the curl stays a curl), Verlet at a fixed 120 Hz. Every selected pixel is bound to its
    nearest guide by (s along it, d across it) and drawn on a triangle strip that follows the guide
    (Live2D-style skinning along a deform path). Nothing outside the strips ever moves.
  * Cloth: guides hang from the waistband at the fold lines (found from the shading along the cloth);
    the fabric between two guides is a pleat panel of fixed material width. Squeezed, a panel turns out
    of the picture (width L·cos φ) and shades accordingly; a hinge spring pulls it back to its fold in
    the photo (nonzero rest angle, Bridson et al. 2003); it can't open past flat or close past the
    fabric's thickness (pleats meet and stack). Panels are drawn back to front by their tilt.
  * Pendant: one rigid piece on a pin; the full nonlinear pendulum about the pin with its own inertia,
    balanced in the photo's pose, so real gravity turns it (tipped 90°, it hangs down).
  * Body: a mask of the person, turned into a distance field; a node (or a pendant's outline point)
    that starts outside the body can't go into it - pushed back out along the gradient (PBD contact).
  * What it uncovers: filled when prepared by exemplar patches from around it (Criminisi et al. 2004).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

W, H = 320, 240
SWAY_MAX = 4
GUIDE_MAX = 6
GUIDE_PTS = 16
# The hips under a skirt: an outline you line up with her. Points, in order round it:
#   0 waist L · 1 hip L · 2 left leg outer (at the hem) · 3 left leg inner · 4 between the legs ·
#   5 right leg inner · 6 right leg outer · 7 hip R · 8 waist R
HIP_POINTS = 9
HIP_NAMES = ("waist", "hip", "leg", "leg", "crotch", "leg", "leg", "hip", "waist")
SR_MAX = 9                           # rods (cloth: fold lines, ≤ 8 panels)
KINDS = ("hair", "cloth", "pendant")
DT = 1.0 / 120.0
BEND_MAX = math.radians(55)         # a joint bends at most this far past its bend in the photo
ON_BODY_REVEAL = 0.3                # lying on the body: it moves at most this × its half width (the fill there is a guess)
BIND_MAX = 12.0                     # hair farther than this from every lock isn't carried (it stays)
EXT_MAX = 6.0                       # a lock's strip runs at most this far past its tip
FREE_SWING = math.radians(30)       # hanging free: at most about this far round from its pin
CLOTH_HOLD = 0.1                    # a skirt holds its shape by gravity: its pull to the photo's pose scales with
                                    # how much of gravity still points down the picture (fabric's own stiffness: this floor)
CLOTH_SWING = math.pi               # a skirt: no limit - turned upside down it falls right up (only her hips stop it)
G_PX = 1500.0                       # gravity, px/s² (a 60 px lock swings about once in 1.25 s)
#          key       label                          lo   hi   default
SPARAMS = [("swing", "Swing", 0.0, 2.5, 1.0),
           ("breeze", "Breeze while still", 0.0, 1.5, 0.25),
           ("stiff", "Stiffness (keeps its shape)", 0.0, 1.0, 0.35),
           ("pleat", "Pleat depth (cloth)", 1.0, 2.5, 1.5)]
SDEF = {p[0]: p[4] for p in SPARAMS}
KIND = {  # damping 1/s, jolt lag px/s² per g, breeze px/s², gain on the change in "down", friction on the body
    "hair": dict(c=4.5, j=500.0, wind=110.0, gain=0.9, mu=0.3),
    "cloth": dict(c=3.4, j=450.0, wind=70.0, gain=1.0, mu=0.0),
    "pendant": dict(c=0.9, j=1100.0, wind=15.0, gain=1.0, mu=0.0),
}


@dataclass
class Sway:
    kind: str = "hair"
    mask: object = None                     # (H, W) bool: what moves
    guides: list = field(default_factory=list)   # hair: [[(x, y) root → tip], ...]; cloth: [[A, B] waistband]; pendant: [[pin]]
    prm: dict = field(default_factory=dict)
    name: str = ""                          # the Studio's layer name
    on: str = ""                            # the figure it rests on / collides with (its layer name)
    arm: list = field(default_factory=list)  # cloth: her hips under it, lined up by you (HIP_POINTS points; [] = fitted)

    def get(self, k):
        return self.prm.get(k, SDEF[k])

    def copy(self):
        return Sway(self.kind, None if self.mask is None else self.mask.copy(), [list(g) for g in self.guides], dict(self.prm),
                    self.name, self.on, [tuple(p) for p in self.arm])

    def ok(self):
        if self.mask is None or self.mask.sum() < 12 or not self.guides:
            return False
        if self.kind == "hair":                          # every lock root → tip
            return all(len(g) >= 2 for g in self.guides)
        return len(self.guides[0]) >= (1 if self.kind == "pendant" else 2)   # (cloth: then any marked folds, one point each)


# ---- helpers -------------------------------------------------------------------------------------------
def grow1(m):
    q = np.pad(m, 1)
    return m | q[:-2, 1:-1] | q[2:, 1:-1] | q[1:-1, :-2] | q[1:-1, 2:]


def edt(seed):
    """Euclidean distance to the nearest seed pixel."""
    try:
        from scipy.ndimage import distance_transform_edt
        if not seed.any():
            return np.full(seed.shape, 1e6, np.float32)
        return distance_transform_edt(~seed).astype(np.float32)
    except ImportError:
        pass
    from mg_depth import _nearest
    ny, nx, d = _nearest(seed, max(W, H))
    return np.sqrt(np.where(ny >= 0, d, 1e12)).astype(np.float32)


def body_sdf(body):
    """Signed distance to the body's outline: > 0 outside, < 0 inside."""
    if body is None or not body.any():
        return None
    out = edt(body)            # outside: distance to the body
    ins = edt(~body)           # inside: distance to the outside
    return np.where(body, -ins + 0.5, out - 0.5).astype(np.float32)


def neck_row(body, hair=None):
    """The row of the neck: the narrowest the figure gets between the top of its head and the
    shoulders (the widest row in its upper part). Hair is counted as not figure. None if unclear."""
    if body is None:
        return None
    b = np.asarray(body, bool) & ~(np.asarray(hair, bool) if hair is not None else False)
    rows = np.nonzero(b.any(1))[0]
    if len(rows) < 20:
        return None
    top, bot = rows.min(), rows.max()
    h = bot - top
    wd = np.array([b[y].sum() for y in range(top, bot + 1)], float)
    up = int(h * 0.45)
    if up < 8:
        return None
    sh = int(np.argmax(wd[:up]))                         # the shoulders: widest in the upper part
    if sh < 4:
        return None
    head = wd[:sh]
    hw = head[: max(2, sh // 3)].max()                  # how wide the head is
    for i in range(max(2, sh // 3), sh):                 # the neck: where it's narrowest under the head
        if wd[i] <= 0.8 * hw and wd[i] == head[max(2, sh // 3):sh].min():
            return int(top + i)
    return int(top + max(2, sh // 3) + int(np.argmin(head[max(2, sh // 3):sh])))


def suggest_locks(mask, body=None, spacing=10.0):
    """Lock strokes for a hair selection, root → tip, the way you'd draw them: the hair is split into its
    separate falls (left of the face, right of it, behind…); each gets one lock per ~`spacing` px of its
    width, traced up from its lowest point through the middle of its strand, row by row. body: the
    figure it's on - a lock starts at the neck's height (where hair leaves the head and hangs; above
    that it lies on the head and stays put)."""
    from scipy.ndimage import label
    m = np.asarray(mask, bool)
    neck = neck_row(body, m)
    lab, n = label(m)
    locks = []
    for c in range(1, n + 1):
        part = lab == c
        if part.sum() < 30:
            continue
        ys, xs = np.nonzero(part)
        y0, y1 = ys.min(), ys.max()
        if y1 - y0 < 8:
            continue
        # the strand's width a third of the way up from its tips: how many locks it gets
        yb = int(y1 - (y1 - y0) * 0.33)
        row = np.nonzero(part[yb])[0]
        w = (row.max() - row.min() + 1) if len(row) else 4
        k = int(max(1, min(GUIDE_MAX - len(locks), round(w / spacing))))
        if k <= 0:
            break
        lo, hi = (row.min(), row.max()) if len(row) else (xs.min(), xs.max())
        for j in range(k):
            fx = lo + (hi - lo) * (j + 0.5) / k
            # from the tips up: start at the lowest pixel of the strand near fx
            col = [y for y in range(y1, y0 - 1, -1) if part[y, int(round(fx))]]
            yy = col[0] if col else y1
            x = fx; pts = [(x, float(yy))]
            off = (fx - (lo + hi) / 2) / max(1.0, (hi - lo) / 2)      # where across the strand (−1..1)
            for y in range(yy - 1, y0 - 1, -1):
                r = np.nonzero(part[y])[0]
                if not len(r):
                    break
                # the run nearest x, and the same place across it
                runs = np.split(r, np.nonzero(np.diff(r) > 1)[0] + 1)
                run = min(runs, key=lambda q: min(abs(q[0] - x), abs(q[-1] - x)))
                a, b = run[0], run[-1]
                if abs((a + b) / 2 - x) > 6 and not (a <= x <= b):
                    break
                nx = (a + b) / 2 + off * (b - a) / 2 * 0.8
                x = x + max(-1.5, min(1.5, nx - x))
                pts.append((x, float(y)))
            pts = pts[::-1]
            if neck is not None:                         # it hangs from where it leaves the head: the neck's
                pts = [p for p in pts if p[1] >= neck] or pts[-2:]   # height (above that it lies on the head and stays)
            if len(pts) < 4:
                continue
            step = max(1, len(pts) // 12)
            pl = pts[::step]
            if pl[-1] != pts[-1]:
                pl.append(pts[-1])
            locks.append([(round(px, 1), round(py, 1)) for px, py in pl[:GUIDE_PTS]])
            if len(locks) >= GUIDE_MAX:
                return locks
    return locks


def fit_hips(skirt, body=None, waist=None):
    """The hips' outline fitted to a skirt (and her legs under its hem): a start for you to line up.
    Points as HIP_POINTS says."""
    sk = np.asarray(skirt, bool)
    ys = np.nonzero(sk.any(1))[0]
    if not len(ys):
        return []
    top, bot = int(ys.min()), int(ys.max())
    if waist is not None and len(waist) >= 2:                 # (the waistband you set: its two ends)
        (wa, ty_a), (wb, ty_b) = sorted([tuple(waist[0]), tuple(waist[1])])
        top = int(round(min(ty_a, ty_b)))
    else:
        r0 = np.nonzero(sk[top:top + max(2, (bot - top) // 10)].any(0))[0]
        wa, wb = float(r0.min()), float(r0.max())
    xa, xb = np.nonzero(sk.any(0))[0][[0, -1]]
    bd = np.asarray(body, bool) if body is not None else np.zeros_like(sk)
    legs = bd & ~sk
    cols = np.zeros(W, bool)
    for y in range(bot + 1, min(H, bot + 8)):
        cols[max(0, xa - 2):min(W, xb + 3)] |= legs[y, max(0, xa - 2):min(W, xb + 3)]
    runs = np.split(np.nonzero(cols)[0], np.nonzero(np.diff(np.nonzero(cols)[0]) > 1)[0] + 1) if cols.any() else []
    runs = sorted([r for r in runs if len(r) >= 2], key=len)[-2:]
    runs = sorted(runs, key=lambda r: r[0])
    c, w = (wa + wb) / 2, wb - wa
    if len(runs) == 2:
        lo, li, ri, ro = float(runs[0][0]), float(runs[0][-1]), float(runs[1][0]), float(runs[1][-1])
    elif len(runs) == 1:
        lo, ro = float(runs[0][0]), float(runs[0][-1])
        li = ri = (lo + ro) / 2
    else:
        lo, ro = c - w * 0.4, c + w * 0.4
        li, ri = c - w * 0.05, c + w * 0.05
    # body proportions: the hip line about 0.4 waist-widths under the waist, the point between the legs
    # about 0.85 (well above the hem of anything knee-length)
    yh = min(top + w * 0.4, top + (bot - top) * 0.45)
    hl, hr = min(wa, lo) - w * 0.06, max(wb, ro) + w * 0.06
    yc = min(top + w * 0.85, bot - 2.0)
    return [(wa, float(top)), (hl, yh), (lo, float(bot)), (li, float(bot)), ((li + ri) / 2, yc),
            (ri, float(bot)), (ro, float(bot)), (hr, yh), (wb, float(top))]


def bikini_zone(pts):
    """Safety: the bikini area of the hips' outline - round the point between the legs (node 4), from a
    little above it down past it, as wide as the gap between the legs and a bit more. Wherever a skirt
    swinging away would show skin there, it shows a white bikini bottom instead (with pink polka dots).
    Returns (H, W) bool."""
    from PIL import Image, ImageDraw
    m = Image.new("L", (W, H), 0)
    if len(pts) == HIP_POINTS:
        q = bikini_poly(pts)
        ImageDraw.Draw(m).polygon(q, fill=1)
    return np.asarray(m, bool)


def bikini_poly(pts):
    """A bikini bottom on the hips' outline: its top edge low across the hips (at the hip line, the
    full width of the outline there), the sides curving in like leg openings, down to a narrow panel
    just past the point between the legs (node 4)."""
    (w0x, w0y), (h0x, h0y), (l0x, l0y), (lix, liy), (cx, cy), (rix, riy), (r0x, r0y), (h1x, h1y), (w1x, w1y) = pts
    hy = (h0y + h1y) / 2
    ty = hy + 0.1 * max(2.0, cy - hy)                  # its top: just under the hip line
    def side(ax, ay, bx, by):                          # the outline's x at height ty, along a side
        return ax + (bx - ax) * ((ty - ay) / (by - ay) if by != ay else 0.0)
    xl = side(h0x, h0y, l0x, l0y) + 1.0
    xr = side(h1x, h1y, r0x, r0y) - 1.0
    gap = max(3.0, rix - lix)
    by = cy + 0.5                                      # down to the point between the legs
    wb = min(gap * 0.3 + 1.0, 3.0)                     # the narrow panel there
    left, right = [], []
    for k in range(6):                                 # leg openings: steep at first, then in
        t = k / 5
        y = ty + (by - ty) * t
        f = (1 - t) ** 1.6
        left.append((cx - wb - (cx - wb - xl) * f, y))
        right.append((cx + wb + (xr - cx - wb) * f, y))
    return left + right[::-1]


def bikini_color(x, y):
    """White, with small pink polka dots on a 6 px grid."""
    gx, gy = (x % 6) - 2.5, (y % 6) - 2.5
    return (255.0, 150.0, 190.0) if gx * gx + gy * gy <= 2.0 else (238.0, 238.0, 238.0)


def hips_mask(pts):
    """Her body under the skirt: the hips' outline filled (and the legs straight down from it)."""
    from PIL import Image, ImageDraw
    m = Image.new("L", (W, H), 0)
    if len(pts) == HIP_POINTS:
        ImageDraw.Draw(m).polygon([(float(x), float(y)) for x, y in pts], fill=1)
    return np.asarray(m, bool)


def is_skin(px):
    """Skin-coloured (YCbCr box, Chai & Ngan 1999), for (..., 3) RGB float."""
    px = np.asarray(px, np.float32)
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    cr = 128 + 0.5 * r - 0.4187 * g - 0.0813 * b
    cb = 128 - 0.1687 * r - 0.3313 * g + 0.5 * b
    return (cr >= 136) & (cr <= 173) & (cb >= 77) & (cb <= 124) & (r > g) & (r > b + 10)


def skin_tone(photo, legs, body, above=None):
    """Her skin colour: the skin-coloured pixels of her arms, shoulders and face (above `above`, the
    skirt's top: never the ground a loose mask took in), else of her legs, else a default."""
    img = np.asarray(photo, np.float32)
    cands = []
    if body is not None and body.any():
        b = np.asarray(body, bool).copy()
        if above is not None:
            b[int(above):] = False
        cands.append(b)
    if legs is not None:
        cands.append(np.asarray(legs, bool))
    for m in cands:
        if m.any():
            px = img[m]
            px = px[is_skin(px)]
            if len(px) >= 12:
                return np.median(px, 0)
    return np.array([224.0, 172.0, 140.0], np.float32)


def cloth_fill(photo, hole, forbid, skirt=None, body=None, hips=None):
    """Behind a skirt, as it really is behind one (object permanence) - so it can swing, fly or fall
    right up (turned upside down) and what shows is always true:
    - inside the hips' outline (lined up by you): her body. Her legs and thighs in her own skin, each
      column carried up from her leg just under the hem (else her skin tone), rounded off darker toward
      the outline; across her pelvis a white bikini bottom with pink polka dots (safety - always).
    - beside her, inside the skirt's swing: the scene behind her (filled from the scene, never from her).
    - the soft rings just outside the skirt: whatever is right next to them (leg, hand, scene)."""
    img = np.asarray(photo, np.float32)
    from scipy.ndimage import distance_transform_edt
    hole = np.asarray(hole, bool)
    _, (ny, nx) = distance_transform_edt(hole, return_indices=True)
    out = img[ny, nx].copy()                                 # (the rings: what's right next to them)
    if skirt is None or not skirt.any():
        return out
    sk = np.asarray(skirt, bool)
    bd = np.asarray(body, bool) if body is not None else np.zeros_like(sk)
    ys = np.nonzero(sk.any(1))[0]
    top, bot = int(ys.min()), int(ys.max())
    legs = bd & ~sk & ~hole
    pts = hips if hips and len(hips) == HIP_POINTS else fit_hips(sk, bd)
    inside = hips_mask(pts)
    scene = hole & ~inside
    rows = np.zeros_like(sk); rows[top:bot + 1] = True
    scene &= rows & sk.any(0)[None, :]
    if scene.any():                                          # beside her, inside its swing: the scene
        filled = inpaint_em(img, scene, forbid=np.asarray(forbid, bool) | bd | inside | sk)
        out[scene] = filled[scene]
    under_hem = legs.copy(); under_hem[:bot + 1] = False; under_hem[min(H, bot + 12):] = False
    tone = skin_tone(img, under_hem, bd, above=top)
    inner = distance_transform_edt(inside)                   # (rounded: darker toward her outline)
    biki = bikini_zone(pts)
    bases = np.tile(tone, (W, 1)).astype(np.float32)
    for x in range(W):                                       # each column: her leg's own shade under the hem
        col = np.nonzero(inside[:, x] & hole[:, x])[0]
        if len(col):
            under = np.nonzero(legs[col.max() + 1:min(H, col.max() + 6), x])[0]
            if len(under) and is_skin(img[col.max() + 1 + under[0], x]):
                bases[x] = img[col.max() + 1 + under[0], x] * 0.6 + tone * 0.4
    k = np.ones(7) / 7                                       # (smoothed across: skin, not stripes)
    bases = np.stack([np.convolve(np.pad(bases[:, c], 3, mode="edge"), k, "valid") for c in range(3)], 1)
    for x in range(W):
        col = np.nonzero(inside[:, x] & hole[:, x])[0]
        if not len(col):
            continue
        base = bases[x]
        for y in col:
            if biki[y, x]:
                out[y, x] = bikini_color(x, y)
                continue
            d = inner[y, x]
            out[y, x] = base * (0.82 if d < 1.5 else (0.92 if d < 3 else 1.0))
    return out


def rest_body(body, masks):
    """What the moving things rest on: the body without them, plus each of their pixels whose nearest
    pixel that doesn't move is the body (a lock over her face lies on her; one hanging past her
    against the wall is in front of the wall). body: the figures; masks: every sway's mask."""
    if body is None or not body.any():
        return None
    mv = np.zeros((H, W), bool)
    for m in masks:
        if m is not None:
            mv |= m
    bp = body & ~mv; sc = ~body & ~mv
    if not mv.any() or not bp.any():
        return body.copy()
    if not sc.any():
        return body | mv
    return bp | (mv & (edt(bp) <= edt(sc)))


def sdf_at(sdf, x, y):
    x = min(max(x, 0.0), W - 1.001); y = min(max(y, 0.0), H - 1.001)
    ix, iy = int(x), int(y); fx, fy = x - ix, y - iy
    a = sdf[iy, ix] * (1 - fx) + sdf[iy, ix + 1] * fx
    b = sdf[iy + 1, ix] * (1 - fx) + sdf[iy + 1, ix + 1] * fx
    v = a * (1 - fy) + b * fy
    gx = (sdf[iy, ix + 1] - sdf[iy, ix]) * (1 - fy) + (sdf[iy + 1, ix + 1] - sdf[iy + 1, ix]) * fy
    gy = (sdf[iy + 1, ix] - sdf[iy, ix]) * (1 - fx) + (sdf[iy + 1, ix + 1] - sdf[iy, ix + 1]) * fx
    l = math.hypot(gx, gy) or 1.0
    return float(v), gx / l, gy / l


def resample(pts, step, nmax):
    pts = np.asarray(pts, np.float64)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    L = float(seg.sum())
    n = int(min(nmax, max(3, round(L / step) + 1)))
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    s = np.linspace(0, L, n)
    return np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], 1)


def bind(xs, ys, poly):
    """For pixels (xs, ys): distance to the polyline, arc length s of the closest point, signed d (left +)."""
    best = np.full(len(xs), 1e18); S = np.zeros(len(xs)); D = np.zeros(len(xs)); cum = 0.0
    for i in range(len(poly) - 1):
        ax, ay = poly[i]; bx, by = poly[i + 1]
        dx, dy = bx - ax, by - ay; l2 = dx * dx + dy * dy; l = math.sqrt(l2)
        t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / l2, 0 if i else -np.inf, 1 if i < len(poly) - 2 else np.inf)
        px, py = ax + t * dx, ay + t * dy
        dd = (xs - px) ** 2 + (ys - py) ** 2
        take = dd < best
        best[take] = dd[take]; S[take] = cum + t[take] * l
        D[take] = ((xs - ax) * (-dy) + (ys - ay) * dx)[take] / l
        cum += l
    return np.sqrt(best), S, D


def inpaint(img, hole, patch=7, win=28):
    """Exemplar-based fill (Criminisi, Pérez, Toyama 2004): fill the hole's edge patch with the most
    structure first (confidence × the isophote meeting the edge), copying the best-matching patch of
    known pixels from nearby; repeat until it's filled."""
    img = img.astype(np.float32).copy(); hole = hole.copy()
    r = patch // 2
    conf = (~hole).astype(np.float32)
    gray = img.mean(-1)
    known = ~hole
    ys, xs = np.nonzero(hole)
    if not len(xs):
        return img
    y0, y1 = max(r, ys.min() - win), min(H - r - 1, ys.max() + win)
    x0, x1 = max(r, xs.min() - win), min(W - r - 1, xs.max() + win)
    for _ in range(int(hole.sum()) + 10):
        if not hole.any():
            break
        q = np.pad(hole, 1)
        front = hole & ~(q[:-2, 1:-1] & q[2:, 1:-1] & q[1:-1, :-2] & q[1:-1, 2:])
        fy, fx = np.nonzero(front)
        # priority
        gy_, gx_ = np.gradient(np.where(known, gray, 0))
        pr = []
        for y, x in zip(fy, fx):
            ya, yb, xa, xb = max(0, y - r), min(H, y + r + 1), max(0, x - r), min(W, x + r + 1)
            c = conf[ya:yb, xa:xb].sum() / ((2 * r + 1) ** 2)
            pr.append(c * (1e-3 + abs(gx_[y, x]) + abs(gy_[y, x])))
        k = int(np.argmax(pr)); y, x = int(fy[k]), int(fx[k])
        y = min(max(y, r), H - r - 1); x = min(max(x, r), W - r - 1)
        tp = img[y - r:y + r + 1, x - r:x + r + 1]; tk = known[y - r:y + r + 1, x - r:x + r + 1]
        best, bp = 1e30, None
        for cy in range(max(r, y - win), min(H - r - 1, y + win) + 1, 1):
            for cx in range(max(r, x - win), min(W - r - 1, x + win) + 1, 1):
                if not known[cy - r:cy + r + 1, cx - r:cx + r + 1].all():
                    continue
                d = ((img[cy - r:cy + r + 1, cx - r:cx + r + 1] - tp) ** 2).sum(-1)[tk].sum()
                if d < best:
                    best, bp = d, (cy, cx)
        sl = hole[y - r:y + r + 1, x - r:x + r + 1]
        if bp is None:
            img[y, x] = img[known].mean(0); hole[y, x] = False; known[y, x] = True; continue
        cy, cx = bp
        src = img[cy - r:cy + r + 1, cx - r:cx + r + 1]
        img[y - r:y + r + 1, x - r:x + r + 1][sl] = src[sl]
        cval = conf[y - r:y + r + 1, x - r:x + r + 1].mean()
        conf[y - r:y + r + 1, x - r:x + r + 1][sl] = cval
        known[y - r:y + r + 1, x - r:x + r + 1][sl] = True
        hole[y - r:y + r + 1, x - r:x + r + 1][sl] = False
    return img


def inpaint_fast(img, hole, patch=7, win=24, forbid=None):
    """The same exemplar fill, vectorised: candidate patches from a window around the hole, matched by
    SSD over the known part of each target patch."""
    from numpy.lib.stride_tricks import sliding_window_view as swv
    img = img.astype(np.float32).copy(); hole = hole.copy(); known = ~hole
    srcok = known if forbid is None else known & ~forbid     # what may be copied in (not the thing itself)
    r = patch // 2
    ys, xs = np.nonzero(hole)
    if not len(xs):
        return img
    ya, yb = max(0, ys.min() - win), min(H, ys.max() + win + 1)
    xa, xb = max(0, xs.min() - win), min(W, xs.max() + win + 1)
    sub = img[ya:yb, xa:xb]; subk = srcok[ya:yb, xa:xb]
    P = swv(sub, (patch, patch), axis=(0, 1))            # (h', w', 3, p, p)
    K = swv(subk, (patch, patch)).all(axis=(-1, -2))     # fully known candidates
    cy, cx = np.nonzero(K)
    cand = P[cy, cx].reshape(len(cy), 3, -1)             # (n, 3, p²)
    conf = known.astype(np.float32)
    gray = img.mean(-1)
    guard = int(hole.sum()) + 10
    while hole.any() and guard > 0:
        guard -= 1
        q = np.pad(hole, 1)
        front = hole & ~(q[:-2, 1:-1] & q[2:, 1:-1] & q[1:-1, :-2] & q[1:-1, 2:])
        fy, fx = np.nonzero(front)
        gy_, gx_ = np.gradient(np.where(known, gray, gray[known].mean()))
        cs = np.array([conf[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].mean() for y, x in zip(fy, fx)])
        pr = cs * (1e-3 + np.abs(gx_[fy, fx]) + np.abs(gy_[fy, fx]))
        k = int(np.argmax(pr))
        y = min(max(int(fy[k]), r), H - r - 1); x = min(max(int(fx[k]), r), W - r - 1)
        tp = img[y - r:y + r + 1, x - r:x + r + 1].transpose(2, 0, 1).reshape(3, -1)
        tk = known[y - r:y + r + 1, x - r:x + r + 1].reshape(-1)
        near = (np.abs(cy + ya + r - y) <= win) & (np.abs(cx + xa + r - x) <= win)
        if not near.any() or not tk.any():
            img[y, x] = img[known].mean(0); hole[y, x] = False; known[y, x] = True; conf[y, x] = 0.5; continue
        c = cand[near]
        d = ((c[:, :, tk] - tp[None, :, tk]) ** 2).sum((1, 2))
        j = int(np.argmin(d))
        src = c[j].reshape(3, patch, patch).transpose(1, 2, 0)
        sl = hole[y - r:y + r + 1, x - r:x + r + 1].copy()
        img[y - r:y + r + 1, x - r:x + r + 1][sl] = src[sl]
        conf[y - r:y + r + 1, x - r:x + r + 1][sl] = conf[y - r:y + r + 1, x - r:x + r + 1].mean()
        known[y - r:y + r + 1, x - r:x + r + 1][sl] = True
        hole[y - r:y + r + 1, x - r:x + r + 1][sl] = False
    return img


def matte(photo, mask, plate):
    """Edge alpha and carried colour: the outline pixel and two rings past it are split c = a·H + (1−a)·B
    (H: nearest solid pixel of it, B: what's behind it - the plate), so it carries its own colour."""
    from mg_depth import _nearest
    r1 = grow1(mask); r2 = grow1(r1)
    c = np.pad(mask, 1, mode="edge")
    edge = mask & ~(c[:-2, 1:-1] & c[2:, 1:-1] & c[1:-1, :-2] & c[1:-1, 2:])
    solid = mask & ~edge
    if not solid.any():
        solid = mask
    ny, nx, _ = _nearest(solid, 5)
    soft = np.where(mask, np.where(edge, 0.82, 1.0), np.where(r1, 0.47, np.where(r2, 0.18, 0.0)))
    band = (soft > 0) & (soft < 1) & (ny >= 0)
    Hc = photo[np.clip(ny, 0, H - 1), np.clip(nx, 0, W - 1)].astype(np.float32)
    cc, B = photo.astype(np.float32), plate.astype(np.float32)
    dv = Hc - B; dd = (dv * dv).sum(-1)
    a = ((cc - B) * dv).sum(-1) / np.maximum(dd, 100.0)
    hi = np.where(edge, 1.0, np.where(r1, 0.8, 0.45)); lo = np.where(edge, 0.4, 0.0)
    a = np.where(dd < 300, soft, np.clip(a, lo, hi))
    alpha = np.where(solid, 1.0, np.where(band, a, np.where(soft > 0, soft, 0.0))).astype(np.float32)
    texd = photo.astype(np.float32).copy(); texd[band] = Hc[band]
    return alpha, texd, r2


# ---- the simulation ------------------------------------------------------------------------------------
class SwaySim:
    def __init__(self, sw: Sway, photo, body=None, sdf=None, forbid=None, plate=None):
        """plate: a fill made before ((H, W, 3) image, (H, W) where it has one) - from a saved splate or
        an earlier run; used where it has one (and the rest filled as usual)."""
        self.s = sw
        self.ok = sw is not None and sw.ok()
        if not self.ok:
            return
        self.K = KIND[sw.kind]
        self.photo = np.asarray(photo, np.float32)
        self.sdf = sdf if sdf is not None else body_sdf(body)
        self.body = body if body is not None else (self.sdf < 0 if self.sdf is not None else None)
        self.t = 0.0; self.acc = 0.0; self.awake = True; self.fade = 0.0
        {"hair": self._prep_hair, "cloth": self._prep_cloth, "pendant": self._prep_pendant}[sw.kind]()
        if sw.kind == "cloth":                          # her hips under it: lined up by you, else fitted
            self.arm = sw.arm if len(sw.arm) == HIP_POINTS else fit_hips(sw.mask, self.body, sw.guides[0] if sw.guides else None)
            self.hipm, self.hipl, self.hipr = self.hips_rows(self.arm)
        if sw.kind != "pendant":                        # only what its rest mesh covers can move (or be uncovered)
            cov = np.full((H, W), -1, np.int32)
            for D, S_, oid, _, _ in self._mesh():
                self._cover(cov, S_, oid)
            self.cover = cov
            self.moving &= (cov >= 0)
            self.owner = np.where(self.moving, cov, -1)                # carried by the mesh piece it lies in
            self.hole = grow1(grow1(self.moving)) & (cov >= 0)      # (the selection and the two rings the matte carries)
        # behind it: everything that can move away, filled from around it; its edge matted
        fb = grow1(grow1(sw.mask)) | (forbid if forbid is not None else False)
        given = None
        if plate is not None:
            pimg, have = plate
            if (self.hole & ~have).sum() == 0:          # (it has all of it: nothing to fill)
                given = np.where(have[..., None], np.asarray(pimg, np.float32), self.photo)
        if given is not None:
            self.plate = given
        elif sw.kind == "cloth":
            self.plate = cloth_fill(self.photo, self.hole, fb, self.moving, self.body, self.arm)
        elif self.body is not None:                       # behind what lies on her is her; behind what hangs free is the scene
            onb = self.hole & self.body
            plate = inpaint_em(self.photo, self.hole & ~self.body, forbid=fb | self.body)
            if onb.any():
                # Behind hair lying on her is the skin / cloth right next to it - never her dress from a
                # hand's width away: sources only from her own pixels close around the hole (widened only
                # if there's too little there).
                near = edt(self.hole)
                for reach in (8.0, 14.0, 24.0, 1e9):
                    src = self.body & ~fb & (near <= reach)
                    if src.sum() >= 60 + 2 * onb.sum() ** 0.5:
                        break
                plate = inpaint_em(plate, onb, forbid=~src | fb, win=max(10, int(min(reach, 40))))
            self.plate = plate
        else:
            self.plate = inpaint_em(self.photo, self.hole, forbid=fb)
        self.fill_hole = self.hole.copy()              # (where the fill is used: what a saved splate carries)
        self.alpha, self.texd, _ = matte(self.photo, self.moving, self.plate)
        if self.s.kind != "pendant":                    # the rings past it belong to its nearest lock / panel
            from mg_depth import _nearest
            ny, nx, _ = _nearest(self.owner >= 0, 4)
            ring = (self.alpha > 0) & (self.owner < 0) & (ny >= 0) & self.hole
            own = self.owner.copy()
            own[ring] = self.owner[ny[ring], nx[ring]]
            self.owner = np.where(ring & (self.cover != own), -1, own)   # (a ring pixel its lock's mesh can't carry stays)
            self.hole &= self.owner >= 0                                  # every uncovered pixel is carried by something
            self.alpha = np.where(self.hole, self.alpha, 0.0)

    # .. hair: rods along the guides, pixels bound to their nearest guide
    def _prep_hair(self):
        m = self.s.mask
        ys, xs = np.nonzero(m)
        self.rods = []
        for g in self.s.guides[:GUIDE_MAX]:
            L = float(np.hypot(*np.diff(np.asarray(g, float), axis=0).T).sum())
            self.rods.append(resample(g, max(4.0, L / 9), 12))
        dist = np.stack([bind(xs, ys, r)[0] for r in self.rods])
        own = dist.argmin(0)
        own[dist.min(0) > BIND_MAX] = -1                   # (too far from every lock: it stays)
        self.owner = np.full((H, W), -1, np.int32)
        self.bs = np.zeros((H, W), np.float32); self.bd = np.zeros((H, W), np.float32)
        self.hw = []
        moving = np.zeros((H, W), bool)
        for k, r in enumerate(self.rods):
            sel = own == k
            _, S, D = bind(xs[sel], ys[sel], r)
            mov = S > 0.5                                   # past the root: it moves
            self.owner[ys[sel][mov], xs[sel][mov]] = k
            self.bs[ys[sel], xs[sel]] = S; self.bd[ys[sel], xs[sel]] = D
            moving[ys[sel][mov], xs[sel][mov]] = True
            seg = np.hypot(*np.diff(r, axis=0).T); cum = np.concatenate(([0], np.cumsum(seg)))
            # half widths at each node (left +, right −) from the pixels it carries
            lw = np.full(len(r), 2.0); rw = np.full(len(r), 2.0)
            Sm, Dm = S[mov], D[mov]
            for i in range(len(r)):
                near = np.abs(Sm - cum[i]) <= max(seg.max(), 3)
                if near.any():                  # (95th percentile: a stray bit of something else doesn't widen the lock)
                    pos, neg = Dm[near][Dm[near] > 0], -Dm[near][Dm[near] < 0]
                    lw[i] = max(2.0, float(np.percentile(pos, 95)) + 2.5) if len(pos) else 2.0
                    rw[i] = max(2.0, float(np.percentile(neg, 95)) + 2.5) if len(neg) else 2.0
            ext = min(EXT_MAX, max(0.0, float(Sm.max() - cum[-1]) + 2.5)) if len(Sm) else 2.5
            self.hw.append((lw, rw, ext))
        self.moving = moving
        self.hole = grow1(grow1(grow1(moving)))
        self.rest = [r.copy() for r in self.rods]
        self.x = [r.copy() for r in self.rods]; self.xp = [r.copy() for r in self.rods]
        self.lens = [np.hypot(*np.diff(r, axis=0).T) for r in self.rods]
        self.kn = [G_PX / np.maximum(np.concatenate(([1e9], np.cumsum(l))), 4.0) for l in self.lens]
        self.rest_sdf = [np.array([sdf_at(self.sdf, *p)[0] for p in r]) if self.sdf is not None else None for r in self.rods]
        self.on_body = [(rs <= 1.0) if rs is not None else np.zeros(len(r), bool) for rs, r in zip(self.rest_sdf, self.rods)]
        self.cap = []
        for k, r in enumerate(self.rods):
            dist = np.concatenate(([0.0], np.cumsum(self.lens[k])))
            lw, rw, _ = self.hw[k]
            free = 2.0 * dist * math.sin(FREE_SWING / 2) + 1e-3
            onb = np.maximum(2.0, ON_BODY_REVEAL * (lw + rw) * 0.5)
            self.cap.append(np.where(self.on_body[k], np.minimum(onb, free), free))

    # .. cloth: guides from the waistband at the fold lines, pleat panels between them
    def hips_rows(self, pts):
        """Per row: the hips' left and right edge (−1 where none) - what the skirt's edges can't go inside."""
        hm = hips_mask(pts)
        hl = np.full(H, -1.0); hr = np.full(H, -1.0)
        for y in range(H):
            r = np.nonzero(hm[y])[0]
            if len(r):
                hl[y], hr[y] = r.min(), r.max()
        return hm, hl, hr

    def _prep_cloth(self):
        m = self.s.mask
        (ax, ay), (bx, by) = self.s.guides[0][:2]
        ex, ey = bx - ax, by - ay; l = math.hypot(ex, ey) or 1.0; ex, ey = ex / l, ey / l
        nx_, ny_ = -ey, ex
        if ny_ < 0:
            nx_, ny_ = -nx_, -ny_
        ys, xs = np.nonzero(m)
        S = (xs - ax) * ex + (ys - ay) * ey; T = (xs - ax) * nx_ + (ys - ay) * ny_
        below = T > 0
        s0, s1 = float(S[below].min()) - 1.5, float(S[below].max()) + 1.5
        tmax = float(T[below].max()) + 1.5
        # fold lines: valleys of brightness across the cloth (averaged down the lower two-thirds)
        lum = self.photo.mean(-1)
        bins = np.arange(int(s0), int(s1) + 1)
        prof = np.array([lum[ys[below & (np.abs(S - b) < 1.0) & (T > tmax / 3)], xs[below & (np.abs(S - b) < 1.0) & (T > tmax / 3)]].mean()
                         if (below & (np.abs(S - b) < 1.0) & (T > tmax / 3)).any() else np.nan for b in bins])
        prof = np.where(np.isnan(prof), np.nanmean(prof), prof)
        sm = np.convolve(prof, np.ones(5) / 5, mode="same")
        cands = [i for i in range(3, len(sm) - 3) if sm[i] < sm[i - 3] - 2 and sm[i] < sm[i + 3] - 2 and sm[i] == sm[i - 3:i + 4].min()]
        cuts = [s0]
        marked = [g[0] for g in self.s.guides[1:GUIDE_MAX]]
        if marked:                                         # the folds you marked (any point on each fold line)
            for fx_, fy_ in marked[:SR_MAX - 2]:
                c = (fx_ - ax) * ex + (fy_ - ay) * ey
                if s0 + 3 < c < s1 - 3:
                    cuts.append(float(c))
        else:
            for i in sorted(cands, key=lambda i: sm[i]):
                b = float(bins[i])
                if all(abs(b - c) >= 9 for c in cuts) and s1 - b >= 9:
                    cuts.append(b)
        cuts = sorted(cuts) + [s1]
        while len(cuts) > 9:                               # (at most 8 panels: merge the narrowest)
            k = int(np.argmin(np.diff(cuts)[1:-1])) + 1 if len(cuts) > 3 else 1
            del cuts[k]
        if len(cuts) < 4 and not marked:                   # no clear folds (and none marked): even panels ~12 px
            n = max(3, int((s1 - s0) / 12))
            cuts = list(np.linspace(s0, s1, n + 1))
        self.cuts = cuts
        nr = 6
        ts = [tmax * j / nr for j in range(nr + 1)]
        self.rods = [np.array([(ax + c * ex + t * nx_, ay + c * ey + t * ny_) for t in ts]) for c in cuts]
        self.rest = [r.copy() for r in self.rods]
        self.x = [r.copy() for r in self.rods]; self.xp = [r.copy() for r in self.rods]
        self.lens = [np.full(nr, tmax / nr) for _ in self.rods]
        self.kn = [G_PX / np.maximum(np.array(ts), 4.0) for _ in self.rods]
        self.W0 = np.diff(cuts)                            # panel width on screen at rest
        self.Lm = self.W0 * self.s.get("pleat")           # its material width: flat, it'd be this wide
        self.cos0 = self.W0 / self.Lm
        self.thick = 1.5
        self.owner = np.full((H, W), -1, np.int32)
        moving = np.zeros((H, W), bool)
        for k in range(len(cuts) - 1):
            sel = below & (S >= cuts[k]) & (S < cuts[k + 1] + (k == len(cuts) - 2))
            self.owner[ys[sel], xs[sel]] = k
            moving[ys[sel], xs[sel]] = True
        self.frame = (ax, ay, ex, ey, nx_, ny_, tmax)
        self.moving = moving
        self.hole = grow1(grow1(grow1(moving)))
        self.rest_sdf = [np.array([sdf_at(self.sdf, *p)[0] for p in r]) if self.sdf is not None else None for r in self.rods]
        self.on_body = [(rs <= 1.0) if rs is not None else np.zeros(len(r), bool) for rs, r in zip(self.rest_sdf, self.rods)]
        self.cap = [np.array(ts) * 2.0 * math.sin(CLOTH_SWING / 2) + 1e-3 for _ in self.rods]
        # round the body: each row's axis, radius (its edges ~70° round) and each fold line's angle
        e = np.array([ex, ey]); self.cyl = [None]
        for j in range(1, nr + 1):
            xs_ = np.array([r[j] @ e for r in self.rods]); c = float((xs_[0] + xs_[-1]) * 0.5)
            R = float((xs_[-1] - xs_[0]) * 0.5) / math.sin(math.radians(70))
            self.cyl.append((c, R, np.arcsin(np.clip((xs_ - c) / R, -0.999, 0.999))))

    # .. pendant: one rigid piece on its pin
    def _prep_pendant(self):
        m = self.s.mask
        px, py = self.s.guides[0][0]
        ys, xs = np.nonzero(m)
        rx, ry = xs - px, ys - py
        self.pin = (float(px), float(py))
        self.mass = float(len(xs))
        self.rc = (float(rx.mean()), float(ry.mean()))
        self.I = float((rx * rx + ry * ry).sum()) + self.mass * 1.0
        self.th = 0.0; self.thp = 0.0
        # balanced in the photo: the photo's gravity (0, 1) exerts no net turn at rest
        self.tau0 = -self.mass * G_PX * (self.rc[0] * 1.0 - self.rc[1] * 0.0)
        # outline points that collide (the far ones)
        c = np.pad(m, 1); edge = m & ~(c[:-2, 1:-1] & c[2:, 1:-1] & c[1:-1, :-2] & c[1:-1, 2:])
        ey_, ex_ = np.nonzero(edge)
        ang = np.arctan2(ey_ - py, ex_ - px)
        pts = []
        for a in np.linspace(-math.pi, math.pi, 17)[:-1]:
            k = np.abs(((ang - a + math.pi) % (2 * math.pi)) - math.pi) < 0.25
            if k.any():
                d = np.hypot(ex_[k] - px, ey_[k] - py); i = int(np.argmax(d))
                pts.append((float(ex_[k][i]), float(ey_[k][i])))
        self.outline = pts
        self.rest_sdf = [sdf_at(self.sdf, *p)[0] for p in pts] if self.sdf is not None else None
        self.moving = m.copy()
        self.hole = grow1(grow1(grow1(m)))
        self.tlen = float(np.hypot(rx, ry).max())

    # ---- physics ---------------------------------------------------------------------------------------
    def kick(self, amp, sign):
        """A beat: the free ends get a sideways kick (music)."""
        if not self.ok:
            return
        if self.s.kind == "pendant":
            self.thp -= sign * amp * 90.0 / max(self.tlen, 4) * DT
        else:
            for x, xp, r in zip(self.x, self.xp, self.rest):
                d = r[-1] - r[0]; l = math.hypot(*d) or 1.0
                nx_, ny_ = -d[1] / l, d[0] / l
                w = np.linspace(0, 1, len(x))[:, None] ** 1.5
                xp -= sign * amp * 120.0 * DT * w * np.array([nx_, ny_])
        self.awake = True

    def poke(self, px, py):
        """A tap on it: a flick away from the tap."""
        if not self.ok or not self.moving[min(max(int(py), 0), H - 1), min(max(int(px), 0), W - 1)]:
            return False
        if self.s.kind == "pendant":
            side = (px - self.pin[0]) * self.rc[1] - (py - self.pin[1]) * self.rc[0]
            self.thp += (1 if side > 0 else -1) * 200.0 / max(self.tlen, 4) * DT
        else:
            for x, xp, r in zip(self.x, self.xp, self.rest):
                d = np.hypot(x[:, 0] - px, x[:, 1] - py)
                f = np.clip(1 - d / 30.0, 0, 1)[:, None]
                if not f.any():
                    continue
                dd = r[-1] - r[0]; l = math.hypot(*dd) or 1.0
                nx_, ny_ = -dd[1] / l, dd[0] / l
                side = (px - r[0][0]) * nx_ + (py - r[0][1]) * ny_
                xp += (1 if side > 0 else -1) * 150.0 * DT * f * np.array([nx_, ny_])
        self.awake = True
        return True

    def _collide(self, k, x):
        if self.sdf is None:
            return
        rs = self.rest_sdf[k]
        for i in range(1, len(x)):
            if rs[i] <= 1.0:                    # it starts on the body (lies in front of it): free
                continue
            v, gx, gy = sdf_at(self.sdf, *x[i])
            if v < 1.0:                         # into the body: back out along the gradient
                x[i, 0] += (1.0 - v) * gx; x[i, 1] += (1.0 - v) * gy

    def _rods_step(self, fx, fy, wind, damp_k, hold=1.0):
        K, h = self.K, DT
        damp = 1.0 - K["c"] * damp_k * h
        stiff = self.s.get("stiff")
        cloth = self.s.kind == "cloth"
        e = None
        if cloth:
            ax, ay, ex, ey, nx_, ny_, tmax = self.frame
            e = (ex, ey)
        for k, (x, xp, r, kn, ln) in enumerate(zip(self.x, self.xp, self.rest, self.kn, self.lens)):
            n = len(x)
            wdir = e if cloth else (-(r[-1] - r[0])[1], (r[-1] - r[0])[0])
            wl = math.hypot(*wdir) or 1.0
            wk = np.linspace(0.3, 1.0, n)[:, None] * wind * np.array([wdir[0] / wl, wdir[1] / wl])
            a = np.array([fx, fy]) + (hold * kn)[:, None] * (r - x) + wk
            on = self.on_body[k]
            if K["mu"] > 0 and on.any():      # lying on the body: rubbing on it takes some of the push and slows it
                ext = a - (hold * kn)[:, None] * (r - x)
                a[on] -= ext[on] * min(0.8, 1.5 * K["mu"])
                xp[on] += (x[on] - xp[on]) * min(0.9, 8.0 * K["mu"] * h)
            new = x + (x - xp) * damp + a * h * h
            new[0] = r[0]
            xp[:] = x; x[:] = new
        if cloth:
            self._panels()
        for k, (x, r, ln) in enumerate(zip(self.x, self.rest, self.lens)):
            # shape memory: each link turns with the one above it (the curl stays a curl)
            if stiff > 0 and not cloth:
                for i in range(2, len(x)):
                    c = x[i - 1] - x[i - 2]; rr = r[i - 1] - r[i - 2]
                    ang = math.atan2(c[1], c[0]) - math.atan2(rr[1], rr[0])
                    cs, sn = math.cos(ang), math.sin(ang)
                    v = r[i] - r[i - 1]
                    tgt = x[i - 1] + np.array([v[0] * cs - v[1] * sn, v[0] * sn + v[1] * cs])
                    x[i] += (tgt - x[i]) * (0.35 * stiff)
            if not cloth:                                  # (a skirt hangs round her: it swings in front of
                self._collide(k, x)                        # her legs, it doesn't collide with them)
            # follow the leader: every link exactly its length, from the root down - and no joint bends
            # more than BEND_MAX past its bend in the photo (a lock folds, but doesn't flip over itself)
            for i in range(1, len(x)):
                d = x[i] - x[i - 1]; l = math.hypot(*d) or 1e-6
                if i >= 2 and not cloth:               # (fabric has no bend limit: it folds right over)
                    pc = x[i - 1] - x[i - 2]; pr = r[i - 1] - r[i - 2]; rr = r[i] - r[i - 1]
                    rest_b = math.atan2(pr[0] * rr[1] - pr[1] * rr[0], pr[0] * rr[0] + pr[1] * rr[1])
                    now_b = math.atan2(pc[0] * d[1] - pc[1] * d[0], pc[0] * d[0] + pc[1] * d[1])
                    db = (now_b - rest_b + math.pi) % (2 * math.pi) - math.pi
                    if abs(db) > BEND_MAX:
                        ang = math.atan2(pc[1], pc[0]) + rest_b + math.copysign(BEND_MAX, db)
                        d = np.array([math.cos(ang), math.sin(ang)]) * l
                # fabric: a link never stretches but can foreshorten - a fold turning toward you or away
                # (upside down the hem comes up in front of her, over the waist); hair keeps its exact length
                x[i] = x[i - 1] + d * (min(1.0, ln[i - 1] / l) if cloth else ln[i - 1] / l)
            # what the fill can't make up limits how far it goes: lying on the body (nothing true to show
            # behind it), a node may leave its spot by a fraction of the lock's width; hanging free in
            # front of the scene, the swing is only capped at a wide angle
            cap = self.cap[k]
            dd = x - r; dm = np.hypot(dd[:, 0], dd[:, 1])
            over = dm > cap
            if over.any():
                x[over] = r[over] + dd[over] * (cap[over] / dm[over])[:, None]
        if cloth:
            self._panels()
            self._hips_collide()

    def _hips_collide(self):
        """The skirt's two edges can't swing in through her hips / thighs: an edge that hangs outside her
        hip line in the photo stays outside it (the far side of a swing hugs her)."""
        if not hasattr(self, "hipl"):
            return
        for k, side in ((0, -1), (len(self.x) - 1, 1)):
            x, r = self.x[k], self.rest[k]
            for j in range(1, len(x)):
                yr, yc = int(round(r[j][1])), int(round(x[j][1]))
                if not (0 <= yr < H and 0 <= yc < H):
                    continue
                if side < 0 and self.hipl[yr] >= 0 and r[j][0] <= self.hipl[yr] + 0.5 and self.hipl[yc] >= 0:
                    x[j][0] = min(x[j][0], self.hipl[yc])
                if side > 0 and self.hipr[yr] >= 0 and r[j][0] >= self.hipr[yr] - 0.5 and self.hipr[yc] >= 0:
                    x[j][0] = max(x[j][0], self.hipr[yc])

    def _panels(self):
        """Pleats: neighbouring guides can't open past the panel's flat width or close past the fabric's
        thickness; between those, a hinge spring pulls toward the fold in the photo."""
        ex, ey = self.frame[2], self.frame[3]
        # A skirt goes round the body: it can't slide sideways through her, so a sideways swing turns it
        # round its axis (the hem's sideways shift u becomes a turn θ = u / R). Each fold line sits at its
        # angle ψ round the body (screen offset R·sin ψ); turned, the pleats coming round to the side bunch
        # up and the ones turning to face you open out - the panels below then keep their width limits.
        e = np.array([ex, ey])
        for j in range(1, len(self.x[0])):
            c, R, psi = self.cyl[j]
            u = float(np.mean([(x[j] - r[j]) @ e for x, r in zip(self.x, self.rest)]))
            th = u / R                                  # the turn that moves it as far, on average (Newton)
            for _ in range(4):
                f_ = R * float(np.mean(np.sin(psi + th) - np.sin(psi))) - u
                df = R * float(np.mean(np.cos(psi + th))) or 1e-6
                th -= f_ / df
            th = max(-0.9, min(0.9, th))
            for k, x in enumerate(self.x):
                want = c + R * math.sin(psi[k] + th)
                cur = float(x[j] @ e)
                x[j] += (want - cur) * 0.5 * e
        # one sheet: neighbouring fold lines move together (each pulled toward its neighbours' displacement)
        # (the row as a whole keeps its shift: only how it's shared out between the folds evens up)
        n = len(self.x)
        for _ in range(2):
            for j in range(1, len(self.x[0])):
                d = [x[j] - r[j] for x, r in zip(self.x, self.rest)]
                nd = [d[0]] + [d[k] + 0.5 * ((d[k - 1] + d[k + 1]) * 0.5 - d[k]) for k in range(1, n - 1)] + [d[-1]]
                fix = (sum(d) - sum(nd)) / n
                for k in range(n):
                    self.x[k][j] = self.rest[k][j] + nd[k] + fix
        for k in range(len(self.x) - 1):
            A, B = self.x[k], self.x[k + 1]
            for j in range(1, len(A)):
                d = B[j] - A[j]; w = d[0] * ex + d[1] * ey          # across the cloth (signed: order kept)
                lo, hi, w0 = self.thick, self.Lm[k], self.W0[k]
                tgt = w + (w0 - w) * 0.12                          # hinge spring toward the photo's fold
                tgt = min(max(tgt, lo), hi)
                corr = (tgt - w) * 0.5
                A[j, 0] -= corr * ex; A[j, 1] -= corr * ey
                B[j, 0] += corr * ex; B[j, 1] += corr * ey

    def _pend_step(self, fx, fy, wind, damp_k, dn):
        """I·θ'' = m·g·(r(θ) × down) + τ0 − c·θ' (+ jolts, breeze): the full pendulum."""
        h = DT
        c, s = math.cos(self.th), math.sin(self.th)
        rx = self.rc[0] * c - self.rc[1] * s; ry = self.rc[0] * s + self.rc[1] * c
        gx, gy = G_PX * dn[0], G_PX * dn[1]
        jx, jy = fx, fy
        tq = self.mass * (rx * (gy + jy) - ry * (gx + jx)) + self.tau0 * math.hypot(dn[0], dn[1])
        tq += self.mass * wind * math.hypot(rx, ry)               # (a breeze: a small sideways push)
        acc = tq / self.I
        new = self.th + (self.th - self.thp) * (1.0 - self.K["c"] * damp_k * h) + acc * h * h
        if self.sdf is not None:                                   # its outline can't go into the body
            cs, sn = math.cos(new), math.sin(new)
            for (ox, oy), r0 in zip(self.outline, self.rest_sdf):
                if r0 <= 1.0:
                    continue
                qx, qy = ox - self.pin[0], oy - self.pin[1]
                v, _, _ = sdf_at(self.sdf, self.pin[0] + qx * cs - qy * sn, self.pin[1] + qx * sn + qy * cs)
                if v < 1.0:
                    new = self.th - (new - self.th) * 0.3           # (a soft bounce)
                    break
        self.thp, self.th = self.th, new

    def step(self, dt, dn, jolt, damp_k=1.0):
        """dn: unit down in screen axes (upright (0, 1, 0)); jolt: g, gravity removed; damp_k: < 1 while the
        motion echo plays (its swaying carries on)."""
        if not self.ok:
            return False
        K, sw = self.K, self.s.get("swing")
        m = math.hypot(dn[0], dn[1])
        self.acc += min(dt, 0.1)
        while self.acc >= DT:
            self.acc -= DT; self.t += DT
            T = 2 * math.pi * self.t
            wind = sw * self.s.get("breeze") * K["wind"] * (0.6 * math.sin(T / 3.1) + 0.4 * math.sin(T / 1.7 + 1.0))
            if self.s.kind == "pendant":
                self._pend_step(-sw * K["j"] * jolt[0], -sw * K["j"] * jolt[1], wind, damp_k, dn)
            else:
                g = sw * K["gain"]
                fx = g * (G_PX * dn[0] - K["j"] * jolt[0])
                hold = 1.0
                if self.s.kind == "cloth":
                    # the photo's pose is held up by her (waist, hips) only while gravity points down the
                    # picture: that part of gravity is carried; the rest (sideways, upside down) pulls it
                    hold = max(CLOTH_HOLD, dn[1])
                    fy = g * (G_PX * min(0.0, dn[1]) - K["j"] * jolt[1])
                else:
                    fy = g * (G_PX * (dn[1] - m) - K["j"] * jolt[1])
                self._rods_step(fx, fy, wind, damp_k, hold)
        if self.s.kind == "pendant":
            disp = abs(self.th) * self.tlen; vel = abs(self.th - self.thp) * self.tlen
        else:
            disp = max(float(np.abs(x - r).max()) for x, r in zip(self.x, self.rest))
            vel = max(float(np.abs(x - xp).max()) for x, xp in zip(self.x, self.xp))
        self.fade = min(1.0, disp)
        self.awake = disp > 0.05 or vel > 0.002 or self.s.get("breeze") > 0
        return self.awake

    # ---- drawing --------------------------------------------------------------------------------------
    def _mesh(self):
        """Triangles to draw: (dest tri, source tri, owner id, shade, depth)."""
        tris = []
        if self.s.kind == "pendant":
            ys, xs = np.nonzero(self.moving)
            x0, x1, y0, y1 = xs.min() - 3, xs.max() + 4, ys.min() - 3, ys.max() + 4
            S = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], float)
            c, s = math.cos(self.th), math.sin(self.th)
            q = S - self.pin
            D = np.stack([self.pin[0] + q[:, 0] * c - q[:, 1] * s, self.pin[1] + q[:, 0] * s + q[:, 1] * c], 1)
            return [(D[[0, 1, 2]], S[[0, 1, 2]], -2, 1.0, 0.0), (D[[0, 2, 3]], S[[0, 2, 3]], -2, 1.0, 0.0)]
        if self.s.kind == "hair":
            for k, (x, r) in enumerate(zip(self.x, self.rest)):
                lw, rw, ext = self.hw[k]
                def strip(p):
                    n = len(p)
                    t = np.zeros_like(p)
                    t[1:-1] = p[2:] - p[:-2]; t[0] = p[1] - p[0]; t[-1] = p[-1] - p[-2]
                    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-6)[:, None]
                    nrm = np.stack([-t[:, 1], t[:, 0]], 1)         # the side bind() measures d > 0 on
                    nrm[0] = rest_n0                               # (the root edge is pinned whole, not just its middle)
                    tip = p[-1] + t[-1] * ext
                    P = np.vstack([p, tip]); Nn = np.vstack([nrm, nrm[-1]])
                    lws = np.concatenate([lw, [lw[-1]]]); rws = np.concatenate([rw, [rw[-1]]])
                    return P + Nn * lws[:, None], P - Nn * rws[:, None]
                t0 = r[1] - r[0]; t0 = t0 / (np.hypot(*t0) or 1.0); rest_n0 = np.array([-t0[1], t0[0]])
                Ls, Rs = strip(r); Ld, Rd = strip(x)
                for i in range(len(Ls) - 1):
                    S4 = np.array([Ls[i], Rs[i], Rs[i + 1], Ls[i + 1]]); D4 = np.array([Ld[i], Rd[i], Rd[i + 1], Ld[i + 1]])
                    for tri in ((0, 1, 2), (0, 2, 3)):
                        tris.append((D4[list(tri)], S4[list(tri)], k, 1.0, float(k)))
            return tris
        # cloth: each panel row as two triangles; shade and depth from its fold
        ex, ey = self.frame[2], self.frame[3]
        for k in range(len(self.x) - 1):
            A, B, Ar, Br = self.x[k], self.x[k + 1], self.rest[k], self.rest[k + 1]
            for j in range(len(A) - 1):
                w = ((B[j] + B[j + 1] - A[j] - A[j + 1]) * 0.5) @ np.array([ex, ey])
                cphi = min(1.0, max(0.05, w / self.Lm[k]))
                sgn = 1 if k % 2 == 0 else -1                          # accordion: panels alternate
                light = 0.35                                           # light a little from the side
                phi, phi0 = math.acos(cphi), math.acos(min(1.0, self.cos0[k]))
                shade = max(0.25, math.cos(sgn * phi - light)) / max(0.25, math.cos(sgn * phi0 - light))
                depth = -sgn * math.sin(phi)                           # turned away: further back
                S4 = np.array([Ar[j], Br[j], Br[j + 1], Ar[j + 1]]); D4 = np.array([A[j], B[j], B[j + 1], A[j + 1]])
                for tri in ((0, 1, 2), (0, 2, 3)):
                    tris.append((D4[list(tri)], S4[list(tri)], k, shade, depth))
        return tris

    @staticmethod
    def _cover(cov, S, oid):
        x0 = max(0, int(math.floor(S[:, 0].min()))); x1 = min(W - 1, int(math.ceil(S[:, 0].max())))
        y0 = max(0, int(math.floor(S[:, 1].min()))); y1 = min(H - 1, int(math.ceil(S[:, 1].max())))
        if x0 > x1 or y0 > y1:
            return
        Y, X = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(np.float64)
        (ax, ay), (bx, by), (cx, cy) = S
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-9:
            return
        l0 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / den
        l1 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / den
        ins = (l0 >= -1e-7) & (l1 >= -1e-7) & (1 - l0 - l1 >= -1e-7)
        sl = cov[y0:y1 + 1, x0:x1 + 1]
        sl[ins & (sl < 0)] = oid

    def render(self, content):
        """content (H, W, 3) array: its plate under it, then the mesh back to front."""
        if not self.ok:
            return content
        out = content.astype(np.float32)
        hole = self.hole
        out[hole] = self.plate[hole]
        fade = self.fade
        tex = self.photo * (1 - fade) + self.texd * fade if fade > 0 else self.photo
        al = np.where(self.alpha > 0, 1.0 * (1 - fade) + self.alpha * fade, 0.0)
        owner = self.owner if self.s.kind == "hair" else None     # (cloth panels tile exactly; locks overlap)
        for D, S, oid, shade, depth in sorted(self._mesh(), key=lambda t: t[4]):
            self._tri(out, D, S, oid, shade, tex, al, owner)
        if content.dtype == np.uint8:
            content[:] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        else:
            content[:] = out
        return content

    def _tri(self, out, D, S, oid, shade, tex, al, owner):
        bx0 = max(0, int(math.floor(D[:, 0].min()))); bx1 = min(W - 1, int(math.ceil(D[:, 0].max())))
        by0 = max(0, int(math.floor(D[:, 1].min()))); by1 = min(H - 1, int(math.ceil(D[:, 1].max())))
        if bx0 > bx1 or by0 > by1:
            return
        Y, X = np.mgrid[by0:by1 + 1, bx0:bx1 + 1].astype(np.float64)
        (ax, ay), (bx, by), (cx, cy) = D
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-9:
            return
        l0 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / den
        l1 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / den
        l2 = 1 - l0 - l1
        inside = (l0 >= -1e-7) & (l1 >= -1e-7) & (l2 >= -1e-7)
        if not inside.any():
            return
        sx = np.clip(l0 * S[0, 0] + l1 * S[1, 0] + l2 * S[2, 0], 0, W - 1.001)
        sy = np.clip(l0 * S[0, 1] + l1 * S[1, 1] + l2 * S[2, 1], 0, H - 1.001)
        ix, iy = sx.astype(np.int32), sy.astype(np.int32); fx, fy = sx - ix, sy - iy
        if owner is not None:                    # only the pixels this lock / panel carries (rings included)
            inside &= owner[np.clip(np.round(sy).astype(int), 0, H - 1), np.clip(np.round(sx).astype(int), 0, W - 1)] == oid
        a = (al[iy, ix] * (1 - fx) + al[iy, ix + 1] * fx) * (1 - fy) + (al[iy + 1, ix] * (1 - fx) + al[iy + 1, ix + 1] * fx) * fy
        a = np.where(inside, a, 0.0)
        if not (a > 0).any():
            return
        f3x, f3y = fx[..., None], fy[..., None]
        col = ((tex[iy, ix] * (1 - f3x) + tex[iy, ix + 1] * f3x) * (1 - f3y) + (tex[iy + 1, ix] * (1 - f3x) + tex[iy + 1, ix + 1] * f3x) * f3y)
        col = col * shade
        sl = out[by0:by1 + 1, bx0:bx1 + 1]
        sl[:] = sl * (1 - a[..., None]) + col * a[..., None]


def inpaint_em(img, hole, forbid=None, patch=5, levels=3, iters=5, win=40):
    """Image completion by patch voting (Wexler, Shechtman, Irani 2007), coarse to fine: every hole patch
    takes its nearest known patch from around it (SSD), every hole pixel becomes the average of what the
    patches over it say, repeat - the fill agrees with itself and with its surroundings (no blocks, no
    seams). Sources never come from `forbid` (the thing that moves)."""
    img = np.asarray(img, np.float32)
    src_ok = ~hole if forbid is None else ~hole & ~forbid
    ys, xs = np.nonzero(hole)
    if not len(xs):
        return img.copy()
    y0, y1 = max(0, ys.min() - win), min(H, ys.max() + win + 1)
    x0, x1 = max(0, xs.min() - win), min(W, xs.max() + win + 1)
    I0, M0, S0 = img[y0:y1, x0:x1].copy(), hole[y0:y1, x0:x1].copy(), src_ok[y0:y1, x0:x1].copy()
    pyr = [(I0, M0, S0)]
    for _ in range(levels - 1):                       # (a 2× pyramid; a coarse pixel is a hole if any of its 4 is)
        I, M, Sk = pyr[-1]
        h2, w2 = I.shape[0] // 2, I.shape[1] // 2
        if h2 < patch * 2 or w2 < patch * 2:
            break
        I2 = I[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, 3).mean((1, 3))
        M2 = M[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).any((1, 3))
        S2 = Sk[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).all((1, 3))
        pyr.append((I2, M2, S2))
    r = patch // 2
    from numpy.lib.stride_tricks import sliding_window_view as swv
    cur = None
    for lvl in range(len(pyr) - 1, -1, -1):
        I, M, Sk = pyr[lvl]
        I = I.copy()
        h, w = M.shape
        if cur is None:                               # coarsest: start from a smooth fill
            from scipy.ndimage import distance_transform_edt
            seed = ~M & Sk if (~M & Sk).any() else ~M
            _, (ny, nx) = distance_transform_edt(~seed, return_indices=True)
            I[M] = I[ny[M], nx[M]]
            for _ in range(60):
                q = np.pad(I, ((1, 1), (1, 1), (0, 0)), mode="edge")
                I[M] = ((q[:-2, 1:-1] + q[2:, 1:-1] + q[1:-1, :-2] + q[1:-1, 2:]) / 4)[M]
        else:                                         # finer: start from the coarser answer
            up = np.repeat(np.repeat(cur, 2, 0), 2, 1)
            uh, uw = min(h, up.shape[0]), min(w, up.shape[1])
            fill = I.copy(); fill[:uh, :uw] = up[:uh, :uw]
            I[M] = fill[M]
        Ip = np.pad(I, ((r, r), (r, r), (0, 0)), mode="edge")
        Mp = np.pad(M, r, constant_values=False)
        Sp = np.pad(Sk, r, constant_values=False)
        okc = swv(Sp, (patch, patch)).all((-1, -2))    # candidate centres: a fully known, allowed patch
        cy, cx = np.nonzero(okc)
        # targets: every patch that touches the hole
        touch = swv(Mp, (patch, patch)).any((-1, -2))
        ty, tx = np.nonzero(touch)
        if not len(cy) or not len(ty):
            cur = I; continue
        wwin = max(8, win >> lvl)
        for it in range(iters):
            Ip = np.pad(I, ((r, r), (r, r), (0, 0)), mode="edge")
            P = swv(Ip, (patch, patch), axis=(0, 1))   # (h, w, 3, p, p) centred on each pixel
            C = P[cy, cx].reshape(len(cy), -1)
            T = P[ty, tx].reshape(len(ty), -1)
            cc = (C * C).sum(1)
            best = np.zeros(len(ty), np.int64); has = np.zeros(len(ty), bool)
            for a in range(0, len(ty), 512):          # nearest candidate within the window (‖t−c‖² = t² + c² − 2tc)
                tb = T[a:a + 512]
                d = cc[None, :] - 2.0 * tb @ C.T
                far = (np.abs(ty[a:a + 512, None] - cy[None, :]) > wwin) | (np.abs(tx[a:a + 512, None] - cx[None, :]) > wwin)
                none = far.all(1)                     # (none within reach: the best from anywhere allowed)
                far[none] = False
                d[far] = np.inf
                best[a:a + 512] = np.argmin(d, 1)
                has[a:a + 512] = True
            acc = np.zeros((h, w, 3), np.float32); wt = np.zeros((h, w), np.float32)
            src = P[cy[best], cx[best]]               # (n, 3, p, p)
            for dy in range(patch):                   # vote: each target patch paints its source over itself
                for dx in range(patch):
                    yy = ty + dy - r; xx = tx + dx - r
                    ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w) & has
                    np.add.at(acc, (yy[ok], xx[ok]), src[ok, :, dy, dx])
                    np.add.at(wt, (yy[ok], xx[ok]), 1.0)
            upd = M & (wt > 0)
            I[upd] = acc[upd] / wt[upd][:, None]
        if lvl == 0:                                  # last: each pixel the centre of its own best patch (sharp)
            Ip = np.pad(I, ((r, r), (r, r), (0, 0)), mode="edge")
            P = swv(Ip, (patch, patch), axis=(0, 1))
            hy, hx = np.nonzero(M)
            T = P[hy, hx].reshape(len(hy), -1)
            C = P[cy, cx].reshape(len(cy), -1); cc = (C * C).sum(1)
            for a in range(0, len(hy), 512):
                d = cc[None, :] - 2.0 * T[a:a + 512] @ C.T
                far = (np.abs(hy[a:a + 512, None] - cy[None, :]) > win) | (np.abs(hx[a:a + 512, None] - cx[None, :]) > win)
                far[far.all(1)] = False
                d[far] = np.inf
                b = np.argmin(d, 1)
                I[hy[a:a + 512], hx[a:a + 512]] = P[cy[b], cx[b]][:, :, r, r]
        cur = I
    out = img.copy()
    out[y0:y1, x0:x1][M0] = cur[M0]
    return out


# ---- .jig lines --------------------------------------------------------------------------------------
#   sway hair [swing=…] [breeze=…] [stiff=…] [pleat=…]
#   sguide x,y x,y …                 (hair: one per lock, root → tip; cloth: the waistband A B; pendant: the pin)
#   srow y a-b c-d …                 (what moves)
#   splate y x RRGGBB…               (optional, from the lab: what's behind it, pixel by pixel from x along row y,
#                                     RGB565 as 4 hex digits each; without it the Core2 makes its own)
#   body / brow y a-b …              (once per photo: the person - what things rest on and collide with)
def _rows(m, tag):
    out = []
    for y in range(H):
        row = m[y]
        if not row.any():
            continue
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        s_, e_ = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        out.append(f"{tag} {y} " + " ".join(f"{a}-{b}" for a, b in zip(s_, e_)))
    return out


def format_lines(sways, body=None, plates=None):
    out = []
    if body is not None and body.any():
        out.append("body")
        out += _rows(body, "brow")
    for i, sw in enumerate(sways[:SWAY_MAX]):
        if not sw.ok():
            continue
        kv = " ".join(f"{k}={sw.prm[k]:g}" for k in SDEF if k in sw.prm and abs(sw.prm[k] - SDEF[k]) > 1e-6)
        nm = (" name=" + sw.name.replace(" ", "_")[:15]) if sw.name else ""
        on = (" on=" + sw.on.replace(" ", "_")[:15]) if sw.on else ""
        out.append(f"sway {sw.kind}" + (" " + kv if kv else "") + nm + on)
        for g in sw.guides[:GUIDE_MAX]:
            out.append("sguide " + " ".join(f"{int(round(x))},{int(round(y))}" for x, y in g[:16]))
        if sw.kind == "cloth" and len(sw.arm) == HIP_POINTS:
            out.append("sarm " + " ".join(f"{int(round(x))},{int(round(y))}" for x, y in sw.arm))
        out += _rows(sw.mask, "srow")
        if plates is not None and plates[i] is not None:
            img, hole = plates[i]
            p = np.clip(img + 0.5, 0, 255).astype(np.uint16)
            v = ((p[..., 0] >> 3) << 11) | ((p[..., 1] >> 2) << 5) | (p[..., 2] >> 3)
            for y in range(H):
                xs = np.flatnonzero(hole[y])
                if not len(xs):
                    continue
                runs = np.split(xs, np.flatnonzero(np.diff(xs) != 1) + 1)
                for r in runs:
                    out.append(f"splate {y} {r[0]} " + "".join(f"{int(v[y, x]):04x}" for x in r))
    return out


def parse_lines(lines):
    """-> (sways, body mask or None, plates: per sway (rgb (H, W, 3), have (H, W)) or None)."""
    sways, plates, body, cur, tgt = [], [], None, None, None
    for raw in lines:
        tok = raw.split()
        if not tok:
            continue
        if tok[0] == "body":
            body = np.zeros((H, W), bool); tgt = "body"
        elif tok[0] == "sway" and len(tok) >= 2 and tok[1] in KINDS and len(sways) < SWAY_MAX:
            prm, nm, on = {}, "", ""
            for kv in tok[2:]:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    if k == "name":
                        nm = v.replace("_", " ")[:15]
                    elif k == "on":
                        on = v.replace("_", " ")[:15]
                    try:
                        if k in SDEF:
                            prm[k] = float(v)
                    except ValueError:
                        pass
            cur = Sway(tok[1], np.zeros((H, W), bool), [], prm, nm, on); sways.append(cur); plates.append(None); tgt = "sway"
        elif tok[0] == "sarm" and cur is not None and tgt == "sway":
            try:
                pts = [tuple(float(v) for v in p.split(",")) for p in tok[1:]]
                if len(pts) == HIP_POINTS:
                    cur.arm = pts
            except ValueError:
                pass
        elif tok[0] == "sguide" and cur is not None and tgt == "sway":
            try:
                cur.guides.append([tuple(float(v) for v in p.split(",")) for p in tok[1:]])
            except ValueError:
                pass
        elif tok[0] in ("srow", "brow") and len(tok) > 1:
            m = cur.mask if (tok[0] == "srow" and cur is not None) else (body if tok[0] == "brow" else None)
            if m is None:
                continue
            try:
                y = int(tok[1])
            except ValueError:
                continue
            for sp in tok[2:]:
                if "-" in sp and 0 <= y < H:
                    a, b = sp.split("-", 1)
                    m[y, max(0, int(a)):min(W, int(b))] = True
        elif tok[0] == "splate" and cur is not None and len(tok) == 4:
            y, x, hx = int(tok[1]), int(tok[2]), tok[3]
            if plates[-1] is None:
                plates[-1] = (np.zeros((H, W, 3), np.float32), np.zeros((H, W), bool))
            img, have = plates[-1]
            for i in range(len(hx) // 4):
                v = int(hx[4 * i:4 * i + 4], 16)
                if 0 <= x + i < W and 0 <= y < H:
                    img[y, x + i] = ((v >> 11) * 255 // 31, ((v >> 5) & 63) * 255 // 63, (v & 31) * 255 // 31)   # (as the Core2 reads it)
                    have[y, x + i] = True
    return sways, body, plates
