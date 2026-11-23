"""Depth planes (parallax) for Mantis Gallery: data, .jig lines, depth-map import, and a live preview.

The Core2 (src/jiggle_parallax.h) is the reference renderer. This preview follows it — who owns each
pixel, what each plane continues behind what it hides, the mirrored fill past a plane's edge, the
gyro-style view with its spring and drift — but samples with bilinear slices so the lab can animate it
live; the Core2 uses a sharper (Catmull-Rom) filter for the final look.

.jig lines (after zones and water, ignored by older firmware):
    parallax amount=10 ambient=0.3 ...        (only values that differ from the defaults)
    plane depth=0.7 [ramp=ax,ay,bx,by,depth_b]
    prow y a-b c-d ...                        (that plane's mask, like wrow)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

W, H = 320, 240
PLANE_MAX = 4
#          key        label                                        lo    hi    default  help
PPARAMS = [("amount", "Depth (px the nearest ground slides)", 0.0, 24.0, 6.0,
            "How far the nearest ground slides at a full tilt. As subtle as the scene needs for the illusion to hold."),
           ("ambient", "Drift while still", 0.0, 1.0, 0.3,
            "A slow wandering of the view when the Core2 sits still, so the depth keeps showing. 0 = off."),
           ("response", "Response (spring)", 0.2, 3.0, 1.0,
            "How quickly the view follows a tilt. Lower feels heavier and floatier."),
           ("focus", "Pinned depth (0 = background)", 0.0, 1.0, 0.0,
            "The depth that stays put. 0 keeps the background still (cheapest, no edges uncovered)."),
           ("tilt", "Tilt for the full slide (deg)", 3.0, 40.0, 14.0,
            "How far you tip the Core2 for the full slide."),
           ("vert", "Up/down share", 0.0, 1.5, 1.0,
            "How much tipping forward / back moves things compared with left / right."),
           ("invert", "Reverse direction", 0.0, 1.0, 0.0,
            "Flip which way the planes slide."),
           ("ground", "Ground slides with the tilt", 0.0, 1.0, 1.0,
            "With a ground grid: the ground itself slides (near rows most). Off: it stays put and only the figures move.")]
PKEYS = [p[0] for p in PPARAMS]
PDEF = {p[0]: p[4] for p in PPARAMS}


@dataclass
class Plane:
    depth: float = 0.5                      # 0 = background depth, 1 = nearest
    mask: object = None                     # (H, W) bool
    ramp: tuple | None = None               # (ax, ay, bx, by, depth_b): depth runs from `depth` at A to depth_b at B
    feet: tuple | None = None               # (x, y): where it meets the ground (the feet line); else its lowest row
    name: str = ""                          # the Studio's layer name
    fixed: bool = False                     # depth set by hand (not from where it meets the ground)

    def copy(self):
        return Plane(self.depth, None if self.mask is None else self.mask.copy(), self.ramp, self.feet, self.name, self.fixed)

    def depth_at(self, X, Y):
        if not self.ramp:
            return np.full(np.broadcast(X, Y).shape, self.depth, np.float32)
        ax, ay, bx, by, db = self.ramp
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        if l2 <= 1:
            return np.full(np.broadcast(X, Y).shape, self.depth, np.float32)
        t = ((X - ax) * dx + (Y - ay) * dy) / l2
        return (self.depth + (db - self.depth) * np.clip(t, 0, 1)).astype(np.float32)

    def dhi(self):
        return max(self.depth, self.ramp[4]) if self.ramp else self.depth

    def drange(self):
        return (min(self.depth, self.ramp[4]), max(self.depth, self.ramp[4])) if self.ramp else (self.depth, self.depth)


@dataclass
class Parallax:
    planes: list = field(default_factory=list)
    prm: dict = field(default_factory=dict)
    grid: list | None = None                # ground grid corners [(x, y)×4]: near-L, near-R, far-R, far-L

    def get(self, k):
        lo, hi = next((p[2], p[3]) for p in PPARAMS if p[0] == k)
        return min(hi, max(lo, float(self.prm[k]))) if k in self.prm else PDEF[k]

    def copy(self):
        return Parallax([p.copy() for p in self.planes], dict(self.prm), None if self.grid is None else list(self.grid))

    def any(self):
        return bool(self.grid) or any(p.mask is not None and p.mask.any() for p in self.planes)

    def ground_moves(self):
        return bool(self.grid) and self.get("ground") >= 0.5

    def ground(self):
        """(c0, cx, cy): the ground's depth = clamp(c0 + cx·x + cy·y, 0, 1) — 1 at the nearest ground the
        frame shows, 0 on the horizon (inverse depth of a flat ground is linear on screen). None if no grid."""
        if not self.grid:
            return None
        import jiggle_lab as J
        a, b, c, d, e, f, g, h = J.water_homog(self.grid)
        G, Hh, I = d * h - e * g, -(a * h - b * g), a * e - b * d
        sgn = -1.0 if (G * self.grid[0][0] + Hh * self.grid[0][1] + I) < 0 else 1.0
        wmax = max(sgn * (G * x + Hh * y + I) for x, y in ((0, 0), (W, 0), (0, H), (W, H)))
        if wmax <= 1e-9:
            return None
        return sgn * I / wmax, sgn * G / wmax, sgn * Hh / wmax

    def card_depths(self):
        """With a grid: each plane stands where its bottom meets the ground."""
        gr = self.ground()
        if gr is None:
            return
        c0, cx, cy = gr
        for p in self.planes:
            if p.mask is None or not p.mask.any() or p.fixed:
                continue
            if p.feet is not None:
                xb, yb = p.feet
            else:
                ys, xs = np.nonzero(p.mask)
                yb = int(ys.max()); xb = xs[ys == yb].mean()
            p.depth = float(min(1.0, max(0.0, c0 + cx * xb + cy * yb)))
            p.ramp = None


# ---- .jig ---------------------------------------------------------------------------------------------
def parse_lines(lines):
    """Parallax from a .jig's lines (anything else is skipped)."""
    P = Parallax()
    cur = None
    for line in lines:
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "parallax":
            for t in tok[1:]:
                k, _, v = t.partition("=")
                if k in PKEYS:
                    try:
                        P.prm[k] = float(v)
                    except ValueError:
                        pass
        elif tok[0] == "plane":
            cur = None
            if len(P.planes) >= PLANE_MAX:
                continue
            p = Plane(mask=np.zeros((H, W), bool))
            for t in tok[1:]:
                k, _, v = t.partition("=")
                try:
                    if k == "depth":
                        p.depth = min(1.0, max(0.0, float(v)))
                    elif k == "ramp":
                        a = v.split(",")
                        p.ramp = (int(a[0]), int(a[1]), int(a[2]), int(a[3]), min(1.0, max(0.0, float(a[4]))))
                    elif k == "feet":
                        a = v.split(",")
                        p.feet = (int(a[0]), int(a[1]))
                    elif k == "name":
                        p.name = v.replace("_", " ")[:15]
                    elif k == "fixed":
                        p.fixed = v not in ("0", "")
                except (ValueError, IndexError):
                    pass
            P.planes.append(p)
            cur = p
        elif tok[0] == "pgrid":
            try:
                pts = [tuple(float(v) for v in t.split(",")) for t in tok[1:5]]
                if len(pts) == 4 and all(len(p_) == 2 for p_ in pts):
                    P.grid = pts
            except ValueError:
                pass
        elif tok[0] == "prow" and cur is not None and len(tok) > 1:
            try:
                y = int(tok[1])
            except ValueError:
                continue
            if 0 <= y < H:
                for sp in tok[2:]:
                    a, _, b = sp.partition("-")
                    try:
                        cur.mask[y, max(0, int(a)):max(0, min(W, int(b)))] = True
                    except ValueError:
                        pass
    return P


def format_lines(P):
    if P is None or not P.any():
        return []
    out = []
    line = "parallax"
    for k in PKEYS:
        if k in P.prm and abs(P.get(k) - PDEF[k]) > 1e-6:
            line += f" {k}={P.get(k):.3g}"
    out.append(line)
    if P.grid:
        out.append("pgrid " + " ".join(f"{x:.1f},{y:.1f}" for x, y in P.grid))
    for p in P.planes[:PLANE_MAX]:
        if p.mask is None or not p.mask.any():
            continue
        line = f"plane depth={p.depth:.3g}"
        if p.ramp:
            ax, ay, bx, by, db = p.ramp
            line += f" ramp={int(ax)},{int(ay)},{int(bx)},{int(by)},{db:.3g}"
        if p.feet is not None:
            line += f" feet={int(p.feet[0])},{int(p.feet[1])}"
        if p.fixed:
            line += " fixed=1"
        if p.name:
            line += " name=" + p.name.replace(" ", "_")[:15]
        out.append(line)
        for y in range(H):
            row = p.mask[y]
            if not row.any():
                continue
            d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
            s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            out.append(f"prow {y} " + " ".join(f"{a}-{b}" for a, b in zip(s, e)))
    return out


# ---- depth map import -----------------------------------------------------------------------------------
def planes_from_depth(depth, n_planes=3, near_is_bright=True, content_box=None):
    """A depth map (any size, grayscale array; e.g. from Depth Anything / MiDaS) → planes.
    The farthest band stays background; the rest become planes at their band's depth. `content_box`
    (x0, y0, w, h) is where the picture sits on the 320×240 screen (smaller pictures are centred)."""
    from PIL import Image
    d = np.asarray(depth, np.float32)
    if d.ndim == 3:
        d = d[..., :3].mean(-1)
    x0, y0, cw, ch = content_box or (0, 0, W, H)
    d = np.asarray(Image.fromarray(d).resize((cw, ch), Image.BILINEAR), np.float32)
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    d = np.clip((d - lo) / max(1e-6, hi - lo), 0, 1)
    if not near_is_bright:
        d = 1 - d
    full = np.zeros((H, W), np.float32)
    full[y0:y0 + ch, x0:x0 + cw] = d
    inside = np.zeros((H, W), bool)
    inside[y0:y0 + ch, x0:x0 + cw] = True
    n = max(2, min(PLANE_MAX + 1, n_planes + 1))
    cuts = np.quantile(full[inside], np.linspace(0, 1, n + 1)[1:-1])     # equal-area bands
    band = np.digitize(full, cuts)
    planes = []
    for b in range(1, n):                                                   # band 0 = background
        m = (band == b) & inside
        if m.sum() < 30:
            continue
        planes.append(Plane(depth=float(np.clip(full[m].mean(), 0.05, 1.0)), mask=m))
    return planes[-PLANE_MAX:]


# ---- preview --------------------------------------------------------------------------------------------
def _nearest(seed, iters):
    """Nearest seed pixel coordinates (8-neighbour propagation, `iters` px deep). −1 where none."""
    ny = np.where(seed, np.arange(H)[:, None], -1).astype(np.int32)
    nx = np.where(seed, np.arange(W)[None, :], -1).astype(np.int32)
    Y, X = np.mgrid[:H, :W]
    big = 1 << 30
    d = np.where(seed, 0, big)
    for _ in range(iters):
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            sy = np.roll(ny, (dy, dx), (0, 1)); sx = np.roll(nx, (dy, dx), (0, 1))
            ok = sy >= 0
            if dy == 1: ok[0, :] = False
            if dy == -1: ok[-1, :] = False
            if dx == 1: ok[:, 0] = False
            if dx == -1: ok[:, -1] = False
            cd = np.where(ok, (sy - Y) ** 2 + (sx - X) ** 2, big)
            better = cd < d
            ny = np.where(better, sy, ny); nx = np.where(better, sx, nx); d = np.where(better, cd, d)
    return ny, nx, d


def soften(cov):
    """cov as alpha: the outer pixel half on an edge row, most on a corner-only touch (the frame isn't an edge)"""
    c = np.pad(cov, 1, mode="edge")
    edge = ~(c[:-2, 1:-1] & c[2:, 1:-1] & c[1:-1, :-2] & c[1:-1, 2:])
    corner = ~(c[:-2, :-2] & c[:-2, 2:] & c[2:, :-2] & c[2:, 2:])
    return np.where(cov, np.where(edge, 128 / 255, np.where(corner, 200 / 255, 1.0)), 0.0).astype(np.float32)


class DepthPreview:
    """Prepared once per edit; render(content, view) per frame."""

    def __init__(self, P: Parallax, photo):
        self.P = P
        self.ok = P is not None and P.any()
        if not self.ok:
            return
        self.photo = np.asarray(photo, np.float32)
        Y, X = np.mgrid[:H, :W].astype(np.float32)
        P.card_depths()
        self.gr = P.ground() if P.ground_moves() else None
        self.planes = [p for p in P.planes if p.mask is not None and p.mask.any()]
        self.ok = True
        dep = np.full((H, W), -1.0, np.float32)
        owner = np.zeros((H, W), np.int32)
        for k, p in enumerate(self.planes):
            dk = p.depth_at(X, Y)
            take = p.mask & (dk > dep)
            owner[take] = k + 1; dep[take] = dk[take]
        self.owner = owner
        f, A, vert, amb = P.get("focus"), P.get("amount"), P.get("vert"), P.get("ambient")
        # the view: a full tilt (1, through the soft limit) plus the idle drift, at most ambient·(0.55 + 0.22)
        # across and ambient·(0.40 + 0.15) up / down (step's two sines each way)
        vx, vy = 1.0 + (0.55 + 0.22) * amb, 1.0 + (0.40 + 0.15) * amb
        # how far two layers slide apart (a nearer one uncovering what's behind it): their depths differ by at
        # most the nearest depth there is (the background is 0)
        span = max([p.drange()[1] for p in self.planes] + ([1.0] if self.gr else [0.0]))
        self.margin = int(math.ceil(max(span * A * vx, span * A * vert * vy))) + 3

        deep = self.margin + 4
        # fills: each layer's colours continued past its edge (mirrored across the nearest own pixel)
        self.fill = []
        for lid in range(len(self.planes) + 1):
            own = owner == lid
            ny, nx, _ = _nearest(own, deep if lid else 40)
            my, mx = np.clip(2 * ny - Y.astype(np.int32), 0, H - 1), np.clip(2 * nx - X.astype(np.int32), 0, W - 1)
            use_m = (ny >= 0) & (owner[my, mx] == lid)
            fy = np.where(use_m, my, np.where(ny >= 0, ny, Y.astype(np.int32)))
            fx = np.where(use_m, mx, np.where(nx >= 0, nx, X.astype(np.int32)))
            fl = self.photo[fy, fx]
            fl[own] = self.photo[own]
            self.fill.append(fl)
        # coverage: own pixels + what each plane continues behind nearer ones (nearest farther surface)
        cov = [owner == k + 1 for k in range(len(self.planes))]
        dhi = [p.dhi() for p in self.planes]
        for o, po in enumerate(self.planes):
            farther = (owner == 0) | ((owner != o + 1) & (np.array([-1.0] + dhi)[owner] < dhi[o]))
            ny, nx, d = _nearest(farther, deep)
            hid = (owner == o + 1) & (ny >= 0) & (d <= self.margin * self.margin)
            behind = owner[np.clip(ny, 0, H - 1), np.clip(nx, 0, W - 1)]
            for k in range(len(self.planes)):
                cov[k] |= hid & (behind == k + 1)
        for k, p in enumerate(self.planes):
            nearer = (owner > 0) & (owner != k + 1) & (np.array([-1.0] + dhi)[owner] >= dhi[k])
            cov[k] |= p.mask & nearer
        self.cov = [c.astype(np.float32) for c in cov]
        self.soft = [soften(c) for c in cov]
        self.order = sorted(range(len(self.planes)), key=lambda k: -dhi[k])
        self.X, self.Y = X, Y
        # no zoom: the photo's own pixels at rest; past the photo's edge a slide reads it mirrored (_mir)
        self.zoom = 1.0
        self.ZX = (X - W * 0.5) / self.zoom + W * 0.5
        self.ZY = (Y - H * 0.5) / self.zoom + H * 0.5
        self.gd = np.clip(self.gr[0] + self.gr[1] * X + self.gr[2] * Y, 0, 1) if self.gr else None

    @staticmethod
    def _mir(i, n):
        """source indices past the photo's edge read it mirrored about that edge (… 1, 0 | 0, 1 …), as the Core2's
        par_mirror: what a slide uncovers there lines up with the photo"""
        i = np.where(i < 0, -1 - i, np.where(i >= n, 2 * n - 1 - i, i))
        return np.clip(i, 0, n - 1)

    @classmethod
    def _shift(cls, img, sx, sy):
        """img sampled at (x − sx, y − sy), bilinear, mirrored past the edges. sx, sy scalars."""
        ix, iy = math.floor(sx), math.floor(sy)
        fx, fy = sx - ix, sy - iy

        def sl(a, ox, oy):              # a[y − oy, x − ox], mirrored past the edges
            ys = cls._mir(np.arange(H) - oy, H); xs = cls._mir(np.arange(W) - ox, W)
            return a[ys][:, xs]
        a00 = sl(img, ix, iy); a10 = sl(img, ix + 1, iy); a01 = sl(img, ix, iy + 1); a11 = sl(img, ix + 1, iy + 1)
        # sampling at x − sx = (x − ix) − fx: between x − ix − 1 (weight fx) and x − ix (1 − fx)
        return (a00 * (1 - fx) + a10 * fx) * (1 - fy) + (a01 * (1 - fx) + a11 * fx) * fy

    @staticmethod
    def _taps(t, q):
        """The Core2's colour filter at phase t (0..1): 4 taps, bilinear eased toward Catmull-Rom by q (0..1)."""
        t2, t3 = t * t, t * t * t
        cr = ((-t3 + 2 * t2 - t) * 0.5, (3 * t3 - 5 * t2 + 2) * 0.5, (-3 * t3 + 4 * t2 + t) * 0.5, (t3 - t2) * 0.5)
        bl = (0.0, 1.0 - t, t, 0.0)
        return [b + (c - b) * q for b, c in zip(bl, cr)]

    @classmethod
    def _shift4(cls, img, sx, sy, q):
        """img (H, W, 3) sampled at (x − sx, y − sy) with the colour filter (_taps), separable, mirrored past the edges."""
        if q <= 0:
            return cls._shift(img, sx, sy)
        # sample position x − sx = (x − ix − 1) + (1 − fx): taps at x − ix − 2 … x − ix + 1, phase 1 − fx
        ix, iy = math.floor(sx), math.floor(sy)
        tx, ty = 1.0 - (sx - ix), 1.0 - (sy - iy)
        if tx >= 1.0: ix, tx = ix - 1, 0.0               # (a whole shift: taps from x − ix − 1, phase 0)
        if ty >= 1.0: iy, ty = iy - 1, 0.0
        wx, wy = cls._taps(tx, q), cls._taps(ty, q)
        xs = np.arange(W); ys = np.arange(H)
        h = sum(w * img[:, cls._mir(xs - ix - 2 + k, W)] for k, w in enumerate(wx))
        out = sum(w * h[cls._mir(ys - iy - 2 + k, H)] for k, w in enumerate(wy))
        return np.clip(out, 0, 255)

    @classmethod
    def _sample4(cls, a, qx, qy, q):
        """a (H, W, 3) sampled at per-pixel (qx, qy) with the colour filter (_taps), mirrored past the edges."""
        ix, iy = np.floor(qx).astype(np.int32), np.floor(qy).astype(np.int32)
        tx, ty = qx - ix, qy - iy
        if q <= 0:
            fx, fy = tx[..., None], ty[..., None]
            x0, x1, y0, y1 = cls._mir(ix, W), cls._mir(ix + 1, W), cls._mir(iy, H), cls._mir(iy + 1, H)
            return (a[y0, x0] * (1 - fx) + a[y0, x1] * fx) * (1 - fy) + (a[y1, x0] * (1 - fx) + a[y1, x1] * fx) * fy
        wx, wy = cls._taps(tx, q), cls._taps(ty, q)
        out = 0.0
        for m in range(4):
            yy = cls._mir(iy - 1 + m, H)
            row = sum(wx[k][..., None] * a[yy, cls._mir(ix - 1 + k, W)] for k in range(4))
            out = out + wy[m][..., None] * row
        return np.clip(out, 0, 255)

    def _alpha(self, k, kx, ky):
        """the plane's silhouette, its edge feather showing as far as it has slid (as the Core2)"""
        fade = min(1.0, self.planes[k].dhi() * max(abs(kx), abs(ky)))
        s = self.soft[k]
        return np.where(s > 0, 1.0 - (1.0 - s) * fade, 0.0).astype(np.float32)

    def render(self, content, view, sharp=1.0):
        """content: (H, W, 3) uint8 (the photo with water / zones drawn in). view: (vx, vy) in −1..1.
        sharp: the colour filter as the Core2 eases it (GyroView.sharp): 1 = Catmull-Rom, 0 = bilinear."""
        if not self.ok:
            return content
        P = self.P
        s = -1.0 if P.get("invert") >= 0.5 else 1.0
        kx, ky = s * P.get("amount") * view[0], s * P.get("amount") * view[1] * P.get("vert")
        f = P.get("focus")
        cont = np.asarray(content, np.float32)
        out = np.zeros((H, W, 3), np.float32)
        rem = np.ones((H, W), np.float32)
        layers = [k + 1 for k in self.order] + [0]
        if self.gr is not None:                      # ground grid: the ground per pixel
            for lid in layers:
                src = np.where((self.owner == lid)[..., None], cont, self.fill[lid])
                d = (self.gd if lid == 0 else np.float32(self.planes[lid - 1].depth)) - f
                qx = self.ZX - d * kx; qy = self.ZY - d * ky
                ix, iy = np.floor(qx).astype(np.int32), np.floor(qy).astype(np.int32); fx, fy = (qx - ix)[..., None], (qy - iy)[..., None]
                x0, x1, y0, y1 = self._mir(ix, W), self._mir(ix + 1, W), self._mir(iy, H), self._mir(iy + 1, H)
                def g(a):
                    a = a if a.ndim == 3 else a[..., None]
                    return (a[y0, x0] * (1 - fx) + a[y0, x1] * fx) * (1 - fy) + (a[y1, x0] * (1 - fx) + a[y1, x1] * fx) * fy
                c = self._sample4(src, qx, qy, sharp)
                a = np.ones((H, W), np.float32) if lid == 0 else g(self._alpha(lid - 1, kx, ky))[..., 0]
                wgt = a * rem
                out += c * wgt[..., None]
                rem -= wgt
            return np.clip(out + 0.5, 0, 255).astype(np.uint8)
        for lid in layers:
            src = np.where((self.owner == lid)[..., None], cont, self.fill[lid])
            if lid:
                p = self.planes[lid - 1]
                if p.ramp:                        # per pixel: shift by the depth here
                    d = p.depth_at(self.X, self.Y) - f
                    qx = self.X - d * kx; qy = self.Y - d * ky
                    ix, iy = np.floor(qx).astype(np.int32), np.floor(qy).astype(np.int32); fx, fy = qx - ix, qy - iy
                    x0, x1, y0, y1 = self._mir(ix, W), self._mir(ix + 1, W), self._mir(iy, H), self._mir(iy + 1, H)
                    def g(a):
                        a = a if a.ndim == 3 else a[..., None]
                        t = a[y0, x0] * (1 - fx[..., None]) + a[y0, x1] * fx[..., None]
                        b = a[y1, x0] * (1 - fx[..., None]) + a[y1, x1] * fx[..., None]
                        return t * (1 - fy[..., None]) + b * fy[..., None]
                    c, a = self._sample4(src, qx, qy, sharp), g(self._alpha(lid - 1, kx, ky))[..., 0]
                else:
                    sx, sy = (p.depth - f) * kx, (p.depth - f) * ky
                    c, a = self._shift4(src, sx, sy, sharp), self._shift(self._alpha(lid - 1, kx, ky), sx, sy)
            else:
                c, a = (self._shift4(src, -f * kx, -f * ky, sharp) if f > 0 else src), np.ones((H, W), np.float32)
            wgt = a * rem
            out += c * wgt[..., None]
            rem -= wgt
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)


class View:
    """The view (−1..1 each way) as the Core2 moves it: a target (from tilting, or the lab's pad),
    a slow drift while still, and a critically damped spring so the depth has weight."""

    def __init__(self, P):
        self.P, self.t = P, 0.0
        self.v = [0.0, 0.0]; self.vv = [0.0, 0.0]

    def step(self, dt, target=(0.0, 0.0)):
        P = self.P
        self.t += dt
        tg = list(target)
        amb = P.get("ambient") if P else 0.0
        if amb > 0:
            T = 2 * math.pi * self.t
            tg[0] += amb * (0.55 * math.sin(T / 11.0) + 0.22 * math.sin(T / 4.3 + 1.0))
            tg[1] += amb * (0.40 * math.sin(T / 8.3 + 2.0) + 0.15 * math.sin(T / 3.7 + 0.5))
        w = 9.0 * (P.get("response") if P else 1.0)
        n = max(1, int(math.ceil(dt * w / 0.25)))
        h = dt / n
        for _ in range(n):
            for a in range(2):
                self.vv[a] += (w * w * (tg[a] - self.v[a]) - 2 * w * self.vv[a]) * h
                self.v[a] += self.vv[a] * h
        return tuple(self.v)


class GyroView:
    """The Core2's view exactly (jiggle_parallax.h parallax_step): gyro rates (deg/s) are integrated into a
    held tilt that eases back to centre over a few seconds, through a soft limit, plus the slow drift,
    then a critically damped spring. rates: (about the screen's horizontal axis: top edge away = +,
    about its vertical axis: right edge away = +)."""
    RECENTER_S, DEAD_DPS = 2.5, 0.6
    SOFT_PXS, SHARP_PXS, SOFT_S, SHARP_S = 10.0, 4.0, 0.1, 0.35   # the filter: soft while it slides fast (as the Core2)

    def __init__(self, P):
        self.P, self.t = P, 0.0
        self.th = [0.0, 0.0]; self.v = [0.0, 0.0]; self.vv = [0.0, 0.0]
        self.sharp = 1.0                                  # 1 = Catmull-Rom, 0 = bilinear (DepthPreview.render)

    def step(self, dt, rates=(0.0, 0.0)):
        P = self.P
        self.t += dt
        dead = lambda r: 0.0 if abs(r) < self.DEAD_DPS else (r - self.DEAD_DPS if r > 0 else r + self.DEAD_DPS)
        rx, ry = rates
        self.th[0] += dead(ry) * math.radians(1) * dt          # right edge away → near planes slide right
        self.th[1] += -dead(rx) * math.radians(1) * dt         # top edge away → near planes slide up
        leak = math.exp(-dt / self.RECENTER_S)
        self.th = [a * leak for a in self.th]
        full = math.radians(P.get("tilt") if P else 14.0)
        tg = []
        for a in range(2):
            u = 1.5 * self.th[a] / full
            tg.append(u / math.sqrt(1 + u * u))
        amb = P.get("ambient") if P else 0.0
        if amb > 0:
            T = 2 * math.pi * self.t
            tg[0] += amb * (0.55 * math.sin(T / 11.0) + 0.22 * math.sin(T / 4.3 + 1.0))
            tg[1] += amb * (0.40 * math.sin(T / 8.3 + 2.0) + 0.15 * math.sin(T / 3.7 + 0.5))
        w = 9.0 * (P.get("response") if P else 1.0)
        n = max(1, int(math.ceil(dt * w / 0.25)))
        h = dt / n
        for _ in range(n):
            for a in range(2):
                self.vv[a] += (w * w * (tg[a] - self.v[a]) - 2 * w * self.vv[a]) * h
                self.v[a] += self.vv[a] * h
        speed = (abs(self.vv[0]) + abs(self.vv[1])) * (P.get("amount") if P else 0.0)   # px/s
        if speed > self.SOFT_PXS:
            self.sharp = max(0.0, self.sharp - dt / self.SOFT_S)
        elif speed < self.SHARP_PXS:
            self.sharp = min(1.0, self.sharp + dt / self.SHARP_S)
        return tuple(self.v)


def tilt_to_view(P, tilt_deg_lr, tilt_deg_ud):
    """A held tilt (degrees, right edge away / top edge away) → view target, like the Core2's soft limit."""
    full = P.get("tilt") if P else 14.0
    out = []
    for th in (tilt_deg_lr, -tilt_deg_ud):
        u = 1.5 * th / full
        out.append(u / math.sqrt(1 + u * u))
    return tuple(out)
