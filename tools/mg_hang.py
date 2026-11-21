"""Hanging things for Mantis Gallery: hair, cloth (skirts, scarves, flags) and charms (earrings, pendants).

How the 2D animation tools do this (Live2D's physics groups, Spine's physics constraints): a chain of
pendulums driven by how the "camera" moves, each link pulled back toward its pose in the artwork
(strength) and slowed (damping), with the picture bound to the chain as a mesh. Here:

  * The user selects the thing (same tools as water / depth) and marks where it hangs from: one point
    (an earring's hook) or a line (a ponytail's tie, a skirt's waistband).
  * A mesh of chains runs from that line down through the selection: `nc` chains side by side, rows of
    nodes along them. Rows at / above the line are pinned.
  * Physics at a fixed 120 Hz (Verlet): each free node is pulled toward where its link would be if it
    kept its pose in the photo relative to the link above it (so a curve stays a curve while it
    swings), pushed by the change in gravity as the Core2 tilts, by its jolts (the hair lags), by a
    breeze, and by taps; links keep their lengths (and side links their spacing / shear).
  * Drawing: what's behind the selection is filled from its surroundings once; every frame the mesh's
    triangles are drawn from their pose in the photo to where they are now, with a soft edge only
    while it moves (so at rest the photo is exact).

The Core2 (src/jiggle_hang.h) runs the same steps in the same order.

.jig lines (after zones, water and depth; ignored by older firmware):
    hang hair 150,60 162,60 swing=1 breeze=0.3      (kind, where it hangs from: A then B, A = B for a point)
    hrow y a-b c-d ...                                (its selection, like wrow)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

W, H = 320, 240
HANG_MAX = 3
KINDS = ("hair", "cloth", "charm")
KIND_LABEL = {"hair": "Hair", "cloth": "Cloth", "charm": "Charm"}
# gain: how fully it follows the new "down" (1 = a free pendulum); g: gravity (px/s²) sets how quickly
# (a 60 px ponytail at 1500 swings about once in 1.25 s); c: damping (1/s); j: lag on a jolt (px/s² per g)
PRESET = {"hair":  dict(nc=3, nr=6, gain=0.45, c=2.2, g=1500.0, j=900.0, shear=0.5, wind=110.0),
          "cloth": dict(nc=5, nr=4, gain=0.25, c=3.0, g=1100.0, j=600.0, shear=0.2, wind=70.0),
          "charm": dict(nc=2, nr=1, gain=0.8, c=1.0, g=1800.0, j=1100.0, shear=1.0, wind=20.0)}
#          key      label                       lo   hi   default
HPARAMS = [("swing", "Swing (how much it moves)", 0.0, 2.5, 1.0),
           ("breeze", "Breeze while still", 0.0, 1.5, 0.3)]
HDEF = {p[0]: p[4] for p in HPARAMS}
DT = 1.0 / 120.0
ITERS = 4
REACH_DEG = 55.0              # the most it's drawn swinging (its drawing region)


@dataclass
class Hang:
    kind: str = "hair"
    a: tuple = (0.0, 0.0)          # where it hangs from: A ... B (A = B: a point)
    b: tuple = (0.0, 0.0)
    mask: object = None            # (H, W) bool
    prm: dict = field(default_factory=dict)

    def get(self, k):
        return self.prm.get(k, HDEF[k])

    def copy(self):
        return Hang(self.kind, tuple(self.a), tuple(self.b), None if self.mask is None else self.mask.copy(), dict(self.prm))

    def ok(self):
        return self.mask is not None and self.mask.sum() >= 12


def grow1(m):
    q = np.pad(m, 1)
    return m | q[:-2, 1:-1] | q[2:, 1:-1] | q[1:-1, :-2] | q[1:-1, 2:]


def soften(m):
    """What it carries, as alpha: the selection, and three rings past it (the wisps and blended edge a selection
    never quite gets) fading out. At rest it's all drawn solid (the photo exactly); the fade shows only as
    it moves - and what it leaves behind is filled out to the third ring, so no outline is left behind."""
    r1 = grow1(m); r2 = grow1(r1); r3 = grow1(r2)
    c = np.pad(m, 1, mode="edge")
    edge = ~(c[:-2, 1:-1] & c[2:, 1:-1] & c[1:-1, :-2] & c[1:-1, 2:])
    return np.where(m, np.where(edge, 210, 255), np.where(r1, 120, np.where(r2, 45, np.where(r3, 20, 0)))).astype(np.int32)


FILL_ITERS = 80
FILL_DETAIL = 0.6
FILL_GRAIN = 9.0               # (8-bit levels)


def behind_fill(photo, hole):
    """What's behind: the surroundings' colours smoothed across the hole (a membrane: nearest colour, then
    averaged with its neighbours), with the fine texture of the surroundings mirrored in on top - so the
    sky stays sky, a shoulder carries on as skin, sand keeps its grain, without streaks or a blur."""
    from mg_depth import _nearest
    ny, nx, _ = _nearest(~hole, 64)
    Y, X = np.mgrid[:H, :W]
    ok = ny >= 0
    my, mx = np.clip(2 * ny - Y, 0, H - 1), np.clip(2 * nx - X, 0, W - 1)
    use_m = ok & ~hole[my, mx]
    fy = np.where(use_m, my, np.where(ok, ny, Y)); fx = np.where(use_m, mx, np.where(ok, nx, X))
    mir = photo.copy(); mir[hole] = photo[fy[hole], fx[hole]]
    near = photo.copy(); near[hole & ok] = photo[ny[hole & ok], nx[hole & ok]]
    g = near.astype(np.float32)
    for _ in range(FILL_ITERS):
        q = np.pad(g, ((1, 1), (1, 1), (0, 0)), mode="edge")
        g[hole] = ((q[:-2, 1:-1] + q[2:, 1:-1] + q[1:-1, :-2] + q[1:-1, 2:]) / 4)[hole]
    q = np.pad(mir.astype(np.float32), ((2, 2), (2, 2), (0, 0)), mode="edge")
    blur = sum(q[2 + dy:H + 2 + dy, 2 + dx:W + 2 + dx] for dy in range(-2, 3) for dx in range(-2, 3)) / 25.0
    out = np.asarray(photo, np.float32).copy()
    det = np.clip(FILL_DETAIL * (mir - blur), -FILL_GRAIN, FILL_GRAIN)   # grain, not stray dark / bright bits
    out[hole] = np.clip(g[hole] + det[hole], 0, 255)
    return out


def decontaminate(photo, mask, fill):
    """Its edge, matted: the pixels along its outline (and the three rings past it) are part it, part what
    was behind it in the photo. Each is split, c = a·H + (1 − a)·B (H: its nearest solid pixel, B: what's
    behind), so when it moves it carries its own colour at that cover - not a seam of the old background.
    Returns (alpha 1..255 over the selection and rings, 0 elsewhere; colours to carry)."""
    from mg_depth import _nearest
    soft = soften(mask)
    edge = (soft > 0) & (soft < 255)
    solid = soft == 255
    if not solid.any():
        solid = mask
    ny, nx, _ = _nearest(solid, 6)
    ok = edge & (ny >= 0)
    Hc = photo[np.clip(ny, 0, H - 1), np.clip(nx, 0, W - 1)].astype(np.float32)
    c, B = photo.astype(np.float32), fill.astype(np.float32)
    d = Hc - B
    dd = (d * d).sum(-1)
    a = ((c - B) * d).sum(-1) / np.maximum(dd, 100.0)
    lo = np.where(soft == 210, 0.4, 0.0); hi = np.where(soft == 210, 1.0, np.where(soft == 120, 0.8, np.where(soft == 45, 0.5, 0.3)))
    a = np.where(dd < 300, soft / 255.0, np.clip(a, lo, hi))           # (too alike to tell: the plain fade)
    alpha = np.where(solid & mask, 255, np.where(edge, np.clip(np.round(a * 255), 1, 254), 0)).astype(np.int32)
    texd = photo.astype(np.float32).copy()
    texd[ok] = Hc[ok]
    return alpha, texd


def hang_dir(xs, ys, ax, ay, bx, by):
    """Which way it hangs: from the pin toward the middle of what's below it (the side of the pin line that
    faces down in the photo; a point pin: below its height). Not below at all: straight down."""
    rx, ry = (ax + bx) / 2, (ay + by) / 2
    nx, ny = -(by - ay), bx - ax
    l = math.hypot(nx, ny)
    if l < 1:
        nx, ny = 0.0, 1.0
    else:
        nx, ny = nx / l, ny / l
        if ny < 0:
            nx, ny = -nx, -ny
    below = (xs - rx) * nx + (ys - ry) * ny > 0
    if below.sum() < 4:
        return nx, ny
    return float(xs[below].mean()) - rx, float(ys[below].mean()) - ry


class HangSim:
    """Prepared once per edit; step(...) per frame; render(...) into the content."""

    def __init__(self, h: Hang, photo):
        self.h = h
        self.ok = h is not None and h.ok()
        if not self.ok:
            return
        P = PRESET[h.kind]
        self.P = P
        ys, xs = np.nonzero(h.mask)
        ax, ay = map(float, h.a); bx, by = map(float, h.b)
        rx, ry = (ax + bx) / 2, (ay + by) / 2
        dx, dy = hang_dir(xs, ys, ax, ay, bx, by)                       # it hangs toward its middle below the pin
        l = math.hypot(dx, dy)
        if l < 4:
            dx, dy, l = 0.0, 1.0, 1.0
        e2 = (dx / l, dy / l)
        e1 = (-e2[1], e2[0])
        if (bx - ax) * e1[0] + (by - ay) * e1[1] < 0:
            e1 = (-e1[0], -e1[1])
        s = (xs - rx) * e1[0] + (ys - ry) * e1[1]
        t = (xs - rx) * e2[0] + (ys - ry) * e2[1]
        smin, smax, tmin, tmax = float(s.min()) - 1.5, float(s.max()) + 1.5, float(t.min()), float(t.max()) + 1.5
        tmax = max(tmax, 4.0)
        nc, nr = P["nc"], P["nr"]
        rows = ([tmin - 1.5] if tmin < -0.5 else []) + [0.0] + [tmax * j / nr for j in range(1, nr + 1)]
        self.npin = len(rows) - nr
        cols = [smin + (smax - smin) * k / (nc - 1) for k in range(nc)]
        self.nc, self.nrows = nc, len(rows)
        self.e1, self.e2, self.R = e1, e2, (rx, ry)
        self.rest = np.array([[(rx + c * e1[0] + r * e2[0], ry + c * e1[1] + r * e2[1]) for c in cols] for r in rows],
                             np.float64)                                   # [row][col] = (x, y)
        self.x = self.rest.copy(); self.xp = self.rest.copy()
        self.tlen = tmax
        # Hanging things are pendulums: pushed off "down" by d, gravity pulls back g·d / (distance to the
        # pivot) - so each node's pull back is g / its distance (a tilt by a swings it by a, to first order).
        self.kn = np.array([P["g"] / max(4.0, r) for r in rows], np.float64)
        self.k = P["g"] / max(4.0, math.hypot(float(xs.mean()) - rx, float(ys.mean()) - ry))   # (charm)
        # what it's drawn over: everything it can swing across
        pts = [self.rest.reshape(-1, 2)]
        for deg in (-REACH_DEG, REACH_DEG):
            c, sn = math.cos(math.radians(deg)), math.sin(math.radians(deg))
            d = self.rest.reshape(-1, 2) - (rx, ry)
            pts.append(np.stack([rx + d[:, 0] * c - d[:, 1] * sn, ry + d[:, 0] * sn + d[:, 1] * c], 1))
        pts = np.concatenate(pts)
        self.x0 = max(0, int(math.floor(pts[:, 0].min())) - 2); self.x1 = min(W, int(math.ceil(pts[:, 0].max())) + 3)
        self.y0 = max(0, int(math.floor(pts[:, 1].min())) - 2); self.y1 = min(H, int(math.ceil(pts[:, 1].max())) + 3)
        ph = np.asarray(photo, np.float32)
        self.tex = ph
        self.hole = grow1(grow1(grow1(h.mask)))
        self.fill = behind_fill(ph, self.hole)
        self.alpha, self.texd = decontaminate(ph, h.mask, self.fill)
        cxm, cym = float(xs.mean()) - rx, float(ys.mean()) - ry
        self.rc = (cxm, cym); self.Lc = max(4.0, math.hypot(cxm, cym))
        self.th = 0.0; self.thp = 0.0                     # charm: its turn about the pivot (now, a step ago)
        self.t = 0.0; self.acc = 0.0; self.awake = True
        self.fade = 0.0

    # ---- physics -------------------------------------------------------------------------------------
    def poke(self, px, py, strength=1.0):
        """A flick: nodes near the tap pushed sideways, away from where it was tapped."""
        if not self.ok:
            return False
        if not (self.x0 <= px < self.x1 and self.y0 <= py < self.y1):
            return False
        d = self.x - (px, py)
        dist = np.hypot(d[..., 0], d[..., 1])
        r = max(14.0, 0.35 * self.tlen)
        if dist[self.npin:].min() > r:
            return False
        side = (px - self.R[0]) * self.e1[0] + (py - self.R[1]) * self.e1[1]
        sgn = -1.0 if side > 0 else 1.0
        f = np.clip(1.0 - dist / (2 * r), 0, 1)
        v = 160.0 * strength * self.h.get("swing") * sgn * f              # px/s
        if self.h.kind == "charm":
            self.thp -= 160.0 * strength * self.h.get("swing") * sgn / self.Lc * DT * 1.5
            self.awake = True
            return True
        for j in range(self.npin, self.nrows):
            self.xp[j, :, 0] -= v[j] * self.e1[0] * DT; self.xp[j, :, 1] -= v[j] * self.e1[1] * DT
        self.awake = True
        return True

    def _substep(self, fx, fy, wind):
        P, h = self.P, DT
        x, xp, rest = self.x, self.xp, self.rest
        nc, n0 = self.nc, self.npin
        damp = 1.0 - P["c"] * h
        new = x.copy()
        for j in range(n0, self.nrows):
            k = self.kn[j]
            wk = wind * (0.55 + 0.45 * (j - n0 + 1) / (self.nrows - n0))
            for c in range(nc):
                ax_ = fx + k * (rest[j, c, 0] - x[j, c, 0]) + wk * self.e1[0]
                ay_ = fy + k * (rest[j, c, 1] - x[j, c, 1]) + wk * self.e1[1]
                new[j, c, 0] = x[j, c, 0] + (x[j, c, 0] - xp[j, c, 0]) * damp + ax_ * h * h
                new[j, c, 1] = x[j, c, 1] + (x[j, c, 1] - xp[j, c, 1]) * damp + ay_ * h * h
        self.xp = x
        self.x = new
        self._constrain()

    def _pendulum(self, fx, fy, wind):
        """A charm: one stiff body turning about its pivot (offset form: gravity's pull is the spring)."""
        h = DT
        c, s = math.cos(self.th), math.sin(self.th)
        rx_, ry_ = self.rc[0] * c - self.rc[1] * s, self.rc[0] * s + self.rc[1] * c
        gx, gy = fx + wind * self.e1[0], fy + wind * self.e1[1]
        acc = (rx_ * gy - ry_ * gx) / (self.Lc * self.Lc) - self.k * self.th
        new = self.th + (self.th - self.thp) * (1.0 - self.P["c"] * h) + acc * h * h
        new = max(-1.2, min(1.2, new))
        self.thp, self.th = self.th, new
        c, s = math.cos(new), math.sin(new)
        d = self.rest - self.R
        self.xp = self.x
        self.x = np.stack([self.R[0] + d[..., 0] * c - d[..., 1] * s, self.R[1] + d[..., 0] * s + d[..., 1] * c], -1)

    def _link(self, a, b, L, stiff):
        (ja, ca), (jb, cb) = a, b
        pa, pb = ja >= self.npin, jb >= self.npin                   # free?
        if not (pa or pb):
            return
        x = self.x
        dx, dy = x[jb, cb, 0] - x[ja, ca, 0], x[jb, cb, 1] - x[ja, ca, 1]
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return
        e = (d - L) / d * stiff
        if pa and pb:
            x[ja, ca, 0] += 0.5 * e * dx; x[ja, ca, 1] += 0.5 * e * dy
            x[jb, cb, 0] -= 0.5 * e * dx; x[jb, cb, 1] -= 0.5 * e * dy
        elif pa:
            x[ja, ca, 0] += e * dx; x[ja, ca, 1] += e * dy
        else:
            x[jb, cb, 0] -= e * dx; x[jb, cb, 1] -= e * dy

    def _constrain(self):
        rest, nc, sh = self.rest, self.nc, self.P["shear"]
        L = lambda a, b: float(np.hypot(*(rest[a] - rest[b])))
        for _ in range(ITERS):
            for j in range(max(1, self.npin), self.nrows):
                for c in range(nc):
                    self._link((j - 1, c), (j, c), L((j - 1, c), (j, c)), 1.0)
                for c in range(nc - 1):
                    self._link((j, c), (j, c + 1), L((j, c), (j, c + 1)), 1.0)
                    if j - 1 >= self.npin:                    # (each chain hinges at its own root)
                        self._link((j - 1, c), (j, c + 1), L((j - 1, c), (j, c + 1)), sh)
                        self._link((j - 1, c + 1), (j, c), L((j - 1, c + 1), (j, c)), sh)

    def step(self, dt, dn, jolt, wind_on=True):
        """dn: unit down in screen axes (upright (0, 1, 0)). jolt: g, gravity removed (screen axes)."""
        if not self.ok:
            return False
        P, h = self.P, self.h
        sw = h.get("swing")
        m = math.hypot(dn[0], dn[1])
        sw = sw * P["gain"]
        fx = sw * (P["g"] * dn[0] - P["j"] * jolt[0])          # the change in "down" (upright: (0, 1))
        fy = sw * (P["g"] * (dn[1] - m) - P["j"] * jolt[1])
        self.acc += min(dt, 0.1)
        moved = False
        while self.acc >= DT:
            self.acc -= DT
            self.t += DT
            T = 2 * math.pi * self.t
            wind = sw * h.get("breeze") * P["wind"] * (0.6 * math.sin(T / 3.1) + 0.4 * math.sin(T / 1.7 + 1.0)) if wind_on else 0.0
            if self.h.kind == "charm":
                self._pendulum(fx, fy, wind)
            else:
                self._substep(fx, fy, wind)
            moved = True
        disp = float(np.abs(self.x - self.rest).max())
        vel = float(np.abs(self.x - self.xp).max())
        if self.h.kind == "charm":
            vel = abs(self.th - self.thp) * self.tlen
        quiet = abs(fx) + abs(fy) < 1.0 and (h.get("breeze") <= 0 or not wind_on)
        if quiet and disp < 0.05 and vel < 0.002:
            self.x = self.rest.copy(); self.xp = self.rest.copy(); self.th = self.thp = 0.0
            self.awake = False
        else:
            self.awake = True
        self.fade = min(1.0, disp)
        return moved

    # ---- drawing -------------------------------------------------------------------------------------
    def render(self, content):
        """content: (H, W, 3) array, drawn into in place over this hang's region."""
        if not self.ok:
            return content
        x0, y0, x1, y1 = self.x0, self.y0, self.x1, self.y1
        reg = content[y0:y1, x0:x1]
        m = self.hole[y0:y1, x0:x1]
        reg[m] = self.fill[y0:y1, x0:x1][m]
        out = reg.astype(np.float32)
        al = self.alpha
        fade = self.fade
        for j in range(self.nrows - 1):
            for c in range(self.nc - 1):
                q = [(j, c), (j, c + 1), (j + 1, c + 1), (j + 1, c)]
                for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                    D = np.array([self.x[a] for a in tri]); S = np.array([self.rest[a] for a in tri])
                    self._tri(out, D, S, al, fade)
        content[y0:y1, x0:x1] = np.clip(out + 0.5, 0, 255).astype(content.dtype) if content.dtype == np.uint8 else out
        return content

    def _tri(self, out, D, S, al, fade):
        x0, y0, x1, y1 = self.x0, self.y0, self.x1, self.y1
        bx0 = max(x0, int(math.floor(D[:, 0].min()))); bx1 = min(x1 - 1, int(math.ceil(D[:, 0].max())))
        by0 = max(y0, int(math.floor(D[:, 1].min()))); by1 = min(y1 - 1, int(math.ceil(D[:, 1].max())))
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
        e = -1e-7
        inside = (l0 >= e) & (l1 >= e) & (l2 >= e)
        if not inside.any():
            return
        sx = l0 * S[0, 0] + l1 * S[1, 0] + l2 * S[2, 0]
        sy = l0 * S[0, 1] + l1 * S[1, 1] + l2 * S[2, 1]
        sx = np.clip(sx, 0, W - 1.001); sy = np.clip(sy, 0, H - 1.001)
        ix, iy = sx.astype(np.int32), sy.astype(np.int32); fx, fy = sx - ix, sy - iy
        def eff(v):                                   # at rest solid (the photo); moving: the estimated cover
            return np.where(v > 0, 256 * (1 - fade) + np.where(v == 255, 256, v) * fade, 0)
        a = ((eff(al[iy, ix]) * (1 - fx) + eff(al[iy, ix + 1]) * fx) * (1 - fy)
             + (eff(al[iy + 1, ix]) * (1 - fx) + eff(al[iy + 1, ix + 1]) * fx) * fy) / 256.0
        a = np.where(inside, a, 0.0)
        if not (a > 0).any():
            return
        T = self.tex * (1 - fade) + self.texd * fade if fade > 0 else self.tex
        col = ((T[iy, ix] * (1 - fx[..., None]) + T[iy, ix + 1] * fx[..., None]) * (1 - fy[..., None])
               + (T[iy + 1, ix] * (1 - fx[..., None]) + T[iy + 1, ix + 1] * fx[..., None]) * fy[..., None])
        sl = out[by0 - y0:by1 + 1 - y0, bx0 - x0:bx1 + 1 - x0]
        sl[:] = sl * (1 - a[..., None]) + col * a[..., None]


# ---- .jig lines --------------------------------------------------------------------------------------
def parse_lines(lines):
    """-> list of Hang (from 'hang' / 'hrow' lines; others ignored)."""
    out, cur = [], None
    for raw in lines:
        tok = raw.split()
        if not tok:
            continue
        if tok[0] == "hang" and len(tok) >= 4 and tok[1] in KINDS and len(out) < HANG_MAX:
            try:
                a = tuple(float(v) for v in tok[2].split(",")); b = tuple(float(v) for v in tok[3].split(","))
            except ValueError:
                cur = None; continue
            prm = {}
            for kv in tok[4:]:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    if k in HDEF:
                        try:
                            prm[k] = float(v)
                        except ValueError:
                            pass
            cur = Hang(tok[1], a, b, np.zeros((H, W), bool), prm)
            out.append(cur)
        elif tok[0] == "hrow" and cur is not None and len(tok) > 1:
            try:
                y = int(tok[1])
            except ValueError:
                continue
            if 0 <= y < H:
                for sp in tok[2:]:
                    if "-" in sp:
                        a, b = sp.split("-", 1)
                        try:
                            cur.mask[y, max(0, int(a)):min(W, int(b))] = True
                        except ValueError:
                            pass
    return out


def format_lines(hangs):
    out = []
    for h in hangs:
        if not h.ok():
            continue
        kv = " ".join(f"{k}={h.prm[k]:g}" for k in (p[0] for p in HPARAMS) if k in h.prm and abs(h.prm[k] - HDEF[k]) > 1e-6)
        out.append(f"hang {h.kind} {int(round(h.a[0]))},{int(round(h.a[1]))} {int(round(h.b[0]))},{int(round(h.b[1]))}" + (" " + kv if kv else ""))
        for y in range(H):
            row = h.mask[y]
            if not row.any():
                continue
            d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
            s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            out.append(f"hrow {y} " + " ".join(f"{a}-{b}" for a, b in zip(s, e)))
    return out
