"""
Mantis Gallery — jiggle engine for the PC editor.

A line-for-line port of src/jiggle_core.h (shapes, weights, springs, warp) so the preview moves the
way the Core2 will, plus .mg / .jig file handling and the editing logic used by the Jiggle tab in
mg_convert.py. No GUI code in here; numpy is needed for the live preview.

.jig file (hidden, next to the image: /images/santa.mg → /images/.santa.mg.jig):
    MGJIG1
    round 120 130 62 0 natural            shape  cx  cy  radius  rotation°  material
    tear 150 92 14 0 soft tip 0 -50       + control point at local (0, -0.50)
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field, replace
from pathlib import Path

try:
    import numpy as np
except ImportError:  # editing still works; the live preview needs numpy
    np = None

# ---- constants (keep in step with jiggle_core.h) -----------------------------------------------
IMG_W, IMG_H = 320, 240
MAX_ZONES = 4
R_MIN, R_MAX, R_DEFAULT = 10.0, 150.0, 45.0
BBOX_K, MARGIN, PLATEAU = 1.13, 1.25, 0.55
TIP_R, TIP_FADE = 0.15, 2.0
TIP_U, TIP_W = 0.0, -0.50
Q_MAX, STRETCH_MAX, REST_DAMP = 0.45, 0.07, 0.35
TIP_TRAVEL, PENDULUM = 0.02, 0.30

ROUND, HALF, TEAR = 0, 1, 2
SOFT, NATURAL, FIRM = 0, 1, 2

# name, label, depth, freq_mul
SHAPES = [("round", "Round", 1.00, 1.00), ("half", "Half", 0.50, 1.10), ("tear", "Tear", 0.55, 0.88)]
# name, label, f1, z1, f2, z2, stiffen, sens, couple
MATERIALS = [
    ("soft", "Soft", 2.0, 0.20, 3.1, 0.17, 1.2, 1.15, 0.85),
    ("natural", "Natural", 2.6, 0.25, 3.9, 0.21, 1.8, 1.00, 0.70),
    ("firm", "Firm", 3.5, 0.31, 5.0, 0.27, 2.6, 0.80, 0.50),
]

# Advanced tuning: key, label, lo, hi, help. Unset = use the material/shape preset. (jiggle_core.h Tune)
TUNES = [
    ("f1", "Main frequency", 0.8, 6.0, "Hz for a 50 px zone; bigger zones swing slower. Lower = heavier, softer."),
    ("z1", "Main damping", 0.05, 0.8, "How fast the swing dies. Flesh ≈ 0.2–0.35; under 0.15 looks rubbery."),
    ("f2", "Ripple frequency", 1.0, 9.0, "The outer layer that trails the mass. ~1.5× the main frequency."),
    ("z2", "Ripple damping", 0.05, 0.8, "How fast the ripple settles."),
    ("stiff", "Stiffening", 0.0, 5.0, "Tissue gets stiffer the further it stretches; stops big swings looking elastic."),
    ("sens", "Response to motion", 0.2, 2.0, "How strongly this zone reacts to moving the Core2."),
    ("couple", "Ripple strength", 0.0, 1.5, "How hard the outer layer is flung by the mass. 0 = moves as one piece."),
    ("travel", "Max swing", 0.3, 1.0, "Caps how far it can move (1 = the safe maximum)."),
    ("bulge", "Bulge", 0.0, 1.0, "Swell / dent when pushed toward or away from the screen."),
    ("stretch", "Stretch / squash", 0.0, 1.0, "Narrows as it stretches, widens as it squashes (keeps volume)."),
    ("sag", "Gravity sag", 0.0, 2.0, "How much it hangs toward the low side when you tilt the Core2."),
    ("pend", "Pendulum lift", 0.0, 0.45, "Swinging sideways also lifts it, pivoting from where it's attached."),
    ("pstiff", "Point stiffness", 1.0, 4.0, "Control point: how much stiffer than the body (it snaps back faster)."),
    ("pdamp", "Point damping", 0.1, 0.9, "Control point: how quickly its own wobble dies."),
    ("psize", "Point firm area", 0.8, 1.6, "Control point: size of the firm area around it."),
    ("dance", "Dance kick", 0.0, 2.0, "Strength of the beat kick in dance mode."),
    ("side", "Dance side", -1.0, 1.0, "-1 left, 0 centre, 1 right: which zones alternate in a twerk. Unset = by position."),
]
TUNE_KEYS = [t[0] for t in TUNES]

# Playback feel (keep in step with jiggle.cpp)
SENS, SAG, BASELINE_TAU, SAG_TAU, SAG_DEAD = 1.4, 0.7, 0.8, 5.0, 0.03

MG_MAGIC = b"MG01"
JIG_MAGIC = "MGJIG1"
BG_RGB = (24, 24, 24)  # COL_BG 0x18C3


# ---- shapes -------------------------------------------------------------------------------------
def _clamp01(t):
    return np.clip(t, 0.0, 1.0) if np is not None and isinstance(t, np.ndarray) else min(1.0, max(0.0, t))


def smooth01(t):
    t = _clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def shape_sdf(shape, u, w):
    """Signed distance (local units, u right, w up, negative inside). Works on floats or arrays."""
    xp = np if (np is not None and isinstance(u, np.ndarray)) else None
    sqrt = xp.sqrt if xp else math.sqrt
    if shape == HALF:
        y = w + 0.5
        arc = sqrt(u * u + y * y) - 1.0
        return xp.maximum(arc, -y) if xp else max(arc, -y)
    if shape == TEAR:
        r1, r2, h = 0.55, 0.10, 1.30
        b = (r1 - r2) / h
        a = math.sqrt(1.0 - b * b)
        px = abs(u) if not xp else xp.abs(u)
        py = w + 0.415
        k = -b * px + a * py
        d_lo = sqrt(px * px + py * py) - r1
        d_hi = sqrt(px * px + (py - h) * (py - h)) - r2
        d_mid = a * px + b * py - r1
        if xp:
            return xp.where(k < 0, d_lo, xp.where(k > a * h, d_hi, d_mid))
        return d_lo if k < 0 else (d_hi if k > a * h else d_mid)
    return sqrt(u * u + w * w) - 1.0


def shape_attach(shape):
    return 0.0, (-0.5 if shape == HALF else (0.95 if shape == TEAR else 0.9))


def shape_tip_default(shape):
    return 0.0, (0.15 if shape == HALF else (TIP_W if shape == TEAR else -0.25))


def shape_free(shape, u, w, has_tip=False, pu=0.0, pw=0.0):
    if has_tip:
        au, aw = shape_attach(shape)
        nu, nw = pu - au, pw - aw
        L = math.sqrt(nu * nu + nw * nw)
        if L > 0.2:
            along = ((u - au) * nu + (w - aw) * nw) / (L * L * 1.15)
            base = 0.35 if shape == ROUND else 0.0
            return base + (1.0 - base) * smooth01(along)
    if shape == HALF:
        return smooth01((w + 0.5) / 1.0)
    if shape == TEAR:
        return smooth01((0.95 - w) / 1.9)
    return 0.35 + 0.65 * smooth01((0.9 - w) / 1.8)


def shape_outline(shape, npts=64):
    """Outline points (local units) by bisection along rays — all shapes are convex."""
    pts = []
    for i in range(npts):
        a = 2 * math.pi * i / npts
        du, dw = math.cos(a), math.sin(a)
        lo, hi = 0.0, 1.25
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            if shape_sdf(shape, du * mid, dw * mid) < 0:
                lo = mid
            else:
                hi = mid
        pts.append((du * lo, dw * lo))
    return pts


# ---- zones --------------------------------------------------------------------------------------
@dataclass
class Zone:
    shape: int = ROUND
    material: int = NATURAL
    cx: float = IMG_W / 2
    cy: float = IMG_H / 2
    r: float = R_DEFAULT
    rot: int = 0
    tip: bool = False
    tip_u: float = TIP_U
    tip_w: float = TIP_W
    tune: dict = field(default_factory=dict)       # key → value overrides (see TUNES)

    def tuned(self, key, default):
        if key not in self.tune:
            return default
        _, _, lo, hi, _ = TUNES[TUNE_KEYS.index(key)]
        return min(hi, max(lo, float(self.tune[key])))

    def tip_radius(self):
        return TIP_R * self.tuned("psize", 1.0)

    def axes(self):
        a = math.radians(self.rot)
        return math.cos(a), math.sin(a)

    def to_local(self, x, y):
        c, s = self.axes()
        dx, dy = x - self.cx, y - self.cy
        return (dx * c + dy * s) / self.r, -(-dx * s + dy * c) / self.r

    def from_local(self, u, w):
        c, s = self.axes()
        a, b = u, -w
        return self.cx + self.r * (a * c - b * s), self.cy + self.r * (a * s + b * c)

    def contains(self, x, y, pad=6.0):
        u, w = self.to_local(x, y)
        return shape_sdf(self.shape, u, w) * self.r < pad

    def tip_xy(self):
        return self.from_local(self.tip_u, self.tip_w)

    def outline(self, scale=1.0, npts=64):
        return [self.from_local(u * scale, w * scale) for u, w in shape_outline(self.shape, npts)]


def zone_bbox(z: Zone):
    e = BBOX_K * MARGIN * z.r + 2.0
    x0, y0 = max(0, math.floor(z.cx - e)), max(0, math.floor(z.cy - e))
    x1, y1 = min(IMG_W, math.ceil(z.cx + e)), min(IMG_H, math.ceil(z.cy + e))
    return x0, y0, x1 - x0, y1 - y0


def _q8(a):
    """Quantise 0..1 to 0..255 exactly like the firmware (×255 + 0.5, truncated)."""
    return np.floor(np.asarray(a, dtype=np.float64) * 255.0 + 0.5).astype(np.uint8)


class ZoneSim:
    """Runtime state of one zone: maps, spring constants, physics (port of prepare/step/stage)."""

    def __init__(self, z: Zone):
        self.z = z
        si, mi = SHAPES[z.shape], MATERIALS[z.material]
        bx, by, bw, bh = zone_bbox(z)
        self.bx, self.by, self.bw, self.bh = bx, by, bw, bh
        self.ok = bw > 0 and bh > 0
        c, s = z.axes()
        if self.ok and np is not None:
            X, Y = np.meshgrid(np.arange(bx, bx + bw, dtype=np.float64), np.arange(by, by + bh, dtype=np.float64))
            dx, dy = X - z.cx, Y - z.cy
            U = (dx * c + dy * s) / z.r
            Wl = -(-dx * s + dy * c) / z.r
            self.w = _q8(smooth01(-shape_sdf(z.shape, U / MARGIN, Wl / MARGIN) / (PLATEAU * si[2]))) / 255.0
            self.g = _q8(shape_free(z.shape, U, Wl, z.tip, z.tip_u, z.tip_w)) / 255.0
            self.X, self.Y = X, Y
            self.t = None
            if z.tip:
                du, dw = U - z.tip_u, Wl - z.tip_w
                d = np.sqrt(du * du + dw * dw)
                tr = z.tip_radius()
                tw = smooth01((tr * TIP_FADE - d) / (tr * (TIP_FADE - 0.5)))
                # firmware only keeps the tip map inside its own box; outside it is 0
                tx, ty = z.tip_xy()
                e = tr * TIP_FADE * z.r + 2.0
                a0, b0 = max(0, math.floor(tx - e)), max(0, math.floor(ty - e))
                a1, b1 = min(IMG_W, math.ceil(tx + e)), min(IMG_H, math.ceil(ty + e))
                inbox = (X >= a0) & (X < a1) & (Y >= b0) & (Y < b1)
                self.t = np.where(inbox, _q8(tw), 0).astype(np.float64) / 255.0
        self.tcx, self.tcy = z.tip_xy()
        self.tip_gc = shape_free(z.shape, z.tip_u, z.tip_w, True, z.tip_u, z.tip_w)
        au, aw = shape_attach(z.shape)
        pu, pw = (z.tip_u, z.tip_w) if z.tip else (0.0, 0.0)
        ax, ay = z.from_local(au, aw)
        px, py = z.from_local(pu, pw)
        hx, hy = px - ax, py - ay
        hl = math.hypot(hx, hy)
        self.hang = (hx / hl, hy / hl) if hl > 1e-3 else (0.0, 1.0)

        size_k = min(1.35, max(0.65, math.sqrt(50.0 / z.r)))
        T = z.tuned
        self.w1 = 2 * math.pi * T("f1", mi[2]) * si[3] * size_k
        self.w2 = 2 * math.pi * T("f2", mi[4]) * si[3] * size_k
        self.wt = self.w1 * 2.3 * T("pstiff", 1.0)
        self.zt = T("pdamp", 0.38)
        self.z1, self.z2 = T("z1", mi[3]), T("z2", mi[5])
        self.stiffen, self.sens, self.couple = T("stiff", mi[6]), T("sens", mi[7]), T("couple", mi[8])
        self.sag, self.pend, self.dance = T("sag", 1.0), T("pend", PENDULUM), T("dance", 1.0)
        self.side = int(round(T("side", 0))) if "side" in z.tune else None
        self.lim_px = 0.22 * si[2] * z.r * T("travel", 1.0)
        self.s_lim = 0.35 / (1.0 + 1.5 / (PLATEAU * si[2])) * T("bulge", 1.0)
        self.st_max = STRETCH_MAX * T("stretch", 1.0)
        self.rest()

    def rest(self):
        self.dx = self.dy = self.vx = self.vy = 0.0
        self.qx = self.qy = self.qvx = self.qvy = 0.0
        self.s = self.vs = 0.0
        self.tx = self.ty = self.tvx = self.tvy = 0.0
        self.awake = False

    @staticmethod
    def _clamp(x, y, vx, vy, m):
        m2 = x * x + y * y
        if m2 <= m * m:
            return x, y, vx, vy
        mm = math.sqrt(m2)
        nx, ny = x / mm, y / mm
        x, y = nx * m, ny * m
        vr = vx * nx + vy * ny
        if vr > 0:
            vx -= vr * nx
            vy -= vr * ny
        return x, y, vx * 0.8, vy * 0.8

    def step(self, h, tgx, tgy, tgs):
        tgx *= self.sens; tgy *= self.sens; tgs *= self.sens
        d2 = self.dx * self.dx + self.dy * self.dy
        k1 = self.w1 * self.w1 * (1.0 + self.stiffen * d2)
        sp2 = self.vx * self.vx + self.vy * self.vy
        c1 = 2.0 * (self.z1 + REST_DAMP / (1.0 + 4.0 * sp2)) * self.w1
        ax = k1 * (tgx - self.dx) - c1 * self.vx
        ay = k1 * (tgy - self.dy) - c1 * self.vy
        self.vx += ax * h; self.vy += ay * h
        self.dx += self.vx * h; self.dy += self.vy * h
        self.dx, self.dy, self.vx, self.vy = self._clamp(self.dx, self.dy, self.vx, self.vy, 1.0)

        drive = self.couple / (self.w1 * self.w1)
        q2 = self.qx * self.qx + self.qy * self.qy
        k2 = self.w2 * self.w2 * (1.0 + 3.0 * q2)
        qs2 = self.qvx * self.qvx + self.qvy * self.qvy
        c2 = 2.0 * (self.z2 + REST_DAMP / (1.0 + 4.0 * qs2)) * self.w2
        qax = -k2 * self.qx - c2 * self.qvx - drive * self.w2 * self.w2 * ax
        qay = -k2 * self.qy - c2 * self.qvy - drive * self.w2 * self.w2 * ay
        self.qvx += qax * h; self.qvy += qay * h
        self.qx += self.qvx * h; self.qy += self.qvy * h
        self.qx, self.qy, self.qvx, self.qvy = self._clamp(self.qx, self.qy, self.qvx, self.qvy, Q_MAX)

        ws = self.w1 * 1.3
        cs = 2.0 * (self.z1 + REST_DAMP / (1.0 + 4.0 * self.vs * self.vs)) * ws
        self.vs += (ws * ws * (1.0 + self.stiffen * self.s * self.s) * (tgs - self.s) - cs * self.vs) * h
        self.s += self.vs * h
        if self.s > 1.0:
            self.s = 1.0; self.vs = min(self.vs, 0.0)
        if self.s < -1.0:
            self.s = -1.0; self.vs = max(self.vs, 0.0)

        if self.z.tip:
            lim_ratio = self.lim_px / (TIP_TRAVEL * self.z.r)
            bx, by = (ax + self.tip_gc * qax) * lim_ratio, (ay + self.tip_gc * qay) * lim_ratio
            ct = 2.0 * self.zt * self.wt
            tax = -self.wt * self.wt * self.tx - ct * self.tvx - 0.6 * bx
            tay = -self.wt * self.wt * self.ty - ct * self.tvy - 0.6 * by
            self.tvx += tax * h; self.tvy += tay * h
            self.tx += self.tvx * h; self.ty += self.tvy * h
            self.tx, self.ty, self.tvx, self.tvy = self._clamp(self.tx, self.ty, self.tvx, self.tvy, 1.0)

    def settled(self, tx, ty, ts):
        e, v = 0.004, 0.03
        return (abs(self.dx - tx * self.sens) < e and abs(self.dy - ty * self.sens) < e and abs(self.s - ts * self.sens) < e
                and abs(self.qx) < e and abs(self.qy) < e and abs(self.tx) < 0.02 and abs(self.ty) < 0.02
                and abs(self.vx) < v and abs(self.vy) < v and abs(self.vs) < v and abs(self.qvx) < v and abs(self.qvy) < v
                and abs(self.tvx) < 0.1 and abs(self.tvy) < 0.1 and abs(tx) < e and abs(ty) < e and abs(ts) < e)

    def poke(self, px, py, strength=1.0):
        ax, ay = self.z.cx - px, self.z.cy - py
        m = math.hypot(ax, ay)
        k = strength * self.w1 * 0.45
        if m > 0.5:
            self.vx += k * ax / m; self.vy += k * ay / m
        self.vs -= strength * self.w1 * 0.5
        self.awake = True

    def dance_kick(self, amp, side, sign):
        if self.side is not None:
            side = self.side
        k = amp * self.w1 * self.dance
        self.vy += k * 0.70 * (1.0 + 0.5 * side * sign)
        self.vx += k * 0.25 * sign
        self.vs += k * 0.20
        self.awake = True

    def stage(self):
        """Per-frame linear terms (port of stage_frame)."""
        z = self.z
        b0x, b0y = self.dx * self.lim_px, self.dy * self.lim_px
        hx, hy = self.hang
        perp = -self.dx * hy + self.dy * hx
        lift = self.pend * perp * perp * self.lim_px
        b0x -= lift * hx; b0y -= lift * hy
        b1x, b1y = self.qx * self.lim_px, self.qy * self.lim_px
        sc = self.s * self.s_lim
        if z.tip:
            b0x -= sc * (self.tcx - z.cx); b0y -= sc * (self.tcy - z.cy)
        m = math.hypot(self.dx, self.dy)
        e = self.st_max * min(m, 1.0)
        nx, ny = (self.dx / m, self.dy / m) if m > 1e-4 else (0.0, 0.0)
        iso, an = sc - 0.5 * e, 1.5 * e
        M = (iso + an * nx * nx, an * nx * ny, an * nx * ny, iso + an * ny * ny)
        ot = None
        if z.tip:
            rx, ry = self.tcx - z.cx, self.tcy - z.cy
            ot = (b0x + self.tip_gc * b1x + M[0] * rx + M[1] * ry + self.tx * TIP_TRAVEL * z.r,
                  b0y + self.tip_gc * b1y + M[2] * rx + M[3] * ry + self.ty * TIP_TRAVEL * z.r)
        return b0x, b0y, b1x, b1y, M, ot

    def add_offsets(self, OX, OY):
        """Add this zone's offset field (pixels) into full-frame arrays."""
        if not self.ok or np is None:
            return
        b0x, b0y, b1x, b1y, M, ot = self.stage()
        RX, RY = self.X - self.z.cx, self.Y - self.z.cy
        Bx = b0x + self.g * b1x + M[0] * RX + M[1] * RY
        By = b0y + self.g * b1y + M[2] * RX + M[3] * RY
        if self.t is not None and ot is not None:
            Bx = Bx + self.t * (ot[0] - Bx)
            By = By + self.t * (ot[1] - By)
        sl = (slice(self.by, self.by + self.bh), slice(self.bx, self.bx + self.bw))
        OX[sl] += self.w * Bx
        OY[sl] += self.w * By


# ---- playback (port of play_tick's motion handling) ---------------------------------------------
class Sim:
    """Drives zones like the Core2 does. Feed screen-space accelerometer readings (g, gravity
    included: the Core2 held upright reads (0, -1, 0)), plus beats and pokes."""

    def __init__(self, zones, image_rgb=None):
        self.zs = [ZoneSim(z) for z in zones]
        self.img = image_rgb
        self.base = None
        self.slow = None

    def step(self, dt, accel=(0.0, -1.0, 0.0), dance=0.0):
        a = list(accel)
        if self.base is None:
            self.base, self.slow = a[:], a[:]
        k, ks = 1 - math.exp(-dt / BASELINE_TAU), 1 - math.exp(-dt / SAG_TAU)
        for i in range(3):
            self.base[i] += (a[i] - self.base[i]) * k
            self.slow[i] += (self.base[i] - self.slow[i]) * ks
        sens = SENS * (1.0 - 0.5 * dance)
        cl = lambda v: max(-1.0, min(1.0, v))
        jx, jy = -sens * (a[0] - self.base[0]), -sens * (a[1] - self.base[1])
        ts = cl(-sens * (a[2] - self.base[2]))
        dz = lambda v: v - SAG_DEAD if v > SAG_DEAD else (v + SAG_DEAD if v < -SAG_DEAD else 0.0)
        gx, gy = -SAG * dz(self.base[0] - self.slow[0]), -SAG * dz(self.base[1] - self.slow[1])
        stirred = abs(jx + gx) > 0.03 or abs(jy + gy) > 0.03 or abs(ts) > 0.03
        for z in self.zs:
            if stirred:
                z.awake = True
            tx, ty = cl(jx + z.sag * gx), cl(jy + z.sag * gy)
            if z.awake:
                n = max(1, math.ceil(dt / 0.004))
                for _ in range(n):
                    z.step(dt / n, tx, ty, ts)
                if z.settled(tx, ty, ts):
                    z.rest()

    def beat(self, amp, sign):
        if not self.zs:
            return
        cen = sum(z.z.cx for z in self.zs) / len(self.zs)
        for z in self.zs:
            side = -1 if z.z.cx < cen - 3 else (1 if z.z.cx > cen + 3 else 0)
            z.dance_kick(amp, side, sign)

    def poke(self, x, y):
        for z in reversed(self.zs):
            if z.z.contains(x, y):
                z.poke(x, y)
                return True
        return False

    def awake(self):
        return any(z.awake for z in self.zs)

    def offsets(self):
        OX = np.zeros((IMG_H, IMG_W)); OY = np.zeros((IMG_H, IMG_W))
        for z in self.zs:
            z.add_offsets(OX, OY)
        return OX, OY

    def render(self):
        """Warped frame as uint8 RGB (H, W, 3)."""
        OX, OY = self.offsets()
        return warp(self.img, OX, OY)


def warp(img, OX, OY):
    """Bilinear sample of img at (x − OX, y − OY), clamped like the firmware."""
    H, W = img.shape[:2]
    X, Y = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    sx = np.clip(X - OX, 0, W - 1.001)
    sy = np.clip(Y - OY, 0, H - 1.001)
    ix, iy = sx.astype(np.int32), sy.astype(np.int32)
    fx, fy = (sx - ix)[..., None], (sy - iy)[..., None]
    f = img.astype(np.float32)
    top = f[iy, ix] * (1 - fx) + f[iy, ix + 1] * fx
    bot = f[iy + 1, ix] * (1 - fx) + f[iy + 1, ix + 1] * fx
    return np.clip(top * (1 - fy) + bot * fy + 0.5, 0, 255).astype(np.uint8)


# ---- files --------------------------------------------------------------------------------------
def read_mg(path) -> "np.ndarray":
    """.mg → (240, 320, 3) uint8, laid out like the Core2 shows it (smaller images centred)."""
    data = Path(path).read_bytes()
    if len(data) < 8 or data[:4] != MG_MAGIC:
        raise ValueError("not an .mg file (missing MG01 header)")
    w, h = struct.unpack("<HH", data[4:8])
    if not (0 < w <= IMG_W and 0 < h <= IMG_H) or len(data) < 8 + w * h * 2:
        raise ValueError(f"bad .mg size {w}x{h}")
    px = np.frombuffer(data, dtype="<u2", count=w * h, offset=8).reshape(h, w)
    rgb = np.stack([((px >> 11) & 31) * 255 // 31, ((px >> 5) & 63) * 255 // 63, (px & 31) * 255 // 31], -1).astype(np.uint8)
    out = np.empty((IMG_H, IMG_W, 3), np.uint8); out[:] = BG_RGB
    x0, y0 = (IMG_W - w) // 2, (IMG_H - h) // 2
    out[y0:y0 + h, x0:x0 + w] = rgb
    return out


def jig_path_for(mg_path) -> Path:
    """The firmware's sidecar name: /images/santa.mg → /images/.santa.mg.jig"""
    p = Path(mg_path)
    return p.with_name("." + p.name + ".jig")


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def parse_jig(text: str) -> list[Zone]:
    lines = [l.strip() for l in text.splitlines()]
    if not lines or not lines[0].startswith(JIG_MAGIC):
        raise ValueError("not a .jig file (missing MGJIG1 header)")
    names = [s[0] for s in SHAPES]
    mats = [m[0] for m in MATERIALS]
    zones = []
    for line in lines[1:]:
        tok = line.split()
        if len(tok) < 5 or tok[0] not in names:
            continue
        try:
            cx, cy, r, rot = (int(t) for t in tok[1:5])
        except ValueError:
            continue
        z = Zone(shape=names.index(tok[0]), cx=_clamp(cx, 0, IMG_W), cy=_clamp(cy, 0, IMG_H),
                 r=_clamp(r, R_MIN, R_MAX), rot=((rot % 360) + 360) % 360)
        rest = tok[5:]
        i = 0
        while i < len(rest):
            t = rest[i]
            if t in mats:
                z.material = mats.index(t)
            elif "=" in t:
                k, _, v = t.partition("=")
                if k in TUNE_KEYS:
                    try:
                        _, _, lo, hi, _ = TUNES[TUNE_KEYS.index(k)]
                        fv = float(v)
                        if fv == fv:
                            z.tune[k] = min(hi, max(lo, fv))
                    except ValueError:
                        pass
            elif t == "tip" and i + 2 < len(rest):
                try:
                    tu, tw = int(rest[i + 1]), int(rest[i + 2])
                    z.tip, z.tip_u, z.tip_w = True, _clamp(tu / 100, -0.6, 0.6), _clamp(tw / 100, -1.0, 0.9)
                    i += 2
                except (ValueError, IndexError):
                    pass
            i += 1
        zones.append(z)
        if len(zones) >= MAX_ZONES:
            break
    return zones


def format_jig(zones: list[Zone]) -> str:
    out = [JIG_MAGIC]
    for z in zones[:MAX_ZONES]:
        line = f"{SHAPES[z.shape][0]} {round(z.cx)} {round(z.cy)} {round(z.r)} {int(z.rot)} {MATERIALS[z.material][0]}"
        if z.tip:
            line += f" tip {round(z.tip_u * 100)} {round(z.tip_w * 100)}"
        for k in TUNE_KEYS:
            if k in z.tune:
                line += f" {k}={z.tuned(k, 0.0):.3g}"
        out.append(line)
    return "\n".join(out) + "\n"


# ---- editing (what the Jiggle tab drives) ---------------------------------------------------------
@dataclass
class Editor:
    """Zones on one image, selection, and mouse/keyboard editing. Coordinates are image pixels."""
    zones: list = field(default_factory=list)
    selected: int = -1
    armed: int = -1                 # shape waiting to be placed by a click
    mg_path: Path | None = None
    image: object = None
    dirty: bool = False
    last_r: float = R_DEFAULT
    last_material: int = NATURAL
    _drag: str = ""                 # "", "zone", "tip"
    _last: tuple = (0, 0)

    # files
    def open_mg(self, path):
        self.image = read_mg(path)
        self.mg_path = Path(path)
        jp = jig_path_for(path)
        self.zones = parse_jig(jp.read_text(encoding="utf-8")) if jp.exists() else []
        self.selected, self.armed, self.dirty = -1, -1, False
        return jp.exists()

    def save(self) -> Path:
        if self.mg_path is None:
            raise ValueError("open an .mg image first")
        jp = jig_path_for(self.mg_path)
        if self.zones:
            jp.write_text(format_jig(self.zones), encoding="utf-8", newline="\n")
        elif jp.exists():
            jp.unlink()
        self.dirty = False
        return jp

    @property
    def zone(self):
        return self.zones[self.selected] if 0 <= self.selected < len(self.zones) else None

    def _changed(self):
        self.dirty = True

    # placing / selecting
    def arm(self, shape):
        if len(self.zones) >= MAX_ZONES and self.armed != shape:
            return False
        self.armed = -1 if self.armed == shape else shape
        return True

    def hit(self, x, y):
        for i in range(len(self.zones) - 1, -1, -1):
            if self.zones[i].contains(x, y):
                return i
        return -1

    def near_tip(self, x, y):
        z = self.zone
        if not z or not z.tip:
            return False
        tx, ty = z.tip_xy()
        rr = z.tip_radius() * z.r + 6.0
        return (x - tx) ** 2 + (y - ty) ** 2 <= rr * rr

    def press(self, x, y):
        self._last = (x, y)
        if self.armed >= 0:
            if len(self.zones) < MAX_ZONES:
                tu, tw = shape_tip_default(self.armed)
                self.zones.append(Zone(shape=self.armed, material=self.last_material, cx=x, cy=y,
                                       r=self.last_r, tip_u=tu, tip_w=tw))
                self.selected = len(self.zones) - 1
                self._changed()
            self.armed = -1
            self._drag = "zone"
            return
        if self.near_tip(x, y):
            self._drag = "tip"
            return
        i = self.hit(x, y)
        self.selected = i
        self._drag = "zone" if i >= 0 else ""

    def drag(self, x, y):
        dx, dy = x - self._last[0], y - self._last[1]
        self._last = (x, y)
        z = self.zone
        if not z or not self._drag or (dx == 0 and dy == 0):
            return
        if self._drag == "tip":
            self.move_tip_screen(dx, dy)
        else:
            z.cx = _clamp(z.cx + dx, 0, IMG_W)
            z.cy = _clamp(z.cy + dy, 0, IMG_H)
        self._changed()

    def release(self, x, y):
        self._drag = ""

    def move_tip_screen(self, dx, dy):
        z = self.zone
        c, s = z.axes()
        nu = _clamp(z.tip_u + (dx * c + dy * s) / z.r, -0.6, 0.6)
        nw = _clamp(z.tip_w - (-dx * s + dy * c) / z.r, -1.0, 0.9)
        if shape_sdf(z.shape, nu, nw) < -z.tip_radius() * 0.5:
            z.tip_u, z.tip_w = nu, nw

    # properties
    def resize(self, factor):
        z = self.zone
        if z:
            nr = z.r * factor
            if abs(nr - z.r) < 1.0:
                nr = z.r + (1.0 if factor > 1 else -1.0)
            z.r = _clamp(nr, R_MIN, R_MAX)
            self.last_r = z.r
            self._changed()

    def rotate(self, deg):
        z = self.zone
        if z:
            z.rot = int((z.rot + deg) % 360)
            self._changed()

    def set_material(self, m):
        z = self.zone
        if z:
            z.material = m
            self.last_material = m
            self._changed()

    def set_tip(self, on):
        z = self.zone
        if z:
            z.tip = bool(on)
            if z.tip and shape_sdf(z.shape, z.tip_u, z.tip_w) > -z.tip_radius() * 0.5:
                z.tip_u, z.tip_w = shape_tip_default(z.shape)
            self._changed()

    def set_shape(self, shape):
        z = self.zone
        if z and z.shape != shape:
            z.shape = shape
            if shape_sdf(shape, z.tip_u, z.tip_w) > -z.tip_radius() * 0.5:
                z.tip_u, z.tip_w = shape_tip_default(shape)
            self._changed()

    def delete(self):
        if self.zone:
            del self.zones[self.selected]
            self.selected = -1
            self._changed()

    def nudge(self, dx, dy):
        z = self.zone
        if z:
            z.cx = _clamp(z.cx + dx, 0, IMG_W)
            z.cy = _clamp(z.cy + dy, 0, IMG_H)
            self._changed()

    def set_tune(self, key, value):
        z = self.zone
        if z:
            if value is None:
                z.tune.pop(key, None)
            else:
                _, _, lo, hi, _ = TUNES[TUNE_KEYS.index(key)]
                z.tune[key] = min(hi, max(lo, float(value)))
            self._changed()

    def snapshot(self):
        return [replace(z, tune=dict(z.tune)) for z in self.zones]


# ---- motion presets for the preview -------------------------------------------------------------
def motion_accel(kind: str, t: float, tilt_deg: float = 0.0):
    """Screen-space accelerometer reading (g) for a preview motion at time t since it started."""
    th = math.radians(tilt_deg)
    gx, gy = -math.sin(th), -math.cos(th)          # held upright, tilted sideways by tilt_deg
    ax = ay = az = 0.0
    pulse = lambda t0, w: math.exp(-((t - t0) / w) ** 2)
    if kind == "shake":                              # quick shove right, then back
        ax = 1.2 * pulse(0.10, 0.05) - 1.0 * pulse(0.25, 0.06)
    elif kind == "bounce":                           # hop: push up, land
        ay = -1.0 * pulse(0.10, 0.05) + 1.2 * pulse(0.35, 0.05)
    elif kind == "walk":                             # steps at ~110 per minute for 4 s
        if t < 4.0:
            ph = (t * 110 / 60) % 1.0
            ay = -0.45 * math.exp(-((ph - 0.1) / 0.06) ** 2) + 0.15 * math.sin(2 * math.pi * ph)
            ax = 0.12 * math.sin(math.pi * t * 110 / 60)
    elif kind == "jolt":                             # push toward the screen
        az = 1.0 * pulse(0.08, 0.04) - 0.8 * pulse(0.2, 0.05)
    return gx + ax, gy + ay, az


# ---- analysis helpers for the editor --------------------------------------------------------------
def free_point(z: Zone):
    """A point toward the zone's free end (where the swing shows most): the control point if any."""
    if z.tip:
        return z.tip_xy()
    zs = ZoneSim(replace(z, tune=dict(z.tune)))
    hx, hy = zs.hang
    return z.cx + hx * 0.55 * z.r, z.cy + hy * 0.55 * z.r


def step_response(z: Zone, kind="shove", seconds=2.5, dt=0.004):
    """Offset (px, along the shove) at the free point after a standard 60 ms sideways shove (or a
    vertical drop for kind="drop"). Returns (times, values, stats)."""
    sim = ZoneSim(replace(z, tune=dict(z.tune)))
    px, py = free_point(z)
    ix, iy = int(round(px)), int(round(py))
    lx, ly = ix - sim.bx, iy - sim.by
    inside = sim.ok and np is not None and 0 <= lx < sim.bw and 0 <= ly < sim.bh
    w = float(sim.w[ly, lx]) if inside else 1.0
    g = float(sim.g[ly, lx]) if inside else 1.0
    t_list, v_list = [], []
    n = int(seconds / dt)
    for i in range(n):
        t = i * dt
        push = 1.0 if t < 0.06 else 0.0
        if kind == "drop":
            sim.step(dt, 0.0, push, 0.0)
        else:
            sim.step(dt, push, 0.0, 0.0)
        b0x, b0y, b1x, b1y, M, ot = sim.stage()
        rx, ry = ix - z.cx, iy - z.cy
        bx = b0x + g * b1x + M[0] * rx + M[1] * ry
        by = b0y + g * b1y + M[2] * rx + M[3] * ry
        if ot is not None and z.tip:
            bx, by = ot
        v = w * (by if kind == "drop" else bx)
        t_list.append(t); v_list.append(v)
    peak = max((abs(v) for v in v_list), default=0.0)
    settle = 0.0
    for t, v in zip(reversed(t_list), reversed(v_list)):
        if abs(v) > 0.25:
            settle = t
            break
    swings, last_ext, direction = 0, 0.0, 0
    for i in range(1, len(v_list)):
        d = 1 if v_list[i] > v_list[i - 1] else -1
        if direction and d != direction and abs(v_list[i - 1]) > 0.5 and abs(v_list[i - 1] - last_ext) > 0.3:
            swings += 1
            last_ext = v_list[i - 1]
        direction = d
    sim_hz = sim.w1 / (2 * math.pi)
    return t_list, v_list, {"peak_px": peak, "settle_ms": settle * 1000, "swings": swings, "hz": sim_hz}


def maps_overlay(img, z: Zone):
    """Tint the image to show the selected zone's maps: red = how much it moves, blue = free end,
    yellow = the control point's firm area."""
    out = img.astype(np.float32).copy()
    sim = ZoneSim(replace(z, tune=dict(z.tune)))
    if not sim.ok:
        return img
    sl = (slice(sim.by, sim.by + sim.bh), slice(sim.bx, sim.bx + sim.bw))
    w, g = sim.w[..., None], sim.g[..., None]
    tint = np.zeros((sim.bh, sim.bw, 3), np.float32)
    tint += np.array([255, 40, 40], np.float32) * w * (1 - g)
    tint += np.array([60, 120, 255], np.float32) * w * g
    a = 0.55 * w
    if sim.t is not None:
        t = sim.t[..., None]
        tint = tint * (1 - t) + np.array([255, 230, 40], np.float32) * t
        a = np.maximum(a, 0.7 * t)
    out[sl] = out[sl] * (1 - a) + tint * a
    return np.clip(out, 0, 255).astype(np.uint8)


def mirrored(z: Zone, axis_x: float = IMG_W / 2) -> Zone:
    """Copy reflected across a vertical line (for symmetric pairs)."""
    m = replace(z, tune=dict(z.tune))
    m.cx = _clamp(2 * axis_x - z.cx, 0, IMG_W)
    m.rot = (-z.rot) % 360
    m.tip_u = -z.tip_u
    if "side" in m.tune:
        m.tune["side"] = -m.tune["side"]
    return m

