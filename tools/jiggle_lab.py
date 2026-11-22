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
R_MIN, R_MAX, R_DEFAULT = 6.0, 150.0, 45.0
BBOX_K, MARGIN, PLATEAU = 1.13, 1.25, 0.55
TIP_R, TIP_FADE = 0.15, 2.0
TIP_U, TIP_W = 0.0, -0.50
Q_MAX, STRETCH_MAX, REST_DAMP = 0.45, 0.07, 0.30
TIP_TRAVEL, PENDULUM = 0.02, 0.30
FORM_MIX, FOLD_MARGIN, AXIS_K = 0.25, 0.12, 0.5

ROUND, HALF, TEAR = 0, 1, 2
SOFT, NATURAL, FIRM = 0, 1, 2

# name, label, depth, freq_mul
SHAPES = [("round", "Round", 1.00, 1.00), ("half", "Half", 0.50, 1.10), ("tear", "Tear", 0.55, 0.88)]
# name, label, f1, z1, f2, z2, stiffen, sens, couple
MATERIALS = [
    ("soft", "Soft", 2.0, 0.17, 3.1, 0.14, 1.2, 1.15, 0.85),
    ("natural", "Natural", 2.6, 0.21, 3.9, 0.18, 1.8, 1.00, 0.70),
    ("firm", "Firm", 3.5, 0.27, 5.0, 0.23, 2.6, 0.80, 0.50),
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
    ("travel", "Max swing", 0.3, 1.6, "How far it can move. Default comes from the form and mass; the fold guard keeps any setting safe."),
    ("bulge", "Bulge", 0.0, 1.0, "Swell / dent when pushed toward or away from the screen."),
    ("stretch", "Stretch / squash", 0.0, 1.0, "Narrows as it stretches, widens as it squashes (keeps volume)."),
    ("sag", "Gravity sag", 0.0, 2.0, "How much it hangs toward the low side when you tilt the Core2."),
    ("pend", "Pendulum lift", 0.0, 0.45, "Swinging sideways also lifts it, pivoting from where it's attached."),
    ("proj", "Projection", 0.2, 1.5, "How far the form sticks out toward the viewer. Deeper = longer lever: heavier, slower swing. Default comes from the point."),
    ("tail", "Teardrop tail", 0.3, 1.0, "Teardrop only: shorten the point end so the bulb gets wide before the tail gets long."),
    ("psize", "Point shape hold", 0.8, 1.6, "Size of the area around the point that keeps its shape (moves as one piece)."),
    ("dance", "Dance kick", 0.0, 2.0, "Strength of the beat kick in dance mode."),
    ("side", "Dance side", -1.0, 1.0, "-1 left, 0 centre, 1 right: which zones alternate in a twerk. Unset = by position."),
    ("mass", "Mass / heaviness", 0.5, 2.0, "One dial for how heavy the real thing is: heavier swings slower and further. Big chest, butt, belly ≈ 1.3–1.8."),
    ("anchor", "Anchored side", 0.0, 0.8, "How firmly the side where it attaches is held. 0 = moves as a whole; high = the free end does all the moving (butt cheek from the waist, belly from the ribs)."),
    ("slack", "Slack when lifted", 0.0, 1.0, "Lifted against its hang it goes loose and floats; dropping, the skin catches. The lift-then-drop of a hop. Hanging forms are slacker."),
    ("down", "Picture's down", -180.0, 180.0, "Degrees: which way gravity points in the drawing. 0 = straight down the screen; 90 = toward the right edge (figure lying on its side); 180 = drawn upside down."),
    ("contact", "Presses on neighbours", 0.0, 1.0, "Two zones that touch press and squish against each other instead of passing through (1 = solid, 0 = off)."),
]
# Which panel group each dial lives in (Tk tab)
TUNE_GROUPS = [
    ("Weight and spring", ["mass", "f1", "z1", "stiff", "slack", "sens"]),
    ("Ripple (outer layer)", ["couple", "f2", "z2"]),
    ("Shape of the motion", ["travel", "anchor", "pend", "sag", "bulge", "stretch", "contact"]),
    ("Form and point", ["proj", "psize", "tail"]),
    ("Picture", ["down"]),
    ("Dance", ["dance", "side"]),
]
TUNE_KEYS = [t[0] for t in TUNES]

# Playback feel (keep in step with jiggle.cpp)
SENS, GRAV_TAU, GRAV_K, GRAV_Z = 1.4, 0.25, 0.72, 0.50   # gravity pinned to the real one (see jiggle.cpp)
PX_M, ROT_SCALE = 0.000127, 4.0                          # twist: metres per pixel, exaggeration
ECHO_FADE_S, ECHO_SLOW_TAU, ECHO_DAMP_CUT = 9.0, 1.5, 0.955   # bobblehead echo (see jiggle.cpp)

MG_MAGIC = b"MG01"
JIG_MAGIC = "MGJIG1"
BG_RGB = (24, 24, 24)  # COL_BG 0x18C3


# ---- shapes -------------------------------------------------------------------------------------
def _clamp01(t):
    return np.clip(t, 0.0, 1.0) if np is not None and isinstance(t, np.ndarray) else min(1.0, max(0.0, t))


def smooth01(t):
    t = _clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def shape_sdf(shape, u, w, tail=1.0):
    """Signed distance (local units, u right, w up, negative inside). Works on floats or arrays.
    tail (teardrop only, 0.3–1) shortens the point end; the shape stays centred on its extent."""
    xp = np if (np is not None and isinstance(u, np.ndarray)) else None
    sqrt = xp.sqrt if xp else math.sqrt
    if shape == HALF:
        y = w + 0.5
        arc = sqrt(u * u + y * y) - 1.0
        return xp.maximum(arc, -y) if xp else max(arc, -y)
    if shape == TEAR:
        r1, r2, h = 0.55, 0.10, 1.30 * tail
        b = (r1 - r2) / h
        a = math.sqrt(1.0 - b * b)
        px = abs(u) if not xp else xp.abs(u)
        py = w + 0.415 + 0.65 * (tail - 1.0)
        k = -b * px + a * py
        d_lo = sqrt(px * px + py * py) - r1
        d_hi = sqrt(px * px + (py - h) * (py - h)) - r2
        d_mid = a * px + b * py - r1
        if xp:
            return xp.where(k < 0, d_lo, xp.where(k > a * h, d_hi, d_mid))
        return d_lo if k < 0 else (d_hi if k > a * h else d_mid)
    return sqrt(u * u + w * w) - 1.0


def shape_attach(shape, tail=1.0):
    return 0.0, (-0.5 if shape == HALF else (0.95 + 0.65 * (tail - 1.0) if shape == TEAR else 0.9))


def shape_tip_default(shape, tail=1.0):
    return 0.0, (0.15 if shape == HALF else (TIP_W - 0.65 * (tail - 1.0) if shape == TEAR else -0.25))


def shape_free(shape, u, w, has_tip=False, pu=0.0, pw=0.0, tail=1.0):
    if has_tip:
        au, aw = shape_attach(shape, tail)
        nu, nw = pu - au, pw - aw
        L = math.sqrt(nu * nu + nw * nw)
        if L > 0.2:
            along = ((u - au) * nu + (w - aw) * nw) / (L * L * 1.15)
            base = 0.35 if shape == ROUND else 0.0
            return base + (1.0 - base) * smooth01(along)
    if shape == HALF:
        return smooth01((w + 0.5) / 1.0)
    if shape == TEAR:
        return smooth01((0.95 + 0.65 * (tail - 1.0) - w) / (1.9 + 1.3 * (tail - 1.0)))
    return 0.35 + 0.65 * smooth01((0.9 - w) / 1.8)


def shape_outline(shape, npts=64, tail=1.0):
    """Outline points (local units) by bisection along rays — all shapes are convex."""
    pts = []
    for i in range(npts):
        a = 2 * math.pi * i / npts
        du, dw = math.cos(a), math.sin(a)
        lo, hi = 0.0, 1.25
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            if shape_sdf(shape, du * mid, dw * mid, tail) < 0:
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
    name: str = ""                                 # the Studio's layer name

    def tuned(self, key, default):
        if key not in self.tune:
            return default
        _, _, lo, hi, _ = TUNES[TUNE_KEYS.index(key)]
        return min(hi, max(lo, float(self.tune[key])))

    def tip_radius(self):
        return TIP_R * self.tuned("psize", 1.0)

    def tail(self):
        return self.tuned("tail", 1.0) if self.shape == TEAR else 1.0

    def sdf(self, u, w):
        return shape_sdf(self.shape, u, w, self.tail())

    def point_ok(self, u, w):
        """Anywhere inside — droopy apexes sit right at the edge."""
        return self.sdf(u, w) < -self.tip_radius() * 0.5

    def clamp_point(self):
        for _ in range(40):
            if self.point_ok(self.tip_u, self.tip_w):
                return
            self.tip_u *= 0.92; self.tip_w *= 0.92
        if not self.point_ok(self.tip_u, self.tip_w):
            self.tip_u, self.tip_w = shape_tip_default(self.shape, self.tail())

    def point_edge(self):
        if not self.tip:
            return 0.0
        d = -self.sdf(self.tip_u, self.tip_w) / (0.45 * SHAPES[self.shape][2])
        return 1.0 - min(1.0, max(0.0, d))

    def margin(self):
        return MARGIN + 0.35 * self.point_edge()

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
        return self.sdf(u, w) * self.r < pad

    def tip_xy(self):
        return self.from_local(self.tip_u, self.tip_w)

    def outline(self, scale=1.0, npts=64):
        return [self.from_local(u * scale, w * scale) for u, w in shape_outline(self.shape, npts, self.tail())]


def zone_bbox(z: Zone):
    e = BBOX_K * z.margin() * z.r + 2.0
    x0, y0 = max(0, math.floor(z.cx - e)), max(0, math.floor(z.cy - e))
    x1, y1 = min(IMG_W, math.ceil(z.cx + e)), min(IMG_H, math.ceil(z.cy + e))
    return x0, y0, x1 - x0, y1 - y0


def _q8(a):
    """Quantise 0..1 to 0..255 exactly like the firmware (×255 + 0.5, truncated)."""
    return np.floor(np.asarray(a, dtype=np.float64) * 255.0 + 0.5).astype(np.uint8)


def form_t(z: Zone, u, w, scale):
    """0 at the control point → 1 at the outline (scaled): where on the dome's slope a pixel sits."""
    dP = np.sqrt((u - z.tip_u) ** 2 + (w - z.tip_w) ** 2) if isinstance(u, np.ndarray) else math.hypot(u - z.tip_u, w - z.tip_w)
    dE = -shape_sdf(z.shape, u / scale, w / scale, z.tail()) * scale
    if isinstance(u, np.ndarray):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(dE <= 0, 1.0, dP / np.where(dE <= 0, 1.0, dP + dE))
    return 1.0 if dE <= 0 else dP / (dP + dE)


def zone_free(z: Zone, u, w):
    g = shape_free(z.shape, u, w, z.tip, z.tip_u, z.tip_w, z.tail())
    if not z.tip:
        return g
    t = form_t(z, u, w, 1.0)
    h = 1.0 - t * t
    return g * (0.75 + 0.25 * (np.maximum(h, 0) if isinstance(h, np.ndarray) else max(h, 0.0)))


def attach_ramp(z: Zone, u, w):
    """1 − anchor·(1−a)², a = 0 at the attachment → 1 at the apex (or centre) and beyond (weight_at)."""
    anchor = z.tuned("anchor", FORM_MIX if z.tip else 0.0)
    if anchor <= 0:
        return 1.0
    au, aw = shape_attach(z.shape, z.tail())
    pu, pw = (z.tip_u, z.tip_w) if z.tip else (0.0, 0.0)
    hu, hw = pu - au, pw - aw
    hl = math.sqrt(hu * hu + hw * hw)
    if hl < 1e-3:
        return 1.0
    # ramp over at least one radius, so a point near its attachment can't make a weight cliff
    a = np.clip(((u - au) * hu + (w - aw) * hw) / (hl * max(hl, 1.0)), 0.0, 1.0)
    return 1.0 - anchor * (1.0 - a) * (1.0 - a)


def zone_droop(z: Zone, cu, cw, r_eq):
    """−1 firm (points up against its attachment) … 0 typical … +1 hanging (points down)."""
    if not z.tip or r_eq <= 0:
        return 0.0
    down = -(z.tip_w - cw) / r_eq
    return max(-1.0, min(1.0, (down - 0.2) / 0.45))


class ZoneSim:
    """Runtime state of one zone: maps, spring constants, physics (port of prepare/step/stage)."""

    def __init__(self, z: Zone):
        self.z = z
        if z.tip:
            z.clamp_point()
        si, mi = SHAPES[z.shape], MATERIALS[z.material]
        bx, by, bw, bh = zone_bbox(z)
        self.bx, self.by, self.bw, self.bh = bx, by, bw, bh
        self.ok = bw > 0 and bh > 0
        c, s = z.axes()
        area, cu, cw = 0.0, 0.0, 0.0
        self.t = None
        if self.ok and np is not None:
            X, Y = np.meshgrid(np.arange(bx, bx + bw, dtype=np.float64), np.arange(by, by + bh, dtype=np.float64))
            dx, dy = X - z.cx, Y - z.cy
            U = (dx * c + dy * s) / z.r
            Wl = -(-dx * s + dy * c) / z.r
            M_ = z.margin()
            w0 = smooth01(-shape_sdf(z.shape, U / M_, Wl / M_, z.tail()) / (PLATEAU * si[2]))
            w0 = w0 * attach_ramp(z, U, Wl)
            self.w = _q8(w0) / 255.0
            self.g = _q8(zone_free(z, U, Wl)) / 255.0
            self.X, self.Y = X, Y
            inside = z.sdf(U, Wl) < 0
            area = float(inside.sum())
            if area > 0:
                cu, cw = float(U[inside].sum()), float(Wl[inside].sum())
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
        self.tip_gc = zone_free(z, z.tip_u, z.tip_w)
        au, aw = shape_attach(z.shape, z.tail())
        pu, pw = (z.tip_u, z.tip_w) if z.tip else (0.0, 0.0)
        ax, ay = z.from_local(au, aw)
        px, py = z.from_local(pu, pw)
        hx, hy = px - ax, py - ay
        hl = math.hypot(hx, hy)
        self.hang = (hx / hl, hy / hl) if hl > 1e-3 else (0.0, 1.0)

        # what the point implies about the form (defaults; sliders win)
        r_eq = droop = 0.0
        if area > 0:
            cu /= area; cw /= area
            r_eq = math.sqrt(area / math.pi) / z.r
            droop = zone_droop(z, cu, cw, r_eq)
        self.droop = droop
        form_k, axis_u, tilt, elong = 1.0, (0.0, 0.0), 0.0, 0.0
        T = z.tuned
        if z.tip and area > 0:
            du, dw = z.tip_u - cu, z.tip_w - cw
            proj = T("proj", 0.6 - 0.25 * droop)
            lever = T("proj", 0.6)
            dl = math.hypot(du, dw)
            elong = dl / r_eq
            L = math.sqrt(dl * dl + (lever * r_eq) ** 2)
            form_k = min(1.3, max(0.9, 1.0 / math.sqrt(L / (0.6 * r_eq))))
            L = math.sqrt(dl * dl + (proj * r_eq) ** 2)
            if dl > 1e-4:
                x0, y0 = z.from_local(cu, cw)
                x1, y1 = z.from_local(z.tip_u, z.tip_w)
                h_ = math.hypot(x1 - x0, y1 - y0)
                axis_u = ((x1 - x0) / h_, (y1 - y0) / h_)
                tilt = dl / L
        self.elong = elong

        fk = form_k * (1.0 - (0.06 if droop > 0 else 0.15) * droop)
        self.mass = T("mass", 1.0)
        fk /= math.sqrt(self.mass)
        size_k = min(1.35, max(0.65, math.sqrt(50.0 / z.r))) * fk
        self.form_travel = min(1.6, max(0.8, 1.0 / fk))
        self.w1 = 2 * math.pi * T("f1", mi[2]) * si[3] * size_k
        self.w2 = 2 * math.pi * T("f2", mi[4]) * si[3] * size_k
        self.wt = self.w1 * 2.3
        self.zt = 0.38
        self.z1, self.z2 = T("z1", mi[3] - 0.03 * droop), T("z2", mi[5])
        self.stiffen, self.sens = T("stiff", mi[6] * (1.0 - 0.3 * droop)), T("sens", mi[7])
        ripple = min(1.6, 1.0 + 0.3 * droop + 0.4 * min(1.0, elong))
        self.couple = T("couple", mi[8] * ripple)
        self.sag = T("sag", 1.0 + 0.5 * droop)
        self.pend = T("pend", min(0.45, PENDULUM * (1.0 + 0.6 * droop)))
        self.dance = T("dance", 1.0)
        self.slack = T("slack", (0.25 + 0.25 * max(droop, 0.0)) if z.tip else 0.2)
        self.contact = T("contact", 1.0)
        a = math.radians(T("down", 0.0))
        self.down_s, self.down_c = math.sin(a), math.cos(a)
        self.pressed = False
        self.cax = self.cay = 0.0
        self.damp_k = 1.0
        self.side = int(round(T("side", 0))) if "side" in z.tune else None
        self.lim_px = 0.22 * si[2] * z.r * T("travel", self.form_travel)
        self.s_lim = 0.35 / (1.0 + 1.5 / (PLATEAU * si[2])) * T("bulge", 1.0)
        self.st_max = STRETCH_MAX * T("stretch", 1.0)
        self.axis = (AXIS_K * tilt * self.lim_px * axis_u[0], AXIS_K * tilt * self.lim_px * axis_u[1])
        self.tip_travel = TIP_TRAVEL * z.r
        self.rest()
        k = self._fold_scale()
        if k < 1.0:
            self.lim_px *= k; self.s_lim *= k; self.st_max *= k
            self.axis = (self.axis[0] * k, self.axis[1] * k); self.tip_travel *= k
        self.fold_k = k

    def _fold_scale(self):
        """Same fold guard as the firmware: largest k ≤ 1 keeping det(I − kJ) ≥ FOLD_MARGIN."""
        if not self.ok or np is None or self.bw < 5 or self.bh < 5:
            return 1.0
        st = 1 if (self.bw <= 96 and self.bh <= 96) else 2       # small zones: every pixel
        margin = FOLD_MARGIN if st == 1 else FOLD_MARGIN + 0.04
        w, g = self.w[::st, ::st], self.g[::st, ::st]
        t = self.t[::st, ::st] if self.t is not None else None
        RX, RY = self.X[::st, ::st] - self.z.cx, self.Y[::st, ::st] - self.z.cy
        kmin = 1.0
        for d in range(8):
            a = 0.7853982 * d
            ca, sa = math.cos(a), math.sin(a)
            for qs, ss, ts in ((q_, s_, t_) for q_ in (-1, 1) for s_ in (-1, 1) for t_ in ((-1, 1) if self.z.tip else (-1,))):
                if True:
                    self.dx, self.dy, self.qx, self.qy = ca, sa, qs * Q_MAX * ca, qs * Q_MAX * sa
                    self.s, self.tx, self.ty = float(ss), ts * ca, ts * sa
                    b0x, b0y, b1x, b1y, M, ot = self.stage()
                    Bx = b0x + g * b1x + M[0] * RX + M[1] * RY
                    By = b0y + g * b1y + M[2] * RX + M[3] * RY
                    if t is not None and ot is not None:
                        Bx = Bx + t * (ot[0] - Bx); By = By + t * (ot[1] - By)
                    Ox, Oy = w * Bx, w * By
                    jxx = (Ox[1:-1, 2:] - Ox[1:-1, :-2]) / (2 * st); jyx = (Oy[1:-1, 2:] - Oy[1:-1, :-2]) / (2 * st)
                    jxy = (Ox[2:, 1:-1] - Ox[:-2, 1:-1]) / (2 * st); jyy = (Oy[2:, 1:-1] - Oy[:-2, 1:-1]) / (2 * st)
                    tr, de = jxx + jyy, jxx * jyy - jxy * jyx
                    c0 = 1.0 - margin
                    kk = np.full(tr.shape, np.inf)
                    lin = np.abs(de) < 1e-9
                    kk = np.where(lin & (tr > 1e-9), c0 / np.where(tr > 1e-9, tr, 1.0), kk)
                    disc = tr * tr - 4.0 * de * c0
                    ok = (~lin) & (disc >= 0)
                    sq = np.sqrt(np.where(ok, disc, 0.0))
                    den = np.where(ok, 2 * de, 1.0)
                    for kr in ((tr - sq) / den, (tr + sq) / den):
                        kk = np.where(ok & (kr > 0) & (kr < kk), kr, kk)
                    if kk.size:
                        kmin = min(kmin, float(kk.min()))
        self.rest()
        return kmin

    def rest(self):
        self.dx = self.dy = self.vx = self.vy = 0.0
        self.qx = self.qy = self.qvx = self.qvy = 0.0
        self.s = self.vs = 0.0
        self.tx = self.ty = self.tvx = self.tvy = 0.0
        self.cax = self.cay = 0.0
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
        if self.slack > 0:
            a = self.dx * self.hang[0] + self.dy * self.hang[1]
            k1 *= (1.0 + 0.8 * self.slack * a) if a >= 0 else (1.0 + 0.5 * self.slack * a)
        sp2 = self.vx * self.vx + self.vy * self.vy
        c1 = 2.0 * (self.z1 + REST_DAMP / (1.0 + 4.0 * sp2)) * self.w1 * self.damp_k
        ax = k1 * (tgx - self.dx) - c1 * self.vx
        ay = k1 * (tgy - self.dy) - c1 * self.vy
        self.vx += ax * h; self.vy += ay * h
        self.dx += self.vx * h; self.dy += self.vy * h
        self.dx, self.dy, self.vx, self.vy = self._clamp(self.dx, self.dy, self.vx, self.vy, 1.0)

        press = math.hypot(self.cax, self.cay) / (self.w1 * self.w1)   # pressed: swells toward the viewer
        tgs += 0.5 * min(press, 1.0)
        ax += self.cax; ay += self.cay; self.cax = self.cay = 0.0   # net push incl. a pressing neighbour
        drive = self.couple / (self.w1 * self.w1)
        q2 = self.qx * self.qx + self.qy * self.qy
        k2 = self.w2 * self.w2 * (1.0 + 3.0 * q2)
        qs2 = self.qvx * self.qvx + self.qvy * self.qvy
        c2 = 2.0 * (self.z2 + REST_DAMP / (1.0 + 4.0 * qs2)) * self.w2 * self.damp_k
        qax = -k2 * self.qx - c2 * self.qvx - drive * self.w2 * self.w2 * ax
        qay = -k2 * self.qy - c2 * self.qvy - drive * self.w2 * self.w2 * ay
        self.qvx += qax * h; self.qvy += qay * h
        self.qx += self.qvx * h; self.qy += self.qvy * h
        self.qx, self.qy, self.qvx, self.qvy = self._clamp(self.qx, self.qy, self.qvx, self.qvy, Q_MAX)

        ws = self.w1 * 1.3
        cs = 2.0 * (self.z1 + REST_DAMP / (1.0 + 4.0 * self.vs * self.vs)) * ws * self.damp_k
        self.vs += (ws * ws * (1.0 + self.stiffen * self.s * self.s) * (tgs - self.s) - cs * self.vs) * h
        self.s += self.vs * h
        if self.s > 1.0:
            self.s = 1.0; self.vs = min(self.vs, 0.0)
        if self.s < -1.0:
            self.s = -1.0; self.vs = max(self.vs, 0.0)

        if self.z.tip:
            lim_ratio = self.lim_px / max(self.tip_travel, 1e-6)
            bx, by = (ax + self.tip_gc * qax) * lim_ratio, (ay + self.tip_gc * qay) * lim_ratio
            ct = 2.0 * self.zt * self.wt
            tax = -self.wt * self.wt * self.tx - ct * self.tvx - 0.6 * bx
            tay = -self.wt * self.wt * self.ty - ct * self.tvy - 0.6 * by
            self.tvx += tax * h; self.tvy += tay * h
            self.tx += self.tvx * h; self.ty += self.tvy * h
            self.tx, self.ty, self.tvx, self.tvy = self._clamp(self.tx, self.ty, self.tvx, self.tvy, 1.0)

    def still(self, tx, ty, ts):
        e, v = 0.015, 0.06
        return (abs(self.dx - tx * self.sens) < e and abs(self.dy - ty * self.sens) < e and abs(self.s - ts * self.sens) < e
                and abs(self.qx) < e and abs(self.qy) < e and abs(self.tx) < 0.02 and abs(self.ty) < 0.02
                and abs(self.vx) < v and abs(self.vy) < v and abs(self.vs) < v and abs(self.qvx) < v and abs(self.qvy) < v
                and abs(self.tvx) < 0.1 and abs(self.tvy) < 0.1)

    def still_pressed(self):
        e, v = 0.015, 0.06
        return (self.pressed and abs(self.qx) < e and abs(self.qy) < e and abs(self.tx) < 0.02 and abs(self.ty) < 0.02
                and abs(self.vx) < v and abs(self.vy) < v and abs(self.vs) < v and abs(self.qvx) < v and abs(self.qvy) < v
                and abs(self.tvx) < 0.1 and abs(self.tvy) < 0.1)

    def gravity_target(self, dn, kxy, kz):
        """Real gravity vs. the picture's own down (firmware gravity_target)."""
        s, c = self.down_s, self.down_c
        wx, wy = dn[1] * s + dn[0] * c, -dn[0] * s + dn[1] * c
        return kxy * (wx - s), kxy * (wy - c), kz * dn[2]

    def freeze(self):
        self.vx = self.vy = self.qx = self.qy = self.qvx = self.qvy = self.vs = 0.0
        self.tx = self.ty = self.tvx = self.tvy = 0.0
        self.cax = self.cay = 0.0
        self.awake = False

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
            b0x += -sc * (self.tcx - z.cx) + self.s * self.axis[0]
            b0y += -sc * (self.tcy - z.cy) + self.s * self.axis[1]
        m = math.hypot(self.dx, self.dy)
        e = self.st_max * min(m, 1.0)
        nx, ny = (self.dx / m, self.dy / m) if m > 1e-4 else (0.0, 0.0)
        iso, an = sc - 0.5 * e, 1.5 * e
        M = (iso + an * nx * nx, an * nx * ny, an * nx * ny, iso + an * ny * ny)
        ot = None
        if z.tip:
            rx, ry = self.tcx - z.cx, self.tcy - z.cy
            ot = (b0x + self.tip_gc * b1x + M[0] * rx + M[1] * ry + self.tx * self.tip_travel,
                  b0y + self.tip_gc * b1y + M[2] * rx + M[3] * ry + self.ty * self.tip_travel)
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


# ---- motion echo (port of MotionEcho in jiggle_dance.h) -------------------------------------------
class MotionEcho:
    """Bobblehead echo: sway it back and forth a couple of times and stop, and the forms keep swaying on
    their own springs, dying out slowly. Only detects the sway; strength() (1 → 0 over fade_s) eases the
    damping off. Same logic as the firmware."""
    FS, N, LAG_MIN, LAG_MAX = 50, 300, 10, 125
    MOVE, CONF, SWAY_HOLD = 0.035, 0.5, 0.6

    def __init__(self, fade_s=9.0):
        self.fade_s = fade_s
        self.h = [[0.0] * 4 for _ in range(self.N)]
        self.head = self.filled = self.since = self.good = 0
        self.t = self.last_t = self.acc_w = self.e = 0.0
        self.acc = [0.0] * 4; self.gprev = [0.0] * 3
        self.period_s = 1.0; self.moving_t = self.still_t = self.fade_t = self.sustain = 0.0
        self.swayed = self.fading = False
        self.peak = 0.0

    def at(self, ago):
        return self.h[(self.head - 1 - ago + 2 * self.N) % self.N]

    def update(self, dt, dyn, gdev, w):
        self.t += dt
        v2 = 0.0
        for k in range(3):
            rate = (gdev[k] - self.gprev[k]) / dt if dt > 1e-6 else 0.0
            self.gprev[k] = gdev[k]
            v2 += dyn[k] * dyn[k] + (0.05 * rate) ** 2
        v2 += (w * 0.05) ** 2
        self.e += (v2 - self.e) * (1.0 - math.exp(-dt / 0.15))
        moving = math.sqrt(self.e) > self.MOVE
        self.peak = self.e if self.e > self.peak else self.peak * math.exp(-dt / 2.0)
        swaying = math.sqrt(self.e) > max(self.MOVE, 0.3 * math.sqrt(self.peak))
        for k in range(3):
            self.acc[k] += (dyn[k] + gdev[k]) * dt
        self.acc[3] += w * dt; self.acc_w += dt
        while self.t - self.last_t >= 1.0 / self.FS:
            self.last_t += 1.0 / self.FS
            kk = 1.0 / self.acc_w if self.acc_w > 1e-6 else 0.0
            self.h[self.head] = [a * kk for a in self.acc]
            self.acc = [0.0] * 4; self.acc_w = 0.0
            self.head = (self.head + 1) % self.N
            self.filled = min(self.N, self.filled + 1)
            self.since += 1
            if self.since >= self.FS // 10:
                self.since = 0; self._analyse(moving)
        if moving:
            self.moving_t += dt
        if swaying:
            self.still_t = 0.0; self.fading = False
            target = self.SWAY_HOLD if self.swayed else 0.0
            self.sustain += (target - self.sustain) * (1.0 - math.exp(-dt / 0.3))
        else:
            self.still_t += dt
            if self.swayed and not self.fading:
                self.fading, self.fade_t = True, 0.0
            if not self.swayed and not moving and self.still_t > 0.6:
                self.moving_t = 0.0
            if self.fading:
                self.fade_t += dt
                u = self.fade_t / self.fade_s
                self.sustain = 1.0 - u ** 3 if u < 1.0 else 0.0
                if u >= 1.0:
                    self.fading = False; self.swayed = False

    def strength(self):
        return self.sustain

    @classmethod
    def _periodicity(cls, x0):
        x = x0 - x0.mean(); n = len(x)
        lmax = min(n // 2, cls.LAG_MAX)
        if lmax <= cls.LAG_MIN:
            return 0.0, 0
        r = np.zeros(cls.LAG_MAX + 2); dip = -1
        for L in range(1, lmax + 2):
            a, b = x[:n - L], x[L:]
            sxx, syy = float(a @ a), float(b @ b)
            r[L] = float(a @ b) / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0
            if dip < 0 and r[L] < -0.1:
                dip = L
        if dip < 0:
            return 0.0, 0
        best, pick = 0.0, 0
        for L in range(max(dip, cls.LAG_MIN), lmax + 1):
            if r[L] >= r[L - 1] and r[L] >= r[L + 1] and r[L] > best:
                best, pick = r[L], L
        return (best if pick else 0.0), pick

    def _analyse(self, moving):
        n = self.filled
        if n < 2 * self.LAG_MIN + 10 or not moving:
            return
        S = np.array([self.at(n - 1 - i) for i in range(n)])
        D = S[:, :3] - S[:, :3].mean(0)
        C = D.T @ D
        ax = np.zeros(3); ax[int(np.argmax(np.diag(C)))] = 1.0
        for _ in range(10):
            nx = C @ ax; l = float(np.linalg.norm(nx))
            if l < 1e-12:
                break
            ax = nx / l
        x = D @ ax + 0.05 * S[:, 3]
        nw = min(n, 3 * self.FS)
        c1, p1 = self._periodicity(x[n - nw:]); c2, p2 = self._periodicity(x)
        conf, pick = (c1, p1) if c1 >= c2 else (c2, p2)
        if pick and conf >= self.CONF and self.moving_t > 1.5 * pick / self.FS:
            self.good += 1
            if self.good >= 2:
                self.swayed = True; self.period_s = pick / self.FS
        else:
            self.good = 0
            if conf < 0.3:
                self.swayed = False


# ---- contact between neighbouring zones (port of contact_gap / contact_pass) ----------------------
def zone_extent(z: Zone, nx, ny):
    c, s_ = z.axes()
    u, w = nx * c + ny * s_, -(-nx * s_ + ny * c)
    lo, hi = 0.0, 2.0
    for _ in range(18):
        m = 0.5 * (lo + hi)
        if z.sdf(u * m, w * m) < 0:
            lo = m
        else:
            hi = m
    return lo * z.r


def contact_gap(a: Zone, b: Zone):
    nx, ny = b.cx - a.cx, b.cy - a.cy
    d = math.hypot(nx, ny)
    if d < 1e-3:
        return -1e9
    nx /= d; ny /= d
    return d - zone_extent(a, nx, ny) - zone_extent(b, -nx, -ny)


def contact_gaps(zones):
    n = len(zones)
    return {(i, j): contact_gap(zones[i], zones[j]) for i in range(n) for j in range(i + 1, n)}


def contact_pass(zs, gaps, h):
    n = len(zs)                                          # (pressed is cleared once per frame)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = zs[i], zs[j]
            if not a.ok or not b.ok or (not a.awake and not b.awake):
                continue
            k = 0.5 * (a.contact + b.contact)
            g = gaps[(i, j)]
            if k <= 0 or g < -0.6 * min(a.z.r, b.z.r) or g > a.lim_px + b.lim_px:   # deep overlap = one mass
                continue
            nx, ny = b.z.cx - a.z.cx, b.z.cy - a.z.cy
            l = math.hypot(nx, ny); nx /= l; ny /= l
            allow = max(g, 0.0)
            close = (a.dx * a.lim_px - b.dx * b.lim_px) * nx + (a.dy * a.lim_px - b.dy * b.lim_px) * ny
            p = close - allow
            if p <= 0:
                continue
            ma, mb = a.z.r * a.z.r * a.mass, b.z.r * b.z.r * b.mass
            wa, wb = mb / (ma + mb), ma / (ma + mb)
            c = k * p
            a.dx -= c * wa * nx / a.lim_px; a.dy -= c * wa * ny / a.lim_px
            b.dx += c * wb * nx / b.lim_px; b.dy += c * wb * ny / b.lim_px
            vc = (a.vx * a.lim_px - b.vx * b.lim_px) * nx + (a.vy * a.lim_px - b.vy * b.lim_px) * ny
            if vc > 0:
                dv = k * (1.0 + 0.15 * smooth01((vc - 10.0) / 30.0)) * vc
                ga, gb = dv * wa / a.lim_px, dv * wb / b.lim_px
                a.vx -= ga * nx; a.vy -= ga * ny; b.vx += gb * nx; b.vy += gb * ny
                a.cax -= ga * nx / h; a.cay -= ga * ny / h
                b.cax += gb * nx / h; b.cay += gb * ny / h
            a.awake = b.awake = True
            a.pressed = b.pressed = True


# ---- playback (port of play_tick's motion handling) ---------------------------------------------
_SWAY_CACHE = {}


def sway_protos(img, sways, body=None):
    """The prepared hair / cloth / pendants (fills behind them made, edges matted), as the Core2 builds
    them from the saved lines: on what they rest (mg_sway.rest_body), never filled from one another.
    Cached: the same ones again come back at once."""
    import mg_sway as SW
    import hashlib
    sways = [s for s in sways if s.ok()][:SW.SWAY_MAX]
    rb = SW.rest_body(body, [s.mask for s in sways])
    sdf = None
    out = []
    for i, sw in enumerate(sways):
        others = np.zeros(sw.mask.shape, bool)
        for j, o in enumerate(sways):
            if j != i:
                others |= o.mask
        key = (sw.kind, sw.mask.tobytes(), repr(sw.guides), repr(sorted(sw.prm.items())), others.tobytes(),
               None if rb is None else rb.tobytes(), img.tobytes() if img is not None else b"",
               repr(sw.arm) + ("back" if sw.back else ""), None if sw.under is None else np.asarray(sw.under).tobytes())
        hk = hashlib.sha1(b"".join(k if isinstance(k, bytes) else k.encode() for k in key if k is not None)).hexdigest()
        proto = _SWAY_CACHE.get(hk)
        if proto is None:
            if sdf is None and rb is not None:
                sdf = SW.body_sdf(rb)
            proto = SW.SwaySim(sw, img, rb, sdf=sdf, forbid=others if others.any() else None)
            _SWAY_CACHE[hk] = proto
            while len(_SWAY_CACHE) > 8:
                _SWAY_CACHE.pop(next(iter(_SWAY_CACHE)))
        out.append(proto)
    return out


class Sim:
    """Drives zones like the Core2 does. Feed screen-space accelerometer readings (g, gravity
    included: the Core2 held upright reads (0, -1, 0)), plus beats and pokes."""

    def __init__(self, zones, image_rgb=None, waters=(), par=None, hangs=()):
        self.zs = [ZoneSim(z) for z in zones]
        self.ws = [WaterSim(w) for w in waters if w is not None]
        import mg_hang as G
        self.hs = [G.HangSim(h, image_rgb) for h in hangs if h is not None and h.ok() and image_rgb is not None]
        self.img = image_rgb
        self.depth = None
        self.ss = []                            # swaying things (mg_sway): add_sways
        self.view_target = (0.0, 0.0)          # (old) a target view; the Core2 uses gyro rates (self.gyro)
        self.gyro = (0.0, 0.0)                 # deg/s: top edge away (+x), right edge away (+y), like the IMU
        if par is not None and par.any() and image_rgb is not None:
            import mg_depth as D
            self.depth = D.DepthPreview(par, image_rgb)
            self.view = D.GyroView(par)
            self.v = (0.0, 0.0)
        self.base = None
        self.slow = None
        self.echo = MotionEcho(ECHO_FADE_S)
        self.dn_slow = None

    def add_sways(self, sways, body=None):
        """Swaying things over the photo (prepared once per mask / guides: their plates take a moment)."""
        import copy
        self.ss = [copy.deepcopy(p) for p in sway_protos(self.img, sways, body)]

    def step(self, dt, accel=(0.0, -1.0, 0.0), dance=0.0, twist=(0.0, 0.0)):
        """Same as the Core2: gravity pinned to the real one, jolts on top. `dance` is unused here
        (dance only adds beat kicks); kept for callers."""
        if self.depth is not None:
            self.v = self.view.step(dt, self.gyro)
        a = list(accel)
        if self.base is None:
            self.base = a[:]
            self.hold = [[0.0, 0.0, 0.0] for _ in self.zs]
        k = 1 - math.exp(-dt / GRAV_TAU)
        for i in range(3):
            self.base[i] += (a[i] - self.base[i]) * k
        gl = math.sqrt(sum(v * v for v in self.base)) or 1.0
        if gl < 0.3:
            gl = 1.0
        dn_live = [-v / gl for v in self.base]
        cl = lambda v: max(-1.0, min(1.0, v))
        dyn = [a[i] - self.base[i] for i in range(3)]
        # motion echo (firmware play_tick): a repeating motion carries on after you stop
        if self.dn_slow is None:
            l0 = math.sqrt(sum(v * v for v in a)) or 1.0
            self.dn_slow = [-v / (l0 if l0 >= 0.3 else 1.0) for v in a]
        ks = 1.0 - math.exp(-dt / ECHO_SLOW_TAU)
        gdev = []
        for i in range(3):
            self.dn_slow[i] += (dn_live[i] - self.dn_slow[i]) * ks
            gdev.append(dn_live[i] - self.dn_slow[i])
        self.echo.update(dt, dyn, gdev, twist[1])
        for ws in self.ws:                      # water: real gravity + your jolts
            ws.step(dt, dn_live, dyn)
        for hs in self.hs:                      # hanging things: the same
            hs.step(dt, dn_live, dyn)
        sdk = 1.0 - ECHO_DAMP_CUT * self.echo.strength()
        push = [(z.z.cx + z.dx, z.z.cy + z.dy, z.z.r, z.vx, z.vy) for z in self.zs if z.ok]
        for ss in self.ss:                      # swaying things: real "down" (screen axes), jolts, the echo
            if getattr(ss, "ok", False) and ss.s.kind == "cloth":
                ss.pushers = push               # (forms jiggling under a skirt carry it along)
            ss.step(dt, tuple(dn_live), tuple(dyn), sdk)
        for z in self.zs:                       # swayed and stopped: keeps swaying, dying out slowly
            z.damp_k = 1.0 - ECHO_DAMP_CUT * self.echo.strength()
        dn = dn_live
        jx, jy, js = (-SENS * dyn[i] for i in range(3))
        tg = []
        alpha, wv = twist
        w2 = wv * wv
        for i, z in enumerate(self.zs):
            # a twist swings the zone's spot around: tangential α·r⊥ and centripetal −ω²·r (firmware)
            rx, ry = (z.z.cx - IMG_W * 0.5) * PX_M * ROT_SCALE, (z.z.cy - IMG_H * 0.5) * PX_M * ROT_SCALE
            arx, ary = (alpha * -ry - w2 * rx) / 9.81, (alpha * rx - w2 * ry) / 9.81
            gx, gy, gs = z.gravity_target(dn, GRAV_K, GRAV_Z)
            m = math.hypot(gx, gy) * z.sag * z.sens   # steady hang inside reachable travel (firmware)
            if m > 0.95:
                gx, gy = gx * 0.95 / m, gy * 0.95 / m
            mz = abs(gs) * z.sag * z.sens
            if mz > 0.95:
                gs *= 0.95 / mz
            tx, ty, ts = cl(jx - SENS * arx + z.sag * gx), cl(jy - SENS * ary + z.sag * gy), cl(js + z.sag * gs)
            tg.append((tx, ty, ts))
            h = self.hold[i]
            if not z.awake and max(abs(tx - h[0]), abs(ty - h[1]), abs(ts - h[2])) > 0.03:
                z.awake = True
        n = max(1, math.ceil(dt / 0.004))
        gaps = contact_gaps([z.z for z in self.zs]) if len(self.zs) > 1 else None
        for z in self.zs:
            z.pressed = False
        for _ in range(n):
            for i, z in enumerate(self.zs):
                if z.ok and z.awake:
                    z.step(dt / n, *tg[i])
            if gaps is not None:
                contact_pass(self.zs, gaps, dt / n)
        for i, z in enumerate(self.zs):
            tx, ty, ts = tg[i]
            if z.awake and self.echo.strength() < 0.02 and (z.still(tx, ty, ts) or z.still_pressed()):
                home = abs(tx) < 0.004 and abs(ty) < 0.004 and abs(ts) < 0.004
                if home and (not z.pressed or (abs(z.dx) < 0.02 and abs(z.dy) < 0.02 and abs(z.s) < 0.02)):
                    z.rest()
                else:
                    z.freeze()
                self.hold[i] = [tx, ty, ts]

    def beat(self, amp, sign):
        for ss in self.ss:
            ss.kick(amp, sign)
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
        for hs in self.hs:                      # a tap on hair / cloth / a charm: a flick
            if hs.poke(x, y):
                return True
        for ss in self.ss:
            if ss.poke(x, y):
                return True
        for ws in self.ws:                      # a tap on water: a ripple
            if ws.contains(x, y):
                ws.poke(x, y, 160.0, 2)
                return True
        return False

    def awake(self):
        return any(z.awake for z in self.zs) or any(h.awake for h in self.hs) or any(s.awake for s in self.ss)

    def offsets(self):
        OX = np.zeros((IMG_H, IMG_W)); OY = np.zeros((IMG_H, IMG_W))
        for z in self.zs:
            z.add_offsets(OX, OY)
        for ws in self.ws:                      # a zone never drags water: it's its own layer (firmware)
            m = getattr(ws.w, "mask", None)
            if m is not None:
                OX[m] = 0.0; OY[m] = 0.0
        return OX, OY

    def render(self):
        """Frame as uint8 RGB (H, W, 3): water first, zones drawn over it, then depth (like the Core2)."""
        OX, OY = self.offsets()
        if not self.ws and not self.hs and not self.ss:
            frame = warp(self.img, OX, OY)
        else:
            layer = self.img.copy()             # the photo with the water drawn in…
            for ws in self.ws:
                ws.render_into(layer, self.img)
            for hs in self.hs:                  # …the hanging things over it…
                hs.render(layer)
            for ss in self.ss:                  # …hair / cloth / pendants…
                ss.render(layer)
            frame = warp(layer, OX, OY)         # …and the zones warp that (effects stack)
        if self.depth is not None:
            frame = self.depth.render(frame, self.v, self.view.sharp)
        return frame


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


# ---- water (port of jiggle_water.h) ---------------------------------------------------------------
W_STILL, W_RIVER, W_OCEAN = 0, 1, 2
WMODES = [("still", "Still water"), ("river", "River"), ("ocean", "Ocean waves")]
WATER_MAX, WLINE_MAX, WPTS_MAX, WCELL = 2, 3, 24, 4
# key, label, lo, hi, default, help
WPARAMS = [
    ("amb", "Ambient motion", 0.0, 3.0, 1.0, "The water's gentle, always-on movement."),
    ("ripple", "Ripple strength", 0.0, 3.0, 1.0, "How much taps, rain and shakes ring the surface."),
    ("slosh", "Tilt / shake response", 0.0, 3.0, 1.0, "How much the body of water leans and sloshes when you tilt or shake the Core2."),
    ("rain", "Rain", 0.0, 1.0, 0.0, "Rain drops ringing the surface (on/off)."),
    ("rate", "Rain amount", 0.1, 4.0, 1.0, "How many drops."),
    ("edge", "Edge hold (px)", 0.0, 16.0, 3.0, "Motion fades to nothing over this many pixels inside the mask, so the banks stay put."),
    ("horizon", "Horizon (y px)", -1.0, 239.0, -1.0, "Waves get smaller toward this line (−1 = the top of the water)."),
    ("speed", "Flow / wave speed", 0.1, 4.0, 1.0, "River current, or how fast ocean swells roll in."),
    ("size", "Wave size", 0.1, 4.0, 1.0, "Ocean swell height."),
    ("spacing", "Wave spacing (px)", 8.0, 160.0, 48.0, "Distance between ocean swells (close up)."),
    ("sets", "Bigger sets", 0.0, 1.0, 0.5, "How much every few waves come in bigger."),
    ("glint", "Sun glints", 0.0, 1.0, 0.4, "Bright sparkles where ripples catch the light."),
    ("light", "Painted wave light", 0.0, 1.0, 0.0, "Paints light and shade onto swells and ripples. For flat or stylised art with little texture; photos look best at 0 (their own texture moves)."),
    ("foam", "Painted shore foam", 0.0, 1.0, 0.0, "Paints foam where waves break at the shore line. For flat or stylised art; leave at 0 for photos."),
]
WPARAM_KEYS = [p_[0] for p_ in WPARAMS]
W_OCEAN_FEEL = 0.15
W_SLOSH_PX, W_RIP_K, W_AMB_PX, W_RIVER_PX, W_RIVER_T, W_OCEAN_A, W_OCEAN_T = 7.0, 0.035, 0.9, 14.0, 1.6, 4.2, 2.4
W_OCEAN_SHADE, W_RIP_SHADE = 0.16, 0.0035                # light on swells and ripples (see jiggle_water.h)
W_SIM_DT = 0.04                                          # surface step: a fixed 25 Hz, drawn blended


@dataclass
class Water:
    mode: int = W_STILL
    prm: dict = field(default_factory=dict)          # dial overrides (see WPARAMS)
    lines: list = field(default_factory=list)        # flow / shore curves: [[(x, y), ...], ...]
    mask: object = None                              # (240, 320) bool
    grid: list | None = None                          # perspective grid corners [(x, y)×4]: near-L, near-R, far-R, far-L
    name: str = ""                                    # the Studio's layer name

    def __post_init__(self):
        if self.mask is None and np is not None:
            self.mask = np.zeros((IMG_H, IMG_W), bool)

    def get(self, key):
        _, _, lo, hi, d, _ = WPARAMS[WPARAM_KEYS.index(key)]
        return min(hi, max(lo, float(self.prm[key]))) if key in self.prm else d

    def copy(self):
        return Water(self.mode, dict(self.prm), [list(l) for l in self.lines], self.mask.copy(),
                     None if self.grid is None else [tuple(p) for p in self.grid], self.name)


def mask_brush(mask, cx, cy, r, v=True):
    for dy in range(-r, r + 1):
        y = cy + dy
        if 0 <= y < IMG_H:
            hw = int(math.sqrt(max(0, r * r - dy * dy)))
            mask[y, max(0, cx - hw):max(0, min(IMG_W, cx + hw + 1))] = v


def mask_poly(mask, pts, v=True):
    """Filled polygon (even-odd) at pixel centres — same rule as the firmware."""
    n = len(pts)
    if n < 3:
        return
    ys = [p_[1] for p_ in pts]
    for y in range(max(0, min(ys)), min(IMG_H - 1, max(ys)) + 1):
        yc = y + 0.5; xs = []
        for i in range(n):
            (xi, yi), (xj, yj) = pts[i], pts[i - 1]
            if (yi > yc) != (yj > yc):
                xs.append(xi + (yc - yi) * (xj - xi) / (yj - yi))
        xs.sort()
        for a in range(0, len(xs) - 1, 2):
            x0, x1 = max(0, math.ceil(xs[a] - 0.5)), min(IMG_W, math.ceil(xs[a + 1] - 0.5))
            if x1 > x0:
                mask[y, x0:x1] = v


def water_curve(pts):
    """Smooth curve through the points (Catmull-Rom), like the firmware: arrays (x, y)."""
    n = len(pts)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    if n == 1:
        return np.array([float(pts[0][0])]), np.array([float(pts[0][1])])
    ox, oy = [], []
    for i in range(n - 1):
        p0, p1, p2, p3 = pts[max(0, i - 1)], pts[i], pts[i + 1], pts[min(n - 1, i + 2)]
        for s_ in range(8):
            t = s_ / 8; t2 = t * t; t3 = t2 * t
            for k, out in ((0, ox), (1, oy)):
                a0, a1, a2, a3 = p0[k], p1[k], p2[k], p3[k]
                out.append(0.5 * (2 * a1 + (-a0 + a2) * t + (2 * a0 - 5 * a1 + 4 * a2 - a3) * t2 + (-a0 + 3 * a1 - 3 * a2 + a3) * t3))
    ox.append(float(pts[-1][0])); oy.append(float(pts[-1][1]))
    return np.array(ox), np.array(oy)


def curve_nearest(cx, cy, PX, PY):
    """Vectorised nearest point on a sampled curve: dist, tangent (tx, ty), unit toward it (nx, ny)."""
    shape = PX.shape
    px, py = PX.ravel(), PY.ravel()
    best = np.full(px.shape, 1e9); TX = np.ones(px.shape); TY = np.zeros(px.shape); NX = np.zeros(px.shape); NY = np.zeros(px.shape)
    if len(cx) == 1:
        dx, dy = cx[0] - px, cy[0] - py; d = np.hypot(dx, dy)
        with np.errstate(divide="ignore", invalid="ignore"):
            return d.reshape(shape), TX.reshape(shape), TY.reshape(shape), np.where(d > 1e-6, dx / d, 0).reshape(shape), np.where(d > 1e-6, dy / d, 0).reshape(shape)
    for i in range(len(cx) - 1):
        ax, ay, bx, by = cx[i], cy[i], cx[i + 1] - cx[i], cy[i + 1] - cy[i]
        l2 = bx * bx + by * by
        if l2 < 1e-6:
            continue
        u = np.clip(((px - ax) * bx + (py - ay) * by) / l2, 0, 1)
        qx, qy = ax + u * bx - px, ay + u * by - py
        d = np.hypot(qx, qy)
        m = d < best
        l = math.sqrt(l2)
        best = np.where(m, d, best); TX = np.where(m, bx / l, TX); TY = np.where(m, by / l, TY)
        with np.errstate(divide="ignore", invalid="ignore"):
            NX = np.where(m, np.where(d > 1e-6, qx / d, -by / l), NX); NY = np.where(m, np.where(d > 1e-6, qy / d, bx / l), NY)
    return best.reshape(shape), TX.reshape(shape), TY.reshape(shape), NX.reshape(shape), NY.reshape(shape)


class WaterSim:
    """Runtime of one water region (port of water_prepare / water_step / water_render_rect)."""

    def __init__(self, w: Water):
        self.w = w
        self.ok = False
        self.t = self.sx = self.sy = self.svx = self.svy = self.rain_acc = self.set_ph = 0.0
        self.sim_acc = self.alpha = 0.0
        self.rng = 12345
        m = w.mask
        if m is None or not m.any():
            return
        ys, xs = np.nonzero(m)
        self.bx, self.by = int(xs.min()), int(ys.min())
        self.bw, self.bh = int(xs.max()) - self.bx + 1, int(ys.max()) - self.by + 1
        self.area = int(m.sum())
        sub = m[self.by:self.by + self.bh, self.bx:self.bx + self.bw]
        # edge hold: distance from the bank (the photo's own border isn't a bank)
        pad = np.pad(m, 1, constant_values=True)
        cur = pad.copy(); dist = np.zeros(pad.shape)
        edge_px = w.get("edge")
        for k in range(int(math.ceil(edge_px)) + 2):
            dist += cur
            if k % 2 == 0:
                cur = cur & np.roll(cur, 1, 0) & np.roll(cur, -1, 0) & np.roll(cur, 1, 1) & np.roll(cur, -1, 1)
            else:
                e = cur.copy()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        e &= np.roll(np.roll(cur, dy, 0), dx, 1)
                cur = e
            cur[0, :] = cur[-1, :] = cur[:, 0] = cur[:, -1] = True
            cur &= pad
        d = dist[1:-1, 1:-1][self.by:self.by + self.bh, self.bx:self.bx + self.bw] - 0.5
        e = smooth01(d / edge_px) if edge_px > 0.01 else np.ones(d.shape)
        self.e = np.where(sub, np.maximum(e, 1 / 255), 0.0)
        # grid
        self.gw, self.gh = (self.bw + WCELL - 1) // WCELL + 2, (self.bh + WCELL - 1) // WCELL + 2
        I, Jg = np.meshgrid(np.arange(self.gw), np.arange(self.gh))
        CX = self.bx + (I - 1) * WCELL + WCELL // 2; CY = self.by + (Jg - 1) * WCELL + WCELL // 2
        inb = (I > 0) & (Jg > 0) & (I < self.gw - 1) & (Jg < self.gh - 1) & (CX < IMG_W) & (CY < IMG_H) & (CX >= 0) & (CY >= 0)
        self.cin = np.zeros((self.gh, self.gw), bool)
        self.cin[inb] = m[CY[inb], CX[inb]]
        c = self.cin.astype(np.int8)
        self.cnx = np.zeros_like(c); self.cny = np.zeros_like(c)
        self.cnx[1:-1, 1:-1] = (c[1:-1, :-2] - c[1:-1, 2:]) * c[1:-1, 1:-1]
        self.cny[1:-1, 1:-1] = (c[:-2, 1:-1] - c[2:, 1:-1]) * c[1:-1, 1:-1]
        self.h0 = np.zeros((self.gh, self.gw)); self.h1 = np.zeros((self.gh, self.gw))
        self.gx = np.zeros((self.gh, self.gw)); self.gy = np.zeros((self.gh, self.gw))
        self.gxp = np.zeros((self.gh, self.gw)); self.gyp = np.zeros((self.gh, self.gw))
        PX = self.bx + (I - 1) * WCELL + WCELL * 0.5; PY = self.by + (Jg - 1) * WCELL + WCELL * 0.5
        curves = [water_curve(l) for l in w.lines[:WLINE_MAX]]
        self.fx = self.fy = self.od = None
        if w.mode == W_RIVER:
            sx = np.zeros(PX.shape); sy = np.zeros(PX.shape); sw = np.zeros(PX.shape)
            for cx_, cy_ in curves:
                if len(cx_) < 2:
                    continue
                d_, tx, ty, _, _ = curve_nearest(cx_, cy_, PX, PY)
                wt = 1.0 / (d_ * d_ + 144.0)
                sx += wt * tx; sy += wt * ty; sw += wt
            l2 = np.hypot(sx, sy)
            ok = (sw > 0) & (l2 > 1e-6)
            with np.errstate(divide="ignore", invalid="ignore"):
                self.fx = np.where(ok, sx / l2, 0.0); self.fy = np.where(ok, sy / l2, 0.0)
        if w.mode == W_OCEAN:
            best = np.full(PX.shape, 1e9); bnx = np.zeros(PX.shape); bny = np.ones(PX.shape)
            for cx_, cy_ in curves:
                if len(cx_) < 1:
                    continue
                d_, _, _, nx, ny = curve_nearest(cx_, cy_, PX, PY)
                m_ = d_ < best
                best = np.where(m_, d_, best); bnx = np.where(m_, nx, bnx); bny = np.where(m_, ny, bny)
            none = best > 1e8
            self.od = np.where(none, (self.by + self.bh) - PY, best)
            self.onx = np.where(none, 0.0, bnx); self.ony = np.where(none, 1.0, bny)
        Y, X = np.mgrid[self.by:self.by + self.bh, self.bx:self.bx + self.bw]
        self.X, self.Y = X.astype(np.float64), Y.astype(np.float64)
        # the node lattice (every second pixel, x and y even) the waves are worked out on, and the four nodes
        # each pixel blends (½ weights: an odd pixel is halfway between two) - as the Core2
        self.ni0, self.nj0 = self.bx >> 1, self.by >> 1
        NY, NX = np.mgrid[2 * self.nj0:2 * (((self.by + self.bh - 1) >> 1) + 2):2, 2 * self.ni0:2 * (((self.bx + self.bw - 1) >> 1) + 2):2]
        self.NX, self.NY = NX.astype(np.float64), NY.astype(np.float64)
        self.pi, self.pj = (X >> 1) - self.ni0, (Y >> 1) - self.nj0
        self.po, self.pq = X & 1, Y & 1
        self.ok = True
        self.pax = self.pay = None                  # perspective grid (see water_perspective)
        self.ghz = -1.0
        self.oworld = False                         # ocean distances on the water's surface (grid)
        self._w = w
        if w.grid:
            self._perspective(w.grid)

    # --- motion
    def _rand(self):
        self.rng = (self.rng * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.rng >> 8

    def _randf(self):
        return (self._rand() & 0xFFFF) / 65536.0

    def _perspective(self, q):
        H = water_homog(q)
        G, Hh, I = water_unmap(H, 0.0, 0.0)[3]
        kx = np.zeros((self.gh, self.gw)); ky = np.zeros((self.gh, self.gw))
        for j in range(self.gh):
            for i in range(self.gw):
                sx, sy = self.bx + (i - 1) * WCELL + WCELL * 0.5, self.by + (j - 1) * WCELL + WCELL * 0.5
                u, v, ww, _ = water_unmap(H, sx, sy)
                if abs(ww) > 1e-9 and H[6] * u + H[7] * v + 1.0 > 1e-4:
                    e = 1e-3
                    x0, y0 = water_hmap(H, u, v); xu, yu = water_hmap(H, u + e, v); xv, yv = water_hmap(H, u, v + e)
                    kx[j, i] = math.hypot(xu - x0, yu - y0) / e; ky[j, i] = math.hypot(xv - x0, yv - y0) / e
        kref = kx[self.cin].max() if self.cin.any() else 0
        if kref <= 0:
            return
        ax, ay = np.clip(kx / kref, 0.05, 1.0), np.clip(ky / kref, 0.05, 1.0)
        self.pax, self.pay = 0.5 * ax * ax, 0.5 * ay * ay
        if abs(Hh) > 1e-9:
            self.ghz = -(G * (IMG_W * 0.5) + I) / Hh
        if self._w.mode == W_OCEAN and self.od is not None:
            # swells on the water's own surface: distance across the plane, rolling toward the near edge
            vshore = -1e9
            for l in self._w.lines[:WLINE_MAX]:
                for sx, sy in l[:WPTS_MAX]:
                    u, v, ww, _ = water_unmap(H, float(sx), float(sy))
                    if abs(ww) > 1e-9:
                        vshore = max(vshore, v)
            vv = np.full((self.gh, self.gw), 1e9); onx = self.onx.copy(); ony = self.ony.copy()
            for j in range(self.gh):
                for i in range(self.gw):
                    sx, sy = self.bx + (i - 1) * WCELL + WCELL * 0.5, self.by + (j - 1) * WCELL + WCELL * 0.5
                    u, v, ww, _ = water_unmap(H, sx, sy)
                    if abs(ww) > 1e-9:
                        vv[j, i] = v
                        x0, y0 = water_hmap(H, u, v); x1, y1 = water_hmap(H, u, v + 1e-3)
                        dx, dy = x0 - x1, y0 - y1; l = math.hypot(dx, dy)
                        if l > 1e-9:
                            onx[j, i], ony[j, i] = dx / l, dy / l
            vmin = vv[self.cin].min() if self.cin.any() else 0.0
            v0 = vshore if vshore > -1e8 else vmin
            self.od = np.maximum(0.0, (vv - v0) * kref)
            self.onx, self.ony = onx, ony
            self.oworld = True

    def poke(self, x, y, amt=160.0, r=2):
        if not self.ok:
            return
        ci, cj = int(math.floor((x - self.bx) / WCELL)) + 1, int(math.floor((y - self.by) / WCELL)) + 1
        if self.pax is not None and 0 <= ci < self.gw and 0 <= cj < self.gh:
            s = math.sqrt(2.0 * self.pax[cj, ci])
            amt *= s
            r = max(1, int(round(r * s)))
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                i, j = ci + di, cj + dj
                if i < 1 or j < 1 or i >= self.gw - 1 or j >= self.gh - 1:
                    continue
                d2, rr = float(di * di + dj * dj), float(r * r) + 0.5
                if d2 > rr:
                    continue
                if self.cin[j, i]:
                    self.h1[j, i] -= amt * (1.0 - d2 / (rr + 0.5))

    def contains(self, x, y):
        return self.ok and 0 <= int(x) < IMG_W and 0 <= int(y) < IMG_H and bool(self.w.mask[int(y), int(x)])

    def step(self, dt, dn, jolt):
        if not self.ok:
            return
        w = self.w
        dt = min(dt, 0.1); self.t += dt
        sl = w.get("slosh") * (W_OCEAN_FEEL if w.mode == W_OCEAN else 1.0)    # an ocean only just feels them
        tx, ty = sl * W_SLOSH_PX * dn[0], sl * W_SLOSH_PX * (dn[1] - 1.0) * 0.6
        ax, ay = (tx - self.sx) * 10.0 - self.svx * 1.6, (ty - self.sy) * 10.0 - self.svy * 1.6
        kx, ky = ax * dt, ay * dt
        self.svx += kx; self.svy += ky; self.sx += self.svx * dt; self.sy += self.svy * dt
        if abs(kx) + abs(ky) > 1e-4:
            self.h1 += 9.0 * sl * (np.sqrt(2.0 * self.pax) if self.pax is not None else 1.0) * (kx * self.cnx + ky * self.cny)
        rip = w.get("ripple")
        jm = math.sqrt(sum(v * v for v in jolt))
        if jm > 0.06 and rip > 0:
            for _ in range(int(jm * 6.0 * sl) + 1):
                px, py = self.bx + self._randf() * self.bw, self.by + self._randf() * self.bh
                self.poke(px, py, 45.0 * jm * rip * (0.4 if w.mode == W_OCEAN else 1.0), 1)
        if w.get("rain") >= 0.5 and rip > 0:
            self.rain_acc += dt * w.get("rate") * self.area / 1500.0
            while self.rain_acc >= 1.0:
                self.rain_acc -= 1.0
                px, py = self.bx + self._randf() * self.bw, self.by + self._randf() * self.bh
                if w.mask[int(py), int(px)]:
                    self.poke(px, py, (45.0 + 35.0 * self._randf()) * rip, 1)
        self.sim_acc += dt
        guard = 0
        while self.sim_acc >= W_SIM_DT and guard < 4:      # fixed 25 Hz, like the firmware
            guard += 1
            self.sim_acc -= W_SIM_DT
            nb = np.zeros_like(self.h0)
            if self.pax is not None:
                hc = self.h1[1:-1, 1:-1]
                nb[1:-1, 1:-1] = (2 * hc + self.pax[1:-1, 1:-1] * (self.h1[1:-1, :-2] + self.h1[1:-1, 2:] - 2 * hc)
                                  + self.pay[1:-1, 1:-1] * (self.h1[:-2, 1:-1] + self.h1[2:, 1:-1] - 2 * hc) - self.h0[1:-1, 1:-1])
            else:
                nb[1:-1, 1:-1] = (self.h1[1:-1, :-2] + self.h1[1:-1, 2:] + self.h1[:-2, 1:-1] + self.h1[2:, 1:-1]) * 0.5 - self.h0[1:-1, 1:-1]
            self.h0 = np.where(self.cin, nb * 0.985, 0.0)
            self.h0, self.h1 = self.h1, self.h0
            self.gxp, self.gyp = self.gx.copy(), self.gy.copy()
            self.gx[1:-1, 1:-1] = self.h1[1:-1, 2:] - self.h1[1:-1, :-2]
            self.gy[1:-1, 1:-1] = self.h1[2:, 1:-1] - self.h1[:-2, 1:-1]
        self.sim_acc = min(self.sim_acc, W_SIM_DT)
        self.alpha = self.sim_acc / W_SIM_DT
        self.set_ph += dt / (W_OCEAN_T / w.get("speed") * 7.0)

    # --- render
    def _lerp(self, a, CX, CY):
        i = np.clip(CX.astype(int), 0, self.gw - 2); j = np.clip(CY.astype(int), 0, self.gh - 2)
        fx = np.clip(CX - i, 0, 1); fy = np.clip(CY - j, 0, 1)
        return (a[j, i] * (1 - fx) + a[j, i + 1] * fx) * (1 - fy) + (a[j + 1, i] * (1 - fx) + a[j + 1, i + 1] * fx) * fy

    def _clamp(self, ox, oy):
        m = self.w.mask
        for _ in range(3):
            sx = np.clip(np.floor(self.X + ox + 0.5).astype(int), 0, IMG_W - 1)
            sy = np.clip(np.floor(self.Y + oy + 0.5).astype(int), 0, IMG_H - 1)
            inside = m[sy, sx] & (self.X + ox + 0.5 >= 0) & (self.X + ox + 0.5 < IMG_W) & (self.Y + oy + 0.5 >= 0) & (self.Y + oy + 0.5 < IMG_H)
            ox = np.where(inside, ox, ox * 0.5); oy = np.where(inside, oy, oy * 0.5)
            if inside.all():
                return ox, oy
        sx = np.clip(np.floor(self.X + ox + 0.5).astype(int), 0, IMG_W - 1)
        sy = np.clip(np.floor(self.Y + oy + 0.5).astype(int), 0, IMG_H - 1)
        inside = m[sy, sx]
        return np.where(inside, ox, 0.0), np.where(inside, oy, 0.0)

    def _up(self, a):
        """node values → pixels: each pixel the mean of its four nodes (two of them, or one, the same)"""
        i, j, o, q = self.pi, self.pj, self.po, self.pq
        return 0.25 * (a[j, i] + a[j, i + o] + a[j + q, i] + a[j + q, i + o])

    def render_into(self, frame, src):
        """Draw this water into `frame` (H, W, 3 uint8), sampling the untouched photo `src`. The ripples, waves
        and swell are worked out on the node lattice (half the picture's resolution, as the Core2); the photo
        is sampled at every pixel, at full resolution, and the fade toward the bank is each pixel's own."""
        if not self.ok:
            return
        w, t, X, Y = self.w, self.t, self.NX, self.NY
        hz = w.get("horizon")
        if hz < 0:
            hz = self.ghz if -200 < self.ghz < self.by + self.bh else float(self.by)   # the grid's, if any
        bottom = float(self.by + self.bh)
        p = np.clip((Y - hz) / max(bottom - hz, 1.0), 0.15, 1.0); ip = 1.0 / p
        amb = w.get("amb") * W_AMB_PX * p
        dy = amb * (0.6 * np.sin(X * 0.05 * ip + Y * 0.13 * ip - 1.4 * t) + 0.4 * np.sin(-X * 0.08 * ip + Y * 0.19 * ip - 2.1 * t + 1.7))
        dx = amb * 0.35 * np.sin(X * 0.07 * ip - Y * 0.06 * ip + 1.2 * t)
        CX, CY = (X - self.bx) / WCELL + 0.5, (Y - self.by) / WCELL + 0.5
        rk = W_RIP_K * w.get("ripple")
        a_ = self.alpha                                   # blended between the last two steps
        sgx = self._lerp(self.gxp + (self.gx - self.gxp) * a_, CX, CY)
        sgy = self._lerp(self.gyp + (self.gy - self.gyp) * a_, CX, CY)
        dx = dx + sgx * rk + self.sx * 0.5; dy = dy + sgy * rk + self.sy * 0.5
        slope = np.hypot(sgx, sgy)
        light, foam_k = w.get("light"), w.get("foam")
        shade = 1.0 - light * W_RIP_SHADE * sgy * (1.0 if rk > 0 else 0.0)     # painted light (optional)
        foam = np.zeros_like(X)
        if w.mode == W_OCEAN and self.od is not None:
            d = self._lerp(self.od, CX, CY); nx = self._lerp(self.onx, CX, CY); ny = self._lerp(self.ony, CX, CY)
            lam = w.get("spacing") * (1.0 if self.oworld else np.maximum(p, 0.3))
            T = W_OCEAN_T / w.get("speed")
            ph = 2 * math.pi * (d / lam + t / T)
            env = 1.0 + w.get("sets") * 0.6 * math.sin(2 * math.pi * self.set_ph)
            A = W_OCEAN_A * w.get("size") * p * env
            sw = np.sin(ph) + 0.35 * np.sin(2.0 * ph - 0.6)
            dy = dy - A * 0.8 * sw
            orb = A * 0.6 * np.cos(ph)
            run = np.where(d < lam * 0.5, A * 0.8 * np.maximum(np.sin(ph), 0) * (1.0 - d / (lam * 0.5)), 0.0)
            dx = dx + nx * (orb + run); dy = dy + ny * (orb + run)
            cp, sp_ = np.cos(ph), np.sin(ph)
            shade = shade + light * W_OCEAN_SHADE * (A / W_OCEAN_A) * (cp + 0.7 * np.cos(2.0 * ph - 0.6))
            zone = lam * 0.6
            foam = np.where((d < zone) & (sp_ > 0.7), foam_k * (sp_ - 0.7) * 3.3 * (1.0 - d / zone) * np.minimum(A / W_OCEAN_A, 1.0), 0.0)
        g = w.get("glint")
        glint = np.clip((slope - 18.0) / 40.0, 0, 1) * g if g > 0 else None
        # … to the pixels: blended from the nodes, faded by each pixel's own distance from the bank
        e, up = self.e, self._up
        f = src.astype(np.float32)

        def sample(ox, oy):
            sx = np.clip(self.X + ox, 0, IMG_W - 1.001); sy = np.clip(self.Y + oy, 0, IMG_H - 1.001)
            ix, iy = sx.astype(int), sy.astype(int)
            fx_, fy_ = (sx - ix)[..., None], (sy - iy)[..., None]
            top = f[iy, ix] * (1 - fx_) + f[iy, ix + 1] * fx_
            bot = f[iy + 1, ix] * (1 - fx_) + f[iy + 1, ix + 1] * fx_
            return top * (1 - fy_) + bot * fy_

        if w.mode == W_RIVER and self.fx is not None:     # the two phases of the flow, cross-faded
            sp = W_RIVER_PX * w.get("speed") * W_RIVER_T
            fx_, fy_ = self._lerp(self.fx, CX, CY), self._lerp(self.fy, CX, CY)
            f1 = (t / W_RIVER_T) % 1.0; f2 = (f1 + 0.5) % 1.0
            a = self._clamp(up(dx - fx_ * (f1 - 0.5) * sp) * e, up(dy - fy_ * (f1 - 0.5) * sp) * e)
            b = self._clamp(up(dx - fx_ * (f2 - 0.5) * sp) * e, up(dy - fy_ * (f2 - 0.5) * sp) * e)
            mixw = abs(2.0 * f1 - 1.0)
            col = sample(*a) * (1 - mixw) + sample(*b) * mixw
        else:
            col = sample(*self._clamp(up(dx) * e, up(dy) * e))
        col = col * ((up(shade) - 1.0) * e + 1.0)[..., None]
        fm = np.clip(up(foam), 0, 0.85) * e
        col = col + (255.0 - col) * fm[..., None]
        col = np.clip(col, 0, 255)
        if glint is not None:
            lum = col.mean(-1)
            k = up(glint) * e * (lum / 255.0) * (lum > 120)
            col = col + (255.0 - col) * k[..., None]
        sub = frame[self.by:self.by + self.bh, self.bx:self.bx + self.bw]
        inside = e > 0
        sub[inside] = np.clip(col[inside] + 0.5, 0, 255).astype(np.uint8)


def parse_jig(text: str) -> list[Zone]:
    """Zones only (older callers). See parse_jig_all for zones + water."""
    return parse_jig_all(text)[0]


def parse_jig_all(text: str):
    """(zones, waters). Old files (zones only, any older keys) load unchanged; unknown words are ignored."""
    lines = [l.strip() for l in text.splitlines()]
    if not lines or not lines[0].startswith(JIG_MAGIC):
        raise ValueError("not a .jig file (missing MGJIG1 header)")
    names = [s[0] for s in SHAPES]
    mats = [m[0] for m in MATERIALS]
    zones, waters = [], []
    for line in lines[1:]:
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "water":                                  # a water region starts
            if len(waters) >= WATER_MAX or np is None:
                waters.append(None) if len(waters) < 99 else None
                continue
            w = Water(mode=[m_[0] for m_ in WMODES].index(tok[1]) if len(tok) > 1 and tok[1] in [m_[0] for m_ in WMODES] else W_STILL)
            for t in tok[2:]:
                k, _, v = t.partition("=")
                if k == "name":
                    w.name = v.replace("_", " ")[:15]
                if k in WPARAM_KEYS:
                    try:
                        fv = float(v)
                        if fv == fv:
                            w.prm[k] = fv
                    except ValueError:
                        pass
            waters.append(w)
            continue
        if tok[0] in ("wline", "wrow", "wgrid"):
            w = waters[-1] if waters else None
            if w is None:
                continue
            try:
                if tok[0] == "wline" and len(w.lines) < WLINE_MAX:
                    pts = [tuple(int(v) for v in t.split(",")) for t in tok[1:WPTS_MAX + 1]]
                    w.lines.append([(int(_clamp(x, 0, IMG_W - 1)), int(_clamp(y, 0, IMG_H - 1))) for x, y in pts])
                elif tok[0] == "wgrid":
                    try:
                        pts = [tuple(float(v) for v in t.split(",")) for t in tok[1:5]]
                        if len(pts) == 4 and all(len(p) == 2 for p in pts):
                            w.grid = pts
                    except ValueError:
                        pass
                elif tok[0] == "wrow":
                    y = int(tok[1])
                    if 0 <= y < IMG_H:
                        for t in tok[2:]:
                            a, _, b = t.partition("-")
                            w.mask[y, max(0, int(a)):max(0, min(IMG_W, int(b)))] = True
            except (ValueError, IndexError):
                pass
            continue
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
                if k == "name":
                    z.name = v.replace("_", " ")[:15]
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
        if len(zones) < MAX_ZONES:
            zones.append(z)
    return zones, [w for w in waters if w is not None]


def parse_parallax(text: str):
    """The depth planes in a .jig (empty Parallax if none)."""
    import mg_depth as D
    return D.parse_lines([l.strip() for l in text.splitlines()[1:]])


def parse_hangs(text: str):
    """The hanging things in a .jig (hair, cloth, charms)."""
    import mg_hang as G
    return G.parse_lines([l.strip() for l in text.splitlines()[1:]])


def format_jig(zones: list[Zone], waters=(), par=None, hangs=()) -> str:
    """Zones first (older firmware reads those and skips the rest), then each water region, then depth."""
    out = [JIG_MAGIC]
    for z in zones[:MAX_ZONES]:
        line = f"{SHAPES[z.shape][0]} {round(z.cx)} {round(z.cy)} {round(z.r)} {int(z.rot)} {MATERIALS[z.material][0]}"
        if z.tip:
            line += f" tip {round(z.tip_u * 100)} {round(z.tip_w * 100)}"
        for k in TUNE_KEYS:
            if k in z.tune:
                line += f" {k}={z.tuned(k, 0.0):.3g}"
        if z.name:
            line += " name=" + z.name.replace(" ", "_")[:15]
        out.append(line)
    for w in list(waters)[:WATER_MAX]:
        if w is None or w.mask is None or not w.mask.any():
            continue
        line = f"water {WMODES[w.mode][0]}"
        for k in WPARAM_KEYS:
            if k in w.prm:
                line += f" {k}={w.get(k):.3g}"
        if w.name:
            line += " name=" + w.name.replace(" ", "_")[:15]
        out.append(line)
        for l in w.lines[:WLINE_MAX]:
            if l:
                out.append("wline " + " ".join(f"{int(x)},{int(y)}" for x, y in l[:WPTS_MAX]))
        if w.grid:
            out.append("wgrid " + " ".join(f"{x:.1f},{y:.1f}" for x, y in w.grid))
        for y in range(IMG_H):
            row = w.mask[y]
            if not row.any():
                continue
            d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
            starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
            out.append(f"wrow {y} " + " ".join(f"{a}-{b}" for a, b in zip(starts, ends)))
    if par is not None:
        import mg_depth as D
        out += D.format_lines(par)
    if hangs:
        import mg_hang as G
        out += G.format_lines([default_pin(h) for h in hangs])
    return "\n".join(out) + "\n"


def default_pin(h):
    """No Pin set: it hangs from the middle of its top row (as the Core2 does)."""
    if h is not None and h.a[0] < 0 and h.mask is not None and h.mask.any():
        ys, xs = np.nonzero(h.mask)
        y = int(ys.min()); row = np.flatnonzero(h.mask[y])
        h.a, h.b = (float(row.min()), float(y)), (float(row.max()), float(y))
    return h


# ---- body-part presets (Jiggle tab) ---------------------------------------------------------------
# name, material, point position (0 = at the attachment … 1 = the far free end; None = no point), dials
BODY_PRESETS = [
    ("Breast · natural", SOFT, 0.65, {}),
    ("Breast · heavy / hanging", SOFT, 0.88, {"mass": 1.5, "slack": 0.75}),
    ("Breast · perky / firm", NATURAL, 0.42, {"slack": 0.15}),
    ("Breast · small / athletic", FIRM, 0.6, {"mass": 0.7}),
    ("Butt cheek · from behind", SOFT, 0.62, {"mass": 1.5, "anchor": 0.55, "slack": 0.6, "pend": 0.12,
                                             "couple": 1.1, "stretch": 0.8}),
    ("Butt · side view", SOFT, 0.55, {"mass": 1.6, "anchor": 0.5, "slack": 0.5, "pend": 0.2, "couple": 1.0}),
    ("Belly", SOFT, 0.55, {"mass": 1.7, "anchor": 0.45, "pend": 0.1, "couple": 1.0}),
    ("Thigh / hip", NATURAL, None, {"mass": 1.3, "anchor": 0.6, "pend": 0.05, "travel": 0.6}),
    ("Cheek / chin", FIRM, None, {"mass": 0.6, "anchor": 0.5, "travel": 0.5}),
]
PRESET_NAMES = [p[0] for p in BODY_PRESETS]


# Zones placed on the Core2 itself start out strong (firmware device_boost); shown as set dials here.
DEVICE_BOOST = {"sag": 1.8, "travel": 1.45, "sens": 1.35, "pend": 0.4, "slack": 0.55, "dance": 1.4}


def point_along(z: Zone, along):
    """Point on the line from the attachment through the centre to the far edge (local units)."""
    au, aw = shape_attach(z.shape, z.tail())
    du, dw = -au, -aw
    L = math.hypot(du, dw) or 1.0
    du, dw = du / L, dw / L
    lo, hi = L, L + 2.5                                  # far edge beyond the centre
    for _ in range(30):
        m = 0.5 * (lo + hi)
        if z.sdf(au + du * m, aw + dw * m) < 0:
            lo = m
        else:
            hi = m
    t = along * lo
    return au + du * t, aw + dw * t


# ---- editing (what the Jiggle tab drives) ---------------------------------------------------------
@dataclass
class Editor:
    """Zones on one image, selection, and mouse/keyboard editing. Coordinates are image pixels."""
    zones: list = field(default_factory=list)
    waters: list = field(default_factory=list)       # water regions (see Water)
    water_sel: int = -1
    par: object = None                               # depth planes (mg_depth.Parallax)
    plane_sel: int = -1
    hangs: list = field(default_factory=list)        # hanging things (mg_hang.Hang)
    hang_sel: int = -1
    target: str = "water"                            # which mask the tools edit: "water", "depth" or "hang"
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
        txt = jp.read_text(encoding="utf-8") if jp.exists() else None
        self.zones, self.waters = parse_jig_all(txt) if txt else ([], [])
        import mg_depth as D
        self.par = parse_parallax(txt) if txt else D.Parallax()
        self.hangs = parse_hangs(txt) if txt else []
        self.hang_sel = 0 if self.hangs else -1
        self.selected, self.armed, self.dirty = -1, -1, False
        self.water_sel = 0 if self.waters else -1
        self.plane_sel = 0 if self.par.planes else -1
        return jp.exists()

    def save(self) -> Path:
        if self.mg_path is None:
            raise ValueError("open an .mg image first")
        jp = jig_path_for(self.mg_path)
        has_water = any(w.mask is not None and w.mask.any() for w in self.waters)
        has_depth = self.par is not None and self.par.any()
        has_hang = any(h.ok() for h in self.hangs)
        if self.zones or has_water or has_depth or has_hang:
            jp.write_text(format_jig(self.zones, self.waters, self.par, self.hangs), encoding="utf-8", newline="\n")
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

    def rot_handle_xy(self, z=None):
        """Rotation handle: just outside the zone on its 'top' (local up) side."""
        z = z or self.zone
        return z.from_local(0.0, 1.0 + 14.0 / z.r)

    def near_rot(self, x, y):
        z = self.zone
        if not z:
            return False
        hx, hy = self.rot_handle_xy(z)
        return (x - hx) ** 2 + (y - hy) ** 2 <= 8.0 ** 2

    def apply_preset(self, name):
        """Body-part starting point: material, point placement and the dials that matter for it.
        Keeps position, size, shape and rotation."""
        z = self.zone
        if not z:
            return False
        pr = next((p for p in BODY_PRESETS if p[0] == name), None)
        if pr is None:
            return False
        _, mat, along, tune = pr
        z.material = mat
        self.last_material = mat
        side = z.tune.get("side")
        z.tune = dict(tune)
        if side is not None:
            z.tune["side"] = side
        if along is None:
            z.tip = False
        else:
            z.tip = True
            z.tip_u, z.tip_w = point_along(z, along)
            z.clamp_point()
        self._changed()
        return True

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
        if self.near_rot(x, y):
            self._drag = "rot"
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
        elif self._drag == "rot":                      # handle follows the mouse, either way round
            z.rot = int(round(math.degrees(math.atan2(x - z.cx, -(y - z.cy))))) % 360
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
        if z.point_ok(nu, nw):
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
            if z.tip:
                z.clamp_point()
            self._changed()

    def set_shape(self, shape):
        z = self.zone
        if z and z.shape != shape:
            z.shape = shape
            if not z.point_ok(z.tip_u, z.tip_w):
                z.tip_u, z.tip_w = shape_tip_default(shape, z.tail())
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
        return ([replace(z, tune=dict(z.tune)) for z in self.zones], [w.copy() for w in self.waters], self.water_sel,
                None if self.par is None else self.par.copy(), self.plane_sel, [h.copy() for h in self.hangs], self.hang_sel)

    def restore(self, snap):
        zones, waters, ws, par, ps, hs, hsel = snap
        self.zones, self.waters, self.water_sel, self.par, self.plane_sel = zones, waters, ws, par, ps
        self.hangs, self.hang_sel = hs, hsel

    # hanging things
    @property
    def hang(self):
        return self.hangs[self.hang_sel] if 0 <= self.hang_sel < len(self.hangs) else None

    def add_hang(self, kind="hair"):
        import mg_hang as G
        if len(self.hangs) >= G.HANG_MAX:
            return False
        self.hangs.append(G.Hang(kind, (-1.0, -1.0), (-1.0, -1.0), np.zeros((IMG_H, IMG_W), bool), {}))
        self.hang_sel = len(self.hangs) - 1
        self._changed()
        return True

    def delete_hang(self):
        if self.hang is None:
            return
        del self.hangs[self.hang_sel]
        self.hang_sel = min(self.hang_sel, len(self.hangs) - 1)
        self._changed()

    def pin(self, x, y, second=False):
        """Where the current hanging thing hangs from: A (and B = A), then B for a line."""
        h = self.hang
        if h is None:
            return
        if second:
            h.b = (float(x), float(y))
        else:
            h.a = h.b = (float(x), float(y))
        self._changed()

    # depth planes
    @property
    def plane(self):
        P = self.par
        return P.planes[self.plane_sel] if P is not None and 0 <= self.plane_sel < len(P.planes) else None

    def add_plane(self, depth=0.6):
        import mg_depth as D
        if self.par is None:
            self.par = D.Parallax()
        if len(self.par.planes) >= D.PLANE_MAX:
            return False
        self.par.planes.append(D.Plane(depth=depth, mask=np.zeros((IMG_H, IMG_W), bool)))
        self.plane_sel = len(self.par.planes) - 1
        self._changed()
        return True

    def delete_plane(self):
        if self.plane is None:
            return
        del self.par.planes[self.plane_sel]
        self.plane_sel = min(self.plane_sel, len(self.par.planes) - 1)
        self._changed()

    def planes_from_depth_map(self, depth, n=3, near_is_bright=True):
        import mg_depth as D
        if self.par is None:
            self.par = D.Parallax()
        box = getattr(self, "content_box", None)
        self.par.planes = D.planes_from_depth(depth, n, near_is_bright, box)
        self.plane_sel = len(self.par.planes) - 1
        self._changed()
        return len(self.par.planes)

    # water
    @property
    def water(self):
        return self.waters[self.water_sel] if 0 <= self.water_sel < len(self.waters) else None

    def add_water(self, mode=W_STILL):
        if len(self.waters) >= WATER_MAX:
            return False
        self.waters.append(Water(mode=mode))
        self.water_sel = len(self.waters) - 1
        self._changed()
        return True

    def delete_water(self):
        if self.water is None:
            return
        del self.waters[self.water_sel]
        self.water_sel = min(self.water_sel, len(self.waters) - 1)
        self._changed()

    def water_paint(self, x, y, r, erase=False):
        w = self.water
        if w is not None:
            mask_brush(w.mask, int(round(x)), int(round(y)), int(r), not erase)
            self._changed()

    # ---- smart selection (same maths as the Core2: mg_select / src/jiggle_select.h) ----
    def _cur_mask(self):
        if self.target == "depth":
            p = self.plane
            return None if p is None else p.mask
        if self.target == "hang":
            h = self.hang
            return None if h is None else h.mask
        w = self.water
        return None if w is None else w.mask

    def mask_paint(self, x, y, r, erase=False):
        m = self._cur_mask()
        if m is not None:
            mask_brush(m, int(round(x)), int(round(y)), int(r), not erase)
            self._changed()

    def smart_pick(self, marks, add=True, step=0, trim=True):
        """Select from a tap / stroke (`marks`: bool array) into the current mask. Keeps what's needed
        to step the edge with smart_step. Returns (pixels changed, Grown)."""
        m = self._cur_mask()
        if m is None or self.image is None or not np.asarray(marks).any():
            return 0, None
        import mg_select as S
        trim = (1 if trim else 0) if self.target not in ("depth", "hang") else -1   # water keeps off the bank; planes / hangs take their rim
        self._smart = {"marks": np.asarray(marks, bool).copy(), "before": m.copy(), "add": add, "step": step, "trim": trim}
        return self._smart_run()

    def _smart_run(self):
        import mg_select as S
        st, m = self._smart, self._cur_mask()
        m[:] = st["before"]
        g = S.grow(self.image, st["marks"], st["step"])
        n = S.apply(m, g.sel, st["add"], st["trim"], seeds=st["marks"])
        st["grown"] = g
        self._changed()
        return n, g

    def smart_step(self, d):
        """Smaller (−1) / bigger (+1) edge for the last smart pick. Returns (changed, Grown) or None."""
        st = getattr(self, "_smart", None)
        if not st or "grown" not in st or self._cur_mask() is None:
            return None
        g = st["grown"]
        if not g.levels or not (0 <= g.index + d < len(g.levels)):
            return None
        st["step"] += d
        return self._smart_run()

    def grow_mask(self, n):
        """Grow (+) / shrink (−) the current mask by |n| px."""
        m = self._cur_mask()
        if m is None:
            return
        import mg_select as S
        m[:] = S.grow_mask(m, n)
        self._changed()

    def smart_forget(self):
        self._smart = None

    def snap_fill(self, rough, trim=True):
        """A rough shape → fitted to the real edges inside it, added to the current mask."""
        m = self._cur_mask()
        if m is None or self.image is None:
            return 0
        import mg_select as S
        n = S.apply(m, S.snap(self.image, np.asarray(rough, bool)), True, -1 if self.target in ("depth", "hang") else (1 if trim else 0))
        self._changed()
        return n

    def snap_mask(self, trim=True):
        """Snap the whole current mask (e.g. painted roughly) to the real edges."""
        m = self._cur_mask()
        if m is None or self.image is None or not m.any():
            return 0
        import mg_select as S
        fitted = S.snap(self.image, m.copy())
        before = m.copy()
        m[:] = False
        S.apply(m, fitted, True, -1 if self.target in ("depth", "hang") else (1 if trim else 0))
        self._changed()
        return int((before ^ m).sum())

    def water_fill(self, pts, erase=False, snap=False):
        m = self._cur_mask()
        if m is not None and len(pts) >= 3:
            p = [(int(round(x)), int(round(y))) for x, y in pts]
            if snap and not erase and self.image is not None:
                rough = np.zeros((IMG_H, IMG_W), bool)
                mask_poly(rough, p, True)
                self.snap_fill(rough)
                return
            mask_poly(m, p, not erase)
            self._changed()

    def water_set_line(self, i, pts):
        w = self.water
        if w is None:
            return
        pts = [(int(round(x)), int(round(y))) for x, y in pts][:WPTS_MAX]
        while len(w.lines) <= i:
            w.lines.append([])
        w.lines[i] = pts
        w.lines = [l for l in w.lines if l][:WLINE_MAX]
        self._changed()


# ---- motion presets for the preview -------------------------------------------------------------
MOTIONS = [("tiltlr", "Tilt L/R"), ("sweep", "Sweep"), ("shake", "Shake"), ("bounce", "Hop"), ("walk", "Walk"),
           ("run", "Run"), ("jolt", "Push"), ("sway", "Sway"), ("spin", "Spin"), ("bumps", "Car bumps"), ("jump", "Big jump")]


def motion_gyro(kind: str, t: float):
    """(about the screen's horizontal axis: top edge away +, about its vertical axis: right edge away +),
    deg/s, for the motions that turn the Core2 (what drives the depth view on the device)."""
    if kind == "tiltlr" and t < 4.0:                 # turn the right edge away and back, twice (±15°)
        w = 2 * math.pi * 0.5
        return 0.0, 15.0 * w * math.degrees(1) * math.radians(1) * math.cos(w * t) * min(1.0, t / 0.3)
    if kind == "sweep" and t < 6.0:                  # a slow look around: ±22° side to side, ±8° up and down
        w1, w2 = 2 * math.pi / 3.0, 2 * math.pi / 4.5
        return 8.0 * w2 * math.cos(w2 * t), 22.0 * w1 * math.cos(w1 * t)
    return 0.0, 0.0


def motion_twist(kind: str, t: float):
    """(angular acceleration rad/s², angular speed rad/s) about the screen normal for a motion."""
    if kind == "spin" and t < 2.4:                   # twist one way, pause, twist back
        w = 9.0 * math.sin(math.pi * min(1.0, t / 0.9)) if t < 0.9 else (
            -9.0 * math.sin(math.pi * min(1.0, (t - 1.3) / 0.9)) if 1.3 < t < 2.2 else 0.0)
        a = (9.0 * math.pi / 0.9) * math.cos(math.pi * t / 0.9) if t < 0.9 else (
            -(9.0 * math.pi / 0.9) * math.cos(math.pi * (t - 1.3) / 0.9) if 1.3 < t < 2.2 else 0.0)
        return a, w
    return 0.0, 0.0


def motion_accel(kind: str, t: float, tilt_deg: float = 0.0, lean_deg: float = 0.0):
    """Screen-space accelerometer reading (g) for a preview motion at time t since it started.
    tilt_deg turns the Core2 in its own plane (90 = on its side, 180 = upside down); lean_deg tips
    the top toward (+) or away from (−) you."""
    th, ph = math.radians(tilt_deg), math.radians(lean_deg)
    gx, gy = -math.sin(th) * math.cos(ph), -math.cos(th) * math.cos(ph)
    gz = math.sin(ph)
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
    elif kind == "run":                              # jogging, ~165 steps a minute for 4 s
        if t < 4.0:
            ph_ = (t * 165 / 60) % 1.0
            ay = -1.1 * math.exp(-((ph_ - 0.1) / 0.05) ** 2) + 0.35 * math.sin(2 * math.pi * ph_)
            ax = 0.25 * math.sin(math.pi * t * 165 / 60)
    elif kind == "sway":                             # rock side to side at 1.6 Hz for 3 s
        if t < 3.0:
            ax = 0.55 * math.sin(2 * math.pi * 1.6 * t) * min(1.0, t / 0.3)
    elif kind == "bumps":                            # car on a rough road: uneven jolts, some sideways
        for i, (t0, a_, b_) in enumerate(((0.15, 0.9, 0.2), (0.55, -0.6, -0.3), (0.8, 1.2, 0.1),
                                          (1.35, 0.5, 0.35), (1.6, -0.9, -0.15), (2.3, 1.4, 0.0),
                                          (2.45, -1.0, 0.25), (3.1, 0.7, -0.3))):
            ay -= a_ * pulse(t0, 0.035); ax += b_ * pulse(t0 + 0.02, 0.04)
    elif kind == "jump":                             # crouch, launch, airborne (reads ~0 g), hard landing
        ay = 0.6 * pulse(0.15, 0.08) - 1.8 * pulse(0.35, 0.06) - 2.6 * pulse(0.9, 0.04) + 0.6 * pulse(1.0, 0.05)
        if 0.45 < t < 0.85:
            return ax, ay, az
    return gx + ax, gy + ay, gz + az


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
    if "down" in m.tune:
        m.tune["down"] = -m.tune["down"]
    return m




# ---- water perspective grid (same maths as water_homog / water_perspective in the firmware) ----------
def water_homog(q):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = q
    dx1, dx2, dx3 = x1 - x2, x3 - x2, x0 - x1 + x2 - x3
    dy1, dy2, dy3 = y1 - y2, y3 - y2, y0 - y1 + y2 - y3
    den = dx1 * dy2 - dx2 * dy1
    g = (dx3 * dy2 - dx2 * dy3) / den if abs(den) > 1e-9 else 0.0
    h = (dx1 * dy3 - dx3 * dy1) / den if abs(den) > 1e-9 else 0.0
    return (x1 - x0 + g * x1, x3 - x0 + h * x3, x0, y1 - y0 + g * y1, y3 - y0 + h * y3, y0, g, h)


def water_hmap(H, u, v):
    a, b, c, d, e, f, g, h = H
    w = g * u + h * v + 1.0
    return (a * u + b * v + c) / w, (d * u + e * v + f) / w


def water_unmap(H, x, y):
    a, b, c, d, e, f, g, h = H
    A, B, C = e - f * h, -(b - c * h), b * f - c * e
    D, E, F = -(d - f * g), a - c * g, -(a * f - c * d)
    G, Hh, I = d * h - e * g, -(a * h - b * g), a * e - b * d
    w = G * x + Hh * y + I
    return (A * x + B * y + C) / w, (D * x + E * y + F) / w, w, (G, Hh, I)


def default_grid(mask):
    """A starting grid over a water mask: its box, narrowed toward the top (far)."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    x0, x1, y0, y1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    inset = (x1 - x0) * 0.2
    return [(x0, y1), (x1, y1), (x1 - inset, y0), (x0 + inset, y0)]
