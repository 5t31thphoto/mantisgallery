"""Mantis Gallery — the Dynamics Studio's scene model (no UI).

Every mask in the Studio is a named, typed layer, and every tool works on the one active layer:

    Sky      above the horizon (stays put)                  built in
    Ground   the ground grid: horizon + how fast it recedes built in
    Figure   something standing in the scene: a depth plane (where its feet meet the ground), a collider
             for hair and cloth, and protected from other layers' painting
    Water    still / river / ocean on the ground's perspective
    Jiggle   a jiggle zone (shape, not a mask)
    Sway     hair lock / cloth / pendant (mg_sway)

The .jig file stays the Core2's format: figures are `plane` lines (with name= and feet=), water `water`
lines, zones their shape lines, sways `sway` lines; names ride along as name= so a save on the Core2 keeps
them. Everything here is plain numpy so the Tk tab (studio_tab.py) is only a view over it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import jiggle_lab as J
import mg_depth as D
import mg_select as S
import mg_sway as SW

W, H = J.IMG_W, J.IMG_H
KINDS = ("sky", "ground", "figure", "water", "jiggle", "sway")
KIND_LABEL = {"sky": "Sky", "ground": "Ground", "figure": "Figure", "water": "Water", "jiggle": "Jiggle", "sway": "Sway"}
FIG_COLS = ["#ff9a3c", "#ff6fae", "#c08cff", "#ffd166"]
WATER_COL, JIG_COL, SWAY_COL, SKY_COL, GROUND_COL = "#3cc8ff", "#b4ff00", "#ffe000", "#8fb8d8", "#c9a86a"
SWAY_KINDS = (("hair", "Hair lock"), ("cloth", "Cloth (pleats)"), ("pendant", "Pendant"))
EXPORT_SCALE = 4                             # masks / photo exported at 4× the Core2's 320×240


@dataclass
class Layer:
    kind: str
    index: int = 0
    obj: object = None

    @property
    def key(self):
        return (self.kind, self.index)

    @property
    def name(self):
        if self.kind == "sky":
            return "Sky"
        if self.kind == "ground":
            return "Ground"
        n = getattr(self.obj, "name", "") or ""
        if n:
            return n
        if self.kind == "jiggle":
            return f"{J.SHAPES[self.obj.shape][1]} zone {self.index + 1}"
        if self.kind == "sway":
            return dict(SWAY_KINDS)[self.obj.kind] + f" {self.index + 1}"
        return f"{KIND_LABEL[self.kind]} {self.index + 1}"

    @property
    def has_mask(self):
        return self.kind in ("figure", "water", "sway")

    @property
    def mask(self):
        return self.obj.mask if self.has_mask and self.obj is not None else None


def _safe_name(s):
    s = "".join(ch for ch in s.strip() if ch.isalnum() or ch in " -_")[:15].strip()
    return s


class Studio:
    """One photo's scene: layers, the active one, tools' maths, undo, files."""

    def __init__(self):
        self.ed = J.Editor()
        self.ed.par = D.Parallax()
        self.sways: list = []
        self.active = ("ground", 0)
        self.undo_stack, self.redo_stack = [], []
        self.protect = {}                       # figure name → protected from other layers' painting (default on)
        self.hidden = set()                     # layer keys not tinted on the canvas
        self.depth_in = None                    # imported / generated depth (H, W) float, near = 1, aligned
        self.depth_raw = None
        self.depth_note = ""
        self._smart = None
        self.dirty = False

    # ---- files ------------------------------------------------------------------------------------
    @property
    def image(self):
        return self.ed.image

    @property
    def par(self):
        if self.ed.par is None:
            self.ed.par = D.Parallax()
        return self.ed.par

    def open(self, path):
        had = self.ed.open_mg(path)
        if self.ed.par is None:
            self.ed.par = D.Parallax()
        jp = J.jig_path_for(path)
        self.sways = []
        if jp.exists():
            sws, _body, _plates = SW.parse_lines([l.strip() for l in jp.read_text(encoding="utf-8").splitlines()[1:]])
            self.sways = sws
        self._adopt_old_water_grid()
        for i, p in enumerate(self.par.planes):
            if not p.name:
                p.name = f"Figure {i + 1}"
        for i, w in enumerate(self.ed.waters):
            if not w.name:
                w.name = ["Water", "Sea", "Pond"][0] + (f" {i + 1}" if len(self.ed.waters) > 1 else "")
        self.active = ("ground", 0)
        self.undo_stack.clear(); self.redo_stack.clear()
        self.depth_in = self.depth_raw = None
        self.depth_note = ""
        self._smart = None
        self.dirty = False
        return had

    def _adopt_old_water_grid(self):
        """Older files: a water's own grid becomes the scene's ground grid (one grid for the picture)."""
        if self.par.grid:
            return
        for w in self.ed.waters:
            if w.grid:
                self.par.grid = [tuple(p) for p in w.grid]
                if not any(p.mask is not None and p.mask.any() for p in self.par.planes):
                    self.par.prm["ground"] = 0.0
                return

    def par_with_sways(self):
        """The depth planes as they move: each figure carries its hair / cloth / pendants (the ones on it,
        or - none named - the figure they overlap most), so nothing of her is left behind when she slides."""
        par = self.par.copy()
        for sw in self.sways:
            if sw.mask is None or not sw.mask.any():
                continue
            idx = [i for i, p in enumerate(par.planes) if sw.on and p.name == sw.on]
            if not idx:
                ov = [((p.mask & S.grow_mask(sw.mask, 2)).sum() if p.mask is not None else 0) for p in par.planes]
                idx = [int(np.argmax(ov))] if ov and max(ov) > 0 else []
            for i in idx:
                par.planes[i].mask = par.planes[i].mask | sw.mask
        return par

    def jig_text(self):
        """The .jig as it will be saved."""
        for w in self.ed.waters:                    # the water lies on the ground: its perspective
            w.grid = None if not self.par.grid else [tuple(p) for p in self.par.grid]
        txt = J.format_jig(self.ed.zones, self.ed.waters, self.par_with_sways(), self.ed.hangs)
        body = self.body_mask()
        plates = None
        ok = [s for s in self.sways if s.ok()]
        if ok and self.image is not None:                # the fill behind each, as the Core2 will draw it
            try:
                protos = J.sway_protos(self.image, ok, body if body.any() else None)
                it = iter(protos)
                plates = [((lambda p: (p.plate, p.fill_hole))(next(it))) if s.ok() else None for s in self.sways]
            except Exception:
                plates = None                            # (the Core2 makes its own)
        out_sw = []
        for sw in self.sways:                            # (a skirt's hips: as lined up, else fitted - the Core2 needs them)
            if sw.kind == "cloth" and len(sw.arm) != SW.HIP_POINTS and sw.mask is not None and sw.mask.any():
                sw = sw.copy(); sw.arm = SW.fit_hips(sw.mask, SW.rest_body(body, [x.mask for x in self.sways]) if body.any() else None,
                                                     sw.guides[0] if sw.guides else None)
            out_sw.append(sw)
        lines = SW.format_lines(out_sw, body if body.any() and self.sways else None, plates)
        if lines:
            txt += "\n".join(lines) + "\n"
        return txt

    def save(self):
        if self.ed.mg_path is None:
            raise ValueError("Open an .mg photo first")
        jp = J.jig_path_for(self.ed.mg_path)
        txt = self.jig_text()
        if txt.strip() != J.JIG_MAGIC:
            jp.write_text(txt, encoding="utf-8", newline="\n")
        elif jp.exists():
            jp.unlink()
        self.dirty = False
        return jp

    # ---- layers -----------------------------------------------------------------------------------
    def layers(self):
        out = [Layer("sky"), Layer("ground")]
        out += [Layer("figure", i, p) for i, p in enumerate(self.par.planes)]
        out += [Layer("water", i, w) for i, w in enumerate(self.ed.waters)]
        out += [Layer("jiggle", i, z) for i, z in enumerate(self.ed.zones)]
        out += [Layer("sway", i, s) for i, s in enumerate(self.sways)]
        return out

    def layer(self, key=None):
        key = key or self.active
        for L in self.layers():
            if L.key == tuple(key):
                return L
        return Layer("ground")

    def by_name(self, name, kind=None):
        for L in self.layers():
            if L.name == name and (kind is None or L.kind == kind):
                return L
        return None

    def set_active(self, key):
        key = tuple(key)
        if self.layer(key).key != key:
            return False
        self.active = key
        self._smart = None
        if key[0] == "jiggle":
            self.ed.selected = key[1]
        else:
            self.ed.selected = -1
        return True

    def color(self, L):
        if L.kind == "figure":
            return FIG_COLS[L.index % len(FIG_COLS)]
        return {"sky": SKY_COL, "ground": GROUND_COL, "water": WATER_COL, "jiggle": JIG_COL, "sway": SWAY_COL}[L.kind]

    def detail(self, L):
        if L.kind == "sky":
            return "above the horizon · stays put"
        if L.kind == "ground":
            return "grid fitted" if self.par.grid else "no grid yet — fit it here"
        if L.kind == "figure":
            d = self.figure_depth(L.index)
            src = "by hand" if L.obj.fixed or not self.par.grid else ("feet line" if L.obj.feet is not None else "lowest point")
            return f"depth {d:.2f} ({src})" + ("" if self.protect.get(L.name, True) else " · unprotected")
        if L.kind == "water":
            return f"{J.WMODES[L.obj.mode][1].lower()}" + (" · on the grid" if self.par.grid else "")
        if L.kind == "jiggle":
            return f"{J.SHAPES[L.obj.shape][1].lower()} · {J.MATERIALS[L.obj.material][1].lower()}"
        sw = L.obj
        return dict(SWAY_KINDS)[sw.kind].lower() + (f" · rests on {sw.on}" if sw.on else "")

    def unique_name(self, base):
        names = {L.name for L in self.layers()}
        if base not in names:
            return base
        k = 2
        while f"{base} {k}" in names:
            k += 1
        return f"{base} {k}"

    # ---- undo -------------------------------------------------------------------------------------
    def _snap(self):
        return (self.ed.snapshot(), [s.copy() for s in self.sways], self.active, dict(self.protect))

    def push_undo(self):
        self.undo_stack.append(self._snap())
        if len(self.undo_stack) > 80:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.dirty = True

    def undo(self):
        if not self.undo_stack:
            return False
        self.redo_stack.append(self._snap())
        self._restore(self.undo_stack.pop())
        return True

    def redo(self):
        if not self.redo_stack:
            return False
        self.undo_stack.append(self._snap())
        self._restore(self.redo_stack.pop())
        return True

    def _restore(self, st):
        ed_snap, sways, active, prot = st
        self.ed.restore(ed_snap)
        if self.ed.par is None:
            self.ed.par = D.Parallax()
        self.sways, self.protect = sways, prot
        self._smart = None
        self.active = active if self.layer(active).key == tuple(active) else ("ground", 0)
        self.dirty = True

    # ---- adding / removing layers -------------------------------------------------------------------
    def add_figure(self, name="Figure"):
        if len(self.par.planes) >= D.PLANE_MAX:
            return None
        self.push_undo()
        self.par.planes.append(D.Plane(depth=0.6, mask=np.zeros((H, W), bool), name=self.unique_name(_safe_name(name) or "Figure")))
        key = ("figure", len(self.par.planes) - 1)
        self.set_active(key)
        return key

    def add_water(self, name="Water", mode=J.W_STILL):
        if len(self.ed.waters) >= J.WATER_MAX:
            return None
        self.push_undo()
        w = J.Water(mode=mode)
        w.name = self.unique_name(_safe_name(name) or "Water")
        self.ed.waters.append(w)
        key = ("water", len(self.ed.waters) - 1)
        self.set_active(key)
        return key

    def add_zone(self, shape, x=W / 2, y=H / 2, name=""):
        if len(self.ed.zones) >= J.MAX_ZONES:
            return None
        self.push_undo()
        tu, tw = J.shape_tip_default(shape)
        z = J.Zone(shape=shape, material=self.ed.last_material, cx=float(x), cy=float(y), r=self.ed.last_r, tip_u=tu, tip_w=tw)
        z.name = _safe_name(name)
        self.ed.zones.append(z)
        key = ("jiggle", len(self.ed.zones) - 1)
        self.set_active(key)
        return key

    def add_sway(self, kind="hair", name=""):
        if len(self.sways) >= SW.SWAY_MAX:
            return None
        self.push_undo()
        base = _safe_name(name) or dict(SWAY_KINDS)[kind].split(" (")[0]
        on = next((L.name for L in self.layers() if L.kind == "figure"), "")
        self.sways.append(SW.Sway(kind, np.zeros((H, W), bool), [], {}, self.unique_name(base), on))
        key = ("sway", len(self.sways) - 1)
        self.set_active(key)
        return key

    def delete_active(self):
        L = self.layer()
        if L.kind in ("sky", "ground"):
            if L.kind == "ground" and self.par.grid:
                self.push_undo()
                self.par.grid = None
                return True
            return False
        self.push_undo()
        if L.kind == "figure":
            del self.par.planes[L.index]
        elif L.kind == "water":
            del self.ed.waters[L.index]
        elif L.kind == "jiggle":
            del self.ed.zones[L.index]
        else:
            del self.sways[L.index]
        self.active = ("ground", 0)
        self.ed.selected = -1
        return True

    def rename_active(self, name):
        L = self.layer()
        name = _safe_name(name)
        if not name or L.kind in ("sky", "ground") or name == L.name:
            return False
        if any(M.name == name for M in self.layers() if M.key != L.key):
            return False
        self.push_undo()
        old = L.name
        L.obj.name = name
        if L.kind == "figure":
            for s in self.sways:
                if s.on == old:
                    s.on = name
            if old in self.protect:
                self.protect[name] = self.protect.pop(old)
        return True

    def move_active(self, d):
        """Reorder within its kind (figures: nearer first matters only for names; kept simple)."""
        L = self.layer()
        lst = {"figure": self.par.planes, "water": self.ed.waters, "jiggle": self.ed.zones, "sway": self.sways}.get(L.kind)
        if lst is None:
            return False
        j = L.index + d
        if not (0 <= j < len(lst)):
            return False
        self.push_undo()
        lst[L.index], lst[j] = lst[j], lst[L.index]
        self.active = (L.kind, j)
        if L.kind == "jiggle":
            self.ed.selected = j
        return True

    # ---- masks: the active layer, protection ----------------------------------------------------------
    def mask(self):
        return self.layer().mask

    def protected(self):
        """Pixels the active layer may not take: other figures (unless unprotected). Hair / cloth may lie
        on a figure, so sways aren't held back."""
        L = self.layer()
        P = np.zeros((H, W), bool)
        if L.kind not in ("figure", "water"):
            return P
        for M in self.layers():
            if M.kind == "figure" and M.key != L.key and self.protect.get(M.name, True) and M.mask is not None:
                P |= M.mask
        return P

    def _trim(self):
        return 1 if self.layer().kind == "water" else -1       # water keeps off the bank; others take their rim

    def _commit(self, m, before):
        """Apply protection to a changed mask: pixels it just took from a protected layer go back."""
        P = self.protected()
        m[P & ~before] = False

    # ---- selection tools ------------------------------------------------------------------------------
    def smart(self, marks, add=True):
        """Smart click / stroke (marks: bool mask of where you clicked / painted)."""
        m = self.mask()
        marks = np.asarray(marks, bool)
        if m is None or self.image is None or not marks.any():
            return 0
        self.push_undo()
        before = m.copy()
        g = S.grow(self.image, marks, 0)
        S.apply(m, g.sel, add, self._trim(), seeds=marks)
        self._commit(m, before)
        self._smart = {"marks": marks.copy(), "before": before, "add": add, "step": 0, "grown": g, "key": self.active}
        return int((before ^ m).sum())

    def cut(self, keep, avoid=None, add=True):
        """Cut out: strokes on it (keep) and on what it isn't (avoid); what's selected already counts as
        maybe-it. The result replaces the layer's mask where the strokes reach (add=False: it's removed)."""
        m = self.mask()
        keep = np.asarray(keep, bool)
        if m is None or self.image is None or not keep.any():
            return 0
        self.push_undo()
        before = m.copy()
        sel = S.cut(self.image, keep, avoid)           # (decided from the strokes alone: all of them, every time)
        if add:                                        # near the strokes it's what the cut says; farther, what
            from scipy.ndimage import distance_transform_edt   # was there stays (brushed bits elsewhere)
            near = distance_transform_edt(~keep) <= 24
            keep_rest = m & ~near
            m[:] = False                               # (small gaps filled, and a figure takes its rim, as the
            S.apply(m, sel, True, self._trim())        # other tools do)
            m |= keep_rest
        else:
            m &= ~sel
        self._commit(m, before)
        self._smart = None
        return int((before ^ m).sum())

    def smart_step(self, d):
        """Tighter (−1) / looser (+1): the last smart pick, a smaller or bigger version."""
        st = self._smart
        m = self.mask()
        if not st or st["key"] != self.active or m is None:
            return None
        g = st["grown"]
        if not g.levels or not (0 <= g.index + d < len(g.levels)):
            return 0
        self.push_undo()
        st["step"] += d
        m[:] = st["before"]
        g = S.grow(self.image, st["marks"], st["step"])
        S.apply(m, g.sel, st["add"], self._trim(), seeds=st["marks"])
        self._commit(m, st["before"])
        st["grown"] = g
        return int((st["before"] ^ m).sum())

    def brush(self, x, y, r, erase=False, start=True):
        m = self.mask()
        if m is None:
            return
        if start:
            self.push_undo()
            self._smart = None
        before = m.copy()
        J.mask_brush(m, int(round(x)), int(round(y)), int(r), not erase)
        if not erase:
            self._commit(m, before)

    def lasso(self, pts, add=True):
        m = self.mask()
        if m is None or len(pts) < 3:
            return 0
        self.push_undo()
        self._smart = None
        before = m.copy()
        J.mask_poly(m, [(int(round(x)), int(round(y))) for x, y in pts], add)
        if add:
            self._commit(m, before)
        return int((before ^ m).sum())

    def refine_edges(self):
        """Snap the whole mask to the real edges (GrabCut where OpenCV is installed). Never empties it."""
        m = self.mask()
        if m is None or self.image is None or not m.any():
            return 0
        self.push_undo()
        before = m.copy()
        fitted = S.snap(self.image, m.copy())
        m[:] = False
        S.apply(m, fitted, True, self._trim())
        self._commit(m, before)
        if not m.any():
            m[:] = before
        self._smart = None
        return int((before ^ m).sum())

    def grow(self, n):
        m = self.mask()
        if m is None:
            return
        self.push_undo()
        before = m.copy()
        m[:] = S.grow_mask(m, n)
        self._commit(m, before)

    def clear_mask(self):
        m = self.mask()
        if m is None or not m.any():
            return
        self.push_undo()
        m[:] = False

    # ---- figures and depth -------------------------------------------------------------------------
    def figure_depth(self, i):
        p = self.par.planes[i]
        gr = self.par.ground()
        if p.fixed or gr is None or p.mask is None or not p.mask.any():
            return p.depth
        c0, cx, cy = gr
        x, y = self.feet(i)
        return float(min(1.0, max(0.0, c0 + cx * x + cy * y)))

    def feet(self, i):
        """Where figure i meets the ground: its feet line, or the middle of its lowest row."""
        p = self.par.planes[i]
        if p.feet is not None:
            return p.feet
        if p.mask is None or not p.mask.any():
            return (W / 2, H - 1)
        ys, xs = np.nonzero(p.mask)
        yb = int(ys.max())
        return (float(xs[ys == yb].mean()), float(yb))

    def set_feet(self, x, y, start=True):
        L = self.layer()
        if L.kind != "figure":
            return
        if start:
            self.push_undo()
        L.obj.feet = (int(round(min(W - 1, max(0, x)))), int(round(min(H - 1, max(0, y)))))

    def set_depth(self, v):
        """Depth by hand (0 = as far as the sky, 1 = nearest): for things that don't stand on the ground."""
        L = self.layer()
        if L.kind == "figure":
            self.push_undo()
            L.obj.depth = float(min(1.0, max(0.0, v)))
            L.obj.fixed = True

    def set_depth_by_feet(self, on):
        L = self.layer()
        if L.kind == "figure" and L.obj.fixed == bool(on):
            self.push_undo()
            if not on:
                L.obj.depth = self.figure_depth(L.index)        # start from where it stood
            L.obj.fixed = not on

    def set_pprm(self, k, v):
        self.push_undo()
        self.par.prm[k] = float(v)

    # ---- the ground grid ------------------------------------------------------------------------------
    def grid_default(self):
        self.push_undo()
        x0, x1, y0, y1 = 20, 300, 150, 230
        inset = (x1 - x0) * 0.3
        self.par.grid = [(x0, y1), (x1, y1), (x1 - inset, y0), (x0 + inset, y0)]
        if not any(p.mask is not None and p.mask.any() for p in self.par.planes):
            self.par.prm.setdefault("ground", 1.0)

    def grid_corner(self, k, x, y, start=True):
        if not self.par.grid:
            return
        if start:
            self.push_undo()
        g = list(self.par.grid)
        g[k] = (float(x), float(y))
        self.par.grid = g

    def grid_move(self, dx, dy, start=True):
        if not self.par.grid:
            return
        if start:
            self.push_undo()
        self.par.grid = [(x + dx, y + dy) for x, y in self.par.grid]

    def grid_turn(self, deg):
        if not self.par.grid:
            return
        self.push_undo()
        cx = sum(p[0] for p in self.par.grid) / 4; cy = sum(p[1] for p in self.par.grid) / 4
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        self.par.grid = [(cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c) for x, y in self.par.grid]

    def horizon(self):
        """The horizon as (y at x=0, y at x=W), or None: where the ground's depth reaches 0."""
        gr = self.par.ground()
        if gr is None:
            return None
        c0, cx, cy = gr
        if abs(cy) < 1e-9:
            return None
        return (-(c0) / cy, -(c0 + cx * W) / cy)

    # ---- water lines -------------------------------------------------------------------------------
    def set_water_line(self, pts):
        L = self.layer()
        if L.kind != "water" or len(pts) < 2:
            return False
        self.push_undo()
        pts = [(int(round(min(W - 1, max(0, x)))), int(round(min(H - 1, max(0, y))))) for x, y in pts][:J.WPTS_MAX]
        L.obj.lines = [pts]
        return True

    def set_water_mode(self, mode):
        L = self.layer()
        if L.kind == "water" and L.obj.mode != mode:
            self.push_undo()
            L.obj.mode = mode

    def set_wprm(self, k, v):
        L = self.layer()
        if L.kind == "water":
            self.push_undo()
            L.obj.prm[k] = float(v)

    # ---- sways ---------------------------------------------------------------------------------------
    def sway_guide(self, pts, replace=False):
        """Hair: a lock stroke root → tip (adds one). Cloth: the waistband's two ends. Pendant: its pin."""
        L = self.layer()
        if L.kind != "sway":
            return False
        sw = L.obj
        pts = [(float(x), float(y)) for x, y in pts]
        need = {"hair": 2, "cloth": 2, "pendant": 1}[sw.kind]
        if len(pts) < need:
            return False
        self.push_undo()
        if sw.kind == "hair":
            g = [pts[0]]
            for p in pts[1:]:                                    # thin to a few px apart
                if math.hypot(p[0] - g[-1][0], p[1] - g[-1][1]) >= 4:
                    g.append(p)
            if len(g) < 2:
                g.append(pts[-1])
            if replace:
                sw.guides = []
            sw.guides = (sw.guides + [g[:16]])[-SW.GUIDE_MAX:]
        elif sw.kind == "cloth":
            if replace or not sw.guides or len(pts) >= 2:
                sw.guides = [[pts[0], pts[-1]]] + sw.guides[1:]           # the waistband (marked folds kept)
            else:
                sw.guides = (sw.guides + [[pts[0]]])[:SW.GUIDE_MAX]    # a fold you mark: one point on it
        else:
            sw.guides = [[pts[0]]]
        return True

    def suggest_locks(self):
        """Hair: lock strokes traced for you (each fall of it, rooted at the neck). Replaces the strokes."""
        L = self.layer()
        if L.kind != "sway" or L.obj.kind != "hair" or not L.obj.mask.any():
            return 0
        self.push_undo()
        B = self.body_mask()
        L.obj.guides = SW.suggest_locks(L.obj.mask, B if B.any() else None)
        return len(L.obj.guides)

    def hips(self):
        """The active skirt's hip outline: as lined up, else fitted (shown for you to drag into place)."""
        L = self.layer()
        if L.kind != "sway" or L.obj.kind != "cloth" or L.obj.mask is None or not L.obj.mask.any():
            return []
        if len(L.obj.arm) == SW.HIP_POINTS:
            return L.obj.arm
        B = self.body_mask()
        return SW.fit_hips(L.obj.mask, SW.rest_body(B, [s.mask for s in self.sways]) if B.any() else None,
                           L.obj.guides[0] if L.obj.guides else None)

    def set_hip(self, i, x, y, start=True):
        L = self.layer()
        pts = list(self.hips())
        if not pts:
            return
        if start:
            self.push_undo()
        pts[i] = (float(x), float(y))
        L.obj.arm = pts

    def fit_hips(self):
        L = self.layer()
        if L.kind == "sway" and L.obj.kind == "cloth":
            self.push_undo()
            L.obj.arm = []

    def set_back(self, back):
        """The skirt's figure is seen from behind (the bikini bottom then covers her seat)."""
        L = self.layer()
        if L.kind == "sway" and L.obj.kind == "cloth" and L.obj.back != bool(back):
            self.push_undo()
            L.obj.back = bool(back)

    def load_under(self, src):
        """What's really under the active sway - your own layer of the same picture (e.g. her in just the
        bikini, same pose): its fill is cut from this instead of made up. Used here to make the fill only:
        never saved (the .jig carries the fill it makes). src: a path or an RGB image."""
        L = self.layer()
        if L.kind != "sway":
            return False
        if isinstance(src, (str, Path)):
            import mg_convert
            im = mg_convert.prepare_image(Path(src), "cover")          # (laid out exactly as the photo was)
            img = np.asarray(im.convert("RGB"), np.uint8)
        else:
            img = np.asarray(src, np.uint8)
        if img.shape[:2] != (H, W):
            return False
        self.push_undo()
        L.obj.under = img.copy()
        return True

    def clear_under(self):
        L = self.layer()
        if L.kind == "sway" and L.obj.under is not None:
            self.push_undo()
            L.obj.under = None

    def undo_guide(self):
        """The last lock / fold / pin off."""
        L = self.layer()
        if L.kind == "sway" and L.obj.guides:
            self.push_undo()
            L.obj.guides = L.obj.guides[:-1]
            return True
        return False

    def clear_guides(self):
        L = self.layer()
        if L.kind == "sway" and L.obj.guides:
            self.push_undo()
            L.obj.guides = []

    def set_sway(self, what, v):
        L = self.layer()
        if L.kind != "sway":
            return
        self.push_undo()
        if what == "kind":
            L.obj.kind = v
            L.obj.guides = []
        elif what == "on":
            L.obj.on = v
        else:
            L.obj.prm[what] = float(v)

    def body_mask(self):
        """What sways rest on and collide with: the figures named by them (all figures if none named)."""
        names = {s.on for s in self.sways if s.on}
        B = np.zeros((H, W), bool)
        for L in self.layers():
            if L.kind == "figure" and L.mask is not None and (not names or L.name in names):
                B |= L.mask
        return B

    # ---- depth maps ---------------------------------------------------------------------------------
    def scene_depth(self):
        """The scene's depth (H, W) in 0..1, near = 1: the ground from the grid (0 above the horizon),
        each figure at its depth on top, nearest wins."""
        gr = self.par.ground()
        Y, X = np.mgrid[:H, :W].astype(np.float32)
        d = np.clip(gr[0] + gr[1] * X + gr[2] * Y, 0, 1) if gr is not None else np.zeros((H, W), np.float32)
        best = np.full((H, W), -1.0, np.float32)
        for i, p in enumerate(self.par.planes):
            if p.mask is None or not p.mask.any():
                continue
            v = self.figure_depth(i)
            take = p.mask & (v > best)
            best[take] = v
        return np.where(best >= 0, best, d).astype(np.float32)

    def set_depth_map(self, arr, near_is_white=True, source=""):
        """A depth map from anywhere (any size, grayscale or RGB, 8 or 16 bit) → aligned to the grid."""
        a = np.asarray(arr, np.float64)
        if a.ndim == 3:
            a = a[..., :3].mean(-1)
        from PIL import Image
        x0, y0, cw, ch = getattr(self.ed, "content_box", None) or (0, 0, W, H)
        a = np.asarray(Image.fromarray(a.astype(np.float32)).resize((cw, ch), Image.BILINEAR), np.float64)
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        a = np.clip((a - lo) / max(1e-9, hi - lo), 0, 1)
        if not near_is_white:
            a = 1 - a
        full = np.zeros((H, W)); full[y0:y0 + ch, x0:x0 + cw] = a
        self.depth_raw = full.astype(np.float32)
        self.depth_in = self._align(self.depth_raw)
        self.depth_note = source
        return self.depth_in

    def _ground_pixels(self):
        gr = self.par.ground()
        if gr is None:
            return None
        Y, X = np.mgrid[:H, :W]
        g = gr[0] + gr[1] * X + gr[2] * Y
        m = g > 0.02
        for p in self.par.planes:
            if p.mask is not None:
                m &= ~S.grow_mask(p.mask, 3)
        for s in self.sways:
            if s.mask is not None:
                m &= ~s.mask
        return m if m.sum() > 200 else None

    def _align(self, d):
        """Depth models give depth with no scale or offset: fit it to the grid's ground (least squares,
        twice, dropping the worst fifth), so its numbers mean the same as the scene's."""
        m = self._ground_pixels()
        if m is None:
            return d
        gr = self.par.ground()
        Y, X = np.mgrid[:H, :W]
        g = (gr[0] + gr[1] * X + gr[2] * Y)[m]
        v = d[m].astype(np.float64)
        keep = np.ones(len(v), bool)
        a = b = 0.0
        for _ in range(2):
            A = np.stack([v[keep], np.ones(keep.sum())], 1)
            (a, b), *_ = np.linalg.lstsq(A, g[keep], rcond=None)
            r = np.abs(a * v + b - g)
            keep = r <= np.quantile(r, 0.8)
        return np.clip(a * d + b, 0, 1.2).astype(np.float32)

    def map_depths(self):
        """Per figure: (name, depth from the scene, depth from the map or None, disagree?)."""
        out = []
        for i, p in enumerate(self.par.planes):
            dm = None
            if self.depth_in is not None and p.mask is not None and p.mask.any():
                dm = float(np.median(self.depth_in[p.mask]))
            ds = self.figure_depth(i)
            out.append((p.name or f"Figure {i + 1}", ds, dm, dm is not None and abs(dm - ds) > 0.12))
        return out

    def use_map_depths(self):
        """Figures take their depth from the map (and stop using the feet line)."""
        if self.depth_in is None:
            return 0
        self.push_undo()
        n = 0
        for p in self.par.planes:
            if p.mask is not None and p.mask.any():
                p.depth = float(np.clip(np.median(self.depth_in[p.mask]), 0, 1))
                p.fixed = True
                n += 1
        if self.par.grid:                 # with a grid the Core2 stands planes on it; a map depth needs it off
            self.par.prm["ground"] = self.par.prm.get("ground", 1.0)
        return n

    def fit_grid_from_depth(self):
        """The ground grid from the depth map: the ground is where the map's inverse depth is a flat
        slope (RANSAC plane fit on the lower part, figures left out); the horizon is where that slope
        reaches the far end of the map. Returns the horizon's height in the middle, or None."""
        d = self.depth_raw
        if d is None:
            return None
        rng = np.random.default_rng(7)
        Y, X = np.mgrid[:H, :W]
        cand = (Y > H * 0.45)
        for p in self.par.planes:
            if p.mask is not None:
                cand &= ~S.grow_mask(p.mask, 3)
        ys, xs = np.nonzero(cand)
        if len(ys) < 300:
            return None
        v = d[ys, xs].astype(np.float64)
        best, best_n = None, -1
        for _ in range(300):
            k = rng.choice(len(ys), 3, replace=False)
            A = np.stack([np.ones(3), xs[k], ys[k]], 1).astype(np.float64)
            try:
                c = np.linalg.solve(A, v[k])
            except np.linalg.LinAlgError:
                continue
            if c[2] <= 1e-5:                                 # inverse depth must grow toward the bottom
                continue
            n = int((np.abs(c[0] + c[1] * xs + c[2] * ys - v) < 0.03).sum())
            if n > best_n:
                best, best_n = c, n
        if best is None or best_n < 200:
            return None
        inl = np.abs(best[0] + best[1] * xs + best[2] * ys - v) < 0.03
        A = np.stack([np.ones(inl.sum()), xs[inl], ys[inl]], 1).astype(np.float64)
        c0, cx, cy = np.linalg.lstsq(A, v[inl], rcond=None)[0]
        far = float(np.percentile(d, 3))                     # the far end of the map: the horizon's value
        yl, yr = (far - c0) / cy, (far - c0 - cx * W) / cy
        yl, yr = float(np.clip(yl, -80, H * 0.9)), float(np.clip(yr, -80, H * 0.9))
        # a grid that keeps that horizon: near edge at the bottom, far edge 45 % of the way up to it
        self.push_undo()
        yb = H - 6.0
        hy = lambda x: yl + (yr - yl) * x / W
        vx = W / 2
        far_y = lambda x: hy(x) + 0.45 * (yb - hy(x))
        nl, nr = (30.0, yb), (W - 30.0, yb)
        def toward_vp(p):
            t = (p[1] - far_y(p[0])) / max(1e-6, p[1] - hy(vx))
            return (p[0] + (vx - p[0]) * t, p[1] + (hy(vx) - p[1]) * t)
        self.par.grid = [nl, nr, toward_vp(nr), toward_vp(nl)]
        return hy(W / 2)

    # ---- import / export ------------------------------------------------------------------------------
    def _type_word(self, L):
        return {"figure": "figure", "water": "water", "sway": "sway"}[L.kind]

    def export_masks(self, folder):
        """photo.png, one Layer.type.png per mask (white = in), depth.png (16-bit, near = white),
        scene.json. All at 4× the Core2's size so they're comfortable to paint on."""
        from PIL import Image
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        S_ = EXPORT_SCALE
        Image.fromarray(self.image).resize((W * S_, H * S_), Image.LANCZOS).save(folder / "photo.png")
        files = []
        for L in self.layers():
            if not L.has_mask or L.mask is None:
                continue
            fn = f"{L.name}.{self._type_word(L)}.png"
            Image.fromarray((L.mask * 255).astype(np.uint8)).resize((W * S_, H * S_), Image.NEAREST).save(folder / fn)
            files.append({"name": L.name, "type": self._type_word(L), "file": fn})
        self._save16(self.scene_depth(), folder / "depth.png")
        meta = {"photo": "photo.png", "size": [W * S_, H * S_], "core2_size": [W, H], "layers": files,
                "depth": {"file": "depth.png", "near": "white", "bits": 16},
                "note": "White = in the layer, black = out. Name a new file Layer.type.png (figure / water / sway) to import it as that type."}
        (folder / "scene.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return [f["file"] for f in files] + ["photo.png", "depth.png", "scene.json"]

    @staticmethod
    def read_mask_file(path):
        """Any image → (H, W) bool at the Core2's size. A transparent PNG uses its alpha; otherwise
        brightness (≥ half = in). 4:3 files fill the frame; others fill the photo's own box."""
        from PIL import Image
        im = Image.open(path)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            a = np.asarray(im.convert("RGBA"))[..., 3]
            v = a if a.min() < 250 else np.asarray(im.convert("L"))
        elif im.mode in ("I;16", "I;16B", "I"):
            a = np.asarray(im, np.float64); v = (a / max(1.0, a.max()) * 255)
        else:
            v = np.asarray(im.convert("L"))
        return v

    def _to_frame(self, v):
        from PIL import Image
        v = np.asarray(v, np.float32)
        h, w = v.shape
        if abs(w / h - W / H) < 0.02:
            box = (0, 0, W, H)
        else:
            box = getattr(self.ed, "content_box", None) or (0, 0, W, H)
        x0, y0, cw, ch = box
        small = np.asarray(Image.fromarray(v).resize((cw, ch), Image.BOX), np.float32)
        out = np.zeros((H, W), bool)
        out[y0:y0 + ch, x0:x0 + cw] = small >= 127.5
        return out

    def import_plan(self, paths):
        """What an import would do, file by file: [{file, name, type, action, note}]. type None = ask."""
        rows = []
        for p in paths:
            p = Path(p)
            if p.suffix.lower() not in (".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg", ".webp"):
                continue
            stem = p.stem
            if stem.lower() in ("photo", "depth"):
                continue
            name, _, typ = stem.rpartition(".")
            if not name:
                name, typ = stem, ""
            typ = typ.lower() if typ.lower() in ("figure", "water", "sway") else None
            name = _safe_name(name.replace(".", " ")) or "Layer"
            L = self.by_name(name, typ) if typ else self.by_name(name)
            if L is not None and L.has_mask:
                rows.append({"file": str(p), "name": name, "type": L.kind, "action": "replace", "note": ""})
            else:
                rows.append({"file": str(p), "name": name, "type": typ, "action": "new",
                             "note": "" if typ else "no type in the name: pick one"})
        return rows

    def import_apply(self, rows, protect=True):
        """Do the import (one undo step). Rows with type None are skipped."""
        if not rows:
            return 0
        self.push_undo()
        n = 0
        for r in rows:
            typ = r.get("type")
            if typ not in ("figure", "water", "sway"):
                continue
            m = self._to_frame(self.read_mask_file(r["file"]))
            L = self.by_name(r["name"], typ) if r.get("action") == "replace" else None
            if L is None:
                if typ == "figure":
                    if len(self.par.planes) >= D.PLANE_MAX:
                        continue
                    self.par.planes.append(D.Plane(depth=0.6, mask=np.zeros((H, W), bool), name=self.unique_name(r["name"])))
                    L = self.layer(("figure", len(self.par.planes) - 1))
                elif typ == "water":
                    if len(self.ed.waters) >= J.WATER_MAX:
                        continue
                    w = J.Water(); w.name = self.unique_name(r["name"]); self.ed.waters.append(w)
                    L = self.layer(("water", len(self.ed.waters) - 1))
                else:
                    if len(self.sways) >= SW.SWAY_MAX:
                        continue
                    self.sways.append(SW.Sway("hair", np.zeros((H, W), bool), [], {}, self.unique_name(r["name"]), ""))
                    L = self.layer(("sway", len(self.sways) - 1))
            if protect and L.kind in ("water",):
                for M in self.layers():
                    if M.kind == "figure" and self.protect.get(M.name, True):
                        m &= ~M.mask
            L.obj.mask[:] = m
            n += 1
        return n

    def export_depth(self, path):
        from PIL import Image
        self._save16(self.scene_depth() if self.depth_in is None else self.depth_in, path)
        return path

    @staticmethod
    def _save16(d, path):
        from PIL import Image
        big = np.asarray(Image.fromarray(np.clip(d, 0, 1).astype(np.float32)).resize((W * EXPORT_SCALE, H * EXPORT_SCALE), Image.BILINEAR))
        Image.fromarray((np.clip(big, 0, 1) * 65535).astype(np.uint16)).save(path)

    def import_depth(self, path, near_is_white=True):
        from PIL import Image
        im = Image.open(path)
        a = np.asarray(im, np.float64) if im.mode.startswith("I") else np.asarray(im.convert("L"), np.float64)
        return self.set_depth_map(a, near_is_white, Path(path).name)

    # ---- preview ---------------------------------------------------------------------------------------
    def sim(self, with_sways=True):
        body = self.body_mask()
        par = self.par_with_sways() if self.par.any() else None
        if par is not None:
            for i, p in enumerate(par.planes):          # (depths as the Core2 will compute them)
                if p.mask is not None and p.mask.any():
                    p.depth = self.figure_depth(i)
        waters = []
        for w in self.ed.waters:
            w2 = w.copy()
            w2.grid = None if not self.par.grid else [tuple(p) for p in self.par.grid]
            waters.append(w2)
        sim = J.Sim(self.ed.zones, self.image, waters, par, self.ed.hangs)
        if with_sways:
            sim.add_sways([s for s in self.sways if s.ok()], body if body.any() else None)
        return sim
