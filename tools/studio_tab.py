"""
Mantis Gallery — "Dynamics Studio" tab for mg_convert.py.

Open an .mg straight off the SD card and build its scene as named layers:

    Scene   Sky (stays put), Ground (the ground grid: horizon + perspective), Figures (things standing
            in the scene: their own depth, protected from other layers' painting, what hair rests on)
    Jiggle  soft-body zones       Water   still / river / ocean       Depth   how the layers slide
    Sway    hair locks, cloth, pendants

Every tool works on the highlighted layer, and the line under the picture always says which.

Select   W Smart · B Brush · E Erase · L Lasso   (Add / Subtract; hold Alt to subtract)
Fix      Tighter / Looser (the last Smart pick) · Refine edges (snap the whole mask to the real edges)
         · Grow / Shrink 1 px · [ ] brush size
Handles  the layer's own: Grid corners (Ground), Feet line (Figure), Shore / Flow line (Water), Lock
         stroke / Waistband / Pin (Sway), Move (Jiggle)
Keys     Ctrl+Z / Ctrl+Y undo / redo · Ctrl+S save · Space live preview · Enter finishes a lasso or line
         · Esc cancels it · Delete removes the active layer
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import jiggle_lab as J
import mg_depth as D
import mg_select as S
import mg_sway as SW
import studio_model as SM

try:
    from PIL import Image, ImageTk
except ImportError:  # mg_convert already requires Pillow
    Image = ImageTk = None

import numpy as np

BG, PANEL, PANEL2, FG, MUTED = "#1a1a1a", "#222222", "#2a2a2a", "#e8e8e8", "#a8a8a8"
TEAL, LIME, YEL, PLUM, RED = "#007373", "#b4ff00", "#ffe000", "#5d005d", "#ff7070"
FONT, FONT_B, FONT_S = ("Segoe UI", 10), ("Segoe UI", 10, "bold"), ("Segoe UI", 9)
SETTINGS = Path.home() / ".mantis_studio.json"

TOOLS = (("cut", "Cut out", "C"), ("smart", "Smart", "W"), ("brush", "Brush", "B"), ("erase", "Erase", "E"), ("lasso", "Lasso", "L"))
MASK_TOOLS = tuple(t for t, _, _ in TOOLS)
HANDLES = {  # the active layer's own handle tools: (tool, label, key)
    "ground": [("grid", "Grid corners", "G")],
    "figure": [("feet", "Feet line", "F")],
    "water": [("line", "Shore line", "S")],
    "jiggle": [("zone", "Move / size", "M")],
    "sway": [("guide", "Lock stroke", "K")],
    "sky": [],
}
HANDLES_SWAY = {"hair": ("guide", "Lock stroke", "K"), "cloth": ("guide", "Waistband", "K"), "pendant": ("guide", "Pin", "K")}
SUBTABS = ("Scene", "Jiggle", "Water", "Depth", "Sway")
TAB_OF = {"sky": "Scene", "ground": "Scene", "figure": "Scene", "water": "Water", "jiggle": "Jiggle", "sway": "Sway"}


def _hex(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def load_settings():
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(d):
    try:
        SETTINGS.write_text(json.dumps(d, indent=1), encoding="utf-8")
    except OSError:
        pass


class Tabs(tk.Frame):
    """Sub-tabs as in the design: a row of flat buttons, the chosen one underlined in lime. Answers the bits
    of ttk.Notebook the tab uses (select / index / the changed event)."""

    def __init__(self, master):
        super().__init__(master, bg=BG)
        self.bar = tk.Frame(self, bg=BG)
        self.bar.pack(fill="x")
        self.body = tk.Frame(self, bg=BG)
        self.body.pack(fill="both", expand=True)
        self.pages, self.btns, self.lines, self.cur, self._cb = [], [], [], 0, []

    def add(self, frame, text=""):
        i = len(self.pages)
        col = tk.Frame(self.bar, bg=BG)
        col.pack(side="left", fill="x", expand=True, padx=1)
        b = tk.Button(col, text=text, command=lambda: self.select(i), relief="flat", bd=0, pady=6,
                      font=("Segoe UI", 9, "bold"), cursor="hand2")
        b.pack(fill="x")
        ln = tk.Frame(col, height=2, bg="#333")
        ln.pack(fill="x")
        self.pages.append(frame); self.btns.append(b); self.lines.append(ln)
        self._paint()

    def _paint(self):
        for j, (b, ln) in enumerate(zip(self.btns, self.lines)):
            on = j == self.cur
            b.configure(bg=PANEL2 if on else BG, fg=FG if on else MUTED, activebackground=PANEL2, activeforeground=FG)
            ln.configure(bg=LIME if on else "#333")
        for j, f in enumerate(self.pages):
            if j == self.cur:
                f.pack(fill="both", expand=True)
            else:
                f.pack_forget()

    def select(self, i=None):
        if i is None:
            return self.cur
        if i != self.cur:
            self.cur = i
            self._paint()
            for cb in self._cb:
                cb(None)
        return self.cur

    def index(self, what):
        return int(what)

    def bind(self, seq, fn, add=None):
        if seq == "<<NotebookTabChanged>>":
            self._cb.append(fn)
        else:
            super().bind(seq, fn, add)


class StudioTab(tk.Frame):
    def __init__(self, master, root=None):
        super().__init__(master, bg=BG)
        self.root = root or self.winfo_toplevel()
        self.st = SM.Studio()
        self.zoom = 2
        self.tool = "cut"
        self.sub = tk.BooleanVar(value=False)          # Subtract mode
        self.brush_r = 6
        self.protect_v = tk.BooleanVar(value=True)
        self.preview = tk.BooleanVar(value=False)
        self.show_layers = tk.BooleanVar(value=True)
        self.show_handles = tk.BooleanVar(value=True)
        self.show_depth = tk.BooleanVar(value=False)
        self.tilt = tk.DoubleVar(value=0.0)
        self.lean = tk.DoubleVar(value=0.0)
        self.dance = tk.BooleanVar(value=False)
        self.bpm = tk.StringVar(value="120")
        self.slowmo = tk.StringVar(value="1")
        self.status = tk.StringVar(value="")
        self.sim = None
        self._sim_t = 0.0
        self._motion = None
        self._dance_amp, self._dance_sign, self._next_beat = 0.0, 1, 0.0
        self._lean_prev = 0.0
        self._last_tick = time.perf_counter()
        self._pts = []                                 # lasso / line / stroke points in progress
        self._marks = None                             # smart stroke in progress
        self._cut = None                               # Cut out: the active layer's keep / not strokes so far
        self._last_xy = None
        self._drag = None
        self._alt = False
        self._photo = None
        self._img_item = None
        self._syncing = False
        self._settings = load_settings()
        self._build()
        self._bind_keys()
        self._refresh_all()
        self.after(33, self._tick)

    # ================================================================================== widgets
    def _btn(self, parent, text, cmd, kind="", **kw):
        bgc = {"": PANEL2, "teal": TEAL, "plum": PLUM, "on": LIME, "sub": "#5a1818"}[kind]
        fgc = "#000" if kind == "on" else FG
        b = tk.Button(parent, text=text, command=cmd, bg=bgc, fg=fgc, activebackground=TEAL, activeforeground=FG,
                      relief="flat", bd=0, padx=kw.pop("padx", 8), pady=kw.pop("pady", 3), font=kw.pop("font", FONT_S),
                      highlightthickness=0, cursor="hand2", **kw)
        return b

    def _label(self, parent, text, fg=MUTED, font=FONT_S, **kw):
        return tk.Label(parent, text=text, fg=fg, bg=kw.pop("bg", parent.cget("bg")), font=font, anchor="w", justify="left", **kw)

    def _section(self, parent, title):
        box = tk.Frame(parent, bg=PANEL, highlightbackground="#333", highlightthickness=1)
        box.pack(fill="x", padx=2, pady=(0, 8))
        tk.Label(box, text=title, fg=YEL, bg=PANEL, font=("Segoe UI", 9, "bold"), anchor="w").pack(fill="x", padx=8, pady=(6, 2))
        inner = tk.Frame(box, bg=PANEL)
        inner.pack(fill="x", padx=8, pady=(0, 8))
        return inner

    def _slider(self, parent, label, lo, hi, res, get, set_, help_=""):
        r = tk.Frame(parent, bg=PANEL)
        r.pack(fill="x")
        lbl = tk.Label(r, text=label, fg=FG, bg=PANEL, font=FONT_S, width=18, anchor="w", wraplength=128, justify="left")
        lbl.pack(side="left")
        var = tk.DoubleVar(value=get())

        def moved(v):                       # (Tk also calls this when the value is set from code: ignore no-ops)
            if self._syncing or abs(float(v) - float(get())) < res * 0.5:
                return
            set_(float(v))
            self._changed(soft=True)
        sc = tk.Scale(r, from_=lo, to=hi, resolution=res, orient="horizontal", variable=var, showvalue=True,
                      bg=PANEL, fg=FG, troughcolor="#3a3a3a", highlightthickness=0, length=170, font=FONT_S, command=moved)
        sc.pack(side="left", fill="x", expand=True)
        if help_:
            for w in (lbl, sc):
                w.bind("<Enter>", lambda e, h=help_, l=label: self._hint(f"{l}: {h}"))
        return var

    def _check(self, parent, text, var, cmd):
        return tk.Checkbutton(parent, text=text, variable=var, command=cmd, bg=parent.cget("bg"), fg=FG,
                              selectcolor="#333", activebackground=parent.cget("bg"), activeforeground=FG,
                              font=FONT_S, anchor="w", highlightthickness=0)

    def _scrolled(self, parent):
        """A vertically scrolling panel; returns the inner frame."""
        outer = tk.Frame(parent, bg=BG)
        outer.pack(fill="both", expand=True)
        cv = tk.Canvas(outer, bg=BG, highlightthickness=0, width=380)
        sb = tk.Scrollbar(outer, orient="vertical", command=cv.yview)
        inner = tk.Frame(cv, bg=BG)
        inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=inner, anchor="nw", width=380)
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        inner.bind("<Enter>", lambda e: cv.bind_all("<MouseWheel>", lambda ev: cv.yview_scroll(-1 if ev.delta > 0 else 1, "units")))
        inner.bind("<Leave>", lambda e: cv.unbind_all("<MouseWheel>"))
        return inner

    # ================================================================================== layout
    def _build(self):
        # ---- top bar
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=8, pady=(8, 4))
        self._btn(top, "Open .mg…", self.open_dialog, "teal").pack(side="left")
        self._btn(top, "Save to SD", self.save, "plum").pack(side="left", padx=4)
        self._btn(top, "Revert", self.revert).pack(side="left")
        mb = tk.Menubutton(top, text="Masks ▾", bg=TEAL, fg=FG, activebackground=TEAL, relief="flat", font=FONT_S, padx=8, pady=3)
        mm = tk.Menu(mb, tearoff=0, bg=PANEL, fg=FG, activebackground=TEAL)
        mm.add_command(label="Export masks…  (one PNG per layer + the photo)", command=self.export_masks)
        mm.add_command(label="Import masks…  (PNGs, matched to layers by name)", command=self.import_masks)
        mb.configure(menu=mm)
        mb.pack(side="left", padx=(12, 4))
        db = tk.Menubutton(top, text="Depth ▾", bg=TEAL, fg=FG, activebackground=TEAL, relief="flat", font=FONT_S, padx=8, pady=3)
        dm = tk.Menu(db, tearoff=0, bg=PANEL, fg=FG, activebackground=TEAL)
        dm.add_command(label="Import depth map…", command=self.import_depth)
        dm.add_command(label="Generate depth…  (Depth Anything V2, run here)", command=self.generate_depth)
        dm.add_command(label="Export depth map…", command=self.export_depth)
        dm.add_separator()
        dm.add_command(label="Fit the ground grid from the depth map", command=self.fit_grid)
        dm.add_command(label="Figures take their depth from the map", command=self.use_map_depths)
        db.configure(menu=dm)
        db.pack(side="left")
        self._btn(top, "↶ Undo", self.undo).pack(side="left", padx=(12, 2))
        self._btn(top, "↷ Redo", self.redo).pack(side="left")
        self.file_lbl = self._label(top, "", MUTED)
        self.file_lbl.pack(side="left", padx=12)
        zf = tk.Frame(top, bg=BG)
        zf.pack(side="right")
        self._label(zf, "Zoom").pack(side="left")
        self.zoom_btns = {}
        for z in (1, 2, 3):
            b = self._btn(zf, f"{z}×", lambda z=z: self.set_zoom(z))
            b.pack(side="left", padx=1)
            self.zoom_btns[z] = b

        # ---- body: layers | canvas | inspector
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=8)
        self._build_layers(body)
        self._build_center(body)
        self._build_right(body)
        self._build_preview_bar()

    def _build_layers(self, body):
        left = tk.Frame(body, bg=PANEL, width=230, highlightbackground="#333", highlightthickness=1)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        hd = tk.Frame(left, bg=PANEL)
        hd.pack(fill="x", padx=8, pady=(8, 4))
        self._label(hd, "SCENE LAYERS", YEL, ("Segoe UI", 9, "bold")).pack(side="left")
        add = tk.Menubutton(hd, text="+ Add ▾", bg=TEAL, fg=FG, activebackground=TEAL, relief="flat", font=FONT_S, padx=6)
        am = tk.Menu(add, tearoff=0, bg=PANEL, fg=FG, activebackground=TEAL)
        am.add_command(label="Figure (a person / thing standing in the scene)", command=lambda: self.add("figure"))
        am.add_command(label="Water", command=lambda: self.add("water"))
        am.add_command(label="Jiggle zone", command=lambda: self.add("jiggle"))
        am.add_separator()
        am.add_command(label="Hair lock (sway)", command=lambda: self.add("hair"))
        am.add_command(label="Cloth (sway)", command=lambda: self.add("cloth"))
        am.add_command(label="Pendant (sway)", command=lambda: self.add("pendant"))
        add.configure(menu=am)
        add.pack(side="right")
        self.layer_box = tk.Frame(left, bg=PANEL)
        self.layer_box.pack(fill="x", padx=6)
        ops = tk.Frame(left, bg=PANEL)
        ops.pack(fill="x", padx=6, pady=6)
        self._btn(ops, "Rename", self.rename).pack(side="left")
        self._btn(ops, "▲", lambda: self._do(lambda: self.st.move_active(-1))).pack(side="left", padx=2)
        self._btn(ops, "▼", lambda: self._do(lambda: self.st.move_active(1))).pack(side="left")
        self._btn(ops, "Delete", self.delete_layer).pack(side="right")
        self._label(left, "Everything you paint goes into the highlighted layer. Click a layer to work on it; "
                          "its tab opens on the right. The eye hides its tint.", MUTED, wraplength=210).pack(
            side="bottom", fill="x", padx=8, pady=8)

    def _build_center(self, body):
        mid = tk.Frame(body, bg=BG)
        mid.pack(side="left", fill="both", expand=True, padx=8)
        strip = tk.Frame(mid, bg=PANEL, highlightbackground="#333", highlightthickness=1)
        strip.pack(fill="x")
        r1 = tk.Frame(strip, bg=PANEL)
        r1.pack(fill="x", padx=6, pady=(5, 2))
        self._label(r1, "SELECT", YEL, ("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 4))
        self.tool_btns = {}
        for t, label, key in TOOLS:
            b = self._btn(r1, f"{label}  {key}", lambda t=t: self.set_tool(t))
            b.pack(side="left", padx=1)
            self.tool_btns[t] = b
        self.add_btn = self._btn(r1, "＋ Add", lambda: self.set_sub(False))
        self.add_btn.pack(side="left", padx=(10, 1))
        self.sub_btn = self._btn(r1, "− Subtract", lambda: self.set_sub(True))
        self.sub_btn.pack(side="left", padx=1)
        self._label(r1, " hold Alt = subtract", MUTED).pack(side="left")
        r2 = tk.Frame(strip, bg=PANEL)
        r2.pack(fill="x", padx=6, pady=(2, 5))
        self._label(r2, "FIX", YEL, ("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 4))
        self.fix_btns = []
        for text, fn in (("Tighter", lambda: self.smart_step(-1)), ("Looser", lambda: self.smart_step(1)),
                         ("Refine edges", self.refine), ("Grow 1 px", lambda: self._do(lambda: self.st.grow(1))),
                         ("Shrink 1 px", lambda: self._do(lambda: self.st.grow(-1))), ("Clear", self.clear_mask)):
            b = self._btn(r2, text, fn)
            b.pack(side="left", padx=1)
            self.fix_btns.append(b)
        self.handle_box = tk.Frame(r2, bg=PANEL)
        self.handle_box.pack(side="left", padx=(12, 0))
        self.canvas = tk.Canvas(mid, width=J.IMG_W * self.zoom, height=J.IMG_H * self.zoom, bg="#000",
                                highlightthickness=1, highlightbackground="#444", cursor="crosshair")
        self.canvas.pack(pady=6)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag_ev)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Motion>", self._hover)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda e: self._wheel(e, 1))
        self.canvas.bind("<Button-5>", lambda e: self._wheel(e, -1))
        self.canvas.bind("<Double-Button-1>", lambda e: self.finish_pts())
        st = tk.Frame(mid, bg="#101010", highlightbackground=LIME, highlightthickness=1)
        st.pack(fill="x")
        self.status_lbl = tk.Label(st, textvariable=self.status, fg=FG, bg="#101010", font=FONT, anchor="w",
                                   justify="left", wraplength=J.IMG_W * 2)
        self.status_lbl.pack(fill="x", padx=8, pady=5)
        self.hint_lbl = tk.Label(mid, text="", fg=MUTED, bg=BG, font=FONT_S, anchor="w", justify="left",
                                 wraplength=J.IMG_W * 2)
        self.hint_lbl.pack(fill="x")

    def _build_right(self, body):
        right = tk.Frame(body, bg=BG, width=400)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self.nb = Tabs(right)
        self.nb.pack(fill="both", expand=True)
        self.tabs = {}
        for name in SUBTABS:
            f = tk.Frame(self.nb.body, bg=BG)
            self.nb.add(f, text=name)
            self.tabs[name] = self._scrolled(f)
        self.nb.bind("<<NotebookTabChanged>>", self._tab_changed)
        self._build_scene_tab(self.tabs["Scene"])
        self._build_jiggle_tab(self.tabs["Jiggle"])
        self._build_water_tab(self.tabs["Water"])
        self._build_depth_tab(self.tabs["Depth"])
        self._build_sway_tab(self.tabs["Sway"])

    # ---- Scene tab: Sky / Ground / Figure inspectors ------------------------------------------------
    def _build_scene_tab(self, p):
        self.scene_title = self._label(p, "", YEL, ("Segoe UI", 10, "bold"))
        self.scene_title.pack(fill="x", padx=4, pady=(6, 2))
        self.scene_box = tk.Frame(p, bg=BG)
        self.scene_box.pack(fill="x")
        f = self._section(p, "WHAT'S IN THIS SCENE")
        self.summary_lbl = self._label(f, "", FG, bg=PANEL, wraplength=340)
        self.summary_lbl.pack(fill="x")
        self.next_lbl = self._label(f, "", LIME, bg=PANEL, wraplength=340)
        self.next_lbl.pack(fill="x", pady=(4, 0))

    def _fill_scene_box(self):
        for w in self.scene_box.winfo_children():
            w.destroy()
        L = self.st.layer()
        kind = L.kind if L.kind in ("sky", "ground", "figure") else None
        if kind == "sky":
            self.scene_title.configure(text="SKY  ·  FAR")
            f = self._section(self.scene_box, "SKY")
            self._label(f, "Everything above the horizon. It stays put when you tilt (the handheld 'window' feel); "
                           "the ground and the figures slide against it. Fit the Ground grid to set where it starts.",
                        FG, bg=PANEL, wraplength=340).pack(fill="x")
        elif kind == "ground" or kind is None:
            self.scene_title.configure(text="GROUND  ·  THE SCENE'S FLOOR")
            f = self._section(self.scene_box, "GROUND GRID")
            self._label(f, "Fit it once: drag the 4 corners until the squares lie flat on the ground. It tells every "
                           "effect where the horizon is and how fast the ground recedes. Wheel over it turns it.",
                        FG, bg=PANEL, wraplength=340).pack(fill="x")
            row = tk.Frame(f, bg=PANEL); row.pack(fill="x", pady=(6, 0))
            if self.st.par.grid:
                self._btn(row, "⟲ Turn", lambda: self._do(lambda: self.st.grid_turn(-1))).pack(side="left")
                self._btn(row, "⟳ Turn", lambda: self._do(lambda: self.st.grid_turn(1))).pack(side="left", padx=2)
                self._btn(row, "Reset", lambda: self._do(self.st.grid_default)).pack(side="left", padx=2)
                self._btn(row, "Remove grid", lambda: self._do(self.st.delete_active)).pack(side="left", padx=2)
            else:
                self._btn(row, "Add the ground grid", lambda: (self._do(self.st.grid_default), self.set_tool("grid")), "teal").pack(side="left")
            self._label(f, "Used by: Depth (how far each pixel is) · Figures (where they stand) · Water (wave size "
                           "and direction). The dashed line is the horizon.", MUTED, bg=PANEL, wraplength=340).pack(fill="x", pady=(6, 0))
        else:
            fig = L.obj
            self.scene_title.configure(text=f"{L.name.upper()}  ·  FIGURE")
            f = self._section(self.scene_box, "FIGURE")
            self._label(f, "A thing standing in the scene: its own depth layer, a collider for hair and cloth, "
                           "and protected from other layers' painting.", FG, bg=PANEL, wraplength=340).pack(fill="x")
            self.protect_v.set(self.st.protect.get(L.name, True))
            self._check(f, "Protect from other layers' painting", self.protect_v, self._protect_toggled).pack(fill="x", pady=(6, 0))
            d = self.st.figure_depth(L.index)
            f1 = self._section(self.scene_box, "ITS DEPTH")
            self.v_feet = tk.BooleanVar(value=bool(self.st.par.grid) and not fig.fixed)
            rb = tk.Radiobutton(f1, text="Stands on the ground — depth from its feet line", variable=self.v_feet, value=True,
                                bg=PANEL, fg=FG, selectcolor="#333", font=FONT_S, activebackground=PANEL,
                                command=lambda: self._do(lambda: self.st.set_depth_by_feet(True)))
            rb.pack(anchor="w")
            if not self.st.par.grid:
                rb.configure(state="disabled")
            tk.Radiobutton(f1, text="Set by hand (floating, hanging, or its feet don't meet the ground)", variable=self.v_feet,
                           value=False, bg=PANEL, fg=FG, selectcolor="#333", font=FONT_S, activebackground=PANEL,
                           command=lambda: self._do(lambda: self.st.set_depth_by_feet(False))).pack(anchor="w")
            if self.st.par.grid and not fig.fixed:
                src = "its feet line" if fig.feet is not None else "its lowest point (drag a Feet line to change it)"
                self._label(f1, f"Depth {d:.2f}  from {src}", LIME, bg=PANEL, wraplength=340).pack(fill="x", pady=(4, 0))
                row = tk.Frame(f1, bg=PANEL); row.pack(fill="x", pady=(4, 0))
                self._btn(row, "Feet line tool  F", lambda: self.set_tool("feet")).pack(side="left")
                if fig.feet is not None:
                    self._btn(row, "Use its lowest point", lambda: self._do(lambda: self._clear_feet())).pack(side="left", padx=4)
            else:
                self._slider(f1, "Depth", 0, 1, 0.01, lambda: fig.depth, lambda v: self.st.set_depth(v),
                             "0 = as far as the sky (stays put), 1 = nearest (slides most). Compare with the Depth tab's list.")
            f2 = self._section(self.scene_box, "HOW TO SELECT IT")
            self._label(f2, "1  Cut out: paint a few strokes on her (head to feet, the arms).\n"
                            "2  Alt-paint (or − Subtract) on what she isn't, where it touches her:\n"
                            "    the background by her sides, the ground between her feet.\n"
                            "3  Every stroke re-cuts. Her hair and clothes count as her.\n"
                            "4  A thin outline left behind when she slides? Grow 1 px.\n"
                            "5  Feet line: drag where she touches the ground.", FG, bg=PANEL).pack(fill="x")

    def _clear_feet(self):
        L = self.st.layer()
        if L.kind == "figure":
            self.st.push_undo()
            L.obj.feet = None

    def _protect_toggled(self):
        L = self.st.layer()
        if L.kind == "figure":
            self.st.push_undo()
            self.st.protect[L.name] = bool(self.protect_v.get())
            self._refresh_all()

    # ---- Jiggle tab ------------------------------------------------------------------------------------
    def _build_jiggle_tab(self, p):
        f = self._section(p, "ADD A ZONE")
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x")
        self.shape_btns = []
        for i, (_, label, _, _) in enumerate(J.SHAPES):
            b = self._btn(r, f"{label}", lambda i=i: self.arm(i))
            b.pack(side="left", padx=(0, 4))
            self.shape_btns.append(b)
        self._label(f, "Pick a shape, then click the photo to place it. Drag a zone to move it, its small handle "
                       "to turn it, the wheel to size it.", MUTED, bg=PANEL, wraplength=340).pack(fill="x", pady=(4, 0))
        self.jig_box = tk.Frame(p, bg=BG)
        self.jig_box.pack(fill="x")
        f = self._section(self.jig_box, "ZONE")
        self.v_mat = tk.IntVar(value=J.NATURAL)
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x", pady=2)
        self._label(r, "Feel", FG, bg=PANEL, width=10).pack(side="left")
        for i, m in enumerate(J.MATERIALS):
            tk.Radiobutton(r, text=m[1], variable=self.v_mat, value=i, bg=PANEL, fg=FG, selectcolor="#333", font=FONT_S,
                           activebackground=PANEL, command=lambda: self._zone_do(lambda: self.st.ed.set_material(self.v_mat.get()))).pack(side="left")
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x", pady=2)
        self._label(r, "Body part", FG, bg=PANEL, width=10).pack(side="left")
        self.v_preset = tk.StringVar(value="")
        pcb = ttk.Combobox(r, textvariable=self.v_preset, values=J.PRESET_NAMES, width=26, state="readonly")
        pcb.pack(side="left")
        pcb.bind("<<ComboboxSelected>>", lambda e: self._zone_do(lambda: self.st.ed.apply_preset(self.v_preset.get())))
        self.v_size = self._slider(f, "Size", J.R_MIN, J.R_MAX, 1, lambda: self._z("r", J.R_DEFAULT),
                                   lambda v: self._zone_set("r", v))
        self.v_rot = self._slider(f, "Rotation", -180, 180, 1, lambda: ((self._z("rot", 0) + 180) % 360) - 180,
                                  lambda v: self._zone_set("rot", int(round(v)) % 360))
        self.v_tip = tk.BooleanVar(value=False)
        self._check(f, "Control point (line it up over the apex)", self.v_tip,
                    lambda: self._zone_do(lambda: self.st.ed.set_tip(self.v_tip.get()))).pack(fill="x", pady=(4, 0))
        self.form_lbl = self._label(f, "", YEL, bg=PANEL, wraplength=340)
        self.form_lbl.pack(fill="x")
        f = self._section(self.jig_box, "FINE TUNING  (grey = automatic · ↺ resets one)")
        self.tune_rows = {}
        info = {t[0]: t for t in J.TUNES}
        for group, keys in J.TUNE_GROUPS:
            self._label(f, group, YEL, bg=PANEL).pack(fill="x", pady=(6, 0))
            for key in keys:
                _, label, lo, hi, help_ = info[key]
                self.tune_rows[key] = self._tune_row(f, key, label, lo, hi, help_)
        self._btn(f, "Reset all to automatic", lambda: self._zone_do(self._reset_tuning)).pack(anchor="w", pady=(6, 0))
        f = self._section(self.jig_box, "RESPONSE  (after a quick sideways shove)")
        self.resp = tk.Canvas(f, width=340, height=100, bg="#111", highlightthickness=0)
        self.resp.pack()
        self.resp_lbl = self._label(f, "", MUTED, bg=PANEL)
        self.resp_lbl.pack(fill="x")
        self.jig_empty = self._label(p, "", MUTED, wraplength=360)

    def _z(self, attr, default):
        z = self.st.ed.zone
        return getattr(z, attr) if z else default

    def _zone_set(self, attr, v):
        z = self.st.ed.zone
        if z is None:
            return
        self.st.push_undo()
        setattr(z, attr, v)

    def _zone_do(self, fn):
        if self.st.ed.zone is None:
            return
        self.st.push_undo()
        fn()
        self._changed()

    def _reset_tuning(self):
        z = self.st.ed.zone
        if z:
            z.tune = {k: v for k, v in z.tune.items() if k == "side"}

    def _tune_row(self, parent, key, label, lo, hi, help_):
        r = tk.Frame(parent, bg=PANEL)
        r.pack(fill="x")
        lbl = tk.Label(r, text=label, fg=MUTED, bg=PANEL, width=18, anchor="w", font=FONT_S)
        lbl.pack(side="left")
        var = tk.DoubleVar(value=lo)
        res = 1.0 if key in ("side", "down") else 0.01
        def moved(v, k=key):
            z = self.st.ed.zone
            if self._syncing or z is None or (k in z.tune and abs(float(v) - z.tuned(k, 0.0)) < res * 0.5):
                return
            if k not in z.tune and abs(float(v) - getattr(self, "_tune_shown", {}).get(k, 1e9)) < res * 0.5:
                return                      # (just showing the automatic value)
            self._zone_do(lambda: self.st.ed.set_tune(k, float(v)))
        s = tk.Scale(r, from_=lo, to=hi, resolution=res, orient="horizontal", variable=var, showvalue=True, font=FONT_S,
                     bg=PANEL, fg=FG, troughcolor="#3a3a3a", highlightthickness=0, length=140, command=moved)
        s.pack(side="left")
        self._btn(r, "↺", lambda k=key: self._zone_do(lambda: self.st.ed.set_tune(k, None)), padx=4, pady=0).pack(side="left", padx=2)
        for w in (lbl, s):
            w.bind("<Enter>", lambda e, h=help_, l=label: self._hint(f"{l}: {h}"))
        return {"var": var, "label": lbl}

    # ---- Water tab ------------------------------------------------------------------------------------
    def _build_water_tab(self, p):
        self.water_title = self._label(p, "", YEL, ("Segoe UI", 10, "bold"))
        self.water_title.pack(fill="x", padx=4, pady=(6, 2))
        self.water_box = tk.Frame(p, bg=BG)
        self.water_box.pack(fill="x")

    def _fill_water_box(self):
        for w in self.water_box.winfo_children():
            w.destroy()
        L = self.st.layer()
        if L.kind != "water":
            self.water_title.configure(text="WATER")
            f = self._section(self.water_box, "NO WATER SELECTED")
            self._label(f, "Pick a water layer on the left, or add one.", FG, bg=PANEL).pack(fill="x")
            self._btn(f, "+ Add water", lambda: self.add("water"), "teal").pack(anchor="w", pady=(6, 0))
            return
        w = L.obj
        self.water_title.configure(text=f"{L.name.upper()}  ·  WATER")
        f = self._section(self.water_box, "KIND")
        row = tk.Frame(f, bg=PANEL); row.pack(fill="x")
        for m, (key, label, *_rest) in enumerate(J.WMODES):
            self._btn(row, label, lambda m=m: self._do(lambda: self.st.set_water_mode(m)), "on" if w.mode == m else "").pack(side="left", padx=1)
        self._label(f, ("Perspective: the ground grid ✓" if self.st.par.grid else
                        "Perspective: none — fit the Ground grid (Scene tab) for waves that recede"),
                    LIME if self.st.par.grid else YEL, bg=PANEL, wraplength=340).pack(fill="x", pady=(6, 0))
        if w.mode != J.W_STILL:
            what = "Shore line" if w.mode == J.W_OCEAN else "Flow line"
            have = bool(w.lines and w.lines[0])
            self._label(f, f"{what}: {'set ✓' if have else 'not set — use the ' + what + ' tool (S): click along it, Enter'}",
                        LIME if have else YEL, bg=PANEL, wraplength=340).pack(fill="x", pady=(2, 0))
            row = tk.Frame(f, bg=PANEL); row.pack(fill="x", pady=(4, 0))
            self._btn(row, f"{what} tool  S", lambda: self.set_tool("line")).pack(side="left")
            if have:
                self._btn(row, "Clear it", lambda: self._do(self._clear_line)).pack(side="left", padx=4)
        f = self._section(self.water_box, "SETTINGS")
        for key, label, lo, hi, dflt, help_ in J.WPARAMS:
            if key in ("horizon",):
                continue
            self._slider(f, label, lo, hi, 0.01 if hi <= 5 else 1, lambda k=key: w.get(k),
                         lambda v, k=key: self.st.set_wprm(k, v), help_)
        f = self._section(self.water_box, "WHAT EACH HANDLE DOES")
        self._label(f, "Shore line (ocean): where waves arrive · Flow line (river): which way and how fast it runs · "
                       "Dashed line: the horizon, from the ground grid. Figures are protected: painting water can't "
                       "cover them.", FG, bg=PANEL, wraplength=340).pack(fill="x")

    def _clear_line(self):
        L = self.st.layer()
        if L.kind == "water":
            self.st.push_undo()
            L.obj.lines = []

    # ---- Depth tab -------------------------------------------------------------------------------------
    def _build_depth_tab(self, p):
        self._label(p, "DEPTH  ·  NEAREST FIRST", YEL, ("Segoe UI", 10, "bold")).pack(fill="x", padx=4, pady=(6, 2))
        self.depth_box = tk.Frame(p, bg=BG)
        self.depth_box.pack(fill="x")

    def _fill_depth_box(self):
        for w in self.depth_box.winfo_children():
            w.destroy()
        st = self.st
        f = self._section(self.depth_box, "LAYERS")
        rows = []
        for i, pl in enumerate(st.par.planes):
            rows.append((st.figure_depth(i), pl.name or f"Figure {i + 1}", "feet line" if pl.feet is not None else
                         ("lowest point" if st.par.grid else "set by hand"), SM.FIG_COLS[i % 4], ("figure", i)))
        rows.sort(key=lambda r: -r[0])
        if st.par.grid:
            rows.append((None, "Ground", "the grid (0 at the horizon … 1 nearest)", SM.GROUND_COL, ("ground", 0)))
        rows.append((0.0, "Sky", "above the horizon", SM.SKY_COL, ("sky", 0)))
        md = {n: (dm, bad) for n, _, dm, bad in st.map_depths()}
        for d, name, src, col, key in rows:
            r = tk.Frame(f, bg=PANEL); r.pack(fill="x", pady=1)
            tk.Label(r, text="■", fg=col, bg=PANEL, font=FONT_S).pack(side="left")
            b = tk.Label(r, text=name, fg=FG, bg=PANEL, font=FONT_B if key == st.active else FONT_S, width=12, anchor="w", cursor="hand2")
            b.pack(side="left")
            b.bind("<Button-1>", lambda e, k=key: self.select_layer(k))
            tk.Label(r, text="0–1" if d is None else f"{d:.2f}", fg=LIME, bg=PANEL, font=FONT_S, width=5).pack(side="left")
            self._label(r, src, MUTED, bg=PANEL).pack(side="left")
            if name in md and md[name][0] is not None:
                tk.Label(r, text=f"map {md[name][0]:.2f}" + (" ⚠" if md[name][1] else ""), fg=YEL if md[name][1] else MUTED,
                         bg=PANEL, font=FONT_S).pack(side="right")
        self._label(f, "Each figure stands where its feet meet the ground grid. Select a figure and drag its Feet line "
                       "if its lowest point isn't where it stands.", MUTED, bg=PANEL, wraplength=340).pack(fill="x", pady=(4, 0))
        if st.depth_in is not None:
            self._label(f, f"Depth map: {st.depth_note or 'imported'} — lined up with the grid. ⚠ = the map disagrees "
                           "with the feet line.", MUTED, bg=PANEL, wraplength=340).pack(fill="x", pady=(2, 0))
        f = self._section(self.depth_box, "PARALLAX")
        P = st.par
        for key, label, lo, hi, dflt, help_ in D.PPARAMS:
            if key in ("ground", "invert"):
                continue
            self._slider(f, label, lo, hi, 0.01 if hi <= 3 else 0.5, lambda k=key: P.get(k), lambda v, k=key: st.set_pprm(k, v), help_)
        self.ground_v = tk.BooleanVar(value=P.get("ground") >= 0.5)
        cb = self._check(f, "The ground slides too (needs the grid; off = only figures move)", self.ground_v,
                         lambda: (self._do(lambda: st.set_pprm("ground", 1.0 if self.ground_v.get() else 0.0))))
        cb.pack(fill="x", pady=(4, 0))
        if not P.grid:
            cb.configure(state="disabled")
        self.invert_v = tk.BooleanVar(value=P.get("invert") >= 0.5)
        self._check(f, "Reverse direction", self.invert_v,
                    lambda: self._do(lambda: st.set_pprm("invert", 1.0 if self.invert_v.get() else 0.0))).pack(fill="x")
        self._label(f, "Tilt with the preview controls below: turned / leaned, or the Tilt L/R and Sweep motions.",
                    MUTED, bg=PANEL, wraplength=340).pack(fill="x", pady=(4, 0))
        f = self._section(self.depth_box, "DEPTH MAPS")
        row = tk.Frame(f, bg=PANEL); row.pack(fill="x")
        self._btn(row, "Import…", self.import_depth).pack(side="left")
        self._btn(row, "Generate…", self.generate_depth).pack(side="left", padx=2)
        self._btn(row, "Export…", self.export_depth).pack(side="left")
        row = tk.Frame(f, bg=PANEL); row.pack(fill="x", pady=(4, 0))
        self._btn(row, "Fit grid from map", self.fit_grid).pack(side="left")
        self._btn(row, "Figures use map depth", self.use_map_depths).pack(side="left", padx=2)

    # ---- Sway tab -------------------------------------------------------------------------------------
    def _build_sway_tab(self, p):
        self.sway_title = self._label(p, "", YEL, ("Segoe UI", 10, "bold"))
        self.sway_title.pack(fill="x", padx=4, pady=(6, 2))
        self.sway_box = tk.Frame(p, bg=BG)
        self.sway_box.pack(fill="x")

    def _fill_sway_box(self):
        for w in self.sway_box.winfo_children():
            w.destroy()
        L = self.st.layer()
        if L.kind != "sway":
            self.sway_title.configure(text="SWAY")
            f = self._section(self.sway_box, "NO SWAYING THING SELECTED")
            self._label(f, "Pick one on the left, or add one:", FG, bg=PANEL).pack(fill="x")
            row = tk.Frame(f, bg=PANEL); row.pack(fill="x", pady=(6, 0))
            for k, lab in SM.SWAY_KINDS:
                self._btn(row, "+ " + lab.split(" (")[0], lambda k=k: self.add(k), "teal").pack(side="left", padx=1)
            self._sway_help(self.sway_box)
            return
        sw = L.obj
        self.sway_title.configure(text=f"{L.name.upper()}  ·  SWAY")
        f = self._section(self.sway_box, "KIND")
        row = tk.Frame(f, bg=PANEL); row.pack(fill="x")
        for k, lab in SM.SWAY_KINDS:
            self._btn(row, lab, lambda k=k: self._do(lambda: self.st.set_sway("kind", k)), "on" if sw.kind == k else "").pack(side="left", padx=1)
        figs = [M.name for M in self.st.layers() if M.kind == "figure"]
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x", pady=(6, 0))
        self._label(r, "Rests on / collides with", FG, bg=PANEL).pack(side="left")
        self.v_on = tk.StringVar(value=sw.on or "(nothing)")
        cb = ttk.Combobox(r, textvariable=self.v_on, values=["(nothing)"] + figs, width=14, state="readonly")
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self._do(lambda: self.st.set_sway("on", "" if self.v_on.get() == "(nothing)" else self.v_on.get())))
        tool = HANDLES_SWAY[sw.kind]
        ng = len(sw.guides)
        what = {"hair": f"Lock strokes: {ng}", "cloth": "Waistband: " + (f"set ✓, folds marked: {ng - 1}" if ng else "not set"),
                "pendant": "Pin: " + ("set ✓" if ng else "not set")}[sw.kind]
        self._label(f, what + ("" if ng else f" — use the {tool[1]} tool ({tool[2]})"), LIME if ng else YEL, bg=PANEL).pack(fill="x", pady=(6, 0))
        row = tk.Frame(f, bg=PANEL); row.pack(fill="x", pady=(4, 0))
        self._btn(row, f"{tool[1]} tool  {tool[2]}", lambda: self.set_tool("guide")).pack(side="left")
        if sw.kind == "hair":
            self._btn(row, "Suggest locks", lambda: self._do(self.st.suggest_locks), "teal").pack(side="left", padx=4)
        if sw.kind == "cloth":
            hr = tk.Frame(f, bg=PANEL); hr.pack(fill="x", pady=(6, 0))
            self._label(hr, "Her hips under it: " + ("lined up ✓" if len(sw.arm) == SW.HIP_POINTS else "fitted — check it"),
                        LIME if len(sw.arm) == SW.HIP_POINTS else YEL, bg=PANEL).pack(side="left")
            self._btn(hr, "Hips tool  H", lambda: self.set_tool("hips")).pack(side="left", padx=4)
            self._btn(hr, "Re-fit", lambda: self._do(self.st.fit_hips)).pack(side="left")
        if ng:
            self._btn(row, "Remove last", lambda: self._do(self.st.undo_guide)).pack(side="left", padx=4)
            self._btn(row, "Clear", lambda: self._do(self.st.clear_guides)).pack(side="left", padx=4)
        f = self._section(self.sway_box, "MOTION")
        for key, label, lo, hi, dflt in SW.SPARAMS:
            if key == "pleat" and sw.kind != "cloth":
                continue
            self._slider(f, label, lo, hi, 0.01, lambda k=key: sw.get(k), lambda v, k=key: self.st.set_sway(k, v))
        self._sway_help(self.sway_box)

    def _sway_help(self, parent):
        f = self._section(parent, "THE THREE KINDS")
        self._label(f, "Hair — Cut out the hair (paint on it; Alt-paint on her face and the background), then "
                       "Suggest locks, or draw each lock with Lock stroke: from the neck's height (where it leaves "
                       "her head) to its tip. Hair above that lies on her head and stays.\nCloth — Cut out the skirt "
                       "(Alt-paint her arms, legs and the ground), then Waistband: click its two ends, then Hips: drag "
                       "the pink outline onto her hips and legs (the skirt won't swing in through it). Optional: click "
                       "the fold lines.\nPendant — Cut it out, then Pin: click where it hangs from.\nHair and cloth "
                       "move with the figure they rest on.", FG, bg=PANEL, wraplength=340).pack(fill="x")

    # ---- preview bar -------------------------------------------------------------------------------------
    def _build_preview_bar(self):
        bar = tk.Frame(self, bg=PANEL, highlightbackground="#333", highlightthickness=1)
        bar.pack(fill="x", padx=8, pady=(4, 8))
        r1 = tk.Frame(bar, bg=PANEL); r1.pack(fill="x", padx=8, pady=(6, 2))
        self.prev_btn = self._btn(r1, "▶ Live preview", self._toggle_preview, font=FONT_B)
        self.prev_btn.pack(side="left", padx=(0, 6))
        for kind, label in J.MOTIONS:
            self._btn(r1, label, lambda k=kind: self.start_motion(k), padx=6).pack(side="left", padx=1)
        self._btn(r1, "● Record…", self.record_preview, "plum").pack(side="right")
        r2 = tk.Frame(bar, bg=PANEL); r2.pack(fill="x", padx=8, pady=(2, 6))
        self._label(r2, "Hold the Core2:  turned", bg=PANEL).pack(side="left")
        tk.Scale(r2, from_=-180, to=180, orient="horizontal", variable=self.tilt, showvalue=False, length=150,
                 bg=PANEL, troughcolor="#3a3a3a", highlightthickness=0, sliderrelief="flat", bd=0,
                 command=lambda v: self.tilt_lbl.configure(text=f"{float(v):+.0f}°")).pack(side="left", padx=4)
        self.tilt_lbl = self._label(r2, "0°", LIME, bg=PANEL, width=5)
        self.tilt_lbl.pack(side="left")
        self._label(r2, "leaned", bg=PANEL).pack(side="left", padx=(6, 0))
        tk.Scale(r2, from_=-90, to=90, orient="horizontal", variable=self.lean, showvalue=False, length=110,
                 bg=PANEL, troughcolor="#3a3a3a", highlightthickness=0, sliderrelief="flat", bd=0,
                 command=lambda v: self.lean_lbl.configure(text=f"{float(v):+.0f}°")).pack(side="left", padx=4)
        self.lean_lbl = self._label(r2, "0°", LIME, bg=PANEL, width=5)
        self.lean_lbl.pack(side="left")
        self._btn(r2, "Upright", lambda: (self.tilt.set(0), self.lean.set(0), self.tilt_lbl.configure(text="0°"),
                                          self.lean_lbl.configure(text="0°")), padx=6).pack(side="left", padx=6)
        self.dance_btn = self._btn(r2, "Dance", lambda: (self.dance.set(not self.dance.get()), self._dance_toggled(),
                                                         self._paint_preview_bar()), padx=6)
        self.dance_btn.pack(side="left", padx=(10, 2))
        tk.Spinbox(r2, from_=60, to=200, increment=1, textvariable=self.bpm, width=4, bg=PANEL2, fg=FG,
                   buttonbackground="#333", font=FONT_S, relief="flat").pack(side="left")
        self._label(r2, "BPM", bg=PANEL).pack(side="left", padx=(2, 10))
        self._label(r2, "Speed", bg=PANEL).pack(side="left", padx=(0, 2))
        self.speed_btns = {}
        for v, l in (("1", "1×"), ("0.5", "½×"), ("0.25", "¼×")):
            b = self._btn(r2, l, lambda v=v: (self.slowmo.set(v), self._paint_preview_bar()), padx=6)
            b.pack(side="left", padx=1)
            self.speed_btns[v] = b
        self._label(r2, "Show:", bg=PANEL).pack(side="left", padx=(10, 2))
        self.show_btns = {}
        for text, var in (("Layers", self.show_layers), ("Handles", self.show_handles), ("Depth map", self.show_depth)):
            b = self._btn(r2, text, lambda var=var: (var.set(not var.get()), self._paint_preview_bar(), self._render()), padx=6)
            b.pack(side="left", padx=1)
            self.show_btns[text] = (b, var)
        self._paint_preview_bar()

    def _paint_preview_bar(self):
        def paint(b, on):
            b.configure(bg=LIME if on else PANEL2, fg="#000" if on else FG)
        paint(self.prev_btn, self.preview.get())
        paint(self.dance_btn, self.dance.get())
        for v, b in self.speed_btns.items():
            paint(b, self.slowmo.get() == v)
        for b, var in self.show_btns.values():
            paint(b, var.get())

    def _toggle_preview(self):
        self.preview.set(not self.preview.get())
        self._preview_toggled()
        self._paint_preview_bar()

    def record_preview(self):
        """Save the preview, played through a set of motions, as a GIF or MP4 (to look at, or share)."""
        if self.st.image is None:
            return
        path = filedialog.asksaveasfilename(parent=self, title="Record the preview", defaultextension=".gif",
                                            filetypes=[("Animated GIF", "*.gif"), ("MP4 video", "*.mp4")],
                                            initialfile=(Path(self.st.ed.mg_path).stem if self.st.ed.mg_path else "preview") + "_preview.gif")
        if not path:
            return
        self._hint("Recording the preview (tilt, sway, shake, walk, dance)… it takes a little while.")
        st = self.st

        def work():
            try:
                import mg_preview as MP
                sim = st.sim()
                fr = list(MP.frames(sim, seconds=17.0, fps=15))
                MP.write(path, fr, fps=15)
                msg = f"Saved {len(fr)} frames: {path}"
            except Exception as ex:                          # (shown, never raised on the Tk thread)
                msg = f"Couldn't record: {ex}"
            self._job_done = lambda: self._hint(msg)
        threading.Thread(target=work, daemon=True).start()

    # ================================================================================== state → widgets
    def _hint(self, text):
        self.hint_lbl.configure(text=text)

    def _refresh_all(self):
        self._syncing = True
        try:
            self._refresh_layers()
            self._refresh_strip()
            self._fill_scene_box()
            self._fill_water_box()
            self._fill_depth_box()
            self._fill_sway_box()
            self._sync_jiggle()
            self._refresh_summary()
            self._refresh_status()
            for z, b in self.zoom_btns.items():
                b.configure(bg=LIME if z == self.zoom else PANEL2, fg="#000" if z == self.zoom else FG)
            p = self.st.ed.mg_path
            self.file_lbl.configure(text=(f"{p}  ·  {len(self.st.layers())} layers" + ("  ·  unsaved changes" if self.st.dirty else ""))
                                    if p else "No photo open")
        finally:
            self._syncing = False
        self._render()

    def _refresh_layers(self):
        for w in self.layer_box.winfo_children():
            w.destroy()
        for L in self.st.layers():
            on = L.key == self.st.active
            row = tk.Frame(self.layer_box, bg="#2f3a1a" if on else PANEL, highlightbackground=LIME if on else PANEL,
                           highlightthickness=1, cursor="hand2")
            row.pack(fill="x", pady=1)
            eye = tk.Label(row, text="◌" if L.key in self.st.hidden else "◉", fg=MUTED, bg=row.cget("bg"), font=FONT_S, cursor="hand2")
            eye.pack(side="left", padx=(4, 2))
            eye.bind("<Button-1>", lambda e, k=L.key: self._toggle_eye(k))
            tk.Label(row, text="■", fg=self.st.color(L), bg=row.cget("bg"), font=FONT_S).pack(side="left")
            col = tk.Frame(row, bg=row.cget("bg"))
            col.pack(side="left", fill="x", padx=4, pady=2)
            nm = tk.Label(col, text=L.name if L.name == SM.KIND_LABEL[L.kind] else f"{L.name}  · {SM.KIND_LABEL[L.kind]}", fg=FG, bg=row.cget("bg"),
                          font=FONT_B if on else FONT_S, anchor="w")
            nm.pack(fill="x")
            dt = tk.Label(col, text=self.st.detail(L), fg=MUTED, bg=row.cget("bg"), font=("Segoe UI", 8), anchor="w")
            dt.pack(fill="x")
            for w in (row, col, nm, dt):
                w.bind("<Button-1>", lambda e, k=L.key: self.select_layer(k))
                w.bind("<Double-Button-1>", lambda e, k=L.key: (self.select_layer(k), self.rename()))

    def _refresh_strip(self):
        L = self.st.layer()
        can = L.has_mask and not self.preview.get()
        for t, b in self.tool_btns.items():
            on = self.tool == t
            b.configure(state="normal" if can else "disabled", bg=LIME if on and can else PANEL2, fg="#000" if on and can else FG)
        sub = self.sub.get()
        self.add_btn.configure(state="normal" if can else "disabled", bg=LIME if can and not sub else PANEL2, fg="#000" if can and not sub else FG)
        self.sub_btn.configure(state="normal" if can else "disabled", bg="#a83232" if can and sub else PANEL2)
        for b in self.fix_btns:
            b.configure(state="normal" if can else "disabled")
        for w in self.handle_box.winfo_children():
            w.destroy()
        hs = self._handles(L)
        if hs:
            self._label(self.handle_box, "THIS LAYER", YEL, ("Segoe UI", 8, "bold"), bg=PANEL).pack(side="left", padx=(0, 4))
        for t, label, key in hs:
            on = self.tool == t
            self._btn(self.handle_box, f"{label}  {key}", lambda t=t: self.set_tool(t), "on" if on else "").pack(side="left", padx=1)

    def _handles(self, L):
        if L.kind == "water":
            if L.obj.mode == J.W_STILL:
                return []
            return [("line", "Shore line" if L.obj.mode == J.W_OCEAN else "Flow line", "S")]
        if L.kind == "sway":
            t = HANDLES_SWAY[L.obj.kind]
            return [("guide", t[1], t[2])] + ([("hips", "Hips", "H")] if L.obj.kind == "cloth" else [])
        return HANDLES.get(L.kind, [])

    def _refresh_summary(self):
        st = self.st
        n = {k: sum(1 for L in st.layers() if L.kind == k) for k in SM.KINDS}
        self.summary_lbl.configure(text=f"Sky · Ground grid {'fitted' if st.par.grid else 'not fitted'} · {n['figure']} figure(s) · "
                                        f"{n['water']} water · {n['jiggle']} jiggle zone(s) · {n['sway']} swaying thing(s)")
        if st.image is None:
            nxt = "Open an .mg photo to start."
        elif not st.par.grid:
            nxt = "Next: Ground ▸ fit the grid (Grid corners tool)."
        elif n["figure"] == 0:
            nxt = "Next: + Add ▸ Figure for each person / thing standing in the scene."
        elif not any(L.mask is not None and L.mask.any() for L in st.layers() if L.kind == "figure"):
            nxt = "Next: select the figure — Smart: click it."
        else:
            nxt = "Next: Water, Jiggle zones or Sway — then ▶ Live preview."
        self.next_lbl.configure(text=nxt)

    def _refresh_status(self):
        L = self.st.layer()
        if self.st.image is None:
            self.status.set("Open an .mg photo (Open .mg…) from the SD card's /images folder.")
            return
        if self.preview.get():
            self.status.set("▶ Live preview — click the picture to poke it · use the motions and turned / leaned below · "
                            "untick Live preview to edit.")
            return
        name, kind = L.name, SM.KIND_LABEL[L.kind]
        prot = [M.name for M in self.st.layers() if M.kind == "figure" and M.key != L.key and self.st.protect.get(M.name, True)]
        ptxt = (f"  Protected (never painted): {', '.join(prot)}." if prot and L.kind in ("figure", "water") else "")
        t = self.tool
        if t in MASK_TOOLS and L.has_mask:
            sub = self.sub.get() or self._alt or t == "erase"
            verb = ("NOT-stroke for" if sub else "KEEP-stroke for") if t == "cut" else ("SUBTRACT from" if sub else "ADD to")
            how = {"cut": "paint a stroke ON it; Alt-paint (or Subtract) on what it ISN'T — the face, sky, ground. "
                          "Each stroke refines the cut.",
                   "smart": "click it, or drag over a sample. Alt-click removes.",
                   "brush": f"paint (size {self.brush_r}; [ ] to change).", "erase": f"paint it away (size {self.brush_r}).",
                   "lasso": "click around it, double-click or Enter to finish."}[t]
            self.status.set(f"{dict((a, b) for a, b, _ in TOOLS)[t]} · {verb} “{name}” ({kind}) — {how}{ptxt}")
        elif t == "grid" and not self.st.par.grid:
            self.status.set("Ground grid — no grid yet: click the picture (or Add the ground grid) to put one down, then "
                            "drag its corners until the squares lie flat on the ground.")
        elif t == "grid":
            self.status.set("Grid corners — drag the 4 corners until the squares lie flat on the ground; drag inside to move "
                            "it; wheel turns it. The dashed line is the horizon.")
        elif t == "feet":
            self.status.set(f"Feet line on “{name}” — click / drag where it touches the ground. Its depth comes from there.")
        elif t == "line":
            what = "shore" if L.kind == "water" and L.obj.mode == J.W_OCEAN else "flow"
            self.status.set(f"{what.title()} line on “{name}” — click along the {what}, double-click or Enter to finish. "
                            + ("Waves run to it." if what == "shore" else "The river runs along it, first point upstream."))
        elif t == "hips" and L.kind == "sway":
            self.status.set(f"Hips under “{name}” — drag the pink points onto her: waist, widest hip, each leg's two "
                            "edges at the hem, between the legs. The skirt can't swing in through it; behind the "
                            "skirt inside it you see skirt or leg, outside it the scene. The white dashed area round the "
                            "point between the legs never shows skin: a white polka-dot bikini bottom shows there.")
        elif t == "guide" and L.kind == "sway":
            self.status.set({"hair": f"Lock stroke on “{name}” — drag from where the lock is tied to its tip. Each stroke = one lock.",
                             "cloth": (f"Folds on “{name}” — click on each fold line to split the pleats there (optional)."
                                       if L.obj.guides and len(L.obj.guides[0]) >= 2 else
                                       f"Waistband on “{name}” — click its left end, then its right end."),
                             "pendant": f"Pin on “{name}” — click where it hangs from."}[L.obj.kind])
        elif t == "zone" or L.kind == "jiggle":
            ar = self.st.ed.armed
            self.status.set(f"Click the photo to place: {J.SHAPES[ar][1]}" if ar >= 0 else
                            (f"Jiggle zone “{name}” — drag to move, drag the ⟳ handle to turn, wheel to size, drag its "
                             "circle to line up the control point." if L.kind == "jiggle" else "Pick a shape in the Jiggle tab, then click the photo."))
        elif L.kind == "sky":
            self.status.set("Sky — nothing to paint: it's everything above the horizon (set by the Ground grid).")
        elif L.kind == "ground":
            self.status.set("Ground — use Grid corners to fit the grid." if self.st.par.grid else
                            "Ground — no grid yet: Add the ground grid (Scene tab), then drag its corners.")
        else:
            self.status.set(f"“{name}” ({kind}) — pick a tool.")

    def _sync_jiggle(self):
        z = self.st.ed.zone
        for i, b in enumerate(self.shape_btns):
            on = self.st.ed.armed == i
            b.configure(bg=LIME if on else PANEL2, fg="#000" if on else FG)
        if z is None:
            self.jig_box.pack_forget()
            self.jig_empty.configure(text="No zone selected — pick a shape above and click the photo, or pick a "
                                          "jiggle layer on the left.")
            self.jig_empty.pack(fill="x", padx=6)
            return
        self.jig_empty.pack_forget()
        self.jig_box.pack(fill="x")
        self.v_mat.set(z.material)
        self.v_size.set(round(z.r)); self.v_rot.set(((z.rot + 180) % 360) - 180)
        self.v_tip.set(z.tip)
        sim = J.ZoneSim(J.replace(z, tune=dict(z.tune)))
        mi = J.MATERIALS[z.material]
        defaults = {"f1": mi[2], "z1": mi[3], "f2": mi[4], "z2": mi[5], "stiff": mi[6], "sens": mi[7], "couple": mi[8],
                    "travel": 1.0, "bulge": 1.0, "stretch": 1.0, "sag": 1.0, "pend": J.PENDULUM, "proj": 0.6, "tail": 1.0,
                    "psize": 1.0, "dance": 1.0, "side": 0.0, "mass": 1.0, "anchor": 0.0, "slack": 0.2, "down": 0.0, "contact": 1.0}
        if z.tip:
            d = sim.droop
            defaults.update(z1=sim.z1, stiff=sim.stiffen, couple=sim.couple, sag=sim.sag, pend=sim.pend, proj=0.6 - 0.25 * d)
            kind = "hanging / droopy" if d > 0.35 else ("firm / perky" if d < -0.35 else "typical")
            self.form_lbl.configure(text=f"Point reads as: {kind}  (droop {d:+.2f})")
        else:
            self.form_lbl.configure(text="No control point: motion follows the outline only.")
        defaults.update(travel=sim.form_travel, slack=sim.slack, anchor=z.tuned("anchor", J.FORM_MIX if z.tip else 0.0))
        self._tune_shown = {}
        for key, row in self.tune_rows.items():
            set_ = key in z.tune
            v = z.tuned(key, 0.0) if set_ else defaults.get(key, 0.0)
            row["var"].set(v)
            self._tune_shown[key] = float(row["var"].get())
            row["label"].configure(fg=LIME if set_ else MUTED)
        self._draw_response(z)

    def _draw_response(self, z):
        c = self.resp
        c.delete("all")
        t, v, st = J.step_response(z)
        Wd, Hd = 340, 100
        peak = max(1.0, st["peak_px"])
        c.create_line(0, Hd / 2, Wd, Hd / 2, fill="#333")
        pts = []
        for i in range(0, len(t), 3):
            pts += [t[i] / t[-1] * Wd, Hd / 2 - v[i] / peak * (Hd / 2 - 6)]
        if len(pts) >= 4:
            c.create_line(*pts, fill=LIME, width=2, smooth=True)
        self.resp_lbl.configure(text=f"{st['hz']:.1f} Hz · peak {st['peak_px']:.1f} px · {st['swings']} visible swings · "
                                     f"settles in {st['settle_ms']:.0f} ms")

    # ================================================================================== actions
    def _changed(self, soft=False):
        """After any edit: the preview restarts with it, the panels follow."""
        self.st.dirty = True
        if self.preview.get():
            self._restart_sim(keep_motion=True)
        if soft:                                         # (a slider being dragged: don't rebuild the panels)
            self._refresh_status()
            self._render()
            return
        self._refresh_all()

    def _do(self, fn):
        r = fn()
        self._changed()
        return r

    def open_dialog(self):
        p = filedialog.askopenfilename(parent=self, title="Open an .mg photo", filetypes=[("Mantis Gallery image", "*.mg"), ("All", "*.*")])
        if p:
            self.open_path(p)

    def open_path(self, path):
        if self.st.dirty and not messagebox.askyesno("Unsaved changes", "Discard the changes to the open photo?", parent=self):
            return False
        try:
            had = self.st.open(path)
        except (OSError, ValueError) as e:
            messagebox.showerror("Can't open", str(e), parent=self)
            return False
        self.preview.set(False)
        self.sim = None
        self.tool = "grid" if not self.st.par.grid else "smart"
        self.nb.select(0)
        self._hint(("Loaded its scene." if had else "No scene for this photo yet.") + "  Start with the Ground grid, then add Figures.")
        self._refresh_all()
        return True

    def save(self):
        try:
            jp = self.st.save()
        except (OSError, ValueError) as e:
            messagebox.showerror("Can't save", str(e), parent=self)
            return None
        self._hint(f"Saved {jp.name} next to the photo — the Core2 plays it and edits the same layers.")
        self._refresh_all()
        return jp

    def revert(self):
        if self.st.ed.mg_path is not None:
            self.st.dirty = False
            self.open_path(self.st.ed.mg_path)

    def undo(self):
        if self.st.undo():
            self._changed()

    def redo(self):
        if self.st.redo():
            self._changed()

    def add(self, what):
        if self.st.image is None:
            return
        names = {"figure": "Figure", "water": "Water", "jiggle": "Zone", "hair": "Hair lock", "cloth": "Cloth", "pendant": "Pendant"}
        if what == "jiggle":
            self.nb.select(SUBTABS.index("Jiggle"))
            self.arm(0)
            return
        name = simpledialog.askstring("Name it", f"Name for the new {names[what].lower()}:", parent=self,
                                      initialvalue=self.st.unique_name(names[what])) or names[what]
        if what == "figure":
            key = self.st.add_figure(name)
            self.tool = "cut"
        elif what == "water":
            key = self.st.add_water(name, J.W_STILL)
            self.tool = "cut"
        else:
            key = self.st.add_sway(what, name)
            self.tool = "cut"
        if key is None:
            messagebox.showinfo("Full", "That's the most of that kind the Core2 can play.", parent=self)
            return
        self.sub.set(False)
        self.nb.select(SUBTABS.index(TAB_OF[key[0]]))
        self._changed()

    def select_layer(self, key):
        if self.st.set_active(key):
            L = self.st.layer()
            hs = [h[0] for h in self._handles(L)]
            if L.has_mask:
                if self.tool not in MASK_TOOLS + tuple(hs):
                    self.tool = "cut"
            else:
                self.tool = hs[0] if hs else "smart"
            self.st.ed.armed = -1
            self._pts = []
            want = TAB_OF[L.kind]
            if self.nb.index(self.nb.select()) != SUBTABS.index(want) and not (want == "Scene" and self._tab() == "Depth"):
                self._syncing_tab = True
                self.nb.select(SUBTABS.index(want))
            self._refresh_all()

    def _tab(self):
        try:
            return SUBTABS[self.nb.index(self.nb.select())]
        except (tk.TclError, ValueError, TypeError):
            return "Scene"

    def _tab_changed(self, e=None):
        if getattr(self, "_syncing_tab", False):
            self._syncing_tab = False
            return
        tab = self._tab()
        L = self.st.layer()
        if tab in ("Depth",) or TAB_OF[L.kind] == tab:
            self._refresh_all()
            return
        kinds = {"Scene": ("figure", "ground"), "Jiggle": ("jiggle",), "Water": ("water",), "Sway": ("sway",)}[tab]
        for k in kinds:
            M = next((M for M in self.st.layers() if M.kind == k), None)
            if M is not None:
                self.st.set_active(M.key)
                hs = [h[0] for h in self._handles(M)]
                self.tool = "cut" if M.has_mask else (hs[0] if hs else "smart")
                break
        self._refresh_all()

    def rename(self):
        L = self.st.layer()
        if L.kind in ("sky", "ground"):
            return
        n = simpledialog.askstring("Rename", "New name:", parent=self, initialvalue=L.name)
        if n and not self.st.rename_active(n):
            messagebox.showinfo("Rename", "That name is taken (or empty).", parent=self)
        self._changed()

    def delete_layer(self):
        L = self.st.layer()
        if L.kind == "sky" or (L.kind == "ground" and not self.st.par.grid):
            return
        what = "the ground grid" if L.kind == "ground" else f"“{L.name}”"
        if messagebox.askyesno("Delete", f"Delete {what}? (Undo brings it back.)", parent=self):
            self._do(self.st.delete_active)

    def _toggle_eye(self, key):
        self.st.hidden ^= {tuple(key)}
        self._refresh_all()

    def set_tool(self, t):
        L = self.st.layer()
        if t in MASK_TOOLS and not L.has_mask:
            return
        if t == "grid" and not self.st.par.grid:
            self.st.grid_default()
        self.tool = t
        self._pts = []
        self.st.ed.armed = -1
        self._refresh_all()

    def set_sub(self, on):
        self.sub.set(on)
        self._refresh_strip()
        self._refresh_status()

    def smart_step(self, d):
        r = self.st.smart_step(d)
        if r is None:
            self._hint("Tighter / Looser change the last Smart pick — click with Smart first.")
        elif r == 0 and self.st._smart:
            self._hint("That's as " + ("tight" if d < 0 else "loose") + " as it goes.")
        self._changed()

    def refine(self):
        if self.st.mask() is None or not self.st.mask().any():
            self._hint("Refine edges snaps the layer's mask to the real edges — select something first.")
            return
        n = self.st.refine_edges()
        self._hint(f"Refined: {n} px changed to follow the real edges." if n else "Already on the edges.")
        self._changed()

    def clear_mask(self):
        if self.st.mask() is not None and self.st.mask().any() and messagebox.askyesno(
                "Clear", f"Clear everything selected in “{self.st.layer().name}”?", parent=self):
            self._do(self.st.clear_mask)

    def arm(self, shape):
        if self.st.image is None:
            return
        ed = self.st.ed
        ed.armed = -1 if ed.armed == shape else shape
        if ed.armed >= 0 and len(ed.zones) >= J.MAX_ZONES:
            ed.armed = -1
            messagebox.showinfo("Full", f"{J.MAX_ZONES} zones is the most the Core2 plays.", parent=self)
        self.tool = "zone"
        self._refresh_all()

    def set_zoom(self, z):
        self.zoom = z
        self.canvas.configure(width=J.IMG_W * z, height=J.IMG_H * z)
        self.status_lbl.configure(wraplength=max(560, J.IMG_W * z))
        self.hint_lbl.configure(wraplength=max(560, J.IMG_W * z))
        self._img_item = None
        self.canvas.delete("all")
        self._refresh_all()

    # ================================================================================== canvas input
    def _xy(self, e):
        return e.x / self.zoom, e.y / self.zoom

    def _subtracting(self, e=None):
        alt = bool(e is not None and (getattr(e, "state", 0) & 0x0008 or getattr(e, "state", 0) & 0x20000))
        self._alt = alt
        return self.sub.get() or alt

    def _press(self, e):
        if self.st.image is None:
            return
        x, y = self._xy(e)
        if self.preview.get():
            if self.sim:
                self.sim.poke(x, y)
            return
        L, t = self.st.layer(), self.tool
        self._drag = None
        if t == "zone" or self.st.ed.armed >= 0:
            if self.st.ed.armed >= 0:
                key = self.st.add_zone(self.st.ed.armed, x, y)
                self.st.ed.armed = -1
                self._drag = ("zone", x, y, True)
                self.st.ed._drag = "zone"; self.st.ed._last = (x, y)
                self._changed()
                return
            ed = self.st.ed
            if ed.zone is not None and (ed.near_tip(x, y) or ed.near_rot(x, y)):
                self.st.push_undo()
                ed.press(x, y)
            else:
                i = ed.hit(x, y)
                if i >= 0:
                    self.st.set_active(("jiggle", i))
                    self.st.push_undo()
                    ed.press(x, y)
                    self._changed()
            self._drag = ("zone", x, y, False)
            return
        if t == "grid":
            if not self.st.par.grid:                    # the first click puts a grid down to fit
                self.st.grid_default()
                self._hint("A ground grid is down: drag its 4 corners until the squares lie flat on the ground.")
                self._changed()
                return
            g = self.st.par.grid
            k = min(range(4), key=lambda k: (g[k][0] - x) ** 2 + (g[k][1] - y) ** 2)
            near = (g[k][0] - x) ** 2 + (g[k][1] - y) ** 2 <= (12 / self.zoom * 2) ** 2
            self.st.push_undo()
            self._drag = ("corner", k) if near else ("gridmove", x, y)
            return
        if t == "feet" and L.kind == "figure":
            self.st.set_feet(x, y, True)
            self._drag = ("feet",)
            self._changed(soft=True)
            return
        if t in ("line", "lasso"):
            self._pts.append((x, y))
            self._render()
            return
        if t == "hips" and L.kind == "sway" and L.obj.kind == "cloth":
            pts = self.st.hips()
            if pts:
                i = min(range(len(pts)), key=lambda i: (pts[i][0] - x) ** 2 + (pts[i][1] - y) ** 2)
                if (pts[i][0] - x) ** 2 + (pts[i][1] - y) ** 2 <= (14 / self.zoom * 2) ** 2:
                    self.st.set_hip(i, x, y, True)
                    self._drag = ("hip", i)
                    self._changed(soft=True)
            return
        if t == "guide" and L.kind == "sway":
            if L.obj.kind == "hair":
                self._pts = [(x, y)]
                self._drag = ("stroke",)
            elif L.obj.kind == "cloth":
                if L.obj.guides and len(L.obj.guides[0]) >= 2 and not self._pts:   # waistband set: a click marks a fold
                    self.st.sway_guide([(x, y)])
                    self._hint("Fold marked. Mark each fold you want the pleats split at (or none: it finds them).")
                    self._changed()
                    return
                self._pts.append((x, y))
                if len(self._pts) >= 2:
                    self.st.sway_guide(self._pts); self._pts = []
                    self._changed()
                else:
                    self._render()
            else:
                self.st.sway_guide([(x, y)])
                self._changed()
            return
        if not L.has_mask:
            return
        sub = self._subtracting(e)
        if t == "cut":                                  # a stroke on it (keep) or on what it isn't (not)
            c = self._cut
            if c is None or c["key"] != L.key:
                c = self._cut = {"key": L.key, "keep": np.zeros((J.IMG_H, J.IMG_W), bool), "not": np.zeros((J.IMG_H, J.IMG_W), bool)}
            self._marks = np.zeros((J.IMG_H, J.IMG_W), bool)
            J.mask_brush(self._marks, int(x), int(y), 1, True)
            self._last_xy = (x, y)
            self._drag = ("cut", sub)
            self._render()
            return
        if t == "smart":
            self._marks = np.zeros((J.IMG_H, J.IMG_W), bool)
            J.mask_brush(self._marks, int(x), int(y), 2, True)
            self._last_xy = (x, y)
            self._drag = ("smart", sub)
            self._render()
        elif t in ("brush", "erase"):
            self.st.brush(x, y, self.brush_r, erase=(t == "erase") or sub, start=True)
            self._drag = ("paint", (t == "erase") or sub)
            self._changed(soft=True)

    def _drag_ev(self, e):
        if self.st.image is None or self.preview.get() or not self._drag:
            return
        x, y = self._xy(e)
        k = self._drag[0]
        if k == "zone":
            self.st.ed.drag(x, y)
            self._changed(soft=True)
        elif k == "corner":
            self.st.grid_corner(self._drag[1], x, y, start=False)
            self._changed(soft=True)
        elif k == "gridmove":
            self.st.grid_move(x - self._drag[1], y - self._drag[2], start=False)
            self._drag = ("gridmove", x, y)
            self._changed(soft=True)
        elif k == "feet":
            self.st.set_feet(x, y, False)
            self._changed(soft=True)
        elif k == "hip":
            self.st.set_hip(self._drag[1], x, y, False)
            self._changed(soft=True)
        elif k == "stroke":
            self._pts.append((x, y))
            self._render()
        elif k in ("smart", "cut"):                    # (a stroke, not dots: fast drags skip pixels)
            r = 2 if k == "smart" else (2 if self._drag[1] else 1)
            lx, ly = self._last_xy or (x, y)
            n = int(max(abs(x - lx), abs(y - ly))) + 1
            for i in range(n + 1):
                J.mask_brush(self._marks, int(lx + (x - lx) * i / n), int(ly + (y - ly) * i / n), r, True)
            self._last_xy = (x, y)
            self._render()
        elif k == "paint":
            self.st.brush(x, y, self.brush_r, erase=self._drag[1], start=False)
            self._changed(soft=True)

    def _release(self, e):
        if not self._drag:
            return
        k = self._drag[0]
        if k == "cut":
            c = self._cut
            c["not" if self._drag[1] else "keep"] |= self._marks
            c["keep"] &= ~c["not"]
            self._marks = None
            if c["keep"].any():
                n = self.st.cut(c["keep"], c["not"] if c["not"].any() else None)
                self._hint(f"Cut out: {n} px changed.  Paint more on it, or Alt / Subtract-paint on what it isn't "
                           "(the face next to hair, the ground between feet) — every stroke refines the same cut.")
            else:
                self._hint("Now paint a stroke ON it (the not-strokes only say what it isn't).")
        elif k == "smart":
            n = self.st.smart(self._marks, add=not self._drag[1])
            self._marks = None
            self._hint(f"{'Removed' if self._drag[1] else 'Added'} {n} px.  Tighter / Looser step it; Refine edges snaps it."
                       if n else "Nothing changed there — try dragging over a bigger sample, or Looser.")
        elif k == "stroke":
            if len(self._pts) >= 2 and math.hypot(self._pts[-1][0] - self._pts[0][0], self._pts[-1][1] - self._pts[0][1]) >= 6:
                self.st.sway_guide(self._pts)
            else:
                self._hint("Drag from where the lock is tied to its tip (a little further).")
            self._pts = []
        elif k == "zone":
            self.st.ed.release(0, 0)
        self._drag = None
        self._changed()

    def finish_pts(self):
        L = self.st.layer()
        if self.tool == "lasso" and len(self._pts) >= 3:
            n = self.st.lasso(self._pts, add=not self.sub.get())
            self._hint(f"Lasso: {n} px.  Refine edges to snap it to the real outline.")
        elif self.tool == "line" and len(self._pts) >= 2 and L.kind == "water":
            self.st.set_water_line(self._pts)
        self._pts = []
        self._changed()

    def _hover(self, e):
        self._subtracting(e)
        if self.tool in ("brush", "erase") or self._pts:
            self._render()

    def _wheel(self, e, direction=None):
        d = direction if direction is not None else (1 if getattr(e, "delta", 0) > 0 else -1)
        L = self.st.layer()
        if self.tool == "grid" and self.st.par.grid:
            self._do(lambda: self.st.grid_turn(0.5 * d))
        elif L.kind == "jiggle" and self.st.ed.zone is not None:
            self._zone_do(lambda: self.st.ed.resize(1.06 if d > 0 else 1 / 1.06))
        elif self.tool in ("brush", "erase"):
            self.brush_r = int(min(40, max(1, self.brush_r + d)))
            self._refresh_status(); self._render()

    def _bind_keys(self):
        r = self.root

        def on(seq, fn):
            r.bind(seq, lambda e: None if not self._active() else fn(e), add="+")
        on("<Control-s>", lambda e: self.save())
        on("<Control-z>", lambda e: self.undo())
        on("<Control-y>", lambda e: self.redo())
        on("<Control-Z>", lambda e: self.redo())
        on("<space>", lambda e: (self.preview.set(not self.preview.get()), self._preview_toggled()))
        on("<Return>", lambda e: self.finish_pts())
        on("<Escape>", lambda e: (self._pts.clear(), setattr(self.st.ed, "armed", -1), self._refresh_all()))
        on("<Delete>", lambda e: self.delete_layer())
        on("<bracketleft>", lambda e: (setattr(self, "brush_r", max(1, self.brush_r - 1)), self._refresh_status(), self._render()))
        on("<bracketright>", lambda e: (setattr(self, "brush_r", min(40, self.brush_r + 1)), self._refresh_status(), self._render()))
        for key, t in (("w", "smart"), ("b", "brush"), ("e", "erase"), ("l", "lasso"), ("g", "grid"), ("f", "feet"),
                       ("s", "line"), ("k", "guide"), ("m", "zone")):
            on(f"<KeyPress-{key}>", lambda e, t=t: self._key_tool(t))
        for k in ("Alt_L", "Alt_R"):
            on(f"<KeyPress-{k}>", lambda e: (setattr(self, "_alt", True), self._refresh_status()))
            on(f"<KeyRelease-{k}>", lambda e: (setattr(self, "_alt", False), self._refresh_status()))

    def _key_tool(self, t):
        L = self.st.layer()
        if t in MASK_TOOLS:
            self.set_tool(t)
        elif t in [h[0] for h in self._handles(L)]:
            self.set_tool(t)

    def _active(self):
        """Keys only while this tab shows and focus isn't in a text field."""
        try:
            if not self.winfo_ismapped():
                return False
            f = self.focus_get()
        except (KeyError, tk.TclError):
            return True
        return not isinstance(f, (tk.Entry, tk.Spinbox, ttk.Entry, ttk.Combobox))

    # ================================================================================== preview
    def _restart_sim(self, keep_motion=False):
        """The preview starts at once; hair / cloth / pendants join it when their fill behind them is ready
        (that takes a few seconds the first time, in the background; then it's cached)."""
        if self.st.image is None:
            self.sim = None
            return
        self.sim = self.st.sim(with_sways=False)
        self._lean_prev = float(self.lean.get())
        if not keep_motion:
            self._motion = None
        sways = [s.copy() for s in self.st.sways if s.ok()]
        if not sways:
            return
        body = self.st.body_mask()
        body = body if body.any() else None
        sim, img = self.sim, self.st.image
        self._sway_wait = True

        def work():
            try:
                tmp = J.Sim([], img)
                tmp.add_sways(sways, body)              # fills the cache
            except Exception as ex:                     # (handed to the Tk thread: _tick picks it up)
                self._sway_result = ("error", str(ex))
                return
            self._sway_result = ("ok", (sim, sways, body))
        self._sway_result = None
        threading.Thread(target=work, daemon=True).start()
        self._hint("Preparing hair / cloth / pendants (the fill behind them)… the rest of the preview runs meanwhile.")

    def _sways_ready(self, sim, sways, body):
        if sim is not self.sim:                         # (the preview restarted meanwhile)
            return
        sim.add_sways(sways, body)                      # (from the cache: instant)
        self._sway_wait = False
        self._hint("Hair / cloth / pendants are in the preview.")

    def _preview_toggled(self):
        if hasattr(self, "prev_btn"):
            self._paint_preview_bar()
        if self.preview.get():
            self._restart_sim()
        else:
            self.sim = None
        self._refresh_all()

    def _dance_toggled(self):
        self._dance_amp = 0.0
        self._next_beat = self._sim_t
        if self.dance.get() and not self.preview.get():
            self.preview.set(True)
            self._preview_toggled()

    def start_motion(self, kind):
        if not self.preview.get():
            self.preview.set(True)
            self._preview_toggled()
        self._motion = (kind, self._sim_t)

    def step_preview(self, dt):
        """Advance the preview by dt seconds (the Core2's own maths: accelerometer, gyro, beats)."""
        if not self.sim:
            return
        t0 = self._sim_t
        self._sim_t += dt
        tilt, lean = float(self.tilt.get()), float(self.lean.get())
        acc = J.motion_accel("none", 0.0, tilt, lean)
        twist, gyro = (0.0, 0.0), [0.0, 0.0]
        if self._motion:
            kind, ts = self._motion
            if self._sim_t - ts > 6.5:
                self._motion = None
            else:
                acc = J.motion_accel(kind, self._sim_t - ts, tilt, lean)
                twist = J.motion_twist(kind, self._sim_t - ts)
                gx, gy = J.motion_gyro(kind, self._sim_t - ts)
                gyro = [gx, gy]
        if dt > 0:                                          # leaning the Core2 turns it about its horizontal axis
            gyro[0] += -(lean - self._lean_prev) / dt
        self._lean_prev = lean
        self.sim.gyro = tuple(gyro)
        period = 60.0 / max(40.0, float(self.bpm.get() or 120))
        self._dance_amp = min(1.0, self._dance_amp + dt / (1.5 * period)) if self.dance.get() else max(0.0, self._dance_amp - dt)
        if self._dance_amp > 0 and self._sim_t >= self._next_beat:
            self._dance_sign = -self._dance_sign
            self.sim.beat(0.35 * self._dance_amp, self._dance_sign)     # (a nudge, as on the Core2)
            self._next_beat = max(self._next_beat + period, t0)
        self.sim.step(dt, acc, self._dance_amp, twist)

    def _tick(self):
        now = time.perf_counter()
        dt = min(0.05, now - self._last_tick)
        self._last_tick = now
        res = getattr(self, "_sway_result", None)
        if res is not None:                             # (work finished on a background thread)
            self._sway_result = None
            if res[0] == "ok":
                self._sways_ready(*res[1])
            else:
                self._hint(f"Couldn't prepare the swaying things: {res[1]}")
        res = getattr(self, "_job_done", None)
        if res is not None:
            self._job_done = None
            res()
        try:
            if self.preview.get() and self.sim:
                self.step_preview(dt * float(self.slowmo.get()))
                self._render()
        finally:
            self.after(33, self._tick)

    # ================================================================================== drawing
    def frame_rgb(self):
        img = self.st.image
        if img is None:
            return None
        if self.preview.get() and self.sim:
            img = self.sim.render()
        else:
            img = img.copy()
        if self.show_depth.get():
            d = self.st.depth_in if self.st.depth_in is not None else self.st.scene_depth()
            v = (np.clip(d, 0, 1) * 255).astype(np.uint8)
            img = (img * 0.35 + np.stack([v, v, v], -1) * 0.65).astype(np.uint8)
        if not self.preview.get() and self.show_layers.get():
            for L in self.st.layers():
                m = L.mask
                if m is None or not m.any() or L.key in self.st.hidden:
                    continue
                a = 0.45 if L.key == self.st.active else 0.18
                img[m] = (img[m] * (1 - a) + np.array(_hex(self.st.color(L))) * a).astype(np.uint8)
        c = self._cut
        if c is not None and c["key"] == self.st.active and not self.preview.get() and self.tool == "cut":
            img[c["keep"]] = _hex(LIME)
            img[c["not"]] = (255, 70, 70)
        if self._marks is not None:
            red = self._drag is not None and self._drag[0] == "cut" and self._drag[1]
            img[self._marks] = (255, 70, 70) if red else ((180, 255, 0) if self.tool == "cut" else (255, 230, 0))
        return img

    def _render(self):
        c = self.canvas
        c.delete("ov")
        img = self.frame_rgb()
        k = self.zoom
        if img is None:
            c.delete("all")
            self._img_item = None
            c.create_text(J.IMG_W * k / 2, J.IMG_H * k / 2, fill=MUTED, font=FONT,
                          text="Open an .mg photo from the SD card\n(the /images folder)", justify="center")
            return
        if Image is not None:
            pil = Image.fromarray(img).resize((J.IMG_W * k, J.IMG_H * k), Image.NEAREST)
            self._photo = ImageTk.PhotoImage(pil)
            if self._img_item is None:
                c.delete("all")
                self._img_item = c.create_image(0, 0, anchor="nw", image=self._photo)
            else:
                c.itemconfigure(self._img_item, image=self._photo)
        if self.show_handles.get() and not self.preview.get():
            self._draw_handles()
        elif self.preview.get():
            self._draw_horizon(dim=True)

    def _tag(self, x, y, text, col, anchor="w"):
        c = self.canvas
        t = c.create_text(x, y, text=text, fill=col, anchor=anchor, font=("Segoe UI", 9, "bold"), tags="ov")
        b = c.bbox(t)
        if b:                                           # keep it inside the picture
            cw, ch = J.IMG_W * self.zoom, J.IMG_H * self.zoom
            dx = (cw - 4 - b[2]) if b[2] > cw - 4 else (4 - b[0] if b[0] < 4 else 0)
            dy = (ch - 4 - b[3]) if b[3] > ch - 4 else (4 - b[1] if b[1] < 4 else 0)
            if dx or dy:
                c.coords(t, x + dx, y + dy)
                b = c.bbox(t)
            r = c.create_rectangle(b[0] - 3, b[1] - 1, b[2] + 3, b[3] + 1, fill="#000", outline=col, tags="ov")
            c.tag_lower(r, t)

    def _draw_horizon(self, dim=False):
        hz = self.st.horizon()
        if hz is None:
            return
        k, c = self.zoom, self.canvas
        y0, y1 = hz
        if -20 < (y0 + y1) / 2 < J.IMG_H + 20:
            c.create_line(0, y0 * k, J.IMG_W * k, y1 * k, fill=YEL, dash=(6, 4), width=1, tags="ov")
            L = self.st.layer()
            if not dim and (L.kind in ("ground", "sky") or self.tool == "grid"):
                self._tag(J.IMG_W * k - 6, y1 * k - 10, "horizon", YEL, anchor="e")

    def _draw_handles(self):
        c, k, st = self.canvas, self.zoom, self.st
        L = st.layer()
        self._draw_horizon()
        g = st.par.grid
        if g:
            act = self.tool == "grid" or L.kind == "ground"
            col = LIME if act else "#7a7a7a"
            H_ = J.water_homog(g)
            for i in range(9):
                for a, b in (((i / 8, 0), (i / 8, 1)), ((0, i / 8), (1, i / 8))):
                    pa, pb = J.water_hmap(H_, *a), J.water_hmap(H_, *b)
                    c.create_line(pa[0] * k, pa[1] * k, pb[0] * k, pb[1] * k, fill=col, width=1, tags="ov")
            if act:
                for j, (x, y) in enumerate(g):
                    cc = LIME if j < 2 else YEL
                    c.create_oval(x * k - 7, y * k - 7, x * k + 7, y * k + 7, fill="#000", outline=cc, width=2, tags="ov")
                    self._tag(x * k + 10, y * k, "near corner" if j < 2 else "far corner", cc)
        for M in st.layers():                               # figures: name + feet line
            if M.kind != "figure" or M.mask is None or not M.mask.any():
                continue
            ys, xs = np.nonzero(M.mask)
            col = st.color(M)
            fx, fy = st.feet(M.index)
            on = M.key == st.active
            if on or self.tool == "feet":
                c.create_line((fx - 18) * k, fy * k, (fx + 18) * k, fy * k, fill=YEL, width=3, tags="ov")
                if on:
                    self._tag((fx + 20) * k, fy * k, "feet line ◆", YEL)
            if on and M.key not in st.hidden:
                self._tag(xs.max() * k + 6, ys.min() * k + 8, f"{M.name} · Figure", col)
        A = st.layer()                                      # the active water / sway: its name
        if A.kind in ("water", "sway") and A.mask is not None and A.mask.any() and A.key not in st.hidden:
            ys, xs = np.nonzero(A.mask)
            self._tag(xs.max() * k + 6, ys.min() * k + 8, f"{A.name} · {SM.KIND_LABEL[A.kind]}", st.color(A))
        for M in st.layers():                               # water lines
            if M.kind == "water" and M.obj.lines and M.obj.mode != J.W_STILL:
                for ln in M.obj.lines:
                    if len(ln) >= 2:
                        pts = [v * k for p in ln for v in p]
                        c.create_line(*pts, fill="#ffffff", width=3, tags="ov")
                        if M.key == st.active:
                            self._tag(ln[-1][0] * k + 6, ln[-1][1] * k, "shore" if M.obj.mode == J.W_OCEAN else "flow", "#ffffff")
                        if M.obj.mode == J.W_RIVER:
                            (ax, ay), (bx, by) = ln[-2], ln[-1]
                            c.create_line(ax * k, ay * k, bx * k, by * k, fill="#ffffff", width=3, arrow="last", tags="ov")
        if L.kind == "sway" and L.obj.kind == "cloth" and (self.tool == "hips" or L.key == st.active):
            pts = st.hips()                                 # her hips under the skirt
            if pts:
                on = self.tool == "hips"
                flat = [v * k for p in pts for v in p]
                c.create_line(*flat, *flat[:2], fill="#ff6fae" if on else "#a0587a", width=2 if on else 1, dash=() if on else (3, 3), tags="ov")
                if on:
                    q = SW.bikini_poly(pts)                 # (safety: never skin here - a white bikini bottom)
                    fq = [v * k for p in q for v in p]
                    c.create_line(*fq, *fq[:2], fill="#ffffff", dash=(2, 2), width=1, tags="ov")
                    for i, (x, y) in enumerate(pts):
                        c.create_oval(x * k - 5, y * k - 5, x * k + 5, y * k + 5, fill="#000", outline="#ff6fae", width=2, tags="ov")
        for M in st.layers():                               # sway guides (the active one's)
            if M.kind != "sway" or M.key != st.active:
                continue
            for gl in M.obj.guides:
                if M.obj.kind == "pendant":
                    x, y = gl[0]
                    c.create_oval(x * k - 6, y * k - 6, x * k + 6, y * k + 6, outline=YEL, width=2, tags="ov")
                    pass
                elif len(gl) >= 2:
                    pts = [v * k for p in gl for v in p]
                    c.create_line(*pts, fill=YEL, width=2, arrow="last" if M.obj.kind == "hair" else "none", tags="ov")
                    c.create_oval(gl[0][0] * k - 4, gl[0][1] * k - 4, gl[0][0] * k + 4, gl[0][1] * k + 4, fill=YEL, outline="#000", tags="ov")

        for i, z in enumerate(st.ed.zones):                 # jiggle zones
            col = LIME
            sel = L.kind == "jiggle" and L.index == i
            pts = [v * k for p in z.outline(1.0, 64) for v in p]
            c.create_line(*pts, *pts[:2], fill="#000", width=4 if sel else 3, tags="ov")
            c.create_line(*pts, *pts[:2], fill=col if sel else "#88aa44", width=2 if sel else 1, tags="ov")
            if sel:
                hx, hy = st.ed.rot_handle_xy(z)
                ex, ey = z.from_local(0.0, 1.0)
                c.create_line(ex * k, ey * k, hx * k, hy * k, fill=col, dash=(2, 2), tags="ov")
                c.create_oval(hx * k - 7, hy * k - 7, hx * k + 7, hy * k + 7, fill="#000", outline=col, width=2, tags="ov")
                c.create_text(hx * k, hy * k, text="⟳", fill=col, font=("Segoe UI", 8, "bold"), tags="ov")
            if z.tip:
                tx, ty = z.tip_xy()
                rr = max(3.0, z.tip_radius() * z.r) * k
                c.create_oval(tx * k - rr, ty * k - rr, tx * k + rr, ty * k + rr, outline=YEL if sel else col, width=1, tags="ov")
            if sel:
                self._tag(z.cx * k, (z.cy - z.r) * k - 10, SM.Layer("jiggle", i, z).name, col, anchor="center")
        if self._pts:                                       # lasso / line / stroke in progress
            pts = [v * k for p in self._pts for v in p]
            if len(pts) >= 4:
                c.create_line(*pts, fill=YEL, width=2, dash=(4, 3) if self.tool == "lasso" else (), tags="ov")
            for x, y in self._pts:
                c.create_oval(x * k - 3, y * k - 3, x * k + 3, y * k + 3, fill=YEL, outline="#000", tags="ov")

    # ================================================================================== masks & depth files
    def export_masks(self):
        if self.st.image is None:
            return
        d = filedialog.askdirectory(parent=self, title="Export masks to a folder (a new one is best)")
        if not d:
            return
        stem = Path(self.st.ed.mg_path).stem if self.st.ed.mg_path else "scene"
        folder = Path(d) / f"{stem}_masks"
        files = self.st.export_masks(folder)
        self._hint(f"Exported {len(files)} files to {folder}. Clean them up anywhere, then Masks ▸ Import masks…")
        messagebox.showinfo("Masks exported", f"Saved to\n{folder}\n\n" + "\n".join(files) +
                            "\n\nWhite = in the layer. Keep the names to bring them back to the same layers.", parent=self)

    def import_masks(self):
        if self.st.image is None:
            return
        paths = filedialog.askopenfilenames(parent=self, title="Import masks (PNG etc.; Layer.type.png matches a layer)",
                                            filetypes=[("Images", "*.png *.tif *.tiff *.bmp *.jpg *.jpeg *.webp"), ("All", "*.*")])
        if not paths:
            return
        rows = self.st.import_plan(paths)
        if not rows:
            messagebox.showinfo("Import masks", "No mask images in that selection.", parent=self)
            return
        ImportDialog(self, rows)

    def _apply_import(self, rows, protect):
        n = self.st.import_apply(rows, protect)
        self._hint(f"Imported {n} layer(s). One Undo takes the whole import back.")
        self._changed()

    def import_depth(self):
        if self.st.image is None:
            return
        p = filedialog.askopenfilename(parent=self, title="Import a depth map", filetypes=[("Images", "*.png *.tif *.tiff *.jpg *.jpeg *.webp *.bmp"), ("All", "*.*")])
        if not p:
            return
        near_white = messagebox.askyesno("Which end is near?", "Is NEAR shown as WHITE (bright) in this map?\n\n"
                                         "(Depth Anything / MiDaS / ControlNet depth: yes.)", parent=self)
        self.st.import_depth(p, near_white)
        self.show_depth.set(True)
        self.nb.select(SUBTABS.index("Depth"))
        self._hint("Depth map imported and lined up with the ground grid. Show ▸ Depth map toggles it.")
        self._changed()

    def export_depth(self):
        if self.st.image is None:
            return
        p = filedialog.asksaveasfilename(parent=self, title="Export the depth map", defaultextension=".png",
                                         filetypes=[("16-bit PNG", "*.png")], initialfile="depth.png")
        if p:
            self.st.export_depth(p)
            self._hint(f"Saved {p} (16-bit, near = white).")

    def fit_grid(self):
        if self.st.depth_raw is None:
            messagebox.showinfo("Fit grid", "Import or generate a depth map first (Depth ▾).", parent=self)
            return
        y = self.st.fit_grid_from_depth()
        if y is None:
            messagebox.showinfo("Fit grid", "Couldn't find a flat ground in that map — fit the grid by hand.", parent=self)
            return
        self.tool = "grid"
        self.st.set_active(("ground", 0))
        self.nb.select(0)
        self._hint(f"Grid fitted from the map (horizon at y≈{y:.0f}). Check the squares lie flat; drag corners to correct.")
        self._changed()

    def use_map_depths(self):
        if self.st.depth_in is None:
            messagebox.showinfo("Map depths", "Import or generate a depth map first (Depth ▾).", parent=self)
            return
        n = self.st.use_map_depths()
        self._hint(f"{n} figure(s) now take their depth from the map.")
        self._changed()

    def generate_depth(self):
        if self.st.image is None:
            return
        GenerateDepthDialog(self)




class ImportDialog(tk.Toplevel):
    """What an import will do, file by file, before anything changes."""

    def __init__(self, tab, rows):
        super().__init__(tab, bg=PANEL)
        self.tab, self.rows = tab, rows
        self.title("Import masks")
        self.transient(tab.winfo_toplevel())
        tk.Label(self, text=f"Import masks — {len(rows)} file(s)", fg=FG, bg=PANEL, font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=12, pady=(10, 6))
        for c, h in enumerate(("FILE", "GOES INTO", "TYPE", "NOTES")):
            tk.Label(self, text=h, fg=YEL, bg=PANEL, font=("Segoe UI", 8, "bold")).grid(row=1, column=c, sticky="w", padx=12)
        self.types = []
        for i, r in enumerate(rows):
            tk.Label(self, text=Path(r["file"]).name, fg=FG, bg=PANEL, font=FONT_S).grid(row=2 + i, column=0, sticky="w", padx=12, pady=2)
            tk.Label(self, text=(f"{r['name']} (replace)" if r["action"] == "replace" else f"New layer: {r['name']}"),
                     fg=LIME if r["type"] else YEL, bg=PANEL, font=FONT_S).grid(row=2 + i, column=1, sticky="w", padx=12)
            v = tk.StringVar(value=r["type"] or "figure")
            cb = ttk.Combobox(self, textvariable=v, values=["figure", "water", "sway", "skip"], width=8, state="readonly")
            cb.grid(row=2 + i, column=2, sticky="w", padx=12)
            if r["action"] == "replace":
                cb.configure(state="disabled")
            self.types.append(v)
            tk.Label(self, text=r["note"] or ("white = in, black = out; scaled to fit" if True else ""), fg=MUTED, bg=PANEL,
                     font=FONT_S).grid(row=2 + i, column=3, sticky="w", padx=12)
        n = len(rows) + 2
        self.protect = tk.BooleanVar(value=True)
        tk.Checkbutton(self, text="Keep figures protected (an imported water mask can't cover a figure)", variable=self.protect,
                       bg=PANEL, fg=FG, selectcolor="#333", activebackground=PANEL, font=FONT_S).grid(
            row=n, column=0, columnspan=4, sticky="w", padx=12, pady=(8, 0))
        tk.Label(self, text="Reads: white = in the layer, black = out. A transparent PNG uses its alpha. One Undo undoes the whole import.",
                 fg=MUTED, bg=PANEL, font=FONT_S).grid(row=n + 1, column=0, columnspan=4, sticky="w", padx=12)
        bf = tk.Frame(self, bg=PANEL)
        bf.grid(row=n + 2, column=0, columnspan=4, sticky="e", padx=12, pady=10)
        tk.Button(bf, text="Cancel", command=self.destroy, bg=PANEL2, fg=FG, relief="flat", padx=10).pack(side="left", padx=4)
        tk.Button(bf, text=f"Import {len(rows)} layer(s)", command=self.ok, bg=LIME, fg="#000", relief="flat", padx=10).pack(side="left")

    def ok(self):
        rows = []
        for r, v in zip(self.rows, self.types):
            r = dict(r)
            if r["action"] != "replace":
                r["type"] = None if v.get() == "skip" else v.get()
            rows.append(r)
        self.destroy()
        self.tab._apply_import(rows, self.protect.get())


class GenerateDepthDialog(tk.Toplevel):
    """Depth Anything V2, run by this Python in its own process. From the ComfyUI folder it only reads the
    checkpoint and the model code as files — ComfyUI and its venv are never run or touched."""

    def __init__(self, tab):
        super().__init__(tab, bg=PANEL)
        self.tab = tab
        self.title("Generate depth")
        self.transient(tab.winfo_toplevel())
        s = tab._settings
        self.comfy = tk.StringVar(value=s.get("comfy_dir", ""))
        self.device = tk.StringVar(value=s.get("depth_device", "cuda"))
        tk.Label(self, text="Generate a depth map (Depth Anything V2)", fg=FG, bg=PANEL, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        r = tk.Frame(self, bg=PANEL); r.pack(fill="x", padx=12)
        tk.Label(r, text="ComfyUI folder", fg=FG, bg=PANEL, font=FONT_S, width=14, anchor="w").pack(side="left")
        tk.Entry(r, textvariable=self.comfy, width=44, bg=PANEL2, fg=FG, insertbackground=FG).pack(side="left")
        tk.Button(r, text="Browse…", command=self.browse, bg=PANEL2, fg=FG, relief="flat").pack(side="left", padx=4)
        r = tk.Frame(self, bg=PANEL); r.pack(fill="x", padx=12, pady=4)
        tk.Label(r, text="Run on", fg=FG, bg=PANEL, font=FONT_S, width=14, anchor="w").pack(side="left")
        tk.Radiobutton(r, text="GPU (half precision, ~0.7 GB for a second or two; CPU if it's full)", variable=self.device,
                       value="cuda", bg=PANEL, fg=FG, selectcolor="#333", font=FONT_S, activebackground=PANEL).pack(anchor="w")
        tk.Radiobutton(r, text="CPU only (~20 s)", variable=self.device, value="cpu", bg=PANEL, fg=FG,
                       selectcolor="#333", font=FONT_S, activebackground=PANEL).pack(anchor="w")
        tk.Label(self, text="Runs in this Python (needs torch here). From the ComfyUI folder it only reads the Depth Anything V2\n"
                            "checkpoint and its model code, as files. ComfyUI and its venv are never run or changed.",
                 fg=MUTED, bg=PANEL, font=FONT_S, justify="left").pack(anchor="w", padx=12)
        self.msg = tk.Label(self, text="", fg=YEL, bg=PANEL, font=FONT_S, justify="left", wraplength=520)
        self.msg.pack(anchor="w", padx=12, pady=4)
        bf = tk.Frame(self, bg=PANEL); bf.pack(anchor="e", padx=12, pady=10)
        tk.Button(bf, text="Close", command=self.destroy, bg=PANEL2, fg=FG, relief="flat", padx=10).pack(side="left", padx=4)
        self.run_btn = tk.Button(bf, text="Generate", command=self.run, bg=LIME, fg="#000", relief="flat", padx=10)
        self.run_btn.pack(side="left")

    def browse(self):
        d = filedialog.askdirectory(parent=self, title="Your ComfyUI folder (the one with custom_nodes)")
        if d:
            self.comfy.set(d)

    def run(self):
        comfy = self.comfy.get().strip()
        if not comfy or not (Path(comfy) / "custom_nodes").is_dir():
            self.msg.configure(text="Pick your ComfyUI folder (it has custom_nodes in it).")
            return
        self.tab._settings.update(comfy_dir=comfy, depth_device=self.device.get())
        save_settings(self.tab._settings)
        tmp = Path(tempfile.mkdtemp(prefix="mg_depth_"))
        inp, out = tmp / "photo.png", tmp / "depth.png"
        Image.fromarray(self.tab.st.image).save(inp)
        py = sys.executable
        if py.lower().endswith("pythonw.exe"):
            py = py[:-5] + ".exe"
        cmd = [py, str(Path(__file__).with_name("mg_depthgen.py")), "--comfy", comfy, "--in", str(inp), "--out", str(out),
               "--device", self.device.get()]
        self.msg.configure(text="Running… (loading the model takes a little while)")
        self.run_btn.configure(state="disabled")

        def work():
            try:
                kw = {"creationflags": 0x08000000} if os.name == "nt" else {}
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, **kw)
                ok, log = p.returncode == 0 and out.exists(), (p.stdout + p.stderr)[-1500:]
            except (OSError, subprocess.SubprocessError) as e:
                ok, log = False, str(e)
            self.tab._job_done = lambda: self.done(ok, log, out)     # (run on the Tk thread by the tab's tick)
        threading.Thread(target=work, daemon=True).start()

    def done(self, ok, log, out):
        try:
            self.run_btn.configure(state="normal")
        except tk.TclError:
            return
        if not ok:
            self.msg.configure(text="It didn't work:\n" + log.strip()[-900:])
            return
        self.tab.st.import_depth(out, True)
        self.tab.st.depth_note = "Depth Anything V2"
        self.tab.show_depth.set(True)
        self.tab.nb.select(SUBTABS.index("Depth"))
        self.tab._hint("Depth generated and lined up with the ground grid. Fit grid from map / Figures use map depth are in Depth ▾.")
        self.tab._changed()
        self.destroy()
