"""
Mantis Gallery — "Jiggle" tab for mg_convert.py.

Open an .mg straight off the SD card, place zones, fine-tune how they move, watch a live preview, and
save the hidden .jig file next to the image (named the way the Core2 expects). All the physics lives
in jiggle_lab.py — the same maths as the firmware, so what you see is what the Core2 does.

Mouse   click a shape button, then click the photo to place it · drag a zone to move it · drag its
        point circle to line it up · drag the small handle above it to rotate (either way) · wheel =
        size · Shift+wheel = rotate · in preview, click = poke
Keys    1/2/3 place Round/Half/Tear · Del delete · arrows nudge (Shift ×10) · +/- size · [ ] rotate
        (Shift ×15)
        Space preview · Ctrl+S save · Ctrl+Z / Ctrl+Y undo / redo · Esc cancel placing
Water   Smart: click / paint over the water · Paint / Erase drag · Shape: click around it, Enter fills (snapped) · Grid: fit the perspective
        · Flow / Shore line: click along the river (or beach), Enter keeps it · Backspace undoes a point
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import jiggle_lab as J

try:
    from PIL import Image, ImageTk
except ImportError:  # mg_convert already requires Pillow
    Image = ImageTk = None

BG, PANEL, FG, MUTED = "#1a1a1a", "#222222", "#e8e8e8", "#9e9e9e"
TEAL, LIME, YEL, PLUM = "#007373", "#b4ff00", "#ffe000", "#5d005d"
ZONE_COLS = ["#b4ff00", "#00e0e0", "#ff9cf0", "#ffb000"]


class JiggleTab(tk.Frame):
    def __init__(self, master, root=None):
        super().__init__(master, bg=BG)
        self.root = root or self.winfo_toplevel()
        self.ed = J.Editor()
        self.zoom = 2
        self.undo_stack: list = []
        self.redo_stack: list = []
        self._undo_t = 0.0
        self._last_dir = None
        self._photo = None
        self._img_item = None
        self._syncing = False

        # preview state
        self.preview = tk.BooleanVar(value=False)
        self.show_outlines = tk.BooleanVar(value=True)
        self.show_maps = tk.BooleanVar(value=False)
        self.slowmo = tk.StringVar(value="1")
        self.tilt = tk.DoubleVar(value=0.0)
        self.lean = tk.DoubleVar(value=0.0)
        self.dance = tk.BooleanVar(value=False)
        self.bpm = tk.DoubleVar(value=120.0)
        self.sim = None
        self._motion = None            # (kind, t0)
        self._dance_t = 0.0
        self._dance_amp = 0.0
        self._dance_sign = 1
        self._next_beat = 0.0
        self._last_tick = time.perf_counter()
        self._sim_t = 0.0

        # water editing
        self.wtool = tk.StringVar(value="off")      # off · smart · paint · erase · shape · line
        self.v_snap = tk.BooleanVar(value=True)      # Shape: fit the filled shape to the real edges
        self._marks = None                           # smart: the tap / stroke being made
        self._mlast = None
        self._shift = False
        self.brush = tk.IntVar(value=8)
        self._wpts = []                              # shape / line being clicked in
        self._wline_edit = -1                        # which line is being redrawn (-1 = a new one)

        self.status = tk.StringVar(value="Open an .mg from the SD card to begin.")
        self._build()
        self._bind_keys()
        self._render()
        self.after(16, self._tick)

    # ------------------------------------------------------------------------------------- layout
    def _btn(self, parent, text, cmd, bg="#2d2d2d", fg=FG, **kw):
        opts = {"activebackground": "#3a3a3a", "relief": "flat", "padx": 8, "pady": 3}
        opts.update(kw)
        return tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg, **opts)

    def _build(self):
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=10, pady=(8, 4))
        self._btn(top, "Open .mg…", self.open_dialog, bg=TEAL, fg="#000").pack(side="left")
        self.save_btn = self._btn(top, "Save to SD", self.save, bg=PLUM, fg="#fff")
        self.save_btn.pack(side="left", padx=6)
        self._btn(top, "Revert", self.revert).pack(side="left")
        self.file_lbl = tk.Label(top, text="", fg=MUTED, bg=BG, anchor="w")
        self.file_lbl.pack(side="left", padx=10, fill="x", expand=True)
        tk.Label(top, text="Zoom", fg=MUTED, bg=BG).pack(side="right")
        self.zoom_var = tk.StringVar(value="2×")
        zoom = ttk.Combobox(top, textvariable=self.zoom_var, values=["2×", "3×"], width=4, state="readonly")
        zoom.pack(side="right", padx=4)
        zoom.bind("<<ComboboxSelected>>", lambda e: self.set_zoom(int(self.zoom_var.get()[0])))

        mid = tk.Frame(self, bg=BG)
        mid.pack(fill="both", expand=True, padx=10)
        left = tk.Frame(mid, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, width=J.IMG_W * self.zoom, height=J.IMG_H * self.zoom,
                                bg="#000", highlightthickness=1, highlightbackground="#333", cursor="crosshair")
        self.canvas.pack(anchor="nw")
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._on_wheel(e, +1))
        self.canvas.bind("<Button-5>", lambda e: self._on_wheel(e, -1))
        self.canvas.bind("<Motion>", self._on_hover)
        self._build_preview_bar(left)

        # right: scrolling properties panel
        right = tk.Frame(mid, bg=PANEL, width=380)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)
        sc = tk.Canvas(right, bg=PANEL, highlightthickness=0, width=360)
        sb = ttk.Scrollbar(right, orient="vertical", command=sc.yview)
        self.panel = tk.Frame(sc, bg=PANEL)
        self.panel.bind("<Configure>", lambda e: sc.configure(scrollregion=sc.bbox("all")))
        sc.create_window((0, 0), window=self.panel, anchor="nw")
        sc.configure(yscrollcommand=sb.set)
        sc.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._build_panel(self.panel)

        tk.Label(self, textvariable=self.status, fg=LIME, bg=BG, anchor="w").pack(fill="x", padx=10, pady=(2, 6))

    def _section(self, parent, title):
        tk.Label(parent, text=title, fg=TEAL, bg=PANEL, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=8, pady=(10, 2))
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill="x", padx=8)
        return f

    def _build_panel(self, p):
        f = self._section(p, "Add a zone")
        self.shape_btns = []
        for i, (_, label, _, _) in enumerate(J.SHAPES):
            b = self._btn(f, f"{i + 1}  {label}", lambda i=i: self.arm(i))
            b.pack(side="left", padx=(0, 4))
            self.shape_btns.append(b)

        f = self._section(p, "Zones")
        self.zlist = tk.Listbox(f, height=4, bg="#161616", fg=FG, selectbackground=TEAL, relief="flat",
                                exportselection=False, activestyle="none")
        self.zlist.pack(fill="x")
        self.zlist.bind("<<ListboxSelect>>", self._on_list)
        row = tk.Frame(f, bg=PANEL)
        row.pack(fill="x", pady=(4, 0))
        self._btn(row, "Delete", self.delete).pack(side="left")
        self._btn(row, "Mirror copy", self.mirror_copy).pack(side="left", padx=4)
        self._btn(row, "Copy tuning to all", self.copy_tuning).pack(side="left")

        f = self._section(p, "Zone")
        self.v_shape = tk.StringVar()
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x")
        tk.Label(r, text="Shape", fg=MUTED, bg=PANEL, width=10, anchor="w").pack(side="left")
        cb = ttk.Combobox(r, textvariable=self.v_shape, values=[s[1] for s in J.SHAPES], width=10, state="readonly")
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self._edit(lambda: self.ed.set_shape([s[1] for s in J.SHAPES].index(self.v_shape.get()))))
        self.v_mat = tk.IntVar(value=J.NATURAL)
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x", pady=2)
        tk.Label(r, text="Material", fg=MUTED, bg=PANEL, width=10, anchor="w").pack(side="left")
        for i, m in enumerate(J.MATERIALS):
            tk.Radiobutton(r, text=m[1], variable=self.v_mat, value=i, bg=PANEL, fg=FG, selectcolor="#333",
                           activebackground=PANEL, command=lambda: self._edit(lambda: self.ed.set_material(self.v_mat.get()))).pack(side="left")
        self.v_x = self._slider(f, "X", 0, J.IMG_W, 1, lambda v: self._set_attr("cx", v))
        self.v_y = self._slider(f, "Y", 0, J.IMG_H, 1, lambda v: self._set_attr("cy", v))
        self.v_r = self._slider(f, "Size", J.R_MIN, J.R_MAX, 1, lambda v: self._set_attr("r", v))
        self.v_rot = self._slider(f, "Rotation", -180, 180, 1, lambda v: self._set_attr("rot", int(round(v)) % 360))
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x", pady=(0, 2))
        tk.Label(r, text="", bg=PANEL, width=10).pack(side="left")
        for d, t in ((-15, "↶ 15°"), (-1, "↶ 1°"), (1, "↷ 1°"), (15, "↷ 15°")):
            self._btn(r, t, lambda d=d: self._edit(lambda: self.ed.rotate(d), coalesce=True), padx=5, pady=0).pack(side="left", padx=1)
        self._btn(r, "0°", lambda: self._edit(lambda: self._set_attr("rot", 0)), padx=5, pady=0).pack(side="left", padx=(6, 0))
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x", pady=(4, 0))
        tk.Label(r, text="Body part", fg=MUTED, bg=PANEL, width=10, anchor="w").pack(side="left")
        self.v_preset = tk.StringVar(value="")
        pcb = ttk.Combobox(r, textvariable=self.v_preset, values=J.PRESET_NAMES, width=26, state="readonly")
        pcb.pack(side="left")
        pcb.bind("<<ComboboxSelected>>", lambda e: self.apply_preset(self.v_preset.get()))
        tk.Label(f, text="A preset sets the material, the point and the dials that matter for that body part;\n"
                         "then fine-tune below. Rotate so the zone's top faces where it joins the body.",
                 fg=MUTED, bg=PANEL, anchor="w", justify="left").pack(fill="x")

        f = self._section(p, "Control point  (line it up over the apex)")
        self.v_tip = tk.BooleanVar(value=False)
        tk.Checkbutton(f, text="Use a control point", variable=self.v_tip, bg=PANEL, fg=FG, selectcolor="#333",
                       activebackground=PANEL, command=lambda: self._edit(lambda: self.ed.set_tip(self.v_tip.get()))).pack(anchor="w")
        self.v_tu = self._slider(f, "Across", -0.6, 0.6, 0.01, lambda v: self._set_tip(v, None))
        self.v_tw = self._slider(f, "Up/down", -1.0, 0.9, 0.01, lambda v: self._set_tip(None, v))
        self.form_lbl = tk.Label(f, text="", fg=YEL, bg=PANEL, anchor="w", justify="left", wraplength=340)
        self.form_lbl.pack(fill="x", pady=(2, 0))
        tk.Label(f, text="Put it on the apex. Low = hanging (swings, ripples, sags); high = firm.\n"
                         "Rotate the zone so its top faces where it attaches to the body.",
                 fg=MUTED, bg=PANEL, anchor="w", justify="left").pack(fill="x")

        f = self._section(p, "Fine tuning  (grey = automatic from material / point · ↺ resets one)")
        self.tune_rows = {}
        info = {t[0]: t for t in J.TUNES}
        for group, keys in J.TUNE_GROUPS:
            tk.Label(f, text=group, fg=YEL, bg=PANEL, anchor="w").pack(fill="x", pady=(6, 0))
            for key in keys:
                _, label, lo, hi, help_ = info[key]
                self.tune_rows[key] = self._tune_row(f, key, label, lo, hi, help_)
        self._btn(f, "Reset all to automatic", self.reset_tuning).pack(anchor="w", pady=(6, 0))

        self._build_water_panel(p)
        self._build_depth_panel(p)
        self._build_hang_panel(p)

        f = self._section(p, "Response  (after a quick sideways shove)")
        self.resp = tk.Canvas(f, width=340, height=110, bg="#111", highlightthickness=0)
        self.resp.pack()
        self.resp_lbl = tk.Label(f, text="", fg=MUTED, bg=PANEL, anchor="w", justify="left")
        self.resp_lbl.pack(fill="x")

    def _build_water_panel(self, p):
        f = self._section(p, "Water  (lakes, rivers, the sea)")
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x")
        for mode, (_, label) in enumerate(J.WMODES):
            self._btn(r, f"+ {label}", lambda m=mode: self.add_water(m), padx=5).pack(side="left", padx=(0, 3))
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x", pady=(4, 0))
        tk.Label(r, text="Area", fg=MUTED, bg=PANEL, width=10, anchor="w").pack(side="left")
        self.v_wsel = tk.StringVar(value="")
        self.wsel_cb = ttk.Combobox(r, textvariable=self.v_wsel, values=[], width=16, state="readonly")
        self.wsel_cb.pack(side="left")
        self.wsel_cb.bind("<<ComboboxSelected>>", lambda e: self._pick_water())
        self._btn(r, "Delete", self.delete_water, padx=5).pack(side="left", padx=4)
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x", pady=2)
        tk.Label(r, text="Kind", fg=MUTED, bg=PANEL, width=10, anchor="w").pack(side="left")
        self.v_wmode = tk.IntVar(value=0)
        for mode, (_, label) in enumerate(J.WMODES):
            tk.Radiobutton(r, text=label, variable=self.v_wmode, value=mode, bg=PANEL, fg=FG, selectcolor="#333", activebackground=PANEL,
                           command=lambda: self._edit(lambda: self._set_wmode(self.v_wmode.get()))).pack(side="left")
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x", pady=2)
        tk.Label(r, text="Tool", fg=MUTED, bg=PANEL, width=10, anchor="w").pack(side="left")
        for key, label in (("off", "Off"), ("smart", "Smart"), ("paint", "Paint"), ("erase", "Erase"), ("shape", "Shape"), ("line", "Line"), ("grid", "Grid")):
            tk.Radiobutton(r, text=label, variable=self.wtool, value=key, bg=PANEL, fg=FG, selectcolor="#333", activebackground=PANEL,
                           indicatoron=False, padx=6, command=lambda: (self._use_water(), self._wtool_changed())).pack(side="left", padx=1)
        self.v_brush_row = self._slider(f, "Brush size", 1, 40, 1, lambda v: self.brush.set(int(v)))
        self.v_brush_row.set(8)
        self._pick_controls(f)
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x", pady=(2, 0))
        self._btn(r, "Finish (Enter)", self.finish_water_pts, padx=5).pack(side="left")
        self._btn(r, "Clear lines", lambda: self._edit(self._clear_lines), padx=5).pack(side="left", padx=3)
        self._btn(r, "Clear mask", lambda: self._edit(self._clear_mask), padx=5).pack(side="left")
        self.v_grid = tk.BooleanVar(value=False)
        tk.Checkbutton(f, text="Perspective grid: fit its squares to the water surface (Grid tool: drag the corners;\n"
                               "drag inside to move it; the wheel turns it). Far ripples then travel slower, smaller, foreshortened.",
                       variable=self.v_grid, bg=PANEL, fg=FG, selectcolor="#333", activebackground=PANEL, justify="left",
                       command=self._grid_toggled).pack(anchor="w")
        self.v_rain = tk.BooleanVar(value=False)
        tk.Checkbutton(f, text="Rain", variable=self.v_rain, bg=PANEL, fg=FG, selectcolor="#333", activebackground=PANEL,
                       command=lambda: self._edit(lambda: self._set_wprm("rain", 1.0 if self.v_rain.get() else 0.0))).pack(anchor="w")
        self.wrows = {}
        for key, label, lo, hi, d, help_ in J.WPARAMS:
            if key == "rain":
                continue
            self.wrows[key] = self._wparam_row(f, key, label, lo, hi, help_)
        self.water_lbl = tk.Label(f, text="", fg=YEL, bg=PANEL, anchor="w", justify="left", wraplength=340)
        self.water_lbl.pack(fill="x", pady=(2, 0))
        tk.Label(f, text="Paint just the water (stop short of the banks). River: draw a Line down the\n"
                         "middle of the current, following its bends, in the direction it flows.\n"
                         "Ocean: draw a Line along the shore where the waves land.",
                 fg=MUTED, bg=PANEL, anchor="w", justify="left").pack(fill="x")

    def _pick_controls(self, f):
        """Smart selection controls (shared by the water and depth panels)."""
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x", pady=(2, 0))
        self._btn(r, "− Smaller", lambda: self._smart_step(-1), padx=5).pack(side="left")
        self._btn(r, "Bigger +", lambda: self._smart_step(+1), padx=5).pack(side="left", padx=3)
        self._btn(r, "Snap edges", self._snap_all, padx=5).pack(side="left")
        self._btn(r, "Grow", lambda: self._edit(lambda: self.ed.grow_mask(1)), padx=4).pack(side="left", padx=(3, 0))
        self._btn(r, "Shrink", lambda: self._edit(lambda: self.ed.grow_mask(-1)), padx=4).pack(side="left")
        tk.Checkbutton(r, text="Shape snaps", variable=self.v_snap, bg=PANEL, fg=FG, selectcolor="#333",
                       activebackground=PANEL).pack(side="left", padx=4)
        tk.Label(f, text="Smart: click or paint over a sample - it finds the edges (Shift = remove).\n"
                         "Smaller / Bigger step to the next natural edge. Shape: click around it,\n"
                         "Enter - it's fitted to the real edges inside. Snap edges: fit a rough painted mask.",
                 fg=MUTED, bg=PANEL, anchor="w", justify="left").pack(fill="x")

    def _smart_done(self, n, g, add):
        if g is None:
            return
        steps = f" - edge {g.index + 1} of {len(g.levels)} (Smaller / Bigger)" if len(g.levels) > 1 else ""
        if n:
            self.status.set(f"{'Added' if add else 'Removed'} {n} px{steps}.")
        else:
            self.status.set("Nothing changed there - try Bigger, or another spot." if add else "Nothing to remove there.")

    def _smart_step(self, d):
        self._push_undo()
        r = self.ed.smart_step(d)
        if r is None:
            self.status.set("Click or paint over something first." if not getattr(self.ed, "_smart", None)
                            else ("That's the smallest edge." if d < 0 else "That's the biggest edge."))
            return
        self._smart_done(r[0], r[1], self.ed._smart["add"])
        self._after_edit()

    def _snap_all(self):
        self._push_undo()
        n = self.ed.snap_mask()
        self.status.set(f"Fitted to the edges ({n} px changed)." if n else "Paint a rough mask first (or nothing to fit).")
        self._after_edit()

    # ---------------------------------------------------------------------------------- depth panel
    def _build_depth_panel(self, p):
        import mg_depth as D
        f = self._section(p, "Depth  (parallax: planes slide as you tilt)")
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x")
        self._btn(r, "+ Plane", lambda: self._edit(lambda: (self.ed.add_plane(), self._use_depth())), padx=5).pack(side="left")
        self.v_psel = tk.StringVar(value="")
        self.psel_cb = ttk.Combobox(r, textvariable=self.v_psel, values=[], width=14, state="readonly")
        self.psel_cb.pack(side="left", padx=4)
        self.psel_cb.bind("<<ComboboxSelected>>", lambda e: self._pick_plane())
        self._btn(r, "Delete", lambda: self._edit(self.ed.delete_plane), padx=5).pack(side="left")
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x", pady=2)
        tk.Label(r, text="Tool", fg=MUTED, bg=PANEL, width=10, anchor="w").pack(side="left")
        self.v_pgrid = tk.BooleanVar(value=False)
        tk.Checkbutton(f, text="Ground grid: fit its squares flat on the ground (Grid tool: drag corners / inside; wheel turns).\n"
                               "It sets the horizon and how fast things recede; every depth comes from it.",
                       variable=self.v_pgrid, bg=PANEL, fg=FG, selectcolor="#333", activebackground=PANEL, justify="left",
                       command=self._pgrid_toggled).pack(anchor="w")
        for key, label in (("off", "Off"), ("smart", "Smart"), ("paint", "Paint"), ("erase", "Erase"), ("shape", "Shape"), ("grid", "Grid")):
            tk.Radiobutton(r, text=label, variable=self.wtool, value=key, bg=PANEL, fg=FG, selectcolor="#333", activebackground=PANEL,
                           indicatoron=False, padx=6, command=lambda: (self._use_depth(), self._wtool_changed())).pack(side="left", padx=1)
        self.v_pdepth = self._slider(f, "Depth", 0.05, 1.0, 0.05, lambda v: self._set_plane("depth", v))
        self.v_pramp = tk.BooleanVar(value=False)
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x")
        tk.Checkbutton(r, text="Ramp (ground running away): Ramp tool = click the near end, then the far end",
                       variable=self.v_pramp, bg=PANEL, fg=FG, selectcolor="#333", activebackground=PANEL,
                       command=lambda: self._edit(lambda: self._set_plane("ramp_on", self.v_pramp.get()))).pack(anchor="w")
        self.v_pfar = self._slider(f, "Far end", 0.0, 1.0, 0.05, lambda v: self._set_plane("far", v))
        self.prows = {}
        for key, label, lo, hi, d, help_ in D.PPARAMS:
            self.prows[key] = self._pparam_row(f, key, label, lo, hi, help_)
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x", pady=(4, 0))
        tk.Label(r, text="Tilt pad\n(drag)", fg=MUTED, bg=PANEL, width=10, anchor="w", justify="left").pack(side="left")
        self.pad = tk.Canvas(r, width=150, height=90, bg="#111", highlightthickness=1, highlightbackground="#444")
        self.pad.pack(side="left")
        self.pad.bind("<B1-Motion>", self._pad_drag); self.pad.bind("<Button-1>", self._pad_drag)
        self.pad.bind("<ButtonRelease-1>", lambda e: self._pad_set(0.0, 0.0))
        self.v_sweep = tk.BooleanVar(value=False)
        c = tk.Frame(r, bg=PANEL); c.pack(side="left", padx=6)
        tk.Checkbutton(c, text="Sweep", variable=self.v_sweep, bg=PANEL, fg=FG, selectcolor="#333", activebackground=PANEL,
                       command=self._depth_preview_on).pack(anchor="w")
        self._btn(c, "Depth map…", self.import_depth_map, padx=5).pack(anchor="w", pady=2)
        self.v_pn = tk.IntVar(value=3)
        tk.Spinbox(c, from_=1, to=D.PLANE_MAX, textvariable=self.v_pn, width=3).pack(anchor="w")
        self.depth_lbl = tk.Label(f, text="", fg=YEL, bg=PANEL, anchor="w", justify="left", wraplength=340)
        self.depth_lbl.pack(fill="x")
        tk.Label(f, text="Make a plane for each thing at a different distance (nearest = highest depth); the\n"
                         "rest is the background, which stays put. Smart: click it. Shape: click around it.\n"
                         "Depth map…: a grayscale depth image (e.g. Depth Anything from ComfyUI) → planes.",
                 fg=MUTED, bg=PANEL, anchor="w", justify="left").pack(fill="x")
        self._pad_target = (0.0, 0.0)

    # ----------------------------------------------------------------------------------- hang panel
    def _build_hang_panel(self, p):
        import mg_hang as G
        f = self._section(p, "Hanging  (hair, cloth, charms: they swing as you tilt)")
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x")
        self._btn(r, "+ Hang", lambda: self._edit(lambda: (self.ed.add_hang(), self._use_hang())), padx=5).pack(side="left")
        self.v_hsel = tk.StringVar(value="")
        self.hsel_cb = ttk.Combobox(r, textvariable=self.v_hsel, values=[], width=14, state="readonly")
        self.hsel_cb.pack(side="left", padx=4)
        self.hsel_cb.bind("<<ComboboxSelected>>", lambda e: self._pick_hang())
        self._btn(r, "Delete", lambda: self._edit(self.ed.delete_hang), padx=5).pack(side="left")
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x", pady=2)
        tk.Label(r, text="Kind", fg=MUTED, bg=PANEL, width=10, anchor="w").pack(side="left")
        self.v_hkind = tk.StringVar(value="hair")
        for key in G.KINDS:
            tk.Radiobutton(r, text=G.KIND_LABEL[key], variable=self.v_hkind, value=key, bg=PANEL, fg=FG, selectcolor="#333",
                           activebackground=PANEL, indicatoron=False, padx=6,
                           command=lambda: self._edit(lambda: self._set_hang("kind", self.v_hkind.get()))).pack(side="left", padx=1)
        r = tk.Frame(f, bg=PANEL); r.pack(fill="x", pady=2)
        tk.Label(r, text="Tool", fg=MUTED, bg=PANEL, width=10, anchor="w").pack(side="left")
        for key, label in (("off", "Off"), ("smart", "Smart"), ("paint", "Paint"), ("erase", "Erase"), ("shape", "Shape"), ("pin", "Pin")):
            tk.Radiobutton(r, text=label, variable=self.wtool, value=key, bg=PANEL, fg=FG, selectcolor="#333", activebackground=PANEL,
                           indicatoron=False, padx=6, command=lambda: (self._use_hang(), self._wtool_changed())).pack(side="left", padx=1)
        self.hrows = {}
        for key, label, lo, hi, d in G.HPARAMS:
            self.hrows[key] = self._slider(f, label.split(" (")[0], lo, hi, 0.05, lambda v, k=key: self._set_hang(k, v))
        self.hang_lbl = tk.Label(f, text="", fg=YEL, bg=PANEL, anchor="w", justify="left", wraplength=340)
        self.hang_lbl.pack(fill="x")
        tk.Label(f, text="Select it (Smart: click it; Shape: click around it, Enter), then Pin: click where it\n"
                         "hangs from - once for a point (an earring's hook), twice for a line (a ponytail's\n"
                         "tie, a waistband). No pin: it hangs from its top. Preview, then tilt or click it.",
                 fg=MUTED, bg=PANEL, anchor="w", justify="left").pack(fill="x")
        self._pin_next = 0

    def _use_hang(self):
        self.ed.target = "hang"
        if self.ed.hang is None and self.ed.image is not None:
            self.ed.add_hang(self.v_hkind.get() if hasattr(self, "v_hkind") else "hair")

    def _pick_hang(self):
        vals = list(self.hsel_cb.cget("values"))
        if self.v_hsel.get() in vals:
            self.ed.hang_sel = vals.index(self.v_hsel.get())
            self.ed.target = "hang"
            self._pin_next = 0
            self._sync_panel(); self._render()

    def _set_hang(self, what, v):
        h = self.ed.hang
        if h is None or self._syncing:
            return
        if what == "kind":
            h.kind = v
        else:
            import mg_hang as G
            if abs(float(v) - G.HDEF[what]) < 1e-6:
                h.prm.pop(what, None)
            else:
                h.prm[what] = float(v)
        self.ed.dirty = True
        self._sync_hang()

    def _sync_hang(self):
        hs = self.ed.hangs
        names = [f"{i + 1}: {h.kind}" for i, h in enumerate(hs)]
        self.hsel_cb.configure(values=names)
        h = self.ed.hang
        if h is None:
            self.v_hsel.set("")
            self.hang_lbl.configure(text="Nothing hanging yet: + Hang, then select it.")
            return
        self.v_hsel.set(names[self.ed.hang_sel])
        self.v_hkind.set(h.kind)
        for key, var in self.hrows.items():
            var.set(h.get(key))
        pin = "no pin: hangs from its top" if h.a[0] < 0 else (
            f"pinned at {int(h.a[0])},{int(h.a[1])}" if h.a == h.b else f"pinned {int(h.a[0])},{int(h.a[1])} – {int(h.b[0])},{int(h.b[1])}")
        tgt = "editing this" if self.ed.target == "hang" else "tools are elsewhere"
        self.hang_lbl.configure(text=f"{int(h.mask.sum())} px · {pin} · {tgt}")

    def _pparam_row(self, parent, key, label, lo, hi, help_):
        r = tk.Frame(parent, bg=PANEL); r.pack(fill="x")
        lbl = tk.Label(r, text=label, fg=MUTED, bg=PANEL, width=17, anchor="w"); lbl.pack(side="left")
        var = tk.DoubleVar(value=lo)
        res = 1.0 if key == "invert" else (0.5 if key in ("amount", "tilt") else 0.05)
        sc = tk.Scale(r, from_=lo, to=hi, resolution=res, orient="horizontal", variable=var, showvalue=True,
                      bg=PANEL, fg=FG, troughcolor="#333", highlightthickness=0, length=150,
                      command=lambda v, k=key: None if self._syncing else self._edit(lambda: self._set_pprm(k, float(v)), coalesce=True))
        sc.pack(side="left")
        self._btn(r, "↺", lambda k=key: self._edit(lambda: self._set_pprm(k, None)), padx=4, pady=0).pack(side="left", padx=2)
        for w_ in (lbl, sc):
            w_.bind("<Enter>", lambda e, h=help_, l=label: self.status.set(f"{l}: {h}"))
        return {"var": var, "label": lbl}

    def _pgrid_toggled(self):
        import mg_depth as D
        if self.ed.image is None:
            self.v_pgrid.set(False); return
        if self.ed.par is None:
            self.ed.par = D.Parallax()
        on = bool(self.v_pgrid.get())
        P = self.ed.par
        self._edit(lambda: setattr(P, "grid", [(20.0, 230.0), (300.0, 230.0), (215.0, 150.0), (105.0, 150.0)] if on else None))
        self.ed.target = "depth"
        if on:
            self.wtool.set("grid"); self._wtool_changed()
            self.status.set("Drag the grid until its squares lie flat on the ground, like floor tiles; the dashed line is its horizon.")

    def _use_depth(self):
        self.ed.target = "depth"
        if self.ed.plane is None and self.ed.image is not None:
            self.ed.add_plane()

    def _use_water(self):
        self.ed.target = "water"

    def _pick_plane(self):
        vals = list(self.psel_cb.cget("values"))
        if self.v_psel.get() in vals:
            self.ed.plane_sel = vals.index(self.v_psel.get())
            self.ed.target = "depth"
            self._sync_panel(); self._render()

    def _set_plane(self, what, v):
        p = self.ed.plane
        if p is None or self._syncing:
            return
        if what == "depth":
            p.depth = float(v)
        elif what == "far":
            if p.ramp:
                p.ramp = p.ramp[:4] + (float(v),)
        elif what == "ramp_on":
            p.ramp = (160, 230, 160, 120, max(0.0, p.depth - 0.4)) if v else None
        self.ed.dirty = True

    def _set_pprm(self, k, v):
        P = self.ed.par
        if P is None:
            return
        if v is None:
            P.prm.pop(k, None)
        else:
            P.prm[k] = v
        self.ed.dirty = True

    def _depth_preview_on(self):
        if not self.preview.get():
            self.preview.set(True); self._preview_toggled()

    def _pad_drag(self, e):
        self._pad_set(max(-1.0, min(1.0, (e.x - 75) / 70.0)), max(-1.0, min(1.0, (e.y - 45) / 40.0)))
        self._depth_preview_on()

    def _pad_set(self, vx, vy):
        self._pad_target = (vx, vy)
        c = self.pad
        c.delete("all")
        c.create_line(75, 0, 75, 90, fill="#333"); c.create_line(0, 45, 150, 45, fill="#333")
        x, y = 75 + vx * 70, 45 + vy * 40
        c.create_oval(x - 5, y - 5, x + 5, y + 5, fill=LIME, outline="")

    def import_depth_map(self):
        if self.ed.image is None:
            return
        path = filedialog.askopenfilename(title="A depth map for this picture (grayscale, bright = near)",
                                          filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("All", "*.*")])
        if not path:
            return
        from PIL import Image as PImage
        import struct as _st
        try:
            d = J.np.asarray(PImage.open(path).convert("L"), J.np.float32)
            w, h = _st.unpack("<HH", self.ed.mg_path.read_bytes()[4:8])
            self.ed.content_box = ((J.IMG_W - w) // 2, (J.IMG_H - h) // 2, w, h)
        except Exception as ex:
            messagebox.showerror("Depth map", f"Couldn't read it: {ex}")
            return
        n = int(self.v_pn.get() or 3)
        self._edit(lambda: self.ed.planes_from_depth_map(d, n, True))
        self.ed.target = "depth"
        self.status.set(f"{len(self.ed.par.planes)} planes from the depth map. Touch them up with Smart / Paint / Erase.")

    def _sync_depth(self):
        P = self.ed.par
        names = [f"{i + 1}: depth {p.depth:.2f}" + (" ramp" if p.ramp else "") for i, p in enumerate(P.planes)] if P else []
        self.psel_cb.configure(values=names)
        p = self.ed.plane
        self.v_pgrid.set(bool(P is not None and P.grid))
        if P is not None:
            for key, row in self.prows.items():
                row["var"].set(P.get(key))
                row["label"].configure(fg=LIME if key in P.prm else MUTED)
        if p is None:
            self.v_psel.set("")
            self.depth_lbl.configure(text="No planes yet: + Plane, then click the nearest thing (Smart).")
            return
        self.v_psel.set(names[self.ed.plane_sel])
        if P.grid:
            P.card_depths()
        self.v_pdepth.set(p.depth)
        self.v_pramp.set(bool(p.ramp))
        self.v_pfar.set(p.ramp[4] if p.ramp else p.depth)
        tgt = "editing this plane" if self.ed.target == "depth" else "tools are on the water"
        src = f"depth {p.depth:.2f} from the ground grid" if P.grid else "no ground grid: depth from the slider"
        self.depth_lbl.configure(text=f"{int(p.mask.sum())} px · {src} · {tgt}")

    def _wparam_row(self, parent, key, label, lo, hi, help_):
        r = tk.Frame(parent, bg=PANEL)
        r.pack(fill="x")
        lbl = tk.Label(r, text=label, fg=MUTED, bg=PANEL, width=17, anchor="w")
        lbl.pack(side="left")
        var = tk.DoubleVar(value=lo)
        res = 1.0 if key in ("horizon", "spacing") else 0.05
        sc = tk.Scale(r, from_=lo, to=hi, resolution=res, orient="horizontal", variable=var, showvalue=True,
                      bg=PANEL, fg=FG, troughcolor="#333", highlightthickness=0, length=150,
                      command=lambda v, k=key: None if self._syncing else self._edit(lambda: self._set_wprm(k, float(v)), coalesce=True))
        sc.pack(side="left")
        self._btn(r, "↺", lambda k=key: self._edit(lambda: self._set_wprm(k, None)), padx=4, pady=0).pack(side="left", padx=2)
        for w_ in (lbl, sc):
            w_.bind("<Enter>", lambda e, h=help_, l=label: self.status.set(f"{l}: {h}"))
        return {"var": var, "label": lbl}

    # --- water actions
    def add_water(self, mode):
        if self.ed.image is None:
            return
        if len(self.ed.waters) >= J.WATER_MAX:
            self.status.set(f"{J.WATER_MAX} water areas is the most — delete one first.")
            return
        self._edit(lambda: self.ed.add_water(mode))
        self.wtool.set("paint" if mode == J.W_STILL else "shape")
        self._wtool_changed()
        self.status.set({J.W_STILL: "Paint over the lake / pond (or use Shape: click around it, Enter).",
                         J.W_RIVER: "Cover the river (Shape or Paint), then a Line down the current.",
                         J.W_OCEAN: "Cover the sea (Shape or Paint), then a Line along the shore."}[mode])

    def delete_water(self):
        if self.ed.water is not None:
            self._edit(self.ed.delete_water)

    def _pick_water(self):
        vals = list(self.wsel_cb.cget("values"))
        if self.v_wsel.get() in vals:
            self.ed.water_sel = vals.index(self.v_wsel.get())
            self._sync_panel(); self._render()

    def _set_wmode(self, m):
        w = self.ed.water
        if w is not None:
            w.mode = m
            self.ed.dirty = True

    def _set_wprm(self, k, v):
        w = self.ed.water
        if w is None:
            return
        if v is None:
            w.prm.pop(k, None)
        else:
            w.prm[k] = v
        self.ed.dirty = True

    def _clear_lines(self):
        if self.ed.water is not None:
            self.ed.water.lines = []
            self.ed.dirty = True

    def _clear_mask(self):
        if self.ed.water is not None:
            self.ed.water.mask[:] = False
            self.ed.dirty = True

    def _wtool_changed(self):
        self._wpts = []
        self.ed.smart_forget()
        if self.wtool.get() != "off":
            if self.preview.get():
                self.preview.set(False); self._preview_toggled()
            if self.ed.target == "water" and self.ed.water is None and self.ed.image is not None:
                self._edit(lambda: self.ed.add_water(J.W_STILL))
            self.arm(-1)
        self._render()

    def finish_water_pts(self):
        tool, pts = self.wtool.get(), list(self._wpts)
        if tool == "shape":
            if len(pts) < 3:
                self.status.set("Click 3 or more points around the water first."); return
            snap = bool(self.v_snap.get())
            self._edit(lambda: self.ed.water_fill(pts, snap=snap))
            self.status.set("Filled and fitted to the edges. Smart / Paint / Erase to touch up." if snap
                            else "Filled. Add more shapes, or Paint / Erase to touch up the edges.")
        elif tool == "line":
            if len(pts) < 2:
                self.status.set("Click 2 or more points along the water first."); return
            w = self.ed.water
            if w is not None and len([l for l in w.lines if l]) >= J.WLINE_MAX:
                self.status.set(f"{J.WLINE_MAX} lines is the most — Clear lines first."); return
            self._edit(lambda: self.ed.water_set_line(len([l for l in self.ed.water.lines if l]), pts))
            self.status.set("Line kept. It bends with the river; the arrow shows the way it flows.")
        self._wpts = []
        self._render()

    def _grid_toggled(self):
        w = self.ed.water
        if w is None:
            self.v_grid.set(False); return
        on = bool(self.v_grid.get())
        self._edit(lambda: setattr(w, "grid", (J.default_grid(w.mask) if w.mask is not None and w.mask.any() else
                                              [(40.0, 220.0), (280.0, 220.0), (230.0, 120.0), (90.0, 120.0)]) if on else None))
        if on:
            self.wtool.set("grid"); self._use_water(); self._wtool_changed()
            self.status.set("Drag the grid's corners until its squares lie flat on the water (like floor tiles).")

    def _grid_obj(self):
        return self.ed.par if self.ed.target == "depth" else self.ed.water

    def _water_press(self, x, y):
        tool = self.wtool.get()
        if tool == "pin":
            self._use_hang()
            self._push_undo()
            self.ed.pin(x, y, second=self._pin_next == 1)
            self._pin_next = 1 - self._pin_next
            self.status.set("Pinned at a point - click again for a line (a tie, a waistband)." if self._pin_next
                            else "Pinned along a line. Click again to start over.")
            self._sync_panel(); self._render()
            return
        if tool == "grid":
            w = self._grid_obj()
            if w is None or not w.grid:
                self.status.set("Tick the grid box first (water: Perspective grid · depth: Ground grid)."); return
            d = [math.hypot(x - px, y - py) for px, py in w.grid]
            k = min(range(4), key=lambda i: d[i])
            self._push_undo()
            self._gdrag = (k, x, y) if d[k] <= 12 / max(1, self.zoom) * 2 + 6 else (-1, x, y)
            return
        if tool == "ramp":
            self._wpts.append((x, y))
            if len(self._wpts) >= 2 and self.ed.plane is not None:
                (ax, ay), (bx, by) = self._wpts[:2]
                p = self.ed.plane
                far = p.ramp[4] if p.ramp else max(0.0, p.depth - 0.4)
                self._edit(lambda: setattr(p, "ramp", (int(ax), int(ay), int(bx), int(by), far)))
                self.status.set("Ramp set: depth runs from this plane's Depth at the first click to Far end at the second.")
                self._wpts = []
            self._render()
            return
        if tool == "smart":
            self._marks = J.np.zeros((J.IMG_H, J.IMG_W), bool)
            J.mask_brush(self._marks, int(round(x)), int(round(y)), 4, True)
            self._mlast = (x, y)
            self._render()
            return
        if tool in ("paint", "erase"):
            self._push_undo()
            self.ed.mask_paint(x, y, self.brush.get(), erase=(tool == "erase"))
            self._wlast = (x, y)
            self._render()
        elif tool in ("shape", "line"):
            self._wpts.append((x, y))
            self._render()

    def _water_drag(self, x, y):
        if self.wtool.get() == "grid" and getattr(self, "_gdrag", None):
            w = self._grid_obj()
            k, lx, ly = self._gdrag
            if w is not None and w.grid:
                g = list(w.grid)
                if k >= 0:
                    g[k] = (min(J.IMG_W + 200, max(-200, x)), min(J.IMG_H + 200, max(-400, y)))
                else:
                    g = [(px + x - lx, py + y - ly) for px, py in g]
                w.grid = g
                self.ed.dirty = True
                self._gdrag = (k, x, y)
                self._render()
            return
        if self.wtool.get() == "smart" and self._marks is not None:
            lx, ly = self._mlast
            n = int(max(abs(x - lx), abs(y - ly)) / 2) + 1
            for k in range(1, n + 1):
                J.mask_brush(self._marks, int(round(lx + (x - lx) * k / n)), int(round(ly + (y - ly) * k / n)), 4, True)
            self._mlast = (x, y)
            self._render()
            return
        if self.wtool.get() in ("paint", "erase") and getattr(self, "_wlast", None):
            lx, ly = self._wlast
            n = int(max(abs(x - lx), abs(y - ly)) / 2) + 1
            for k in range(1, n + 1):
                self.ed.mask_paint(lx + (x - lx) * k / n, ly + (y - ly) * k / n, self.brush.get(), erase=(self.wtool.get() == "erase"))
            self._wlast = (x, y)
            self._render()

    def _water_release(self):
        if self.wtool.get() == "grid":
            self._gdrag = None
            self._after_edit()
            return
        if self.wtool.get() == "smart" and self._marks is not None:
            marks, self._marks = self._marks, None
            self._push_undo()
            add = not self._shift
            self.status.set("Selecting..."); self.update_idletasks()
            n, g = self.ed.smart_pick(marks, add=add)
            self._smart_done(n, g, add)
            self._after_edit()
            return
        if self.wtool.get() in ("paint", "erase"):
            self._wlast = None
            self._after_edit()

    def _slider(self, parent, label, lo, hi, res, cb):
        r = tk.Frame(parent, bg=PANEL)
        r.pack(fill="x")
        tk.Label(r, text=label, fg=MUTED, bg=PANEL, width=10, anchor="w").pack(side="left")
        var = tk.DoubleVar(value=lo)
        s = tk.Scale(r, from_=lo, to=hi, resolution=res, orient="horizontal", variable=var, showvalue=True,
                     bg=PANEL, fg=FG, troughcolor="#333", highlightthickness=0, length=230,
                     command=lambda v: None if self._syncing else self._edit(lambda: cb(float(v)), coalesce=True))
        s.pack(side="left", fill="x", expand=True)
        return var

    def _tune_row(self, parent, key, label, lo, hi, help_):
        r = tk.Frame(parent, bg=PANEL)
        r.pack(fill="x")
        lbl = tk.Label(r, text=label, fg=MUTED, bg=PANEL, width=17, anchor="w")
        lbl.pack(side="left")
        var = tk.DoubleVar(value=lo)
        res = 1.0 if key in ("side", "down") else 0.01
        s = tk.Scale(r, from_=lo, to=hi, resolution=res, orient="horizontal", variable=var, showvalue=True,
                     bg=PANEL, fg=FG, troughcolor="#333", highlightthickness=0, length=150,
                     command=lambda v, k=key: None if self._syncing else self._edit(lambda: self.ed.set_tune(k, float(v)), coalesce=True))
        s.pack(side="left")
        self._btn(r, "↺", lambda k=key: self._edit(lambda: self.ed.set_tune(k, None)), padx=4, pady=0).pack(side="left", padx=2)
        for w in (lbl, s):
            w.bind("<Enter>", lambda e, h=help_, l=label: self.status.set(f"{l}: {h}"))
        return {"var": var, "label": lbl, "scale": s}

    def _build_preview_bar(self, parent):
        bar = tk.Frame(parent, bg=BG)
        bar.pack(fill="x", pady=(8, 0))
        tk.Checkbutton(bar, text="▶ Live preview", variable=self.preview, bg=BG, fg=LIME, selectcolor="#333",
                       activebackground=BG, font=("Segoe UI", 10, "bold"), command=self._preview_toggled).pack(side="left")
        for kind, label in J.MOTIONS:
            self._btn(bar, label, lambda k=kind: self.start_motion(k), padx=5).pack(side="left", padx=1)

        bar3 = tk.Frame(parent, bg=BG)
        bar3.pack(fill="x", pady=(2, 0))
        tk.Label(bar3, text="Hold the Core2:  turned", fg=MUTED, bg=BG).pack(side="left")
        tk.Scale(bar3, from_=-180, to=180, orient="horizontal", variable=self.tilt, showvalue=True, length=150,
                 bg=BG, fg=FG, troughcolor="#333", highlightthickness=0).pack(side="left")
        tk.Label(bar3, text="°  leaned", fg=MUTED, bg=BG).pack(side="left")
        tk.Scale(bar3, from_=-90, to=90, orient="horizontal", variable=self.lean, showvalue=True, length=110,
                 bg=BG, fg=FG, troughcolor="#333", highlightthickness=0).pack(side="left")
        tk.Label(bar3, text="°", fg=MUTED, bg=BG).pack(side="left")
        self._btn(bar3, "Upright", lambda: (self.tilt.set(0), self.lean.set(0)), padx=5).pack(side="left", padx=4)

        bar2 = tk.Frame(parent, bg=BG)
        bar2.pack(fill="x", pady=(2, 0))
        tk.Checkbutton(bar2, text="Dance", variable=self.dance, bg=BG, fg=FG, selectcolor="#333",
                       activebackground=BG, command=self._dance_toggled).pack(side="left")
        tk.Spinbox(bar2, from_=60, to=200, increment=1, textvariable=self.bpm, width=5,
                   bg="#2a2a2a", fg=FG, buttonbackground="#333").pack(side="left")
        tk.Label(bar2, text="BPM   Speed", fg=MUTED, bg=BG).pack(side="left")
        for v, l in (("1", "1×"), ("0.5", "½×"), ("0.25", "¼×")):
            tk.Radiobutton(bar2, text=l, variable=self.slowmo, value=v, bg=BG, fg=FG, selectcolor="#333",
                           activebackground=BG).pack(side="left")
        tk.Checkbutton(bar2, text="Outlines", variable=self.show_outlines, bg=BG, fg=FG, selectcolor="#333",
                       activebackground=BG, command=self._render).pack(side="left", padx=(10, 0))
        tk.Checkbutton(bar2, text="Show maps", variable=self.show_maps, bg=BG, fg=FG, selectcolor="#333",
                       activebackground=BG, command=self._render).pack(side="left")

    def _bind_keys(self):
        r = self.root
        def on(seq, fn):
            r.bind(seq, lambda e: None if not self._active() else fn(e), add="+")
        on("<Delete>", lambda e: self.delete())
        on("<BackSpace>", lambda e: (self._wpts.pop(), self._render()) if self._wpts else self.delete())
        on("<Escape>", lambda e: (self._wpts.clear(), self._render()) if self._wpts else self.arm(-1))
        on("<Return>", lambda e: self.finish_water_pts() if self.wtool.get() in ("shape", "line") else None)
        on("1", lambda e: self.arm(0)); on("2", lambda e: self.arm(1)); on("3", lambda e: self.arm(2))
        on("<plus>", lambda e: self._edit(lambda: self.ed.resize(1.06)))
        on("<equal>", lambda e: self._edit(lambda: self.ed.resize(1.06)))
        on("<minus>", lambda e: self._edit(lambda: self.ed.resize(1 / 1.06)))
        on("<bracketleft>", lambda e: self._edit(lambda: self.ed.rotate(-5)))
        on("<bracketright>", lambda e: self._edit(lambda: self.ed.rotate(5)))
        on("<braceleft>", lambda e: self._edit(lambda: self.ed.rotate(-15)))
        on("<braceright>", lambda e: self._edit(lambda: self.ed.rotate(15)))
        for key, dx, dy in (("Left", -1, 0), ("Right", 1, 0), ("Up", 0, -1), ("Down", 0, 1)):
            on(f"<{key}>", lambda e, dx=dx, dy=dy: self._edit(lambda: self.ed.nudge(dx, dy), coalesce=True))
            on(f"<Shift-{key}>", lambda e, dx=dx, dy=dy: self._edit(lambda: self.ed.nudge(dx * 10, dy * 10), coalesce=True))
        on("<space>", lambda e: (self.preview.set(not self.preview.get()), self._preview_toggled()))
        on("<Control-s>", lambda e: self.save())
        on("<Control-z>", lambda e: self.undo())
        on("<Control-y>", lambda e: self.redo())
        on("<Control-Z>", lambda e: self.redo())

    def _active(self):
        """Keyboard shortcuts only while this tab is showing and focus isn't in a text field."""
        try:
            nb = self.master
            if hasattr(nb, "select") and nb.select() != str(self):
                return False
            f = self.root.focus_get()
            return not isinstance(f, (tk.Entry, tk.Spinbox, ttk.Entry, ttk.Combobox))
        except Exception:
            return True

    # ------------------------------------------------------------------------------------- files
    def open_dialog(self):
        p = filedialog.askopenfilename(title="Open an image from the SD card (/images/…)",
                                       initialdir=self._last_dir or None,
                                       filetypes=[("Mantis Gallery image", "*.mg"), ("All files", "*.*")])
        if p:
            self.open_path(p)

    def open_path(self, path):
        if self.ed.dirty and self.ed.mg_path and not messagebox.askyesno(
                "Jiggle", "Discard unsaved changes to this image's zones?"):
            return False
        try:
            had = self.ed.open_mg(path)
        except Exception as e:
            messagebox.showerror("Jiggle", f"Couldn't open {path}:\n{e}")
            return False
        self._last_dir = str(Path(path).parent)
        self.undo_stack.clear(); self.redo_stack.clear()
        self.file_lbl.configure(text=f"{Path(path).name}  →  {J.jig_path_for(path).name}")
        self.status.set(f"Loaded {len(self.ed.zones)} zone(s) from {J.jig_path_for(path).name}" if had
                        else "No zones yet — pick a shape (1/2/3) and click the photo.")
        self._restart_sim()
        self._sync_panel()
        self._render()
        return True

    def save(self):
        if not self.ed.mg_path:
            messagebox.showinfo("Jiggle", "Open an .mg image first.")
            return None
        try:
            jp = self.ed.save()
        except OSError as e:
            messagebox.showerror("Jiggle", f"Couldn't save:\n{e}")
            return None
        self.status.set(f"Saved {len(self.ed.zones)} zone(s) → {jp}" if self.ed.zones
                        else f"No zones — removed {jp.name}")
        self._refresh_list()
        return jp

    def revert(self):
        if self.ed.mg_path:
            self.ed.dirty = False
            self.open_path(self.ed.mg_path)

    # ------------------------------------------------------------------------------------- editing
    def _push_undo(self, coalesce=False):
        now = time.perf_counter()
        if coalesce and now - self._undo_t < 0.6 and self.undo_stack:
            self._undo_t = now
            return
        self._undo_t = now
        self.undo_stack.append((self.ed.snapshot(), self.ed.selected))
        if len(self.undo_stack) > 200:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _edit(self, fn, coalesce=False):
        """Run an edit with undo, then refresh everything that depends on the zones."""
        self._push_undo(coalesce)
        fn()
        self._after_edit()

    def _after_edit(self):
        self._restart_sim(keep_motion=True)
        self._sync_panel()
        self._render()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append((self.ed.snapshot(), self.ed.selected))
            snap, self.ed.selected = self.undo_stack.pop()
            self.ed.restore(snap)
            self.ed.dirty = True
            self._after_edit()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append((self.ed.snapshot(), self.ed.selected))
            snap, self.ed.selected = self.redo_stack.pop()
            self.ed.restore(snap)
            self.ed.dirty = True
            self._after_edit()

    def arm(self, shape):
        if shape < 0:
            self.ed.armed = -1
        elif not self.ed.arm(shape):
            self.status.set(f"{J.MAX_ZONES} zones is the most the Core2 handles — delete one first.")
        else:
            if self.ed.armed >= 0:
                self.status.set(f"Click the photo to place a {J.SHAPES[shape][1]} zone.")
        for i, b in enumerate(self.shape_btns):
            b.configure(bg=TEAL if self.ed.armed == i else "#2d2d2d")

    def delete(self):
        if self.ed.zone:
            self._edit(self.ed.delete)

    def mirror_copy(self):
        z = self.ed.zone
        if not z:
            return
        if len(self.ed.zones) >= J.MAX_ZONES:
            self.status.set(f"{J.MAX_ZONES} zones is the most — delete one first.")
            return
        def do():
            self.ed.zones.append(J.mirrored(z))
            self.ed.selected = len(self.ed.zones) - 1
            self.ed.dirty = True
        self._edit(do)
        self.status.set("Mirrored across the middle of the photo — drag it into place.")

    def copy_tuning(self):
        z = self.ed.zone
        if not z:
            return
        def do():
            for o in self.ed.zones:
                if o is not z:
                    o.material = z.material
                    side = o.tune.get("side")
                    o.tune = {k: v for k, v in z.tune.items() if k != "side"}
                    if side is not None:
                        o.tune["side"] = side
            self.ed.dirty = True
        self._edit(do)
        self.status.set("Material and fine tuning copied to every other zone (dance side left as is).")

    def apply_preset(self, name):
        if not self.ed.zone:
            self.status.set("Select a zone first, then pick a body part.")
            return
        self._edit(lambda: self.ed.apply_preset(name))
        self.status.set(f"{name}: preset applied — drag the point onto the apex, then fine-tune.")

    def reset_tuning(self):
        if self.ed.zone:
            self._edit(lambda: self.ed.zone.tune.clear())

    def _set_attr(self, attr, v):
        z = self.ed.zone
        if z:
            setattr(z, attr, v)
            if attr == "r":
                self.ed.last_r = v
            self.ed.dirty = True

    def _set_tip(self, u, w):
        z = self.ed.zone
        if not z:
            return
        nu = z.tip_u if u is None else u
        nw = z.tip_w if w is None else w
        if z.point_ok(nu, nw):
            z.tip_u, z.tip_w = nu, nw
            z.tip = True
            self.ed.dirty = True

    def set_zoom(self, z):
        self.zoom = z
        self.canvas.configure(width=J.IMG_W * z, height=J.IMG_H * z)
        self._render()

    # ------------------------------------------------------------------------------------- mouse
    def _img_xy(self, e):
        return e.x / self.zoom, e.y / self.zoom

    def _on_press(self, e):
        if self.ed.image is None:
            return
        x, y = self._img_xy(e)
        self._shift = bool(getattr(e, "state", 0) & 0x0001)
        if self.wtool.get() != "off" and not self.preview.get():
            self._water_press(x, y)
            return
        if self.preview.get():
            if self.sim and self.sim.poke(x, y):
                self.status.set("Poke!")
            return
        self._push_undo()
        self.ed.press(x, y)
        self.arm(-1)
        self._after_edit()

    def _on_drag(self, e):
        if self.ed.image is None or self.preview.get():
            return
        x, y = self._img_xy(e)
        if self.wtool.get() != "off":
            self._water_drag(x, y)
            return
        self.ed.drag(x, y)
        self._sync_panel()
        self._render()

    def _on_release(self, e):
        if self.ed.image is None or self.preview.get():
            return
        if self.wtool.get() != "off":
            self._water_release()
            return
        self.ed.release(*self._img_xy(e))
        self._after_edit()

    def _on_wheel(self, e, direction=None):
        d = direction if direction is not None else (1 if getattr(e, "delta", 0) > 0 else -1)
        if self.wtool.get() == "grid" and not self.preview.get():   # Grid tool: the wheel turns it (levels the horizon)
            w = self._grid_obj()
            if w is not None and w.grid:
                cx = sum(p[0] for p in w.grid) / 4; cy = sum(p[1] for p in w.grid) / 4
                c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
                self._push_undo()
                w.grid = [(cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c) for x, y in w.grid]
                self.ed.dirty = True
                self._after_edit()
            return
        if not self.ed.zone or self.preview.get():
            return
        if getattr(e, "state", 0) & 0x0001:            # Shift: rotate
            self._edit(lambda: self.ed.rotate(5 * d), coalesce=True)
        else:
            self._edit(lambda: self.ed.resize(1.04 if d > 0 else 1 / 1.04), coalesce=True)

    def _on_hover(self, e):
        if self.ed.image is None or self.preview.get():
            return
        x, y = self._img_xy(e)
        if self.ed.near_rot(x, y):
            self.canvas.configure(cursor="exchange")
            self.status.set("Drag to rotate — either way round.")
            return
        self.canvas.configure(cursor="fleur" if (self.ed.near_tip(x, y) or self.ed.hit(x, y) >= 0) else "crosshair")

    def _on_list(self, e=None):
        sel = self.zlist.curselection()
        if sel:
            self.ed.selected = int(sel[0])
            self._sync_panel()
            self._render()

    # ------------------------------------------------------------------------------------- panel
    def _refresh_list(self):
        self.zlist.delete(0, "end")
        for i, z in enumerate(self.ed.zones):
            extra = " • point" if z.tip else ""
            tuned = f" • {len(z.tune)} tuned" if z.tune else ""
            self.zlist.insert("end", f"{i + 1}  {J.SHAPES[z.shape][1]}  ({J.MATERIALS[z.material][1]}){extra}{tuned}")
        if 0 <= self.ed.selected < len(self.ed.zones):
            self.zlist.selection_clear(0, "end")
            self.zlist.selection_set(self.ed.selected)

    def _sync_water(self):
        names = [f"{i + 1}: {J.WMODES[w.mode][1]}" for i, w in enumerate(self.ed.waters)]
        self.wsel_cb.configure(values=names)
        w = self.ed.water
        if w is None:
            self.v_wsel.set("")
            self.water_lbl.configure(text="No water yet: use + Still water / + River / + Ocean waves.")
            return
        self.v_wsel.set(names[self.ed.water_sel])
        self.v_wmode.set(w.mode)
        self.v_grid.set(bool(w.grid))
        self.v_rain.set(w.get("rain") >= 0.5)
        for key, row in self.wrows.items():
            row["var"].set(w.get(key))
            row["label"].configure(fg=LIME if key in w.prm else MUTED)
        px = int(w.mask.sum()) if w.mask is not None else 0
        nl = len([l for l in w.lines if l])
        need = ""
        if w.mode == J.W_RIVER and not nl:
            need = " · add a Line down the river"
        if w.mode == J.W_OCEAN and not nl:
            need = " · add a Line along the shore"
        self.water_lbl.configure(text=f"{px} px of water · {nl} line(s){need}")

    def _sync_panel(self):
        self._syncing = True
        try:
            self._refresh_list()
            self._sync_water()
            self._sync_depth()
            self._sync_hang()
            z = self.ed.zone
            if z:
                self.v_shape.set(J.SHAPES[z.shape][1])
                self.v_mat.set(z.material)
                self.v_x.set(round(z.cx)); self.v_y.set(round(z.cy))
                self.v_r.set(round(z.r)); self.v_rot.set(((z.rot + 180) % 360) - 180)
                self.v_tip.set(z.tip)
                self.v_tu.set(round(z.tip_u, 2)); self.v_tw.set(round(z.tip_w, 2))
                sim = J.ZoneSim(J.replace(z, tune=dict(z.tune)))
                defaults = self._tune_defaults(z, sim)
                # what the point implies, so the unset sliders show the values actually in use
                if z.tip:
                    d = sim.droop
                    defaults.update(z1=sim.z1, stiff=sim.stiffen, couple=sim.couple, sag=sim.sag, pend=sim.pend,
                                    proj=0.6 - 0.25 * d)
                    kind = "hanging / droopy" if d > 0.35 else ("firm / perky" if d < -0.35 else "typical")
                    self.form_lbl.configure(text=f"Point reads as: {kind}  (droop {d:+.2f}, elongation {sim.elong:.2f})"
                                                 + (f" · fold guard trimmed travel to {sim.fold_k:.0%}" if sim.fold_k < 0.995 else ""))
                else:
                    self.form_lbl.configure(text="No control point: motion follows the outline only.")
                defaults.update(travel=sim.form_travel, slack=sim.slack, anchor=z.tuned("anchor", J.FORM_MIX if z.tip else 0.0))
                for key, row in self.tune_rows.items():
                    set_ = key in z.tune
                    row["var"].set(z.tuned(key, 0.0) if set_ else defaults[key])
                    row["label"].configure(fg=LIME if set_ else MUTED)
            self._draw_response()
        finally:
            self._syncing = False

    @staticmethod
    def _tune_defaults(z, sim):
        mi = J.MATERIALS[z.material]
        return {"f1": mi[2], "z1": mi[3], "f2": mi[4], "z2": mi[5], "stiff": mi[6], "sens": mi[7], "couple": mi[8],
                "travel": 1.0, "bulge": 1.0, "stretch": 1.0, "sag": 1.0, "pend": J.PENDULUM,
                "proj": 0.6, "tail": 1.0, "psize": 1.0, "dance": 1.0, "side": 0.0,
                "mass": 1.0, "anchor": 0.0, "slack": 0.2, "down": 0.0, "contact": 1.0}

    def _draw_response(self):
        c = self.resp
        c.delete("all")
        z = self.ed.zone
        if not z or J.np is None:
            self.resp_lbl.configure(text="Select a zone to see how it settles.")
            return
        t, v, st = J.step_response(z)
        W, H = 340, 110
        peak = max(1.0, st["peak_px"])
        c.create_line(0, H / 2, W, H / 2, fill="#333")
        pts = []
        for i in range(0, len(t), 3):
            pts += [t[i] / t[-1] * W, H / 2 - v[i] / peak * (H / 2 - 6)]
        if len(pts) >= 4:
            c.create_line(*pts, fill=LIME, width=2, smooth=True)
        c.create_line(st["settle_ms"] / 1000 / t[-1] * W, 0, st["settle_ms"] / 1000 / t[-1] * W, H, fill=TEAL, dash=(3, 3))
        self.resp_lbl.configure(text=f"{st['hz']:.1f} Hz here · peak {st['peak_px']:.1f} px · "
                                     f"{st['swings']} visible swings · settles in {st['settle_ms']:.0f} ms")

    # ------------------------------------------------------------------------------------- preview
    def _restart_sim(self, keep_motion=False):
        if J.np is None or self.ed.image is None:
            self.sim = None
            return
        self.sim = J.Sim(self.ed.zones, self.ed.image, self.ed.waters, self.ed.par, [J.default_pin(h.copy()) for h in self.ed.hangs])
        if not keep_motion:
            self._motion = None

    def _preview_toggled(self):
        self._restart_sim()
        self.status.set("Preview: pick a motion, turn / lean the Core2, dance, or click a zone to poke it."
                        if self.preview.get() else "Editing.")
        self._render()

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
        """Advance the preview by dt seconds of simulated time (also used by tests)."""
        if not self.sim:
            return
        t0 = self._sim_t
        self._sim_t += dt
        acc = J.motion_accel("none", 0.0, self.tilt.get(), self.lean.get())
        twist = (0.0, 0.0)
        if self._motion:
            kind, ts = self._motion
            if self._sim_t - ts > 5.0:
                self._motion = None
            else:
                acc = J.motion_accel(kind, self._sim_t - ts, self.tilt.get(), self.lean.get())
                twist = J.motion_twist(kind, self._sim_t - ts)
        # dance: steady beats, strength eases in over ~1.5 beats and out over 1 s when switched off
        period = 60.0 / max(40.0, float(self.bpm.get() or 120))
        if self.dance.get():
            self._dance_amp = min(1.0, self._dance_amp + dt / (1.5 * period))
        else:
            self._dance_amp = max(0.0, self._dance_amp - dt)
        if self._dance_amp > 0 and self._sim_t >= self._next_beat:
            self._dance_sign = -self._dance_sign
            self.sim.beat(self._dance_amp, self._dance_sign)
            self._next_beat = max(self._next_beat + period, t0)
        if getattr(self.sim, "depth", None) is not None:
            tx, ty = self._pad_target
            if self.v_sweep.get():
                tx, ty = 0.85 * math.sin(self._sim_t * 1.3), 0.5 * math.sin(self._sim_t * 0.9 + 1.0)
            self.sim.view_target = (tx, ty)
        self.sim.step(dt, acc, self._dance_amp, twist)

    def _tick(self):
        now = time.perf_counter()
        dt = min(0.05, now - self._last_tick)
        self._last_tick = now
        try:
            if self.preview.get() and self.sim:
                speed = float(self.slowmo.get())
                self.step_preview(dt * speed)
                self._render()
        finally:
            self.after(16, self._tick)

    # ------------------------------------------------------------------------------------- drawing
    def frame_rgb(self):
        """What the canvas shows (without outlines), as a uint8 RGB array."""
        img = self.ed.image
        if img is None:
            return None
        if self.preview.get() and self.sim:
            img = self.sim.render()
        if self.show_maps.get() and self.ed.zone and J.np is not None:
            img = J.maps_overlay(img, self.ed.zone)
        if not self.preview.get() and self.ed.waters and J.np is not None and ((self.wtool.get() != "off" and self.ed.target == "water") or self.show_outlines.get()):
            img = img.copy()                       # tint the water so you can see what's covered
            for i, w in enumerate(self.ed.waters):
                if w.mask is None:
                    continue
                a = 0.42 if (i == self.ed.water_sel and self.wtool.get() != "off") else 0.22
                img[w.mask] = (img[w.mask] * (1 - a) + J.np.array([0, 190, 255]) * a).astype(J.np.uint8)
        P = self.ed.par
        if not self.preview.get() and P is not None and P.planes and J.np is not None and (self.ed.target == "depth" and self.wtool.get() != "off"):
            img = img.copy()
            cols = ((255, 60, 220), (255, 140, 0), (255, 235, 0), (60, 230, 60))
            for i, p in enumerate(P.planes):
                if p.mask is None:
                    continue
                a = 0.45 if i == self.ed.plane_sel else 0.2
                img[p.mask] = (img[p.mask] * (1 - a) + J.np.array(cols[i % 4]) * a).astype(J.np.uint8)
        if not self.preview.get() and self.ed.hangs and J.np is not None and self.ed.target == "hang" and self.wtool.get() != "off":
            img = img.copy()
            for i, h in enumerate(self.ed.hangs):
                if h.mask is None:
                    continue
                a = 0.45 if i == self.ed.hang_sel else 0.2
                img[h.mask] = (img[h.mask] * (1 - a) + J.np.array([255, 165, 0]) * a).astype(J.np.uint8)
        if self._marks is not None:
            img = img.copy(); img[self._marks] = (255, 230, 0)
        return img

    def _render(self):
        c = self.canvas
        c.delete("ov")
        img = self.frame_rgb()
        if img is None:
            c.delete("all")
            c.create_text(J.IMG_W * self.zoom / 2, J.IMG_H * self.zoom / 2, fill=MUTED,
                          text="Open an .mg image from the SD card\n(the /images folder)", justify="center")
            return
        if Image is not None:
            pil = Image.fromarray(img).resize((J.IMG_W * self.zoom, J.IMG_H * self.zoom), Image.NEAREST)
            self._photo = ImageTk.PhotoImage(pil)
            if self._img_item is None:
                c.delete("all")
                self._img_item = c.create_image(0, 0, anchor="nw", image=self._photo)
            else:
                c.itemconfigure(self._img_item, image=self._photo)
        if self.show_outlines.get():
            self._draw_overlays()
        elif self.wtool.get() != "off":
            self._draw_water_overlays()

    def _draw_grid(self, w, sel):
        c, k = self.canvas, self.zoom
        H = J.water_homog(w.grid)
        col = "#7fd" if sel else "#476"
        for i in range(9):
            for a, b in (((i / 8, 0.0), (i / 8, 1.0)), ((0.0, i / 8), (1.0, i / 8))):
                pts = []
                for s in range(9):
                    u = a[0] + (b[0] - a[0]) * s / 8; v = a[1] + (b[1] - a[1]) * s / 8
                    x, y = J.water_hmap(H, u, v); pts += [x * k, y * k]
                c.create_line(*pts, fill=col, width=2 if i in (0, 8) else 1, tags="ov")
        G, Hh, I = J.water_unmap(H, 0.0, 0.0)[3]
        if abs(Hh) > 1e-9:                         # the horizon it implies
            y0, y1 = -(G * 0 + I) / Hh, -(G * J.IMG_W + I) / Hh
            c.create_line(0, y0 * k, J.IMG_W * k, y1 * k, fill="#fd7", dash=(6, 4), tags="ov")
        if sel and self.wtool.get() == "grid":
            for x, y in w.grid:
                c.create_rectangle(x * k - 5, y * k - 5, x * k + 5, y * k + 5, outline="#000", fill=LIME, tags="ov")

    def _draw_water_overlays(self):
        c, k = self.canvas, self.zoom
        P = self.ed.par
        if self.ed.target == "depth" and P is not None and P.grid:
            self._draw_grid(P, True)
        h = self.ed.hang
        if self.ed.target == "hang" and h is not None and h.a[0] >= 0 and not self.preview.get():   # the pin
            z = self.zoom
            (ax, ay), (bx, by) = h.a, h.b
            c.create_line(ax * z, ay * z, bx * z, by * z, fill="#000", width=4, tags="ov")
            c.create_line(ax * z, ay * z, bx * z, by * z, fill=YEL, width=2, tags="ov")
            for px_, py_ in (h.a, h.b):
                c.create_oval(px_ * z - 4, py_ * z - 4, px_ * z + 4, py_ * z + 4, fill=YEL, outline="#000", tags="ov")
            if h.mask is not None and h.mask.any():
                import mg_hang as G
                ys, xs = J.np.nonzero(h.mask)
                rx, ry = (ax + bx) / 2, (ay + by) / 2
                dx, dy = G.hang_dir(xs, ys, ax, ay, bx, by)
                l = math.hypot(dx, dy)
                if l >= 4:
                    c.create_line(rx * z, ry * z, (rx + dx / l * 18) * z, (ry + dy / l * 18) * z, fill=YEL, width=2, arrow="last", tags="ov")
        for i, w in enumerate(self.ed.waters):
            if w.grid and self.ed.target == "water":
                self._draw_grid(w, i == self.ed.water_sel)
        for i, w in enumerate(self.ed.waters):
            sel = i == self.ed.water_sel
            for l in w.lines:
                if len(l) < 2:
                    continue
                cx, cy = J.water_curve(l)
                pts = [v * k for xy in zip(cx, cy) for v in xy]
                c.create_line(*pts, fill="#000", width=4, tags="ov")
                c.create_line(*pts, fill=YEL if sel else "#bba", width=2, arrow="last", arrowshape=(10, 12, 5), tags="ov")
                for x, y in l:
                    c.create_oval(x * k - 3, y * k - 3, x * k + 3, y * k + 3, outline="#000", fill=YEL if sel else "#bba", tags="ov")
        if self._wpts:
            col = YEL if self.wtool.get() == "line" else LIME
            if len(self._wpts) >= 2:
                if self.wtool.get() == "line":
                    cx, cy = J.water_curve(self._wpts)
                    pts = [v * k for xy in zip(cx, cy) for v in xy]
                else:
                    pts = [v * k for p_ in self._wpts for v in p_]
                c.create_line(*pts, fill=col, width=2, dash=(4, 3), tags="ov")
            for x, y in self._wpts:
                c.create_oval(x * k - 3, y * k - 3, x * k + 3, y * k + 3, outline="#000", fill=col, tags="ov")

    def _draw_overlays(self):
        c, k = self.canvas, self.zoom
        self._draw_water_overlays()
        for i, z in enumerate(self.ed.zones):
            col = ZONE_COLS[i % len(ZONE_COLS)]
            sel = i == self.ed.selected
            if sel:
                ring = [v * k for p in z.outline(z.margin(), 48) for v in p]
                c.create_line(*ring, *ring[:2], fill="#7a8a9a", dash=(3, 4), tags="ov")
            pts = [v * k for p in z.outline(1.0, 64) for v in p]
            c.create_line(*pts, *pts[:2], fill="#000", width=4 if sel else 3, tags="ov")
            c.create_line(*pts, *pts[:2], fill=col, width=2 if sel else 1, tags="ov")
            if sel:
                x, y = z.cx * k, z.cy * k
                c.create_line(x - 6, y, x + 7, y, fill=col, tags="ov")
                c.create_line(x, y - 6, x, y + 7, fill=col, tags="ov")
                hx, hy = self.ed.rot_handle_xy(z)
                ex, ey = z.from_local(0.0, 1.0)
                c.create_line(ex * k, ey * k, hx * k, hy * k, fill=col, dash=(2, 2), tags="ov")
                c.create_oval(hx * k - 6, hy * k - 6, hx * k + 6, hy * k + 6, fill="#000", outline=col, width=2, tags="ov")
                c.create_text(hx * k, hy * k, text="⟳", fill=col, font=("Segoe UI", 8, "bold"), tags="ov")
            if z.tip:
                tx, ty = z.tip_xy()
                rr = max(3.0, z.tip_radius() * z.r) * k
                c.create_oval(tx * k - rr, ty * k - rr, tx * k + rr, ty * k + rr, outline="#000", width=3, tags="ov")
                c.create_oval(tx * k - rr, ty * k - rr, tx * k + rr, ty * k + rr, outline=YEL if sel else col, width=1, tags="ov")
            c.create_text(z.cx * k + 1, (z.cy - z.r) * k - 7, text=str(i + 1), fill=col, tags="ov",
                          font=("Segoe UI", 9, "bold"))
