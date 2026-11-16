"""
Mantis Gallery — "Jiggle" tab for mg_convert.py.

Open an .mg straight off the SD card, place zones, fine-tune how they move, watch a live preview, and
save the hidden .jig file next to the image (named the way the Core2 expects). All the physics lives
in jiggle_lab.py — the same maths as the firmware, so what you see is what the Core2 does.

Mouse   click a shape button, then click the photo to place it · drag a zone to move it · drag its
        point circle to line it up · wheel = size · Shift+wheel = rotate · in preview, click = poke
Keys    1/2/3 place Round/Half/Tear · Del delete · arrows nudge (Shift ×10) · +/- size · [ ] rotate
        Space preview · Ctrl+S save · Ctrl+Z / Ctrl+Y undo / redo · Esc cancel placing
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
        self.v_rot = self._slider(f, "Rotation", 0, 359, 1, lambda v: self._set_attr("rot", int(v)))

        f = self._section(p, "Control point  (line it up over the apex)")
        self.v_tip = tk.BooleanVar(value=False)
        tk.Checkbutton(f, text="Use a control point", variable=self.v_tip, bg=PANEL, fg=FG, selectcolor="#333",
                       activebackground=PANEL, command=lambda: self._edit(lambda: self.ed.set_tip(self.v_tip.get()))).pack(anchor="w")
        self.v_tu = self._slider(f, "Across", -0.6, 0.6, 0.01, lambda v: self._set_tip(v, None))
        self.v_tw = self._slider(f, "Up/down", -1.0, 0.9, 0.01, lambda v: self._set_tip(None, v))

        f = self._section(p, "Fine tuning  (blank = material preset)")
        self.tune_rows = {}
        for key, label, lo, hi, help_ in J.TUNES:
            self.tune_rows[key] = self._tune_row(f, key, label, lo, hi, help_)
        self._btn(f, "Reset all to preset", self.reset_tuning).pack(anchor="w", pady=(4, 0))

        f = self._section(p, "Response  (after a quick sideways shove)")
        self.resp = tk.Canvas(f, width=340, height=110, bg="#111", highlightthickness=0)
        self.resp.pack()
        self.resp_lbl = tk.Label(f, text="", fg=MUTED, bg=PANEL, anchor="w", justify="left")
        self.resp_lbl.pack(fill="x")

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
        res = 1.0 if key == "side" else 0.01
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
        for kind, label in (("shake", "Shake"), ("bounce", "Hop"), ("walk", "Walk"), ("jolt", "Push")):
            self._btn(bar, label, lambda k=kind: self.start_motion(k)).pack(side="left", padx=2)
        tk.Label(bar, text="  Tilt", fg=MUTED, bg=BG).pack(side="left")
        tk.Scale(bar, from_=-45, to=45, orient="horizontal", variable=self.tilt, showvalue=False, length=90,
                 bg=BG, troughcolor="#333", highlightthickness=0).pack(side="left")

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
        on("<BackSpace>", lambda e: self.delete())
        on("<Escape>", lambda e: self.arm(-1))
        on("1", lambda e: self.arm(0)); on("2", lambda e: self.arm(1)); on("3", lambda e: self.arm(2))
        on("<plus>", lambda e: self._edit(lambda: self.ed.resize(1.06)))
        on("<equal>", lambda e: self._edit(lambda: self.ed.resize(1.06)))
        on("<minus>", lambda e: self._edit(lambda: self.ed.resize(1 / 1.06)))
        on("<bracketleft>", lambda e: self._edit(lambda: self.ed.rotate(-5)))
        on("<bracketright>", lambda e: self._edit(lambda: self.ed.rotate(5)))
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
            self.ed.zones, self.ed.selected = self.undo_stack.pop()
            self.ed.dirty = True
            self._after_edit()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append((self.ed.snapshot(), self.ed.selected))
            self.ed.zones, self.ed.selected = self.redo_stack.pop()
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
        if J.shape_sdf(z.shape, nu, nw) < -z.tip_radius() * 0.5:
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
        self.ed.drag(x, y)
        self._sync_panel()
        self._render()

    def _on_release(self, e):
        if self.ed.image is None or self.preview.get():
            return
        self.ed.release(*self._img_xy(e))
        self._after_edit()

    def _on_wheel(self, e, direction=None):
        if not self.ed.zone or self.preview.get():
            return
        d = direction if direction is not None else (1 if getattr(e, "delta", 0) > 0 else -1)
        if getattr(e, "state", 0) & 0x0001:            # Shift: rotate
            self._edit(lambda: self.ed.rotate(5 * d), coalesce=True)
        else:
            self._edit(lambda: self.ed.resize(1.04 if d > 0 else 1 / 1.04), coalesce=True)

    def _on_hover(self, e):
        if self.ed.image is None or self.preview.get():
            return
        x, y = self._img_xy(e)
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

    def _sync_panel(self):
        self._syncing = True
        try:
            self._refresh_list()
            z = self.ed.zone
            if z:
                self.v_shape.set(J.SHAPES[z.shape][1])
                self.v_mat.set(z.material)
                self.v_x.set(round(z.cx)); self.v_y.set(round(z.cy))
                self.v_r.set(round(z.r)); self.v_rot.set(z.rot)
                self.v_tip.set(z.tip)
                self.v_tu.set(round(z.tip_u, 2)); self.v_tw.set(round(z.tip_w, 2))
                sim = J.ZoneSim(z)
                defaults = self._tune_defaults(z, sim)
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
                "travel": 1.0, "bulge": 1.0, "stretch": 1.0, "sag": 1.0, "pend": J.PENDULUM, "pstiff": 1.0,
                "pdamp": 0.38, "psize": 1.0, "dance": 1.0, "side": 0.0}

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
        self.sim = J.Sim(self.ed.zones, self.ed.image)
        if not keep_motion:
            self._motion = None

    def _preview_toggled(self):
        self._restart_sim()
        self.status.set("Preview: shake / hop / walk / push, tilt, dance, or click a zone to poke it."
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
        acc = J.motion_accel("none", 0.0, self.tilt.get())
        if self._motion:
            kind, ts = self._motion
            if self._sim_t - ts > 5.0:
                self._motion = None
            else:
                acc = J.motion_accel(kind, self._sim_t - ts, self.tilt.get())
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
        self.sim.step(dt, acc, self._dance_amp)

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

    def _draw_overlays(self):
        c, k = self.canvas, self.zoom
        for i, z in enumerate(self.ed.zones):
            col = ZONE_COLS[i % len(ZONE_COLS)]
            sel = i == self.ed.selected
            if sel:
                ring = [v * k for p in z.outline(J.MARGIN, 48) for v in p]
                c.create_line(*ring, *ring[:2], fill="#7a8a9a", dash=(3, 4), tags="ov")
            pts = [v * k for p in z.outline(1.0, 64) for v in p]
            c.create_line(*pts, *pts[:2], fill="#000", width=4 if sel else 3, tags="ov")
            c.create_line(*pts, *pts[:2], fill=col, width=2 if sel else 1, tags="ov")
            if sel:
                x, y = z.cx * k, z.cy * k
                c.create_line(x - 6, y, x + 7, y, fill=col, tags="ov")
                c.create_line(x, y - 6, x, y + 7, fill=col, tags="ov")
            if z.tip:
                tx, ty = z.tip_xy()
                rr = max(3.0, z.tip_radius() * z.r) * k
                c.create_oval(tx * k - rr, ty * k - rr, tx * k + rr, ty * k + rr, outline="#000", width=3, tags="ov")
                c.create_oval(tx * k - rr, ty * k - rr, tx * k + rr, ty * k + rr, outline=YEL if sel else col, width=1, tags="ov")
            c.create_text(z.cx * k + 1, (z.cy - z.r) * k - 7, text=str(i + 1), fill=col, tags="ov",
                          font=("Segoe UI", 9, "bold"))
