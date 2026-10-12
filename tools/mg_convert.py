#!/usr/bin/env python3
"""
Mantis Gallery image converter — laptop tool

Converts JPG/PNG/BMP/WEBP → .mg (RGB565 320×240) for fast SD playback
on M5Stack Core2.

Requires:  pip install pillow

Usage:
  python mg_convert.py
  python mg_convert.py /path/to/folder
  python mg_convert.py /path/to/folder -o /path/to/output
  python mg_convert.py --cli /path/to/folder -o /path/to/output   # no GUI
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Install Pillow first:  pip install pillow")
    sys.exit(1)

W, H = 320, 240
MAGIC = b"MG01"


def to_rgb565(r: int, g: int, b: int) -> int:
    return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)


def convert_one(src: Path, dst: Path, mode: str = "cover") -> None:
    """
    mode:
      cover  — scale to fill 320x240, center-crop overflow (default)
      fit    — letterbox (pad with black) keeping full image
    """
    im = Image.open(src)
    im = im.convert("RGB")
    sw, sh = im.size

    if mode == "fit":
        scale = min(W / sw, H / sh)
        nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        canvas.paste(im, ((W - nw) // 2, (H - nh) // 2))
        im = canvas
    else:
        scale = max(W / sw, H / sh)
        nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
        left = (nw - W) // 2
        top = (nh - H) // 2
        im = im.crop((left, top, left + W, top + H))

    pixels = im.load()
    body = bytearray()
    for y in range(H):
        for x in range(W):
            r, g, b = pixels[x, y]
            body += struct.pack("<H", to_rgb565(r, g, b))

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<HH", W, H))
        f.write(body)


def batch(src_dir: Path, out_dir: Path | None, mode: str) -> tuple[int, int]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in sorted(src_dir.rglob("*")) if p.suffix.lower() in exts and p.is_file()]
    ok = fail = 0
    for p in files:
        rel = p.relative_to(src_dir)
        target_root = out_dir if out_dir else src_dir
        dst = target_root / rel.with_suffix(".mg")
        try:
            convert_one(p, dst, mode=mode)
            print(f"OK  {p} → {dst}")
            ok += 1
        except Exception as e:
            print(f"FAIL {p}: {e}")
            fail += 1
    return ok, fail


def run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Mantis Gallery — MG Converter")
    root.geometry("520x280")
    root.configure(bg="#1a1a1a")

    src_var = tk.StringVar()
    out_var = tk.StringVar()
    mode_var = tk.StringVar(value="cover")
    status = tk.StringVar(value="Pick a folder of JPG/PNG images.")

    def browse_src():
        d = filedialog.askdirectory(title="Source folder")
        if d:
            src_var.set(d)

    def browse_out():
        d = filedialog.askdirectory(title="Output folder (optional)")
        if d:
            out_var.set(d)

    def go():
        s = src_var.get().strip()
        if not s:
            messagebox.showerror("Mantis", "Choose a source folder")
            return
        o = out_var.get().strip() or None
        status.set("Converting…")
        root.update_idletasks()
        ok, fail = batch(Path(s), Path(o) if o else None, mode_var.get())
        status.set(f"Done — {ok} converted, {fail} failed")
        messagebox.showinfo("Mantis", f"Converted {ok} image(s)\nFailed: {fail}")

    pad = {"padx": 12, "pady": 6}
    fg = "#e8e8e8"
    tk.Label(root, text="Mantis Gallery converter", fg="#007373", bg="#1a1a1a",
             font=("Segoe UI", 14, "bold")).pack(anchor="w", **pad)
    tk.Label(root, text="Output is 320×240 RGB565 .mg for Core2 SD",
             fg="#9e9e9e", bg="#1a1a1a").pack(anchor="w", padx=12)

    row = tk.Frame(root, bg="#1a1a1a")
    row.pack(fill="x", **pad)
    tk.Entry(row, textvariable=src_var, width=48).pack(side="left", padx=(0, 8))
    tk.Button(row, text="Source…", command=browse_src).pack(side="left")

    row2 = tk.Frame(root, bg="#1a1a1a")
    row2.pack(fill="x", **pad)
    tk.Entry(row2, textvariable=out_var, width=48).pack(side="left", padx=(0, 8))
    tk.Button(row2, text="Output…", command=browse_out).pack(side="left")

    row3 = tk.Frame(root, bg="#1a1a1a")
    row3.pack(fill="x", **pad)
    tk.Label(row3, text="Scale:", fg=fg, bg="#1a1a1a").pack(side="left")
    ttk.Radiobutton(row3, text="Cover (crop)", variable=mode_var, value="cover").pack(side="left", padx=8)
    ttk.Radiobutton(row3, text="Fit (letterbox)", variable=mode_var, value="fit").pack(side="left")

    tk.Button(root, text="Convert", command=go, bg="#007373", fg="white",
              font=("Segoe UI", 11, "bold")).pack(**pad)
    tk.Label(root, textvariable=status, fg="#b4ff00", bg="#1a1a1a").pack(anchor="w", padx=12)

    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Mantis Gallery .mg converter")
    ap.add_argument("src", nargs="?", help="Source folder")
    ap.add_argument("-o", "--out", help="Output folder (default: beside sources)")
    ap.add_argument("--mode", choices=("cover", "fit"), default="cover")
    ap.add_argument("--cli", action="store_true", help="Force CLI (no GUI)")
    args = ap.parse_args()

    if args.src or args.cli:
        if not args.src:
            print("Source folder required with --cli")
            sys.exit(2)
        ok, fail = batch(Path(args.src), Path(args.out) if args.out else None, args.mode)
        print(f"\n{ok} ok, {fail} failed")
        sys.exit(0 if fail == 0 else 1)

    run_gui()


if __name__ == "__main__":
    main()
