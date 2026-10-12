#!/usr/bin/env python3
"""
Mantis Gallery — MG Converter (Tkinter GUI)

Converts JPG / PNG / BMP / WEBP → .mg (RGB565 320×240) for fast
playback on M5Stack Core2 from the SD card.

Preserves folder structure:
    Input:  FolderA/cats/a.jpg
            FolderA/dogs/b.png
    Output: out/cats/a.mg
            out/dogs/b.mg

Requires:
    pip install pillow

Run (opens the GUI):
    python mg_convert.py

Optional CLI (no window):
    python mg_convert.py --cli /path/to/folder -o /path/to/out
"""

from __future__ import annotations

import struct
import sys
import threading
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Install Pillow first:\n  pip install pillow")
    sys.exit(1)

W, H = 320, 240
MAGIC = b"MG01"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def to_rgb565_bytes(im: Image.Image) -> bytes:
    """RGB image → little-endian RGB565 byte string (row-major)."""
    im = im.convert("RGB")
    if im.size != (W, H):
        raise ValueError(f"expected {W}x{H}, got {im.size}")
    rgb = im.tobytes()
    out = bytearray(W * H * 2)
    j = 0
    for i in range(0, len(rgb), 3):
        r, g, b = rgb[i], rgb[i + 1], rgb[i + 2]
        pix = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
        out[j] = pix & 0xFF
        out[j + 1] = (pix >> 8) & 0xFF
        j += 2
    return bytes(out)


def prepare_image(src: Path, mode: str) -> Image.Image:
    im = Image.open(src).convert("RGB")
    sw, sh = im.size
    if mode == "fit":
        scale = min(W / sw, H / sh)
        nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        canvas.paste(im, ((W - nw) // 2, (H - nh) // 2))
        return canvas
    # cover — fill and center-crop
    scale = max(W / sw, H / sh)
    nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - W) // 2
    top = (nh - H) // 2
    return im.crop((left, top, left + W, top + H))


def convert_one(src: Path, dst: Path, mode: str = "cover") -> None:
    im = prepare_image(src, mode)
    body = to_rgb565_bytes(im)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<HH", W, H))
        f.write(body)


def collect_files(src_dir: Path) -> list[Path]:
    """All images under src_dir, including nested subfolders."""
    return sorted(
        p for p in src_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS
    )


# ---------------------------------------------------------------------------
#  Tkinter GUI
# ---------------------------------------------------------------------------
def run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk, scrolledtext

    root = tk.Tk()
    root.title("Mantis Gallery — MG Converter")
    root.geometry("640x480")
    root.minsize(520, 400)
    root.configure(bg="#1a1a1a")

    # theme-ish ttk
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("TRadiobutton", background="#1a1a1a", foreground="#e8e8e8")
    style.configure("TProgressbar", troughcolor="#2a2a2a", background="#007373")

    src_var = tk.StringVar()
    out_var = tk.StringVar()
    mode_var = tk.StringVar(value="cover")
    status_var = tk.StringVar(value="Choose source + output folders, then Convert.")
    busy = {"on": False}

    def log(msg: str) -> None:
        log_box.configure(state="normal")
        log_box.insert("end", msg + "\n")
        log_box.see("end")
        log_box.configure(state="disabled")

    def browse_src() -> None:
        d = filedialog.askdirectory(title="Source folder (JPG/PNG/…, subfolders kept)")
        if d:
            src_var.set(d)
            n = len(collect_files(Path(d)))
            status_var.set(f"Source set — {n} image(s) found (including subfolders)")

    def browse_out() -> None:
        d = filedialog.askdirectory(title="Output folder (same folder tree as .mg)")
        if d:
            out_var.set(d)

    def do_convert() -> None:
        if busy["on"]:
            return
        src = src_var.get().strip()
        out_s = out_var.get().strip()
        if not src:
            messagebox.showerror("Mantis Gallery", "Choose a source folder first.")
            return
        if not out_s:
            messagebox.showerror(
                "Mantis Gallery",
                "Choose an output folder.\n\n"
                "Folder structure is preserved, e.g.:\n"
                "  source/cats/a.jpg  →  output/cats/a.mg",
            )
            return
        src_path = Path(src)
        out_path = Path(out_s)
        if not src_path.is_dir():
            messagebox.showerror("Mantis Gallery", "Source path is not a folder.")
            return
        if src_path.resolve() == out_path.resolve():
            messagebox.showerror("Mantis Gallery", "Source and output must be different folders.")
            return
        mode = mode_var.get()
        files = collect_files(src_path)
        if not files:
            messagebox.showwarning("Mantis Gallery", "No JPG/PNG/BMP/WEBP files found.")
            return

        busy["on"] = True
        convert_btn.configure(state="disabled")
        progress["value"] = 0
        progress["maximum"] = len(files)
        status_var.set(f"Converting 0 / {len(files)} …")
        log_box.configure(state="normal")
        log_box.delete("1.0", "end")
        log_box.configure(state="disabled")
        log(f"Source: {src_path}")
        log(f"Output: {out_path}")
        log(f"Mode:   {mode}  →  320×240 RGB565 .mg")
        log(f"Files:  {len(files)}  (folder structure preserved)")
        log("——")

        def worker() -> None:
            ok = fail = 0
            for i, p in enumerate(files):
                rel = p.relative_to(src_path)
                # Preserve tree: source/cats/a.jpg → output/cats/a.mg
                dst = out_path / rel.with_suffix(".mg")
                try:
                    convert_one(p, dst, mode=mode)
                    ok += 1
                    root.after(0, lambda m=f"OK  {rel} → {rel.with_suffix('.mg')}": log(m))
                except Exception as e:
                    fail += 1
                    root.after(0, lambda m=f"FAIL {rel}: {e}": log(m))
                root.after(0, lambda v=i + 1, t=len(files): (
                    progress.configure(value=v),
                    status_var.set(f"Converting {v} / {t} …"),
                ))
            def done() -> None:
                busy["on"] = False
                convert_btn.configure(state="normal")
                status_var.set(f"Done — {ok} converted, {fail} failed")
                log(f"—— finished: {ok} ok, {fail} failed ——")
                messagebox.showinfo(
                    "Mantis Gallery",
                    f"Converted {ok} image(s)\nFailed: {fail}\n\n"
                    f"Folder structure was kept under:\n{out_path}\n\n"
                    f"Copy that tree onto the Core2 SD card under /images",
                )
            root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    # ---- layout ----
    pad = {"padx": 14, "pady": 6}
    fg = "#e8e8e8"
    muted = "#9e9e9e"
    teal = "#007373"
    lime = "#b4ff00"

    tk.Label(
        root, text="Mantis Gallery", fg=teal, bg="#1a1a1a",
        font=("Segoe UI", 16, "bold"),
    ).pack(anchor="w", padx=14, pady=(14, 0))
    tk.Label(
        root,
        text="Convert photos to .mg  ·  320×240 RGB565  ·  keeps subfolder structure",
        fg=muted, bg="#1a1a1a", font=("Segoe UI", 10),
    ).pack(anchor="w", padx=14, pady=(0, 8))

    # source
    fr = tk.Frame(root, bg="#1a1a1a")
    fr.pack(fill="x", **pad)
    tk.Label(fr, text="Source folder", fg=muted, bg="#1a1a1a", width=14, anchor="w").pack(side="left")
    tk.Entry(fr, textvariable=src_var, bg="#2a2a2a", fg=fg, insertbackground=fg,
             relief="flat").pack(side="left", fill="x", expand=True, padx=(0, 8))
    tk.Button(fr, text="Browse…", command=browse_src, bg="#2d2d2d", fg=fg,
              relief="flat", padx=10).pack(side="left")

    # output
    fr2 = tk.Frame(root, bg="#1a1a1a")
    fr2.pack(fill="x", **pad)
    tk.Label(fr2, text="Output folder", fg=muted, bg="#1a1a1a", width=14, anchor="w").pack(side="left")
    tk.Entry(fr2, textvariable=out_var, bg="#2a2a2a", fg=fg, insertbackground=fg,
             relief="flat").pack(side="left", fill="x", expand=True, padx=(0, 8))
    tk.Button(fr2, text="Browse…", command=browse_out, bg="#2d2d2d", fg=fg,
              relief="flat", padx=10).pack(side="left")

    # mode
    fr3 = tk.Frame(root, bg="#1a1a1a")
    fr3.pack(fill="x", **pad)
    tk.Label(fr3, text="Scale mode", fg=muted, bg="#1a1a1a", width=14, anchor="w").pack(side="left")
    ttk.Radiobutton(fr3, text="Cover (crop to fill)", variable=mode_var, value="cover").pack(side="left", padx=4)
    ttk.Radiobutton(fr3, text="Fit (letterbox)", variable=mode_var, value="fit").pack(side="left", padx=4)

    convert_btn = tk.Button(
        root, text="Convert to .mg", command=do_convert,
        bg=teal, fg="#000000", activebackground="#00a0a0",
        font=("Segoe UI", 12, "bold"), relief="flat", padx=16, pady=8,
    )
    convert_btn.pack(pady=10)

    progress = ttk.Progressbar(root, mode="determinate", length=400)
    progress.pack(fill="x", padx=14, pady=(0, 4))

    tk.Label(root, textvariable=status_var, fg=lime, bg="#1a1a1a",
             font=("Segoe UI", 10)).pack(anchor="w", padx=14)

    log_box = scrolledtext.ScrolledText(
        root, height=12, bg="#121212", fg="#c8c8c8",
        insertbackground=fg, relief="flat", font=("Consolas", 9),
    )
    log_box.pack(fill="both", expand=True, padx=14, pady=(6, 14))
    log_box.configure(state="disabled")

    root.mainloop()


def run_cli(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Mantis Gallery .mg converter (CLI)")
    ap.add_argument("--cli", action="store_true")
    ap.add_argument("src", help="Source folder")
    ap.add_argument("-o", "--out", required=True, help="Output folder (structure preserved)")
    ap.add_argument("--mode", choices=("cover", "fit"), default="cover")
    args = ap.parse_args(argv)
    src_path = Path(args.src)
    out_path = Path(args.out)
    if src_path.resolve() == out_path.resolve():
        print("Source and output must differ")
        return 2
    files = collect_files(src_path)
    ok = fail = 0
    for p in files:
        rel = p.relative_to(src_path)
        dst = out_path / rel.with_suffix(".mg")
        try:
            convert_one(p, dst, mode=args.mode)
            print(f"OK  {rel} → {dst}")
            ok += 1
        except Exception as e:
            print(f"FAIL {rel}: {e}")
            fail += 1
    print(f"\n{ok} ok, {fail} failed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    # Default: always open the Tkinter GUI.
    # Pass --cli for headless batch use.
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        sys.exit(run_cli(sys.argv[1:]))
    run_gui()
