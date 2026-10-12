#!/usr/bin/env python3
"""
Mantis Gallery — Basalt Softworks MG image converter

Converts JPG/PNG/BMP/WEBP to 320x240 RGB565 .mg files for M5Stack Core2.

Requires: pip install pillow

Usage:
  python mg_convert.py
  python mg_convert.py /path/to/folder
  python mg_convert.py /path/to/folder -o /path/to/output
  python mg_convert.py --cli /path/to/folder -o /path/to/output
"""

from __future__ import annotations

import argparse
import struct
import sys
import threading
import queue
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Install Pillow first: pip install pillow")
    sys.exit(1)

W, H = 320, 240
MAGIC = b"MG01"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Basalt Softworks / Mantis palette
BG = "#111315"
PANEL = "#1a1d20"
PANEL_ALT = "#202428"
BORDER = "#30383b"
TEAL = "#007373"
TEAL_HI = "#00a0a0"
PLUM = "#5d005d"
LIME = "#b4ff00"
TEXT = "#e8eeee"
MUTED = "#8e9a9d"
RED = "#ff6b72"
FONT = "Segoe UI"


def to_rgb565(r: int, g: int, b: int) -> int:
    return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)


def convert_one(src: Path, dst: Path, mode: str = "cover") -> None:
    """Convert one image to the MG01 header + little-endian RGB565 pixel payload."""
    with Image.open(src) as opened:
        im = opened.convert("RGB")
    sw, sh = im.size
    if sw < 1 or sh < 1:
        raise ValueError("Image has invalid dimensions")

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
        left, top = (nw - W) // 2, (nh - H) // 2
        im = im.crop((left, top, left + W, top + H))

    # Pack efficiently while preserving the existing file format.
    pixels = im.tobytes()
    body = bytearray(W * H * 2)
    j = 0
    for i in range(0, len(pixels), 3):
        r, g, b = pixels[i], pixels[i + 1], pixels[i + 2]
        value = to_rgb565(r, g, b)
        body[j] = value & 0xFF
        body[j + 1] = (value >> 8) & 0xFF
        j += 2

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<HH", W, H))
        f.write(body)


def find_images(src_dir: Path) -> list[Path]:
    return sorted(p for p in src_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in EXTENSIONS)


def batch(src_dir: Path, out_dir: Path | None, mode: str) -> tuple[int, int]:
    files = find_images(src_dir)
    ok = fail = 0
    for p in files:
        rel = p.relative_to(src_dir)
        dst = (out_dir if out_dir else src_dir) / rel.with_suffix(".mg")
        try:
            convert_one(p, dst, mode=mode)
            print(f"OK  {p} -> {dst}")
            ok += 1
        except Exception as exc:
            print(f"FAIL {p}: {exc}")
            fail += 1
    return ok, fail


def run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("MANTIS GALLERY  /  MG CONVERTER")
    root.geometry("780x690")
    root.minsize(700, 620)
    root.configure(bg=BG)

    src_var = tk.StringVar()
    out_var = tk.StringVar()
    mode_var = tk.StringVar(value="cover")
    status_var = tk.StringVar(value="READY  •  Choose a source folder to scan.")
    source_count_var = tk.StringVar(value="0 images detected")
    mode_hint_var = tk.StringVar(value="Fills the 320 × 240 frame; trims edges as needed.")
    event_queue: queue.Queue = queue.Queue()
    converting = False

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure(
        "Mantis.Horizontal.TProgressbar",
        troughcolor=PANEL_ALT, background=LIME, bordercolor=PANEL_ALT,
        lightcolor=LIME, darkcolor=LIME, thickness=9
    )
    style.configure(
        "Mantis.TRadiobutton", background=PANEL, foreground=TEXT,
        indicatorcolor=PANEL_ALT, focuscolor=PANEL, font=(FONT, 10)
    )
    style.map(
        "Mantis.TRadiobutton",
        background=[("active", PANEL)],
        foreground=[("disabled", MUTED), ("active", LIME)]
    )

    def label(parent, text, *, color=TEXT, size=10, weight="normal", **kwargs):
        return tk.Label(parent, text=text, bg=parent.cget("bg"), fg=color,
                        font=(FONT, size, weight), **kwargs)

    def section(parent, title, subtitle=None):
        outer = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        heading = tk.Frame(outer, bg=PANEL)
        heading.pack(fill="x", padx=16, pady=(13, 3))
        label(heading, title.upper(), color=LIME, size=9, weight="bold").pack(anchor="w")
        if subtitle:
            label(outer, subtitle, color=MUTED, size=9, justify="left",
                  wraplength=650).pack(anchor="w", padx=16, pady=(0, 10))
        return outer

    def styled_button(parent, text, command, *, primary=False, width=None):
        bg = TEAL if primary else PANEL_ALT
        fg = "#ffffff" if primary else TEXT
        btn = tk.Button(
            parent, text=text, command=command, width=width,
            bg=bg, fg=fg, activebackground=TEAL_HI if primary else BORDER,
            activeforeground="#ffffff", disabledforeground=MUTED,
            relief="flat", bd=0, padx=14, pady=9,
            font=(FONT, 9, "bold"), cursor="hand2",
            highlightthickness=1, highlightbackground=TEAL if primary else BORDER
        )
        return btn

    def choose_src():
        d = filedialog.askdirectory(title="Choose source image folder")
        if d:
            src_var.set(d)
            refresh_scan()

    def choose_out():
        d = filedialog.askdirectory(title="Choose output folder")
        if d:
            out_var.set(d)

    def refresh_scan():
        s = src_var.get().strip()
        if not s:
            source_count_var.set("0 images detected")
            return
        folder = Path(s).expanduser()
        if not folder.is_dir():
            source_count_var.set("Folder not found")
            return
        try:
            count = len(find_images(folder))
            source_count_var.set(f"{count} image{'s' if count != 1 else ''} detected")
            status_var.set(f"SCAN COMPLETE  •  {folder.name or str(folder)}")
        except Exception as exc:
            source_count_var.set("Could not scan folder")
            log(f"[WARN] Scan failed: {exc}")

    def log(message):
        log_box.configure(state="normal")
        log_box.insert("end", message + "\n")
        log_box.see("end")
        log_box.configure(state="disabled")

    def set_busy(busy):
        nonlocal converting
        converting = busy
        convert_btn.configure(state="disabled" if busy else "normal",
                              text="CONVERTING…" if busy else "▶   CONVERT IMAGES")
        scan_btn.configure(state="disabled" if busy else "normal")
        src_entry.configure(state="disabled" if busy else "normal")
        out_entry.configure(state="disabled" if busy else "normal")
        src_browse.configure(state="disabled" if busy else "normal")
        out_browse.configure(state="disabled" if busy else "normal")

    def start_conversion():
        s = Path(src_var.get().strip()).expanduser() if src_var.get().strip() else None
        if s is None or not s.is_dir():
            messagebox.showerror("Mantis Gallery", "Choose a valid source folder first.", parent=root)
            return
        files = find_images(s)
        if not files:
            messagebox.showinfo("Mantis Gallery", "No supported images were found in that folder.", parent=root)
            return
        o_text = out_var.get().strip()
        o = Path(o_text).expanduser() if o_text else None
        mode = mode_var.get()
        if o is not None:
            try:
                o.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                messagebox.showerror("Mantis Gallery", f"Cannot use output folder:\n{exc}", parent=root)
                return

        set_busy(True)
        progress.configure(maximum=len(files), value=0)
        progress_text.configure(text=f"0 / {len(files)}")
        status_var.set(f"CONVERTING  •  {len(files)} image(s) queued")
        log("")
        log(f"--- CONVERSION START  |  {len(files)} file(s)  |  {mode.upper()} ---")
        log(f"Destination: {o if o else 'Alongside source images'}")

        def worker():
            good = bad = 0
            for index, p in enumerate(files, start=1):
                rel = p.relative_to(s)
                dst = (o if o else s) / rel.with_suffix(".mg")
                try:
                    convert_one(p, dst, mode=mode)
                    good += 1
                    event_queue.put(("log", f"[ OK ] {rel}  →  {dst.name}"))
                except Exception as exc:
                    bad += 1
                    event_queue.put(("log", f"[FAIL] {rel}  |  {exc}"))
                event_queue.put(("progress", index, len(files)))
            event_queue.put(("done", good, bad, len(files)))

        threading.Thread(target=worker, daemon=True).start()
        poll_queue()

    def poll_queue():
        try:
            while True:
                item = event_queue.get_nowait()
                if item[0] == "log":
                    log(item[1])
                elif item[0] == "progress":
                    progress.configure(maximum=item[2], value=item[1])
                    progress_text.configure(text=f"{item[1]} / {item[2]}")
                elif item[0] == "done":
                    good, bad, total = item[1:]
                    set_busy(False)
                    status_var.set(f"COMPLETE  •  {good} converted  •  {bad} failed")
                    log(f"--- COMPLETE  |  {good} OK  |  {bad} FAILED  |  {total} TOTAL ---")
                    if bad:
                        messagebox.showwarning(
                            "Mantis Gallery — Finished",
                            f"Converted: {good}\nFailed: {bad}\n\nSee the run log for details.",
                            parent=root
                        )
                    else:
                        messagebox.showinfo(
                            "Mantis Gallery — Finished",
                            f"All {good} image(s) converted successfully.",
                            parent=root
                        )
        except queue.Empty:
            pass
        if converting:
            root.after(80, poll_queue)

    # Header / brand strip
    top = tk.Frame(root, bg=BG)
    top.pack(fill="x", padx=22, pady=(20, 14))
    brand = tk.Frame(top, bg=BG)
    brand.pack(side="left", fill="x", expand=True)
    label(brand, "MANTIS", color=TEXT, size=22, weight="bold").pack(anchor="w")
    label(brand, "B A S A L T   S O F T W O R K S", color=TEAL_HI, size=8, weight="bold").pack(anchor="w", pady=(0, 4))
    label(brand, "GALLERY  /  IMAGE CONVERTER", color=MUTED, size=10, weight="bold").pack(anchor="w")
    mark = tk.Frame(top, bg=PLUM, width=6, height=62)
    mark.pack(side="right", padx=(12, 0), fill="y")
    label(top, "MG01", color=LIME, size=15, weight="bold").pack(side="right", padx=8, pady=12)

    # Output format summary strip
    strip = tk.Frame(root, bg=TEAL, height=54)
    strip.pack(fill="x", padx=22, pady=(0, 14))
    strip.pack_propagate(False)
    label(strip, "TARGET FORMAT", color="#bce9e9", size=8, weight="bold").pack(side="left", padx=(15, 8), pady=14)
    label(strip, "320 × 240   /   RGB565   /   .MG", color="#ffffff", size=11, weight="bold").pack(side="left", pady=13)
    label(strip, "M5STACK CORE2  •  SD PLAYBACK", color="#d5ffff", size=8, weight="bold").pack(side="right", padx=15, pady=15)

    # Folder controls
    folders = section(root, "01  /  Image folders", "Source folders are scanned recursively. Subfolder structure is preserved in the output.")
    folders.pack(fill="x", padx=22, pady=(0, 12))
    source_row = tk.Frame(folders, bg=PANEL)
    source_row.pack(fill="x", padx=16, pady=(8, 6))
    label(source_row, "SOURCE", color=MUTED, size=8, weight="bold", width=9, anchor="w").pack(side="left")
    src_entry = tk.Entry(source_row, textvariable=src_var, bg=BG, fg=TEXT, insertbackground=LIME,
                         relief="flat", bd=0, font=(FONT, 10), highlightthickness=1,
                         highlightbackground=BORDER, highlightcolor=TEAL)
    src_entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 8))
    src_browse = styled_button(source_row, "BROWSE…", choose_src)
    src_browse.pack(side="left")

    output_row = tk.Frame(folders, bg=PANEL)
    output_row.pack(fill="x", padx=16, pady=(4, 4))
    label(output_row, "OUTPUT", color=MUTED, size=8, weight="bold", width=9, anchor="w").pack(side="left")
    out_entry = tk.Entry(output_row, textvariable=out_var, bg=BG, fg=TEXT, insertbackground=LIME,
                         relief="flat", bd=0, font=(FONT, 10), highlightthickness=1,
                         highlightbackground=BORDER, highlightcolor=TEAL)
    out_entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 8))
    out_browse = styled_button(output_row, "BROWSE…", choose_out)
    out_browse.pack(side="left")

    bottom_row = tk.Frame(folders, bg=PANEL)
    bottom_row.pack(fill="x", padx=16, pady=(4, 13))
    label(bottom_row, "Blank output = save .mg beside each source image.", color=MUTED, size=9).pack(side="left")
    scan_btn = styled_button(bottom_row, "SCAN FOLDER", refresh_scan)
    scan_btn.pack(side="right")
    label(bottom_row, textvariable=source_count_var, color=LIME, size=9, weight="bold").pack(side="right", padx=(0, 12))

    # Scaling mode
    options = section(root, "02  /  Image scaling", "Choose how images are composed into the Core2's 4:3 display.")
    options.pack(fill="x", padx=22, pady=(0, 12))
    mode_row = tk.Frame(options, bg=PANEL)
    mode_row.pack(fill="x", padx=16, pady=(8, 13))
    ttk.Radiobutton(mode_row, text="COVER  ·  crop edges to fill", variable=mode_var,
                    value="cover", style="Mantis.TRadiobutton",
                    command=lambda: mode_hint_var.set("Fills the entire display; trims overflow from the center.")).pack(side="left", padx=(0, 22))
    ttk.Radiobutton(mode_row, text="FIT  ·  preserve full image", variable=mode_var,
                    value="fit", style="Mantis.TRadiobutton",
                    command=lambda: mode_hint_var.set("Shows the complete image; adds black bars where needed.")).pack(side="left")
    label(options, textvariable=mode_hint_var, color=MUTED, size=9).pack(anchor="w", padx=17, pady=(0, 12))

    # Conversion action and progress
    action = tk.Frame(root, bg=BG)
    action.pack(fill="x", padx=22, pady=(0, 12))
    convert_btn = styled_button(action, "▶   CONVERT IMAGES", start_conversion, primary=True)
    convert_btn.configure(font=(FONT, 10, "bold"), padx=20, pady=12)
    convert_btn.pack(side="left")
    progress_text = label(action, "Waiting", color=LIME, size=9, weight="bold")
    progress_text.pack(side="right", padx=(10, 0))
    progress = ttk.Progressbar(action, style="Mantis.Horizontal.TProgressbar",
                               mode="determinate", maximum=100)
    progress.pack(side="right", fill="x", expand=True, padx=(18, 0), pady=8)

    # Log panel
    log_panel = tk.Frame(root, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
    log_panel.pack(fill="both", expand=True, padx=22, pady=(0, 10))
    log_header = tk.Frame(log_panel, bg=PANEL_ALT)
    log_header.pack(fill="x")
    label(log_header, "RUN LOG", color=LIME, size=9, weight="bold").pack(side="left", padx=12, pady=8)
    label(log_header, "LIVE OUTPUT", color=MUTED, size=8, weight="bold").pack(side="right", padx=12, pady=8)
    log_box = tk.Text(log_panel, height=7, bg=BG, fg=TEXT, insertbackground=LIME,
                      selectbackground=TEAL, relief="flat", bd=0, wrap="word",
                      font=("Consolas", 9), padx=12, pady=10, state="disabled")
    log_box.pack(fill="both", expand=True, padx=1, pady=1)
    log_box.tag_configure("success", foreground=LIME)
    log("MANTIS GALLERY // MG01 CONVERTER READY")
    log("Supported: JPG  JPEG  PNG  BMP  WEBP")
    log("Output: 320 × 240 / RGB565 / little-endian pixel data")

    # Footer
    footer = tk.Frame(root, bg=BG)
    footer.pack(fill="x", padx=22, pady=(0, 12))
    label(footer, "BASALT SOFTWORKS", color=TEAL_HI, size=8, weight="bold").pack(side="left")
    label(footer, textvariable=status_var, color=MUTED, size=8).pack(side="right")

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
