# Mantis Gallery — M5Stack Core2

SD-card photo gallery. Scans `/images` on the microSD card, plays images in a **random no-repeat shuffle**, and lets you cycle **albums** (subfolders) with the side buttons.

## SD layout

```
/images/                 default album (files here)
  shot1.mg
  shot2.jpg
  vacation/              album "vacation"
    a.mg
    b.png
  family/
    …
```

Supported extensions: **`.mg`** (fast), **`.jpg` / `.jpeg`**, **`.png`**.

See **`IMAGE_SPEC.txt`** for exact resolution and the `.mg` binary format.

## Controls (no on-screen labels)

**Default album** (`/images` root files):

| Input | Action |
|-------|--------|
| **Hold B** | Yellow→red bar; hold until full → random subfolder |
| **Swipe right / left** | Next / previous image |
| **Long-press screen** | Slideshow on/off |

**Subfolders:**

| Input | Action |
|-------|--------|
| **A / C** | Previous / next subfolder |
| **Short B** | Back to default album |
| Swipes / long-press | Same as default |

**Slideshow:** 5–8 s random dwell; **tap** doubles time on current photo; swipes still work.

## Fast path (recommended)

JPEG/PNG decode on ESP32 is fine for a few images; for large folders use the converter:

```bash
pip install pillow
python tools/mg_convert.py
```

Produces 320×240 RGB565 `.mg` files the firmware can push almost instantly.

## Build

```bash
pio run -e m5stack-core2
pio run -e m5stack-core2 -t upload
```

## Flash via web

GitHub Actions builds firmware and publishes the `web/` flasher to Pages (same drop-zip workflow as DeviceRadar).

Direct download on the site: **merged `mantis-gallery.bin`** only (plus this repo’s `IMAGE_SPEC.txt` / converter in source).
