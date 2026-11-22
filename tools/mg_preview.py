"""Record the Studio's live preview to an animated GIF (or frames): the same maths the preview runs -
the Core2's own - driven by a script of motions, tilt / lean and dance, so a scene can be checked (and
shown) exactly as it will play."""
import math

import numpy as np

import jiggle_lab as J

DEFAULT_SCRIPT = [            # (start s, what): a motion name from jiggle_lab.MOTIONS, "dance on/off", "still",
    (0.0, "tiltlr"), (4.2, "sway"), (7.6, "shake"), (8.8, "walk"), (13.0, "dance on"), (19.0, "dance off")]
#                               "turn A S" (ease the turn to A° over S s), "lean A S" (top away / toward you)
SHOWCASE = [                  # turned right over and wiggled upside down, rocked, and danced (29 s)
    (0.0, "still"), (1.5, "sway"), (5.0, "turn 180 3"), (8.5, "shake"), (9.8, "tiltlr"), (13.2, "turn 0 3"),
    (16.5, "lean 45 1.2"), (17.7, "lean -45 1.6"), (19.3, "lean 0 1.2"), (20.6, "dance on"), (27.0, "dance off")]
SCRIPTS = {"Mixed motions": (DEFAULT_SCRIPT, 21.0), "Showcase (over, upside down, dance)": (SHOWCASE, 29.0)}


def _ease(a, b, u):
    u = 0.0 if u < 0 else (1.0 if u > 1 else u)
    return a + (b - a) * u * u * (3 - 2 * u)


def frames(sim, script=DEFAULT_SCRIPT, seconds=21.0, fps=20, bpm=112, crop=None, tilt=0.0, lean=0.0, info=False):
    """Yields uint8 RGB frames of the preview (crop: (x0, y0, x1, y1) of the 320×240 picture)."""
    dt = 1.0 / fps
    t, motion, dance, amp, sign, next_beat = 0.0, None, False, 0.0, 1, 0.0
    ev = sorted(script)
    period = 60.0 / bpm
    turn = [tilt, tilt, 0.0, 1.0]                       # (from, to, start, seconds): eased
    lean_ = [lean, lean, 0.0, 1.0]
    lean_prev = lean
    for f in range(int(seconds * fps)):
        while ev and ev[0][0] <= t + 1e-9:
            _, what = ev.pop(0)
            w = what.split()
            if w[0] in ("turn", "lean") and len(w) == 3:
                ease = turn if w[0] == "turn" else lean_
                cur = _ease(ease[0], ease[1], (t - ease[2]) / ease[3])
                ease[:] = [cur, float(w[1]), t, max(0.05, float(w[2]))]
                continue
            if what == "dance on":
                dance, next_beat = True, t
            elif what == "dance off":
                dance = False
            elif what == "still":
                motion = None
            else:
                motion = (what, t)
        tl = _ease(turn[0], turn[1], (t - turn[2]) / turn[3])
        ln = _ease(lean_[0], lean_[1], (t - lean_[2]) / lean_[3])
        acc = J.motion_accel("none", 0.0, tl, ln)
        twist, gyro = (0.0, 0.0), (0.0, 0.0)
        if motion and t - motion[1] <= 6.5:
            k, ts = motion
            acc = J.motion_accel(k, t - ts, tl, ln)
            twist = J.motion_twist(k, t - ts)
            gyro = J.motion_gyro(k, t - ts)
        sim.gyro = (gyro[0] - (ln - lean_prev) / dt, gyro[1])   # (leaning it is a turn the depth view sees)
        lean_prev = ln
        amp = min(1.0, amp + dt / (1.5 * period)) if dance else max(0.0, amp - dt)
        if amp > 0 and t >= next_beat:
            sign = -sign
            sim.beat(0.35 * amp, sign)
            next_beat = max(next_beat + period, t)
        sim.step(dt, acc, amp, twist)
        t += dt
        img = sim.render()
        if crop:
            x0, y0, x1, y1 = crop
            img = img[y0:y1, x0:x1]
        yield (img, dict(t=t, turned=tl, leaned=ln, dance=amp, motion=motion[0] if motion and t - motion[1] <= 6.5 else "")) if info else img


def write(path, frames_, fps=15, scale=2):
    """Frames → an animated GIF, or an MP4 when the name ends .mp4 (ffmpeg on the PATH; else a GIF beside it)."""
    import shutil
    import subprocess
    from PIL import Image
    frames_ = [np.asarray(f) for f in frames_]
    h, w = frames_[0].shape[:2]
    if str(path).lower().endswith(".mp4") and shutil.which("ffmpeg"):
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                              f"{w * scale}x{h * scale}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset",
                              "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                             stdin=subprocess.PIPE)
        for f in frames_:
            p.stdin.write(np.asarray(Image.fromarray(f).resize((w * scale, h * scale), Image.LANCZOS)).tobytes())
        p.stdin.close()
        p.wait()
        return path
    if str(path).lower().endswith(".mp4"):
        path = str(path)[:-4] + ".gif"
    im = [Image.fromarray(f).resize((w * scale, h * scale), Image.LANCZOS) for f in frames_]
    im[0].save(path, save_all=True, append_images=im[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return path


def save_gif(path, sim, scale=2, **kw):
    from PIL import Image
    fps = kw.get("fps", 20)
    fr = [Image.fromarray(np.asarray(f)).resize((f.shape[1] * scale, f.shape[0] * scale), Image.LANCZOS)
          for f in frames(sim, **kw)]
    fr[0].save(path, save_all=True, append_images=fr[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return len(fr)
