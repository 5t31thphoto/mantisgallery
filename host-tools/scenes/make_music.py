"""The test music (16 kHz mono, 26 s): silence, then 120 BPM drums and bass from 16 s to 24 s - what play.cpp's
microphone hears when dance mode is switched on."""
import os, numpy as np
sr = 16000; T = 26.0; a = np.zeros(int(sr * T)); bpm = 120; beat = 60 / bpm
rng = np.random.default_rng(1)
t0, t1 = 16.0, 24.0
for k in range(int((t1 - t0) / beat * 2)):          # eighths
    t = t0 + k * beat / 2; i = int(t * sr)
    if k % 2 == 0:                                  # kick on every beat
        n = int(0.18 * sr); tt = np.arange(n) / sr; f = 50 + 90 * np.exp(-tt * 30)
        a[i:i + n] += 0.9 * np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-tt * 14)
    if k % 4 == 2:                                  # snare on 2 and 4
        n = int(0.12 * sr); tt = np.arange(n) / sr
        a[i:i + n] += 0.35 * rng.standard_normal(n) * np.exp(-tt * 25)
    n = int(0.03 * sr); tt = np.arange(n) / sr      # hats
    a[i:i + n] += 0.12 * np.diff(np.r_[0, rng.standard_normal(n)]) * np.exp(-tt * 120)
tt = np.arange(int((t1 - t0) * sr)) / sr            # a bass line
notes = [55, 55, 65.4, 49]; f = np.array([notes[int(x / (beat * 4)) % 4] for x in tt])
a[int(t0 * sr):int(t0 * sr) + len(tt)] += 0.25 * np.sin(2 * np.pi * np.cumsum(f) / sr)
a = np.clip(a / np.abs(a).max() * 0.6, -1, 1)
(a * 32767).astype('<i2').tofile(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'music16k.raw'))
