"""Smart selection for Mantis Gallery masks (water areas, depth planes) — the same integer maths as the
Core2 firmware (src/jiggle_select.h), so a selection made in the lab is the one the device would make.

Two ways in, both "simply work" from loose input:

  grow(img, marks, step)   a tap or a stroke over a sample. Grows over the marked stuff to its natural
                           boundary: every pixel gets a crossing cost (edges of a lightly smoothed image
                           and colour difference from the sample), the region at each cost threshold is
                           the set reachable without ever crossing more than that (a bottleneck path, so
                           long thin rivers and wide seas behave the same), and the thresholds where the
                           region stops growing are its natural boundaries. The best guess climbs those
                           while each step still adds the same stuff, and stops at a different material
                           or a flood through a weak spot. step = −1/+1 picks the next smaller / bigger
                           boundary. Pieces of the same stuff cut off by something thin (a pole, an arm)
                           come along.
  snap(img, rough, band)   a rough shape drawn around something. Keeps what's surely inside and outside
                           and finds the real edge in between: object and background grow toward each
                           other (costly across edges and through the other side's colours), learning
                           both colour models from the shape and then from their own answer (GrabCut-
                           style), with a mild preference for "inside the shape you drew".

Inputs are boolean (240, 320) arrays; the image is (240, 320, 3) uint8 RGB.
"""
from __future__ import annotations

import numpy as np

W, H = 320, 240
N = W * H
INF = 65535
K_MAX, NS_MAX = 8, 2048
NEAR_PX = 24          # cut-off pieces of the same stuff within this distance come along
HOLE_PX = 120         # enclosed gaps up to this size are filled (glints, flecks)
D_HIST = 1200


# ---- colour ------------------------------------------------------------------------------------------
def planes(img):
    """Luma / chroma of the RGB565-quantised picture (int32, (H, W, 3)): Y, B−Y, R−Y."""
    a = np.asarray(img)[..., :3].astype(np.int32)
    r, g, b = (a[..., 0] >> 3) << 3, (a[..., 1] >> 2) << 2, (a[..., 2] >> 3) << 3
    y = (77 * r + 150 * g + 29 * b) >> 8
    return np.stack([y, b - y, r - y], -1)


def dist(f, c):
    """weighted L1, luma counted half: f (..., 3), c (K, 3) → (..., K)"""
    d = f[..., None, :] - c
    return (np.abs(d[..., 0]) >> 1) + np.abs(d[..., 1]) + np.abs(d[..., 2])


def tdiv(s, n):               # C-style integer division (truncates toward zero)
    return np.sign(s) * (np.abs(s) // n)


def fdiv9(s):                 # floor(s / 9) for any sign (as the firmware does it)
    return np.floor_divide(s, 9)


def blur3(F):
    p = np.pad(F, ((1, 1), (1, 1), (0, 0)), mode="edge")
    s = sum(p[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    return fdiv9(s)


def edges(F):
    E = np.zeros((H, W), np.int32)
    dx = (np.abs(F[:, 1:, 0] - F[:, :-1, 0]) >> 1) + np.abs(F[:, 1:, 1] - F[:, :-1, 1]) + np.abs(F[:, 1:, 2] - F[:, :-1, 2])
    dy = (np.abs(F[1:, :, 0] - F[:-1, :, 0]) >> 1) + np.abs(F[1:, :, 1] - F[:-1, :, 1]) + np.abs(F[1:, :, 2] - F[:-1, :, 2])
    E[:, 1:] = np.maximum(E[:, 1:], dx); E[:, :-1] = np.maximum(E[:, :-1], dx)
    E[1:, :] = np.maximum(E[1:, :], dy); E[:-1, :] = np.maximum(E[:-1, :], dy)
    return np.minimum(E, 255)


def take(idx):
    n = len(idx)
    if n <= NS_MAX:
        return idx
    return idx[::-(-n // NS_MAX)]


def kmeans(s):
    """Deterministic k-means (farthest-point start, 6 rounds). Returns centres, mean distance."""
    s = s.astype(np.int64)
    k = int(min(K_MAX, max(1, len(s) // 16)))
    c = [tdiv(s.sum(0), len(s))]
    for _ in range(1, k):
        dm = dist(s, np.array(c)).min(1)
        c.append(s[int(np.argmax(dm))])
    c = np.array(c, np.int64)
    for _ in range(6):
        lab = dist(s, c).argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = tdiv(s[m].sum(0), int(m.sum()))
    return c, int(dist(s, c).min(1).sum() // len(s))


def kth(vals, k):
    """k-th smallest (0-based) of non-negative ints — what the firmware gets from a histogram."""
    return int(np.partition(np.asarray(vals).ravel(), k)[k])


BAR_RGB = (24, 24, 24)     # the gallery's backdrop (0x18C3) that smaller pictures are centred on


def bars(img):
    """Letterbox / pillarbox bars: whole columns / rows of exactly the gallery's backdrop colour at the
    frame's edges. Never part of a selection."""
    a = np.asarray(img)[..., :3].astype(np.int32)
    isbg = (a == np.array(BAR_RGB)).all(-1)
    m = np.zeros((H, W), bool)
    for rng in (range(W), range(W - 1, -1, -1)):
        for x in rng:
            if isbg[:, x].all(): m[:, x] = True
            else: break
    for rng in (range(H), range(H - 1, -1, -1)):
        for y in rng:
            if isbg[y].all(): m[y] = True
            else: break
    return m


# ---- distances ---------------------------------------------------------------------------------------
def _scan(d, c):
    C = np.cumsum(c)
    d = np.minimum(d, C + np.minimum.accumulate(d - C))
    cr = c[::-1]; Cr = np.cumsum(cr); dr = d[::-1]
    dr = np.minimum(dr, Cr + np.minimum.accumulate(dr - Cr))
    return dr[::-1]


def geodesic(d, c, rounds=3):
    """Cheapest path cost from the zeros of d (cost c to enter a pixel), by raster sweeps."""
    d = d.astype(np.int64); c = c.astype(np.int64)
    for _ in range(rounds):
        for y in range(H):
            if y:
                d[y] = np.minimum(d[y], d[y - 1] + c[y])
            d[y] = _scan(d[y], c[y])
        for y in range(H - 1, -1, -1):
            if y < H - 1:
                d[y] = np.minimum(d[y], d[y + 1] + c[y])
            d[y] = _scan(d[y], c[y])
    return np.minimum(d, INF)


def chamfer(m):
    """city-block distance to the True pixels (exact)"""
    return geodesic(np.where(m, 0, 4096), np.ones((H, W), np.int64), 1)


def majority(m):
    p = np.pad(m.astype(np.int32), 1)
    s = sum(p[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    return s >= 5


def _flood(avail, x, y):
    """Scanline flood over avail (claimed pixels cleared). Returns (count, touched border, spans)."""
    spans, n, edge = [], 0, False
    stack = [(x, y)]
    while stack:
        xx, yy = stack.pop()
        row = avail[yy]
        if not row[xx]:
            continue
        a = xx
        while a > 0 and row[a - 1]:
            a -= 1
        b = xx
        while b < W - 1 and row[b + 1]:
            b += 1
        row[a:b + 1] = False
        spans.append((yy, a, b)); n += b - a + 1
        if a == 0 or b == W - 1 or yy == 0 or yy == H - 1:
            edge = True
        for ny in (yy - 1, yy + 1):
            if 0 <= ny < H:
                want = avail[ny, a:b + 1]
                starts = np.flatnonzero(want & ~np.concatenate(([False], want[:-1])))
                stack.extend((a + int(k), ny) for k in starts)
    return n, edge, spans


def flood_le(cost, seeds, v):
    """seeds + every pixel reachable from them through pixels of cost <= v"""
    avail = (cost <= v) | seeds
    sel = np.zeros((H, W), bool)
    for y, x in zip(*np.nonzero(seeds)):
        if avail[y, x]:
            for r, a, b in _flood(avail, int(x), int(y))[2]:
                sel[r, a:b + 1] = True
    return sel


# ---- grow (tap / stroke) -----------------------------------------------------------------------------
def seed_core(F, seeds):
    """A tap's colour = what's under the finger, without the stray pixels of an edge it straddles."""
    idx = np.flatnonzero(seeds)
    if len(idx) <= 8:
        return idx
    f = F.reshape(-1, 3)[idx].astype(np.int64)
    m = np.sort(f, 0)[len(f) // 2]
    d = (np.abs(f[:, 0] - m[0]) >> 1) + np.abs(f[:, 1] - m[1]) + np.abs(f[:, 2] - m[2])
    return idx[d <= kth(d, (len(d) * 6) // 10)]


def area_curve(cost, seeds):
    """Size of the marks' region among pixels of cost <= v, for v = 0..255 (union-find by level)."""
    c = np.where(seeds, 0, cost).ravel()
    order = np.argsort(c, kind="stable").tolist()
    cl = c.tolist()
    par = list(range(N)); size = [1] * N; on = [False] * N
    s0 = int(np.flatnonzero(seeds.ravel())[0])

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            if size[a] < size[b]:
                a, b = b, a
            par[b] = a; size[a] += size[b]

    area = np.zeros(256, np.int64)
    k = 0
    for v in range(256):
        while k < N and cl[order[k]] == v:
            i = order[k]; k += 1; on[i] = True
            x = i % W
            if x > 0 and on[i - 1]: union(i, i - 1)
            if x < W - 1 and on[i + 1]: union(i, i + 1)
            if i >= W and on[i - W]: union(i, i - W)
            if i + W < N and on[i + W]: union(i, i + W)
        if v == 0:                                   # (marks placed before s0 joined it too)
            for i in np.flatnonzero(seeds.ravel()).tolist():
                union(i, s0)
        area[v] = size[find(s0)]
    return area


def stable_levels(area, n_seed, delta=10):
    """Natural boundaries: local minima of relative growth (×1024), smallest region first."""
    lo = max(48, 4 * n_seed)
    q = [None] * 256
    for v in range(delta, 256 - delta):
        a = int(area[v])
        if lo <= a <= (6 * N) // 10:
            q[v] = ((int(area[v + delta]) - int(area[v - delta])) * 1024) // a
    lv = []
    for v in range(delta, 256 - delta):
        if q[v] is None or q[v] > 900:
            continue
        if (q[v - 1] is not None and q[v - 1] < q[v]) or (q[v + 1] is not None and q[v + 1] <= q[v]):
            continue
        a = int(area[v])
        if lv and a * 10 <= int(area[lv[-1][0]]) * 12:      # same region (within 20 %): keep the steadier
            if q[v] < lv[-1][1]:
                lv[-1] = (v, q[v])
            continue
        lv.append((v, q[v]))
    return lv


class Grown:
    """A grow's answer and its ladder (so Smaller / Bigger can step without starting over)."""
    def __init__(self, sel, levels, index, best):
        self.sel, self.levels, self.index, self.best = sel, levels, index, best


def grow(img, seeds, step=0, F=None):
    F = planes(img) if F is None else F
    seeds = np.asarray(seeds, bool)
    if not seeds.any():
        return Grown(seeds.copy(), [], -1, -1)
    core = seed_core(F, seeds)
    c, spread = kmeans(F.reshape(-1, 3)[take(core)])
    Fb = blur3(F)
    D = dist(Fb, c).min(-1)
    E = edges(Fb)
    cost = np.minimum(255, np.maximum(E, np.minimum((D * 24) // (spread + 8), 255)))
    cost[bars(img) & ~seeds] = 255                       # the bars around a small picture: never
    area = area_curve(cost, seeds)
    lv = stable_levels(area, int(seeds.sum()))
    if not lv:
        return Grown(seeds.copy(), [], -1, -1)
    tau = 40 + spread // 2
    best, prev, masks = 0, seeds, []
    for i, (v, q) in enumerate(lv):              # climb while each step adds the same stuff
        m = flood_le(cost, seeds, v)
        masks.append(m)
        inc = D[m & ~prev]
        prev = m
        if i == 0:
            continue
        if len(inc) == 0 or kth(inc, len(inc) // 2) > tau:
            break
        a0, a1 = int(area[lv[i - 1][0]]), int(area[v])
        if a1 > 4 * a0 and a1 - a0 > N // 20:     # a flood through a weak spot
            break
        best = i
    i = min(max(best + step, 0), len(lv) - 1)
    while len(masks) <= i:
        masks.append(flood_le(cost, seeds, lv[len(masks)][0]))
    sel = masks[i].copy()
    # pieces of the same stuff cut off by something thin
    n_sel = int(sel.sum())
    t_piece = kth(cost[sel], (n_sel * 95) // 100)
    tau_p = min(kth(D[sel], (n_sel * 3) // 4), tau)   # like the typical selected pixel, never looser than a climb step
    near = chamfer(sel) <= NEAR_PX
    avail = (cost <= t_piece) & ~sel
    add = np.zeros((H, W), bool)
    for y, x in zip(*np.nonzero(avail & near)):
        if not avail[y, x]:
            continue
        n, _, spans = _flood(avail, int(x), int(y))
        if n < 12 or n > 3 * n_sel:
            continue
        vals = np.concatenate([D[r, a:b + 1] for r, a, b in spans])
        if kth(vals, len(vals) // 2) <= tau_p:
            for r, a, b in spans:
                add[r, a:b + 1] = True
    sel |= add
    return Grown(majority(sel) | seeds, lv, i, best)


# ---- snap (rough shape → real edges) -----------------------------------------------------------------
def _models(F, fg_idx, bg_idx):
    Ff = F.reshape(-1, 3)
    cf, _ = kmeans(Ff[take(fg_idx)])
    cb, _ = kmeans(Ff[take(bg_idx)])
    Df = dist(F, cf).min(-1); Db = dist(F, cb).min(-1)
    return (32 * Df) // (Df + Db + 1), (32 * Db) // (Df + Db + 1)


SNAP_BG = 21


def _keep_core(sel, core):
    out = np.zeros_like(sel)
    av = sel.copy()
    for y, x in zip(*np.nonzero(core & sel)):
        if av[y, x]:
            for r, a, b in _flood(av, int(x), int(y))[2]:
                out[r, a:b + 1] = True
    return out


def grabcut(img, rough, iters=5):
    """GrabCut (Rother, Kolmogorov & Blake 2004: colour mixture models + a global min-cut), via OpenCV.
    Inside the rough shape = probably the object, outside and the bars = background. None without OpenCV."""
    try:
        import cv2
    except ImportError:
        return None
    rough = np.asarray(rough, bool) & ~bars(img)
    if rough.sum() < 12:
        return rough.copy()
    m = np.where(rough, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    bgr = np.ascontiguousarray(np.asarray(img)[..., 2::-1])
    try:
        cv2.grabCut(bgr, m, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    sel = (m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)
    if not sel.any():
        return sel
    # drop stray specks; keep every substantial piece (an object can be split by what's in front of it)
    av = sel.copy(); comps = []
    for y, x in zip(*np.nonzero(sel)):
        if av[y, x]:
            c = _flood(av, int(x), int(y)); comps.append(c)
    big = max(c[0] for c in comps)
    out = np.zeros_like(sel)
    for n, _, spans in comps:
        if n >= max(30, big // 10):
            for r, a, b in spans:
                out[r, a:b + 1] = True
    return out


def snap(img, rough, band=8, F=None, engine="auto"):
    """A rough shape → the object inside it. engine "auto": GrabCut when OpenCV is installed (the lab),
    else the built-in fit (the same maths the Core2 runs)."""
    if engine in ("auto", "grabcut"):
        g = grabcut(img, rough)
        if g is not None:
            return g
    return snap_builtin(img, rough, band, F)


def snap_builtin(img, rough, band=8, F=None):
    F = planes(img) if F is None else F
    B = bars(img)
    rough = np.asarray(rough, bool) & ~B
    if rough.sum() < 12:
        return rough.copy()
    E = edges(F)
    dist_in = chamfer(~rough); dist_out = chamfer(rough)
    b_in = max(band, int(dist_in.max()) // 2)        # how loose a rough shape can be, inward
    b_out = max(3, band // 2)                         # and a little outward
    fg_seed = dist_in > b_in
    if fg_seed.sum() < 20:
        fg_seed = dist_in > b_in // 2
    bg_seed = (dist_out > b_out) | B
    ring_b = bg_seed & (dist_out <= 3 * b_in) & ~B
    if not ring_b.any():
        ring_b = bg_seed
    if not fg_seed.any() or not ring_b.any():
        return rough.copy()
    unknown = ~(fg_seed | bg_seed)
    ce = 1 + (E >> 3)
    prior = np.where(rough, (np.minimum(dist_in, b_in) * 12) // max(1, b_in), 0)   # "inside what you drew"
    fg_s, bg_s = rough, ring_b
    sel = rough
    for _ in range(2):                                # learn both sides from the shape, then the answer
        Lb, Lf = _models(F, np.flatnonzero(fg_s), np.flatnonzero(bg_s))
        dfg = geodesic(np.where(fg_seed, 0, INF), ce + Lb)
        dbg = geodesic(np.where(bg_seed & ~fg_seed, 0, INF), ce + Lf + prior)
        sel = (((dfg < dbg) & (dfg < INF)) & unknown) | (fg_seed & ~unknown)
        sel &= ~(unknown & (Lb >= SNAP_BG))           # GrabCut's data term: clearly the background's colour
        sel = _keep_core(sel, fg_seed)                # …and only what's joined to the object
        fg_s = sel
        bg_s = ring_b | (rough & ~sel)
        if not fg_s.any():
            break
    return (majority(sel) | fg_seed) & ~bg_seed


# ---- applying to a mask ------------------------------------------------------------------------------
def apply(mask, sel, add=True, trim=1):
    """Add (or remove) a selection. Enclosed small gaps are filled. trim=1: an added area keeps its
    outer pixel off whatever isn't already in the mask (water stays off the bank) and a removed area
    takes one extra; 0: exact; −1: an added area takes its edge pixels too (a depth plane carries its
    own rim, so no ghost halo is left behind when it slides). Returns pixels changed."""
    t = sel.copy()
    avail = ~t
    start = np.zeros_like(t)                          # gaps are looked for from next to the selection
    start[1:-1, 1:-1] = avail[1:-1, 1:-1] & (t[1:-1, :-2] | t[:-2, 1:-1])
    for yy, xx in zip(*np.nonzero(start)):
        if not avail[yy, xx]:
            continue
        n, edge, spans = _flood(avail, int(xx), int(yy))
        if n <= HOLE_PX and not edge:
            for r, a, b in spans:
                t[r, a:b + 1] = True
    before = mask.copy()
    if add and trim < 0:                                # take the edge pixels too (depth planes)
        q = np.pad(t, 1)
        mask |= t | q[:-2, 1:-1] | q[2:, 1:-1] | q[1:-1, :-2] | q[1:-1, 2:]
    elif add:
        if trim:
            p = np.pad(t | mask, 1, mode="edge")
            t = t & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
        mask |= t
    else:
        if trim:
            q = np.pad(t, 1)
            t = t | q[:-2, 1:-1] | q[2:, 1:-1] | q[1:-1, :-2] | q[1:-1, 2:]
        mask &= ~t
    return int((before ^ mask).sum())


def disc(x, y, r=4):
    yy, xx = np.mgrid[:H, :W]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r * r


def grow_mask(mask, n=1):
    """Expand (n > 0) or contract (n < 0) a mask by |n| px (4-neighbour steps)."""
    m = np.asarray(mask, bool).copy()
    for _ in range(abs(n)):
        q = np.pad(m, 1, constant_values=(n < 0))
        nb = [q[:-2, 1:-1], q[2:, 1:-1], q[1:-1, :-2], q[1:-1, 2:]]
        m = (m | nb[0] | nb[1] | nb[2] | nb[3]) if n > 0 else (m & nb[0] & nb[1] & nb[2] & nb[3])
    return m
