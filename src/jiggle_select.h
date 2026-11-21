#pragma once
// Smart selection for masks (water areas, depth planes) — the same integer maths as tools/mg_select.py,
// so the Core2 and the lab agree pixel for pixel.
//
//   sel_grow   a tap or a stroke over a sample → the marked stuff, out to its natural boundary.
//              Each pixel gets a crossing cost (edges of a lightly smoothed picture, colour difference
//              from the sample); the region at each threshold is what's reachable without ever crossing
//              more (bottleneck paths: long rivers and wide seas behave alike); thresholds where it stops
//              growing are its natural boundaries. The best guess climbs them while each step adds the
//              same stuff and stops at a new material or a flood through a weak spot; step −1/+1 = the
//              next smaller / bigger boundary. Pieces of the same stuff cut off by something thin (a pole,
//              an arm) come along.
//   sel_snap   a rough shape around something → its real edge: object and background grow toward each
//              other (costly across edges and through each other's colours), colour models learnt from
//              the shape and then from their own answer, with a mild preference for "inside the shape".
//   sel_apply  add / remove a selection on a mask: small enclosed gaps filled; optional 1 px trim.
// Work memory comes from `alloc` (PSRAM) and is freed before returning; ~1.3 MB at peak.
#include <algorithm>
#include "jiggle_water.h"

namespace jig {

constexpr int SEL_N = IMG_W * IMG_H;
constexpr int SEL_INF = 65535;
constexpr int SEL_KMAX = 8, SEL_NSMAX = 2048;
constexpr int SEL_NEAR = 24, SEL_HOLE = 120, SEL_DHIST = 1200, SEL_STACK = 65536;
constexpr int SEL_SNAP_BG = 21;                         // snap: colour this background-like (of 32) is dropped

struct SelC { int y, cb, cr; };
inline int sel_d(const SelC& a, const SelC& b) { return (abs(a.y - b.y) >> 1) + abs(a.cb - b.cb) + abs(a.cr - b.cr); }
inline SelC sel_col(const uint16_t* img, bool sw, int i) {
    PickColor p = pick_color(sw ? (uint16_t)((img[i] >> 8) | (img[i] << 8)) : img[i]);
    return {p.y, p.cb, p.cr};
}
inline int sel_floor9(int s) { int q = s / 9; if ((s % 9) != 0 && s < 0) --q; return q; }

// ---- small helpers --------------------------------------------------------------------------------------
struct SelBuf {                                      // frees everything it handed out
    void* (*alloc)(size_t);
    void* p[16]; int n = 0; bool ok = true;
    explicit SelBuf(void* (*a)(size_t)) : alloc(a) {}
    template <typename T> T* get(size_t count) {
        T* q = (T*)alloc(count * sizeof(T));
        if (!q) ok = false; else if (n < 16) p[n++] = q;
        return q;
    }
    ~SelBuf() { for (int i = 0; i < n; ++i) free(p[i]); }
};
inline bool bget(const uint8_t* m, int i) { return (m[i >> 3] >> (i & 7)) & 1; }
inline void bset(uint8_t* m, int i) { m[i >> 3] |= (uint8_t)(1 << (i & 7)); }
inline void bclr(uint8_t* m, int i) { m[i >> 3] &= (uint8_t)~(1 << (i & 7)); }

// k-th smallest (0-based) via a histogram of values 0..nb−1
struct SelHist {
    int* h; int nb, n = 0;
    void clear() { memset(h, 0, sizeof(int) * nb); n = 0; }
    void add(int v) { ++h[v < nb ? v : nb - 1]; ++n; }
    int kth(int k) const { int s = 0; for (int v = 0; v < nb; ++v) { s += h[v]; if (s > k) return v; } return nb - 1; }
};

// deterministic k-means over samples (farthest-point start, 6 rounds); returns k, sets spread
inline int sel_kmeans(const SelC* s, int n, SelC* c, int& spread, int* lab) {
    int k = n / 16; if (k < 1) k = 1; if (k > SEL_KMAX) k = SEL_KMAX;
    long sy = 0, sb = 0, sr = 0;
    for (int i = 0; i < n; ++i) { sy += s[i].y; sb += s[i].cb; sr += s[i].cr; }
    c[0] = {(int)(sy / n), (int)(sb / n), (int)(sr / n)};
    for (int j = 1; j < k; ++j) {
        int bi = 0, bd = -1;
        for (int i = 0; i < n; ++i) {
            int dm = 1 << 30;
            for (int q = 0; q < j; ++q) { int d = sel_d(s[i], c[q]); if (d < dm) dm = d; }
            if (dm > bd) { bd = dm; bi = i; }
        }
        c[j] = s[bi];
    }
    for (int it = 0; it < 6; ++it) {
        for (int i = 0; i < n; ++i) {
            int bq = 0, bd = sel_d(s[i], c[0]);
            for (int q = 1; q < k; ++q) { int d = sel_d(s[i], c[q]); if (d < bd) { bd = d; bq = q; } }
            lab[i] = bq;
        }
        for (int q = 0; q < k; ++q) {
            long ay = 0, ab = 0, ar = 0; int cnt = 0;
            for (int i = 0; i < n; ++i) if (lab[i] == q) { ay += s[i].y; ab += s[i].cb; ar += s[i].cr; ++cnt; }
            if (cnt) c[q] = {(int)(ay / cnt), (int)(ab / cnt), (int)(ar / cnt)};
        }
    }
    long tot = 0;
    for (int i = 0; i < n; ++i) {
        int bd = sel_d(s[i], c[0]);
        for (int q = 1; q < k; ++q) { int d = sel_d(s[i], c[q]); if (d < bd) bd = d; }
        tot += bd;
    }
    spread = (int)(tot / n);
    return k;
}
// samples of the pixels where pick(i) (raster order, thinned to ≤ SEL_NSMAX)
template <typename Pick, typename Col>
inline int sel_take(Pick pick, Col col, SelC* out) {
    int n = 0;
    for (int i = 0; i < SEL_N; ++i) n += pick(i) ? 1 : 0;
    if (!n) return 0;
    int st = n <= SEL_NSMAX ? 1 : (n + SEL_NSMAX - 1) / SEL_NSMAX, k = 0, m = 0;
    for (int i = 0; i < SEL_N; ++i)
        if (pick(i)) { if (k % st == 0) out[m++] = col(i); ++k; }
    return m;
}

// cheapest path from d == 0 pixels (cost c[i] to enter i), raster sweeps; d holds 0 / SEL_INF on entry
inline void sel_geodesic(uint16_t* d, const uint8_t* c, int rounds, const uint8_t* c2 = nullptr) {
    auto cc = [&](int i) { return (int)c[i] + (c2 ? (int)c2[i] : 0); };
    for (int r = 0; r < rounds; ++r) {
        for (int pass = 0; pass < 2; ++pass)
            for (int yy = 0; yy < IMG_H; ++yy) {
                const int y = pass ? IMG_H - 1 - yy : yy, row = y * IMG_W;
                const int py = pass ? y + 1 : y - 1;
                if (py >= 0 && py < IMG_H)
                    for (int x = 0; x < IMG_W; ++x) { int v = d[py * IMG_W + x] + cc(row + x); if (v < d[row + x]) d[row + x] = (uint16_t)v; }
                for (int x = 1; x < IMG_W; ++x) { int v = d[row + x - 1] + cc(row + x); if (v < d[row + x]) d[row + x] = (uint16_t)v; }
                for (int x = IMG_W - 2; x >= 0; --x) { int v = d[row + x + 1] + cc(row + x); if (v < d[row + x]) d[row + x] = (uint16_t)v; }
            }
    }
}
// city-block distance to the set bits of m (or to the clear bits, inv)
inline void sel_chamfer(const uint8_t* m, bool inv, uint16_t* d, uint8_t* ones) {
    for (int i = 0; i < SEL_N; ++i) { d[i] = (bget(m, i) != inv) ? 0 : 4096; ones[i] = 1; }
    sel_geodesic(d, ones, 1);
}
inline void sel_majority(const uint8_t* m, uint8_t* out) {
    memset(out, 0, MASK_BYTES);
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x) {
            int s = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) s += mask_get(m, x + dx, y + dy) ? 1 : 0;
            if (s >= 5) bset(out, y * IMG_W + x);
        }
}

// Scanline flood over avail(x, y) (claim() must make avail false). Spans go to span(y, a, b).
// Returns the count; edge = touched the frame. The stack (SEL_STACK seeds) is far beyond what a
// 320×240 fill can queue at once.
struct SelStack { int16_t* s; int cap; };
template <typename Avail, typename Claim, typename Span>
inline int sel_flood(int x, int y, Avail avail, Claim claim, Span span, SelStack& S, bool& edge) {
    int sp = 0, n = 0;
    edge = false;
    S.s[sp++] = (int16_t)x; S.s[sp++] = (int16_t)y;
    auto run = [&]() {
        while (sp > 0) {
            int yy = S.s[--sp], xx = S.s[--sp];
            if (!avail(xx, yy)) continue;
            int a = xx, b = xx;
            while (a > 0 && avail(a - 1, yy)) --a;
            while (b < IMG_W - 1 && avail(b + 1, yy)) ++b;
            for (int k = a; k <= b; ++k) claim(k, yy);
            span(yy, a, b);
            n += b - a + 1;
            if (a == 0 || b == IMG_W - 1 || yy == 0 || yy == IMG_H - 1) edge = true;
            for (int ny = yy - 1; ny <= yy + 1; ny += 2) {
                if (ny < 0 || ny >= IMG_H) continue;
                bool prev = false;
                for (int k = a; k <= b; ++k) {
                    bool w = avail(k, ny);
                    if (w && !prev && sp + 2 <= S.cap * 2) { S.s[sp++] = (int16_t)k; S.s[sp++] = (int16_t)ny; }
                    prev = w;
                }
            }
        }
    };
    run();
    return n;
}

// Letterbox / pillarbox bars: a picture smaller than the screen sits centred on the gallery's backdrop
// colour (0x18C3). Whole columns / rows of exactly that colour at the frame's edges are never selected.
constexpr uint16_t SEL_BAR = 0x18C3;
inline void sel_bars(const uint16_t* img, bool sw, uint8_t* B) {
    memset(B, 0, MASK_BYTES);
    auto px = [&](int i) { uint16_t v = img[i]; return sw ? (uint16_t)((v >> 8) | (v << 8)) : v; };
    auto col = [&](int x) { for (int y = 0; y < IMG_H; ++y) if (px(y * IMG_W + x) != SEL_BAR) return false; return true; };
    auto row = [&](int y) { for (int x = 0; x < IMG_W; ++x) if (px(y * IMG_W + x) != SEL_BAR) return false; return true; };
    for (int x = 0; x < IMG_W && col(x); ++x) for (int y = 0; y < IMG_H; ++y) bset(B, y * IMG_W + x);
    for (int x = IMG_W - 1; x >= 0 && col(x); --x) for (int y = 0; y < IMG_H; ++y) bset(B, y * IMG_W + x);
    for (int y = 0; y < IMG_H && row(y); ++y) for (int x = 0; x < IMG_W; ++x) bset(B, y * IMG_W + x);
    for (int y = IMG_H - 1; y >= 0 && row(y); --y) for (int x = 0; x < IMG_W; ++x) bset(B, y * IMG_W + x);
}

// ---- grow ----------------------------------------------------------------------------------------------
struct SelGrow { int levels = 0, index = -1, best = -1; };

// seeds: bitmap of the marks. out: result bitmap. Returns the ladder (levels, chosen index, best).
inline SelGrow sel_grow(const uint16_t* img, bool sw, const uint8_t* seeds, int step, uint8_t* out,
                        void* (*alloc)(size_t)) {
    SelGrow R;
    memset(out, 0, MASK_BYTES);
    int nseed = 0, s0 = -1;
    for (int i = 0; i < SEL_N; ++i) if (bget(seeds, i)) { if (s0 < 0) s0 = i; ++nseed; }
    if (!nseed) return R;
    SelBuf B(alloc);
    int16_t* fb = B.get<int16_t>((size_t)SEL_N * 3);
    uint16_t* D = B.get<uint16_t>(SEL_N);
    uint8_t* cost = B.get<uint8_t>(SEL_N);
    int32_t* par = B.get<int32_t>(SEL_N);
    int32_t* ord = B.get<int32_t>(SEL_N);
    SelC* samp = B.get<SelC>(nseed > SEL_NSMAX ? nseed : SEL_NSMAX);
    int* lab = B.get<int>(nseed > SEL_NSMAX ? nseed : SEL_NSMAX);
    int* hist = B.get<int>(SEL_DHIST);
    uint8_t* prev = B.get<uint8_t>(MASK_BYTES);
    uint8_t* cur = B.get<uint8_t>(MASK_BYTES);
    uint8_t* add = B.get<uint8_t>(MASK_BYTES);
    uint8_t* avail = B.get<uint8_t>(MASK_BYTES);
    int16_t* stk = B.get<int16_t>((size_t)SEL_STACK * 2);
    int32_t* spans = B.get<int32_t>((size_t)24000 * 3);
    uint8_t* ones = B.get<uint8_t>(SEL_N);
    uint8_t* bars = B.get<uint8_t>(MASK_BYTES);
    if (!B.ok) { memcpy(out, seeds, MASK_BYTES); return R; }
    SelStack S{stk, SEL_STACK};
    SelHist H{hist, SEL_DHIST};

    // the sample: the marks' colours, minus the stray pixels of an edge a tap straddles
    int n = 0;
    for (int i = 0; i < SEL_N; ++i) if (bget(seeds, i)) samp[n++] = sel_col(img, sw, i);
    if (n > 8) {
        int* ch = lab;
        int med[3];
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < n; ++i) ch[i] = c == 0 ? samp[i].y : (c == 1 ? samp[i].cb : samp[i].cr);
            std::sort(ch, ch + n);
            med[c] = ch[n / 2];
        }
        SelC m{med[0], med[1], med[2]};
        for (int i = 0; i < n; ++i) ch[i] = sel_d(samp[i], m);
        int* tmp = (int*)par;                            // (par is free until the area curve)
        memcpy(tmp, ch, sizeof(int) * n);
        std::nth_element(tmp, tmp + (n * 6) / 10, tmp + n);
        const int thr = tmp[(n * 6) / 10];
        int k = 0;
        for (int i = 0; i < n; ++i) if (ch[i] <= thr) samp[k++] = samp[i];
        n = k;
    }
    if (n > SEL_NSMAX) { int st = (n + SEL_NSMAX - 1) / SEL_NSMAX, k = 0; for (int i = 0; i < n; i += st) samp[k++] = samp[i]; n = k; }
    SelC cen[SEL_KMAX];
    int spread = 0;
    const int K = sel_kmeans(samp, n, cen, spread, lab);

    // smoothed planes, distance to the sample, edges → crossing cost
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x) {
            int s0y = 0, s0b = 0, s0r = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = x + dx < 0 ? 0 : (x + dx >= IMG_W ? IMG_W - 1 : x + dx);
                    int yy = y + dy < 0 ? 0 : (y + dy >= IMG_H ? IMG_H - 1 : y + dy);
                    SelC c = sel_col(img, sw, yy * IMG_W + xx);
                    s0y += c.y; s0b += c.cb; s0r += c.cr;
                }
            const int i = y * IMG_W + x;
            fb[i * 3] = (int16_t)sel_floor9(s0y); fb[i * 3 + 1] = (int16_t)sel_floor9(s0b); fb[i * 3 + 2] = (int16_t)sel_floor9(s0r);
        }
    auto FB = [&](int i) { return SelC{fb[i * 3], fb[i * 3 + 1], fb[i * 3 + 2]}; };
    for (int i = 0; i < SEL_N; ++i) {
        SelC f = FB(i);
        int bd = sel_d(f, cen[0]);
        for (int q = 1; q < K; ++q) { int d = sel_d(f, cen[q]); if (d < bd) bd = d; }
        D[i] = (uint16_t)bd;
    }
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x) {
            const int i = y * IMG_W + x;
            SelC f = FB(i);
            int e = 0;
            if (x > 0) { int d = sel_d(f, FB(i - 1)); if (d > e) e = d; }
            if (x < IMG_W - 1) { int d = sel_d(f, FB(i + 1)); if (d > e) e = d; }
            if (y > 0) { int d = sel_d(f, FB(i - IMG_W)); if (d > e) e = d; }
            if (y < IMG_H - 1) { int d = sel_d(f, FB(i + IMG_W)); if (d > e) e = d; }
            if (e > 255) e = 255;
            int dn = (D[i] * 24) / (spread + 8);
            if (dn > 255) dn = 255;
            cost[i] = (uint8_t)(e > dn ? e : dn);
        }
    sel_bars(img, sw, bars);                               // the bars around a small picture: never
    for (int i = 0; i < SEL_N; ++i) if (bget(bars, i) && !bget(seeds, i)) cost[i] = 255;

    // area of the marks' region at each threshold (union-find, level by level)
    static int area[256];
    {
        int cnt[257] = {0};
        for (int i = 0; i < SEL_N; ++i) ++cnt[(bget(seeds, i) ? 0 : cost[i]) + 1];
        for (int v = 0; v < 256; ++v) cnt[v + 1] += cnt[v];
        for (int i = 0; i < SEL_N; ++i) ord[cnt[bget(seeds, i) ? 0 : cost[i]]++] = i;
        for (int i = 0; i < SEL_N; ++i) par[i] = -1;    // root: −size; "off" pixels: not yet seen
        memset(avail, 0, MASK_BYTES);                   // (avail = on)
        auto find = [&](int i) { while (par[i] >= 0) { if (par[par[i]] >= 0) par[i] = par[par[i]]; i = par[i]; } return i; };
        auto unite = [&](int i, int j) {
            int a = find(i), b = find(j);
            if (a == b) return;
            if (-par[a] < -par[b]) { int t = a; a = b; b = t; }
            par[a] += par[b]; par[b] = a;
        };
        int k = 0;
        for (int v = 0; v < 256; ++v) {
            while (k < SEL_N) {
                const int i = ord[k];
                const int ci = bget(seeds, i) ? 0 : cost[i];
                if (ci != v) break;
                ++k;
                bset(avail, i);
                const int x = i % IMG_W;
                if (x > 0 && bget(avail, i - 1)) unite(i, i - 1);
                if (x < IMG_W - 1 && bget(avail, i + 1)) unite(i, i + 1);
                if (i >= IMG_W && bget(avail, i - IMG_W)) unite(i, i - IMG_W);
                if (i + IMG_W < SEL_N && bget(avail, i + IMG_W)) unite(i, i + IMG_W);
            }
            if (v == 0) for (int i = 0; i < SEL_N; ++i) if (bget(seeds, i)) unite(i, s0);
            area[v] = -par[find(s0)];
        }
    }

    // natural boundaries
    static int lvv[256], lvq[256];
    int nl = 0;
    {
        const int lo = 48 > 4 * nseed ? 48 : 4 * nseed;
        static int q[256];
        for (int v = 0; v < 256; ++v) q[v] = -1;
        for (int v = 10; v < 246; ++v) {
            const int a = area[v];
            if (a >= lo && a <= (6 * SEL_N) / 10) q[v] = (int)(((long)(area[v + 10] - area[v - 10]) * 1024) / a);
        }
        for (int v = 10; v < 246; ++v) {
            if (q[v] < 0 || q[v] > 900) continue;
            if ((q[v - 1] >= 0 && q[v - 1] < q[v]) || (q[v + 1] >= 0 && q[v + 1] <= q[v])) continue;
            const int a = area[v];
            if (nl && (long)a * 10 <= (long)area[lvv[nl - 1]] * 12) { if (q[v] < lvq[nl - 1]) { lvv[nl - 1] = v; lvq[nl - 1] = q[v]; } continue; }
            lvv[nl] = v; lvq[nl] = q[v]; ++nl;
        }
    }
    R.levels = nl;
    if (!nl) { memcpy(out, seeds, MASK_BYTES); return R; }

    auto flood_le = [&](int v, uint8_t* m) {          // marks + what they reach under threshold v
        memset(m, 0, MASK_BYTES);
        bool edge;
        for (int i = 0; i < SEL_N; ++i) {
            if (!bget(seeds, i) || bget(m, i)) continue;
            sel_flood(i % IMG_W, i / IMG_W,
                      [&](int x, int y) { int j = y * IMG_W + x; return !bget(m, j) && (cost[j] <= v || bget(seeds, j)); },
                      [&](int x, int y) { bset(m, y * IMG_W + x); }, [](int, int, int) {}, S, edge);
        }
    };
    const int tau = 40 + spread / 2;
    int best = 0;
    memcpy(prev, seeds, MASK_BYTES);
    for (int i = 0; i < nl; ++i) {                    // climb while each step adds the same stuff
        flood_le(lvv[i], cur);
        H.clear();
        for (int j = 0; j < SEL_N; ++j) if (bget(cur, j) && !bget(prev, j)) H.add(D[j]);
        memcpy(prev, cur, MASK_BYTES);
        if (i == 0) continue;
        if (H.n == 0 || H.kth(H.n / 2) > tau) break;
        const int a0 = area[lvv[i - 1]], a1 = area[lvv[i]];
        if (a1 > 4 * a0 && a1 - a0 > SEL_N / 20) break;
        best = i;
    }
    int ix = best + step;
    if (ix < 0) ix = 0;
    if (ix > nl - 1) ix = nl - 1;
    R.best = best; R.index = ix;
    flood_le(lvv[ix], cur);

    // pieces of the same stuff cut off by something thin
    {
        int nsel = 0;
        int ch[256] = {0};
        H.clear();
        for (int j = 0; j < SEL_N; ++j) if (bget(cur, j)) { ++nsel; ++ch[cost[j]]; H.add(D[j]); }
        int t_piece = 255;
        for (int v = 0, s = 0; v < 256; ++v) { s += ch[v]; if (s > (nsel * 95) / 100) { t_piece = v; break; } }
        int tau_p = H.kth((nsel * 3) / 4);
        if (tau_p > tau) tau_p = tau;
        uint16_t* dist = (uint16_t*)par;               // (par is free now)
        {
            sel_chamfer(cur, false, dist, ones);
            for (int j = 0; j < SEL_N; ++j) {
                if (!bget(cur, j) && cost[j] <= t_piece) bset(avail, j); else bclr(avail, j);
            }
            memset(add, 0, MASK_BYTES);
            for (int j = 0; j < SEL_N; ++j) {
                if (!bget(avail, j) || dist[j] > SEL_NEAR) continue;
                int ns = 0;
                bool edge;
                H.clear();
                int cnt = sel_flood(j % IMG_W, j / IMG_W,
                                    [&](int x, int y) { return bget(avail, y * IMG_W + x); },
                                    [&](int x, int y) { bclr(avail, y * IMG_W + x); H.add(D[y * IMG_W + x]); },
                                    [&](int y, int a, int b) { if (ns < 24000) { spans[ns * 3] = y; spans[ns * 3 + 1] = a; spans[ns * 3 + 2] = b; } ++ns; },
                                    S, edge);
                if (cnt < 12 || cnt > 3 * nsel || ns > 24000) continue;
                if (H.kth(H.n / 2) > tau_p) continue;
                for (int s = 0; s < ns; ++s)
                    for (int x = spans[s * 3 + 1]; x <= spans[s * 3 + 2]; ++x) bset(add, spans[s * 3] * IMG_W + x);
            }
            for (int b = 0; b < MASK_BYTES; ++b) cur[b] |= add[b];
        }
    }
    sel_majority(cur, out);
    for (int b = 0; b < MASK_BYTES; ++b) out[b] |= seeds[b];
    return R;
}

// ---- snap -----------------------------------------------------------------------------------------------
inline void sel_snap(const uint16_t* img, bool sw, const uint8_t* rough, int band, uint8_t* out, void* (*alloc)(size_t)) {
    memcpy(out, rough, MASK_BYTES);
    int nr = 0;
    for (int i = 0; i < SEL_N; ++i) nr += bget(rough, i);
    if (nr < 12) return;
    SelBuf B(alloc);
    uint8_t* E = B.get<uint8_t>(SEL_N);
    uint8_t* ones = B.get<uint8_t>(SEL_N);
    uint16_t* din = B.get<uint16_t>(SEL_N);
    uint16_t* dout = B.get<uint16_t>(SEL_N);
    uint16_t* dfg = B.get<uint16_t>(SEL_N);
    uint16_t* dbg = B.get<uint16_t>(SEL_N);
    uint8_t* cf = B.get<uint8_t>(SEL_N);               // FG front's step cost
    uint8_t* cb = B.get<uint8_t>(SEL_N);               // BG front's step cost (without the prior)
    uint8_t* prior = B.get<uint8_t>(SEL_N);
    uint8_t* fgs = B.get<uint8_t>(MASK_BYTES);
    uint8_t* bgs = B.get<uint8_t>(MASK_BYTES);
    uint8_t* ring = B.get<uint8_t>(MASK_BYTES);
    uint8_t* fgm = B.get<uint8_t>(MASK_BYTES);         // FG model's pixels
    uint8_t* bgm = B.get<uint8_t>(MASK_BYTES);         // BG model's pixels
    uint8_t* sel = B.get<uint8_t>(MASK_BYTES);
    SelC* samp = B.get<SelC>(SEL_NSMAX);
    int* lab = B.get<int>(SEL_NSMAX);
    uint8_t* bars = B.get<uint8_t>(MASK_BYTES);
    uint8_t* rgh = B.get<uint8_t>(MASK_BYTES);
    uint8_t* lbm = B.get<uint8_t>(SEL_N);              // how background-like each pixel's colour is (0..32)
    uint8_t* tmp = B.get<uint8_t>(MASK_BYTES);
    int16_t* stk = B.get<int16_t>((size_t)SEL_STACK * 2);
    if (!B.ok) return;
    SelStack S{stk, SEL_STACK};
    sel_bars(img, sw, bars);
    int nr2 = 0;
    for (int b = 0; b < MASK_BYTES; ++b) { rgh[b] = (uint8_t)(rough[b] & ~bars[b]); nr2 += __builtin_popcount(rgh[b]); }
    rough = rgh;
    memcpy(out, rough, MASK_BYTES);
    if (nr2 < 12) return;
    auto col = [&](int i) { return sel_col(img, sw, i); };
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x) {
            const int i = y * IMG_W + x;
            SelC f = col(i);
            int e = 0;
            if (x > 0) { int d = sel_d(f, col(i - 1)); if (d > e) e = d; }
            if (x < IMG_W - 1) { int d = sel_d(f, col(i + 1)); if (d > e) e = d; }
            if (y > 0) { int d = sel_d(f, col(i - IMG_W)); if (d > e) e = d; }
            if (y < IMG_H - 1) { int d = sel_d(f, col(i + IMG_W)); if (d > e) e = d; }
            E[i] = (uint8_t)(e > 255 ? 255 : e);
        }
    sel_chamfer(rough, true, din, ones);               // distance in from the shape's edge
    sel_chamfer(rough, false, dout, ones);             // distance out
    int mx = 0;
    for (int i = 0; i < SEL_N; ++i) if (din[i] > mx) mx = din[i];
    const int b_in = band > mx / 2 ? band : mx / 2;
    const int b_out = band / 2 > 3 ? band / 2 : 3;
    int nf = 0;
    memset(fgs, 0, MASK_BYTES); memset(bgs, 0, MASK_BYTES); memset(ring, 0, MASK_BYTES);
    for (int i = 0; i < SEL_N; ++i) if (din[i] > b_in) { bset(fgs, i); ++nf; }
    if (nf < 20) { nf = 0; for (int i = 0; i < SEL_N; ++i) if (din[i] > b_in / 2) { bset(fgs, i); ++nf; } }
    int nb = 0;
    for (int i = 0; i < SEL_N; ++i)
        if (dout[i] > b_out || bget(bars, i)) { bset(bgs, i); if (dout[i] <= 3 * b_in && !bget(bars, i)) { bset(ring, i); ++nb; } }
    if (!nb) for (int i = 0; i < SEL_N; ++i) if (bget(bgs, i)) { bset(ring, i); ++nb; }
    if (!nf || !nb) return;
    for (int i = 0; i < SEL_N; ++i)
        prior[i] = bget(rough, i) ? (uint8_t)(((din[i] < b_in ? din[i] : b_in) * 12) / (b_in > 1 ? b_in : 1)) : 0;
    memcpy(fgm, rough, MASK_BYTES);
    memcpy(bgm, ring, MASK_BYTES);
    memcpy(sel, rough, MASK_BYTES);
    for (int it = 0; it < 2; ++it) {
        SelC cfc[SEL_KMAX], cbc[SEL_KMAX];
        int sp, kf, kb;
        int n = sel_take([&](int i) { return bget(fgm, i); }, col, samp);
        kf = sel_kmeans(samp, n, cfc, sp, lab);
        n = sel_take([&](int i) { return bget(bgm, i); }, col, samp);
        kb = sel_kmeans(samp, n, cbc, sp, lab);
        for (int i = 0; i < SEL_N; ++i) {
            SelC f = col(i);
            int df = sel_d(f, cfc[0]), db = sel_d(f, cbc[0]);
            for (int q = 1; q < kf; ++q) { int d = sel_d(f, cfc[q]); if (d < df) df = d; }
            for (int q = 1; q < kb; ++q) { int d = sel_d(f, cbc[q]); if (d < db) db = d; }
            const int ce = 1 + (E[i] >> 3);
            lbm[i] = (uint8_t)((32 * df) / (df + db + 1));
            cf[i] = (uint8_t)(ce + (32 * df) / (df + db + 1));
            cb[i] = (uint8_t)(ce + (32 * db) / (df + db + 1) + prior[i]);
            dfg[i] = bget(fgs, i) ? 0 : SEL_INF;
            dbg[i] = (bget(bgs, i) && !bget(fgs, i)) ? 0 : SEL_INF;
        }
        sel_geodesic(dfg, cf, 3);
        sel_geodesic(dbg, cb, 3);
        bool any = false;
        memset(tmp, 0, MASK_BYTES);
        for (int i = 0; i < SEL_N; ++i) {
            const bool unknown = !bget(fgs, i) && !bget(bgs, i);
            if ((unknown && dfg[i] < dbg[i] && dfg[i] < SEL_INF && lbm[i] < SEL_SNAP_BG) || bget(fgs, i)) bset(tmp, i);
        }
        memset(sel, 0, MASK_BYTES);                    // only what's joined to the object's core
        for (int i = 0; i < SEL_N; ++i) {
            if (!bget(fgs, i) || !bget(tmp, i) || bget(sel, i)) continue;
            bool edge;
            sel_flood(i % IMG_W, i / IMG_W, [&](int x, int y) { return bget(tmp, y * IMG_W + x) && !bget(sel, y * IMG_W + x); },
                      [&](int x, int y) { bset(sel, y * IMG_W + x); }, [](int, int, int) {}, S, edge);
            any = true;
        }
        memcpy(fgm, sel, MASK_BYTES);
        for (int b = 0; b < MASK_BYTES; ++b) bgm[b] = (uint8_t)(ring[b] | (rough[b] & ~sel[b]));
        if (!any) break;
    }
    sel_majority(sel, out);
    for (int b = 0; b < MASK_BYTES; ++b) out[b] = (uint8_t)((out[b] | fgs[b]) & ~bgs[b]);
}

// ---- apply ---------------------------------------------------------------------------------------------
// trim: 1 = an added area keeps 1 px off what isn't in the mask (water off the bank), 0 = exact,
// −1 = an added area also takes its edge pixels (a depth plane carries its own rim, no ghost halo).
// Removing with `seeds` (what was tapped / stroked): only the selected pieces under them go - holding on
// a stray blob takes that blob off, not everything of a similar colour nearby.
inline int sel_apply(uint8_t* mask, const uint8_t* sel_in, bool add, int trim, void* (*alloc)(size_t),
                     const uint8_t* seeds = nullptr) {
    SelBuf B(alloc);
    uint8_t* t = B.get<uint8_t>(MASK_BYTES);
    uint8_t* avail = B.get<uint8_t>(MASK_BYTES);
    uint8_t* t2 = B.get<uint8_t>(MASK_BYTES);
    int16_t* stk = B.get<int16_t>((size_t)SEL_STACK * 2);
    int32_t* spans = B.get<int32_t>((size_t)(SEL_HOLE + 8) * 3);
    if (!B.ok) return 0;
    SelStack S{stk, SEL_STACK};
    memcpy(t, sel_in, MASK_BYTES);
    if (!add && seeds) {
        bool hit = false;
        for (int b = 0; b < MASK_BYTES && !hit; ++b) hit = (t[b] & mask[b] & seeds[b]) != 0;
        if (hit) {
            for (int b = 0; b < MASK_BYTES; ++b) { avail[b] = t[b] & mask[b]; t2[b] = 0; }
            for (int i = 0; i < IMG_W * IMG_H; ++i) {
                if (!bget(seeds, i) || !bget(avail, i)) continue;
                bool edge;
                sel_flood(i % IMG_W, i / IMG_W, [&](int xx, int yy) { return bget(avail, yy * IMG_W + xx); },
                          [&](int xx, int yy) { bclr(avail, yy * IMG_W + xx); bset(t2, yy * IMG_W + xx); },
                          [&](int, int, int) {}, S, edge);
            }
            memcpy(t, t2, MASK_BYTES);
        }
    }
    for (int b = 0; b < MASK_BYTES; ++b) avail[b] = (uint8_t)~t[b];
    for (int y = 1; y < IMG_H - 1; ++y)                // small enclosed gaps
        for (int x = 1; x < IMG_W - 1; ++x) {
            const int i = y * IMG_W + x;
            if (!bget(avail, i) || !(bget(t, i - 1) || bget(t, i - IMG_W))) continue;
            int ns = 0;
            bool edge;
            int n = sel_flood(x, y, [&](int xx, int yy) { return bget(avail, yy * IMG_W + xx); },
                              [&](int xx, int yy) { bclr(avail, yy * IMG_W + xx); },
                              [&](int yy, int a, int b) { if (ns < SEL_HOLE + 8) { spans[ns * 3] = yy; spans[ns * 3 + 1] = a; spans[ns * 3 + 2] = b; } ++ns; },
                              S, edge);
            if (n <= SEL_HOLE && !edge)
                for (int s = 0; s < ns; ++s)
                    for (int xx = spans[s * 3 + 1]; xx <= spans[s * 3 + 2]; ++xx) bset(t, spans[s * 3] * IMG_W + xx);
        }
    memcpy(t2, t, MASK_BYTES);
    if (trim < 0 && add) {
        for (int y = 0; y < IMG_H; ++y)
            for (int x = 0; x < IMG_W; ++x)
                if (mask_get(t, x - 1, y) || mask_get(t, x + 1, y) || mask_get(t, x, y - 1) || mask_get(t, x, y + 1)) bset(t2, y * IMG_W + x);
    } else if (trim) {
        for (int y = 0; y < IMG_H; ++y)
            for (int x = 0; x < IMG_W; ++x) {
                const int i = y * IMG_W + x;
                if (add) {
                    if (!bget(t, i)) continue;
                    auto in = [&](int xx, int yy) {
                        xx = xx < 0 ? 0 : (xx >= IMG_W ? IMG_W - 1 : xx);
                        yy = yy < 0 ? 0 : (yy >= IMG_H ? IMG_H - 1 : yy);
                        return mask_get(t, xx, yy) || mask_get(mask, xx, yy);
                    };
                    if (!(in(x - 1, y) && in(x + 1, y) && in(x, y - 1) && in(x, y + 1))) bclr(t2, i);
                } else if (mask_get(t, x - 1, y) || mask_get(t, x + 1, y) || mask_get(t, x, y - 1) || mask_get(t, x, y + 1)) {
                    bset(t2, i);
                }
            }
    }
    int changed = 0;
    for (int b = 0; b < MASK_BYTES; ++b) {
        uint8_t nv = add ? (uint8_t)(mask[b] | t2[b]) : (uint8_t)(mask[b] & ~t2[b]);
        if (nv != mask[b]) { changed += __builtin_popcount((unsigned)(nv ^ mask[b])); mask[b] = nv; }
    }
    return changed;
}

}  // namespace jig
