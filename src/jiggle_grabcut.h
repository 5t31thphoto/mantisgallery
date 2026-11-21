#pragma once
// GrabCut on the Core2 (Rother, Kolmogorov & Blake, "GrabCut", SIGGRAPH 2004): a rough shape around
// something → the thing. Inside the shape is "probably it", outside (and the bars around a small
// picture) is background. Two 5-component Gaussian colour mixtures (it / background) and a global
// min-cut with edge-aware neighbour links, iterated: the same method as the lab's OpenCV GrabCut.
//
// To fit memory and time the cut runs at half resolution (160×120: ~0.8 MB of work memory); the edge
// is then settled at full resolution against the colour models, inside a 2 px band.
// Max-flow: Boykov–Kolmogorov (search trees from both terminals, reused between augmentations) on the
// 4-connected pixel grid, with integer capacities.
#include <algorithm>
#include "jiggle_select.h"

namespace jig {

constexpr int GC_K = 5;                 // components per colour model
constexpr int GC_ITERS = 4;
constexpr float GC_GAMMA = 50.0f;       // edge-link strength (the paper's / OpenCV's)
constexpr int GC_SCALE = 64;            // capacities ×64 → integers

// ---- Gaussian mixture ----------------------------------------------------------------------------
struct GcGauss { float w, m[3], inv[9], logn; };   // logn = log(w) − ½ log det
struct GcGmm {
    GcGauss g[GC_K]; int k = 0;
    // −log(Σ w N) up to a shared constant
    float nll(const float* z) const {
        float best = -1e30f, s = 0;
        float lv[GC_K];
        for (int i = 0; i < k; ++i) {
            const GcGauss& c = g[i];
            const float d0 = z[0] - c.m[0], d1 = z[1] - c.m[1], d2 = z[2] - c.m[2];
            const float q = d0 * (c.inv[0] * d0 + c.inv[1] * d1 + c.inv[2] * d2) + d1 * (c.inv[3] * d0 + c.inv[4] * d1 + c.inv[5] * d2) +
                            d2 * (c.inv[6] * d0 + c.inv[7] * d1 + c.inv[8] * d2);
            lv[i] = c.logn - 0.5f * q;
            if (lv[i] > best) best = lv[i];
        }
        for (int i = 0; i < k; ++i) s += expf(lv[i] - best);
        return -(best + logf(s));
    }
    int which(const float* z) const {               // the component that explains z best
        int bi = 0; float best = -1e30f;
        for (int i = 0; i < k; ++i) {
            const GcGauss& c = g[i];
            const float d0 = z[0] - c.m[0], d1 = z[1] - c.m[1], d2 = z[2] - c.m[2];
            const float q = d0 * (c.inv[0] * d0 + c.inv[1] * d1 + c.inv[2] * d2) + d1 * (c.inv[3] * d0 + c.inv[4] * d1 + c.inv[5] * d2) +
                            d2 * (c.inv[6] * d0 + c.inv[7] * d1 + c.inv[8] * d2);
            const float l = c.logn - 0.5f * q;
            if (l > best) { best = l; bi = i; }
        }
        return bi;
    }
};
// Learn from samples z (n×3) with component labels lab.
inline void gc_learn(GcGmm& G, const float* z, const uint8_t* lab, int n) {
    double s[GC_K][3] = {{0}}, ss[GC_K][9] = {{0}};
    int cnt[GC_K] = {0};
    for (int i = 0; i < n; ++i) {
        const int c = lab[i]; const float* v = z + i * 3;
        ++cnt[c];
        for (int a = 0; a < 3; ++a) { s[c][a] += v[a]; for (int b = 0; b < 3; ++b) ss[c][a * 3 + b] += (double)v[a] * v[b]; }
    }
    G.k = 0;
    for (int c = 0; c < GC_K; ++c) {
        if (!cnt[c]) continue;
        GcGauss& g = G.g[G.k++];
        double m[3], C[9];
        for (int a = 0; a < 3; ++a) m[a] = s[c][a] / cnt[c];
        for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) C[a * 3 + b] = ss[c][a * 3 + b] / cnt[c] - m[a] * m[b];
        for (int a = 0; a < 3; ++a) C[a * 4] += 0.01 + 4.0;    // (a little extra spread: RGB565 steps of ~4–8)
        const double det = C[0] * (C[4] * C[8] - C[5] * C[7]) - C[1] * (C[3] * C[8] - C[5] * C[6]) + C[2] * (C[3] * C[7] - C[4] * C[6]);
        const double id = det > 1e-12 ? 1.0 / det : 0.0;
        double I[9] = {(C[4] * C[8] - C[5] * C[7]) * id, -(C[1] * C[8] - C[2] * C[7]) * id, (C[1] * C[5] - C[2] * C[4]) * id,
                       -(C[3] * C[8] - C[5] * C[6]) * id, (C[0] * C[8] - C[2] * C[6]) * id, -(C[0] * C[5] - C[2] * C[3]) * id,
                       (C[3] * C[7] - C[4] * C[6]) * id, -(C[0] * C[7] - C[1] * C[6]) * id, (C[0] * C[4] - C[1] * C[3]) * id};
        g.w = (float)cnt[c] / n;
        for (int a = 0; a < 3; ++a) g.m[a] = (float)m[a];
        for (int a = 0; a < 9; ++a) g.inv[a] = (float)I[a];
        g.logn = logf(g.w) - 0.5f * (float)log(det > 1e-12 ? det : 1e-12);
    }
}
// k-means start (farthest-point seeds, 8 rounds) → component labels
inline void gc_kmeans(const float* z, int n, uint8_t* lab) {
    if (n <= 0) return;
    float c[GC_K][3];
    const int k = n < GC_K ? n : GC_K;
    double s0[3] = {0, 0, 0};
    for (int i = 0; i < n; ++i) for (int a = 0; a < 3; ++a) s0[a] += z[i * 3 + a];
    for (int a = 0; a < 3; ++a) c[0][a] = (float)(s0[a] / n);
    for (int j = 1; j < k; ++j) {
        int bi = 0; float bd = -1;
        for (int i = 0; i < n; ++i) {
            float dm = 1e30f;
            for (int q = 0; q < j; ++q) {
                const float d0 = z[i * 3] - c[q][0], d1 = z[i * 3 + 1] - c[q][1], d2 = z[i * 3 + 2] - c[q][2];
                const float d = d0 * d0 + d1 * d1 + d2 * d2;
                if (d < dm) dm = d;
            }
            if (dm > bd) { bd = dm; bi = i; }
        }
        for (int a = 0; a < 3; ++a) c[j][a] = z[bi * 3 + a];
    }
    for (int it = 0; it < 8; ++it) {
        double sum[GC_K][3] = {{0}}; int cnt[GC_K] = {0};
        for (int i = 0; i < n; ++i) {
            int bq = 0; float bd = 1e30f;
            for (int q = 0; q < k; ++q) {
                const float d0 = z[i * 3] - c[q][0], d1 = z[i * 3 + 1] - c[q][1], d2 = z[i * 3 + 2] - c[q][2];
                const float d = d0 * d0 + d1 * d1 + d2 * d2;
                if (d < bd) { bd = d; bq = q; }
            }
            lab[i] = (uint8_t)bq; ++cnt[bq];
            for (int a = 0; a < 3; ++a) sum[bq][a] += z[i * 3 + a];
        }
        for (int q = 0; q < k; ++q) if (cnt[q]) for (int a = 0; a < 3; ++a) c[q][a] = (float)(sum[q][a] / cnt[q]);
    }
}

// ---- max-flow on a 4-connected grid (Boykov–Kolmogorov) -------------------------------------------
// Directions 0 left, 1 right, 2 up, 3 down; opp(d) = d ^ 1. Node capacities: tr > 0 from the source
// (object side), tr < 0 to the sink. After run(): side(i) true = object.
struct GcFlow {
    int W = 0, H = 0, N = 0;
    int32_t* rc = nullptr;      // residual capacity of arc (i → neighbour d): rc[i*4 + d]
    int32_t* tr = nullptr;
    int8_t*  par = nullptr;     // −1 none, 4 terminal, 5 orphan, 0..3 direction to the parent
    uint8_t* tree = nullptr;    // 0 free, 1 source tree, 2 sink tree
    int32_t* dist = nullptr;
    int32_t* ts = nullptr;
    int32_t* q = nullptr;       // active FIFO (ring)
    uint8_t* inq = nullptr;
    int32_t* orph = nullptr;
    int qh = 0, qt = 0, no = 0, T = 0;

    int nb(int i, int d) const {
        const int x = i % W;
        switch (d) {
        case 0: return x > 0 ? i - 1 : -1;
        case 1: return x < W - 1 ? i + 1 : -1;
        case 2: return i >= W ? i - W : -1;
        default: return i + W < N ? i + W : -1;
        }
    }
    void push(int i) { if (!inq[i]) { inq[i] = 1; q[qt] = i; qt = (qt + 1) % (N + 1); } }
    int pop() { while (qh != qt) { int i = q[qh]; qh = (qh + 1) % (N + 1); inq[i] = 0; if (tree[i]) return i; } return -1; }

    bool alloc(int w, int h, void* (*al)(size_t)) {
        W = w; H = h; N = w * h;
        rc = (int32_t*)al((size_t)N * 16); tr = (int32_t*)al((size_t)N * 4); par = (int8_t*)al(N); tree = (uint8_t*)al(N);
        dist = (int32_t*)al((size_t)N * 4); ts = (int32_t*)al((size_t)N * 4); q = (int32_t*)al((size_t)(N + 1) * 4);
        inq = (uint8_t*)al(N); orph = (int32_t*)al((size_t)N * 4);
        return rc && tr && par && tree && dist && ts && q && inq && orph;
    }
    void release() { for (void* p : {(void*)rc, (void*)tr, (void*)par, (void*)tree, (void*)dist, (void*)ts, (void*)q, (void*)inq, (void*)orph}) if (p) free(p); }

    // origin check for adoption: does j's path reach its terminal? (timestamps cache the answer)
    bool origin(int j, int& d_out) {
        int k = j, d = 0;
        while (true) {
            if (ts[k] == T) { d += dist[k]; break; }
            const int p = par[k];
            if (p == 4) { ts[k] = T; dist[k] = 1; d += 1; break; }
            if (p < 0 || p == 5) return false;
            ++d; k = nb(k, p);
        }
        // mark the path
        int dd = d;
        for (k = j; ts[k] != T; k = nb(k, par[k])) { ts[k] = T; dist[k] = dd--; }
        d_out = d;
        return true;
    }

    void run() {
        qh = qt = 0; no = 0; T = 1;
        for (int i = 0; i < N; ++i) {
            inq[i] = 0; ts[i] = 0; dist[i] = 0;
            if (tr[i] > 0) { tree[i] = 1; par[i] = 4; ts[i] = T; dist[i] = 1; push(i); }
            else if (tr[i] < 0) { tree[i] = 2; par[i] = 4; ts[i] = T; dist[i] = 1; push(i); }
            else { tree[i] = 0; par[i] = -1; }
        }
        int cur = -1;
        while (true) {
            // grow
            int s = -1, t = -1, sd = -1;
            if (cur >= 0 && tree[cur] == 0) cur = -1;
            while (s < 0) {
                if (cur < 0) { cur = pop(); if (cur < 0) return; }
                const int i = cur;
                for (int d = 0; d < 4 && s < 0; ++d) {
                    const int j = nb(i, d);
                    if (j < 0) continue;
                    const int32_t cap = tree[i] == 1 ? rc[i * 4 + d] : rc[j * 4 + (d ^ 1)];
                    if (cap <= 0) continue;
                    if (!tree[j]) { tree[j] = tree[i]; par[j] = (int8_t)(d ^ 1); ts[j] = ts[i]; dist[j] = dist[i] + 1; push(j); }
                    else if (tree[j] != tree[i]) { if (tree[i] == 1) { s = i; t = j; sd = d; } else { s = j; t = i; sd = d ^ 1; } }
                    else if (ts[j] <= ts[i] && dist[j] > dist[i]) { par[j] = (int8_t)(d ^ 1); ts[j] = ts[i]; dist[j] = dist[i] + 1; }
                }
                if (s < 0) cur = -1;                     // nothing more from this node
            }
            // augment along source…s → t…sink
            int32_t f = rc[s * 4 + sd];
            for (int k = s; par[k] != 4; ) { const int p = par[k], pk = nb(k, p); const int32_t c = rc[pk * 4 + (p ^ 1)]; if (c < f) f = c; k = pk; }
            { int k = s; while (par[k] != 4) k = nb(k, par[k]); if (tr[k] < f) f = tr[k]; }
            for (int k = t; par[k] != 4; ) { const int p = par[k]; const int32_t c = rc[k * 4 + p]; if (c < f) f = c; k = nb(k, p); }
            { int k = t; while (par[k] != 4) k = nb(k, par[k]); if (-tr[k] < f) f = -tr[k]; }
            rc[s * 4 + sd] -= f; rc[t * 4 + (sd ^ 1)] += f;
            for (int k = s; ; ) {
                const int p = par[k];
                if (p == 4) { tr[k] -= f; if (tr[k] == 0) { par[k] = 5; orph[no++] = k; } break; }
                const int pk = nb(k, p);
                rc[pk * 4 + (p ^ 1)] -= f; rc[k * 4 + p] += f;
                if (rc[pk * 4 + (p ^ 1)] == 0) { par[k] = 5; orph[no++] = k; }
                k = pk;
            }
            for (int k = t; ; ) {
                const int p = par[k];
                if (p == 4) { tr[k] += f; if (tr[k] == 0) { par[k] = 5; orph[no++] = k; } break; }
                const int pk = nb(k, p);
                rc[k * 4 + p] -= f; rc[pk * 4 + (p ^ 1)] += f;
                if (rc[k * 4 + p] == 0) { par[k] = 5; orph[no++] = k; }
                k = pk;
            }
            // adopt
            ++T;
            while (no > 0) {
                const int o = orph[--no];
                const uint8_t tt = tree[o];
                int bd = -1, bdist = 1 << 30;
                for (int d = 0; d < 4; ++d) {
                    const int j = nb(o, d);
                    if (j < 0 || tree[j] != tt) continue;
                    const int32_t cap = tt == 1 ? rc[j * 4 + (d ^ 1)] : rc[o * 4 + d];
                    if (cap <= 0) continue;
                    int dj;
                    if (origin(j, dj) && dj < bdist) { bdist = dj; bd = d; }
                }
                if (bd >= 0) { par[o] = (int8_t)bd; ts[o] = T; dist[o] = bdist + 1; continue; }
                for (int d = 0; d < 4; ++d) {           // no parent: it leaves the tree
                    const int j = nb(o, d);
                    if (j < 0 || tree[j] != tt) continue;
                    const int32_t cap = tt == 1 ? rc[j * 4 + (d ^ 1)] : rc[o * 4 + d];
                    if (cap > 0) push(j);
                    if (par[j] >= 0 && par[j] < 4 && nb(j, par[j]) == o) { par[j] = 5; orph[no++] = j; }
                }
                tree[o] = 0; par[o] = -1;
            }
        }
    }
    bool side(int i) const { return tree[i] == 1; }        // reachable from the source in the residual
};

// ---- GrabCut ----------------------------------------------------------------------------------------
// rough: the shape drawn (full res). out: the thing inside it. Returns false if out of memory.
inline bool sel_grabcut(const uint16_t* img, bool sw, const uint8_t* rough, uint8_t* out, void* (*al)(size_t)) {
    memcpy(out, rough, MASK_BYTES);
    const int w = IMG_W / 2, h = IMG_H / 2, n = w * h;
    SelBuf B(al);
    float* z = B.get<float>((size_t)n * 3);              // half-res colour
    uint8_t* lab = B.get<uint8_t>(n);                    // trimap: 0 background (fixed), 1 maybe-object
    uint8_t* seg = B.get<uint8_t>(n);                    // current answer
    uint8_t* comp = B.get<uint8_t>(n);
    float* zs = B.get<float>((size_t)n * 3);
    uint8_t* bars = B.get<uint8_t>(MASK_BYTES);
    int32_t* wr = B.get<int32_t>((size_t)n * 4);          // edge links (each direction)
    if (!B.ok) return false;
    auto rgb = [&](int i, float* o) {
        uint16_t v = img[i]; if (sw) v = (uint16_t)((v >> 8) | (v << 8));
        o[0] = ((v >> 11) & 31) * (255.0f / 31); o[1] = ((v >> 5) & 63) * (255.0f / 63); o[2] = (v & 31) * (255.0f / 31);
    };
    sel_bars(img, sw, bars);
    int nin = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float a[3], s[3] = {0, 0, 0};
            int ins = 0;
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                    const int fi = (2 * y + dy) * IMG_W + 2 * x + dx;
                    rgb(fi, a); s[0] += a[0]; s[1] += a[1]; s[2] += a[2];
                    if (bget(rough, fi) && !bget(bars, fi)) ++ins;
                }
            const int i = y * w + x;
            z[i * 3] = s[0] * 0.25f; z[i * 3 + 1] = s[1] * 0.25f; z[i * 3 + 2] = s[2] * 0.25f;
            lab[i] = ins >= 2 ? 1 : 0;
            seg[i] = lab[i];
            nin += lab[i];
        }
    if (nin < 8 || nin == n) return true;
    // β from the mean colour step between neighbours; edge links
    double acc = 0; long np = 0;
    for (int i = 0; i < n; ++i) {
        const int x = i % w;
        if (x < w - 1) { float d0 = z[i * 3] - z[i * 3 + 3], d1 = z[i * 3 + 1] - z[i * 3 + 4], d2 = z[i * 3 + 2] - z[i * 3 + 5]; acc += d0 * d0 + d1 * d1 + d2 * d2; ++np; }
        if (i + w < n) { float d0 = z[i * 3] - z[(i + w) * 3], d1 = z[i * 3 + 1] - z[(i + w) * 3 + 1], d2 = z[i * 3 + 2] - z[(i + w) * 3 + 2]; acc += d0 * d0 + d1 * d1 + d2 * d2; ++np; }
    }
    const float beta = acc > 0 ? (float)(np / (2.0 * acc)) : 0.0f;
    GcFlow F;
    if (!F.alloc(w, h, al)) { F.release(); return false; }
    for (int i = 0; i < n; ++i)
        for (int d = 0; d < 4; ++d) {
            const int j = F.nb(i, d);
            int32_t c = 0;
            if (j >= 0) {
                const float d0 = z[i * 3] - z[j * 3], d1 = z[i * 3 + 1] - z[j * 3 + 1], d2 = z[i * 3 + 2] - z[j * 3 + 2];
                c = (int32_t)lroundf(GC_GAMMA * expf(-beta * (d0 * d0 + d1 * d1 + d2 * d2)) * GC_SCALE);
            }
            wr[i * 4 + d] = c;
        }
    const int32_t BIG = (int32_t)(9.0f * GC_GAMMA * 4 * GC_SCALE);
    GcGmm fg, bg;
    for (int it = 0; it < GC_ITERS; ++it) {
        // colour models from the current answer (k-means start on the first round)
        int nf = 0, nb = 0;
        for (int i = 0; i < n; ++i) if (seg[i]) { memcpy(zs + nf * 3, z + i * 3, 12); ++nf; }
        if (!nf) break;
        if (it == 0) gc_kmeans(zs, nf, comp); else for (int i = 0; i < nf; ++i) comp[i] = (uint8_t)fg.which(zs + i * 3);
        gc_learn(fg, zs, comp, nf);
        for (int i = 0; i < n; ++i) if (!seg[i]) { memcpy(zs + nb * 3, z + i * 3, 12); ++nb; }
        if (!nb) break;
        if (it == 0) gc_kmeans(zs, nb, comp); else for (int i = 0; i < nb; ++i) comp[i] = (uint8_t)bg.which(zs + i * 3);
        gc_learn(bg, zs, comp, nb);
        // graph: object side = source
        memcpy(F.rc, wr, (size_t)n * 16);
        for (int i = 0; i < n; ++i) {
            if (!lab[i]) { F.tr[i] = -BIG; continue; }
            const float df = fg.nll(z + i * 3), db = bg.nll(z + i * 3);   // costs of calling it object / background
            float m = df < db ? df : db;
            F.tr[i] = (int32_t)lroundf((db - m) * GC_SCALE) - (int32_t)lroundf((df - m) * GC_SCALE);
        }
        F.run();
        for (int i = 0; i < n; ++i) seg[i] = lab[i] && F.side(i) ? 1 : 0;
    }
    F.release();
    // full resolution: settle the edge (2 px band) against the colour models, keep the shape's limits
    memset(out, 0, MASK_BYTES);
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x)
            if (seg[(y >> 1) * w + (x >> 1)]) bset(out, y * IMG_W + x);
    uint8_t* band = bars;                                 // (reuse: bars no longer needed after this)
    uint8_t* fin = B.get<uint8_t>(MASK_BYTES);
    if (fin && fg.k && bg.k) {
        memcpy(fin, out, MASK_BYTES);
        for (int y = 0; y < IMG_H; ++y)
            for (int x = 0; x < IMG_W; ++x) {
                const bool v = mask_get(out, x, y);
                bool edge = false;
                for (int dy = -2; dy <= 2 && !edge; ++dy)
                    for (int dx = -2; dx <= 2; ++dx) if (mask_get(out, x + dx, y + dy) != v) { edge = true; break; }
                if (!edge) continue;
                const int i = y * IMG_W + x;
                if (!bget(rough, i)) { bclr(fin, i); continue; }
                float a[3]; rgb(i, a);
                if (fg.nll(a) < bg.nll(a)) bset(fin, i); else bclr(fin, i);
            }
        sel_majority(fin, out);
        (void)band;
    }
    int kept = 0, area = 0;
    for (int b = 0; b < MASK_BYTES; ++b) { out[b] &= rough[b]; kept += __builtin_popcount(out[b]); area += __builtin_popcount(rough[b]); }
    // Collapsed: nothing inside the shape stands apart from what's around it (a strip drawn
    // across one continuous surface). The drawn shape is then the intent - let the caller
    // fit its outline to the nearby edges instead.
    return kept * 5 >= area;
}

}  // namespace jig
