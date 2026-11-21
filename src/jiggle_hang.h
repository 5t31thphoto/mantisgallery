#pragma once
// Hanging things: hair, cloth (skirts, scarves, flags) and charms (earrings, pendants).
//
// The way the 2D animation tools do it (Live2D's physics groups, Spine's physics constraints): chains
// of pendulums moved by how the "camera" moves, each link pulled back toward its pose in the artwork
// and slowed, with the picture bound to the chains as a mesh.
//   * The user selects the thing (the same tools as water and depth) and marks where it hangs from:
//     a point (an earring's hook) or a line (a ponytail's tie, a skirt's waistband).
//   * A mesh of `nc` chains runs from there down through the selection (it hangs toward its own
//     middle); rows at / above the line are pinned and every chain hinges at its own root.
//   * Physics at a fixed 120 Hz (Verlet). A hanging thing is a pendulum: pushed off "down" by d,
//     gravity pulls it back by g·d / (its distance from the pivot) - so each node is pulled to its pose
//     in the photo with g / its distance, and pushed by the change in "down" as the Core2 tilts (to
//     first order a tilt by a swings it by a), by jolts (it lags), a breeze and taps. Links keep their
//     lengths; side links keep the width, and (between free rows) the shear. A charm is one stiff body
//     turning about its hook.
//   * Drawing: what's behind is filled once from the surroundings (the selection and three rings past
//     it, so no outline of it is left behind); each frame the mesh's triangles are drawn from their
//     pose in the photo to where they are now. The rings come along as a soft fringe - only while
//     it moves: at rest the photo is exact.
// tools/mg_hang.py is the same, step for step.
#include "jiggle_parallax.h"

namespace jig {

constexpr int HANG_MAX = 3;
constexpr int HC_MAX = 5;                   // chains side by side
constexpr int HR_MAX = 8;                   // rows (≤ 2 pinned + 6 free)
constexpr int HN_MAX = HC_MAX * HR_MAX;
constexpr int HL_MAX = 4 * HN_MAX;          // links
constexpr float HANG_DT = 1.0f / 120.0f;
constexpr int HANG_ITERS = 4;
constexpr float HANG_REACH_DEG = 55.0f;     // its drawing region: as far as it can swing
constexpr int HANG_FILL_ITERS = 80;         // what's behind: smoothing passes…
constexpr float HANG_FILL_DETAIL = 0.6f;    // …and how much of the surroundings' grain goes on top…
constexpr float HANG_FILL_GRAIN = 9.0f;     // …at most (8-bit levels: grain, not stray dark / bright bits)

enum HangKind : uint8_t { HANG_HAIR = 0, HANG_CLOTH, HANG_CHARM, HANG_KIND_COUNT };
enum HParam : uint8_t { HP_SWING = 0, HP_BREEZE, HP_COUNT };

struct HangKindInfo {
    const char* key; const char* label;
    int nc, nr;                             // chains, free rows
    float gain, c, g, j, shear, wind;       // follows the new down (1: freely); damping 1/s; gravity, jolt lag px/s²; shear; breeze px/s²
};
inline const HangKindInfo& hang_kind_info(uint8_t k) {
    static const HangKindInfo t[HANG_KIND_COUNT] = {
        {"hair", "Hair", 3, 6, 0.45f, 2.2f, 1500.0f, 900.0f, 0.5f, 110.0f},
        {"cloth", "Cloth", 5, 4, 0.25f, 3.0f, 1100.0f, 600.0f, 0.2f, 70.0f},
        {"charm", "Charm", 2, 1, 0.8f, 1.0f, 1800.0f, 1100.0f, 1.0f, 20.0f},
    };
    return t[k < HANG_KIND_COUNT ? k : 0];
}
struct HParamInfo { const char* key; const char* label; float lo, hi, dflt; };
inline const HParamInfo& hparam_info(uint8_t p) {
    static const HParamInfo t[HP_COUNT] = {
        {"swing", "Swing (how much it moves)", 0.0f, 2.5f, 1.0f},
        {"breeze", "Breeze while still", 0.0f, 1.5f, 0.3f},
    };
    return t[p < HP_COUNT ? p : 0];
}

struct HLink { uint8_t a, b; float len, stiff; };

struct Hang {
    // ---- definition (saved)
    uint8_t kind = HANG_HAIR;
    int16_t ax = 0, ay = 0, bx = 0, by = 0;   // where it hangs from (A = B: a point)
    float   prm[HP_COUNT];
    uint8_t* mask = nullptr;
    // ---- runtime
    bool  ready = false, awake = false, drawn = false;
    int   nc = 0, nrows = 0, npin = 0, nl = 0;
    float e1[2], e2[2], R[2];
    float rest[HN_MAX][2], x[HN_MAX][2], xp[HN_MAX][2];
    float kn[HR_MAX];
    HLink link[HL_MAX];
    float rc[2] = {0, 0}, Lc = 4, kc = 0, th = 0, thp = 0, tlen = 4;
    float t = 0, acc = 0, fade = 0;
    int   x0 = 0, y0 = 0, x1 = 0, y1 = 0;    // drawing region
    uint16_t* tex = nullptr;                 // region: the photo (native RGB565)
    uint16_t* texd = nullptr;                // region: what it carries when it moves (its edge matted)
    uint16_t* fill = nullptr;                // region: what's behind it
    uint8_t*  alpha = nullptr;               // region: what it carries (0 = not it)
    Hang() { for (int i = 0; i < HP_COUNT; ++i) prm[i] = hparam_info((uint8_t)i).dflt; }
};

inline void hang_free_runtime(Hang& h) {
    if (h.tex) free(h.tex);
    if (h.texd) free(h.texd);
    h.texd = nullptr;
    if (h.fill) free(h.fill);
    if (h.alpha) free(h.alpha);
    h.tex = h.fill = nullptr; h.alpha = nullptr; h.ready = false;
}
inline void hang_free(Hang& h) {
    hang_free_runtime(h);
    if (h.mask) free(h.mask);
    h.mask = nullptr;
}
inline bool hang_ensure_mask(Hang& h, void* (*alloc)(size_t)) {
    if (h.mask) return true;
    h.mask = (uint8_t*)alloc(MASK_BYTES);
    if (h.mask) memset(h.mask, 0, MASK_BYTES);
    return h.mask != nullptr;
}
inline void hang_copy_def(Hang& dst, const Hang& src, void* (*alloc)(size_t)) {
    hang_free(dst);
    dst = Hang();
    dst.kind = src.kind; dst.ax = src.ax; dst.ay = src.ay; dst.bx = src.bx; dst.by = src.by;
    memcpy(dst.prm, src.prm, sizeof(dst.prm));
    if (src.mask && hang_ensure_mask(dst, alloc)) memcpy(dst.mask, src.mask, MASK_BYTES);
}
inline int hang_count(const Hang& h) {
    if (!h.mask) return 0;
    int n = 0;
    for (int b = 0; b < MASK_BYTES; ++b) n += __builtin_popcount(h.mask[b]);
    return n;
}

inline float hng(const Hang& h, uint8_t p) { return h.prm[p]; }

// Which way it hangs: from the pin toward the middle of what's below it (the side of the pin line that
// faces down in the photo; a point pin: below its height). Nothing below: straight down.
inline void hang_dir(const Hang& h, float& dx, float& dy) {
    const float rx = (h.ax + h.bx) * 0.5f, ry = (h.ay + h.by) * 0.5f;
    float nx = -(float)(h.by - h.ay), ny = (float)(h.bx - h.ax);
    const float l = sqrtf(nx * nx + ny * ny);
    if (l < 1) { nx = 0; ny = 1; }
    else { nx /= l; ny /= l; if (ny < 0) { nx = -nx; ny = -ny; } }
    double sx = 0, sy = 0; int n = 0;
    if (h.mask)
        for (int y = 0; y < IMG_H; ++y)
            for (int x = 0; x < IMG_W; ++x)
                if (mask_get(h.mask, x, y) && (x - rx) * nx + (y - ry) * ny > 0) { sx += x; sy += y; ++n; }
    if (n < 4) { dx = nx; dy = ny; return; }
    dx = (float)(sx / n) - rx; dy = (float)(sy / n) - ry;
}

// ---- prepare ----------------------------------------------------------------------------------------
inline bool hang_prepare(Hang& h, const uint16_t* photo, bool sw, void* (*alloc)(size_t)) {
    hang_free_runtime(h);
    if (!h.mask || !photo) return false;
    const HangKindInfo& K = hang_kind_info(h.kind);
    double sx = 0, sy = 0; int n = 0;
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x)
            if (mask_get(h.mask, x, y)) { sx += x; sy += y; ++n; }
    if (n < 12) return false;
    const float mx = (float)(sx / n), my = (float)(sy / n);
    const float rx = (h.ax + h.bx) * 0.5f, ry = (h.ay + h.by) * 0.5f;
    float dx, dy;
    hang_dir(h, dx, dy);                              // it hangs toward its middle below the pin
    float l = sqrtf(dx * dx + dy * dy);
    if (l < 4) { dx = 0; dy = 1; l = 1; }
    h.e2[0] = dx / l; h.e2[1] = dy / l;
    h.e1[0] = -h.e2[1]; h.e1[1] = h.e2[0];
    if ((h.bx - h.ax) * h.e1[0] + (h.by - h.ay) * h.e1[1] < 0) { h.e1[0] = -h.e1[0]; h.e1[1] = -h.e1[1]; }
    h.R[0] = rx; h.R[1] = ry;
    float smin = 1e9f, smax = -1e9f, tmin = 1e9f, tmax = -1e9f;
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x)
            if (mask_get(h.mask, x, y)) {
                const float s = (x - rx) * h.e1[0] + (y - ry) * h.e1[1], t = (x - rx) * h.e2[0] + (y - ry) * h.e2[1];
                if (s < smin) smin = s;
                if (s > smax) smax = s;
                if (t < tmin) tmin = t;
                if (t > tmax) tmax = t;
            }
    smin -= 1.5f; smax += 1.5f; tmax += 1.5f;
    if (tmax < 4) tmax = 4;
    float rows[HR_MAX];
    int nr = 0;
    if (tmin < -0.5f) rows[nr++] = tmin - 1.5f;
    rows[nr++] = 0.0f;
    h.npin = nr;
    for (int j = 1; j <= K.nr; ++j) rows[nr++] = tmax * j / K.nr;
    h.nrows = nr; h.nc = K.nc; h.tlen = tmax;
    for (int j = 0; j < nr; ++j) {
        h.kn[j] = K.g / (rows[j] > 4.0f ? rows[j] : 4.0f);
        for (int c = 0; c < K.nc; ++c) {
            const float s = smin + (smax - smin) * c / (K.nc - 1);
            float* p = h.rest[j * K.nc + c];
            p[0] = rx + s * h.e1[0] + rows[j] * h.e2[0];
            p[1] = ry + s * h.e1[1] + rows[j] * h.e2[1];
            h.x[j * K.nc + c][0] = h.xp[j * K.nc + c][0] = p[0];
            h.x[j * K.nc + c][1] = h.xp[j * K.nc + c][1] = p[1];
        }
    }
    h.rc[0] = mx - rx; h.rc[1] = my - ry;
    h.Lc = sqrtf(h.rc[0] * h.rc[0] + h.rc[1] * h.rc[1]);
    if (h.Lc < 4) h.Lc = 4;
    h.kc = K.g / h.Lc;
    h.th = h.thp = 0;
    // links: along each chain, across each row, and (between free rows) the shear
    h.nl = 0;
    auto L = [&](int a, int b, float st) {
        if (h.nl >= HL_MAX) return;
        const float ddx = h.rest[b][0] - h.rest[a][0], ddy = h.rest[b][1] - h.rest[a][1];
        h.link[h.nl++] = HLink{(uint8_t)a, (uint8_t)b, sqrtf(ddx * ddx + ddy * ddy), st};
    };
    const int nc = K.nc;
    for (int j = (h.npin > 1 ? h.npin : 1); j < nr; ++j) {
        for (int c = 0; c < nc; ++c) L((j - 1) * nc + c, j * nc + c, 1.0f);
        for (int c = 0; c < nc - 1; ++c) {
            L(j * nc + c, j * nc + c + 1, 1.0f);
            if (j - 1 >= h.npin) {
                L((j - 1) * nc + c, j * nc + c + 1, K.shear);
                L((j - 1) * nc + c + 1, j * nc + c, K.shear);
            }
        }
    }
    // drawing region: the mesh swung either way as far as it's drawn
    float bx0 = 1e9f, by0 = 1e9f, bx1 = -1e9f, by1 = -1e9f;
    for (int d = -1; d <= 1; ++d) {
        const float a = d * HANG_REACH_DEG * 0.0174533f, cs = cosf(a), sn = sinf(a);
        for (int i = 0; i < nr * nc; ++i) {
            const float qx = h.rest[i][0] - rx, qy = h.rest[i][1] - ry;
            const float px = rx + qx * cs - qy * sn, py = ry + qx * sn + qy * cs;
            if (px < bx0) bx0 = px;
            if (px > bx1) bx1 = px;
            if (py < by0) by0 = py;
            if (py > by1) by1 = py;
        }
    }
    h.x0 = (int)floorf(bx0) - 2; h.x1 = (int)ceilf(bx1) + 3; h.y0 = (int)floorf(by0) - 2; h.y1 = (int)ceilf(by1) + 3;
    if (h.x0 < 0) h.x0 = 0;
    if (h.y0 < 0) h.y0 = 0;
    if (h.x1 > IMG_W) h.x1 = IMG_W;
    if (h.y1 > IMG_H) h.y1 = IMG_H;
    const int rw = h.x1 - h.x0, rh = h.y1 - h.y0;
    if (rw <= 0 || rh <= 0) return false;
    // what it carries: the selection and three rings past it; what's behind: filled from the surroundings
    const int N = IMG_W * IMG_H;
    uint8_t* hole = (uint8_t*)alloc(N);
    int16_t* nx = (int16_t*)alloc((size_t)N * 2);
    int16_t* ny = (int16_t*)alloc((size_t)N * 2);
    h.tex = (uint16_t*)alloc((size_t)rw * rh * 2);
    h.texd = (uint16_t*)alloc((size_t)rw * rh * 2);
    h.fill = (uint16_t*)alloc((size_t)rw * rh * 2);
    h.alpha = (uint8_t*)alloc((size_t)rw * rh);
    if (!hole || !nx || !ny || !h.tex || !h.texd || !h.fill || !h.alpha) {
        if (hole) free(hole);
        if (nx) free(nx);
        if (ny) free(ny);
        hang_free_runtime(h);
        return false;
    }
    for (int i = 0; i < N; ++i) hole[i] = mask_get(h.mask, i % IMG_W, i / IMG_W) ? 4 : 0;   // 4: it, 3…1: rings 1…3
    for (int r = 3; r >= 1; --r)
        for (int i = 0; i < N; ++i) {
            if (hole[i]) continue;
            const int x = i % IMG_W, y = i / IMG_W;
            if ((x > 0 && hole[i - 1] > r) || (x < IMG_W - 1 && hole[i + 1] > r) || (y > 0 && hole[i - IMG_W] > r) || (y < IMG_H - 1 && hole[i + IMG_W] > r))
                hole[i] = (uint8_t)r;
        }
    auto px = [&](int i) -> uint16_t { uint16_t v = photo[i]; return sw ? (uint16_t)((v >> 8) | (v << 8)) : v; };
    for (int i = 0; i < N; ++i) {
        if (!hole[i]) { nx[i] = (int16_t)(i % IMG_W); ny[i] = (int16_t)(i / IMG_W); }
        else nx[i] = -1;
    }
    par_nearest(nx, ny);
    for (int y = h.y0; y < h.y1; ++y)
        for (int x = h.x0; x < h.x1; ++x) {
            const int i = y * IMG_W + x, o = (y - h.y0) * rw + (x - h.x0);
            h.tex[o] = px(i);
            uint8_t a = 0;
            if (hole[i] == 4) {
                bool edge = false;
                if (x > 0 && hole[i - 1] != 4) edge = true;
                if (x < IMG_W - 1 && hole[i + 1] != 4) edge = true;
                if (y > 0 && hole[i - IMG_W] != 4) edge = true;
                if (y < IMG_H - 1 && hole[i + IMG_W] != 4) edge = true;
                a = edge ? 210 : 255;
            } else if (hole[i] == 3) a = 120;
            else if (hole[i] == 2) a = 45;
            else if (hole[i] == 1) a = 20;
            h.alpha[o] = a;
            if (!hole[i] || nx[i] < 0) { h.fill[o] = px(i); continue; }
            const int ex = nx[i], ey = ny[i], qx = 2 * ex - x, qy = 2 * ey - y;   // mirrored across the edge
            if (qx >= 0 && qy >= 0 && qx < IMG_W && qy < IMG_H && !hole[qy * IMG_W + qx]) h.fill[o] = px(qy * IMG_W + qx);
            else h.fill[o] = px(ey * IMG_W + ex);
        }
    // What's behind, better: the surroundings' colours smoothed across the hole (nearest colour, then
    // averaged with its neighbours), plus the fine texture of the mirrored fill on top - the sky stays
    // sky, a shoulder carries on as skin, sand keeps its grain; no streaks, no blur.
    {
        const int rn = rw * rh;
        float* g = (float*)alloc((size_t)rn * 3 * sizeof(float));
        if (g) {
            for (int y = h.y0; y < h.y1; ++y)
                for (int x = h.x0; x < h.x1; ++x) {
                    const int i = y * IMG_W + x, o = (y - h.y0) * rw + (x - h.x0);
                    const uint16_t c = (hole[i] && nx[i] >= 0) ? px(ny[i] * IMG_W + nx[i]) : px(i);
                    g[o * 3] = (float)(c >> 11); g[o * 3 + 1] = (float)((c >> 5) & 63); g[o * 3 + 2] = (float)(c & 31);
                }
            auto at = [&](int x, int y) { if (x < 0) x = 0; if (x >= rw) x = rw - 1; if (y < 0) y = 0; if (y >= rh) y = rh - 1; return y * rw + x; };
            for (int it = 0; it < HANG_FILL_ITERS; ++it)
                for (int y = 0; y < rh; ++y)
                    for (int x = 0; x < rw; ++x) {
                        if (!hole[(y + h.y0) * IMG_W + x + h.x0]) continue;
                        const int o = y * rw + x, a = at(x - 1, y), b = at(x + 1, y), c = at(x, y - 1), d = at(x, y + 1);
                        for (int ch = 0; ch < 3; ++ch) g[o * 3 + ch] = 0.25f * (g[a * 3 + ch] + g[b * 3 + ch] + g[c * 3 + ch] + g[d * 3 + ch]);
                    }
            for (int y = 0; y < rh; ++y)
                for (int x = 0; x < rw; ++x) {
                    if (!hole[(y + h.y0) * IMG_W + x + h.x0]) continue;
                    const int o = y * rw + x;
                    float bl[3] = {0, 0, 0};
                    for (int dy = -2; dy <= 2; ++dy)
                        for (int dx = -2; dx <= 2; ++dx) {
                            const uint16_t c = h.fill[at(x + dx, y + dy)];
                            bl[0] += (float)(c >> 11); bl[1] += (float)((c >> 5) & 63); bl[2] += (float)(c & 31);
                        }
                    const uint16_t m = h.fill[o];
                    const float mv[3] = {(float)(m >> 11), (float)((m >> 5) & 63), (float)(m & 31)};
                    int v[3];
                    static const float lim[3] = {HANG_FILL_GRAIN * 31 / 255, HANG_FILL_GRAIN * 63 / 255, HANG_FILL_GRAIN * 31 / 255};
                    for (int ch = 0; ch < 3; ++ch) {
                        float d = HANG_FILL_DETAIL * (mv[ch] - bl[ch] / 25.0f);
                        d = d < -lim[ch] ? -lim[ch] : (d > lim[ch] ? lim[ch] : d);
                        v[ch] = (int)lroundf(g[o * 3 + ch] + d);
                    }
                    v[0] = v[0] < 0 ? 0 : (v[0] > 31 ? 31 : v[0]);
                    v[1] = v[1] < 0 ? 0 : (v[1] > 63 ? 63 : v[1]);
                    v[2] = v[2] < 0 ? 0 : (v[2] > 31 ? 31 : v[2]);
                    g[o * 3] = (float)v[0]; g[o * 3 + 1] = (float)v[1]; g[o * 3 + 2] = (float)v[2];
                }
            for (int y = 0; y < rh; ++y)
                for (int x = 0; x < rw; ++x)
                    if (hole[(y + h.y0) * IMG_W + x + h.x0]) {
                        const int o = y * rw + x;
                        h.fill[o] = (uint16_t)(((int)g[o * 3] << 11) | ((int)g[o * 3 + 1] << 5) | (int)g[o * 3 + 2]);
                    }
            free(g);
        }
    }
    // Its edge, matted: the pixels along its outline and the rings are part it, part what was behind
    // it. Each is split, c = a·H + (1 − a)·B (H: its nearest solid pixel, B: what's behind), so when it
    // moves it carries its own colour at that cover - not a seam of the old background.
    for (int i = 0; i < N; ++i) {
        const int x = i % IMG_W, y = i / IMG_W;
        const bool solid = x >= h.x0 && x < h.x1 && y >= h.y0 && y < h.y1 && h.alpha[(y - h.y0) * rw + (x - h.x0)] == 255;
        if (solid) { nx[i] = (int16_t)x; ny[i] = (int16_t)y; } else nx[i] = -1;
    }
    par_nearest(nx, ny);
    for (int y = h.y0; y < h.y1; ++y)
        for (int x = h.x0; x < h.x1; ++x) {
            const int i = y * IMG_W + x, o = (y - h.y0) * rw + (x - h.x0);
            const uint8_t v = h.alpha[o];
            h.texd[o] = h.tex[o];
            if (v == 0 || v == 255) continue;
            float a = v / 255.0f;
            if (nx[i] >= 0) {
                const uint16_t hc = px(ny[i] * IMG_W + nx[i]), bc = h.fill[o], cc = h.tex[o];
                auto f8 = [](uint16_t c, float* o3) { o3[0] = (c >> 11) * (255.0f / 31); o3[1] = ((c >> 5) & 63) * (255.0f / 63); o3[2] = (c & 31) * (255.0f / 31); };
                float H3[3], B3[3], C3[3];
                f8(hc, H3); f8(bc, B3); f8(cc, C3);
                float dd = 0, num = 0;
                for (int k = 0; k < 3; ++k) { const float d = H3[k] - B3[k]; dd += d * d; num += (C3[k] - B3[k]) * d; }
                if (dd >= 300.0f) {                     // (too alike to tell: the plain fade)
                    const float lo = v == 210 ? 0.4f : 0.0f, hi = v == 210 ? 1.0f : (v == 120 ? 0.8f : (v == 45 ? 0.5f : 0.3f));
                    a = num / (dd > 100.0f ? dd : 100.0f);
                    a = a < lo ? lo : (a > hi ? hi : a);
                }
                h.texd[o] = hc;
            }
            int q = (int)lroundf(a * 255.0f);
            h.alpha[o] = (uint8_t)(q < 1 ? 1 : (q > 254 ? 254 : q));
        }
    free(hole); free(nx); free(ny);
    h.t = h.acc = h.fade = 0;
    h.awake = true; h.drawn = false;
    h.ready = true;
    return true;
}

// ---- physics ----------------------------------------------------------------------------------------
inline void hang_constrain(Hang& h) {
    for (int it = 0; it < HANG_ITERS; ++it)
        for (int l = 0; l < h.nl; ++l) {
            const HLink& k = h.link[l];
            const bool fa = k.a / h.nc >= h.npin, fb = k.b / h.nc >= h.npin;   // free?
            if (!fa && !fb) continue;
            float* A = h.x[k.a]; float* B = h.x[k.b];
            const float dx = B[0] - A[0], dy = B[1] - A[1], d = sqrtf(dx * dx + dy * dy);
            if (d < 1e-6f) continue;
            const float e = (d - k.len) / d * k.stiff;
            if (fa && fb) { A[0] += 0.5f * e * dx; A[1] += 0.5f * e * dy; B[0] -= 0.5f * e * dx; B[1] -= 0.5f * e * dy; }
            else if (fa) { A[0] += e * dx; A[1] += e * dy; }
            else { B[0] -= e * dx; B[1] -= e * dy; }
        }
}
inline void hang_substep(Hang& h, float fx, float fy, float wind) {
    const HangKindInfo& K = hang_kind_info(h.kind);
    const float dt = HANG_DT, damp = 1.0f - K.c * dt;
    if (h.kind == HANG_CHARM) {                       // one stiff body turning about its hook
        const float c = cosf(h.th), s = sinf(h.th);
        const float rx = h.rc[0] * c - h.rc[1] * s, ry = h.rc[0] * s + h.rc[1] * c;
        const float gx = fx + wind * h.e1[0], gy = fy + wind * h.e1[1];
        const float a = (rx * gy - ry * gx) / (h.Lc * h.Lc) - h.kc * h.th;
        float nt = h.th + (h.th - h.thp) * damp + a * dt * dt;
        if (nt > 1.2f) nt = 1.2f;
        if (nt < -1.2f) nt = -1.2f;
        h.thp = h.th; h.th = nt;
        const float c2 = cosf(nt), s2 = sinf(nt);
        for (int i = 0; i < h.nrows * h.nc; ++i) {
            h.xp[i][0] = h.x[i][0]; h.xp[i][1] = h.x[i][1];
            const float qx = h.rest[i][0] - h.R[0], qy = h.rest[i][1] - h.R[1];
            h.x[i][0] = h.R[0] + qx * c2 - qy * s2; h.x[i][1] = h.R[1] + qx * s2 + qy * c2;
        }
        return;
    }
    const int nc = h.nc, n0 = h.npin, nr = h.nrows;
    for (int j = n0; j < nr; ++j) {
        const float k = h.kn[j], wk = wind * (0.55f + 0.45f * (j - n0 + 1) / (nr - n0));
        for (int c = 0; c < nc; ++c) {
            const int i = j * nc + c;
            const float ax = fx + k * (h.rest[i][0] - h.x[i][0]) + wk * h.e1[0];
            const float ay = fy + k * (h.rest[i][1] - h.x[i][1]) + wk * h.e1[1];
            const float nx = h.x[i][0] + (h.x[i][0] - h.xp[i][0]) * damp + ax * dt * dt;
            const float ny = h.x[i][1] + (h.x[i][1] - h.xp[i][1]) * damp + ay * dt * dt;
            h.xp[i][0] = h.x[i][0]; h.xp[i][1] = h.x[i][1];
            h.x[i][0] = nx; h.x[i][1] = ny;
        }
    }
    hang_constrain(h);
}
// dn: unit "down" in screen axes (upright (0, 1, 0)); jolt: g, gravity removed. Returns true while moving.
inline bool hang_step(Hang& h, float dt, const float dn[3], const float jolt[3]) {
    if (!h.ready) return false;
    const HangKindInfo& K = hang_kind_info(h.kind);
    const float sw = hng(h, HP_SWING) * K.gain;
    const float m = sqrtf(dn[0] * dn[0] + dn[1] * dn[1]);
    const float fx = sw * (K.g * dn[0] - K.j * jolt[0]);          // the change in "down" (upright: (0, 1))
    const float fy = sw * (K.g * (dn[1] - m) - K.j * jolt[1]);
    const float breeze = hng(h, HP_BREEZE);
    h.acc += dt > 0.1f ? 0.1f : dt;
    while (h.acc >= HANG_DT) {
        h.acc -= HANG_DT;
        h.t += HANG_DT;
        const float T = 6.2831853f * h.t;
        const float wind = sw * breeze * K.wind * (0.6f * sinf(T / 3.1f) + 0.4f * sinf(T / 1.7f + 1.0f));
        hang_substep(h, fx, fy, wind);
    }
    float disp = 0, vel = 0;
    for (int i = h.npin * h.nc; i < h.nrows * h.nc; ++i)
        for (int a = 0; a < 2; ++a) {
            const float d = fabsf(h.x[i][a] - h.rest[i][a]), v = fabsf(h.x[i][a] - h.xp[i][a]);
            if (d > disp) disp = d;
            if (v > vel) vel = v;
        }
    if (h.kind == HANG_CHARM) {
        disp = 0; vel = fabsf(h.th - h.thp) * h.tlen;
        for (int i = 0; i < h.nrows * h.nc; ++i)
            for (int a = 0; a < 2; ++a) { const float d = fabsf(h.x[i][a] - h.rest[i][a]); if (d > disp) disp = d; }
    }
    const bool quiet = fabsf(fx) + fabsf(fy) < 1.0f && breeze <= 0.0f;
    if (quiet && disp < 0.05f && vel < 0.002f) {          // home: exactly the photo
        for (int i = 0; i < h.nrows * h.nc; ++i) { h.x[i][0] = h.xp[i][0] = h.rest[i][0]; h.x[i][1] = h.xp[i][1] = h.rest[i][1]; }
        h.th = h.thp = 0;
        h.awake = false;
    } else {
        h.awake = true;
    }
    h.fade = disp < 1.0f ? disp : 1.0f;
    return h.awake;
}

// A flick: the nodes near a tap pushed sideways, away from where it was tapped.
inline bool hang_poke(Hang& h, float px, float py, float strength = 1.0f) {
    if (!h.ready || px < h.x0 || px >= h.x1 || py < h.y0 || py >= h.y1) return false;
    const float r = 0.35f * h.tlen > 14.0f ? 0.35f * h.tlen : 14.0f;
    float best = 1e9f;
    for (int i = h.npin * h.nc; i < h.nrows * h.nc; ++i) {
        const float d = sqrtf((h.x[i][0] - px) * (h.x[i][0] - px) + (h.x[i][1] - py) * (h.x[i][1] - py));
        if (d < best) best = d;
    }
    if (best > r) return false;
    const float side = (px - h.R[0]) * h.e1[0] + (py - h.R[1]) * h.e1[1];
    const float sgn = side > 0 ? -1.0f : 1.0f;
    const float sw = hng(h, HP_SWING);
    if (h.kind == HANG_CHARM) {
        h.thp -= 160.0f * strength * sw * sgn / h.Lc * HANG_DT * 1.5f;
        h.awake = true;
        return true;
    }
    for (int i = h.npin * h.nc; i < h.nrows * h.nc; ++i) {
        const float d = sqrtf((h.x[i][0] - px) * (h.x[i][0] - px) + (h.x[i][1] - py) * (h.x[i][1] - py));
        float f = 1.0f - d / (2 * r);
        if (f <= 0) continue;
        const float v = 160.0f * strength * sw * sgn * f;     // px/s
        h.xp[i][0] -= v * h.e1[0] * HANG_DT; h.xp[i][1] -= v * h.e1[1] * HANG_DT;
    }
    h.awake = true;
    return true;
}

// ---- drawing ----------------------------------------------------------------------------------------
// out: the content (sw = byte-swapped). First the region's selection (+ rings) is filled with what's
// behind, then the mesh is drawn where it is now. (The caller restores the region from the photo and
// redraws water there first.)
inline void hang_render(const Hang& h, uint16_t* out, bool osw) {
    if (!h.ready) return;
    const int rw = h.x1 - h.x0, rh = h.y1 - h.y0;
    auto put = [&](int i, uint16_t c) { out[i] = osw ? (uint16_t)((c >> 8) | (c << 8)) : c; };
    auto get = [&](int i) -> uint16_t { const uint16_t v = out[i]; return osw ? (uint16_t)((v >> 8) | (v << 8)) : v; };
    for (int y = h.y0; y < h.y1; ++y)
        for (int x = h.x0; x < h.x1; ++x) {
            const int o = (y - h.y0) * rw + (x - h.x0);
            if (h.alpha[o]) put(y * IMG_W + x, h.fill[o]);
        }
    const int fd = (int)(h.fade * 256.0f);
    auto eff = [fd](int v) { return v ? (256 * (256 - fd) + (v == 255 ? 256 : v) * fd) >> 8 : 0; };   // rest: solid
    auto tri = [&](int ia, int ib, int ic) {
        const float ax = h.x[ia][0], ay = h.x[ia][1], bx = h.x[ib][0], by = h.x[ib][1], cx = h.x[ic][0], cy = h.x[ic][1];
        const float den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
        if (fabsf(den) < 1e-9f) return;
        const float inv = 1.0f / den;
        int X0 = (int)floorf(fminf(ax, fminf(bx, cx))), X1 = (int)ceilf(fmaxf(ax, fmaxf(bx, cx)));
        int Y0 = (int)floorf(fminf(ay, fminf(by, cy))), Y1 = (int)ceilf(fmaxf(ay, fmaxf(by, cy)));
        if (X0 < h.x0) X0 = h.x0;
        if (X1 > h.x1 - 1) X1 = h.x1 - 1;
        if (Y0 < h.y0) Y0 = h.y0;
        if (Y1 > h.y1 - 1) Y1 = h.y1 - 1;
        const float sax = h.rest[ia][0], say = h.rest[ia][1], sbx = h.rest[ib][0], sby = h.rest[ib][1], scx = h.rest[ic][0], scy = h.rest[ic][1];
        for (int y = Y0; y <= Y1; ++y)
            for (int x = X0; x <= X1; ++x) {
                const float l0 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) * inv;
                const float l1 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) * inv;
                const float l2 = 1.0f - l0 - l1;
                if (l0 < -1e-7f || l1 < -1e-7f || l2 < -1e-7f) continue;
                float sx = l0 * sax + l1 * sbx + l2 * scx, sy = l0 * say + l1 * sby + l2 * scy;
                sx -= h.x0; sy -= h.y0;                  // into the region
                if (sx < 0) sx = 0;
                if (sy < 0) sy = 0;
                if (sx > rw - 1.001f) sx = rw - 1.001f;
                if (sy > rh - 1.001f) sy = rh - 1.001f;
                const int ix = (int)sx, iy = (int)sy;
                const int ixb = ix + 1 < rw ? ix + 1 : ix, iyb = iy + 1 < rh ? iy + 1 : iy;
                const int fx = (int)((sx - ix) * 256.0f), fy = (int)((sy - iy) * 256.0f);
                const int o00 = iy * rw + ix, o10 = iy * rw + ixb, o01 = iyb * rw + ix, o11 = iyb * rw + ixb;
                const int a = ((eff(h.alpha[o00]) * (256 - fx) + eff(h.alpha[o10]) * fx) * (256 - fy) +
                               (eff(h.alpha[o01]) * (256 - fx) + eff(h.alpha[o11]) * fx) * fy) >> 16;
                if (a <= 0) continue;
                const uint16_t* T = fd >= 256 ? h.texd : h.tex;
                uint16_t c00 = T[o00], c10 = T[o10], c01 = T[o01], c11 = T[o11];
                if (fd > 0 && fd < 256) {               // (between: blend the two)
                    auto mix = [fd](uint16_t a, uint16_t b) {
                        const int r = ((a >> 11) * (256 - fd) + (b >> 11) * fd) >> 8, g = (((a >> 5) & 63) * (256 - fd) + ((b >> 5) & 63) * fd) >> 8;
                        const int bl = ((a & 31) * (256 - fd) + (b & 31) * fd) >> 8;
                        return (uint16_t)((r << 11) | (g << 5) | bl);
                    };
                    c00 = mix(c00, h.texd[o00]); c10 = mix(c10, h.texd[o10]); c01 = mix(c01, h.texd[o01]); c11 = mix(c11, h.texd[o11]);
                }
                auto ch = [&](int sh, int mk) {
                    const int v00 = (c00 >> sh) & mk, v10 = (c10 >> sh) & mk, v01 = (c01 >> sh) & mk, v11 = (c11 >> sh) & mk;
                    return ((v00 * (256 - fx) + v10 * fx) * (256 - fy) + (v01 * (256 - fx) + v11 * fx) * fy);   // ×65536
                };
                const int R = ch(11, 31), G = ch(5, 63), B = ch(0, 31);
                const int i = y * IMG_W + x;
                const uint16_t d = get(i);
                const int dr = d >> 11, dg = (d >> 5) & 63, db = d & 31;
                const int r5 = (int)(((int64_t)R * a + (int64_t)(dr << 16) * (256 - a) + (1 << 23)) >> 24);
                const int g6 = (int)(((int64_t)G * a + (int64_t)(dg << 16) * (256 - a) + (1 << 23)) >> 24);
                const int b5 = (int)(((int64_t)B * a + (int64_t)(db << 16) * (256 - a) + (1 << 23)) >> 24);
                put(i, (uint16_t)(((r5 > 31 ? 31 : r5) << 11) | ((g6 > 63 ? 63 : g6) << 5) | (b5 > 31 ? 31 : b5)));
            }
    };
    for (int j = 0; j < h.nrows - 1; ++j)
        for (int c = 0; c < h.nc - 1; ++c) {
            const int q0 = j * h.nc + c, q1 = q0 + 1, q2 = q0 + h.nc + 1, q3 = q0 + h.nc;
            tri(q0, q1, q2);
            tri(q0, q2, q3);
        }
}

// ---- .jig lines: "hang hair ax,ay bx,by [swing=…] [breeze=…]", "hrow y a-b …" --------------------------
inline bool parse_hang_line(char* line, Hang* hs, int& nh, int& cur, void* (*alloc)(size_t)) {
    char kind[10];
    int used = 0, n = 0;
    if (sscanf(line, "%9s%n", kind, &used) != 1) return false;
    char* rest = line + used;
    if (strcmp(kind, "hang") == 0) {
        cur = -1;
        char kn[10];
        int ax, ay, bx, by;
        if (nh >= HANG_MAX || sscanf(rest, "%9s %d,%d %d,%d%n", kn, &ax, &ay, &bx, &by, &n) != 5) return true;
        rest += n;
        int k = -1;
        for (int i = 0; i < HANG_KIND_COUNT; ++i) if (strcmp(kn, hang_kind_info((uint8_t)i).key) == 0) k = i;
        if (k < 0) return true;
        Hang& h = hs[nh];
        hang_free(h);
        h = Hang();
        if (!hang_ensure_mask(h, alloc)) return true;
        h.kind = (uint8_t)k; h.ax = (int16_t)ax; h.ay = (int16_t)ay; h.bx = (int16_t)bx; h.by = (int16_t)by;
        char tok[48];
        while (sscanf(rest, "%47s%n", tok, &n) == 1) {
            rest += n;
            char* eq = strchr(tok, '=');
            if (!eq) continue;
            *eq = 0;
            for (int p = 0; p < HP_COUNT; ++p)
                if (strcmp(tok, hparam_info((uint8_t)p).key) == 0) {
                    char* end = nullptr;
                    const float v = strtof(eq + 1, &end);
                    const HParamInfo& pi = hparam_info((uint8_t)p);
                    if (end != eq + 1 && !isnan(v)) h.prm[p] = v < pi.lo ? pi.lo : (v > pi.hi ? pi.hi : v);
                }
        }
        cur = nh++;
        return true;
    }
    if (strcmp(kind, "hrow") == 0) {
        if (cur < 0) return true;
        int y;
        if (sscanf(rest, "%d%n", &y, &n) != 1) return true;
        rest += n;
        int a, b;
        while (sscanf(rest, " %d-%d%n", &a, &b, &n) == 2) { rest += n; mask_span(hs[cur].mask, y, a, b, true); }
        return true;
    }
    return false;
}

template <typename F>
inline void save_hangs(F& f, const Hang* hs, int nh) {
    for (int k = 0; k < nh; ++k) {
        const Hang& h = hs[k];
        if (hang_count(h) < 12) continue;
        f.printf("hang %s %d,%d %d,%d", hang_kind_info(h.kind).key, h.ax, h.ay, h.bx, h.by);
        for (int p = 0; p < HP_COUNT; ++p)
            if (fabsf(h.prm[p] - hparam_info((uint8_t)p).dflt) > 1e-6f) f.printf(" %s=%.3g", hparam_info((uint8_t)p).key, h.prm[p]);
        f.printf("\n");
        save_mask_rows(f, h.mask, "hrow");
    }
}

}  // namespace jig
