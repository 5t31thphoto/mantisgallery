#pragma once
// Parallax: the photo split into depth planes that slide against each other as you tilt the Core2.
//
// Planes are painted masks with a depth (0 = as far as the background, 1 = nearest), optionally a
// depth ramp (a ground plane running off to the horizon). Each frame the screen samples every plane
// at its own offset, nearest first:
//   * the background is pinned by default ("focus" 0) — parallax is relative motion, so only the
//     nearer planes and the strip they uncover are ever redrawn (no edge overscan needed either);
//   * every plane carries a fill: its own pixels, continued past its silhouette by mirroring its own
//     texture across the edge — what's uncovered behind a moving plane looks like the surface it
//     belongs to (no smear, no halo of the foreground's colours);
//   * planes are sampled with a separable Catmull-Rom filter (weights per plane per frame, from a
//     table), so a crisp edge stays crisp while it slides by fractions of a pixel, and edges blend
//     with sub-pixel coverage;
//   * the view comes from the gyro (any tilt or turn, recentring itself over a few seconds), through
//     a spring so the depth has weight, plus an optional slow drift for a Core2 standing on a desk.
// Water and jiggle zones are drawn into the content first, so all three effects stack.
#include "jiggle_water.h"

namespace jig {

constexpr int PLANE_MAX = 4;
constexpr float PAR_RECENTER_S = 2.5f;     // a held tilt eases back to centre over this
constexpr float PAR_BIAS_S = 3.0f;         // gyro bias learnt while still
constexpr float PAR_DEAD_DPS = 0.6f;       // gyro noise floor: slower than this isn't a turn

enum PParam : uint8_t { PP_AMOUNT = 0, PP_AMBIENT, PP_RESPONSE, PP_FOCUS, PP_TILT, PP_VERT, PP_INVERT, PP_COUNT };
struct PParamInfo { const char* key; const char* label; float lo, hi, dflt; };
inline const PParamInfo& pparam_info(uint8_t p) {
    static const PParamInfo k[PP_COUNT] = {
        {"amount", "Depth (px the nearest ground slides)", 0.0f, 24.0f, 6.0f},
        {"ambient", "Drift while still", 0.0f, 1.0f, 0.3f},
        {"response", "Response (spring speed)", 0.2f, 3.0f, 1.0f},
        {"focus", "Pinned depth (0 = background stays put)", 0.0f, 1.0f, 0.0f},
        {"tilt", "Tilt for the full slide (deg)", 3.0f, 40.0f, 14.0f},
        {"vert", "Up/down share", 0.0f, 1.5f, 1.0f},
        {"invert", "Reverse direction", 0.0f, 1.0f, 0.0f},
    };
    return k[p < PP_COUNT ? p : 0];
}

struct Plane {
    // ---- definition (saved)
    float   depth = 0.5f;                 // 0 = background depth, 1 = nearest
    bool    ramp = false;                 // depth runs from `depth` at A to `depth_b` at B
    int16_t ax = 0, ay = 0, bx = 0, by = 0;
    float   depth_b = 0.5f;
    uint8_t* mask = nullptr;
    // ---- runtime
    uint8_t*  cov = nullptr;              // where it can show: its pixels + a margin under nearer planes
    uint16_t* fill = nullptr;             // its colours, continued past its edge (native RGB565)
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;   // bbox of cov
    float c0 = 0, cx = 0, cy = 0, dlo = 0, dhi = 0;   // depth = clamp(c0 + cx·x + cy·y, dlo, dhi)
};

struct Parallax {
    float prm[PP_COUNT];
    int   n = 0;
    Plane pl[PLANE_MAX];
    bool  grid = false;                   // ground grid (fitted by eye): corners near-L, near-R, far-R, far-L
    float gq[8] = {0};                    //   → the horizon and how fast the ground recedes
    // ---- runtime
    bool      ready = false;
    uint8_t*  owner = nullptr;            // per pixel: 0 = background, k+1 = plane k
    uint16_t* bg_fill = nullptr;
    int       order[PLANE_MAX];           // nearest first
    int       margin = 0;
    Plane     gp;                         // the ground's depth field (grid): 1 at the nearest ground, 0 at the horizon
    float     zoom = 1.0f;                // a touch of zoom so sliding never shows past the photo's edges
    int16_t   zx[IMG_W], zy[IMG_H];       // where each screen column / row samples before sliding (×16)
    // view (screen units: +x = near planes slide right, +y = down; 1 = full slide)
    float th[2] = {0, 0}, bias[2] = {0, 0}, v[2] = {0, 0}, vv[2] = {0, 0}, t = 0;
    float dv[2] = {0, 0};                 // view the screen was last drawn with
    bool  drawn = false;
    Parallax() { for (int i = 0; i < PP_COUNT; ++i) prm[i] = pparam_info((uint8_t)i).dflt; }
};

// ---- Catmull-Rom weights, 64 phases, Q8 (sum 256) ------------------------------------------------
inline int16_t (&par_w())[64][4] { static int16_t w[64][4]; return w; }
inline void par_tab() {
    auto& W = par_w();
    for (int i = 0; i < 64; ++i) {
        float t = i / 64.0f, t2 = t * t, t3 = t2 * t;
        float f[4] = {(-t3 + 2 * t2 - t) * 0.5f, (3 * t3 - 5 * t2 + 2) * 0.5f, (-3 * t3 + 4 * t2 + t) * 0.5f, (t3 - t2) * 0.5f};
        int s = 0;
        for (int k = 0; k < 4; ++k) { W[i][k] = (int16_t)lroundf(f[k] * 256.0f); s += W[i][k]; }
        W[i][1] += (int16_t)(256 - s);
    }
}

inline float pprm(const Parallax& P, uint8_t p) { return P.prm[p]; }

inline void plane_free_runtime(Plane& p) {
    if (p.cov) free(p.cov);
    if (p.fill) free(p.fill);
    p.cov = nullptr; p.fill = nullptr;
}
inline void plane_free(Plane& p) {
    plane_free_runtime(p);
    if (p.mask) free(p.mask);
    p.mask = nullptr;
}
inline void parallax_free_runtime(Parallax& P) {
    for (int k = 0; k < PLANE_MAX; ++k) plane_free_runtime(P.pl[k]);
    if (P.owner) free(P.owner);
    if (P.bg_fill) free(P.bg_fill);
    P.owner = nullptr; P.bg_fill = nullptr; P.ready = false;
}
inline void parallax_free(Parallax& P) {
    parallax_free_runtime(P);
    for (int k = 0; k < PLANE_MAX; ++k) plane_free(P.pl[k]);
    P.n = 0;
}
inline bool plane_ensure_mask(Plane& p, void* (*alloc)(size_t)) {
    if (p.mask) return true;
    p.mask = (uint8_t*)alloc(MASK_BYTES);
    if (p.mask) memset(p.mask, 0, MASK_BYTES);
    return p.mask != nullptr;
}
inline void plane_copy_def(Plane& dst, const Plane& src, void* (*alloc)(size_t)) {
    plane_free(dst);
    uint8_t* keep = nullptr;
    dst = Plane();
    dst.depth = src.depth; dst.ramp = src.ramp; dst.depth_b = src.depth_b;
    dst.ax = src.ax; dst.ay = src.ay; dst.bx = src.bx; dst.by = src.by;
    if (src.mask && plane_ensure_mask(dst, alloc)) memcpy(dst.mask, src.mask, MASK_BYTES);
    (void)keep;
}
inline bool plane_any(const Plane& p) {
    if (!p.mask) return false;
    for (int i = 0; i < MASK_BYTES; ++i) if (p.mask[i]) return true;
    return false;
}

inline float plane_depth_at(const Plane& p, float x, float y) {
    float d = p.c0 + p.cx * x + p.cy * y;
    return d < p.dlo ? p.dlo : (d > p.dhi ? p.dhi : d);
}

// ---- build: owner map, coverage, fills ---------------------------------------------------------------
inline bool bget_(const uint8_t* m, int i) { return (m[i >> 3] >> (i & 7)) & 1; }
inline void bset_(uint8_t* m, int i) { m[i >> 3] |= (uint8_t)(1 << (i & 7)); }
// Nearest marked pixel for every pixel (marked: nx ≥ 0 holds its own coordinates; others nx = −1):
// two-pass propagation of coordinates over 8 neighbours.
inline void par_nearest(int16_t* nx, int16_t* ny) {
    auto d2 = [&](int x, int y, int j) { if (nx[j] < 0) return 0x7fffffff; int dx = nx[j] - x, dy = ny[j] - y; return dx * dx + dy * dy; };
    auto relax = [&](int x, int y, int j) {
        int i = y * IMG_W + x;
        int c = d2(x, y, i), e = d2(x, y, j);
        if (e < c) { nx[i] = nx[j]; ny[i] = ny[j]; }
    };
    for (int y = 0; y < IMG_H; ++y) {
        for (int x = 0; x < IMG_W; ++x) {
            if (y > 0) { int r = (y - 1) * IMG_W; if (x > 0) relax(x, y, r + x - 1); relax(x, y, r + x); if (x < IMG_W - 1) relax(x, y, r + x + 1); }
            if (x > 0) relax(x, y, y * IMG_W + x - 1);
        }
        for (int x = IMG_W - 2; x >= 0; --x) relax(x, y, y * IMG_W + x + 1);
    }
    for (int y = IMG_H - 1; y >= 0; --y) {
        for (int x = IMG_W - 1; x >= 0; --x) {
            if (y < IMG_H - 1) { int r = (y + 1) * IMG_W; if (x < IMG_W - 1) relax(x, y, r + x + 1); relax(x, y, r + x); if (x > 0) relax(x, y, r + x - 1); }
            if (x < IMG_W - 1) relax(x, y, y * IMG_W + x + 1);
        }
        for (int x = 1; x < IMG_W; ++x) relax(x, y, y * IMG_W + x - 1);
    }
}

// Nearest own pixel for every pixel (two-pass propagation of coordinates, 8 neighbours), then each
// pixel outside takes the colour mirrored across that nearest edge point (the texture carries on).
inline void par_fill(const uint8_t* owner, uint8_t id, const uint16_t* photo, bool sw, uint16_t* out,
                     int16_t* nx, int16_t* ny) {
    const int N = IMG_W * IMG_H;
    auto px = [&](int i) -> uint16_t { uint16_t v = photo[i]; return sw ? (uint16_t)((v >> 8) | (v << 8)) : v; };
    bool any = false;
    for (int i = 0; i < N; ++i) {
        if (owner[i] == id) { nx[i] = (int16_t)(i % IMG_W); ny[i] = (int16_t)(i / IMG_W); any = true; }
        else nx[i] = -1;
    }
    if (!any) { for (int i = 0; i < N; ++i) out[i] = px(i); return; }
    par_nearest(nx, ny);
    for (int i = 0; i < N; ++i) {
        if (owner[i] == id) { out[i] = px(i); continue; }
        int x = i % IMG_W, y = i / IMG_W, ex = nx[i], ey = ny[i];
        int mx = 2 * ex - x, my = 2 * ey - y;           // mirrored across the edge
        if (mx >= 0 && my >= 0 && mx < IMG_W && my < IMG_H && owner[my * IMG_W + mx] == id) out[i] = px(my * IMG_W + mx);
        else out[i] = px(ey * IMG_W + ex);
    }
}

inline int parallax_max_shift(const Parallax& P) {
    float f = P.prm[PP_FOCUS], m = 0;
    for (int k = 0; k < P.n; ++k) {
        float a = fabsf(P.pl[k].dlo - f), b = fabsf(P.pl[k].dhi - f);
        if (a > m) m = a;
        if (b > m) m = b;
    }
    if (f > m) m = f;                                // the background, if it's not the pinned depth
    if (P.grid && 1.0f - f > m) m = 1.0f - f;        // the ground, nearest to the horizon
    float vert = P.prm[PP_VERT] > 1.0f ? P.prm[PP_VERT] : 1.0f;
    return (int)ceilf(m * P.prm[PP_AMOUNT] * vert * 1.05f);
}

// Build everything the renderer needs. `photo` is the picture (sw = byte-swapped).
inline bool parallax_prepare(Parallax& P, const uint16_t* photo, bool sw, void* (*alloc)(size_t)) {
    parallax_free_runtime(P);
    if ((P.n <= 0 && !P.grid) || !photo) return false;
    const int N = IMG_W * IMG_H;
    P.gp = Plane();
    P.gp.dlo = P.gp.dhi = 0;
    if (P.grid) {
        // Inverse depth of a flat ground is linear on screen and 0 on the horizon: the third row of the
        // grid's inverse homography. Scaled to 1 at the nearest ground the frame shows.
        const WHomog H = water_homog(P.gq);
        const float G = H.d * H.h - H.e * H.g, Hh = -(H.a * H.h - H.b * H.g), I = H.a * H.e - H.b * H.d;
        const float sgn = (G * P.gq[0] + Hh * P.gq[1] + I) < 0 ? -1.0f : 1.0f;
        float wmax = 0;
        const float cx_[4] = {0, (float)IMG_W, 0, (float)IMG_W}, cy_[4] = {0, 0, (float)IMG_H, (float)IMG_H};
        for (int c = 0; c < 4; ++c) { float w = sgn * (G * cx_[c] + Hh * cy_[c] + I); if (w > wmax) wmax = w; }
        if (wmax > 1e-9f) {
            P.gp.c0 = sgn * I / wmax; P.gp.cx = sgn * G / wmax; P.gp.cy = sgn * Hh / wmax;
            P.gp.dlo = 0; P.gp.dhi = 1;
        }
        for (int k = 0; k < P.n; ++k) {               // a card stands where its bottom meets the ground
            Plane& p = P.pl[k];
            if (!p.mask) continue;
            int yb = -1; long sx = 0; int nx_ = 0;
            for (int y = IMG_H - 1; y >= 0 && yb < 0; --y)
                for (int x = 0; x < IMG_W; ++x) if (mask_get(p.mask, x, y)) { yb = y; sx += x; ++nx_; }
            if (yb < 0) continue;
            p.depth = plane_depth_at(P.gp, (float)sx / nx_, (float)yb);
            p.ramp = false;
        }
    }
    for (int k = 0; k < P.n; ++k) {
        Plane& p = P.pl[k];
        p.c0 = p.depth; p.cx = p.cy = 0; p.dlo = p.dhi = p.depth;
        if (p.ramp) {
            float dx = p.bx - p.ax, dy = p.by - p.ay, l2 = dx * dx + dy * dy;
            if (l2 > 1.0f) {
                float g = (p.depth_b - p.depth) / l2;
                p.cx = g * dx; p.cy = g * dy; p.c0 = p.depth - p.cx * p.ax - p.cy * p.ay;
                p.dlo = p.depth < p.depth_b ? p.depth : p.depth_b;
                p.dhi = p.depth < p.depth_b ? p.depth_b : p.depth;
            }
        }
    }
    // nearest first (by the plane's nearest depth)
    for (int k = 0; k < P.n; ++k) P.order[k] = k;
    for (int a = 1; a < P.n; ++a) {
        int t = P.order[a], b = a - 1;
        while (b >= 0 && P.pl[P.order[b]].dhi < P.pl[t].dhi) { P.order[b + 1] = P.order[b]; --b; }
        P.order[b + 1] = t;
    }
    P.owner = (uint8_t*)alloc(N);
    P.bg_fill = (uint16_t*)alloc((size_t)N * 2);
    int16_t* nx = (int16_t*)alloc((size_t)N * 2);
    int16_t* ny = (int16_t*)alloc((size_t)N * 2);
    bool ok = P.owner && P.bg_fill && nx && ny;
    for (int k = 0; k < P.n && ok; ++k) {
        P.pl[k].cov = (uint8_t*)alloc(MASK_BYTES);
        P.pl[k].fill = (uint16_t*)alloc((size_t)N * 2);
        ok = P.pl[k].cov && P.pl[k].fill && P.pl[k].mask;
    }
    if (!ok) { if (nx) free(nx); if (ny) free(ny); parallax_free_runtime(P); return false; }
    // owner: the nearest plane painted over each pixel
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x) {
            int best = 0;
            float bd = -1;
            for (int k = 0; k < P.n; ++k) {
                if (!mask_get(P.pl[k].mask, x, y)) continue;
                float d = plane_depth_at(P.pl[k], (float)x, (float)y);
                if (d > bd) { bd = d; best = k + 1; }
            }
            P.owner[y * IMG_W + x] = (uint8_t)best;
        }
    P.margin = parallax_max_shift(P) + 3;
    par_fill(P.owner, 0, photo, sw, P.bg_fill, nx, ny);
    for (int k = 0; k < P.n; ++k) {
        Plane& p = P.pl[k];
        memset(p.cov, 0, MASK_BYTES);
        for (int i = 0; i < N; ++i) if (P.owner[i] == k + 1) bset_(p.cov, i);
    }
    // What's behind each plane: every pixel it hides continues the nearest surface that's farther than
    // it (sky behind an arm stays sky, sea stays sea), out to the most it can ever be uncovered.
    const int M2 = P.margin * P.margin;
    for (int o = 0; o < P.n; ++o) {
        const float dov = P.pl[o].dhi;
        for (int i = 0; i < N; ++i) {
            const int w = P.owner[i];
            const bool farther = w == 0 || (w != o + 1 && P.pl[w - 1].dhi < dov);
            if (farther) { nx[i] = (int16_t)(i % IMG_W); ny[i] = (int16_t)(i / IMG_W); } else nx[i] = -1;
        }
        par_nearest(nx, ny);
        for (int i = 0; i < N; ++i) {
            if (P.owner[i] != o + 1 || nx[i] < 0) continue;
            const int x = i % IMG_W, y = i / IMG_W, dx = nx[i] - x, dy = ny[i] - y;
            if (dx * dx + dy * dy > M2) continue;
            const int behind = P.owner[ny[i] * IMG_W + nx[i]];
            if (behind > 0) bset_(P.pl[behind - 1].cov, i);
        }
    }
    for (int k = 0; k < P.n; ++k) {                   // planes painted on past their edge keep that too
        Plane& p = P.pl[k];
        for (int i = 0; i < N; ++i)
            if (P.owner[i] > 0 && P.owner[i] != k + 1 && P.pl[P.owner[i] - 1].dhi >= p.dhi && bget_(p.mask, i)) bset_(p.cov, i);
        par_fill(P.owner, (uint8_t)(k + 1), photo, sw, p.fill, nx, ny);
        p.x0 = IMG_W; p.y0 = IMG_H; p.x1 = 0; p.y1 = 0;
        for (int i = 0; i < N; ++i) {
            if (!bget_(p.cov, i)) continue;
            const int x = i % IMG_W, y = i / IMG_W;
            if (x < p.x0) p.x0 = x;
            if (y < p.y0) p.y0 = y;
            if (x + 1 > p.x1) p.x1 = x + 1;
            if (y + 1 > p.y1) p.y1 = y + 1;
        }
    }
    free(nx); free(ny);
    P.zoom = P.grid ? 1.0f + 2.0f * (float)(parallax_max_shift(P) + 1) / IMG_W : 1.0f;
    for (int x = 0; x < IMG_W; ++x) P.zx[x] = (int16_t)lroundf(((x - IMG_W * 0.5f) / P.zoom + IMG_W * 0.5f) * 16.0f);
    for (int y = 0; y < IMG_H; ++y) P.zy[y] = (int16_t)lroundf(((y - IMG_H * 0.5f) / P.zoom + IMG_H * 0.5f) * 16.0f);
    P.ready = true;
    P.drawn = false;
    return true;
}

// ---- view ------------------------------------------------------------------------------------------
// rate_x: turning about the screen's horizontal axis (top edge away = +), rate_y: about its vertical
// axis (right edge away = +), deg/s. Returns true while the view is moving.
inline bool parallax_step(Parallax& P, float dt, float rate_x, float rate_y, bool have) {
    P.t += dt;
    const float D2R = 0.017453293f;
    if (have) {
        const float kb = 1.0f - expf(-dt / PAR_BIAS_S);
        if (fabsf(rate_x - P.bias[0]) < 3.0f && fabsf(rate_y - P.bias[1]) < 3.0f) {   // still: learn the gyro's offset
            P.bias[0] += (rate_x - P.bias[0]) * kb;
            P.bias[1] += (rate_y - P.bias[1]) * kb;
        }
        auto dead = [](float r) { return fabsf(r) < PAR_DEAD_DPS ? 0.0f : (r > 0 ? r - PAR_DEAD_DPS : r + PAR_DEAD_DPS); };
        P.th[0] += dead(rate_y - P.bias[1]) * D2R * dt;   // right edge away → near planes slide right
        P.th[1] += -dead(rate_x - P.bias[0]) * D2R * dt;  // top edge away → near planes slide up
    }
    const float leak = expf(-dt / PAR_RECENTER_S);
    P.th[0] *= leak; P.th[1] *= leak;
    const float full = P.prm[PP_TILT] * D2R;
    float tg[2];
    for (int a = 0; a < 2; ++a) {
        float u = 1.5f * P.th[a] / full;
        tg[a] = u / sqrtf(1.0f + u * u);                // soft limit
    }
    const float amb = P.prm[PP_AMBIENT];
    if (amb > 0) {                                      // slow, never-repeating-looking drift
        const float T = 6.2831853f * P.t;
        tg[0] += amb * (0.55f * sinf(T / 11.0f) + 0.22f * sinf(T / 4.3f + 1.0f));
        tg[1] += amb * (0.40f * sinf(T / 8.3f + 2.0f) + 0.15f * sinf(T / 3.7f + 0.5f));
    }
    const float w = 9.0f * P.prm[PP_RESPONSE];
    int steps = (int)ceilf(dt * w / 0.25f);
    if (steps < 1) steps = 1;
    const float h = dt / steps;
    for (int s = 0; s < steps; ++s)
        for (int a = 0; a < 2; ++a) {                   // critically damped spring: weight without wobble
            P.vv[a] += (w * w * (tg[a] - P.v[a]) - 2.0f * w * P.vv[a]) * h;
            P.v[a] += P.vv[a] * h;
        }
    const float A = P.prm[PP_AMOUNT];
    if (amb <= 0 && fabsf(tg[0]) + fabsf(tg[1]) < 0.002f && fabsf(P.v[0]) + fabsf(P.v[1]) < 0.012f &&
        fabsf(P.vv[0]) + fabsf(P.vv[1]) < 0.02f) {      // settled: exactly centred (the untouched picture)
        P.v[0] = P.v[1] = P.vv[0] = P.vv[1] = 0;
        if (fabsf(P.th[0]) + fabsf(P.th[1]) < 1e-3f) P.th[0] = P.th[1] = 0;
    }
    float moved = (fabsf(P.v[0] - P.dv[0]) + fabsf(P.v[1] - P.dv[1])) * A;
    return !P.drawn || moved > 1.0f / 128.0f;
}

// Per-frame slide of each plane: shift = (depth − focus) · (kx, ky)  (px, + = right / down)
inline void parallax_k(const Parallax& P, float& kx, float& ky) {
    const float s = P.prm[PP_INVERT] >= 0.5f ? -1.0f : 1.0f, A = P.prm[PP_AMOUNT];
    kx = s * A * P.v[0];
    ky = s * A * P.v[1] * P.prm[PP_VERT];
}

// Screen rect (x0, y0, x1, y1) a plane can touch between the last drawn view and this one.
inline void parallax_plane_rect(const Parallax& P, int k, int* r) {
    const Plane& p = P.pl[k];
    const float f = P.prm[PP_FOCUS], A = P.prm[PP_AMOUNT];
    const float s = P.prm[PP_INVERT] >= 0.5f ? -1.0f : 1.0f;
    float m = fabsf(p.dlo - f) > fabsf(p.dhi - f) ? fabsf(p.dlo - f) : fabsf(p.dhi - f);
    float ex = 0, ey = 0;
    for (int a = 0; a < 2; ++a) {
        const float* vw = a ? P.v : P.dv;
        float sx = fabsf(s * A * vw[0]) * m, sy = fabsf(s * A * vw[1] * P.prm[PP_VERT]) * m;
        if (sx > ex) ex = sx;
        if (sy > ey) ey = sy;
    }
    r[0] = p.x0 - (int)ceilf(ex) - 3; r[1] = p.y0 - (int)ceilf(ey) - 3;
    r[2] = p.x1 + (int)ceilf(ex) + 3; r[3] = p.y1 + (int)ceilf(ey) + 3;
    if (r[0] < 0) r[0] = 0;
    if (r[1] < 0) r[1] = 0;
    if (r[2] > IMG_W) r[2] = IMG_W;
    if (r[3] > IMG_H) r[3] = IMG_H;
}

// ---- render ----------------------------------------------------------------------------------------
// Content: what the planes show of themselves (the photo with water / jiggle drawn in). When `live`,
// a plane's own pixels come from it; everything else (and everything when not live) from the fills.
struct ParFrame {
    const Parallax* P;
    const uint16_t* anim;
    bool anim_sw, live;
    float kx, ky;
    // per layer (index 0..n-1 = draw order nearest first, n = background)
    int   nl;
    uint8_t id[PLANE_MAX + 1];
    const uint16_t* fill[PLANE_MAX + 1];
    const uint8_t*  cov[PLANE_MAX + 1];
    bool  flat[PLANE_MAX + 1];
    int32_t sx[PLANE_MAX + 1], sy[PLANE_MAX + 1];       // flat: shift, Q8
    const Plane* pl[PLANE_MAX + 1];
};

inline void parallax_frame(ParFrame& F, const Parallax& P, const uint16_t* anim, bool anim_sw, bool live) {
    F.P = &P; F.anim = anim; F.anim_sw = anim_sw; F.live = live;
    parallax_k(P, F.kx, F.ky);
    const float f = P.prm[PP_FOCUS];
    F.nl = P.n + 1;
    for (int j = 0; j < P.n; ++j) {
        const Plane& p = P.pl[P.order[j]];
        F.id[j] = (uint8_t)(P.order[j] + 1);
        F.fill[j] = p.fill; F.cov[j] = p.cov; F.pl[j] = &p;
        F.flat[j] = p.dlo == p.dhi;
        F.sx[j] = (int32_t)lroundf((p.depth - f) * F.kx * 256.0f);
        F.sy[j] = (int32_t)lroundf((p.depth - f) * F.ky * 256.0f);
    }
    const int b = P.n;
    F.id[b] = 0; F.fill[b] = P.bg_fill; F.cov[b] = nullptr; F.pl[b] = &P.gp; F.flat[b] = !P.grid;
    F.sx[b] = (int32_t)lroundf(-f * F.kx * 256.0f);
    F.sy[b] = (int32_t)lroundf(-f * F.ky * 256.0f);
}

inline int par_clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Render screen rows [y0, y1) × columns [x0, x1) into out (display byte order, width x1 − x0).
inline void parallax_render(const ParFrame& F, uint16_t* out, int x0, int y0, int x1, int y1) {
    const auto& W = par_w();
    const int ow = x1 - x0;
    const uint8_t* owner = F.P->owner;
    const float f = F.P->prm[PP_FOCUS];
    for (int y = y0; y < y1; ++y) {
        uint16_t* o = out + (y - y0) * ow;
        for (int x = x0; x < x1; ++x) {
            int rem = 256, ar = 0, ag = 0, ab = 0;
            for (int j = 0; j < F.nl && rem > 0; ++j) {
                int32_t SX, SY;
                if (F.flat[j]) { SX = F.sx[j]; SY = F.sy[j]; }
                else {
                    float d = plane_depth_at(*F.pl[j], (float)x, (float)y) - f;
                    SX = (int32_t)(d * F.kx * 256.0f); SY = (int32_t)(d * F.ky * 256.0f);
                }
                const int32_t QX = ((int32_t)F.P->zx[x] << 4) - SX, QY = ((int32_t)F.P->zy[y] << 4) - SY;
                const int ix = QX >> 8, iy = QY >> 8, fx = QX & 255, fy = QY & 255;
                int a = 256;
                if (F.cov[j]) {                         // sub-pixel coverage of the plane's silhouette
                    const int xa = par_clampi(ix, 0, IMG_W - 1), xb = par_clampi(ix + 1, 0, IMG_W - 1);
                    const int ya = par_clampi(iy, 0, IMG_H - 1), yb = par_clampi(iy + 1, 0, IMG_H - 1);
                    const int c00 = mask_get(F.cov[j], xa, ya), c10 = mask_get(F.cov[j], xb, ya);
                    const int c01 = mask_get(F.cov[j], xa, yb), c11 = mask_get(F.cov[j], xb, yb);
                    if (!(c00 | c10 | c01 | c11)) continue;
                    if (!(c00 & c10 & c01 & c11))
                        a = ((c00 * (256 - fx) + c10 * fx) * (256 - fy) + (c01 * (256 - fx) + c11 * fx) * fy) >> 8;
                    if (a <= 0) continue;
                }
                // colour: separable Catmull-Rom over the plane's content
                const int16_t* wx = W[fx >> 2];
                const int16_t* wy = W[fy >> 2];
                int xs[4];
                for (int k = 0; k < 4; ++k) xs[k] = par_clampi(ix - 1 + k, 0, IMG_W - 1);
                const uint8_t id = F.id[j];
                const uint16_t* fill = F.fill[j];
                int R = 0, G = 0, B = 0;
                for (int r = 0; r < 4; ++r) {
                    if (!wy[r]) continue;
                    const int row = par_clampi(iy - 1 + r, 0, IMG_H - 1) * IMG_W;
                    int hr = 0, hg = 0, hb = 0;
                    for (int k = 0; k < 4; ++k) {
                        if (!wx[k]) continue;
                        const int i = row + xs[k];
                        uint16_t v;
                        if (F.live && owner[i] == id) { v = F.anim[i]; if (F.anim_sw) v = (uint16_t)((v >> 8) | (v << 8)); }
                        else v = fill[i];
                        hr += wx[k] * (v >> 11); hg += wx[k] * ((v >> 5) & 63); hb += wx[k] * (v & 31);
                    }
                    R += wy[r] * hr; G += wy[r] * hg; B += wy[r] * hb;
                }
                R = par_clampi(R >> 8, 0, 31 << 8); G = par_clampi(G >> 8, 0, 63 << 8); B = par_clampi(B >> 8, 0, 31 << 8);
                const int wgt = a * rem >> 8;           // front to back
                ar += R * wgt; ag += G * wgt; ab += B * wgt;
                rem -= wgt;
                if (a >= 256) rem = 0;
            }
            const int r5 = (ar + 32768) >> 16, g6 = (ag + 32768) >> 16, b5 = (ab + 32768) >> 16;
            const uint16_t c = (uint16_t)((par_clampi(r5, 0, 31) << 11) | (par_clampi(g6, 0, 63) << 5) | par_clampi(b5, 0, 31));
            o[x - x0] = (uint16_t)((c >> 8) | (c << 8));
        }
    }
}

// Where on the picture a tap on the screen lands (the nearest plane showing there).
inline void parallax_to_content(const Parallax& P, int x, int y, int& qx, int& qy) {
    qx = x; qy = y;
    if (!P.ready) return;
    float kx, ky;
    parallax_k(P, kx, ky);
    const float f = P.prm[PP_FOCUS];
    for (int j = 0; j < P.n; ++j) {
        const Plane& p = P.pl[P.order[j]];
        float d = plane_depth_at(p, (float)x, (float)y) - f;
        const float zx = (x - IMG_W * 0.5f) / P.zoom + IMG_W * 0.5f, zy = (y - IMG_H * 0.5f) / P.zoom + IMG_H * 0.5f;
        int cx = (int)lroundf(zx - d * kx), cy = (int)lroundf(zy - d * ky);
        if (mask_get(p.cov, cx, cy)) { qx = cx; qy = cy; return; }
    }
    const float zx = (x - IMG_W * 0.5f) / P.zoom + IMG_W * 0.5f, zy = (y - IMG_H * 0.5f) / P.zoom + IMG_H * 0.5f;
    const float d = (P.grid ? plane_depth_at(P.gp, (float)x, (float)y) : 0.0f) - f;
    qx = (int)lroundf(zx - d * kx); qy = (int)lroundf(zy - d * ky);
}

// ---- .jig lines: "parallax k=v…", "plane depth=… [ramp=ax,ay,bx,by,depth_b]", "prow y a-b …" -----------
inline bool parse_parallax_line(char* line, Parallax& P, int& cur, void* (*alloc)(size_t)) {
    char kind[10];
    int used = 0, n = 0;
    if (sscanf(line, "%9s%n", kind, &used) != 1) return false;
    char* rest = line + used;
    char tok[48];
    if (strcmp(kind, "parallax") == 0) {
        while (sscanf(rest, "%47s%n", tok, &n) == 1) {
            rest += n;
            char* eq = strchr(tok, '=');
            if (!eq) continue;
            *eq = 0;
            for (int p = 0; p < PP_COUNT; ++p)
                if (strcmp(tok, pparam_info((uint8_t)p).key) == 0) {
                    char* end = nullptr;
                    float v = strtof(eq + 1, &end);
                    const PParamInfo& pi = pparam_info((uint8_t)p);
                    if (end != eq + 1 && !isnan(v)) P.prm[p] = v < pi.lo ? pi.lo : (v > pi.hi ? pi.hi : v);
                }
        }
        return true;
    }
    if (strcmp(kind, "plane") == 0) {
        cur = -1;
        if (P.n >= PLANE_MAX) return true;
        Plane& p = P.pl[P.n];
        plane_free(p);
        p = Plane();
        if (!plane_ensure_mask(p, alloc)) return true;
        while (sscanf(rest, "%47s%n", tok, &n) == 1) {
            rest += n;
            float d, db;
            int ax, ay, bx, by;
            if (sscanf(tok, "depth=%f", &d) == 1 && !isnan(d)) p.depth = d < 0 ? 0 : (d > 1 ? 1 : d);
            else if (sscanf(tok, "ramp=%d,%d,%d,%d,%f", &ax, &ay, &bx, &by, &db) == 5 && !isnan(db)) {
                p.ramp = true;
                p.ax = (int16_t)ax; p.ay = (int16_t)ay; p.bx = (int16_t)bx; p.by = (int16_t)by;
                p.depth_b = db < 0 ? 0 : (db > 1 ? 1 : db);
            }
        }
        cur = P.n++;
        return true;
    }
    if (strcmp(kind, "pgrid") == 0) {                 // the ground grid: 4 corners
        float q[8]; int got = 0;
        for (int k = 0; k < 4; ++k) { float x, y; if (sscanf(rest, " %f,%f%n", &x, &y, &n) != 2) break; rest += n; q[2 * k] = x; q[2 * k + 1] = y; ++got; }
        if (got == 4) { P.grid = true; memcpy(P.gq, q, sizeof(q)); }
        return true;
    }
    if (strcmp(kind, "prow") == 0) {
        if (cur < 0) return true;
        int y;
        if (sscanf(rest, "%d%n", &y, &n) != 1) return true;
        rest += n;
        int a, b;
        while (sscanf(rest, " %d-%d%n", &a, &b, &n) == 2) { rest += n; mask_span(P.pl[cur].mask, y, a, b, true); }
        return true;
    }
    return false;
}

template <typename F>
inline void save_mask_rows(F& f, const uint8_t* m, const char* tag) {
    for (int y = 0; y < IMG_H; ++y) {
        int x = 0;
        bool row = false;
        while (x < IMG_W) {
            while (x < IMG_W && !mask_get(m, x, y)) ++x;
            if (x >= IMG_W) break;
            int a = x;
            while (x < IMG_W && mask_get(m, x, y)) ++x;
            if (!row) { f.printf("%s %d", tag, y); row = true; }
            f.printf(" %d-%d", a, x);
        }
        if (row) f.printf("\n");
    }
}

template <typename F>
inline void save_parallax(F& f, const Parallax& P) {
    bool any = false;
    for (int k = 0; k < P.n; ++k) any |= plane_any(P.pl[k]);
    if (!any && !P.grid) return;
    f.printf("parallax");
    for (int p = 0; p < PP_COUNT; ++p)
        if (fabsf(P.prm[p] - pparam_info((uint8_t)p).dflt) > 1e-6f) f.printf(" %s=%.3g", pparam_info((uint8_t)p).key, P.prm[p]);
    f.printf("\n");
    if (P.grid) f.printf("pgrid %.1f,%.1f %.1f,%.1f %.1f,%.1f %.1f,%.1f\n", P.gq[0], P.gq[1], P.gq[2], P.gq[3], P.gq[4], P.gq[5], P.gq[6], P.gq[7]);
    for (int k = 0; k < P.n; ++k) {
        const Plane& p = P.pl[k];
        if (!plane_any(p)) continue;
        f.printf("plane depth=%.3g", p.depth);
        if (p.ramp) f.printf(" ramp=%d,%d,%d,%d,%.3g", p.ax, p.ay, p.bx, p.by, p.depth_b);
        f.printf("\n");
        save_mask_rows(f, p.mask, "prow");
    }
}

}  // namespace jig
