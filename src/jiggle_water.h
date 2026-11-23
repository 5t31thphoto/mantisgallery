/**
 * Water for landscape photos — still water, rivers, ocean waves. Moves the photo's own pixels inside a
 * mask; the banks never move. No Arduino dependencies (built and tested on a PC too; mirrored in
 * tools/jiggle_lab.py).
 *
 *  - Mask: a 1-bit map of the water (painted / polygons), stored in the .jig as row spans.
 *  - Edge hold: motion fades to zero over the last `edge` px inside the mask; samples that would land
 *    outside the mask are pulled back in, so bank pixels never get smeared into the water.
 *  - Still water: subtle ambient waves (smaller toward a horizon), a wave-equation surface for taps,
 *    rain and shakes, and the whole body sloshing with real gravity.
 *  - River: curved flow lines → a flow direction per spot; the texture streams along them (two phases
 *    cross-faded, so it loops with no visible jump). Ripples ride on top.
 *  - Ocean: a curved shore line; swells roll in toward it, bigger up close, with sets and run-up.
 */
#pragma once
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <new>
#if defined(ESP_PLATFORM)
#include <esp_heap_caps.h>
#endif
#include "jiggle_core.h"

namespace jig {

enum WaterMode : uint8_t { W_STILL = 0, W_RIVER = 1, W_OCEAN = 2, W_MODE_COUNT = 3 };
inline const char* water_mode_name(uint8_t m) {
    return m == W_RIVER ? "river" : (m == W_OCEAN ? "ocean" : "still");
}
inline const char* water_mode_label(uint8_t m) {
    return m == W_RIVER ? "River" : (m == W_OCEAN ? "Ocean" : "Still");
}

static constexpr int   WATER_MAX = 2;          // water regions per photo
static constexpr int   WLINE_MAX = 3;          // flow / shore curves per region
static constexpr int   WPTS_MAX = 24;          // points per curve
static constexpr int   WCELL = 4;              // surface simulation cell (px)
static constexpr int   MASK_BYTES = IMG_W * IMG_H / 8;

// Dials (saved as key=value on the "water" line; unset = default).
enum WParam : uint8_t {
    WP_AMB, WP_RIPPLE, WP_SLOSH, WP_RAIN, WP_RATE, WP_EDGE, WP_HORIZON,
    WP_SPEED, WP_SIZE, WP_SPACING, WP_SETS, WP_GLINT, WP_LIGHT, WP_FOAM, WP_COUNT
};
struct WParamInfo { const char* key; const char* label; float lo, hi, dflt; };
inline const WParamInfo& wparam_info(uint8_t p) {
    static const WParamInfo k[WP_COUNT] = {
        {"amb", "Ambient motion", 0.0f, 3.0f, 1.0f},
        {"ripple", "Ripple strength", 0.0f, 3.0f, 1.0f},
        {"slosh", "Tilt / shake response", 0.0f, 3.0f, 1.0f},
        {"rain", "Rain", 0.0f, 1.0f, 0.0f},
        {"rate", "Rain amount", 0.1f, 4.0f, 1.0f},
        {"edge", "Edge hold (px)", 0.0f, 16.0f, 3.0f},
        {"horizon", "Horizon (y px, -1 = top of water)", -1.0f, 239.0f, -1.0f},
        {"speed", "Flow / wave speed", 0.1f, 4.0f, 1.0f},
        {"size", "Wave size", 0.1f, 4.0f, 1.0f},
        {"spacing", "Wave spacing (px)", 8.0f, 160.0f, 48.0f},
        {"sets", "Bigger sets", 0.0f, 1.0f, 0.5f},
        {"glint", "Sun glints", 0.0f, 1.0f, 0.4f},
        {"light", "Painted wave light", 0.0f, 1.0f, 0.0f},   // for flat / stylised art; photos move their own
        {"foam", "Painted shore foam", 0.0f, 1.0f, 0.0f},    // texture, so both are off by default
    };
    return k[p < WP_COUNT ? p : 0];
}

struct WLine { uint8_t n = 0; int16_t x[WPTS_MAX], y[WPTS_MAX]; };

struct Water {
    // ---- definition (saved)
    uint8_t mode = W_STILL;
    char    name[16] = {0};                     // the Studio's layer name (kept through a Core2 save)
    float   prm[WP_COUNT];
    uint8_t nlines = 0;
    WLine   line[WLINE_MAX];
    uint8_t* mask = nullptr;                    // MASK_BYTES, bit (x,y) = water
    bool    grid = false;                       // perspective grid on the surface:
    float   gq[8] = {0};                        //   corners near-left, near-right, far-right, far-left (x, y)

    // ---- runtime (water_prepare)
    int bx = 0, by = 0, bw = 0, bh = 0, area = 0;
    uint8_t* edge = nullptr;                     // bw*bh: 0 outside, 1..255 motion weight inside
    uint8_t* dsafe = nullptr;                    // bw*bh: px to the nearest bank (a shift shorter can't reach it)
    uint16_t* sp_off = nullptr;                  // bh+1: where each row's water spans start in sp
    int16_t* sp = nullptr;                       // spans: x0, x1 pairs (x1 exclusive)
    int gw = 0, gh = 0;                          // cell grid (one cell border all round)
    float *h0 = nullptr, *h1 = nullptr;          // surface heights (two time levels)
    float *gx = nullptr, *gy = nullptr;          // surface slope after the latest step
    float *gxp = nullptr, *gyp = nullptr;        // …and the step before (drawn blended between them)
    float *sgx = nullptr, *sgy = nullptr;        // the blend as it stood when this frame began (water_snapshot)
    uint8_t* cin = nullptr;                      // cell is water
    int8_t *cnx = nullptr, *cny = nullptr;       // outward normal at the water's edge (else 0)
    int32_t* bank = nullptr;                     // the edge cells (cnx | cny), listed once …
    float* bank_k = nullptr;                     // … and how hard each throws a wave (far banks smaller: √(2·pax))
    int nbank = 0;
    float *fx = nullptr, *fy = nullptr;          // river: flow (px/s) per cell
    float *od = nullptr, *onx = nullptr, *ony = nullptr;   // ocean: distance to shore, unit toward it
    float t = 0, sx = 0, sy = 0, svx = 0, svy = 0, rain_acc = 0, set_ph = 0;
    float sim_acc = 0, alpha = 0;                // fixed-rate surface: time since the last step, blend
    float* pax = nullptr;                        // perspective (grid): per cell ½·(across scale)², ½·(depth scale)²
    float* pay = nullptr;                        //   — far water: slower, smaller, foreshortened ripples
    float ghz = -1.0f;                           // the grid's horizon (vanishing line), y px
    bool  oworld = false;                        // ocean distances are on the water's surface (grid), not on screen
    uint32_t rng = 12345;

    Water() { for (int i = 0; i < WP_COUNT; ++i) prm[i] = wparam_info((uint8_t)i).dflt; }
};

inline float wprm(const Water& w, uint8_t p) { return w.prm[p]; }

// ---- mask bits ----------------------------------------------------------------------------------
inline bool mask_get(const uint8_t* m, int x, int y) {
    if (!m || x < 0 || y < 0 || x >= IMG_W || y >= IMG_H) return false;
    int i = y * IMG_W + x;
    return (m[i >> 3] >> (i & 7)) & 1;
}
inline void mask_put(uint8_t* m, int x, int y, bool v) {
    if (!m || x < 0 || y < 0 || x >= IMG_W || y >= IMG_H) return;
    int i = y * IMG_W + x;
    if (v) m[i >> 3] |= (uint8_t)(1 << (i & 7)); else m[i >> 3] &= (uint8_t)~(1 << (i & 7));
}
inline void mask_span(uint8_t* m, int y, int x0, int x1, bool v) {   // [x0, x1)
    if (y < 0 || y >= IMG_H) return;
    if (x0 < 0) x0 = 0;
    if (x1 > IMG_W) x1 = IMG_W;
    for (int x = x0; x < x1; ++x) mask_put(m, x, y, v);
}
// Round brush (paint v = true, erase v = false).
inline void mask_brush(uint8_t* m, int cx, int cy, int r, bool v) {
    for (int dy = -r; dy <= r; ++dy) {
        int hw = (int)sqrtf((float)(r * r - dy * dy));
        mask_span(m, cy + dy, cx - hw, cx + hw + 1, v);
    }
}
// Filled polygon (even-odd), sampled at pixel centres.
inline void mask_poly(uint8_t* m, const int16_t* xs, const int16_t* ys, int n, bool v) {
    if (n < 3) return;
    int ymin = IMG_H, ymax = -1;
    for (int i = 0; i < n; ++i) { if (ys[i] < ymin) ymin = ys[i]; if (ys[i] > ymax) ymax = ys[i]; }
    if (ymin < 0) ymin = 0;
    if (ymax > IMG_H - 1) ymax = IMG_H - 1;
    float xs_[64];
    for (int y = ymin; y <= ymax; ++y) {
        float yc = y + 0.5f;
        int k = 0;
        for (int i = 0, j = n - 1; i < n && k < 64; j = i++) {
            float yi = ys[i], yj = ys[j];
            if ((yi > yc) != (yj > yc)) xs_[k++] = xs[i] + (yc - yi) * (xs[j] - xs[i]) / (yj - yi);
        }
        for (int a = 1; a < k; ++a) { float t = xs_[a]; int b = a - 1; while (b >= 0 && xs_[b] > t) { xs_[b + 1] = xs_[b]; --b; } xs_[b + 1] = t; }
        for (int a = 0; a + 1 < k; a += 2) mask_span(m, y, (int)ceilf(xs_[a] - 0.5f), (int)ceilf(xs_[a + 1] - 0.5f), v);
    }
}

// ---- colour (luma / chroma of RGB565; used by the smart selection) ---------------------------------
struct PickColor { int y, cb, cr; };
inline PickColor pick_color(uint16_t p) {
    int r = (p >> 11) << 3, g = ((p >> 5) & 63) << 2, b = (p & 31) << 3;
    int y = (77 * r + 150 * g + 29 * b) >> 8;
    return {y, b - y, r - y};
}
inline int pick_dist(const PickColor& a, const PickColor& b) {
    return (abs(a.y - b.y) >> 1) + abs(a.cb - b.cb) + abs(a.cr - b.cr);
}
// ---- curves -------------------------------------------------------------------------------------
// Smooth curve through a line's points (Catmull-Rom), as up to `max` samples. Returns the count.
inline int water_curve(const WLine& L, float* ox, float* oy, int max) {
    if (L.n == 0) return 0;
    if (L.n == 1) { ox[0] = L.x[0]; oy[0] = L.y[0]; return 1; }
    int k = 0;
    const int sub = 8;
    for (int i = 0; i + 1 < L.n && k < max; ++i) {
        int i0 = i > 0 ? i - 1 : 0, i3 = i + 2 < L.n ? i + 2 : L.n - 1;
        float x0 = L.x[i0], y0 = L.y[i0], x1 = L.x[i], y1 = L.y[i], x2 = L.x[i + 1], y2 = L.y[i + 1], x3 = L.x[i3], y3 = L.y[i3];
        for (int s = 0; s < sub && k < max; ++s) {
            float t = (float)s / sub, t2 = t * t, t3 = t2 * t;
            ox[k] = 0.5f * (2 * x1 + (-x0 + x2) * t + (2 * x0 - 5 * x1 + 4 * x2 - x3) * t2 + (-x0 + 3 * x1 - 3 * x2 + x3) * t3);
            oy[k] = 0.5f * (2 * y1 + (-y0 + y2) * t + (2 * y0 - 5 * y1 + 4 * y2 - y3) * t2 + (-y0 + 3 * y1 - 3 * y2 + y3) * t3);
            ++k;
        }
    }
    if (k < max) { ox[k] = L.x[L.n - 1]; oy[k] = L.y[L.n - 1]; ++k; }
    return k;
}

// Nearest point on a sampled curve: distance, unit tangent (along the curve), unit toward it.
inline void curve_nearest(const float* cx, const float* cy, int n, float px, float py,
                          float& dist, float& tx, float& ty, float& nx, float& ny) {
    dist = 1e9f; tx = 1; ty = 0; nx = 0; ny = 0;
    if (n == 1) { float dx = cx[0] - px, dy = cy[0] - py; dist = sqrtf(dx * dx + dy * dy); if (dist > 1e-6f) { nx = dx / dist; ny = dy / dist; } return; }
    for (int i = 0; i + 1 < n; ++i) {
        float ax = cx[i], ay = cy[i], bx = cx[i + 1] - ax, by = cy[i + 1] - ay;
        float l2 = bx * bx + by * by;
        if (l2 < 1e-6f) continue;
        float u = ((px - ax) * bx + (py - ay) * by) / l2;
        u = u < 0 ? 0 : (u > 1 ? 1 : u);
        float qx = ax + u * bx - px, qy = ay + u * by - py, d = sqrtf(qx * qx + qy * qy);
        if (d < dist) {
            dist = d;
            float l = sqrtf(l2);
            tx = bx / l; ty = by / l;
            if (d > 1e-6f) { nx = qx / d; ny = qy / d; } else { nx = -ty; ny = tx; }
        }
    }
}

// ---- allocation ---------------------------------------------------------------------------------
using WAlloc = void* (*)(size_t);
inline void* water_default_alloc(size_t n) { return malloc(n); }

inline void water_free_runtime(Water& w) {
    void* ps[] = {w.edge, w.dsafe, w.sp_off, w.sp, w.h0, w.h1, w.gx, w.gy, w.gxp, w.gyp, w.sgx, w.sgy, w.cin, w.cnx, w.cny,
                  w.fx, w.fy, w.od, w.onx, w.ony, w.pax, w.pay, w.bank, w.bank_k};
    w.pax = w.pay = nullptr; w.ghz = -1.0f; w.oworld = false;
    w.bank = nullptr; w.bank_k = nullptr; w.nbank = 0;
    for (void* p : ps) if (p) free(p);
    w.edge = nullptr; w.dsafe = nullptr; w.sp_off = nullptr; w.sp = nullptr;
    w.h0 = w.h1 = w.gx = w.gy = w.gxp = w.gyp = w.sgx = w.sgy = nullptr; w.cin = nullptr; w.cnx = w.cny = nullptr;
    w.fx = w.fy = w.od = w.onx = w.ony = nullptr;
    w.bw = w.bh = w.area = 0;
}
inline void water_free(Water& w) {
    water_free_runtime(w);
    if (w.mask) free(w.mask);
    w.mask = nullptr;
}
inline bool water_ensure_mask(Water& w, WAlloc al = water_default_alloc) {
    if (w.mask) return true;
    w.mask = (uint8_t*)al(MASK_BYTES);
    if (w.mask) memset(w.mask, 0, MASK_BYTES);
    return w.mask != nullptr;
}
inline void water_copy_def(Water& dst, const Water& src, WAlloc al = water_default_alloc) {
    water_free(dst);
    dst.mode = src.mode;
    memcpy(dst.name, src.name, sizeof(dst.name));
    memcpy(dst.prm, src.prm, sizeof(dst.prm));
    dst.nlines = src.nlines;
    for (int i = 0; i < WLINE_MAX; ++i) dst.line[i] = src.line[i];
    dst.grid = src.grid;
    memcpy(dst.gq, src.gq, sizeof(dst.gq));
    if (src.mask && water_ensure_mask(dst, al)) memcpy(dst.mask, src.mask, MASK_BYTES);
}

// ---- perspective grid -----------------------------------------------------------------------------
// The unit square (u, v) → the grid's quad on screen (Heckbert's square-to-quad homography).
struct WHomog { float a, b, c, d, e, f, g, h; };
inline WHomog water_homog(const float* q) {
    const float x0 = q[0], y0 = q[1], x1 = q[2], y1 = q[3], x2 = q[4], y2 = q[5], x3 = q[6], y3 = q[7];
    const float dx1 = x1 - x2, dx2 = x3 - x2, dx3 = x0 - x1 + x2 - x3;
    const float dy1 = y1 - y2, dy2 = y3 - y2, dy3 = y0 - y1 + y2 - y3;
    WHomog H{};
    const float den = dx1 * dy2 - dx2 * dy1;
    H.g = fabsf(den) > 1e-9f ? (dx3 * dy2 - dx2 * dy3) / den : 0.0f;
    H.h = fabsf(den) > 1e-9f ? (dx1 * dy3 - dx3 * dy1) / den : 0.0f;
    H.a = x1 - x0 + H.g * x1; H.b = x3 - x0 + H.h * x3; H.c = x0;
    H.d = y1 - y0 + H.g * y1; H.e = y3 - y0 + H.h * y3; H.f = y0;
    return H;
}
inline void water_hmap(const WHomog& H, float u, float v, float& x, float& y) {
    const float w = H.g * u + H.h * v + 1.0f;
    x = (H.a * u + H.b * v + H.c) / w; y = (H.d * u + H.e * v + H.f) / w;
}
// Per cell: how much a step on the water surface shrinks on screen across (ax) and in depth (ay),
// relative to the nearest water (1). Also the vanishing line, used as the horizon when none is set.
inline bool water_perspective(Water& w, WAlloc al) {
    if (!w.grid || !w.h1) return false;
    const WHomog H = water_homog(w.gq);
    // inverse (adjugate of [[a b c][d e f][g h 1]])
    const float A = H.e - H.f * H.h, B = -(H.b - H.c * H.h), C = H.b * H.f - H.c * H.e;
    const float D = -(H.d - H.f * H.g), E = H.a - H.c * H.g, F = -(H.a * H.f - H.c * H.d);
    const float G = H.d * H.h - H.e * H.g, Hh = -(H.a * H.h - H.b * H.g), I = H.a * H.e - H.b * H.d;
    const int gn = w.gw * w.gh;
    w.pax = (float*)al(gn * 4); w.pay = (float*)al(gn * 4);
    if (!w.pax || !w.pay) return false;
    float kref = 0;
    for (int j = 0; j < w.gh; ++j)
        for (int i = 0; i < w.gw; ++i) {
            const float sx = w.bx + (i - 1) * WCELL + WCELL * 0.5f, sy = w.by + (j - 1) * WCELL + WCELL * 0.5f;
            const float ww = G * sx + Hh * sy + I;
            float kx = 0, ky = 0;
            if (fabsf(ww) > 1e-9f) {
                const float u = (A * sx + B * sy + C) / ww, v = (D * sx + E * sy + F) / ww;
                if (H.g * u + H.h * v + 1.0f > 1e-4f) {          // on the near side of the horizon
                    const float e = 1e-3f;
                    float x0, y0, xu, yu, xv, yv;
                    water_hmap(H, u, v, x0, y0); water_hmap(H, u + e, v, xu, yu); water_hmap(H, u, v + e, xv, yv);
                    kx = sqrtf((xu - x0) * (xu - x0) + (yu - y0) * (yu - y0)) / e;
                    ky = sqrtf((xv - x0) * (xv - x0) + (yv - y0) * (yv - y0)) / e;
                }
            }
            w.pax[j * w.gw + i] = kx; w.pay[j * w.gw + i] = ky;
            if (w.cin[j * w.gw + i] && kx > kref) kref = kx;
        }
    if (kref <= 0) { free(w.pax); free(w.pay); w.pax = w.pay = nullptr; return false; }
    for (int k = 0; k < gn; ++k) {
        float ax = w.pax[k] / kref, ay = w.pay[k] / kref;
        ax = ax < 0.05f ? 0.05f : (ax > 1.0f ? 1.0f : ax);
        ay = ay < 0.05f ? 0.05f : (ay > 1.0f ? 1.0f : ay);
        w.pax[k] = 0.5f * ax * ax; w.pay[k] = 0.5f * ay * ay;   // (½: the stable step of the plain surface)
    }
    if (fabsf(Hh) > 1e-9f) w.ghz = -(G * (IMG_W * 0.5f) + I) / Hh;   // where the plane meets the sky
    if (w.mode == W_OCEAN && w.od) {
        // Swells on the water's own surface: distance across the plane (in px as seen up close), rolling
        // toward the near edge — so far swells are closer together, smaller and line up with the grid.
        float* vv = w.pay ? (float*)al(gn * 4) : nullptr;
        if (vv) {
            float vshore = -1e9f;
            for (int l = 0; l < w.nlines; ++l)               // a shore line, if drawn, is where they land
                for (int k = 0; k < w.line[l].n; ++k) {
                    const float sx = w.line[l].x[k], sy = w.line[l].y[k], ww = G * sx + Hh * sy + I;
                    if (fabsf(ww) > 1e-9f) { const float v = (D * sx + E * sy + F) / ww; if (v > vshore) vshore = v; }
                }
            float vmin = 1e9f;
            for (int j = 0; j < w.gh; ++j)
                for (int i = 0; i < w.gw; ++i) {
                    const int k = j * w.gw + i;
                    const float sx = w.bx + (i - 1) * WCELL + WCELL * 0.5f, sy = w.by + (j - 1) * WCELL + WCELL * 0.5f;
                    const float ww = G * sx + Hh * sy + I;
                    vv[k] = fabsf(ww) > 1e-9f ? (D * sx + E * sy + F) / ww : 1e9f;
                    if (w.cin[k] && vv[k] < vmin) vmin = vv[k];
                    // toward the near edge: −(screen direction of +v)
                    float u = fabsf(ww) > 1e-9f ? (A * sx + B * sy + C) / ww : 0.0f, x0, y0, x1, y1;
                    water_hmap(H, u, vv[k], x0, y0); water_hmap(H, u, vv[k] + 1e-3f, x1, y1);
                    float dx = x0 - x1, dy = y0 - y1, l = sqrtf(dx * dx + dy * dy);
                    if (l > 1e-9f) { w.onx[k] = dx / l; w.ony[k] = dy / l; }
                }
            const float v0 = vshore > -1e8f ? vshore : vmin;
            for (int k = 0; k < gn; ++k) {
                float d = (vv[k] - v0) * kref;                // px of surface, as they'd measure up close
                w.od[k] = d > 0 ? d : 0;
            }
            free(vv);
            w.oworld = true;
        }
    }
    return true;
}

// ---- prepare ------------------------------------------------------------------------------------
// Builds the edge-hold map, the simulation grid and the flow / shore fields. Resets the motion.
inline bool water_prepare(Water& w, WAlloc al = water_default_alloc) {
    water_free_runtime(w);
    w.t = w.sx = w.sy = w.svx = w.svy = w.rain_acc = w.set_ph = w.sim_acc = w.alpha = 0;
    w.rng = 12345;
    if (!w.mask) return false;
    int x0 = IMG_W, y0 = IMG_H, x1 = -1, y1 = -1, area = 0;
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x)
            if (mask_get(w.mask, x, y)) {
                ++area;
                if (x < x0) x0 = x;
                if (x > x1) x1 = x;
                if (y < y0) y0 = y;
                if (y > y1) y1 = y;
            }
    if (!area) return false;
    w.bx = x0; w.by = y0; w.bw = x1 - x0 + 1; w.bh = y1 - y0 + 1; w.area = area;
    const int n = w.bw * w.bh;
    // Edge hold: chamfer distance (in 1/3 px) to the nearest non-water pixel. The photo's own border
    // isn't a bank (the water carries on past it), so it doesn't count.
    uint16_t* d = (uint16_t*)malloc((size_t)n * 2);
    w.edge = (uint8_t*)al((size_t)n);
    w.dsafe = (uint8_t*)al((size_t)n);
    if (!d || !w.edge || !w.dsafe) { if (d) free(d); water_free_runtime(w); return false; }
    const uint16_t BIG = 60000;
    for (int y = 0; y < w.bh; ++y)
        for (int x = 0; x < w.bw; ++x) d[y * w.bw + x] = mask_get(w.mask, x0 + x, y0 + y) ? BIG : 0;
    auto inside_or_border = [&](int X, int Y) {      // outside the bbox: water only past the photo edge
        if (X < 0 || Y < 0 || X >= IMG_W || Y >= IMG_H) return true;
        return mask_get(w.mask, X, Y);
    };
    for (int y = 0; y < w.bh; ++y)
        for (int x = 0; x < w.bw; ++x) {
            uint16_t& v = d[y * w.bw + x];
            if (!v) continue;
            int best = v;
            const int ox[4] = {-1, 1, -1, 0}, oy[4] = {0, -1, -1, -1}, oc[4] = {3, 4, 4, 3};
            for (int k = 0; k < 4; ++k) {
                int X = x + ox[k], Y = y + oy[k];
                int dv = (X < 0 || Y < 0 || X >= w.bw || Y >= w.bh) ? (inside_or_border(x0 + X, y0 + Y) ? BIG : 0) : d[Y * w.bw + X];
                if (dv + oc[k] < best) best = dv + oc[k];
            }
            v = (uint16_t)(best > BIG ? BIG : best);
        }
    for (int y = w.bh - 1; y >= 0; --y)
        for (int x = w.bw - 1; x >= 0; --x) {
            uint16_t& v = d[y * w.bw + x];
            if (!v) continue;
            int best = v;
            const int ox[4] = {1, -1, 1, 0}, oy[4] = {0, 1, 1, 1}, oc[4] = {3, 4, 4, 3};
            for (int k = 0; k < 4; ++k) {
                int X = x + ox[k], Y = y + oy[k];
                int dv = (X < 0 || Y < 0 || X >= w.bw || Y >= w.bh) ? (inside_or_border(x0 + X, y0 + Y) ? BIG : 0) : d[Y * w.bw + X];
                if (dv + oc[k] < best) best = dv + oc[k];
            }
            v = (uint16_t)(best > BIG ? BIG : best);
        }
    const float edge_px = wprm(w, WP_EDGE);
    for (int i = 0; i < n; ++i) {
        if (!d[i]) { w.edge[i] = 0; w.dsafe[i] = 0; continue; }
        float px = d[i] / 3.0f - 0.5f;                       // distance from the bank, px
        w.dsafe[i] = (uint8_t)(px < 0 ? 0 : (px > 255 ? 255 : (int)px));
        float e = edge_px > 0.01f ? smooth01(px / edge_px) : 1.0f;
        int v = (int)(e * 255.0f + 0.5f);
        w.edge[i] = (uint8_t)(v < 1 ? 1 : v);                 // ≥1 = water (even right at the bank)
    }
    free(d);
    // Each row's water spans: the renderer jumps straight across the land.
    {
        int ns = 0;
        for (int y = 0; y < w.bh; ++y)
            for (int x = 0; x < w.bw; ++x)
                if (w.edge[y * w.bw + x] && (x == 0 || !w.edge[y * w.bw + x - 1])) ++ns;
        w.sp_off = (uint16_t*)al((size_t)(w.bh + 1) * 2);
        w.sp = (int16_t*)al((size_t)(ns ? ns : 1) * 4);
        if (!w.sp_off || !w.sp) { water_free_runtime(w); return false; }
        int k = 0;
        for (int y = 0; y < w.bh; ++y) {
            w.sp_off[y] = (uint16_t)k;
            const uint8_t* er = w.edge + y * w.bw;
            for (int x = 0; x < w.bw;) {
                if (!er[x]) { ++x; continue; }
                int a = x;
                while (x < w.bw && er[x]) ++x;
                w.sp[2 * k] = (int16_t)(w.bx + a); w.sp[2 * k + 1] = (int16_t)(w.bx + x); ++k;
            }
        }
        w.sp_off[w.bh] = (uint16_t)k;
    }

    // Simulation grid: cell (i, j) covers pixels from bx + (i-1)·WCELL, one spare cell all round.
    w.ghz = -1.0f;
    w.gw = (w.bw + WCELL - 1) / WCELL + 2;
    w.gh = (w.bh + WCELL - 1) / WCELL + 2;
    const int gn = w.gw * w.gh;
    w.h0 = (float*)al(gn * 4); w.h1 = (float*)al(gn * 4); w.gx = (float*)al(gn * 4); w.gy = (float*)al(gn * 4);
    w.gxp = (float*)al(gn * 4); w.gyp = (float*)al(gn * 4); w.sgx = (float*)al(gn * 4); w.sgy = (float*)al(gn * 4);
    w.cin = (uint8_t*)al(gn); w.cnx = (int8_t*)al(gn); w.cny = (int8_t*)al(gn);
    if (!w.h0 || !w.h1 || !w.gx || !w.gy || !w.gxp || !w.gyp || !w.sgx || !w.sgy || !w.cin || !w.cnx || !w.cny) { water_free_runtime(w); return false; }
    memset(w.h0, 0, gn * 4); memset(w.h1, 0, gn * 4); memset(w.gx, 0, gn * 4); memset(w.gy, 0, gn * 4);
    memset(w.gxp, 0, gn * 4); memset(w.gyp, 0, gn * 4); memset(w.sgx, 0, gn * 4); memset(w.sgy, 0, gn * 4);
    for (int j = 0; j < w.gh; ++j)
        for (int i = 0; i < w.gw; ++i) {
            int cx = x0 + (i - 1) * WCELL + WCELL / 2, cy = y0 + (j - 1) * WCELL + WCELL / 2;
            w.cin[j * w.gw + i] = (i > 0 && j > 0 && i < w.gw - 1 && j < w.gh - 1 && mask_get(w.mask, cx, cy)) ? 1 : 0;
        }
    for (int j = 0; j < w.gh; ++j)
        for (int i = 0; i < w.gw; ++i) {
            int k = j * w.gw + i;
            w.cnx[k] = w.cny[k] = 0;
            if (!w.cin[k]) continue;
            int l = w.cin[k - 1], r = w.cin[k + 1], u = w.cin[k - w.gw], dn = w.cin[k + w.gw];
            w.cnx[k] = (int8_t)(l - r);          // toward the missing side
            w.cny[k] = (int8_t)(u - dn);
        }

    // Curves → fields.
    static float cxs[WLINE_MAX][WPTS_MAX * 8 + 1], cys[WLINE_MAX][WPTS_MAX * 8 + 1];   // (static: small stack)
    int cn[WLINE_MAX] = {0, 0, 0};
    for (int l = 0; l < w.nlines && l < WLINE_MAX; ++l) cn[l] = water_curve(w.line[l], cxs[l], cys[l], WPTS_MAX * 8 + 1);
    if (w.mode == W_RIVER) {
        w.fx = (float*)al(gn * 4); w.fy = (float*)al(gn * 4);
        if (!w.fx || !w.fy) { water_free_runtime(w); return false; }
        for (int j = 0; j < w.gh; ++j)
            for (int i = 0; i < w.gw; ++i) {
                float px = x0 + (i - 1) * WCELL + WCELL * 0.5f, py = y0 + (j - 1) * WCELL + WCELL * 0.5f;
                float sx = 0, sy = 0, sw = 0;
                for (int l = 0; l < w.nlines; ++l) {
                    if (cn[l] < 2) continue;
                    float dist, tx, ty, nx, ny;
                    curve_nearest(cxs[l], cys[l], cn[l], px, py, dist, tx, ty, nx, ny);
                    float wt = 1.0f / (dist * dist + 144.0f);
                    sx += wt * tx; sy += wt * ty; sw += wt;
                }
                float l2 = sqrtf(sx * sx + sy * sy);
                const int k = j * w.gw + i;
                if (sw > 0 && l2 > 1e-6f) { w.fx[k] = sx / l2; w.fy[k] = sy / l2; } else { w.fx[k] = 0; w.fy[k] = 0; }
            }
    }
    if (w.mode == W_OCEAN) {
        w.od = (float*)al(gn * 4); w.onx = (float*)al(gn * 4); w.ony = (float*)al(gn * 4);
        if (!w.od || !w.onx || !w.ony) { water_free_runtime(w); return false; }
        for (int j = 0; j < w.gh; ++j)
            for (int i = 0; i < w.gw; ++i) {
                float px = x0 + (i - 1) * WCELL + WCELL * 0.5f, py = y0 + (j - 1) * WCELL + WCELL * 0.5f;
                float best = 1e9f, bnx = 0, bny = 1;
                for (int l = 0; l < w.nlines; ++l) {
                    if (cn[l] < 1) continue;
                    float dist, tx, ty, nx, ny;
                    curve_nearest(cxs[l], cys[l], cn[l], px, py, dist, tx, ty, nx, ny);
                    if (dist < best) { best = dist; bnx = nx; bny = ny; }
                }
                const int k = j * w.gw + i;
                if (best > 1e8f) { best = (float)(y0 + w.bh) - py; bnx = 0; bny = 1; }   // no line: toward the bottom
                w.od[k] = best; w.onx[k] = bnx; w.ony[k] = bny;
            }
    }
    water_perspective(w, al);
    // The edge cells, listed once with their push (water_step leans the water every tick it tilts: it would
    // otherwise look through every cell for them, and take a square root at each). Without the memory: it does.
    int nb = 0;
    for (int k = 0; k < gn; ++k) if (w.cnx[k] | w.cny[k]) ++nb;
    w.bank = (int32_t*)al((size_t)(nb > 0 ? nb : 1) * 4); w.bank_k = (float*)al((size_t)(nb > 0 ? nb : 1) * 4);
    if (!w.bank || !w.bank_k) { if (w.bank) free(w.bank); if (w.bank_k) free(w.bank_k); w.bank = nullptr; w.bank_k = nullptr; }
    else
        for (int k = 0; k < gn; ++k)
            if (w.cnx[k] | w.cny[k]) { w.bank[w.nbank] = k; w.bank_k[w.nbank] = w.pax ? sqrtf(2.0f * w.pax[k]) : 1.0f; ++w.nbank; }
    return true;
}

// ---- motion -------------------------------------------------------------------------------------
inline uint32_t wrand(Water& w) { w.rng = w.rng * 1664525u + 1013904223u; return w.rng >> 8; }
inline float wrandf(Water& w) { return (wrand(w) & 0xFFFF) / 65536.0f; }

// A drop / tap / splash at screen (x, y): amt ≈ how hard, r = radius in cells.
inline void water_poke(Water& w, float x, float y, float amt, int r) {
    if (!w.h1) return;
    int ci = (int)floorf((x - w.bx) / WCELL) + 1, cj = (int)floorf((y - w.by) / WCELL) + 1;
    if (w.pax && ci >= 0 && cj >= 0 && ci < w.gw && cj < w.gh) {   // far away: a smaller splash
        const float s = sqrtf(2.0f * w.pax[cj * w.gw + ci]);
        amt *= s;
        r = (int)lroundf(r * s); if (r < 1) r = 1;
    }
    for (int dj = -r; dj <= r; ++dj)
        for (int di = -r; di <= r; ++di) {
            int i = ci + di, j = cj + dj;
            if (i < 1 || j < 1 || i >= w.gw - 1 || j >= w.gh - 1) continue;
            float d2 = (float)(di * di + dj * dj), rr = (float)(r * r) + 0.5f;
            if (d2 > rr) continue;
            int k = j * w.gw + i;
            if (w.cin[k]) w.h1[k] -= amt * (1.0f - d2 / (rr + 0.5f));
        }
}
inline bool water_contains(const Water& w, int x, int y) { return w.edge && mask_get(w.mask, x, y); }

static constexpr float W_SLOSH_PX = 7.0f;       // how far the body of water leans at a full tilt
static constexpr float W_OCEAN_FEEL = 0.15f;    // an ocean barely feels the Core2's tilts and shakes
static constexpr float W_RIP_K = 0.035f;        // surface slope → pixel shift
static constexpr float W_AMB_PX = 0.9f;         // ambient wave height (px, close up)
static constexpr float W_RIVER_PX = 14.0f;      // river flow (px/s) at speed 1
static constexpr float W_RIVER_T = 1.6f;        // s per flow cycle (two phases cross-fade)
static constexpr float W_OCEAN_A = 4.2f;        // swell size (px, close up) at size 1
static constexpr float W_OCEAN_SHADE = 0.16f;   // swells catch the light: back of a swell lighter, face darker
static constexpr float W_RIP_SHADE = 0.0035f;   // ripples catch the light too (per unit of surface slope)
static constexpr float W_OCEAN_T = 2.4f;        // s per swell at speed 1
static constexpr float W_SIM_DT = 0.04f;        // surface simulation step (25 Hz)

// dn: unit "down" in screen axes (from the IMU; upright = (0,1,0)). jolt: g, gravity removed.
inline void water_step(Water& w, float dt, const float dn[3], const float jolt[3]) {
    if (!w.h1) return;
    if (dt > 0.1f) dt = 0.1f;
    w.t += dt;
    // The body of water leans toward the real down and overshoots a little; its swings throw waves
    // off the banks.
    // (An ocean is too big to slosh in a frame: it only just feels them.)
    const float sl = wprm(w, WP_SLOSH) * (w.mode == W_OCEAN ? W_OCEAN_FEEL : 1.0f);
    // (toward the side that went down: tipped right, right; top edge tipped away (dn[2] < 0), toward the top -
    // the far side of the water - and toward you, the bottom)
    const float tx = sl * W_SLOSH_PX * dn[0], ty = sl * W_SLOSH_PX * dn[2] * 0.6f;
    const float ax = (tx - w.sx) * 10.0f - w.svx * 1.6f, ay = (ty - w.sy) * 10.0f - w.svy * 1.6f;
    const float kx = ax * dt, ky = ay * dt;
    w.svx += kx; w.svy += ky; w.sx += w.svx * dt; w.sy += w.svy * dt;
    const int gn = w.gw * w.gh;
    const float kick = 9.0f * sl;
    if (fabsf(kx) + fabsf(ky) > 1e-4f) {
        if (w.bank)                                   // (far banks throw smaller waves)
            for (int b = 0; b < w.nbank; ++b) {
                const int k = w.bank[b];
                w.h1[k] += kick * w.bank_k[b] * (kx * w.cnx[k] + ky * w.cny[k]);
            }
        else
            for (int k = 0; k < gn; ++k)
                if (w.cnx[k] | w.cny[k])
                    w.h1[k] += kick * (w.pax ? sqrtf(2.0f * w.pax[k]) : 1.0f) * (kx * w.cnx[k] + ky * w.cny[k]);
    }
    // Shakes: chop where you'd least expect it.
    const float rip = wprm(w, WP_RIPPLE);
    const float jm = sqrtf(jolt[0] * jolt[0] + jolt[1] * jolt[1] + jolt[2] * jolt[2]);
    if (jm > 0.06f && rip > 0) {
        int np = (int)(jm * 6.0f * sl) + 1;
        for (int i = 0; i < np; ++i) {
            float px = w.bx + wrandf(w) * w.bw, py = w.by + wrandf(w) * w.bh;
            water_poke(w, px, py, 45.0f * jm * rip * (w.mode == W_OCEAN ? 0.4f : 1.0f), 1);
        }
    }
    // Rain.
    if (wprm(w, WP_RAIN) >= 0.5f && rip > 0) {
        w.rain_acc += dt * wprm(w, WP_RATE) * w.area / 1500.0f;
        while (w.rain_acc >= 1.0f) {
            w.rain_acc -= 1.0f;
            float px = w.bx + wrandf(w) * w.bw, py = w.by + wrandf(w) * w.bh;
            if (mask_get(w.mask, (int)px, (int)py)) water_poke(w, px, py, (45.0f + 35.0f * wrandf(w)) * rip, 1);
        }
    }
    // Surface: the wave equation at a fixed 25 steps a second (same speed at any frame rate); the
    // slopes are drawn blended between the last two steps, so it moves smoothly at any frame rate.
    w.sim_acc += dt;
    int guard = 0;
    while (w.sim_acc >= W_SIM_DT && guard++ < 4) {
        w.sim_acc -= W_SIM_DT;
        for (int j = 1; j < w.gh - 1; ++j)
            for (int i = 1; i < w.gw - 1; ++i) {
                int k = j * w.gw + i;
                if (!w.cin[k]) { w.h0[k] = 0; continue; }
                float nb;
                if (w.pax) {                                  // perspective: slower, foreshortened far away
                    const float hc = w.h1[k];
                    nb = 2.0f * hc + w.pax[k] * (w.h1[k - 1] + w.h1[k + 1] - 2.0f * hc)
                                   + w.pay[k] * (w.h1[k - w.gw] + w.h1[k + w.gw] - 2.0f * hc) - w.h0[k];
                } else {
                    nb = (w.h1[k - 1] + w.h1[k + 1] + w.h1[k - w.gw] + w.h1[k + w.gw]) * 0.5f - w.h0[k];
                }
                w.h0[k] = nb * 0.985f;
            }
        float* tmp = w.h0; w.h0 = w.h1; w.h1 = tmp;
        tmp = w.gxp; w.gxp = w.gx; w.gx = tmp;
        tmp = w.gyp; w.gyp = w.gy; w.gy = tmp;
        for (int j = 1; j < w.gh - 1; ++j)
            for (int i = 1; i < w.gw - 1; ++i) {
                int k = j * w.gw + i;
                w.gx[k] = w.h1[k + 1] - w.h1[k - 1];
                w.gy[k] = w.h1[k + w.gw] - w.h1[k - w.gw];
            }
    }
    if (w.sim_acc > W_SIM_DT) w.sim_acc = W_SIM_DT;
    w.alpha = w.sim_acc / W_SIM_DT;
    w.set_ph += dt / (W_OCEAN_T / wprm(w, WP_SPEED) * 7.0f);
}

// ---- render -------------------------------------------------------------------------------------
// The waves, ripples and swell are smooth: worked out at half the picture's resolution (a node every
// second pixel each way), with everything smooth at cell scale interpolated once per row and integer-
// phase table sines. Per pixel: a blend of its nodes, its own fade toward the bank, and the bilinear
// sample of the photo - at full resolution (the photo itself is never scaled).
static int16_t s_wsin[1025];                        // sin, Q14, 1024 steps (+1 for the lerp)
static bool s_wsin_ok = false;
inline __attribute__((noinline)) const int16_t* wsin_tab() {
    if (!s_wsin_ok) { for (int i = 0; i <= 1024; ++i) s_wsin[i] = (int16_t)lrintf(16384.0f * sinf(6.2831853f * i / 1024.0f)); s_wsin_ok = true; }
    return s_wsin;
}
JIG_INLINE float wsin_u(uint32_t ph) {                  // ph: a full turn = 2^32
    if (__builtin_expect(!s_wsin_ok, 0)) wsin_tab();   // (built once, at start: a read, not a call, per sine)
    const int16_t* T = s_wsin;
    const uint32_t i = ph >> 22;
    const int f = (int)((ph >> 12) & 1023);
    const int a = T[i], b = T[i + 1];
    return (float)(a + (((b - a) * f) >> 10)) * (1.0f / 16384.0f);
}
inline uint32_t wturns(float turns) {               // turns (any sign) → phase
    turns -= floorf(turns);
    return (uint32_t)(turns * 4294967296.0f);
}
inline float wsin(float a) { return wsin_u(wturns(a * 0.15915494f)); }

JIG_INLINE uint16_t rgb565_mix(uint16_t a, uint16_t b, int t8) {   // t8: 0 → a, 256 → b
    int ar = a >> 11, ag = (a >> 5) & 63, ab = a & 31, br = b >> 11, bg = (b >> 5) & 63, bb = b & 31;
    int r = ar + (((br - ar) * t8) >> 8), g = ag + (((bg - ag) * t8) >> 8), bl = ab + (((bb - ab) * t8) >> 8);
    return (uint16_t)((r << 11) | (g << 5) | bl);
}
JIG_INLINE uint16_t rgb565_brighten(uint16_t c, float k) {          // toward white by k (0..1)
    int r = c >> 11, g = (c >> 5) & 63, b = c & 31;
    r += (int)((31 - r) * k); g += (int)((63 - g) * k); b += (int)((31 - b) * k);
    return (uint16_t)((r << 11) | (g << 5) | b);
}


JIG_INLINE uint16_t rgb565_scale(uint16_t c, float f) {             // brightness × f (clamped)
    int r = (int)((c >> 11) * f), g = (int)(((c >> 5) & 63) * f), b = (int)((c & 31) * f);
    r = r > 31 ? 31 : r; g = g > 63 ? 63 : g; b = b > 31 ? 31 : b;
    return (uint16_t)((r << 11) | (g << 5) | b);
}
// Water is drawn per frame (WFrame: what holds for all of it), per node row (WRow) and per pixel.
constexpr int W_NC = IMG_W / WCELL + 4;
struct WFrame {
    float t, al, rk, slx, sly, hz, span_h, amb0, glint, light, foam_k;
    bool ocean, river;
    float tT, ospc, oA0;
    float k1, k2; int mix8;
    const float* sgx = nullptr;                      // (optional) the ripple field as it stood when the frame
    const float* sgy = nullptr;                      //   began (water_snapshot) - every row from the same moment
    uint32_t id = 0;                                 // (which frame: the node rows cached for it)
};
// That snapshot: gw·gh blended slopes each, exactly as water_row_setup would blend them live. A frame drawn
// over several ticks then shows one moment, however far the surface has stepped since.
inline void water_snapshot(const Water& w, float* sgx, float* sgy) {
    const float al = w.alpha;
    for (int k = 0; k < w.gw * w.gh; ++k) {
        sgx[k] = w.gxp[k] + (w.gx[k] - w.gxp[k]) * al;
        sgy[k] = w.gyp[k] + (w.gy[k] - w.gyp[k]) * al;
    }
}
struct WRow {
    int y = -1, ry;                                  // the picture row it's for (-1: none yet)
    const Water* w = nullptr;
    float Aamb, inv_lam, half_lam, lam, oA, oA8, oAr, oAr1;
    int32_t d1, d2, d3;
    uint32_t b1, b2, b3;
    float rRX[W_NC], rRY[W_NC], rGX[W_NC], rGY[W_NC], rA[W_NC], rB[W_NC], rC[W_NC];
};
inline void water_frame_setup(const Water& w, WFrame& F) {
    static uint32_t s_id = 0;
    F.id = ++s_id;
    F.t = w.t; F.al = w.alpha;
    F.rk = W_RIP_K * wprm(w, WP_RIPPLE);
    F.slx = -w.sx * 0.5f; F.sly = -w.sy * 0.5f;     // (a pixel shows the photo from x + d: the lean moves it by −d)
    float hz = wprm(w, WP_HORIZON);
    if (hz < 0) hz = (w.ghz > -200.0f && w.ghz < (float)(w.by + w.bh)) ? w.ghz : (float)w.by;   // the grid's, if any
    F.hz = hz;
    const float bottom = (float)(w.by + w.bh);
    F.span_h = bottom - hz > 1 ? bottom - hz : 1.0f;
    F.amb0 = wprm(w, WP_AMB) * W_AMB_PX;
    F.glint = wprm(w, WP_GLINT);
    F.light = wprm(w, WP_LIGHT); F.foam_k = wprm(w, WP_FOAM);
    F.ocean = w.mode == W_OCEAN && w.od; F.river = w.mode == W_RIVER && w.fx;
    // ocean (per frame)
    const float oT = W_OCEAN_T / wprm(w, WP_SPEED);
    const float oenv = 1.0f + wprm(w, WP_SETS) * 0.6f * wsin(6.2831853f * w.set_ph);
    F.oA0 = W_OCEAN_A * wprm(w, WP_SIZE) * oenv; F.ospc = wprm(w, WP_SPACING);
    float tT = F.t / oT; tT -= floorf(tT);
    F.tT = tT;
    // river (per frame)
    const float rsp = W_RIVER_PX * wprm(w, WP_SPEED) * W_RIVER_T;
    float f1 = F.t / W_RIVER_T; f1 -= floorf(f1);
    float f2 = f1 + 0.5f; f2 -= floorf(f2);
    F.k1 = -(f1 - 0.5f) * rsp; F.k2 = -(f2 - 0.5f) * rsp;
    F.mix8 = (int)(fabsf(2.0f * f1 - 1.0f) * 256.0f);
}
// per row: perspective, wave phases, and the cell fields blended between cell rows (y: a row of the water)
inline void water_row_setup(const Water& w, const WFrame& F, int y, WRow& R) {
    const float TURN = 4294967296.0f / 6.2831853f;
    R.y = y; R.ry = y - w.by; R.w = &w;
    const float t = F.t, al = F.al;
    float p = (y - F.hz) / F.span_h;
    p = p < 0.15f ? 0.15f : (p > 1 ? 1 : p);
    const float ip = 1.0f / p;
    R.Aamb = F.amb0 * p;
    R.d1 = (int32_t)(0.05f * ip * TURN); R.d2 = (int32_t)(-0.08f * ip * TURN); R.d3 = (int32_t)(0.07f * ip * TURN);
    R.b1 = wturns((y * 0.13f * ip - 1.4f * t) * 0.15915494f);
    R.b2 = wturns((y * 0.19f * ip - 2.1f * t + 1.7f) * 0.15915494f);
    R.b3 = wturns((-y * 0.06f * ip + 1.2f * t) * 0.15915494f);
    const float v = R.ry / (float)WCELL + 0.5f;
    int j = (int)v; if (j > w.gh - 2) j = w.gh - 2;
    float fy = v - j; fy = fy < 0 ? 0 : (fy > 1 ? 1 : fy);
    const float gy0 = 1.0f - fy;
    for (int i = 0; i < w.gw; ++i) {
        const int ka = j * w.gw + i, kb = ka + w.gw;
        float gxa, gxb, gya, gyb;
        if (F.sgx) { gxa = F.sgx[ka]; gxb = F.sgx[kb]; gya = F.sgy[ka]; gyb = F.sgy[kb]; }
        else {
            gxa = w.gxp[ka] + (w.gx[ka] - w.gxp[ka]) * al; gxb = w.gxp[kb] + (w.gx[kb] - w.gxp[kb]) * al;
            gya = w.gyp[ka] + (w.gy[ka] - w.gyp[ka]) * al; gyb = w.gyp[kb] + (w.gy[kb] - w.gyp[kb]) * al;
        }
        R.rGX[i] = gxa * gy0 + gxb * fy; R.rGY[i] = gya * gy0 + gyb * fy;
        R.rRX[i] = R.rGX[i] * F.rk; R.rRY[i] = R.rGY[i] * F.rk;
        if (F.ocean) { R.rA[i] = w.od[ka] * gy0 + w.od[kb] * fy; R.rB[i] = w.onx[ka] * gy0 + w.onx[kb] * fy; R.rC[i] = w.ony[ka] * gy0 + w.ony[kb] * fy; }
        else if (F.river) { R.rA[i] = w.fx[ka] * gy0 + w.fx[kb] * fy; R.rB[i] = w.fy[ka] * gy0 + w.fy[kb] * fy; }
    }
    R.lam = w.oworld ? F.ospc : F.ospc * (p > 0.3f ? p : 0.3f);   // (on the surface: one spacing)
    R.inv_lam = 1.0f / R.lam; R.half_lam = R.lam * 0.5f; R.oA = F.oA0 * p;
    R.oA8 = R.oA * 0.8f; R.oAr = R.oA / W_OCEAN_A; R.oAr1 = R.oAr > 1 ? 1 : R.oAr;   // (per row, not per pixel)
}
// The ripples, waves and swell are smooth: they're worked out on a lattice of every second pixel (x and y
// even, "nodes": half the picture's resolution), and each pixel blends its nodes (½ weights). Only the
// photo is sampled at every pixel - at full resolution, at the spot its blend displaces it to - and the
// fade toward the bank is the pixel's own. (Everything here is before that fade.)
struct alignas(4) WNode {                            // 12 bytes
    int16_t ax, ay, bx, by;                          // displacement, 1/64 px (river: the two flow phases; else b = a)
    int16_t sh;                                      // painted light: shade − 1, ×8192 …
    int8_t  fo, gl;                                  // … foam, glint strength: ×127 (these three: one aligned word)
};
JIG_INLINE uint32_t wnode_extras(const WNode& n) { uint32_t v; memcpy(&v, &n.sh, 4); return v; }   // (0: none)
JIG_INLINE int wq(float v, float k, int lim) {       // (to the node's fixed point, rounded, clamped)
    const float q = v * k;
    return q > (float)lim ? lim : (q < -(float)lim ? -lim : (int)(q + (q >= 0 ? 0.5f : -0.5f)));
}
// One node (x even, R a row set up for y = R.y even). P1..P3: its wave phases (b + d·x).
JIG_INLINE WNode water_node(const Water& w, const WFrame& F, const WRow& R, int x, uint32_t P1, uint32_t P2, uint32_t P3) {
    const float u = (x - w.bx) * (1.0f / WCELL) + 0.5f;
    int i = (int)u; if (i < 0) i = 0; if (i > w.gw - 2) i = w.gw - 2;
    float fx = u - i; fx = fx < 0 ? 0 : (fx > 1 ? 1 : fx);
    const float fx0 = 1.0f - fx;
    const float gxr = R.rGX[i] * fx0 + R.rGX[i + 1] * fx, gyr = R.rGY[i] * fx0 + R.rGY[i + 1] * fx;
    float shade = 1.0f - F.light * W_RIP_SHADE * gyr * (F.rk > 0 ? 1.0f : 0.0f);   // painted light (optional)
    float foam = 0.0f;
    float dx = R.Aamb * 0.35f * wsin_u(P3) + R.rRX[i] * fx0 + R.rRX[i + 1] * fx + F.slx;
    float dy = R.Aamb * (0.6f * wsin_u(P1) + 0.4f * wsin_u(P2)) + R.rRY[i] * fx0 + R.rRY[i + 1] * fx + F.sly;
    if (F.ocean) {
        const float c06 = 0.82533561f, s06 = 0.56464247f;            // cos 0.6, sin 0.6
        const float d = R.rA[i] * fx0 + R.rA[i + 1] * fx, nx = R.rB[i] * fx0 + R.rB[i + 1] * fx, ny = R.rC[i] * fx0 + R.rC[i + 1] * fx;
        float q = d * R.inv_lam + F.tT;
        q -= floorf(q);
        const uint32_t U = (uint32_t)(q * 4294967296.0f);
        const float sp_ = wsin_u(U), cp = wsin_u(U + 0x40000000u);
        const float s2 = 2.0f * sp_ * cp * c06 - (cp * cp - sp_ * sp_) * s06;   // sin(2φ − 0.6)
        dy -= R.oA8 * (sp_ + 0.35f * s2);
        float push = R.oA * 0.6f * cp;
        if (d < R.half_lam && sp_ > 0) push += R.oA8 * sp_ * (1.0f - d / R.half_lam);
        dx += nx * push; dy += ny * push;
        // optional painted light: the slope of the swell (d/dφ of its height) — backs lighter, faces darker
        const float c2 = (cp * cp - sp_ * sp_) * c06 + 2.0f * sp_ * cp * s06;   // cos(2φ − 0.6)
        shade += F.light * W_OCEAN_SHADE * R.oAr * (cp + 0.7f * c2);
        // foam where the swells break near the shore line
        const float zone = R.lam * 0.6f;
        if (F.foam_k > 0 && d < zone && sp_ > 0.7f) foam = F.foam_k * (sp_ - 0.7f) * 3.3f * (1.0f - d / zone) * R.oAr1;
    }
    WNode n;
    if (F.river) {                                   // two phases of the flow, cross-faded per pixel
        const float fxv = R.rA[i] * fx0 + R.rA[i + 1] * fx, fyv = R.rB[i] * fx0 + R.rB[i + 1] * fx;
        n.ax = (int16_t)wq(dx + fxv * F.k1, 64.0f, 32767); n.ay = (int16_t)wq(dy + fyv * F.k1, 64.0f, 32767);
        n.bx = (int16_t)wq(dx + fxv * F.k2, 64.0f, 32767); n.by = (int16_t)wq(dy + fyv * F.k2, 64.0f, 32767);
    } else { n.ax = n.bx = (int16_t)wq(dx, 64.0f, 32767); n.ay = n.by = (int16_t)wq(dy, 64.0f, 32767); }
    n.sh = (int16_t)wq(shade - 1.0f, 8192.0f, 32767);
    n.fo = (int8_t)wq(foam, 127.0f, 127);
    float g = 0.0f;                                  // light caught on a steep ripple (scaled by the pixel's light)
    if (F.glint > 0) {
        const float sl2 = gxr * gxr + gyr * gyr;
        if (sl2 > 324.0f) { g = (sqrtf(sl2) - 18.0f) * (1.0f / 40.0f); g = (g > 1 ? 1 : g) * F.glint; }
    }
    n.gl = (int8_t)wq(g, 127.0f, 127);
    return n;
}
// The node rows of the frame being drawn, shared by both cores: a slice of pixel rows first has its new node
// rows worked out (water_nodes - the two cores take half each), then its pixels blend them (water_render).
// Slices go down the water, so the row a slice shares with the last is kept, not worked out again.
constexpr int W_NN = IMG_W / 2 + 3;                  // nodes per row (x up to IMG_W + 2: past the picture is fine)
constexpr int W_NR = 8;                              // node rows kept (a slice of STRIP_ROWS ≤ 12 rows needs ≤ 7)
struct WNodeCache {
    const Water* w = nullptr; uint32_t id = 0;
    int tag[W_NR];                                   // the node row in each slot (INT32_MIN: none)
    int dmax[W_NR];                                  // its largest displacement (1/64 px, either axis, either phase)
    WNode n[W_NR][W_NN];
    WNode mid[2][W_NN];                              // per core: an odd pixel row, halfway between its two node rows
};
// (~20 KB, read at every water pixel: internal RAM when there's comfortably room, else PSRAM - on the heap,
// so it can never crowd the firmware's fixed data out of internal RAM. Made by the first water_nodes_due, on
// one core.)
inline WNodeCache*& water_nodes_ptr() { static WNodeCache* c = nullptr; return c; }
inline WNodeCache* water_nodes_cache() {
    WNodeCache*& c = water_nodes_ptr();
    if (!c) {
        void* p = nullptr;
#if defined(ESP_PLATFORM)                             // (the same room kept free as the other fast scratch)
        if (heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT) > sizeof(WNodeCache) + 48 * 1024)
            p = heap_caps_malloc(sizeof(WNodeCache), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
#endif
        if (!p) p = malloc(sizeof(WNodeCache));
        if (p) c = new (p) WNodeCache();
    }
    return c;
}
inline WRow& water_row_scratch(int lane) { static WRow r[2]; return r[lane & 1]; }   // (a core's)
// Pixel row y's water reaches nodes x0/2 … (x1−1)/2 + 1 of each span [x0, x1).
JIG_INLINE void water_need(const Water& w, int y, uint8_t* need) {
    if (y < w.by || y >= w.by + w.bh) return;
    for (int k = w.sp_off[y - w.by]; k < w.sp_off[y - w.by + 1]; ++k) {
        const int a = w.sp[2 * k] >> 1, b = ((w.sp[2 * k + 1] - 1) >> 1) + 2;
        if (a < b) memset(need + a, 1, (size_t)(b - a));
    }
}
// Node rows [j0, j1) into their slots (R: this core's row scratch). Only the nodes pixels in rows 2j−1 … 2j+1
// blend - not across dry gaps.
inline void water_node_rows(const Water& w, const WFrame& F, WRow& R, int j0, int j1) {
    WNodeCache* Cp = water_nodes_ptr();
    if (!Cp) return;
    WNodeCache& C = *Cp;
    uint8_t need[W_NN];
    for (int j = j0; j < j1; ++j) {
        memset(need, 0, sizeof need);
        for (int y = 2 * j - 1; y <= 2 * j + 1; ++y) water_need(w, y, need);
        const int slot = ((j % W_NR) + W_NR) % W_NR;
        WNode* n = C.n[slot];
        bool set = false;
        int dm = 0;
        for (int i = 0; i < W_NN; ++i) {
            if (!need[i]) continue;
            if (!set) { water_row_setup(w, F, 2 * j, R); set = true; }
            const uint32_t x = 2u * (uint32_t)i;
            const WNode& m = n[i] = water_node(w, F, R, (int)x, R.b1 + (uint32_t)R.d1 * x, R.b2 + (uint32_t)R.d2 * x, R.b3 + (uint32_t)R.d3 * x);
            const int d = abs(m.ax) | abs(m.ay) | abs(m.bx) | abs(m.by);   // (≥ each: a bound, cheaply)
            if (d > dm) dm = d;
        }
        C.dmax[slot] = dm;
    }
}
// The node rows pixel rows [y0, y1) need that aren't kept yet: claims their slots, returns them as [j0, j1)
// (call water_node_rows for those - on both cores - before water_render). One core at a time.
inline void water_nodes_due(const Water& w, const WFrame& F, int y0, int y1, int& j0, int& j1) {
    j0 = j1 = 0;
    WNodeCache* Cp = water_nodes_cache();
    if (!Cp) return;
    WNodeCache& C = *Cp;
    if (C.w != &w || C.id != F.id) { C.w = &w; C.id = F.id; for (int k = 0; k < W_NR; ++k) C.tag[k] = INT32_MIN; }
    const int ja = y0 >> 1, jb = ((y1 - 1) >> 1) + ((y1 - 1) & 1) + 1;   // (the last row, if odd, needs the next)
    j0 = jb; j1 = jb;
    for (int j = ja; j < jb; ++j) {
        int& t = C.tag[((j % W_NR) + W_NR) % W_NR];
        if (t == j) continue;
        t = j;
        if (j0 == jb) j0 = j;
        j1 = j + 1;
    }
}
// Keep a sample inside the water (displacement in 1/64 px): back off toward the pixel itself if it would
// land on the bank.
JIG_INLINE void water_clamp_q(const Water& w, int x, int y, int& ax, int& ay) {
    for (int k = 0; k < 3; ++k) {
        const int sx = x + ((ax + 32) >> 6), sy = y + ((ay + 32) >> 6);   // ⌊x + ax/64 + ½⌋
        if (w.mask && (unsigned)sx < (unsigned)IMG_W && (unsigned)sy < (unsigned)IMG_H) {
            const int i = sy * IMG_W + sx;
            if ((w.mask[i >> 3] >> (i & 7)) & 1) return;
        }
        ax /= 2; ay /= 2;
    }
    ax = ay = 0;
}
// One water pixel (x, y) from its nodes P[0], P[o] (o = x & 1), its fade e and bank distance. IN: every
// sample it can take lies inside the picture (sample_fx's clamps would change nothing: none).
template <bool IN>
JIG_INLINE uint16_t water_px(const Water& w, const WFrame& F, const WNode* P, int x, int y, int e, int ds,
                             const uint16_t* srow, const uint16_t* src, bool src_swapped, bool river) {
    const int o = x & 1;                                 // odd: halfway to the next node (even: (2a) >> 1 = a)
    int ax = (P[0].ax + P[o].ax) >> 1, ay = (P[0].ay + P[o].ay) >> 1;
    int bx = 0, by = 0;
    if (river) { bx = (P[0].bx + P[o].bx) >> 1; by = (P[0].by + P[o].by) >> 1; }
    if (e != 255) {                                      // (·e·257 >> 16 ≈ ·e / 255)
        const int ek = e * 257;
        ax = (ax * ek) >> 16; ay = (ay * ek) >> 16; bx = (bx * ek) >> 16; by = (by * ek) >> 16;
    }
    const int lim = (ds << 6) - 64;                      // shorter shifts can't reach the bank
    auto smp = [&](int dx, int dy) {
        return IN ? sample_fx_in(src, src_swapped, (x << 8) + dx * 4, (y << 8) + dy * 4)
                  : sample_fx(src, src_swapped, (x << 8) + dx * 4, (y << 8) + dy * 4);
    };
    uint16_t c;
    if (river) {                                         // the two phases of the flow, cross-faded
        if (abs(ax) + abs(ay) >= lim) water_clamp_q(w, x, y, ax, ay);
        if (abs(bx) + abs(by) >= lim) water_clamp_q(w, x, y, bx, by);
        c = rgb565_mix(smp(ax, ay), smp(bx, by), F.mix8);
    } else {
        if (abs(ax) + abs(ay) >= lim) water_clamp_q(w, x, y, ax, ay);
        if ((ax | ay) == 0) c = src_swapped ? bswap16(srow[x]) : srow[x];
        else c = smp(ax, ay);
    }
    if (wnode_extras(P[0]) | wnode_extras(P[o])) {       // painted light, foam, glints (only where a node has any)
        const int sh = (P[0].sh + P[o].sh) >> 1, fo = (P[0].fo + P[o].fo) >> 1, gl = (P[0].gl + P[o].gl) >> 1;
        if (sh) {
            const float k = 1.0f + (float)((sh * e * 257) >> 16) * (1.0f / 8192);
            if (k < 0.995f || k > 1.005f) c = rgb565_scale(c, k);
        }
        if (fo > 1) c = rgb565_brighten(c, (fo > 108 ? 108 : fo) * (1.0f / 127) * (e * (1.0f / 255)));
        if (gl > 0) {                                    // light caught on a steep ripple
            const int lum = ((c >> 11) * 8 + ((c >> 5) & 63) * 4 + (c & 31) * 8) / 3;
            if (lum > 120) c = rgb565_brighten(c, gl * (1.0f / 127) * (e * (1.0f / 255)) * (lum * (1.0f / 255)));
        }
    }
    return c;
}
// Render the water in rows [y0,y1) × cols [x0,x1) into `out` — a full frame (stride IMG_W) in out_swapped
// byte order — sampling the untouched photo `src` (src_swapped order). Only water pixels are written.
// F: the frame (water_frame_setup + water_snapshot at its start); its node rows must be worked out first
// (water_nodes_due / water_node_rows). lane: 0 or 1 — which core's scratch to use.
JIG_HOT void water_render(const Water& w, const WFrame& F, const uint16_t* src, bool src_swapped, uint16_t* out, bool out_swapped,
                          int x0, int y0, int x1, int y1, int lane = 0) {
    const WNodeCache* Cp = water_nodes_ptr();
    if (!w.edge || !w.sp_off || !Cp) return;
    const WNodeCache& C = *Cp;
    WNode* mid = water_nodes_ptr()->mid[lane & 1];
    if (y0 < w.by) y0 = w.by;
    if (y1 > w.by + w.bh) y1 = w.by + w.bh;
    const bool river = F.river;
    for (int y = y0; y < y1; ++y) {
        const int ry = y - w.by;
        const int s0 = w.sp_off[ry], s1 = w.sp_off[ry + 1];
        if (s0 == s1) continue;
        // this row's nodes: row y/2 (even y), else halfway between rows (y−1)/2 and (y+1)/2
        const WNode* N = C.n[(y >> 1) % W_NR];
        if (y & 1) {
            const WNode* N1 = C.n[((y >> 1) + 1) % W_NR];
            for (int sidx = s0; sidx < s1; ++sidx) {
                const int a = w.sp[2 * sidx] >> 1, b = ((w.sp[2 * sidx + 1] - 1) >> 1) + 2;
                for (int i = a; i < b; ++i) {
                    const WNode& p = N[i]; const WNode& q = N1[i]; WNode& m = mid[i];
                    m.ax = (int16_t)((p.ax + q.ax) >> 1); m.ay = (int16_t)((p.ay + q.ay) >> 1);
                    m.bx = (int16_t)((p.bx + q.bx) >> 1); m.by = (int16_t)((p.by + q.by) >> 1);
                    m.sh = (int16_t)((p.sh + q.sh) >> 1); m.fo = (int8_t)((p.fo + q.fo) >> 1); m.gl = (int8_t)((p.gl + q.gl) >> 1);
                }
            }
            N = mid;
        }
        // Where no sample can leave the picture (this row's largest displacement, its node rows' bound, can't
        // reach the edge): sampled without the clamps - the same pixels.
        const int D = 4 * ((y & 1) ? (C.dmax[(y >> 1) % W_NR] > C.dmax[((y >> 1) + 1) % W_NR] ? C.dmax[(y >> 1) % W_NR] : C.dmax[((y >> 1) + 1) % W_NR])
                                   : C.dmax[(y >> 1) % W_NR]);   // (1/256 px)
        const bool vin = (y << 8) - D >= 0 && (y << 8) + D <= ((IMG_H - 1) << 8) - 1;
        const int sxl = (D + 255) >> 8, sxr = ((((IMG_W - 1) << 8) - 1 - D) >> 8) + 1;   // x in [sxl, sxr): inside
        uint16_t* orow = out + y * IMG_W;
        const uint8_t* erow = w.edge + ry * w.bw - w.bx;
        const uint8_t* drow = w.dsafe + ry * w.bw - w.bx;
        const uint16_t* srow = src + y * IMG_W;
        for (int sidx = s0; sidx < s1; ++sidx) {
            int xa = w.sp[2 * sidx], xb = w.sp[2 * sidx + 1];
            if (xa < x0) xa = x0;
            if (xb > x1) xb = x1;
            const int ia = vin ? (xa > sxl ? xa : sxl) : xb, ib = vin ? (xb < sxr ? xb : sxr) : xb;
            int x = xa;
            for (; x < xb && x < ia; ++x) {
                const uint16_t c = water_px<false>(w, F, N + (x >> 1), x, y, erow[x], drow[x], srow, src, src_swapped, river);
                orow[x] = out_swapped ? bswap16(c) : c;
            }
            for (; x < ib; ++x) {
                const uint16_t c = water_px<true>(w, F, N + (x >> 1), x, y, erow[x], drow[x], srow, src, src_swapped, river);
                orow[x] = out_swapped ? bswap16(c) : c;
            }
            for (; x < xb; ++x) {
                const uint16_t c = water_px<false>(w, F, N + (x >> 1), x, y, erow[x], drow[x], srow, src, src_swapped, river);
                orow[x] = out_swapped ? bswap16(c) : c;
            }
        }
    }
}

// Each row's water extent within [y0,y1): the columns a strip actually needs to send.
inline void water_rows_extent(const Water& w, int y0, int y1, int& xa, int& xb) {
    xa = IMG_W; xb = 0;
    if (!w.sp_off) return;
    if (y0 < w.by) y0 = w.by;
    if (y1 > w.by + w.bh) y1 = w.by + w.bh;
    for (int y = y0; y < y1; ++y) {
        const int r = y - w.by;
        for (int k = w.sp_off[r]; k < w.sp_off[r + 1]; ++k) {
            if (w.sp[2 * k] < xa) xa = w.sp[2 * k];
            if (w.sp[2 * k + 1] > xb) xb = w.sp[2 * k + 1];
        }
    }
}

}  // namespace jig
