/**
 * Jiggle core — no Arduino / M5 dependencies, so it can be unit-tested on a PC.
 *
 * A "zone" is a soft shape (round / half-round / teardrop) placed on the image. Each zone owns a
 * weight map: 1.0 deep inside the shape, easing to 0.0 at its edge. Every frame, each pixel inside a
 * zone samples the pristine image at (pixel - weight * offset), where offset = spring displacement
 * plus a radial bulge. Because the weight is 0 at the edge, the warped patch always meets the rest of
 * the photo seamlessly.
 *
 * Pixel buffers: `src` is the M5Canvas buffer (RGB565, byte-swapped when `swapped` is true — the
 * sprite's usual layout; jiggle.cpp detects it at boot). `out` is plain little-endian RGB565: push it
 * with setSwapBytes(true), exactly like the .mg path in main.cpp.
 */
#pragma once
#include <stdint.h>
#include <math.h>

namespace jig {

enum Shape : uint8_t { SHAPE_ROUND = 0, SHAPE_HALF = 1, SHAPE_TEAR = 2, SHAPE_COUNT = 3 };

static constexpr int   IMG_W = 320;
static constexpr int   IMG_H = 240;
static constexpr int   MAX_ZONES = 4;
static constexpr float R_MIN = 10.0f;
static constexpr float R_MAX = 150.0f;
static constexpr float R_DEFAULT = 45.0f;
static constexpr float BBOX_K = 1.13f;     // every shape fits inside a circle of radius 1.12 (local units)
// The drawn outline is the object (fit it to the belly). The jiggle field reaches MARGIN× further so
// the object's own silhouette moves too, and reaches full strength at PLATEAU of the depth so the
// middle moves as one solid mass instead of swirling.
static constexpr float MARGIN = 1.25f;
static constexpr float PLATEAU = 0.55f;

// Feel per shape. depth = how deep the shape's interior is (local units) — sets the falloff width.
// freq/zeta = spring frequency (Hz, for a ~50 px zone) and damping ratio (lower = wobblier).
struct ShapeInfo { const char* name; const char* label; float depth; float freq_hz; float zeta; };
inline const ShapeInfo& shape_info(uint8_t t) {
    static const ShapeInfo k[SHAPE_COUNT] = {
        {"round", "Round", 1.00f, 2.6f, 0.11f},   // bellies, cheeks, bums
        {"half",  "Half",  0.50f, 3.4f, 0.14f},   // domes: flat side = attached edge
        {"tear",  "Tear",  0.55f, 2.1f, 0.09f},   // hanging drops: heavy bulb, pointed top
    };
    return k[t < SHAPE_COUNT ? t : 0];
}

// Signed distance in local units (u → right, w → up; negative inside). Shapes are centred on (0,0).
inline float shape_sdf(uint8_t t, float u, float w) {
    switch (t) {
    case SHAPE_HALF: {                       // flat edge at w = -0.5, dome top at w = +0.5, width ±1
        float y = w + 0.5f;
        float arc = sqrtf(u * u + y * y) - 1.0f;
        return arc > -y ? arc : -y;
    }
    case SHAPE_TEAR: {                       // uneven capsule: bulb r=0.55 below, tip r=0.10 above
        const float r1 = 0.55f, r2 = 0.10f, h = 1.30f;
        const float b = (r1 - r2) / h;
        const float a = sqrtf(1.0f - b * b);
        float px = fabsf(u), py = w + 0.415f;
        float k = -b * px + a * py;
        if (k < 0.0f) return sqrtf(px * px + py * py) - r1;
        if (k > a * h) return sqrtf(px * px + (py - h) * (py - h)) - r2;
        return a * px + b * py - r1;
    }
    default:
        return sqrtf(u * u + w * w) - 1.0f;
    }
}

struct Zone {
    // Definition (image pixels; rot in degrees, clockwise on screen). This is what gets saved.
    uint8_t shape = SHAPE_ROUND;
    float   cx = IMG_W / 2, cy = IMG_H / 2, r = R_DEFAULT;
    int16_t rot = 0;

    // Runtime (rebuilt by prepare_zone)
    int16_t  bx = 0, by = 0, bw = 0, bh = 0;  // bounding box on screen (clipped)
    uint8_t* wmap = nullptr;                  // bw*bh weights, 0..255
    float omega = 0, zeta = 0, lim_px = 0, s_lim = 0;
    float dx = 0, dy = 0, vx = 0, vy = 0;     // displacement, normalised: |d| <= 1 → lim_px pixels
    float s = 0, vs = 0;                      // bulge, normalised: |s| <= 1 → s_lim scale
    float dpx = 0, dpy = 0, sc = 0;           // per-frame pixel offsets (set by stage_frame)
    bool  awake = false;
};

inline void zone_axes(const Zone& z, float& c, float& s) {
    float a = z.rot * 0.017453293f;
    c = cosf(a); s = sinf(a);
}

inline void to_local(const Zone& z, float c, float s, float x, float y, float& u, float& w) {
    float dx = x - z.cx, dy = y - z.cy;
    u = ( dx * c + dy * s) / z.r;
    w = -(-dx * s + dy * c) / z.r;
}

inline void from_local(const Zone& z, float c, float s, float u, float w, float& x, float& y) {
    float a = u, b = -w;
    x = z.cx + z.r * (a * c - b * s);
    y = z.cy + z.r * (a * s + b * c);
}

// Hit test against the drawn outline, plus `pad` pixels so small zones are still easy to tap.
inline bool zone_contains(const Zone& z, float x, float y, float pad = 6.0f) {
    float c, s, u, w;
    zone_axes(z, c, s);
    to_local(z, c, s, x, y, u, w);
    return shape_sdf(z.shape, u, w) * z.r < pad;
}

// Screen bbox of the zone, clipped. Returns false if it's entirely off screen.
inline bool zone_bbox(const Zone& z, int& bx, int& by, int& bw, int& bh) {
    float e = BBOX_K * MARGIN * z.r + 2.0f;
    int x0 = (int)floorf(z.cx - e), y0 = (int)floorf(z.cy - e);
    int x1 = (int)ceilf(z.cx + e),  y1 = (int)ceilf(z.cy + e);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > IMG_W) x1 = IMG_W;
    if (y1 > IMG_H) y1 = IMG_H;
    bx = x0; by = y0; bw = x1 - x0; bh = y1 - y0;
    return bw > 0 && bh > 0;
}

inline size_t zone_wmap_bytes(const Zone& z) {
    int bx, by, bw, bh;
    return zone_bbox(z, bx, by, bw, bh) ? (size_t)bw * (size_t)bh : 0;
}

// Fill bbox, weight map (caller supplies a buffer of zone_wmap_bytes) and spring constants.
inline void prepare_zone(Zone& z, uint8_t* wmap) {
    int bx, by, bw, bh;
    if (!zone_bbox(z, bx, by, bw, bh) || !wmap) { z.bw = z.bh = 0; z.wmap = nullptr; return; }
    z.bx = (int16_t)bx; z.by = (int16_t)by; z.bw = (int16_t)bw; z.bh = (int16_t)bh;
    z.wmap = wmap;

    const ShapeInfo& si = shape_info(z.shape);
    float c, s;
    zone_axes(z, c, s);
    for (int y = 0; y < bh; ++y) {
        for (int x = 0; x < bw; ++x) {
            float u, w;
            to_local(z, c, s, (float)(bx + x), (float)(by + y), u, w);
            float t = -shape_sdf(z.shape, u / MARGIN, w / MARGIN) / (PLATEAU * si.depth);
            t = t < 0 ? 0 : (t > 1 ? 1 : t);
            float wt = t * t * (3.0f - 2.0f * t);
            wmap[y * bw + x] = (uint8_t)(wt * 255.0f + 0.5f);
        }
    }

    // Bigger zones swing slower, like a heavier mass.
    float size_k = sqrtf(50.0f / z.r);
    size_k = size_k < 0.6f ? 0.6f : (size_k > 1.6f ? 1.6f : size_k);
    z.omega = 6.2831853f * si.freq_hz * size_k;
    z.zeta = si.zeta;
    // Limits keep the warp from folding over itself: (weight slope × offset) must stay below 1.
    // Max slope is 1.5 / (PLATEAU·depth·MARGIN·r), so translation ≤ ~0.25·depth·r and bulge ≤ the
    // scale term below keep the sum under 1 even when both peak together.
    z.lim_px = 0.25f * si.depth * z.r;
    z.s_lim = 0.4f / (1.0f + 1.5f / (PLATEAU * si.depth));
    z.dx = z.dy = z.vx = z.vy = z.s = z.vs = 0;
    z.awake = false;
}

// One semi-implicit Euler step. tx/ty/ts are the targets the springs pull toward (normalised).
inline void step_zone(Zone& z, float h, float tx, float ty, float ts) {
    const float w = z.omega, w2 = w * w, c = 2.0f * z.zeta * w;
    z.vx += (w2 * (tx - z.dx) - c * z.vx) * h;
    z.vy += (w2 * (ty - z.dy) - c * z.vy) * h;
    z.dx += z.vx * h;
    z.dy += z.vy * h;

    const float ws = w * 1.3f;
    z.vs += (ws * ws * (ts - z.s) - 2.0f * z.zeta * ws * z.vs) * h;
    z.s += z.vs * h;

    float m2 = z.dx * z.dx + z.dy * z.dy;
    if (m2 > 1.0f) {                                 // hit the stop: clamp, drop outward speed
        float m = sqrtf(m2), nx = z.dx / m, ny = z.dy / m;
        z.dx = nx; z.dy = ny;
        float vr = z.vx * nx + z.vy * ny;
        if (vr > 0) { z.vx -= vr * nx; z.vy -= vr * ny; }
        z.vx *= 0.85f; z.vy *= 0.85f;
    }
    if (z.s > 1.0f)  { z.s = 1.0f;  if (z.vs > 0) z.vs = 0; }
    if (z.s < -1.0f) { z.s = -1.0f; if (z.vs < 0) z.vs = 0; }
}

inline bool zone_settled(const Zone& z, float tx, float ty, float ts) {
    return fabsf(z.dx - tx) < 0.004f && fabsf(z.dy - ty) < 0.004f && fabsf(z.s - ts) < 0.004f &&
           fabsf(z.vx) < 0.03f && fabsf(z.vy) < 0.03f && fabsf(z.vs) < 0.03f &&
           fabsf(tx) < 0.004f && fabsf(ty) < 0.004f && fabsf(ts) < 0.004f;
}

// Push the zone like a finger poke at (px,py): shove content away from the finger, dent the middle.
inline void poke_zone(Zone& z, float px, float py, float strength = 1.0f) {
    float ax = z.cx - px, ay = z.cy - py;
    float m = sqrtf(ax * ax + ay * ay);
    if (m > 0.5f) { z.vx += strength * 5.0f * ax / m; z.vy += strength * 5.0f * ay / m; }
    z.vs -= strength * 6.0f;
    z.awake = true;
}

// Dance beat: drop and rebound. side = -1 left zone / +1 right / 0 centre; sign alternates each beat,
// so a left/right pair takes turns (hips twerk, chest bounces a touch out of step). Scaled by the
// zone's own frequency so small, stiff zones and big, soft ones swing a similar share of their travel.
inline void dance_kick(Zone& z, float amp, int side, int sign) {
    float k = amp * z.omega;
    z.vy += k * 0.75f * (1.0f + 0.5f * side * sign);
    z.vx += k * 0.30f * sign;
    z.vs += k * 0.25f;
    z.awake = true;
}

inline void stage_frame(Zone* zs, int n) {
    for (int i = 0; i < n; ++i) {
        Zone& z = zs[i];
        z.dpx = z.dx * z.lim_px;
        z.dpy = z.dy * z.lim_px;
        z.sc = z.s * z.s_lim;
    }
}

// ---- pixels ----
static constexpr uint32_t MASK_565X = 0x07E0F81Fu;   // RGB565 spread so channels can be lerped in one go

inline uint32_t spread565(uint16_t p) { return ((uint32_t)p | ((uint32_t)p << 16)) & MASK_565X; }
inline uint16_t pack565(uint32_t v) { return (uint16_t)((v | (v >> 16)) & 0xFFFFu); }
inline uint16_t bswap16(uint16_t v) { return (uint16_t)((v >> 8) | (v << 8)); }

inline uint16_t rd(const uint16_t* p, bool swapped) { return swapped ? bswap16(*p) : *p; }

inline uint16_t sample_bilinear(const uint16_t* src, bool swapped, float sx, float sy) {
    if (sx < 0) sx = 0;
    if (sy < 0) sy = 0;
    if (sx > IMG_W - 1.001f) sx = IMG_W - 1.001f;
    if (sy > IMG_H - 1.001f) sy = IMG_H - 1.001f;
    int ix = (int)sx, iy = (int)sy;
    uint32_t fx = (uint32_t)((sx - ix) * 32.0f), fy = (uint32_t)((sy - iy) * 32.0f);
    const uint16_t* p = src + iy * IMG_W + ix;
    uint32_t a = spread565(rd(p, swapped)), b = spread565(rd(p + 1, swapped));
    uint32_t c = spread565(rd(p + IMG_W, swapped)), d = spread565(rd(p + IMG_W + 1, swapped));
    uint32_t top = ((a * (32 - fx) + b * fx) >> 5) & MASK_565X;
    uint32_t bot = ((c * (32 - fx) + d * fx) >> 5) & MASK_565X;
    return pack565(((top * (32 - fy) + bot * fy) >> 5) & MASK_565X);
}

// Render zone i's bbox into `out` (bw*bh, little-endian RGB565). Offsets from every zone that
// overlaps are summed, so overlapping zones blend instead of seaming.
inline void render_zone(const uint16_t* src, bool swapped, uint16_t* out, const Zone* zs, int n, int i) {
    const Zone& Z = zs[i];
    int ov[MAX_ZONES], no = 0;
    for (int j = 0; j < n && no < MAX_ZONES; ++j) {
        const Zone& g = zs[j];
        if (!g.wmap) continue;
        if (g.bx >= Z.bx + Z.bw || g.bx + g.bw <= Z.bx || g.by >= Z.by + Z.bh || g.by + g.bh <= Z.by) continue;
        ov[no++] = j;
    }
    for (int y = Z.by; y < Z.by + Z.bh; ++y) {
        const uint16_t* row = src + y * IMG_W;
        for (int x = Z.bx; x < Z.bx + Z.bw; ++x) {
            float ox = 0, oy = 0;
            for (int k = 0; k < no; ++k) {
                const Zone& g = zs[ov[k]];
                unsigned lx = (unsigned)(x - g.bx), ly = (unsigned)(y - g.by);
                if (lx >= (unsigned)g.bw || ly >= (unsigned)g.bh) continue;
                uint8_t wv = g.wmap[ly * g.bw + lx];
                if (!wv) continue;
                float wf = wv * (1.0f / 255.0f);
                ox += wf * (g.dpx + g.sc * (x - g.cx));
                oy += wf * (g.dpy + g.sc * (y - g.cy));
            }
            if (fabsf(ox) + fabsf(oy) < 0.03f) *out++ = rd(row + x, swapped);
            else *out++ = sample_bilinear(src, swapped, x - ox, y - oy);
        }
    }
}

// Outline of a shape in local units, by bisecting along rays from the centre (all shapes are convex).
inline void shape_outline(uint8_t shape, int npts, float* us, float* ws) {
    for (int i = 0; i < npts; ++i) {
        float a = 6.2831853f * i / npts;
        float du = cosf(a), dw = sinf(a);
        float lo = 0.0f, hi = 1.25f;
        for (int it = 0; it < 16; ++it) {
            float mid = 0.5f * (lo + hi);
            if (shape_sdf(shape, du * mid, dw * mid) < 0) lo = mid; else hi = mid;
        }
        us[i] = du * lo;
        ws[i] = dw * lo;
    }
}

}  // namespace jig
