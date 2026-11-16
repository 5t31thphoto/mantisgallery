/**
 * Jiggle core — no Arduino / M5 dependencies, so it is unit-tested on a PC and mirrored 1:1 by
 * tools/jiggle_lab.py (the PC editor's preview). Keep the two in step.
 *
 * A zone is a soft shape (round / half-round / teardrop) on the photo. Every pixel inside samples the
 * pristine image at (pixel − offset), where
 *
 *     B      = d + g·q + M·(p − c)                     body motion at p
 *     offset = w · ( B + wt·(R − B) )                   R = rigid motion of the control point + t
 *
 *   w  weight: 1 in the body, easing to 0 past the outline (seamless edge)
 *   d  main spring (the mass swinging)          q  outer layer: lags d, so motion ripples outward
 *   g  how "free" a point is: 0 at the attached side, 1 at the free end
 *   M  bulge + volume-preserving stretch (narrows as it stretches, widens as it squashes)
 *   t  control point ("tip", optional, any shape): a small circle lined up over the apex. It shapes
 *      the zone — the mass hangs from the attached side toward it and bulges around it — and the
 *      area under it stays firm, moving as one piece and snapping slightly after the body.
 *
 * Flesh, not rubber: springs are well damped (settle in 2–3 wobbles), stiffen as they stretch, get
 * extra damping near rest (no endless shimmer), and bigger zones swing slower.
 *
 * Pixel buffers: `src` is the image canvas (RGB565, byte-swapped when `src_swapped`). Output from
 * render_rect is in display order (byte-swapped RGB565) for direct DMA.
 */
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <math.h>

namespace jig {

enum Shape : uint8_t { SHAPE_ROUND = 0, SHAPE_HALF = 1, SHAPE_TEAR = 2, SHAPE_COUNT = 3 };
enum Material : uint8_t { MAT_SOFT = 0, MAT_NATURAL = 1, MAT_FIRM = 2, MAT_COUNT = 3 };

static constexpr int   IMG_W = 320;
static constexpr int   IMG_H = 240;
static constexpr int   MAX_ZONES = 4;
static constexpr float R_MIN = 10.0f;
static constexpr float R_MAX = 150.0f;
static constexpr float R_DEFAULT = 45.0f;
static constexpr float BBOX_K = 1.13f;     // every shape fits inside radius 1.12 (local units)
static constexpr float MARGIN = 1.25f;     // jiggle field reaches past the drawn outline
static constexpr float PLATEAU = 0.55f;    // full strength from this fraction of the depth inward

static constexpr float TIP_R = 0.15f;      // control point radius (local units)
static constexpr float TIP_FADE = 2.0f;    // its firm area blends out to TIP_R·TIP_FADE
static constexpr float TIP_U = 0.0f, TIP_W = -0.50f;   // teardrop default: middle of the bulb

static constexpr float Q_MAX = 0.45f;      // outer-layer travel (fraction of main travel)
static constexpr float STRETCH_MAX = 0.07f;
static constexpr float REST_DAMP = 0.35f;  // extra damping ratio near rest
static constexpr float TIP_TRAVEL = 0.02f;  // control point's own swing, × r pixels (subtle)
static constexpr float PENDULUM = 0.30f;   // sideways swing lifts the mass toward its attachment

struct ShapeInfo {
    const char* name; const char* label;
    float depth;      // interior depth (local units): sets falloff width and travel
    float freq_mul;   // shape's effect on frequency (teardrops hang, so swing slower)
};
inline const ShapeInfo& shape_info(uint8_t t) {
    static const ShapeInfo k[SHAPE_COUNT] = {
        {"round", "Round", 1.00f, 1.00f},
        {"half",  "Half",  0.50f, 1.10f},
        {"tear",  "Tear",  0.55f, 0.88f},
    };
    return k[t < SHAPE_COUNT ? t : 0];
}

struct MaterialInfo {
    const char* name; const char* label;
    float f1, z1;      // main spring Hz (for a 50 px zone) and damping ratio
    float f2, z2;      // outer layer
    float stiffen;    // strain stiffening: k·(1 + stiffen·|d|²)
    float sens;       // response to motion
    float couple;     // how hard the outer layer is flung by the main mass
};
inline const MaterialInfo& material_info(uint8_t m) {
    static const MaterialInfo k[MAT_COUNT] = {
        {"soft",    "Soft",    2.0f, 0.20f, 3.1f, 0.17f, 1.2f, 1.15f, 0.85f},
        {"natural", "Natural", 2.6f, 0.25f, 3.9f, 0.21f, 1.8f, 1.00f, 0.70f},
        {"firm",    "Firm",    3.5f, 0.31f, 5.0f, 0.27f, 2.6f, 0.80f, 0.50f},
    };
    return k[m < MAT_COUNT ? m : (uint8_t)MAT_NATURAL];
}

// ---- Advanced tuning: per-zone overrides on top of the material preset (PC editor). -------------
// Saved in the .jig line as key=value; unset (NaN) means "use the preset". Ranges are enforced on
// load and keep the warp fold-free (checked in tests at every extreme).
enum Tune : uint8_t {
    T_F1, T_Z1, T_F2, T_Z2, T_STIFF, T_SENS, T_COUPLE,     // springs
    T_TRAVEL, T_BULGE, T_STRETCH, T_SAG, T_PEND,            // shape of the motion
    T_PSTIFF, T_PDAMP, T_PSIZE,                             // control point
    T_DANCE, T_SIDE,                                        // dance (side: -1 left, 0 centre, 1 right)
    TUNE_COUNT
};
struct TuneInfo { const char* key; const char* label; float lo, hi; };
inline const TuneInfo& tune_info(uint8_t t) {
    static const TuneInfo k[TUNE_COUNT] = {
        {"f1", "Main frequency (Hz @50px)", 0.8f, 6.0f},
        {"z1", "Main damping", 0.05f, 0.8f},
        {"f2", "Ripple frequency (Hz @50px)", 1.0f, 9.0f},
        {"z2", "Ripple damping", 0.05f, 0.8f},
        {"stiff", "Stiffening when stretched", 0.0f, 5.0f},
        {"sens", "Response to motion", 0.2f, 2.0f},
        {"couple", "Ripple strength", 0.0f, 1.5f},
        {"travel", "Max swing", 0.3f, 1.0f},
        {"bulge", "Bulge", 0.0f, 1.0f},
        {"stretch", "Stretch / squash", 0.0f, 1.0f},
        {"sag", "Gravity sag", 0.0f, 2.0f},
        {"pend", "Pendulum lift", 0.0f, 0.45f},
        {"pstiff", "Point stiffness", 1.0f, 4.0f},
        {"pdamp", "Point damping", 0.1f, 0.9f},
        {"psize", "Point firm area", 0.8f, 1.6f},
        {"dance", "Dance kick", 0.0f, 2.0f},
        {"side", "Dance side", -1.0f, 1.0f},
    };
    return k[t < TUNE_COUNT ? t : 0];
}

inline float clamp01(float v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }
inline float smooth01(float t) { t = clamp01(t); return t * t * (3.0f - 2.0f * t); }

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

// Where each shape is attached (local units): round hangs from its top, half sits on its flat side,
// the teardrop hangs from its point.
inline void shape_attach(uint8_t t, float& u, float& w) {
    u = 0;
    w = t == SHAPE_HALF ? -0.5f : (t == SHAPE_TEAR ? 0.95f : 0.9f);
}
// Default control-point spot per shape: round a little below centre, half near the dome top.
inline void shape_tip_default(uint8_t t, float& u, float& w) {
    u = 0;
    w = t == SHAPE_HALF ? 0.15f : (t == SHAPE_TEAR ? TIP_W : -0.25f);
}

// 0 = attached (moves least), 1 = free end (lags and swings most). In local units, before rotation.
// Without a control point: round — top attached, bottom free; half — flat side attached, dome top
// free; tear — pointed top attached, bulb bottom free. With one, "free" runs from the attachment
// toward the control point, so moving the point re-aims the whole hang.
inline float shape_free(uint8_t t, float u, float w, bool has_tip = false, float pu = 0, float pw = 0) {
    if (has_tip) {
        float au, aw;
        shape_attach(t, au, aw);
        float nu = pu - au, nw = pw - aw, L = sqrtf(nu * nu + nw * nw);
        if (L > 0.2f) {
            float along = ((u - au) * nu + (w - aw) * nw) / (L * L * 1.15f);
            float base = t == SHAPE_ROUND ? 0.35f : 0.0f;
            return base + (1.0f - base) * smooth01(along);
        }
    }
    switch (t) {
    case SHAPE_HALF: return smooth01((w + 0.5f) / 1.0f);
    case SHAPE_TEAR: return smooth01((0.95f - w) / 1.9f);
    default:         return 0.35f + 0.65f * smooth01((0.9f - w) / 1.8f);
    }
}

struct Zone {
    // ---- Definition (image pixels; rot in degrees clockwise). Saved in the .jig file.
    uint8_t shape = SHAPE_ROUND;
    uint8_t material = MAT_NATURAL;
    float   cx = IMG_W / 2, cy = IMG_H / 2, r = R_DEFAULT;
    int16_t rot = 0;
    bool    tip = false;                     // control point on/off (any shape)
    float   tip_u = TIP_U, tip_w = TIP_W;    // its position, local units (see shape_tip_default)
    float   tune[TUNE_COUNT] = {NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN};

    // ---- Runtime (prepare_zone)
    int16_t bx = 0, by = 0, bw = 0, bh = 0;  // bbox on screen (clipped)
    uint8_t* wmap = nullptr;                 // bw·bh body weight 0..255
    uint8_t* gmap = nullptr;                 // bw·bh free-ness 0..255
    int16_t* span = nullptr;                 // per row: first, last+1 x (screen) with w > 0
    int16_t tbx = 0, tby = 0, tbw = 0, tbh = 0;  // tip bbox (screen, clipped)
    uint8_t* tmap = nullptr;                 // tbw·tbh tip weight
    float tcx = 0, tcy = 0, tip_wc = 0, tip_gc = 0;  // tip centre (screen) and w, g there

    float w1 = 0, z1 = 0, w2 = 0, z2 = 0, wt = 0, zt = 0.38f, stiffen = 0, sens = 1, couple = 0;
    float lim_px = 0, s_lim = 0, st_max = STRETCH_MAX, sag = 1, pend = PENDULUM, dance = 1, tip_r = TIP_R;
    int8_t side = 2;                             // 2 = automatic (from position)

    // ---- Physics state (normalised: |d| ≤ 1 ↔ lim_px pixels)
    float dx = 0, dy = 0, vx = 0, vy = 0;        // main
    float qx = 0, qy = 0, qvx = 0, qvy = 0;      // outer layer
    float s = 0, vs = 0;                         // bulge
    float tx = 0, ty = 0, tvx = 0, tvy = 0;      // tip (|t| ≤ 1 ↔ TIP_TRAVEL·r px)
    bool  awake = false;

    // ---- Per-frame (stage_frame)
    float b0x = 0, b0y = 0, b1x = 0, b1y = 0;    // px
    float m00 = 0, m01 = 0, m10 = 0, m11 = 0;
    float otx = 0, oty = 0;                      // R: control point's rigid motion (px, unweighted)
    float hang_x = 0, hang_y = 1;                // unit vector (screen) from attachment toward the mass
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

inline bool tip_used(const Zone& z) { return z.tip; }

inline bool tune_set(const Zone& z, uint8_t t) { return !isnan(z.tune[t]); }
inline float tune_or(const Zone& z, uint8_t t, float dflt) {
    if (!tune_set(z, t)) return dflt;
    const TuneInfo& ti = tune_info(t);
    float v = z.tune[t];
    return v < ti.lo ? ti.lo : (v > ti.hi ? ti.hi : v);
}
inline float tip_radius(const Zone& z) { return TIP_R * tune_or(z, T_PSIZE, 1.0f); }

inline void tip_bbox(const Zone& z, int& x0, int& y0, int& w, int& h) {
    float c, s, px, py;
    zone_axes(z, c, s);
    from_local(z, c, s, z.tip_u, z.tip_w, px, py);
    float e = tip_radius(z) * TIP_FADE * z.r + 2.0f;
    int a0 = (int)floorf(px - e), b0 = (int)floorf(py - e), a1 = (int)ceilf(px + e), b1 = (int)ceilf(py + e);
    if (a0 < 0) a0 = 0;
    if (b0 < 0) b0 = 0;
    if (a1 > IMG_W) a1 = IMG_W;
    if (b1 > IMG_H) b1 = IMG_H;
    x0 = a0; y0 = b0; w = a1 > a0 ? a1 - a0 : 0; h = b1 > b0 ? b1 - b0 : 0;
}

// Bytes of working memory prepare_zone needs for this zone (one block).
inline size_t zone_mem_bytes(const Zone& z) {
    int bx, by, bw, bh;
    if (!zone_bbox(z, bx, by, bw, bh)) return 0;
    size_t n = (size_t)bw * bh;
    size_t bytes = ((2 * n + 3) & ~(size_t)3) + (size_t)bh * 2 * sizeof(int16_t);
    if (tip_used(z)) { int a, b, w, h; tip_bbox(z, a, b, w, h); bytes += (size_t)w * h; }
    return bytes;
}

inline float weight_at(const Zone& z, float u, float w) {
    const ShapeInfo& si = shape_info(z.shape);
    return smooth01(-shape_sdf(z.shape, u / MARGIN, w / MARGIN) / (PLATEAU * si.depth));
}

inline float tip_weight_at(const Zone& z, float u, float w) {
    float du = u - z.tip_u, dw = w - z.tip_w;
    float d = sqrtf(du * du + dw * dw), tr = tip_radius(z);
    return smooth01((tr * TIP_FADE - d) / (tr * (TIP_FADE - 0.5f)));
}

// Build maps (in `mem`, zone_mem_bytes long) and spring constants. Resets motion.
inline void prepare_zone(Zone& z, uint8_t* mem) {
    int bx, by, bw, bh;
    z.wmap = z.gmap = z.tmap = nullptr; z.span = nullptr;
    if (!zone_bbox(z, bx, by, bw, bh) || !mem) { z.bw = z.bh = 0; return; }
    z.bx = (int16_t)bx; z.by = (int16_t)by; z.bw = (int16_t)bw; z.bh = (int16_t)bh;
    size_t n = (size_t)bw * bh;
    z.wmap = mem;
    z.gmap = mem + n;
    z.span = (int16_t*)(mem + ((2 * n + 3) & ~(size_t)3));
    uint8_t* tip_mem = (uint8_t*)(z.span + 2 * bh);

    const ShapeInfo& si = shape_info(z.shape);
    float c, s;
    zone_axes(z, c, s);
    for (int y = 0; y < bh; ++y) {
        int first = -1, last = -1;
        for (int x = 0; x < bw; ++x) {
            float u, w;
            to_local(z, c, s, (float)(bx + x), (float)(by + y), u, w);
            uint8_t wv = (uint8_t)(weight_at(z, u, w) * 255.0f + 0.5f);
            z.wmap[y * bw + x] = wv;
            z.gmap[y * bw + x] = (uint8_t)(shape_free(z.shape, u, w, z.tip, z.tip_u, z.tip_w) * 255.0f + 0.5f);
            if (wv) { if (first < 0) first = x; last = x; }
        }
        z.span[2 * y]     = (int16_t)(first < 0 ? 0 : bx + first);
        z.span[2 * y + 1] = (int16_t)(first < 0 ? 0 : bx + last + 1);
    }

    if (tip_used(z)) {
        int a, b, w, h;
        tip_bbox(z, a, b, w, h);
        z.tbx = (int16_t)a; z.tby = (int16_t)b; z.tbw = (int16_t)w; z.tbh = (int16_t)h;
        z.tmap = tip_mem;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                float u, ww;
                to_local(z, c, s, (float)(a + x), (float)(b + y), u, ww);
                z.tmap[y * w + x] = (uint8_t)(tip_weight_at(z, u, ww) * 255.0f + 0.5f);
            }
        from_local(z, c, s, z.tip_u, z.tip_w, z.tcx, z.tcy);
        z.tip_wc = weight_at(z, z.tip_u, z.tip_w);
        z.tip_gc = shape_free(z.shape, z.tip_u, z.tip_w, true, z.tip_u, z.tip_w);
    } else {
        z.tbw = z.tbh = 0;
    }

    // Hang axis: from the attachment toward the control point (or the centre), on screen.
    {
        float au, aw, pu = 0, pw = 0, ax, ay, px, py;
        shape_attach(z.shape, au, aw);
        if (z.tip) { pu = z.tip_u; pw = z.tip_w; }
        from_local(z, c, s, au, aw, ax, ay);
        from_local(z, c, s, pu, pw, px, py);
        float hx = px - ax, hy = py - ay, hl = sqrtf(hx * hx + hy * hy);
        if (hl > 1e-3f) { z.hang_x = hx / hl; z.hang_y = hy / hl; }
    }

    // Bigger zones swing slower, like a heavier mass.
    float size_k = sqrtf(50.0f / z.r);
    size_k = size_k < 0.65f ? 0.65f : (size_k > 1.35f ? 1.35f : size_k);
    const MaterialInfo& mi = material_info(z.material);
    z.w1 = 6.2831853f * tune_or(z, T_F1, mi.f1) * si.freq_mul * size_k;
    z.w2 = 6.2831853f * tune_or(z, T_F2, mi.f2) * si.freq_mul * size_k;
    z.wt = z.w1 * 2.3f * tune_or(z, T_PSTIFF, 1.0f);   // control point: stiffer
    z.zt = tune_or(z, T_PDAMP, 0.38f);
    z.z1 = tune_or(z, T_Z1, mi.z1); z.z2 = tune_or(z, T_Z2, mi.z2);
    z.stiffen = tune_or(z, T_STIFF, mi.stiffen); z.sens = tune_or(z, T_SENS, mi.sens);
    z.couple = tune_or(z, T_COUPLE, mi.couple);
    z.sag = tune_or(z, T_SAG, 1.0f); z.pend = tune_or(z, T_PEND, PENDULUM);
    z.dance = tune_or(z, T_DANCE, 1.0f);
    z.side = tune_set(z, T_SIDE) ? (int8_t)lroundf(tune_or(z, T_SIDE, 0)) : (int8_t)2;
    z.tip_r = tip_radius(z);
    // Travel limits keep the warp from folding over itself (checked numerically in the tests).
    z.lim_px = 0.22f * si.depth * z.r * tune_or(z, T_TRAVEL, 1.0f);
    z.s_lim = 0.35f / (1.0f + 1.5f / (PLATEAU * si.depth)) * tune_or(z, T_BULGE, 1.0f);
    z.st_max = STRETCH_MAX * tune_or(z, T_STRETCH, 1.0f);

    z.dx = z.dy = z.vx = z.vy = z.qx = z.qy = z.qvx = z.qvy = 0;
    z.s = z.vs = z.tx = z.ty = z.tvx = z.tvy = 0;
    z.awake = false;
}

inline void clamp_vec(float& x, float& y, float& vx, float& vy, float maxlen) {
    float m2 = x * x + y * y;
    if (m2 <= maxlen * maxlen) return;
    float m = sqrtf(m2), nx = x / m, ny = y / m;
    x = nx * maxlen; y = ny * maxlen;
    float vr = vx * nx + vy * ny;
    if (vr > 0) { vx -= vr * nx; vy -= vr * ny; }
    vx *= 0.8f; vy *= 0.8f;
}

// One semi-implicit Euler step (h ≤ ~5 ms). tx/ty/ts: where the motion pushes the mass (normalised).
inline void step_zone(Zone& z, float h, float tgx, float tgy, float tgs) {
    tgx *= z.sens; tgy *= z.sens; tgs *= z.sens;

    // Main mass: stiffens as it stretches, extra damping near rest so it settles instead of shimmering.
    float d2 = z.dx * z.dx + z.dy * z.dy;
    float k1 = z.w1 * z.w1 * (1.0f + z.stiffen * d2);
    float sp2 = z.vx * z.vx + z.vy * z.vy;
    float c1 = 2.0f * (z.z1 + REST_DAMP / (1.0f + 4.0f * sp2)) * z.w1;
    float ax = k1 * (tgx - z.dx) - c1 * z.vx;
    float ay = k1 * (tgy - z.dy) - c1 * z.vy;
    z.vx += ax * h; z.vy += ay * h;
    z.dx += z.vx * h; z.dy += z.vy * h;
    clamp_vec(z.dx, z.dy, z.vx, z.vy, 1.0f);

    // Outer layer: flung by the main mass's acceleration, so it trails and overshoots a little.
    float drive = z.couple / (z.w1 * z.w1);
    float q2 = z.qx * z.qx + z.qy * z.qy;
    float k2 = z.w2 * z.w2 * (1.0f + 3.0f * q2);
    float qs2 = z.qvx * z.qvx + z.qvy * z.qvy;
    float c2 = 2.0f * (z.z2 + REST_DAMP / (1.0f + 4.0f * qs2)) * z.w2;
    float qax = -k2 * z.qx - c2 * z.qvx - drive * z.w2 * z.w2 * ax;
    float qay = -k2 * z.qy - c2 * z.qvy - drive * z.w2 * z.w2 * ay;
    z.qvx += qax * h; z.qvy += qay * h;
    z.qx += z.qvx * h; z.qy += z.qvy * h;
    clamp_vec(z.qx, z.qy, z.qvx, z.qvy, Q_MAX);

    // Bulge (breathing / squash from pushes toward or away from the screen).
    float ws = z.w1 * 1.3f;
    float cs = 2.0f * (z.z1 + REST_DAMP / (1.0f + 4.0f * z.vs * z.vs)) * ws;
    z.vs += (ws * ws * (1.0f + z.stiffen * z.s * z.s) * (tgs - z.s) - cs * z.vs) * h;
    z.s += z.vs * h;
    if (z.s > 1.0f)  { z.s = 1.0f;  if (z.vs > 0) z.vs = 0; }
    if (z.s < -1.0f) { z.s = -1.0f; if (z.vs < 0) z.vs = 0; }

    // Tip: stiff and well damped; lags the body at its spot, then snaps back.
    if (tip_used(z)) {
        float lim_ratio = z.lim_px / (TIP_TRAVEL * z.r);       // body units → tip units
        float bx = (ax + z.tip_gc * qax) * lim_ratio, by = (ay + z.tip_gc * qay) * lim_ratio;
        float ct = 2.0f * z.zt * z.wt;
        float tax = -z.wt * z.wt * z.tx - ct * z.tvx - 0.6f * bx;
        float tay = -z.wt * z.wt * z.ty - ct * z.tvy - 0.6f * by;
        z.tvx += tax * h; z.tvy += tay * h;
        z.tx += z.tvx * h; z.ty += z.tvy * h;
        clamp_vec(z.tx, z.ty, z.tvx, z.tvy, 1.0f);
    }
}

inline bool zone_settled(const Zone& z, float tx, float ty, float ts) {
    const float e = 0.004f, v = 0.03f;
    return fabsf(z.dx - tx * z.sens) < e && fabsf(z.dy - ty * z.sens) < e && fabsf(z.s - ts * z.sens) < e &&
           fabsf(z.qx) < e && fabsf(z.qy) < e && fabsf(z.tx) < 0.02f && fabsf(z.ty) < 0.02f &&
           fabsf(z.vx) < v && fabsf(z.vy) < v && fabsf(z.vs) < v && fabsf(z.qvx) < v && fabsf(z.qvy) < v &&
           fabsf(z.tvx) < 0.1f && fabsf(z.tvy) < 0.1f &&
           fabsf(tx) < e && fabsf(ty) < e && fabsf(ts) < e;
}

inline void zone_rest(Zone& z) {
    z.dx = z.dy = z.vx = z.vy = z.qx = z.qy = z.qvx = z.qvy = 0;
    z.s = z.vs = z.tx = z.ty = z.tvx = z.tvy = 0;
    z.awake = false;
}

// Finger poke at (px,py): shove content away from the finger, dent the middle.
inline void poke_zone(Zone& z, float px, float py, float strength = 1.0f) {
    float ax = z.cx - px, ay = z.cy - py;
    float m = sqrtf(ax * ax + ay * ay);
    float k = strength * z.w1 * 0.45f;
    if (m > 0.5f) { z.vx += k * ax / m; z.vy += k * ay / m; }
    z.vs -= strength * z.w1 * 0.5f;
    z.awake = true;
}

// Dance beat: drop and rebound. side = -1 left zone / +1 right / 0 centre; sign alternates each beat.
inline void dance_kick(Zone& z, float amp, int side, int sign) {
    if (z.side != 2) side = z.side;           // explicit side from the editor
    float k = amp * z.w1 * z.dance;
    z.vy += k * 0.70f * (1.0f + 0.5f * side * sign);
    z.vx += k * 0.25f * sign;
    z.vs += k * 0.20f;
    z.awake = true;
}

// Per-frame linear terms of the offset field.
inline void stage_frame(Zone* zs, int n) {
    for (int i = 0; i < n; ++i) {
        Zone& z = zs[i];
        z.b0x = z.dx * z.lim_px; z.b0y = z.dy * z.lim_px;
        z.b1x = z.qx * z.lim_px; z.b1y = z.qy * z.lim_px;
        // Pendulum: swinging across the hang axis pivots about the attachment, so the mass rises.
        float perp = -z.dx * z.hang_y + z.dy * z.hang_x;
        float lift = z.pend * perp * perp * z.lim_px;
        z.b0x -= lift * z.hang_x; z.b0y -= lift * z.hang_y;
        float sc = z.s * z.s_lim;
        // Bulge grows around the control point when there is one: sc·(p − P) = sc·(p − c) − sc·(P − c)
        if (tip_used(z)) { z.b0x -= sc * (z.tcx - z.cx); z.b0y -= sc * (z.tcy - z.cy); }
        float m = sqrtf(z.dx * z.dx + z.dy * z.dy);
        float e = z.st_max * (m < 1.0f ? m : 1.0f);
        float nx = m > 1e-4f ? z.dx / m : 0.0f, ny = m > 1e-4f ? z.dy / m : 0.0f;
        // Stretch along the swing, half as much squeeze across it (area roughly kept).
        float iso = sc - 0.5f * e, an = 1.5f * e;
        z.m00 = iso + an * nx * nx; z.m01 = an * nx * ny;
        z.m10 = an * nx * ny;       z.m11 = iso + an * ny * ny;
        if (tip_used(z)) {
            float rx = z.tcx - z.cx, ry = z.tcy - z.cy;
            z.otx = z.b0x + z.tip_gc * z.b1x + z.m00 * rx + z.m01 * ry + z.tx * TIP_TRAVEL * z.r;
            z.oty = z.b0y + z.tip_gc * z.b1y + z.m10 * rx + z.m11 * ry + z.ty * TIP_TRAVEL * z.r;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Pixels
// ---------------------------------------------------------------------------------------------
static constexpr uint32_t MASK_565X = 0x07E0F81Fu;
inline uint32_t spread565(uint16_t p) { return ((uint32_t)p | ((uint32_t)p << 16)) & MASK_565X; }
inline uint16_t pack565(uint32_t v) { return (uint16_t)((v | (v >> 16)) & 0xFFFFu); }
inline uint16_t bswap16(uint16_t v) { return (uint16_t)((v >> 8) | (v << 8)); }
inline uint16_t rd(const uint16_t* p, bool swapped) { return swapped ? bswap16(*p) : *p; }

// Bilinear sample at 8.8 fixed-point coordinates. Returns native RGB565.
inline uint16_t sample_fx(const uint16_t* src, bool swapped, int32_t sx8, int32_t sy8) {
    const int32_t maxx = ((IMG_W - 1) << 8) - 1, maxy = ((IMG_H - 1) << 8) - 1;
    if (sx8 < 0) sx8 = 0; else if (sx8 > maxx) sx8 = maxx;
    if (sy8 < 0) sy8 = 0; else if (sy8 > maxy) sy8 = maxy;
    int ix = sx8 >> 8, iy = sy8 >> 8;
    uint32_t fx = (uint32_t)(sx8 & 255) >> 3, fy = (uint32_t)(sy8 & 255) >> 3;
    const uint16_t* p = src + iy * IMG_W + ix;
    uint32_t a = spread565(rd(p, swapped)), b = spread565(rd(p + 1, swapped));
    uint32_t c = spread565(rd(p + IMG_W, swapped)), d = spread565(rd(p + IMG_W + 1, swapped));
    uint32_t top = ((a * (32 - fx) + b * fx) >> 5) & MASK_565X;
    uint32_t bot = ((c * (32 - fx) + d * fx) >> 5) & MASK_565X;
    return pack565(((top * (32 - fy) + bot * fy) >> 5) & MASK_565X);
}

// Float reference of the offset field at one pixel (tests + the Python port check against this).
inline void offset_ref(const Zone* zs, const int* ids, int n, float x, float y, float& ox, float& oy) {
    ox = oy = 0;
    for (int k = 0; k < n; ++k) {
        const Zone& z = zs[ids[k]];
        int lx = (int)x - z.bx, ly = (int)y - z.by;
        if (!z.wmap || lx < 0 || ly < 0 || lx >= z.bw || ly >= z.bh) continue;
        float w = z.wmap[ly * z.bw + lx] / 255.0f, g = z.gmap[ly * z.bw + lx] / 255.0f;
        float rx = x - z.cx, ry = y - z.cy;
        float bx = z.b0x + g * z.b1x + z.m00 * rx + z.m01 * ry;
        float by = z.b0y + g * z.b1y + z.m10 * rx + z.m11 * ry;
        if (z.tmap) {
            int tx = (int)x - z.tbx, ty = (int)y - z.tby;
            if (tx >= 0 && ty >= 0 && tx < z.tbw && ty < z.tbh) {
                float t = z.tmap[ty * z.tbw + tx] / 255.0f;
                bx += t * (z.otx - bx);
                by += t * (z.oty - by);
            }
        }
        ox += w * bx;
        oy += w * by;
    }
}

// Fast renderer: the rectangle [x0,x1)×[y0,y1) into `out` (stride x1−x0), display byte order.
// Integer maths, one pass per zone over its row span, each pixel sampled once however many zones
// overlap. `acc` must hold 2·IMG_W int32 of scratch.
inline void render_rect(const uint16_t* src, bool src_swapped, uint16_t* out,
                        const Zone* zs, const int* ids, int n,
                        int x0, int y0, int x1, int y1, int32_t* acc) {
    int32_t* accx = acc;
    int32_t* accy = acc + IMG_W;
    const int ow = x1 - x0;
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) { accx[x] = 0; accy[x] = 0; }
        bool any = false;
        for (int k = 0; k < n; ++k) {
            const Zone& z = zs[ids[k]];
            if (!z.wmap || y < z.by || y >= z.by + z.bh) continue;
            int ly = y - z.by;
            int sa = z.span[2 * ly], sb = z.span[2 * ly + 1];
            if (sa < x0) sa = x0;
            if (sb > x1) sb = x1;
            if (sa >= sb) continue;
            any = true;
            // lin = b0 + M·(p − c), 16.16 fixed, stepped along the row
            float ry = (float)y - z.cy, rx0 = (float)sa - z.cx;
            int32_t linx = (int32_t)lroundf((z.b0x + z.m00 * rx0 + z.m01 * ry) * 65536.0f);
            int32_t liny = (int32_t)lroundf((z.b0y + z.m10 * rx0 + z.m11 * ry) * 65536.0f);
            const int32_t dlx = (int32_t)lroundf(z.m00 * 65536.0f), dly = (int32_t)lroundf(z.m10 * 65536.0f);
            const int32_t b1x = (int32_t)lroundf(z.b1x * 256.0f), b1y = (int32_t)lroundf(z.b1y * 256.0f);
            const uint8_t* wr = z.wmap + ly * z.bw - z.bx;
            const uint8_t* gr = z.gmap + ly * z.bw - z.bx;
            // Control point rows: firm area pulls the body motion toward its rigid motion R.
            const uint8_t* tr = nullptr;
            int ta = 0, tb = 0;
            int32_t rx8 = 0, ry8 = 0;
            if (z.tmap && y >= z.tby && y < z.tby + z.tbh) {
                tr = z.tmap + (y - z.tby) * z.tbw - z.tbx;
                ta = z.tbx; tb = z.tbx + z.tbw;
                rx8 = (int32_t)lroundf(z.otx * 256.0f); ry8 = (int32_t)lroundf(z.oty * 256.0f);
            }
            for (int x = sa; x < sb; ++x, linx += dlx, liny += dly) {
                int32_t w = wr[x];
                if (!w) continue;
                int32_t g = gr[x];
                // ×257 >> 16 is an exact-enough ÷255 that stays in 32 bits (|t| < 32768 · 255 · 257 < 2³¹)
                int32_t tx = (linx >> 8) + ((g * b1x * 257) >> 16);   // 1/256 px
                int32_t ty = (liny >> 8) + ((g * b1y * 257) >> 16);
                if (tr && x >= ta && x < tb) {
                    int32_t t = tr[x];
                    if (t) {
                        t += t >> 7;                                    // 0..256
                        tx += (t * (rx8 - tx)) >> 8;
                        ty += (t * (ry8 - ty)) >> 8;
                    }
                }
                accx[x] += (w * tx * 257) >> 16;
                accy[x] += (w * ty * 257) >> 16;
            }
        }
        const uint16_t* row = src + y * IMG_W;
        uint16_t* o = out + (y - y0) * ow - x0;
        if (!any) {
            for (int x = x0; x < x1; ++x) o[x] = src_swapped ? row[x] : bswap16(row[x]);
            continue;
        }
        const int32_t y8 = y << 8;
        for (int x = x0; x < x1; ++x) {
            int32_t ox = accx[x], oy = accy[x];
            if (ox == 0 && oy == 0) { o[x] = src_swapped ? row[x] : bswap16(row[x]); continue; }
            o[x] = bswap16(sample_fx(src, src_swapped, (x << 8) - ox, y8 - oy));
        }
    }
}

// Union of the zones' row spans over rows [y0,y1): the strip's horizontal extent. false if empty.
inline bool strip_extent(const Zone* zs, const int* ids, int n, int y0, int y1, int& x0, int& x1) {
    x0 = IMG_W; x1 = 0;
    for (int k = 0; k < n; ++k) {
        const Zone& z = zs[ids[k]];
        if (!z.wmap) continue;
        int a = y0 > z.by ? y0 : z.by, b = y1 < z.by + z.bh ? y1 : z.by + z.bh;
        for (int y = a; y < b; ++y) {
            int sa = z.span[2 * (y - z.by)], sb = z.span[2 * (y - z.by) + 1];
            if (sa >= sb) continue;
            if (sa < x0) x0 = sa;
            if (sb > x1) x1 = sb;
        }
    }
    return x1 > x0;
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
