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
static constexpr float REST_DAMP = 0.30f;  // extra damping ratio near rest
static constexpr float TIP_TRAVEL = 0.02f;  // control point's own swing, × r pixels (subtle)
static constexpr float PENDULUM = 0.30f;   // sideways swing lifts the mass toward its attachment
// Control point = apex of the 3D form whose base is the outline. It reshapes the weights into a dome
// that peaks at the point (the thickest, most distal tissue swings most), aims the lag/ripple there,
// and gives the form an axis: pushes toward/away from the screen move the apex along it.
static constexpr float FORM_MIX = 0.25f;     // weight eases off by this much toward the attachment (full at the apex)
static constexpr float FOLD_MARGIN = 0.12f;  // fold guard: the warp's area may shrink to no less than this
static constexpr float AXIS_K = 0.5f;        // apex travel along the form axis per unit of bulge

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
        {"soft",    "Soft",    2.0f, 0.17f, 3.1f, 0.14f, 1.2f, 1.15f, 0.85f},
        {"natural", "Natural", 2.6f, 0.21f, 3.9f, 0.18f, 1.8f, 1.00f, 0.70f},
        {"firm",    "Firm",    3.5f, 0.27f, 5.0f, 0.23f, 2.6f, 0.80f, 0.50f},
    };
    return k[m < MAT_COUNT ? m : (uint8_t)MAT_NATURAL];
}

// ---- Advanced tuning: per-zone overrides on top of the material preset (PC editor). -------------
// Saved in the .jig line as key=value; unset (NaN) means "use the preset". Ranges are enforced on
// load and keep the warp fold-free (checked in tests at every extreme).
enum Tune : uint8_t {
    T_F1, T_Z1, T_F2, T_Z2, T_STIFF, T_SENS, T_COUPLE,     // springs
    T_TRAVEL, T_BULGE, T_STRETCH, T_SAG, T_PEND,            // shape of the motion
    T_PROJ, T_TAIL, T_PSIZE,                                // form depth, teardrop tail, point hold
    T_DANCE, T_SIDE,                                        // dance (side: -1 left, 0 centre, 1 right)
    T_MASS, T_ANCHOR, T_SLACK, T_DOWN, T_CONTACT,           // matching the picture (see prepare_zone)
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
        {"travel", "Max swing", 0.3f, 1.6f},
        {"bulge", "Bulge", 0.0f, 1.0f},
        {"stretch", "Stretch / squash", 0.0f, 1.0f},
        {"sag", "Gravity sag", 0.0f, 2.0f},
        {"pend", "Pendulum lift", 0.0f, 0.45f},
        {"proj", "Projection (form depth)", 0.2f, 1.5f},
        {"tail", "Teardrop tail", 0.3f, 1.0f},
        {"psize", "Point shape hold", 0.8f, 1.6f},
        {"dance", "Dance kick", 0.0f, 2.0f},
        {"side", "Dance side", -1.0f, 1.0f},
        {"mass", "Mass / heaviness", 0.5f, 2.0f},
        {"anchor", "Anchored at the attachment", 0.0f, 0.8f},
        {"slack", "Slack when lifted", 0.0f, 1.0f},
        {"down", "Picture's down (degrees)", -180.0f, 180.0f},
        {"contact", "Presses on neighbours", 0.0f, 1.0f},
    };
    return k[t < TUNE_COUNT ? t : 0];
}

inline float clamp01(float v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }
inline float smooth01(float t) { t = clamp01(t); return t * t * (3.0f - 2.0f * t); }

// Signed distance in local units (u → right, w → up; negative inside). Shapes are centred on (0,0).
inline float shape_sdf(uint8_t t, float u, float w, float tail = 1.0f) {
    switch (t) {
    case SHAPE_HALF: {                       // flat edge at w = -0.5, dome top at w = +0.5, width ±1
        float y = w + 0.5f;
        float arc = sqrtf(u * u + y * y) - 1.0f;
        return arc > -y ? arc : -y;
    }
    case SHAPE_TEAR: {                       // uneven capsule: bulb r=0.55 below, tip r=0.10 above
        // tail (0.3–1) shortens the point end; the shape stays centred on its extent.
        const float r1 = 0.55f, r2 = 0.10f, h = 1.30f * tail;
        const float b = (r1 - r2) / h;
        const float a = sqrtf(1.0f - b * b);
        float px = fabsf(u), py = w + 0.415f + 0.65f * (tail - 1.0f);
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
inline void shape_attach(uint8_t t, float& u, float& w, float tail = 1.0f) {
    u = 0;
    w = t == SHAPE_HALF ? -0.5f : (t == SHAPE_TEAR ? 0.95f + 0.65f * (tail - 1.0f) : 0.9f);
}
// Default control-point spot per shape: round a little below centre, half near the dome top.
inline void shape_tip_default(uint8_t t, float& u, float& w, float tail = 1.0f) {
    u = 0;
    w = t == SHAPE_HALF ? 0.15f : (t == SHAPE_TEAR ? TIP_W - 0.65f * (tail - 1.0f) : -0.25f);
}

// 0 = attached (moves least), 1 = free end (lags and swings most). In local units, before rotation.
// Without a control point: round — top attached, bottom free; half — flat side attached, dome top
// free; tear — pointed top attached, bulb bottom free. With one, "free" runs from the attachment
// toward the control point, so moving the point re-aims the whole hang.
inline float shape_free(uint8_t t, float u, float w, bool has_tip = false, float pu = 0, float pw = 0, float tail = 1.0f) {
    if (has_tip) {
        float au, aw;
        shape_attach(t, au, aw, tail);
        float nu = pu - au, nw = pw - aw, L = sqrtf(nu * nu + nw * nw);
        if (L > 0.2f) {
            float along = ((u - au) * nu + (w - aw) * nw) / (L * L * 1.15f);
            float base = t == SHAPE_ROUND ? 0.35f : 0.0f;
            return base + (1.0f - base) * smooth01(along);
        }
    }
    switch (t) {
    case SHAPE_HALF: return smooth01((w + 0.5f) / 1.0f);
    case SHAPE_TEAR: return smooth01((0.95f + 0.65f * (tail - 1.0f) - w) / (1.9f + 1.3f * (tail - 1.0f)));
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
    char    name[16] = {0};                  // the Studio's layer name (kept through a Core2 save)
    float   tip_u = TIP_U, tip_w = TIP_W;    // its position, local units (see shape_tip_default)
    float   tune[TUNE_COUNT] = {NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
                                NAN, NAN, NAN, NAN, NAN};
    static_assert(TUNE_COUNT == 22, "extend the NAN list above");

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
    float slack = 0, contact = 1, mass = 1;      // lift-slack, neighbour contact, heaviness
    float down_s = 0, down_c = 1;                // picture's down: (sin, cos) of its angle from screen-down
    bool  pressed = false;                       // touching a neighbour this frame (contact_pass)
    float cax = 0, cay = 0;                      // neighbour's push last substep (accel, normalised)
    float damp_k = 1;                            // damping scale (< 1 while a sway is dying out: echo)

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
    float axis_x = 0, axis_y = 0;                // form axis on screen (px per unit bulge), 0 without a point
    float tip_travel = 0;                        // point's own swing (px)
    float fold_k = 1;                            // how much the fold guard scaled the motion (1 = none)
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


inline float zone_margin(const Zone& z);
inline bool zone_bbox(const Zone& z, int& bx, int& by, int& bw, int& bh) {
    float e = BBOX_K * zone_margin(z) * z.r + 2.0f;
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
inline float zone_tail(const Zone& z) { return z.shape == SHAPE_TEAR ? tune_or(z, T_TAIL, 1.0f) : 1.0f; }
inline float zone_sdf(const Zone& z, float u, float w) { return shape_sdf(z.shape, u, w, zone_tail(z)); }

// Is (u, w) a valid spot for the control point? An apex sits inside the form, not on its edge.
inline bool point_ok(const Zone& z, float u, float w) {
    return zone_sdf(z, u, w) < -tip_radius(z) * 0.5f;      // anywhere inside — droopy apexes sit at the edge
}

// How close the point is to the outline (0 = deep inside, 1 = on the edge). Points near an edge get a
// wider margin so the motion has room to fade out beyond the outline on that side.
inline float point_edge(const Zone& z) {
    if (!z.tip) return 0.0f;
    float d = -zone_sdf(z, z.tip_u, z.tip_w) / (0.45f * shape_info(z.shape).depth);
    return 1.0f - (d < 0 ? 0 : (d > 1 ? 1 : d));
}
inline float zone_margin(const Zone& z) { return MARGIN + 0.35f * point_edge(z); }

// Which way the form points, from the apex's offset to the base centre (cu, cw) in the zone's own
// frame — the zone's top is where it attaches to the body, so rotate the zone with the figure.
//   pointing down (hangs)  → droop +1: softer, heavier, swings and ripples, sags more
//   pointing up (holds itself up against its attachment) → droop −1: firm, springy
//   sideways only aims the axis. Typical rest pose (slightly below centre) → 0.
inline float zone_droop(const Zone& z, float cu, float cw, float r_eq) {
    (void)cu;
    if (!z.tip || r_eq <= 0) return 0.0f;
    float down = -(z.tip_w - cw) / r_eq;           // + when the apex is below the base centre
    float d = (down - 0.2f) / 0.45f;
    return d < -1 ? -1 : (d > 1 ? 1 : d);
}
// Pull the point inward (toward the shape's middle) until it's valid.
inline void clamp_point(Zone& z) {
    for (int i = 0; i < 40 && !point_ok(z, z.tip_u, z.tip_w); ++i) { z.tip_u *= 0.92f; z.tip_w *= 0.92f; }
    if (!point_ok(z, z.tip_u, z.tip_w)) shape_tip_default(z.shape, z.tip_u, z.tip_w, zone_tail(z));
}

// Fraction of the way from the control point out to the outline (scaled by `scale`): 0 at the apex,
// 1 on the edge. Cheap stand-in for "where along the dome's slope is this pixel".
inline float form_t(const Zone& z, float u, float w, float scale) {
    float du = u - z.tip_u, dw = w - z.tip_w;
    float dP = sqrtf(du * du + dw * dw);
    float dE = -shape_sdf(z.shape, u / scale, w / scale, zone_tail(z)) * scale;
    if (dE <= 0.0f) return 1.0f;
    return dP / (dP + dE);
}

// Lag / ripple map: free end away from the attachment; with a point, also scaled by how thick the
// form is there (1 − t²: a dome over the outline peaking at the point).
inline float zone_free(const Zone& z, float u, float w) {
    float g = shape_free(z.shape, u, w, z.tip, z.tip_u, z.tip_w, zone_tail(z));
    if (!z.tip) return g;
    float t = form_t(z, u, w, 1.0f);
    float h = 1.0f - t * t;
    return g * (0.75f + 0.25f * (h > 0 ? h : 0));
}

// Zones placed on the Core2 itself (no PC to fine-tune on) start out strong: clearly hanging with
// gravity, bigger swing, livelier response, more lift-and-drop, harder dance kicks. They're written
// to the .jig as ordinary dial settings, so the PC Jiggle tab shows them and can tune them back.
inline void device_boost(Zone& z) {
    z.tune[T_SAG] = 1.8f;
    z.tune[T_TRAVEL] = 1.45f;
    z.tune[T_SENS] = 1.35f;
    z.tune[T_PEND] = 0.4f;
    z.tune[T_SLACK] = 0.55f;
    z.tune[T_DANCE] = 1.4f;
}

// Hit test against the drawn outline, plus `pad` pixels so small zones are still easy to tap.
inline bool zone_contains(const Zone& z, float x, float y, float pad = 6.0f) {
    float c, s, u, w;
    zone_axes(z, c, s);
    to_local(z, c, s, x, y, u, w);
    return zone_sdf(z, u, w) * z.r < pad;
}

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
inline void clamp_point(Zone& z);
inline size_t zone_mem_bytes(const Zone& z0) {
    Zone z = z0;                                  // sized for the point where prepare_zone will put it
    if (z.tip) clamp_point(z);
    int bx, by, bw, bh;
    if (!zone_bbox(z, bx, by, bw, bh)) return 0;
    size_t n = (size_t)bw * bh;
    size_t bytes = ((2 * n + 3) & ~(size_t)3) + (size_t)bh * 2 * sizeof(int16_t);
    if (tip_used(z)) { int a, b, w, h; tip_bbox(z, a, b, w, h); bytes += (size_t)w * h; }
    return bytes;
}

inline float weight_at(const Zone& z, float u, float w) {
    const ShapeInfo& si = shape_info(z.shape);
    const float M = zone_margin(z);
    float w0 = smooth01(-zone_sdf(z, u / M, w / M) / (PLATEAU * si.depth));
    float anchor = tune_or(z, T_ANCHOR, z.tip ? FORM_MIX : 0.0f);
    if (anchor <= 0.0f) return w0;
    // Anchor only the attached side: weight ramps up from the attachment and is full from the apex
    // on, so the bulb and free end keep all their swing and the apex leads.
    float au, aw;
    shape_attach(z.shape, au, aw, zone_tail(z));
    float pu = z.tip ? z.tip_u : 0.0f, pw = z.tip ? z.tip_w : 0.0f;   // toward the apex (or the centre)
    float hu = pu - au, hw = pw - aw, hl = sqrtf(hu * hu + hw * hw);
    if (hl < 1e-3f) return w0;
    // ramp over at least one radius, so a point near its attachment can't make a weight cliff
    float a = ((u - au) * hu + (w - aw) * hw) / (hl * (hl > 1.0f ? hl : 1.0f));
    a = a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a);
    return w0 * (1.0f - anchor * (1.0f - a) * (1.0f - a));
}

inline float tip_weight_at(const Zone& z, float u, float w) {
    float du = u - z.tip_u, dw = w - z.tip_w;
    float d = sqrtf(du * du + dw * dw), tr = tip_radius(z);
    return smooth01((tr * TIP_FADE - d) / (tr * (TIP_FADE - 0.5f)));
}

inline void stage_frame(Zone* zs, int n);
inline void offset_ref(const Zone* zs, const int* ids, int n, float x, float y, float& ox, float& oy);
inline float fold_scale(Zone& z);

// Build maps (in `mem`, zone_mem_bytes long) and spring constants. Resets motion.
inline void prepare_zone(Zone& z, uint8_t* mem) {
    if (z.tip) clamp_point(z);
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
    float cu = 0, cw = 0, area = 0;                 // outline centroid and area (local units)
    for (int y = 0; y < bh; ++y) {
        int first = -1, last = -1;
        for (int x = 0; x < bw; ++x) {
            float u, w;
            to_local(z, c, s, (float)(bx + x), (float)(by + y), u, w);
            uint8_t wv = (uint8_t)(weight_at(z, u, w) * 255.0f + 0.5f);
            z.wmap[y * bw + x] = wv;
            z.gmap[y * bw + x] = (uint8_t)(zone_free(z, u, w) * 255.0f + 0.5f);
            if (wv) { if (first < 0) first = x; last = x; }
            if (zone_sdf(z, u, w) < 0) { cu += u; cw += w; area += 1; }
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
        z.tip_gc = zone_free(z, z.tip_u, z.tip_w);
    } else {
        z.tbw = z.tbh = 0;
    }

    // Hang axis: from the attachment toward the control point (or the centre), on screen.
    {
        float au, aw, pu = 0, pw = 0, ax, ay, px, py;
        shape_attach(z.shape, au, aw, zone_tail(z));
        if (z.tip) { pu = z.tip_u; pw = z.tip_w; }
        from_local(z, c, s, au, aw, ax, ay);
        from_local(z, c, s, pu, pw, px, py);
        float hx = px - ax, hy = py - ay, hl = sqrtf(hx * hx + hy * hy);
        if (hl > 1e-3f) { z.hang_x = hx / hl; z.hang_y = hy / hl; }
    }

    // What the point implies about the form (droop −1 firm/pointing up … +1 hanging). Defaults only —
    // any slider the user set wins.
    float r_eq = 0, droop = 0;
    if (area > 0) {
        cu /= area; cw /= area;
        r_eq = sqrtf(area / 3.14159265f) / z.r;                  // local units
        droop = zone_droop(z, cu, cw, r_eq);
    }
    // Form axis: base centre → apex is (Δ on screen, proj·R toward the viewer). A longer form swings
    // slower (heavier lever) and ripples more; a bulge moves the apex along the screen part of the axis.
    float form_k = 1.0f, axis_ux = 0, axis_uy = 0, tilt = 0, elong = 0;
    z.axis_x = z.axis_y = 0;
    if (z.tip && area > 0) {
        float du = z.tip_u - cu, dw = z.tip_w - cw;
        float proj = tune_or(z, T_PROJ, 0.6f - 0.25f * droop);   // perky projects, droopy hangs (aims the axis)
        float lever = tune_or(z, T_PROJ, 0.6f);                   // …but only a set dial lengthens the lever
        float dl = sqrtf(du * du + dw * dw);
        elong = dl / r_eq;
        float L = sqrtf(dl * dl + lever * r_eq * lever * r_eq);
        float rho = L / (0.6f * r_eq);
        form_k = 1.0f / sqrtf(rho);
        form_k = form_k < 0.9f ? 0.9f : (form_k > 1.3f ? 1.3f : form_k);
        L = sqrtf(dl * dl + proj * r_eq * proj * r_eq);           // axis direction uses the implied depth
        if (dl > 1e-4f) {
            float x0, y0, x1, y1;
            from_local(z, c, s, cu, cw, x0, y0);
            from_local(z, c, s, z.tip_u, z.tip_w, x1, y1);
            float hl = sqrtf((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
            axis_ux = (x1 - x0) / hl; axis_uy = (y1 - y0) / hl;
            tilt = dl / L;                                        // how much of the axis lies in the screen
        }
    }

    // Bigger zones swing slower, like a heavier mass (and a deeper form, via form_k).
    float size_k = sqrtf(50.0f / z.r);
    size_k = size_k < 0.65f ? 0.65f : (size_k > 1.35f ? 1.35f : size_k);
    float fk = form_k * (1.0f - (droop > 0 ? 0.06f : 0.15f) * droop);    // droopy: heavier, slower · perky: springier
    z.mass = tune_or(z, T_MASS, 1.0f);
    fk /= sqrtf(z.mass);                            // heavier: slower (ω ∝ 1/√m) …
    size_k *= fk;
    // Softer spring = more deflection for the same jolt (static deflection ∝ 1/ω²; we take 1/ω so
    // a heavy form swings visibly further while a perky one stays tighter).
    float form_travel = 1.0f / fk;
    form_travel = form_travel < 0.8f ? 0.8f : (form_travel > 1.6f ? 1.6f : form_travel);   // … and further
    const MaterialInfo& mi = material_info(z.material);
    z.w1 = 6.2831853f * tune_or(z, T_F1, mi.f1) * si.freq_mul * size_k;
    z.w2 = 6.2831853f * tune_or(z, T_F2, mi.f2) * si.freq_mul * size_k;
    z.wt = z.w1 * 2.3f;                       // the point's small follow-through: stiffer than the body
    z.zt = 0.38f;
    z.z1 = tune_or(z, T_Z1, mi.z1 - 0.03f * droop); z.z2 = tune_or(z, T_Z2, mi.z2);
    z.stiffen = tune_or(z, T_STIFF, mi.stiffen * (1.0f - 0.3f * droop)); z.sens = tune_or(z, T_SENS, mi.sens);
    float ripple = 1.0f + 0.3f * droop + 0.4f * (elong > 1 ? 1 : elong);
    z.couple = tune_or(z, T_COUPLE, mi.couple * (ripple > 1.6f ? 1.6f : ripple));
    z.sag = tune_or(z, T_SAG, 1.0f + 0.5f * droop);
    float pend = PENDULUM * (1.0f + 0.6f * droop);
    z.pend = tune_or(z, T_PEND, pend > 0.45f ? 0.45f : pend);
    z.dance = tune_or(z, T_DANCE, 1.0f);
    // Suspension slack: lifted against its hang the tissue goes loose (floats up), stretched along it
    // the skin catches (firm stop) — the lift-and-drop of a real hop. Hanging forms are slacker.
    z.slack = tune_or(z, T_SLACK, z.tip ? 0.25f + 0.25f * (droop > 0 ? droop : 0.0f) : 0.2f);
    z.contact = tune_or(z, T_CONTACT, 1.0f);
    {
        float a = tune_or(z, T_DOWN, 0.0f) * 0.017453293f;
        z.down_s = sinf(a); z.down_c = cosf(a);
    }
    z.side = tune_set(z, T_SIDE) ? (int8_t)lroundf(tune_or(z, T_SIDE, 0)) : (int8_t)2;
    z.tip_r = tip_radius(z);
    // Travel limits keep the warp from folding over itself (checked numerically in the tests).
    // A hanging (droopy) form is heavier and slower — and swings further, not less.
    z.lim_px = 0.22f * si.depth * z.r * tune_or(z, T_TRAVEL, form_travel);
    z.s_lim = 0.35f / (1.0f + 1.5f / (PLATEAU * si.depth)) * tune_or(z, T_BULGE, 1.0f);
    z.st_max = STRETCH_MAX * tune_or(z, T_STRETCH, 1.0f);
    // Axis motion: up to AXIS_K of the travel, scaled by how far the axis tilts into the screen plane.
    z.axis_x = AXIS_K * tilt * z.lim_px * axis_ux;
    z.axis_y = AXIS_K * tilt * z.lim_px * axis_uy;
    z.tip_travel = TIP_TRAVEL * z.r;

    // Fold guard: every motion amount scales the offsets linearly, so try the extreme poses once and
    // shrink them all together just enough that no pose folds the image. Safe zones keep full travel.
    float k = fold_scale(z);
    if (k < 1.0f) {
        z.lim_px *= k; z.s_lim *= k; z.st_max *= k;
        z.axis_x *= k; z.axis_y *= k; z.tip_travel *= k;
    }
    z.fold_k = k;

    z.dx = z.dy = z.vx = z.vy = z.qx = z.qy = z.qvx = z.qvy = 0;
    z.s = z.vs = z.tx = z.ty = z.tvx = z.tvy = 0;
    z.cax = z.cay = 0;
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
    if (z.slack > 0.0f) {
        float a = z.dx * z.hang_x + z.dy * z.hang_y;   // + stretched along the hang, − lifted
        k1 *= a >= 0.0f ? 1.0f + 0.8f * z.slack * a : 1.0f + 0.5f * z.slack * a;
    }
    float sp2 = z.vx * z.vx + z.vy * z.vy;
    float c1 = 2.0f * (z.z1 + REST_DAMP / (1.0f + 4.0f * sp2)) * z.w1 * z.damp_k;
    float ax = k1 * (tgx - z.dx) - c1 * z.vx;
    float ay = k1 * (tgy - z.dy) - c1 * z.vy;
    z.vx += ax * h; z.vy += ay * h;
    z.dx += z.vx * h; z.dy += z.vy * h;
    clamp_vec(z.dx, z.dy, z.vx, z.vy, 1.0f);

    // Outer layer: flung by the main mass's acceleration, so it trails and overshoots a little.
    // what the outer layer and the point feel is the net push: spring plus any neighbour pressing;
    // pressed flesh also swells toward the viewer in proportion to how hard it's pressed
    float press = sqrtf(z.cax * z.cax + z.cay * z.cay) / (z.w1 * z.w1);
    tgs += 0.5f * (press < 1.0f ? press : 1.0f);
    ax += z.cax; ay += z.cay; z.cax = z.cay = 0;
    float drive = z.couple / (z.w1 * z.w1);
    float q2 = z.qx * z.qx + z.qy * z.qy;
    float k2 = z.w2 * z.w2 * (1.0f + 3.0f * q2);
    float qs2 = z.qvx * z.qvx + z.qvy * z.qvy;
    float c2 = 2.0f * (z.z2 + REST_DAMP / (1.0f + 4.0f * qs2)) * z.w2 * z.damp_k;
    float qax = -k2 * z.qx - c2 * z.qvx - drive * z.w2 * z.w2 * ax;
    float qay = -k2 * z.qy - c2 * z.qvy - drive * z.w2 * z.w2 * ay;
    z.qvx += qax * h; z.qvy += qay * h;
    z.qx += z.qvx * h; z.qy += z.qvy * h;
    clamp_vec(z.qx, z.qy, z.qvx, z.qvy, Q_MAX);

    // Bulge (breathing / squash from pushes toward or away from the screen).
    float ws = z.w1 * 1.3f;
    float cs = 2.0f * (z.z1 + REST_DAMP / (1.0f + 4.0f * z.vs * z.vs)) * ws * z.damp_k;
    z.vs += (ws * ws * (1.0f + z.stiffen * z.s * z.s) * (tgs - z.s) - cs * z.vs) * h;
    z.s += z.vs * h;
    if (z.s > 1.0f)  { z.s = 1.0f;  if (z.vs > 0) z.vs = 0; }
    if (z.s < -1.0f) { z.s = -1.0f; if (z.vs < 0) z.vs = 0; }

    // Tip: stiff and well damped; lags the body at its spot, then snaps back.
    if (tip_used(z)) {
        float lim_ratio = z.lim_px / (z.tip_travel > 1e-6f ? z.tip_travel : 1e-6f);   // body units → tip units
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

// At rest where the targets hold it — possibly displaced (tilted: it hangs that way and stays).
inline bool zone_still(const Zone& z, float tx, float ty, float ts) {
    const float e = 0.015f, v = 0.06f;           // loose enough for accelerometer noise (≈0.05 px)
    return fabsf(z.dx - tx * z.sens) < e && fabsf(z.dy - ty * z.sens) < e && fabsf(z.s - ts * z.sens) < e &&
           fabsf(z.qx) < e && fabsf(z.qy) < e && fabsf(z.tx) < 0.02f && fabsf(z.ty) < 0.02f &&
           fabsf(z.vx) < v && fabsf(z.vy) < v && fabsf(z.vs) < v && fabsf(z.qvx) < v && fabsf(z.qvy) < v &&
           fabsf(z.tvx) < 0.1f && fabsf(z.tvy) < 0.1f;
}

// Real gravity (unit "down" from the IMU, screen axes) → where it pulls this zone, vs. how the picture
// was drawn. The picture's own down is (sin, cos) of T_DOWN from screen-down; turning the Core2 turns
// that world with it. Upright with down = 0 gives no offset. Scaled by the zone's sag by the caller.
inline void gravity_target(const Zone& z, const float dn[3], float kxy, float kz, float& gx, float& gy, float& gs) {
    const float s = z.down_s, c = z.down_c;
    const float wx = dn[1] * s + dn[0] * c, wy = -dn[0] * s + dn[1] * c;
    gx = kxy * (wx - s); gy = kxy * (wy - c); gs = kz * dn[2];
}

// Pixels from a zone's centre to its outline along screen direction (nx, ny).
inline float zone_extent(const Zone& z, float nx, float ny) {
    float c, s;
    zone_axes(z, c, s);
    const float u = nx * c + ny * s, w = -(-nx * s + ny * c);
    float lo = 0.0f, hi = 2.0f;
    for (int i = 0; i < 18; ++i) {
        float m = 0.5f * (lo + hi);
        if (zone_sdf(z, u * m, w * m) < 0.0f) lo = m; else hi = m;
    }
    return lo * z.r;
}

// Gap (px) between two zones' outlines along the line joining their centres (< 0: drawn overlapping).
inline float contact_gap(const Zone& a, const Zone& b) {
    float nx = b.cx - a.cx, ny = b.cy - a.cy, d = sqrtf(nx * nx + ny * ny);
    if (d < 1e-3f) return -1e9f;                          // same spot: never "contact" (degenerate)
    nx /= d; ny /= d;
    return d - zone_extent(a, nx, ny) - zone_extent(b, -nx, -ny);
}

// Neighbours press on each other instead of passing through: the mass parts are kept from closing
// in past where they were drawn touching (or their gap), shared by mass, closing speed taken out
// (a soft collision; real impacts bounce a little). The push is passed to step_zone (cax/cay):
// the outer layer feels it, and the squeeze swells the form toward the viewer.
// gap: n×n from contact_gap, computed once per load.
inline void contact_pass(Zone* zs, int n, const float* gap, float h) {
    for (int i = 0; i < n; ++i)                  // (pressed is cleared once per frame by the caller)
        for (int j = i + 1; j < n; ++j) {
            Zone& a = zs[i]; Zone& b = zs[j];
            if (!a.wmap || !b.wmap || (!a.awake && !b.awake)) continue;
            const float k = 0.5f * (a.contact + b.contact);
            const float g = gap[i * n + j];
            const float rmin = a.r < b.r ? a.r : b.r;           // drawn deep inside each other: one mass
            if (k <= 0.0f || g < -0.6f * rmin || g > a.lim_px + b.lim_px) continue;
            float nx = b.cx - a.cx, ny = b.cy - a.cy, l = sqrtf(nx * nx + ny * ny);
            nx /= l; ny /= l;
            const float allow = g > 0.0f ? g : 0.0f;
            const float close = (a.dx * a.lim_px - b.dx * b.lim_px) * nx + (a.dy * a.lim_px - b.dy * b.lim_px) * ny;
            const float p = close - allow;
            if (p <= 0.0f) continue;
            const float ma = a.r * a.r * a.mass, mb = b.r * b.r * b.mass;
            const float wa = mb / (ma + mb), wb = ma / (ma + mb);   // lighter one gives way more
            const float c = k * p;
            a.dx -= c * wa * nx / a.lim_px; a.dy -= c * wa * ny / a.lim_px;
            b.dx += c * wb * nx / b.lim_px; b.dy += c * wb * ny / b.lim_px;
            const float vc = (a.vx * a.lim_px - b.vx * b.lim_px) * nx + (a.vy * a.lim_px - b.vy * b.lim_px) * ny;
            if (vc > 0.0f) {
                const float dv = k * (1.0f + 0.15f * smooth01((vc - 10.0f) / 30.0f)) * vc;   // impacts bounce a little
                const float ga = dv * wa / a.lim_px, gb = dv * wb / b.lim_px;
                a.vx -= ga * nx; a.vy -= ga * ny; b.vx += gb * nx; b.vy += gb * ny;
                a.cax -= ga * nx / h; a.cay -= ga * ny / h;            // felt by the outer layer next step
                b.cax += gb * nx / h; b.cay += gb * ny / h;
            }
            a.awake = b.awake = true;
            a.pressed = b.pressed = true;
        }
}

// Pressed against a neighbour and no longer moving: rest there even though it can't reach its target.
inline bool zone_still_pressed(const Zone& z) {
    const float e = 0.015f, v = 0.06f;
    return z.pressed && fabsf(z.qx) < e && fabsf(z.qy) < e && fabsf(z.tx) < 0.02f && fabsf(z.ty) < 0.02f &&
           fabsf(z.vx) < v && fabsf(z.vy) < v && fabsf(z.vs) < v && fabsf(z.qvx) < v && fabsf(z.qvy) < v &&
           fabsf(z.tvx) < 0.1f && fabsf(z.tvy) < 0.1f;
}
inline bool zone_displaced(const Zone& z) {
    return fabsf(z.dx) > 1e-4f || fabsf(z.dy) > 1e-4f || fabsf(z.s) > 1e-4f;
}
// Stop moving but keep the current (held) shape.
inline void zone_freeze(Zone& z) {
    z.vx = z.vy = z.qx = z.qy = z.qvx = z.qvy = z.vs = z.tx = z.ty = z.tvx = z.tvy = 0;
    z.cax = z.cay = 0;
    z.awake = false;
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
// groove_x/y: direction and strength of your recent movement (0 = none); each beat swings that way too.
inline void dance_kick(Zone& z, float amp, int side, int sign, float groove_x = 0, float groove_y = 0) {
    if (z.side != 2) side = z.side;           // explicit side from the editor
    float k = amp * z.w1 * z.dance;
    z.vy += k * (0.70f * (1.0f + 0.5f * side * sign) + 0.6f * groove_y * sign);
    z.vx += k * (0.25f * sign + 0.6f * groove_x * sign);
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
        // Bulge grows around the control point when there is one: sc·(p − P) = sc·(p − c) − sc·(P − c),
        // and moves the apex along the form's axis (out when it swells, back toward the base when pressed).
        if (tip_used(z)) {
            z.b0x += -sc * (z.tcx - z.cx) + z.s * z.axis_x;
            z.b0y += -sc * (z.tcy - z.cy) + z.s * z.axis_y;
        }
        float m = sqrtf(z.dx * z.dx + z.dy * z.dy);
        float e = z.st_max * (m < 1.0f ? m : 1.0f);
        float nx = m > 1e-4f ? z.dx / m : 0.0f, ny = m > 1e-4f ? z.dy / m : 0.0f;
        // Stretch along the swing, half as much squeeze across it (area roughly kept).
        float iso = sc - 0.5f * e, an = 1.5f * e;
        z.m00 = iso + an * nx * nx; z.m01 = an * nx * ny;
        z.m10 = an * nx * ny;       z.m11 = iso + an * ny * ny;
        if (tip_used(z)) {
            float rx = z.tcx - z.cx, ry = z.tcy - z.cy;
            z.otx = z.b0x + z.tip_gc * z.b1x + z.m00 * rx + z.m01 * ry + z.tx * z.tip_travel;
            z.oty = z.b0y + z.tip_gc * z.b1y + z.m10 * rx + z.m11 * ry + z.ty * z.tip_travel;
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

// Fold guard. Offsets are linear in the motion amounts, so for a pose with Jacobian J (of the offset
// field) the warp at scale k has det(I − kJ) = 1 − k·tr J + k²·det J. Try the extreme poses on a
// 2-pixel grid and return the largest k ≤ 1 that keeps every det ≥ FOLD_MARGIN.
inline float fold_scale(Zone& z) {
    if (!z.wmap || z.bw < 5 || z.bh < 5) return 1.0f;
    const int id = 0, st = (z.bw <= 96 && z.bh <= 96) ? 1 : 2;   // small zones: every pixel
    const float margin = st == 1 ? FOLD_MARGIN : FOLD_MARGIN + 0.04f;  // coarse grid: keep more in hand
    const int gw = (z.bw - 1) / st + 1, gh = (z.bh - 1) / st + 1;
    static float ro[3][IMG_W / 2 + 2][2];            // rolling rows of offsets on the grid
    float kmin = 1.0f;
    for (int dir = 0; dir < 8; ++dir)
        for (int qs = -1; qs <= 1; qs += 2)
            for (int ss = -1; ss <= 1; ss += 2)
            for (int ts = -1; ts <= (tip_used(z) ? 1 : -1); ts += 2) {   // point swings either way
                float a = 0.7853982f * dir, ca = cosf(a), sa = sinf(a);
                z.dx = ca; z.dy = sa; z.qx = qs * Q_MAX * ca; z.qy = qs * Q_MAX * sa;
                z.s = (float)ss; z.tx = ts * ca; z.ty = ts * sa;
                stage_frame(&z, 1);
                for (int gy = 0; gy < gh; ++gy) {
                    float (*row)[2] = ro[gy % 3];
                    for (int gx = 0; gx < gw; ++gx)
                        offset_ref(&z, &id, 1, (float)(z.bx + gx * st), (float)(z.by + gy * st), row[gx][0], row[gx][1]);
                    if (gy < 2) continue;
                    float (*up)[2] = ro[(gy - 2) % 3], (*mid)[2] = ro[(gy - 1) % 3];
                    for (int gx = 1; gx < gw - 1; ++gx) {
                        float jxx = (mid[gx + 1][0] - mid[gx - 1][0]) / (2.0f * st);
                        float jyx = (mid[gx + 1][1] - mid[gx - 1][1]) / (2.0f * st);
                        float jxy = (row[gx][0] - up[gx][0]) / (2.0f * st);
                        float jyy = (row[gx][1] - up[gx][1]) / (2.0f * st);
                        float tr = jxx + jyy, de = jxx * jyy - jxy * jyx;
                        // smallest k > 0 with de·k² − tr·k + (1 − m) = 0
                        float c0 = 1.0f - margin, kk = 1e9f;
                        if (fabsf(de) < 1e-9f) { if (tr > 1e-9f) kk = c0 / tr; }
                        else {
                            float disc = tr * tr - 4.0f * de * c0;
                            if (disc >= 0) {
                                float sq = sqrtf(disc), k1 = (tr - sq) / (2 * de), k2 = (tr + sq) / (2 * de);
                                if (k1 > 0 && k1 < kk) kk = k1;
                                if (k2 > 0 && k2 < kk) kk = k2;
                            }
                        }
                        if (kk < kmin) kmin = kk;
                    }
                }
            }
    z.dx = z.dy = z.qx = z.qy = z.s = z.tx = z.ty = 0;
    return kmin;
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
inline void shape_outline(uint8_t shape, int npts, float* us, float* ws, float tail = 1.0f) {
    for (int i = 0; i < npts; ++i) {
        float a = 6.2831853f * i / npts;
        float du = cosf(a), dw = sinf(a);
        float lo = 0.0f, hi = 1.25f;
        for (int it = 0; it < 16; ++it) {
            float mid = 0.5f * (lo + hi);
            if (shape_sdf(shape, du * mid, dw * mid, tail) < 0) lo = mid; else hi = mid;
        }
        us[i] = du * lo;
        ws[i] = dw * lo;
    }
}

}  // namespace jig
