#pragma once
// Sway: hair locks, pleated cloth and pendants (earrings, charms). tools/mg_sway.py is the reference;
// this follows it step for step (the Studio's lab and the Core2 move them the same way).
//
//  * Hair: each lock is a guide drawn root → tip. The guide is a rod of nodes moved at a fixed 120 Hz
//    (Verlet): pulled back to its pose in the photo by g / (distance from the root) - a hanging thing's
//    own pendulum - and pushed by the change in "down" as the Core2 tilts, by jolts (it lags), a breeze,
//    beats and taps. Follow-the-leader keeps every link its length (Müller, Kim, Chentanez 2012); shape
//    memory relative to the link above keeps a curl a curl; no joint bends more than BEND_MAX past its
//    bend in the photo. Every pixel of the lock is bound to its nearest guide by (along, across) and
//    drawn on a triangle strip that follows it. Nothing outside the strips moves.
//  * Cloth: guides hang from the waistband at the fold lines (the dark valleys of the shading, or the
//    folds you mark); between two guides is a pleat panel of fixed material width. A skirt goes round the
//    body: a sideways swing turns it about its axis, so the pleats coming round the side bunch and the
//    ones facing you open. Squeezed, a panel turns out of the picture and darkens; it can't open past
//    flat or close past the fabric's thickness. Panels are drawn back to front.
//  * Pendant: one rigid piece on its pin - the full pendulum, balanced in the photo's pose, so real
//    gravity turns it (tip the Core2 90° and it hangs straight down).
//  * The body (the figure it rests on) is a collider: a node, or a pendant's outline point, that starts
//    off the body can't go into it.
//  * What it uncovers: the Studio's fill (splate lines, patch-voted); without one the Core2 continues the
//    surroundings itself - on the body only from the body right around it. How far a thing may move is
//    limited by what the fill can make true: lying on the body it moves a fraction of its width.
#include "jiggle_parallax.h"
#include <algorithm>
#include <vector>

namespace jig {

constexpr int   SWAY_MAX = 4;
constexpr int   SG_MAX = 6;                 // guides (hair locks; cloth: the waistband + up to 5 marked folds)
constexpr int   SG_PTS = 16;
constexpr int   SR_MAX = 9;                 // rods (cloth: fold lines, ≤ 8 panels)
constexpr int   SN_MAX = 12;                // nodes per rod
constexpr float SW_DT = 1.0f / 120.0f;
constexpr float SW_G = 1500.0f;             // gravity, px/s²
constexpr float SW_BEND = 0.95993109f;      // 55°
constexpr float SW_ON_BODY = 0.3f;
constexpr float SW_FREE = 0.5235988f;       // 30°
constexpr int SW_PUSH_MAX = 6;
constexpr float SW_PUSH_K = 45.0f;          // a form jiggling under a skirt (a zone on her) drags the cloth on it (mg_sway.PUSH_K)
constexpr float SW_HAIR_HOLD = 0.25f;         // (mg_sway.HAIR_HOLD)
constexpr float SW_CLOTH_HOLD = 0.1f;        // a skirt holds its pose by gravity: its pull to the photo pose scales with how much gravity points down the picture
constexpr float SW_CLOTH_FREE = 3.1415927f; // a skirt: no limit (upside down it falls right up; her hips stop it)
constexpr float SW_BIND_MAX = 12.0f;        // hair farther than this from every lock isn't carried (it stays)
constexpr float SW_EXT_MAX = 6.0f;          // a lock's strip runs at most this far past its tip

enum SwayKind : uint8_t { SK_HAIR = 0, SK_CLOTH, SK_PENDANT, SK_COUNT };
enum SParam : uint8_t { SP_SWING = 0, SP_BREEZE, SP_STIFF, SP_PLEAT, SP_COUNT };
struct SwayKindInfo { const char* key; const char* label; float c, j, wind, gain, mu; };
inline const SwayKindInfo& sway_kind_info(uint8_t k) {
    static const SwayKindInfo t[SK_COUNT] = {
        {"hair", "Hair", 4.5f, 500.0f, 110.0f, 0.9f, 0.3f},
        {"cloth", "Cloth", 3.4f, 450.0f, 70.0f, 1.0f, 0.0f},
        {"pendant", "Pendant", 0.9f, 1100.0f, 15.0f, 1.0f, 0.0f},
    };
    return t[k < SK_COUNT ? k : 0];
}
struct SParamInfo { const char* key; const char* label; float lo, hi, dflt; };
inline const SParamInfo& sparam_info(uint8_t p) {
    static const SParamInfo t[SP_COUNT] = {
        {"swing", "Swing", 0.0f, 2.5f, 1.0f},
        {"breeze", "Breeze while still", 0.0f, 1.5f, 0.25f},
        {"stiff", "Stiffness", 0.0f, 1.0f, 0.35f},
        {"pleat", "Pleat depth", 1.0f, 2.5f, 1.5f},
    };
    return t[p < SP_COUNT ? p : 0];
}

struct Sway {
    // ---- definition (saved)
    uint8_t  kind = SK_HAIR;
    float    prm[SP_COUNT];
    uint8_t* mask = nullptr;                  // MASK_BYTES: what moves
    int      ng = 0;
    int      gn[SG_MAX] = {0};
    float    gx[SG_MAX][SG_PTS], gy[SG_MAX][SG_PTS];
    char     name[16] = {0}, on[16] = {0};    // the Studio's name; the figure it rests on
    int16_t  lax = -1, lay = -1, lbx = -1, lby = -1;   // (an older file's "hang" pin: its guides are made from it)
    int      narm = 0;                        // cloth: her hips under it (the Studio's outline: kept, saved)
    int16_t  armx[9] = {0}, army[9] = {0};
    bool     arm_back = false;                // … she's seen from behind (the bikini bottom covers her seat)
    // runtime: forms jiggling under it (zones): cx, cy, r, vx, vy - set before each step
    float    push[SW_PUSH_MAX][5];
    int      npush = 0;
    int16_t* hipl = nullptr;                  // runtime: per row, the hips' left / right edge (−1 none)
    int16_t* hipr = nullptr;
    uint16_t* pdef = nullptr;                 // the Studio's fill (full frame, native RGB565) + which pixels it has
    uint8_t* pdef_has = nullptr;
    bool     pdef_ok = false;                 // (still matches the mask: saved back)
    // ---- runtime
    bool ready = false, awake = false, drawn = true;
    int  x0 = 0, y0 = 0, x1 = 0, y1 = 0;      // the screen rect it can touch (rest region + reach)
    int  rx0 = 0, ry0 = 0, rw = 0, rh = 0;    // its buffers' rect
    int8_t*   own = nullptr;                  // per pixel: rod / panel carrying it (−1 none)
    uint8_t*  hole = nullptr;                 // what it uncovers (the plate shows there)
    uint8_t*  alpha = nullptr;                // edge alpha (0..255)
    uint16_t* plate = nullptr;                // what's behind (native RGB565)
    uint16_t* photo = nullptr;                // the photo here (native)
    uint16_t* texd = nullptr;                 // edge pixels carrying their own colour (native)
    float*    c8 = nullptr;                   // the photo / the edge colour as RGB (3 floats a pixel), for drawing
    float*    t8 = nullptr;
    uint8_t*  solid = nullptr;                // per pixel: fully its own, unmatted (alpha 255, photo = edge colour)
    int  lx0 = 0, ly0 = 0, lx1 = 0, ly1 = 0;  // what it drew last time (screen)
    int  ux0 = 0, uy0 = 0, ux1 = 0, uy1 = 0;  // what this frame redraws: that and what it draws now
    int   nr = 0, nn[SR_MAX] = {0};
    float rx[SR_MAX][SN_MAX], ry[SR_MAX][SN_MAX];     // rest
    float x[SR_MAX][SN_MAX], y[SR_MAX][SN_MAX], xp[SR_MAX][SN_MAX], yp[SR_MAX][SN_MAX];
    float len[SR_MAX][SN_MAX], kn[SR_MAX][SN_MAX], cap[SR_MAX][SN_MAX], rsdf[SR_MAX][SN_MAX];
    bool  onb[SR_MAX][SN_MAX];
    float lw[SR_MAX][SN_MAX], rwd[SR_MAX][SN_MAX], ext[SR_MAX];  // hair: half widths, tip overhang
    float cuts[SR_MAX], W0[SR_MAX], Lm[SR_MAX], cos0[SR_MAX];      // cloth
    float fax = 0, fay = 0, fex = 1, fey = 0, fnx = 0, fny = 1, tmax = 0;
    float cc[SN_MAX], cR[SN_MAX], cpsi[SN_MAX][SR_MAX];
    float pinx = 0, piny = 0, rcx = 0, rcy = 0, mass = 1, I = 1, th = 0, thp = 0, tau0 = 0, tlen = 1;   // pendant
    int   no = 0;
    float ox[16], oy[16], ors[16];
    float t = 0, acc = 0, fade = 0, disp = 0;
    Sway() { for (int p = 0; p < SP_COUNT; ++p) prm[p] = sparam_info((uint8_t)p).dflt; }
};

inline void sway_free_runtime(Sway& s) {
    void* bufs[] = {s.own, s.hole, s.alpha, s.plate, s.photo, s.texd, s.c8, s.t8, s.solid, s.hipl, s.hipr};
    for (void* b : bufs) if (b) free(b);
    s.own = nullptr; s.hole = nullptr; s.alpha = nullptr; s.plate = s.photo = s.texd = nullptr; s.c8 = s.t8 = nullptr;
    s.hipl = s.hipr = nullptr; s.solid = nullptr;
    s.ready = false;
}
inline void sway_free(Sway& s) {
    sway_free_runtime(s);
    if (s.mask) free(s.mask);
    if (s.pdef) free(s.pdef);
    if (s.pdef_has) free(s.pdef_has);
    s.mask = nullptr; s.pdef = nullptr; s.pdef_has = nullptr; s.pdef_ok = false;
}
inline bool sway_ensure_mask(Sway& s, void* (*alloc)(size_t)) {
    if (s.mask) return true;
    s.mask = (uint8_t*)alloc(MASK_BYTES);
    if (s.mask) memset(s.mask, 0, MASK_BYTES);
    return s.mask != nullptr;
}
inline void sway_copy_def(Sway& d, const Sway& s, void* (*alloc)(size_t)) {
    sway_free(d);
    d = Sway();
    d.kind = s.kind; memcpy(d.prm, s.prm, sizeof(d.prm));
    d.narm = s.narm; memcpy(d.armx, s.armx, sizeof(d.armx)); memcpy(d.army, s.army, sizeof(d.army)); d.arm_back = s.arm_back;
    d.ng = s.ng; memcpy(d.gn, s.gn, sizeof(d.gn)); memcpy(d.gx, s.gx, sizeof(d.gx)); memcpy(d.gy, s.gy, sizeof(d.gy));
    memcpy(d.name, s.name, sizeof(d.name)); memcpy(d.on, s.on, sizeof(d.on));
    if (s.mask && sway_ensure_mask(d, alloc)) memcpy(d.mask, s.mask, MASK_BYTES);
    if (s.pdef && s.pdef_has && s.pdef_ok) {
        d.pdef = (uint16_t*)alloc((size_t)IMG_W * IMG_H * 2);
        d.pdef_has = (uint8_t*)alloc(MASK_BYTES);
        if (d.pdef && d.pdef_has) { memcpy(d.pdef, s.pdef, (size_t)IMG_W * IMG_H * 2); memcpy(d.pdef_has, s.pdef_has, MASK_BYTES); d.pdef_ok = true; }
    }
}
inline int sway_count(const Sway& s) {
    if (!s.mask) return 0;
    int n = 0;
    for (int i = 0; i < MASK_BYTES; ++i) { uint8_t b = s.mask[i]; while (b) { n += b & 1; b >>= 1; } }
    return n;
}
inline bool sway_ok(const Sway& s) {
    if (sway_count(s) < 12 || s.ng < 1) return false;
    if (s.kind == SK_HAIR) { for (int g = 0; g < s.ng; ++g) if (s.gn[g] < 2) return false; return true; }
    return s.gn[0] >= (s.kind == SK_PENDANT ? 1 : 2);   // (cloth: then any marked folds, one point each)
}

// Guides where none were drawn (or from an older file's pin): hair - one lock down the middle of the
// selection, root at its top; cloth - the waistband across its top row; pendant - pinned at its top.
inline void sway_default_guides(Sway& s) {
    if (!s.mask || s.ng > 0) return;
    int top = -1, bot = -1;
    for (int y = 0; y < IMG_H && top < 0; ++y) for (int x = 0; x < IMG_W; ++x) if (mask_get(s.mask, x, y)) { top = y; break; }
    for (int y = IMG_H - 1; y >= 0 && bot < 0; --y) for (int x = 0; x < IMG_W; ++x) if (mask_get(s.mask, x, y)) { bot = y; break; }
    if (top < 0) return;
    auto row_span = [&](int y, int& a, int& b) {
        a = -1; b = -1;
        for (int x = 0; x < IMG_W; ++x) if (mask_get(s.mask, x, y)) { if (a < 0) a = x; b = x; }
    };
    int a, b;
    row_span(top, a, b);
    if (s.kind == SK_CLOTH) {
        float ax = (float)a, ay = (float)top, bx = (float)b, by = (float)top;
        if (s.lax >= 0 && s.lbx >= 0 && (s.lax != s.lbx || s.lay != s.lby)) { ax = s.lax; ay = s.lay; bx = s.lbx; by = s.lby; }
        if (bx - ax < 2 && fabsf(by - ay) < 2) bx = ax + 2;
        s.gx[0][0] = ax; s.gy[0][0] = ay; s.gx[0][1] = bx; s.gy[0][1] = by; s.gn[0] = 2; s.ng = 1;
        return;
    }
    float px = (a + b) * 0.5f, py = (float)top;
    if (s.lax >= 0) { px = (s.lax + (s.lbx >= 0 ? s.lbx : s.lax)) * 0.5f; py = (s.lay + (s.lby >= 0 ? s.lby : s.lay)) * 0.5f; }
    if (s.kind == SK_PENDANT) { s.gx[0][0] = px; s.gy[0][0] = py; s.gn[0] = 1; s.ng = 1; return; }
    // hair: root, then the middle of the selection every few rows down to its lowest row
    int n = 0;
    s.gx[0][n] = px; s.gy[0][n] = py; ++n;
    const int steps = std::min(SG_PTS - 1, std::max(2, (bot - (int)py) / 8));
    for (int k = 1; k <= steps; ++k) {
        const int y = (int)py + (bot - (int)py) * k / steps;
        row_span(y, a, b);
        if (a < 0) continue;
        s.gx[0][n] = (a + b) * 0.5f; s.gy[0][n] = (float)y; ++n;
    }
    if (n < 2) { s.gx[0][1] = px; s.gy[0][1] = py + 4; n = 2; }
    s.gn[0] = n; s.ng = 1;
}

// ---- body: a signed distance field (px × 8, > 0 off the body), shared by all of them -----------------
inline int16_t* sway_body_sdf(const uint8_t* body, void* (*alloc)(size_t)) {
    const int N = IMG_W * IMG_H;
    int16_t* sdf = (int16_t*)alloc((size_t)N * 2);
    int16_t* nx = (int16_t*)alloc((size_t)N * 2);
    int16_t* ny = (int16_t*)alloc((size_t)N * 2);
    if (!sdf || !nx || !ny) { if (sdf) free(sdf); if (nx) free(nx); if (ny) free(ny); return nullptr; }
    for (int pass = 0; pass < 2; ++pass) {               // 0: distance to the body; 1: to off it
        bool any = false;
        for (int i = 0; i < N; ++i) {
            const bool b = bget_(body, i);
            if (b == (pass == 0)) { nx[i] = (int16_t)(i % IMG_W); ny[i] = (int16_t)(i / IMG_W); any = true; } else nx[i] = -1;
        }
        if (any) par_nearest(nx, ny);
        for (int i = 0; i < N; ++i) {
            const bool b = bget_(body, i);
            if (b == (pass == 1)) {
                float d = 4000.0f;
                if (nx[i] >= 0) { const float dx = nx[i] - (i % IMG_W), dy = ny[i] - (i / IMG_W); d = sqrtf(dx * dx + dy * dy); }
                const float v = pass == 0 ? d - 0.5f : -d + 0.5f;
                sdf[i] = (int16_t)lroundf(std::max(-4000.0f, std::min(4000.0f, v * 8.0f)));
            }
        }
    }
    free(nx); free(ny);
    return sdf;
}
// What they rest on (mg_sway.rest_body): the body without the things that move, plus each of their
// pixels whose nearest pixel that doesn't move is the body. In place; nx/ny: 2 × IMG_W·IMG_H int16 scratch.
inline void sway_rest_body(uint8_t* body, const uint8_t* const* masks, int n, int16_t* nx, int16_t* ny) {
    const int N = IMG_W * IMG_H;
    auto mv = [&](int i) { for (int k = 0; k < n; ++k) if (masks[k] && bget_(masks[k], i)) return true; return false; };
    bool anym = false, anyb = false, anys = false;
    for (int i = 0; i < N; ++i) { const bool m = mv(i), b = bget_(body, i); anym |= m; anyb |= b && !m; anys |= !b && !m; }
    if (!anym || !anyb) return;
    if (!anys) { for (int i = 0; i < N; ++i) if (mv(i)) bset_(body, i); return; }
    std::vector<float> db(N);
    for (int pass = 0; pass < 2; ++pass) {              // 0: distance to the body, 1: to the scene
        for (int i = 0; i < N; ++i) {
            const bool m = mv(i), b = bget_(body, i);
            if (!m && (pass == 0 ? b : !b)) { nx[i] = (int16_t)(i % IMG_W); ny[i] = (int16_t)(i / IMG_W); } else nx[i] = -1;
        }
        par_nearest(nx, ny);
        for (int i = 0; i < N; ++i) {
            if (!mv(i)) continue;
            const float dx = nx[i] - (i % IMG_W), dy = ny[i] - (i / IMG_W), d = nx[i] >= 0 ? sqrtf(dx * dx + dy * dy) : 1e6f;
            if (pass == 0) db[i] = d;
            else if (db[i] <= d) bset_(body, i); else body[i >> 3] &= (uint8_t)~(1 << (i & 7));
        }
    }
}

inline float sdf_at(const int16_t* sdf, float x, float y, float* gx = nullptr, float* gy = nullptr) {
    x = x < 0 ? 0 : (x > IMG_W - 1.001f ? IMG_W - 1.001f : x);
    y = y < 0 ? 0 : (y > IMG_H - 1.001f ? IMG_H - 1.001f : y);
    const int ix = (int)x, iy = (int)y;
    const float fx = x - ix, fy = y - iy;
    const float s00 = sdf[iy * IMG_W + ix] * 0.125f, s10 = sdf[iy * IMG_W + ix + 1] * 0.125f;
    const float s01 = sdf[(iy + 1) * IMG_W + ix] * 0.125f, s11 = sdf[(iy + 1) * IMG_W + ix + 1] * 0.125f;
    if (gx) {
        float ax = (s10 - s00) * (1 - fy) + (s11 - s01) * fy, ay = (s01 - s00) * (1 - fx) + (s11 - s10) * fx;
        float l = sqrtf(ax * ax + ay * ay);
        if (l < 1e-9f) l = 1.0f;
        *gx = ax / l; *gy = ay / l;
    }
    return (s00 * (1 - fx) + s10 * fx) * (1 - fy) + (s01 * (1 - fx) + s11 * fx) * fy;
}

// ---- small helpers ------------------------------------------------------------------------------------
// RGB565 → 0..255 per channel, as the Studio's lab reads it (x·255 / 31 or 63, integer): tables, no divides
struct SwLevels {
    float f5[32], f6[64];
    SwLevels() { for (int i = 0; i < 32; ++i) f5[i] = (float)(i * 255 / 31); for (int i = 0; i < 64; ++i) f6[i] = (float)(i * 255 / 63); }
};
static const SwLevels s_swlv;
JIG_INLINE void sw_unpack(uint16_t v, float& r, float& g, float& b) {
    r = s_swlv.f5[v >> 11]; g = s_swlv.f6[(v >> 5) & 63]; b = s_swlv.f5[v & 31];
}
// One channel back to 5 / 6 bits: round(v·m / 255), clamped - bit for bit as (int)(v·m / 255.0f + 0.5f).
// The Core2's FPU has no divide (a divide is a long instruction sequence), so this multiplies by m / 255
// instead: that lands within ~3e-5 of the divided value, so whenever the result isn't that close to a
// rounding edge it's the same integer; the rare one that is takes the exact divide.
JIG_INLINE int sw_q(float v, int m, float k, int hi) {
    const float u = v * k + 0.5f;
    if (u < -1e-3f) return 0;
    if (u >= 1e-3f) {
        const int i = (int)u;
        const float f = u - (float)i;
        if (f > 1e-4f && f < 1.0f - 1e-4f) return i > hi ? hi : i;
    }
    const int i = (int)(v * m / 255.0f + 0.5f);
    return i < 0 ? 0 : (i > hi ? hi : i);
}
// rintf (to nearest, ties to even) for 0 ≤ v < 2^23, without the library call
JIG_INLINE int sw_rint_pos(float v) {
    const int i = (int)v;
    const float f = v - (float)i;                       // (exact)
    return (f > 0.5f || (f == 0.5f && (i & 1))) ? i + 1 : i;
}
JIG_INLINE uint16_t sw_pack(float r, float g, float b) {
    return (uint16_t)((sw_q(r, 31, 31.0f / 255.0f, 31) << 11) | (sw_q(g, 63, 63.0f / 255.0f, 63) << 5) | sw_q(b, 31, 31.0f / 255.0f, 31));
}
// a polyline resampled to n evenly spaced nodes (in double, as the Studio's lab does it)
inline int sw_resample(const float* px, const float* py, int np, float step, int nmax, float* ox, float* oy) {
    double cum[SG_PTS] = {0};
    for (int i = 1; i < np; ++i) cum[i] = cum[i - 1] + sqrt((double)(px[i] - px[i - 1]) * (px[i] - px[i - 1]) + (double)(py[i] - py[i - 1]) * (py[i] - py[i - 1]));
    const double L = cum[np - 1];
    const double q = L / step;
    long r = lround(q);
    if (fabs(q - floor(q) - 0.5) < 1e-12 && (r & 1)) --r;     // (round half to even)
    int n = (int)r + 1;
    n = n < 3 ? 3 : (n > nmax ? nmax : n);
    for (int k = 0; k < n; ++k) {
        const double s = L * k / (n - 1);
        int j = 0;
        while (j < np - 2 && cum[j + 1] < s) ++j;
        const double seg = cum[j + 1] - cum[j], t = seg > 1e-12 ? (s - cum[j]) / seg : 0;
        ox[k] = (float)(px[j] + (px[j + 1] - px[j]) * t); oy[k] = (float)(py[j] + (py[j + 1] - py[j]) * t);
    }
    return n;
}
// distance to a polyline, arc length at the closest point, signed distance across (left +)
inline void sw_bind(float X, float Y, const float* px, const float* py, int n, float& dist, float& S, float& D) {
    double best = 1e18, cum = 0, Sd = 0, Dd = 0;
    for (int i = 0; i < n - 1; ++i) {
        const double dx = (double)px[i + 1] - px[i], dy = (double)py[i + 1] - py[i], l2 = dx * dx + dy * dy, l = sqrt(l2);
        if (l2 < 1e-12) continue;
        double t = ((X - px[i]) * dx + (Y - py[i]) * dy) / l2;
        if (i > 0 && t < 0) t = 0;
        if (i < n - 2 && t > 1) t = 1;
        const double qx = px[i] + t * dx, qy = py[i] + t * dy, dd = (X - qx) * (X - qx) + (Y - qy) * (Y - qy);
        if (dd < best) { best = dd; Sd = cum + t * l; Dd = ((X - px[i]) * (-dy) + (Y - py[i]) * dx) / l; }
        cum += l;
    }
    dist = (float)sqrt(best); S = (float)Sd; D = (float)Dd;
}
inline bool sw_in(const Sway& s, int x, int y) { return x >= s.rx0 && y >= s.ry0 && x < s.rx0 + s.rw && y < s.ry0 + s.rh; }
inline int sw_i(const Sway& s, int x, int y) { return (y - s.ry0) * s.rw + (x - s.rx0); }

// The mesh: every triangle (dest, source, owner, shade, depth) - the pose now, or at rest.
struct SwTri { float dx[3], dy[3], sx[3], sy[3]; int8_t oid; float shade, depth; };
inline int sway_mesh(const Sway& s, SwTri* tris, int maxt, bool rest) {
    int nt = 0;
    auto add = [&](const float* DX, const float* DY, const float* SX, const float* SY, int a, int b, int c, int8_t oid, float shade, float depth) {
        if (nt >= maxt) return;
        SwTri& T = tris[nt++];
        const int ix[3] = {a, b, c};
        for (int k = 0; k < 3; ++k) { T.dx[k] = DX[ix[k]]; T.dy[k] = DY[ix[k]]; T.sx[k] = SX[ix[k]]; T.sy[k] = SY[ix[k]]; }
        T.oid = oid; T.shade = shade; T.depth = depth;
    };
    if (s.kind == SK_PENDANT) {
        const int8_t* o = s.own;
        int ax = IMG_W, ay = IMG_H, bx = -1, by = -1;
        for (int y = 0; y < s.rh; ++y) for (int x = 0; x < s.rw; ++x) if (o[y * s.rw + x] >= 0) {
            ax = std::min(ax, x + s.rx0); bx = std::max(bx, x + s.rx0); ay = std::min(ay, y + s.ry0); by = std::max(by, y + s.ry0); }
        if (bx < 0) return 0;
        const float SX[4] = {(float)ax - 3, (float)bx + 4, (float)bx + 4, (float)ax - 3}, SY[4] = {(float)ay - 3, (float)ay - 3, (float)by + 4, (float)by + 4};
        const float th = rest ? 0.0f : s.th, c = cosf(th), sn = sinf(th);
        float DX[4], DY[4];
        for (int k = 0; k < 4; ++k) { const float qx = SX[k] - s.pinx, qy = SY[k] - s.piny; DX[k] = s.pinx + qx * c - qy * sn; DY[k] = s.piny + qx * sn + qy * c; }
        add(DX, DY, SX, SY, 0, 1, 2, -2, 1.0f, 0); add(DX, DY, SX, SY, 0, 2, 3, -2, 1.0f, 0);
        return nt;
    }
    if (s.kind == SK_HAIR) {
        for (int k = 0; k < s.nr; ++k) {
            const int n = s.nn[k];
            float t0x = s.rx[k][1] - s.rx[k][0], t0y = s.ry[k][1] - s.ry[k][0], l0 = sqrtf(t0x * t0x + t0y * t0y);
            if (l0 < 1e-6f) l0 = 1;
            t0x /= l0; t0y /= l0;
            float SLx[SN_MAX + 1], SLy[SN_MAX + 1], SRx[SN_MAX + 1], SRy[SN_MAX + 1], DLx[SN_MAX + 1], DLy[SN_MAX + 1], DRx[SN_MAX + 1], DRy[SN_MAX + 1];
            for (int pass = 0; pass < 2; ++pass) {
                const float* px = (pass == 0 || rest) ? s.rx[k] : s.x[k];
                const float* py = (pass == 0 || rest) ? s.ry[k] : s.y[k];
                float* Lx = pass ? DLx : SLx; float* Ly = pass ? DLy : SLy; float* Rx = pass ? DRx : SRx; float* Ry = pass ? DRy : SRy;
                float tx = 0, ty = 0;
                for (int i = 0; i <= n; ++i) {
                    if (i < n) {
                        const int a = i > 0 ? i - 1 : 0, b = i < n - 1 ? i + 1 : n - 1;
                        tx = px[b] - px[a]; ty = py[b] - py[a];
                        float l = sqrtf(tx * tx + ty * ty); if (l < 1e-6f) l = 1e-6f; tx /= l; ty /= l;
                    }
                    float nxv = -ty, nyv = tx;
                    if (i == 0) { nxv = -t0y; nyv = t0x; }                     // (the root edge is pinned whole)
                    float cx, cy;
                    const int j = i < n ? i : n - 1;
                    if (i < n) { cx = px[i]; cy = py[i]; } else { cx = px[n - 1] + tx * s.ext[k]; cy = py[n - 1] + ty * s.ext[k]; }
                    Lx[i] = cx + nxv * s.lw[k][j]; Ly[i] = cy + nyv * s.lw[k][j];
                    Rx[i] = cx - nxv * s.rwd[k][j]; Ry[i] = cy - nyv * s.rwd[k][j];
                }
            }
            for (int i = 0; i < n; ++i) {
                const float SX[4] = {SLx[i], SRx[i], SRx[i + 1], SLx[i + 1]}, SY[4] = {SLy[i], SRy[i], SRy[i + 1], SLy[i + 1]};
                const float DX[4] = {DLx[i], DRx[i], DRx[i + 1], DLx[i + 1]}, DY[4] = {DLy[i], DRy[i], DRy[i + 1], DLy[i + 1]};
                add(DX, DY, SX, SY, 0, 1, 2, (int8_t)k, 1.0f, (float)k); add(DX, DY, SX, SY, 0, 2, 3, (int8_t)k, 1.0f, (float)k);
            }
        }
        return nt;
    }
    for (int k = 0; k < s.nr - 1; ++k) {                // cloth panels: shade and depth from the fold
        const int n = s.nn[k];
        for (int j = 0; j < n - 1; ++j) {
            const float* Ax = rest ? s.rx[k] : s.x[k]; const float* Ay = rest ? s.ry[k] : s.y[k];
            const float* Bx = rest ? s.rx[k + 1] : s.x[k + 1]; const float* By = rest ? s.ry[k + 1] : s.y[k + 1];
            const float w = ((Bx[j] + Bx[j + 1] - Ax[j] - Ax[j + 1]) * s.fex + (By[j] + By[j + 1] - Ay[j] - Ay[j + 1]) * s.fey) * 0.5f;
            float cphi = w / s.Lm[k]; cphi = cphi < 0.05f ? 0.05f : (cphi > 1 ? 1 : cphi);
            const float sgn = (k % 2 == 0) ? 1.0f : -1.0f, light = 0.35f;
            const float phi = acosf(cphi), phi0 = acosf(std::min(1.0f, s.cos0[k]));
            const float shade = std::max(0.25f, cosf(sgn * phi - light)) / std::max(0.25f, cosf(sgn * phi0 - light));
            const float depth = -sgn * sinf(phi);
            const float SX[4] = {s.rx[k][j], s.rx[k + 1][j], s.rx[k + 1][j + 1], s.rx[k][j + 1]}, SY[4] = {s.ry[k][j], s.ry[k + 1][j], s.ry[k + 1][j + 1], s.ry[k][j + 1]};
            const float DX[4] = {Ax[j], Bx[j], Bx[j + 1], Ax[j + 1]}, DY[4] = {Ay[j], By[j], By[j + 1], Ay[j + 1]};
            add(DX, DY, SX, SY, 0, 1, 2, (int8_t)k, shade, depth); add(DX, DY, SX, SY, 0, 2, 3, (int8_t)k, shade, depth);
        }
    }
    return nt;
}
constexpr int SW_TRIS = 2 * SR_MAX * (SN_MAX + 1);

// ---- prepare ------------------------------------------------------------------------------------------
// grow a region bitmap (one byte per pixel) by one 4-neighbour step
inline void sw_grow(uint8_t* m, int w, int h, uint8_t* tmp) {
    memcpy(tmp, m, (size_t)w * h);
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        if (tmp[y * w + x]) continue;
        if ((x > 0 && tmp[y * w + x - 1]) || (x < w - 1 && tmp[y * w + x + 1]) || (y > 0 && tmp[(y - 1) * w + x]) || (y < h - 1 && tmp[(y + 1) * w + x])) m[y * w + x] = 1;
    }
}
// nearest marked pixel within the region (two-pass propagation); -1 where none
inline void sw_nearest(const uint8_t* seed, int w, int h, int16_t* nx, int16_t* ny) {
    for (int i = 0; i < w * h; ++i) { if (seed[i]) { nx[i] = (int16_t)(i % w); ny[i] = (int16_t)(i / w); } else nx[i] = -1; }
    auto d2 = [&](int x, int y, int j) { if (nx[j] < 0) return 0x7fffffff; const int dx = nx[j] - x, dy = ny[j] - y; return dx * dx + dy * dy; };
    auto rel = [&](int x, int y, int j) { const int i = y * w + x; if (d2(x, y, j) < d2(x, y, i)) { nx[i] = nx[j]; ny[i] = ny[j]; } };
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (y > 0) { const int r = (y - 1) * w; if (x > 0) rel(x, y, r + x - 1); rel(x, y, r + x); if (x < w - 1) rel(x, y, r + x + 1); }
            if (x > 0) rel(x, y, y * w + x - 1);
        }
        for (int x = w - 2; x >= 0; --x) rel(x, y, y * w + x + 1);
    }
    for (int y = h - 1; y >= 0; --y) {
        for (int x = w - 1; x >= 0; --x) {
            if (y < h - 1) { const int r = (y + 1) * w; if (x < w - 1) rel(x, y, r + x + 1); rel(x, y, r + x); if (x > 0) rel(x, y, r + x - 1); }
            if (x < w - 1) rel(x, y, y * w + x + 1);
        }
        for (int x = 1; x < w; ++x) rel(x, y, y * w + x - 1);
    }
}

// The same, but only `iters` px deep (8-neighbour steps, as mg_depth._nearest - the Studio's lab gets the
// same pixels); d2: squared distance (or a huge number where none).
inline void sw_nearest_it(const uint8_t* seed, int w, int h, int iters, int16_t* nx, int16_t* ny, int32_t* d2, int16_t* tx, int16_t* ty) {
    const int N = w * h;
    const int32_t BIG = 1 << 30;
    for (int i = 0; i < N; ++i) {
        if (seed[i]) { nx[i] = (int16_t)(i % w); ny[i] = (int16_t)(i / w); d2[i] = 0; } else { nx[i] = ny[i] = -1; d2[i] = BIG; }
    }
    static const int8_t DIRS[8][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    for (int it = 0; it < iters; ++it)
        for (int q = 0; q < 8; ++q) {
            const int dy = DIRS[q][0], dx = DIRS[q][1];
            memcpy(tx, nx, (size_t)N * 2); memcpy(ty, ny, (size_t)N * 2);
            for (int y = 0; y < h; ++y) {
                const int sy = y - dy;
                if (sy < 0 || sy >= h) continue;
                for (int x = 0; x < w; ++x) {
                    const int sx = x - dx;
                    if (sx < 0 || sx >= w) continue;
                    const int j = sy * w + sx;
                    if (ty[j] < 0) continue;
                    const int32_t cd = (int32_t)(ty[j] - y) * (ty[j] - y) + (int32_t)(tx[j] - x) * (tx[j] - x);
                    const int i = y * w + x;
                    if (cd < d2[i]) { nx[i] = tx[j]; ny[i] = ty[j]; d2[i] = cd; }
                }
            }
        }
}

inline void sw_prep_hair(Sway& s, std::vector<int>& px_, std::vector<int>& py_) {
    s.nr = 0;
    for (int g = 0; g < s.ng && s.nr < SR_MAX; ++g) {
        if (s.gn[g] < 2) continue;
        float L = 0;
        for (int i = 1; i < s.gn[g]; ++i) L += sqrtf((s.gx[g][i] - s.gx[g][i - 1]) * (s.gx[g][i] - s.gx[g][i - 1]) + (s.gy[g][i] - s.gy[g][i - 1]) * (s.gy[g][i] - s.gy[g][i - 1]));
        const int k = s.nr++;
        s.nn[k] = sw_resample(s.gx[g], s.gy[g], s.gn[g], std::max(4.0f, L / 9.0f), SN_MAX, s.rx[k], s.ry[k]);
    }
    const int np = (int)px_.size();
    std::vector<int8_t> owner(np, -1);
    std::vector<float> S(np), D(np);
    for (int p = 0; p < np; ++p) {
        float best = 1e18f;
        for (int k = 0; k < s.nr; ++k) {
            float d, Sk, Dk;
            sw_bind((float)px_[p], (float)py_[p], s.rx[k], s.ry[k], s.nn[k], d, Sk, Dk);
            if (d < best) { best = d; owner[p] = (int8_t)k; S[p] = Sk; D[p] = Dk; }
        }
        if (best > SW_BIND_MAX) owner[p] = -1;            // (too far from every lock: it stays)
    }
    for (int p = 0; p < np; ++p) {                     // past the root: it moves
        if (owner[p] < 0 || S[p] <= 0.5f) continue;
        if (sw_in(s, px_[p], py_[p])) s.own[sw_i(s, px_[p], py_[p])] = owner[p];
    }
    std::vector<float> pos, neg;
    for (int k = 0; k < s.nr; ++k) {
        const int n = s.nn[k];
        float cum[SN_MAX] = {0}, segmax = 0;
        for (int i = 1; i < n; ++i) {
            const float l = sqrtf((s.rx[k][i] - s.rx[k][i - 1]) * (s.rx[k][i] - s.rx[k][i - 1]) + (s.ry[k][i] - s.ry[k][i - 1]) * (s.ry[k][i] - s.ry[k][i - 1]));
            cum[i] = cum[i - 1] + l; segmax = std::max(segmax, l);
        }
        float smax = -1e9f;
        for (int i = 0; i < n; ++i) {
            pos.clear(); neg.clear();
            for (int p = 0; p < np; ++p) {
                if (owner[p] != k || S[p] <= 0.5f) continue;
                if (i == 0) smax = std::max(smax, S[p]);
                if (fabsf(S[p] - cum[i]) <= std::max(segmax, 3.0f)) { if (D[p] > 0) pos.push_back(D[p]); else if (D[p] < 0) neg.push_back(-D[p]); }
            }
            auto p95 = [](std::vector<float>& v) {     // (95th percentile: a stray bit of something else doesn't widen it)
                if (v.empty()) return 2.0f;
                const float q = 0.95f * (v.size() - 1);
                const size_t a = (size_t)q;
                std::nth_element(v.begin(), v.begin() + a, v.end());
                float va = v[a], vb = va;
                if (a + 1 < v.size()) { std::nth_element(v.begin(), v.begin() + a + 1, v.end()); vb = v[a + 1]; }
                return std::max(2.0f, va + (vb - va) * (q - a) + 2.5f);
            };
            s.lw[k][i] = p95(pos); s.rwd[k][i] = p95(neg);
        }
        s.ext[k] = smax > -1e8f ? std::min(SW_EXT_MAX, std::max(0.0f, smax - cum[n - 1] + 2.5f)) : 2.5f;
    }
}

inline void sw_prep_cloth(Sway& s, const uint16_t* photo, bool psw, std::vector<int>& px_, std::vector<int>& py_) {
    const float ax = s.gx[0][0], ay = s.gy[0][0], bx = s.gx[0][1], by = s.gy[0][1];
    float ex = bx - ax, ey = by - ay, l = sqrtf(ex * ex + ey * ey);
    if (l < 1e-6f) l = 1;
    ex /= l; ey /= l;
    float nxv = -ey, nyv = ex;
    if (nyv < 0) { nxv = -nxv; nyv = -nyv; }
    const int np = (int)px_.size();
    std::vector<float> S(np), T(np);
    float s0 = 1e9f, s1 = -1e9f, tm = 0;
    for (int p = 0; p < np; ++p) {
        S[p] = (px_[p] - ax) * ex + (py_[p] - ay) * ey; T[p] = (px_[p] - ax) * nxv + (py_[p] - ay) * nyv;
        if (T[p] > 0) { s0 = std::min(s0, S[p]); s1 = std::max(s1, S[p]); tm = std::max(tm, T[p]); }
    }
    s0 -= 1.5f; s1 += 1.5f; tm += 1.5f;
    std::vector<float> cuts;
    cuts.push_back(s0);
    if (s.ng > 1) {                                     // folds you marked
        for (int g = 1; g < s.ng && (int)cuts.size() < SR_MAX - 1; ++g) {
            const float c = (s.gx[g][0] - ax) * ex + (s.gy[g][0] - ay) * ey;
            if (c > s0 + 3 && c < s1 - 3) cuts.push_back(c);
        }
        std::sort(cuts.begin(), cuts.end());
    } else {                                            // folds from the shading: valleys of brightness across
        const int nb = (int)s1 - (int)s0 + 1;
        std::vector<float> prof(nb, 0), cnt(nb, 0);
        for (int p = 0; p < np; ++p) {
            if (T[p] <= 0 || T[p] <= tm / 3) continue;
            for (int b = 0; b < nb; ++b) {
                if (fabsf(S[p] - ((int)s0 + b)) < 1.0f) {
                    const uint16_t v0 = photo[py_[p] * IMG_W + px_[p]], v = psw ? (uint16_t)((v0 >> 8) | (v0 << 8)) : v0;
                    float r, g, bb; sw_unpack(v, r, g, bb);
                    prof[b] += (r + g + bb) / 3; cnt[b] += 1;
                }
            }
        }
        float mean = 0; int nm = 0;
        for (int b = 0; b < nb; ++b) if (cnt[b] > 0) { prof[b] /= cnt[b]; mean += prof[b]; ++nm; }
        mean = nm ? mean / nm : 0;
        for (int b = 0; b < nb; ++b) if (cnt[b] <= 0) prof[b] = mean;
        std::vector<float> sm(nb, 0);
        for (int b = 0; b < nb; ++b) {                  // (5-tap box, zero-padded like numpy's "same")
            float a = 0; for (int q = -2; q <= 2; ++q) if (b + q >= 0 && b + q < nb) a += prof[b + q];
            sm[b] = a / 5;
        }
        std::vector<int> cand;
        for (int i = 3; i < nb - 3; ++i) {
            float mn = sm[i]; for (int q = -3; q <= 3; ++q) mn = std::min(mn, sm[i + q]);
            if (sm[i] < sm[i - 3] - 2 && sm[i] < sm[i + 3] - 2 && sm[i] == mn) cand.push_back(i);
        }
        std::stable_sort(cand.begin(), cand.end(), [&](int a, int b) { return sm[a] < sm[b]; });
        for (int i : cand) {
            const float b = (float)((int)s0 + i);
            bool ok = s1 - b >= 9;
            for (float c : cuts) ok = ok && fabsf(b - c) >= 9;
            if (ok) cuts.push_back(b);
        }
        std::sort(cuts.begin(), cuts.end());
    }
    cuts.push_back(s1);
    while ((int)cuts.size() > SR_MAX) {                 // (at most 8 panels: merge the narrowest)
        int k = 1; float best = 1e9f;
        for (int i = 1; i < (int)cuts.size() - 1; ++i) if (cuts[i + 1] - cuts[i] < best) { best = cuts[i + 1] - cuts[i]; k = i; }
        cuts.erase(cuts.begin() + k);
    }
    if ((int)cuts.size() < 4 && s.ng <= 1) {            // no clear folds (and none marked): even panels ~12 px
        const int n = std::max(3, (int)((s1 - s0) / 12));
        cuts.clear();
        for (int i = 0; i <= n && i < SR_MAX; ++i) cuts.push_back(s0 + (s1 - s0) * i / n);
    }
    const int nrw = 6;
    s.nr = (int)cuts.size();
    for (int k = 0; k < s.nr; ++k) {
        s.cuts[k] = cuts[k];
        s.nn[k] = nrw + 1;
        for (int j = 0; j <= nrw; ++j) {
            const float tt = tm * j / nrw;
            s.rx[k][j] = ax + cuts[k] * ex + tt * nxv; s.ry[k][j] = ay + cuts[k] * ey + tt * nyv;
        }
    }
    for (int k = 0; k < s.nr - 1; ++k) { s.W0[k] = cuts[k + 1] - cuts[k]; s.Lm[k] = s.W0[k] * s.prm[SP_PLEAT]; s.cos0[k] = s.W0[k] / s.Lm[k]; }
    s.fax = ax; s.fay = ay; s.fex = ex; s.fey = ey; s.fnx = nxv; s.fny = nyv; s.tmax = tm;
    for (int p = 0; p < np; ++p) {
        if (T[p] <= 0 || !sw_in(s, px_[p], py_[p])) continue;
        for (int k = 0; k < s.nr - 1; ++k)
            if (S[p] >= cuts[k] && (S[p] < cuts[k + 1] || (k == s.nr - 2 && S[p] <= cuts[k + 1]))) { s.own[sw_i(s, px_[p], py_[p])] = (int8_t)k; break; }
    }
    for (int j = 1; j <= nrw; ++j) {                    // round the body: each row's axis, radius, each fold's angle
        float a = s.rx[0][j] * ex + s.ry[0][j] * ey, b = s.rx[s.nr - 1][j] * ex + s.ry[s.nr - 1][j] * ey;
        s.cc[j] = (a + b) * 0.5f;
        s.cR[j] = (b - a) * 0.5f / sinf(1.2217305f);
        for (int k = 0; k < s.nr; ++k) {
            float v = ((s.rx[k][j] * ex + s.ry[k][j] * ey) - s.cc[j]) / s.cR[j];
            v = v < -0.999f ? -0.999f : (v > 0.999f ? 0.999f : v);
            s.cpsi[j][k] = asinf(v);
        }
    }
}

inline void sw_prep_pendant(Sway& s, std::vector<int>& px_, std::vector<int>& py_) {
    s.pinx = s.gx[0][0]; s.piny = s.gy[0][0];
    const int np = (int)px_.size();
    double sx = 0, sy = 0, sI = 0;
    float tl = 0;
    for (int p = 0; p < np; ++p) {
        const float qx = px_[p] - s.pinx, qy = py_[p] - s.piny;
        sx += qx; sy += qy; sI += qx * qx + qy * qy; tl = std::max(tl, sqrtf(qx * qx + qy * qy));
        if (sw_in(s, px_[p], py_[p])) s.own[sw_i(s, px_[p], py_[p])] = 0;
    }
    s.mass = (float)np; s.rcx = (float)(sx / np); s.rcy = (float)(sy / np);
    s.I = (float)sI + s.mass; s.th = s.thp = 0;
    s.tau0 = -s.mass * SW_G * s.rcx;                  // balanced in the photo's pose
    s.tlen = std::max(tl, 1.0f);
    // outline points that collide: the farthest edge pixel in each of 16 directions
    s.no = 0;
    for (int a = 0; a < 16; ++a) {
        const float ang = -3.14159265f + 6.2831853f * a / 16;
        float best = -1, bx = 0, by = 0;
        for (int p = 0; p < np; ++p) {
            const int x = px_[p], y = py_[p];
            const bool edge = !(x > 0 && mask_get(s.mask, x - 1, y)) || !(x < IMG_W - 1 && mask_get(s.mask, x + 1, y)) ||
                              !(y > 0 && mask_get(s.mask, x, y - 1)) || !(y < IMG_H - 1 && mask_get(s.mask, x, y + 1));
            if (!edge) continue;
            float da = atan2f(y - s.piny, x - s.pinx) - ang;
            while (da > 3.14159265f) da -= 6.2831853f;
            while (da < -3.14159265f) da += 6.2831853f;
            if (fabsf(da) >= 0.25f) continue;
            const float d = sqrtf((x - s.pinx) * (x - s.pinx) + (y - s.piny) * (y - s.piny));
            if (d > best) { best = d; bx = (float)x; by = (float)y; }
        }
        if (best >= 0) { s.ox[s.no] = bx; s.oy[s.no] = by; ++s.no; }
    }
}

// The fill behind it, where the Studio didn't send one: each hole pixel takes the colour mirrored across
// the nearest pixel it may copy from (the texture carries on). On the body: only from the body right
// around the hole; off it: only from off the body.
inline void sw_fill(Sway& s, const uint8_t* body, const uint8_t* fb, int16_t* nx, int16_t* ny, uint8_t* tmp) {
    const int w = s.rw, h = s.rh, N = w * h;
    std::vector<uint8_t> src(N), want(N);
    for (int pass = 0; pass < 2; ++pass) {              // 0: off the body, 1: on it
        int nw = 0;
        for (int i = 0; i < N; ++i) {
            const int X = s.rx0 + i % w, Y = s.ry0 + i / w;
            const bool on = body && bget_(body, Y * IMG_W + X);
            want[i] = s.hole[i] && (on == (pass == 1)) && !(s.pdef_ok && s.pdef_has && bget_(s.pdef_has, Y * IMG_W + X));
            nw += want[i];
        }
        if (!nw) continue;
        for (int reach = 8; ; reach = reach < 24 ? reach + 8 : 1000) {
            uint8_t* nearh = tmp;                       // within `reach` of the hole (on the body only)
            for (int i = 0; i < N; ++i) nearh[i] = s.hole[i];
            if (pass == 1 && reach < 1000) for (int r = 0; r < reach; ++r) sw_grow(nearh, w, h, nearh + N);
            int ns = 0;
            for (int i = 0; i < N; ++i) {
                const int X = s.rx0 + i % w, Y = s.ry0 + i / w;
                const bool on = body && bget_(body, Y * IMG_W + X);
                src[i] = !s.hole[i] && !fb[i] && (on == (pass == 1)) && (pass == 0 || reach >= 1000 || nearh[i]);
                ns += src[i];
            }
            if (ns >= 60 + 2 * (int)sqrtf((float)nw) || reach >= 1000) {
                if (!ns) break;
                sw_nearest(src.data(), w, h, nx, ny);
                for (int i = 0; i < N; ++i) {
                    if (!want[i] || nx[i] < 0) continue;
                    const int x = i % w, y = i / w, mx = 2 * nx[i] - x, my = 2 * ny[i] - y;
                    const int j = (mx >= 0 && my >= 0 && mx < w && my < h && src[my * w + mx]) ? my * w + mx : ny[i] * w + nx[i];
                    s.plate[i] = s.photo[j];
                }
                break;
            }
        }
    }
}

// Safety: the bikini area of a skirt's hip outline (mg_sway.bikini_poly) - round the point between the
// legs. Where the fill behind a swinging skirt would show skin there, a white bikini bottom with pink
// polka dots shows instead.
constexpr int SW_BIKINI_N = 12;
inline bool sw_bikini_poly(const Sway& s, float* qx, float* qy) {   // (mg_sway.bikini_poly; SW_BIKINI_N points)
    if (s.narm != 9) return false;
    const float h0x = s.armx[1], h0y = s.army[1], l0x = s.armx[2], l0y = s.army[2], lix = s.armx[3];
    const float cx = s.armx[4], cy = s.army[4], rix = s.armx[5], r0x = s.armx[6], r0y = s.army[6], h1x = s.armx[7], h1y = s.army[7];
    const bool bk = s.arm_back;                         // (seen from behind: her seat - higher, steeper, wider panel)
    const float hy = (h0y + h1y) / 2, ty = hy + (bk ? -0.05f : 0.1f) * std::max(2.0f, cy - hy);
    auto side = [&](float ax, float ay, float bx, float by) { return ax + (bx - ax) * (by != ay ? (ty - ay) / (by - ay) : 0.0f); };
    const float xl = side(h0x, h0y, l0x, l0y) + 1.0f, xr = side(h1x, h1y, r0x, r0y) - 1.0f;
    const float gap = std::max(3.0f, rix - lix), by = cy + 0.5f;
    const float wb = bk ? std::min(gap * 0.6f + 2.0f, 6.0f) : std::min(gap * 0.3f + 1.0f, 3.0f);
    for (int k = 0; k < 6; ++k) {
        const float t = k / 5.0f, y = ty + (by - ty) * t, f = powf(1 - t, bk ? 0.75f : 1.6f);
        qx[k] = cx - wb - (cx - wb - xl) * f; qy[k] = y;
        qx[SW_BIKINI_N - 1 - k] = cx + wb + (xr - cx - wb) * f; qy[SW_BIKINI_N - 1 - k] = y;
    }
    return true;
}
inline bool sw_in_poly(const float* qx, const float* qy, int n, float x, float y) {
    bool in = false;
    for (int a = 0, b = n - 1; a < n; b = a++)
        if ((qy[a] > y) != (qy[b] > y) && x < qx[a] + (y - qy[a]) * (qx[b] - qx[a]) / (qy[b] - qy[a])) in = !in;
    return in;
}
inline uint16_t sw_bikini_color(int x, int y) {
    const float gx = (x % 6) - 2.5f, gy = (y % 6) - 2.5f;
    return gx * gx + gy * gy <= 2.0f ? sw_pack(255, 150, 190) : sw_pack(238, 238, 238);
}

// Edge alpha and carried colour (mg_sway.matte): the outline pixel and two rings past it split
// c = a·H + (1 − a)·B, H = nearest solid pixel of it, B = what's behind (the plate).
inline void sw_matte(Sway& s, const uint8_t* moving, int16_t* nx, int16_t* ny, uint8_t* tmp, int32_t* d2, int16_t* tx, int16_t* ty) {
    const int w = s.rw, h = s.rh, N = w * h;
    uint8_t* r1 = tmp; uint8_t* r2 = tmp + N; uint8_t* t2 = tmp + 2 * N; uint8_t* solid = tmp + 3 * N;
    memcpy(r1, moving, N); sw_grow(r1, w, h, t2);
    memcpy(r2, r1, N); sw_grow(r2, w, h, t2);
    auto mv = [&](int x, int y) { x = x < 0 ? 0 : (x >= w ? w - 1 : x); y = y < 0 ? 0 : (y >= h ? h - 1 : y); return moving[y * w + x] != 0; };
    int nsolid = 0;
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        const int i = y * w + x;
        const bool edge = moving[i] && !(mv(x - 1, y) && mv(x + 1, y) && mv(x, y - 1) && mv(x, y + 1));
        solid[i] = moving[i] && !edge;
        t2[i] = edge;
        nsolid += solid[i];
    }
    if (!nsolid) memcpy(solid, moving, N);
    sw_nearest_it(solid, w, h, 5, nx, ny, d2, tx, ty);
    for (int i = 0; i < N; ++i) {
        const bool edge = t2[i];
        const float soft = moving[i] ? (edge ? 0.82f : 1.0f) : (r1[i] ? 0.47f : (r2[i] ? 0.18f : 0.0f));
        const bool band = soft > 0 && soft < 1 && ny[i] >= 0;
        float a;
        s.texd[i] = s.photo[i];
        if (solid[i]) a = 1.0f;
        else if (band) {
            const int j = ny[i] * w + nx[i];
            float hr, hg, hb, cr, cg, cb, br, bg, bb;
            sw_unpack(s.photo[j], hr, hg, hb); sw_unpack(s.photo[i], cr, cg, cb); sw_unpack(s.plate[i], br, bg, bb);
            const float dr = hr - br, dg = hg - bg, db = hb - bb, dd = dr * dr + dg * dg + db * db;
            const float hi = edge ? 1.0f : (r1[i] ? 0.8f : 0.45f), lo = edge ? 0.4f : 0.0f;
            if (dd < 300) a = soft;
            else { a = ((cr - br) * dr + (cg - bg) * dg + (cb - bb) * db) / std::max(dd, 100.0f); a = a < lo ? lo : (a > hi ? hi : a); }
            s.texd[i] = s.photo[j];
        } else a = soft;
        s.alpha[i] = (uint8_t)lroundf(std::max(0.0f, std::min(1.0f, a)) * 255.0f);
    }
}

inline void sw_cover(int8_t* cov, int w, int h, int x0, int y0, const SwTri& T) {   // (rest mesh: who carries each pixel)
    const double X[3] = {T.sx[0], T.sx[1], T.sx[2]}, Y[3] = {T.sy[0], T.sy[1], T.sy[2]};
    const double den = (Y[1] - Y[2]) * (X[0] - X[2]) + (X[2] - X[1]) * (Y[0] - Y[2]);
    if (fabs(den) < 1e-9) return;
    const int bx0 = std::max(0, (int)floor(std::min(X[0], std::min(X[1], X[2])))), bx1 = std::min(IMG_W - 1, (int)ceil(std::max(X[0], std::max(X[1], X[2]))));
    const int by0 = std::max(0, (int)floor(std::min(Y[0], std::min(Y[1], Y[2])))), by1 = std::min(IMG_H - 1, (int)ceil(std::max(Y[0], std::max(Y[1], Y[2]))));
    for (int y = by0; y <= by1; ++y) for (int x = bx0; x <= bx1; ++x) {
        if (x < x0 || y < y0 || x >= x0 + w || y >= y0 + h) continue;
        const double l0 = ((Y[1] - Y[2]) * (x - X[2]) + (X[2] - X[1]) * (y - Y[2])) / den;
        const double l1 = ((Y[2] - Y[0]) * (x - X[2]) + (X[0] - X[2]) * (y - Y[2])) / den;
        if (l0 >= -1e-7 && l1 >= -1e-7 && 1 - l0 - l1 >= -1e-7 && cov[(y - y0) * w + (x - x0)] < 0) cov[(y - y0) * w + (x - x0)] = T.oid;
    }
}

// Build everything to move and draw it. body: the figures it rests on (MASK_BYTES bits) or null; sdf
// from sway_body_sdf (or null).
// forbid: other things that move (MASK_BYTES bits, or null) - never copied in behind it.
inline bool sway_prepare(Sway& s, const uint16_t* photo, bool psw, const uint8_t* body, const int16_t* sdf, const uint8_t* forbid,
                         void* (*alloc)(size_t)) {
    sway_free_runtime(s);
    if (!sway_ok(s) || !photo) return false;
    std::vector<int> px_, py_;
    int mx0 = IMG_W, my0 = IMG_H, mx1 = -1, my1 = -1;
    for (int y = 0; y < IMG_H; ++y) for (int x = 0; x < IMG_W; ++x) if (mask_get(s.mask, x, y)) {
        px_.push_back(x); py_.push_back(y); mx0 = std::min(mx0, x); mx1 = std::max(mx1, x); my0 = std::min(my0, y); my1 = std::max(my1, y); }
    // buffers: the mask's box and what the plate / rings need around it
    s.rx0 = std::max(0, mx0 - 8); s.ry0 = std::max(0, my0 - 8);
    s.rw = std::min(IMG_W, mx1 + 9) - s.rx0; s.rh = std::min(IMG_H, my1 + 9) - s.ry0;
    const int N = s.rw * s.rh;
    s.own = (int8_t*)alloc(N); s.hole = (uint8_t*)alloc(N); s.alpha = (uint8_t*)alloc(N);
    s.plate = (uint16_t*)alloc((size_t)N * 2); s.photo = (uint16_t*)alloc((size_t)N * 2); s.texd = (uint16_t*)alloc((size_t)N * 2);
    s.c8 = (float*)alloc((size_t)N * 12); s.t8 = (float*)alloc((size_t)N * 12);
    int16_t* nx = (int16_t*)alloc((size_t)N * 2); int16_t* ny = (int16_t*)alloc((size_t)N * 2);
    uint8_t* tmp = (uint8_t*)alloc((size_t)N * 4);
    uint8_t* moving = (uint8_t*)alloc(N); uint8_t* fb = (uint8_t*)alloc(N);
    int8_t* cov = (int8_t*)alloc(N);
    int32_t* d2 = (int32_t*)alloc((size_t)N * 4);
    int16_t* tx = (int16_t*)alloc((size_t)N * 2); int16_t* ty = (int16_t*)alloc((size_t)N * 2);
    SwTri* tris = (SwTri*)alloc(sizeof(SwTri) * SW_TRIS);
    auto cleanup = [&]() { void* b[] = {nx, ny, tmp, moving, fb, cov, tris, d2, tx, ty}; for (void* p : b) if (p) free(p); };
    if (!s.own || !s.hole || !s.alpha || !s.plate || !s.photo || !s.texd || !s.c8 || !s.t8 || !nx || !ny || !tmp || !moving || !fb || !cov || !tris || !d2 || !tx || !ty) {
        cleanup(); sway_free_runtime(s); return false;
    }
    memset(s.own, -1, N); memset(s.hole, 0, N);
    for (int i = 0; i < N; ++i) {
        const uint16_t v = photo[(s.ry0 + i / s.rw) * IMG_W + s.rx0 + i % s.rw];
        s.photo[i] = psw ? (uint16_t)((v >> 8) | (v << 8)) : v;
        s.plate[i] = s.photo[i];
    }
    if (s.kind == SK_HAIR) sw_prep_hair(s, px_, py_);
    else if (s.kind == SK_CLOTH) sw_prep_cloth(s, photo, psw, px_, py_);
    else sw_prep_pendant(s, px_, py_);
    for (int i = 0; i < N; ++i) moving[i] = s.own[i] >= 0;
    if (s.kind != SK_PENDANT) {                         // only what its rest mesh covers can move (or be uncovered)
        memset(cov, -1, N);
        const int nt = sway_mesh(s, tris, SW_TRIS, true);
        for (int k = 0; k < nt; ++k) sw_cover(cov, s.rw, s.rh, s.rx0, s.ry0, tris[k]);
        for (int i = 0; i < N; ++i) { if (cov[i] < 0) moving[i] = 0; s.own[i] = moving[i] ? cov[i] : -1; }
        memcpy(s.hole, moving, N); sw_grow(s.hole, s.rw, s.rh, tmp); sw_grow(s.hole, s.rw, s.rh, tmp);
        for (int i = 0; i < N; ++i) if (cov[i] < 0) s.hole[i] = 0;
    } else {
        memcpy(s.hole, moving, N); for (int r = 0; r < 3; ++r) sw_grow(s.hole, s.rw, s.rh, tmp);
    }
    // what may be copied in behind it: not the thing itself (nor its two rings)
    memset(fb, 0, N);
    for (int i = 0; i < N; ++i) { const int X = s.rx0 + i % s.rw, Y = s.ry0 + i / s.rw; fb[i] = mask_get(s.mask, X, Y); }
    sw_grow(fb, s.rw, s.rh, tmp); sw_grow(fb, s.rw, s.rh, tmp);
    if (forbid) for (int i = 0; i < N; ++i) if (mask_get(forbid, s.rx0 + i % s.rw, s.ry0 + i / s.rw)) fb[i] = 1;
    for (int i = 0; i < N; ++i) {                       // the Studio's fill where it has one
        const int X = s.rx0 + i % s.rw, Y = s.ry0 + i / s.rw;
        if (s.pdef_ok && s.pdef && s.pdef_has && bget_(s.pdef_has, Y * IMG_W + X)) s.plate[i] = s.pdef[Y * IMG_W + X];
    }
    sw_fill(s, body, fb, nx, ny, tmp);
    if (s.kind == SK_CLOTH) {                           // (its own fill: never skin in the bikini area)
        float qx[SW_BIKINI_N], qy[SW_BIKINI_N];
        if (sw_bikini_poly(s, qx, qy))
            for (int i = 0; i < N; ++i) {
                const int X = s.rx0 + i % s.rw, Y = s.ry0 + i / s.rw;
                if (!s.hole[i] || (s.pdef_ok && s.pdef_has && bget_(s.pdef_has, Y * IMG_W + X))) continue;
                if (sw_in_poly(qx, qy, SW_BIKINI_N, X + 0.5f, Y + 0.5f)) s.plate[i] = sw_bikini_color(X, Y);
            }
    }
    sw_matte(s, moving, nx, ny, tmp, d2, tx, ty);
    if (s.kind != SK_PENDANT) {                         // the rings past it belong to its nearest lock / panel
        uint8_t* seed = tmp;
        for (int i = 0; i < N; ++i) seed[i] = s.own[i] >= 0;
        sw_nearest_it(seed, s.rw, s.rh, 4, nx, ny, d2, tx, ty);
        for (int i = 0; i < N; ++i) {
            if (!(s.alpha[i] > 0 && s.own[i] < 0 && ny[i] >= 0 && s.hole[i])) continue;
            const int8_t o = s.own[ny[i] * s.rw + nx[i]];
            if (cov[i] == o) s.own[i] = o;
        }
        for (int i = 0; i < N; ++i) {                   // every uncovered pixel is carried by something
            if (s.own[i] < 0) s.hole[i] = 0;
            if (!s.hole[i]) s.alpha[i] = 0;
        }
    }
    // physics state
    for (int k = 0; k < s.nr; ++k) {
        const int n = s.nn[k];
        float cum = 0;
        for (int i = 0; i < n; ++i) {
            s.x[k][i] = s.xp[k][i] = s.rx[k][i]; s.y[k][i] = s.yp[k][i] = s.ry[k][i];
            s.len[k][i] = i ? sqrtf((s.rx[k][i] - s.rx[k][i - 1]) * (s.rx[k][i] - s.rx[k][i - 1]) + (s.ry[k][i] - s.ry[k][i - 1]) * (s.ry[k][i] - s.ry[k][i - 1])) : 0;
            cum += s.len[k][i];
            s.kn[k][i] = i ? SW_G / std::max(cum, 4.0f) : SW_G / 1e9f;
            s.rsdf[k][i] = sdf ? sdf_at(sdf, s.rx[k][i], s.ry[k][i]) : 1e9f;
            s.onb[k][i] = sdf && s.rsdf[k][i] <= 1.0f;
            const float fr = 2.0f * cum * sinf(SW_FREE / 2) + 1e-3f;
            if (s.kind == SK_CLOTH) s.cap[k][i] = (s.tmax * i / (n - 1)) * 2.0f * sinf(SW_CLOTH_FREE / 2) + 1e-3f;
            else {
                const float onb = std::max(2.0f, SW_ON_BODY * (s.lw[k][i] + s.rwd[k][i]) * 0.5f);
                s.cap[k][i] = s.onb[k][i] ? std::min(onb, fr) : fr;
            }
        }
        if (s.kind == SK_CLOTH) {                       // (cloth: kn from each row's distance down)
            for (int i = 0; i < n; ++i) s.kn[k][i] = SW_G / std::max(s.tmax * i / (n - 1), 4.0f);
        }
    }
    for (int i = 0; i < s.no; ++i) s.ors[i] = sdf ? sdf_at(sdf, s.ox[i], s.oy[i]) : 1e9f;
    // what it can touch on screen: its region plus how far it can swing
    float reach = 0;
    if (s.kind == SK_PENDANT) {                         // (its box's farthest corner, swung round the pin)
        for (int i = 0; i < N; ++i) if (s.own[i] >= 0) {
            const float dx = s.rx0 + i % s.rw - s.pinx, dy = s.ry0 + i / s.rw - s.piny;
            reach = std::max(reach, sqrtf((fabsf(dx) + 4) * (fabsf(dx) + 4) + (fabsf(dy) + 4) * (fabsf(dy) + 4)));
        }
        reach += 2;
    } else for (int k = 0; k < s.nr; ++k) for (int i = 0; i < s.nn[k]; ++i)
        reach = std::max(reach, s.cap[k][i] + (s.kind == SK_HAIR ? std::max(s.lw[k][i], s.rwd[k][i]) + s.ext[k] : 6.0f));
    if (s.kind == SK_PENDANT) { s.x0 = (int)(s.pinx - reach); s.y0 = (int)(s.piny - reach); s.x1 = (int)(s.pinx + reach) + 1; s.y1 = (int)(s.piny + reach) + 1; }
    else { s.x0 = s.rx0 - (int)reach; s.y0 = s.ry0 - (int)reach; s.x1 = s.rx0 + s.rw + (int)reach; s.y1 = s.ry0 + s.rh + (int)reach; }
    s.x0 = std::max(0, s.x0); s.y0 = std::max(0, s.y0); s.x1 = std::min(IMG_W, s.x1); s.y1 = std::min(IMG_H, s.y1);
    for (int i = 0; i < N; ++i) {                      // 8-bit colour of the photo and the matted edge, for drawing
        sw_unpack(s.photo[i], s.c8[3 * i], s.c8[3 * i + 1], s.c8[3 * i + 2]);
        sw_unpack(s.texd[i], s.t8[3 * i], s.t8[3 * i + 1], s.t8[3 * i + 2]);
    }
    s.solid = (uint8_t*)alloc((size_t)N);              // (drawing: one test a corner instead of three reads)
    if (s.solid) for (int i = 0; i < N; ++i) s.solid[i] = (uint8_t)(s.alpha[i] == 255 && s.photo[i] == s.texd[i]);
    if (s.kind == SK_CLOTH && s.narm == 9) {             // her hips: each row's edges (even-odd scanline of the outline)
        s.hipl = (int16_t*)alloc(IMG_H * 2); s.hipr = (int16_t*)alloc(IMG_H * 2);
        if (s.hipl && s.hipr)
            for (int y = 0; y < IMG_H; ++y) {
                float lo = 1e9f, hi = -1e9f;
                const float yc = y + 0.5f;
                for (int a = 0; a < 9; ++a) {
                    const int b = (a + 1) % 9;
                    const float ya = s.army[a], yb2 = s.army[b];
                    if ((ya <= yc) == (yb2 <= yc)) continue;
                    const float xx = s.armx[a] + (yc - ya) * (s.armx[b] - s.armx[a]) / (yb2 - ya);
                    lo = std::min(lo, xx); hi = std::max(hi, xx);
                }
                s.hipl[y] = lo <= hi ? (int16_t)lroundf(lo) : (int16_t)-1;
                s.hipr[y] = lo <= hi ? (int16_t)lroundf(hi) : (int16_t)-1;
            }
    }
    s.t = s.acc = s.fade = 0; s.disp = 0; s.awake = true; s.drawn = false;
    s.lx0 = s.rx0; s.ly0 = s.ry0; s.lx1 = s.rx0 + s.rw; s.ly1 = s.ry0 + s.rh;
    cleanup();
    s.ready = true;
    return true;
}

// ---- physics --------------------------------------------------------------------------------------------
inline void sw_collide(Sway& s, int k, const int16_t* sdf) {
    if (!sdf) return;
    for (int i = 1; i < s.nn[k]; ++i) {
        if (s.rsdf[k][i] <= 1.0f) continue;              // it starts on the body (lies in front of it): free
        float gx, gy;
        const float v = sdf_at(sdf, s.x[k][i], s.y[k][i], &gx, &gy);
        if (v < 1.0f) { s.x[k][i] += (1.0f - v) * gx; s.y[k][i] += (1.0f - v) * gy; }
    }
}
inline void sw_panels(Sway& s) {
    const float ex = s.fex, ey = s.fey;
    const int nrow = s.nn[0];
    for (int j = 1; j < nrow; ++j) {                     // a sideways swing turns the skirt round the body
        const float c = s.cc[j], R = s.cR[j];
        float u = 0;
        for (int k = 0; k < s.nr; ++k) u += (s.x[k][j] - s.rx[k][j]) * ex + (s.y[k][j] - s.ry[k][j]) * ey;
        u /= s.nr;
        float th = u / R;
        for (int it = 0; it < 4; ++it) {
            float f = 0, df = 0;
            for (int k = 0; k < s.nr; ++k) { f += sinf(s.cpsi[j][k] + th) - sinf(s.cpsi[j][k]); df += cosf(s.cpsi[j][k] + th); }
            f = R * f / s.nr - u; df = R * df / s.nr;
            if (fabsf(df) < 1e-6f) df = 1e-6f;
            th -= f / df;
        }
        th = th < -0.9f ? -0.9f : (th > 0.9f ? 0.9f : th);
        for (int k = 0; k < s.nr; ++k) {
            const float want = c + R * sinf(s.cpsi[j][k] + th), cur = s.x[k][j] * ex + s.y[k][j] * ey;
            s.x[k][j] += (want - cur) * 0.5f * ex; s.y[k][j] += (want - cur) * 0.5f * ey;
        }
    }
    for (int it = 0; it < 2; ++it)                      // one sheet: neighbouring fold lines move together
        for (int j = 1; j < nrow; ++j) {
            float dx[SR_MAX], dy[SR_MAX];
            for (int k = 0; k < s.nr; ++k) { dx[k] = s.x[k][j] - s.rx[k][j]; dy[k] = s.y[k][j] - s.ry[k][j]; }
            float nx_[SR_MAX], ny_[SR_MAX], sx = 0, sy = 0;   // (the row keeps its shift: only its sharing-out evens up)
            for (int k = 0; k < s.nr; ++k) {
                nx_[k] = dx[k]; ny_[k] = dy[k];
                if (k > 0 && k < s.nr - 1) { nx_[k] += 0.5f * ((dx[k - 1] + dx[k + 1]) * 0.5f - dx[k]); ny_[k] += 0.5f * ((dy[k - 1] + dy[k + 1]) * 0.5f - dy[k]); }
                sx += dx[k] - nx_[k]; sy += dy[k] - ny_[k];
            }
            for (int k = 0; k < s.nr; ++k) { s.x[k][j] = s.rx[k][j] + nx_[k] + sx / s.nr; s.y[k][j] = s.ry[k][j] + ny_[k] + sy / s.nr; }
        }
    for (int k = 0; k < s.nr - 1; ++k)                  // pleats: open no wider than flat, close no tighter than the fabric
        for (int j = 1; j < nrow; ++j) {
            const float w = (s.x[k + 1][j] - s.x[k][j]) * ex + (s.y[k + 1][j] - s.y[k][j]) * ey;
            float tg = w + (s.W0[k] - w) * 0.12f;
            tg = tg < 1.5f ? 1.5f : (tg > s.Lm[k] ? s.Lm[k] : tg);
            const float corr = (tg - w) * 0.5f;
            s.x[k][j] -= corr * ex; s.y[k][j] -= corr * ey; s.x[k + 1][j] += corr * ex; s.y[k + 1][j] += corr * ey;
        }
}
// The skirt's two edges can't swing in through her hips / thighs (an edge outside her hip line in the
// photo stays outside it).
inline void sw_hips(Sway& s) {
    if (!s.hipl || !s.hipr) return;
    for (int side = 0; side < 2; ++side) {
        const int k = side ? s.nr - 1 : 0;
        for (int j = 1; j < s.nn[k]; ++j) {
            const int yr = (int)lroundf(s.ry[k][j]), yc = (int)lroundf(s.y[k][j]);
            if (yr < 0 || yr >= IMG_H || yc < 0 || yc >= IMG_H) continue;
            if (!side && s.hipl[yr] >= 0 && s.rx[k][j] <= s.hipl[yr] + 0.5f && s.hipl[yc] >= 0) s.x[k][j] = std::min(s.x[k][j], (float)s.hipl[yc]);
            if (side && s.hipr[yr] >= 0 && s.rx[k][j] >= s.hipr[yr] - 0.5f && s.hipr[yc] >= 0) s.x[k][j] = std::max(s.x[k][j], (float)s.hipr[yc]);
        }
    }
}
inline void sw_rods(Sway& s, float fx, float fy, float wind, float damp_k, const int16_t* sdf, float hold) {
    const SwayKindInfo& K = sway_kind_info(s.kind);
    const float h = SW_DT, damp = 1.0f - K.c * damp_k * h, stiff = s.prm[SP_STIFF];
    const bool cloth = s.kind == SK_CLOTH;
    for (int k = 0; k < s.nr; ++k) {
        const int n = s.nn[k];
        float wdx, wdy;
        if (cloth) { wdx = s.fex; wdy = s.fey; }
        else { wdx = -(s.ry[k][n - 1] - s.ry[k][0]); wdy = s.rx[k][n - 1] - s.rx[k][0]; }
        float wl = sqrtf(wdx * wdx + wdy * wdy); if (wl < 1e-9f) wl = 1;
        for (int i = 0; i < n; ++i) {
            const float wk = (0.3f + 0.7f * i / (n - 1)) * wind;
            const float kn = hold * s.kn[k][i];
            float ax = fx + kn * (s.rx[k][i] - s.x[k][i]) + wk * wdx / wl;
            float ay = fy + kn * (s.ry[k][i] - s.y[k][i]) + wk * wdy / wl;
            if (K.mu > 0 && s.onb[k][i]) {               // lying on the body: rubbing on it takes some push, slows it
                const float ex_ = ax - kn * (s.rx[k][i] - s.x[k][i]), ey_ = ay - kn * (s.ry[k][i] - s.y[k][i]);
                const float m = std::min(0.8f, 1.5f * K.mu);
                ax -= ex_ * m; ay -= ey_ * m;
                const float v = std::min(0.9f, 8.0f * K.mu * h);
                s.xp[k][i] += (s.x[k][i] - s.xp[k][i]) * v; s.yp[k][i] += (s.y[k][i] - s.yp[k][i]) * v;
            }
            for (int p = 0; cloth && p < s.npush; ++p) {   // under it, a form jiggling: the cloth on it carried along
                const float* P = s.push[p];
                const float ddx = s.x[k][i] - P[0], ddy = s.y[k][i] - P[1];
                float wq = 1.0f - sqrtf(ddx * ddx + ddy * ddy) / (1.6f * P[2]);
                if (wq <= 0.0f) continue;
                if (wq > 1.0f) wq = 1.0f;
                ax += SW_PUSH_K * wq * (P[3] - (s.x[k][i] - s.xp[k][i]) / h);
                ay += SW_PUSH_K * wq * (P[4] - (s.y[k][i] - s.yp[k][i]) / h);
            }
            float nxp = s.x[k][i] + (s.x[k][i] - s.xp[k][i]) * damp + ax * h * h;
            float nyp = s.y[k][i] + (s.y[k][i] - s.yp[k][i]) * damp + ay * h * h;
            if (i == 0) { nxp = s.rx[k][0]; nyp = s.ry[k][0]; }
            s.xp[k][i] = s.x[k][i]; s.yp[k][i] = s.y[k][i]; s.x[k][i] = nxp; s.y[k][i] = nyp;
        }
    }
    if (cloth) sw_panels(s);
    for (int k = 0; k < s.nr; ++k) {
        const int n = s.nn[k];
        float* X = s.x[k]; float* Y = s.y[k]; const float* RX = s.rx[k]; const float* RY = s.ry[k];
        if (stiff > 0 && !cloth)                         // shape memory: each link turns with the one above it
            for (int i = 2; i < n; ++i) {
                const float ang = atan2f(Y[i - 1] - Y[i - 2], X[i - 1] - X[i - 2]) - atan2f(RY[i - 1] - RY[i - 2], RX[i - 1] - RX[i - 2]);
                const float cs = cosf(ang), sn = sinf(ang), vx = RX[i] - RX[i - 1], vy = RY[i] - RY[i - 1];
                X[i] += (X[i - 1] + vx * cs - vy * sn - X[i]) * (0.35f * stiff);
                Y[i] += (Y[i - 1] + vx * sn + vy * cs - Y[i]) * (0.35f * stiff);
            }
        if (!cloth) sw_collide(s, k, sdf);              // (a skirt swings in front of her legs: no collision)
        for (int i = 1; i < n; ++i) {                    // follow the leader, and no joint bends past BEND_MAX of its photo bend
            float dx = X[i] - X[i - 1], dy = Y[i] - Y[i - 1], l = sqrtf(dx * dx + dy * dy);
            if (l < 1e-6f) l = 1e-6f;
            if (i >= 2 && !cloth) {                      // (fabric has no bend limit: it folds right over)
                const float pcx = X[i - 1] - X[i - 2], pcy = Y[i - 1] - Y[i - 2], prx = RX[i - 1] - RX[i - 2], pry = RY[i - 1] - RY[i - 2];
                const float rrx = RX[i] - RX[i - 1], rry = RY[i] - RY[i - 1];
                const float rest_b = atan2f(prx * rry - pry * rrx, prx * rrx + pry * rry);
                const float now_b = atan2f(pcx * dy - pcy * dx, pcx * dx + pcy * dy);
                float db = now_b - rest_b + 3.14159265f;
                db -= 6.2831853f * floorf(db / 6.2831853f); db -= 3.14159265f;
                if (fabsf(db) > SW_BEND) {
                    const float ang = atan2f(pcy, pcx) + rest_b + (db > 0 ? SW_BEND : -SW_BEND);
                    dx = cosf(ang) * l; dy = sinf(ang) * l;
                }
            }
            // fabric: a link never stretches, but it can foreshorten - a fold turning toward you or away
            // (upside down the hem comes up in front of her, over the waist); hair keeps its exact length
            const float f = cloth ? std::min(1.0f, s.len[k][i] / l) : s.len[k][i] / l;
            X[i] = X[i - 1] + dx * f; Y[i] = Y[i - 1] + dy * f;
        }
        for (int i = 0; i < n; ++i) {                    // what the fill can't make true limits how far it goes
            const float ddx = X[i] - RX[i], ddy = Y[i] - RY[i], dm = sqrtf(ddx * ddx + ddy * ddy);
            if (dm > s.cap[k][i]) { X[i] = RX[i] + ddx * s.cap[k][i] / dm; Y[i] = RY[i] + ddy * s.cap[k][i] / dm; }
        }
    }
    if (cloth) { sw_panels(s); sw_hips(s); }
}
inline void sw_pend(Sway& s, float fx, float fy, float wind, float damp_k, const float dn[3], const int16_t* sdf) {
    const float h = SW_DT, c = cosf(s.th), sn = sinf(s.th);
    const float rx = s.rcx * c - s.rcy * sn, ry = s.rcx * sn + s.rcy * c;
    const float gx = SW_G * dn[0], gy = SW_G * dn[1];
    float tq = s.mass * (rx * (gy + fy) - ry * (gx + fx)) + s.tau0 * sqrtf(dn[0] * dn[0] + dn[1] * dn[1]);
    tq += s.mass * wind * sqrtf(rx * rx + ry * ry);
    float nw = s.th + (s.th - s.thp) * (1.0f - sway_kind_info(s.kind).c * damp_k * h) + tq / s.I * h * h;
    if (sdf) {                                           // its outline can't go into the body
        const float cs = cosf(nw), ss = sinf(nw);
        for (int i = 0; i < s.no; ++i) {
            if (s.ors[i] <= 1.0f) continue;
            const float qx = s.ox[i] - s.pinx, qy = s.oy[i] - s.piny;
            if (sdf_at(sdf, s.pinx + qx * cs - qy * ss, s.piny + qx * ss + qy * cs) < 1.0f) { nw = s.th - (nw - s.th) * 0.3f; break; }
        }
    }
    s.thp = s.th; s.th = nw;
}
// dn: unit "down" (screen axes, upright (0, 1, 0)); jolt: g, gravity removed; damp_k < 1 while the motion
// echo plays. Returns true while it moves.
inline bool sway_step(Sway& s, float dt, const float dn[3], const float jolt[3], float damp_k, const int16_t* sdf) {
    if (!s.ready) return false;
    const SwayKindInfo& K = sway_kind_info(s.kind);
    const float sw = s.prm[SP_SWING], m = sqrtf(dn[0] * dn[0] + dn[1] * dn[1]);
    s.acc += dt < 0.1f ? dt : 0.1f;
    while (s.acc >= SW_DT) {
        s.acc -= SW_DT; s.t += SW_DT;
        const float T = 6.2831853f * s.t;
        const float wind = sw * s.prm[SP_BREEZE] * K.wind * (0.6f * sinf(T / 3.1f) + 0.4f * sinf(T / 1.7f + 1.0f));
        if (s.kind == SK_PENDANT) sw_pend(s, -sw * K.j * jolt[0], -sw * K.j * jolt[1], wind, damp_k, dn, sdf);
        else {
            const float g = sw * K.gain;
            // a skirt: the photo's pose is held up by her only while gravity points down the picture -
            // that part is carried; the rest (sideways, upside down) pulls it
            const bool cl = s.kind == SK_CLOTH || s.kind == SK_HAIR;   // (hair too: it keeps a little more of its shape)
            const float gy = cl ? SW_G * std::min(0.0f, dn[1]) : SW_G * (dn[1] - m);
            sw_rods(s, g * (SW_G * dn[0] - K.j * jolt[0]), g * (gy - K.j * jolt[1]), wind, damp_k, sdf,
                    cl ? std::max(s.kind == SK_CLOTH ? SW_CLOTH_HOLD : SW_HAIR_HOLD, dn[1]) : 1.0f);
        }
    }
    float disp = 0, vel = 0;
    if (s.kind == SK_PENDANT) { disp = fabsf(s.th) * s.tlen; vel = fabsf(s.th - s.thp) * s.tlen; }
    else for (int k = 0; k < s.nr; ++k) for (int i = 0; i < s.nn[k]; ++i) {
        disp = std::max(disp, std::max(fabsf(s.x[k][i] - s.rx[k][i]), fabsf(s.y[k][i] - s.ry[k][i])));
        vel = std::max(vel, std::max(fabsf(s.x[k][i] - s.xp[k][i]), fabsf(s.y[k][i] - s.yp[k][i])));
    }
    s.fade = disp < 1.0f ? disp : 1.0f;
    s.disp = disp;
    s.awake = disp > 0.05f || vel > 0.002f || s.prm[SP_BREEZE] > 0;
    return s.awake;
}
// A beat: the free ends get a small sideways nudge (music).
inline void sway_kick(Sway& s, float amp, int sign) {
    if (!s.ready) return;
    if (s.kind == SK_PENDANT) { s.thp -= sign * amp * 90.0f / std::max(s.tlen, 4.0f) * SW_DT; }
    else for (int k = 0; k < s.nr; ++k) {
        const int n = s.nn[k];
        float dx = s.rx[k][n - 1] - s.rx[k][0], dy = s.ry[k][n - 1] - s.ry[k][0], l = sqrtf(dx * dx + dy * dy);
        if (l < 1e-6f) l = 1;
        const float nx = -dy / l, ny = dx / l;
        for (int i = 0; i < n; ++i) {
            const float w = powf((float)i / (n - 1), 1.5f) * sign * amp * 120.0f * SW_DT;
            s.xp[k][i] -= w * nx; s.yp[k][i] -= w * ny;
        }
    }
    s.awake = true;
}
// A tap on it: a flick away from the tap.
inline bool sway_poke(Sway& s, float px, float py) {
    if (!s.ready) return false;
    const int X = (int)px, Y = (int)py;
    if (!sw_in(s, X, Y) || s.own[sw_i(s, X, Y)] < 0) return false;
    if (s.kind == SK_PENDANT) {
        const float side = (px - s.pinx) * s.rcy - (py - s.piny) * s.rcx;
        s.thp += (side > 0 ? 1 : -1) * 200.0f / std::max(s.tlen, 4.0f) * SW_DT;
    } else for (int k = 0; k < s.nr; ++k) {
        const int n = s.nn[k];
        float dx = s.rx[k][n - 1] - s.rx[k][0], dy = s.ry[k][n - 1] - s.ry[k][0], l = sqrtf(dx * dx + dy * dy);
        if (l < 1e-6f) l = 1;
        const float nx = -dy / l, ny = dx / l, side = (px - s.rx[k][0]) * nx + (py - s.ry[k][0]) * ny;
        for (int i = 0; i < n; ++i) {
            const float d = sqrtf((s.x[k][i] - px) * (s.x[k][i] - px) + (s.y[k][i] - py) * (s.y[k][i] - py));
            const float f = std::max(0.0f, 1.0f - d / 30.0f);
            if (f <= 0) continue;
            s.xp[k][i] += (side > 0 ? 1 : -1) * 150.0f * SW_DT * f * nx; s.yp[k][i] += (side > 0 ? 1 : -1) * 150.0f * SW_DT * f * ny;
        }
    }
    s.awake = true;
    return true;
}

// ---- drawing --------------------------------------------------------------------------------------------
// Where it draws now (screen): its buffers' rect (what it uncovers) and its mesh as posed.
inline void sway_bounds(const Sway& s, int& x0, int& y0, int& x1, int& y1) {
    static SwTri tris[SW_TRIS];
    x0 = s.rx0; y0 = s.ry0; x1 = s.rx0 + s.rw; y1 = s.ry0 + s.rh;
    const int nt = sway_mesh(s, tris, SW_TRIS, false);
    for (int k = 0; k < nt; ++k)
        for (int v = 0; v < 3; ++v) {
            x0 = std::min(x0, (int)floorf(tris[k].dx[v])); x1 = std::max(x1, (int)ceilf(tris[k].dx[v]) + 1);
            y0 = std::min(y0, (int)floorf(tris[k].dy[v])); y1 = std::max(y1, (int)ceilf(tris[k].dy[v]) + 1);
        }
    x0 = std::max(x0, s.x0); y0 = std::max(y0, s.y0); x1 = std::min(x1, s.x1); y1 = std::min(y1, s.y1);
}
// This frame's rect to redraw: what it drew last time and what it draws now (ux*); then remembered.
inline void sway_dirty(Sway& s) {
    int x0, y0, x1, y1;
    sway_bounds(s, x0, y0, x1, y1);
    s.ux0 = std::min(x0, s.lx0); s.uy0 = std::min(y0, s.ly0); s.ux1 = std::max(x1, s.lx1); s.uy1 = std::max(y1, s.ly1);
    s.lx0 = x0; s.ly0 = y0; s.lx1 = x1; s.ly1 = y1;
}

// Drawing it: the mesh as posed, back to front (made once a frame; both cores then draw rows of it).
struct SwDraw { SwTri tris[SW_TRIS]; int order[SW_TRIS]; int nt = 0; };
inline void sway_draw_prep(const Sway& s, SwDraw& D) {
    D.nt = s.ready ? sway_mesh(s, D.tris, SW_TRIS, false) : 0;
    for (int k = 0; k < D.nt; ++k) D.order[k] = k;
    std::stable_sort(D.order, D.order + D.nt, [&](int a, int b) { return D.tris[a].depth < D.tris[b].depth; });
}
// This frame's rect from a prepared mesh (as sway_dirty, without making it again).
inline void sway_dirty_d(Sway& s, const SwDraw& D) {
    int x0 = s.rx0, y0 = s.ry0, x1 = s.rx0 + s.rw, y1 = s.ry0 + s.rh;
    for (int k = 0; k < D.nt; ++k)
        for (int v = 0; v < 3; ++v) {
            x0 = std::min(x0, (int)floorf(D.tris[k].dx[v])); x1 = std::max(x1, (int)ceilf(D.tris[k].dx[v]) + 1);
            y0 = std::min(y0, (int)floorf(D.tris[k].dy[v])); y1 = std::max(y1, (int)ceilf(D.tris[k].dy[v]) + 1);
        }
    x0 = std::max(x0, s.x0); y0 = std::max(y0, s.y0); x1 = std::min(x1, s.x1); y1 = std::min(y1, s.y1);
    s.ux0 = std::min(x0, s.lx0); s.uy0 = std::min(y0, s.ly0); s.ux1 = std::max(x1, s.lx1); s.uy1 = std::max(y1, s.ly1);
    s.lx0 = x0; s.ly0 = y0; s.lx1 = x1; s.ly1 = y1;
}
// Into out (the layer, RGB565; osw = byte-swapped), rows [ya, yb): the plate under it, then the mesh.
// At rest it draws the photo exactly. (Each triangle row by row over just its span.)
JIG_INLINE void sw_put(uint16_t* out, bool osw, int X, int Y, uint16_t v) { out[Y * IMG_W + X] = osw ? (uint16_t)((v >> 8) | (v << 8)) : v; }
JIG_INLINE float sw_al(const uint8_t* AL, int i, float keep, float fade) { return AL[i] ? keep + AL[i] * (1.0f / 255.0f) * fade : 0.0f; }
JIG_HOT void sway_draw_rows(const Sway& s, const SwDraw& D, uint16_t* out, bool osw, int ya, int yb) {
    if (!s.ready) return;
    for (int y = std::max(ya, s.ry0); y < std::min(yb, s.ry0 + s.rh); ++y)
        for (int x = 0; x < s.rw; ++x) { const int i = (y - s.ry0) * s.rw + x; if (s.hole[i]) sw_put(out, osw, s.rx0 + x, y, s.plate[i]); }
    const SwTri* tris = D.tris; const int* order = D.order; const int nt = D.nt;
    const float fade = s.fade, keep = 1.0f - fade;
    const bool hair = s.kind == SK_HAIR;
    const int RW = s.rw, RH = s.rh;
    const float* C8 = s.c8; const float* T8 = s.t8; const uint8_t* AL = s.alpha;
    const uint16_t* PH = s.photo; const uint16_t* TX = s.texd; const uint8_t* SO = s.solid;
    for (int q = 0; q < nt; ++q) {
        const SwTri& T = tris[order[q]];
        const float* X = T.dx; const float* Y = T.dy;
        const float den = (Y[1] - Y[2]) * (X[0] - X[2]) + (X[2] - X[1]) * (Y[0] - Y[2]);
        if (fabsf(den) < 1e-9f) continue;
        const float id = 1.0f / den;
        const int by0 = std::max(ya, (int)floorf(std::min(Y[0], std::min(Y[1], Y[2])))), by1 = std::min(yb - 1, (int)ceilf(std::max(Y[0], std::max(Y[1], Y[2]))));
        if (by0 > by1) continue;
        const int bx0 = std::max(0, (int)floorf(std::min(X[0], std::min(X[1], X[2])))), bx1 = std::min(IMG_W - 1, (int)ceilf(std::max(X[0], std::max(X[1], X[2]))));
        // barycentrics are affine in x: l(x) = a + b·x per row
        const float a0x = (Y[1] - Y[2]) * id, a1x = (Y[2] - Y[0]) * id;
        const float sx0 = T.sx[0] - s.rx0, sx1 = T.sx[1] - s.rx0, sx2 = T.sx[2] - s.rx0;
        const float sy0 = T.sy[0] - s.ry0, sy1 = T.sy[1] - s.ry0, sy2 = T.sy[2] - s.ry0;
        for (int y = by0; y <= by1; ++y) {
            const float b0 = ((X[2] - X[1]) * (y - Y[2])) * id - a0x * X[2];
            const float b1 = ((X[0] - X[2]) * (y - Y[2])) * id - a1x * X[2];
            // the span where all three are ≥ 0 (then the exact test per pixel, as the Studio's lab)
            float lo = (float)bx0, hi = (float)bx1;
            auto clip = [&](float a, float b) {          // a·x + b ≥ −eps
                if (fabsf(a) < 1e-12f) { if (b < -1e-5f) { lo = 1; hi = 0; } return; }
                const float r = (-1e-5f - b) / a;
                if (a > 0) lo = std::max(lo, r); else hi = std::min(hi, r);
            };
            clip(a0x, b0); clip(a1x, b1); clip(-a0x - a1x, 1.0f - b0 - b1);
            if (lo > hi) continue;
            const int xa = std::max(bx0, (int)floorf(lo) - 1), xb = std::min(bx1, (int)ceilf(hi) + 1);
            for (int x = xa; x <= xb; ++x) {
                const float l0 = a0x * x + b0, l1 = a1x * x + b1, l2 = 1 - l0 - l1;
                if (l0 < -1e-6f || l1 < -1e-6f || l2 < -1e-6f) continue;
                float sx = l0 * sx0 + l1 * sx1 + l2 * sx2, sy = l0 * sy0 + l1 * sy1 + l2 * sy2;
                sx = sx < 0 ? 0 : (sx > RW - 1.001f ? RW - 1.001f : sx); sy = sy < 0 ? 0 : (sy > RH - 1.001f ? RH - 1.001f : sy);
                const int ix = (int)sx, iy = (int)sy;
                if (hair) {                              // only the pixels this lock carries
                    const int ox_ = sw_rint_pos(sx), oy_ = sw_rint_pos(sy);
                    if (s.own[std::min(oy_, RH - 1) * RW + std::min(ox_, RW - 1)] != T.oid) continue;
                }
                const float fx = sx - ix, fy = sy - iy;
                const int ix1 = ix + 1 < RW ? ix + 1 : RW - 1, iy1 = iy + 1 < RH ? iy + 1 : RH - 1;
                const int i00 = iy * RW + ix, i10 = iy * RW + ix1, i01 = iy1 * RW + ix, i11 = iy1 * RW + ix1;
                const float w0 = (1 - fx) * (1 - fy), w1 = fx * (1 - fy), w2 = (1 - fx) * fy, w3 = fx * fy;
                if (SO ? (SO[i00] & SO[i10] & SO[i01] & SO[i11]) != 0
                       : ((AL[i00] & AL[i10] & AL[i01] & AL[i11]) == 255 && PH[i00] == TX[i00] && PH[i10] == TX[i10] &&
                          PH[i01] == TX[i01] && PH[i11] == TX[i11])) {   // inside it (most of it): solid, its own colour
                    const float r = (C8[3 * i00] * w0 + C8[3 * i10] * w1 + C8[3 * i01] * w2 + C8[3 * i11] * w3) * T.shade;
                    const float g = (C8[3 * i00 + 1] * w0 + C8[3 * i10 + 1] * w1 + C8[3 * i01 + 1] * w2 + C8[3 * i11 + 1] * w3) * T.shade;
                    const float b = (C8[3 * i00 + 2] * w0 + C8[3 * i10 + 2] * w1 + C8[3 * i01 + 2] * w2 + C8[3 * i11 + 2] * w3) * T.shade;
                    sw_put(out, osw, x, y, sw_pack(r, g, b));
                    continue;
                }
                const float A = sw_al(AL, i00, keep, fade) * w0 + sw_al(AL, i10, keep, fade) * w1 +
                                sw_al(AL, i01, keep, fade) * w2 + sw_al(AL, i11, keep, fade) * w3;
                if (A <= 0) continue;
                float col[3];
                for (int c = 0; c < 3; ++c) {
                    float v = C8[3 * i00 + c] * w0 + C8[3 * i10 + c] * w1 + C8[3 * i01 + c] * w2 + C8[3 * i11 + c] * w3;
                    if (fade > 0) {
                        const float t = T8[3 * i00 + c] * w0 + T8[3 * i10 + c] * w1 + T8[3 * i01 + c] * w2 + T8[3 * i11 + c] * w3;
                        v = v * keep + t * fade;
                    }
                    col[c] = v * T.shade;
                }
                const uint16_t ov0 = out[y * IMG_W + x], ov = osw ? (uint16_t)((ov0 >> 8) | (ov0 << 8)) : ov0;
                float orr, og, ob; sw_unpack(ov, orr, og, ob);
                sw_put(out, osw, x, y, sw_pack(orr * (1 - A) + col[0] * A, og * (1 - A) + col[1] * A, ob * (1 - A) + col[2] * A));
            }
        }
    }
}
inline void sway_render(const Sway& s, uint16_t* out, bool osw) {
    static SwDraw D;
    sway_draw_prep(s, D);
    sway_draw_rows(s, D, out, osw, 0, IMG_H);
}

// ---- .jig lines (mg_sway.py) ----------------------------------------------------------------------------
//   sway <hair|cloth|pendant> [swing= breeze= stiff= pleat=] [name=] [on=]
//   sguide x,y x,y …        (hair: a lock root → tip; cloth: the waistband, then any marked folds (one point); pendant: its pin)
//   srow y a-b …            (what moves)
//   splate y x HHHH…        (the Studio's fill behind it: RGB565, 4 hex digits a pixel)
//   body / brow y a-b …     (what they rest on: the figures, once per photo)
inline bool parse_sway_line(char* line, Sway* ss, int& ns, int& cur, uint8_t*& body, void* (*alloc)(size_t)) {
    char kind[8];
    int used = 0, n = 0;
    if (sscanf(line, "%7s%n", kind, &used) != 1) return false;
    char* rest = line + used;
    if (strcmp(kind, "body") == 0) {
        cur = -2;
        if (!body) { body = (uint8_t*)alloc(MASK_BYTES); if (body) memset(body, 0, MASK_BYTES); }
        return true;
    }
    if (strcmp(kind, "brow") == 0) {
        if (cur != -2 || !body) return true;
        int y; if (sscanf(rest, "%d%n", &y, &n) != 1) return true;
        rest += n; int a, b;
        while (sscanf(rest, " %d-%d%n", &a, &b, &n) == 2) { rest += n; mask_span(body, y, a, b, true); }
        return true;
    }
    if (strcmp(kind, "sway") == 0) {
        cur = -1;
        if (ns >= SWAY_MAX) return true;
        Sway& s = ss[ns];
        sway_free(s); s = Sway();
        char k[16];
        if (sscanf(rest, "%15s%n", k, &n) == 1) {
            for (int q = 0; q < SK_COUNT; ++q) if (strcmp(k, sway_kind_info((uint8_t)q).key) == 0) s.kind = (uint8_t)q;
            rest += n;
        }
        char tok[40];
        while (sscanf(rest, "%39s%n", tok, &n) == 1) {
            rest += n;
            char* eq = strchr(tok, '='); if (!eq) continue; *eq = 0;
            auto cp = [](char* d, const char* sname) { size_t L = strlen(sname); if (L > 15) L = 15; memcpy(d, sname, L); d[L] = 0; };
            if (strcmp(tok, "name") == 0) { cp(s.name, eq + 1); continue; }
            if (strcmp(tok, "on") == 0) { cp(s.on, eq + 1); continue; }
            for (int p = 0; p < SP_COUNT; ++p)
                if (strcmp(tok, sparam_info((uint8_t)p).key) == 0) {
                    char* end = nullptr; float v = strtof(eq + 1, &end);
                    const SParamInfo& pi = sparam_info((uint8_t)p);
                    if (end != eq + 1 && !isnan(v)) s.prm[p] = v < pi.lo ? pi.lo : (v > pi.hi ? pi.hi : v);
                }
        }
        if (!sway_ensure_mask(s, alloc)) return true;
        cur = ns++;
        return true;
    }
    if (strcmp(kind, "hang") == 0) {                    // (an older file: "hang hair|cloth|charm ax,ay bx,by [swing=…]")
        cur = -1;
        if (ns >= SWAY_MAX) return true;
        char kn[10]; int ax, ay, bx, by;
        if (sscanf(rest, "%9s %d,%d %d,%d%n", kn, &ax, &ay, &bx, &by, &n) != 5) return true;
        rest += n;
        Sway& s = ss[ns];
        sway_free(s); s = Sway();
        s.kind = strcmp(kn, "cloth") == 0 ? SK_CLOTH : (strcmp(kn, "charm") == 0 ? SK_PENDANT : SK_HAIR);
        s.lax = (int16_t)ax; s.lay = (int16_t)ay; s.lbx = (int16_t)bx; s.lby = (int16_t)by;
        char tok[40];
        while (sscanf(rest, "%39s%n", tok, &n) == 1) {
            rest += n;
            char* eq = strchr(tok, '='); if (!eq) continue; *eq = 0;
            const int p = strcmp(tok, "swing") == 0 ? SP_SWING : (strcmp(tok, "breeze") == 0 ? SP_BREEZE : -1);
            if (p >= 0) { const float v = strtof(eq + 1, nullptr); if (!isnan(v)) s.prm[p] = std::max(sparam_info((uint8_t)p).lo, std::min(sparam_info((uint8_t)p).hi, v)); }
        }
        if (!sway_ensure_mask(s, alloc)) return true;
        cur = ns++;
        return true;
    }
    if (strcmp(kind, "hrow") == 0) strcpy(kind, "srow");
    if (strcmp(kind, "sarm") == 0) {
        if (cur < 0) return true;
        Sway& s = ss[cur];
        int k = 0, x, y;
        while (k < 9 && sscanf(rest, " %d,%d%n", &x, &y, &n) == 2) { rest += n; s.armx[k] = (int16_t)x; s.army[k] = (int16_t)y; ++k; }
        s.narm = k == 9 ? 9 : 0;
        s.arm_back = strstr(rest, "back") != nullptr;
        return true;
    }
    if (strcmp(kind, "sguide") == 0) {
        if (cur < 0) return true;
        Sway& s = ss[cur];
        if (s.ng >= SG_MAX) return true;
        int k = 0; float x, y;
        while (k < SG_PTS && sscanf(rest, " %f,%f%n", &x, &y, &n) == 2) { rest += n; s.gx[s.ng][k] = x; s.gy[s.ng][k] = y; ++k; }
        if (k) s.gn[s.ng++] = k;
        return true;
    }
    if (strcmp(kind, "srow") == 0) {
        if (cur < 0) return true;
        int y; if (sscanf(rest, "%d%n", &y, &n) != 1) return true;
        rest += n; int a, b;
        while (sscanf(rest, " %d-%d%n", &a, &b, &n) == 2) { rest += n; mask_span(ss[cur].mask, y, a, b, true); }
        return true;
    }
    if (strcmp(kind, "splate") == 0) {
        if (cur < 0) return true;
        Sway& s = ss[cur];
        int y, x0;
        if (sscanf(rest, "%d %d%n", &y, &x0, &n) != 2 || y < 0 || y >= IMG_H) return true;
        rest += n;
        while (*rest == ' ') ++rest;
        if (!s.pdef) { s.pdef = (uint16_t*)alloc((size_t)IMG_W * IMG_H * 2); s.pdef_has = (uint8_t*)alloc(MASK_BYTES); if (s.pdef_has) memset(s.pdef_has, 0, MASK_BYTES); }
        if (!s.pdef || !s.pdef_has) return true;
        s.pdef_ok = true;
        for (int x = x0; x < IMG_W && rest[0] && rest[1] && rest[2] && rest[3]; ++x, rest += 4) {
            char hx[5] = {rest[0], rest[1], rest[2], rest[3], 0};
            const uint16_t v = (uint16_t)strtoul(hx, nullptr, 16);
            if (x >= 0) { s.pdef[y * IMG_W + x] = v; bset_(s.pdef_has, y * IMG_W + x); }
        }
        return true;
    }
    return false;
}
template <typename F>
inline void save_sways(F& f, const Sway* ss, int ns, const uint8_t* body) {
    if (body && ns > 0) {
        bool any = false;
        for (int i = 0; i < MASK_BYTES && !any; ++i) any = body[i] != 0;
        if (any) { f.printf("body\n"); save_mask_rows(f, body, "brow"); }
    }
    for (int i = 0; i < ns; ++i) {
        const Sway& s = ss[i];
        if (!sway_ok(s)) continue;
        f.printf("sway %s", sway_kind_info(s.kind).key);
        for (int p = 0; p < SP_COUNT; ++p)
            if (fabsf(s.prm[p] - sparam_info((uint8_t)p).dflt) > 1e-6f) f.printf(" %s=%g", sparam_info((uint8_t)p).key, s.prm[p]);
        if (s.name[0]) f.printf(" name=%s", s.name);       // (kept as written: spaces are _)
        if (s.on[0]) f.printf(" on=%s", s.on);
        f.printf("\n");
        for (int g = 0; g < s.ng; ++g) {
            f.printf("sguide");
            for (int k = 0; k < s.gn[g]; ++k) f.printf(" %d,%d", (int)lroundf(s.gx[g][k]), (int)lroundf(s.gy[g][k]));
            f.printf("\n");
        }
        if (s.kind == SK_CLOTH && s.narm == 9) {
            f.printf("sarm");
            for (int k = 0; k < 9; ++k) f.printf(" %d,%d", s.armx[k], s.army[k]);
            f.printf(s.arm_back ? " back\n" : "\n");
        }
        save_mask_rows(f, s.mask, "srow");
        if (s.pdef_ok && s.pdef && s.pdef_has)
            for (int y = 0; y < IMG_H; ++y) {
                int x = 0;
                while (x < IMG_W) {
                    while (x < IMG_W && !bget_(s.pdef_has, y * IMG_W + x)) ++x;
                    if (x >= IMG_W) break;
                    f.printf("splate %d %d ", y, x);
                    while (x < IMG_W && bget_(s.pdef_has, y * IMG_W + x)) { f.printf("%04x", s.pdef[y * IMG_W + x]); ++x; }
                    f.printf("\n");
                }
            }
    }
}

}  // namespace jig
