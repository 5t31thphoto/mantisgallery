/**
 * Jiggle zones — device side: canvases, IMU drive, rendering, .jig files and the on-screen editor.
 * The maths (shapes, weights, springs, warp) is in jiggle_core.h; the beat tracker in jiggle_dance.h.
 *
 * .jig file (hidden, next to the image: /images/santa.mg → /images/.santa.mg.jig):
 *     MGJIG1
 *     round 120 130 62 0 natural          ← shape  cx  cy  radius  rotation°  [material]
 *     tear 150 92 14 0 soft tip 0 -50     ← teardrop with a stiffer tip at local (0, -0.50)
 * Material and control point ("tip u w", local units ×100, any shape) are optional; older files load
 * as "natural" with no control point. The PC editor can add fine-tuning as key=value (f1=2.4 z1=0.3 …,
 * see Tune in jiggle_core.h); the Core2 keeps them when you edit a zone here.
 *
 * Motion is physics: gravity is pinned to the real one (tip it, turn it upside down, lay it flat),
 * jolts and twists push the flesh around, and a repeating motion you make is recorded and echoed for a
 * few seconds after you stop. Dance is music only: while the microphone hears music, zones also get a
 * kick on each beat (weighted by each zone's dance slider), on top of whatever the physics is doing.
 *
 * Rendering: zones that moved are grouped (overlapping ones render together, each pixel once), then
 * drawn in 12-row strips into two small internal-RAM buffers: while one strip goes to the screen by
 * DMA the next is being rendered. Each strip only covers the columns the zones actually reach.
 */
#include "jiggle.h"
#include "jiggle_core.h"
#include "jiggle_dance.h"
#include "jiggle_water.h"
#include "jiggle_select.h"
#include "jiggle_grabcut.h"
#include "jiggle_parallax.h"
#include "jiggle_sway.h"
#include <new>
#include <SD.h>
#if defined(ESP_PLATFORM)
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>
#endif
#include <string.h>
#include <stdio.h>

namespace jig {
namespace {

constexpr uint16_t C_BG    = 0x18C3;
constexpr uint16_t C_TEAL  = 0x038E;
constexpr uint16_t C_LIME  = 0xB7E0;
constexpr uint16_t C_TEXT  = 0xEF5D;
constexpr uint16_t C_HINT  = 0x9CF3;
constexpr uint16_t C_DIM   = 0x8410;
constexpr uint16_t C_BLACK = 0x0000;
constexpr uint16_t C_YEL   = 0xFFE0;

// ---- Feel ------------------------------------------------------------------------------------
// IMU → screen axes for the Core2 in landscape (rotation 1). Zones should LAG behind a shove (move
// the Core2 right → belly swings left, then wobbles back). If an axis does the opposite, flip its sign.
constexpr float IMU_SX = 1.0f, IMU_SY = -1.0f, IMU_SZ = 1.0f;
constexpr float SENS = 1.4f;             // swing per g of jolt / flick (1.0 = zone's full travel)
// Gravity is pinned to the real one: the photo's own pose is "held upright". Tip it and the flesh hangs
// toward the real down; upside down it pulls toward the top of the image; face-up it flattens.
constexpr float GRAV_TAU = 0.25f;        // s: splits slow gravity from fast jolts
constexpr float GRAV_K = 0.72f;          // hang per g of gravity away from "upright" (× zone's sag slider)
constexpr float GRAV_Z = 0.50f;          // face-up flattens / face-down bulges (× zone's sag slider)
// Twisting in the screen plane (gyro): zones lag the twist and fling outward when spun.
constexpr float GYRO_CW = -1.0f;         // gyro z → clockwise-on-screen; flip if twists make zones lead
constexpr float PX_M = 0.000127f;        // Core2 screen ≈ 0.127 mm per pixel
constexpr float ROT_SCALE = 4.0f;        // treat the picture as bigger than the screen so twists read
// Motion echo: a repeating motion (swing, arc, pendulum…) is recorded and keeps going after you stop.
constexpr float ECHO_FADE_S = 9.0f;      // after a back-and-forth sway, it keeps swaying this long
constexpr float ECHO_DAMP_CUT = 0.955f;  // …with damping eased off by this much: slow, smooth die-out
constexpr float ECHO_SLOW_TAU = 1.5f;    // s: "average down" the gravity swing is measured from

// ---- Music ---------------------------------------------------------------------------------------
constexpr bool     MIC_DANCE = true;      // dance to music heard by the microphone (when switched on: swipe up)
constexpr float    DANCE_GAIN = 0.35f;    // beat nudge strength (×zone's own dance setting): subtle, additive
constexpr float    DANCE_ROOM = 0.5f;     // no kick while a zone is already this far out (of its travel)
constexpr float    DANCE_SWAY_ROOM = 3.0f;  // nor while hair / cloth / a pendant is this many px out
constexpr uint32_t MIC_SR = 16000;
constexpr size_t   MIC_N = 320;           // 20 ms chunks
constexpr float    MIC_LEAD = 0.03f;      // beat lead for audio (spring swing − recording latency)
constexpr bool     USE_DMA = true;        // set false to push strips without DMA

// ---- Rendering ---------------------------------------------------------------------------------
constexpr int STRIP_ROWS = 12;
constexpr size_t STRIP_PIX = (size_t)STRIP_ROWS * IMG_W;

// ---- Editor layout -----------------------------------------------------------------------------
constexpr int ROW_H = 30;                // each toolbar row
constexpr int TB1_N = 8;                 // Round · Half · Tear · Water · Depth · Sway · Grid · Cancel
constexpr int TB1_WATER = 3;
constexpr int TB1_DEPTH = 4;
constexpr int TB1_SWAY = 5;
constexpr int TB1_GRID = 6;
constexpr int TB2G_N = 5;                // grid: turn · turn · reset · on/off · the ground slides
constexpr int TB2S_N = 5;                // sway: Tool · Item · Kind · Swing · Del
constexpr int TB2D_N = 5;                // depth: Tool · Plane · Depth · Drift · Del
constexpr int TB2W_N = 5;                // water: Tool · Mode · Rain · Region · Del
constexpr int TB2_N = 4;                 // Rot · Feel · Tip · Del   (only while a zone is selected)
constexpr int HINT_H = 16;
constexpr int FOOT_H = 16;
constexpr int OUTLINE_PTS = 48;

M5Canvas g_img(&M5.Display);             // pristine image (PSRAM)
M5Canvas g_scratch(&M5.Display);         // editor frame
bool g_ok = false;
bool g_swapped = true;
uint16_t* g_src_fast = nullptr;          // copy of the image in internal RAM, if there's room
uint16_t* g_strip[2] = {nullptr, nullptr};
bool g_dma = false;
int32_t g_acc[2 * IMG_W];
int32_t g_acc1[2 * IMG_W];               // the second core's row accumulator
constexpr uint32_t LCD_WRITE_HZ = 40000000;   // display SPI write clock (raised to this if lower)

Zone g_z[MAX_ZONES];
bool g_shown[MAX_ZONES];                 // screen currently shows this zone displaced
Water    g_wat[WATER_MAX];               // water regions
int      g_nw = 0;
Sway*    g_sway = nullptr;               // hair, cloth, pendants (SWAY_MAX of them, in PSRAM: begin())
int      g_ns = 0;
uint8_t* g_body = nullptr;               // what they rest on (the Studio's "body" lines; else the depth figures)
bool     g_body_saved = false;           // (it came from the file: written back)
int16_t* g_sdf = nullptr;                // its signed distance (×8), for collisions
Parallax g_par;                          // depth planes (none = no parallax)
uint16_t* g_anim = nullptr;              // parallax with zones: the content with the zones drawn in
bool      g_anim_sw = true;
bool      g_par_moving = false;
uint16_t* g_front = nullptr;             // depth: what the screen shows (display order) — only changes get sent
constexpr float PAR_GX_S = -1.0f;        // gyro x → "top edge away" (flip if up/down depth runs backwards)
constexpr float PAR_GY_S = 1.0f;         // gyro y → "right edge away" (flip if left/right runs backwards)
int  g_n = 0;
char g_jig_path[320] = {0};

MotionEcho g_motion_echo;                // repeating motion (jolts, rocking, tipping, twisting) → echo
Rhythm   g_rhythm_mic;                   // beats from music (microphone) → dance
AudioOnset g_onset;
int16_t  g_mic_buf[MIC_N];
bool     g_mic_on = false, g_mic_busy = false;
uint32_t g_mic_at = 0;
float    g_mic_hit = 0;                  // latest chunk's hit strength…
float    g_mic_hold = 0;                 // …fed to the tracker for one chunk's worth of time (s)
bool     g_mic_new = false;              // a chunk was just analysed
float    g_dance = 0;                    // dance strength 0..1 (music)
bool     g_dance_on = false;             // swipe up switches dancing on / off
int      g_kick_sign = 1;
uint32_t g_last_kick = 0;
bool     g_was_music = false, g_was_echo = false;
float    g_acc_last[3] = {0, 0, 1};
float    g_gyro_last = 0;
float    g_gyro_xy[2] = {0, 0};          // deg/s about the IMU's x and y (parallax view)
float    g_grav[3] = {0, 0, 0};          // low-passed accelerometer = gravity (reads "up")
bool     g_init = false;
float    g_w = 0, g_alpha = 0;           // twist rate (rad/s, clockwise) and its change (rad/s²)
float    g_dn_slow[3] = {0, 1, 0};       // real down, slowly averaged (gravity swing = down − this)
float    g_echo = 0;                     // echo strength 0..1
float    g_sway_damp = 1.0f;             // hair / cloth / pendants: their damping while the echo plays
float    g_hold[MAX_ZONES][3];           // targets a resting zone is holding
uint32_t g_last_us = 0;
// Drawing never holds the loop for long: a frame that takes longer than this is finished over the next
// ticks (the gallery reads the touch screen and buttons in between, so taps and swipes always land).
constexpr uint32_t SLICE_US = 12000;
uint32_t g_tick_us0 = 0;
// The work is done in pieces (a water slice, a hair slice, a strip sent, the content step, the zones) whose
// cost is known from the last ones: a piece starts only if it still fits in this tick's budget, so pieces
// never pile up past it and every tick is about as long as the last. The first piece of a tick always
// runs: if one tick's physics ever takes the whole budget, drawing slows down - it never stops.
bool g_tick_did = false;                               // a piece ran this tick
struct Cost {                                          // a kind of piece's cost (µs): up at once, down slowly
    uint32_t us = 0;
    void note(uint32_t d) { us = d >= us ? d : us - ((us - d) >> 2); }
};
Cost g_c_wpush, g_c_swstart, g_c_swpush, g_c_content, g_c_frame, g_c_push, g_c_pushc, g_c_zones;
bool fits(uint32_t est) { return !g_tick_did || micros() - g_tick_us0 + est <= SLICE_US; }
bool fits(const Cost& c) { return fits(c.us); }
struct Piece {                                         // times one piece (scope) into its cost
    Cost& c; uint32_t t0;
    explicit Piece(Cost& c_) : c(c_), t0(micros()) {}
    ~Piece() { c.note(micros() - t0); g_tick_did = true; }
};
// Slices that vary in size (water, hair): cost per unit of work (µs per 1024 units; up at once, down slowly)
// times the units in the next one - the budget fills right up to its end, not to the costliest slice's.
struct Rate {
    uint32_t q = 0;
    uint32_t est(uint32_t units) const { return (uint32_t)(((uint64_t)q * units) >> 10); }
    void note(uint32_t d, uint32_t units) {
        if (!units) return;
        const uint32_t r = (uint32_t)(((uint64_t)d << 10) / units);
        q = r >= q ? r : q - ((q - r) >> 2);
    }
};
Rate g_r_water, g_r_sway;

// ---- Frame clock --------------------------------------------------------------------------------
// Frames start on a steady beat - 60, 30, 25 or 20 a second - not whenever the last one happens to finish:
// evenly spaced frames look smooth where a faster but uneven rate looks cheap. The beat is the fastest one
// the frames' measured work (start to on screen) fits with room to spare; it slows at once when they don't,
// and speeds up again only once they've fit the faster one for a while (so it never hunts).
constexpr uint32_t CLOCK_MS[] = {16, 33, 40, 50};
struct FrameClock {
    int lvl = 1;
    uint32_t next = 0, t0 = 0;
    bool open = false;                                 // a frame started and not yet on screen
    uint32_t work = 0;                                 // a frame's work (ms): up at once, down slowly
    uint32_t fit0 = 0;                                 // since when it has fit the faster beat (0: it doesn't)
    uint32_t period() const { return CLOCK_MS[lvl]; }
    bool due(uint32_t now) const { return !open && (int32_t)(now - next) >= 0; }
    void begin(uint32_t now) {
        next += period();
        if ((int32_t)(now - next) >= 0) next = now + period();   // (behind: on the beat from now - no catching up)
        t0 = now; open = true;
    }
    void end(uint32_t now) {
        if (!open) return;
        open = false;
        const uint32_t d = now - t0;
        work = d >= work ? d : work - ((work - d + 7) >> 3);
        const uint32_t need = work + work / 8 + 2;
        if (need > period() && lvl < 3) {              // doesn't fit: slower, now
            while (lvl < 3 && need > period()) ++lvl;
            fit0 = 0;
        } else if (lvl > 0 && need * 5 <= CLOCK_MS[lvl - 1] * 4) {   // fits the faster one with room …
            if (!fit0) fit0 = now | 1;
            else if (now - fit0 > 1500) { --lvl; fit0 = 0; }          // … for 1.5 s: faster
        } else fit0 = 0;
    }
    void cancel() { open = false; }                    // (started, but there was nothing to draw)
    void soon() { open = false; next = millis(); }     // (draw at the next chance)
};
FrameClock g_clock;
struct RPiece {
    Rate& r; uint32_t u, t0;
    RPiece(Rate& r_, uint32_t u_) : r(r_), u(u_), t0(micros()) {}
    ~RPiece() { r.note(micros() - t0, u); g_tick_did = true; }
};

// perf
uint32_t g_pf_t0 = 0, g_pf_frames = 0, g_pf_us = 0, g_pf_px = 0;

// ---- helpers -----------------------------------------------------------------------------------
uint8_t* alloc_psram(size_t n) {
    void* p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!p) p = malloc(n);
    return (uint8_t*)p;
}

void free_map(Zone& z) {
    if (z.wmap) free(z.wmap);            // wmap is the start of the zone's single block
    z.wmap = z.gmap = z.tmap = nullptr;
    z.span = nullptr;
}

void zone_off_water(Zone& z);
void rebuild(Zone& z) {
    free_map(z);
    size_t b = zone_mem_bytes(z);
    prepare_zone(z, b ? alloc_psram(b) : nullptr);
    zone_off_water(z);
}

int shape_from_name(const char* s) {
    for (int i = 0; i < SHAPE_COUNT; ++i)
        if (strcmp(s, shape_info(i).name) == 0) return i;
    return -1;
}

int material_from_name(const char* s) {
    for (int i = 0; i < MAT_COUNT; ++i)
        if (strcmp(s, material_info(i).name) == 0) return i;
    return -1;
}

float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

void* water_alloc(size_t n) { return alloc_psram(n); }
// Small, very hot scratch: internal RAM when there's comfortably room (the same margin the photo copy
// keeps), else PSRAM. Freed with free() either way.
void* alloc_fast(size_t n) {
#if defined(ESP_PLATFORM)
    if (heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT) > n + 48 * 1024) {
        void* p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (p) return p;
    }
#endif
    return alloc_psram(n);
}

void layer_sync();
void clear_waters() {
    for (int i = 0; i < g_nw; ++i) water_free(g_wat[i]);
    g_nw = 0;
    layer_sync();                                     // back to the plain photo
}

// A jiggle zone never drags water: the water is its own layer, animated by itself. Where a zone's
// falloff reaches into water its weight is cleared, so the zone moves only what it covers.
void zone_off_water(Zone& z) {
    if (!z.wmap) return;
    for (int i = 0; i < g_nw; ++i) {
        const Water& w = g_wat[i];
        if (!w.mask) continue;
        for (int y = 0; y < z.bh; ++y)
            for (int x = 0; x < z.bw; ++x)
                if (z.wmap[y * z.bw + x] && mask_get(w.mask, z.bx + x, z.by + y)) z.wmap[y * z.bw + x] = 0;
    }
}
// Each hair / cloth / pendant: how far (px) its strands lie from the nearest water that's drawn - water
// redrawn further away than it has swung (+ its soft edge) can't show through it (sway_due). A chamfer
// distance over the frame, in PSRAM for the moment; without the memory, 0 (always: as close as can be).
void sway_wet_dist() {
    for (int i = 0; i < g_ns; ++i) g_sway[i].wetd = 0;
    if (g_ns == 0) return;
    bool any = false;
    for (int w = 0; w < g_nw; ++w) any |= g_wat[w].edge && g_wat[w].mask;
    if (!any) return;
    uint16_t* D = (uint16_t*)alloc_psram((size_t)IMG_W * IMG_H * 2);
    if (!D) return;
    for (int k = 0; k < IMG_W * IMG_H; ++k) {
        bool wet = false;
        for (int w = 0; w < g_nw && !wet; ++w) wet = g_wat[w].edge && g_wat[w].mask && ((g_wat[w].mask[k >> 3] >> (k & 7)) & 1);
        D[k] = wet ? 0 : 60000;
    }
    auto lo = [](uint16_t& d, int v) { if (v < d) d = (uint16_t)v; };   // (3 a step across, 4 diagonally)
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x) {
            uint16_t& d = D[y * IMG_W + x];
            if (x) lo(d, D[y * IMG_W + x - 1] + 3);
            if (y) {
                lo(d, D[(y - 1) * IMG_W + x] + 3);
                if (x) lo(d, D[(y - 1) * IMG_W + x - 1] + 4);
                if (x < IMG_W - 1) lo(d, D[(y - 1) * IMG_W + x + 1] + 4);
            }
        }
    for (int y = IMG_H - 1; y >= 0; --y)
        for (int x = IMG_W - 1; x >= 0; --x) {
            uint16_t& d = D[y * IMG_W + x];
            if (x < IMG_W - 1) lo(d, D[y * IMG_W + x + 1] + 3);
            if (y < IMG_H - 1) {
                lo(d, D[(y + 1) * IMG_W + x] + 3);
                if (x < IMG_W - 1) lo(d, D[(y + 1) * IMG_W + x + 1] + 4);
                if (x) lo(d, D[(y + 1) * IMG_W + x - 1] + 4);
            }
        }
    for (int i = 0; i < g_ns; ++i) {
        Sway& h = g_sway[i];
        if (!h.ready || !h.own) continue;              // (0: as close as can be)
        int m = 60000;
        for (int y = 0; y < h.rh; ++y)
            for (int x = 0; x < h.rw; ++x) {
                const int sx = h.rx0 + x, sy = h.ry0 + y;
                if (h.own[y * h.rw + x] >= 0 && sx >= 0 && sy >= 0 && sx < IMG_W && sy < IMG_H && D[sy * IMG_W + sx] < m) m = D[sy * IMG_W + sx];
            }
        h.wetd = m / 3.0f;
    }
    free(D);
}
void prepare_waters() {
    for (int i = 0; i < g_nw; ++i) water_prepare(g_wat[i], water_alloc);
    for (int i = 0; i < g_n; ++i) zone_off_water(g_z[i]);
    layer_sync();
    sway_wet_dist();
}
void clear_sways() {
    for (int i = 0; i < g_ns; ++i) sway_free(g_sway[i]);
    g_ns = 0;
    if (g_body) { free(g_body); g_body = nullptr; }
    if (g_sdf) { free(g_sdf); g_sdf = nullptr; }
    g_body_saved = false;
}
// Build hair / cloth / pendants: what they rest on (the body less them, plus their parts that lie on
// it - mg_sway.rest_body), its distance field, then each one. (Before prepare_waters: that builds the layer.)
void prepare_sways() {
    if (g_sdf) { free(g_sdf); g_sdf = nullptr; }
    for (int i = 0; i < g_ns; ++i) sway_free_runtime(g_sway[i]);
    if (g_ns == 0) return;
    for (int i = 0; i < g_ns; ++i) sway_default_guides(g_sway[i]);
    uint8_t* body = (uint8_t*)water_alloc(MASK_BYTES);
    int16_t* nx = (int16_t*)water_alloc((size_t)IMG_W * IMG_H * 2);
    int16_t* ny = (int16_t*)water_alloc((size_t)IMG_W * IMG_H * 2);
    uint8_t* forbid = (uint8_t*)water_alloc(MASK_BYTES);
    bool anyb = false;
    if (body) {
        memset(body, 0, MASK_BYTES);
        if (g_body) memcpy(body, g_body, MASK_BYTES);
        else for (int k = 0; k < g_par.n; ++k)          // (no body lines: the depth figures)
            if (g_par.pl[k].mask) for (int b = 0; b < MASK_BYTES; ++b) body[b] |= g_par.pl[k].mask[b];
        for (int b = 0; b < MASK_BYTES && !anyb; ++b) anyb = body[b] != 0;
    }
    if (anyb && nx && ny) {
        const uint8_t* ms[SWAY_MAX];
        for (int i = 0; i < g_ns; ++i) ms[i] = g_sway[i].mask;
        sway_rest_body(body, ms, g_ns, nx, ny);
        g_sdf = sway_body_sdf(body, water_alloc);
    }
    if (nx) free(nx);
    if (ny) free(ny);
    const uint16_t* photo = (const uint16_t*)g_img.getBuffer();
    for (int i = 0; i < g_ns; ++i) {
        if (forbid) {                                   // (never filled from one another)
            memset(forbid, 0, MASK_BYTES);
            for (int j = 0; j < g_ns; ++j) if (j != i && g_sway[j].mask) for (int b = 0; b < MASK_BYTES; ++b) forbid[b] |= g_sway[j].mask[b];
        }
        const bool ok = sway_prepare(g_sway[i], photo, g_swapped, anyb ? body : nullptr, g_sdf, forbid, water_alloc);
        Serial.printf("jig: %s %d %s\n", sway_kind_info(g_sway[i].kind).label, i + 1, ok ? "ready" : "not ready (no memory, or nothing selected)");
    }
    if (forbid) free(forbid);
    if (body) free(body);
}
bool any_sway_def() {
    for (int i = 0; i < g_ns; ++i) if (sway_count(g_sway[i]) >= 12) return true;
    return false;
}

int water_mode_from_name(const char* s) {
    for (int m = 0; m < W_MODE_COUNT; ++m) if (strcmp(s, water_mode_name((uint8_t)m)) == 0) return m;
    return -1;
}

// One .jig line about water: "water <mode> k=v…", "wline x,y x,y …", "wrow y a-b c-d …".
// `cur` is the region the wline/wrow lines belong to (-1 = none / one too many). True if handled.
bool parse_water_line(char* line, Water* ws, int& nw, int& cur) {
    char kind[8];
    int used = 0;
    if (sscanf(line, "%7s%n", kind, &used) != 1) return false;
    char* rest = line + used;
    char tok[40];
    int n = 0;
    if (strcmp(kind, "water") == 0) {
        cur = -1;
        if (nw >= WATER_MAX) return true;
        Water& w = ws[nw];
        water_free(w);
        w = Water();
        if (!water_ensure_mask(w, water_alloc)) return true;
        if (sscanf(rest, "%39s%n", tok, &n) == 1) {
            int m = water_mode_from_name(tok);
            if (m >= 0) { w.mode = (uint8_t)m; rest += n; }
        }
        while (sscanf(rest, "%39s%n", tok, &n) == 1) {
            rest += n;
            char* eq = strchr(tok, '=');
            if (!eq) continue;
            *eq = 0;
            if (strcmp(tok, "name") == 0) { size_t L = strlen(eq + 1); if (L > sizeof(w.name) - 1) L = sizeof(w.name) - 1; memcpy(w.name, eq + 1, L); w.name[L] = 0; continue; }
            for (int p = 0; p < WP_COUNT; ++p)
                if (strcmp(tok, wparam_info((uint8_t)p).key) == 0) {
                    char* end = nullptr;
                    float v = strtof(eq + 1, &end);
                    const WParamInfo& pi = wparam_info((uint8_t)p);
                    if (end != eq + 1 && !isnan(v)) w.prm[p] = v < pi.lo ? pi.lo : (v > pi.hi ? pi.hi : v);
                }
        }
        cur = nw++;
        return true;
    }
    if (strcmp(kind, "wline") == 0) {
        if (cur < 0) return true;
        Water& w = ws[cur];
        if (w.nlines >= WLINE_MAX) return true;
        WLine& L = w.line[w.nlines];
        L.n = 0;
        int x, y;
        while (L.n < WPTS_MAX && sscanf(rest, " %d,%d%n", &x, &y, &n) == 2) {
            rest += n;
            L.x[L.n] = (int16_t)clampf((float)x, 0, IMG_W - 1);
            L.y[L.n] = (int16_t)clampf((float)y, 0, IMG_H - 1);
            ++L.n;
        }
        if (L.n) ++w.nlines;
        return true;
    }
    if (strcmp(kind, "wgrid") == 0) {                 // perspective grid: 4 corners
        if (cur < 0) return true;
        Water& w = ws[cur];
        float q[8];
        int got = 0;
        for (int k = 0; k < 4; ++k) {
            float x, y;
            if (sscanf(rest, " %f,%f%n", &x, &y, &n) != 2) break;
            rest += n; q[2 * k] = x; q[2 * k + 1] = y; ++got;
        }
        if (got == 4) { w.grid = true; memcpy(w.gq, q, sizeof(q)); }
        return true;
    }
    if (strcmp(kind, "wrow") == 0) {
        if (cur < 0) return true;
        int y;
        if (sscanf(rest, "%d%n", &y, &n) != 1) return true;
        rest += n;
        int a, b;
        while (sscanf(rest, " %d-%d%n", &a, &b, &n) == 2) {
            rest += n;
            mask_span(ws[cur].mask, y, a, b, true);
        }
        return true;
    }
    return false;
}

template <typename F>
void save_waters(F& f, const Water* ws, int nw) {
    for (int i = 0; i < nw; ++i) {
        const Water& w = ws[i];
        if (!w.mask) continue;
        bool any = false;
        for (int k = 0; k < MASK_BYTES && !any; ++k) any = w.mask[k] != 0;
        if (!any) continue;
        f.printf("water %s", water_mode_name(w.mode));
        for (int p = 0; p < WP_COUNT; ++p)
            if (fabsf(w.prm[p] - wparam_info((uint8_t)p).dflt) > 1e-6f) f.printf(" %s=%.3g", wparam_info((uint8_t)p).key, w.prm[p]);
        if (w.name[0]) f.printf(" name=%s", w.name);
        f.printf("\n");
        for (int l = 0; l < w.nlines; ++l) {
            if (!w.line[l].n) continue;
            f.printf("wline");
            for (int k = 0; k < w.line[l].n; ++k) f.printf(" %d,%d", w.line[l].x[k], w.line[l].y[k]);
            f.printf("\n");
        }
        if (w.grid) f.printf("wgrid %.1f,%.1f %.1f,%.1f %.1f,%.1f %.1f,%.1f\n", w.gq[0], w.gq[1], w.gq[2], w.gq[3], w.gq[4], w.gq[5], w.gq[6], w.gq[7]);
        for (int y = 0; y < IMG_H; ++y) {
            int x = 0;
            bool row = false;
            while (x < IMG_W) {
                while (x < IMG_W && !mask_get(w.mask, x, y)) ++x;
                if (x >= IMG_W) break;
                int a = x;
                while (x < IMG_W && mask_get(w.mask, x, y)) ++x;
                if (!row) { f.printf("wrow %d", y); row = true; }
                f.printf(" %d-%d", a, x);
            }
            if (row) f.printf("\n");
        }
    }
}

// With water, the zones warp a composited layer (the photo with the water drawn in) instead of the
// raw photo — so effects stack: a figure jiggling in water shows the water moving inside its warp.
// The layer reuses the internal-RAM copy of the photo when there is one (else the editor's canvas),
// so it costs no extra memory.
uint16_t* g_layer = nullptr;
bool      g_layer_sw = false;
uint16_t* g_photo_n = nullptr;                      // the photo in native byte order, for the water to sample (while any)

const uint16_t* render_src() {
    if (g_layer) return g_layer;
    return g_src_fast ? g_src_fast : (const uint16_t*)g_img.getBuffer();
}
bool render_src_swapped() {
    if (g_layer) return g_layer_sw;
    return g_src_fast ? true : g_swapped;
}

// (Re)build the layer from the photo; called whenever the water or the photo changes.
void layer_sync() {
    const uint16_t* photo = (const uint16_t*)g_img.getBuffer();
    const size_t n = (size_t)IMG_W * IMG_H;
    if (g_src_fast) {                                   // restore the fast copy (it may hold old water)
        if (g_swapped) memcpy(g_src_fast, photo, n * 2);
        else for (size_t i = 0; i < n; ++i) g_src_fast[i] = bswap16(photo[i]);
    }
    bool wet = false;                                   // (the water samples the photo: native order, no swap per read)
    for (int i = 0; i < g_nw; ++i) wet |= g_wat[i].edge != nullptr;
    if (wet && !g_photo_n) g_photo_n = (uint16_t*)alloc_psram(n * 2);
    if (!wet && g_photo_n) { free(g_photo_n); g_photo_n = nullptr; }
    if (g_photo_n) {
        if (g_swapped) for (size_t i = 0; i < n; ++i) g_photo_n[i] = bswap16(photo[i]);
        else memcpy(g_photo_n, photo, n * 2);
    }
    bool live = wet;
    for (int i = 0; i < g_ns; ++i) live |= g_sway[i].ready;          // (hair / cloth / pendants draw into it too)
    if (!live) { g_layer = nullptr; return; }
    if (g_src_fast) { g_layer = g_src_fast; g_layer_sw = true; return; }
    g_layer = (uint16_t*)g_scratch.getBuffer();
    g_layer_sw = g_swapped;
    if (g_layer) memcpy(g_layer, photo, n * 2);
}

bool any_water(const Water* ws, int nw) {
    for (int i = 0; i < nw; ++i)
        if (ws[i].mask) for (int k = 0; k < MASK_BYTES; ++k) if (ws[i].mask[k]) return true;
    return false;
}

bool any_plane(const Parallax& P);
// Lines this firmware doesn't play yet (the Studio's swaying hair / cloth, anything newer) are kept as
// they were and written back on save, so editing on the Core2 never loses them.
char*  g_keep = nullptr;
size_t g_keep_len = 0, g_keep_cap = 0;
void keep_line(const char* ln) {
    const size_t n = strlen(ln);
    if (g_keep_len + n + 2 > g_keep_cap) {
        size_t cap = g_keep_cap * 2 > g_keep_len + n + 4096 ? g_keep_cap * 2 : g_keep_len + n + 4096;
        char* p = (char*)alloc_psram(cap);
        if (!p) return;
        if (g_keep) { memcpy(p, g_keep, g_keep_len); free(g_keep); }
        g_keep = p; g_keep_cap = cap;
    }
    memcpy(g_keep + g_keep_len, ln, n);
    g_keep_len += n;
    g_keep[g_keep_len++] = '\n';
}

void save_zones() {
    if (!g_jig_path[0]) return;
    if (g_n == 0 && !any_water(g_wat, g_nw) && !any_plane(g_par) && !any_sway_def() && g_keep_len == 0) {
        if (SD.exists(g_jig_path)) SD.remove(g_jig_path);
        return;
    }
    File f = SD.open(g_jig_path, FILE_WRITE);
    if (!f) { Serial.printf("jig: can't write %s\n", g_jig_path); return; }
    f.println("MGJIG1");
    for (int i = 0; i < g_n; ++i) {
        const Zone& z = g_z[i];
        f.printf("%s %d %d %d %d %s", shape_info(z.shape).name, (int)lroundf(z.cx), (int)lroundf(z.cy),
                 (int)lroundf(z.r), (int)z.rot, material_info(z.material).name);
        if (tip_used(z)) f.printf(" tip %d %d", (int)lroundf(z.tip_u * 100), (int)lroundf(z.tip_w * 100));
        if (z.name[0]) f.printf(" name=%s", z.name);
        for (int t = 0; t < TUNE_COUNT; ++t)
            if (tune_set(z, (uint8_t)t)) f.printf(" %s=%.3g", tune_info((uint8_t)t).key, z.tune[t]);
        f.printf("\n");
    }
    save_waters(f, g_wat, g_nw);                     // after the zones: older firmware reads those and stops
    save_parallax(f, g_par);
    save_sways(f, g_sway, g_ns, g_body_saved ? g_body : nullptr);
    if (g_keep_len) f.write((const uint8_t*)g_keep, g_keep_len);
    f.close();
    Serial.printf("jig: saved %d zone(s), %d water -> %s\n", g_n, g_nw, g_jig_path);
}

// Parse one zone line. Returns false if it isn't a zone.
bool parse_zone(char* line, Zone& z) {
    char name[12];
    int cx, cy, r, rot, used = 0;
    if (sscanf(line, "%11s %d %d %d %d%n", name, &cx, &cy, &r, &rot, &used) != 5) return false;
    int shape = shape_from_name(name);
    if (shape < 0) return false;
    z = Zone();
    z.shape = (uint8_t)shape;
    z.cx = clampf((float)cx, 0, IMG_W);
    z.cy = clampf((float)cy, 0, IMG_H);
    z.r = clampf((float)r, R_MIN, R_MAX);
    z.rot = (int16_t)(((rot % 360) + 360) % 360);
    char* rest = line + used;
    char tok[24];
    int n = 0;
    while (sscanf(rest, "%23s%n", tok, &n) == 1) {
        rest += n;
        int m = material_from_name(tok);
        if (m >= 0) { z.material = (uint8_t)m; continue; }
        if (char* eq = strchr(tok, '=')) {        // advanced tuning from the PC editor
            *eq = 0;
            if (strcmp(tok, "name") == 0) { size_t L = strlen(eq + 1); if (L > sizeof(z.name) - 1) L = sizeof(z.name) - 1; memcpy(z.name, eq + 1, L); z.name[L] = 0; continue; }
            for (int t = 0; t < TUNE_COUNT; ++t)
                if (strcmp(tok, tune_info((uint8_t)t).key) == 0) {
                    char* end = nullptr;
                    float v = strtof(eq + 1, &end);
                    if (end != eq + 1 && !isnan(v)) {
                        const TuneInfo& ti = tune_info((uint8_t)t);
                        z.tune[t] = v < ti.lo ? ti.lo : (v > ti.hi ? ti.hi : v);
                    }
                }
            continue;
        }
        if (strcmp(tok, "tip") == 0) {
            int tu = 0, tw = 0, k = 0;
            if (sscanf(rest, "%d %d%n", &tu, &tw, &k) == 2) {
                rest += k;
                z.tip = true;
                z.tip_u = clampf(tu / 100.0f, -0.6f, 0.6f);
                z.tip_w = clampf(tw / 100.0f, -1.0f, 0.9f);
            }
        }
    }
    return true;
}

// ---- Rendering ---------------------------------------------------------------------------------
void push_strip(int x, int y, int w, int h, uint16_t* buf, bool& inflight) {
    if (USE_DMA && g_dma) {
        if (inflight) M5.Display.waitDMA();
        M5.Display.pushImageDMA(x, y, w, h, buf);     // display byte order → straight to the bus
        inflight = true;
    } else {
        M5.Display.pushImage(x, y, w, h, buf);
    }
}

// Draw every zone that moved (or must return to rest). Each overlapping group renders once.
// ---- Two cores --------------------------------------------------------------------------------
// The gallery loop runs on core 1; core 0 is mostly idle. A worker pinned to core 0 renders half the
// rows of each water area and each jiggle strip while the loop renders the other half (rows are
// independent: separate scratch per core). PC builds run both halves in turn.
using RowJob = void (*)(void* ctx, int lane, int y0, int y1);
#if defined(ESP_PLATFORM)
SemaphoreHandle_t g_go = nullptr, g_done = nullptr;
struct { RowJob fn; void* ctx; int y0, y1; } volatile g_job;
void row_worker(void*) {
    for (;;) {
        xSemaphoreTake(g_go, portMAX_DELAY);
        g_job.fn(g_job.ctx, 1, g_job.y0, g_job.y1);
        xSemaphoreGive(g_done);
    }
}
void par_start() {                                   // (its stack: the depth render takes ~2.6 KB of it - room to spare)
    if (g_go) return;
    g_go = xSemaphoreCreateBinary();
    g_done = xSemaphoreCreateBinary();
    if (!g_go || !g_done || xTaskCreatePinnedToCore(row_worker, "jig-rows", 6144, nullptr, 3, nullptr, 0) != pdPASS) {
        g_go = nullptr;                                 // run everything on this core
        Serial.printf("jig: second core unavailable, rendering on one\n");
    }
}
// ym: where the second core's rows start (the two halves' work about equal).
void par_rows_at(RowJob fn, void* ctx, int y0, int ym, int y1) {
    if (!g_go || y1 - y0 < 2 || ym <= y0 || ym >= y1) { fn(ctx, 0, y0, y1); return; }
    g_job.fn = fn; g_job.ctx = ctx; g_job.y0 = ym; g_job.y1 = y1;
    xSemaphoreGive(g_go);
    fn(ctx, 0, y0, ym);
    xSemaphoreTake(g_done, portMAX_DELAY);
}
#else
void par_start() {}
void par_rows_at(RowJob fn, void* ctx, int y0, int ym, int y1) {
    fn(ctx, 0, y0, ym);
    fn(ctx, 1, ym, y1);
}
#endif
void par_rows(RowJob fn, void* ctx, int y0, int y1) { par_rows_at(fn, ctx, y0, (y0 + y1) / 2, y1); }

struct StripJob { const uint16_t* src; bool sw; uint16_t* out; const int* members; int nm, xa, xb, ys; };
void strip_rows(void* c, int lane, int y0, int y1) {
    StripJob& j = *(StripJob*)c;
    render_rect(j.src, j.sw, j.out + (y0 - j.ys) * (j.xb - j.xa), g_z, j.members, j.nm, j.xa, y0, j.xb, y1, lane ? g_acc1 : g_acc);
}
struct WaterJob { const Water* w; const WFrame* fr; const uint16_t* photo; bool psw; };
void water_rows(void* c, int lane, int y0, int y1) {
    WaterJob& j = *(WaterJob*)c;
    water_render(*j.w, *j.fr, j.photo, j.psw, g_layer, g_layer_sw, 0, y0, IMG_W, y1, lane);
}
void water_node_rows_job(void* c, int lane, int j0, int j1) {   // (rows here: node rows)
    WaterJob& j = *(WaterJob*)c;
    water_node_rows(*j.w, *j.fr, water_row_scratch(lane), j0, j1);
}

struct SwayJob { const Sway* s; const SwDraw* d; };
void sway_rows(void* c, int lane, int y0, int y1) {
    SwayJob& j = *(SwayJob*)c;
    sway_draw_rows(*j.s, *j.d, g_layer, g_layer_sw, y0, y1);
}

// Content rects for parallax: where the content changed this frame (picture coordinates).
struct DirtyRects { int r[24][4]; int n = 0;
    void add(int x0, int y0, int x1, int y1) {
        if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0; if (x1 > IMG_W) x1 = IMG_W; if (y1 > IMG_H) y1 = IMG_H;
        if (x0 >= x1 || y0 >= y1) return;
        for (int i = 0; i < n; ++i)                    // merge with one it overlaps / nearly touches
            if (!(x0 > r[i][2] + 8 || r[i][0] > x1 + 8 || y0 > r[i][3] + 8 || r[i][1] > y1 + 8)) {
                int a = r[i][0] < x0 ? r[i][0] : x0, b = r[i][1] < y0 ? r[i][1] : y0;
                int c = r[i][2] > x1 ? r[i][2] : x1, d = r[i][3] > y1 ? r[i][3] : y1;
                r[i][0] = r[--n][0]; r[i][1] = r[n][1]; r[i][2] = r[n][2]; r[i][3] = r[n][3];
                add(a, b, c, d);
                return;
            }
        if (n < 24) { r[n][0] = x0; r[n][1] = y0; r[n][2] = x1; r[n][3] = y1; ++n; }
        else { int a = r[0][0], b = r[0][1], c = r[0][2], d = r[0][3];   // (full: fold into the first)
               r[0][0] = a < x0 ? a : x0; r[0][1] = b < y0 ? b : y0; r[0][2] = c > x1 ? c : x1; r[0][3] = d > y1 ? d : y1; }
    }
};

// Draw every zone that moved (or must return to rest) — to the screen, or (parallax) into the content
// buffer g_anim, noting where in `dr`.
void draw_frame(DirtyRects* dr = nullptr) {
    const bool to_anim = dr != nullptr;
    stage_frame(g_z, g_n);
    int ids[MAX_ZONES], nd = 0;
    for (int i = 0; i < g_n; ++i)
        if (g_z[i].wmap && (g_z[i].awake || g_shown[i])) ids[nd++] = i;
    if (!nd) return;
    // A zone resting in a held (tilted) pose still offsets pixels it shares with a moving zone.
    for (int i = 0, n0 = nd; i < g_n; ++i) {
        const Zone& A = g_z[i];
        bool in = false;
        for (int k = 0; k < nd; ++k) in |= ids[k] == i;
        if (in || !A.wmap || !zone_displaced(A)) continue;
        for (int k = 0; k < n0; ++k) {
            const Zone& B = g_z[ids[k]];
            if (!(A.bx >= B.bx + B.bw || B.bx >= A.bx + A.bw || A.by >= B.by + B.bh || B.by >= A.by + A.bh)) { ids[nd++] = i; break; }
        }
    }

    // Group overlapping zones (tiny n: simple merge).
    int grp[MAX_ZONES];
    for (int i = 0; i < nd; ++i) grp[i] = i;
    for (int a = 0; a < nd; ++a)
        for (int b = a + 1; b < nd; ++b) {
            const Zone& A = g_z[ids[a]];
            const Zone& B = g_z[ids[b]];
            bool ov = !(A.bx >= B.bx + B.bw || B.bx >= A.bx + A.bw || A.by >= B.by + B.bh || B.by >= A.by + A.bh);
            if (ov) { int ga = grp[a], gb = grp[b]; for (int k = 0; k < nd; ++k) if (grp[k] == gb) grp[k] = ga; }
        }

    uint32_t t0 = micros();
    uint32_t px = 0;
    const uint16_t* src = render_src();
    bool src_swapped = render_src_swapped();
    int cur = 0;
    bool inflight = false;
    if (!to_anim) M5.Display.startWrite();
    for (int g = 0; g < nd; ++g) {
        int members[MAX_ZONES], nm = 0, y0 = IMG_H, y1 = 0;
        bool first = true;
        for (int k = 0; k < nd; ++k) {
            if (grp[k] != g) continue;
            members[nm++] = ids[k];
            const Zone& z = g_z[ids[k]];
            if (z.by < y0) y0 = z.by;
            if (z.by + z.bh > y1) y1 = z.by + z.bh;
            first = false;
        }
        if (first) continue;
        for (int y = y0; y < y1; y += STRIP_ROWS) {
            int ye = y + STRIP_ROWS < y1 ? y + STRIP_ROWS : y1;
            int xa, xb;
            if (!strip_extent(g_z, members, nm, y, ye, xa, xb)) continue;
            StripJob job{src, src_swapped, g_strip[cur], members, nm, xa, xb, y};
            par_rows(strip_rows, &job, y, ye);           // both cores, half the rows each
            if (to_anim) {                               // into the content (display order → its order)
                const int w = xb - xa;
                for (int yy = y; yy < ye; ++yy) {
                    const uint16_t* sr = g_strip[cur] + (yy - y) * w;
                    uint16_t* dst = g_anim + yy * IMG_W + xa;
                    if (g_anim_sw) memcpy(dst, sr, (size_t)w * 2);
                    else for (int x = 0; x < w; ++x) dst[x] = bswap16(sr[x]);
                }
                dr->add(xa, y, xb, ye);
                continue;
            }
            push_strip(xa, y, xb - xa, ye - y, g_strip[cur], inflight);
            px += (uint32_t)(xb - xa) * (ye - y);
            cur ^= 1;
        }
    }
    if (to_anim) { for (int k = 0; k < nd; ++k) g_shown[ids[k]] = false; return; }
    if (inflight) M5.Display.waitDMA();
    M5.Display.endWrite();

    for (int k = 0; k < nd; ++k) g_shown[ids[k]] = false;

    uint32_t now = millis();
    g_pf_us += micros() - t0;
    g_pf_px += px;
    ++g_pf_frames;
    if (now - g_pf_t0 >= 3000) {
        if (g_pf_frames)
            Serial.printf("jig: %.1f fps, frame %.1f ms, %lu px/frame (%s, src %s)\n",
                          g_pf_frames * 1000.0f / (now - g_pf_t0), g_pf_us / 1000.0f / g_pf_frames,
                          (unsigned long)(g_pf_px / g_pf_frames), g_dma ? "DMA" : "no DMA",
                          g_src_fast ? "SRAM" : "PSRAM");
        g_pf_t0 = now; g_pf_frames = g_pf_us = g_pf_px = 0;
    }
}

// The water frame in progress: 1 = rendering into the layer, 2 = hair / cloth / pendants over it, 3 = sending.
struct WaterJobState { uint8_t phase = 0; int wi = 0, y = -1, nd = 0; int which[8]; bool par = false, sw = false; WFrame wf[WATER_MAX]; };
WaterJobState g_wj;
void water_job_begin(bool par) {                       // (each water as it stands now: the whole frame shows this moment)
    g_wj = WaterJobState(); g_wj.phase = 1; g_wj.par = par;
    for (int i = 0; i < g_nw; ++i) {
        const Water& w = g_wat[i];
        if (!w.edge) continue;
        water_frame_setup(w, g_wj.wf[i]);
        water_snapshot(w, w.sgx, w.sgy);
        g_wj.wf[i].sgx = w.sgx; g_wj.wf[i].sgy = w.sgy;
    }
}
bool sway_redraw_run(bool budget);
void sway_redraw_start(bool water_moved);
void sway_push(const int* which, int nd);
int g_sj_nd();
int g_sj_which(int k);
// A water slice's work: its wet pixels (+ some per row). ym: where to split it between the two cores so
// each gets about half; returns the larger half.
uint32_t water_units(const Water& w, int y0, int y1, int& ym) {
    uint32_t n[STRIP_ROWS], tot = 0;
    for (int y = y0; y < y1; ++y) {
        uint32_t u = 16;
        if (!w.sp_off) u += w.bw;
        else for (int k = w.sp_off[y - w.by]; k < w.sp_off[y - w.by + 1]; ++k) u += w.sp[2 * k + 1] - w.sp[2 * k];
        n[y - y0] = u; tot += u;
    }
    uint32_t a = 0, best = tot;
    ym = y0;
    for (int y = y0; y < y1; ++y) {                    // (the split with the smaller larger half)
        const uint32_t big = a > tot - a ? a : tot - a;
        if (big < best) { best = big; ym = y; }
        a += n[y - y0];
    }
    return best;
}
// Advance it until done (true) or this tick's time is up (false).
bool water_job() {
    const uint16_t* photo = g_photo_n ? g_photo_n : (const uint16_t*)g_img.getBuffer();
    const bool psw = g_photo_n ? false : g_swapped;
    while (g_wj.phase == 1) {
        if (g_wj.wi >= g_nw) { g_wj.phase = 2; break; }
        const Water& w = g_wat[g_wj.wi];
        if (g_wj.y < 0) g_wj.y = w.by;
        if (!w.edge || g_wj.y >= w.by + w.bh) { ++g_wj.wi; g_wj.y = -1; continue; }
        const int ye = g_wj.y + STRIP_ROWS < w.by + w.bh ? g_wj.y + STRIP_ROWS : w.by + w.bh;
        int ym;
        const uint32_t units = water_units(w, g_wj.y, ye, ym);
        if (!fits(g_r_water.est(units))) return false;
        RPiece pc(g_r_water, units);
        WaterJob job{&w, &g_wj.wf[g_wj.wi], photo, psw};
        int j0, j1;                                    // its new node rows (the ripple field at half resolution) …
        water_nodes_due(w, g_wj.wf[g_wj.wi], g_wj.y, ye, j0, j1);
        if (j0 < j1) par_rows(water_node_rows_job, &job, j0, j1);
        par_rows_at(water_rows, &job, g_wj.y, ym, ye);   // … then its pixels: both cores, about half the wet pixels each
        g_wj.y = ye;
    }
    if (g_wj.phase == 2 && g_wj.par) { g_wj.phase = 0; return true; }   // (depth: the rest is its content step)
    if (g_wj.phase == 2) {                             // hair / cloth / pendants over the new water (into the layer)
        if (g_ns > 0 && !g_wj.sw) {
            if (!fits(g_c_swstart)) return false;
            { Piece pc(g_c_swstart); sway_redraw_start(true); }
            g_wj.sw = true;
        }
        if (g_wj.sw && !sway_redraw_run(true)) return false;
        g_wj.nd = g_wj.sw ? g_sj_nd() : 0;
        for (int k = 0; k < g_wj.nd; ++k) g_wj.which[k] = g_sj_which(k);
        g_wj.phase = 3; g_wj.wi = 0; g_wj.y = -1;
    }
    // Sending: the last strip is followed at once by the hair over it and the zones on top (a zone left out
    // for a tick would flash at rest), so every strip leaves room for those.
    const uint32_t tail = (g_wj.nd ? g_c_swpush.us : 0) + (g_n > 0 ? g_c_zones.us : 0);
    int cur = 0;
    bool inflight = false;
    M5.Display.startWrite();
    while (g_wj.phase == 3) {
        if (g_wj.wi >= g_nw) { g_wj.phase = 4; break; }
        const Water& w = g_wat[g_wj.wi];
        if (g_wj.y < 0) g_wj.y = w.by;
        if (!w.edge || g_wj.y >= w.by + w.bh) { ++g_wj.wi; g_wj.y = -1; continue; }
        if (!fits(g_c_wpush.us + tail)) break;
        Piece pc(g_c_wpush);
        const int y = g_wj.y, ye = y + STRIP_ROWS < w.by + w.bh ? y + STRIP_ROWS : w.by + w.bh;
        g_wj.y = ye;
        int xa, xb;
        water_rows_extent(w, y, ye, xa, xb);           // just the columns that hold water
        if (xa >= xb) continue;
        uint16_t* out = g_strip[cur];
        if (inflight && USE_DMA && g_dma) { M5.Display.waitDMA(); inflight = false; }
        const int ow = xb - xa;
        for (int yy = y; yy < ye; ++yy) {
            const uint16_t* row = g_layer + yy * IMG_W + xa;
            uint16_t* o = out + (yy - y) * ow;
            if (g_layer_sw) memcpy(o, row, (size_t)ow * 2);
            else for (int x = 0; x < ow; ++x) o[x] = bswap16(row[x]);
        }
        push_strip(xa, y, ow, ye - y, out, inflight);
        cur ^= 1;
    }
    if (inflight) M5.Display.waitDMA();
    M5.Display.endWrite();
    if (g_wj.phase != 4) return false;
    if (g_wj.nd) { Piece pc(g_c_swpush); sway_push(g_wj.which, g_wj.nd); }
    g_wj.phase = 0;
    return true;
}

// ---- hair, cloth, pendants ---------------------------------------------------------------------------
bool regions_meet(int ax0, int ay0, int ax1, int ay1, int bx0, int by0, int bx1, int by1) {
    return !(ax0 >= bx1 || bx0 >= ax1 || ay0 >= by1 || by0 >= ay1);
}
// Which to redraw this frame: moving, settling (one last frame at rest), or over water that moved.
// How far (px) its pose is from the one on screen, and that one noted as it starts drawing.
float sway_moved(const Sway& h) {
    if (h.kind == SK_PENDANT) return fabsf(h.th - h.drth) * h.tlen;
    float m = 0;
    for (int k = 0; k < h.nr; ++k)
        for (int j = 0; j < h.nn[k]; ++j) {
            const float d = fmaxf(fabsf(h.x[k][j] - h.drx[k][j]), fabsf(h.y[k][j] - h.dry[k][j]));
            if (d > m) m = d;
        }
    return m;
}
void sway_mark(Sway& h) { memcpy(h.drx, h.x, sizeof(h.x)); memcpy(h.dry, h.y, sizeof(h.y)); h.drth = h.th; }
// Due a redraw: new, poked, kicked or put back (not drawn); moved 1/4 px from the pose on screen - finer
// steps don't show, and a breeze moving it a fiftieth of a pixel a frame would otherwise redraw all of it
// every frame; come to rest (once more, exactly); or water redrawn within its strands' reach (the water
// under a strand's soft edge shows through it).
bool sway_due(int i, bool water_moved) {
    const Sway& h = g_sway[i];
    if (!h.ready) return false;
    if (!h.drawn) return true;
    const float m = sway_moved(h);
    if (m >= 0.25f || (!h.awake && m > 0.0f)) return true;
    if (water_moved && h.wetd <= h.disp + 3.0f)
        for (int k = 0; k < g_nw; ++k)
            if (g_wat[k].edge && regions_meet(h.lx0, h.ly0, h.lx1, h.ly1, g_wat[k].bx, g_wat[k].by, g_wat[k].bx + g_wat[k].bw, g_wat[k].by + g_wat[k].bh))
                return true;
    return false;
}
// Into the layer: each due one's region back to the photo (and its water), then every one there drawn.
// Returns how many regions were redrawn (their indices in `which`).
// The redraw in progress (the depth frame draws it a slice at a time; elsewhere at once): each due
// one's rect back to the photo, then every one there drawn - all from meshes posed at its start.
struct SwayRedraw { bool on = false; bool wm = false; int nd = 0, which[SWAY_MAX]; int tn = 0, touch[SWAY_MAX]; int k = 0, y = -1;
                    int by0 = 0, by1 = IMG_H; };                  // (the rows put back: only those are drawn)
SwayRedraw g_sj;
int g_sj_nd() { return g_sj.nd; }
bool g_sw_send = false;                                // (flat) hair redrawn, waiting to be sent
bool g_ff = false;                                     // (flat) a frame is under way (its zones still to draw)
bool g_sw_start = false;                               // (flat) … its hair / cloth / pendants still to start
int g_sj_which(int k) { return g_sj.which[k]; }
SwDraw* g_sdraw[SWAY_MAX] = {nullptr};
void sway_redraw_start(bool water_moved) {
    g_sj = SwayRedraw();
    if (!g_layer || g_ns == 0) return;
    for (int i = 0; i < g_ns; ++i) if (sway_due(i, water_moved)) g_sj.which[g_sj.nd++] = i;
    if (!g_sj.nd) return;
    for (int i = 0; i < g_ns; ++i)
        if (!g_sdraw[i]) { g_sdraw[i] = (SwDraw*)water_alloc(sizeof(SwDraw)); if (!g_sdraw[i]) { g_sj.nd = 0; return; } }
    for (int i = 0; i < g_ns; ++i) sway_draw_prep(g_sway[i], *g_sdraw[i]);   // (the pose it draws, fixed now)
    for (int k = 0; k < g_sj.nd; ++k) sway_dirty_d(g_sway[g_sj.which[k]], *g_sdraw[g_sj.which[k]]);   // (what it drew last time and now)
    for (int k = 0; k < g_sj.nd; ++k) sway_mark(g_sway[g_sj.which[k]]);   // (the pose it draws)
    const uint16_t* photo = (const uint16_t*)g_img.getBuffer();
    for (int k = 0; k < g_sj.nd; ++k) {
        const Sway& h = g_sway[g_sj.which[k]];
        // Back to the photo - except on the water, which was just drawn afresh this frame (water_moved:
        // it always is when the water moves; without water nothing is skipped).
        for (int y = h.uy0; y < h.uy1; ++y) {
            const uint16_t* sr = photo + y * IMG_W + h.ux0;
            uint16_t* d = g_layer + y * IMG_W + h.ux0;
            const int n = h.ux1 - h.ux0;
            if (!water_moved) {
                if (g_layer_sw == g_swapped) memcpy(d, sr, (size_t)n * 2);
                else for (int x = 0; x < n; ++x) d[x] = bswap16(sr[x]);
                continue;
            }
            const uint8_t* wm[WATER_MAX]; int nm = 0;     // (the water masks that are drawn: bit reads, no bounds tests)
            for (int w = 0; w < g_nw; ++w) if (g_wat[w].edge && g_wat[w].mask) wm[nm++] = g_wat[w].mask;
            const bool same = g_layer_sw == g_swapped;
            for (int x = 0, i = y * IMG_W + h.ux0; x < n; ++x, ++i) {
                bool wet = false;
                for (int w = 0; w < nm && !wet; ++w) wet = (wm[w][i >> 3] >> (i & 7)) & 1;
                if (!wet) d[x] = same ? sr[x] : bswap16(sr[x]);
            }
        }
    }
    for (int i = g_ns - 1; i >= 0; --i) {             // (one at rest draws itself exactly as the photo; the first on top)
        bool touched = false;
        for (int k = 0; k < g_sj.nd; ++k) {
            const Sway& a = g_sway[g_sj.which[k]];
            touched |= regions_meet(a.ux0, a.uy0, a.ux1, a.uy1, g_sway[i].x0, g_sway[i].y0, g_sway[i].x1, g_sway[i].y1);
        }
        if (touched) g_sj.touch[g_sj.tn++] = i;
    }
    g_sj.by0 = IMG_H; g_sj.by1 = 0;                    // only the rows put back need drawing (the rest is as it was)
    for (int k = 0; k < g_sj.nd; ++k) {
        const Sway& a = g_sway[g_sj.which[k]];
        if (a.uy0 < g_sj.by0) g_sj.by0 = a.uy0;
        if (a.uy1 > g_sj.by1) g_sj.by1 = a.uy1;
    }
    g_sj.on = true;
}
// Draw on (both cores, 24 rows at a time); budget: stop when this tick's time is up. True when done.
bool sway_redraw_run(bool budget) {
    while (g_sj.on && g_sj.k < g_sj.tn) {
        const int i = g_sj.touch[g_sj.k];
        const Sway& s = g_sway[i];
        const int sy0 = s.y0 > g_sj.by0 ? s.y0 : g_sj.by0, sy1 = s.y1 < g_sj.by1 ? s.y1 : g_sj.by1;
        if (g_sj.y < 0) g_sj.y = sy0;
        if (g_sj.y >= sy1) { ++g_sj.k; g_sj.y = -1; continue; }
        const int ye = g_sj.y + 24 < sy1 ? g_sj.y + 24 : sy1;
        const uint32_t units = (uint32_t)((ye - g_sj.y + 1) / 2) * (uint32_t)(s.x1 - s.x0 + 16);   // (a core's half)
        if (budget && !fits(g_r_sway.est(units))) return false;
        RPiece pc(g_r_sway, units);
        SwayJob job{&s, g_sdraw[i]};
        par_rows(sway_rows, &job, g_sj.y, ye);
        g_sj.y = ye;
    }
    if (g_sj.on)
        for (int k = 0; k < g_sj.nd; ++k) g_sway[g_sj.which[k]].drawn = true;   // (moving on: sway_due)
    g_sj.on = false;
    return true;
}
bool any_sway_due() {
    for (int i = 0; i < g_ns; ++i) if (sway_due(i, false)) return true;
    return false;
}
// Send the redrawn regions to the screen.
void sway_push(const int* which, int nd) {
    int cur = 0;
    bool inflight = false;
    M5.Display.startWrite();
    for (int k = 0; k < nd; ++k) {
        const Sway& h = g_sway[which[k]];
        const int ow = h.ux1 - h.ux0;
        if (ow <= 0) continue;
        for (int y = h.uy0; y < h.uy1; y += STRIP_ROWS) {
            const int ye = y + STRIP_ROWS < h.uy1 ? y + STRIP_ROWS : h.uy1;
            uint16_t* out = g_strip[cur];
            if (inflight && USE_DMA && g_dma) { M5.Display.waitDMA(); inflight = false; }
            for (int yy = y; yy < ye; ++yy) {
                const uint16_t* row = g_layer + yy * IMG_W + h.ux0;
                uint16_t* o = out + (yy - y) * ow;
                if (g_layer_sw) memcpy(o, row, (size_t)ow * 2);
                else for (int x = 0; x < ow; ++x) o[x] = bswap16(row[x]);
            }
            push_strip(h.ux0, y, ow, ye - y, out, inflight);
            cur ^= 1;
        }
    }
    if (inflight) M5.Display.waitDMA();
    M5.Display.endWrite();
}

// ---- parallax -------------------------------------------------------------------------------------
// Content (what the planes show of themselves): the photo with the water drawn in (the layer), and the
// zones drawn into g_anim on top of that when there are zones.
// The parallax frame being sent (it may take a few ticks).
struct ParJobState { bool on = false; ParFrame F; DirtyRects dr; int ri = 0, y = 0; uint32_t px = 0, us = 0; float v0[2] = {0, 0}, s0 = 1; };
ParJobState g_pj;
uint8_t g_ps = 0;                                      // the depth frame's stage (par_tick; 0: none)
void frame_reset() {                                  // (no frame in progress)
    g_pj.on = false; g_wj = WaterJobState(); g_sj = SwayRedraw(); g_sw_send = g_ff = g_sw_start = false; g_ps = 0;
    g_clock.cancel();
}   // (no frame in progress)
void par_push(uint32_t now);
const uint16_t* par_content(bool& sw) {
    if (g_anim) { sw = g_anim_sw; return g_anim; }
    sw = render_src_swapped();
    return render_src();
}
bool g_par_full_sync = true;                           // the planes' fills need the whole content again
void anim_sync() {                                     // g_anim ← the layer (zones redraw themselves)
    g_par_full_sync = true;
    if (!g_anim) return;
    const uint16_t* s = render_src();
    g_anim_sw = render_src_swapped();
    memcpy(g_anim, s, (size_t)IMG_W * IMG_H * 2);
}
void prepare_parallax() {
    frame_reset();
    parallax_prepare(g_par, (const uint16_t*)g_img.getBuffer(), g_swapped, water_alloc, alloc_fast);
    if (g_par.ready && g_n > 0) {
        if (!g_anim) g_anim = (uint16_t*)alloc_psram((size_t)IMG_W * IMG_H * 2);
    } else if (g_anim) { free(g_anim); g_anim = nullptr; }
    if (g_par.ready && !g_front) g_front = (uint16_t*)alloc_psram((size_t)IMG_W * IMG_H * 2);
    if (!g_par.ready && g_front) { free(g_front); g_front = nullptr; }
    anim_sync();
    g_par.drawn = false;
}
void clear_parallax() {
    frame_reset();
    parallax_free(g_par);
    g_par = Parallax();
    if (g_anim) { free(g_anim); g_anim = nullptr; }
    if (g_front) { free(g_front); g_front = nullptr; }
}
bool any_plane(const Parallax& P) {
    if (par_ground(P)) return true;
    for (int k = 0; k < P.n; ++k) if (plane_any(P.pl[k])) return true;
    return false;
}

struct ParJob { const ParFrame* F; uint16_t* out; int x0, x1, ys; int16_t* rng; };
void par_rows_job(void* c, int lane, int y0, int y1) {
    ParJob& j = *(ParJob*)c;
    parallax_render(*j.F, j.out + (y0 - j.ys) * (j.x1 - j.x0), j.x0, y0, j.x1, y1, lane, j.rng + 2 * (y0 - j.ys));
}


// One parallax frame: step the view, update the content (water, zones), redraw what changed - in stages,
// each started only in a tick that has room for it: 1 water into the layer, 2 hair / cloth / pendants to
// start, 3 them drawn, 4 the content step (zones, the planes' fills, what to redraw); then it's sent.
void par_content_begin(bool zones_any, bool water_done);
bool par_content_sync();
void par_content_finish(uint32_t now, bool moved);
void par_tick(uint32_t now, float dt, bool may_draw, bool zones_any) {
    const bool moved = parallax_step(g_par, dt, PAR_GX_S * g_gyro_xy[0], PAR_GY_S * g_gyro_xy[1], M5.Imu.isEnabled());
    g_par_moving = moved || g_par.prm[PP_AMBIENT] > 0.0f || g_pj.on;
    if (!may_draw) return;
    if (g_pj.on) { par_push(now); return; }            // finish sending the frame in progress first
    static bool s_water_done = false, s_moved = false, s_zones = false;   // (carried while the stages run)
    if (g_ps == 0) {
        if (!g_clock.due(now)) return;
        g_clock.begin(now);
        s_moved = moved; s_zones = zones_any; s_water_done = false;
        if (g_nw > 0 && g_layer) { water_job_begin(true); g_ps = 1; } else g_ps = 2;
    } else { s_moved |= moved; s_zones |= zones_any; }
    if (g_ps == 1) {                                   // water: into the layer, a slice at a time
        if (!water_job()) return;
        s_water_done = true; g_ps = 2;
    }
    if (g_ps == 2) {                                   // hair / cloth / pendants into the layer: a slice at a time
        if (g_ns > 0) {
            if (!fits(g_c_swstart)) return;
            { Piece pc(g_c_swstart); sway_redraw_start(g_nw > 0); }
        } else g_sj = SwayRedraw();
        g_ps = 3;
    }
    if (g_ps == 3) {
        if (!sway_redraw_run(true)) return;
        g_ps = 4;
    }
    if (g_ps == 4) {
        if (!fits(g_c_content)) return;
        { Piece pc(g_c_content); par_content_begin(s_zones, s_water_done); }
        g_ps = 5;
    }
    if (g_ps == 5) {
        if (!par_content_sync()) return;
        g_ps = 6;
    }
    if (!fits(g_c_frame)) return;
    g_ps = 0;
    { Piece pc(g_c_frame); par_content_finish(now, s_moved); }
    if (g_pj.on) par_push(now);
}

// Each water's wet stretch per row (x0, x1; x0 >= x1: none) - what changes when it's redrawn.
static int16_t* s_wrow[WATER_MAX] = {nullptr};
const int16_t* water_rows(int i) {
    const Water& w = g_wat[i];
    if (!w.sp || !w.sp_off) return nullptr;
    if (!s_wrow[i]) s_wrow[i] = (int16_t*)alloc_psram((size_t)IMG_H * 4);
    int16_t* r = s_wrow[i];
    if (!r) return nullptr;
    for (int y = 0; y < IMG_H; ++y) { r[2 * y] = IMG_W; r[2 * y + 1] = 0; }
    for (int y = w.by; y < w.by + w.bh; ++y)
        for (int k = w.sp_off[y - w.by]; k < w.sp_off[y - w.by + 1]; ++k) {
            if (w.sp[2 * k] < r[2 * y]) r[2 * y] = w.sp[2 * k];
            if (w.sp[2 * k + 1] > r[2 * y + 1]) r[2 * y + 1] = w.sp[2 * k + 1];
        }
    return r;
}

// The rest of starting a depth frame, once the water and hair / cloth are drawn into the layer - in pieces
// over as many ticks as it takes (nothing reaches the screen before it's done): 1 the changed content into
// the content buffer and the zones drawn over it, 2 that into the planes' fills (both cores, some rows at a
// time), 3 what to redraw.
struct ContentJob {
    DirtyRects content, wbox, zc;                      // what changed; the water's boxes (it changed only where wet); the zones
    const int16_t* wrows[WATER_MAX] = {nullptr};       // each water's wet stretch per row
    bool water_done = false, full = false;
    int si = 0, sy = -1;                               // (2: which rect / water, which row)
};
ContentJob g_cj;
// 1: the content buffer.
void par_content_begin(bool zones_any, bool water_done) {
    ContentJob& J = g_cj;
    J = ContentJob();
    J.water_done = water_done;
    DirtyRects& content = J.content;
    DirtyRects& wbox = J.wbox;
    const int16_t** wrows = J.wrows;
    if (water_done) {                                  // (and the content buffer)
        for (int i = 0; i < g_nw; ++i) {
            const Water& w = g_wat[i];
            if (!w.edge) continue;
            const int16_t* wr = wrows[i] = water_rows(i);
            if (g_anim)
                for (int y = w.by; y < w.by + w.bh; ++y) {
                    const int xa = wr ? wr[2 * y] : w.bx, xb = wr ? wr[2 * y + 1] : w.bx + w.bw;
                    if (xa >= xb) continue;
                    const uint16_t* sr = g_layer + y * IMG_W + xa;
                    uint16_t* d = g_anim + y * IMG_W + xa;
                    if (g_anim_sw == g_layer_sw) memcpy(d, sr, (size_t)(xb - xa) * 2);
                    else for (int x = 0; x < xb - xa; ++x) d[x] = bswap16(sr[x]);
                }
            wbox.add(w.bx, w.by, w.bx + w.bw, w.by + w.bh);
            for (int k = 0; k < g_n; ++k) {            // zones over water redraw on top of it
                const Zone& z = g_z[k];
                if (z.wmap && !(z.bx >= w.bx + w.bw || w.bx >= z.bx + z.bw || z.by >= w.by + w.bh || w.by >= z.by + z.bh))
                    g_shown[k] = true, zones_any = true;
            }
        }
    }
    {                                                  // hair / cloth / pendants (drawn into the layer): into the content
        const int nd = g_sj.nd;
        for (int k = 0; k < nd; ++k) {
            const Sway& h = g_sway[g_sj.which[k]];
            if (g_anim)
                for (int y = h.uy0; y < h.uy1; ++y) {
                    const uint16_t* sr = g_layer + y * IMG_W + h.ux0;
                    uint16_t* d = g_anim + y * IMG_W + h.ux0;
                    if (g_anim_sw == g_layer_sw) memcpy(d, sr, (size_t)(h.ux1 - h.ux0) * 2);
                    else for (int x = 0; x < h.ux1 - h.ux0; ++x) d[x] = bswap16(sr[x]);
                }
            content.add(h.ux0, h.uy0, h.ux1, h.uy1);
            for (int z = 0; z < g_n; ++z)
                if (g_z[z].wmap && regions_meet(g_z[z].bx, g_z[z].by, g_z[z].bx + g_z[z].bw, g_z[z].by + g_z[z].bh, h.ux0, h.uy0, h.ux1, h.uy1))
                    g_shown[z] = true, zones_any = true;
        }
    }
    DirtyRects& zc = J.zc;                             // (zones drawn: kept apart, for the exact redraw below)
    if (zones_any && g_anim) draw_frame(&zc);
    for (int i = 0; i < zc.n; ++i) content.add(zc.r[i][0], zc.r[i][1], zc.r[i][2], zc.r[i][3]);
    J.full = g_par_full_sync;                          // (the first frame: all of it)
    g_par_full_sync = false;
}
// 2: into the planes' fills, a piece at a time (true when done).
struct SyncJob { const uint16_t* c; bool sw; int x0, x1; const int16_t* wr; };
void sync_rows(void* p, int lane, int y0, int y1) {
    const SyncJob& j = *(const SyncJob*)p;
    for (int y = y0; y < y1; ++y) {
        const int xa = j.wr ? j.wr[2 * y] : j.x0, xb = j.wr ? j.wr[2 * y + 1] : j.x1;
        if (xa < xb) parallax_sync_content(g_par, j.c, j.sw, xa, y, xb, y + 1);
    }
}
Rate g_r_sync;
bool par_content_sync() {
    ContentJob& J = g_cj;
    bool csw;
    const uint16_t* c = par_content(csw);
    const int nr = J.full ? 1 : J.content.n;           // the rects, then the waters
    for (;;) {
        if (J.si >= nr + g_nw) return true;
        int x0, y0, x1, y1;
        const int16_t* wr = nullptr;
        if (J.full) { x0 = 0; y0 = 0; x1 = IMG_W; y1 = IMG_H; }
        else if (J.si < nr) { x0 = J.content.r[J.si][0]; y0 = J.content.r[J.si][1]; x1 = J.content.r[J.si][2]; y1 = J.content.r[J.si][3]; }
        else {
            const Water& w = g_wat[J.si - nr];
            if (J.full || !J.water_done || !w.edge) { ++J.si; J.sy = -1; continue; }
            x0 = w.bx; y0 = w.by; x1 = w.bx + w.bw; y1 = w.by + w.bh; wr = J.wrows[J.si - nr];
        }
        if (J.sy < 0) J.sy = y0;
        if (J.sy >= y1) { ++J.si; J.sy = -1; continue; }
        const int ye = J.sy + 24 < y1 ? J.sy + 24 : y1;
        uint32_t units = 0;
        for (int y = J.sy; y < ye; ++y) units += 16 + (uint32_t)(wr ? (wr[2 * y] < wr[2 * y + 1] ? wr[2 * y + 1] - wr[2 * y] : 0) : x1 - x0);
        units = units / 2 + 16;                        // (a core's half)
        if (!fits(g_r_sync.est(units))) return false;
        RPiece pc(g_r_sync, units);
        SyncJob job{c, csw, x0, x1, wr};
        par_rows(sync_rows, &job, J.sy, ye);           // (rows apart: each pixel goes to its own owner's fill)
        J.sy = ye;
    }
}
// 3: what to redraw, and the frame to send.
void par_content_finish(uint32_t now, bool moved) {
    ContentJob& J = g_cj;
    DirtyRects& content = J.content;
    const DirtyRects& wbox = J.wbox;
    const DirtyRects& zc = J.zc;
    const int16_t* const* wrows = J.wrows;
    const bool water_done = J.water_done, full_sync = J.full;
    for (int i = 0; i < wbox.n; ++i) content.add(wbox.r[i][0], wbox.r[i][1], wbox.r[i][2], wbox.r[i][3]);   // (for what to redraw)
    if (!moved && content.n == 0) { g_clock.cancel(); return; }

    // screen rects: planes that slid (or the whole frame if the background slides too), plus content
    // changes, spread by the most any plane is shifted
    DirtyRects& dr = g_pj.dr;
    dr.n = 0;
    if (moved) {
        if (g_par.prm[PP_FOCUS] > 0.0f || par_ground(g_par) || !g_par.drawn) dr.add(0, 0, IMG_W, IMG_H);   // (the ground: all of it)
        else for (int k = 0; k < g_par.n; ++k) { int r[4]; parallax_plane_rect(g_par, k, r); dr.add(r[0], r[1], r[2], r[3]); }
    }
    // A changed content pixel shows on screen up to 2 margins away (its slide, plus the zoom's offset toward
    // the edges) - plus the filter's reach, inside the margin's slack.
    // (Whole rows: the render then redraws only the pixels it must and copies the rest from the screen.)
    const int m = 2 * g_par.margin;
    for (int i = 0; i < content.n; ++i) dr.add(0, content.r[i][1] - m, IMG_W, content.r[i][3] + m);
    if (moved) {
        DirtyRects wide;
        for (int i = 0; i < dr.n; ++i) wide.add(0, dr.r[i][1], IMG_W, dr.r[i][3]);
        dr = wide;
    }
    bool sw;
    const uint16_t* anim = par_content(sw);
    // Not moved (yet): the content is drawn at the view the screen shows, not one it hasn't caught up to -
    // or the redrawn bits would sit a fraction of a pixel off the rest, and the screen would never catch up.
    // (Likewise as sharp as the screen shows it.)
    const float vnow[2] = {g_par.v[0], g_par.v[1]}, snow = g_par.sharp;
    if (!moved && g_par.drawn) { g_par.v[0] = g_par.dv[0]; g_par.v[1] = g_par.dv[1]; g_par.sharp = g_par.dsharp; }
    parallax_frame(g_pj.F, g_par, anim, sw, g_nw > 0 || g_n > 0);
    const float vshow[2] = {g_par.v[0], g_par.v[1]}, sshow = g_par.sharp;
    g_par.v[0] = vnow[0]; g_par.v[1] = vnow[1]; g_par.sharp = snow;
    // pixels drawn exactly as the screen shows them are copied, not redrawn: not across what changed
    g_pj.F.front = g_par.drawn ? g_front : nullptr;
    ParFrame& F = g_pj.F;                              // what changed, exactly: the render works out which pixels
    F.force_all = full_sync; F.nforce = 0; F.nfw = 0; //  each layer's filter would see it through
    auto frect = [&F](int x0, int y0, int x1, int y1) {
        if (F.nforce >= 24) { F.force_all = true; return; }
        int16_t* r = F.force[F.nforce++];
        r[0] = (int16_t)x0; r[1] = (int16_t)y0; r[2] = (int16_t)x1; r[3] = (int16_t)y1;
    };
    if (water_done)
        for (int i = 0; i < g_nw; ++i) {
            if (!g_wat[i].edge) continue;
            const int16_t* wr = wrows[i];
            if (wr) F.fwrow[F.nfw++] = wr;
            else frect(g_wat[i].bx, g_wat[i].by, g_wat[i].bx + g_wat[i].bw, g_wat[i].by + g_wat[i].bh);
        }
    for (int k = 0; k < g_sj.nd; ++k) { const Sway& h = g_sway[g_sj.which[k]]; frect(h.ux0, h.uy0, h.ux1, h.uy1); }
    for (int i = 0; i < zc.n; ++i) frect(zc.r[i][0], zc.r[i][1], zc.r[i][2], zc.r[i][3]);
    g_pj.v0[0] = vshow[0]; g_pj.v0[1] = vshow[1]; g_pj.s0 = sshow;
    g_pj.ri = 0; g_pj.y = dr.n ? dr.r[0][1] : 0; g_pj.px = 0; g_pj.us = 0;
    g_pj.on = true;
}


// Will a strip be rendered (the view slid, or content changed near it), or mostly copied from the screen?
// (Only the time it's expected to take: which pixels are drawn is the render's business.)
bool par_strip_renders(int y0, int y1) {
    const ParFrame& F = g_pj.F;
    if (!F.front || F.force_all || g_pj.v0[0] != g_par.dv[0] || g_pj.v0[1] != g_par.dv[1]) return true;
    const int m = 2 * g_par.margin;
    for (int i = 0; i < F.nforce; ++i) if (F.force[i][1] - m < y1 && F.force[i][3] + m > y0) return true;
    for (int i = 0; F.nfw && i < g_nw; ++i)          // (the waters, when they were redrawn)
        if (g_wat[i].edge && g_wat[i].by - m < y1 && g_wat[i].by + g_wat[i].bh + m > y0) return true;
    return false;
}
// Send the frame in progress, strip by strip, until it's done or this tick's time is up.
void par_push(uint32_t now) {
    DirtyRects& dr = g_pj.dr;
    uint32_t t0 = micros();
    int cur = 0;
    bool inflight = false;
    M5.Display.startWrite();
    while (g_pj.ri < dr.n) {
        const int x0 = dr.r[g_pj.ri][0], x1 = dr.r[g_pj.ri][2];
        if (g_pj.y >= dr.r[g_pj.ri][3]) {
            if (++g_pj.ri < dr.n) g_pj.y = dr.r[g_pj.ri][1];
            continue;
        }
        const int y = g_pj.y, ye = y + STRIP_ROWS < dr.r[g_pj.ri][3] ? y + STRIP_ROWS : dr.r[g_pj.ri][3];
        Cost& cost = par_strip_renders(y, ye) ? g_c_push : g_c_pushc;   // (rendered, or copied from the screen)
        if (!fits(cost)) break;
        Piece pc(cost);
        g_pj.y = ye;
        int16_t rng[2 * STRIP_ROWS];                // (per row: the columns worked out afresh - the rest is the screen's)
        ParJob job{&g_pj.F, g_strip[cur], x0, x1, y, rng};
        par_rows(par_rows_job, &job, y, ye);        // both cores
        int bx0 = x0, bx1 = x1, by0 = y, by1 = ye;
        uint16_t* buf = g_strip[cur];
        const int w = x1 - x0;
        if (g_front && g_par.drawn) {               // send only what changed (exact: same picture)
            bx0 = IMG_W; bx1 = -1; by0 = IMG_H; by1 = -1;
            for (int yy = y; yy < ye; ++yy) {     // (only where it was worked out afresh: the rest is the screen's)
                const uint16_t* a = buf + (yy - y) * w;
                const uint16_t* f = g_front + yy * IMG_W + x0;
                const int lo = rng[2 * (yy - y)] - x0, hi = rng[2 * (yy - y) + 1] - x0;
                if (lo >= hi) continue;
                int l = lo, r = hi - 1;
                while (l < hi && a[l] == f[l]) ++l;
                if (l == hi) continue;
                while (a[r] == f[r]) --r;
                if (x0 + l < bx0) bx0 = x0 + l;
                if (x0 + r + 1 > bx1) bx1 = x0 + r + 1;
                if (yy < by0) by0 = yy;
                by1 = yy + 1;
            }
            if (by1 < 0) continue;                  // nothing changed in this strip
            const int w2 = bx1 - bx0;
            if (w2 != w || by0 != y)                // pack the changed box to the buffer's start
                for (int yy = by0; yy < by1; ++yy)
                    memmove(buf + (yy - by0) * w2, buf + (yy - y) * w + (bx0 - x0), (size_t)w2 * 2);
        }
        if (inflight && USE_DMA && g_dma) { M5.Display.waitDMA(); inflight = false; }   // (the other buffer is free)
        if (g_front)
            for (int yy = by0; yy < by1; ++yy) memcpy(g_front + yy * IMG_W + bx0, buf + (yy - by0) * (bx1 - bx0), (size_t)(bx1 - bx0) * 2);
        push_strip(bx0, by0, bx1 - bx0, by1 - by0, buf, inflight);
        g_pj.px += (uint32_t)(bx1 - bx0) * (by1 - by0);
        cur ^= 1;
    }
    if (inflight) M5.Display.waitDMA();
    M5.Display.endWrite();
    g_pj.us += micros() - t0;
    if (g_pj.ri < dr.n) return;                     // (more next tick)
    g_pj.on = false;
    g_par.dv[0] = g_pj.v0[0]; g_par.dv[1] = g_pj.v0[1];   // the view this frame shows (and its filter)
    g_par.dsharp = g_pj.s0;
    g_par.drawn = true;
    g_clock.end(now);
    g_pf_us += g_pj.us; g_pf_px += g_pj.px; ++g_pf_frames;
    if (now - g_pf_t0 >= 3000) {
        if (g_pf_frames)
            Serial.printf("jig: depth %.1f fps, frame %.1f ms, %lu px/frame\n", g_pf_frames * 1000.0f / (now - g_pf_t0),
                          g_pf_us / 1000.0f / g_pf_frames, (unsigned long)(g_pf_px / g_pf_frames));
        g_pf_t0 = now; g_pf_frames = g_pf_us = g_pf_px = 0;
    }
}

void rest_all() {
    for (int i = 0; i < g_n; ++i) {
        zone_rest(g_z[i]);
        g_shown[i] = false;
        g_hold[i][0] = g_hold[i][1] = g_hold[i][2] = 0;
    }
    for (int i = 0; i < g_ns; ++i) {                   // hair / cloth / pendants: home, and drawn once more
        Sway& h = g_sway[i];
        if (!h.ready) continue;
        for (int k = 0; k < h.nr; ++k)
            for (int j = 0; j < h.nn[k]; ++j) { h.x[k][j] = h.xp[k][j] = h.rx[k][j]; h.y[k][j] = h.yp[k][j] = h.ry[k][j]; }
        h.th = h.thp = 0; h.fade = 0; h.disp = 0; h.drawn = false;
    }
}

void kick_all() {
    for (int i = 0; i < g_n; ++i) {
        g_z[i].vy += 0.6f * g_z[i].w1;
        g_z[i].vs += 0.4f * g_z[i].w1;
        g_z[i].awake = true;
    }
    for (int i = 0; i < g_ns; ++i) {                   // and a small swing of each hair / cloth / pendant
        Sway& h = g_sway[i];
        if (!h.ready) continue;
        sway_kick(h, 0.8f, (i & 1) ? -1 : 1);
        h.drawn = false;
    }
}

// ================================================================================================
// Editor
// ================================================================================================
bool     e_open = false;
Zone     e_z[MAX_ZONES];                 // working copy (definitions only, no maps)
int      e_n = 0;
int      e_active = -1;                  // zone being edited
int      e_armed = -1;                   // shape waiting to be placed by a tap
float    e_last_r = R_DEFAULT;
uint8_t  e_last_mat = MAT_NATURAL;
bool     e_dirty = true;
uint32_t e_last_draw = 0;
bool     e_btn_ready = false;            // ignore buttons that were already down when we opened
uint32_t e_rep_a = 0, e_rep_c = 0;
char     e_toast[48] = {0};
uint32_t e_toast_until = 0;

// Water editing
enum WTool : uint8_t { WT_BRUSH = 0, WT_SMART, WT_ERASE, WT_POLY, WT_LINE, WT_GRID, WT_COUNT };
Water    e_w[WATER_MAX];                 // working copies (masks included)
int      e_nw = 0, e_wi = -1;            // regions, the one being edited
bool     e_wmode = false;                // editing water (not zones)
bool     e_dmode = false;                // editing depth planes
Plane    e_p[PLANE_MAX];                 // working copies of the planes (masks included)
int      e_np = 0, e_pi = -1;
float    e_pprm[PP_COUNT];
bool     e_pgrid = false;               // the ground grid (Grid tool in depth mode)
float    e_gq[8];
const float DEPTHS[] = {0.15f, 0.3f, 0.5f, 0.7f, 0.9f};
Sway*    e_s = nullptr;                  // working copies of hair / cloth / pendants (PSRAM: begin())
Sway*    e_sdflt = nullptr;              // (the dimmed guides one gets when none are drawn)
int      e_ns = 0, e_si = -1;
bool     e_smode = false;                // editing hanging things
bool     e_stroke = false;               // a hair lock being drawn (root → tip, in e_px / e_py)
bool e_gmode = false;                         // fitting the ground grid (shared by water and depth)
bool e_mm() { return e_wmode || e_dmode || e_smode || e_gmode; }   // editing a mask (water area, depth plane, hanging thing)
uint8_t  e_wtool = WT_BRUSH;
int      e_brush = 8;                    // brush radius (px)
// Smart selection: what you tapped / stroked, the mask before it (so A/C can step the edge), a result
uint8_t* e_marks = nullptr;
uint8_t* e_before = nullptr;
uint8_t* e_selbuf = nullptr;
bool     e_glive = false, e_gadd = true;     // the last smart pick can still be stepped
int      e_gstep = 0, e_glevels = 0, e_gidx = -1;
int16_t  e_px[32], e_py[32];             // polygon / line being drawn
int      e_pn = 0;
int      e_bx = -1, e_by = -1;           // last brush dab (to join a drag smoothly)

bool     t_down = false, t_moved = false, t_bar = false, t_tip = false;
int      t_x0 = 0, t_y0 = 0, t_lx = 0, t_ly = 0;
uint32_t t_t0 = 0;

float g_out_u[SHAPE_COUNT][OUTLINE_PTS], g_out_w[SHAPE_COUNT][OUTLINE_PTS];
bool  g_out_init = false;

void toast(const char* msg);
int bar_bottom() { return (e_active >= 0 || e_wmode || e_dmode || e_smode || e_gmode) ? 2 * ROW_H : ROW_H; }
// The bars sit at the top, or (a tap on the hint line moves them) just above the footer, so what's
// under them can be edited too - hair is usually right at the top.
bool e_low = false;
int row_y(int row) { return e_low ? IMG_H - FOOT_H - (row + 1) * ROW_H : row * ROW_H; }   // (low: row 1 nearest the footer)
int bars_y() { return e_low ? IMG_H - FOOT_H - bar_bottom() : 0; }
int hint_y() { return e_low ? bars_y() - HINT_H : bar_bottom(); }
bool in_bars(int y) { const int a = e_low ? hint_y() : 0, b = e_low ? bars_y() + bar_bottom() : bar_bottom() + HINT_H; return y >= a && y < b; }
// The picture you can see and touch between the bars (and the footer).
int vis_top() { return e_low ? 0 : bar_bottom() + HINT_H; }
int vis_bot() { return e_low ? hint_y() : IMG_H - FOOT_H; }

const char* wtool_label(uint8_t t) {
    if (t == WT_SMART) return "Smart";
    if (t == WT_GRID) return "Grid";
    if (t == WT_LINE && e_smode) {                     // the guide tool, per kind
        const Sway* h = (e_si >= 0 && e_si < e_ns) ? &e_s[e_si] : nullptr;
        if (!h || h->kind == SK_HAIR) return "Locks";
        return h->kind == SK_CLOTH ? (h->ng == 0 || h->gn[0] < 2 ? "Waist" : "Folds") : "Pin";
    }
    return t == WT_ERASE ? "Erase" : (t == WT_POLY ? "Shape" : (t == WT_LINE ? (e_wi >= 0 && e_w[e_wi].mode == W_OCEAN ? "Shore" : "Flow") : "Paint"));
}

Water* e_water() { return (e_wi >= 0 && e_wi < e_nw) ? &e_w[e_wi] : nullptr; }

bool e_new_water() {
    if (e_nw >= WATER_MAX) { return false; }
    Water& w = e_w[e_nw];
    water_free(w);
    w = Water();
    if (!water_ensure_mask(w, water_alloc)) return false;
    e_wi = e_nw++;
    return true;
}

// Paint / erase a dab, joined to the previous one while dragging.
uint8_t* e_cur_mask();
void e_dab(int x, int y) {
    uint8_t* m = e_cur_mask();
    if (!m) return;
    bool v = e_wtool == WT_BRUSH;
    if (e_bx >= 0) {
        int dx = x - e_bx, dy = y - e_by, n = (abs(dx) > abs(dy) ? abs(dx) : abs(dy)) / 2 + 1;
        for (int k = 1; k <= n; ++k) mask_brush(m, e_bx + dx * k / n, e_by + dy * k / n, e_brush, v);
    } else {
        mask_brush(m, x, y, e_brush, v);
    }
    e_bx = x; e_by = y;
}

// ---- smart selection -----------------------------------------------------------------------------
void e_draw();
Plane* e_plane() { return (e_pi >= 0 && e_pi < e_np) ? &e_p[e_pi] : nullptr; }
Sway* e_sway() { return (e_si >= 0 && e_si < e_ns) ? &e_s[e_si] : nullptr; }
bool e_new_sway() {
    if (e_ns >= SWAY_MAX) return false;
    Sway& h = e_s[e_ns];
    sway_free(h);
    new (&h) Sway();                                    // (in place: never an ~8 KB temporary on the stack)
    if (!sway_ensure_mask(h, water_alloc)) return false;
    e_si = e_ns++;
    return true;
}
uint8_t* e_cur_mask() {
    if (e_dmode) { Plane* p = e_plane(); return p ? p->mask : nullptr; }
    if (e_smode) { Sway* h = e_sway(); return h ? h->mask : nullptr; }
    Water* w = e_water(); return w ? w->mask : nullptr;
}
// ---- the ground grid: fitted by eye to the ground (Grid button). Depth planes stand on it and the water
// takes its perspective from it (one grid for the whole picture).
int8_t e_gk = -1;                                  // corner being dragged (-1: the whole grid)
bool* e_grid_on() { return &e_pgrid; }
float* e_grid_q() { return e_gq; }
// A starting grid: over the water, its box narrowed toward the top; the ground, the lower frame.
void e_grid_default() {
    float* q = e_grid_q(); bool* on = e_grid_on();
    if (!q || !on) return;
    int x0 = 20, x1 = 300, y0 = 150, y1 = 230;
    const uint8_t* m = nullptr;                    // (the lower part of the frame)
    if (m) {
        int a = IMG_W, b = -1, c = IMG_H, d = -1;
        for (int y = 0; y < IMG_H; ++y)
            for (int x = 0; x < IMG_W; ++x)
                if (mask_get(m, x, y)) { if (x < a) a = x; if (x > b) b = x; if (y < c) c = y; if (y > d) d = y; }
        if (b > a + 8 && d > c + 4) { x0 = a; x1 = b; y0 = c; y1 = d; }
    }
    const float in = (x1 - x0) * 0.3f;
    const float g[8] = {(float)x0, (float)y1, (float)x1, (float)y1, x1 - in, (float)y0, x0 + in, (float)y0};
    memcpy(q, g, sizeof(g));
    *on = true;
}
// Turn it about its middle (levels the horizon with the photo's).
void e_grid_turn(float deg) {
    float* q = e_grid_q(); bool* on = e_grid_on();
    if (!q || !on || !*on) return;
    const float cx = (q[0] + q[2] + q[4] + q[6]) * 0.25f, cy = (q[1] + q[3] + q[5] + q[7]) * 0.25f;
    const float cs = cosf(deg * 0.0174533f), sn = sinf(deg * 0.0174533f);
    for (int k = 0; k < 4; ++k) {
        const float dx = q[2 * k] - cx, dy = q[2 * k + 1] - cy;
        q[2 * k] = cx + dx * cs - dy * sn; q[2 * k + 1] = cy + dx * sn + dy * cs;
    }
}
int bar_bottom();
// Where a corner's handle shows: on the corner, or pulled in to the visible picture (under the bars and
// the footer, or past the frame) - it still drags the corner itself.
void e_grid_handle(const float* q, int k, int& hx, int& hy) {
    const int top = vis_top() + 6, bot = vis_bot() - 7;
    hx = (int)clampf(q[2 * k], 7.0f, IMG_W - 8.0f);
    hy = (int)clampf(q[2 * k + 1], (float)top, (float)bot);
}
void e_grid_press(int x, int y) {
    float* q = e_grid_q(); bool* on = e_grid_on();
    e_gk = -1;
    if (!q || !on || !*on) return;
    int best = 28 * 28;
    for (int k = 0; k < 4; ++k) {
        int hx, hy;
        e_grid_handle(q, k, hx, hy);
        const int dx = x - hx, dy = y - hy;
        if (dx * dx + dy * dy < best) { best = dx * dx + dy * dy; e_gk = (int8_t)k; }
    }
}
void e_grid_drag(int dx, int dy) {
    float* q = e_grid_q(); bool* on = e_grid_on();
    if (!q || !on || !*on) return;
    for (int k = 0; k < 4; ++k) {
        if (e_gk >= 0 && k != e_gk) continue;
        q[2 * k] = clampf(q[2 * k] + dx, -200.0f, IMG_W + 200.0f);
        q[2 * k + 1] = clampf(q[2 * k + 1] + dy, -400.0f, IMG_H + 200.0f);
    }
}
void e_draw_grid(M5Canvas& c) {
    const bool* on = e_grid_on(); const float* q = e_grid_q();
    if (!on || !*on || !q) return;
    const bool act = e_wtool == WT_GRID;
    const uint16_t col = act ? C_LIME : C_DIM;
    const WHomog H = water_homog(q);
    for (int k = 0; k <= 8; ++k) {                     // 8 × 8 squares
        float ax, ay, bx, by;
        water_hmap(H, k / 8.0f, 0, ax, ay); water_hmap(H, k / 8.0f, 1, bx, by);
        c.drawLine((int)ax, (int)ay, (int)bx, (int)by, col);
        water_hmap(H, 0, k / 8.0f, ax, ay); water_hmap(H, 1, k / 8.0f, bx, by);
        c.drawLine((int)ax, (int)ay, (int)bx, (int)by, col);
    }
    if (!act) return;
    // where it meets the sky (dashed): the vanishing line
    const float G = H.d * H.h - H.e * H.g, Hh = -(H.a * H.h - H.b * H.g), I = H.a * H.e - H.b * H.d;
    if (fabsf(Hh) > 1e-9f)
        for (int x = 0; x < IMG_W; x += 12) {
            const int y0 = (int)(-(G * x + I) / Hh), y1 = (int)(-(G * (x + 6) + I) / Hh);
            if (y0 > -10 && y0 < IMG_H + 10) c.drawLine(x, y0, x + 6, y1, C_YEL);
        }
    if (fabsf(Hh) > 1e-9f) {                           // named, so it's not a mystery line
        const int yh = (int)(-(G * 30 + I) / Hh);
        if (yh > vis_top() + 8 && yh < vis_bot() - 8) {
            c.setTextDatum(middle_left); c.setTextColor(C_YEL);
            c.drawString("horizon", 6, yh - 8);
            c.setTextDatum(middle_center);
        }
    }
    for (int k = 0; k < 4; ++k) {
        int hx, hy;
        e_grid_handle(q, k, hx, hy);
        const uint16_t hc = k < 2 ? C_LIME : C_YEL;                       // near corners / far corners
        if (hx != (int)q[2 * k] || hy != (int)q[2 * k + 1]) {             // pulled in: points the way to it
            float dx = q[2 * k] - hx, dy = q[2 * k + 1] - hy, l = sqrtf(dx * dx + dy * dy);
            c.drawLine(hx, hy, hx + (int)(dx / l * 11), hy + (int)(dy / l * 11), hc);
        }
        c.fillCircle(hx, hy, 6, C_BLACK);
        c.drawCircle(hx, hy, 6, hc);
        c.drawCircle(hx, hy, 5, hc);
    }
}

void e_tool_picked() {
    if (e_wtool != WT_GRID) return;
    bool* on = e_grid_on();
    if (on && !*on) { e_grid_default(); toast("Drag corners to fit - hold: grid off"); }
    else toast("Drag corners or inside - hold: grid off");
}

bool e_new_plane() {
    if (e_np >= PLANE_MAX) return false;
    Plane& p = e_p[e_np];
    plane_free(p);
    p = Plane();
    p.depth = e_np == 0 ? 0.7f : 0.5f;
    if (!plane_ensure_mask(p, water_alloc)) return false;
    e_pi = e_np++;
    return true;
}

bool e_smart_mem() {
    if (!e_marks) e_marks = (uint8_t*)water_alloc(MASK_BYTES);
    if (!e_before) e_before = (uint8_t*)water_alloc(MASK_BYTES);
    if (!e_selbuf) e_selbuf = (uint8_t*)water_alloc(MASK_BYTES);
    return e_marks && e_before && e_selbuf;
}
void e_smart_free() {
    for (uint8_t** p : {&e_marks, &e_before, &e_selbuf}) { if (*p) free(*p); *p = nullptr; }
    e_glive = false;
}
void e_busy(const char* msg) {                         // a selection takes a moment: say so first
    toast(msg);
    e_draw();
}

// (Re)apply the last smart pick at boundary step e_gstep, starting from the mask as it was before it.
void e_smart_run() {
    uint8_t* m = e_cur_mask();
    if (!m || !e_smart_mem()) return;
    memcpy(m, e_before, MASK_BYTES);
    SelGrow g = sel_grow((const uint16_t*)g_img.getBuffer(), g_swapped, e_marks, e_gstep, e_selbuf, water_alloc);
    int n = sel_apply(m, e_selbuf, e_gadd, (e_dmode || e_smode) ? -1 : 1, water_alloc, e_marks);   // water keeps off the bank; planes / hangs take their rim
    e_glevels = g.levels; e_gidx = g.index; e_glive = g.levels > 0;
    char msg[48];
    if (!n) snprintf(msg, sizeof(msg), e_gadd ? "Already in - try another spot" : "Nothing to remove there");
    else if (g.levels > 1) snprintf(msg, sizeof(msg), "%s - edge %d of %d - A/C", e_gadd ? "Added" : "Removed", g.index + 1, g.levels);
    else snprintf(msg, sizeof(msg), e_gadd ? "Added" : "Removed");
    toast(msg);
}
void e_smart_go(bool add) {                            // the marks are in: pick from them
    uint8_t* m = e_cur_mask();
    if (!m || !e_smart_mem()) return;
    memcpy(e_before, m, MASK_BYTES);
    e_gstep = 0; e_gadd = add;
    e_busy(add ? "Selecting..." : "Removing...");
    e_smart_run();
}
void e_smart_step(int d) {                             // A smaller / C bigger
    if (!e_glive) { toast("Tap or paint over something first"); return; }
    if (e_gidx + d < 0) { toast("That's the smallest edge"); return; }
    if (e_gidx + d >= e_glevels) { toast("That's the biggest edge"); return; }
    e_gstep += d;
    e_busy(d < 0 ? "Smaller..." : "Bigger...");
    e_smart_run();
}
void e_mark(int x, int y) {                            // a stroke over the sample (joined while dragging)
    if (!e_smart_mem()) return;
    if (e_bx >= 0) {
        int dx = x - e_bx, dy = y - e_by, n = (abs(dx) > abs(dy) ? abs(dx) : abs(dy)) / 2 + 1;
        for (int k = 1; k <= n; ++k) mask_brush(e_marks, e_bx + dx * k / n, e_by + dy * k / n, 4, true);
    } else {
        memset(e_marks, 0, MASK_BYTES);
        mask_brush(e_marks, x, y, 4, true);
    }
    e_bx = x; e_by = y;
}

// Finish what A/C mean for the current tool: C = fill the shape / keep the line.
void e_finish() {
    Water* w = e_water();
    uint8_t* m = e_cur_mask();
    if (!m) return;
    if (e_wtool == WT_POLY) {
        if (e_pn < 3) { toast("Tap 3+ points around it"); return; }
        if (e_smart_mem()) {                           // fill it, fitted to the real edges inside
            e_busy("Fitting to the edges...");
            memset(e_marks, 0, MASK_BYTES);
            mask_poly(e_marks, e_px, e_py, e_pn, true);
            if (!sel_grabcut((const uint16_t*)g_img.getBuffer(), g_swapped, e_marks, e_selbuf, water_alloc))   // GrabCut
                sel_snap((const uint16_t*)g_img.getBuffer(), g_swapped, e_marks, 8, e_selbuf, water_alloc);      // (low memory)
            sel_apply(m, e_selbuf, true, (e_dmode || e_smode) ? -1 : 1, water_alloc);
        } else {
            mask_poly(m, e_px, e_py, e_pn, true);
        }
        e_pn = 0; e_glive = false;
        toast("Fitted - Smart/Paint/Erase to touch up");
    } else if (e_wtool == WT_LINE && e_smode) {
        const Sway* h = e_sway();
        toast(!h || h->ng == 0 ? "No guides: it gets the dimmed ones" : "Guides kept - B saves");
    } else if (e_wtool == WT_LINE && w) {
        if (e_pn < 2) { toast("Tap 2+ points along the water"); return; }
        if (w->nlines >= WLINE_MAX) { toast("3 lines max - A removes the last"); return; }
        WLine& L = w->line[w->nlines++];
        L.n = (uint8_t)(e_pn < WPTS_MAX ? e_pn : WPTS_MAX);
        for (int k = 0; k < L.n; ++k) { L.x[k] = e_px[k]; L.y[k] = e_py[k]; }
        e_pn = 0;
        toast(w->mode == W_OCEAN ? "Shore line kept" : "Flow line kept");
    }
    e_dirty = true;
}

void e_undo_point() {
    if (e_smode && e_wtool == WT_LINE) {               // guides: the last lock / fold / pin off
        Sway* h = e_sway();
        if (!h || h->ng == 0) { toast("No guides yet"); return; }
        if (h->kind == SK_CLOTH && h->ng == 1 && h->gn[0] == 2) h->gn[0] = 1;   // (the waistband's second end)
        else { --h->ng; h->gn[h->ng] = 0; }
        toast(h->kind == SK_HAIR ? "Removed the last lock" : (h->kind == SK_CLOTH ? "Removed the last mark" : "Pin cleared"));
        e_dirty = true;
        return;
    }
    Water* w = e_water();
    if (e_pn > 0) { --e_pn; e_dirty = true; return; }
    if (w && e_wtool == WT_LINE && w->nlines > 0) { --w->nlines; toast("Removed the last line"); e_dirty = true; }
}

// Tint the water on the editor frame so you can see what's covered.
void e_tint_water(M5Canvas& c) {
    uint16_t* buf = (uint16_t*)c.getBuffer();
    for (int i = 0; i < e_nw; ++i) {
        const uint8_t* m = e_w[i].mask;
        if (!m) continue;
        const bool act = e_wmode && i == e_wi;
        for (int y = 0; y < IMG_H; ++y)
            for (int x = 0; x < IMG_W; ++x) {
                if (!mask_get(m, x, y)) continue;
                uint16_t v = buf[y * IMG_W + x];
                uint16_t px = g_swapped ? bswap16(v) : v;
                px = rgb565_mix(px, act ? (uint16_t)0x05FF : (uint16_t)0x0459, act ? 110 : 60);   // cyan wash
                buf[y * IMG_W + x] = g_swapped ? bswap16(px) : px;
            }
    }
}

void e_tint_planes(M5Canvas& c) {
    static const uint16_t PC[PLANE_MAX] = {0xF81F, 0xFC00, 0xFFE0, 0x07E0};   // magenta, orange, yellow, green
    uint16_t* buf = (uint16_t*)c.getBuffer();
    for (int k = 0; k < e_np; ++k) {
        const uint8_t* m = e_p[k].mask;
        if (!m) continue;
        const bool act = e_dmode && k == e_pi;
        for (int i = 0; i < IMG_W * IMG_H; ++i) {
            if (!((m[i >> 3] >> (i & 7)) & 1)) continue;
            uint16_t px = g_swapped ? bswap16(buf[i]) : buf[i];
            px = rgb565_mix(px, PC[k], act ? 110 : (e_dmode ? 50 : 0));
            buf[i] = g_swapped ? bswap16(px) : px;
        }
    }
}

// Hair / cloth / pendants: their selection tinted, and the guides of the one being edited.
void e_tint_sways(M5Canvas& c) {
    if (!e_smode) return;
    uint16_t* buf = (uint16_t*)c.getBuffer();
    for (int k = 0; k < e_ns; ++k) {
        const uint8_t* m = e_s[k].mask;
        if (!m) continue;
        const bool act = k == e_si;
        for (int i = 0; i < IMG_W * IMG_H; ++i) {
            if (!((m[i >> 3] >> (i & 7)) & 1)) continue;
            uint16_t px = g_swapped ? bswap16(buf[i]) : buf[i];
            px = rgb565_mix(px, (uint16_t)0xFD20, act ? 110 : 45);   // orange
            buf[i] = g_swapped ? bswap16(px) : px;
        }
    }
    const Sway* h = e_sway();
    if (!h) return;
    if (!e_sdflt) return;
    Sway& d = *e_sdflt;                                 // (none drawn yet: the ones it will get, dimmed)
    new (&d) Sway();
    const Sway* g = h;
    bool dflt = false;
    if (h->ng == 0 && h->mask && sway_count(*h) >= 12) {
        d.kind = h->kind; d.mask = h->mask;
        sway_default_guides(d);
        d.mask = nullptr;
        g = &d; dflt = true;
    }
    const uint16_t col = dflt ? C_DIM : C_YEL;
    for (int k = 0; k < g->ng; ++k) {
        const int n = g->gn[k];
        if (g->kind == SK_HAIR || (g->kind == SK_CLOTH && k == 0)) {
            for (int i = 0; i + 1 < n; ++i) {
                c.drawWideLine(g->gx[k][i], g->gy[k][i], g->gx[k][i + 1], g->gy[k][i + 1], 3.0f, C_BLACK);
                c.drawWideLine(g->gx[k][i], g->gy[k][i], g->gx[k][i + 1], g->gy[k][i + 1], 1.5f, col);
            }
            c.fillCircle((int)g->gx[k][0], (int)g->gy[k][0], 3, col);          // the root / waistband ends
            if (g->kind == SK_CLOTH && n > 1) c.fillCircle((int)g->gx[k][n - 1], (int)g->gy[k][n - 1], 3, col);
            if (g->kind == SK_HAIR && n > 1) {                                   // the tip: which way it runs
                float dx = g->gx[k][n - 1] - g->gx[k][n - 2], dy = g->gy[k][n - 1] - g->gy[k][n - 2], l = sqrtf(dx * dx + dy * dy);
                if (l > 0.5f) {
                    dx /= l; dy /= l;
                    const float ex = g->gx[k][n - 1], ey = g->gy[k][n - 1];
                    c.fillTriangle((int)(ex + dx * 5), (int)(ey + dy * 5), (int)(ex - dy * 3), (int)(ey + dx * 3), (int)(ex + dy * 3), (int)(ey - dx * 3), col);
                }
            }
        } else if (g->kind == SK_CLOTH) {                                        // a fold: a short line down the cloth
            float ex = 1, ey = 0;
            if (g->gn[0] >= 2) { ex = g->gx[0][1] - g->gx[0][0]; ey = g->gy[0][1] - g->gy[0][0]; const float l = sqrtf(ex * ex + ey * ey); if (l > 0.5f) { ex /= l; ey /= l; } }
            float nx = -ey, ny = ex;
            if (ny < 0) { nx = -nx; ny = -ny; }
            const float x0 = g->gx[k][0], y0 = g->gy[k][0];
            c.drawWideLine(x0 - nx * 6, y0 - ny * 6, x0 + nx * 6, y0 + ny * 6, 3.0f, C_BLACK);
            c.drawWideLine(x0 - nx * 6, y0 - ny * 6, x0 + nx * 6, y0 + ny * 6, 1.5f, col);
        } else {                                                                 // pendant: its pin
            c.fillCircle((int)g->gx[k][0], (int)g->gy[k][0], 4, C_BLACK);
            c.fillCircle((int)g->gx[k][0], (int)g->gy[k][0], 3, col);
        }
    }
    if (e_stroke)                                                                // the lock being drawn
        for (int k = 1; k < e_pn; ++k) c.drawWideLine(e_px[k - 1], e_py[k - 1], e_px[k], e_py[k], 1.5f, C_LIME);
}

void e_draw_lines(M5Canvas& c) {
    static float cx[WPTS_MAX * 8 + 1], cy[WPTS_MAX * 8 + 1];
    for (int i = 0; i < e_nw; ++i) {
        const Water& w = e_w[i];
        for (int l = 0; l < w.nlines; ++l) {
            int n = water_curve(w.line[l], cx, cy, WPTS_MAX * 8 + 1);
            for (int k = 0; k + 1 < n; ++k) {
                c.drawWideLine(cx[k], cy[k], cx[k + 1], cy[k + 1], 2.0f, C_BLACK);
                c.drawWideLine(cx[k], cy[k], cx[k + 1], cy[k + 1], 1.0f, C_YEL);
            }
            if (n >= 2) {                                  // arrowhead: which way it flows
                float dx = cx[n - 1] - cx[n - 2], dy = cy[n - 1] - cy[n - 2], l2 = sqrtf(dx * dx + dy * dy);
                if (l2 > 0.01f) {
                    dx /= l2; dy /= l2;
                    float ex = cx[n - 1], ey = cy[n - 1];
                    c.fillTriangle((int)(ex + dx * 6), (int)(ey + dy * 6), (int)(ex - dy * 4), (int)(ey + dx * 4), (int)(ex + dy * 4), (int)(ey - dx * 4), C_YEL);
                }
            }
        }
    }
    for (int k = 0; k < e_pn; ++k) {                        // the shape / line being tapped in
        if (k) c.drawLine(e_px[k - 1], e_py[k - 1], e_px[k], e_py[k], e_wtool == WT_LINE ? C_YEL : C_LIME);
        c.fillCircle(e_px[k], e_py[k], 2, e_wtool == WT_LINE ? C_YEL : C_LIME);
    }
    if (e_wtool == WT_POLY && e_pn >= 3) c.drawLine(e_px[e_pn - 1], e_py[e_pn - 1], e_px[0], e_py[0], C_DIM);
}

void toast(const char* msg) {
    strncpy(e_toast, msg, sizeof(e_toast) - 1);
    e_toast[sizeof(e_toast) - 1] = 0;
    e_toast_until = millis() + 1600;
    e_dirty = true;
}

void draw_outline(M5Canvas& c, const Zone& z, float k, uint16_t col, float w, bool shadow, bool dotted) {
    if (!g_out_init) {
        for (int s = 0; s < SHAPE_COUNT; ++s) shape_outline(s, OUTLINE_PTS, g_out_u[s], g_out_w[s]);
        g_out_init = true;
    }
    float cs, sn;
    zone_axes(z, cs, sn);
    const float* us = g_out_u[z.shape < SHAPE_COUNT ? z.shape : 0];
    const float* ws = g_out_w[z.shape < SHAPE_COUNT ? z.shape : 0];
    float tu[OUTLINE_PTS], tw[OUTLINE_PTS];
    if (zone_tail(z) != 1.0f) {                       // a shortened teardrop: its own outline
        shape_outline(z.shape, OUTLINE_PTS, tu, tw, zone_tail(z));
        us = tu; ws = tw;
    }
    for (int pass = shadow ? 0 : 1; pass < 2; ++pass) {
        float px, py;
        from_local(z, cs, sn, us[0] * k, ws[0] * k, px, py);
        for (int i = 1; i <= OUTLINE_PTS; ++i) {
            int j = i % OUTLINE_PTS;
            float nx, ny;
            from_local(z, cs, sn, us[j] * k, ws[j] * k, nx, ny);
            if (!dotted || (i & 1)) {
                if (pass == 0) c.drawWideLine(px, py, nx, ny, w + 1.0f, C_BLACK);
                else           c.drawWideLine(px, py, nx, ny, w, col);
            }
            px = nx; py = ny;
        }
    }
}

void tip_screen(const Zone& z, float& x, float& y) {
    float cs, sn;
    zone_axes(z, cs, sn);
    from_local(z, cs, sn, z.tip_u, z.tip_w, x, y);
}

const char* context_hint() {
    if (e_gmode) return e_pgrid ? "Squares flat on the ground  -  A/C turn" : "Grid off  -  tap Grid: on";
    if (e_dmode) {
        switch (e_wtool) {
        case WT_BRUSH: return "Paint what's at this depth  -  A/C size";
        case WT_SMART: return "Tap or paint over it  -  hold: remove";
        case WT_ERASE: return "Erase  -  A/C size";
        case WT_GRID:  return "Squares flat on the ground  -  A/C turn";
        default:       return "Tap around it  -  A undo  -  C fill (fitted)";
        }
    }
    if (e_smode) {
        switch (e_wtool) {
        case WT_BRUSH: return "Paint what sways (hair, cloth...)  -  A/C size";
        case WT_SMART: return "Tap or paint over it  -  hold: remove";
        case WT_ERASE: return "Erase  -  A/C size";
        case WT_LINE: {
            const Sway* h = e_sway();
            if (!h || h->kind == SK_HAIR) return "Drag each lock, root to tip  -  A undo";
            if (h->kind == SK_PENDANT) return "Tap where it hangs from  -  A clear";
            return (h->ng == 0 || h->gn[0] < 2) ? "Tap both ends of the waistband  -  A undo" : "Tap the folds (optional)  -  A undo";
        }
        default:       return "Tap around it  -  A undo  -  C fill (fitted)";
        }
    }
    if (e_wmode) {
        switch (e_wtool) {
        case WT_BRUSH: return "Paint the water  -  A/C brush size  -  B saves";
        case WT_SMART: return "Tap or paint over the water  -  hold: remove";
        case WT_ERASE: return "Erase water off the banks  -  A/C size";
        case WT_POLY:  return "Tap around the water  -  A undo  -  C fill";
        case WT_GRID:  return "Squares flat on the water  -  A/C turn";
        default:       return (e_water() && e_water()->mode == W_OCEAN) ? "Tap along the shore  -  A undo  -  C done"
                                                                      : "Tap along the flow  -  A undo  -  C done";
        }
    }
    if (e_armed >= 0) {
        static char buf[40];
        snprintf(buf, sizeof(buf), "Tap the photo to place: %s", shape_info(e_armed).label);
        return buf;
    }
    if (e_active >= 0) {
        if (tip_used(e_z[e_active])) return "Drag zone, or its point circle";
        return "Drag to move  -  A/C size  -  B saves";
    }
    if (e_n > 0) return "Pick a shape, or tap a zone to edit it";
    return "Pick a shape above";
}

void bar_button(M5Canvas& c, int row, int i, int n, bool on, bool live) {
    int x0 = i * IMG_W / n, x1 = (i + 1) * IMG_W / n, y0 = row_y(row);
    if (on) c.fillRoundRect(x0 + 2, y0 + 2, x1 - x0 - 4, ROW_H - 4, 5, C_TEAL);
    else    c.drawRoundRect(x0 + 2, y0 + 2, x1 - x0 - 4, ROW_H - 4, 5, live ? C_DIM : C_BG);
}

void e_draw() {
    M5Canvas& c = g_scratch;
    memcpy(c.getBuffer(), g_img.getBuffer(), (size_t)IMG_W * IMG_H * 2);
    if (!e_dmode && !e_smode) e_tint_water(c);
    e_tint_planes(c);
    e_tint_sways(c);
    if (!e_dmode && !e_smode) e_draw_lines(c);
    if (e_mm()) e_draw_grid(c);
    if (e_mm() && e_wtool == WT_SMART && t_down && e_marks) {      // the stroke being painted
        uint16_t* buf = (uint16_t*)c.getBuffer();
        for (int i = 0; i < IMG_W * IMG_H; ++i)
            if ((e_marks[i >> 3] >> (i & 7)) & 1) buf[i] = g_swapped ? bswap16(C_YEL) : C_YEL;
    }

    // Zones: thin dotted ring = how far the jiggle reaches, solid line = the object you're fitting.
    for (int i = 0; i < e_n; ++i) {
        bool act = (i == e_active);
        if (act) draw_outline(c, e_z[i], zone_margin(e_z[i]), C_HINT, 0.5f, false, true);
        draw_outline(c, e_z[i], 1.0f, act ? C_LIME : C_TEXT, act ? 1.2f : 0.7f, true, false);
        if (tip_used(e_z[i])) {
            float tx, ty;
            tip_screen(e_z[i], tx, ty);
            float tr = tip_radius(e_z[i]) * e_z[i].r;
            if (tr < 3) tr = 3;
            c.drawCircle((int)lroundf(tx), (int)lroundf(ty), (int)lroundf(tr) + 1, C_BLACK);
            c.drawCircle((int)lroundf(tx), (int)lroundf(ty), (int)lroundf(tr), act ? C_YEL : C_TEXT);
        }
    }
    if (e_active >= 0) {
        int x = (int)lroundf(e_z[e_active].cx), y = (int)lroundf(e_z[e_active].cy);
        c.fillRect(x - 5, y - 1, 11, 3, C_BLACK);
        c.fillRect(x - 1, y - 5, 3, 11, C_BLACK);
        c.drawFastHLine(x - 4, y, 9, C_LIME);
        c.drawFastVLine(x, y - 4, 9, C_LIME);
    }

    c.setFont(&fonts::Font2);
    c.setTextDatum(middle_center);

    // Row 1: shapes + Cancel
    const int by = row_y(0), by2 = row_y(1);
    c.fillRect(0, by, IMG_W, ROW_H, C_BG);
    for (int i = 0; i < TB1_N; ++i) {
        int x0 = i * IMG_W / TB1_N, x1 = (i + 1) * IMG_W / TB1_N;
        int cx = (x0 + x1) / 2, cy = by + ROW_H / 2;
        bool on = (i < SHAPE_COUNT && e_armed == i) || (i == TB1_WATER && e_wmode) || (i == TB1_DEPTH && e_dmode) || (i == TB1_SWAY && e_smode) || (i == TB1_GRID && e_gmode);
        bar_button(c, 0, i, TB1_N, on, true);
        if (i == TB1_WATER || i == TB1_DEPTH || i == TB1_SWAY || i == TB1_GRID) {
            c.setTextColor(on ? C_LIME : C_TEXT);
            c.drawString(i == TB1_WATER ? "Water" : (i == TB1_DEPTH ? "Depth" : (i == TB1_SWAY ? "Sway" : "Grid")), cx, cy);
        } else if (i < SHAPE_COUNT) {
            Zone icon;
            icon.shape = (uint8_t)i;
            icon.cx = (float)cx;
            icon.cy = (float)cy + (i == SHAPE_HALF ? 2.0f : 0.0f);
            icon.r = (i == SHAPE_TEAR) ? 11.0f : 9.0f;
            draw_outline(c, icon, 1.0f, on ? C_LIME : C_TEXT, 0.8f, false, false);
        } else {
            c.setTextColor(C_TEXT);
            c.drawString("Cancel", cx, cy);
        }
    }

    // Row 2 (grid): turn · turn · reset · on/off
    if (e_gmode) {
        c.fillRect(0, by2, IMG_W, ROW_H, C_BG);
        const bool mv = e_pprm[PP_GROUND] >= 0.5f;
        const char* lbl[TB2G_N] = {"Turn <", "Turn >", "Reset", e_pgrid ? "Grid: on" : "Grid: off", mv ? "Moves: on" : "Moves: off"};
        for (int i = 0; i < TB2G_N; ++i) {
            bar_button(c, 1, i, TB2G_N, (i == 3 && e_pgrid) || (i == 4 && mv), true);
            c.setTextColor(C_TEXT);
            c.drawString(lbl[i], (2 * i + 1) * IMG_W / (2 * TB2G_N), by2 + ROW_H / 2);
        }
    }
    // Row 2 (water): tool · mode · rain · region · delete
    if (e_wmode) {
        c.fillRect(0, by2, IMG_W, ROW_H, C_BG);
        const Water* w = e_water();
        char region[32];
        snprintf(region, sizeof(region), "Area %d/%d", e_wi + 1, e_nw);
        const char* lbl[TB2W_N] = {wtool_label(e_wtool), w ? water_mode_label(w->mode) : "-",
                                   (w && w->prm[WP_RAIN] >= 0.5f) ? "Rain: on" : "Rain: off", region, "Del"};
        for (int i = 0; i < TB2W_N; ++i) {
            bar_button(c, 1, i, TB2W_N, (i == 2 && w && w->prm[WP_RAIN] >= 0.5f), true);
            c.setTextColor(C_TEXT);
            c.drawString(lbl[i], (2 * i + 1) * IMG_W / (2 * TB2W_N), by2 + ROW_H / 2);
        }
    }
    // Row 2 (depth): tool · plane · its depth · drift · delete
    if (e_dmode) {
        c.fillRect(0, by2, IMG_W, ROW_H, C_BG);
        const Plane* p = e_plane();
        char pl[32], dp[16];
        snprintf(pl, sizeof(pl), "Plane %d/%d", e_pi + 1, e_np);
        if (e_pgrid && p && !p->fixed) snprintf(dp, sizeof(dp), "On ground");
        else snprintf(dp, sizeof(dp), "Depth %.2f", p ? p->depth : 0.0f);
        const bool drift = e_pprm[PP_AMBIENT] > 0.0f;
        const char* lbl[TB2D_N] = {wtool_label(e_wtool), pl, dp, drift ? "Drift: on" : "Drift: off", "Del"};
        for (int i = 0; i < TB2D_N; ++i) {
            bar_button(c, 1, i, TB2D_N, i == 3 && drift, true);
            c.setTextColor(C_TEXT);
            c.drawString(lbl[i], (2 * i + 1) * IMG_W / (2 * TB2D_N), by2 + ROW_H / 2);
        }
    }
    // Row 2 (hair / cloth / pendants): tool · which · kind · swing · delete
    if (e_smode) {
        c.fillRect(0, by2, IMG_W, ROW_H, C_BG);
        const Sway* h = e_sway();
        char it[24], sw[16];
        if (h && h->name[0]) snprintf(it, sizeof(it), "%.9s", h->name);
        else snprintf(it, sizeof(it), "Item %d/%d", e_si + 1, e_ns);
        const float s = h ? h->prm[SP_SWING] : 1.0f;
        snprintf(sw, sizeof(sw), "Swing: %s", s < 0.75f ? "low" : (s < 1.3f ? "mid" : "high"));
        const char* lbl[TB2S_N] = {wtool_label(e_wtool), it, h ? sway_kind_info(h->kind).label : "-", sw, "Del"};
        for (int i = 0; i < TB2S_N; ++i) {
            bar_button(c, 1, i, TB2S_N, false, true);
            c.setTextColor(C_TEXT);
            c.drawString(lbl[i], (2 * i + 1) * IMG_W / (2 * TB2S_N), by2 + ROW_H / 2);
        }
    }
    // Row 2: selected zone's options
    if (e_active >= 0 && !e_wmode && !e_dmode && !e_smode && !e_gmode) {
        const Zone& z = e_z[e_active];
        c.fillRect(0, by2, IMG_W, ROW_H, C_BG);
        char feel[16];
        snprintf(feel, sizeof(feel), "%s", material_info(z.material).label);
        const char* lbl[TB2_N] = {"Rot", feel, z.tip ? "Point: on" : "Point: off", "Del"};
        for (int i = 0; i < TB2_N; ++i) {
            bar_button(c, 1, i, TB2_N, i == 2 && z.tip, true);
            c.setTextColor(C_TEXT);
            c.drawString(lbl[i], (2 * i + 1) * IMG_W / (2 * TB2_N), by2 + ROW_H / 2);
        }
    }

    // Hint line
    int hy = hint_y();
    bool toasting = e_toast[0] && (int32_t)(millis() - e_toast_until) < 0;
    const char* hint = toasting ? e_toast : context_hint();
    c.fillRect(0, hy, IMG_W, HINT_H, C_BLACK);
    c.setTextColor(toasting ? C_YEL : C_HINT);
    c.drawString(hint, IMG_W / 2, hy + HINT_H / 2);

    // Footer over the physical buttons
    int fy = IMG_H - FOOT_H;
    c.fillRect(0, fy, IMG_W, FOOT_H, C_BLACK);
    if (e_mm()) {
        const bool pts = e_wtool == WT_POLY || e_wtool == WT_LINE;
        c.setTextColor(C_TEXT);
        if (e_wtool == WT_GRID) {
            c.drawString("A  turn", 55, fy + FOOT_H / 2);
            c.drawString("turn  C", 265, fy + FOOT_H / 2);
        } else if (e_wtool == WT_SMART) {
            c.setTextColor(e_glive ? C_TEXT : C_DIM);
            c.drawString("A  smaller", 55, fy + FOOT_H / 2);
            c.drawString("bigger  C", 265, fy + FOOT_H / 2);
        } else {
            c.drawString(pts ? "A  undo" : "A  smaller", 55, fy + FOOT_H / 2);
            if (!(pts && e_smode)) c.drawString(pts ? (e_wtool == WT_POLY ? "fill  C" : "done  C") : "bigger  C", 265, fy + FOOT_H / 2);
        }
        if (!pts && e_wtool != WT_SMART && e_wtool != WT_GRID) {                   // brush size preview
            c.drawCircle(IMG_W - 14, fy - 14, e_brush < 12 ? e_brush : 12, C_LIME);
        }
    } else {
        c.setTextColor(e_active >= 0 ? C_TEXT : C_DIM);
        c.drawString("A  smaller", 55, fy + FOOT_H / 2);
        c.drawString("bigger  C", 265, fy + FOOT_H / 2);
    }
    c.setTextColor(C_LIME);
    c.drawString("B  save", 160, fy + FOOT_H / 2);

    c.pushSprite(0, 0);
    e_dirty = false;
}

void scale_active(int dir) {
    if (e_active < 0) { toast("Select or place a zone first"); return; }
    Zone& z = e_z[e_active];
    float nr = dir > 0 ? z.r * 1.06f : z.r / 1.06f;
    if (fabsf(nr - z.r) < 1.0f) nr = z.r + (dir > 0 ? 1.0f : -1.0f);
    z.r = clampf(nr, R_MIN, R_MAX);
    e_last_r = z.r;
    e_dirty = true;
}

void wait_release() {
    for (int i = 0; i < 300; ++i) {
        M5.update();
        auto t = M5.Touch.getDetail();
        if (!t.isPressed() && !M5.BtnA.isPressed() && !M5.BtnB.isPressed() && !M5.BtnC.isPressed()) break;
        delay(10);
    }
}

void e_close(bool save) {
    if (save) {
        for (int i = 0; i < g_n; ++i) free_map(g_z[i]);
        g_n = e_n;
        for (int i = 0; i < g_n; ++i) {
            g_z[i] = e_z[i];
            g_z[i].wmap = nullptr;
            rebuild(g_z[i]);
            g_shown[i] = false;
        }
        clear_waters();
        g_nw = 0;
        for (int i = 0; i < e_nw; ++i) {
            bool any = false;
            for (int k = 0; k < MASK_BYTES && e_w[i].mask && !any; ++k) any = e_w[i].mask[k] != 0;
            if (!any) continue;                           // an empty area isn't kept
            water_copy_def(g_wat[g_nw], e_w[i], water_alloc);
            g_wat[g_nw].grid = e_pgrid;                   // the water lies on the ground: its perspective
            if (e_pgrid) memcpy(g_wat[g_nw].gq, e_gq, sizeof(e_gq));
            ++g_nw;
        }
        for (int i = 0; i < g_ns; ++i) sway_free(g_sway[i]);   // (the body lines stay)
        g_ns = 0;
        for (int k = 0; k < e_ns; ++k) {
            if (sway_count(e_s[k]) < 12) continue;       // an empty one isn't kept
            if (e_s[k].kind == SK_CLOTH && e_s[k].ng > 0 && e_s[k].gn[0] < 2) e_s[k].ng = 0;   // (half a waistband: its own)
            sway_default_guides(e_s[k]);                 // (no guides drawn: the ones shown dimmed)
            sway_copy_def(g_sway[g_ns], e_s[k], water_alloc);
            ++g_ns;
        }
        parallax_free(g_par);
        g_par = Parallax();
        for (int p = 0; p < PP_COUNT; ++p) g_par.prm[p] = e_pprm[p];
        g_par.grid = e_pgrid; memcpy(g_par.gq, e_gq, sizeof(e_gq));
        for (int k = 0; k < e_np; ++k) {
            if (!plane_any(e_p[k])) continue;           // an empty plane isn't kept
            plane_copy_def(g_par.pl[g_par.n], e_p[k], water_alloc);
            ++g_par.n;
        }
        for (int i = 0; i < g_ns; ++i) {                  // each figure carries its hair / cloth (nothing of her left behind)
            const Sway& sw = g_sway[i];
            if (!sw.mask || g_par.n == 0) continue;
            int best = -1, bo = 0;
            for (int k = 0; k < g_par.n; ++k) {
                const Plane& pl = g_par.pl[k];
                if (!pl.mask) continue;
                if (sw.on[0] && strncmp(sw.on, pl.name, sizeof(pl.name)) == 0) { best = k; break; }
                int o = 0;
                for (int b = 0; b < MASK_BYTES; ++b) o += __builtin_popcount(pl.mask[b] & sw.mask[b]);
                if (!sw.on[0] && o > bo) { bo = o; best = k; }
            }
            if (best >= 0) for (int b = 0; b < MASK_BYTES; ++b) g_par.pl[best].mask[b] |= sw.mask[b];
        }
        prepare_sways();                                  // (they rest on the figures: after the planes)
        prepare_waters();
        save_zones();
    }
    for (int i = 0; i < e_nw; ++i) water_free(e_w[i]);
    for (int k = 0; k < e_np; ++k) plane_free(e_p[k]);
    e_np = 0; e_pi = -1; e_dmode = false;
    for (int k = 0; k < e_ns; ++k) sway_free(e_s[k]);
    e_ns = 0; e_si = -1; e_smode = false;
    e_smart_free();
    e_nw = 0; e_wi = -1; e_wmode = false; e_pn = 0;
    layer_sync();                                     // the editor drew on the canvas the layer may use
    prepare_parallax();                               // (its content mirrors the layer)
    e_open = false;
    present();
    rest_all();
    if (save) kick_all();          // show it off straight away
    wait_release();
    g_last_us = micros();
}

void e_delete_active() {
    if (e_active < 0) { toast("Tap a zone to select it"); return; }
    for (int i = e_active; i < e_n - 1; ++i) e_z[i] = e_z[i + 1];
    --e_n;
    e_active = -1;
    e_dirty = true;
}

void e_bar_tap(int x, int y, uint32_t held) {
    if (y >= row_y(0) && y < row_y(0) + ROW_H) y = y - row_y(0);                       // row 1
    else if (bar_bottom() > ROW_H && y >= row_y(1) && y < row_y(1) + ROW_H) y = ROW_H + y - row_y(1);   // row 2
    else {                                             // the hint line: move the bars to the other end
        e_low = !e_low;
        toast(e_low ? "Bars at the bottom - tap this line to move them" : "Bars at the top");
        e_dirty = true;
        return;
    }
    if (y < ROW_H) {
        int i = x * TB1_N / IMG_W;
        if (i == TB1_GRID) {                          // the ground grid on / off
            e_gmode = !e_gmode;
            e_wmode = false; e_dmode = false; e_smode = false;
            e_armed = -1; e_active = -1; e_pn = 0; e_glive = false;
            if (e_gmode) {
                e_wtool = WT_GRID;
                if (!e_pgrid) {
                    e_grid_default();
                    bool planes = false;
                    for (int k = 0; k < e_np; ++k) planes |= plane_any(e_p[k]);
                    if (!planes) e_pprm[PP_GROUND] = 0.0f;   // (just the grid: the ground doesn't start sliding)
                }
                toast("Drag the corners: squares flat on the ground");
            } else e_wtool = WT_SMART;
            e_dirty = true;
            return;
        }
        if (e_gmode) { e_gmode = false; e_wtool = WT_SMART; }
        if (i == TB1_SWAY) {                          // hanging things on / off
            e_smode = !e_smode;
            e_wmode = false; e_dmode = false;
            e_armed = -1; e_active = -1; e_pn = 0; e_glive = false;
            if (e_wtool == WT_GRID) e_wtool = WT_SMART;
            if (e_smode && e_ns == 0) e_new_sway();
            if (e_smode) toast(e_low ? "Select it, then its guides (Tool)" : "At the top? Tap this line: bars go down");
            e_dirty = true;
            return;
        }
        if (i == TB1_DEPTH) {                         // depth planes on / off
            e_dmode = !e_dmode;
            e_wmode = false; e_smode = false;
            e_armed = -1; e_active = -1; e_pn = 0; e_glive = false;
            if (e_wtool == WT_LINE) e_wtool = WT_SMART;
            if (e_dmode && e_np == 0) e_new_plane();
            if (e_dmode) toast(e_pgrid ? "Select what stands on the ground (Smart)" : "Tap what's nearest (Smart), or Shape around it");
            e_dirty = true;
            return;
        }
        if (i == TB1_WATER) {                         // water editing on / off
            e_dmode = false; e_smode = false; e_glive = false;
            e_wmode = !e_wmode;
            e_armed = -1; e_active = -1; e_pn = 0;
            if (e_wmode && e_nw == 0) e_new_water();
            if (e_wmode) toast("Paint over the water (or Shape: tap around it)");
            e_dirty = true;
            return;
        }
        if (i < SHAPE_COUNT) {
            e_wmode = false; e_dmode = false; e_smode = false; e_pn = 0;
            if (e_armed == i) { e_armed = -1; e_dirty = true; return; }
            if (e_n >= MAX_ZONES) { toast("4 zones max - Del one first"); return; }
            e_armed = i;
            e_dirty = true;
        } else {
            e_close(false);
        }
        return;
    }
    if (e_gmode) {
        switch (x * TB2G_N / IMG_W) {
        case 0: e_grid_turn(-2.0f); break;
        case 1: e_grid_turn(+2.0f); break;
        case 2: e_grid_default(); toast("Grid reset"); break;
        case 3:
            e_pgrid = !e_pgrid;
            if (e_pgrid && e_gq[0] == 0 && e_gq[1] == 0 && e_gq[2] == 0) e_grid_default();
            toast(e_pgrid ? "Grid on" : "Grid off - flat water, planes at their own depth");
            break;
        case 4:                                   // does the ground itself slide when you tilt?
            e_pprm[PP_GROUND] = e_pprm[PP_GROUND] >= 0.5f ? 0.0f : 1.0f;
            toast(e_pprm[PP_GROUND] >= 0.5f ? "The ground slides too when you tilt" : "The ground stays put (planes still stand on it)");
            break;
        }
        e_dirty = true;
        return;
    }
    if (e_smode) {
        Sway* h = e_sway();
        switch (x * TB2S_N / IMG_W) {
        case 0:                                   // tool: Paint → Smart → Erase → Shape → its guides
            e_wtool = (uint8_t)((e_wtool + 1) % WT_COUNT);
            if (e_wtool == WT_GRID) e_wtool = WT_BRUSH;
            e_pn = 0; e_glive = false; e_stroke = false;
            if (e_wtool == WT_LINE && h)
                toast(h->kind == SK_HAIR ? "Drag along each lock, from its root" : (h->kind == SK_CLOTH ? "Tap the waistband's two ends" : "Tap where it hangs from"));
            break;
        case 1:                                   // next one (hold: a new one)
            if (held >= 500) { if (!e_new_sway()) toast("4 max - Del one first"); else toast("New - select it"); }
            else if (e_ns > 0) e_si = (e_si + 1) % e_ns;
            e_pn = 0; e_glive = false; e_stroke = false;
            break;
        case 2:                                   // hair → cloth → pendant (its guides start over)
            if (h) {
                h->kind = (uint8_t)((h->kind + 1) % SK_COUNT);
                h->ng = 0; memset(h->gn, 0, sizeof(h->gn));
                h->lax = h->lay = h->lbx = h->lby = -1;
                toast(h->kind == SK_HAIR ? "Hair: draw its locks (Tool)" : (h->kind == SK_CLOTH ? "Cloth: its waistband (Tool)" : "Pendant: its pin (Tool)"));
            }
            break;
        case 3:                                   // swing: low → mid → high
            if (h) h->prm[SP_SWING] = h->prm[SP_SWING] < 0.75f ? 1.0f : (h->prm[SP_SWING] < 1.3f ? 1.6f : 0.5f);
            break;
        case 4:                                   // delete this one
            if (h) {
                sway_free(*h);
                for (int k = e_si; k < e_ns - 1; ++k) { e_s[k] = e_s[k + 1]; new (&e_s[k + 1]) Sway(); }
                --e_ns;
                if (e_ns == 0) e_new_sway(); else if (e_si >= e_ns) e_si = e_ns - 1;
                toast("Deleted");
                e_glive = false;
            }
            break;
        }
        e_dirty = true;
        return;
    }
    if (e_dmode) {
        Plane* p = e_plane();
        switch (x * TB2D_N / IMG_W) {
        case 0:                                   // tool: Paint → Smart → Erase → Shape (the grid: Grid button)
            e_wtool = (uint8_t)((e_wtool + 1) % WT_COUNT);
            if (e_wtool == WT_LINE || e_wtool == WT_GRID) e_wtool = WT_BRUSH;
            e_pn = 0; e_glive = false;
            e_tool_picked();
            break;
        case 1:                                   // next plane (hold: a new one)
            if (held >= 500) { if (!e_new_plane()) toast("4 planes max"); else toast("New plane"); }
            else if (e_np > 0) e_pi = (e_pi + 1) % e_np;
            e_pn = 0; e_glive = false;
            break;
        case 2:                                   // its depth: far → near
            // its depth: with a grid it stands where it meets the ground, or (tap on) set by hand
            // 0.15 … 0.9, then back on the ground. Without a grid: by hand only.
            if (p) {
                if (e_pgrid && !p->fixed) { p->fixed = true; p->depth = DEPTHS[0]; toast("Depth by hand (it doesn't stand on the ground)"); }
                else {
                    int k = 0;
                    while (k < 5 && DEPTHS[k] < p->depth - 0.01f) ++k;
                    if (e_pgrid && k >= 4) { p->fixed = false; toast("Stands on the ground: depth from where it meets it"); }
                    else p->depth = DEPTHS[(k + 1) % 5];
                    if (!e_pgrid) p->fixed = false;
                }
                p->ramp = false;
            }
            break;
        case 3:                                   // drift while still
            e_pprm[PP_AMBIENT] = e_pprm[PP_AMBIENT] > 0.0f ? 0.0f : pparam_info(PP_AMBIENT).dflt;
            break;
        case 4:                                   // delete this plane
            if (p) {
                plane_free(*p);
                for (int k = e_pi; k < e_np - 1; ++k) { e_p[k] = e_p[k + 1]; e_p[k + 1] = Plane(); }
                --e_np;
                if (e_np == 0) e_new_plane(); else if (e_pi >= e_np) e_pi = e_np - 1;
                toast("Plane deleted");
                e_glive = false;
            }
            break;
        }
        e_dirty = true;
        return;
    }
    if (e_wmode) {
        Water* w = e_water();
        switch (x * TB2W_N / IMG_W) {
        case 0:                                   // tool: Paint → Smart → Erase → Shape → Flow/Shore line (grid: Grid button)
            e_wtool = (uint8_t)((e_wtool + 1) % WT_COUNT);
            if (e_wtool == WT_GRID) e_wtool = WT_BRUSH;
            e_glive = false;
            e_pn = 0;
            e_tool_picked();
            break;
        case 1:                                   // still → river → ocean
            if (w) {
                w->mode = (uint8_t)((w->mode + 1) % W_MODE_COUNT);
                if (w->mode != W_STILL && w->nlines == 0)
                    toast(w->mode == W_RIVER ? "Add a Flow line along the river" : "Add a Shore line where waves land");
            }
            break;
        case 2:                                   // rain on / off
            if (w) w->prm[WP_RAIN] = w->prm[WP_RAIN] >= 0.5f ? 0.0f : 1.0f;
            break;
        case 3:                                   // next area (hold: a new one)
            if (held >= 500) { if (!e_new_water()) toast("2 water areas max"); else toast("New water area"); }
            else if (e_nw > 0) e_wi = (e_wi + 1) % e_nw;
            e_pn = 0; e_glive = false;
            break;
        case 4:                                   // delete this area
            if (w) {
                water_free(*w);
                for (int k = e_wi; k < e_nw - 1; ++k) { e_w[k] = e_w[k + 1]; e_w[k + 1] = Water(); }
                --e_nw;
                if (e_nw == 0) e_new_water(); else if (e_wi >= e_nw) e_wi = e_nw - 1;
                toast("Water area deleted");
                e_glive = false;
            }
            break;
        }
        e_dirty = true;
        return;
    }
    if (e_active < 0) return;
    Zone& z = e_z[e_active];
    switch (x * TB2_N / IMG_W) {
    case 0:                                   // Rot: tap = 15°, hold = 90°
        z.rot = (int16_t)((z.rot + (held >= 500 ? 90 : 15)) % 360);
        break;
    case 1:                                   // Feel: Soft → Natural → Firm
        z.material = (uint8_t)((z.material + 1) % MAT_COUNT);
        e_last_mat = z.material;
        break;
    case 2:                                   // control point: line it up over the apex
        z.tip = !z.tip;
        if (z.tip) clamp_point(z);
        if (z.tip) toast("Drag the circle onto the apex");
        break;
    case 3:
        e_delete_active();
        return;
    }
    e_dirty = true;
}

void e_image_tap(int x, int y, uint32_t held) {
    (void)held;
    if (e_mm()) {
        if (e_smode && e_wtool == WT_LINE) {           // guides by tap: cloth's waistband and folds, a pendant's pin
            Sway* h = e_sway();
            if (h && h->kind == SK_PENDANT) {
                h->ng = 1; h->gn[0] = 1; h->gx[0][0] = (float)x; h->gy[0][0] = (float)y;
                toast("Pinned - it swings from here");
            } else if (h && h->kind == SK_CLOTH) {
                if (h->ng == 0) { h->ng = 1; h->gn[0] = 1; h->gx[0][0] = (float)x; h->gy[0][0] = (float)y; toast("Now the waistband's other end"); }
                else if (h->gn[0] < 2) { h->gn[0] = 2; h->gx[0][1] = (float)x; h->gy[0][1] = (float)y; toast("Waistband set - tap folds, or done"); }
                else if (h->ng < SG_MAX) { h->gn[h->ng] = 1; h->gx[h->ng][0] = (float)x; h->gy[h->ng][0] = (float)y; ++h->ng; toast("Fold marked"); }
                else toast("That's the most folds");
            } else toast("Drag along the lock, from its root to its tip");
            e_dirty = true;
            return;
        }
        if (e_wtool == WT_GRID) {
            bool* on = e_grid_on();
            if (on && *on && held >= 600) { *on = false; toast("Grid off - flat water, planes at their own depth"); }
            else if (on && !*on) { e_grid_default(); toast("Drag corners to fit - hold: grid off"); }
        }
        if (e_wtool == WT_POLY || e_wtool == WT_LINE) {
            int maxp = e_wtool == WT_LINE ? WPTS_MAX : 32;
            if (e_pn < maxp) { e_px[e_pn] = (int16_t)x; e_py[e_pn] = (int16_t)y; ++e_pn; }
            else toast("That's the most points");
        }
        e_dirty = true;                            // (brush taps paint on touch-down)
        return;
    }
    if (e_armed >= 0) {
        if (e_n < MAX_ZONES) {
            Zone z;
            z.shape = (uint8_t)e_armed;
            z.material = e_last_mat;
            z.cx = (float)x;
            z.cy = (float)y;
            z.r = e_last_r;
            z.rot = 0;
            shape_tip_default(z.shape, z.tip_u, z.tip_w);
            device_boost(z);                           // made on the Core2: start out strong
            e_z[e_n] = z;
            e_active = e_n++;
        }
        e_armed = -1;
        e_dirty = true;
        return;
    }
    int hit = -1;
    for (int i = e_n - 1; i >= 0; --i) {
        if (i != e_active && zone_contains(e_z[i], (float)x, (float)y)) { hit = i; break; }
    }
    if (hit >= 0) { e_active = hit; e_dirty = true; return; }
    if (e_active >= 0) {
        e_z[e_active].cx = (float)x;
        e_z[e_active].cy = (float)y;
        e_dirty = true;
        return;
    }
    toast("Pick a shape in the top bar");
}

bool near_tip(int x, int y) {
    if (e_active < 0 || !tip_used(e_z[e_active])) return false;
    float tx, ty;
    tip_screen(e_z[e_active], tx, ty);
    float rr = tip_radius(e_z[e_active]) * e_z[e_active].r + 8.0f;
    return (x - tx) * (x - tx) + (y - ty) * (y - ty) <= rr * rr;
}

// Drag the tip dot: convert the screen delta to local units and keep it inside the bulb.
void move_tip(int dx, int dy) {
    Zone& z = e_z[e_active];
    float cs, sn;
    zone_axes(z, cs, sn);
    float du = (dx * cs + dy * sn) / z.r;
    float dw = -(-dx * sn + dy * cs) / z.r;
    float nu = clampf(z.tip_u + du, -0.6f, 0.6f), nw = clampf(z.tip_w + dw, -1.0f, 0.9f);
    if (point_ok(z, nu, nw)) { z.tip_u = nu; z.tip_w = nw; }
}

template <typename Btn>
void e_scale_button(Btn& b, uint32_t& next, int dir, uint32_t now) {
    if (b.wasPressed()) {
        scale_active(dir);
        next = now + 380;
    } else if (b.isPressed() && (int32_t)(now - next) >= 0) {
        scale_active(dir);
        next = now + 60;
    }
}

}  // namespace

// ================================================================================================
// Public API
// ================================================================================================

bool begin() {
    if (!g_sway) {                                     // (hair / cloth / pendants: ~7 KB each - PSRAM, not DRAM)
        g_sway = (Sway*)alloc_psram(sizeof(Sway) * SWAY_MAX);
        e_s = (Sway*)alloc_psram(sizeof(Sway) * SWAY_MAX);
        e_sdflt = (Sway*)alloc_psram(sizeof(Sway));
        if (!g_sway || !e_s || !e_sdflt) return false;
        for (int i = 0; i < SWAY_MAX; ++i) { new (&g_sway[i]) Sway(); new (&e_s[i]) Sway(); }
        new (e_sdflt) Sway();
    }
    g_img.setColorDepth(16);
    g_img.setPsram(true);
    g_scratch.setColorDepth(16);
    g_scratch.setPsram(true);
    if (!g_img.createSprite(IMG_W, IMG_H)) return false;
    if (!g_scratch.createSprite(IMG_W, IMG_H)) return false;

    // Learn the canvas byte order once: draw pure red (RGB565 0xF800) and look at the raw word.
    g_img.drawPixel(0, 0, (uint16_t)0xF800);
    uint16_t raw = ((const uint16_t*)g_img.getBuffer())[0];
    g_swapped = (raw != 0xF800);
    g_img.fillScreen(C_BG);

    // Strip buffers in DMA-capable internal RAM (fall back to plain pushes if that fails).
    for (int i = 0; i < 2; ++i)
        g_strip[i] = (uint16_t*)heap_caps_malloc(STRIP_PIX * 2, MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
    g_dma = g_strip[0] && g_strip[1];
    if (!g_dma) {
        for (int i = 0; i < 2; ++i) {
            if (g_strip[i]) free(g_strip[i]);
            g_strip[i] = (uint16_t*)alloc_psram(STRIP_PIX * 2);
        }
        if (!g_strip[0] || !g_strip[1]) return false;
    }
    // A copy of the photo in internal RAM makes the warp's reads several times faster. Only if
    // there's comfortably enough room left for everything else.
    size_t img_bytes = (size_t)IMG_W * IMG_H * 2;
    if (heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT) > img_bytes + 48 * 1024)
        g_src_fast = (uint16_t*)heap_caps_malloc(img_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    g_rhythm_mic.set_music(true);
    g_rhythm_mic.set_lead(MIC_LEAD);
    g_motion_echo.fade_s = ECHO_FADE_S;      // motion echo fades over this after you stop

    wsin_tab();                                       // build the sine table before two cores share it
    par_tab();                                        // parallax filter weights (likewise)
    par_start();
#if defined(ESP_PLATFORM)
    {   // display write clock: make sure it's at LCD_WRITE_HZ (M5GFX normally sets 40 MHz on the Core2)
        auto* bus = static_cast<lgfx::Bus_SPI*>(M5.Display.getPanel()->getBus());
        if (bus && M5.getBoard() == m5::board_t::board_M5StackCore2) {
            auto cfg = bus->config();
            const uint32_t was = cfg.freq_write;
            if (was < LCD_WRITE_HZ) {
                cfg.freq_write = LCD_WRITE_HZ;
                bus->config(cfg);
                bus->init();
            }
            Serial.printf("jig: LCD SPI write %u Hz (was %u)\n", (unsigned)cfg.freq_write, (unsigned)was);
        }
    }
#endif
    Serial.printf("jig: ready (swapped=%d imu=%d dma=%d fast-src=%d)\n", (int)g_swapped,
                  (int)M5.Imu.isEnabled(), (int)g_dma, g_src_fast != nullptr);
    g_ok = true;
    g_last_us = micros();
    return true;
}

M5Canvas& image() { return g_img; }
bool image_swapped() { return g_swapped; }

void present() {
    if (!g_ok) return;
    if (g_src_fast) {
        const uint16_t* s = (const uint16_t*)g_img.getBuffer();
        size_t n = (size_t)IMG_W * IMG_H;
        if (g_swapped) memcpy(g_src_fast, s, n * 2);
        else for (size_t i = 0; i < n; ++i) g_src_fast[i] = bswap16(s[i]);   // keep display order
    }
    g_img.pushSprite(0, 0);
    if (g_front) {                                    // depth: the screen now shows the plain photo
        const uint16_t* s = (const uint16_t*)g_img.getBuffer();
        const size_t n = (size_t)IMG_W * IMG_H;
        if (g_swapped) memcpy(g_front, s, n * 2);
        else for (size_t i = 0; i < n; ++i) g_front[i] = bswap16(s[i]);
        g_par.dv[0] = g_par.dv[1] = 0;                // …i.e. at the centred view: redraw what differs
        parallax_forget_screen(g_par);
        g_par.drawn = true;
        g_par_moving = true;
    } else {
        g_par.drawn = false;
    }
}

void clear_zones() {
    for (int i = 0; i < g_n; ++i) free_map(g_z[i]);
    g_n = 0;
    clear_sways();
    clear_waters();
    clear_parallax();
    g_keep_len = 0;
    g_jig_path[0] = 0;
}

int zone_count() { return g_n; }

void load_zones(const char* image_path) {
    frame_reset();                   // (no frame in progress for the old picture)
    clear_zones();
    if (!g_ok || !image_path) return;
    // Hidden sidecar: "/images/santa.mg" → "/images/.santa.mg.jig" (dot-files are skipped by scans).
    const char* slash = strrchr(image_path, '/');
    if (slash) snprintf(g_jig_path, sizeof(g_jig_path), "%.*s.%s.jig", (int)(slash - image_path + 1), image_path, slash + 1);
    else       snprintf(g_jig_path, sizeof(g_jig_path), ".%s.jig", image_path);
    if (!SD.exists(g_jig_path)) return;
    File f = SD.open(g_jig_path, FILE_READ);
    if (!f) return;
    // Line by line, any length of file (water masks make it a few KB). Lines longer than the buffer
    // are cut (the rest of that line is skipped).
    static char line[1400];
    static uint8_t chunk[512];
    int len = 0, line_no = 0, wcur = -1, pcur = -1, hcur = -1;
    bool ok = true, skipping = false, done = false;
    auto take_line = [&](char* ln) {
        if (line_no++ == 0) { if (strncmp(ln, "MGJIG1", 6) != 0) ok = false; return; }
        if (!ok) return;
        if (parse_water_line(ln, g_wat, g_nw, wcur)) return;
        if (parse_parallax_line(ln, g_par, pcur, water_alloc)) return;
        if (parse_sway_line(ln, g_sway, g_ns, hcur, g_body, water_alloc)) { g_body_saved |= g_body != nullptr; return; }
        Zone z;
        if (!parse_zone(ln, z)) { keep_line(ln); return; }
        if (g_n >= MAX_ZONES) return;
        g_z[g_n] = z;
        rebuild(g_z[g_n]);
        g_shown[g_n] = false;
        g_hold[g_n][0] = g_hold[g_n][1] = g_hold[g_n][2] = 0;
        ++g_n;
    };
    while (!done && ok) {
        size_t got = f.read(chunk, sizeof(chunk));
        if (got == 0) done = true;
        for (size_t i = 0; i < got; ++i) {
            char ch = (char)chunk[i];
            if (ch == '\n' || ch == '\r') {
                if (!skipping && len > 0) { line[len] = 0; take_line(line); }
                len = 0; skipping = false;
            } else if (!skipping) {
                if (len < (int)sizeof(line) - 1) line[len++] = ch;
                else { line[len] = 0; take_line(line); len = 0; skipping = true; }
            }
        }
    }
    if (len > 0 && !skipping && ok) { line[len] = 0; take_line(line); }
    f.close();
    prepare_sways();                                   // (rests on the body lines, else the depth figures' masks)
    prepare_waters();
    prepare_parallax();
    Serial.printf("jig: %d zone(s), %d water, %d depth plane(s), %d hair / cloth / pendant from %s\n", g_n, g_nw, g_par.n, g_ns, g_jig_path);
}

void play_tick(uint32_t now, bool may_draw) {
    uint32_t us = micros();
    g_tick_us0 = us;
    g_tick_did = false;
    float dt = (us - g_last_us) * 1e-6f;
    g_last_us = us;
    if (!g_ok) return;
    if (dt > 0.05f) dt = 0.05f;
    if (dt <= 0.0f) return;

    // ---- IMU: gravity (pinned to the real one), jolts, twist ---------------------------------
    float a[3] = {0, 0, 0};
    bool have = M5.Imu.isEnabled();
    if (have) {
        if (M5.Imu.update()) {
            auto d = M5.Imu.getImuData();
            g_acc_last[0] = d.accel.x; g_acc_last[1] = d.accel.y; g_acc_last[2] = d.accel.z;
            g_gyro_last = d.gyro.z;
            g_gyro_xy[0] = d.gyro.x; g_gyro_xy[1] = d.gyro.y;
        }
        a[0] = IMU_SX * g_acc_last[0]; a[1] = IMU_SY * g_acc_last[1]; a[2] = IMU_SZ * g_acc_last[2];
        if (!g_init) {
            float l0 = sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); if (l0 < 0.3f) l0 = 1.0f;
            for (int i = 0; i < 3; ++i) { g_grav[i] = a[i]; g_dn_slow[i] = -a[i] / l0; }
            g_init = true;
        }
        float k = 1.0f - expf(-dt / GRAV_TAU);
        for (int i = 0; i < 3; ++i) g_grav[i] += (a[i] - g_grav[i]) * k;
        float w = GYRO_CW * g_gyro_last * 0.017453293f;
        g_alpha += ((w - g_w) / dt - g_alpha) * (1.0f - expf(-dt / 0.03f));
        g_w = w;
    } else {
        a[1] = -1.0f; g_grav[0] = 0; g_grav[1] = -1.0f; g_grav[2] = 0;
    }
    float dyn[3] = {a[0] - g_grav[0], a[1] - g_grav[1], a[2] - g_grav[2]};   // jolts, flicks, swings
    float gl = sqrtf(g_grav[0] * g_grav[0] + g_grav[1] * g_grav[1] + g_grav[2] * g_grav[2]);
    if (gl < 0.3f) gl = 1.0f;

    // ---- Motion echo: a repeating motion carries on after you stop, fading out ------------------
    float dn_live[3] = {-g_grav[0] / gl, -g_grav[1] / gl, -g_grav[2] / gl};
    float gdev[3];
    {
        const float k = 1.0f - expf(-dt / ECHO_SLOW_TAU);
        for (int i = 0; i < 3; ++i) { g_dn_slow[i] += (dn_live[i] - g_dn_slow[i]) * k; gdev[i] = dn_live[i] - g_dn_slow[i]; }
    }
    if (have) g_motion_echo.update(dt, dyn, gdev, g_w);
    for (int i = 0; i < g_nw; ++i)                     // water: real gravity + your jolts
        if (g_wat[i].edge) water_step(g_wat[i], dt, dn_live, dyn);
    // Swayed back and forth and stopped: the forms keep swaying on their own springs, dying out
    // slowly (damping eased off, coming back over ECHO_FADE_S). Nothing is replayed.
    g_echo = g_motion_echo.strength();
    for (int i = 0; i < g_n; ++i) g_z[i].damp_k = 1.0f - ECHO_DAMP_CUT * g_echo;
    g_sway_damp = 1.0f - ECHO_DAMP_CUT * g_echo;
    for (int i = 0; i < g_ns; ++i) {                   // hair / cloth / pendants: real gravity + your jolts (+ the echo)
        Sway& sw = g_sway[i];
        if (!sw.ready) continue;
        sw.npush = 0;
        if (sw.kind == SK_CLOTH)                       // forms jiggling under a skirt carry it along
            for (int k = 0; k < g_n && sw.npush < SW_PUSH_MAX; ++k) {
                const Zone& z = g_z[k];
                if (!z.wmap) continue;
                float* P = sw.push[sw.npush++];
                P[0] = z.cx + z.dx; P[1] = z.cy + z.dy; P[2] = z.r; P[3] = z.vx; P[4] = z.vy;
            }
        sway_step(sw, dt, dn_live, dyn, g_sway_damp, g_sdf);   // (redrawn once it has moved enough: sway_due)
    }
    bool echoing = g_echo > 0.02f;
    if (echoing != g_was_echo) {
        g_was_echo = echoing;
        Serial.printf(echoing ? "jig: still swaying after your %.1f s sway\n" : "jig: sway died out\n", g_motion_echo.period());
    }
    float motion[3];
    for (int k = 0; k < 3; ++k) motion[k] = dyn[k];

    // ---- Dance = music, only when you've switched it on (swipe up). ----------------------------
    // It dances only to a steady tempo the beat tracker has locked onto: a quiet room, talking, a fan
    // or clicks never lock, so they never kick anything. Before the lock nothing moves.
    bool mic_beat = false;
    g_mic_new = false;
    const bool want_mic = MIC_DANCE && g_dance_on && (g_n > 0 || g_ns > 0);
    if (want_mic && !g_mic_on) {
        if (M5.Speaker.isEnabled()) M5.Speaker.end(); // Core2: mic and speaker share one I2S port
        g_mic_on = M5.Mic.isEnabled() || M5.Mic.begin();
        g_mic_busy = false;
        g_onset = AudioOnset();                       // learn this room afresh (the beat tracker unlocks by itself)
        Serial.printf("jig: dance on, microphone %s\n", g_mic_on ? "listening" : "unavailable");
    } else if (!want_mic && g_mic_on) {
        for (int k = 0; k < 40 && M5.Mic.isRecording(); ++k) delay(1);   // (let the last chunk finish)
        M5.Mic.end();
        g_mic_on = false; g_mic_busy = false;
        Serial.println("jig: dance off, microphone off");
    }
    if (g_mic_on) {                                   // record-and-poll, one 20 ms chunk at a time
        if (!g_mic_busy) {
            g_mic_busy = M5.Mic.record(g_mic_buf, MIC_N, MIC_SR);
            g_mic_at = now;
        } else if (now - g_mic_at >= 18 && M5.Mic.isRecording() == 0) {
            g_mic_hit = g_onset.process(g_mic_buf, MIC_N, MIC_SR);
            g_mic_hold = (float)MIC_N / MIC_SR;
            g_mic_new = true;
            g_mic_busy = M5.Mic.record(g_mic_buf, MIC_N, MIC_SR);
            g_mic_at = now;
        }
        float on = g_mic_hold > 0 ? (dt < g_mic_hold ? 1.0f : g_mic_hold / dt) : 0.0f;
        mic_beat = g_rhythm_mic.update(dt, 0, 0, g_mic_hit * on, 0, 0, 1);
        g_mic_hold -= dt;
    }
    const bool music = g_mic_on && g_onset.music() && g_rhythm_mic.steady();
    g_dance = music ? (g_dance + dt < 1.0f ? g_dance + dt : 1.0f) : (g_dance - dt * 0.5f > 0.0f ? g_dance - dt * 0.5f : 0.0f);
    if (music != g_was_music) {
        g_was_music = music;
        if (music) Serial.printf("jig: music at %.0f BPM - dancing (level %.0f dB, floor %.0f dB)\n", g_rhythm_mic.bpm(), g_onset.level_db(), g_onset.floor_db());
        else Serial.printf("jig: no steady beat - not dancing (level %.0f dB, floor %.0f dB)\n", g_onset.level_db(), g_onset.floor_db());
    }
    const bool beat = music && mic_beat;
    const int sign = g_rhythm_mic.beat_sign();

    if (g_n == 0 && g_nw == 0 && g_ns == 0 && !g_par.ready) return;   // nothing to animate (water / depth / sways alone still do)

    if (beat)                                          // hair / cloth / pendants: the same small nudge, only if near rest
        for (int i = 0; i < g_ns; ++i)
            if (g_sway[i].ready && g_sway[i].disp < DANCE_SWAY_ROOM) { sway_kick(g_sway[i], DANCE_GAIN * g_dance * g_sway_damp, sign); g_sway[i].drawn = false; }
    if (beat && g_n > 0) {
        float cen = 0;
        for (int i = 0; i < g_n; ++i) cen += g_z[i].cx;
        cen /= g_n;
        for (int i = 0; i < g_n; ++i) {
            int side = g_z[i].cx < cen - 3 ? -1 : (g_z[i].cx > cen + 3 ? 1 : 0);
            // Dance only adds a small nudge on top of whatever else moves the zone; it changes nothing else.
            // A zone already swinging far gets no nudge, and while the echo has eased the damping the
            // nudge is scaled down by as much (so the two can't pump each other up).
            if (g_z[i].wmap && fabsf(g_z[i].dx) + fabsf(g_z[i].dy) < DANCE_ROOM)
                dance_kick(g_z[i], DANCE_GAIN * g_dance * g_z[i].damp_k, side, sign);
        }
    }

    // ---- Targets: jolts + echo + twist (fast) and real gravity (held) ------------------------
    const float jx = -SENS * motion[0], jy = -SENS * motion[1], js = -SENS * motion[2];
    const float dn[3] = {dn_live[0], dn_live[1], dn_live[2]};                                // real "down"
    const float tw_w = g_w, tw_al = g_alpha;                  // twist
    const float w2 = tw_w * tw_w;

    float tg[MAX_ZONES][3];
    int steps = (int)ceilf(dt / 0.004f);
    if (steps < 1) steps = 1;
    const float h = dt / steps;
    for (int i = 0; i < g_n; ++i) {
        Zone& z = g_z[i];
        if (!z.wmap) continue;
        float gx, gy, gs;                                  // real gravity vs. the picture's own down
        gravity_target(z, dn, GRAV_K, GRAV_Z, gx, gy, gs);
        {   // the steady hang stays inside reachable travel, so a zone held there can come to rest
            float m = sqrtf(gx * gx + gy * gy) * z.sag * z.sens;
            if (m > 0.95f) { gx *= 0.95f / m; gy *= 0.95f / m; }
            float mz = fabsf(gs) * z.sag * z.sens;
            if (mz > 0.95f) gs *= 0.95f / mz;
        }
        // the zone's spot is swung around by a twist: tangential α·r⊥ and centripetal −ω²·r (in g)
        float rx = (z.cx - IMG_W * 0.5f) * PX_M * ROT_SCALE, ry = (z.cy - IMG_H * 0.5f) * PX_M * ROT_SCALE;
        float arx = (tw_al * -ry - w2 * rx) / 9.81f, ary = (tw_al * rx - w2 * ry) / 9.81f;
        float tx = clampf(jx - SENS * arx + z.sag * gx, -1, 1);
        float ty = clampf(jy - SENS * ary + z.sag * gy, -1, 1);
        float ts = clampf(js + z.sag * gs, -1, 1);
        tg[i][0] = tx; tg[i][1] = ty; tg[i][2] = ts;
        float moved = fabsf(tx - g_hold[i][0]);
        if (fabsf(ty - g_hold[i][1]) > moved) moved = fabsf(ty - g_hold[i][1]);
        if (fabsf(ts - g_hold[i][2]) > moved) moved = fabsf(ts - g_hold[i][2]);
        if (!z.awake && moved > 0.03f) z.awake = true;
    }
    // Step all zones together so neighbours can press on each other between substeps.
    float gap[MAX_ZONES * MAX_ZONES];
    const bool pairs = g_n > 1;
    if (pairs)
        for (int i = 0; i < g_n; ++i)
            for (int j = i + 1; j < g_n; ++j) gap[i * g_n + j] = contact_gap(g_z[i], g_z[j]);
    for (int i = 0; i < g_n; ++i) g_z[i].pressed = false;   // touched a neighbour at any substep
    for (int s = 0; s < steps; ++s) {
        for (int i = 0; i < g_n; ++i)
            if (g_z[i].wmap && g_z[i].awake) step_zone(g_z[i], h, tg[i][0], tg[i][1], tg[i][2]);
        if (pairs) contact_pass(g_z, g_n, gap, h);
    }
    bool any = false;
    for (int i = 0; i < g_n; ++i) {
        Zone& z = g_z[i];
        if (!z.wmap) continue;
        const float tx = tg[i][0], ty = tg[i][1], ts = tg[i][2];
        if (z.awake && g_echo < 0.02f && (zone_still(z, tx, ty, ts) || zone_still_pressed(z))) {   // rest where it hangs (not while still swaying)
            const bool home = fabsf(tx) < 0.004f && fabsf(ty) < 0.004f && fabsf(ts) < 0.004f;
            if (home && (!z.pressed || (fabsf(z.dx) < 0.02f && fabsf(z.dy) < 0.02f && fabsf(z.s) < 0.02f))) zone_rest(z);
            else zone_freeze(z);
            g_hold[i][0] = tx; g_hold[i][1] = ty; g_hold[i][2] = ts;
            g_shown[i] = true;                             // draw that final pose once
        }
        if (z.awake || g_shown[i]) any = true;
    }
    if (g_par.ready) { par_tick(now, dt, may_draw, any); return; }   // depth: everything goes through it
    // The flat picture: a frame on each beat (g_clock) - the water into the layer and sent a slice at a time
    // (a big sea can't hold the loop up), the hanging things over it, then the zones on top.
    if (!g_ff && may_draw && g_clock.due(now)) {
        const bool water = g_nw > 0 && g_layer, sways = !water && g_ns > 0 && any_sway_due();
        if (water || sways || any) {
            g_clock.begin(now); g_ff = true;
            if (water) water_job_begin(false);
            else g_sw_start = sways;
        }
    }
    if (!g_ff || !may_draw) return;
    bool under = false;                                // something under the zones was just sent: they follow now
    if (g_wj.phase) {
        if (!water_job()) return;                      // (zones wait for the water to finish)
        under = true;
        for (int i = 0; i < g_n; ++i)                  // zones over water draw on top of it
            for (int k = 0; k < g_nw; ++k) {
                const Water& w = g_wat[k];
                const Zone& z = g_z[i];
                if (w.edge && z.wmap && (z.awake || zone_displaced(z)) &&
                    !(z.bx >= w.bx + w.bw || w.bx >= z.bx + z.bw || z.by >= w.by + w.bh || w.by >= z.by + z.bh)) {
                    g_shown[i] = true; any = true;
                }
            }
    }
    if (g_sw_start) {                                  // hair / cloth / pendants (no water): into the layer a slice
        if (!fits(g_c_swstart)) return;                //  at a time, then sent - with the zones over them
        g_sw_start = false;
        Piece pc(g_c_swstart);
        sway_redraw_start(false);
    }
    if (g_sj.on) {
        if (!sway_redraw_run(true)) return;            // (zones wait for it, as for the water)
        g_sw_send = true;
    }
    if (g_sw_send) {
        if (!fits(g_c_swpush.us + (g_n > 0 ? g_c_zones.us : 0))) return;
        g_sw_send = false;
        { Piece pc(g_c_swpush); sway_push(g_sj.which, g_sj.nd); }
        under = true;
        for (int k = 0; k < g_sj.nd; ++k)              // zones over them draw on top
            for (int i = 0; i < g_n; ++i) {
                const Sway& h = g_sway[g_sj.which[k]];
                const Zone& z = g_z[i];
                if (z.wmap && regions_meet(z.bx, z.by, z.bx + z.bw, z.by + z.bh, h.ux0, h.uy0, h.ux1, h.uy1)) { g_shown[i] = true; any = true; }
            }
    }
    static bool s_zone_skipped = false;                 // (the zones next tick - but never two ticks running)
    if (!under && !s_zone_skipped && any && !fits(g_c_zones)) { s_zone_skipped = true; return; }
    s_zone_skipped = false;
    g_ff = false;
    if (any) { Piece pc(g_c_zones); draw_frame(); }
    g_clock.end(now);
}

bool poke(int x, int y) {
    if (!g_ok) return false;
    if (g_par.ready) parallax_to_content(g_par, x, y, x, y);   // the tap lands on what's shown there
    for (int i = g_n - 1; i >= 0; --i) {
        if (g_z[i].wmap && zone_contains(g_z[i], (float)x, (float)y)) {
            poke_zone(g_z[i], (float)x, (float)y);
            return true;
        }
    }
    for (int i = 0; i < g_ns; ++i)                     // a tap on hair / cloth / a pendant: a flick
        if (g_sway[i].ready && sway_poke(g_sway[i], (float)x, (float)y)) { g_sway[i].drawn = false; return true; }
    for (int i = 0; i < g_nw; ++i)                     // a tap on water: a ripple
        if (water_contains(g_wat[i], x, y)) { water_poke(g_wat[i], (float)x, (float)y, 160.0f, 2); return true; }
    return false;
}

int water_count() { return g_nw; }
bool dance_toggle() {
    g_dance_on = !g_dance_on;
    if (!g_dance_on) g_dance = 0.0f;
    return g_dance_on;
}
bool dance_is_on() { return g_dance_on; }
// Something was drawn over the photo: show it again as it is now (zones in their poses, depth redrawn).
void refresh() {
    present();
    for (int i = 0; i < g_n; ++i) g_shown[i] = true;
    for (int i = 0; i < g_ns; ++i) g_sway[i].drawn = false;
    g_clock.soon();
}
bool animating() {
    if (!g_ok) return false;
    if (g_layer) return true;
    if (g_par.ready && g_par_moving) return true;
    for (int i = 0; i < g_n; ++i) if (g_z[i].awake) return true;
    for (int i = 0; i < g_ns; ++i) if (g_sway[i].awake) return true;
    return false;
}

void editor_open() {
    frame_reset();
    if (!g_ok || e_open) return;
    rest_all();
    for (int i = 0; i < g_n; ++i) { e_z[i] = g_z[i]; e_z[i].wmap = e_z[i].gmap = e_z[i].tmap = nullptr; e_z[i].span = nullptr; }
    e_n = g_n;
    for (int i = 0; i < WATER_MAX; ++i) water_free(e_w[i]);
    e_nw = 0;
    for (int i = 0; i < g_nw; ++i) { water_copy_def(e_w[e_nw], g_wat[i], water_alloc); ++e_nw; }
    e_wi = e_nw ? 0 : -1;
    for (int k = 0; k < PLANE_MAX; ++k) plane_free(e_p[k]);
    e_np = 0;
    for (int k = 0; k < g_par.n; ++k) { plane_copy_def(e_p[e_np], g_par.pl[k], water_alloc); ++e_np; }
    e_pi = e_np ? 0 : -1;
    for (int p = 0; p < PP_COUNT; ++p) e_pprm[p] = g_par.prm[p];
    e_pgrid = g_par.grid; memcpy(e_gq, g_par.gq, sizeof(e_gq));
    if (!e_pgrid)                                      // (an older file: a water's own grid becomes the ground grid)
        for (int i = 0; i < g_nw; ++i)
            if (g_wat[i].grid) {
                e_pgrid = true; memcpy(e_gq, g_wat[i].gq, sizeof(e_gq));
                bool planes = false;
                for (int k = 0; k < g_par.n; ++k) planes |= plane_any(g_par.pl[k]);
                if (!planes) e_pprm[PP_GROUND] = 0.0f;
                break;
            }
    e_gmode = false;
    for (int k = 0; k < SWAY_MAX; ++k) sway_free(e_s[k]);
    e_ns = 0;
    for (int k = 0; k < g_ns; ++k) { sway_copy_def(e_s[e_ns], g_sway[k], water_alloc); ++e_ns; }
    e_si = e_ns ? 0 : -1;
    e_stroke = false;
    e_wmode = false; e_dmode = false; e_smode = false; e_wtool = WT_BRUSH; e_pn = 0;
    e_active = -1;
    e_armed = -1;
    e_btn_ready = false;
    e_toast[0] = 0;
    t_down = false;
    e_open = true;
    e_draw();
}

bool editor_is_open() { return e_open; }

void editor_tick(uint32_t now) {
    if (!e_open) return;

    // Buttons: A smaller, C bigger (hold to repeat), B save.
    if (!e_btn_ready) {
        if (!M5.BtnA.isPressed() && !M5.BtnB.isPressed() && !M5.BtnC.isPressed()) e_btn_ready = true;
    } else {
        if (e_mm()) {
            if (e_wtool == WT_POLY || e_wtool == WT_LINE) {
                if (M5.BtnA.wasPressed()) e_undo_point();
                if (M5.BtnC.wasPressed()) e_finish();
            } else if (e_wtool == WT_GRID) {
                if (M5.BtnA.wasPressed() || (M5.BtnA.isPressed() && (int32_t)(now - e_rep_a) >= 0)) {
                    e_grid_turn(-1.0f); e_rep_a = now + (M5.BtnA.wasPressed() ? 380 : 80); e_dirty = true;
                }
                if (M5.BtnC.wasPressed() || (M5.BtnC.isPressed() && (int32_t)(now - e_rep_c) >= 0)) {
                    e_grid_turn(+1.0f); e_rep_c = now + (M5.BtnC.wasPressed() ? 380 : 80); e_dirty = true;
                }
            } else if (e_wtool == WT_SMART) {
                if (M5.BtnA.wasPressed()) e_smart_step(-1);
                if (M5.BtnC.wasPressed()) e_smart_step(+1);
            } else {
                if (M5.BtnA.wasPressed() || (M5.BtnA.isPressed() && (int32_t)(now - e_rep_a) >= 0)) {
                    e_brush = e_brush > 2 ? e_brush - 1 : 2; e_rep_a = now + (M5.BtnA.wasPressed() ? 380 : 60); e_dirty = true;
                }
                if (M5.BtnC.wasPressed() || (M5.BtnC.isPressed() && (int32_t)(now - e_rep_c) >= 0)) {
                    e_brush = e_brush < 40 ? e_brush + 1 : 40; e_rep_c = now + (M5.BtnC.wasPressed() ? 380 : 60); e_dirty = true;
                }
            }
        } else {
            e_scale_button(M5.BtnA, e_rep_a, -1, now);
            e_scale_button(M5.BtnC, e_rep_c, +1, now);
        }
        if (M5.BtnB.wasClicked()) { e_close(true); return; }
    }

    // Touch: bar taps, image taps; drags move the selected zone (or its tip dot).
    auto td = M5.Touch.getDetail();
    bool pressed = td.isPressed() && td.x >= 0 && td.y >= 0 && td.y < IMG_H;
    if (pressed) {
        if (!t_down) {
            t_down = true;
            t_moved = false;
            t_x0 = t_lx = td.x;
            t_y0 = t_ly = td.y;
            t_t0 = now;
            t_bar = in_bars(td.y);
            t_tip = !t_bar && !e_mm() && near_tip(td.x, td.y);
            if (!t_bar && e_mm() && (e_wtool == WT_BRUSH || e_wtool == WT_ERASE)) {
                e_bx = -1; e_dab(td.x, td.y); e_dirty = true; e_glive = false;
            }
            if (!t_bar && e_mm() && e_wtool == WT_SMART) { e_bx = -1; e_mark(td.x, td.y); e_dirty = true; }
            if (!t_bar && e_mm() && e_wtool == WT_GRID) e_grid_press(td.x, td.y);
            if (!t_bar && e_smode && e_wtool == WT_LINE && e_sway() && e_sway()->kind == SK_HAIR) {   // a lock: root here
                e_stroke = true; e_pn = 1; e_px[0] = (int16_t)td.x; e_py[0] = (int16_t)td.y;
            }
        } else if (!t_bar) {
            if (!t_moved && (abs(td.x - t_x0) > 6 || abs(td.y - t_y0) > 6)) t_moved = true;
            if (e_mm()) {                     // masks: drags paint / erase / mark (shape & line take taps)
                if ((e_wtool == WT_BRUSH || e_wtool == WT_ERASE) && (td.x != t_lx || td.y != t_ly)) {
                    e_dab(td.x, td.y); e_dirty = true;
                }
                if (e_wtool == WT_SMART && (td.x != t_lx || td.y != t_ly)) { e_mark(td.x, td.y); e_dirty = true; }
                if (e_wtool == WT_GRID && t_moved && (td.x != t_lx || td.y != t_ly)) { e_grid_drag(td.x - t_lx, td.y - t_ly); e_dirty = true; }
                if (e_stroke && e_pn < 32) {          // (a point every few px along the lock)
                    const int dx = td.x - e_px[e_pn - 1], dy = td.y - e_py[e_pn - 1];
                    if (dx * dx + dy * dy >= 25) { e_px[e_pn] = (int16_t)td.x; e_py[e_pn] = (int16_t)td.y; ++e_pn; e_dirty = true; }
                }
                if (e_wtool != WT_GRID || t_moved) { t_lx = td.x; t_ly = td.y; }
            } else if (t_moved) {             // first move applies the whole distance since touch-down
                if (e_active >= 0 && (td.x != t_lx || td.y != t_ly)) {
                    if (t_tip) {
                        move_tip(td.x - t_lx, td.y - t_ly);
                    } else {
                        Zone& z = e_z[e_active];
                        z.cx = clampf(z.cx + (td.x - t_lx), 0, IMG_W);
                        z.cy = clampf(z.cy + (td.y - t_ly), 0, IMG_H);
                    }
                    e_dirty = true;
                }
                t_lx = td.x;
                t_ly = td.y;
            }
        }
    } else if (t_down) {
        t_down = false;
        if (e_stroke) {                       // a lock drawn: kept (at most 16 points along it)
            e_stroke = false;
            Sway* h = e_sway();
            if (t_moved && h && e_pn >= 2) {
                if (h->ng >= SG_MAX) toast("6 locks max - A removes the last");
                else {
                    const int n = e_pn < SG_PTS ? e_pn : SG_PTS;
                    for (int k = 0; k < n; ++k) {
                        const int src = n > 1 ? k * (e_pn - 1) / (n - 1) : 0;
                        h->gx[h->ng][k] = e_px[src]; h->gy[h->ng][k] = e_py[src];
                    }
                    h->gn[h->ng++] = n;
                    toast(h->ng == 1 ? "Lock kept - draw the others the same way" : "Lock kept");
                }
            } else toast("Drag along the lock, from its root to its tip");
            e_pn = 0; e_dirty = true;
        } else if (!t_bar && e_mm() && e_wtool == WT_SMART && e_marks) {    // tap or stroke done: select from it
            e_smart_go(!(!t_moved && now - t_t0 >= 500));
        } else if (!t_moved) {
            if (t_bar)      e_bar_tap(t_x0, t_y0, now - t_t0);
            else if (!t_tip) e_image_tap(t_x0, t_y0, now - t_t0);   // a tap on the point circle does nothing
            if (!e_open) return;              // Cancel closed us
        }
    }

    if (e_toast[0] && (int32_t)(now - e_toast_until) >= 0) { e_toast[0] = 0; e_dirty = true; }
    if (e_dirty && now - e_last_draw >= 30) {
        e_last_draw = now;
        e_draw();
    }
}

}  // namespace jig
