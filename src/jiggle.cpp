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
 * Dance: the IMU and the microphone each run a beat tracker. Music sets the default dance (a twerk on
 * the beat); if your own movement locks a rhythm, that takes over until it stops.
 *
 * Rendering: zones that moved are grouped (overlapping ones render together, each pixel once), then
 * drawn in 12-row strips into two small internal-RAM buffers: while one strip goes to the screen by
 * DMA the next is being rendered. Each strip only covers the columns the zones actually reach.
 */
#include "jiggle.h"
#include "jiggle_core.h"
#include "jiggle_dance.h"
#include <SD.h>
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
constexpr float SENS = 1.4f;             // swing per g of motion (1.0 = zone's full travel)
constexpr float BASELINE_TAU = 0.8f;     // s: jolts register, then settle
constexpr float SAG = 0.7f;              // tilt the Core2 → flesh hangs toward the low side…
constexpr float SAG_TAU = 5.0f;          // …and slowly recentres over a few seconds
constexpr float SAG_DEAD = 0.03f;        // g: ignore tiny tilts, so a quick shake leaves nothing behind
constexpr uint32_t FRAME_MS = 16;        // redraw cap (~60 fps); big zones are limited by the SPI bus

// ---- Music ---------------------------------------------------------------------------------------
constexpr bool     MIC_DANCE = true;      // dance to music heard by the microphone
constexpr uint32_t MIC_SR = 16000;
constexpr size_t   MIC_N = 320;           // 20 ms chunks
constexpr float    MIC_LEAD = 0.03f;      // beat lead for audio (spring swing − recording latency)
constexpr bool     USE_DMA = true;        // set false to push strips without DMA

// ---- Rendering ---------------------------------------------------------------------------------
constexpr int STRIP_ROWS = 12;
constexpr size_t STRIP_PIX = (size_t)STRIP_ROWS * IMG_W;

// ---- Editor layout -----------------------------------------------------------------------------
constexpr int ROW_H = 30;                // each toolbar row
constexpr int TB1_N = 4;                 // Round · Half · Tear · Cancel
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

Zone g_z[MAX_ZONES];
bool g_shown[MAX_ZONES];                 // screen currently shows this zone displaced
int  g_n = 0;
char g_jig_path[320] = {0};

Rhythm   g_rhythm;                       // beats from your movement (IMU)
Rhythm   g_rhythm_mic;                   // beats from music (microphone)
AudioOnset g_onset;
int16_t  g_mic_buf[MIC_N];
bool     g_mic_on = false, g_mic_busy = false;
uint32_t g_mic_at = 0;
float    g_mic_hit = 0;                  // latest chunk's hit strength…
float    g_mic_hold = 0;                 // …fed to the tracker for one chunk's worth of time (s)
int      g_dance_src = 0;                // 0 none, 1 IMU, 2 music (for the log)
float    g_acc_last[3] = {0, 0, 1};
float    g_groove[2] = {0, 0};               // decaying memory of your movement's direction
bool     g_was_dancing = false;
float    g_base[3] = {0, 0, 0};
float    g_slow[3] = {0, 0, 0};
bool     g_base_init = false;
uint32_t g_last_us = 0;
uint32_t g_last_draw = 0;

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

void rebuild(Zone& z) {
    free_map(z);
    size_t b = zone_mem_bytes(z);
    prepare_zone(z, b ? alloc_psram(b) : nullptr);
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

const uint16_t* render_src() { return g_src_fast ? g_src_fast : (const uint16_t*)g_img.getBuffer(); }

void save_zones() {
    if (!g_jig_path[0]) return;
    if (g_n == 0) {
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
        for (int t = 0; t < TUNE_COUNT; ++t)
            if (tune_set(z, (uint8_t)t)) f.printf(" %s=%.3g", tune_info((uint8_t)t).key, z.tune[t]);
        f.printf("\n");
    }
    f.close();
    Serial.printf("jig: saved %d zone(s) -> %s\n", g_n, g_jig_path);
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
void draw_frame() {
    stage_frame(g_z, g_n);
    int ids[MAX_ZONES], nd = 0;
    for (int i = 0; i < g_n; ++i)
        if (g_z[i].wmap && (g_z[i].awake || g_shown[i])) ids[nd++] = i;
    if (!nd) return;

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
    bool src_swapped = g_src_fast ? true : g_swapped;
    int cur = 0;
    bool inflight = false;
    M5.Display.startWrite();
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
            render_rect(src, src_swapped, g_strip[cur], g_z, members, nm, xa, y, xb, ye, g_acc);
            push_strip(xa, y, xb - xa, ye - y, g_strip[cur], inflight);
            px += (uint32_t)(xb - xa) * (ye - y);
            cur ^= 1;
        }
    }
    if (inflight) M5.Display.waitDMA();
    M5.Display.endWrite();

    for (int k = 0; k < nd; ++k) g_shown[ids[k]] = g_z[ids[k]].awake;

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

void rest_all() {
    for (int i = 0; i < g_n; ++i) { zone_rest(g_z[i]); g_shown[i] = false; }
}

void kick_all() {
    for (int i = 0; i < g_n; ++i) {
        g_z[i].vy += 0.6f * g_z[i].w1;
        g_z[i].vs += 0.4f * g_z[i].w1;
        g_z[i].awake = true;
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

bool     t_down = false, t_moved = false, t_bar = false, t_tip = false;
int      t_x0 = 0, t_y0 = 0, t_lx = 0, t_ly = 0;
uint32_t t_t0 = 0;

float g_out_u[SHAPE_COUNT][OUTLINE_PTS], g_out_w[SHAPE_COUNT][OUTLINE_PTS];
bool  g_out_init = false;

int bar_bottom() { return e_active >= 0 ? 2 * ROW_H : ROW_H; }

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
    int x0 = i * IMG_W / n, x1 = (i + 1) * IMG_W / n, y0 = row * ROW_H;
    if (on) c.fillRoundRect(x0 + 2, y0 + 2, x1 - x0 - 4, ROW_H - 4, 5, C_TEAL);
    else    c.drawRoundRect(x0 + 2, y0 + 2, x1 - x0 - 4, ROW_H - 4, 5, live ? C_DIM : C_BG);
}

void e_draw() {
    M5Canvas& c = g_scratch;
    memcpy(c.getBuffer(), g_img.getBuffer(), (size_t)IMG_W * IMG_H * 2);

    // Zones: thin dotted ring = how far the jiggle reaches, solid line = the object you're fitting.
    for (int i = 0; i < e_n; ++i) {
        bool act = (i == e_active);
        if (act) draw_outline(c, e_z[i], MARGIN, C_HINT, 0.5f, false, true);
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
    c.fillRect(0, 0, IMG_W, ROW_H, C_BG);
    for (int i = 0; i < TB1_N; ++i) {
        int x0 = i * IMG_W / TB1_N, x1 = (i + 1) * IMG_W / TB1_N;
        int cx = (x0 + x1) / 2, cy = ROW_H / 2;
        bool on = (i < SHAPE_COUNT && e_armed == i);
        bar_button(c, 0, i, TB1_N, on, true);
        if (i < SHAPE_COUNT) {
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

    // Row 2: selected zone's options
    if (e_active >= 0) {
        const Zone& z = e_z[e_active];
        c.fillRect(0, ROW_H, IMG_W, ROW_H, C_BG);
        char feel[16];
        snprintf(feel, sizeof(feel), "%s", material_info(z.material).label);
        const char* lbl[TB2_N] = {"Rot", feel, z.tip ? "Point: on" : "Point: off", "Del"};
        for (int i = 0; i < TB2_N; ++i) {
            bar_button(c, 1, i, TB2_N, i == 2 && z.tip, true);
            c.setTextColor(C_TEXT);
            c.drawString(lbl[i], (2 * i + 1) * IMG_W / (2 * TB2_N), ROW_H + ROW_H / 2);
        }
    }

    // Hint line
    int hy = bar_bottom();
    bool toasting = e_toast[0] && (int32_t)(millis() - e_toast_until) < 0;
    const char* hint = toasting ? e_toast : context_hint();
    c.fillRect(0, hy, IMG_W, HINT_H, C_BLACK);
    c.setTextColor(toasting ? C_YEL : C_HINT);
    c.drawString(hint, IMG_W / 2, hy + HINT_H / 2);

    // Footer over the physical buttons
    int fy = IMG_H - FOOT_H;
    c.fillRect(0, fy, IMG_W, FOOT_H, C_BLACK);
    c.setTextColor(e_active >= 0 ? C_TEXT : C_DIM);
    c.drawString("A  smaller", 55, fy + FOOT_H / 2);
    c.drawString("bigger  C", 265, fy + FOOT_H / 2);
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
        save_zones();
    }
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
    if (y < ROW_H) {
        int i = x * TB1_N / IMG_W;
        if (i < SHAPE_COUNT) {
            if (e_armed == i) { e_armed = -1; e_dirty = true; return; }
            if (e_n >= MAX_ZONES) { toast("4 zones max - Del one first"); return; }
            e_armed = i;
            e_dirty = true;
        } else {
            e_close(false);
        }
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
        if (z.tip && shape_sdf(z.shape, z.tip_u, z.tip_w) > -tip_radius(z) * 0.5f) shape_tip_default(z.shape, z.tip_u, z.tip_w);
        if (z.tip) toast("Drag the circle onto the apex");
        break;
    case 3:
        e_delete_active();
        return;
    }
    e_dirty = true;
}

void e_image_tap(int x, int y) {
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
    if (shape_sdf(z.shape, nu, nw) < -tip_radius(z) * 0.5f) { z.tip_u = nu; z.tip_w = nw; }
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
    g_rhythm.set_coast(4.0f);                // your rhythm keeps going ~4 s after you stop

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
}

void clear_zones() {
    for (int i = 0; i < g_n; ++i) free_map(g_z[i]);
    g_n = 0;
    g_jig_path[0] = 0;
}

int zone_count() { return g_n; }

void load_zones(const char* image_path) {
    clear_zones();
    if (!g_ok || !image_path) return;
    // Hidden sidecar: "/images/santa.mg" → "/images/.santa.mg.jig" (dot-files are skipped by scans).
    const char* slash = strrchr(image_path, '/');
    if (slash) snprintf(g_jig_path, sizeof(g_jig_path), "%.*s.%s.jig", (int)(slash - image_path + 1), image_path, slash + 1);
    else       snprintf(g_jig_path, sizeof(g_jig_path), ".%s.jig", image_path);
    if (!SD.exists(g_jig_path)) return;
    File f = SD.open(g_jig_path, FILE_READ);
    if (!f) return;
    char buf[768];
    size_t got = f.read((uint8_t*)buf, sizeof(buf) - 1);
    f.close();
    buf[got] = 0;

    char* save = nullptr;
    int line_no = 0;
    for (char* line = strtok_r(buf, "\r\n", &save); line; line = strtok_r(nullptr, "\r\n", &save), ++line_no) {
        if (line_no == 0) {
            if (strncmp(line, "MGJIG1", 6) != 0) return;
            continue;
        }
        if (g_n >= MAX_ZONES) break;
        Zone z;
        if (!parse_zone(line, z)) continue;
        g_z[g_n] = z;
        rebuild(g_z[g_n]);
        g_shown[g_n] = false;
        ++g_n;
    }
    Serial.printf("jig: %d zone(s) from %s\n", g_n, g_jig_path);
}

void play_tick(uint32_t now, bool may_draw) {
    uint32_t us = micros();
    float dt = (us - g_last_us) * 1e-6f;
    g_last_us = us;
    if (!g_ok) return;
    if (dt > 0.05f) dt = 0.05f;
    if (dt <= 0.0f) return;

    // Track the IMU even on photos without zones, so the baselines are current when one appears.
    float a[3] = {0, 0, 0};
    bool have = M5.Imu.isEnabled();
    if (have) {
        if (M5.Imu.update()) {
            auto d = M5.Imu.getImuData();
            g_acc_last[0] = d.accel.x; g_acc_last[1] = d.accel.y; g_acc_last[2] = d.accel.z;
        }
        a[0] = IMU_SX * g_acc_last[0]; a[1] = IMU_SY * g_acc_last[1]; a[2] = IMU_SZ * g_acc_last[2];
        if (!g_base_init) {
            for (int i = 0; i < 3; ++i) g_base[i] = g_slow[i] = a[i];
            g_base_init = true;
        }
        float k = 1.0f - expf(-dt / BASELINE_TAU), ks = 1.0f - expf(-dt / SAG_TAU);
        for (int i = 0; i < 3; ++i) {
            g_base[i] += (a[i] - g_base[i]) * k;
            g_slow[i] += (g_base[i] - g_slow[i]) * ks;
        }
    }
    // Dance: your movement (IMU) and the music (mic) each run a beat tracker.
    bool imu_beat = false, mic_beat = false;
    if (have) {
        float bl = sqrtf(g_base[0] * g_base[0] + g_base[1] * g_base[1] + g_base[2] * g_base[2]);
        if (bl < 0.2f) bl = 1.0f;
        imu_beat = g_rhythm.update(dt, a[0] - g_base[0], a[1] - g_base[1], a[2] - g_base[2],
                                   g_base[0] / bl, g_base[1] / bl, g_base[2] / bl);
    }
    if (MIC_DANCE && g_n > 0 && !g_mic_on) {          // start listening once a photo has zones
        if (M5.Speaker.isEnabled()) M5.Speaker.end(); // Core2: mic and speaker share one I2S port
        g_mic_on = M5.Mic.isEnabled() || M5.Mic.begin();
        g_mic_busy = false;
        Serial.printf("jig: microphone %s\n", g_mic_on ? "listening" : "unavailable");
    }
    if (g_mic_on) {                                   // record-and-poll, one 20 ms chunk at a time
        if (!g_mic_busy) {
            g_mic_busy = M5.Mic.record(g_mic_buf, MIC_N, MIC_SR);
            g_mic_at = now;
        } else if (now - g_mic_at >= 18 && M5.Mic.isRecording() == 0) {
            g_mic_hit = g_onset.process(g_mic_buf, MIC_N, MIC_SR);
            g_mic_hold = (float)MIC_N / MIC_SR;
            g_mic_busy = M5.Mic.record(g_mic_buf, MIC_N, MIC_SR);
            g_mic_at = now;
        }
        // Feed the hit for exactly one chunk of time, whatever the loop rate, so it weighs the same.
        float on = g_mic_hold > 0 ? (dt < g_mic_hold ? 1.0f : g_mic_hold / dt) : 0.0f;
        mic_beat = g_rhythm_mic.update(dt, 0, 0, g_mic_hit * on, 0, 0, 1);
        g_mic_hold -= dt;
    }
    // Your own rhythm wins while it's locked (or fading out); otherwise the music leads.
    int src = g_rhythm.locked() ? 1 : (g_rhythm_mic.amp() > 0.0f ? 2 : (g_rhythm.amp() > 0.0f ? 1 : 0));
    const Rhythm& R = src == 2 ? g_rhythm_mic : g_rhythm;
    const bool beat = src == 1 ? imu_beat : (src == 2 ? mic_beat : false);
    const float dance = src ? R.amp() : 0.0f;
    if (src != g_dance_src || (src && R.locked() != g_was_dancing)) {
        if (src && R.locked()) Serial.printf("jig: dancing to %s at %.0f BPM\n", src == 1 ? "movement" : "music", R.bpm());
        else if (!src)          Serial.println("jig: dance stopped");
        g_dance_src = src;
        g_was_dancing = src && R.locked();
    }

    // Inertia: content lags the device (pushed opposite to the measured acceleration). A tilt also
    // makes it hang toward the new "down" for a few seconds. While dancing, raw motion is turned down.
    const float sens = SENS * (1.0f - 0.5f * dance);
    float jx = 0, jy = 0, ts = 0, gx = 0, gy = 0;    // jolt (fast) and sag (slow tilt) parts
    if (have) {
        jx = -sens * (a[0] - g_base[0]); jy = -sens * (a[1] - g_base[1]);
        ts = clampf(-sens * (a[2] - g_base[2]), -1, 1);
        auto dz = [](float v) { return v > SAG_DEAD ? v - SAG_DEAD : (v < -SAG_DEAD ? v + SAG_DEAD : 0.0f); };
        gx = -SAG * dz(g_base[0] - g_slow[0]); gy = -SAG * dz(g_base[1] - g_slow[1]);
    }
    bool stirred = fabsf(jx + gx) > 0.03f || fabsf(jy + gy) > 0.03f || fabsf(ts) > 0.03f;
    // Groove: rectified recent motion per axis, rising fast, fading over ~4 s, only while dancing.
    {
        float up = 1.0f - expf(-dt / 0.3f), down = 1.0f - expf(-dt / 4.0f);
        float m[2] = {fabsf(jx), fabsf(jy)};
        for (int i = 0; i < 2; ++i) {
            float target = clampf(m[i], 0, 1);
            g_groove[i] += (target - g_groove[i]) * (target > g_groove[i] ? up : down);
            if (dance <= 0.0f) g_groove[i] = 0;
        }
    }
    if (g_n == 0) return;

    if (beat) {
        float cen = 0;
        for (int i = 0; i < g_n; ++i) cen += g_z[i].cx;
        cen /= g_n;
        for (int i = 0; i < g_n; ++i) {
            int side = g_z[i].cx < cen - 3 ? -1 : (g_z[i].cx > cen + 3 ? 1 : 0);
            if (g_z[i].wmap) dance_kick(g_z[i], dance, side, R.beat_sign(), g_groove[0], g_groove[1]);
        }
    }

    bool any = false;
    for (int i = 0; i < g_n; ++i) {
        Zone& z = g_z[i];
        if (!z.wmap) continue;
        if (stirred) z.awake = true;
        float tx = clampf(jx + z.sag * gx, -1, 1), ty = clampf(jy + z.sag * gy, -1, 1);
        if (z.awake) {
            int steps = (int)ceilf(dt / 0.004f);
            float h = dt / steps;
            for (int s = 0; s < steps; ++s) step_zone(z, h, tx, ty, ts);
            if (zone_settled(z, tx, ty, ts)) zone_rest(z);
        }
        if (z.awake || g_shown[i]) any = true;
    }
    if (!any || !may_draw) return;
    if (now - g_last_draw < FRAME_MS) return;
    g_last_draw = now;
    draw_frame();
}

bool poke(int x, int y) {
    if (!g_ok) return false;
    for (int i = g_n - 1; i >= 0; --i) {
        if (g_z[i].wmap && zone_contains(g_z[i], (float)x, (float)y)) {
            poke_zone(g_z[i], (float)x, (float)y);
            return true;
        }
    }
    return false;
}

void editor_open() {
    if (!g_ok || e_open) return;
    rest_all();
    for (int i = 0; i < g_n; ++i) { e_z[i] = g_z[i]; e_z[i].wmap = e_z[i].gmap = e_z[i].tmap = nullptr; e_z[i].span = nullptr; }
    e_n = g_n;
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
        e_scale_button(M5.BtnA, e_rep_a, -1, now);
        e_scale_button(M5.BtnC, e_rep_c, +1, now);
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
            t_bar = td.y < bar_bottom();
            t_tip = !t_bar && near_tip(td.x, td.y);
        } else if (!t_bar) {
            if (!t_moved && (abs(td.x - t_x0) > 6 || abs(td.y - t_y0) > 6)) t_moved = true;
            if (t_moved) {                    // first move applies the whole distance since touch-down
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
        if (!t_moved) {
            if (t_bar)      e_bar_tap(t_x0, t_y0, now - t_t0);
            else if (!t_tip) e_image_tap(t_x0, t_y0);   // a tap on the point circle does nothing
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
