/**
 * Jiggle zones — device side: PSRAM canvases, IMU drive, .jig files, and the on-screen editor.
 * The maths (shapes, weights, springs, warp) is in jiggle_core.h.
 *
 * .jig file (hidden, next to the image: /images/santa.mg → /images/.santa.mg.jig):
 *     MGJIG1
 *     round 120 130 62 0        ← shape  centre-x  centre-y  radius  rotation°   (image pixels)
 *     tear 260 64 46 15
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
constexpr float BASELINE_TAU = 0.8f;     // s: a tilt makes zones sag, then they ease back to centre
constexpr uint32_t FRAME_MS = 25;        // jiggle redraw cap (~40 fps)

// ---- Editor layout -----------------------------------------------------------------------------
constexpr int TB_H = 30;                 // top toolbar height
constexpr int TB_N = 6;                  // Round · Half · Tear · Rot · Del · Cancel
constexpr int HINT_H = 16;
constexpr int FOOT_H = 16;
constexpr int OUTLINE_PTS = 48;

M5Canvas g_img(&M5.Display);             // pristine image
M5Canvas g_scratch(&M5.Display);         // editor frame + jiggle patch buffer
bool g_ok = false;
bool g_swapped = true;

Zone g_z[MAX_ZONES];
bool g_shown[MAX_ZONES];                 // screen currently shows this zone displaced
int  g_n = 0;
char g_jig_path[320] = {0};

Rhythm   g_rhythm;                       // dance mode: beat tracker on the IMU
bool     g_was_dancing = false;
float    g_base[3] = {0, 0, 0};
bool     g_base_init = false;
uint32_t g_last_us = 0;
uint32_t g_last_draw = 0;

// ---- helpers -----------------------------------------------------------------------------------
uint8_t* alloc_psram(size_t n) {
    void* p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!p) p = malloc(n);
    return (uint8_t*)p;
}

void free_map(Zone& z) {
    if (z.wmap) free(z.wmap);
    z.wmap = nullptr;
}

void rebuild(Zone& z) {
    free_map(z);
    size_t b = zone_wmap_bytes(z);
    prepare_zone(z, b ? alloc_psram(b) : nullptr);
}

int shape_from_name(const char* s) {
    for (int i = 0; i < SHAPE_COUNT; ++i)
        if (strcmp(s, shape_info(i).name) == 0) return i;
    return -1;
}

float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

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
        f.printf("%s %d %d %d %d\n", shape_info(z.shape).name,
                 (int)lroundf(z.cx), (int)lroundf(z.cy), (int)lroundf(z.r), (int)z.rot);
    }
    f.close();
    Serial.printf("jig: saved %d zone(s) → %s\n", g_n, g_jig_path);
}

void push_zone(int i) {
    Zone& z = g_z[i];
    if (!z.wmap) return;
    uint16_t* out = (uint16_t*)g_scratch.getBuffer();
    render_zone((const uint16_t*)g_img.getBuffer(), g_swapped, out, g_z, g_n, i);
    M5.Display.setSwapBytes(true);
    M5.Display.pushImage(z.bx, z.by, z.bw, z.bh, out);
    M5.Display.setSwapBytes(false);
}

void rest_all() {
    for (int i = 0; i < g_n; ++i) {
        Zone& z = g_z[i];
        z.dx = z.dy = z.vx = z.vy = z.s = z.vs = 0;
        z.awake = false;
        g_shown[i] = false;
    }
}

void kick_all() {
    for (int i = 0; i < g_n; ++i) {
        g_z[i].vy += 4.0f;
        g_z[i].vs += 3.0f;
        g_z[i].awake = true;
    }
}

// ================================================================================================
// Editor
// ================================================================================================
bool     e_open = false;
Zone     e_z[MAX_ZONES];                 // working copy (definitions only, no weight maps)
int      e_n = 0;
int      e_active = -1;                  // zone being edited
int      e_armed = -1;                   // shape waiting to be placed by a tap
float    e_last_r = R_DEFAULT;
bool     e_dirty = true;
uint32_t e_last_draw = 0;
bool     e_btn_ready = false;            // ignore buttons that were already down when we opened
uint32_t e_rep_a = 0, e_rep_c = 0;
char     e_toast[48] = {0};
uint32_t e_toast_until = 0;

bool     t_down = false, t_moved = false, t_bar = false;
int      t_x0 = 0, t_y0 = 0, t_lx = 0, t_ly = 0;
uint32_t t_t0 = 0;

float g_out_u[SHAPE_COUNT][OUTLINE_PTS], g_out_w[SHAPE_COUNT][OUTLINE_PTS];
bool  g_out_init = false;

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

const char* context_hint() {
    if (e_armed >= 0) {
        static char buf[40];
        snprintf(buf, sizeof(buf), "Tap the photo to place: %s", shape_info(e_armed).label);
        return buf;
    }
    if (e_active >= 0) return "Drag to move  -  Rot turns  -  B saves";
    if (e_n > 0) return "Pick a shape, or tap a zone to edit it";
    return "Pick a shape above";
}

void e_draw() {
    M5Canvas& c = g_scratch;
    memcpy(c.getBuffer(), g_img.getBuffer(), (size_t)IMG_W * IMG_H * 2);

    // Zones: thin dotted ring = how far the jiggle reaches, solid line = the object you're fitting.
    for (int i = 0; i < e_n; ++i) {
        bool act = (i == e_active);
        if (act) draw_outline(c, e_z[i], MARGIN, C_HINT, 0.5f, false, true);
        draw_outline(c, e_z[i], 1.0f, act ? C_LIME : C_TEXT, act ? 1.2f : 0.7f, true, false);
    }
    if (e_active >= 0) {
        int x = (int)lroundf(e_z[e_active].cx), y = (int)lroundf(e_z[e_active].cy);
        c.fillRect(x - 5, y - 1, 11, 3, C_BLACK);
        c.fillRect(x - 1, y - 5, 3, 11, C_BLACK);
        c.drawFastHLine(x - 4, y, 9, C_LIME);
        c.drawFastVLine(x, y - 4, 9, C_LIME);
    }

    // Top bar
    c.fillRect(0, 0, IMG_W, TB_H, C_BG);
    c.setFont(&fonts::Font2);
    c.setTextDatum(middle_center);
    for (int i = 0; i < TB_N; ++i) {
        int x0 = i * IMG_W / TB_N, x1 = (i + 1) * IMG_W / TB_N;
        int cx = (x0 + x1) / 2, cy = TB_H / 2;
        bool on = (i < SHAPE_COUNT && e_armed == i);
        if (on) c.fillRoundRect(x0 + 2, 2, x1 - x0 - 4, TB_H - 4, 5, C_TEAL);
        else    c.drawRoundRect(x0 + 2, 2, x1 - x0 - 4, TB_H - 4, 5, C_DIM);
        if (i < SHAPE_COUNT) {
            Zone icon;
            icon.shape = (uint8_t)i;
            icon.cx = (float)cx;
            icon.cy = (float)cy + (i == SHAPE_HALF ? 2.0f : 0.0f);
            icon.r = (i == SHAPE_TEAR) ? 11.0f : 9.0f;
            draw_outline(c, icon, 1.0f, on ? C_LIME : C_TEXT, 0.8f, false, false);
        } else {
            static const char* lbl[] = {"Rot", "Del", "Cancel"};
            bool live = (i == 5) || e_active >= 0;
            c.setTextColor(live ? C_TEXT : C_DIM);
            c.drawString(lbl[i - SHAPE_COUNT], cx, cy);
        }
    }

    // Hint line under the bar
    bool toasting = e_toast[0] && (int32_t)(millis() - e_toast_until) < 0;
    const char* hint = toasting ? e_toast : context_hint();
    c.fillRect(0, TB_H, IMG_W, HINT_H, C_BLACK);
    c.setTextColor(toasting ? C_YEL : C_HINT);
    c.drawString(hint, IMG_W / 2, TB_H + HINT_H / 2);

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

void e_bar_tap(int x, uint32_t held) {
    int i = x * TB_N / IMG_W;
    if (i < 0 || i >= TB_N) return;
    if (i < SHAPE_COUNT) {
        if (e_armed == i) { e_armed = -1; e_dirty = true; return; }
        if (e_n >= MAX_ZONES) { toast("4 zones max - Del one first"); return; }
        e_armed = i;
        e_dirty = true;
        return;
    }
    switch (i) {
    case 3:                                   // Rot: tap = 15°, hold = 90°
        if (e_active < 0) { toast("Select a zone to rotate"); return; }
        e_z[e_active].rot = (int16_t)((e_z[e_active].rot + (held >= 500 ? 90 : 15)) % 360);
        e_dirty = true;
        break;
    case 4:
        e_delete_active();
        break;
    case 5:
        e_close(false);
        break;
    }
}

void e_image_tap(int x, int y) {
    if (e_armed >= 0) {
        if (e_n < MAX_ZONES) {
            Zone z;
            z.shape = (uint8_t)e_armed;
            z.cx = (float)x;
            z.cy = (float)y;
            z.r = e_last_r;
            z.rot = 0;
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
    Serial.printf("jig: canvases ok, byte-swapped=%d, imu=%d\n", (int)g_swapped, (int)M5.Imu.isEnabled());

    g_ok = true;
    g_last_us = micros();
    return true;
}

M5Canvas& image() { return g_img; }
bool image_swapped() { return g_swapped; }

void present() {
    if (g_ok) g_img.pushSprite(0, 0);
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
    char buf[512];
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
        char name[12];
        int cx, cy, r, rot;
        if (sscanf(line, "%11s %d %d %d %d", name, &cx, &cy, &r, &rot) != 5) continue;
        int shape = shape_from_name(name);
        if (shape < 0) continue;
        Zone z;
        z.shape = (uint8_t)shape;
        z.cx = clampf((float)cx, 0, IMG_W);
        z.cy = clampf((float)cy, 0, IMG_H);
        z.r = clampf((float)r, R_MIN, R_MAX);
        z.rot = (int16_t)(((rot % 360) + 360) % 360);
        g_z[g_n] = z;
        g_z[g_n].wmap = nullptr;
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

    // Track the IMU even on photos without zones, so the baseline is current when one appears.
    float a[3] = {0, 0, 0};
    bool have = M5.Imu.isEnabled();
    if (have) {
        M5.Imu.update();
        float ax = 0, ay = 0, az = 0;
        M5.Imu.getAccel(&ax, &ay, &az);
        a[0] = IMU_SX * ax; a[1] = IMU_SY * ay; a[2] = IMU_SZ * az;
        if (!g_base_init) {
            for (int i = 0; i < 3; ++i) g_base[i] = a[i];
            g_base_init = true;
        }
        float k = 1.0f - expf(-dt / BASELINE_TAU);
        for (int i = 0; i < 3; ++i) g_base[i] += (a[i] - g_base[i]) * k;
    }
    // Dance mode: a steady rhythm in the motion locks a metronome that kicks every zone on the beat.
    bool beat = false;
    if (have) {
        float bl = sqrtf(g_base[0] * g_base[0] + g_base[1] * g_base[1] + g_base[2] * g_base[2]);
        if (bl < 0.2f) bl = 1.0f;
        beat = g_rhythm.update(dt, a[0] - g_base[0], a[1] - g_base[1], a[2] - g_base[2],
                               g_base[0] / bl, g_base[1] / bl, g_base[2] / bl);
        if (g_rhythm.locked() != g_was_dancing) {
            g_was_dancing = g_rhythm.locked();
            if (g_was_dancing) Serial.printf("jig: dancing at %.0f BPM\n", g_rhythm.bpm());
            else               Serial.println("jig: rhythm stopped");
        }
    }
    const float dance = g_rhythm.amp();

    // Inertia: content lags the device, so it's pushed opposite to the measured acceleration.
    // While dancing the beat kicks lead and the raw motion is turned down so they don't fight.
    const float sens = SENS * (1.0f - 0.5f * dance);
    float tx = have ? clampf(-sens * (a[0] - g_base[0]), -1, 1) : 0;
    float ty = have ? clampf(-sens * (a[1] - g_base[1]), -1, 1) : 0;
    float ts = have ? clampf(-sens * (a[2] - g_base[2]), -1, 1) : 0;
    bool stirred = fabsf(tx) > 0.03f || fabsf(ty) > 0.03f || fabsf(ts) > 0.03f;
    if (g_n == 0) return;

    if (beat) {
        float cen = 0;
        for (int i = 0; i < g_n; ++i) cen += g_z[i].cx;
        cen /= g_n;
        for (int i = 0; i < g_n; ++i) {
            int side = g_z[i].cx < cen - 3 ? -1 : (g_z[i].cx > cen + 3 ? 1 : 0);
            if (g_z[i].wmap) dance_kick(g_z[i], dance, side, g_rhythm.beat_sign());
        }
    }

    bool any = false;
    for (int i = 0; i < g_n; ++i) {
        Zone& z = g_z[i];
        if (!z.wmap) continue;
        if (stirred) z.awake = true;
        if (z.awake) {
            int steps = (int)ceilf(dt / 0.004f);
            float h = dt / steps;
            for (int s = 0; s < steps; ++s) step_zone(z, h, tx, ty, ts);
            if (zone_settled(z, tx, ty, ts)) {
                z.dx = z.dy = z.vx = z.vy = z.s = z.vs = 0;
                z.awake = false;
            }
        }
        if (z.awake || g_shown[i]) any = true;
    }
    if (!any || !may_draw) return;
    if (now - g_last_draw < FRAME_MS) return;
    g_last_draw = now;

    stage_frame(g_z, g_n);
    for (int i = 0; i < g_n; ++i) {
        if (g_z[i].awake) { push_zone(i); g_shown[i] = true; }
        else if (g_shown[i]) { push_zone(i); g_shown[i] = false; }   // final frame at rest
    }
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
    for (int i = 0; i < g_n; ++i) { e_z[i] = g_z[i]; e_z[i].wmap = nullptr; }
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

    // Touch: bar taps, image taps, drags move the selected zone.
    auto td = M5.Touch.getDetail();
    bool pressed = td.isPressed() && td.x >= 0 && td.y >= 0 && td.y < IMG_H;
    if (pressed) {
        if (!t_down) {
            t_down = true;
            t_moved = false;
            t_x0 = t_lx = td.x;
            t_y0 = t_ly = td.y;
            t_t0 = now;
            t_bar = td.y < TB_H;
        } else if (!t_bar) {
            if (!t_moved && (abs(td.x - t_x0) > 6 || abs(td.y - t_y0) > 6)) t_moved = true;
            if (t_moved) {                    // first move applies the whole distance since touch-down
                if (e_active >= 0 && (td.x != t_lx || td.y != t_ly)) {
                    Zone& z = e_z[e_active];
                    z.cx = clampf(z.cx + (td.x - t_lx), 0, IMG_W);
                    z.cy = clampf(z.cy + (td.y - t_ly), 0, IMG_H);
                    e_dirty = true;
                }
                t_lx = td.x;
                t_ly = td.y;
            }
        }
    } else if (t_down) {
        t_down = false;
        if (!t_moved) {
            if (t_bar) e_bar_tap(t_x0, now - t_t0);
            else       e_image_tap(t_x0, t_y0);
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
