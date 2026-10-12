/**
 * Mantis Gallery — M5Stack Core2
 *
 * SD card slideshow:
 *   /images/*.jpg|png|mg          → default album
 *   /images/<folder>/…            → extra albums (Btn A / C cycle)
 *
 * Playback: random shuffle, no repeats until the album is exhausted,
 * then reshuffle. Prefers .mg (RGB565) for speed; JPG/PNG fallback.
 */

#include <M5Unified.h>
#include <SD.h>
#include <SPI.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

// Mantis theme
static constexpr uint16_t COL_BG     = 0x18C3;
static constexpr uint16_t COL_TEAL   = 0x038E;
static constexpr uint16_t COL_PURPLE = 0x580B;
static constexpr uint16_t COL_LIME   = 0xB7E0;
static constexpr uint16_t COL_TEXT   = 0xEF5D;
static constexpr uint16_t COL_HINT   = 0x9CF3;
static constexpr uint16_t COL_DIM    = 0x8410;

static constexpr int SCREEN_W = 320;
static constexpr int SCREEN_H = 240;

static constexpr const char* IMAGES_ROOT = "/images";

// Controls
static constexpr uint32_t HOLD_B_MS        = 1200;  // long-press B to enter random subfolder
static constexpr uint32_t TOUCH_LONG_MS    = 650;   // long-press screen → slideshow
static constexpr int      SWIPE_MIN_PX     = 40;
static constexpr uint32_t SLIDE_MIN_MS     = 5000;
static constexpr uint32_t SLIDE_MAX_MS     = 8000;

static constexpr uint16_t COL_BAR_YEL = 0xFFE0;  // yellow
static constexpr uint16_t COL_BAR_RED = 0xF800;  // red
static constexpr uint16_t COL_BAR_BG  = 0x2104;  // dark under bar

// Max path length for SD paths
static constexpr size_t PATH_MAX_LEN = 96;
static constexpr size_t NAME_MAX_LEN = 48;
static constexpr int MAX_ALBUMS = 32;
static constexpr int MAX_FILES_PER_ALBUM = 512;

// ---------------------------------------------------------------------------
//  SD helpers (Core2: SCK=18 MISO=38 MOSI=23 CS=4)
// ---------------------------------------------------------------------------
static bool sd_ok = false;

static bool sd_begin() {
    SPI.begin(18, 38, 23, 4);
    if (SD.begin(4, SPI, 25000000)) {
        sd_ok = true;
        return true;
    }
    // retry slower
    if (SD.begin(4, SPI, 4000000)) {
        sd_ok = true;
        return true;
    }
    sd_ok = false;
    return false;
}

static bool is_image_name(const char* name) {
    if (!name) return false;
    size_t n = strlen(name);
    auto low = [](char c) -> char {
        return (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
    };
    auto match = [&](const char* suf, size_t sl) -> bool {
        if (n < sl) return false;
        for (size_t i = 0; i < sl; ++i) {
            if (low(name[n - sl + i]) != suf[i]) return false;
        }
        return true;
    };
    return match(".mg", 3) || match(".jpg", 4) || match(".png", 4) || match(".jpeg", 5);
}

static bool ends_with_mg(const char* name) {
    size_t n = strlen(name);
    if (n < 3) return false;
    const char* e = name + n - 3;
    return e[0] == '.' && (e[1] == 'm' || e[1] == 'M') && (e[2] == 'g' || e[2] == 'G');
}

static bool ends_with_png(const char* name) {
    size_t n = strlen(name);
    if (n < 4) return false;
    const char* e = name + n - 4;
    auto low = [](char c) { return (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c; };
    return e[0] == '.' && low(e[1]) == 'p' && low(e[2]) == 'n' && low(e[3]) == 'g';
}

// ---------------------------------------------------------------------------
//  Album model
// ---------------------------------------------------------------------------
struct Album {
    char name[NAME_MAX_LEN];   // "" = default (files in /images root)
    char path[PATH_MAX_LEN];   // full directory path
    std::vector<String> files; // filenames only (not full path)
    std::vector<int> order;    // shuffled indices into files
    int cursor = 0;
};

static std::vector<Album> albums;
static int album_i = 0;
static uint16_t* pixel_buf = nullptr; // PSRAM RGB565 frame for .mg
static constexpr size_t PIXEL_BYTES = SCREEN_W * SCREEN_H * 2;

static void show_msg(const char* line1, const char* line2 = nullptr) {
    M5.Display.fillScreen(COL_BG);
    M5.Display.setTextDatum(middle_center);
    M5.Display.setTextColor(COL_TEAL);
    M5.Display.setFont(&fonts::Font4);
    M5.Display.drawString("Mantis Gallery", SCREEN_W / 2, 70);
    M5.Display.setFont(&fonts::Font2);
    M5.Display.setTextColor(COL_TEXT);
    if (line1) M5.Display.drawString(line1, SCREEN_W / 2, 120);
    if (line2) {
        M5.Display.setTextColor(COL_HINT);
        M5.Display.drawString(line2, SCREEN_W / 2, 150);
    }
}

static void reshuffle(Album& a) {
    a.order.clear();
    a.order.reserve(a.files.size());
    for (size_t i = 0; i < a.files.size(); ++i) a.order.push_back((int)i);
    // Fisher–Yates
    for (int i = (int)a.order.size() - 1; i > 0; --i) {
        int j = (int)(esp_random() % (uint32_t)(i + 1));
        std::swap(a.order[i], a.order[j]);
    }
    a.cursor = 0;
}

static bool scan_dir_images(const char* dir, std::vector<String>& out) {
    out.clear();
    File d = SD.open(dir);
    if (!d || !d.isDirectory()) {
        if (d) d.close();
        return false;
    }
    for (;;) {
        File f = d.openNextFile();
        if (!f) break;
        if (!f.isDirectory()) {
            const char* nm = f.name();
            // SD lib may return full path or basename depending on version
            const char* base = strrchr(nm, '/');
            base = base ? base + 1 : nm;
            if (is_image_name(base)) {
                out.push_back(String(base));
            }
        }
        f.close();
    }
    d.close();
    // Prefer listing stability: sort names, shuffle is separate
    std::sort(out.begin(), out.end(), [](const String& a, const String& b) {
        return a.compareTo(b) < 0;
    });
    return true;
}

static bool build_albums() {
    albums.clear();
    album_i = 0;

    // Ensure /images exists
    if (!SD.exists(IMAGES_ROOT)) {
        return false;
    }

    // Default album: files directly in /images
    {
        Album def;
        strncpy(def.name, "default", sizeof(def.name) - 1);
        strncpy(def.path, IMAGES_ROOT, sizeof(def.path) - 1);
        scan_dir_images(IMAGES_ROOT, def.files);
        if (!def.files.empty()) {
            reshuffle(def);
            albums.push_back(std::move(def));
        }
    }

    // Subfolders of /images
    File root = SD.open(IMAGES_ROOT);
    if (root && root.isDirectory()) {
        for (;;) {
            File f = root.openNextFile();
            if (!f) break;
            if (f.isDirectory()) {
                const char* nm = f.name();
                const char* base = strrchr(nm, '/');
                base = base ? base + 1 : nm;
                // skip hidden
                if (base[0] == '.') { f.close(); continue; }

                Album al;
                strncpy(al.name, base, sizeof(al.name) - 1);
                snprintf(al.path, sizeof(al.path), "%s/%s", IMAGES_ROOT, base);
                scan_dir_images(al.path, al.files);
                if (!al.files.empty() && (int)albums.size() < MAX_ALBUMS) {
                    reshuffle(al);
                    albums.push_back(std::move(al));
                }
            }
            f.close();
        }
        root.close();
    }

    return !albums.empty();
}

// ---------------------------------------------------------------------------
//  Image loaders
// ---------------------------------------------------------------------------
static bool load_mg(const char* path) {
    File f = SD.open(path, FILE_READ);
    if (!f) return false;

    char magic[4];
    if (f.read((uint8_t*)magic, 4) != 4) { f.close(); return false; }
    if (memcmp(magic, "MG01", 4) != 0) { f.close(); return false; }

    uint16_t w = 0, h = 0;
    if (f.read((uint8_t*)&w, 2) != 2 || f.read((uint8_t*)&h, 2) != 2) {
        f.close(); return false;
    }
    if (w == 0 || h == 0 || w > 640 || h > 480) { f.close(); return false; }

    size_t need = (size_t)w * (size_t)h * 2;
    // Fast path: exact screen size → pushImage full screen
    if (w == SCREEN_W && h == SCREEN_H && pixel_buf) {
        size_t got = f.read((uint8_t*)pixel_buf, need);
        f.close();
        if (got != need) return false;
        M5.Display.setSwapBytes(true);
        M5.Display.pushImage(0, 0, SCREEN_W, SCREEN_H, pixel_buf);
        M5.Display.setSwapBytes(false);
        return true;
    }

    // Non-native size: read into temp and push centered (still faster than JPG)
    uint16_t* tmp = (uint16_t*)heap_caps_malloc(need, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!tmp) tmp = (uint16_t*)malloc(need);
    if (!tmp) { f.close(); return false; }
    size_t got = f.read((uint8_t*)tmp, need);
    f.close();
    if (got != need) { free(tmp); return false; }

    M5.Display.fillScreen(COL_BG);
    int x = ((int)SCREEN_W - (int)w) / 2;
    int y = ((int)SCREEN_H - (int)h) / 2;
    M5.Display.setSwapBytes(true);
    M5.Display.pushImage(x, y, w, h, tmp);
    M5.Display.setSwapBytes(false);
    free(tmp);
    return true;
}

// M5GFX DataWrapperT<fs::SDFS> is abstract on current cores — avoid draw*File(SD, …).
// Load the whole file into PSRAM/heap and decode from the buffer instead.
static bool load_jpg_png(const char* path, bool png) {
    File f = SD.open(path, FILE_READ);
    if (!f) return false;
    size_t sz = f.size();
    if (sz == 0 || sz > 2 * 1024 * 1024) { // hard cap 2 MB
        f.close();
        return false;
    }
    uint8_t* buf = (uint8_t*)heap_caps_malloc(sz, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!buf) buf = (uint8_t*)malloc(sz);
    if (!buf) { f.close(); return false; }

    size_t got = f.read(buf, sz);
    f.close();
    if (got != sz) { free(buf); return false; }

    M5.Display.fillScreen(COL_BG);
    bool ok = false;
    if (png) {
        ok = M5.Display.drawPng(buf, sz, 0, 0);
    } else {
        ok = M5.Display.drawJpg(buf, sz, 0, 0);
    }
    free(buf);
    return ok;
}

// default album is always index 0 when present (built first in build_albums)
static bool in_default_album() {
    return !albums.empty() && album_i == 0 &&
           strcmp(albums[0].name, "default") == 0;
}

static bool show_current() {
    if (albums.empty()) return false;
    Album& a = albums[album_i];
    if (a.files.empty()) return false;

    if (a.cursor < 0) a.cursor = 0;
    if (a.cursor >= (int)a.order.size()) {
        reshuffle(a);
    }
    int fi = a.order[a.cursor];
    const String& fname = a.files[fi];

    char path[PATH_MAX_LEN];
    snprintf(path, sizeof(path), "%s/%s", a.path, fname.c_str());

    bool ok = false;
    if (ends_with_mg(fname.c_str())) {
        ok = load_mg(path);
    } else if (ends_with_png(fname.c_str())) {
        ok = load_jpg_png(path, true);
    } else {
        ok = load_jpg_png(path, false);
    }

    // No on-screen text or labels — blank frame if load fails
    if (!ok) {
        M5.Display.fillScreen(COL_BG);
    }
    return ok;
}

static void next_image() {
    if (albums.empty()) return;
    Album& a = albums[album_i];
    if (a.files.empty()) return;
    a.cursor++;
    if (a.cursor >= (int)a.order.size()) {
        reshuffle(a);
    }
    show_current();
}

static void prev_image() {
    if (albums.empty()) return;
    Album& a = albums[album_i];
    if (a.files.empty()) return;
    if (a.cursor <= 0) {
        // wrap to end of current shuffle without reshuffling mid-cycle
        a.cursor = (int)a.order.size() - 1;
    } else {
        a.cursor--;
    }
    show_current();
}

// Cycle subfolders only (skip default at index 0 when it is "default")
static void change_subfolder(int delta) {
    if (albums.size() < 2) return;
    int start = 0;
    if (strcmp(albums[0].name, "default") == 0) start = 1;
    int n = (int)albums.size() - start;
    if (n <= 0) return;

    int rel = album_i - start;
    if (rel < 0) rel = 0;
    rel = (rel + delta) % n;
    if (rel < 0) rel += n;
    album_i = start + rel;
    show_current();
}

static void go_default_album() {
    if (albums.empty()) return;
    album_i = 0;
    show_current();
}

static void enter_random_subfolder() {
    if (albums.size() < 2) return;
    int start = (strcmp(albums[0].name, "default") == 0) ? 1 : 0;
    int n = (int)albums.size() - start;
    if (n <= 0) return;
    album_i = start + (int)(esp_random() % (uint32_t)n);
    show_current();
}

// Progress bar at bottom: yellow → red as t goes 0..1
static void draw_hold_bar(float t) {
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    const int bar_h = 8;
    const int y = SCREEN_H - bar_h;
    M5.Display.fillRect(0, y, SCREEN_W, bar_h, COL_BAR_BG);
    // lerp yellow → red
    uint8_t r = 255;
    uint8_t g = (uint8_t)(255 * (1.0f - t));
    uint8_t b = 0;
    uint16_t col = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
    int w = (int)(t * SCREEN_W);
    if (w > 0) M5.Display.fillRect(0, y, w, bar_h, col);
}

static uint32_t random_slide_ms() {
    return SLIDE_MIN_MS + (esp_random() % (SLIDE_MAX_MS - SLIDE_MIN_MS + 1));
}

// ---------------------------------------------------------------------------
//  Interaction state
// ---------------------------------------------------------------------------
static bool     slideshow      = false;
static uint32_t slide_due_ms   = 0;   // when to auto-advance in slideshow
static uint32_t slide_span_ms  = 0;   // current photo dwell

static bool     btn_b_holding  = false;
static uint32_t btn_b_t0       = 0;
static float    btn_b_last_t   = -1;

static bool     touch_down     = false;
static int16_t  touch_x0       = 0, touch_y0 = 0;
static int16_t  touch_x1       = 0, touch_y1 = 0;  // last point while pressed
static uint32_t touch_t0       = 0;
static bool     touch_moved    = false;

// ---------------------------------------------------------------------------
void setup() {
    auto cfg = M5.config();
    cfg.clear_display = true;
    M5.begin(cfg);

    M5.Display.setBrightness(200);
    M5.Display.setRotation(1); // landscape 320x240
    M5.Display.fillScreen(COL_BG);

    Serial.begin(115200);
    Serial.println("Mantis Gallery");

    pixel_buf = (uint16_t*)heap_caps_malloc(PIXEL_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!pixel_buf) {
        pixel_buf = (uint16_t*)heap_caps_malloc(PIXEL_BYTES, MALLOC_CAP_8BIT);
    }

    show_msg("Mounting SD…");
    if (!sd_begin()) {
        show_msg("No SD card", "Insert FAT32 card");
        while (true) { M5.update(); delay(200); }
    }

    show_msg("Scanning /images…");
    if (!build_albums()) {
        show_msg("No images found", "See IMAGE_SPEC.txt");
        while (true) { M5.update(); delay(200); }
    }

    Serial.printf("Albums: %d\n", (int)albums.size());
    for (auto& a : albums) {
        Serial.printf("  %s (%s) %d files\n", a.name, a.path, (int)a.files.size());
    }

    show_current();
}

void loop() {
    M5.update();
    uint32_t now = millis();

    // ---- Btn C hold = power off (always) ----
    if (M5.BtnC.wasHold()) {
        M5.Power.powerOff();
    }

    // ---- Album navigation ----
    // Default: A/C ignored for folders. Subfolders: short A/C cycle peers.
    if (!in_default_album()) {
        if (M5.BtnA.wasClicked()) change_subfolder(-1);
        if (M5.BtnC.wasClicked()) change_subfolder(+1);
    }

    // ---- Btn B ----
    // Default: long-press with yellow→red bar → random subfolder
    // Subfolder: short press → jump back to default
    if (in_default_album()) {
        if (M5.BtnB.isPressed()) {
            if (!btn_b_holding) {
                btn_b_holding = true;
                btn_b_t0 = now;
                btn_b_last_t = -1;
            }
            float t = (float)(now - btn_b_t0) / (float)HOLD_B_MS;
            if (t > 1.0f) t = 1.0f;
            if (t - btn_b_last_t >= 0.02f || t >= 1.0f) {
                draw_hold_bar(t);
                btn_b_last_t = t;
            }
            if (t >= 1.0f) {
                btn_b_holding = false;
                btn_b_last_t = -1;
                // wait for release so we don't retrigger
                while (M5.BtnB.isPressed()) { M5.update(); delay(10); }
                enter_random_subfolder();
                if (slideshow) {
                    slide_span_ms = random_slide_ms();
                    slide_due_ms = millis() + slide_span_ms;
                }
            }
        } else if (btn_b_holding) {
            // released early — cancel bar, restore image
            btn_b_holding = false;
            btn_b_last_t = -1;
            show_current();
        }
    } else {
        btn_b_holding = false;
        if (M5.BtnB.wasClicked()) {
            go_default_album();
            if (slideshow) {
                slide_span_ms = random_slide_ms();
                slide_due_ms = millis() + slide_span_ms;
            }
        }
    }

    // ---- Touch: swipe L/R, long-press = slideshow, tap in slideshow = 2× dwell ----
    auto td = M5.Touch.getDetail();
    if (td.isPressed() && td.x >= 0) {
        if (!touch_down) {
            touch_down = true;
            touch_x0 = touch_x1 = td.x;
            touch_y0 = touch_y1 = td.y;
            touch_t0 = now;
            touch_moved = false;
        } else {
            touch_x1 = td.x;
            touch_y1 = td.y;
            if (abs(touch_x1 - touch_x0) > 12 || abs(touch_y1 - touch_y0) > 12)
                touch_moved = true;
        }
    } else if (touch_down) {
        touch_down = false;
        int16_t dx = touch_x1 - touch_x0;
        uint32_t held = now - touch_t0;

        if (!touch_moved && held >= TOUCH_LONG_MS) {
            // Long press on screen → toggle slideshow on/off
            slideshow = !slideshow;
            if (slideshow) {
                slide_span_ms = random_slide_ms();
                slide_due_ms = now + slide_span_ms;
            }
        } else if (abs(dx) >= SWIPE_MIN_PX) {
            // Swipe right → next, swipe left → previous
            if (dx > 0) next_image();
            else        prev_image();
            if (slideshow) {
                slide_span_ms = random_slide_ms();
                slide_due_ms = millis() + slide_span_ms;
            }
        } else if (!touch_moved && held < TOUCH_LONG_MS) {
            // Short tap during slideshow → double remaining/current dwell
            if (slideshow) {
                slide_span_ms = slide_span_ms * 2;
                if (slide_span_ms > 60000) slide_span_ms = 60000;
                slide_due_ms = millis() + slide_span_ms;
            }
        }
    }

    // ---- Slideshow auto-advance ----
    if (slideshow && !btn_b_holding && (int32_t)(now - slide_due_ms) >= 0) {
        next_image();
        slide_span_ms = random_slide_ms();
        slide_due_ms = millis() + slide_span_ms;
    }

    delay(12);
}
