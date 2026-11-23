// Host harness common: the globals the stubs need, a scene loader, the IMU poses.
// Include after "jiggle.cpp". Everything runs the firmware's own code; only the board is faked.
#pragma once
#include <string>
uint32_t g_now_ms = 0; HardwareSerial Serial; m5::M5Unified M5; SDFS SD; std::map<std::string, std::string> g_fs;
int g_push_count = 0, g_present_count = 0; m5::touch_detail_t m5::g_touch; const lgfx::IFont fonts::Font2, fonts::Font4;
std::vector<int16_t> g_audio;                        // what the microphone hears (16 kHz), from time 0
void (*g_push_hook)(int, int, uint16_t) = nullptr;   // every pixel pushed to the screen (flash.cpp)
void stub_present(const uint16_t* s) { for (int i = 0; i < 320 * 240; i++) M5.Display.fb[i] = (uint16_t)((s[i] >> 8) | (s[i] << 8)); }
static uint32_t fake_us() { return g_now_ms * 1000u; }
uint32_t (*g_micros_hook)() = nullptr;

// A scene: its .jig and its picture (raw RGB565, big-endian - the .mg's pixels in display order).
inline void load_scene(const char* jig, const char* be) {
    using namespace jig;
    FILE* jf = fopen(jig, "rb"); if (!jf) { fprintf(stderr, "no %s\n", jig); exit(1); }
    std::string jb; char b[4096]; size_t n; while ((n = fread(b, 1, sizeof b, jf)) > 0) jb.append(b, n); fclose(jf);
    begin();
    FILE* f = fopen(be, "rb"); if (!f) { fprintf(stderr, "no %s\n", be); exit(1); }
    if (fread(image().getBuffer(), 2, 320 * 240, f)) {} fclose(f);
    g_fs["/images/.p.mg.jig"] = jb; present(); load_zones("/images/p.mg");
    g_micros_hook = fake_us;
}
inline void load_audio(const char* raw) {
    FILE* af = raw ? fopen(raw, "rb") : nullptr; if (!af) return;
    int16_t s; while (fread(&s, 2, 1, af) == 1) g_audio.push_back(s); fclose(af);
}
inline void dump_frame(const char* path) { FILE* o = fopen(path, "wb"); fwrite(M5.Display.fb, 2, 320 * 240, o); fclose(o); }
inline uint64_t frame_hash() { uint64_t h = 1469598103934665603ull; for (int k = 0; k < 320 * 240; k++) { h ^= M5.Display.fb[k]; h *= 1099511628211ull; } return h; }
