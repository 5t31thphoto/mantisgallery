// The preview run: a scripted hand plays the scene through the firmware - held still, turned left and right,
// tipped forward and back, a few quick shakes, then music with dance mode on (swipe up) - and every 5th
// 10 ms tick's frame is dumped (20 fps), with what was happening in script.txt.
//   play <scene.jig> <picture.be> <out dir> [music16k.raw]
#include "jiggle.cpp"
#include "common.h"
static float ease(float u) { u = u < 0 ? 0 : (u > 1 ? 1 : u); return u * u * (3 - 2 * u); }
int main(int argc, char** argv) {
    using namespace jig;
    load_audio(argc > 4 ? argv[4] : nullptr);
    load_scene(argv[1], argv[2]);
    const char* out = argv[3];
    const float T = 26.0f; const int every = 5, step = 10;
    FILE* cap = fopen((std::string(out) + "/script.txt").c_str(), "w");
    for (int i = 0; g_now_ms < T * 1000; i++) {
        const float t = g_now_ms / 1000.f;
        float roll = 0, pitch = 0, droll = 0, dpitch = 0, jx = 0, jy = 0;
        const char* what = "Held still: depth drift, the water";
        if (t >= 3 && t < 8.5f) { float u = t - 3, A = 18.f * ease(u / 0.8f) * (1 - ease((u - 4.7f) / 0.8f)), w = 2 * 3.14159f / 2.75f; roll = A * sinf(w * u); droll = A * w * cosf(w * u); what = "Turned left and right: depth, the water leans, hair and skirt swing"; }
        else if (t >= 8.5f && t < 11) { float u = t - 8.5f, A = 14.f * ease(u / 0.6f) * (1 - ease((u - 1.9f) / 0.6f)), w = 2 * 3.14159f / 2.2f; pitch = A * sinf(w * u); dpitch = A * w * cosf(w * u); what = "Tipped forward and back"; }
        else if (t >= 11 && t < 13.5f) { float u = t - 11, e = expf(-1.6f * fmodf(u, 0.85f)); jx = 1.3f * e * sinf(2 * 3.14159f * 5.0f * u); jy = 0.4f * e * sinf(2 * 3.14159f * 3.3f * u + 1); what = "A few quick shakes: jiggle, hair, skirt, ripples"; }
        else if (t >= 13.5f && t < 16) what = "Settling";
        else if (t >= 16 && t < 24) what = "Music playing - dance mode on";
        else if (t >= 24) what = "Music stopped";
        if (t >= 16 && !g_dance_on) dance_toggle();
        if (t >= 24.5f && g_dance_on) dance_toggle();
        const float r = roll * 3.14159f / 180, p = pitch * 3.14159f / 180;
        M5.Imu.ax = sinf(r) * cosf(p) + jx; M5.Imu.ay = cosf(r) * cosf(p) + jy; M5.Imu.az = sinf(p);
        M5.Imu.gy = droll * 0.8f; M5.Imu.gx = dpitch; M5.Imu.gz = 0;
        play_tick(g_now_ms, true);
        if (i % every == 0) {
            char fn[256]; snprintf(fn, 256, "%s/f%04d.raw", out, i / every); dump_frame(fn);
            fprintf(cap, "%d %.2f %s\n", i / every, t, what);
        }
        g_now_ms += step;
    }
    fclose(cap);
}
