// Flicker check: per tick, the pixels the screen showed that the same tick then overwrote with something else
// (e.g. her rest pose flashing under a zone). Optionally the zones' motion per tick, to compare runs.
//   flash <scene.jig> <picture.be> <ticks> [zones.txt]
#include "jiggle.cpp"
#include "common.h"
static uint16_t first[320 * 240]; static uint8_t nw[320 * 240];
static void hook(int x, int y, uint16_t v) { int i = y * 320 + x; if (!nw[i]) first[i] = v; if (nw[i] < 255) nw[i]++; }
int main(int argc, char** argv) {
    using namespace jig;
    load_scene(argv[1], argv[2]);
    g_push_hook = hook;
    const int N = atoi(argv[3]); long flash = 0, fticks = 0, maxf = 0;
    FILE* zf = argc > 4 ? fopen(argv[4], "w") : nullptr;
    for (int i = 0; i < N; i++) {
        const float t = g_now_ms / 1000.f;
        M5.Imu.ax = 0.5f * sinf(1.7f * t); M5.Imu.ay = 1; M5.Imu.gy = 40 * cosf(2 * t); M5.Imu.gx = 25 * sinf(1.3f * t);
        memset(nw, 0, sizeof nw);
        play_tick(g_now_ms, true);
        long fl = 0; for (int k = 0; k < 320 * 240; k++) if (nw[k] >= 2 && first[k] != M5.Display.fb[k]) fl++;
        flash += fl; if (fl) { fticks++; if (fl > maxf) maxf = fl; }
        if (zf) { fprintf(zf, "%d", i); for (int z = 0; z < g_n; z++) fprintf(zf, " %.5f %.5f %.5f %.5f", g_z[z].dx, g_z[z].dy, g_z[z].vx, g_z[z].vy); fprintf(zf, "\n"); }
        g_now_ms += 16;
    }
    printf("ticks with flashes %ld/%d, flashed px total %ld, max in a tick %ld\n", fticks, N, flash, maxf);
}
