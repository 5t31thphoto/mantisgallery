// Deterministic run: fixed 16 ms ticks (the frame budget never trips: every frame completes in its tick), the
// IMU swaying the device. Prints a hash of every presented frame - the exactness check (regress.sh).
//   det <scene.jig> <picture.be> <ticks> [hashes.txt]      env: STILL=1 | BIG=1 (motion), DUMPDIR=dir (every 10th frame)
#include "jiggle.cpp"
#include "common.h"
int main(int argc, char** argv) {
    using namespace jig;
    load_scene(argv[1], argv[2]);
    const int N = atoi(argv[3]); uint64_t H = 1469598103934665603ull;
    FILE* hf = argc > 4 ? fopen(argv[4], "w") : nullptr;
    for (int i = 0; i < N; i++) {
        const float t = g_now_ms / 1000.f;
        if (getenv("STILL")) { M5.Imu.ax = 0; M5.Imu.ay = 1; M5.Imu.az = 0; M5.Imu.gx = 0; M5.Imu.gy = 0; }
        else if (getenv("BIG")) { float a = 0.9f * t; M5.Imu.ax = sinf(a); M5.Imu.ay = cosf(a); M5.Imu.az = 0.3f * sinf(0.7f * t); M5.Imu.gy = 60 * cosf(1.1f * t); M5.Imu.gx = 50 * sinf(1.9f * t); }
        else { M5.Imu.ax = 0.5f * sinf(1.7f * t); M5.Imu.ay = 1; M5.Imu.gy = 40 * cosf(2 * t); M5.Imu.gx = 25 * sinf(1.3f * t); }
        play_tick(g_now_ms, true);
        const uint64_t h = frame_hash();
        if (hf) fprintf(hf, "%d %016llx\n", i, (unsigned long long)h);
        if (getenv("DUMPDIR") && i % 10 == 9) { char fn[256]; snprintf(fn, 256, "%s/f%03d.raw", getenv("DUMPDIR"), i); dump_frame(fn); }
        H ^= h; H *= 1099511628211ull;
        g_now_ms += 16;
    }
    if (hf) fclose(hf);
    printf("ticks %d  hash %016llx\n", N, (unsigned long long)H);
}
