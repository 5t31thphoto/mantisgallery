// A skirt's fill as the firmware makes it: its hole + fill (full frame), the body it was given, its mask and
// the other sways' (to compare with the Studio's mg_sway.cloth_fill - see fill_compare.py).
//   fill <scene.jig> <picture.be> <out.bin>   → out.bin (plate u16 LE + hole u8), .body, .skirt, .forbid (bit masks)
#include "jiggle.cpp"
#include "common.h"
int main(int argc, char** argv) {
    using namespace jig;
    load_scene(argv[1], argv[2]);
    static uint16_t plate[320 * 240]; static uint8_t hole[320 * 240];
    int ci = -1;
    for (int i = 0; i < g_ns; i++) { const Sway& s = g_sway[i]; if (s.kind != SK_CLOTH || !s.plate) continue; ci = i;
        for (int y = 0; y < s.rh; y++) for (int x = 0; x < s.rw; x++) { int k = y * s.rw + x, j = (s.ry0 + y) * 320 + s.rx0 + x; hole[j] = s.hole[k]; plate[j] = s.plate[k]; }
        printf("hips"); for (int k = 0; k < 9; k++) printf(" %d,%d", s.armx[k], s.army[k]); printf(" back=%d\n", s.arm_back); }
    if (ci < 0) { fprintf(stderr, "no skirt\n"); return 1; }
    FILE* o = fopen(argv[3], "wb"); fwrite(plate, 2, 320 * 240, o); fwrite(hole, 1, 320 * 240, o); fclose(o);
    static uint8_t body[MASK_BYTES], fo[MASK_BYTES]; static int16_t nx[320 * 240], ny[320 * 240];
    sway_body_into(body, nx, ny, g_par.pl, g_par.n, g_sway, g_ns);
    for (int i = 0; i < g_ns; i++) sway_hips_body(g_sway[i], body);
    for (int j = 0; j < g_ns; j++) if (j != ci) for (int q = 0; q < MASK_BYTES; q++) fo[q] |= g_sway[j].mask[q];
    auto put = [&](const char* ext, const uint8_t* m) { o = fopen((std::string(argv[3]) + ext).c_str(), "wb"); fwrite(m, 1, MASK_BYTES, o); fclose(o); };
    put(".body", body); put(".skirt", g_sway[ci].mask); put(".forbid", fo);
}
