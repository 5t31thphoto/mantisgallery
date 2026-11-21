/**
 * Jiggle zones — soft-body "puppet" warps over the current photo, driven by the IMU.
 *
 * Every image is decoded into image() (a PSRAM canvas) and shown with present(). Zones for an image
 * live next to it on the SD card as a hidden ".<image file>.jig" (plain text, see jiggle.cpp).
 *
 * Editor (triple-tap a photo — undocumented Easter egg): pick Round / Half / Tear in the top bar, tap the photo to place it, drag to move,
 * A / C to shrink / grow, Rot to turn, Del to remove, B to save and watch it jiggle.
 */
#pragma once
#include <M5Unified.h>

namespace jig {

bool begin();                                // allocate PSRAM canvases; call once after M5.begin()
M5Canvas& image();                           // decode target: the pristine current image
bool image_swapped();                        // canvas stores RGB565 byte-swapped (normally true)
void present();                              // push the pristine image to the screen

void load_zones(const char* image_path);     // reads "<dir>/.<name>.jig" if it exists
void clear_zones();
int  zone_count();

void play_tick(uint32_t now_ms, bool may_draw);  // IMU → springs → redraw moving zones
bool poke(int x, int y);
int water_count();                             // water regions on this photo
bool animating();                              // water, or a zone in motion: keep the loop tight                     // tap on a zone: give it a shove. true if a zone was hit

void editor_open();
bool editor_is_open();
void editor_tick(uint32_t now_ms);           // owns touch + buttons while open

}  // namespace jig
