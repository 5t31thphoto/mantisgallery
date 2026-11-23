# host-tools

Runs the firmware's own code (`../src`) on a PC: the board (display, IMU, microphone, SD, clock) is faked by
`stubs/`, nothing else. Used to check changes and to make previews before anything goes on the Core2.
Needs g++ (C++17), Python 3 with numpy/scipy/Pillow (and OpenCV for the scene builders), ffmpeg.

## The test scenes
`scenes/hiker-front` and `scenes/hiker-rear`: the hiker by the lake, seen from the front and from behind -
depth planes on a ground grid, water (falls + lake / lake), jiggle zones (chest / seat under the skirt), hair,
the skirt with its hip outline (front / `back`). Each folder holds the picture and everything made from it:
`<name>.mg` (the gallery file), `<name>.be` (its pixels for the harness), `masks.npz`, `hair_under.png`, and
`<name>.jig` - the scene. To put one on the Core2: copy `<name>.mg` to `/images/` and the `.jig` beside it
as `/images/.<name>.mg.jig`.

Rebuild a scene (only after changing its masks or setup - the files are kept so you don't have to):
    python scenes/<scene>/masks.py      # the masks (hand-placed outlines snapped to the picture) → masks.npz, masks.png
    python scenes/<scene>/build.py      # set up through the Studio's own model (tools/studio_model) → <name>.jig
A new picture: `python scenes/convert.py <picture> <dir> <name>`. The music: `python scenes/make_music.py`.

## Checks
    ./preview.sh [src dir]              # both scenes played through the firmware → out/<scene>.gif / .mp4 (+ frames)
    ./regress.sh <baseline src dir>     # every frame of every scene, 3 kinds of motion: old src vs ../src - "ALL SAME"
Harness programs (`harness/`, built by the scripts, or by hand:
`g++ -std=gnu++17 -O2 -Istubs -Iharness -I../src harness/<x>.cpp -o build/<x> -lpthread`):
- `play` - the scripted hand: still, turned, tipped, shaken, then music with dance mode on (26 s, 20 fps frames)
- `det` - fixed ticks, prints a hash of every frame (what regress.sh compares)
- `flash` - pixels the screen showed and then overwrote within one frame (flicker)
- `fill` + `fill_compare.py` - a skirt's own fill (no Studio fill in the scene) against the Studio's
- `compare.py` - two or more `play` runs side by side → one MP4 (e.g. before / after a change)

The harness lets every frame finish (no frame budget), so motion is smoother than on the device.
