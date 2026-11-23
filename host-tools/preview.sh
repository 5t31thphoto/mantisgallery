#!/bin/bash
# Previews of the two test scenes (the hiker from the front and from behind) played through the firmware in
# ../src (or the src dir given): out/<scene>.gif and .mp4, and the raw frames in out/<scene>/ (for compare.py).
#   ./preview.sh [src dir]
set -e
HT="$(cd "$(dirname "$0")" && pwd)"; SRC="$(cd "${1:-$HT/../src}" && pwd)"
B="$HT/build"; O="$HT/out"; mkdir -p "$B" "$O"
g++ -std=gnu++17 -O2 -w -I"$HT/stubs" -I"$HT/harness" -I"$SRC" "$HT/harness/play.cpp" -o "$B/play" -lpthread
for s in "hiker-front hiker" "hiker-rear rear"; do set -- $s
  rm -rf "$O/$1"; mkdir -p "$O/$1"
  "$B/play" "$HT/scenes/$1/$2.jig" "$HT/scenes/$1/$2.be" "$O/$1" "$HT/scenes/music16k.raw" | grep -E "dance|music" || true
  python3 "$HT/harness/video.py" "$O/$1" "$O/$1"
done
