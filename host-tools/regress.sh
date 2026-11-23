#!/bin/bash
# Exactness: every presented frame of every regression scene, in three kinds of motion, from the firmware in
# <baseline src dir> against the firmware in ../src. "ALL SAME" = the change didn't alter a single pixel.
#   ./regress.sh <baseline src dir> [ticks]
set -e
HT="$(cd "$(dirname "$0")" && pwd)"; NEW="$HT/../src"; OLD="$(cd "$1" && pwd)"; N=${2:-150}
B="$HT/build"; mkdir -p "$B"
g++ -std=gnu++17 -O2 -w -I"$HT/stubs" -I"$HT/harness" -I"$OLD" "$HT/harness/det.cpp" -o "$B/det_old" -lpthread &
g++ -std=gnu++17 -O2 -w -I"$HT/stubs" -I"$HT/harness" -I"$NEW" "$HT/harness/det.cpp" -o "$B/det_new" -lpthread &
wait
R="$HT/regress"; bad=0; n=0
pic() { case "$1" in F*) echo F.be;; R*) echo R.be;; pony*) echo pony.be;; *) echo shot.be;; esac; }
for env in "" "STILL=1" "BIG=1"; do
  for j in "$R"/*.jig "$HT"/scenes/hiker-front/hiker.jig "$HT"/scenes/hiker-rear/rear.jig; do
    case "$j" in *hiker-front*) be="$HT/scenes/hiker-front/hiker.be";; *hiker-rear*) be="$HT/scenes/hiker-rear/rear.be";; *) be="$R/$(pic "$(basename "$j")")";; esac
    env $env "$B/det_old" "$j" "$be" $N "$B/a.txt" >/dev/null 2>&1; env $env "$B/det_new" "$j" "$be" $N "$B/b.txt" >/dev/null 2>&1
    n=$((n+1)); cmp -s "$B/a.txt" "$B/b.txt" || { echo "DIFF ${env:-move} $(basename "$j")"; bad=1; }
  done
done
[ $bad = 0 ] && echo "ALL SAME ($n runs, $N frames each)"
