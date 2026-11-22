# Dynamics Studio: what the user needs (from building the four test scenes)

Learned while building the four test scenes (a stream and a lake, each stylized and photoreal) with
only the Studio's own tools. These notes drive the UI rebuild.

## Selecting things

- **Cut out (keep / not strokes)** is the tool that gets clean masks. Paint a few strokes on the thing
  and a few on what it isn't: the face next to the hair, the sky behind, the ground between the feet.
  GrabCut does the edge (`Studio.cut`, `mg_select.cut`). Smart-select alone grabbed the face with the
  hair and the ground shadow with the feet.
  - UI: one tool with two stroke colours, **Keep** (left drag) and **Not** (right drag, or Alt).
  - Re-cutting with more strokes refines what's there; it doesn't start over.
- **Paint "Not" over the whole face** with a broad stroke, not one line. Skin and hair colours are
  close, especially in stylized art.
- **Figures need their soft edge.** Without the outermost pixel, a ghost outline is left behind when the
  figure slides (depth). Cut out now takes the rim the way the other tools do. Also offer **Grow 1**
  next to it, because some edges need two pixels.
- **"Not" strokes must not touch the subject.** One placed a few pixels too far in cut half an arm off
  the figure, and the water then took that arm.
  - The overlay should show each layer's mask clearly while cutting.
  - Each stroke should be undoable on its own.
- **A figure carries its own hair, cloth and pendants** (`Studio.par_with_sways`; the Core2 does the
  same on save). Without this, the hair left of her head stayed behind when she slid in depth.
  - Users shouldn't have to think about it.
  - The UI says so in one line under the figure's mask.
- **The water must stay off the figure** (other figures are protected). A water mask leaking onto an
  arm shows up as ripples on the skin.

## Hair

- **Lock strokes start at the neck, not the crown.** Hair above the neck lies on the head and must
  stay. A lock drawn from the top of the head pivots the whole fall round the crown, so it reads as
  "not attached to her head". Pixels above the root don't move.
- **Suggest locks** (`mg_sway.suggest_locks`) traces them for you: one per ~10 px of width, each
  fall's middle, root at the neck (`mg_sway.neck_row`, the narrowest row of the figure under the
  head).
  - UI: a **Suggest locks** button on the Hair tab.
  - The user can then drag a lock's root or tip, remove one, or draw their own.
- **Hair more than 12 px from every lock stays put** (`BIND_MAX`). Before this, a short lock's strip
  ran past its tip and dragged a piece of hair onto her chest. A lock's strip now runs at most 6 px
  past its tip (`EXT_MAX`).
- **Short hair that ends above the neck doesn't get a sway layer** (the photoreal stream scene). It
  doesn't swing in life either.
- **Lying on the body** (over the shoulder, down the back), hair moves only a little; hanging free, at
  most 30° from its root. This is right for the device's tilt and shakes.

## Cloth

- **A skirt moves as one sheet.** Neighbouring fold lines now pull toward each other's motion.
  Before, they swung separately and the hem went wavy and zigzagged.
- **The waistband** is the selection's top row, end to end. Drawing folds is optional; without them
  the folds come from the shading.
- **Seams opening to show skin** is the right fill, as you said, but it should only show at extreme
  swings.

## Water

- **A lake is Still water**: ripples and the wobble of reflections.
- **A waterfall or stream is a River**, with a flow line drawn down the fall or along the current.
- **The lake needs a "Not" stroke along the far shore** or it climbs into the trees. Shore lines are
  where users will need to look closely.

## Jiggle

- **Round zones on the chest**, minimum size. Under a hard shake they can push past the silhouette.
  The preview should show the zone's reach while it's being placed.

## Preview / checking

- **`mg_preview`** records the live preview as frames, GIF or video, from the same maths and a motion
  script (tilt, sway, shake, walk, dance). The four test videos came from it.
  - UI: **Record preview…**, saving a GIF or MP4 of the current scene and script, so users can share
    and compare.
- **What I checked every scene for:**
  - anything left behind when something moves (ghost edges, detached pieces);
  - masks taking a neighbour (face in hair, ground at the feet);
  - water on skin;
  - hair pivoting from the wrong place.

  The UI could flag the first two automatically: a mask edge that crosses a strong colour edge.
