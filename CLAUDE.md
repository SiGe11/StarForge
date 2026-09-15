# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A 3D real-time strategy game (StarCraft-like) in C++20 + Metal, for Apple Silicon
Macs. **Zero external dependencies** — no SDL, no CMake, no Homebrew, no engine.
Keep it that way; adding a dependency defeats the point of the project.

`README.md` is the player-facing doc (controls, unit stats, honest scope limits).

## Build and verify

```bash
make            # clean build, currently warning-free at -Wall -Wextra
make run
make bench      # ./starforge --bench 300 --no-help
make aieval     # headless AI vs four scripted opponents; GAMES= and SECS= override
make clean
```

There is **no test suite**. Verification is done two ways, both of which work
without a human at the keyboard — use them, don't ask the user to eyeball things:

```bash
./starforge --shot out.png --shot-frame 150   # render N frames, write a PNG, exit
./starforge --bench 300                        # print fps / gpu ms / draws / instances / tris
make meshcheck                                 # audit the mesh library; no GPU, no Mac
make meshcheck PNG=/tmp/sheet.png              # ...and render a contact sheet of all 12
```

`make meshcheck` is the one check that runs on a machine without Metal. It
builds the same vertex/index soup the renderer uploads and audits winding,
degenerate triangles, bounds, per-mesh `radius`/`height` and triangle budget;
with `PNG=` it also rasterises every mesh in software (`tools/mesh_preview.h`),
which is the only way to look at a model at all off a Mac. It exits non-zero on
failure, so it works as a pre-commit gate.

Other flags: `--seed N`, `--speed X`, `--cam-dist D`, `--cam-yaw R`,
`--stress N` (spawns N units per side for load testing), `--select-all`
(populates the selection at the shot frame so the command card and selection
panel render), `--ai-debug` (opens the AI inspector), `--bench-influence`
(CPU vs Metal timing for the influence field), `--edge-test SIDE` (pins a virtual
cursor to a screen edge and reports camera travel, so edge scrolling is testable
without a mouse), `--fullscreen` / `--windowed`, `--no-help`.

The game starts **full screen**; `--bench` and `--shot` force windowed so their
measurements stay comparable run to run. Anything that changes the drawable size
changes GPU timings, so don't compare a fullscreen number against a windowed one.

`tools/ai_eval.cpp` also has a `--trace <kind> <seconds>` mode that prints a
45-second-interval timeline of one match — economy, army, strategy, belief and
the raw perception signals. Reach for it before tuning any AI coefficient; every
AI bug found so far was diagnosed from that table rather than from the summary.

GUI apps here must be run with a watchdog — they do not exit on their own unless
`--bench` or `--shot` is given, and `timeout` is not installed:

```bash
( ./starforge --shot /tmp/s.png --shot-frame 150 & APP=$!; \
  ( sleep 45; kill -9 $APP 2>/dev/null ) & WD=$!; wait $APP; kill $WD 2>/dev/null )
```

For simulation work, prefer a throwaway harness over launching the GUI — the sim
is pure C++ and links standalone in about a second, running ~1800× realtime:

```bash
clang++ -std=c++20 -O2 -I src harness.cpp src/sim/{Game,Nav,Terrain}.cpp -o /tmp/h && /tmp/h
```

## Architecture

### The layering rule

`core/` and `sim/` are portable C++ with **no knowledge of Metal or AppKit**.
Only `gfx/Renderer.mm` and `app/main.mm` are Objective-C++, and the renderer is
reached through a plain C++ facade (`Renderer.h`, opaque `Impl`). Don't leak
Objective-C types into the sim, and don't let the renderer reach into game state.

The one-way contract is `RenderFrame` (in `Renderer.h`): `main.mm` fills it each
frame from game state, hands it over, and the renderer only reads it.

### Shaders are compiled at runtime, on purpose

The whole Metal Shading Language source lives as C string literals in
`gfx/Shaders.h`, split into `kShaderSource`, `kShaderSource2`, `kShaderSource3`,
concatenated **in that order** by `Renderer::init` (parts 2 and 3 use helpers —
`fbm2`, `pbrDirect`, `sampleShadow`, `skyColor`, `hash21` — defined in part 1).

This is not a stylistic choice: the machine has Xcode **Command Line Tools only**,
so the offline `metal` compiler does not exist and a `.metallib` cannot be built.
`newLibraryWithSource:` works fine without it. Don't "fix" this by adding a
`.metal` file to the build.

### RenderTypes.h is the CPU↔GPU ABI

Every struct there is mirrored by hand in the MSL in `Shaders.h`. Sizes are
locked by `static_assert` (`MeshVertex` 64, `InstanceData` 112, `Billboard` 48,
`UIVertex` 32, `FrameUniforms` 336) and must stay 16-byte multiples. **Changing a
struct means changing both sides**; the assert catches size drift but not field
reordering. `packed_float3` on the MSL side is what makes the 32-byte
`TerrainVertex` line up.

### Meshes

All meshes go into one shared vertex/index buffer pair; at upload
`Renderer::init` folds each mesh's `baseVertex` into its indices so every draw
uses `baseVertex: 0`.

Meshes are wound **counter-clockwise as seen from outside**, and the renderer
sets `frontFacingWinding: CCW` + `cullMode: back`. A primitive wound the wrong
way is invisible rather than obviously broken, so after touching `MeshBuilder`,
validate: for each triangle, the geometric normal `cross(b-a, c-a)` must agree
with the average of the three stored vertex normals. `make meshcheck` does
exactly that and runs anywhere, Mac or not.

There are two sources of geometry, and the second one is optional:

- `gfx/MeshGen.cpp` builds every mesh from boxes, cylinders, spheres and
  jittered blobs. This is the fallback and it must keep working on its own.
- `assets/models.bin` holds Blender-authored versions of ten of the twelve
  meshes, authored by `tools/blender/build_models.py` and loaded by
  `readModelPack` in MeshGen.cpp. The projectile and the selection ring stay
  procedural -- a box and a flat ring gain nothing from a modelling package.

Delete `assets/models.bin` and the game still runs, just with the primitives;
this is the same contract as the textures beside it. **Nothing in the build
depends on Blender** -- it is an offline authoring tool, `bpy` from PyPI, and
the pack is the only thing it produces.

Two properties the pack must preserve, because the rest of the game reads them
off `MeshRange`:

- `radius` sizes the **selection ring** (`max(D.radius * 1.15, meshR * 1.06)`),
  so a part hanging off the back of a model -- a drive sprocket overhanging its
  track, say -- silently inflates the ring under every unit of that type.
- `height` positions the **health bar** and the **build-in dissolve cutoff**, so
  a stack poking above the roofline lifts both.

`make meshcheck` prints both next to the triangle counts. Compare against the
procedural numbers (run it with no argument) before accepting a new pack.

### Entities

`sim/Game.h` stores entities in one vector with slot reuse plus a generation
counter. References are `EntId`, **not `Handle`** — Apple's `MacTypes.h` defines a
global `Handle`, which collides the moment Cocoa is included.

`Game::spawn()` can reallocate `ents`. Any code holding an `Entity&` across a
spawn is holding a dangling reference. `updateProduction` is written to stop
touching `e` after it spawns; preserve that property when editing the update path.

### Terrain

`sim/Terrain.cpp` generates a fractal field, **renormalises it** (value noise
clusters near its midpoint and never reaches 0 or 1 — skipping this gives a map
with no low ground and no water), then *terraces* it into plateaus where the
narrow smoothstep band between levels becomes a cliff face. Passability is a
corner-height delta test.

If the two bases end up disconnected it routes a least-resistance corridor
between them and smooths its height profile into a ramp, looping until connected.
Every generated map is therefore playable — don't remove that step.

### The AI plays by the same rules as the player

`src/ai/` is a separate layer that drives team 1 through Game's **public** `cmd*`
API — the same entry points the mouse uses. Three invariants must survive any
edit:

1. **It must not read enemy state directly.** Everything about the opponent goes
   through `Perception`, which filters on `Game::visible(team, pos)`. Iterating
   `g_->ents` for enemy information is cheating. (Reading *own* units directly is
   fine — you can always see your own army.)
2. **Every order costs actions.** `Commander::order/train/build` are the only
   paths to the game, and each spends from `ActionBudget`. Adding a code path
   that calls `cmdX` directly silently removes the APM limit.
3. **`Game` owns no AI.** `updateAI` was deleted; the app and the eval harness
   each construct a `Commander` and tick it after `Game::update`.

Pipeline: `Perception` → `OpponentModel` (HMM-style filter + behaviour profile) →
`StrategySelector` (linear contextual bandit, hand-authored prior + online
gradient) → `runMacro` / `runScouts` / `runTactics` / `runMicro`, each rate-limited
and each spending from the shared budget.

### Opponent modelling only works on persistent signals

The recurring bug in this layer is reading an **instantaneous** quantity for an
**event-like** phenomenon. Killed attackers vanish from memory immediately, so
"how much army do they have at my base right now" is almost always zero even
against a relentless rusher. Anything used for classification must be a
slow-decaying peak or EMA (`eArmyEstimate`, `eCommitPeak`, `pressure`), never a
raw per-frame count. Fresh counts are still correct for *tactical* decisions like
whether to defend right now.

### Textures are detail, not replacement

`assets/` holds eight generated textures: four materials (ground, cliff, armor,
crystal) in one `2DArray` pair, two sprite sheets (explosion, smoke), one 2x2
decal sheet (scorch) and one water ripple height map. Rules that keep them
working:

- The meshes have **no UVs**. Terrain uses world XZ; cliffs use the dominant
  horizontal axis with world height as V; objects use dominant *object-space*
  axis (world space makes panelling swim as units rotate).
- Each texture is rescaled at load so its **mean linear luminance is 0.5**, and
  the shader multiplies by two. Never normalise by a per-pixel mean -- dividing a
  near-greyscale texture by its own pixel mean yields exactly 1.0 everywhere and
  the texture silently does nothing.
- Normal maps are Sobel-derived from luminance on the CPU, with wrapped sampling.
- Albedo and the sprite sheet are `RGBA8Unorm_sRGB`; normals are plain
  `RGBA8Unorm` because they are data, not colour. Coverage masks (scorch, smoke)
  are `R8Unorm` built from luminance and deliberately **not** linearised: for a
  hand-painted mask the perceptual ramp is the one that looks right.
- **Sobel gain is per-texture and varies by an order of magnitude.** Gravel and
  rock already differ sharply between neighbouring texels, so 3-4 is plenty. A
  smooth water swell has tiny one-pixel gradients and needs **26** before any
  ripple is visible -- at 2.6 the water rendered dead flat and looked like a
  missing texture rather than a mis-tuned one.
- **Object material is chosen per draw**, not per vertex: a `float4` at fragment
  buffer 3 carrying (array slice, uv scale, detail blend, normal strength). Draws
  are already grouped by mesh, so this costs nothing. Ore get the crystal
  slice, boulders the cliff slice, everything else armour.
- Object detail is skipped where `mte.z >= 0.5` (emissive panels), which is why
  the glowing crystal spikes stay clean and only the rock they grow from picks up
  the crystal texture.

### Alpha-blended sprites live in their own pass

Every sprite in `RenderFrame::billboards` is **purely additive** -- it can only
brighten. Scorch marks have to darken the ground, so kinds 5 (scorch) and 6
(smoke) go to `RenderFrame::decals` and a second pipeline (`psDecal`, blend mode
1, `decalFS`), drawn after the water and before the additive particles. Emission
order within that list is the draw order, so scorch is emitted first.

Two traps:

- **Check a sheet's black level before trusting luminance as coverage.** The
  smoke sheet averages 27/255 between plumes; used directly that haze draws the
  whole quad as a grey rectangle. `decalFS` subtracts a floor and masks the
  border.
- **Camera-facing sprites centred on a ground-level origin get sliced.** Half the
  quad is under the terrain and the depth test cuts it along a dead-straight
  horizontal line. `billboardVS` lifts kinds above 3 by `0.85 * size` so they
  grow upward out of their origin. The flat/camera-facing split is by kind, and
  kinds 2, 3 and 5 (rings, shockwaves, scorch) are the flat ones -- an earlier
  `kind >= 1.5` test swept the explosion sheet into the flat branch by accident.

### Fog of war is rendered from the simulation's own grid

`Game` has always kept a per-team 64x64 visibility grid -- the AI is fogged by
the same data. `updateFOW()` in `main.mm` turns team 0's copy into an `RG8Unorm`
texture (r = currently seen, g = ever seen) that terrain, water, object and
sprite shaders sample through `applyFOW`.

- Cells are **eased toward their target**, not copied. The grid flips in discrete
  steps on its own timer, so a straight copy pops a 4-metre square at a time.
- Terrain that is explored but unwatched is dimmed and desaturated rather than
  hidden; terrain never seen goes near-black. Sprites are cut outright, because
  an explosion is live information rather than a memory.
- `--no-fow` disables it. `RenderFrame::fowStrength` of 0 skips the sample
  entirely, so the fallback costs nothing.

### Measuring GPU time

`MTLCommandBuffer.GPUStartTime/GPUEndTime` spans the present wait and reports
*more* time as the scene gets lighter. Use the timestamp counters instead, and
note three traps that all produced convincing but wrong numbers:

1. **One counter buffer per frame in flight.** A single shared buffer is
   overwritten by later frames before the completion handler resolves it.
2. **Counter timestamps are nanoseconds**, a different clock domain from
   `-sampleTimestamps:gpuTimestamp:` (mach ticks). Correlating the two is wrong.
3. **Per-pass spans overlap** on a TBDR -- the next pass's vertex stage runs
   during the current pass's fragment stage -- so they must not be summed. And
   the composite pass absorbs the wait for the drawable, so `RenderStats::
   gpuFrameMs` deliberately measures shadow->bloom only.

Report an **average over many frames**; a single frame's timing is far too noisy,
and run-to-run variance is about +/-1 ms.

## Gotchas that cost real debugging time

- **A texture is not automatically cheaper than the noise it replaces.** Swapping
  the sky's nine octaves of value noise for two samples of `clouds.jpg` measured
  *slower* -- five out of five alternating pairs, about 0.45 ms. The cloud deck is
  projected as `dir.xz / dir.y`, so near the horizon the UV derivative explodes,
  adjacent pixels land on different mips and the cache thrashes. Sampling with an
  explicit `level(lod)` -- the scale grows as `1/dir.y^2`, so the level is just
  its log -- removed the penalty. On a TBDR part ALU is cheap and the texture
  cache is not; a fetch only wins when neighbouring pixels want neighbouring
  texels. Check any sampler fed by a perspective divide for this.

- **The software preview must mirror `buildScene`, not approximate it.**
  `tools/mesh_preview.h` hardcodes the sun direction, intensity, colour,
  ambient and exposure from `app/main.mm`, plus the exact composite chain
  (Narkowicz ACES, explicit 1/2.2 gamma because the drawable is `BGRA8Unorm`
  rather than sRGB, then the 1.20 saturation lift). An earlier version
  estimated all five, which is the mistake the calibrated-set note warns
  about in another form: it rendered a neutral white key instead of the warm
  (1.00, 0.90, 0.74) one and at roughly half the real exposure, so it was not
  predicting what the game would show. If `buildScene` is retuned, retune this
  with it.

- **Silhouette is the whole game at RTS camera distance.** The first pass at
  the Blender models ported dimensions straight across from MeshGen.cpp and
  added bevels and panel insets to them. Triangle count went up nine times and
  the result looked the same, because every shape was still a rectangular
  prism and detail below a few pixels does not survive the camera. What
  changed the read was rebuilding the outlines: `frustum()` (a box whose top
  face has its own X and Z scale and can be slid sideways) is what most of
  them are made of, because it is the cheapest way to get a shape that is not
  a box. Judge a model by its cast shadow, not by its wireframe.

- **The old palette was about four times too bright.** Nearly every surface
  was at 0.70 albedo, against a `buildScene` calibrated so ~0.18 lands near
  mid-grey after ACES and gamma. Everything clipped toward white, so no
  geometry read regardless of how much of it there was. The model pack's
  palette spans 0.04 to 0.46; `MeshGen.cpp`'s procedural palette has **not**
  been changed to match, so the fallback still renders brighter than the pack.

- **Rock needs flat shading, and 38 degrees is not enough to get it.**
  `SMOOTH_ANGLE` averages normals across any two faces meeting at less than 38
  degrees, which is what lets a bevel blend into the curve it rounds. The
  facets of a displaced-sphere rock meet at 20 to 30 degrees, so the same rule
  smooths the entire surface and stone renders as a balloon. `mark_flat()`
  opts a part out and keeps its face normals.

- **Selecting a plate by angle fails on a plate that is itself tilted.** The
  bevel-strip trap below has a second form that a tighter angle cannot fix. A
  battered fortress wall leans about 14 degrees, so any threshold loose enough
  to admit the wall also admits its bevel trim, and one tight enough to
  exclude the trim excludes the wall as well -- measured at 15 degrees it
  selected five faces and tore the part open, at 8 degrees it selected none.
  Area separates them at any slope, because a wall is orders of magnitude
  larger than the strips around it: `face_plate()` takes the largest candidate
  and anything within 45% of it. Use it for anything that gets inset;
  `face_facing()` is only safe on a genuinely axis-aligned face.

- **Selecting faces by normal after a bevel picks up the bevel.** The Blender
  models are bevelled before anything is selected on them, and an n-segment
  bevel replaces each sharp edge with faces at evenly spaced angles -- 30 and
  60 degrees for the two-segment bevel used throughout. A loose "faces pointing
  +Z" test therefore returns the plate *and* the ring of strips wrapped around
  it, and `inset_region` over a ring of corner strips does not recess a panel,
  it tears the part open. That inflated the trooper's torso from 0.76 units
  wide to 1.67 and put two spikes through its hips -- with every triangle
  perfectly wound, so the winding audit passed. `face_facing` now thresholds on
  an angle (15 degrees), which excludes bevel faces with margin.

- **Aspect ratio cannot detect torn geometry; intended bounds can.** The
  obvious follow-up check -- flag long thin triangles -- is useless here,
  because bevelling a 5-metre plate by 2 cm legitimately produces triangles
  with aspect ratios in the hundreds. It fired on the untouched procedural ore
  mesh. What works is the one fact only the authoring side has: how big the
  part was asked to be. `sf_model.check_bounds()` records each primitive's
  requested box, carries it through transforms, and raises at export if the
  part has escaped it.

- **Radial placement has two independent ways to be wrong.** `create_cone` puts
  its first vertex at angle 0, so the flat faces of an octagonal drum are
  centred half a segment round -- placing a rib at `i*2pi/seg` straddles a
  corner. And a rotation of theta about Y sends +Z to
  `(sin theta, 0, cos theta)`, so pointing a part outward needs `pi/2 - a`, not
  `-a`. Getting the second one wrong leaves ribs lying across their wall
  rather than standing out of it, which survives a glance. Use `face_angle()`
  and `orient_radial()` in `build_models.py` rather than open-coding either.

- **A band on a tapered drum has to beat the taper at that exact height.** The
  foundry's window band was authored at radius 3.57 against a drum that is 3.62
  wide where the band sits, so it rendered nothing at all -- emissive geometry
  buried inside the hull costs triangles and produces no symptom beyond the
  detail silently not being there. `cone_radius_at()` derives the radius
  instead. `make meshcheck` reports emissive coverage over a full orbit, but
  only per mesh: it catches a model whose glow is *entirely* buried, not one
  band of several, so the contact sheet is still the review that matters.

- **`MeshBuilder::sphere` used to emit degenerate polar triangles.** The polar
  rings collapse to a point at one end, so a quad there has two coincident
  corners. They were zero-area and rasterised to nothing, but they cost index
  bandwidth and their geometric normals are pure sin/cos rounding noise, which
  makes any winding audit report them as backfacing. The poles are fans now.

- **Unit names are data and will get longer.** Command card labels are drawn from
  `UnitDef::name`, and the rename from "Depot" to "Bunkhouse" ran straight off the
  edge of the button. The label now measures itself with `textWidth` and shrinks
  to fit. Anything that renders a name into a fixed box needs the same treatment.

- **Native macOS full screen is wrong for this game.** It keeps the menu bar and
  the title bar one mouse-move from the top of the display, and pushing the
  pointer at the top edge is how the camera pans north. Full screen is therefore
  a borderless window (`styleMask` 0, verified as such by `--fs-debug`) plus
  `NSApplicationPresentationHideMenuBar | HideDock` -- *Hide*, not *AutoHide*.
  Hiding the menu bar without also hiding the Dock raises. Changing `styleMask`
  drops the first responder, so it has to be re-set afterwards, and a borderless
  window returns NO from `canBecomeKeyWindow` unless the subclass overrides it.
  Presentation options revert whenever the app is not frontmost, and an app that
  sets its own while it has focus can leave ours dropped on the way back -- so
  they are reasserted from `applicationDidBecomeActive:`, not set once.

- **The drawable is not the view, so never convert mouse input with
  `backingScaleFactor`.** Adaptive resolution resizes the drawable underneath a
  stationary cursor; `backingScaleFactor` only equals the drawable's scale while
  render scale is 1.0. Map view points through `mapMouse()`, which divides by the
  view's own size and multiplies by `pixelW/pixelH`, so the two cannot diverge.
  The failure mode is nasty: everything is correct for the first few seconds,
  then the cursor silently drifts away from the game the moment the renderer
  drops to 0.7 -- and the error is proportional to distance from the top-left
  corner, so the middle of the screen looks nearly right. `--mouse-test` checks
  the mapping at a reduced render scale without needing a window or an event.
  Anything else cached in drawable pixels (the drag anchor) has to be rescaled
  when the drawable resizes, for the same reason.

- **An adaptive-resolution controller has to predict the cost at the scale it is
  stepping *to*, and back off after failed climbs.** Two separate traps. Testing
  "is the current scale comfortable?" against a fixed threshold ignores that a
  step up multiplies pixel count by `(new/old)^2`. And a machine sharing its GPU
  with something bursty -- a chat client compositing a notification panel -- can
  pass the step-up test during every quiet spell and fail it during every busy
  one, so the resolution changes every few seconds *forever*. That reads to a
  player as the whole HUD flickering. Each undone climb therefore doubles the
  clean run the next one must see. `--adapt-test` replays measurement sequences
  through the controller without a GPU and asserts it stops changing; the shipped
  version failed three of its six scenarios, still cycling at window 158.

- **Shadow depth bias is in normalised depth over a ~519-unit frustum.** A clamp
  of `0.02` is 10 *world units*. Flat terrain never hits the clamp; a building's
  steep depth slope always does, which silently deletes all object shadows while
  leaving terrain shadows working.
- **Sun azimuth vs. camera yaw decides whether shadows are visible at all.** With
  the sun on the camera's side, every shadow hides behind its own caster and the
  scene looks like shadows are broken. The sun is currently roughly perpendicular
  to the default view axis.
- **Lighting is a calibrated set**: `sunIntensity`, `ambient` and `exposure` in
  `buildScene` are tuned together so a ~0.18 albedo lands near mid-grey after ACES
  + gamma. Raising one alone clips everything to white.
- **Don't trust `GPUEndTime - GPUStartTime` as a profiling signal.** `presentDrawable`
  is in the same command buffer, so the timestamp spans the vsync wait — it
  *decreases* as scene load increases. Use `--stress` and watch fps instead.
- **Edge-scrolling must be gated on the cursor actually being inside the view.**
  Mouse position defaults to `(0,0)`, which reads as "top-left edge" and pans the
  camera off the map before the user touches anything.
- **A capture/readback must be bound to the command buffer that encodes its blit**,
  or the previous frame's completion handler consumes the request and writes an
  empty file.
- Instance buffer offsets must stay 16-byte aligned (`InstanceData` is 112, so
  `offset * sizeof(InstanceData)` is safe).
- **Stale enemy memories will pin the AI's army at home.** Home-defence threat
  must be computed from sightings in the last few seconds and weighed against our
  own army value, or a couple of remembered troopers freeze a 3000-value army for
  the whole game.
- **A macro pass that `return`s after one action starves whatever is below it.**
  Worker production sat above army production and consumed every tick; the AI
  banked 1200 ore with two garrison. Macro issues a prioritised *burst* and
  lets `ActionBudget` be the limiter.
- **`s.pending == 0` serialises all construction.** Gate each structure on its own
  pending count, or a garrison under construction blocks the supply bunkhouse that
  would unblock production.
- Don't tune AI coefficients against the summary table — the per-archetype
  confusion matrix and `--trace` are what actually show which signal is wrong.
- **The shell here is zsh, which does not word-split unquoted `$1`.** Passing
  flags to a benchmark helper as `run "--msaa 2"` sends one argv entry, every
  flag is silently ignored, and you measure the baseline over and over while
  believing you are comparing settings. Use `${=1}`.
- **Verify the build actually succeeded before trusting a measurement.** Rapid
  edit/rebuild cycles left the binary newer than the sources, `make` skipped the
  rebuild, and several rounds of "optimisation results" were the old binary. Have
  benchmark helpers fail loudly on a build error.
- `python3` string-replacement patches fail silently when the source text does not
  match exactly. Assert on the replacement, and re-grep the file afterwards.

## Conventions

- **Camera is on the arrow keys**, not WASD, so the letter keys stay free for
  commands the way StarCraft binds them. Don't rebind camera onto letters.
- Geometry is either procedural (`MeshGen.cpp`) or Blender-authored through
  `assets/models.bin` -- see **Meshes**. Whichever is edited, the procedural
  path must still produce a complete, correct library on its own. Surface
  detail comes from the four textures in `assets/`; the game must keep working
  when that directory is missing (`U.misc.z` is the strength flag, 0 =
  procedural fallback).
- MSAA targets are `storageModeMemoryless` and resolve on store — they never
  leave tile memory on Apple GPUs. Keep new render targets in that pattern.
