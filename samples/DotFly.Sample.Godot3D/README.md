# dotFly Godot 3D sample — closed visual loop, odor search, feeding

> The full story of this scene — what the fly senses, what the brain computes, what the decoder
> decides, and the code behind it — is in the docs: [the room demo](https://kkokosa.github.io/dotFly/demo.html)
> and [inside the room demo](https://kkokosa.github.io/dotFly/demo-internals.html).

A Godot 4 (.NET) 3D scene in which a fly flies around a sunlit living room (windows, a couch, two
tables, four columns) and finds sugar, with the MaleCNS connectome (166,700 neurons, 24.9 M
synapses) running in real time on dotFly as the only thing between its senses and its actions. The
left third of the screen shows what the network is doing — its eyes, its brain lit by real spikes,
and the readout time-graphs wired to the decisions; bottom-right an isometric map of the room with
the flight path.

```
world ──► senses (encoders, hand-written)                              inputs   7,189 neurons
   two eye cameras, 32×32 each, 120° FOV, 45° left / right ─ Retina.cs   (luminance only, see below)
        per eye: dark fraction & growth ─────────────────► LC4          looming
                 horizontal shift outward / inward ──────► T4a / T4b    front-to-back / back-to-front
                 vertical shift up / down ───────────────► T4c / T4d
   odor field at the two antennae ─ Odor.cs ─ 120 Hz/unit ─► ORN_DM1 L/R  smell (one glomerulus)
   sugar contact when landed on a patch ─────────────────► claw_tpGRN    tarsal taste
                                                          ► LB3*         labellar taste
   wind on the antennae (side it comes from) ───────────► JO-C/E L/R    Johnston's organ, wind direction

MaleCNS v1.0 (dotFly, gain 1.0, 4,225 neurons silenced for stability, see below)
   measured circuits:  LC4 → DNp04 L/R, DNp01 (Giant Fiber)             escape
                       T4a → HS (HSE/HSN/HSS), T4b → H2, T4c → VS       optic flow (rotation vs translation)
                       ORN_DM1 → DM1_lPN                                 odor detection
                       claw_tpGRN & LB3 → MN9 (proboscis motor neuron)   the Shiu et al. feeding pathway
                       JO-C/E → AMMC012, WED080 (one per antenna)       wind direction, lateralized
                                                                       readouts 38 neurons in 13 groups

decoders/dn_fixed_3d.json + Behaviour.cs (decoder, hand-written, labelled Fixed)
   yaw   = 1.2·(DNp04_L − DNp04_R)                          escape away
         + 0.2·((HS_R + H2_L) − (HS_L + H2_R))              optomotor counter-turn (rotation)
         − 0.3·(HS_R − HS_L)                                 centering (translation along a wall)
   climb = 0.07·mean(VS) + random drift ±8 BL/s              no altimeter: heights are sampled at random;
           in odor ±12 BL/s, reversed half the time the readout falls    vertical klinotaxis
   speed = 120 + 1.0·DNp01 body lengths/s                    a real fly cruises at 100–170 BL/s; GF burst
   odor  : DM1_lPN > 110 Hz = detected → surge upwind while the rate rises (yaw −= 0.6·(wind_L − wind_R)),
           tumble 150°/s for 0.65 s when it has fallen 25 Hz/s for 0.25 s; 80 BL/s;
           lost (< 110 Hz, remembered 8 s) → U-turn back into the tube, then zigzag casts (35 BL/s)
           of growing duration until the readout returns; DM1_lPN > 270 Hz = near → descend, land
   taste : landed → GRN input → MN9 (smoothed 0.3 s) > 15 Hz = feed; take off when MN9 stops
FlyBody3D.cs: CharacterBody3D flight (body length 0.03 m = 10× a real fly, `DOTFLY_BODY_LENGTH`),
              pitch with altitude, landing on tables and the floor, panels
models/fly.glb + BlurredWings.cs: the rigged fly (leg clips per state, rowing wing beat);
              FlyModel.cs is the procedural fallback
```

## What is measured and what is not

- **Measured:** every synapse. LC4 reaches the DNp04/GF escape neurons ipsilaterally, T4 reaches the
  HS/VS lobula-plate tangential cells with the right lateralization, ORN_DM1 reaches the DM1
  projection neurons, and the tarsal/labellar gustatory neurons reach MN9 — all found in the
  checkpoint (`dotfly inspect --upstream-of <MN9 id>` / `--downstream-of`), not assumed.
- **Measured, and it mattered:** HS cells alone cannot tell a turn from flying along a wall (both
  give front-to-back flow on one eye), and the fly steered into walls and hugged corners. Reading
  the back-to-front cells too (T4b → H2, both in the checkpoint) separates the two: rotation lights
  HS on one side and H2 on the other → counter-turn; translation lights HS on one side only → the
  centering term turns away. The rule is the decoder's; the cells and their tuning are the wiring's.
- **Measured: wind direction.** The antennae's Johnston's organ (JO-C/E, 203 neurons per side)
  reaches AMMC/wedge interneurons that answer one antenna only — left JO drives AMMC012 11960 at
  130 Hz (and WED080 13072 at 134 Hz) while their right partners stay at 0–2 Hz, and vice versa.
  The readout uses AMMC012 alone: its two sides answer alike (130 / 112 Hz), whereas WED080's do
  not (134 / 65 Hz) and biased the surge to the left. That is the
  cue real flies use to *surge upwind* on odor (Álvarez-Salvado et al. 2018); the tubes run
  downwind from the sugar, so surging upwind along one leads to it.
- **Not derivable from the wiring, done by the decoder:** *odor direction*. A single glomerulus
  reports concentration; both DM1 PNs respond alike whichever antenna is stimulated, so the search
  combines the upwind surge with klinotaxis (tumble when the PN rate falls) and casts when it is
  lost, implemented in `Behaviour.cs`. Likewise landing,
  take-off and the flight model are the demo's rules. **Wall walking is not in this demo**: nothing in
  the readouts distinguishes a wall from air, so it would be pure scripting and is left out.
- **Every number** in `dn_fixed_3d.json` is a hand choice. The HUD labels the decoder `Fixed`.

## Stability control (gain 1.0)

At the model's calibration (`w_syn` = 0.275 mV) the whole network has self-sustaining attractors:
a few hertz of one-sided LC4 ignites an optic-lobe loop (~450 k spikes/s), and *any* ORN input
ignites an antennal-lobe / mushroom-body loop (Kenyon cells, PAM, lLN local neurons) that never
stops. Rather than lowering the gain, this demo **silences** (zero outgoing weights of) the Kenyon
cells and the antennal-lobe `lLN*` local neurons — 4,225 neurons, `FlyBrain3D.StabilityControl` —
after which every readout is graded in its input and returns to baseline when the input stops
(verified with `dotfly run --stim-seconds`). The HUD shows it: *gain 1.00 (model) control: 4225
neurons silenced (KC + AL LNs)*. Set `StabilityControl` to `""` to watch the runaway.

## The room, the eyes and the odor

- The room is rendered with Godot's Forward+ pipeline: SDFGI global illumination (bounced light
  from the sun patches and the lamps), screen-space reflections on the floor, SSAO/SSIL, soft PCSS
  sun shadows through real window openings, a physical sky, ACES tonemapping, and PBR materials
  with procedural albedo + normal maps (`ProceduralTextures.cs`, FastNoiseLite) — no downloaded
  assets. Furniture: couch with cushions on a kilim, coffee table, dining table with chairs, four
  columns, bookshelf, floor lamp, pendants, curtains, a plant, a picture, a door.
- **The fly's eyes see luminance only** — the photoreceptors that feed motion vision and looming
  (R1–R6 → T4/T5, LC4) are achromatic; flies do have colour receptors (R7/R8, UV/blue/green) but
  those feed a pathway this demo does not read. Two compound eyes → two cameras, each looking 45°
  to its side with a 120° field, 32×32 samples (Drosophila has ~800 ommatidia per eye). The eye
  panel shows the very L8 images the retina encoder receives. "Dark" for the looming channel is
  relative to each eye's mean luminance (adaptation), so a dim room and a bright arena both work.
- **The odor is drawn where it is, raymarched.** The field is the demo's physics (`Odor.cs`: three
  cones, 0.5 m wide and 0.35 m tall at the sugar and growing as (1 + distance / 2.5 m), 18 m long,
  the axis diluted by the square root of that growth so "near" means within ~2 m; cores starting
  0.45 m above the surface the sugar sits on, rising 28 cm per metre downwind (warm sugar air is
  buoyant: they reach the ceiling and spread along it) and meandering in three dimensions — two
  sinusoids sideways, one vertically, drifting slowly — with the wind turning a full circle every
  15 minutes with a wobble; a cone ends 1.5 m before its axis meets a wall and the field fades to
  zero within 0.8 m of any wall, so nothing pools in corners), so it is known everywhere;
  `odor_raymarch.gdshader` marches every pixel of the chase camera through the room (48 jittered
  steps), evaluates the same function, stops at the depth buffer, draws a soft-edged tube around
  the centreline (never between the camera and the fly) in warm gold full of glitter — motes at hashed positions in 3.5 cm cells (no
  tiling texture, so no grid), each with its own flicker, bright enough to bloom, like dust
  catching the sun — and multiplies it by a wind-advected 3D noise for wisps and gaps (the antennae sample the smooth
  mean field — the noise is only visual). The eye cameras do not draw that layer; the room map
  draws the cones' centrelines so you can see the streams the fly is trying to follow.
- **The fly cannot see itself**: its meshes live on a visual layer the eye cameras cull (with 120°
  eyes at the head, the wings and legs were in view). She casts a shadow like anything else.
- Two cameras (`C`): chase — third-person with inertia (a low-passed heading, ≈0.7 s), 13.5 body
  lengths back, orbiting to her front when she lands; follow — a trail camera sliding along the
  path she flew, 7.8 body lengths behind, never showing her front. Both keep her centred in the
  visible area by a lens shift, and anything between her and the camera turns translucent.
- **The fly model.** `models/fly.glb` is a 6,654-triangle low-poly *Drosophila* with a 25-bone
  leg rig (body PBR atlas, wing veins baked; a reconstruction referencing "Fruit Fly Drosophila"
  by yuqian.ye / Alicia Hidalgo Lab on Sketchfab — a visual asset, not a calibrated biological
  model). It is loaded at runtime with `GltfDocument` (no editor import), auto-scaled to the body
  length (0.03 m by default), faced −Z (`ModelYawDegrees` = 180), placed with its feet where the
  landed body rests, and put on the fly's own visual layer. Its five clips are driven by the
  behaviour state: `flight` (fore and middle legs retracted, hind legs trailing), `landing_pose`
  (forelegs reaching) while descending onto sugar, `standing` when landed and feeding, `takeoff`
  → `flight` when leaving. Its wings are separate hinge nodes (`Fly_Wing_L/R`), so
  `BlurredWings.cs` beats them in flight the way high-speed video shows a fruit fly does
  (Bergou, Ristroph, Cohen & Wang; physics.aps.org story v25/st13): the wings **row forward and
  back in a near-horizontal stroke plane through ~140° and flip their pitch (±45°) at each stroke
  reversal**, leading edge forward on the forward stroke — like feathering an oar. The beat runs
  at 14 Hz so the eye can follow it (a real fly: ~200 Hz), with two short trailing ghosts of each
  wing for the blur; on landing the wings fold back over the abdomen.
  Any other glTF works the same way (a model without those wing nodes just flies rigidly;
  `ModelPath` = "" uses the procedural fly in `FlyModel.cs`).

## The panels

- **eyes** — the two 32×32 luminance renders (L, R), the retina channel rates per eye (LC4,
  T4a–d), and the smell/taste rates below.
- **brain** — the checkpoint's real soma positions projected to a map (head left, ventral nerve cord
  right; the texture takes the panel's aspect so the map fills it, the brain — where all inputs
  and readouts are — is drawn to scale and the VNC gets the remaining width, compressed only if it
  does not fit — the neck is found as the narrowest z-bin — and labelled);
  every neuron's recent spikes colour its soma (dark → red → yellow → white, auto-scaled).
  Blue pixels are the input populations, coloured discs the readout groups sized by their rate.
  This is the propagation itself: watch it light from the optic lobes / antennal lobe outward.
- **traces** — three 10 s time-graphs of the readout groups (escape; optic flow; smell & taste),
  each captioned with the decoder rule it feeds, and the decoded actions (yaw / climb / speed) as
  live bars with the behaviour state, its yaw source, the odor flag and the total feeding time.
- **room** (bottom-right) — an isometric projection of the room with furniture, columns, sugar,
  the wind, the flight path coloured by age (feeding stops as dots) and the fly with a drop line.

The left column is full height and scales with the window (34 % of its width); labels are
sized for a 1280×720 window and grow with it.

## More flies

`Godot_v4.7.1-stable_mono_win64.exe --path . -- --flies 4` (or `DOTFLY_FLIES=4`) spawns extra
flies. **Each one is a full, independent MaleCNS simulation** — its own `Simulation` with its own
seed, eyes, antennae, readouts and decoder — not a puppet: the physical cores are split evenly
and each simulation's workers are pinned to their own cores (`SimulationOptions.ThreadPinOffset` /
`ThreadPinTotal`, added for this). Only the primary fly (orange on the room map) has panels, HUD,
camera and keys; the others just live in the room and show as grey dots on the map. On the 8-core
laptop 4 flies × 2 threads run at RTF 1.00 with ~90 % of each simulation thread busy — that is the
practical limit there; more flies fall behind real time (the HUD's `behind` shows it). Memory is
~150 MB per fly (the blocked CSR).

## Performance

Measured with `--print-fps` on the 8-core laptop at 1280×720 (the network itself runs on its own
threads at RTF 1.00 throughout):

| quality (`DOTFLY_QUALITY` / `Arena.Quality`) | odor | frame |
|---|---|---|
| `low` — no screen-space effects | off | 8.8 ms |
| `medium` (default) — SSR + SSAO + soft sun shadows | on | 10.3 ms (~98 FPS) |
| `high` — + SDFGI + SSIL + shadowed lamps | on | 12.8 ms (~78 FPS) |

Two things used to cost 28 ms per frame and are now gone: the eye images were read back with
`GetImage()` (a GPU→CPU stall, twice per frame, ~12 ms) — they now come back asynchronously via
`RenderingDevice.TextureGetDataAsync` (a frame late, which a fly does not mind); and the brain map
recomputed its 166,700-neuron heat map and uploaded the texture every frame (~16 ms) — it now
updates at 15 Hz and bakes the input-population dots into the texture. `DOTFLY_ODOR_RENDER=0`
turns the raymarch off; `DOTFLY_PERF=nopanels|nomap|noshadow|nopcss` isolates the rest.

## Keys

- `R` recurrent transmission off → the network becomes a pass-through of the inputs
- `E` external input off → the network runs on nothing
- `V` vision off → retina channels held at 0 Hz
- `O` smell/taste off → ORN and GRN channels held at 0 Hz
- `S` silence DNp04 + HS → the readout cells still spike but no longer transmit
- `Space` pause
- `C` camera: **chase** (third-person with inertia, 13.5 body lengths back, orbits to her front when
  she lands) or **follow** (a trail camera: it slides along the path she actually flew, 7.8 body
  lengths behind her along the trail, so in a turn it goes through the same bend she did and keeps
  her rear in view; if she reverses and flies at it, it swings behind her heading — you never see
  her front). In both she stays centred in the visible area by a lens shift (an off-axis frustum),
  not by moving the camera — a camera translated beside her trail shows her flank in every turn
  toward it. In both, anything
  between her and the camera (walls, columns, furniture) turns translucent while it is in the way.

## Run

1. Build the MaleCNS checkpoint (repository README): `.data/malecns-v1.0/malecns-v1.0-superclass.dfb`.
2. `DOTNET_ROLL_FORWARD_TO_PRERELEASE=1` in the environment (until .NET 11 GA).
3. Open this folder in Godot 4.7 (.NET) and press Play, or run the real executable (not the winget shim):
   `Godot_v4.7.1-stable_mono_win64.exe --path .` — add `--quit-after 4800` for an 80 s unattended run.
   `DOTFLY_SCREENSHOT_FRAME=<n>` saves `screenshot.png` at that frame, `DOTFLY_SCREENSHOT_STATE=Feeding`
   0.8 s into that behaviour state (handy for unattended runs).
   `--headless` runs the network but the eye sees nothing (no renderer).

The HUD prints once per two seconds to the console as well, so `--quit-after` runs are inspectable
(state transitions Flying → Landing → Landed → Feeding → TakeOff appear there). On an 8-core laptop
the network keeps up with the wall clock (RTF 1.00, 0 ms behind) at ~20–35 % of the simulation
thread; a typical run finds sugar and feeds within a minute (the volumetric fog needs a GPU that
runs Forward+).
