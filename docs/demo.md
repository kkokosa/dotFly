# The room demo

`samples/DotFly.Sample.Godot3D` is a Godot 4 (.NET) scene: a fly flies around a sunlit living room
and looks for sugar, with the whole MaleCNS connectome (166,700 neurons, 24.9 M synapses) running
in real time on dotFly as the only thing between her senses and her actions. The left third of
the screen shows what the network is doing; the bottom-right map shows the room, the odor streams
and her flight path.

## The loop

```
world ─► senses (encoders, hand-written)                                   inputs: 7,524 neurons
   two eye cameras, 32×32 luminance, 45° left / right ─ Retina.cs
        per eye: dark fraction & growth ──────────────────► LC4            looming
                 horizontal shift outward / inward ───────► T4a / T4b      front-to-back / back-to-front motion
                 vertical shift up / down ────────────────► T4c / T4d
   odor concentration at the two antennae ─ Odor.cs ─────► ORN_DM1 L/R    smell (one glomerulus)
   wind on the antennae (the side it comes from) ────────► JO-C/E L/R     Johnston's organ
   sugar contact when landed ─────────────────────────────► claw_tpGRN, LB3   tarsal / labellar taste

MaleCNS v1.0, Shiu et al. dynamics, gain 1.0, 4,225 neurons silenced for stability
   measured circuits:  LC4 → DNp04 L/R, DNp01 (Giant Fiber)          escape
                       T4a → HS, T4b → H2, T4c → VS                   optic flow
                       ORN_DM1 → DM1_lPN                              odor amount
                       JO-C/E → AMMC012 (one per antenna)             wind direction, lateralized
                       claw_tpGRN & LB3 → MN9                         the Shiu et al. feeding pathway
                                                                     readouts: 38 neurons in 13 groups

decoder (Behaviour.cs + decoders/dn_fixed_3d.json, hand-written, labelled Fixed)
   yaw   = 1.2·(DNp04_L − DNp04_R) + 0.2·((HS_R + H2_L) − (HS_L + H2_R)) − 0.3·(HS_R − HS_L)
           escape away · optomotor counter-turn · centering along walls
   odor  : DM1_lPN > 110 Hz → surge upwind (yaw −= 0.6·(wind_L − wind_R)); tumble when it falls;
           lost → U-turn, then casts; > 270 Hz and sugar below → land
   climb : random vertical drift (no altimeter); in odor ±12 BL/s, reversed when the readout falls
   speed : 120 body lengths/s cruise (a real fly: 100–170), 80 while tracking, 35 while casting
   taste : MN9 (0.3 s smoothed) > 15 Hz → feed; take off when it stops
```

Speeds are in body lengths per second and scale with `DOTFLY_BODY_LENGTH` (default 0.03 m — ten
times a real fly; the room, at 24 × 18 × 5 m, is at its own scale).

## What the brain does and what the decoder does

The brain supplies five graded, lateralized signals — looming side, optic-flow pattern, odor
amount, wind side, taste — through circuits that are all in the checkpoint and were found with
the CLI (`dotfly inspect --upstream-of`, `dotfly run --stimulate`), not assumed. The decision of
*where to go* (surge, tumble, cast, height search, landing) is the decoder's: the fly's own
navigation circuit — wind and odor meeting in the fan-shaped body, compared with the compass and
turned into a steering command by PFL3 → DNa02 — is silent in this uniform-LIF model (see
[Findings](findings.md)). The panels print the decoder's rules beside the readouts they use.

## The panels

- **eyes** — the two 32×32 luminance images the retina encoder receives (the motion/looming
  photoreceptors R1–R6 are achromatic; the eye cameras never see the odor rendering or the fly's
  own body), the channel rates per eye, and the smell/taste/wind input rates.
- **brain** — every neuron's recent spikes at its real soma position (dark → red → yellow →
  white), input populations in blue, readout groups as rate-sized discs; the brain to scale, the
  VNC compressed into the remaining width.
- **traces** — 10 s time-graphs of the readout groups in three panels (escape; optic flow;
  smell/taste/wind), each captioned with the decoder rule it feeds, plus the decoded actions as
  live bars and the behaviour state.
- **room** — an isometric map: furniture, columns, sugar, wind, the odor cones' centrelines, the
  flight path coloured by age with feeding stops marked, the fly with a drop line; other flies as
  grey dots.
- **HUD** — the checkpoint, neural time vs wall time (RTF, "behind"), spikes/s, the gain and the
  stability control, the causal switches, the camera mode, the state, odor at the antennae.

## The world and the rendering

- Forward+ with SDFGI, SSR, SSAO/SSIL, soft sun shadows through real window openings, a physical
  sky, procedural PBR materials — no downloaded assets except the fly.
- The odor field is the demo's physics (`Odor.cs`): three cones from the sugar, narrow at the
  source and broad downwind, buoyant (rising to the ceiling), meandering in 3-D, ending before
  walls; the wind turns once per 15 minutes. `odor_raymarch.gdshader` evaluates the *same*
  function per pixel — the gold haze with glitter is exactly what the antennae sample (the glitter
  and wisps are visual only) — and never draws between the camera and the fly.
- The fly: a 6,654-triangle rigged *Drosophila* (`models/fly.glb`, loaded at runtime), leg clips
  driven by the behaviour state (flight, landing reach, standing, takeoff), wings rowed fore–aft
  with a pitch flip at each stroke reversal as in high-speed video of real flies, at 14 Hz so the
  eye can follow (a real fly: ~200 Hz).
- Cameras (`C`): **chase** — third-person with inertia, 13.5 body lengths back, orbits to her
  front when she lands; **follow** — a trail camera sliding along the path she flew, 7.8 body
  lengths behind, never her front. Both keep her centred in the visible area by a lens shift, and
  anything between her and the camera turns translucent.

## Keys and switches

`R` recurrent transmission (off = the network is a pass-through of its inputs), `E` external
input, `V` vision, `O` smell/taste, `S` silence DNp04 + HS (they still spike, they no longer
transmit), `Space` pause, `C` camera.

Environment: `DOTFLY_BODY_LENGTH` (m), `DOTFLY_QUALITY=high|medium|low`, `DOTFLY_ODOR_RENDER=0`,
`DOTFLY_CAMERA=follow`, `DOTFLY_FLIES=N` (or `-- --flies N`: each extra fly is a full independent
simulation, cores split evenly), `DOTFLY_SCREENSHOT_FRAME=n`, `DOTFLY_SCREENSHOT_STATE=Feeding|Landed|Faded|Odor`.

## Performance (8-core Ryzen laptop, 1280×720)

| quality | odor | frame |
|---|---|---|
| `low` — no screen-space effects | off | 8.8 ms |
| `medium` (default) — SSR + SSAO + soft shadows | on | 10.3 ms (~98 FPS) |
| `high` — + SDFGI + SSIL + shadowed lamps | on | 12.8 ms (~78 FPS) |

The network keeps RTF 1.00 throughout (its own threads; ~20–35 % busy for one fly). Four flies:
2 threads each, every fly at RTF 1.00, ~90 % busy — the practical limit on 8 cores.
