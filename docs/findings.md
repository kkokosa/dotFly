# Findings

What the model did — and did not do — when asked to fly. Each is measured with the CLI or in the
scene and recorded in detail in `PLAN.md` §12; this is the short list.

## About the model

- **Brian2 agrees with dotFly, not with the published rates file.** On the same inputs, Brian2
  2.5.1 and 2.9 reproduce dotFly's sugar → MN9 result (MN9 83 Hz, per-neuron rate correlation
  0.995 with the published file, Jaccard 0.95). The remaining difference is the published file's,
  not ours.
- **Runaway attractors at gain 1.0.** A few hertz of one-sided LC4 input ignites an optic-lobe loop
  (~450 k spikes/s that never stops; 10 Hz does not — it is coincidence-triggered), and *any*
  ORN input ignites an antennal-lobe / mushroom-body loop (Kenyon cells KCg-m / KCab-*, PAM, lLN
  local neurons). The demo keeps gain 1.0 and silences the Kenyon cells and the `lLN*` local
  neurons (4,225 neurons): every readout used is then graded and returns to baseline within a
  second of the input stopping. The uniform LIF lacks the sparsening these populations have in
  vivo (APL inhibition, high KC thresholds, LN gain control).
- **The fly's steering circuit is silent.** Driving wind (JO-C/E) and odor (ORN_DM1): DNa02 0–8 Hz,
  DNa03 ≤ 10 Hz, EPG 3–13 Hz and symmetric, PFL3 / hΔC ≈ 0, while wind strongly and symmetrically
  drives neck/GNG descending neurons (DNg73, DNge027, DNge143 at 150–175 Hz). The central complex
  is a tuned recurrent system (ring attractor, Δ7 inhibition, tonic drive) that one uniform LIF
  with no inputs to it cannot run. See the [roadmap](roadmap.md).
- **Odor and wind suppress feeding.** GRNs alone drive MN9 at 95 Hz; with ORN_DM1 and JO inputs on
  (as when she sits at food) MN9 gives 35 Hz. Real, and unexplained.

## About the circuits we read

- **LC4 → DNp04 / DNp01 / DNp02 / DNp11 ipsilaterally**, e.g. LC4_L at 120 Hz → DNp04 (left) 305 Hz,
  DNp01 255 Hz; the readouts other demos use for escape (DNa02, DNp09) stay silent under this input.
- **HS alone steers into walls.** Front-to-back flow on one eye is the same for a turn and for
  flying past a wall, so `yaw ∝ HS_R − HS_L` hugged walls and corners. Reading T4b → H2
  (back-to-front) as well separates rotation (HS one side + H2 the other → counter-turn) from
  translation (HS one side only → centering, turn away).
- **Odor direction is not in one glomerulus**: both DM1 projection neurons respond alike whichever
  antenna is stimulated. Hence the decoder's klinotaxis (surge / tumble / cast).
- **Wind direction is.** Left JO-C/E → AMMC012 11960 at 130 Hz, WED080 13072 at 134 Hz; their
  right partners 0–2 Hz; and vice versa. AMMC012 alone is used (its sides answer alike, 130/112
  Hz; WED080's 134/65 Hz made the fly circle left).
- **Detection thresholds have to come from the response curve**: DM1_lPN gives 58 / 88 / 134 / 214
  / 280 / 302 Hz for ORN input at 5 / 10 / 20 / 40 / 80 / 120 Hz, so a 40 Hz "detected" threshold
  fired on the far tail (concentration 0.03); it is 110 Hz.

## About encoding and decoding

- A fixed dark threshold saturated LC4 in a realistic room; the looming channel is relative to each
  eye's mean luminance (adaptation), as photoreceptors are.
- Two-cell readouts over 30 ms jump ±33 Hz; one-cell readouts step 0/75/100 Hz. Decisions use
  smoothed readouts (τ 0.15–0.3 s); a threshold on a single window flickers.
- A plume whose far end sweeps faster than the fly tracks is lost: the wind now turns once per 15
  minutes.
- Remembering the absolute height where odor was strongest is a cheat (a fly has no altimeter);
  the vertical search uses only the readout trend and the decoder's own last move.

## About engineering

- Godot builds in Debug and drags referenced projects into Debug: pin the library references with
  `SetConfiguration="Configuration=Release"` or the kernel is 25× slower.
- The tiered JIT leaves hot per-step methods at tier 0 for most of a short run (3.5× slower):
  `AggressiveOptimization` on every per-step method.
- Two synchronous `ViewportTexture.GetImage()` readbacks per frame cost ~12 ms (a GPU stall):
  `RenderingDevice.TextureGetDataAsync` instead. Redrawing a 166k-neuron heat map every frame cost
  ~16 ms: 15 Hz.
- Several `Simulation`s in one process pinned their workers to the same cores:
  `SimulationOptions.ThreadPinOffset / ThreadPinTotal`.
