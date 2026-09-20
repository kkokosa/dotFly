# Measured vs engineered

Every number and rule in dotFly is one of three kinds, and the demos say which on screen:

- **measured** — read from the connectome: which neuron connects to which, with how many synapses,
  with what neurotransmitter sign (from the release's prediction), where each soma is.
- **calibrated** — a parameter of the published model: the leaky integrate-and-fire constants and
  the single synaptic weight `w_syn = 0.275 mV` that Shiu et al. fitted on the sugar → proboscis
  experiment. dotFly reproduces that model exactly (spike-for-spike against Brian2). The gain
  multiplier (`SynapticGain`) is shown on the HUD; anything other than 1.0 is labelled *calibrated*.
- **fixed / trained** — ours: the sensory encoders, the motor decoders, the stability control,
  and any readout adapter (hand rules, a linear map, an ONNX/ML.NET model — the graph itself never
  learns).

## Inventory for the room demo

| part | kind | what exactly |
|---|---|---|
| MaleCNS v1.0 wiring, 166,700 neurons / 24.9 M synapses, signs, soma positions | measured | `superclass IS NOT NULL` filter; unclear/histamine edges dropped as no-ops (counts in the provenance) |
| LIF dynamics, delays, refractoriness, Poisson semantics | calibrated (Shiu et al.) | see [Model semantics](model.md) |
| LC4 → DNp04 L/R, DNp01 (escape) | measured | found with `dotfly run --stimulate type:LC4@L` |
| T4a → HS, T4b → H2, T4c → VS (optic flow) | measured | lateralized exactly as the literature says |
| ORN_DM1 → DM1_lPN (odor amount) | measured | both PNs answer alike to either antenna: no direction |
| JO-C/E → AMMC012 L/R (wind direction) | measured | one antenna's JO drives one side only (130 vs 0–2 Hz) |
| claw_tpGRN, LB3 → MN9 (feeding) | measured | with odor and wind inputs on, MN9 drops from 95 to 35 Hz |
| silencing 4,225 Kenyon cells + antennal-lobe `lLN*` | fixed (stability control) | without it any ORN input ignites a loop that never stops; shown on the HUD |
| retina: luminance → dark fraction, image shift → LC4/T4 rates | fixed (encoder) | replaces what the optic lobe computes from photoreceptors — see the [roadmap](roadmap.md) |
| odor concentration → ORN_DM1 rate; wind side → JO-C/E rate; contact → GRN rate | fixed (encoders) | |
| yaw from DNp04, HS/H2; speed from GF; feed from MN9 | fixed (decoder) | direct readouts of the measured circuits |
| surge upwind, tumble, cast, U-turn, height search, landing trigger | fixed (decoder) | the *navigation decision*; the brain's own circuit for it is silent in this model |
| body: flight model, collisions, bumper and hop reflexes, wing beat, leg clips | fixed (body) | labelled on the HUD when they act |

## What is not derivable from the wiring at all

- Odor *direction* from one glomerulus (both DM1 projection neurons respond alike).
- Wall walking, grooming, courtship — nothing in the readouts we use distinguishes them; they
  would be pure scripting and are left out.
- Anything needing neuromodulation, gap junctions or dendritic computation: the model has none.

## Comparison with other connectome demos

"Fly Mario"-style demos run the same Shiu model and map a few descending neurons (DNa02, DNp09)
to keyboard buttons through a hand-made table. That table is an adapter in the same sense as ours;
the difference dotFly tries to make is (a) the inputs are the fly's own senses in a physical world
rather than a stimulus schedule, (b) every readout is a circuit shown to exist and to respond in
the checkpoint, (c) the adapters are printed on screen and documented as such, and (d) what the
model *cannot* do is measured and stated: under wind and odor input, DNa02 fires 0–8 Hz here.
