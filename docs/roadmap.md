# Roadmap

What exists is the **initial prototype**: the engine (M0–M4 in `PLAN.md`) and the room demo. The
items below are features, tracked as GitHub issues; the order is the order that moves the demo
from "adapters around a connectome" toward "the connectome runs the fly".

## F1 — Move the sensory boundary outward: vision

We inject rates into T4 and LC4, i.e. C# computes what the optic lobe should compute from light.

1. Test whether the uniform-LIF optic lobe computes motion and looming from photoreceptor /
   lamina input: drive R1–R6 (or L1/L2) with a moving pattern via the CLI and check whether
   T4a–d become direction-selective and LC4 expansion-tuned.
2. If it does: replace the retina encoder by a photoreceptor-level one (luminance per ommatidium →
   photoreceptor rate) and let the connectome compute the rest.
3. Landing triggered by visual expansion (LC4 / Giant Fiber) gated by odor, as in flying flies
   (van Breugel & Dickinson 2012), instead of "odor high and sugar below".
4. Read the object-approach cells (LC10 / LC11 → their descending targets) if they respond — a
   measured "approach dark objects" behaviour.

Acceptance: no C# computes what a neuron in the checkpoint computes; encoders reduce to physically
defined light / odor / wind / contact transductions.

## F2 — Connectome-native navigation (per-cell-type fitting)

The fly's own navigation circuit — wind (WPN → LNa → PFNa) and odor meeting in the fan-shaped body
(hΔC), the EPG compass, PFL3 turning goal − heading into a DNa02 steering command — is in the
checkpoint and silent in the uniform LIF. Following Lappalainen et al. 2024 (connectome-constrained
model of the visual system: synapses fixed, a few parameters per cell type fitted, physiology
predicted):

1. Recover the EPG / PEN / Δ7 ring order from the connectivity; inject a compass bump; test
   persistence, single-ness, rotation under PEN drive, and whether PFNa wind + FB odor reach PFL3 /
   DNa02.
2. Per-cell-type parameters (synaptic gain, τm, tonic current, threshold) as type-gain tables in
   the checkpoint and `SimulationOptions.TypeGains`; a subgraph checkpoint extractor (CX + wind +
   olfactory + steering, ~5–10 k neurons) for hundreds× real-time evaluation.
3. CMA-ES / evolution strategies against quantitative targets (EPG bump persistence and rotation,
   hΔC and PFN wind tuning, PFL3 L−R ∝ sin(goal − heading), DNa02 asymmetry) — or a differentiable
   rate surrogate trained by backprop and ported back.
4. Validation on behaviour never fitted: ON surge / OFF cast in the room with the decoder's
   surge/cast rules switched off and yaw read from DNa02 L−R.

Risk: the LIF with per-type gains may still not hold a bump; then a neuron-model term
(adaptation, conductance synapses) is next — still the connectome, with better dynamics.

## F3 — CUDA backend

PTX kernels through the driver API, CUDA graphs, lanes (batched trials) on the GPU; ≥ 10× real
time on one GPU; cross-backend validation against the CPU reference.

## F4 — Lanes, Native AOT, packages, docs

Batched trials on CPU, a Native AOT CLI, NuGet packages, the API reference published.

## Literature behind F1 and F2

Shiu et al. 2024 (Nature) — the model. Lappalainen et al. 2024 (Nature) — connectome-constrained
fitting. Dombrovski et al. 2023 (Nature) — LC4 → DNp02/DNp04 gradients. Álvarez-Salvado et al.
2018 (eLife), van Breugel & Dickinson 2012/2014 — surge and cast. Suver et al. 2019, Okubo et al.
2020, Matheson et al. 2022 — wind into the central complex. Seelig & Jayaraman 2015, Kim et al.
2017, Turner-Evans et al. 2017/2020, Hulse et al. 2021 — the compass ring attractor. Westeinde et
al. 2024, Mussells Pires et al. 2024 — PFL3 → DNa02 steering. Maisak et al. 2013, Kim, Fitzgerald &
Maimon 2015, Straw, Lee & Dickinson 2010 — motion vision and its use.
