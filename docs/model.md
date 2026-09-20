# Model semantics (Brian2)

dotFly reproduces the leaky integrate-and-fire model of Shiu et al. 2024
(`philshiu/Drosophila_brain_model`, `model.py`) exactly. "Exactly" is checked against Brian2 with
golden fixtures: on a 2,001-neuron / 160 k-synapse sugar circuit every spike step is identical and
membrane traces agree to 1e-11 mV over 3,000 steps.

## The model

Per neuron: `dv/dt = (v0 − v + g)/τm`, `dg/dt = −g/τs`, spike when `v > v_th`, reset `v = v_rst,
g = 0`, refractory `rfc`. Parameters: `v0 = v_rst = −52 mV`, `v_th = −45 mV`, `τm = 20 ms`,
`τs = 5 ms`, `rfc = 2.2 ms`, delay `1.8 ms`, `w_syn = 0.275 mV` per synaptic contact, `dt = 0.1 ms`.
A synapse adds `w_syn × sign(NT) × contacts` to `g` of the target, 18 steps after the source spike.
Poisson inputs add `68.75 mV` to `v` (so every accepted event fires the target) and set the
target's `rfc = 0`.

## What has to be right, and is tested

1. **Exact integration, not Euler** — the closed form per step: `a = e^{−dt/τm}`, `c = e^{−dt/τs}`,
   `b = τs/(τs−τm)·(c − a)`, `v' = v0 + a(v − v0) + b·g`, `g' = c·g`. Two FMAs and a multiply per
   neuron.
2. **`unless refractory` guards every write** to `v` and `g` except the reset: synaptic deliveries and
   Poisson events that arrive while a neuron is refractory (spike step through `t + R − 1`) are
   discarded, not accumulated. A port that accumulates them diverges within ~100 ms. Refractoriness
   is an integer countdown (Brian2 compares timesteps with `floor(t/dt + 1e-3)`).
3. **Reset zeroes `g`.**
4. **Schedule order per step**: state update → threshold → synapses (deliveries and Poisson) →
   reset. A delivery arriving in the step a neuron spikes is wiped by the reset.
5. **Homogeneous delay** of 18 steps: a ring of 19 spike lists, no per-synapse delay.
6. **Poisson targets have no refractory period** and an event landing in a spike step is discarded,
   so a continuously driven target fires at most every other step.
7. **Silencing zeroes only outgoing weights** — the neuron still integrates and spikes.
8. **Sign policy lives in the data**: ACh +1, GABA/Glu −1, DA/OA/5-HT +1; MaleCNS histamine and
   unknown transmitters are dropped (recorded in the checkpoint provenance).
9. **Precision**: the reference backend is float64 and matches Brian2 to rounding; the fast SIMD
   backend is float32 and is validated statistically (per-neuron spike counts identical on the full
   v630 sugar run, ≥ 99 % of golden spike events) and is bit-identical across thread counts.

## A note on scale

One synaptic contact raises `v` by at most ≈ 0.1575 × 0.275 mV; a single presynaptic spike needs
~162 contacts to fire a resting neuron on its own. Firing is convergence. This is also why the
model has no tonic activity: nothing fires without input, and recurrent loops either die or — when
the convergence is high enough, as in the Kenyon cells — run away. See [Findings](findings.md).
