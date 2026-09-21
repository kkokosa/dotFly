# dotFly

**A native .NET inference engine for fly-connectome spiking models — and an honest demo of what such simulations are.**

dotFly runs the *Drosophila* connectomes — [MaleCNS v1.0](https://male-cns.janelia.org/) (166,700
neurons, 24.9 M synapses) and [FlyWire](https://flywire.ai/) v630/v783 — as the leaky
integrate-and-fire network of [Shiu et al. 2024](https://doi.org/10.1038/s41586-024-07763-9), in
pure C# on .NET 11, fast enough to sit inside a game loop. It is a library: Godot, console apps,
tests and benchmarks link it in-process. No Python, no server, no sidecar.

> **Status: initial prototype.** Everything described here works end to end on an 8-core laptop:
> the whole MaleCNS in real time, spike-for-spike against Brian2, inside a Godot scene where a
> fly with her own eyes, antennae and legs searches a living room for sugar. The
> [roadmap](roadmap.md) lists what comes next as features, not promises.

## What you are looking at

Demos of "a fly brain playing a game" are appearing — a connectome driving Mario through a
keyboard mapping, a FlyWire brain steering a cursor. Every one of them, including this one, is
three things stacked:

| layer | in dotFly | its nature |
|---|---|---|
| the **wiring** — which neuron connects to which, how strongly, with what sign | MaleCNS v1.0 / FlyWire, every synapse | **measured** by electron microscopy |
| the **dynamics** — how a neuron turns input into spikes | one leaky integrate-and-fire model with one synaptic weight for all 24.9 M synapses (Shiu et al.), reproduced exactly | **one calibrated parameter** for the whole brain |
| the **adapters** — what feeds the sensory neurons, what the motor neurons' spikes are turned into | retina → LC4/T4, odor → ORN, wind → JO, contact → GRN; descending neurons → yaw, climb, speed, feed | **engineering**, ours |

dotFly's point is to make the third layer as thin and as visible as possible, and to say on
screen which is which. The room demo labels every number as *measured*, *calibrated* or *fixed*,
draws the brain's real spikes at real soma positions, and prints the decoder's rules next to the
readouts they use. See [Measured vs engineered](honesty.md) for the full inventory, and
[Findings](findings.md) for what the model turned out to do — and not do — when asked to fly.

## What it is not

Not a digital fly, not an upload, not a brain that plays by itself. A moving body driven by
descending-neuron rates through a hand-tuned decoder is a programmable, testable model of neural
circuitry — a valuable thing, and the only claim made here.

## Where to go next

- [Getting started](getting-started.md) — build, download the data, build a checkpoint, run the demos.
- [Library guide](library.md) — open a checkpoint, select neurons, simulate, read out, run in real time, record, adapt.
- [The room demo](demo.md) — what the fly senses, what the brain computes, what the decoder decides.
- [Inside the room demo](demo-internals.md) — the frame loop, the encoders, the readouts and the decoder, with code.
- [CLI](cli.md) — `dotfly info | build | inspect | run | bench | explore`.
- [API reference](api/DotFly.yml) — generated from the source.
