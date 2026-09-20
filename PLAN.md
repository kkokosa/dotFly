# dotFly — a native .NET inference engine for fly-connectome spiking models

> Brainstorm result, 2026-09-18. Companion to the research note
> `Fruit fly brain connectomes and simulations (MaleCNS).md`. This is a plan, not a status report:
> every number that is not marked *measured* is an estimate to be verified by the benchmarks in §8.

## 0. One-paragraph pitch

dotFly runs the MaleCNS (166,700 neurons / 25.58 M directed edges / 124.2 M contacts) and smaller
"checkpoints" (FlyWire v630/v783 for reproducing Shiu et al., hand-cut subcircuits, shuffled controls)
as leaky integrate-and-fire point-neuron networks, in pure C# on .NET 11, fast enough to sit inside a
game loop. It is the fly-brain analogue of dotLLM: a memory-mapped model file, unmanaged SoA state,
SIMD kernels on CPU, PTX kernels on CUDA via the driver API, zero allocations on the hot path, and a
demo-facing API (stimuli in, spikes/rates out, full inspection) so that Godot or console demos in C#
never touch Python.

What it is **not** (and the README must say so, per the note): not a digital animal, not a fly that
"plays games" by itself. The graph is measured; the neuron equations, the encoder and the decoder are
engineering choices, and the engine must make them explicit and switchable so demos can show what the
wiring contributes (causal controls, §7.6).

## 1. Scope and goals

| Goal | Concretely |
|---|---|
| G1 Faithful reference | Bit-for-bit (float64) match of the Shiu et al. Brian2 model on small graphs with fixed input spike trains; statistical match (Eon-style metrics) on the sugar→MN9 experiment. |
| G2 Fast | ≥ 1× real time (10,000 steps of 0.1 ms per wall second) on a 16-core desktop CPU for typical demo activity; ≥ 10× on a single consumer GPU; batched trials for experiments. |
| G3 Demo-friendly | Stimuli, readouts, recording/replay, inspection, real-time clock driver, thread-safe snapshotting for a game thread. Works in Godot 4 (.NET) and plain console apps. |
| G4 Checkpoints | One binary format for MaleCNS v1.0, FlyWire v630/v783, and any derived subgraph or control (shuffled, ablated), with provenance pinned in the header. |
| G5 Honest | Neural time, simulated duration and wall time reported separately; every fixed/calibrated/trained interface element named as such. |

Non-goals for v1: body/biomechanics (NeuroMechFly-class), multi-compartment neurons, gap junctions,
receptor-resolved neuromodulation, learning rules (only hooks, §10).

## 2. The reference model (verified against `philshiu/Drosophila_brain_model@91bdd1e7`)

Fetched from `model.py` at the pinned commit; this is what "correct" means for G1.

```python
default_params = {
    't_run': 1000 * ms, 'n_run': 30,
    'v_0': -52 * mV, 'v_rst': -52 * mV, 'v_th': -45 * mV,
    't_mbr': 20 * ms,          # membrane time constant
    'tau':   5 * ms,           # synaptic time constant
    't_rfc': 2.2 * ms,         # refractory period
    't_dly': 1.8 * ms,         # synaptic delay
    'w_syn': .275 * mV,        # weight per contact
    'r_poi': 150 * Hz, 'r_poi2': 0 * Hz,
    'f_poi': 250,              # Poisson weight = w_syn * f_poi = 68.75 mV
    'eqs': """
        dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau              : volt (unless refractory)
        rfc : second""",
    'eq_th':  'v > v_th',
    'eq_rst': 'v = v_rst; w = 0; g = 0 * mV',
}
NeuronGroup(N, model=eqs, method='linear', threshold=eq_th, reset=eq_rst, refractory='rfc')
Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=t_dly)
syn.w = df_con['Excitatory x Connectivity'] * w_syn      # signed contact count x 0.275 mV
PoissonInput(target=neu[i], target_var='v', N=1, rate=r_poi, weight=w_syn * f_poi)
```

Consequences the engine must reproduce exactly (each becomes a golden test, §9):

1. **Exact ("linear") integration, not Euler.** Per step of `dt` the closed form is
   `a = exp(-dt/τm)`, `c = exp(-dt/τs)`, `b = τs/(τs-τm) · (c - a)` and
   `v' = v0 + a·(v - v0) + b·g`, `g' = c·g` — two FMAs and one multiply per neuron. With
   dt = 0.1 ms: a ≈ 0.995012, c ≈ 0.980199, b ≈ 0.00493794. Brian2 computes the same matrix
   exponential symbolically; we precompute constants in float64 and round once.
2. **`unless refractory`** freezes *both* v and g during refractoriness; the threshold is not evaluated.
   Brian2 compares in integer timesteps (`timestep(t - lastspike) >= timestep(rfc)`, with `timestep =
   floor(t/dt + 1e-3)`), so refractoriness is an integer countdown, not a float comparison. A neuron that
   spikes at step *t* is frozen at *t+1 … t+R−1* and integrates again at *t+R* (R = 22).
   **Verified in M1 (Brian2 2.9 codegen + golden traces): the flag makes *every* write to v and g
   conditional on `not_refractory`, not just the ODE update — the thresholder clears `not_refractory`
   at the spike step and only the resetter overrides the guard. Hence synaptic deliveries (`g += w`)
   and Poisson/injection events (`v += …`) arriving at steps *t … t+R−1* are silently discarded, never
   accumulated.** A port that accumulates them diverges within ~100 ms on a real circuit.
3. **Reset zeroes g** (`g = 0*mV`) — a port that only resets v diverges immediately. The stray `w = 0`
   in the reset string is to be checked against a real Brian2 run (it is either a no-op or an error in
   newer Brian2 versions); it does not affect state.
4. **Schedule order per step** (Brian2 default slots): state update → threshold → synapses (delayed
   deliveries *and* PoissonInput) → reset. Hence a delivery arriving in the same step a neuron spikes is
   wiped by the reset, and a Poisson event lands on v after the threshold check, firing next step.
5. **Delay = 18 steps** (1.8 ms), homogeneous → a ring buffer of 19 spike lists, no per-synapse delay.
6. **PoissonInput writes to `v`, not `g`, and stimulated neurons get `rfc = 0`.** `model.py`'s `poi()`
   sets `neu[i].rfc = 0*ms` for every Poisson target, so they have no refractory period at all.
   68.75 mV ≫ (v_th − v_0) = 7 mV, so every accepted event causes a spike at the next threshold check;
   the event that lands in the same step as a spike is discarded (item 2), so a continuously driven
   target fires at most every other step (5 kHz). Per stimulated neuron per step: `Bernoulli(r·dt)`
   (Brian2 draws Binomial(N=1, p=rate·dt)). *(Corrected in M1: the earlier "v accumulates during
   refractoriness" was wrong — and moot for Poisson targets, which have none.)*
6a. **Silencing (`silence()`) zeroes only the neuron's outgoing weights**: a silenced neuron still
   integrates and spikes, it just transmits nothing. The engine models this as a per-neuron
   `Transmits` flag, not as removing the neuron.
6b. **Scale of one synapse.** From the closed-form impulse response, a jump `g0` in g raises v by at most
   ≈ 0.1575·g0 (peak ≈ 9 ms later), so one presynaptic spike needs ≈ 162 contacts (7 / (0.275 × 0.1575))
   to fire a resting neuron on its own; typical firing is convergence of many inputs. Useful when
   designing stimuli and reading rate readouts.
7. **Sign policy** lives in the data, not the equations: `Excitatory x Connectivity` is
   `sign(NT) × contact count` with ACh = +1, GABA/Glu = −1, DA/OA/5-HT = +1. MaleCNS has no such
   column; the checkpoint builder applies the policy and records it (Xenova's MaleCNS config adds
   `histamine: 0, unknown: 0`, which we adopt as the default MaleCNS policy and expose as a switch).
8. **Float precision**: Brian2 is float64. The reference backend is float64; the fast path is float32.
   Spiking networks are chaotic, so only the float64 reference is expected to match Brian2 to rounding;
   float32 is validated statistically (§9). *Measured in M1:* Brian2 arranges the same closed form as
   `g·τs·(e^{dt/τs} − e^{dt/τm})·e^{−dt/τm}·e^{−dt/τs}/(τm − τs) + v·a + v0 − v0·a`, which differs from our
   `v0 + a(v − v0) + b·g` by a few ulps; the observed max deviation is ~1e-13 mV over 3,000 steps and
   every spike step is identical, so the golden tolerance is 1e-11 mV.
9. **Edge order** is irrelevant for spike identity but not for the last ulp: Brian2 sums deliveries in
   synapse-creation order. The golden generator sorts edges by (pre, post) — the checkpoint's CSR order —
   so the two sides sum in the same order. Builders must keep rows sorted and duplicate-free.

Model variants to support from day one via a `NeuronModel` descriptor (parameters + integration
constants), not via code forks: Shiu-v630 (paper), Shiu-v783, MaleCNS-Xenova (same equations, MaleCNS
sign policy), and "user" (any parameter override, e.g. Fly64's tonic drive as an added stimulus).

### 2.1 Why spiking LIF is the v1 model (and what "variations" mean)

Decision: v1 simulates **spiking point neurons** (Shiu LIF). Reasons:

- It is the only connectome-wide model with a published experimental validation, and it is what every
  MaleCNS demo (Xenova, DOOMFLY, Fly64, Eon) runs — results stay comparable.
- It is the cheap one. Spiking is *event-driven*: a step touches ~2 MB of state plus only the out-edges
  of neurons that spiked. A "smooth"/graded model (Flyvis-style rate units, or any continuous-activity
  model) must propagate through **every** edge every step — a 25.6 M-edge sparse matrix-vector product
  (~150 MB of memory traffic) per 0.1 ms. That is not a real-time CPU workload; it is a GPU-and-coarser-dt
  regime.
- The commonly discussed variations are mostly still point-neuron spiking models and share the same
  kernel shape (update state → threshold → scatter): current-based vs conductance-based synapses,
  adaptive threshold / AdEx, Izhikevich, per-type parameter tables, plasticity on a declared edge subset.
  All of these fit the `NeuronModel` descriptor + extra state arrays without changing the architecture.
- Graded/rate models need a different kernel (dense SpMV per step). They are planned as a later,
  separate backend mode (§10), mainly for optic-lobe hybrids (Flyvis-like units feeding spiking DNs), not
  for v1.

## 3. Data and the checkpoint format

### 3.1 Sources (all bulk downloads, CC BY 4.0, no tokens)

| Source | Files | Use |
|---|---|---|
| MaleCNS v1.0 | `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (1.05 GB; `traced-only` 508 MB), `body-annotations-…-minconf-0.5.feather` (14.5 MB), `body-neurotransmitters-…v1.0.feather` (43.3 MB) | demos; 166,700-neuron graph |
| FlyWire v630 / v783 | Shiu repo inputs (v630 = paper), Zenodo `proofread_connections_783.feather` (852 MB) | G1 reproduction |
| Shiu experiment outputs | Edmond archive `10.17617/3.CZODIW` | statistical validation |

Feather = Arrow IPC. Use the `Apache.Arrow` NuGet package (pure managed) to read; Parquet via
`Parquet.Net` if any source ships Parquet (Eon's `2025_Connectivity_783.parquet`). No pandas.

### 3.2 `.dfb` — dotFly brain checkpoint (memory-mapped, like GGUF in dotLLM)

Single file, little-endian, 64-byte-aligned sections, header with section table. Loaded with
`MemoryMappedFile`; nothing copied to the managed heap.

```
header        magic "DFB1", version, section table, provenance block (JSON: source file SHA-256s,
              dataset+materialization, inclusion filter, NT sign policy, model params, builder commit,
              build time, seed for any shuffle)
neurons       N x { bodyId u64 (exact — never through double), typeId u32, classId u16, ntId u8,
              hemisphere u8, soma xyz f32x3 (nullable), flags }
strings       type/class/NT/superclass string tables (+ per-neuron free-form annotation offsets)
model         NeuronModel descriptor (§2), dt, precomputed a/b/c in f64
csr_out       rowPtr i64[N+1], col i32[E] (sorted ascending within row), w i16[E] (signed contact
              count; i32 fallback flag if any |count| > 32767)
csc_in        optional mirror (post → pre) for inspection/ablation; built by `dotfly build --with-csc`
sets          optional named neuron sets (e.g. "sugar_GRN_R", "MN9", "DNa02_L") — literature-guided
              candidate lists shipped with demos, versioned in the file
```

Sizes for MaleCNS: `col` 102 MB + `w` 51 MB + rowPtr 1.3 MB ≈ 155 MB (CSR), ×2 with CSC. The whole
graph sits in RAM; the *state* (v, g, refractory) is ~2 MB and lives in L2 — this is the core reason
the engine can be fast (§5).

### 3.3 Sub-checkpoints and controls (`dotfly build`)

The same builder produces:
- `--filter` by class/superclass/type/region/hemisphere/status (e.g. central brain only, optic lobe
  only, "traced-only" status filter to reproduce the official export — the note warns the retained demo
  graph (25,582,938 edges) ≠ `traced-only` (25,563,197 edges); both must be buildable and named).
- `--hops k --seed-set name` — k-hop neighbourhood of a set (sugar GRNs → MN9 circuit; a few-thousand
  neuron checkpoint for tests, CI and low-end demos).
- `--min-weight n` — drop weak edges (an explicit, recorded degradation knob).
- `--shuffle degree-preserving|weight|sign --seed s` — control graphs for causal experiments (§7.6).
- `--ablate set` — zero in/out edges of a set (Shiu-style silencing) baked into a checkpoint, in
  addition to runtime masks.

Every derived file carries its parent's provenance chain.

## 4. Architecture

Layered like dotLLM; each project is a NuGet package.

```
+--------------------------------------------------------------+
| DotFly.Cli            build / inspect / run / bench / export  |  Spectre.Console, Native AOT
+--------------------------------------------------------------+
| DotFly.Engine         Simulation, clock, stimuli, probes,     |  the demo-facing API (§7)
|                       recorder/replay, snapshots, queries     |
+--------------+--------------+--------------------------------+
| DotFly.Data  | DotFly.Cpu   | DotFly.Cuda                    |  Arrow/Parquet → .dfb builder;
| (builder,    | (SIMD, pool) | (PTX via cuda driver API)      |  backends implement IBackend
|  loaders)    |              |                                |
+--------------+--------------+--------------------------------+
| DotFly.Core   .dfb reader, graph/neuron tables, NeuronModel,  |  no dependencies beyond BCL
|               IBackend/ISimulationState, RNG, unmanaged mem   |
+--------------------------------------------------------------+
src/       DotFly.Adapters.Onnx, DotFly.Adapters.MLNet — optional readout-adapter packages (§7.9);
           the engine itself has no ML dependency
samples/   DotFly.Sample.Console (sugar→MN9), DotFly.Sample.Explorer (TUI raster/inspector),
           DotFly.Sample.Godot (embodied readout demo), DotFly.Sample.TrainReadout (record frames,
           train a LinearAdapter / ML.NET model, run it back as an adapter)
tests/     DotFly.Tests.Unit (golden traces), DotFly.Tests.Integration (real checkpoints, opt-in)
benchmarks/DotFly.Benchmarks (BenchmarkDotNet)
tools/     brian2_golden.py — the only Python in the repo, used offline to produce test fixtures
```

Conventions copied from dotLLM's `CLAUDE.md` (file-scoped namespaces, `readonly record struct`,
`Span<T>` signatures, `[MethodImpl(AggressiveInlining)]`, `[SkipLocalsInit]`, `NativeMemory.AlignedAlloc`,
`[LibraryImport]`, XML docs, TreatWarningsAsErrors, central package management, MinVer). Target
`net11.0` (preview SDK pinned in `global.json`, `rollForward: latestFeature`).

## 5. Core engine design

### 5.1 State layout (SoA, unmanaged, 64-byte aligned, N padded to the vector width)

```
v[N]      float32 (float64 in the reference backend)  membrane potential
g[N]      float32                                     synaptic variable
rfc[N]    uint8 / int16                               refractory countdown in steps (0 = free)
spikeRing[19][cap]  int32 spike lists (delay ring) + counts
stim      sparse list of stimulated neurons + p = rate·dt
silence[N] bit                                        runtime ablation mask (§7.6)
```
~2 MB for MaleCNS. Everything is `IDisposable`, allocated once per `Simulation`.

### 5.2 One step (dt = 0.1 ms); thread `p` owns neuron range `[lo_p, hi_p)` for *all* phases

```
phase A (parallel, own range, SIMD):
    for i in range:
        if rfc[i] > 0: rfc[i]--; continue
        v[i] = v0 + a*(v[i]-v0) + b*g[i];  g[i] *= c
        if v[i] > vth: localSpikes.add(i)            (compress via MoveMask + LUT)
barrier
phase B (parallel, own range):
    publish localSpikes into spikeRing[(t+18)%19]      (each thread writes its own sub-slot)
    due = spikeRing[t%19]                              (written 18 steps ago, complete, read-only)
    for each spiking pre s in due (all threads iterate the same list in the same order):
        for e in blockCSR(s, p):  g[col[e]] += wsyn * w[e]     <- only cols inside my range
    Poisson: for stimulated i in my range: if philox(seed, t, i) < p: v[i] += 68.75 mV
    Reset for my range's spikes of *this* step: v = vrst, g = 0, rfc = 22
barrier
```

Two barriers per step. Order inside phase B (deliver → Poisson → reset) reproduces Brian2's slot order (§2.4).

### 5.3 Blocked CSR: race-free, deterministic scatter without atomics

*CSR (Compressed Sparse Row)* is the standard sparse-graph layout used in §3.2: `rowPtr[N+1]`,
`col[E]`, `w[E]`; the out-edges of neuron `s` are the contiguous slice `col[rowPtr[s] .. rowPtr[s+1])`,
so a spike costs one sequential read of that slice instead of a scan of a 166,700² matrix.

Columns are sorted within each row (guaranteed by the builder). At load, for a partition of the post
range into `P` blocks (P = thread count, or a multiple for load balance), compute
`blockPtr[row][P+1]` = offsets where each row's columns cross block boundaries: one linear pass over E
(~25 M ints, ~50 ms), 166,700 × (P+1) × 4 B ≈ 11 MB at P = 16. Thread `p` then walks
`col[blockPtr[s][p] .. blockPtr[s][p+1])` for each spiking `s` — no binary search, no atomics, writes
confined to its own L1-resident slice of `g`, and **results are independent of the thread count**
because each `g[i]` receives its additions in the (fixed) spike-list order.

Trade-off: threads whose block has few edges idle while others work; mitigated by P = 4× threads
with dynamic block claiming (still deterministic: the order of additions to any one `g[i]` is unchanged).

### 5.4 Spike list construction

Vectorised threshold: `Vector512<float>.GreaterThan` → `ExtractMostSignificantBits` → popcount +
per-nibble index LUT (`Vector128.Shuffle`) to compress indices; scalar fallback. Spike lists are
per-thread fixed-capacity arrays (capacity = range length: worst case is every neuron spiking).

### 5.5 Threading

A dedicated `ComputeThreadPool` (dotLLM pattern): spinning workers pinned to cores, a sense-reversing
barrier, no `Task`/TPL per step (10,000 steps per second cannot afford ~10 µs of scheduling per phase).
Single-threaded fast path when N is small (sub-checkpoints) or activity is tiny: adaptively skip the
parallel scatter when `Σ deg(spiking) < threshold` (measured, not guessed).

### 5.6 Randomness

Counter-based RNG (Philox4x32-10) keyed by `(seed, step, neuronIndex, stream)`. The same seed gives
the identical Poisson event stream on CPU, GPU, any thread count, and for replay — no per-thread
generator state, no allocation. Brian2's own RNG is not reproduced; golden tests inject explicit spike
trains instead (§9).

### 5.7 Precision & generic math

Kernels are generic over `TFloat : IBinaryFloatingPointIeee754<TFloat>` so `double` (reference) and
`float` (fast) share one implementation; the JIT specialises. `Half`/bf16 state is deliberately *not*
planned — 7 mV of headroom over 0.275 mV steps is where half precision fails.

### 5.8 Batched trials ("lanes")

Shiu runs 30 trials per condition. Layout `[N][B]` with B = 8/16 lanes puts the trial dimension in
SIMD lanes: the state update is unchanged, and a delivery for spiking row `s` becomes
`g[col][0..B) += w · spikeMaskLanes(s)` — a contiguous FMA instead of a scalar scatter, amortising the
CSR traversal over trials. Big win for experiments; irrelevant for a single real-time demo. Lanes are
a `Simulation` option, not a separate engine.

## 6. Backends

### 6.1 CPU (`DotFly.Cpu`) — first

- `Vector512`/`Vector256`/`Vector128` with width-agnostic code paths, scalar fallback.
- Weight arrays as `i16` counts × `wsyn` at delivery (6 B/edge) vs prescaled `f32` (8 B/edge):
  benchmark; the scatter is bound by random writes into `g`, not by reading weights, so the smaller one
  probably wins.
- NUMA: keep the graph and state on the pool's node (dotLLM `NumaTopology`); single-socket desktops
  are the primary target.
- Native AOT for the CLI (startup < 100 ms matters for a tool that runs in a game's launch path).

### 6.2 CUDA (`DotFly.Cuda`) — second

Same approach as dotLLM: PTX text embedded as resources, loaded with `cuModuleLoadData` through a
`[LibraryImport]` binding of the CUDA driver API (`nvcuda.dll` / `libcuda.so`), no custom native
library, no toolkit at runtime. `nvcc -ptx` at development time, PTX checked in.

Kernels:
- `lif_step`: one thread per neuron, update + threshold, warp-ballot compaction of spike indices into a
  device list.
- `deliver`: one warp per spiking row (rows > 4096 edges split across blocks), `atomicAdd` on `g`
  (non-deterministic order → float32 non-determinism accepted on GPU; a deterministic post-blocked
  variant is optional).
- `poisson_inject`: Philox, identical event stream to CPU (§5.6).
- `reset`, `record_rates` (per-neuron spike counters + EMA rates on device).
- Multi-step launch: a CUDA graph of K steps (K = 10–100) per host round-trip; spike lists and rates
  copied back once per K steps into a pinned host buffer. Persistent cooperative kernel evaluated later.
- Lanes (§5.8) map to `blockDim.x` naturally.

### 6.3 Later / maybe

ROCm/HIP (would need a native lib; the PTX-free trick is CUDA-only). WebGPU/browser viewer is out of
scope (Xenova already covers it).

## 7. Engine API (`DotFly.Engine`) — what demos program against

dotFly is a **library** (NuGet packages); Godot, console apps, tests and benchmarks all link it
in-process. There is no server, no bridge, no sidecar process. Two usage shapes share one engine:

**Shape 1 — the script drives the clock** (experiments, tests, console demos; this is
`samples/DotFly.Sample.SugarExperiment`):

```csharp
using DotFly;

using Brain brain = Brain.Open(".data/shiu2024/flywire-v630.dfb");        // mmapped, immutable, shareable
using Simulation sim = brain.CreateSimulation(new SimulationOptions { Seed = 1000, Threads = 8 });

NeuronSet sugar = brain.ByBodyIds("sugar_GRN_R", sugarIds);                  // exact 64-bit IDs
NeuronSet mn9   = brain.ByBodyId(720575940660219265);
sim.Input(sugar, InputKind.PoissonToV).Fill(150f);                           // 150 Hz each (Shiu semantics: rfc = 0)
sim.Silence(brain.Query(type: "DNa02", side: Side.Left));                    // outgoing synapses zeroed
OutputPort mn9Rate = sim.Output(mn9, OutputKind.Rate(50.Ms()), publishEvery: 20.Ms());
sim.OnSpikes += (step, ReadOnlySpan<int> ids) => raster.Push(step, ids);     // zero-alloc callback
using Recorder rec = sim.Record("run1.dfs");                                 // spikes + inputs + output frames

for (int trial = 0; trial < 30; trial++)
{
    sim.Reset(1000 + (ulong)trial);                                          // fresh trial, new Poisson stream
    sim.Run(1.Seconds());                                                    // neural time, blocking
}

Console.WriteLine($"MN9 {mn9Rate.Snapshot.Mean():F1} Hz, RTF {sim.Clock.RealTimeFactor:F1}×");
```

**Shape 2 — the app has its own loop** (Godot, any game/UI): the simulation runs on its own thread via
`RealtimeDriver`; the app writes input ports and reads output snapshots, never blocking the
simulation and never touching engine state from the game thread.

```csharp
public partial class FlyBrain : Node
{
    Brain _brain; Simulation _sim; RealtimeDriver _driver;
    InputPort _eyes; OutputPort _dn;                 // K_in floats in, K_out floats out
    IReadoutAdapter _decoder;                        // fixed / calibrated / trained — the demo's choice
    float[] _rates, _actions;

    public override void _Ready()
    {
        _brain = Brain.Open(ProjectSettings.GlobalizePath("res://brains/malecns-v1.0-superclass.dfb"));
        _sim   = _brain.CreateSimulation(new SimulationOptions { Threads = 8, Seed = 1 });

        // inputs: one Poisson rate per LC4 neuron (the retina → rate mapping is the demo's encoder)
        _eyes  = _sim.Input(_brain.Query(type: "LC4"), InputKind.PoissonToV);
        _rates = new float[_eyes.Count];

        // outputs: per-neuron rates of descending neurons, published every 20 ms of neural time
        NeuronSet dn = _brain.Query(type: "DNa02") | _brain.Query(type: "DNp09");
        _dn = _sim.Output(dn, OutputKind.Rate(30.Ms()), publishEvery: 20.Ms());

        // decoder: features (K_out rates) → actions (walk, yaw). Linear here; an ONNX/ML.NET
        // adapter consumes the same span (M4)
        _decoder = LinearAdapter.Load("res://decoders/dn_linear.json");
        _actions = new float[_decoder.ActionCount];

        _driver = new RealtimeDriver(_sim, new RealtimeOptions { Ratio = 1.0, MaxStepsPerTick = 500 });
        _driver.Start();                      // dedicated thread; keeps neural time ≤ wall time
    }

    public override void _Process(double delta)
    {
        _retina.Encode(_rates);               // app code: pixels → K_in Hz values
        _eyes.Write(_rates);                  // thread-safe; applied from the next step

        OutputFrame f = _dn.Snapshot;         // lock-free, last published frame
        _decoder.Evaluate(f.Values, _actions);
        _body.Apply(walk: _actions[0], yaw: _actions[1]);
        _hud.Text = $"t={f.NeuralTime.TotalSeconds:F2}s behind={_driver.BehindBy.TotalMilliseconds:F0}ms " +
                    $"RTF={_sim.Clock.RecentRealTimeFactor:F2} spikes/s={_sim.Clock.RecentSpikesPerSecond:N0}";
    }

    public override void _ExitTree() { _driver.Dispose(); _sim.Dispose(); _brain.Dispose(); }
}
```

Toggles for a demo UI map to engine calls: `_sim.RecurrentTransmission = false`,
`_sim.ExternalInput = false`, `_sim.Silence(set)` / `Unsilence(set)`, `_sim.Reset(seed)`, or
swapping the checkpoint for a `--shuffle degree-preserving` build.

### 7.1 Objects

- `Brain` — immutable, mmapped checkpoint: `Neurons` (index ↔ bodyId, type, class, NT, hemisphere, soma),
  `OutEdges(i)`, `InEdges(i)` (needs CSC), `Sets`, `Query(...)`, `Downstream(set, hops, minWeight)`,
  `Provenance`.
- `Simulation` — mutable state + backend + stimuli + probes; `Step()`, `Run(TimeSpan simulated)`,
  `Advance(steps)`, `Reset(seed)`, `Snapshot()` / `Restore()` (state save/restore for branching
  experiments), `Clock`.
- `InputPort` (§7.8): bound to a `NeuronSet`, one float per neuron; kinds `PoissonToV` (Shiu semantics),
  `PoissonToG` (gentler), `Current` (constant/tonic drive per step, Fly64-style), `SpikeTimes` (explicit
  events for replay/golden tests).
- `OutputPort` (§7.8): bound to a `NeuronSet`; kinds `Rate(window)`, `SpikeCount`, `Membrane` (v, g;
  small sets only), `Population` aggregates; published at a fixed neural-time interval into a
  double-buffered snapshot so a game thread never blocks the simulation thread.
- `Recorder` / `Replay` — compact `.dfs` binary (step, id) with sidecar JSON of stimuli and probe series;
  exporters to CSV/Parquet/NumPy `.npy` for comparisons with Brian2/Eon outputs.

### 7.2 Time

`SimulationClock` separates **neural time** (steps × dt), **simulated duration** and **wall time**, and
reports the real-time factor. `RealtimeDriver` runs the simulation on its own thread and advances it to
keep neural time ≤ wall time (or a fixed ratio), exposing "steps completed this frame" and "behind by"
so a demo can degrade honestly (switch to a smaller checkpoint, never silently slow neural time).

### 7.3 Game-loop integration (Godot)

- Simulation thread + `RealtimeDriver`; the game's `_Process` reads the latest `ReadoutSnapshot` and
  enqueues encoder frames/stimuli with neural timestamps (lock-free SPSC queues).
- Decoders are the demo's code, in the demo project, explicitly labelled *fixed*, *calibrated* or
  *trained* (the Fly Marksman vs Fly Dino distinction from the note). The engine ships none by default;
  the Godot sample ships a Xenova-style DN → speed/yaw/escape mapping with its gains in a JSON the user
  can read.
- No out-of-process bridge (decided 2026-09-18): the engine is consumed as a library in-process. If a
  Godot release cannot host the .NET 11 preview runtime, the mitigation is multi-targeting
  `net10.0;net11.0` for the library packages, not a sidecar process.

### 7.4 Inspection

CLI `dotfly inspect`: neuron by bodyId/type, in/out partners with weights and NT, degree histograms,
path search, set membership, provenance dump. Library-level equivalents on `Brain`.

### 7.5 Replay & determinism guarantee

Given (checkpoint SHA, options, seed, recorded stimuli), a run reproduces exactly on the same backend
and precision, independent of thread count (§5.3, §5.6). Cross-backend (CPU float32 vs CUDA) is
statistically comparable only.

### 7.6 Causal controls as first-class features (from the note's "minimal causal controls")

`sim.RecurrentTransmission = false` (deliver nothing), `sim.SensoryInput = false`,
`sim.Silence(set)`, shuffled/degree-preserving control checkpoints (§3.3), fixed decoders across
conditions, multi-seed runs via lanes. Demos are expected to expose these toggles in their UI.

### 7.7 Addressing neurons: `NeuronSet`

Every input, output, silence mask and query resolves to a `NeuronSet`: an immutable, sorted `int[]` of
neuron indices plus a name and provenance (how it was made). Four ways to get one:

| Way | Example | Notes |
|---|---|---|
| exact IDs | `brain.ByBodyId(10001UL)`, `brain.ByBodyIds(span)` | u64 preserved exactly; unknown IDs are errors, not silent drops |
| annotation query | `brain.Query(type: "LC4", side: Side.Left)`, `brain.Query(cls: "descending", nt: Nt.Gaba)`, `brain.Query(typeGlob: "DNa*")` | matches the string tables in the checkpoint; the column is explicit because MaleCNS and FlyWire name things differently (`type`, `flywireType`, `hemibrainType`, `class`, `superclass`, `rootSide`, …) |
| shipped named sets | `brain.Sets["sugar_GRN_R"]`, `brain.Sets["DNa02_L"]` | literature-guided lists demos rely on, versioned in the file with their source (paper/table) |
| set algebra | `a \| b`, `a & b`, `a - b`, `brain.Downstream(set, hops: 2, minWeight: 5)`, `brain.Upstream(...)` | results can be saved back into a checkpoint (`brain.Sets.Add(name, set)` in the builder) |

Sets are what you name in a config file too: a demo's `decoder.json` lists `"features": ["DNa02_L",
"DNa02_R", "DNp09"]` and the engine resolves it against whichever checkpoint is loaded, failing loudly
if a set is missing.

### 7.8 Inputs and outputs are vectors

An `InputPort` over K neurons accepts K floats (`Write(ReadOnlySpan<float>)`, `Fill(value)`); values
take effect from the next step and stay until overwritten. Meaning per kind: `PoissonToV` → rate in Hz
converted to a per-step Bernoulli probability; `Current` → mV added per step; `SpikeTimes` → seconds.
Converting camera pixels, sound, or game state into those K numbers is the **encoder** and lives in the
app; the engine only promises "K floats in, applied at step t+1, recorded".

An `OutputPort` over K neurons publishes K floats (`Snapshot.Values` as `ReadOnlySpan<float>`, plus
`NeuralTime`, `Step`, `FrameIndex`) at a fixed neural-time interval (`publishEvery`, e.g. 20 ms, the
control tick used by Fly64/Eon). Views for interop: `Memory<float>`, `System.Numerics.Tensors.Tensor<float>`,
and a pinned buffer address for zero-copy hand-off to native runtimes. Publication is double-buffered
and lock-free; the game thread reads the last complete frame.

### 7.9 Trainable readouts: ML.NET, ONNX, TorchSharp

The output vector *is* the feature vector, so a decoder is any function `features → actions`:

- `IReadoutAdapter { int FeatureCount; int ActionCount; void Evaluate(ReadOnlySpan<float> features, Span<float> actions); }`
  in `DotFly.Engine` with no ML dependency. Built-ins: `ThresholdAdapter` (Fly Marksman-style hand rules),
  `LinearAdapter` (Fly Dino's 243-parameter reader is exactly this; trainable in-process with plain
  regression / REINFORCE in pure C#), `CompositeAdapter` (chain, smoothing, hysteresis).
- `DotFly.Adapters.Onnx` — `OnnxAdapter` over Microsoft.ML.OnnxRuntime; input `OrtValue` created over
  the port's pinned buffer (no copy), output written into `actions`. Load anything trained in PyTorch,
  ML.NET or scikit-learn exported to ONNX.
- `DotFly.Adapters.MLNet` — `MLNetAdapter` wrapping a `PredictionEngine<TIn,TOut>` for models trained
  with ML.NET (`MLContext`) on recorded frames. TorchSharp needs no package: it consumes the same span.
- Training data: `Record.Frames` writes control frames `(frameIndex, neuralTime, inputs[], features[],
  actions[], reward, done)` to `.dfs`; `dotfly export --frames run.dfs --parquet|--npy|--csv` produces
  the tables for offline training in any stack. `Simulation.Snapshot()/Restore()` gives episode resets;
  **lanes** (§5.8) run B episodes in one simulation for in-loop RL with one graph traversal.
- Provenance: the run header records which adapter ran and whether it was *fixed*, *calibrated* or
  *trained* (and on what), so a demo cannot accidentally present a trained readout as "the wiring".

## 8. Performance plan

### 8.1 Budget (estimates, to be measured in M2)

Per step, MaleCNS, float32, 16 threads, AVX-512:
- Phase A: 166,700 neurons × ~6 ops, 2 MB streamed from L2 → ~10–20 µs single-threaded, ~2–4 µs split.
- Phase B: cost ∝ Σ out-degree of spiking neurons (mean degree 153). At sugar-experiment activity
  (order 10¹ spikes/step) it is negligible; at heavy visual stimulation (10⁴ spikes/step ≈ 1.5 M edge
  updates) ~1–2 ms single-threaded, ~150–250 µs across 16 threads.
- Barriers: 2 × ~1 µs.

*Measured in M2 (see §11 M2 findings and §12.12):* phase A ≈ 0.8 cycles/neuron after the fixed-point
rewrite (47 µs single-threaded for MaleCNS, memory-bound; ~6 µs per thread on 8 cores); one barrier per
18-step batch; delivery ≈ 10 ns per edge effective.

So real time (100 µs/step) is plausible on CPU for moderate activity and needs the GPU for heavy
activity or lanes. Reference point from the note: Brian2 ≈ 5 min per simulated second per CPU thread
for the sugar experiment (≈ 30 ms/step) — the CPU target is > 300× that.

### 8.2 Metrics (reported by `dotfly bench` and BenchmarkDotNet)

steps/s; real-time factor; active-edge updates/s (Gedge/s); spikes/s; peak RSS; per-phase µs; and
always alongside: checkpoint SHA, precision, threads, seed, simulated duration, wall duration, mean
activity. Scenarios: `idle` (no input), `sugar` (Shiu 100/150 Hz), `visual-heavy` (LC-family
stimulated at high rate), `lanes-16`.

### 8.3 Optimisation order

1. Correct scalar float64 → 2. SIMD phase A → 3. blocked-CSR parallel phase B → 4. weight dtype /
row sorting / prefetch of the next spiking row's block → 5. adaptive single-vs-parallel B → 6. lanes →
7. CUDA → 8. CUDA graphs / persistent kernel. Each step lands with a benchmark diff.

## 9. Validation strategy

1. **Unit golden traces (synthetic)**: 1–3 neurons with hand-derived expected v/g sequences for the
   exact integrator, refractory countdown, reset-zeroes-g, same-step delivery-then-reset, delayed
   delivery at exactly 18 steps, Poisson-during-refractory deferred spike.
2. **Brian2 golden traces**: `tools/brian2_golden.py` (Brian2 2.5.1, pinned as in the Shiu env) builds
   sub-checkpoints of a few hundred neurons from v630, replaces `PoissonInput` by a `SpikeGeneratorGroup`
   with explicit event times (so RNG is out of the loop), runs 200 ms, dumps v/g per step and spike
   times. The float64 CPU reference must match to ≤ 1e-12 mV and identical spike steps. Fixtures are
   committed; the script is not part of the build.
3. **Sugar → MN9 reproduction** (integration, needs v630 data): 30 trials at 150 Hz, compare MN9 and
   published downstream rates against the Edmond archive using Eon's metrics (active-set Jaccard,
   per-neuron rate correlation, spike-count ratio) with thresholds fixed *before* running.
4. **Cross-backend/precision**: float32 vs float64, CPU vs CUDA, threads 1 vs 16 (must be identical
   on CPU), lanes vs sequential trials (identical event streams ⇒ identical spikes).
5. **Determinism test**: same seed twice ⇒ byte-identical `.dfs`.
6. **Data tests**: MaleCNS builder reproduces the note's counts (166,700 / 25,582,938 / 124,177,617 and
   the `traced-only` 25,563,197 / 124,025,046); bodyId round-trips exactly; sign policy table applied.

## 10. Extension hooks (designed in, implemented later)

- `ISynapsePlasticity` invoked on a *declared subset* of edges (e.g. KC→MBON, DOOMFLY-style) with
  its own weight buffer; the base graph stays immutable so "plasticity on/off" is a flag. Any learning
  claim is the demo's, with the controls from the note (contingent vs yoked, retention).
- `INeuronModel` variants: current-based vs conductance-like, per-type parameter tables, Flyvis-style
  graded units for optic-lobe types (mixed-model networks) — the state layout supports extra arrays.
- Heterogeneous delays (per-edge delay classes) if a future dataset provides them.
- Multi-GPU: not planned; N is small, the problem is latency, not capacity.

## 11. Milestones

| # | Milestone | Deliverable | Acceptance |
|---|---|---|---|
| M0 ✅ 2026-09-18 | Scaffold | solution, props, CI, CLAUDE.md, README with scope/caveats; `NeuronModel` + exact integrator constants, `SignPolicy`, `NativeBuffer<T>`, `dotfly info` | `dotnet build` on .NET 11 RC1 (`11.0.100-rc.1`), 25 unit tests green |
| M1 ✅ 2026-09-18 | Core + data | `.dfb` format (writer + mmapped reader), `MaleCnsSource` (Arrow/LZ4) + `FlyWireSource` (Parquet) builders, `Brain`/`NeuronSet`/query/downstream, Philox RNG, float64 `ReferenceBackend`, `dotfly build/inspect/run`, `tools/brian2_golden.py` | Brian2 golden fixtures pass (chain3; sugar 2-hop 130 n / 3.3 k e / 1 k steps; sugar 3-hop 2,001 n / 160 k e / 3 k steps — all spikes identical, traces ≤ 1e-11 mV); data tests reproduce the note's counts; 53 unit + 4 integration tests green |
| M2 ✅ 2026-09-18 | Fast CPU | `CpuBackend`: float32 SIMD phase A (`LifKernel`), blocked-CSR delivery, `ComputeThreadPool` (pinned, one per physical core), Philox + geometric Poisson, per-batch barrier, adaptive contiguous block ownership, `dotfly bench` | *Measured on a Ryzen 7 5800HS laptop (8 cores, AVX2), 8 threads:* v630 sugar **19.6×** real time (196 k steps/s; 3.3× on one thread), MaleCNS idle **14.5×**, MaleCNS with 1,684 T4a neurons driven at 150 Hz **6.2×** (~0.9 G synaptic deliveries/s). Spikes bit-identical for 1/2/3/7/16 threads; float32 backend reproduces the float64 reference's per-neuron spike counts exactly on the full v630 sugar run and ≥ 99 % of Brian2 golden events; 76 unit + 5 integration tests green |
| M3 ✅ 2026-09-18 | Engine API | `Simulation` (`CreateSimulation`, `Run/Advance`, `Reset(seed)`, `Snapshot/Restore`, `ScheduleAt`, `Inject`, `Silence`), `InputPort` (PoissonToV, Current), `OutputPort` (Rate/SpikeCount/Membrane, lock-free frames), `SimulationClock`, `RealtimeDriver`, `.dfs` `Recorder`/`Recording` (+ `Replay`, CSV export), `IReadoutAdapter` (Linear/Threshold/Smoothing, JSON) | `samples/DotFly.Sample.SugarExperiment` runs the Shiu example (30 trials + 3 silencing runs) in ~2 s per 30 trials: MN9 83 Hz, rate correlation 0.995 / Jaccard 0.95 with the published per-neuron rates; Brian2 2.5.1 and 2.9 on the same inputs agree with dotFly, not with the published file (see findings). 87 unit + 5 integration tests |
| M4 ✅ 2026-09-18 | Demos + adapters | `dotfly explore` (live raster, stats, causal toggles, `+/-` rate); `samples/DotFly.Sample.Godot` (LC4 retina encoder → escape descending neurons DNp04/DNp01 → fixed decoder → 2D fly, R/E/S toggles); `DotFly.Adapters.Onnx` (`OnnxAdapter`, zero-copy feature buffer, tolerant of extra graph inputs) and `DotFly.Adapters.MLNet` (`MLNetAdapter`, runtime-sized feature schema); `samples/DotFly.Sample.TrainReadout` (record frames → Linear / ML.NET / ONNX readouts) | Godot runs the full 166,700-neuron MaleCNS at **RTF 1.00, 0 ms behind, ~25 % busy** on the 8-core laptop (headless smoke test, 883 k spikes/s); TrainReadout: held-out accuracy Linear 95.8 %, ML.NET 98.3 %, ONNX 98.3 % (3 classes, chance 33 %); 93 unit + 5 integration tests |
| M4.5 ✅ 2026-09-20 | 3D room demo + presentation | `samples/DotFly.Sample.Godot3D`: the fly's own eyes (two 32×32 luminance cameras), odor cones with wind, sugar on tables and floor, measured readouts LC4→DNp04/GF, T4→HS/H2/VS, ORN_DM1→DM1_lPN, JO→AMMC012, GRN→MN9, a labelled decoder (surge/tumble/cast, landing, feeding), real-spike brain map, traces wired to decisions, isometric room map, rigged fly model with clips, chase/follow cameras, occlusion fading, `--flies N`; library additions it needed (`Brain.Upstream/SomaPositions`, `SynapticGain`, thread pin offsets, `--stim-seconds`, `--upstream-of`); the DocFX documentation site (`docs/`: guides, measured-vs-engineered, findings, CLI, API reference) | **This is the initial prototype.** RTF 1.00 with all panels; 94 tests green; docs build |

**M0–M4.5 are the initial prototype.** What follows are *features*, tracked as GitHub issues
(see `docs/roadmap.md` for the full text):

| # | Feature | Summary |
|---|---|---|
| F1 | Move the sensory boundary outward (vision) | test whether the uniform-LIF optic lobe computes motion/looming from photoreceptor input; if so, a photoreceptor-level retina encoder; landing by visual expansion; LC10/LC11 object-approach readouts |
| F2 | Connectome-native navigation (§12 finding p) | EPG ring order and bump tests; per-cell-type parameters (`.dfb` type-gain tables, `SimulationOptions.TypeGains`); subgraph extractor; CMA-ES / rate surrogate against compass, hΔC/PFN, PFL3, DNa02 targets; validation on surge/cast with the decoder off |
| F3 | CUDA backend | PTX kernels, driver-API binding, CUDA graphs, lanes on GPU; ≥ 10× real time on one GPU |
| F4 | Lanes + AOT + packages | batched trials on CPU, Native AOT CLI, NuGet packages, published API docs |

F1 and F2 are the "run the real fly" features and come before the performance work; F3 can proceed in parallel.

## 12. Decisions to confirm and open risks

1. **Name/licence** — *decided*: "dotFly", GPLv3 like dotLLM. Data is CC BY 4.0; attribution files
   ship with the checkpoints; Shiu code (MIT) is only *read*, not vendored.
2. **Delivery** — *decided*: a library consumed in-process (Godot or anything else); no bridge/sidecar.
   Risk: Godot 4.x .NET hosting may lag behind a preview runtime; mitigation is multi-targeting
   `net10.0;net11.0` for `Core/Engine/Cpu` only if it turns out to be needed — not by default.
3. **Datasets** — *decided*: FlyWire (v630 for the paper reproduction, v783) **and** MaleCNS v1.0; one
   format, one engine; validation runs on v630 first, demos default to MaleCNS.
4. **Simulation form** — *decided*: spiking LIF (Shiu) for v1; graded/rate models later as a separate
   backend mode (§2.1, §10).
5. **`Apache.Arrow` Feather support**: Janelia files are Arrow IPC (Feather v2); if any turn out to be
   Feather v1 or use a compression the managed reader lacks, the fallback is a one-off Python conversion
   to Parquet, documented, never required at runtime.
6. **Blocked-CSR memory** at high P (P = 64 ⇒ ~43 MB) is fine; verify the load-balance assumption
   with the `visual-heavy` scenario before committing to it over atomics.
7. **Row sorting** changes nothing numerically on CPU (deterministic order is defined by spike-list
   order, not column order), but the builder must guarantee sorted, duplicate-free (pre, post) pairs.
8. **Which graph is "the" MaleCNS demo checkpoint**: ship both the retained-all-classified and the
   official `traced-only` variants with distinct names; demos default to the former (matches Xenova),
   science paths default to the latter.
9. The `w = 0` reset oddity and Brian2's exact refractory boundary semantics are settled empirically
   by the golden-trace generator, not by reading docs.
11. *M1 findings*: MaleCNS Feather files are LZ4-compressed Arrow IPC (`Apache.Arrow.Compression`
    needed; the 1.05 GB weights file streams in ~10 s). The 166,700-neuron demo graph is exactly
    `superclass IS NOT NULL`; the official `traced-only` export is exactly `status == 'Traced'` —
    both reproduce the note's edge/contact counts to the digit. Edges whose presynaptic transmitter has
    sign 0 (unclear/histamine: 677,985 edges, 2,733,429 contacts) are dropped at build time as
    `g += 0` no-ops; the structural counts are kept in provenance. Max contact count in either dataset
    is 2,591 → `int16` weights hold. Shiu's v630 inputs are Parquet with pandas-nullable int64 columns:
    read via Parquet.Net's raw API with definition-level checks. Brian2 2.5.1 has no Python-3.11
    Windows wheel; the golden generator runs on Brian2 2.9.0 (same equations, same slot semantics,
    version recorded in each fixture). The scalar float64 reference already runs the full v630 sugar
    experiment at 0.39× real time single-threaded (2.6 s per simulated second vs Brian2's ~5 min).
    *First full-graph comparison (v630, 21 sugar GRNs @ 150 Hz, 1 s, one trial each, independent
    Poisson streams):* Brian2 2.9 (numpy target) 13,739 spikes / 376 active / MN9 79 Hz in 25 s wall;
    dotFly reference 13,169 / 375 / 77 Hz in 2.6 s. Active-set Jaccard 0.93, rate correlation 0.99,
    spike-count ratio 0.96; every disagreement is a neuron with ≤ 2 spikes. The published
    `results/example/sugarR.parquet` (Brian2 2.5.1, 30 trials) is ~20 % more active (17,050 spikes /
    403 active / MN9 93 ± 3 Hz per trial) than *both*, so that gap sits between Brian2 versions or run
    environments, not between dotFly and Brian2 — to be investigated in M3 (validation 3) by pinning
    Brian2 2.5.1 on a Python 3.10 environment.
12. *M2 findings* (each worth remembering before touching the hot path):
    - **Tiered JIT was the biggest single factor**: the batch loop and kernel ran as tier-0/OSR code for
      most of a short run — 21 µs/step instead of 6 µs. `[MethodImpl(AggressiveOptimization)]` on
      `LifKernel.Update`, `Run` and the delivery/input methods fixed it (3.5× on MaleCNS idle).
    - **Refractory neurons are a fixed point** when `v_rst == v_0` (Shiu): they sit at `(v0, 0)`, cannot
      reach threshold, and discard inputs — so phase A needs no blend at all, only the countdown and
      a *blocked bit*. Kernel went from 2.5 to 0.8 cycles/neuron. The general (`v_rst ≠ v_0`) path
      keeps the masked update.
    - **Blocked flags as bits** (`ushort` per 16 neurons, from `vmovmskps`) instead of bytes: no
      narrow/pack sequence in phase A and a 20 KB, L1-resident mask for the delivery scatter.
    - **One barrier per 18-step batch** (the synaptic delay) instead of two per step: within a step a
      block touches only its own state plus spike lists published ≥ 18 steps earlier, so threads run
      whole batches unsynchronised; the caller merges/publishes per-step lists at the boundary.
      Sinks are notified after the batch (`ISpikeSink.OnStep` per step, in order).
    - **Poisson as geometric inter-arrival times** keyed by (seed, neuron, event#): one compare per
      stimulated neuron per step and ~p draws instead of one Philox call each (28 → 1 µs/step for
      1,684 stimulated neurons). Same scheme in the reference backend → identical event streams.
    - **Adaptive contiguous block ownership**: 4 blocks per thread; the cut points between threads
      move per batch according to measured phase-A ticks + delivered edges × self-calibrated
      per-edge cost. Contiguity matters: delivery is one column walk per spike per thread (per-block
      walks quadrupled the random offset-table reads and made balancing slower than imbalance).
    - Delivery is DRAM-latency-bound on the per-spike offset-row and column-slice fetches (~10 ns/edge
      effective at 8 threads); prefetching the next two steps' due lists (known 18 steps ahead) is
      worth ~15 %. The `visual-heavy` regime (10⁴ spikes/step) remains GPU territory as planned.
    - Laptop reality: all-core clocks and power limits cap phase-A scaling at ~5× on 8 cores; the
      pinned pool (workers on even logical processors) avoids SMT-sibling co-scheduling.
    - BenchmarkDotNet 0.15 does not know the .NET 11 runtime moniker; use `--inProcess`.
13. *M3 findings*:
    - **Validation 3 resolved**: the published `results/example/sugarR.parquet` (~17 k spikes/trial,
      MN9 93 Hz) is *not* what the shipped code and v630 inputs produce. Brian2 **2.5.1** (the pinned
      version, run in a Python 3.10 venv) gives 13,177 spikes / 378 active / MN9 78 Hz on one trial,
      Brian2 2.9 gives 13,739 / 376 / 79, dotFly gives 13.2–13.9 k / 369–384 / 77–87 across seeds and
      83.3 Hz over 30 trials. The published file was produced under other conditions (data version,
      parameters or code revision unknown). Per-neuron rate correlation with it is still 0.995.
      The comparison target for future work is Brian2 on the same inputs, which dotFly matches.
    - Silencing the three most active sugar GRNs one at a time: MN9 83.3 → 83.3 / 78.7 / 79.6 Hz.
    - Engine overhead is negligible: 30 reseeded trials on v630 run at 14× real time through the
      API (vs 19.6× raw); creating a `Simulation` per trial costs ~0.2 s (blocked-CSR build), hence
      `Reset(seed)`.
    - `Recording.Replay` reproduces a run's spikes exactly from the recorded input writes and
      injections (same seed); thread-safe port writes take effect at the next `Advance`.
15. *M4b — 3D closed visual loop* (`samples/DotFly.Sample.Godot3D`): the fly's own 64×32 eye render
    → per-eye looming (LC4) and optic-flow channels (T4a–d, from row/column profile shifts) →
    6,987 input neurons → measured readouts LC4→DNp04/DNp01 (escape) and T4a→HS, T4b→H2, T4c→VS
    (optic flow, all lateralized exactly as the literature says) → fixed decoder (escape yaw +
    optomotor counter-turn + VS altitude hold + GF speed burst). Runs at RTF 1.00, 0 ms behind,
    ~20–35 % busy on the 8-core laptop, and the fly navigates the arena for as long as it runs.
    Findings: (a) a `SubViewport` is not a `Node3D` — a camera inside it does not follow its parent,
    set its `GlobalTransform` per frame; (b) the optomotor sign: HS on the side seeing front-to-back
    motion means the fly rotates *towards* that side, so the reflex is yaw ∝ HS_R − HS_L (the other
    sign is positive feedback and spins); (c) **the model has a bistable optic-lobe attractor**: at
    gain 1.0, 3 Hz of one-sided LC4 input ignites ~450 k self-sustaining spikes/s (10 Hz does not —
    it is coincidence-triggered, not rate-driven), while at synaptic gain 0.85 nothing ignites and
    the escape response is unchanged (DNp04 293 vs 305 Hz). `SimulationOptions.SynapticGain` (and
    `dotfly run --gain`) exposes the model's one free parameter; every use other than 1.0 is shown
    as a calibration. Xenova's `omittedFastOutputNeurons: 11068` is plausibly their answer to the
    same phenomenon.
16. *M4c — the "fancy" 3D scene* (same sample): altitude/pitch, an odor field from three sugar
    patches (wandering wind, downwind plume with a gradient to climb), ORN_DM1 L/R smell input,
    gustatory input (claw_tpGRN tarsal + LB3* labellar) when landed, and the measured GRN → MN9
    proboscis pathway as the "feeding" readout — found with the new `Brain.Upstream` /
    `dotfly inspect --upstream-of`. Left-side panels: eye render + channel bars, a brain map from
    the checkpoint's real soma positions (`Brain.SomaPositions`) lit by real spikes, and readout
    time-graphs captioned with the decoder rule each one feeds. Findings: (a) **the olfactory side
    has its own runaway**: *any* ORN input, at any gain, ignites an antennal-lobe / mushroom-body loop
    (Kenyon cells KCg-m/KCab-*, PAM, lLN* local neurons) that never stops — the sugar-only gain 0.85
    trick does not help; (b) the fix that keeps gain 1.0 is **silencing** Kenyon cells + `lLN*`
    (4,225 neurons, `FlyBrain3D.StabilityControl`), after which every readout (DNp04, HS/VS, DM1_lPN,
    MN9) is graded and returns to baseline within a second of the input stopping (verified with the
    new `dotfly run --stim-seconds`); (c) odor *direction* is not derivable from one glomerulus (both
    DM1 PNs respond alike to either antenna), so the search is decoder klinotaxis (run-and-tumble on
    the PN rate trend); wall walking is not derivable from any readout and is left out; (d) visual
    cross-talk — T4 optic flow at tens of Hz per pixel swamps the other readouts, so T4 is encoded
    at 12 Hz/px, capped at 60 Hz, and LC4 at 100 Hz for full looming. Verified end to end: odor
    tracking → landing → MN9 at ~75 Hz → feeding → take-off, RTF 1.00.
    Second pass (room scene): a 24×18×5 m living room (Forward+, procedural textures, real window
    openings with sun shadows, couch, two tables, four columns), sugar on the tables and the floor
    (patches carry their surface height; landing is relative to it), **two eye cameras** (one per
    compound eye, 45° to each side, 32×32 luminance — the motion pathway's R1–R6 are achromatic, so
    grayscale is the faithful encoding), **the odor rendered as volumetric fog by a fog shader that
    evaluates the same plume function as `Odor.cs`** (the eye cameras use an environment override
    without it — the fly does not see smell), a third-person camera with a low-passed heading
    (turns are the fly's, not the camera's; it orbits to the front-side when the fly lands), a
    part-based animated fly (legs, wings, halteres, two-segment proboscis; feeding pose), a
    full-height scalable left column and an isometric room map with the flight path. Findings:
    (e) the Poly/Google CC-BY low-poly fly is a single baked mesh (no rig), so a downloaded model
    cannot feed — the procedural fly can; (f) volumetric-fog density must saturate
    (`d = k·c/(1+2c)`) or the chase camera inside the plume core sees nothing; (g) a `Node3D.Scale`
    applies before its rotation — an elongated sphere rotated to lie along the body must be scaled
    along the axis that ends up along the body.
    Third pass (photoreal room): SDFGI + SSR + SSIL + PCSS + physical sky + ACES, procedural PBR
    materials with normal maps (FastNoiseLite height → `Image.BumpMapToNormalMap`), more furniture;
    the fly at a 0.36 m body length; the eye panel shows the retina's own L8 images (luminance,
    not the colour render); odor **raymarched** per pixel (`odor_raymarch.gdshader`: same plume
    function, depth-buffer stop, wind-advected `NoiseTexture3D` billows, on a visual layer the eye
    cameras cull); panel labels ×1.5 with clipping. Findings: (h) with a realistic room the fixed
    dark threshold (90/255) made half of every eye "dark" and saturated LC4 on both sides — the
    looming threshold is now relative to each eye's mean luminance (adaptation), back to 1–15 %;
    (i) a constant sink made the small fly crawl the floor and wedge into corners — the flight
    model now holds a cruise altitude (1.6 m, 0.9 m while tracking odor) and a body "hop" reflex
    (turn + climb, labelled on the HUD) frees it when it has not moved for 3 s; (j) volumetric-fog
    density units and raymarch optical depth differ by an order of magnitude — the raymarch needs
    `k ≈ 0.07` with a saturating `c/(1+c)` to stay a haze rather than a wall.
    Fourth pass (odor channels + FPS): plumes are now narrow meandering ribbons (0.9 m, 14 m,
    two drifting sinusoids on the centreline, identical in `Odor.cs` and the shader via a shared
    `time_s`), drawn on the room map as centrelines; the fly still finds them (feeds at ~110 s).
    Profiled with `--print-fps` and env toggles: the pretty rendering was **not** the offender —
    `low`/no-odor was 28.7 ms/frame because (k) two synchronous `ViewportTexture.GetImage()`
    readbacks per physics frame stalled the GPU pipeline (~12 ms) → replaced by
    `RenderingDevice.TextureGetDataAsync` with a callback (RGBA8 → luma in C#, sync fallback for
    the compatibility renderer), and (l) the brain map's per-frame 166k-neuron heat + splat +
    texture upload plus 7,189 `DrawRect` calls for the input dots (~16 ms) → 15 Hz updates, dots
    baked into the texture; the minimap path is one `DrawPolylineColors` call instead of up to
    6,000 `DrawLine`s. Result 8.8 ms (`low`), 10.3 ms (`medium`, default), 12.8 ms (`high`).
    Fifth pass: odor as soft-edged **tubes** (0.7 × 0.45 m, core 0.45 m above the surface, cyan
    with a self-luminous core against the warm room), clipped at the walls and faded within 0.8 m
    of any wall (nothing pools in corners); brain map with the VNC compressed past the neck so the
    brain fills the panel; the fly on its own visual layer (its 120° eyes saw its wings and legs)
    and without shadows. Finding (m): **HS alone steers into walls** — front-to-back flow on one
    eye is the same for a turn and for flying past a wall, so `yaw ∝ HS_R − HS_L` hugged walls and
    corners. Reading T4b → H2 (back-to-front, measured) as well separates them: rotation = HS one
    side + H2 the other → counter-turn; translation = HS one side only → centering (turn away),
    `yaw += 0.2·((HS_R+H2_L) − (HS_L+H2_R)) − 0.3·(HS_R − HS_L)`. The fly now crosses the room
    freely (three feedings in 150 s, no corner dwelling), 94 FPS.
    Sixth pass: `--flies N` — N independent full simulations in one process. Needed a library
    addition: `ComputeThreadPool(threadCount, pin, pinOffset, pinTotal)` and
    `SimulationOptions.ThreadPinOffset/ThreadPinTotal`, because every pool pinned its workers
    from logical CPU 0 upward and N pools would have stacked on the same cores. 4 flies × 2
    threads on the 8-core laptop: every fly at RTF 1.00, ~90 % busy, 36 FPS (the render thread
    now competes with 8 pinned workers). The brain map takes the panel's aspect (brain to scale,
    VNC in the rest, compressed only when needed).
    Seventh pass — "sticky" odor tracking. Measured first: the DM1_lPN response to ORN_DM1 input is
    58 / 88 / 134 / 214 / 280 / 302 Hz at 5 / 10 / 20 / 40 / 80 / 120 Hz, so the old 40 Hz "detected"
    threshold fired on the far tail (c ≈ 0.03) — now 110 Hz (inside the tube), "near" 270 Hz.
    Decoder additions: odor memory (8 s) with a U-turn back into the tube and zigzag casts of
    growing duration at 1.0 m/s; a height memory (the tubes have a core height) with ±0.35 m
    vertical casting; cruise at 1.0 m (between floor-level and table-level tubes). The decisive
    addition is **wind**: the antennae's Johnston's organ JO-C/E neurons (203 per side) reach
    AMMC012 (11960 / 11702) and WED080 (13072 / 12566), each answering one antenna only (130 vs
    0 Hz) — a measured lateralized wind-direction readout — so the fly *surges upwind* on odor
    (`yaw −= 0.6·(wind_L − wind_R)`), which along a downwind tube leads to the sugar (the real
    fly's strategy, Álvarez-Salvado 2018). Result over 5 min: tracking 68 % of the time, 85 % of
    odor losses recovered by the casts, four feedings (was ~one).
    Eighth pass — no altimeter, 3-D tunnels. The height memory (remember the world height where
    the readout peaked) was dropped as a cheat: a fly has no altimeter. Now the vertical search
    uses only the readout and the decoder's own last move — without odor a random vertical drift
    (±0.25 m/s, re-drawn every 4–10 s, slightly down-biased, bounded by floor and ceiling); in
    odor ±0.35 m/s, reversed half the times the readout falls (vertical klinotaxis) and at random
    while casting. The VS climb term had to drop to 0.002 m/s/Hz: at 0.01 it pushed the fly to the
    ceiling once the hold was gone. The tunnels meander vertically too (±0.55 m, λ 7.5 m) and rise
    5 cm per metre downwind (buoyancy), capped 0.5 m below the ceiling — same function in
    `Odor.cs`, the shader and the map. Wind readout narrowed to AMMC012 alone (its sides answer
    alike, 130/112 Hz; WED080's 134/65 Hz made the fly circle left). Cost of honesty: 1–2 feedings
    per 5 min (was 4 with the memory), tracking ~38 %, ~90 % of losses recovered.
    Ninth pass: the user's 6,654-triangle low-poly Drosophila (`models/fly.glb`, faces +Z, wings
    as hinge nodes) replaces the procedural fly. Wings: two wrong attempts (ghost copies fanned
    over the stroke plus an envelope disc = a helicopter rotor; wings held out with a smear = a
    static blur), then the right one from the high-speed kinematics (Bergou/Ristroph/Cohen/Wang,
    physics.aps.org v25/st13): the real wing nodes row fore–aft in a near-horizontal plane
    through 140° and flip pitch ±45° at each reversal, at a visible 14 Hz with two short trailing
    ghosts; folded on landing. Feet placed at the landed rest height from the model's AABB.
    Tenth pass — scale. The fly was 0.36 m (1:6.4 against a 2.3 m door; a real Drosophila is
    3 mm, 1:770). Body length is now one parameter (`FlyBody3D.BodyLength`, `DOTFLY_BODY_LENGTH`),
    default **0.03 m** (10× a real fly: a fly's world that the chase camera can still show);
    everything body-relative follows it — eyes, antennae, collider, landing heights, stuck
    detector, camera distance and near planes — and the decoder's speeds and climb rates are now
    in **body lengths per second** (cruise 120 BL/s, tracking 55, casting 35; a real fly cruises
    at 100–170 BL/s), multiplied by the body length at run time. The room and the odor tubes stay
    at room scale for now. Finding (n): the chase camera's first-order position smoothing lags
    the fly by speed/rate (≈ 0.6 m) — invisible at 0.36 m, fatal at 3 cm; the camera now rides
    rigidly with the fly and only its offset is smoothed.
    Camera modes (`C`): chase (as before) and follow — FPV-like, locked 2.6 body lengths behind
    on the fly's own heading with τ ≈ 0.07 s, low and on-axis, no orbit on landing, so her front is
    never seen; `DOTFLY_CAMERA=follow` starts there. Then: follow rides on the fly's full
    orientation (yaw + 0.6·pitch) through a quaternion slerp (τ ≈ 0.08 s), chase moved out to 9
    body lengths, and both modes fade what lies between fly and camera: a ray from the fly to the
    camera (up to 6 hits, the fly's body excluded), each hit static body's meshes get a cached
    translucent copy of their material (alpha 0.3) and are restored 0.25 s after they clear
    (`OcclusionFader.cs`); `DOTFLY_SCREENSHOT_STATE=Faded` captures the first such moment.
    Then: the fly was "centred" on the whole window while the panels hide its left third, so she
    sat left of the visible area — both modes now translate the camera sideways (`Camera3D.HOffset`
    = −panelFraction · tan(hfov/2) · distance) so she projects to the centre of what is visible
    (measured: x = 864 / 859 of the 860 target). Follow is on a rubber band: the camera's yaw and
    pitch are a damped spring (ω = 9 rad/s, ζ = 0.55) driven by the fly's unwrapped yaw and pitch,
    the camera sits on that smoothed axis, so turns are followed with momentum, not on a stick.
    That still showed her side in turns (any camera pinned to her heading pivots around her), so
    follow became a **trail camera**: her path is recorded (≈40 body lengths, cumulative length),
    the camera sits at the trail point 2.6 body lengths of path behind her and looks at her — in a
    turn it passes through the same bend she did; when she reverses toward it the offset blends to
    behind-her-heading (smoothstep on the rear-view dot). Measured over a 3-min run: rear-view dot
    > 0.7 in 100 % of the 2-s samples, including tumbles, casts and U-turns.
    Finding (o): the `HOffset` centring was a camera *translation* — it put the camera 0.39·d to
    the left of her trail, so left turns showed her flank and right turns did not. Replaced by a
    lens shift (`Projection.Frustum` + `FrustumOffset`, image slides, camera stays on the trail;
    she projects to x = 860 exactly in both modes); follow distance doubled to 5.2 body lengths.
    Then both cameras moved out 1.5× (chase 13.5, follow 7.8 body lengths); the odor back to warm
    gold with glitter (fine noise^9 specks twinkling with time and view direction, ×12 so they
    bloom) and the tunnels made buoyant (rise 0.28 m per metre, vertical width growing with
    distance, capped 0.35 m below the ceiling and spreading along it).
    Rigged model (2026-09-20): `models/fly.glb` replaced by the user's rigged version — same
    6,654 triangles, a 25-bone leg rig and five clips (standing, flight, landing_pose, takeoff,
    land) — loaded at runtime with `GltfDocument` (the AnimationPlayer and Skeleton3D come out of
    `GenerateScene`), clips driven by the behaviour state (Flying → flight, Landing →
    landing_pose, Landed/Feeding → standing, TakeOff → takeoff then flight); the wings stay
    separate hinge nodes, so the rowing wing beat is unchanged; the model's GDScript controller
    is not used.
    Cones: plumes are now cones (0.5 × 0.35 m at the sugar, width × (1 + a/2.5) downwind, 18 m,
    axis diluted by √(1 + a/2.5) so "near" means within ~2 m): detectable odor fills ~28 % of the
    room volume (was 3–5 %); the fly is in odor 60–76 % of the time. Finding (q): she then landed
    on the bowl and left "nothing tasted" — with odor and wind inputs on, GRN → MN9 drops from 95
    to 35 Hz in this model (measured with the CLI), about one spike per 30 ms readout window, so
    the 20 Hz single-window test flickered; feeding is now judged on a 0.3 s smoothed MN9 > 15 Hz
    and she waits 6 s on the food before giving up.
    Then: the raymarch keeps the segment from the camera to the fly clear (`clear_distance`
    uniform = camera–fly distance × 1.15), the plates got colliders so the occlusion fader sees
    them, the landed chase camera sits higher; decoder readouts smoothed (PN τ 0.15 s, wind τ
    0.3 s), tumble only below −25 Hz/s, tracking at 80 BL/s; wind turns once per 15 min and the
    meander drifts at half speed — a plume whose far end sweeps faster than the fly tracks is lost.
    Glitter re-done as hashed motes (3.5 cm cells, one mote in twelve, own flicker) — the tiling
    64³ noise texture sampled at 11× produced a visible grid; the fly's shadow is back.
    **Finding (p) — is odor navigation in the connectome?** Measured with the CLI: driving JO-C/E
    (wind) and ORN_DM1 (odor) leaves the steering pathway of the literature silent in this model —
    DNa02 0–8 Hz, DNa03 ≤ 10 Hz, EPG 3–13 Hz and symmetric, PFL3/hΔC ~0 — while wind strongly and
    symmetrically drives neck/GNG descending neurons (DNg73, DNge027, DNge143, 150–175 Hz) and a
    few unnamed-function DNs lateralize with wind side (DNg33 94/190, DNg29 86/2, DNb05 0/70,
    DNp33 72/0). Real flies steer upwind through wind (WPN → LAL → PFN) and odor converging in the
    fan-shaped body (hΔC; Matheson et al. 2022; Okubo et al. 2020) against the EPG compass, with
    PFL3 turning goal − heading into a left/right drive of DNa02 (Westeinde et al. 2024; Mussells
    Pires et al. 2024; Rayshubskiy et al.). The central complex is a tuned recurrent system (ring
    attractor, Δ7 inhibition, tonic drive, self-motion and landmark inputs) that the uniform-LIF
    model with no inputs to it cannot run; the same way the KC/LN loop runs away, the CX does
    nothing. Paths to a connectome-native tracker, in order of honesty: (1) connectome-constrained
    parameter fitting per cell type (Lappalainen et al. 2024, Nature, did this for the visual
    system): per-type synaptic gains, time constants and thresholds fitted so that measured
    behaviours (ON surge / OFF cast, Álvarez-Salvado et al. 2018; van Breugel & Dickinson 2014)
    or recordings are reproduced — dotFly has the fast backend for the inner loop, needs
    per-type gain tables in the checkpoint builder and a search loop; (2) feed the CX its real
    inputs (ring-neuron visual heading, PFN self-motion) and test whether a bump forms — unlikely
    without (1); (3) read the wind-lateralized DNs of unknown function as "steering" — measurable
    but not defensible. The current decoder (surge/cast on PN + AMMC012 wind) remains labelled
    engineering.
    **M6 — connectome-native navigation (written up 2026-09-20 as the M5 candidate; now M6, after the vision-boundary milestone M5).** Today the wiring is
    measured, the dynamics are one fitted parameter (Shiu), the encoders/decoders are adapters,
    and the *navigation decision* (surge/tumble/cast/height) is ours. The real circuit: wind
    JO-C/E → WPN (Suver 2019) → LNa → PFNa (Okubo 2020, Neuron) and odor → FB tangentials meet in
    the fan-shaped body at hΔC, giving an upwind goal (Matheson 2022, Nat Commun); the EPG compass
    is a ring attractor with PEN shifting and Δ7 inhibition, anchored by ER ring neurons (Seelig
    & Jayaraman 2015; Kim 2017; Turner-Evans 2017, 2020; Hulse 2021); PFL3 computes goal − heading
    and drives DNa02 left/right (Westeinde 2024; Mussells Pires 2024; Rayshubskiy). All present in
    the checkpoint; all silent in the uniform LIF (no tonic activity, one weight for every
    synapse, recurrent loops die or explode). Route, following Lappalainen et al. 2024 (Nature —
    connectome-constrained visual model: synapses fixed from the connectome, a few parameters per
    cell type fitted on a task, physiology predicted): (1) experiments first — recover the EPG
    ring order from EPG/PEN/Δ7 connectivity, inject a bump and test persistence, single-ness,
    rotation under PEN drive, and whether PFNa wind + FB odor reach PFL3/DNa02 (expected: the bump
    dies — that failure is the objective); (2) per-cell-type parameters (synaptic gain, τm, tonic
    current, threshold) as `.dfb` type-gain tables + `SimulationOptions.TypeGains`; a subgraph
    checkpoint extractor (CX + wind + olfactory + steering, ~5–10 k neurons) so the fast backend
    runs hundreds× real time; CMA-ES / evolution strategies against quantitative targets: EPG
    bump persistence/rotation, hΔC & PFN wind tuning, PFL3 L−R ∝ sin(goal − heading), DNa02
    asymmetry; alternatively a differentiable rate surrogate trained by backprop and ported back;
    (3) validation on behaviour never fitted: ON surge / OFF cast (Álvarez-Salvado 2018; van
    Breugel & Dickinson 2014) in the 3D scene with the surge/cast decoder switched off and yaw read
    from DNa02 L−R. Only the encoders and the last motor mapping stay as adapters. Risk: the LIF
    with per-type gains still cannot hold a bump → a neuron-model term (adaptation, conductance
    synapses) is next, still "the connectome with better dynamics". CUDA becomes M7, lanes/AOT M8.
14. *M4 findings*:
    - **Godot hosting.** (a) The winget `godot.exe` is a shim in a `Links` folder; Godot's .NET module
      looks for `GodotSharp/` next to the *real* executable and fails silently through the shim — run
      the real `Godot_v4.7.1-stable_mono_win64[_console].exe`. (b) Godot's plugin host resolves the
      highest *stable* runtime, so a `net11.0` game assembly fails with "System.Runtime 11.0.0.0 not
      found" until .NET 11 GA; `DOTNET_ROLL_FORWARD_TO_PRERELEASE=1` fixes it (documented in the
      sample README). (c) Godot builds the game in **Debug**, and that configuration flows into
      referenced projects — the SIMD kernel ran 25× slower (150 µs/step per thread) until the library
      `ProjectReference`s were pinned with `SetConfiguration="Configuration=Release"`. (d) Pinning
      the caller thread to CPU 0 is now opt-in (`DOTFLY_PIN_CALLER=1`): in a host it competes with
      the host's main thread; workers stay pinned.
    - **LC4 → escape circuit, measured in the checkpoint.** Driving LC4 (looming-sensitive visual
      projection neurons) at 120 Hz on one side lights DNp04, DNp01 (Giant Fiber), DNp02 and DNp11
      *ipsilaterally* (e.g. LC4_L → DNp04 531898 (left) 305 Hz, DNp01 10010 (left) 255 Hz). DNa02/DNp09
      (the Xenova readouts) stay silent under this input, so the Godot decoder reads DNp04_L/R and
      DNp01 instead — an escape reflex: turn away from the looming side, speed up with the GF.
    - **Unilateral LC4 drive is a heavy regime**: 787 k spikes/s from 14 k neurons versus 166 k spikes/s
      for bilateral drive (contralateral balance suppresses it). The CLI runs it at 3× real time; the
      Godot scene at 1.00× with 75 % headroom.
    - ML.NET's ONNX export keeps the multiclass trainer's `Label` input even when only `Score` is
      requested; `OnnxAdapter` feeds zero tensors to any extra graph inputs, so exported models run
      with just the feature vector.
    - The 30 ms/20 ms rate readout over 1,314 DNs separates gustatory / mechanosensory / olfactory
      stimulation at 96–98 % with a linear readout — the anatomy-chosen readout is informative, the
      adapter is small, and only the adapter was trained.
10. *M0 findings*: xunit v3 on the .NET 10+ SDK requires `"test": { "runner": "Microsoft.Testing.Platform" }`
    in `global.json` and an `Exe` test project; Spectre.Console.Cli is not AOT-annotated, so the CLI
    opts out of the AOT analyzers while the library packages stay `IsAotCompatible` (Godot consumers
    are the ones that matter). This machine has AVX2/FMA but no AVX-512 — `Vector256` is the local
    fast path; AVX-512 numbers need CI or another box.

## 13. References (pinned)

- Shiu et al. model: `philshiu/Drosophila_brain_model@91bdd1e7dcf193f3e7ca5a8933497fcef63b7960` (`model.py` — equations quoted in §2)
- Eon backends & comparison metrics: `eonsystemspbc/fly-brain@a3db62f9436074e485c0278290c2164ed6150808`
- Xenova MaleCNS config: `huggingface.co/spaces/Xenova/fruit-fly-simulation@776d115ee5aa934578a87fd6d260d138084f59c1` (`public/model.json`)
- MaleCNS downloads / release history: https://male-cns.janelia.org/download/ , https://male-cns.janelia.org/release/
- FlyWire v783 connectivity: https://zenodo.org/records/10676866
- Shiu experiment outputs: https://doi.org/10.17617/3.CZODIW
- dotLLM conventions and CUDA-via-PTX pattern: `C:\github\ai\dotLLM\CLAUDE.md`, `src/DotLLM.Cuda/CudaModule.cs`, `src/DotLLM.Cpu/Threading/`
