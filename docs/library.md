# Library guide

Everything in `DotFly.Core` / `DotFly.Engine`, by task, with the code you would actually write.
The API reference has every member; this page is the tour.

## 1. Open a checkpoint

```csharp
using DotFly;                       // Simulation, ports, RealtimeDriver, units (Ms(), Seconds())
using DotFly.Core.Graph;            // Brain, NeuronSet, Side, NeuronInfo, EdgeSpan
using DotFly.Core.Models;           // Neurotransmitter, NeuronModel

using Brain brain = Brain.Open(".data/malecns-v1.0/malecns-v1.0-superclass.dfb");

Console.WriteLine($"{brain.Provenance.Name}: {brain.NeuronCount:N0} neurons, {brain.EdgeCount:N0} edges");
Console.WriteLine($"w_syn {brain.Model.SynapseWeightMv} mV, sign policy {brain.Provenance.SignPolicy}");
```

`Brain` is a memory-mapped, immutable view of the `.dfb` file: opening is instant, nothing is
copied, and several `Simulation`s can share one `Brain`. The provenance block records the source
files (SHA-256), the filter, the sign policy and the dropped-edge counts, so a result can always be
tied to the data it came from.

## 2. Select neurons

A `NeuronSet` is an immutable, sorted list of neuron indices with a name. Everything — inputs,
outputs, silencing, exploration — takes one.

```csharp
NeuronSet lc4Left  = brain.Query(type: "LC4", side: Side.Left, name: "LC4_L");
NeuronSet hs       = brain.Query(type: "HSE") | brain.Query(type: "HSN") | brain.Query(type: "HSS");   // union
NeuronSet lb3      = brain.Query(type: "LB3*");                                   // trailing/leading * = glob
NeuronSet kenyon   = brain.Query(cls: "Kenyon_Cell");                             // by class
NeuronSet gaba     = brain.Query(superclass: "descending_neuron", nt: Neurotransmitter.Gaba);
NeuronSet wind     = brain.ByBodyIds("wind_L", [11960UL, 13072UL]);              // exact 64-bit IDs
NeuronSet one      = brain.ByBodyId(720575940660219265);                          // FlyWire MN9

NeuronSet notKc    = hs - kenyon;                                                 // difference
NeuronSet both     = lc4Left & brain.Query(nt: Neurotransmitter.Acetylcholine);   // intersection

NeuronInfo info = brain[lc4Left[0]];   // type, class, superclass, instance, NT, side, soma (voxels)
Console.WriteLine($"{info.BodyId} {info.Type} {info.Side} soma {info.Soma}");
```

Body IDs are `ulong` and never pass through a floating-point type. Named sets stored in the
checkpoint (from the builder) are in `brain.Sets`.

### Following the wiring

```csharp
NeuronSet targets = brain.Downstream(lc4Left, hops: 1, minWeight: 5);   // who LC4_L drives, ≥ 5 contacts
NeuronSet drivers = brain.Upstream(one, hops: 2, minWeight: 5);         // who drives MN9, two hops back

EdgeSpan edges = brain.OutEdges(lc4Left[0]);                            // raw CSR row: targets + signed weights
for (int k = 0; k < edges.Length; k++)
{
    Console.WriteLine($"→ {brain[edges.Targets[k]].Type}  {edges.Weights[k]:+#;-#} contacts");
}
```

This is how the demo's readouts were found: stimulate a population with the CLI, see who fires,
confirm the path with `Upstream`.

## 3. Create a simulation

```csharp
using Simulation sim = brain.CreateSimulation(new SimulationOptions
{
    Seed = 1,                 // same checkpoint + options + seed ⇒ identical spikes, at any thread count
    Threads = 0,              // 0 = one per physical core
    SynapticGain = 1.0,       // the model's one free parameter; ≠ 1.0 is a *calibration* — say so
    Backend = Backend.Cpu,    // float32 SIMD; Backend.Reference = float64, matches Brian2 to rounding
});
```

The model constants come from the checkpoint (`NeuronModel.Shiu2024`: exact integrator, 18-step
delay, 22-step refractoriness, Poisson weight 68.75 mV) — you don't set them per run.

## 4. Inputs

```csharp
InputPort odor = sim.Input(brain.Query(type: "ORN_DM1", side: Side.Left), InputKind.PoissonToV);
odor.Fill(60f);                          // 60 Hz Poisson drive on every neuron of the set
odor.Set1(position: 3, value: 120f);     // one neuron of the set
odor.Write(rates);                       // a whole span, one value per neuron of the set

InputPort tonic = sim.Input(brain.Query(type: "EPG"), InputKind.Current);
tonic.Fill(0.05f);                       // +0.05 mV added to v every step (a tonic drive; not in Shiu's model)
```

`PoissonToV` is the Shiu et al. stimulation (Brian2 `PoissonInput(target_var='v')`): each event
adds 68.75 mV, so every accepted event fires the target, and driven neurons get no refractory
period. Port writes are thread-safe and take effect at the next `Advance` — a game thread can
write them while the simulation thread runs.

## 5. Outputs

```csharp
NeuronSet mn9 = brain.Query(type: "MN9", name: "MN9");
OutputPort rate  = sim.Output(mn9, OutputKind.Rate(30.Ms()), publishEvery: 20.Ms());   // Hz per neuron
OutputPort count = sim.Output(mn9, OutputKind.SpikeCount(200.Ms()));                   // spikes per window
OutputPort v     = sim.Output(mn9, OutputKind.Membrane);                                // v in mV, every publish

OutputFrame f = rate.Snapshot;           // lock-free: the latest published frame
float mean = f.Mean();                   // over the set
float first = f[0];                      // per neuron, in set order
Console.WriteLine($"frame {f.FrameIndex} at {f.NeuralTime}: MN9 {mean:F0} Hz");

rate.Published += (port, frame) => { /* called on the simulation thread, every publishEvery */ };
```

A `Rate` window of 30 ms over two neurons has a resolution of ~17 Hz per spike — for decisions,
smooth over several frames (the room demo uses τ = 0.15–0.3 s).

## 6. Every spike

```csharp
long total = 0;
sim.OnSpikes += (long step, ReadOnlySpan<int> ids) =>
{
    total += ids.Length;                 // ids are neuron indices; runs on the simulation thread, no allocation
};
```

## 7. Running, scheduling, injecting

```csharp
sim.Run(1.Seconds());                    // neural time; blocks until done
sim.Advance(180);                        // 180 steps of 0.1 ms

sim.ScheduleAt(250.Ms(), s => s.Inputs[0].Fill(0f));           // an action at a neural time
sim.Inject(brain.IndexOf(720575940660219265), at: 300.Ms());   // one Poisson-sized event on one neuron
sim.Inject(mn9, at: 300.Ms());                                  // on a whole set

Console.WriteLine($"{sim.NeuralTime} neural in {sim.Clock.WallTime} wall = {sim.Clock.RealTimeFactor:F1}×");
```

Neural time, simulated duration and wall time are reported separately and never conflated.

## 8. Trials: reset, snapshot, restore

```csharp
for (int trial = 0; trial < 30; trial++)
{
    sim.Reset(seed: 1000 + (ulong)trial);   // same inputs, a fresh Poisson stream; ports keep their values
    sim.Run(1.Seconds());
}

SimulationSnapshot before = sim.Snapshot();   // full state (v, g, refractoriness, delay ring, RNG)
sim.Run(500.Ms());
sim.Restore(before);                          // back to the exact same state
```

Use `Reset` for trials rather than a new `Simulation`: constructing one rebuilds the blocked CSR
(~0.2 s for MaleCNS).

## 9. Silencing and causal switches

```csharp
sim.Silence(brain.Query(cls: "Kenyon_Cell") | brain.Query(type: "lLN*"));   // outgoing weights → 0; they still spike
sim.Unsilence(kenyon);

sim.RecurrentTransmission = false;   // no synaptic delivery at all: the network is a pass-through of its inputs
sim.ExternalInput = false;           // no input events: the network runs on nothing
```

Silencing is Shiu et al.'s `silence()`. The two switches are what the demos' `R` / `E` keys flip:
they let an audience see what the wiring contributes versus what the adapters contribute.

## 10. Real time, in a game loop

```csharp
using var driver = new RealtimeDriver(sim, new RealtimeOptions
{
    Ratio = 1.0,                              // neural seconds per wall second
    MaxLag = 250.Ms(),                        // beyond this the driver skips ahead and reports it
});
driver.Start();                               // its own thread; the game thread never blocks

// each frame:
odor.Fill(concentration * 120f);              // thread-safe write
float pn = rate.Snapshot.Mean();              // lock-free read
if (driver.BehindBy > 50.Ms()) { /* show "behind" on the HUD */ }

driver.Pause(); driver.Start();               // Space
```

Threading contract: `Simulation` methods run on one thread (the driver's); `InputPort.Write/Fill`,
`OutputPort.Snapshot`, the switches and read-only properties are safe from any thread. Inside a
host (Godot), don't pin the caller thread; the workers are pinned to their own cores.

## 11. Recording and replay

```csharp
using (Recorder rec = sim.Record("run.dfs"))       // spikes, input writes, injections, output frames
{
    sim.Run(2.Seconds());
}

Recording r = Recording.Read("run.dfs");
r.WriteSpikesCsv("spikes.csv");
r.WriteFramesCsv("mn9.csv", port: 0);

using Simulation again = brain.CreateSimulation(new SimulationOptions { Seed = sim.Seed });
r.Replay(again);                                  // reproduces the run's spikes exactly
```

## 12. Readout adapters

An adapter turns a frame's values into actions and carries its origin — *Fixed*, *Calibrated* or
*Trained* — so a demo can label it honestly.

```csharp
using DotFly.Adapters;

// Hand rules: action 0 = "feed" when feature 0 (MN9 rate) exceeds 15 Hz
var rules = new ThresholdAdapter(featureCount: 1, [new ThresholdAdapter.Rule(Feature: 0, Threshold: 15f)],
                                 AdapterOrigin.Fixed, "MN9 > 15 Hz → feed");

// A trained linear map (weights from samples/DotFly.Sample.TrainReadout), smoothed
IReadoutAdapter linear = new SmoothingAdapter(LinearAdapter.Load("readout.json"), alpha: 0.3f);

// ONNX / ML.NET models (separate packages: DotFly.Adapters.Onnx / .MLNet)
IReadoutAdapter onnx  = new OnnxAdapter("readout.onnx");
IReadoutAdapter mlnet = MLNetAdapter.Load(new MLContext(), "readout.zip", featureCount: 9);

Span<float> actions = stackalloc float[rules.ActionCount];
rules.Evaluate(rate.Snapshot, actions);
```

The graph never learns; only the adapter is trained. Training data come from recordings
(`Recording.Outputs`) or live frames.

## 13. Several simulations in one process

```csharp
int flies = 4, cores = DotFly.Cpu.Fast.CpuBackend.DefaultThreads(), each = Math.Max(1, cores / flies);
var sims = Enumerable.Range(0, flies).Select(i => brain.CreateSimulation(new SimulationOptions
{
    Seed = 1 + (ulong)i, Threads = each,
    ThreadPinOffset = i * each, ThreadPinTotal = flies * each,   // disjoint cores per simulation
})).ToArray();
```

Without the pin offsets every pool pins its workers from CPU 0 upward and they stack on the same
cores.

## 14. Pitfalls we hit so you don't

- **Runaway attractors at gain 1.0**: one-sided LC4 input or any ORN input can ignite a loop that
  never stops (optic lobe; Kenyon cells + antennal-lobe LNs). Check with `dotfly run --stim-seconds`
  that your readouts return to baseline; silence the loop or say the gain is calibrated.
- **Godot builds in Debug** and drags referenced projects into Debug — pin the dotFly references
  with `SetConfiguration="Configuration=Release"` or the kernel is 25× slower.
- **Tiered JIT**: per-step methods must carry `[MethodImpl(MethodImplOptions.AggressiveOptimization)]`
  (already the case inside dotFly; relevant if you write your own `IParallelBody`).
- **Readout noise**: rates over 30 ms from a handful of cells jump in ~17–33 Hz steps; smooth
  before thresholding.
