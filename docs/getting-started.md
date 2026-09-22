# Getting started

This repository is three things: the **library** (`src/DotFly.*`, NuGet-style projects), the
**CLI** (`src/DotFly.Cli`) and the **demos** (`samples/`). The connectome data are not in the
repository (CC BY 4.0, downloaded once into the git-ignored `.data/` folder).

## 1. Prerequisites

- **.NET 11 SDK**, RC1 or later — the exact version is pinned in `global.json`
  ([download](https://dotnet.microsoft.com/download/dotnet/11.0)).
- **Godot 4.7, .NET edition** — only for the two Godot demos ([download](https://godotengine.org/download)).
  Until .NET 11 GA, Godot needs `DOTNET_ROLL_FORWARD_TO_PRERELEASE=1` in its environment.
- Disk: ~300 MB for the FlyWire v630 files, ~3 GB for MaleCNS v1.0 plus its checkpoints.

## 2. Build and test

```powershell
git clone https://github.com/kkokosa/dotFly
cd dotFly
dotnet build -c Release
dotnet test  -c Release                                   # 94 unit + integration tests; no data needed
dotnet run --project src/DotFly.Cli -c Release -- info
```

## 3. Get a connectome

**Option A — FlyWire v630 (90 MB): enough for the library, the CLI and the sugar experiment.**
The files are in the Shiu et al. model repository:

```powershell
git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model .data/shiu2024
dotnet run --project src/DotFly.Cli -c Release -- build flywire .data/shiu2024/2023_03_23_completeness_630_final.csv .data/shiu2024/2023_03_23_connectivity_630_final.parquet -m 630
dotnet run --project samples/DotFly.Sample.SugarExperiment -c Release   # 30 trials of sugar → MN9 in ~2 s
```

**Option B — MaleCNS v1.0 (1.1 GB): needed by the room demo and the 2D demo.**
From <https://male-cns.janelia.org/download/> ("flat files"), put these three files into
`.data/malecns-v1.0/`:

```
body-annotations-male-cns-v1.0-minconf-0.5.feather              (~15 MB)
body-neurotransmitters-male-cns-v1.0.feather                    (~43 MB)
connectome-weights-male-cns-v1.0-minconf-0.5.feather            (~1.05 GB; or the 508 MB -traced-only file)
```

then build the checkpoint (a minute; writes `.data/malecns-v1.0/malecns-v1.0-superclass.dfb`):

```powershell
dotnet run --project src/DotFly.Cli -c Release -- build malecns .data/malecns-v1.0 --filter superclass
dotnet run --project src/DotFly.Cli -c Release -- inspect .data/malecns-v1.0/malecns-v1.0-superclass.dfb
```

`--filter superclass` keeps bodies with a `superclass` annotation (166,700 neurons; the graph the
demos use); `--filter traced` keeps `status == Traced` (165,122; the official `traced-only` export;
needs the `-traced-only` weights file). Edges whose presynaptic transmitter is unclear or histamine
are dropped as no-ops; the structural counts stay in the checkpoint's provenance.

## 4. Run the room demo

```powershell
$env:DOTNET_ROLL_FORWARD_TO_PRERELEASE = "1"
& "<path to>\Godot_v4.7.1-stable_mono_win64.exe" --path samples\DotFly.Sample.Godot3D
```

or open `samples/DotFly.Sample.Godot3D` in the Godot editor and press Play. Keys: `C` camera,
`R` recurrent transmission, `E` external input, `V` vision, `O` smell/taste, `S` silence DNp04+HS,
`Space` pause. `-- --flies 4` puts four flies in the room. The
[room demo page](demo.md) explains
everything on screen. Note: on Windows run the real Godot executable, not the winget shim
(`godot.exe` in a `Links` folder) — the .NET module fails silently through the shim.

## 5. Try the CLI

```powershell
dotnet run --project src/DotFly.Cli -c Release -- explore .data/malecns-v1.0/malecns-v1.0-superclass.dfb --stimulate type:T4a
dotnet run --project src/DotFly.Cli -c Release -- run .data/malecns-v1.0/malecns-v1.0-superclass.dfb --stimulate "type:LC4@L" --hz 100 --seconds 1 --silence "class:Kenyon_Cell;type:lLN*"
dotnet run --project src/DotFly.Cli -c Release -- bench .data/malecns-v1.0/malecns-v1.0-superclass.dfb --scenario type:T4a --threads 1,4,8
```

`tools/brian2_golden.py` regenerates the Brian2 reference fixtures (needs a Python environment with
`brian2`, `pandas`, `pyarrow`; it is not part of the build).

## Use the engine

```csharp
using DotFly;
using DotFly.Core.Graph;

using Brain brain = Brain.Open(".data/shiu2024/flywire-v630.dfb");
using Simulation sim = brain.CreateSimulation(new SimulationOptions { Seed = 1 });

NeuronSet sugar = brain.ByBodyIds("sugar_GRN_R", sugarIds);          // exact 64-bit IDs (the 21 sugar GRNs of Shiu et al.)
sim.Input(sugar, InputKind.PoissonToV).Fill(150f);                   // Poisson drive, 150 Hz each
OutputPort mn9 = sim.Output(brain.ByBodyId(720575940660219265), OutputKind.Rate(50.Ms()));
long spikes = 0;
sim.OnSpikes += (long step, ReadOnlySpan<int> ids) => spikes += ids.Length;   // every spike, no allocation

sim.Run(1.Seconds());                                                 // neural time
Console.WriteLine($"{brain.Provenance.Name}: {brain.NeuronCount:N0} neurons, {brain.EdgeCount:N0} edges");
Console.WriteLine($"MN9 {mn9.Snapshot.Mean():F1} Hz in the last 50 ms; {spikes:N0} spikes in {sim.NeuralTime.TotalSeconds:F1} s neural time, " +
                  $"{sim.Clock.WallTime.TotalMilliseconds:F0} ms wall ({sim.Clock.RealTimeFactor:F1}× real time)");
```

Output (8-core laptop, default thread count; the spike count is the same on every run and at any
thread count — only the wall time changes):

```
flywire-v630: 127,400 neurons, 14,687,178 edges
MN9 66.7 Hz in the last 50 ms; 13,776 spikes in 1.0 s neural time, 170 ms wall (5.9× real time)
```

For a game loop, `new RealtimeDriver(sim).Start()` runs the network on its own thread; the game
thread writes `InputPort`s and reads `OutputPort.Snapshot` frames (lock-free), and a readout
adapter (`LinearAdapter`, `ThresholdAdapter`, or an ONNX/ML.NET model) turns a frame's values into
actions — labelled *fixed*, *calibrated* or *trained*, because the graph never learns.
`sim.Record("run.dfs")` captures spikes, input writes and output frames; `Recording.Read(...)
.Replay(sim)` reproduces the run. `samples/DotFly.Sample.SugarExperiment` is the worked example.

## Samples

| sample | what it shows |
|---|---|
| `samples/DotFly.Sample.SugarExperiment` | the Shiu et al. sugar → MN9 experiment (30 trials + silencing) against the published rates |
| `samples/DotFly.Sample.TrainReadout` | recording output frames and training a readout three ways (C# logistic regression, ML.NET, ONNX) |
| `samples/DotFly.Sample.Godot` | the 2D closed loop: retina → LC4 → DNp04/GF → a fixed decoder steers a fly away from a looming target |
| `samples/DotFly.Sample.Godot3D` | the room: eyes, antennae, wind, sugar, feeding, panels, cameras, several flies |
