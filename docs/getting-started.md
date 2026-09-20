# Getting started

## Build

Requires the .NET 11 SDK (RC1 or later; pinned in `global.json`).

```powershell
dotnet build -c Release
dotnet test  -c Release                                   # unit tests need no data
dotnet run --project src/DotFly.Cli -c Release -- info
```

## Data and checkpoints

Connectome releases are downloaded into the git-ignored `.data/` folder (never committed; CC BY 4.0)
and converted once into memory-mapped `.dfb` checkpoints:

```powershell
# MaleCNS v1.0 flat connectome (https://male-cns.janelia.org/download/), ~1.6 GB:
#   .data/malecns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather
#   .data/malecns-v1.0/body-neurotransmitters-male-cns-v1.0.feather
#   .data/malecns-v1.0/connectome-weights-male-cns-v1.0-minconf-0.5.feather          (or -traced-only)
dotnet run --project src/DotFly.Cli -c Release -- build malecns .data/malecns-v1.0 --filter superclass
dotnet run --project src/DotFly.Cli -c Release -- build malecns .data/malecns-v1.0 --filter traced

# FlyWire v630 as shipped with the Shiu et al. model (github.com/philshiu/Drosophila_brain_model):
#   .data/shiu2024/2023_03_23_completeness_630_final.csv, 2023_03_23_connectivity_630_final.parquet
dotnet run --project src/DotFly.Cli -c Release -- build flywire .data/shiu2024/2023_03_23_completeness_630_final.csv .data/shiu2024/2023_03_23_connectivity_630_final.parquet -m 630
```

`--filter superclass` keeps bodies with a `superclass` annotation (166,700 neurons; the graph the
demos use); `--filter traced` keeps `status == Traced` (165,122; the official `traced-only` export).
Edges whose presynaptic transmitter is unclear or histamine are dropped as no-ops; the structural
counts stay in the checkpoint's provenance.

## Use the engine

```csharp
using DotFly;

using Brain brain = Brain.Open(".data/shiu2024/flywire-v630.dfb");
using Simulation sim = brain.CreateSimulation(new SimulationOptions { Seed = 1 });

NeuronSet sugar = brain.ByBodyIds("sugar_GRN_R", sugarIds);          // exact 64-bit IDs
sim.Input(sugar, InputKind.PoissonToV).Fill(150f);                   // Poisson drive, 150 Hz each
OutputPort mn9 = sim.Output(brain.ByBodyId(720575940660219265), OutputKind.Rate(50.Ms()));
sim.OnSpikes += (step, ReadOnlySpan<int> ids) => { /* raster, counters… */ };

sim.Run(1.Seconds());                                                 // neural time
Console.WriteLine($"MN9 {mn9.Snapshot.Mean():F1} Hz at {sim.Clock.RealTimeFactor:F1}× real time");
```

For a game loop, `new RealtimeDriver(sim).Start()` runs the network on its own thread; the game
thread writes `InputPort`s and reads `OutputPort.Snapshot` frames (lock-free), and a readout
adapter (`LinearAdapter`, `ThresholdAdapter`, or an ONNX/ML.NET model) turns a frame's values into
actions — labelled *fixed*, *calibrated* or *trained*, because the graph never learns.
`sim.Record("run.dfs")` captures spikes, input writes and output frames; `Recording.Read(...)
.Replay(sim)` reproduces the run. `samples/DotFly.Sample.SugarExperiment` is the worked example.

## Run the room demo

1. Build the MaleCNS checkpoint as above (`.data/malecns-v1.0/malecns-v1.0-superclass.dfb`).
2. Install Godot 4.7 (.NET). Until .NET 11 GA, set `DOTNET_ROLL_FORWARD_TO_PRERELEASE=1`.
3. Open `samples/DotFly.Sample.Godot3D` in the editor and press Play, or run the real executable
   (not the winget shim): `Godot_v4.7.1-stable_mono_win64.exe --path samples/DotFly.Sample.Godot3D`.

Keys: `C` camera mode, `R` recurrent transmission, `E` external input, `V` vision, `O` smell/taste,
`S` silence DNp04 + HS, `Space` pause. `-- --flies 4` puts four flies in the room, each a full
independent simulation. Details, environment variables and the performance table are in
[The room demo](demo.md).

## Samples

| sample | what it shows |
|---|---|
| `samples/DotFly.Sample.SugarExperiment` | the Shiu et al. sugar → MN9 experiment (30 trials + silencing) against the published rates |
| `samples/DotFly.Sample.TrainReadout` | recording output frames and training a readout three ways (C# logistic regression, ML.NET, ONNX) |
| `samples/DotFly.Sample.Godot` | the 2D closed loop: retina → LC4 → DNp04/GF → a fixed decoder steers a fly away from a looming target |
| `samples/DotFly.Sample.Godot3D` | the room: eyes, antennae, wind, sugar, feeding, panels, cameras, several flies |
