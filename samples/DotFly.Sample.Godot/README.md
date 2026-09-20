# dotFly Godot sample

A Godot 4 (.NET) scene in which a 2D "fly" is steered by descending-neuron rates of the MaleCNS
connectome running in real time on the dotFly engine.

- `FlyBrain.cs` — opens the checkpoint, creates input ports over the LC4 looming-sensitive visual
  projection neurons (left/right) and an output port over the escape descending neurons LC4 drives
  in this checkpoint — DNp04 (left/right) and DNp01, the Giant Fiber — and runs a `RealtimeDriver`.
- `FlyBody.cs` — every frame: encodes the target's bearing into LC4 rates (the **encoder**, hand-written),
  reads the latest output frame, applies the **fixed** decoder gains from `decoders/dn_fixed.json`
  (yaw ∝ DNp04_L − DNp04_R, i.e. turn away from the looming side; speed ∝ DNp01) and moves the sprite.
- Keys: `R` recurrent transmission on/off, `E` external input on/off, `S` silence DNp04, `Space` pause.
  These are the causal controls: watch what the steering does when the wiring is cut versus when the
  input is cut — the decoder is unchanged either way.

What this is: a programmable model of neural circuitry driving a sprite through engineering choices
that are all visible in this folder. What it is not: a fly.

## Run

1. Build the MaleCNS checkpoint (see the repository README): `.data/malecns-v1.0/malecns-v1.0-superclass.dfb`.
2. Until .NET 11 ships, tell the .NET host to accept the RC runtime: set `DOTNET_ROLL_FORWARD_TO_PRERELEASE=1`
   in the environment Godot starts from (otherwise the game assembly fails to load with
   "System.Runtime, Version=11.0.0.0 not found").
3. Open this folder in Godot 4.7 (.NET) and press Play, or headless:
   `Godot_v4.7.1-stable_mono_win64_console.exe --headless --path . --quit-after 300`
   (run the real executable — the winget `godot` shim cannot find Godot's own .NET assemblies).
   The `CheckpointPath` export on the `FlyBrain` node defaults to the repository's `.data` folder.

The project file pins the dotFly library references to the Release configuration; Godot builds the
game in Debug, and an unoptimised kernel is ~25× slower.
