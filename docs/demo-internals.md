# Inside the room demo

How `samples/DotFly.Sample.Godot3D` is wired, with the code that matters. Read this to change a
rule, add a readout, or feed the brain a new sense.

## The frame loop

`FlyBody3D._PhysicsProcess` runs 60 times a second on Godot's physics thread while the network
runs on its own thread through `RealtimeDriver`:

```csharp
// ---- sense ------------------------------------------------------------
ReadEyes();                                        // async GPU readback → two 32×32 L8 images
if (_eyeFresh[0] && _eyeFresh[1])
    _retina.Process(_eyeL8[0], _eyeL8[1]);         // → LC4, T4a–d rates per eye
float odorL = _odor.Concentration(leftAntenna);    // the world's odor field at each antenna
float odorR = _odor.Concentration(rightAntenna);
Vector3 from = -_odor.Wind;                        // the side the air comes from → JO-C/E rates
float windL = Math.Clamp(0.35f + from.Dot(-right), 0f, 1f), windR = Math.Clamp(0.35f - from.Dot(-right), 0f, 1f);
_brain.Encode(_retina, odorL, odorR, tarsalContact: onSugar, labellarContact: onSugar && StateTime > 0.5f, windL, windR);

// ---- read out ---------------------------------------------------------
_brain.ReadOut();                                  // latest published frame → mean rate per readout group

// ---- decide -----------------------------------------------------------
(float yaw, float climb, float speed) = _behaviour.Step(_brain.Rates, patchUnder, altitude, dt);

// ---- act --------------------------------------------------------------
_yaw -= Mathf.DegToRad(yaw) * dt;  Velocity = forward * speed + Vector3.Up * climb;  MoveAndSlide();
```

Nothing else touches the network. The 20 ms publish period of the output port and the 60 Hz frame
are the only coupling between the two clocks; the HUD's *RTF* and *behind* fields report how well
the driver keeps up.

## Encoding: what the sensory neurons are told

`FlyBrain3D.Encode` writes one Poisson rate per input population; the populations are queried by
type from the checkpoint at startup:

```csharp
foreach (string type in new[] { "LC4", "T4a", "T4b", "T4c", "T4d", "ORN_DM1" })
{
    AddInput(_brain.Query(type: type, side: Side.Left,  name: type + "_L"));
    AddInput(_brain.Query(type: type, side: Side.Right, name: type + "_R"));
}
AddInput(_brain.Query(type: "claw_tpGRN"));                                   // tarsal sugar receptors
AddInput(_brain.Query(type: "LB3*"));                                         // labellar sugar receptors
AddInput((_brain.Query(type: "JO-C*", side: Side.Left) | _brain.Query(type: "JO-E*", side: Side.Left)).Rename("JO-CE_L"));
AddInput((_brain.Query(type: "JO-C*", side: Side.Right) | _brain.Query(type: "JO-E*", side: Side.Right)).Rename("JO-CE_R"));
```

```csharp
public void Encode(Retina retina, float odorLeft, float odorRight, bool tarsal, bool labellar, float windLeft, float windRight)
{
    for (int c = 0; c < 5; c++)                                    // LC4, T4a, T4b, T4c, T4d
        for (int eye = 0; eye < 2; eye++)
            InputRates[2 * c + eye] = Vision ? retina.Rates[eye, c] : 0f;
    InputRates[10] = Chemosensation ? Mathf.Clamp(odorLeft  * 120f, 0, 150) : 0;   // ORN_DM1_L, Hz
    InputRates[11] = Chemosensation ? Mathf.Clamp(odorRight * 120f, 0, 150) : 0;
    InputRates[12] = Chemosensation && tarsal   ? 100f : 0f;                        // claw_tpGRN
    InputRates[13] = Chemosensation && labellar ? 100f : 0f;                        // LB3
    InputRates[14] = Chemosensation ? Mathf.Clamp(windLeft  * 60f, 0, 60) : 0;      // JO-CE_L
    InputRates[15] = Chemosensation ? Mathf.Clamp(windRight * 60f, 0, 60) : 0;
    for (int p = 0; p < _inputs.Count; p++) _inputs[p].Fill(InputRates[p]);        // thread-safe port writes
}
```

The retina (`Retina.cs`) is the largest encoder: per eye it takes the fraction of pixels darker
than 0.55 × the eye's mean luminance and its growth (→ LC4), and the best 1-D shift of the row and
column intensity profiles between frames (→ T4a/b horizontal outward/inward, T4c/d vertical).
Everything here is a hand choice and is labelled so.

## Reading out: what the brain answers

Every readout group is a `NeuronSet` found in the checkpoint; all of them share one `OutputPort`
(30 ms rate window, published every 20 ms), and `ReadOut` averages the frame per group:

```csharp
_groups =
[
    _brain.Query(type: "DNp04", side: Side.Left,  name: "DNp04_L"),  _brain.Query(type: "DNp04", side: Side.Right, name: "DNp04_R"),
    _brain.Query(type: "DNp01", name: "DNp01"),                       // Giant Fiber
    Hs(Side.Left), Hs(Side.Right),                                    // HSE | HSN | HSS per side
    _brain.Query(type: "VS", side: Side.Left, name: "VS_L"),  _brain.Query(type: "VS", side: Side.Right, name: "VS_R"),
    _brain.Query(type: "H2", side: Side.Left, name: "H2_L"),  _brain.Query(type: "H2", side: Side.Right, name: "H2_R"),
    _brain.Query(type: "DM1_lPN", name: "DM1_lPN"),
    _brain.Query(type: "MN9", name: "MN9"),
    _brain.ByBodyIds("wind_L", [11960UL]), _brain.ByBodyIds("wind_R", [11702UL]),   // AMMC012, one per antenna (measured)
];
_out = _sim.Output(all, OutputKind.Rate(30.Ms()), publishEvery: 20.Ms());

public void ReadOut()
{
    OutputFrame f = _out.Snapshot;
    Array.Clear(Rates);
    for (int p = 0; p < _groupOfPosition.Length; p++) Rates[_groupOfPosition[p]] += f[p];
    for (int g = 0; g < Rates.Length; g++) Rates[g] /= Math.Max(1, _groupCount[g]);
}
```

`Rates[]` is in `GroupNames` order; `Behaviour.Step` reads it by index.

## Deciding: the decoder

`Behaviour.cs` is a small state machine (Flying, Landing, Landed, Feeding, TakeOff) over the
readouts; every constant comes from `decoders/dn_fixed_3d.json`, whose `origin` is `Fixed`. The
core of the flying state, condensed:

```csharp
float escape    = 1.2f * (dnp04L - dnp04R);                                  // looming side → turn away
float optomotor = 0.2f * ((hsR + h2L) - (hsL + h2R)) - 0.3f * (hsR - hsL);   // rotation → counter-turn; wall → centre
float speed     = (120f + 1.0f * gf) * bodyLength;                           // BL/s; the GF bursts

OdorDetected = pnSmoothed > 110f;                                            // from the measured PN response curve
if (OdorDetected)
{
    yaw = escape + optomotor - 0.6f * (windL - windR);                      // surge upwind
    if (fallingFor > 0.25f) { tumble(150°/s, 0.65 s); flipVertical50%(); }   // klinotaxis on the trend
    if (pn > 270f && patchUnder && altitude < 4 * bodyLength) Transition(Landing);
}
else if (OdorMemory)   // lost within the last 8 s
{
    yaw += castSign * 150f;  speed = 35f * bodyLength;                       // U-turn, then widening casts
}
```

Landed → `Feeding` when the 0.3 s-smoothed MN9 exceeds 15 Hz; `TakeOff` when it stops or after 8 s.
The trace panel prints these rules next to the traces they use.

## Where the panels get their data

- **brain map**: `FlyBrain3D` subscribes to `sim.OnSpikes` and keeps a per-neuron `Activity[]`
  that decays with τ = 0.15 s; `BrainMapView` projects every neuron's soma (from
  `Brain.SomaPositions`) once, then at 15 Hz splats the activity into a 400-wide heat texture.
- **traces**: `TraceView` samples `Rates[]` at 30 Hz into 10 s rings.
- **eyes**: the two L8 images the retina received, uploaded as textures — not the colour render.
- **room map**: the odor cones' centrelines from `Odor.Centreline`, the trail from
  `FlyBody3D`'s recorded path.

## Recipes

**Add a readout.** Find the cells (`dotfly inspect --upstream-of`, `dotfly run --stimulate`), add a
`NeuronSet` to `_groups` and a name to `GroupNames`, a colour to `BrainMapView.GroupColors`, an
index to the right `TraceView` panel, then use `rates[i]` in `Behaviour.Step`. Keep the order in
sync — `Behaviour` unpacks by position.

**Change a rule.** Edit `decoders/dn_fixed_3d.json` (no rebuild) for gains and thresholds; edit
`Behaviour.cs` for structure. Update the caption strings in `TraceView.cs` so the screen still
tells the truth.

**Add a sense.** A new `AddInput(...)` in `FlyBrain3D._Ready`, a rate in `Encode`, a name in
`InputNames`. Say on the HUD what the encoder does.

**Try the causal switches from code.** `sim.RecurrentTransmission = false` (the `R` key) turns the
brain into a pass-through; `sim.Silence(set)` (the `S` key) keeps a readout spiking but stops it
transmitting — the cleanest way to show what a circuit contributes.
