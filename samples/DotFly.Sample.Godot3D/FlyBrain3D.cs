using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using DotFly.Core.Graph;
using Godot;
using Side = DotFly.Core.Graph.Side;

namespace DotFly.Sample.Godot3D;

/// <summary>
/// The connectome in the 3D scene.
/// <para><b>Inputs</b> (all Poisson-to-v, one rate per population): per eye LC4, T4a–d (vision);
/// per antenna ORN_DM1 (the vinegar/attraction olfactory channel); claw_tpGRN (tarsal sugar
/// contact) and LB3 (labellar sugar contact).</para>
/// <para><b>Readouts</b> (30 ms rates, 20 ms frames), all measured circuits of this checkpoint:
/// LC4 → DNp04 L/R + DNp01 (escape); T4a → HS L/R, T4c → VS L/R, T4b → H2 L/R (optic flow; HS and
/// H2 together tell rotation from translation); ORN_DM1 → DM1_lPN (odor
/// detected — odor <em>direction</em> is not resolved by the wiring, both PNs respond alike);
/// GRNs → MN9 (proboscis motor: feed); JO-C/E (wind on the antennae) → AMMC012 / WED080 per side
/// (wind direction: lateralized, measured) — the surge-upwind cue.</para>
/// <para>Also keeps a per-neuron activity trace (real spikes, decayed) for the brain map.</para>
/// </summary>
public partial class FlyBrain3D : Node
{
    /// <summary>Checkpoint path; relative paths are resolved against the project directory.</summary>
    [Export]
    public string CheckpointPath { get; set; } = "../../.data/malecns-v1.0/malecns-v1.0-superclass.dfb";

    /// <summary>Simulation threads (0 = one per physical core).</summary>
    [Export]
    public int Threads { get; set; }

    /// <summary>Neural seconds per wall second.</summary>
    [Export]
    public double Ratio { get; set; } = 1.0;

    /// <summary>Random seed of the stochastic inputs (each fly gets its own).</summary>
    [Export]
    public int Seed { get; set; } = 1;

    /// <summary>Worker slots taken by the other flies' simulations (see <c>SimulationOptions.ThreadPinOffset</c>).</summary>
    [Export]
    public int PinOffset { get; set; }

    /// <summary>Worker slots of all flies' simulations.</summary>
    [Export]
    public int PinTotal { get; set; }

    /// <summary>Whether this brain reacts to the R/E/V/O/S/Space keys (only the primary fly's does).</summary>
    [Export]
    public bool HandlesKeys { get; set; } = true;

    /// <summary>Multiplier on the model's unitary synaptic weight; 1.0 is the published model. Shown on the HUD.</summary>
    [Export]
    public double SynapticGain { get; set; } = 1.0;

    /// <summary>
    /// Stability control: populations whose outgoing synapses are silenced at start, as a
    /// ';'-separated list of <c>class:NAME</c> / <c>type:NAME</c> specs (globs allowed). Default: the
    /// Kenyon cells and the antennal-lobe local neurons (4,225 of 166,700 neurons). The uniform
    /// LIF model lacks the sparsening these populations have in vivo (APL inhibition, high KC
    /// thresholds, LN gain control); without this control a few hertz of odor — or of one-sided
    /// LC4 input — ignites a self-sustaining attractor (~450–600 k spikes/s that never stops, PN
    /// and MN9 saturated with no input). With it every readout used here is graded and returns to
    /// baseline within a second of the input stopping (measured with `dotfly run --stim-seconds`).
    /// Empty = no control (watch the runaway).
    /// </summary>
    [Export]
    public string StabilityControl { get; set; } = "class:Kenyon_Cell;type:lLN*";

    /// <summary>Readout groups, in the order of <see cref="Rates"/>.</summary>
    public static readonly string[] GroupNames = ["DNp04_L", "DNp04_R", "DNp01 (GF)", "HS_L", "HS_R", "VS_L", "VS_R", "H2_L", "H2_R", "DM1_lPN", "MN9", "wind_L", "wind_R"];

    /// <summary>Input populations, in the order of <see cref="InputRates"/>.</summary>
    public static readonly string[] InputNames = ["LC4_L", "LC4_R", "T4a_L", "T4a_R", "T4b_L", "T4b_R", "T4c_L", "T4c_R", "T4d_L", "T4d_R", "ORN_DM1_L", "ORN_DM1_R", "claw_tpGRN", "LB3", "JO-CE_L", "JO-CE_R"];

    private Brain? _brain;
    private Simulation? _sim;
    private RealtimeDriver? _driver;
    private readonly List<InputPort> _inputs = [];
    private readonly List<NeuronSet> _inputSets = [];
    private OutputPort? _out;
    private int[] _groupOfPosition = [];
    private readonly int[] _groupCount = new int[GroupNames.Length];
    private NeuronSet[] _groups = [];
    private Decoder _decoder = new();
    private bool _silenced;
    private string _status = "loading…";
    private float[] _activity = [];
    private NeuronSet _control = NeuronSet.Empty;

    /// <summary>The silenced stability-control population.</summary>
    public NeuronSet Control => _control;

    /// <summary>Resolves <c>class:NAME</c> / <c>type:NAME</c> specs (globs, optional <c>@L</c>/<c>@R</c>), ';' = union.</summary>
    public static NeuronSet ResolveSpec(Brain brain, string spec)
    {
        NeuronSet union = NeuronSet.Empty;
        foreach (string part in spec.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            int colon = part.IndexOf(':');
            string field = colon < 0 ? "type" : part[..colon].ToLowerInvariant();
            string value = colon < 0 ? part : part[(colon + 1)..];
            Side? side = null;
            int at = value.LastIndexOf('@');
            if (at >= 0)
            {
                side = value[(at + 1)..].ToUpperInvariant() == "L" ? Side.Left : Side.Right;
                value = value[..at];
            }

            union |= field == "class" ? brain.Query(cls: value, side: side) : brain.Query(type: value, side: side);
        }

        return union;
    }

    /// <summary>Latest readout rates (Hz) per group.</summary>
    public float[] Rates { get; } = new float[GroupNames.Length];

    /// <summary>Latest input rates (Hz) per population.</summary>
    public float[] InputRates { get; } = new float[InputNames.Length];

    /// <summary>The decoder gains.</summary>
    public Decoder Gains => _decoder;

    /// <summary>Whether the network is running.</summary>
    public bool IsRunning => _driver is not null;

    /// <summary>Whether visual channels are applied (V key).</summary>
    public bool Vision { get; set; } = true;

    /// <summary>Whether olfactory/gustatory channels are applied (O key).</summary>
    public bool Chemosensation { get; set; } = true;

    /// <summary>The brain (null until loaded).</summary>
    public Brain? Brain => _brain;

    /// <summary>The simulation (null until loaded).</summary>
    public Simulation? Simulation => _sim;

    /// <summary>The driver (null until loaded).</summary>
    public RealtimeDriver? Driver => _driver;

    /// <summary>Per-neuron recent activity (spike counts with exponential decay), written on the simulation thread.</summary>
    public float[] Activity => _activity;

    /// <summary>The readout groups (for the brain map markers).</summary>
    public IReadOnlyList<NeuronSet> Groups => _groups;

    /// <summary>The input populations (for the brain map markers).</summary>
    public IReadOnlyList<NeuronSet> InputSets => _inputSets;

    /// <summary>Whether DNp04 + HS are silenced (S key).</summary>
    public bool Silenced => _silenced;

    /// <summary>Status line.</summary>
    public string Status => _status;

    public override void _Ready()
    {
        string projectDir = ProjectSettings.GlobalizePath("res://");
        _decoder = JsonSerializer.Deserialize(File.ReadAllText(Path.Combine(projectDir, "decoders", "dn_fixed_3d.json")), Decoder3DJsonContext.Default.Decoder) ?? new Decoder();

        string path = Path.IsPathRooted(CheckpointPath) ? CheckpointPath : Path.GetFullPath(Path.Combine(projectDir, CheckpointPath));
        if (!File.Exists(path))
        {
            _status = $"checkpoint not found: {path}\nBuild it with `dotfly build malecns` (see README).";
            GD.PushError(_status);
            return;
        }

        _brain = Brain.Open(path);
        int flies = Arena.FliesFromArgs();
        if (flies > 1 && Threads == 0)
        {
            // Several flies share the machine: an even split of the physical cores, pinned apart.
            Threads = Math.Max(1, DotFly.Cpu.Fast.CpuBackend.DefaultThreads() / flies);
            PinTotal = flies * Threads;
        }

        _sim = _brain.CreateSimulation(new SimulationOptions { Threads = Threads, Seed = (ulong)Seed, SynapticGain = SynapticGain, ThreadPinOffset = PinOffset, ThreadPinTotal = PinTotal });
        _activity = new float[_brain.NeuronCount];
        if (!string.IsNullOrWhiteSpace(StabilityControl))
        {
            _control = ResolveSpec(_brain, StabilityControl).Rename("stability control");
            _sim.Silence(_control);
        }

        foreach (string type in new[] { "LC4", "T4a", "T4b", "T4c", "T4d", "ORN_DM1" })
        {
            AddInput(_brain.Query(type: type, side: Side.Left, name: type + "_L"));
            AddInput(_brain.Query(type: type, side: Side.Right, name: type + "_R"));
        }

        AddInput(_brain.Query(type: "claw_tpGRN", name: "claw_tpGRN"));
        AddInput(_brain.Query(type: "LB3*", name: "LB3"));
        // Wind: the antennae's Johnston's organ, the wind-direction-sensitive JO-C/E neurons per side.
        AddInput((_brain.Query(type: "JO-C*", side: Side.Left) | _brain.Query(type: "JO-E*", side: Side.Left)).Rename("JO-CE_L"));
        AddInput((_brain.Query(type: "JO-C*", side: Side.Right) | _brain.Query(type: "JO-E*", side: Side.Right)).Rename("JO-CE_R"));

        _groups =
        [
            _brain.Query(type: "DNp04", side: Side.Left, name: "DNp04_L"),
            _brain.Query(type: "DNp04", side: Side.Right, name: "DNp04_R"),
            _brain.Query(type: "DNp01", name: "DNp01"),
            Hs(Side.Left), Hs(Side.Right),
            _brain.Query(type: "VS", side: Side.Left, name: "VS_L"),
            _brain.Query(type: "VS", side: Side.Right, name: "VS_R"),
            _brain.Query(type: "H2", side: Side.Left, name: "H2_L"),
            _brain.Query(type: "H2", side: Side.Right, name: "H2_R"),
            _brain.Query(type: "DM1_lPN", name: "DM1_lPN"),
            _brain.Query(type: "MN9", name: "MN9"),
            // Wind direction: AMMC/wedge interneurons that respond to one antenna's JO-C/E only
            // (measured: left JO → AMMC012 11960 130 Hz, WED080 13072 134 Hz, their partners 0–2 Hz).
            // AMMC012 only: its two sides answer alike (130 vs 112 Hz); adding WED080 (134 vs 65 Hz)
            // made the readout left-heavy and the fly circled left.
            _brain.ByBodyIds("wind_L", [11960UL]),
            _brain.ByBodyIds("wind_R", [11702UL]),
        ];
        NeuronSet all = NeuronSet.Empty;
        for (int g = 0; g < _groups.Length; g++)
        {
            _groupCount[g] = _groups[g].Count;
            all |= _groups[g];
        }

        _out = _sim.Output(all, OutputKind.Rate(30.Ms()), publishEvery: 20.Ms());
        _groupOfPosition = new int[all.Count];
        for (int g = 0; g < _groups.Length; g++)
        {
            foreach (int i in _groups[g].Indices)
            {
                _groupOfPosition[IndexOf(all, i)] = g;
            }
        }

        float[] activity = _activity;
        _sim.OnSpikes += (_, ids) =>
        {
            foreach (int i in ids)
            {
                activity[i] += 1f;
            }
        };

        _driver = new RealtimeDriver(_sim, new RealtimeOptions { Ratio = Ratio, MaxStepsPerTick = 500 });
        _driver.Start();
        GD.Print($"dotFly: opened {path}");
        int totalIn = 0;
        foreach (NeuronSet s in _inputSets)
        {
            totalIn += s.Count;
        }

        _status = $"{_brain.Provenance.Name}: {_brain.NeuronCount:N0} neurons, {_brain.EdgeCount:N0} edges; {totalIn} input neurons in {_inputSets.Count} populations; {all.Count} readout neurons in {_groups.Length} groups";
    }

    private void AddInput(NeuronSet set)
    {
        _inputSets.Add(set);
        _inputs.Add(_sim!.Input(set, InputKind.PoissonToV));
    }

    private NeuronSet Hs(Side side) =>
        (_brain!.Query(type: "HSE", side: side) | _brain.Query(type: "HSN", side: side) | _brain.Query(type: "HSS", side: side)).Rename($"HS_{(side == Side.Left ? "L" : "R")}");

    private static int IndexOf(NeuronSet set, int neuron)
    {
        int lo = 0, hi = set.Count - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >> 1;
            if (set[mid] == neuron)
            {
                return mid;
            }

            if (set[mid] < neuron)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid - 1;
            }
        }

        return -1;
    }

    /// <summary>Writes all input rates: visual channels from the retina, chemosensory from the world.</summary>
    public void Encode(Retina retina, float odorLeft, float odorRight, bool tarsalContact, bool labellarContact, float windLeft = 0f, float windRight = 0f)
    {
        if (_sim is null)
        {
            return;
        }

        // Visual: population index = 2 * channel + eye.
        for (int c = 0; c < 5; c++)
        {
            for (int eye = 0; eye < 2; eye++)
            {
                InputRates[2 * c + eye] = Vision ? retina.Rates[eye, c] : 0f;
            }
        }

        InputRates[10] = Chemosensation ? Mathf.Clamp(odorLeft * _decoder.OdorHzPerUnit, 0, 150) : 0;
        InputRates[11] = Chemosensation ? Mathf.Clamp(odorRight * _decoder.OdorHzPerUnit, 0, 150) : 0;
        InputRates[12] = Chemosensation && tarsalContact ? 100f : 0f;
        InputRates[13] = Chemosensation && labellarContact ? 100f : 0f;
        InputRates[14] = Chemosensation ? Mathf.Clamp(windLeft * _decoder.WindHz, 0, _decoder.WindHz) : 0;    // antennal deflection → JO-C/E
        InputRates[15] = Chemosensation ? Mathf.Clamp(windRight * _decoder.WindHz, 0, _decoder.WindHz) : 0;
        for (int p = 0; p < _inputs.Count; p++)
        {
            _inputs[p].Fill(InputRates[p]);
        }
    }

    /// <summary>Reads the latest frame and averages per group into <see cref="Rates"/>.</summary>
    public void ReadOut()
    {
        if (_out is null)
        {
            return;
        }

        OutputFrame f = _out.Snapshot;
        Array.Clear(Rates);
        for (int p = 0; p < _groupOfPosition.Length; p++)
        {
            Rates[_groupOfPosition[p]] += f[p];
        }

        for (int g = 0; g < Rates.Length; g++)
        {
            Rates[g] = _groupCount[g] > 0 ? Rates[g] / _groupCount[g] : 0f;
        }
    }

    /// <summary>Decays the activity trace (call once per rendered frame).</summary>
    public void DecayActivity(float factor)
    {
        float[] a = _activity;
        for (int i = 0; i < a.Length; i++)
        {
            a[i] *= factor;
        }
    }

    /// <summary>Compact clock line for the HUD.</summary>
    public string ClockLine()
    {
        if (_sim is null || _driver is null)
        {
            return _status;
        }

        SimulationClock c = _sim.Clock;
        return $"neural {c.NeuralTime.TotalSeconds,7:F2} s   busy {c.WallTime.TotalSeconds,6:F2} s   RTF {c.RecentRealTimeFactor,5:F2}   behind {_driver.BehindBy.TotalMilliseconds,4:F0} ms   spikes/s {c.RecentSpikesPerSecond,9:N0}   gain {SynapticGain:F2} ({(Math.Abs(SynapticGain - 1) < 1e-6 ? "model" : "calibrated")})   control: {(_control.Count == 0 ? "none" : $"{_control.Count} neurons silenced (KC + AL LNs)")}";
    }

    /// <summary>Switch state line for the HUD.</summary>
    public string SwitchLine()
    {
        if (_sim is null || _driver is null)
        {
            return string.Empty;
        }

        return $"[R] recurrent {(_sim.RecurrentTransmission ? "on" : "OFF")}   [E] external input {(_sim.ExternalInput ? "on" : "OFF")}   [V] vision {(Vision ? "on" : "OFF")}   [O] smell/taste {(Chemosensation ? "on" : "OFF")}   [S] DNp04+HS silenced {(_silenced ? "YES" : "no")}   [Space] {(_driver.Running ? "pause" : "resume")}";
    }

    public override void _UnhandledKeyInput(InputEvent @event)
    {
        if (!HandlesKeys || _sim is null || _driver is null || @event is not InputEventKey { Pressed: true } key)
        {
            return;
        }

        switch (key.Keycode)
        {
            case Key.R:
                _sim.RecurrentTransmission = !_sim.RecurrentTransmission;
                break;
            case Key.E:
                _sim.ExternalInput = !_sim.ExternalInput;
                break;
            case Key.V:
                Vision = !Vision;
                break;
            case Key.O:
                Chemosensation = !Chemosensation;
                break;
            case Key.S:
                _silenced = !_silenced;
                bool s = _silenced;
                NeuronSet set = _brain!.Query(type: "DNp04") | Hs(Side.Left) | Hs(Side.Right);
                _sim.ScheduleAt(_sim.Step, sim => { if (s) sim.Silence(set); else sim.Unsilence(set); });
                break;
            case Key.Space:
                if (_driver.Running)
                {
                    _driver.Pause();
                }
                else
                {
                    _driver.Start();
                }

                break;
        }
    }

    public override void _ExitTree()
    {
        _driver?.Dispose();
        _sim?.Dispose();
        _brain?.Dispose();
    }

    /// <summary>Decoder gains loaded from <c>decoders/dn_fixed_3d.json</c>.</summary>
    public sealed class Decoder
    {
        public string Origin { get; set; } = "Fixed";
        public float EscapeYawGainDegPerSecPerHz { get; set; } = 1.2f;
        public float OptomotorYawGainDegPerSecPerHz { get; set; } = 0.2f;
        public float CenteringYawGainDegPerSecPerHz { get; set; } = 0.3f;
        // Speeds are in body lengths per second, so the fly flies like a fly at any size: a real
        // Drosophila cruises at 100–170 BL/s (0.3–0.5 m/s at 3 mm).
        public float OptomotorClimbGainBlPerSecPerHz { get; set; } = 0.07f;
        public float RoamClimbBlPerSec { get; set; } = 8f;
        public float VerticalCastBlPerSec { get; set; } = 12f;
        public float IdleSpeedBlPerSec { get; set; } = 120f;
        public float GfSpeedGainBlPerSecPerHz { get; set; } = 1.0f;
        public float OdorHzPerUnit { get; set; } = 120f;
        public float OdorDetectHz { get; set; } = 110f;
        public float OdorNearHz { get; set; } = 270f;
        public float CastYawDegPerSec { get; set; } = 150f;
        public float OdorMemorySeconds { get; set; } = 8f;
        public float CastSpeedBlPerSec { get; set; } = 35f;
        public float TumbleSeconds { get; set; } = 0.65f;
        public float WindHz { get; set; } = 60f;
        public float SurgeYawGainDegPerSecPerHz { get; set; } = 0.6f;
        public float TrackingSpeedBlPerSec { get; set; } = 80f;
        public float FeedMn9Hz { get; set; } = 15f;
    }
}

[System.Text.Json.Serialization.JsonSourceGenerationOptions(PropertyNameCaseInsensitive = true)]
[System.Text.Json.Serialization.JsonSerializable(typeof(FlyBrain3D.Decoder))]
internal sealed partial class Decoder3DJsonContext : System.Text.Json.Serialization.JsonSerializerContext
{
}
