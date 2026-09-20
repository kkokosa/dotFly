using System;
using System.IO;
using System.Text.Json;
using DotFly.Core.Graph;
using Side = DotFly.Core.Graph.Side;
using Godot;

namespace DotFly.Sample.Godot;

/// <summary>
/// Runs a MaleCNS checkpoint on a <see cref="RealtimeDriver"/> and exposes three things to the
/// scene: an input port over the LC4 looming-sensitive visual projection neurons (left/right), an
/// output port over the DNp04 (left/right) and DNp01 (Giant Fiber) escape descending neurons that
/// LC4 drives, and the causal switches. The decoder is an escape reflex: turn away from the side
/// whose DNp04 fires, speed up with the Giant Fiber. The retina encoder
/// (<see cref="Encode"/>) and the decoder gains (<c>decoders/dn_fixed.json</c>) are hand-written
/// engineering, labelled as such — the graph is the only measured part.
/// </summary>
public partial class FlyBrain : Node
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

    /// <summary>Peak LC4 drive when the target is dead ahead on that side, in Hz.</summary>
    [Export]
    public float MaxRetinaHz { get; set; } = 120f;

    private Brain? _brain;
    private Simulation? _sim;
    private RealtimeDriver? _driver;
    private InputPort? _eyesLeft;
    private InputPort? _eyesRight;
    private OutputPort? _dn;
    private NeuronSet _dnp04L = NeuronSet.Empty;
    private NeuronSet _dnp04R = NeuronSet.Empty;
    private NeuronSet _dnp01 = NeuronSet.Empty;
    private float[] _left = [];
    private float[] _right = [];
    private Decoder _decoder = new();
    private bool _silenced;
    private string _status = "loading…";

    /// <summary>Whether the network is running.</summary>
    public bool IsRunning => _driver is not null;

    /// <summary>The decoder gains (fixed).</summary>
    public Decoder Gains => _decoder;

    /// <summary>Last decoded actions: yaw rate (deg/s) and forward speed (px/s).</summary>
    public (float YawDegPerSec, float SpeedPxPerSec) Actions { get; private set; }

    /// <summary>Latest readout rates: DNp04_L, DNp04_R, DNp01 (Hz, mean over the set).</summary>
    public (float DNp04L, float DNp04R, float DNp01) Rates { get; private set; }

    /// <summary>Status line for the HUD.</summary>
    public string Status => _status;

    public override void _Ready()
    {
        string projectDir = ProjectSettings.GlobalizePath("res://");
        string decoderPath = Path.Combine(projectDir, "decoders", "dn_fixed.json");
        _decoder = JsonSerializer.Deserialize(File.ReadAllText(decoderPath), DecoderJsonContext.Default.Decoder) ?? new Decoder();

        string path = Path.IsPathRooted(CheckpointPath) ? CheckpointPath : Path.GetFullPath(Path.Combine(projectDir, CheckpointPath));
        if (!File.Exists(path))
        {
            _status = $"checkpoint not found: {path}\nBuild it with `dotfly build malecns` (see README).";
            GD.PushError(_status);
            return;
        }

        _brain = Brain.Open(path);
        _sim = _brain.CreateSimulation(new SimulationOptions { Threads = Threads, Seed = 1 });

        NeuronSet lc4L = _brain.Query(type: "LC4", side: Side.Left, name: "LC4_L");
        NeuronSet lc4R = _brain.Query(type: "LC4", side: Side.Right, name: "LC4_R");
        _dnp04L = _brain.Query(type: "DNp04", side: Side.Left, name: "DNp04_L");
        _dnp04R = _brain.Query(type: "DNp04", side: Side.Right, name: "DNp04_R");
        _dnp01 = _brain.Query(type: "DNp01", name: "DNp01");

        _eyesLeft = _sim.Input(lc4L, InputKind.PoissonToV);
        _eyesRight = _sim.Input(lc4R, InputKind.PoissonToV);
        _left = new float[lc4L.Count];
        _right = new float[lc4R.Count];
        _dn = _sim.Output(_dnp04L | _dnp04R | _dnp01, OutputKind.Rate(30.Ms()), publishEvery: 20.Ms());

        _driver = new RealtimeDriver(_sim, new RealtimeOptions { Ratio = Ratio, MaxStepsPerTick = 500 });
        _driver.Start();
        GD.Print($"dotFly: opened {path}");
        _status = $"{_brain.Provenance.Name}: {_brain.NeuronCount:N0} neurons, {_brain.EdgeCount:N0} edges; LC4 {lc4L.Count}+{lc4R.Count}, DNp04 {_dnp04L.Count}+{_dnp04R.Count}, DNp01 {_dnp01.Count}";
    }

    /// <summary>
    /// Retina encoder (hand-written): the target's bearing relative to the fly's heading drives
    /// the LC4 population on the side the target is on, proportionally to how far off-axis it is
    /// (0 straight ahead, max at ±90°), scaled by proximity.
    /// </summary>
    public void Encode(float bearingRad, float distancePx)
    {
        if (_eyesLeft is null || _eyesRight is null)
        {
            return;
        }

        float proximity = Mathf.Clamp(1f - distancePx / 700f, 0.1f, 1f);
        float offAxis = Mathf.Clamp(MathF.Abs(bearingRad) / (MathF.PI / 2), 0f, 1f);
        float hz = MaxRetinaHz * offAxis * proximity;
        Array.Fill(_left, bearingRad < 0 ? hz : 0f);
        Array.Fill(_right, bearingRad > 0 ? hz : 0f);
        _eyesLeft.Write(_left);
        _eyesRight.Write(_right);
    }

    /// <summary>Reads the latest output frame and applies the fixed decoder.</summary>
    public void Decode()
    {
        if (_dn is null)
        {
            return;
        }

        OutputFrame f = _dn.Snapshot;
        float l = Mean(f, 0, _dnp04L.Count);
        float r = Mean(f, _dnp04L.Count, _dnp04R.Count);
        float p = Mean(f, _dnp04L.Count + _dnp04R.Count, _dnp01.Count);
        Rates = (l, r, p);
        // Turn away from the looming side (positive yaw = clockwise = towards +bearing).
        Actions = (_decoder.YawGainDegPerSecPerHz * (l - r), _decoder.IdleSpeedPxPerSec + _decoder.SpeedGainPxPerSecPerHz * p);
    }

    private static float Mean(OutputFrame f, int start, int count)
    {
        if (count == 0)
        {
            return 0;
        }

        double sum = 0;
        for (int i = 0; i < count; i++)
        {
            sum += f[start + i];
        }

        return (float)(sum / count);
    }

    /// <summary>Backend profile summary when DOTFLY_PROFILE=1.</summary>
    public string? Profile() => _sim?.Backend is DotFly.Cpu.Fast.CpuBackend cb ? cb.ProfileSummary() : null;

    /// <summary>HUD text: neural vs wall time, lag, activity and switches.</summary>
    public string Hud()
    {
        if (_sim is null || _driver is null)
        {
            return _status;
        }

        SimulationClock c = _sim.Clock;
        return $"{_status}\n" +
               $"neural {c.NeuralTime.TotalSeconds,7:F2} s   wall {c.WallTime.TotalSeconds,7:F2} s busy   RTF {c.RecentRealTimeFactor,5:F2}   behind {_driver.BehindBy.TotalMilliseconds,5:F0} ms   spikes/s {c.RecentSpikesPerSecond,9:N0}\n" +
               $"DNp04_L {Rates.DNp04L,6:F1} Hz   DNp04_R {Rates.DNp04R,6:F1} Hz   DNp01 {Rates.DNp01,6:F1} Hz   →   yaw {Actions.YawDegPerSec,7:F1} °/s   speed {Actions.SpeedPxPerSec,6:F0} px/s   (decoder: {_decoder.Origin})\n" +
               $"[R] recurrent {(_sim.RecurrentTransmission ? "on" : "OFF")}   [E] external input {(_sim.ExternalInput ? "on" : "OFF")}   [S] DNp04 silenced {(_silenced ? "YES" : "no")}   [Space] {(_driver.Running ? "pause" : "resume")}";
    }

    public override void _UnhandledKeyInput(InputEvent @event)
    {
        if (_sim is null || _driver is null || @event is not InputEventKey { Pressed: true } key)
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
            case Key.S:
                _silenced = !_silenced;
                // Silence/unsilence is a simulation-thread operation: schedule it.
                bool s = _silenced;
                NeuronSet set = _dnp04L | _dnp04R;
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

    /// <summary>Decoder gains loaded from <c>decoders/dn_fixed.json</c>.</summary>
    public sealed class Decoder
    {
        /// <summary>Provenance label.</summary>
        public string Origin { get; set; } = "Fixed";

        /// <summary>Yaw gain.</summary>
        public float YawGainDegPerSecPerHz { get; set; } = 4f;

        /// <summary>Speed gain.</summary>
        public float SpeedGainPxPerSecPerHz { get; set; } = 6f;

        /// <summary>Speed with no DNp01 activity.</summary>
        public float IdleSpeedPxPerSec { get; set; } = 30f;
    }
}

[System.Text.Json.Serialization.JsonSourceGenerationOptions(PropertyNameCaseInsensitive = true)]
[System.Text.Json.Serialization.JsonSerializable(typeof(FlyBrain.Decoder))]
internal sealed partial class DecoderJsonContext : System.Text.Json.Serialization.JsonSerializerContext
{
}
