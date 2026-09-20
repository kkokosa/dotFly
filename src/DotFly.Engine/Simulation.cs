using System.Diagnostics;
using DotFly.Core.Backends;
using DotFly.Core.Graph;
using DotFly.Core.Models;
using DotFly.Cpu.Fast;
using DotFly.Cpu.Reference;

namespace DotFly;

/// <summary>Receives the spikes of one step (neuron indices ascending). Do not retain the span.</summary>
public delegate void SpikeHandler(long step, ReadOnlySpan<int> spikes);

/// <summary>
/// A running network: one <see cref="Brain"/>, one backend, the input/output ports bound to
/// neuron sets, the clock, the recorder and the causal switches. All members are meant to be
/// called from one thread (the caller of <see cref="Run"/>/<see cref="Advance"/>, or the
/// <see cref="RealtimeDriver"/>'s thread) except <see cref="InputPort.Write"/>,
/// <see cref="OutputPort.Snapshot"/>, the switches and the read-only properties, which are safe
/// from any thread.
/// </summary>
public sealed class Simulation : IDisposable, ISpikeSink
{
    private readonly List<InputPort> _inputs = [];
    private readonly List<OutputPort> _outputs = [];
    private readonly List<SpikeHandler> _handlers = [];
    private readonly SortedDictionary<long, List<Action<Simulation>>> _scheduled = new();
    private readonly HashSet<int> _silenced = [];
    private readonly bool[] _poissonDriven;
    private Recorder? _recorder;
    private volatile bool _recurrent = true;
    private volatile bool _external = true;
    private long _batchSpikes;

    /// <summary>The checkpoint.</summary>
    public Brain Brain { get; }

    /// <summary>The backend (kernel-level access; prefer the engine API).</summary>
    public ISpikingBackend Backend { get; }

    /// <summary>Options this simulation was created with.</summary>
    public SimulationOptions Options { get; }

    /// <summary>Per-step constants in use.</summary>
    public LifStepConstants Constants => Backend.Constants;

    /// <summary>Timestep in ms.</summary>
    public double DtMs => Options.DtMs;

    /// <summary>Neural / wall time bookkeeping.</summary>
    public SimulationClock Clock { get; }

    /// <summary>Steps completed.</summary>
    public long Step => Backend.Step;

    /// <summary>Neural time elapsed.</summary>
    public TimeSpan NeuralTime => Clock.NeuralTime;

    /// <summary>Input ports created so far.</summary>
    public IReadOnlyList<InputPort> Inputs => _inputs;

    /// <summary>Output ports created so far.</summary>
    public IReadOnlyList<OutputPort> Outputs => _outputs;

    /// <summary>Causal switch: deliver spikes to their targets (true) or cut all recurrent transmission (false).</summary>
    public bool RecurrentTransmission
    {
        get => _recurrent;
        set => _recurrent = value;
    }

    /// <summary>Causal switch: apply external inputs (Poisson, currents, injections) or ignore them all.</summary>
    public bool ExternalInput
    {
        get => _external;
        set => _external = value;
    }

    /// <summary>Neurons currently silenced (outgoing synapses zeroed; they still spike).</summary>
    public IReadOnlyCollection<int> Silenced => _silenced;

    /// <summary>Called on the simulation thread with each step's spikes.</summary>
    public event SpikeHandler OnSpikes
    {
        add => _handlers.Add(value);
        remove => _handlers.Remove(value);
    }

    internal Simulation(Brain brain, SimulationOptions options)
    {
        Brain = brain;
        Options = options;
        if (options.SynapticGain <= 0 || double.IsNaN(options.SynapticGain))
        {
            throw new ArgumentOutOfRangeException(nameof(options), "SynapticGain must be positive.");
        }

        LifStepConstants constants = (brain.Model with { SynapseWeightMv = brain.Model.SynapseWeightMv * options.SynapticGain }).Discretize(options.DtMs);
        Backend = options.Backend switch
        {
            DotFly.Backend.Cpu => new CpuBackend(brain, constants, options.Seed, options.Threads, options.ThreadPinOffset, options.ThreadPinTotal),
            DotFly.Backend.Reference => new ReferenceBackend(brain, constants, options.Seed),
            _ => throw new ArgumentOutOfRangeException(nameof(options), $"Unknown backend {options.Backend}."),
        };
        Clock = new SimulationClock(options.DtMs);
        _poissonDriven = new bool[brain.NeuronCount];
    }

    // ---- ports -----------------------------------------------------------------------------

    /// <summary>Creates an input port over <paramref name="set"/>; values start at 0 (no input).</summary>
    public InputPort Input(NeuronSet set, InputKind kind)
    {
        ArgumentNullException.ThrowIfNull(set);
        CheckSet(set);
        var port = new InputPort(this, _inputs.Count, set, kind);
        _inputs.Add(port);
        _recorder?.WritePortDefinition(port.Id, isInput: true, set, kind.ToString());
        return port;
    }

    /// <summary>Creates an output port over <paramref name="set"/>, published every <paramref name="publishEvery"/> (default 20 ms).</summary>
    public OutputPort Output(NeuronSet set, OutputKind kind, TimeSpan? publishEvery = null)
    {
        ArgumentNullException.ThrowIfNull(set);
        CheckSet(set);
        var port = new OutputPort(this, _outputs.Count, set, kind, publishEvery ?? 20.Ms(), Brain.NeuronCount);
        _outputs.Add(port);
        _recorder?.WritePortDefinition(port.Id, isInput: false, set, kind.Kind.ToString());
        return port;
    }

    internal void ApplyInput(InputPort port, ReadOnlySpan<float> values)
    {
        NeuronSet set = port.Set;
        switch (port.Kind)
        {
            case InputKind.PoissonToV:
                double dtSeconds = DtMs * 1e-3;
                for (int p = 0; p < set.Count; p++)
                {
                    int i = set[p];
                    double prob = Math.Clamp(values[p] * dtSeconds, 0.0, 1.0);
                    Backend.SetPoissonProbability(i, prob);
                    if (Options.ShiuPoissonSemantics)
                    {
                        bool driven = prob > 0;
                        if (driven != _poissonDriven[i])
                        {
                            _poissonDriven[i] = driven;
                            Backend.SetRefractorySteps(i, driven ? 0 : Constants.RefractorySteps);
                        }
                    }
                }

                break;
            case InputKind.Current:
                for (int p = 0; p < set.Count; p++)
                {
                    Backend.SetCurrent(set[p], values[p]);
                }

                break;
        }

        _recorder?.WriteInput(Step, port.Id, values);
    }

    // ---- controls --------------------------------------------------------------------------

    /// <summary>Zeroes the outgoing synapses of the set's neurons (Shiu-style silencing). They still integrate and spike.</summary>
    public void Silence(NeuronSet set)
    {
        ArgumentNullException.ThrowIfNull(set);
        CheckSet(set);
        foreach (int i in set.Indices)
        {
            if (_silenced.Add(i))
            {
                Backend.SetTransmits(i, false);
            }
        }
    }

    /// <summary>Restores the outgoing synapses of the set's neurons.</summary>
    public void Unsilence(NeuronSet set)
    {
        ArgumentNullException.ThrowIfNull(set);
        foreach (int i in set.Indices)
        {
            if (_silenced.Remove(i))
            {
                Backend.SetTransmits(i, true);
            }
        }
    }

    /// <summary>Schedules one input event of the model's Poisson weight on <paramref name="neuron"/> at neural time <paramref name="at"/> (≥ now).</summary>
    public void Inject(int neuron, TimeSpan at) => Inject(neuron, at, Constants.PoissonWeightMv);

    /// <summary>Schedules a voltage injection of <paramref name="mv"/> on <paramref name="neuron"/> at neural time <paramref name="at"/> (≥ now).</summary>
    public void Inject(int neuron, TimeSpan at, double mv)
    {
        long step = StepOf(at);
        Backend.ScheduleInjection(step, neuron, mv);
        _recorder?.WriteInjection(step, neuron, mv);
    }

    /// <summary>Schedules an event on every neuron of <paramref name="set"/> at <paramref name="at"/>.</summary>
    public void Inject(NeuronSet set, TimeSpan at)
    {
        ArgumentNullException.ThrowIfNull(set);
        foreach (int i in set.Indices)
        {
            Inject(i, at);
        }
    }

    /// <summary>Runs <paramref name="action"/> on the simulation thread just before step <paramref name="step"/> executes (≥ current step).</summary>
    public void ScheduleAt(long step, Action<Simulation> action)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(step, Step);
        if (!_scheduled.TryGetValue(step, out var list))
        {
            _scheduled[step] = list = [];
        }

        list.Add(action);
    }

    /// <summary>Runs <paramref name="action"/> just before the step at neural time <paramref name="at"/>.</summary>
    public void ScheduleAt(TimeSpan at, Action<Simulation> action) => ScheduleAt(StepOf(at), action);

    /// <summary>Step index of a neural time (rounded to the nearest step).</summary>
    public long StepOf(TimeSpan neuralTime) => (long)Math.Round(neuralTime.TotalMilliseconds / DtMs);

    // ---- running ---------------------------------------------------------------------------

    /// <summary>Advances by <paramref name="neural"/> of neural time (rounded to whole steps).</summary>
    public void Run(TimeSpan neural) => Advance((int)Math.Round(neural.TotalMilliseconds / DtMs));

    /// <summary>Advances by <paramref name="steps"/> steps.</summary>
    public void Advance(int steps)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(steps);
        while (steps > 0)
        {
            long now = Step;
            // Apply pending port writes and switches on this thread.
            foreach (InputPort port in _inputs)
            {
                port.Apply();
            }

            Backend.RecurrentTransmission = _recurrent;
            Backend.ExternalInput = _external;

            if (_scheduled.Count > 0)
            {
                long first = _scheduled.Keys.First();
                if (first == now)
                {
                    List<Action<Simulation>> actions = _scheduled[first];
                    _scheduled.Remove(first);
                    foreach (Action<Simulation> a in actions)
                    {
                        a(this);
                    }

                    continue; // re-apply inputs the actions may have changed
                }

                if (first < now + steps)
                {
                    int chunk = (int)(first - now);
                    RunChunk(chunk);
                    steps -= chunk;
                    continue;
                }
            }

            RunChunk(steps);
            steps = 0;
        }
    }

    private void RunChunk(int steps)
    {
        _batchSpikes = 0;
        Clock.Begin();
        Backend.Advance(steps, this);
        Clock.End(steps, _batchSpikes);
    }

    /// <inheritdoc />
    void ISpikeSink.OnStep(long step, ReadOnlySpan<int> spikes)
    {
        _batchSpikes += spikes.Length;
        foreach (OutputPort port in _outputs)
        {
            port.Accumulate(spikes);
            if (port.ShouldPublish(step))
            {
                port.Publish(step);
                _recorder?.WriteOutput(step, port.Id, port.Snapshot);
            }
        }

        _recorder?.WriteSpikes(step, spikes);
        foreach (SpikeHandler h in _handlers)
        {
            h(step, spikes);
        }
    }

    /// <summary>Resets the network state and clock to step 0 with a new seed (a fresh trial). Ports, silencing and switches are kept.</summary>
    public void Reset(ulong seed)
    {
        Backend.Reset(seed);
        AfterReset(0);
    }

    /// <summary>Resets the network state and clock to step 0 (same seed ⇒ same run). Ports, silencing and switches are kept.</summary>
    public void Reset()
    {
        Backend.Reset();
        AfterReset(0);
    }

    private void AfterReset(long step)
    {
        _scheduled.Clear();
        Clock.Reset();
        foreach (OutputPort port in _outputs)
        {
            port.Reset();
        }

        _recorder?.WriteReset(step);
    }

    /// <summary>Current seed of the backend.</summary>
    public ulong Seed => Backend.Seed;

    /// <summary>Captures the dynamic state for later <see cref="Restore"/> (branching experiments, episode resets).</summary>
    public SimulationSnapshot Snapshot() => new(Step, Backend.SaveState());

    /// <summary>Restores a snapshot taken from this simulation (or one with the same checkpoint and options). Scheduled actions and injections are dropped.</summary>
    public void Restore(SimulationSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        Backend.RestoreState(snapshot.State);
        _scheduled.Clear();
        Clock.Reset();
        foreach (OutputPort port in _outputs)
        {
            port.Reset();
        }

        _recorder?.WriteReset(snapshot.Step);
    }

    // ---- recording -------------------------------------------------------------------------

    /// <summary>Starts recording to a <c>.dfs</c> file; dispose the returned recorder (or this simulation) to finish it.</summary>
    public Recorder Record(string path, RecordFlags what = RecordFlags.All)
    {
        if (_recorder is not null)
        {
            throw new InvalidOperationException("Already recording.");
        }

        _recorder = new Recorder(path, this, what);
        foreach (InputPort p in _inputs)
        {
            _recorder.WritePortDefinition(p.Id, true, p.Set, p.Kind.ToString());
        }

        foreach (OutputPort p in _outputs)
        {
            _recorder.WritePortDefinition(p.Id, false, p.Set, p.Kind.Kind.ToString());
        }

        return _recorder;
    }

    internal void RecorderClosed(Recorder r)
    {
        if (ReferenceEquals(_recorder, r))
        {
            _recorder = null;
        }
    }

    private void CheckSet(NeuronSet set)
    {
        if (set.Count > 0 && set[set.Count - 1] >= Brain.NeuronCount)
        {
            throw new ArgumentOutOfRangeException(nameof(set), $"Set '{set.Name}' refers to neuron {set[set.Count - 1]} but the checkpoint has {Brain.NeuronCount} neurons.");
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _recorder?.Dispose();
        Backend.Dispose();
    }
}

/// <summary>An opaque copy of a simulation's dynamic state.</summary>
public sealed class SimulationSnapshot
{
    internal byte[] State { get; }

    /// <summary>Step the snapshot was taken at.</summary>
    public long Step { get; }

    /// <summary>Size in bytes.</summary>
    public int Size => State.Length;

    internal SimulationSnapshot(long step, byte[] state)
    {
        Step = step;
        State = state;
    }
}

/// <summary>Factory methods on <see cref="Brain"/>.</summary>
public static class BrainExtensions
{
    /// <summary>Creates a simulation of this brain.</summary>
    public static Simulation CreateSimulation(this Brain brain, SimulationOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(brain);
        return new Simulation(brain, options ?? new SimulationOptions());
    }
}
