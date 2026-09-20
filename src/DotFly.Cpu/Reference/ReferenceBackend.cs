using DotFly.Core.Backends;
using DotFly.Core.Graph;
using DotFly.Core.Models;
using DotFly.Core.Randomness;

namespace DotFly.Cpu.Reference;

/// <summary>
/// Scalar, single-threaded, float64 reference implementation of the Shiu et al. 2024 LIF network.
/// Written for clarity and fidelity to Brian2's step schedule, not for speed; the SIMD backends are
/// validated against it.
/// <para>Per step <c>t</c> (Brian2 default slot order):</para>
/// <list type="number">
/// <item><b>State update</b> — non-refractory neurons: <c>v ← v0 + a(v−v0) + b·g</c>, <c>g ← c·g</c>; refractory neurons: countdown only (v and g frozen).</item>
/// <item><b>Threshold</b> — non-refractory neurons with <c>v &gt; vth</c> spike.</item>
/// <item><b>Synapses</b> — spikes emitted at step <c>t − delay</c> are delivered: <c>g[post] += w·wsyn</c>; Poisson events and scheduled injections add to <c>v</c>. <b>Both are discarded for a refractory target</b> (see below).</item>
/// <item><b>Reset</b> — this step's spikers: <c>v ← vrst</c>, <c>g ← 0</c>, refractory countdown armed (always applied).</item>
/// </list>
/// A neuron spiking at step <c>t</c> stays frozen for steps <c>t+1 … t+R−1</c> and integrates again
/// at <c>t+R</c> (Brian2: <c>timestep(t − lastspike) ≥ timestep(rfc)</c>), hence the countdown is
/// armed with <c>R − 1</c>. Because <c>v</c> and <c>g</c> carry Brian2's <c>(unless refractory)</c>
/// flag, every write to them outside the resetter is guarded by <c>not_refractory</c> — which the
/// thresholder clears at the spike step — so synaptic deliveries and input events arriving at steps
/// <c>t … t+R−1</c> are silently dropped, not accumulated. Verified against Brian2 2.9 codegen
/// (<c>set_conditional_write</c>; only the resetter uses <c>override_conditional_write</c>).
/// </summary>
public sealed class ReferenceBackend : ISpikingBackend
{
    private readonly Brain _brain;
    private readonly LifStepConstants _k;
    private readonly int _n;

    private readonly double[] _v;
    private readonly double[] _g;
    private readonly int[] _rfcCountdown;
    private readonly int[] _rfcSteps;
    private readonly bool[] _blocked;      // refractory (frozen or spiked) at this step → inputs discarded
    private readonly double[] _poissonP;
    private readonly long[] _nextEvent;    // step of the next Poisson event per stimulated neuron
    private readonly uint[] _eventIndex;   // events scheduled so far per neuron (RNG counter)
    private readonly bool[] _transmits;
    private readonly double[] _current;
    private readonly List<int> _poissonNeurons = [];
    private readonly List<int> _currentNeurons = [];

    // Delay ring: slot s holds the spikes to deliver at step (s ≡ step mod RingSlots).
    private readonly int[][] _ring;
    private readonly int[] _ringCount;
    private readonly int[] _spikesThisStep;
    private int _spikeCount;

    private readonly SortedDictionary<long, List<(int Neuron, double Mv)>> _injections = new();

    /// <inheritdoc />
    public Brain Brain => _brain;

    /// <inheritdoc />
    public LifStepConstants Constants => _k;

    /// <inheritdoc />
    public int NeuronCount => _n;

    /// <inheritdoc />
    public long Step { get; private set; }

    /// <inheritdoc />
    public ulong Seed { get; private set; }

    /// <inheritdoc />
    public bool RecurrentTransmission { get; set; } = true;

    /// <inheritdoc />
    public bool ExternalInput { get; set; } = true;

    /// <summary>Creates a backend for <paramref name="brain"/> with the checkpoint's model discretised at <paramref name="dtMs"/>.</summary>
    public ReferenceBackend(Brain brain, ulong seed, double dtMs = 0.1)
        : this(brain, brain.Model.Discretize(dtMs), seed)
    {
    }

    /// <summary>Creates a backend with explicit step constants.</summary>
    public ReferenceBackend(Brain brain, LifStepConstants constants, ulong seed)
    {
        ArgumentNullException.ThrowIfNull(brain);
        _brain = brain;
        _k = constants;
        _n = brain.NeuronCount;
        Seed = seed;

        _v = new double[_n];
        _g = new double[_n];
        _rfcCountdown = new int[_n];
        _rfcSteps = new int[_n];
        _blocked = new bool[_n];
        _poissonP = new double[_n];
        _nextEvent = new long[_n];
        _eventIndex = new uint[_n];
        _transmits = new bool[_n];
        _current = new double[_n];
        Array.Fill(_rfcSteps, constants.RefractorySteps);
        Array.Fill(_transmits, true);

        _ring = new int[constants.RingSlots][];
        _ringCount = new int[constants.RingSlots];
        for (int i = 0; i < _ring.Length; i++)
        {
            _ring[i] = new int[Math.Max(16, _n)];
        }

        _spikesThisStep = new int[_n];
        Reset();
    }

    /// <inheritdoc />
    public double GetV(int neuron) => _v[neuron];

    /// <inheritdoc />
    public double GetG(int neuron) => _g[neuron];

    /// <inheritdoc />
    public void SetRefractorySteps(int neuron, int steps)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(steps);
        _rfcSteps[neuron] = steps;
    }

    /// <inheritdoc />
    public void SetPoissonProbability(int neuron, double probabilityPerStep)
    {
        if (probabilityPerStep is < 0 or > 1 || double.IsNaN(probabilityPerStep))
        {
            throw new ArgumentOutOfRangeException(nameof(probabilityPerStep));
        }

        bool was = _poissonP[neuron] > 0;
        _poissonP[neuron] = probabilityPerStep;
        bool now = probabilityPerStep > 0;
        if (now)
        {
            // (Re)start the stream from the current step with a fresh draw.
            _nextEvent[neuron] = PoissonProcess.NextEventStep(Seed, neuron, _eventIndex[neuron]++, Step - 1, probabilityPerStep);
        }

        if (now && !was)
        {
            int pos = _poissonNeurons.BinarySearch(neuron);
            _poissonNeurons.Insert(~pos, neuron);
        }
        else if (!now && was)
        {
            _poissonNeurons.Remove(neuron);
        }
    }

    /// <inheritdoc />
    public void SetCurrent(int neuron, double mvPerStep)
    {
        if (double.IsNaN(mvPerStep))
        {
            throw new ArgumentOutOfRangeException(nameof(mvPerStep));
        }

        bool was = _current[neuron] != 0;
        _current[neuron] = mvPerStep;
        bool now = mvPerStep != 0;
        if (now && !was)
        {
            _currentNeurons.Insert(~_currentNeurons.BinarySearch(neuron), neuron);
        }
        else if (!now && was)
        {
            _currentNeurons.Remove(neuron);
        }
    }

    /// <inheritdoc />
    public void SetTransmits(int neuron, bool transmits) => _transmits[neuron] = transmits;

    /// <inheritdoc />
    public byte[] SaveState()
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write(Step);
        w.Write(_n);
        foreach (double x in _v) w.Write(x);
        foreach (double x in _g) w.Write(x);
        foreach (int x in _rfcCountdown) w.Write(x);
        foreach (bool x in _blocked) w.Write(x);
        foreach (uint x in _eventIndex) w.Write(x);
        foreach (long x in _nextEvent) w.Write(x);
        w.Write(_ring.Length);
        for (int slot = 0; slot < _ring.Length; slot++)
        {
            w.Write(_ringCount[slot]);
            for (int i = 0; i < _ringCount[slot]; i++) w.Write(_ring[slot][i]);
        }

        w.Flush();
        return ms.ToArray();
    }

    /// <inheritdoc />
    public void RestoreState(ReadOnlySpan<byte> state)
    {
        using var ms = new MemoryStream(state.ToArray());
        using var r = new BinaryReader(ms);
        long step = r.ReadInt64();
        if (r.ReadInt32() != _n)
        {
            throw new InvalidDataException("State was saved for a different neuron count.");
        }

        for (int i = 0; i < _n; i++) _v[i] = r.ReadDouble();
        for (int i = 0; i < _n; i++) _g[i] = r.ReadDouble();
        for (int i = 0; i < _n; i++) _rfcCountdown[i] = r.ReadInt32();
        for (int i = 0; i < _n; i++) _blocked[i] = r.ReadBoolean();
        for (int i = 0; i < _n; i++) _eventIndex[i] = r.ReadUInt32();
        for (int i = 0; i < _n; i++) _nextEvent[i] = r.ReadInt64();
        if (r.ReadInt32() != _ring.Length)
        {
            throw new InvalidDataException("State was saved for a different delay.");
        }

        for (int slot = 0; slot < _ring.Length; slot++)
        {
            _ringCount[slot] = r.ReadInt32();
            for (int i = 0; i < _ringCount[slot]; i++) _ring[slot][i] = r.ReadInt32();
        }

        _injections.Clear();
        _spikeCount = 0;
        Step = step;
    }

    /// <inheritdoc />
    public void ScheduleInjection(long step, int neuron, double mv)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(step, Step);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual((uint)neuron, (uint)_n);
        if (!_injections.TryGetValue(step, out var list))
        {
            _injections[step] = list = [];
        }

        list.Add((neuron, mv));
    }

    /// <inheritdoc />
    public void Advance(int steps, ISpikeSink? sink)
    {
        for (int s = 0; s < steps; s++)
        {
            StepOnce();
            sink?.OnStep(Step - 1, _spikesThisStep.AsSpan(0, _spikeCount));
        }
    }

    /// <inheritdoc />
    public void Reset(ulong seed)
    {
        Seed = seed;
        Reset();
    }

    /// <inheritdoc />
    public void Reset()
    {
        Array.Fill(_v, _k.RestMv);
        Array.Clear(_g);
        Array.Clear(_rfcCountdown);
        Array.Clear(_blocked);
        Array.Clear(_ringCount);
        _injections.Clear();
        _spikeCount = 0;
        Step = 0;
        Array.Clear(_eventIndex);
        foreach (int i in _poissonNeurons)
        {
            _nextEvent[i] = PoissonProcess.NextEventStep(Seed, i, _eventIndex[i]++, -1, _poissonP[i]);
        }
    }

    private void StepOnce()
    {
        long t = Step;
        double a = _k.A, b = _k.B, c = _k.C, v0 = _k.RestMv, vth = _k.ThresholdMv;
        int slots = _k.RingSlots;

        // 1 + 2: state update and threshold.
        int count = 0;
        for (int i = 0; i < _n; i++)
        {
            if (_rfcCountdown[i] > 0)
            {
                _rfcCountdown[i]--;
                _blocked[i] = true;
                continue;
            }

            double v = v0 + a * (_v[i] - v0) + b * _g[i];
            _g[i] *= c;
            _v[i] = v;
            bool spiked = v > vth;
            _blocked[i] = spiked;
            if (spiked)
            {
                _spikesThisStep[count++] = i;
            }
        }

        _spikeCount = count;

        // Publish this step's spikes for delivery at t + delay.
        int publishSlot = (int)((t + _k.DelaySteps) % slots);
        Array.Copy(_spikesThisStep, _ring[publishSlot], count);
        _ringCount[publishSlot] = count;

        // 3: synapses — deliver spikes emitted at t − delay (they sit in slot t mod slots).
        if (RecurrentTransmission)
        {
            int dueSlot = (int)(t % slots);
            int[] due = _ring[dueSlot];
            int dueCount = _ringCount[dueSlot];
            ReadOnlySpan<long> rp = _brain.RowPtr;
            ReadOnlySpan<int> cols = _brain.Columns;
            ReadOnlySpan<short> w = _brain.Weights;
            double wsyn = _k.SynapseWeightMv;
            for (int k = 0; k < dueCount; k++)
            {
                int pre = due[k];
                if (!_transmits[pre])
                {
                    continue;
                }

                for (long e = rp[pre]; e < rp[pre + 1]; e++)
                {
                    int post = cols[(int)e];
                    if (!_blocked[post])
                    {
                        _g[post] += w[(int)e] * wsyn;
                    }
                }
            }
        }

        // Consumed; the slot will be overwritten when reused.
        _ringCount[(int)(t % slots)] = 0;

        if (ExternalInput)
        {
            // Poisson events (Brian2 PoissonInput, N=1: Bernoulli(rate·dt) per step) → v, sampled
            // as geometric inter-arrival times; the stream depends only on the seed.
            double pw = _k.PoissonWeightMv;
            foreach (int i in _poissonNeurons)
            {
                if (_nextEvent[i] == t)
                {
                    if (!_blocked[i])
                    {
                        _v[i] += pw;
                    }

                    _nextEvent[i] = PoissonProcess.NextEventStep(Seed, i, _eventIndex[i]++, t, _poissonP[i]);
                }
            }

            // Constant currents → v.
            foreach (int i in _currentNeurons)
            {
                if (!_blocked[i])
                {
                    _v[i] += _current[i];
                }
            }

            // Scheduled injections (explicit spike trains for replay/golden tests) → v.
            if (_injections.Count > 0 && _injections.TryGetValue(t, out var list))
            {
                foreach ((int neuron, double mv) in list)
                {
                    if (!_blocked[neuron])
                    {
                        _v[neuron] += mv;
                    }
                }

                _injections.Remove(t);
            }
        }

        // 4: reset.
        double vrst = _k.ResetMv;
        for (int k = 0; k < count; k++)
        {
            int i = _spikesThisStep[k];
            _v[i] = vrst;
            _g[i] = 0.0;
            _rfcCountdown[i] = Math.Max(0, _rfcSteps[i] - 1);
        }

        Step = t + 1;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        // Managed arrays only; nothing to release. Present for interface symmetry with native backends.
    }
}
