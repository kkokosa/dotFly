using DotFly.Core.Graph;

namespace DotFly;

/// <summary>What an output port publishes per neuron.</summary>
public readonly record struct OutputKind
{
    /// <summary>Kind of readout.</summary>
    public enum Readout
    {
        /// <summary>Firing rate in Hz over the trailing window.</summary>
        Rate,

        /// <summary>Spike count over the trailing window.</summary>
        SpikeCount,

        /// <summary>Membrane potential in mV at publish time.</summary>
        Membrane,
    }

    /// <summary>Readout kind.</summary>
    public Readout Kind { get; init; }

    /// <summary>Trailing window for rate/count readouts.</summary>
    public TimeSpan Window { get; init; }

    /// <summary>Firing rate over a trailing window.</summary>
    public static OutputKind Rate(TimeSpan window) => new() { Kind = Readout.Rate, Window = window };

    /// <summary>Spike count over a trailing window.</summary>
    public static OutputKind SpikeCount(TimeSpan window) => new() { Kind = Readout.SpikeCount, Window = window };

    /// <summary>Membrane potential.</summary>
    public static OutputKind Membrane => new() { Kind = Readout.Membrane };
}

/// <summary>One published readout: K values in set order, with the neural time it describes.</summary>
public sealed class OutputFrame
{
    private readonly float[] _values;

    /// <summary>Sequential frame number, starting at 0 after each reset.</summary>
    public long FrameIndex { get; }

    /// <summary>Step at the end of which the frame was published.</summary>
    public long Step { get; }

    /// <summary>Neural time at the end of that step.</summary>
    public TimeSpan NeuralTime { get; }

    /// <summary>The values (one per neuron of the port's set).</summary>
    public ReadOnlySpan<float> Values => _values;

    /// <summary>The values as memory (for tensor / interop views).</summary>
    public ReadOnlyMemory<float> Memory => _values;

    /// <summary>Value by position.</summary>
    public float this[int position] => _values[position];

    /// <summary>Arithmetic mean of the values (0 for an empty port).</summary>
    public float Mean()
    {
        if (_values.Length == 0)
        {
            return 0;
        }

        double sum = 0;
        foreach (float v in _values)
        {
            sum += v;
        }

        return (float)(sum / _values.Length);
    }

    internal OutputFrame(long frameIndex, long step, TimeSpan neuralTime, float[] values)
    {
        FrameIndex = frameIndex;
        Step = step;
        NeuralTime = neuralTime;
        _values = values;
    }
}

/// <summary>
/// K floats out: a per-neuron readout of a <see cref="NeuronSet"/>, published every
/// <see cref="PublishEvery"/> of neural time into an immutable <see cref="Snapshot"/> that any
/// thread may read without blocking the simulation.
/// </summary>
public sealed class OutputPort
{
    private readonly Simulation _sim;
    private readonly int[] _position;     // neuron index → position in set, or -1
    private readonly int[][] _buckets;    // ring of per-bucket spike counts [bucket][position]
    private readonly int _publishSteps;
    private readonly int _bucketCount;
    private int _bucket;
    private long _frameIndex;
    private OutputFrame _snapshot;

    /// <summary>Port id within the simulation (used by recordings).</summary>
    public int Id { get; }

    /// <summary>The neurons read out.</summary>
    public NeuronSet Set { get; }

    /// <summary>Readout kind.</summary>
    public OutputKind Kind { get; }

    /// <summary>Publication interval (neural time).</summary>
    public TimeSpan PublishEvery { get; }

    /// <summary>Number of values (= neurons).</summary>
    public int Count => Set.Count;

    /// <summary>The last published frame (initially all zeros at step −1). Lock-free.</summary>
    public OutputFrame Snapshot => Volatile.Read(ref _snapshot);

    /// <summary>Raised on the simulation thread when a frame is published.</summary>
    public event Action<OutputPort, OutputFrame>? Published;

    internal OutputPort(Simulation sim, int id, NeuronSet set, OutputKind kind, TimeSpan publishEvery, int neuronCount)
    {
        _sim = sim;
        Id = id;
        Set = set;
        Kind = kind;
        PublishEvery = publishEvery;
        _publishSteps = Math.Max(1, (int)Math.Round(publishEvery.TotalMilliseconds / sim.DtMs));
        _bucketCount = kind.Kind == OutputKind.Readout.Membrane ? 1 : Math.Max(1, (int)Math.Ceiling(kind.Window.TotalMilliseconds / (_publishSteps * sim.DtMs)));
        _position = new int[neuronCount];
        Array.Fill(_position, -1);
        for (int p = 0; p < set.Count; p++)
        {
            _position[set[p]] = p;
        }

        _buckets = new int[_bucketCount][];
        for (int b = 0; b < _bucketCount; b++)
        {
            _buckets[b] = new int[set.Count];
        }

        _snapshot = new OutputFrame(-1, -1, TimeSpan.Zero, new float[set.Count]);
    }

    /// <summary>Effective window in steps (rounded up to whole publish intervals).</summary>
    public int WindowSteps => _bucketCount * _publishSteps;

    internal void Accumulate(ReadOnlySpan<int> spikes)
    {
        int[] bucket = _buckets[_bucket];
        foreach (int i in spikes)
        {
            int p = _position[i];
            if (p >= 0)
            {
                bucket[p]++;
            }
        }
    }

    internal bool ShouldPublish(long step) => (step + 1) % _publishSteps == 0;

    internal void Publish(long step)
    {
        var values = new float[Set.Count];
        switch (Kind.Kind)
        {
            case OutputKind.Readout.Membrane:
                for (int p = 0; p < values.Length; p++)
                {
                    values[p] = (float)_sim.Backend.GetV(Set[p]);
                }

                break;
            default:
                double scale = Kind.Kind == OutputKind.Readout.Rate ? 1000.0 / (WindowSteps * _sim.DtMs) : 1.0;
                for (int b = 0; b < _bucketCount; b++)
                {
                    int[] bucket = _buckets[b];
                    for (int p = 0; p < values.Length; p++)
                    {
                        values[p] += bucket[p];
                    }
                }

                if (scale != 1.0)
                {
                    for (int p = 0; p < values.Length; p++)
                    {
                        values[p] = (float)(values[p] * scale);
                    }
                }

                break;
        }

        var frame = new OutputFrame(_frameIndex++, step, TimeSpan.FromMilliseconds((step + 1) * _sim.DtMs), values);
        Volatile.Write(ref _snapshot, frame);
        _bucket = (_bucket + 1) % _bucketCount;
        Array.Clear(_buckets[_bucket]);
        Published?.Invoke(this, frame);
    }

    internal void Reset()
    {
        foreach (int[] b in _buckets)
        {
            Array.Clear(b);
        }

        _bucket = 0;
        _frameIndex = 0;
        Volatile.Write(ref _snapshot, new OutputFrame(-1, -1, TimeSpan.Zero, new float[Set.Count]));
    }
}
