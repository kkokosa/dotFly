using DotFly.Core.Graph;

namespace DotFly;

/// <summary>How an input port's values act on its neurons.</summary>
public enum InputKind
{
    /// <summary>
    /// Values are Poisson rates in Hz; each event adds the model's Poisson weight (68.75 mV) to
    /// <c>v</c> — Brian2 <c>PoissonInput(target_var='v')</c>, the Shiu et al. stimulation.
    /// </summary>
    PoissonToV,

    /// <summary>Values are a constant depolarisation in mV added to <c>v</c> every step (tonic drive).</summary>
    Current,
}

/// <summary>
/// K floats in: one value per neuron of a <see cref="NeuronSet"/>, in set order. Writes are
/// thread-safe and take effect from the next simulation step; converting sensor data (pixels,
/// sound, game state) into these values is the caller's encoder.
/// </summary>
public sealed class InputPort
{
    private readonly Simulation _sim;
    private readonly float[] _pending;
    private readonly float[] _applied;
    private readonly object _lock = new();
    private bool _dirty;
    private long _writes;

    /// <summary>Port id within the simulation (used by recordings).</summary>
    public int Id { get; }

    /// <summary>The neurons this port drives.</summary>
    public NeuronSet Set { get; }

    /// <summary>Kind of input.</summary>
    public InputKind Kind { get; }

    /// <summary>Number of values (= neurons).</summary>
    public int Count => Set.Count;

    /// <summary>Number of writes so far (diagnostic).</summary>
    public long Writes => Interlocked.Read(ref _writes);

    internal InputPort(Simulation sim, int id, NeuronSet set, InputKind kind)
    {
        _sim = sim;
        Id = id;
        Set = set;
        Kind = kind;
        _pending = new float[set.Count];
        _applied = new float[set.Count];
    }

    /// <summary>Sets all values (length must equal <see cref="Count"/>).</summary>
    public void Write(ReadOnlySpan<float> values)
    {
        if (values.Length != Count)
        {
            throw new ArgumentException($"Expected {Count} values, got {values.Length}.", nameof(values));
        }

        lock (_lock)
        {
            values.CopyTo(_pending);
            _dirty = true;
        }

        Interlocked.Increment(ref _writes);
    }

    /// <summary>Sets every value to <paramref name="value"/>.</summary>
    public void Fill(float value)
    {
        lock (_lock)
        {
            Array.Fill(_pending, value);
            _dirty = true;
        }

        Interlocked.Increment(ref _writes);
    }

    /// <summary>Sets one value by position in the set.</summary>
    public void Set1(int position, float value)
    {
        lock (_lock)
        {
            _pending[position] = value;
            _dirty = true;
        }

        Interlocked.Increment(ref _writes);
    }

    /// <summary>Values currently applied to the backend (simulation thread's view).</summary>
    public ReadOnlySpan<float> Applied => _applied;

    /// <summary>Applies pending writes to the backend. Called on the simulation thread.</summary>
    internal bool Apply()
    {
        lock (_lock)
        {
            if (!_dirty)
            {
                return false;
            }

            _pending.CopyTo(_applied, 0);
            _dirty = false;
        }

        _sim.ApplyInput(this, _applied);
        return true;
    }
}
