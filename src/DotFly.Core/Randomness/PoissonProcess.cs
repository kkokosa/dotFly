using System.Runtime.CompilerServices;

namespace DotFly.Core.Randomness;

/// <summary>
/// Per-neuron Bernoulli(p)-per-step event streams sampled as geometric inter-arrival times, so
/// that a neuron costs one comparison per step and one random draw per event. The draw for the
/// <c>k</c>-th event of neuron <c>i</c> is <see cref="Philox4x32"/>-keyed by <c>(seed, i, k)</c>,
/// hence the stream is a pure function of the seed: identical on every backend, for any thread
/// count, and on replay. The distribution is exactly that of an independent Bernoulli(p) draw at
/// every step (number of failures before a success ~ Geometric(p)).
/// </summary>
public static class PoissonProcess
{
    /// <summary>Stream id used for the geometric draws.</summary>
    public const uint Stream = 1;

    /// <summary>
    /// Step of the next event given that the previous event (or the (re)start of stimulation) was
    /// at <paramref name="fromStep"/> exclusive: <c>fromStep + 1 + Geometric(p)</c>.
    /// </summary>
    /// <param name="seed">Simulation seed.</param>
    /// <param name="neuron">Neuron index.</param>
    /// <param name="eventIndex">Ordinal of the event being scheduled for this neuron.</param>
    /// <param name="fromStep">Step after which the search starts.</param>
    /// <param name="p">Per-step probability in (0, 1].</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long NextEventStep(ulong seed, int neuron, uint eventIndex, long fromStep, double p)
    {
        if (p >= 1.0)
        {
            return fromStep + 1;
        }

        // U in [0,1) → 1−U in (0,1] keeps log finite; floor(log(1−U)/log(1−p)) ~ Geometric(p).
        double u = 1.0 - Philox4x32.Uniform(seed, eventIndex, (uint)neuron, Stream);
        double k = Math.Floor(Math.Log(u) / Math.Log(1.0 - p));
        long failures = k >= long.MaxValue / 4 ? long.MaxValue / 4 : (long)k;
        return fromStep + 1 + failures;
    }
}
