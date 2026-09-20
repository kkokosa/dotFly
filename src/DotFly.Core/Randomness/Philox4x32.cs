using System.Runtime.CompilerServices;

namespace DotFly.Core.Randomness;

/// <summary>
/// Philox4x32-10 counter-based random number generator (Salmon et al. 2011). Stateless: the
/// output is a pure function of (key, counter), so the same seed yields the identical stream for
/// a given (step, neuron, stream) on any backend, any thread count, and on replay.
/// </summary>
public static class Philox4x32
{
    private const uint M0 = 0xD2511F53;
    private const uint M1 = 0xCD9E8D57;
    private const uint W0 = 0x9E3779B9;
    private const uint W1 = 0xBB67AE85;

    /// <summary>
    /// Produces four 32-bit outputs for the given 128-bit counter and 64-bit key.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (uint R0, uint R1, uint R2, uint R3) Generate(uint c0, uint c1, uint c2, uint c3, uint k0, uint k1)
    {
        for (int round = 0; round < 10; round++)
        {
            ulong p0 = (ulong)M0 * c0;
            ulong p1 = (ulong)M1 * c2;
            uint hi0 = (uint)(p0 >> 32), lo0 = (uint)p0;
            uint hi1 = (uint)(p1 >> 32), lo1 = (uint)p1;

            uint n0 = hi1 ^ c1 ^ k0;
            uint n1 = lo1;
            uint n2 = hi0 ^ c3 ^ k1;
            uint n3 = lo0;

            c0 = n0;
            c1 = n1;
            c2 = n2;
            c3 = n3;

            k0 += W0;
            k1 += W1;
        }

        return (c0, c1, c2, c3);
    }

    /// <summary>
    /// A uniform double in [0, 1) derived from (seed, step, index, stream). Uses 53 bits of the
    /// first two outputs.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Uniform(ulong seed, ulong step, uint index, uint stream)
    {
        (uint r0, uint r1, _, _) = Generate(
            (uint)step, (uint)(step >> 32), index, stream,
            (uint)seed, (uint)(seed >> 32));
        ulong bits = ((ulong)r0 << 21) ^ r1;           // 53 significant bits
        return (bits & ((1UL << 53) - 1)) * (1.0 / (1UL << 53));
    }
}
