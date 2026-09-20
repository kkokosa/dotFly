using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace DotFly.Cpu.Kernels;

/// <summary>
/// Phase A of a step for a contiguous neuron range: exact LIF update, threshold, refractory
/// bookkeeping and spike compaction. Float32 state; <see cref="Vector256{T}"/> when accelerated,
/// scalar otherwise. Both paths produce bit-identical results (same operations, same order).
/// <para>
/// Per neuron: <c>frozen = cd &gt; 0</c>, <c>cd ← max(cd − 1, 0)</c>. Non-frozen:
/// <c>v ← v0 + a(v − v0) + b·g</c>, <c>g ← c·g</c>, <c>spiked = v &gt; vth</c>, spikers get
/// <c>cd ← arm</c>. <c>blocked = frozen | spiked</c> is written as one bit per neuron
/// (<c>blockedBits[i &gt;&gt; 4]</c>, bit <c>i &amp; 15</c>) for the delivery phase.
/// </para>
/// <para>
/// When the reset potential equals the resting potential (Shiu 2024: both −52 mV) a refractory
/// neuron sits at <c>(v0, 0)</c>, which is a fixed point of the update and cannot reach threshold,
/// and its inputs are discarded — so the update can run unconditionally on every lane and the
/// only refractory bookkeeping left is the countdown and the blocked bit. That is the
/// <see cref="Constants.RestEqualsReset"/> fast path; the general path masks the update.
/// </para>
/// </summary>
public static unsafe class LifKernel
{
    /// <summary>Per-step constants in float32.</summary>
    public readonly record struct Constants(float A, float B, float C, float Rest, float Threshold, bool RestEqualsReset)
    {
        /// <summary>Converts from the double-precision descriptor.</summary>
        public static Constants From(Core.Models.LifStepConstants k) =>
            new((float)k.A, (float)k.B, (float)k.C, (float)k.RestMv, (float)k.ThresholdMv, (float)k.RestMv == (float)k.ResetMv);
    }

    /// <summary>
    /// Runs phase A on neurons <c>[start, end)</c>; <paramref name="start"/> must be a multiple of
    /// 16 (block boundaries are). Appends spiking neuron indices to <paramref name="spikes"/> and
    /// returns the count appended.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int Update(
        in Constants k,
        float* v,
        float* g,
        int* countdown,
        int* arm,
        ushort* blockedBits,
        int start,
        int end,
        int* spikes)
    {
        System.Diagnostics.Debug.Assert((start & 15) == 0, "range start must be 16-aligned");
        int count = 0;
        int i = start;

        if (Vector256.IsHardwareAccelerated && end - start >= 16)
        {
            Vector256<float> va = Vector256.Create(k.A);
            Vector256<float> vb = Vector256.Create(k.B);
            Vector256<float> vc = Vector256.Create(k.C);
            Vector256<float> v0 = Vector256.Create(k.Rest);
            Vector256<float> vth = Vector256.Create(k.Threshold);
            Vector256<int> zero = Vector256<int>.Zero;
            Vector256<int> one = Vector256.Create(1);

            if (k.RestEqualsReset)
            {
                for (; i + 16 <= end; i += 16)
                {
                    uint b0 = LaneFixedPoint(va, vb, vc, v0, vth, zero, one, v + i, g + i, countdown + i, out uint s0);
                    uint b1 = LaneFixedPoint(va, vb, vc, v0, vth, zero, one, v + i + 8, g + i + 8, countdown + i + 8, out uint s1);
                    blockedBits[i >> 4] = (ushort)(b0 | (b1 << 8));
                    count += Compact(s0 | (s1 << 8), i, countdown, arm, spikes + count);
                }
            }
            else
            {
                for (; i + 16 <= end; i += 16)
                {
                    uint b0 = LaneMasked(va, vb, vc, v0, vth, zero, one, v + i, g + i, countdown + i, out uint s0);
                    uint b1 = LaneMasked(va, vb, vc, v0, vth, zero, one, v + i + 8, g + i + 8, countdown + i + 8, out uint s1);
                    blockedBits[i >> 4] = (ushort)(b0 | (b1 << 8));
                    count += Compact(s0 | (s1 << 8), i, countdown, arm, spikes + count);
                }
            }
        }

        // Scalar tail (also the whole range when not accelerated). Bits are read-modify-written
        // per neuron so partially vectorised blocks stay consistent.
        for (; i < end; i++)
        {
            int cd = countdown[i];
            bool frozen = cd > 0;
            bool spiked = false;
            if (frozen)
            {
                countdown[i] = cd - 1;
                if (k.RestEqualsReset)
                {
                    // Fixed point: identical arithmetic to the vector path, no observable change.
                    float nvf = k.Rest + k.A * (v[i] - k.Rest) + k.B * g[i];
                    g[i] *= k.C;
                    v[i] = nvf;
                }
            }
            else
            {
                float nv = k.Rest + k.A * (v[i] - k.Rest) + k.B * g[i];
                g[i] *= k.C;
                v[i] = nv;
                spiked = nv > k.Threshold;
                if (spiked)
                {
                    countdown[i] = arm[i];
                    spikes[count++] = i;
                }
            }

            int bit = 1 << (i & 15);
            ref ushort word = ref blockedBits[i >> 4];
            word = (ushort)(frozen | spiked ? word | bit : word & ~bit);
        }

        return count;
    }

    /// <summary>Emits spike indices for the set bits, arms their countdowns; returns the count.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Compact(uint bits, int baseIndex, int* countdown, int* arm, int* spikes)
    {
        int n = 0;
        while (bits != 0)
        {
            int idx = baseIndex + BitOperations.TrailingZeroCount(bits);
            countdown[idx] = arm[idx];
            spikes[n++] = idx;
            bits &= bits - 1;
        }

        return n;
    }

    /// <summary>Fast path (rest == reset): unconditional update; returns the blocked bits, out the spiked bits.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint LaneFixedPoint(
        Vector256<float> va, Vector256<float> vb, Vector256<float> vc, Vector256<float> v0, Vector256<float> vth,
        Vector256<int> zero, Vector256<int> one, float* v, float* g, int* countdown, out uint spikedBits)
    {
        Vector256<float> vv = Vector256.Load(v);
        Vector256<float> gg = Vector256.Load(g);
        Vector256<int> cd = Vector256.Load(countdown);

        Vector256<float> nv = v0 + va * (vv - v0) + vb * gg;
        (gg * vc).Store(g);
        nv.Store(v);

        Vector256<int> frozen = Vector256.GreaterThan(cd, zero);
        Vector256.Max(cd - one, zero).Store(countdown);

        // A frozen neuron is at rest with g = 0 and cannot exceed threshold, so no mask is needed.
        Vector256<int> spiked = Vector256.GreaterThan(nv, vth).AsInt32();
        spikedBits = spiked.ExtractMostSignificantBits();
        return (frozen | spiked).ExtractMostSignificantBits();
    }

    /// <summary>General path: masked update for frozen lanes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint LaneMasked(
        Vector256<float> va, Vector256<float> vb, Vector256<float> vc, Vector256<float> v0, Vector256<float> vth,
        Vector256<int> zero, Vector256<int> one, float* v, float* g, int* countdown, out uint spikedBits)
    {
        Vector256<float> vv = Vector256.Load(v);
        Vector256<float> gg = Vector256.Load(g);
        Vector256<int> cd = Vector256.Load(countdown);
        Vector256<int> frozen = Vector256.GreaterThan(cd, zero);
        Vector256<float> frozenF = frozen.AsSingle();

        Vector256<float> nv = v0 + va * (vv - v0) + vb * gg;
        Vector256<float> ng = gg * vc;
        Vector256.ConditionalSelect(frozenF, vv, nv).Store(v);
        Vector256.ConditionalSelect(frozenF, gg, ng).Store(g);
        Vector256.Max(cd - one, zero).Store(countdown);

        Vector256<int> spiked = Vector256.GreaterThan(nv, vth).AsInt32() & ~frozen;
        spikedBits = spiked.ExtractMostSignificantBits();
        return (frozen | spiked).ExtractMostSignificantBits();
    }
}
