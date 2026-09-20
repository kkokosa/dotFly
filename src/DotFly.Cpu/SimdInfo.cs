using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace DotFly.Cpu;

/// <summary>
/// Reports which vector widths and instruction sets the CPU backend will use on this machine.
/// </summary>
public static class SimdInfo
{
    /// <summary>Widest hardware-accelerated vector in bits (512, 256, 128) or 0 for scalar-only.</summary>
    public static int WidestVectorBits =>
        Vector512.IsHardwareAccelerated ? 512 :
        Vector256.IsHardwareAccelerated ? 256 :
        Vector128.IsHardwareAccelerated ? 128 : 0;

    /// <summary>Number of <see cref="float"/> lanes in the widest accelerated vector (1 for scalar).</summary>
    public static int FloatLanes => Math.Max(1, WidestVectorBits / 32);

    /// <summary>Short description of the instruction sets available, for diagnostics.</summary>
    public static string Describe()
    {
        var parts = new List<string>(8);
        if (Avx512F.IsSupported)
        {
            parts.Add("AVX-512F");
        }

        if (Avx512BW.IsSupported)
        {
            parts.Add("AVX-512BW");
        }

        if (Avx2.IsSupported)
        {
            parts.Add("AVX2");
        }

        if (Fma.IsSupported)
        {
            parts.Add("FMA");
        }

        if (Sse42.IsSupported)
        {
            parts.Add("SSE4.2");
        }

        if (AdvSimd.IsSupported)
        {
            parts.Add("NEON");
        }

        if (parts.Count == 0)
        {
            parts.Add("scalar");
        }

        return $"{string.Join(", ", parts)} (Vector{WidestVectorBits}, {FloatLanes} float lanes)";
    }
}
