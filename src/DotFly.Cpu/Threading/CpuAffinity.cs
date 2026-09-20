using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace DotFly.Cpu.Threading;

/// <summary>
/// Pins the current thread to one logical processor (Windows, Linux). Used so that spinning
/// compute workers spread over physical cores instead of SMT siblings. Best effort: returns false
/// when unsupported.
/// </summary>
internal static partial class CpuAffinity
{
    /// <summary>Pins the calling thread to <paramref name="logicalProcessorId"/>.</summary>
    public static bool PinCurrentThread(int logicalProcessorId)
    {
        if (logicalProcessorId < 0)
        {
            return false;
        }

        if (OperatingSystem.IsWindows())
        {
            return PinWindows(logicalProcessorId);
        }

        if (OperatingSystem.IsLinux())
        {
            return PinLinux(logicalProcessorId);
        }

        return false;
    }

    /// <summary>
    /// Logical processor for worker <paramref name="workerIndex"/> of <paramref name="workerCount"/>:
    /// when SMT is likely (even logical count ≥ 4 and workers ≤ half), use every second logical
    /// processor so each worker lands on its own physical core; otherwise spread sequentially.
    /// </summary>
    public static int ProcessorFor(int workerIndex, int workerCount)
    {
        int logical = Environment.ProcessorCount;
        if (logical >= 4 && logical % 2 == 0 && workerCount <= logical / 2)
        {
            return (workerIndex * 2) % logical;
        }

        return workerIndex % logical;
    }

    [SupportedOSPlatform("windows")]
    private static bool PinWindows(int logicalProcessorId)
    {
        if (logicalProcessorId >= 64)
        {
            return false;
        }

        try
        {
            nint mask = (nint)(1UL << logicalProcessorId);
            return WindowsNative.SetThreadAffinityMask(WindowsNative.GetCurrentThread(), mask) != 0;
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException)
        {
            return false;
        }
    }

    [SupportedOSPlatform("windows")]
    private static partial class WindowsNative
    {
        [LibraryImport("kernel32.dll")]
        internal static partial nint GetCurrentThread();

        [LibraryImport("kernel32.dll")]
        internal static partial nint SetThreadAffinityMask(nint hThread, nint dwThreadAffinityMask);
    }

    [SupportedOSPlatform("linux")]
    private static bool PinLinux(int logicalProcessorId)
    {
        try
        {
            const int CpuSetSize = 128;
            Span<ulong> mask = stackalloc ulong[CpuSetSize / sizeof(ulong)];
            mask.Clear();
            mask[logicalProcessorId / 64] = 1UL << (logicalProcessorId % 64);
            unsafe
            {
                fixed (ulong* p = mask)
                {
                    return LinuxNative.sched_setaffinity(0, (nuint)CpuSetSize, p) == 0;
                }
            }
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException)
        {
            return false;
        }
    }

    [SupportedOSPlatform("linux")]
    private static unsafe partial class LinuxNative
    {
        [LibraryImport("libc")]
        internal static partial int sched_setaffinity(int pid, nuint cpusetsize, ulong* mask);
    }
}
