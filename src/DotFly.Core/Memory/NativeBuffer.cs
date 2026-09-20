using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DotFly.Core.Memory;

/// <summary>
/// A fixed-length, cache-line-aligned block of unmanaged memory. Used for all simulation state
/// so the hot path never allocates on the managed heap and SIMD loads are always aligned.
/// </summary>
/// <typeparam name="T">Unmanaged element type.</typeparam>
public sealed unsafe class NativeBuffer<T> : IDisposable
    where T : unmanaged
{
    /// <summary>Alignment in bytes; 64 covers AVX-512 and a full x64 cache line.</summary>
    public const int Alignment = 64;

    private T* _ptr;

    /// <summary>Number of elements.</summary>
    public int Length { get; }

    /// <summary>Raw pointer to the first element; valid until <see cref="Dispose"/>.</summary>
    public T* Pointer
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _ptr;
    }

    /// <summary>The buffer as a span.</summary>
    public Span<T> Span
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => new(_ptr, Length);
    }

    /// <summary>
    /// Allocates <paramref name="length"/> zero-initialised elements, rounding the byte size up to
    /// a multiple of <see cref="Alignment"/> so vector loops may read a full trailing vector.
    /// </summary>
    public NativeBuffer(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        Length = length;
        nuint bytes = (nuint)length * (nuint)sizeof(T);
        nuint rounded = (bytes + Alignment - 1) / Alignment * Alignment;
        if (rounded == 0)
        {
            rounded = Alignment;
        }

        _ptr = (T*)NativeMemory.AlignedAlloc(rounded, Alignment);
        NativeMemory.Clear(_ptr, rounded);
    }

    /// <summary>Element accessor without bounds checks beyond the debug assertion.</summary>
    public ref T this[int index]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            System.Diagnostics.Debug.Assert((uint)index < (uint)Length);
            return ref _ptr[index];
        }
    }

    /// <summary>Releases the unmanaged memory.</summary>
    public void Dispose()
    {
        if (_ptr != null)
        {
            NativeMemory.AlignedFree(_ptr);
            _ptr = null;
        }

        GC.SuppressFinalize(this);
    }

    /// <summary>Finaliser as a safety net; deterministic <see cref="Dispose"/> is expected.</summary>
    ~NativeBuffer()
    {
        if (_ptr != null)
        {
            NativeMemory.AlignedFree(_ptr);
        }
    }
}
