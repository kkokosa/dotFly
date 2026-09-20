using DotFly.Core.Graph;
using DotFly.Core.Memory;

namespace DotFly.Cpu.Fast;

/// <summary>
/// Per-row offsets at which the (sorted) columns of the checkpoint's CSR cross the boundaries of
/// <see cref="Blocks"/> contiguous post-neuron ranges. Thread <c>p</c> delivering the spike of row
/// <c>s</c> walks <c>col[Offset(s, p) .. Offset(s, p + 1))</c> — exactly the targets inside its own
/// range — so no two threads write the same <c>g[i]</c> and the summation order for every target
/// is the spike-list order regardless of thread count.
/// </summary>
public sealed unsafe class BlockedCsr : IDisposable
{
    private readonly NativeBuffer<int> _offsets;

    /// <summary>Number of post-neuron blocks.</summary>
    public int Blocks { get; }

    /// <summary>Block boundaries: block <c>p</c> covers neurons <c>[Bounds[p], Bounds[p+1])</c>.</summary>
    public int[] Bounds { get; }

    /// <summary>Number of rows (neurons).</summary>
    public int Rows { get; }

    /// <summary>Builds the block offsets for <paramref name="blocks"/> equal-sized ranges.</summary>
    public BlockedCsr(Brain brain, int blocks)
    {
        ArgumentNullException.ThrowIfNull(brain);
        ArgumentOutOfRangeException.ThrowIfLessThan(blocks, 1);
        Rows = brain.NeuronCount;
        Blocks = blocks;
        Bounds = new int[blocks + 1];
        for (int p = 0; p <= blocks; p++)
        {
            // 16-aligned so every block starts on a full SIMD/bit-mask group; the last takes the tail.
            Bounds[p] = p == blocks ? Rows : (int)((long)Rows * p / blocks) & ~15;
        }

        _offsets = new NativeBuffer<int>(checked(Rows * (blocks + 1)));
        int* off = _offsets.Pointer;
        long* rowPtr = brain.RowPtrPointer;
        int* cols = brain.ColumnsPointer;
        int stride = blocks + 1;

        for (int s = 0; s < Rows; s++)
        {
            long e = rowPtr[s];
            long rowEnd = rowPtr[s + 1];
            int* row = off + (long)s * stride;
            row[0] = checked((int)e);
            for (int p = 1; p <= blocks; p++)
            {
                int boundary = Bounds[p];
                while (e < rowEnd && cols[e] < boundary)
                {
                    e++;
                }

                row[p] = checked((int)e);
            }
        }
    }

    /// <summary>Offset of the first edge of row <paramref name="row"/> whose target lies in block <paramref name="block"/> or later.</summary>
    public int Offset(int row, int block) => _offsets.Pointer[(long)row * (Blocks + 1) + block];

    /// <summary>Raw pointer to the offsets table, row-major with stride <c>Blocks + 1</c>.</summary>
    public int* OffsetsPointer => _offsets.Pointer;

    /// <inheritdoc />
    public void Dispose() => _offsets.Dispose();
}
