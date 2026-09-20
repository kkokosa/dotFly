using DotFly.Core.Graph;
using DotFly.Core.Models;

namespace DotFly.Core.Checkpoints;

/// <summary>
/// In-memory representation of everything that goes into a <c>.dfb</c> file. Produced by the
/// builders in <c>DotFly.Data</c> and consumed by <see cref="DfbWriter"/>.
/// </summary>
public sealed class CheckpointContent
{
    /// <summary>Provenance record.</summary>
    public required Provenance Provenance { get; init; }

    /// <summary>String table referenced by the annotation columns.</summary>
    public required StringTable Strings { get; init; }

    /// <summary>Body / root IDs, one per neuron, in neuron-index order.</summary>
    public required ulong[] BodyIds { get; init; }

    /// <summary>Cell type string indices.</summary>
    public required uint[] TypeIndex { get; init; }

    /// <summary>Superclass string indices.</summary>
    public required uint[] SuperclassIndex { get; init; }

    /// <summary>Class string indices.</summary>
    public required uint[] ClassIndex { get; init; }

    /// <summary>Instance string indices.</summary>
    public required uint[] InstanceIndex { get; init; }

    /// <summary>FlyWire cross-reference type string indices.</summary>
    public required uint[] FlywireTypeIndex { get; init; }

    /// <summary>Hemibrain cross-reference type string indices.</summary>
    public required uint[] HemibrainTypeIndex { get; init; }

    /// <summary>Neurotransmitter per neuron.</summary>
    public required Neurotransmitter[] Neurotransmitters { get; init; }

    /// <summary>Side per neuron.</summary>
    public required Side[] Sides { get; init; }

    /// <summary>Soma coordinates, 3 ints per neuron, <see cref="int.MinValue"/> when unknown.</summary>
    public required int[] Soma { get; init; }

    /// <summary>CSR row pointers, length N+1.</summary>
    public required long[] RowPtr { get; init; }

    /// <summary>CSR column indices (postsynaptic neuron index), sorted ascending within each row.</summary>
    public required int[] Columns { get; init; }

    /// <summary>CSR weights: signed contact counts.</summary>
    public required short[] Weights { get; init; }

    /// <summary>Named neuron sets.</summary>
    public required IReadOnlyList<NeuronSet> Sets { get; init; }

    /// <summary>Number of neurons.</summary>
    public int NeuronCount => BodyIds.Length;

    /// <summary>Number of directed edges.</summary>
    public long EdgeCount => Columns.LongLength;

    /// <summary>Validates internal consistency; throws on the first violation.</summary>
    public void Validate()
    {
        int n = NeuronCount;
        Check(TypeIndex.Length == n, "TypeIndex length");
        Check(SuperclassIndex.Length == n, "SuperclassIndex length");
        Check(ClassIndex.Length == n, "ClassIndex length");
        Check(InstanceIndex.Length == n, "InstanceIndex length");
        Check(FlywireTypeIndex.Length == n, "FlywireTypeIndex length");
        Check(HemibrainTypeIndex.Length == n, "HemibrainTypeIndex length");
        Check(Neurotransmitters.Length == n, "Neurotransmitters length");
        Check(Sides.Length == n, "Sides length");
        Check(Soma.Length == 3 * n, "Soma length");
        Check(RowPtr.Length == n + 1, "RowPtr length");
        Check(RowPtr[0] == 0 && RowPtr[n] == Columns.LongLength, "RowPtr bounds");
        Check(Weights.Length == Columns.Length, "Weights length");

        for (int i = 0; i < n; i++)
        {
            Check(RowPtr[i] <= RowPtr[i + 1], $"RowPtr monotonic at {i}");
            int prev = -1;
            for (long e = RowPtr[i]; e < RowPtr[i + 1]; e++)
            {
                int c = Columns[e];
                Check((uint)c < (uint)n, $"column out of range at edge {e}");
                Check(c > prev, $"columns not strictly ascending in row {i}");
                Check(Weights[e] != 0, $"zero weight at edge {e}");
                prev = c;
            }
        }

        foreach (NeuronSet s in Sets)
        {
            Check(s.Count == 0 || s[s.Count - 1] < n, $"set {s.Name} index out of range");
        }

        static void Check(bool ok, string what)
        {
            if (!ok)
            {
                throw new InvalidDataException($"Checkpoint content invalid: {what}.");
            }
        }
    }
}
