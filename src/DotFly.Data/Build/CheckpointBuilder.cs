using DotFly.Core.Checkpoints;
using DotFly.Core.Graph;
using DotFly.Core.Models;

namespace DotFly.Data.Build;

/// <summary>
/// Assembles neurons, edges and named sets into a validated <see cref="CheckpointContent"/>:
/// merges duplicate (pre, post) pairs by summing weights, drops zero-weight edges, sorts columns
/// within rows and packs weights as <see cref="short"/>.
/// </summary>
public static class CheckpointBuilder
{
    /// <summary>Builds checkpoint content.</summary>
    /// <param name="provenance">Provenance record.</param>
    /// <param name="neurons">Neurons in index order.</param>
    /// <param name="edges">Edges in neuron indices with signed contact counts.</param>
    /// <param name="sets">Named sets over the same indices.</param>
    /// <exception cref="OverflowException">A merged weight does not fit in 16 bits.</exception>
    public static CheckpointContent Build(
        Provenance provenance,
        IReadOnlyList<NeuronRecord> neurons,
        EdgeList edges,
        IReadOnlyList<NeuronSet> sets)
    {
        ArgumentNullException.ThrowIfNull(provenance);
        ArgumentNullException.ThrowIfNull(neurons);
        ArgumentNullException.ThrowIfNull(edges);
        ArgumentNullException.ThrowIfNull(sets);

        int n = neurons.Count;
        var strings = new StringTable();
        var bodyIds = new ulong[n];
        var type = new uint[n];
        var superclass = new uint[n];
        var cls = new uint[n];
        var instance = new uint[n];
        var fwType = new uint[n];
        var hbType = new uint[n];
        var nt = new Neurotransmitter[n];
        var side = new Side[n];
        var soma = new int[3 * n];
        var seen = new HashSet<ulong>(n);

        for (int i = 0; i < n; i++)
        {
            NeuronRecord r = neurons[i];
            if (!seen.Add(r.BodyId))
            {
                throw new InvalidDataException($"Duplicate body ID {r.BodyId}.");
            }

            bodyIds[i] = r.BodyId;
            type[i] = strings.Intern(r.Type);
            superclass[i] = strings.Intern(r.Superclass);
            cls[i] = strings.Intern(r.Class);
            instance[i] = strings.Intern(r.Instance);
            fwType[i] = strings.Intern(r.FlywireType);
            hbType[i] = strings.Intern(r.HemibrainType);
            nt[i] = r.Neurotransmitter;
            side[i] = r.Side;
            if (r.Soma is { } s)
            {
                soma[3 * i] = s.X;
                soma[3 * i + 1] = s.Y;
                soma[3 * i + 2] = s.Z;
            }
            else
            {
                soma[3 * i] = soma[3 * i + 1] = soma[3 * i + 2] = int.MinValue;
            }
        }

        (long[] rowPtr, int[] cols, short[] weights) = ToCsr(n, edges);

        return new CheckpointContent
        {
            Provenance = provenance,
            Strings = strings,
            BodyIds = bodyIds,
            TypeIndex = type,
            SuperclassIndex = superclass,
            ClassIndex = cls,
            InstanceIndex = instance,
            FlywireTypeIndex = fwType,
            HemibrainTypeIndex = hbType,
            Neurotransmitters = nt,
            Sides = side,
            Soma = soma,
            RowPtr = rowPtr,
            Columns = cols,
            Weights = weights,
            Sets = sets,
        };
    }

    /// <summary>
    /// Converts an edge list to CSR: counting sort by pre, sort by post within row, merge
    /// duplicates by summing, drop zeros, pack to 16 bits.
    /// </summary>
    internal static (long[] RowPtr, int[] Columns, short[] Weights) ToCsr(int n, EdgeList edges)
    {
        ReadOnlySpan<int> pre = edges.Pre;
        ReadOnlySpan<int> post = edges.Post;
        ReadOnlySpan<int> w = edges.Weight;
        int m = edges.Count;

        var degree = new long[n + 1];
        for (int e = 0; e < m; e++)
        {
            if ((uint)pre[e] >= (uint)n || (uint)post[e] >= (uint)n)
            {
                throw new InvalidDataException($"Edge {e} references neuron index out of range ({pre[e]} → {post[e]}, N = {n}).");
            }

            degree[pre[e] + 1]++;
        }

        for (int i = 0; i < n; i++)
        {
            degree[i + 1] += degree[i];
        }

        // Scatter into row order.
        var fill = (long[])degree.Clone();
        var tmpCol = new int[m];
        var tmpW = new int[m];
        for (int e = 0; e < m; e++)
        {
            long pos = fill[pre[e]]++;
            tmpCol[pos] = post[e];
            tmpW[pos] = w[e];
        }

        // Sort within rows, merge duplicates, drop zeros.
        var rowPtr = new long[n + 1];
        var colOut = new List<int>(m);
        var wOut = new List<short>(m);
        for (int i = 0; i < n; i++)
        {
            int start = (int)degree[i];
            int len = (int)(degree[i + 1] - degree[i]);
            if (len > 1)
            {
                Array.Sort(tmpCol, tmpW, start, len);
            }

            int e = start;
            while (e < start + len)
            {
                int c = tmpCol[e];
                long sum = 0;
                while (e < start + len && tmpCol[e] == c)
                {
                    sum += tmpW[e++];
                }

                if (sum != 0)
                {
                    if (sum is > short.MaxValue or < short.MinValue)
                    {
                        throw new OverflowException($"Merged weight {sum} for edge {i} → {c} does not fit in 16 bits.");
                    }

                    colOut.Add(c);
                    wOut.Add((short)sum);
                }
            }

            rowPtr[i + 1] = colOut.Count;
        }

        return (rowPtr, colOut.ToArray(), wOut.ToArray());
    }
}
