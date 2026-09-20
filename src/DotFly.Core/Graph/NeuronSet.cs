using System.Collections;

namespace DotFly.Core.Graph;

/// <summary>
/// An immutable, sorted, duplicate-free set of neuron indices with a name. Every input, output,
/// silence mask and query in dotFly resolves to one of these.
/// </summary>
public sealed class NeuronSet : IReadOnlyList<int>
{
    private readonly int[] _indices;

    /// <summary>Set name (a shipped set name, a query description, or user-supplied).</summary>
    public string Name { get; }

    /// <summary>Number of neurons.</summary>
    public int Count => _indices.Length;

    /// <summary>The sorted neuron indices.</summary>
    public ReadOnlySpan<int> Indices => _indices;

    /// <inheritdoc />
    public int this[int position] => _indices[position];

    private NeuronSet(string name, int[] sortedUnique)
    {
        Name = name;
        _indices = sortedUnique;
    }

    /// <summary>The empty set.</summary>
    public static NeuronSet Empty { get; } = new("empty", []);

    /// <summary>Creates a set from arbitrary indices (sorted and de-duplicated).</summary>
    public static NeuronSet From(string name, ReadOnlySpan<int> indices)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        if (indices.IsEmpty)
        {
            return new NeuronSet(name, []);
        }

        int[] copy = indices.ToArray();
        Array.Sort(copy);
        int w = 1;
        for (int r = 1; r < copy.Length; r++)
        {
            if (copy[r] != copy[w - 1])
            {
                copy[w++] = copy[r];
            }
        }

        if (w != copy.Length)
        {
            Array.Resize(ref copy, w);
        }

        if (copy[0] < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(indices), "Neuron indices must be non-negative.");
        }

        return new NeuronSet(name, copy);
    }

    /// <summary>Creates a set from indices already known to be sorted and unique (no copy validation beyond a debug check).</summary>
    internal static NeuronSet FromSortedUnique(string name, int[] sortedUnique)
    {
        System.Diagnostics.Debug.Assert(IsSortedUnique(sortedUnique));
        return new NeuronSet(name, sortedUnique);
    }

    /// <summary>Returns a copy of this set under a different name.</summary>
    public NeuronSet Rename(string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        return new NeuronSet(name, _indices);
    }

    /// <summary>Whether the set contains the given neuron index (binary search).</summary>
    public bool Contains(int neuron) => Array.BinarySearch(_indices, neuron) >= 0;

    /// <summary>Union.</summary>
    public static NeuronSet operator |(NeuronSet a, NeuronSet b)
    {
        var result = new List<int>(a.Count + b.Count);
        int i = 0, j = 0;
        while (i < a.Count && j < b.Count)
        {
            int x = a._indices[i], y = b._indices[j];
            if (x < y)
            {
                result.Add(x);
                i++;
            }
            else if (y < x)
            {
                result.Add(y);
                j++;
            }
            else
            {
                result.Add(x);
                i++;
                j++;
            }
        }

        result.AddRange(a._indices.AsSpan(i));
        result.AddRange(b._indices.AsSpan(j));
        return new NeuronSet($"({a.Name} | {b.Name})", result.ToArray());
    }

    /// <summary>Intersection.</summary>
    public static NeuronSet operator &(NeuronSet a, NeuronSet b)
    {
        var result = new List<int>(Math.Min(a.Count, b.Count));
        int i = 0, j = 0;
        while (i < a.Count && j < b.Count)
        {
            int x = a._indices[i], y = b._indices[j];
            if (x < y)
            {
                i++;
            }
            else if (y < x)
            {
                j++;
            }
            else
            {
                result.Add(x);
                i++;
                j++;
            }
        }

        return new NeuronSet($"({a.Name} & {b.Name})", result.ToArray());
    }

    /// <summary>Difference (elements of <paramref name="a"/> not in <paramref name="b"/>).</summary>
    public static NeuronSet operator -(NeuronSet a, NeuronSet b)
    {
        var result = new List<int>(a.Count);
        int j = 0;
        foreach (int x in a._indices)
        {
            while (j < b.Count && b._indices[j] < x)
            {
                j++;
            }

            if (j >= b.Count || b._indices[j] != x)
            {
                result.Add(x);
            }
        }

        return new NeuronSet($"({a.Name} - {b.Name})", result.ToArray());
    }

    /// <inheritdoc />
    public IEnumerator<int> GetEnumerator() => ((IEnumerable<int>)_indices).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <inheritdoc />
    public override string ToString() => $"{Name} ({Count} neurons)";

    private static bool IsSortedUnique(int[] a)
    {
        for (int i = 1; i < a.Length; i++)
        {
            if (a[i] <= a[i - 1])
            {
                return false;
            }
        }

        return a.Length == 0 || a[0] >= 0;
    }
}
