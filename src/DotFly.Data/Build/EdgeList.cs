namespace DotFly.Data.Build;

/// <summary>
/// A growable list of directed, weighted edges expressed in neuron indices. Weights are signed
/// contact counts. Parallel arrays keep memory tight for the 10⁷–10⁸ edges of a full connectome.
/// </summary>
public sealed class EdgeList
{
    private int[] _pre;
    private int[] _post;
    private int[] _weight;
    private int _count;

    /// <summary>Creates a list with the given initial capacity.</summary>
    public EdgeList(int capacity = 1 << 20)
    {
        _pre = new int[capacity];
        _post = new int[capacity];
        _weight = new int[capacity];
    }

    /// <summary>Number of edges.</summary>
    public int Count => _count;

    /// <summary>Presynaptic indices.</summary>
    public ReadOnlySpan<int> Pre => _pre.AsSpan(0, _count);

    /// <summary>Postsynaptic indices.</summary>
    public ReadOnlySpan<int> Post => _post.AsSpan(0, _count);

    /// <summary>Signed contact counts.</summary>
    public ReadOnlySpan<int> Weight => _weight.AsSpan(0, _count);

    /// <summary>Appends an edge.</summary>
    public void Add(int pre, int post, int weight)
    {
        if (_count == _pre.Length)
        {
            int cap = checked(_pre.Length * 2);
            Array.Resize(ref _pre, cap);
            Array.Resize(ref _post, cap);
            Array.Resize(ref _weight, cap);
        }

        _pre[_count] = pre;
        _post[_count] = post;
        _weight[_count] = weight;
        _count++;
    }
}
