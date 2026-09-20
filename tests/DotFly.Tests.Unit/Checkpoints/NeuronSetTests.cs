using DotFly.Core.Graph;

namespace DotFly.Tests.Unit.Checkpoints;

public sealed class NeuronSetTests
{
    [Fact]
    public void From_Sorts_And_Deduplicates()
    {
        NeuronSet s = NeuronSet.From("s", [5, 1, 3, 1, 5, 0]);
        Assert.Equal([0, 1, 3, 5], s.Indices.ToArray());
        Assert.True(s.Contains(3));
        Assert.False(s.Contains(2));
        Assert.Equal(4, s.Count);
    }

    [Fact]
    public void Set_Algebra()
    {
        NeuronSet a = NeuronSet.From("a", [1, 2, 3, 4]);
        NeuronSet b = NeuronSet.From("b", [3, 4, 5]);
        Assert.Equal([1, 2, 3, 4, 5], (a | b).Indices.ToArray());
        Assert.Equal([3, 4], (a & b).Indices.ToArray());
        Assert.Equal([1, 2], (a - b).Indices.ToArray());
        Assert.Equal([5], (b - a).Indices.ToArray());
        Assert.Equal("(a | b)", (a | b).Name);
        Assert.Empty(NeuronSet.Empty | NeuronSet.Empty);
    }

    [Fact]
    public void Rejects_Negative_Indices()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => NeuronSet.From("x", [-1, 2]));
    }
}
