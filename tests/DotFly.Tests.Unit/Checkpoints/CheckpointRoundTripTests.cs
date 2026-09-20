using DotFly.Core.Checkpoints;
using DotFly.Core.Graph;
using DotFly.Core.Models;
using DotFly.Data.Build;

namespace DotFly.Tests.Unit.Checkpoints;

public sealed class CheckpointRoundTripTests
{
    [Fact]
    public void Writes_And_Reads_Back_Everything()
    {
        var sets = new[] { NeuronSet.From("evens", [0, 2, 4]), NeuronSet.From("one", [1]) };
        using var t = TestBrains.Create(
            5,
            [(0, 1, 3), (0, 4, -2), (0, 2, 5), (2, 3, 1), (4, 0, -7)],
            sets,
            i => new NeuronRecord
            {
                BodyId = 720575940600000000UL + (ulong)i,   // FlyWire-scale IDs, beyond double precision
                Type = i == 0 ? "DNp01" : "KC",
                Superclass = "descending_neuron",
                Class = i % 2 == 0 ? "CX" : null,
                Instance = $"inst{i}",
                FlywireType = "fw",
                HemibrainType = i == 4 ? "Giant Fiber" : null,
                Neurotransmitter = i == 4 ? Neurotransmitter.Gaba : Neurotransmitter.Acetylcholine,
                Side = i % 2 == 0 ? Side.Left : Side.Right,
                Soma = i == 3 ? null : (i, 10 * i, 100 * i),
            });
        Brain b = t.Brain;

        Assert.Equal(5, b.NeuronCount);
        Assert.Equal(5, b.EdgeCount);
        Assert.Equal(18, b.ContactCount);
        Assert.Equal("test", b.Provenance.Name);
        Assert.Equal(NeuronModel.Shiu2024, b.Model);

        // Exact 64-bit IDs.
        Assert.Equal(720575940600000004UL, b.BodyIds[4]);
        Assert.Equal(4, b.IndexOf(720575940600000004UL));
        Assert.Equal(-1, b.IndexOf(42));

        // Columns sorted within row 0: targets 1, 2, 4.
        EdgeSpan e0 = b.OutEdges(0);
        Assert.Equal([1, 2, 4], e0.Targets.ToArray());
        Assert.Equal([3, 5, -2], e0.Weights.ToArray());
        Assert.Equal(0, b.OutEdges(1).Length);
        Assert.Equal([0], b.OutEdges(4).Targets.ToArray());

        NeuronInfo n4 = b[4];
        Assert.Equal("KC", n4.Type);
        Assert.Equal("Giant Fiber", n4.HemibrainType);
        Assert.Equal(Neurotransmitter.Gaba, n4.Neurotransmitter);
        Assert.Equal(Side.Left, n4.Side);
        Assert.Equal((4, 40, 400), n4.Soma);
        Assert.Equal(1, n4.OutDegree);
        Assert.Null(b[3].Soma);
        Assert.Equal(string.Empty, b[1].Class);

        Assert.Equal(2, b.Sets.Count);
        Assert.Equal([0, 2, 4], b.Sets["evens"].Indices.ToArray());
        Assert.Equal([1], b.Sets["one"].Indices.ToArray());
    }

    [Fact]
    public void Builder_Merges_Duplicates_And_Drops_Zero_Sum()
    {
        using var t = TestBrains.Create(3, [(0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 2, -4), (1, 0, 1)]);
        Assert.Equal(2, t.Brain.EdgeCount);
        Assert.Equal([1], t.Brain.OutEdges(0).Targets.ToArray());
        Assert.Equal([5], t.Brain.OutEdges(0).Weights.ToArray());
    }

    [Fact]
    public void Builder_Rejects_Weight_Overflow_And_Bad_Indices()
    {
        Assert.Throws<OverflowException>(() => TestBrains.Create(2, [(0, 1, 40000)]));
        Assert.Throws<InvalidDataException>(() => TestBrains.Create(2, [(0, 5, 1)]));
    }

    [Fact]
    public void Query_Matches_Exact_And_Glob()
    {
        using var t = TestBrains.Create(6, [], neuron: i => new NeuronRecord
        {
            BodyId = (ulong)i + 1,
            Type = i switch { 0 => "DNa02", 1 => "DNa02", 2 => "DNp09", 3 => "LC4", 4 => "LC9", _ => null },
            Side = i == 0 ? Side.Left : Side.Right,
            Neurotransmitter = i == 3 ? Neurotransmitter.Gaba : Neurotransmitter.Acetylcholine,
        });
        Brain b = t.Brain;

        Assert.Equal([0, 1], b.Query(type: "DNa02").Indices.ToArray());
        Assert.Equal([0], b.Query(type: "DNa02", side: Side.Left).Indices.ToArray());
        Assert.Equal([0, 1, 2], b.Query(type: "DN*").Indices.ToArray());
        Assert.Equal([3, 4], b.Query(type: "LC*").Indices.ToArray());
        Assert.Equal([2, 4], b.Query(type: "*9").Indices.ToArray());   // DNp09 and LC9
        Assert.Equal([0, 1], b.Query(type: "*a0*").Indices.ToArray());
        Assert.Equal([3], b.Query(nt: Neurotransmitter.Gaba).Indices.ToArray());
        Assert.Empty(b.Query(type: "nope"));
        Assert.Equal(6, b.Query().Count);
    }

    [Fact]
    public void Downstream_Walks_Hops_With_Min_Weight()
    {
        using var t = TestBrains.Create(5, [(0, 1, 5), (1, 2, 1), (2, 3, 9), (0, 4, 1)]);
        Brain b = t.Brain;
        NeuronSet seed = NeuronSet.From("seed", [0]);

        Assert.Equal([0], b.Downstream(seed, 0).Indices.ToArray());
        Assert.Equal([0, 1, 4], b.Downstream(seed, 1).Indices.ToArray());
        Assert.Equal([0, 1], b.Downstream(seed, 1, minWeight: 2).Indices.ToArray());
        Assert.Equal([0, 1, 2, 3, 4], b.Downstream(seed, 10).Indices.ToArray());
        Assert.Equal([0, 1], b.Downstream(seed, 10, minWeight: 2).Indices.ToArray());
    }

    [Fact]
    public void ByBodyIds_Fails_Loudly_On_Unknown_Id()
    {
        using var t = TestBrains.Create(2, []);
        Assert.Throws<KeyNotFoundException>(() => t.Brain.ByBodyIds("x", [1000UL, 999UL]));
        Assert.Equal([0, 1], t.Brain.ByBodyIds("x", [1001UL, 1000UL]).Indices.ToArray());
    }

    [Fact]
    public void Open_Rejects_Garbage()
    {
        string path = Path.Combine(Path.GetTempPath(), $"dotfly-bad-{Guid.NewGuid():N}.dfb");
        File.WriteAllBytes(path, new byte[128]);
        try
        {
            Assert.Throws<InvalidDataException>(() => DfbFile.Open(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void StringTable_Round_Trips()
    {
        var t = new StringTable();
        uint a = t.Intern("alpha");
        uint b = t.Intern("βeta");
        Assert.Equal(a, t.Intern("alpha"));
        Assert.Equal(0u, t.Intern(null));
        Assert.Equal(0u, t.Intern(string.Empty));

        StringTable r = StringTable.Deserialize(t.Serialize());
        Assert.Equal(3, r.Count);
        Assert.Equal("alpha", r[a]);
        Assert.Equal("βeta", r[b]);
        Assert.Equal(string.Empty, r[0]);
        Assert.Equal(b, r.TryGetIndex("βeta"));
    }
}
