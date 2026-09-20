using DotFly.Core.Graph;

namespace DotFly.Tests.Integration;

/// <summary>
/// The published counts from the research note, reproduced from the built checkpoints
/// (<c>dotfly build malecns</c> / <c>dotfly build flywire</c> must have been run into <c>.data/</c>).
/// </summary>
public sealed class CheckpointCountTests
{
    [Fact(Skip = "MaleCNS checkpoint not built in .data/", SkipUnless = nameof(DataPaths.HasMaleCnsCheckpoint), SkipType = typeof(DataPaths))]
    public void MaleCns_Superclass_Graph_Matches_The_Retained_Demo_Graph()
    {
        using Brain b = Brain.Open(Path.Combine(DataPaths.MaleCns!, "malecns-v1.0-superclass.dfb"));
        Assert.Equal(166_700, b.NeuronCount);
        Assert.Equal(25_582_938, b.Provenance.SourceEdges!.Edges);
        Assert.Equal(124_177_617, b.Provenance.SourceEdges.Contacts);
        Assert.Equal(b.Provenance.SourceEdges.Edges - b.Provenance.SourceEdges.ZeroSignEdges, b.EdgeCount);
        Assert.Equal(b.Provenance.SourceEdges.Contacts - b.Provenance.SourceEdges.ZeroSignContacts, b.ContactCount);

        // Giant Fiber → TTMn is the strongest output of DNp01 (escape circuit).
        int gf = b.IndexOf(10001);
        Assert.Equal("DNp01", b[gf].Type);
        EdgeSpan e = b.OutEdges(gf);
        int best = 0;
        for (int i = 1; i < e.Length; i++)
        {
            if (e.Weights[i] > e.Weights[best])
            {
                best = i;
            }
        }

        Assert.Equal("TTMn", b[e.Targets[best]].Type);
    }

    [Fact(Skip = "MaleCNS traced checkpoint not built in .data/", SkipUnless = nameof(HasTraced), SkipType = typeof(CheckpointCountTests))]
    public void MaleCns_Traced_Graph_Matches_The_Official_Export()
    {
        using Brain b = Brain.Open(Path.Combine(DataPaths.MaleCns!, "malecns-v1.0-traced.dfb"));
        Assert.Equal(165_122, b.NeuronCount);
        Assert.Equal(25_563_197, b.Provenance.SourceEdges!.Edges);
        Assert.Equal(124_025_046, b.Provenance.SourceEdges.Contacts);
    }

    public static bool HasTraced => DataPaths.MaleCns is not null && File.Exists(Path.Combine(DataPaths.MaleCns, "malecns-v1.0-traced.dfb"));

    [Fact(Skip = "FlyWire v630 checkpoint not built in .data/", SkipUnless = nameof(DataPaths.HasFlyWireCheckpoint), SkipType = typeof(DataPaths))]
    public void FlyWire_V630_Matches_Shiu_Inputs()
    {
        using Brain b = Brain.Open(Path.Combine(DataPaths.Shiu!, "flywire-v630.dfb"));
        Assert.Equal(127_400, b.NeuronCount);
        Assert.Equal(14_687_178, b.EdgeCount);
        // First row of the connectivity table: 720575940596125868 → 720575940640198784 (index 119081), weight 1.
        int pre = b.IndexOf(720575940596125868);
        Assert.Equal(0, pre);
        Assert.Equal(119081, b.IndexOf(720575940640198784));
        EdgeSpan e = b.OutEdges(0);
        int k = e.Targets.IndexOf(119081);
        Assert.True(k >= 0);
        Assert.Equal(1, e.Weights[k]);
    }
}
