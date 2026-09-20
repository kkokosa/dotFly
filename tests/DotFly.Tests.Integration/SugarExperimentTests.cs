using System.Diagnostics;
using DotFly.Core.Backends;
using DotFly.Core.Graph;
using DotFly.Cpu.Fast;
using DotFly.Cpu.Reference;

namespace DotFly.Tests.Integration;

/// <summary>
/// The Shiu et al. example experiment on the full v630 graph with the reference backend: 21 sugar
/// GRNs driven at 150 Hz for one second. This is a smoke test of the full-graph path (MN9 must
/// respond); the statistical comparison with the published 30-trial results is M3 work.
/// </summary>
public sealed class SugarExperimentTests
{
    private static readonly ulong[] SugarV630 =
    [
        720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
        720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
        720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
        720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
        720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
        720575940611875570,
    ];

    private const ulong Mn9V630 = 720575940660219265;

    private sealed class Counter(int n) : ISpikeSink
    {
        public int[] Counts { get; } = new int[n];

        public long Total { get; private set; }

        public void OnStep(long step, ReadOnlySpan<int> spiked)
        {
            foreach (int i in spiked)
            {
                Counts[i]++;
            }

            Total += spiked.Length;
        }
    }

    [Fact(Skip = "FlyWire v630 checkpoint not built in .data/", SkipUnless = nameof(DataPaths.HasFlyWireCheckpoint), SkipType = typeof(DataPaths))]
    public void Sugar_GRNs_At_150Hz_Drive_MN9_On_The_Full_V630_Graph()
    {
        using Brain b = Brain.Open(Path.Combine(DataPaths.Shiu!, "flywire-v630.dfb"));
        NeuronSet sugar = b.ByBodyIds("sugar", SugarV630);
        int mn9 = b.IndexOf(Mn9V630);
        Assert.True(mn9 >= 0);

        using var sim = new ReferenceBackend(b, seed: 1);
        foreach (int i in sugar.Indices)
        {
            sim.SetRefractorySteps(i, 0);                       // Shiu: Poisson targets have rfc = 0
            sim.SetPoissonProbability(i, 150.0 * 0.1e-3);
        }

        var counter = new Counter(b.NeuronCount);
        var sw = Stopwatch.StartNew();
        sim.Advance(10_000, counter);                          // 1 s of neural time
        sw.Stop();

        int active = counter.Counts.Count(c => c > 0);
        Console.Error.WriteLine($"v630 sugar @150 Hz, 1 s: {counter.Total:N0} spikes from {active:N0} neurons; MN9 = {counter.Counts[mn9]} Hz; wall {sw.Elapsed.TotalSeconds:F1} s (reference backend, scalar float64)");

        // Published example (sugarR.parquet, 30 trials): 448 distinct active neurons, MN9 ≈ tens of Hz.
        Assert.InRange(counter.Counts[mn9], 5, 200);
        Assert.InRange(active, 200, 1500);
    }

    [Fact(Skip = "FlyWire v630 checkpoint not built in .data/", SkipUnless = nameof(DataPaths.HasFlyWireCheckpoint), SkipType = typeof(DataPaths))]
    public void Fast_Backend_Matches_Reference_On_The_Full_V630_Graph()
    {
        using Brain b = Brain.Open(Path.Combine(DataPaths.Shiu!, "flywire-v630.dfb"));
        NeuronSet sugar = b.ByBodyIds("sugar", SugarV630);
        int mn9 = b.IndexOf(Mn9V630);

        Counter Run(ISpikingBackend sim)
        {
            foreach (int i in sugar.Indices)
            {
                sim.SetRefractorySteps(i, 0);
                sim.SetPoissonProbability(i, 150.0 * 0.1e-3);
            }

            var c = new Counter(b.NeuronCount);
            sim.Advance(10_000, c);
            return c;
        }

        Counter reference, fast;
        var sw = Stopwatch.StartNew();
        using (var r = new ReferenceBackend(b, seed: 3))
        {
            reference = Run(r);
        }

        double refWall = sw.Elapsed.TotalSeconds;
        sw.Restart();
        using (var f = new CpuBackend(b, seed: 3))
        {
            fast = Run(f);
        }

        double fastWall = sw.Elapsed.TotalSeconds;

        int[] cr = reference.Counts, cf = fast.Counts;
        var activeR = Enumerable.Range(0, b.NeuronCount).Where(i => cr[i] > 0).ToHashSet();
        var activeF = Enumerable.Range(0, b.NeuronCount).Where(i => cf[i] > 0).ToHashSet();
        double jaccard = (double)activeR.Intersect(activeF).Count() / activeR.Union(activeF).Count();
        double ratio = (double)fast.Total / reference.Total;
        int[] union = activeR.Union(activeF).ToArray();
        double corr = Pearson(union.Select(i => (double)cr[i]).ToArray(), union.Select(i => (double)cf[i]).ToArray());
        Console.Error.WriteLine($"v630 sugar, same seed: reference {reference.Total:N0} spikes / MN9 {cr[mn9]} Hz in {refWall:F2} s; cpu {fast.Total:N0} / MN9 {cf[mn9]} Hz in {fastWall:F2} s; jaccard {jaccard:F3}, count ratio {ratio:F3}, rate corr {corr:F4}");

        Assert.True(jaccard > 0.9, $"jaccard {jaccard:F3}");
        Assert.InRange(ratio, 0.95, 1.05);
        Assert.True(corr > 0.98, $"rate correlation {corr:F4}");

        static double Pearson(double[] a, double[] c)
        {
            double ma = a.Average(), mc = c.Average(), num = 0, da = 0, dc = 0;
            for (int i = 0; i < a.Length; i++)
            {
                num += (a[i] - ma) * (c[i] - mc);
                da += (a[i] - ma) * (a[i] - ma);
                dc += (c[i] - mc) * (c[i] - mc);
            }

            return num / Math.Sqrt(da * dc);
        }
    }
}
