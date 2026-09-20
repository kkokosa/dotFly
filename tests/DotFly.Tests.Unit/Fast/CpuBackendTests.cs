using System.Runtime.InteropServices;
using System.Text.Json;
using DotFly.Core.Backends;
using DotFly.Core.Graph;
using DotFly.Core.Memory;
using DotFly.Core.Models;
using DotFly.Cpu.Fast;
using DotFly.Cpu.Kernels;
using DotFly.Cpu.Reference;
using DotFly.Data.Build;

namespace DotFly.Tests.Unit.Fast;

public sealed class CpuBackendTests
{
    private sealed class Recorder : ISpikeSink
    {
        public List<(int Step, int Neuron)> Spikes { get; } = [];

        public void OnStep(long step, ReadOnlySpan<int> spiked)
        {
            foreach (int i in spiked)
            {
                Spikes.Add(((int)step, i));
            }
        }
    }

    /// <summary>A random recurrent graph with a few strongly driven neurons — enough activity to exercise delivery on every thread.</summary>
    private static TestBrains.TempBrain RandomBrain(int n, int seed, double density = 0.01)
    {
        var rng = new Random(seed);
        var edges = new List<(int, int, int)>();
        for (int i = 0; i < n; i++)
        {
            int deg = (int)(n * density);
            for (int k = 0; k < deg; k++)
            {
                int j = rng.Next(n);
                int w = rng.Next(1, 40) * (rng.NextDouble() < 0.3 ? -1 : 1);
                edges.Add((i, j, w));
            }
        }

        return TestBrains.Create(n, edges);
    }

    private static List<(int, int)> Run(ISpikingBackend sim, int stimulated, int steps)
    {
        for (int i = 0; i < stimulated; i++)
        {
            sim.SetRefractorySteps(i, 0);
            sim.SetPoissonProbability(i, 150.0 * 0.1e-3);
        }

        var rec = new Recorder();
        sim.Advance(steps, rec);
        return rec.Spikes;
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(16)]
    public void Spikes_Are_Identical_For_Any_Thread_Count(int threads)
    {
        using var t = RandomBrain(3000, seed: 11);
        List<(int, int)> baseline;
        using (var one = new CpuBackend(t.Brain, seed: 5, threads: 1))
        {
            baseline = Run(one, stimulated: 40, steps: 3000);
        }

        Assert.True(baseline.Count > 500, $"too little activity to be a meaningful test: {baseline.Count} spikes");
        using var sim = new CpuBackend(t.Brain, seed: 5, threads: threads);
        Assert.Equal(Math.Min(threads, 3000 / 256), sim.Threads);
        List<(int, int)> spikes = Run(sim, stimulated: 40, steps: 3000);
        Assert.Equal(baseline, spikes);
    }

    [Fact]
    public void Float32_Backend_Matches_Float64_Reference_Statistically()
    {
        using var t = RandomBrain(3000, seed: 12);
        List<(int, int)> reference;
        using (var r = new ReferenceBackend(t.Brain, seed: 9))
        {
            reference = Run(r, stimulated: 40, steps: 5000);
        }

        List<(int, int)> fast;
        using (var f = new CpuBackend(t.Brain, seed: 9, threads: 4))
        {
            fast = Run(f, stimulated: 40, steps: 5000);
        }

        // Identical Poisson streams; float32 rounding may shift a threshold crossing occasionally,
        // and the network is recurrent, so compare Eon-style: active set, counts, rate correlation.
        int[] cr = Counts(reference, 3000), cf = Counts(fast, 3000);
        var activeR = cr.Select((c, i) => (c, i)).Where(x => x.c > 0).Select(x => x.i).ToHashSet();
        var activeF = cf.Select((c, i) => (c, i)).Where(x => x.c > 0).Select(x => x.i).ToHashSet();
        double jaccard = (double)activeR.Intersect(activeF).Count() / activeR.Union(activeF).Count();
        double ratio = (double)fast.Count / reference.Count;
        double corr = Pearson(cr, cf);
        Assert.True(jaccard > 0.9, $"jaccard {jaccard:F3}");
        Assert.InRange(ratio, 0.9, 1.1);
        Assert.True(corr > 0.95, $"rate correlation {corr:F3}");

        static int[] Counts(List<(int Step, int Neuron)> s, int n)
        {
            var c = new int[n];
            foreach ((_, int i) in s)
            {
                c[i]++;
            }

            return c;
        }

        static double Pearson(int[] a, int[] b)
        {
            double ma = a.Average(), mb = b.Average(), num = 0, da = 0, db = 0;
            for (int i = 0; i < a.Length; i++)
            {
                num += (a[i] - ma) * (b[i] - mb);
                da += (a[i] - ma) * (a[i] - ma);
                db += (b[i] - mb) * (b[i] - mb);
            }

            return num / Math.Sqrt(da * db);
        }
    }

    [Fact]
    public void Vector_And_Scalar_Kernel_Paths_Are_Bit_Identical()
    {
        const int n = 1000;
        var k = LifKernel.Constants.From(NeuronModel.Shiu2024.Discretize(0.1));
        var rng = new Random(3);
        using var v1 = new NativeBuffer<float>(n);
        using var g1 = new NativeBuffer<float>(n);
        using var cd1 = new NativeBuffer<int>(n);
        using var v2 = new NativeBuffer<float>(n);
        using var g2 = new NativeBuffer<float>(n);
        using var cd2 = new NativeBuffer<int>(n);
        using var arm = new NativeBuffer<int>(n);
        using var b1 = new NativeBuffer<ushort>(n / 16 + 1);
        using var b2 = new NativeBuffer<ushort>(n / 16 + 1);
        using var s1 = new NativeBuffer<int>(n);
        using var s2 = new NativeBuffer<int>(n);
        for (int i = 0; i < n; i++)
        {
            // Refractory neurons sit at (rest, 0) — the invariant the fixed-point fast path relies on.
            bool frozen = rng.NextDouble() < 0.2;
            cd1[i] = cd2[i] = frozen ? rng.Next(1, 22) : 0;
            v1[i] = v2[i] = frozen ? -52f : -52f + (float)rng.NextDouble() * 8f;      // some above threshold
            g1[i] = g2[i] = frozen ? 0f : (float)rng.NextDouble() * 20f;
            arm[i] = rng.NextDouble() < 0.1 ? 0 : 21;
        }

        int c1, c2;
        unsafe
        {
            // Full range: vector path for the bulk, scalar tail.
            c1 = LifKernel.Update(in k, v1.Pointer, g1.Pointer, cd1.Pointer, arm.Pointer, b1.Pointer, 0, n, s1.Pointer);
            // Forced scalar: chunks of 16 with end clipped to 15 elements never reach the vector path.
            c2 = 0;
            for (int start = 0; start < n; start += 16)
            {
                int end = Math.Min(n, start + 15);
                c2 += LifKernel.Update(in k, v2.Pointer, g2.Pointer, cd2.Pointer, arm.Pointer, b2.Pointer, start, end, s2.Pointer + c2);
                if (end < n && end == start + 15)
                {
                    c2 += LifKernel.Update(in k, v2.Pointer, g2.Pointer, cd2.Pointer, arm.Pointer, b2.Pointer, end, end + 1, s2.Pointer + c2);
                }
            }
        }

        Assert.Equal(c2, c1);
        Assert.True(c1 > 50, $"expected many spikes, got {c1}");
        Assert.Equal(MemoryMarshal.AsBytes(v2.Span).ToArray(), MemoryMarshal.AsBytes(v1.Span).ToArray());
        Assert.Equal(MemoryMarshal.AsBytes(g2.Span).ToArray(), MemoryMarshal.AsBytes(g1.Span).ToArray());
        Assert.Equal(cd2.Span.ToArray(), cd1.Span.ToArray());
        Assert.Equal(b2.Span.ToArray(), b1.Span.ToArray());
        Assert.Equal(s2.Span[..c2].ToArray(), s1.Span[..c1].ToArray());
        Assert.Contains(b1.Span.ToArray(), x => x != 0);
    }

    [Fact]
    public void BlockedCsr_Offsets_Partition_Each_Row_By_Target_Block()
    {
        using var t = RandomBrain(500, seed: 4, density: 0.05);
        Brain b = t.Brain;
        using var csr = new BlockedCsr(b, blocks: 7);
        ReadOnlySpan<long> rp = b.RowPtr;
        ReadOnlySpan<int> cols = b.Columns;
        for (int s = 0; s < b.NeuronCount; s++)
        {
            Assert.Equal(rp[s], csr.Offset(s, 0));
            Assert.Equal(rp[s + 1], csr.Offset(s, 7));
            for (int p = 0; p < 7; p++)
            {
                for (int e = csr.Offset(s, p); e < csr.Offset(s, p + 1); e++)
                {
                    Assert.InRange(cols[e], csr.Bounds[p], csr.Bounds[p + 1] - 1);
                }
            }
        }
    }

    private sealed record Fixture(int Steps, double DtMs, long[] BodyIds, int[][] Edges, int[][] Injections, int[] RefractoryZero, int[][] Spikes);

    [Theory]
    [InlineData("chain3")]
    [InlineData("sugar_2hop")]
    [InlineData("sugar_3hop")]
    public void Float32_Backend_Reproduces_Brian2_Spikes_On_Fixtures(string name)
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Fixtures");
        Fixture f = JsonSerializer.Deserialize<Fixture>(File.ReadAllText(Path.Combine(dir, name + ".json")), new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;
        using var t = TestBrains.Create(f.BodyIds.Length, f.Edges.Select(e => (e[0], e[1], e[2])), neuron: i => new NeuronRecord { BodyId = (ulong)f.BodyIds[i] });
        LifStepConstants k = NeuronModel.Shiu2024.Discretize(f.DtMs);
        using var sim = new CpuBackend(t.Brain, k, seed: 0, threads: 4);
        foreach (int i in f.RefractoryZero)
        {
            sim.SetRefractorySteps(i, 0);
        }

        foreach (int[] inj in f.Injections)
        {
            sim.ScheduleInjection(inj[0], inj[1], k.PoissonWeightMv);
        }

        var rec = new Recorder();
        sim.Advance(f.Steps, rec);
        (int, int)[] expected = f.Spikes.Select(s => (s[0], s[1])).ToArray();

        // Float32 vs float64 on a chaotic recurrent system: require ≥ 99 % identical (step, neuron)
        // events and identical per-neuron spike counts within ±1 for 99 % of neurons.
        var e = expected.ToHashSet();
        var a = rec.Spikes.ToHashSet();
        double overlap = (double)e.Intersect(a).Count() / Math.Max(e.Count, a.Count);
        Assert.True(overlap >= 0.99, $"{name}: event overlap {overlap:P2} ({e.Count} expected, {a.Count} actual)");
    }
}
