using System.Runtime.InteropServices;
using System.Text.Json;
using DotFly.Core.Backends;
using DotFly.Core.Models;
using DotFly.Cpu.Reference;
using DotFly.Data.Build;

namespace DotFly.Tests.Unit.Reference;

/// <summary>
/// Replays fixtures produced by <c>tools/brian2_golden.py</c> (Brian2 running the Shiu et al.
/// equations on a sub-graph with explicit input events) and requires the reference backend to
/// reproduce every spike step and the recorded v/g traces to floating-point rounding.
/// </summary>
public sealed class Brian2GoldenTests
{
    /// <summary>Tolerance in mV. Brian2 arranges the closed-form update differently, so the last few ulps differ.</summary>
    private const double ToleranceMv = 1e-11;

    private sealed record Fixture(
        string Name,
        double DtMs,
        int Steps,
        string Brian2,
        string StateUpdaterCode,
        long[] BodyIds,
        int[][] Edges,
        int[][] Injections,
        int[] RefractoryZero,
        int[] Recorded,
        int[][] Spikes);

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

    public static TheoryData<string> Fixtures => new() { "chain3", "sugar_2hop", "sugar_3hop" };

    [Theory]
    [MemberData(nameof(Fixtures))]
    public void Reference_Backend_Matches_Brian2(string name)
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Fixtures");
        Fixture f = JsonSerializer.Deserialize<Fixture>(File.ReadAllText(Path.Combine(dir, name + ".json")), new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;
        byte[] raw = File.ReadAllBytes(Path.Combine(dir, name + ".f64"));
        ReadOnlySpan<double> traces = MemoryMarshal.Cast<byte, double>(raw);
        int r = f.Recorded.Length;
        Assert.Equal(f.Steps * r * 2, traces.Length);

        using var t = TestBrains.Create(
            f.BodyIds.Length,
            f.Edges.Select(e => (e[0], e[1], e[2])),
            neuron: i => new NeuronRecord { BodyId = (ulong)f.BodyIds[i] });
        LifStepConstants k = NeuronModel.Shiu2024.Discretize(f.DtMs);
        using var sim = new ReferenceBackend(t.Brain, k, seed: 0);
        foreach (int i in f.RefractoryZero)
        {
            sim.SetRefractorySteps(i, 0);
        }

        foreach (int[] inj in f.Injections)
        {
            sim.ScheduleInjection(inj[0], inj[1], k.PoissonWeightMv);
        }

        var rec = new Recorder();
        double maxErr = 0;
        for (int step = 0; step < f.Steps; step++)
        {
            // Brian2's StateMonitor samples at the start of step `step` == dotFly state when Step == step.
            for (int j = 0; j < r; j++)
            {
                int neuron = f.Recorded[j];
                double v = traces[(step * r + j) * 2];
                double g = traces[(step * r + j) * 2 + 1];
                double ev = Math.Abs(sim.GetV(neuron) - v);
                double eg = Math.Abs(sim.GetG(neuron) - g);
                maxErr = Math.Max(maxErr, Math.Max(ev, eg));
                Assert.True(ev <= ToleranceMv, $"{name}: v mismatch at step {step}, neuron {neuron}: dotFly {sim.GetV(neuron):R} vs Brian2 {v:R}");
                Assert.True(eg <= ToleranceMv, $"{name}: g mismatch at step {step}, neuron {neuron}: dotFly {sim.GetG(neuron):R} vs Brian2 {g:R}");
            }

            sim.Advance(1, rec);
        }

        (int, int)[] expected = f.Spikes.Select(s => (s[0], s[1])).ToArray();
        Assert.Equal(expected, rec.Spikes.ToArray());
        Assert.True(maxErr < ToleranceMv, $"max abs error {maxErr:E2} mV");
    }
}
