using DotFly.Core.Backends;
using DotFly.Core.Models;
using DotFly.Cpu.Reference;

namespace DotFly.Tests.Unit.Reference;

/// <summary>
/// Hand-derived golden traces for the Brian2 step semantics (PLAN.md §2). Each test states the
/// expected numbers from the closed-form update so a regression here is a semantics change, not
/// a tolerance issue. Runs against every backend (see the derived classes).
/// </summary>
public abstract class BackendSemanticsTests
{
    private static readonly LifStepConstants K = NeuronModel.Shiu2024.Discretize(0.1);

    /// <summary>Tolerance for state comparisons: exact for float64, float32 rounding otherwise.</summary>
    protected abstract int Digits { get; }

    protected abstract ISpikingBackend Create(DotFly.Core.Graph.Brain brain, ulong seed);

    private sealed class Recorder : ISpikeSink
    {
        public List<(long Step, int Neuron)> Spikes { get; } = [];

        public void OnStep(long step, ReadOnlySpan<int> spiked)
        {
            foreach (int i in spiked)
            {
                Spikes.Add((step, i));
            }
        }
    }

    private ISpikingBackend CreateWith(DotFly.Core.Graph.Brain brain, ulong seed, bool recurrent)
    {
        ISpikingBackend b = Create(brain, seed);
        b.RecurrentTransmission = recurrent;
        return b;
    }

    [Fact]
    public void Idle_Network_Stays_At_Rest()
    {
        using var t = TestBrains.Create(3, [(0, 1, 5)]);
        using ISpikingBackend sim = Create(t.Brain, 1);
        var rec = new Recorder();
        sim.Advance(1000, rec);
        Assert.Empty(rec.Spikes);
        Assert.Equal(K.RestMv, sim.GetV(0), Digits);
        Assert.Equal(0.0, sim.GetG(0));
        Assert.Equal(1000, sim.Step);
    }

    [Fact]
    public void Injection_Fires_Next_Step_And_Reset_Zeroes_State()
    {
        using var t = TestBrains.Create(1, []);
        using ISpikingBackend sim = Create(t.Brain, 1);
        var rec = new Recorder();

        // Injection at step 5 lands after the threshold check of step 5 → spike at step 6.
        sim.ScheduleInjection(5, 0, K.PoissonWeightMv);
        sim.Advance(6, rec);
        Assert.Empty(rec.Spikes);
        // At the end of step 5: v = rest + 68.75 (injected after update).
        Assert.Equal(K.RestMv + K.PoissonWeightMv, sim.GetV(0), Digits);

        sim.Advance(1, rec);
        Assert.Equal([(6L, 0)], rec.Spikes);
        // Reset: v = reset, g = 0.
        Assert.Equal(K.ResetMv, sim.GetV(0), Digits);
        Assert.Equal(0.0, sim.GetG(0));
    }

    [Fact]
    public void Refractory_Freezes_For_R_Minus_One_Steps_Then_Integrates()
    {
        using var t = TestBrains.Create(1, []);
        using ISpikingBackend sim = Create(t.Brain, 1);
        var rec = new Recorder();

        sim.ScheduleInjection(0, 0, K.PoissonWeightMv);       // spike at step 1
        // Constant drive that would fire every step if not refractory: inject 68.75 mV every step.
        for (long s = 1; s < 60; s++)
        {
            sim.ScheduleInjection(s, 0, K.PoissonWeightMv);
        }

        sim.Advance(60, rec);
        long[] steps = rec.Spikes.Select(x => x.Step).ToArray();
        // Spike at 1; refractory at 1..22, where injections are discarded (Brian2 conditional write);
        // integrates at 23 (v = rest, no spike), accepts the injection at 23, fires at 24; period 23.
        Assert.Equal([1, 24, 47], steps);
    }

    [Fact]
    public void Zero_Refractory_Neuron_Fires_Every_Other_Step_Under_Constant_Injection()
    {
        // Brian2 order: update → threshold → synapses (incl. Poisson/injection) → reset. The event
        // that arrives in the same step as a spike is wiped by the reset, so even with rfc = 0 a
        // neuron driven every step fires every other step (5 kHz at dt = 0.1 ms).
        using var t = TestBrains.Create(1, []);
        using ISpikingBackend sim = Create(t.Brain, 1);
        sim.SetRefractorySteps(0, 0);                          // Shiu: Poisson targets have rfc = 0
        var rec = new Recorder();
        for (long s = 0; s < 10; s++)
        {
            sim.ScheduleInjection(s, 0, K.PoissonWeightMv);
        }

        sim.Advance(11, rec);
        Assert.Equal([1, 3, 5, 7, 9], rec.Spikes.Select(x => x.Step).ToArray());
    }

    [Fact]
    public void Delivery_Arrives_After_Exactly_DelaySteps_With_Signed_Weight()
    {
        // 0 → 1 excitatory (3 contacts), 0 → 2 inhibitory (−2 contacts).
        using var t = TestBrains.Create(3, [(0, 1, 3), (0, 2, -2)]);
        using ISpikingBackend sim = Create(t.Brain, 1);
        sim.ScheduleInjection(0, 0, K.PoissonWeightMv);       // neuron 0 spikes at step 1
        sim.Advance(1 + K.DelaySteps, null);                   // through step 1 + 18 = 19 exclusive → last step executed is 18
        Assert.Equal(0.0, sim.GetG(1));
        Assert.Equal(0.0, sim.GetG(2));

        sim.Advance(1, null);                                  // step 19 = 1 + 18: delivery
        Assert.Equal(3 * K.SynapseWeightMv, sim.GetG(1), Digits);
        Assert.Equal(-2 * K.SynapseWeightMv, sim.GetG(2), Digits);
        // v not yet affected at the delivery step (delivery happens after the update).
        Assert.Equal(K.RestMv, sim.GetV(1), Digits);

        sim.Advance(1, null);                                  // step 20: v integrates g, g decays
        Assert.Equal(K.RestMv + K.B * 3 * K.SynapseWeightMv, sim.GetV(1), Digits);
        Assert.Equal(K.C * 3 * K.SynapseWeightMv, sim.GetG(1), Digits);
    }

    [Fact]
    public void Same_Step_Delivery_To_A_Spiking_Neuron_Is_Discarded()
    {
        using var t = TestBrains.Create(2, [(0, 1, 10)]);
        using ISpikingBackend sim = Create(t.Brain, 1);
        sim.ScheduleInjection(0, 0, K.PoissonWeightMv);       // 0 spikes at 1 → delivery to 1 at step 19
        sim.ScheduleInjection(18, 1, K.PoissonWeightMv);      // 1 spikes at 19 — same step as the delivery
        var rec = new Recorder();
        sim.Advance(20, rec);
        Assert.Contains((19L, 1), rec.Spikes);
        Assert.Equal(0.0, sim.GetG(1));                        // discarded (refractory) and reset → g = 0
    }

    [Fact]
    public void Deliveries_During_Refractory_Period_Are_Discarded_Not_Accumulated()
    {
        // 0 → 2 and 1 → 2 (both 10 contacts). Neuron 2 spikes at step 5; 0 spikes at 1 (delivery at 19,
        // inside 2's refractory window 5..26) and 1 spikes at 10 (delivery at 28, accepted).
        using var t = TestBrains.Create(3, [(0, 2, 10), (1, 2, 10)]);
        using ISpikingBackend sim = Create(t.Brain, 1);
        sim.ScheduleInjection(0, 0, K.PoissonWeightMv);
        sim.ScheduleInjection(4, 2, K.PoissonWeightMv);
        sim.ScheduleInjection(9, 1, K.PoissonWeightMv);
        sim.Advance(28, null);                                 // steps 0..27 done
        Assert.Equal(0.0, sim.GetG(2));                        // the step-19 delivery was dropped
        sim.Advance(1, null);                                  // step 28: accepted
        Assert.Equal(10 * K.SynapseWeightMv, sim.GetG(2), Digits);
    }

    [Fact]
    public void Silenced_Neuron_Still_Spikes_But_Does_Not_Transmit()
    {
        using var t = TestBrains.Create(2, [(0, 1, 10)]);
        using ISpikingBackend sim = Create(t.Brain, 1);
        sim.SetTransmits(0, false);
        sim.ScheduleInjection(0, 0, K.PoissonWeightMv);
        var rec = new Recorder();
        sim.Advance(25, rec);
        Assert.Equal([(1L, 0)], rec.Spikes);
        Assert.Equal(0.0, sim.GetG(1));
    }

    [Fact]
    public void Global_Switches_Cut_Recurrence_And_External_Input()
    {
        using var t = TestBrains.Create(2, [(0, 1, 10)]);
        using ISpikingBackend sim = CreateWith(t.Brain, 1, recurrent: false);
        sim.ScheduleInjection(0, 0, K.PoissonWeightMv);
        sim.Advance(25, null);
        Assert.Equal(0.0, sim.GetG(1));

        sim.Reset();
        sim.RecurrentTransmission = true;
        sim.ExternalInput = false;
        sim.SetPoissonProbability(0, 1.0);
        sim.ScheduleInjection(0, 0, K.PoissonWeightMv);
        var rec = new Recorder();
        sim.Advance(25, rec);
        Assert.Empty(rec.Spikes);
    }

    [Fact]
    public void Poisson_Input_Is_Seed_Deterministic_And_Rate_Correct()
    {
        using var t = TestBrains.Create(1, []);
        var rec1 = new Recorder();
        using (ISpikingBackend sim = Create(t.Brain, 7))
        {
            sim.SetRefractorySteps(0, 0);
            sim.SetPoissonProbability(0, 150.0 * 0.1e-3);       // 150 Hz at dt = 0.1 ms
            sim.Advance(100_000, rec1);                          // 10 s
        }

        var rec2 = new Recorder();
        using (ISpikingBackend sim = Create(t.Brain, 7))
        {
            sim.SetRefractorySteps(0, 0);
            sim.SetPoissonProbability(0, 150.0 * 0.1e-3);
            sim.Advance(100_000, rec2);
        }

        Assert.Equal(rec1.Spikes, rec2.Spikes);
        // Expect ≈ 1500 events; 5σ ≈ 194.
        Assert.InRange(rec1.Spikes.Count, 1300, 1700);

        var rec3 = new Recorder();
        using (ISpikingBackend sim = Create(t.Brain, 8))
        {
            sim.SetRefractorySteps(0, 0);
            sim.SetPoissonProbability(0, 150.0 * 0.1e-3);
            sim.Advance(100_000, rec3);
        }

        Assert.NotEqual(rec1.Spikes, rec3.Spikes);
    }

    [Fact]
    public void Single_Spike_Needs_About_162_Contacts_To_Fire_A_Resting_Neuron()
    {
        Assert.False(FiresWithContacts(160));
        Assert.True(FiresWithContacts(164));

        bool FiresWithContacts(int contacts)
        {
            using var t = TestBrains.Create(2, [(0, 1, contacts)]);
            using ISpikingBackend sim = Create(t.Brain, 1);
            sim.ScheduleInjection(0, 0, K.PoissonWeightMv);
            var rec = new Recorder();
            sim.Advance(400, rec);
            return rec.Spikes.Any(x => x.Neuron == 1);
        }
    }

    [Fact]
    public void Excitatory_Chain_Propagates_With_Delay()
    {
        // The impulse response of v to a jump g0 in g peaks at g0·τs/(τs−τm)·(e^{−t/τs} − e^{−t/τm})
        // ≈ 0.1575·g0 (t ≈ 9.2 ms), so one presynaptic spike needs ≈ 7 / (0.275·0.1575) ≈ 162
        // contacts to fire a resting neuron. Use 200: g0 = 55 mV, threshold crossed ≈ 3.5 ms later.
        using var t = TestBrains.Create(3, [(0, 1, 200), (1, 2, 200)]);
        using ISpikingBackend sim = Create(t.Brain, 1);
        sim.ScheduleInjection(0, 0, K.PoissonWeightMv);
        var rec = new Recorder();
        sim.Advance(200, rec);

        // g jumps at 19; v then rises over a few steps until it crosses −45 mV.
        (long, int)[] s = rec.Spikes.ToArray();
        Assert.Equal((1L, 0), s[0]);
        Assert.Equal(1, s[1].Item2);
        Assert.Equal(2, s[2].Item2);
        Assert.True(s[1].Item1 > 19 && s[1].Item1 < 100, $"neuron 1 spiked at {s[1].Item1}");
        Assert.Equal(s[1].Item1 + (s[1].Item1 - 1), s[2].Item1); // same latency for the second hop
    }
}

public sealed class ReferenceBackendSemanticsTests : BackendSemanticsTests
{
    protected override int Digits => 12;

    protected override ISpikingBackend Create(DotFly.Core.Graph.Brain brain, ulong seed) => new ReferenceBackend(brain, seed);
}

public sealed class CpuBackendSemanticsTests : BackendSemanticsTests
{
    protected override int Digits => 4;

    protected override ISpikingBackend Create(DotFly.Core.Graph.Brain brain, ulong seed) => new DotFly.Cpu.Fast.CpuBackend(brain, seed, threads: 3);
}
