using DotFly.Core.Models;

namespace DotFly.Tests.Unit.Models;

public sealed class NeuronModelTests
{
    [Fact]
    public void Defaults_Match_Shiu2024()
    {
        NeuronModel m = NeuronModel.Shiu2024;

        Assert.Equal(-52.0, m.RestMv);
        Assert.Equal(-52.0, m.ResetMv);
        Assert.Equal(-45.0, m.ThresholdMv);
        Assert.Equal(20.0, m.MembraneTauMs);
        Assert.Equal(5.0, m.SynapticTauMs);
        Assert.Equal(2.2, m.RefractoryMs);
        Assert.Equal(1.8, m.DelayMs);
        Assert.Equal(0.275, m.SynapseWeightMv);
        Assert.Equal(68.75, m.PoissonWeightMv);
    }

    [Fact]
    public void Discretize_Produces_Exact_Integrator_Constants_At_0_1_Ms()
    {
        LifStepConstants k = NeuronModel.Shiu2024.Discretize(0.1);

        Assert.Equal(Math.Exp(-0.1 / 20.0), k.A, 15);
        Assert.Equal(Math.Exp(-0.1 / 5.0), k.C, 15);
        Assert.Equal(5.0 / (5.0 - 20.0) * (k.C - k.A), k.B, 15);

        // Magnitudes quoted in PLAN.md §2.
        Assert.Equal(0.995012, k.A, 6);
        Assert.Equal(0.980199, k.C, 6);
        Assert.Equal(0.00493794, k.B, 8);

        Assert.Equal(22, k.RefractorySteps);
        Assert.Equal(18, k.DelaySteps);
        Assert.Equal(19, k.RingSlots);
    }

    [Fact]
    public void Discretize_Matches_Brute_Force_Matrix_Exponential()
    {
        // Integrate the ODE with a tiny Euler step over one 0.1 ms interval and compare with
        // the closed form; tolerance reflects the Euler error at that sub-step.
        var m = NeuronModel.Shiu2024;
        LifStepConstants k = m.Discretize(0.1);

        double v = -48.0, g = 3.0;
        const int subSteps = 100_000;
        double h = 0.1 / subSteps;
        for (int i = 0; i < subSteps; i++)
        {
            double dv = (m.RestMv - v + g) / m.MembraneTauMs;
            double dg = -g / m.SynapticTauMs;
            v += h * dv;
            g += h * dg;
        }

        double vExact = k.RestMv + k.A * (-48.0 - k.RestMv) + k.B * 3.0;
        double gExact = k.C * 3.0;

        Assert.Equal(vExact, v, 7);
        Assert.Equal(gExact, g, 7);
    }

    [Theory]
    [InlineData(2.2, 0.1, 22)]
    [InlineData(1.8, 0.1, 18)]
    [InlineData(0.3, 0.1, 3)]   // 0.3/0.1 = 2.9999999999999996 in floating point; Brian2's epsilon fixes it
    [InlineData(2.25, 0.1, 22)] // floor, not round
    public void BrianTimestep_Uses_Floor_With_Epsilon(double t, double dt, int expected)
    {
        Assert.Equal(expected, NeuronModel.BrianTimestep(t, dt));
    }

    [Theory]
    [InlineData(1.8, 0.1, 18)]
    [InlineData(1.85, 0.1, 19)]
    [InlineData(0.04, 0.1, 0)]
    public void BrianDelaySteps_Rounds_To_Nearest(double delay, double dt, int expected)
    {
        Assert.Equal(expected, NeuronModel.BrianDelaySteps(delay, dt));
    }

    [Fact]
    public void Discretize_Rejects_Equal_Time_Constants()
    {
        var m = new NeuronModel { MembraneTauMs = 5.0, SynapticTauMs = 5.0 };
        Assert.Throws<ArgumentOutOfRangeException>(() => m.Discretize(0.1));
    }

    [Fact]
    public void Discretize_Rejects_NonPositive_Dt()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => NeuronModel.Shiu2024.Discretize(0.0));
    }
}
