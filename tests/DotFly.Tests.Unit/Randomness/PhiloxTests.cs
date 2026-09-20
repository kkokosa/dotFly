using DotFly.Core.Randomness;

namespace DotFly.Tests.Unit.Randomness;

public sealed class PhiloxTests
{
    [Fact]
    public void Known_Answer_Test_Vector()
    {
        // Random123 philox4x32-10 KAT: counter = 0, key = 0.
        (uint r0, uint r1, uint r2, uint r3) = Philox4x32.Generate(0, 0, 0, 0, 0, 0);
        Assert.Equal(0x6627e8d5u, r0);
        Assert.Equal(0xe169c58du, r1);
        Assert.Equal(0xbc57ac4cu, r2);
        Assert.Equal(0x9b00dbd8u, r3);

        // counter = key = all ones.
        (r0, r1, r2, r3) = Philox4x32.Generate(uint.MaxValue, uint.MaxValue, uint.MaxValue, uint.MaxValue, uint.MaxValue, uint.MaxValue);
        Assert.Equal(0x408f276du, r0);
        Assert.Equal(0x41c83b0eu, r1);
        Assert.Equal(0xa20bc7c6u, r2);
        Assert.Equal(0x6d5451fdu, r3);
    }

    [Fact]
    public void Uniform_Is_Deterministic_And_In_Range()
    {
        double u1 = Philox4x32.Uniform(42, 7, 12345, 0);
        double u2 = Philox4x32.Uniform(42, 7, 12345, 0);
        Assert.Equal(u1, u2);
        Assert.NotEqual(u1, Philox4x32.Uniform(43, 7, 12345, 0));
        Assert.NotEqual(u1, Philox4x32.Uniform(42, 8, 12345, 0));
        Assert.NotEqual(u1, Philox4x32.Uniform(42, 7, 12346, 0));
        Assert.NotEqual(u1, Philox4x32.Uniform(42, 7, 12345, 1));

        double sum = 0;
        const int n = 200_000;
        for (uint i = 0; i < n; i++)
        {
            double u = Philox4x32.Uniform(1, 0, i, 0);
            Assert.InRange(u, 0.0, 0.999999999999999);
            sum += u;
        }

        Assert.InRange(sum / n, 0.495, 0.505);
    }
}
