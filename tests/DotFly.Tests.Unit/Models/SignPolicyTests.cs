using DotFly.Core.Models;

namespace DotFly.Tests.Unit.Models;

public sealed class SignPolicyTests
{
    [Theory]
    [InlineData(Neurotransmitter.Acetylcholine, 1)]
    [InlineData(Neurotransmitter.Gaba, -1)]
    [InlineData(Neurotransmitter.Glutamate, -1)]
    [InlineData(Neurotransmitter.Dopamine, 1)]
    [InlineData(Neurotransmitter.Octopamine, 1)]
    [InlineData(Neurotransmitter.Serotonin, 1)]
    [InlineData(Neurotransmitter.Histamine, 0)]
    [InlineData(Neurotransmitter.Unknown, 0)]
    public void Shiu2024_Signs(Neurotransmitter nt, int expected)
    {
        Assert.Equal(expected, SignPolicy.Shiu2024.Sign(nt));
    }

    [Fact]
    public void Custom_Rejects_Wrong_Length_And_Bad_Signs()
    {
        Assert.Throws<ArgumentException>(() => SignPolicy.Custom("x", new sbyte[7]));
        Assert.Throws<ArgumentException>(() => SignPolicy.Custom("x", [0, 2, 0, 0, 0, 0, 0, 0]));
    }
}
