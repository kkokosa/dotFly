using DotFly.Core.Memory;

namespace DotFly.Tests.Unit.Memory;

public sealed class NativeBufferTests
{
    [Fact]
    public unsafe void Allocates_Aligned_And_Zeroed()
    {
        using var buf = new NativeBuffer<float>(1000);

        Assert.Equal(1000, buf.Length);
        Assert.True((nuint)buf.Pointer % NativeBuffer<float>.Alignment == 0);
        foreach (float f in buf.Span)
        {
            Assert.Equal(0f, f);
        }
    }

    [Fact]
    public void Indexer_And_Span_Alias_The_Same_Memory()
    {
        using var buf = new NativeBuffer<int>(16);
        buf[3] = 42;
        Assert.Equal(42, buf.Span[3]);
        buf.Span[5] = 7;
        Assert.Equal(7, buf[5]);
    }

    [Fact]
    public unsafe void Zero_Length_Is_Allowed()
    {
        using var buf = new NativeBuffer<double>(0);
        Assert.Equal(0, buf.Length);
        Assert.True(buf.Pointer != null);
        Assert.True(buf.Span.IsEmpty);
    }

    [Fact]
    public void Dispose_Is_Idempotent()
    {
        var buf = new NativeBuffer<byte>(8);
        buf.Dispose();
        buf.Dispose();
    }
}
