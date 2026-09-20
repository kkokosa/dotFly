namespace DotFly;

/// <summary>Readable time literals for engine calls: <c>50.Ms()</c>, <c>1.5.Seconds()</c>.</summary>
public static class Units
{
    /// <summary>Milliseconds as a <see cref="TimeSpan"/>.</summary>
    public static TimeSpan Ms(this double value) => TimeSpan.FromMilliseconds(value);

    /// <summary>Milliseconds as a <see cref="TimeSpan"/>.</summary>
    public static TimeSpan Ms(this int value) => TimeSpan.FromMilliseconds(value);

    /// <summary>Seconds as a <see cref="TimeSpan"/>.</summary>
    public static TimeSpan Seconds(this double value) => TimeSpan.FromSeconds(value);

    /// <summary>Seconds as a <see cref="TimeSpan"/>.</summary>
    public static TimeSpan Seconds(this int value) => TimeSpan.FromSeconds(value);
}
