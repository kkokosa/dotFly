using System.Diagnostics;

namespace DotFly;

/// <summary>
/// Keeps neural time, simulated duration and wall time apart. Wall time accumulates only while the
/// simulation is advancing, so the real-time factor measures the engine, not the caller's pauses.
/// </summary>
public sealed class SimulationClock
{
    private readonly double _dtMs;
    private readonly Stopwatch _wall = new();
    private long _spikes;
    private long _windowSteps;
    private long _windowSpikes;
    private long _windowWallTicks;

    internal SimulationClock(double dtMs)
    {
        _dtMs = dtMs;
    }

    /// <summary>Steps completed since the last reset.</summary>
    public long Steps { get; private set; }

    /// <summary>Neural time elapsed since the last reset (steps × dt).</summary>
    public TimeSpan NeuralTime => TimeSpan.FromMilliseconds(Steps * _dtMs);

    /// <summary>Wall-clock time spent advancing since the last reset.</summary>
    public TimeSpan WallTime => _wall.Elapsed;

    /// <summary>Total spikes since the last reset.</summary>
    public long Spikes => _spikes;

    /// <summary>Neural seconds per wall second since the last reset (∞ if no wall time yet).</summary>
    public double RealTimeFactor => _wall.Elapsed.TotalSeconds > 0 ? NeuralTime.TotalSeconds / _wall.Elapsed.TotalSeconds : double.PositiveInfinity;

    /// <summary>Steps per wall second since the last reset.</summary>
    public double StepsPerSecond => _wall.Elapsed.TotalSeconds > 0 ? Steps / _wall.Elapsed.TotalSeconds : 0;

    /// <summary>Spikes per neural second over the most recent <see cref="Sample"/> window.</summary>
    public double RecentSpikesPerSecond { get; private set; }

    /// <summary>Real-time factor over the most recent <see cref="Sample"/> window.</summary>
    public double RecentRealTimeFactor { get; private set; }

    internal void Begin() => _wall.Start();

    internal void End(int steps, long spikes)
    {
        _wall.Stop();
        Steps += steps;
        _spikes += spikes;
        _windowSteps += steps;
        _windowSpikes += spikes;
    }

    internal void MarkWindow(long wallTicksNow)
    {
        long wallDelta = wallTicksNow - _windowWallTicks;
        if (_windowSteps > 0 && wallDelta > 0)
        {
            double neuralSeconds = _windowSteps * _dtMs / 1000.0;
            RecentSpikesPerSecond = _windowSpikes / neuralSeconds;
            RecentRealTimeFactor = neuralSeconds / (wallDelta / (double)Stopwatch.Frequency);
        }

        _windowSteps = 0;
        _windowSpikes = 0;
        _windowWallTicks = wallTicksNow;
    }

    /// <summary>Closes the current statistics window: <see cref="RecentSpikesPerSecond"/> and <see cref="RecentRealTimeFactor"/> describe everything since the previous call.</summary>
    public void Sample() => MarkWindow(Stopwatch.GetTimestamp());

    internal void Reset()
    {
        _wall.Reset();
        Steps = 0;
        _spikes = 0;
        _windowSteps = 0;
        _windowSpikes = 0;
        _windowWallTicks = Stopwatch.GetTimestamp();
        RecentSpikesPerSecond = 0;
        RecentRealTimeFactor = 0;
    }
}
