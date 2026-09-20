using System.Diagnostics;

namespace DotFly;

/// <summary>Options for <see cref="RealtimeDriver"/>.</summary>
public sealed record RealtimeOptions
{
    /// <summary>Neural seconds per wall second to aim for. 1 = real time.</summary>
    public double Ratio { get; init; } = 1.0;

    /// <summary>Most steps advanced in one go before re-checking the clock.</summary>
    public int MaxStepsPerTick { get; init; } = 500;

    /// <summary>
    /// If the simulation falls behind the wall clock by more than this, the deficit is forgiven
    /// (neural time is never silently slowed, but it is not "caught up" either after a stall).
    /// </summary>
    public TimeSpan MaxLag { get; init; } = TimeSpan.FromMilliseconds(250);

    /// <summary>Statistics window for <see cref="SimulationClock.RecentRealTimeFactor"/>.</summary>
    public TimeSpan StatsWindow { get; init; } = TimeSpan.FromMilliseconds(500);
}

/// <summary>
/// Runs a <see cref="Simulation"/> on its own thread, keeping neural time in step with the wall
/// clock (times <see cref="RealtimeOptions.Ratio"/>). The game or UI thread writes input ports and
/// reads output snapshots; it never touches the simulation directly. <see cref="BehindBy"/> tells
/// the host how far the network lags, so it can degrade honestly (e.g. switch to a smaller
/// checkpoint) instead of pretending.
/// </summary>
public sealed class RealtimeDriver : IDisposable
{
    private readonly Simulation _sim;
    private readonly RealtimeOptions _options;
    private readonly Thread _thread;
    private readonly ManualResetEventSlim _resume = new(false);
    private volatile bool _running;
    private volatile bool _stop;
    private long _behindTicks;
    private long _stepsThisTick;
    private Exception? _error;

    /// <summary>The simulation being driven.</summary>
    public Simulation Simulation => _sim;

    /// <summary>Whether the driver thread is currently advancing (not paused/stopped).</summary>
    public bool Running => _running && !_stop;

    /// <summary>How far neural time (÷ ratio) lags the wall clock right now; zero or negative when caught up.</summary>
    public TimeSpan BehindBy => TimeSpan.FromTicks(Volatile.Read(ref _behindTicks));

    /// <summary>Steps advanced in the most recent tick.</summary>
    public long StepsLastTick => Volatile.Read(ref _stepsThisTick);

    /// <summary>The exception that stopped the driver, if any.</summary>
    public Exception? Error => _error;

    /// <summary>Creates a driver; call <see cref="Start"/> to begin.</summary>
    public RealtimeDriver(Simulation sim, RealtimeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(sim);
        _sim = sim;
        _options = options ?? new RealtimeOptions();
        _thread = new Thread(Loop)
        {
            IsBackground = true,
            Name = "dotfly-realtime",
            Priority = ThreadPriority.AboveNormal,
        };
    }

    /// <summary>Starts (or resumes) advancing.</summary>
    public void Start()
    {
        ObjectDisposedException.ThrowIf(_stop, this);
        _running = true;
        _resume.Set();
        if (!_thread.IsAlive)
        {
            _thread.Start();
        }
    }

    /// <summary>Pauses; neural time stops, wall time keeps going (the deficit is forgiven up to <see cref="RealtimeOptions.MaxLag"/>).</summary>
    public void Pause()
    {
        _running = false;
        _resume.Reset();
    }

    private void Loop()
    {
        double ticksPerStep = Stopwatch.Frequency * _sim.DtMs / 1000.0 / _options.Ratio;
        long maxLagTicks = (long)(_options.MaxLag.TotalSeconds * Stopwatch.Frequency);
        long statsTicks = (long)(_options.StatsWindow.TotalSeconds * Stopwatch.Frequency);
        long origin = Stopwatch.GetTimestamp();
        long stepsAtOrigin = _sim.Step;
        long lastStats = origin;
        var spinner = new SpinWait();

        try
        {
            while (!_stop)
            {
                if (!_running)
                {
                    _resume.Wait(10);
                    // Re-anchor so the pause is not "caught up".
                    origin = Stopwatch.GetTimestamp();
                    stepsAtOrigin = _sim.Step;
                    continue;
                }

                long now = Stopwatch.GetTimestamp();
                long dueSteps = (long)((now - origin) / ticksPerStep) + stepsAtOrigin;
                long behind = dueSteps - _sim.Step;
                Volatile.Write(ref _behindTicks, (long)(behind * ticksPerStep));

                if (behind <= 0)
                {
                    Volatile.Write(ref _stepsThisTick, 0);
                    spinner.SpinOnce(sleep1Threshold: 4);
                    continue;
                }

                if (behind * ticksPerStep > maxLagTicks)
                {
                    // Forgive the deficit beyond MaxLag: re-anchor rather than racing to catch up.
                    origin = now - (long)(maxLagTicks / ticksPerStep * ticksPerStep);
                    stepsAtOrigin = _sim.Step;
                    behind = (long)(maxLagTicks / ticksPerStep);
                }

                int steps = (int)Math.Min(behind, _options.MaxStepsPerTick);
                _sim.Advance(steps);
                Volatile.Write(ref _stepsThisTick, steps);
                spinner.Reset();

                if (now - lastStats >= statsTicks)
                {
                    _sim.Clock.MarkWindow(now);
                    lastStats = now;
                }
            }
        }
        catch (Exception e)
        {
            _error = e;
            _running = false;
        }
    }

    /// <summary>Stops the driver thread and waits for it. The simulation is left as is.</summary>
    public void Dispose()
    {
        if (_stop)
        {
            return;
        }

        _stop = true;
        _resume.Set();
        if (_thread.IsAlive && Thread.CurrentThread != _thread)
        {
            _thread.Join();
        }

        _resume.Dispose();
    }
}
