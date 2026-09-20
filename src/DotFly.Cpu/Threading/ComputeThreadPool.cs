namespace DotFly.Cpu.Threading;

/// <summary>A unit of work executed once per pool thread.</summary>
public interface IParallelBody
{
    /// <summary>Runs the body on thread <paramref name="threadIndex"/> of <paramref name="threadCount"/>.</summary>
    void Run(int threadIndex, int threadCount);
}

/// <summary>
/// A dedicated fork-join pool for the simulation step: <c>N−1</c> background workers plus the
/// calling thread. <see cref="Dispatch"/> runs a body on every thread and returns when all have
/// finished — one barrier. Workers spin on a generation counter while the simulation is hot and
/// fall back to an event after a while so an idle pool does not burn CPU. No allocations per
/// dispatch.
/// </summary>
public sealed class ComputeThreadPool : IDisposable
{
    private const int SpinsBeforeSleep = 20_000;

    private readonly Thread[] _workers;
    private readonly ManualResetEventSlim _wake = new(false);
    private volatile bool _shutdown;
    private int _generation;
    private int _pending;
    private IParallelBody? _body;

    /// <summary>Total number of threads, including the caller.</summary>
    public int ThreadCount { get; }

    /// <summary>Whether worker threads are pinned to logical processors.</summary>
    public bool Pinned { get; }

    private readonly int _pinOffset;
    private readonly int _pinTotal;

    /// <summary>
    /// Creates a pool with <paramref name="threadCount"/> threads (1 = caller only, no workers).
    /// With <paramref name="pin"/>, workers are pinned to distinct logical processors (one per
    /// physical core when SMT is detected); the caller thread is not pinned. Several pools in one
    /// process (several simulations) pass a <paramref name="pinOffset"/> (worker slots taken by
    /// the other pools) and <paramref name="pinTotal"/> (worker slots of all pools) so they land
    /// on different cores.
    /// </summary>
    public ComputeThreadPool(int threadCount, bool pin = true, int pinOffset = 0, int pinTotal = 0)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(threadCount, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(pinOffset);
        ThreadCount = threadCount;
        Pinned = pin && threadCount > 1;
        _pinOffset = pinOffset;
        _pinTotal = pinTotal > 0 ? pinTotal : threadCount + pinOffset;
        _workers = new Thread[threadCount - 1];
        for (int i = 0; i < _workers.Length; i++)
        {
            int index = i + 1;
            _workers[i] = new Thread(() => WorkerLoop(index))
            {
                IsBackground = true,
                Name = $"dotfly-compute-{index}",
                Priority = ThreadPriority.AboveNormal,
            };
            _workers[i].Start();
        }
    }

    /// <summary>Runs <paramref name="body"/> on all threads (index 0 = caller) and waits for completion.</summary>
    public void Dispatch(IParallelBody body)
    {
        ObjectDisposedException.ThrowIf(_shutdown, this);
        if (_workers.Length == 0)
        {
            body.Run(0, 1);
            return;
        }

        _body = body;
        Volatile.Write(ref _pending, _workers.Length);
        Interlocked.Increment(ref _generation);
        _wake.Set();

        body.Run(0, ThreadCount);

        var spinner = new SpinWait();
        while (Volatile.Read(ref _pending) != 0)
        {
            spinner.SpinOnce(sleep1Threshold: -1);
        }

        _wake.Reset();
        _body = null;
    }

    private void WorkerLoop(int index)
    {
        if (Pinned)
        {
            CpuAffinity.PinCurrentThread(CpuAffinity.ProcessorFor(index + _pinOffset, _pinTotal));
        }

        int seen = 0;
        while (!_shutdown)
        {
            // Wait for a new generation: spin first, then block.
            int spins = 0;
            while (Volatile.Read(ref _generation) == seen)
            {
                if (_shutdown)
                {
                    return;
                }

                if (++spins < SpinsBeforeSleep)
                {
                    Thread.SpinWait(4);
                }
                else
                {
                    _wake.Wait(1);
                }
            }

            seen = Volatile.Read(ref _generation);
            IParallelBody? body = _body;
            if (body is not null)
            {
                body.Run(index, ThreadCount);
            }

            Interlocked.Decrement(ref _pending);
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_shutdown)
        {
            return;
        }

        _shutdown = true;
        Interlocked.Increment(ref _generation);
        _wake.Set();
        foreach (Thread t in _workers)
        {
            t.Join();
        }

        _wake.Dispose();
    }
}
