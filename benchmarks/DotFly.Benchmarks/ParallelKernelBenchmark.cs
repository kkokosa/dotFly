using BenchmarkDotNet.Attributes;
using DotFly.Core.Memory;
using DotFly.Core.Models;
using DotFly.Cpu.Kernels;
using DotFly.Cpu.Threading;

namespace DotFly.Benchmarks;

/// <summary>
/// Phase A over a MaleCNS-sized population split across the compute pool: measures how the
/// per-thread slice cost behaves under concurrency (clock, cache, bandwidth) and the dispatch cost.
/// </summary>
public unsafe class ParallelKernelBenchmark
{
    private const int N = 166_720; // MaleCNS rounded to 16

    /// <summary>Threads (pinned one per physical core).</summary>
    [Params(1, 2, 4, 8)]
    public int Threads { get; set; }

    /// <summary>Steps per dispatch.</summary>
    [Params(1, 18)]
    public int Batch { get; set; }

    private ComputeThreadPool _pool = null!;
    private Body _body = null!;

    private sealed class Body(int n, int threads, int batch) : IParallelBody, IDisposable
    {
        private readonly NativeBuffer<float> _v = new(n);
        private readonly NativeBuffer<float> _g = new(n);
        private readonly NativeBuffer<int> _cd = new(n);
        private readonly NativeBuffer<int> _arm = new(n);
        private readonly NativeBuffer<ushort> _bits = new(n / 16 + 1);
        private readonly NativeBuffer<int> _spikes = new(n);
        private readonly int[] _bounds = Enumerable.Range(0, threads + 1).Select(p => p == threads ? n : (int)((long)n * p / threads) & ~15).ToArray();
        private readonly LifKernel.Constants _k = LifKernel.Constants.From(NeuronModel.Shiu2024.Discretize(0.1));

        public void Init()
        {
            var rng = new Random(1);
            for (int i = 0; i < n; i++)
            {
                _v[i] = -52f + (float)rng.NextDouble() * 6f;
                _g[i] = (float)rng.NextDouble();
                _arm[i] = 21;
            }
        }

        public void Run(int threadIndex, int threadCount)
        {
            int start = _bounds[threadIndex], end = _bounds[threadIndex + 1];
            for (int s = 0; s < batch; s++)
            {
                LifKernel.Update(in _k, _v.Pointer, _g.Pointer, _cd.Pointer, _arm.Pointer, _bits.Pointer, start, end, _spikes.Pointer + start);
            }
        }

        public void Dispose()
        {
            _v.Dispose();
            _g.Dispose();
            _cd.Dispose();
            _arm.Dispose();
            _bits.Dispose();
            _spikes.Dispose();
        }
    }

    [GlobalSetup]
    public void Setup()
    {
        _pool = new ComputeThreadPool(Threads);
        _body = new Body(N, Threads, Batch);
        _body.Init();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _pool.Dispose();
        _body.Dispose();
    }

    /// <summary>One dispatch = <see cref="Batch"/> steps over all neurons.</summary>
    [Benchmark]
    public void Dispatch() => _pool.Dispatch(_body);
}
