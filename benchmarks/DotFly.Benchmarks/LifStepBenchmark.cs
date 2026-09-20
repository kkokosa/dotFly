using BenchmarkDotNet.Attributes;
using DotFly.Core.Memory;
using DotFly.Core.Models;
using DotFly.Cpu.Kernels;

namespace DotFly.Benchmarks;

/// <summary>
/// Phase A over a neuron range: the scalar baseline loop versus <see cref="LifKernel.Update"/>
/// (Vector256 + scalar tail), at an L2-resident slice size and at full MaleCNS size.
/// </summary>
[MemoryDiagnoser]
public unsafe class LifStepBenchmark
{
    /// <summary>Neurons per call: one thread's slice of MaleCNS on 8 cores, and the whole graph.</summary>
    [Params(20_838, 166_700)]
    public int N { get; set; }

    private NativeBuffer<float> _v = null!;
    private NativeBuffer<float> _g = null!;
    private NativeBuffer<int> _cd = null!;
    private NativeBuffer<int> _arm = null!;
    private NativeBuffer<ushort> _blocked = null!;
    private NativeBuffer<int> _spikes = null!;
    private LifStepConstants _k;
    private LifKernel.Constants _kf;

    [GlobalSetup]
    public void Setup()
    {
        _v = new NativeBuffer<float>(N);
        _g = new NativeBuffer<float>(N);
        _cd = new NativeBuffer<int>(N);
        _arm = new NativeBuffer<int>(N);
        _blocked = new NativeBuffer<ushort>(N / 16 + 1);
        _spikes = new NativeBuffer<int>(N);
        _k = NeuronModel.Shiu2024.Discretize(0.1);
        _kf = LifKernel.Constants.From(_k);
        var rng = new Random(1);
        for (int i = 0; i < N; i++)
        {
            _v[i] = -52f + (float)rng.NextDouble() * 6f;   // below threshold: realistic (few spikes)
            _g[i] = (float)rng.NextDouble();
            _cd[i] = rng.NextDouble() < 0.02 ? rng.Next(1, 22) : 0;
            _arm[i] = 21;
        }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _v.Dispose();
        _g.Dispose();
        _cd.Dispose();
        _arm.Dispose();
        _blocked.Dispose();
        _spikes.Dispose();
    }

    [Benchmark(Baseline = true)]
    public int ScalarBaseline()
    {
        float a = _kf.A, b = _kf.B, c = _kf.C, v0 = _kf.Rest, vth = _kf.Threshold;
        float* v = _v.Pointer;
        float* g = _g.Pointer;
        int* cd = _cd.Pointer;
        ushort* blocked = _blocked.Pointer;
        int spikes = 0;
        for (int i = 0; i < N; i++)
        {
            int d = cd[i];
            bool bl;
            if (d > 0)
            {
                cd[i] = d - 1;
                bl = true;
            }
            else
            {
                float nv = v0 + a * (v[i] - v0) + b * g[i];
                g[i] *= c;
                v[i] = nv;
                bl = nv > vth;
                if (bl)
                {
                    spikes++;
                }
            }

            int bit = 1 << (i & 15);
            blocked[i >> 4] = (ushort)(bl ? blocked[i >> 4] | bit : blocked[i >> 4] & ~bit);
        }

        return spikes;
    }

    [Benchmark]
    public int LifKernelUpdate() =>
        LifKernel.Update(in _kf, _v.Pointer, _g.Pointer, _cd.Pointer, _arm.Pointer, _blocked.Pointer, 0, N, _spikes.Pointer);
}
