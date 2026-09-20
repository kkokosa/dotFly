using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using DotFly.Core.Backends;
using DotFly.Core.Graph;
using DotFly.Core.Memory;
using DotFly.Core.Models;
using DotFly.Core.Randomness;
using DotFly.Cpu.Kernels;
using DotFly.Cpu.Threading;

namespace DotFly.Cpu.Fast;

/// <summary>
/// The fast CPU backend: float32 state in aligned unmanaged memory, SIMD phase A
/// (<see cref="LifKernel"/>), blocked-CSR delivery without atomics, a dedicated fork-join pool and
/// counter-based randomness. Semantics are those of <see cref="Reference.ReferenceBackend"/>;
/// results are bit-identical for any thread count.
/// <para>
/// <b>Blocks.</b> Neurons are split into contiguous, 16-aligned blocks (several per thread). Within
/// one step a block touches only its own state (v, g, countdowns, blocked bits, spikers) plus the
/// read-only graph and the spike lists published <em>delay</em> steps earlier, so blocks are
/// independent within a step. Which thread runs a block does not affect the arithmetic; each
/// thread owns a contiguous run of blocks, and the cut points are recomputed per batch from the
/// measured cost of the previous batch so clustered stimulation or clustered targets do not leave
/// threads idle. Contiguity keeps the per-spike cost of delivery at one offset-row read per thread.
/// </para>
/// <para>
/// <b>Batching.</b> Because of the delay, a thread can run up to <em>delay</em> consecutive steps
/// over its blocks without synchronising; the caller merges the per-block, per-step spike lists at
/// the batch boundary and publishes them into the delay ring. One barrier per batch.
/// </para>
/// </summary>
public sealed unsafe class CpuBackend : ISpikingBackend, IParallelBody
{
    /// <summary>Blocks per thread (DOTFLY_BLOCKS_PER_THREAD overrides for experiments).</summary>
    private static readonly int BlocksPerThread = int.TryParse(Environment.GetEnvironmentVariable("DOTFLY_BLOCKS_PER_THREAD"), out int bpt) && bpt > 0 ? bpt : 4;

    private static readonly bool NoPrefetch = Environment.GetEnvironmentVariable("DOTFLY_NOPREFETCH") == "1";

    /// <summary>
    /// Whether the calling (simulation) thread is pinned to a logical processor like the workers.
    /// Off by default: inside a host such as a game engine the caller shares a core with the host's
    /// own threads and pinning it makes it the straggler. Workers are always pinned.
    /// </summary>
    public static bool PinCaller { get; set; } = Environment.GetEnvironmentVariable("DOTFLY_PIN_CALLER") == "1";

    /// <summary>Whether per-batch profiling is enabled (DOTFLY_PROFILE=1).</summary>
    public static readonly bool Profile = Environment.GetEnvironmentVariable("DOTFLY_PROFILE") == "1";

    private readonly Brain _brain;
    private readonly LifStepConstants _k;
    private readonly LifKernel.Constants _kf;
    private readonly int _n;
    private readonly ComputeThreadPool _pool;
    private readonly BlockedCsr _blocked;
    private readonly int _threads;
    private readonly int _blocks;
    private readonly int _batchSteps;

    // State.
    private readonly NativeBuffer<float> _v;
    private readonly NativeBuffer<float> _g;
    private readonly NativeBuffer<int> _countdown;
    private readonly NativeBuffer<int> _arm;            // per-neuron max(R−1, 0)
    private readonly NativeBuffer<ushort> _blockedBits; // bit i&15 of word i>>4: refractory this step → inputs discarded
    private readonly NativeBuffer<byte> _transmits;     // 1 = outgoing synapses active
    private readonly NativeBuffer<double> _poissonP;    // per-step probability, 0 = none
    private readonly NativeBuffer<long> _nextEvent;     // step of the next Poisson event
    private readonly NativeBuffer<uint> _eventIndex;    // events scheduled so far (RNG counter)
    private readonly List<int> _poissonNeurons = [];    // sorted
    private readonly NativeBuffer<float> _current;      // mV added per step, 0 = none
    private readonly List<int> _currentNeurons = [];    // sorted

    // Spikes. Per block and per batch step: local list at [_local + s*N + blockStart].
    private readonly NativeBuffer<int> _local;          // K × N
    private readonly int[] _localCounts;                // [s * blocks + b]
    private readonly NativeBuffer<int> _reported;       // K × N merged lists of the last batch
    private readonly int[] _reportedStart;              // K + 1
    private readonly NativeBuffer<int>[] _ring;         // delay ring of merged lists
    private readonly int[] _ringCount;

    private readonly SortedDictionary<long, List<(int Neuron, double Mv)>> _injections = new();
    private readonly List<(int Neuron, double Mv)>?[] _batchInjections;

    // Block ownership (contiguous runs, reassigned per batch) and measured cost.
    private readonly int[] _threadFirstBlock;           // threads + 1 cut points
    private readonly long[] _blockCost;                 // ticks in the previous batch
    private readonly long[] _blockCostNow;

    private bool _callerPinned;
    private int _batchCount;   // steps in the current batch
    private long _step;        // first step of the current batch while dispatched; completed steps otherwise
    private bool _recurrent;
    private bool _external;

    private readonly long[] _edgesDeliveredByThread = new long[64];

    /// <summary>Total synaptic deliveries performed (edge updates, blocked targets excluded from the update but counted) since construction.</summary>
    public long EdgesDelivered
    {
        get
        {
            long sum = 0;
            for (int t = 0; t < _threads; t++)
            {
                sum += _edgesDeliveredByThread[t];
            }

            return sum;
        }
    }

    // Profiling (DOTFLY_PROFILE=1).
    private long _profDispatch, _profMerge, _profBatches;
    private readonly long[] _profBody = new long[64];
    private readonly long[] _profKernel = new long[64];
    private readonly long[] _profDeliver = new long[64];
    private readonly long[] _profPoisson = new long[64];

    /// <inheritdoc />
    public Brain Brain => _brain;

    /// <inheritdoc />
    public LifStepConstants Constants => _k;

    /// <inheritdoc />
    public int NeuronCount => _n;

    /// <inheritdoc />
    public long Step => _step;

    /// <inheritdoc />
    public ulong Seed { get; private set; }

    /// <inheritdoc />
    public bool RecurrentTransmission { get; set; } = true;

    /// <inheritdoc />
    public bool ExternalInput { get; set; } = true;

    /// <summary>Number of threads in use.</summary>
    public int Threads => _threads;

    /// <summary>Number of neuron blocks (several per thread).</summary>
    public int Blocks => _blocks;

    /// <summary>Maximum steps executed per barrier (= synaptic delay in steps, at least 1).</summary>
    public int BatchSteps => _batchSteps;

    /// <summary>Creates a backend with the checkpoint's model at <paramref name="dtMs"/>.</summary>
    public CpuBackend(Brain brain, ulong seed, int threads = 0, double dtMs = 0.1)
        : this(brain, brain.Model.Discretize(dtMs), seed, threads)
    {
    }

    /// <summary>
    /// Creates a backend with explicit constants; <paramref name="threads"/> ≤ 0 uses one thread per
    /// physical core (half the logical processors when SMT is present).
    /// </summary>
    public CpuBackend(Brain brain, LifStepConstants constants, ulong seed, int threads = 0, int pinOffset = 0, int pinTotal = 0)
    {
        ArgumentNullException.ThrowIfNull(brain);
        _brain = brain;
        _k = constants;
        _kf = LifKernel.Constants.From(constants);
        _n = brain.NeuronCount;
        Seed = seed;
        _threads = threads <= 0 ? DefaultThreads() : threads;
        _threads = Math.Max(1, Math.Min(_threads, Math.Max(1, _n / 256)));
        _blocks = _threads == 1 ? 1 : Math.Min(_threads * BlocksPerThread, Math.Max(1, _n / 64));
        _batchSteps = Math.Max(1, constants.DelaySteps);
        _pool = new ComputeThreadPool(_threads, pin: true, pinOffset, pinTotal);
        _blocked = new BlockedCsr(brain, _blocks);

        _v = new NativeBuffer<float>(_n);
        _g = new NativeBuffer<float>(_n);
        _countdown = new NativeBuffer<int>(_n);
        _arm = new NativeBuffer<int>(_n);
        _blockedBits = new NativeBuffer<ushort>((_n + 15) / 16);
        _transmits = new NativeBuffer<byte>(_n);
        _poissonP = new NativeBuffer<double>(_n);
        _nextEvent = new NativeBuffer<long>(_n);
        _eventIndex = new NativeBuffer<uint>(_n);
        _current = new NativeBuffer<float>(_n);
        _local = new NativeBuffer<int>(checked(_batchSteps * _n));
        _localCounts = new int[_batchSteps * _blocks];
        _reported = new NativeBuffer<int>(checked(_batchSteps * _n));
        _reportedStart = new int[_batchSteps + 1];
        _ring = new NativeBuffer<int>[constants.RingSlots];
        _ringCount = new int[constants.RingSlots];
        for (int i = 0; i < _ring.Length; i++)
        {
            _ring[i] = new NativeBuffer<int>(_n);
        }

        _batchInjections = new List<(int, double)>?[_batchSteps];
        _threadFirstBlock = new int[_threads + 1];
        _blockCost = new long[_blocks];
        _blockCostNow = new long[_blocks];
        AssignBlocks();

        _arm.Span.Fill(Math.Max(0, constants.RefractorySteps - 1));
        _transmits.Span.Fill(1);
        Reset();
    }

    /// <summary>One thread per physical core when the OS reports SMT (2 logical per core), else all logical.</summary>
    public static int DefaultThreads()
    {
        int logical = Environment.ProcessorCount;
        return logical >= 4 && logical % 2 == 0 ? logical / 2 : logical;
    }

    /// <inheritdoc />
    public double GetV(int neuron) => _v[neuron];

    /// <inheritdoc />
    public double GetG(int neuron) => _g[neuron];

    /// <inheritdoc />
    public void SetRefractorySteps(int neuron, int steps)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(steps);
        _arm[neuron] = Math.Max(0, steps - 1);
    }

    /// <inheritdoc />
    public void SetPoissonProbability(int neuron, double probabilityPerStep)
    {
        if (probabilityPerStep is < 0 or > 1 || double.IsNaN(probabilityPerStep))
        {
            throw new ArgumentOutOfRangeException(nameof(probabilityPerStep));
        }

        bool was = _poissonP[neuron] > 0;
        _poissonP[neuron] = probabilityPerStep;
        bool now = probabilityPerStep > 0;
        if (now)
        {
            _nextEvent[neuron] = PoissonProcess.NextEventStep(Seed, neuron, _eventIndex[neuron]++, _step - 1, probabilityPerStep);
        }

        if (now && !was)
        {
            int pos = _poissonNeurons.BinarySearch(neuron);
            _poissonNeurons.Insert(~pos, neuron);
        }
        else if (!now && was)
        {
            _poissonNeurons.Remove(neuron);
        }
    }

    /// <inheritdoc />
    public void SetCurrent(int neuron, double mvPerStep)
    {
        if (double.IsNaN(mvPerStep))
        {
            throw new ArgumentOutOfRangeException(nameof(mvPerStep));
        }

        bool was = _current[neuron] != 0;
        _current[neuron] = (float)mvPerStep;
        bool now = mvPerStep != 0;
        if (now && !was)
        {
            _currentNeurons.Insert(~_currentNeurons.BinarySearch(neuron), neuron);
        }
        else if (!now && was)
        {
            _currentNeurons.Remove(neuron);
        }
    }

    /// <inheritdoc />
    public void SetTransmits(int neuron, bool transmits) => _transmits[neuron] = transmits ? (byte)1 : (byte)0;

    /// <inheritdoc />
    public byte[] SaveState()
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write(_step);
        w.Write(_n);
        w.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(_v.Span));
        w.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(_g.Span));
        w.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(_countdown.Span));
        w.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(_blockedBits.Span));
        w.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(_eventIndex.Span));
        w.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(_nextEvent.Span));
        w.Write(_ring.Length);
        for (int slot = 0; slot < _ring.Length; slot++)
        {
            w.Write(_ringCount[slot]);
            w.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(_ring[slot].Span[.._ringCount[slot]]));
        }

        w.Flush();
        return ms.ToArray();
    }

    /// <inheritdoc />
    public void RestoreState(ReadOnlySpan<byte> state)
    {
        using var ms = new MemoryStream(state.ToArray());
        using var r = new BinaryReader(ms);
        long step = r.ReadInt64();
        if (r.ReadInt32() != _n)
        {
            throw new InvalidDataException("State was saved for a different neuron count.");
        }

        Read(r, System.Runtime.InteropServices.MemoryMarshal.AsBytes(_v.Span));
        Read(r, System.Runtime.InteropServices.MemoryMarshal.AsBytes(_g.Span));
        Read(r, System.Runtime.InteropServices.MemoryMarshal.AsBytes(_countdown.Span));
        Read(r, System.Runtime.InteropServices.MemoryMarshal.AsBytes(_blockedBits.Span));
        Read(r, System.Runtime.InteropServices.MemoryMarshal.AsBytes(_eventIndex.Span));
        Read(r, System.Runtime.InteropServices.MemoryMarshal.AsBytes(_nextEvent.Span));
        if (r.ReadInt32() != _ring.Length)
        {
            throw new InvalidDataException("State was saved for a different delay.");
        }

        for (int slot = 0; slot < _ring.Length; slot++)
        {
            _ringCount[slot] = r.ReadInt32();
            Read(r, System.Runtime.InteropServices.MemoryMarshal.AsBytes(_ring[slot].Span[.._ringCount[slot]]));
        }

        _injections.Clear();
        Array.Clear(_localCounts);
        Array.Clear(_reportedStart);
        _step = step;

        static void Read(BinaryReader reader, Span<byte> target)
        {
            int read = 0;
            while (read < target.Length)
            {
                int n = reader.Read(target[read..]);
                if (n <= 0)
                {
                    throw new EndOfStreamException("Truncated backend state.");
                }

                read += n;
            }
        }
    }

    /// <inheritdoc />
    public void ScheduleInjection(long step, int neuron, double mv)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(step, _step);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual((uint)neuron, (uint)_n);
        if (!_injections.TryGetValue(step, out var list))
        {
            _injections[step] = list = [];
        }

        list.Add((neuron, mv));
    }

    /// <inheritdoc />
    public void Advance(int steps, ISpikeSink? sink)
    {
        while (steps > 0)
        {
            int k = Math.Min(steps, _batchSteps);
            RunBatch(k);
            if (sink is not null)
            {
                for (int s = 0; s < k; s++)
                {
                    int start = _reportedStart[s];
                    sink.OnStep(_step - k + s, new ReadOnlySpan<int>(_reported.Pointer + start, _reportedStart[s + 1] - start));
                }
            }

            steps -= k;
        }
    }

    /// <inheritdoc />
    public void Reset(ulong seed)
    {
        Seed = seed;
        Reset();
    }

    /// <inheritdoc />
    public void Reset()
    {
        _v.Span.Fill((float)_k.RestMv);
        _g.Span.Clear();
        _countdown.Span.Clear();
        _blockedBits.Span.Clear();
        Array.Clear(_ringCount);
        Array.Clear(_localCounts);
        Array.Clear(_reportedStart);
        _injections.Clear();
        _step = 0;
        _eventIndex.Span.Clear();
        foreach (int i in _poissonNeurons)
        {
            _nextEvent[i] = PoissonProcess.NextEventStep(Seed, i, _eventIndex[i]++, -1, _poissonP[i]);
        }
    }

    private void RunBatch(int k)
    {
        long first = _step;
        _batchCount = k;
        _recurrent = RecurrentTransmission;
        _external = ExternalInput;
        for (int s = 0; s < k; s++)
        {
            _batchInjections[s] = null;
            if (_injections.Count > 0 && _injections.Remove(first + s, out var list))
            {
                _batchInjections[s] = list;
            }
        }

        if (_pool.Pinned && !_callerPinned && PinCaller)
        {
            CpuAffinity.PinCurrentThread(CpuAffinity.ProcessorFor(0, _threads));
            _callerPinned = true;
        }

        long t0 = Profile ? Stopwatch.GetTimestamp() : 0;
        _pool.Dispatch(this);
        long t1 = Profile ? Stopwatch.GetTimestamp() : 0;

        // Merge per-block lists per step (block order = ascending neuron index), publish into the
        // ring for delivery at step + delay, and keep a copy for the sink.
        int slots = _k.RingSlots;
        int total = 0;
        for (int s = 0; s < k; s++)
        {
            _reportedStart[s] = total;
            int stepTotal = 0;
            int* local = _local.Pointer + (long)s * _n;
            for (int b = 0; b < _blocks; b++)
            {
                int c = _localCounts[s * _blocks + b];
                if (c > 0)
                {
                    Buffer.MemoryCopy(local + _blocked.Bounds[b], _reported.Pointer + total, ((long)_batchSteps * _n - total) * sizeof(int), (long)c * sizeof(int));
                    total += c;
                    stepTotal += c;
                }
            }

            long step = first + s;
            int publishSlot = (int)((step + _k.DelaySteps) % slots);
            if (stepTotal > 0)
            {
                Buffer.MemoryCopy(_reported.Pointer + _reportedStart[s], _ring[publishSlot].Pointer, (long)_n * sizeof(int), (long)stepTotal * sizeof(int));
            }

            _ringCount[publishSlot] = stepTotal;
        }

        _reportedStart[k] = total;
        _step = first + k;

        // Rebalance from the measured costs of this batch.
        Array.Copy(_blockCostNow, _blockCost, _blocks);
        AssignBlocks();

        if (Profile)
        {
            _profDispatch += t1 - t0;
            _profMerge += Stopwatch.GetTimestamp() - t1;
            _profBatches++;
        }
    }

    /// <summary>
    /// Cuts the block sequence into <c>threads</c> contiguous runs of (nearly) equal measured cost.
    /// Unmeasured blocks count 1 each, which yields an equal split on the first batch.
    /// </summary>
    private void AssignBlocks()
    {
        long total = 0;
        for (int b = 0; b < _blocks; b++)
        {
            total += Math.Max(1, _blockCost[b]);
        }

        _threadFirstBlock[0] = 0;
        long acc = 0;
        int next = 0;
        for (int t = 1; t < _threads; t++)
        {
            long target = total * t / _threads;
            // Advance until the cumulative cost reaches this thread's share, leaving enough blocks
            // for the remaining threads.
            while (next < _blocks - (_threads - t) && acc + Math.Max(1, _blockCost[next]) / 2 < target)
            {
                acc += Math.Max(1, _blockCost[next]);
                next++;
            }

            _threadFirstBlock[t] = next;
        }

        _threadFirstBlock[_threads] = _blocks;
    }

    /// <inheritdoc />
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    void IParallelBody.Run(int threadIndex, int threadCount)
    {
        long first = _step;
        int bFirst = _threadFirstBlock[threadIndex];
        int bEnd = _threadFirstBlock[threadIndex + 1];
        int start = _blocked.Bounds[bFirst];
        int end = _blocked.Bounds[bEnd];
        long b0 = Profile ? Stopwatch.GetTimestamp() : 0;
        long deliverTicks = 0;
        long edgesDelivered = 0;
        Span<long> kernelTicks = stackalloc long[bEnd - bFirst];
        Span<long> edgesPerBlock = stackalloc long[bEnd - bFirst];
        kernelTicks.Clear();
        edgesPerBlock.Clear();

        for (int s = 0; s < _batchCount; s++)
        {
            // Phase A per block (measured), spikes stored per block.
            for (int b = bFirst; b < bEnd; b++)
            {
                long k0 = Stopwatch.GetTimestamp();
                int bs = _blocked.Bounds[b];
                _localCounts[s * _blocks + b] = LifKernel.Update(
                    in _kf, _v.Pointer, _g.Pointer, _countdown.Pointer, _arm.Pointer, _blockedBits.Pointer,
                    bs, _blocked.Bounds[b + 1], _local.Pointer + (long)s * _n + bs);
                kernelTicks[b - bFirst] += Stopwatch.GetTimestamp() - k0;
            }

            long d0 = Stopwatch.GetTimestamp();
            edgesDelivered += Deliver(bFirst, bEnd, first + s, edgesPerBlock);
            deliverTicks += Stopwatch.GetTimestamp() - d0;

            long p0 = Profile ? Stopwatch.GetTimestamp() : 0;
            ExternalInputs(start, end, first + s, s);
            if (Profile)
            {
                _profPoisson[threadIndex] += Stopwatch.GetTimestamp() - p0;
            }

            // 4. Reset this step's spikers of my blocks (countdown was armed in phase A).
            float vrst = (float)_k.ResetMv;
            float* v = _v.Pointer;
            float* g = _g.Pointer;
            for (int b = bFirst; b < bEnd; b++)
            {
                int* mine = _local.Pointer + (long)s * _n + _blocked.Bounds[b];
                int count = _localCounts[s * _blocks + b];
                for (int q = 0; q < count; q++)
                {
                    int i = mine[q];
                    v[i] = vrst;
                    g[i] = 0f;
                }
            }
        }

        // Cost model for rebalancing: measured phase-A ticks per block plus delivered edges at the
        // per-edge cost this thread just observed.
        _edgesDeliveredByThread[threadIndex] += edgesDelivered;
        double ticksPerEdge = edgesDelivered > 0 ? (double)deliverTicks / edgesDelivered : 0;
        for (int b = bFirst; b < bEnd; b++)
        {
            _blockCostNow[b] = kernelTicks[b - bFirst] + (long)(edgesPerBlock[b - bFirst] * ticksPerEdge);
        }

        if (Profile)
        {
            _profBody[threadIndex] += Stopwatch.GetTimestamp() - b0;
            _profDeliver[threadIndex] += deliverTicks;
            for (int b = bFirst; b < bEnd; b++)
            {
                _profKernel[threadIndex] += kernelTicks[b - bFirst];
            }
        }
    }

    /// <summary>
    /// 3a. Delivers the spikes emitted at <c>t - delay</c> into blocks <c>[bFirst, bEnd)</c> as one
    /// contiguous column walk per spike; accumulates delivered edges per block. Returns edges delivered.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private long Deliver(int bFirst, int bEnd, long t, Span<long> edgesPerBlock)
    {
        if (!_recurrent)
        {
            return 0;
        }

        float* g = _g.Pointer;
        ushort* blocked = _blockedBits.Pointer;
        int slots = _k.RingSlots;
        int dueSlot = (int)(t % slots);
        int* due = _ring[dueSlot].Pointer;
        int dueCount = _ringCount[dueSlot];
        int* cols = _brain.ColumnsPointer;
        short* w = _brain.WeightsPointer;
        int* off = _blocked.OffsetsPointer;
        int stride = _blocked.Blocks + 1;
        byte* transmits = _transmits.Pointer;
        float wsyn = (float)_k.SynapseWeightMv;
        long delivered = 0;

        // Software prefetch: the due lists of the next two steps are already known, so pull their
        // offset rows (t + 2) and column slices (t + 1) into cache while this step's deliveries
        // run. Lists for the last steps of a batch may be stale - harmless.
        if (Sse.IsSupported && !NoPrefetch)
        {
            int* next = _ring[(int)((t + 1) % slots)].Pointer;
            int nextCount = Math.Min(_ringCount[(int)((t + 1) % slots)], 256);
            for (int q = 0; q < nextCount; q++)
            {
                Sse.Prefetch0(cols + off[(long)next[q] * stride + bFirst]);
            }

            int* after = _ring[(int)((t + 2) % slots)].Pointer;
            int afterCount = Math.Min(_ringCount[(int)((t + 2) % slots)], 256);
            for (int q = 0; q < afterCount; q++)
            {
                Sse.Prefetch0(off + (long)after[q] * stride + bFirst);
            }
        }

        for (int q = 0; q < dueCount; q++)
        {
            int pre = due[q];
            if (transmits[pre] == 0)
            {
                continue;
            }

            int* row = off + (long)pre * stride;
            int e = row[bFirst];
            int eEnd = row[bEnd];
            delivered += eEnd - e;
            for (int b = bFirst; b < bEnd; b++)
            {
                edgesPerBlock[b - bFirst] += row[b + 1] - row[b];
            }

            for (; e < eEnd; e++)
            {
                int post = cols[e];
                if ((blocked[post >> 4] & (1 << (post & 15))) == 0)
                {
                    g[post] += w[e] * wsyn;
                }
            }
        }

        return delivered;
    }

    /// <summary>3b. Poisson events and scheduled injections for neurons <c>[start, end)</c> at step <paramref name="t"/>.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ExternalInputs(int start, int end, long t, int s)
    {
        if (!_external)
        {
            return;
        }

        float* v = _v.Pointer;
        ushort* blocked = _blockedBits.Pointer;
        float pw = (float)_k.PoissonWeightMv;
        long* nextEvent = _nextEvent.Pointer;
        uint* eventIndex = _eventIndex.Pointer;
        double* pp = _poissonP.Pointer;
        int lo = LowerBound(_poissonNeurons, start);
        int hi = LowerBound(_poissonNeurons, end);
        for (int q = lo; q < hi; q++)
        {
            int i = _poissonNeurons[q];
            if (nextEvent[i] == t)
            {
                if ((blocked[i >> 4] & (1 << (i & 15))) == 0)
                {
                    v[i] += pw;
                }

                nextEvent[i] = PoissonProcess.NextEventStep(Seed, i, eventIndex[i]++, t, pp[i]);
            }
        }

        float* cur = _current.Pointer;
        int clo = LowerBound(_currentNeurons, start);
        int chi = LowerBound(_currentNeurons, end);
        for (int q = clo; q < chi; q++)
        {
            int i = _currentNeurons[q];
            if ((blocked[i >> 4] & (1 << (i & 15))) == 0)
            {
                v[i] += cur[i];
            }
        }

        if (_batchInjections[s] is { } inj)
        {
            foreach ((int neuron, double mv) in inj)
            {
                if (neuron >= start && neuron < end && (blocked[neuron >> 4] & (1 << (neuron & 15))) == 0)
                {
                    v[neuron] += (float)mv;
                }
            }
        }
    }

    private static int LowerBound(List<int> sorted, int value)
    {
        int lo = 0, hi = sorted.Count;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (sorted[mid] < value)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }

        return lo;
    }

    /// <summary>Profile summary (only meaningful with DOTFLY_PROFILE=1).</summary>
    public string ProfileSummary()
    {
        double us = 1e6 / Stopwatch.Frequency;
        long steps = Math.Max(1, _step);
        string per = string.Join(" ", Enumerable.Range(0, _threads).Select(p => $"{_profBody[p] * us / steps:F1}(k{_profKernel[p] * us / steps:F1}/d{_profDeliver[p] * us / steps:F1}/p{_profPoisson[p] * us / steps:F1})"));
        return $"batches {_profBatches}: dispatch wall {_profDispatch * us / steps:F1} µs/step, merge {_profMerge * us / steps:F1} µs/step; per-thread body(kernel/deliver/poisson) µs/step: {per}";
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _pool.Dispose();
        _blocked.Dispose();
        _v.Dispose();
        _g.Dispose();
        _countdown.Dispose();
        _arm.Dispose();
        _blockedBits.Dispose();
        _transmits.Dispose();
        _poissonP.Dispose();
        _nextEvent.Dispose();
        _eventIndex.Dispose();
        _current.Dispose();
        _local.Dispose();
        _reported.Dispose();
        foreach (var r in _ring)
        {
            r.Dispose();
        }
    }
}
