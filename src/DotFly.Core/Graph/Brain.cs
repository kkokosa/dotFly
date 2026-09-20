using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using DotFly.Core.Checkpoints;
using DotFly.Core.Models;

namespace DotFly.Core.Graph;

/// <summary>
/// An opened, immutable checkpoint: the neuron table, the CSR graph and the named sets, all read
/// in place from the memory-mapped file. Thread-safe for concurrent readers; share one instance
/// between simulations.
/// </summary>
public sealed class Brain : IDisposable
{
    private readonly DfbFile _file;
    private readonly StringTable _strings;
    private readonly Dictionary<string, NeuronSet> _sets;
    private readonly Dictionary<ulong, int> _byBodyId;

    /// <summary>Provenance record of the checkpoint.</summary>
    public Provenance Provenance { get; }

    /// <summary>Number of neurons.</summary>
    public int NeuronCount { get; }

    /// <summary>Number of directed edges.</summary>
    public long EdgeCount { get; }

    /// <summary>Sum of absolute contact counts over all edges.</summary>
    public long ContactCount { get; }

    /// <summary>Neuron model the checkpoint was built for.</summary>
    public NeuronModel Model => Provenance.Model;

    /// <summary>Named sets shipped in the checkpoint.</summary>
    public IReadOnlyDictionary<string, NeuronSet> Sets => _sets;

    /// <summary>The interned annotation strings.</summary>
    public StringTable Strings => _strings;

    /// <summary>Path of the underlying file.</summary>
    public string Path => _file.Path;

    private Brain(DfbFile file)
    {
        _file = file;
        Provenance = Provenance.FromJson(System.Text.Encoding.UTF8.GetString(file.Bytes(DfbFormat.TagProvenance)));
        _strings = StringTable.Deserialize(file.Bytes(DfbFormat.TagStrings));
        NeuronCount = checked((int)file.Section(DfbFormat.TagBodyId).Count);
        EdgeCount = file.Section(DfbFormat.TagColumns).Count;

        long contacts = 0;
        foreach (short w in Weights)
        {
            contacts += Math.Abs((int)w);
        }

        ContactCount = contacts;

        _byBodyId = new Dictionary<ulong, int>(NeuronCount);
        ReadOnlySpan<ulong> ids = BodyIds;
        for (int i = 0; i < ids.Length; i++)
        {
            _byBodyId[ids[i]] = i;
        }

        _sets = ReadSets(file);
    }

    /// <summary>Opens a checkpoint file.</summary>
    public static Brain Open(string path)
    {
        DfbFile file = DfbFile.Open(path);
        try
        {
            return new Brain(file);
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    // ---- columns -------------------------------------------------------------------------

    /// <summary>Body / root IDs by neuron index.</summary>
    public ReadOnlySpan<ulong> BodyIds => _file.Column<ulong>(DfbFormat.TagBodyId);

    /// <summary>Neurotransmitter by neuron index.</summary>
    public ReadOnlySpan<Neurotransmitter> Neurotransmitters => _file.Column<Neurotransmitter>(DfbFormat.TagNeurotransmitter);

    /// <summary>Side by neuron index.</summary>
    public ReadOnlySpan<Side> Sides => _file.Column<Side>(DfbFormat.TagSide);

    /// <summary>Soma voxel coordinates, 3 ints per neuron (<see cref="int.MinValue"/> = unknown).</summary>
    public ReadOnlySpan<int> SomaPositions => _file.Column<int>(DfbFormat.TagSoma);

    /// <summary>CSR row pointers (length N+1).</summary>
    public ReadOnlySpan<long> RowPtr => _file.Column<long>(DfbFormat.TagRowPtr);

    /// <summary>CSR column indices (postsynaptic neuron index).</summary>
    public ReadOnlySpan<int> Columns => _file.Column<int>(DfbFormat.TagColumns);

    /// <summary>CSR weights (signed contact counts).</summary>
    public ReadOnlySpan<short> Weights => _file.Column<short>(DfbFormat.TagWeights);

    /// <summary>Unsafe pointer access for kernels; valid until <see cref="Dispose"/>.</summary>
    public unsafe long* RowPtrPointer => _file.Pointer<long>(DfbFormat.TagRowPtr);

    /// <summary>Unsafe pointer access for kernels; valid until <see cref="Dispose"/>.</summary>
    public unsafe int* ColumnsPointer => _file.Pointer<int>(DfbFormat.TagColumns);

    /// <summary>Unsafe pointer access for kernels; valid until <see cref="Dispose"/>.</summary>
    public unsafe short* WeightsPointer => _file.Pointer<short>(DfbFormat.TagWeights);

    // ---- neuron access -------------------------------------------------------------------

    /// <summary>Full annotation record of one neuron.</summary>
    public NeuronInfo this[int index]
    {
        get
        {
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual((uint)index, (uint)NeuronCount);
            ReadOnlySpan<int> soma = _file.Column<int>(DfbFormat.TagSoma).Slice(3 * index, 3);
            return new NeuronInfo(
                index,
                BodyIds[index],
                Str(DfbFormat.TagType, index),
                Str(DfbFormat.TagSuperclass, index),
                Str(DfbFormat.TagClass, index),
                Str(DfbFormat.TagInstance, index),
                Str(DfbFormat.TagFlywireType, index),
                Str(DfbFormat.TagHemibrainType, index),
                Neurotransmitters[index],
                Sides[index],
                soma[0] == int.MinValue ? null : (soma[0], soma[1], soma[2]),
                (int)(RowPtr[index + 1] - RowPtr[index]));
        }
    }

    /// <summary>Neuron index of a body ID, or −1.</summary>
    public int IndexOf(ulong bodyId) => _byBodyId.TryGetValue(bodyId, out int i) ? i : -1;

    /// <summary>A single-neuron set by body ID.</summary>
    /// <exception cref="KeyNotFoundException">The ID is not in this checkpoint.</exception>
    public NeuronSet ByBodyId(ulong bodyId)
    {
        int i = IndexOf(bodyId);
        return i < 0
            ? throw new KeyNotFoundException($"Body ID {bodyId} is not in checkpoint {Provenance.Name}.")
            : NeuronSet.From(bodyId.ToString(), [i]);
    }

    /// <summary>A set from body IDs; every ID must exist (no silent drops).</summary>
    public NeuronSet ByBodyIds(string name, ReadOnlySpan<ulong> bodyIds)
    {
        var idx = new int[bodyIds.Length];
        for (int k = 0; k < bodyIds.Length; k++)
        {
            int i = IndexOf(bodyIds[k]);
            idx[k] = i < 0 ? throw new KeyNotFoundException($"Body ID {bodyIds[k]} is not in checkpoint {Provenance.Name}.") : i;
        }

        return NeuronSet.From(name, idx);
    }

    /// <summary>Out-edges of a neuron: (postsynaptic index, signed contact count) pairs.</summary>
    public EdgeSpan OutEdges(int neuron)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual((uint)neuron, (uint)NeuronCount);
        ReadOnlySpan<long> rp = RowPtr;
        int start = checked((int)rp[neuron]);
        int len = checked((int)(rp[neuron + 1] - rp[neuron]));
        return new EdgeSpan(Columns.Slice(start, len), Weights.Slice(start, len));
    }

    /// <summary>
    /// Neurons matching all given criteria. String criteria match exactly, or as a prefix/suffix
    /// glob when they contain a trailing/leading <c>*</c>. Returns an empty set when nothing matches.
    /// </summary>
    public NeuronSet Query(
        string? type = null,
        string? superclass = null,
        string? cls = null,
        string? instance = null,
        string? flywireType = null,
        string? hemibrainType = null,
        Side? side = null,
        Neurotransmitter? nt = null,
        string? name = null)
    {
        var matches = new List<int>();
        ReadOnlySpan<uint> types = _file.Column<uint>(DfbFormat.TagType);
        ReadOnlySpan<uint> supers = _file.Column<uint>(DfbFormat.TagSuperclass);
        ReadOnlySpan<uint> classes = _file.Column<uint>(DfbFormat.TagClass);
        ReadOnlySpan<uint> instances = _file.Column<uint>(DfbFormat.TagInstance);
        ReadOnlySpan<uint> fw = _file.Column<uint>(DfbFormat.TagFlywireType);
        ReadOnlySpan<uint> hb = _file.Column<uint>(DfbFormat.TagHemibrainType);
        ReadOnlySpan<Side> sides = Sides;
        ReadOnlySpan<Neurotransmitter> nts = Neurotransmitters;

        StringMatcher mType = new(type), mSuper = new(superclass), mCls = new(cls), mInst = new(instance), mFw = new(flywireType), mHb = new(hemibrainType);

        for (int i = 0; i < NeuronCount; i++)
        {
            if (side is { } s && sides[i] != s)
            {
                continue;
            }

            if (nt is { } n && nts[i] != n)
            {
                continue;
            }

            if (!mType.Matches(_strings, types[i]) || !mSuper.Matches(_strings, supers[i]) || !mCls.Matches(_strings, classes[i])
                || !mInst.Matches(_strings, instances[i]) || !mFw.Matches(_strings, fw[i]) || !mHb.Matches(_strings, hb[i]))
            {
                continue;
            }

            matches.Add(i);
        }

        string setName = name ?? DescribeQuery(type, superclass, cls, instance, flywireType, hemibrainType, side, nt);
        return NeuronSet.FromSortedUnique(setName, matches.ToArray());
    }

    /// <summary>
    /// Neurons reachable from <paramref name="seed"/> within <paramref name="hops"/> forward hops
    /// over edges with |contact count| ≥ <paramref name="minWeight"/>, including the seed.
    /// </summary>
    public NeuronSet Downstream(NeuronSet seed, int hops, int minWeight = 1)
    {
        ArgumentNullException.ThrowIfNull(seed);
        ArgumentOutOfRangeException.ThrowIfNegative(hops);
        var visited = new bool[NeuronCount];
        var frontier = new List<int>(seed.Count);
        var result = new List<int>(seed.Count);
        foreach (int i in seed.Indices)
        {
            visited[i] = true;
            frontier.Add(i);
            result.Add(i);
        }

        ReadOnlySpan<long> rp = RowPtr;
        ReadOnlySpan<int> cols = Columns;
        ReadOnlySpan<short> w = Weights;
        for (int h = 0; h < hops && frontier.Count > 0; h++)
        {
            var next = new List<int>();
            foreach (int s in frontier)
            {
                for (long e = rp[s]; e < rp[s + 1]; e++)
                {
                    int c = cols[(int)e];
                    if (!visited[c] && Math.Abs((int)w[(int)e]) >= minWeight)
                    {
                        visited[c] = true;
                        next.Add(c);
                        result.Add(c);
                    }
                }
            }

            frontier = next;
        }

        return NeuronSet.From($"downstream({seed.Name}, hops={hops}, minWeight={minWeight})", result.ToArray());
    }

    /// <summary>
    /// Neurons from which <paramref name="target"/> is reachable within <paramref name="hops"/>
    /// forward hops over edges with |contact count| ≥ <paramref name="minWeight"/>, including the
    /// target. Each hop is one pass over the CSR (no in-edge index is stored), ~25 M edges for MaleCNS.
    /// </summary>
    public NeuronSet Upstream(NeuronSet target, int hops, int minWeight = 1)
    {
        ArgumentNullException.ThrowIfNull(target);
        ArgumentOutOfRangeException.ThrowIfNegative(hops);
        var inSet = new bool[NeuronCount];
        var frontier = new bool[NeuronCount];
        var result = new List<int>(target.Count);
        foreach (int i in target.Indices)
        {
            inSet[i] = true;
            frontier[i] = true;
            result.Add(i);
        }

        ReadOnlySpan<long> rp = RowPtr;
        ReadOnlySpan<int> cols = Columns;
        ReadOnlySpan<short> w = Weights;
        for (int h = 0; h < hops; h++)
        {
            var next = new bool[NeuronCount];
            bool any = false;
            for (int s = 0; s < NeuronCount; s++)
            {
                if (inSet[s])
                {
                    continue;
                }

                for (long e = rp[s]; e < rp[s + 1]; e++)
                {
                    if (frontier[cols[(int)e]] && Math.Abs((int)w[(int)e]) >= minWeight)
                    {
                        inSet[s] = true;
                        next[s] = true;
                        result.Add(s);
                        any = true;
                        break;
                    }
                }
            }

            if (!any)
            {
                break;
            }

            frontier = next;
        }

        return NeuronSet.From($"upstream({target.Name}, hops={hops}, minWeight={minWeight})", result.ToArray());
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _file.Dispose();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private string Str(uint tag, int index)
    {
        uint idx = _file.Column<uint>(tag)[index];
        return idx == 0 ? string.Empty : _strings[idx];
    }

    private static Dictionary<string, NeuronSet> ReadSets(DfbFile file)
    {
        var result = new Dictionary<string, NeuronSet>(StringComparer.Ordinal);
        if (!file.Has(DfbFormat.TagSets))
        {
            return result;
        }

        ReadOnlySpan<byte> dir = file.Bytes(DfbFormat.TagSets);
        uint count = BinaryPrimitives.ReadUInt32LittleEndian(dir);
        int namesStart = checked(4 + 12 * (int)count);
        StringTable names = StringTable.Deserialize(dir[namesStart..]);
        ReadOnlySpan<int> idx = file.Column<int>(DfbFormat.TagSetIndices);
        for (int i = 0; i < count; i++)
        {
            uint nameIdx = BinaryPrimitives.ReadUInt32LittleEndian(dir[(4 + 12 * i)..]);
            uint off = BinaryPrimitives.ReadUInt32LittleEndian(dir[(8 + 12 * i)..]);
            uint len = BinaryPrimitives.ReadUInt32LittleEndian(dir[(12 + 12 * i)..]);
            string name = names[nameIdx];
            result[name] = NeuronSet.FromSortedUnique(name, idx.Slice((int)off, (int)len).ToArray());
        }

        return result;
    }

    private static string DescribeQuery(string? type, string? superclass, string? cls, string? instance, string? fw, string? hb, Side? side, Neurotransmitter? nt)
    {
        var parts = new List<string>(8);
        Add("type", type);
        Add("superclass", superclass);
        Add("class", cls);
        Add("instance", instance);
        Add("flywireType", fw);
        Add("hemibrainType", hb);
        if (side is { } s)
        {
            parts.Add($"side={s}");
        }

        if (nt is { } n)
        {
            parts.Add($"nt={n}");
        }

        return parts.Count == 0 ? "all" : "query(" + string.Join(", ", parts) + ")";

        void Add(string k, string? v)
        {
            if (v is not null)
            {
                parts.Add($"{k}={v}");
            }
        }
    }

    private readonly ref struct StringMatcher
    {
        private readonly string? _pattern;
        private readonly bool _prefix;
        private readonly bool _suffix;

        public StringMatcher(string? pattern)
        {
            if (pattern is null)
            {
                _pattern = null;
                return;
            }

            _suffix = pattern.StartsWith('*');
            _prefix = pattern.EndsWith('*');
            _pattern = pattern.Trim('*');
        }

        public bool Matches(StringTable strings, uint idx)
        {
            if (_pattern is null)
            {
                return true;
            }

            string s = idx == 0 ? string.Empty : strings[idx];
            if (_prefix && _suffix)
            {
                return s.Contains(_pattern, StringComparison.Ordinal);
            }

            if (_prefix)
            {
                return s.StartsWith(_pattern, StringComparison.Ordinal);
            }

            if (_suffix)
            {
                return s.EndsWith(_pattern, StringComparison.Ordinal);
            }

            return string.Equals(s, _pattern, StringComparison.Ordinal);
        }
    }
}

/// <summary>Annotation record of one neuron.</summary>
/// <param name="Index">Neuron index within the checkpoint.</param>
/// <param name="BodyId">Body / root ID in the source dataset.</param>
/// <param name="Type">Cell type ("" if unannotated).</param>
/// <param name="Superclass">Superclass ("" if unannotated).</param>
/// <param name="Class">Class ("" if unannotated).</param>
/// <param name="Instance">Instance name ("" if unannotated).</param>
/// <param name="FlywireType">FlyWire cross-reference type ("" if none).</param>
/// <param name="HemibrainType">Hemibrain cross-reference type ("" if none).</param>
/// <param name="Neurotransmitter">Predicted transmitter.</param>
/// <param name="Side">Soma/root side.</param>
/// <param name="Soma">Soma voxel coordinates, if known.</param>
/// <param name="OutDegree">Number of outgoing edges.</param>
public readonly record struct NeuronInfo(
    int Index,
    ulong BodyId,
    string Type,
    string Superclass,
    string Class,
    string Instance,
    string FlywireType,
    string HemibrainType,
    Neurotransmitter Neurotransmitter,
    Side Side,
    (int X, int Y, int Z)? Soma,
    int OutDegree);

/// <summary>The out-edges of one neuron as parallel spans.</summary>
public readonly ref struct EdgeSpan
{
    /// <summary>Postsynaptic neuron indices.</summary>
    public ReadOnlySpan<int> Targets { get; }

    /// <summary>Signed contact counts, parallel to <see cref="Targets"/>.</summary>
    public ReadOnlySpan<short> Weights { get; }

    /// <summary>Number of edges.</summary>
    public int Length => Targets.Length;

    internal EdgeSpan(ReadOnlySpan<int> targets, ReadOnlySpan<short> weights)
    {
        Targets = targets;
        Weights = weights;
    }
}
