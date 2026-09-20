using Apache.Arrow;
using Apache.Arrow.Compression;
using Apache.Arrow.Ipc;
using DotFly.Core.Checkpoints;
using DotFly.Core.Graph;
using DotFly.Core.Models;
using DotFly.Data.Build;

namespace DotFly.Data.Sources;

/// <summary>Neuron inclusion filters for MaleCNS.</summary>
public enum MaleCnsInclusion
{
    /// <summary>Bodies with a non-null <c>superclass</c> — the graph retained by the public demos (166,700 neurons in v1.0).</summary>
    SuperclassAnnotated,

    /// <summary>Bodies with <c>status == "Traced"</c> — matches the official <c>traced-only</c> export (165,122 neurons in v1.0).</summary>
    TracedOnly,
}

/// <summary>
/// Reads the MaleCNS flat-connectome Feather release (annotations, neurotransmitters, weights) and
/// produces neurons, edges and provenance for the checkpoint builder.
/// </summary>
public sealed class MaleCnsSource
{
    private static readonly CompressionCodecFactory s_codecs = new();

    /// <summary>Path to <c>body-annotations-…feather</c>.</summary>
    public required string AnnotationsPath { get; init; }

    /// <summary>Path to <c>body-neurotransmitters-…feather</c>.</summary>
    public required string NeurotransmittersPath { get; init; }

    /// <summary>Path to a <c>connectome-weights-…feather</c> file (full or traced-only).</summary>
    public required string WeightsPath { get; init; }

    /// <summary>Which bodies to include.</summary>
    public MaleCnsInclusion Inclusion { get; init; } = MaleCnsInclusion.SuperclassAnnotated;

    /// <summary>Sign policy applied to the presynaptic neuron's transmitter.</summary>
    public SignPolicy SignPolicy { get; init; } = SignPolicy.Shiu2024;

    /// <summary>Neuron model recorded in the checkpoint.</summary>
    public NeuronModel Model { get; init; } = NeuronModel.Shiu2024;

    /// <summary>Progress callback (message).</summary>
    public Action<string>? Log { get; init; }

    /// <summary>Reads everything and builds the checkpoint content.</summary>
    public CheckpointContent Read(string checkpointName)
    {
        Log?.Invoke("Reading annotations…");
        List<NeuronRecord> neurons = ReadNeurons();
        Log?.Invoke($"  {neurons.Count:N0} neurons included ({Inclusion}).");

        Log?.Invoke("Reading neurotransmitters…");
        Dictionary<ulong, Neurotransmitter> nts = ReadNeurotransmitters();
        for (int i = 0; i < neurons.Count; i++)
        {
            if (nts.TryGetValue(neurons[i].BodyId, out Neurotransmitter nt))
            {
                neurons[i] = neurons[i] with { Neurotransmitter = nt };
            }
        }

        var index = new Dictionary<ulong, int>(neurons.Count);
        for (int i = 0; i < neurons.Count; i++)
        {
            index[neurons[i].BodyId] = i;
        }

        Log?.Invoke("Reading weights…");
        (EdgeList edges, EdgeStats stats) = ReadEdges(index, neurons);
        Log?.Invoke($"  {stats.Edges:N0} edges / {stats.Contacts:N0} contacts among included neurons; {stats.ZeroSignEdges:N0} edges / {stats.ZeroSignContacts:N0} contacts dropped (presynaptic sign 0).");

        var provenance = new Provenance
        {
            Name = checkpointName,
            Dataset = "male-cns:v1.0",
            Sources = [SourceFiles.Describe(AnnotationsPath), SourceFiles.Describe(NeurotransmittersPath), SourceFiles.Describe(WeightsPath)],
            InclusionFilter = Inclusion switch
            {
                MaleCnsInclusion.SuperclassAnnotated => "superclass IS NOT NULL",
                MaleCnsInclusion.TracedOnly => "status == 'Traced'",
                _ => Inclusion.ToString(),
            },
            SignPolicy = SignPolicy.Name,
            SourceEdges = stats,
            Model = Model,
            BuilderVersion = SourceFiles.BuilderVersion,
            BuiltAt = DateTimeOffset.UtcNow,
        };

        Log?.Invoke("Building CSR…");
        return CheckpointBuilder.Build(provenance, neurons, edges, []);
    }

    private List<NeuronRecord> ReadNeurons()
    {
        var neurons = new List<NeuronRecord>(170_000);
        foreach (RecordBatch batch in Batches(AnnotationsPath))
        {
            var bodyId = (Int64Array)batch.Column("bodyId");
            var type = (StringArray)batch.Column("type");
            var superclass = (StringArray)batch.Column("superclass");
            var cls = (StringArray)batch.Column("class");
            var instance = (StringArray)batch.Column("instance");
            var flywireType = (StringArray)batch.Column("flywireType");
            var hemibrainType = (StringArray)batch.Column("hemibrainType");
            var somaSide = (StringArray)batch.Column("somaSide");
            var rootSide = (StringArray)batch.Column("rootSide");
            var status = (StringArray)batch.Column("status");
            var soma = (ListArray)batch.Column("somaLocation");

            for (int i = 0; i < batch.Length; i++)
            {
                bool include = Inclusion switch
                {
                    MaleCnsInclusion.SuperclassAnnotated => !superclass.IsNull(i),
                    MaleCnsInclusion.TracedOnly => !status.IsNull(i) && status.GetString(i) == "Traced",
                    _ => false,
                };
                if (!include)
                {
                    continue;
                }

                (int, int, int)? somaXyz = null;
                if (!soma.IsNull(i))
                {
                    var values = (Int64Array)soma.GetSlicedValues(i);
                    if (values.Length == 3)
                    {
                        somaXyz = (checked((int)values.GetValue(0)!.Value), checked((int)values.GetValue(1)!.Value), checked((int)values.GetValue(2)!.Value));
                    }
                }

                neurons.Add(new NeuronRecord
                {
                    BodyId = checked((ulong)bodyId.GetValue(i)!.Value),
                    Type = Nullable(type, i),
                    Superclass = Nullable(superclass, i),
                    Class = Nullable(cls, i),
                    Instance = Nullable(instance, i),
                    FlywireType = Nullable(flywireType, i),
                    HemibrainType = Nullable(hemibrainType, i),
                    Side = ParseSide(Nullable(somaSide, i) ?? Nullable(rootSide, i)),
                    Soma = somaXyz,
                });
            }
        }

        return neurons;
    }

    private Dictionary<ulong, Neurotransmitter> ReadNeurotransmitters()
    {
        var map = new Dictionary<ulong, Neurotransmitter>(200_000);
        foreach (RecordBatch batch in Batches(NeurotransmittersPath))
        {
            var body = (Int64Array)batch.Column("body");
            var consensus = (StringArray)batch.Column("consensus_nt");
            for (int i = 0; i < batch.Length; i++)
            {
                Neurotransmitter nt = ParseNt(Nullable(consensus, i));
                if (nt != Neurotransmitter.Unknown)
                {
                    map[checked((ulong)body.GetValue(i)!.Value)] = nt;
                }
            }
        }

        return map;
    }

    private (EdgeList Edges, EdgeStats Stats) ReadEdges(Dictionary<ulong, int> index, List<NeuronRecord> neurons)
    {
        var edges = new EdgeList(1 << 24);
        long rows = 0, structuralEdges = 0, structuralContacts = 0, zeroEdges = 0, zeroContacts = 0;
        foreach (RecordBatch batch in Batches(WeightsPath))
        {
            var pre = (Int64Array)batch.Column("body_pre");
            var post = (Int64Array)batch.Column("body_post");
            var weight = (Int64Array)batch.Column("weight");
            ReadOnlySpan<long> preV = pre.Values;
            ReadOnlySpan<long> postV = post.Values;
            ReadOnlySpan<long> wV = weight.Values;
            for (int i = 0; i < batch.Length; i++)
            {
                rows++;
                if (!index.TryGetValue((ulong)preV[i], out int a) || !index.TryGetValue((ulong)postV[i], out int b))
                {
                    continue;
                }

                structuralEdges++;
                structuralContacts += wV[i];
                int sign = SignPolicy.Sign(neurons[a].Neurotransmitter);
                if (sign == 0)
                {
                    zeroEdges++;
                    zeroContacts += wV[i];
                    continue;
                }

                edges.Add(a, b, checked((int)(sign * wV[i])));
            }
        }

        Log?.Invoke($"  {rows:N0} weight rows scanned.");
        return (edges, new EdgeStats(structuralEdges, structuralContacts, zeroEdges, zeroContacts));
    }

    private static IEnumerable<RecordBatch> Batches(string path)
    {
        using FileStream fs = new(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20);
        using var reader = new ArrowFileReader(fs, s_codecs);
        while (reader.ReadNextRecordBatch() is { } batch)
        {
            using (batch)
            {
                yield return batch;
            }
        }
    }

    private static string? Nullable(StringArray a, int i) => a.IsNull(i) ? null : a.GetString(i) is { Length: > 0 } s ? s : null;

    private static Side ParseSide(string? s) => s switch
    {
        "L" => Side.Left,
        "R" => Side.Right,
        "M" => Side.Middle,
        _ => Side.Unknown,
    };

    /// <summary>Maps a MaleCNS <c>consensus_nt</c> label to the enum; unknown labels (incl. "unclear") → <see cref="Neurotransmitter.Unknown"/>.</summary>
    public static Neurotransmitter ParseNt(string? s) => s switch
    {
        "acetylcholine" => Neurotransmitter.Acetylcholine,
        "gaba" => Neurotransmitter.Gaba,
        "glutamate" => Neurotransmitter.Glutamate,
        "dopamine" => Neurotransmitter.Dopamine,
        "octopamine" => Neurotransmitter.Octopamine,
        "serotonin" => Neurotransmitter.Serotonin,
        "histamine" => Neurotransmitter.Histamine,
        _ => Neurotransmitter.Unknown,
    };
}
