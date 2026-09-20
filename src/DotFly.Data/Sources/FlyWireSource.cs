using System.Globalization;
using DotFly.Core.Checkpoints;
using DotFly.Core.Models;
using DotFly.Data.Build;
using Parquet;
using Parquet.Schema;

namespace DotFly.Data.Sources;

/// <summary>
/// Reads the FlyWire inputs shipped with the Shiu et al. 2024 model: a completeness CSV whose
/// row order defines the neuron index, and a connectivity Parquet with precomputed signed
/// weights (<c>Excitatory x Connectivity</c>). Neuron indices in the checkpoint equal the Brian2
/// indices of the original model, so results can be compared directly.
/// </summary>
public sealed class FlyWireSource
{
    /// <summary>Path to the completeness CSV (<c>root_id,Completed</c>).</summary>
    public required string CompletenessPath { get; init; }

    /// <summary>Path to the connectivity Parquet.</summary>
    public required string ConnectivityPath { get; init; }

    /// <summary>Materialisation label, e.g. <c>630</c> or <c>783</c>.</summary>
    public required string Materialization { get; init; }

    /// <summary>Neuron model recorded in the checkpoint.</summary>
    public NeuronModel Model { get; init; } = NeuronModel.Shiu2024;

    /// <summary>Progress callback (message).</summary>
    public Action<string>? Log { get; init; }

    /// <summary>Reads everything and builds the checkpoint content.</summary>
    public async Task<CheckpointContent> ReadAsync(string checkpointName, CancellationToken cancellationToken = default)
    {
        Log?.Invoke("Reading completeness…");
        List<NeuronRecord> neurons = ReadCompleteness();
        Log?.Invoke($"  {neurons.Count:N0} neurons.");

        Log?.Invoke("Reading connectivity…");
        EdgeList edges = await ReadConnectivityAsync(neurons, cancellationToken).ConfigureAwait(false);
        Log?.Invoke($"  {edges.Count:N0} edges.");

        var provenance = new Provenance
        {
            Name = checkpointName,
            Dataset = $"flywire:v{Materialization}",
            Sources = [SourceFiles.Describe(CompletenessPath), SourceFiles.Describe(ConnectivityPath)],
            InclusionFilter = "all rows of the completeness table (Shiu et al. 2024 proofread set)",
            SignPolicy = "precomputed (Excitatory x Connectivity)",
            Model = Model,
            BuilderVersion = SourceFiles.BuilderVersion,
            BuiltAt = DateTimeOffset.UtcNow,
        };

        Log?.Invoke("Building CSR…");
        return CheckpointBuilder.Build(provenance, neurons, edges, []);
    }

    private List<NeuronRecord> ReadCompleteness()
    {
        var neurons = new List<NeuronRecord>(140_000);
        using var reader = new StreamReader(CompletenessPath);
        string? header = reader.ReadLine();
        if (header is null)
        {
            throw new InvalidDataException("Completeness CSV is empty.");
        }

        while (reader.ReadLine() is { } line)
        {
            if (line.Length == 0)
            {
                continue;
            }

            int comma = line.IndexOf(',');
            ReadOnlySpan<char> id = comma < 0 ? line : line.AsSpan(0, comma);
            neurons.Add(new NeuronRecord { BodyId = ulong.Parse(id, NumberStyles.None, CultureInfo.InvariantCulture) });
        }

        return neurons;
    }

    private async Task<EdgeList> ReadConnectivityAsync(IReadOnlyList<NeuronRecord> neurons, CancellationToken ct)
    {
        var edges = new EdgeList(1 << 24);
        await using ParquetReader reader = await ParquetReader.CreateAsync(ConnectivityPath, cancellationToken: ct).ConfigureAwait(false);
        DataField[] fields = reader.Schema.DataFields;
        DataField preField = fields.Single(f => f.Name == "Presynaptic_Index");
        DataField postField = fields.Single(f => f.Name == "Postsynaptic_Index");
        DataField wField = fields.Single(f => f.Name == "Excitatory x Connectivity");
        DataField idField = fields.Single(f => f.Name == "Presynaptic_ID");

        for (int g = 0; g < reader.RowGroupCount; g++)
        {
            using ParquetRowGroupReader rg = reader.OpenRowGroupReader(g);
            int rows = checked((int)rg.RowCount);
            var pre = new long[rows];
            var post = new long[rows];
            var w = new long[rows];
            var preId = new long[rows];
            var defLevels = new int[rows];
            await ReadRequiredAsync(rg, preField, pre, defLevels, ct).ConfigureAwait(false);
            await ReadRequiredAsync(rg, postField, post, defLevels, ct).ConfigureAwait(false);
            await ReadRequiredAsync(rg, wField, w, defLevels, ct).ConfigureAwait(false);
            await ReadRequiredAsync(rg, idField, preId, defLevels, ct).ConfigureAwait(false);
            for (int i = 0; i < rows; i++)
            {
                int a = checked((int)pre[i]);
                if ((uint)a >= (uint)neurons.Count || neurons[a].BodyId != (ulong)preId[i])
                {
                    throw new InvalidDataException($"Connectivity row {i} of group {g}: Presynaptic_Index {a} does not map to Presynaptic_ID {preId[i]} in the completeness table.");
                }

                if (w[i] == 0)
                {
                    continue;
                }

                edges.Add(a, checked((int)post[i]), checked((int)w[i]));
            }
        }

        return edges;
    }

    /// <summary>
    /// Reads a column that the writer marked optional but that must not contain nulls, using the
    /// raw API so values land directly in <paramref name="values"/> without a <c>long?[]</c> copy.
    /// </summary>
    private static async Task ReadRequiredAsync(ParquetRowGroupReader rg, DataField field, long[] values, int[] defLevels, CancellationToken ct)
    {
        await rg.ReadRawAsync<long>(field, values, field.MaxDefinitionLevel > 0 ? defLevels : null, null, ct).ConfigureAwait(false);
        if (field.MaxDefinitionLevel > 0)
        {
            foreach (int d in defLevels)
            {
                if (d != field.MaxDefinitionLevel)
                {
                    throw new InvalidDataException($"Column '{field.Name}' contains nulls; the connectivity table must be complete.");
                }
            }
        }
    }
}
