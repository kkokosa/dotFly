using System.Text.Json;
using System.Text.Json.Serialization;
using DotFly.Core.Models;

namespace DotFly.Core.Checkpoints;

/// <summary>
/// Everything needed to reproduce a checkpoint: where the data came from, which filter selected
/// the neurons, which sign policy produced the edge signs, and the neuron model the checkpoint was
/// built for. Stored as JSON inside the <c>.dfb</c> file.
/// </summary>
public sealed class Provenance
{
    /// <summary>Short checkpoint name, e.g. <c>malecns-v1.0-superclass</c>.</summary>
    public required string Name { get; init; }

    /// <summary>Source dataset identifier, e.g. <c>male-cns:v1.0</c> or <c>flywire:v630</c>.</summary>
    public required string Dataset { get; init; }

    /// <summary>Source files with SHA-256 hashes.</summary>
    public required IReadOnlyList<SourceFile> Sources { get; init; }

    /// <summary>Human-readable description of the neuron inclusion filter.</summary>
    public required string InclusionFilter { get; init; }

    /// <summary>Name of the <see cref="SignPolicy"/> that produced the edge signs, or <c>precomputed</c>.</summary>
    public required string SignPolicy { get; init; }

    /// <summary>The neuron model the checkpoint targets.</summary>
    public required NeuronModel Model { get; init; }

    /// <summary>Edge statistics of the source selection before zero-sign edges were dropped, when known.</summary>
    public EdgeStats? SourceEdges { get; init; }

    /// <summary>Name of the parent checkpoint when derived (sub-graph, shuffle, ablation), else null.</summary>
    public string? Parent { get; init; }

    /// <summary>Derivation description (e.g. <c>hops=2 from set sugar_GRN_R</c>), else null.</summary>
    public string? Derivation { get; init; }

    /// <summary>Seed of any randomised derivation (shuffles), else null.</summary>
    public ulong? Seed { get; init; }

    /// <summary>Version of the builder that produced the file.</summary>
    public required string BuilderVersion { get; init; }

    /// <summary>UTC build time.</summary>
    public required DateTimeOffset BuiltAt { get; init; }

    /// <summary>Serialises to JSON (indented).</summary>
    public string ToJson() => JsonSerializer.Serialize(this, ProvenanceJsonContext.Default.Provenance);

    /// <summary>Deserialises from JSON.</summary>
    public static Provenance FromJson(string json) =>
        JsonSerializer.Deserialize(json, ProvenanceJsonContext.Default.Provenance)
        ?? throw new InvalidDataException("Provenance JSON deserialised to null.");
}

/// <summary>A source file reference with its hash.</summary>
/// <param name="FileName">File name (no directory).</param>
/// <param name="Sha256">Lower-case hex SHA-256 of the file contents.</param>
/// <param name="Length">File length in bytes.</param>
public sealed record SourceFile(string FileName, string Sha256, long Length);

/// <summary>
/// Structural edge counts of the selected neurons in the source data. <see cref="Edges"/> and
/// <see cref="Contacts"/> count every (pre, post) pair among included neurons; the checkpoint
/// itself keeps only pairs whose sign is non-zero (the rest would contribute <c>g += 0</c>).
/// </summary>
/// <param name="Edges">Directed (pre, post) pairs among included neurons.</param>
/// <param name="Contacts">Sum of contact counts over those pairs.</param>
/// <param name="ZeroSignEdges">Pairs dropped because the presynaptic transmitter maps to sign 0.</param>
/// <param name="ZeroSignContacts">Contacts on those dropped pairs.</param>
public sealed record EdgeStats(long Edges, long Contacts, long ZeroSignEdges, long ZeroSignContacts);

/// <summary>Source-generated JSON context (AOT-safe).</summary>
[JsonSourceGenerationOptions(WriteIndented = true, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(Provenance))]
[JsonSerializable(typeof(NeuronModel))]
[JsonSerializable(typeof(SourceFile))]
[JsonSerializable(typeof(EdgeStats))]
internal sealed partial class ProvenanceJsonContext : JsonSerializerContext
{
}
