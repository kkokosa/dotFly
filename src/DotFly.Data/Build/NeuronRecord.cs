using DotFly.Core.Graph;
using DotFly.Core.Models;

namespace DotFly.Data.Build;

/// <summary>One neuron as gathered from a source dataset before it is written to a checkpoint.</summary>
public sealed record NeuronRecord
{
    /// <summary>Body / root ID.</summary>
    public required ulong BodyId { get; init; }

    /// <summary>Cell type, or null.</summary>
    public string? Type { get; init; }

    /// <summary>Superclass, or null.</summary>
    public string? Superclass { get; init; }

    /// <summary>Class, or null.</summary>
    public string? Class { get; init; }

    /// <summary>Instance name, or null.</summary>
    public string? Instance { get; init; }

    /// <summary>FlyWire cross-reference type, or null.</summary>
    public string? FlywireType { get; init; }

    /// <summary>Hemibrain cross-reference type, or null.</summary>
    public string? HemibrainType { get; init; }

    /// <summary>Neurotransmitter label.</summary>
    public Neurotransmitter Neurotransmitter { get; init; } = Neurotransmitter.Unknown;

    /// <summary>Side.</summary>
    public Side Side { get; init; } = Side.Unknown;

    /// <summary>Soma coordinates, or null.</summary>
    public (int X, int Y, int Z)? Soma { get; init; }
}
