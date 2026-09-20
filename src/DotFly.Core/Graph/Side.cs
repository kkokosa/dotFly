namespace DotFly.Core.Graph;

/// <summary>Soma / root side of a neuron as annotated in the source dataset.</summary>
public enum Side : byte
{
    /// <summary>Not annotated.</summary>
    Unknown = 0,

    /// <summary>Left.</summary>
    Left = 1,

    /// <summary>Right.</summary>
    Right = 2,

    /// <summary>Midline.</summary>
    Middle = 3,
}
