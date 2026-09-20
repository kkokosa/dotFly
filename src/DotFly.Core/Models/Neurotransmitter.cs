namespace DotFly.Core.Models;

/// <summary>
/// Predicted neurotransmitter of a neuron, as labelled in the connectome releases
/// (MaleCNS <c>consensusNt</c>, FlyWire <c>nt_type</c>). A single label per neuron is a modelling
/// convention, not receptor-resolved physiology.
/// </summary>
public enum Neurotransmitter : byte
{
    /// <summary>No confident prediction.</summary>
    Unknown = 0,

    /// <summary>Acetylcholine — the main fast excitatory transmitter in the fly.</summary>
    Acetylcholine = 1,

    /// <summary>GABA — fast inhibitory.</summary>
    Gaba = 2,

    /// <summary>Glutamate — treated as inhibitory in the fly central brain models.</summary>
    Glutamate = 3,

    /// <summary>Dopamine — a neuromodulator, reduced to a fast sign in point-neuron models.</summary>
    Dopamine = 4,

    /// <summary>Octopamine — a neuromodulator, reduced to a fast sign in point-neuron models.</summary>
    Octopamine = 5,

    /// <summary>Serotonin — a neuromodulator, reduced to a fast sign in point-neuron models.</summary>
    Serotonin = 6,

    /// <summary>Histamine — the photoreceptor transmitter; inhibitory in the optic lobe.</summary>
    Histamine = 7,
}

/// <summary>
/// Maps a neurotransmitter label to the sign of the presynaptic neuron's outgoing weights.
/// The policy is applied once by the checkpoint builder and recorded in the checkpoint's
/// provenance; kernels only ever see signed contact counts.
/// </summary>
public sealed record SignPolicy
{
    private readonly sbyte[] _signs;

    /// <summary>Human-readable name recorded in checkpoint provenance.</summary>
    public string Name { get; }

    private SignPolicy(string name, sbyte[] signs)
    {
        Name = name;
        _signs = signs;
    }

    /// <summary>Returns +1, −1 or 0 for the given transmitter.</summary>
    public int Sign(Neurotransmitter nt) => _signs[(int)nt];

    /// <summary>
    /// Shiu et al. 2024 convention: acetylcholine +1; GABA and glutamate −1; dopamine, octopamine
    /// and serotonin +1. The FlyWire v630/v783 inputs ship the signed column precomputed, so this
    /// policy is documentary for those files; for MaleCNS histamine and unknown labels are
    /// neutral (0), matching the Xenova MaleCNS configuration.
    /// </summary>
    public static SignPolicy Shiu2024 { get; } = new(
        "shiu2024",
        [
            /* Unknown       */ 0,
            /* Acetylcholine */ 1,
            /* Gaba          */ -1,
            /* Glutamate     */ -1,
            /* Dopamine      */ 1,
            /* Octopamine    */ 1,
            /* Serotonin     */ 1,
            /* Histamine     */ 0,
        ]);

    /// <summary>Creates a custom policy; the array is indexed by <see cref="Neurotransmitter"/>.</summary>
    /// <exception cref="ArgumentException">Wrong length or a sign outside {−1, 0, +1}.</exception>
    public static SignPolicy Custom(string name, ReadOnlySpan<sbyte> signs)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        if (signs.Length != 8)
        {
            throw new ArgumentException("Expected one sign per Neurotransmitter value (8).", nameof(signs));
        }

        foreach (sbyte s in signs)
        {
            if (s is < -1 or > 1)
            {
                throw new ArgumentException("Signs must be -1, 0 or +1.", nameof(signs));
            }
        }

        return new SignPolicy(name, signs.ToArray());
    }
}
