namespace DotFly;

/// <summary>Which backend runs the network.</summary>
public enum Backend
{
    /// <summary>Float32 SIMD, multi-threaded (default).</summary>
    Cpu,

    /// <summary>Float64 scalar reference — slow, for validation.</summary>
    Reference,
}

/// <summary>Options for <see cref="BrainExtensions.CreateSimulation"/>.</summary>
public sealed record SimulationOptions
{
    /// <summary>Backend. Default <see cref="Backend.Cpu"/>.</summary>
    public Backend Backend { get; init; } = Backend.Cpu;

    /// <summary>Threads for the CPU backend; 0 = one per physical core.</summary>
    public int Threads { get; init; }

    /// <summary>
    /// For several simulations in one process: the worker slots already taken by the others
    /// (<see cref="ThreadPinOffset"/>) and the slots of all of them (<see cref="ThreadPinTotal"/>),
    /// so each simulation's pinned workers land on different cores. 0 = a single simulation.
    /// </summary>
    public int ThreadPinOffset { get; init; }

    /// <inheritdoc cref="ThreadPinOffset"/>
    public int ThreadPinTotal { get; init; }

    /// <summary>Random seed for stochastic inputs. Same seed + same inputs ⇒ same spikes.</summary>
    public ulong Seed { get; init; } = 1;

    /// <summary>Integration timestep in ms. Default 0.1 (Brian2 default; all inspected demos).</summary>
    public double DtMs { get; init; } = 0.1;

    /// <summary>
    /// Multiplier on the checkpoint's unitary synaptic weight (<c>w_syn</c>, 0.275 mV in Shiu et
    /// al., calibrated on the sugar → feeding experiment). It is the model's one global free
    /// parameter; anything other than 1 is a <em>calibration</em> and must be reported as such.
    /// </summary>
    public double SynapticGain { get; init; } = 1.0;

    /// <summary>
    /// Shiu et al. semantics for Poisson-driven neurons: their refractory period is set to zero
    /// while a Poisson port drives them (<c>model.py</c>, <c>poi()</c>). Default true.
    /// </summary>
    public bool ShiuPoissonSemantics { get; init; } = true;
}
