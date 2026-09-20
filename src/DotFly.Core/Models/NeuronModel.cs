namespace DotFly.Core.Models;

/// <summary>
/// Parameters of the leaky integrate-and-fire (LIF) point-neuron model used by dotFly.
/// Defaults reproduce Shiu et al. 2024 (<c>philshiu/Drosophila_brain_model@91bdd1e7</c>, <c>model.py</c>):
/// <code>
/// dv/dt = (v_0 - v + g) / t_mbr   (unless refractory)
/// dg/dt = -g / tau                (unless refractory)
/// threshold: v &gt; v_th;  reset: v = v_rst, g = 0;  refractory t_rfc
/// synapse:   g += w (after delay t_dly), w = signed contact count × w_syn
/// </code>
/// All voltages are in millivolts and all times in milliseconds so the descriptor is
/// unit-free at the kernel boundary. Use <see cref="Discretize"/> to obtain the per-step
/// constants of the exact (matrix-exponential) integrator.
/// </summary>
public readonly record struct NeuronModel
{
    /// <summary>Resting potential <c>v_0</c> in mV. Default −52.</summary>
    public double RestMv { get; init; } = -52.0;

    /// <summary>Reset potential <c>v_rst</c> in mV. Default −52.</summary>
    public double ResetMv { get; init; } = -52.0;

    /// <summary>Spike threshold <c>v_th</c> in mV; a spike is emitted when <c>v &gt; v_th</c>. Default −45.</summary>
    public double ThresholdMv { get; init; } = -45.0;

    /// <summary>Membrane time constant <c>t_mbr</c> in ms. Default 20.</summary>
    public double MembraneTauMs { get; init; } = 20.0;

    /// <summary>Synaptic time constant <c>tau</c> in ms. Default 5.</summary>
    public double SynapticTauMs { get; init; } = 5.0;

    /// <summary>Absolute refractory period <c>t_rfc</c> in ms. Default 2.2.</summary>
    public double RefractoryMs { get; init; } = 2.2;

    /// <summary>Homogeneous synaptic transmission delay <c>t_dly</c> in ms. Default 1.8.</summary>
    public double DelayMs { get; init; } = 1.8;

    /// <summary>Unitary synaptic weight <c>w_syn</c> in mV per contact. Default 0.275.</summary>
    public double SynapseWeightMv { get; init; } = 0.275;

    /// <summary>
    /// Voltage added to <c>v</c> by one Poisson input event, <c>w_syn × f_poi</c>, in mV.
    /// Default 0.275 × 250 = 68.75, i.e. every event fires the neuron at the next threshold check.
    /// </summary>
    public double PoissonWeightMv { get; init; } = 0.275 * 250.0;

    /// <summary>Creates a model with the Shiu et al. 2024 defaults.</summary>
    public NeuronModel()
    {
    }

    /// <summary>The Shiu et al. 2024 model with all defaults.</summary>
    public static NeuronModel Shiu2024 => new();

    /// <summary>
    /// Computes the per-step constants of the exact integrator for a given timestep.
    /// </summary>
    /// <param name="dtMs">Integration timestep in ms (Brian2 default and all inspected demos: 0.1).</param>
    /// <exception cref="ArgumentOutOfRangeException">Timestep is not positive, or the two time constants are equal (the closed form has a removable singularity there that is not implemented).</exception>
    public LifStepConstants Discretize(double dtMs)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dtMs);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MembraneTauMs);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(SynapticTauMs);
        if (MembraneTauMs == SynapticTauMs)
        {
            throw new ArgumentOutOfRangeException(nameof(dtMs), "Membrane and synaptic time constants must differ.");
        }

        // Closed-form solution over one step of the linear system
        //   du/dt = (-u + g) / τm,   dg/dt = -g / τs,   u = v - v0
        //   u' = a·u + b·g,   g' = c·g
        // with a = e^{-dt/τm}, c = e^{-dt/τs}, b = τs/(τs − τm)·(c − a).
        // This is what Brian2's method='linear' produces (as a matrix exponential).
        double a = Math.Exp(-dtMs / MembraneTauMs);
        double c = Math.Exp(-dtMs / SynapticTauMs);
        double b = SynapticTauMs / (SynapticTauMs - MembraneTauMs) * (c - a);

        return new LifStepConstants(
            DtMs: dtMs,
            A: a,
            B: b,
            C: c,
            RestMv: RestMv,
            ResetMv: ResetMv,
            ThresholdMv: ThresholdMv,
            SynapseWeightMv: SynapseWeightMv,
            PoissonWeightMv: PoissonWeightMv,
            RefractorySteps: BrianTimestep(RefractoryMs, dtMs),
            DelaySteps: BrianDelaySteps(DelayMs, dtMs));
    }

    /// <summary>
    /// Brian2's <c>timestep(t, dt)</c>: <c>floor(t/dt + 1e-3)</c>. Used for the refractory
    /// comparison, which Brian2 performs in integer steps, not in floating-point time.
    /// </summary>
    internal static int BrianTimestep(double tMs, double dtMs)
        => checked((int)Math.Floor(tMs / dtMs + 1e-3));

    /// <summary>
    /// Brian2's spike-queue delay quantisation: <c>round(delay/dt)</c> to the nearest step.
    /// </summary>
    internal static int BrianDelaySteps(double delayMs, double dtMs)
        => checked((int)Math.Round(delayMs / dtMs, MidpointRounding.AwayFromZero));
}

/// <summary>
/// Per-step constants of the exact LIF integrator for a fixed timestep, as consumed by kernels.
/// Produced by <see cref="NeuronModel.Discretize"/>; all values are in mV / ms / steps.
/// </summary>
/// <param name="DtMs">Timestep in ms.</param>
/// <param name="A">Membrane decay factor per step, <c>e^{-dt/τm}</c>.</param>
/// <param name="B">Coupling of <c>g</c> into <c>v</c> per step (see <see cref="NeuronModel.Discretize"/>).</param>
/// <param name="C">Synaptic decay factor per step, <c>e^{-dt/τs}</c>.</param>
/// <param name="RestMv">Resting potential in mV.</param>
/// <param name="ResetMv">Reset potential in mV.</param>
/// <param name="ThresholdMv">Spike threshold in mV (strict <c>&gt;</c>).</param>
/// <param name="SynapseWeightMv">Weight per contact in mV.</param>
/// <param name="PoissonWeightMv">Voltage added by one Poisson event in mV.</param>
/// <param name="RefractorySteps">Number of steps a neuron stays frozen after a spike.</param>
/// <param name="DelaySteps">Number of steps between a presynaptic spike and its delivery.</param>
public readonly record struct LifStepConstants(
    double DtMs,
    double A,
    double B,
    double C,
    double RestMv,
    double ResetMv,
    double ThresholdMv,
    double SynapseWeightMv,
    double PoissonWeightMv,
    int RefractorySteps,
    int DelaySteps)
{
    /// <summary>Length of the spike ring buffer needed to honour <see cref="DelaySteps"/>: one slot per delay step plus the current one.</summary>
    public int RingSlots => DelaySteps + 1;
}
