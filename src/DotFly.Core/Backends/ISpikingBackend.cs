using DotFly.Core.Graph;
using DotFly.Core.Models;

namespace DotFly.Core.Backends;

/// <summary>
/// Receives the spikes emitted at each step. Implementations must not retain the span.
/// </summary>
public interface ISpikeSink
{
    /// <summary>Called once per step with the (ascending) indices of the neurons that spiked.</summary>
    void OnStep(long step, ReadOnlySpan<int> spiked);
}

/// <summary>
/// The kernel-level contract of a simulation backend: owns the state of one network instance,
/// advances it in fixed steps, and exposes the per-neuron controls the engine layer builds on.
/// Everything here is index-based and allocation-free on the step path.
/// </summary>
public interface ISpikingBackend : IDisposable
{
    /// <summary>The graph being simulated.</summary>
    Brain Brain { get; }

    /// <summary>Per-step constants in use.</summary>
    LifStepConstants Constants { get; }

    /// <summary>Number of neurons.</summary>
    int NeuronCount { get; }

    /// <summary>Number of steps completed since the last <see cref="Reset()"/>.</summary>
    long Step { get; }

    /// <summary>Random seed used for stochastic inputs.</summary>
    ulong Seed { get; }

    /// <summary>Resets all state like <see cref="Reset()"/> and changes the seed, so the next run draws a different Poisson stream.</summary>
    void Reset(ulong seed);

    /// <summary>Membrane potential of a neuron in mV (diagnostic; not for hot paths).</summary>
    double GetV(int neuron);

    /// <summary>Synaptic variable of a neuron in mV (diagnostic).</summary>
    double GetG(int neuron);

    /// <summary>
    /// Sets the refractory duration of one neuron in steps. Shiu et al. set it to 0 for
    /// Poisson-stimulated neurons.
    /// </summary>
    void SetRefractorySteps(int neuron, int steps);

    /// <summary>
    /// Sets the per-step probability of a Poisson event on a neuron (0 disables). Each event
    /// adds <see cref="LifStepConstants.PoissonWeightMv"/> to <c>v</c> after delivery and
    /// before reset, matching Brian2's <c>PoissonInput(target_var='v')</c>.
    /// </summary>
    void SetPoissonProbability(int neuron, double probabilityPerStep);

    /// <summary>
    /// Sets a constant depolarising current for a neuron: <paramref name="mvPerStep"/> is added to
    /// <c>v</c> every step (0 disables), in the delivery phase, discarded while refractory like any input.
    /// </summary>
    void SetCurrent(int neuron, double mvPerStep);

    /// <summary>Whether a neuron transmits to its targets (false = Shiu-style silencing: outgoing weights zeroed).</summary>
    void SetTransmits(int neuron, bool transmits);

    /// <summary>
    /// Schedules an external voltage injection <paramref name="mv"/> on <paramref name="neuron"/>
    /// at absolute <paramref name="step"/> (≥ current step); applied in the delivery phase of that step.
    /// </summary>
    void ScheduleInjection(long step, int neuron, double mv);

    /// <summary>Global switch: when false, no spike is delivered to any target (recurrent transmission off).</summary>
    bool RecurrentTransmission { get; set; }

    /// <summary>Global switch: when false, Poisson inputs and scheduled injections are ignored.</summary>
    bool ExternalInput { get; set; }

    /// <summary>Advances the network by <paramref name="steps"/> steps, reporting spikes to <paramref name="sink"/> (may be null).</summary>
    void Advance(int steps, ISpikeSink? sink);

    /// <summary>Resets all state (v, g, refractory counters, pending spikes and injections) and the step counter; controls (rates, masks) are kept.</summary>
    void Reset();

    /// <summary>Serialises the complete dynamic state (v, g, refractory, pending spikes, RNG counters, step) so that <see cref="RestoreState"/> continues bit-identically. Controls are not included.</summary>
    byte[] SaveState();

    /// <summary>Restores a state produced by <see cref="SaveState"/> on a backend of the same checkpoint and constants. Pending scheduled injections are cleared.</summary>
    void RestoreState(ReadOnlySpan<byte> state);
}
