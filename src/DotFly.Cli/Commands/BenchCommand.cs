using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using DotFly.Core.Backends;
using DotFly.Core.Graph;
using DotFly.Cpu;
using DotFly.Cpu.Fast;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotFly.Cli.Commands;

/// <summary>
/// Throughput benchmark of the CPU backend on a checkpoint: runs a scenario for a fixed neural
/// duration at several thread counts and reports steps/s, real-time factor, spikes and synaptic
/// deliveries per second, and memory. Neural time, simulated duration and wall time are reported
/// separately, as PLAN.md §8.2 requires.
/// </summary>
public sealed class BenchCommand : Command<BenchCommand.Settings>
{
    /// <summary>Command settings.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Checkpoint path.</summary>
        [CommandArgument(0, "<checkpoint>")]
        [Description("Path to a .dfb checkpoint")]
        public required string Checkpoint { get; init; }

        /// <summary>Scenario.</summary>
        [CommandOption("--scenario <SPEC>")]
        [Description("idle (no input; default), or a stimulation spec: comma-separated body IDs, set:NAME or type:NAME driven at --hz")]
        [DefaultValue("idle")]
        public string Scenario { get; init; } = "idle";

        /// <summary>Poisson rate.</summary>
        [CommandOption("--hz <RATE>")]
        [Description("Poisson rate per stimulated neuron in Hz")]
        [DefaultValue(150.0)]
        public double Hz { get; init; } = 150.0;

        /// <summary>Duration.</summary>
        [CommandOption("--seconds <S>")]
        [Description("Simulated duration per run in seconds")]
        [DefaultValue(1.0)]
        public double Seconds { get; init; } = 1.0;

        /// <summary>Thread counts.</summary>
        [CommandOption("--threads <LIST>")]
        [Description("Comma-separated thread counts to try (default: 1 and one per physical core)")]
        public string? Threads { get; init; }

        /// <summary>Seed.</summary>
        [CommandOption("--seed <SEED>")]
        [DefaultValue(1UL)]
        public ulong Seed { get; init; } = 1;
    }

    private sealed class Counter : ISpikeSink
    {
        public long Total { get; private set; }

        public void OnStep(long step, ReadOnlySpan<int> spiked) => Total += spiked.Length;
    }

    /// <inheritdoc />
    protected override int Execute(CommandContext context, Settings settings, CancellationToken cancellationToken)
    {
        using Brain brain = Brain.Open(settings.Checkpoint);
        NeuronSet stim = settings.Scenario.Equals("idle", StringComparison.OrdinalIgnoreCase) ? NeuronSet.Empty : RunCommand.Resolve(brain, settings.Scenario, "stimulated");
        int[] threads = settings.Threads is null
            ? [1, CpuBackend.DefaultThreads()]
            : settings.Threads.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse).ToArray();

        const double dtMs = 0.1;
        int steps = (int)Math.Round(settings.Seconds * 1000.0 / dtMs);
        AnsiConsole.MarkupLineInterpolated($"[bold]{brain.Provenance.Name}[/]: {brain.NeuronCount:N0} neurons, {brain.EdgeCount:N0} edges; scenario [bold]{settings.Scenario}[/] ({stim.Count} neurons @ {settings.Hz} Hz), {settings.Seconds} s neural time (dt = {dtMs} ms, {steps:N0} steps), seed {settings.Seed}.");
        AnsiConsole.MarkupLineInterpolated($"[grey]{System.Runtime.InteropServices.RuntimeInformation.OSDescription}; {SimdInfo.Describe()}; {Environment.ProcessorCount} logical processors[/]");

        var table = new Table().Border(TableBorder.Rounded)
            .AddColumns("Threads", "Wall (s)", "Steps/s", "Real-time ×", "Spikes/s", "Deliveries/s", "Spikes", "Peak RSS (MB)");
        foreach (int th in threads)
        {
            using var sim = new CpuBackend(brain, settings.Seed, th, dtMs);
            double p = settings.Hz * dtMs * 1e-3;
            foreach (int i in stim.Indices)
            {
                sim.SetRefractorySteps(i, 0);
                sim.SetPoissonProbability(i, p);
            }

            // Warm-up: JIT and first-touch, then reset the network to the same initial state.
            sim.Advance(Math.Min(steps, 500), null);
            sim.Reset();
            long edges0 = sim.EdgesDelivered;

            var counter = new Counter();
            var sw = Stopwatch.StartNew();
            sim.Advance(steps, counter);
            sw.Stop();

            double wall = sw.Elapsed.TotalSeconds;
            long deliveries = sim.EdgesDelivered - edges0;
            table.AddRow(
                sim.Threads.ToString(CultureInfo.InvariantCulture),
                wall.ToString("F3", CultureInfo.InvariantCulture),
                (steps / wall).ToString("N0", CultureInfo.InvariantCulture),
                (settings.Seconds / wall).ToString("F2", CultureInfo.InvariantCulture),
                (counter.Total / wall).ToString("N0", CultureInfo.InvariantCulture),
                (deliveries / wall).ToString("N0", CultureInfo.InvariantCulture),
                counter.Total.ToString("N0", CultureInfo.InvariantCulture),
                (Process.GetCurrentProcess().PeakWorkingSet64 / 1e6).ToString("F0", CultureInfo.InvariantCulture));
        }

        AnsiConsole.Write(table);
        return 0;
    }
}
