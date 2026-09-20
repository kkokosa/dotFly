using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using DotFly.Core.Backends;
using DotFly.Core.Graph;
using DotFly.Cpu.Fast;
using DotFly.Cpu.Reference;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotFly.Cli.Commands;

/// <summary>
/// Runs a Shiu-style activation experiment: Poisson-drive a set of neurons and report firing rates.
/// </summary>
public sealed class RunCommand : Command<RunCommand.Settings>
{
    /// <summary>Command settings.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Checkpoint path.</summary>
        [CommandArgument(0, "<checkpoint>")]
        [Description("Path to a .dfb checkpoint")]
        public required string Checkpoint { get; init; }

        /// <summary>Neurons to stimulate.</summary>
        [CommandOption("--stimulate <IDS>")]
        [Description("Comma-separated body IDs, set:NAME, or type:/superclass:/class:NAME[@L|@R] to drive with Poisson input")]
        public string? Stimulate { get; init; }

        /// <summary>Poisson rate.</summary>
        [CommandOption("--hz <RATE>")]
        [Description("Poisson rate per stimulated neuron in Hz")]
        [DefaultValue(150.0)]
        public double Hz { get; init; } = 150.0;

        /// <summary>Silenced neurons.</summary>
        [CommandOption("--silence <IDS>")]
        [Description("Comma-separated body IDs whose outgoing synapses are zeroed")]
        public string? Silence { get; init; }

        /// <summary>Duration.</summary>
        [CommandOption("--seconds <S>")]
        [Description("Simulated duration in seconds")]
        [DefaultValue(1.0)]
        public double Seconds { get; init; } = 1.0;

        /// <summary>Stimulus duration.</summary>
        [CommandOption("--stim-seconds <S>")]
        [Description("Stimulate only for the first S seconds (default: the whole run); spike counts are reported per phase")]
        public double? StimSeconds { get; init; }

        /// <summary>Seed.</summary>
        [CommandOption("--seed <SEED>")]
        [Description("Random seed for the Poisson inputs")]
        [DefaultValue(1UL)]
        public ulong Seed { get; init; } = 1;

        /// <summary>Backend.</summary>
        [CommandOption("--backend <NAME>")]
        [Description("cpu (float32 SIMD, multi-threaded; default) or reference (float64 scalar)")]
        [DefaultValue("cpu")]
        public string Backend { get; init; } = "cpu";

        /// <summary>Threads.</summary>
        [CommandOption("--threads <N>")]
        [Description("Threads for the cpu backend (0 = all logical cores)")]
        [DefaultValue(0)]
        public int Threads { get; init; }

        /// <summary>Synaptic gain.</summary>
        [CommandOption("--gain <G>")]
        [Description("Multiplier on the unitary synaptic weight (1 = the checkpoint's model; anything else is a calibration)")]
        [DefaultValue(1.0)]
        public double Gain { get; init; } = 1.0;

        /// <summary>Rows to print.</summary>
        [CommandOption("--top <N>")]
        [Description("Print the N most active neurons")]
        [DefaultValue(25)]
        public int Top { get; init; } = 25;

        /// <summary>CSV output.</summary>
        [CommandOption("--csv <FILE>")]
        [Description("Write per-neuron spike counts (bodyId,index,type,count) to a CSV")]
        public string? Csv { get; init; }
    }

    private sealed class Counter(int n) : ISpikeSink
    {
        public int[] Counts { get; } = new int[n];

        public long Total { get; private set; }

        public void OnStep(long step, ReadOnlySpan<int> spiked)
        {
            foreach (int i in spiked)
            {
                Counts[i]++;
            }

            Total += spiked.Length;
        }
    }

    /// <inheritdoc />
    protected override int Execute(CommandContext context, Settings settings, CancellationToken cancellationToken)
    {
        using Brain brain = Brain.Open(settings.Checkpoint);
        NeuronSet stim = settings.Stimulate is null ? NeuronSet.Empty : Resolve(brain, settings.Stimulate, "stimulated");
        NeuronSet silence = settings.Silence is null ? NeuronSet.Empty : Resolve(brain, settings.Silence, "silenced");

        const double dtMs = 0.1;
        int steps = (int)Math.Round(settings.Seconds * 1000.0 / dtMs);
        Core.Models.LifStepConstants constants = (brain.Model with { SynapseWeightMv = brain.Model.SynapseWeightMv * settings.Gain }).Discretize(dtMs);
        using ISpikingBackend sim = settings.Backend.ToLowerInvariant() switch
        {
            "cpu" => new CpuBackend(brain, constants, settings.Seed, settings.Threads),
            "reference" => new ReferenceBackend(brain, constants, settings.Seed),
            _ => throw new ArgumentException($"Unknown backend '{settings.Backend}'."),
        };
        string backendName = sim is CpuBackend c ? $"cpu backend, {c.Threads} threads, float32" : "reference backend, float64";
        double p = settings.Hz * dtMs * 1e-3;
        foreach (int i in stim.Indices)
        {
            sim.SetRefractorySteps(i, 0);
            sim.SetPoissonProbability(i, p);
        }

        foreach (int i in silence.Indices)
        {
            sim.SetTransmits(i, false);
        }

        AnsiConsole.MarkupLineInterpolated($"[grey]{brain.Provenance.Name}[/]: {brain.NeuronCount:N0} neurons, {brain.EdgeCount:N0} edges. Stimulating {stim.Count} neurons at {settings.Hz} Hz, silencing {silence.Count}, {settings.Seconds} s neural time, seed {settings.Seed}.");
        var counter = new Counter(brain.NeuronCount);
        var sw = Stopwatch.StartNew();
        if (settings.StimSeconds is { } stimSeconds && stimSeconds < settings.Seconds)
        {
            int stimSteps = (int)Math.Round(stimSeconds * 1000.0 / dtMs);
            sim.Advance(stimSteps, counter);
            long duringSpikes = counter.Total;
            var during = (int[])counter.Counts.Clone();
            foreach (int i in stim.Indices)
            {
                sim.SetPoissonProbability(i, 0);
            }

            sim.Advance(steps - stimSteps, counter);
            long afterSpikes = counter.Total - duringSpikes;
            AnsiConsole.MarkupLineInterpolated($"[grey]stimulus on for {stimSeconds} s: {duringSpikes:N0} spikes ({duringSpikes / stimSeconds:N0}/s); off for {settings.Seconds - stimSeconds} s: {afterSpikes:N0} spikes ({afterSpikes / (settings.Seconds - stimSeconds):N0}/s)[/]");
            foreach (int i in brain.Query(type: "DM1_lPN").Indices.ToArray().Concat(brain.Query(type: "MN9").Indices.ToArray()))
            {
                AnsiConsole.MarkupLineInterpolated($"[grey]  {brain[i].Type} {brain.BodyIds[i]}: {during[i] / stimSeconds:F0} Hz during, {(counter.Counts[i] - during[i]) / (settings.Seconds - stimSeconds):F0} Hz after[/]");
            }
        }
        else
        {
            sim.Advance(steps, counter);
        }

        sw.Stop();

        if (sim is CpuBackend cb && CpuBackend.Profile)
        {
            AnsiConsole.MarkupLineInterpolated($"[grey]{cb.ProfileSummary()}[/]");
        }

        int active = counter.Counts.Count(c => c > 0);
        double rtf = settings.Seconds / sw.Elapsed.TotalSeconds;
        AnsiConsole.MarkupLineInterpolated($"[green]Done[/]: {counter.Total:N0} spikes from {active:N0} neurons in {sw.Elapsed.TotalSeconds:F2} s wall ({rtf:F2}× real time, {steps / sw.Elapsed.TotalSeconds:N0} steps/s; {backendName}).");

        int[] order = Enumerable.Range(0, brain.NeuronCount).Where(i => counter.Counts[i] > 0).OrderByDescending(i => counter.Counts[i]).ToArray();
        var table = new Table().Border(TableBorder.Rounded).AddColumns("Body ID", "Type", "Superclass", "Spikes", "Hz", "Stim");
        foreach (int i in order.Take(settings.Top))
        {
            NeuronInfo n = brain[i];
            table.AddRow(n.BodyId.ToString(), n.Type, n.Superclass, counter.Counts[i].ToString("N0"), (counter.Counts[i] / settings.Seconds).ToString("F1"), stim.Contains(i) ? "*" : "");
        }

        AnsiConsole.Write(table);

        if (settings.Csv is not null)
        {
            using var w = new StreamWriter(settings.Csv);
            w.WriteLine("bodyId,index,type,count");
            foreach (int i in order)
            {
                w.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{brain.BodyIds[i]},{i},{brain[i].Type},{counter.Counts[i]}"));
            }

            AnsiConsole.MarkupLineInterpolated($"Wrote {order.Length} rows to {settings.Csv}.");
        }

        return 0;
    }

    /// <summary>Resolves a stimulation/silencing spec: comma-separated body IDs, <c>set:NAME</c> or <c>type:NAME</c>.</summary>
    internal static NeuronSet Resolve(Brain brain, string spec, string name)
    {
        // Several specs joined with ';' form a union.
        if (spec.Contains(';'))
        {
            NeuronSet union = NeuronSet.Empty;
            foreach (string part in spec.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                union |= Resolve(brain, part, name);
            }

            return union.Rename(name);
        }

        if (spec.StartsWith("set:", StringComparison.OrdinalIgnoreCase))
        {
            return brain.Sets[spec[4..]];
        }

        if (spec.StartsWith("type:", StringComparison.OrdinalIgnoreCase) || spec.StartsWith("superclass:", StringComparison.OrdinalIgnoreCase) || spec.StartsWith("class:", StringComparison.OrdinalIgnoreCase))
        {
            // type:NAME[@L|@R|@M] — optional side suffix.
            int colon = spec.IndexOf(':');
            string field = spec[..colon].ToLowerInvariant();
            string value = spec[(colon + 1)..];
            Side? side = null;
            int at = value.LastIndexOf('@');
            if (at >= 0)
            {
                side = value[(at + 1)..].ToUpperInvariant() switch { "L" => Side.Left, "R" => Side.Right, "M" => Side.Middle, _ => throw new ArgumentException($"Unknown side in '{spec}'.") };
                value = value[..at];
            }

            return field switch
            {
                "type" => brain.Query(type: value, side: side, name: name),
                "superclass" => brain.Query(superclass: value, side: side, name: name),
                _ => brain.Query(cls: value, side: side, name: name),
            };
        }

        ulong[] ids = spec.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(s => ulong.Parse(s, CultureInfo.InvariantCulture)).ToArray();
        return brain.ByBodyIds(name, ids);
    }
}
