using System.ComponentModel;
using System.Diagnostics;
using DotFly.Core.Checkpoints;
using DotFly.Data.Sources;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotFly.Cli.Commands;

/// <summary>Builds a <c>.dfb</c> checkpoint from the MaleCNS flat-connectome Feather release.</summary>
public sealed class BuildMaleCnsCommand : Command<BuildMaleCnsCommand.Settings>
{
    /// <summary>Command settings.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Directory containing the three Feather files.</summary>
        [CommandArgument(0, "<data-dir>")]
        [Description("Directory with body-annotations-*.feather, body-neurotransmitters-*.feather and connectome-weights-*.feather")]
        public required string DataDir { get; init; }

        /// <summary>Output path.</summary>
        [CommandOption("-o|--output <FILE>")]
        [Description("Output .dfb path (default: <data-dir>/malecns-v1.0-<filter>.dfb)")]
        public string? Output { get; init; }

        /// <summary>Inclusion filter.</summary>
        [CommandOption("--filter <FILTER>")]
        [Description("superclass (default; 166,700 neurons) or traced (status == Traced; 165,122 neurons)")]
        [DefaultValue("superclass")]
        public string Filter { get; init; } = "superclass";

        /// <summary>Weights file override.</summary>
        [CommandOption("--weights <FILE>")]
        [Description("Weights file name inside data-dir (default: the full minconf-0.5 file, or the traced-only file for --filter traced when present)")]
        public string? Weights { get; init; }
    }

    /// <inheritdoc />
    protected override int Execute(CommandContext context, Settings settings, CancellationToken cancellationToken)
    {
        MaleCnsInclusion inclusion = settings.Filter.ToLowerInvariant() switch
        {
            "superclass" => MaleCnsInclusion.SuperclassAnnotated,
            "traced" => MaleCnsInclusion.TracedOnly,
            _ => throw new ArgumentException($"Unknown filter '{settings.Filter}'."),
        };

        string dir = settings.DataDir;
        string annotations = Find(dir, "body-annotations-*.feather");
        string nts = Find(dir, "body-neurotransmitters-*.feather");
        string weights;
        if (settings.Weights is not null)
        {
            weights = Path.Combine(dir, settings.Weights);
        }
        else
        {
            string? traced = inclusion == MaleCnsInclusion.TracedOnly ? Directory.GetFiles(dir, "connectome-weights-*-traced-only.feather").FirstOrDefault() : null;
            weights = traced ?? Directory.GetFiles(dir, "connectome-weights-*.feather")
                .FirstOrDefault(f => !f.Contains("-traced-only") && !f.Contains("-significant-only"))
                ?? throw new FileNotFoundException("No connectome-weights-*.feather in data dir.");
        }

        string name = $"malecns-v1.0-{(inclusion == MaleCnsInclusion.TracedOnly ? "traced" : "superclass")}";
        string output = settings.Output ?? Path.Combine(dir, name + ".dfb");

        var sw = Stopwatch.StartNew();
        var source = new MaleCnsSource
        {
            AnnotationsPath = annotations,
            NeurotransmittersPath = nts,
            WeightsPath = weights,
            Inclusion = inclusion,
            Log = m => AnsiConsole.MarkupLineInterpolated($"[grey]{sw.Elapsed:mm\\:ss}[/] {m}"),
        };

        CheckpointContent content = source.Read(name);
        AnsiConsole.MarkupLineInterpolated($"[grey]{sw.Elapsed:mm\\:ss}[/] Writing {output}…");
        DfbWriter.Write(output, content);

        long contacts = 0;
        foreach (short w in content.Weights)
        {
            contacts += Math.Abs((int)w);
        }

        AnsiConsole.MarkupLineInterpolated($"[green]Done[/] in {sw.Elapsed.TotalSeconds:F1} s: {content.NeuronCount:N0} neurons, {content.EdgeCount:N0} edges, {contacts:N0} contacts → {new FileInfo(output).Length / 1e6:F1} MB");
        return 0;
    }

    private static string Find(string dir, string pattern) =>
        Directory.GetFiles(dir, pattern).FirstOrDefault() ?? throw new FileNotFoundException($"No {pattern} in {dir}.");
}
