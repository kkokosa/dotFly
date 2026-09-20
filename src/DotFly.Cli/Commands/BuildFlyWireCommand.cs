using System.ComponentModel;
using System.Diagnostics;
using DotFly.Core.Checkpoints;
using DotFly.Data.Sources;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotFly.Cli.Commands;

/// <summary>Builds a <c>.dfb</c> checkpoint from the FlyWire inputs of the Shiu et al. 2024 model.</summary>
public sealed class BuildFlyWireCommand : AsyncCommand<BuildFlyWireCommand.Settings>
{
    /// <summary>Command settings.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Completeness CSV.</summary>
        [CommandArgument(0, "<completeness-csv>")]
        [Description("Completeness CSV (e.g. 2023_03_23_completeness_630_final.csv)")]
        public required string Completeness { get; init; }

        /// <summary>Connectivity Parquet.</summary>
        [CommandArgument(1, "<connectivity-parquet>")]
        [Description("Connectivity Parquet (e.g. 2023_03_23_connectivity_630_final.parquet)")]
        public required string Connectivity { get; init; }

        /// <summary>Materialisation label.</summary>
        [CommandOption("-m|--materialization <LABEL>")]
        [Description("Materialisation label, e.g. 630 or 783")]
        [DefaultValue("630")]
        public string Materialization { get; init; } = "630";

        /// <summary>Output path.</summary>
        [CommandOption("-o|--output <FILE>")]
        [Description("Output .dfb path (default: next to the CSV as flywire-v<label>.dfb)")]
        public string? Output { get; init; }
    }

    /// <inheritdoc />
    protected override async Task<int> ExecuteAsync(CommandContext context, Settings settings, CancellationToken cancellationToken)
    {
        string name = $"flywire-v{settings.Materialization}";
        string output = settings.Output ?? Path.Combine(Path.GetDirectoryName(Path.GetFullPath(settings.Completeness))!, name + ".dfb");
        var sw = Stopwatch.StartNew();
        var source = new FlyWireSource
        {
            CompletenessPath = settings.Completeness,
            ConnectivityPath = settings.Connectivity,
            Materialization = settings.Materialization,
            Log = m => AnsiConsole.MarkupLineInterpolated($"[grey]{sw.Elapsed:mm\\:ss}[/] {m}"),
        };

        CheckpointContent content = await source.ReadAsync(name, cancellationToken).ConfigureAwait(false);
        AnsiConsole.MarkupLineInterpolated($"[grey]{sw.Elapsed:mm\\:ss}[/] Writing {output}…");
        DfbWriter.Write(output, content);
        AnsiConsole.MarkupLineInterpolated($"[green]Done[/] in {sw.Elapsed.TotalSeconds:F1} s: {content.NeuronCount:N0} neurons, {content.EdgeCount:N0} edges → {new FileInfo(output).Length / 1e6:F1} MB");
        return 0;
    }
}
