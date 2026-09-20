using System.ComponentModel;
using DotFly.Core.Graph;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotFly.Cli.Commands;

/// <summary>Prints checkpoint summary, provenance, and optionally one neuron with its partners.</summary>
public sealed class InspectCommand : Command<InspectCommand.Settings>
{
    /// <summary>Command settings.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Checkpoint path.</summary>
        [CommandArgument(0, "<checkpoint>")]
        [Description("Path to a .dfb checkpoint")]
        public required string Checkpoint { get; init; }

        /// <summary>Body ID to describe.</summary>
        [CommandOption("--body <ID>")]
        [Description("Describe one neuron by body/root ID, with its strongest outgoing partners")]
        public ulong? BodyId { get; init; }

        /// <summary>Type query.</summary>
        [CommandOption("--type <TYPE>")]
        [Description("List neurons of a cell type (supports * prefix/suffix)")]
        public string? Type { get; init; }

        /// <summary>Upstream query.</summary>
        [CommandOption("--upstream-of <ID>")]
        [Description("List neurons within --hops forward hops of this body ID (i.e. its presynaptic neighbourhood)")]
        public ulong? UpstreamOf { get; init; }

        /// <summary>Hops.</summary>
        [CommandOption("--hops <N>")]
        [DefaultValue(1)]
        public int Hops { get; init; } = 1;

        /// <summary>Minimum weight.</summary>
        [CommandOption("--min-weight <W>")]
        [DefaultValue(1)]
        public int MinWeight { get; init; } = 1;

        /// <summary>Class filter for listings.</summary>
        [CommandOption("--class <CLASS>")]
        [Description("Only list neurons of this class (with --upstream-of)")]
        public string? Class { get; init; }

        /// <summary>Row limit.</summary>
        [CommandOption("--top <N>")]
        [Description("Rows to print for partner / type listings")]
        [DefaultValue(20)]
        public int Top { get; init; } = 20;
    }

    /// <inheritdoc />
    protected override int Execute(CommandContext context, Settings settings, CancellationToken cancellationToken)
    {
        using Brain brain = Brain.Open(settings.Checkpoint);
        var p = brain.Provenance;

        var summary = new Table().Border(TableBorder.Rounded).AddColumns("Property", "Value");
        summary.AddRow("Name", p.Name);
        summary.AddRow("Dataset", p.Dataset);
        summary.AddRow("Inclusion", p.InclusionFilter);
        summary.AddRow("Sign policy", p.SignPolicy);
        summary.AddRow("Neurons", brain.NeuronCount.ToString("N0"));
        summary.AddRow("Edges", brain.EdgeCount.ToString("N0"));
        summary.AddRow("Contacts", brain.ContactCount.ToString("N0"));
        if (p.SourceEdges is { } se)
        {
            summary.AddRow("Source edges", $"{se.Edges:N0} edges / {se.Contacts:N0} contacts ({se.ZeroSignEdges:N0} / {se.ZeroSignContacts:N0} dropped, sign 0)");
        }

        summary.AddRow("Named sets", brain.Sets.Count.ToString());
        summary.AddRow("Built", $"{p.BuiltAt:u} by {p.BuilderVersion}");
        foreach (var s in p.Sources)
        {
            summary.AddRow("Source", $"{s.FileName} ({s.Length / 1e6:F1} MB, sha256 {s.Sha256[..12]}…)");
        }

        AnsiConsole.Write(summary);

        if (settings.BodyId is { } id)
        {
            int idx = brain.IndexOf(id);
            if (idx < 0)
            {
                AnsiConsole.MarkupLine($"[red]Body ID {id} not found.[/]");
                return 1;
            }

            NeuronInfo n = brain[idx];
            var t = new Table().Border(TableBorder.Rounded).Title($"Neuron {id} (index {idx})").AddColumns("Field", "Value");
            t.AddRow("Type", n.Type);
            t.AddRow("Superclass", n.Superclass);
            t.AddRow("Class", n.Class);
            t.AddRow("Instance", n.Instance);
            t.AddRow("FlyWire type", n.FlywireType);
            t.AddRow("Hemibrain type", n.HemibrainType);
            t.AddRow("Neurotransmitter", n.Neurotransmitter.ToString());
            t.AddRow("Side", n.Side.ToString());
            t.AddRow("Soma", n.Soma is { } s ? $"({s.X}, {s.Y}, {s.Z})" : "-");
            t.AddRow("Out-degree", n.OutDegree.ToString("N0"));
            AnsiConsole.Write(t);

            EdgeSpan edges = brain.OutEdges(idx);
            var order = new int[edges.Length];
            for (int i = 0; i < order.Length; i++)
            {
                order[i] = i;
            }

            short[] w = edges.Weights.ToArray();
            int[] targets = edges.Targets.ToArray();
            Array.Sort(order, (a, b) => Math.Abs((int)w[b]).CompareTo(Math.Abs((int)w[a])));
            var partners = new Table().Border(TableBorder.Rounded).Title("Strongest outgoing partners").AddColumns("Body ID", "Type", "Superclass", "Weight");
            foreach (int i in order.Take(settings.Top))
            {
                NeuronInfo m = brain[targets[i]];
                partners.AddRow(m.BodyId.ToString(), m.Type, m.Superclass, w[i].ToString());
            }

            AnsiConsole.Write(partners);
        }

        if (settings.UpstreamOf is { } upId)
        {
            int target = brain.IndexOf(upId);
            if (target < 0)
            {
                AnsiConsole.MarkupLine($"[red]Body ID {upId} not found.[/]");
                return 1;
            }

            NeuronSet up = brain.Upstream(NeuronSet.From(upId.ToString(), [target]), settings.Hops, settings.MinWeight);
            var rows = up.Indices.ToArray().Select(i => brain[i]).Where(n => settings.Class is null || n.Class == settings.Class).ToArray();
            AnsiConsole.MarkupLineInterpolated($"[bold]{rows.Length}[/] neurons{(settings.Class is null ? "" : $" of class '{settings.Class}'")} within {settings.Hops} hop(s) upstream of {upId} (min weight {settings.MinWeight}); by type:");
            var byType = new Table().Border(TableBorder.Rounded).AddColumns("Type", "Class", "Count", "Sides");
            foreach (var g in rows.GroupBy(n => (n.Type, n.Class)).OrderByDescending(g => g.Count()).Take(settings.Top))
            {
                byType.AddRow(g.Key.Type, g.Key.Class, g.Count().ToString(), string.Join("/", g.GroupBy(n => n.Side).Select(s => $"{s.Key}:{s.Count()}")));
            }

            AnsiConsole.Write(byType);
        }

        if (settings.Type is { } type)
        {
            NeuronSet set = brain.Query(type: type);
            AnsiConsole.MarkupLineInterpolated($"[bold]{set.Count}[/] neurons match type '{type}'.");
            var list = new Table().Border(TableBorder.Rounded).AddColumns("Index", "Body ID", "Type", "Side", "NT", "Out-degree");
            foreach (int i in set.Indices[..Math.Min(set.Count, settings.Top)])
            {
                NeuronInfo m = brain[i];
                list.AddRow(i.ToString(), m.BodyId.ToString(), m.Type, m.Side.ToString(), m.Neurotransmitter.ToString(), m.OutDegree.ToString("N0"));
            }

            AnsiConsole.Write(list);
        }

        return 0;
    }
}
