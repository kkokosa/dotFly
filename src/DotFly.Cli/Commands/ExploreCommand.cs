using System.ComponentModel;
using System.Text;
using DotFly.Core.Graph;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotFly.Cli.Commands;

/// <summary>
/// Live terminal explorer: runs a checkpoint in real time on a <see cref="RealtimeDriver"/> and
/// shows a spike raster of the most active (or watched) neurons, population statistics and the
/// causal switches. Keys: <c>q</c> quit, <c>space</c> pause/resume, <c>r</c> toggle recurrent
/// transmission, <c>e</c> toggle external input, <c>+</c>/<c>-</c> stimulation rate.
/// </summary>
public sealed class ExploreCommand : Command<ExploreCommand.Settings>
{
    /// <summary>Command settings.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Checkpoint path.</summary>
        [CommandArgument(0, "<checkpoint>")]
        [Description("Path to a .dfb checkpoint")]
        public required string Checkpoint { get; init; }

        /// <summary>Stimulated neurons.</summary>
        [CommandOption("--stimulate <SPEC>")]
        [Description("Neurons to drive with Poisson input: comma-separated body IDs, set:NAME or type:NAME")]
        public string? Stimulate { get; init; }

        /// <summary>Rate.</summary>
        [CommandOption("--hz <RATE>")]
        [DefaultValue(150.0)]
        public double Hz { get; init; } = 150.0;

        /// <summary>Watched neurons.</summary>
        [CommandOption("--watch <SPEC>")]
        [Description("Neurons to show in the raster (default: the most active ones)")]
        public string? Watch { get; init; }

        /// <summary>Speed.</summary>
        [CommandOption("--ratio <R>")]
        [Description("Neural seconds per wall second")]
        [DefaultValue(1.0)]
        public double Ratio { get; init; } = 1.0;

        /// <summary>Threads.</summary>
        [CommandOption("--threads <N>")]
        [DefaultValue(0)]
        public int Threads { get; init; }

        /// <summary>Seed.</summary>
        [CommandOption("--seed <SEED>")]
        [DefaultValue(1UL)]
        public ulong Seed { get; init; } = 1;

        /// <summary>Auto-exit.</summary>
        [CommandOption("--duration <SECONDS>")]
        [Description("Exit after this many wall seconds (default: run until q)")]
        public double? Duration { get; init; }

        /// <summary>Rows.</summary>
        [CommandOption("--rows <N>")]
        [Description("Raster rows")]
        [DefaultValue(24)]
        public int Rows { get; init; } = 24;
    }

    private const int Bins = 240;          // ring capacity; the visible part adapts to the terminal width
    private const double BinMs = 10;
    private const int LabelWidth = 12 + 1 + 18 + 1 + 5 + 1;

    /// <inheritdoc />
    protected override int Execute(CommandContext context, Settings settings, CancellationToken cancellationToken)
    {
        using Brain brain = Brain.Open(settings.Checkpoint);
        NeuronSet stim = settings.Stimulate is null ? NeuronSet.Empty : RunCommand.Resolve(brain, settings.Stimulate, "stimulated");
        NeuronSet? watch = settings.Watch is null ? null : RunCommand.Resolve(brain, settings.Watch, "watched");

        using Simulation sim = brain.CreateSimulation(new SimulationOptions { Seed = settings.Seed, Threads = settings.Threads });
        InputPort? input = stim.Count > 0 ? sim.Input(stim, InputKind.PoissonToV) : null;
        float hz = (float)settings.Hz;
        input?.Fill(hz);

        // Per-neuron activity: total counts over a sliding 2 s window (bucketed) and a raster ring.
        int n = brain.NeuronCount;
        var window = new int[n];
        var raster = new byte[Bins * n];
        int binSteps = Math.Max(1, (int)Math.Round(BinMs / sim.DtMs));
        long lastBin = -1;
        sim.OnSpikes += (step, ids) =>
        {
            long bin = step / binSteps;
            int slot = (int)(bin % Bins);
            if (bin != lastBin)
            {
                Array.Clear(raster, slot * n, n);
                lastBin = bin;
            }

            foreach (int i in ids)
            {
                window[i]++;
                raster[slot * n + i] = 1;
            }
        };

        using var driver = new RealtimeDriver(sim, new RealtimeOptions { Ratio = settings.Ratio });
        driver.Start();

        int[] rows = watch?.Indices[..Math.Min(watch.Count, settings.Rows)].ToArray() ?? [];
        var layout = new Layout("root").SplitRows(new Layout("header").Size(7), new Layout("raster"), new Layout("footer").Size(3));
        bool paused = false;
        long lastDecay = Environment.TickCount64;

        long deadline = settings.Duration is { } d ? Environment.TickCount64 + (long)(d * 1000) : long.MaxValue;
        if (!AnsiConsole.Profile.Capabilities.Interactive || Console.IsOutputRedirected)
        {
            // No live terminal (piped output, CI): print a frame per second until the deadline.
            if (settings.Duration is null)
            {
                AnsiConsole.MarkupLine("[yellow]Not an interactive terminal; pass --duration to run non-interactively.[/]");
                return 1;
            }

            while (Environment.TickCount64 < deadline && driver.Error is null)
            {
                Thread.Sleep(1000);
                if (watch is null)
                {
                    rows = Enumerable.Range(0, n).Where(i => window[i] > 0).OrderByDescending(i => window[i]).Take(settings.Rows).ToArray();
                }

                AnsiConsole.Write(Header(brain, sim, driver, stim, hz, paused));
                AnsiConsole.Write(Raster(brain, rows, raster, n, lastBin, window));
            }

            return driver.Error is null ? 0 : 1;
        }

        AnsiConsole.Live(layout).AutoClear(true).Start(ctx =>
        {
            while (!cancellationToken.IsCancellationRequested && Environment.TickCount64 < deadline)
            {
                while (!Console.IsInputRedirected && Console.KeyAvailable)
                {
                    ConsoleKeyInfo key = Console.ReadKey(true);
                    switch (key.KeyChar)
                    {
                        case 'q':
                            return;
                        case ' ':
                            paused = !paused;
                            if (paused)
                            {
                                driver.Pause();
                            }
                            else
                            {
                                driver.Start();
                            }

                            break;
                        case 'r':
                            sim.RecurrentTransmission = !sim.RecurrentTransmission;
                            break;
                        case 'e':
                            sim.ExternalInput = !sim.ExternalInput;
                            break;
                        case '+':
                            hz = Math.Min(2000f, hz * 1.25f);
                            input?.Fill(hz);
                            break;
                        case '-':
                            hz /= 1.25f;
                            input?.Fill(hz);
                            break;
                    }
                }

                // Decay the activity window every 2 s so the top list follows the present.
                if (Environment.TickCount64 - lastDecay > 2000)
                {
                    for (int i = 0; i < n; i++)
                    {
                        window[i] >>= 1;
                    }

                    lastDecay = Environment.TickCount64;
                }

                if (watch is null)
                {
                    rows = Enumerable.Range(0, n).Where(i => window[i] > 0).OrderByDescending(i => window[i]).Take(settings.Rows).ToArray();
                }

                layout["header"].Update(Header(brain, sim, driver, stim, hz, paused));
                layout["raster"].Update(Raster(brain, rows, raster, n, lastBin, window));
                layout["footer"].Update(new Panel("[grey]q[/] quit  [grey]space[/] pause  [grey]r[/] recurrent on/off  [grey]e[/] external input on/off  [grey]+/-[/] rate").Border(BoxBorder.None));
                ctx.Refresh();
                Thread.Sleep(100);
                if (driver.Error is { } err)
                {
                    AnsiConsole.WriteException(err);
                    return;
                }
            }
        });

        return 0;
    }

    private static Panel Header(Brain brain, Simulation sim, RealtimeDriver driver, NeuronSet stim, float hz, bool paused)
    {
        SimulationClock c = sim.Clock;
        var t = new Table().Border(TableBorder.None).HideHeaders().AddColumns("k", "v", "k2", "v2");
        t.AddRow("checkpoint", $"{brain.Provenance.Name} ({brain.NeuronCount:N0} neurons, {brain.EdgeCount:N0} edges)", "state", paused ? "[yellow]paused[/]" : "[green]running[/]");
        t.AddRow("neural time", $"{c.NeuralTime.TotalSeconds:F2} s", "wall time", $"{c.WallTime.TotalSeconds:F2} s");
        t.AddRow("real-time ×", $"{c.RecentRealTimeFactor:F2} (recent)  {c.RealTimeFactor:F2} (overall)", "behind by", $"{driver.BehindBy.TotalMilliseconds:F0} ms");
        t.AddRow("spikes/s", $"{c.RecentSpikesPerSecond:N0} (neural)", "total spikes", $"{c.Spikes:N0}");
        t.AddRow("stimulation", stim.Count > 0 ? $"{stim.Count} neurons @ {hz:F0} Hz ({stim.Name})" : "none", "switches", $"recurrent {(sim.RecurrentTransmission ? "[green]on[/]" : "[red]off[/]")}, external {(sim.ExternalInput ? "[green]on[/]" : "[red]off[/]")}");
        return new Panel(t).Header("dotfly explore").Border(BoxBorder.Rounded);
    }

    private static Panel Raster(Brain brain, int[] rows, byte[] raster, int n, long lastBin, int[] window)
    {
        var sb = new StringBuilder();
        long newest = lastBin;
        int visible = Math.Clamp(AnsiConsole.Profile.Width - LabelWidth - 4, 20, Bins);
        foreach (int i in rows)
        {
            NeuronInfo info = brain[i];
            string label = $"{Truncate(info.Type, 12),-12} {info.BodyId,18} {window[i],5} ";
            sb.Append(Markup.Escape(label));
            for (int b = visible - 1; b >= 0; b--)
            {
                long bin = newest - b;
                sb.Append(bin >= 0 && raster[(int)(bin % Bins) * n + i] != 0 ? '█' : '·');
            }

            sb.Append('\n');
        }

        if (rows.Length == 0)
        {
            sb.Append("[grey](no activity yet)[/]");
        }

        return new Panel(new Markup(sb.ToString())).Header($"raster — last {visible * BinMs / 1000:F1} s, {BinMs} ms bins (type, body ID, recent spikes)").Border(BoxBorder.Rounded);
    }

    private static string Truncate(string s, int max) => s.Length <= max ? s : s[..max];
}
