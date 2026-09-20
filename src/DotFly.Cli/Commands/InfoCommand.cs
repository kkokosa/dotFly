using System.Reflection;
using System.Runtime.InteropServices;
using DotFly.Core.Models;
using DotFly.Cpu;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotFly.Cli.Commands;

/// <summary>Prints runtime, SIMD and default-model information.</summary>
public sealed class InfoCommand : Command
{
    /// <inheritdoc />
    protected override int Execute(CommandContext context, CancellationToken cancellationToken)
    {
        string version = typeof(NeuronModel).Assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "unknown";

        var table = new Table().Border(TableBorder.Rounded).AddColumns("Property", "Value");
        table.AddRow("dotFly", version);
        table.AddRow("Runtime", RuntimeInformation.FrameworkDescription);
        table.AddRow("OS", RuntimeInformation.OSDescription);
        table.AddRow("Architecture", RuntimeInformation.ProcessArchitecture.ToString());
        table.AddRow("Logical cores", Environment.ProcessorCount.ToString());
        table.AddRow("SIMD", SimdInfo.Describe());
        table.AddRow("Server GC", System.Runtime.GCSettings.IsServerGC ? "yes" : "no");

        LifStepConstants k = NeuronModel.Shiu2024.Discretize(0.1);
        table.AddRow("Default model", "Shiu et al. 2024 LIF, dt = 0.1 ms");
        table.AddRow("  a, b, c", $"{k.A:R}, {k.B:R}, {k.C:R}");
        table.AddRow("  refractory / delay", $"{k.RefractorySteps} / {k.DelaySteps} steps");

        AnsiConsole.Write(table);
        return 0;
    }
}
