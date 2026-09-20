using DotFly.Cli.Commands;
using Spectre.Console.Cli;

var app = new CommandApp();
app.Configure(config =>
{
    config.SetApplicationName("dotfly");
    config.AddCommand<InfoCommand>("info")
        .WithDescription("Show runtime, SIMD and hardware information relevant to simulation performance.");
    config.AddBranch("build", build =>
    {
        build.SetDescription("Build a .dfb checkpoint from a connectome release.");
        build.AddCommand<BuildMaleCnsCommand>("malecns")
            .WithDescription("MaleCNS v1.0 flat-connectome Feather files.");
        build.AddCommand<BuildFlyWireCommand>("flywire")
            .WithDescription("FlyWire completeness CSV + connectivity Parquet as shipped with the Shiu et al. 2024 model.");
    });
    config.AddCommand<RunCommand>("run")
        .WithDescription("Run a Shiu-style activation experiment and report firing rates.");
    config.AddCommand<BenchCommand>("bench")
        .WithDescription("Benchmark the CPU backend on a checkpoint at several thread counts.");
    config.AddCommand<ExploreCommand>("explore")
        .WithDescription("Live raster and statistics of a checkpoint running in real time (interactive).");
    config.AddCommand<InspectCommand>("inspect")
        .WithDescription("Inspect a checkpoint: provenance, counts, a neuron and its partners, or a cell type.");
});

return app.Run(args);
