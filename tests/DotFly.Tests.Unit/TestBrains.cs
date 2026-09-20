using DotFly.Core.Checkpoints;
using DotFly.Core.Graph;
using DotFly.Core.Models;
using DotFly.Data.Build;

namespace DotFly.Tests.Unit;

/// <summary>Builds small synthetic checkpoints in temp files for tests.</summary>
internal static class TestBrains
{
    public static Provenance Provenance(string name = "test") => new()
    {
        Name = name,
        Dataset = "synthetic",
        Sources = [],
        InclusionFilter = "all",
        SignPolicy = "precomputed",
        Model = NeuronModel.Shiu2024,
        BuilderVersion = "test",
        BuiltAt = new DateTimeOffset(2026, 9, 18, 0, 0, 0, TimeSpan.Zero),
    };

    /// <summary>
    /// Writes a checkpoint with <paramref name="neuronCount"/> neurons (body IDs 1000+i) and the
    /// given (pre, post, weight) edges, then opens it. The caller disposes the brain; the file is
    /// deleted on dispose via <see cref="TempBrain"/>.
    /// </summary>
    public static TempBrain Create(int neuronCount, IEnumerable<(int Pre, int Post, int Weight)> edges, IReadOnlyList<NeuronSet>? sets = null, Func<int, NeuronRecord>? neuron = null)
    {
        var neurons = new List<NeuronRecord>(neuronCount);
        for (int i = 0; i < neuronCount; i++)
        {
            neurons.Add(neuron?.Invoke(i) ?? new NeuronRecord { BodyId = 1000UL + (ulong)i, Type = $"T{i % 3}" });
        }

        var list = new EdgeList(16);
        foreach ((int pre, int post, int w) in edges)
        {
            list.Add(pre, post, w);
        }

        CheckpointContent content = CheckpointBuilder.Build(Provenance(), neurons, list, sets ?? []);
        string path = Path.Combine(Path.GetTempPath(), $"dotfly-test-{Guid.NewGuid():N}.dfb");
        DfbWriter.Write(path, content);
        return new TempBrain(Brain.Open(path), path);
    }

    public sealed class TempBrain(Brain brain, string path) : IDisposable
    {
        public Brain Brain { get; } = brain;

        public string Path { get; } = path;

        public void Dispose()
        {
            Brain.Dispose();
            try
            {
                File.Delete(Path);
            }
            catch (IOException)
            {
                // Best effort.
            }
        }
    }
}
