using System.Text;
using DotFly.Core.Graph;

namespace DotFly;

/// <summary>What a recording captures.</summary>
[Flags]
public enum RecordFlags
{
    /// <summary>Every spike (step, neuron).</summary>
    Spikes = 1,

    /// <summary>Every input-port write and injection.</summary>
    Inputs = 2,

    /// <summary>Every published output frame.</summary>
    Outputs = 4,

    /// <summary>Everything.</summary>
    All = Spikes | Inputs | Outputs,
}

/// <summary>
/// Writes a <c>.dfs</c> recording: a little-endian stream of tagged records after a JSON header
/// that pins the checkpoint, seed and options. Spikes, inputs and output frames carry the step
/// they belong to, so a run can be replayed (<see cref="Recording.Replay"/>) or exported.
/// </summary>
public sealed class Recorder : IDisposable
{
    internal const uint Magic = 0x31534644; // "DFS1"

    internal enum Tag : byte
    {
        Spikes = 1,
        Input = 2,
        Output = 3,
        Injection = 4,
        Port = 5,
        Reset = 6,
    }

    private readonly BinaryWriter _w;
    private readonly Simulation _sim;
    private bool _closed;

    /// <summary>What is being recorded.</summary>
    public RecordFlags Flags { get; }

    /// <summary>Output path.</summary>
    public string Path { get; }

    internal Recorder(string path, Simulation sim, RecordFlags flags)
    {
        Path = path;
        _sim = sim;
        Flags = flags;
        _w = new BinaryWriter(new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read, 1 << 16), Encoding.UTF8);
        _w.Write(Magic);
        _w.Write(1);
        string header = System.Text.Json.JsonSerializer.Serialize(new RecordingHeader(
            sim.Brain.Provenance.Name,
            sim.Brain.Provenance.Sources.Select(s => s.Sha256).ToArray(),
            sim.Options.Seed,
            sim.Options.DtMs,
            sim.Options.Backend.ToString(),
            sim.Step,
            DateTimeOffset.UtcNow), RecordingJsonContext.Default.RecordingHeader);
        _w.Write(header);
    }

    internal void WritePortDefinition(int id, bool isInput, NeuronSet set, string kind)
    {
        _w.Write((byte)Tag.Port);
        _w.Write(id);
        _w.Write(isInput);
        _w.Write(set.Name);
        _w.Write(kind);
        _w.Write(set.Count);
        foreach (int i in set.Indices)
        {
            _w.Write(i);
        }
    }

    internal void WriteSpikes(long step, ReadOnlySpan<int> spikes)
    {
        if ((Flags & RecordFlags.Spikes) == 0 || spikes.IsEmpty)
        {
            return;
        }

        _w.Write((byte)Tag.Spikes);
        _w.Write(step);
        _w.Write(spikes.Length);
        _w.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(spikes));
    }

    internal void WriteInput(long step, int port, ReadOnlySpan<float> values)
    {
        if ((Flags & RecordFlags.Inputs) == 0)
        {
            return;
        }

        _w.Write((byte)Tag.Input);
        _w.Write(step);
        _w.Write(port);
        _w.Write(values.Length);
        _w.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(values));
    }

    internal void WriteInjection(long step, int neuron, double mv)
    {
        if ((Flags & RecordFlags.Inputs) == 0)
        {
            return;
        }

        _w.Write((byte)Tag.Injection);
        _w.Write(step);
        _w.Write(neuron);
        _w.Write(mv);
    }

    internal void WriteOutput(long step, int port, OutputFrame frame)
    {
        if ((Flags & RecordFlags.Outputs) == 0)
        {
            return;
        }

        _w.Write((byte)Tag.Output);
        _w.Write(step);
        _w.Write(port);
        _w.Write(frame.FrameIndex);
        _w.Write(frame.Values.Length);
        _w.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(frame.Values));
    }

    internal void WriteReset(long step)
    {
        _w.Write((byte)Tag.Reset);
        _w.Write(step);
    }

    /// <summary>Flushes buffered records to disk.</summary>
    public void Flush() => _w.Flush();

    /// <inheritdoc />
    public void Dispose()
    {
        if (_closed)
        {
            return;
        }

        _closed = true;
        _w.Dispose();
        _sim.RecorderClosed(this);
    }
}

/// <summary>Header of a <c>.dfs</c> recording.</summary>
/// <param name="Checkpoint">Checkpoint name.</param>
/// <param name="SourceSha256">SHA-256 of the checkpoint's source files.</param>
/// <param name="Seed">Simulation seed.</param>
/// <param name="DtMs">Timestep in ms.</param>
/// <param name="Backend">Backend name.</param>
/// <param name="StartStep">Step at which recording started.</param>
/// <param name="RecordedAt">UTC time.</param>
public sealed record RecordingHeader(string Checkpoint, string[] SourceSha256, ulong Seed, double DtMs, string Backend, long StartStep, DateTimeOffset RecordedAt);

[System.Text.Json.Serialization.JsonSerializable(typeof(RecordingHeader))]
internal sealed partial class RecordingJsonContext : System.Text.Json.Serialization.JsonSerializerContext
{
}
