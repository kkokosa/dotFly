using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;

namespace DotFly;

/// <summary>A port as recorded.</summary>
/// <param name="Id">Port id.</param>
/// <param name="IsInput">Input (true) or output port.</param>
/// <param name="Name">Set name.</param>
/// <param name="Kind">Kind name.</param>
/// <param name="Neurons">Neuron indices in set order.</param>
public sealed record RecordedPort(int Id, bool IsInput, string Name, string Kind, int[] Neurons);

/// <summary>An input-port write as recorded.</summary>
/// <param name="Step">Step from which the values applied.</param>
/// <param name="Port">Port id.</param>
/// <param name="Values">Values.</param>
public sealed record RecordedInput(long Step, int Port, float[] Values);

/// <summary>A published output frame as recorded.</summary>
/// <param name="Step">Step the frame was published after.</param>
/// <param name="Port">Port id.</param>
/// <param name="FrameIndex">Frame index.</param>
/// <param name="Values">Values.</param>
public sealed record RecordedOutput(long Step, int Port, long FrameIndex, float[] Values);

/// <summary>An injection as recorded.</summary>
/// <param name="Step">Step.</param>
/// <param name="Neuron">Neuron index.</param>
/// <param name="Mv">Millivolts.</param>
public sealed record RecordedInjection(long Step, int Neuron, double Mv);

/// <summary>A parsed <c>.dfs</c> recording.</summary>
public sealed class Recording
{
    /// <summary>Header.</summary>
    public required RecordingHeader Header { get; init; }

    /// <summary>Port definitions.</summary>
    public required IReadOnlyList<RecordedPort> Ports { get; init; }

    /// <summary>All spikes as (step, neuron), in order.</summary>
    public required IReadOnlyList<(long Step, int Neuron)> Spikes { get; init; }

    /// <summary>Input writes.</summary>
    public required IReadOnlyList<RecordedInput> Inputs { get; init; }

    /// <summary>Output frames.</summary>
    public required IReadOnlyList<RecordedOutput> Outputs { get; init; }

    /// <summary>Injections.</summary>
    public required IReadOnlyList<RecordedInjection> Injections { get; init; }

    /// <summary>Steps at which the simulation was reset/restored (the restored step).</summary>
    public required IReadOnlyList<long> Resets { get; init; }

    /// <summary>Reads a <c>.dfs</c> file.</summary>
    public static Recording Read(string path)
    {
        using var r = new BinaryReader(new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 1 << 16), Encoding.UTF8);
        if (r.ReadUInt32() != Recorder.Magic)
        {
            throw new InvalidDataException("Not a .dfs recording.");
        }

        int version = r.ReadInt32();
        if (version != 1)
        {
            throw new InvalidDataException($"Unsupported recording version {version}.");
        }

        RecordingHeader header = System.Text.Json.JsonSerializer.Deserialize(r.ReadString(), RecordingJsonContext.Default.RecordingHeader)
            ?? throw new InvalidDataException("Bad recording header.");

        var ports = new List<RecordedPort>();
        var spikes = new List<(long, int)>();
        var inputs = new List<RecordedInput>();
        var outputs = new List<RecordedOutput>();
        var injections = new List<RecordedInjection>();
        var resets = new List<long>();

        while (r.BaseStream.Position < r.BaseStream.Length)
        {
            var tag = (Recorder.Tag)r.ReadByte();
            switch (tag)
            {
                case Recorder.Tag.Spikes:
                {
                    long step = r.ReadInt64();
                    int n = r.ReadInt32();
                    var ids = new int[n];
                    ReadExactly(r, MemoryMarshal.AsBytes<int>(ids));
                    foreach (int i in ids)
                    {
                        spikes.Add((step, i));
                    }

                    break;
                }

                case Recorder.Tag.Input:
                {
                    long step = r.ReadInt64();
                    int port = r.ReadInt32();
                    int n = r.ReadInt32();
                    var v = new float[n];
                    ReadExactly(r, MemoryMarshal.AsBytes<float>(v));
                    inputs.Add(new RecordedInput(step, port, v));
                    break;
                }

                case Recorder.Tag.Output:
                {
                    long step = r.ReadInt64();
                    int port = r.ReadInt32();
                    long frame = r.ReadInt64();
                    int n = r.ReadInt32();
                    var v = new float[n];
                    ReadExactly(r, MemoryMarshal.AsBytes<float>(v));
                    outputs.Add(new RecordedOutput(step, port, frame, v));
                    break;
                }

                case Recorder.Tag.Injection:
                    injections.Add(new RecordedInjection(r.ReadInt64(), r.ReadInt32(), r.ReadDouble()));
                    break;
                case Recorder.Tag.Port:
                {
                    int id = r.ReadInt32();
                    bool isInput = r.ReadBoolean();
                    string name = r.ReadString();
                    string kind = r.ReadString();
                    int n = r.ReadInt32();
                    var idx = new int[n];
                    ReadExactly(r, MemoryMarshal.AsBytes<int>(idx));
                    ports.Add(new RecordedPort(id, isInput, name, kind, idx));
                    break;
                }

                case Recorder.Tag.Reset:
                    resets.Add(r.ReadInt64());
                    break;
                default:
                    throw new InvalidDataException($"Unknown record tag {tag} at {r.BaseStream.Position}.");
            }
        }

        return new Recording
        {
            Header = header,
            Ports = ports,
            Spikes = spikes,
            Inputs = inputs,
            Outputs = outputs,
            Injections = injections,
            Resets = resets,
        };
    }

    private static void ReadExactly(BinaryReader r, Span<byte> target)
    {
        int read = 0;
        while (read < target.Length)
        {
            int n = r.Read(target[read..]);
            if (n <= 0)
            {
                throw new EndOfStreamException("Truncated recording.");
            }

            read += n;
        }
    }

    /// <summary>
    /// Re-applies the recorded input writes and injections to <paramref name="sim"/> at their
    /// recorded steps (relative to the simulation's current step being the recording's start step).
    /// Ports are matched by id and must have been created in the same order with the same sets.
    /// </summary>
    public void Replay(Simulation sim)
    {
        ArgumentNullException.ThrowIfNull(sim);
        if (sim.Step != Header.StartStep)
        {
            throw new InvalidOperationException($"Recording starts at step {Header.StartStep}; the simulation is at {sim.Step}.");
        }

        foreach (RecordedPort port in Ports)
        {
            if (port.IsInput)
            {
                InputPort live = sim.Inputs[port.Id];
                if (!live.Set.Indices.SequenceEqual(port.Neurons))
                {
                    throw new InvalidOperationException($"Input port {port.Id} ('{port.Name}') does not match the recording.");
                }
            }
        }

        foreach (RecordedInput input in Inputs)
        {
            float[] values = input.Values;
            int id = input.Port;
            sim.ScheduleAt(input.Step, s => s.Inputs[id].Write(values));
        }

        foreach (RecordedInjection inj in Injections)
        {
            sim.Backend.ScheduleInjection(inj.Step, inj.Neuron, inj.Mv);
        }
    }

    /// <summary>Writes the spikes as CSV (<c>step,neuron</c>).</summary>
    public void WriteSpikesCsv(string path)
    {
        using var w = new StreamWriter(path);
        w.WriteLine("step,neuron");
        foreach ((long step, int neuron) in Spikes)
        {
            w.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{step},{neuron}"));
        }
    }

    /// <summary>
    /// Writes the output frames of one port as CSV: one row per frame, columns
    /// <c>frame,step,neural_ms,v0..vK-1</c>. Feature vectors for training readout adapters.
    /// </summary>
    public void WriteFramesCsv(string path, int port)
    {
        using var w = new StreamWriter(path);
        RecordedPort p = Ports.First(x => x.Id == port && !x.IsInput);
        w.WriteLine("frame,step,neural_ms," + string.Join(',', p.Neurons.Select((_, i) => "v" + i)));
        foreach (RecordedOutput o in Outputs.Where(o => o.Port == port))
        {
            w.Write(string.Create(CultureInfo.InvariantCulture, $"{o.FrameIndex},{o.Step},{(o.Step + 1) * Header.DtMs}"));
            foreach (float v in o.Values)
            {
                w.Write(',');
                w.Write(v.ToString("R", CultureInfo.InvariantCulture));
            }

            w.WriteLine();
        }
    }
}
