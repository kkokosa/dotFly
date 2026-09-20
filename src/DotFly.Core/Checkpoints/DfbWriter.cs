using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;
using DotFly.Core.Graph;

namespace DotFly.Core.Checkpoints;

/// <summary>Writes a <see cref="CheckpointContent"/> to a <c>.dfb</c> file.</summary>
public static class DfbWriter
{
    private readonly record struct Pending(uint Tag, long Count, Func<Stream, long> Write, long Length);

    /// <summary>Writes the checkpoint to <paramref name="path"/>, replacing any existing file.</summary>
    public static void Write(string path, CheckpointContent content)
    {
        ArgumentNullException.ThrowIfNull(content);
        content.Validate();

        byte[] provenance = Encoding.UTF8.GetBytes(content.Provenance.ToJson());
        byte[] strings = content.Strings.Serialize();
        (byte[] setsDir, int[] setIdx) = SerializeSets(content.Sets);
        var soma = content.Soma;

        var sections = new List<Pending>
        {
            Bytes(DfbFormat.TagProvenance, provenance, 1),
            Bytes(DfbFormat.TagStrings, strings, content.Strings.Count),
            Column(DfbFormat.TagBodyId, content.BodyIds),
            Column(DfbFormat.TagType, content.TypeIndex),
            Column(DfbFormat.TagSuperclass, content.SuperclassIndex),
            Column(DfbFormat.TagClass, content.ClassIndex),
            Column(DfbFormat.TagInstance, content.InstanceIndex),
            Column(DfbFormat.TagFlywireType, content.FlywireTypeIndex),
            Column(DfbFormat.TagHemibrainType, content.HemibrainTypeIndex),
            Column(DfbFormat.TagNeurotransmitter, MemoryMarshal.Cast<Models.Neurotransmitter, byte>(content.Neurotransmitters).ToArray()),
            Column(DfbFormat.TagSide, MemoryMarshal.Cast<Side, byte>(content.Sides).ToArray()),
            Column(DfbFormat.TagSoma, soma, soma.Length / 3),
            Column(DfbFormat.TagRowPtr, content.RowPtr),
            Column(DfbFormat.TagColumns, content.Columns),
            Column(DfbFormat.TagWeights, content.Weights),
            Bytes(DfbFormat.TagSets, setsDir, content.Sets.Count),
            Column(DfbFormat.TagSetIndices, setIdx),
        };

        // Layout: header, table, then sections at aligned offsets.
        long tableOffset = DfbFormat.HeaderSize;
        long cursor = DfbFormat.Align(tableOffset + sections.Count * 32L);
        var entries = new SectionEntry[sections.Count];
        for (int i = 0; i < sections.Count; i++)
        {
            entries[i] = new SectionEntry(sections[i].Tag, cursor, sections[i].Length, sections[i].Count);
            cursor = DfbFormat.Align(cursor + sections[i].Length);
        }

        string tmp = path + ".tmp";
        using (var fs = new FileStream(tmp, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20))
        {
            Span<byte> header = stackalloc byte[DfbFormat.HeaderSize];
            header.Clear();
            BinaryPrimitives.WriteUInt32LittleEndian(header, DfbFormat.Magic);
            BinaryPrimitives.WriteUInt32LittleEndian(header[4..], DfbFormat.Version);
            BinaryPrimitives.WriteUInt32LittleEndian(header[8..], (uint)sections.Count);
            BinaryPrimitives.WriteUInt32LittleEndian(header[12..], DfbFormat.HeaderSize);
            BinaryPrimitives.WriteInt64LittleEndian(header[16..], cursor); // total file length
            fs.Write(header);

            Span<byte> entry = stackalloc byte[32];
            foreach (SectionEntry e in entries)
            {
                BinaryPrimitives.WriteUInt32LittleEndian(entry, e.Tag);
                BinaryPrimitives.WriteUInt32LittleEndian(entry[4..], 0);
                BinaryPrimitives.WriteInt64LittleEndian(entry[8..], e.Offset);
                BinaryPrimitives.WriteInt64LittleEndian(entry[16..], e.Length);
                BinaryPrimitives.WriteInt64LittleEndian(entry[24..], e.Count);
                fs.Write(entry);
            }

            for (int i = 0; i < sections.Count; i++)
            {
                PadTo(fs, entries[i].Offset);
                long written = sections[i].Write(fs);
                if (written != sections[i].Length)
                {
                    throw new InvalidOperationException($"Section {DfbFormat.TagToString(sections[i].Tag)} wrote {written} bytes, expected {sections[i].Length}.");
                }
            }

            PadTo(fs, cursor);
        }

        File.Move(tmp, path, overwrite: true);
    }

    private static Pending Bytes(uint tag, byte[] data, long count) =>
        new(tag, count, s => { s.Write(data); return data.Length; }, data.Length);

    private static Pending Column<T>(uint tag, T[] data, long? count = null)
        where T : unmanaged
    {
        long bytes = data.LongLength * Marshal.SizeOf<T>();
        return new Pending(tag, count ?? data.LongLength, s =>
        {
            ReadOnlySpan<byte> raw = MemoryMarshal.AsBytes<T>(data);
            // Write in chunks so very large columns do not exceed Stream.Write span limits.
            const int chunk = 1 << 24;
            long total = 0;
            for (int off = 0; off < raw.Length; off += chunk)
            {
                int len = Math.Min(chunk, raw.Length - off);
                s.Write(raw.Slice(off, len));
                total += len;
            }

            return total;
        }, bytes);
    }

    private static (byte[] Directory, int[] Indices) SerializeSets(IReadOnlyList<NeuronSet> sets)
    {
        var strings = new StringTable();
        var dir = new byte[4 + 12 * sets.Count];
        BinaryPrimitives.WriteUInt32LittleEndian(dir, (uint)sets.Count);
        long total = 0;
        foreach (NeuronSet s in sets)
        {
            total += s.Count;
        }

        var idx = new int[checked((int)total)];
        int pos = 0;
        for (int i = 0; i < sets.Count; i++)
        {
            NeuronSet s = sets[i];
            // Names are stored inline as UTF-8 length-prefixed after the fixed part? No: keep it
            // simple and fixed-width — the name goes into a per-directory mini string table.
            uint nameIdx = strings.Intern(s.Name);
            BinaryPrimitives.WriteUInt32LittleEndian(dir.AsSpan(4 + 12 * i), nameIdx);
            BinaryPrimitives.WriteUInt32LittleEndian(dir.AsSpan(8 + 12 * i), (uint)pos);
            BinaryPrimitives.WriteUInt32LittleEndian(dir.AsSpan(12 + 12 * i), (uint)s.Count);
            s.Indices.CopyTo(idx.AsSpan(pos));
            pos += s.Count;
        }

        byte[] names = strings.Serialize();
        var combined = new byte[dir.Length + names.Length];
        dir.CopyTo(combined, 0);
        names.CopyTo(combined, dir.Length);
        return (combined, idx);
    }

    private static void PadTo(FileStream fs, long offset)
    {
        long pad = offset - fs.Position;
        if (pad < 0)
        {
            throw new InvalidOperationException("Section layout overflow.");
        }

        if (pad > 0)
        {
            Span<byte> zeros = stackalloc byte[64];
            zeros.Clear();
            while (pad > 0)
            {
                int n = (int)Math.Min(pad, zeros.Length);
                fs.Write(zeros[..n]);
                pad -= n;
            }
        }
    }
}
