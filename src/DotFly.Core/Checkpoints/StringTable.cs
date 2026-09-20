using System.Buffers.Binary;
using System.Text;

namespace DotFly.Core.Checkpoints;

/// <summary>
/// Interned string table used by checkpoint columns. Index 0 is always the empty string and
/// stands for "not annotated".
/// </summary>
public sealed class StringTable
{
    private readonly List<string> _strings = [string.Empty];
    private readonly Dictionary<string, uint> _index = new(StringComparer.Ordinal) { [string.Empty] = 0 };

    /// <summary>Number of strings, including the empty string at index 0.</summary>
    public int Count => _strings.Count;

    /// <summary>Returns the string at an index.</summary>
    public string this[uint index] => _strings[checked((int)index)];

    /// <summary>Interns a string, returning its index; null and empty map to 0.</summary>
    public uint Intern(string? value)
    {
        if (string.IsNullOrEmpty(value))
        {
            return 0;
        }

        if (_index.TryGetValue(value, out uint existing))
        {
            return existing;
        }

        uint idx = (uint)_strings.Count;
        _strings.Add(value);
        _index[value] = idx;
        return idx;
    }

    /// <summary>Looks up the index of a string without interning; returns null if absent.</summary>
    public uint? TryGetIndex(string value) => _index.TryGetValue(value, out uint i) ? i : null;

    /// <summary>All strings in index order.</summary>
    public IReadOnlyList<string> Strings => _strings;

    /// <summary>
    /// Serialises as <c>uint32 count, uint32 offsets[count+1], utf8 bytes</c>.
    /// </summary>
    public byte[] Serialize()
    {
        int n = _strings.Count;
        var encoded = new byte[n][];
        long total = 0;
        for (int i = 0; i < n; i++)
        {
            encoded[i] = Encoding.UTF8.GetBytes(_strings[i]);
            total += encoded[i].Length;
        }

        checked
        {
            var buffer = new byte[4 + 4 * (n + 1) + (int)total];
            BinaryPrimitives.WriteUInt32LittleEndian(buffer, (uint)n);
            int dataStart = 4 + 4 * (n + 1);
            int pos = dataStart;
            for (int i = 0; i < n; i++)
            {
                BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(4 + 4 * i), (uint)(pos - dataStart));
                encoded[i].CopyTo(buffer, pos);
                pos += encoded[i].Length;
            }

            BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(4 + 4 * n), (uint)(pos - dataStart));
            return buffer;
        }
    }

    /// <summary>Deserialises a table produced by <see cref="Serialize"/>.</summary>
    public static StringTable Deserialize(ReadOnlySpan<byte> data)
    {
        uint n = BinaryPrimitives.ReadUInt32LittleEndian(data);
        int dataStart = checked(4 + 4 * ((int)n + 1));
        var table = new StringTable();
        if (n == 0 || Encoding.UTF8.GetString(Slice(data, 0, dataStart)).Length != 0)
        {
            throw new InvalidDataException("String table must start with the empty string.");
        }

        for (uint i = 1; i < n; i++)
        {
            string s = Encoding.UTF8.GetString(Slice(data, (int)i, dataStart));
            if (s.Length == 0)
            {
                // Preserve indices even if a duplicate empty string sneaked in.
                table._strings.Add(string.Empty);
                continue;
            }

            uint idx = (uint)table._strings.Count;
            table._strings.Add(s);
            table._index.TryAdd(s, idx);
        }

        return table;

        static ReadOnlySpan<byte> Slice(ReadOnlySpan<byte> d, int i, int dataStart)
        {
            uint start = BinaryPrimitives.ReadUInt32LittleEndian(d[(4 + 4 * i)..]);
            uint end = BinaryPrimitives.ReadUInt32LittleEndian(d[(4 + 4 * (i + 1))..]);
            return d.Slice(dataStart + (int)start, (int)(end - start));
        }
    }
}
