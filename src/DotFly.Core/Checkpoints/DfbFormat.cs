using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;

namespace DotFly.Core.Checkpoints;

/// <summary>
/// Layout constants of the dotFly brain checkpoint (<c>.dfb</c>) file.
/// <para>
/// The file is a little-endian container of fixed-width columns, each 64-byte aligned so that it
/// can be memory-mapped and used in place:
/// </para>
/// <code>
/// [0..64)      header: magic "DFB1", version, section count, header size, reserved
/// [64..)       section table: <see cref="SectionEntry"/> × count
/// ...          sections, each aligned to 64 bytes
/// </code>
/// Section tags are four ASCII characters. Neuron columns have one element per neuron; edge
/// columns have one element per directed edge in CSR order (rows = presynaptic neuron index,
/// columns sorted ascending within a row).
/// </summary>
public static class DfbFormat
{
    /// <summary>File magic, "DFB1".</summary>
    public const uint Magic = 0x31424644; // 'D','F','B','1' little-endian

    /// <summary>Current format version.</summary>
    public const uint Version = 1;

    /// <summary>Section alignment in bytes.</summary>
    public const int Alignment = 64;

    /// <summary>Header size in bytes.</summary>
    public const int HeaderSize = 64;

    /// <summary>Provenance JSON (UTF-8), see <see cref="Provenance"/>.</summary>
    public static readonly uint TagProvenance = Tag("PROV");

    /// <summary>String table: <c>uint32 count</c>, <c>uint32 offsets[count+1]</c>, UTF-8 bytes. Index 0 is always the empty string (used as "null").</summary>
    public static readonly uint TagStrings = Tag("STRS");

    /// <summary>Neuron column, <c>ulong</c>: body/root ID.</summary>
    public static readonly uint TagBodyId = Tag("NBID");

    /// <summary>Neuron column, <c>uint</c>: string index of the cell type.</summary>
    public static readonly uint TagType = Tag("NTYP");

    /// <summary>Neuron column, <c>uint</c>: string index of the superclass.</summary>
    public static readonly uint TagSuperclass = Tag("NSUP");

    /// <summary>Neuron column, <c>uint</c>: string index of the class.</summary>
    public static readonly uint TagClass = Tag("NCLS");

    /// <summary>Neuron column, <c>uint</c>: string index of the instance name.</summary>
    public static readonly uint TagInstance = Tag("NINS");

    /// <summary>Neuron column, <c>uint</c>: string index of the FlyWire type (MaleCNS cross-reference).</summary>
    public static readonly uint TagFlywireType = Tag("NFWT");

    /// <summary>Neuron column, <c>uint</c>: string index of the hemibrain type (MaleCNS cross-reference).</summary>
    public static readonly uint TagHemibrainType = Tag("NHBT");

    /// <summary>Neuron column, <c>byte</c>: <see cref="Models.Neurotransmitter"/>.</summary>
    public static readonly uint TagNeurotransmitter = Tag("NNT_");

    /// <summary>Neuron column, <c>byte</c>: <see cref="Graph.Side"/>.</summary>
    public static readonly uint TagSide = Tag("NSID");

    /// <summary>Neuron column, <c>int × 3</c>: soma voxel coordinates, <see cref="int.MinValue"/> when unknown.</summary>
    public static readonly uint TagSoma = Tag("NSOM");

    /// <summary>Edge row pointers, <c>long[N+1]</c>.</summary>
    public static readonly uint TagRowPtr = Tag("ERPT");

    /// <summary>Edge column indices, <c>int[E]</c>: postsynaptic neuron index.</summary>
    public static readonly uint TagColumns = Tag("ECOL");

    /// <summary>Edge weights, <c>short[E]</c>: signed contact count.</summary>
    public static readonly uint TagWeights = Tag("EWGT");

    /// <summary>Named sets directory: <c>uint32 count</c>, then per set <c>uint32 nameStringIndex, uint32 offset, uint32 length</c> into <see cref="TagSetIndices"/>.</summary>
    public static readonly uint TagSets = Tag("SETS");

    /// <summary>Concatenated, per-set-sorted neuron indices of all named sets, <c>int[]</c>.</summary>
    public static readonly uint TagSetIndices = Tag("SIDX");

    /// <summary>Builds a section tag from four ASCII characters.</summary>
    public static uint Tag(string fourCc)
    {
        if (fourCc.Length != 4)
        {
            throw new ArgumentException("Tag must be exactly four ASCII characters.", nameof(fourCc));
        }

        Span<byte> b = stackalloc byte[4];
        Encoding.ASCII.GetBytes(fourCc, b);
        return BinaryPrimitives.ReadUInt32LittleEndian(b);
    }

    /// <summary>Formats a tag as its four characters, for diagnostics.</summary>
    public static string TagToString(uint tag)
    {
        Span<byte> b = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(b, tag);
        return Encoding.ASCII.GetString(b);
    }

    /// <summary>Rounds a byte offset up to the next multiple of <see cref="Alignment"/>.</summary>
    public static long Align(long offset) => (offset + Alignment - 1) / Alignment * Alignment;
}

/// <summary>An entry of the section table.</summary>
/// <param name="Tag">Four-character section tag.</param>
/// <param name="Offset">Byte offset of the section from the start of the file (64-byte aligned).</param>
/// <param name="Length">Section length in bytes.</param>
/// <param name="Count">Number of logical elements in the section (rows for columns; 0 if not applicable).</param>
[StructLayout(LayoutKind.Sequential, Pack = 8, Size = 32)]
public readonly record struct SectionEntry(uint Tag, long Offset, long Length, long Count);
