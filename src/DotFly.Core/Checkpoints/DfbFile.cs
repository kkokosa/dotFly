using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;

namespace DotFly.Core.Checkpoints;

/// <summary>
/// A memory-mapped <c>.dfb</c> file. Sections are exposed as spans over the mapping — nothing is
/// copied to the managed heap. The file must stay open while any span obtained from it is in use.
/// </summary>
public sealed unsafe class DfbFile : IDisposable
{
    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _view;
    private readonly byte* _base;
    private readonly Dictionary<uint, SectionEntry> _sections = new();

    /// <summary>Path the file was opened from.</summary>
    public string Path { get; }

    /// <summary>Total file length in bytes as recorded in the header.</summary>
    public long Length { get; }

    /// <summary>All section entries.</summary>
    public IReadOnlyCollection<SectionEntry> Sections => _sections.Values;

    private DfbFile(string path, MemoryMappedFile mmf, MemoryMappedViewAccessor view, byte* basePtr, long length)
    {
        Path = path;
        _mmf = mmf;
        _view = view;
        _base = basePtr;
        Length = length;
    }

    /// <summary>Opens and validates a checkpoint file.</summary>
    /// <exception cref="InvalidDataException">Bad magic, version or section table.</exception>
    public static DfbFile Open(string path)
    {
        var info = new FileInfo(path);
        if (!info.Exists)
        {
            throw new FileNotFoundException("Checkpoint not found.", path);
        }

        if (info.Length < DfbFormat.HeaderSize)
        {
            throw new InvalidDataException("File too small to be a .dfb checkpoint.");
        }

        MemoryMappedFile mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        MemoryMappedViewAccessor view = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        byte* ptr = null;
        view.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
        ptr += view.PointerOffset;

        var header = new ReadOnlySpan<byte>(ptr, DfbFormat.HeaderSize);
        if (BinaryPrimitives.ReadUInt32LittleEndian(header) != DfbFormat.Magic)
        {
            Release(view, mmf);
            throw new InvalidDataException("Not a .dfb checkpoint (bad magic).");
        }

        uint version = BinaryPrimitives.ReadUInt32LittleEndian(header[4..]);
        if (version != DfbFormat.Version)
        {
            Release(view, mmf);
            throw new InvalidDataException($"Unsupported .dfb version {version}; expected {DfbFormat.Version}.");
        }

        uint count = BinaryPrimitives.ReadUInt32LittleEndian(header[8..]);
        uint headerSize = BinaryPrimitives.ReadUInt32LittleEndian(header[12..]);
        long length = BinaryPrimitives.ReadInt64LittleEndian(header[16..]);
        if (length > info.Length)
        {
            Release(view, mmf);
            throw new InvalidDataException("Checkpoint is truncated.");
        }

        var file = new DfbFile(path, mmf, view, ptr, length);
        for (uint i = 0; i < count; i++)
        {
            var e = new ReadOnlySpan<byte>(ptr + headerSize + 32 * i, 32);
            var entry = new SectionEntry(
                BinaryPrimitives.ReadUInt32LittleEndian(e),
                BinaryPrimitives.ReadInt64LittleEndian(e[8..]),
                BinaryPrimitives.ReadInt64LittleEndian(e[16..]),
                BinaryPrimitives.ReadInt64LittleEndian(e[24..]));
            if (entry.Offset < 0 || entry.Length < 0 || entry.Offset + entry.Length > length)
            {
                file.Dispose();
                throw new InvalidDataException($"Section {DfbFormat.TagToString(entry.Tag)} is out of bounds.");
            }

            file._sections[entry.Tag] = entry;
        }

        return file;
    }

    /// <summary>Whether the file has a section with the given tag.</summary>
    public bool Has(uint tag) => _sections.ContainsKey(tag);

    /// <summary>Returns the section entry, throwing if absent.</summary>
    public SectionEntry Section(uint tag) =>
        _sections.TryGetValue(tag, out SectionEntry e)
            ? e
            : throw new InvalidDataException($"Checkpoint has no section {DfbFormat.TagToString(tag)}.");

    /// <summary>Raw bytes of a section.</summary>
    public ReadOnlySpan<byte> Bytes(uint tag)
    {
        SectionEntry e = Section(tag);
        return new ReadOnlySpan<byte>(_base + e.Offset, checked((int)e.Length));
    }

    /// <summary>A fixed-width column as a typed span over the mapping.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> Column<T>(uint tag)
        where T : unmanaged
    {
        SectionEntry e = Section(tag);
        long elems = e.Length / sizeof(T);
        return new ReadOnlySpan<T>(_base + e.Offset, checked((int)elems));
    }

    /// <summary>Raw pointer to a section start (for kernels that outlive a span scope).</summary>
    public T* Pointer<T>(uint tag)
        where T : unmanaged
        => (T*)(_base + Section(tag).Offset);

    /// <inheritdoc />
    public void Dispose()
    {
        Release(_view, _mmf);
        GC.SuppressFinalize(this);
    }

    private static void Release(MemoryMappedViewAccessor view, MemoryMappedFile mmf)
    {
        view.SafeMemoryMappedViewHandle.ReleasePointer();
        view.Dispose();
        mmf.Dispose();
    }
}
