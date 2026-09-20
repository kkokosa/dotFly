using System.Security.Cryptography;
using DotFly.Core.Checkpoints;

namespace DotFly.Data.Sources;

/// <summary>Helpers shared by the source readers.</summary>
internal static class SourceFiles
{
    /// <summary>Computes a <see cref="SourceFile"/> record (SHA-256) for a file.</summary>
    public static SourceFile Describe(string path)
    {
        using FileStream fs = File.OpenRead(path);
        byte[] hash = SHA256.HashData(fs);
        return new SourceFile(Path.GetFileName(path), Convert.ToHexStringLower(hash), fs.Length);
    }

    /// <summary>The version string of the running DotFly.Data assembly.</summary>
    public static string BuilderVersion =>
        typeof(SourceFiles).Assembly.GetCustomAttributes(typeof(System.Reflection.AssemblyInformationalVersionAttribute), false)
            is [System.Reflection.AssemblyInformationalVersionAttribute a] ? a.InformationalVersion : "unknown";
}
