namespace DotFly.Tests.Integration;

/// <summary>
/// Locates the local connectome data. Set <c>DOTFLY_DATA</c> or run from the repository (which has
/// a git-ignored <c>.data/</c> folder). Tests skip when the required files are absent.
/// </summary>
internal static class DataPaths
{
    public static string? Root { get; } = Locate();

    public static string? MaleCns => Dir("malecns-v1.0", "body-annotations-male-cns-v1.0-minconf-0.5.feather");

    public static string? Shiu => Dir("shiu2024", "2023_03_23_connectivity_630_final.parquet");

    public static bool HasMaleCns => MaleCns is not null;

    public static bool HasShiu => Shiu is not null;

    public static bool HasMaleCnsCheckpoint => MaleCns is not null && File.Exists(Path.Combine(MaleCns, "malecns-v1.0-superclass.dfb"));

    public static bool HasFlyWireCheckpoint => Shiu is not null && File.Exists(Path.Combine(Shiu, "flywire-v630.dfb"));

    private static string? Dir(string sub, string marker)
    {
        if (Root is null)
        {
            return null;
        }

        string d = Path.Combine(Root, sub);
        return File.Exists(Path.Combine(d, marker)) ? d : null;
    }

    private static string? Locate()
    {
        string? env = Environment.GetEnvironmentVariable("DOTFLY_DATA");
        if (!string.IsNullOrEmpty(env) && Directory.Exists(env))
        {
            return env;
        }

        DirectoryInfo? dir = new(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string candidate = Path.Combine(dir.FullName, ".data");
            if (Directory.Exists(candidate))
            {
                return candidate;
            }

            dir = dir.Parent;
        }

        return null;
    }
}
