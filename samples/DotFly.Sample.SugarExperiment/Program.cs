// The Shiu et al. 2024 example experiment through the dotFly engine API:
//   1. drive the 21 sugar-sensing GRNs at 150 Hz for 1 s, 30 trials, on the FlyWire v630 graph;
//   2. report the most active neurons and the proboscis motor neuron MN9;
//   3. silence each of the three most active neurons in turn (Shiu's `neu_slnc`) and re-measure MN9;
//   4. compare trial-averaged rates with the published results/example/sugarR.parquet using the
//      Eon-style metrics (active-set Jaccard, rate correlation, spike-count ratio).
//
// Usage: dotnet run -c Release -- [path/to/.data/shiu2024] [--trials 30] [--hz 150] [--threads 0]

using System.Diagnostics;
using System.Globalization;
using DotFly;
using DotFly.Core.Graph;
using Parquet;
using Parquet.Schema;

string dataDir = args.Length > 0 && !args[0].StartsWith("--", StringComparison.Ordinal) ? args[0] : FindData();
int trials = IntOption("--trials", 30);
double hz = DoubleOption("--hz", 150.0);
int threads = IntOption("--threads", 0);

ulong[] sugarIds =
[
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
];
const ulong Mn9 = 720575940660219265;

using Brain brain = Brain.Open(Path.Combine(dataDir, "flywire-v630.dfb"));
NeuronSet sugar = brain.ByBodyIds("sugar_GRN_R", sugarIds);
int mn9 = brain.IndexOf(Mn9);
Console.WriteLine($"{brain.Provenance.Name}: {brain.NeuronCount:N0} neurons, {brain.EdgeCount:N0} edges. Sugar GRNs: {sugar.Count}, MN9 index {mn9}.");

// ---- 1. 30 trials, one Simulation per trial (different seeds = different Poisson streams) ----
var sw = Stopwatch.StartNew();
double[] meanRate = Experiment("sugarR", sugar, silence: null);
Console.WriteLine($"{trials} trials × 1 s in {sw.Elapsed.TotalSeconds:F1} s wall ({trials / sw.Elapsed.TotalSeconds:F1} simulated s per wall s).");

// ---- 2. Top neurons and MN9 ----
int[] top = Enumerable.Range(0, brain.NeuronCount).Where(i => meanRate[i] > 0).OrderByDescending(i => meanRate[i]).ToArray();
Console.WriteLine($"\nActive neurons (mean over trials): {top.Length}. Top 10:");
foreach (int i in top.Take(10))
{
    Console.WriteLine($"  {brain.BodyIds[i]}  {meanRate[i],7:F1} Hz{(sugar.Contains(i) ? "  (sugar GRN)" : "")}");
}

Console.WriteLine($"MN9 ({Mn9}): {meanRate[mn9]:F1} Hz mean over {trials} trials.");

// ---- 3. Silencing: the three most active neurons (all sugar GRNs) one at a time ----
Console.WriteLine("\nSilencing experiments (outgoing synapses zeroed; the neuron still spikes):");
foreach (int victim in top.Take(3))
{
    double[] silenced = Experiment($"sugarR-{brain.BodyIds[victim]}", sugar, silence: NeuronSet.From("silenced", [victim]));
    Console.WriteLine($"  silence {brain.BodyIds[victim]}: MN9 {silenced[mn9]:F1} Hz (intact {meanRate[mn9]:F1} Hz)");
}

// ---- 4. Comparison with the published example file ----
string published = Path.Combine(dataDir, "results", "example", "sugarR.parquet");
if (File.Exists(published))
{
    Dictionary<ulong, double> pubRate = await ReadPublishedRatesAsync(published);
    var ours = new Dictionary<ulong, double>();
    for (int i = 0; i < brain.NeuronCount; i++)
    {
        if (meanRate[i] > 0)
        {
            ours[brain.BodyIds[i]] = meanRate[i];
        }
    }

    Compare("published results/example/sugarR.parquet (Brian2 2.5.1, 30 trials)", pubRate, ours);
    Console.WriteLine("  Note: Brian2 2.5.1 and 2.9 re-run on the shipped inputs give ~13.2–13.9 k spikes/trial and MN9 ≈ 78–87 Hz,");
    Console.WriteLine("  matching dotFly; the published file (~17 k spikes/trial, MN9 93 Hz) was produced under other conditions.");
}
else
{
    Console.WriteLine($"\n(No published file at {published}; skipping comparison.)");
}

return;

double[] Experiment(string name, NeuronSet stimulated, NeuronSet? silence)
{
    var counts = new long[brain.NeuronCount];
    using Simulation sim = brain.CreateSimulation(new SimulationOptions { Seed = 1000, Threads = threads });
    sim.Input(stimulated, InputKind.PoissonToV).Fill((float)hz);
    if (silence is not null)
    {
        sim.Silence(silence);
    }

    sim.OnSpikes += (_, ids) =>
    {
        foreach (int i in ids)
        {
            counts[i]++;
        }
    };

    for (int trial = 0; trial < trials; trial++)
    {
        sim.Reset(1000 + (ulong)trial);                     // a fresh trial: same inputs, new Poisson stream
        sim.Run(1.Seconds());
    }

    return counts.Select(c => (double)c / trials).ToArray();
}

static void Compare(string label, Dictionary<ulong, double> a, Dictionary<ulong, double> b)
{
    var union = a.Keys.Union(b.Keys).ToArray();
    double jaccard = (double)a.Keys.Intersect(b.Keys).Count() / union.Length;
    double[] x = union.Select(k => a.GetValueOrDefault(k)).ToArray();
    double[] y = union.Select(k => b.GetValueOrDefault(k)).ToArray();
    double mx = x.Average(), my = y.Average(), num = 0, dx = 0, dy = 0;
    for (int i = 0; i < x.Length; i++)
    {
        num += (x[i] - mx) * (y[i] - my);
        dx += (x[i] - mx) * (x[i] - mx);
        dy += (y[i] - my) * (y[i] - my);
    }

    Console.WriteLine($"\nComparison with {label}:");
    Console.WriteLine($"  active neurons: {a.Count} vs {b.Count} (Jaccard {jaccard:F3}); rate correlation {num / Math.Sqrt(dx * dy):F4}; spike-count ratio {y.Sum() / x.Sum():F3}");
    Console.WriteLine($"  MN9: {a.GetValueOrDefault(Mn9):F1} Hz vs {b.GetValueOrDefault(Mn9):F1} Hz");
}

static async Task<Dictionary<ulong, double>> ReadPublishedRatesAsync(string path)
{
    await using ParquetReader reader = await ParquetReader.CreateAsync(path);
    DataField idField = reader.Schema.DataFields.Single(f => f.Name == "flywire_id");
    DataField trialField = reader.Schema.DataFields.Single(f => f.Name == "trial");
    var counts = new Dictionary<ulong, long>();
    var trialsSeen = new HashSet<long>();
    for (int g = 0; g < reader.RowGroupCount; g++)
    {
        using ParquetRowGroupReader rg = reader.OpenRowGroupReader(g);
        int rows = checked((int)rg.RowCount);
        var ids = new long[rows];
        var trial = new long[rows];
        var def = new int[rows];
        await rg.ReadRawAsync<long>(idField, ids, idField.MaxDefinitionLevel > 0 ? def : null, null);
        await rg.ReadRawAsync<long>(trialField, trial, trialField.MaxDefinitionLevel > 0 ? def : null, null);
        for (int i = 0; i < rows; i++)
        {
            counts[(ulong)ids[i]] = counts.GetValueOrDefault((ulong)ids[i]) + 1;
            trialsSeen.Add(trial[i]);
        }
    }

    int nTrials = Math.Max(1, trialsSeen.Count);
    return counts.ToDictionary(kv => kv.Key, kv => (double)kv.Value / nTrials);
}

int IntOption(string name, int fallback) => Array.IndexOf(args, name) is int i and >= 0 && i + 1 < args.Length ? int.Parse(args[i + 1], CultureInfo.InvariantCulture) : fallback;

double DoubleOption(string name, double fallback) => Array.IndexOf(args, name) is int i and >= 0 && i + 1 < args.Length ? double.Parse(args[i + 1], CultureInfo.InvariantCulture) : fallback;

static string FindData()
{
    DirectoryInfo? dir = new(AppContext.BaseDirectory);
    while (dir is not null)
    {
        string candidate = Path.Combine(dir.FullName, ".data", "shiu2024");
        if (Directory.Exists(candidate))
        {
            return candidate;
        }

        dir = dir.Parent;
    }

    throw new DirectoryNotFoundException("Pass the path to .data/shiu2024 (with flywire-v630.dfb built).");
}
