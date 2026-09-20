// Train a readout adapter on recorded output frames — the "Fly Dino" pattern:
//   anatomy picks the readout cells (all descending neurons), gameplay-like episodes produce
//   feature frames, and ONLY the adapter is trained. The graph never changes.
//
// Task: three sensory classes on MaleCNS — gustatory, mechanosensory and olfactory neurons — are
// driven one class at a time; from the descending-neuron rates alone, which class is on?
//
//   1. record training episodes (frames = DN rates every 20 ms, label = active class);
//   2. train three adapters on the same frames:
//        a. LinearAdapter — multinomial logistic regression in plain C# (no ML dependency);
//        b. MLNetAdapter  — ML.NET SDCA maximum entropy, saved as a .zip;
//        c. OnnxAdapter   — the ML.NET model exported to ONNX, run through ONNX Runtime;
//   3. evaluate all three on held-out episodes (different seeds) through the live engine.
//
// Usage: dotnet run -c Release -- [path/to/checkpoint.dfb] [--train 8] [--test 4] [--out dir]

using System.Diagnostics;
using System.Globalization;
using DotFly;
using DotFly.Adapters;
using DotFly.Adapters.MLNet;
using DotFly.Adapters.Onnx;
using DotFly.Core.Graph;
using Microsoft.ML;
using Microsoft.ML.Data;

string checkpoint = args.Length > 0 && !args[0].StartsWith("--", StringComparison.Ordinal) ? args[0] : FindCheckpoint();
int trainEpisodes = IntOption("--train", 8);
int testEpisodes = IntOption("--test", 4);
string outDir = StringOption("--out", Path.Combine(Path.GetDirectoryName(checkpoint)!, "readout"));
Directory.CreateDirectory(outDir);

const int NeuronsPerClass = 60;
const float StimulusHz = 100f;
TimeSpan episodeLength = 300.Ms();
TimeSpan settle = 100.Ms();                   // frames before this are transient and not used
TimeSpan publishEvery = 20.Ms();
TimeSpan window = 50.Ms();

using Brain brain = Brain.Open(checkpoint);
var rng = new Random(7);
string[] classNames = ["gustatory", "mechanosensory", "olfactory"];
NeuronSet[] classes = classNames.Select(c => Subset(brain.Query(cls: c), NeuronsPerClass, rng, c)).ToArray();
NeuronSet readout = brain.Query(superclass: "descending_neuron", name: "DN");
Console.WriteLine($"{brain.Provenance.Name}: {brain.NeuronCount:N0} neurons. Readout: {readout.Count} descending neurons. Classes: {string.Join(", ", classes.Select(c => $"{c.Name} ({c.Count})"))}.");

using Simulation sim = brain.CreateSimulation(new SimulationOptions { Seed = 1 });
InputPort[] inputs = classes.Select(c => sim.Input(c, InputKind.PoissonToV)).ToArray();
OutputPort dn = sim.Output(readout, OutputKind.Rate(window), publishEvery);
int k = readout.Count;

// ---- 1. Record episodes -------------------------------------------------------------------
var sw = Stopwatch.StartNew();
List<(float[] Features, int Label)> train = RunEpisodes(seedBase: 100, episodesPerClass: trainEpisodes);
List<(float[] Features, int Label)> test = RunEpisodes(seedBase: 900, episodesPerClass: testEpisodes);
Console.WriteLine($"Recorded {train.Count} training and {test.Count} test frames ({(trainEpisodes + testEpisodes) * classes.Length} episodes × {episodeLength.TotalMilliseconds} ms) in {sw.Elapsed.TotalSeconds:F1} s.");

// ---- 2a. LinearAdapter: multinomial logistic regression in C# -----------------------------
sw.Restart();
float[] scale = FeatureScale(train, k);
LinearAdapter linear = TrainLogistic(train, k, classes.Length, scale, epochs: 300, lr: 0.5f);
linear.Save(Path.Combine(outDir, "dn_linear.json"));
Console.WriteLine($"LinearAdapter trained in {sw.Elapsed.TotalSeconds:F1} s ({k * classes.Length + classes.Length} parameters) → dn_linear.json");

// ---- 2b. ML.NET ------------------------------------------------------------------------------
sw.Restart();
var ml = new MLContext(seed: 1);
SchemaDefinition schema = MLNetAdapter.FeatureSchema(k);
IDataView trainView = ml.Data.LoadFromEnumerable(train.Select(t => new FeatureRow { Features = t.Features, Label = t.Label }), schema);
// The label→key mapping stays outside the model so the exported graph needs only "Features".
IDataView keyed = ml.Transforms.Conversion.MapValueToKey("Label").Fit(trainView).Transform(trainView);
var pipeline = ml.Transforms.NormalizeMinMax("Features")
    .Append(ml.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"));
ITransformer mlModel = pipeline.Fit(keyed);
string mlPath = Path.Combine(outDir, "dn_mlnet.zip");
ml.Model.Save(mlModel, keyed.Schema, mlPath);
var mlnet = new MLNetAdapter(ml, mlModel, k, AdapterOrigin.Trained, $"ML.NET SDCA maximum entropy on {train.Count} frames");
Console.WriteLine($"ML.NET model trained in {sw.Elapsed.TotalSeconds:F1} s → dn_mlnet.zip ({mlnet.ActionCount} class scores)");

// ---- 2c. ONNX export of the ML.NET model -----------------------------------------------------
string onnxPath = Path.Combine(outDir, "dn_mlnet.onnx");
using (FileStream fs = File.Create(onnxPath))
{
    ml.Model.ConvertToOnnx(mlModel, keyed, fs, "Score");
}

using var onnx = new OnnxAdapter(onnxPath, AdapterOrigin.Trained, "ML.NET model exported to ONNX", outputName: OnnxScoreOutput(onnxPath), featureCount: k);
Console.WriteLine($"ONNX export → dn_mlnet.onnx ({onnx.ActionCount} outputs)");

// ---- 3. Evaluate ---------------------------------------------------------------------------
Console.WriteLine();
Console.WriteLine($"{"Adapter",-16} {"Origin",-10} {"Train acc",10} {"Test acc",10}  Description");
Report("Linear", linear, train, test, scale);
Report("ML.NET", mlnet, train, test, null);
Report("ONNX", onnx, train, test, null);
Console.WriteLine("\nChance level is 33.3 %. The graph was never modified; only the readout was trained. The DN cells were chosen by anatomy (superclass), not by the optimiser.");
return;

// ---- helpers ---------------------------------------------------------------------------------

List<(float[], int)> RunEpisodes(ulong seedBase, int episodesPerClass)
{
    var frames = new List<(float[], int)>();
    int settleFrames = (int)(settle.TotalMilliseconds / publishEvery.TotalMilliseconds);
    for (int e = 0; e < episodesPerClass; e++)
    {
        for (int c = 0; c < classes.Length; c++)
        {
            sim.Reset(seedBase + (ulong)(e * classes.Length + c));
            for (int j = 0; j < inputs.Length; j++)
            {
                inputs[j].Fill(j == c ? StimulusHz : 0f);
            }

            int label = c;
            void Collect(OutputPort _, OutputFrame f)
            {
                if (f.FrameIndex >= settleFrames)
                {
                    frames.Add((f.Values.ToArray(), label));
                }
            }

            dn.Published += Collect;
            sim.Run(episodeLength);
            dn.Published -= Collect;
        }
    }

    return frames;
}

static float[] FeatureScale(List<(float[] Features, int Label)> data, int k)
{
    var max = new float[k];
    foreach ((float[] f, _) in data)
    {
        for (int i = 0; i < k; i++)
        {
            max[i] = Math.Max(max[i], f[i]);
        }
    }

    for (int i = 0; i < k; i++)
    {
        max[i] = max[i] > 0 ? 1f / max[i] : 0f;
    }

    return max;
}

static LinearAdapter TrainLogistic(List<(float[] Features, int Label)> data, int k, int classCount, float[] scale, int epochs, float lr)
{
    // Full-batch gradient descent on the multinomial cross-entropy with L2 = 1e-3, on max-scaled
    // features; the scale is folded into the weights so the adapter consumes raw rates.
    var w = new float[classCount * k];
    var b = new float[classCount];
    var gw = new float[classCount * k];
    var gb = new float[classCount];
    var logits = new float[classCount];
    float[][] x = data.Select(d => d.Features.Select((v, i) => v * scale[i]).ToArray()).ToArray();
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        Array.Clear(gw);
        Array.Clear(gb);
        for (int n = 0; n < x.Length; n++)
        {
            float[] xi = x[n];
            for (int c = 0; c < classCount; c++)
            {
                logits[c] = b[c] + System.Numerics.Tensors.TensorPrimitives.Dot(w.AsSpan(c * k, k), xi);
            }

            System.Numerics.Tensors.TensorPrimitives.SoftMax(logits, logits);
            for (int c = 0; c < classCount; c++)
            {
                float g = logits[c] - (c == data[n].Label ? 1f : 0f);
                gb[c] += g;
                System.Numerics.Tensors.TensorPrimitives.MultiplyAdd(xi, g, gw.AsSpan(c * k, k), gw.AsSpan(c * k, k));
            }
        }

        float inv = lr / x.Length;
        for (int i = 0; i < w.Length; i++)
        {
            w[i] -= inv * (gw[i] + 1e-3f * x.Length * w[i]);
        }

        for (int c = 0; c < classCount; c++)
        {
            b[c] -= inv * gb[c];
        }
    }

    for (int c = 0; c < classCount; c++)
    {
        for (int i = 0; i < k; i++)
        {
            w[c * k + i] *= scale[i];
        }
    }

    return new LinearAdapter(k, classCount, w, b, AdapterOrigin.Trained, $"multinomial logistic regression, {data.Count} frames, {epochs} epochs");
}

void Report(string name, IReadoutAdapter adapter, List<(float[] Features, int Label)> trainSet, List<(float[] Features, int Label)> testSet, float[]? unused)
{
    double Accuracy(List<(float[] Features, int Label)> set)
    {
        var actions = new float[adapter.ActionCount];
        int correct = 0;
        foreach ((float[] f, int label) in set)
        {
            adapter.Evaluate(f, actions);
            int best = 0;
            for (int i = 1; i < actions.Length; i++)
            {
                if (actions[i] > actions[best])
                {
                    best = i;
                }
            }

            correct += best == label ? 1 : 0;
        }

        return (double)correct / set.Count;
    }

    Console.WriteLine($"{name,-16} {adapter.Origin,-10} {Accuracy(trainSet),10:P1} {Accuracy(testSet),10:P1}  {adapter.Description}");
}

static string OnnxScoreOutput(string path)
{
    using var session = new Microsoft.ML.OnnxRuntime.InferenceSession(path);
    return session.OutputMetadata.Keys.FirstOrDefault(n => n.StartsWith("Score", StringComparison.OrdinalIgnoreCase))
        ?? session.OutputMetadata.First(kv => kv.Value.ElementType == typeof(float)).Key;
}

static NeuronSet Subset(NeuronSet set, int count, Random rng, string name)
{
    if (set.Count == 0)
    {
        throw new InvalidOperationException($"No neurons for class '{name}' in this checkpoint.");
    }

    int[] idx = set.Indices.ToArray();
    rng.Shuffle(idx);
    return NeuronSet.From(name, idx.AsSpan(0, Math.Min(count, idx.Length)));
}

int IntOption(string name, int fallback) => Array.IndexOf(args, name) is int i and >= 0 && i + 1 < args.Length ? int.Parse(args[i + 1], CultureInfo.InvariantCulture) : fallback;

string StringOption(string name, string fallback) => Array.IndexOf(args, name) is int i and >= 0 && i + 1 < args.Length ? args[i + 1] : fallback;

static string FindCheckpoint()
{
    DirectoryInfo? dir = new(AppContext.BaseDirectory);
    while (dir is not null)
    {
        string candidate = Path.Combine(dir.FullName, ".data", "malecns-v1.0", "malecns-v1.0-superclass.dfb");
        if (File.Exists(candidate))
        {
            return candidate;
        }

        dir = dir.Parent;
    }

    throw new FileNotFoundException("Pass the path to a MaleCNS .dfb checkpoint.");
}
