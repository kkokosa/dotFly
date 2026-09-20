using DotFly.Adapters;
using DotFly.Adapters.MLNet;
using DotFly.Adapters.Onnx;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace DotFly.Tests.Unit.Engine;

/// <summary>
/// The two ML-backed adapters over a tiny synthetic task: train with ML.NET, run through
/// <see cref="MLNetAdapter"/>, export to ONNX, run through <see cref="OnnxAdapter"/>, and require
/// identical class decisions from both.
/// </summary>
public sealed class AdapterPackageTests
{
    private const int K = 6;

    private static List<FeatureRow> Synthetic(int perClass, int seed)
    {
        var rng = new Random(seed);
        var rows = new List<FeatureRow>();
        for (int c = 0; c < 3; c++)
        {
            for (int n = 0; n < perClass; n++)
            {
                var f = new float[K];
                for (int i = 0; i < K; i++)
                {
                    f[i] = (float)rng.NextDouble() * 10f + (i % 3 == c ? 40f : 0f);   // class c lights features i ≡ c (mod 3)
                }

                rows.Add(new FeatureRow { Features = f, Label = c });
            }
        }

        return rows;
    }

    [Fact]
    public void MLNet_And_Onnx_Adapters_Agree_On_A_Trained_Classifier()
    {
        var ml = new MLContext(seed: 1);
        SchemaDefinition schema = MLNetAdapter.FeatureSchema(K);
        IDataView data = ml.Data.LoadFromEnumerable(Synthetic(40, 1), schema);
        IDataView keyed = ml.Transforms.Conversion.MapValueToKey("Label").Fit(data).Transform(data);
        ITransformer model = ml.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features").Fit(keyed);

        var mlnet = new MLNetAdapter(ml, model, K);
        Assert.Equal(K, mlnet.FeatureCount);
        Assert.Equal(3, mlnet.ActionCount);
        Assert.Equal(AdapterOrigin.Trained, mlnet.Origin);

        string dir = Path.Combine(Path.GetTempPath(), $"dotfly-adapters-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            string zip = Path.Combine(dir, "m.zip");
            ml.Model.Save(model, keyed.Schema, zip);
            MLNetAdapter loaded = MLNetAdapter.Load(ml, zip, K);

            string onnxPath = Path.Combine(dir, "m.onnx");
            using (FileStream fs = File.Create(onnxPath))
            {
                ml.Model.ConvertToOnnx(model, keyed, fs, "Score");
            }

            using var onnx = new OnnxAdapter(onnxPath, featureCount: K, outputName: FindScoreOutput(onnxPath));
            Assert.Equal(K, onnx.FeatureCount);
            Assert.Equal(3, onnx.ActionCount);

            int correct = 0, agree = 0, total = 0;
            var a = new float[3];
            var b = new float[3];
            var c = new float[3];
            foreach (FeatureRow row in Synthetic(20, 2))
            {
                mlnet.Evaluate(row.Features, a);
                loaded.Evaluate(row.Features, b);
                onnx.Evaluate(row.Features, c);
                int pa = ArgMax(a), pb = ArgMax(b), pc = ArgMax(c);
                Assert.Equal(pa, pb);
                agree += pa == pc ? 1 : 0;
                correct += pa == (int)row.Label ? 1 : 0;
                total++;
            }

            Assert.True(correct >= total * 0.95, $"accuracy {correct}/{total}");
            Assert.True(agree >= total * 0.98, $"ML.NET vs ONNX agreement {agree}/{total}");
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    private static string FindScoreOutput(string path)
    {
        using var s = new Microsoft.ML.OnnxRuntime.InferenceSession(path);
        return s.OutputMetadata.Keys.First(n => n.StartsWith("Score", StringComparison.OrdinalIgnoreCase));
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
        {
            if (v[i] > v[best])
            {
                best = i;
            }
        }

        return best;
    }
}
