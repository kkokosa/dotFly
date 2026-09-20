using DotFly.Adapters;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace DotFly.Adapters.Onnx;

/// <summary>
/// Runs an ONNX model as a readout: features in (a <c>[1, K]</c> float tensor), actions out (the
/// first float output, flattened). Load anything trained elsewhere — PyTorch, scikit-learn,
/// ML.NET — as long as it takes one float vector and returns one.
/// </summary>
public sealed class OnnxAdapter : IReadoutAdapter, IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly string _outputName;
    private readonly float[] _inputBuffer;
    private readonly long[] _inputShape;
    private readonly Dictionary<string, OrtValue> _feeds = new();   // feature input + zero-filled extras

    /// <inheritdoc />
    public int FeatureCount { get; }

    /// <inheritdoc />
    public int ActionCount { get; }

    /// <inheritdoc />
    public AdapterOrigin Origin { get; }

    /// <inheritdoc />
    public string Description { get; }

    /// <summary>
    /// Loads a model. <paramref name="inputName"/>/<paramref name="outputName"/> default to the
    /// model's first input/output. The feature count comes from the input shape's last dimension
    /// (or <paramref name="featureCount"/> when the shape is symbolic); the action count from a
    /// probe run with zeros.
    /// </summary>
    public OnnxAdapter(string modelPath, AdapterOrigin origin = AdapterOrigin.Trained, string? description = null, string? inputName = null, string? outputName = null, int? featureCount = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        _session = new InferenceSession(modelPath);
        Origin = origin;
        Description = description ?? $"ONNX model {Path.GetFileName(modelPath)}";

        _inputName = inputName ?? _session.InputMetadata.Keys.First();
        NodeMetadata input = _session.InputMetadata[_inputName];
        int[] dims = input.Dimensions;
        int k = featureCount ?? (dims.Length > 0 && dims[^1] > 0 ? dims[^1] : throw new ArgumentException("Input shape is symbolic; pass featureCount.", nameof(featureCount)));
        FeatureCount = k;
        _inputShape = dims.Length == 0 ? [1, k] : dims.Select((d, i) => (long)(d > 0 ? d : (i == dims.Length - 1 ? k : 1))).ToArray();
        _inputBuffer = new float[_inputShape.Aggregate(1L, (a, b) => a * b)];

        _outputName = outputName ?? _session.OutputMetadata.First(kv => kv.Value.ElementType == typeof(float)).Key;

        // Extra graph inputs (e.g. the label column an ML.NET export drags along) are fed zeros of
        // their declared type and shape, symbolic dimensions taken as 1.
        _feeds[_inputName] = OrtValue.CreateTensorValueFromMemory(_inputBuffer, _inputShape);
        foreach ((string name, NodeMetadata meta) in _session.InputMetadata)
        {
            if (name == _inputName || !meta.IsTensor)
            {
                continue;
            }

            long[] shape = meta.Dimensions.Length == 0 ? [1] : meta.Dimensions.Select(d => (long)Math.Max(d, 1)).ToArray();
            _feeds[name] = ZeroTensor(meta.ElementType, shape);
        }

        ActionCount = Probe();
    }

    private static OrtValue ZeroTensor(Type elementType, long[] shape)
    {
        long n = shape.Aggregate(1L, (a, b) => a * b);
        return elementType == typeof(float) ? OrtValue.CreateTensorValueFromMemory(new float[n], shape)
            : elementType == typeof(double) ? OrtValue.CreateTensorValueFromMemory(new double[n], shape)
            : elementType == typeof(long) ? OrtValue.CreateTensorValueFromMemory(new long[n], shape)
            : elementType == typeof(int) ? OrtValue.CreateTensorValueFromMemory(new int[n], shape)
            : elementType == typeof(bool) ? OrtValue.CreateTensorValueFromMemory(new bool[n], shape)
            : elementType == typeof(string) ? OrtValue.CreateFromStringTensor(new DenseTensor<string>(Enumerable.Repeat(string.Empty, (int)n).ToArray(), shape.Select(d => (int)d).ToArray()))
            : throw new NotSupportedException($"Unsupported extra input element type {elementType.Name}.");
    }

    private int Probe()
    {
        using IDisposableReadOnlyCollection<OrtValue> outputs = _session.Run(new RunOptions(), _feeds, [_outputName]);
        return (int)outputs[0].GetTensorTypeAndShape().ElementCount;
    }

    /// <inheritdoc />
    public void Evaluate(ReadOnlySpan<float> features, Span<float> actions)
    {
        if (features.Length != FeatureCount || actions.Length != ActionCount)
        {
            throw new ArgumentException("Feature or action span has the wrong length.");
        }

        features.CopyTo(_inputBuffer);            // the feature OrtValue wraps this buffer (pinned, no copy)
        using IDisposableReadOnlyCollection<OrtValue> outputs = _session.Run(new RunOptions(), _feeds, [_outputName]);
        outputs[0].GetTensorDataAsSpan<float>()[..ActionCount].CopyTo(actions);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        foreach (OrtValue v in _feeds.Values)
        {
            v.Dispose();
        }

        _session.Dispose();
    }
}
