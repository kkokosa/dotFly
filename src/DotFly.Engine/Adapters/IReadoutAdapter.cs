using System.Text.Json;
using System.Text.Json.Serialization;

namespace DotFly.Adapters;

/// <summary>How an adapter's parameters came to be — the label every demo must show.</summary>
public enum AdapterOrigin
{
    /// <summary>Hand-written rule or gain, not fitted to anything.</summary>
    Fixed,

    /// <summary>Hand-tuned by probing responses (Fly Marksman-style).</summary>
    Calibrated,

    /// <summary>Fitted by an optimiser on recorded frames or in the loop (Fly Dino-style).</summary>
    Trained,
}

/// <summary>
/// Maps a feature vector (an <see cref="OutputFrame"/>'s values) to an action vector. The graph
/// stays frozen; this is the only thing a demo trains, and it says so via <see cref="Origin"/>.
/// </summary>
public interface IReadoutAdapter
{
    /// <summary>Expected feature count.</summary>
    int FeatureCount { get; }

    /// <summary>Produced action count.</summary>
    int ActionCount { get; }

    /// <summary>Fixed, calibrated or trained.</summary>
    AdapterOrigin Origin { get; }

    /// <summary>Free-text provenance (what it was calibrated/trained on).</summary>
    string Description { get; }

    /// <summary>Computes <paramref name="actions"/> from <paramref name="features"/>.</summary>
    void Evaluate(ReadOnlySpan<float> features, Span<float> actions);
}

/// <summary>Convenience overloads.</summary>
public static class ReadoutAdapterExtensions
{
    /// <summary>Evaluates on a published frame.</summary>
    public static void Evaluate(this IReadoutAdapter adapter, OutputFrame frame, Span<float> actions) =>
        adapter.Evaluate(frame.Values, actions);

    /// <summary>Evaluates and returns a new action array.</summary>
    public static float[] Evaluate(this IReadoutAdapter adapter, ReadOnlySpan<float> features)
    {
        var a = new float[adapter.ActionCount];
        adapter.Evaluate(features, a);
        return a;
    }
}

/// <summary>Affine readout: <c>actions = W · features + b</c>. Fly Dino's 243-parameter reader is one of these.</summary>
public sealed class LinearAdapter : IReadoutAdapter
{
    private readonly float[] _w;   // [action * features + feature]
    private readonly float[] _b;

    /// <inheritdoc />
    public int FeatureCount { get; }

    /// <inheritdoc />
    public int ActionCount { get; }

    /// <inheritdoc />
    public AdapterOrigin Origin { get; }

    /// <inheritdoc />
    public string Description { get; }

    /// <summary>Weights, row-major [action][feature].</summary>
    public ReadOnlySpan<float> Weights => _w;

    /// <summary>Biases per action.</summary>
    public ReadOnlySpan<float> Biases => _b;

    /// <summary>Creates an adapter from weights (row-major [action][feature]) and biases.</summary>
    public LinearAdapter(int featureCount, int actionCount, ReadOnlySpan<float> weights, ReadOnlySpan<float> biases, AdapterOrigin origin, string description)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(featureCount);
        ArgumentOutOfRangeException.ThrowIfNegative(actionCount);
        if (weights.Length != featureCount * actionCount || biases.Length != actionCount)
        {
            throw new ArgumentException("Weight/bias sizes do not match the feature and action counts.");
        }

        FeatureCount = featureCount;
        ActionCount = actionCount;
        Origin = origin;
        Description = description;
        _w = weights.ToArray();
        _b = biases.ToArray();
    }

    /// <inheritdoc />
    public void Evaluate(ReadOnlySpan<float> features, Span<float> actions)
    {
        if (features.Length != FeatureCount || actions.Length != ActionCount)
        {
            throw new ArgumentException("Feature or action span has the wrong length.");
        }

        for (int a = 0; a < ActionCount; a++)
        {
            ReadOnlySpan<float> row = _w.AsSpan(a * FeatureCount, FeatureCount);
            actions[a] = _b[a] + System.Numerics.Tensors.TensorPrimitives.Dot(row, features);
        }
    }

    /// <summary>Saves as JSON.</summary>
    public void Save(string path) =>
        File.WriteAllText(path, JsonSerializer.Serialize(new LinearAdapterJson(FeatureCount, ActionCount, _w, _b, Origin.ToString(), Description), AdapterJsonContext.Default.LinearAdapterJson));

    /// <summary>Loads from JSON written by <see cref="Save"/>.</summary>
    public static LinearAdapter Load(string path)
    {
        LinearAdapterJson j = JsonSerializer.Deserialize(File.ReadAllText(path), AdapterJsonContext.Default.LinearAdapterJson)
            ?? throw new InvalidDataException("Empty adapter file.");
        return new LinearAdapter(j.Features, j.Actions, j.Weights, j.Biases, Enum.Parse<AdapterOrigin>(j.Origin), j.Description);
    }
}

/// <summary>
/// Rule-based readout: each action is <c>onValue</c> when its feature exceeds a threshold, else
/// <c>offValue</c> (Fly Marksman-style hand-written rules).
/// </summary>
public sealed class ThresholdAdapter : IReadoutAdapter
{
    /// <summary>One rule.</summary>
    /// <param name="Feature">Feature index.</param>
    /// <param name="Threshold">Threshold (strict &gt;).</param>
    /// <param name="OnValue">Action value when above.</param>
    /// <param name="OffValue">Action value otherwise.</param>
    public sealed record Rule(int Feature, float Threshold, float OnValue = 1f, float OffValue = 0f);

    private readonly Rule[] _rules;

    /// <inheritdoc />
    public int FeatureCount { get; }

    /// <inheritdoc />
    public int ActionCount => _rules.Length;

    /// <inheritdoc />
    public AdapterOrigin Origin { get; }

    /// <inheritdoc />
    public string Description { get; }

    /// <summary>Creates a rule adapter.</summary>
    public ThresholdAdapter(int featureCount, IReadOnlyList<Rule> rules, AdapterOrigin origin = AdapterOrigin.Fixed, string description = "hand-written thresholds")
    {
        ArgumentNullException.ThrowIfNull(rules);
        foreach (Rule r in rules)
        {
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual((uint)r.Feature, (uint)featureCount);
        }

        FeatureCount = featureCount;
        _rules = rules.ToArray();
        Origin = origin;
        Description = description;
    }

    /// <inheritdoc />
    public void Evaluate(ReadOnlySpan<float> features, Span<float> actions)
    {
        for (int a = 0; a < _rules.Length; a++)
        {
            Rule r = _rules[a];
            actions[a] = features[r.Feature] > r.Threshold ? r.OnValue : r.OffValue;
        }
    }
}

/// <summary>Exponential smoothing of another adapter's actions (stateful; call <see cref="Reset"/> between episodes).</summary>
public sealed class SmoothingAdapter : IReadoutAdapter
{
    private readonly IReadoutAdapter _inner;
    private readonly float _alpha;
    private readonly float[] _state;
    private readonly float[] _scratch;
    private bool _primed;

    /// <inheritdoc />
    public int FeatureCount => _inner.FeatureCount;

    /// <inheritdoc />
    public int ActionCount => _inner.ActionCount;

    /// <inheritdoc />
    public AdapterOrigin Origin => _inner.Origin;

    /// <inheritdoc />
    public string Description => $"{_inner.Description} (EMA α = {_alpha})";

    /// <summary>Wraps <paramref name="inner"/>; <paramref name="alpha"/> in (0, 1] is the weight of the newest value.</summary>
    public SmoothingAdapter(IReadoutAdapter inner, float alpha)
    {
        ArgumentNullException.ThrowIfNull(inner);
        if (alpha is <= 0 or > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha));
        }

        _inner = inner;
        _alpha = alpha;
        _state = new float[inner.ActionCount];
        _scratch = new float[inner.ActionCount];
    }

    /// <inheritdoc />
    public void Evaluate(ReadOnlySpan<float> features, Span<float> actions)
    {
        _inner.Evaluate(features, _scratch);
        for (int a = 0; a < _state.Length; a++)
        {
            _state[a] = _primed ? _state[a] + _alpha * (_scratch[a] - _state[a]) : _scratch[a];
            actions[a] = _state[a];
        }

        _primed = true;
    }

    /// <summary>Forgets the smoothing state.</summary>
    public void Reset() => _primed = false;
}

internal sealed record LinearAdapterJson(int Features, int Actions, float[] Weights, float[] Biases, string Origin, string Description);

[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(LinearAdapterJson))]
internal sealed partial class AdapterJsonContext : JsonSerializerContext
{
}
