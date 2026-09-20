using DotFly.Adapters;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace DotFly.Adapters.MLNet;

/// <summary>Feature row for ML.NET: one float vector. The vector size is set at runtime via a schema definition.</summary>
public sealed class FeatureRow
{
    /// <summary>The feature vector (an output frame's values).</summary>
    public float[] Features { get; set; } = [];

    /// <summary>Label for training (class index or regression target); unused at prediction time.</summary>
    public float Label { get; set; }
}

/// <summary>Prediction row: the model's score vector (class scores/probabilities or a regression value).</summary>
public sealed class ScoreRow
{
    /// <summary>Scores.</summary>
    public float[] Score { get; set; } = [];
}

/// <summary>
/// Runs a trained ML.NET model as a readout. The model's <c>Score</c> column is the action vector
/// (per-class scores for classifiers, one value for regressors).
/// </summary>
public sealed class MLNetAdapter : IReadoutAdapter
{
    private readonly PredictionEngine<FeatureRow, ScoreRow> _engine;
    private readonly FeatureRow _row;

    /// <inheritdoc />
    public int FeatureCount { get; }

    /// <inheritdoc />
    public int ActionCount { get; }

    /// <inheritdoc />
    public AdapterOrigin Origin { get; }

    /// <inheritdoc />
    public string Description { get; }

    /// <summary>Wraps a trained transformer.</summary>
    public MLNetAdapter(MLContext ml, ITransformer model, int featureCount, AdapterOrigin origin = AdapterOrigin.Trained, string description = "ML.NET model")
    {
        ArgumentNullException.ThrowIfNull(ml);
        ArgumentNullException.ThrowIfNull(model);
        FeatureCount = featureCount;
        Origin = origin;
        Description = description;
        SchemaDefinition schema = FeatureSchema(featureCount);
        _engine = ml.Model.CreatePredictionEngine<FeatureRow, ScoreRow>(model, inputSchemaDefinition: schema);
        _row = new FeatureRow { Features = new float[featureCount] };
        ActionCount = _engine.Predict(_row).Score.Length;
    }

    /// <summary>Loads a model saved with <see cref="MLContext.Model"/>.</summary>
    public static MLNetAdapter Load(MLContext ml, string path, int featureCount, AdapterOrigin origin = AdapterOrigin.Trained, string? description = null)
    {
        ITransformer model = ml.Model.Load(path, out _);
        return new MLNetAdapter(ml, model, featureCount, origin, description ?? $"ML.NET model {Path.GetFileName(path)}");
    }

    /// <summary>Schema definition with the feature vector sized at runtime.</summary>
    public static SchemaDefinition FeatureSchema(int featureCount)
    {
        SchemaDefinition schema = SchemaDefinition.Create(typeof(FeatureRow));
        schema[nameof(FeatureRow.Features)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, featureCount);
        return schema;
    }

    /// <inheritdoc />
    public void Evaluate(ReadOnlySpan<float> features, Span<float> actions)
    {
        if (features.Length != FeatureCount || actions.Length != ActionCount)
        {
            throw new ArgumentException("Feature or action span has the wrong length.");
        }

        features.CopyTo(_row.Features);
        ScoreRow score = _engine.Predict(_row);
        score.Score.AsSpan(0, ActionCount).CopyTo(actions);
    }
}
