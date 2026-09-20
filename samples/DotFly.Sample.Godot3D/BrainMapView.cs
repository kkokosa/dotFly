using System;
using System.Collections.Generic;
using DotFly.Core.Graph;
using Godot;

namespace DotFly.Sample.Godot3D;

/// <summary>
/// The brain, drawn from the checkpoint's real soma positions (MaleCNS voxel coordinates; head to
/// the left, ventral nerve cord to the right) and lit by real spikes: every neuron's recent
/// activity (decayed spike count) colours its soma pixel. Input populations are outlined in blue,
/// readout groups carry coloured markers. This is what the network is doing right now.
/// </summary>
public partial class BrainMapView : Control
{
    private int MapW = 400;      // set at bind time from the anatomy's aspect (z-extent / x-extent)
    private const int MapH = 260;
    private const int Margin = 8;

    private FlyBrain3D? _brain;
    private int[] _pixelOfNeuron = [];      // −1 when the soma is unknown
    private byte[] _rgba = [];
    private float[] _heat = [];
    private byte[] _baseCount = [];
    private Image? _image;
    private ImageTexture? _texture;
    private readonly List<(Vector2 Pos, Color Color, string Label)> _markers = [];
    private readonly List<Vector2> _inputPixels = [];
    private float _peak = 1f;
    private float[]? _splat;
    private bool[] _isInput = [];
    private float _neckX = 0.6f;
    private bool _vncCompressed = true;
    private double _accum;
    private const double UpdateInterval = 1.0 / 15;   // the map is the most expensive panel: 15 Hz is plenty for a 150 ms trace

    /// <summary>Colours of the readout groups (shared with the trace view).</summary>
    public static readonly Color[] GroupColors =
    [
        new(1f, 0.35f, 0.35f), new(1f, 0.6f, 0.2f), new(1f, 0.9f, 0.3f),
        new(0.4f, 0.9f, 1f), new(0.3f, 0.6f, 1f), new(0.6f, 1f, 0.6f), new(0.3f, 0.8f, 0.4f),
        new(0.75f, 0.85f, 1f), new(0.55f, 0.65f, 0.9f),
        new(0.9f, 0.5f, 1f), new(1f, 0.4f, 0.8f),
        new(0.7f, 0.95f, 0.85f), new(0.45f, 0.8f, 0.7f),
    ];

    /// <summary>Binds the view to the brain once it is loaded.</summary>
    public void Bind(FlyBrain3D brain)
    {
        _brain = brain;
        Brain? b = brain.Brain;
        if (b is null)
        {
            return;
        }

        // Projection: horizontal = z (anterior → posterior), vertical = x (left → right), from the
        // 1st–99th percentile of known soma positions so outliers do not squash the map.
        ReadOnlySpan<int> soma = b.SomaPositions;
        int n = b.NeuronCount;
        var xs = new List<int>(n);
        var zs = new List<int>(n);
        for (int i = 0; i < n; i++)
        {
            if (soma[3 * i] != int.MinValue)
            {
                xs.Add(soma[3 * i]);
                zs.Add(soma[3 * i + 2]);
            }
        }

        xs.Sort();
        zs.Sort();
        float x0 = xs[xs.Count / 100], x1 = xs[xs.Count * 99 / 100];
        float z0 = zs[zs.Count / 100], z1 = zs[zs.Count * 99 / 100];

        // The neck: the z-bin (in the middle 25–75 % of the axis) where the somata are narrowest.
        // The brain (before the neck) is drawn to scale in BrainShare of the width; the VNC is
        // compressed into the rest, so the brain — where all inputs and readouts are — is large.
        const int bins = 64;
        var lo = new float[bins];
        var hi = new float[bins];
        Array.Fill(lo, float.MaxValue);
        Array.Fill(hi, float.MinValue);
        for (int i = 0; i < n; i++)
        {
            if (soma[3 * i] == int.MinValue)
            {
                continue;
            }

            int bin = Math.Clamp((int)((soma[3 * i + 2] - z0) / (z1 - z0) * bins), 0, bins - 1);
            lo[bin] = MathF.Min(lo[bin], soma[3 * i]);
            hi[bin] = MathF.Max(hi[bin], soma[3 * i]);
        }

        int neck = bins / 2;
        float narrowest = float.MaxValue;
        for (int bin = bins / 4; bin < bins * 3 / 4; bin++)
        {
            float spread = hi[bin] - lo[bin];
            if (spread < narrowest)
            {
                narrowest = spread;
                neck = bin;
            }
        }

        float zNeck = z0 + (neck + 0.5f) * (z1 - z0) / bins;
        // The texture takes the panel's aspect so the map fills the panel; the brain is to scale,
        // the VNC gets whatever width is left (compressed only if it does not fit).
        float areaW = MathF.Max(100f, Size.X - 2 * Margin), areaH = MathF.Max(60f, Size.Y - 56);
        MapW = Math.Clamp((int)MathF.Round(MapH * areaW / areaH), 300, 1400);
        float brainPx = MapH * (zNeck - z0) / MathF.Max(1f, x1 - x0);
        float vncPx = MapH * (z1 - zNeck) / MathF.Max(1f, x1 - x0);
        float brainShare = Math.Clamp(brainPx / MapW, 0.3f, 0.85f);
        _vncCompressed = vncPx > (1 - brainShare) * MapW + 1;
        float MapX(int z) => z < zNeck
            ? (z - z0) / (zNeck - z0) * brainShare * (MapW - 1)
            : (brainShare + (z - zNeck) / (z1 - zNeck) * (1 - brainShare)) * (MapW - 1);
        _neckX = brainShare;

        _pixelOfNeuron = new int[n];
        _baseCount = new byte[MapW * MapH];
        _isInput = new bool[MapW * MapH];
        for (int i = 0; i < n; i++)
        {
            if (soma[3 * i] == int.MinValue)
            {
                _pixelOfNeuron[i] = -1;
                continue;
            }

            int px = (int)Math.Clamp(MapX(soma[3 * i + 2]), 0, MapW - 1);
            int py = (int)Math.Clamp((soma[3 * i] - x0) / (x1 - x0) * (MapH - 1), 0, MapH - 1);
            int p = py * MapW + px;
            _pixelOfNeuron[i] = p;
            if (_baseCount[p] < 255)
            {
                _baseCount[p]++;
            }
        }

        foreach (NeuronSet set in brain.InputSets)
        {
            foreach (int i in set.Indices)
            {
                if (_pixelOfNeuron[i] >= 0)
                {
                    _inputPixels.Add(new Vector2(_pixelOfNeuron[i] % MapW, _pixelOfNeuron[i] / MapW));
                    _isInput[_pixelOfNeuron[i]] = true;
                }
            }
        }

        for (int g = 0; g < brain.Groups.Count; g++)
        {
            Vector2 sum = Vector2.Zero;
            int count = 0;
            foreach (int i in brain.Groups[g].Indices)
            {
                if (_pixelOfNeuron[i] >= 0)
                {
                    sum += new Vector2(_pixelOfNeuron[i] % MapW, _pixelOfNeuron[i] / MapW);
                    count++;
                }
            }

            if (count > 0)
            {
                _markers.Add((sum / count, GroupColors[g], FlyBrain3D.GroupNames[g]));
            }
        }

        _rgba = new byte[MapW * MapH * 4];
        _heat = new float[MapW * MapH];
        _image = Image.CreateEmpty(MapW, MapH, false, Image.Format.Rgba8);
        _texture = ImageTexture.CreateFromImage(_image);
    }

    public override void _Process(double delta)
    {
        if (_brain?.Brain is null || _image is null || _texture is null || !IsVisibleInTree())
        {
            return;
        }

        _accum += delta;
        if (_accum < UpdateInterval)
        {
            return;
        }

        _accum = 0;
        float drive = 0;
        foreach (float r in _brain.InputRates)
        {
            drive = MathF.Max(drive, r);
        }

        float inputBlue = 0.25f + 0.5f * MathF.Min(1f, drive / 100f);
        float[] activity = _brain.Activity;
        Array.Clear(_heat);
        float peak = 0.5f;
        for (int i = 0; i < _pixelOfNeuron.Length; i++)
        {
            int p = _pixelOfNeuron[i];
            float a = activity[i];
            if (p >= 0 && a > 0.02f)
            {
                float h = _heat[p] + a;
                _heat[p] = h;
                if (h > peak)
                {
                    peak = h;
                }
            }
        }

        _peak += (peak - _peak) * 0.1f;   // slow auto-scale
        float inv = 1f / MathF.Max(_peak * 0.5f, 1f);

        // Splat each hot pixel onto its 3×3 neighbourhood so single spiking neurons stay visible.
        Span<float> src = _heat;
        var splat = _splat ??= new float[_heat.Length];
        Array.Clear(splat);
        for (int p = 0; p < src.Length; p++)
        {
            float h = src[p];
            if (h <= 0.02f)
            {
                continue;
            }

            int x = p % MapW, y = p / MapW;
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    int xx = x + dx, yy = y + dy;
                    if ((uint)xx < MapW && (uint)yy < MapH)
                    {
                        float w = dx == 0 && dy == 0 ? 1f : 0.45f;
                        int q = yy * MapW + xx;
                        splat[q] = MathF.Max(splat[q], h * w);
                    }
                }
            }
        }

        for (int p = 0; p < _heat.Length; p++)
        {
            float h = MathF.Sqrt(MathF.Min(1f, splat[p] * inv));   // gamma lifts faint activity
            float baseGrey = _baseCount[p] > 0 ? 0.16f + MathF.Min(0.14f, _baseCount[p] * 0.02f) : 0.03f;
            // Heat colormap: dark → red → yellow → white.
            float r = baseGrey + (1f - baseGrey) * MathF.Min(1f, h * 2f);
            float g = baseGrey + (1f - baseGrey) * MathF.Max(0f, MathF.Min(1f, h * 2f - 0.6f));
            float bl = baseGrey + (1f - baseGrey) * MathF.Max(0f, h * 2f - 1.4f);
            if (_isInput[p])
            {
                // Input populations tinted blue, brighter when driven (baked here: 7k draw calls otherwise).
                r = r * (1 - inputBlue) + 0.3f * inputBlue;
                g = g * (1 - inputBlue) + 0.5f * inputBlue;
                bl = bl * (1 - inputBlue) + 1f * inputBlue;
            }

            int o = p * 4;
            _rgba[o] = (byte)(r * 255);
            _rgba[o + 1] = (byte)(g * 255);
            _rgba[o + 2] = (byte)(bl * 255);
            _rgba[o + 3] = 255;
        }

        _image.SetData(MapW, MapH, false, Image.Format.Rgba8, _rgba);
        _texture.Update(_image);
        QueueRedraw();
    }

    public override void _Draw()
    {
        DrawRect(new Rect2(0, 0, Size), new Color(0.05f, 0.05f, 0.07f, 0.9f));
        Font font = ThemeDB.FallbackFont;
        DrawString(font, new Vector2(Margin, 22), "brain — real somata, real spikes (head ◀ · VNC ▶)", HorizontalAlignment.Left, -1, 18, new Color(0.8f, 0.8f, 0.85f));
        if (_texture is null)
        {
            DrawString(font, new Vector2(Margin, 40), "loading…", HorizontalAlignment.Left, -1, 18);
            return;
        }

        // Fit the map into the control (it scales with the panel).
        float scale = MathF.Min((Size.X - 2 * Margin) / MapW, (Size.Y - 56) / MapH);
        var origin = new Vector2(Margin, 30);
        DrawTextureRect(_texture, new Rect2(origin, new Vector2(MapW * scale, MapH * scale)), false);

        // Readout markers.
        for (int m = 0; m < _markers.Count; m++)
        {
            (Vector2 pos, Color color, string label) = _markers[m];
            float rate = _brain is not null && m < _brain.Rates.Length ? _brain.Rates[m] : 0;
            float radius = (3f + MathF.Min(6f, rate / 40f)) * MathF.Max(0.8f, scale);
            Vector2 at = origin + pos * scale;
            DrawCircle(at, radius + 1.5f, new Color(0, 0, 0, 0.7f));
            DrawCircle(at, radius, color);
            DrawString(font, at + new Vector2(radius + 3, 4), label, HorizontalAlignment.Left, -1, 15, color);
        }

        float neckPx = origin.X + _neckX * MapW * scale;
        DrawLine(new Vector2(neckPx, origin.Y), new Vector2(neckPx, origin.Y + MapH * scale), new Color(0.5f, 0.5f, 0.6f, 0.5f), 1f);
        DrawString(font, new Vector2(neckPx + 4, origin.Y + MapH * scale - 4), _vncCompressed ? "VNC (compressed)" : "VNC", HorizontalAlignment.Left, -1, 12, new Color(0.6f, 0.6f, 0.7f));
        DrawString(font, new Vector2(Margin, origin.Y + MapH * scale + 18), "blue = inputs (LC4, T4, ORN, GRN)   ● = readouts, size = rate", HorizontalAlignment.Left, -1, 15, new Color(0.6f, 0.6f, 0.7f));
    }
}
