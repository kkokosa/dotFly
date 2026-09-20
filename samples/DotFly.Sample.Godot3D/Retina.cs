using System;
using Godot;

namespace DotFly.Sample.Godot3D;

/// <summary>
/// The encoder: turns the fly's own eye images (two low-resolution renders, one camera per
/// compound eye, each looking 45° to its side; luminance only, as the R1–R6 photoreceptors that
/// feed motion vision are achromatic) into rates for ten visual input channels, per eye:
/// <list type="bullet">
/// <item><b>LC4</b> — looming: fraction of dark pixels in the eye and its growth.</item>
/// <item><b>T4a / T4b</b> — horizontal motion outward (front-to-back) / inward (back-to-front).</item>
/// <item><b>T4c / T4d</b> — vertical motion upward / downward.</item>
/// </list>
/// Motion is estimated by cross-correlating the eye's row/column intensity profiles with
/// the previous frame over small shifts. Everything here is hand-written engineering — it decides
/// what the network is told, and it is deliberately simple and inspectable.
/// </summary>
public sealed class Retina
{
    private readonly int _w;
    private readonly int _h;
    private readonly float[] _prevCols;   // per eye: column means (2 × w)
    private readonly float[] _prevRows;   // per eye: row means (2 × h)
    private readonly float[] _cols;
    private readonly float[] _rows;
    private readonly float[] _prevDark = new float[2];
    private bool _primed;

    /// <summary>Channel rates in Hz, per eye: index by <see cref="Channel"/>.</summary>
    public float[,] Rates { get; } = new float[2, 5];

    /// <summary>Raw measurements for the HUD: dark fraction, horizontal shift, vertical shift per eye.</summary>
    public (float Dark, float Dx, float Dy)[] Measured { get; } = new (float, float, float)[2];

    /// <summary>Channel indices of <see cref="Rates"/>.</summary>
    public enum Channel
    {
        /// <summary>Looming.</summary>
        LC4 = 0,

        /// <summary>Front-to-back horizontal motion.</summary>
        T4a = 1,

        /// <summary>Back-to-front horizontal motion.</summary>
        T4b = 2,

        /// <summary>Upward motion.</summary>
        T4c = 3,

        /// <summary>Downward motion.</summary>
        T4d = 4,
    }

    /// <summary>Peak rate for a fully looming eye (Hz).</summary>
    public float LoomHz { get; set; } = 100f;

    /// <summary>Rate per pixel-per-frame of image motion (Hz).</summary>
    public float MotionHzPerPx { get; set; } = 12f;

    /// <summary>
    /// A pixel counts as an object when darker than this fraction of the eye's mean luminance —
    /// relative, like photoreceptor adaptation, so a dim room and a bright arena both work.
    /// </summary>
    public float DarkFraction { get; set; } = 0.55f;

    /// <summary>Creates a retina for two eye images (left, right) of the given size each (L8 format).</summary>
    public Retina(int width, int height)
    {
        _w = width;
        _h = height;
        _prevCols = new float[2 * width];
        _cols = new float[2 * width];
        _prevRows = new float[2 * height];
        _rows = new float[2 * height];
    }

    /// <summary>Processes the two eye images (grayscale, row-major, <c>width × height</c> bytes each).</summary>
    public void Process(ReadOnlySpan<byte> left, ReadOnlySpan<byte> right)
    {
        Array.Clear(_cols);
        Array.Clear(_rows);
        Span<int> dark = stackalloc int[2];
        for (int eye = 0; eye < 2; eye++)
        {
            ReadOnlySpan<byte> l8 = eye == 0 ? left : right;
            long sum = 0;
            for (int i = 0; i < l8.Length; i++)
            {
                sum += l8[i];
            }

            float threshold = DarkFraction * sum / MathF.Max(1, l8.Length);
            for (int y = 0; y < _h; y++)
            {
                ReadOnlySpan<byte> row = l8.Slice(y * _w, _w);
                for (int x = 0; x < _w; x++)
                {
                    byte v = row[x];
                    _cols[eye * _w + x] += v;
                    _rows[eye * _h + y] += v;
                    if (v < threshold)
                    {
                        dark[eye]++;
                    }
                }
            }
        }

        for (int i = 0; i < 2 * _w; i++)
        {
            _cols[i] /= _h;
        }

        for (int i = 0; i < 2 * _h; i++)
        {
            _rows[i] /= _w;
        }

        for (int eye = 0; eye < 2; eye++)
        {
            float darkFrac = dark[eye] / (float)(_w * _h);
            float dx = 0, dy = 0;
            if (_primed)
            {
                dx = BestShift(_prevCols.AsSpan(eye * _w, _w), _cols.AsSpan(eye * _w, _w));
                dy = BestShift(_prevRows.AsSpan(eye * _h, _h), _rows.AsSpan(eye * _h, _h));
            }

            // Each eye camera looks 45° to its side: outward (front-to-back) image motion is
            // leftward (negative dx) on the left eye and rightward on the right eye.
            float outward = eye == 0 ? -dx : dx;
            float growth = MathF.Max(0f, darkFrac - _prevDark[eye]) * 10f;
            Rates[eye, (int)Channel.LC4] = Mathf.Clamp(LoomHz * (darkFrac * 0.6f + growth), 0f, LoomHz);
            Rates[eye, (int)Channel.T4a] = Mathf.Clamp(MotionHzPerPx * MathF.Max(0f, outward), 0f, 60f);
            Rates[eye, (int)Channel.T4b] = Mathf.Clamp(MotionHzPerPx * MathF.Max(0f, -outward), 0f, 60f);
            Rates[eye, (int)Channel.T4c] = Mathf.Clamp(MotionHzPerPx * MathF.Max(0f, -dy), 0f, 60f);   // image moving up = negative y shift
            Rates[eye, (int)Channel.T4d] = Mathf.Clamp(MotionHzPerPx * MathF.Max(0f, dy), 0f, 60f);
            Measured[eye] = (darkFrac, dx, dy);
            _prevDark[eye] = darkFrac;
        }

        _cols.CopyTo(_prevCols, 0);
        _rows.CopyTo(_prevRows, 0);
        _primed = true;
    }

    /// <summary>Shift (in samples, −3..3) that best aligns <paramref name="prev"/> to <paramref name="cur"/> (cur ≈ prev shifted by +shift).</summary>
    private static float BestShift(ReadOnlySpan<float> prev, ReadOnlySpan<float> cur)
    {
        const int maxShift = 3;
        float bestErr = float.MaxValue;
        int best = 0;
        for (int s = -maxShift; s <= maxShift; s++)
        {
            float err = 0;
            int n = 0;
            for (int i = maxShift; i < cur.Length - maxShift; i++)
            {
                float d = cur[i] - prev[i - s];
                err += d * d;
                n++;
            }

            err /= Math.Max(1, n);
            if (err < bestErr - 1e-3f)
            {
                bestErr = err;
                best = s;
            }
        }

        // No structure (flat image) → no motion.
        return bestErr < 1e-4f && best != 0 ? 0 : best;
    }
}
