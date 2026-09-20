using System;
using Godot;

namespace DotFly.Sample.Godot3D;

/// <summary>
/// What the fly sees: the two eye renders (left eye, right eye — each a 32×32 luminance image from
/// a camera looking 45° to its side; colour is dropped because the motion/looming photoreceptors
/// R1–R6 are achromatic) and the channel rates the retina encoder derives from them.
/// </summary>
public partial class EyeView : Control
{
    private static readonly string[] Channels = ["LC4", "T4a", "T4b", "T4c", "T4d"];
    private Texture2D? _left;
    private Texture2D? _right;
    private Retina? _retina;
    private FlyBrain3D? _brain;

    /// <summary>Binds the view.</summary>
    public void Bind(Texture2D left, Texture2D right, Retina retina, FlyBrain3D brain)
    {
        _left = left;
        _right = right;
        _retina = retina;
        _brain = brain;
    }

    public override void _Process(double delta) => QueueRedraw();

    public override void _Draw()
    {
        DrawRect(new Rect2(0, 0, Size), new Color(0.05f, 0.05f, 0.07f, 0.9f));
        Font font = ThemeDB.FallbackFont;
        DrawString(font, new Vector2(8, 22), "eyes — 32×32 luminance, L / R cameras → retina", HorizontalAlignment.Left, -1, 18, new Color(0.8f, 0.8f, 0.85f));
        if (_left is null || _right is null || _retina is null || _brain is null)
        {
            return;
        }

        // Two square eye images side by side, then the channel bars; everything scales with the width.
        float barsW = 190f;
        float eyeSize = MathF.Min((Size.X - 24 - barsW) / 2f, Size.Y - 84);
        float rowH = 26f;
        var rectL = new Rect2(8, 32, eyeSize, eyeSize);
        var rectR = new Rect2(8 + eyeSize + 4, 32, eyeSize, eyeSize);
        DrawTextureRect(_left, rectL, false, new Color(1, 1, 1));
        DrawTextureRect(_right, rectR, false, new Color(1, 1, 1));
        DrawString(font, new Vector2(rectL.Position.X + 4, rectL.Position.Y + 12), "L", HorizontalAlignment.Left, -1, 16, new Color(1, 1, 1, 0.8f));
        DrawString(font, new Vector2(rectR.Position.X + 4, rectR.Position.Y + 12), "R", HorizontalAlignment.Left, -1, 16, new Color(1, 1, 1, 0.8f));

        float bx = rectR.End.X + 10;
        for (int c = 0; c < Channels.Length; c++)
        {
            float y = rectL.Position.Y + c * rowH;
            DrawString(font, new Vector2(bx, y + 14), Channels[c], HorizontalAlignment.Left, -1, 15, new Color(0.7f, 0.7f, 0.8f));
            for (int eye = 0; eye < 2; eye++)
            {
                float rate = _brain.InputRates[2 * c + eye];
                var track = new Rect2(bx + 40 + eye * 72, y + 4, 66, 10);
                DrawRect(track, new Color(0.15f, 0.15f, 0.2f));
                DrawRect(new Rect2(track.Position, new Vector2(MathF.Min(1f, rate / 100f) * track.Size.X, track.Size.Y)), new Color(0.3f, 0.6f, 1f));
                DrawString(font, new Vector2(track.Position.X, y + 27), $"{rate:F0}", HorizontalAlignment.Left, -1, 12, new Color(0.6f, 0.7f, 0.9f));
            }
        }

        (float dark0, float dx0, float dy0) = _retina.Measured[0];
        (float dark1, float dx1, float dy1) = _retina.Measured[1];
        float ty = MathF.Max(rectL.End.Y, rectL.Position.Y + 5 * rowH) + 14;
        DrawString(font, new Vector2(8, ty), $"L: dark {dark0:P0} dx {dx0:+0;-0;0} dy {dy0:+0;-0;0}    R: dark {dark1:P0} dx {dx1:+0;-0;0} dy {dy1:+0;-0;0}    (px/frame)", HorizontalAlignment.Left, -1, 15, new Color(0.6f, 0.6f, 0.7f));
        DrawString(font, new Vector2(8, ty + 18), $"smell ORN L {_brain.InputRates[10]:F0} / R {_brain.InputRates[11]:F0}   taste {_brain.InputRates[12]:F0}/{_brain.InputRates[13]:F0}   wind JO L {_brain.InputRates[14]:F0} / R {_brain.InputRates[15]:F0} Hz", HorizontalAlignment.Left, -1, 15, new Color(0.9f, 0.6f, 1f));
    }
}
