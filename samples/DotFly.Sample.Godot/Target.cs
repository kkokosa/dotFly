using System;
using Godot;

namespace DotFly.Sample.Godot;

/// <summary>A stimulus that circles the screen; the retina encoder looks at it.</summary>
public partial class Target : Node2D
{
    private double _t;

    public override void _Process(double delta)
    {
        _t += delta;
        Vector2 size = GetViewportRect().Size;
        Position = size / 2 + new Vector2((float)Math.Cos(_t * 0.3) * size.X * 0.35f, (float)Math.Sin(_t * 0.5) * size.Y * 0.35f);
        QueueRedraw();
    }

    public override void _Draw() => DrawCircle(Vector2.Zero, 10, Colors.DeepSkyBlue);
}
