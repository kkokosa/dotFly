using System;
using Godot;

namespace DotFly.Sample.Godot;

/// <summary>
/// A 2D fly: heading and position driven by the decoded actions of <see cref="FlyBrain"/>. No
/// legs, no physics — the same open-loop "readout → controller" shape as the public demos.
/// </summary>
public partial class FlyBody : Node2D
{
    private FlyBrain? _brain;
    private Node2D? _target;
    private Label? _hud;
    private float _heading;   // radians, 0 = +x
    private int _frame;

    public override void _Ready()
    {
        _brain = GetNode<FlyBrain>("../FlyBrain");
        _target = GetNode<Node2D>("../Target");
        _hud = GetNode<Label>("../HUD");
    }

    public override void _Process(double delta)
    {
        if (_brain is null || _target is null)
        {
            return;
        }

        // Sense: bearing of the target relative to the heading, in (−π, π]; positive = to the right.
        Vector2 toTarget = _target.GlobalPosition - GlobalPosition;
        float bearing = Mathf.Wrap(toTarget.Angle() - _heading, -MathF.PI, MathF.PI);
        _brain.Encode(bearing, toTarget.Length());

        // Act: the brain's latest frame → decoder → heading and speed.
        _brain.Decode();
        (float yawDegPerSec, float speed) = _brain.Actions;
        _heading += Mathf.DegToRad(yawDegPerSec) * (float)delta;
        Position += new Vector2(MathF.Cos(_heading), MathF.Sin(_heading)) * speed * (float)delta;

        // Keep it on screen.
        Vector2 size = GetViewportRect().Size;
        Position = new Vector2(Mathf.Wrap(Position.X, 0, size.X), Mathf.Wrap(Position.Y, 0, size.Y));
        Rotation = _heading;

        if (_hud is not null)
        {
            _hud.Text = _brain.Hud();
        }

        if (++_frame % 120 == 0)
        {
            GD.Print(_brain.Hud());       // also visible when running headless
            if (DotFly.Cpu.Fast.CpuBackend.Profile && _brain.Profile() is { } prof)
            {
                GD.Print(prof);
            }
        }

        QueueRedraw();
    }

    public override void _Draw()
    {
        DrawPolygon([new Vector2(18, 0), new Vector2(-12, 10), new Vector2(-6, 0), new Vector2(-12, -10)], [Colors.Orange]);
        DrawCircle(Vector2.Zero, 4, Colors.Black);
    }
}
