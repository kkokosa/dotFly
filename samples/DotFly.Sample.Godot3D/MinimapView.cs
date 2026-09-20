using System;
using System.Collections.Generic;
using Godot;

namespace DotFly.Sample.Godot3D;

/// <summary>
/// Isometric projection of the room with the fly's trajectory: floor, columns, furniture, sugar
/// patches, the wind, the flight path coloured by age (with feeding stops marked) and the fly's
/// current position with a drop line to the floor.
/// </summary>
public partial class MinimapView : Control
{
    private const float Cos30 = 0.8660254f;
    private const float Sin30 = 0.5f;
    private const int MaxPoints = 6000;      // 10 minutes at 10 Hz

    private Arena? _arena;
    private FlyBody3D? _fly;
    private readonly List<(Vector3 Pos, byte Feeding)> _path = [];
    private Vector2[] _pathPoints = [];
    private Color[] _pathColors = [];
    private double _accum;
    private float _scale = 6f;
    private Vector2 _center;

    /// <summary>Binds the view.</summary>
    public void Bind(Arena arena, FlyBody3D fly)
    {
        _arena = arena;
        _fly = fly;
    }

    public override void _Process(double delta)
    {
        if (_fly is null)
        {
            return;
        }

        _accum += delta;
        if (_accum >= 0.1)
        {
            _accum = 0;
            _path.Add((_fly.GlobalPosition, (byte)(_fly.State == Behaviour.State.Feeding ? 1 : 0)));
            if (_path.Count > MaxPoints)
            {
                _path.RemoveRange(0, _path.Count - MaxPoints);
            }
        }

        QueueRedraw();
    }

    private Vector2 Iso(Vector3 p) => _center + new Vector2((p.X - p.Z) * Cos30, (p.X + p.Z) * Sin30 - p.Y * 0.9f) * _scale;

    public override void _Draw()
    {
        DrawRect(new Rect2(0, 0, Size), new Color(0.05f, 0.05f, 0.07f, 0.85f));
        Font font = ThemeDB.FallbackFont;
        DrawString(font, new Vector2(8, 16), "room — isometric: odor ribbons, flight path (bright = recent, ● = feeding)", HorizontalAlignment.Left, -1, 11, new Color(0.8f, 0.8f, 0.85f));
        if (_arena is null || _fly is null)
        {
            return;
        }

        float hx = _arena.HalfX, hz = _arena.HalfZ, h = _arena.Height;
        _scale = MathF.Min((Size.X - 16) / ((hx + hz) * 2 * Cos30), (Size.Y - 40) / ((hx + hz) * 2 * Sin30 + h * 0.9f));
        _center = new Vector2(Size.X / 2, 24 + ((hx + hz) * Sin30 + h * 0.9f) * _scale + (Size.Y - 40 - ((hx + hz) * 2 * Sin30 + h * 0.9f) * _scale) / 2);

        // Floor and the wall edges.
        var floor = new Color(0.18f, 0.16f, 0.14f);
        var edge = new Color(0.5f, 0.5f, 0.55f);
        Vector2[] corners = [Iso(new Vector3(-hx, 0, -hz)), Iso(new Vector3(hx, 0, -hz)), Iso(new Vector3(hx, 0, hz)), Iso(new Vector3(-hx, 0, hz))];
        DrawColoredPolygon(corners, floor);
        DrawPolyline([.. corners, corners[0]], edge, 1f);
        foreach (Vector3 c in new[] { new Vector3(-hx, 0, -hz), new Vector3(hx, 0, -hz), new Vector3(-hx, 0, hz) })
        {
            DrawLine(Iso(c), Iso(c + new Vector3(0, h, 0)), new Color(edge, 0.5f), 1f);
        }

        DrawLine(Iso(new Vector3(-hx, h, -hz)), Iso(new Vector3(hx, h, -hz)), new Color(edge, 0.35f), 1f);
        DrawLine(Iso(new Vector3(-hx, h, -hz)), Iso(new Vector3(-hx, h, hz)), new Color(edge, 0.35f), 1f);

        // Sugar patches (ellipses on their surface), their meandering plumes, and the wind.
        foreach (Odor.Patch patch in _arena.Odor.Patches)
        {
            DrawEllipse(patch.Position, patch.Radius, new Color(1f, 0.8f, 0.25f, 0.8f));
            var ribbon = new Vector2[24];
            var ribbonColors = new Color[24];
            for (int i = 0; i < ribbon.Length; i++)
            {
                float along = i * _arena.Odor.PlumeLength / (ribbon.Length - 1);
                Vector3 c = _arena.Odor.Centreline(patch, along);   // 3-D: the map shows the tunnel's height too
                c.X = Math.Clamp(c.X, -hx, hx);
                c.Z = Math.Clamp(c.Z, -hz, hz);
                ribbon[i] = Iso(c);
                ribbonColors[i] = new Color(1f, 0.85f, 0.4f, 0.75f * MathF.Exp(-along / _arena.Odor.PlumeLength));
            }

            DrawPolylineColors(ribbon, ribbonColors, 2f);
        }

        Vector3 wind = _arena.Odor.Wind;
        Vector2 w0 = Iso(new Vector3(hx - 3, h, hz - 2)), w1 = Iso(new Vector3(hx - 3, h, hz - 2) + wind * 2f);
        DrawLine(w0, w1, new Color(0.7f, 0.85f, 1f), 1.5f);
        DrawCircle(w1, 2f, new Color(0.7f, 0.85f, 1f));
        DrawString(font, w0 + new Vector2(4, -4), "wind", HorizontalAlignment.Left, -1, 9, new Color(0.7f, 0.85f, 1f));

        // Furniture and columns.
        foreach (Arena.Box box in _arena.Boxes)
        {
            DrawBox(box.Center, box.Size, box.Color);
        }

        foreach (Arena.Column col in _arena.Columns)
        {
            var c = new Color(0.55f, 0.4f, 0.3f);
            DrawEllipse(col.Base, col.Radius, c);
            DrawLine(Iso(col.Base + new Vector3(-col.Radius, 0, col.Radius)), Iso(col.Base + new Vector3(-col.Radius, col.Height, col.Radius)), c, 1.5f);
            DrawLine(Iso(col.Base + new Vector3(col.Radius, 0, -col.Radius)), Iso(col.Base + new Vector3(col.Radius, col.Height, -col.Radius)), c, 1.5f);
            DrawEllipse(col.Base + new Vector3(0, col.Height, 0), col.Radius, c);
        }

        // Trajectory, bright when recent (one polyline call); feeding stops as dots.
        int n = _path.Count;
        if (n >= 2)
        {
            if (_pathPoints.Length != n)
            {
                _pathPoints = new Vector2[n];
                _pathColors = new Color[n];
            }

            for (int i = 0; i < n; i++)
            {
                float recent = i / (float)n;
                _pathPoints[i] = Iso(_path[i].Pos);
                _pathColors[i] = new Color(0.4f + 0.6f * recent, 0.7f, 1f, 0.15f + 0.85f * recent);
            }

            DrawPolylineColors(_pathPoints, _pathColors, 1f);
            for (int i = 1; i < n; i++)
            {
                if (_path[i].Feeding == 1 && _path[i - 1].Feeding == 0)
                {
                    DrawCircle(_pathPoints[i], 3f, new Color(1f, 0.4f, 0.8f));
                }
            }
        }

        // Other flies (grey), then the primary: a drop line to the floor, a shadow dot and a heading tick.
        foreach (FlyBody3D other in _arena.FlyBodies)
        {
            if (other == _fly)
            {
                continue;
            }

            Vector3 q = other.GlobalPosition;
            Vector3 f = -other.GlobalTransform.Basis.Z;
            var grey = new Color(0.8f, 0.8f, 0.85f, other.State == Behaviour.State.Feeding ? 1f : 0.7f);
            DrawLine(Iso(new Vector3(q.X, 0, q.Z)), Iso(q), new Color(1f, 1f, 1f, 0.15f), 1f);
            DrawLine(Iso(q), Iso(q + f * 0.8f), grey, 1.5f);
            DrawCircle(Iso(q), 2.5f, grey);
        }

        Vector3 p = _fly.GlobalPosition;
        Vector3 fwd = -_fly.GlobalTransform.Basis.Z;
        DrawLine(Iso(new Vector3(p.X, 0, p.Z)), Iso(p), new Color(1f, 1f, 1f, 0.35f), 1f);
        DrawCircle(Iso(new Vector3(p.X, 0, p.Z)), 2f, new Color(0, 0, 0, 0.6f));
        DrawLine(Iso(p), Iso(p + fwd * 1.2f), new Color(1f, 0.6f, 0.2f), 2f);
        DrawCircle(Iso(p), 3.5f, new Color(1f, 0.6f, 0.2f));
        DrawString(font, new Vector2(8, Size.Y - 6), $"alt {p.Y:F1} m   path {n / 10f:F0} s" + (_arena.FlyBodies.Count > 1 ? $"   {_arena.FlyBodies.Count} flies (● orange = the one you watch)" : string.Empty), HorizontalAlignment.Left, -1, 9, new Color(0.6f, 0.6f, 0.7f));
    }

    private void DrawEllipse(Vector3 center, float radius, Color color)
    {
        const int segments = 20;
        var pts = new Vector2[segments + 1];
        for (int i = 0; i <= segments; i++)
        {
            float a = i * MathF.Tau / segments;
            pts[i] = Iso(center + new Vector3(MathF.Cos(a) * radius, 0, MathF.Sin(a) * radius));
        }

        DrawPolyline(pts, color, 1.2f);
    }

    private void DrawBox(Vector3 center, Vector3 size, Color color)
    {
        Vector3 e = size / 2;
        Vector3[] c =
        [
            center + new Vector3(-e.X, -e.Y, -e.Z), center + new Vector3(e.X, -e.Y, -e.Z), center + new Vector3(e.X, -e.Y, e.Z), center + new Vector3(-e.X, -e.Y, e.Z),
            center + new Vector3(-e.X, e.Y, -e.Z), center + new Vector3(e.X, e.Y, -e.Z), center + new Vector3(e.X, e.Y, e.Z), center + new Vector3(-e.X, e.Y, e.Z),
        ];
        DrawColoredPolygon([Iso(c[4]), Iso(c[5]), Iso(c[6]), Iso(c[7])], new Color(color, 0.55f));
        DrawColoredPolygon([Iso(c[3]), Iso(c[2]), Iso(c[6]), Iso(c[7])], new Color(color * 0.7f, 0.55f));   // +Z face
        DrawColoredPolygon([Iso(c[1]), Iso(c[2]), Iso(c[6]), Iso(c[5])], new Color(color * 0.5f, 0.55f));   // +X face
    }
}
