using System;
using Godot;

namespace DotFly.Sample.Godot3D;

/// <summary>
/// Time graphs of every readout group (last ~10 s) in three panels — escape, optic flow, smell &amp;
/// taste — each wired to the decision it feeds, with the decoded actions as live bars. The arrows
/// are the decoder's rules, literally.
/// </summary>
public partial class TraceView : Control
{
    private const int History = 300;
    private FlyBrain3D? _brain;
    private Behaviour? _behaviour;
    private readonly float[,] _trace = new float[FlyBrain3D.GroupNames.Length, History];
    private int _head;
    private double _accum;
    private (float Yaw, float Climb, float Speed) _actions;

    /// <summary>Binds the view.</summary>
    public void Bind(FlyBrain3D brain, Behaviour behaviour)
    {
        _brain = brain;
        _behaviour = behaviour;
    }

    private float _bodyLength = 0.03f;

    /// <summary>Latest actions to display (climb and speed in m/s; shown in body lengths per second).</summary>
    public void SetActions(float yaw, float climb, float speed, float bodyLength)
    {
        _actions = (yaw, climb, speed);
        _bodyLength = MathF.Max(1e-3f, bodyLength);
    }

    public override void _Process(double delta)
    {
        if (_brain is null)
        {
            return;
        }

        _accum += delta;
        if (_accum >= 1.0 / 30)
        {
            _accum = 0;
            for (int g = 0; g < FlyBrain3D.GroupNames.Length; g++)
            {
                _trace[g, _head] = _brain.Rates[g];
            }

            _head = (_head + 1) % History;
        }

        QueueRedraw();
    }

    public override void _Draw()
    {
        DrawRect(new Rect2(0, 0, Size), new Color(0.05f, 0.05f, 0.07f, 0.9f));
        Font font = ThemeDB.FallbackFont;
        if (_brain is null || _behaviour is null)
        {
            return;
        }

        float graphW = Size.X - 190;
        float panelH = (Size.Y - 8) / 3f;
        Panel(font, 0, panelH, graphW, "escape — LC4 → DNp04, GF", [0, 1, 2], ["DNp04_L", "DNp04_R", "GF"], 350f,
            $"yaw += 1.2·(DNp04_L − DNp04_R)\nspeed = 120 + 1.0·GF BL/s");
        Panel(font, panelH, panelH, graphW, "optic flow — T4 → HS, H2, VS", [3, 4, 7, 8, 5, 6], ["HS_L", "HS_R", "H2_L", "H2_R", "VS_L", "VS_R"], 350f,
            $"yaw += 0.2·(HS_R+H2_L − HS_L−H2_R) − 0.3·(HS_R − HS_L)\nclimb = 0.07·VS + random drift | ±12 BL/s in odor");
        Panel(font, 2 * panelH, panelH, graphW, "smell/taste/wind — ORN→PN · GRN→MN9 · JO→AMMC", [9, 10, 11, 12], ["PN", "MN9", "wind_L", "wind_R"], 300f,
            $"PN > 110: surge upwind (wind_L−wind_R), tumble on fall\nMN9 (0.3 s) > 15 Hz: feed");

        // Decision bars.
        float bx = graphW + 12;
        DrawString(font, new Vector2(bx, 22), "decisions", HorizontalAlignment.Left, -1, 18, new Color(0.8f, 0.8f, 0.85f));
        Bar(font, bx, 30, "yaw °/s", _actions.Yaw, 240f, new Color(1f, 0.6f, 0.3f), true);
        Bar(font, bx, 70, "climb BL/s", _actions.Climb / _bodyLength, 50f, new Color(0.5f, 0.9f, 0.6f), true);
        Bar(font, bx, 110, "speed BL/s", _actions.Speed / _bodyLength, 200f, new Color(0.9f, 0.9f, 0.4f), false);
        DrawString(font, new Vector2(bx, 158), $"state: {_behaviour.Current}", HorizontalAlignment.Left, -1, 18, new Color(1f, 1f, 1f));
        DrawString(font, new Vector2(bx, 178), _behaviour.YawSource, HorizontalAlignment.Left, -1, 16, new Color(0.7f, 0.7f, 0.8f));
        DrawString(font, new Vector2(bx, 200), $"odor {(_behaviour.OdorDetected ? "DETECTED" : _behaviour.OdorMemory ? "lost, casting" : "—")} {(_behaviour.OdorTrend >= 0 ? "↑" : "↓")}{MathF.Abs(_behaviour.OdorTrend):F0}", HorizontalAlignment.Left, -1, 16, new Color(0.9f, 0.5f, 1f));
        DrawString(font, new Vector2(bx, 220), $"fed {_behaviour.FeedTime:F1} s", HorizontalAlignment.Left, -1, 16, new Color(1f, 0.4f, 0.8f));
    }

    private void Panel(Font font, float y, float h, float w, string title, int[] groups, string[] shortNames, float maxHz, string rule)
    {
        DrawString(font, new Vector2(6, y + 18), title, HorizontalAlignment.Left, -1, 16, new Color(0.8f, 0.8f, 0.85f));
        var area = new Rect2(6, y + 24, w - 12, h - 32);
        DrawString(font, new Vector2(area.End.X - 94, area.End.Y - 5), $"0–{maxHz:F0} Hz · 10 s", HorizontalAlignment.Right, 90, 11, new Color(0.5f, 0.5f, 0.55f));
        DrawRect(area, new Color(0.1f, 0.1f, 0.13f));
        DrawLine(area.Position + new Vector2(0, area.Size.Y), area.End, new Color(0.3f, 0.3f, 0.35f));
        foreach (int g in groups)
        {
            var pts = new Vector2[History];
            for (int k = 0; k < History; k++)
            {
                float v = _trace[g, (_head + k) % History];
                pts[k] = new Vector2(area.Position.X + k * area.Size.X / (History - 1), area.End.Y - MathF.Min(1f, v / maxHz) * area.Size.Y);
            }

            DrawPolyline(pts, BrainMapView.GroupColors[g], 1.2f, true);
        }

        // Legend (current rates) along the top edge; the decoder rule bottom-left.
        float lx = area.Position.X + 4;
        float ly = area.Position.Y + 13;
        for (int i = 0; i < groups.Length; i++)
        {
            int g = groups[i];
            string txt = $"{shortNames[i]} {_brain!.Rates[g]:F0}";
            float tw = font.GetStringSize(txt, HorizontalAlignment.Left, -1, 13).X;
            if (lx + tw > area.End.X - 4)
            {
                lx = area.Position.X + 4;   // wrap the legend
                ly += 13;
            }

            DrawString(font, new Vector2(lx, ly), txt, HorizontalAlignment.Left, -1, 13, BrainMapView.GroupColors[g]);
            lx += tw + 9;
        }

        DrawMultilineString(font, new Vector2(area.Position.X + 4, area.End.Y - 19), rule, HorizontalAlignment.Left, area.Size.X - 8, 12, 2, new Color(0.95f, 0.95f, 0.7f));
    }

    private void Bar(Font font, float x, float y, string label, float value, float max, Color color, bool signed)
    {
        DrawString(font, new Vector2(x, y + 14), label, HorizontalAlignment.Left, -1, 15, new Color(0.7f, 0.7f, 0.8f));
        var track = new Rect2(x, y + 20, 120, 12);
        DrawRect(track, new Color(0.15f, 0.15f, 0.2f));
        float frac = Mathf.Clamp(value / max, -1f, 1f);
        if (signed)
        {
            float mid = track.Position.X + track.Size.X / 2;
            DrawLine(new Vector2(mid, track.Position.Y), new Vector2(mid, track.End.Y), new Color(0.4f, 0.4f, 0.5f));
            float wdt = frac * track.Size.X / 2;
            DrawRect(new Rect2(MathF.Min(mid, mid + wdt), track.Position.Y, MathF.Abs(wdt), track.Size.Y), color);
        }
        else
        {
            DrawRect(new Rect2(track.Position, new Vector2(MathF.Max(0, frac) * track.Size.X, track.Size.Y)), color);
        }

        DrawString(font, new Vector2(x + 126, y + 31), value.ToString("F0"), HorizontalAlignment.Left, -1, 15, color);
    }
}
