using System;
using Godot;

namespace DotFly.Sample.Godot3D;

/// <summary>
/// Wing beat for a loaded fly model whose wings are separate hinge nodes (<c>Fly_Wing_L</c> /
/// <c>Fly_Wing_R</c>, resting folded back along −Z). The kinematics follow the high-speed
/// reconstructions of Drosophila flight (Bergou, Ristroph, Cohen &amp; Wang; physics.aps.org
/// story v25/st13): the wings row <b>forward and back in a near-horizontal stroke plane</b> through
/// ~140°, and <b>flip their pitch at each stroke reversal</b> — leading edge tilted forward on the
/// forward stroke, backward on the return, like feathering an oar. A real fly beats at ~200 Hz;
/// here the beat runs at a rate the eye can follow, with two short trailing ghosts of each wing
/// for the blur. On the ground the wings fold back. Visual only.
/// </summary>
public sealed class BlurredWings
{
    private const float RestSweepDeg = 54f;      // the folded wing points back-out; +54° puts it straight out sideways
    private const float AmplitudeDeg = 70f;      // ±70° about that: a 140° stroke
    private const float PitchRad = 0.8f;         // ±45° pitch flip
    private const float BeatHz = 14f;            // visible beat rate (a real fly: ~200 Hz)
    private static readonly float[] GhostLag = [0.012f, 0.024f];   // seconds behind, for the blur

    private readonly MeshInstance3D[] _wings;
    private readonly MeshInstance3D[][] _ghosts;
    private readonly Vector3[] _longAxis;        // the wing's own axis, hinge → tip, in the hinge frame
    private readonly float[] _sign;              // +1: the left wing sweeps forward with a +Y rotation; −1 right
    private float _time;
    private float _flight;                       // 0 grounded … 1 flying (smoothed)

    private BlurredWings(MeshInstance3D[] wings, MeshInstance3D[][] ghosts, Vector3[] longAxis, float[] sign)
    {
        _wings = wings;
        _ghosts = ghosts;
        _longAxis = longAxis;
        _sign = sign;
    }

    /// <summary>Finds the wing nodes under <paramref name="root"/>; null when the model has none.</summary>
    public static BlurredWings? TryCreate(Node root)
    {
        var left = root.FindChild("Fly_Wing_L", true, false) as MeshInstance3D;
        var right = root.FindChild("Fly_Wing_R", true, false) as MeshInstance3D;
        if (left is null || right is null)
        {
            return null;
        }

        MeshInstance3D[] wings = [left, right];
        float[] sign = [1f, -1f];
        var ghosts = new MeshInstance3D[2][];
        var axes = new Vector3[2];
        for (int w = 0; w < 2; w++)
        {
            MeshInstance3D wing = wings[w];
            Node parent = wing.GetParent();

            // The wing's long axis from its mesh bounds: from the hinge (the node origin) to the
            // far corner of the blade, in the horizontal plane.
            Aabb b = wing.GetAabb();
            float tipX = MathF.Abs(b.Position.X) > MathF.Abs(b.End.X) ? b.Position.X : b.End.X;
            float tipZ = MathF.Abs(b.Position.Z) > MathF.Abs(b.End.Z) ? b.Position.Z : b.End.Z;
            axes[w] = new Vector3(tipX, 0, tipZ).Normalized();

            Material? ghostMat = null;
            if (wing.GetActiveMaterial(0) is StandardMaterial3D std)
            {
                var m = (StandardMaterial3D)std.Duplicate();
                m.Transparency = BaseMaterial3D.TransparencyEnum.Alpha;
                m.AlbedoColor = new Color(m.AlbedoColor.R, m.AlbedoColor.G, m.AlbedoColor.B, 0.28f);
                m.CullMode = BaseMaterial3D.CullModeEnum.Disabled;
                m.ShadingMode = BaseMaterial3D.ShadingModeEnum.Unshaded;
                ghostMat = m;
            }

            ghosts[w] = new MeshInstance3D[GhostLag.Length];
            for (int g = 0; g < GhostLag.Length; g++)
            {
                var ghost = (MeshInstance3D)wing.Duplicate();
                ghost.Name = $"{wing.Name}_trail{g}";
                ghost.MaterialOverride = ghostMat;
                ghost.CastShadow = GeometryInstance3D.ShadowCastingSetting.Off;
                parent.AddChild(ghost);
                ghosts[w][g] = ghost;
            }
        }

        return new BlurredWings(wings, ghosts, axes, sign);
    }

    /// <summary>Advances the beat; <paramref name="flying"/> selects beating vs folded wings.</summary>
    public void Animate(float dt, bool flying)
    {
        _flight += ((flying ? 1f : 0f) - _flight) * MathF.Min(1f, dt * 6f);
        _time += dt;
        for (int w = 0; w < 2; w++)
        {
            _wings[w].Basis = Pose(w, _time);
            for (int g = 0; g < _ghosts[w].Length; g++)
            {
                MeshInstance3D ghost = _ghosts[w][g];
                ghost.Visible = _flight > 0.5f;
                ghost.Basis = Pose(w, _time - GhostLag[g]);
            }
        }
    }

    /// <summary>
    /// Wing pose at time <paramref name="t"/>: sweep about the hinge's vertical axis (fore–aft
    /// rowing), then pitch about the wing's own long axis, flipping with the stroke direction.
    /// Blended with the folded rest pose by the flight factor.
    /// </summary>
    private Basis Pose(int w, float t)
    {
        float phase = MathF.Tau * BeatHz * t;
        float sweepDeg = _flight * (RestSweepDeg + AmplitudeDeg * MathF.Sin(phase));   // 0 = folded back
        float pitch = _flight * PitchRad * MathF.Cos(phase);                            // ∝ stroke velocity: flips at reversal
        var sweep = new Basis(Vector3.Up, _sign[w] * Mathf.DegToRad(sweepDeg));
        var feather = new Basis(_longAxis[w], -_sign[w] * pitch);
        return sweep * feather;   // feather in the wing's own frame first, then row it around the hinge
    }
}
