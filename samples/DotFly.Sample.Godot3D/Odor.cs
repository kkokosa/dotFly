using System;
using System.Collections.Generic;
using Godot;

namespace DotFly.Sample.Godot3D;

/// <summary>
/// Sugar patches and the odor field they emit. Concentration is a sum of narrow tubes stretched
/// downwind (exponential decay along the wind, a Gaussian across a centreline that meanders like
/// a smoke ribbon, a short Gaussian upwind) that hug the surface the patch sits on and thin out
/// with height; the wind slowly wanders. Purely the world's physics — nothing neural. The same function is evaluated by <c>odor_raymarch.gdshader</c>
/// for the volumetric rendering, so what is drawn is exactly what the antennae sample.
/// </summary>
public sealed class Odor
{
    /// <summary>A sugar patch.</summary>
    /// <param name="Position">Centre; <c>Y</c> is the surface height the patch sits on.</param>
    /// <param name="Radius">Contact radius in metres.</param>
    public sealed record Patch(Vector3 Position, float Radius);

    private readonly List<Patch> _patches = [];
    private double _t;

    /// <summary>The patches.</summary>
    public IReadOnlyList<Patch> Patches => _patches;

    /// <summary>Current wind direction (unit, horizontal).</summary>
    public Vector3 Wind { get; private set; } = new(1, 0, 0);

    /// <summary>Plume length scale in metres.</summary>
    public float PlumeLength { get; set; } = 18f;

    /// <summary>Plume width scale in metres (at the source; it widens slowly downwind).</summary>
    public float PlumeWidth { get; set; } = 0.5f;

    /// <summary>Cone growth: the width (lateral and vertical) is multiplied by 1 + along / ConeScale — a narrow tube at the sugar, a broad stream 15 m downwind.</summary>
    public float ConeScale { get; set; } = 2.5f;

    /// <summary>Meander amplitude in metres (lateral swing of the plume centreline).</summary>
    public float MeanderAmplitude { get; set; } = 1.3f;

    /// <summary>Meander wavelength in metres along the plume.</summary>
    public float MeanderWavelength { get; set; } = 6f;

    /// <summary>Meander period in seconds (the ribbon drifts sideways over time).</summary>
    public float MeanderPeriod { get; set; } = 60f;

    /// <summary>Vertical meander amplitude in metres (the tunnel rises and dips).</summary>
    public float VerticalMeanderAmplitude { get; set; } = 0.55f;

    /// <summary>Vertical meander wavelength in metres along the tunnel.</summary>
    public float VerticalMeanderWavelength { get; set; } = 7.5f;

    /// <summary>Buoyant rise of the tunnel's core per metre downwind.</summary>
    public float RisePerMetre { get; set; } = 0.28f;   // warm sugar air is buoyant: the tunnel climbs to the ceiling within ~15 m

    /// <summary>Ceiling height; the core stays 0.5 m below it.</summary>
    public float RoomHeight { get; set; } = 1000f;

    /// <summary>Elapsed time (drives the wind and the meander; shared with the renderer).</summary>
    public float Time => (float)_t;

    /// <summary>Room half-extents (X, Z): plumes stop at the walls and fade within <see cref="WallMargin"/> of them.</summary>
    public Vector2 RoomHalfSize { get; set; } = new(1000, 1000);

    /// <summary>Distance from a wall over which the odor fades to zero (no odor gathers in corners).</summary>
    public float WallMargin { get; set; } = 0.8f;

    /// <summary>A ribbon ends this far before its axis meets a wall.</summary>
    public float EndMargin { get; set; } = 1.5f;

    /// <summary>Seconds per full turn of the wind direction.</summary>
    public float WindPeriod { get; set; } = 900f;   // slow: a plume that sweeps faster than the fly tracks is lost

    /// <summary>Height of the ribbon's core above the patch surface (metres).</summary>
    public float CoreHeight { get; set; } = 0.45f;

    /// <summary>Vertical Gaussian half-width of the ribbon (metres) — with the lateral width this makes it a tube.</summary>
    public float VerticalWidth { get; set; } = 0.35f;

    /// <summary>Adds a patch.</summary>
    public void Add(Vector3 position, float radius) => _patches.Add(new Patch(position, radius));

    /// <summary>Advances the wind.</summary>
    public void Update(double delta)
    {
        _t += delta;
        // A slow full rotation (one turn per WindPeriod) with a wobble: over minutes the ribbons
        // sweep every direction instead of always pointing at one wall.
        float a = (float)(Math.Tau * _t / WindPeriod + 0.5 * Math.Sin(_t * 0.05) + 0.3 * Math.Sin(_t * 0.023));
        Wind = new Vector3(MathF.Cos(a), 0, MathF.Sin(a));
    }

    /// <summary>Odor concentration (0..1+) at a point.</summary>
    public float Concentration(Vector3 p)
    {
        float c = 0;
        foreach (Patch patch in _patches)
        {
            c += Concentration(p, patch);
        }

        return c * WallFactor(p);
    }

    /// <summary>0 at a wall → 1 at <see cref="WallMargin"/> from every wall.</summary>
    public float WallFactor(Vector3 p)
    {
        float dx = RoomHalfSize.X - MathF.Abs(p.X);
        float dz = RoomHalfSize.Y - MathF.Abs(p.Z);
        float d = MathF.Min(dx, dz) / WallMargin;
        d = Math.Clamp(d, 0f, 1f);
        return d * d * (3f - 2f * d);
    }

    /// <summary>Downwind distance from the patch at which the plume's axis meets a wall.</summary>
    public float AlongToWall(Patch patch)
    {
        float tx = Wind.X > 1e-4f ? (RoomHalfSize.X - patch.Position.X) / Wind.X : Wind.X < -1e-4f ? (-RoomHalfSize.X - patch.Position.X) / Wind.X : float.MaxValue;
        float tz = Wind.Z > 1e-4f ? (RoomHalfSize.Y - patch.Position.Z) / Wind.Z : Wind.Z < -1e-4f ? (-RoomHalfSize.Y - patch.Position.Z) / Wind.Z : float.MaxValue;
        return MathF.Min(tx, tz);
    }

    /// <summary>Odor concentration from one patch (mirrored in <c>odor_raymarch.gdshader</c>).</summary>
    public float Concentration(Vector3 p, Patch patch)
    {
        Vector3 d = p - patch.Position;
        float height = d.Y;
        d.Y = 0;
        float along = d.Dot(Wind);                              // positive = downwind of the patch
        var perp = new Vector3(-Wind.Z, 0, Wind.X);
        float across = d.Dot(perp);                             // signed lateral offset
        // The centreline meanders: two sinusoids along the plume, drifting with time, growing
        // from zero at the source (each patch gets its own phase from its position).
        float phase = patch.Position.X * 0.7f + patch.Position.Z * 1.3f;
        float a = MathF.Max(0f, along);
        float grow = MathF.Min(1f, a / 3f);
        float centre = MeanderAmplitude * grow * (MathF.Sin(MathF.Tau * (a / MeanderWavelength - Time / MeanderPeriod) + phase)
                                                  + 0.45f * MathF.Sin(MathF.Tau * (a / (MeanderWavelength * 0.37f) + Time / (MeanderPeriod * 0.61f)) + 2.1f * phase));
        float off = across - centre;
        float coreY = CoreY(patch, a, phase);
        // Downwind: exponential decay along the plume (a gradient to climb); upwind: short Gaussian.
        float alongTerm = along > 0 ? -along / PlumeLength : -(along * along) / (2 * 1.2f * 1.2f);
        float width = PlumeWidth * (1f + a / ConeScale);                       // a cone: narrow at the sugar, broad downwind
        float dh = height - coreY;
        float vw = VerticalWidth * (1f + a / ConeScale);                       // and in height too
        float vertical = -(dh * dh) / (2 * vw * vw) - MathF.Max(0f, -height) * 4f;   // a tube, not a sheet
        float wallCut = -MathF.Max(0f, a - AlongToWall(patch) + EndMargin) * 2.5f;   // the ribbon ends before the wall
        // Dilution: as the cone widens the odor spreads over it (a mild √ of the cross-section, so the
        // far field stays detectable): "near" (PN > 270 Hz) is then reached only within ~1–2 m of the sugar.
        return MathF.Exp(alongTerm - (off * off) / (2 * width * width) + vertical + wallCut) / MathF.Sqrt(1f + a / ConeScale);
    }

    /// <summary>Height of the tunnel core above the patch surface, <paramref name="a"/> metres downwind: base height, buoyant rise, vertical meander, capped below the ceiling.</summary>
    private float CoreY(Patch patch, float a, float phase)
    {
        float grow = MathF.Min(1f, a / 3f);
        float y = CoreHeight + RisePerMetre * a + VerticalMeanderAmplitude * grow * MathF.Sin(MathF.Tau * (a / VerticalMeanderWavelength + Time / (MeanderPeriod * 0.8f)) + 3.7f * phase);
        return MathF.Min(y, RoomHeight - 0.35f - patch.Position.Y);   // then spreads along the ceiling
    }

    /// <summary>A point on a plume's meandering centreline, <paramref name="along"/> metres downwind (for the map).</summary>
    public Vector3 Centreline(Patch patch, float along)
    {
        var perp = new Vector3(-Wind.Z, 0, Wind.X);
        float phase = patch.Position.X * 0.7f + patch.Position.Z * 1.3f;
        float a = MathF.Max(0f, along);
        float grow = MathF.Min(1f, a / 3f);
        float centre = MeanderAmplitude * grow * (MathF.Sin(MathF.Tau * (a / MeanderWavelength - Time / MeanderPeriod) + phase)
                                                  + 0.45f * MathF.Sin(MathF.Tau * (a / (MeanderWavelength * 0.37f) + Time / (MeanderPeriod * 0.61f)) + 2.1f * phase));
        return patch.Position + Wind * along + perp * centre + Vector3.Up * CoreY(patch, a, phase);
    }

    /// <summary>The patch under a point (horizontal distance ≤ radius), if any.</summary>
    public Patch? PatchUnder(Vector3 p)
    {
        foreach (Patch patch in _patches)
        {
            Vector3 d = p - patch.Position;
            d.Y = 0;
            if (d.Length() <= patch.Radius)
            {
                return patch;
            }
        }

        return null;
    }
}
