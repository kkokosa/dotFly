using System.Collections.Generic;
using Godot;

namespace DotFly.Sample.Godot3D;

/// <summary>
/// Keeps the fly visible: every frame a ray is cast from the fly to the camera, and each static
/// body it passes through (walls, columns, furniture, glass) has its meshes swapped to a translucent
/// copy of their material while they are in the way, restored a moment after they are not.
/// </summary>
public sealed class OcclusionFader
{
    private const float Alpha = 0.3f;
    private const float Grace = 0.25f;   // seconds an object stays faded after it stops occluding (no flicker)
    private const int MaxHits = 6;

    private readonly Dictionary<Material, Material> _translucent = [];          // original → translucent copy (shared materials share copies)
    private readonly Dictionary<MeshInstance3D, (Material? Original, float Timer)> _faded = [];
    private readonly List<MeshInstance3D> _restore = [];

    /// <summary>Objects currently faded (for the HUD).</summary>
    public int Count => _faded.Count;

    /// <summary>Fades what lies between <paramref name="from"/> (the fly) and <paramref name="to"/> (the camera).</summary>
    public void Update(PhysicsDirectSpaceState3D space, Vector3 from, Vector3 to, Rid self, float dt)
    {
        var hitNow = new HashSet<MeshInstance3D>();
        var exclude = new Godot.Collections.Array<Rid> { self };
        for (int i = 0; i < MaxHits; i++)
        {
            var query = PhysicsRayQueryParameters3D.Create(from, to);
            query.Exclude = exclude;
            Godot.Collections.Dictionary hit = space.IntersectRay(query);
            if (hit.Count == 0)
            {
                break;
            }

            exclude.Add(hit["rid"].AsRid());
            if (hit["collider"].AsGodotObject() is Node body)
            {
                foreach (Node child in body.GetChildren())
                {
                    if (child is MeshInstance3D mesh)
                    {
                        hitNow.Add(mesh);
                        Fade(mesh);
                    }
                }
            }
        }

        _restore.Clear();
        foreach ((MeshInstance3D mesh, (Material? original, float timer)) in _faded)
        {
            if (hitNow.Contains(mesh))
            {
                _faded[mesh] = (original, Grace);
            }
            else if (timer - dt <= 0)
            {
                _restore.Add(mesh);
            }
            else
            {
                _faded[mesh] = (original, timer - dt);
            }
        }

        foreach (MeshInstance3D mesh in _restore)
        {
            if (GodotObject.IsInstanceValid(mesh))
            {
                mesh.MaterialOverride = _faded[mesh].Original;
            }

            _faded.Remove(mesh);
        }
    }

    private void Fade(MeshInstance3D mesh)
    {
        if (_faded.ContainsKey(mesh))
        {
            return;
        }

        Material? original = mesh.MaterialOverride ?? mesh.GetActiveMaterial(0);
        if (original is not StandardMaterial3D std)
        {
            return;   // shader materials (the odor) are left alone
        }

        if (!_translucent.TryGetValue(std, out Material? copy))
        {
            var m = (StandardMaterial3D)std.Duplicate();
            m.Transparency = BaseMaterial3D.TransparencyEnum.Alpha;
            m.AlbedoColor = new Color(m.AlbedoColor.R, m.AlbedoColor.G, m.AlbedoColor.B, Alpha);
            m.CullMode = BaseMaterial3D.CullModeEnum.Back;
            copy = m;
            _translucent[std] = copy;
        }

        _faded[mesh] = (mesh.MaterialOverride, Grace);
        mesh.MaterialOverride = copy;
    }
}
