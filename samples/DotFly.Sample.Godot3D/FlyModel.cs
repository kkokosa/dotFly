using System;
using Godot;

namespace DotFly.Sample.Godot3D;

/// <summary>
/// A part-based fruit fly built from primitives so it can be animated: red compound eyes,
/// antennae, a tan thorax, a striped abdomen, six two-segment legs, veined wings, halteres and a
/// two-segment proboscis. Poses: flying (legs tucked, wings beating), landed (legs down, wings
/// folded over the abdomen), feeding (head dips, proboscis extends and pumps, abdomen pulses).
/// Forward is −Z, body length ≈ 0.9 m. Visual only — nothing here feeds the network.
/// </summary>
public partial class FlyModel : Node3D
{
    private readonly Node3D[] _femur = new Node3D[6];
    private readonly Node3D[] _knee = new Node3D[6];
    private Node3D _wingL = null!;
    private Node3D _wingR = null!;
    private Node3D _head = null!;
    private Node3D _proboscis = null!;
    private Node3D _haustellum = null!;
    private MeshInstance3D _labellum = null!;
    private MeshInstance3D _abdomen = null!;
    private float _landed;       // 0 flying … 1 landed (smoothed)
    private float _feeding;      // 0 … 1 (smoothed)
    private float _wingPhase;
    private float _pumpPhase;

    public override void _Ready()
    {
        var cuticle = new StandardMaterial3D { AlbedoColor = new Color(0.72f, 0.5f, 0.26f), Roughness = 0.6f };
        var dark = new StandardMaterial3D { AlbedoColor = new Color(0.18f, 0.12f, 0.08f), Roughness = 0.7f };
        var eyeMat = new StandardMaterial3D { AlbedoColor = new Color(0.75f, 0.08f, 0.05f), Roughness = 0.35f, EmissionEnabled = true, Emission = new Color(0.25f, 0.02f, 0.01f) };
        var abdomenMat = new StandardMaterial3D { AlbedoTexture = ProceduralTextures.Abdomen(), Roughness = 0.55f };
        var wingMat = new StandardMaterial3D { AlbedoTexture = ProceduralTextures.Wing(), Transparency = BaseMaterial3D.TransparencyEnum.Alpha, CullMode = BaseMaterial3D.CullModeEnum.Disabled, Roughness = 0.2f, Metallic = 0.3f };

        // Thorax and abdomen.
        AddChild(new MeshInstance3D { Mesh = new SphereMesh { Radius = 0.2f, Height = 0.4f }, MaterialOverride = cuticle, Position = new Vector3(0, 0.02f, -0.05f), Scale = new Vector3(1f, 0.95f, 1.15f) });
        _abdomen = new MeshInstance3D { Mesh = new SphereMesh { Radius = 0.2f, Height = 0.4f }, MaterialOverride = abdomenMat, Position = new Vector3(0, -0.02f, 0.32f), Scale = new Vector3(0.9f, 1.55f, 0.8f), RotationDegrees = new Vector3(-90, 0, 0) };   // poles along the body: stripes run around it
        AddChild(_abdomen);

        // Head with compound eyes, antennae and the proboscis.
        _head = new Node3D { Position = new Vector3(0, 0.05f, -0.3f) };
        AddChild(_head);
        _head.AddChild(new MeshInstance3D { Mesh = new SphereMesh { Radius = 0.13f, Height = 0.26f }, MaterialOverride = cuticle, Position = new Vector3(0, 0, -0.05f), Scale = new Vector3(1.1f, 1f, 0.9f) });
        foreach (float s in new[] { -1f, 1f })
        {
            _head.AddChild(new MeshInstance3D { Mesh = new SphereMesh { Radius = 0.085f, Height = 0.17f }, MaterialOverride = eyeMat, Position = new Vector3(s * 0.1f, 0.01f, -0.08f), Scale = new Vector3(0.9f, 1.1f, 1f) });
            var antenna = new MeshInstance3D { Mesh = new CylinderMesh { TopRadius = 0.006f, BottomRadius = 0.012f, Height = 0.1f }, MaterialOverride = dark, Position = new Vector3(s * 0.03f, -0.01f, -0.19f), RotationDegrees = new Vector3(100, 0, s * 25) };
            _head.AddChild(antenna);
        }

        _proboscis = new Node3D { Position = new Vector3(0, -0.09f, -0.1f) };
        _head.AddChild(_proboscis);
        _proboscis.AddChild(new MeshInstance3D { Mesh = new CylinderMesh { TopRadius = 0.04f, BottomRadius = 0.035f, Height = 0.17f }, MaterialOverride = cuticle, Position = new Vector3(0, -0.085f, 0) });
        _haustellum = new Node3D { Position = new Vector3(0, -0.17f, 0) };
        _proboscis.AddChild(_haustellum);
        _haustellum.AddChild(new MeshInstance3D { Mesh = new CylinderMesh { TopRadius = 0.035f, BottomRadius = 0.03f, Height = 0.17f }, MaterialOverride = cuticle, Position = new Vector3(0, -0.085f, 0) });
        _labellum = new MeshInstance3D { Mesh = new SphereMesh { Radius = 0.055f, Height = 0.11f }, MaterialOverride = new StandardMaterial3D { AlbedoColor = new Color(0.85f, 0.45f, 0.4f), Roughness = 0.4f }, Position = new Vector3(0, -0.18f, 0), Scale = new Vector3(1.4f, 0.5f, 1.2f) };
        _haustellum.AddChild(_labellum);

        // Six legs: coxa pivot → femur → knee pivot → tibia + tarsus.
        (float x, float y, float z)[] coxae = [(0.12f, -0.08f, -0.2f), (0.16f, -0.1f, -0.03f), (0.14f, -0.1f, 0.14f)];
        for (int i = 0; i < 6; i++)
        {
            float side = i < 3 ? -1f : 1f;
            (float x, float y, float z) = coxae[i % 3];
            var femur = new Node3D { Position = new Vector3(side * x, y, z) };
            femur.AddChild(new MeshInstance3D { Mesh = new CylinderMesh { TopRadius = 0.02f, BottomRadius = 0.015f, Height = 0.28f }, MaterialOverride = cuticle, Position = new Vector3(0, -0.14f, 0) });
            var knee = new Node3D { Position = new Vector3(0, -0.28f, 0) };
            knee.AddChild(new MeshInstance3D { Mesh = new CylinderMesh { TopRadius = 0.014f, BottomRadius = 0.008f, Height = 0.34f }, MaterialOverride = dark, Position = new Vector3(0, -0.17f, 0) });
            femur.AddChild(knee);
            AddChild(femur);
            _femur[i] = femur;
            _knee[i] = knee;
        }

        // Wings (root pivots on the thorax) and halteres behind them.
        _wingL = MakeWing(-1f, wingMat);
        _wingR = MakeWing(1f, wingMat);
        foreach (float s in new[] { -1f, 1f })
        {
            var haltere = new Node3D { Position = new Vector3(s * 0.15f, 0.08f, 0.12f) };
            haltere.AddChild(new MeshInstance3D { Mesh = new CylinderMesh { TopRadius = 0.006f, BottomRadius = 0.006f, Height = 0.1f }, MaterialOverride = cuticle, Position = new Vector3(s * 0.05f, 0, 0.02f), RotationDegrees = new Vector3(0, 0, s * 75) });
            haltere.AddChild(new MeshInstance3D { Mesh = new SphereMesh { Radius = 0.018f, Height = 0.036f }, MaterialOverride = cuticle, Position = new Vector3(s * 0.1f, 0.01f, 0.04f) });
            AddChild(haltere);
        }

        MakeInvisibleToEyes(this);
    }

    private Node3D MakeWing(float side, Material material)
    {
        var root = new Node3D { Position = new Vector3(side * 0.12f, 0.18f, -0.02f) };
        var mesh = new MeshInstance3D
        {
            Mesh = new PlaneMesh { Size = new Vector2(0.78f, 0.3f) },
            MaterialOverride = material,
            Position = new Vector3(side * 0.4f, 0, 0.04f),
            RotationDegrees = new Vector3(0, side > 0 ? 0 : 180, 0),   // the vein texture's root is at u = 0
        };
        root.AddChild(mesh);
        AddChild(root);
        return root;
    }

    /// <summary>Puts every mesh on the fly's own visual layer (her eyes do not draw it); shadows stay on.</summary>
    public static void MakeInvisibleToEyes(Node node)
    {
        if (node is MeshInstance3D mi)
        {
            mi.Layers = Arena.FlyLayer;
        }

        foreach (Node child in node.GetChildren())
        {
            MakeInvisibleToEyes(child);
        }
    }

    /// <summary>Advances the animation. <paramref name="feedingPump"/> is 0..1 (e.g. MN9 rate / 60 Hz) and drives the pumping speed.</summary>
    public void Animate(float dt, Behaviour.State state, float feedingPump)
    {
        bool landed = state is Behaviour.State.Landed or Behaviour.State.Feeding;
        bool feeding = state == Behaviour.State.Feeding;
        _landed += ((landed ? 1f : 0f) - _landed) * MathF.Min(1f, dt * 6f);
        _feeding += ((feeding ? 1f : 0f) - _feeding) * MathF.Min(1f, dt * 3f);

        // Legs: tucked back in flight, spread and planted when landed.
        for (int i = 0; i < 6; i++)
        {
            float side = i < 3 ? -1f : 1f;
            int pair = i % 3;
            float flyRoll = side * 0.35f, flyPitch = 1.1f, flyKnee = -2.6f;
            float landRoll = side * (0.9f + pair * 0.1f), landPitch = (pair - 1) * 0.6f, landKnee = -1.9f + pair * 0.1f;
            _femur[i].Rotation = new Vector3(Mathf.Lerp(flyPitch, landPitch, _landed), 0, Mathf.Lerp(flyRoll, landRoll, _landed));
            _knee[i].Rotation = new Vector3(0, 0, Mathf.Lerp(flyKnee, landKnee, _landed) * side * -1f);
        }

        // Wings: beat in flight (aliased on purpose — it reads as a blur), fold back when landed.
        _wingPhase += dt * 42f;
        float flap = MathF.Sin(_wingPhase) * 0.7f * (1f - _landed);
        float fold = 1.35f * _landed;
        _wingL.Rotation = new Vector3(0, fold, flap + 0.1f * _landed);
        _wingR.Rotation = new Vector3(0, -fold, -flap - 0.1f * _landed);

        // Feeding: head dips, proboscis extends toward the surface and pumps; abdomen swells.
        _pumpPhase += dt * (2.5f + 4f * feedingPump) * _feeding;
        float pump = 0.5f + 0.5f * MathF.Sin(_pumpPhase);
        _head.Rotation = new Vector3(-0.6f * _feeding, 0, 0);
        Rotation = new Vector3(-0.2f * _feeding, 0, 0);   // the whole body tips forward onto the food
        _proboscis.Rotation = new Vector3(Mathf.Lerp(-1.4f, -0.35f - 0.1f * pump, _feeding), 0, 0);
        _proboscis.Scale = new Vector3(1, Mathf.Lerp(0.3f, 1f, _feeding), 1);
        _haustellum.Rotation = new Vector3(Mathf.Lerp(2.6f, 0.6f - 0.2f * pump, _feeding), 0, 0);
        _labellum.Scale = new Vector3(1.4f + 0.3f * pump * _feeding, 0.5f, 1.2f + 0.3f * pump * _feeding);
        _abdomen.Scale = new Vector3(0.9f + 0.05f * _feeding, 1.55f + 0.05f * _feeding, 0.8f + 0.06f * _feeding * pump);
    }
}
