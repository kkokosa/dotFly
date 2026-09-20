using System;
using System.Collections.Generic;
using Godot;

namespace DotFly.Sample.Godot3D;

/// <summary>
/// Builds the room in code: a sunlit living room with windows, curtains, a couch with cushions,
/// two tables, four columns, a rug, lamps, a bookshelf, sugar on the tables and the floor, the
/// odor field (raymarched by <c>odor_raymarch.gdshader</c>), the fly (<see cref="FlyBody3D"/>) and
/// the chase camera. Rendering: Forward+, SDFGI global illumination, screen-space reflections,
/// soft sun shadows, PBR materials with procedural normal maps. The fly's eyes see it as luminance
/// only, which is what the motion/looming pathway uses (R1–R6 are achromatic).
/// </summary>
public partial class Arena : Node3D
{
    /// <summary>Visual layer of things the fly must not see (the odor rendering).</summary>
    public const uint HumanOnlyLayer = 2;

    /// <summary>Visual layer of the fly's own body (its eye cameras do not draw it).</summary>
    public const uint FlyLayer = 4;

    /// <summary>A box-shaped obstacle for the minimap.</summary>
    public readonly record struct Box(Vector3 Center, Vector3 Size, Color Color);

    /// <summary>A column for the minimap.</summary>
    public readonly record struct Column(Vector3 Base, float Radius, float Height);

    /// <summary>Room half-width along X.</summary>
    public float HalfX { get; } = 12f;

    /// <summary>Room half-depth along Z.</summary>
    public float HalfZ { get; } = 9f;

    /// <summary>Ceiling height.</summary>
    public float Height { get; } = 5f;

    /// <summary>The sugar patches and their odor field.</summary>
    public Odor Odor { get; } = new();

    /// <summary>Furniture, for the minimap.</summary>
    public List<Box> Boxes { get; } = [];

    /// <summary>Columns, for the minimap.</summary>
    public List<Column> Columns { get; } = [];

    /// <summary>The chase camera (a child of the room so the fly's rotation does not drag it).</summary>
    public Camera3D Chase { get; private set; } = null!;

    /// <summary>
    /// Rendering quality: <c>high</c> = SDFGI + SSIL + SSR + shadowed lamps, <c>medium</c> = SSR and
    /// sun shadows only (default), <c>low</c> = no screen-space effects. <c>DOTFLY_QUALITY</c> overrides.
    /// </summary>
    [Export]
    public string Quality { get; set; } = System.Environment.GetEnvironmentVariable("DOTFLY_QUALITY") ?? "medium";

    /// <summary>
    /// Number of flies. Each extra fly is a full, independent MaleCNS simulation (its own seed,
    /// inputs, readouts and decoder) sharing the machine: threads are split evenly and pinned to
    /// distinct cores. Only the primary fly has panels, HUD and camera. <c>--flies N</c> after the
    /// Godot arguments' <c>--</c>, or <c>DOTFLY_FLIES</c>, override.
    /// </summary>
    [Export]
    public int Flies { get; set; } = FliesFromArgs();

    /// <summary>All flies (index 0 = primary), for the room map.</summary>
    public List<FlyBody3D> FlyBodies { get; } = [];

    /// <summary>Draw the odor (raymarched). <c>DOTFLY_ODOR_RENDER=0</c> disables it.</summary>
    [Export]
    public bool RenderOdor { get; set; } = System.Environment.GetEnvironmentVariable("DOTFLY_ODOR_RENDER") != "0";

    private bool High => Quality == "high";
    private bool AtLeastMedium => Quality != "low";
    private ShaderMaterial? _odorMaterial;
    private Sky? _sky;
    private Vector3 _sunDir = Vector3.Up;

    /// <summary>
    /// The environment for the fly's eye cameras: same sky and light, no glow, no SSAO — plain
    /// luminance for the retina. The odor rendering is on a layer the eye cameras do not draw.
    /// </summary>
    public Godot.Environment CreateEyeEnvironment() => new()
    {
        BackgroundMode = Godot.Environment.BGMode.Sky,
        Sky = _sky,
        AmbientLightSource = Godot.Environment.AmbientSource.Sky,
        AmbientLightSkyContribution = 0.7f,
        AmbientLightEnergy = 1.0f,
        TonemapMode = Godot.Environment.ToneMapper.Filmic,
        TonemapExposure = 1.3f,
    };

    public override void _Ready()
    {
        Odor.RoomHalfSize = new Vector2(HalfX - 0.15f, HalfZ - 0.15f);   // inner faces of the walls
        Odor.RoomHeight = Height;
        BuildEnvironment();
        BuildRoom();
        BuildFurniture();
        BuildSugar();
        BuildOdorRender();

        Chase = new Camera3D { Position = new Vector3(0, 2.5f, 6), Current = true, Fov = 65, Near = 0.02f };
        AddChild(Chase);

        SpawnFlies();
    }

    /// <summary>The fly count from <c>--flies N</c> / <c>DOTFLY_FLIES</c> (1 when absent).</summary>
    public static int FliesFromArgs()
    {
        string[] args = OS.GetCmdlineUserArgs();
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--flies" && i + 1 < args.Length && int.TryParse(args[i + 1], out int n))
            {
                return Math.Clamp(n, 1, 32);
            }

            if (args[i].StartsWith("--flies=", StringComparison.Ordinal) && int.TryParse(args[i]["--flies=".Length..], out int m))
            {
                return Math.Clamp(m, 1, 32);
            }
        }

        return int.TryParse(System.Environment.GetEnvironmentVariable("DOTFLY_FLIES"), out int e) ? Math.Clamp(e, 1, 32) : 1;
    }

    private void SpawnFlies()
    {
        var primaryBrain = GetNode<FlyBrain3D>("FlyBrain");
        int n = Math.Max(1, Flies);
        int cores = DotFly.Cpu.Fast.CpuBackend.DefaultThreads();
        int threadsEach = Math.Max(1, cores / n);
        if (n > 1)
        {
            // The scene's brain readied itself before this (children first): it read the fly
            // count itself and took the first thread slots.
            GD.Print($"dotFly: {n} flies × {threadsEach} threads ({cores} physical cores); extra flies are full simulations without panels");
        }

        var rng = new Random(11);
        for (int i = 0; i < n; i++)
        {
            FlyBrain3D brain = primaryBrain;
            if (i > 0)
            {
                brain = new FlyBrain3D
                {
                    CheckpointPath = primaryBrain.CheckpointPath,
                    Threads = threadsEach,
                    PinOffset = i * threadsEach,
                    PinTotal = n * threadsEach,
                    Seed = 1 + i,
                    SynapticGain = primaryBrain.SynapticGain,
                    StabilityControl = primaryBrain.StabilityControl,
                    HandlesKeys = false,
                };
                AddChild(brain);
            }

            Vector3 at = i == 0 ? new Vector3(-9f, 1.6f, 5f) : new Vector3((float)(rng.NextDouble() * 2 - 1) * (HalfX - 3), 1.2f + (float)rng.NextDouble() * 2f, (float)(rng.NextDouble() * 2 - 1) * (HalfZ - 3));
            float heading = i == 0 ? Mathf.DegToRad(150) : (float)(rng.NextDouble() * MathF.Tau);
            var fly = new FlyBody3D { BrainNode = brain, Primary = i == 0, Position = at, Rotation = new Vector3(0, heading, 0) };
            FlyBodies.Add(fly);
            AddChild(fly);
        }
    }

    public override void _Process(double delta)
    {
        _odorMaterial?.SetShaderParameter("wind", Odor.Wind);
        _odorMaterial?.SetShaderParameter("time_s", Odor.Time);
        if (FlyBodies.Count > 0)
        {
            _odorMaterial?.SetShaderParameter("clear_distance", (FlyBodies[0].GlobalPosition - Chase.GlobalPosition).Length() * 1.15f);
        }
    }

    private void BuildEnvironment()
    {
        var sky = new PhysicalSkyMaterial { RayleighColor = new Color(0.26f, 0.41f, 0.58f), MieCoefficient = 0.006f, GroundColor = new Color(0.35f, 0.4f, 0.3f), EnergyMultiplier = 1.0f };
        _sky = new Sky { SkyMaterial = sky, ProcessMode = Sky.ProcessModeEnum.Realtime, RadianceSize = Sky.RadianceSizeEnum.Size256 };
        var env = new WorldEnvironment
        {
            Environment = new Godot.Environment
            {
                BackgroundMode = Godot.Environment.BGMode.Sky,
                Sky = _sky,
                AmbientLightSource = Godot.Environment.AmbientSource.Sky,
                AmbientLightSkyContribution = 0.7f,
                AmbientLightEnergy = 1.0f,
                ReflectedLightSource = Godot.Environment.ReflectionSource.Sky,
                TonemapMode = Godot.Environment.ToneMapper.Aces,
                TonemapExposure = 1.35f,
                TonemapWhite = 6f,
                SdfgiEnabled = High,                 // bounced light from the sun patches and the lamps
                SdfgiUseOcclusion = true,
                SdfgiBounceFeedback = 0.5f,
                SdfgiCascades = 4,
                SdfgiMinCellSize = 0.15f,
                SdfgiEnergy = 1.2f,
                SsrEnabled = AtLeastMedium,          // the glossy floor reflects the windows
                SsrMaxSteps = 96,
                SsaoEnabled = AtLeastMedium,
                SsaoRadius = 0.8f,
                SsaoIntensity = 1.5f,
                SsilEnabled = High,
                GlowEnabled = true,
                GlowIntensity = 0.3f,
                GlowBloom = 0.03f,
                GlowHdrThreshold = 1.2f,
                AdjustmentEnabled = true,
                AdjustmentContrast = 1.05f,
                AdjustmentSaturation = 1.05f,
            },
        };
        AddChild(env);

        // Sun through the windows: soft PCSS shadows; the sky's sun follows it.
        var sun = new DirectionalLight3D
        {
            LightEnergy = 4.2f,
            LightColor = new Color(1f, 0.93f, 0.82f),
            ShadowEnabled = System.Environment.GetEnvironmentVariable("DOTFLY_PERF") != "noshadow",
            LightAngularDistance = System.Environment.GetEnvironmentVariable("DOTFLY_PERF") == "nopcss" ? 0f : 0.9f,
            ShadowBlur = 1.2f,
            DirectionalShadowMaxDistance = 50f,
            DirectionalShadowMode = DirectionalLight3D.ShadowMode.Parallel4Splits,
            DirectionalShadowSplit1 = 0.05f,
            DirectionalShadowSplit2 = 0.15f,
            DirectionalShadowSplit3 = 0.4f,
        };
        sun.RotationDegrees = new Vector3(-30, 58, 0);
        AddChild(sun);
        _sunDir = sun.GlobalTransform.Basis.Z;   // a light shines along −Z; +Z points toward the sun
    }

    private void BuildRoom()
    {
        StandardMaterial3D wood = ProceduralTextures.Wood();
        StandardMaterial3D plaster = ProceduralTextures.Plaster(new Color(0.9f, 0.87f, 0.8f));
        var ceilingMat = new StandardMaterial3D { AlbedoColor = new Color(0.96f, 0.96f, 0.94f), Roughness = 0.95f };
        var outside = new StandardMaterial3D { AlbedoColor = new Color(0.28f, 0.45f, 0.22f), Roughness = 1f };
        var trim = new StandardMaterial3D { AlbedoColor = new Color(0.97f, 0.96f, 0.93f), Roughness = 0.35f };

        // Floor, ceiling and the ground outside (seen through the windows).
        AddStatic(new Vector3(0, -0.1f, 0), new Vector3(HalfX * 2, 0.2f, HalfZ * 2), wood);
        AddChild(new MeshInstance3D { Mesh = new PlaneMesh { Size = new Vector2(300, 300) }, MaterialOverride = outside, Position = new Vector3(0, -0.25f, 0) });
        AddStatic(new Vector3(0, Height + 0.1f, 0), new Vector3(HalfX * 2, 0.2f, HalfZ * 2), ceilingMat);

        // Walls with window openings (real holes, so the sun comes in with shadows).
        BuildWall(alongX: true, coord: -HalfZ, length: HalfX * 2, plaster, trim, windows: [(-5f, 3f), (5f, 3f)]);
        BuildWall(alongX: true, coord: HalfZ, length: HalfX * 2, plaster, trim, windows: [(-5f, 3f), (5f, 3f)]);
        BuildWall(alongX: false, coord: HalfX, length: HalfZ * 2, plaster, trim, windows: [(0f, 4f)]);
        BuildWall(alongX: false, coord: -HalfX, length: HalfZ * 2, plaster, trim, windows: []);

        // Baseboards and a cornice.
        foreach ((Vector3 c, Vector3 s) in new[]
        {
            (new Vector3(0, 0.06f, -HalfZ + 0.17f), new Vector3(HalfX * 2, 0.12f, 0.04f)),
            (new Vector3(0, 0.06f, HalfZ - 0.17f), new Vector3(HalfX * 2, 0.12f, 0.04f)),
            (new Vector3(HalfX - 0.17f, 0.06f, 0), new Vector3(0.04f, 0.12f, HalfZ * 2)),
            (new Vector3(-HalfX + 0.17f, 0.06f, 0), new Vector3(0.04f, 0.12f, HalfZ * 2)),
        })
        {
            AddChild(new MeshInstance3D { Mesh = new BoxMesh { Size = s }, MaterialOverride = trim, Position = c });
            AddChild(new MeshInstance3D { Mesh = new BoxMesh { Size = s }, MaterialOverride = trim, Position = new Vector3(c.X, Height - 0.06f, c.Z) });
        }

        // A door on the windowless wall, with a frame.
        var doorMat = new StandardMaterial3D { AlbedoColor = new Color(0.33f, 0.2f, 0.11f), Roughness = 0.45f, MetallicSpecular = 0.6f };
        AddChild(new MeshInstance3D { Mesh = new BoxMesh { Size = new Vector3(0.06f, 2.3f, 1.1f) }, MaterialOverride = doorMat, Position = new Vector3(-HalfX + 0.15f + 0.04f, 1.15f, 3f) });
        AddChild(new MeshInstance3D { Mesh = new BoxMesh { Size = new Vector3(0.1f, 2.45f, 1.3f) }, MaterialOverride = trim, Position = new Vector3(-HalfX + 0.15f + 0.01f, 1.22f, 3f) });
        AddChild(new MeshInstance3D { Mesh = new SphereMesh { Radius = 0.04f, Height = 0.08f }, MaterialOverride = new StandardMaterial3D { AlbedoColor = new Color(0.8f, 0.7f, 0.3f), Metallic = 1f, Roughness = 0.25f }, Position = new Vector3(-HalfX + 0.25f, 1.05f, 3.45f) });

        // Columns: dark stained wood, 4 of them — the vertical obstacles.
        var columnMat = new StandardMaterial3D { AlbedoColor = new Color(0.17f, 0.11f, 0.07f), Roughness = 0.5f, MetallicSpecular = 0.5f };
        foreach ((float x, float z) in new[] { (-4f, -3f), (4f, -3f), (-4f, 3f), (4f, 3f) })
        {
            const float r = 0.4f;
            var col = new StaticBody3D { Position = new Vector3(x, Height / 2, z) };
            col.AddChild(new MeshInstance3D { Mesh = new CylinderMesh { TopRadius = r, BottomRadius = r, Height = Height, RadialSegments = 32 }, MaterialOverride = columnMat });
            col.AddChild(new MeshInstance3D { Mesh = new CylinderMesh { TopRadius = r + 0.1f, BottomRadius = r + 0.1f, Height = 0.25f, RadialSegments = 32 }, MaterialOverride = columnMat, Position = new Vector3(0, -Height / 2 + 0.125f, 0) });
            col.AddChild(new MeshInstance3D { Mesh = new CylinderMesh { TopRadius = r + 0.12f, BottomRadius = r, Height = 0.3f, RadialSegments = 32 }, MaterialOverride = columnMat, Position = new Vector3(0, Height / 2 - 0.15f, 0) });
            col.AddChild(new CollisionShape3D { Shape = new CylinderShape3D { Radius = r, Height = Height } });
            AddChild(col);
            Columns.Add(new Column(new Vector3(x, 0, z), r, Height));
        }

        // Ceiling pendants over the two halves of the room.
        foreach (float x in new[] { -6f, 6f })
        {
            AddChild(new MeshInstance3D { Mesh = new CylinderMesh { TopRadius = 0.01f, BottomRadius = 0.01f, Height = 1.2f }, MaterialOverride = new StandardMaterial3D { AlbedoColor = new Color(0.1f, 0.1f, 0.1f) }, Position = new Vector3(x, Height - 0.6f, 0) });
            AddChild(new MeshInstance3D { Mesh = new CylinderMesh { TopRadius = 0.18f, BottomRadius = 0.42f, Height = 0.35f, RadialSegments = 32 }, MaterialOverride = new StandardMaterial3D { AlbedoColor = new Color(0.95f, 0.9f, 0.8f), Roughness = 0.6f, CullMode = BaseMaterial3D.CullModeEnum.Disabled }, Position = new Vector3(x, Height - 1.35f, 0) });
            AddChild(new MeshInstance3D { Mesh = new SphereMesh { Radius = 0.07f, Height = 0.14f }, MaterialOverride = new StandardMaterial3D { AlbedoColor = new Color(1f, 0.95f, 0.85f), EmissionEnabled = true, Emission = new Color(1f, 0.9f, 0.7f), EmissionEnergyMultiplier = 6f }, Position = new Vector3(x, Height - 1.45f, 0) });
            AddChild(new OmniLight3D { Position = new Vector3(x, Height - 1.5f, 0), LightEnergy = 2.5f, OmniRange = 14f, OmniAttenuation = 1.3f, LightColor = new Color(1f, 0.92f, 0.78f), ShadowEnabled = High, LightSize = 0.1f });
        }
    }

    private void BuildWall(bool alongX, float coord, float length, Material material, Material trim, (float Center, float Width)[] windows)
    {
        const float t = 0.3f;
        const float sillY = 1.2f, topY = 3.4f;
        float half = length / 2;
        var edges = new List<float> { -half };
        foreach ((float c, float w) in windows)
        {
            edges.Add(c - w / 2);
            edges.Add(c + w / 2);
        }

        edges.Add(half);
        for (int i = 0; i < edges.Count - 1; i++)
        {
            float a = edges[i], b = edges[i + 1];
            float mid = (a + b) / 2, len = b - a;
            if (i % 2 == 0)
            {
                Segment(mid, len, 0, Height);                     // full-height pier
            }
            else
            {
                Segment(mid, len, 0, sillY);                      // under the window
                Segment(mid, len, topY, Height);                  // above the window
                Window(mid, len);
            }
        }

        void Segment(float mid, float len, float y0, float y1)
        {
            if (y1 - y0 <= 0 || len <= 0)
            {
                return;
            }

            Vector3 center = alongX ? new Vector3(mid, (y0 + y1) / 2, coord) : new Vector3(coord, (y0 + y1) / 2, mid);
            Vector3 size = alongX ? new Vector3(len, y1 - y0, t) : new Vector3(t, y1 - y0, len);
            AddStatic(center, size, material);
        }

        void Window(float mid, float len)
        {
            float inward = coord > 0 ? -1f : 1f;      // toward the room centre
            Vector3 center = alongX ? new Vector3(mid, (sillY + topY) / 2, coord) : new Vector3(coord, (sillY + topY) / 2, mid);
            Vector3 size = alongX ? new Vector3(len, topY - sillY, 0.02f) : new Vector3(0.02f, topY - sillY, len);
            var glass = new StandardMaterial3D
            {
                AlbedoColor = new Color(0.85f, 0.92f, 1f, 0.12f),
                Transparency = BaseMaterial3D.TransparencyEnum.Alpha,
                Roughness = 0.02f,
                Metallic = 0.1f,
                MetallicSpecular = 1f,
                CullMode = BaseMaterial3D.CullModeEnum.Disabled,
            };
            AddStatic(center, size, glass);   // keeps the fly inside; the sun passes through

            // Frame, mullion, sill and curtains.
            Vector3 fs = alongX ? new Vector3(len + 0.2f, 0.1f, t + 0.06f) : new Vector3(t + 0.06f, 0.1f, len + 0.2f);
            AddChild(new MeshInstance3D { Mesh = new BoxMesh { Size = fs }, MaterialOverride = trim, Position = center + new Vector3(0, -(topY - sillY) / 2, 0) });
            AddChild(new MeshInstance3D { Mesh = new BoxMesh { Size = fs }, MaterialOverride = trim, Position = center + new Vector3(0, (topY - sillY) / 2, 0) });
            Vector3 ms = alongX ? new Vector3(0.06f, topY - sillY, t + 0.06f) : new Vector3(t + 0.06f, topY - sillY, 0.06f);
            AddChild(new MeshInstance3D { Mesh = new BoxMesh { Size = ms }, MaterialOverride = trim, Position = center });
            Vector3 hs = alongX ? new Vector3(len, 0.05f, t + 0.06f) : new Vector3(t + 0.06f, 0.05f, len);
            AddChild(new MeshInstance3D { Mesh = new BoxMesh { Size = hs }, MaterialOverride = trim, Position = center + new Vector3(0, 0.3f, 0) });
            Vector3 sillSize = alongX ? new Vector3(len + 0.3f, 0.05f, t + 0.3f) : new Vector3(t + 0.3f, 0.05f, len + 0.3f);
            Vector3 sillPos = center + new Vector3(0, -(topY - sillY) / 2 - 0.05f, 0) + (alongX ? new Vector3(0, 0, inward * 0.12f) : new Vector3(inward * 0.12f, 0, 0));
            AddChild(new MeshInstance3D { Mesh = new BoxMesh { Size = sillSize }, MaterialOverride = trim, Position = sillPos });

            var curtain = new StandardMaterial3D { AlbedoColor = new Color(0.75f, 0.65f, 0.5f), Roughness = 1f, MetallicSpecular = 0.1f };
            foreach (float side in new[] { -1f, 1f })
            {
                Vector3 cs = alongX ? new Vector3(0.5f, topY + 0.4f, 0.12f) : new Vector3(0.12f, topY + 0.4f, 0.5f);
                Vector3 cp = center + (alongX ? new Vector3(side * (len / 2 + 0.3f), 0, inward * (t / 2 + 0.12f)) : new Vector3(inward * (t / 2 + 0.12f), 0, side * (len / 2 + 0.3f)));
                cp.Y = (topY + 0.4f) / 2 + 0.02f;
                AddChild(new MeshInstance3D { Mesh = new BoxMesh { Size = cs }, MaterialOverride = curtain, Position = cp });
            }

            Vector3 rod = alongX ? new Vector3(len + 1.4f, 0.03f, 0.03f) : new Vector3(0.03f, 0.03f, len + 1.4f);
            Vector3 rp = center + new Vector3(0, (topY - sillY) / 2 + 0.35f, 0) + (alongX ? new Vector3(0, 0, inward * (t / 2 + 0.12f)) : new Vector3(inward * (t / 2 + 0.12f), 0, 0));
            AddChild(new MeshInstance3D { Mesh = new BoxMesh { Size = rod }, MaterialOverride = new StandardMaterial3D { AlbedoColor = new Color(0.3f, 0.3f, 0.32f), Metallic = 0.9f, Roughness = 0.3f }, Position = rp });
        }
    }

    private void BuildFurniture()
    {
        var couchColor = new Color(0.22f, 0.34f, 0.4f);
        StandardMaterial3D fabric = ProceduralTextures.Fabric(couchColor);
        StandardMaterial3D cushionA = ProceduralTextures.Fabric(new Color(0.75f, 0.6f, 0.35f));
        StandardMaterial3D cushionB = ProceduralTextures.Fabric(new Color(0.5f, 0.2f, 0.18f));
        var darkWood = new StandardMaterial3D { AlbedoColor = new Color(0.26f, 0.16f, 0.09f), Roughness = 0.3f, MetallicSpecular = 0.7f };
        var couch = new Vector3(-7f, 0, -6.5f);

        // Couch on the rug, facing +Z: base, seat cushions, back, armrests, throw cushions.
        Furniture(couch + new Vector3(0, 0.2f, 0.1f), new Vector3(3.2f, 0.4f, 1.2f), fabric, couchColor);
        foreach (float sx in new[] { -0.75f, 0.75f })
        {
            AddChild(new MeshInstance3D { Mesh = new BoxMesh { Size = new Vector3(1.4f, 0.22f, 1.05f) }, MaterialOverride = fabric, Position = couch + new Vector3(sx, 0.51f, 0.12f) });
        }

        Furniture(couch + new Vector3(0, 0.95f, -0.45f), new Vector3(3.2f, 1.0f, 0.3f), fabric, couchColor);
        Furniture(couch + new Vector3(-1.5f, 0.6f, 0.1f), new Vector3(0.25f, 0.85f, 1.2f), fabric, couchColor);
        Furniture(couch + new Vector3(1.5f, 0.6f, 0.1f), new Vector3(0.25f, 0.85f, 1.2f), fabric, couchColor);
        AddChild(new MeshInstance3D { Mesh = new BoxMesh { Size = new Vector3(0.45f, 0.45f, 0.14f) }, MaterialOverride = cushionA, Position = couch + new Vector3(-1.05f, 0.88f, -0.2f), RotationDegrees = new Vector3(-12, 8, 0) });
        AddChild(new MeshInstance3D { Mesh = new BoxMesh { Size = new Vector3(0.45f, 0.45f, 0.14f) }, MaterialOverride = cushionB, Position = couch + new Vector3(1.0f, 0.88f, -0.2f), RotationDegrees = new Vector3(-12, -10, 0) });
        AddChild(new MeshInstance3D { Mesh = new PlaneMesh { Size = new Vector2(6f, 4f) }, MaterialOverride = ProceduralTextures.Rug(), Position = new Vector3(-7f, 0.012f, -5f) });

        // Coffee table in front of the couch (low) and a dining table (high) with chairs.
        Table(new Vector3(-7f, 0, -3.5f), new Vector2(2.0f, 1.2f), 0.55f, darkWood);
        Table(new Vector3(6f, 0, 4f), new Vector2(2.4f, 1.4f), 1.0f, darkWood);
        foreach ((float sx, float sz) in new[] { (-0.85f, 0f), (0.85f, 0f), (0f, -0.55f), (0f, 0.55f) })
        {
            Chair(new Vector3(6f + sx * 1.6f, 0, 4f + sz * 1.7f), darkWood);
        }

        // Bookshelf on the windowless wall, a floor lamp, a plant.
        var shelfMat = new StandardMaterial3D { AlbedoColor = new Color(0.55f, 0.42f, 0.28f), Roughness = 0.5f };
        var shelfAt = new Vector3(-HalfX + 0.5f, 0, -3f);
        Furniture(shelfAt + new Vector3(0, 1.1f, 0), new Vector3(0.4f, 2.2f, 2.4f), shelfMat, new Color(0.5f, 0.4f, 0.3f));
        var rng = new Random(4);
        for (int shelf = 0; shelf < 4; shelf++)
        {
            float z = -1.05f;
            while (z < 1.05f)
            {
                float w = 0.04f + (float)rng.NextDouble() * 0.05f;
                float h = 0.22f + (float)rng.NextDouble() * 0.12f;
                var c = Color.FromHsv((float)rng.NextDouble(), 0.5f + 0.4f * (float)rng.NextDouble(), 0.35f + 0.5f * (float)rng.NextDouble());
                AddChild(new MeshInstance3D { Mesh = new BoxMesh { Size = new Vector3(0.28f, h, w) }, MaterialOverride = new StandardMaterial3D { AlbedoColor = c, Roughness = 0.8f }, Position = shelfAt + new Vector3(0.22f, 0.3f + shelf * 0.5f + h / 2, z + w / 2) });
                z += w + 0.01f;
            }
        }

        var lampBase = new Vector3(-HalfX + 1.2f, 0, -8f);
        AddChild(new MeshInstance3D { Mesh = new CylinderMesh { TopRadius = 0.015f, BottomRadius = 0.015f, Height = 1.6f }, MaterialOverride = new StandardMaterial3D { AlbedoColor = new Color(0.2f, 0.2f, 0.22f), Metallic = 0.9f, Roughness = 0.3f }, Position = lampBase + new Vector3(0, 0.8f, 0) });
        AddChild(new MeshInstance3D { Mesh = new CylinderMesh { TopRadius = 0.2f, BottomRadius = 0.28f, Height = 0.35f, RadialSegments = 32 }, MaterialOverride = new StandardMaterial3D { AlbedoColor = new Color(0.95f, 0.85f, 0.7f), Roughness = 0.7f, CullMode = BaseMaterial3D.CullModeEnum.Disabled }, Position = lampBase + new Vector3(0, 1.7f, 0) });
        AddChild(new OmniLight3D { Position = lampBase + new Vector3(0, 1.65f, 0), LightEnergy = 1.4f, OmniRange = 6f, LightColor = new Color(1f, 0.85f, 0.65f), ShadowEnabled = High, LightSize = 0.1f });

        var pot = new StaticBody3D { Position = new Vector3(10.5f, 0.35f, -7.5f) };
        pot.AddChild(new MeshInstance3D { Mesh = new CylinderMesh { TopRadius = 0.45f, BottomRadius = 0.35f, Height = 0.7f, RadialSegments = 32 }, MaterialOverride = new StandardMaterial3D { AlbedoColor = new Color(0.55f, 0.28f, 0.17f), Roughness = 0.8f } });
        pot.AddChild(new CollisionShape3D { Shape = new CylinderShape3D { Radius = 0.45f, Height = 0.7f } });
        AddChild(pot);
        var leaves = new StaticBody3D { Position = new Vector3(10.5f, 1.6f, -7.5f) };
        var leafMat = new StandardMaterial3D { AlbedoColor = new Color(0.12f, 0.32f, 0.12f), Roughness = 0.6f };
        for (int i = 0; i < 9; i++)
        {
            float a = i * 0.7f;
            leaves.AddChild(new MeshInstance3D { Mesh = new SphereMesh { Radius = 0.45f, Height = 0.9f }, MaterialOverride = leafMat, Position = new Vector3(MathF.Cos(a) * 0.5f, (i % 3) * 0.35f - 0.3f, MathF.Sin(a) * 0.5f), Scale = new Vector3(1.3f, 0.5f, 0.8f), Rotation = new Vector3(0, -a, 0.3f) });
        }

        leaves.AddChild(new CollisionShape3D { Shape = new SphereShape3D { Radius = 0.9f } });
        AddChild(leaves);
        Boxes.Add(new Box(new Vector3(10.5f, 1.2f, -7.5f), new Vector3(1.8f, 2.4f, 1.8f), new Color(0.2f, 0.5f, 0.2f)));

        // A picture on the wall.
        AddChild(new MeshInstance3D { Mesh = new BoxMesh { Size = new Vector3(1.4f, 1.0f, 0.04f) }, MaterialOverride = new StandardMaterial3D { AlbedoColor = new Color(0.1f, 0.08f, 0.06f), Roughness = 0.4f }, Position = new Vector3(-7f, 2.4f, -HalfZ + 0.17f) });
        AddChild(new MeshInstance3D { Mesh = new BoxMesh { Size = new Vector3(1.25f, 0.85f, 0.02f) }, MaterialOverride = new StandardMaterial3D { AlbedoColor = new Color(0.55f, 0.65f, 0.75f), Roughness = 0.9f }, Position = new Vector3(-7f, 2.4f, -HalfZ + 0.2f) });
    }

    private void Chair(Vector3 at, Material wood)
    {
        Furniture(at + new Vector3(0, 0.45f, 0), new Vector3(0.45f, 0.05f, 0.45f), wood, new Color(0.45f, 0.3f, 0.18f));
        AddChild(new MeshInstance3D { Mesh = new BoxMesh { Size = new Vector3(0.45f, 0.5f, 0.04f) }, MaterialOverride = wood, Position = at + new Vector3(0, 0.72f, MathF.Sign(at.Z - 4f) * 0.2f) });
        foreach ((float sx, float sz) in new[] { (-1f, -1f), (1f, -1f), (-1f, 1f), (1f, 1f) })
        {
            AddChild(new MeshInstance3D { Mesh = new CylinderMesh { TopRadius = 0.02f, BottomRadius = 0.02f, Height = 0.45f }, MaterialOverride = wood, Position = at + new Vector3(sx * 0.19f, 0.22f, sz * 0.19f) });
        }
    }

    private void Table(Vector3 at, Vector2 top, float height, Material wood)
    {
        Furniture(at + new Vector3(0, height - 0.03f, 0), new Vector3(top.X, 0.06f, top.Y), wood, new Color(0.45f, 0.3f, 0.18f));
        foreach ((float sx, float sz) in new[] { (-1f, -1f), (1f, -1f), (-1f, 1f), (1f, 1f) })
        {
            AddChild(new MeshInstance3D { Mesh = new CylinderMesh { TopRadius = 0.035f, BottomRadius = 0.03f, Height = height - 0.06f }, MaterialOverride = wood, Position = at + new Vector3(sx * (top.X / 2 - 0.1f), (height - 0.06f) / 2, sz * (top.Y / 2 - 0.1f)) });
        }
    }

    private void Furniture(Vector3 center, Vector3 size, Material material, Color mapColor)
    {
        AddStatic(center, size, material);
        Boxes.Add(new Box(center, size, mapColor));
    }

    private void AddStatic(Vector3 center, Vector3 size, Material material)
    {
        var body = new StaticBody3D { Position = center };
        body.AddChild(new MeshInstance3D { Mesh = new BoxMesh { Size = size }, MaterialOverride = material });
        body.AddChild(new CollisionShape3D { Shape = new BoxShape3D { Size = size } });
        AddChild(body);
    }

    private void BuildSugar()
    {
        // Sugar: a spilled juice puddle on the floor, a bowl on the dining table, a plate on the
        // coffee table. Each emits an odor plume (Odor.cs). Y is the surface the patch sits on.
        var syrup = new StandardMaterial3D { AlbedoColor = new Color(0.95f, 0.65f, 0.15f, 0.85f), Transparency = BaseMaterial3D.TransparencyEnum.Alpha, Roughness = 0.05f, MetallicSpecular = 1f, ClearcoatEnabled = true, Clearcoat = 1f };
        var china = new StandardMaterial3D { AlbedoColor = new Color(0.95f, 0.95f, 0.92f), Roughness = 0.15f, MetallicSpecular = 0.8f };
        foreach ((Vector3 p, float r, bool dish) in new[] { (new Vector3(2f, 0f, -5.5f), 1.1f, false), (new Vector3(6f, 1.0f, 4f), 0.8f, true), (new Vector3(-7f, 0.55f, -3.5f), 0.7f, true) })
        {
            Odor.Add(p, r);
            if (dish)
            {
                var plate = new StaticBody3D { Position = p + new Vector3(0, 0.025f, 0) };
                plate.AddChild(new MeshInstance3D { Mesh = new CylinderMesh { TopRadius = r + 0.1f, BottomRadius = r - 0.05f, Height = 0.05f, RadialSegments = 48 }, MaterialOverride = china });
                plate.AddChild(new CollisionShape3D { Shape = new CylinderShape3D { Radius = r + 0.1f, Height = 0.05f } });
                AddChild(plate);   // a body, so the occlusion fader can see it in the way
            }

            AddChild(new MeshInstance3D { Mesh = new CylinderMesh { TopRadius = r * 0.92f, BottomRadius = r * 0.92f, Height = 0.02f, RadialSegments = 48 }, MaterialOverride = syrup, Position = p + new Vector3(0, dish ? 0.06f : 0.01f, 0) });
        }
    }

    private void BuildOdorRender()
    {
        if (!RenderOdor)
        {
            return;
        }

        var shader = GD.Load<Shader>("res://odor_raymarch.gdshader");
        _odorMaterial = new ShaderMaterial { Shader = shader };
        IReadOnlyList<Odor.Patch> patches = Odor.Patches;
        _odorMaterial.SetShaderParameter("patch_count", patches.Count);
        for (int i = 0; i < patches.Count && i < 3; i++)
        {
            _odorMaterial.SetShaderParameter($"patch{i}", patches[i].Position);
        }

        _odorMaterial.SetShaderParameter("plume_length", Odor.PlumeLength);
        _odorMaterial.SetShaderParameter("plume_width", Odor.PlumeWidth);
        _odorMaterial.SetShaderParameter("cone_scale", Odor.ConeScale);
        _odorMaterial.SetShaderParameter("core_height", Odor.CoreHeight);
        _odorMaterial.SetShaderParameter("vertical_meander_amplitude", Odor.VerticalMeanderAmplitude);
        _odorMaterial.SetShaderParameter("vertical_meander_wavelength", Odor.VerticalMeanderWavelength);
        _odorMaterial.SetShaderParameter("rise_per_metre", Odor.RisePerMetre);
        _odorMaterial.SetShaderParameter("vertical_width", Odor.VerticalWidth);
        _odorMaterial.SetShaderParameter("meander_amplitude", Odor.MeanderAmplitude);
        _odorMaterial.SetShaderParameter("meander_wavelength", Odor.MeanderWavelength);
        _odorMaterial.SetShaderParameter("meander_period", Odor.MeanderPeriod);
        _odorMaterial.SetShaderParameter("wind", Odor.Wind);
        _odorMaterial.SetShaderParameter("box_min", new Vector3(-Odor.RoomHalfSize.X, 0, -Odor.RoomHalfSize.Y));   // inner wall faces, as Odor.cs
        _odorMaterial.SetShaderParameter("box_max", new Vector3(Odor.RoomHalfSize.X, Height, Odor.RoomHalfSize.Y));
        _odorMaterial.SetShaderParameter("wall_margin", Odor.WallMargin);
        _odorMaterial.SetShaderParameter("end_margin", Odor.EndMargin);
        _odorMaterial.SetShaderParameter("sun_dir", _sunDir.Normalized());
        _odorMaterial.SetShaderParameter("noise_tex", new NoiseTexture3D
        {
            Width = 64,
            Height = 64,
            Depth = 64,
            Seamless = true,
            Noise = new FastNoiseLite { NoiseType = FastNoiseLite.NoiseTypeEnum.SimplexSmooth, Frequency = 0.06f, FractalOctaves = 4, FractalLacunarity = 2.1f, FractalGain = 0.55f, Seed = 21 },
        });

        // Back faces of a box around the room, drawn only for the chase camera (layer 2).
        AddChild(new MeshInstance3D
        {
            Mesh = new BoxMesh { Size = new Vector3(HalfX * 2, Height, HalfZ * 2) },
            Position = new Vector3(0, Height / 2, 0),
            MaterialOverride = _odorMaterial,
            Layers = HumanOnlyLayer,
            CastShadow = GeometryInstance3D.ShadowCastingSetting.Off,
            ExtraCullMargin = 16f,
        });
    }
}
