using System;
using System.Collections.Generic;
using System.Linq;
using Godot;

namespace DotFly.Sample.Godot3D;

/// <summary>
/// The fly: an animated body (<see cref="FlyModel"/>), two eye cameras rendered to 32×32 textures,
/// the third-person chase camera with inertia, and the instrument panels. Each frame: eye images →
/// <see cref="Retina"/>, odor at the antennae and sugar contact → <see cref="FlyBrain3D.Encode"/>;
/// readouts → <see cref="Behaviour"/> → yaw / climb / speed → flight with pitch, or landing /
/// feeding on a patch. Nothing here is biomechanics.
/// </summary>
public partial class FlyBody3D : CharacterBody3D
{
    private const int EyeSize = 32;

    /// <summary>
    /// Body length of the fly in metres. A real Drosophila is 0.003 m (1:770 against the 2.3 m
    /// door); the default 0.03 m is ten times that — a fly's world, still visible from the chase
    /// camera. Everything body-relative scales with it: eyes, antennae, collider, landing heights,
    /// camera distance and the flight speeds (the decoder's speeds are body lengths per second).
    /// The room and the odor tubes do not. <c>DOTFLY_BODY_LENGTH</c> overrides.
    /// </summary>
    [Export]
    public float BodyLength { get; set; } = float.TryParse(System.Environment.GetEnvironmentVariable("DOTFLY_BODY_LENGTH"), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float bl) ? bl : 0.03f;

    private float FlyScale => BodyLength / 0.9f;   // the procedural model is built at 0.9 m

    /// <summary>Save <c>screenshot.png</c> at this physics frame (0 = never); for unattended runs.</summary>
    [Export]
    public int ScreenshotAtFrame { get; set; } = int.TryParse(System.Environment.GetEnvironmentVariable("DOTFLY_SCREENSHOT_FRAME"), out int f) ? f : 0;

    /// <summary>Save <c>screenshot.png</c> 0.3 s into this behaviour state (e.g. <c>Feeding</c>), once.</summary>
    [Export]
    public string? ScreenshotState { get; set; } = System.Environment.GetEnvironmentVariable("DOTFLY_SCREENSHOT_STATE");

    /// <summary>
    /// Optional glTF model (<c>res://models/fly.glb</c> by default): loaded at runtime (no editor
    /// import needed), auto-scaled to <see cref="BodyLength"/>, its first animation played in
    /// flight. Feeding is then shown by the body dipping. Without the file, the procedural fly is used.
    /// </summary>
    [Export]
    public string ModelPath { get; set; } = "res://models/fly.glb";

    /// <summary>Yaw (degrees) to turn a loaded model so it faces −Z.</summary>
    [Export]
    public float ModelYawDegrees { get; set; } = 180f;   // the bundled model faces +Z

    private bool _stateShotDone;
    private readonly ImageTexture[] _eyeGray = new ImageTexture[2];
    private readonly Image[] _eyeL8Image = new Image[2];
    private readonly byte[][] _eyeL8 = [new byte[EyeSize * EyeSize], new byte[EyeSize * EyeSize]];
    private readonly bool[] _eyeFresh = new bool[2];
    private readonly bool[] _eyePending = new bool[2];
    private RenderingDevice? _rd;
    private bool _asyncReadback = true;
    private Node3D? _loadedModel;
    private AnimationPlayer? _loadedAnimation;
    private BlurredWings? _blurredWings;
    private float _loadedDip;
    private FlyBrain3D? _brain;
    private Arena? _arena;
    private Odor? _odor;
    private Behaviour? _behaviour;
    private readonly SubViewport[] _eyes = new SubViewport[2];
    private readonly Camera3D[] _eyeCameras = new Camera3D[2];
    private Label? _hud;
    private EyeView? _eyeView;
    private BrainMapView? _mapView;
    private TraceView? _traces;
    private MinimapView? _minimap;
    private readonly Retina _retina = new(EyeSize, EyeSize);
    private FlyModel? _model;
    private Node3D? _visual;
    private float _yaw;
    private float _pitch;
    private float _bumper;
    private float _hop;
    private float _stuckTimer;
    private Vector3 _stuckAnchor;
    private int _frame;
    private Vector3 _camHeading = new(0, 0, -1);
    private Vector3 _camPos;
    private Vector3 _camOffset;
    private Vector3 _camLook;
    private float _camDistance = 4.5f;
    private readonly OcclusionFader _occlusion = new();

    /// <summary>The brain driving this fly; null = the scene's <c>FlyBrain</c> node.</summary>
    public FlyBrain3D? BrainNode { get; set; }

    /// <summary>The primary fly owns the panels, the HUD, the chase camera and the console log.</summary>
    public bool Primary { get; set; } = true;

    /// <summary>Current behaviour state (for the panels).</summary>
    public Behaviour.State State => _behaviour?.Current ?? Behaviour.State.Flying;

    public override void _Ready()
    {
        _brain = BrainNode ?? GetNode<FlyBrain3D>("../FlyBrain");
        _arena = GetParent<Arena>();
        _odor = _arena.Odor;
        _behaviour = new Behaviour(_brain.Gains, BodyLength);

        // Visual parts hang off a pivot so the body can pitch without affecting the collider.
        _visual = new Node3D();
        AddChild(_visual);
        if (!TryLoadModel())
        {
            _model = new FlyModel { Scale = new Vector3(FlyScale, FlyScale, FlyScale) };
            _visual.AddChild(_model);
        }

        AddChild(new CollisionShape3D { Shape = new SphereShape3D { Radius = 0.42f * BodyLength } });

        // Two eyes: one camera per compound eye, 120° each, looking 45° to its side.
        Godot.Environment eyeEnv = _arena.CreateEyeEnvironment();
        for (int eye = 0; eye < 2; eye++)
        {
            _eyes[eye] = new SubViewport { Size = new Vector2I(EyeSize, EyeSize), RenderTargetUpdateMode = SubViewport.UpdateMode.Always, OwnWorld3D = false };
            _eyeCameras[eye] = new Camera3D { Fov = 120, Near = MathF.Max(0.001f, 0.05f * BodyLength), Far = 100, Environment = eyeEnv, CullMask = 1 };   // layer 1 only: no odor rendering
            _eyeL8Image[eye] = Image.CreateEmpty(EyeSize, EyeSize, false, Image.Format.L8);
            _eyeGray[eye] = ImageTexture.CreateFromImage(_eyeL8Image[eye]);
            _eyes[eye].AddChild(_eyeCameras[eye]);
            AddChild(_eyes[eye]);
        }

        _camOffset = new Vector3(0, 1.7f, 5f) * BodyLength;
        _camPos = GlobalPosition + _camOffset;
        _camDistance = 4.2f * BodyLength;
        _arena.Chase.Near = MathF.Max(0.001f, 0.05f * BodyLength);
        _camLook = GlobalPosition;

        if (!Primary)
        {
            return;   // extra flies: no panels, no HUD, no camera
        }

        // Instrument panels: the left column (eyes, brain map, traces) is full height and scales
        // with the window; the room map sits bottom-right; HUD text on top of the 3D view.
        var canvas = new CanvasLayer();
        _eyeView = new EyeView { ClipContents = true };
        _mapView = new BrainMapView { ClipContents = true };
        _traces = new TraceView { ClipContents = true };
        _minimap = new MinimapView { ClipContents = true };
        _hud = new Label();
        _hud.AddThemeFontSizeOverride("font_size", 14);
        canvas.AddChild(_eyeView);
        canvas.AddChild(_mapView);
        canvas.AddChild(_traces);
        canvas.AddChild(_minimap);
        canvas.AddChild(_hud);
        AddChild(canvas);
        Layout();
        GetTree().Root.SizeChanged += Layout;
        if (System.Environment.GetEnvironmentVariable("DOTFLY_PERF") == "nopanels")
        {
            canvas.Visible = false;
            _mapView.ProcessMode = ProcessModeEnum.Disabled;
            _traces.ProcessMode = ProcessModeEnum.Disabled;
            _eyeView.ProcessMode = ProcessModeEnum.Disabled;
            _minimap.ProcessMode = ProcessModeEnum.Disabled;
        }
        else if (System.Environment.GetEnvironmentVariable("DOTFLY_PERF") == "nomap")
        {
            _mapView.ProcessMode = ProcessModeEnum.Disabled;
        }

        if (_brain.Brain is not null)
        {
            _eyeView.Bind(_eyeGray[0], _eyeGray[1], _retina, _brain);
            _mapView.Bind(_brain);
            _traces.Bind(_brain, _behaviour);
            _minimap.Bind(_arena, this);
        }
    }

    private void Layout()
    {
        Vector2 vs = GetViewport().GetVisibleRect().Size;
        float panelW = MathF.Max(440f, vs.X * 0.34f);
        float eyeSize = (panelW - 24 - 190) / 2f;
        float eyeH = MathF.Max(eyeSize, 5 * 26f) + 72;
        float mapH = (vs.Y - eyeH) * 0.55f;
        _eyeView!.Position = Vector2.Zero;
        _eyeView.Size = new Vector2(panelW, eyeH);
        _mapView!.Position = new Vector2(0, eyeH + 2);
        _mapView.Size = new Vector2(panelW, mapH);
        _traces!.Position = new Vector2(0, eyeH + mapH + 4);
        _traces.Size = new Vector2(panelW, vs.Y - eyeH - mapH - 4);
        float miniW = MathF.Min(340f, (vs.X - panelW) * 0.4f);
        _minimap!.Size = new Vector2(miniW, miniW * 0.8f);
        _minimap.Position = new Vector2(vs.X - miniW - 8, vs.Y - miniW * 0.8f - 8);
        _hud!.Position = new Vector2(panelW + 12, 8);
        _hud.Size = new Vector2(vs.X - panelW - 24, 90);
    }

    public override void _PhysicsProcess(double delta)
    {
        if (_brain is null || _odor is null || _behaviour is null || _arena is null)
        {
            return;
        }

        float dt = (float)delta;
        _odor.Update(delta);

        // Eye cameras ride on the head (pitching with the body), 45° to each side.
        for (int eye = 0; eye < 2; eye++)
        {
            float side = eye == 0 ? -1f : 1f;
            _eyeCameras[eye].GlobalTransform = GlobalTransform * new Transform3D(Basis.FromEuler(new Vector3(_pitch, -side * MathF.PI / 4, 0)), new Vector3(side * 0.08f, 0.1f, -0.4f) * FlyScale);
        }

        // ---- sense -------------------------------------------------------------------------
        // Eye images come back asynchronously from the GPU (no pipeline stall); the retina runs
        // whenever both eyes have a fresh frame, at ~30–60 Hz.
        ReadEyes();
        if (_eyeFresh[0] && _eyeFresh[1])
        {
            _eyeFresh[0] = _eyeFresh[1] = false;
            _retina.Process(_eyeL8[0], _eyeL8[1]);
            for (int eye = 0; eye < 2; eye++)
            {
                _eyeL8Image[eye].SetData(EyeSize, EyeSize, false, Image.Format.L8, _eyeL8[eye]);
                _eyeGray[eye].Update(_eyeL8Image[eye]);   // the panel shows exactly what the retina got: luminance
            }
        }

        Vector3 rightVec = Transform.Basis.X;
        Vector3 antennae = GlobalPosition + new Vector3(0, 0.05f, 0) * FlyScale - Transform.Basis.Z * (0.45f * FlyScale);
        float odorL = _odor.Concentration(antennae - rightVec * (0.1f * FlyScale));
        float odorR = _odor.Concentration(antennae + rightVec * (0.1f * FlyScale));
        Odor.Patch? patch = _odor.PatchUnder(GlobalPosition);
        float surfaceY = patch?.Position.Y ?? 0f;
        bool landed = _behaviour.Current is Behaviour.State.Landed or Behaviour.State.Feeding;
        bool onSugar = landed && patch is not null;
        // Wind on the antennae (encoder): the air comes from −Wind; the antenna on the side it
        // comes from is deflected most, head-on wind deflects both a little.
        Vector3 from = -_odor.Wind;
        float fromLeft = from.Dot(-rightVec);
        float windL = Math.Clamp(0.35f + fromLeft, 0f, 1f);
        float windR = Math.Clamp(0.35f - fromLeft, 0f, 1f);
        _brain.Encode(_retina, odorL, odorR, tarsalContact: onSugar, labellarContact: onSugar && _behaviour.StateTime > 0.5f, windL, windR);

        // ---- decide ------------------------------------------------------------------------
        _brain.ReadOut();
        _brain.DecayActivity(MathF.Exp(-dt / 0.15f));
        float altitude = GlobalPosition.Y - surfaceY;
        (float yawDegPerSec, float climb, float speed) = _behaviour.Step(_brain.Rates, patch is not null, altitude, dt);
        bool grounded = _behaviour.Current is Behaviour.State.Landed or Behaviour.State.Feeding;
        // Body reflexes (not the network): a half-turn after hitting something, and a half-turn
        // plus a hop when the fly has not moved for 3 s (wedged in a corner). Distances in body lengths.
        if (_bumper > 0)
        {
            yawDegPerSec += 180f;
            _bumper -= dt;
        }

        if (!grounded && (GlobalPosition - _stuckAnchor).Length() > 1.1f * BodyLength)
        {
            _stuckAnchor = GlobalPosition;
            _stuckTimer = 0;
        }
        else if (!grounded && (_stuckTimer += dt) > 3f)
        {
            _stuckTimer = 0;
            _bumper = 1f;
            _hop = 1f;
        }

        if (_hop > 0)
        {
            _hop -= dt;
            climb = MathF.Max(climb, 33f * BodyLength);
        }

        _traces?.SetActions(yawDegPerSec, climb, speed, BodyLength);

        // ---- act ---------------------------------------------------------------------------
        _yaw -= Mathf.DegToRad(yawDegPerSec) * dt;                 // positive = turn right
        Rotation = new Vector3(0, _yaw, 0);
        Vector3 forward = -Transform.Basis.Z;
        Velocity = grounded ? Vector3.Zero : forward * speed + Vector3.Up * climb;
        if (grounded)
        {
            Position = new Vector3(Position.X, surfaceY + 0.36f * FlyScale, Position.Z);
        }

        MoveAndSlide();
        if (!grounded)
        {
            float floor = _behaviour.Current == Behaviour.State.Landing ? surfaceY + 0.36f * FlyScale : 1.4f * BodyLength;
            Position = new Vector3(Position.X, Mathf.Clamp(Position.Y, floor, _arena.Height - 0.5f), Position.Z);
            if (IsOnWall() && _bumper <= 0)
            {
                _bumper = 1f;
            }
        }

        // Pitch the body along the velocity; animate the model.
        float targetPitch = grounded ? 0 : MathF.Atan2(climb, MathF.Max(speed, 0.5f));
        _pitch += (targetPitch - _pitch) * MathF.Min(1f, dt * 6f);
        if (_visual is not null)
        {
            _visual.Rotation = new Vector3(_pitch, 0, 0);
        }

        _model?.Animate(dt, _behaviour.Current, MathF.Min(1f, _brain.Rates[10] / 60f));
        AnimateLoadedModel(dt, grounded);
        if (!Primary)
        {
            return;
        }

        UpdateChaseCamera(dt, grounded);

        if (_hud is not null)
        {
            _hud.Text = $"{_brain.Status}\n{_brain.ClockLine()}\n{_brain.SwitchLine()}\n" +
                        $"[C] camera {(Camera == CameraMode.Chase ? "chase" : "FOLLOW")}   state {_behaviour.Current} ({_behaviour.YawSource})   odor L {odorL:F2} R {odorR:F2}   patch {(patch is null ? "—" : "under")}   alt {altitude:F1} m   pitch {Mathf.RadToDeg(_pitch):F0}°" +
                        (_bumper > 0 ? "   BUMPER (body reflex, not the network)" : string.Empty) + (_hop > 0 ? " + HOP" : string.Empty);
        }

        bool stateShot = ScreenshotState is not null && !_stateShotDone
            && (ScreenshotState == "Faded" ? _occlusion.Count > 0 : ScreenshotState == "Odor" ? odorL > 0.3f : _behaviour.Current.ToString() == ScreenshotState && _behaviour.StateTime > 0.3f);
        if ((ScreenshotAtFrame > 0 && _frame == ScreenshotAtFrame) || stateShot)
        {
            _stateShotDone |= stateShot;
            string file = ProjectSettings.GlobalizePath("res://screenshot.png");
            GetViewport().GetTexture().GetImage().SavePng(file);
            GD.Print($"screenshot saved: {file}  fly on screen at {_arena!.Chase.UnprojectPosition(GlobalPosition)} (viewport {GetViewport().GetVisibleRect().Size}), camera {_arena.Chase.GlobalPosition} fly {GlobalPosition} look {_camLook}");
        }

        if (++_frame % 120 == 0)
        {
            GD.Print($"pos ({Position.X:F1}, {Position.Y:F1}, {Position.Z:F1}) heading {Mathf.RadToDeg(_yaw):F0}° {_behaviour.Current} [{_behaviour.YawSource}] lost/recovered {_behaviour.Losses}/{_behaviour.Recoveries} faded {_occlusion.Count} rearview {(-GlobalTransform.Basis.Z).Dot((GlobalPosition - _arena!.Chase.GlobalPosition).Normalized()):F2} odor {odorL:F2}/{odorR:F2} | {_brain.ClockLine()}");
            GD.Print($"  rates: {string.Join(" ", FlyBrain3D.GroupNames.Select((n, i) => $"{n} {_brain.Rates[i]:F0}"))} | yaw {yawDegPerSec:F0} climb {climb:F2} speed {speed:F1}");
        }
    }

    private void ReadEyes()
    {
        for (int eye = 0; eye < 2; eye++)
        {
            if (_eyePending[eye])
            {
                continue;
            }

            ViewportTexture? tex = _eyes[eye].GetTexture();
            if (tex is null)
            {
                continue;
            }

            if (_asyncReadback)
            {
                _rd ??= RenderingServer.GetRenderingDevice();
                Rid rdTex = _rd is null ? default : RenderingServer.TextureGetRdTexture(tex.GetRid());
                if (_rd is not null && rdTex.IsValid)
                {
                    int e = eye;
                    _eyePending[eye] = true;
                    _rd.TextureGetDataAsync(rdTex, 0, Callable.From((byte[] data) => OnEyeData(e, data)));
                    continue;
                }

                _asyncReadback = false;   // no RenderingDevice (compatibility renderer): synchronous fallback
                GD.Print("dotFly: eye readback falls back to synchronous GetImage()");
            }

            Image img = tex.GetImage();
            if (!img.IsEmpty())
            {
                img.Convert(Image.Format.L8);
                img.GetData().CopyTo(_eyeL8[eye], 0);
                _eyeFresh[eye] = true;
            }
        }
    }

    private void OnEyeData(int eye, byte[] data)
    {
        _eyePending[eye] = false;
        int n = EyeSize * EyeSize;
        byte[] l8 = _eyeL8[eye];
        if (data.Length == n * 4)
        {
            // RGBA8 (sRGB-encoded, the same bytes GetImage() returns) → Rec. 601 luma.
            for (int i = 0; i < n; i++)
            {
                l8[i] = (byte)((data[4 * i] * 77 + data[4 * i + 1] * 150 + data[4 * i + 2] * 29) >> 8);
            }
        }
        else if (data.Length == n * 8)
        {
            // RGBA16F render target: linear half floats → sRGB-ish luma.
            for (int i = 0; i < n; i++)
            {
                float r = (float)BitConverter.Int16BitsToHalf(BitConverter.ToInt16(data, 8 * i));
                float g = (float)BitConverter.Int16BitsToHalf(BitConverter.ToInt16(data, 8 * i + 2));
                float b = (float)BitConverter.Int16BitsToHalf(BitConverter.ToInt16(data, 8 * i + 4));
                float y = MathF.Pow(Math.Clamp(0.299f * r + 0.587f * g + 0.114f * b, 0f, 1f), 1f / 2.2f);
                l8[i] = (byte)(y * 255f);
            }
        }
        else
        {
            _asyncReadback = false;
            GD.Print($"dotFly: unexpected eye texture size {data.Length} bytes; synchronous fallback");
            return;
        }

        _eyeFresh[eye] = true;
    }

    private bool TryLoadModel()
    {
        if (string.IsNullOrEmpty(ModelPath) || !Godot.FileAccess.FileExists(ModelPath))
        {
            return false;
        }

        var doc = new GltfDocument();
        var state = new GltfState();
        Error err = doc.AppendFromFile(ProjectSettings.GlobalizePath(ModelPath), state);
        if (err != Error.Ok)
        {
            GD.PushWarning($"fly model {ModelPath}: {err}; using the procedural fly");
            return false;
        }

        if (doc.GenerateScene(state) is not Node3D root)
        {
            return false;
        }

        // Fit: longest horizontal extent = BodyLength, centred; yaw as configured.
        var pivot = new Node3D { RotationDegrees = new Vector3(0, ModelYawDegrees, 0) };
        pivot.AddChild(root);
        _visual!.AddChild(pivot);
        Aabb box = MergedAabb(root);
        float extent = MathF.Max(box.Size.X, box.Size.Z);
        float k = extent > 1e-4f ? BodyLength / extent : 1f;
        root.Scale = new Vector3(k, k, k);
        // Centre it horizontally; put the feet where the landed body rests (0.36·scale below the centre).
        root.Position = new Vector3(-box.GetCenter().X * k, -box.Position.Y * k - 0.36f * FlyScale + 0.01f, -box.GetCenter().Z * k);
        _loadedModel = pivot;
        _blurredWings = BlurredWings.TryCreate(root);   // before the layer pass: the ghosts must not reach the eyes either
        FlyModel.MakeInvisibleToEyes(root);
        _loadedAnimation = FindAnimationPlayer(root);
        if (_loadedAnimation is not null)
        {
            string[] clips = _loadedAnimation.GetAnimationList();
            _hasPoseClips = clips.Contains("flight") && clips.Contains("standing");
            if (_hasPoseClips)
            {
                _loadedAnimation.Play("flight");   // the rigged model: leg poses per behaviour state (see AnimateLoadedModel)
            }
            else if (clips.Length > 0)
            {
                _loadedAnimation.Play(clips[0]);   // any other model: its first clip loops in flight
            }
        }

        GD.Print($"fly model loaded: {ModelPath} (extent {extent:F2} m → scale {k:F3}, clips: {(_loadedAnimation is null ? "none" : string.Join("/", _loadedAnimation.GetAnimationList()))}, wing nodes: {(_blurredWings is not null ? "yes" : "no")})");
        return true;
    }

    private static AnimationPlayer? FindAnimationPlayer(Node node)
    {
        foreach (Node child in node.GetChildren())
        {
            if (child is AnimationPlayer ap)
            {
                return ap;
            }

            if (FindAnimationPlayer(child) is { } found)
            {
                return found;
            }
        }

        return null;
    }

    private static Aabb MergedAabb(Node node)
    {
        Aabb? acc = null;
        void Visit(Node n)
        {
            if (n is MeshInstance3D mi)
            {
                Aabb a = mi.Transform * mi.GetAabb();
                acc = acc is null ? a : acc.Value.Merge(a);
            }

            foreach (Node c in n.GetChildren())
            {
                Visit(c);
            }
        }

        Visit(node);
        return acc ?? new Aabb(Vector3.Zero, Vector3.One);
    }

    private bool _hasPoseClips;
    private Behaviour.State _clipState = Behaviour.State.Flying;

    /// <summary>
    /// Drives the rigged model's leg clips from the behaviour state — <c>flight</c> (fore and
    /// middle legs retracted, hind legs trailing), <c>landing_pose</c> (forelegs reaching) while
    /// descending onto sugar, <c>standing</c> once landed and while feeding, <c>takeoff</c> then
    /// <c>flight</c> when leaving — plus the feeding dip and pump on the whole body, and the rowing
    /// wing beat (<see cref="BlurredWings"/>; the clips leave the wing nodes alone).
    /// </summary>
    private void AnimateLoadedModel(float dt, bool grounded)
    {
        if (_loadedModel is null)
        {
            return;
        }

        Behaviour.State state = _behaviour!.Current;
        bool feeding = state == Behaviour.State.Feeding;
        _loadedDip += ((feeding ? 1f : 0f) - _loadedDip) * MathF.Min(1f, dt * 3f);
        float pump = feeding ? 0.04f * MathF.Sin(Time.GetTicksMsec() * 0.012f) : 0f;
        _loadedModel.Rotation = new Vector3(-0.35f * _loadedDip + pump, Mathf.DegToRad(ModelYawDegrees), 0);

        if (_loadedAnimation is not null && _hasPoseClips && state != _clipState)
        {
            _clipState = state;
            switch (state)
            {
                case Behaviour.State.Landing:
                    _loadedAnimation.Play("landing_pose", 0.15);
                    break;
                case Behaviour.State.Landed:
                case Behaviour.State.Feeding:
                    _loadedAnimation.Play("standing", 0.25);
                    break;
                case Behaviour.State.TakeOff:
                    _loadedAnimation.Play("takeoff", 0.1);
                    _loadedAnimation.Queue("flight");
                    break;
                default:
                    if (_loadedAnimation.CurrentAnimation != "takeoff")
                    {
                        _loadedAnimation.Play("flight", 0.2);
                    }

                    break;
            }
        }

        _blurredWings?.Animate(dt, !grounded);
    }

    /// <summary>
    /// Third-person camera with inertia: it follows a low-passed heading (time constant ≈ 0.7 s),
    /// so tumbles and escape turns are seen as the fly turning, not the camera; the position
    /// eases toward a point behind and above the fly, shifted so the fly sits right of the panels.
    /// </summary>
    /// <summary>Camera modes, toggled with <c>C</c>.</summary>
    public enum CameraMode
    {
        /// <summary>Third-person with inertia: a low-passed heading, an offset to the side, orbits to the front when landed.</summary>
        Chase,

        /// <summary>FPV-like: locked behind the fly on its own heading, close and low; we never see her front.</summary>
        Follow,
    }

    /// <summary>Current camera mode.</summary>
    public CameraMode Camera { get; set; } = System.Environment.GetEnvironmentVariable("DOTFLY_CAMERA") == "follow" ? CameraMode.Follow : CameraMode.Chase;   // DOTFLY_CAMERA=follow starts in follow mode

    public override void _UnhandledKeyInput(InputEvent @event)
    {
        if (Primary && @event is InputEventKey { Pressed: true, Keycode: Key.C })
        {
            Camera = Camera == CameraMode.Chase ? CameraMode.Follow : CameraMode.Chase;
        }
    }

    private void UpdateChaseCamera(float dt, bool grounded)
    {
        Camera3D cam = _arena!.Chase;
        if (Camera == CameraMode.Follow)
        {
            UpdateFollowCamera(dt, grounded);
        }
        else
        {
            UpdateChaseCameraInertial(dt, grounded);
        }

        // Anything between the fly and the camera turns translucent while it is in the way.
        _occlusion.Update(GetWorld3D().DirectSpaceState, GlobalPosition, cam.GlobalPosition, GetRid(), dt);
    }

    /// <summary>
    /// Third-person with inertia: a low-passed heading (τ ≈ 0.7 s) so turns read as the fly
    /// turning, an offset to the side so she sits right of the panels, 9 body lengths back; orbits
    /// to her front when she lands.
    /// </summary>
    private void UpdateChaseCameraInertial(float dt, bool grounded)
    {
        Vector3 fwd = -GlobalTransform.Basis.Z;
        fwd.Y = 0;
        fwd = fwd.Normalized();
        if (grounded)
        {
            fwd = fwd.Rotated(Vector3.Up, 1.9f);   // on the ground, orbit to the front-side: the proboscis is the point
        }

        _camHeading = (_camHeading + (fwd - _camHeading) * MathF.Min(1f, dt * 1.4f)).Normalized();
        float targetDistance = (grounded ? 7.5f : 13.5f) * BodyLength;   // body lengths, so any fly size frames the same
        _camDistance += (targetDistance - _camDistance) * MathF.Min(1f, dt * 1.5f);
        // The camera rides rigidly with the fly's position (no translation lag — at a 3 cm body a
        // first-order lag of v/rate would be metres); only its offset around the fly is smoothed.
        float up = grounded ? 3.5f : 2.2f;   // landed: look down over the plate rim and the syrup
        Vector3 desiredOffset = -_camHeading * _camDistance + Vector3.Up * (up * BodyLength);   // on her axis: she stays centred
        _camOffset += (desiredOffset - _camOffset) * MathF.Min(1f, dt * 4f);
        _camPos = GlobalPosition + _camOffset;
        _camPos.Y = Mathf.Clamp(_camPos.Y, 0.4f * BodyLength, _arena!.Height - 0.2f);
        _camLook = GlobalPosition + Vector3.Up * (0.3f * BodyLength);   // at her (slightly above her centre), so she is centred
        Camera3D cam = _arena.Chase;
        cam.GlobalPosition = _camPos;
        cam.LookAt(_camLook, Vector3.Up);
        CentreInVisibleArea(cam, _camDistance);
    }

    /// <summary>
    /// The panels cover the left of the window, so "centred" means the middle of what is left. A
    /// lens shift does it: an off-axis frustum slides the image sideways while the camera itself
    /// stays put (a camera translation would put it beside her trail and show her flank in turns).
    /// </summary>
    private void CentreInVisibleArea(Camera3D cam, float distance)
    {
        Vector2 vs = GetViewport().GetVisibleRect().Size;
        float panelFraction = _eyeView is null ? 0f : _eyeView.Size.X / vs.X;           // 0 when there are no panels
        float nearHeight = 2f * cam.Near * MathF.Tan(Mathf.DegToRad(FollowFovDegrees) / 2f);
        float nearHalfWidth = nearHeight / 2f * (vs.X / vs.Y);
        cam.Projection = Camera3D.ProjectionType.Frustum;
        cam.KeepAspect = Camera3D.KeepAspectEnum.Height;
        cam.Size = nearHeight;
        cam.FrustumOffset = new Vector2(-panelFraction * nearHalfWidth, 0f);   // she projects to the centre of the visible area
        cam.HOffset = 0f;
    }

    private const float FollowFovDegrees = 65f;

    /// <summary>
    /// FPV-like: the camera rides on the fly's own orientation — yaw and pitch, smoothed by a
    /// quaternion slerp (τ ≈ 0.08 s) so tumbles and pitch changes are followed, not jolted — 2.6
    /// body lengths behind and a little above her, looking where she looks. Never her front.
    /// </summary>
    /// <summary>
    /// Trail camera: it slides along the path she actually flew, <c>FollowDistance</c> body lengths
    /// behind her along that trail, looking at her — so in a turn it goes through the same bend she
    /// did and keeps her rear in view instead of pivoting to her side. When she reverses and flies
    /// at the camera (a wall bump, a U-turn) the offset blends to "behind her heading", so her
    /// front is never shown. The offset around her is smoothed (τ ≈ 0.1 s); her position is not,
    /// so there is no translation lag.
    /// </summary>
    private void UpdateFollowCamera(float dt, bool grounded)
    {
        const float followDistance = 7.8f;
        float d = (grounded ? 6.0f : followDistance) * BodyLength;
        RecordTrail();
        Vector3 up = Vector3.Up;
        Vector3 forward = -GlobalTransform.Basis.Z;
        forward.Y = 0;
        forward = forward.LengthSquared() > 1e-6f ? forward.Normalized() : new Vector3(0, 0, -1);

        Vector3 offsetTrail = TrailPointBehind(d) - GlobalPosition;
        if (offsetTrail.LengthSquared() < (0.3f * d) * (0.3f * d))
        {
            offsetTrail = -forward * d;   // not enough trail yet (start, or she has been sitting still)
        }

        Vector3 offsetBehind = -forward * d;
        // Is the current camera in front of her? dot = 1 straight behind, −1 straight ahead.
        Vector3 camDir = _camOffset.LengthSquared() > 1e-8f ? _camOffset.Normalized() : -forward;
        float dot = forward.Dot(-camDir);
        float w = Mathf.SmoothStep(-0.2f, 0.4f, dot);            // 1 = trust the trail, 0 = swing behind her heading
        Vector3 offsetTarget = offsetBehind.Lerp(offsetTrail, w) + up * (0.5f * BodyLength);
        _camOffset += (offsetTarget - _camOffset) * MathF.Min(1f, dt * 10f);
        _camPos = GlobalPosition + _camOffset;
        _camPos.Y = Mathf.Clamp(_camPos.Y, 0.3f * BodyLength, _arena!.Height - 0.2f);
        _camLook = GlobalPosition + up * (0.3f * BodyLength);
        Camera3D cam = _arena.Chase;
        cam.GlobalPosition = _camPos;
        cam.LookAt(_camLook, up);
        _camDistance = d;
        CentreInVisibleArea(cam, _camDistance);
    }

    private readonly List<(Vector3 Pos, float S)> _trail = [];   // her path, with cumulative length
    private float _trailLength;

    private void RecordTrail()
    {
        Vector3 p = GlobalPosition;
        if (_trail.Count == 0)
        {
            _trail.Add((p, 0));
            return;
        }

        (Vector3 last, float s) = _trail[^1];
        float step = (p - last).Length();
        if (step < 0.02f * BodyLength)
        {
            return;
        }

        _trailLength = s + step;
        _trail.Add((p, _trailLength));
        while (_trail.Count > 2 && _trailLength - _trail[1].S > 40f * BodyLength)
        {
            _trail.RemoveAt(0);   // keep ~40 body lengths of path
        }
    }

    /// <summary>The point on her trail <paramref name="distance"/> metres of path behind her (interpolated).</summary>
    private Vector3 TrailPointBehind(float distance)
    {
        if (_trail.Count == 0)
        {
            return GlobalPosition;
        }

        float target = _trailLength - distance;
        for (int i = _trail.Count - 1; i > 0; i--)
        {
            (Vector3 a, float sa) = _trail[i - 1];
            (Vector3 b, float sb) = _trail[i];
            if (sa <= target)
            {
                float t = sb > sa ? (target - sa) / (sb - sa) : 0f;
                return a.Lerp(b, Math.Clamp(t, 0f, 1f));
            }
        }

        return _trail[0].Pos;
    }

}
