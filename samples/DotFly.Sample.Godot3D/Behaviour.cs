using System;
using Godot;

namespace DotFly.Sample.Godot3D;

/// <summary>
/// The decoder as a small state machine over the readout rates. Every rule here is the demo's
/// choice (labelled <c>Fixed</c>); the network supplies the rates. States: <c>Flying</c> (escape +
/// optomotor + odor search), <c>Landing</c> (descending onto a patch once odor is strong),
/// <c>Landed</c> (on the sugar: tarsal + labellar contact), <c>Feeding</c> (MN9 fires), <c>TakeOff</c>.
/// </summary>
public sealed class Behaviour
{
    /// <summary>Behaviour states.</summary>
    public enum State
    {
        Flying,
        Landing,
        Landed,
        Feeding,
        TakeOff,
    }

    private readonly FlyBrain3D.Decoder _d;
    private float _castSign = 1f;
    private float _tumbleLeft;          // seconds left of the current tumble
    private float _fallingFor;          // seconds the odor readout has been falling
    private float _sinceOdor = 1e6f;    // seconds since the odor readout was last above threshold
    private readonly Random _rng = new(7);
    private float _vertSign = 1f;       // current vertical search direction while tracking
    private float _vertLeft;            // seconds left of the current vertical cast (odor lost)
    private float _mn9Smooth;
    private float _pnSmooth, _windLSmooth, _windRSmooth;
    private float _roamClimb;           // vertical drift when there is no odor …
    private float _roamLeft;            // … re-drawn at random every few seconds
    private float _castLeft;            // seconds left of the current cast (odor lost)
    private int _castIndex;             // casts since the odor was lost (0 = the U-turn)
    private float _pnPrev;
    private float _pnTrend;
    private float _stateTime;
    private float _feedTime;

    /// <summary>Current state.</summary>
    public State Current { get; private set; } = State.Flying;

    /// <summary>Seconds spent in the current state.</summary>
    public float StateTime => _stateTime;

    /// <summary>Odor "detected" flag (DM1_lPN above the detection rate).</summary>
    public bool OdorDetected { get; private set; }

    /// <summary>Rate-of-change of the DM1_lPN readout (Hz/s, smoothed).</summary>
    public float OdorTrend => _pnTrend;

    /// <summary>Times the odor readout dropped below threshold, and times it came back within the memory window (for tuning).</summary>
    public int Losses { get; private set; }

    /// <inheritdoc cref="Losses"/>
    public int Recoveries { get; private set; }

    /// <summary>Odor lost recently: still searching where it was (casting), not cruising.</summary>
    public bool OdorMemory => !OdorDetected && _sinceOdor < _d.OdorMemorySeconds;

    /// <summary>Which decision each action came from, for the HUD.</summary>
    public string YawSource { get; private set; } = string.Empty;

    /// <summary>Creates the state machine with the decoder gains.</summary>
    public Behaviour(FlyBrain3D.Decoder decoder, float bodyLength = 0.36f)
    {
        _d = decoder;
        _bodyLength = bodyLength;
    }

    private readonly float _bodyLength;   // landing distances are in body lengths

    /// <summary>
    /// Computes the actions from the readouts. <paramref name="rates"/> is in <see cref="FlyBrain3D.GroupNames"/>
    /// order; <paramref name="overPatch"/> and <paramref name="altitude"/> come from the world.
    /// </summary>
    public (float YawDegPerSec, float ClimbMPerSec, float SpeedMPerSec) Step(float[] rates, bool overPatch, float altitude, float delta)
    {
        float dnp04L = rates[0], dnp04R = rates[1], gf = rates[2], hsL = rates[3], hsR = rates[4], vsL = rates[5], vsR = rates[6], h2L = rates[7], h2R = rates[8], pn = rates[9], mn9 = rates[10], windL = rates[11], windR = rates[12];
        _stateTime += delta;
        // MN9 is two cells read over 30 ms: 35 Hz is one spike per window. Feeding decisions use a
        // 0.3 s smoothed rate so a real but modest response is not missed.
        _mn9Smooth += (mn9 - _mn9Smooth) * MathF.Min(1f, delta / 0.3f);
        mn9 = _mn9Smooth;

        // Odor: detected when the DM1 projection neurons exceed the threshold; trend = smoothed d/dt.
        bool wasDetected = OdorDetected;
        // Readout smoothing: DM1_lPN is two cells over 30 ms (±33 Hz jumps) and each wind readout
        // one cell (0/75/100 Hz steps); the decisions below use τ = 0.15 s / 0.3 s versions.
        _pnSmooth += (pn - _pnSmooth) * MathF.Min(1f, delta / 0.15f);
        _windLSmooth += (windL - _windLSmooth) * MathF.Min(1f, delta / 0.3f);
        _windRSmooth += (windR - _windRSmooth) * MathF.Min(1f, delta / 0.3f);
        pn = _pnSmooth;
        windL = _windLSmooth;
        windR = _windRSmooth;
        OdorDetected = pn > _d.OdorDetectHz;
        if (wasDetected && !OdorDetected)
        {
            Losses++;
        }
        else if (!wasDetected && OdorDetected && _sinceOdor < _d.OdorMemorySeconds)
        {
            Recoveries++;
        }

        _sinceOdor = OdorDetected ? 0f : _sinceOdor + delta;

        // No altimeter: the fly does not know its height. Without odor it drifts up or down at
        // random (a new direction every few seconds; the room's floor and ceiling bound it); with
        // odor the vertical search below uses only the readout trend and its own last move.
        if ((_roamLeft -= delta) <= 0)
        {
            _roamLeft = 4f + 6f * (float)_rng.NextDouble();
            _roamClimb = (float)(_rng.NextDouble() * 2 - 1.2) * _d.RoamClimbBlPerSec;   // slightly more down than up: the sugar is low
        }

        float trend = (pn - _pnPrev) / MathF.Max(delta, 1e-3f);
        _pnTrend += (trend - _pnTrend) * MathF.Min(1f, delta * 4f);
        _pnPrev = pn;

        float escape = _d.EscapeYawGainDegPerSecPerHz * (dnp04L - dnp04R);
        // Optic flow: a yaw to the left puts front-to-back flow on the right eye (HS_R) and
        // back-to-front on the left (H2_L) — counter-turn right. Flying along a wall on the right
        // puts front-to-back flow on the right eye only, which HS alone cannot tell from a left
        // turn (the fly would steer into the wall); the centering term turns away from the side
        // with more front-to-back flow, and outweighs the optomotor term for one-sided flow.
        float optomotor = _d.OptomotorYawGainDegPerSecPerHz * ((hsR + h2L) - (hsL + h2R))
                        - _d.CenteringYawGainDegPerSecPerHz * (hsR - hsL);
        // Altitude: the VS optic-flow readout nudges it; the flight model holds a cruise altitude
        // (lower while tracking odor, because the plume hugs the surface).
        bool tracking = OdorDetected || OdorMemory;
        float bl = _bodyLength;   // decoder speeds are body lengths per second
        float climbHold = (_d.OptomotorClimbGainBlPerSecPerHz * (vsL + vsR) / 2f + _roamClimb) * bl;
        float speed = (_d.IdleSpeedBlPerSec + _d.GfSpeedGainBlPerSecPerHz * gf) * bl;

        switch (Current)
        {
            case State.Flying:
            {
                float yaw = escape + optomotor;
                YawSource = MathF.Abs(escape) > MathF.Abs(optomotor) ? "escape (DNp04)" : "optic flow (HS/H2)";
                float climb = climbHold;
                if (tracking)
                {
                    speed = MathF.Min(speed, _d.TrackingSpeedBlPerSec * bl);   // do not overshoot the tube
                }

                if (OdorDetected)
                {
                    _castIndex = 0;
                    _castLeft = 0;
                    climb = _vertSign * _d.VerticalCastBlPerSec * bl;   // vertical klinotaxis: keep going while it pays
                    // Run-and-tumble (decoder rule): run straight while the odor readout rises or
                    // holds; when it has been falling for a moment, tumble — a decisive turn of
                    // ~120° over 0.8 s, alternating sides — then run again. Near the source (PN
                    // high) slow down and descend; land when a patch is within reach.
                    if (_tumbleLeft > 0)
                    {
                        _tumbleLeft -= delta;
                        yaw += _castSign * _d.CastYawDegPerSec;
                        YawSource = "tumble (odor fading)";
                    }
                    else
                    {
                        _fallingFor = _pnTrend < -25f ? _fallingFor + delta : 0f;   // a real fall, not readout noise
                        if (_fallingFor > 0.25f)
                        {
                            // The readout fell: the tube was left sideways or vertically, and one
                            // glomerulus cannot say which — so tumble, and reverse the vertical
                            // direction half of the time.
                            _tumbleLeft = _d.TumbleSeconds;
                            _castSign = -_castSign;
                            if (_rng.NextDouble() < 0.5)
                            {
                                _vertSign = -_vertSign;
                            }

                            _fallingFor = 0;
                        }
                        else
                        {
                            // Surge upwind (decoder rule; the fly's own strategy): the tubes run
                            // downwind from the sugar, so upwind along one leads to it. wind_L
                            // high = air from the left = upwind is to the left = turn left.
                            yaw -= _d.SurgeYawGainDegPerSecPerHz * (windL - windR);
                            YawSource = "surge upwind (odor)";
                        }
                    }

                    if (pn > _d.OdorNearHz)
                    {
                        climb = MathF.Min(climb, -17f * bl);
                        speed = MathF.Min(speed, 47f * bl);
                        if (overPatch && altitude < 4f * _bodyLength)
                        {
                            Transition(State.Landing);
                        }
                    }
                }
                else if (OdorMemory)
                {
                    // Lost the tube (decoder rule, the moth/fly "cast"): the odor was just here,
                    // so first a U-turn back into it, then zigzags of alternating sign and growing
                    // duration, slow and low, until the readout comes back or the memory expires.
                    if (_castLeft <= 0)
                    {
                        _castLeft = _castIndex == 0 ? 180f / _d.CastYawDegPerSec : MathF.Min(1.6f, 0.5f + 0.3f * _castIndex);
                        _castSign = _castIndex == 0 ? _castSign : -_castSign;
                        _castIndex++;
                    }

                    _castLeft -= delta;
                    yaw += _castSign * _d.CastYawDegPerSec;
                    speed = MathF.Min(speed, _d.CastSpeedBlPerSec * bl);   // stay near where the odor was
                    if ((_vertLeft -= delta) <= 0)
                    {
                        _vertLeft = 0.8f + 1.2f * (float)_rng.NextDouble();
                        _vertSign = _rng.NextDouble() < 0.5 ? -1f : 1f;   // and cast vertically at random
                    }

                    climb = _vertSign * _d.VerticalCastBlPerSec * bl;
                    _tumbleLeft = 0;
                    _fallingFor = 0;
                    YawSource = _castIndex == 1 ? "U-turn (odor lost)" : $"cast {_castIndex} (odor lost)";
                }

                return (yaw, climb, speed);
            }

            case State.Landing:
                YawSource = "landing";
                if (!overPatch)
                {
                    Transition(State.Flying);
                }
                else if (altitude <= 1.4f * _bodyLength)
                {
                    Transition(State.Landed);
                }

                return (0f, -40f * bl, 20f * bl);

            case State.Landed:
                YawSource = "landed";
                if (mn9 > _d.FeedMn9Hz)
                {
                    Transition(State.Feeding);
                }
                else if (_stateTime > 6f)
                {
                    Transition(State.TakeOff);   // nothing tasted: leave
                }

                return (0f, 0f, 0f);

            case State.Feeding:
                YawSource = "feeding (MN9)";
                _feedTime += delta;
                if (mn9 < _d.FeedMn9Hz * 0.5f && _stateTime > 1f || _stateTime > 8f)
                {
                    Transition(State.TakeOff);
                }

                return (0f, 0f, 0f);

            case State.TakeOff:
                YawSource = "take-off";
                if (_stateTime > 1.5f)
                {
                    Transition(State.Flying);
                }

                return (60f, 50f * bl, 85f * bl);

            default:
                return (0f, 0f, 0f);
        }
    }

    /// <summary>Total seconds spent feeding.</summary>
    public float FeedTime => _feedTime;

    private void Transition(State next)
    {
        Current = next;
        _stateTime = 0;
    }
}
