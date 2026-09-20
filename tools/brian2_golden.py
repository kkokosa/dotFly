#!/usr/bin/env python
"""Generate Brian2 golden traces for dotFly's reference backend.

Runs the Shiu et al. 2024 equations (model.py @ 91bdd1e7) with Brian2 2.5.1 on a small
sub-graph of the FlyWire v630 connectivity, driving chosen neurons with *explicit* input events
(a SpikeGeneratorGroup + `v += w_syn*f_poi` synapse with zero delay) instead of PoissonInput, so
that randomness is out of the loop and the trace is a pure function of the equations.

Output (two files per fixture, committed under tests/DotFly.Tests.Unit/Fixtures/):
  <name>.json  metadata, neurons (body ids), edges (pre, post, signed count), injections
               (step, neuron), refractory-zero neurons, recorded neurons, all spikes (step, neuron)
  <name>.f64   little-endian float64 traces: for each recorded step k in 0..T-1, for each
               recorded neuron r, v then g — i.e. shape (T, R, 2), the state *before* step k's
               update (Brian2 StateMonitor semantics) = dotFly state when Step == k.

This is an offline tool. It is never part of the .NET build.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from brian2 import (Hz, NeuronGroup, Network, SpikeGeneratorGroup, SpikeMonitor, StateMonitor,
                    Synapses, defaultclock, ms, mV, prefs, second)

SUGAR_V630 = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9_V630 = 720575940660219265

PARAMS = dict(v_0=-52 * mV, v_rst=-52 * mV, v_th=-45 * mV, t_mbr=20 * ms, tau=5 * ms,
              t_rfc=2.2 * ms, t_dly=1.8 * ms, w_syn=.275 * mV, f_poi=250)
EQS = '''
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau               : volt (unless refractory)
rfc                            : second
'''


def load_v630(data: Path):
    comp = pd.read_csv(data / '2023_03_23_completeness_630_final.csv', index_col=0)
    con = pd.read_parquet(data / '2023_03_23_connectivity_630_final.parquet')
    ids = comp.index.to_numpy(dtype=np.int64)
    return ids, con


def subgraph(ids, con, seed_ids, hops, min_weight):
    """Neurons reachable from seed_ids within `hops` forward hops over |w| >= min_weight (plus seeds)."""
    id2i = {int(v): i for i, v in enumerate(ids)}
    pre = con['Presynaptic_Index'].to_numpy()
    post = con['Postsynaptic_Index'].to_numpy()
    w = con['Excitatory x Connectivity'].to_numpy()
    strong = np.abs(w) >= min_weight
    order = np.argsort(pre, kind='stable')
    pre_s, post_s, strong_s = pre[order], post[order], strong[order]
    starts = np.searchsorted(pre_s, np.arange(len(ids) + 1))
    keep = set(id2i[s] for s in seed_ids)
    frontier = list(keep)
    for _ in range(hops):
        nxt = []
        for s in frontier:
            for e in range(starts[s], starts[s + 1]):
                if strong_s[e] and post_s[e] not in keep:
                    keep.add(int(post_s[e]))
                    nxt.append(int(post_s[e]))
        frontier = nxt
    sel = np.array(sorted(keep), dtype=np.int64)
    local = {int(g): i for i, g in enumerate(sel)}
    mask = np.isin(pre, sel) & np.isin(post, sel)
    edges = sorted((local[int(a)], local[int(b)], int(c)) for a, b, c in zip(pre[mask], post[mask], w[mask]) if c != 0)
    return sel, edges, local


def run(name, body_ids, edges, injections, rfc_zero, record, steps, out: Path, dt_ms=0.1):
    """Simulate the sub-graph with Brian2 and write <name>.json + <name>.f64."""
    prefs.codegen.target = 'numpy'
    defaultclock.dt = dt_ms * ms
    n = len(body_ids)

    neu = NeuronGroup(n, EQS, method='linear', threshold='v > v_th', reset='v = v_rst; g = 0*mV',
                      refractory='rfc', namespace=PARAMS, name='neu')
    neu.v = PARAMS['v_0']
    neu.g = 0 * mV
    neu.rfc = PARAMS['t_rfc']
    for i in rfc_zero:
        neu.rfc[i] = 0 * ms

    syn = Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=PARAMS['t_dly'], name='syn')
    if edges:
        pre = np.array([e[0] for e in edges]); post = np.array([e[1] for e in edges])
        syn.connect(i=pre, j=post)
        syn.w = np.array([e[2] for e in edges], dtype=float) * PARAMS['w_syn']

    objs = [neu, syn]
    if injections:
        inj_idx = np.array([j for _, j in injections]); inj_t = np.array([s for s, _ in injections]) * dt_ms * ms
        gen = SpikeGeneratorGroup(n, inj_idx, inj_t, name='gen')
        inj = Synapses(gen, neu, on_pre='v += w_syn*f_poi', delay=0 * ms, namespace=PARAMS, name='inj')
        inj.connect(j='i')
        objs += [gen, inj]

    spk = SpikeMonitor(neu, name='spk')
    st = StateMonitor(neu, ['v', 'g'], record=list(record), name='st')
    net = Network(*objs, spk, st)
    net.run(steps * dt_ms * ms)

    abstract = neu.state_updater.abstract_code
    spikes = sorted((int(round(float(t / ms) / dt_ms)), int(i)) for i, t in zip(spk.i, spk.t))

    meta = dict(
        name=name, dtMs=dt_ms, steps=steps, brian2=__import__('brian2').__version__,
        model='shiu2024', params={k: float(v) if k == 'f_poi' else str(v) for k, v in PARAMS.items()},
        stateUpdaterCode=abstract,
        bodyIds=[int(b) for b in body_ids], edges=edges,
        injections=[[int(s), int(j)] for s, j in injections],
        refractoryZero=[int(i) for i in rfc_zero],
        recorded=[int(i) for i in record],
        spikes=[[s, i] for s, i in spikes],
    )
    (out / f'{name}.json').write_text(json.dumps(meta, separators=(',', ':')), encoding='utf-8')

    v = np.asarray(st.v / mV)  # shape (R, T)
    g = np.asarray(st.g / mV)
    assert v.shape == (len(record), steps), v.shape
    traces = np.empty((steps, len(record), 2), dtype='<f8')
    traces[:, :, 0] = v.T
    traces[:, :, 1] = g.T
    (out / f'{name}.f64').write_bytes(traces.tobytes())
    print(f'{name}: {n} neurons, {len(edges)} edges, {len(injections)} injections, {len(spikes)} spikes, {steps} steps')


def bernoulli_injections(rng, neurons, steps, rate_hz, dt_ms):
    p = rate_hz * dt_ms * 1e-3
    return [(s, j) for s in range(steps) for j in neurons if rng.random() < p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('.data/shiu2024'))
    ap.add_argument('--out', type=Path, default=Path('tests/DotFly.Tests.Unit/Fixtures'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260918)

    # 1. Tiny hand-checkable chain: 0 -> 1 (200 contacts), 1 -> 2 (-50), 2 -> 0 (10).
    run('chain3', [1, 2, 3], [(0, 1, 200), (1, 2, -50), (2, 0, 10)],
        injections=[(0, 0), (40, 0), (41, 0)], rfc_zero=[], record=[0, 1, 2], steps=600, out=args.out)

    # 2. Sugar circuit from v630: sugar GRNs + 2 forward hops over strong edges, 150 Hz-like
    #    Bernoulli input on the sugar neurons for 100 ms; record a handful of neurons incl. MN9 if present.
    ids, con = load_v630(args.data)
    sel, edges, local = subgraph(ids, con, SUGAR_V630, hops=2, min_weight=25)
    sugar_local = [local[__import__('numpy').where(ids == s)[0][0]] for s in SUGAR_V630]
    steps = 1000
    inj = bernoulli_injections(rng, sugar_local, steps, 150.0, 0.1)
    mn9 = np.where(ids == MN9_V630)[0][0]
    record = sorted(set(sugar_local[:3] + ([local[mn9]] if mn9 in local else []) + list(rng.choice(len(sel), 6, replace=False))))
    run('sugar_2hop', [int(ids[i]) for i in sel], edges, inj, rfc_zero=sugar_local, record=record, steps=steps, out=args.out)

    # 3. Larger: 3 hops over weaker edges, 300 ms; records 12 neurons.
    sel, edges, local = subgraph(ids, con, SUGAR_V630, hops=3, min_weight=15)
    sugar_local = [local[np.where(ids == s)[0][0]] for s in SUGAR_V630]
    steps = 3000
    inj = bernoulli_injections(rng, sugar_local, steps, 150.0, 0.1)
    mn9 = np.where(ids == MN9_V630)[0][0]
    record = sorted(set(sugar_local[:2] + ([local[mn9]] if mn9 in local else []) + [int(x) for x in rng.choice(len(sel), 10, replace=False)]))
    run('sugar_3hop', [int(ids[i]) for i in sel], edges, inj, rfc_zero=sugar_local, record=record, steps=steps, out=args.out)


if __name__ == '__main__':
    main()
