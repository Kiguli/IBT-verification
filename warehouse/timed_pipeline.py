"""Time the warehouse BFS-VI pipeline on the FULL model in a fresh process.

    python timed_pipeline.py hand        # case3.build with the hand-compiled outcomes() (reference)
    python timed_pipeline.py td_dropin   # case3.build's loop, the compilation as the edge oracle (paper's timings)
    python timed_pipeline.py td_compile  # tickdist.compile_imc (the library's Phase 1), then numpy VI

Phases are timed exactly where case3.run times them: t_bfs covers only the BFS loop (successor
enumeration + state indexing + edge memo); the (n, K) array fill is timed separately; each robust
VI sweep (case3.robust_vi, H=280) is timed alone; the midpoint point-VI is run afterwards because
case3.run takes its peak after it. Peak memory: psutil peak_wset (bytes/1e6 = MB), the branch
case3._peak_rss_mb takes on Windows, sampled after every phase.

Appends one JSON line per run to timed_runs.jsonl in this folder.
"""
import json
import os
import sys
import time
from collections import deque

import exp_common as ec

import numpy as np
import case3
import tickdist as td
from tickdist_case3_check import case3_ibt

CERT_LO = 0.26794320082507844
CERT_HI = 0.9743589226308836
CERT_MID = 0.6776746417641166


def build_with(cfg, oracle):
    """case3.build, verbatim except that the successor function is a parameter."""
    s0 = (cfg.HOME[0], cfg.HOME[1], 0, cfg.BATMAX, 0, 0)
    idx = {s0: 0}; order = [s0]; q = deque([s0])
    edges = []
    t0 = time.perf_counter()
    while q:
        s = q.popleft()
        outs = oracle(s)
        for (t, fac) in outs:
            if t not in idx:
                idx[t] = len(order); order.append(t); q.append(t)
        edges.append([(idx[t], fac) for (t, fac) in outs])
    t_bfs = time.perf_counter() - t0
    return order, idx, edges, t_bfs


def arrays_from_edges(order, edges, cfg):
    n = len(order)
    K = max(len(e) for e in edges)
    succ = np.zeros((n, K), dtype=np.int32)
    lo = np.zeros((n, K)); hi = np.zeros((n, K))
    for i, es in enumerate(edges):
        for j, (t, fac) in enumerate(es):
            l, h = case3.edge_interval(fac)
            succ[i, j] = t; lo[i, j] = l; hi[i, j] = h
    goal = np.array([s[2] == cfg.NSTAGE - 1 for s in order])
    return succ, lo, hi, goal


def main(mode):
    rec = {'mode': mode, 'peak_mb': {}}
    rec['peak_mb']['after_imports'] = round(ec.peak_wset_mb(), 1)
    cfg = case3.Config(small=False)
    tree = case3_ibt(cfg)
    LID = case3.LID
    t_all = time.perf_counter()

    if mode == 'hand':
        order, idx, edges, t_bfs = build_with(cfg, lambda s: case3.outcomes(cfg, s))
    elif mode == 'td_dropin':
        def oracle(s):
            return [(t, tuple((LID[nm], sg) for (nm, sg) in fac))
                    for (t, fac) in td.tickdist_full(tree, s, check=True).imc_edges()]
        order, idx, edges, t_bfs = build_with(cfg, oracle)
    elif mode == 'td_compile':
        s0 = (cfg.HOME[0], cfg.HOME[1], 0, cfg.BATMAX, 0, 0)
        t0 = time.perf_counter()
        imc = td.compile_imc(tree, s0)
        t_bfs = time.perf_counter() - t0
        order = imc.states
        rec['tickdist_calls'] = imc.calls
    else:
        raise SystemExit('unknown mode ' + mode)
    rec['peak_mb']['after_bfs'] = round(ec.peak_wset_mb(), 1)

    t0 = time.perf_counter()
    if mode == 'td_compile':
        n = imc.n()
        K = max(len(r) for r in imc.succ)
        succ = np.zeros((n, K), dtype=np.int32)
        lo = np.zeros((n, K)); hi = np.zeros((n, K))
        for i in range(n):
            for j, t in enumerate(imc.succ[i]):
                succ[i, j] = t; lo[i, j] = imc.lo[i][j]; hi[i, j] = imc.hi[i][j]
        goal = np.array([s[2] == cfg.NSTAGE - 1 for s in imc.states])
        # factor form for the midpoint sweep (leaf names -> case3 leaf ids)
        edges = [[(t, tuple((LID[nm], sg) for (nm, sg) in m)) for (t, m) in imc.edges[i]]
                 for i in range(n)]
    else:
        succ, lo, hi, goal = arrays_from_edges(order, edges, cfg)
    t_arrays = time.perf_counter() - t0
    n, K = succ.shape
    rec['peak_mb']['after_arrays'] = round(ec.peak_wset_mb(), 1)

    t0 = time.perf_counter(); Vlo = case3.robust_vi(succ, lo, hi, goal, cfg.H, True)
    t_lo = time.perf_counter() - t0
    rec['peak_mb']['after_vi_lower'] = round(ec.peak_wset_mb(), 1)
    t0 = time.perf_counter(); Vhi = case3.robust_vi(succ, lo, hi, goal, cfg.H, False)
    t_hi = time.perf_counter() - t0
    rec['peak_mb']['after_vi_upper'] = round(ec.peak_wset_mb(), 1)
    t_through_vi = time.perf_counter() - t_all

    mids = [case3.MID[l] for l in case3.LEAVES]
    P = case3.probs_for(edges, n, K, mids)
    t0 = time.perf_counter(); Vm = case3.point_vi(succ, P, goal, cfg.H)
    t_mid = time.perf_counter() - t0
    rec['peak_mb']['after_midpoint'] = round(ec.peak_wset_mb(), 1)

    lower, upper, mid = float(Vlo[0]), float(Vhi[0]), float(Vm[0])
    rec.update({
        'R': n, 'K': int(K), 'H': cfg.H, 'n_edges': int((lo + hi > 0).sum()),
        'lower': lower, 'upper': upper, 'midpoint': mid,
        'lower_bit_identical': lower == CERT_LO, 'upper_bit_identical': upper == CERT_HI,
        'lower_hex': lower.hex(), 'upper_hex': upper.hex(),
        'midpoint_bit_identical': mid == CERT_MID,
        't_bfs_s': round(t_bfs, 3), 't_arrays_s': round(t_arrays, 3),
        't_vi_lower_s': round(t_lo, 3), 't_vi_upper_s': round(t_hi, 3),
        't_vi_midpoint_s': round(t_mid, 3),
        't_bfs_plus_two_vi_s': round(t_bfs + t_lo + t_hi, 3),
        't_wall_through_upper_vi_s': round(t_through_vi, 3),
        'peak_rss_gb_after_two_vi': round(rec['peak_mb']['after_vi_upper'] / 1000, 2),
        'peak_rss_gb_as_case3_run': round(rec['peak_mb']['after_midpoint'] / 1000, 2),
        'peak_rss_method': 'psutil.Process().memory_info().peak_wset / 1e6 (MB), same branch as '
                           'case3._peak_rss_mb on Windows; process-wide peak, fresh process per run',
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'command': 'python timed_pipeline.py ' + mode,
    })
    print(json.dumps(rec))
    with open(os.path.join(ec.HERE, 'timed_runs.jsonl'), 'a') as fh:
        fh.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
