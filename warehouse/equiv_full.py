"""Edge-for-edge comparison of the compilation (tickdist) against case3.outcomes on EVERY state
reachable in the FULL warehouse configuration (push policy), plus array-level and compiled-IMC-level bit checks.

Writes equiv_full.json in this folder.
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
from tickdist_case3_check import case3_ibt, handwritten, compiled


def main():
    cfg = case3.Config(small=False)
    tree = case3_ibt(cfg)
    wf = td.check_wellformed(tree, strict=False)
    s0 = (cfg.HOME[0], cfg.HOME[1], 0, cfg.BATMAX, 0, 0)

    # 1. reachable set by BFS with the hand-compiled successor function (case3.build order)
    t0 = time.perf_counter()
    order, idx, edges_h, succ_h, lo_h, hi_h, goal_h, _ = case3.build(cfg)
    t_build = time.perf_counter() - t0
    n = len(order)

    # 2. edge-for-edge comparison on every reachable state
    t0 = time.perf_counter()
    n_identical = 0
    n_edges_h = n_edges_t = 0
    n_factors = 0
    mism = []
    order_only = 0
    for s in order:
        a = handwritten(cfg, s)
        b = compiled(tree, s)            # tickdist_full(tree, s, check=True).imc_edges()
        n_edges_h += len(a)
        n_edges_t += len(b)
        n_factors += sum(len(f) for (_t, f) in a)
        if a == b:
            n_identical += 1
        else:
            if sorted(map(str, a)) == sorted(map(str, b)):
                order_only += 1
            if len(mism) < 10:
                mism.append({'state': str(s), 'handwritten': [str(x) for x in a],
                             'tickdist': [str(x) for x in b]})
    t_cmp = time.perf_counter() - t0

    # 3. BFS with the compilation as the oracle: same states in the same order? same arrays bit for bit?
    LID = case3.LID
    t0 = time.perf_counter()
    idx_t = {s0: 0}; order_t = [s0]; q = deque([s0]); edges_t = []
    while q:
        s = q.popleft()
        outs = [(t, tuple((LID[nm], sg) for (nm, sg) in fac))
                for (t, fac) in td.tickdist_full(tree, s, check=True).imc_edges()]
        for (t, fac) in outs:
            if t not in idx_t:
                idx_t[t] = len(order_t); order_t.append(t); q.append(t)
        edges_t.append([(idx_t[t], fac) for (t, fac) in outs])
    t_bfs_t = time.perf_counter() - t0
    same_order = order_t == order
    same_edges = edges_t == edges_h
    K = max(len(e) for e in edges_t)
    succ_t = np.zeros((len(order_t), K), dtype=np.int32)
    lo_t = np.zeros((len(order_t), K)); hi_t = np.zeros((len(order_t), K))
    for i, es in enumerate(edges_t):
        for j, (t, fac) in enumerate(es):
            l, h = case3.edge_interval(fac)
            succ_t[i, j] = t; lo_t[i, j] = l; hi_t[i, j] = h
    arrays_equal = (succ_t.shape == succ_h.shape and np.array_equal(succ_t, succ_h)
                    and np.array_equal(lo_t, lo_h) and np.array_equal(hi_t, hi_h))

    # 4. the library's own compile_imc (interval recursion: imul / capped Minkowski sum) -> its
    #    per-successor intervals, compared bit for bit with case3's per-edge intervals.
    t0 = time.perf_counter()
    imc = td.compile_imc(tree, s0)
    t_compile = time.perf_counter() - t0
    cmp_same_order = imc.states == order
    n_merged = sum(1 for i in range(imc.n()) if len(imc.succ[i]) != len(edges_h[i]))
    ivl_equal = True
    first_bad = None
    if cmp_same_order and n_merged == 0:
        for i in range(imc.n()):
            row_succ = [t for (t, _f) in edges_h[i]]
            if imc.succ[i] != row_succ:
                ivl_equal = False; first_bad = ('succ', i); break
            for j in range(len(row_succ)):
                if imc.lo[i][j] != lo_h[i, j] or imc.hi[i][j] != hi_h[i, j]:
                    ivl_equal = False; first_bad = ('ivl', i, j, imc.lo[i][j], lo_h[i, j],
                                                    imc.hi[i][j], hi_h[i, j])
                    break
            if not ivl_equal:
                break
    else:
        ivl_equal = False

    res = {
        'what': 'the compilation (tickdist_full(tree, s, check=True).imc_edges()) vs hand-compiled '
                'case3.outcomes(cfg, s), FULL config, push policy, every reachable state; '
                'exact equality of the ordered list of (successor, ((leaf, sign), ...)) pairs',
        'config': 'full', 'policy': cfg.policy,
        'tree_leaf_wellformed': wf['ok'],
        'leaf_ids_used': sorted(td.leaf_boxes(tree)),
        'n_reachable_states': n,
        'n_states_checked': n,
        'n_states_identical': n_identical,
        'n_states_mismatch': n - n_identical,
        'n_states_order_only_mismatch': order_only,
        'n_edges_handwritten': n_edges_h,
        'n_edges_tickdist': n_edges_t,
        'n_leaf_factors_compared': n_factors,
        'mismatch_examples': mism,
        'seconds_compare': round(t_cmp, 2),
        'seconds_handwritten_build_incl_arrays': round(t_build, 2),
        'tickdist_bfs': {'same_states_same_order': same_order, 'same_edge_lists': same_edges,
                         'n_states': len(order_t),
                         'arrays_succ_lo_hi_bit_identical': bool(arrays_equal),
                         'K': int(K), 'seconds_bfs_loop': round(t_bfs_t, 2)},
        'tickdist_compile_imc': {'n_states': imc.n(), 'same_states_same_order': cmp_same_order,
                                 'rows_with_merged_parallel_edges': n_merged,
                                 'per_successor_intervals_bit_identical_to_case3': ivl_equal,
                                 'first_difference': first_bad,
                                 'tickdist_calls': imc.calls,
                                 'seconds': round(t_compile, 2)},
        'provenance': ec.provenance(),
        'command': 'python equiv_full.py',
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    out = os.path.join(ec.HERE, 'equiv_full.json')
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
