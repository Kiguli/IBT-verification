"""The static-calibration family at the corners of the leaf box.

The certificate brackets every leaf assignment theta in Theta(T). This script evaluates
P_{T_theta} by point VI at each of the 2^L = 128 corners of the leaf box and reports
the range over the 128 corners. Every corner is a leaf assignment, so that range lies
inside the certified bracket, and its distance to the bracket is an upper bound on the
conservatism of the interval relaxation for fixed leaf assignments. It is not claimed
to be the exact range over Theta(T): over H = 280 ticks a leaf's value enters a path
once for every tick in which that leaf fires, so the reachability probability need
not be multi-affine in the leaf vector, and its extremes may lie inside the box.

Writes results_case3_corners.json (reads results_case3_full.json). Run: python corners.py
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import case3 as m  # noqa: E402


def main():
    cfg = m.Config(small=False)
    order, idx, edges, succ, lo, hi, goal, t_bfs = m.build(cfg)
    n, K = succ.shape
    fid, fsg, mask = m.factor_arrays(edges, n, K)

    L = len(m.LEAVES)
    best, worst = -1.0, 2.0
    argmin = argmax = None
    t0 = time.perf_counter()
    for c, bits in enumerate(itertools.product((0, 1), repeat=L), 1):
        vals = [m.IVL[leaf][b] for leaf, b in zip(m.LEAVES, bits)]
        P = m.probs_for_vec(fid, fsg, mask, vals)
        v = float(m.point_vi(succ, P, goal, cfg.H)[0])
        if v > best:
            best, argmax = v, bits
        if v < worst:
            worst, argmin = v, bits
        if c % 16 == 0:
            print(f'  {c}/{2**L} corners, {time.perf_counter()-t0:.0f}s, '
                  f'range [{worst:.6f}, {best:.6f}]', flush=True)

    res = json.loads((HERE / 'results_case3_full.json').read_text())
    lo_c, hi_c = res['lower'], res['upper']
    width = hi_c - lo_c
    out = {
        'n_corners': 2 ** L,
        'leaves': m.LEAVES,
        'corner_min': worst, 'corner_max': best,
        'argmin_bits': list(argmin), 'argmax_bits': list(argmax),
        'certified_lower': lo_c, 'certified_upper': hi_c,
        'certified_width': width,
        'static_width': best - worst,
        'slack_lower': worst - lo_c,
        'slack_upper': hi_c - best,
        'conservatism_fraction': ((worst - lo_c) + (hi_c - best)) / width,
        't_s': round(time.perf_counter() - t0, 1),
    }
    print(json.dumps(out, indent=2))
    (HERE / 'results_case3_corners.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
