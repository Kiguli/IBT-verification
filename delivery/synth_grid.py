"""Delivery mission: min / max of the H=80 delivery probability over Theta(T) on a 9-point-per-leaf
grid (endpoints included; 9^4 = 6561 leaf assignments), evaluated exactly as synth_case.py does:
td.point_vi(imc, leafvals, TGT, H)[0][i0] on imc = td.compile_imc(build(), S0).

Writes synth_grid.json in this folder.
"""
import itertools
import json
import os
import sys
import time

import exp_common as ec
sys.path.insert(0, ec.HERE)

import numpy as np
import tickdist as td
import synth_case as sc


def main():
    T = sc.build()
    imc = td.compile_imc(T, sc.S0)
    i0 = imc.index[sc.S0]
    _, tlo = td.robust_vi(imc, sc.TGT, sc.H, True)
    _, thi = td.robust_vi(imc, sc.TGT, sc.H, False)
    cert = [tlo[sc.H][i0], thi[sc.H][i0]]

    grids = {l: [float(v) for v in np.linspace(sc.BOX[l][0], sc.BOX[l][1], 9)] for l in sc.LEAVES}
    for l in sc.LEAVES:
        assert grids[l][0] == sc.BOX[l][0] and grids[l][-1] == sc.BOX[l][1]
    t0 = time.perf_counter()
    vals = []
    for combo in itertools.product(*[range(9) for _ in sc.LEAVES]):
        lv = {l: grids[l][k] for l, k in zip(sc.LEAVES, combo)}
        v = td.point_vi(imc, lv, sc.TGT, sc.H)[0][i0]
        vals.append((v, combo))
    t_grid = time.perf_counter() - t0
    vmin, amin = min(vals)
    vmax, amax = max(vals)
    is_corner = lambda c: all(k in (0, 8) for k in c)  # noqa: E731
    corners = [(v, c) for (v, c) in vals if is_corner(c)]
    cmin, cargmin = min(corners)
    cmax, cargmax = max(corners)
    # all assignments attaining the min (ties)
    ties_min = [c for (v, c) in vals if v == vmin]
    # monotonicity along each leaf axis on the grid (holding the others fixed)
    arr = np.empty([9] * len(sc.LEAVES))
    for v, c in vals:
        arr[c] = v
    mono = {}
    for ax, l in enumerate(sc.LEAVES):
        d = np.diff(arr, axis=ax)
        mono[l] = {'min_step': float(d.min()), 'max_step': float(d.max()),
                   'nondecreasing_everywhere': bool((d >= -1e-15).all()),
                   'nonincreasing_everywhere': bool((d <= 1e-15).all())}
    named = lambda c: {l: grids[l][k] for l, k in zip(sc.LEAVES, c)}  # noqa: E731
    res = {
        'what': 'Pr[F<=80 delivered=2] over a 9x9x9x9 grid of leaf assignments in Theta(T) '
                '(np.linspace(lo, hi, 9) per leaf, endpoints exact), via td.point_vi on the compiled IMC',
        'leaves': sc.LEAVES, 'box': sc.BOX, 'grid_values': grids,
        'n_assignments': len(vals),
        'min': vmin, 'argmin': named(amin), 'argmin_grid_index': amin,
        'argmin_is_corner': is_corner(amin), 'n_assignments_attaining_min': len(ties_min),
        'max': vmax, 'argmax': named(amax), 'argmax_grid_index': amax,
        'argmax_is_corner': is_corner(amax),
        'corner_min_16': cmin, 'corner_argmin_16': named(cargmin),
        'corner_max_16': cmax, 'corner_argmax_16': named(cargmax),
        'grid_min_minus_corner_min': vmin - cmin,
        'grid_max_minus_corner_max': vmax - cmax,
        'certificate_H': cert,
        'gap_floor_to_grid_min': vmin - cert[0],
        'per_leaf_monotonicity_on_grid': mono,
        'compiled_imc': {'R': imc.n(), 'max_out_degree': max(len(x) for x in imc.succ)},
        'seconds_grid': round(t_grid, 2),
        'provenance': ec.provenance(),
        'command': 'python synth_grid.py',
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(ec.HERE, 'synth_grid.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
