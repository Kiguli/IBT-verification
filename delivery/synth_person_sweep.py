"""Supplement to synth_grid.py: the grid shows Pr[F<=80 delivered=2] nondecreasing in handover, grasp and move at
every grid step, and non-monotone in the shared person detector. This sweeps the person leaf densely
(3001 points, endpoints exact) with the other three leaves at their low ends (the min side) and at
their high ends (the max side), using the same td.point_vi on the compiled IMC.

Writes synth_person_sweep.json in this folder.
"""
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
    ps = [float(v) for v in np.linspace(sc.BOX['person'][0], sc.BOX['person'][1], 3001)]
    out = {}
    for side, k in (('others_low', 0), ('others_high', 1)):
        others = {l: sc.BOX[l][k] for l in sc.LEAVES if l != 'person'}
        vals = [td.point_vi(imc, {'person': p, **others}, sc.TGT, sc.H)[0][i0] for p in ps]
        j_min = int(np.argmin(vals)); j_max = int(np.argmax(vals))
        out[side] = {'others': others,
                     'min': vals[j_min], 'argmin_person': ps[j_min],
                     'max': vals[j_max], 'argmax_person': ps[j_max],
                     'at_person_0.20': vals[0], 'at_person_0.50': vals[-1]}
    res = {'what': 'dense 1-D sweep of the person leaf (3001 points on [0.20, 0.50])',
           'sweeps': out, 'command': 'python synth_person_sweep.py',
           'generated': time.strftime('%Y-%m-%dT%H:%M:%S')}
    with open(os.path.join(ec.HERE, 'synth_person_sweep.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
