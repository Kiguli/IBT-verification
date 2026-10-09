"""The warehouse model as stated in case3.py: |Sigma| factorisation, the seven leaf intervals,
and how they were derived (recomputed from the trial counts and compared bit for bit with
results_case3_full.json). Writes model_facts.json in this folder.
"""
import json
import os
import time

import exp_common as ec
import case3

full = json.load(open(os.path.join(ec.HERE, 'results_case3_full.json')))
cfg = case3.Config(small=False)
n_cells = len(cfg.cells)
fact = {'passable_cells': n_cells, 'grid': '%dx%d' % (cfg.N, cfg.N), 'rack_cells': len(cfg.racks),
        'stages': cfg.NSTAGE, 'battery_levels': cfg.BATMAX + 1, 'worker_phases': cfg.NW,
        'charge_flag_values': 2}
sigma = n_cells * cfg.NSTAGE * (cfg.BATMAX + 1) * cfg.NW * 2
leaves = {}
for name, (k, n) in case3.COUNTS.items():
    cp = case3.clopper_pearson(k, n)
    wl = case3.wilson(k, n)
    leaves[name] = {'successes': k, 'trials': n, 'clopper_pearson': list(cp),
                    'equals_results_case3_full_bitwise': list(cp) == list(full['intervals'][name]),
                    'wilson_z2.69_for_comparison': list(wl)}
res = {
    'sigma': sigma, 'sigma_factorisation': fact,
    'sigma_matches_results_case3_full': sigma == full['sigma'],
    'start_state': [cfg.HOME[0], cfg.HOME[1], 0, cfg.BATMAX, 0, 0],
    'H': cfg.H, 'charge_hysteresis_lo_hi_gain': [cfg.chg_lo, cfg.chg_hi, cfg.chg_gain],
    'stations': {'home': cfg.HOME, 'charger': cfg.CHARGER, 'picks': cfg.picks, 'packs': cfg.packs},
    'patrol': cfg.patrol,
    'calib_function_used': case3.calib.__name__,
    'alpha': case3.ALPHA, 'n_leaves': case3.NLEAVES, 'alpha_per_leaf': case3.ALPHA_PER_LEAF,
    'leaves': leaves,
    'all_intervals_match': all(v['equals_results_case3_full_bitwise'] for v in leaves.values()),
    'command': 'python model_facts.py',
    'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
}
with open(os.path.join(ec.HERE, 'model_facts.json'), 'w') as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res, indent=1))
